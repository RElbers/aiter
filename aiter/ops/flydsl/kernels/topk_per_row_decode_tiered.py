"""FlyDSL decode TopK-per-row kernel (tiered persistent multi-block radix-select)

Computes an unordered Top-K index set per decode row, fusing a single-workgroup and
a multi-block radix-select into one persistent launch grid=(blocks_per_row, num_rows)
that picks a per-row strategy by valid length. Which is required when decode batch's
sequences differ in length. Each row derives how many of its blocks_per_row
workgroups cooperate (active_parts); the rest return immediately.

Inputs/outputs:
  - logits: fp32, logical shape (num_rows, L), strides (stride0, stride1) with
    stride1 == 1 (contiguous within a row).
  - seq_lens: int32 causal lengths per sequence; row r scores sequence r // next_n at
    decode slot r % next_n, valid length seq_len - next_n + slot + 1.
  - indices: flattened int32 output with shape (num_rows, top_k); each row writes its
    unordered Top-K index set. A row with fewer than top_k valid entries is
    identity-filled and padded with -1.
  - workspace: row-major int32 scratch sized by topk_workspace_slots(num_rows,
    bits_per_pass). The multi-block tiers merge per-block LDS histograms into its
    pass-private global histograms over an inter-workgroup acquire/release barrier and
    coordinate through its counters; the single-workgroup tier never touches it.

Paths (per row, by valid length row_len):
  - short (row_len <= short_max): active_parts = 1; part 0 runs the whole radix-select
    in one workgroup - LDS-only histograms, no inter-workgroup barrier, no workspace
    round-trip.
  - mid (short_max < row_len <= mid_max): active_parts = min(blocks_per_row, mid_cap).
  - long (row_len > mid_max): active_parts = min(blocks_per_row, long_cap).

Constraints:
  - logits are fp32; the order-preserving radix key twiddle is fp32-specific.
  - bits_per_pass is 10 or 11; the short tier requires 11 bits (2048-bin LDS histogram).
  - BLOCK_THREADS is fixed at 1024 (wave64); the histogram/scan layout and the
    occupancy deadlock guard rely on it.
  - workspace must be zeroed before any launch that enters a multi-block tier; its
    counters and histograms accumulate from zero (needs_workspace_zero reports when).
  - The row barrier spins (s_sleep), so a row's blocks_per_row workgroups must be
    co-resident. This is a regular launch, not hipLaunchCooperativeKernel: it is safe
    only because HIP flattens the grid x-fastest, so a row's parts launch contiguously
    and drain in order - scheduler launch order, not a cooperative guarantee. Keep
    num_rows * blocks_per_row co-resident; the wrapper's deadlock guard enforces this,
    forcing larger batches onto the barrier-free short tier.
"""

from functools import cache
from typing import Any, Literal

import flydsl.compiler as flyc
import flydsl.expr as fx
from flydsl._mlir.dialects import llvm
from flydsl.expr import (
    arith,
    buffer_ops,
    const_expr,
    gpu,
    range_constexpr,
    rocdl,
    vector,
)
from flydsl.expr.typing import T

# HW max block size; also assumed by the bucket scan (2 bins/thread -> 2048 bins)
# and the occupancy=2 deadlock guard. Changing it breaks both.
BLOCK_THREADS = 1024
WARP_SIZE = 64
LOAD_VEC = 4
# Default histogram-scan staging (one of 1/2/4/8)
SCAN_STAGES = 2

# 128B-spaced inter-workgroup counter groups (32 int32 == 128B each), kept in the int32 workspace.
COUNTER_STRIDE = 32
COUNTER_SLOTS = 6 * COUNTER_STRIDE
COUNTER_ARRIVALS = 2 * COUNTER_STRIDE
COUNTER_OUT_FRONT = 3 * COUNTER_STRIDE
COUNTER_OUT_BACK = 4 * COUNTER_STRIDE
COUNTER_PASS_DONE = 5 * COUNTER_STRIDE

SMEM_META_K = 0
SMEM_META_LEN = 1
SMEM_META_THRESHOLD = 2
SMEM_META_ABOVE = 3

# Short-tier one-workgroup metadata reuses the same 8-int LDS block after zeroing.
SMEM_META_SHORT_ABOVE_H = 0
SMEM_META_SHORT_THRESHOLD_H = 1
SMEM_META_SHORT_ABOVE_M = 2
SMEM_META_SHORT_THRESHOLD_M = 3
SMEM_META_SHORT_ABOVE_L = 4
SMEM_META_SHORT_THRESHOLD_L = 5
SMEM_META_SHORT_FRONT_COUNT = 6
SMEM_META_SHORT_BACK_COUNT = 7


def _num_passes(bits_per_pass: int) -> int:
    return (32 + bits_per_pass - 1) // bits_per_pass


def topk_workspace_slots(
    num_rows: int,
    bits_per_pass: int = 11,
) -> int:
    """Return int32 workspace slots for the tiered path (row-major, per row)."""
    if bits_per_pass not in (10, 11):
        raise ValueError(f"bits_per_pass must be 10 or 11, got {bits_per_pass}")
    row_slots = COUNTER_SLOTS + _num_passes(bits_per_pass) * (1 << bits_per_pass)
    return int(num_rows) * row_slots


def needs_workspace_zero(
    max_row_len: int,
    top_k: int,
    short_max: int,
    tier_mode: str = "auto",
) -> bool:
    """Return whether any row can enter the persistent multi-block path."""
    if tier_mode == "short":
        return False
    if tier_mode in ("mid", "long"):
        return True
    return max_row_len > max(short_max, top_k)


@cache
def create_topk_per_row_decode_tiered_kernel(
    top_k: int,
    *,
    blocks_per_row: int = 8,
    bits_per_pass: int = 11,
    scan_stages: int = SCAN_STAGES,
    tier_mode: Literal["auto", "short", "mid", "long"] = "auto",
    tiered_short_max: int = 16384,
    tiered_mid_cap: int = 16,
    tiered_mid_max: int = 65536,
    tiered_long_cap: int = 32,
    mask_non_finite: bool = True,
) -> Any:
    """Build a launcher that selects the Top-K largest values' column indices per
    decode row (unordered set, matching torch.topk by value). Implemented as a
    tiered persistent multi-block radix-select; the returned launcher is cached.

    top_k: number of indices selected per row (compile-time; any positive value).
    blocks_per_row: workgroups launched per row (grid width); the mid/long tiers cap
        how many actually cooperate, excess workgroups return immediately.
    bits_per_pass: radix digit width, 10 or 11; 11 = 2048-bin LDS histogram, required
        by the short tier.
    scan_stages: histogram block-scan staging, one of 1/2/4/8.
    tier_mode: "auto" picks a tier per row by valid length; "short"/"mid"/"long"
        force that tier for every row.
    tiered_short_max: row_len <= this -> short tier (single workgroup, barrier-free).
    tiered_mid_max: short_max < row_len <= this -> mid tier; longer -> long tier.
    tiered_mid_cap / tiered_long_cap: max cooperating workgroups per row in the mid /
        long tier (clamped to blocks_per_row).
    mask_non_finite: clamp inf/NaN to -inf so they never rank into the top-k.
    """
    short_max = tiered_short_max
    mid_cap = tiered_mid_cap
    mid_max = tiered_mid_max
    long_cap = tiered_long_cap

    if bits_per_pass not in (10, 11):
        raise ValueError(f"bits_per_pass must be 10 or 11, got {bits_per_pass}")
    if scan_stages not in (1, 2, 4, 8):
        raise ValueError(f"scan_stages must be one of (1, 2, 4, 8), got {scan_stages}")
    if not 2 <= blocks_per_row <= 32:
        raise ValueError(f"blocks_per_row must be in [2, 32], got {blocks_per_row}")
    if mid_cap < 2 or long_cap < 2:
        raise ValueError(f"mid_cap/long_cap must be >= 2, got {mid_cap}/{long_cap}")
    if mid_max < short_max:
        raise ValueError(f"mid_max must be >= short_max, got {mid_max} < {short_max}")
    if tier_mode not in ("auto", "short", "mid", "long"):
        raise ValueError(
            f"tier_mode must be one of auto/short/mid/long, got {tier_mode!r}"
        )
    # The short tier runs the standalone one-workgroup radix-select (2048-bin LDS
    # histogram), so it needs bits_per_pass == 11. It is compiled in for "auto"
    # (short rows) and "short" (all rows); forcing "short" without bpp==11 is an
    # error rather than a silent fallback.
    if tier_mode == "short" and bits_per_pass != 11:
        raise ValueError(
            f"tier_mode='short' requires bits_per_pass == 11, got {bits_per_pass}"
        )

    short_tier = tier_mode in ("auto", "short") and bits_per_pass == 11
    block_threads = BLOCK_THREADS
    waves_per_block = (block_threads + WARP_SIZE - 1) // WARP_SIZE
    num_passes = _num_passes(bits_per_pass)
    num_buckets = 1 << bits_per_pass
    row_workspace_slots = COUNTER_SLOTS + num_passes * num_buckets

    # Caps/thresholds only affect codegen for modes that use them; include them in
    # the name for those modes so distinct configs cache separately.
    _cap_tag = (
        ""
        if tier_mode == "short"
        else (
            f"_s{tiered_short_max}_mc{tiered_mid_cap}_mm{tiered_mid_max}_lc{tiered_long_cap}"
        )
    )
    kernel_name = f"topk_per_row_decode_persistent_k{top_k}_bpp{bits_per_pass}_g{blocks_per_row}_v2_stage{scan_stages}_{tier_mode}{_cap_tag}{'_1wg' if short_tier else ''}{'_mf' if mask_non_finite else ''}"

    @fx.struct
    class SharedStorage:
        s_hist: fx.Array[fx.Int32, num_buckets, 16]
        s_scan: fx.Array[fx.Int32, waves_per_block * 2, 16]
        s_meta: fx.Array[fx.Int32, 8, 16]

    @flyc.kernel(name=kernel_name, known_block_size=[block_threads, 1, 1])
    def topk_per_row_decode_tiered_kernel(
        logits: fx.Tensor,
        next_n: fx.Int32,
        seq_lens: fx.Tensor,
        indices: fx.Tensor,
        workspace: fx.Tensor,
        stride0: fx.Int32,
    ) -> None:
        block_x = gpu.block_id("x")
        block_y = gpu.block_id("y")
        thread_x = gpu.thread_id("x")
        part = fx.Int32(block_x)
        row = fx.Int32(block_y)
        tid = fx.Int32(thread_x)
        tid_idx = thread_x
        lane = tid % WARP_SIZE
        wave = tid // WARP_SIZE

        lds = fx.SharedAllocator().allocate(SharedStorage).peek()
        s_hist = lds.s_hist.view(fx.make_layout(num_buckets, 1))
        s_scan = lds.s_scan.view(fx.make_layout(waves_per_block * 2, 1))
        s_meta = lds.s_meta.view(fx.make_layout(8, 1))

        logits_rsc = buffer_ops.create_buffer_resource(logits, max_size=True)
        seq_lens_rsc = buffer_ops.create_buffer_resource(seq_lens, max_size=True)
        indices_rsc = buffer_ops.create_buffer_resource(indices, max_size=True)
        workspace_rsc = buffer_ops.create_buffer_resource(workspace, max_size=True)
        workspace_base_idx = buffer_ops.extract_base_index(workspace, address_space=1)

        hist_base_ptr = fx.ptrtoint(lds.s_hist.ptr)
        meta_base_ptr = fx.ptrtoint(lds.s_meta.ptr)

        # Decode row geometry.
        seq_row = row // next_n
        slot = row - seq_row * next_n
        seq_len = buffer_ops.buffer_load(
            seq_lens_rsc, seq_row, vec_width=1, dtype=T.i32
        )
        row_len = seq_len - next_n + slot + 1
        row_len = (row_len > 0).select(row_len, 0)
        row_base = row * stride0
        row_out = row * top_k
        row_ws_base = row * row_workspace_slots

        # Active cooperating workgroups per row over the fixed grid (excess blocks
        # return immediately). "auto" picks per row by length; short/mid/long force
        # that tier for every row. Caps are clamped to the grid.
        c_mid_cap = fx.Int32(mid_cap)
        c_long_cap = fx.Int32(long_cap)
        mid_parts = (blocks_per_row < c_mid_cap).select(
            fx.Int32(blocks_per_row), c_mid_cap
        )
        long_parts = (blocks_per_row < c_long_cap).select(
            fx.Int32(blocks_per_row), c_long_cap
        )

        if const_expr(tier_mode == "short"):
            active_parts = fx.Int32(1)
        elif const_expr(tier_mode == "mid"):
            active_parts = mid_parts
        elif const_expr(tier_mode == "long"):
            active_parts = long_parts
        else:  # "auto": pick per row by valid length
            c_short = short_max
            c_mid = mid_max
            active_parts = (row_len <= c_short).select(
                1,
                (row_len <= c_mid).select(mid_parts, long_parts),
            )
        active_threads = active_parts * block_threads
        single_part_active = active_parts == 1

        def active_stride_idx(mult: int = 1):
            return fx.Index(active_threads * mult)

        def counter_slot(slot_const: int):
            return row_ws_base + slot_const

        def histogram_slot(pass_id: int, bin_i32):
            return row_ws_base + (COUNTER_SLOTS + pass_id * num_buckets) + bin_i32

        def i32_ptr(base, elem_i32, address_space: int):
            elem_idx = fx.Index(elem_i32)
            base_idx = fx.Index(base)
            addr = base_idx + elem_idx * fx.Index(4)
            ptr = buffer_ops.create_llvm_ptr(addr, address_space=address_space)
            return ptr._value if const_expr(hasattr(ptr, "_value")) else ptr

        def global_atomic_add_i32(
            elem_i32, value, ordering=llvm.AtomicOrdering.monotonic
        ):
            return llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                i32_ptr(workspace_base_idx, elem_i32, 1),
                arith.unwrap(value),
                ordering,
                syncscope="agent",
                alignment=4,
            ).result

        def global_atomic_xchg_i32(elem_i32, value, ordering):
            return llvm.AtomicRMWOp(
                llvm.AtomicBinOp.xchg,
                i32_ptr(workspace_base_idx, elem_i32, 1),
                arith.unwrap(value),
                ordering,
                syncscope="agent",
                alignment=4,
            ).result

        def global_atomic_load_i32_acquire(elem_i32):
            # Volatile agent-scoped acquire load for the row-barrier spin. The
            # matching release publish below makes histogram updates visible to
            # peer workgroups without issuing a read-modify-write on the polled slot.
            return llvm.LoadOp(
                T.i32,
                i32_ptr(workspace_base_idx, elem_i32, 1),
                alignment=4,
                volatile_=True,
                ordering=llvm.AtomicOrdering.acquire,
                syncscope="agent",
            ).result

        def lds_atomic_add_i32(base, elem_i32, value):
            return llvm.AtomicRMWOp(
                llvm.AtomicBinOp.add,
                i32_ptr(base, elem_i32, 3),
                arith.unwrap(value),
                llvm.AtomicOrdering.monotonic,
                syncscope="workgroup",
                alignment=4,
            ).result

        def spin_until_slot_ge(elem_i32, target):
            cur = fx.Int32(0).ir_value()
            while cur < target:
                rocdl.s_sleep(1)
                cur = global_atomic_load_i32_acquire(elem_i32)

        def row_barrier(token: int):
            # Intentional no-drain acquire/release protocol: workgroup barriers
            # bracket local LDS work, the last workgroup release-publishes pass_done,
            # and peers spin with acquire loads. A full waitcnt drain here is
            # performance/correctness sensitive.
            token_value = fx.Int32(token)
            target_arrivals = token_value * active_parts
            gpu.barrier()
            if tid == 0:
                prev = global_atomic_add_i32(
                    counter_slot(COUNTER_ARRIVALS), 1, llvm.AtomicOrdering.acq_rel
                )
                last = (prev + 1) == target_arrivals
                if last:
                    global_atomic_xchg_i32(
                        counter_slot(COUNTER_PASS_DONE),
                        token_value,
                        llvm.AtomicOrdering.release,
                    )
                else:
                    spin_until_slot_ge(counter_slot(COUNTER_PASS_DONE), token_value)
            gpu.barrier()

        def mask_nonfinite(val):
            # inf/NaN (exponent all-ones) -> -inf so they sort below every finite
            # value and are never selected.
            if const_expr(not mask_non_finite):
                return val
            c_exp_mask = 0x7F800000  # fp32 exponent bits (all-ones => inf/NaN)
            bits = val.bitcast(T.i32)
            is_nonfinite = bits & c_exp_mask == c_exp_mask
            return is_nonfinite.select(float("-inf"), val)

        def radix_twiddle_key(val):
            # Map larger fp32 values to smaller unsigned keys so ascending
            # bucket scans select descending values. Normalize signed zero to
            # keep tie handling value-equivalent.
            val = mask_nonfinite(val)
            key_val = (val == 0.0).select(0.0, val)
            bits = key_val.bitcast(T.i32)
            sign = bits.shrui(fx.Int32(31))
            positive_mask = bits ^ 0x7FFFFFFF
            return (sign == 0).select(positive_mask, bits)

        def bucket_for_key(key, start_bit: int):
            return key.shrui(fx.Int32(start_bit)) & (num_buckets - 1)

        def prefix_for_key(key, previous_start_bit: int):
            b = fx.Int32(previous_start_bit)
            return arith.shli(key.shrui(b), b)

        def load_row_vec(col_base_i32):
            return buffer_ops.buffer_load(
                logits_rsc,
                row_base + col_base_i32,
                vec_width=LOAD_VEC,
                dtype=T.f32,
            )

        def clear_local_histogram():
            for hist_idx in range(
                tid_idx, fx.Index(num_buckets), fx.Index(block_threads)
            ):
                fx.memref_store(0, s_hist, hist_idx)
            gpu.barrier()

        def wave_inclusive_scan_i32(value):
            cur = value
            for sh in range_constexpr(int.bit_length(WARP_SIZE) - 1):
                d = 1 << sh
                src_lane = lane - d
                byte_addr = src_lane * 4
                peer = rocdl.ds_bpermute(T.i32, byte_addr, cur)
                take = lane >= d
                cur = take.select(cur + peer, cur)
            return cur

        def choose_bucket_prefix(target_k):
            # Multi-block ascending block scan over the LDS histogram; each thread owns a bin pair.
            prefix_base = waves_per_block

            bin0 = tid * 2
            bin1 = bin0 + 1

            bin0_valid = bin0 < num_buckets
            bin1_valid = bin1 < num_buckets
            safe_bin0 = bin0_valid.select(bin0, 0)
            safe_bin1 = bin1_valid.select(bin1, 0)
            count0 = bin0_valid.select(fx.memref_load(s_hist, safe_bin0), 0)
            count1 = bin1_valid.select(fx.memref_load(s_hist, safe_bin1), 0)

            pair_count = count0 + count1
            thread_incl = wave_inclusive_scan_i32(pair_count)
            thread_prefix = thread_incl - pair_count

            if lane == (WARP_SIZE - 1):
                fx.memref_store(thread_incl, s_scan, wave)
            gpu.barrier()

            if wave == 0:
                has_slot = lane < waves_per_block
                slot_index = has_slot.select(lane, 0)
                slot_total = has_slot.select(fx.memref_load(s_scan, slot_index), 0)
                slot_prefix = wave_inclusive_scan_i32(slot_total) - slot_total
                if has_slot:
                    fx.memref_store(slot_prefix, s_scan, prefix_base + lane)
            gpu.barrier()

            wave_prefix = fx.memref_load(s_scan, prefix_base + wave)
            excl0 = wave_prefix + thread_prefix
            incl0 = excl0 + count0
            incl1 = incl0 + count1

            def mark_threshold(bin_index, excl, incl, bucket_count):
                is_threshold = (excl < target_k) & (incl >= target_k)
                if is_threshold:
                    fx.memref_store(target_k - excl, s_meta, SMEM_META_K)
                    fx.memref_store(bucket_count, s_meta, SMEM_META_LEN)
                    fx.memref_store(bin_index, s_meta, SMEM_META_THRESHOLD)
                    fx.memref_store(excl, s_meta, SMEM_META_ABOVE)

            mark_threshold(bin0, excl0, incl0, count0)
            mark_threshold(bin1, incl0, incl1, count1)
            gpu.barrier()

        def flush_local_histogram(pass_id: int):
            for hist_idx in range(
                tid_idx, fx.Index(num_buckets), fx.Index(block_threads)
            ):
                hist_i32 = hist_idx
                count = fx.memref_load(s_hist, hist_i32)
                if count != 0:
                    global_atomic_add_i32(histogram_slot(pass_id, hist_i32), count)

        def load_global_histogram(pass_id: int):
            # Vectorized reload
            n_vec = num_buckets // LOAD_VEC
            c_nvec_idx = fx.Index(n_vec)
            for grp in range(tid_idx, c_nvec_idx, fx.Index(block_threads)):
                base_bin = grp * LOAD_VEC
                vec = buffer_ops.buffer_load(
                    workspace_rsc,
                    histogram_slot(pass_id, base_bin),
                    vec_width=LOAD_VEC,
                    dtype=T.i32,
                )
                for j in range_constexpr(LOAD_VEC):
                    total = vector.extract(vec, [j], [])
                    fx.memref_store(total, s_hist, base_bin + j)
            gpu.barrier()

        def process_loaded_scan_vec(
            col_base,
            vec,
            pass_id: int,
            start_bit: int,
            previous_start_bit: int,
            current_bits,
        ):
            for j in range_constexpr(LOAD_VEC):
                col_i32 = col_base + j
                if col_i32 < row_len:
                    val = vector.extract(vec, [j], [])
                    key = radix_twiddle_key(val)
                    matches_prefix = True
                    if const_expr(pass_id != 0):
                        matches_prefix = (
                            prefix_for_key(key, previous_start_bit) == current_bits
                        )
                    if matches_prefix:
                        lds_atomic_add_i32(
                            hist_base_ptr, bucket_for_key(key, start_bit), 1
                        )

        def scan_vec_block(
            vblk, pass_id: int, start_bit: int, previous_start_bit: int, current_bits
        ):
            col_base = fx.Int32(vblk) * LOAD_VEC
            process_loaded_scan_vec(
                col_base,
                load_row_vec(col_base),
                pass_id,
                start_bit,
                previous_start_bit,
                current_bits,
            )

        def staged_scan_vec_blocks(
            vblk,
            pass_id: int,
            start_bit: int,
            previous_start_bit: int,
            current_bits,
        ):
            if const_expr(scan_stages == 1):
                strides = [0]
            elif const_expr(scan_stages == 2):
                strides = [0, active_stride_idx()]
            elif const_expr(scan_stages == 4):
                strides = [
                    0,
                    active_stride_idx(),
                    active_stride_idx(2),
                    active_stride_idx(3),
                ]
            else:
                strides = [
                    0,
                    active_stride_idx(),
                    active_stride_idx(2),
                    active_stride_idx(3),
                    active_stride_idx(4),
                    active_stride_idx(5),
                    active_stride_idx(6),
                    active_stride_idx(7),
                ]
            cols_v = [fx.Int32(vblk + s) * LOAD_VEC for s in strides]
            vecs = [load_row_vec(cb) for cb in cols_v]
            for cb, vc in zip(cols_v, vecs):
                process_loaded_scan_vec(
                    cb, vc, pass_id, start_bit, previous_start_bit, current_bits
                )

        def process_loaded_write_vec(col_base, vec, local_k, kth_bits):
            for j in range_constexpr(LOAD_VEC):
                col_i32 = col_base + j
                if col_i32 < row_len:
                    val = vector.extract(vec, [j], [])
                    key = radix_twiddle_key(val)
                    if fx.Uint32(key) < fx.Uint32(kth_bits):
                        pos = global_atomic_add_i32(counter_slot(COUNTER_OUT_FRONT), 1)
                        if pos < top_k:
                            buffer_ops.buffer_store(col_i32, indices_rsc, row_out + pos)

                    if key == kth_bits:
                        back = global_atomic_add_i32(counter_slot(COUNTER_OUT_BACK), 1)
                        if back < local_k:
                            out_pos = top_k - 1 - back
                            buffer_ops.buffer_store(
                                col_i32, indices_rsc, row_out + out_pos
                            )

        def write_vec_block(vblk, local_k, kth_bits):
            col_base = fx.Int32(vblk) * LOAD_VEC
            process_loaded_write_vec(
                col_base, load_row_vec(col_base), local_k, kth_bits
            )

        global_vec_tid = part * block_threads + tid
        global_vec_tid_idx = fx.Index(global_vec_tid)
        vec_blocks_i32 = (row_len + LOAD_VEC - 1) >> 2
        vec_blocks_idx = fx.Index(vec_blocks_i32)

        def scan_pass(pass_id: int, current_k, current_bits, barrier_token: int):
            start_bit = max(32 - (pass_id + 1) * bits_per_pass, 0)
            previous_start_bit = max(32 - pass_id * bits_per_pass, 0)

            clear_local_histogram()
            if const_expr(scan_stages == 8):
                unroll_limit_idx = vec_blocks_idx - active_stride_idx(7)
                staged_stride_idx = active_stride_idx(8)
            elif const_expr(scan_stages == 4):
                unroll_limit_idx = vec_blocks_idx - active_stride_idx(3)
                staged_stride_idx = active_stride_idx(4)
            elif const_expr(scan_stages == 2):
                unroll_limit_idx = vec_blocks_idx - active_stride_idx()
                staged_stride_idx = active_stride_idx(2)
            else:
                unroll_limit_idx = vec_blocks_idx
                staged_stride_idx = active_stride_idx()

            for vblk, pass_state in range(
                global_vec_tid_idx,
                unroll_limit_idx,
                staged_stride_idx,
                init=[global_vec_tid_idx],
            ):
                staged_scan_vec_blocks(
                    vblk, pass_id, start_bit, previous_start_bit, current_bits
                )
                pass_results = yield [vblk + staged_stride_idx]

            for vblk, pass_state in range(
                pass_results,
                vec_blocks_idx,
                active_stride_idx(),
                init=[0],
            ):
                scan_vec_block(
                    vblk, pass_id, start_bit, previous_start_bit, current_bits
                )
                pass_results = yield [pass_state[0]]
            gpu.barrier()

            if single_part_active:
                choose_bucket_prefix(current_k)
            if ~single_part_active:
                flush_local_histogram(pass_id)
                row_barrier(barrier_token)
                load_global_histogram(pass_id)
                choose_bucket_prefix(current_k)

            chosen_bucket = fx.memref_load(s_meta, SMEM_META_THRESHOLD)
            next_k = fx.memref_load(s_meta, SMEM_META_K)
            next_len = fx.memref_load(s_meta, SMEM_META_LEN)
            next_bits = current_bits | arith.shli(chosen_bucket, fx.Int32(start_bit))
            if const_expr(pass_id == num_passes - 1):
                unroll_limit_idx = vec_blocks_idx - active_stride_idx(3)
                for vblk, write_state in range(
                    global_vec_tid_idx,
                    unroll_limit_idx,
                    active_stride_idx(4),
                    init=[global_vec_tid_idx],
                ):
                    for unroll_id in range_constexpr(4):
                        write_vec_block(
                            vblk + active_stride_idx(unroll_id),
                            next_k,
                            next_bits,
                        )
                    write_results = yield [vblk + active_stride_idx(4)]

                for vblk, write_state in range(
                    write_results,
                    vec_blocks_idx,
                    active_stride_idx(),
                    init=[0],
                ):
                    write_vec_block(vblk, next_k, next_bits)
                    write_results = yield [write_state[0]]

            return next_k, next_len, next_bits

        def one_workgroup_short_tier():
            # Faithful copy of the standalone one-workgroup unordered radix-select.
            # Runs entirely within a single workgroup (part 0):
            # LDS-only histograms, a hierarchical block scan to locate each
            # radix threshold, and a direct LDS-counter atomic-append write.
            # Unlike the persistent multi-block path, this short tier uses the
            # standalone ascending ordered key and total-k threshold convention.
            # Three order-preserving passes peel 11/11/10 bits of that key, so
            # it requires a 2048-bin histogram
            # (num_buckets == 2048, i.e. bits_per_pass == 11). It reuses the
            # persistent kernel's existing LDS (s_hist 2048 ints,
            # s_scan 32 ints, s_meta 8 ints) with no extra shared memory.

            def ordered_key(val):
                val = mask_nonfinite(val)
                key_val = (val == 0.0).select(0.0, val)
                bits = key_val.bitcast(T.i32)
                sign = bits.shrui(fx.Int32(31))
                neg_key = ~bits
                pos_key = bits ^ ~0x7FFFFFFF
                return (sign != 0).select(neg_key, pos_key)

            def radix_bucket(val, shift, nbits: int):
                if const_expr(32 - shift <= nbits):
                    return ordered_key(val).shrui(fx.Int32(shift))

                return ordered_key(val).shrui(fx.Int32(shift)) & ((1 << nbits) - 1)

            def clear_hist():
                for h in range(tid_idx, fx.Index(num_buckets), fx.Index(block_threads)):
                    fx.memref_store(0, s_hist, h)
                gpu.barrier()

            def choose_threshold(target_k, above_slot, threshold_slot):
                # Hierarchical inclusive block scan over the 2048-bin histogram;
                # each thread owns the contiguous bin pair (2*tid, 2*tid+1). The
                # kth-largest boundary is the first bucket whose inclusive prefix
                # passes ``K' = total - target_k`` (excl <= K' < incl).
                prefix_base = waves_per_block

                bin0 = tid * 2
                count0 = fx.memref_load(s_hist, bin0)
                count1 = fx.memref_load(s_hist, bin0 + 1)

                pair_count = count0 + count1
                thread_incl = wave_inclusive_scan_i32(pair_count)
                thread_prefix = thread_incl - pair_count

                if lane == (WARP_SIZE - 1):
                    fx.memref_store(thread_incl, s_scan, wave)
                gpu.barrier()

                if wave == 0:
                    has_slot = lane < waves_per_block
                    slot_index = has_slot.select(lane, 0)
                    slot_total = has_slot.select(fx.memref_load(s_scan, slot_index), 0)
                    slot_prefix = wave_inclusive_scan_i32(slot_total) - slot_total
                    if has_slot:
                        fx.memref_store(slot_prefix, s_scan, prefix_base + lane)
                gpu.barrier()

                wave_prefix = fx.memref_load(s_scan, prefix_base + wave)
                last_wave_prefix = fx.memref_load(
                    s_scan, prefix_base + waves_per_block - 1
                )
                last_wave_total = fx.memref_load(s_scan, waves_per_block - 1)
                total = last_wave_prefix + last_wave_total
                below_target = total - target_k

                excl0 = wave_prefix + thread_prefix
                incl0 = excl0 + count0
                incl1 = incl0 + count1

                def mark_threshold(bin_index, excl, incl):
                    is_threshold = (excl <= below_target) & (incl > below_target)
                    if is_threshold:
                        fx.memref_store(bin_index, s_meta, threshold_slot)
                        fx.memref_store(total - incl, s_meta, above_slot)

                mark_threshold(bin0, excl0, incl0)
                mark_threshold(bin0 + 1, incl0, incl1)
                gpu.barrier()

                return fx.memref_load(s_meta, threshold_slot)

            if tid == 0:
                for meta_slot in range_constexpr(8):
                    fx.memref_store(0, s_meta, meta_slot)
            gpu.barrier()

            # Per-chunk bodies for the three radix passes plus the final scatter.
            # Each takes the chunk's first column index and its already-loaded
            # vec4 and feeds a fresh HBM load through the exact same logic.
            def hist_pass1_chunk(col_base, vec):
                for j in range_constexpr(LOAD_VEC):
                    col_i32 = col_base + j
                    if col_i32 < row_len:
                        val = vector.extract(vec, [j], [])
                        bucket_h = radix_bucket(val, 21, 11)
                        lds_atomic_add_i32(hist_base_ptr, bucket_h, 1)

            def hist_pass2_chunk(col_base, vec, threshold_h):
                for j in range_constexpr(LOAD_VEC):
                    col_i32 = col_base + j
                    if col_i32 < row_len:
                        val = vector.extract(vec, [j], [])
                        bucket_h = radix_bucket(val, 21, 11)
                        if bucket_h == threshold_h:
                            bucket_m = radix_bucket(val, 10, 11)
                            lds_atomic_add_i32(hist_base_ptr, bucket_m, 1)

            def hist_pass3_chunk(col_base, vec, threshold_h, threshold_m):
                for j in range_constexpr(LOAD_VEC):
                    col_i32 = col_base + j
                    if col_i32 < row_len:
                        val = vector.extract(vec, [j], [])
                        bucket_h = radix_bucket(val, 21, 11)
                        bucket_m = radix_bucket(val, 10, 11)
                        if (bucket_h == threshold_h) & (bucket_m == threshold_m):
                            bucket_l = radix_bucket(val, 0, 10)
                            lds_atomic_add_i32(hist_base_ptr, bucket_l, 1)

            def final_scatter_chunk(
                col_base,
                vec,
                threshold_h,
                threshold_m,
                threshold_l,
                num_needed,
            ):
                for j in range_constexpr(LOAD_VEC):
                    col_i32 = col_base + j
                    if col_i32 < row_len:
                        val = vector.extract(vec, [j], [])
                        bucket_h = radix_bucket(val, 21, 11)
                        bucket_m = radix_bucket(val, 10, 11)
                        bucket_l = radix_bucket(val, 0, 10)

                        gt_h = bucket_h > threshold_h
                        eq_h = bucket_h == threshold_h
                        gt_m = bucket_m > threshold_m
                        eq_m = bucket_m == threshold_m
                        gt_l = bucket_l > threshold_l
                        eq_l = bucket_l == threshold_l

                        # Ranked above kth key -> append from the front.
                        strictly_above = gt_h | (eq_h & (gt_m | (eq_m & gt_l)))
                        if strictly_above:
                            front_count = fx.Int32(SMEM_META_SHORT_FRONT_COUNT)
                            pos = lds_atomic_add_i32(meta_base_ptr, front_count, 1)
                            buffer_ops.buffer_store(col_i32, indices_rsc, row_out + pos)

                        # Matches kth key exactly -> backfill num_needed slots.
                        at_boundary = eq_h & eq_m & eq_l
                        if at_boundary:
                            back_count = fx.Int32(SMEM_META_SHORT_BACK_COUNT)
                            back = lds_atomic_add_i32(meta_base_ptr, back_count, 1)
                            if back < num_needed:
                                out_pos = top_k - 1 - back
                                buffer_ops.buffer_store(
                                    col_i32, indices_rsc, row_out + out_pos
                                )

            # Reread driver: stream the whole valid row from HBM once per pass.
            # This keeps the short tier's VGPR footprint low while sharing the
            # same per-chunk logic across radix passes and final scatter.
            def reread_pass(chunk_fn):
                for vblk in range(
                    tid_idx,
                    vec_blocks_idx,
                    fx.Index(block_threads),
                ):
                    col_base = vblk * LOAD_VEC
                    chunk_fn(col_base, load_row_vec(col_base))

            # Pass 1: high 11 bits over the whole valid row.
            clear_hist()
            reread_pass(lambda cb, v: hist_pass1_chunk(cb, v))
            gpu.barrier()
            threshold_h = choose_threshold(
                top_k,
                SMEM_META_SHORT_ABOVE_H,
                SMEM_META_SHORT_THRESHOLD_H,
            )

            # Pass 2: mid 11 bits within the high boundary bucket.
            clear_hist()
            reread_pass(lambda cb, v: hist_pass2_chunk(cb, v, threshold_h))
            gpu.barrier()
            above_h = fx.memref_load(s_meta, SMEM_META_SHORT_ABOVE_H)
            need_after_h = top_k - above_h
            threshold_m = choose_threshold(
                need_after_h,
                SMEM_META_SHORT_ABOVE_M,
                SMEM_META_SHORT_THRESHOLD_M,
            )

            # Pass 3: low 10 bits within the high+mid boundary.
            clear_hist()
            reread_pass(lambda cb, v: hist_pass3_chunk(cb, v, threshold_h, threshold_m))
            gpu.barrier()
            above_m = fx.memref_load(s_meta, SMEM_META_SHORT_ABOVE_M)
            need_after_m = need_after_h - above_m
            threshold_l = choose_threshold(
                need_after_m,
                SMEM_META_SHORT_ABOVE_L,
                SMEM_META_SHORT_THRESHOLD_L,
            )
            above_l = fx.memref_load(s_meta, SMEM_META_SHORT_ABOVE_L)
            need_after_l = need_after_m - above_l

            # Final phase: direct atomic-append write (LDS counters only).
            reread_pass(
                lambda cb, v: final_scatter_chunk(
                    cb,
                    v,
                    threshold_h,
                    threshold_m,
                    threshold_l,
                    need_after_l,
                )
            )

        # Direct-fill: rows with row_len <= top_k (part 0 only) emit identity indices + -1.
        direct_fill = row_len <= top_k
        direct_fill_active = (part == 0) & direct_fill
        direct_fill_iters = direct_fill_active.select(fx.Index(top_k), 0)
        for out_col in range(tid_idx, direct_fill_iters, fx.Index(block_threads)):
            out_col_i32 = fx.Int32(out_col)
            valid = out_col_i32 < row_len
            out_val = valid.select(out_col_i32, -1)
            buffer_ops.buffer_store(out_val, indices_rsc, row_out + out_col_i32)

        if const_expr(short_tier):
            short_active = single_part_active & (part == 0) & (row_len > top_k)
            if short_active:
                one_workgroup_short_tier()
            persistent_active = (
                (row_len > top_k) & (part < active_parts) & (~single_part_active)
            )
        else:
            persistent_active = (row_len > top_k) & (part < active_parts)

        if persistent_active:
            local_k = top_k
            kth_bits = 0
            for pass_id in range_constexpr(num_passes):
                local_k, _local_len, kth_bits = scan_pass(
                    pass_id, local_k, kth_bits, pass_id + 1
                )

    @flyc.jit
    def launcher(
        logits: fx.Tensor,
        next_n: fx.Int32,
        seq_lens: fx.Tensor,
        indices: fx.Tensor,
        workspace: fx.Tensor,
        num_rows: fx.Int32,
        stride0: fx.Int32,
        stride1: fx.Int32,
        stream: fx.Stream,
    ) -> None:
        grid_y = fx.Index(num_rows)
        topk_per_row_decode_tiered_kernel(
            logits, next_n, seq_lens, indices, workspace, stride0
        ).launch(
            grid=(blocks_per_row, grid_y, 1),
            block=(block_threads, 1, 1),
            stream=stream,
        )

    return launcher
