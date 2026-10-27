# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Cache policies: warm, rotate (input copies), flush (cache buster)."""

import copy
import math
import statistics
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from typing import Any

import torch

from .config import BenchmarkConfig, ResolvedCachePolicy
from .result import CacheResult

# Last-level cache per arch. CDNA3/CDNA4 put a 256 MB Infinity Cache (MALL)
# behind the per-XCD L2; torch only reports the 4 MB L2, so a buster sized from
# L2_cache_size would leave inputs resident in the MALL.
_LLC_BYTES = {
    "gfx942": 256 * 1024 * 1024,
    "gfx950": 256 * 1024 * 1024,
}

# Never spend more than this fraction of free device memory on input copies.
_ROTATE_FREE_MEM_FRACTION = 0.5
# auto cache policy: treat the kernel as cache-sensitive when a cold call is
# this much slower than a warm one.
_AUTO_COLD_THRESHOLD = 0.05
_AUTO_PROBE_COUNT = 5

_ArgumentSet = tuple[tuple[Any, ...], dict[str, Any]]


def llc_bytes(device: int | None = None) -> int:
    """Size of the cache a cold measurement has to defeat."""
    properties = torch.cuda.get_device_properties(device)
    arch = getattr(properties, "gcnArchName", "").split(":", 1)[0]
    return _LLC_BYTES.get(
        arch,
        getattr(properties, "L2_cache_size", 4 * 1024 * 1024),
    )


_flush_buffers: dict[int, torch.Tensor] = {}


def flush_cache(device: int | None = None) -> None:
    """Evict the last-level cache by overwriting a buffer twice its size."""
    device_index = torch.cuda.current_device() if device is None else device
    buffer = _flush_buffers.get(device_index)
    if buffer is None:
        buffer = torch.empty(
            2 * llc_bytes(device_index) // 4,
            dtype=torch.int32,
            device=device_index,
        )
        _flush_buffers[device_index] = buffer

    buffer.zero_()


def _iter_tensors(value: Any) -> Iterator[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        yield value
    elif isinstance(value, (list, tuple)):
        for item in value:
            yield from _iter_tensors(item)
    elif isinstance(value, dict):
        for item in value.values():
            yield from _iter_tensors(item)


def argument_storage_bytes(args: tuple[Any, ...], kwargs: dict[str, Any]) -> int:
    """Bytes of distinct device tensors reachable from the call's arguments."""
    seen: set[int] = set()
    total_bytes = 0

    for tensor in _iter_tensors((args, kwargs)):
        if tensor.is_cuda and tensor.data_ptr() not in seen:
            seen.add(tensor.data_ptr())
            total_bytes += tensor.untyped_storage().nbytes()

    return total_bytes


@dataclass(frozen=True)
class ArgumentCopies:
    argument_sets: list[_ArgumentSet]  # the caller's originals first
    llc_coverage: float  # bytes cycled through / LLC size
    # every timed call reads inputs that are not cache-resident
    meets_cold_sizing_target: bool


def copy_arguments(
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    max_copies: int,
    *,
    storage_bytes: int,
    calls_to_cover: int | None = None,
    device: int | None = None,
) -> ArgumentCopies:
    """Deep-copy the inputs so that each timed call reads cold inputs.

    Like run_perftest, but sized from the LLC instead of filling free memory.
    Inputs are cold when either every timed call gets its own copy
    (``calls_to_cover`` copies; None means calls repeat without bound, as in a
    graph replay) or cycling through the copies streams at least twice the
    LLC. The copy count is capped by ``max_copies`` and half the free memory.
    The cache is flushed after copying so no copy starts resident.
    """
    cache_bytes = llc_bytes(device)
    required_copies = max(1, math.ceil(2 * cache_bytes / storage_bytes))
    if calls_to_cover is not None:
        required_copies = min(required_copies, calls_to_cover)

    free_bytes = torch.cuda.mem_get_info(device)[0]
    memory_limit = max(1, int(free_bytes * _ROTATE_FREE_MEM_FRACTION // storage_bytes))

    copy_count = min(required_copies, max_copies, memory_limit)
    argument_sets = [(args, kwargs)] + [
        (copy.deepcopy(args), copy.deepcopy(kwargs)) for _ in range(copy_count - 1)
    ]

    flush_cache(device)
    torch.cuda.synchronize()

    meets_cold_sizing_target = copy_count * storage_bytes >= 2 * cache_bytes or (
        calls_to_cover is not None and copy_count >= calls_to_cover
    )

    return ArgumentCopies(
        argument_sets=argument_sets,
        llc_coverage=copy_count * storage_bytes / cache_bytes,
        meets_cold_sizing_target=meets_cold_sizing_target,
    )


def _probe_median_us(call: Callable[[], Any], *, flush: bool) -> float:
    samples_us: list[float] = []

    for _ in range(_AUTO_PROBE_COUNT):
        if flush:
            flush_cache()

        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        torch.cuda.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)

    return statistics.median(samples_us)


def _probe_speedup(call: Callable[[], Any]) -> float:
    warm_us = _probe_median_us(call, flush=False)
    cold_us = _probe_median_us(call, flush=True)
    return (cold_us - warm_us) / cold_us if cold_us > 0 else 0.0


def _resolve_cache_policy(
    call: Callable[[], Any],
    cfg: BenchmarkConfig,
    has_device_args: bool,
    notes: list[str],
) -> tuple[ResolvedCachePolicy, float | None]:
    policy = cfg.cache_policy_or_default
    speedup = None

    if policy == "auto":
        speedup = _probe_speedup(call)
        if speedup <= _AUTO_COLD_THRESHOLD:
            policy = "warm"
        else:
            policy = "rotate" if cfg.mode == "graph" else "flush"

    if policy == "flush" and cfg.mode == "graph":
        raise ValueError(
            "graph mode cannot flush the cache between recorded calls; use "
            "cache='rotate' to time graph replay with cold inputs"
        )

    if policy != "rotate" or has_device_args:
        return policy, speedup

    if cfg.cache == "rotate":
        raise ValueError(
            "cache='rotate' needs the inputs passed as arguments (run_bench(fn, "
            "args=...)); a closure has nothing to "
            "rotate. Use cache='flush' instead."
        )

    if cfg.mode != "graph":
        return "flush", speedup

    if cfg.cache == "auto":
        notes.append(
            "auto: kernel is cache-sensitive, but graph mode can only run cold by "
            "rotating inputs and none were passed; timed warm"
        )
    else:
        notes.append(
            "graph mode runs cold only by rotating inputs, and none were "
            "passed (pass them via args); timed warm"
        )
    return "warm", speedup


def _bind_call(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
) -> Callable[[], Any]:
    return lambda: fn(*args, **kwargs)


@dataclass(frozen=True)
class CacheSetup:
    cache: CacheResult
    bound_calls: list[Callable[[], Any]]
    before_call: Callable[[], None] | None
    notes: list[str]


def prepare_cache(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    cfg: BenchmarkConfig,
    *,
    calls_to_cover: int | None,
    device: int,
) -> CacheSetup:
    notes: list[str] = []

    storage_bytes = argument_storage_bytes(args, kwargs)
    call = _bind_call(fn, args, kwargs)
    policy, speedup = _resolve_cache_policy(call, cfg, storage_bytes > 0, notes)
    cache = CacheResult(
        requested_policy=cfg.cache,
        effective_policy=policy,
        cache_speedup_fraction=speedup,
    )

    if policy == "warm":
        bound_calls = [call]
        before_call = None
    elif policy == "flush":
        bound_calls = [call]
        before_call = flush_cache
    else:
        # A graph replay records one call per copy, so copies beyond
        # graph_max_calls would never run.
        max_copies = cfg.rotate_max_copies
        if cfg.mode == "graph":
            max_copies = min(max_copies, cfg.graph_max_calls)

        copies = copy_arguments(
            args,
            kwargs,
            max_copies,
            storage_bytes=storage_bytes,
            calls_to_cover=calls_to_cover,
            device=device,
        )
        bound_calls = [
            _bind_call(fn, copy_args, copy_kwargs)
            for copy_args, copy_kwargs in copies.argument_sets
        ]
        before_call = None
        cache.argument_set_count = len(bound_calls)
        cache.llc_coverage = copies.llc_coverage
        if not copies.meets_cold_sizing_target:
            notes.append(
                f"rotate: only {len(bound_calls)} argument sets fit (covers "
                f"{copies.llc_coverage:.2f}x the last-level cache); timing "
                "may be partly warm"
            )

    return CacheSetup(
        cache=cache,
        bound_calls=bound_calls,
        before_call=before_call,
        notes=notes,
    )
