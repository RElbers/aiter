# aiter.bench: unified kernel benchmarking

One timing interface for every aiter backend (FlyDSL, CK, ASM, HIP, Triton).
Every result records which mode, cache policy and environment produced it.

## Source layout

| Module | Responsibility |
|---|---|
| [`__init__.py`](__init__.py) | Public imports from `aiter.bench` |
| [`api.py`](api.py) | Benchmark entry-point implementations and mode iteration |
| [`runner.py`](runner.py) | Measurement lifecycle and result assembly |
| [`config.py`](config.py) | Measurement configuration and validation |
| [`result.py`](result.py) | Result dataclasses, statistics and serialization |
| [`cache.py`](cache.py) | Cache selection and argument preparation |
| [`env.py`](env.py) | System metadata, GPU snapshots and telemetry |
| [`timers/result.py`](timers/result.py) | Typed timer outputs and mode-specific details |
| [`timers/device.py`](timers/device.py) | Profiler event attribution and aggregation |
| [`timers/eager.py`](timers/eager.py) | Per-call GPU-event timing |
| [`timers/graph.py`](timers/graph.py) | Graph capture, replay sizing and timing |

## Quick start

```python
from aiter.bench import BenchmarkConfig, run_bench, run_bench_fn, run_bench_modes

# Like run_perftest, but returns the full result.
out, result = run_bench(my_kernel, args=(x, w, out_buf), config=BenchmarkConfig(mode="graph"))
print(result.timing.median_us, result.cache.effective_policy, result.details.calls_per_replay)

# Any callable. Pass inputs through `args` so cache="rotate" can copy them.
result = run_bench_fn(my_kernel, args=(x, w, out_buf), config=BenchmarkConfig(mode="device"),
                      flops=2 * m * n * k, nbytes=(m * k + n * k + m * n) * 2)
print(result.to_json())

# All three modes on the same inputs.
results = run_bench_modes(my_kernel, args=(x, w, out_buf))
host_overhead_us = results["eager"].timing.median_us - results["graph"].timing.median_us
```

`BenchmarkConfig` is the only place timing is configured. Pass the same object
to every call you want to compare.

## Timing modes

| Mode | Measures | How | Use it for |
|---|---|---|---|
| `device` (default) | GPU busy time per call: the sum of every kernel, memcpy and memset the call launches | torch.profiler hardware timestamps. Each call runs in its own profiler range, and a GPU operation belongs to the call whose range issued its launch (matched by correlation id, not kernel name). | Kernel work in isolation; per-kernel breakdown (`details.per_operation_us`); comparable to `run_perftest` |
| `graph` | Steady-state time per call when replayed from a HIP graph, as graph-based serving runs it | K calls recorded into one graph; each sample is one replay ÷ K. K is chosen so one replay takes at least `graph_target_replay_us` (default 1000 µs). Fails with `EmptyGraphCaptureError` when nothing was captured. |  Production-style A/B coparisons|
| `eager` | Per-call time including host launch overhead | GPU events around every call; one sync at the end (`do_bench` pattern). `eager_sync_each=True` drains every call, adding the full host round trip. | Non-graph serving; launch-overhead questions |

Reading the benchmarking times

| Observation | Meaning |
|---|---|
| `eager` ≫ `graph` | Host launch overhead dominates at this shape. |
| `device` `details.span_us` > `timing.median_us` | Gaps between the call's kernels (`span_us − median_us`). |
| `graph` > `device` for kernels under ~2 µs | Graph replay pays a per-kernel GPU dispatch cost that profiler timestamps leave out. Real in serving. |
| `graph` < `device` for kernels of ~2–10 µs | Back-to-back kernels in a graph overlap dispatch with the previous kernel's tail; device mode times each kernel separately. |
| `graph` ≈ `device` above ~20 µs | Fixed per-kernel effects are negligible. |

So `device` and `graph` are different measurements, not two estimates of one
number. Compare like with like: same mode, cache policy and `BenchmarkConfig`.

### Why graph mode records K calls

A `g.replay()` costs the host about 6 µs and the GPU a further 5–10 µs of
fixed setup. With one call per graph, every kernel faster than that reads as
the replay cost. Spreading the fixed cost over K calls bounds its share at roughly
`fixed_cost / graph_target_replay_us`, about 1% at the default.

## Cache policies

| Policy | Effect | Default for |
|---|---|---|
| `warm` | Same inputs every call; they stay cache-resident. | — (opt in) |
| `rotate` | Deep copies of the inputs, cycled so each call reads data that is not cache-resident (sized from the last-level cache, 256 MB Infinity Cache on gfx942/gfx950). Needs inputs passed as arguments. In graph mode each copy is one recorded call, so at most `graph_max_calls` copies. | `device` (matches `run_perftest`), `graph` |
| `flush` | Overwrite a buffer twice the last-level cache size before each call, outside the timed span. Not available in graph mode (use `rotate`). | `eager` (matches `do_bench`) |
| `auto` | Probe warm vs flushed calls; go cold (`flush`, or `rotate` in graph mode) when cold is more than 5% slower. The measured difference is in `cache.cache_speedup_fraction`. | — |

`cache=None` (the default) picks the per-mode default above: **every mode
defaults to cold inputs**. A callable without inputs (a closure) has nothing
to rotate, so the default falls back to `flush`, or to `warm` in graph mode
with a note. The policy that actually ran is always in `result.cache.effective_policy`.
A `rotate` that could not fit enough copies adds a "partly warm" note. This
happens in graph mode for small inputs: being cold needs about
2 × 256 MB / input size copies, more than `graph_max_calls` (4096) for inputs
under about 128 KB. Kernels with inputs that small are rarely limited by
memory traffic.

## Results

`BenchmarkResult` (`to_dict()`, `to_json()`, `from_dict()`, versioned by
`schema_version`):

- **Timing** (`timing`): `samples_us`, `median_us`, `mean_us`, `min_us`, `max_us`,
  `std_us`, `outliers` (IQR count; outliers are reported, never dropped), and
  `selected_us`: the statistic chosen by `BenchmarkConfig.statistic` (`median` by
  default, `mean` to match `run_perftest` and the tuners).
- **Mode details:** `cache.effective_policy`, `cache.argument_set_count`,
  `details.calls_per_replay`, `details.per_operation_us`, `details.span_us`,
  `details.median_operations_per_call`, `details.all_device_events_linked`.
- **Cost:** `first_call_wall_ms` (wall time of the first call, including JIT).
- **Throughput:** `tflops`, `gb_s` when `flops` / `nbytes` are given.
- **Context** (`environment`): `system` (GPU, gfx, CU count, HIP/torch/Triton/FlyDSL
  versions, aiter git SHA), `gpu_state_before` / `gpu_state_after` (clocks,
  performance level, activity, power, temperature), `telemetry`; plus `notes`.

`notes` carries anything that weakens a number: a GPU already busy before
timing, a varying kernel count, operations the profiler could not attribute,
a partly warm rotation.

## Environment and telemetry

Nothing in `aiter.bench` changes GPU settings. Clocks and the performance
level are only recorded: from the `amdsmi` Python binding when it imports,
otherwise from the `amd-smi` CLI (`record_gpu_state=False` skips this).
`telemetry=True` samples clocks, power and activity on a background thread
during the timed window; it needs the `amdsmi` binding.

## Not yet implemented

Tracked in the plan: rocprofiler-sdk as an alternative device backend,
per-arch peaks (`tflops_peak_ratio`, `gbps_peak_ratio` are `None`), and the
case-level API (`bench`, `sweep`, `compare`, `run_candidates`, CLI).
