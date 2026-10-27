"""aiter.bench: unified kernel benchmarking.

CPU tests cover configuration, statistics, the result schema and the device
timer's kernel-to-call matching (on synthetic profiler events). GPU tests run
each timing mode on small torch kernels and skip cleanly without a GPU.
"""

import contextlib
import json
import pathlib
import subprocess
import sys
import textwrap
from typing import Any, cast
from unittest.mock import patch

import pytest
import torch

_REPO_ROOT = pathlib.Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from aiter import bench
from aiter.bench import (
    BenchmarkConfig,
    BenchmarkResult,
    CacheResult,
    DeviceDetails,
    EmptyGraphCaptureError,
    EnvironmentResult,
    GraphDetails,
    TimingResult,
    run_bench,
    run_bench_fn,
    run_bench_modes,
    summarize,
)
from aiter.bench import env as bench_env
from aiter.bench.result import count_outliers
from aiter.bench.timers import device as device_timer
from aiter.bench.timers import graph as graph_timer
from aiter.bench.timers.graph import time_graph

needs_gpu = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs a GPU")
FAST: dict[str, Any] = {
    "sample_count": 20,
    "warmup_iters": 2,
    "record_gpu_state": False,
}


# --------------------------------------------------------------------------- #
# Configuration, statistics, result schema
# --------------------------------------------------------------------------- #


def test_config_defaults_per_mode():
    assert BenchmarkConfig(mode="device").cache_policy_or_default == "rotate"
    assert BenchmarkConfig(mode="graph").cache_policy_or_default == "rotate"
    assert BenchmarkConfig(mode="eager").cache_policy_or_default == "flush"
    assert (
        BenchmarkConfig(mode="graph", cache="rotate").cache_policy_or_default
        == "rotate"
    )


@pytest.mark.parametrize(
    "kwargs",
    [
        {"mode": "wall"},
        {"cache": "cold"},
        {"statistic": "p50"},
        {"sample_count": 1},
        {"warmup_iters": -1},
        {"graph_target_replay_us": 0},
        {"graph_max_calls": 0},
        {"rotate_max_copies": 0},
    ],
)
def test_config_rejects_invalid(kwargs):
    with pytest.raises(ValueError):
        BenchmarkConfig(**kwargs)


def test_summarize():
    stats = summarize([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 99.0])
    assert stats.median_us == 5.0
    assert stats.mean_us == 15.0
    assert (stats.min_us, stats.max_us) == (1.0, 99.0)
    assert stats.outliers == 1
    assert summarize([5.0]).std_us == 0.0
    with pytest.raises(ValueError):
        summarize([])


def test_outliers_are_counted_not_removed():
    samples = [10.0] * 20 + [50.0]
    assert count_outliers(samples) == 1
    assert summarize(samples).max_us == 50.0


def test_result_json_round_trip():
    result = BenchmarkResult(
        name="k",
        mode="graph",
        timing=TimingResult(selected_us=1.5, samples_us=[1.0, 2.0]),
        cache=CacheResult(requested_policy="warm", effective_policy="warm"),
        details=GraphDetails(calls_per_replay=4, captured_nodes=8),
        environment=EnvironmentResult(system={"gfx": "gfx942"}),
    )
    again = BenchmarkResult.from_json(result.to_json())
    assert again == result
    data = result.to_dict()
    data["field_from_the_future"] = 1
    for group in ("timing", "cache", "details", "environment"):
        data[group]["field_from_the_future"] = 1
    assert BenchmarkResult.from_dict(data) == result
    assert data["schema_version"] == bench.SCHEMA_VERSION


@pytest.mark.parametrize("run", [run_bench, run_bench_fn, run_bench_modes])
@pytest.mark.parametrize("positional", [False, True], ids=["keyword", "positional"])
def test_callable_keywords_do_not_conflict_with_benchmark_options(
    monkeypatch, run, positional
):
    calls = []

    def operation(value, *, name, config):
        calls.append((value, name, config))
        return value * config

    def measure(fn, args, kwargs, config, **metadata):
        assert config.sample_count == 4
        assert metadata == {"name": "benchmark", "flops": 6, "nbytes": 8}
        return fn(*args, **kwargs), BenchmarkResult(name=metadata["name"])

    monkeypatch.setattr(bench.api, "run_measurement", measure)
    options = {
        "config": BenchmarkConfig(sample_count=4),
        "name": "benchmark",
        "flops": 6,
        "nbytes": 8,
    }
    if positional:
        measured = run(operation, (2,), {"name": "callable", "config": 3}, **options)
    else:
        measured = run(
            operation, args=(2,), kwargs={"name": "callable", "config": 3}, **options
        )
    assert calls == [(2, "callable", 3)] * (3 if run is run_bench_modes else 1)
    if run is run_bench:
        output, result = measured
        assert output == 6
        assert result.name == "benchmark"
    elif run is run_bench_modes:
        assert set(measured) == {"device", "graph", "eager"}
        assert all(result.name == "benchmark" for result in measured.values())
    else:
        assert measured.name == "benchmark"


# --------------------------------------------------------------------------- #
# Device timer: matching GPU operations to calls (synthetic events)
# --------------------------------------------------------------------------- #


class _Event:
    def __init__(self, name, activity, device, start, end, corr=0):
        self._name, self._activity, self._device = name, activity, device
        self._start, self._end, self._corr = start, end, corr

    def name(self):
        return self._name

    def is_user_annotation(self):
        return self._activity in ("user_annotation", "gpu_user_annotation")

    def device_type(self):
        return self._device

    def correlation_id(self):
        return self._corr

    def start_ns(self):
        return self._start

    def end_ns(self):
        return self._end

    def duration_ns(self):
        return self._end - self._start


class _TypedEvent(_Event):
    def activity_type(self):
        return self._activity


CPU = torch.autograd.DeviceType.CPU
GPU = torch.autograd.DeviceType.CUDA
MARK = device_timer._CALL_RANGE_PREFIX


def _launch(corr, at):
    return _Event("hipLaunchKernel", "cuda_runtime", CPU, at, at + 1, corr)


def _kernel(name, corr, start, end):
    return _Event(name, "kernel", GPU, start, end, corr)


def test_split_calls_by_launch_range():
    events = [
        _Event(f"{MARK}0", "user_annotation", CPU, 0, 100, corr=1),
        _launch(1, 10),  # correlation ids may collide with CPU op ids
        _launch(2, 20),
        _launch(3, 30),
        _Event(f"{MARK}1", "user_annotation", CPU, 200, 300, corr=2),
        _launch(4, 210),
        _launch(5, 150),  # between ranges: the cache flush, ignored
        _Event("custom::op", "cpu_op", CPU, 150, 160, corr=1),
        _kernel("gemm", 1, 1000, 1010),
        _kernel("reduce", 2, 1010, 1013),
        _kernel("gemm", 3, 1015, 1025),  # same kernel twice in one call
        _kernel("gemm", 4, 2000, 2012),
        _kernel("flush", 5, 1500, 1600),
        _kernel("orphan", 99, 3000, 3001),  # no launch record
        _Event(f"{MARK}0", "gpu_user_annotation", GPU, 1000, 1025),
    ]
    per_call, unmatched = device_timer._group_device_events_by_call(events, 2)
    assert [op.name() for op in per_call[0]] == ["gemm", "reduce", "gemm"]
    assert [op.name() for op in per_call[1]] == ["gemm"]
    assert unmatched == 1


@pytest.mark.parametrize(
    "activity", ["cuda_runtime", "cuda_driver", "privateuse1_runtime"]
)
def test_device_prefers_activity_type_over_runtime_name(activity):
    events = [
        _TypedEvent(f"{MARK}0", "user_annotation", CPU, 0, 100),
        _TypedEvent("vendorLaunch", activity, CPU, 10, 11, corr=1),
        _TypedEvent("cudaFrontend", "cpu_op", CPU, 150, 160, corr=1),
        _TypedEvent("gemm", "kernel", GPU, 1000, 1010, corr=1),
        _TypedEvent(f"{MARK}0", "gpu_user_annotation", GPU, 1000, 1010),
    ]
    per_call, unmatched = device_timer._group_device_events_by_call(events, 1)
    assert [op.name() for op in per_call[0]] == ["gemm"]
    assert unmatched == 0


def test_device_summary_counts_overlapping_and_repeated_operations():
    calls = [
        [
            _kernel("gemm", 1, 1000, 4000),
            _kernel("gemm", 2, 2000, 4000),
            _kernel("copy", 3, 4000, 5000),
        ],
        [_kernel("gemm", 4, 6000, 8000), _kernel("zero", 5, 8000, 8500)],
    ]
    output = device_timer._summarize_device_events(calls, unlinked_count=1)
    assert isinstance(output.details, DeviceDetails)
    assert output.samples_us == [6.0, 2.5]
    assert output.details.span_us == 3.25
    assert output.details.per_operation_us == {"gemm": 3.5, "copy": 1.0, "zero": 0.5}
    assert output.details.median_operations_per_call == 2
    assert output.details.unlinked_device_event_count == 1
    assert not output.details.all_device_events_linked


def test_capture_rejects_zero_nodes_without_warning(monkeypatch):
    class EmptyGraph:
        def num_nodes(self):
            return 0

    monkeypatch.setattr(graph_timer.torch.cuda, "CUDAGraph", EmptyGraph)
    monkeypatch.setattr(
        graph_timer.torch.cuda,
        "graph",
        lambda graph, stream: contextlib.nullcontext(),
    )

    with pytest.raises(EmptyGraphCaptureError):
        graph_timer._capture([lambda: None], 1, cast(torch.cuda.Stream, object()))


def test_captured_node_count_supports_attribute_and_missing_api():
    class GraphWithNodeAttribute:
        _num_nodes = 3

    graph = cast(torch.cuda.CUDAGraph, GraphWithNodeAttribute())
    assert graph_timer._captured_node_count(graph) == 3
    assert (
        graph_timer._captured_node_count(cast(torch.cuda.CUDAGraph, object())) is None
    )


# --------------------------------------------------------------------------- #
# Environment snapshot parsing (no GPU, amd-smi output mocked)
# --------------------------------------------------------------------------- #


def test_snapshot_cli_parsing():
    metric = {
        "gpu_data": [
            {
                "clock": {
                    "gfx_0": {"clk": {"value": 2100}, "clk_locked": "DISABLED"},
                    "gfx_1": {"clk": {"value": 2050}, "clk_locked": "DISABLED"},
                    "gfx_2": "N/A",
                    "mem_0": {"clk": {"value": 2000}},
                },
                "usage": {"gfx_activity": {"value": 3}, "umc_activity": {"value": 1}},
                "perf_level": "AMDSMI_DEV_PERF_LEVEL_AUTO",
                "power": {"socket_power": {"value": 120}},
                "temperature": {"hotspot": {"value": 40}},
            }
        ]
    }
    with (
        patch.object(bench_env, "_smi_index", return_value=0),
        patch.object(bench_env, "pci_bdf", return_value="0000:00:00.0"),
        patch.object(bench_env, "_run_smi", return_value=metric),
    ):
        state = bench_env._snapshot_cli(0)
    assert state["gfx_clk_mhz"] == 2100
    assert state["mem_clk_mhz"] == 2000
    assert state["gfx_clk_locked"] == ["DISABLED"]
    assert state["gfx_activity_pct"] == 3
    assert state["perf_level"] == "AMDSMI_DEV_PERF_LEVEL_AUTO"


# --------------------------------------------------------------------------- #
# GPU: every mode on real kernels
# --------------------------------------------------------------------------- #


def _mul(x, y):
    return torch.mul(x, 2.0, out=y)


def _tensors(n=1 << 16):
    x = torch.randn(n, device="cuda")
    return x, torch.empty_like(x)


@needs_gpu
@pytest.mark.parametrize("mode", ["device", "graph", "eager"])
def test_modes_time_a_kernel(mode):
    x, y = _tensors()
    result = run_bench_fn(
        _mul,
        args=(x, y),
        config=BenchmarkConfig(mode=mode, **FAST),
        nbytes=2 * x.nbytes,
    )
    assert result.mode == mode
    assert len(result.timing.samples_us) == 20
    assert 0 < result.timing.min_us <= result.timing.median_us <= result.timing.max_us
    assert result.timing.selected_us == result.timing.median_us
    assert result.gb_s == pytest.approx(2 * x.nbytes / result.timing.median_us / 1e3)
    assert (
        result.cache.effective_policy
        == BenchmarkConfig(mode=mode).cache_policy_or_default
    )
    assert result.first_call_wall_ms is not None
    assert result.environment.system is not None
    assert result.environment.system["gfx"]


@needs_gpu
def test_metric_mean():
    x, y = _tensors()
    result = run_bench_fn(
        _mul, args=(x, y), config=BenchmarkConfig(statistic="mean", **FAST)
    )
    assert result.timing.selected_us == result.timing.mean_us


@needs_gpu
def test_device_counts_every_kernel():
    x, y = _tensors()
    z = torch.empty_like(x)

    def three_kernels(x, y, z):
        torch.mul(x, 2.0, out=y)
        torch.add(y, 1.0, out=z)
        torch.mul(x, 2.0, out=y)

    result = run_bench_fn(
        three_kernels, args=(x, y, z), config=BenchmarkConfig(mode="device", **FAST)
    )
    assert isinstance(result.details, DeviceDetails)
    assert result.details.median_operations_per_call == 3
    assert result.details.all_device_events_linked
    assert len(result.details.per_operation_us) == 2
    assert all(duration > 0 for duration in result.details.per_operation_us.values())


@needs_gpu
def test_graph_rejects_empty_capture():
    with pytest.raises(EmptyGraphCaptureError):
        run_bench_fn(lambda: None, config=BenchmarkConfig(mode="graph", **FAST))


@needs_gpu
def test_graph_refines_calls_per_replay(monkeypatch):
    x, y = _tensors(128)
    durations = iter([100.0, 1000.0])
    monkeypatch.setattr(graph_timer, "_estimate_call_us", lambda calls: 10.0)
    monkeypatch.setattr(
        graph_timer, "_measure_replay_us", lambda graph, count: next(durations)
    )
    output = time_graph([lambda: _mul(x, y)], 2, target_replay_us=1000, max_calls=4096)
    assert isinstance(output.details, GraphDetails)
    assert output.details.calls_per_replay == 1000
    assert len(output.samples_us) == 2
    assert not output.notes
    torch.testing.assert_close(y, x * 2)


@needs_gpu
def test_graph_respects_call_cap_and_normalizes_samples(monkeypatch):
    x, y = _tensors(128)
    elapsed_ms = []
    elapsed_time = torch.cuda.Event.elapsed_time

    def elapsed(start, end):
        duration = elapsed_time(start, end)
        elapsed_ms.append(duration)
        return duration

    monkeypatch.setattr(torch.cuda.Event, "elapsed_time", elapsed)
    output = time_graph([lambda: _mul(x, y)], 2, target_replay_us=1000, max_calls=4)
    assert isinstance(output.details, GraphDetails)
    assert 0 < output.details.calls_per_replay <= 4
    assert output.samples_us == pytest.approx(
        [
            duration * 1000 / output.details.calls_per_replay
            for duration in elapsed_ms[-2:]
        ]
    )


@needs_gpu
def test_rotate_needs_inputs():
    x, y = _tensors()
    with pytest.raises(ValueError, match="rotate"):
        run_bench_fn(
            lambda: _mul(x, y),
            config=BenchmarkConfig(mode="device", cache="rotate", **FAST),
        )


@needs_gpu
def test_rotate_copies_cover_samples_and_profiler_warmup():
    x, y = _tensors()
    result = run_bench_fn(
        _mul, args=(x, y), config=BenchmarkConfig(mode="device", cache="rotate", **FAST)
    )
    assert result.cache.argument_set_count == 20 + device_timer.PROFILER_WARMUP_CALLS
    assert not any("partly warm" in note for note in result.notes)


@needs_gpu
@pytest.mark.parametrize(
    "mode, fallback", [("device", "flush"), ("eager", "flush"), ("graph", "warm")]
)
def test_default_cache_on_closure_falls_back(mode, fallback):
    x, y = _tensors()
    result = run_bench_fn(lambda: _mul(x, y), config=BenchmarkConfig(mode=mode, **FAST))
    assert result.cache.effective_policy == fallback
    assert (mode == "graph") == any("timed warm" in note for note in result.notes)


@needs_gpu
def test_graph_rotates_by_default():
    x, y = _tensors(1 << 22)  # 32 MB of inputs: 16 copies cover 2x the LLC
    result = run_bench_fn(
        _mul, args=(x, y), config=BenchmarkConfig(mode="graph", **FAST)
    )
    assert result.cache.effective_policy == "rotate"
    assert isinstance(result.details, GraphDetails)
    assert result.cache.argument_set_count is not None
    assert result.details.calls_per_replay % result.cache.argument_set_count == 0
    assert not any("partly warm" in note for note in result.notes)


@needs_gpu
def test_graph_cannot_flush():
    x, y = _tensors()
    with pytest.raises(ValueError, match="rotate"):
        run_bench_fn(
            _mul,
            args=(x, y),
            config=BenchmarkConfig(mode="graph", cache="flush", **FAST),
        )


@needs_gpu
def test_auto_cache_resolves():
    x, y = _tensors()
    result = run_bench_fn(
        _mul, args=(x, y), config=BenchmarkConfig(mode="eager", cache="auto", **FAST)
    )
    assert result.cache.requested_policy == "auto"
    assert result.cache.effective_policy in ("warm", "flush")
    assert result.cache.cache_speedup_fraction is not None


@needs_gpu
def test_run_bench_returns_first_output():
    x, _ = _tensors()
    outputs = []

    def double(x):
        output = x * 2
        outputs.append(output)
        return output

    data, result = run_bench(double, args=(x,), config=BenchmarkConfig(**FAST))
    assert len(outputs) > 1
    assert data is outputs[0]
    torch.testing.assert_close(data, x * 2)
    assert isinstance(result, BenchmarkResult)
    assert result.name == "double"


@needs_gpu
def test_time_all_and_json():
    x, y = _tensors()
    results = run_bench_modes(_mul, args=(x, y), config=BenchmarkConfig(**FAST))
    assert set(results) == {"device", "graph", "eager"}
    assert len({r.first_call_wall_ms for r in results.values()}) == 1
    for result in results.values():
        assert BenchmarkResult.from_dict(json.loads(result.to_json())) == result


@needs_gpu
def test_device_timing_excludes_custom_operators_from_runtime_correlations():
    script = textwrap.dedent(
        """
        import torch
        from aiter.bench.timers.device import time_device

        library = torch.library.Library("custom_bench", "DEF")
        library.define("sleep(Tensor x) -> Tensor")

        def workload(x):
            for _ in range(10):
                torch.cuda._sleep(100)
            return x

        library.impl("sleep", workload, "CUDA")
        x = torch.ones(1, device="cuda")
        workload(x)
        torch.cuda.synchronize()

        output = time_device([lambda: torch.ops.custom_bench.sleep(x)], 4)

        assert len(output.samples_us) == 4
        assert output.details.median_operations_per_call == 10, output.details
        assert output.details.all_device_events_linked, output.details
        assert not output.notes, output.notes
        """
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=_REPO_ROOT,
        capture_output=True,
        check=False,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
