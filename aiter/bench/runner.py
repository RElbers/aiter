# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

import time
from collections.abc import Callable
from typing import Any

import torch

from .cache import CacheSetup, prepare_cache
from .config import BenchmarkConfig
from .env import gpu_state, system_info, telemetry_window
from .result import BenchmarkResult, CacheResult, EnvironmentResult, summarize
from .timers import TimerOutput
from .timers.device import PROFILER_WARMUP_CALLS, time_device
from .timers.eager import time_eager
from .timers.graph import time_graph

# Notes flag a GPU that was already busy before timing started.
_BUSY_ACTIVITY_PCT = 5


def run_measurement(
    fn: Callable[..., Any],
    args: tuple[Any, ...],
    kwargs: dict[str, Any],
    config: BenchmarkConfig,
    *,
    name: str | None,
    flops: float | None,
    nbytes: float | None,
) -> tuple[Any, BenchmarkResult]:
    device = torch.cuda.current_device()
    notes: list[str] = []

    state_before = gpu_state(device) if config.record_gpu_state else None
    activity = (state_before or {}).get("gfx_activity_pct")
    if activity is not None and activity > _BUSY_ACTIVITY_PCT:
        notes.append(f"GPU was already busy before timing (gfx_activity={activity}%)")

    torch.cuda.synchronize()
    start = time.perf_counter()
    first_call_output = fn(*args, **kwargs)
    torch.cuda.synchronize()
    first_call_wall_ms = (time.perf_counter() - start) * 1000.0

    for _ in range(config.warmup_iters):
        fn(*args, **kwargs)
    torch.cuda.synchronize()

    # Graph replays repeat the recorded calls, so copies are reused.
    calls_to_cover = None if config.mode == "graph" else config.sample_count
    # Device mode profiles a few extra calls that it discards.
    if config.mode == "device":
        calls_to_cover = config.sample_count + PROFILER_WARMUP_CALLS

    setup = prepare_cache(
        fn,
        args,
        kwargs,
        config,
        calls_to_cover=calls_to_cover,
        device=device,
    )
    notes.extend(setup.notes)

    with telemetry_window(device, config.telemetry) as window:
        output = _run_timer(setup, config)

    notes.extend(output.notes)
    if window.error:
        notes.append(window.error)

    state_after = gpu_state(device) if config.record_gpu_state else None
    environment = EnvironmentResult(
        system=system_info(device).to_dict(),
        gpu_state_before=state_before,
        gpu_state_after=state_after,
        telemetry=window.summary,
    )

    result = _build_result(
        config,
        setup.cache,
        output,
        environment,
        name=name or getattr(fn, "__name__", None),
        first_call_wall_ms=first_call_wall_ms,
        flops=flops,
        nbytes=nbytes,
        notes=notes,
    )
    return first_call_output, result


def _run_timer(
    setup: CacheSetup,
    config: BenchmarkConfig,
) -> TimerOutput:
    if config.mode == "device":
        return time_device(
            setup.bound_calls,
            config.sample_count,
            before_call=setup.before_call,
        )

    if config.mode == "graph":
        return time_graph(
            setup.bound_calls,
            config.sample_count,
            target_replay_us=config.graph_target_replay_us,
            max_calls=config.graph_max_calls,
        )

    return time_eager(
        setup.bound_calls,
        config.sample_count,
        before_call=setup.before_call,
        sync_each=config.eager_sync_each,
    )


def _build_result(
    config: BenchmarkConfig,
    cache: CacheResult,
    output: TimerOutput,
    environment: EnvironmentResult,
    *,
    name: str | None,
    first_call_wall_ms: float,
    flops: float | None,
    nbytes: float | None,
    notes: list[str],
) -> BenchmarkResult:
    timing = summarize(
        output.samples_us,
        statistic=config.statistic,
        warmup_iters=config.warmup_iters,
    )

    return BenchmarkResult(
        name=name,
        mode=config.mode,
        timing=timing,
        cache=cache,
        details=output.details,
        environment=environment,
        first_call_wall_ms=first_call_wall_ms,
        flops=flops,
        nbytes=nbytes,
        tflops=flops / timing.median_us / 1e6 if flops else None,
        gb_s=nbytes / timing.median_us / 1e3 if nbytes else None,
        notes=notes,
    )
