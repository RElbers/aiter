# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Public entry points: ``run_bench_fn``, ``run_bench`` and ``run_bench_modes``."""

from collections.abc import Callable, Mapping, Sequence
from typing import Any, ParamSpec, TypeVar

from .config import MODES, BenchmarkConfig, Mode
from .result import BenchmarkResult
from .runner import run_measurement

_P = ParamSpec("_P")
_R = TypeVar("_R")


def run_bench_fn(
    fn: Callable[_P, _R],
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    *,
    config: BenchmarkConfig | None = None,
    name: str | None = None,
    flops: float | None = None,
    nbytes: float | None = None,
) -> BenchmarkResult:
    """Time ``fn(*args, **kwargs)``. Pass inputs via ``args`` for cache='rotate'.

    ``flops`` and ``nbytes`` per call give ``tflops`` and ``gb_s``.
    """
    _, result = run_bench(
        fn,
        args=args,
        kwargs=kwargs,
        config=config,
        name=name,
        flops=flops,
        nbytes=nbytes,
    )
    return result


def run_bench(
    fn: Callable[_P, _R],
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    *,
    config: BenchmarkConfig | None = None,
    name: str | None = None,
    flops: float | None = None,
    nbytes: float | None = None,
) -> tuple[_R, BenchmarkResult]:
    """Returns ``(data, BenchmarkResult)``.

    ``data`` is the output of the first call, made on the caller's original
    inputs before any timing.
    """
    return run_measurement(
        fn,
        tuple(args),
        dict(kwargs or {}),
        config or BenchmarkConfig(),
        name=name,
        flops=flops,
        nbytes=nbytes,
    )


def run_bench_modes(
    fn: Callable[_P, _R],
    args: Sequence[Any] = (),
    kwargs: Mapping[str, Any] | None = None,
    *,
    config: BenchmarkConfig | None = None,
    modes: Sequence[Mode] = MODES,
    name: str | None = None,
    flops: float | None = None,
    nbytes: float | None = None,
) -> dict[Mode, BenchmarkResult]:
    """Time ``fn`` in each of ``modes``; returns ``{mode: BenchmarkResult}``.

    ``eager - graph`` is the host launch overhead per call; a device-mode
    ``span_us - median_us`` is the time between the call's kernels.
    """
    base = config or BenchmarkConfig()
    results: dict[Mode, BenchmarkResult] = {}

    for mode in modes:
        cfg = base.replace(mode=mode)
        if base.cache == "flush" and mode == "graph":
            cfg = cfg.replace(cache="rotate")

        results[mode] = run_bench_fn(
            fn,
            args=args,
            kwargs=kwargs,
            config=cfg,
            name=name,
            flops=flops,
            nbytes=nbytes,
        )

    if results:
        first_call_wall_ms = results[modes[0]].first_call_wall_ms
        for result in results.values():
            result.first_call_wall_ms = first_call_wall_ms

    return results
