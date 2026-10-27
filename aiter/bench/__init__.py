# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Unified kernel benchmarking. See aiter/bench/README.md."""

from .api import run_bench, run_bench_fn, run_bench_modes
from .config import CACHE_POLICIES, DEFAULT_CACHE, MODES, STATISTICS, BenchmarkConfig
from .result import (
    SCHEMA_VERSION,
    BenchmarkResult,
    CacheResult,
    EnvironmentResult,
    TimingResult,
    summarize,
)
from .timers import DeviceDetails, EagerDetails, GraphDetails
from .timers.graph import EmptyGraphCaptureError, GraphCaptureError

__all__ = [
    "CACHE_POLICIES",
    "DEFAULT_CACHE",
    "MODES",
    "SCHEMA_VERSION",
    "STATISTICS",
    "BenchmarkConfig",
    "BenchmarkResult",
    "CacheResult",
    "DeviceDetails",
    "EagerDetails",
    "EmptyGraphCaptureError",
    "EnvironmentResult",
    "GraphCaptureError",
    "GraphDetails",
    "TimingResult",
    "run_bench",
    "run_bench_fn",
    "run_bench_modes",
    "summarize",
]
