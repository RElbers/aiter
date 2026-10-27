# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""``BenchmarkResult``: everything one timing run produced, JSON-serializable."""

import dataclasses
import json
import statistics
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any

from .config import CachePolicy, ResolvedCachePolicy, Statistic
from .timers.result import DeviceDetails, EagerDetails, GraphDetails

# Bump when a field is renamed or its meaning changes; adding a field is fine.
SCHEMA_VERSION = 1

# Tukey fences: samples outside [Q1 - k*IQR, Q3 + k*IQR] count as outliers.
_OUTLIER_IQR_FACTOR = 1.5


@dataclass
class TimingResult:
    statistic: Statistic = "median"
    selected_us: float = 0.0
    samples_us: list[float] = field(default_factory=list)
    median_us: float = 0.0
    mean_us: float = 0.0
    min_us: float = 0.0
    max_us: float = 0.0
    std_us: float = 0.0
    outliers: int = 0  # reported only; outliers stay in every other statistic
    warmup_iters: int = 0
    sample_count: int = 0


@dataclass
class CacheResult:
    requested_policy: CachePolicy | None = None
    effective_policy: ResolvedCachePolicy | None = None
    cache_speedup_fraction: float | None = None  # auto: (cold - warm) / cold
    argument_set_count: int | None = None
    llc_coverage: float | None = None


@dataclass
class EnvironmentResult:
    system: dict[str, Any] | None = None
    gpu_state_before: dict[str, Any] | None = None
    gpu_state_after: dict[str, Any] | None = None
    telemetry: dict[str, Any] | None = None


@dataclass
class BenchmarkResult:
    name: str | None = None
    mode: str = ""
    timing: TimingResult = field(default_factory=TimingResult)
    cache: CacheResult = field(default_factory=CacheResult)
    details: EagerDetails | GraphDetails | DeviceDetails | None = None
    environment: EnvironmentResult = field(default_factory=EnvironmentResult)
    first_call_wall_ms: float | None = None
    notes: list[str] = field(default_factory=list)

    # Throughput (from median_us)
    flops: float | None = None
    nbytes: float | None = None
    tflops: float | None = None
    gb_s: float | None = None
    tflops_peak_ratio: float | None = None
    gbps_peak_ratio: float | None = None

    backend: str | None = None
    kernel: str | None = None
    config: dict[str, Any] | None = None
    tunables: dict[str, Any] | None = None
    # Correctness: None (not checked), "pass", "fail" or "skipped"
    verified: str | None = None
    err: float | None = None
    schema_version: int = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), default=str, **kwargs)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "BenchmarkResult":
        fields = _known_fields(cls, data)
        fields["timing"] = TimingResult(
            **_known_fields(TimingResult, data.get("timing", {}))
        )
        fields["cache"] = CacheResult(
            **_known_fields(CacheResult, data.get("cache", {}))
        )
        fields["environment"] = EnvironmentResult(
            **_known_fields(EnvironmentResult, data.get("environment", {}))
        )

        details = data.get("details")
        if details is not None:
            detail_types = {
                "eager": EagerDetails,
                "graph": GraphDetails,
                "device": DeviceDetails,
            }
            mode = data.get("mode")
            if mode not in detail_types:
                raise ValueError(f"cannot reconstruct timing details for mode={mode!r}")

            detail_type = detail_types[mode]
            fields["details"] = detail_type(**_known_fields(detail_type, details))

        return cls(**fields)

    @classmethod
    def from_json(cls, text: str) -> "BenchmarkResult":
        return cls.from_dict(json.loads(text))

    def summary(self) -> dict[str, Any]:
        """Compact columns for @benchmark tables (no samples, no metadata)."""
        return {
            "mode": self.mode,
            "cache": self.cache.effective_policy,
            "selected_us": self.timing.selected_us,
            "median_us": self.timing.median_us,
            "mean_us": self.timing.mean_us,
            "std_us": self.timing.std_us,
            "tflops": self.tflops,
            "gb_s": self.gb_s,
        }


def count_outliers(
    samples_us: Sequence[float],
    iqr_factor: float = _OUTLIER_IQR_FACTOR,
) -> int:
    if len(samples_us) < 4:
        return 0

    q1, _, q3 = statistics.quantiles(samples_us, n=4)
    lower = q1 - iqr_factor * (q3 - q1)
    upper = q3 + iqr_factor * (q3 - q1)
    return sum(sample < lower or sample > upper for sample in samples_us)


def summarize(
    samples_us: Sequence[float],
    *,
    statistic: Statistic = "median",
    warmup_iters: int = 0,
) -> TimingResult:
    samples = list(samples_us)
    if not samples:
        raise ValueError("no samples to summarize")

    median_us = statistics.median(samples)
    mean_us = statistics.fmean(samples)
    return TimingResult(
        statistic=statistic,
        selected_us=median_us if statistic == "median" else mean_us,
        samples_us=samples,
        median_us=median_us,
        mean_us=mean_us,
        min_us=min(samples),
        max_us=max(samples),
        std_us=statistics.stdev(samples) if len(samples) > 1 else 0.0,
        outliers=count_outliers(samples),
        warmup_iters=warmup_iters,
        sample_count=len(samples),
    )


def _known_fields(cls: Any, data: dict[str, Any]) -> dict[str, Any]:
    names = {field.name for field in dataclasses.fields(cls) if field.init}
    return {name: value for name, value in data.items() if name in names}
