# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

from dataclasses import dataclass, field


@dataclass(frozen=True)
class EagerDetails:
    sync_each: bool


@dataclass(frozen=True)
class GraphDetails:
    calls_per_replay: int
    captured_nodes: int | None


@dataclass(frozen=True)
class DeviceDetails:
    per_operation_us: dict[str, float]
    span_us: float
    median_operations_per_call: int
    unlinked_device_event_count: int
    all_device_events_linked: bool = field(init=False)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "all_device_events_linked",
            self.unlinked_device_event_count == 0,
        )


@dataclass(frozen=True)
class TimerOutput:
    samples_us: list[float]
    details: EagerDetails | GraphDetails | DeviceDetails
    notes: list[str] = field(default_factory=list)
