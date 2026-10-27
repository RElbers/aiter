# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Device timing: hardware kernel timestamps from torch.profiler.

Each timed call runs inside a ``record_function`` range. A GPU operation
(kernel, memcpy, memset) belongs to call ``i`` when the runtime call that
launched it -- linked by correlation id -- was issued inside range ``i``. No
kernel-name or repeating-pattern guess is involved, so repeated kernels and a
data-dependent kernel count are handled. Work launched outside the ranges
(the cache flush, profiler internals) is ignored.

Each sample is the busy time of one call: the sum of its GPU operations'
durations. ``span_us`` is the median first-start to last-end time per call;
``span_us - median busy`` is time spent in gaps between the call's kernels.
"""

import bisect
import statistics
from collections import defaultdict
from collections.abc import Callable, Sequence
from typing import Any

import torch
from torch.profiler import ProfilerActivity, profile, record_function

from .result import DeviceDetails, TimerOutput

_CALL_RANGE_PREFIX = "__aiter_bench_call_"
# Calls profiled but discarded: the profiler's first ranges are slow to start.
PROFILER_WARMUP_CALLS = 2
_RUNTIME_ACTIVITY_TYPES = ("cuda_runtime", "cuda_driver", "privateuse1_runtime")
_RUNTIME_API_PREFIXES = ("hip", "cuda", "cu")


def _is_runtime_event(event: Any) -> bool:
    activity_type = getattr(event, "activity_type", None)
    if activity_type is not None:
        return activity_type() in _RUNTIME_ACTIVITY_TYPES

    return (
        not event.is_user_annotation()
        and "::" not in event.name()
        and event.name().startswith(_RUNTIME_API_PREFIXES)
    )


def _group_device_events_by_call(
    events: Sequence[Any],
    call_count: int,
) -> tuple[list[list[Any]], int]:
    """Assign device operations to calls. Returns (events_by_call, unlinked_count)."""
    call_ranges: list[tuple[int, int, int]] = []  # (start_ns, end_ns, call index)
    launch_start_by_correlation: dict[int, int] = {}
    device_events: list[Any] = []

    for event in events:
        if event.device_type() != torch.autograd.DeviceType.CPU:
            if not event.is_user_annotation():
                device_events.append(event)
        elif event.is_user_annotation() and event.name().startswith(_CALL_RANGE_PREFIX):
            call_index = int(event.name()[len(_CALL_RANGE_PREFIX) :])
            call_ranges.append((event.start_ns(), event.end_ns(), call_index))
        elif _is_runtime_event(event):
            launch_start_by_correlation[event.correlation_id()] = event.start_ns()

    call_ranges.sort()
    range_starts_ns = [start_ns for start_ns, _, _ in call_ranges]

    events_by_call: list[list[Any]] = [[] for _ in range(call_count)]
    unlinked_count = 0

    for event in device_events:
        launch_start_ns = launch_start_by_correlation.get(event.correlation_id())
        if launch_start_ns is None:
            unlinked_count += 1
            continue

        range_index = bisect.bisect_right(range_starts_ns, launch_start_ns) - 1
        if range_index >= 0 and launch_start_ns <= call_ranges[range_index][1]:
            call_index = call_ranges[range_index][2]
            events_by_call[call_index].append(event)

    return events_by_call, unlinked_count


def _summarize_device_events(
    events_by_call: Sequence[Sequence[Any]],
    unlinked_count: int,
) -> TimerOutput:
    empty_call_count = sum(not events for events in events_by_call)
    if empty_call_count:
        raise RuntimeError(
            f"device timing found no GPU work for {empty_call_count} of {len(events_by_call)} calls. "
            "The callable launched nothing, or the profiler could not link its "
            "operations to their launches."
        )

    samples_us: list[float] = []
    spans_us: list[float] = []
    operation_counts: list[int] = []
    durations_by_name: dict[str, list[float]] = defaultdict(list)

    for events in events_by_call:
        samples_us.append(sum(event.duration_ns() for event in events) / 1000.0)
        spans_us.append(
            (
                max(event.end_ns() for event in events)
                - min(event.start_ns() for event in events)
            )
            / 1000.0
        )
        operation_counts.append(len(events))

        call_durations_by_name: dict[str, float] = defaultdict(float)
        for event in events:
            call_durations_by_name[event.name()] += event.duration_ns() / 1000.0

        for name, duration_us in call_durations_by_name.items():
            durations_by_name[name].append(duration_us)

    notes: list[str] = []
    if min(operation_counts) != max(operation_counts):
        notes.append(
            f"device operation count per call varies from {min(operation_counts)} "
            f"to {max(operation_counts)}"
        )

    if unlinked_count:
        notes.append(
            f"{unlinked_count} GPU operations could not be linked to a launch and "
            "were not counted"
        )

    details = DeviceDetails(
        # A kernel missing from some calls contributes its median over the
        # calls where it ran.
        per_operation_us={
            name: statistics.median(durations)
            for name, durations in durations_by_name.items()
        },
        span_us=statistics.median(spans_us),
        median_operations_per_call=statistics.median_low(operation_counts),
        unlinked_device_event_count=unlinked_count,
    )
    return TimerOutput(
        samples_us=samples_us,
        details=details,
        notes=notes,
    )


def time_device(
    bound_calls: Sequence[Callable[[], Any]],
    sample_count: int,
    *,
    before_call: Callable[[], None] | None = None,
) -> TimerOutput:
    call_count = PROFILER_WARMUP_CALLS + sample_count

    torch.cuda.synchronize()
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as profiler:
        for call_index in range(call_count):
            if before_call is not None:
                before_call()

            with record_function(f"{_CALL_RANGE_PREFIX}{call_index}"):
                bound_calls[call_index % len(bound_calls)]()
        torch.cuda.synchronize()

    events_by_call, unlinked_count = _group_device_events_by_call(
        profiler.profiler.kineto_results.events(),
        call_count,
    )
    events_by_call = events_by_call[PROFILER_WARMUP_CALLS:]
    return _summarize_device_events(events_by_call, unlinked_count)
