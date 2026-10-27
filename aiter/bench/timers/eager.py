# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Eager timing: one GPU event pair around each call.

Default is the triton.testing.do_bench pattern: events around every call, one
synchronize at the end, so the host runs ahead and per-launch dispatch cost is
amortized the way a steady non-graph serving loop amortizes it. With
``sync_each`` every call is drained before the next one, so each sample also
contains the host launch round trip.
"""

from collections.abc import Callable, Sequence
from typing import Any

import torch

from .result import EagerDetails, TimerOutput


def time_eager(
    bound_calls: Sequence[Callable[[], Any]],
    sample_count: int,
    *,
    before_call: Callable[[], None] | None = None,
    sync_each: bool = False,
) -> TimerOutput:
    starts = [torch.cuda.Event(enable_timing=True) for _ in range(sample_count)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(sample_count)]

    torch.cuda.synchronize()
    for i in range(sample_count):
        if before_call is not None:
            # Issued before the start event, so it is never inside a sample.
            before_call()

        starts[i].record()
        bound_calls[i % len(bound_calls)]()
        ends[i].record()

        if sync_each:
            torch.cuda.synchronize()

    torch.cuda.synchronize()
    samples_us = [start.elapsed_time(end) * 1000.0 for start, end in zip(starts, ends)]
    details = EagerDetails(sync_each=sync_each)
    return TimerOutput(samples_us=samples_us, details=details)
