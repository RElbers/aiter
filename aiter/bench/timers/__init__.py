# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Timing strategies.

Every timer takes ``bound_calls`` -- zero-argument callables, one per input
copy; timed call ``i`` runs ``bound_calls[i % len(bound_calls)]`` -- plus an
optional ``before_call`` hook (the cache flush), and returns a ``TimerOutput``.
"""

from .result import DeviceDetails, EagerDetails, GraphDetails, TimerOutput

__all__ = ["DeviceDetails", "EagerDetails", "GraphDetails", "TimerOutput"]
