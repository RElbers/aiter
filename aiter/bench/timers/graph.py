# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
"""Graph timing: replay K calls recorded into one HIP graph.

Each ``g.replay()`` costs the host a few microseconds (about 6 us on MI350P).
With one call per graph, any kernel shorter than that measures the host's
replay rate instead of device work. Recording K calls per graph, with K chosen
so one replay takes at least ``target_replay_us``, keeps the GPU ahead of the
host; each sample is one replay divided by K.

All samples are enqueued back to back with a single synchronize at the end,
so no sample contains a host-side gap.
"""

import math
import warnings
from collections.abc import Callable, Sequence
from typing import Any

import torch

from .result import GraphDetails, TimerOutput


class EmptyGraphCaptureError(RuntimeError):
    def __init__(self) -> None:
        super().__init__(
            "graph capture recorded no GPU work. The callable launched its "
            "kernels outside the capture stream (for example on the NULL "
            "stream) or launched nothing. Pass torch.cuda.current_stream() to "
            "the kernel launch (FlyDSL: a 'stream: fx.Stream' launcher "
            "argument) so it can be captured."
        )


class GraphCaptureError(RuntimeError):
    pass


def _captured_node_count(graph: torch.cuda.CUDAGraph) -> int | None:
    """Return the captured node count when the torch build exposes it."""
    for attr in ("num_nodes", "_num_nodes"):
        try:
            probe = getattr(graph, attr)
            value = probe() if callable(probe) else probe
        except (AttributeError, RuntimeError, TypeError):
            continue
        if isinstance(value, int):
            return value
    return None


def _capture(
    bound_calls: Sequence[Callable[[], Any]],
    calls_per_replay: int,
    stream: torch.cuda.Stream,
) -> tuple[torch.cuda.CUDAGraph, int | None]:
    graph = torch.cuda.CUDAGraph()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        try:
            with torch.cuda.graph(graph, stream=stream):
                for j in range(calls_per_replay):
                    bound_calls[j % len(bound_calls)]()
        except Exception as error:
            raise GraphCaptureError(
                f"callable is not graph-capturable ({type(error).__name__}: "
                f"{error}). Common causes: host synchronization, .item(), "
                "host-to-device copies or allocations outside the caching "
                "allocator inside the call."
            ) from error
    captured_nodes = _captured_node_count(graph)
    empty_warning = any("empty" in str(w.message).lower() for w in caught)
    if captured_nodes == 0 or (captured_nodes is None and empty_warning):
        raise EmptyGraphCaptureError()
    return graph, captured_nodes


def _measure_replay_us(graph: torch.cuda.CUDAGraph, replay_count: int) -> float:
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(replay_count):
        graph.replay()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) * 1000.0 / replay_count


def _estimate_call_us(
    bound_calls: Sequence[Callable[[], Any]],
    probe_count: int = 5,
) -> float:
    samples_us = []
    for i in range(probe_count):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        bound_calls[i % len(bound_calls)]()
        end.record()
        torch.cuda.synchronize()
        samples_us.append(start.elapsed_time(end) * 1000.0)
    return max(sorted(samples_us)[len(samples_us) // 2], 1e-3)


def _choose_calls_per_replay(
    target_us: float,
    call_us: float,
    copy_count: int,
    max_calls: int,
) -> int:
    """Calls per replay: a whole number of passes over the input copies."""
    passes = math.ceil(target_us / call_us / copy_count)
    passes = max(1, min(passes, max_calls // copy_count))
    return passes * copy_count


def _prepare_graph(
    bound_calls: Sequence[Callable[[], Any]],
    target_replay_us: float,
    max_calls: int,
) -> tuple[torch.cuda.CUDAGraph, GraphDetails, list[str]]:
    notes: list[str] = []

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())

    # Warm on the capture stream: lazy initialization and allocator growth are
    # not allowed during capture.
    with torch.cuda.stream(stream):
        for call in bound_calls:
            call()
    torch.cuda.synchronize()

    # Every rotated copy appears equally often per replay, otherwise the
    # replay would favour a subset of the copies.
    copy_count = min(len(bound_calls), max_calls)
    # The eager estimate includes host overhead, so it overestimates small
    # kernels and K comes out too small; one refinement from the measured
    # replay corrects that.
    call_us = _estimate_call_us(bound_calls)
    call_count = _choose_calls_per_replay(
        target_replay_us,
        call_us,
        copy_count,
        max_calls,
    )
    graph, captured_nodes = _capture(
        bound_calls,
        call_count,
        stream,
    )

    replay_us = _measure_replay_us(graph, 3)
    if replay_us < target_replay_us / 2 and call_count < max_calls:
        refined_count = _choose_calls_per_replay(
            target_replay_us,
            replay_us / call_count,
            copy_count,
            max_calls,
        )

        if refined_count > call_count:
            graph.reset()
            call_count = refined_count
            graph, captured_nodes = _capture(
                bound_calls,
                call_count,
                stream,
            )
            replay_us = _measure_replay_us(graph, 3)

    if replay_us < target_replay_us / 2:
        notes.append(
            f"graph replay is only {replay_us:.1f} us with {call_count} calls "
            f"(graph_max_calls={max_calls}); per-replay host cost may be "
            "included. Raise graph_max_calls."
        )

    details = GraphDetails(
        calls_per_replay=call_count,
        captured_nodes=captured_nodes,
    )
    return graph, details, notes


def time_graph(
    bound_calls: Sequence[Callable[[], Any]],
    sample_count: int,
    *,
    target_replay_us: float,
    max_calls: int,
) -> TimerOutput:
    graph, details, notes = _prepare_graph(
        bound_calls,
        target_replay_us,
        max_calls,
    )
    graph.replay()

    starts = [torch.cuda.Event(enable_timing=True) for _ in range(sample_count)]
    ends = [torch.cuda.Event(enable_timing=True) for _ in range(sample_count)]
    for i in range(sample_count):
        starts[i].record()
        graph.replay()
        ends[i].record()

    torch.cuda.synchronize()
    samples_us = [
        start.elapsed_time(end) * 1000.0 / details.calls_per_replay
        for start, end in zip(starts, ends)
    ]
    graph.reset()
    return TimerOutput(samples_us=samples_us, details=details, notes=notes)
