# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tests any operation can opt into; see ``signatures`` for what they receive."""

from __future__ import annotations

import ctypes
import unittest
from collections.abc import Callable
from typing import Any

import torch
from torch._subclasses.fake_tensor import FakeTensorMode

from aiter.catalog.registry import KernelEntry, OperationEntry
from aiter.catalog.testing.signatures import VerifyOutputs
from aiter.catalog.testing.tensors import (
    assign,
    clone,
    equal,
    map_tensors,
    poison,
    storage,
    tensors,
)

DEVICE = "cuda"


# Kernel tests


def matches_reference(
    kernel: KernelEntry, instance, case, op_input, verify_outputs: VerifyOutputs
) -> None:
    verify_outputs(op_input, kernel.run(clone(op_input), instance))


def leaves_inputs_unchanged(
    kernel: KernelEntry, instance, case, op_input, verify_outputs: VerifyOutputs
) -> None:
    copy = clone(op_input)
    kernel.run(copy, instance)
    assert equal(copy, op_input), "the kernel wrote to its input"


def does_not_alias_inputs(
    kernel: KernelEntry, instance, case, op_input, verify_outputs: VerifyOutputs
) -> None:
    copy = clone(op_input)
    output = kernel.run(copy, instance)
    input_storages = {storage(t) for t in tensors(copy)}
    aliased = [t for t in tensors(output) if storage(t) in input_storages]
    assert not aliased, "an output shares memory with an input"


def is_deterministic(
    kernel: KernelEntry, instance, case, op_input, verify_outputs: VerifyOutputs
) -> None:
    first = kernel.run(clone(op_input), instance)
    second = kernel.run(clone(op_input), instance)
    assert equal(first, second), "two launches on the same input differ"


def graph_capture(
    kernel: KernelEntry, instance, case, op_input, verify_outputs: VerifyOutputs
) -> None:
    _check_graph_capture(
        lambda x: kernel.run(x, instance), case, op_input, verify_outputs
    )


def on_side_stream(
    kernel: KernelEntry, instance, case, op_input, verify_outputs: VerifyOutputs
) -> None:
    _check_on_side_stream(lambda x: kernel.run(x, instance), op_input)


# Op tests


def op_matches_reference(
    op: OperationEntry, case, op_input, verify_outputs: VerifyOutputs
) -> None:
    verify_outputs(op_input, op.run(clone(op_input)))


def op_graph_capture(
    op: OperationEntry, case, op_input, verify_outputs: VerifyOutputs
) -> None:
    _check_graph_capture(op.run, case, op_input, verify_outputs)


def fake_matches_real(
    op: OperationEntry, case, op_input, verify_outputs: VerifyOutputs
) -> None:
    """The fake implementation torch.compile traces with predicts the real output."""
    real = op.run(clone(op_input))
    with FakeTensorMode() as mode:
        fake = op.run(map_tensors(op_input, mode.from_tensor))

    real_tensors, fake_tensors = tensors(real), tensors(fake)
    assert len(fake_tensors) == len(real_tensors)
    for r, f in zip(real_tensors, fake_tensors):
        assert (f.shape, f.dtype, f.device, f.stride()) == (
            r.shape,
            r.dtype,
            r.device,
            r.stride(),
        )


def compiles_without_graph_breaks(
    op: OperationEntry, case, op_input, verify_outputs: VerifyOutputs
) -> None:
    torch._dynamo.reset()
    compiled = torch.compile(op.run, fullgraph=True)
    expected = op.run(clone(op_input))
    assert equal(compiled(clone(op_input)), expected)


# Graph capture

# A capture that fails leaves torch's capture bookkeeping inconsistent for the
# rest of the process; later captures then fail regardless of the kernel.
_capture_broken_by: str | None = None


def _check_graph_capture(
    launch: Callable[[Any], Any], case, op_input, verify_outputs: VerifyOutputs
) -> None:
    """A captured launch, replayed on new input values, gives the right output for them.

    The outputs are poisoned before the replay, so a launch that writes nothing
    or reads values baked in at capture fails.
    """
    if _capture_broken_by is not None:
        raise unittest.SkipTest(f"graph capture unusable since: {_capture_broken_by}")

    static_input = clone(op_input)
    _warm_up_on_side_stream(launch, static_input)
    graph, static_output = _capture(launch, static_input)

    new_input = case.make_input(seed=1, device=DEVICE)
    assign(static_input, new_input)
    poison(static_output)
    graph.replay()
    torch.cuda.synchronize()
    verify_outputs(new_input, static_output)


def _warm_up_on_side_stream(launch: Callable[[Any], Any], op_input) -> None:
    """Launches twice outside capture; the second, warm launch must not sync with the host.

    First-call work (JIT loads, workspace allocation) has to happen outside the
    capture, on a stream other than the capturing one. A sync in the warm launch
    would fail the capture anyway; caught here, it fails with a clear message.
    """
    side = torch.cuda.Stream()
    side.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(side):
        launch(op_input)
        torch.cuda.set_sync_debug_mode("error")
        try:
            launch(op_input)
        finally:
            torch.cuda.set_sync_debug_mode("default")
    torch.cuda.current_stream().wait_stream(side)


def _capture(
    launch: Callable[[Any], Any], op_input
) -> tuple[torch.cuda.CUDAGraph, Any]:
    global _capture_broken_by

    graph = torch.cuda.CUDAGraph()
    try:
        with torch.cuda.graph(graph):
            output = launch(op_input)
    except Exception as e:
        _capture_broken_by = f"a capture failed with {type(e).__name__}: {e}"
        # Destroying a graph whose capture failed aborts the process, even at
        # interpreter exit, so it is leaked instead.
        ctypes.pythonapi.Py_IncRef(ctypes.py_object(graph))
        _end_failed_capture()
        raise
    return graph, output


def _end_failed_capture() -> None:
    # A capture that fails inside torch.cuda.graph leaves its stream capturing,
    # and every later HIP call in the process then fails. torch has no API to
    # end it, so end it through HIP and return to the default stream.
    if not torch.cuda.is_current_stream_capturing():
        return

    hip = ctypes.CDLL("libamdhip64.so")
    stream = ctypes.c_void_p(torch.cuda.current_stream().cuda_stream)
    graph = ctypes.c_void_p()
    hip.hipStreamEndCapture(stream, ctypes.byref(graph))
    if graph.value:
        hip.hipGraphDestroy(graph)
    hip.hipGetLastError()
    torch.cuda.set_stream(torch.cuda.default_stream())


# Streams

# Long enough for a launch on the wrong stream to overtake the input copy.
_SIDE_STREAM_DELAY_CYCLES = 50_000_000


def _check_on_side_stream(launch: Callable[[Any], Any], op_input) -> None:
    """Launched on a side stream right after its inputs are written there, it matches."""
    expected = launch(clone(op_input))

    # Poisoned first, so a launch that overtakes the write below reads NaN
    # rather than an earlier copy's values left in a reused allocation.
    side_input = clone(op_input)
    poison(side_input)

    default = torch.cuda.current_stream()
    side = torch.cuda.Stream()
    side.wait_stream(default)
    with torch.cuda.stream(side):
        torch.cuda._sleep(_SIDE_STREAM_DELAY_CYCLES)
        assign(side_input, op_input)
        output = launch(side_input)
    default.wait_stream(side)
    torch.cuda.synchronize()
    assert equal(output, expected), "the result differs when launched on a side stream"
