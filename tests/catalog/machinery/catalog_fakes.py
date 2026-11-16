# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""A fake operation on the CPU for testing the catalog's machinery.

Import after ``requires_built_aiter()``: it imports ``aiter.catalog``.
"""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import torch

from aiter.catalog import registry
from aiter.catalog.registry import AnyOperationEntry, DeviceInfo, NoTunables, _registry
from aiter.catalog.testing.pytest_class import to_pytest_class


@dataclass(frozen=True)
class ScaleInput:
    x: torch.Tensor
    factor: float = 2.0


@dataclass(frozen=True)
class ScaleShape:
    n: int
    factor: float


@dataclass(frozen=True)
class ScaleCase:
    n: int
    factor: float = 2.0

    def make_input(self, *, seed: int, device: str) -> ScaleInput:
        return ScaleInput(torch.arange(self.n, dtype=torch.float32), self.factor)


def scale(x: torch.Tensor, factor: float = 2.0) -> torch.Tensor:
    return x * factor


def scale_reference(i: ScaleInput) -> torch.Tensor:
    return i.x * i.factor


def compare_outputs(
    case: ScaleCase, actual: torch.Tensor, expected: torch.Tensor
) -> None:
    torch.testing.assert_close(actual, expected)


def register_scale_op(monkeypatch) -> AnyOperationEntry:
    """Registers the fake operation in a registry emptied for the calling test."""
    monkeypatch.setattr(_registry, "_operations", {})
    return registry.register_operation(
        scale,
        lambda i: scale(i.x, i.factor),
        input_type=ScaleInput,
        shape_type=ScaleShape,
        input_to_shape=lambda i: ScaleShape(i.x.numel(), i.factor),
    )


def one_instance(shape: ScaleShape, device: DeviceInfo) -> Iterator[NoTunables]:
    yield NoTunables()


def register(op, name, run, available_instances=one_instance, instance_type=NoTunables):
    return op.register_kernel(
        run,
        run,
        name=name,
        backend="test",
        instance_type=instance_type,
        available_instances=available_instances,
    )


DEFAULT_CASES = (ScaleCase(4),)


def make_suite(
    op, *, kernel_tests=(), op_tests=(), cases=DEFAULT_CASES, max_instances=None
):
    """An instance of ``op``'s pytest class on a fake gfx950; its test methods can be called directly."""
    tests = op.operation_tests(
        cases=cases,
        reference=scale_reference,
        compare_outputs=compare_outputs,
        kernel_tests=kernel_tests,
        op_tests=op_tests,
        max_instances=max_instances,
    )
    return to_pytest_class(tests, DeviceInfo("gfx950", 256))()
