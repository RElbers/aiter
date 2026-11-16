# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Turns an operation's opted-in tests into a pytest class.

Each kernel test becomes one test method, parametrized over the operation's
kernels and its cases; each op test one method parametrized over the cases. For
a kernel, the method launches the test once per instance the kernel has available
for the case's shape on the given device, or an evenly spaced
``max_instances`` of them. A kernel with none available is skipped for that case;
an instance the backend raises Unsupported for is left out with a warning, and
the method is skipped if none ran.

Every class also checks that each case makes an input of the operation's input
type, which its declaration cannot check statically.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable
from typing import Any

import pytest

from aiter.catalog.registry import (
    DeviceInfo,
    KernelEntry,
    OperationTestsEntry,
    Unsupported,
)
from aiter.catalog.testing.signatures import KernelTest, OpTest, VerifyOutputs

DEVICE = "cuda"


def to_pytest_class(tests: OperationTestsEntry, device: DeviceInfo) -> type:
    """A pytest class running ``tests`` on ``device``; bind it to a ``Test*`` name."""
    op = tests.op

    def verify_outputs_for(case) -> VerifyOutputs:
        def verify_outputs(op_input, actual) -> None:
            tests.compare_outputs(case, actual, tests.reference(op_input))

        return verify_outputs

    def kernel_method(test: KernelTest):
        def method(self, kernel: KernelEntry, case) -> None:
            op_input = case.make_input(seed=0, device=DEVICE)
            instances = _available_instances(
                kernel, op_input, device, tests.max_instances
            )
            verify_outputs = verify_outputs_for(case)

            ran = 0
            unsupported = []
            for instance in instances:
                try:
                    test(kernel, instance, case, op_input, verify_outputs)
                except Unsupported as e:
                    unsupported.append(f"{instance}: {e}")
                    continue
                except AssertionError as e:
                    raise AssertionError(f"{kernel.name}[{instance}]: {e}") from e
                ran += 1

            if not ran:
                pytest.skip(
                    f"{kernel.name}: every instance Unsupported at launch: "
                    + "; ".join(unsupported)
                )
            if unsupported:
                warnings.warn(
                    f"{kernel.name}: {len(unsupported)} instances Unsupported at "
                    "launch, not tested: " + "; ".join(unsupported),
                    stacklevel=2,
                )

        return method

    def op_method(test: OpTest):
        def method(self, case) -> None:
            op_input = case.make_input(seed=0, device=DEVICE)
            test(op, case, op_input, verify_outputs_for(case))

        return method

    def pytest_generate_tests(self, metafunc: pytest.Metafunc) -> None:
        if "kernel" in metafunc.fixturenames:
            kernels = list(op.kernels.values())
            metafunc.parametrize("kernel", kernels, ids=[k.name for k in kernels])
        if "case" in metafunc.fixturenames:
            metafunc.parametrize("case", tests.cases, ids=str)

    def test_case_makes_the_operation_input(self, case) -> None:
        op_input = case.make_input(seed=0, device=DEVICE)
        expected = op.input_type
        if isinstance(expected, type):
            assert isinstance(
                op_input, expected
            ), f"{case} made a {type(op_input).__name__}, not a {expected.__name__}"

    attrs: dict[str, Any] = {
        "pytest_generate_tests": pytest_generate_tests,
        "test_case_makes_the_operation_input": test_case_makes_the_operation_input,
    }

    def add(test: KernelTest | OpTest, method: Callable[..., None]) -> None:
        name = f"test_{test.__name__}"
        if name in attrs:
            raise ValueError(f"{op.name} opts into {name} twice")
        attrs[name] = method

    for kernel_test in tests.kernel_tests:
        add(kernel_test, kernel_method(kernel_test))
    for op_test in tests.op_tests:
        add(op_test, op_method(op_test))

    return type(f"Test_{op.name}", (), attrs)


def _available_instances(
    kernel: KernelEntry, op_input, device: DeviceInfo, max_instances: int | None
) -> list[Any]:
    shape = kernel.op.input_to_shape(op_input)
    instances = list(kernel.available_instances(shape, device))
    if not instances:
        pytest.skip(f"{kernel.name} has no instance available for {shape}")

    if max_instances is None or len(instances) <= max_instances:
        return instances

    warnings.warn(
        f"{kernel.name} runs {max_instances} of {len(instances)} instances for {shape}",
        stacklevel=2,
    )
    step = len(instances) / max_instances
    return [instances[int(i * step)] for i in range(max_instances)]
