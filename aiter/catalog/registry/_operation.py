# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Generic, ParamSpec, TypeVar

from aiter.catalog.registry._common import (
    DeviceInfo,
    Func,
    _function_name,
    _key,
)
from aiter.catalog.registry._kernel import KernelEntry
from aiter.catalog.registry._operation_tests import Case, OperationTestsEntry

if TYPE_CHECKING:
    from aiter.catalog.testing.signatures import KernelTest, OpTest

P = ParamSpec("P")
R = TypeVar("R")
Input = TypeVar("Input")
Shape = TypeVar("Shape")
Instance = TypeVar("Instance")


@dataclass(frozen=True)
class OperationEntry(Generic[P, R, Input, Shape]):
    func: Callable[P, R]
    run: Callable[[Input], R]
    name: str

    # Called with the operation's own arguments, so it shares the signature P.
    input_type: Callable[P, Input]
    shape_type: type[Shape]
    input_to_shape: Callable[[Input], Shape]

    kernels: dict[str, KernelEntry[Any, Input, Shape, R]] = field(default_factory=dict)
    tests: OperationTestsEntry[Input, R, Any] | None = None

    def register_kernel(
        self,
        func: Callable[..., Any],
        run: Callable[[Input, Instance], R],
        /,
        *,
        name: str | None = None,
        backend: str,
        instance_type: type[Instance],
        available_instances: Callable[[Shape, DeviceInfo], Iterable[Instance]],
    ) -> KernelEntry[Instance, Input, Shape, R]:
        entry = KernelEntry(
            func,
            run,
            name or _function_name(func),
            self,
            backend,
            instance_type,
            available_instances,
        )
        if entry.name in self.kernels:
            raise ValueError(
                f"kernel {entry.name} is already registered for {self.name}"
            )

        self.kernels[entry.name] = entry
        return entry

    def operation_tests(
        self,
        *,
        cases: Iterable[Case],
        reference: Callable[[Input], R],
        compare_outputs: Callable[[Case, R, R], None],
        kernel_tests: Sequence[KernelTest[Any, Case, Input, R]] = (),
        op_tests: Sequence[OpTest[Case, Input, R]] = (),
        max_instances: int | None = None,
    ) -> OperationTestsEntry[Input, R, Case]:
        if self.tests is not None:
            raise ValueError(f"tests are already declared for {self.name}")

        entry = OperationTestsEntry(
            self,
            tuple(cases),
            reference,
            compare_outputs,
            tuple(kernel_tests),
            tuple(op_tests),
            max_instances,
        )
        # The entry is frozen; tests are declared after the operation is registered.
        object.__setattr__(self, "tests", entry)
        return entry

    def get_kernel(self, kernel: str | Func, /) -> KernelEntry[Any, Input, Shape, R]:
        key = _key(kernel)
        try:
            return self.kernels[key]
        except KeyError:
            raise ValueError(f"{key} is not a registered kernel for {self.name}")


AnyOperationEntry = OperationEntry[..., Any, Any, Any]
