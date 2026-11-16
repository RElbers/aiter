# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, Protocol, TypeVar

if TYPE_CHECKING:
    from aiter.catalog.registry._common import DeviceInfo
    from aiter.catalog.registry._operation import OperationEntry
    from aiter.catalog.testing.signatures import KernelTest, OpTest

R = TypeVar("R")
Input = TypeVar("Input")
Input_co = TypeVar("Input_co", covariant=True)


class OperationCase(Protocol[Input_co]):
    def make_input(self, *, seed: int, device: str) -> Input_co: ...


# A bound cannot mention Input, so make_input's result is not tied to the
# operation's input statically; the test runner checks it instead.
Case = TypeVar("Case", bound=OperationCase[Any])


@dataclass(frozen=True)
class OperationTestsEntry(Generic[Input, R, Case]):
    """The tests an operation opts into and what they judge outputs against."""

    op: OperationEntry[..., R, Input, Any]
    cases: tuple[Case, ...]
    reference: Callable[[Input], R]
    # Raises unless ``actual`` is acceptable given the reference's ``expected``.
    compare_outputs: Callable[[Case, R, R], None]
    kernel_tests: tuple[KernelTest[Any, Case, Input, R], ...]
    op_tests: tuple[OpTest[Case, Input, R], ...]
    max_instances: int | None

    def to_pytest_class(self, device: DeviceInfo) -> type:
        """See ``aiter.catalog.testing.pytest_class.to_pytest_class``."""
        # Imported here so the registry does not import pytest.
        from aiter.catalog.testing.pytest_class import to_pytest_class

        return to_pytest_class(self, device)
