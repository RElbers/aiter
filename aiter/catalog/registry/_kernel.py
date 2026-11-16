# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.


from __future__ import annotations

from collections.abc import Callable, Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Generic, TypeVar

from aiter.catalog.registry._common import DeviceInfo

if TYPE_CHECKING:
    from aiter.catalog.registry._operation import OperationEntry

R = TypeVar("R")
Input = TypeVar("Input")
Shape = TypeVar("Shape")
Instance = TypeVar("Instance")


@dataclass(frozen=True)
class KernelEntry(Generic[Instance, Input, Shape, R]):
    func: Callable[..., Any]
    # Taking the instance as a parameter pins Instance to instance_type; with
    # Instance only in yielded positions, mismatched available_instances widen it
    # to a union.
    run: Callable[[Input, Instance], R]

    name: str
    op: OperationEntry[..., R, Input, Shape]

    backend: str

    instance_type: type[Instance]
    available_instances: Callable[[Shape, DeviceInfo], Iterable[Instance]]


AnyKernelEntry = KernelEntry[Any, Any, Any, Any]
