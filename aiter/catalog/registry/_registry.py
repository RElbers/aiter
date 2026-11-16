# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Declarations of aiter operations and the kernels that implement them.

An operation is a public aiter function; its signature is the contract. A
kernel runs one backend on the operation's input and one of its own instances,
those available for the input's shape on a device. Registration records metadata:
it does not wrap functions, change dispatch or launch work, and nothing on the
hot path reads it.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import ParamSpec, TypeVar

from aiter.catalog.registry._common import Func, _function_name, _key
from aiter.catalog.registry._operation import AnyOperationEntry, OperationEntry

P = ParamSpec("P")
R = TypeVar("R")
Input = TypeVar("Input")
Shape = TypeVar("Shape")

_operations: dict[str, AnyOperationEntry] = {}


def register_operation(
    func: Callable[P, R],
    run: Callable[[Input], R],
    /,
    *,
    name: str | None = None,
    input_type: Callable[P, Input],
    shape_type: type[Shape],
    input_to_shape: Callable[[Input], Shape],
) -> OperationEntry[P, R, Input, Shape]:
    entry = OperationEntry(
        func,
        run,
        name or _function_name(func),
        input_type=input_type,
        shape_type=shape_type,
        input_to_shape=input_to_shape,
    )

    if entry.name in _operations:
        raise ValueError(f"operation {entry.name} is already registered")

    _operations[entry.name] = entry
    return entry


def operations() -> list[AnyOperationEntry]:
    return list(_operations.values())


def get_operation(op: str | Func, /) -> AnyOperationEntry:
    key = _key(op)
    try:
        return _operations[key]
    except KeyError:
        raise ValueError(f"{op} is not a registered operation")
