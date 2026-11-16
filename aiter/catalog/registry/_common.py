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

import inspect
from collections.abc import Callable
from dataclasses import dataclass
from types import FunctionType
from typing import Any

Func = Callable[..., Any]


@dataclass(frozen=True)
class DeviceInfo:
    arch: str
    cu_num: int

    @classmethod
    def current(cls) -> DeviceInfo:
        """The device this process runs on, as aiter's runtime detects it."""
        from aiter.jit.utils.chip_info import get_cu_num, get_gfx_runtime

        return cls(get_gfx_runtime(), get_cu_num())


@dataclass(frozen=True)
class NoTunables:
    """The instance type of a kernel with nothing to tune."""


class Unsupported(Exception):
    """A kernel's backend rejected a yielded instance at launch.

    For backends that only decide support inside the compiled library.
    Operation tests skip that instance: never a pass.
    """


def _function_name(func: Func, /) -> str:
    # Decorators such as torch_compile_guard return a generic wrapper whose
    # qualname every guarded op shares; the original sits behind __wrapped__.
    func = inspect.unwrap(func)
    if isinstance(func, FunctionType):
        return func.__qualname__
    raise TypeError(f"{func!r} has no name")


def _key(obj: str | Func, /) -> str:
    if isinstance(obj, str):
        return obj

    return _function_name(obj)
