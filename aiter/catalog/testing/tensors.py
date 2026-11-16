# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Tensor helpers for operation inputs and outputs.

Each walks tensors nested in tuples, lists, dicts and dataclasses, which is how
operation inputs and outputs are built.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import fields, is_dataclass, replace
from typing import Any

import torch


def clone(value: Any) -> Any:
    """``value`` with every tensor cloned."""
    return map_tensors(value, _clone_tensor)


def map_tensors(value: Any, fn: Callable[[torch.Tensor], Any]) -> Any:
    """``value`` rebuilt with ``fn`` applied to every tensor in it."""
    if isinstance(value, torch.Tensor):
        return fn(value)
    if isinstance(value, (list, tuple)):
        return type(value)(map_tensors(v, fn) for v in value)
    if isinstance(value, dict):
        return {k: map_tensors(v, fn) for k, v in value.items()}
    if is_dataclass(value) and not isinstance(value, type):
        mapped = {
            f.name: map_tensors(getattr(value, f.name), fn)
            for f in fields(value)
            if f.init
        }
        return replace(value, **mapped)
    return value


def tensors(value: Any) -> list[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        return [value]
    return [t for child in _children(value) for t in tensors(child)]


def equal(a: Any, b: Any) -> bool:
    """Bit-exact for tensors, ``==`` otherwise."""
    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        return (
            isinstance(a, torch.Tensor)
            and isinstance(b, torch.Tensor)
            and a.dtype == b.dtype
            and torch.equal(a, b)
        )

    a_children, b_children = _children(a), _children(b)
    if not a_children and not b_children:
        return a == b
    if type(a) is not type(b) or len(a_children) != len(b_children):
        return False

    return all(equal(x, y) for x, y in zip(a_children, b_children))


def storage(t: torch.Tensor) -> int:
    """Identifies the memory ``t`` lives in; views of one tensor share it."""
    return t.untyped_storage().data_ptr()


def assign(dst: Any, src: Any) -> None:
    """Writes ``src``'s tensor values into ``dst``'s tensors, which keep their storage."""
    for d, s in zip(tensors(dst), tensors(src), strict=True):
        d.copy_(s)


def poison(value: Any) -> None:
    """Fills every floating-point tensor in ``value`` with NaN."""
    for t in tensors(value):
        if t.is_floating_point():
            t.fill_(float("nan"))


def _clone_tensor(t: torch.Tensor) -> torch.Tensor:
    copy = t.clone()
    # Operations read flags set on tensors, such as ``is_shuffled``.
    copy.__dict__.update(t.__dict__)
    return copy


def _children(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple)):
        return list(value)
    if isinstance(value, dict):
        return list(value.values())
    if is_dataclass(value) and not isinstance(value, type):
        return [getattr(value, f.name) for f in fields(value)]
    return []
