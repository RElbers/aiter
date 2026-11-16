# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The signatures operation tests are written against.

A kernel test checks one kernel launched with one instance; an op test checks
what the public operation adds on top of its kernels. Both get the case, the
input made from it, and ``verify_outputs(op_input, actual)``, which raises
unless ``actual`` is the right output for ``op_input``. A test raises
``unittest.SkipTest`` to skip and ``Unsupported`` when the backend rejects the
instance at launch.

Parameters are positional-only: the names document what each one is, and a
test may name them as it likes. A test is a named function: its name is the
pytest test's name.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeVar

from aiter.catalog.registry import KernelEntry, OperationEntry

Instance = TypeVar("Instance")
Case = TypeVar("Case")
Input = TypeVar("Input")
R = TypeVar("R")
Input_contra = TypeVar("Input_contra", contravariant=True)
R_contra = TypeVar("R_contra", contravariant=True)


class VerifyOutputs(Protocol[Input_contra, R_contra]):
    def __call__(self, op_input: Input_contra, actual: R_contra, /) -> None: ...


class KernelTest(Protocol[Instance, Case, Input, R]):
    __name__: str

    def __call__(
        self,
        kernel: KernelEntry[Instance, Input, Any, R],
        instance: Instance,
        case: Case,
        op_input: Input,
        verify_outputs: VerifyOutputs[Input, R],
        /,
    ) -> None: ...


class OpTest(Protocol[Case, Input, R]):
    __name__: str

    def __call__(
        self,
        op: OperationEntry[..., R, Input, Any],
        case: Case,
        op_input: Input,
        verify_outputs: VerifyOutputs[Input, R],
        /,
    ) -> None: ...
