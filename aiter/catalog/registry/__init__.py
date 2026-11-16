# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The catalog of aiter operations and the kernels that implement them.

Importing ``aiter.catalog`` registers every declaration module listed here.
``aiter`` itself does not import it, so nothing on the hot path depends on it.
"""

from ._common import DeviceInfo, NoTunables, Unsupported
from ._kernel import AnyKernelEntry, KernelEntry
from ._operation import AnyOperationEntry, OperationEntry
from ._operation_tests import OperationCase, OperationTestsEntry
from ._registry import (
    get_operation,
    operations,
    register_operation,
)

__all__ = [
    "AnyKernelEntry",
    "AnyOperationEntry",
    "DeviceInfo",
    "KernelEntry",
    "NoTunables",
    "OperationCase",
    "OperationEntry",
    "OperationTestsEntry",
    "Unsupported",
    "get_operation",
    "operations",
    "register_operation",
]
