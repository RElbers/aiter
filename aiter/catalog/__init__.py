# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The catalog of aiter operations and the kernels that implement them.

Importing ``aiter.catalog`` imports every module under ``aiter.catalog.ops``,
which registers the operations, kernels and tests they declare. ``aiter``
itself does not import it, so nothing on the hot path depends on it.
"""

import importlib
import pkgutil

from aiter.catalog import ops
from aiter.catalog.registry import register_operation


def _discover_ops():
    for _module in pkgutil.walk_packages(ops.__path__, prefix=f"{ops.__name__}."):
        importlib.import_module(_module.name)


_discover_ops()


__all__ = [
    "register_operation",
]
