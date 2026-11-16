# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

from __future__ import annotations

from conftest import requires_built_aiter

requires_built_aiter()

import torch

from aiter.catalog.testing.tensors import clone


def test_clone_keeps_tensor_flags():
    x = torch.ones(2)
    x.is_shuffled = True  # ty: ignore[unresolved-attribute]
    (copy,) = clone((x,))
    assert copy is not x and copy.is_shuffled is True
