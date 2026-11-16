# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""The registry's own rules, on a fake operation."""

from __future__ import annotations

import pytest
from conftest import requires_built_aiter

requires_built_aiter()

from catalog_fakes import make_suite, register_scale_op


def test_declaring_tests_twice_raises(monkeypatch):
    scale_op = register_scale_op(monkeypatch)
    make_suite(scale_op)
    with pytest.raises(ValueError, match="already declared"):
        make_suite(scale_op)
