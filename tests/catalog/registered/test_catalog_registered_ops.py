# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Checks over every operation aiter's catalog registers."""

from __future__ import annotations

from conftest import requires_built_aiter

requires_built_aiter()

from aiter.catalog import registry


def test_every_operation_has_a_kernel():
    ops_without_kernels = [op.name for op in registry.operations() if not op.kernels]
    assert not ops_without_kernels, f"operations without kernels: {ops_without_kernels}"


def test_every_operation_declares_tests():
    ops_without_tests = [op.name for op in registry.operations() if op.tests is None]
    assert not ops_without_tests, f"operations without tests: {ops_without_tests}"
