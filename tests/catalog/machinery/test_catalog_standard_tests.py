# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Each standard test passes a correct kernel and fails the broken one it targets.

Only the tests that run on the CPU; graph capture, streams, fake tensors and
compilation need a GPU.
"""

from __future__ import annotations

import pytest
from conftest import requires_built_aiter

requires_built_aiter()

import torch
from catalog_fakes import ScaleCase, make_suite, register, register_scale_op

from aiter.catalog.testing import standard_tests as st


def _wrong(i, _):
    return i.x * i.factor + 1


def _writes_input(i, _):
    i.x.mul_(i.factor)
    return i.x.clone()


def _returns_input(i, _):
    return i.x.mul_(1.0)


def _nondeterministic(i, _):
    return i.x * i.factor + torch.rand(()) * 1e-3


@pytest.mark.parametrize(
    ("test", "broken"),
    [
        (st.matches_reference, _wrong),
        (st.leaves_inputs_unchanged, _writes_input),
        (st.does_not_alias_inputs, _returns_input),
        (st.is_deterministic, _nondeterministic),
    ],
)
def test_a_standard_test_fails_the_kernel_it_targets(monkeypatch, test, broken):
    scale_op = register_scale_op(monkeypatch)
    correct = register(scale_op, "correct", lambda i, _: i.x * i.factor)
    broken = register(scale_op, "broken", broken)
    suite = make_suite(scale_op, kernel_tests=[test])
    method = getattr(suite, f"test_{test.__name__}")

    method(correct, ScaleCase(4))
    with pytest.raises(AssertionError):
        method(broken, ScaleCase(4))
