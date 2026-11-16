# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Every registered operation's test declaration is well formed."""

from __future__ import annotations

import inspect

import pytest
from conftest import requires_built_aiter

requires_built_aiter()

from aiter.catalog import registry

DECLARED = [op.tests for op in registry.operations() if op.tests is not None]
DECLARED_IDS = [tests.op.name for tests in DECLARED]


def _arity(fn) -> int:
    return len(inspect.signature(fn).parameters)


@pytest.mark.parametrize("tests", DECLARED, ids=DECLARED_IDS)
def test_has_cases(tests):
    assert tests.cases


@pytest.mark.parametrize("tests", DECLARED, ids=DECLARED_IDS)
def test_case_ids_are_unique(tests):
    # pytest builds test IDs from str(case); duplicates make failures ambiguous.
    ids = [str(case) for case in tests.cases]
    duplicates = sorted({i for i in ids if ids.count(i) > 1})
    assert not duplicates


@pytest.mark.parametrize("tests", DECLARED, ids=DECLARED_IDS)
def test_reference_and_compare_outputs_take_the_right_arguments(tests):
    assert _arity(tests.reference) == 1, "reference(op_input)"
    assert _arity(tests.compare_outputs) == 3, "compare_outputs(case, actual, expected)"


@pytest.mark.parametrize("tests", DECLARED, ids=DECLARED_IDS)
def test_opted_in_tests_have_their_list_signature(tests):
    wrong = [
        *(f"kernel test {t.__name__}" for t in tests.kernel_tests if _arity(t) != 5),
        *(f"op test {t.__name__}" for t in tests.op_tests if _arity(t) != 4),
    ]
    assert not wrong, (
        "kernel tests take (kernel, instance, case, op_input, verify_outputs), "
        f"op tests (op, case, op_input, verify_outputs): {wrong}"
    )
