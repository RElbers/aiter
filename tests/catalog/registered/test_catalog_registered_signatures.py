# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Every registered operation's input type matches its public function's signature.

When they match, ``op.func(**fields)`` is a valid call for every input, so the
operation's ``run`` adapter can be the plain unpacking it is meant to be.
"""

from __future__ import annotations

import inspect
import typing

import pytest
from conftest import requires_built_aiter

requires_built_aiter()

from aiter.catalog import registry

OPS = registry.operations()


@pytest.mark.parametrize("op", OPS, ids=[op.name for op in OPS])
def test_public_signature_names_its_parameters(op):
    # A decorator that does not set __wrapped__ hides the op's parameters.
    kinds = {p.kind for p in inspect.signature(op.func).parameters.values()}
    assert kinds != {
        inspect.Parameter.VAR_POSITIONAL,
        inspect.Parameter.VAR_KEYWORD,
    }, f"{op.name}'s signature is a bare (*args, **kwargs) wrapper"


@pytest.mark.parametrize("op", OPS, ids=[op.name for op in OPS])
def test_input_type_takes_exactly_the_operation_parameters(op):
    op_params = inspect.signature(op.func).parameters
    input_params = inspect.signature(op.input_type).parameters
    assert list(input_params) == list(op_params)

    for name, expected in op_params.items():
        actual = input_params[name]
        assert actual.kind == expected.kind, name
        assert actual.default == expected.default, name


@pytest.mark.parametrize("op", OPS, ids=[op.name for op in OPS])
def test_input_type_annotations_match_the_operation(op):
    op_hints = typing.get_type_hints(inspect.unwrap(op.func))
    input_hints = typing.get_type_hints(op.input_type)
    mismatched = {
        name: (input_hints.get(name), hint)
        for name, hint in op_hints.items()
        if name != "return" and input_hints.get(name) != hint
    }
    assert not mismatched, f"(input type, operation) annotations: {mismatched}"
