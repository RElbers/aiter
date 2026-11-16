# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""How ``to_pytest_class`` runs an operation's tests: instances, skips, and failures."""

from __future__ import annotations

import pytest
from conftest import requires_built_aiter

requires_built_aiter()

import torch
from catalog_fakes import (
    ScaleCase,
    ScaleInput,
    make_suite,
    register,
    register_scale_op,
)

from aiter.catalog.registry import NoTunables, Unsupported
from aiter.catalog.testing import standard_tests as st


def test_opting_into_a_test_twice_raises(monkeypatch):
    scale_op = register_scale_op(monkeypatch)
    with pytest.raises(ValueError, match="twice"):
        make_suite(scale_op, kernel_tests=[st.matches_reference, st.matches_reference])


def test_a_kernel_test_runs_once_per_available_instance(monkeypatch):
    scale_op = register_scale_op(monkeypatch)

    def available(shape, device):
        yield from (1, 2, 3)

    def run(i: ScaleInput, tile: int) -> torch.Tensor:
        if tile == 3:
            raise Unsupported("tile 3 rejected by the library")
        return i.x * i.factor

    kernel = register(
        scale_op, "tiled", run, available_instances=available, instance_type=int
    )
    seen = []

    def records_instances(kernel, instance, case, op_input, verify_outputs):
        kernel.run(op_input, instance)
        seen.append(instance)

    suite = make_suite(scale_op, kernel_tests=[records_instances])
    with pytest.warns(UserWarning, match="1 instances Unsupported"):
        suite.test_records_instances(kernel, ScaleCase(4))
    assert seen == [1, 2]


def test_a_kernel_with_no_instance_for_a_shape_is_skipped(monkeypatch):
    scale_op = register_scale_op(monkeypatch)

    def available(shape, device):
        if shape.factor == 2.0:
            yield NoTunables()

    kernel = register(
        scale_op,
        "factor_two_only",
        lambda i, _: i.x * i.factor,
        available_instances=available,
    )
    suite = make_suite(scale_op, kernel_tests=[st.matches_reference])
    with pytest.raises(pytest.skip.Exception, match="no instance available"):
        suite.test_matches_reference(kernel, ScaleCase(4, factor=5.0))


def test_a_kernel_unsupported_for_every_instance_is_skipped(monkeypatch):
    scale_op = register_scale_op(monkeypatch)

    def run(i, _):
        raise Unsupported("not built for this target")

    kernel = register(scale_op, "unbuilt", run)
    suite = make_suite(scale_op, kernel_tests=[st.matches_reference])
    with pytest.raises(pytest.skip.Exception, match="every instance Unsupported"):
        suite.test_matches_reference(kernel, ScaleCase(4))


def test_max_instances_runs_an_evenly_spaced_subset(monkeypatch):
    scale_op = register_scale_op(monkeypatch)
    kernel = register(
        scale_op,
        "ten_tiles",
        lambda i, tile: i.x * i.factor,
        available_instances=lambda shape, device: range(10),
        instance_type=int,
    )
    seen = []

    def records_instances(kernel, instance, case, op_input, verify_outputs):
        seen.append(instance)

    suite = make_suite(scale_op, kernel_tests=[records_instances], max_instances=4)
    with pytest.warns(UserWarning, match="runs 4 of 10"):
        suite.test_records_instances(kernel, ScaleCase(4))
    assert seen == [0, 2, 5, 7]


def test_a_failure_names_the_kernel_and_instance(monkeypatch):
    scale_op = register_scale_op(monkeypatch)
    kernel = register(scale_op, "wrong", lambda i, _: i.x * 3.0)
    suite = make_suite(scale_op, kernel_tests=[st.matches_reference])
    with pytest.raises(AssertionError, match=r"^wrong\[NoTunables\(\)\]: "):
        suite.test_matches_reference(kernel, ScaleCase(4))


def test_an_op_test_gets_the_op_and_judges_against_the_reference(monkeypatch):
    scale_op = register_scale_op(monkeypatch)
    register(scale_op, "correct", lambda i, _: i.x * i.factor)
    suite = make_suite(scale_op, op_tests=[st.op_matches_reference])
    suite.test_op_matches_reference(ScaleCase(4))


def test_every_class_checks_each_case_makes_the_operation_input(monkeypatch):
    scale_op = register_scale_op(monkeypatch)

    class MakesATensor:
        def make_input(self, *, seed: int, device: str) -> torch.Tensor:
            return torch.ones(2)

    suite = make_suite(scale_op)
    suite.test_case_makes_the_operation_input(ScaleCase(4))
    with pytest.raises(AssertionError, match="made a Tensor, not a ScaleInput"):
        suite.test_case_makes_the_operation_input(MakesATensor())
