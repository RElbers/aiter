# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Operation tests for ``aiter.rms_norm``."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from aiter.catalog.ops.rmsnorm.api import RmsNorm_Input, rmsnorm_op
from aiter.catalog.testing import standard_tests as st
from aiter.test_common import assertAllclose


@dataclass(frozen=True)
class RmsNorm_Case:
    dims: tuple[int, ...]
    dtype: str
    model_sensitive: int = 0
    row_strided: bool = False

    def make_input(self, *, seed: int, device: str) -> RmsNorm_Input:
        dtype = getattr(torch, self.dtype)
        g = torch.Generator(device=device).manual_seed(seed)
        hidden = self.dims[-1]
        if self.row_strided:
            padded = (*self.dims[:-1], hidden + 64)
            x = torch.randn(padded, generator=g, device=device).to(dtype)
            x = x[..., :hidden]
        else:
            x = torch.randn(self.dims, generator=g, device=device).to(dtype)
        weight = torch.randn(hidden, generator=g, device=device).to(dtype)
        return RmsNorm_Input(x, weight, 1e-5, self.model_sensitive)


cases = (
    RmsNorm_Case((1, 4096), "bfloat16"),
    RmsNorm_Case((64, 8192), "bfloat16"),
    RmsNorm_Case((32, 4096), "float16"),
    # Row lengths that are not a multiple of the kernels' 8-element vectors.
    RmsNorm_Case((64, 1002), "bfloat16"),
    RmsNorm_Case((1, 8190), "float16"),
    RmsNorm_Case((16, 4096), "bfloat16", row_strided=True),
    RmsNorm_Case((16, 4096), "float32"),
    RmsNorm_Case((8, 16384), "bfloat16"),
    RmsNorm_Case((2, 8, 4096), "bfloat16"),
    RmsNorm_Case((32, 4096), "bfloat16", model_sensitive=1),
)


def reference(op_input: RmsNorm_Input) -> torch.Tensor:
    x = op_input.input.float()
    normed = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + op_input.epsilon)
    if op_input.use_model_sensitive_rmsnorm:
        # T5: round to the input dtype before applying the weight.
        normed = normed.to(op_input.input.dtype).float()
    return (normed * op_input.weight.float()).to(op_input.input.dtype)


def compare_outputs(
    case: RmsNorm_Case,
    actual: torch.Tensor,
    expected: torch.Tensor,
) -> None:
    assert actual.shape == expected.shape and actual.dtype == expected.dtype
    assertAllclose(actual, expected, rtol=1e-2, atol=1e-2, tol_err_ratio=0)
    # The T5 variant differs from plain rmsnorm by one 16-bit rounding step,
    # which the tolerance cannot see; bit-exactness can.
    if actual.element_size() == 2:
        exact = (actual == expected).float().mean().item()
        assert exact >= 0.99, f"only {exact:.2%} of elements match the reference"


rmsnorm_tests = rmsnorm_op.operation_tests(
    cases=cases,
    reference=reference,
    compare_outputs=compare_outputs,
    kernel_tests=[
        st.matches_reference,
        st.leaves_inputs_unchanged,
        st.does_not_alias_inputs,
        st.is_deterministic,
        st.graph_capture,
        st.on_side_stream,
    ],
    op_tests=[
        st.op_matches_reference,
        st.op_graph_capture,
        st.fake_matches_real,
        st.compiles_without_graph_breaks,
    ],
)
