# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""``aiter.rms_norm`` and its kernels: the HIP fast path and the OPUS generic one."""

from __future__ import annotations

from dataclasses import dataclass

import torch

from aiter.catalog.registry import register_operation
from aiter.ops.rmsnorm import rms_norm


@dataclass(frozen=True)
class RmsNorm_Input:
    input: torch.Tensor
    weight: torch.Tensor
    epsilon: float
    use_model_sensitive_rmsnorm: int = 0


@dataclass(frozen=True)
class RmsNorm_Shape:
    rows: int
    hidden: int
    ndim: int
    dtype: str
    model_sensitive: bool


def input_to_shape(inputs: RmsNorm_Input) -> RmsNorm_Shape:
    dtypes = ("torch.float16", "torch.bfloat16", "torch.float32")
    hidden = inputs.input.shape[-1]
    if (
        str(inputs.input.dtype) not in dtypes
        or inputs.weight.dtype != inputs.input.dtype
    ):
        raise ValueError("input and weight must share an fp16, bf16 or fp32 dtype")
    if tuple(inputs.weight.shape) != (hidden,):
        raise ValueError(
            f"weight must be ({hidden},), got {tuple(inputs.weight.shape)}"
        )

    return RmsNorm_Shape(
        inputs.input.numel() // hidden,
        hidden,
        inputs.input.dim(),
        str(inputs.input.dtype),
        bool(inputs.use_model_sensitive_rmsnorm),
    )


def run_rms_norm(x: RmsNorm_Input) -> torch.Tensor:
    return rms_norm(x.input, x.weight, x.epsilon, x.use_model_sensitive_rmsnorm)


rmsnorm_op = register_operation(
    rms_norm,
    run_rms_norm,
    name="rms_norm",
    input_type=RmsNorm_Input,
    shape_type=RmsNorm_Shape,
    input_to_shape=input_to_shape,
)
