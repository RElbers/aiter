# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""``aiter.rms_norm`` and its kernels: the HIP fast path and the OPUS generic one."""

from __future__ import annotations

from collections.abc import Iterator

import torch

from aiter.catalog.ops.rmsnorm.api import RmsNorm_Input, RmsNorm_Shape, rmsnorm_op
from aiter.catalog.registry import DeviceInfo, NoTunables
from aiter.ops.rmsnorm import rms_norm_opus as opus_binding
from aiter.ops.rmsnorm import rmsnorm as hip_binding


def run_hip(x: RmsNorm_Input, _: NoTunables) -> torch.Tensor:
    out = torch.empty_like(x.input)
    hip_binding(out, x.input, x.weight, x.epsilon)
    return out


def rms_norm_hip_instances(
    shape: RmsNorm_Shape, target: DeviceInfo
) -> Iterator[NoTunables]:
    if (
        not shape.model_sensitive
        and shape.ndim == 2
        and shape.dtype in ("torch.float16", "torch.bfloat16")
        and shape.hidden <= 8192
        and (shape.hidden % 8 == 0 or target.arch not in ("gfx1250", "gfx1250-strict"))
    ):
        yield NoTunables()


rmsnorm_op.register_kernel(
    hip_binding,
    run_hip,
    backend="hip",
    instance_type=NoTunables,
    available_instances=rms_norm_hip_instances,
)


def run_opus(x: RmsNorm_Input, _: NoTunables) -> torch.Tensor:
    out = torch.empty_like(x.input)
    opus_binding(out, x.input, x.weight, x.epsilon, x.use_model_sensitive_rmsnorm)
    return out


def rms_norm_opus_instances(
    shape: RmsNorm_Shape, target: DeviceInfo
) -> Iterator[NoTunables]:
    yield NoTunables()


rmsnorm_op.register_kernel(
    opus_binding,
    run_opus,
    backend="opus",
    instance_type=NoTunables,
    available_instances=rms_norm_opus_instances,
)
