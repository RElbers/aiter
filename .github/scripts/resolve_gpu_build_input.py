# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Translate the workflow's existing gpu_archs input into build environment.

Accepting pairs in this UI input preserves the ten workflow_dispatch inputs.
The GPU_ARCHS environment variable itself retains its architecture-only syntax.
"""

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "aiter/jit/utils"))
from build_targets import format_build_targets, parse_build_targets


def resolve_input(value):
    if "\n" in value or "\r" in value:
        raise ValueError("GPU build input must be a single line")
    if ":" not in value:
        return value, ""
    targets = parse_build_targets(value)
    return ";".join(sorted({gfx for gfx, _ in targets})), format_build_targets(targets)


def main():
    archs, targets = resolve_input(os.environ["GPU_ARCHS"])
    with open(os.environ["GITHUB_ENV"], "a") as output:
        output.write(f"GPU_ARCHS={archs}\nAITER_BUILD_TARGETS={targets}\n")
    print(f"Build architectures: {archs}; explicit variants: {targets or '(legacy)'}")


if __name__ == "__main__":
    main()
