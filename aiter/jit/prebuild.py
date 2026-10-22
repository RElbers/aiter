# SPDX-License-Identifier: MIT
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.
"""Build a selected module without importing runtime operators or probing a GPU."""

import argparse
import inspect
import logging
import sys
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("module", help="Module name from optCompilerConfig.json")
    parser.add_argument(
        "--build-targets", help="gfx:CU pairs; defaults to AITER_BUILD_TARGETS"
    )
    args = parser.parse_args()

    # Import the build package directly, just as setup.py does. Importing aiter
    # would initialize runtime operators before resolving the explicit request.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from jit.utils.build_targets import resolve_build_targets

    targets = resolve_build_targets(args.build_targets)
    if targets is None:
        parser.error(
            "set --build-targets or AITER_BUILD_TARGETS for an explicit prebuild"
        )

    from jit import core

    logging.basicConfig(level=logging.INFO)
    build_args = core.get_args_of_build(args.module, build_targets=targets)
    build_args["md_name"] = args.module
    parameters = inspect.signature(core.build_module).parameters
    build_args = {key: value for key, value in build_args.items() if key in parameters}
    core.build_module(**build_args)


if __name__ == "__main__":
    main()
