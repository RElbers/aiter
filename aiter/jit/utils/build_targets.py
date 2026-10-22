# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Pure-Python arch constants and env-driven build target resolution.
# No torch dependency — safe to import in build scripts, gen_instances, and tests
# that run without a GPU or a full PyTorch install.
import os
import re

GFX_MAP = {
    0: "native",
    1: "gfx90a",
    2: "gfx908",
    3: "gfx940",
    4: "gfx941",
    5: "gfx942",
    6: "gfx945",
    7: "gfx1100",
    8: "gfx950",
    9: "gfx1101",
    10: "gfx1102",
    11: "gfx1103",
    12: "gfx1150",
    13: "gfx1151",
    14: "gfx1152",
    15: "gfx1153",
    16: "gfx1200",
    17: "gfx1201",
    18: "gfx1250",
}

# Maps gfx arch to the default (SPX / full-GPU) CU count used when no live GPU is
# present at build time (e.g. CI nodes with GPU_ARCHS set but no device visible).
# For live GPU builds, get_cu_num() is used instead and correctly reflects the
# actual visible CU count, including non-SPX partition modes (DPX / QPX / CPX)
# and binned variants (e.g. MI308X is gfx942 but has fewer CUs than MI300X).
# If building without a GPU for a binned or partitioned target, set CU_NUM
# explicitly alongside GPU_ARCHS to override the default here.
# Extend this table when adding support for new GPU targets.
GFX_CU_NUM_MAP = {
    "gfx942": 304,  # MI300X (SPX, full GPU); MI308X shares gfx942 — use CU_NUM override
    "gfx950": 256,  # MI350
    "gfx1100": 96,  # Radeon RX 7900 XTX
    "gfx1101": 60,  # Radeon RX 7800 XT
    "gfx1102": 32,  # Radeon RX 7600
    "gfx1103": 12,  # Radeon 780M
    "gfx1150": 16,  # Radeon 890M; set CU_NUM=12 for Radeon 880M
    "gfx1151": 40,  # Strix Halo / Radeon 8060S
    "gfx1200": 32,  # Radeon RX 9060 XT; set CU_NUM=28 for Radeon RX 9060
    "gfx1201": 64,  # Radeon RX 9070 XT / Radeon AI PRO R9700
    "gfx1250": 256,  # Gfx1250
}

BuildTargets = tuple[tuple[str, int], ...]


def parse_build_targets(value: str | BuildTargets) -> BuildTargets:
    """Validate and canonicalize explicit destination pairs without probing a GPU."""
    if isinstance(value, str):
        pairs = []
        for entry in value.split(";"):
            match = re.fullmatch(r"\s*(gfx[0-9a-z]+)\s*:\s*([0-9]+)\s*", entry.lower())
            if match is None:
                raise ValueError(
                    f"Invalid build target {entry!r}; expected gfx:positive_cu_count "
                    "pairs separated by ';', e.g. gfx950:128;gfx950:256"
                )
            pairs.append((match[1], int(match[2])))
    else:
        pairs = list(value)

    if not pairs:
        raise ValueError("An explicit build request must contain at least one target")

    targets = set()
    for gfx, cu_num in pairs:
        if gfx not in GFX_CU_NUM_MAP:
            raise ValueError(
                f"Unsupported build architecture {gfx!r}; supported: {list(GFX_CU_NUM_MAP)}"
            )
        if type(cu_num) is not int or cu_num <= 0:
            raise ValueError(
                f"Invalid CU count {cu_num!r} for {gfx}; expected a positive integer"
            )
        targets.add((gfx, cu_num))

    return tuple(sorted(targets))


def resolve_build_targets(
    value: str | BuildTargets | None = None,
) -> BuildTargets | None:
    """Read build-only input at an explicit build entry point, never at runtime."""
    if value is None:
        value = os.getenv("AITER_BUILD_TARGETS", "")
    if isinstance(value, str) and not value.strip():
        return None

    targets = parse_build_targets(value)
    legacy_archs = os.getenv("GPU_ARCHS", "").strip()
    if legacy_archs and set(_parse_gpu_archs_env(legacy_archs)) != {
        gfx for gfx, _ in targets
    }:
        raise ValueError(
            f"GPU_ARCHS={legacy_archs!r} conflicts with explicit build targets {targets}"
        )
    legacy_cu = os.getenv("CU_NUM", "").strip()
    if legacy_cu and {int(legacy_cu)} != {cu for _, cu in targets}:
        raise ValueError(
            f"CU_NUM={legacy_cu!r} conflicts with explicit build targets {targets}"
        )

    return targets


def format_build_targets(targets: BuildTargets) -> str:
    return ";".join(f"{gfx}:{cu_num}" for gfx, cu_num in targets)


def torch_processor_count_to_cu(gfx: str, processor_count: int) -> int:
    """Return physical CUs from HIP's mode-dependent processor count."""
    gfx = gfx.split(":", 1)[0].lower()
    rdna = gfx.startswith("gfx11") or gfx in ("gfx1200", "gfx1201")
    if rdna:
        wgp_mode = os.getenv("GPU_ENABLE_WGP_MODE", "1")
        numeric_mode = re.match(r"\s*([+-]?[0-9]+)", wgp_mode)
        wgp_enabled = wgp_mode == "true" or (
            numeric_mode is not None and int(numeric_mode.group(1)) != 0
        )
        if wgp_enabled:
            return processor_count * 2
    return processor_count


def ck_tile_target_families(targets: BuildTargets) -> str:
    """CK Tile generator --targets value; gfx9 is its family name for gfx942."""
    return ",".join(sorted({"gfx9" if gfx == "gfx942" else gfx for gfx, _ in targets}))


def _parse_gpu_archs_env(gfx_env: str) -> list[str]:
    """Split a GPU_ARCHS string into a list of non-empty architecture names.

    Raises RuntimeError if no valid architecture names remain after splitting
    on ';' and stripping whitespace — e.g. GPU_ARCHS=" ; " would otherwise
    silently produce an empty target list and fall back to heuristic kernels.
    """
    archs = [g.strip() for g in gfx_env.split(";") if g.strip()]
    if not archs:
        raise RuntimeError(
            f"GPU_ARCHS={gfx_env!r} contains no valid architecture names after splitting on ';'. "
            f"Known targets: {list(GFX_CU_NUM_MAP.keys())}"
        )
    return archs


def get_build_targets_env() -> list[tuple[str, int]]:
    """Resolve build targets from GPU_ARCHS env var only.  No live GPU detection.

    Raises RuntimeError if GPU_ARCHS is not set or contains an unknown arch.
    Intended for CI nodes, build scripts, and tests that run without a GPU.
    Use chip_info.get_build_targets() when live GPU fallback is also desired.
    """
    gfx_env = os.getenv("GPU_ARCHS")
    if not gfx_env:
        raise RuntimeError(
            "GPU_ARCHS is not set. "
            "Set GPU_ARCHS=gfx942 (or similar) to resolve build targets without a GPU."
        )
    targets = []
    for gfx in _parse_gpu_archs_env(gfx_env):
        if gfx not in GFX_CU_NUM_MAP:
            raise RuntimeError(
                f"Unknown gfx '{gfx}' in GPU_ARCHS — add it to "
                f"GFX_CU_NUM_MAP in build_targets.py. Known targets: "
                f"{list(GFX_CU_NUM_MAP.keys())}"
            )
        cu_num = int(os.getenv("CU_NUM", GFX_CU_NUM_MAP[gfx]))
        targets.append((gfx, cu_num))
    return targets


def filter_tune_df(tune_df, targets: list):
    """Return the subset of tune_df whose (gfx, cu_num) matches any entry in targets.

    Args:
        tune_df:  pandas DataFrame loaded from a tuning CSV (must have 'gfx' and
                  'cu_num' columns).
        targets:  list of (gfx, cu_num) tuples, as returned by get_build_targets()
                  or get_build_targets_env().

    Returns:
        Filtered DataFrame (original index preserved, no reset).
    """
    import pandas as pd

    mask = pd.Series([False] * len(tune_df), index=tune_df.index)
    for gfx, cu_num in targets:
        mask |= (tune_df["gfx"] == gfx) & (tune_df["cu_num"] == cu_num)
    return tune_df[mask]
