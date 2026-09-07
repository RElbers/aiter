# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.
#
# Pure-Python arch constants and env-driven build target resolution.
# No torch dependency — safe to import in build scripts, gen_instances, and tests
# that run without a GPU or a full PyTorch install.
import logging
import os
import re

logger = logging.getLogger("aiter")

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
    "gfx1250": 256,  # Gfx1250
}

# Arch names a target may name. GFX_CU_NUM_MAP holds only those with a default
# CU count; the rest are valid targets but have to name their count explicitly.
KNOWN_GFX = frozenset(v for v in GFX_MAP.values() if v != "native")


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


def _parse_cu_num(value, ctx: str) -> int:
    """Parse a CU count, naming ``ctx`` in the error rather than raising a bare
    ValueError whose traceback does not say which variable was wrong."""
    try:
        cu_num = int(value)
    except (TypeError, ValueError):
        raise RuntimeError(f"{ctx}: CU count {value!r} is not an integer.") from None
    if cu_num <= 0:
        raise RuntimeError(f"{ctx}: CU count {cu_num} must be positive.")
    return cu_num


def _parse_gpu_targets_env() -> list[tuple[str, int]] | None:
    """Parse AITER_GPU_TARGETS into (gfx, cu_num) targets, or None if it is unset.

    gfx950:128;gfx950:256  -> [("gfx950", 128), ("gfx950", 256)]
    gfx950                 -> [("gfx950", 256)]  # CU from GFX_CU_NUM_MAP
    gfx942;gfx950:128      -> [("gfx942", 304), ("gfx950", 128)]
    (unset or blank)       -> None

    Entries split on ';' or ','. Arch names are case-folded and checked against
    KNOWN_GFX, and CU counts must be positive integers: filter_tune_df compares
    the arch string exactly, so an unvalidated typo matches no row and detunes
    the whole build without an error.
    """
    targets_env = os.getenv("AITER_GPU_TARGETS")
    if not targets_env or not targets_env.strip():
        return None
    ctx = f"AITER_GPU_TARGETS={targets_env!r}"

    targets = []
    for entry in re.split(r"[;,]", targets_env):
        entry = entry.strip()
        if not entry:
            continue
        gfx, sep, cu = entry.partition(":")
        gfx = gfx.strip().lower()
        if gfx not in KNOWN_GFX:
            raise RuntimeError(
                f"{ctx}: unknown gfx {gfx!r} in entry {entry!r}. "
                f"Known targets: {sorted(KNOWN_GFX)}"
            )
        if sep:
            targets.append((gfx, _parse_cu_num(cu.strip(), f"{ctx}, entry {entry!r}")))
        elif gfx in GFX_CU_NUM_MAP:
            targets.append((gfx, GFX_CU_NUM_MAP[gfx]))
        else:
            raise RuntimeError(
                f"{ctx}: {gfx!r} has no default CU count — add it to "
                f"GFX_CU_NUM_MAP in build_targets.py, or name the count "
                f"explicitly as '{gfx}:<cu_num>'."
            )

    if not targets:
        raise RuntimeError(
            f"{ctx} names no targets. Expected entries of the form "
            f"'gfx' or 'gfx:cu_num'."
        )

    # Preserve caller order for backward compatibility while removing exact
    # duplicates. Cache identities canonicalize separately.
    return list(dict.fromkeys(targets))


def get_build_archs_env() -> list[str] | None:
    """Deduped arch names from AITER_GPU_TARGETS, or None if it is unset.

    AITER_GPU_TARGETS decides what a build compiles as well as which tuned rows
    it bakes. An arch named only here still has to reach --offload-arch, or the
    module ships dispatch entries for device code it does not contain.
    """
    targets = _parse_gpu_targets_env()
    if targets is None:
        return None
    return list(dict.fromkeys(gfx for gfx, _ in targets))


def has_named_targets() -> bool:
    """True when the env names build targets explicitly.

    Lets a caller tell a configuration error (a typo'd arch, a bad CU count)
    apart from "no GPU here and nothing named", which is the only case that may
    quietly fall back to an unfiltered frame.
    """
    if (os.getenv("AITER_GPU_TARGETS") or "").strip():
        return True
    gfx = (os.getenv("GPU_ARCHS") or "").strip()
    return bool(gfx) and gfx.lower() != "native"


def get_build_targets_env() -> list[tuple[str, int]]:
    """Resolve build targets from env only.  No live GPU detection.

    Use AITER_GPU_TARGETS to pick which tuned (gfx, cu_num) kernels get built.
    Use GPU_ARCHS to pick which archs get compiled: it also feeds --offload-arch
    and the hsaco cache path, so it stays a bare arch list.
    If set, AITER_GPU_TARGETS overrides GPU_ARCHS + CU_NUM for resolving build
    targets.

    Raises RuntimeError if neither is set or an arch is unknown.
    Intended for CI nodes, build scripts, and tests that run without a GPU.
    Use chip_info.get_build_targets() when live GPU fallback is also desired.

    AITER_GPU_TARGETS=gfx950:128;gfx950:256 -> [("gfx950", 128), ("gfx950", 256)]
    GPU_ARCHS=gfx942;gfx950                 -> [("gfx942", 304), ("gfx950", 256)]
    GPU_ARCHS=gfx942 CU_NUM=80              -> [("gfx942", 80)]
    """
    targets = _parse_gpu_targets_env()
    if targets is not None:
        return targets

    gfx_env = os.getenv("GPU_ARCHS")
    if not gfx_env:
        raise RuntimeError(
            "Neither AITER_GPU_TARGETS nor GPU_ARCHS is set. "
            "Set GPU_ARCHS=gfx942 (or similar) to resolve build targets without a GPU."
        )
    cu_env = os.getenv("CU_NUM")
    targets = []
    for gfx in _parse_gpu_archs_env(gfx_env):
        gfx = gfx.lower()
        if gfx not in GFX_CU_NUM_MAP:
            raise RuntimeError(
                f"Unknown gfx '{gfx}' in GPU_ARCHS — add it to "
                f"GFX_CU_NUM_MAP in build_targets.py. Known targets: "
                f"{list(GFX_CU_NUM_MAP.keys())}"
            )
        cu_num = (
            _parse_cu_num(cu_env, f"CU_NUM={cu_env!r}")
            if cu_env
            else GFX_CU_NUM_MAP[gfx]
        )
        targets.append((gfx, cu_num))
    return list(dict.fromkeys(targets))


def filter_tune_df(tune_df, targets: list, source: str = ""):
    """Return the subset of tune_df whose (gfx, cu_num) matches any entry in targets.

    Warns for every target that matched no row. Filtering happens at build time,
    so a target with no tuned rows compiles only the default kernel for every
    shape and leaves no runtime signal that anything is missing.

    Args:
        tune_df:  pandas DataFrame loaded from a tuning CSV (must have 'gfx' and
                  'cu_num' columns).
        targets:  list of (gfx, cu_num) tuples, as returned by get_build_targets()
                  or get_build_targets_env().
        source:   optional subject for the warning, e.g. a CSV basename.

    Returns:
        Filtered DataFrame (original index preserved, no reset).
    """
    import pandas as pd

    mask = pd.Series([False] * len(tune_df), index=tune_df.index)
    unmatched = []
    for gfx, cu_num in targets:
        hit = (tune_df["gfx"] == gfx) & (tune_df["cu_num"] == cu_num)
        if not hit.any():
            unmatched.append(f"{gfx}:{cu_num}")
        mask |= hit
    if unmatched and len(tune_df):
        logger.warning(
            "%s has no rows for build target(s) %s; every shape there falls "
            "back to the default kernel. Tune those targets, or drop them from "
            "AITER_GPU_TARGETS / GPU_ARCHS.",
            source or "The tuned config CSV",
            ", ".join(unmatched),
        )
    return tune_df[mask]
