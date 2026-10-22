# SPDX-License-Identifier: MIT
# Copyright (C) 2024-2026, Advanced Micro Devices, Inc. All rights reserved.

"""GPU target resolution and cache identity tests with mocked GPU probes."""

import contextlib
import os
import sys
from unittest import TestCase, mock

# Ensure the repo-local aiter is imported, not any system/site-packages install.
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)


def test_runtime_arch_resolution():
    from aiter.jit.utils import chip_info

    gpu_target_strings = ("gfx950:256;gfx942:304", "gfx942:304;gfx950:256")
    live_expected_gfxs = (
        ("gfx942", "gfx942"),
        ("gfx1201", "gfx950"),
    )
    for gpu_target_string in gpu_target_strings:
        for live_gfx, expected_gfx in live_expected_gfxs:
            with (
                _target_env(AITER_BUILD_TARGETS=gpu_target_string),
                _cleared_cache(chip_info.get_gfx_custom_op_core),
                mock.patch.object(chip_info, "_detect_native", return_value=[live_gfx]),
            ):
                resolved_gfx = chip_info.GFX_MAP[chip_info.get_gfx_custom_op_core()]

            assert (
                resolved_gfx == expected_gfx
            ), f"{gpu_target_string}, live {live_gfx}: got {resolved_gfx}"


def test_gpu_archs_takes_the_live_cu_count():
    from aiter.jit.utils import chip_info

    with (
        _target_env(GPU_ARCHS="gfx950"),
        mock.patch.object(chip_info, "get_gfx_runtime", return_value="gfx950"),
        mock.patch.object(chip_info, "get_cu_num", return_value=128),
    ):
        targets = chip_info.get_build_targets()

    assert targets == [("gfx950", 128)], targets


def test_build_target_environment_contract():
    from aiter.jit.utils.build_targets import get_build_targets_env

    cases = (
        ({}, "Neither AITER_BUILD_TARGETS nor GPU_ARCHS is set"),
        ({"GPU_ARCHS": "native"}, "needs a live GPU"),
        ({"GPU_ARCHS": " ; "}, "contains no valid architecture names"),
        ({"GPU_ARCHS": "native,gfx950"}, "cannot be combined"),
        ({"GPU_ARCHS": "gfx_unknown"}, "unknown gfx"),
        ({"GPU_ARCHS": "gfx950", "CU_NUM": "0"}, "must be a positive integer"),
        # The env-only entry point diagnoses CU_NUM before parsing GPU_ARCHS.
        ({"GPU_ARCHS": " ; ", "CU_NUM": "bad"}, "CU_NUM="),
        ({"GPU_ARCHS": "gfx90a", "CU_NUM": "64"}, "no default CU count"),
        ({"AITER_BUILD_TARGETS": " ; "}, "names no targets"),
        ({"AITER_BUILD_TARGETS": "native"}, "unknown gfx"),
        ({"AITER_BUILD_TARGETS": "gfx950:-1"}, "must be a positive integer"),
        ({"AITER_BUILD_TARGETS": "gfx950:1:2"}, "must be a positive integer"),
        ({"AITER_BUILD_TARGETS": "gfx90a"}, "no default CU count"),
        ({"AITER_BUILD_TARGETS": "gfx90a:64"}, [("gfx90a", 64)]),
        ({"AITER_BUILD_TARGETS": " \t", "GPU_ARCHS": "gfx950"}, [("gfx950", 256)]),
        (
            {"GPU_ARCHS": " GFX950, gfx942;gfx950 ", "CU_NUM": "80"},
            [("gfx950", 80), ("gfx942", 80)],
        ),
        (
            {"AITER_BUILD_TARGETS": " GFX950:128,gfx942;gfx950:128;gfx950 "},
            [("gfx950", 128), ("gfx942", 304), ("gfx950", 256)],
        ),
        (
            {"AITER_BUILD_TARGETS": "gfx950", "GPU_ARCHS": " ; ", "CU_NUM": "bad"},
            [("gfx950", 256)],
        ),
    )
    for env, expected in cases:
        with _target_env(**env):
            if isinstance(expected, str):
                with TestCase().assertRaisesRegex(RuntimeError, expected, msg=env):
                    get_build_targets_env()
            else:
                assert get_build_targets_env() == expected, env


def test_build_target_live_fallback_contract():
    from aiter.jit.utils import chip_info

    # A named bare arch stays pinned; only GPU_ARCHS adopts the live CU count.
    cases = (
        ({"AITER_BUILD_TARGETS": "gfx950"}, "gfx950", [("gfx950", 256)]),
        ({"GPU_ARCHS": "gfx950;gfx942"}, "gfx950", [("gfx950", 128), ("gfx942", 304)]),
        ({"GPU_ARCHS": "gfx950", "CU_NUM": "80"}, "gfx950", [("gfx950", 80)]),
        ({"GPU_ARCHS": "gfx950"}, "gfx942", [("gfx950", 256)]),
        ({"GPU_ARCHS": "gfx950"}, RuntimeError("no GPU"), [("gfx950", 256)]),
        ({"GPU_ARCHS": "gfx950"}, KeyError("unknown GPU"), [("gfx950", 256)]),
        ({}, "gfx950", [("gfx950", 128)]),
        ({"GPU_ARCHS": " NATIVE "}, "gfx950", [("gfx950", 128)]),
        ({}, RuntimeError("no GPU"), "No GPU detected"),
        # This entry point parses GPU_ARCHS before inspecting CU_NUM.
        ({"GPU_ARCHS": " ; ", "CU_NUM": "bad"}, "gfx950", "contains no valid"),
    )
    for env, live, expected in cases:
        probe = (
            mock.Mock(side_effect=live)
            if isinstance(live, Exception)
            else mock.Mock(return_value=live)
        )
        with (
            _target_env(**env),
            mock.patch.object(chip_info, "get_gfx_runtime", probe),
            mock.patch.object(chip_info, "get_gfx", probe),
            mock.patch.object(chip_info, "get_cu_num", return_value=128),
        ):
            if isinstance(expected, str):
                with TestCase().assertRaisesRegex(RuntimeError, expected, msg=env):
                    chip_info.get_build_targets()
            else:
                assert chip_info.get_build_targets() == expected, (env, live)


def test_target_resolution_warnings():
    import pandas as pd

    from aiter.jit import core
    from aiter.jit.utils import chip_info

    for gpu_archs, warning in (
        ("gfx942", "disagrees"),
        (" ; ", "Ignoring GPU_ARCHS"),
        ("gfx950", None),
    ):
        with (
            _target_env(AITER_BUILD_TARGETS="gfx950", GPU_ARCHS=gpu_archs),
            mock.patch.object(core.logger, "warning") as log,
        ):
            assert core.validate_and_update_archs() == ["gfx950"]
            assert log.call_count == int(warning is not None)
            if warning:
                assert warning in log.call_args.args[0]

    with (
        _target_env(AITER_BUILD_TARGETS="gfx950", CU_NUM="128"),
        mock.patch.object(chip_info.logger, "warning") as log,
    ):
        assert chip_info.get_build_targets() == [("gfx950", 256)]
        assert "does not match any build target" in log.call_args.args[0]

    df = pd.DataFrame({"gfx": ["gfx950"], "cu_num": [256], "libtype": ["ck"]})
    with (
        _target_env(AITER_BUILD_TARGETS="gfx950"),
        mock.patch.object(chip_info.logger, "warning") as log,
    ):
        assert chip_info._select_tuned_rows(df, "cktile").empty
        assert log.call_args.args[1] == "gfx950:256"


def test_native_resolution_depends_on_the_caller():
    from aiter.jit import core
    from aiter.jit.utils import chip_info

    for windows, expected in ((False, ["native"]), (True, ["gfx950"])):
        with (
            _target_env(),
            mock.patch.object(core, "IS_WINDOWS", windows),
            mock.patch.object(core, "_detect_native", return_value=["gfx950"]),
        ):
            assert core.validate_and_update_archs() == expected

    with (
        _target_env(),
        _cleared_cache(chip_info.get_gfx_list),
        _cleared_cache(chip_info.get_gfx_custom_op_core),
        mock.patch.object(
            chip_info, "_detect_native", side_effect=RuntimeError("no GPU")
        ),
    ):
        assert chip_info.get_gfx_list() == ["cpu"]
        assert os.environ["AITER_GPU_ARCHS"] == "cpu"
        with TestCase().assertRaisesRegex(RuntimeError, "no GPU"):
            chip_info.get_gfx_custom_op_core()


def test_unparseable_gpu_archs_defers_to_build_targets():
    from aiter.jit import core

    for gpu_archs in ("   ", " ; "):
        with _target_env(AITER_BUILD_TARGETS="gfx950", GPU_ARCHS=gpu_archs):
            archs = core.validate_and_update_archs()

        assert archs == ["gfx950"], f"GPU_ARCHS={gpu_archs!r}: got {archs}"


def test_gpu_archs_parses_the_same_everywhere():
    from aiter.jit.utils import build_targets, chip_info

    resolvers = (
        build_targets.gpu_archs_env_names,
        build_targets.get_build_targets_env,
        chip_info.get_gfx_list,
    )
    with (
        _target_env(GPU_ARCHS=" GFX942, gfx950;gfx942 "),
        _cleared_cache(chip_info.get_gfx_list),
    ):
        archs = [resolve() for resolve in resolvers]

    assert archs == [
        ["gfx942", "gfx950"],
        [("gfx942", 304), ("gfx950", 256)],
        ["gfx942", "gfx950"],
    ], archs

    for gpu_archs in ("", " ; ", "native;gfx942"):
        for resolve in resolvers:
            with (
                _target_env(GPU_ARCHS=gpu_archs),
                _cleared_cache(chip_info.get_gfx_list),
            ):
                try:
                    got = resolve()
                except RuntimeError:
                    continue
            raise AssertionError(
                f"GPU_ARCHS={gpu_archs!r}: {resolve.__name__} -> {got}"
            )

    # get_gfx_list runs at import, so an arch only some modules support must
    # not fail it; resolving build targets is where unknown names are rejected.
    for gpu_archs in ("gfx942;gfx1030", "gfx950;gfx942:xnack-"):
        with (
            _target_env(GPU_ARCHS=gpu_archs),
            _cleared_cache(chip_info.get_gfx_list),
        ):
            assert chip_info.get_gfx_list() == gpu_archs.split(";"), gpu_archs
            try:
                got = build_targets.get_build_targets_env()
            except RuntimeError:
                continue
        raise AssertionError(f"GPU_ARCHS={gpu_archs!r}: get_build_targets_env -> {got}")


@contextlib.contextmanager
def _target_env(**values):
    with mock.patch.dict(os.environ):
        for name in ("AITER_BUILD_TARGETS", "GPU_ARCHS", "CU_NUM"):
            os.environ.pop(name, None)
        os.environ.update(values)
        yield


@contextlib.contextmanager
def _cleared_cache(function):
    function.cache_clear()
    try:
        yield
    finally:
        function.cache_clear()


if __name__ == "__main__":
    test_build_target_environment_contract()
    test_build_target_live_fallback_contract()
    test_target_resolution_warnings()
    test_native_resolution_depends_on_the_caller()
    test_runtime_arch_resolution()
    test_gpu_archs_takes_the_live_cu_count()
    test_unparseable_gpu_archs_defers_to_build_targets()
    test_gpu_archs_parses_the_same_everywhere()
    print("ALL_PASS")
