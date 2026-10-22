# Explicit GPU build targets

`AITER_BUILD_TARGETS` describes the destination GPU variants for an explicit
build. It supports several CU counts of one architecture and mixed architectures,
without using the build host's GPU as a substitute. Runtime hardware detection,
launch geometry, and an operation's cold JIT build ignore this variable.

The shared request APIs are available while backend adapters are introduced.
Explicit native builds reject module recipes whose adapters are not available
in this revision. Legacy builds retain their existing behavior. Full-package
explicit builds are not enabled yet; use selected native modules when supported.

```sh
AITER_BUILD_TARGETS='gfx950:128;gfx950:256' \
  python aiter/jit/prebuild.py module_gemm_a8w8

PREBUILD_KERNELS=1 PREBUILD_MODULES='module_gemm_a8w8' \
  AITER_BUILD_TARGETS='gfx942:304;gfx950:128' python setup.py bdist_wheel
```

`PREBUILD_MODULES` selects native modules and skips package-wide FlyDSL AOT.
Module names come from `aiter/jit/optCompilerConfig.json`. Run live-GPU pretuning
separately from explicit cross-builds.

## Input and compatibility

The syntax is semicolon-separated `gfx:positive_cu_count` pairs. Architecture
names are validated against `GFX_CU_NUM_MAP` in `aiter/jit/utils/build_targets.py`,
including gfx942, gfx950, gfx1250 and the listed Radeon architectures. Names and
whitespace are normalized, duplicates removed, and order canonicalized. Parser
acceptance does not extend any backend's hardware support. Bare architectures, `native`,
wildcards, and comma-separated pairs are not part of the new syntax. Unset or blank retains legacy behavior.

Existing `GPU_ARCHS` and `CU_NUM` retain their syntax and behavior. When supplied
alongside explicit pairs, their architecture set and scalar CU count must agree
with the request. Otherwise the entry point reports a conflict without changing
the caller's environment. Leave `CU_NUM` unset when requesting different counts.

Each instruction-set architecture appears once in the compiler flags. CU count
selects tuning rows or compile-time specialization only where the backend uses
it. Required coverage does not imply that every other compatible kernel is
removed from the artifact.

## Backend coverage

| Build path | Destination handling |
| --- | --- |
| CK GEMM, batched GEMM and CK Tile GEMM | Filter tuned rows by exact `(gfx, cu_num)` pairs, preserve fallback kernels and existing lookup keys, and compile for the unique architectures. CK Tile registries resolve destinations lazily; explicit builds do not probe the host during imports. |
| CK two-stage MoE | Select applicable prebuild variants from tuning metadata and generate destination-specific registries. Mixed builds select the matching heuristic using execution hardware. |
| CK Tile MXFP4 MoE | Include its existing gfx950 implementation when requested. This does not add a gfx942 or gfx1250 implementation. |
| CK Tile attention recipes | Pass the destination families to the upstream generator, including its `gfx9` name for gfx942. |

## Programmatic boundary

Resolve the request at an explicit entry point, before evaluating recipes.
Filter the arguments against the build function's signature: recipes also
contain metadata such as `is_experimental` that `build_module` does not accept.

```python
# As setup.py does, add the checkout's aiter/ directory to sys.path to import
# build tooling without initializing runtime operators.
import inspect

from jit import core
from jit.utils.build_targets import resolve_build_targets

targets = resolve_build_targets("gfx942:304;gfx950:128")
module = "module_gemm_a8w8"
args = core.get_args_of_build(module, build_targets=targets)
args["md_name"] = module
parameters = inspect.signature(core.build_module).parameters
args = {key: value for key, value in args.items() if key in parameters}
core.build_module(**args)
```

Shared build functions receive immutable pairs and never read
`AITER_BUILD_TARGETS` implicitly. Destination-aware codegen receives explicit
CLI arguments. Runtime JIT children omit the inherited variable, even when it
contains an invalid request. Parallel builds keep their requests local.

## Tuning data

Requested CU variants need existing tuning rows. Missing rows warn with the
target and table/backend while keeping supported fallbacks or runtime JIT.
Unknown kernel IDs/names in applicable CK rows still fail strictly.
Target selection never substitutes the build host's GPU.
