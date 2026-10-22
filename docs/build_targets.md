# Explicit GPU build targets

`AITER_BUILD_TARGETS` describes the destination GPU variants for an explicit
build. It supports several CU counts of one architecture and mixed architectures,
without using the build host's GPU as a substitute. Runtime hardware detection,
launch geometry, and an operation's cold JIT build ignore this variable.

```sh
# One instruction-set target, two tuning variants.
AITER_BUILD_TARGETS='gfx950:128;gfx950:256' \
  python aiter/jit/prebuild.py module_gemm_a8w8

# Cross-compile with no GPU visible.
HIP_VISIBLE_DEVICES='' ROCR_VISIBLE_DEVICES='' \
  python aiter/jit/prebuild.py module_gemm_a8w8 \
  --build-targets 'gfx942:304;gfx950:128'

# Use the existing full package prebuild selection, including FlyDSL AOT.
PREBUILD_KERNELS=1 AITER_BUILD_TARGETS='gfx942:304;gfx950:128;gfx950:256' \
  python setup.py bdist_wheel

# Or prebuild selected native modules only.
PREBUILD_KERNELS=1 PREBUILD_MODULES='module_gemm_a8w8;module_deepgemm_opus' \
  AITER_BUILD_TARGETS='gfx942:304;gfx950:128' python setup.py bdist_wheel
```

Module names come from `aiter/jit/optCompilerConfig.json`. `PREBUILD_MODULES`
replaces the usual native module selection and skips package-wide FlyDSL AOT.
Without it, the existing `PREBUILD_KERNELS` selection and exclusions apply.
Live-GPU pretuning must run separately from explicit cross-builds. Keep
`AITER_JIT_DIR` unset for a wheel so native modules land in its normal package
location; standalone prebuilds can use it to select their output directory.

The CK Tile preshuffled GEMM tuning module can also be cross-built through
`aiter/jit/prebuild.py` or `PREBUILD_MODULES`. Its candidate tables retain each
architecture's kernel IDs, and execution selects the table for the current GPU.
Several CU variants of one architecture share the same candidate table.

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
| OPUS | Filter by architecture ownership, retaining the mandatory/default compiled kernels and applicable tuned kernels. CU pruning is deliberately not applied. Per-source compiler options survive package prebuilds. |
| CK Tile attention recipes | Pass the destination families to the upstream generator, including its `gfx9` name for gfx942. |
| Assembly manifest codegen | Select the requested architecture entries. Existing object availability and runtime guards still apply. |
| FlyDSL AOT | Package and all AOT CLIs use the same job selector. GEMM, convolution and MoE use job metadata; chunk GDN honors explicit gfx and expands portable jobs. Persistent MXFP4 retains CU specialization. Fixed gfx950 families keep their actual compiler target. |
| Generic template HIP libraries | The explicit Python API supports multi-architecture compilation, with compatible cache lookup shared by Python and C++. CU count adds no cache dimension to these libraries. |

FlyDSL CLI example (requires the FlyDSL version pinned by this checkout):

```sh
AITER_AOT_IMPORT=1 python -m aiter.aot.flydsl.gemm \
  --build-targets 'gfx950:128;gfx950:256' --csv /path/to/tuned.csv
```

Legacy FlyDSL `ARCH`/`GPU_ARCHS` filtering now applies consistently to package
builds and CLIs. `native` is resolved once to the detected architecture before
selection, fixing the previous literal-string filter. The compiler still uses
the job's destination, including the legacy CU-to-architecture fallback for old
rows without gfx metadata. MoE workers clear cached CU answers both when entering
and when leaving a compilation override.

AOT selection does not add missing algorithms or configurations. Grouped MoE's
existing AOT implementation covers auxiliary kernels. Mega MoE retains its fixed
gfx950/256-CU compilation constants. Unsupported batched FlyDSL kernel families
still emit the existing missing-AOT-wiring warning. These constraints also apply
when a package requests additional variants.

## Programmatic boundary and template cache

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

`csrc.cpp_itfs.utils.compile_template_op` and `compile_lib` accept the same
`build_targets` keyword. Their runtime callers omit it. Template artifacts live
under `AITER_ROOT_DIR/build/targets/<sorted architectures joined by +>/<folder>`.
Python and C++ prefer an exact match, then an existing compatible superset.
Runtime can therefore find a fat library after build variables are removed.
Libraries without architecture-qualified paths are rebuilt once. Publication is
atomic, locks use the same directory, and loaded handles are keyed by resolved
file path. Custom folders are supported. `AITER_ROOT_DIR` is the exact root in
both languages, fixing C++'s previous extra `/.aiter` suffix.

## Tuning data and workflow entry points

Requested CU variants need existing tuning rows. Missing rows warn with the
target and table/backend while keeping supported fallbacks or runtime JIT.
Unknown kernel IDs/names in applicable CK rows still fail strictly. A missing
embedded object is a separate error; neither case changes the destination to
the build host's GPU.

The release, nightly and release-automation workflows accept pairs in their
existing `gpu_archs` **workflow input**, for example `gfx950:128;gfx950:256`.
A build step translates this into architecture-only `GPU_ARCHS` and explicit
`AITER_BUILD_TARGETS`, forwards both into the build container, and records them
in the summary. Legacy architecture-only workflow inputs remain unchanged. This
keeps the release workflow within the ten `workflow_dispatch` input limit; it
does not extend the syntax of the `GPU_ARCHS` environment variable.

When the workflow builds an older revision without explicit-target support,
architecture-only inputs still work. Explicit pairs are rejected with an
unsupported-revision error rather than silently losing the requested CU variants.
