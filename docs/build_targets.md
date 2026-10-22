# Explicit GPU build targets

`AITER_BUILD_TARGETS` describes the destination GPU variants for an explicit
build. It supports several CU counts of one architecture and mixed architectures,
without using the build host's GPU as a substitute. Runtime hardware detection,
launch geometry, and an operation's cold JIT build ignore this variable.

The shared request APIs are available while backend adapters are introduced.
Explicit native builds reject module recipes whose adapters are not available
in this revision. Builds without explicit targets retain their existing behavior.
Full-package explicit builds are not enabled yet; use selected native modules
when supported.

## Input and compatibility

The syntax is semicolon-separated `gfx:positive_cu_count` pairs. Architecture
names are validated against `GFX_CU_NUM_MAP` in `aiter/jit/utils/build_targets.py`,
including gfx942, gfx950, gfx1250 and the listed Radeon architectures. Names and
whitespace are normalized, duplicates removed, and order canonicalized. Parser
acceptance does not extend any backend's hardware support. Bare architectures,
`native`, wildcards, and comma-separated pairs are not part of the new syntax.
Unset or blank keeps the existing `GPU_ARCHS`/`CU_NUM` and hardware-detection
behavior.

`GPU_ARCHS` and `CU_NUM` remain supported configuration inputs with their existing
syntax and behavior. When supplied alongside explicit pairs, their architecture
set and scalar CU count must agree with the request. Otherwise the entry point
reports a conflict without changing the caller's environment. Leave `CU_NUM`
unset when requesting different counts.

Each instruction-set architecture appears once in the compiler flags. CU count
selects tuning rows or compile-time specialization only where the backend uses
it. Required coverage does not imply that every other compatible kernel is
removed from the artifact.

## Programmatic boundary

Resolve with `resolve_build_targets` at an explicit entry point. Pass the result
as `build_targets` to `get_args_of_build` and `build_module` once the module's
adapter is available. Shared runtime callers omit that argument. Runtime JIT
children omit inherited `AITER_BUILD_TARGETS`, and parallel requests keep their
state local. `parse_build_targets` and `format_build_targets` expose the canonical
representation without probing a GPU.
