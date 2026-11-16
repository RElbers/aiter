# Proposal: using the catalog to make review easier

Status: proposal. Builds on [architecture-decisions.md](architecture-decisions.md);
decision numbers below refer to it.

## Summary

Review findings on aiter kernel PRs keep returning to a few bug classes, and so
do the bugs that get past review. Most of them are mechanical: an unwritten
tail, an out-of-bounds read, a stride the wrapper did not check, a dispatch
branch no test reaches. People and AI reviewers find them by reading kernel code
one PR at a time, and miss them often enough that about half of recent escaped
bugs were found by downstream frameworks rather than by aiter's own tests.

The catalog already knows, for every registered op, its inputs, its kernels,
which instances each kernel offers for a shape, and how to make inputs and judge
outputs. That is enough to check most of these classes once, generically, for
every op. This proposal adds those checks, the few catalog features they need,
and review tooling that reads declarations instead of kernel code.

In line with the guiding principle of the ADR, the cost should land in the
catalog, written once, and per-op declarations should grow by a few
small declarations at most.

## What review currently catches and misses

A survey of ROCm/aiter PRs from 2026-09-16 to 2026-11-16 looked at human review
threads on 49 kernel PRs, automated review comments on 813 kernel PRs, and 42
fix and revert PRs traced back to where the bug came in.

| Bug class | Seen in review | Escaped bugs | Caught by the catalog today |
| --- | --- | --- | --- |
| Unwritten tails, uninitialised output, out-of-bounds reads and writes | often | ~15 | no |
| Missing input validation in the wrapper (dtype, device, stride, alignment, range) | most frequent human finding | some | no |
| 32-bit offset overflow at large extents | recurs across Triton, FlyDSL and HIP | 2 | no |
| Dispatch and gate holes, optional arguments left at default | most frequent automated finding | 9 | only if a case reaches the path |
| Arch detection that an environment override can change | recurs | | no |
| Strided, expanded and offset views | often | some | aliasing only |
| Streams, graph capture, fake tensors, compile, races | sometimes | ~11 | mostly |
| Tests that are weak, never reach the new branch, or never run in CI | often | "the failing branch had no case" | no |

Of the 42 escaped bugs, about 5 would have been caught by today's standard
tests given the right case, about 15 by new generic checks, about 20 only by a
case that targets the bug, and about 3 not by tests at all (build, process).
The new generic checks below address the second group; coverage checks and
review tooling address part of the third.

## Proposed checks

### Standard tests (run on the GPU, opted into per op)

1. **Dirty memory and poisoned outputs.** Before each launch, fill the caching
   allocator's free memory with NaN bit patterns, so anything the kernel reads
   without writing first is garbage rather than leftover correct data. Where
   the op takes output or scratch buffers, poison those too. Require a finite
   output that matches the reference. Catches uninitialised merges and partial
   writes (#7613, #7182, #6926, #7653, #7047).
2. **Guard canaries.** Place each input and output tensor inside a larger
   allocation with a sentinel region before and after it, and require the
   sentinels unchanged after the launch. Catches out-of-bounds writes that leave
   results correct (#5793, #7085, #6677).
3. **Edge-input sweep.** Derive variants of each case: zero-size batch or token
   dimensions, sizes that are not a multiple of the declared vector or tile
   widths, strided, expanded and storage-offset views, padding rows filled with
   NaN or -1. For each kernel, each variant must either match the reference or
   be declined (no instance available, or `Unsupported`, or a validation error).
   A silently wrong result fails. Catches #7376, #7302, #7603, #7031, #7771,
   #7601.
4. **Negative validation.** Mutate one input field at a time to something the
   op must reject: wrong dtype, wrong device, unsupported stride or alignment,
   out-of-range scalar. The public op must raise a validation error, without
   hanging. Optionally, it must raise before allocating. Covers the most
   frequent human review finding (#7825, #7416, #7817, #7530).
5. **Determinism under interference.** Extend `is_deterministic`: repeat
   launches with unrelated work on a side stream, with the case's rows embedded
   in a larger batch (batch invariance), and as two concurrent launches of the
   same instance on two streams with different inputs. Catches races and
   shared scratch (#7574, #7622, #6954, #7825).
6. **Cold graph capture.** A variant of `graph_capture` without the warm-up, so
   a first-call allocation happens inside the capture (#7825).
7. **Device placement.** On a machine with two GPUs: inputs on a non-current
   device must work, mixed devices must raise (#7529, #7824, #7825).
8. **Large extents.** Opt-in: a case reaching 2^31 elements or bytes in some
   index, through real memory or an `as_strided` view over a small allocation
   for ops that do not touch the whole range. Expensive, so declared per op.

### Registered checks (static, CPU, over everything in the catalog)

1. **Coverage.** Every kernel, and every instance class it declares (see
   "Instance space" below), is reached by at least one case on at least one
   architecture. A new kernel or branch without a case fails here instead of in
   a downstream framework.
2. **CI discovery.** Every op with declared tests has a test module under
   `op_tests/catalog/`, and every such module is in the CI lists.
3. **Non-trivial cases.** Every case's reference is finite and not constant, so
   a comparison cannot pass vacuously.
4. **Arch gating.** Catalog declarations and the kernels they register gate on
   `available_instances` and `DeviceInfo`, never on `gcnArchName`, `GPU_ARCHS`
   or a backend's own target query.
5. **Names.** No model names in op, kernel or case names.

### Review tooling

1. **Declaration diff.** A command comparing the catalog between two revisions:
   ops and kernels added or removed, and for each kernel, which
   (architecture, dtype, shape region) cells gained or lost instances, and
   whether a case reaches each newly admitted cell. Reviewers and review agents
   read this instead of reverse-engineering gates from kernel code; it covers
   the largest automated-review category (dispatch and scope holes) and
   "no case reaches the new branch".
2. **Review instructions for catalog ops.** The review rules that the
   standard tests now enforce deterministically are skipped for an op that is in
   the catalog with those tests opted in and green: fake vs real metadata
   (review-pr D6, D7), stream synchronisation (G1), uninitialised buffers (D1),
   contiguity and raw-pointer contracts (D8, D14), gate holes and
   over-strict asserts (B1, B4, B7), new branch without a test (A2, A4). Rules
   that need judgement stay (design, performance, process). Fewer, sharper
   findings also address the noise: automated reviewers repeat findings on
   every push, and most style findings go unaddressed.
3. **Unregistered kernels.** Flag a PR that adds a kernel binding or a public
   op without registering it.

## What needs to change in the catalog

### Field roles on the input (needed by checks 1 to 4)

The input dataclass says nothing about what each field is. The checks need to
know which tensors are outputs or scratch (to poison), which may be re-laid out
(for the stride sweep), which dimensions have alignment or tile constraints,
and which rows are padding. Proposed: a way to declare a role per field
(output, scratch, layout-flexible, range-limited), optional and defaulting to a
plain input, so ops that need none declare none. How roles are expressed is not
decided.

### A decline and error contract (needed by checks 3 and 4)

The sweep and negative validation need to know what counts as a correct
refusal. Proposed:

- a kernel that cannot serve a shape yields no instance from
  `available_instances`; an instance that turns out unsupported when compiled or
  launched raises `Unsupported`;
- the public op raises a validation error (`ValueError` or `TypeError`) for
  inputs outside its contract, before launching anything;
- anything else (a hang, an illegal memory access, a wrong result) fails.

### Shape constraints for case variants (needed by check 3)

To derive tail and boundary variants without per-op code, a case or shape needs
to state its alignment-relevant dimensions and their units (vector width, tile
sizes). Proposed: a small declaration on the shape type saying, for example,
that a dimension must be a multiple of 8, from which the runner derives sizes
just below, at and just above each multiple. Ops that declare nothing get no tail
variants.

### Instance space (needed by registered check 1)

Coverage needs the set of instance classes a kernel can offer, independent of a
case, so it can compare what exists with what cases reach. Proposed: an optional
way for a kernel to declare its instance classes for a device, without
compiling anything, in keeping with the constraint on `available_instances`
(decision 3). Kernels with one instance (`NoTunables`) need nothing.

### Workspace ownership (needed by check 5)

Concurrent launches need each launch to get its own scratch. With a `Scratch`
field role, the runner allocates one per launch; kernels with hidden
process-wide scratch are what the concurrent variant is meant to expose.

### Smaller changes

- **Reference caching** per case, since several checks judge the same input.
- **Per-kernel opt-out** of a single check with a reason (an open question in
  the ADR), for kernels that legitimately cannot pass one, such as a kernel that
  syncs with the host.
- **Arch matrix in the declaration diff.** `available_instances` takes a
  `DeviceInfo`, so the diff can evaluate it for every architecture in
  `GFX_CU_NUMS_MAP` on a CPU, without the hardware.

## Plan

**Phase 1: no API change.**

- Checks: dirty memory, empty-size variants where a case can express them,
  determinism under interference, cold capture, device placement.
- Registered checks: kernel coverage by case, CI discovery, non-trivial cases,
  arch gating, names.

**Phase 2: field roles and the decline and error contract.**

- Checks: poisoned output and scratch buffers, guard canaries, the edge-input
  sweep with stride and tail variants, negative validation, concurrent launches.
- Catalog: shape constraints, reference caching.

**Phase 3: review tooling.**

- The declaration diff, instance space and instance-class coverage.
- Review-pr integration: rules skipped for catalog ops, the unregistered-kernel
  check.
- Large extents, opt-in.

Each phase is useful on its own; phase 1 adds no per-op code at all.

## What it costs an op

| Phase | Per-op addition |
| --- | --- |
| 1 | nothing; the new standard tests are listed in `kernel_tests` / `op_tests` |
| 2 | roles for the input fields that are outputs, scratch, layout-flexible or range-limited; the alignment of aligned shape dimensions |
| 3 | a declaration of the instance classes, for kernels with more than one |

## Limits

- Prebuilt assembly kernels: tests catch the symptom, but review cannot see the
  cause.
- Collectives (several ranks, repeated graph capture across ranks, IPC) need
  multi-GPU infrastructure beyond per-op tests.
- Bugs that need a specific numeric situation (cancellation with large finite
  logits, bf16 rounding bias, clamping at extreme magnitudes) still need cases
  written for them; the catalog makes such a case one line, not a check.
- The benefit applies to ops in the catalog, so it grows with migration.

## Open questions

- How field roles, shape alignments and instance classes are declared, and
  whether field roles can be inferred for common patterns (a tensor argument named `out`).
- Whether negative validation should require refusal before allocation, or only
  a clean refusal.
- How dirty memory interacts with the caching allocator's per-stream pools and
  with graph capture's private pool.
- Whether the declaration diff runs in CI on every PR, or on demand for
  reviewers and review agents.
