# Operation catalog: architecture decisions

The catalog describes aiter's public operations and the kernels that implement
them, and runs one set of tests against every implementation. This record lists
the decisions behind its current shape, the reason for each, and what each one
costs. Open questions are at the end.

Code: `aiter/catalog/`. Tests: `tests/catalog/` (the catalog itself) and
`op_tests/catalog/` (the operations it describes).

## Guiding principle: little, simple boilerplate per op

Describing an operation and its kernels necessarily takes some code per op and
per kernel. The aim is to keep that code small and plain, and to move the
complexity into the code behind the catalog, which is written once: the
registry, the standard tests and the test runner. When a choice trades one for
the other, the per-op code gets the simpler side.

What an op's author writes is declarations and small adapters:

| For | Code |
| --- | --- |
| the operation | an `Input` dataclass mirroring the public signature, a `Shape` dataclass and `input_to_shape`, a `run` that unpacks the input, one `register_operation` call |
| each kernel | a `run(input, instance)` adapter over its binding, `available_instances`, one `register_kernel` call |
| its tests | a case class with `make_input`, the cases, `reference`, `compare_outputs`, and the list of tests it opts into |
| its test file | one line in `op_tests/catalog/test_<op>.py` |

What the code behind the catalog handles, once for every op: discovering and
registering declarations, type-checking them against each other, choosing and
capping instances per shape and device, copying inputs, skipping unavailable or
unsupported instances, naming failures, and the mechanics of the standard
tests (graph capture and its recovery, streams, fake tensors, compilation).

The decisions below are measured against this principle; several of them
(decisions 3, 5, 6 and 8 in particular) exist to keep per-op code out of the
mechanics.

## 1. The catalog is a description, not a dispatcher

**Context.** aiter already routes each public op to a backend inside the op
(for `rms_norm`, `_use_hip_common` picks HIP or OPUS). A second routing layer
would compete with it.

**Decision.** The catalog records metadata and runs tests. It does not wrap
functions, change dispatch or launch work on the hot path. `aiter` never
imports `aiter.catalog`; the catalog imports `aiter`.

**Consequences.** Registering an op changes nothing for its users. The catalog
can describe an op without the op's code changing, and its declarations live in
their own package rather than inside `aiter/ops/`.

## 2. An operation is the existing public function

**Decision.** `register_operation(func, run, *, input_type, shape_type,
input_to_shape)` registers the public function itself (`aiter.rms_norm`), with:

- `input_type`: a frozen dataclass whose fields are the function's parameters,
  in order, with the same kinds, defaults and annotations;
- `run(input) -> R`: calls the function with the input's fields;
- `input_to_shape(input) -> Shape`: the input's shape, as kernels need it to
  say what they can serve.

**Consequences.** Every implementation, the op and its kernels, is launched the
same way: on an `Input`. Tests make and copy one value instead of argument
lists. The match between `input_type` and the function signature is checked for
every registered op (`tests/catalog/registered/test_catalog_registered_signatures.py`).

`torch_compile_guard` is annotated to return a function with the guarded
function's signature, so the type checker sees the op's real parameters through
the decorator. That annotation also makes the checker see existing call sites
across the repo that it previously treated as untyped.

## 3. A kernel is a binding, a run adapter and its available instances

**Context.** An earlier design required each kernel wrapper to take the op's
parameters plus keyword-only tunables, with the signatures compared. Bindings
have their own signatures (output buffers, extra flags), so every kernel needed
a wrapper that mirrored the op.

**Decision.** `op.register_kernel(func, run, *, backend, instance_type,
available_instances)`:

- `func` is the kernel's own binding (`aiter.ops.rmsnorm.rmsnorm`). It gives
  the kernel its identity and default name.
- `run(input, instance) -> R` adapts the op's input to the binding.
- An *instance* is one configuration of the kernel, of type `instance_type`;
  `NoTunables` for kernels with nothing to configure.
- `available_instances(shape, device)` yields the instances available for a
  shape on a device. It answers what the kernel supports, not what dispatch
  would choose. It is not checked against dispatch's routing. It must be cheap
  and must not compile anything, so it may claim instances that only turn out
  to be unsupported when compiled or launched; the kernel then raises
  `Unsupported`.

**Consequences.** There is no signature contract between kernels and ops to
maintain. A kernel with no instance available for a shape yields none, and
tests skip it for that shape. Because `run` takes the instance as a parameter,
the type checker rejects `available_instances` whose instances do not match
`instance_type`.

## 4. Declarations are discovered, one package per op

**Decision.** Each op has a package under `aiter/catalog/ops/<op>/` with
`api.py` (the operation), `kernels.py` (its kernels) and `tests.py` (its test
declaration). Importing `aiter.catalog` imports every module under
`aiter.catalog.ops` (`pkgutil.walk_packages`), which registers everything they
declare.

**Consequences.** Adding an op is adding a package; nothing lists ops by hand.
Every op directory needs an `__init__.py` so packaging includes it. Importing
the catalog imports the test declarations too, which need torch but not pytest.

## 5. Tests are declared per op and opted into explicitly

**Decision.** `op.operation_tests(cases=, reference=, compare_outputs=,
kernel_tests=[...], op_tests=[...], max_instances=)` records an
`OperationTestsEntry` on the op. An op has at most one; a second declaration
raises. Only the tests listed run.

- `reference(input) -> R` is what the op should compute.
- `compare_outputs(case, actual, expected)` raises unless `actual` is
  acceptable; tolerances come from the case.
- Each case has `make_input(*, seed, device) -> Input`.

**Consequences.** Which checks apply to an op is visible in one place, and an op
that legitimately cannot pass a check (one that syncs with the host, for
example) leaves it out rather than carrying a skip list. Every registered op is
checked to have declared tests.

## 6. Kernel tests and op tests are different things

**Context.** Running the same checks against dispatch as against each kernel
mostly re-tests whichever kernel dispatch picked.

**Decision.** Two kinds of test, with their own signatures
(`aiter/catalog/testing/signatures.py`):

- a **kernel test** `(kernel, instance, case, op_input, verify_outputs)` checks
  one kernel with one instance;
- an **op test** `(op, case, op_input, verify_outputs)` checks what the public
  op adds: routing, its wrapper, and its torch registration.

`verify_outputs(op_input, actual)` is built by the runner from the declaration's
`reference` and `compare_outputs`, so tests that judge an output need neither.
The signatures are protocols with positional-only parameters: the names document
each argument and tests may name their own. A test's function name is its
pytest name.

**Consequences.** The standard tests are plain functions in
`aiter/catalog/testing/standard_tests.py`, with no pytest import. A test skips
by raising `unittest.SkipTest`, and reports a launch-time rejection by raising
`Unsupported`.

## 7. Cases are typed by their own parameter

**Context.** Typing the case as `OperationCase[Input]` rejects every real
`compare_outputs`, which reads fields of its own case class. A type variable
bounded by `OperationCase[Input]` is not allowed: a bound cannot mention another
type variable.

**Decision.** `OperationTestsEntry[Input, R, Case]` with `Case` bounded by
`OperationCase[Any]`, inferred from `cases`. `compare_outputs` and the tests
receive the concrete case type.

**Consequences.** Misspelled case fields and mismatched `compare_outputs`
signatures are type errors. That `make_input` returns the op's `Input` is no
longer checked statically, so every generated test class checks it at run time
for each case (`test_case_makes_the_operation_input`).

## 8. Defining tests and running them are separate

**Decision.** `aiter/catalog/testing/pytest_class.py` turns an
`OperationTestsEntry` into a pytest class (`to_pytest_class(entry, device)`,
or `entry.to_pytest_class(device)`). Kernel tests are parametrized over kernel
and case, op tests over case. For a kernel the runner:

- asks for the instances available for the case's shape on `device`, and skips
  the kernel for that case if there are none;
- takes an evenly spaced `max_instances` of them, with a warning;
- runs the test once per instance, leaving out instances that raise
  `Unsupported` (warning), and skips if none ran;
- prefixes a failure with `kernel[instance]`.

The device is passed in, not read by the runner: the test module creates it
with `DeviceInfo.current()`.

**Consequences.** The registry never imports pytest. The runner can be tested on
the CPU with a fake op and a fake device (`tests/catalog/machinery/`). The
device is read when pytest imports the op's test module, so a machine where it
cannot be detected fails at collection.

## 9. The standard tests and what they rely on

| Test | Kind | Checks |
| --- | --- | --- |
| `matches_reference` | kernel | output passes `verify_outputs` |
| `leaves_inputs_unchanged` | kernel | the input copy is bit-identical after the launch |
| `does_not_alias_inputs` | kernel | no output tensor shares storage with an input tensor |
| `is_deterministic` | kernel | two launches give bit-identical outputs |
| `graph_capture` | kernel | a captured launch replayed on new input values is right for them |
| `on_side_stream` | kernel | a launch on a side stream right after its inputs are written there matches |
| `op_matches_reference` | op | the public op's output passes `verify_outputs` |
| `op_graph_capture` | op | as `graph_capture`, through the public op |
| `fake_matches_real` | op | the fake implementation predicts shape, dtype, device and stride |
| `compiles_without_graph_breaks` | op | `torch.compile(fullgraph=True)` matches eager |

Graph capture warms up twice on a side stream, the second time under
`set_sync_debug_mode("error")` so that a host sync fails with a clear message
before capture. Outputs are filled with NaN before the replay, so a launch that
writes nothing or reads values baked in at capture fails.

A capture that fails leaves the process unusable: the stream stays in capture
mode and torch's capture bookkeeping stays inconsistent. The test ends the
capture through HIP with `ctypes`, deliberately leaks the failed graph object
(destroying it aborts the process), and skips every later capture test in the
process with the reason. This is limited by one process per test file in CI
(decision 10), as it already is for existing op tests.

The side-stream test poisons the input copy before writing the real values on
the side stream behind a delay; without that, the allocator reuses a block that
already holds the values and a launch on the wrong stream reads correct-looking
stale data.

## 10. Where tests live

| Path | Contents |
| --- | --- |
| `aiter/catalog/ops/<op>/tests.py` | the op's test declaration: cases, reference, comparison, opted-in tests |
| `op_tests/catalog/test_<op>.py` | the pytest class for the op: `Test<Op> = <op>_tests.to_pytest_class(DeviceInfo.current())` |
| `tests/catalog/machinery/` | the catalog's own behaviour, on a fake op on the CPU |
| `tests/catalog/registered/` | checks over everything registered: kernels, declared tests, signatures, declarations |

One file per op in `op_tests/catalog/` keeps one process per op in CI, so a
failed capture or a timeout in one op does not affect another. A module that
generated a class for every registered op was considered and not taken for that
reason.

`tests/catalog/` holds only generic checks; nothing in it is specific to an op
or a kernel.

CI: `.github/scripts/aiter_test.sh` and `split_tests.sh` list
`op_tests/catalog/` and `tests/catalog/` recursively, without widening the
depth for other directories, and run both through pytest.

## Open questions

- **Whether and how to standardize tuning.** Each kernel already declares its
  instances as typed configurations and which of them are available for a
  shape on a device, and the runner can launch any of them on an op's input.
  That is most of what a tuner enumerates today, per kernel and in its own
  form. Open: whether the catalog should only expose instances to the existing
  tuners, or provide a common way to time them and record the best per shape;
  where results would live relative to the existing tuned configuration
  files; and how dispatch would consume them without the catalog becoming a
  dispatcher (decision 1).
- **An optional instance, and which instances tests cover.** A kernel's `run`
  could accept `instance=None`, meaning the kernel's own default (its tuned
  lookup or heuristic, as when dispatch calls it). Tests could then cover, per
  case:
  - every instance from `available_instances` (today's behaviour, capped by
    `max_instances`): the widest coverage, and the most launches. It is also
    the least realistic: `available_instances` may over-claim, since it must
    not compile, so testing everything it yields compiles every claimed
    instance, including those that then raise `Unsupported`;
  - only the default, `instance=None`: what users actually run, but no
    coverage of instances the default never picks;
  - the instances that appear in tuned configuration rows for the case's
    shape: what deployments run, at a fraction of the launches, but it ties
    the tests to the tuned files and leaves untuned shapes to the default.

  Kernels with nothing to tune (`NoTunables`) would not change. The choice
  affects the instance type (`Instance | None`) and the tuning question above.
- **Deriving command lines and benchmarks from the declarations.** Most op
  test scripts and benchmarks carry their own argument parser (shape and dtype
  lists, backend choices) and their own timing loop. A catalog declaration
  already has what those are built from: a case dataclass whose fields are the
  parameters, a way to make inputs from a case, the operation and its kernels
  as launchable functions, and each kernel's instances. A command line could be
  derived from the case fields, and a benchmark could time each kernel and
  instance on those inputs, the way the tests already launch them. Open:
  whether that is worth it; whether it replaces the existing scripts or sits
  beside them as one generic entry point; how sweeps (several values per field)
  and the scripts' summary tables would be expressed; and how it relates to the
  tuning question above, which would time the same launches.
- **The side-stream test can miss a wrong-stream launch** after enough earlier
  side-stream tests in the same process. The cause is not known; until it is, a
  pass is weak evidence.
- **Graph-capture recovery** depends on `ctypes` calls into HIP and a leaked
  graph object. The alternative is to keep only the sync check and the skip
  after a failed capture, and accept an abort at exit.
- **Meta-device checks.** Checking `input_to_shape`, instance types and that
  every case is served by some kernel needs inputs; cases that use a
  `torch.Generator` cannot build them on the meta device.
- **Per-field input metadata** (which fields are outputs, which may be
  re-laid out) would allow standard tests for poisoned output buffers and
  strided inputs.
- **Per-kernel opt-out** of a single test (a kernel with a shared workspace and
  the side-stream test, for example) has no mechanism yet.
- **The reference is not cached** across the tests of a case.
