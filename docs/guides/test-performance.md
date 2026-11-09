# Antecedent test performance

Pick the narrowest command that covers your edit. During development, use
`bash scripts/test.sh dev` with a package and test target, or
`bash scripts/test.sh python-test` for Python-only edits. Use
`bash scripts/test.sh engineering` for every engineering test (script tests, Rust
tests, doctests, optional feature tests and Python tests), and plain
`bash scripts/test.sh` on a development machine to run that and then earn the
scientific and calibration records your change owes.

## Selections

`scripts/test.sh` names each selection explicitly; none silently substitutes
for another, and an empty selection fails.

| Mode | Selects |
| --- | --- |
| `engineering` | Script tests, workspace nextest run, doctests, optional-feature suites and the Python suite, excluding registered scientific assertions and calibration tests. |
| `rust`, `docs`, `dev` | Workspace nextest run, doctests, or a focused selection. The nextest default filter (`python3 scripts/evidence_audit.py rust-engineering-filter --check`) excludes registered scientific assertions and calibration tests. |
| (none), `all` | The development-machine run: `engineering`, then the owed `scientific` assertions, then the owed `calibration` measurements; stops at the first failure. See [Earning and checking the records](scientific-verification.md#earning-and-checking-the-records). |
| `rust-calibration-acceptance` | Only those retained acceptance assertions; no calibration is performed. |
| `integration` | Integration targets, without library unit tests. |
| `scientific` | Only the owed scientific assertions (cited by the evidence registries or listed in `parity/scientific_tests.toml`), one nextest run per crate and one pytest batch, each batch recorded when it finishes (`scripts/scientific_records.py run`). `--only PATTERN` limits it to matching owed rows; `--all` (manual) reruns every one. No engineering run. |
| `calibration` | Only the owed coverage records (`scripts/measure_calibration.sh --owed`); `--only PATTERN` narrows by group, record id/test or route; `--all` (manual) measures every group. |
| `scripts` | Discovered script unit tests in their locked environment; skips, empty discovery and failures fail. |
| `python`, `python-test` | pytest without scientific, measured-record acceptance and calibration-simulation assertions (with or without sync and native rebuild). |
| `python-calibration-acceptance` | pytest with `-m requires_calibration`; it reads the attesting coverage records, so run it when they are current. |
| `slow` | The slow correctness sweeps listed in `parity/slow_correctness_tests.toml` (see [Lanes](#lanes)). |

The pytest modes first check the native extension identity
(`scripts/native_build_identity.py check`). CI's coverage-floor and wheel tests
exclude the measured-record acceptance marker. The explicit acceptance lane
retains those tests for use after calibration. Plain pytest uses the same
default; `-m requires_calibration` selects the acceptance tests explicitly.
For Rust, the default nextest filter is
`python3 scripts/evidence_audit.py rust-engineering-filter`, which is derived
from `parity/test_registry.toml`. After registering a scientific or
measured-record assertion, regenerate it with `--write`; the static gate runs
`--check` and fails while `.config/nextest.toml` is stale.

Most integration files are compiled into a few consolidated `suite_NN` targets
per crate (`scripts/consolidate_tests.py`); the files stay where they are, are
included with `#[path]`, and are listed in the crate's `tests/suite_map.txt`.
Calibration harnesses and any target a script names with `--test` stay
separate. `scripts/check_test_targets.py` fails when a test file is not
compiled, and `scripts/check_test_registry.py` fails when a registry row
names a missing file or a stale suite-map line.

## Lanes

Software, scientific-evidence and calibration-acceptance assertions run
separately. A test may be in the software and scientific lanes at once; every
ordinary test is in the software lane, so none falls between lanes.

| Lane | Command | Contents |
| --- | --- | --- |
| Engineering | `bash scripts/test.sh engineering` (Rust tests through `rust`, pytest through `python`, script tests through `scripts`) | Every non-ignored workspace test except the registered scientific assertions and the registered calibration tests. |
| Scientific evidence | `bash scripts/test.sh scientific` | Only the assertions that `parity/*.toml` claims cite as known-truth or external-oracle evidence (`scientific` role), in Rust and Python. Discovery proves each cited test exists and is not ignored; only the scientific lane runs them, on a development machine. |
| Calibration acceptance | `bash scripts/test.sh rust-calibration-acceptance`; `bash scripts/test.sh python-calibration-acceptance` | Tests registered under `calibration_acceptance_assertions` in `parity/test_registry.toml`, or marked `requires_calibration`. They read attesting coverage records and fail while those records are owed. |
| Calibration measurement | `bash scripts/test.sh calibration` (`scripts/measure_calibration.sh --owed`) | `#[ignore]`d replicate-heavy coverage, precision and measurement diagnostics. No ordinary lane runs them. |
| Slow correctness | `bash scripts/test.sh slow` | `#[ignore]`d exhaustive enumerations and heavy acceptance fits that are too slow for the ordinary lanes (for example `mag_id_bruteforce::exhaustive_four_observed_two_latent` and the ring-with-triangles sampler references in `graph_posterior_wide`). Selected by exact name from `parity/slow_correctness_tests.toml`, which gives each a reason and an expected runtime. Manual or nightly; not part of ordinary CI. |
| Integration | `bash scripts/test.sh integration` | All integration targets, without library unit tests. |

Every `#[ignore]`d Rust test is in exactly one class, and
`python3 scripts/check_ignored_tests.py` (static gate) fails
otherwise: calibration or measurement (the calibration scripts own them), fixture
writers (`writes ...`), timing probes (`timing harness ...`), the slow correctness
lane above, or an explicit `[[other]]` exemption in `parity/slow_correctness_tests.toml`.
An ignored test needs a written reason; a new one that fits no class, or a registry row whose
test is gone or no longer ignored, is a failure, so nothing falls out of every lane.
`bash scripts/test.sh slow` repeats that check against a nextest listing before it
runs the selection with `--run-ignored only`; run it before a release or nightly.

The scientific lane builds with the workspace feature set, so it shares its
build with `rust`. Doctests (`docs`) and the optional learner and
Gaussian-process feature suites run in `engineering`.

## Scheduling long tests

A handful of long tests bound the wall time of the Rust suite, so shortening it
means shortening or overlapping those tests, not tuning the many short ones.
`.config/nextest.toml` gives the longest tests a scheduling `priority`, so they
start first instead of after the pool has drained. Priority changes order only;
every test still runs once with its assertions.

Nextest runs one test per core, so a test that spreads independent cases over
scoped threads uses idle cores at the tail of the run; it costs about the same
CPU time (somewhat more through cache contention) and shortens the wall time.
Read outcomes back in a fixed order so the assertions do not depend on thread
timing. A shared fixture across tests (`OnceLock`) does not help under nextest,
which gives each test its own process. Never remove or weaken an assertion,
tolerance or seed to make a test faster.

## Rust fixture reuse

Exhaustive numerical tests can reuse fixtures that the swept parameter does not
change. `transport_exact_scm.rs` builds the source population laws once per
graph and parameter combination and reuses them across the target selections,
because source laws depend only on the graph and parameter values. Each
selection still receives a fresh target law and a freshly enumerated target
intervention truth, and the independent oracle's summation order, probability
calculations and tolerances are unchanged. The cache is local to one graph
family and parameter combination, so memory does not accumulate across the
sweep. The sweep schedulers spawn no more workers than there are CPUs, and
visit counters assert that all 512 three-node and 65,536 four-node diagrams are
examined.

## Local commands

Install nextest once using the [official installation instructions](https://nexte.st/docs/installation/).
From the repository root:

```bash
# Every engineering test: script, Rust, doctest, optional-feature and Python.
bash scripts/test.sh engineering

# Engineering, then the scientific assertions and calibration records that are owed.
bash scripts/test.sh

# One owed scientific test while iterating.
bash scripts/test.sh scientific --only graph_posterior

# Reuse the same runner settings for a focused development loop.
bash scripts/test.sh rust -p antecedent-identify --test transport_exact_scm
bash scripts/test.sh rust -p antecedent --test response_facade -E 'test(response)'
bash scripts/test.sh docs -p antecedent-identify

# Rebuild the native extension from current sources, then profile pytest.
bash scripts/test.sh python --durations=30

# Pure Python edits can reuse an extension already built from current Rust sources.
ANTECEDENT_SKIP_NATIVE_BUILD=1 bash scripts/test.sh python -q --durations=30
```

With no Rust selection arguments, the runner uses the whole workspace and the
calibration feature set. With arguments, it forwards your exact selection;
include required features for the target you are changing. `engineering` also
executes the separate learner and Gaussian-process feature configurations. The
runner is the test suite, not the release gate: run `scripts/gate_release.sh` and
the [release procedure](../development.md#releases) when those checks are required.

The runner sets `CARGO_PROFILE_TEST_OPT_LEVEL=2` and line-table debug information
unless you have already supplied overrides. Cargo.toml stays unchanged, so
ordinary `cargo check`, development builds and the calibration surface retain
their existing settings. For a compile-heavy loop on a small test:

```bash
CARGO_PROFILE_TEST_OPT_LEVEL=0 bash scripts/test.sh rust -p antecedent-core --lib
```

Switching optimization, features, toolchain, RUSTFLAGS or target changes Cargo
fingerprints and can trigger recompilation. Keep settings stable while
measuring repeated runs. Release LTO remains reserved for release builds and
statistical calibration; the ordinary optimized test loop does not enable it.

Python runs synchronize the locked dev environment, use explicit release
`maturin develop`, then run pytest with synchronization disabled. Editable
builds also use the release profile (`tool.maturin.profile = "release"`; see
[native extension builds](../development.md#native-extension-builds)).
See [Maturin configuration](https://www.maturin.rs/config).

Ignored statistical measurements remain explicit through
`scripts/measure_calibration.sh`. No faster command silently adds ignored
calibration experiments to the ordinary suite or substitutes smoke results for
measured coverage records.

## Local development build strategy

For the edit, build, test loop, reduce compilation and linking before adding
more test workers. The facade has many integration-test binaries, and running a
small nextest name filter against it can still build all of them. Select the
Cargo target first, then apply a test-name filter within it:

```bash
# Cheap compilation for a focused Rust edit. Existing numerical package
# overrides keep faer, stats, kernels and estimate optimized.
bash scripts/test.sh dev -p antecedent --test response_facade
bash scripts/test.sh dev -p antecedent-core --lib

# Use optimized execution for a numerically expensive target.
bash scripts/test.sh rust -p antecedent-identify --test transport_exact_scm

# Python/test-only edits use the current native extension directly.
bash scripts/test.sh python-test tests/test_response_api.py -q
```

`dev` defaults to test opt-level 0 and requires an explicit selection. Caller
optimization overrides still apply. The optimized `rust` mode is appropriate
for larger numerical runs; it can cost more compilation during active editing.
`python-test` skips both environment synchronization and native compilation.
After changing Rust, rebuild the extension explicitly before using that mode.

Support generation keeps the bytes and modification times of outputs it would
not change. Regenerating an identical support matrix therefore does not rewrite
`reason_codes_data.rs` or the embedded support and coverage JSON files, so
Cargo's timestamp-based invalidation does not rebuild core and downstream
crates.

Local sccache is secondary to these choices: it cannot cache linker invocations,
and incremental Rust compilation is not cacheable through sccache. Disabling
incremental compilation to improve compiler-cache hit rates can hurt successive
local edits. See [sccache Rust support](https://github.com/mozilla/sccache/blob/main/docs/Rust.md).

## CI build strategy

The `python-native` job builds one release extension for Linux x86_64 and
CPython 3.12. The `cited-tests` shards and the coverage/lint job
consume this run's artifact. The static, public-API, bench and `rust` jobs start
independently while the extension builds.

Consumers verify the commit, Python ABI, operating system, architecture,
package version, filename and checksum. The helper installs the extension in
the checkout package and writes an editable-style `.pth` file into the venv.
It then verifies the exact native import path in a fresh interpreter outside
the checkout, matching tests that launch child processes from temporary
directories. Python source coverage still covers the checkout package.

CI runs in four tiers: `static` (source
reading only, seconds), the build jobs (`public-api`, `cited-tests`, `benches`, `features`,
`python-native`, `deny`), the engineering tests (`rust`, `development`, `python-lint`,
`python-wheels`) and one `scientific` job; see [CI vs local gates](../development.md#ci-vs-local-gates).
CI executes no scientific assertion and no calibration
measurement: the `scientific` job builds nothing and only checks that every record in
`parity/scientific_records.toml` and `parity/coverage_records.toml` is current. Those records are
earned on development machines ([Earning and checking the records](scientific-verification.md#earning-and-checking-the-records)); the engineering jobs deselect the scientific tests. The
16 platform/interpreter wheel combinations still build and test
their own installed wheels. Sharing the Linux source-tree extension does not
replace packaged-wheel testing. Wheel test environments and the local release
candidate environment also install mypy, which the two typing tests require.

The cited-tests job uses the same test optimization and debug-info settings
as the main Rust jobs. Compiler caches remain enabled on
the jobs that already use them; cache hits do not replace same-run artifact
reuse.

## Timing reports and verification

Nextest's `ci` profile runs all selected ordinary tests without retries or a
default filter and writes `target/nextest/ci/junit.xml`. CI retains separate
reports for the workspace shards and optional feature suites, including
failed runs. The local runner copies each suite's report into a separate file
under `target/test-performance/`, so a run that executes several suites keeps
every result. Cargo HTML timings separate compilation from execution. Python
coverage and wheel runs print their thirty slowest durations.

For an isolated local comparison:

```bash
# First run includes any required optimized compilation.
bash scripts/test.sh rust -p antecedent-identify --features test-util \
  --test transport_exact_scm --offline

# Repeat unchanged to measure a warm build and execution separately.
bash scripts/test.sh rust -p antecedent-identify --features test-util \
  --test transport_exact_scm --offline

# Display oracle visit and numerical comparison counts.
bash scripts/test.sh rust -p antecedent-identify --features test-util \
  --test transport_exact_scm --offline --success-output immediate
```

When comparing CI runs, compare by job and step: the native producer's build
time, gate execution, Rust execution from JUnit, artifact transfer time,
runner-seconds and the final required-check completion time. These distinguish
saved compilation from a bottleneck that has simply moved elsewhere.

## Cost report

Every `scripts/test.sh` run writes a receipt under `target/test-performance/`
(command, settings, host, checkout identity, wall, user and system time, peak
child memory) and copies each nextest JUnit report beside it. To compare two
runs:

```bash
python3 scripts/cost_report.py BEFORE.json AFTER.json \
  --before-junit BEFORE.xml --after-junit AFTER.xml --output cost.json
```

The report prints each measurement as a signed difference (after minus before)
and calls the comparison controlled only if all of these hold: identical
command, settings, platform and CPU count; the same commit with no uncommitted
inputs and no change during either run; both runs exited 0; JUnit reports for
both with equal selected-case counts and no skipped or failed cases. Otherwise
it lists the reasons and states that the differences establish neither an
improvement nor a regression. Values that were not recorded are `null`; none is
estimated. A receipt's wall time includes discovery, compilation and lock waits;
only the JUnit case durations isolate execution, and build time must be taken
from Cargo timings under matching cache conditions. A single cold run per
revision gives no variance estimate, so treat one paired comparison as
indicative only.

CI runs these controls in the `script unit tests` and `Test-suite controls`
steps of the Python lint job (`scripts/test_evidence_audit.py` and
`scripts/test_cost_report.py` through script discovery; the assertion scan and
registry self-tests, and the source-only evidence audit report, directly). The
`static` job runs the registry check itself, and the provenance
schema gate runs in it. The required checks fail, with
fault controls in `scripts/test_evidence_audit.py` and
`scripts/test_cost_report.py`, for: a stale native extension, a deleted one
with a fresh stamp, or one built for another interpreter ABI,
an empty scientific or calibration selection, an unregistered or undiscovered
calibration-acceptance test, a script test that is failed, skipped, expected to
fail, deselected unknown or discovered as nothing, a Python session that skips
more than 5% of its tests, and a Python selection that collects nothing.
