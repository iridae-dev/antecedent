# Antecedent test performance

The largest avoidable costs are repeated native builds in CI, unoptimized local
test binaries, and repeated fixture generation in exhaustive numerical tests.
The changes described here retain the test cases and assertions while reducing
that work. During development, use `bash scripts/test.sh dev` with a package and
test target, or `bash scripts/test.sh python-test` for Python-only edits. Use
`bash scripts/test.sh` for optimized Rust verification and `bash scripts/test.sh
all` for Rust tests, doctests, optional feature tests and Python tests.

## Measured costs

The local measurements below were taken on 14 October 2026 on macOS arm64 with
10 logical CPUs, 64 GiB RAM, Rust 1.85 and CPython 3.13. Compilation and test
execution are reported separately; an execution speedup is not a promise about
the full workspace build.

| Measurement | Before | After | Interpretation |
| --- | ---: | ---: | --- |
| Four-node transport oracle with the ordinary local test profile | 134.96 s | 48.44 s | Source fixture reuse and CPU-aware scheduling reduce execution by about 64%. |
| Four-node transport oracle with optimized test binaries | 15.18 s | 7.36 s | Fixture reuse reduces execution by about 52% with both versions at opt-level 2. |
| Both Python typing tests | 11.91 s | 6.41 s | Private mypy cache reuse reduces the combined calls by about 46%. |
| Benchmark example correctness test | 0.73 s | 0.61 s | Smoke mode retains all workload/format combinations with fewer timing samples. |
| Optimized transport integration target | — | 19 passing tests in 7.98 s | All ordinary tests in this target execute. |
| First optimized transport build and run | — | 104.24 s total | Changing optimization settings costs a rebuild; the test phase accounts for 8.17 s. |

The full Python baseline collected 5,478 tests in 6.51 s and executed in
175.76 s: 5,391 passed, 55 skipped and 32 failed. Thirty-one failures concerned
calibration records that no longer attest the modified checkout; one concerned
an older native reason-code registry than the edited registry. These baseline
failures prevent treating this checkout as a green full-suite benchmark.
The Python suite already used a release native extension.

These are individual run measurements. The four-node comparisons use the original
and changed test sources with the same dependencies, once at the ordinary
local profile and once with opt-level 2. Combining the optimized runner and
fixture change takes that test from 134.96 s to 7.36 s; CI already used
opt-level 2, so its relevant comparison is 15.18 s to 7.36 s. The first optimized
run took 7.54 s for this test and 8.17 s for the target, consistent with the
final verification run.

The [successful CI run from 11 October 2026](https://github.com/iridae-dev/antecedent/actions/runs/38133678023)
provides the build baseline:

| CI work | Recorded time |
| --- | ---: |
| Native extension builds across 16 jobs | 9,203 runner-seconds, or 153.4 runner-minutes |
| One native extension build | 428–669 s; median 589 s |
| Slowest job, Domain transport-contracts | 1,543 s, including 665 s native build and 803 s gate execution |
| Ubuntu Rust remaining packages | 615 s in the test step, including build, tests and doctests |
| Ubuntu Rust facade | 705 s in the test step, including build, tests and doctests |
| Python coverage job | 539 s native build and 275 s pytest |

Runner-seconds sum work across machines. They do not equal elapsed time when
jobs run concurrently. The shared build removes 15 native compilations per
run; the resulting CI wall time needs a run of the updated workflow.

## Selections

`scripts/test.sh` names each selection explicitly; none silently substitutes
for another, and an empty selection fails.

| Mode | Selects |
| --- | --- |
| `rust` (default), `docs`, `dev` | Workspace nextest run, doctests, or a focused forwarded selection. |
| `rust-software` | Ordinary workspace assertions with registered measured-record acceptance assertions explicitly deselected (they await calibration). |
| `rust-calibration-acceptance` | Only those retained acceptance assertions; no calibration is performed. |
| `integration` | Integration targets, without library unit tests. |
| `scientific` | Exact scientific assertions cited by the evidence registries, via `scripts/evidence_audit.py run-scientific`. |
| `scripts` | Discovered script unit tests in their locked environment; skips, empty discovery and failures fail. |
| `python`, `python-test` | Full pytest (with or without sync and native rebuild). |
| `python-software` | pytest with `-m 'not requires_calibration'`. |
| `python-calibration-acceptance` | pytest with `-m requires_calibration`, after branch calibration. |

Both pytest modes first check the native extension identity
(`scripts/native_build_identity.py check`). CI's coverage-floor pytest step runs
the whole suite (software and acceptance) because the calibration attestation
runs first in the same workflow.

Most integration files are compiled into a few consolidated `suite_NN` targets
per crate (`scripts/consolidate_tests.py`); the files stay where they are, are
included with `#[path]`, and are listed in the crate's `tests/suite_map.txt`.
Calibration harnesses and any target a script names with `--test` stay
separate. `scripts/check_test_targets.py` fails when a test file is not
compiled, and `scripts/check_v3_test_registry.py` fails when a registry row
names a missing file or a stale suite-map line.

## Rust fixture reuse

`transport_exact_scm.rs` formerly generated the same source population laws
for every target selection. Source laws depend on the graph and parameter
values; target selections change only target mechanisms. The test now builds
the 27 three-node or 81 four-node source laws once per graph and parameter
combination and reuses them across the eight or sixteen selections.

Each selection still receives a fresh target law and a freshly enumerated
target intervention truth. The independent oracle's summation order,
probability calculations and tolerances remain unchanged. Immutable source
law clones can also reuse their marginal caches. The cache is local to one
graph family and parameter combination, so memory does not accumulate across
the exhaustive sweep.

The three-node sweep also shares its evidence catalog and binds each proof
once per selection. Both sweep schedulers distribute directed masks among at
most the available CPUs. The four-node oracle previously always spawned eight
workers, including on smaller CI machines. Nextest continues to reserve the
machine for these internally parallel sweeps.

Visit counters assert all 512 three-node and 65,536 four-node diagrams are
examined. The parameter admissibility rules, negative-witness checks, treatment
positions, joint interventions and numerical comparisons remain in place.
The final four-node run checked 123,480 graph-selection-parameter cases and
246,960 intervention comparisons.

## Local commands

Install nextest once using the [official installation instructions](https://nexte.st/docs/installation/).
From the repository root:

```bash
# Workspace Rust unit and integration tests, including CI calibration features.
bash scripts/test.sh

# Include doctests, optional learner/GP features and the Python suite.
bash scripts/test.sh all

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
include required features for the target you are changing. `all` also executes
the separate learner and Gaussian-process feature configurations. It is the
test suite, not the release gate: run `scripts/gate_release.sh` and the standing
release procedure when those checks are required.

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
`maturin develop`, then run pytest with synchronization disabled. The existing
`tool.maturin.profile = "release"` already applied to editable builds; native
artifact reuse does not depend on correcting an unoptimized Python build.
See [Maturin configuration](https://www.maturin.rs/config).

Ignored statistical measurements remain explicit through
`scripts/measure_calibration.sh`. No faster command silently adds ignored
calibration experiments to the ordinary suite or substitutes smoke results for
accepted coverage records.

## CI build strategy

The `python-native` job builds one release extension for Linux x86_64 and
CPython 3.12 after the calibration attestation. Sixteen Python domain sections
and the coverage/lint job consume this run's artifact. Rust-only domain
sections start independently while the extension builds.

Consumers verify the commit, Python ABI, operating system, architecture,
package version, filename and checksum. The helper installs the extension in
the checkout package and writes an editable-style `.pth` file into the venv.
It then verifies the exact native import path in a fresh interpreter outside
the checkout, matching tests that launch child processes from temporary
directories. Python source coverage still covers the checkout package.

All 27 domain section names (16 Python, 11 Rust-only) and the required `Domain gates` check remain.
The aggregate check requires both the Python and Rust domain matrices to
succeed. The 16 platform/interpreter wheel combinations still build and test
their own installed wheels. Sharing the Linux source-tree extension does not
replace packaged-wheel testing. Wheel test environments and the local release
candidate environment also install mypy, which the two typing tests require.

Domain Rust tests now use the same test optimization and debug-info settings
as the main Rust jobs. The named-test facade listing includes
`calibration-internal`, matching the preceding facade test build and avoiding
another feature-specific set of binaries. Compiler caches remain enabled on
the jobs that already use them; cache hits do not replace same-run artifact
reuse.

## Timing reports and verification

Nextest's `ci` profile runs all selected ordinary tests without retries or a
default filter and writes `target/nextest/ci/junit.xml`. CI retains separate
reports for the workspace shards and optional feature suites, including
failed runs. The local runner copies each suite's report into a separate file
under `target/test-performance/`, so `all` retains workspace and optional
feature results. Cargo HTML timings separate compilation from execution. Python
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

The transport target passes all 19 tests. The changed Python example and typing
files pass 83 checks; six notebook executions were excluded from that focused
run because the local environment lacks matplotlib. Native artifact integrity
tests, a real artifact install/import roundtrip in an isolated venv, named-test
self-tests, Rust feature parity, workflow matrix preservation and shell/YAML
validation also pass. The performance fixes do not change production algorithms or calibration records.

Compare the next updated CI run against the baseline by job and step. Track
the native producer's build time, domain gate execution after the profile
change, Rust execution from JUnit, artifact transfer time, runner-seconds and
the final required-check completion time. These distinguish saved compilation
from a bottleneck that has simply moved elsewhere.

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
from Cargo timings under matching cache conditions.

The one comparison already on file,
`parity/evidence_audit/ordinary_rust_comparison.json`, is not controlled: the
selections, checkout and Cargo settings differ, and its execution time rose
from 744.162 to 903.397 seconds. No workspace speedup is claimed.

### Paired runs: the test-binary consolidation

`parity/evidence_audit/paired_rust_cost_runs.json` records three cold,
sequential ordinary runs on one 10-CPU host with identical settings
(`CARGO_PROFILE_TEST_OPT_LEVEL=2`, line-table debug info, no incremental
compilation), the same workspace features and a fresh target directory per
revision. Build and execution are timed separately.

| Revision | Test binaries | Tests | Build wall / CPU | Execution (nextest) | Failures |
| --- | ---: | ---: | ---: | ---: | ---: |
| before the consolidation (`99a9c55b0`) | 640 | 8,423 | 796 s / 5,795 s | 86 s | 12 |
| after it (`d69837638`) | 231 | 8,436 | 673 s / 4,807 s | 73 s | 0 |
| current ordinary run (`fa3ecc207`) | 480 | 13,486 | 1,160 s / 9,175 s | 464 s | 13 |

Only the first pair is comparable: same code and settings, with the
consolidation as the intended difference. It is indicative only. It shows about
15% less build wall time, 17% less build CPU time and 15% less execution time,
at about 19% higher peak build memory, which is within the run-to-run variation
to expect from a single cold build. Each revision was run once, so there is no variance
estimate; interactive applications shared the host; and the earlier revision had
12 failing tests (and 13 fewer tests) that the later one does not. The current
run has far more tests and binaries, so it shows growth, not an improvement. Its
13 failures are the calibration-pending measured-record tests that
`rust-software` deselects. The comparison is not called controlled: the
checkouts differ by design.

CI runs these controls in the `script unit tests` and `Test-suite controls`
steps of the Python lint job (`scripts/test_evidence_audit.py` and
`scripts/test_cost_report.py` through script discovery; the assertion scan and
registry self-tests, and the source-only evidence audit report, directly). The
Domain `inventory` section runs the registry check itself, and the provenance
schema gate runs in every release-gate section. The required checks fail, with
fault controls in `scripts/test_evidence_audit.py` and
`scripts/test_cost_report.py`, for: a stale native extension, a deleted one
with a fresh stamp, or one built for another interpreter ABI,
an empty scientific or calibration selection, an unregistered or undiscovered
calibration-acceptance test, a script test that is failed, skipped, expected to
fail, deselected unknown or discovered as nothing, a Python session that skips
more than 5% of its tests, and a Python selection that collects nothing.

## Further performance work

The remaining opportunities should be evaluated from durations before
changing test topology:

1. **Standalone Python imports.** The 130 import checks took 48.27 s locally.
   Each module needs a fresh interpreter to preserve the import-order oracle.
   Bounded concurrent subprocess launches could reduce their elapsed time;
   verify process isolation and preserve one result per module.
2. **Python worker limits.** The suite is sequential. Start an xdist experiment
   with two workers, measure wall time and memory, and audit native thread
   pools before increasing the count. Automatic CPU-count workers can multiply
   the native library's own parallelism. No xdist dependency or new test
   sharding policy is introduced here.
3. **Rust gate build topology.** Some evidence gates resolve and build cited
   targets with different feature sets. Use the new timing artifacts to decide
   whether a shared nextest archive or consolidated feature build saves more
   than artifact transfer and lost parallelism cost. Preserve citation and
   no-test-match checks.
4. **MAG numerical fixtures.** The exhaustive `mag_id_bruteforce` oracle already
   caches identification results and observed laws. Reusing compiled exact
   plans and free-variable discovery per MAG is another candidate, with the
   same numeric worlds and assertions retained.
5. **Calibration scheduling.** Statistical calibration already uses optimized
   release builds, explicit ignored suites, narrowed owed groups and scheduled
   workers. Optimize its CPU allocation separately and retain the measured
   replicate counts, sample-size grid and attestation rules.

## Local development build strategy

For the edit, build, test loop, reduce compilation and linking before adding
more test workers. After consolidation the workspace has 449 integration-test binaries;
197 belong to the facade. Running a small nextest name filter against the
facade can still build all its targets. Select the Cargo target first, then
apply a test-name filter within it:

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

Nineteen facade targets that use only the known-truth fixtures now import a
narrow helper wrapper. Previously each included all ten common helper modules,
including unrelated calibration and data generators. Each now includes only
the unchanged fixtures and their required calibration seed helpers. This cuts
the included helper text from 4,168 to 2,437 lines per target and removes
invalidation from eight unrelated helper modules. Test files, function names,
data generators and citation paths remain unchanged. This is a reduction in
compiler input and dependencies; it is not a measured percentage reduction in
build time.

The next broader changes, in priority order, are:

| Change | Local benefit | Required validation |
| --- | --- | --- |
| Dedicated native development build with incremental compilation and LTO off | Avoid repeated whole-graph release link optimization after a Rust edit while retaining optimized numerical code. | Compare cold build, no-op build and a real edit/rebuild against the release path, then verify Python results and runtime. |
| Measure the consolidated `suite_NN` targets | The grouping is in place; one paired cold run measured about 15% less build time (see the paired-runs table). | Compare library-edit and test-file-edit rebuilds before and after under matching conditions with `scripts/cost_report.py`. |
| Exclude purely ignored calibration targets from ordinary builds | A conservative source inventory found 27 such integration targets, containing 112 ignored tests. These still cost compilation and links today. | Confirm actual collection, add an explicit measurement feature, and have the calibration driver enable it. |
| Reduce dependency fan-out in core and artifact/facade layers | Limit how far a common source edit propagates. | Separate stable data/types from frequently edited implementations without weakening public API or provenance checks. |

For the native profile experiment, keep all current features and release
optimization. Use a separate target directory so repeated experiments do not
replace the normal release build cache:

```bash
cd python
CARGO_TARGET_DIR=../target/native-dev \
  CARGO_INCREMENTAL=1 \
  CARGO_PROFILE_RELEASE_LTO=off \
  uv run --no-sync maturin develop --release --locked
```

This candidate removes LTO and enables incremental compilation without changing
Cargo.toml. The first build creates a new cache. Benchmark warm builds after
representative edits before adopting it as the default; full release and
calibration validation retain their existing settings. Cargo documents the
compile/runtime tradeoffs in [profiles](https://doc.rust-lang.org/cargo/reference/profiles.html).

Local sccache is secondary to these changes: it cannot cache linker invocations,
and incremental Rust compilation is not cacheable through sccache. Disabling
incremental compilation to improve compiler-cache hit rates can hurt successive
local edits. See [sccache Rust support](https://github.com/mozilla/sccache/blob/main/docs/Rust.md).

Support generation now also preserves modification times when its output bytes
are unchanged. Previously regenerating an identical support matrix rewrote
`reason_codes_data.rs` and the embedded support/coverage JSON files. Cargo's
timestamp-based invalidation could then rebuild core and downstream crates
even though no generated content changed. The generator still recomputes live
attestation and writes changed outputs; it simply avoids no-op writes. Eight
snapshot/write checks, including the existing attestation guards, pass.

Validation of the focused development changes includes all nineteen narrowed
facade targets passing `cargo check`, a real focused Rust `dev` run, and 29
Python tests through `python-test` without a native rebuild. The native profile
candidate remains a benchmark proposal, and the consolidated targets' build benefit is measured once, with the caveats above.


