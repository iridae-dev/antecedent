# Development

[How Antecedent checks its scientific methods](guides/scientific-verification.md) explains how provenance, independent references, executing assertions and calibration support a claim. This page describes the development and measurement procedures that maintain that evidence.

## Documentation preview and checks

In a separate checkout or worktree with the matching Antecedent installed:

```bash
python -m pip install -r requirements-docs.txt
python scripts/generate_example_catalog.py --check
python scripts/generate_doc_release_links.py --check
python scripts/check_docs_experience.py
python scripts/render_notebooks.py
mkdocs build --strict
python scripts/build_python_api_docs.py site/python
mkdocs serve
```

Notebook rendering executes setup cells and code in real isolated kernels. It produces HTML in `target/doc-notebook-previews`; source notebooks keep no saved outputs. Avoid running calibration or release-registry generators just to preview prose. The publication manifest excludes branch audits and one-off measurements while retaining evidence source paths. See [documentation sources](https://github.com/iridae-dev/antecedent/blob/v2.4.0/docs/README.md) for editorial conventions.


Product name: **Antecedent**. Python distribution/import: `antecedent`. Rust
day-1 facade: `antecedent` (`cargo add antecedent`). Supporting crates are
`antecedent-*` on crates.io.

## CI vs local gates

For optimized local test commands, measured bottlenecks and the CI build
strategy, see [test performance](guides/test-performance.md).

GitHub Actions CI (`ci.yml`) runs the gate in tiers. Each CI job is one section of
`scripts/gate_release.sh`, so a local run and CI cannot drift; cheap checks come first and the
record-currency job last. CI executes no scientific assertion and no calibration measurement:

0. **`static`** — source reading only, no build and no test: DCO, rustfmt, lint allows, test
   targets, crate ownership, the test registry and lane filters, parity inventories, provenance,
   capability records, the one support count (`scripts/gate_support.py`), the release inventory
   (`scripts/check_release_inventory.py`), bounds, release claims, links and
   every generated file being current, and that every scientific and calibration record is current
   (`scientific_records.py status --check`, `calibration_status.py --check`).
1. **Builds** — compile, list or lint; no test runs. **`features`** (Clippy and the feature
   combinations crates.io users get: default features, each optional feature alone,
   `--no-default-features`), **`public-api`** (public items match `parity/public_api/*.txt`),
   **`cited-tests`** (eight shards; every cited test is a compiled, non-ignored test),
   **`benches`** (Criterion smoke, skipped on a pull request that touches no bench),
   **`python-native`** (the extension the Python jobs share), **`deny`** (`cargo deny check` and
   the crates.io publish dry-run) and **`fuzz`**.
2. **Engineering tests** — tests of the software with no scientific claim. **`rust`** (the
   engineering nextest selection and doctests; the facade and remaining packages run in parallel on
   Ubuntu, both also on macOS after merge), **`development`** (the development build that runs
   `not_supported` routes), **`python-lint`** (Ruff, mypy, script tests, then pytest without the
   scientific and calibration markers, with an 85% coverage floor) and **`python-wheels`** (builds
   and tests the supported wheel matrix). The optional learner and Gaussian-process feature tests
   run in `features` on every pull request.
3. **`scientific`** — one job, `bash scripts/gate_release.sh --section scientific`: the same two
   currency checks as `static`, alone and without a build. It fails while any scientific or
   calibration record is owed; the records a change owes are earned on a development machine
   (`bash scripts/test.sh`) and committed with it.
4. **Release candidate** — `scripts/gate_release_candidate.sh`, run locally (below): the CI-run
   check, the record-currency checks, scale cases and the installed wheel. It re-runs no
   scientific assertion.

Running `scripts/gate_release.sh` without `--section` executes tiers 0 to 3 locally in order.

**Scientific tests and statistical calibration are earned on a development machine, never on
GitHub.** CI only checks that the committed records are current. See [Earning and checking the records](guides/scientific-verification.md#earning-and-checking-the-records) for the
model and every lane command. Criterion benchmark smokes check
execution, not timing regressions. `cargo deny` runs inside `gate_release.sh` only when `cargo-deny`
is on PATH, which it is not in CI's `benches` job (the `deny` job owns it there).

## Gates (local)

Each gate below is also a step of `scripts/gate_release.sh`. Run one on its own when a change might
break something unintended:

```bash
python3 scripts/check_parity_inventories.py   # every parity inventory: done rows have evidence, exit artifacts exist
bash scripts/gate_upstream_names.sh
bash scripts/gate_composition.sh   # composition consuming contract/claim tests resolve to real tests
bash scripts/gate_transport.sh     # transport stage contracts; every fixture family has a positive and a counterexample
bash scripts/gate_gcm_conformance.sh   # policies, mechanism selection, falsification, noise replay, intrinsic influence: every clause cites a test
bash scripts/gate_metadata_consistency.sh
bash scripts/gate_evidence_reachability.sh
python3 scripts/gate_support.py       # the one support rule over every unit, and the one support count
python3 scripts/gate_support.py --cited-tests   # every cited test is compiled and not ignored (builds test binaries)
python3 scripts/gate_not_supported_sites.py   # `not_supported` is built only in require_supported
python3 scripts/check_release_inventory.py   # acceptance manifest, counterparts, scope, children, routes, specifications
bash scripts/gate_checked_execution.sh   # every supported estimator cites a test that executes from its retained checked plan
bash scripts/gate_docs_support_matrix.sh
bash scripts/gate_public_api.sh   # published crates' public items match parity/public_api/*.txt
bash scripts/test.sh                  # engineering, then owed scientific and owed calibration
bash scripts/measure_calibration.sh   # coverage / CI Type I: measure what is not current, collect, print the status
bash scripts/gate_release.sh       # tiers 0 to 3 in order
bash scripts/gate_python_lint.sh   # local equivalent of the CI lint/type checks
```

## Public API snapshots

`scripts/gate_public_api.sh` lists every published crate's public items (paths,
signatures, fields, variants, trait impls, `#[doc(hidden)]` items included) from
rustdoc JSON on the pinned toolchain (`RUSTC_BOOTSTRAP=1`, default features) and
compares the listing with the committed `parity/public_api/<crate>.txt`. Any
difference fails the gate with a line diff. An intended API change is made
deliberate by regenerating the snapshots (`bash scripts/gate_public_api.sh
--write`) and committing the diff, which a reviewer then reads as the API
change. The gate runs in the `public-api` section of `gate_release.sh`. Keep
cross-crate plumbing that is not a supported API out of the snapshot by making
it `pub(crate)`, or, when another workspace crate or a fixture needs it, behind
a `test-util`-style feature (see `antecedent-identify`) rather than
`#[doc(hidden)] pub`.

## Coverage records

`parity/coverage_records.toml` holds only measurements. Each row is a
`calibration-record` line a coverage test emitted through
`CoverageTally::for_record` (`crates/antecedent/tests/common/calibration.rs`),
keyed with the construction the runtime reports for the execution it scored,
and stamped with the commit it was measured at. The runtime claim and the
independent consumer both match a reported interval against these rows through
`antecedent_io::calibration`, so a row that no test emitted cannot make an
interval `calibrated`.

### Sample-size grid

A record's scope is a measured range of row counts, not one row count. Every
record-keyed design draws its sample size through `SampleGrid`
(`crates/antecedent/tests/common/calibration.rs`), and the gate runs each
record-emitting group once per grid point with
`ANTECEDENT_CALIBRATION_GRID_POINT=0|1|2`:

| grid | points | used for |
| --- | --- | --- |
| `STANDARD` | `n/2, n, 2n` | every design whose base `n` exceeds 100 (`grid_n`) |
| `SHORT_SERIES` | `3n/4, n, 2n` | series of at most 100 steps, whose base is already the shortest series the construction is supported for (`grid_n`) |
| `HEAVY` | `n/2, n, 3n/2` | designs that run for hours at their base (Bayesian derivative and Jacobian bands, ADMG front door, the 2500-row counterfactual designs) |

The base point (point 1, also what an unset variable means) is the design as it
was measured before the grid, on the same replicate data; points 0 and 2 salt
every generator in the harness, so each point measures independent data and
stays deterministic. A test name's `nNNN` names the base point.

Each point is gated on its own: the band, the 2000-replicate recheck (logged as
`<group>.p<k>.recheck.log`) and the precision floor apply per point.
`scripts/collect_coverage_records.py` merges the lines of one record id into one
row: `grid` holds the coverage measured at each point, `n_min..n_max` spans the
points, and the row is a boundary when any point is. It refuses a record that
misses a point, or whose points do not measure strictly growing sample sizes (a
design that does not scale its `n`). The matcher labels an execution
`calibrated` only inside `n_min..n_max` of a record that passed at every point;
a record that failed at any point is `measured_below_nominal` /
`boundary_record` over its whole range, reporting the failing point's
coverage; outside the range it is `outside_measured_scope` /
`sample_size_outside_measured_range`. Nothing is extrapolated.

A boundary record can recover the passing part of its range, but only from an
independent confirmation. `ANTECEDENT_CALIBRATION_CONFIRMATION=1` (the gate's
confirmation mode, run at the grid points that passed, for example with
`ANTECEDENT_CALIBRATION_GRID_POINTS="1 2"`) measures the same designs again
with a second fixed seed salt, so no replicate is shared with the first
measurement, and writes `<group>.p<k>.confirm.log` (and `.confirm.recheck.log`)
whose lines carry `"confirmation": true`. The collector keeps those tallies
apart, in the record's `confirmation` (shaped like `grid`). When a contiguous
run of at least two grid points reaching the top or the bottom of the grid
passed in the measurement and again in the confirmation, the record states
`confirmed_passing_range = [n_lo, n_hi]`. The record stays a boundary and keeps
its failing points; a confirmation point that failed its band leaves that
point unconfirmed. A named boundary never recovers a range, and a failing point
in the middle of the grid leaves nothing to recover. The matcher is
`calibrated` for a row count inside `confirmed_passing_range` (with every other
axis, the attestation and the scope checks unchanged), governed by the lowest
coverage over those points in the measurement and the confirmation, and it
reports that range as the scope. A row count in the failing region, or between
it and the range, is still `measured_below_nominal` / `boundary_record`.
`scripts/gate_parity_schema.sh` checks that the stated range is exactly what the
grid and the confirmation establish.

`antecedent_io_core::calibration::resolve_passing_range(record, rows)` says why a
row count of a boundary record is, or is not, recovered: `Recovered` (with the
confirmed region), `FailedPoint` (the grid point whose range or neighbourhood
governs, with its coverage), `Unconfirmed` (the points passed the measurement and
reach an end of the grid, but no confirmation passed at every one of them),
`NotContiguous` (a failed interior point separates the passing points and is
named), `InteriorRun` (failed points on both sides) and `LonePassingPoint`. The
slot of the row count stays the matcher's: `calibrated` exactly when this is
`Recovered`. The collector combines a confirmation with its measurement only when
every construction field (estimator, interval method, standard-error kind,
dependence, estimand, identification) and the nominal level agree, and only when
both were measured at the commit it collects; a confirmation that differs in any
of them is refused naming the field, and nothing of it is combined.

A named boundary asserts its measured coverage at every grid point with
`CoverageTally::assert_boundary_at([Some(m0), Some(m1), Some(m2)])`, so a
regression at any point fails (`None` gates a point at nominal). The values come
from the gate's own deterministic seeds: the band is a change detector around
the measured level, not a claim about the true coverage. A gated design that
fails at one point is named the boundary it measures the same way.

A wiring smoke run (`ANTECEDENT_CALIBRATION_SMOKE=1` with a small
`ANTECEDENT_CALIBRATION_NSIM`) never gates and flags its lines
`"smoke": true`. The collector refuses them except with
`--smoke --log-dir <dir> --out <scratch.toml>`, which writes a scratch registry
and never touches `parity/`.

### Two flows: engineering and calibration

Engineering and calibration are separate flows. A measurement is made on a development machine
and never in CI; CI fails while a coverage record is owed (`calibration_status.py --check`), but
calibration status never changes support or refuses a route. See [Earning and checking the records](guides/scientific-verification.md#earning-and-checking-the-records).

The measurement is deterministic: every replicate's data and resamples come
from fixed seeds, so re-running unchanged code reproduces the same numbers
(`crates/antecedent-estimate/tests/coverage_determinism.rs` holds that in
ordinary CI). A record is never re-measured on a timer.

**Fingerprint.** `scripts/evidence_fingerprint.py` computes a sha256 over
everything a test depends on: the normalized sources and manifests of the
workspace crates in its dependency closure (comments and workspace version
lines ignored), the test file with its helpers and the DGP file, `Cargo.lock`
and the workspace manifest, the toolchain file and the feature set, plus embedded and declared inputs
(`parity/fingerprint_inputs.toml`), build scripts and `ANTECEDENT_*` values. The
scientific records (`scripts/scientific_records.py`) and the calibration status
use this one implementation; the full list is in the canonical page above.

**Status.** `python3 scripts/calibration_status.py status` classifies every
record in `parity/coverage_records.toml`:

- `current` — the record's stored `fingerprint` (written by the collector when it
  stamps a measurement at HEAD) equals the fingerprint of the tree, or a passing
  probe attestation made at a tree with that fingerprint covers its measurement
  (see [Probes](#probes));
- `probe_owed` — not current, and the measurement stored replicate digests, so a
  probe can reproduce them;
- `measurement_owed` — not current and nothing to probe; only a re-measurement
  makes it current.

A record without a stored `fingerprint` is never current through its
measurement. `status --owed` prints the ids that are not current, `status --json`
the counts and per-record status. `status` always exits 0; `calibration_status.py --check`
exits 1 while any item is owed and is what CI and the release candidate run. The runtime snapshot of attesting record ids is the
`current` set.

### Measuring

See [calibration runtime and evidence reuse](calibration-runtime.md) for the
measured runtime audit, scheduling fixes, prefix-extension limits and shared
resource budgeting.

One command, from a clean checkout of the commit you are about to push:

```bash
bash scripts/measure_calibration.sh --dry-run   # the groups it would run
bash scripts/measure_calibration.sh --pilot     # 8-replicate smoke timings; no coverage evidence
bash scripts/measure_calibration.sh --dry-run   # projection from this commit's pilot
bash scripts/measure_calibration.sh             # measure what is not current, collect, print the status
bash scripts/measure_calibration.sh --owed      # the same set, spelled out (`calibration_status.py status --owed`)
bash scripts/measure_calibration.sh --owed --only PATTERN  # owed work whose group, record id/test or route matches
bash scripts/measure_calibration.sh --all       # re-measure every group
bash scripts/measure_calibration.sh --missing-only # fill missing evidence; keep prior evidence
bash scripts/measure_calibration.sh --missing-only --group 'weighted_interval_calibration:*'
bash scripts/measure_calibration.sh --jobs 6    # parallel groups (default: the core count)
```

It refuses a dirty working tree, because a measurement must correspond to a real
commit. It then works in four steps:

1. It asks `scripts/calibration_status.py status` which records are not current;
   probe first, so a record whose digests reproduce is not measured again. `scripts/calibration_groups.py` maps
   those records to the gate groups that measure them.
2. It builds the selected test binaries once, then runs each group alone through
   `scripts/measure_calibration_groups.sh`, several at a time; a record-emitting group runs
   as one job per sample-size grid point
   (`ANTECEDENT_CALIBRATION_GRID_POINTS=<k>`), so the points of a long group run
   side by side. The run includes the gate's inline 2000-replicate rechecks, per
   point. It prints each job's start and its result with elapsed time. The gate's logs go to
   `target/calibration-records/`, where the collector reads them, and earlier
   logs and consoles move to durable, SHA-scoped run directories under
   `target/calibration-archives/`. Each group's console
   output goes to `target/calibration-console/`, and its wall time is appended
   to `target/calibration-timings.tsv` for post-run review.
3. It runs `scripts/collect_coverage_records.py --keep-not_measured` (without the
   flag under `--all`). Each record is stamped with the commit it was measured
   at and the fingerprint of its test there. Records the logs do not re-measure
   keep their own `calibration_sha` and `fingerprint`.
4. It prints the calibration status of the result.

Commit `parity/coverage_records.toml` and the files the collector regenerates,
then push. A normal run that fails its calibration band stops before collection.
Missing-only and explicitly filtered runs harvest complete passing groups with
`--keep-existing --passing-only` even when other groups fail. A group with any
failed recheck, incomplete grid, mismatched SHA, or malformed records is skipped
as a whole, with reasons; its logs stay available. The wrapper still exits with a
failure after saving the completed evidence. Replacement IDs belonging to skipped
groups are deferred, leaving their original evidence intact.

Without `--all`, the selection has three parts:

- the groups behind records that are not current;
- any group in a calibration suite that has no record in the registry yet;
- when anything else runs, the pass/fail gates that emit no record (CI Type I,
  discovery FPR, SBC). No record describes
  those gates, so they run with every re-measurement.

Formal runner measurements use `suite_coverage_fwer.v1` after the complete source
family and hypothesis audit are frozen in `scripts/calibration_family_manifest.json`
and `scripts/calibration_family_labels.json`. Its exact binomial coverage decisions
allocate total risk 0.05 across every source gate group, three grid slots, 256
source-possible coverage cells per group/grid, and predetermined 400/2,000/20,000
attempt looks with weights 1/2, 1/4, 1/4. Selection and sharding never shrink this
family. The initial `release_2_4_1_validation_v1` seed domain supplies fresh validation
data; each registered domain's SHA256 first eight bytes, interpreted little-endian,
are XORed into the existing seed salt. Source SHA, family, grid and domain must agree when extending
saved tallies. Fresh-seed confirmations are outside this protocol's allocated risk
and are refused. Smoke pilots and legacy replays retain their original semantics.

Scientific methods, reusable examples and behavioral tests are named for their
purpose, independently of the package release. A method revision may have its own
version when its definition changes. Package versions belong in packaging,
release materials and explicit historical-compatibility fixtures.

New calibration seed domains must describe the statistical procedure and its
revision, for example `pag_bounded_outcome_confidence_set_validation_v1`.
Existing domain strings, their derived salts, allocations and collected evidence
are immutable historical identities, including experiments already reserved before
outcomes. Do not rename them as a cleanup: an evidence-safe migration needs an
explicit review of every reference, seed stream and cumulative risk allocation.

A method revision informed by earlier results needs a new independent validation
domain and an unused cumulative hypothesis slot before new outcomes are observed.
`parity/calibration_decision_allocations.json` retains the original risk budget and
spent HC1 slot while reserving a separate HC3 construction under
`release_2_4_1_weighted_hc3_validation_v1`. The new family manifest binds this receipt's
hash. HC3 was selected after the older results; it was not present in the initial
source roster. Its independent validation consumes another slot within the original
256-cell allowance. Changing the domain never supplies another family alpha or
recycles an earlier slot. Unknown slot reuse and changed constructions fail closed.

This controls false rejections of coverage compatibility for that frozen validation
family. It does not establish exact nominal coverage or control every usefulness,
reporting, width or other scientific check. Record counts and all intermediate raw
decisions remain durable, including a first look superseded by its registered
recheck. Earlier collected evidence retains its original decision protocol.

Use `--missing-only` to finish an incomplete collection while existing evidence awaits
a probe or a re-measurement. It selects groups with neither matching coverage records nor a
complete passed entry in `parity/calibration_gates.toml`, irrespective of status.
The collector uses `--keep-existing`: it preserves every old record, its original SHA,
and earlier gate evidence, and refuses collisions with existing record IDs.
For incremental evidence-only batches, `--no-cells --defer-generated` also leaves
supported support metadata and generated assets for the final combined harvest.
Raw-log validation and whole-group completeness still apply.
`--group` restricts the selected set by a quoted full-label glob; repeat it for multiple
groups. A filter matching nothing fails before measurement. To remeasure a numerically changed construction already in the registry, select its
groups with `--all --group 'suite: test'` and repeat `--replace-record ID` for every
existing record those groups emit. Filtered runs preserve unrelated evidence; the
collector rejects unlisted collisions and any replacement ID lacking both old and
fresh evidence. For a fixed higher-precision protocol, add `--replicates 20000` before measuring;
the chosen budget applies only to the selected groups. The default remains 400
(with the existing 2,000-replicate precision recheck). Runs starting at 2,000 or
more replicates apply their registered final-look decision directly, without an extra extension.
The formal family protocol accepts only these three registered replicate budgets.
Decide the budget before examining results, retain the outcomes, and do not search
seeds or repeat until a pass. These options do not
make a record current: only new evidence or a passing probe does. Passing collected records whose numerical outputs changed must be
remeasured separately; `--missing-only` does not select them.

The fingerprint and probes keep the cost proportional to the change:

- an edit to a crate only the owed tests do not depend on moves no fingerprint;
- a comment, whitespace or workspace-version edit moves none either;
- a change that cannot move a number is cleared once a probe reproduces the
  records' digests at the new fingerprint;
- a change to a shared crate, the lock file or the toolchain moves every
  fingerprint that reads it.

**How long it takes.** Groups run side by side (`measure_calibration.sh
--jobs`). Independent seeds inside a group run across `map_replicates` workers;
each seed uses a serial `ExecutionContext::for_tests`, so changing the worker
count does not change its interval. The smoke pilot times every selected group
and grid point on the current clean commit with eight replicates. It stores
logs separately in `target/calibration-pilot-records/` and timings in
`target/calibration-pilot-timings.tsv`. Those logs cannot enter the coverage
collector. The next `--dry-run` projects the 400-replicate pass and expected
2,000-replicate extensions from these timings and the previous recheck rates.
Only the test runner's simulation time scales with replicate count; Cargo and
gate startup are charged once per job. The projection is approximate: convergence
does not scale perfectly with replicates, and a changed coverage rate changes the recheck
count. Without a pilot at HEAD, `--dry-run` gives no duration estimate. An old
forecast that summed wall times of suites competing on the same CPU substantially
overstated the full-pass time. The measurement runs the longest previously measured grid
jobs first (falling back to pilot timings) to reduce the final straggler. A change
costs only the groups behind the records whose fingerprint it moves.

The collector refuses to stamp HEAD while the sources a fingerprint reads
(`crates/`, `python/`, the manifests and the toolchain file) differ from HEAD. It keeps a rechecked point's more precise run, rewrites the
`calibration` / `calibration_reason` pair on every supported cell and estimator
row (a supported cell whose every reported-level record is a boundary also states
`calibration_reason = "boundary_record"` beside its list), and regenerates `crates/antecedent-io-core/data/coverage_records.json`
(the snapshot `coverage_records_data` loads).
`scripts/gate_parity_schema.sh` then checks what those rows claim. To collect
logs measured at another commit, pass `--sha <commit>`; those records carry no
fingerprint and stay owed to a probe or a re-measurement.

### Probes

A measurement stores, for every grid point, the chained digests of its first
10 / 32 / 100 / 320 / 1000 replicates (the record's `probe` table). Replicate
`k` is a deterministic function of `k`, and the harness folds each replicate's
interval endpoints, truth and bound scope facts, as raw bits in a fixed order,
into a fingerprint (`CoverageTally::record`, `ReplicateDigest`). Re-running the
record's test at `ANTECEDENT_CALIBRATION_NSIM=k` therefore reproduces the stored
prefix digest exactly when the first `k` replicates are bit-identical. A match
for `k` replicates bounds the per-replicate rate of any undetected change at
`3/k`; the bound has to fit inside the record's margin, the observed coverage
minus the acceptance floor at its worst grid point (3 MCSE for a boundary
record).

```bash
python3 scripts/calibration_probe.py quick   --dry-run   # k = 10 per grid point; the commands only
python3 scripts/calibration_probe.py bounded             # k = smallest stored prefix >= 3/margin
python3 scripts/calibration_probe.py bounded --record <id>
```

Each probe runs the record's group through `scripts/measure_calibration_groups.sh` under
the measurement's frozen decision environment, in smoke mode (never gated), on a
clean commit, with logs in `target/calibration-probe/`. It writes
`parity/calibration_probe_attestations.toml`, one row per record and
measurement: `record_id`, `calibration_sha`, `sha` (the probed commit),
`fingerprint` (the record's fingerprint at that commit), `k`, `bound = 3/k`,
`result`. A mismatch leaves the record owed. A record whose
margin needs more replicates than its longest stored prefix, or that stores no
digests (measured before probes existed), is reported `owed`; only a
re-measurement makes it current. `quick` is a cheap detector: a record stands on a row
only when `k` is at least what its margin needs.

`calibration_status.py status` counts a record as current when a passing row
covers it and the row's `fingerprint` equals the fingerprint of the tree.
Commit the attestation file with the change it attests; re-measuring a record
drops its rows.

Mark a `parity/*.toml` capability `done` only with conformance under `conformance/`
**or** a named harness in the gate script, plus a recorded reference-generation
command when black-box comparison applies.

Statuses: `pending` | `done`.

## Release candidates

The independent [practitioner acceptance suite](practitioner-acceptance.md)
is an additional cut requirement for each release. Run its Python and Rust jobs and both scale
sizes against the candidate, and close its leftover ledger. It remains outside
`gate_release.sh`; passing the commands below alone does not discharge S.

`gate_release.sh` is the pull-request gate; a release is cut with
`scripts/gate_release_candidate.sh` (which `scripts/tag_release.sh` runs before
tagging). It refuses a dirty tree and requires `cargo-deny`. It needs one input:

```bash
CI_RUN_ID=<GitHub Actions ci run on this exact HEAD> \
  bash scripts/gate_release_candidate.sh
```

- **`CI_RUN_ID`** is the database id of a `ci` workflow run whose `headSha` is
  the commit being cut (`gh run list --workflow ci.yml --commit "$(git rev-parse HEAD)"`).
  The gate reads it with `gh run view <id> --json headSha,jobs` and requires
  every job id listed in `parity/release.toml` `required_jobs` to have
  succeeded. Job ids are `ci.yml` keys (`static`, `rust`, `features`, `public-api`,
  `cited-tests`, `benches`, `deny`, `development`, `python-lint`, `scientific`,
  `python-wheels`); a run reports display names instead, one per matrix
  combination ("Rust ubuntu-latest", "Rust ubuntu-latest facade",
  "Rust macos-latest facade", "Wheel macos-14 py3.12"). The Rust facade
  and remaining-workspace suites are both required on Ubuntu and macOS;
  together they execute the workspace's tests and doctests.
  `scripts/ci_workflow.py` parses `ci.yml` as YAML and expands every matrix
  combination, so a missing or failed wheel leg fails the cut.
- The gate then runs every other section of `gate_release.sh` with `cargo-deny` required (its
  `static` section requires zero owed scientific records and zero owed calibration records;
  nothing is re-run or re-recorded), runs the release inventory with no unresolved row,
  the composition and decision exit stories, the release-only scale cases and the release-closure
  receipts, and builds one wheel into a fresh directory, installs it into a fresh venv and runs
  the documented examples, the cyclic-systems gate and the full Python test suite against that
  installed wheel (outside `python/`, so the source tree cannot shadow it).

Each gate that decides a release has a `--self-test` mode that feeds it
deliberately broken input and requires a failure: `gate_composition.sh`, `gate_transport.sh`, `gate_gcm_conformance.sh`,
`check_release_inventory.py`, `gate_parity_schema.sh`, `gate_docs_support_matrix.sh`,
`gate_release_candidate.sh`, plus the
citation, reachability, metadata and support gates.
`scripts/gate_selftests.sh` runs all of them. Each case builds a repo overlay and
runs a whole gate, so it takes tens of minutes and is run on demand (after
changing a gate or its cases), not by `gate_release.sh` or CI.

## Python lint / types

`scripts/gate_python_lint.sh` runs **ruff** (check + format) and **mypy** over
`python/antecedent` (including hand-written `.pyi` stubs for `_native`). It is a
local equivalent of the checks in the separate CI `python-lint` job; the
wheel-matrix job does not run lint or type checks.

```bash
cd python
uv sync --group dev
# If CONDA_PREFIX and VIRTUAL_ENV are both set, unset CONDA_PREFIX first.
bash ../scripts/gate_python_lint.sh
```

Or individually:

```bash
cd python && uv run ruff check antecedent tests ../examples/python
uv run ruff format --check antecedent tests ../examples/python
uv run mypy
```

## Native extension builds

All Python builds of `antecedent._native` — wheels, `maturin develop`, and the
editable rebuild uv performs when the install goes stale — compile with Cargo's
release profile (`[tool.maturin] profile = "release"` in `python/pyproject.toml`).
Do not remove that pin: the PEP 517/660 default is the debug profile, which
produces bit-identical estimates at roughly 50× the wall time, so nothing
downstream notices. A debug-profile extension warns on import (the flag is
`antecedent._native.__build_optimized__`), and `tests/test_build_profile.py`
fails the pytest suite — locally and in CI — if the extension under test is
unoptimized (`ANTECEDENT_ALLOW_DEBUG_NATIVE=1` opts out deliberately).

## Recorded scientific tests

The scientific assertions (those whose claims carry the `scientific` role, plus the uncited ones in
`parity/scientific_tests.toml`; calibration simulations and acceptance measurements are excluded)
are recorded in `parity/scientific_records.toml`: per assertion, its exact selector, roles, the
commit it passed at, a fingerprint of its dependencies and the time. While the fingerprint still
matches, the recorded pass carries forward and the assertion does not rerun. What the fingerprint
covers, and what owed means for support, is in [Earning and checking the records](guides/scientific-verification.md#earning-and-checking-the-records).

```bash
bash scripts/test.sh                                  # engineering, then owed scientific, then owed calibration
bash scripts/test.sh scientific                       # owed scientific assertions only
bash scripts/test.sh scientific --only PATTERN        # owed rows matching PATTERN (substring or glob)
bash scripts/test.sh scientific --all                 # manual: rerun and re-record everything
python3 scripts/scientific_records.py status          # which assertions are owed
python3 scripts/scientific_records.py status --check  # exit 1 if any owed; executes nothing (CI)
```

`scientific_records.py run` is the one entry point that executes scientific assertions. The
engineering lane excludes them (the nextest default filter and the pytest `scientific` marker are
both derived from the registries). The owed Rust assertions run in one `cargo nextest run` per
crate with the optimized test profile and the owed Python assertions in one pytest batch; each
batch is recorded when it finishes. The capability, transport, composition, checked-execution and
GCM conformance gates only resolve their rows to exact tests (`run_evidence_rows.py --static`).
The runner proves the exact assertion executed and was not ignored; failures are reported and never
recorded. A recorded pass is evidence for support only; it is never a calibration measurement or
coverage record.

## Tests that matter

| Kind | Role |
|------|------|
| Unit / property | Algorithm invariants, graph witnesses, numeric edge cases |
| Conformance | Frozen fixtures vs expected outputs (`conformance/`) |
| Calibration | Coverage / Type I / null FPR (`measure_calibration_groups.sh`, measured locally by `measure_calibration.sh`) |
| Cross-language | Python bindings exercise the same semantics |
| Criterion benches | Designated hot paths; release gate locally and in CI |
| Fuzz | Parsers / graph / artifact surfaces under `fuzz/` |

Tolerance classes live in `antecedent-core` (ADR 0010). Do not tighten or loosen a
conformance band without an ADR-level reason.

## Performance rules (merge blockers)

- Data layout and copy policy are designed with the algorithm, not after.
- No per-observation dynamic dispatch / Python / hash / heap in scalar inner loops
  unless the slow path is API-explicit and separately benched.
- Scalar kernels are the correctness reference; SIMD/BLAS/parallel paths pass the
  same tests.
- Do not change statistical semantics to go faster (sample selection, masking,
  conditioning order, randomization, stopping rules, estimands).
- Parallelism is bounded by `ExecutionContext`.
- Superlinear storage must expose bounds, streaming, or refuse — not OOM later.

See [hot_paths.md](hot_paths.md).

## Feature flags

Cargo features mean “optional adapter / heavy backend,” never “different numbers
on the default path.”

Always on: `faer`, portable kernels, `ExecutionContext` parallelism (`rayon`
rejected).

Present today (examples): `antecedent-data/arrow`, `antecedent-model/gaussian-process`,
`antecedent-learn/ml-gbdt`, `ml-forest`, `ml-neural`. A feature that gates no code is not
declared (the HMC backend is always compiled), and there is no `simd-runtime`
feature, so `KernelPolicy::allow_arch_simd` always selects the portable kernels.
Ingest and exchange adapters are optional features and never reshape core types.
The `development` feature (and `scripts/dev_python.sh` for Python, which
installs a development build and checks `antecedent.build_flavor()`) turns the
support gate into a recorder: a route that lacks provenance or an independent
scientific check runs instead of refusing, and its result carries `would_refuse`
(the `not_supported` code, the detail and the missing evidence). Scientific
refusals still refuse. A development build is for exercising unsupported routes
and developing their evidence; artifacts it seals are stamped `development` and
release consumers refuse them, so never ship, publish or write fixtures from one.
The only place a `not_supported` refusal is built is
`antecedent_core::support_gate::require_supported`, and
`scripts/gate_not_supported_sites.py` enforces that. See
[How Antecedent checks its scientific methods](guides/scientific-verification.md#the-evidence-model).
CI compiles the feature combinations crates.io users get (default features, each
optional feature alone, `--no-default-features`) in the `features` job.

## Unsafe / deps

Reviewed `unsafe` is concentrated in `antecedent-kernels` (SIMD), the
`antecedent-data` buffer/Arrow FFI adapters (`unsafe from_ffi`), and IO mmap
(`unsafe open_path_mapped`; safe `open_path` is an owned snapshot).
New `unsafe` needs justification in review. Dependency and license policy: ADR 0008 and `deny.toml`.

## Versions

Workspace and Python package versions follow `[workspace.package].version` in `Cargo.toml`.
Artifact format is frozen separately — see [artifacts.md](artifacts.md).

MSRV: Rust 1.85, edition 2024. Python: CPython 3.11–3.14.

Keep `[workspace.package].version` in `Cargo.toml` and `version` in
`python/pyproject.toml` in sync:

```bash
bash scripts/set_version.sh X.Y.Z
```

`set_version.sh` also updates path-dependency pins under `crates/*/Cargo.toml`,
the Python fallback `__version__`, and the local package entry in
`python/uv.lock`, and freezes the previous cut's supported-cell block in
`docs/release-notes/` so a later matrix regen cannot overwrite it. Refresh
`Cargo.lock` with `cargo update -p antecedent` (or a workspace check) before
committing. The generator rewrites live supported-cell markers only in
`docs/release-notes/vX.Y.Z.md` for the current workspace version.

## Releases

Head each changelog section with the bare version, `## X.Y.Z`. Do not add
status markers such as `draft`, `unreleased`, or `in preparation` to the
changelog, release notes, `SECURITY.md`, or other documentation: they are a
manual step at the cut and get missed. Write release-neutral text that stays
true once the version ships. The tag, not the document, records that a release
was published; a package version bump is not proof of that.

Before merging the release PR:

1. Commit the reviewed implementation, tests, and documentation with DCO sign-off.
   Keep the generated support/conformance output current and the worktree clean.
   Review release notes as the user-facing version overview, the changelog as
   the delta from the previous version, and affected guides and API docs for
   examples, formulas, assumptions, refusal reasons, artifact formats, and
   Python/Rust agreement. Remove temporary cut notes and provisional wording.
2. Run the relevant tests, formatters, linters, and documentation checks locally.
   Open a PR and require the CI jobs on the committed HEAD, including the
   every job in `parity/release.toml` `required_jobs` (`static`, `scientific`,
   `python-lint`, `python-wheels` and the rest). The pull-request gate runs in CI; a local full run
   is optional.
3. For an actual release cut, run
   `CI_RUN_ID=<run> bash scripts/gate_release_candidate.sh`. It prints the
   coverage records that are not current for HEAD (`calibration_status.py`);
   measure or probe them before the cut so the release labels them `calibrated`.
   They do not fail the gate.
4. Run `cargo deny check` with a freshly fetched advisory database. A cached
   offline audit is useful evidence, but does not establish current advisory status.
5. Run `bash scripts/publish_crates.sh --dry-run`. Inspect any fallback to
   `cargo check` for unpublished workspace dependencies: that is not a completed
   package verification for those crates.
6. Build wheel and source-distribution artifacts and test installation outside
   the checkout, without an editable install. Verify the supported OS/Python wheel
   matrix before publishing; one local extension build covers only that environment.
7. Check the changelog, release notes, user examples, refusal/compatibility scope,
   and evidence ledger. Record measured timings separately from test ceilings.

Before tagging, confirm the changelog has a `## X.Y.Z` section for the version
being cut and no status markers anywhere. `docs/release-notes/preparation.toml`
is optional: when it is present, `target_version` names the release notes
(`docs/release-notes/vX.Y.Z.md`), which may be ahead of the package; when it is
absent, X.Y.Z is the workspace package version in `Cargo.toml` and the missing
file is not an error. Those notes must be current, the supported-versions table
in `SECURITY.md` and the version in `CITATION.cff` must name that version, and
release-status text must match the cut. Coverage records that are not current
for HEAD are listed by the candidate gate. Tag only the approved, clean commit after these checks pass.
Do not remove the release gate's clean-diff check to accommodate pending edits.

Tagged releases drive docs and artifact publishing (GitHub Release assets and
public PyPI). The tag `vX.Y.Z` must point to a commit with a successful exact-SHA
`main` CI run. The tag workflow checks the committed version and all 16 wheels
from that run before any publishing job starts; it does not rebuild the wheels.

```bash
# Optional: bump and commit on main first (X.Y.Z is the version being cut)
bash scripts/set_version.sh X.Y.Z
cargo update -p antecedent
git add Cargo.toml Cargo.lock python/pyproject.toml python/uv.lock \
  python/antecedent/__init__.py crates/*/Cargo.toml fuzz/Cargo.lock \
  CHANGELOG.md CITATION.cff docs/release-notes/
git commit -s -m "chore: bump version to X.Y.Z"

# Tag current (or just-bumped) version and push
CI_RUN_ID=<ci run on HEAD> bash scripts/tag_release.sh   # runs gate_release_candidate.sh
git push origin vX.Y.Z
```

Workflow [`.github/workflows/publish-release.yml`](https://github.com/iridae-dev/antecedent/blob/main/.github/workflows/publish-release.yml)
reuses the 16 wheels built and fully tested by the green `main` CI run on the
exact tag commit, then publishes to public PyPI and the GitHub Release as
**independent jobs**. It builds the docs bundle from that run's Python 3.12
Linux wheel. Trusted publishing (`id-token: write`) must not
wait on GitHub asset uploads: a unicorn on `uploads.github.com` skipped PyPI
for 1.10.0 while crates.io succeeded. Release assets are
uploaded one file at a time with retries. A manual retry may supply
`reuse_run_id`, but that run must be a successful `main` push CI run on the
exact tag commit and contain the complete wheel matrix.

Configure a pending/trusted publisher on [pypi.org](https://pypi.org) for this
repo and workflow file `publish-release.yml` (Environment blank unless the job
sets `environment:`).

Install with `pip install antecedent`, or download a wheel from the GitHub
Release. (GitHub Packages has no supported Python registry; do not use
`upload.pypi.pkg.github.com`.)

Azure / non-GitHub deploys: bake a Release `.whl` into the image, or install
from PyPI.

## crates.io (Rust)

Publish the library graph (facade `antecedent` + `antecedent-*` deps). **Do not**
publish `antecedent-py` (`publish = false`).

```bash
# Local dry-run (default)
bash scripts/publish_crates.sh

# Real upload (CARGO_REGISTRY_TOKEN or CRATES_IO_TOKEN)
bash scripts/publish_crates.sh --execute
```

The `crates-dry-run` and `publish-crates` jobs of
[`.github/workflows/publish-release.yml`](https://github.com/iridae-dev/antecedent/blob/main/.github/workflows/publish-release.yml)
publish the crates as part of the same tagged release as the wheels: every crate is
dry-run before PyPI is touched, and the upload runs after PyPI succeeded, in the
`release` environment. Set repository secret `CRATES_IO_TOKEN`.

Checklist before the first public crate release:

1. `cargo test --workspace` green; run `bash scripts/gate_release.sh` locally before the cut.
2. `bash scripts/publish_crates.sh --dry-run` succeeds.
3. Root `CHANGELOG.md` has notes for the version.
4. Tag `vX.Y.Z` (or dispatch the workflow) with `CRATES_IO_TOKEN` configured.
5. Confirm `cargo add antecedent` resolves on crates.io / docs.rs.

## Repo create checklist

1. Create a GitHub repository and push this tree.
2. Enable Actions.
3. Confirm `workspace.package.repository` in `Cargo.toml` matches the remote.
4. Configure PyPI trusted publisher for `publish-release.yml`.
5. Tag `vX.Y.Z` for the version being cut (or bump first) to cut wheels + PyPI (+ crates.io with token).
