# Development

Product name: **Antecedent**. Python distribution/import: `antecedent`. Rust
day-1 facade: `antecedent` (`cargo add antecedent`). Supporting crates are
`antecedent-*` on crates.io.

## CI vs local gates

GitHub Actions CI (`ci.yml`) runs the following checks on every PR:

- **`rust`** — fmt, clippy, `cargo test --workspace`, DCO (plus an optional
  crates.io publish dry-run when manifests change).
- **`gates`** — first the calibration attestation
  (`scripts/gate_calibration_attestation.sh`, seconds), then
  `scripts/gate_release.sh`, which runs the parity-manifest schema check,
  provenance and metadata checks, support-matrix and evidence checks, feature
  gates, artifact tests, and Criterion benchmark smokes.
- **`python-lint`** — Ruff, mypy, and pytest with an 85% coverage floor after
  building the native extension.
- **`python-wheels`** — builds and tests the supported wheel matrix.

**Statistical calibration is measured on a development machine before you push,
never on GitHub.** CI only checks that the committed coverage records match the
code being uploaded, and the `gates` job fails when they do not (see
[Coverage records](#coverage-records)). Criterion benchmark smokes do run
through `gate_release.sh`; they check execution, not timing regressions.
`cargo deny` runs inside `gate_release.sh` only when `cargo-deny` is on PATH,
which it is not in CI.

## Gates (local / slow path)

Feature gates own inventory honesty + conformance for a domain. Run them locally
before a release, or when a change might break something unintended:

```bash
bash scripts/gate_estimate_ci.sh
bash scripts/gate_bayesian.sh
bash scripts/gate_gcm.sh
bash scripts/gate_pag.sh
bash scripts/gate_context.sh
bash scripts/gate_attribution.sh
bash scripts/gate_design_state.sh
bash scripts/gate_upstream_names.sh
bash scripts/gate_response_calibration.sh
bash scripts/gate_causal_artifacts.sh
bash scripts/gate_estimate_reuse.sh
bash scripts/gate_composition.sh   # 1.10 consuming contract/claim tests must actually run
bash scripts/gate_metadata_consistency.sh
bash scripts/gate_evidence_reachability.sh
bash scripts/gate_support_matrix.sh   # public license cells; default refused
bash scripts/gate_docs_support_matrix.sh
bash scripts/measure_calibration.sh   # coverage / CI Type I: measure what is owed, collect, attest
bash scripts/gate_release.sh       # prior gates + inventory + benches + optional deny
bash scripts/gate_python_lint.sh   # local equivalent of the CI lint/type checks
```

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
| `SHORT_SERIES` | `3n/4, n, 2n` | series of at most 100 steps, whose base is already the shortest series the construction is licensed for (`grid_n`) |
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
a record that failed at any point is `scope_not_assessed` /
`boundary_record` over its whole range, reporting the failing point's
coverage; outside the range it is `scope_not_assessed` /
`sample_size_outside_measured_range`. Nothing is extrapolated.

A named boundary asserts its measured coverage at the base point with
`CoverageTally::assert_boundary(m)` and is recorded, not gated, at the other
points; once those points are measured, `assert_boundary_at([m0, m1, m2])`
holds each point to its own value (`None` gates a point at nominal). A gated
design that fails at one point is named the boundary it measures the same way.

A wiring smoke run (`ANTECEDENT_CALIBRATION_SMOKE=1` with a small
`ANTECEDENT_CALIBRATION_NSIM`) never gates and flags its lines
`"smoke": true`. The collector refuses them except with
`--smoke --log-dir <dir> --out <scratch.toml>`, which writes a scratch registry
and never touches `parity/`.

### When a record stands

The measurement is deterministic: every replicate's data and resamples come
from fixed seeds, so re-running unchanged code reproduces the same numbers
(`crates/antecedent-estimate/tests/coverage_determinism.rs` holds that in
ordinary CI). A record therefore stands until the code it measured changes,
and is never re-measured on a timer.

`scripts/calibration_surface.list` is the only owner of that code. It assigns
every path that can move a number (crate sources, manifests, the toolchain, the
shared harness, each calibration suite) to a facet:

- `core` — shared numerical and harness code (least-squares, conjugate,
  bootstrap, RNG, facade dispatch, manifests, toolchain). Every record
  carries it: a core edit owes a re-measurement unless a reviewed replay
  waiver covers the change.
- `estimator.*` / `identity.*` — one estimator or identification
  implementation. A change owes only the records that use that estimator or
  identity.
- narrower facets — `mechanism` (the fitted-SCM crates and the dispatch path
  that alone reaches them), `design`, one `suite.<file>` per calibration
  suite, and `registry_mirror` (generated tables no measurement reads). A
  change invalidates only the records carrying the facet.

Each record's `facets` are derived from the record itself by
`scripts/calibration_facets.py`: the suite of its test and DGP file, and every
`estimator.*` / `identity.*` (or `mechanism` / `design`) a `key` line assigns
to its fields. Every record also carries `core`. `scripts/gate_parity_schema.sh`
rejects a record whose facets are not exactly the derived set. `estimator.*` /
`identity.*` isolation is the `key` line: a change owes only records whose
fields match. For `mechanism` / `design`, `check` still fails when a file
outside the facet names the facet's items, unless an `allow` line records that
reviewed reference exactly; a new reference or a stale entry fails.

A record is **attested** while none of its facets differs between its
`calibration_sha` and the tree.

### The rule: measure locally, CI rejects what does not match

A change to statistical code is measured on your machine before you push. CI
rejects an upload whose records do not match the code:
`scripts/gate_calibration_attestation.sh` runs first in the `gates` job on every
pull request and every push to `main`, and again inside `gate_release.sh`. It
never runs a replicate. It compares file contents through git and takes a few
seconds. It fails when:

- a record **owes a re-measurement**: a path in a facet it depends on changed
  between its `calibration_sha` and the tree, and no valid replay waiver covers
  the change. The failure names the drifted facets, their changed paths, the
  number of records owed, and the command to run;
- a record's `calibration_sha` **does not resolve**. A rebase, squash or deleted
  branch orphans the commit a record names. Push the branch or a tag that
  preserves it (`git tag calibration/<name> <sha>` and
  `git push origin calibration/<name>`), or re-measure. The `gates` job checks
  out with `fetch-depth: 0`, so any pushed commit or tag resolves;
- a replay waiver is invalid, or the list or its facet boundaries are broken
  (`calibration_facets.py check`).

Records that are attested, or `attested_by_replay` under a valid waiver, pass.
The same check is what a release cut requires, so there is no separate release
mode.

### Measuring

One command, from a clean checkout of the commit you are about to push:

```bash
bash scripts/measure_calibration.sh --dry-run   # the groups it would run, with a rough duration
bash scripts/measure_calibration.sh             # measure what is owed, collect, re-run the gate
bash scripts/measure_calibration.sh --all       # re-measure every group
bash scripts/measure_calibration.sh --jobs 6    # parallel groups (default: the core count)
```

It refuses a dirty working tree, because a measurement must correspond to a real
commit. It then works in four steps:

1. It asks `scripts/calibration_facets.py` which records owe a re-measurement
   (or name a commit this clone lacks). `scripts/calibration_groups.py` maps
   those records to the gate groups that measure them.
2. It builds the selected test binaries once, then runs each group alone through
   `scripts/gate_calibration.sh`, several at a time; a record-emitting group runs
   as one job per sample-size grid point
   (`ANTECEDENT_CALIBRATION_GRID_POINTS=<k>`), so the points of a long group run
   side by side. The run includes the gate's inline 2000-replicate rechecks, per
   point. It prints each job's start and its result with elapsed time. The gate's logs go to
   `target/calibration-records/`, where the collector reads them, and earlier
   logs move to `target/calibration-records.previous/`. Each group's console
   output goes to `target/calibration-console/`, and its wall time is appended
   to `target/calibration-timings.tsv`, which later `--dry-run` estimates use.
3. It runs `scripts/collect_coverage_records.py --keep-attested` (without the
   flag under `--all`). Each record is stamped with the commit it was measured
   at. Records that still stand keep their own `calibration_sha`.
4. It runs the attestation gate on the result.

Commit `parity/coverage_records.toml` and the files the collector regenerates,
then push. A group that fails its calibration band stops the command before
collection.

Without `--all`, the selection has three parts:

- the groups behind owed records;
- any group in a calibration suite that has no record in the registry yet;
- when anything else runs, the pass/fail gates that emit no record (CI Type I,
  discovery FPR, SBC, `gate_response_calibration.sh`). Nothing can attest those
  gates, so they run with every re-measurement.

Facets and replay waivers keep the cost proportional to the change:

- an edit confined to one suite owes only that suite's records;
- an estimator or identification-path edit owes only the records that use it;
- a `mechanism` edit owes only the fitted-SCM records;
- a reviewed change that cannot move a number owes nothing, through a waiver;
- a `core` edit owes every record, unless a reviewed replay waiver covers it;

**How long it takes.** Groups already run side by side (`measure_calibration.sh
--jobs`). Independent seeds inside a group run across `available_parallelism`
workers (`map_replicates` in the coverage harness). Each seed still builds a
serial `ExecutionContext::for_tests` study, so the same seed is the same
interval.

A full re-measurement is a few hours on an M-series laptop, not an overnight
one-core job. A Bayesian derivative 2000-replicate recheck that used to pin one
core for ~10 h is about 1–1.5 h. A change that drifts one suite or facet costs
only the groups behind its records. `--dry-run` still scales suite timings by
the three-point grid factor (about 3.5× for linear-in-`n` designs) until local
per-point timings replace them.

The collector refuses to stamp HEAD while the surface its records depend on
differs from HEAD. It keeps a rechecked point's more precise run, rewrites the
`calibration` / `calibration_reason` pair on every licensed cell and estimator
row, and regenerates `crates/antecedent-io/src/coverage_records_data.rs`.
`scripts/gate_parity_schema.sh` then checks what those rows claim. To collect
logs measured at another commit, pass `--sha <commit>`. After you edit
`scripts/calibration_surface.list`, `collect_coverage_records.py --retag`
recomputes the facets without measuring anything.
`python3 scripts/calibration_facets.py status` prints the per-commit, per-facet
drift report on its own.

### Replay waivers

A reviewed change that cannot move a measured number (a new field with a
default, a diagnostic) can stand in for a re-measurement through a waiver in
`parity/calibration_waivers.toml`, which is outside the surface. A waiver names
`from` (the commit or tag the records were measured at), `to` (the commit it
attests forward to), the exact surface `paths` that changed between them, a
`justification`, `reviewed_by`, and `replay` records, each with the waived
paths its test `exercises`. A record is **attested_by_replay** only when it was
measured at `from`, every drifted path in its facets is one the waiver names,
and none of those facets changed between `to` and the tree. Any other change
leaves it owing, as before.

The evidence is a replay: `calibration_facets.py replay --waiver <id>`, run on a
clean checkout of `to`, re-runs the gate groups behind the replay records
through the unchanged `scripts/gate_calibration.sh` and compares each emitted
`calibration-record` payload with the stored record bit for bit (covered count,
observed, mcse, replicates and every other emitted field). It writes the
outcome (the commit replayed at, the record ids, `identical`, any differing
fields, and a fingerprint of each stored record) into the waiver. `check`
fails a waiver whose replay was not identical, whose outcome is missing or was
run elsewhere, whose stored records changed since, whose `from` or `to` does not
resolve, which names a path that is off the surface or unchanged within its
range, or whose `exercises` are empty, name a path outside the waiver, leave a
waived path unexercised, or name a file the record's test cannot reach or a facet
it does not carry. The attestation gate reports covered records as
`attested_by_replay (waiver <id>)`, accepts them, and prints their count and
waiver ids.

```bash
python3 scripts/calibration_facets.py replay-candidates --from <measured tag> --to <sha>
# write the waiver (without `outcome`), check out `to`, then:
python3 scripts/calibration_facets.py replay --waiver <id> --dry-run   # the gate groups it runs
python3 scripts/calibration_facets.py replay --waiver <id>
```

Mark a `parity/*.toml` capability `done` only with conformance under `conformance/`
**or** a named harness in the gate script, plus a recorded reference-generation
command when black-box comparison applies.

Statuses: `pending` | `in_progress` | `done`. No waiver vocabulary.

## Release candidates

For 1.11, the independent [practitioner acceptance suite](practitioner-acceptance.md)
is an additional cut requirement. Run its Python and Rust jobs and both scale
sizes against the candidate, and close its leftover ledger. It remains outside
`gate_release.sh`; passing the commands below alone does not discharge S.

`gate_release.sh` is the PR inventory; a release is cut with
`scripts/gate_release_candidate.sh` (which `scripts/tag_release.sh` runs before
tagging). It needs one input:

```bash
CI_RUN_ID=<GitHub Actions ci run on this exact HEAD> \
  bash scripts/gate_release_candidate.sh
```

- **`CI_RUN_ID`** is the database id of a `ci` workflow run whose `headSha` is
  the commit being cut (`gh run list --workflow ci.yml --commit "$(git rev-parse HEAD)"`).
  The gate reads it with `gh run view <id> --json headSha,jobs` and requires
  every job id listed in `parity/release.toml` `required_jobs` to have
  succeeded. Job ids are `ci.yml` keys (`rust`, `gates`, `python-lint`,
  `python-wheels`); a run reports display names instead, one per matrix
  combination ("Rust ubuntu-latest", "Wheel macos-14 py3.12").
  `scripts/ci_workflow.py` parses `ci.yml` as YAML and expands every matrix
  combination, so a missing or failed wheel leg fails the cut.
- The gate then runs `gate_release.sh` (which includes the same calibration
  attestation every PR passes), Python lint/types, the Python suite with
  its coverage floor, and builds one wheel into a fresh directory, installs it
  into a fresh venv and runs the full Python test suite against that installed
  wheel (outside `python/`, so the source tree cannot shadow it).

Each gate that decides a release has a `--self-test` mode that feeds it
deliberately broken input and requires a failure: `gate_composition.sh`,
`gate_parity_schema.sh`, `gate_docs_support_matrix.sh`,
`gate_release_candidate.sh` and `gate_calibration_attestation.sh`.
`gate_release.sh` runs all of them.

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

## Tests that matter

| Kind | Role |
|------|------|
| Unit / property | Algorithm invariants, graph witnesses, numeric edge cases |
| Conformance | Frozen fixtures vs expected outputs (`conformance/`) |
| Calibration | Coverage / Type I / null FPR (`gate_calibration.sh`, measured locally by `measure_calibration.sh`) |
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
`antecedent-prob/hmc`. `antecedent-prob/smc` is an empty feature that enables no
backend, and there is no `simd-runtime` feature, so `KernelPolicy::allow_arch_simd`
always selects the portable kernels. Ingest and exchange adapters are optional
features and never reshape core types.

## Unsafe / deps

Reviewed `unsafe` is concentrated in `antecedent-kernels` (SIMD), the
`antecedent-data` buffer/Arrow FFI adapters, and thin IO mmap.
New `unsafe` needs justification in review. Dependency and license policy:
[security_review.md](security_review.md), ADR 0008.

## Versions

Workspace and Python package version are kept in sync (currently **1.11.0**).
Artifact format is frozen separately — see [artifacts.md](artifacts.md).

MSRV: Rust 1.85, edition 2024. Python: CPython 3.11–3.14.

Keep `[workspace.package].version` in `Cargo.toml` and `version` in
`python/pyproject.toml` in sync:

```bash
bash scripts/set_version.sh X.Y.Z
```

`set_version.sh` also updates path-dependency pins under `crates/*/Cargo.toml`,
the Python fallback `__version__`, and the local package entry in
`python/uv.lock`, and freezes the previous cut's licensed-cell block in
`docs/release-notes/` so a later matrix regen cannot overwrite it. Refresh
`Cargo.lock` with `cargo update -p antecedent` (or a workspace check) before
committing. The generator rewrites live licensed-cell markers only in
`docs/release-notes/vX.Y.Z.md` for the current workspace version.

## Releases

Keep the changelog under **Unreleased** until a cut is approved and
its date is known. A package version bump is not proof that a release has been
published.

Before merging the release PR:

1. Commit the reviewed implementation, tests, and documentation with DCO sign-off.
   Keep the generated support/conformance output current and the worktree clean.
2. Run `cargo test --workspace`, strict all-target Clippy, Python tests against
   the rebuilt extension, Python lint/type checks, and `bash scripts/gate_release.sh`.
   `gate_release.sh` is the PR inventory / composition umbrella; a green
   local run is not an RC. Run `bash scripts/gate_codeql.sh` with the existing
   query configuration. Require CI on the same commit, including the Python
   lint/pytest and `python-wheels` jobs.
3. For an actual release cut, run
   `CI_RUN_ID=<run> bash scripts/gate_release_candidate.sh`. Every coverage
   record must be attested, exactly as on every PR: measured, in a run that
   passed, at a commit whose calibration surface matches HEAD in every facet the
   record depends on.
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

Before tagging, confirm the dated 1.10.0 changelog section is present, Unreleased
is empty, its comparison link is `v1.10.0...HEAD`, and release-status text matches
the cut. Tag only the approved, clean commit after these checks pass.
Do not remove the release gate's clean-diff check to accommodate pending edits.

Tagged releases drive wheel + docs publishing (GitHub Release assets and public
PyPI). The tag `vX.Y.Z` is the source of truth for the release build; CI runs
`scripts/set_version.sh` before maturin.

```bash
# Optional: bump and commit on main first
bash scripts/set_version.sh 1.10.0
cargo update -p antecedent
git add Cargo.toml Cargo.lock python/pyproject.toml python/uv.lock \
  python/antecedent/__init__.py crates/*/Cargo.toml fuzz/Cargo.lock \
  CHANGELOG.md CITATION.cff docs/release-notes/
git commit -s -m "chore: bump version to 1.10.0"

# Tag current (or just-bumped) version and push
CI_RUN_ID=<ci run on HEAD> bash scripts/tag_release.sh   # runs gate_release_candidate.sh
git push origin v1.10.0
```

Workflow [`.github/workflows/publish-release.yml`](https://github.com/iridae-dev/antecedent/blob/main/.github/workflows/publish-release.yml)
builds the full wheel matrix, then publishes to public PyPI and the GitHub
Release as **independent jobs**. Trusted publishing (`id-token: write`) must not
wait on GitHub asset uploads: a unicorn on `uploads.github.com` skipped PyPI
for 1.10.0 while crates.io (a separate workflow) succeeded. Release assets are
uploaded one file at a time with retries. If wheels already exist, dispatch
with `version` plus `reuse_run_id` set to the Actions run that built them —
do not rebuild a newer branch and stamp it as an older version.

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

Tag workflow [`.github/workflows/publish-crates.yml`](https://github.com/iridae-dev/antecedent/blob/main/.github/workflows/publish-crates.yml)
runs on `v*` tags (and `workflow_dispatch`) separately from the Python
`publish-release.yml` wheel pipeline. Set repository secret `CRATES_IO_TOKEN`.

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
5. Tag `v1.10.0` (or bump first) to cut wheels + PyPI (+ crates.io with token).
