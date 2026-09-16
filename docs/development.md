# Development

Product name: **Antecedent**. Python distribution/import: `antecedent`. Rust
day-1 facade: `antecedent` (`cargo add antecedent`). Supporting crates are
`antecedent-*` on crates.io.

## CI vs local gates

GitHub Actions CI (`ci.yml`) runs the following checks on every PR:

- **`rust`** — fmt, clippy, `cargo test --workspace`, DCO (plus an optional
  crates.io publish dry-run when manifests change).
- **`gates`** — `scripts/gate_release.sh`, which runs the parity-manifest schema
  check, provenance and metadata checks, support-matrix and evidence checks,
  feature gates, artifact tests, and Criterion benchmark smokes.
- **`python-lint`** — Ruff, mypy, and pytest with an 85% coverage floor after
  building the native extension.
- **`python-wheels`** — builds and tests the supported wheel matrix.

CI does **not** run `gate_calibration.sh` per PR. Criterion benchmark smokes
do run through `gate_release.sh`; they check execution, not timing regressions.
The statistical calibration measurement has no schedule: coverage records stand
until the surface they measured changes (see [Coverage records](#coverage-records)),
so it runs on demand through
[`.github/workflows/calibration.yml`](https://github.com/iridae-dev/antecedent/blob/main/.github/workflows/calibration.yml)
(`workflow_dispatch`) and is owed only when the `gates` job reports a drifted
facet. `cargo deny` runs inside `gate_release.sh` only when `cargo-deny` is on
PATH, which it is not in CI.

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
bash scripts/gate_calibration.sh   # SE coverage / CI Type I — on demand, when records owe it
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

### When a record stands

The measurement is deterministic: every replicate's data and resamples come
from fixed seeds, so re-running unchanged code reproduces the same numbers
(`crates/antecedent-estimate/tests/coverage_determinism.rs` holds that in
ordinary CI). A record therefore stands until the code it measured changes,
and is never re-measured on a timer.

`scripts/calibration_surface.list` is the only owner of that code. It assigns
every path that can move a number (crate sources, manifests, the toolchain, the
shared harness, each calibration suite) to a facet:

- `core` — code every record depends on; a change invalidates every record.
  Any path under the surface that no narrower line claims is `core`, and a
  crate, manifest or record-emitting suite the list does not cover fails the
  gate, so an unmapped edit never looks harmless.
- narrower facets — `mechanism` (the fitted-SCM crates and the dispatch path
  that alone reaches them), `design`, one `suite.<file>` per calibration
  suite, and `registry_mirror` (generated tables no measurement reads). A
  change invalidates only the records carrying the facet.

Each record's `facets` are derived from the record itself by
`scripts/calibration_facets.py`: `core`, the facet of its test and DGP file,
facets the list's `key` lines assign to its query or estimator, and any facet
whose items its suite names. `scripts/gate_parity_schema.sh` rejects a record
whose facets are not exactly the derived set. A narrow facet is only sound
while no other code runs it, so `calibration_facets.py check` fails when a file
outside a facet names the facet's items, unless an `allow` line in the list
records that reviewed reference exactly (an error conversion, a result slot,
the query-keyed dispatch arm); a new reference or a stale entry fails.

A record is **attested** while none of its facets differs between its
`calibration_sha` and the tree. `scripts/gate_calibration_attestation.sh`
reports this on every PR (a `DRIFTED` facet names the changed paths and the
records that now owe a re-measurement) and requires it at a release cut.

### Measuring

Measure on a committed worktree; the collector refuses to stamp HEAD while
the surface its records depend on differs from HEAD:

```bash
python3 scripts/calibration_facets.py status          # which facets drifted, which records owe
python3 scripts/calibration_shards.py run all --only-stale   # or dispatch calibration.yml
python3 scripts/collect_coverage_records.py --keep-attested
```

`--only-stale` runs only the gate groups behind records that owe a
re-measurement, plus the pass/fail gates that emit no record (CI Type I,
discovery FPR, SBC), which nothing can attest. `--keep-attested` keeps every
existing record the logs did not re-measure whose facets have not drifted, at
its own `calibration_sha`, and drops (naming them) the ones that still owe.
Without it the registry is exactly the logs present. The collector keeps a
rechecked group's more precise run, rewrites the `calibration` /
`calibration_reason` pair on every licensed cell and estimator row, and
regenerates `crates/antecedent-io/src/coverage_records_data.rs`;
`scripts/gate_parity_schema.sh` then checks what those rows claim (`--sha
<commit>` collects logs measured at another commit). After editing
`scripts/calibration_surface.list`, `collect_coverage_records.py --retag`
recomputes the facets without measuring anything.

`calibration.yml` splits the gate across 24 runners with
`scripts/calibration_shards.py`: each long group (a Bayesian derivative cell,
or another whose 2000-replicate recheck has measured tens of minutes to hours)
gets a runner to itself and the short groups are dealt across the rest, the
same way on every runner. When the gate grows, the scheduler refuses a shard
count that no longer fits; raise the shard count, not the timeout.

Mark a `parity/*.toml` capability `done` only with conformance under `conformance/`
**or** a named harness in the gate script, plus a recorded reference-generation
command when black-box comparison applies.

Statuses: `pending` | `in_progress` | `done`. No waiver vocabulary.

## Release candidates

`gate_release.sh` is the PR inventory; a release is cut with
`scripts/gate_release_candidate.sh` (which `scripts/tag_release.sh` runs before
tagging). It needs two inputs:

```bash
REQUIRE_CALIBRATION_ATTESTATION=1 \
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
- **`REQUIRE_CALIBRATION_ATTESTATION=1`** makes `gate_calibration_attestation.sh`
  require every coverage record to be attested: each record's facets unchanged
  between its own `calibration_sha` (a commit that must be in the clone) and
  this tree. An unchanged surface passes with no re-measurement. A drifted
  facet fails, naming the changed paths and the records that owe a
  re-measurement.
- The gate then runs `gate_release.sh`, Python lint/types, the Python suite with
  its coverage floor, and builds one wheel into a fresh directory, installs it
  into a fresh venv and runs the full Python test suite against that installed
  wheel (outside `python/`, so the source tree cannot shadow it).

Each gate that decides a release has a `--self-test` mode that feeds it
deliberately broken input and requires a failure: `gate_composition.sh`,
`gate_parity_schema.sh`, `gate_docs_support_matrix.sh` and
`gate_release_candidate.sh`. `gate_release.sh` runs all of them.

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
| Calibration | Coverage / Type I / null FPR (`gate_calibration.sh`) |
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

Workspace and Python package version are kept in sync (currently **1.10.0**).
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
   `REQUIRE_CALIBRATION_ATTESTATION=1 CI_RUN_ID=<run> bash scripts/gate_release_candidate.sh`.
   Every coverage record must be attested: measured at a commit whose
   calibration surface matches HEAD in every facet the record depends on, in a
   run that passed. Everyday PRs do not run the 400-replicate gate; they report
   which records a change leaves owing a re-measurement.
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
REQUIRE_CALIBRATION_ATTESTATION=1 CI_RUN_ID=<ci run on HEAD> \
  bash scripts/tag_release.sh          # runs gate_release_candidate.sh
git push origin v1.10.0
```

Workflow [`.github/workflows/publish-release.yml`](https://github.com/iridae-dev/antecedent/blob/main/.github/workflows/publish-release.yml)
builds the full wheel matrix, attaches wheels + `docs.tar.gz` to the GitHub
Release, and publishes to public PyPI via trusted publishing (`id-token: write`).
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
