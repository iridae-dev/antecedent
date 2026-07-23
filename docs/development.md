# Development

Product name: **Antecedent**. Python distribution/import: `antecedent`. Rust
day-1 facade: `antecedent` (`cargo add antecedent`). Supporting crates are
`antecedent-*` on crates.io.

## CI vs local gates

GitHub Actions Rust CI (`ci.yml`) runs **fmt**, **clippy**, **`cargo test --workspace`**,
and **DCO** (plus an optional crates.io publish dry-run when manifests change).
It does **not** run feature gates, `gate_release.sh`, Criterion smokes, or
`cargo deny`.

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
bash scripts/gate_calibration.sh   # SE coverage / CI Type I — weekly / pre-release
bash scripts/gate_release.sh       # prior gates + inventory + benches + optional deny
```

Mark a `parity/*.toml` capability `done` only with conformance under `conformance/`
**or** a named harness in the gate script, plus a recorded reference-generation
command when black-box comparison applies.

Statuses: `pending` | `in_progress` | `done`. No waiver vocabulary.

## Tests that matter

| Kind | Role |
|------|------|
| Unit / property | Algorithm invariants, graph witnesses, numeric edge cases |
| Conformance | Frozen fixtures vs expected outputs (`conformance/`) |
| Calibration | Coverage / Type I / null FPR (`gate_calibration.sh`) |
| Cross-language | Python bindings exercise the same semantics |
| Criterion benches | Designated hot paths; local `gate_release` / bench smokes |
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
`antecedent-prob/hmc`. Reserved / unfinished: `smc`, `simd-runtime`. Optional ingest
and exchange adapters may land later without reshaping core types.

## Unsafe / deps

Reviewed `unsafe` is concentrated in `antecedent-kernels` (SIMD) and thin IO mmap.
New `unsafe` needs justification in review. Dependency and license policy:
[security_review.md](security_review.md), ADR 0008.

## Versions

Packages stay at **0.1.0** until an explicit 1.0 decision (ADR 0017). Artifact
format is frozen separately — see [artifacts.md](artifacts.md).

MSRV: Rust 1.85, edition 2024. Python: CPython 3.11–3.14.

Keep `[workspace.package].version` in `Cargo.toml` and `version` in
`python/pyproject.toml` in sync:

```bash
bash scripts/set_version.sh X.Y.Z
```

## Releases

Tagged releases drive wheel + docs publishing (GitHub Release assets and public
PyPI). The tag `vX.Y.Z` is the source of truth for the release build; CI runs
`scripts/set_version.sh` before maturin.

```bash
# Optional: bump and commit on main first
bash scripts/set_version.sh 0.1.0
git add Cargo.toml python/pyproject.toml
git commit -m "chore: bump version to 0.1.0"

# Tag current (or just-bumped) version and push
bash scripts/tag_release.sh          # or: bash scripts/tag_release.sh 0.1.0
git push origin v0.1.0
```

Workflow [`.github/workflows/publish-release.yml`](../.github/workflows/publish-release.yml)
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

Tag workflow [`.github/workflows/publish-crates.yml`](../.github/workflows/publish-crates.yml)
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
5. Tag `v0.1.0` (or bump first) to cut wheels + PyPI (+ crates.io with token).
