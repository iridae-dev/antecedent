# Security, licensing, unsafe-code, and dependency review

Date: 2026-09-14
Scope: workspace crates + `python` extension (package version **1.9.0**)
ADR: [0017](https://github.com/iridae-dev/antecedent/blob/main/adr/0017-release-prep.md)

The 1.9.0 source diff (against the 1.8.0 cut) changes estimators, calibration
tests, the Python facade, and the artifact format (0.5 adds an optional
identified-set interval on structural-mixture analysis results; 0.4 artifacts
migrate unchanged through the existing bounded decoder). Review of the diff
found no new `unsafe` block and no workflow change. Its only manifest change
adds `arrow-array` and `arrow-schema`, already workspace dependencies, as
dev-dependencies of the `antecedent` crate; the lockfile gains no package. The
Arrow C Data Interface import now borrows a Float64 column with nulls only when
its null slots already hold NaN, and copies otherwise. An offline
`cargo deny --offline check` against the cached advisory database passed
advisories, bans, licenses, and sources. This is a source review, not a fresh
advisory refresh or CodeQL run.

The 1.8.0 source diff adds Bayesian functional evaluation, weighted statistical
fits, mechanism composition, and prior-transfer arguments to existing prepared
entry points. Review of this diff found no new external dependency, unsafe
block, workflow permission, or artifact decoder. Artifact input continues
through the existing bounded posterior decoder. Weighted numerical APIs
validate shape, finiteness, and weight totals; posterior loops retain
cancellation checks. This source review does not constitute a fresh advisory
refresh or a new CodeQL run; dated tool results below retain their original scope.
The 1.8.0 offline cargo-deny check passed advisories, bans, licenses, and
sources using the unchanged policy and a writable copy of the cached advisory
database. The standard release wrapper cannot acquire a lock in the read-only
default advisory-cache directory in this environment.


This review was re-run against the 0.9.1 cut, including the workspace unsafe-
code policy, the current lockfile's advisory/license/source rules, default
feature linkage, and workflow permissions/action pins. 0.9.1 is a behavior
patch: it adds no dependencies, changes no lockfile entry beyond the workspace
version bump, and introduces no `unsafe` block, so the 0.9.0 findings carry
forward unchanged. 1.0.0 is the version stamp of that 0.9.1 tree. The 1.1.0
diff adds prepared-identification caches, conformance fixtures, and four
additive PyO3 entry points for supplying a constructed graph posterior
(`GraphPosterior.from_atoms`, `analyze_ate_graph_posterior`,
`analyze_temporal_graph_posterior`, and a `posterior=` argument on the
prepared graph-posterior constructors). Those entry points validate atom
masks, weights, and lag packing and bind the posterior to the data schema
before it reaches the engine. There is no dependency, unsafe-code,
artifact-decoding, or workflow change, and the lockfile diff is limited to
workspace package versions, so the existing threat boundary and scoped unsafe
findings continue to apply.


The 1.2.0 diff adds safe Rust statistical composition and diagnostic code plus
additive PyO3 prepared arguments and artifact export. `antecedent-estimate`
now uses the existing workspace `antecedent-graph` crate at runtime for finite
DAG unfolding; it previously depended on that crate only in tests. The existing `serde_json` dependency enables `float_roundtrip` so decimal
JSON crossings retain exact binary float values. No external dependency,
unsafe block, artifact decoder, or workflow permission is added.
Artifact export uses the existing bounded canonical writers. New provenance
cards record independent implementations and their statistical restrictions.

The 1.4.0 diff adds class-preserving graph completion and estimation routes
and Python handoff APIs. Its lockfile changes are workspace version updates;
no external dependency, unsafe block, or artifact decoder was added. Temporal
handoffs export certified offsets and trim series boundaries; incomplete
graph results retain their identification limitations. The local CodeQL gate passed on
2026-09-10 with zero Rust, Python, and Actions findings under the existing
documented query exclusions. A redundant test import was removed and Python
rescanned before the final combined findings audit passed.

The 1.5.0 diff adds retargetable score tables, exceedance functionals,
cell-saturated joint AIPW, tier-background identification as a fast path, and
joint influence-function mixing for static envelopes. It adds no external
dependency, unsafe block, artifact decoder, or workflow permission. New Python
entry points (`analyze_ate_tiered`, `PreparedAnalysis.retarget`, outcome
functional kwargs) validate declared `depends_on` names and refuse weights that
depend on treatment. Score-table payloads reuse the existing bounded artifact
writers.

The 1.6.0 diff adds temporal identification caches, sequential g-computation,
observation-adjusted temporal response, DBN-posterior execution, and prior-bank
metadata. Lockfile changes are workspace version updates; no external
dependency, unsafe block, or workflow permission was added. Prior metadata uses
the existing checksummed, size-bounded artifact container; the catalog API
filters incompatible transfer metadata before the caller requests hydration.

The 1.7.0 diff adds caller-supplied class priors, class-preserving Bayesian
temporal execution on `TemporalCpdag` / `TemporalPag`, a completion-search cap,
and shared numerical corrections in `antecedent-stats`, `antecedent-prob`, and
`antecedent-validate`. Lockfile changes are workspace version updates only; no
external dependency, unsafe block, artifact decoder, or workflow permission was
added. Class priors are validated (finite, nonnegative, positive total, no
duplicate keys) before any arithmetic, and structural atoms reuse the existing
checksummed, size-bounded posterior artifact container.

The 1.8.0 diff adds query-aware Bayesian remainder execution (functional
row-law evaluator, mechanism-posterior mediation, weighted GCM
counterfactuals, Riesz/point/GAM derivatives, ConditionalEffect
graph-posterior envelopes) and prior-transfer hydrate metadata
(`treatment_contrast`). Lockfile changes are workspace version updates
only; no external dependency, unsafe block, artifact decoder, or workflow
permission was added. Mapped priors fail closed without a declared
mapping; `n_draws < 2` is a typed refuse.

## Unsafe code policy

| Crate | Policy | Notes |
|-------|--------|-------|
| Most semantic crates (`antecedent-*` except below / kernels) | `#![forbid(unsafe_code)]` | Verified locally by `scripts/gate_release.sh` |
| `antecedent-data` | `#![deny(unsafe_code)]` + scoped `allow` | Foreign buffers (`buffer.rs`) and Arrow CDI (`arrow_ffi.rs`) |
| `antecedent-io` | `#![deny(unsafe_code)]` + scoped `allow` | Thin mmap (`mmap_file.rs`) only |
| `antecedent-kernels` | `#![allow(unsafe_code)]` | Only reviewed SIMD / aliasing kernels |
| `python` / `antecedent-py` | `#![allow(unsafe_code)]` | Required by PyO3 |

Gate fails if a forbid-crate loses `forbid(unsafe_code)`, or if data/io lose `deny` / their scoped escape modules.

## Licensing

- Project: `MIT OR Apache-2.0` (see `LICENSE-MIT`, `LICENSE-APACHE`, ADR 0008).
- Dependencies audited with **cargo-deny** (`deny.toml` license allow-list); run
  locally (`cargo deny check`) — not part of CI.
- Default features must remain wheel-distributable without system BLAS
  ([ADR 0001](https://github.com/iridae-dev/antecedent/blob/main/adr/0001-linear-algebra-backend.md)).

## Dependency notes

| Component | Role | Review |
|-----------|------|--------|
| `faer` | Default linear algebra | Pure Rust; no system BLAS in default wheels |
| `paste` (transitive via `gemm`) | faer build-time macro | Unmaintained (`RUSTSEC-2024-0436`); ignored in `deny.toml` with reason — no runtime use; revisit when faer drops it |
| `arrow-array` / `arrow-schema` / `arrow-buffer` | Tabular / IPC sections | Feature-gated where needed; no algorithm duplication in Python |
| `pyo3` 0.29 / `numpy` 0.29 | Python boundary | Current Python bindings; the lockfile passes the advisory policy in `deny.toml` |
| `blake3` / `ciborium` / `serde` | Artifact container | CBOR + checksums under the format-0.5 artifact contract |
| `thiserror` | Error types | No runtime concerns |

`cargo deny check` passed on 2026-09-07: advisories, bans, licenses, and
sources were all `ok`. Its configured warning-level duplicate dependency and
unused license-allowance reports remain non-failing maintenance signals;
`yanked = "warn"` is unchanged in `deny.toml`.

## Wheel purity

Default maturin wheels use the `faer` path and must not link system BLAS.
Optional `blas` features (if added later) are non-default.

## CodeQL

- **CI:** `.github/workflows/codeql.yml` runs on every push to `main`, every pull
  request, and weekly (Mon 04:00 UTC). Uses the same
  `.github/codeql/codeql-config.yml` and `security-and-quality` suites as below.
- **Local strict gate** (requires `codeql` on `PATH`): `bash scripts/gate_codeql.sh`
  — not wired into Actions; fails unless rust / python / actions SARIF reports have
  **0** findings (same config and query filters as CI). Run before release or when
  touching security-sensitive surfaces.
- Third-party Actions in workflows are pinned to commit SHAs; workflows set explicit `permissions`.

## Published surface

| Surface | Destination | Notes |
|---------|-------------|--------|
| Rust facade `antecedent` + `antecedent-*` library crates | crates.io | Tag workflow `publish-crates.yml`; see `scripts/publish_crates.sh` |
| Python package `antecedent` (PyO3 crate `antecedent-py`) | PyPI / GitHub Release assets | **Not** on crates.io (`publish = false`) |

## Evidence commands

```bash
# Local slow path: unsafe forbid scan + inventory + benches (+ optional deny)
bash scripts/gate_release.sh

# License / advisory / source policy (optional local)
cargo deny check

# CodeQL (strict local gate — 0 findings; CI uses .github/workflows/codeql.yml)
bash scripts/gate_codeql.sh
```

## 1.3.0 boundary review

The added estimators use safe Rust and existing linear algebra. Conditional Cox IPCW validates input dimensions, numeric covariates, event indicators, convergence, information rank and survival positivity. Nested counterfactuals remain refused. The additive `static_result` artifact uses the existing bounded CBOR container and validates its query, identification and numeric fields. No new runtime dependency or workflow permission is introduced. R survival is an executing test oracle only; no upstream source or executable is bundled.
