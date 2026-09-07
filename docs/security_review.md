# Security, licensing, unsafe-code, and dependency review

Date: 2026-09-07
Scope: workspace crates + `python` extension (package version **1.1.0**)
ADR: [0017](../adr/0017-release-prep.md)

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
  ([ADR 0001](../adr/0001-linear-algebra-backend.md)).

## Dependency notes

| Component | Role | Review |
|-----------|------|--------|
| `faer` | Default linear algebra | Pure Rust; no system BLAS in default wheels |
| `paste` (transitive via `gemm`) | faer build-time macro | Unmaintained (`RUSTSEC-2024-0436`); ignored in `deny.toml` with reason — no runtime use; revisit when faer drops it |
| `arrow-array` / `arrow-schema` / `arrow-buffer` | Tabular / IPC sections | Feature-gated where needed; no algorithm duplication in Python |
| `pyo3` 0.29 / `numpy` 0.29 | Python boundary | Current Python bindings; the lockfile passes the advisory policy in `deny.toml` |
| `blake3` / `ciborium` / `serde` | Artifact container | CBOR + checksums under the format-0.4 artifact contract |
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
