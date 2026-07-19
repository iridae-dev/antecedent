# Security, licensing, unsafe-code, and dependency review 

Date: 2026-07-21 
Scope: workspace crates + `python` extension (package version **0.1.0**) 
ADR: [0017](../adr/0017-release-prep.md)

## Unsafe code policy

| Crate | Policy | Notes |
|-------|--------|-------|
| Most semantic crates (`causal-*` except below / kernels) | `#![forbid(unsafe_code)]` | Verified by `scripts/gate_release.sh` |
| `causal-data` | `#![deny(unsafe_code)]` + scoped `allow` | Foreign buffers (`buffer.rs`) and Arrow CDI (`arrow_ffi.rs`) |
| `causal-io` | `#![deny(unsafe_code)]` + scoped `allow` | Thin mmap (`mmap_file.rs`) only |
| `causal-kernels` | `#![allow(unsafe_code)]` | Only reviewed SIMD / aliasing kernels |
| `python` / `causal-py` | `#![allow(unsafe_code)]` | Required by PyO3 |

Gate fails if a forbid-crate loses `forbid(unsafe_code)`, or if data/io lose `deny` / their scoped escape modules.

## Licensing

- Project: `MIT OR Apache-2.0` (see `LICENSE-MIT`, `LICENSE-APACHE`, ADR 0008).
- Dependencies audited with **cargo-deny** (`deny.toml` license allow-list).
- Default features must remain wheel-distributable without system BLAS (ADR 0001 / ).

## Dependency notes

| Component | Role | Review |
|-----------|------|--------|
| `faer` | Default linear algebra | Pure Rust; no system BLAS in default wheels |
| `paste` (transitive via `gemm`) | faer build-time macro | Unmaintained (`RUSTSEC-2024-0436`); ignored in `deny.toml` with reason — no runtime use; revisit when faer drops it |
| `arrow-array` / `arrow-schema` / `arrow-buffer` | Tabular / IPC sections | Feature-gated where needed; no algorithm duplication in Python |
| `pyo3` 0.29 / `numpy` 0.29 | Python boundary | Upgraded in to clear `RUSTSEC-2025-0020` and `RUSTSEC-2026-0177` |
| `blake3` / `ciborium` / `serde` | Artifact container | CBOR + checksums per |
| `thiserror` | Error types | No runtime concerns |

Unmaintained / yanked crates: cargo-deny advisories job in CI (`yanked = "warn"`).

## Wheel purity

Default maturin wheels use the `faer` path and must not link system BLAS.
Optional `blas` features (if added later) are non-default.

## CodeQL

- CI: `.github/workflows/codeql.yml` (rust / python / actions, `build-mode: none`, `security-and-quality` suites via `.github/codeql/codeql-config.yml`).
- Local gate (requires `codeql` on `PATH`): `bash scripts/gate_codeql.sh` — fails unless all three SARIF reports have **0** findings.
- Third-party Actions in workflows are pinned to commit SHAs; workflows set explicit `permissions`.

## Evidence commands

```bash
# Unsafe forbid scan + inventory + benches (excerpt)
bash scripts/gate_release.sh

# License / advisory / source policy
cargo deny check

# CodeQL (0 findings)
bash scripts/gate_codeql.sh
```
