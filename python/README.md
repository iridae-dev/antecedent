# Python package for causal-library

Requires CPython 3.11–3.14 and a Rust 1.85 toolchain. CI builds and smoke-tests
wheels for that range on Linux x86_64/aarch64 (manylinux), macOS x86_64/arm64,
and Windows x86_64 (default `faer` path; no system BLAS).

```bash
cd python
uv venv && source .venv/bin/activate
uv sync --group dev
maturin develop
pytest
```

The `causal` package exposes a coarse FFI surface over the Rust facade:

- `analyze` / `analyze_ate` — static and temporal effect analysis
- `discover_pcmci*` / `discover_lpcmci` / … — temporal discovery (links + oriented graph edges)
- `gcm_counterfactual_ite` / `gcm_sample_do` — GCM counterfactuals and interventional draws
- `load_float64_columns` — DESIGN §25.6 conversion probe (same Arrow→tabular path as analysis)

Build artifacts (`_native.*.so`) are gitignored; always `maturin develop` (or install a wheel) on a fresh checkout.

Typed exceptions (`CausalError` and subclasses) mirror Rust `AnalysisError` categories.
Errors from argument-shape checks remain `ValueError`.
