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

## Public API

Primary entry point is the OO facade:

```python
import causal

result = causal.analyze(
    data,  # dict[str, array] or pandas DataFrame
    graph=[("z", "t"), ("z", "y"), ("t", "y")],
    query=causal.AverageEffect(treatment="t", outcome="y"),
    inference=causal.Frequentist(),  # or causal.Bayesian(...)
)
print(result.identification, result.estimate, result.validation)
```

Also exposed:

- `discover_pcmci*` / `discover_lpcmci` / … — temporal discovery
- `counterfactual_ite` / `sample_do` — GCM counterfactuals and interventional draws
- `dag_from_*` / `dag_to_*` — graph interchange
- `load_float64_columns` — DESIGN §25.6 conversion probe

Build artifacts (`_native.*.so`) are gitignored; always `maturin develop` (or install a wheel) on a fresh checkout.

Typed exceptions (`CausalError` and subclasses) mirror Rust `AnalysisError` categories.
The native module `causal._native` remains available for advanced FFI use.
