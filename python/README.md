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

g = causal.Dag.from_edges(["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")])
result = causal.analyze(
    data,  # dict[str, array] or pandas DataFrame
    graph=g,  # or an edge list
    query=causal.AverageEffect(treatment="t", outcome="y"),
    inference=causal.Frequentist(),  # or causal.Bayesian(...)
)
print(result.identification, result.estimate, result.validation)

gcm = causal.fit_gcm(["z", "t", "y"], columns, list(g.edges()))
draws = gcm.sample_do({"t": 1.0}, n=200)

# Or discover then fit (never invents orientations; refuses incomplete PAG/FCI):
fitted, edges = causal.fit_gcm_discovered(data, discovery=causal.PC(alpha=0.05))
```

Also exposed:

- Typed graphs: `Dag` / `Cpdag` / `Pag` / `Admg` / `TemporalDag`
- `discover_*` (PC, GES, LiNGAM, NOTEARS, FCI/RFCI, PCMCI family, Bayesian posteriors)
- `validate_pcmci_*` / discovery stability validators (block bootstrap, FPR, grids, …)
- `FittedGcm` / `counterfactual_ite` / `sample_do` — GCM counterfactuals
- `PopulationRegistry` / `target_*` helpers for named predicates and custom-distribution IPW
- `fit_gcm_discovered` / `attribute_*_discovered` — discover-then-attribute composition
- `CausalState` — incremental state with retained batches, events, suff-stats, and particle filter
- Queries: `AverageEffect`, `PulseEffect`, `SustainedEffect`,
  `InterventionalDistribution`, `PathSpecificEffect`, `ConditionalEffect`,
  `TemporalMediationEffect`
- `refute=True|"full"|"placebo"|False` on static and temporal `analyze`
- RD: `estimator="rd.sharp"` with `running_variable` / `cutoff` / `bandwidth`
- `dag_from_*` / `dag_to_*` — graph interchange
- Design / state examples: [`examples/rank_designs.py`](examples/rank_designs.py),
  [`examples/causal_state_workflow.py`](examples/causal_state_workflow.py)
  (see ADR 0016 — no auto-rerun)

Build artifacts (`_native.*.so`) are gitignored; always `maturin develop` (or install a wheel) on a fresh checkout.

Typed exceptions (`CausalError` and subclasses) mirror Rust `AnalysisError` categories.
The native module `causal._native` remains available for advanced FFI use.
