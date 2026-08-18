# Antecedent

Antecedent is an identification-first causal inference engine for Python (with
a native Rust core). It takes an analysis from causal structure through
estimation, diagnostics, interventions, and counterfactuals — without silently
treating discovered graphs as ground truth.

Use it when you need a causal effect you can defend: identification is checked
before anything is estimated, refuters run against the estimate by default, and
uncertainty about the causal graph — a CPDAG, a PAG, a posterior over graphs —
is carried through to the effect instead of being resolved by fiat.

Requires CPython 3.11–3.14. Install from PyPI:

```bash
pip install antecedent
```

Paste this block and run it. It simulates a confounded dataset, checks that the
effect is identified, estimates it, and runs refuters against the estimate:

```python
import numpy as np
from antecedent import AverageEffect, analyze

rng = np.random.default_rng(0)
n = 2000
season = rng.normal(size=n)                               # confounder
price = 0.7 * season + rng.normal(size=n)                 # treatment
sales = 1.5 * price + 2.0 * season + rng.normal(size=n)   # outcome, true effect = 1.5

result = analyze(
    data={"season": season, "price": price, "sales": sales},
    graph=[("season", "price"), ("season", "sales"), ("price", "sales")],
    query=AverageEffect(treatment="price", outcome="sales"),
)

print(result.identification)   # NonparametricallyIdentified via backdoor.adjustment, adjusting for season
print(result.estimate)         # ate=1.48, analytic and bootstrap standard errors
print(result.validation)       # refuters ran and passed
```

The same `analyze()` call scales from this to temporal pulse effects,
Bayesian graph-posterior mixtures that report unidentified structure mass,
mediation, counterfactuals, and root-cause attribution — see the
[project README](https://github.com/iridae-dev/antecedent#readme) for worked
examples and the [documentation](https://antecedent.readthedocs.io/) for the
full API.

## Development

For local development you need a Rust 1.85 toolchain. CI builds and smoke-tests
wheels for the supported CPython range on Linux x86_64/aarch64 (manylinux),
macOS arm64, and Windows x86_64 (default `faer` path; no system BLAS). Tagged
releases publish wheels to PyPI and GitHub Release assets (see
[docs/development.md](https://github.com/iridae-dev/antecedent/blob/main/docs/development.md)).

```bash
cd python
uv venv && source .venv/bin/activate
uv sync --group dev
maturin develop
pytest
```

Lint and types (local gate only — not part of wheel CI):

```bash
bash ../scripts/gate_python_lint.sh   # ruff check/format + mypy
```

## Public API

Primary entry point is the OO facade:

```python
import antecedent

g = antecedent.Dag.from_edges(["z", "t", "y"], [("z", "t"), ("z", "y"), ("t", "y")])
result = antecedent.analyze(
    data,  # dict[str, array] or pandas DataFrame
    graph=g,  # or an edge list
    query=antecedent.AverageEffect(treatment="t", outcome="y"),
    inference=antecedent.Frequentist(),  # or antecedent.Bayesian(...)
)
print(result.identification, result.estimate, result.validation)

# Identify without estimating:
id_only = antecedent.identify(
    graph=g,
    query=antecedent.AverageEffect(treatment="t", outcome="y"),
    identifier=antecedent.Identifier.BACKDOOR_ADJUSTMENT,
)

gcm = antecedent.model.fit_gcm(["z", "t", "y"], columns, list(g.edges()))
draws = gcm.sample_do({"t": 1.0}, n=200)

# Or discover then fit (never invents orientations; refuses incomplete PAG/FCI):
fitted, edges = antecedent.gcm.fit_gcm_discovered(
    data, discovery=antecedent.discovery.PC(alpha=0.05)
)
```

The root namespace is frozen at 49 names for 0.5; everything else is reached through a
stage module (`antecedent.discovery`, `antecedent.priors`, `antecedent.errors`, …).

Also exposed:

- Typed graphs: `Dag` / `Cpdag` / `Pag` / `Admg` / `TemporalDag`
  (`d_separated` / `latent_project` on `Dag`; `m_separated` on `Admg` / `Pag`)
- `Identifier` / `Estimator` enums (wire ids) plus string kwargs
- `identify(graph=…, query=AverageEffect(…))` — identify without estimating
- Queries: `AverageEffect`, `MediationEffect`, `Counterfactual`, `PulseEffect`, `SustainedEffect`,
  `InterventionalDistribution`, `PathSpecificEffect`, `ConditionalEffect`,
  `TemporalMediationEffect`
- `antecedent.discovery` — PC, GES, LiNGAM, NOTEARS, FCI/RFCI, PCMCI family, Bayesian posteriors
- `antecedent.validation.validate_pcmci_*` — discovery stability (block bootstrap, FPR, grids, …)
- `antecedent.model` / `antecedent.counterfactual` — `FittedGcm`, `sample_do`, `counterfactual_ite`
- `antecedent.population` — `PopulationRegistry` / `target_*` for named predicates and custom-distribution IPW
- `antecedent.gcm` — `fit_gcm_discovered` / `attribute_*_discovered` discover-then-attribute composition
- `antecedent.state.CausalState` — incremental state with retained batches, events, suff-stats, particle filter
- `refute=True|"full"|"placebo"|False` on static and temporal `analyze`
- RD: `estimator="rd.sharp"` with `running_variable` / `cutoff` / `bandwidth`
- Graph interchange on the classes: `Dag.from_dot` / `.to_dot` and the JSON / GML / NetworkX peers
- Design / state examples: [`examples/python/rank_designs.py`](https://github.com/iridae-dev/antecedent/blob/main/examples/python/rank_designs.py),
  [`examples/python/causal_state_workflow.py`](https://github.com/iridae-dev/antecedent/blob/main/examples/python/causal_state_workflow.py)
  (see ADR 0016 — no auto-rerun); catalog in [`examples/README.md`](https://github.com/iridae-dev/antecedent/blob/main/examples/README.md)

Build artifacts (`_native.*.so`) are gitignored; always `maturin develop` (or install a wheel) on a fresh checkout.

Typed exceptions (`CausalError` and subclasses) mirror Rust `CausalError` categories.
The native module `antecedent._native` remains available for advanced FFI use
(including the flat `AteAnalysisResult` DTO; prefer nested `AnalysisResult`).
