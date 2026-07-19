# Causal

**Fast, explicit causal inference for Python and Rust.**

Discovery, identification, estimation, counterfactuals, attribution, and validation in one API—across tabular, temporal, panel, and multi-environment data.

```python
result = causal.analyze(
    data,
    graph=graph,
    query=causal.AverageEffect(
        treatment="price",
        outcome="demand",
    ),
)

print(result.identification)
print(result.estimate)
print(result.validation)
```

Identify first. Estimate second. Estimators never silently choose confounders or invent a number for an unidentified query.

## Installation

**Python** (CPython 3.11–3.14; wheels for Linux, macOS, Windows):

```bash
pip install causal
```

**Rust** (1.85+, edition 2024):

```bash
cargo add causal
```

## Python quick start

```python
import causal

result = causal.analyze(
    data,  # dict[str, ndarray] or pandas DataFrame
    graph=[("z", "campaign"), ("z", "revenue"), ("campaign", "revenue")],
    query=causal.AverageEffect(
        treatment="campaign",
        outcome="revenue",
    ),
    inference=causal.Frequentist(),
)

print(result.identification.status, result.estimate.ate)
```

Stages are available separately: `identification`, `estimate`, `posterior`, `validation`, `diagnostics`, `provenance`.

Temporal discovery then estimation:

```python
discovered = causal.discover_pcmci(
    names, columns, max_lag=12, alpha=0.05, seed=1
)
result = causal.analyze(
    {"pressure": pressure, "defect": defect},
    graph=[("pressure", 1, "defect", 0)],
    query=causal.PulseEffect(
        treatment="pressure",
        outcome="defect",
        active_level=-0.03,
        treatment_lag=1,
        horizon_steps=1,
    ),
)
```

## Rust quick start

```rust
use causal::prelude::*;

fn main() -> Result<(), CausalError> {
    let ctx = ExecutionContext::for_tests(1);
    let t = schema.id_of("campaign")?;
    let y = schema.id_of("revenue")?;

    let result = CausalAnalysis::builder()
        .data(tabular)
        .graph(dag)
        .query(AverageEffectQuery::binary_ate(t, y))
        .inference(InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(1000)))
        .build()?
        .run(&ctx)?;

    if let Some(posterior) = &result.posterior {
        println!("P(effect < 0) = {}", posterior.probability_below(0.0)?);
    }
    println!("ATE point = {}", result.estimate.ate);
    Ok(())
}
```

`use causal::prelude::*` for the facade. Lower crates expose graph, discovery, identification, estimation, model, and validation components.

## What it covers

| Area | Includes |
| --- | --- |
| Data | Tabular, time series, panel, multi-environment, events; NumPy/pandas/Arrow/Polars |
| Graphs | DAG, ADMG, CPDAG, PAG, temporal; d/m-separation, projection, interventions |
| Discovery | Constraint- and score-based, NOTEARS, PCMCI family, Bayesian search; evidence retained |
| Identification | Backdoor, front-door, IV, ID/IDC, mediation, partial ID, temporal |
| Estimation | Adjustment, matching, weighting, DR, IV, RD; frequentist and Bayesian |
| SCMs / CFs | Interventions, abduction–action–prediction, trajectories |
| Attribution | Anomaly, distribution/mechanism change, path-specific, Shapley, root cause |
| Validation | Placebos, sensitivity, overlap, graph/discovery stability, PPC |

## Design rules

* **Identification ≠ estimation** — the estimand is decided before fitting.
* **Priors ≠ nonparametric ID** — Bayesian restrictions are recorded as such.
* **Graph ≠ parameter uncertainty** — kept separate in results.
* **Discovery is evidence** — not auto-promoted to ground truth.
* **Static ≠ temporal graphs** — lag/time-index semantics are explicit.

Hot paths run in Rust (batched APIs, reusable workspaces, optimized kernels). Results keep assumptions, diagnostics, and provenance. Artifacts use versioned serialization, not internal struct dumps.

## Documentation

* [Architecture](docs/architecture.md) · [Development](docs/development.md) · [Artifacts](docs/artifacts.md)
* [Hot paths](docs/hot_paths.md) · [Conformance](docs/conformance/README.md) · [ADRs](adr/README.md)
* [Rust API](https://docs.rs/causal) · [Examples](examples/)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Algorithm work should include method basis, tests, calibration where relevant, a benchmark, and provenance. DCO sign-off required.

## License

MIT OR Apache-2.0, at your option.

```text
SPDX-License-Identifier: MIT OR Apache-2.0
```
