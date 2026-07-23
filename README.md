# Antecedent

**Fast, explicit causal inference for Python and Rust.**

Discovery, identification, estimation, counterfactuals, attribution, and validation in one API—across tabular, temporal, panel, multi-environment, and event data.

```python
result = antecedent.analyze(
    data,
    graph=graph,
    query=antecedent.AverageEffect(
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

**Python** (CPython 3.11–3.14; wheels for Linux, macOS, Windows).

Private builds publish to **GitHub Packages** and as assets on each GitHub
Release (`vX.Y.Z`). Public PyPI is not used yet; the distribution name is
`antecedent`.

GitHub Packages (replace `OWNER` / use a PAT with `read:packages`):

```bash
# uv
export UV_INDEX_GITHUB_USERNAME=TOKEN
export UV_INDEX_GITHUB_PASSWORD=ghp_...   # or fine-scoped token
uv add antecedent --index https://pypi.pkg.github.com/OWNER/antecedent/simple/
```

```toml
# pyproject.toml (uv)
[[tool.uv.index]]
name = "github"
url = "https://pypi.pkg.github.com/OWNER/antecedent/simple/"
authenticate = "always"

[tool.uv.sources]
antecedent = { index = "github" }
```

```bash
# pip
pip install antecedent \
  --index-url https://OWNER:ghp_...@pypi.pkg.github.com/OWNER/antecedent/simple/
```

Or download the platform wheel from the Release assets and
`pip install ./antecedent-*.whl`.

**Rust** (1.85+, edition 2024) — facade on crates.io:

```bash
cargo add antecedent
```

```toml
antecedent = "0.1"
```

```rust
use antecedent::prelude::*;
```

Supporting crates (`causal-core`, `causal-graph`, …) publish alongside the facade
and are public dependencies of `antecedent`. The Python extension crate
(`causal-py`) is not on crates.io.

For a private checkout before crates.io mirrors catch up:

```toml
antecedent = { git = "ssh://git@github.com/iridae-dev/antecedent.git" }
```

See [docs/development.md](docs/development.md) for tagging releases and the
crates.io publish checklist.

## Python quick start

```python
import antecedent

result = antecedent.analyze(
    data,  # prefer PyArrow / Arrow CDI for interactive; pandas remains correct
    graph=[("z", "campaign"), ("z", "revenue"), ("campaign", "revenue")],
    query=antecedent.AverageEffect(
        treatment="campaign",
        outcome="revenue",
    ),
    inference=antecedent.Frequentist(),
    latency="interactive",  # optional: analytic/cheap path; pass Arrow for zero-copy
)

print(result.identification.status, result.estimate.ate)
```

Stages are available separately: `identification`, `estimate`, `posterior`, `validation`, `diagnostics`, `provenance`.

Temporal discovery then estimation:

```python
discovered = antecedent.discover_pcmci(
    names, columns, max_lag=12, alpha=0.05, seed=1
)
result = antecedent.analyze(
    {"pressure": pressure, "defect": defect},
    graph=[("pressure", 1, "defect", 0)],
    query=antecedent.PulseEffect(
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
use antecedent::prelude::*;
use antecedent::RefuteSuite;

fn main() -> Result<(), CausalError> {
    let schema = CausalSchemaBuilder::new()
        .continuous("campaign")
        .treatment()
        .continuous("revenue")
        .outcome()
        .continuous("z")
        .context()
        .build()?;
    let data = TabularData::from_f64_columns([
        ("campaign", campaign.as_slice()),
        ("revenue", revenue.as_slice()),
        ("z", z.as_slice()),
    ])?;
    let dag = Dag::from_named_edges(
        &schema,
        &[("z", "campaign"), ("z", "revenue"), ("campaign", "revenue")],
    )?;
    let t = schema.id_of("campaign")?;
    let y = schema.id_of("revenue")?;
    let ctx = ExecutionContext::for_tests(1);

    let result = CausalAnalysis::builder()
        .data(data)
        .graph(dag)
        .query(AverageEffectQuery::binary_ate(t, y))
        .inference(InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(1000)))
        .refute(RefuteSuite::None)
        .build()?
        .run(&ctx)?;

    if let Some(posterior) = &result.posterior {
        println!("P(effect < 0) = {}", posterior.probability_below(0.0)?);
    }
    println!("ATE point = {}", result.effect());
    Ok(())
}
```

`use antecedent::prelude::*` for day-1 imports. Prefer modules (`antecedent::discovery`, `antecedent::gcm`, `antecedent::io`, …) for stage depth — those are no longer re-exported at the crate root (0.1.x breaking). Examples: `cargo run -p antecedent --example ate_quickstart`.

## What it covers

| Area | Includes |
| --- | --- |
| Data | Tabular, time series, panel (multi-unit; J-PCMCI+ / clustered SE), multi-environment (J-PCMCI+ discover), events (duration-bin align → temporal); NumPy/pandas/Arrow CDI |
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
* [API naming (Rust ↔ Python)](docs/api_naming.md) · [Hot paths](docs/hot_paths.md) · [Conformance](docs/conformance/README.md) · [ADRs](adr/README.md)
* API docs: `docs.tar.gz` on each Release (markdown + rustdoc + Python pdoc); locally `cargo doc -p antecedent --open`
* [Examples](crates/antecedent/examples/) · [Python examples](python/examples/)

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). Algorithm work should include method basis, tests, calibration where relevant, a benchmark, and provenance. DCO sign-off required.

## License

MIT OR Apache-2.0, at your option.

```text
SPDX-License-Identifier: MIT OR Apache-2.0
```
