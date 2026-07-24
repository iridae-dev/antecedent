# Antecedent

Antecedent is an identification-first causal inference engine for Python and Rust. It takes an analysis from causal structure through estimation, diagnostics, interventions, and counterfactuals — without silently treating discovered graphs as ground truth.

Most causal tooling asks you to supply the true DAG and then conditions every downstream number on it, even though the graph is usually the least certain input you have. Antecedent is built for **causal inference under structural uncertainty**: discovered structure — a CPDAG, a PAG, a posterior over graphs — is treated as evidence about the causal graph, and that uncertainty is propagated into identification, estimation, and effect intervals rather than resolved by fiat.

Three rules are enforced throughout:

* identification is evaluated before estimation;
* priors and parametric assumptions do not upgrade nonparametric identification;
* uncertainty about causal structure is retained rather than silently resolved.

Everything else in the library supports that position:

* **One engine, whole workflow.** Discovery, identification, estimation, Bayesian inference, interventions, counterfactuals, attribution, validation, and experimental design share one API and one set of guarantees — assumptions are not lost at the seams between libraries.
* **Temporal and online analysis.** Temporal graphs with their own semantics, PCMCI-family discovery, temporal identification and estimation, and incremental `CausalState` for streaming workflows.
* **A Rust core.** The scientific engine is written in Rust with a first-class Python API, so the same computation runs in notebooks and in production.

## Quick start

```bash
pip install antecedent
```

Paste this block and run it. It simulates a confounded dataset, checks that the effect is identified, estimates it, and runs refuters against the estimate:

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

print(result.identification)
print(result.estimate)
print(result.validation)
```

```text
IdentificationView(status='NonparametricallyIdentified', method='backdoor.adjustment', adjustment_set=['season'], assumption_count=1, derivation_step_count=2)
EstimateView(ate=1.4809859501787295, se_analytic=0.0222..., se_bootstrap=0.0215..., estimator_id='linear.adjustment.ate', method='backdoor.adjustment', ...)
ValidationView(passed=True, ran=True, count=2, ...)
```

The corresponding Rust interface uses `CausalAnalysis::builder()`:

```rust
use antecedent::prelude::*;

let result = CausalAnalysis::builder()
    .data(data)
    .graph(graph)
    .query(query)
    .build()?
    .run(&ctx)?;
```

```bash
cargo add antecedent
```

An analysis can include:

* identification status and assumptions;
* an estimand and applicable identification strategies;
* a frequentist estimate or Bayesian posterior;
* uncertainty across compatible graphs;
* diagnostics, refuters, and sensitivity analyses;
* provenance and serialized artifacts.

## See it work

### Estimate an intervention effect from tabular data

Treatment assignment depends on a confounder, so the naive contrast is biased. Adjusting for the graph recovers the true effect of 2:

```python
import numpy as np
from antecedent import AverageEffect, analyze

rng = np.random.default_rng(11)
n = 1200
severity = rng.normal(size=n)                                   # confounder
treated = (rng.random(n) < 1 / (1 + np.exp(0.4 - 0.9 * severity))).astype(float)
recovery = 2.0 * treated + severity + 0.4 * rng.normal(size=n)  # true effect = 2

result = analyze(
    data={"treated": treated, "recovery": recovery, "severity": severity},
    graph=[("severity", "treated"), ("severity", "recovery"), ("treated", "recovery")],
    query=AverageEffect(treatment="treated", outcome="recovery"),
    estimator="propensity.weighting",
    seed=11,
)
naive = recovery[treated == 1].mean() - recovery[treated == 0].mean()
print(f"naive difference in means: {naive:.2f}")
print(f"adjusted causal effect:    {result.ate:.2f}  (se {result.estimate.se_bootstrap:.2f})")
```

```text
naive difference in means: 2.73
adjusted causal effect:    2.02  (se 0.03)
```

### Discover a graph while retaining structural uncertainty

On linear-Gaussian data, PC recovers the skeleton but cannot orient a single edge — every DAG in the equivalence class fits equally well. Antecedent will not pick an orientation for you. Either review the graph explicitly (`AcceptedGraph`), or propagate the uncertainty through a posterior over graphs:

```python
import numpy as np
import antecedent
from antecedent import AverageEffect, Bayesian, ExactDagPosterior, analyze

rng = np.random.default_rng(7)
n = 2000
z = rng.normal(size=n)
t = 0.8 * z + rng.normal(size=n)
y = 1.5 * t + z + rng.normal(size=n)
data = {"z": z, "t": t, "y": y}

pc = antecedent.discover_pc(data, alpha=0.05)
print(f"edges oriented by PC: {pc.cpdag_directed_edges} directed, {pc.cpdag_undirected_edges} undirected")

result = analyze(
    data=data,
    discovery=ExactDagPosterior(),
    query=AverageEffect(treatment="t", outcome="y"),
    inference=Bayesian(n_draws=200, backend="conjugate"),
    seed=7,
)
p = result.posterior
print(f"effect posterior over compatible graphs: {p.effect_mean:.2f} ± {p.effect_sd:.2f}")
print(f"posterior mass on graphs where the effect is unidentified: {p.unidentified_mass:.2f}")
```

```text
edges oriented by PC: 0 directed, 3 undirected
effect posterior over compatible graphs: 1.83 ± 0.02
posterior mass on graphs where the effect is unidentified: 0.50
```

Half of the structure posterior cannot identify this effect at all. That mass is reported alongside the estimate rather than renormalized away — the number tells you how much of what you believe about the structure the estimate silently ignores.

### Diagnose a temporal mechanism

A manufacturing line where pressure drives the defect rate one step later. The temporal graph uses lagged edges, identification goes through the unfolded temporal backdoor, and the query asks what a one-step pressure pulse does to defects:

```python
import numpy as np
from antecedent import PulseEffect, analyze

rng = np.random.default_rng(42)
n = 400
pressure = rng.normal(size=n)
defect = np.zeros(n)
for t in range(1, n):
    defect[t] = 0.9 * pressure[t - 1] + 0.1 * rng.normal()

result = analyze(
    data={"pressure": pressure, "defect": defect},
    graph=[("pressure", 1, "defect", 0)],   # pressure at lag 1 → defect now
    query=PulseEffect(
        treatment="pressure", outcome="defect",
        treatment_lag=1, horizon_steps=1, active_level=1.0,
    ),
    seed=42,
)
print(result.identification)
print(f"one-step effect of a pressure pulse on defect rate: {result.ate:.3f}")
```

```text
IdentificationView(status='NonparametricallyIdentified', method='temporal.backdoor.unfolded', adjustment_set=[], assumption_count=2, derivation_step_count=3)
one-step effect of a pressure pulse on defect rate: 0.895
```

Longer, runnable versions of these and more live in [`python/examples/`](python/examples/) — discover-once-then-estimate-many with `AcceptedGraph`, sequential Bayesian analysis with prior transfer, propensity weighting with overlap diagnostics, experimental-design ranking, and streaming `CausalState` workflows — and in [`crates/antecedent/examples/`](crates/antecedent/examples/) for Rust.

## Workflow

```text
data
  │
  ├── discover
  │     └── DAG · CPDAG · PAG · temporal graph · graph posterior
  │
  └── graph
        ├── identify
        ├── estimate
        ├── infer posterior
        ├── intervene
        ├── evaluate counterfactuals
        ├── attribute change
        ├── validate
        └── rank experimental designs
```

Discovery results are treated as evidence about causal structure. They are not automatically treated as the true graph.

## Queries

Antecedent uses typed causal queries rather than a string query language.

| Query                        | Purpose                                        |
| ---------------------------- | ---------------------------------------------- |
| `AverageEffect`              | Average treatment effects and contrasts        |
| `ConditionalEffect`          | Conditional and heterogeneous effects          |
| `PulseEffect`                | Temporary temporal interventions               |
| `SustainedEffect`            | Sustained temporal interventions               |
| `InterventionalDistribution` | Distributions under `do(·)`                    |
| `PathSpecificEffect`         | Direct, mediated, and path-specific effects    |
| Counterfactual queries       | Nested and unit-level counterfactuals          |
| Attribution queries          | Anomaly, mechanism, path, and unit attribution |

## Capabilities

The full inventory — every graph class, algorithm, estimator, refuter, and
mechanism — lives in [docs/capabilities.md](docs/capabilities.md) (also on
[Read the Docs](https://antecedent.readthedocs.io/en/latest/capabilities/)).
The highlights:

* **Graphs.** DAG, ADMG, CPDAG, PAG, and their temporal variants, with
  d-/m-separation, latent projection, and Markov-equivalence operations.
  Static and temporal semantics are distinct. Interchange via NetworkX, DOT,
  JSON, GML, and versioned CBOR.
* **Discovery.** PC, FCI, RFCI, GES, DirectLiNGAM, NOTEARS; the temporal
  PCMCI family (PCMCI, PCMCI+, LPCMCI, J-PCMCI+, regime-specific RPCMCI);
  Bayesian structure posteriors that propagate into downstream effect
  analyses; discovery stability validators.
* **Identification.** Backdoor, front-door, IV, sharp RD, ID/IDC on DAGs and
  ADMGs, generalized adjustment for partial graphs, and temporal strategies.
  Every query is reported as identified, partially identified,
  graph-dependent, or not identified.
* **Estimation.** Regression, g-computation, IPW, matching, AIPW, 2SLS, RD,
  and temporal estimators on the frequentist side; Bayesian g-computation,
  HMC GLMs, prior transfer, and graph-by-effect posterior envelopes on the
  Bayesian side.
* **Interventions and counterfactuals.** An SCM layer with hard, soft,
  stochastic, sequenced, and policy interventions; abduction–action–prediction
  counterfactuals, nested counterfactuals, and temporal trajectories.
* **Attribution and diagnostics.** Anomaly, distribution-shift, change-point,
  and unit-level attribution; Shapley-based root-cause ranking.
* **Validation and sensitivity.** Placebo, common-cause, bootstrap, and
  data-subset refuters; overlap diagnostics; E-values; linear through
  nonparametric sensitivity; Bayesian predictive checks.
* **Experimental design.** Rank measure/intervene/observe actions by expected
  information gain, probability of identification, or decision utility.
* **Incremental state.** `CausalState` for online workflows: streaming
  sufficient statistics, particle filters, prepared analyses, and explicit
  invalidation that never silently reruns an analysis.
* **Data and artifacts.** NumPy, pandas, and Arrow in Python; tabular,
  time-series, panel, and multi-environment data; schema-versioned CBOR
  artifacts with memory-mapped access.

## Scientific scope

Antecedent follows several explicit constraints:

1. Priors do not upgrade nonparametric identification.
2. Discovery results are not assumed to be ground truth.
3. Static and temporal graph semantics are not interchangeable.
4. Unidentified graph-posterior mass is preserved.
5. Partial graphs are not silently completed.
6. PAG-native full ID and IDC are not claimed.
7. Unsupervised regime discovery is outside the RPCMCI workflow.

## Platform support

Python wheels cover CPython 3.11–3.14 on Linux, macOS, and Windows:
`pip install antecedent` (PyPI; also GitHub Release wheels). The scientific
engine and native API are written in Rust: `cargo add antecedent` (crates.io) —
see [docs/development.md](docs/development.md). No other language bindings are
currently provided.

## Documentation

Narrative docs and the Python API reference are on
[Read the Docs](https://antecedent.readthedocs.io/) ([Python API](https://antecedent.readthedocs.io/en/latest/python/antecedent.html));
the Rust API is on [docs.rs/antecedent](https://docs.rs/antecedent). Locally:
`mkdocs serve`, `cargo doc -p antecedent --open`.

Also: [Full capabilities](docs/capabilities.md) · [Comparison with DoWhy, EconML, Tigramite, causal-learn](docs/comparison.md) · [Architecture](docs/architecture.md) · [Development](docs/development.md) · [API naming](docs/api_naming.md) · [ADRs](adr/README.md) · [Examples](crates/antecedent/examples/) · [Python examples](python/examples/).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). DCO sign-off required.

## License

MIT OR Apache-2.0 — see `LICENSE-MIT` and `LICENSE-APACHE`.
