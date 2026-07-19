# Causal

**Fast, explicit causal inference for Python and Rust.**

Causal brings causal discovery, identification, estimation, counterfactuals, attribution, and validation into one coherent API—across tabular, temporal, panel, and multi-environment data.

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

The central rule is simple:

> Identify the causal quantity first. Estimate it second.

Estimators never silently choose confounders or turn an unsupported causal question into a plausible-looking number.

## Why Causal?

Causal analysis usually requires stitching together several kinds of tooling:

* graph representation and traversal;
* causal discovery;
* adjustment and identification;
* statistical estimation;
* temporal analysis;
* structural causal models;
* sensitivity and validation;
* counterfactual and attribution methods.

Causal provides these as compatible components with shared data, graph, query, assumption, and result types.

That means you can move from:

```text
data
  → graph evidence
  → identified estimand
  → estimate or posterior
  → validation
```

without discarding the assumptions, uncertainty, or provenance produced by the previous stage.

## Highlights

* **Static and temporal workflows**
  Analyze tabular, time-series, panel, event, and multi-environment data through compatible APIs.

* **Identification as a first-class result**
  Distinguish nonparametric identification, identification under added restrictions, partial identification, graph-dependent identification, and non-identification.

* **Frequentist and Bayesian inference**
  Evaluate the same identified causal functional using either framework where appropriate.

* **Multiple graph semantics**
  Work with DAGs, ADMGs, CPDAGs, PAGs, temporal graphs, and weighted graph collections without collapsing them into one generic graph type.

* **Causal discovery**
  Discover static and lagged structure, retain edge evidence, apply domain constraints, and propagate unresolved graph uncertainty.

* **Interventions and counterfactuals**
  Evaluate hard, soft, stochastic, simultaneous, and temporal interventions, including unit-level counterfactuals.

* **Attribution and root-cause analysis**
  Explain anomalies, distribution shifts, mechanism changes, and path-specific contributions.

* **Validation built in**
  Run overlap checks, placebo tests, bootstrap refutations, hidden-confounding sensitivity, graph stability, prior sensitivity, and posterior predictive diagnostics.

* **Native performance**
  Heavy computation runs in Rust using batched APIs, reusable workspaces, explicit memory planning, and optimized numerical kernels.

* **Auditable outputs**
  Results retain assumptions, graph evidence, derivations, diagnostics, execution choices, and provenance.

## Installation

### Python

```bash
pip install causal
```

Supported Python versions:

```text
CPython 3.11–3.14
```

Prebuilt wheels are available for Linux, macOS, and Windows. The default build does not require a system BLAS installation.

### Rust

```bash
cargo add causal
```

Minimum supported toolchain:

```text
Rust 1.85
Edition 2024
```

## Python quick start

### Estimate an effect from a supplied graph

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
print(result.validation.count)
```

The result exposes each stage independently:

```python
result.identification
result.estimate
result.posterior
result.validation
result.diagnostics
result.provenance
result.performance
```

### Discover structure, then estimate a temporal effect

```python
discovered = causal.discover_pcmci(
    names, columns, max_lag=12, alpha=0.05, seed=1
)
# Orient / accept edges, then estimate with a lagged graph:
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

Wiring `discovery=` directly into `analyze()` for a single-call discover→estimate path is still partial.

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

The `causal` crate provides the high-level API (`use causal::prelude::*`). Lower-level crates expose graph, discovery, identification, estimation, model, validation, and numerical components.

## Capabilities

### Data

* tabular data;
* regular and irregular time series;
* panel data;
* multiple environments and regimes;
* event-indexed data;
* masks and missingness;
* observation weights;
* categorical and ordinal variables;
* vector-valued variables;
* temporal and grouped split strategies.

NumPy, pandas, PyArrow, and Polars inputs are supported through native or Arrow-based conversion paths.

### Graphs

* DAG, ADMG, CPDAG, and PAG types;
* temporal and context-aware graph variants;
* ancestry and reachability;
* topological ordering;
* d-separation and m-separation;
* active-path and separation witnesses;
* districts and ancestral subgraphs;
* latent projection;
* intervention overlays;
* equivalence-class operations;
* temporal unfolding;
* graph completion and graph sampling.

Safe constructors enforce graph-specific endpoint and acyclicity rules.

### Discovery

Static methods include:

* constraint-based discovery;
* latent-variable discovery;
* score-based DAG search;
* continuous SEM discovery (NOTEARS);
* Bayesian graph search.

Temporal methods include:

* PCMCI;
* PCMCI+;
* LPCMCI;
* J-PCMCI+;
* RPCMCI.

Discovery results can retain:

* raw and adjusted p-values;
* test statistics;
* confidence intervals;
* separating sets;
* selection frequencies;
* posterior edge probabilities;
* orientation probabilities;
* expert constraints;
* unresolved marks and conflicts.

A discovered graph is represented together with its evidence.

### Identification

Supported identification problems include:

* backdoor adjustment;
* minimal, maximal, and cost-ranked adjustment sets;
* generalized adjustment over graph classes;
* front-door identification;
* instrumental-variable validation;
* mediation;
* ID and IDC;
* partial identification;
* graph-ensemble identification;
* temporal identification.

```rust
let result = identifier.identify(
    &prepared_graph,
    &query,
    &mut workspace,
)?;
```

Identification results include the estimand, required assumptions, derivation trace, status, and diagnostics.

### Estimation

Frequentist estimators include:

* linear and generalized linear adjustment;
* distance and propensity matching;
* propensity stratification;
* ATE, ATT, and ATC weighting;
* doubly robust estimation;
* instrumental variables and 2SLS;
* regression discontinuity;
* front-door and mediation estimation;
* conditional and temporal effects;
* analytic and resampling-based uncertainty.

Bayesian inference includes:

* Gaussian, logistic, Poisson, and temporal mechanism models;
* posterior evaluation of identified functionals;
* Bayesian g-computation;
* graph-model averaging;
* posterior interventional sampling;
* prior-sensitivity analysis;
* posterior predictive diagnostics.

Identification status remains attached to the estimate or posterior.

### Structural causal models

* probabilistic causal models;
* structural causal models;
* invertible mechanisms;
* static and temporal models;
* automatic mechanism assignment;
* observational sampling;
* hard, soft, stochastic, and simultaneous interventions;
* posterior predictive intervention sampling;
* model falsification and predictive checks.

Models compile into reusable execution plans for repeated simulation and intervention.

### Counterfactuals

Counterfactual evaluation follows abduction–action–prediction.

Supported operations include:

* point and distributional counterfactuals;
* individual treatment effects;
* shared-noise counterfactual worlds;
* counterfactual trajectories;
* missing factual variables;
* posterior uncertainty over exogenous state;
* nested counterfactuals where supported by the model assumptions.

### Attribution

* anomaly attribution;
* distribution-change attribution;
* mechanism-change attribution;
* unit-change attribution;
* path-specific contributions;
* feature relevance;
* causal influence and arrow strength;
* Shapley decompositions;
* root-cause ranking.

```rust
let result = ChangeAttribution::new()
    .outcome("defect_probability")
    .baseline(january)
    .comparison(february)
    .components(AttributionComponents::All)
    .allocation(AllocationMethod::Shapley {
        approximation: ShapleyConfig::monte_carlo(2_000),
    })
    .run(&model, &posterior, &ctx)?;
```

Approximate methods report their compute budget and Monte Carlo error.

### Validation and sensitivity

```rust
let report = ValidationSuite::new()
    .with(PlaceboTreatment::default())
    .with(GraphStability::block_bootstrap(200))
    .with(PriorSensitivity::standard_grid())
    .run(&result, &ctx)?;
```

Available validators include:

* placebo treatment;
* random common cause;
* data-subset refutation;
* bootstrap refutation;
* dummy outcome;
* unobserved-confounding sensitivity;
* overlap and effective-sample-size diagnostics;
* graph refutation;
* discovery stability;
* lag-window sensitivity;
* orientation stability;
* environment and regime holdouts;
* prior predictive checks;
* posterior predictive checks;
* prior sensitivity;
* simulation-based calibration.

Inapplicable checks return an explicit `NotApplicable` result.

### Experiment and measurement design

Rank candidate measurements and interventions by:

* graph information gain;
* identification probability;
* expected posterior-width reduction;
* expected utility;
* posterior regret;
* model discrimination;
* cost and constraints.

Results retain approximation error and ranking uncertainty.

## Explicit causal semantics

Causal keeps several distinctions visible throughout the API.

### Identification is not estimation

An estimator receives an identified estimand. It does not decide which adjustment set makes the query valid.

### Priors do not remove non-identifiability

Bayesian assumptions can make a model-specific quantity estimable, but they are recorded as additional restrictions rather than reported as nonparametric identification.

### Graph uncertainty is not parameter uncertainty

Sampling, parameter, graph, orientation, identification, mechanism, regime, and measurement uncertainty are represented separately.

### Discovered structure is evidence

Discovery output can be reviewed, constrained, completed, sampled, or propagated downstream. It is not automatically promoted to ground truth.

### Static and temporal graphs are not interchangeable

Temporal algorithms operate on explicit lag and time-index semantics. Summary graphs are not accepted as substitutes for expanded causal structures without a declared interpretation.

## Performance

Performance and correctness are co-equal implementation requirements.

Core execution paths use:

* compact integer graph indexes;
* borrowed typed column views;
* columnar data layouts;
* prepared sample and design plans;
* reusable numerical and graph workspaces;
* batched conditional-independence tests;
* batched resampling and simulation;
* bounded parallel execution;
* explicit memory budgets;
* scalar reference kernels;
* runtime-selected optimized kernels;
* coarse-grained Python-to-Rust calls;
* GIL release during native computation.

Repeated hot-path operations avoid per-row dynamic dispatch, Python callbacks, and unnecessary scratch allocation.

Optimized implementations must match scalar reference behavior under the same semantic and numerical tolerance contracts.

## Observable execution

Results can report execution choices that materially affect performance:

* borrowed and copied bytes;
* materializations and transpositions;
* selected numerical backend;
* scalar or optimized kernel path;
* fallback reasons;
* batch sizes;
* thread use;
* cache behavior;
* estimated peak memory;
* Python boundary crossings.

Memory-intensive algorithms estimate their requirements before allocation and use bounded, streaming, or chunked execution where supported.

## Reproducibility

Artifacts can retain:

* data schema and category domains;
* masks and preprocessing;
* row-selection policy;
* graph version and evidence;
* assumptions and their sources;
* query and target population;
* identification derivation;
* estimator or inference configuration;
* random seeds;
* backend and kernel information;
* warnings and diagnostics;
* provenance;
* performance records.

Models, queries, graphs, and results use explicitly versioned serialization formats rather than serializing internal Rust structs as the durable contract.

## API layers

### High-level analysis

```python
result = causal.analyze(...)
```

Use the facade to compose graph handling, identification, estimation, and optional validation.

### Component APIs

Use individual crates or Python modules when you need a specific stage:

```rust
use causal_graph::Dag;
use causal_discovery::Pcmci;
use causal_identify::BackdoorIdentifier;
use causal_estimate::AipwAte;
use causal_validate::PlaceboTreatment;
```

### Prepared and batched APIs

Performance-sensitive workloads can prepare reusable state explicitly:

```rust
let prepared = estimator.prepare(
    &data,
    &estimand,
    &ctx,
)?;

let fit = estimator.fit(
    &prepared,
    &mut workspace,
    &ctx,
)?;
```

Preparation, allocation, and execution costs are kept distinct where repeated evaluation is expected.

## Repository structure

```text
crates/
  causal-core/
  causal-data/
  causal-graph/
  causal-expr/
  causal-kernels/
  causal-stats/
  causal-prob/
  causal-discovery/
  causal-identify/
  causal-estimate/
  causal-model/
  causal-counterfactual/
  causal-attribution/
  causal-validate/
  causal-design/
  causal-state/
  causal-io/
  causal/

python/
docs/
examples/
benches/
conformance/
provenance/
fuzz/
adr/
```

## Project scope

Causal provides causal computation and related numerical, graph, serialization, and validation primitives.

Data ingestion services, orchestration, dashboards, authentication, registries, deployment, and action execution are outside the project scope.

## Documentation

* [Getting started](docs/getting-started.md)
* [Python guide](docs/python/)
* [Rust API](https://docs.rs/causal)
* [Data model](docs/data/)
* [Graphs](docs/graphs/)
* [Discovery](docs/discovery/)
* [Identification](docs/identification/)
* [Estimation](docs/estimation/)
* [Structural causal models](docs/models/)
* [Counterfactuals](docs/counterfactuals/)
* [Attribution](docs/attribution/)
* [Validation](docs/validation/)
* [Temporal analysis](docs/temporal/)
* [Performance](docs/performance.md)
* [Artifact format](docs/artifacts.md)
* [Examples](examples/)
* [Technical design](DESIGN.md)

## Contributing

Contributions are welcome across algorithms, numerical methods, APIs, interoperability, benchmarks, testing, and documentation.

Algorithm contributions should include:

* the mathematical or published-method basis;
* correctness and edge-case tests;
* statistical calibration where applicable;
* a representative benchmark;
* memory or allocation coverage for hot paths;
* provenance metadata.

See [CONTRIBUTING.md](CONTRIBUTING.md).

Contributions use Developer Certificate of Origin sign-off.

## License

Licensed under either:

* Apache License, Version 2.0
* MIT License

at your option.

```text
SPDX-License-Identifier: MIT OR Apache-2.0
```
