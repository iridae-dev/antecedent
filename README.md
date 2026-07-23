# Antecedent

Antecedent is a causal inference library written in Rust with a Python API.

It provides a single workflow for causal discovery, identification, estimation, Bayesian inference, interventions, counterfactuals, attribution, validation, experimental design, and incremental causal state.

The library is built around three rules:

* identification is evaluated before estimation;
* priors and parametric assumptions do not upgrade nonparametric identification;
* uncertainty about causal structure is retained rather than silently resolved.

## Quick start

```bash
pip install antecedent
```

```python
from antecedent import AverageEffect, analyze

result = analyze(
    data=data,
    graph=graph,
    query=AverageEffect(
        treatment="treatment",
        outcome="outcome",
    ),
)

print(result.identification)
print(result.estimate)
print(result.validation)
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

## Graphs

Supported graph classes include:

* DAG;
* ADMG;
* CPDAG;
* PAG;
* temporal DAG;
* temporal CPDAG;
* temporal PAG.

Graph operations include:

* d-separation;
* m-separation;
* districts;
* latent projection;
* Markov-equivalence completions;
* definite-status separation;
* temporal unfolding;
* intervention overlays.

Static and temporal graphs have separate semantics. A static graph is not interpreted as temporal by default.

Graph interchange is available through NetworkX, DOT, JSON, GML, and versioned CBOR artifacts.

## Discovery

### Static

* PC
* FCI
* RFCI
* GES
* DirectLiNGAM
* NOTEARS

### Temporal and multi-context

* PCMCI
* PCMCI+
* LPCMCI
* J-PCMCI+
* regime-specific RPCMCI workflows

### Bayesian structure learning

* exact DAG posterior;
* order MCMC;
* structure MCMC;
* CI-screened graph posterior;
* DBN posterior.

Posterior graph samples can be propagated into downstream effect analyses.

### Conditional independence tests

Supported tests include:

* partial correlation;
* weighted and robust partial correlation;
* regression CI;
* k-nearest-neighbour CI;
* mixed k-nearest-neighbour CI;
* symbolic conditional mutual information;
* GPDC;
* G²;
* oracle tests;
* Bayesian CI tests.

Multiplicity corrections include BH, BY, Bonferroni, and Holm.

Discovery stability tools include block bootstrap, lag and threshold sensitivity, orientation stability, environment holdout, synthetic-null checks, and permutation or phase-randomized surrogates.

## Identification

Antecedent reports whether a query is:

* nonparametrically identified;
* partially identified;
* graph-dependent;
* not identified.

Implemented identification strategies include:

* backdoor adjustment;
* efficient backdoor adjustment;
* front-door identification;
* instrumental variables;
* sharp regression discontinuity;
* ID and IDC for DAGs and ADMGs;
* hedge certificates;
* nonparametric path-specific identification;
* generalized adjustment for partial graphs;
* unfolded temporal backdoor;
* temporal mediation.

`AutoIdentifier` reports applicable strategies. It does not silently choose an estimator.

For PAGs, Antecedent uses identification envelopes or explicit graph completions. Full PAG-native ID and IDC are outside the supported scope.

## Estimation

### Frequentist

* linear and generalized-linear outcome regression;
* g-computation;
* inverse probability weighting;
* propensity matching;
* covariate-distance matching;
* stratification;
* AIPW;
* front-door two-stage estimation;
* Wald estimation;
* 2SLS;
* sharp local-linear regression discontinuity;
* linear conditional effect models;
* temporal adjustment;
* temporal mediation;
* functional plug-in estimation.

### Bayesian

* Bayesian g-computation;
* temporal Bayesian g-computation;
* conjugate Gaussian models;
* Laplace GLM approximation;
* HMC GLMs;
* graph-by-effect posterior envelopes;
* same-design prior transfer;
* effect-level and mapped prior transfer;
* prior catalogs and compatibility filtering;
* power-prior mixtures;
* conflict-sensitive prior weighting;
* transport policies across compatible designs.

Unidentified graph-posterior mass is retained rather than silently renormalized away.

## Interventions and counterfactuals

Antecedent includes a structural causal model layer.

Supported mechanisms include:

* linear-Gaussian models;
* constant mechanisms;
* discrete mechanisms;
* hierarchical linear and generalized-linear models;
* Minnesota BVAR;
* linear Gaussian state-space models;
* Gaussian-process mechanisms.

Supported interventions include:

* hard interventions;
* soft interventions;
* stochastic interventions;
* sequenced interventions;
* temporal policies;
* dynamic policies;
* mechanism overrides.

Do-sampling methods include weighting, KDE, and MCMC.

Counterfactual support includes:

* abduction–action–prediction;
* nested counterfactuals;
* temporal trajectories;
* unit-level counterfactual analysis.

## Attribution and diagnostics

Antecedent can analyze:

* anomalous outcomes;
* distribution shifts;
* structural changes;
* mechanism changes;
* change points;
* unit-level change;
* path contributions;
* arrow strength;
* feature relevance;
* root-cause rankings.

Implemented techniques include:

* likelihood-ratio tests;
* mean-difference tests;
* classifier-based tests;
* MMD;
* Gaussian KL divergence;
* CUSUM-style scans;
* Shapley attribution;
* coalition caching.

## Validation and sensitivity

Estimate validation includes:

* placebo refuters;
* random common-cause refuters;
* unobserved common-cause refuters;
* bootstrap refuters;
* data-subset refuters;
* dummy-outcome refuters;
* overlap diagnostics;
* E-values;
* graph refutation.

Sensitivity methods include:

* linear sensitivity;
* partial-linear sensitivity;
* nonparametric sensitivity;
* Reisz sensitivity.

Bayesian validation includes:

* prior predictive checks;
* prior sensitivity;
* MCMC diagnostics;
* simulation-based calibration hooks.

Resampling support includes:

* IID bootstrap;
* Bayesian bootstrap;
* moving-block bootstrap;
* circular-block bootstrap;
* column permutation;
* phase-randomized surrogates.

## Experimental design

Antecedent can rank candidate actions such as:

* measuring a variable;
* intervening on a variable;
* observing an environment;
* changing a sampling plan.

Ranking criteria include:

* expected information gain;
* probability of identification;
* expected effect-interval width;
* decision utility.

The design layer supports batched Monte Carlo evaluation, common random numbers, and early stopping.

## Incremental state

`CausalState` supports stateful and online workflows.

Available components include:

* explicit invalidation;
* incremental OLS;
* streaming covariance;
* particle-filter state-space models;
* local score caches;
* rolling mechanism diagnostics;
* configurable cache budgets;
* prepared analyses;
* progressive and cancellable execution;
* adaptive resampling.

Invalidation does not automatically rerun an analysis.

## Data support

Antecedent supports:

* tabular data;
* time series;
* panel data;
* multi-environment data;
* event data converted into temporal frames.

Python interfaces support NumPy, pandas, and Arrow CDI. Rust uses `TableView`.

## Artifacts

Versioned artifacts include:

* graphs;
* graph posteriors;
* model bundles;
* analysis traces;
* causal state.

Artifacts use schema-versioned CBOR containers with optional Zstandard-compressed sections, selective reads, and memory-mapped access.

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

Python wheels are provided for:

* CPython 3.11–3.14;
* Linux;
* macOS;
* Windows.

The scientific engine and native API are written in Rust. No additional language bindings are currently provided.

Until public PyPI is enabled, install from [GitHub Packages](docs/development.md) or a Release wheel. Rust: `cargo add antecedent` (crates.io) or a git dependency — see [docs/development.md](docs/development.md).

## Documentation

The documentation covers:

* installation;
* typed queries;
* graph construction;
* discovery;
* identification;
* estimation;
* Bayesian inference;
* temporal and panel analysis;
* interventions;
* counterfactuals;
* attribution;
* validation;
* experimental design;
* artifacts;
* Rust API;
* Python API.

Narrative docs: [docs/](docs/index.md) (MkDocs / Read the Docs). Rust API: [docs.rs/antecedent](https://docs.rs/antecedent). Python API HTML ships in Release `docs.tar.gz` via pdoc. Locally: `mkdocs serve`, `cargo doc -p antecedent --open`.

Also: [Architecture](docs/architecture.md) · [Development](docs/development.md) · [API naming](docs/api_naming.md) · [ADRs](adr/README.md) · [Examples](crates/antecedent/examples/) · [Python examples](python/examples/).

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md). DCO sign-off required.

## License

MIT OR Apache-2.0 — see `LICENSE-MIT` and `LICENSE-APACHE`.
