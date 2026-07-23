# Cross-language naming dictionary

Rust and Python expose the **same capabilities** with idiomatic shapes on each side
(architecture invariant 9: capability parity ≠ API cloning).

| Capability | Rust | Python |
|---|---|---|
| Day-1 import | `use antecedent::prelude::*` (`cargo add antecedent`) | `import antecedent` |
| Run analysis | `CausalAnalysis::builder()…run(&ctx)` | `antecedent.analyze(data, graph=…, query=…)` |
| Average effect | `AverageEffectQuery` | `AverageEffect` |
| Temporal pulse / sustained | `TemporalEffectQuery` | `PulseEffect` / `SustainedEffect` |
| Mediation (static) | `MediationQuery` | `MediationEffect` |
| Mediation (temporal) | `MediationQuery` + temporal data | `TemporalMediationEffect` |
| Counterfactual ITE | `CausalQuery::Counterfactual` / `gcm::counterfactual_ite` | `Counterfactual` on `analyze` / `FittedGcm.counterfactual_ite` |
| Identify only | `CausalAnalysis::identify_only` | `antecedent.identify(graph=…, query=…)` |
| Identifier strategy | `IdentifierId::BackdoorAdjustment` | `Identifier.BACKDOOR_ADJUSTMENT` / `"backdoor.adjustment"` |
| Estimator strategy | `EstimatorId::LinearAdjustmentAte` | `Estimator.LINEAR_ADJUSTMENT_ATE` / `"linear.adjustment.ate"` |
| Inference | `InferenceMode::Bayesian(BayesianConfig::…)` | `Bayesian(...)` / `Frequentist()` |
| Tabular data | `TabularData::from_f64_columns` | `dict[str, array]` / pandas / Arrow |
| Named DAG | `Dag::from_named_edges(&schema, &[…])` | `Dag.from_edges(names, edges)` or edge list |
| d-separation | `Dag::is_d_separated` | `Dag.d_separated(x, y, z=…)` |
| Latent projection | `latent_project` | `Dag.latent_project(observed)` |
| Primary scalar effect | `result.effect()` | `result.effect` (`.ate` alias) |
| Errors | `CausalError` | `CausalError` (+ typed subclasses) |
| Latency tier | `LatencyMode::Interactive` | `Latency.INTERACTIVE` / `"interactive"` |
| Refute suite | `RefuteSuite::…` | `Refute.FULL` / `bool` / `"placebo"` |
| Plan inspection | `result.logical_plan()` / `PreparedAnalysis::plan()` | `result.plan` / `PreparedAnalysis.plan` |
| Stage modules | `antecedent::discovery`, `antecedent::gcm`, `antecedent::io` | `antecedent.discovery`, `antecedent.gcm`, `antecedent.graph` |

Prefer package / module paths for stage depth; keep day-1 at the crate / package root.
Rust stage APIs are **not** re-exported at the crate root — use `antecedent::io::…`, `antecedent::discovery::…`, `antecedent::gcm::…`, etc.
