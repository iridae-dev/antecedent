# Cross-language naming dictionary

Rust and Python expose the **same capabilities** with idiomatic shapes on each side
(architecture invariant 9: capability parity ≠ API cloning).

| Capability | Rust | Python |
|---|---|---|
| Day-1 import | `use causal::prelude::*` | `import causal` |
| Run analysis | `CausalAnalysis::builder()…run(&ctx)` | `causal.analyze(data, graph=…, query=…)` |
| Average effect | `AverageEffectQuery` | `AverageEffect` |
| Temporal pulse / sustained | `TemporalEffectQuery` | `PulseEffect` / `SustainedEffect` |
| Mediation (static) | `MediationQuery` | `MediationEffect` |
| Mediation (temporal) | `MediationQuery` + temporal data | `TemporalMediationEffect` |
| Counterfactual ITE | `CausalQuery::Counterfactual` / `gcm::counterfactual_ite` | `Counterfactual` on `analyze` / `FittedGcm.counterfactual_ite` |
| Identify only | `CausalAnalysis::identify_only` | `causal.identify(graph=…, query=…)` |
| Identifier strategy | `IdentifierId::BackdoorAdjustment` | `Identifier.BACKDOOR_ADJUSTMENT` / `"backdoor.adjustment"` |
| Estimator strategy | `EstimatorId::LinearAdjustmentAte` | `Estimator.LINEAR_ADJUSTMENT_ATE` / `"linear.adjustment.ate"` |
| Inference | `InferenceMode::Bayesian(BayesianConfig::…)` | `Bayesian(...)` / `Frequentist()` |
| Tabular data | `TabularData::from_f64_columns` | `dict[str, array]` / pandas / Arrow |
| Named DAG | `Dag::from_named_edges(&schema, &[…])` | `Dag.from_edges(names, edges)` or edge list |
| d-separation | `Dag::is_d_separated` | `Dag.d_separated(x, y, z=…)` |
| Latent projection | `latent_project` | `Dag.latent_project(observed)` |
| Primary scalar effect | `result.effect()` | `result.estimate.ate` / `result.ate` |
| Errors | `CausalError` | `CausalError` (+ typed subclasses) |
| Stage modules | `causal::discovery`, `causal::gcm`, `causal::io` | `causal.discovery`, `causal.gcm`, `causal.graph` |

Prefer package / module paths for stage depth; keep day-1 at the crate / package root.
