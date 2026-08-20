# Cross-language naming dictionary

Rust and Python expose the **same capabilities** with idiomatic shapes on each side
(architecture invariant 9: capability parity ≠ API cloning).

## The shape

Every analysis is three verbs:

- `analyze(data, graph=..., query=...)` — identify, then estimate, in one call.
- `identify(graph=..., query=...)` — identify only; returns a staged
  `Identification` that can continue into `.estimate(data)` / `.validate(data)`
  while retaining the resolved strategy and query. The one-shot execution
  pipeline deterministically rechecks identification before estimation.
- `estimate(identification, data, ...)` — the module-level mirror of
  `Identification.estimate`, for callers that already hold a staged
  `Identification`.

The root namespace (`import antecedent`) is **frozen at 49 names through 0.6**: the three verbs
above; the accepted-structure and result types (`AcceptedGraph`, `Identification`,
`AnalysisResult`); the nine typed queries (`AverageEffect`, `PulseEffect`,
`SustainedEffect`, `InterventionalDistribution`, `PathSpecificEffect`,
`ConditionalEffect`, `MediationEffect`, `Counterfactual`,
`TemporalMediationEffect`) plus the eight response-family queries
(`ResponseCurve`, `AverageDerivative`, `PointDerivative`, `Elasticity`,
`SemiElasticity`, `DirectionalDerivative`, `ResponseJacobian`,
`InterventionResponse`); the five graph classes (`Dag`, `Cpdag`, `Pag`, `Admg`,
`TemporalDag`); the inference / identifier / estimator / latency / refute selectors
(`Frequentist`, `Bayesian`, `Identifier`, `Estimator`, `Latency`, `Refute`); the two
error names most callers catch (`CausalError`, `ReviewRequired`); the twelve stage
modules themselves; and `__version__`.

**The rule for where a name lives**: if it's part of the day-1 workflow — run an
analysis, describe a query, hold a graph, catch an error — it's at the root. Anything
more specialized lives in the stage module that owns it, and you reach it by walking
the module path rather than importing it flat:

``antecedent.attribution``, ``antecedent.data``, ``antecedent.design``,
``antecedent.discovery``, ``antecedent.errors``, ``antecedent.estimation``,
``antecedent.extensibility``, ``antecedent.gcm``, ``antecedent.graph``,
``antecedent.priors``, ``antecedent.state``, ``antecedent.validation``.

A handful of further modules are reachable as ``antecedent.<name>`` (nothing stops
`import antecedent; antecedent.population.AllRows` from working) but are deliberately
left out of the frozen `__all__` list because their public content is already
re-exported above, or because they're a narrower surface than the twelve stage
modules: ``antecedent.counterfactual``, ``antecedent.inference``, ``antecedent.model``,
``antecedent.population``, ``antecedent.query``.

Discovery algorithms follow the same rule: what used to be sixteen free
`discover_*()` functions are now config dataclasses on ``antecedent.discovery``
(`PC`, `GES`, `LiNGAM`, `NOTEARS`, `FCI`, `RFCI`, `PCMCI`, `PCMCIPlus`, `LPCMCI`,
`JPCMCIPlus`, `RPCMCI`, and the graph-posterior family). Each one owns its own
`run()` (and, where a holdable graph artifact makes sense, `.accept()`):

```python
result = antecedent.discovery.PC(alpha=0.05).run(data, seed=1)
accepted = antecedent.discovery.PC(alpha=0.05).accept(data, seed=1)  # -> AcceptedGraph
```

Graph interchange is on the classes: ``Dag.from_dot`` / ``Dag.to_dot`` and the
JSON / GML / NetworkX peers, likewise on ``Cpdag`` / ``Pag`` / ``Admg`` — not free
`dag_from_*` / `dag_to_*` functions. These constructors round-trip variable names
(earlier versions discarded them).

Public analysis results are nested ``AnalysisResult`` views. The native DTOs
live on ``antecedent._native`` only, which is an advanced FFI surface.

## Rust ↔ Python capability map

| Capability | Rust | Python |
|---|---|---|
| Day-1 import | `use antecedent::prelude::*` (`cargo add antecedent`) | `import antecedent` |
| Run analysis | `Study::tabular(data)…build()?.run(&ctx)` (or `::series` / `::series_multi` / `::panel` / `::events` for other modalities) | `antecedent.analyze(data, graph=…, query=…)` |
| Identify only (staged) | `Study::tabular(data)…build()?.identify_only()` | `antecedent.identify(graph=…, query=…)` → `Identification.estimate()` / `.validate()` |
| Average effect | `AverageEffectQuery` | `AverageEffect` |
| Continuous response | `ResponseQuery` / `ResponseFunctional` | `ResponseCurve` / `AverageDerivative` / `PointDerivative` / `Elasticity` / `SemiElasticity` |
| Vector response derivative | `ResponseFunctional::DirectionalDerivative` / `::Jacobian` | `DirectionalDerivative` / `ResponseJacobian` |
| Intervention response | `ResponseFunctional::InterventionResponse` | `InterventionResponse(..., intervention=intervention.Set/Shift/Bernoulli/Gaussian/Categorical(...))` |
| Observation mechanism | `ObservationSpec` + explicit `ObservationAssumption` | `antecedent.observation` specs attached to a response query |
| Structural transport | `TransportQuery` + `SelectionDiagram` | `antecedent.transport.TransportQuery` / `SelectionDiagram` |
| Randomized interference | `InterferenceQuery` + `AssignmentDesign` + `ExposureMapping` | `antecedent.interference` stage types |
| Temporal pulse / sustained | `TemporalEffectQuery` | `PulseEffect` / `SustainedEffect` |
| Temporal dose × horizon response | `ResponseQuery` + `TemporalResponseSpec` on `ResponseFunctional::MeanCurve` / `::InterventionResponse` | `ResponseCurve(..., horizons=…, policy="pulse"|"sustained"|"dynamic", treatment_lag=…, max_history_lag=…)` / matching `InterventionResponse(..., horizons=…, …)` — keyword-only after treatment/outcome names; absent `horizons` = static Dag cell |
| Mediation (static) | `MediationQuery` | `MediationEffect` |
| Mediation (temporal) | `MediationQuery` + temporal data | `TemporalMediationEffect` |
| Counterfactual ITE | `CausalQuery::Counterfactual` / `gcm::counterfactual_ite` | `Counterfactual` on `analyze` / `FittedGcm.counterfactual_ite` |
| Identifier strategy | `IdentifierId::BackdoorAdjustment` | `Identifier.BACKDOOR_ADJUSTMENT` / `"backdoor.adjustment"` |
| Estimator strategy | `EstimatorId::LinearAdjustmentAte` | `Estimator.LINEAR_ADJUSTMENT_ATE` / `"linear.adjustment.ate"` |
| Per-estimator tuning | `EstimatorSpec::LinearAdjustmentAte { .. }` (builder setters) | `analyze(..., estimator_config={...})` — one table-driven dict kwarg; see `python/src/estimator_config.rs` for the estimator-id → valid-keys table |
| Discovery algorithm | `discover_pc` / `discover_ges` / … (still free functions) | `antecedent.discovery.PC(...).run(...)` / `.accept(...)` (config dataclass, not a free `discover_*` function) |
| Accepted-graph session | `DiscoveryArtifact` / re-run identify+estimate | `AcceptedGraph.accepted(...)` / `.asserted(...)`; `.review({edge: mark})`; `.pending`; `len()` / `iter()` / `in` |
| Target population | Rust `TargetPopulation` enum | `antecedent.population.AllRows` / `Treated` / `Untreated` / `Named` / `Rows` / `CustomDistribution` dataclasses (module not at root; `target_all()`-style constructors still work and return these types) |
| Inference | `InferenceMode::Bayesian(BayesianConfig::…)` | `Bayesian(...)` / `Frequentist()` |
| Refutation suite | `RefuteSuite::…` | `Refute.FULL` / `"placebo"` / `"cheap"` / `"full"` — `refute=True` is rejected (`TypeError`); leave `refute` unset for the default suite. Temporal `ResponseCurve` / `InterventionResponse` record `refute.temporal_response.skipped` (scalar refuters do not apply to function-valued surfaces). |
| Tabular data | `TabularData::from_f64_columns` | `dict[str, array]` / pandas / Arrow |
| Named DAG | `Dag::from_named_edges(&schema, &[…])` | `Dag.from_edges(names, edges)` or edge list |
| d-separation | `Dag::is_d_separated` | `Dag.d_separated(x, y, z=…)` |
| Latent projection | `latent_project` | `Dag.latent_project(observed)` |
| External prior bank | `antecedent-prob::conjugate_moment_match` / `compose_external_priors` | `antecedent.priors.beta_from_moments` / `compose_external_priors` / `PriorCatalog` (module renamed from `prior_bank` to `priors`) |
| Primary scalar effect | `result.effect()` | `result.effect` (`.ate` alias) |
| Rich result display | `Debug` / `Display` impls | `AnalysisResult.__repr__` / `_repr_html_` (amber callout when `unidentified_mass > 0`); `ValidationView` supports `len()` / iteration / indexing / `.failed` / `.to_pandas()`; `PosteriorView` supports `__array__` / `.interval()` |
| Errors | `CausalError` | `CausalError` (+ typed subclasses); `ReviewRequired` carries structured `pending_edges` |
| Latency tier | `LatencyMode::Interactive` | `Latency.INTERACTIVE` / `"interactive"` |
| Plan inspection | `result.logical_plan()` / `PreparedAnalysis::plan()` | `result.plan` / `PreparedAnalysis.plan` |
| Stage modules | `antecedent::discovery`, `antecedent::gcm`, `antecedent::io` | `antecedent.discovery`, `antecedent.gcm`, `antecedent.graph` |

Prefer package / module paths for stage depth; keep day-1 at the crate / package root.
Rust stage APIs are **not** re-exported at the crate root — use `antecedent::io::…`, `antecedent::discovery::…`, `antecedent::gcm::…`, etc.

## A sharp edge worth knowing

**Pulse / Sustained vs temporal `ResponseCurve`.** On licensed `TemporalDag`
cells, `PulseEffect` and single-step `SustainedEffect` agree numerically with
the dose × horizon surface on the shared known-truth fixture, but standard
errors can differ: the surface uses analytic delta-method bands while the
contrast path uses the `Study` bootstrap configuration.

**Temporal response refuters.** Scalar ATE refuters are not applicable to a
function-valued temporal surface; licensed runs emit diagnostic
`refute.temporal_response.skipped` rather than running placebo / random-common-cause
checks on a single scalar summary.

`__init__.py` does `from .identify import Identification, estimate, identify`, which
rebinds the `identify` attribute on the `antecedent` package to the **function**.
`import a.b as m` is spec'd as "import `a.b`, then `m = a.b`" — an attribute lookup
on the already-imported parent, not a `sys.modules` lookup — so
`import antecedent.identify as m` binds `m` to that function, not to the
`identify.py` module. `from antecedent.identify import Identification` still works
as expected, because `from X import Y` extracts `Y` directly off the
freshly-imported module object rather than walking `X`'s (possibly rebound)
attributes. In practice this rarely matters: everything `identify.py` exports
(`Identification`, `IdentifyResult`, `estimate`, `identify`, `validate`) is
reachable without naming the module at all — `Identification` and
`identify`/`estimate` are already at the root, and `validate`/`IdentifyResult` are
one `from antecedent.identify import ...` away.
