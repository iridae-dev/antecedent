# Compiler layers

The six layers are ownership boundaries inside the existing compiler. They are
not new crates and not a second public workflow. `PreparedStudy` and Python
`PreparedAnalysis` stay the practitioner handle. `antecedent-expr` stays the
identified-functional IR.

The layers exist as code. `Study::compile` lowers every route cell to one
`LoweredProgram` (`crates/antecedent/src/planner/lowered.rs`) and attaches it to
the `PhysicalExecutionPlan` it returns; dispatch refuses a plan without one. The
program holds the scientific problem, the identification product, the statistical
program, the evidence and decision program, the physical layer, and the claim the
execution must publish. See [The lowered program](#the-lowered-program).

`antecedent-contract::SemanticLayer` (`Target`, `Identification`, `Program`,
`Inference`, `Support`, `Data`, `Execution`, `Results`) is the identity and
transform axis from ADR 0022. It is not this map. In particular its `Program`
is the licensed contract digest, not layer 3.

| Layer | Owns |
| --- | --- |
| 1. Scientific problem | Population, units, variables, interventions, outcome functional, horizon, structural assumptions, evidence declarations, decision objective. |
| 2. Identification product | Identified functional or set, derivation, required evidence, unresolved obligations, search status. |
| 3. Statistical program | Models, nuisance fits, scores or likelihood, priors, folds, resampling, posterior computation, diagnostics. |
| 4. Evidence/decision program | Source contributions, transformations, dependence, joint uncertainty, actions, utility, constraints, information value. |
| 5. Physical plan | Scans, layouts, materialization, batching, kernels, threads, memory, cancellation, retained state. |
| 6. Claim/receipt | Answer, uncertainty reading, support, assumptions, identities, provenance, measured execution facts. |

A later layer may do more work. It must not strengthen an earlier layer.
Cheap preflight (`Study::inspect`) stays off the identification, fitting,
enumeration, and simulation path.

## Planner

Logical compile writes a route record. It does not build an
`IdentificationResult` or a fit.

| Type or function | Path | Layer | Crate | Notes |
| --- | --- | --- | --- | --- |
| `LogicalAnalysisPlan` | `crates/antecedent/src/planner/plan.rs` | 1 | `antecedent` | Exists today. Holds `LogicalAnalysisPlanRecord`, `CausalQuery`, optional `DiscoveryEstimationSplit`, `row_count_hint`. `validate` checks modality against the query and a few discovery-algorithm names. |
| `LogicalAnalysisPlan::compile_physical` and `compile_physical_with_all_graphs` | `crates/antecedent/src/planner/plan.rs` | 5 | `antecedent` | Exists today. Lowers the route record under `ExecutionContext`. |
| `PhysicalExecutionPlan` | `crates/antecedent/src/planner/plan.rs` | 5 | `antecedent` | Exists today. Record plus the logical plan and optional resolved `TemporalDag`, `Dag`, `Pag`. |
| `reject_dag_only_on_pag` | `crates/antecedent/src/planner/plan.rs` | 1 | `antecedent` | Exists today. Refuses a DAG-only `IdentifierId` on `GraphClass::Pag` or `TemporalPag`. |
| `LogicalAnalysisPlanRecord` | `crates/antecedent-contract/src/plan.rs` | 1 | `antecedent-contract` | Exists today. `plan_id`, `DataClassification`, discovery algorithm, review flag, identifier and estimator name strings, validation suite, `query_variables`. No estimand, score, prior, or evidence ledger. |
| `PhysicalExecutionPlanRecord` | `crates/antecedent-contract/src/plan.rs` | 5 | `antecedent-contract` | Exists today. Estimator-free record: materializations, kernels, batch size, byte estimates, one `ParallelTaskSpec` list, worker count, `deterministic_reductions`, `expected_python_crossings`. |
| `BufferMaterialization`, `KernelSelection`, `ParallelTaskSpec`, `DataClassification` | `crates/antecedent-contract/src/plan.rs` | 5 | `antecedent-contract` | Exists today. `Chunked` is a variant. The planner does not select it. |
| `ExecutionPerformanceRecord` | `crates/antecedent-contract/src/plan.rs` | 6 | `antecedent-contract` | Exists today. Measured wall time, RSS, copies, stage timings, bootstrap and draw counts. A receipt field, not a plan. |
| `PlanBinding`, `PlanName`, `RouteDefaults` | `crates/antecedent/src/planner/binding.rs` | 1 | `antecedent` | Exists today. Parses the record's identifier and estimator strings into `IdentifierId` / `EstimatorId`. Unnamed slots take the route default. Unrecognised names refuse. |
| `StaticCompileInput` and the `Static*CompileInput` aliases | `crates/antecedent/src/planner/logical.rs` | 1 | `antecedent` | Exists today. Tabular data, graph, query, optional `DiscoveryEstimationSplit`, identifier and estimator strings. |
| `TemporalClassEffectCompileInput` | `crates/antecedent/src/planner/logical.rs` | 1 | `antecedent` | Exists today. Series, temporal class nodes, `TemporalEffectQuery`, split, classification, plan id, identifier, estimator. |
| `compile_logical_static_ate` | `crates/antecedent/src/planner/logical.rs` | 1 | `antecedent` | Exists today. Plan id `static_ate`. Refuses a target population that linear adjustment cannot carry. |
| `compile_logical_static_response` | `crates/antecedent/src/planner/logical.rs` | 1 | `antecedent` | Exists today. Plan id from the response route. |
| `compile_logical_static_pag_ate`, `compile_logical_static_cpdag_ate` | `crates/antecedent/src/planner/logical.rs` | 1 | `antecedent` | Exists today. Class average-effect records. |
| `compile_logical_static_pag_response`, `compile_logical_static_cpdag_response` | `crates/antecedent/src/planner/logical.rs` | 1 | `antecedent` | Exists today. Class response records. Requires a class-aware response query. |
| `compile_logical_distribution` | `crates/antecedent/src/planner/logical.rs` | 1 | `antecedent` | Exists today. Interventional-distribution route record. |
| `compile_logical_path_specific` | `crates/antecedent/src/planner/logical.rs` | 1 | `antecedent` | Exists today. Path-specific route record. |
| `compile_logical_codetermined_joint` | `crates/antecedent/src/planner/logical.rs` | 1 | `antecedent` | Exists today. Co-determined ADMG joint response record. |
| `compile_logical_temporal_effect`, `compile_logical_temporal_effect_classified` | `crates/antecedent/src/planner/logical.rs` | 1 | `antecedent` | Exists today. Plan id `temporal_effect`. Records `temporal.backdoor.unfolded` and `temporal.linear.adjustment` on the DAG path. Refuses a planned target population. |
| `compile_logical_temporal_response` | `crates/antecedent/src/planner/logical.rs` | 1 | `antecedent` | Exists today. Temporal response route record. |
| `compile_logical_temporal_class_effect` | `crates/antecedent/src/planner/logical.rs` | 1 | `antecedent` | Exists today. Pulse or single-step Sustained on a temporal CPDAG or PAG. |
| `compile_logical_admg_response` | `crates/antecedent/src/analysis/execute/compile.rs` | 1 | `antecedent` | Exists today. ADMG response logical record. |
| `IdentifierId`, `EstimatorId` | `crates/antecedent/src/strategy_table/ids.rs` | 1 | `antecedent` | Exists today. Closed wire names the logical record stores. Selection is not identification and not a fit. |

`compile_physical_with_all_graphs` sizes the record from the logical plan's row count and its query variables, materializes `design.matrix` as `CopiedContiguous`, picks `DenseBackend` under the estimator's `kernel_label` (fallback `ols.faer`), schedules one task dimension from `EstimatorId::parallel_task_dimension` (fallback `analysis`), and sets `expected_python_crossings` to 1. A soft memory limit with no chunked alternative returns `CausalError::Resource`. That sizing is a starting record, not the design: when the attached lowered program quotes a design (a linear adjustment `[1 | T | Z…]` with its contrast codings, or an ANCOVA randomized effect), `PhysicalExecutionPlan::requote` re-plans rows, columns, byte estimates and whether the design is borrowed from the source columns, from the quoted width. A route that fits no single adjustment design says so on its layer five (`WidthSource::NotDesignBased`, with the reason) instead of passing the starting record off as a quote. A plan from `compile_physical` alone carries no lowered program and does not dispatch.

## Compile

`Study::route_input` (`crates/antecedent/src/analysis/execute/route_input.rs`) admits one `RouteInput` cell. `Study::compile_logical` and `Study::compile` (`crates/antecedent/src/analysis/execute/compile.rs`) dispatch that cell. A graph posterior has no single structure: `compile_logical` returns `CausalError::Unsupported` and `compile` calls `compile_graph_posterior`. Its single logical plan is the mixture section of its [plan view](#the-plan-view).

| Type or function | Path | Layer | Crate | Notes |
| --- | --- | --- | --- | --- |
| `RouteInput` | `crates/antecedent/src/analysis/execute/route_input.rs` | 1 | `antecedent` | Exists today. Borrowed data, query, and graph for one admitted cell, including `StaticAteMixed`, which logical compile still refuses. |
| `StaticMediationSource` | `crates/antecedent/src/analysis/execute/route_input.rs` | 1 | `antecedent` | Exists today. `MediationQuery` or `NestedCounterfactualQuery`. |
| `PlanId` | `crates/antecedent/src/analysis/execute/plan_id.rs` | 1 | `antecedent` | Exists today. Wire `plan_id` for the routes that do not use the planner's fixed ids (`static_conditional`, `transport_trial`, `gcm_query`, longitudinal regime ids, and the rest of that enum). |
| `Study::compile_logical` | `crates/antecedent/src/analysis/execute/compile.rs` | 1 | `antecedent` | Exists today. Inspectable route record only. |
| `Study::compile` | `crates/antecedent/src/analysis/execute/compile.rs` | 1 to 6 | `antecedent` | Exists today. Logical record, then `compile_physical*`, then the route lowering, which attaches the `LoweredProgram`. Attaches a resolved DAG, temporal DAG, or PAG when the cell has one. `StaticAteMixed` is physical-only. A context already inside a lowered dispatch is refused (`public_builder_reentry`). |
| `Study::compile_for_decision`, `Study::compile_for_evsi` | `crates/antecedent/src/analysis/execute/compile_decision.rs` | 1 to 6 | `antecedent` | Exists today. Compile that starts from a `DecisionProblem` or an `EvsiRequest`; layer four carries the decision or the EVSI program. |
| `RouteCell`, `GraphPosteriorCell`, the registry table | `crates/antecedent/src/analysis/lowering/registry.rs` | 1 to 6 | `antecedent` | Exists today. One row per `RouteInput` variant mapping it to its family lowering (`static_effect`, `static_response`, `structural`, `temporal`, `design_based`), plus the graph-posterior cells. A `RouteInput` variant without a row does not compile. The cell name is the `lowering` column of `parity/licensed_routes.toml`. |
| `Study::compile_logical_from` and the `compile_logical_*` route builders in this file | `crates/antecedent/src/analysis/execute/compile.rs` | 1 | `antecedent` | Exists today. One builder per `RouteInput` arm: static response DAG/PAG/CPDAG, co-determined joint, mixed ADMG response, static ATE DAG/PAG/CPDAG, distribution DAG and mixed, path-specific, temporal effect and response, multi-env temporal effect, panel effect and response, conditional DAG/CPDAG/PAG, temporal and static mediation, transport, interference, randomized, policy value, panel DiD, synthetic control, continuous dose, local-polynomial ratio, survival, longitudinal regime, GCM, cross-world. Each fills a `LogicalAnalysisPlanRecord`. |
| `Study::run` | `crates/antecedent/src/analysis/execute/dispatch.rs` | 3 | `antecedent` | Exists today. One-shot sugar over compile, prepare, and estimate. Returns `StudyResult`. |
| `Study::execute` and the dispatch entry points | `crates/antecedent/src/analysis/execute/dispatch.rs` | 3 to 6 | `antecedent` | Exists today. Require the plan's lowered program for the cell the study classifies as (`route_not_lowered` otherwise), run inside its lowered-dispatch scope, and stamp its identity on the result. |

## Prepared execution

`PreparedStudy` is the workflow handle. Its fields already sit on different layers. Estimate and refresh do not re-identify or rebuild the physical plan. Sharp RD is the exception: prepare does not cache that identification product; each estimate click re-identifies.

| Type or function | Path | Layer | Crate | Notes |
| --- | --- | --- | --- | --- |
| `PreparedStudy`, `SampledPreparedState` | `crates/antecedent/src/analysis/prepared/mod.rs` | workflow | `antecedent` | Exists today. Freezes `Study`, `PhysicalExecutionPlan`, `CausalSchema`, modality, `SamplingRegularity`, optional `ScoreTable`, and `PreparedExecution`. `program_cache` holds contract payloads. |
| `Study::prepare` | `crates/antecedent/src/analysis/prepared/prepare.rs` | 2 then 5 | `antecedent` | Exists today. Refuses discovery and review-required graphs. Caches identification, seals an operation, compiles the physical plan once. |
| `PreparedStudy::estimate`, `estimate_series`, `estimate_panel`, `estimate_multi_env`, `estimate_retained` | `crates/antecedent/src/analysis/prepared/estimate.rs`, `lifecycle.rs` | 3 | `antecedent` | Exists today. Same schema and frozen plan. Numeric knobs, seeds, and `ExecutionContext` may change. |
| `PreparedStudy::refresh`, `refresh_series`, `refresh_panel`, `refresh_multi_env` | `crates/antecedent/src/analysis/prepared/lifecycle.rs` | 3 | `antecedent` | Exists today. Schema-checked data swap. A frozen-field change is an error, not a recompile. |
| `PreparedStudy::retarget` | `crates/antecedent/src/analysis/prepared/retarget.rs` | 1 | `antecedent` | Exists today. Row-weight population on a prepared score table. Identification is unchanged. |
| `PreparedStudy::refute` | `crates/antecedent/src/analysis/prepared/lifecycle.rs` | 4 | `antecedent` | Exists today. Same frozen estimand. `CausalQuery::AverageEffect` only. |
| `PreparedStudy::contract` | `crates/antecedent/src/analysis/contract.rs` | 6 | `antecedent` | Exists today. Builds `CausalContract` from cached identification products. |
| `Study::inspect` | `crates/antecedent/src/analysis/contract.rs` | 1 | `antecedent` | Exists today. Cheap structural inspection. Identification slot and program identity are absent. It does not identify (`compiler.inspect_does_not_identify`); the [plan view](#the-plan-view) is where identification is shown. |
| `CachedStaticIdentification`, `CachedTemporalIdentification`, `CachedTemporalHorizonIdentification` | `crates/antecedent/src/analysis/prepared/cached.rs` | 2 | `antecedent` | Exists today. Public cached products. |
| `CachedPagIdentification`, `CachedCpdagIdentification`, `CachedTemporalClassIdentification` | `crates/antecedent/src/analysis/prepared/cached.rs` | 2 | `antecedent` | Exists today. Crate-private class caches. |
| `CachedGraphPosteriorIdentification`, `CachedGraphPosteriorAtomIdentification`, `CachedClassPosteriorAtomIdentification`, `CachedClassPosteriorCase` | `crates/antecedent/src/analysis/identification_cache/graph_posterior.rs` | 2 | `antecedent` | Exists today. Per-atom identification. The plan view's `MixtureView` reads it as one logical plan. |
| `CachedDbnPosteriorIdentification`, `CachedDbnPosteriorAtomIdentification` | `crates/antecedent/src/analysis/identification_cache/dbn.rs` | 2 | `antecedent` | Exists today. |
| `CachedTemporalClassPosteriorIdentification`, `CachedTemporalClassPosteriorAtomIdentification` | `crates/antecedent/src/analysis/identification_cache/temporal_class.rs` | 2 | `antecedent` | Exists today. |
| `PreparedExecution`, `ExecutionKind` | `crates/antecedent/src/analysis/prepared/execution.rs` | 3 | `antecedent` | Exists today. Sealed operation or `OrdinaryDispatch`. |
| `CheckedProgramBinding` | `crates/antecedent/src/analysis/prepared/execution.rs` | 2 | `antecedent` | Exists today. One of AIPW, front-door, IV, linear, `FunctionalProgram`, functional-response members, distribution program, or nested counterfactual. |
| `RetainedOperation`, `PreparedStudy::retained_operation` | `crates/antecedent/src/analysis/prepared/info.rs` | 3 | `antecedent` | Exists today. One variant per `PreparedExecution` variant, matched exhaustively: the operation's public inspection record, the checked lowering or functional program it retains, or its name. It replaces the per-route `checked_*` and `has_checked_*_operation` accessors. |
| `PlanView`, `PreparedStudy::plan_view`, `Study::plan_view`, `PhysicalExecutionPlan::plan_view` | `crates/antecedent/src/planner/plan_view.rs` | 1 to 6 | `antecedent` | Exists today. See [The plan view](#the-plan-view). |

`PreparedExecution` variants, all in `crates/antecedent/src/analysis/prepared/execution.rs`, crate `antecedent`, layer 3 (each retains a checked operation and calls estimators on the estimate click): `FunctionalEffect`, `PathSpecificEffect`, `AdmgResponseCurve`, `AdmgGraphPosteriorResponse`, `CheckedLinear`, `CheckedGlmAdjustment`, `CheckedRd`, `CheckedPropensity`, `CheckedConditional`, `BayesianConditional`, `GraphPosteriorEffect`, `ClassGraphPosteriorEffect`, `BayesianGraphPosteriorAte`, `StaticClassEffect`, `CheckedAipw`, `Counterfactual`, `NestedCounterfactual`, `DerivativeResponse`, `CellAipwResponse`, `StaticDagResponse`, `StaticMediation`, `BayesianStaticMediation`, `Attribution`, `TemporalDagResponse`, `TemporalDagEffect`, `BayesianTemporalDagEffect`, `TemporalMediation`, `Interference`, `Randomized`, `PolicyValue`, `ContinuousDoseResponse`, `PanelDid`, `SyntheticControl`, `LocalPolynomialRatio`, `Survival`, `LongitudinalRegime`, `TemporalClassEffect`, `Distribution`, `BayesianGcomp`, `BayesianBasisAte`, `BayesianBasisCate`, `BayesianRobustAte`, `StaticResponseCurve`, `FrontDoorLinear`, `Iv`, `UnknownTieredAverage`, `BayesianClassConditional`, `TemporalClassResponse`, `TemporalGraphPosteriorResponse`, `TemporalClassMediation`, `BayesianTemporalClassEffect`, `TemporalGraphPosteriorEffect`, `StaticClassResponse`, `GraphPosteriorResponse`, `BayesianSpecialist`, `TransportTrial`.

Python `PreparedAnalysis` (`python/antecedent/estimation.py`) is the same handle: `inspect`, `inspect_plan`, `plan_view`, `estimate`, `retarget`, `refresh`, `refute`.

## Scientific problem

Exists today in `antecedent-core` unless noted. These types are the problem statement the compiler reads. They are not identification products.

| Type or function | Path | Layer | Crate | Notes |
| --- | --- | --- | --- | --- |
| `CausalQuery` and the query structs (`AverageEffectQuery`, `ResponseQuery`, `TemporalEffectQuery`, `ConditionalEffectQuery`, `MediationQuery`, `InterventionalDistributionQuery`, `PathSpecificEffectQuery`, `TransportQuery`, `InterferenceQuery`, `RandomizedEffectQuery`, `PolicyValueQuery`, `PanelDidQuery`, `SyntheticControlQuery`, `ContinuousDoseResponseQuery`, `LocalPolynomialRatioQuery`, `SurvivalQuery`, `LongitudinalRegimeQuery`, `CrossWorldQuery`, and the other `CausalQuery` arms) | `crates/antecedent-core/src/query.rs` | 1 | `antecedent-core` | Exists today. |
| `CausalSchema`, `VariableSchema` | `crates/antecedent-core/src/schema.rs` | 1 | `antecedent-core` | Exists today. Names at the boundary, `VariableId` on the hot path. |
| `Intervention`, `InterventionSequence`, `TemporalPolicy`, `StochasticPolicy` | `crates/antecedent-core/src/intervention.rs` | 1 | `antecedent-core` | Exists today. |
| `ValueExpr` | `crates/antecedent-core/src/value_expr.rs` | 1 | `antecedent-core` | Exists today. Value language for `Intervention::Transform` and `Assign`. Compiled into `antecedent-model` overlays, not into `antecedent-expr`. |
| `TargetPopulation`, `OutcomeFunctional`, `ResponseFunctional`, `EvidenceCatalog` | `crates/antecedent-core/src/query.rs` | 1 | `antecedent-core` | Exists today. `EvidenceCatalog` declares evidence. It is not the ledger. |
| `Assumption`, `AssumptionSet`, `AssumptionRecord` | `crates/antecedent-core/src/assumption.rs` | 1 | `antecedent-core` | Exists today. |
| `ObligationRecord` | `crates/antecedent-contract/src/obligation.rs` | 1 | `antecedent-contract` | Exists today. Extends assumption records. Carried into layer 2 when identification leaves one unresolved. |
| `Dag`, `Admg`, `Cpdag`, `Pag`, temporal graph types | `crates/antecedent-graph` | 1 | `antecedent-graph` | Exists today. Causal graph. Distinct from the evidence ledger and from the physical schedule. |
| `TabularData`, `TimeSeriesData`, `PanelData`, `MultiEnvironmentData`, `DiscoveryEstimationSplit` | `crates/antecedent-data` | 1 | `antecedent-data` | Exists today. Observation storage and the discovery/estimation window. Content digest is computed in `OwnedColumnarStorage` without an IO dependency. |

## Identification product

| Type or function | Path | Layer | Crate | Notes |
| --- | --- | --- | --- | --- |
| `IdentifiedEstimand`, `EstimandMethod` | `crates/antecedent-expr/src/estimand.rs` | 2 | `antecedent-expr` | Exists today. Re-exported by `antecedent-identify`. |
| `CausalExprArena`, `ExprId` | `crates/antecedent-expr` | 2 | `antecedent-expr` | Exists today. Identified-functional IR. Arena ids are not durable identity. |
| `FunctionalProgram`, `ProgramSchema`, `ProgramEvaluator` | `crates/antecedent-expr/src/program.rs` | 2 | `antecedent-expr` | Exists today. Checked expression program. The module states that structural checking does not establish identification. Not a statistical program and not a physical plan. |
| `ExactEvaluationPlan` | `crates/antecedent-expr/src/exact_plan.rs` | 2 | `antecedent-expr` | Exists today. Discrete-law evaluation plan for an identified functional. Not `PhysicalExecutionPlan`. |
| `Identifier`, `IdentificationWorkspace` | `crates/antecedent-identify/src/identifier.rs` | 2 | `antecedent-identify` | Exists today. Sealed identify trait. |
| `IdentificationResult`, `IdentificationResultParts`, `EstimandClaim`, `DerivationTrace`, `DerivationStep`, `IdentificationPerformanceRecord` | `crates/antecedent-identify/src/result.rs` | 2 | `antecedent-identify` | Exists today. Status, estimands, arena, derivation, assumptions, diagnostics, hedge. `candidates_examined` is execution accounting and stays out of the identification-product digest. |
| `IdentificationStatus` | `crates/antecedent-core/src/identification.rs` | 2 | `antecedent-core` | Exists today. Vocabulary only. |
| `BackdoorIdentifier`, `EfficientBackdoorIdentifier`, `FrontDoorIdentifier`, `IdIdentifier`, `IdcIdentifier`, `AutoIdentifier`, `GeneralizedAdjustmentIdentifier`, `InstrumentalVariableIdentifier`, `PathSpecificIdentifier`, `ResponseIdentifier`, `SharpRdIdentifier`, `TemporalBackdoorIdentifier`, `TemporalMediationIdentifier`, `TransportIdentifier` | `crates/antecedent-identify/src/` | 2 | `antecedent-identify` | Exists today. Algorithm entry points. `identify_tiered` and `identify_tiered_joint` are the tiered entries. |
| `IdentificationEnvelope`, `HedgeCertificate` | `crates/antecedent-identify/src/envelope.rs`, `hedge.rs` | 2 | `antecedent-identify` | Exists today. Class mass and the non-identification witness. |
| `TemporalIdentificationResult`, `TemporalClassEnvelope` | `crates/antecedent-identify/src/temporal_backdoor.rs`, `temporal_generalized.rs` | 2 | `antecedent-identify` | Exists today. |
| Transport and recovery decisions (`decide_z_transport_with_catalog`, `decide_mz_transport`, `decide_mixed_source`, `decide_observation_recovery`, `identify_counterfactual_transport`, and the bound functionals those modules return) | `crates/antecedent-identify/src/sid/`, `recovery.rs` | 2 | `antecedent-identify` | Exists today. Identification decisions with explicit missing-evidence and budget receipts. Not design ranking. |

## Statistical program

`StatisticalProgram` (`crates/antecedent/src/planner/statistical.rs`) is layer three of every lowered program: the resolved identifier and estimator, the `Realization` the inference mode selects (a frequentist score, an ordinary or generalized Bayesian likelihood, an identified-set envelope, or an exact point), the fitted models as `ModelSlot`s with their roles and variables, the cross-fitting `FoldPlan`, the `ResamplingPlan` under the declared sampling design, and the `PosteriorPlan`. Its identity enters the lowered program's identity. The estimators below remain the entry points the program is realized by; `antecedent-learn` fits predictors and does not own the causal statistical program.

| Type or function | Path | Layer | Crate | Notes |
| --- | --- | --- | --- | --- |
| `Estimator::prepare`, `Estimator::fit` | `crates/antecedent-estimate/src/estimator.rs` | 3 | `antecedent-estimate` | Exists today. Sealed. Implemented for `LinearAdjustmentAte`. Most estimators use inherent `prepare` / `fit` instead. |
| `PreparedEstimationProblem`, `EstimationWorkspace`, `LinearAdjustmentAte`, `CheckedLinearAdjustmentAte` | `crates/antecedent-estimate/src/adjustment.rs` | 3 | `antecedent-estimate` | Exists today. |
| `compile_adjustment_design`, `design_width::quote_adjustment_design` | `crates/antecedent-estimate/src/design_compile.rs`, `design_width.rs` | 3 | `antecedent-estimate` | Exists today. Expands covariates to a float design. Contrast matrices come from `antecedent-data`. `CompiledDesign` lives in `antecedent-stats`. The lowering quotes the same width at compile and layer five re-plans the physical record from it. |
| `JointFitPlan` | `crates/antecedent-estimate/src/fit_plan.rs` | 3 | `antecedent-estimate` | Exists today. Joint fit schedule inside estimate. Not a `PhysicalExecutionPlan`. |
| `ScoreTable`, `ScoreInference` | `crates/antecedent-estimate/src/scores.rs` | 3 | `antecedent-estimate` | Exists today. AIPW and cell-AIPW scores. Frozen on `PreparedStudy` when the cell exports them. |
| `CheckedAipwPreparation`, `CheckedFrontDoorPreparation`, `CheckedIvPreparation` | `crates/antecedent-estimate/src/aipw.rs`, `frontdoor.rs`, `iv.rs` | 3 | `antecedent-estimate` | Exists today. Bound into `CheckedProgramBinding`. |
| `ContinuousResponseEstimator`, `TemporalResponseEstimator`, `TemporalMediationEstimator`, `ObservationMechanismEstimator` | `crates/antecedent-estimate/src/response/`, `temporal_response/`, `temporal_mediation.rs`, `observation.rs` | 3 | `antecedent-estimate` | Exists today. Representative inherent estimators. The crate surface is the module list in `antecedent-estimate/src/lib.rs`. |
| `InferenceBackend`, `BayesFitOptions`, `BayesFitResult` | `crates/antecedent-prob/src/backend.rs` | 3 | `antecedent-prob` | Exists today. |
| `fit_conjugate_gaussian`, `fit_laplace_glm`, `fit_hmc_glm` | `crates/antecedent-prob/src/conjugate.rs`, `laplace.rs`, `hmc.rs` | 3 | `antecedent-prob` | Exists today. Posterior algorithms. |
| `PriorSpec`, `PriorSet`, `PosteriorDraws`, `PosteriorSummary` | `crates/antecedent-prob/src/prior.rs`, `posterior.rs` | 3 | `antecedent-prob` | Exists today. |
| `LearnerFactory`, `FittedPredictor`, `DesignView` | `crates/antecedent-learn/src/learner.rs`, `design.rs` | prediction | `antecedent-learn` | Exists today. Prediction contract used as a nuisance by estimate and as a mechanism family by model. Not layer 3 ownership. |
| `CompiledCausalModel`, `StructuralCausalModel`, `ProbabilisticCausalModel`, `ParentGatherPlan` | `crates/antecedent-model/src/compile.rs` | model companion | `antecedent-model` | Exists today. Mechanism compilation for simulation and counterfactuals. `Study::compile` does not lower these into `PhysicalExecutionPlan`. |
| `NodeProgram` | `crates/antecedent-model/src/program.rs` | model companion | `antecedent-model` | Exists today. Per-node mechanism or policy program. Not `FunctionalProgram`. |
| `TemporalPlan` | `crates/antecedent-model/src/temporal_rollout.rs` | model companion | `antecedent-model` | Exists today. Rollout plan. Not the analysis physical plan. |

## Evidence and decision program

`Study::compile_for_decision` and `Study::compile_for_evsi` (`crates/antecedent/src/analysis/execute/compile_decision.rs`) start compile from a `DecisionProblem` or an `EvsiRequest`. Layer four of the lowered program (`EvidenceDecisionProgram`, `DecisionProgram`, `EvsiProgram` in `crates/antecedent/src/planner/lowered/decision.rs`) then carries the demanded carrier (mean, parameter posterior or outcome law), the actions, constraints, chance threshold, utility form and unit, and the contract identity. For EVSI it also carries the canonical candidates, the prior observations, the ranking schedule, the seed, the tolerances and the quoted candidate × draw Monte Carlo store. The lowering checks the demanded carrier against the layer-three realization and refuses with a typed `DecisionCarrierError` that names the next attempt. A confidence interval is not a parameter posterior, and parameter draws are not individual outcome risk. An EVSI store the admission pool cannot hold is refused before compile allocates. `design_ranking::evaluate_for_study` binds a ranking to that lowered program. The remaining evidence and decision types exist beside that path.

| Type or function | Path | Layer | Crate | Notes |
| --- | --- | --- | --- | --- |
| `EvidenceLedger`, `EvidenceSource`, `LedgerOperation`, `SynthesisOperation` | `crates/antecedent-contract/src/evidence_ledger.rs` | 4 | `antecedent-contract` | Exists today. The ledger does no statistics. Pooling numbers stay in `antecedent-stats`. |
| `check_propagation`, `PropagationPlan`, `PropagationRequest` | `crates/antecedent-contract/src/joint_law_rule.rs` | 4 | `antecedent-contract` | Exists today. Admits or refuses a joint-law combination. |
| `MechanismCompositionPlan`, `check_mechanism_composition` | `crates/antecedent-contract/src/mechanism_composition.rs` | 4 | `antecedent-contract` | Exists today. Declared mechanism composition against the ledger. |
| `EvidenceObligation`, `EvidenceOffer` | `crates/antecedent-contract/src/evidence_obligation.rs` | 4 | `antecedent-contract` | Exists today. |
| `ScientificQuantity` | `crates/antecedent-contract/src/quantity.rs` | 4 | `antecedent-contract` | Exists today. Quantity, units, regime, and role on a composed object. |
| `DecisionProblem`, `AffineUtility`, `evaluate_decision` | `crates/antecedent-design/src/decision.rs` | 4 | `antecedent-design` | Exists today. |
| `EvsiRequest`, `evaluate_evsi` | `crates/antecedent-design/src/evsi.rs` | 4 | `antecedent-design` | Exists today. Preposterior value of sample information. |
| `DesignObjective`, ranker and `ScoredDesignRanking` | `crates/antecedent-design/src/objective.rs`, `ranker.rs`, `result.rs` | 4 | `antecedent-design` | Exists today. Experiment ranking. Depends on `antecedent-io-core`, not on `antecedent-estimate`. |
| `ReasoningView` and the four slots | `crates/antecedent-contract/src/reasoning.rs` | 6 | `antecedent-contract` | Exists today. Projection of identification, support, uncertainty, and assumptions onto a claim. Unknown is `Unavailable`. |
| `SemanticLayer`, `TransformEffect`, `TransformationReport` | `crates/antecedent-contract/src/transform.rs` | 6 | `antecedent-contract` | Exists today. Preview and apply effects for refresh and retarget. Different axis from this document's six layers. |

## Claim and receipt

| Type or function | Path | Layer | Crate | Notes |
| --- | --- | --- | --- | --- |
| `ContractIdentities`, `IdentityDomain`, `SemanticDigest`, `IdentityRef` | `crates/antecedent-core/src/identity.rs` | 6 | `antecedent-core` | Exists today. Domain-separated digest vocabulary. Bytes are produced in IO. |
| `ExecutionReceipt`, `RequestIdentity`, `ExecutionRequestState` | `crates/antecedent-core/src/execution.rs` | 6 | `antecedent-core` | Exists today. Host request around `ExecutionContext`. |
| `ExecutionContext`, `AdmissionPool`, `CancellationToken`, `KernelPolicy` | `crates/antecedent-core/src/execution.rs` | 5 | `antecedent-core` | Exists today. Budget and policy the physical planner reads. Not the plan. |
| `CausalContract` | `crates/antecedent/src/analysis/contract.rs` | 6 | `antecedent` | Exists today. Immutable companion of `Study` / `PreparedStudy`. Not a second builder. |
| `ClaimEnvelope`, `HandoffReceipt`, `AcceptanceReport`, `DerivedClaim` | `crates/antecedent-contract/src/claim.rs` | 6 | `antecedent-contract` | Exists today. Portable claim over a contract and one execution. |
| `StudyResult` | `crates/antecedent/src/result.rs` | 6 | `antecedent` | Exists today. Bundles logical record, physical record, `IdentificationResult`, `IdentifiedEstimand`, `PrimaryEstimate`, payload, optional posterior, refutations, diagnostics. One facade result, not a separated receipt. |
| Artifact consume and `verify_contract_against_body` | `crates/antecedent-io`, `crates/antecedent-io-core` | 6 | `antecedent-io` / `antecedent-io-core` | Exists today. Estimator-free replay stays in `antecedent-io-core`. Estimator-bound replay stays in `antecedent-io`. |

## The lowered program

`LoweredProgram` (`crates/antecedent/src/planner/lowered.rs`) is the mandatory output of compile for every `RouteCell`. Every cell lowers at `LoweringDepth::Complete`; there is no partial lowering.

| Layer | Field | Notes |
| --- | --- | --- |
| 1. Scientific problem | `ScientificProblem` | `ProblemTarget::Target` (a world-indexed `ScientificTarget`: worlds, a contrast, a dose-time surface, or mediation), `ProblemTarget::Targets` (one target per requested horizon, for a multi-horizon mediation or a per-horizon response), or `ProblemTarget::QueryFunctional` (a typed `FunctionalKind` with the canonical digest of the query, for a query no world form states: a distribution, a derivative or Jacobian, a cross-world or attribution query, a callback action, a schedule outside its horizon). The digest is the query's `CanonicalEncode` encoding (every query type, grid, weighting, observation declaration, design array and evidence catalog has one beside its definition in `antecedent-core`); set-valued inputs are written in a canonical order. Two queries of one kind lower to different programs. Also the route record, data modality, graph class and support coordinate. |
| 2. Identification product | `IdentificationLayer` | `Retained`: identified once at compile with the products preparation installs, keyed by the retention key (below). `PerExecution`: the route identifies on each execution (sharp RD, cross-world probes), with the reason. `Deferred`: identification refused at compile, with the reason; preparation identifies and reports the same refusal. |
| 3. Statistical program | `StatisticalProgram` | See [Statistical program](#statistical-program). |
| 4. Evidence and decision program | `EvidenceDecisionProgram` | Source contributions and composition, and the decision or EVSI program when compile started from one. |
| 5. Physical layer | `PhysicalLayer` | Physical operator contracts and the `DesignShape`: a quoted design with its rows, encoded columns and gathering, or `NotDesignBased(reason)` with no rows or columns (the planner's sizing stays on the physical record, which is not re-planned). Execution accounting: it stays out of the identity. |
| 6. Claim | `ClaimDemand` | The uncertainty reading and claim kind the execution must publish. The measured receipt stays on the result. |

Preparation installs a retained identification instead of identifying again. Every executor (static, class, mixed-graph, temporal and temporal-class, structural and attribution, transport, design and graph-posterior) takes its identification in one order: the prepared cache, then the product the plan's lowered program retained, then a fresh identification. A retained product is taken only when the study's retention key matches the one compile recorded. The key digests every input identification reads: the query (with its effect modifiers), the data modality and schema variables, the structure of any graph class or the atoms and weights of a graph posterior, the tier background, the completion cap and class prior, the selection diagram and transport trial, the identifier, estimator and discontinuity design, and the inference mode. A reused product that did not come from a prepared handle publishes the diagnostics of a fresh identification (no `exec.identify.cached`). Compile reports `identify.compute` once when it retains a product, so a one-shot run reports exactly one identification. `scripts/gate_checked_execution.py` fails on a licensed route without a registered `lowering` cell, a surviving partial lowering, library code that re-enters the public builder, or an `ExecutionContext` literal outside `antecedent-core`.

## The plan view

`PlanView` (`crates/antecedent/src/planner/plan_view.rs`) is one view over every layer of a lowered program. It is built from the program, before execution. Building it does not identify, fit, scan rows, enumerate graphs, simulate or lease from the admission pool. Its identity is the lowered program's identity. Every route a view describes is a `PlanRoute`. The route table in `registry.rs` generates `PlanRoute` and each cell's `RouteFamily` from the same rows that dispatch the lowering. The view also matches every `RouteFamily` exhaustively. So a route cell, or a family, added without a view does not compile. The guide is [Plan view](../guides/plan-view.md).

| Stage | Entry | Work done | Shown |
| --- | --- | --- | --- |
| Preflight | `Study::inspect`, `PreparedStudy::inspect` | None beyond the structural record; no identification. | No view. Identification and uncertainty slots stay `Unavailable` (`compiler.inspect_does_not_identify`). |
| Compiled | `Study::plan_view(ctx)` | `Study::compile`. Identification retained with the program runs once, and a linear adjustment quotes its design width. No fit, sample, simulation or estimate. | Every layer, the physical record, the support status and the compile context's kernel policy. A graph posterior's atoms come from the study's posterior. |
| Compiled | `PhysicalExecutionPlan::plan_view()` | None. | Every layer and the physical record. No support status, and no graph-posterior atom kind. |
| Prepared | `PreparedStudy::plan_view()` | None. | The compiled view, the identification preparation installed (`installed_at_prepare`), and the name of the sealed operation it retained. `retained_operation()` returns that operation as a typed `RetainedOperation`. |
| Prepared | `plan_view()` on a cyclic family handle; `PlanningPlan::plan_view()` | None. | Every layer of the cyclic or planning program. There is no physical record. |

| View section | Reads | Notes |
| --- | --- | --- |
| `scientific()` | Layer 1 | Target, plan id, modality, graph class, query variables. |
| `identification()` | Layer 2, or what preparation installed | Kind (`retained`, `deferred`, `per_execution`, `non_graphical`) with its reason; the checked product (outcome, claim class, estimands, assumptions, required factors, unresolved obligations); one `RetainedProductView` per installed product (status, estimands with adjustment sets, assumptions, horizons, completions, masses, transport outcome); the retention key. |
| `statistical()` | Layer 3 | `StatisticalProgram` itself. |
| `evidence()` | Layer 4 | `EvidenceDecisionProgram` itself: sources, composition, decision with carrier and EVSI, planning request. |
| `physical()` | Layer 5 and the physical record | Operators and their summed scratch, retained and output byte estimates; the quoted design width or the not-design-based reason; the materializations, kernels, batch, workspace, peak and copy bytes, task schedule and workers; the kernel policy. |
| `uncertainty()`, `claim()` | Layers 3, 4 and 6 | Uncertainty reading, claim kind, realization, resampling and posterior plans, the decision's evidence-port request. |
| `support()` | Layer 1 and the study | Coordinate, licensed flag, cell status. |
| `mixture()` | Layer 2 and 4 of a graph-posterior program | One atom per distinct structure: key, weight, samples, identified flag, status, estimand, class completions and envelope mass; total, identified and unidentified mass; the atom mixture composition. This is the single logical plan of a graph-posterior analysis. |
| `to_json()` | All of the above | Python `prepared.plan_view()` decodes it into `ProgramPlanView` with variables named. |

## Remaining gaps

These are observations about the current tree.

- Layer 1 is still read from `CausalQuery` plus schema, graph, assumptions, and data; `LogicalAnalysisPlan` stays route, query, split, and strategy-name metadata. A functional target is keyed by the query's canonical digest, not stated as worlds.
- Layer 2: a one-shot executor whose published claim is not the claim compile retains identifies its own claim, so its published result stays that of a fresh identification. These are a static claim narrowed to one of several per-estimand alternatives, or carrying a discontinuity design the route ignores; an IDC claim with several estimands; and a mechanism-overlay schedule whose levels the route strips. Every other one-shot execution takes the retained product.
- Layer 4: an outcome law is supplied only by the interventional-distribution cells. The EVSI Monte Carlo dimension sits on `EvsiProgram`, not on the physical layer's operators.
- Layer 5 is one dense-design estimate per route: one copied or borrowed buffer, one kernel label, one task dimension, no scan or projection operators, no spill, and no use of `BufferMaterialization::Chunked`. Only linear adjustment and ANCOVA designs are quoted; the other routes say what they fit instead. `antecedent-state` (`CausalState`, `InvalidationLog`, sufficient statistics) is retained state, not this plan.
- Layer 6 exists as identities, `CausalContract`, `ClaimEnvelope`, and `StudyResult`. The result still carries the route records and the identification product in one struct. `ExecutionPerformanceRecord` can store measured facts; the physical planner's byte counts are estimates, not measurements.
- Three graphs are not three IRs. The causal graph is `antecedent-graph`. Provenance and overlap are `EvidenceLedger` and `ProvenanceGraph`. The execution schedule is the single `task_schedule` on `PhysicalExecutionPlanRecord`.
