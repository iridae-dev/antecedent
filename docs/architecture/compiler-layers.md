# Compiler layers

The six layers are ownership boundaries inside the existing compiler. They are
not new crates and not a second public workflow. `PreparedStudy` and Python
`PreparedAnalysis` stay the practitioner handle. `antecedent-expr` stays the
identified-functional IR.

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

`compile_physical_with_all_graphs` estimates peak bytes as `n_rows * 8 * 8` for the design plus the same again for workspace, always materializes `design.matrix` as `CopiedContiguous`, picks `DenseBackend` under the estimator's `kernel_label` (fallback `ols.faer`), schedules one task dimension from `EstimatorId::parallel_task_dimension` (fallback `analysis`), and sets `expected_python_crossings` to 1. A soft memory limit with no chunked alternative returns `CausalError::Resource`.

## Compile

`Study::route_input` (`crates/antecedent/src/analysis/execute/route_input.rs`) admits one `RouteInput` cell. `Study::compile_logical` and `Study::compile` (`crates/antecedent/src/analysis/execute/compile.rs`) dispatch that cell. A graph posterior has no single logical plan: `compile_logical` returns `CausalError::Unsupported`; `compile` calls `compile_graph_posterior`.

| Type or function | Path | Layer | Crate | Notes |
| --- | --- | --- | --- | --- |
| `RouteInput` | `crates/antecedent/src/analysis/execute/route_input.rs` | 1 | `antecedent` | Exists today. Borrowed data, query, and graph for one admitted cell, including `StaticAteMixed`, which logical compile still refuses. |
| `StaticMediationSource` | `crates/antecedent/src/analysis/execute/route_input.rs` | 1 | `antecedent` | Exists today. `MediationQuery` or `NestedCounterfactualQuery`. |
| `PlanId` | `crates/antecedent/src/analysis/execute/plan_id.rs` | 1 | `antecedent` | Exists today. Wire `plan_id` for the routes that do not use the planner's fixed ids (`static_conditional`, `transport_trial`, `gcm_query`, longitudinal regime ids, and the rest of that enum). |
| `Study::compile_logical` | `crates/antecedent/src/analysis/execute/compile.rs` | 1 | `antecedent` | Exists today. Inspectable route record only. |
| `Study::compile` | `crates/antecedent/src/analysis/execute/compile.rs` | 5 | `antecedent` | Exists today. Logical record, then `compile_physical*`. Attaches a resolved DAG, temporal DAG, or PAG when the cell has one. `StaticAteMixed` is physical-only. |
| `Study::compile_logical_from` and the `compile_logical_*` route builders in this file | `crates/antecedent/src/analysis/execute/compile.rs` | 1 | `antecedent` | Exists today. One builder per `RouteInput` arm: static response DAG/PAG/CPDAG, co-determined joint, mixed ADMG response, static ATE DAG/PAG/CPDAG, distribution DAG and mixed, path-specific, temporal effect and response, multi-env temporal effect, panel effect and response, conditional DAG/CPDAG/PAG, temporal and static mediation, transport, interference, randomized, policy value, panel DiD, synthetic control, continuous dose, local-polynomial ratio, survival, longitudinal regime, GCM, cross-world. Each fills a `LogicalAnalysisPlanRecord`. |
| `Study::run` | `crates/antecedent/src/analysis/execute/dispatch.rs` | 3 | `antecedent` | Exists today. One-shot sugar over identify, prepare, and estimate. Returns `StudyResult`. |

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
| `Study::inspect` | `crates/antecedent/src/analysis/contract.rs` | 1 | `antecedent` | Exists today. Cheap structural inspection. Identification slot and program identity are absent. |
| `CachedStaticIdentification`, `CachedTemporalIdentification`, `CachedTemporalHorizonIdentification` | `crates/antecedent/src/analysis/prepared/cached.rs` | 2 | `antecedent` | Exists today. Public cached products. |
| `CachedPagIdentification`, `CachedCpdagIdentification`, `CachedTemporalClassIdentification` | `crates/antecedent/src/analysis/prepared/cached.rs` | 2 | `antecedent` | Exists today. Crate-private class caches. |
| `CachedGraphPosteriorIdentification`, `CachedGraphPosteriorAtomIdentification`, `CachedClassPosteriorAtomIdentification`, `CachedClassPosteriorCase` | `crates/antecedent/src/analysis/identification_cache/graph_posterior.rs` | 2 | `antecedent` | Exists today. Per-atom identification. Not one logical plan. |
| `CachedDbnPosteriorIdentification`, `CachedDbnPosteriorAtomIdentification` | `crates/antecedent/src/analysis/identification_cache/dbn.rs` | 2 | `antecedent` | Exists today. |
| `CachedTemporalClassPosteriorIdentification`, `CachedTemporalClassPosteriorAtomIdentification` | `crates/antecedent/src/analysis/identification_cache/temporal_class.rs` | 2 | `antecedent` | Exists today. |
| `PreparedExecution`, `ExecutionKind` | `crates/antecedent/src/analysis/prepared/execution.rs` | 3 | `antecedent` | Exists today. Sealed operation or `OrdinaryDispatch`. |
| `CheckedProgramBinding` | `crates/antecedent/src/analysis/prepared/execution.rs` | 2 | `antecedent` | Exists today. One of AIPW, front-door, IV, linear, `FunctionalProgram`, functional-response members, distribution program, or nested counterfactual. |

`PreparedExecution` variants, all in `crates/antecedent/src/analysis/prepared/execution.rs`, crate `antecedent`, layer 3 (each retains a checked operation and calls estimators on the estimate click): `FunctionalEffect`, `PathSpecificEffect`, `AdmgResponseCurve`, `AdmgGraphPosteriorResponse`, `CheckedLinear`, `CheckedGlmAdjustment`, `CheckedRd`, `CheckedPropensity`, `CheckedConditional`, `BayesianConditional`, `GraphPosteriorEffect`, `ClassGraphPosteriorEffect`, `BayesianGraphPosteriorAte`, `StaticClassEffect`, `CheckedAipw`, `Counterfactual`, `NestedCounterfactual`, `DerivativeResponse`, `CellAipwResponse`, `StaticDagResponse`, `StaticMediation`, `BayesianStaticMediation`, `Attribution`, `TemporalDagResponse`, `TemporalDagEffect`, `BayesianTemporalDagEffect`, `TemporalMediation`, `Interference`, `Randomized`, `PolicyValue`, `ContinuousDoseResponse`, `PanelDid`, `SyntheticControl`, `LocalPolynomialRatio`, `Survival`, `LongitudinalRegime`, `TemporalClassEffect`, `Distribution`, `BayesianGcomp`, `BayesianBasisAte`, `BayesianBasisCate`, `BayesianRobustAte`, `StaticResponseCurve`, `FrontDoorLinear`, `Iv`, `UnknownTieredAverage`, `BayesianClassConditional`, `TemporalClassResponse`, `TemporalGraphPosteriorResponse`, `TemporalClassMediation`, `BayesianTemporalClassEffect`, `TemporalGraphPosteriorEffect`, `StaticClassResponse`, `GraphPosteriorResponse`, `BayesianSpecialist`, `TransportTrial`.

Python `PreparedAnalysis` (`python/antecedent/estimation.py`) is the same handle: `inspect`, `inspect_plan`, `estimate`, `retarget`, `refresh`, `refute`.

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

No type today is the statistical program. Fitting is a set of estimator entry points selected by `EstimatorId` and by the sealed prepared operation. `antecedent-learn` fits predictors; it does not own the causal statistical program.

| Type or function | Path | Layer | Crate | Notes |
| --- | --- | --- | --- | --- |
| `Estimator::prepare`, `Estimator::fit` | `crates/antecedent-estimate/src/estimator.rs` | 3 | `antecedent-estimate` | Exists today. Sealed. Implemented for `LinearAdjustmentAte`. Most estimators use inherent `prepare` / `fit` instead. |
| `PreparedEstimationProblem`, `EstimationWorkspace`, `LinearAdjustmentAte`, `CheckedLinearAdjustmentAte` | `crates/antecedent-estimate/src/adjustment.rs` | 3 | `antecedent-estimate` | Exists today. |
| `compile_adjustment_design` | `crates/antecedent-estimate/src/design_compile.rs` | 3 | `antecedent-estimate` | Exists today. Expands covariates to a float design. Contrast matrices come from `antecedent-data`. `CompiledDesign` lives in `antecedent-stats`. The physical planner does not read this width. |
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

The Study planner does not read a decision and then request a statistical program. Evidence and decision types exist beside that path.

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
| `ExecutionContext`, `MemoryBudget`, `CancellationToken`, `KernelPolicy` | `crates/antecedent-core/src/execution.rs` | 5 | `antecedent-core` | Exists today. Budget and policy the physical planner reads. Not the plan. |
| `CausalContract` | `crates/antecedent/src/analysis/contract.rs` | 6 | `antecedent` | Exists today. Immutable companion of `Study` / `PreparedStudy`. Not a second builder. |
| `ClaimEnvelope`, `HandoffReceipt`, `AcceptanceReport`, `DerivedClaim` | `crates/antecedent-contract/src/claim.rs` | 6 | `antecedent-contract` | Exists today. Portable claim over a contract and one execution. |
| `StudyResult` | `crates/antecedent/src/result.rs` | 6 | `antecedent` | Exists today. Bundles logical record, physical record, `IdentificationResult`, `IdentifiedEstimand`, `PrimaryEstimate`, payload, optional posterior, refutations, diagnostics. One facade result, not a separated receipt. |
| Artifact consume and `verify_contract_against_body` | `crates/antecedent-io`, `crates/antecedent-io-core` | 6 | `antecedent-io` / `antecedent-io-core` | Exists today. Estimator-free replay stays in `antecedent-io-core`. Estimator-bound replay stays in `antecedent-io`. |

## What is missing

The six-layer model is not implemented as internal IR. The gaps below are observations about the current tree.

- Layer 1 is `CausalQuery` plus schema, graph, assumptions, and data. `LogicalAnalysisPlan` is route, query, split, and strategy-name metadata. It is not a scientific-problem operator with evidence-use effects.
- Layer 2 exists (`IdentificationResult`, `IdentifiedEstimand`, the expr arena) and is cached on `PreparedStudy`. The physical plan does not take that product as input. Graph-posterior analyses have no single logical plan.
- Layer 3 has no program object. Folds, scores, priors, and resampling are fields of individual estimators and of `CheckedProgramBinding`, not a plan the next layer schedules. `FunctionalProgram` is layer 2. `NodeProgram` and `CompiledCausalModel` are model companions and are not inputs to `Study::compile`.
- Layer 4 exists as the ledger, joint-law rule, and design crate. `Study::compile` does not start from `DecisionProblem` or `EvsiRequest`, and design does not see the prepared physical plan.
- Layer 5 is one dense-design estimate: fixed column-width heuristic, one copied buffer, one kernel label, one task dimension, no scan or projection operators, no spill, and no use of `BufferMaterialization::Chunked`. `compile_adjustment_design` already knows a real design width and is not consulted. `antecedent-state` (`CausalState`, `InvalidationLog`, sufficient statistics) is retained state, not this plan.
- Layer 6 exists as identities, `CausalContract`, `ClaimEnvelope`, and `StudyResult`. The result still carries the route records and the identification product in one struct. `ExecutionPerformanceRecord` can store measured facts; the physical planner's byte counts are estimates, not measurements.
- Three graphs are not three IRs. The causal graph is `antecedent-graph`. Provenance and overlap are `EvidenceLedger` and `ProvenanceGraph`. The execution schedule is the single `task_schedule` on `PhysicalExecutionPlanRecord`.
