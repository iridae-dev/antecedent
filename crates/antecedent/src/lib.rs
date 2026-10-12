//! Unified static/temporal `Study` facade (identify → estimate → refute).
//!
//! Antecedent is an identification-first causal inference engine: identification
//! is evaluated before estimation, refuters run against the estimate, and
//! uncertainty about causal structure (equivalence classes, graph posteriors) is
//! retained rather than silently resolved.
//!
//! # Quick start
//!
//! ```
//! use antecedent::prelude::*;
//! use antecedent::RefuteSuite;
//!
//! let schema = CausalSchemaBuilder::new()
//!     .continuous("t")
//!     .treatment()
//!     .continuous("y")
//!     .outcome()
//!     .continuous("z")
//!     .context()
//!     .build()
//!     .unwrap();
//! let data = TabularData::from_f64_columns([
//!     ("t", &[0.0_f64, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0][..]),
//!     ("y", &[1.0_f64, 3.1, 1.2, 2.9, 0.9, 3.0, 1.1, 3.2][..]),
//!     ("z", &[0.0_f64, 1.0, 0.2, 0.8, 0.1, 0.9, 0.3, 0.7][..]),
//! ])
//! .unwrap();
//! let dag = Dag::from_named_edges(&schema, &[("z", "t"), ("z", "y"), ("t", "y")]).unwrap();
//! let q = AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap());
//! let result = Study::tabular(data)
//!     .graph(dag)
//!     .query(q)
//!     .refute(RefuteSuite::None)
//!     .bootstrap_replicates(0)
//!     .build()
//!     .unwrap()
//!     .run(&ExecutionContext::for_tests(1))
//!     .unwrap();
//! assert!(result.effect().is_finite());
//! ```
//!
//! Prefer [`prelude`] for day-1 imports. Component crates remain available for stage-specific work.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![warn(clippy::missing_errors_doc, clippy::missing_panics_doc)]

pub mod accepted;
pub mod analysis;
pub mod callback_plan;
pub mod class_prior;
pub mod counterfactual_id;
pub mod cross_world;
pub mod design;
pub mod discovery;
pub mod discovery_defaults;
pub mod error;
pub mod estimator_spec;
pub mod gcm;
mod graph_identity;
pub mod inference;
pub mod options;
pub mod planner;
pub mod prelude;
pub mod result;
pub mod review;
pub mod state;
pub mod strategy_table;
pub mod support;
pub use antecedent_io::coverage_records_data;

pub mod estimate;
pub mod graph;
pub mod identify;
pub mod identify_api;
pub mod io;
pub mod query;
pub mod validate;

// --- Day-1 crate-root surface (stage depth lives under modules) ---
pub use accepted::{AcceptedGraph, GraphClass, GraphRef, IntoAccepted};
#[doc(hidden)]
pub use analysis::temporal_dependent_interval_candidate;
pub use analysis::{
    ActionConstraint, ActionOutcome, ActionSpec, ActionStatus, EnumeratedStatus, ForwardEvaluation,
    ForwardInterval, INVERSE_SCOPE_NOTE, IntervalMeta, IntervalScope, InverseOutcomeError,
    InverseOutcomeReport, InverseQuery, MAX_INVERSE_ACTIONS, MAX_INVERSE_FORWARD_POINTS,
    SupportBasis, TargetDirection, classify_inverse_outcome,
};
/// The decision contract beside [`CausalContract`]: admissibility, support and
/// uncertainty requirements, claim adapters, robustness verdicts and the durable
/// artifacts with external-callback receipts.
pub use analysis::{
    ActionRobustness, AdaptedClaims, AdaptedDecision, AdapterError, AdmissibilityError,
    AdmissibilityRules, AdmissibleContractArtifact, AdmissibleDecisionContract, AtomSupport,
    BoundDecisionContract, ClaimProbability, ClaimProfile, DecisionClaimKind,
    DecisionDeclaredExclusion, DecisionFacadeRefusal, DecisionSupportRule, DecisionUncertaintyKind,
    ExternalCallbackReceipt, ExternalTrustLimit, IdentifiedActionRange, IdentifiedSetDecision,
    IdentifiedUtility, IdentifiedVerdict, InputSupport, ReplayReceipt, RobustDecisionResult,
    RobustResultArtifact, RobustVerdict, RobustnessError, SuppliedClaim, SupportShortfall,
    UncertaintyRequirement, adapt_finite_scenarios, adapt_graph_dependent_claims,
    adapt_point_claim, adapt_weighted_graph_atoms, assess_robustness, evaluate_adapted,
    evaluate_identified_sets, evaluate_robust, utility_interval,
};
pub use analysis::{
    AdmgConditionalResult, PreparedAdmgConditionalTransport,
    consume_admg_conditional_obstruction_artifact, consume_admg_conditional_transport_artifact,
    export_admg_conditional_obstruction,
};
pub use analysis::{
    ArmCount, ArmSpec, ArmWeightEss, BatchCostEstimate, BatchPreflightReport, ColumnMissingness,
    ColumnPriority, ColumnWeight, CostEstimate, DependentColumn, DroppedColumn, DuplicateGroup,
    FindingSeverity, FittedPropensity, InferenceDefault, JointCellCostEstimate,
    NuisanceFitDiagnostics, PreflightFinding, PreflightInput, PreflightReport, PropensityOutcome,
    RankDropEstimate, RankDropPlan, RankDropPolicy, RankReport, RetargetCostEstimate,
    ScoreQuantile, SpanCheck, estimate_joint_cell_cost, estimate_with_rank_drop,
    fit_diagnostics_design, plan_rank_drop, preflight_design,
};
pub use analysis::{
    BATCH_RETARGET_SCOPE_NOTE, BandMember, BatchRetargetError, BatchRetargetReport,
    BatchRetargetRequest, BatchScores, ClaimPoint, ClaimReport, ContrastPoint, ContrastReport,
    FamilyCovariance, MAX_T_MAX_DRAWS, MAX_T_MIN_DRAWS, MemberFailure, RetargetClaim,
    RetargetContrast, ScoreSource, SimultaneousBand, TidyKind, TidyRow, UncertaintyKind,
};
pub use analysis::{
    BatchQuery, BatchStudy, BlockedOperation, CandidateProcedure, CandidateScreen,
    CandidateSelection, CausalContract, CellFamilyContrast, CheckedAdmgGraphPosteriorResponseInfo,
    CheckedBayesianSpecialistInfo, CheckedGraphPosteriorResponseInfo, CheckedInterferenceInfo,
    CheckedStaticClassResponseInfo, CheckedTemporalClassMediationInfo,
    CheckedTemporalMediationInfo, CheckedTransportTrialInfo, CheckedUnknownTieredAverageInfo,
    ComputeBudget, ExactFactorRequirement, ExactPreparedState, ExactStudyIdentities,
    ExactStudyInspection, ExactStudyResult, InterferenceSpec, LatencyMode, LearnedTrialResult,
    LearnedTrialState, LicensedNeighbor, NextAction, OperationKind, OperationReadiness,
    OperationReport, PremiseChange, PreparedBatch, PreparedStudy, PreparedZTransport, RdConfig,
    RefuteSuite, SemanticApplicability, SharedBatchDesign, SharedCovariateDesign, StageEvent,
    StageResultSink, StatisticalBindingView, StatisticalContrast, StatisticalPreparedState,
    StatisticalStudyInspection, StatisticalStudyResult, Study, StudyBuilder, TransportTrialSpec,
    ZTransportResult, ZTransportSensitivityArtifactWire, consume_z_transport_artifact,
};
pub use analysis::{
    ConstituentRole, DeclaredExclusion, DerivedTreatmentDeclaration, DerivedTreatmentPlan,
    ExclusionRule, InterventionMeaning, SourceColumn, TemporalPosition, Transformation,
    check_derived_treatment, estimate_derived_joint_cells,
};
pub use analysis::{CpdagEvaluation, EmpiricalScenarioInput, EvidenceSearch, SequenceEndpoints};
pub use analysis::{
    IntervalEstimand, TemporalInitialState, TemporalWindow,
    consume_temporal_initial_state_artifact, consume_temporal_refresh_artifact,
    temporal_dependent_interval,
};
pub use analysis::{
    JOINT_SENSITIVITY_ARTIFACT_VERSION, JointFactorWire, JointOutcomeWire, JointPerturbationWire,
    JointProvenanceWire, JointReceiptWire, JointSamplingWire, JointSensitivityArtifactWire,
    JointSensitivityBodyWire, JointSensitivityConsumeLimits, JointTippingWire,
};
pub use analysis::{
    LearnedContinuousResult, PreparedLearnedContinuous, consume_learned_continuous_artifact,
};
pub use analysis::{MixedSourceResult, PreparedMixedSource, consume_mixed_source_artifact};
pub use analysis::{MzTransportResult, PreparedMzTransport, consume_mz_transport_artifact};
pub use analysis::{
    ObservationRecoveryResult, PreparedObservationRecovery, RecoveryEvaluation,
    consume_observation_recovery_artifact,
};
pub use analysis::{
    PreparedCpdagCompletionScenarios, consume_cpdag_scenarios_artifact,
    consume_scenario_covariance_artifact, export_scenario_covariance,
};
pub use analysis::{PreparedSmoothedDose, SmoothedDoseResult, consume_smoothed_dose_artifact};
pub use analysis::{PreparedTemporalTransport, consume_temporal_transport_artifact};
pub use analysis::{PreparedTransportScenarios, consume_transport_scenarios_artifact};
pub use analysis::{ScreenSplitReceipt, ScreenUnits};
pub use analysis::{
    TierDesign, TierDiagnostics, TierDiagnosticsError, TierEvalue, TierPointEvalue, TierScenario,
    tier_diagnostics,
};
#[doc(hidden)]
pub use analysis::{max_t_critical_value, simultaneous_band_unpublished};
pub use class_prior::ClassPrior;
pub use error::{CausalError, CompileCause, CompileFailure, ReviewKind};
pub use estimate::{CausalPosterior, EffectEstimate, EstimatorId, IdentifierId};
pub use estimator_spec::EstimatorSpec;
pub use graph::{Dag, DenseNodeId, TemporalDag};
pub use identify_api::{Identification, identify, identify_dag, identify_with};
pub use inference::{BayesianConfig, InferenceMode};
pub use options::FdrControl;
pub use query::{
    AnomalyAttributionQuery, AssignmentDesign, AverageEffectQuery, CausalQuery,
    ChangeAttributionQuery, ConditionalEffectQuery, ContinuousDomain, CounterfactualQuery,
    DerivativeScale, DerivativeWeighting, ExposureLevel, ExposureMapping, GridSpec,
    InterferenceFunctional, InterferenceQuery, InterventionalDistributionQuery,
    LocalPolynomialRatioQuery, LongitudinalRegimeQuery, MechanismChangeQuery, MediationContrast,
    MediationQuery, OutcomeFunctional, PanelDidQuery, PathSpecificEffectQuery, RandomizationDesign,
    RandomizedEffectQuery, ResponseFunctional, ResponseIdentification, ResponseQuery,
    ResponseUncertainty, ResponseValue, SyntheticControlQuery, SyntheticPanelMethod,
    TemporalEffectQuery, TemporalPolicy, TemporalResponseSpec, TransportQuery, UnitChangeQuery,
};
pub use result::{
    AnalysisIdentification, ContinuousDoseResponseEstimate, ExecutedContract,
    LocalPolynomialRatioEstimate, LongitudinalRegimeEstimate, PERCENTILE_95_MIN_REPLICATES,
    PanelDidEstimate, PolicyValueEstimate, PrimaryEstimate, PublishedScalarUncertainty,
    RandomizedEffectEstimate, ResultPayload, RowWeightsBinding, StructuralAggregationPolicy,
    StructuralWeightBasis, StudyResult, SurvivalEstimate, SyntheticControlEstimate,
    SyntheticDidEstimate,
};
pub use support::{
    CellStatus, IntoGraphInput, StructureSource, SupportCell, SupportRefusal, cell_coordinate,
    classify, licensed_neighbors, licensed_support_cells, refused_message,
    support_cell_from_coordinate,
};

// Strategy helpers and peer APIs: use `antecedent::estimate`, `antecedent::discovery`,
// `antecedent::gcm`, `antecedent::io`, `antecedent::design`, `antecedent::state`, `antecedent::graph`,
// `antecedent::identify`, `antecedent::review`, `antecedent::validate`, `antecedent::planner`.

#[cfg(test)]
mod tests;
