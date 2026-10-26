//! Curated analysis surface.
//!
//! Stage depth stays under `analysis`. The crate root re-exports this module,
//! so existing `antecedent::Study` paths stay where they are. The decision
//! contract sits beside [`crate::CausalContract`]: admissibility, support and
//! uncertainty requirements, claim adapters, robustness verdicts and the durable
//! artifacts with external-callback receipts.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use super::structure_route::{
    AcceptedPosterior, AttributionOutcome, AttributionPayload, CompiledExecution,
    DiscoveredStructure, DiscoveryPayload, DiscoverySpec, MechanismProgram, SimulationOutcome,
    SimulationPayload, accept_discovered, accept_discovered_posterior, panel_of_units,
};
pub use super::{
    ActionConstraint, ActionOutcome, ActionReason, ActionSpec, ActionStatus, EnumeratedStatus,
    ForwardEvaluation,
    ForwardInterval, INVERSE_SCOPE_NOTE, IntervalMeta, IntervalScope, InverseOutcomeError,
    InverseOutcomeReport, InverseQuery, SupportBasis, TargetDirection, classify_inverse_outcome,
};
pub use super::{
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
pub use super::{
    AdaptiveInferenceOutcome, AdaptiveInferenceRefusal, AdaptiveInferenceSession,
    ConsumedAdaptiveResult, ConsumedMonitoringContract, consume_adaptive_inference_result_artifact,
    consume_adaptive_monitoring_contract_artifact,
};
pub use super::{
    AdmgConditionalResult, PreparedAdmgConditionalTransport,
    consume_admg_conditional_obstruction_artifact, consume_admg_conditional_transport_artifact,
    export_admg_conditional_obstruction,
};
pub use super::{
    ArmCount, ArmSpec, ArmWeightEss, BatchCostEstimate, BatchPreflightReport, ColumnMissingness,
    ColumnPriority, ColumnWeight, CostEstimate, CostModel, CostModelError, DependentColumn, DroppedColumn, DuplicateGroup,
    FindingSeverity, FittedPropensity, InferenceDefault, JointCellCostEstimate,
    NuisanceFitDiagnostics, PreflightFinding, PreflightInput, PreflightReport, PropensityOutcome,
    RankDropEstimate, RankDropPlan, RankDropPolicy, RankReport, RetargetCostEstimate,
    ScoreQuantile, SpanCheck, estimate_joint_cell_cost, estimate_with_rank_drop,
    fit_diagnostics_design, plan_rank_drop, preflight_design,
};
pub use super::{
    BATCH_RETARGET_SCOPE_NOTE, BandMember, BatchRetargetError, BatchRetargetReport,
    BatchRetargetRequest, BatchScores, ClaimPoint, ClaimReport, ContrastPoint, ContrastReport,
    FamilyCovariance, MemberFailure, RetargetClaim,
    RetargetContrast, ScoreSource, SimultaneousBand, TidyKind, TidyRow, UncertaintyKind,
};
pub use super::{
    BatchQuery, BatchStudy, BlockedOperation, CandidateProcedure, CandidateScreen,
    CandidateSelection, CausalContract, CellFamilyContrast, CheckedAdmgGraphPosteriorResponseInfo,
    CheckedBayesianSpecialistInfo, CheckedGraphPosteriorResponseInfo, CheckedInterferenceInfo,
    CheckedStaticClassResponseInfo, CheckedTemporalClassMediationInfo,
    CheckedTemporalMediationInfo, CheckedTemporalMediationTargetInfo, CheckedTransportTrialInfo,
    CheckedUnknownTieredAverageInfo, ComputeBudget, ExactFactorRequirement, ExactPreparedState,
    ExactStudyIdentities, ExactStudyInspection, ExactStudyResult, IntoSharedPlan, InterferenceSpec,
    LatencyMode,
    LearnedTrialResult, LearnedTrialState, LicensedNeighbor, NextAction, OperationKind,
    OperationReadiness, OperationReport, PremiseChange, PreparedBatch, PreparedStudy,
    PreparedZTransport, RdConfig, RefuteSuite, SemanticApplicability, SharedBatchDesign,
    SharedCovariateDesign, StageEvent, StageResultSink, StatisticalBindingView,
    StatisticalContrast, StatisticalPreparedState, StatisticalStudyInspection,
    StatisticalStudyResult, Study, StudyBuilder, TransportTrialSpec, ZTransportResult,
    ZTransportSensitivityArtifactWire, consume_z_transport_artifact,
};
pub use super::{
    BetweenStudy, BorrowingRequest, ConsumedEvidenceLedger, DependenceRangeRequest,
    EvidenceLedgerRefusal, EvidenceLedgerSession, LedgerExport, LedgerRefresh, ObservationAccount,
    PoolingDeclaration, PoolingRequest, RevisionReport, SourceChange, SourcePairRelation,
    SourceRevision, StudyEstimate, StudySummary, SynthesisOutcome, SynthesisValue,
    consume_evidence_ledger_artifact,
};
pub use super::{
    BridgeInspection, ConsumedScientificBridge, ConsumedScientificBridgeDistribution,
    DistributionBridgeInspection, PreparedScientificBridge, PreparedScientificBridgeDistribution,
    ScientificBridgeDistributionResult, ScientificBridgeExport, ScientificBridgeRefusal,
    ScientificBridgeResult, consume_scientific_bridge_artifact,
    consume_scientific_bridge_distribution_artifact, scientific_bridge_artifact_shape,
};
pub use super::{
    CategoricalTarget, ConsumedMeasurement, MeasurementAnalysisBinding, MeasurementDetail,
    MeasurementEstimate, MeasurementInput, MeasurementModelRefusal, MeasurementRefresh,
    PreparedMeasurementModel, consume_measurement_model_artifact,
};
pub use super::{
    ConstituentRole, DeclaredExclusion, DerivedTreatmentDeclaration, DerivedTreatmentPlan,
    ExclusionRule, InterventionMeaning, SourceColumn, TemporalPosition, Transformation,
    check_derived_treatment, estimate_derived_joint_cells,
};
pub use super::{
    ConsumedComposition, DependenceInput, MechanismCompositionExport, MechanismCompositionRefusal,
    MechanismCompositionRequest, MechanismInput, MechanismRefit, MechanismSource,
    PreparedMechanismComposition, consume_mechanism_composition_artifact,
};
pub use super::{ContinuousErrorInput, ContinuousMeasurementEstimate};
pub use super::{
    CounterfactualTransportConsumption, CounterfactualTransportObstructionReport,
    CounterfactualTransportOptions, CounterfactualTransportPreparation,
    CounterfactualTransportResult, PreparedCounterfactualTransport,
    consume_counterfactual_transport_artifact, prepare_counterfactual_transport,
};
pub use super::{CpdagEvaluation, EmpiricalScenarioInput, EvidenceSearch, SequenceEndpoints};
pub use super::{
    IntervalEstimand, TemporalInitialState, TemporalWindow, consume_period_refresh_artifact,
    consume_temporal_initial_state_artifact, consume_temporal_refresh_artifact,
    temporal_dependent_interval,
};
pub use super::{
    JOINT_SENSITIVITY_ARTIFACT_VERSION, JointFactorWire, JointOutcomeWire, JointPerturbationWire,
    JointProvenanceWire, JointReceiptWire, JointSamplingWire, JointSensitivityArtifactWire,
    JointSensitivityBodyWire, JointSensitivityConsumeLimits, JointTippingWire,
};
pub use super::{
    LearnedContinuousResult, PreparedLearnedContinuous, consume_learned_continuous_artifact,
};
pub use super::{
    MissingInference, MissingModality, MissingRouteCell, MissingRouteSupport,
    missing_support_table, route_missing_support,
};
pub use super::{MixedSourceResult, PreparedMixedSource, consume_mixed_source_artifact};
pub use super::{MzTransportResult, PreparedMzTransport, consume_mz_transport_artifact};
pub use super::{
    ObservationRecoveryResult, PreparedObservationRecovery, RecoveryEvaluation,
    consume_observation_recovery_artifact,
};
pub use super::{
    PredictiveAssessmentRefusal, PreparedPredictiveAssessment,
    consume_predictive_assessment_artifact, fit_declaration_from_ledger,
    observation_keys_from_ledger, posterior_predictive_predictor,
};
pub use super::{
    PreparedCpdagCompletionScenarios, consume_cpdag_scenarios_artifact,
    consume_scenario_covariance_artifact, export_scenario_covariance,
};
pub use super::{PreparedSmoothedDose, SmoothedDoseResult, consume_smoothed_dose_artifact};
pub use super::{PreparedTemporalTransport, consume_temporal_transport_artifact};
pub use super::{PreparedTransportScenarios, consume_transport_scenarios_artifact};
pub use super::{ScreenSplitReceipt, ScreenUnits};
pub use super::{
    TierDesign, TierDiagnostics, TierDiagnosticsError, TierEvalue, TierPointEvalue, TierScenario,
    tier_diagnostics,
};

pub use super::fallback::{
    AnalysisWithFallback, AssociationalFallback, EvidenceNeeds, FallbackOutcome, FallbackPart,
    FallbackRequest, FallbackRoles, FallbackUnavailable, RankedPlan,
};
pub use super::{JointResponseResult, PreparedJointResponse};
