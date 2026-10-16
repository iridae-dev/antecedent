//! Unified `Study` facade.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Reference getters on checked operations. The expansion is the function the
/// invocation writes: the same visibility, attributes, signature, and
/// `&self.field` body.
macro_rules! checked_ref {
    ($(
        $(#[$meta:meta])*
        $vis:vis fn $name:ident(&self) -> &$ty:ty { &self.$field:ident }
    )+) => {$(
        $(#[$meta])*
        $vis fn $name(&self) -> &$ty {
            &self.$field
        }
    )+};
}
pub(crate) use checked_ref;

pub mod api;

pub mod adaptive_inference;
pub mod adaptive_participant_procedure;
pub mod adaptive_procedure_declaration;
pub mod adaptive_procedure_report;
mod admg_conditional_transport;
mod batch;
mod batch_export;
mod batch_retarget;
mod builder;
mod candidate_screen_units;
pub mod categorical_treatment;
mod checked_bayesian_class_conditional;
mod checked_bayesian_graph_posterior;
mod checked_bayesian_temporal_class_effect;
mod checked_bayesian_temporal_effect;
mod checked_class_graph_posterior_effect;
mod checked_conditional;
mod checked_graph_posterior;
mod checked_propensity;
mod checked_temporal_class_effect;
mod checked_temporal_class_response;
mod checked_temporal_effect;
mod checked_temporal_graph_posterior_effect;
mod checked_temporal_graph_posterior_response;
mod checked_temporal_response;
pub mod closed_pilots;
pub mod cohort_decision;
pub mod compact_export;
pub mod composition;
pub mod composition_verification;
pub mod conditional_study_ranking;
mod contract;
mod contract_identity;
mod cost;
pub mod count_fitted_observation;
mod counterfactual_transport;
mod cpdag_scenarios;
pub mod cross_world_execution;
pub mod cross_world_interval;
pub mod decision_claims;
mod decision_contract_facade;
pub mod decision_reliability;
mod derived_treatment;
mod design_exports;
pub mod design_ranking;
pub mod discovered_transport;
pub mod discretized_composition;
pub mod dose_grid_functional;
pub mod effect_constancy;
pub mod effect_constancy_consumers;
pub mod evidence_ledger;
mod exact;
mod execute;
pub mod interface_contract;
mod inverse_outcome;
pub mod inverse_query;
pub mod inverse_sources;
pub mod joint_law_composed_basis;
pub mod joint_law_propagation;
mod joint_sensitivity;
mod joint_sensitivity_uncertainty;
pub mod latent_class_effects;
mod learned_continuous;
pub mod learned_joint_closed;
mod learned_trial;
pub mod measurement_model;
pub mod mechanism_composition;
pub mod mechanism_discrepancy;
pub(crate) mod mechanism_exports;
mod mixed_source;
pub mod msm_sensitivity;
mod mz_transport;
pub mod native_claims;
pub mod nonlinear_mediation;
pub mod observation_evidence;
pub mod observation_posterior;
pub mod observation_query;
pub mod per_population_discovery;
pub mod predictive_assessment;
mod preflight;
pub mod provider_callbacks;
pub(crate) mod query_license;
mod rank_drop_estimate;
pub mod recalc_adjusted;
pub mod recalc_cell;
mod recalc_common;
pub mod recalc_design;
pub mod recalc_dr;
pub mod recalc_receipt;
pub mod recalc_temporal;
pub mod recalc_temporal_measured;
mod recovery;
pub mod recovery_chain;
pub mod repair;
pub mod sample_split_analysis;
pub mod sample_split_effect;
pub mod scientific_bridge;
pub mod selective_discovery;
pub mod sensitivity_decision;
pub mod sequential_evaluation;
mod smoothed_dose;
mod statistical;
pub mod temporal_counterfactual;
mod temporal_extensions;
#[doc(hidden)]
pub use temporal_extensions::temporal_dependent_interval_candidate;
mod temporal_transport;
mod tier_diagnostics;
mod transport_grid;
pub mod transport_handoff;
pub mod transport_revalidation;
mod transport_scenarios;
pub mod transport_sensitivity;
pub mod transported_cohort;
pub mod transported_constructors;
pub mod transported_counterfactual;
pub mod transported_measurement;
pub mod transported_mechanism;
pub mod vector_treatment;
mod z_transport;
mod z_transport_sensitivity_artifact;
pub use adaptive_inference::{
    AdaptiveInferenceOutcome, AdaptiveInferenceRefusal, AdaptiveInferenceSession,
    ConsumedAdaptiveResult, ConsumedMonitoringContract, consume_adaptive_inference_result_artifact,
    consume_adaptive_monitoring_contract_artifact,
};
pub use adaptive_participant_procedure::{
    ParticipantHistoryReport, ParticipantProcedure, ParticipantSimulationReport,
};
pub use adaptive_procedure_declaration::{AdaptiveProcedure, ProcedureElements, Retention};
pub use adaptive_procedure_report::{
    ProcedureLawReport, ProcedureReport, ProcedureReportBody, ProcedureReportRequest,
    ProcedureSimulationReport,
};
pub use admg_conditional_transport::{
    AdmgConditionalResult, PreparedAdmgConditionalTransport,
    consume_admg_conditional_obstruction_artifact, consume_admg_conditional_transport_artifact,
    export_admg_conditional_obstruction,
};
pub(crate) use checked_bayesian_class_conditional::CheckedBayesianClassConditional;
pub(crate) use checked_bayesian_graph_posterior::{
    CheckedBayesianGraphPosteriorAte, CheckedBayesianGraphPosteriorAtePremises,
};
pub(crate) use checked_bayesian_temporal_class_effect::CheckedBayesianTemporalClassEffectOperation;
pub(crate) use checked_bayesian_temporal_effect::CheckedBayesianTemporalEffectOperation;
pub(crate) use checked_class_graph_posterior_effect::{
    CheckedClassGraphPosteriorEffect, CheckedClassGraphPosteriorEffectPremises,
};
pub(crate) use checked_conditional::{CheckedConditionalOperation, ConditionalProcedure};
pub(crate) use checked_graph_posterior::{
    CheckedAdmgGraphPosteriorResponse, CheckedGraphPosteriorEffect,
    CheckedGraphPosteriorEffectPremises, CheckedStaticClassEffect,
    CheckedStaticClassEffectPremises, StaticClassGraph, StaticClassIdentification,
};
pub(crate) use checked_temporal_class_effect::{
    CheckedTemporalClassEffectOperation, CheckedTemporalClassEffectOperationPremises,
};
pub(crate) use checked_temporal_class_response::{
    CheckedTemporalClassResponseOperation, CheckedTemporalClassResponseOperationPremises,
};
pub(crate) use checked_temporal_effect::{
    CheckedTemporalEffectOperation, CheckedTemporalEffectOperationPremises,
};
pub(crate) use checked_temporal_graph_posterior_effect::{
    CheckedTemporalGraphPosteriorEffect, CheckedTemporalGraphPosteriorEffectPremises,
    CheckedTemporalGraphPosteriorProof,
};
pub(crate) use checked_temporal_graph_posterior_response::{
    CheckedTemporalGraphPosteriorResponse, CheckedTemporalGraphPosteriorResponsePremises,
    TemporalPosteriorResponseProof,
};
pub use composition_verification::{
    CompositionVerification, CompositionVerificationRefusal, ConsumedCompositionVerification,
    consume_composition_verification_artifact,
};
pub use cost::{
    BatchCostEstimate, CostEstimate, InferenceDefault, JointCellCostEstimate, RetargetCostEstimate,
    estimate_joint_cell_cost,
};
pub use counterfactual_transport::{
    CounterfactualTransportConsumption, CounterfactualTransportObstructionReport,
    CounterfactualTransportOptions, CounterfactualTransportPreparation,
    CounterfactualTransportResult, PreparedCounterfactualTransport,
    consume_counterfactual_transport_artifact, prepare_counterfactual_transport,
};
pub use cpdag_scenarios::{
    CpdagEvaluation, PreparedCpdagCompletionScenarios, consume_cpdag_scenarios_artifact,
    consume_scenario_covariance_artifact, export_scenario_covariance,
};
pub use decision_reliability::{
    DecisionReliabilityInspection, DecisionReliabilityRefusal, PreparedDecisionReliability,
    consume_decision_reliability_artifact,
};
pub use derived_treatment::{
    ConstituentRole, DeclaredExclusion, DerivedTreatmentDeclaration, DerivedTreatmentPlan,
    ExclusionRule, InterventionMeaning, SourceColumn, TemporalPosition, Transformation,
    check_derived_treatment, estimate_derived_joint_cells,
};
pub use evidence_ledger::{
    BetweenStudy, BorrowingRequest, ConsumedEvidenceLedger, DependenceRangeRequest,
    EvidenceLedgerRefusal, EvidenceLedgerSession, LedgerExport, LedgerRefresh, ObservationAccount,
    PooledInterval, PoolingDeclaration, PoolingRequest, RevisionReport, SourceChange,
    SourcePairRelation, SourceRevision, StudyEstimate, StudySummary, SynthesisOutcome,
    SynthesisValue, consume_evidence_ledger_artifact,
};
pub use exact::{
    ExactFactorRequirement, ExactPreparedState, ExactStudyIdentities, ExactStudyInspection,
    ExactStudyResult,
};
pub use interface_contract::{
    ConsumedInterfaceContract, InterfaceContractExport, InterfaceContractRefusal,
    check_interface_composition, check_interface_composition_with_ledger,
    check_interface_connection, consume_interface_contract_artifact, export_interface_contract,
    inspect_interface_contract_artifact,
};
pub use inverse_outcome::{
    ActionConstraint, ActionOutcome, ActionSpec, ActionStatus, EnumeratedStatus, ForwardEvaluation,
    ForwardInterval, INVERSE_SCOPE_NOTE, IntervalMeta, IntervalScope, InverseOutcomeError,
    InverseOutcomeReport, InverseQuery, MAX_INVERSE_ACTIONS, MAX_INVERSE_FORWARD_POINTS,
    SupportBasis, TargetDirection, classify_inverse_outcome,
};
pub use joint_law_propagation::{
    ComposedInterval, CorrelationRangeSummary, ExceedanceQuery, JointLawPropagationRefusal,
    LinearDependence, MixtureComponent, MixtureDraws, MomentInput, PropagationClaim,
    PropagationDraws, alignment_identity, claim_from_artifact,
    consume_joint_law_propagation_artifact, input_alignment, propagate_aligned_functional,
    propagate_aligned_linear, propagate_aligned_linear_spread, propagate_composed_exceedance,
    propagate_correlation_range, propagate_dependence_envelope, propagate_linear_moments,
    propagate_linear_moments_spread, propagate_mixture, propagate_mixture_exceedance,
    propagate_spread, propagate_transported_spread, propagate_transported_spread_aligned,
};
pub use joint_sensitivity::{
    JOINT_SENSITIVITY_ARTIFACT_VERSION, JointFactorWire, JointOutcomeWire, JointPerturbationWire,
    JointProvenanceWire, JointReceiptWire, JointSensitivityArtifactWire, JointSensitivityBodyWire,
    JointSensitivityConsumeLimits, JointTippingWire,
};
pub use joint_sensitivity_uncertainty::JointSamplingWire;
pub use learned_continuous::{
    LearnedContinuousResult, PreparedLearnedContinuous, consume_learned_continuous_artifact,
};
pub use learned_trial::{LearnedTrialResult, LearnedTrialState};
pub use measurement_model::{
    CategoricalTarget, ConsumedMeasurement, MeasurementAnalysisBinding, MeasurementDetail,
    MeasurementEstimate, MeasurementInput, MeasurementModelRefusal, MeasurementRefresh,
    PreparedMeasurementModel, consume_measurement_model_artifact,
};
pub use mechanism_composition::{
    ConsumedComposition, DependenceInput, MechanismCompositionExport, MechanismCompositionRefusal,
    MechanismCompositionRequest, MechanismInput, MechanismRefit, MechanismSource,
    PreparedMechanismComposition, consume_mechanism_composition_artifact,
};
pub use mixed_source::{MixedSourceResult, PreparedMixedSource, consume_mixed_source_artifact};
pub use mz_transport::{MzTransportResult, PreparedMzTransport, consume_mz_transport_artifact};
pub use predictive_assessment::{
    PredictiveAssessmentRefusal, PreparedPredictiveAssessment,
    consume_predictive_assessment_artifact, fit_declaration_from_ledger,
    observation_keys_from_ledger, posterior_predictive_predictor,
};
pub use preflight::{
    ArmCount, ArmSpec, ArmWeightEss, BatchPreflightReport, ColumnMissingness, ColumnPriority,
    ColumnWeight, DependentColumn, DroppedColumn, DuplicateGroup, FindingSeverity,
    FittedPropensity, NuisanceFitDiagnostics, PreflightFinding, PreflightInput, PreflightReport,
    PropensityOutcome, RankDropPlan, RankDropPolicy, RankReport, ScoreQuantile, SpanCheck,
    fit_diagnostics_design, plan_rank_drop, preflight_design,
};
pub use provider_callbacks::{
    CallbackConnectionError, CallbackExecution, CallbackExecutionError, CallbackExecutionPolicy,
    CallbackRegistrationError, CallbackResume, CallbackSession, FnCallback, ProviderCallback,
    check_callback_connection, check_claim_to_callback, consume_callback_result,
    export_callback_result, forecast_law_from_sample, inspect_callback_result,
    predictive_law_source, refuse_identification_licence, refuse_interval, refuse_joint_law,
    refuse_native_trust, refuse_posterior_guarantee, register_callback_source,
    supersede_callback_source, verification_holds,
};
pub use rank_drop_estimate::{RankDropEstimate, estimate_with_rank_drop};
pub use recovery::{
    ObservationRecoveryResult, PreparedObservationRecovery, RecoveryEvaluation,
    consume_observation_recovery_artifact,
};
pub use scientific_bridge::{
    BridgeInspection, ConsumedScientificBridge, PreparedScientificBridge, ScientificBridgeExport,
    ScientificBridgeRefusal, ScientificBridgeResult, consume_scientific_bridge_artifact,
};
pub use smoothed_dose::{PreparedSmoothedDose, SmoothedDoseResult, consume_smoothed_dose_artifact};
pub use statistical::{
    StatisticalBindingView, StatisticalContrast, StatisticalPreparedState,
    StatisticalStudyInspection, StatisticalStudyResult,
};
pub use temporal_extensions::{
    IntervalEstimand, TemporalInitialState, TemporalWindow,
    consume_temporal_initial_state_artifact, consume_temporal_refresh_artifact,
    temporal_dependent_interval,
};
pub use temporal_transport::{
    PreparedTemporalTransport, SequenceEndpoints, consume_temporal_transport_artifact,
};
pub use tier_diagnostics::{
    TierDesign, TierDiagnostics, TierDiagnosticsError, TierEvalue, TierPointEvalue, TierScenario,
    tier_diagnostics,
};
pub use transport_common::EvidenceSearch;
pub use transport_grid::{
    TransportGridData, TransportGridFailure, TransportGridPoint, TransportGridQuery,
    TransportGridResult, TransportGridState,
};
pub use transport_scenarios::{
    EmpiricalScenarioInput, PreparedTransportScenarios, consume_transport_scenarios_artifact,
};
pub use z_transport::{PreparedZTransport, ZTransportResult, consume_z_transport_artifact};
pub use z_transport_sensitivity_artifact::ZTransportSensitivityArtifactWire;
mod graph_posterior_target;
mod helpers;
mod identification_cache;
mod latency;
mod posterior_binding;
mod prepared;
pub(crate) mod proof_support;
mod route_guards;
mod stage;
mod transport_common;
pub(crate) use graph_posterior_target::GraphPosteriorEffectTarget;

pub use antecedent_contract::{
    BlockedOperation, LicensedNeighbor, NextAction, OperationKind, OperationReadiness,
    OperationReport, PremiseChange, SemanticApplicability,
};
pub use batch::{
    BatchQuery, BatchStudy, CandidateProcedure, CandidateScreen, CandidateSelection,
    CellFamilyContrast, PreparedBatch, SharedBatchDesign, SharedCovariateDesign,
};
pub use batch_export::{TidyKind, TidyRow};
pub use batch_retarget::{
    BATCH_RETARGET_SCOPE_NOTE, BandMember, BatchRetargetError, BatchRetargetReport,
    BatchRetargetRequest, BatchScores, ClaimPoint, ClaimReport, ContrastPoint, ContrastReport,
    FamilyCovariance, MAX_T_MAX_DRAWS, MAX_T_MIN_DRAWS, MemberFailure, RetargetClaim,
    RetargetContrast, ScoreSource, SimultaneousBand, UncertaintyKind,
};
// Max-t evaluator entries kept for the calibration wiring (the published route is
// `BatchRetargetReport::simultaneous_interval`).
#[doc(hidden)]
pub use batch_retarget::{max_t_critical_value, simultaneous_band_unpublished};
pub use builder::{InterferenceSpec, RdConfig, RefuteSuite, StudyBuilder, TransportTrialSpec};
pub use candidate_screen_units::{ScreenSplitReceipt, ScreenUnits};
pub use contract::CausalContract;
pub use decision_contract_facade::{
    ActionRobustness, AdaptedClaims, AdaptedDecision, AdapterError, AdmissibilityError,
    AdmissibilityRules, AdmissibleContractArtifact, AdmissibleDecisionContract, AtomSupport,
    BoundDecisionContract, BoundTransportedDecision, ClaimProbability, ClaimProfile,
    DecisionClaimKind, DecisionDeclaredExclusion, DecisionFacadeRefusal, DecisionSupportRule,
    DecisionUncertaintyKind, ExternalCallbackReceipt, ExternalTrustLimit, IdentifiedActionRange,
    IdentifiedSetDecision, IdentifiedUtility, IdentifiedVerdict, InputSupport, ReplayReceipt,
    RobustDecisionResult, RobustResultArtifact, RobustVerdict, RobustnessError, SuppliedClaim,
    SupportPolicy, SupportShortfall, TransportDeclaration, TransportReceipt, TransportedDecision,
    TransportedDecisionInspection, TransportedDemand, TransportedRefusal,
    TransportedRepresentation, TransportedResponse, TransportedRoute, TransportedValue,
    UncertaintyRequirement, adapt_finite_scenarios, adapt_graph_dependent_claims,
    adapt_point_claim, adapt_weighted_graph_atoms, assess_robustness, contract_needs_law,
    evaluate_adapted, evaluate_identified_sets, evaluate_robust, evaluate_transported_decision,
    utility_interval,
};
pub use execute::DagResponseOrigin;
pub use execute::Study;
pub use latency::{
    ComputeBudget, INTERACTIVE_BOOTSTRAP, INTERACTIVE_MAX_ENVELOPE_GRAPHS, INTERACTIVE_N_DRAWS,
    LatencyMode, REPORT_BOOTSTRAP, REPORT_N_DRAWS, ResolvedLatencyBudget, STANDARD_BOOTSTRAP,
    STANDARD_N_DRAWS, refuse_non_report_hmc,
};
pub use prepared::{
    CachedTemporalIdentification, CheckedAdmgGraphPosteriorResponseInfo, CheckedAttributionInfo,
    CheckedBayesianBasisAteInfo, CheckedBayesianClassConditionalInfo,
    CheckedBayesianGraphPosteriorAteInfo, CheckedBayesianRobustAteInfo,
    CheckedBayesianTemporalDagEffectInfo, CheckedCellAipwResponseInfo,
    CheckedClassGraphPosteriorEffectInfo, CheckedConditionalEffectInfo,
    CheckedGraphPosteriorEffectInfo, CheckedGraphPosteriorResponseInfo, CheckedInterferenceInfo,
    CheckedStaticClassEffectInfo, CheckedStaticClassResponseInfo, CheckedStaticMediationInfo,
    CheckedTemporalClassEffectInfo, CheckedTemporalClassMediationInfo,
    CheckedTemporalDagEffectInfo, CheckedTemporalDagResponseInfo, CheckedTemporalMediationInfo,
    CheckedUnknownTieredAverageInfo, PreparedStudy,
};
pub use prepared::{
    CheckedBayesianTemporalClassEffectInfo, CheckedTemporalGraphPosteriorEffectInfo,
};
pub use prepared::{CheckedTemporalClassResponseInfo, CheckedTemporalGraphPosteriorResponseInfo};
pub use stage::{StageEvent, StageResultSink};

pub(crate) use checked_temporal_response::{
    CheckedTemporalResponseOperation, TemporalResponseHorizonEvidence, temporal_response_is_direct,
};
pub use execute::{CheckedBayesianSpecialistInfo, CheckedTransportTrialInfo};
pub(crate) use execute::{parametric_scm_identification, response_witness_ate};

pub mod proposal_arrival;
pub mod recalc_attempt;
pub mod recalc_bayesian;
pub mod recalc_external;
#[cfg(feature = "calibration-internal")]
pub mod recalc_joint_bayesian;
/// Shared finite static response and multi-source transport recalculation.
pub mod recalc_static;

/// Shared native response and two executing attested callback branches.
pub mod recalc_composite;

pub mod sensitivity_source;
/// Independently retained original source diagnostics and contributor mappings.
pub mod source_evidence;
pub mod source_projection;

#[doc(hidden)]
pub use temporal_extensions::validate_temporal_dependent_interval;
