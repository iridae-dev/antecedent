//! Unified `Study` facade.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines, clippy::doc_markdown, clippy::too_many_arguments)]

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
pub mod compact_export;
pub mod composition;
pub mod conditional_study_ranking;
mod contract;
mod contract_identity;
mod cost;
mod cpdag_scenarios;
pub mod decision_claims;
mod decision_contract_facade;
mod derived_treatment;
pub mod design_ranking;
pub mod dose_grid_functional;
pub mod effect_constancy;
pub mod effect_constancy_consumers;
mod exact;
mod execute;
mod inverse_outcome;
pub mod inverse_query;
mod joint_sensitivity;
mod joint_sensitivity_uncertainty;
pub mod latent_class_effects;
mod learned_continuous;
pub mod learned_joint_closed;
mod learned_trial;
pub mod mechanism_discrepancy;
mod mixed_source;
pub mod msm_sensitivity;
mod mz_transport;
pub mod native_claims;
pub mod nonlinear_mediation;
mod preflight;
mod rank_drop_estimate;
pub mod recalc_adjusted;
pub mod recalc_cell;
pub mod recalc_design;
pub mod recalc_dr;
pub mod recalc_receipt;
pub mod recalc_temporal;
pub mod recalc_temporal_measured;
mod recovery;
pub mod recovery_chain;
pub mod repair;
pub mod sensitivity_decision;
mod smoothed_dose;
mod statistical;
pub mod temporal_counterfactual;
mod temporal_extensions;
#[doc(hidden)]
pub use temporal_extensions::temporal_dependent_interval_candidate;
mod temporal_transport;
mod tier_diagnostics;
mod transport_grid;
mod transport_scenarios;
pub mod transported_counterfactual;
pub mod vector_treatment;
mod z_transport;
mod z_transport_sensitivity_artifact;
pub use admg_conditional_transport::{
    AdmgConditionalResult, PreparedAdmgConditionalTransport,
    consume_admg_conditional_obstruction_artifact, consume_admg_conditional_transport_artifact,
    export_admg_conditional_obstruction,
};
pub(crate) use checked_bayesian_class_conditional::CheckedBayesianClassConditional;
pub(crate) use checked_bayesian_graph_posterior::CheckedBayesianGraphPosteriorAte;
pub(crate) use checked_bayesian_temporal_class_effect::CheckedBayesianTemporalClassEffectOperation;
pub(crate) use checked_bayesian_temporal_effect::CheckedBayesianTemporalEffectOperation;
pub(crate) use checked_class_graph_posterior_effect::CheckedClassGraphPosteriorEffect;
pub(crate) use checked_conditional::{CheckedConditionalOperation, ConditionalProcedure};
pub(crate) use checked_graph_posterior::{
    CheckedAdmgGraphPosteriorResponse, CheckedGraphPosteriorEffect, CheckedStaticClassEffect,
    StaticClassGraph, StaticClassIdentification,
};
pub(crate) use checked_temporal_class_effect::CheckedTemporalClassEffectOperation;
pub(crate) use checked_temporal_class_response::CheckedTemporalClassResponseOperation;
pub(crate) use checked_temporal_effect::CheckedTemporalEffectOperation;
pub(crate) use checked_temporal_graph_posterior_effect::{
    CheckedTemporalGraphPosteriorEffect, CheckedTemporalGraphPosteriorProof,
};
pub(crate) use checked_temporal_graph_posterior_response::{
    CheckedTemporalGraphPosteriorResponse, TemporalPosteriorResponseProof,
};
pub use cost::{
    BatchCostEstimate, CostEstimate, InferenceDefault, JointCellCostEstimate, RetargetCostEstimate,
    estimate_joint_cell_cost,
};
pub use cpdag_scenarios::{
    PreparedCpdagCompletionScenarios, consume_cpdag_scenarios_artifact,
    consume_scenario_covariance_artifact, export_scenario_covariance,
};
pub use derived_treatment::{
    ConstituentRole, DeclaredExclusion, DerivedTreatmentDeclaration, DerivedTreatmentPlan,
    ExclusionRule, InterventionMeaning, SourceColumn, TemporalPosition, Transformation,
    check_derived_treatment, estimate_derived_joint_cells,
};
pub use exact::{
    ExactFactorRequirement, ExactPreparedState, ExactStudyIdentities, ExactStudyInspection,
    ExactStudyResult,
};
pub use inverse_outcome::{
    ActionConstraint, ActionOutcome, ActionSpec, ActionStatus, EnumeratedStatus, ForwardEvaluation,
    ForwardInterval, INVERSE_SCOPE_NOTE, IntervalMeta, IntervalScope, InverseOutcomeError,
    InverseOutcomeReport, InverseQuery, MAX_INVERSE_ACTIONS, MAX_INVERSE_FORWARD_POINTS,
    SupportBasis, TargetDirection, classify_inverse_outcome,
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
pub use mixed_source::{MixedSourceResult, PreparedMixedSource, consume_mixed_source_artifact};
pub use mz_transport::{MzTransportResult, PreparedMzTransport, consume_mz_transport_artifact};
pub use preflight::{
    ArmCount, ArmSpec, ArmWeightEss, BatchPreflightReport, ColumnMissingness, ColumnPriority,
    ColumnWeight, DependentColumn, DroppedColumn, DuplicateGroup, FindingSeverity,
    FittedPropensity, NuisanceFitDiagnostics, PreflightFinding, PreflightInput, PreflightReport,
    PropensityOutcome, RankDropPlan, RankDropPolicy, RankReport, ScoreQuantile, SpanCheck,
    fit_diagnostics_design, plan_rank_drop, preflight_design,
};
pub use rank_drop_estimate::{RankDropEstimate, estimate_with_rank_drop};
pub use recovery::{
    ObservationRecoveryResult, PreparedObservationRecovery, consume_observation_recovery_artifact,
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
pub use temporal_transport::{PreparedTemporalTransport, consume_temporal_transport_artifact};
pub use tier_diagnostics::{
    TierDesign, TierDiagnostics, TierDiagnosticsError, TierEvalue, TierPointEvalue, TierScenario,
    tier_diagnostics,
};
pub use transport_grid::{
    TransportGridData, TransportGridFailure, TransportGridPoint, TransportGridQuery,
    TransportGridResult, TransportGridState,
};
pub use transport_scenarios::{PreparedTransportScenarios, consume_transport_scenarios_artifact};
pub use z_transport::{PreparedZTransport, ZTransportResult, consume_z_transport_artifact};
pub use z_transport_sensitivity_artifact::ZTransportSensitivityArtifactWire;
mod graph_posterior_target;
mod helpers;
mod identification_cache;
mod latency;
mod prepared;
mod route_guards;
mod stage;
mod transport_common;
pub(crate) use graph_posterior_target::GraphPosteriorEffectTarget;

pub use antecedent_core::{
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
    BoundDecisionContract, ClaimProbability, ClaimProfile, DecisionClaimKind,
    DecisionDeclaredExclusion, DecisionFacadeRefusal, DecisionSupportRule, DecisionUncertaintyKind,
    ExternalCallbackReceipt, ExternalTrustLimit, IdentifiedActionRange, IdentifiedSetDecision,
    IdentifiedUtility, IdentifiedVerdict, InputSupport, ReplayReceipt, RobustDecisionResult,
    RobustResultArtifact, RobustVerdict, RobustnessError, SuppliedClaim, SupportShortfall,
    UncertaintyRequirement, adapt_finite_scenarios, adapt_graph_dependent_claims,
    adapt_point_claim, adapt_weighted_graph_atoms, assess_robustness, evaluate_adapted,
    evaluate_identified_sets, evaluate_robust, utility_interval,
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
