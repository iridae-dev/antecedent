//! Unified `Study` facade execution (split by modality for SRP).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub(super) use std::sync::Arc;
pub(super) use std::time::Instant;

pub(super) use super::latency::{INTERACTIVE_MAX_ENVELOPE_GRAPHS, LatencyMode};
pub(super) use super::stage::elapsed_ns;
pub(super) use antecedent_contract::{DataClassification, ProvenanceGraph};
pub(super) use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalResponse, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, ExecutionContext, Intervention, ObservationSpec, PopulationRegistry,
    ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue,
    TemporalEffectQuery, VariableId,
};
pub(super) use antecedent_data::{
    DiscoveryEstimationSplit, PanelData, TableView, TabularData, TemporalIndexer, TimeIndex,
    TimeSeriesData,
};
pub(super) use antecedent_discovery::GraphPosterior;
pub(super) use antecedent_estimate::{
    AnalyticSeKind, BayesianGCompWorkspace, BayesianGComputationAte, BayesianTemporalGcomp,
    BootstrapAccounting, CausalPosterior, ConditionalLinearAdjustment, ContinuousResponseEstimator,
    EffectEstimate, EnvelopeOptions, EstimationWorkspace, FunctionalDistribution,
    FunctionalDistributionWorkspace, FunctionalEffect, GraphEffectDraws, LinearAdjustmentAte,
    ObservationMechanismEstimator, OverlapPolicy, PreparedBayesianProblem, RdWorkspace,
    SharpRegressionDiscontinuity, TemporalLinearAdjustment, TemporalMediationEstimate,
    TemporalMediationEstimator, TemporalResponseEstimator, aggregate_effect_envelope,
    functional_cell_unevaluable, nonidentified_with_prior, support_from_functional_eval,
};
pub(super) use antecedent_expr::{
    CausalExprArena, DerivationMeta, DomainRef, EstimandMethod, EvalError, ExprNode,
    IdentifiedEstimand, OutcomeExprId,
};
pub(super) use antecedent_graph::{Admg, Dag, DenseNodeId, Pag, TemporalDag};
pub(super) use antecedent_identify::{
    DerivationTrace, IdentificationEnvelope, IdentificationPerformanceRecord, IdentificationResult,
    IdentificationStatus, SharpRdConfig, SharpRdIdentifier, TemporalBackdoorIdentifier,
    TemporalMediationIdentifier,
};
pub(super) use antecedent_prob::{
    GraphIdentFlag, InferenceDiagnostics, PriorSet, WeightedGraphSamples,
};
pub(super) use antecedent_validate::{
    BayesianSuiteContext, PosteriorPredictiveCheck, PredictiveCheckReport, PriorPredictiveCheck,
    QueryRefutationPlan, TemporalRefitContext, ValidationSuite, ValidatorId, stack_panel_tabular,
    with_conflict_summary, with_prior_sensitivity,
};

pub(super) use crate::accepted::{AcceptedGraph, GraphClass, GraphRef};
pub(super) use crate::callback_plan::mark_python_callback_plan;
pub(super) use crate::error::CausalError;
pub(super) use crate::gcm::{
    anomaly_attribution_query_with, attribute_distribution_change, attribute_unit_change, fit_gcm,
    map_mechanism_fit, mechanism_change_detection,
};
pub(super) use crate::inference::{
    BayesianConfig, InferenceMode, resolve_bayesian_prior_with_conflict,
};
pub(super) use crate::planner::{
    LogicalAnalysisPlan, PhysicalExecutionPlan, StaticAdmgPathSpecificCompileInput,
    StaticAteCompileInput, StaticCpdagPathSpecificCompileInput, StaticCpdagResponseCompileInput,
    StaticDistributionCompileInput, StaticPagAteCompileInput, StaticPagPathSpecificCompileInput,
    StaticPagResponseCompileInput, StaticPathSpecificCompileInput, StaticResponseCompileInput,
    TemporalClassEffectCompileInput,
    compile_logical_codetermined_joint, compile_logical_distribution,
    compile_logical_path_specific, compile_logical_path_specific_admg,
    compile_logical_path_specific_cpdag, compile_logical_path_specific_pag,
    compile_logical_static_ate, compile_logical_static_cpdag_ate,
    compile_logical_static_cpdag_response,
    compile_logical_static_pag_ate, compile_logical_static_pag_response,
    compile_logical_static_response, compile_logical_temporal_class_effect,
    compile_logical_temporal_effect, compile_logical_temporal_effect_classified,
    compile_logical_temporal_response, reject_dag_only_on_pag,
};
pub(super) use crate::result::StructuralAggregationPolicy;
pub(super) use crate::result::StudyResult;
pub(super) use crate::strategy_table::{
    DEFAULT_ADMG_ESTIMATOR_ID, DEFAULT_ADMG_IDENTIFIER_ID, DEFAULT_CONDITIONAL_ESTIMATOR_ID,
    DEFAULT_CONDITIONAL_IDENTIFIER_ID, DEFAULT_DISTRIBUTION_ESTIMATOR_ID,
    DEFAULT_DISTRIBUTION_IDENTIFIER_ID, DEFAULT_ESTIMATOR_ID, DEFAULT_IDENTIFIER_ID,
    DEFAULT_PAG_ESTIMATOR_ID, DEFAULT_PAG_IDENTIFIER_ID, DEFAULT_PATH_ESTIMATOR_ID,
    DEFAULT_PATH_IDENTIFIER_ID, DEFAULT_RESPONSE_IDENTIFIER_ID, EstimatorId, IdentifierId,
    StaticEstimateInput, StaticEstimateWorkspaces, estimate_provenance_step,
    estimate_static_effect, identify_admg, identify_admg_query,
    identify_cpdag_distribution, identify_cpdag_path_specific,
    identify_pag_distribution, identify_pag_path_specific,
    identify_provenance_step, identify_static,
    identify_static_query, identify_static_query_in_context, identify_static_query_with_rd,
    identify_temporal_cpdag_configured,
    identify_temporal_pag_configured, require_identified, select_claim, select_estimand,
    validate_static_pair,
};

pub(super) use super::builder::{DataInput, RdConfig, RefuteSuite};
pub(super) use super::helpers::{
    AssembleArgs, RefutationRun, assemble_result, credible_scalar_uncertainty,
    effect_from_posterior, evaluate_bayesian_prior_sensitivity, overlap_diagnostic,
    project_for_ate_estimate, projection_diagnostic, provenance_pair, push_conflict_diagnostics,
    validator_not_applicable_diagnostic, validator_not_applicable_diagnostics,
};

/// Build-time declarations no click rebinds.
///
/// A [`Study`] is cloned once per click and once per posterior atom; these
/// fields are identical in every such copy, so they are shared behind one
/// `Arc` and a clone costs a reference-count bump instead of a field-by-field
/// copy. Only build and prepare (never a click) write them, through
/// `Arc::make_mut` on the owning study.
// Hidden, but public: it is `Study`'s `Deref` target.
#[doc(hidden)]
#[derive(Clone, Debug)]
pub struct StudyConfig {
    /// Set at [`crate::StudyBuilder::build`] when the caller did not call
    /// `.refute(..)` explicitly and the default refute suite
    /// (`RefuteSuite::PlaceboAndRcc`) was silently downgraded to
    /// `RefuteSuite::None` because the requested cell was
    /// `NotApplicable`/`Refused` while the `RefuteSuite::None` cell for the
    /// same query was `Licensed`/`Allowlisted`. Holds the suite that was
    /// requested before the downgrade (always some suite other than
    /// `None`); `None` here means no downgrade happened. Surfaced to the
    /// caller as diagnostic `exec.refute.default_suite_unsupported`.
    pub(crate) refute_default_downgrade: Option<RefuteSuite>,
    /// `(tier_replicates, configured_replicates)` when a latency tier mapped a
    /// bootstrap budget but the configured estimator's own replicate count won.
    /// Surfaced as diagnostic `latency.bootstrap_not_applied`.
    pub(crate) latency_bootstrap_not_applied: Option<(u32, u32)>,
    /// Structured identity of [`Study::estimator_spec`], digested once at build
    /// so data-sized cluster / panel vectors are never rehashed per contract.
    pub(crate) estimator_spec_identity: Option<antecedent_io::EstimatorSpecWire>,
    pub(crate) response_options: Option<antecedent_estimate::ContinuousResponseOptions>,
    pub(crate) observation_options: antecedent_estimate::ObservationEstimatorOptions,
    pub(crate) draw_controls: Option<crate::PosteriorDrawControls>,
    pub(crate) observation_delayed_entry: Option<antecedent_core::VariableId>,
    pub(crate) rd: Option<RdConfig>,
    pub(crate) stage_sink: Option<Arc<dyn super::stage::StageResultSink>>,
    /// Refused: coarsened continuous coordinate is not a point CDE.
    ///
    /// `StudyBuilder::build` refuses a study that declares one, so a built
    /// study always carries `None` and no identity layer binds it.
    pub(crate) continuous_cell: Option<(antecedent_core::VariableId, Arc<[f64]>)>,
    /// Selection diagram frozen for a transport query.
    pub(crate) selection_diagram: Option<antecedent_graph::SelectionDiagram>,
    /// Trial columns frozen for a transport query.
    pub(crate) transport_trial: Option<super::builder::TransportTrialSpec>,
    /// Prepare-time transport formula + certificate.
    pub(crate) transport_identification_cache:
        Option<Arc<antecedent_identify::TransportIdentification>>,
    /// Executable search of a structure-discovery query.
    pub(crate) discovery: Option<Arc<super::structure_route::DiscoverySpec>>,
    /// Mechanism fit of an attribution or simulation query; `None` runs the
    /// standard registry.
    pub(crate) mechanisms: Option<Arc<super::structure_route::MechanismProgram>>,
    /// Provenance of a graph posterior accepted from a compiled structure search.
    pub(crate) posterior_provenance: Option<Arc<crate::accepted::StructureProvenance>>,
    /// The `AverageEffect` the caller asked for on series data under a `TemporalDag`,
    /// `TemporalCpdag` or `TemporalPag`, when the build compiled it as its lag-0 pulse contrast. The support
    /// cell is the `AverageEffect` cell; every result discloses the reading.
    pub(crate) lag_zero_average_effect: Option<antecedent_core::AverageEffectQuery>,
}

/// Prepared analysis (static or temporal).
///
/// Build-time declarations live in a shared [`StudyConfig`] reachable through
/// `Deref`. The fields below are the per-click overlay: everything a rebinding
/// (`click`, graph-posterior atom, refresh) may replace. Heavy members are
/// held behind an `Arc`, so cloning a `Study` never copies a graph, an
/// estimator specification, or a prepare-time identification cache.
#[derive(Clone, Debug)]
pub struct Study {
    pub(crate) config: Arc<StudyConfig>,
    pub(crate) data: DataInput,
    pub(crate) graph: Arc<AcceptedGraph>,
    /// Set instead of a single `graph` atom when [`crate::StudyBuilder::graph_posterior`]
    /// was used; `graph` then holds only a placeholder shape (variable count / modality
    /// only — never consulted for identification). Mutually exclusive with a "real" `graph`.
    pub(crate) graph_posterior: Option<GraphPosterior>,
    /// Caller-supplied mass over incomplete-temporal class members.
    pub(crate) class_prior: Option<crate::ClassPrior>,
    /// Optional cap on `TemporalCpdag` / `TemporalPag` completion search.
    pub(crate) max_completions: Option<usize>,
    /// Matrix structure-source axis recorded at [`crate::StudyBuilder::build`].
    pub(crate) structure_source: crate::support::StructureSource,
    /// Licensed vs allowlisted evidence status recorded at build. `None` when
    /// the query is not on the public matrix axis.
    pub(crate) support_status: Option<crate::support::CellStatus>,
    pub(crate) query: CausalQuery,
    pub(crate) refute: RefuteSuite,
    pub(crate) bootstrap_replicates: u32,
    /// Retain the joint replicates of a temporal response surface (see
    /// [`crate::StudyBuilder::retain_joint_horizons`]).
    pub(crate) retain_joint_horizons: bool,
    pub(crate) split: Option<DiscoveryEstimationSplit>,
    pub(crate) identifier: Option<IdentifierId>,
    pub(crate) estimator: Option<EstimatorId>,
    pub(crate) estimator_spec: Option<Arc<crate::estimator_spec::EstimatorSpec>>,
    pub(crate) inference: InferenceMode,
    pub(crate) overlap_policy: Option<OverlapPolicy>,
    pub(crate) population_registry: Option<PopulationRegistry>,
    /// Digests of the participant labels bound to the rows of the data.
    pub(crate) participants: Option<antecedent_io::ParticipantBindingWire>,
    /// Declared handling of missing values, when the caller declared one.
    pub(crate) missing: Option<antecedent_data::MissingDeclaration>,
    /// Declared sampling design of the rows, held by every estimator that bootstraps or splits.
    pub(crate) sampling_design: Option<antecedent_core::DeclaredSamplingDesign>,
    pub(crate) custom_validators: Vec<Arc<dyn antecedent_validate::CustomEffectValidator>>,
    pub(crate) latency_mode: Option<super::latency::LatencyMode>,
    /// Prepare-time identification for the static ATE path (including the
    /// general-ID result and functional estimand of a bidirected ADMG), set
    /// only by [`Study::prepare`]. Identification depends solely on
    /// (identifier, graph, query, rd) — all frozen at prepare — so reusing it
    /// per estimate click is exact; `None` (every builder-constructed study)
    /// identifies on each run.
    pub(crate) identification_cache: Option<Arc<super::prepared::CachedStaticIdentification>>,
    /// Prepare-time generalized-adjustment envelope for a supplied PAG.
    pub(crate) pag_identification_cache: Option<Arc<super::prepared::CachedPagIdentification>>,
    /// Prepare-time MEC envelope for a supplied CPDAG.
    pub(crate) cpdag_identification_cache: Option<Arc<super::prepared::CachedCpdagIdentification>>,
    /// Prepare-time temporal-backdoor identification + indexer for temporal response.
    pub(crate) temporal_identification_cache:
        Option<Arc<super::prepared::CachedTemporalIdentification>>,
    /// Prepare-time TemporalCpdag/Pag envelope.
    pub(crate) temporal_class_identification_cache:
        Option<Arc<super::prepared::CachedTemporalClassIdentification>>,
    /// Prepare-time per-atom identification and weights for a static graph posterior.
    pub(crate) graph_posterior_identification_cache:
        Option<Arc<super::prepared::CachedGraphPosteriorIdentification>>,
    /// Prepare-time per-atom identification, indexers, and weights for a DBN posterior.
    pub(crate) dbn_posterior_identification_cache:
        Option<Arc<super::prepared::CachedDbnPosteriorIdentification>>,
    /// Prepare-time TemporalCpdag/Pag graph-posterior class envelopes.
    pub(crate) temporal_class_posterior_identification_cache:
        Option<Arc<super::prepared::CachedTemporalClassPosteriorIdentification>>,
    /// Optional tier-rule background for O(p) closure certification.
    pub(crate) tiered: Option<antecedent_graph::TieredBackground>,
    /// Network + assignment of an interference query, rebound to each click's rows.
    pub(crate) interference: Option<super::builder::InterferenceSpec>,
    /// Shared fold assignment / covariate design when this study is part of a batch.
    pub(crate) shared_batch_design: Option<Arc<super::batch::SharedBatchDesign>>,
}

impl std::ops::Deref for Study {
    type Target = StudyConfig;

    fn deref(&self) -> &StudyConfig {
        &self.config
    }
}

mod admg_posterior;
mod admg_posterior_response;
mod attribution_path;
mod checked_attribution;
mod class_attribution;
mod temporal_attribution;
mod temporal_admg_mediation;
mod temporal_pag_mediation;
mod temporal_admg_posterior;
pub(crate) use attribution_path::{CheckedCounterfactualPlan, CounterfactualProcedure};
pub(crate) use checked_attribution::{
    CheckedAttributionOperation, attribute_bootstrap_supported, attribute_posterior_draws,
    attribute_posterior_supported, attribute_runs_checked, is_mechanism_change_query,
    mechanism_change_posterior_declared,
};
mod bayesian_basis_path;
mod checked_bayesian_basis_ate;
mod checked_bayesian_basis_cate;
mod checked_bayesian_robust_ate;
pub(crate) use checked_bayesian_basis_ate::CheckedBayesianBasisAteExecution;
pub(crate) use checked_bayesian_basis_cate::CheckedBayesianBasisCateExecution;
pub(crate) use checked_bayesian_robust_ate::CheckedBayesianRobustAteExecution;
mod checked_bayesian_dag_ate;
pub(crate) use checked_bayesian_dag_ate::{
    CheckedBayesianDagAteExecution, CheckedBayesianDagAteExecutionPremises,
};
mod bayesian_graph_posterior_atoms;
mod checked_bayesian_dag_conditional;
pub(crate) use bayesian_graph_posterior_atoms::{
    BayesianGraphPosteriorAtomFits, BayesianGraphPosteriorAtomInput,
    BayesianGraphPosteriorPriorAnchor,
};
mod checked_cell_aipw_response;
pub(crate) use checked_bayesian_dag_conditional::{
    CheckedBayesianConditionalOperation, CheckedBayesianConditionalOperationPremises,
};
mod bayesian_level_refutation;
mod bayesian_path;
mod bayesian_quantile_path;
mod bayesian_robust_ate_path;
mod bayesian_specialist_path;
mod checked_bayesian_static_mediation;
mod checked_continuous_dose;
mod checked_interference;
mod checked_local_polynomial_ratio;
mod checked_longitudinal_regime;
mod checked_panel_did;
mod checked_randomized;
mod class_response_refutation;
mod class_derivative_refutation;
mod front_door_derivative_refutation;
mod derivative_refutation;
mod gcm_bootstrap;
mod randomized_fit;
mod randomized_trial;
mod rollout_finish;
mod posterior_witness_state;
mod temporal_response_refutation;
mod transport_refutation;
use randomized_trial::{RandomizedTrialOperation, RandomizedTrialQuery};
mod checked_static_mediation;
mod checked_survival;
mod checked_synthetic_control;
#[path = "exec_checked_temporal_class_effect.rs"]
mod checked_temporal_class_effect;
#[path = "exec_checked_temporal_effect.rs"]
mod checked_temporal_effect;
mod checked_temporal_mediation;
pub(crate) use checked_bayesian_static_mediation::CheckedBayesianStaticMediationOperation;
pub use checked_cell_aipw_response::DagResponseOrigin;
pub(crate) use checked_cell_aipw_response::{
    CheckedCellAipwResponseOperation, CheckedCellAipwResponseOperationPremises,
};
pub(crate) use checked_continuous_dose::CheckedContinuousDoseOperation;
pub(crate) use checked_interference::CheckedInterferenceOperation;
pub(crate) use checked_local_polynomial_ratio::{
    CheckedLocalPolynomialRatioOperation, local_polynomial_ratio_identification,
};
pub(crate) use checked_longitudinal_regime::CheckedLongitudinalRegimeOperation;
mod checked_sequential_contrast;
pub(crate) use checked_sequential_contrast::CheckedSequentialContrastOperation;
mod checked_history_match;
mod checked_route;
pub(crate) use checked_history_match::CheckedHistoryMatchOperation;
pub(crate) use checked_route::{
    CheckedAssociationOperation, CheckedDiscoveryOperation, CheckedRouteOperation,
    finish_family_result,
};
mod checked_association;
mod checked_iv_bounds;
mod checked_temporal_design;
pub(crate) use checked_iv_bounds::CheckedIvBoundsOperation;
pub(crate) use checked_panel_did::CheckedPanelDidOperation;
pub(crate) use checked_panel_did::panel_did_graphless_key;
pub(crate) use checked_panel_did::panel_did_identification;
pub(crate) use checked_randomized::{
    CheckedPolicyValueOperation, CheckedRandomizedOperation, randomized_identification,
};
pub(crate) use checked_static_mediation::CheckedStaticMediationOperation;
pub(crate) use checked_survival::{CheckedSurvivalOperation, survival_identification};
pub(crate) use checked_synthetic_control::{
    CheckedSyntheticControlOperation, synthetic_control_identification,
};
pub(crate) use checked_temporal_design::CheckedTemporalDesignOperation;
pub(crate) use checked_temporal_mediation::CheckedTemporalMediationOperation;
mod atom_mix;
mod atom_route_posterior;
pub(crate) use atom_route_posterior::atom_route_response;
mod dbn_route_posterior;
pub(crate) use dbn_route_posterior::{dbn_atoms, dbn_route_query};
mod interference_posterior;
mod transport_class;
mod transport_posterior;
pub(crate) use transport_posterior::ipw_estimable;
mod class_counterfactual;
mod class_distribution;
mod class_envelope_se;
mod class_path_specific;
mod class_posterior;
mod cross_world_posterior;
mod class_posterior_response;
mod completion_fit_memo;
pub(crate) use checked_temporal_effect::CheckedTemporalEffectExecution;
mod compile;
mod compile_decision;
pub(crate) use compile_decision::require_carrier;
pub use compile_decision::{
    BAYESIAN_REALIZATION, DECISION_NEXT_ATTEMPTS, DISTRIBUTIONAL_CARRIER, DecisionCarrierError,
    DecisionCompileRequest, DecisionDemand, INTERVENTIONAL_LAW, MissingCarrier, PublishedCarriers,
    ROBUST_OVER_IDENTIFIED_SET, ScoredDecision, confidence_interval_as_posterior, score_decision,
};
mod compile_graph_posterior;
mod compile_planning;
pub(crate) use compile_planning::compile_planning;
pub use compile_planning::{PlannedExperiment, PlanningPlan, STANDALONE_PLANNING};
mod plan_id;
mod planning_simulation;
pub(super) use compile::compile_logical_admg_response;
mod route_input;
pub(crate) use route_input::{PathSpecificMixedSource, RouteInput, StaticMediationSource};
mod structural_route;
pub(crate) use structural_route::structure_search_identification;
mod dbn_mediation_frequentist;
mod mediation_streams;
use mediation_streams::{MediationRoute, mediation_block_stream, mediation_mechanism_seed};
mod checked_static_response;
mod dispatch;
mod identified_set_diagnostics;
mod interference_refutation;
mod mediation_mixed;
mod mediation_refutation;
mod multi_env_path;
mod pag_path;
mod panel_link_path;
mod panel_path;
mod response_path;
mod temporal_iv_path;
pub(crate) use checked_static_response::{
    CheckedDerivativeResponseOperation, CheckedStaticDagResponseOperation,
    CheckedStaticDagResponseOperationPremises,
};
pub(crate) use checked_temporal_class_effect::CheckedTemporalClassEffectExecution;
pub(crate) use temporal_path::checked_temporal_response::CheckedTemporalResponseExecution;
mod bayesian_quasi_common;
mod checked_bayesian_specialist;
#[path = "exec_checked_bayesian_temporal_class_effect.rs"]
mod checked_bayesian_temporal_class_effect;
mod checked_graph_posterior_effect;
mod checked_graph_posterior_response;
mod checked_static;
mod checked_static_class_response;
mod checked_temporal_class_mediation;
#[path = "exec_checked_temporal_class_response.rs"]
mod checked_temporal_class_response;
#[path = "exec_checked_temporal_graph_posterior_effect.rs"]
mod checked_temporal_graph_posterior_effect;
#[path = "exec_checked_temporal_graph_posterior_response.rs"]
mod checked_temporal_graph_posterior_response;
mod checked_transport_trial;
mod sequential_validation;
mod static_missing;
mod static_path;
#[cfg(test)]
mod study_clone_tests;
mod temporal_admg_path;
mod temporal_class_effect_core;
mod temporal_class_mediation_posterior;
mod temporal_class_point;
mod temporal_class_posterior;
mod temporal_class_posterior_response;
mod temporal_effect_core;
mod temporal_mediation_core;
mod temporal_path;
mod temporal_point_uncertainty;
mod temporal_posterior_response;
mod transport_interference_path;
mod tuple_bootstrap;
pub(crate) use checked_cell_aipw_response::CellAipwStructure;
pub(crate) use checked_graph_posterior_response::{
    CheckedGraphPosteriorResponse, CheckedGraphPosteriorResponsePremises,
};
pub(crate) use checked_static::CheckedStaticCommon;
pub(crate) use checked_static_class_response::{
    CheckedStaticClassResponse, CheckedStaticClassResponsePremises,
};
pub(crate) use checked_temporal_class_mediation::CheckedTemporalClassMediationOperation;
include!("execute_helpers_route.rs");
include!("execute_helpers_result.rs");
include!("execute_helpers_envelope.rs");
include!("execute_helpers_parametric.rs");

#[cfg(test)]
pub(super) use class_envelope_se::{
    BlockBootstrapPlan, circular_block_length, mixture_block_length,
    shared_circular_block_mixture_se_with_length,
};
pub(super) use class_envelope_se::{
    TemporalAtomDesign, envelope_shared_block_diagnostics, shared_block_mixture_message,
    shared_circular_block_mixture_se,
};
pub(super) use identified_set_diagnostics::{
    IDENTIFIED_SET_INTERVAL_LEVEL, completion_fit_seed, identified_set_interval_diagnostics,
    posterior_identified_set_interval, same_fit_inputs, same_fitted_mechanisms,
    same_fitted_problem,
};
pub(super) use tuple_bootstrap::{
    TupleObservationTarget, TupleReplicates, TupleSurface, tuple_block_observation_replicates,
};

pub(crate) use dispatch::push_gaussian_likelihood_disclosure;
pub(crate) use dispatch::push_lag_zero_average_effect_disclosure;
pub(crate) use dispatch::{
    ANOMALY_DIRICHLET_BACKEND, ATTRIBUTE_DIRICHLET_BACKEND, CHANGE_DIRICHLET_BACKEND,
    COUNTERFACTUAL_DIRICHLET_BACKEND, ROBUST_ATE_BOOTSTRAP_BACKEND,
};
pub(crate) use static_path::{CheckedUnknownTiered, DistributionGraph};
pub(crate) use transport_interference_path::live_transport_identification;

pub use checked_attribution::CheckedAttributionInfo;
pub use checked_bayesian_basis_ate::CheckedBayesianBasisAteInfo;
pub use checked_bayesian_dag_conditional::CheckedBayesianConditionalInfo;
pub use checked_bayesian_robust_ate::CheckedBayesianRobustAteInfo;
pub use checked_bayesian_specialist::CheckedBayesianSpecialistInfo;
pub(crate) use checked_bayesian_specialist::CheckedBayesianSpecialistOperation;
pub use checked_cell_aipw_response::CheckedCellAipwResponseInfo;
pub use checked_graph_posterior_response::CheckedGraphPosteriorResponseInfo;
pub use checked_interference::CheckedInterferenceInfo;
pub use checked_static_class_response::CheckedStaticClassResponseInfo;
pub use checked_static_mediation::CheckedStaticMediationInfo;
pub use checked_static_response::{CheckedDerivativeResponseInfo, CheckedStaticDagResponseInfo};
pub use checked_temporal_class_effect::CheckedTemporalClassEffectInfo;
pub use checked_temporal_class_mediation::{
    CheckedTemporalClassMediationInfo, CheckedTemporalMediationTargetInfo,
};
pub use checked_temporal_effect::CheckedTemporalDagEffectInfo;
pub use checked_temporal_mediation::CheckedTemporalMediationInfo;
pub use checked_transport_trial::CheckedTransportTrialInfo;
pub(crate) use checked_transport_trial::CheckedTransportTrialOperation;
pub(crate) use response_path::{
    class_aware_response_supported, graph_posterior_response_supported, response_witness_ate,
};
pub use temporal_path::checked_temporal_response::CheckedTemporalDagResponseInfo;
pub(crate) use temporal_posterior_response::dbn_posterior_response_supported;

#[cfg(test)]
mod block_length_tests;

#[cfg(test)]
mod quasi_plan_self_check_tests;

#[cfg(test)]
mod tests;
