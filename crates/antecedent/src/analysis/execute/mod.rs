//! Unified `CausalAnalysis` facade execution (split by modality for SRP).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::too_many_arguments,
    clippy::cast_precision_loss,
    clippy::wildcard_imports
)]

pub(super) use std::sync::Arc;
pub(super) use std::time::Instant;

pub(super) use super::latency::{INTERACTIVE_MAX_ENVELOPE_GRAPHS, LatencyMode};
pub(super) use antecedent_core::{
    AverageEffectQuery, CausalQuery, DataClassification, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, ExecutionContext, Intervention, MediationContrast, PopulationRegistry,
    ProvenanceGraph, TemporalEffectQuery, VariableId,
};
pub(super) use antecedent_data::{
    DiscoveryEstimationSplit, PanelData, TableView, TabularData, TimeSeriesData,
};
pub(super) use antecedent_discovery::{dag_from_adjacency_mask, temporal_dag_from_dbn_masks};
pub(super) use antecedent_estimate::{
    AnalyticSeKind, BayesianGCompWorkspace, BayesianGComputationAte, BayesianTemporalGcomp,
    ConditionalLinearAdjustment, EffectEstimate, EnvelopeOptions, EstimationWorkspace,
    FunctionalDistribution, FunctionalDistributionWorkspace, FunctionalEffect, GraphEffectDraws,
    LinearAdjustmentAte, OverlapPolicy, RdWorkspace, SharpRegressionDiscontinuity,
    TemporalLinearAdjustment, TemporalMediationEstimate, TemporalMediationEstimator,
    aggregate_effect_envelope, nonidentified_with_prior,
};
pub(super) use antecedent_expr::{CausalExprArena, IdentifiedEstimand};
pub(super) use antecedent_graph::{
    Admg, Dag, DenseNodeId, Pag, PagReview, TemporalCpdagReview, TemporalDag, TemporalGraphReview,
};
pub(super) use antecedent_identify::{
    DerivationTrace, IdentificationEnvelope, IdentificationPerformanceRecord, IdentificationResult,
    IdentificationStatus, SharpRdConfig, SharpRdIdentifier, TemporalBackdoorIdentifier,
    TemporalMediationIdentifier,
};
pub(super) use antecedent_prob::{
    GraphIdentFlag, InferenceDiagnostics, PriorSet, WeightedGraphSamples,
};
pub(super) use antecedent_validate::{
    BayesianSuiteContext, PosteriorPredictiveCheck, PriorPredictiveCheck, TemporalRefitContext,
    ValidationSuite, ValidatorId, stack_panel_tabular, with_conflict_summary,
    with_prior_sensitivity,
};

pub(super) use crate::callback_plan::mark_python_callback_plan;
pub(super) use crate::discovery::{
    BayesianDiscoverParams, GraphMcmcSchedule, StaticDiscoverParams,
    discover_ci_screened_posterior, discover_dbn_posterior, discover_exact_dag_posterior,
    discover_order_mcmc, discover_structure_mcmc,
};
pub(super) use crate::error::CausalError;
pub(super) use crate::gcm::{
    anomaly_attribution, attribute_distribution_change, attribute_unit_change, counterfactual_ite,
    fit_gcm, mechanism_change_detection,
};
pub(super) use crate::inference::{
    BayesianConfig, InferenceMode, resolve_bayesian_prior, resolve_bayesian_prior_with_conflict,
};
pub(super) use crate::planner::{
    CompiledAnalysis, GraphInput, LogicalAnalysisPlan, PhysicalExecutionPlan,
    StaticAteCompileInput, StaticDistributionCompileInput, StaticPagAteCompileInput,
    StaticPathSpecificCompileInput, compile_logical_distribution, compile_logical_path_specific,
    compile_logical_static_ate, compile_logical_static_pag_ate, compile_logical_temporal_effect,
    compile_logical_temporal_effect_classified, reject_dag_only_on_pag,
};
pub(super) use crate::result::CausalAnalysisResult;
pub(super) use crate::review::{
    PendingCpdagReview, PendingGraphReview, compile_review_required, compile_review_required_cpdag,
    compile_review_required_pag, compile_review_required_static_cpdag,
    compile_review_required_static_dag, compile_review_required_static_pag, ensure_review_complete,
};
pub(super) use crate::strategy_table::{
    DEFAULT_ADMG_ESTIMATOR_ID, DEFAULT_ADMG_IDENTIFIER_ID, DEFAULT_CONDITIONAL_ESTIMATOR_ID,
    DEFAULT_CONDITIONAL_IDENTIFIER_ID, DEFAULT_DISTRIBUTION_ESTIMATOR,
    DEFAULT_DISTRIBUTION_ESTIMATOR_ID, DEFAULT_DISTRIBUTION_IDENTIFIER,
    DEFAULT_DISTRIBUTION_IDENTIFIER_ID, DEFAULT_ESTIMATOR, DEFAULT_ESTIMATOR_ID,
    DEFAULT_IDENTIFIER, DEFAULT_IDENTIFIER_ID, DEFAULT_PAG_ESTIMATOR_ID, DEFAULT_PAG_IDENTIFIER_ID,
    DEFAULT_PATH_ESTIMATOR, DEFAULT_PATH_ESTIMATOR_ID, DEFAULT_PATH_IDENTIFIER,
    DEFAULT_PATH_IDENTIFIER_ID, EstimatorId, IdentifierId, StaticEstimateWorkspaces,
    estimate_provenance_step, estimate_static_effect, identify_admg, identify_pag,
    identify_provenance_step, identify_static, identify_static_query,
    identify_static_query_with_rd, require_identified, select_estimand, validate_static_pair,
};

pub(super) use super::builder::{CausalAnalysisBuilder, DataInput, RdConfig, RefuteSuite};
pub(super) use super::helpers::{
    AssembleArgs, assemble_result, effect_from_posterior, evaluate_bayesian_prior_sensitivity,
    overlap_diagnostic, project_for_ate_estimate, projection_diagnostic, provenance_pair,
    push_conflict_diagnostics, resolve_analysis_ci, run_fci_review, run_ges_review,
    run_jpcmci_plus_review, run_lingam_review, run_lpcmci_review, run_notears_review,
    run_pc_review, run_pcmci_plus_review, run_pcmci_review, run_refuters, run_rfci_review,
    run_rpcmci_discovery,
};

/// Prepared analysis (static or temporal).
#[derive(Clone)]
pub struct CausalAnalysis {
    pub(crate) data: DataInput,
    pub(crate) graph: GraphInput,
    pub(crate) query: CausalQuery,
    pub(crate) refute: RefuteSuite,
    pub(crate) bootstrap_replicates: u32,
    pub(crate) split: Option<DiscoveryEstimationSplit>,
    pub(crate) identifier: Option<IdentifierId>,
    pub(crate) estimator: Option<EstimatorId>,
    pub(crate) rd: Option<RdConfig>,
    pub(crate) inference: InferenceMode,
    pub(crate) overlap_policy: Option<OverlapPolicy>,
    pub(crate) population_registry: Option<PopulationRegistry>,
    pub(crate) discovery_ci:
        Option<Arc<dyn antecedent_stats::ConditionalIndependence + Send + Sync>>,
    pub(crate) custom_validators: Vec<Arc<dyn antecedent_validate::CustomEffectValidator>>,
    pub(crate) latency_mode: Option<super::latency::LatencyMode>,
    pub(crate) stage_sink: Option<Arc<dyn super::stage::StageResultSink>>,
}

impl std::fmt::Debug for CausalAnalysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CausalAnalysis")
            .field("data", &"<data>")
            .field("graph", &self.graph)
            .field("query", &"<query>")
            .field("refute", &self.refute)
            .field("bootstrap_replicates", &self.bootstrap_replicates)
            .field("split", &self.split)
            .field("identifier", &self.identifier)
            .field("estimator", &self.estimator)
            .field("rd", &self.rd)
            .field("inference", &self.inference)
            .field("overlap_policy", &self.overlap_policy)
            .field("population_registry", &self.population_registry.as_ref().map(|_| "<registry>"))
            .field("discovery_ci", &self.discovery_ci.as_ref().map(|_| "<dyn CI>"))
            .field("custom_validators", &self.custom_validators.len())
            .field("latency_mode", &self.latency_mode)
            .field("stage_sink_is_some", &self.stage_sink.is_some())
            .finish()
    }
}

mod attribution_path;
mod bayesian_path;
mod compile;
mod dispatch;
mod pag_path;
mod panel_path;
mod static_path;
mod temporal_path;
include!("support.rs");
