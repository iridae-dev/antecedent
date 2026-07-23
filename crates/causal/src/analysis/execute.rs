//! Unified `CausalAnalysis` facade.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

//! Analysis execution.

#![allow(
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::too_many_arguments,
    clippy::cast_precision_loss
)]

use std::sync::Arc;
use std::time::Instant;

use causal_core::{
    AverageEffectQuery, CausalQuery, DataClassification, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, ExecutionContext, Intervention, MediationContrast, PopulationRegistry,
    ProvenanceGraph, TemporalEffectQuery, VariableId,
};
use causal_data::{DiscoveryEstimationSplit, PanelData, TableView, TabularData, TimeSeriesData};
use causal_discovery::{dag_from_adjacency_mask, temporal_dag_from_dbn_masks};
use causal_estimate::{
    AnalyticSeKind, BayesianGCompWorkspace, BayesianGComputationAte, BayesianTemporalGcomp,
    ConditionalLinearAdjustment, EffectEstimate, EnvelopeOptions, EstimationWorkspace,
    FunctionalDistribution, FunctionalDistributionWorkspace, FunctionalEffect, GraphEffectDraws,
    LinearAdjustmentAte, OverlapPolicy, RdWorkspace, SharpRegressionDiscontinuity,
    TemporalLinearAdjustment, TemporalMediationEstimate, TemporalMediationEstimator,
    aggregate_effect_envelope, nonidentified_with_prior,
};
use causal_expr::{CausalExprArena, IdentifiedEstimand};
use causal_graph::{
    Admg, Dag, DenseNodeId, Pag, PagReview, TemporalCpdagReview, TemporalDag, TemporalGraphReview,
};
use causal_identify::{
    DerivationTrace, IdentificationEnvelope, IdentificationPerformanceRecord, IdentificationResult,
    IdentificationStatus, SharpRdConfig, SharpRdIdentifier, TemporalBackdoorIdentifier,
    TemporalMediationIdentifier,
};
use causal_prob::{GraphIdentFlag, InferenceDiagnostics, PriorSet, WeightedGraphSamples};
use super::latency::{INTERACTIVE_MAX_ENVELOPE_GRAPHS, LatencyMode};
use causal_validate::{
    BayesianSuiteContext, ExternalAlphaSensitivity, PosteriorPredictiveCheck, PriorPredictiveCheck,
    PriorSensitivity, TemporalRefitContext, ValidationSuite, ValidatorId, stack_panel_tabular,
    with_conflict_summary, with_prior_sensitivity,
};

use crate::callback_plan::mark_python_callback_plan;
use crate::discovery::{
    BayesianDiscoverParams, GraphMcmcSchedule, StaticDiscoverParams,
    discover_ci_screened_posterior, discover_dbn_posterior, discover_exact_dag_posterior,
    discover_order_mcmc, discover_structure_mcmc,
};
use crate::error::CausalError;
use crate::gcm::{
    anomaly_attribution, attribute_distribution_change, attribute_unit_change, counterfactual_ite,
    fit_gcm, mechanism_change_detection,
};
use crate::inference::{
    BayesianConfig, InferenceMode, resolve_bayesian_prior, resolve_bayesian_prior_with_conflict,
};
use crate::planner::{
    CompiledAnalysis, GraphInput, LogicalAnalysisPlan, PhysicalExecutionPlan,
    StaticAteCompileInput, StaticDistributionCompileInput, StaticPagAteCompileInput,
    StaticPathSpecificCompileInput, compile_logical_distribution, compile_logical_path_specific,
    compile_logical_static_ate, compile_logical_static_pag_ate, compile_logical_temporal_effect,
    compile_logical_temporal_effect_classified, reject_dag_only_on_pag,
};
use crate::result::CausalAnalysisResult;
use crate::review::{
    PendingCpdagReview, PendingGraphReview, compile_review_required, compile_review_required_cpdag,
    compile_review_required_pag, compile_review_required_static_cpdag,
    compile_review_required_static_dag, compile_review_required_static_pag, ensure_review_complete,
};
use crate::strategy_table::{
    DEFAULT_ADMG_ESTIMATOR_ID, DEFAULT_ADMG_IDENTIFIER_ID, DEFAULT_CONDITIONAL_ESTIMATOR_ID,
    DEFAULT_CONDITIONAL_IDENTIFIER_ID, DEFAULT_DISTRIBUTION_ESTIMATOR,
    DEFAULT_DISTRIBUTION_ESTIMATOR_ID, DEFAULT_DISTRIBUTION_IDENTIFIER,
    DEFAULT_DISTRIBUTION_IDENTIFIER_ID, DEFAULT_ESTIMATOR, DEFAULT_ESTIMATOR_ID,
    DEFAULT_IDENTIFIER, DEFAULT_IDENTIFIER_ID, DEFAULT_PAG_ESTIMATOR_ID, DEFAULT_PAG_IDENTIFIER_ID,
    DEFAULT_PATH_ESTIMATOR, DEFAULT_PATH_ESTIMATOR_ID, DEFAULT_PATH_IDENTIFIER,
    DEFAULT_PATH_IDENTIFIER_ID, EstimatorId, IdentifierId, StaticEstimateWorkspaces,
    estimate_provenance_step, estimate_static_effect, identify_admg, identify_pag,
    identify_provenance_step, identify_static, identify_static_query, identify_static_query_with_rd,
    require_identified, select_estimand, validate_static_pair,
};

use super::builder::{CausalAnalysisBuilder, DataInput, RdConfig, RefuteSuite};
use super::helpers::{
    AssembleArgs, assemble_result, effect_from_posterior, overlap_diagnostic, project_for_ate_estimate,
    projection_diagnostic, provenance_pair, push_conflict_diagnostics, resolve_analysis_ci,
    run_fci_review, run_ges_review, run_jpcmci_plus_review, run_lingam_review, run_lpcmci_review,
    run_notears_review, run_pc_review, run_pcmci_plus_review, run_pcmci_review, run_refuters,
    run_rfci_review, run_rpcmci_discovery,
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
    pub(crate) discovery_ci: Option<Arc<dyn causal_stats::ConditionalIndependence + Send + Sync>>,
    pub(crate) custom_validators: Vec<Arc<dyn causal_validate::CustomEffectValidator>>,
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

impl CausalAnalysis {
    /// Builder entry point.
    ///
    /// # Examples
    ///
    /// ```
    /// use causal::CausalAnalysis;
    ///
    /// let builder = CausalAnalysis::builder().bootstrap_replicates(50);
    /// let _ = builder;
    /// ```
    #[must_use]
    pub fn builder() -> CausalAnalysisBuilder {
        CausalAnalysisBuilder::new()
    }

    /// Mark physical plan when discovery CI override or custom validators are present.
    fn apply_callback_plan_marks(
        &self,
        mut record: causal_core::PhysicalExecutionPlanRecord,
        diagnostics: &mut Vec<causal_core::Diagnostic>,
    ) -> causal_core::PhysicalExecutionPlanRecord {
        if self.discovery_ci.is_some() {
            let (r, d) = mark_python_callback_plan(record, "ci");
            record = r;
            diagnostics.push(d);
        }
        if !self.custom_validators.is_empty() {
            let (r, d) = mark_python_callback_plan(record, "validator");
            record = r;
            diagnostics.push(d);
        }
        record
    }

    /// Compile logical plan only (inspectable semantics).
    ///
    /// # Errors
    ///
    /// Modality / query validation failures. Does not run discovery.
    pub fn compile_logical(&self) -> Result<LogicalAnalysisPlan, CausalError> {
        self.ensure_supported_combination()?;
        match (&self.data, &self.query, &self.graph) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::Static(graph),
            ) => {
                let (identifier, estimator) = self.resolve_static_pair();
                self.ensure_rd_config_present(&estimator)?;
                compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })
            }
            (DataInput::Tabular(data), CausalQuery::Distribution(q), GraphInput::Static(graph)) => {
                let (identifier, estimator) = self.resolve_distribution_pair();
                compile_logical_distribution(StaticDistributionCompileInput {
                    data,
                    graph,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })
            }
            (DataInput::Tabular(data), CausalQuery::PathSpecific(q), GraphInput::Static(graph)) => {
                let (identifier, estimator) = self.resolve_path_pair();
                compile_logical_path_specific(StaticPathSpecificCompileInput {
                    data,
                    graph,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::Temporal(graph),
            ) => {
                let class = match &self.data {
                    DataInput::Event(_) => DataClassification::Event,
                    _ => DataClassification::Temporal,
                };
                compile_logical_temporal_effect_classified(data, graph, q, self.split, false, class)
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverPcmci { .. }
                | GraphInput::DiscoverPcmciPlus { .. }
                | GraphInput::DiscoverRpcmci { .. }
                | GraphInput::DiscoverLpcmci { .. }
                | GraphInput::TemporalPag(_),
            ) => {
                let class = match &self.data {
                    DataInput::Event(_) => DataClassification::Event,
                    _ => DataClassification::Temporal,
                };
                compile_logical_temporal_effect_classified(
                    data,
                    &TemporalDag::empty(),
                    q,
                    self.split,
                    true,
                    class,
                )
            }
            (
                DataInput::MultiEnv(multi),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverJpcmciPlus { .. },
            ) => {
                let data = multi.environment(0).map_err(|e| CausalError::Compile {
                    message: format!("jpcmci+ multi-env: {e}"),
                })?;
                compile_logical_temporal_effect_classified(
                    data,
                    &TemporalDag::empty(),
                    q,
                    self.split,
                    true,
                    DataClassification::MultiEnvironment,
                )
            }
            (
                DataInput::Panel(panel),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverJpcmciPlus { .. }
                | GraphInput::DiscoverPcmci { .. }
                | GraphInput::DiscoverPcmciPlus { .. }
                | GraphInput::DiscoverLpcmci { .. }
                | GraphInput::Temporal(_),
            ) => {
                let data = &panel
                    .unit(0)
                    .map_err(|e| CausalError::Compile { message: format!("panel: {e}") })?
                    .series;
                let review = matches!(
                    self.graph,
                    GraphInput::DiscoverJpcmciPlus { .. }
                        | GraphInput::DiscoverPcmci { .. }
                        | GraphInput::DiscoverPcmciPlus { .. }
                        | GraphInput::DiscoverLpcmci { .. }
                );
                compile_logical_temporal_effect_classified(
                    data,
                    &TemporalDag::empty(),
                    q,
                    self.split,
                    review,
                    DataClassification::Panel,
                )
            }
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q), GraphInput::Pag(pag)) => {
                let (identifier, estimator) = self.resolve_pag_pair();
                reject_dag_only_on_pag(&self.graph, IdentifierId::parse(&identifier))?;
                compile_logical_static_pag_ate(StaticPagAteCompileInput {
                    data,
                    pag,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })
            }
            (
                DataInput::Tabular(data),
                CausalQuery::ConditionalEffect(q),
                GraphInput::Static(graph),
            ) => {
                let (identifier, estimator) = self.resolve_conditional_pair();
                // Logical plan reuses static ATE metadata with conditional estimator.
                let mut plan = compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph,
                    query: &q.inner,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?;
                plan.record.plan_id = Arc::from("static_conditional");
                plan.query = CausalQuery::ConditionalEffect(q.clone());
                Ok(plan)
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::Mediation(q),
                GraphInput::Temporal(graph),
            ) => {
                q.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
                let mut plan = compile_logical_temporal_effect(
                    data,
                    graph,
                    &TemporalEffectQuery::pulse(q.treatment, q.outcome, 1.0),
                    self.split,
                    false,
                )?;
                plan.record.plan_id = Arc::from("temporal_mediation");
                plan.record.identifier = Some(Arc::from("temporal.mediation"));
                plan.record.estimator = Some(Arc::from("temporal.mediation"));
                plan.record.query_variables = Arc::from([q.treatment, q.outcome]);
                plan.query = CausalQuery::Mediation(q.clone());
                Ok(plan)
            }
            (DataInput::Tabular(data), CausalQuery::Mediation(q), GraphInput::Static(graph)) => {
                q.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
                if !matches!(q.contrast, MediationContrast::Total) {
                    return Err(CausalError::Unsupported {
                        message: "static Mediation natural/direct/mediated contrasts require \
                             temporal data + TemporalDag; only MediationContrast::Total \
                             (front-door) is supported on a static DAG",
                    });
                }
                let ate = AverageEffectQuery::binary_ate(q.treatment, q.outcome);
                let mut plan = compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph,
                    query: &ate,
                    validation_suite: self.validation_suite_id(),
                    identifier: Arc::from("frontdoor"),
                    estimator: Arc::from("frontdoor.two_stage"),
                })?;
                plan.record.plan_id = Arc::from("static_mediation_total");
                plan.query = CausalQuery::Mediation(q.clone());
                Ok(plan)
            }
            (
                DataInput::Tabular(data),
                CausalQuery::Counterfactual(_)
                | CausalQuery::AnomalyAttribution(_)
                | CausalQuery::ChangeAttribution(_)
                | CausalQuery::MechanismChange(_)
                | CausalQuery::UnitChange(_),
                GraphInput::Static(_),
            ) => {
                // Parametric SCM paths: logical metadata only (no classic identifier/estimator).
                let (treatment, outcome) = gcm_query_vars(&self.query)?;
                self.query
                    .validate()
                    .map_err(|e| CausalError::Compile { message: e.to_string() })?;
                Ok(LogicalAnalysisPlan {
                    record: causal_core::LogicalAnalysisPlanRecord {
                        plan_id: Arc::from("gcm_query"),
                        data_classification: causal_core::DataClassification::Tabular,
                        discovery_algorithm: None,
                        graph_review_required: false,
                        identifier: Some(Arc::from("gcm.parametric")),
                        estimator: Some(Arc::from("gcm.fit")),
                        validation_suite: self.validation_suite_id(),
                        query_variables: Arc::from([treatment, outcome]),
                    },
                    query: self.query.clone(),
                    split: None,
                    row_count_hint: data.row_count() as u64,
                })
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverPc { .. },
            ) => {
                let (identifier, estimator) = self.resolve_static_pair();
                let n_vars = u32::try_from(data.schema().len()).unwrap_or(0);
                let empty = Dag::with_variables(n_vars);
                let mut plan = compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph: &empty,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?;
                plan.record.discovery_algorithm = Some(Arc::from("pc"));
                plan.record.graph_review_required = true;
                Ok(plan)
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverGes { .. },
            ) => {
                let (identifier, estimator) = self.resolve_static_pair();
                let n_vars = u32::try_from(data.schema().len()).unwrap_or(0);
                let empty = Dag::with_variables(n_vars);
                let mut plan = compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph: &empty,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?;
                plan.record.discovery_algorithm = Some(Arc::from("ges"));
                plan.record.graph_review_required = true;
                Ok(plan)
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverLingam { .. },
            ) => {
                let (identifier, estimator) = self.resolve_static_pair();
                let n_vars = u32::try_from(data.schema().len()).unwrap_or(0);
                let empty = Dag::with_variables(n_vars);
                let mut plan = compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph: &empty,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?;
                plan.record.discovery_algorithm = Some(Arc::from("direct_lingam"));
                plan.record.graph_review_required = true;
                Ok(plan)
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverNotears { .. },
            ) => {
                let (identifier, estimator) = self.resolve_static_pair();
                let n_vars = u32::try_from(data.schema().len()).unwrap_or(0);
                let empty = Dag::with_variables(n_vars);
                let mut plan = compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph: &empty,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?;
                plan.record.discovery_algorithm = Some(Arc::from("notears"));
                plan.record.graph_review_required = true;
                Ok(plan)
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                graph @ GraphInput::DiscoverFci { .. },
            ) => {
                let (identifier, estimator) = self.resolve_static_pair();
                reject_dag_only_on_pag(graph, &identifier)?;
                let n_vars = u32::try_from(data.schema().len()).unwrap_or(0);
                let empty = Dag::with_variables(n_vars);
                let mut plan = compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph: &empty,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?;
                plan.record.discovery_algorithm = Some(Arc::from("fci"));
                plan.record.graph_review_required = true;
                Ok(plan)
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                graph @ GraphInput::DiscoverRfci { .. },
            ) => {
                let (identifier, estimator) = self.resolve_static_pair();
                reject_dag_only_on_pag(graph, &identifier)?;
                let n_vars = u32::try_from(data.schema().len()).unwrap_or(0);
                let empty = Dag::with_variables(n_vars);
                let mut plan = compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph: &empty,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?;
                plan.record.discovery_algorithm = Some(Arc::from("rfci"));
                plan.record.graph_review_required = true;
                Ok(plan)
            }
            _ => Err(CausalError::Unsupported {
                message: "unsupported data/graph/query combination",
            }),
        }
    }

    /// Compile logical → physical plan (or review-required).
    ///
    /// # Errors
    ///
    /// Modality / resource / discovery failures.
    pub fn compile(&self, ctx: &ExecutionContext) -> Result<CompiledAnalysis, CausalError> {
        self.ensure_supported_combination()?;
        match (&self.data, &self.query, &self.graph) {
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::Static(graph),
            ) => {
                let (identifier, estimator) = self.resolve_static_pair();
                self.ensure_rd_config_present(&estimator)?;
                let logical = compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?;
                let physical = logical.compile_physical(ctx)?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (DataInput::Tabular(data), CausalQuery::Distribution(q), GraphInput::Static(graph)) => {
                let (identifier, estimator) = self.resolve_distribution_pair();
                let logical = compile_logical_distribution(StaticDistributionCompileInput {
                    data,
                    graph,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?;
                let physical = logical.compile_physical(ctx)?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (DataInput::Tabular(data), CausalQuery::PathSpecific(q), GraphInput::Static(graph)) => {
                let (identifier, estimator) = self.resolve_path_pair();
                let logical = compile_logical_path_specific(StaticPathSpecificCompileInput {
                    data,
                    graph,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?;
                let physical = logical.compile_physical(ctx)?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::Temporal(graph),
            ) => {
                let class = match &self.data {
                    DataInput::Event(_) => DataClassification::Event,
                    _ => DataClassification::Temporal,
                };
                let logical = compile_logical_temporal_effect_classified(
                    data, graph, q, self.split, false, class,
                )?;
                ensure_review_complete(&logical)?;
                let physical = logical.compile_physical_with_graph(ctx, Some(graph.clone()))?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverPcmci { max_lag, alpha, fdr, accept_discovered },
            ) => {
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let review = run_pcmci_review(data, *max_lag, *alpha, *fdr, ci, ctx)?;
                if *accept_discovered {
                    PendingGraphReview::new(review, data.row_count(), q.clone(), self.split)
                        .accept_all()
                        .finish(data, ctx)
                } else {
                    Ok(compile_review_required(review))
                }
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverPcmciPlus { max_lag, alpha, fdr, accept_discovered },
            ) => {
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let review = run_pcmci_plus_review(data, *max_lag, *alpha, *fdr, ci, ctx)?;
                if *accept_discovered && review.pending_undirected.is_empty() {
                    PendingCpdagReview::new(review, data.row_count(), q.clone(), self.split)
                        .accept_all_directed()
                        .finish(data, ctx)
                } else {
                    Ok(compile_review_required_cpdag(review))
                }
            }
            (
                DataInput::MultiEnv(multi),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverJpcmciPlus {
                    max_lag,
                    alpha,
                    fdr,
                    accept_discovered,
                    multi_dataset,
                },
            ) => {
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let review =
                    run_jpcmci_plus_review(multi, *max_lag, *alpha, *fdr, multi_dataset, ci, ctx)?;
                let data = multi.environment(0).map_err(|e| CausalError::Compile {
                    message: format!("jpcmci+ multi-env: {e}"),
                })?;
                if *accept_discovered && review.pending_undirected.is_empty() {
                    PendingCpdagReview::new(review, data.row_count(), q.clone(), self.split)
                        .accept_all_directed()
                        .finish(data, ctx)
                } else {
                    Ok(compile_review_required_cpdag(review))
                }
            }
            (
                DataInput::Panel(panel),
                CausalQuery::TemporalEffect(q),
                GraphInput::Temporal(graph),
            ) => {
                let data = &panel
                    .unit(0)
                    .map_err(|e| CausalError::Compile { message: format!("panel: {e}") })?
                    .series;
                let logical = compile_logical_temporal_effect_classified(
                    data,
                    graph,
                    q,
                    self.split,
                    false,
                    DataClassification::Panel,
                )?;
                ensure_review_complete(&logical)?;
                let physical = logical.compile_physical_with_graph(ctx, Some(graph.clone()))?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (
                DataInput::Panel(panel),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverJpcmciPlus {
                    max_lag,
                    alpha,
                    fdr,
                    accept_discovered,
                    multi_dataset,
                },
            ) => {
                let multi = panel.as_multi_env().map_err(|e| CausalError::Compile {
                    message: format!("panel as multi-env: {e}"),
                })?;
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let review =
                    run_jpcmci_plus_review(&multi, *max_lag, *alpha, *fdr, multi_dataset, ci, ctx)?;
                let data = &panel
                    .unit(0)
                    .map_err(|e| CausalError::Compile { message: format!("panel: {e}") })?
                    .series;
                if *accept_discovered && review.pending_undirected.is_empty() {
                    let compiled =
                        PendingCpdagReview::new(review, data.row_count(), q.clone(), self.split)
                            .accept_all_directed()
                            .finish(data, ctx)?;
                    Ok(mark_panel_classification(compiled))
                } else {
                    Ok(compile_review_required_cpdag(review))
                }
            }
            (
                DataInput::Panel(panel),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverPcmci { max_lag, alpha, fdr, accept_discovered },
            ) => {
                let pooled = stack_panel_tabular(panel).map_err(CausalError::from)?;
                let n = pooled.row_count();
                let series = TimeSeriesData::try_new(
                    pooled.storage().clone(),
                    causal_data::TimeIndex {
                        regularity: causal_data::SamplingRegularity::Regular { interval_ns: 1 },
                        length: n,
                    },
                )
                .map_err(CausalError::from)?;
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let review = run_pcmci_review(&series, *max_lag, *alpha, *fdr, ci, ctx)?;
                let data = &panel
                    .unit(0)
                    .map_err(|e| CausalError::Compile { message: format!("panel: {e}") })?
                    .series;
                if *accept_discovered {
                    let compiled =
                        PendingGraphReview::new(review, data.row_count(), q.clone(), self.split)
                            .accept_all()
                            .finish(data, ctx)?;
                    Ok(mark_panel_classification(compiled))
                } else {
                    Ok(compile_review_required(review))
                }
            }
            (
                DataInput::Panel(panel),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverPcmciPlus { max_lag, alpha, fdr, accept_discovered },
            ) => {
                let pooled = stack_panel_tabular(panel).map_err(CausalError::from)?;
                let n = pooled.row_count();
                let series = TimeSeriesData::try_new(
                    pooled.storage().clone(),
                    causal_data::TimeIndex {
                        regularity: causal_data::SamplingRegularity::Regular { interval_ns: 1 },
                        length: n,
                    },
                )
                .map_err(CausalError::from)?;
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let review = run_pcmci_plus_review(&series, *max_lag, *alpha, *fdr, ci, ctx)?;
                let data = &panel
                    .unit(0)
                    .map_err(|e| CausalError::Compile { message: format!("panel: {e}") })?
                    .series;
                if *accept_discovered && review.pending_undirected.is_empty() {
                    let compiled =
                        PendingCpdagReview::new(review, data.row_count(), q.clone(), self.split)
                            .accept_all_directed()
                            .finish(data, ctx)?;
                    Ok(mark_panel_classification(compiled))
                } else {
                    Ok(compile_review_required_cpdag(review))
                }
            }
            (
                DataInput::Panel(panel),
                CausalQuery::TemporalEffect(_q),
                GraphInput::DiscoverLpcmci { max_lag, alpha, fdr, accept_discovered: _ },
            ) => {
                let pooled = stack_panel_tabular(panel).map_err(CausalError::from)?;
                let n = pooled.row_count();
                let series = TimeSeriesData::try_new(
                    pooled.storage().clone(),
                    causal_data::TimeIndex {
                        regularity: causal_data::SamplingRegularity::Regular { interval_ns: 1 },
                        length: n,
                    },
                )
                .map_err(CausalError::from)?;
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let review = run_lpcmci_review(&series, *max_lag, *alpha, *fdr, ci, ctx)?;
                Ok(compile_review_required_pag(review))
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(_q),
                GraphInput::DiscoverRpcmci {
                    max_lag,
                    alpha,
                    fdr,
                    accept_discovered,
                    regime_assignment,
                },
            ) => {
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let result =
                    run_rpcmci_discovery(data, *max_lag, *alpha, *fdr, regime_assignment, ci, ctx)?;
                // Multi-regime estimation is not auto-wired; surface the first regime's CPDAG
                // for review. Auto-accept only when a single fully-oriented regime exists.
                let Some(first) = result.per_regime.first() else {
                    return Err(CausalError::Compile {
                        message: "RPCMCI returned no regime graphs".into(),
                    });
                };
                let review = first.review.clone();
                if *accept_discovered
                    && result.per_regime.len() == 1
                    && review.pending_undirected.is_empty()
                {
                    let q = match &self.query {
                        CausalQuery::TemporalEffect(q) => q.clone(),
                        _ => unreachable!(),
                    };
                    PendingCpdagReview::new(review, data.row_count(), q, self.split)
                        .accept_all_directed()
                        .finish(data, ctx)
                } else {
                    Ok(compile_review_required_cpdag(review))
                }
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverDbnPosterior { .. },
            ) => self.compile_dbn_posterior_temporal(data, q, ctx),
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverLpcmci { max_lag, alpha, fdr, accept_discovered },
            ) => {
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let review = run_lpcmci_review(data, *max_lag, *alpha, *fdr, ci, ctx)?;
                // Temporal backdoor is DAG-only. Auto-accept only when the PAG is already
                // fully definite-directed (no circle/ambiguous marks) — never invent orientations.
                if *accept_discovered && review.is_complete() {
                    match review.graph.try_into_temporal_dag() {
                        Ok(dag) => {
                            let mut logical =
                                compile_logical_temporal_effect(data, &dag, q, self.split, false)?;
                            // Completion→DAG (not class-aware temporal PAG ID).
                            logical.record.discovery_algorithm =
                                Some(Arc::from("lpcmci.pag_completed_to_dag"));
                            let physical = logical.compile_physical_with_graph(ctx, Some(dag))?;
                            Ok(CompiledAnalysis::Ready(physical))
                        }
                        Err(_) => Ok(compile_review_required_pag(review)),
                    }
                } else {
                    Ok(compile_review_required_pag(review))
                }
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::TemporalPag(pag),
            ) => {
                let review =
                    causal_graph::TemporalPagReview::from_pag(pag.clone(), "supplied.temporal_pag");
                if review.is_complete() {
                    match review.graph.try_into_temporal_dag() {
                        Ok(dag) => {
                            let mut logical =
                                compile_logical_temporal_effect(data, &dag, q, self.split, false)?;
                            logical.record.discovery_algorithm =
                                Some(Arc::from("supplied.temporal_pag.completed_to_dag"));
                            let physical = logical.compile_physical_with_graph(ctx, Some(dag))?;
                            Ok(CompiledAnalysis::Ready(physical))
                        }
                        Err(_) => Ok(compile_review_required_pag(review)),
                    }
                } else {
                    Ok(compile_review_required_pag(review))
                }
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::TemporalCpdag(cpdag),
            ) => match cpdag.try_into_temporal_dag() {
                Ok(dag) => {
                    let logical =
                        compile_logical_temporal_effect(data, &dag, q, self.split, false)?;
                    let physical = logical.compile_physical_with_graph(ctx, Some(dag))?;
                    Ok(CompiledAnalysis::Ready(physical))
                }
                Err(_) => Ok(compile_review_required_cpdag(
                    causal_graph::TemporalCpdagReview::from_cpdag(
                        cpdag.clone(),
                        "supplied.temporal_cpdag",
                    ),
                )),
            },
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverPc { alpha, max_cond_size, fdr, accept_discovered },
            ) => {
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let review = run_pc_review(data, *alpha, *max_cond_size, *fdr, ci, ctx)?;
                if *accept_discovered && review.pending_undirected.is_empty() {
                    let mut accepted = review;
                    accepted.pending_edges = Arc::from([]);
                    let dag = accepted
                        .try_into_dag()
                        .map_err(|e| CausalError::review_required_msg(e.to_string()))?;
                    let (identifier, estimator) = self.resolve_static_pair();
                    self.ensure_rd_config_present(&estimator)?;
                    let mut logical = compile_logical_static_ate(StaticAteCompileInput {
                        data,
                        graph: &dag,
                        query: q,
                        validation_suite: self.validation_suite_id(),
                        identifier,
                        estimator,
                    })?;
                    logical.record.discovery_algorithm = Some(Arc::from("pc"));
                    let physical = logical.compile_physical_with_graphs(ctx, None, Some(dag))?;
                    Ok(CompiledAnalysis::Ready(physical))
                } else {
                    Ok(compile_review_required_static_cpdag(review))
                }
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverGes { alpha, max_cond_size, fdr, accept_discovered },
            ) => {
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let review = run_ges_review(data, *alpha, *max_cond_size, *fdr, ci, ctx)?;
                if *accept_discovered && review.pending_undirected.is_empty() {
                    let mut accepted = review;
                    accepted.pending_edges = Arc::from([]);
                    let dag = accepted
                        .try_into_dag()
                        .map_err(|e| CausalError::review_required_msg(e.to_string()))?;
                    let (identifier, estimator) = self.resolve_static_pair();
                    self.ensure_rd_config_present(&estimator)?;
                    let mut logical = compile_logical_static_ate(StaticAteCompileInput {
                        data,
                        graph: &dag,
                        query: q,
                        validation_suite: self.validation_suite_id(),
                        identifier,
                        estimator,
                    })?;
                    logical.record.discovery_algorithm = Some(Arc::from("ges"));
                    let physical = logical.compile_physical_with_graphs(ctx, None, Some(dag))?;
                    Ok(CompiledAnalysis::Ready(physical))
                } else {
                    Ok(compile_review_required_static_cpdag(review))
                }
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverLingam { max_cond_size, prune_threshold, accept_discovered },
            ) => {
                let review = run_lingam_review(data, *max_cond_size, *prune_threshold, ctx)?;
                if *accept_discovered {
                    let dag = review
                        .accept_all()
                        .try_into_dag()
                        .map_err(|e| CausalError::review_required_msg(e.to_string()))?;
                    let (identifier, estimator) = self.resolve_static_pair();
                    self.ensure_rd_config_present(&estimator)?;
                    let mut logical = compile_logical_static_ate(StaticAteCompileInput {
                        data,
                        graph: &dag,
                        query: q,
                        validation_suite: self.validation_suite_id(),
                        identifier,
                        estimator,
                    })?;
                    logical.record.discovery_algorithm = Some(Arc::from("direct_lingam"));
                    let physical = logical.compile_physical_with_graphs(ctx, None, Some(dag))?;
                    Ok(CompiledAnalysis::Ready(physical))
                } else {
                    Ok(compile_review_required_static_dag(review))
                }
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverNotears {
                    max_cond_size,
                    lambda,
                    threshold,
                    standardize,
                    accept_discovered,
                },
            ) => {
                let review = run_notears_review(
                    data,
                    *max_cond_size,
                    *lambda,
                    *threshold,
                    *standardize,
                    ctx,
                )?;
                if *accept_discovered {
                    let dag = review
                        .accept_all()
                        .try_into_dag()
                        .map_err(|e| CausalError::review_required_msg(e.to_string()))?;
                    let (identifier, estimator) = self.resolve_static_pair();
                    self.ensure_rd_config_present(&estimator)?;
                    let mut logical = compile_logical_static_ate(StaticAteCompileInput {
                        data,
                        graph: &dag,
                        query: q,
                        validation_suite: self.validation_suite_id(),
                        identifier,
                        estimator,
                    })?;
                    logical.record.discovery_algorithm = Some(Arc::from("notears"));
                    let physical = logical.compile_physical_with_graphs(ctx, None, Some(dag))?;
                    Ok(CompiledAnalysis::Ready(physical))
                } else {
                    Ok(compile_review_required_static_dag(review))
                }
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverExactDagPosterior
                | GraphInput::DiscoverOrderMcmc { .. }
                | GraphInput::DiscoverStructureMcmc { .. }
                | GraphInput::DiscoverCiScreenedPosterior { .. },
            ) => self.compile_graph_posterior_static_ate(data, q, ctx),
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                graph @ GraphInput::DiscoverFci { alpha, max_cond_size, fdr, accept_discovered },
            ) => {
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let (identifier, estimator) = self.resolve_pag_pair();
                reject_dag_only_on_pag(graph, IdentifierId::parse(&identifier))?;
                let review = run_fci_review(data, *alpha, *max_cond_size, *fdr, ci, ctx)?;
                // Accept-as-PAG: circle marks are handled by generalized adjustment over
                // MAG completions (same path as GraphInput::Pag). Review is only when
                // accept_discovered is false.
                if *accept_discovered {
                    let mut logical = compile_logical_static_pag_ate(StaticPagAteCompileInput {
                        data,
                        pag: &review.graph,
                        query: q,
                        validation_suite: self.validation_suite_id(),
                        identifier,
                        estimator,
                    })?;
                    logical.record.discovery_algorithm = Some(Arc::from("fci"));
                    let physical = logical.compile_physical_with_all_graphs(
                        ctx,
                        None,
                        None,
                        Some(review.graph.clone()),
                    )?;
                    Ok(CompiledAnalysis::Ready(physical))
                } else {
                    Ok(compile_review_required_static_pag(review))
                }
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                graph @ GraphInput::DiscoverRfci { alpha, max_cond_size, fdr, accept_discovered },
            ) => {
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let (identifier, estimator) = self.resolve_pag_pair();
                reject_dag_only_on_pag(graph, IdentifierId::parse(&identifier))?;
                let review = run_rfci_review(data, *alpha, *max_cond_size, *fdr, ci, ctx)?;
                if *accept_discovered {
                    let mut logical = compile_logical_static_pag_ate(StaticPagAteCompileInput {
                        data,
                        pag: &review.graph,
                        query: q,
                        validation_suite: self.validation_suite_id(),
                        identifier,
                        estimator,
                    })?;
                    logical.record.discovery_algorithm = Some(Arc::from("rfci"));
                    let physical = logical.compile_physical_with_all_graphs(
                        ctx,
                        None,
                        None,
                        Some(review.graph.clone()),
                    )?;
                    Ok(CompiledAnalysis::Ready(physical))
                } else {
                    Ok(compile_review_required_static_pag(review))
                }
            }
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q), GraphInput::Pag(pag)) => {
                let (identifier, estimator) = self.resolve_pag_pair();
                reject_dag_only_on_pag(&self.graph, IdentifierId::parse(&identifier))?;
                let logical = compile_logical_static_pag_ate(StaticPagAteCompileInput {
                    data,
                    pag,
                    query: q,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?;
                let physical =
                    logical.compile_physical_with_all_graphs(ctx, None, None, Some(pag.clone()))?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q), GraphInput::Cpdag(cpdag)) => {
                match cpdag.try_into_dag() {
                    Ok(dag) => {
                        let (identifier, estimator) = self.resolve_static_pair();
                        self.ensure_rd_config_present(&estimator)?;
                        let logical = compile_logical_static_ate(StaticAteCompileInput {
                            data,
                            graph: &dag,
                            query: q,
                            validation_suite: self.validation_suite_id(),
                            identifier,
                            estimator,
                        })?;
                        let physical =
                            logical.compile_physical_with_graphs(ctx, None, Some(dag))?;
                        Ok(CompiledAnalysis::Ready(physical))
                    }
                    Err(_) => Ok(compile_review_required_static_cpdag(
                        causal_graph::CpdagReview::from_cpdag(cpdag.clone(), "supplied.cpdag"),
                    )),
                }
            }
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q), GraphInput::Admg(admg)) => {
                if admg_has_bidirected(admg) {
                    let (identifier, estimator) = self.resolve_admg_pair();
                    validate_static_pair(
                        IdentifierId::parse(&identifier),
                        EstimatorId::parse(&estimator),
                    )?;
                    q.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
                    let record = causal_core::LogicalAnalysisPlanRecord {
                        plan_id: Arc::from("static_admg_ate"),
                        data_classification: causal_core::DataClassification::Tabular,
                        discovery_algorithm: None,
                        graph_review_required: false,
                        identifier: Some(identifier),
                        estimator: Some(estimator),
                        validation_suite: self.validation_suite_id(),
                        query_variables: Arc::from([q.treatment, q.outcome]),
                    };
                    let logical = LogicalAnalysisPlan {
                        record,
                        query: CausalQuery::AverageEffect(q.clone()),
                        split: None,
                        row_count_hint: data.row_count() as u64,
                    };
                    logical.validate()?;
                    let physical = logical.compile_physical(ctx)?;
                    Ok(CompiledAnalysis::Ready(physical))
                } else {
                    let dag = admg_to_dag(admg)?;
                    let (identifier, estimator) = self.resolve_static_pair();
                    self.ensure_rd_config_present(&estimator)?;
                    let logical = compile_logical_static_ate(StaticAteCompileInput {
                        data,
                        graph: &dag,
                        query: q,
                        validation_suite: self.validation_suite_id(),
                        identifier,
                        estimator,
                    })?;
                    let physical = logical.compile_physical_with_graphs(ctx, None, Some(dag))?;
                    Ok(CompiledAnalysis::Ready(physical))
                }
            }
            (
                DataInput::Tabular(data),
                CausalQuery::ConditionalEffect(q),
                GraphInput::Static(graph),
            ) => {
                let (identifier, estimator) = self.resolve_conditional_pair();
                let mut logical = compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph,
                    query: &q.inner,
                    validation_suite: self.validation_suite_id(),
                    identifier,
                    estimator,
                })?;
                logical.record.plan_id = Arc::from("static_conditional");
                logical.query = CausalQuery::ConditionalEffect(q.clone());
                let physical =
                    logical.compile_physical_with_graphs(ctx, None, Some(graph.clone()))?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::Mediation(q),
                GraphInput::Temporal(graph),
            ) => {
                q.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
                let mut logical = compile_logical_temporal_effect(
                    data,
                    graph,
                    &TemporalEffectQuery::pulse(q.treatment, q.outcome, 1.0),
                    self.split,
                    false,
                )?;
                logical.record.plan_id = Arc::from("temporal_mediation");
                logical.record.identifier = Some(Arc::from("temporal.mediation"));
                logical.record.estimator = Some(Arc::from("temporal.mediation"));
                logical.record.query_variables = Arc::from([q.treatment, q.outcome]);
                logical.query = CausalQuery::Mediation(q.clone());
                let physical = logical.compile_physical_with_graph(ctx, Some(graph.clone()))?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (DataInput::Tabular(data), CausalQuery::Mediation(q), GraphInput::Static(graph)) => {
                q.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
                if !matches!(q.contrast, MediationContrast::Total) {
                    return Err(CausalError::Unsupported {
                        message: "static Mediation natural/direct/mediated contrasts require \
                             temporal data + TemporalDag; only MediationContrast::Total \
                             (front-door) is supported on a static DAG",
                    });
                }
                let ate = AverageEffectQuery::binary_ate(q.treatment, q.outcome);
                let mut logical = compile_logical_static_ate(StaticAteCompileInput {
                    data,
                    graph,
                    query: &ate,
                    validation_suite: self.validation_suite_id(),
                    identifier: Arc::from("frontdoor"),
                    estimator: Arc::from("frontdoor.two_stage"),
                })?;
                logical.record.plan_id = Arc::from("static_mediation_total");
                logical.query = CausalQuery::Mediation(q.clone());
                let physical =
                    logical.compile_physical_with_graphs(ctx, None, Some(graph.clone()))?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (
                DataInput::Tabular(data),
                CausalQuery::Counterfactual(_)
                | CausalQuery::AnomalyAttribution(_)
                | CausalQuery::ChangeAttribution(_)
                | CausalQuery::MechanismChange(_)
                | CausalQuery::UnitChange(_),
                GraphInput::Static(graph),
            ) => {
                let logical = self.compile_logical()?;
                let physical =
                    logical.compile_physical_with_graphs(ctx, None, Some(graph.clone()))?;
                let _ = data;
                Ok(CompiledAnalysis::Ready(physical))
            }
            _ => Err(CausalError::Unsupported {
                message: "unsupported data/graph/query combination",
            }),
        }
    }

    fn validation_suite_id(&self) -> Option<Arc<str>> {
        match self.refute {
            RefuteSuite::None => None,
            RefuteSuite::Cheap => Some(Arc::from("overlap+evalue")),
            RefuteSuite::PlaceboAndRcc => Some(Arc::from("placebo+rcc")),
            RefuteSuite::Full => Some(Arc::from("validation.full")),
        }
    }

    fn ensure_supported_combination(&self) -> Result<(), CausalError> {
        match (&self.data, &self.query, &self.graph) {
            (_, CausalQuery::Distribution(_), graph)
                if !matches!(
                    (&self.data, graph),
                    (DataInput::Tabular(_), GraphInput::Static(_))
                ) =>
            {
                return Err(CausalError::Unsupported {
                    message: "CausalQuery::Distribution requires tabular data and a static DAG",
                });
            }
            (_, CausalQuery::PathSpecific(_), graph)
                if !matches!(
                    (&self.data, graph),
                    (DataInput::Tabular(_), GraphInput::Static(_))
                ) =>
            {
                return Err(CausalError::Unsupported {
                    message: "CausalQuery::PathSpecific requires tabular data and a static DAG",
                });
            }
            (DataInput::Tabular(_), CausalQuery::TemporalEffect(_), _) => {
                return Err(CausalError::Compile {
                    message: "temporal effect query requires temporal data".into(),
                });
            }
            (
                DataInput::Temporal(_) | DataInput::Event(_) | DataInput::MultiEnv(_),
                CausalQuery::AverageEffect(_),
                _,
            ) => {
                return Err(CausalError::Compile {
                    message: "static ATE on temporal data is unsupported; use TemporalEffect"
                        .into(),
                });
            }
            (DataInput::Panel(_), CausalQuery::AverageEffect(_), _) => {
                return Err(CausalError::Compile {
                    message: "static ATE on panel data is unsupported; use TemporalEffect".into(),
                });
            }
            (
                DataInput::Tabular(_),
                _,
                GraphInput::DiscoverPcmci { .. }
                | GraphInput::DiscoverPcmciPlus { .. }
                | GraphInput::DiscoverJpcmciPlus { .. }
                | GraphInput::DiscoverRpcmci { .. }
                | GraphInput::DiscoverLpcmci { .. }
                | GraphInput::TemporalPag(_),
            ) => {
                return Err(CausalError::Compile {
                    message:
                        "PCMCI-family / temporal PAG discovery requires temporal data and a temporal effect query"
                            .into(),
                });
            }
            (
                DataInput::Temporal(_) | DataInput::Event(_) | DataInput::MultiEnv(_),
                _,
                GraphInput::DiscoverPc { .. }
                | GraphInput::DiscoverGes { .. }
                | GraphInput::DiscoverLingam { .. }
                | GraphInput::DiscoverNotears { .. }
                | GraphInput::DiscoverFci { .. }
                | GraphInput::DiscoverRfci { .. },
            ) => {
                return Err(CausalError::Compile {
                    message: "static PC/GES/LiNGAM/NOTEARS/FCI/RFCI discovery requires tabular data and AverageEffect"
                        .into(),
                });
            }
            (
                DataInput::Temporal(_) | DataInput::Event(_),
                _,
                GraphInput::DiscoverJpcmciPlus { .. },
            ) => {
                return Err(CausalError::Compile {
                    message:
                        "J-PCMCI+ discovery requires series_multi (MultiEnvironmentData) or panel"
                            .into(),
                });
            }
            (DataInput::MultiEnv(_), _, graph)
                if !matches!(graph, GraphInput::DiscoverJpcmciPlus { .. }) =>
            {
                return Err(CausalError::Compile {
                    message: "multi-environment data currently supports only DiscoverJpcmciPlus"
                        .into(),
                });
            }
            (DataInput::Panel(_), _, graph)
                if !matches!(
                    graph,
                    GraphInput::DiscoverJpcmciPlus { .. }
                        | GraphInput::DiscoverPcmci { .. }
                        | GraphInput::DiscoverPcmciPlus { .. }
                        | GraphInput::DiscoverLpcmci { .. }
                        | GraphInput::Temporal(_)
                ) =>
            {
                return Err(CausalError::Compile {
                    message: "panel data supports DiscoverJpcmciPlus, DiscoverPcmci/Plus/Lpcmci \
                              (pooled units), or a supplied TemporalDag"
                        .into(),
                });
            }
            (
                DataInput::Temporal(_) | DataInput::Event(_) | DataInput::MultiEnv(_),
                _,
                GraphInput::Pag(_),
            ) => {
                return Err(CausalError::Compile {
                    message: "static Pag requires tabular data and an average-effect query".into(),
                });
            }
            (DataInput::Tabular(_), CausalQuery::AverageEffect(_), GraphInput::Pag(_)) => {
                let (identifier, _) = self.resolve_pag_pair();
                reject_dag_only_on_pag(&self.graph, IdentifierId::parse(&identifier))?;
            }
            _ => {}
        }
        // The temporal path is linear/temporal-backdoor only; refuse an explicitly
        // selected non-temporal identifier/estimator rather than silently ignoring it.
        if matches!(&self.query, CausalQuery::TemporalEffect(_)) {
            if let Some(id) = &self.identifier {
                if *id != IdentifierId::TemporalBackdoorUnfolded {
                    return Err(CausalError::Compile {
                        message: format!(
                            "temporal path only supports identifier \"temporal.backdoor.unfolded\"; got {id:?}"
                        ),
                    });
                }
            }
            if let Some(est) = &self.estimator {
                if *est != EstimatorId::TemporalLinearAdjustment {
                    return Err(CausalError::Compile {
                        message: format!(
                            "temporal path only supports estimator \"temporal.linear.adjustment\"; got {est:?}"
                        ),
                    });
                }
            }
        }
        Ok(())
    }

    /// Resolve builder-selected identifier/estimator ids, applying static-ATE defaults.
    fn resolve_static_pair(&self) -> (Arc<str>, Arc<str>) {
        let identifier = self.identifier.as_ref().unwrap_or(&DEFAULT_IDENTIFIER_ID);
        let estimator = self.estimator.as_ref().unwrap_or(&DEFAULT_ESTIMATOR_ID);
        (Arc::from(identifier.as_str()), Arc::from(estimator.as_str()))
    }

    /// Resolve identifier/estimator for PAG ATE (generalized adjustment).
    fn resolve_pag_pair(&self) -> (Arc<str>, Arc<str>) {
        let identifier = self.identifier.as_ref().unwrap_or(&DEFAULT_PAG_IDENTIFIER_ID);
        let estimator = self.estimator.as_ref().unwrap_or(&DEFAULT_PAG_ESTIMATOR_ID);
        (Arc::from(identifier.as_str()), Arc::from(estimator.as_str()))
    }

    /// Resolve identifier/estimator for ADMG ATE (general ID + functional effect).
    fn resolve_admg_pair(&self) -> (Arc<str>, Arc<str>) {
        let identifier = self.identifier.as_ref().unwrap_or(&DEFAULT_ADMG_IDENTIFIER_ID);
        let estimator = self.estimator.as_ref().unwrap_or(&DEFAULT_ADMG_ESTIMATOR_ID);
        (Arc::from(identifier.as_str()), Arc::from(estimator.as_str()))
    }

    /// Resolve identifier/estimator for ConditionalEffect.
    fn resolve_conditional_pair(&self) -> (Arc<str>, Arc<str>) {
        let identifier = self.identifier.as_ref().unwrap_or(&DEFAULT_CONDITIONAL_IDENTIFIER_ID);
        let estimator = self.estimator.as_ref().unwrap_or(&DEFAULT_CONDITIONAL_ESTIMATOR_ID);
        (Arc::from(identifier.as_str()), Arc::from(estimator.as_str()))
    }

    /// Resolve identifier/estimator for Distribution queries.
    fn resolve_distribution_pair(&self) -> (Arc<str>, Arc<str>) {
        let identifier = self.identifier.as_ref().unwrap_or(&DEFAULT_DISTRIBUTION_IDENTIFIER_ID);
        let estimator = self.estimator.as_ref().unwrap_or(&DEFAULT_DISTRIBUTION_ESTIMATOR_ID);
        (Arc::from(identifier.as_str()), Arc::from(estimator.as_str()))
    }

    /// Resolve identifier/estimator for PathSpecific queries.
    fn resolve_path_pair(&self) -> (Arc<str>, Arc<str>) {
        let identifier = self.identifier.as_ref().unwrap_or(&DEFAULT_PATH_IDENTIFIER_ID);
        let estimator = self.estimator.as_ref().unwrap_or(&DEFAULT_PATH_ESTIMATOR_ID);
        (Arc::from(identifier.as_str()), Arc::from(estimator.as_str()))
    }

    fn ensure_rd_config_present(&self, estimator: &str) -> Result<(), CausalError> {
        if matches!(EstimatorId::parse(estimator), EstimatorId::RdSharp) && self.rd.is_none() {
            return Err(CausalError::Compile {
                message: "estimator \"rd.sharp\" requires builder.rd_config(running_variable, cutoff, bandwidth)".into(),
            });
        }
        Ok(())
    }

    /// Execute a Ready physical plan.
    ///
    /// # Errors
    ///
    /// Identification / estimation / validation failures.
    pub fn execute(
        &self,
        plan: &CompiledAnalysis,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let CompiledAnalysis::Ready(physical) = plan else {
            let (kind, algorithm, pending, hint) = match plan {
                CompiledAnalysis::ReviewRequired(r) => (
                    "temporal_dag",
                    Some(r.algorithm.to_string()),
                    r.pending_edges.len(),
                    "call finish_review_and_run after accepting pending edges",
                ),
                CompiledAnalysis::ReviewRequiredCpdag(r) => (
                    "temporal_cpdag",
                    Some(r.algorithm.to_string()),
                    r.pending_edges.len() + r.pending_undirected.len(),
                    "orient undirected edges then finish_cpdag_review_and_run",
                ),
                CompiledAnalysis::ReviewRequiredStaticCpdag(r) => (
                    "static_cpdag",
                    Some(r.algorithm.to_string()),
                    r.pending_edges.len() + r.pending_undirected.len(),
                    "orient undirected CPDAG edges or supply a Dag",
                ),
                CompiledAnalysis::ReviewRequiredStaticDag(r) => (
                    "static_dag",
                    Some(r.algorithm.to_string()),
                    r.pending_edges.len(),
                    "accept pending directed edges or supply a fully oriented Dag",
                ),
                CompiledAnalysis::ReviewRequiredStaticPag(r) => (
                    "static_pag",
                    Some(r.algorithm.to_string()),
                    r.pending_circles.len(),
                    "finish_static_pag_review_and_run or supply a completed Pag/Dag",
                ),
                CompiledAnalysis::ReviewRequiredPag(r) => (
                    "temporal_pag",
                    Some(r.algorithm.to_string()),
                    r.pending_circles.len(),
                    "complete TemporalPag to a TemporalDag (no circle/ambiguous marks), \
                     or finish PAG review; temporal backdoor does not run on PAG class ID",
                ),
                CompiledAnalysis::Ready(_) => unreachable!(),
            };
            return Err(CausalError::review_required(
                kind,
                algorithm,
                pending,
                "cannot execute while graph review is required",
                hint,
            ));
        };
        ensure_review_complete(&physical.logical)?;
        match (&self.data, &self.query) {
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q)) => match &self.graph {
                GraphInput::Static(graph) => self.execute_static(data, graph, q, physical, ctx),
                GraphInput::DiscoverPc { .. }
                | GraphInput::DiscoverGes { .. }
                | GraphInput::DiscoverLingam { .. }
                | GraphInput::DiscoverNotears { .. }
                | GraphInput::Cpdag(_) => {
                    let graph = physical.static_graph().ok_or(CausalError::Compile {
                            message:
                                "Ready PC/GES/LiNGAM/NOTEARS/CPDAG plan missing resolved static DAG (complete review first)"
                                    .into(),
                        })?;
                    self.execute_static(data, graph, q, physical, ctx)
                }
                GraphInput::Admg(admg) => {
                    if admg_has_bidirected(admg) {
                        self.execute_admg(data, admg, q, physical, ctx)
                    } else {
                        let graph = physical.static_graph().ok_or(CausalError::Compile {
                            message: "Ready ADMG (DAG-coerced) plan missing resolved static DAG"
                                .into(),
                        })?;
                        self.execute_static(data, graph, q, physical, ctx)
                    }
                }
                GraphInput::Pag(_)
                | GraphInput::DiscoverFci { .. }
                | GraphInput::DiscoverRfci { .. } => {
                    let pag = physical.static_pag().ok_or(CausalError::Compile {
                        message:
                            "Ready PAG plan missing resolved static PAG (complete review first)"
                                .into(),
                    })?;
                    self.execute_pag(data, pag, q, physical, ctx)
                }
                GraphInput::DiscoverExactDagPosterior
                | GraphInput::DiscoverOrderMcmc { .. }
                | GraphInput::DiscoverStructureMcmc { .. }
                | GraphInput::DiscoverCiScreenedPosterior { .. } => {
                    self.execute_graph_posterior_bayesian(data, q, physical, ctx)
                }
                _ => Err(CausalError::Unsupported {
                    message: "static ATE execute requires a supplied static DAG/PAG/CPDAG/ADMG or DiscoverPc/Ges/Lingam/Notears/Fci/Rfci/graph-posterior",
                }),
            },
            (DataInput::Tabular(data), CausalQuery::Distribution(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(CausalError::Unsupported {
                        message: "Distribution execute requires a supplied static DAG",
                    });
                };
                self.execute_distribution(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::PathSpecific(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(CausalError::Unsupported {
                        message: "PathSpecific execute requires a supplied static DAG",
                    });
                };
                self.execute_path_specific(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::ConditionalEffect(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(CausalError::Unsupported {
                        message: "ConditionalEffect execute requires a supplied static DAG",
                    });
                };
                self.execute_conditional(data, graph, q, physical, ctx)
            }
            (DataInput::Temporal(data) | DataInput::Event(data), CausalQuery::Mediation(q)) => {
                let graph = physical.temporal_graph().ok_or(CausalError::Compile {
                    message: "Ready temporal mediation plan missing resolved graph".into(),
                })?;
                self.execute_temporal_mediation(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::Mediation(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(CausalError::Unsupported {
                        message: "static Mediation execute requires a supplied static DAG",
                    });
                };
                self.execute_static_mediation_total(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::Counterfactual(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(CausalError::Unsupported {
                        message: "Counterfactual execute requires a supplied static DAG",
                    });
                };
                self.execute_counterfactual(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::AnomalyAttribution(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(CausalError::Unsupported {
                        message: "AnomalyAttribution execute requires a supplied static DAG",
                    });
                };
                self.execute_anomaly(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::ChangeAttribution(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(CausalError::Unsupported {
                        message: "ChangeAttribution execute requires a supplied static DAG",
                    });
                };
                self.execute_change_attribution(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::MechanismChange(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(CausalError::Unsupported {
                        message: "MechanismChange execute requires a supplied static DAG",
                    });
                };
                self.execute_mechanism_change(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::UnitChange(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(CausalError::Unsupported {
                        message: "UnitChange execute requires a supplied static DAG",
                    });
                };
                self.execute_unit_change(data, graph, q, physical, ctx)
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
            ) => {
                if matches!(self.graph, GraphInput::DiscoverDbnPosterior { .. }) {
                    return self.execute_dbn_posterior_bayesian(data, q, physical, ctx);
                }
                let graph = physical.temporal_graph().ok_or(CausalError::Compile {
                    message: "Ready temporal plan missing resolved graph (complete review first)"
                        .into(),
                })?;
                self.execute_temporal(data, graph, q, physical, ctx)
            }
            (DataInput::Panel(panel), CausalQuery::TemporalEffect(q)) => {
                let graph = physical.temporal_graph().ok_or(CausalError::Compile {
                    message: "Ready panel plan missing resolved graph (complete review first)"
                        .into(),
                })?;
                self.execute_panel(panel, graph, q, physical, ctx)
            }
            _ => Err(CausalError::Unsupported {
                message: "execute path unsupported for this configuration",
            }),
        }
    }

    /// Compile and run when Ready; error if review is required.
    ///
    /// # Errors
    ///
    /// Compile / review / execute failures.
    pub fn run(&self, ctx: &ExecutionContext) -> Result<CausalAnalysisResult, CausalError> {
        let compiled = self.compile(ctx)?;
        self.execute(&compiled, ctx)
    }

    /// Identify only (no estimation). Supports static DAG average-effect / related queries.
    ///
    /// # Errors
    ///
    /// Missing DAG graph, unsupported graph class, or identification failure.
    pub fn identify_only(&self) -> Result<IdentificationResult, CausalError> {
        use crate::strategy_table::{DEFAULT_IDENTIFIER_ID, identify_static_query};

        let GraphInput::Static(graph) = &self.graph else {
            return Err(CausalError::Unsupported {
                message: "identify_only currently supports static DAG graphs only",
            });
        };
        let id = self.identifier.clone().unwrap_or(DEFAULT_IDENTIFIER_ID);
        identify_static_query(id, graph, &self.query)
    }

    /// Inspectable compile result (logical + physical when Ready).
    ///
    /// Prefer this over `compile` when documenting plan inspection in user code.
    ///
    /// # Errors
    ///
    /// Same as [`Self::compile`].
    pub fn plan(&self, ctx: &ExecutionContext) -> Result<CompiledAnalysis, CausalError> {
        self.compile(ctx)
    }

    /// Continue after DAG review: accept all pending edges then execute.
    ///
    /// # Errors
    ///
    /// Review / execute failures.
    pub fn finish_review_and_run(
        &self,
        review: TemporalGraphReview,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let (DataInput::Temporal(data) | DataInput::Event(data)) = &self.data else {
            return Err(CausalError::Compile {
                message: "finish_review_and_run requires temporal data".into(),
            });
        };
        let CausalQuery::TemporalEffect(q) = &self.query else {
            return Err(CausalError::Compile {
                message: "finish_review_and_run requires temporal effect query".into(),
            });
        };
        let compiled = PendingGraphReview::new(review, data.row_count(), q.clone(), self.split)
            .accept_all()
            .finish(data, ctx)?;
        self.execute(&compiled, ctx)
    }

    /// Continue after PCMCI+ CPDAG review once undirected marks are oriented and directed accepted.
    ///
    /// # Errors
    ///
    /// Incomplete CPDAG review (undirected remain) or execute failures.
    pub fn finish_cpdag_review_and_run(
        &self,
        review: TemporalCpdagReview,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let (DataInput::Temporal(data) | DataInput::Event(data)) = &self.data else {
            return Err(CausalError::Compile {
                message: "finish_cpdag_review_and_run requires temporal data".into(),
            });
        };
        let CausalQuery::TemporalEffect(q) = &self.query else {
            return Err(CausalError::Compile {
                message: "finish_cpdag_review_and_run requires temporal effect query".into(),
            });
        };
        let compiled = PendingCpdagReview::new(review, data.row_count(), q.clone(), self.split)
            .finish(data, ctx)?;
        self.execute(&compiled, ctx)
    }

    /// Dispatch identify → estimate for the static ATE path, routing on the plan's resolved
    /// identifier/estimator .
    fn execute_static(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let mut clock = super::stage::StageClock::new();
        let identifier =
            physical.logical.record.identifier.as_deref().unwrap_or(DEFAULT_IDENTIFIER);
        let estimator = physical.logical.record.estimator.as_deref().unwrap_or(DEFAULT_ESTIMATOR);
        let estimator_id = EstimatorId::parse(estimator);

        // rd.sharp has no graph-based identification step; dispatch to its
        // own path before touching `graph`.
        if matches!(estimator_id, EstimatorId::RdSharp) {
            return self.execute_rd(data, query, physical, ctx);
        }
        if matches!(estimator_id, EstimatorId::BayesianGcomp) {
            return self.execute_bayesian(data, graph, query, physical, ctx);
        }

        clock.begin(ctx, super::stage::STAGE_IDENTIFY, 0.05)?;
        let rd = self.rd.map(|c| SharpRdConfig {
            running_variable: c.running_variable,
            cutoff: c.cutoff,
            bandwidth: c.bandwidth,
        });
        let identification = identify_static_query_with_rd(
            identifier,
            graph,
            &CausalQuery::AverageEffect(query.clone()),
            rd,
        )?;
        let estimand = select_estimand(&identification, estimator_id.clone())?;
        let assumptions = identification.required_assumptions.clone();
        clock.finish(super::stage::STAGE_IDENTIFY);
        super::stage::emit_stage(
            self.stage_sink.as_ref(),
            super::stage::AnalysisStageEvent::Identify {
                identification: identification.clone(),
                estimand: estimand.clone(),
            },
        );

        let full_cols = data.schema().len();
        let (data_est, query_est, estimand_est) =
            project_for_ate_estimate(data, query, &estimand)?;
        let projected_cols = data_est.schema().len();

        // Point estimate first (no bootstrap); uncertainty stage fills SE separately.
        clock.begin(ctx, super::stage::STAGE_ESTIMATE_POINT, 0.25)?;
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled { stage: super::stage::STAGE_ESTIMATE_POINT });
        }
        let mut estimate_ws = StaticEstimateWorkspaces::default();
        let point = estimate_static_effect(
            estimator,
            &data_est,
            &estimand_est,
            &query_est,
            assumptions.clone(),
            0, // point stage: no bootstrap
            self.overlap_policy,
            self.population_registry.as_ref(),
            ctx,
            &mut estimate_ws,
        )?;
        clock.finish(super::stage::STAGE_ESTIMATE_POINT);
        super::stage::emit_stage(
            self.stage_sink.as_ref(),
            super::stage::AnalysisStageEvent::Point { estimate: point.clone() },
        );

        // Uncertainty: bootstrap fills (real work when replicates > 0).
        let estimate = if self.bootstrap_replicates == 0 {
            if ctx.cancellation.is_cancelled() {
                clock.mark_cancelled();
                point
            } else {
                clock.begin(ctx, super::stage::STAGE_UNCERTAINTY, 0.55)?;
                clock.finish(super::stage::STAGE_UNCERTAINTY);
                super::stage::emit_stage(
                    self.stage_sink.as_ref(),
                    super::stage::AnalysisStageEvent::Uncertainty { estimate: point.clone() },
                );
                point
            }
        } else if matches!(estimator_id, EstimatorId::LinearAdjustmentAte) {
            // Reuse warmed OLS workspace: re-prepare + attach bootstrap without refitting point.
            let cancelled_before = ctx.cancellation.is_cancelled();
            if cancelled_before {
                clock.mark_cancelled();
                if let Some(p) = &ctx.progress {
                    p.report(0.55, super::stage::STAGE_UNCERTAINTY);
                }
                point
            } else {
                clock.begin(ctx, super::stage::STAGE_UNCERTAINTY, 0.55)?;
                let mut est = LinearAdjustmentAte::new();
                est.bootstrap_replicates = self.bootstrap_replicates;
                est.overlap = OverlapPolicy::ExplicitOverride;
                let prep = est.prepare(&data_est, &estimand_est, &query_est).map_err(|e| {
                    CausalError::from(e)
                })?;
                let filled = est
                    .attach_bootstrap(&prep, &mut estimate_ws.linear, ctx, point)
                    .map_err(CausalError::from)?;
                let cancelled = filled.bootstrap_cancelled || ctx.cancellation.is_cancelled();
                if cancelled {
                    clock.mark_cancelled();
                } else {
                    clock.finish(super::stage::STAGE_UNCERTAINTY);
                }
                super::stage::emit_stage(
                    self.stage_sink.as_ref(),
                    super::stage::AnalysisStageEvent::Uncertainty { estimate: filled.clone() },
                );
                filled
            }
        } else {
            // Non-linear static estimators: re-run with bootstrap for uncertainty fills.
            let cancelled_before = ctx.cancellation.is_cancelled();
            if cancelled_before {
                clock.mark_cancelled();
                if let Some(p) = &ctx.progress {
                    p.report(0.55, super::stage::STAGE_UNCERTAINTY);
                }
                point
            } else {
                clock.begin(ctx, super::stage::STAGE_UNCERTAINTY, 0.55)?;
                let filled = estimate_static_effect(
                    estimator,
                    &data_est,
                    &estimand_est,
                    &query_est,
                    assumptions,
                    self.bootstrap_replicates,
                    self.overlap_policy,
                    self.population_registry.as_ref(),
                    ctx,
                    &mut estimate_ws,
                )?;
                let cancelled = filled.bootstrap_cancelled || ctx.cancellation.is_cancelled();
                if cancelled {
                    clock.mark_cancelled();
                } else {
                    clock.finish(super::stage::STAGE_UNCERTAINTY);
                }
                super::stage::emit_stage(
                    self.stage_sink.as_ref(),
                    super::stage::AnalysisStageEvent::Uncertainty { estimate: filled.clone() },
                );
                filled
            }
        };

        let cancelled = estimate.bootstrap_cancelled || clock.cancelled();

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        if let Some(d) = projection_diagnostic(full_cols, projected_cols) {
            diagnostics.push(d);
        }

        let refutations = if cancelled {
            Vec::new()
        } else {
            clock.begin(ctx, super::stage::STAGE_VALIDATE, 0.8)?;
            let prop_scratch = match estimator_id {
                EstimatorId::Aipw => &mut estimate_ws.aipw.propensity,
                _ => &mut estimate_ws.propensity.propensity,
            };
            let reports = run_refuters(
                &data_est,
                &estimand_est,
                &query_est,
                &estimate,
                &mut estimate_ws.linear,
                Some(prop_scratch),
                ctx,
                self.refute,
                estimator,
                &self.custom_validators,
                None,
            )?;
            clock.finish(super::stage::STAGE_VALIDATE);
            super::stage::emit_stage(
                self.stage_sink.as_ref(),
                super::stage::AnalysisStageEvent::Validate {
                    refutations: reports.clone(),
                    predictive_checks: Vec::new(),
                },
            );
            reports
        };

        let (id_artifact, id_op) = identify_provenance_step(identifier);
        let (est_artifact, est_op) = estimate_provenance_step(estimator);
        let provenance = provenance_pair(
            (id_artifact, id_op, &[], &identification.required_assumptions),
            (est_artifact, est_op, &[id_artifact], &estimate.assumptions),
        );

        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        let bootstrap_ok = estimate.bootstrap_replicates_ok;
        let early_stopped = estimate.bootstrap_early_stopped;
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior: None,
            mediation: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
            wall_time_ns: clock.wall_time_ns(),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: clock.timings(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: bootstrap_ok,
            n_draws: None,
            cancelled: clock.cancelled(),
            early_stopped,
        }))
    }

    /// Identify + plug-in estimate for an interventional distribution.
    fn execute_distribution(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &causal_core::InterventionalDistributionQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let started = Instant::now();
        let identifier = physical
            .logical
            .record
            .identifier
            .as_deref()
            .unwrap_or(DEFAULT_DISTRIBUTION_IDENTIFIER);
        let estimator =
            physical.logical.record.estimator.as_deref().unwrap_or(DEFAULT_DISTRIBUTION_ESTIMATOR);
        if !matches!(EstimatorId::parse(estimator), EstimatorId::FunctionalDistribution) {
            return Err(CausalError::Compile {
                message: format!(
                    "Distribution execute requires estimator functional.distribution; got {estimator}"
                ),
            });
        }

        let cq = CausalQuery::Distribution(query.clone());
        let identification = identify_static_query(identifier, graph, &cq)?;
        let estimand = select_estimand(&identification, EstimatorId::parse(estimator))?;

        let est = FunctionalDistribution {
            bootstrap_replicates: self.bootstrap_replicates,
            ..FunctionalDistribution::new()
        };
        let prepared = est
            .prepare(
                data,
                query,
                &estimand,
                &identification.arena,
                identification.required_assumptions.clone(),
            )
            .map_err(CausalError::from)?;
        let mut ws = FunctionalDistributionWorkspace::default();
        let dist = est.estimate(&prepared, &[], &mut ws, ctx).map_err(CausalError::from)?;

        let estimate = EffectEstimate {
            ate: dist.mean,
            se_analytic: dist.se_analytic,
            se_bootstrap: dist.se_bootstrap,
            bootstrap_replicates_ok: dist.bootstrap_replicates_ok,
            bootstrap_replicates_failed: dist.bootstrap_replicates_failed,
            bootstrap_cancelled: dist.bootstrap_cancelled,
            bootstrap_early_stopped: dist.bootstrap_early_stopped,
            assumptions: dist.assumptions.clone(),
            overlap: dist.overlap,
            overlap_report: None,
            retained_memory_bytes: dist.retained_memory_bytes,
        };

        let treatment =
            query.interventions.first().and_then(Intervention::primary_variable).ok_or_else(
                || CausalError::Compile {
                    message: "distribution query missing intervention target".into(),
                },
            )?;
        let outcome = *query.outcomes.first().ok_or_else(|| CausalError::Compile {
            message: "distribution query missing outcome".into(),
        })?;
        let early_stopped = estimate.bootstrap_early_stopped;
        let bootstrap_ok = estimate.bootstrap_replicates_ok;
        let cancelled = estimate.bootstrap_cancelled;

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));

        let mut refute_ws = EstimationWorkspace::default();
        let ate_q = AverageEffectQuery::binary_ate(treatment, outcome);
        let refutations = if estimate.ate.is_finite() {
            run_refuters(
                data,
                &estimand,
                &ate_q,
                &estimate,
                &mut refute_ws,
                None,
                ctx,
                self.refute,
                estimator,
                &self.custom_validators,
                None,
            )?
        } else {
            diagnostics.push(Diagnostic::new(
                "refute.distribution.skipped",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "effect refuters skipped: interventional mean is not a finite scalar",
            ));
            Vec::new()
        };

        let (id_artifact, id_op) = identify_provenance_step(identifier);
        let (est_artifact, est_op) = estimate_provenance_step(estimator);
        let provenance = provenance_pair(
            (id_artifact, id_op, &[], &identification.required_assumptions),
            (est_artifact, est_op, &[id_artifact], &estimate.assumptions),
        );

        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: Some(dist),
            posterior: None,
            mediation: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment,
            outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: bootstrap_ok,
            n_draws: None,
            cancelled,
            early_stopped,
        }))
    }

    /// Identify + plug-in estimate for a path-specific natural effect.
    fn execute_path_specific(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &causal_core::PathSpecificEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let started = Instant::now();
        let identifier =
            physical.logical.record.identifier.as_deref().unwrap_or(DEFAULT_PATH_IDENTIFIER);
        let estimator =
            physical.logical.record.estimator.as_deref().unwrap_or(DEFAULT_PATH_ESTIMATOR);
        if !matches!(EstimatorId::parse(estimator), EstimatorId::FunctionalEffect) {
            return Err(CausalError::Compile {
                message: format!(
                    "PathSpecific execute requires estimator functional.effect; got {estimator}"
                ),
            });
        }

        let cq = CausalQuery::PathSpecific(query.clone());
        let identification = identify_static_query(identifier, graph, &cq)?;
        let estimand = select_estimand(&identification, EstimatorId::parse(estimator))?;

        let mut extra = vec![query.treatment, query.outcome];
        extra.extend(query.path_nodes.iter().copied());
        let est = FunctionalEffect {
            bootstrap_replicates: self.bootstrap_replicates,
            ..FunctionalEffect::new()
        };
        let prepared = est
            .prepare(
                data,
                &estimand,
                &identification.arena,
                identification.required_assumptions.clone(),
                &extra,
            )
            .map_err(CausalError::from)?;
        let mut ws = FunctionalDistributionWorkspace::default();
        let estimate = est.estimate(&prepared, &mut ws, ctx).map_err(CausalError::from)?;

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));

        let mut refute_ws = EstimationWorkspace::default();
        let ate_q = AverageEffectQuery::binary_ate(query.treatment, query.outcome);
        let refutations = run_refuters(
            data,
            &estimand,
            &ate_q,
            &estimate,
            &mut refute_ws,
            None,
            ctx,
            self.refute,
            estimator,
            &self.custom_validators,
            None,
        )?;

        let (id_artifact, id_op) = identify_provenance_step(identifier);
        let (est_artifact, est_op) = estimate_provenance_step(estimator);
        let provenance = provenance_pair(
            (id_artifact, id_op, &[], &identification.required_assumptions),
            (est_artifact, est_op, &[id_artifact], &estimate.assumptions),
        );

        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior: None,
            mediation: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
        }))
    }

    /// Bayesian g-computation execute path.
    fn execute_bayesian(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let mut clock = super::stage::StageClock::new();
        let identifier =
            physical.logical.record.identifier.as_deref().unwrap_or(DEFAULT_IDENTIFIER);
        clock.begin(ctx, super::stage::STAGE_IDENTIFY, 0.05)?;
        let identification = identify_static(identifier, graph, query)?;
        let estimand = select_estimand(&identification, EstimatorId::BayesianGcomp)?;
        clock.finish(super::stage::STAGE_IDENTIFY);
        super::stage::emit_stage(
            self.stage_sink.as_ref(),
            super::stage::AnalysisStageEvent::Identify {
                identification: identification.clone(),
                estimand: estimand.clone(),
            },
        );

        let full_cols = data.schema().len();
        let (data_est, query_est, estimand_est) =
            project_for_ate_estimate(data, query, &estimand)?;
        let projected_cols = data_est.schema().len();

        let cfg = match &self.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => BayesianConfig::laplace(),
        };
        let mut est = BayesianGComputationAte {
            backend: cfg.backend,
            likelihood: cfg.likelihood,
            n_draws: cfg.n_draws,
            seed: ctx.rng.master_seed(),
            overlap: OverlapPolicy::ExplicitOverride,
            prior_scale: cfg.prior_scale,
            prior: None,
        };
        clock.begin(ctx, super::stage::STAGE_ESTIMATE_POINT, 0.25)?;
        let prep = est.prepare(&data_est, &estimand_est, &query_est).map_err(CausalError::from)?;
        let (resolved_prior, conflict_summary) =
            resolve_bayesian_prior_with_conflict(&cfg, &prep, Some(ctx))?;
        est.prior = resolved_prior;
        let mut ws = BayesianGCompWorkspace::default();
        let mut posterior =
            est.fit(&prep, identification.status, &mut ws, ctx).map_err(CausalError::from)?;
        if let Some(summary) = conflict_summary {
            posterior = with_conflict_summary(posterior, summary);
        }
        let estimate = effect_from_posterior(&posterior)?;
        clock.finish(super::stage::STAGE_ESTIMATE_POINT);
        super::stage::emit_stage(
            self.stage_sink.as_ref(),
            super::stage::AnalysisStageEvent::Point { estimate: estimate.clone() },
        );
        clock.begin(ctx, super::stage::STAGE_UNCERTAINTY, 0.55)?;
        clock.finish(super::stage::STAGE_UNCERTAINTY);
        super::stage::emit_stage(
            self.stage_sink.as_ref(),
            super::stage::AnalysisStageEvent::Uncertainty { estimate: estimate.clone() },
        );

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        if let Some(d) = projection_diagnostic(full_cols, projected_cols) {
            diagnostics.push(d);
        }
        if let Some(cs) = posterior.conflict_summary.as_ref() {
            push_conflict_diagnostics(&mut diagnostics, cs);
        }

        clock.begin(ctx, super::stage::STAGE_VALIDATE, 0.8)?;
        let mut refute_ws = EstimationWorkspace::default();
        let mut refutations = match self.refute {
            RefuteSuite::None => Vec::new(),
            RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => run_refuters(
                &data_est,
                &estimand_est,
                &query_est,
                &estimate,
                &mut refute_ws,
                None,
                ctx,
                self.refute,
                "bayesian.gcomp",
                &self.custom_validators,
                None,
            )?,
        };
        // Prior + posterior PPC whenever refute is enabled (full PredictiveCheckReport retained).
        let mut predictive_checks = Vec::new();
        if !matches!(self.refute, RefuteSuite::None) {
            const PPC_ALPHA: f64 = 0.05;
            let ppc_prior = est
                .prior
                .clone()
                .unwrap_or_else(|| PriorSet::weakly_informative(prep.design.ncols));
            let prior_rep = PriorPredictiveCheck {
                n_sims: 200,
                seed: ctx.rng.master_seed(),
                ..PriorPredictiveCheck::new()
            }
            .check_with_prior(&prep, &ppc_prior, ctx)
            .map_err(CausalError::from)?;
            refutations.push(prior_rep.to_refutation_report(estimate.ate, PPC_ALPHA));
            predictive_checks.push(prior_rep);

            let post_rep = PosteriorPredictiveCheck::new()
                .check(&prep, &posterior)
                .map_err(CausalError::from)?;
            refutations.push(post_rep.to_refutation_report(estimate.ate, PPC_ALPHA));
            predictive_checks.push(post_rep);
        }
        // Prior sensitivity / MCMC stay behind the full suite (Shared Bayesian UX).
        // Mode-select: α-multiplier grid when an external composed prior is present;
        // isotropic scale grid otherwise (avoids clearing banked priors).
        if matches!(self.refute, RefuteSuite::Full) {
            let (summary, sens) = if let Some(ext) = cfg.external_compose.as_ref() {
                let alphas_applied: Arc<[f64]> = posterior.conflict_summary.as_ref().map_or_else(
                    || Arc::clone(&ext.composed.alphas_applied),
                    |cs| Arc::clone(&cs.alphas_applied),
                );
                let sens = PriorSensitivity::standard_alpha_grid();
                let (summary, _) = sens
                    .evaluate_external_alpha(
                        &est,
                        &prep,
                        identification.status,
                        &mut ws,
                        ctx,
                        ExternalAlphaSensitivity {
                            sources: &ext.sources,
                            alphas_applied: &alphas_applied,
                        },
                    )
                    .map_err(CausalError::from)?;
                (summary, sens)
            } else {
                let sens = PriorSensitivity::standard_grid();
                let (summary, _) = sens
                    .evaluate(&est, &prep, identification.status, &mut ws, ctx)
                    .map_err(CausalError::from)?;
                (summary, sens)
            };
            refutations.push(sens.to_report(&summary, estimate.ate));
            posterior = with_prior_sensitivity(posterior, summary);

            let suite = ValidationSuite::new().with(ValidatorId::McmcDiagnostics);
            let mut bayes_ctx = BayesianSuiteContext::new(
                &est,
                &prep,
                &posterior,
                identification.status,
                &mut ws,
                estimate.ate,
            );
            let outcomes = suite.run_bayesian(&mut bayes_ctx, ctx).map_err(CausalError::from)?;
            refutations.extend(ValidationSuite::reports_only(&outcomes));
        }
        clock.finish(super::stage::STAGE_VALIDATE);
        super::stage::emit_stage(
            self.stage_sink.as_ref(),
            super::stage::AnalysisStageEvent::Validate {
                refutations: refutations.clone(),
                predictive_checks: predictive_checks.clone(),
            },
        );

        let (id_artifact, id_op) = identify_provenance_step(identifier);
        let provenance = provenance_pair(
            (id_artifact, id_op, &[], &identification.required_assumptions),
            (
                "estimate.bayesian_gcomp",
                "estimate.bayesian_gcomp",
                &[id_artifact],
                &estimate.assumptions,
            ),
        );

        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        let n_draws = u32::try_from(posterior.draws.n_draws).ok();
        let early_stopped = posterior.early_stopped;
        let mut result = assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior: Some(posterior),
            mediation: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
            wall_time_ns: clock.wall_time_ns(),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: clock.timings(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: None,
            n_draws,
            cancelled: clock.cancelled(),
            early_stopped,
        });
        result.predictive_checks = predictive_checks;
        Ok(result)
    }

    /// `rd.sharp` execute path: identify via [`SharpRdIdentifier`], then estimate.
    fn execute_rd(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let started = Instant::now();
        let rd = self.rd.ok_or_else(|| CausalError::Compile {
            message: "estimator \"rd.sharp\" requires builder.rd_config(running_variable, cutoff, bandwidth)".into(),
        })?;
        let identification = SharpRdIdentifier::new(SharpRdConfig {
            running_variable: rd.running_variable,
            cutoff: rd.cutoff,
            bandwidth: rd.bandwidth,
        })
        .identify(CausalQuery::AverageEffect(query.clone()))
        .map_err(CausalError::from)?;
        require_identified(&identification)?;
        let estimand = select_estimand(&identification, EstimatorId::RdSharp)?;

        let mut est =
            SharpRegressionDiscontinuity::new(rd.running_variable, rd.cutoff, rd.bandwidth);
        est.bootstrap_replicates = self.bootstrap_replicates;
        let prep = est.prepare(data, &estimand, query).map_err(CausalError::from)?;
        let mut ws = RdWorkspace::default();
        let estimate = est
            .fit(&prep, &mut ws, ctx, identification.required_assumptions.clone())
            .map_err(CausalError::from)?;

        let mut diagnostics = vec![overlap_diagnostic(estimate.overlap)];
        let mut refute_ws = EstimationWorkspace::default();
        let refutations = run_refuters(
            data,
            &estimand,
            query,
            &estimate,
            &mut refute_ws,
            None,
            ctx,
            self.refute,
            "rd.sharp",
            &self.custom_validators,
            None,
        )?;

        let provenance = provenance_pair(
            ("identify.rd_design", "identify.rd_sharp", &[], &identification.required_assumptions),
            ("estimate.rd", "estimate.rd_sharp", &["identify.rd_design"], &estimate.assumptions),
        );

        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior: None,
            mediation: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
        }))
    }

    fn compile_dbn_posterior_temporal(
        &self,
        data: &TimeSeriesData,
        query: &TemporalEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<CompiledAnalysis, CausalError> {
        if matches!(self.inference, InferenceMode::Frequentist) {
            return Err(CausalError::Unsupported {
                message: "DBN graph-posterior discovery requires inference=Bayesian for effect mixture",
            });
        }
        let class = match &self.data {
            DataInput::Event(_) => DataClassification::Event,
            _ => DataClassification::Temporal,
        };
        let mut logical = compile_logical_temporal_effect_classified(
            data,
            &TemporalDag::empty(),
            query,
            self.split,
            false,
            class,
        )?;
        logical.record.discovery_algorithm = Some(Arc::from("dbn_posterior"));
        let physical = logical.compile_physical_with_graphs(ctx, None, None)?;
        Ok(CompiledAnalysis::Ready(physical))
    }

    fn execute_dbn_posterior_bayesian(
        &self,
        data: &TimeSeriesData,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let started = Instant::now();
        let cfg = match &self.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => {
                return Err(CausalError::Unsupported {
                    message: "DBN graph-posterior discovery requires inference=Bayesian for effect mixture",
                });
            }
        };
        let (max_lag, force_mcmc, n_chains, n_warmup, n_draws) = match &self.graph {
            GraphInput::DiscoverDbnPosterior {
                max_lag,
                force_mcmc,
                n_chains,
                n_warmup,
                n_draws,
            } => (*max_lag, *force_mcmc, *n_chains, *n_warmup, *n_draws),
            _ => {
                return Err(CausalError::Compile {
                    message: "execute_dbn_posterior_bayesian: unexpected GraphInput".into(),
                });
            }
        };
        let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
        let params = BayesianDiscoverParams::default();
        let schedule = GraphMcmcSchedule { n_chains, n_warmup, n_draws, thin: 1 };
        let gp = discover_dbn_posterior(data, &vars, &params, max_lag, force_mcmc, &schedule, ctx)?;
        let lag_masks = gp.lag_masks.as_ref().ok_or_else(|| CausalError::Compile {
            message: "DBN posterior missing per-atom lag masks".into(),
        })?;
        let max_lag = gp.max_lag.unwrap_or(max_lag);

        let est_inner = BayesianGComputationAte {
            backend: cfg.backend,
            likelihood: cfg.likelihood,
            n_draws: cfg.n_draws,
            seed: ctx.rng.master_seed(),
            overlap: OverlapPolicy::ExplicitOverride,
            prior_scale: cfg.prior_scale,
            prior: None,
        };
        let mut bayes = BayesianTemporalGcomp { inner: est_inner };

        let mut weights = Vec::with_capacity(gp.n_graphs);
        let mut flags = Vec::with_capacity(gp.n_graphs);
        let mut keys = Vec::with_capacity(gp.n_graphs);
        let mut per_graph = Vec::new();
        let mut primary_estimand: Option<IdentifiedEstimand> = None;
        let mut primary_identification: Option<IdentificationResult> = None;
        let mut envelope_prior: Option<PriorSet> = None;
        let mut envelope_conflict: Option<causal_prob::ConflictSummary> = None;

        for i in 0..gp.n_graphs {
            if ctx.cancellation.is_cancelled() {
                for j in i..gp.n_graphs {
                    keys.push(gp.graph_keys[j]);
                    weights.push(gp.weights[j]);
                    flags.push(GraphIdentFlag::Unidentified);
                }
                break;
            }
            if let Some(p) = &ctx.progress {
                #[allow(clippy::cast_precision_loss)]
                p.report(i as f64 / gp.n_graphs.max(1) as f64, "envelope");
            }
            let cmask = gp.adjacency[i];
            let lmask = lag_masks[i];
            let key = gp.graph_keys[i];
            keys.push(key);
            weights.push(gp.weights[i]);
            let Ok(tdag) = temporal_dag_from_dbn_masks(cmask, lmask, gp.n_vars, max_lag, &vars)
            else {
                flags.push(GraphIdentFlag::Unidentified);
                continue;
            };
            let Ok(id_res) = TemporalBackdoorIdentifier::new().identify_temporal(&tdag, query)
            else {
                flags.push(GraphIdentFlag::Unidentified);
                continue;
            };
            let identification = id_res.result;
            if !identification_status_ok_for_case(identification.status)
                || identification.estimands.is_empty()
            {
                flags.push(GraphIdentFlag::Unidentified);
                continue;
            }
            let Ok(estimand) =
                select_estimand(&identification, EstimatorId::TemporalLinearAdjustment)
            else {
                flags.push(GraphIdentFlag::Unidentified);
                continue;
            };
            flags.push(GraphIdentFlag::Identified);
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand.clone());
                primary_identification = Some(identification.clone());
            }
            let mut temporal_est = TemporalLinearAdjustment::new();
            temporal_est.inner.overlap = OverlapPolicy::ExplicitOverride;
            let Ok(prep) = temporal_est.prepare(
                data,
                &estimand,
                query,
                &id_res.indexer,
                self.split.as_ref(),
                &ctx.kernel_policy,
            ) else {
                // Already pushed Identified — fix by continuing without draws.
                // Re-mark last flag as Unidentified.
                if let Some(f) = flags.last_mut() {
                    *f = GraphIdentFlag::Unidentified;
                }
                continue;
            };
            let bprep = BayesianGComputationAte::from_prepared_estimation(&prep);
            if envelope_prior.is_none() {
                let (resolved, conflict) =
                    resolve_bayesian_prior_with_conflict(&cfg, &bprep, Some(ctx))?;
                envelope_prior = resolved;
                envelope_conflict = conflict;
            }
            bayes.inner.prior.clone_from(&envelope_prior);
            let mut ws = BayesianGCompWorkspace::default();
            let Ok(posterior) = bayes.fit(&bprep, identification.status, &mut ws, ctx) else {
                if let Some(f) = flags.last_mut() {
                    *f = GraphIdentFlag::Unidentified;
                }
                continue;
            };
            let Some(col) = posterior.effect_column() else {
                if let Some(f) = flags.last_mut() {
                    *f = GraphIdentFlag::Unidentified;
                }
                continue;
            };
            let Ok(d) = posterior.draws.column(col) else {
                if let Some(f) = flags.last_mut() {
                    *f = GraphIdentFlag::Unidentified;
                }
                continue;
            };
            let draws = d.to_vec();
            per_graph.push(GraphEffectDraws { graph_key: key, effect_draws: Arc::from(draws) });
        }

        let graphs = WeightedGraphSamples::new(weights, flags, keys)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let mut subsample_notes = Vec::new();
        let (graphs, per_graph) = maybe_interactive_envelope_subsample(
            self.latency_mode,
            graphs,
            per_graph,
            ctx,
            &mut subsample_notes,
        )?;
        let mut posterior = aggregate_effect_envelope(
            &graphs,
            &per_graph,
            InferenceDiagnostics::analytic("dbn_posterior_envelope"),
            EnvelopeOptions::default(),
        )
        .map_err(CausalError::from)?;
        if let Some(summary) = envelope_conflict {
            posterior = with_conflict_summary(posterior, summary);
        }
        let estimate = effect_from_posterior(&posterior)?;
        let identification = primary_identification.ok_or_else(|| CausalError::Compile {
            message: "DBN posterior envelope: no identified graph atoms".into(),
        })?;
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "DBN posterior envelope: missing estimand".into(),
        })?;

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.extend(subsample_notes);
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        diagnostics.push(Diagnostic::new(
            "estimate.dbn_posterior.envelope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!("unidentified_mass={}", posterior.unidentified_mass),
        ));
        if let Some(cs) = posterior.conflict_summary.as_ref() {
            push_conflict_diagnostics(&mut diagnostics, cs);
        }

        let provenance = provenance_pair(
            ("discover.dbn_posterior", "dbn_posterior", &[], &identification.required_assumptions),
            (
                "estimate.aggregate_effect_envelope",
                "estimate.bayesian.temporal.gcomp",
                &["discover.dbn_posterior"],
                &estimate.assumptions,
            ),
        );
        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior: Some(posterior),
            mediation: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations: Vec::new(),
            diagnostics,
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
        }))
    }

    fn execute_temporal(
        &self,
        data: &TimeSeriesData,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let started = Instant::now();
        let id_res = TemporalBackdoorIdentifier::new()
            .identify_temporal(graph, query)
            .map_err(CausalError::from)?;
        let identification = id_res.result;
        require_identified(&identification)?;
        let estimand = select_estimand(&identification, EstimatorId::TemporalLinearAdjustment)?;

        let mut estimator = TemporalLinearAdjustment::new();
        estimator.inner.bootstrap_replicates = self.bootstrap_replicates;
        estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
        let prep = estimator
            .prepare(
                data,
                &estimand,
                query,
                &id_res.indexer,
                self.split.as_ref(),
                &ctx.kernel_policy,
            )
            .map_err(CausalError::from)?;

        let (estimate, posterior, estimate_artifact, estimate_op) = match &self.inference {
            InferenceMode::Bayesian(cfg) => {
                let mut bayes = BayesianTemporalGcomp {
                    inner: BayesianGComputationAte {
                        backend: cfg.backend,
                        likelihood: cfg.likelihood,
                        n_draws: cfg.n_draws,
                        seed: ctx.rng.master_seed(),
                        overlap: OverlapPolicy::ExplicitOverride,
                        prior_scale: cfg.prior_scale,
                        prior: None,
                    },
                };
                let bprep = BayesianGComputationAte::from_prepared_estimation(&prep);
                let (resolved_prior, conflict_summary) =
                    resolve_bayesian_prior_with_conflict(cfg, &bprep, Some(ctx))?;
                bayes.inner.prior = resolved_prior;
                let mut ws = BayesianGCompWorkspace::default();
                let mut posterior = bayes
                    .fit(&bprep, identification.status, &mut ws, ctx)
                    .map_err(CausalError::from)?;
                if let Some(summary) = conflict_summary {
                    posterior = with_conflict_summary(posterior, summary);
                }
                let estimate = effect_from_posterior(&posterior)?;
                (
                    estimate,
                    Some(posterior),
                    "estimate.bayesian_temporal_gcomp",
                    "estimate.bayesian.temporal.gcomp",
                )
            }
            InferenceMode::Frequentist => {
                let mut workspace = EstimationWorkspace::default();
                let estimate = estimator
                    .fit(&prep, &mut workspace, ctx, identification.required_assumptions.clone())
                    .map_err(CausalError::from)?;
                (
                    estimate,
                    None,
                    "estimate.temporal_linear_adjustment",
                    "estimate.temporal.linear.adjustment",
                )
            }
        };

        let provenance = provenance_pair(
            (
                "identify.temporal_backdoor",
                "identify.temporal.backdoor.unfolded",
                &[],
                &identification.required_assumptions,
            ),
            (
                estimate_artifact,
                estimate_op,
                &["identify.temporal_backdoor"],
                &estimate.assumptions,
            ),
        );

        let mut diagnostics = Vec::new();
        if physical
            .logical
            .record
            .discovery_algorithm
            .as_deref()
            .is_some_and(|a| a.contains("pag_completed_to_dag") || a.contains("completed_to_dag"))
        {
            diagnostics.push(Diagnostic::new(
                "temporal.pag.completed_to_dag",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "TemporalPag completed to TemporalDag before temporal.backdoor \
                 (completion path; not class-aware temporal PAG identification)",
            ));
        }
        let tabular = TabularData::new(data.storage().clone());
        let ate_q = AverageEffectQuery::binary_ate(query.treatment, query.outcome);
        let mut refute_ws = EstimationWorkspace::default();
        let temporal_ctx = TemporalRefitContext {
            indexer: &id_res.indexer,
            temporal_query: query,
            split: self.split.as_ref(),
            kernel_policy: &ctx.kernel_policy,
            time_index: Some(data.time_index()),
            panel: None,
        };
        let mut refutations = run_refuters(
            &tabular,
            &estimand,
            &ate_q,
            &estimate,
            &mut refute_ws,
            None,
            ctx,
            self.refute,
            if posterior.is_some() {
                "bayesian.temporal.gcomp"
            } else {
                "temporal.linear.adjustment"
            },
            &self.custom_validators,
            Some(temporal_ctx),
        )?;

        // Bayesian temporal: prior/posterior PPC + prior sensitivity on Full (mirror static).
        let mut posterior = posterior;
        if matches!(&self.inference, InferenceMode::Bayesian(_))
            && !matches!(self.refute, RefuteSuite::None)
        {
            if let Some(ref post) = posterior {
                const PPC_ALPHA: f64 = 0.05;
                let bprep = BayesianGComputationAte::from_prepared_estimation(&prep);
                let prior_rep = PriorPredictiveCheck {
                    n_sims: 200,
                    seed: ctx.rng.master_seed(),
                    ..PriorPredictiveCheck::new()
                }
                .check(&bprep, ctx)
                .map_err(CausalError::from)?;
                refutations.push(prior_rep.to_refutation_report(estimate.ate, PPC_ALPHA));

                let post_rep = PosteriorPredictiveCheck::new()
                    .check(&bprep, post)
                    .map_err(CausalError::from)?;
                refutations.push(post_rep.to_refutation_report(estimate.ate, PPC_ALPHA));

                if matches!(self.refute, RefuteSuite::Full) {
                    let cfg = match &self.inference {
                        InferenceMode::Bayesian(c) => c.clone(),
                        InferenceMode::Frequentist => unreachable!(),
                    };
                    let mut est = BayesianTemporalGcomp {
                        inner: BayesianGComputationAte {
                            backend: cfg.backend,
                            likelihood: cfg.likelihood,
                            n_draws: cfg.n_draws,
                            seed: ctx.rng.master_seed(),
                            overlap: OverlapPolicy::ExplicitOverride,
                            prior_scale: cfg.prior_scale,
                            prior: None,
                        },
                    };
                    let mut ws = BayesianGCompWorkspace::default();
                    let (summary, sens) = if let Some(ext) = cfg.external_compose.as_ref() {
                        let alphas_applied: Arc<[f64]> =
                            post.conflict_summary.as_ref().map_or_else(
                                || Arc::clone(&ext.composed.alphas_applied),
                                |cs| Arc::clone(&cs.alphas_applied),
                            );
                        est.inner.prior = Some(ext.composed.prior.clone());
                        let sens = PriorSensitivity::standard_alpha_grid();
                        let (summary, _) = sens
                            .evaluate_external_alpha(
                                &est.inner,
                                &bprep,
                                identification.status,
                                &mut ws,
                                ctx,
                                ExternalAlphaSensitivity {
                                    sources: &ext.sources,
                                    alphas_applied: &alphas_applied,
                                },
                            )
                            .map_err(CausalError::from)?;
                        (summary, sens)
                    } else {
                        est.inner.prior = resolve_bayesian_prior(&cfg, &bprep)?;
                        let sens = PriorSensitivity::standard_grid();
                        let (summary, _) = sens
                            .evaluate(&est.inner, &bprep, identification.status, &mut ws, ctx)
                            .map_err(CausalError::from)?;
                        (summary, sens)
                    };
                    refutations.push(sens.to_report(&summary, estimate.ate));
                    posterior = Some(with_prior_sensitivity(post.clone(), summary));
                }
            }
        }

        if let Some(cs) = posterior.as_ref().and_then(|p| p.conflict_summary.as_ref()) {
            push_conflict_diagnostics(&mut diagnostics, cs);
        }

        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior,
            mediation: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
        }))
    }

    /// Panel temporal effect: identify on the shared graph, estimate on stacked units
    /// with [`AnalyticSeKind::PanelClusterHac`] and per-unit `cluster_ids`.
    ///
    /// Bayesian mode fits [`BayesianTemporalGcomp`] on the stacked lag-aligned design
    /// (no hierarchical unit random effects; cluster-HAC is frequentist-only).
    fn execute_panel(
        &self,
        panel: &PanelData,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let started = Instant::now();
        let id_res = TemporalBackdoorIdentifier::new()
            .identify_temporal(graph, query)
            .map_err(CausalError::from)?;
        let identification = id_res.result;
        require_identified(&identification)?;
        let estimand = select_estimand(&identification, EstimatorId::TemporalLinearAdjustment)?;

        let mut estimator = TemporalLinearAdjustment::new();
        estimator.inner.bootstrap_replicates = self.bootstrap_replicates;
        estimator.inner.overlap = OverlapPolicy::ExplicitOverride;
        let (prep, cluster_ids) = estimator
            .prepare_panel(
                panel,
                &estimand,
                query,
                &id_res.indexer,
                self.split.as_ref(),
                &ctx.kernel_policy,
            )
            .map_err(CausalError::from)?;
        let max_lag = query.max_history_lag.unwrap_or(1).max(1) as usize;

        let (estimate, mut posterior, estimate_artifact, estimate_op) = match &self.inference {
            InferenceMode::Bayesian(cfg) => {
                let mut bayes = BayesianTemporalGcomp {
                    inner: BayesianGComputationAte {
                        backend: cfg.backend,
                        likelihood: cfg.likelihood,
                        n_draws: cfg.n_draws,
                        seed: ctx.rng.master_seed(),
                        overlap: OverlapPolicy::ExplicitOverride,
                        prior_scale: cfg.prior_scale,
                        prior: None,
                    },
                };
                let bprep = BayesianGComputationAte::from_prepared_estimation(&prep);
                let (resolved_prior, conflict_summary) =
                    resolve_bayesian_prior_with_conflict(cfg, &bprep, Some(ctx))?;
                bayes.inner.prior = resolved_prior;
                let mut ws = BayesianGCompWorkspace::default();
                let mut posterior = bayes
                    .fit(&bprep, identification.status, &mut ws, ctx)
                    .map_err(CausalError::from)?;
                if let Some(summary) = conflict_summary {
                    posterior = with_conflict_summary(posterior, summary);
                }
                let estimate = effect_from_posterior(&posterior)?;
                (
                    estimate,
                    Some(posterior),
                    "estimate.bayesian_temporal_gcomp.panel",
                    "estimate.bayesian.temporal.gcomp.panel",
                )
            }
            InferenceMode::Frequentist => {
                estimator.inner.cluster_ids = Some(cluster_ids);
                estimator.inner.se_kind = AnalyticSeKind::PanelClusterHac { lag: max_lag };
                let mut workspace = EstimationWorkspace::default();
                let estimate = estimator
                    .fit(&prep, &mut workspace, ctx, identification.required_assumptions.clone())
                    .map_err(CausalError::from)?;
                (
                    estimate,
                    None,
                    "estimate.temporal_linear_adjustment.panel",
                    "estimate.temporal.linear.adjustment.panel",
                )
            }
        };

        let provenance = provenance_pair(
            (
                "identify.temporal_backdoor",
                "identify.temporal.backdoor.unfolded",
                &[],
                &identification.required_assumptions,
            ),
            (
                estimate_artifact,
                estimate_op,
                &["identify.temporal_backdoor"],
                &estimate.assumptions,
            ),
        );

        let mut diagnostics = Vec::new();
        let stacked = stack_panel_tabular(panel).map_err(CausalError::from)?;
        let ate_q = AverageEffectQuery::binary_ate(query.treatment, query.outcome);
        let mut refute_ws = EstimationWorkspace::default();
        let temporal_ctx = TemporalRefitContext {
            indexer: &id_res.indexer,
            temporal_query: query,
            split: self.split.as_ref(),
            kernel_policy: &ctx.kernel_policy,
            time_index: None,
            panel: Some(panel),
        };
        let mut refutations = run_refuters(
            &stacked,
            &estimand,
            &ate_q,
            &estimate,
            &mut refute_ws,
            None,
            ctx,
            self.refute,
            if posterior.is_some() {
                "bayesian.temporal.gcomp"
            } else {
                "temporal.linear.adjustment"
            },
            &self.custom_validators,
            Some(temporal_ctx),
        )?;

        // Panel Bayesian: α-grid under Full when external compose is present (mirror temporal).
        if matches!(self.refute, RefuteSuite::Full) {
            if let (InferenceMode::Bayesian(cfg), Some(post)) = (&self.inference, &posterior) {
                let bprep = BayesianGComputationAte::from_prepared_estimation(&prep);
                let mut est = BayesianTemporalGcomp {
                    inner: BayesianGComputationAte {
                        backend: cfg.backend,
                        likelihood: cfg.likelihood,
                        n_draws: cfg.n_draws,
                        seed: ctx.rng.master_seed(),
                        overlap: OverlapPolicy::ExplicitOverride,
                        prior_scale: cfg.prior_scale,
                        prior: None,
                    },
                };
                let mut ws = BayesianGCompWorkspace::default();
                let (summary, sens) = if let Some(ext) = cfg.external_compose.as_ref() {
                    let alphas_applied: Arc<[f64]> = post.conflict_summary.as_ref().map_or_else(
                        || Arc::clone(&ext.composed.alphas_applied),
                        |cs| Arc::clone(&cs.alphas_applied),
                    );
                    est.inner.prior = Some(ext.composed.prior.clone());
                    let sens = PriorSensitivity::standard_alpha_grid();
                    let (summary, _) = sens
                        .evaluate_external_alpha(
                            &est.inner,
                            &bprep,
                            identification.status,
                            &mut ws,
                            ctx,
                            ExternalAlphaSensitivity {
                                sources: &ext.sources,
                                alphas_applied: &alphas_applied,
                            },
                        )
                        .map_err(CausalError::from)?;
                    (summary, sens)
                } else {
                    est.inner.prior = resolve_bayesian_prior(cfg, &bprep)?;
                    let sens = PriorSensitivity::standard_grid();
                    let (summary, _) = sens
                        .evaluate(&est.inner, &bprep, identification.status, &mut ws, ctx)
                        .map_err(CausalError::from)?;
                    (summary, sens)
                };
                refutations.push(sens.to_report(&summary, estimate.ate));
                posterior = Some(with_prior_sensitivity(post.clone(), summary));
            }
        }

        if let Some(cs) = posterior.as_ref().and_then(|p| p.conflict_summary.as_ref()) {
            push_conflict_diagnostics(&mut diagnostics, cs);
        }

        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior,
            mediation: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
        }))
    }

    /// ADMG ATE via general ID + functional plug-in (bidirected case).
    fn execute_admg(
        &self,
        data: &TabularData,
        admg: &Admg,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let started = Instant::now();
        let identifier = physical
            .logical
            .record
            .identifier
            .as_deref()
            .unwrap_or(crate::strategy_table::DEFAULT_ADMG_IDENTIFIER);
        let estimator = physical
            .logical
            .record
            .estimator
            .as_deref()
            .unwrap_or(crate::strategy_table::DEFAULT_ADMG_ESTIMATOR);
        if !matches!(EstimatorId::parse(estimator), EstimatorId::FunctionalEffect) {
            return Err(CausalError::Compile {
                message: format!("ADMG ATE requires estimator functional.effect; got {estimator}"),
            });
        }

        let identification = identify_admg(identifier, admg, query)?;
        let estimand = select_estimand(&identification, EstimatorId::parse(estimator))?;
        let est = FunctionalEffect {
            bootstrap_replicates: self.bootstrap_replicates,
            ..FunctionalEffect::new()
        };
        let prepared = est
            .prepare(
                data,
                &estimand,
                &identification.arena,
                identification.required_assumptions.clone(),
                &[query.treatment, query.outcome],
            )
            .map_err(CausalError::from)?;
        let mut ws = FunctionalDistributionWorkspace::default();
        let estimate = est.estimate(&prepared, &mut ws, ctx).map_err(CausalError::from)?;

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));

        let mut refute_ws = EstimationWorkspace::default();
        let refutations = run_refuters(
            data,
            &estimand,
            query,
            &estimate,
            &mut refute_ws,
            None,
            ctx,
            self.refute,
            estimator,
            &self.custom_validators,
            None,
        )?;

        let (id_artifact, id_op) = identify_provenance_step(identifier);
        let (est_artifact, est_op) = estimate_provenance_step(estimator);
        let provenance = provenance_pair(
            (id_artifact, id_op, &[], &identification.required_assumptions),
            (est_artifact, est_op, &[id_artifact], &estimate.assumptions),
        );

        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior: None,
            mediation: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
        }))
    }

    /// PAG ATE via generalized-adjustment envelope + mass-weighted estimates.
    fn execute_pag(
        &self,
        data: &TabularData,
        pag: &Pag,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let started = Instant::now();
        let identifier = physical
            .logical
            .record
            .identifier
            .as_deref()
            .unwrap_or(DEFAULT_PAG_IDENTIFIER_ID.as_str());
        let estimator = physical
            .logical
            .record
            .estimator
            .as_deref()
            .unwrap_or(DEFAULT_PAG_ESTIMATOR_ID.as_str());
        let estimator_id = EstimatorId::parse(estimator);
        let envelope = identify_pag(identifier, pag, query)?;
        if matches!(envelope.status, IdentificationStatus::NotIdentified)
            || envelope.identified_weight.0 <= 0.0
        {
            if matches!(self.inference, InferenceMode::Bayesian(_))
                || matches!(estimator_id, EstimatorId::BayesianGcomp)
            {
                return self
                    .execute_pag_nonidentified_prior(query, physical, ctx, &envelope, started);
            }
            return Err(CausalError::Compile {
                message: "PAG effect not identified (no identified mass in envelope)".into(),
            });
        }

        let mut diagnostics = Vec::new();
        diagnostics.push(Diagnostic::new(
            "identify.pag.envelope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "generalized.adjustment envelope: identified_mass={}, unidentified_mass={}, cases={}",
                envelope.identified_weight.0,
                envelope.unidentified_weight.0,
                envelope.cases.len()
            ),
        ));

        let identification = envelope_to_identification_result(&envelope, query);

        if matches!(estimator_id, EstimatorId::BayesianGcomp) {
            return self.execute_pag_bayesian(
                data,
                query,
                physical,
                ctx,
                &envelope,
                identification,
                started,
            );
        }

        let mut weighted_ate = 0.0;
        let mut weighted_se2 = 0.0;
        let mut total_w = 0.0;
        let mut primary_estimand: Option<IdentifiedEstimand> = None;
        let mut assumptions = causal_core::AssumptionSet::default();
        for (i, case) in envelope.cases.iter().enumerate() {
            if !identification_status_ok_for_case(case.result.status)
                || case.result.estimands.is_empty()
            {
                continue;
            }
            let mut estimand = select_estimand(&case.result, estimator_id.clone())?;
            // Generalized-adjustment estimands are backdoor-shaped; estimators expect
            // the canonical backdoor method tag.
            if estimand.method.as_ref().starts_with("generalized.adjustment") {
                estimand.method = Arc::from("backdoor.adjustment");
            }
            let mut case_ws = StaticEstimateWorkspaces::default();
            let estimate = estimate_static_effect(
                estimator_id.clone(),
                data,
                &estimand,
                query,
                case.result.required_assumptions.clone(),
                self.bootstrap_replicates,
                self.overlap_policy,
                self.population_registry.as_ref(),
                ctx,
                &mut case_ws,
            )?;
            let w = case.weight.0;
            weighted_ate += w * estimate.ate;
            if estimate.se_analytic.is_finite() {
                weighted_se2 += w * estimate.se_analytic * estimate.se_analytic;
            }
            total_w += w;
            if primary_estimand.is_none() {
                primary_estimand = Some(estimand);
                assumptions = estimate.assumptions.clone();
            }
            let _ = i;
        }
        if !matches!(total_w.partial_cmp(&0.0), Some(std::cmp::Ordering::Greater)) {
            return Err(CausalError::Compile {
                message: "PAG envelope had no estimable identified cases".into(),
            });
        }
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "PAG envelope missing estimand".into(),
        })?;
        let estimate = EffectEstimate {
            ate: weighted_ate / total_w,
            se_analytic: (weighted_se2 / total_w).sqrt(),
            se_bootstrap: None,
            bootstrap_replicates_ok: None,
            bootstrap_replicates_failed: None,
            bootstrap_cancelled: false,
            bootstrap_early_stopped: false,
            assumptions: assumptions.clone(),
            overlap: OverlapPolicy::ExplicitOverride,
            overlap_report: None,
            retained_memory_bytes: None,
        };

        let mut refute_ws = EstimationWorkspace::default();
        let refutations = run_refuters(
            data,
            &estimand,
            query,
            &estimate,
            &mut refute_ws,
            None,
            ctx,
            self.refute,
            estimator,
            &self.custom_validators,
            None,
        )?;

        let (id_artifact, id_op) = identify_provenance_step(identifier);
        let (est_artifact, est_op) = estimate_provenance_step(estimator);
        let provenance = provenance_pair(
            (id_artifact, id_op, &[], &identification.required_assumptions),
            (est_artifact, est_op, &[id_artifact], &estimate.assumptions),
        );
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior: None,
            mediation: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
        }))
    }

    /// Non-identified PAG with Bayesian inference: prior-predictive draws, no invented ID.
    fn execute_pag_nonidentified_prior(
        &self,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        envelope: &IdentificationEnvelope<Pag>,
        started: Instant,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let cfg = match &self.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => BayesianConfig::laplace(),
        };
        let scale = cfg.prior_scale.max(1e-6);
        let mut prior = PriorSet::weakly_informative(1);
        if let Some(g) = prior.specs.iter_mut().find_map(|s| match s {
            causal_prob::PriorSpec::GaussianCoefficients(p) => Some(p),
            _ => None,
        }) {
            *g = causal_prob::GaussianCoefficientPrior::isotropic(1, scale);
        }
        let posterior = nonidentified_with_prior(
            &prior,
            InferenceDiagnostics::analytic("pag_nonidentified_prior"),
            cfg.n_draws.max(1),
            ctx.rng.master_seed(),
        );
        let estimate = effect_from_posterior(&posterior)?;
        let identification = envelope_to_identification_result(envelope, query);
        let estimand = envelope.invariant.clone().unwrap_or_else(|| {
            IdentifiedEstimand::backdoor(
                "pag.nonidentified",
                Arc::from([]),
                causal_expr::ExprId::from_raw(0),
            )
        });
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(Diagnostic::new(
            "estimate.pag.nonidentified_prior",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "PAG not identified; returning prior-predictive draws (unidentified_mass={})",
                posterior.unidentified_mass
            ),
        ));
        let provenance = provenance_pair(
            (
                "identify.generalized_adjustment",
                "identify.generalized_adjustment",
                &[],
                &identification.required_assumptions,
            ),
            (
                "estimate.bayesian_gcomp",
                "estimate.nonidentified_with_prior",
                &["identify.generalized_adjustment"],
                &estimate.assumptions,
            ),
        );
        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior: Some(posterior),
            mediation: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations: Vec::new(),
            diagnostics,
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
        }))
    }

    fn compile_graph_posterior_static_ate(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        ctx: &ExecutionContext,
    ) -> Result<CompiledAnalysis, CausalError> {
        if matches!(self.inference, InferenceMode::Frequentist) {
            return Err(CausalError::Unsupported {
                message: "graph-posterior discovery requires inference=Bayesian for effect mixture",
            });
        }
        let algo = match &self.graph {
            GraphInput::DiscoverExactDagPosterior => "exact_dag_posterior",
            GraphInput::DiscoverOrderMcmc { .. } => "order_mcmc",
            GraphInput::DiscoverStructureMcmc { .. } => "structure_mcmc",
            GraphInput::DiscoverCiScreenedPosterior { .. } => "ci_screened_posterior",
            _ => "graph_posterior",
        };
        let n_vars = u32::try_from(data.schema().len()).map_err(|_| CausalError::Compile {
            message: "too many variables for graph-posterior compile".into(),
        })?;
        let stub = Dag::with_variables(n_vars);
        let identifier = Arc::from("backdoor.adjustment");
        let estimator = Arc::from("bayesian.gcomp");
        let mut logical = compile_logical_static_ate(StaticAteCompileInput {
            data,
            graph: &stub,
            query,
            validation_suite: self.validation_suite_id(),
            identifier,
            estimator,
        })?;
        logical.record.discovery_algorithm = Some(Arc::from(algo));
        let physical = logical.compile_physical_with_graphs(ctx, None, None)?;
        Ok(CompiledAnalysis::Ready(physical))
    }

    fn discover_graph_posterior_for_ate(
        &self,
        data: &TabularData,
        ctx: &ExecutionContext,
    ) -> Result<causal_discovery::GraphPosterior, CausalError> {
        let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
        let params = BayesianDiscoverParams::default();
        match &self.graph {
            GraphInput::DiscoverExactDagPosterior => {
                discover_exact_dag_posterior(data, &vars, &params, ctx)
            }
            GraphInput::DiscoverOrderMcmc {
                n_chains,
                n_warmup,
                n_draws,
                thin,
                require_diagnostics_gate,
            } => {
                let schedule = GraphMcmcSchedule {
                    n_chains: *n_chains,
                    n_warmup: *n_warmup,
                    n_draws: *n_draws,
                    thin: *thin,
                };
                discover_order_mcmc(data, &vars, &params, &schedule, *require_diagnostics_gate, ctx)
            }
            GraphInput::DiscoverStructureMcmc { n_chains, n_warmup, n_draws, thin } => {
                let schedule = GraphMcmcSchedule {
                    n_chains: *n_chains,
                    n_warmup: *n_warmup,
                    n_draws: *n_draws,
                    thin: *thin,
                };
                discover_structure_mcmc(data, &vars, &params, &schedule, ctx)
            }
            GraphInput::DiscoverCiScreenedPosterior {
                alpha,
                fdr,
                max_cond_size,
                soft_weight,
                n_chains,
                n_warmup,
                n_draws,
                thin,
            } => {
                let ci = resolve_analysis_ci(self.discovery_ci.as_ref())?;
                let screen = StaticDiscoverParams {
                    alpha: *alpha,
                    max_cond_size: *max_cond_size,
                    fdr: *fdr,
                    ci,
                    screen_pc: false,
                    max_subset: None,
                };
                let schedule = GraphMcmcSchedule {
                    n_chains: *n_chains,
                    n_warmup: *n_warmup,
                    n_draws: *n_draws,
                    thin: *thin,
                };
                discover_ci_screened_posterior(
                    data,
                    &vars,
                    &params,
                    &screen,
                    &schedule,
                    *soft_weight,
                    ctx,
                )
            }
            _ => Err(CausalError::Compile {
                message: "discover_graph_posterior_for_ate: unexpected GraphInput".into(),
            }),
        }
    }

    fn execute_graph_posterior_bayesian(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let started = Instant::now();
        let cfg = match &self.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => {
                return Err(CausalError::Unsupported {
                    message: "graph-posterior discovery requires inference=Bayesian for effect mixture",
                });
            }
        };
        let gp = self.discover_graph_posterior_for_ate(data, ctx)?;
        let mut est = BayesianGComputationAte {
            backend: cfg.backend,
            likelihood: cfg.likelihood,
            n_draws: cfg.n_draws,
            seed: ctx.rng.master_seed(),
            overlap: OverlapPolicy::ExplicitOverride,
            prior_scale: cfg.prior_scale,
            prior: None,
        };

        let mut weights = Vec::with_capacity(gp.n_graphs);
        let mut flags = Vec::with_capacity(gp.n_graphs);
        let mut keys = Vec::with_capacity(gp.n_graphs);
        let mut per_graph = Vec::new();
        let mut primary_estimand: Option<IdentifiedEstimand> = None;
        let mut primary_identification: Option<IdentificationResult> = None;
        let mut envelope_prior: Option<PriorSet> = None;
        let mut envelope_conflict: Option<causal_prob::ConflictSummary> = None;

        for i in 0..gp.n_graphs {
            if ctx.cancellation.is_cancelled() {
                for j in i..gp.n_graphs {
                    keys.push(gp.graph_keys[j]);
                    weights.push(gp.weights[j]);
                    flags.push(GraphIdentFlag::Unidentified);
                }
                break;
            }
            if let Some(p) = &ctx.progress {
                #[allow(clippy::cast_precision_loss)]
                p.report(i as f64 / gp.n_graphs.max(1) as f64, "envelope");
            }
            let mask = gp.adjacency[i];
            let key = gp.graph_keys[i];
            keys.push(key);
            weights.push(gp.weights[i]);
            let Ok(dag) = dag_from_adjacency_mask(mask, gp.n_vars) else {
                flags.push(GraphIdentFlag::Unidentified);
                continue;
            };
            let Ok(identification) = identify_static(DEFAULT_IDENTIFIER, &dag, query) else {
                flags.push(GraphIdentFlag::Unidentified);
                continue;
            };
            if identification_status_ok_for_case(identification.status)
                && !identification.estimands.is_empty()
            {
                flags.push(GraphIdentFlag::Identified);
                let estimand = select_estimand(&identification, EstimatorId::BayesianGcomp)?;
                if primary_estimand.is_none() {
                    primary_estimand = Some(estimand.clone());
                    primary_identification = Some(identification.clone());
                }
                let prep = est.prepare(data, &estimand, query).map_err(CausalError::from)?;
                if envelope_prior.is_none() {
                    let (resolved, conflict) =
                        resolve_bayesian_prior_with_conflict(&cfg, &prep, Some(ctx))?;
                    envelope_prior = resolved;
                    envelope_conflict = conflict;
                }
                est.prior.clone_from(&envelope_prior);
                let mut ws = BayesianGCompWorkspace::default();
                let posterior = est
                    .fit(&prep, identification.status, &mut ws, ctx)
                    .map_err(CausalError::from)?;
                let col = posterior.effect_column().ok_or_else(|| CausalError::Compile {
                    message: "Bayesian posterior missing effect column".into(),
                })?;
                let draws = posterior
                    .draws
                    .column(col)
                    .map_err(|e| CausalError::Compile { message: e.to_string() })?
                    .to_vec();
                per_graph.push(GraphEffectDraws { graph_key: key, effect_draws: Arc::from(draws) });
            } else {
                flags.push(GraphIdentFlag::Unidentified);
            }
        }

        let graphs = WeightedGraphSamples::new(weights, flags, keys)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let mut subsample_notes = Vec::new();
        let (graphs, per_graph) = maybe_interactive_envelope_subsample(
            self.latency_mode,
            graphs,
            per_graph,
            ctx,
            &mut subsample_notes,
        )?;
        let mut posterior = aggregate_effect_envelope(
            &graphs,
            &per_graph,
            InferenceDiagnostics::analytic("graph_posterior_envelope"),
            EnvelopeOptions::default(),
        )
        .map_err(CausalError::from)?;
        if let Some(summary) = envelope_conflict {
            posterior = with_conflict_summary(posterior, summary);
        }
        let estimate = effect_from_posterior(&posterior)?;
        let identification = primary_identification.ok_or_else(|| CausalError::Compile {
            message: "graph-posterior envelope: no identified graph atoms".into(),
        })?;
        let estimand = primary_estimand.ok_or_else(|| CausalError::Compile {
            message: "graph-posterior envelope: missing estimand".into(),
        })?;

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.extend(subsample_notes);
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        diagnostics.push(Diagnostic::new(
            "estimate.graph_posterior.envelope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!("unidentified_mass={}", posterior.unidentified_mass),
        ));
        if let Some(cs) = posterior.conflict_summary.as_ref() {
            push_conflict_diagnostics(&mut diagnostics, cs);
        }

        let mut refute_ws = EstimationWorkspace::default();
        let refutations = match self.refute {
            RefuteSuite::None => Vec::new(),
            RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => run_refuters(
                data,
                &estimand,
                query,
                &estimate,
                &mut refute_ws,
                None,
                ctx,
                self.refute,
                "bayesian.gcomp",
                &self.custom_validators,
                None,
            )?,
        };

        let algo =
            physical.logical.record.discovery_algorithm.as_deref().unwrap_or("graph_posterior");
        let provenance = provenance_pair(
            ("discover.graph_posterior", algo, &[], &identification.required_assumptions),
            (
                "estimate.aggregate_effect_envelope",
                "estimate.bayesian_gcomp",
                &["discover.graph_posterior"],
                &estimate.assumptions,
            ),
        );

        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior: Some(posterior),
            mediation: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
        }))
    }

    fn execute_pag_bayesian(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
        envelope: &IdentificationEnvelope<Pag>,
        identification: IdentificationResult,
        started: Instant,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let cfg = match &self.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => BayesianConfig::laplace(),
        };
        let mut est = BayesianGComputationAte {
            backend: cfg.backend,
            likelihood: cfg.likelihood,
            n_draws: cfg.n_draws,
            seed: ctx.rng.master_seed(),
            overlap: OverlapPolicy::ExplicitOverride,
            prior_scale: cfg.prior_scale,
            prior: None,
        };

        let mut weights = Vec::new();
        let mut flags = Vec::new();
        let mut keys = Vec::new();
        let mut per_graph = Vec::new();
        let mut primary_estimand: Option<IdentifiedEstimand> = None;
        let mut envelope_prior: Option<PriorSet> = None;
        let mut envelope_conflict: Option<causal_prob::ConflictSummary> = None;
        for (i, case) in envelope.cases.iter().enumerate() {
            let key = i as u64 + 1;
            keys.push(key);
            weights.push(case.weight.0);
            if identification_status_ok_for_case(case.result.status)
                && !case.result.estimands.is_empty()
            {
                flags.push(GraphIdentFlag::Identified);
                let mut estimand = select_estimand(&case.result, EstimatorId::BayesianGcomp)?;
                if estimand.method.as_ref().starts_with("generalized.adjustment") {
                    estimand.method = Arc::from("backdoor.adjustment");
                }
                if primary_estimand.is_none() {
                    primary_estimand = Some(estimand.clone());
                }
                let prep = est.prepare(data, &estimand, query).map_err(CausalError::from)?;
                if envelope_prior.is_none() {
                    let (resolved, conflict) =
                        resolve_bayesian_prior_with_conflict(&cfg, &prep, Some(ctx))?;
                    envelope_prior = resolved;
                    envelope_conflict = conflict;
                }
                est.prior.clone_from(&envelope_prior);
                let mut ws = BayesianGCompWorkspace::default();
                let posterior = est
                    .fit(&prep, case.result.status, &mut ws, ctx)
                    .map_err(CausalError::from)?;
                let col = posterior.effect_column().ok_or_else(|| CausalError::Compile {
                    message: "Bayesian posterior missing effect column".into(),
                })?;
                let draws = posterior
                    .draws
                    .column(col)
                    .map_err(|e| CausalError::Compile { message: e.to_string() })?
                    .to_vec();
                per_graph.push(GraphEffectDraws { graph_key: key, effect_draws: Arc::from(draws) });
            } else {
                flags.push(GraphIdentFlag::Unidentified);
            }
        }
        let graphs = WeightedGraphSamples::new(weights, flags, keys)
            .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let mut subsample_notes = Vec::new();
        let (graphs, per_graph) = maybe_interactive_envelope_subsample(
            self.latency_mode,
            graphs,
            per_graph,
            ctx,
            &mut subsample_notes,
        )?;
        let mut posterior = aggregate_effect_envelope(
            &graphs,
            &per_graph,
            InferenceDiagnostics::analytic("pag_envelope"),
            EnvelopeOptions::default(),
        )
        .map_err(CausalError::from)?;
        if let Some(summary) = envelope_conflict {
            posterior = with_conflict_summary(posterior, summary);
        }
        let estimate = effect_from_posterior(&posterior)?;
        let estimand = primary_estimand.or(envelope.invariant.clone()).ok_or_else(|| {
            CausalError::Compile { message: "PAG Bayesian envelope missing estimand".into() }
        })?;

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.extend(subsample_notes);
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        diagnostics.push(Diagnostic::new(
            "estimate.pag.envelope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!("unidentified_mass={}", posterior.unidentified_mass),
        ));
        if let Some(cs) = posterior.conflict_summary.as_ref() {
            push_conflict_diagnostics(&mut diagnostics, cs);
        }

        let mut refute_ws = EstimationWorkspace::default();
        let refutations = match self.refute {
            RefuteSuite::None => Vec::new(),
            RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => run_refuters(
                data,
                &estimand,
                query,
                &estimate,
                &mut refute_ws,
                None,
                ctx,
                self.refute,
                "bayesian.gcomp",
                &self.custom_validators,
                None,
            )?,
        };
        if matches!(self.refute, RefuteSuite::Full) {
            // PPC suite needs a single-graph fit context; skip with diagnostic when envelope-only.
            diagnostics.push(Diagnostic::new(
                "refute.bayesian.ppc.skipped",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "Bayesian PPC suite skipped for multi-graph PAG envelope; effect refuters ran on mixture mean",
            ));
        }

        let provenance = provenance_pair(
            (
                "identify.generalized_adjustment",
                "identify.generalized_adjustment",
                &[],
                &identification.required_assumptions,
            ),
            (
                "estimate.bayesian_gcomp",
                "estimate.aggregate_effect_envelope",
                &["identify.generalized_adjustment"],
                &estimate.assumptions,
            ),
        );
        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior: Some(posterior),
            mediation: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
        }))
    }

    fn execute_conditional(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &causal_core::ConditionalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let started = Instant::now();
        let (identifier, _) = self.resolve_conditional_pair();
        let identification = identify_static(identifier.as_ref(), graph, &query.inner)?;
        let estimand = select_estimand(&identification, EstimatorId::ConditionalLinearAdjustment)?;
        let est = ConditionalLinearAdjustment::new();
        let estimate = est.estimate(data, &estimand, query, ctx).map_err(CausalError::from)?;
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        let mut refute_ws = EstimationWorkspace::default();
        let refutations = run_refuters(
            data,
            &estimand,
            &query.inner,
            &estimate,
            &mut refute_ws,
            None,
            ctx,
            self.refute,
            "conditional.linear.adjustment",
            &self.custom_validators,
            None,
        )?;
        let (id_artifact, id_op) = identify_provenance_step(identifier.as_ref());
        let provenance = provenance_pair(
            (id_artifact, id_op, &[], &identification.required_assumptions),
            (
                "estimate.conditional_linear",
                "estimate.conditional_linear_adjustment",
                &[id_artifact],
                &estimate.assumptions,
            ),
        );
        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior: None,
            mediation: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment: query.inner.treatment,
            outcome: query.inner.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
        }))
    }

    fn execute_temporal_mediation(
        &self,
        data: &TimeSeriesData,
        graph: &TemporalDag,
        query: &causal_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let started = Instant::now();
        let identification = TemporalMediationIdentifier {
            allow_natural_controlled_alias: true,
            ..TemporalMediationIdentifier::new()
        }
        .identify(graph, query)
        .map_err(CausalError::from)?;
        require_identified(&identification)?;
        let estimand = select_estimand(&identification, EstimatorId::TemporalMediation)?;
        let mut est = TemporalMediationEstimator::new();
        est.allow_natural_controlled_alias = true;
        let mediation = est.estimate(data, &estimand, query, ctx).map_err(CausalError::from)?;
        let estimate = mediation.effect.clone();
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        let provenance = provenance_pair(
            (
                "identify.temporal_mediation",
                "identify.temporal_mediation",
                &[],
                &identification.required_assumptions,
            ),
            (
                "estimate.temporal_mediation",
                "estimate.temporal_mediation",
                &["identify.temporal_mediation"],
                &estimate.assumptions,
            ),
        );
        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior: None,
            mediation: Some(mediation),
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations: Vec::new(),
            diagnostics,
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
        }))
    }

    fn execute_static_mediation_total(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &causal_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let started = Instant::now();
        if !matches!(query.contrast, MediationContrast::Total) {
            return Err(CausalError::Unsupported {
                message: "static Mediation supports only MediationContrast::Total via front-door",
            });
        }
        let ate = AverageEffectQuery {
            treatment: query.treatment,
            outcome: query.outcome,
            control: query.control.clone(),
            active: query.active.clone(),
            effect_modifiers: Arc::from([]),
            target_population: query.target_population.clone(),
        };
        let identification = identify_static("frontdoor", graph, &ate)?;
        let estimand = select_estimand(&identification, EstimatorId::FrontDoorTwoStage)?;
        let mut estimate_ws = StaticEstimateWorkspaces::default();
        let estimate = estimate_static_effect(
            EstimatorId::FrontDoorTwoStage,
            data,
            &estimand,
            &ate,
            identification.required_assumptions.clone(),
            self.bootstrap_replicates,
            self.overlap_policy,
            self.population_registry.as_ref(),
            ctx,
            &mut estimate_ws,
        )?;
        let mediation = TemporalMediationEstimate {
            effect: estimate.clone(),
            total: Some(estimate.ate),
            direct: None,
            mediated: None,
        };
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        let refutations = run_refuters(
            data,
            &estimand,
            &ate,
            &estimate,
            &mut estimate_ws.linear,
            None,
            ctx,
            self.refute,
            "frontdoor.two_stage",
            &self.custom_validators,
            None,
        )?;
        let provenance = provenance_pair(
            ("identify.frontdoor", "identify.frontdoor", &[], &identification.required_assumptions),
            (
                "estimate.frontdoor",
                "estimate.frontdoor_two_stage",
                &["identify.frontdoor"],
                &estimate.assumptions,
            ),
        );
        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior: None,
            mediation: Some(mediation),
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations,
            diagnostics,
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
        }))
    }

    fn execute_counterfactual(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &causal_core::CounterfactualQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let fitted = fit_gcm(graph.clone(), data)?;
        let outcome = *query.outcomes.first().ok_or_else(|| CausalError::Compile {
            message: "counterfactual query missing outcome".into(),
        })?;
        let (treatment, active, control) = binary_cf_interventions(query)?;
        let ite = counterfactual_ite(fitted.model, data, treatment, outcome, active, control, ctx)?;
        let estimate = EffectEstimate {
            ate: ite.mean_ite,
            se_analytic: f64::NAN,
            se_bootstrap: None,
            bootstrap_replicates_ok: None,
            bootstrap_replicates_failed: None,
            bootstrap_cancelled: false,
            bootstrap_early_stopped: false,
            assumptions: causal_core::AssumptionSet::default(),
            overlap: OverlapPolicy::ExplicitOverride,
            overlap_report: None,
            retained_memory_bytes: None,
        };
        let (identification, estimand) = parametric_scm_identification(
            CausalQuery::Counterfactual(query.clone()),
            treatment,
            outcome,
        );
        let mut diagnostics = vec![Diagnostic::new(
            "gcm.counterfactual",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!("noise_inference={:?}", ite.noise_inference),
        )];
        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior: None,
            mediation: None,
            counterfactual: Some(ite),
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations: Vec::new(),
            diagnostics,
            provenance: ProvenanceGraph::new(),
            treatment,
            outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
        }))
    }

    fn execute_anomaly(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &causal_core::AnomalyAttributionQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let _ = ctx;
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let fitted = fit_gcm(graph.clone(), data)?;
        let scores = anomaly_attribution(
            &fitted.model,
            data,
            query.targets.iter().copied(),
            query.max_units,
        )?;
        let outcome = *query.targets.first().unwrap_or(&VariableId::from_raw(0));
        let treatment = outcome;
        let estimate = nan_effect();
        let (identification, estimand) = parametric_scm_identification(
            CausalQuery::AnomalyAttribution(query.clone()),
            treatment,
            outcome,
        );
        let mut diagnostics = Vec::new();
        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior: None,
            mediation: None,
            counterfactual: None,
            anomaly: Some(scores),
            change_attribution: None,
            mechanism_change: None,
            unit_change: None,
            refutations: Vec::new(),
            diagnostics,
            provenance: ProvenanceGraph::new(),
            treatment,
            outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
        }))
    }

    fn execute_change_attribution(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &causal_core::ChangeAttributionQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let fitted = fit_gcm(graph.clone(), data)?;
        let result = attribute_distribution_change(
            &fitted.model,
            data,
            query,
            &causal_attribution::DistributionChangeOptions::default(),
            ctx,
        )?;
        let outcome = query.outcome;
        let treatment = outcome;
        let estimate = EffectEstimate {
            ate: result.total_change,
            se_analytic: f64::NAN,
            se_bootstrap: None,
            bootstrap_replicates_ok: None,
            bootstrap_replicates_failed: None,
            bootstrap_cancelled: false,
            bootstrap_early_stopped: false,
            assumptions: causal_core::AssumptionSet::default(),
            overlap: OverlapPolicy::ExplicitOverride,
            overlap_report: None,
            retained_memory_bytes: None,
        };
        let (identification, estimand) = parametric_scm_identification(
            CausalQuery::ChangeAttribution(query.clone()),
            treatment,
            outcome,
        );
        let mut diagnostics = Vec::new();
        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior: None,
            mediation: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: Some(result),
            mechanism_change: None,
            unit_change: None,
            refutations: Vec::new(),
            diagnostics,
            provenance: ProvenanceGraph::new(),
            treatment,
            outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
        }))
    }

    fn execute_mechanism_change(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &causal_core::MechanismChangeQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let fitted = fit_gcm(graph.clone(), data)?;
        let detections = mechanism_change_detection(
            &fitted.model,
            data,
            query,
            causal_attribution::MechanismChangeMethod::LikelihoodRatio,
            ctx,
        )?;
        let outcome = *query.targets.first().unwrap_or(&VariableId::from_raw(0));
        let treatment = outcome;
        let estimate = nan_effect();
        let (identification, estimand) = parametric_scm_identification(
            CausalQuery::MechanismChange(query.clone()),
            treatment,
            outcome,
        );
        let mut diagnostics = Vec::new();
        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior: None,
            mediation: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: Some(detections),
            unit_change: None,
            refutations: Vec::new(),
            diagnostics,
            provenance: ProvenanceGraph::new(),
            treatment,
            outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
        }))
    }

    fn execute_unit_change(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &causal_core::UnitChangeQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let started = Instant::now();
        query.validate().map_err(|e| CausalError::Compile { message: e.to_string() })?;
        let fitted = fit_gcm(graph.clone(), data)?;
        let result = attribute_unit_change(&fitted.model, data, query, ctx)?;
        let outcome = query.outcome;
        let treatment = outcome;
        let estimate = nan_effect();
        let (identification, estimand) = parametric_scm_identification(
            CausalQuery::UnitChange(query.clone()),
            treatment,
            outcome,
        );
        let mut diagnostics = Vec::new();
        let physical_record =
            self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
        Ok(assemble_result(AssembleArgs {
            logical: &physical.logical.record,
            physical: &physical_record,
            identification,
            estimand,
            estimate,
            distribution: None,
            posterior: None,
            mediation: None,
            counterfactual: None,
            anomaly: None,
            change_attribution: None,
            mechanism_change: None,
            unit_change: Some(result),
            refutations: Vec::new(),
            diagnostics,
            provenance: ProvenanceGraph::new(),
            treatment,
            outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
            latency_mode: self.latency_mode.map(|m| Arc::from(m.as_str())),
            stage_timings_ns: Vec::new(),
            bootstrap_replicates_requested: Some(self.bootstrap_replicates),
            bootstrap_replicates_ok: None,
            n_draws: None,
            cancelled: false,
            early_stopped: false,
        }))
    }

    /// Continue after static PAG review once circle marks are resolved.
    ///
    /// # Errors
    ///
    /// Incomplete review or execute failures.
    pub fn finish_static_pag_review_and_run(
        &self,
        review: PagReview,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, CausalError> {
        let DataInput::Tabular(data) = &self.data else {
            return Err(CausalError::Compile {
                message: "finish_static_pag_review_and_run requires tabular data".into(),
            });
        };
        let CausalQuery::AverageEffect(q) = &self.query else {
            return Err(CausalError::Compile {
                message: "finish_static_pag_review_and_run requires AverageEffect".into(),
            });
        };
        // Circles are fine: generalized adjustment samples MAG completions.
        let (identifier, estimator) = self.resolve_pag_pair();
        let logical = compile_logical_static_pag_ate(StaticPagAteCompileInput {
            data,
            pag: &review.graph,
            query: q,
            validation_suite: self.validation_suite_id(),
            identifier,
            estimator,
        })?;
        let physical =
            logical.compile_physical_with_all_graphs(ctx, None, None, Some(review.graph))?;
        self.execute(&CompiledAnalysis::Ready(physical), ctx)
    }
}

fn gcm_query_vars(query: &CausalQuery) -> Result<(VariableId, VariableId), CausalError> {
    match query {
        CausalQuery::Counterfactual(q) => {
            let outcome = *q.outcomes.first().ok_or_else(|| CausalError::Compile {
                message: "counterfactual missing outcome".into(),
            })?;
            let treatment =
                q.interventions.first().and_then(Intervention::primary_variable).unwrap_or(outcome);
            Ok((treatment, outcome))
        }
        CausalQuery::AnomalyAttribution(q) => {
            let outcome = *q.targets.first().unwrap_or(&VariableId::from_raw(0));
            Ok((outcome, outcome))
        }
        CausalQuery::ChangeAttribution(q) => Ok((q.outcome, q.outcome)),
        CausalQuery::MechanismChange(q) => {
            let outcome = *q.targets.first().unwrap_or(&VariableId::from_raw(0));
            Ok((outcome, outcome))
        }
        CausalQuery::UnitChange(q) => Ok((q.outcome, q.outcome)),
        _ => Err(CausalError::Compile { message: "gcm_query_vars: unsupported query".into() }),
    }
}

fn nan_effect() -> EffectEstimate {
    EffectEstimate {
        ate: f64::NAN,
        se_analytic: f64::NAN,
        se_bootstrap: None,
        bootstrap_replicates_ok: None,
        bootstrap_replicates_failed: None,
        bootstrap_cancelled: false,
        bootstrap_early_stopped: false,
        assumptions: causal_core::AssumptionSet::default(),
        overlap: OverlapPolicy::ExplicitOverride,
        overlap_report: None,
        retained_memory_bytes: None,
    }
}

/// Interactive graph×effect: stratified subsample of Identified graphs; leftover
/// identified mass is flipped to Unidentified (never silent renormalize to 1).
fn maybe_interactive_envelope_subsample(
    latency_mode: Option<LatencyMode>,
    graphs: WeightedGraphSamples,
    per_graph: Vec<GraphEffectDraws>,
    ctx: &ExecutionContext,
    diagnostics: &mut Vec<Diagnostic>,
) -> Result<(WeightedGraphSamples, Vec<GraphEffectDraws>), CausalError> {
    if latency_mode != Some(LatencyMode::Interactive) {
        return Ok((graphs, per_graph));
    }
    let mut rng = ctx.rng.stream(0xE11E_u64);
    let sub = graphs
        .stratified_interactive_subsample(INTERACTIVE_MAX_ENVELOPE_GRAPHS, &mut rng)
        .map_err(|e| CausalError::Compile { message: e.to_string() })?;
    if !sub.approximate {
        return Ok((sub.graphs, per_graph));
    }
    let keep_keys: std::collections::HashSet<u64> = sub
        .graphs
        .graph_keys
        .iter()
        .zip(sub.graphs.identified.iter())
        .filter(|(_, f)| **f == GraphIdentFlag::Identified)
        .map(|(k, _)| *k)
        .collect();
    let filtered: Vec<GraphEffectDraws> =
        per_graph.into_iter().filter(|g| keep_keys.contains(&g.graph_key)).collect();
    diagnostics.push(Diagnostic::new(
        "estimate.envelope.interactive_subsample",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "approximate=true leftover_identified_mass={} max_identified={}",
            sub.leftover_identified_mass, INTERACTIVE_MAX_ENVELOPE_GRAPHS
        ),
    ));
    Ok((sub.graphs, filtered))
}

fn parametric_scm_identification(
    query: CausalQuery,
    _treatment: VariableId,
    _outcome: VariableId,
) -> (IdentificationResult, IdentifiedEstimand) {
    let estimand = IdentifiedEstimand::backdoor(
        "gcm.parametric",
        Arc::from([]),
        causal_expr::ExprId::from_raw(0),
    );
    let identification = IdentificationResult {
        status: IdentificationStatus::IdentifiedUnderParametricRestrictions,
        query,
        estimands: vec![estimand.clone()],
        arena: CausalExprArena::new(),
        derivation: DerivationTrace::default(),
        required_assumptions: causal_core::AssumptionSet::default(),
        diagnostics: Vec::new(),
        performance: IdentificationPerformanceRecord::default(),
        hedge: None,
    };
    (identification, estimand)
}

fn binary_cf_interventions(
    query: &causal_core::CounterfactualQuery,
) -> Result<(VariableId, f64, f64), CausalError> {
    if query.interventions.len() != 1 {
        return Err(CausalError::Unsupported {
            message: "CausalAnalysis counterfactual path currently supports a single hard \
                 intervention for ITE (use gcm helpers for multi-world predict)",
        });
    }
    let Intervention::Set { variable, value } = &query.interventions[0] else {
        return Err(CausalError::Unsupported {
            message: "CausalAnalysis counterfactual path requires a hard Set intervention",
        });
    };
    let active = value.as_f64().ok_or_else(|| CausalError::Compile {
        message: "counterfactual intervention value must be f64".into(),
    })?;
    Ok((*variable, active, 0.0))
}

fn identification_status_ok_for_case(status: IdentificationStatus) -> bool {
    matches!(
        status,
        IdentificationStatus::NonparametricallyIdentified
            | IdentificationStatus::PartiallyIdentified
            | IdentificationStatus::IdentifiedUnderParametricRestrictions
            | IdentificationStatus::IdentifiedUnderPriorRestrictions
    )
}

fn envelope_to_identification_result(
    envelope: &IdentificationEnvelope<Pag>,
    query: &AverageEffectQuery,
) -> IdentificationResult {
    let mut estimands = Vec::new();
    let mut assumptions = causal_core::AssumptionSet::default();
    let mut diagnostics = Vec::new();
    for case in &envelope.cases {
        if identification_status_ok_for_case(case.result.status) {
            estimands.extend(case.result.estimands.iter().cloned());
            assumptions = case.result.required_assumptions.clone();
            diagnostics.extend(case.result.diagnostics.iter().cloned());
        }
    }
    if let Some(inv) = &envelope.invariant {
        if estimands.is_empty() {
            estimands.push(inv.clone());
        }
    }
    IdentificationResult {
        status: envelope.status,
        query: CausalQuery::AverageEffect(query.clone()),
        estimands,
        arena: CausalExprArena::new(),
        derivation: DerivationTrace::default(),
        required_assumptions: assumptions,
        diagnostics,
        performance: IdentificationPerformanceRecord::default(),
        hedge: None,
    }
}

fn mark_panel_classification(compiled: CompiledAnalysis) -> CompiledAnalysis {
    match compiled {
        CompiledAnalysis::Ready(mut physical) => {
            physical.logical.record.data_classification = DataClassification::Panel;
            CompiledAnalysis::Ready(physical)
        }
        other => other,
    }
}

fn admg_has_bidirected(admg: &Admg) -> bool {
    (0..admg.node_count()).any(|i| {
        let id = DenseNodeId::from_raw(u32::try_from(i).unwrap_or(u32::MAX));
        !admg.bidirected_neighbors(id).is_empty()
    })
}

fn admg_to_dag(admg: &Admg) -> Result<Dag, CausalError> {
    let n = u32::try_from(admg.node_count())
        .map_err(|_| CausalError::Compile { message: "ADMG too large".into() })?;
    let mut dag = Dag::with_variables(n);
    for i in 0..admg.node_count() {
        let from = DenseNodeId::from_raw(u32::try_from(i).unwrap_or(u32::MAX));
        for &to in admg.children(from) {
            dag.insert_directed(from, to)
                .map_err(|e| CausalError::Compile { message: e.to_string() })?;
        }
    }
    Ok(dag)
}
