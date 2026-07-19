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
    AverageEffectQuery, CausalQuery, DataClassification, ExecutionContext, PopulationRegistry, Intervention,
    MediationContrast, ProvenanceGraph, TemporalEffectQuery, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, VariableId,
};
use causal_data::{
    DiscoveryEstimationSplit, PanelData, TableView, TabularData, TimeSeriesData,
};
use causal_estimate::{
    AnalyticSeKind, BayesianGCompWorkspace, BayesianGComputationAte, ConditionalLinearAdjustment, EffectEstimate,
    EnvelopeOptions, EstimationWorkspace, FunctionalDistribution, FunctionalDistributionWorkspace,
    FunctionalEffect, GraphEffectDraws, OverlapPolicy, RdWorkspace, SharpRegressionDiscontinuity,
    TemporalLinearAdjustment, TemporalMediationEstimate, TemporalMediationEstimator,
    aggregate_effect_envelope, nonidentified_with_prior,
};
use causal_expr::{CausalExprArena, IdentifiedEstimand};
use causal_graph::{
    Admg, Dag, DenseNodeId, Pag, PagReview, TemporalCpdagReview, TemporalDag, TemporalGraphReview,
};
use causal_identify::{
    DerivationTrace, IdentificationPerformanceRecord, IdentificationResult, IdentificationStatus,
    IdentificationEnvelope, SharpRdConfig, SharpRdIdentifier, TemporalBackdoorIdentifier,
    TemporalMediationIdentifier,
};
use causal_prob::{GraphIdentFlag, InferenceDiagnostics, PriorSet, WeightedGraphSamples};
use causal_validate::{
    BayesianSuiteContext, ValidationSuite,
};

use crate::callback_plan::mark_python_callback_plan;
use crate::error::AnalysisError;
use crate::gcm::{
    anomaly_attribution, attribute_distribution_change, attribute_unit_change, counterfactual_ite,
    fit_gcm, mechanism_change_detection,
};
use crate::inference::{BayesianConfig, InferenceMode};
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
    DEFAULT_DISTRIBUTION_IDENTIFIER_ID, DEFAULT_ESTIMATOR, DEFAULT_ESTIMATOR_ID, DEFAULT_IDENTIFIER,
    DEFAULT_IDENTIFIER_ID, DEFAULT_PAG_ESTIMATOR_ID, DEFAULT_PAG_IDENTIFIER_ID, DEFAULT_PATH_ESTIMATOR,
    DEFAULT_PATH_ESTIMATOR_ID, DEFAULT_PATH_IDENTIFIER, DEFAULT_PATH_IDENTIFIER_ID, EstimatorId,
    IdentifierId, estimate_provenance_step, estimate_static_effect, identify_admg, identify_pag,
    identify_provenance_step, identify_static, identify_static_query, identify_static_query_with_rd,
    require_identified, select_estimand, validate_static_pair,
};

use super::builder::{CausalAnalysisBuilder, DataInput, RdConfig, RefuteSuite};
use super::helpers::{
    AssembleArgs, assemble_result, effect_from_posterior, overlap_diagnostic,
    provenance_pair, resolve_analysis_ci, run_jpcmci_plus_review, run_lpcmci_review,
    run_pcmci_plus_review, run_pcmci_review, run_fci_review, run_rfci_review, run_ges_review,
    run_lingam_review, run_notears_review, run_pc_review, run_refuters,
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
    pub(crate) discovery_ci: Option<Arc<dyn causal_stats::ConditionalIndependence + Send + Sync>>,
    pub(crate) custom_validators: Vec<Arc<dyn causal_validate::CustomEffectValidator>>,
}

impl std::fmt::Debug for CausalAnalysis {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CausalAnalysis")
            .field("graph", &self.graph)
            .field("refute", &self.refute)
            .field("bootstrap_replicates", &self.bootstrap_replicates)
            .field("identifier", &self.identifier)
            .field("estimator", &self.estimator)
            .field("discovery_ci", &self.discovery_ci.as_ref().map(|_| "<dyn CI>"))
            .field("custom_validators", &self.custom_validators.len())
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
    pub fn compile_logical(&self) -> Result<LogicalAnalysisPlan, AnalysisError> {
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
            (
                DataInput::Tabular(data),
                CausalQuery::Distribution(q),
                GraphInput::Static(graph),
            ) => {
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
            (
                DataInput::Tabular(data),
                CausalQuery::PathSpecific(q),
                GraphInput::Static(graph),
            ) => {
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
                compile_logical_temporal_effect_classified(
                    data, graph, q, self.split, false, class,
                )
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
                    data, &TemporalDag::empty(), q, self.split, true, class,
                )
            }
            (
                DataInput::MultiEnv(multi),
                CausalQuery::TemporalEffect(q),
                GraphInput::DiscoverJpcmciPlus { .. },
            ) => {
                let data = multi.environment(0).map_err(|e| AnalysisError::Compile {
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
                GraphInput::DiscoverJpcmciPlus { .. } | GraphInput::Temporal(_),
            ) => {
                let data = &panel.unit(0).map_err(|e| AnalysisError::Compile {
                    message: format!("panel: {e}"),
                })?.series;
                let review = matches!(self.graph, GraphInput::DiscoverJpcmciPlus { .. });
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
                q.validate().map_err(|e| AnalysisError::Compile { message: e.to_string() })?;
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
            (
                DataInput::Tabular(data),
                CausalQuery::Mediation(q),
                GraphInput::Static(graph),
            ) => {
                q.validate().map_err(|e| AnalysisError::Compile { message: e.to_string() })?;
                if !matches!(q.contrast, MediationContrast::Total) {
                    return Err(AnalysisError::Unsupported {
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
                self.query.validate().map_err(|e| AnalysisError::Compile {
                    message: e.to_string(),
                })?;
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
            _ => Err(AnalysisError::Unsupported {
                message: "unsupported data/graph/query combination",
            }),
        }
    }

    /// Compile logical → physical plan (or review-required).
    ///
    /// # Errors
    ///
    /// Modality / resource / discovery failures.
    pub fn compile(&self, ctx: &ExecutionContext) -> Result<CompiledAnalysis, AnalysisError> {
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
            (
                DataInput::Tabular(data),
                CausalQuery::Distribution(q),
                GraphInput::Static(graph),
            ) => {
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
            (
                DataInput::Tabular(data),
                CausalQuery::PathSpecific(q),
                GraphInput::Static(graph),
            ) => {
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
                let ci = resolve_analysis_ci(&self.discovery_ci)?;
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
                let ci = resolve_analysis_ci(&self.discovery_ci)?;
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
                let ci = resolve_analysis_ci(&self.discovery_ci)?;
                let review = run_jpcmci_plus_review(
                    multi,
                    *max_lag,
                    *alpha,
                    *fdr,
                    multi_dataset,
                    ci,
                    ctx,
                )?;
                let data = multi.environment(0).map_err(|e| AnalysisError::Compile {
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
                    .map_err(|e| AnalysisError::Compile {
                        message: format!("panel: {e}"),
                    })?
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
                let multi = panel.as_multi_env().map_err(|e| AnalysisError::Compile {
                    message: format!("panel as multi-env: {e}"),
                })?;
                let ci = resolve_analysis_ci(&self.discovery_ci)?;
                let review = run_jpcmci_plus_review(
                    &multi,
                    *max_lag,
                    *alpha,
                    *fdr,
                    multi_dataset,
                    ci,
                    ctx,
                )?;
                let data = &panel
                    .unit(0)
                    .map_err(|e| AnalysisError::Compile {
                        message: format!("panel: {e}"),
                    })?
                    .series;
                if *accept_discovered && review.pending_undirected.is_empty() {
                    let compiled = PendingCpdagReview::new(
                        review,
                        data.row_count(),
                        q.clone(),
                        self.split,
                    )
                    .accept_all_directed()
                    .finish(data, ctx)?;
                    Ok(mark_panel_classification(compiled))
                } else {
                    Ok(compile_review_required_cpdag(review))
                }
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
                let ci = resolve_analysis_ci(&self.discovery_ci)?;
                let result =
                    run_rpcmci_discovery(data, *max_lag, *alpha, *fdr, regime_assignment, ci, ctx)?;
                // Multi-regime estimation is not auto-wired; surface the first regime's CPDAG
                // for review. Auto-accept only when a single fully-oriented regime exists.
                let Some(first) = result.per_regime.first() else {
                    return Err(AnalysisError::Compile {
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
                CausalQuery::TemporalEffect(_),
                GraphInput::DiscoverLpcmci { max_lag, alpha, fdr, accept_discovered },
            ) => {
                let ci = resolve_analysis_ci(&self.discovery_ci)?;
                let review = run_lpcmci_review(data, *max_lag, *alpha, *fdr, ci, ctx)?;
                // Temporal backdoor is DAG-only; never auto-finish a PAG into Ready.
                let _ = accept_discovered;
                Ok(compile_review_required_pag(review))
            }
            (
                DataInput::Temporal(_) | DataInput::Event(_),
                CausalQuery::TemporalEffect(_),
                GraphInput::TemporalPag(pag),
            ) => {
                // Never silently estimate on a PAG with temporal.backdoor (DAG-only).
                Ok(compile_review_required_pag(causal_graph::TemporalPagReview::from_pag(
                    pag.clone(),
                    "supplied.temporal_pag",
                )))
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::TemporalEffect(q),
                GraphInput::TemporalCpdag(cpdag),
            ) => {
                match cpdag.try_into_temporal_dag() {
                    Ok(dag) => {
                        let logical = compile_logical_temporal_effect(
                            data,
                            &dag,
                            q,
                            self.split,
                            false,
                        )?;
                        let physical = logical.compile_physical_with_graph(ctx, Some(dag))?;
                        Ok(CompiledAnalysis::Ready(physical))
                    }
                    Err(_) => Ok(compile_review_required_cpdag(
                        causal_graph::TemporalCpdagReview::from_cpdag(
                            cpdag.clone(),
                            "supplied.temporal_cpdag",
                        ),
                    )),
                }
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverPc {
                    alpha,
                    max_cond_size,
                    fdr,
                    accept_discovered,
                },
            ) => {
                let ci = resolve_analysis_ci(&self.discovery_ci)?;
                let review = run_pc_review(data, *alpha, *max_cond_size, *fdr, ci, ctx)?;
                if *accept_discovered && review.pending_undirected.is_empty() {
                    let mut accepted = review;
                    accepted.pending_edges = Arc::from([]);
                    let dag = accepted.try_into_dag().map_err(|e| AnalysisError::ReviewRequired {
                        message: e.to_string(),
                    })?;
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
                    let physical =
                        logical.compile_physical_with_graphs(ctx, None, Some(dag))?;
                    Ok(CompiledAnalysis::Ready(physical))
                } else {
                    Ok(compile_review_required_static_cpdag(review))
                }
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverGes {
                    alpha,
                    max_cond_size,
                    fdr,
                    accept_discovered,
                },
            ) => {
                let ci = resolve_analysis_ci(&self.discovery_ci)?;
                let review = run_ges_review(data, *alpha, *max_cond_size, *fdr, ci, ctx)?;
                if *accept_discovered && review.pending_undirected.is_empty() {
                    let mut accepted = review;
                    accepted.pending_edges = Arc::from([]);
                    let dag = accepted.try_into_dag().map_err(|e| AnalysisError::ReviewRequired {
                        message: e.to_string(),
                    })?;
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
                    let physical =
                        logical.compile_physical_with_graphs(ctx, None, Some(dag))?;
                    Ok(CompiledAnalysis::Ready(physical))
                } else {
                    Ok(compile_review_required_static_cpdag(review))
                }
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                GraphInput::DiscoverLingam {
                    max_cond_size,
                    prune_threshold,
                    accept_discovered,
                },
            ) => {
                let review = run_lingam_review(data, *max_cond_size, *prune_threshold, ctx)?;
                if *accept_discovered {
                    let dag = review.accept_all().try_into_dag().map_err(|e| {
                        AnalysisError::ReviewRequired {
                            message: e.to_string(),
                        }
                    })?;
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
                    let physical =
                        logical.compile_physical_with_graphs(ctx, None, Some(dag))?;
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
                    let dag = review.accept_all().try_into_dag().map_err(|e| {
                        AnalysisError::ReviewRequired {
                            message: e.to_string(),
                        }
                    })?;
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
                    let physical =
                        logical.compile_physical_with_graphs(ctx, None, Some(dag))?;
                    Ok(CompiledAnalysis::Ready(physical))
                } else {
                    Ok(compile_review_required_static_dag(review))
                }
            }
            (
                DataInput::Tabular(data),
                CausalQuery::AverageEffect(q),
                graph @ GraphInput::DiscoverFci {
                    alpha,
                    max_cond_size,
                    fdr,
                    accept_discovered,
                },
            ) => {
                let ci = resolve_analysis_ci(&self.discovery_ci)?;
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
                graph @ GraphInput::DiscoverRfci {
                    alpha,
                    max_cond_size,
                    fdr,
                    accept_discovered,
                },
            ) => {
                let ci = resolve_analysis_ci(&self.discovery_ci)?;
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
                if !admg_has_bidirected(admg) {
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
                } else {
                    let (identifier, estimator) = self.resolve_admg_pair();
                    validate_static_pair(
                        IdentifierId::parse(&identifier),
                        EstimatorId::parse(&estimator),
                    )?;
                    q.validate().map_err(|e| AnalysisError::Compile { message: e.to_string() })?;
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
                let physical = logical.compile_physical_with_graphs(ctx, None, Some(graph.clone()))?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (
                DataInput::Temporal(data) | DataInput::Event(data),
                CausalQuery::Mediation(q),
                GraphInput::Temporal(graph),
            ) => {
                q.validate().map_err(|e| AnalysisError::Compile { message: e.to_string() })?;
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
                let physical =
                    logical.compile_physical_with_graph(ctx, Some(graph.clone()))?;
                Ok(CompiledAnalysis::Ready(physical))
            }
            (
                DataInput::Tabular(data),
                CausalQuery::Mediation(q),
                GraphInput::Static(graph),
            ) => {
                q.validate().map_err(|e| AnalysisError::Compile { message: e.to_string() })?;
                if !matches!(q.contrast, MediationContrast::Total) {
                    return Err(AnalysisError::Unsupported {
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
                let physical = logical.compile_physical_with_graphs(ctx, None, Some(graph.clone()))?;
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
            _ => Err(AnalysisError::Unsupported {
                message: "unsupported data/graph/query combination",
            }),
        }
    }

    fn validation_suite_id(&self) -> Option<Arc<str>> {
        match self.refute {
            RefuteSuite::None => None,
            RefuteSuite::PlaceboAndRcc => Some(Arc::from("placebo+rcc")),
            RefuteSuite::Full => Some(Arc::from("validation.full")),
        }
    }

    fn ensure_supported_combination(&self) -> Result<(), AnalysisError> {
        match (&self.data, &self.query, &self.graph) {
            (DataInput::Tabular(_), CausalQuery::Distribution(_), GraphInput::Static(_)) => {}
            (_, CausalQuery::Distribution(_), _) => {
                return Err(AnalysisError::Unsupported {
                    message: "CausalQuery::Distribution requires tabular data and a static DAG",
                });
            }
            (DataInput::Tabular(_), CausalQuery::PathSpecific(_), GraphInput::Static(_)) => {}
            (_, CausalQuery::PathSpecific(_), _) => {
                return Err(AnalysisError::Unsupported {
                    message: "CausalQuery::PathSpecific requires tabular data and a static DAG",
                });
            }
            (DataInput::Tabular(_), CausalQuery::TemporalEffect(_), _) => {
                return Err(AnalysisError::Compile {
                    message: "temporal effect query requires temporal data".into(),
                });
            }
            (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::AverageEffect(_), _) => {
                return Err(AnalysisError::Compile {
                    message: "static ATE on temporal data is unsupported; use TemporalEffect"
                        .into(),
                });
            }
            (DataInput::MultiEnv(_), CausalQuery::AverageEffect(_), _) => {
                return Err(AnalysisError::Compile {
                    message: "static ATE on temporal data is unsupported; use TemporalEffect"
                        .into(),
                });
            }
            (DataInput::Panel(_), CausalQuery::AverageEffect(_), _) => {
                return Err(AnalysisError::Compile {
                    message: "static ATE on panel data is unsupported; use TemporalEffect"
                        .into(),
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
                return Err(AnalysisError::Compile {
                    message:
                        "PCMCI-family / temporal PAG discovery requires temporal data and a temporal effect query"
                            .into(),
                });
            }
            (DataInput::Temporal(_) | DataInput::Event(_), _, GraphInput::DiscoverPc { .. })
            | (DataInput::MultiEnv(_), _, GraphInput::DiscoverPc { .. })
            | (DataInput::Temporal(_) | DataInput::Event(_), _, GraphInput::DiscoverGes { .. })
            | (DataInput::MultiEnv(_), _, GraphInput::DiscoverGes { .. })
            | (DataInput::Temporal(_) | DataInput::Event(_), _, GraphInput::DiscoverLingam { .. })
            | (DataInput::MultiEnv(_), _, GraphInput::DiscoverLingam { .. })
            | (DataInput::Temporal(_) | DataInput::Event(_), _, GraphInput::DiscoverNotears { .. })
            | (DataInput::MultiEnv(_), _, GraphInput::DiscoverNotears { .. })
            | (DataInput::Temporal(_) | DataInput::Event(_), _, GraphInput::DiscoverFci { .. })
            | (DataInput::MultiEnv(_), _, GraphInput::DiscoverFci { .. })
            | (DataInput::Temporal(_) | DataInput::Event(_), _, GraphInput::DiscoverRfci { .. })
            | (DataInput::MultiEnv(_), _, GraphInput::DiscoverRfci { .. }) => {
                return Err(AnalysisError::Compile {
                    message: "static PC/GES/LiNGAM/NOTEARS/FCI/RFCI discovery requires tabular data and AverageEffect"
                        .into(),
                });
            }
            (DataInput::Temporal(_) | DataInput::Event(_), _, GraphInput::DiscoverJpcmciPlus { .. }) => {
                return Err(AnalysisError::Compile {
                    message: "J-PCMCI+ discovery requires series_multi (MultiEnvironmentData) or panel"
                        .into(),
                });
            }
            (DataInput::MultiEnv(_), _, graph) if !matches!(graph, GraphInput::DiscoverJpcmciPlus { .. }) => {
                return Err(AnalysisError::Compile {
                    message: "multi-environment data currently supports only DiscoverJpcmciPlus"
                        .into(),
                });
            }
            (DataInput::Panel(_), _, graph)
                if !matches!(
                    graph,
                    GraphInput::DiscoverJpcmciPlus { .. } | GraphInput::Temporal(_)
                ) =>
            {
                return Err(AnalysisError::Compile {
                    message: "panel data supports DiscoverJpcmciPlus or a supplied TemporalDag"
                        .into(),
                });
            }
            (DataInput::Temporal(_) | DataInput::Event(_), _, GraphInput::Pag(_)) => {
                return Err(AnalysisError::Compile {
                    message: "static Pag requires tabular data and an average-effect query".into(),
                });
            }
            (DataInput::MultiEnv(_), _, GraphInput::Pag(_)) => {
                return Err(AnalysisError::Compile {
                    message: "static Pag requires tabular data and an average-effect query".into(),
                });
            }
            (DataInput::Tabular(_), CausalQuery::AverageEffect(_), GraphInput::Pag(_)) => {
                let (identifier, _) = self.resolve_pag_pair();
                reject_dag_only_on_pag(&self.graph, IdentifierId::parse(&identifier))?;
            }
            (DataInput::Tabular(_), CausalQuery::ConditionalEffect(_), GraphInput::Static(_)) => {}
            (DataInput::Temporal(_) | DataInput::Event(_), CausalQuery::Mediation(_), GraphInput::Temporal(_)) => {}
            (DataInput::Panel(_), CausalQuery::TemporalEffect(_), GraphInput::Temporal(_)) => {}
            (DataInput::Panel(_), CausalQuery::TemporalEffect(_), GraphInput::DiscoverJpcmciPlus { .. }) => {}
            (DataInput::Tabular(_), CausalQuery::Mediation(_), GraphInput::Static(_)) => {}
            (
                DataInput::Tabular(_),
                CausalQuery::Counterfactual(_)
                | CausalQuery::AnomalyAttribution(_)
                | CausalQuery::ChangeAttribution(_)
                | CausalQuery::MechanismChange(_)
                | CausalQuery::UnitChange(_),
                GraphInput::Static(_),
            ) => {}
            _ => {}
        }
        // The temporal path is linear/temporal-backdoor only; refuse an explicitly
        // selected non-temporal identifier/estimator rather than silently ignoring it.
        if matches!(&self.query, CausalQuery::TemporalEffect(_)) {
            if let Some(id) = &self.identifier {
                if *id != IdentifierId::TemporalBackdoorUnfolded {
                    return Err(AnalysisError::Compile {
                        message: format!(
                            "temporal path only supports identifier \"temporal.backdoor.unfolded\"; got {id:?}"
                        ),
                    });
                }
            }
            if let Some(est) = &self.estimator {
                if *est != EstimatorId::TemporalLinearAdjustment {
                    return Err(AnalysisError::Compile {
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

    fn ensure_rd_config_present(&self, estimator: &str) -> Result<(), AnalysisError> {
        if matches!(EstimatorId::parse(estimator), EstimatorId::RdSharp) && self.rd.is_none() {
            return Err(AnalysisError::Compile {
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
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let CompiledAnalysis::Ready(physical) = plan else {
            return Err(AnalysisError::ReviewRequired {
                message: "cannot execute while graph review is required".into(),
            });
        };
        ensure_review_complete(&physical.logical)?;
        match (&self.data, &self.query) {
            (DataInput::Tabular(data), CausalQuery::AverageEffect(q)) => {
                match &self.graph {
                    GraphInput::Static(graph) => {
                        self.execute_static(data, graph, q, physical, ctx)
                    }
                    GraphInput::DiscoverPc { .. }
                    | GraphInput::DiscoverGes { .. }
                    | GraphInput::DiscoverLingam { .. }
                    | GraphInput::DiscoverNotears { .. }
                    | GraphInput::Cpdag(_) => {
                        let graph = physical.static_graph().ok_or(AnalysisError::Compile {
                            message:
                                "Ready PC/GES/LiNGAM/NOTEARS/CPDAG plan missing resolved static DAG (complete review first)"
                                    .into(),
                        })?;
                        self.execute_static(data, graph, q, physical, ctx)
                    }
                    GraphInput::Admg(admg) => {
                        if !admg_has_bidirected(admg) {
                            let graph = physical.static_graph().ok_or(AnalysisError::Compile {
                                message: "Ready ADMG (DAG-coerced) plan missing resolved static DAG"
                                    .into(),
                            })?;
                            self.execute_static(data, graph, q, physical, ctx)
                        } else {
                            self.execute_admg(data, admg, q, physical, ctx)
                        }
                    }
                    GraphInput::Pag(_)
                    | GraphInput::DiscoverFci { .. }
                    | GraphInput::DiscoverRfci { .. } => {
                        let pag = physical.static_pag().ok_or(AnalysisError::Compile {
                            message:
                                "Ready PAG plan missing resolved static PAG (complete review first)"
                                    .into(),
                        })?;
                        self.execute_pag(data, pag, q, physical, ctx)
                    }
                    _ => Err(AnalysisError::Unsupported {
                        message:
                            "static ATE execute requires a supplied static DAG/PAG/CPDAG/ADMG or DiscoverPc/Ges/Lingam/Notears/Fci/Rfci",
                    }),
                }
            }
            (DataInput::Tabular(data), CausalQuery::Distribution(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(AnalysisError::Unsupported {
                        message: "Distribution execute requires a supplied static DAG",
                    });
                };
                self.execute_distribution(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::PathSpecific(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(AnalysisError::Unsupported {
                        message: "PathSpecific execute requires a supplied static DAG",
                    });
                };
                self.execute_path_specific(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::ConditionalEffect(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(AnalysisError::Unsupported {
                        message: "ConditionalEffect execute requires a supplied static DAG",
                    });
                };
                self.execute_conditional(data, graph, q, physical, ctx)
            }
            (DataInput::Temporal(data) | DataInput::Event(data), CausalQuery::Mediation(q)) => {
                let graph = physical.temporal_graph().ok_or(AnalysisError::Compile {
                    message: "Ready temporal mediation plan missing resolved graph".into(),
                })?;
                self.execute_temporal_mediation(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::Mediation(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(AnalysisError::Unsupported {
                        message: "static Mediation execute requires a supplied static DAG",
                    });
                };
                self.execute_static_mediation_total(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::Counterfactual(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(AnalysisError::Unsupported {
                        message: "Counterfactual execute requires a supplied static DAG",
                    });
                };
                self.execute_counterfactual(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::AnomalyAttribution(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(AnalysisError::Unsupported {
                        message: "AnomalyAttribution execute requires a supplied static DAG",
                    });
                };
                self.execute_anomaly(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::ChangeAttribution(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(AnalysisError::Unsupported {
                        message: "ChangeAttribution execute requires a supplied static DAG",
                    });
                };
                self.execute_change_attribution(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::MechanismChange(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(AnalysisError::Unsupported {
                        message: "MechanismChange execute requires a supplied static DAG",
                    });
                };
                self.execute_mechanism_change(data, graph, q, physical, ctx)
            }
            (DataInput::Tabular(data), CausalQuery::UnitChange(q)) => {
                let GraphInput::Static(graph) = &self.graph else {
                    return Err(AnalysisError::Unsupported {
                        message: "UnitChange execute requires a supplied static DAG",
                    });
                };
                self.execute_unit_change(data, graph, q, physical, ctx)
            }
            (DataInput::Temporal(data) | DataInput::Event(data), CausalQuery::TemporalEffect(q)) => {
                let graph = physical.temporal_graph().ok_or(AnalysisError::Compile {
                    message: "Ready temporal plan missing resolved graph (complete review first)"
                        .into(),
                })?;
                self.execute_temporal(data, graph, q, physical, ctx)
            }
            (DataInput::Panel(panel), CausalQuery::TemporalEffect(q)) => {
                let graph = physical.temporal_graph().ok_or(AnalysisError::Compile {
                    message: "Ready panel plan missing resolved graph (complete review first)"
                        .into(),
                })?;
                self.execute_panel(panel, graph, q, physical, ctx)
            }
            _ => Err(AnalysisError::Unsupported {
                message: "execute path unsupported for this configuration",
            }),
        }
    }

    /// Compile and run when Ready; error if review is required.
    ///
    /// # Errors
    ///
    /// Compile / review / execute failures.
    pub fn run(&self, ctx: &ExecutionContext) -> Result<CausalAnalysisResult, AnalysisError> {
        let compiled = self.compile(ctx)?;
        self.execute(&compiled, ctx)
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
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let (DataInput::Temporal(data) | DataInput::Event(data)) = &self.data else {
            return Err(AnalysisError::Compile {
                message: "finish_review_and_run requires temporal data".into(),
            });
        };
        let CausalQuery::TemporalEffect(q) = &self.query else {
            return Err(AnalysisError::Compile {
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
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let (DataInput::Temporal(data) | DataInput::Event(data)) = &self.data else {
            return Err(AnalysisError::Compile {
                message: "finish_cpdag_review_and_run requires temporal data".into(),
            });
        };
        let CausalQuery::TemporalEffect(q) = &self.query else {
            return Err(AnalysisError::Compile {
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
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let started = Instant::now();
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
        let estimand = select_estimand(&identification, estimator_id)?;
        let assumptions = identification.required_assumptions.clone();

        let estimate = estimate_static_effect(
            estimator,
            data,
            &estimand,
            query,
            assumptions,
            self.bootstrap_replicates,
            self.overlap_policy,
            self.population_registry.as_ref(),
            ctx,
        )?;

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));

        // ValidationSuite skips incompatible validators with NotApplicable rather than failing.
        let refutations = {
            let mut refute_ws = EstimationWorkspace::default();
            run_refuters(
                data,
                &estimand,
                query,
                &estimate,
                &mut refute_ws,
                ctx,
                self.refute,
                estimator,
                &self.custom_validators,
            )?
        };

        let (id_artifact, id_op) = identify_provenance_step(identifier);
        let (est_artifact, est_op) = estimate_provenance_step(estimator);
        let provenance = provenance_pair(
            (id_artifact, id_op, &[], &identification.required_assumptions),
            (est_artifact, est_op, &[id_artifact], &estimate.assumptions),
        );

        let physical_record = self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
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
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let started = Instant::now();
        let identifier = physical
            .logical
            .record
            .identifier
            .as_deref()
            .unwrap_or(DEFAULT_DISTRIBUTION_IDENTIFIER);
        let estimator = physical
            .logical
            .record
            .estimator
            .as_deref()
            .unwrap_or(DEFAULT_DISTRIBUTION_ESTIMATOR);
        if !matches!(EstimatorId::parse(estimator), EstimatorId::FunctionalDistribution) {
            return Err(AnalysisError::Compile {
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
            .prepare(data, query, &estimand, &identification.arena, identification.required_assumptions.clone())
            .map_err(AnalysisError::from)?;
        let mut ws = FunctionalDistributionWorkspace::default();
        let dist = est.estimate(&prepared, &[], &mut ws, ctx).map_err(AnalysisError::from)?;

        let estimate = EffectEstimate {
            ate: dist.mean,
            se_analytic: dist.se_analytic,
            se_bootstrap: dist.se_bootstrap,
            bootstrap_replicates_ok: dist.bootstrap_replicates_ok,
            bootstrap_replicates_failed: dist.bootstrap_replicates_failed,
            assumptions: dist.assumptions.clone(),
            overlap: dist.overlap,
            overlap_report: None,
            retained_memory_bytes: dist.retained_memory_bytes,
        };

        let treatment = query
            .interventions
            .first()
            .and_then(|iv| iv.primary_variable())
            .ok_or_else(|| AnalysisError::Compile {
                message: "distribution query missing intervention target".into(),
            })?;
        let outcome = *query.outcomes.first().ok_or_else(|| AnalysisError::Compile {
            message: "distribution query missing outcome".into(),
        })?;

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
                ctx,
                self.refute,
                estimator,
                &self.custom_validators,
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

        let physical_record = self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
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
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let started = Instant::now();
        let identifier =
            physical.logical.record.identifier.as_deref().unwrap_or(DEFAULT_PATH_IDENTIFIER);
        let estimator =
            physical.logical.record.estimator.as_deref().unwrap_or(DEFAULT_PATH_ESTIMATOR);
        if !matches!(EstimatorId::parse(estimator), EstimatorId::FunctionalEffect) {
            return Err(AnalysisError::Compile {
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
            .map_err(AnalysisError::from)?;
        let mut ws = FunctionalDistributionWorkspace::default();
        let estimate = est.estimate(&prepared, &mut ws, ctx).map_err(AnalysisError::from)?;

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
            ctx,
            self.refute,
            estimator,
            &self.custom_validators,
        )?;

        let (id_artifact, id_op) = identify_provenance_step(identifier);
        let (est_artifact, est_op) = estimate_provenance_step(estimator);
        let provenance = provenance_pair(
            (id_artifact, id_op, &[], &identification.required_assumptions),
            (est_artifact, est_op, &[id_artifact], &estimate.assumptions),
        );

        let physical_record = self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
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
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let started = Instant::now();
        let identifier =
            physical.logical.record.identifier.as_deref().unwrap_or(DEFAULT_IDENTIFIER);
        let identification = identify_static(identifier, graph, query)?;
        let estimand = select_estimand(&identification, EstimatorId::BayesianGcomp)?;

        let cfg = match &self.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => BayesianConfig::laplace(),
        };
        let est = BayesianGComputationAte {
            backend: cfg.backend,
            likelihood: cfg.likelihood,
            n_draws: cfg.n_draws,
            seed: ctx.rng.master_seed(),
            overlap: OverlapPolicy::ExplicitOverride,
            prior_scale: cfg.prior_scale,
        };
        let prep = est.prepare(data, &estimand, query).map_err(AnalysisError::from)?;
        let mut ws = BayesianGCompWorkspace::default();
        let posterior =
            est.fit(&prep, identification.status, &mut ws, ctx).map_err(AnalysisError::from)?;
        let estimate = effect_from_posterior(&posterior)?;

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));

        let mut refute_ws = EstimationWorkspace::default();
        let mut refutations = match self.refute {
            RefuteSuite::None => Vec::new(),
            RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => run_refuters(
                data,
                &estimand,
                query,
                &estimate,
                &mut refute_ws,
                ctx,
                self.refute,
                "bayesian.gcomp",
                &self.custom_validators,
            )?,
        };
        if matches!(self.refute, RefuteSuite::Full) {
            let suite = ValidationSuite::bayesian_diagnostics();
            let mut bayes_ctx = BayesianSuiteContext::new(
                &est,
                &prep,
                &posterior,
                identification.status,
                &mut ws,
                estimate.ate,
            );
            let outcomes = suite.run_bayesian(&mut bayes_ctx, ctx).map_err(AnalysisError::from)?;
            refutations.extend(ValidationSuite::reports_only(&outcomes));
        }

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

        let physical_record = self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
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
        }))
    }

    /// `rd.sharp` execute path: identify via [`SharpRdIdentifier`], then estimate.
    fn execute_rd(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let started = Instant::now();
        let rd = self.rd.ok_or_else(|| AnalysisError::Compile {
            message: "estimator \"rd.sharp\" requires builder.rd_config(running_variable, cutoff, bandwidth)".into(),
        })?;
        let identification = SharpRdIdentifier::new(SharpRdConfig {
            running_variable: rd.running_variable,
            cutoff: rd.cutoff,
            bandwidth: rd.bandwidth,
        })
        .identify(CausalQuery::AverageEffect(query.clone()))
        .map_err(AnalysisError::from)?;
        require_identified(&identification)?;
        let estimand = select_estimand(&identification, EstimatorId::RdSharp)?;

        let mut est =
            SharpRegressionDiscontinuity::new(rd.running_variable, rd.cutoff, rd.bandwidth);
        est.bootstrap_replicates = self.bootstrap_replicates;
        let prep = est.prepare(data, &estimand, query).map_err(AnalysisError::from)?;
        let mut ws = RdWorkspace::default();
        let estimate = est
            .fit(&prep, &mut ws, ctx, identification.required_assumptions.clone())
            .map_err(AnalysisError::from)?;

        let mut diagnostics = vec![overlap_diagnostic(estimate.overlap)];
        let mut refute_ws = EstimationWorkspace::default();
        let refutations = run_refuters(
            data,
            &estimand,
            query,
            &estimate,
            &mut refute_ws,
            ctx,
            self.refute,
            "rd.sharp",
            &self.custom_validators,
        )?;

        let provenance = provenance_pair(
            ("identify.rd_design", "identify.rd_sharp", &[], &identification.required_assumptions),
            ("estimate.rd", "estimate.rd_sharp", &["identify.rd_design"], &estimate.assumptions),
        );

        let physical_record = self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
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
        }))
    }

    fn execute_temporal(
        &self,
        data: &TimeSeriesData,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let started = Instant::now();
        let id_res = TemporalBackdoorIdentifier::new()
            .identify_temporal(graph, query)
            .map_err(AnalysisError::from)?;
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
            .map_err(AnalysisError::from)?;
        let mut workspace = EstimationWorkspace::default();
        let estimate = estimator
            .fit(&prep, &mut workspace, ctx, identification.required_assumptions.clone())
            .map_err(AnalysisError::from)?;

        let provenance = provenance_pair(
            (
                "identify.temporal_backdoor",
                "identify.temporal.backdoor.unfolded",
                &[],
                &identification.required_assumptions,
            ),
            (
                "estimate.temporal_linear_adjustment",
                "estimate.temporal.linear.adjustment",
                &["identify.temporal_backdoor"],
                &estimate.assumptions,
            ),
        );

        let mut diagnostics = Vec::new();
        let tabular = TabularData::new(data.storage().clone());
        let ate_q = AverageEffectQuery::binary_ate(query.treatment, query.outcome);
        let mut refute_ws = EstimationWorkspace::default();
        let refutations = run_refuters(
            &tabular,
            &estimand,
            &ate_q,
            &estimate,
            &mut refute_ws,
            ctx,
            self.refute,
            "temporal.linear.adjustment",
            &self.custom_validators,
        )?;

        let physical_record = self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
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
        }))
    }

    /// Panel temporal effect: identify on the shared graph, estimate on stacked units
    /// with [`AnalyticSeKind::PanelClusterHac`] and per-unit `cluster_ids`.
    fn execute_panel(
        &self,
        panel: &PanelData,
        graph: &TemporalDag,
        query: &TemporalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let started = Instant::now();
        let id_res = TemporalBackdoorIdentifier::new()
            .identify_temporal(graph, query)
            .map_err(AnalysisError::from)?;
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
            .map_err(AnalysisError::from)?;
        let max_lag = query.max_history_lag.unwrap_or(1).max(1) as usize;
        estimator.inner.cluster_ids = Some(cluster_ids);
        estimator.inner.se_kind = AnalyticSeKind::PanelClusterHac { lag: max_lag };
        let mut workspace = EstimationWorkspace::default();
        let estimate = estimator
            .fit(&prep, &mut workspace, ctx, identification.required_assumptions.clone())
            .map_err(AnalysisError::from)?;

        let provenance = provenance_pair(
            (
                "identify.temporal_backdoor",
                "identify.temporal.backdoor.unfolded",
                &[],
                &identification.required_assumptions,
            ),
            (
                "estimate.temporal_linear_adjustment.panel",
                "estimate.temporal.linear.adjustment.panel",
                &["identify.temporal_backdoor"],
                &estimate.assumptions,
            ),
        );

        let mut diagnostics = Vec::new();
        let physical_record = self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
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
            refutations: Vec::new(),
            diagnostics,
            provenance,
            treatment: query.treatment,
            outcome: query.outcome,
            wall_time_ns: u64::try_from(started.elapsed().as_nanos()).unwrap_or(u64::MAX),
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
    ) -> Result<CausalAnalysisResult, AnalysisError> {
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
            return Err(AnalysisError::Compile {
                message: format!(
                    "ADMG ATE requires estimator functional.effect; got {estimator}"
                ),
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
            .map_err(AnalysisError::from)?;
        let mut ws = FunctionalDistributionWorkspace::default();
        let estimate = est.estimate(&prepared, &mut ws, ctx).map_err(AnalysisError::from)?;

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));

        let mut refute_ws = EstimationWorkspace::default();
        let refutations = run_refuters(
            data,
            &estimand,
            query,
            &estimate,
            &mut refute_ws,
            ctx,
            self.refute,
            estimator,
            &self.custom_validators,
        )?;

        let (id_artifact, id_op) = identify_provenance_step(identifier);
        let (est_artifact, est_op) = estimate_provenance_step(estimator);
        let provenance = provenance_pair(
            (id_artifact, id_op, &[], &identification.required_assumptions),
            (est_artifact, est_op, &[id_artifact], &estimate.assumptions),
        );

        let physical_record = self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
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
    ) -> Result<CausalAnalysisResult, AnalysisError> {
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
                return self.execute_pag_nonidentified_prior(
                    query,
                    physical,
                    ctx,
                    &envelope,
                    started,
                );
            }
            return Err(AnalysisError::Compile {
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
            if !identification_status_ok_for_case(&case.result.status)
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
        if !(total_w > 0.0) {
            return Err(AnalysisError::Compile {
                message: "PAG envelope had no estimable identified cases".into(),
            });
        }
        let estimand = primary_estimand.ok_or_else(|| AnalysisError::Compile {
            message: "PAG envelope missing estimand".into(),
        })?;
        let estimate = EffectEstimate {
            ate: weighted_ate / total_w,
            se_analytic: (weighted_se2 / total_w).sqrt(),
            se_bootstrap: None,
            bootstrap_replicates_ok: None,
            bootstrap_replicates_failed: None,
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
            ctx,
            self.refute,
            estimator,
            &self.custom_validators,
        )?;

        let (id_artifact, id_op) = identify_provenance_step(identifier);
        let (est_artifact, est_op) = estimate_provenance_step(estimator);
        let provenance = provenance_pair(
            (id_artifact, id_op, &[], &identification.required_assumptions),
            (est_artifact, est_op, &[id_artifact], &estimate.assumptions),
        );
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        let physical_record = self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
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
    ) -> Result<CausalAnalysisResult, AnalysisError> {
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
            cfg.n_draws.max(1) as usize,
            ctx.rng.master_seed(),
        );
        let estimate = effect_from_posterior(&posterior)?;
        let identification = envelope_to_identification_result(envelope, query);
        let estimand = envelope.invariant.clone().unwrap_or_else(|| {
            IdentifiedEstimand::backdoor("pag.nonidentified", Arc::from([]), causal_expr::ExprId::from_raw(0))
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
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let cfg = match &self.inference {
            InferenceMode::Bayesian(c) => c.clone(),
            InferenceMode::Frequentist => BayesianConfig::laplace(),
        };
        let est = BayesianGComputationAte {
            backend: cfg.backend,
            likelihood: cfg.likelihood,
            n_draws: cfg.n_draws,
            seed: ctx.rng.master_seed(),
            overlap: OverlapPolicy::ExplicitOverride,
            prior_scale: cfg.prior_scale,
        };

        let mut weights = Vec::new();
        let mut flags = Vec::new();
        let mut keys = Vec::new();
        let mut per_graph = Vec::new();
        let mut primary_estimand: Option<IdentifiedEstimand> = None;
        for (i, case) in envelope.cases.iter().enumerate() {
            let key = i as u64 + 1;
            keys.push(key);
            weights.push(case.weight.0);
            if identification_status_ok_for_case(&case.result.status)
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
                let prep = est.prepare(data, &estimand, query).map_err(AnalysisError::from)?;
                let mut ws = BayesianGCompWorkspace::default();
                let posterior = est
                    .fit(&prep, case.result.status, &mut ws, ctx)
                    .map_err(AnalysisError::from)?;
                let col = posterior.effect_column().ok_or_else(|| AnalysisError::Compile {
                    message: "Bayesian posterior missing effect column".into(),
                })?;
                let draws = posterior
                    .draws
                    .column(col)
                    .map_err(|e| AnalysisError::Compile { message: e.to_string() })?
                    .to_vec();
                per_graph.push(GraphEffectDraws {
                    graph_key: key,
                    effect_draws: Arc::from(draws),
                });
            } else {
                flags.push(GraphIdentFlag::Unidentified);
            }
        }
        let graphs = WeightedGraphSamples::new(weights, flags, keys)
            .map_err(|e| AnalysisError::Compile { message: e.to_string() })?;
        let posterior = aggregate_effect_envelope(
            &graphs,
            &per_graph,
            InferenceDiagnostics::analytic("pag_envelope"),
            EnvelopeOptions::default(),
        )
        .map_err(AnalysisError::from)?;
        let estimate = effect_from_posterior(&posterior)?;
        let estimand = primary_estimand.or(envelope.invariant.clone()).ok_or_else(|| {
            AnalysisError::Compile { message: "PAG Bayesian envelope missing estimand".into() }
        })?;

        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        diagnostics.push(Diagnostic::new(
            "estimate.pag.envelope",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!("unidentified_mass={}", posterior.unidentified_mass),
        ));

        let mut refute_ws = EstimationWorkspace::default();
        let refutations = match self.refute {
            RefuteSuite::None => Vec::new(),
            RefuteSuite::PlaceboAndRcc | RefuteSuite::Full => run_refuters(
                data,
                &estimand,
                query,
                &estimate,
                &mut refute_ws,
                ctx,
                self.refute,
                "bayesian.gcomp",
                &self.custom_validators,
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
        let physical_record = self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
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
        }))
    }

    fn execute_conditional(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &causal_core::ConditionalEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let started = Instant::now();
        let (identifier, _) = self.resolve_conditional_pair();
        let identification = identify_static(identifier.as_ref(), graph, &query.inner)?;
        let estimand = select_estimand(&identification, EstimatorId::ConditionalLinearAdjustment)?;
        let est = ConditionalLinearAdjustment::new();
        let estimate = est
            .estimate(data, &estimand, query, ctx)
            .map_err(AnalysisError::from)?;
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        let mut refute_ws = EstimationWorkspace::default();
        let refutations = run_refuters(
            data,
            &estimand,
            &query.inner,
            &estimate,
            &mut refute_ws,
            ctx,
            self.refute,
            "conditional.linear.adjustment",
            &self.custom_validators,
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
        let physical_record = self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
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
        }))
    }

    fn execute_temporal_mediation(
        &self,
        data: &TimeSeriesData,
        graph: &TemporalDag,
        query: &causal_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let started = Instant::now();
        let identification = TemporalMediationIdentifier {
            allow_natural_controlled_alias: true,
            ..TemporalMediationIdentifier::new()
        }
        .identify(graph, query)
        .map_err(AnalysisError::from)?;
        require_identified(&identification)?;
        let estimand = select_estimand(&identification, EstimatorId::TemporalMediation)?;
        let mut est = TemporalMediationEstimator::new();
        est.allow_natural_controlled_alias = true;
        let mediation = est
            .estimate(data, &estimand, query, ctx)
            .map_err(AnalysisError::from)?;
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
        let physical_record = self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
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
        }))
    }

    fn execute_static_mediation_total(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &causal_core::MediationQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let started = Instant::now();
        if !matches!(query.contrast, MediationContrast::Total) {
            return Err(AnalysisError::Unsupported {
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
        )?;
        let mediation = TemporalMediationEstimate {
            effect: estimate.clone(),
            total: Some(estimate.ate),
            direct: None,
            mediated: None,
        };
        let mut diagnostics = identification.diagnostics.clone();
        diagnostics.push(overlap_diagnostic(estimate.overlap));
        let mut refute_ws = EstimationWorkspace::default();
        let refutations = run_refuters(
            data,
            &estimand,
            &ate,
            &estimate,
            &mut refute_ws,
            ctx,
            self.refute,
            "frontdoor.two_stage",
            &self.custom_validators,
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
        let physical_record = self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
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
        }))
    }

    fn execute_counterfactual(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &causal_core::CounterfactualQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let started = Instant::now();
        query.validate().map_err(|e| AnalysisError::Compile { message: e.to_string() })?;
        let fitted = fit_gcm(graph.clone(), data)?;
        let outcome = *query.outcomes.first().ok_or_else(|| AnalysisError::Compile {
            message: "counterfactual query missing outcome".into(),
        })?;
        let (treatment, active, control) = binary_cf_interventions(query)?;
        let ite = counterfactual_ite(
            fitted.model,
            data,
            treatment,
            outcome,
            active,
            control,
            ctx,
        )?;
        let estimate = EffectEstimate {
            ate: ite.mean_ite,
            se_analytic: f64::NAN,
            se_bootstrap: None,
            bootstrap_replicates_ok: None,
            bootstrap_replicates_failed: None,
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
        let physical_record = self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
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
        }))
    }

    fn execute_anomaly(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &causal_core::AnomalyAttributionQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let _ = ctx;
        let started = Instant::now();
        query.validate().map_err(|e| AnalysisError::Compile { message: e.to_string() })?;
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
        let physical_record = self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
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
        }))
    }

    fn execute_change_attribution(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &causal_core::ChangeAttributionQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let started = Instant::now();
        query.validate().map_err(|e| AnalysisError::Compile { message: e.to_string() })?;
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
        let physical_record = self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
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
        }))
    }

    fn execute_mechanism_change(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &causal_core::MechanismChangeQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let started = Instant::now();
        query.validate().map_err(|e| AnalysisError::Compile { message: e.to_string() })?;
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
        let physical_record = self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
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
        }))
    }

    fn execute_unit_change(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &causal_core::UnitChangeQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let started = Instant::now();
        query.validate().map_err(|e| AnalysisError::Compile { message: e.to_string() })?;
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
        let physical_record = self.apply_callback_plan_marks(physical.record.clone(), &mut diagnostics);
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
    ) -> Result<CausalAnalysisResult, AnalysisError> {
        let DataInput::Tabular(data) = &self.data else {
            return Err(AnalysisError::Compile {
                message: "finish_static_pag_review_and_run requires tabular data".into(),
            });
        };
        let CausalQuery::AverageEffect(q) = &self.query else {
            return Err(AnalysisError::Compile {
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

fn gcm_query_vars(query: &CausalQuery) -> Result<(VariableId, VariableId), AnalysisError> {
    match query {
        CausalQuery::Counterfactual(q) => {
            let outcome = *q.outcomes.first().ok_or_else(|| AnalysisError::Compile {
                message: "counterfactual missing outcome".into(),
            })?;
            let treatment = q
                .interventions
                .first()
                .and_then(|iv| iv.primary_variable())
                .unwrap_or(outcome);
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
        _ => Err(AnalysisError::Compile {
            message: "gcm_query_vars: unsupported query".into(),
        }),
    }
}

fn nan_effect() -> EffectEstimate {
    EffectEstimate {
        ate: f64::NAN,
        se_analytic: f64::NAN,
        se_bootstrap: None,
        bootstrap_replicates_ok: None,
        bootstrap_replicates_failed: None,
        assumptions: causal_core::AssumptionSet::default(),
        overlap: OverlapPolicy::ExplicitOverride,
        overlap_report: None,
        retained_memory_bytes: None,
    }
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
) -> Result<(VariableId, f64, f64), AnalysisError> {
    if query.interventions.len() != 1 {
        return Err(AnalysisError::Unsupported {
            message: "CausalAnalysis counterfactual path currently supports a single hard \
                 intervention for ITE (use gcm helpers for multi-world predict)",
        });
    }
    let Intervention::Set { variable, value } = &query.interventions[0] else {
        return Err(AnalysisError::Unsupported {
            message: "CausalAnalysis counterfactual path requires a hard Set intervention",
        });
    };
    let active = value.as_f64().ok_or_else(|| AnalysisError::Compile {
        message: "counterfactual intervention value must be f64".into(),
    })?;
    Ok((*variable, active, 0.0))
}

fn identification_status_ok_for_case(status: &IdentificationStatus) -> bool {
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
        if identification_status_ok_for_case(&case.result.status) {
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

fn admg_to_dag(admg: &Admg) -> Result<Dag, AnalysisError> {
    let n = u32::try_from(admg.node_count()).map_err(|_| AnalysisError::Compile {
        message: "ADMG too large".into(),
    })?;
    let mut dag = Dag::with_variables(n);
    for i in 0..admg.node_count() {
        let from = DenseNodeId::from_raw(u32::try_from(i).unwrap_or(u32::MAX));
        for &to in admg.children(from) {
            dag.insert_directed(from, to).map_err(|e| AnalysisError::Compile {
                message: e.to_string(),
            })?;
        }
    }
    Ok(dag)
}
