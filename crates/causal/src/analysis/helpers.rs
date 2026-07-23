//! Unified `CausalAnalysis` facade.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

//! Private execution helpers.

#![allow(
    clippy::similar_names,
    clippy::too_many_lines,
    clippy::doc_markdown,
    clippy::too_many_arguments,
    clippy::cast_precision_loss
)]

use std::sync::Arc;

use causal_core::{
    AssumptionSet, AverageEffectQuery, BufferMaterialization, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, ExecutionContext, ExecutionPerformanceRecord, Intervention,
    InterventionSequence, LogicalAnalysisPlanRecord, PhysicalExecutionPlanRecord, ProvenanceGraph,
    ProvenanceNode, SequencedIntervention, VERSION, VariableId,
};
use causal_data::{IdRemap, MultiEnvironmentData, TableView, TabularData, TimeSeriesData, dedupe_variable_ids};
use causal_estimate::{CausalPosterior, EffectEstimate, EstimationWorkspace, OverlapPolicy};
use causal_expr::{IdentifiedEstimand, RdDesignParams};
use causal_graph::{CpdagReview, TemporalCpdagReview, TemporalGraphReview};
use causal_validate::{RefutationProblem, RefutationReport, ValidationSuite};

use crate::discovery::{
    DiscoverParams, StaticDiscoverParams, discover_fci, discover_ges, discover_jpcmci_plus,
    discover_lingam, discover_lpcmci, discover_notears, discover_pc, discover_pcmci,
    discover_pcmci_plus, discover_rfci, discover_rpcmci,
};
use crate::discovery_defaults::resolve_ci;
use crate::error::AnalysisError;
use crate::result::CausalAnalysisResult;
use causal_discovery::{MultiDatasetConstraints, RegimeAssignment};

use super::builder::RefuteSuite;

pub(crate) struct AssembleArgs<'a> {
    pub(crate) logical: &'a LogicalAnalysisPlanRecord,
    pub(crate) physical: &'a PhysicalExecutionPlanRecord,
    pub(crate) identification: causal_identify::IdentificationResult,
    pub(crate) estimand: IdentifiedEstimand,
    pub(crate) estimate: EffectEstimate,
    pub(crate) distribution: Option<causal_estimate::InterventionalDistributionEstimate>,
    pub(crate) posterior: Option<causal_estimate::CausalPosterior>,
    pub(crate) mediation: Option<causal_estimate::TemporalMediationEstimate>,
    pub(crate) counterfactual: Option<crate::gcm::IteResult>,
    pub(crate) anomaly: Option<Vec<causal_attribution::AnomalyScores>>,
    pub(crate) change_attribution: Option<causal_attribution::ChangeAttributionResult>,
    pub(crate) mechanism_change: Option<Vec<causal_attribution::MechanismChangeDetection>>,
    pub(crate) unit_change: Option<causal_attribution::UnitChangeResult>,
    pub(crate) refutations: Vec<RefutationReport>,
    pub(crate) diagnostics: Vec<Diagnostic>,
    pub(crate) provenance: ProvenanceGraph,
    pub(crate) treatment: VariableId,
    pub(crate) outcome: VariableId,
    /// Wall-clock nanoseconds for identify→estimate→refute.
    pub(crate) wall_time_ns: u64,
    /// Latency mode label when a tier was requested.
    pub(crate) latency_mode: Option<Arc<str>>,
    /// Per-stage timings.
    pub(crate) stage_timings_ns: Vec<(Arc<str>, u64)>,
    /// Bootstrap replicates requested.
    pub(crate) bootstrap_replicates_requested: Option<u32>,
    /// Bootstrap replicates that succeeded.
    pub(crate) bootstrap_replicates_ok: Option<u32>,
    /// Posterior draws (Bayesian).
    pub(crate) n_draws: Option<u32>,
    /// Cancellation observed during execute.
    pub(crate) cancelled: bool,
    /// Adaptive early-stop (bootstrap SE and/or Bayesian draws).
    pub(crate) early_stopped: bool,
}

pub(crate) fn assemble_result(args: AssembleArgs<'_>) -> CausalAnalysisResult {
    let copy_count = args
        .physical
        .materializations
        .iter()
        .filter(|(_, m)| !matches!(m, BufferMaterialization::Borrowed))
        .count() as u64;
    CausalAnalysisResult {
        logical_plan: args.logical.clone(),
        physical_plan: args.physical.clone(),
        identification: args.identification,
        estimand: args.estimand,
        estimate: args.estimate,
        distribution: args.distribution,
        posterior: args.posterior,
        mediation: args.mediation,
        counterfactual: args.counterfactual,
        anomaly: args.anomaly,
        change_attribution: args.change_attribution,
        mechanism_change: args.mechanism_change,
        unit_change: args.unit_change,
        refutations: args.refutations,
        predictive_checks: Vec::new(),
        diagnostics: args.diagnostics,
        provenance: args.provenance,
        performance: ExecutionPerformanceRecord {
            wall_time_ns: Some(args.wall_time_ns),
            peak_rss_bytes: None,
            copy_count,
            scalar_fallback_count: 0,
            latency_mode: args.latency_mode,
            stage_timings_ns: args.stage_timings_ns,
            bootstrap_replicates_requested: args.bootstrap_replicates_requested,
            bootstrap_replicates_ok: args.bootstrap_replicates_ok,
            n_draws: args.n_draws,
            cancelled: args.cancelled,
            early_stopped: args.early_stopped,
        },
        treatment: args.treatment,
        outcome: args.outcome,
    }
}

pub(crate) type ProvStep<'a> = (&'a str, &'a str, &'a [&'a str], &'a AssumptionSet);

pub(crate) fn provenance_pair(first: ProvStep<'_>, second: ProvStep<'_>) -> ProvenanceGraph {
    let mut provenance = ProvenanceGraph::new();
    for (artifact_id, operation, parents, assumptions) in [first, second] {
        let parent_arcs: Arc<[Arc<str>]> =
            parents.iter().map(|p| Arc::<str>::from(*p)).collect::<Vec<_>>().into();
        provenance.push(ProvenanceNode {
            artifact_id: Arc::from(artifact_id),
            operation: Arc::from(operation),
            parents: parent_arcs,
            assumptions: assumptions.clone(),
            library_version: Arc::from(VERSION),
            config_digest: Some(Arc::from("temporal")),
        });
    }
    provenance
}

pub(crate) fn run_pcmci_review(
    data: &TimeSeriesData,
    max_lag: u32,
    alpha: f64,
    fdr: Option<causal_stats::FdrAdjustment>,
    ci: Arc<dyn causal_stats::ConditionalIndependence + Send + Sync>,
    ctx: &ExecutionContext,
) -> Result<TemporalGraphReview, AnalysisError> {
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let params = DiscoverParams {
        max_lag,
        alpha,
        fdr,
        ci,
        multi_dataset: MultiDatasetConstraints::default(),
    };
    let result = discover_pcmci(data, &vars, &params, ctx)?;
    Ok(result.review)
}

pub(crate) fn run_pcmci_plus_review(
    data: &TimeSeriesData,
    max_lag: u32,
    alpha: f64,
    fdr: Option<causal_stats::FdrAdjustment>,
    ci: Arc<dyn causal_stats::ConditionalIndependence + Send + Sync>,
    ctx: &ExecutionContext,
) -> Result<TemporalCpdagReview, AnalysisError> {
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let params = DiscoverParams {
        max_lag,
        alpha,
        fdr,
        ci,
        multi_dataset: MultiDatasetConstraints::default(),
    };
    let result = discover_pcmci_plus(data, &vars, &params, ctx)?;
    Ok(result.review)
}

pub(crate) fn run_jpcmci_plus_review(
    data: &MultiEnvironmentData,
    max_lag: u32,
    alpha: f64,
    fdr: Option<causal_stats::FdrAdjustment>,
    multi_dataset: &MultiDatasetConstraints,
    ci: Arc<dyn causal_stats::ConditionalIndependence + Send + Sync>,
    ctx: &ExecutionContext,
) -> Result<TemporalCpdagReview, AnalysisError> {
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let system: Vec<VariableId> =
        vars.into_iter().filter(|v| !multi_dataset.is_context(*v)).collect();
    if system.is_empty() {
        return Err(AnalysisError::Compile {
            message: "jpcmci+ needs ≥1 system variable after excluding context_variables".into(),
        });
    }
    let params = DiscoverParams { max_lag, alpha, fdr, ci, multi_dataset: multi_dataset.clone() };
    let result = discover_jpcmci_plus(data, &system, &params, ctx)?;
    Ok(result.review)
}

pub(crate) fn run_rpcmci_discovery(
    data: &TimeSeriesData,
    max_lag: u32,
    alpha: f64,
    fdr: Option<causal_stats::FdrAdjustment>,
    assignment: &RegimeAssignment,
    ci: Arc<dyn causal_stats::ConditionalIndependence + Send + Sync>,
    ctx: &ExecutionContext,
) -> Result<causal_discovery::RpcmciDiscoveryResult, AnalysisError> {
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let params = DiscoverParams {
        max_lag,
        alpha,
        fdr,
        ci,
        multi_dataset: MultiDatasetConstraints::default(),
    };
    if assignment.len() != data.row_count() {
        return Err(AnalysisError::Compile {
            message: format!(
                "RPCMCI regime_assignment length {} != series length {}",
                assignment.len(),
                data.row_count()
            ),
        });
    }
    discover_rpcmci(data, &vars, assignment, &params, None, ctx)
}

pub(crate) fn run_lpcmci_review(
    data: &TimeSeriesData,
    max_lag: u32,
    alpha: f64,
    fdr: Option<causal_stats::FdrAdjustment>,
    ci: Arc<dyn causal_stats::ConditionalIndependence + Send + Sync>,
    ctx: &ExecutionContext,
) -> Result<causal_graph::TemporalPagReview, AnalysisError> {
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let params = DiscoverParams {
        max_lag,
        alpha,
        fdr,
        ci,
        multi_dataset: MultiDatasetConstraints::default(),
    };
    let result = discover_lpcmci(data, &vars, &params, ctx)?;
    Ok(result.review)
}

pub(crate) fn run_pc_review(
    data: &TabularData,
    alpha: f64,
    max_cond_size: usize,
    fdr: Option<causal_stats::FdrAdjustment>,
    ci: Arc<dyn causal_stats::ConditionalIndependence + Send + Sync>,
    ctx: &ExecutionContext,
) -> Result<CpdagReview, AnalysisError> {
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let params =
        StaticDiscoverParams { alpha, max_cond_size, fdr, ci, screen_pc: false, max_subset: None };
    let result = discover_pc(data, &vars, &params, ctx)?;
    Ok(result.review)
}

pub(crate) fn run_ges_review(
    data: &TabularData,
    alpha: f64,
    max_cond_size: usize,
    fdr: Option<causal_stats::FdrAdjustment>,
    ci: Arc<dyn causal_stats::ConditionalIndependence + Send + Sync>,
    ctx: &ExecutionContext,
) -> Result<CpdagReview, AnalysisError> {
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let params =
        StaticDiscoverParams { alpha, max_cond_size, fdr, ci, screen_pc: false, max_subset: None };
    let result = discover_ges(data, &vars, &params, ctx)?;
    Ok(result.review)
}

pub(crate) fn run_lingam_review(
    data: &TabularData,
    max_cond_size: usize,
    prune_threshold: f64,
    ctx: &ExecutionContext,
) -> Result<causal_graph::DagReview, AnalysisError> {
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let params = StaticDiscoverParams {
        alpha: 0.05,
        max_cond_size,
        fdr: None,
        ci: resolve_ci("parcorr", None)?,
        screen_pc: false,
        max_subset: None,
    };
    let result = discover_lingam(data, &vars, &params, prune_threshold, ctx)?;
    Ok(result.review)
}

pub(crate) fn run_notears_review(
    data: &TabularData,
    max_cond_size: usize,
    lambda: f64,
    threshold: f64,
    standardize: bool,
    ctx: &ExecutionContext,
) -> Result<causal_graph::DagReview, AnalysisError> {
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let params = StaticDiscoverParams {
        alpha: 0.05,
        max_cond_size,
        fdr: None,
        ci: resolve_ci("parcorr", None)?,
        screen_pc: false,
        max_subset: None,
    };
    let result = discover_notears(data, &vars, &params, lambda, threshold, standardize, ctx)?;
    Ok(result.discovery.review)
}

pub(crate) fn run_fci_review(
    data: &TabularData,
    alpha: f64,
    max_cond_size: usize,
    fdr: Option<causal_stats::FdrAdjustment>,
    ci: Arc<dyn causal_stats::ConditionalIndependence + Send + Sync>,
    ctx: &ExecutionContext,
) -> Result<causal_graph::PagReview, AnalysisError> {
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let params =
        StaticDiscoverParams { alpha, max_cond_size, fdr, ci, screen_pc: false, max_subset: None };
    let result = discover_fci(data, &vars, &params, ctx)?;
    Ok(result.review)
}

pub(crate) fn run_rfci_review(
    data: &TabularData,
    alpha: f64,
    max_cond_size: usize,
    fdr: Option<causal_stats::FdrAdjustment>,
    ci: Arc<dyn causal_stats::ConditionalIndependence + Send + Sync>,
    ctx: &ExecutionContext,
) -> Result<causal_graph::PagReview, AnalysisError> {
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let params =
        StaticDiscoverParams { alpha, max_cond_size, fdr, ci, screen_pc: false, max_subset: None };
    let result = discover_rfci(data, &vars, &params, ctx)?;
    Ok(result.review)
}

pub(crate) fn run_refuters(
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
    estimate: &EffectEstimate,
    workspace: &mut EstimationWorkspace,
    propensity: Option<&mut causal_stats::PropensityWorkspace>,
    ctx: &ExecutionContext,
    suite: RefuteSuite,
    estimator: &str,
    custom: &[Arc<dyn causal_validate::CustomEffectValidator>],
    temporal: Option<causal_validate::TemporalRefitContext<'_>>,
) -> Result<Vec<RefutationReport>, AnalysisError> {
    let problem = RefutationProblem {
        data,
        estimand,
        query,
        original: estimate,
        estimator: Some(estimator),
        temporal,
    };
    let mut validation = match suite {
        RefuteSuite::None => {
            if custom.is_empty() {
                return Ok(Vec::new());
            }
            ValidationSuite::new()
        }
        RefuteSuite::Cheap => ValidationSuite::overlap_and_evalue(),
        RefuteSuite::PlaceboAndRcc => ValidationSuite::placebo_and_rcc(),
        RefuteSuite::Full => ValidationSuite::full_effect(),
    };
    for v in custom {
        validation = validation.with_custom(Arc::clone(v));
    }
    let outcomes = match propensity {
        Some(pws) => validation
            .run_with_propensity(&problem, workspace, pws, ctx)
            .map_err(AnalysisError::from)?,
        None => validation.run(&problem, workspace, ctx).map_err(AnalysisError::from)?,
    };
    Ok(ValidationSuite::reports_only(&outcomes))
}

pub(crate) fn resolve_analysis_ci(
    discovery_ci: Option<&Arc<dyn causal_stats::ConditionalIndependence + Send + Sync>>,
) -> Result<Arc<dyn causal_stats::ConditionalIndependence + Send + Sync>, AnalysisError> {
    match discovery_ci {
        Some(ci) => Ok(Arc::clone(ci)),
        None => resolve_ci("parcorr", None),
    }
}

pub(crate) fn effect_from_posterior(
    posterior: &CausalPosterior,
) -> Result<EffectEstimate, AnalysisError> {
    let eq = posterior.effect_column().ok_or_else(|| AnalysisError::Compile {
        message: "Bayesian posterior missing effect column".into(),
    })?;
    let ate = posterior.summaries.mean[eq];
    // Report posterior SD of the effect (sampling uncertainty), not MCSE of the mean.
    let se = posterior.summaries.sd[eq];
    Ok(EffectEstimate {
        ate,
        se_analytic: se,
        se_bootstrap: None,
        bootstrap_replicates_ok: None,
        bootstrap_replicates_failed: None,
        bootstrap_cancelled: false,
            bootstrap_early_stopped: false,
        assumptions: posterior.assumptions.clone(),
        overlap: OverlapPolicy::ExplicitOverride,
        overlap_report: None,
        retained_memory_bytes: None,
    })
}

/// Diagnostic recording which overlap policy an estimator applied.
pub(crate) fn overlap_diagnostic(overlap: OverlapPolicy) -> Diagnostic {
    match overlap {
        OverlapPolicy::ExplicitOverride => Diagnostic::new(
            "estimate.overlap.explicit_override",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "estimator used ExplicitOverride for positivity (not a propensity-based method)",
        ),
        OverlapPolicy::RequireDiagnostics { .. } => Diagnostic::new(
            "estimate.overlap.require_diagnostics",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "estimator used RequireDiagnostics for mandatory positivity diagnostics",
        ),
    }
}

/// Surface applied external-prior alphas after conflict shrink.
pub(crate) fn push_conflict_diagnostics(
    diagnostics: &mut Vec<Diagnostic>,
    summary: &causal_prob::ConflictSummary,
) {
    for (i, id) in summary.source_ids.iter().enumerate() {
        let req = summary.alphas_requested.get(i).copied().unwrap_or(f64::NAN);
        let app = summary.alphas_applied.get(i).copied().unwrap_or(f64::NAN);
        let p = summary
            .p_values
            .get(i)
            .and_then(|x| *x)
            .map_or_else(|| "none".to_string(), |v| format!("{v}"));
        let kl = summary
            .kl_values
            .get(i)
            .and_then(|x| *x)
            .map_or_else(|| "none".to_string(), |v| format!("{v}"));
        let mut d = Diagnostic::new(
            "bayes.prior_bank.conflict",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "external prior {id}: alpha_requested={req}, alpha_applied={app}, p={p}, kl={kl}"
            ),
        );
        d.fields = Arc::from([
            (Arc::from("source_id"), Arc::clone(id)),
            (Arc::from("alpha_requested"), Arc::from(format!("{req}"))),
            (Arc::from("alpha_applied"), Arc::from(format!("{app}"))),
        ]);
        diagnostics.push(d);
    }
}

/// Columns required for estimation after identification (treatment, outcome, Z, …).
pub(crate) fn columns_for_ate_estimand(
    query: &AverageEffectQuery,
    estimand: &IdentifiedEstimand,
) -> Vec<VariableId> {
    dedupe_variable_ids(
        std::iter::once(query.treatment)
            .chain(std::iter::once(query.outcome))
            .chain(query.effect_modifiers.iter().copied())
            .chain(estimand.adjustment_set.iter().copied())
            .chain(estimand.instruments.iter().copied())
            .chain(estimand.mediators.iter().copied())
            .chain(estimand.rd_design.map(|rd| rd.running_variable)),
    )
}

/// Project table to estimand columns and remap query/estimand for kernel work.
///
/// Returns projected data + remapped query/estimand. The caller should keep the
/// original estimand for result name resolution.
///
/// # Errors
///
/// Projection or id remap failures.
pub(crate) fn project_for_ate_estimate(
    data: &TabularData,
    query: &AverageEffectQuery,
    estimand: &IdentifiedEstimand,
) -> Result<(TabularData, AverageEffectQuery, IdentifiedEstimand), AnalysisError> {
    let ids = columns_for_ate_estimand(query, estimand);
    // Already thin — skip rebuild when every column is required.
    if ids.len() == data.schema().len() {
        return Ok((data.clone(), query.clone(), estimand.clone()));
    }
    let (projected, remap) = data.project(&ids)?;
    let query_p = remap_average_effect_query(query, &remap)?;
    let estimand_p = remap_identified_estimand(estimand, &remap)?;
    Ok((projected, query_p, estimand_p))
}

fn remap_variable_slice(
    ids: &[VariableId],
    remap: &IdRemap,
) -> Result<Arc<[VariableId]>, AnalysisError> {
    let mapped: Result<Vec<_>, _> = ids.iter().map(|id| remap.map(*id)).collect();
    Ok(Arc::from(mapped?))
}

fn remap_intervention(
    intervention: &Intervention,
    remap: &IdRemap,
) -> Result<Intervention, AnalysisError> {
    match intervention {
        Intervention::Set { variable, value } => {
            Ok(Intervention::Set { variable: remap.map(*variable)?, value: value.clone() })
        }
        Intervention::Shift { variable, delta } => {
            Ok(Intervention::Shift { variable: remap.map(*variable)?, delta: delta.clone() })
        }
        Intervention::Stochastic { variable, policy } => Ok(Intervention::Stochastic {
            variable: remap.map(*variable)?,
            policy: policy.clone(),
        }),
        Intervention::Soft { variable, mechanism } => Ok(Intervention::Soft {
            variable: remap.map(*variable)?,
            mechanism: mechanism.clone(),
        }),
        Intervention::Sequence(seq) => {
            let steps: Result<Vec<_>, AnalysisError> = seq
                .steps
                .iter()
                .map(|s| {
                    Ok(SequencedIntervention {
                        intervention: remap_intervention(&s.intervention, remap)?,
                        temporal: s.temporal.clone(),
                    })
                })
                .collect();
            Ok(Intervention::Sequence(InterventionSequence::new(steps?)))
        }
        other => Err(AnalysisError::Compile {
            message: format!("cannot remap unsupported intervention variant: {other:?}"),
        }),
    }
}

fn remap_average_effect_query(
    query: &AverageEffectQuery,
    remap: &IdRemap,
) -> Result<AverageEffectQuery, AnalysisError> {
    Ok(AverageEffectQuery {
        treatment: remap.map(query.treatment)?,
        outcome: remap.map(query.outcome)?,
        effect_modifiers: remap_variable_slice(&query.effect_modifiers, remap)?,
        control: remap_intervention(&query.control, remap)?,
        active: remap_intervention(&query.active, remap)?,
        target_population: query.target_population.clone(),
    })
}

fn remap_identified_estimand(
    estimand: &IdentifiedEstimand,
    remap: &IdRemap,
) -> Result<IdentifiedEstimand, AnalysisError> {
    let rd_design = match &estimand.rd_design {
        None => None,
        Some(rd) => Some(RdDesignParams {
            running_variable: remap.map(rd.running_variable)?,
            cutoff: rd.cutoff,
            bandwidth: rd.bandwidth,
        }),
    };
    Ok(IdentifiedEstimand {
        method: Arc::clone(&estimand.method),
        adjustment_set: remap_variable_slice(&estimand.adjustment_set, remap)?,
        instruments: remap_variable_slice(&estimand.instruments, remap)?,
        mediators: remap_variable_slice(&estimand.mediators, remap)?,
        functional: estimand.functional,
        rd_design,
    })
}

/// Diagnostic when a wide table was narrowed after identification.
pub(crate) fn projection_diagnostic(
    full_cols: usize,
    projected_cols: usize,
) -> Option<Diagnostic> {
    if projected_cols >= full_cols {
        return None;
    }
    Some(Diagnostic::new(
        "exec.project.columns",
        DiagnosticKind::Execution,
        DiagnosticSeverity::Info,
        format!("projected {full_cols} → {projected_cols} columns after identification"),
    ))
}
