//! Unified `CausalAnalysis` facade (DESIGN.md §21).
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
    AssumptionSet, AverageEffectQuery, BufferMaterialization, Diagnostic,
    DiagnosticKind, DiagnosticSeverity, ExecutionContext, ExecutionPerformanceRecord,
    LogicalAnalysisPlanRecord, PhysicalExecutionPlanRecord, ProvenanceGraph, ProvenanceNode, VERSION, VariableId,
};
use causal_data::{
    MultiEnvironmentData, TableView, TabularData, TimeSeriesData,
};
use causal_estimate::{
    CausalPosterior, EffectEstimate,
    EstimationWorkspace, OverlapPolicy,
};
use causal_expr::IdentifiedEstimand;
use causal_graph::{CpdagReview, TemporalCpdagReview, TemporalGraphReview};
use causal_validate::{
    RefutationProblem, RefutationReport, ValidationSuite,
};

use crate::discovery::{
    DiscoverParams, StaticDiscoverParams, discover_jpcmci_plus, discover_lpcmci, discover_pc,
    discover_pcmci, discover_pcmci_plus, discover_rpcmci,
};
use crate::discovery_defaults::resolve_ci;
use causal_discovery::{MultiDatasetConstraints, RegimeAssignment};
use crate::error::AnalysisError;
use crate::result::CausalAnalysisResult;

use super::builder::RefuteSuite;


pub(crate) struct AssembleArgs<'a> {
    pub(crate) logical: &'a LogicalAnalysisPlanRecord,
    pub(crate) physical: &'a PhysicalExecutionPlanRecord,
    pub(crate) identification: causal_identify::IdentificationResult,
    pub(crate) estimand: IdentifiedEstimand,
    pub(crate) estimate: EffectEstimate,
    pub(crate) posterior: Option<causal_estimate::CausalPosterior>,
    pub(crate) refutations: Vec<RefutationReport>,
    pub(crate) diagnostics: Vec<Diagnostic>,
    pub(crate) provenance: ProvenanceGraph,
    pub(crate) treatment: VariableId,
    pub(crate) outcome: VariableId,
    /// Wall-clock nanoseconds for identify→estimate→refute.
    pub(crate) wall_time_ns: u64,
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
        posterior: args.posterior,
        refutations: args.refutations,
        diagnostics: args.diagnostics,
        provenance: args.provenance,
        performance: ExecutionPerformanceRecord {
            wall_time_ns: Some(args.wall_time_ns),
            peak_rss_bytes: None,
            copy_count,
            scalar_fallback_count: 0,
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
    ctx: &ExecutionContext,
) -> Result<TemporalGraphReview, AnalysisError> {
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let params = DiscoverParams {
        max_lag,
        alpha,
        fdr,
        ci: resolve_ci("parcorr", None)?,
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
    ctx: &ExecutionContext,
) -> Result<TemporalCpdagReview, AnalysisError> {
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let params = DiscoverParams {
        max_lag,
        alpha,
        fdr,
        ci: resolve_ci("parcorr", None)?,
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
    ctx: &ExecutionContext,
) -> Result<TemporalCpdagReview, AnalysisError> {
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let system: Vec<VariableId> = vars
        .into_iter()
        .filter(|v| !multi_dataset.is_context(*v))
        .collect();
    if system.is_empty() {
        return Err(AnalysisError::Compile {
            message: "jpcmci+ needs ≥1 system variable after excluding context_variables".into(),
        });
    }
    let params = DiscoverParams {
        max_lag,
        alpha,
        fdr,
        ci: resolve_ci("parcorr", None)?,
        multi_dataset: multi_dataset.clone(),
    };
    let result = discover_jpcmci_plus(data, &system, &params, ctx)?;
    Ok(result.review)
}

pub(crate) fn run_rpcmci_discovery(
    data: &TimeSeriesData,
    max_lag: u32,
    alpha: f64,
    fdr: Option<causal_stats::FdrAdjustment>,
    assignment: &RegimeAssignment,
    ctx: &ExecutionContext,
) -> Result<causal_discovery::RpcmciDiscoveryResult, AnalysisError> {
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let params = DiscoverParams {
        max_lag,
        alpha,
        fdr,
        ci: resolve_ci("parcorr", None)?,
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
    ctx: &ExecutionContext,
) -> Result<causal_graph::TemporalPagReview, AnalysisError> {
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let params = DiscoverParams {
        max_lag,
        alpha,
        fdr,
        ci: resolve_ci("parcorr", None)?,
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
    ctx: &ExecutionContext,
) -> Result<CpdagReview, AnalysisError> {
    let vars: Vec<VariableId> = data.schema().variables().iter().map(|v| v.id).collect();
    let params = StaticDiscoverParams {
        alpha,
        max_cond_size,
        fdr,
        ci: resolve_ci("parcorr", None)?,
    };
    let result = discover_pc(data, &vars, &params, ctx)?;
    Ok(result.review)
}

pub(crate) fn run_refuters(
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
    estimate: &EffectEstimate,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
    suite: RefuteSuite,
    estimator: &str,
) -> Result<Vec<RefutationReport>, AnalysisError> {
    let problem =
        RefutationProblem { data, estimand, query, original: estimate, estimator: Some(estimator) };
    let validation = match suite {
        RefuteSuite::None => return Ok(Vec::new()),
        RefuteSuite::PlaceboAndRcc => ValidationSuite::placebo_and_rcc(),
        RefuteSuite::Full => ValidationSuite::full_effect(),
    };
    let outcomes = validation.run(&problem, workspace, ctx).map_err(AnalysisError::from)?;
    Ok(ValidationSuite::reports_only(&outcomes))
}

pub(crate) fn effect_from_posterior(posterior: &CausalPosterior) -> Result<EffectEstimate, AnalysisError> {
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
