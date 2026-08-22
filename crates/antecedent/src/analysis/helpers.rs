//! Unified `Study` facade.
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

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, BufferMaterialization, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, ExecutionContext, ExecutionPerformanceRecord, Intervention,
    InterventionSequence, LogicalAnalysisPlanRecord, PhysicalExecutionPlanRecord, ProvenanceGraph,
    ProvenanceNode, SequencedIntervention, VERSION, VariableId,
};
use antecedent_data::{IdRemap, TableView, TabularData, dedupe_variable_ids};
use antecedent_estimate::{CausalPosterior, EffectEstimate, EstimationWorkspace, OverlapPolicy};
use antecedent_expr::{IdentifiedEstimand, RdDesignParams};
use antecedent_validate::{RefutationProblem, RefutationReport, ValidationSuite};

use crate::error::CausalError;
use crate::result::StudyResult;

use super::builder::RefuteSuite;

pub(crate) struct AssembleArgs<'a> {
    pub(crate) logical: &'a LogicalAnalysisPlanRecord,
    pub(crate) physical: &'a PhysicalExecutionPlanRecord,
    pub(crate) identification: antecedent_identify::IdentificationResult,
    pub(crate) estimand: IdentifiedEstimand,
    pub(crate) estimate: EffectEstimate,
    pub(crate) distribution: Option<antecedent_estimate::InterventionalDistributionEstimate>,
    pub(crate) posterior: Option<antecedent_estimate::CausalPosterior>,
    pub(crate) mediation: Option<antecedent_estimate::TemporalMediationEstimate>,
    pub(crate) counterfactual: Option<crate::gcm::IteResult>,
    pub(crate) anomaly: Option<Vec<antecedent_attribution::AnomalyScores>>,
    pub(crate) change_attribution: Option<antecedent_attribution::ChangeAttributionResult>,
    pub(crate) mechanism_change: Option<Vec<antecedent_attribution::MechanismChangeDetection>>,
    pub(crate) unit_change: Option<antecedent_attribution::UnitChangeResult>,
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

pub(crate) fn assemble_result(args: AssembleArgs<'_>) -> StudyResult {
    let copy_count = args
        .physical
        .materializations
        .iter()
        .filter(|(_, m)| !matches!(m, BufferMaterialization::Borrowed))
        .count() as u64;
    StudyResult {
        logical_plan: args.logical.clone(),
        physical_plan: args.physical.clone(),
        identification: args.identification,
        estimand: args.estimand,
        estimate: args.estimate,
        response: None,
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
        support_status: None,
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
            bytes_borrowed: None,
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

/// Diagnostic surfacing one validator skipped as [`antecedent_validate::ValidationOutcome::NotApplicable`]
/// for this run.
///
/// This is a **per-run, data-dependent** skip — e.g. `OverlapRefuter` on a temporal
/// design, `EValue` on a non-binary treatment, an MCMC diagnostic on a Laplace posterior
/// — not the support matrix's permanent typed impossibility
/// (`SupportRefusal::NotApplicable` / wire `not_applicable`). The same validator can run
/// cleanly on a different call against the same licensed cell; a matrix `not_applicable`
/// cell never reaches validation at all. See `docs/capabilities.md`'s "'Not applicable'
/// means three different things".
pub(crate) fn validator_not_applicable_diagnostic(
    validator: antecedent_validate::ValidatorId,
    reason: &str,
) -> Diagnostic {
    let mut d = Diagnostic::new(
        "refute.validator.not_applicable",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "validator '{validator}' was requested but is not applicable to this run's data/estimand \
             (per-run skip, not a permanent support-matrix refusal): {reason}"
        ),
    );
    d.fields = Arc::from([(Arc::from("validator"), Arc::from(validator.as_str()))]);
    d
}

/// Build one [`validator_not_applicable_diagnostic`] per skipped outcome.
pub(crate) fn validator_not_applicable_diagnostics(
    outcomes: &[antecedent_validate::ValidationOutcome],
) -> Vec<Diagnostic> {
    ValidationSuite::not_applicable_only(outcomes)
        .into_iter()
        .map(|(validator, reason)| validator_not_applicable_diagnostic(validator, &reason))
        .collect()
}

/// Run the requested refuter suite, returning both the produced reports and one
/// diagnostic per validator that was requested but skipped as `NotApplicable` for this
/// run (see [`validator_not_applicable_diagnostic`]).
pub(crate) fn run_refuters(
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
    estimate: &EffectEstimate,
    workspace: &mut EstimationWorkspace,
    propensity: Option<&mut antecedent_stats::PropensityWorkspace>,
    ctx: &ExecutionContext,
    suite: RefuteSuite,
    estimator: &str,
    custom: &[Arc<dyn antecedent_validate::CustomEffectValidator>],
    temporal: Option<antecedent_validate::TemporalRefitContext<'_>>,
) -> Result<(Vec<RefutationReport>, Vec<Diagnostic>), CausalError> {
    let problem =
        RefutationProblem::new(data, estimand, query, estimate, Some(estimator), temporal);
    let mut validation = match suite {
        RefuteSuite::None => {
            if custom.is_empty() {
                return Ok((Vec::new(), Vec::new()));
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
            .map_err(CausalError::from)?,
        None => validation.run(&problem, workspace, ctx).map_err(CausalError::from)?,
    };
    let diagnostics = validator_not_applicable_diagnostics(&outcomes);
    Ok((ValidationSuite::reports_only(&outcomes), diagnostics))
}

pub(crate) fn effect_from_posterior(
    posterior: &CausalPosterior,
) -> Result<EffectEstimate, CausalError> {
    let eq = posterior.effect_column().ok_or_else(|| CausalError::Compile {
        message: "Bayesian posterior missing effect column".into(),
    })?;
    let ate = posterior.summaries.mean[eq];
    // Report posterior SD of the effect (sampling uncertainty), not MCSE of the mean.
    let se = posterior.summaries.sd[eq];
    Ok(EffectEstimate::new(ate, se, posterior.assumptions.clone(), OverlapPolicy::ExplicitOverride))
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
    summary: &antecedent_prob::ConflictSummary,
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
) -> Result<(TabularData, AverageEffectQuery, IdentifiedEstimand), CausalError> {
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
) -> Result<Arc<[VariableId]>, CausalError> {
    let mapped: Result<Vec<_>, _> = ids.iter().map(|id| remap.map(*id)).collect();
    Ok(Arc::from(mapped?))
}

fn remap_intervention(
    intervention: &Intervention,
    remap: &IdRemap,
) -> Result<Intervention, CausalError> {
    match intervention {
        Intervention::Set { variable, value } => {
            Ok(Intervention::Set { variable: remap.map(*variable)?, value: value.clone() })
        }
        Intervention::Shift { variable, delta } => {
            Ok(Intervention::Shift { variable: remap.map(*variable)?, delta: delta.clone() })
        }
        Intervention::Stochastic { variable, policy } => {
            Ok(Intervention::Stochastic { variable: remap.map(*variable)?, policy: policy.clone() })
        }
        Intervention::Soft { variable, mechanism } => {
            Ok(Intervention::Soft { variable: remap.map(*variable)?, mechanism: mechanism.clone() })
        }
        Intervention::Sequence(seq) => {
            let steps: Result<Vec<_>, CausalError> = seq
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
        other => Err(CausalError::Compile {
            message: format!("cannot remap unsupported intervention variant: {other:?}"),
        }),
    }
}

fn remap_average_effect_query(
    query: &AverageEffectQuery,
    remap: &IdRemap,
) -> Result<AverageEffectQuery, CausalError> {
    Ok(AverageEffectQuery::new(
        remap.map(query.treatment)?,
        remap.map(query.outcome)?,
        remap_variable_slice(&query.effect_modifiers, remap)?,
        remap_intervention(&query.control, remap)?,
        remap_intervention(&query.active, remap)?,
        query.target_population.clone(),
    ))
}

fn remap_identified_estimand(
    estimand: &IdentifiedEstimand,
    remap: &IdRemap,
) -> Result<IdentifiedEstimand, CausalError> {
    let rd_design = match &estimand.rd_design {
        None => None,
        Some(rd) => {
            Some(RdDesignParams::new(remap.map(rd.running_variable)?, rd.cutoff, rd.bandwidth))
        }
    };
    Ok(IdentifiedEstimand::new(
        Arc::clone(&estimand.method),
        remap_variable_slice(&estimand.adjustment_set, remap)?,
        remap_variable_slice(&estimand.instruments, remap)?,
        remap_variable_slice(&estimand.mediators, remap)?,
        estimand.functional,
        rd_design,
    ))
}

/// Diagnostic when a wide table was narrowed after identification.
pub(crate) fn projection_diagnostic(full_cols: usize, projected_cols: usize) -> Option<Diagnostic> {
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

/// Full-suite prior sensitivity: α-grid when external compose is present, else isotropic scale.
pub(crate) fn evaluate_bayesian_prior_sensitivity(
    cfg: &crate::inference::BayesianConfig,
    est: &antecedent_estimate::BayesianGComputationAte,
    prep: &antecedent_estimate::PreparedBayesianProblem,
    status: antecedent_identify::IdentificationStatus,
    posterior: &CausalPosterior,
    ws: &mut antecedent_estimate::BayesianGCompWorkspace,
    ctx: &ExecutionContext,
) -> Result<
    (antecedent_prob::PriorSensitivitySummary, antecedent_validate::PriorSensitivity),
    CausalError,
> {
    use antecedent_validate::{ExternalAlphaSensitivity, PriorSensitivity};
    if let Some(ext) = cfg.external_compose.as_ref() {
        let alphas_applied: Arc<[f64]> = posterior.conflict_summary.as_ref().map_or_else(
            || Arc::clone(&ext.composed.alphas_applied),
            |cs| Arc::clone(&cs.alphas_applied),
        );
        let sens = PriorSensitivity::standard_alpha_grid();
        let (summary, _) = sens
            .evaluate_external_alpha(
                est,
                prep,
                status,
                ws,
                ctx,
                ExternalAlphaSensitivity { sources: &ext.sources, alphas_applied: &alphas_applied },
            )
            .map_err(CausalError::from)?;
        Ok((summary, sens))
    } else {
        let sens = PriorSensitivity::standard_grid();
        let (summary, _) = sens.evaluate(est, prep, status, ws, ctx).map_err(CausalError::from)?;
        Ok((summary, sens))
    }
}
