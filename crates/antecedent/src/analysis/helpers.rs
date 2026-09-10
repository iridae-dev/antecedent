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
    AssumptionSet, AverageEffectQuery, BufferMaterialization, ConditionalEffectQuery, Diagnostic,
    DiagnosticKind,
    DiagnosticSeverity, ExecutionContext, ExecutionPerformanceRecord, Intervention,
    InterventionSequence, LogicalAnalysisPlanRecord, OutcomeFunctional,
    PhysicalExecutionPlanRecord, ProvenanceGraph, ProvenanceNode, SequencedIntervention, VERSION,
    VariableId,
};
use antecedent_data::{IdRemap, TableView, TabularData, dedupe_variable_ids};
use antecedent_estimate::{
    AipwAte, CausalPosterior, EffectEstimate, EstimationWorkspace, OverlapPolicy, ScoreTable,
    crossfit_binary_scores, exceedance_cdf_values, summarize_functional,
};
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
        certificate: None,
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
        structure_source: crate::support::StructureSource::Explicit,
        candidate_selection: None,
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
    let mut diagnostics: Vec<_> = ValidationSuite::not_applicable_only(outcomes)
        .into_iter()
        .map(|(validator, reason)| validator_not_applicable_diagnostic(validator, &reason))
        .collect();
    for outcome in outcomes {
        if let antecedent_validate::ValidationOutcome::Failed { validator, reason } = outcome {
            let mut d = Diagnostic::new(
                "refute.validator.failed",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Warning,
                format!("validator {validator} could not compute a verdict: {reason}"),
            );
            d.fields = Arc::from([
                (Arc::from("validator"), validator.clone()),
                (Arc::from("reason"), reason.clone()),
            ]);
            diagnostics.push(d);
        }
    }
    diagnostics
}

/// Run the requested refuter suite, returning raw outcomes (reports and
/// per-run `NotApplicable` skips) without collapsing them.
pub(crate) fn refute_outcomes(
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
) -> Result<Vec<antecedent_validate::ValidationOutcome>, CausalError> {
    let problem =
        RefutationProblem::new(data, estimand, query, estimate, Some(estimator), temporal);
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
    match propensity {
        Some(pws) => {
            validation.run_with_propensity(&problem, workspace, pws, ctx).map_err(CausalError::from)
        }
        None => validation.run(&problem, workspace, ctx).map_err(CausalError::from),
    }
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
    let outcomes = refute_outcomes(
        data, estimand, query, estimate, workspace, propensity, ctx, suite, estimator, custom,
        temporal,
    )?;
    let diagnostics = validator_not_applicable_diagnostics(&outcomes);
    Ok((ValidationSuite::reports_only(&outcomes), diagnostics))
}

/// Cheap/full on a plugin intervention *level*. Contrast-shaped refuters are
/// not licensed. Cheap is overlap only; full is overlap plus sampling-stability.
pub(crate) fn run_plugin_level_refuters(
    data: &TabularData,
    estimand: &IdentifiedEstimand,
    query: &AverageEffectQuery,
    estimate: &EffectEstimate,
    workspace: &mut EstimationWorkspace,
    ctx: &ExecutionContext,
    suite: RefuteSuite,
    estimator: &str,
    custom: &[Arc<dyn antecedent_validate::CustomEffectValidator>],
) -> Result<(Vec<antecedent_validate::RefutationReport>, Vec<Diagnostic>), CausalError> {
    let problem =
        RefutationProblem::new(data, estimand, query, estimate, Some(estimator), None);
    let mut validation = match suite {
        RefuteSuite::None => {
            if custom.is_empty() {
                return Ok((Vec::new(), Vec::new()));
            }
            ValidationSuite::new()
        }
        RefuteSuite::Cheap | RefuteSuite::PlaceboAndRcc => ValidationSuite::overlap_only(),
        RefuteSuite::Full => ValidationSuite::plugin_level_full(),
    };
    for v in custom {
        validation = validation.with_custom(Arc::clone(v));
    }
    let outcomes = validation.run(&problem, workspace, ctx).map_err(CausalError::from)?;
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

/// Replace `Y` with `1{Y > c}` when the query asks for a single exceedance.
///
/// Grids are not reduced to the first threshold. Licensed grid paths build a
/// score table or evaluate each `c` explicitly.
///
/// # Errors
///
/// Missing outcome column, length mismatch, or an exceedance grid (refuse
/// rather than silently return `1{Y > c₀}`).
pub(crate) fn apply_outcome_functional(
    data: &TabularData,
    outcome: VariableId,
    functional: &OutcomeFunctional,
) -> Result<TabularData, CausalError> {
    match functional {
        OutcomeFunctional::Exceedance(c) => {
            let y = data.float64_values(outcome)?;
            let threshold = c.to_f64();
            let ind: Arc<[f64]> = y.iter().map(|&yi| f64::from(yi > threshold)).collect();
            data.with_replaced_float(outcome, ind).map_err(CausalError::from)
        }
        OutcomeFunctional::ExceedanceGrid(_) => Err(CausalError::Unsupported {
            message: "exceedance grids cannot be reduced to the first threshold; use the score-table or per-threshold path",
        }),
        _ => Ok(data.clone()),
    }
}

/// Apply a scalar outcome transform. Grids leave `Y` unchanged for a later
/// per-threshold or score-table path.
pub(crate) fn apply_scalar_outcome_functional(
    data: &TabularData,
    outcome: VariableId,
    functional: &OutcomeFunctional,
) -> Result<TabularData, CausalError> {
    match functional {
        OutcomeFunctional::ExceedanceGrid(_) => Ok(data.clone()),
        other => apply_outcome_functional(data, outcome, other),
    }
}

/// Cross-fitted AIPW scores for an exceedance / grid functional on original `Y`.
pub(crate) fn maybe_build_functional_scores(
    data: &TabularData,
    query: &AverageEffectQuery,
    estimand: &IdentifiedEstimand,
    est: &AipwAte,
) -> Result<Option<ScoreTable>, CausalError> {
    if !matches!(
        query.outcome_functional,
        OutcomeFunctional::Exceedance(_) | OutcomeFunctional::ExceedanceGrid(_)
    ) {
        return Ok(None);
    }
    let method = estimand.method.as_ref();
    if !(method.contains("adjustment")
        || method.contains("backdoor")
        || method.starts_with("tiered."))
    {
        return Ok(None);
    }
    let problem = est.prepare(data, estimand, query)?;
    Ok(Some(crossfit_binary_scores(
        &problem,
        query,
        antecedent_estimate::DEFAULT_AIPW_FOLDS,
        &est.glm_options,
        est.backend,
    )?))
}

/// Attach `F_a(c)`, joint IF covariance, and the score table used to form them.
///
/// # Errors
///
/// Score summarize or inference failure. A grid query must not keep a
/// first-threshold mean after the score table failed.
pub(crate) fn attach_score_functional_grid(
    mut estimate: EffectEstimate,
    table: ScoreTable,
) -> Result<(EffectEstimate, Vec<Diagnostic>), CausalError> {
    let (summary, monotone_rearranged, mut diagnostics) = summarize_functional(&table, None)?;
    let n_thresholds = distinct_threshold_count(&table);
    if table.intervened.is_empty() && summary.means.len() >= 2 && n_thresholds <= 1 {
        let mut coefficients = vec![0.0; table.n_columns()];
        coefficients[0] = -1.0;
        coefficients[1] = 1.0;
        let contrast = table.linear_contrast(&summary, &coefficients)?;
        estimate.ate = contrast.value;
        estimate.se_analytic = contrast.se;
        estimate.influence = Some(
            table
                .column(0)?
                .iter()
                .zip(table.column(1)?)
                .map(|(a, b)| b - a)
                .collect::<Vec<_>>()
                .into(),
        );
        if estimate.se_bootstrap.take().is_some() {
            diagnostics.push(Diagnostic::new(
                "estimate.functional.crossfit_inference",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "cross-fitted grid reports joint influence uncertainty; bootstrap from the earlier full-sample fit is not reused",
            ));
        }
    } else if n_thresholds > 1 {
        estimate.ate = f64::NAN;
        estimate.se_analytic = f64::NAN;
        estimate.se_bootstrap = None;
        estimate.influence = None;
        estimate.simultaneous_interval = None;
        diagnostics.push(Diagnostic::new(
            "estimate.functional.grid_scalar_cleared",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "exceedance grids do not publish a first-threshold scalar ATE; use exceedance_cdf and the score table",
        ));
    }
    let cdf = exceedance_cdf_values(&summary, &table);
    let inference = table.inference(None)?;
    let mut estimate = estimate
        .with_score_table(Some(table))
        .with_joint_covariance(Some(summary.covariance))
        .with_exceedance_cdf(cdf)
        .with_monotone_rearranged(monotone_rearranged);
    estimate.score_inference = Some(inference);
    Ok((estimate, diagnostics))
}

fn distinct_threshold_count(table: &ScoreTable) -> usize {
    let mut thresholds: Vec<f64> = table.columns.iter().filter_map(|c| c.threshold).collect();
    thresholds.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    thresholds.dedup_by(|a, b| *a == *b);
    thresholds.len()
}

/// Build scores on original `Y` and attach the exceedance grid when licensed.
pub(crate) fn attach_average_functional_grid(
    estimate: EffectEstimate,
    data: &TabularData,
    query: &AverageEffectQuery,
    estimand: &IdentifiedEstimand,
    extra_diagnostics: &mut Vec<Diagnostic>,
    estimator_id: crate::strategy_table::EstimatorId,
    study: &super::execute::Study,
) -> Result<EffectEstimate, CausalError> {
    if estimator_id != crate::strategy_table::EstimatorId::Aipw
        || !matches!(query.target_population, antecedent_core::TargetPopulation::AllObserved)
    {
        return Ok(estimate);
    }
    let mut config = match study.estimator_spec.as_ref() {
        Some(crate::estimator_spec::EstimatorSpec::Aipw(config)) => *config.clone(),
        _ => AipwAte::new(),
    };
    if let Some(overlap) = study.overlap_policy {
        config.overlap = overlap;
    }
    if !matches!(query.outcome_functional, OutcomeFunctional::Mean)
        && (matches!(config.overlap, OverlapPolicy::RequireDiagnostics { trim: Some(_), .. })
            || config.se_kind != antecedent_estimate::AnalyticSeKind::Homoskedastic)
    {
        return Err(CausalError::Unsupported {
            message: "cross-fitted functional scores require iid inference without propensity trimming",
        });
    }
    let Some(table) = maybe_build_functional_scores(data, query, estimand, &config)? else {
        if !query.outcome_functional.is_mean() {
            return Err(CausalError::Unsupported {
                message: "exceedance requires cross-fitted AIPW scores; refusing a first-threshold or mean substitute",
            });
        }
        return Ok(estimate);
    };
    let (estimate, diagnostics) = attach_score_functional_grid(estimate, table)?;
    extra_diagnostics.extend(diagnostics);
    Ok(estimate)
}

/// Attach a point E-value for the tier-closure no-latent-to-outcome premise.
pub(crate) fn attach_tiered_evalue(
    estimate: &mut EffectEstimate,
    data: &TabularData,
    outcome: VariableId,
    diagnostics: &mut Vec<Diagnostic>,
) {
    if estimate.evalue.is_some() || !estimate.ate.is_finite() {
        return;
    }
    let Ok(y) = data.float64_values(outcome) else {
        diagnostics.push(Diagnostic::new(
            "tiered.evalue.unavailable",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "CoDetermined no-latent-to-outcome E-value could not be computed (outcome column missing)",
        ));
        return;
    };
    let mut n = 0usize;
    let mut sum = 0.0;
    for v in y.iter().copied() {
        if v.is_finite() {
            n += 1;
            sum += v;
        }
    }
    if n < 2 {
        diagnostics.push(Diagnostic::new(
            "tiered.evalue.unavailable",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "CoDetermined no-latent-to-outcome E-value could not be computed (need ≥2 finite outcome rows)",
        ));
        return;
    }
    let mean = sum / n as f64;
    let mut ss = 0.0;
    for v in y.iter().copied() {
        if v.is_finite() {
            ss += (v - mean) * (v - mean);
        }
    }
    let sd = (ss / (n - 1) as f64).sqrt();
    if !(sd.is_finite() && sd > 0.0) {
        diagnostics.push(Diagnostic::new(
            "tiered.evalue.unavailable",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "CoDetermined no-latent-to-outcome E-value could not be computed (outcome SD is not positive and finite)",
        ));
        return;
    }
    let std_diff = estimate.ate / sd;
    let mut rr = (0.91 * std_diff).exp();
    if rr < 1.0 {
        rr = 1.0 / rr;
    }
    let evalue = rr + (rr * (rr - 1.0)).sqrt();
    if !evalue.is_finite() {
        diagnostics.push(Diagnostic::new(
            "tiered.evalue.unavailable",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            "CoDetermined no-latent-to-outcome E-value overflowed",
        ));
        return;
    }
    estimate.evalue = Some(evalue);
    let mut report = Diagnostic::new(
        "tiered.evalue.vanderweele_approx",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "VanderWeele E-value approximation using ATE/SD(Y) (RR≈exp(0.91·Δ/s)) = {evalue}; this is not a tier-identification certificate. Tier-closure is a fast path on a complete-tier ADMG"
        ),
    );
    report.fields = Arc::from([
        (Arc::from("evalue"), Arc::from(evalue.to_string())),
        (Arc::from("method"), Arc::from("vanderweele_outcome_sd")),
        (Arc::from("not_a_tier_id_certificate"), Arc::from("true")),
    ]);
    diagnostics.push(report);
}

/// Evaluate each exceedance threshold on ConditionalEffect and attach `F_a(c)`.
pub(crate) fn attach_conditional_functional_grid(
    estimate: EffectEstimate,
    data: &TabularData,
    query: &ConditionalEffectQuery,
    estimand: &IdentifiedEstimand,
    ctx: &ExecutionContext,
) -> Result<EffectEstimate, CausalError> {
    let _ = ctx;
    let Some(thresholds) = query.inner.outcome_functional.thresholds() else {
        return Ok(estimate);
    };
    let est = antecedent_estimate::ConditionalLinearAdjustment::new();
    let mut cdf = Vec::with_capacity(thresholds.len() * 2);
    let mut columns = Vec::with_capacity(thresholds.len());
    let mut first = None;
    for &threshold in &thresholds {
        let data_c = apply_outcome_functional(
            data,
            query.inner.outcome,
            &OutcomeFunctional::exceedance(threshold),
        )?;
        let (point, arms) = est.estimate_with_means(&data_c, estimand, query)?;
        cdf.extend([1.0 - arms[0], 1.0 - arms[1]]);
        if let Some(inf) = point.influence.as_ref() {
            columns.push(inf.to_vec());
        }
        if first.is_none() {
            first = Some(point);
        }
    }
    let mut out = first.unwrap_or(estimate);
    if columns.len() != thresholds.len() {
        return Err(CausalError::Unsupported {
            message: "ConditionalEffect grid refused: missing influence columns for joint covariance",
        });
    }
    if columns.len() >= 2 {
        let refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
        out.joint_covariance = Some(antecedent_estimate::joint_influence_covariance(&refs, None)?);
    }
    out.exceedance_cdf = Some(Arc::from(cdf));
    if thresholds.len() > 1 {
        out.ate = f64::NAN;
        out.se_analytic = f64::NAN;
        out.se_bootstrap = None;
        out.influence = None;
        out.simultaneous_interval = None;
    }
    Ok(out)
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
    )
    .with_outcome_functional(query.outcome_functional.clone()))
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
