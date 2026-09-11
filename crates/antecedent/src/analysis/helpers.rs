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
    DiagnosticKind, DiagnosticSeverity, ExecutionContext, ExecutionPerformanceRecord, Intervention,
    InterventionSequence, LogicalAnalysisPlanRecord, OutcomeFunctional,
    PhysicalExecutionPlanRecord, ProvenanceGraph, ProvenanceNode, SequencedIntervention, VERSION,
    VariableId,
};
use antecedent_data::{IdRemap, TableView, TabularData, dedupe_variable_ids};
use antecedent_estimate::{
    AipwAte, CausalPosterior, EffectEstimate, EstimationWorkspace, OverlapPolicy, ScoreTable,
    crossfit_binary_scores, exceedance_cdf_values, inference_from_influence_columns,
    summarize_functional,
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
    let problem = RefutationProblem::new(data, estimand, query, estimate, Some(estimator), None);
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
        OutcomeFunctional::Quantile(_) | OutcomeFunctional::Mean | _ => Ok(data.clone()),
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
    shared: Option<&super::batch::SharedBatchDesign>,
) -> Result<Option<ScoreTable>, CausalError> {
    if !matches!(
        query.outcome_functional,
        OutcomeFunctional::Exceedance(_)
            | OutcomeFunctional::ExceedanceGrid(_)
            | OutcomeFunctional::Quantile(_)
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
    let mut problem = est.prepare(data, estimand, query)?;
    if let Some(shared) = shared {
        shared.apply_to_propensity(&mut problem)?;
    }
    if matches!(query.outcome_functional, OutcomeFunctional::Quantile(_)) {
        let grid = antecedent_estimate::empirical_threshold_grid(&problem.outcome, 19)?;
        return Ok(Some(antecedent_estimate::build_binary_scores(
            &problem,
            query.treatment,
            &grid.iter().copied().map(Some).collect::<Vec<_>>(),
            antecedent_estimate::DEFAULT_AIPW_FOLDS,
            &est.glm_options,
            est.backend,
        )?));
    }
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
    thresholds.dedup_by(|a, b| a.total_cmp(b).is_eq());
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
        if estimator_id == crate::strategy_table::EstimatorId::Aipw
            && matches!(
                query.outcome_functional,
                OutcomeFunctional::ExceedanceGrid(_) | OutcomeFunctional::Quantile(_)
            )
            && !matches!(query.target_population, antecedent_core::TargetPopulation::AllObserved)
        {
            return Err(CausalError::Unsupported {
                message: "exceedance grids and quantiles require AllObserved AIPW scores; apply target weights with prepare + retarget",
            });
        }
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
    let Some(table) = maybe_build_functional_scores(
        data,
        query,
        estimand,
        &config,
        study.shared_batch_design.as_deref(),
    )?
    else {
        if !query.outcome_functional.is_mean() {
            return Err(CausalError::Unsupported {
                message: "exceedance requires cross-fitted AIPW scores; refusing a first-threshold or mean substitute",
            });
        }
        return Ok(estimate);
    };
    if let Some(tau) = query.outcome_functional.quantile_level() {
        let (estimate, diagnostics) = attach_quantile_from_table(estimate, table, tau)?;
        extra_diagnostics.extend(diagnostics);
        return Ok(estimate);
    }
    let (estimate, diagnostics) = attach_score_functional_grid(estimate, table)?;
    extra_diagnostics.extend(diagnostics);
    Ok(estimate)
}

pub(crate) fn attach_quantile_from_table(
    mut estimate: EffectEstimate,
    table: ScoreTable,
    tau: f64,
) -> Result<(EffectEstimate, Vec<Diagnostic>), CausalError> {
    let qte = antecedent_estimate::quantile::quantile_contrast(&table, None, tau)?;
    let [q0, q1] = qte.quantiles;
    let [d0, d1] = qte.densities;
    let influence = qte.influence;
    let mut diagnostics = Vec::new();
    let n = influence.len() as f64;
    let mean = influence.iter().sum::<f64>() / n;
    let var = influence.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    estimate.ate = q1 - q0;
    estimate.se_analytic = (var / n).sqrt();
    estimate.influence = Some(influence.clone().into());
    estimate.se_bootstrap = None;
    diagnostics.push(Diagnostic::new(
        "estimate.functional.quantile",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "QTE at τ={tau}: Q_0={q0:.6} Q_1={q1:.6}; densities ({d0:.6}, {d1:.6}); \
             piecewise-linear AIPW CDF inversion conditional on the estimation grid; grid-selection uncertainty and interpolation bias are excluded"
        ),
    ));
    let (estimate, grid_diags) = attach_score_functional_grid(estimate, table)?;
    diagnostics.extend(
        grid_diags
            .into_iter()
            .filter(|d| d.code.as_ref() != "estimate.functional.grid_scalar_cleared"),
    );
    // attach_score_functional_grid clears multi-threshold ate; restore the QTE.
    let mut estimate = estimate;
    estimate.ate = q1 - q0;
    estimate.se_analytic = (var / n).sqrt();
    estimate.influence = Some(influence.into());
    Ok((estimate, diagnostics))
}

pub(crate) fn requested_joint_arm(
    query: &antecedent_core::ResponseQuery,
) -> Result<u32, CausalError> {
    let antecedent_core::ResponseFunctional::InterventionResponse { interventions, .. } =
        &query.functional
    else {
        return Err(CausalError::Unsupported { message: "joint quantile requires Set response" });
    };
    let mut arm = 0u32;
    for (j, iv) in interventions.iter().enumerate() {
        let Intervention::Set { value, .. } = iv else {
            return Err(CausalError::Unsupported {
                message: "joint quantile requires binary Set levels",
            });
        };
        let v = value.as_f64();
        if j >= 3 || !matches!(v, Some(0.0 | 1.0)) {
            return Err(CausalError::Unsupported {
                message: "joint quantile requires at most three binary Set levels",
            });
        }
        arm |= u32::from(v == Some(1.0)) << j;
    }
    Ok(arm)
}

pub(crate) fn attach_joint_quantile_from_table(
    estimate: EffectEstimate,
    table: ScoreTable,
    query: &antecedent_core::ResponseQuery,
    tau: f64,
) -> Result<(EffectEstimate, Vec<Diagnostic>), CausalError> {
    let q = antecedent_estimate::quantile::quantile_arm(
        &table,
        None,
        tau,
        requested_joint_arm(query)?,
    )?;
    let (mut out, mut diagnostics) = attach_score_functional_grid(estimate, table)?;
    out.ate = q.value;
    out.se_analytic = antecedent_estimate::joint_influence_covariance(&[&q.influence], None)?.se(0);
    out.influence = Some(q.influence.into());
    out.se_bootstrap = None;
    out.simultaneous_interval = None;
    diagnostics.retain(|d| d.code.as_ref() != "estimate.functional.grid_scalar_cleared");
    diagnostics.push(quantile_scope_diagnostic());
    Ok((out, diagnostics))
}

pub(crate) fn quantile_scope_diagnostic() -> Diagnostic {
    Diagnostic::new(
        "estimate.functional.quantile",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "piecewise-linear CDF inversion conditional on the frozen grid; interpolation bias and grid-selection uncertainty are excluded; a response quantile is a level, not an arm contrast",
    )
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

/// Bound and monotonize threshold-major two-arm CDF values. Covariance
/// remains covariance of the raw fitted contrasts, not these projected CDFs.
pub(crate) fn project_conditional_cdf(cdf: &mut [f64]) -> Result<bool, CausalError> {
    if cdf.len() % 2 != 0 || cdf.iter().any(|x| !x.is_finite()) {
        return Err(CausalError::Compile { message: "invalid conditional CDF grid".into() });
    }
    let mut changed = false;
    for arm in 0..2 {
        let raw: Vec<_> = cdf.iter().skip(arm).step_by(2).copied().collect();
        let projected = antecedent_estimate::monotone_increasing(&raw);
        for (slot, value) in cdf.iter_mut().skip(arm).step_by(2).zip(projected) {
            let bounded = value.clamp(0.0, 1.0);
            changed |= slot.total_cmp(&bounded).is_ne();
            *slot = bounded;
        }
    }
    Ok(changed)
}

/// Choose one outcome grid from complete rows for every contributing conditional atom.
pub(crate) fn conditional_thresholds(
    data: &TabularData,
    query: &ConditionalEffectQuery,
    adjustment: impl Iterator<Item = VariableId>,
) -> Result<Option<Vec<f64>>, CausalError> {
    if query.inner.outcome_functional.quantile_level().is_none() {
        return Ok(query.inner.outcome_functional.thresholds());
    }
    if !matches!((&query.inner.control, &query.inner.active),
        (Intervention::Set { value: c, .. }, Intervention::Set { value: a, .. })
        if c.as_f64() == Some(0.0) && a.as_f64() == Some(1.0))
    {
        return Err(CausalError::Unsupported {
            message: "conditional quantiles require binary 0/1 arms; a linear mean model is not a distribution model",
        });
    }
    let mut ids: Vec<_> = adjustment.chain(query.inner.effect_modifiers.iter().copied()).collect();
    ids.extend([query.inner.treatment, query.inner.outcome]);
    let mask = data.complete_case_mask(&ids)?;
    let y = data.float64_masked(query.inner.outcome, &mask)?;
    Ok(Some(antecedent_estimate::empirical_threshold_grid(&y, 19)?))
}

/// Publish the actual conditional grid, whose coordinates have no score-table payload.
pub(crate) fn conditional_quantile_grid_diagnostic(
    data: &TabularData,
    query: &ConditionalEffectQuery,
    adjustment: impl Iterator<Item = VariableId>,
) -> Result<Option<Diagnostic>, CausalError> {
    if query.inner.outcome_functional.quantile_level().is_none() {
        return Ok(None);
    }
    let thresholds = conditional_thresholds(data, query, adjustment)?.unwrap_or_default();
    Ok(Some(Diagnostic::new(
        "estimate.functional.quantile_grid",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "frozen empirical thresholds: {thresholds:?}; CDF coordinates are threshold-major, control then active"
        ),
    )))
}

/// Invert the two raw CDF arms, after any frozen-weight atom mixture.
pub(crate) fn attach_conditional_quantile(
    out: &mut EffectEstimate,
    thresholds: &[f64],
    raw_cdf: &[f64],
    columns: &[Vec<f64>],
    supported: &[bool],
    tau: f64,
) -> Result<(), CausalError> {
    let mut arms = Vec::new();
    for arm in 0..2 {
        let means: Vec<_> = raw_cdf.iter().skip(arm).step_by(2).copied().collect();
        let support: Vec<_> = supported.iter().skip(arm).step_by(2).copied().collect();
        let influences: Vec<Vec<f64>> = columns
            .iter()
            .skip(arm)
            .step_by(2)
            .map(|c| {
                let mean = c.iter().sum::<f64>() / c.len() as f64;
                c.iter().map(|v| v - mean).collect()
            })
            .collect();
        arms.push(antecedent_estimate::quantile::invert_supported_cdf(
            thresholds,
            &means,
            &influences,
            &support,
            tau,
        )?);
    }
    let influence: Vec<_> =
        arms[1].influence.iter().zip(&arms[0].influence).map(|(a, b)| a - b).collect();
    out.ate = arms[1].value - arms[0].value;
    out.se_analytic = antecedent_estimate::joint_influence_covariance(&[&influence], None)?.se(0);
    out.influence = Some(influence.into());
    out.se_bootstrap = None;
    out.simultaneous_interval = None;
    Ok(())
}

/// Evaluate each exceedance threshold on ConditionalEffect and attach `F_a(c)`.
///
/// Joint covariance and simultaneous bands are the 2K raw per-arm CDF
/// coordinates. Sparse tails refuse a finite band rather than publishing an
/// empty-cell or first-threshold SE.
pub(crate) fn attach_conditional_functional_grid(
    estimate: EffectEstimate,
    data: &TabularData,
    query: &ConditionalEffectQuery,
    estimand: &IdentifiedEstimand,
    ctx: &ExecutionContext,
) -> Result<EffectEstimate, CausalError> {
    let _ = ctx;
    let Some(thresholds) =
        conditional_thresholds(data, query, estimand.adjustment_set.iter().copied())?
    else {
        return Ok(estimate);
    };
    let y_orig = data.float64_values(query.inner.outcome).map_err(CausalError::from)?;
    let est = antecedent_estimate::ConditionalLinearAdjustment::new();
    let mut raw_cdf = Vec::with_capacity(thresholds.len() * 2);
    let mut columns = Vec::with_capacity(thresholds.len() * 2);
    let mut event_n_eff = Vec::with_capacity(thresholds.len() * 2);
    let mut threshold_supported = Vec::with_capacity(thresholds.len() * 2);
    let mut first = None;
    let mut n_eff_by_arm = [0.0, 0.0];
    for &threshold in &thresholds {
        let data_c = apply_outcome_functional(
            data,
            query.inner.outcome,
            &OutcomeFunctional::exceedance(threshold),
        )?;
        let mut transformed_query = query.clone();
        transformed_query.inner.outcome_functional = OutcomeFunctional::Mean;
        let (point, scores) =
            est.estimate_with_arm_scores(&data_c, estimand, &transformed_query)?;
        raw_cdf.extend([1.0 - scores.means[0], 1.0 - scores.means[1]]);
        if scores.influence[0].len() != scores.influence[1].len()
            || scores.influence[0].len() != scores.row_index.len()
        {
            return Err(CausalError::Unsupported {
                message: "ConditionalEffect grid refused: missing per-arm influence for joint covariance",
            });
        }
        for arm in 0..2 {
            columns.push(scores.influence[arm].iter().map(|v| -v).collect::<Vec<_>>());
            let (events, supported) =
                tail_event_support(&scores.treatment, &scores.row_index, &y_orig, arm, threshold);
            event_n_eff.push(events);
            threshold_supported.push(supported);
        }
        n_eff_by_arm = arm_n_eff(&scores.treatment);
        if first.is_none() {
            first = Some(point);
        }
    }
    let mut out = first.unwrap_or(estimate);
    if columns.len() != thresholds.len() * 2 {
        return Err(CausalError::Unsupported {
            message: "ConditionalEffect grid refused: missing influence columns for joint covariance",
        });
    }
    let refs: Vec<&[f64]> = columns.iter().map(Vec::as_slice).collect();
    out.joint_covariance = Some(antecedent_estimate::joint_influence_covariance(&refs, None)?);
    let n_eff = n_eff_by_arm[0] + n_eff_by_arm[1];
    out.score_inference = Some(inference_from_influence_columns(
        &raw_cdf,
        &refs,
        &event_n_eff,
        &threshold_supported,
        antecedent_estimate::WeightedSupport {
            n_eff,
            n_eff_by_arm: n_eff_by_arm.to_vec(),
            propensity_range: None,
            overlap_ok: n_eff_by_arm
                .iter()
                .all(|n| *n >= antecedent_estimate::scores::MIN_THRESHOLD_EVENTS),
        },
    )?);
    if thresholds.len() == 1 && threshold_supported.iter().any(|&supported| !supported) {
        out.se_analytic = f64::NAN;
        out.se_bootstrap = None;
        out.influence = None;
        out.simultaneous_interval = None;
    }
    let mut cdf = raw_cdf.clone();
    let rearranged = project_conditional_cdf(&mut cdf)?;
    out = out.with_monotone_rearranged(rearranged);
    out.exceedance_cdf = Some(Arc::from(cdf));
    if thresholds.len() > 1 {
        out.ate = f64::NAN;
        out.se_analytic = f64::NAN;
        out.se_bootstrap = None;
        out.influence = None;
        out.simultaneous_interval = None;
    }
    if let Some(tau) = query.inner.outcome_functional.quantile_level() {
        attach_conditional_quantile(
            &mut out,
            &thresholds,
            &raw_cdf,
            &columns,
            &threshold_supported,
            tau,
        )?;
    }
    Ok(out)
}

pub(crate) fn tail_event_support(
    treatment: &[f64],
    row_index: &[u32],
    outcome: &[f64],
    arm: usize,
    threshold: f64,
) -> (f64, bool) {
    let target = if arm == 0 { 0.0 } else { 1.0 };
    let mut events = 0.0;
    let mut non_events = 0.0;
    for (i, &row) in row_index.iter().enumerate() {
        let Some(&t) = treatment.get(i) else {
            continue;
        };
        if (t - target).abs() > 1e-12 {
            continue;
        }
        let Some(&y) = outcome.get(row as usize) else {
            continue;
        };
        if !y.is_finite() {
            continue;
        }
        if y > threshold {
            events += 1.0;
        } else {
            non_events += 1.0;
        }
    }
    (
        events,
        events >= antecedent_estimate::scores::MIN_THRESHOLD_EVENTS
            && non_events >= antecedent_estimate::scores::MIN_THRESHOLD_EVENTS,
    )
}

fn arm_n_eff(treatment: &[f64]) -> [f64; 2] {
    let mut counts = [0.0, 0.0];
    for &t in treatment {
        if t.abs() <= 1e-12 {
            counts[0] += 1.0;
        } else if (t - 1.0).abs() <= 1e-12 {
            counts[1] += 1.0;
        }
    }
    counts
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

#[cfg(test)]
mod cdf_review_tests {
    #[test]
    fn conditional_cdf_is_bounded_and_monotone_in_each_arm() {
        let mut values = [-0.2, 0.8, 0.7, 0.3, 1.3, 1.2];
        assert!(super::project_conditional_cdf(&mut values).unwrap());
        assert!(
            values.iter().zip([0.0, 0.55, 0.7, 0.55, 1.0, 1.0]).all(|(a, b)| (a - b).abs() < 1e-12)
        );
        assert!(!super::project_conditional_cdf(&mut values).unwrap());
        assert!(super::project_conditional_cdf(&mut [f64::NAN, 0.5]).is_err());
    }
}
