//! Unified `Study` facade.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

//! Private execution helpers.

use std::sync::Arc;

use antecedent_contract::{
    BufferMaterialization, ExecutionPerformanceRecord, LogicalAnalysisPlanRecord,
    PhysicalExecutionPlanRecord, ProvenanceGraph, ProvenanceNode,
};
use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ConditionalEffectQuery, Diagnostic, DiagnosticKind,
    DiagnosticSeverity, ExecutionContext, Intervention, InterventionSequence, OutcomeFunctional,
    SequencedIntervention, VERSION, VariableId,
};
use antecedent_data::{IdRemap, TableView, TabularData, dedupe_variable_ids};
use antecedent_estimate::{
    AipwAte, CausalPosterior, EffectEstimate, EstimationWorkspace, OverlapPolicy, ScoreTable,
    crossfit_binary_scores, exceedance_cdf_values, inference_from_influence_columns,
    summarize_functional,
};
use antecedent_expr::{IdentifiedEstimand, RdDesignParams};
use antecedent_validate::{EValue, RefutationProblem, RefutationReport, ValidationSuite};

use crate::error::CausalError;
use crate::result::StudyResult;

use super::builder::RefuteSuite;

mod assembly;

pub(crate) use assembly::{AssembleArgs, assemble_result, provenance_pair, stamp_result_context};

mod refutation;

pub(crate) use refutation::{
    RefutationRun, validator_not_applicable_diagnostic, validator_not_applicable_diagnostics,
};

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

/// 0.95 credible interval of a one-column posterior, or of its effect column.
///
/// A checked ADMG intervention rebuilds the retained posterior as a single
/// scalar quantity, so the effect-column name is gone and column 0 is the
/// intervention mean. A missing or non-finite summary publishes no interval.
pub(crate) fn credible_scalar_uncertainty(
    posterior: &CausalPosterior,
) -> Option<antecedent_core::ResponseUncertainty> {
    let column = posterior
        .effect_column()
        .or_else(|| (posterior.draws.schema.n_quantities() == 1).then_some(0))?;
    let sd = *posterior.summaries.sd.get(column)?;
    let lower = *posterior.summaries.q025.get(column)?;
    let upper = *posterior.summaries.q975.get(column)?;
    if ![sd, lower, upper].iter().all(|value| value.is_finite()) {
        return None;
    }
    Some(antecedent_core::ResponseUncertainty::Scalar {
        standard_error: sd,
        level: crate::result::REPORTED_SE_INTERVAL_LEVEL,
        lower,
        upper,
        interpretation: antecedent_core::IntervalInterpretation::Credible,
        draws: None,
    })
}

/// Pointwise 0.95 credible band, one column of `posterior` per grid level.
///
/// The checked ADMG curve stores each level's intervention-mean draws as its
/// own scalar quantity, in grid order. A non-finite endpoint publishes no band.
pub(crate) fn credible_pointwise_uncertainty(
    posterior: &CausalPosterior,
) -> Option<antecedent_core::ResponseUncertainty> {
    let n = posterior.draws.schema.n_quantities();
    if n == 0 {
        return None;
    }
    let lower: Vec<f64> = posterior.summaries.q025.iter().take(n).copied().collect();
    let upper: Vec<f64> = posterior.summaries.q975.iter().take(n).copied().collect();
    if lower.len() != n
        || upper.len() != n
        || lower.iter().chain(upper.iter()).any(|value| !value.is_finite())
    {
        return None;
    }
    Some(antecedent_core::ResponseUncertainty::PointwiseBand {
        level: crate::result::REPORTED_SE_INTERVAL_LEVEL,
        lower: Arc::from(lower),
        upper: Arc::from(upper),
        interpretation: antecedent_core::IntervalInterpretation::Credible,
        draws: None,
    })
}

/// Share of rows whose raw propensity a clip moves above which the clip is said to bind.
const CLIP_BINDING_FRACTION: f64 = 0.05;
/// Distance from 0 or 1 within which a raw propensity is saturated.
const SATURATION_MARGIN: f64 = 1e-8;

/// Diagnostic recording that the applied overlap policy changed the answer's target or
/// weights, which limits its empirical support ([`antecedent_io::OVERLAP_LIMITED_DIAGNOSTIC`]).
///
/// A clip binds when it moves a saturated propensity (within `1e-8` of 0 or 1) or at least
/// 5% of the rows; a trim binds when it excludes any row. `None` when no report exists or the
/// policy did not bind.
pub(crate) fn overlap_limit_diagnostic(
    report: Option<&antecedent_estimate::OverlapReport>,
) -> Option<Diagnostic> {
    let report = report?;
    let clipped = report.clip.map_or(0.0, |_| 1.0 - report.target_population_support);
    let saturated = report.propensity_min <= SATURATION_MARGIN
        || report.propensity_max >= 1.0 - SATURATION_MARGIN;
    let clip_binds = report.clip.is_some() && (saturated || clipped >= CLIP_BINDING_FRACTION);
    let trim_binds = report.trim.is_some() && report.excluded_fraction > 0.0;
    if !(clip_binds || trim_binds) {
        return None;
    }
    let message = format!(
        "the overlap policy changed the answer: the clip moved {:.1}% of propensities{}; the trim \
         excluded {:.1}% of rows. The estimate holds for the clipped weights or the retained \
         common-support rows, not the full declared population",
        100.0 * clipped,
        if saturated { ", including scores within 1e-8 of 0 or 1" } else { "" },
        100.0 * report.excluded_fraction,
    );
    Some(
        Diagnostic::new(
            antecedent_io::OVERLAP_LIMITED_DIAGNOSTIC,
            DiagnosticKind::Support,
            DiagnosticSeverity::Warning,
            message,
        )
        .with_fields([
            ("clipped_fraction", clipped.to_string()),
            ("excluded_fraction", report.excluded_fraction.to_string()),
            ("saturated", saturated.to_string()),
        ]),
    )
}

/// Diagnostic recording that the resampling interval silently dropped replicates, which
/// limits its support ([`antecedent_io::BOOTSTRAP_LIMITED_DIAGNOSTIC`]).
///
/// Fires when any replicate failed, or when fewer succeeded than were requested and the loop
/// was not stopped early on purpose. `None` when no bootstrap ran or every replicate survived.
pub(crate) fn bootstrap_limit_diagnostic(
    requested: Option<u32>,
    ok: Option<u32>,
    failed: Option<u32>,
    early_stopped: bool,
) -> Option<Diagnostic> {
    let failed_count = failed.unwrap_or(0);
    let shortfall = match (requested, ok) {
        (Some(requested), Some(ok)) if !early_stopped && ok < requested => requested - ok,
        _ => 0,
    };
    if failed_count == 0 && shortfall == 0 {
        return None;
    }
    let requested_text = requested.map_or_else(|| "unrecorded".to_owned(), |n| n.to_string());
    let ok_text = ok.map_or_else(|| "unrecorded".to_owned(), |n| n.to_string());
    Some(
        Diagnostic::new(
            antecedent_io::BOOTSTRAP_LIMITED_DIAGNOSTIC,
            DiagnosticKind::Support,
            DiagnosticSeverity::Warning,
            format!(
                "the resampling interval dropped replicates: {requested_text} requested, \
                 {ok_text} succeeded, {failed_count} failed. The interval rests on the \
                 surviving replicates only"
            ),
        )
        .with_fields([
            ("replicates_requested", requested_text),
            ("replicates_ok", ok_text),
            ("replicates_failed", failed_count.to_string()),
        ]),
    )
}

/// Diagnostic recording that a returned posterior's sampler or optimizer diagnostics fall short
/// of the publication criteria ([`antecedent_io::MCMC_LIMITED_DIAGNOSTIC`]).
///
/// An MCMC factorization is judged by the full publication predicate; any other posterior that
/// records sampler measurements (split R-hat, bulk ESS) is judged by split R-hat at most 1.01
/// and bulk ESS at least 100 per chain, and every posterior by its `converged` flag. `None`
/// when every recorded diagnostic is adequate.
pub(crate) fn sampler_limit_diagnostic(
    diagnostics: &antecedent_prob::InferenceDiagnostics,
) -> Option<Diagnostic> {
    let mut reasons: Vec<String> = Vec::new();
    if diagnostics.factorization == antecedent_prob::HessianFactorization::Mcmc {
        reasons.extend(diagnostics.mcmc_publication_failures().into_iter().map(str::to_owned));
    } else {
        if let Some(rhat) = diagnostics.rhat_max.filter(|r| !r.is_finite() || *r > 1.01) {
            reasons.push(format!("split R-hat {rhat} is above 1.01"));
        }
        let floor = 100.0 * f64::from(diagnostics.n_chains.unwrap_or(1));
        if let Some(ess) = diagnostics.ess_bulk_min.filter(|e| !e.is_finite() || *e < floor) {
            reasons.push(format!("bulk ESS {ess} is below {floor} (100 per chain)"));
        }
    }
    if !diagnostics.converged && reasons.is_empty() {
        reasons.push("the optimizer or sampler did not report convergence".to_owned());
    }
    if reasons.is_empty() {
        return None;
    }
    Some(
        Diagnostic::new(
            antecedent_io::MCMC_LIMITED_DIAGNOSTIC,
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "the posterior was returned with weak sampler diagnostics: {}. Treat its draws \
                 and intervals as provisional",
                reasons.join("; ")
            ),
        )
        .with_fields([
            (
                "rhat_max",
                diagnostics.rhat_max.map_or_else(|| "unrecorded".into(), |v| v.to_string()),
            ),
            (
                "ess_bulk_min",
                diagnostics.ess_bulk_min.map_or_else(|| "unrecorded".into(), |v| v.to_string()),
            ),
            ("converged", diagnostics.converged.to_string()),
            ("reasons", reasons.join("; ")),
        ]),
    )
}

/// Diagnostic recording which overlap policy an estimator applied.
pub(crate) fn overlap_diagnostic(overlap: OverlapPolicy) -> Diagnostic {
    match overlap.bounds() {
        None => Diagnostic::new(
            "estimate.overlap.explicit_override",
            DiagnosticKind::Support,
            DiagnosticSeverity::Info,
            "estimator used ExplicitOverride for positivity (not a propensity-based method)",
        ),
        Some(_) => Diagnostic::new(
            "estimate.overlap.require_diagnostics",
            DiagnosticKind::Support,
            DiagnosticSeverity::Info,
            "estimator used RequireDiagnostics for mandatory positivity diagnostics",
        ),
    }
}

/// Replace `Y` with `1{Y > c}` when the query asks for a single exceedance.
///
/// Grids are not reduced to the first threshold. Supported grid paths build a
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
        OutcomeFunctional::Quantile(_) | OutcomeFunctional::Mean => Ok(data.clone()),
        _ => Err(CausalError::Unsupported {
            message: "this outcome functional has no data reduction on this path",
        }),
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
    fold_seed: u64,
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
    // The recorded master seed governs the fold plan, not only the learners.
    problem.fold_seed = fold_seed;
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
    let n_thresholds = table.distinct_threshold_count();
    if table.intervened.is_empty() && summary.means.len() >= 2 && n_thresholds <= 1 {
        let mut coefficients = vec![0.0; table.n_columns()];
        coefficients[0] = -1.0;
        coefficients[1] = 1.0;
        let contrast = table.linear_contrast(&summary, &coefficients)?;
        estimate.ate = contrast.value;
        estimate.uncertainty.analytic.se = contrast.se;
        estimate.scores.influence = Some(
            table
                .column(0)?
                .iter()
                .zip(table.column(1)?)
                .map(|(a, b)| b - a)
                .collect::<Vec<_>>()
                .into(),
        );
        if estimate.se_bootstrap().is_some() {
            diagnostics.push(Diagnostic::new(
                "estimate.functional.crossfit_inference",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                "cross-fitted grid reports joint influence uncertainty; bootstrap from the earlier full-sample fit is not reused",
            ));
        }
    } else if n_thresholds > 1 {
        estimate.ate = f64::NAN;
        estimate.uncertainty.analytic.se = f64::NAN;
        estimate.uncertainty.bootstrap.se = None;
        estimate.scores.influence = None;
        estimate.family.simultaneous_interval = None;
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
    estimate.scores.inference = Some(inference);
    Ok((estimate, diagnostics))
}

/// The data, query and estimand an average-effect functional is computed for.
#[derive(Clone, Copy)]
pub(crate) struct AverageTarget<'a> {
    pub(crate) data: &'a TabularData,
    pub(crate) query: &'a AverageEffectQuery,
    pub(crate) estimand: &'a IdentifiedEstimand,
}

/// Build scores on original `Y` and attach the exceedance grid when supported.
pub(crate) fn attach_average_functional_grid(
    estimate: EffectEstimate,
    target: AverageTarget<'_>,
    extra_diagnostics: &mut Vec<Diagnostic>,
    estimator_id: crate::strategy_table::EstimatorId,
    study: &super::execute::Study,
    fold_seed: u64,
) -> Result<EffectEstimate, CausalError> {
    let AverageTarget { data, query, estimand } = target;
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
    let mut config = match study.declared.estimator_spec.as_deref() {
        Some(crate::estimator_spec::EstimatorSpec::Aipw(config)) => *config.clone(),
        _ => AipwAte::new(),
    };
    if let Some(design) = study.declared.sampling_design {
        config.inference.sampling_design =
            antecedent_estimate::util::SamplingDeclaration::Declared(design);
    }
    if let Some(overlap) = study.declared.overlap_policy {
        config.overlap = overlap;
    }
    if !matches!(query.outcome_functional, OutcomeFunctional::Mean)
        && (matches!(
            config.overlap.canonical(),
            OverlapPolicy::RequireDiagnostics { trim: Some(_), .. }
        ) || config.inference.se_kind != antecedent_estimate::AnalyticSeKind::Homoskedastic)
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
        fold_seed,
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
    let se = antecedent_stats::iid_se(&influence);
    estimate.ate = q1 - q0;
    estimate.uncertainty.analytic.se = se;
    estimate.scores.influence = Some(influence.clone().into());
    estimate.uncertainty.bootstrap.se = None;
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
    estimate.uncertainty.analytic.se = se;
    estimate.scores.influence = Some(influence.into());
    Ok((estimate, diagnostics))
}

/// The joint-cell arm index of a binary `Set` response: the one decoder shared by quantile,
/// batch family-contrast and retarget paths.
pub(crate) fn requested_joint_arm(
    query: &antecedent_core::ResponseQuery,
) -> Result<u32, CausalError> {
    let antecedent_core::ResponseFunctional::InterventionResponse { interventions, .. } =
        &query.functional
    else {
        return Err(CausalError::Unsupported {
            message: "joint-cell response requires Set response",
        });
    };
    let mut arm = 0u32;
    for (j, iv) in interventions.iter().enumerate() {
        let Intervention::Set { value, .. } = iv else {
            return Err(CausalError::Unsupported {
                message: "joint-cell response requires binary Set levels",
            });
        };
        let v = value.as_f64();
        if j >= antecedent_estimate::cell_aipw::MAX_JOINT_BINARY || !matches!(v, Some(0.0 | 1.0)) {
            return Err(CausalError::Unsupported {
                message: "joint-cell response requires binary Set levels on a joint cell code's coordinates",
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
    out.uncertainty.analytic.se =
        antecedent_estimate::joint_influence_covariance(&[&q.influence], None)?.se(0);
    out.scores.influence = Some(q.influence.into());
    out.uncertainty.bootstrap.se = None;
    out.family.simultaneous_interval = None;
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

/// Mirror the E-value the `sensitivity.evalue` refuter reported onto the typed
/// estimate fields, so a consumer reads the number and the threshold it was judged
/// against without parsing report prose.
///
/// The refuter stays the single owner of the number: this copies
/// [`antecedent_validate::RefutationReport::comparison`] verbatim and pairs it with
/// the threshold of the very [`EValue`] the suite constructs. Nothing is recomputed.
/// When the refuter did not run, a value this function previously mirrored is cleared
/// (a threshold-less E-value from another premise is left alone).
pub(crate) fn mirror_refuted_evalue(
    estimate: &mut crate::PrimaryEstimate,
    refutations: &[RefutationReport],
) {
    let Some(estimate) = estimate.as_effect_mut() else {
        return;
    };
    match refutations.iter().find(|r| r.refuter.as_ref() == EVALUE_REFUTER) {
        Some(report) => {
            estimate.diagnostics.evalue = Some(report.comparison);
            estimate.diagnostics.evalue_threshold = Some(EValue::new().threshold);
        }
        None if estimate.diagnostics.evalue_threshold.is_some() => {
            estimate.diagnostics.evalue = None;
            estimate.diagnostics.evalue_threshold = None;
        }
        None => {}
    }
}

/// Report name the [`EValue`] validator publishes.
pub(crate) const EVALUE_REFUTER: &str = "sensitivity.evalue";

/// Attach the tier cell's point E-value to a finalized estimate when the executed study is
/// a `CoDetermined` tiered one and its functional is not a quantile.
///
/// The single predicate shared by the fresh-run finalizer and the prepared lifecycle, so a
/// result carries the same point E-value however it was produced.
pub(crate) fn attach_tiered_evalue_for_cell(
    estimate: &mut crate::PrimaryEstimate,
    data: &TabularData,
    outcome: VariableId,
    is_quantile: bool,
    tiered: Option<&antecedent_graph::TieredBackground>,
    diagnostics: &mut Vec<Diagnostic>,
) {
    let Some(estimate) = estimate.as_effect_mut() else {
        return;
    };
    if !is_quantile
        && tiered.is_some_and(|b| b.within_tier == antecedent_graph::WithinTier::CoDetermined)
    {
        attach_tiered_evalue(estimate, data, outcome, diagnostics);
    }
}

/// Attach a point E-value for the tier-closure no-latent-to-outcome premise.
fn attach_tiered_evalue(
    estimate: &mut EffectEstimate,
    data: &TabularData,
    outcome: VariableId,
    diagnostics: &mut Vec<Diagnostic>,
) {
    if estimate.diagnostics.evalue.is_some() || !estimate.ate.is_finite() {
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
    estimate.diagnostics.evalue = Some(evalue);
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

mod conditional;

pub(crate) use conditional::{
    attach_conditional_functional_grid, attach_conditional_quantile,
    conditional_quantile_grid_diagnostic, conditional_score_estimator_diagnostic,
    conditional_thresholds, conditional_uses_crossfit_aipw, project_conditional_cdf,
};

pub(crate) fn tail_event_support(
    treatment: &[f64],
    row_index: &[antecedent_data::GlobalId],
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
        let Some(&y) = outcome.get(usize::try_from(row.get()).expect("row id")) else {
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
        Intervention::Transform { variable, expr, .. } => {
            let expr = expr.try_map_variables(&mut |v| remap.map(v))?;
            Ok(Intervention::transform(remap.map(*variable)?, expr))
        }
        Intervention::Assign { variable, expr, .. } => {
            let expr = expr.try_map_variables(&mut |v| remap.map(v))?;
            Ok(Intervention::assign(remap.map(*variable)?, expr))
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

/// Diagnostic when a latency tier's replicate count was set aside for the
/// count a configured estimator owns. `tier` is the tier's count and
/// `configured` the count that ran.
pub(crate) fn latency_bootstrap_not_applied_diagnostic(tier: u32, configured: u32) -> Diagnostic {
    Diagnostic {
        code: Arc::from("latency.bootstrap_not_applied"),
        kind: DiagnosticKind::Scientific,
        severity: DiagnosticSeverity::Info,
        message: Arc::from(format!(
            "the latency tier maps to {tier} bootstrap replicates, but the configured \
             estimator owns its replicate count; {configured} replicates ran"
        )),
        artifact_id: None,
        fields: Arc::from([
            (Arc::from("tier_replicates"), Arc::from(tier.to_string())),
            (Arc::from("configured_replicates"), Arc::from(configured.to_string())),
        ]),
    }
}

/// Full-suite prior sensitivity around the prior actually in force.
///
/// - external prior-bank compose → α-multiplier grid on the post-conflict alphas;
/// - any other resolved prior on `est` (explicit `cfg.prior`, a transferred
///   `cfg.prior_artifact`) → variance-multiplier grid around that prior;
/// - no prior supplied → isotropic scale grid (the isotropic prior *is* the prior in force).
///
/// `est.prior` must hold the resolved prior the reported posterior used; the
/// summary's [`antecedent_prob::PriorSensitivityFamily`] records which family was perturbed.
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
        // An explicit or transferred prior gets its variance-multiplier grid; with none the
        // isotropic scale grid is the prior in force.
        let sens = PriorSensitivity::for_estimator(est);
        let (summary, _) =
            sens.evaluate_in_force(est, prep, status, ws, ctx).map_err(CausalError::from)?;
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

    fn field<'a>(diagnostic: &'a antecedent_core::Diagnostic, key: &str) -> &'a str {
        diagnostic.fields.iter().find(|(k, _)| &**k == key).map(|(_, v)| &**v).unwrap()
    }

    #[test]
    fn dropped_bootstrap_replicates_limit_support() {
        let d = super::bootstrap_limit_diagnostic(Some(200), Some(190), Some(10), false).unwrap();
        assert_eq!(&*d.code, antecedent_io::BOOTSTRAP_LIMITED_DIAGNOSTIC);
        assert_eq!(d.severity, antecedent_core::DiagnosticSeverity::Warning);
        assert_eq!(field(&d, "replicates_requested"), "200");
        assert_eq!(field(&d, "replicates_ok"), "190");
        assert_eq!(field(&d, "replicates_failed"), "10");
        // A shortfall with no recorded failure still limits support; a deliberate early stop,
        // a clean run and no bootstrap do not.
        assert!(super::bootstrap_limit_diagnostic(Some(200), Some(150), None, false).is_some());
        assert!(super::bootstrap_limit_diagnostic(Some(200), Some(150), Some(0), true).is_none());
        assert!(super::bootstrap_limit_diagnostic(Some(200), Some(200), Some(0), false).is_none());
        assert!(super::bootstrap_limit_diagnostic(None, None, None, false).is_none());
    }

    #[test]
    fn weak_sampler_diagnostics_limit_support() {
        let mut inference = antecedent_prob::InferenceDiagnostics::analytic("test");
        assert!(super::sampler_limit_diagnostic(&inference).is_none());
        inference.n_chains = Some(4);
        inference.rhat_max = Some(1.2);
        inference.ess_bulk_min = Some(800.0);
        let d = super::sampler_limit_diagnostic(&inference).unwrap();
        assert_eq!(&*d.code, antecedent_io::MCMC_LIMITED_DIAGNOSTIC);
        assert_eq!(field(&d, "rhat_max"), "1.2");
        inference.rhat_max = Some(1.0);
        assert!(super::sampler_limit_diagnostic(&inference).is_none());
        inference.ess_bulk_min = Some(50.0);
        assert!(super::sampler_limit_diagnostic(&inference).is_some());
        inference.ess_bulk_min = Some(800.0);
        inference.converged = false;
        assert!(super::sampler_limit_diagnostic(&inference).is_some());
    }
}
