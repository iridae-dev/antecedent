//! Mechanism-refit uncertainty for checked counterfactuals.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    Arc, CausalError, CausalPosterior, Diagnostic, EffectEstimate, ExecutionContext,
    IdentificationStatus, OverlapPolicy, StreamDomain, TableView, TabularData, UnitContrast,
    map_mechanism_fit,
};

/// Inputs every Dirichlet row-weight refit of a counterfactual contrast reads.
pub(super) struct CounterfactualRefit<'a> {
    pub(super) data: &'a TabularData,
    pub(super) fitted: &'a crate::gcm::FittedGcm,
    pub(super) contrast: UnitContrast,
    pub(super) assumptions: &'a antecedent_core::AssumptionSet,
    pub(super) status: IdentificationStatus,
}

/// Bayesian-bootstrap refit of the mechanisms per draw, publishing the mean-ITE posterior and per-unit intervals.
pub(super) fn dirichlet_counterfactual_estimate(
    refit: &CounterfactualRefit<'_>,
    ite: crate::gcm::IteResult,
    n_draws: usize,
    ctx: &ExecutionContext,
) -> Result<(EffectEstimate, Option<CausalPosterior>, crate::gcm::IteResult), CausalError> {
    let CounterfactualRefit { data, fitted, contrast, assumptions, status } = *refit;
    let UnitContrast { treatment, outcome, active, control } = contrast;
    let base_model = &fitted.model;
    let mechanism_assignments = &fitted.assignments;
    let n_units = ite.unit_effects.len();
    let mut values = Vec::with_capacity(n_draws);
    // Column-major `units × draws`: one posterior column per unit, read by
    // the same summary that publishes the mean-ITE interval.
    let mut unit_draws = vec![0.0; n_units * n_draws];
    let mut rng = ctx.rng.stream_for(StreamDomain::Attribution, 0x0CF0);
    // One stream through every draw: a categorical node's rank-preserving noise is drawn once
    // per posterior draw of the mechanisms, independently across draws.
    let mut noise_rng = ctx.rng.stream_for(StreamDomain::Counterfactual, 0x0CF1);
    let n = data.row_count();
    for draw in 0..n_draws {
        if ctx.cancellation.is_cancelled() {
            return Err(CausalError::Cancelled {
                stage: super::super::super::stage::STAGE_ESTIMATE_POINT,
            });
        }
        let weights = antecedent_kernels::bayesian_bootstrap_weights(&mut rng, n, false);
        // The same registry the point fit selected from, so every family a
        // node can have selected is one `refit_weighted` accepts.
        let store = crate::gcm::MechanismRegistry::with_heterogeneity_families()
            .refit_weighted(base_model, data, mechanism_assignments, &weights)
            .map_err(|e| match e {
                crate::gcm::ModelError::NotConverged { .. } => map_mechanism_fit(e),
                other => CausalError::rejected(other),
            })?;
        let draw_model = base_model.clone().with_mechanisms(store);
        let draw_ite = crate::gcm::counterfactual_ite_with_noise(
            draw_model,
            data,
            (treatment, outcome, active, control),
            ctx,
            &mut noise_rng,
        )?;
        if draw_ite.unit_effects.len() != n_units {
            return Err(CausalError::Compile {
                message: "counterfactual Bayesian draw changed the unit set".into(),
            });
        }
        values.push(draw_ite.mean_ite);
        for (unit, effect) in draw_ite.unit_effects.iter().enumerate() {
            unit_draws[unit * n_draws + draw] = *effect;
        }
    }
    let posterior = counterfactual_posterior(values, assumptions.clone(), status)?;
    let eq = posterior.effect_column().ok_or_else(|| CausalError::Compile {
        message: "counterfactual posterior missing effect column".into(),
    })?;
    // Published unit effects stay the plain draw average (sequential sum,
    // divided once), exactly as before per-unit intervals existed.
    let scale = n_draws.max(1) as f64;
    let unit_means: Vec<f64> = (0..n_units)
        .map(|unit| {
            unit_draws[unit * n_draws..(unit + 1) * n_draws].iter().fold(0.0, |s, v| s + v) / scale
        })
        .collect();
    let unit_summary = unit_effect_posterior_summary(unit_draws, n_units, n_draws)?;
    let mut ite = ite;
    ite.unit_effects = std::sync::Arc::from(unit_means);
    ite.unit_effect_intervals = Some(crate::gcm::UnitEffectIntervals {
        lower: std::sync::Arc::clone(&unit_summary.q025),
        upper: std::sync::Arc::clone(&unit_summary.q975),
        level: crate::result::REPORTED_SE_INTERVAL_LEVEL,
        method: antecedent_core::IntervalMethod::UnitPosteriorQuantile.as_str(),
    });
    ite.mean_ite = posterior.summaries.mean[eq];
    let mut estimate = EffectEstimate::new(
        posterior.summaries.mean[eq],
        posterior.summaries.sd[eq],
        posterior.assumptions.clone(),
        OverlapPolicy::ExplicitOverride,
    );
    estimate.uncertainty.analytic.se = posterior.summaries.sd[eq];
    Ok((estimate, Some(posterior), ite))
}

/// Frequentist whole-procedure row bootstrap of the mean ITE: each replicate refits every
/// mechanism under multinomial row counts (conditional on the selected families) and
/// abducts–acts–predicts on the original units. The point stays the original fit's mean
/// ITE; the replicates give its bootstrap standard error. Per-unit intervals are not
/// formed. Failed replicates are retained as failed mass.
pub(super) fn bootstrap_counterfactual_estimate(
    refit: &CounterfactualRefit<'_>,
    ite: &crate::gcm::IteResult,
    replicates: u32,
    ctx: &ExecutionContext,
) -> Result<(EffectEstimate, Diagnostic), CausalError> {
    let CounterfactualRefit { data, fitted, contrast, assumptions, status: _ } = *refit;
    let UnitContrast { treatment, outcome, active, control } = contrast;
    let n = data.row_count();
    let _admitted = super::super::gcm_bootstrap::admit_replicates(ctx, replicates, 1, n)?;
    let registry = crate::gcm::MechanismRegistry::with_heterogeneity_families();
    let cancelled = || ctx.cancellation.is_cancelled();
    let mut values = Vec::with_capacity(usize::try_from(replicates).unwrap_or(0));
    let mut failed = 0_u32;
    for replicate in 0..replicates {
        if cancelled() {
            return Err(CausalError::Cancelled {
                stage: super::super::super::stage::STAGE_UNCERTAINTY,
            });
        }
        let mut rng = super::super::gcm_bootstrap::replicate_stream(ctx, 0x0CF0_B007, replicate);
        let counts = super::super::gcm_bootstrap::bootstrap_counts(&mut rng, n);
        let store = match registry.refit_frequency_weighted_polled(
            &fitted.model,
            data,
            &fitted.assignments,
            &counts,
            &cancelled,
        ) {
            Ok(store) => store,
            Err(_) if cancelled() => {
                return Err(CausalError::Cancelled {
                    stage: super::super::super::stage::STAGE_UNCERTAINTY,
                });
            }
            Err(_) => {
                failed += 1;
                continue;
            }
        };
        // One noise stream per replicate: a categorical node's rank-preserving noise is
        // drawn once per replicate, keyed by the replicate.
        let mut noise_rng = ctx.rng.stream_for(
            StreamDomain::Counterfactual,
            0x0CF1_B007_u64.wrapping_add(u64::from(replicate)),
        );
        match crate::gcm::counterfactual_ite_with_noise(
            fitted.model.clone().with_mechanisms(store),
            data,
            (treatment, outcome, active, control),
            ctx,
            &mut noise_rng,
        ) {
            Ok(draw) if draw.unit_effects.len() == ite.unit_effects.len() => {
                values.push(draw.mean_ite);
            }
            Err(_) if cancelled() => {
                return Err(CausalError::Cancelled {
                    stage: super::super::super::stage::STAGE_UNCERTAINTY,
                });
            }
            _ => failed += 1,
        }
    }
    let accounting = super::super::gcm_bootstrap::accounting(&values, failed, replicates);
    let mut estimate = EffectEstimate::new(
        ite.mean_ite,
        f64::NAN,
        assumptions.clone(),
        OverlapPolicy::ExplicitOverride,
    );
    estimate.uncertainty.bootstrap = accounting;
    let note = super::super::gcm_bootstrap::bootstrap_diagnostic(
        "mean ITE over the observed units (per-unit intervals are not formed)",
        &accounting,
    );
    Ok((estimate, note))
}

/// Posterior summary of per-unit ITE draws, through the same summary that
/// publishes the mean-ITE interval, so a unit interval and the mean interval
/// share one quantile rule.
pub(super) fn unit_effect_posterior_summary(
    unit_draws: Vec<f64>,
    n_units: usize,
    n_draws: usize,
) -> Result<antecedent_prob::PosteriorSummary, CausalError> {
    let quantities: Vec<antecedent_prob::PosteriorQuantityKind> = (0..n_units)
        .map(|unit| antecedent_prob::PosteriorQuantityKind::Effect {
            name: std::sync::Arc::from(format!("ite[{unit}]")),
        })
        .collect();
    let schema = antecedent_prob::PosteriorSchema { quantities: std::sync::Arc::from(quantities) };
    let draws = antecedent_prob::PosteriorDraws::from_column_major(
        schema,
        n_draws,
        std::sync::Arc::<[f64]>::from(unit_draws),
    )?;
    Ok(draws.summarize())
}

fn counterfactual_posterior(
    values: Vec<f64>,
    assumptions: antecedent_core::AssumptionSet,
    identification: antecedent_core::IdentificationStatus,
) -> Result<CausalPosterior, CausalError> {
    counterfactual_posterior_columns(vec![("ite".into(), values)], assumptions, identification)
}

/// [`counterfactual_posterior`] with one `Effect` column per contrast, each a named series of the
/// same number of draws. The first column is the result's primary effect.
pub(super) fn counterfactual_posterior_columns(
    columns: Vec<(String, Vec<f64>)>,
    mut assumptions: antecedent_core::AssumptionSet,
    identification: antecedent_core::IdentificationStatus,
) -> Result<CausalPosterior, CausalError> {
    assumptions.push(antecedent_core::AssumptionRecord {
        assumption: antecedent_core::Assumption::ParametricRestriction(antecedent_core::ParametricAssumption {
            id: Arc::from("counterfactual.weighted_mechanisms"),
            description: Arc::from("Dirichlet row-weight posterior of standard mechanism fits, conditional on selected mechanism families and empirical support; abduction is repeated on the original units for every draw, the factual and counterfactual worlds sharing that draw's disturbances (a categorical node's disturbance is drawn once per posterior draw). This is not a parametric coefficient-prior posterior. The interval is for mean_ite over the observed units and reflects mechanism-refit uncertainty, plus the rank-preserving noise posterior of categorical nodes where present; it is not a unit-level predictive interval."),
        }),
        source: antecedent_core::AssumptionSource::AlgorithmDefault { algorithm: Arc::from("gcm.fit.bayesian") },
        scope: antecedent_core::AssumptionScope::Estimation,
        status: antecedent_core::AssumptionStatus::Declared,
    });
    let quantities: Vec<antecedent_prob::PosteriorQuantityKind> = columns
        .iter()
        .map(|(name, _)| antecedent_prob::PosteriorQuantityKind::Effect {
            name: std::sync::Arc::from(name.as_str()),
        })
        .collect();
    let schema = antecedent_prob::PosteriorSchema { quantities: std::sync::Arc::from(quantities) };
    let n = columns.first().map_or(0, |(_, values)| values.len());
    let values: Vec<f64> = columns.into_iter().flat_map(|(_, values)| values).collect();
    let draws = antecedent_prob::PosteriorDraws::from_column_major(
        schema,
        n,
        std::sync::Arc::<[f64]>::from(values),
    )?;
    let summaries = draws.summarize();
    Ok(CausalPosterior {
        subsampled_out_mass: 0.0,
        draws,
        summaries,
        identification,
        prior_sensitivity: None,
        conflict_summary: None,
        diagnostics: antecedent_prob::InferenceDiagnostics::analytic(
            super::super::COUNTERFACTUAL_DIRICHLET_BACKEND,
        ),
        assumptions,
        unidentified_mass: 0.0,
        unevaluable_mass: 0.0,
        early_stopped: false,
        treatment_contrast: None,
        preparation: None,
    })
}
