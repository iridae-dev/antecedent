//! Bayesian quantile treatment effect of a randomized (unconfounded) binary treatment.
//!
//! The target is the quantile coordinate of the frequentist cell: the `tau`-quantile of each
//! potential outcome law and their difference, read by the same piecewise-linear inversion of
//! the distribution function on a threshold grid. The likelihood is stated: within each arm the
//! outcomes fall into the bins cut by the declared grid, independently, with one multinomial
//! probability vector per arm and a symmetric Dirichlet prior, so the posterior is the
//! Dirichlet of the prior plus the bin counts. The grid is declared on the study's Bayesian
//! configuration, never selected from the data. The arms are independent under randomization,
//! so the difference is taken draw by draw. A draw whose quantile the grid cannot resolve is
//! counted; more than one percent unresolved refuses the posterior. The treatment must be
//! unconfounded (an empty adjustment set): the Dirichlet law carries no covariates. The result
//! is an unlicensed, uncalibrated posterior.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::bayesian_specialist_path::BayesianSpecialistInputs;
use super::{
    Arc, AverageEffectQuery, BayesianConfig, CausalError, CausalPosterior, CausalQuery, Dag,
    Diagnostic, DiagnosticKind, DiagnosticSeverity, EffectEstimate, EstimatorId, ExecutionContext,
    IdentifiedEstimand, IdentifiedExecuteExtras, IdentifiedExecuteFinish, IdentifierId,
    InferenceDiagnostics, InferenceMode, Instant, OverlapPolicy, PhysicalExecutionPlan, Strategy,
    Study, StudyResult, TabularData, elapsed_ns, identify_static_query, require_identified,
    select_estimand,
};
use crate::DesignPosteriorModel;
use antecedent_estimate::EstimationError;
use antecedent_estimate::quantile_posterior::randomized_quantile_posterior;
use antecedent_prob::{PosteriorDraws, PosteriorQuantityKind, PosteriorSchema};

fn refuse(code: &'static str, detail: &str, message: &str) -> CausalError {
    CausalError::Estimate(EstimationError::refused(code, format!("{detail}: {message}")))
}

/// Model-scope checks of the checked and one-shot randomized quantile routes.
pub(super) fn check_bayesian_quantile_scope(
    query: &AverageEffectQuery,
    estimand: &IdentifiedEstimand,
    config: &BayesianConfig,
) -> Result<(), CausalError> {
    let binary = |value: f64| {
        antecedent_core::Intervention::set(query.treatment, antecedent_core::Value::f64(value))
    };
    if query.outcome_functional.quantile_level().is_none()
        || !query.effect_modifiers.is_empty()
        || query.target_population != antecedent_core::TargetPopulation::AllObserved
        || query.control != binary(0.0)
        || query.active != binary(1.0)
    {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "quantile_posterior.scope",
            "the randomized quantile posterior answers the AllObserved quantile treatment effect \
             of a binary 0/1 treatment without effect modifiers. Next attempt: a quantile \
             outcome functional on an unmodified binary AverageEffect, or the frequentist AIPW \
             quantile",
        ));
    }
    if !estimand.adjustment_set.is_empty() {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "quantile_posterior.adjustment_not_supported",
            "the Dirichlet-multinomial arm law carries no covariates, so it is the posterior of \
             the potential-outcome quantiles only when the treatment is unconfounded (an empty \
             adjustment set). Next attempt: a graph in which the treatment has no confounding \
             parents, or the frequentist AIPW quantile that adjusts",
        ));
    }
    if config.backend != antecedent_estimate::BayesianBackendKind::ConjugateGaussian
        || config.has_transfer()
    {
        return Err(refuse(
            antecedent_core::reason_code!("option_not_applicable"),
            "quantile_posterior.backend_not_applicable",
            "the quantile posterior is its own conjugate Dirichlet model, so a backend choice, a \
             transferred prior or a custom prior set does not apply. Next attempt: Bayesian \
             inference with the conjugate backend and n_draws only",
        ));
    }
    if !matches!(config.design_model, Some(DesignPosteriorModel::QuantileGrid { .. })) {
        return Err(refuse(
            antecedent_core::reason_code!("option_not_applicable"),
            "quantile_posterior.model_not_declared",
            "the quantile posterior needs its threshold grid and Dirichlet concentration \
             declared. Next attempt: BayesianConfig::design_model with \
             DesignPosteriorModel::QuantileGrid; the grid is part of the model and is never \
             selected from the data",
        ));
    }
    Ok(())
}

/// A posterior over `(quantile difference, control quantile, treated quantile)` from aligned
/// resolved draws, the difference first as the primary effect.
fn quantile_posterior_draws(
    posterior: &antecedent_estimate::quantile_posterior::QuantilePosterior,
    identification: antecedent_core::IdentificationStatus,
    assumptions: antecedent_core::AssumptionSet,
) -> Result<CausalPosterior, CausalError> {
    let difference: Vec<f64> =
        posterior.treated_draws.iter().zip(&posterior.control_draws).map(|(a, c)| a - c).collect();
    let n = difference.len();
    let mut values = Vec::with_capacity(3 * n);
    values.extend_from_slice(&difference);
    values.extend_from_slice(&posterior.control_draws);
    values.extend_from_slice(&posterior.treated_draws);
    let schema = PosteriorSchema {
        quantities: Arc::from([
            PosteriorQuantityKind::Effect { name: Arc::from("ate") },
            PosteriorQuantityKind::Scalar { name: Arc::from("control_quantile") },
            PosteriorQuantityKind::Scalar { name: Arc::from("treated_quantile") },
        ]),
    };
    let draws = PosteriorDraws::from_column_major(schema, n, Arc::<[f64]>::from(values))
        .map_err(CausalError::from)?;
    let summaries = draws.summarize();
    let mut diagnostics = InferenceDiagnostics::analytic("bayesian_quantile.dirichlet_grid");
    diagnostics.notes.push(Arc::from(format!(
        "interval=equal_tailed_95 tau={} draws_requested={} draws_resolved={}",
        posterior.tau, posterior.draws_requested, posterior.draws_resolved
    )));
    Ok(CausalPosterior {
        draws,
        summaries,
        identification,
        prior_sensitivity: None,
        conflict_summary: None,
        diagnostics,
        assumptions,
        unidentified_mass: 0.0,
        subsampled_out_mass: 0.0,
        unevaluable_mass: 0.0,
        early_stopped: false,
        treatment_contrast: Some(1.0),
    })
}

impl Study {
    /// One-shot randomized quantile posterior: identify the back-door claim, then run.
    pub(super) fn execute_bayesian_quantile(
        &self,
        data: &TabularData,
        graph: &Dag,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let identification = identify_static_query(
            IdentifierId::BackdoorAdjustment,
            graph,
            &CausalQuery::AverageEffect(query.clone()),
        )?;
        require_identified(&identification)?;
        let estimand = select_estimand(&identification, EstimatorId::BayesianRandomizedQuantile)?;
        let InferenceMode::Bayesian(config) = &self.inference else {
            return Err(CausalError::Compile {
                message: "bayesian.randomized_quantile requires Bayesian inference".into(),
            });
        };
        self.execute_bayesian_quantile_identified(
            data,
            query,
            physical,
            BayesianSpecialistInputs {
                identification: &identification,
                estimand: &estimand,
                config,
                identify_cached: false,
            },
            ctx,
        )
    }

    /// Posterior of the randomized quantile treatment effect from already-identified products.
    #[allow(
        clippy::too_many_lines,
        reason = "one posterior: read the identified products, draw the quantile effect, and assemble the claim"
    )]
    pub(super) fn execute_bayesian_quantile_identified(
        &self,
        data: &TabularData,
        query: &AverageEffectQuery,
        physical: &PhysicalExecutionPlan,
        inputs: BayesianSpecialistInputs<'_>,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = Instant::now();
        let BayesianSpecialistInputs { identification, estimand, config, identify_cached } = inputs;
        check_bayesian_quantile_scope(query, estimand, config)?;
        let Some(DesignPosteriorModel::QuantileGrid { thresholds, bin_concentration }) =
            &config.design_model
        else {
            return Err(CausalError::Compile {
                message: "the randomized quantile posterior lost its declared grid".into(),
            });
        };
        let tau = query.outcome_functional.quantile_level().ok_or(CausalError::Compile {
            message: "the randomized quantile posterior lost its quantile level".into(),
        })?;
        let mask = data
            .complete_case_mask(&[query.treatment, query.outcome])
            .map_err(CausalError::from)?;
        let raw = data.float64_masked(query.treatment, &mask).map_err(CausalError::from)?;
        let outcome = data.float64_masked(query.outcome, &mask).map_err(CausalError::from)?;
        if raw.iter().any(|value| {
            !(value.to_bits() == 0.0_f64.to_bits() || value.to_bits() == 1.0_f64.to_bits())
        }) {
            return Err(CausalError::Unsupported {
                message: "bayesian.randomized_quantile requires treatment values coded exactly 0/1",
            });
        }
        let treated: Vec<bool> = raw.iter().map(|value| *value > 0.5).collect();
        let posterior = randomized_quantile_posterior(
            &outcome,
            &treated,
            thresholds,
            *bin_concentration,
            tau,
            config.n_draws,
            ctx.rng.master_seed(),
        )
        .map_err(|error| match error {
            EstimationError::Unsupported { message } if message.contains("threshold grid") => {
                refuse(
                    antecedent_core::reason_code!("route_not_supported"),
                    "quantile_posterior.unresolved_draws",
                    &format!(
                        "{message}. Next attempt: a threshold grid that brackets the quantile \
                         of both arms, or a level inside the grid's range"
                    ),
                )
            }
            other => CausalError::from(other),
        })?;
        let row = antecedent_io::bayesian_posterior_row_artifact::quantile_row(&posterior);
        let causal = quantile_posterior_draws(
            &posterior,
            identification.status,
            identification.required_assumptions.clone(),
        )?;
        let estimate = EffectEstimate::new(
            causal.summaries.mean[0],
            causal.summaries.sd[0],
            identification.required_assumptions.clone(),
            OverlapPolicy::ExplicitOverride,
        )
        .with_n_obs(outcome.len() as u64);
        let diagnostics = vec![
            Diagnostic::new(
                "bayesian_quantile.model_scope",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "Dirichlet-multinomial arm laws on a declared grid of {} thresholds with \
                     symmetric concentration {}; piecewise-linear inversion at tau={tau}; \
                     treatment unconfounded (empty adjustment set); unlicensed, calibration \
                     unmeasured",
                    thresholds.len(),
                    bin_concentration
                ),
            ),
            Diagnostic::new(
                "bayesian_quantile.unresolved_draws",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "{} of {} posterior draws resolved the quantile in both arms; more than one \
                     percent unresolved refuses the posterior",
                    posterior.draws_resolved, posterior.draws_requested
                ),
            ),
        ];
        let mut result = self.finish_identified_execute(
            IdentifiedExecuteFinish::new(
                physical,
                identification.clone(),
                estimand.clone(),
                estimate,
                Strategy::new(
                    IdentifierId::BackdoorAdjustment,
                    EstimatorId::BayesianRandomizedQuantile,
                ),
                query.treatment,
                query.outcome,
            )
            .identify_cached(identify_cached)
            .extra_diagnostics(diagnostics)
            .wall_time_ns(elapsed_ns(started))
            .extras(IdentifiedExecuteExtras {
                posterior: Some(causal),
                n_draws: Some(u32::try_from(posterior.draws_resolved).unwrap_or(u32::MAX)),
                ..Default::default()
            }),
        );
        result.payload = crate::result::ResultPayload::PosteriorRow(row);
        Ok(result)
    }
}
