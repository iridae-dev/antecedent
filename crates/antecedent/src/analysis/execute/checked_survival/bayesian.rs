//! Bayesian piecewise-exponential survival for the randomized two-arm survival query.
//!
//! The target is the one the frequentist row reports: the treated-minus-control survival at
//! the restriction horizon and the difference in restricted mean survival time through it.
//! The likelihood is stated: hazards are constant on the declared intervals, subjects are
//! independent, censoring and delayed entry are non-informative given the arm, and the arms
//! are randomized. The interval partition and the gamma prior are declared on the study's
//! Bayesian configuration and never selected from the data; the prior supplies no identifying
//! premise. Competing causes, known censoring weights and a bootstrap request are refused
//! with the next attempt named. The result is an unlicensed, uncalibrated posterior.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::bayesian_quasi_common::{
    DesignFinish, DesignPosterior, effect_quantity, finish_design_posterior_as,
    require_built_in_model,
};
use super::{
    CausalError, CheckedSurvivalOperation, EstimatorId, ExecutionContext, IdentifierId, Strategy,
    StudyResult, SurvivalFunctional, SurvivalRows, TabularData,
};
use crate::{BayesianConfig, DesignPosteriorModel};
use antecedent_estimate::EstimationError;
use antecedent_estimate::survival_posterior::{
    HazardGammaPrior, randomized_piecewise_survival_posterior,
};

fn refuse(code: &'static str, detail: &str, message: &str) -> CausalError {
    CausalError::Estimate(EstimationError::refused_detail(code, detail, message))
}

fn competing_causes() -> CausalError {
    refuse(
        antecedent_core::reason_code!("route_not_supported"),
        "survival_posterior.competing_causes_unsupported",
        "a piecewise-exponential survival posterior needs events coded zero or one; competing \
         causes need cause-specific hazards with a competing-risk likelihood. Next attempt: \
         collapse the causes to one all-cause event, or fit a hazard posterior per cause and \
         report cumulative incidence",
    )
}

/// The declared hazard partition and gamma prior, checked against the query's horizon.
fn declared_model<'a>(
    query: &antecedent_core::SurvivalQuery,
    config: &'a BayesianConfig,
) -> Result<(&'a [f64], HazardGammaPrior), CausalError> {
    let Some(DesignPosteriorModel::PiecewiseHazard { cuts, prior_shape, prior_rate }) =
        &config.design_model
    else {
        return Err(refuse(
            antecedent_core::reason_code!("option_not_applicable"),
            "survival_posterior.model_not_declared",
            "the piecewise-exponential survival posterior needs its hazard partition and gamma \
             prior declared. Next attempt: BayesianConfig::design_model with \
             DesignPosteriorModel::PiecewiseHazard; the partition and the prior are part of the \
             answer and are never selected from the data",
        ));
    };
    let horizon = cuts.last().copied().unwrap_or(f64::NAN);
    if horizon.is_nan() || (horizon - query.tau).abs() > 1e-10 {
        return Err(refuse(
            antecedent_core::reason_code!("option_not_applicable"),
            "survival_posterior.horizon_mismatch",
            "the last hazard cut is the restriction horizon and must equal the query's tau. \
             Next attempt: end the declared partition at tau",
        ));
    }
    Ok((cuts, HazardGammaPrior { shape: *prior_shape, rate: *prior_rate }))
}

/// Refuse what this posterior does not model, before any row is read.
pub(super) fn require_scope(
    query: &antecedent_core::SurvivalQuery,
    config: &BayesianConfig,
    bootstrap_replicates: u32,
) -> Result<(), CausalError> {
    require_built_in_model(config, "the Bayesian piecewise-exponential survival posterior")?;
    if !matches!(query.functional, SurvivalFunctional::SurvivalAndRmst) {
        return Err(competing_causes());
    }
    if query.known_censoring.is_some() {
        return Err(refuse(
            antecedent_core::reason_code!("route_not_supported"),
            "survival_posterior.known_censoring_unsupported",
            "the posterior models non-informative censoring given the arm and has no likelihood \
             for caller-supplied censoring weights. Next attempt: drop the known censoring \
             survival and declare marginally independent censoring, or use the frequentist \
             IPCW product limit",
        ));
    }
    if bootstrap_replicates != 0 {
        return Err(refuse(
            antecedent_core::reason_code!("option_not_applicable"),
            "survival_posterior.bootstrap_not_applicable",
            "a posterior has credible intervals from its draws and no bootstrap. Next attempt: \
             omit the bootstrap count, or use frequentist inference for the bootstrap interval",
        ));
    }
    declared_model(query, config).map(drop)
}

impl CheckedSurvivalOperation {
    /// Execute the retained design as the posterior of the survival and restricted-mean
    /// contrast under the declared piecewise-exponential model.
    pub(super) fn execute_bayesian(
        &self,
        data: &TabularData,
        config: &BayesianConfig,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let started = std::time::Instant::now();
        let (cuts, prior) = declared_model(&self.query, config)?;
        let columns = self.survival_columns(data)?;
        if columns.event.iter().any(|code| *code > 1) {
            return Err(competing_causes());
        }
        let rows = SurvivalRows {
            duration: &columns.duration,
            event_code: &columns.event,
            treated: &columns.treated,
            delayed_entry: columns.entry.as_deref(),
        };
        let posterior = randomized_piecewise_survival_posterior(
            &rows,
            cuts,
            prior,
            config.n_draws,
            ctx.rng.master_seed(),
        )
        .map_err(CausalError::from)?;
        let row = antecedent_io::bayesian_posterior_row_artifact::survival_row(&posterior);
        let design = DesignPosterior {
            columns: vec![
                (
                    effect_quantity("survival_difference"),
                    posterior.survival_difference_draws.clone(),
                ),
                (effect_quantity("rmst_difference"), posterior.rmst_difference_draws.clone()),
            ],
            point: posterior.survival_difference_mean,
            point_sd: f64::NAN,
            n_obs: self.rows as u64,
            interval_withheld: false,
            backend: "bayesian_survival.piecewise_exponential_gamma",
            model: format!(
                "piecewise-constant hazards on {} declared intervals with independent \
                 Gamma({}, {}) priors; conjugate posterior, non-informative censoring and entry \
                 given the arm, randomized assignment",
                cuts.len(),
                prior.shape,
                prior.rate
            ),
            assumptions: vec![(
                "piecewise_exponential_gamma_hazard_model",
                "event hazards are constant on every declared interval and independent a priori \
                 under a gamma prior; censoring and delayed entry are non-informative about the \
                 event time given the arm"
                    .to_string(),
            )],
            diagnostics: Vec::new(),
            notes: vec![format!(
                "horizon={} intervals={} probability_treated_survives_longer={:.6}",
                self.query.tau,
                cuts.len(),
                posterior.probability_treated_survives_longer
            )],
            sampler: None,
        };
        let mut result = finish_design_posterior_as(
            DesignFinish {
                context: &self.result_context,
                data,
                physical: &self.physical,
                identification: &self.identification,
                estimand: &self.estimand,
                outcome: self.query.duration,
                config,
                started,
            },
            design,
            Strategy::new(IdentifierId::RandomizedDesign, EstimatorId::BayesianPiecewiseSurvival),
        )?;
        result.estimate = crate::PrimaryEstimate::NotAnEffect;
        result.payload = crate::result::ResultPayload::PosteriorRow(row);
        Ok(result)
    }
}
