//! Observational excursion, static regime, and regime-survival fits.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::{
    CausalError, CheckedTemporalDesignOperation, ExecutionContext, FitOutput, Summary, TabularData,
    TemporalRegimeMethod, indicator, read_column,
};
use antecedent_estimate::temporal_design::observational::{
    OBSERVATIONAL_COORDINATES, ObservationalRows, fit_observational_excursion,
};
use antecedent_estimate::temporal_design::regime::{
    REGIME_COORDINATES, RegimeMethod, RegimePanel, RegimeRows, StaticRegimes, fit_static_regimes,
};
use antecedent_estimate::temporal_design::survival::{
    SURVIVAL_COORDINATES, Trajectories, TrajectoryRows, fit_regime_survival,
    fit_regime_survival_posterior,
};

impl CheckedTemporalDesignOperation {
    /// The observational excursion: the assignment law is fitted on the measured history, and
    /// the fit is refitted inside every bootstrap replicate or posterior draw.
    pub(super) fn fit_observational(
        &self,
        data: &TabularData,
        outcome: &[f64],
        unit: &[usize],
        pairs: &antecedent_estimate::temporal_design::LaggedPairs,
        columns: (antecedent_core::VariableId, &[antecedent_core::VariableId]),
        ctx: &ExecutionContext,
    ) -> Result<FitOutput, CausalError> {
        let (treatment_id, history_ids) = columns;
        let treated = indicator(&read_column(data, treatment_id)?, "treatment")?;
        let history: Vec<Vec<f64>> =
            history_ids.iter().map(|id| read_column(data, *id)).collect::<Result<_, _>>()?;
        let rows = ObservationalRows { pairs, outcome, treated: &treated, history: &history, unit };
        let floor = self.query.min_probability;
        let (prior_scale, draws) = self
            .extra
            .bayesian
            .as_ref()
            .map_or((1.0, None), |config| (config.prior_scale, Some(config.n_draws)));
        let fit = fit_observational_excursion(
            rows,
            floor,
            prior_scale,
            self.extra.replicates as usize,
            draws,
            ctx,
        )
        .map_err(CausalError::from)?;
        let coordinates = OBSERVATIONAL_COORDINATES.iter().map(|name| (*name).to_owned()).collect();
        let summary = match (&fit.posterior, fit.standard_error, fit.interval) {
            (Some(posterior), _, _) => Summary::Posterior(posterior.clone()),
            (None, Some(se), Some(interval)) => Summary::Bootstrap {
                estimate: fit.estimate.to_vec(),
                standard_error: se.to_vec(),
                interval: interval.to_vec(),
            },
            _ => {
                return Err(CausalError::Compile {
                    message: "the observational excursion reported no uncertainty".into(),
                });
            }
        };
        let posterior = fit.posterior.is_some();
        Ok(FitOutput {
            coordinates,
            family: vec![0],
            summary,
            pairs: fit.pairs,
            clusters: fit.clusters,
            unpaired: pairs.unpaired,
            minimum_probability: fit.minimum_probability,
            replicates: (!posterior).then_some((fit.replicates.0 as u64, fit.replicates.1 as u64)),
            functional: format!("observational_excursion:history_columns={}", history.len()),
            uncertainty: if posterior {
                "weighted_likelihood_bootstrap_refitting_assignment_model"
            } else {
                "whole_procedure_unit_bootstrap_refitting_assignment_model"
            },
        })
    }

    /// The value of whole static regimes on a continuous end-of-follow-up outcome: every
    /// treatment and outcome model is refitted inside each bootstrap replicate or posterior draw.
    pub(super) fn fit_regime(
        &self,
        data: &TabularData,
        outcome: &[f64],
        unit: &[usize],
        columns: (
            antecedent_core::VariableId,
            &[antecedent_core::VariableId],
            &[bool],
            &[bool],
            TemporalRegimeMethod,
        ),
        ctx: &ExecutionContext,
    ) -> Result<FitOutput, CausalError> {
        let (treatment_id, confounder_ids, active, reference, method) = columns;
        let treated = indicator(&read_column(data, treatment_id)?, "treatment")?;
        let confounders: Vec<Vec<f64>> =
            confounder_ids.iter().map(|id| read_column(data, *id)).collect::<Result<_, _>>()?;
        let panel = RegimePanel::new(RegimeRows {
            unit,
            period: &self.query.periods,
            outcome,
            treated: &treated,
            confounders: &confounders,
        })
        .map_err(CausalError::from)?;
        let estimator = match method {
            TemporalRegimeMethod::GFormula => RegimeMethod::GFormula,
            TemporalRegimeMethod::InverseProbability => RegimeMethod::InverseProbability,
            TemporalRegimeMethod::DoublyRobust => RegimeMethod::DoublyRobust,
        };
        let (prior_scale, draws) = self
            .extra
            .bayesian
            .as_ref()
            .map_or((1.0, None), |config| (config.prior_scale, Some(config.n_draws)));
        let fit = fit_static_regimes(
            &panel,
            StaticRegimes { active, reference },
            estimator,
            self.query.min_probability,
            prior_scale,
            self.extra.replicates as usize,
            draws,
            ctx,
        )
        .map_err(CausalError::from)?;
        let coordinates = REGIME_COORDINATES.iter().map(|name| (*name).to_owned()).collect();
        let summary = match (&fit.posterior, fit.standard_error, fit.interval) {
            (Some(posterior), _, _) => Summary::Posterior(posterior.clone()),
            (None, Some(se), Some(interval)) => Summary::Bootstrap {
                estimate: fit.estimate.to_vec(),
                standard_error: se.to_vec(),
                interval: interval.to_vec(),
            },
            _ => {
                return Err(CausalError::Compile {
                    message: "the regime contrast reported no uncertainty".into(),
                });
            }
        };
        let posterior = fit.posterior.is_some();
        Ok(FitOutput {
            coordinates,
            family: vec![0],
            summary,
            pairs: fit.units * fit.periods,
            clusters: fit.units,
            unpaired: 0,
            minimum_probability: fit.minimum_probability,
            replicates: (!posterior).then_some((fit.replicates.0 as u64, fit.replicates.1 as u64)),
            functional: format!(
                "observational_regime:method={};periods={};confounder_columns={}",
                method.name(),
                fit.periods,
                confounders.len()
            ),
            uncertainty: if posterior {
                "weighted_likelihood_bootstrap_refitting_regime_models"
            } else {
                "whole_procedure_unit_bootstrap_refitting_regime_models"
            },
        })
    }

    /// Regime survival on subject trajectories.
    pub(super) fn fit_survival(
        &self,
        data: &TabularData,
        event: &[f64],
        unit: &[usize],
        columns: (
            antecedent_core::VariableId,
            antecedent_core::VariableId,
            &[antecedent_core::VariableId],
            u32,
        ),
        ctx: &ExecutionContext,
    ) -> Result<FitOutput, CausalError> {
        let (treatment_id, censored_id, confounder_ids, horizon) = columns;
        let treatment = indicator(&read_column(data, treatment_id)?, "treatment")?;
        let censored = indicator(&read_column(data, censored_id)?, "censoring")?;
        let event = indicator(event, "event")?;
        let confounders: Vec<Vec<f64>> =
            confounder_ids.iter().map(|id| read_column(data, *id)).collect::<Result<_, _>>()?;
        let rows = TrajectoryRows {
            unit,
            period: &self.query.periods,
            treatment: &treatment,
            censored: &censored,
            event: &event,
            confounders: &confounders,
        };
        let trajectories = Trajectories::new(rows, horizon as usize).map_err(CausalError::from)?;
        let floor = self.query.min_probability;
        let fit = match &self.extra.bayesian {
            Some(config) => fit_regime_survival_posterior(
                &trajectories,
                floor,
                config.prior_scale,
                config.n_draws,
                ctx,
            ),
            None => fit_regime_survival(&trajectories, floor, self.extra.replicates as usize, ctx),
        }
        .map_err(CausalError::from)?;
        let coordinates = SURVIVAL_COORDINATES.iter().map(|name| (*name).to_owned()).collect();
        let summary = match (&fit.posterior, fit.standard_error, fit.interval) {
            (Some(posterior), _, _) => Summary::Posterior(posterior.clone()),
            (None, Some(se), Some(interval)) => Summary::Bootstrap {
                estimate: fit.estimate.to_vec(),
                standard_error: se.to_vec(),
                interval: interval.to_vec(),
            },
            _ => {
                return Err(CausalError::Compile {
                    message: "the regime survival fit reported no uncertainty".into(),
                });
            }
        };
        let posterior = fit.posterior.is_some();
        Ok(FitOutput {
            coordinates,
            family: vec![0],
            summary,
            pairs: fit.person_periods,
            clusters: fit.subjects,
            unpaired: 0,
            minimum_probability: fit.minimum_regime_probability,
            replicates: (!posterior).then_some((fit.replicates.0 as u64, fit.replicates.1 as u64)),
            functional: match fit.posterior_positivity_shortfall {
                Some(shortfall) => format!(
                    "survival:horizon={horizon}; posterior share of draws below the positivity \
                     floor {shortfall}"
                ),
                None => format!("survival:horizon={horizon}"),
            },
            uncertainty: if posterior {
                "joint_likelihood_parametric_g_formula_posterior"
            } else {
                "whole_procedure_subject_bootstrap_refitting_treatment_and_censoring_models"
            },
        })
    }
}
