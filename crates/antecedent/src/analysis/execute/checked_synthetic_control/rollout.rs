//! Synthetic control with intervals: conformal inference for one treated unit, placebo variance
//! for a staggered multi-unit rollout, and the Bayesian posterior of the multi-unit average.
//!
//! The frequentist intervals are unlicensed and their calibration is unmeasured. The Bayesian
//! multi-unit posterior runs the simplex donor-weight posterior of each adopting unit against
//! the never-treated donors and averages the per-unit draws with post-period weights; treating
//! the per-unit posteriors as independent given the shared donors is a declared modelling
//! premise, not a property of the design.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::bayesian_quasi_common::{
    DesignFinish, DesignPosterior, SamplerRecord, effect_quantity, finish_design_posterior,
    scalar_quantity,
};
use super::super::rollout_finish::{
    RolloutPayloadParts, RolloutReport, finish_rollout, named, rollout_payload,
};
use super::{
    CausalError, Diagnostic, DiagnosticKind, DiagnosticSeverity, ExecutionContext, StudyResult,
    TabularData,
};
use antecedent_core::{SyntheticInterval, SyntheticPanelMethod};
use antecedent_estimate::DrawSpec;
use antecedent_estimate::bayesian_synthetic_control::{
    BayesianSyntheticControlSpec, fit_bayesian_synthetic_control,
};
use antecedent_estimate::synthetic_rollout::{
    SyntheticRolloutPanel, conformal_synthetic_control, multi_unit_synthetic,
};
use std::time::Instant;

/// Refuse a declared sampling design the synthetic interval cannot honor, naming the next
/// attempt.
pub(super) fn refuse_sampling_design(
    query: &antecedent_core::SyntheticControlQuery,
    sampling: Option<antecedent_core::DeclaredSamplingDesign>,
) -> Result<(), CausalError> {
    use antecedent_core::DeclaredSamplingDesign as Design;
    let Some(rollout) = &query.rollout else {
        // A declared serial or clustered design is honoured by the study builder (conformal
        // interval, placebo-variance interval). It reaches this point unhonoured only when the
        // route cannot carry that interval: the Bayesian donor-weight posterior, or a query
        // that also asks for a unit-randomization test, a ridge augmentation or (for the serial
        // design) the synthetic difference in differences.
        return match sampling {
            Some(Design::ClusteredPanel | Design::DependentTimeSeries) => {
                Err(crate::compile_reason!(
                    "option_not_applicable",
                    "the declared clustered_panel or dependent_time_series design is honoured by \
                     the placebo-variance or conformal interval of the frequentist synthetic \
                     control, and this query cannot carry it (the Bayesian donor-weight posterior \
                     reads no cluster label or lag, and the interval cannot be combined with a \
                     unit-randomization test, a ridge augmentation or, for a serial design, the \
                     synthetic difference in differences), so the declaration would be silently \
                     ignored. Next attempt: drop the conflicting option, or declare \
                     independent_subject_trajectories (or no design)"
                ))
            }
            _ => Ok(()),
        };
    };
    match (rollout.interval, sampling) {
        (
            SyntheticInterval::Conformal,
            Some(Design::ClusteredPanel | Design::IndependentSubjectTrajectories),
        ) => Err(crate::compile_reason!(
            "option_not_applicable",
            "a conformal synthetic interval permutes the treated unit's residual series over \
                 time, which is a dependent-series procedure; the declared design resamples \
                 independent units. Next attempt: declare dependent_time_series (or no design) \
                 for the conformal interval, or the placebo-variance interval"
        )),
        (SyntheticInterval::PlaceboVariance, Some(Design::DependentTimeSeries)) => {
            Err(crate::compile_reason!(
                "option_not_applicable",
                "placebo variance reassigns adoptions across whole units, and the declared \
                 dependent_time_series design is one series. Next attempt: declare \
                 clustered_panel or independent_subject_trajectories, or a single treated unit \
                 with the conformal interval"
            ))
        }
        _ => Ok(()),
    }
}

fn estimation(error: antecedent_estimate::EstimationError) -> CausalError {
    CausalError::Estimate(error)
}

impl super::CheckedSyntheticControlOperation {
    /// Execute a synthetic panel with conformal or placebo-variance intervals.
    #[allow(
        clippy::too_many_lines,
        reason = "one rollout execution: the fit, its interval method and the rollout payload share the same fit"
    )]
    pub(super) fn execute_rollout(
        &self,
        data: &TabularData,
        y: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let Some(rollout) = self.query.rollout.as_ref() else {
            return Err(CausalError::Compile {
                message: "a synthetic rollout requires its adoptions".into(),
            });
        };
        let panel = SyntheticRolloutPanel {
            outcome: y,
            units: &self.query.units,
            periods: &self.query.periods,
        };
        let report = match rollout.interval {
            SyntheticInterval::Conformal => {
                let fit = conformal_synthetic_control(
                    &panel,
                    &self.query.treated_unit,
                    self.query.intervention_period,
                    ctx,
                )
                .map_err(estimation)?;
                let mut effects = vec![fit.overall.clone()];
                effects.extend(fit.periods.iter().cloned());
                RolloutReport {
                    design: "conformal_synthetic_control",
                    method: "simplex_donor_weights.conformal_moving_block".to_string(),
                    effects,
                    uncertainty: "conformal_test_inversion_refit_under_each_null",
                    replicates: [0, 0, 0],
                    critical_value: None,
                    diagnostics: named(&[
                        ("pre_treatment_rmse", fit.pre_treatment_rmse),
                        ("pre_periods", fit.pre_periods as f64),
                        ("post_periods", fit.post_periods as f64),
                        ("smallest_period_p_value", fit.smallest_period_p_value),
                        ("unbounded_intervals", fit.unbounded_intervals as f64),
                        ("donors", fit.donor_weights.len() as f64),
                    ]),
                    notes: vec![Diagnostic::new(
                        "estimate.quasi.synthetic_control.conformal",
                        DiagnosticKind::Scientific,
                        DiagnosticSeverity::Info,
                        "each per-period interval inverts a permutation test that refits the \
                         donor weights with the hypothesized effect removed; the average-effect \
                         interval assumes a constant effect over the post periods; exact when \
                         the residual series is exchangeable over time, approximate under weak \
                         serial dependence; an interval reaching the searched range is reported \
                         as absent",
                    )],
                }
            }
            SyntheticInterval::PlaceboVariance => {
                let fit = multi_unit_synthetic(
                    &panel,
                    &rollout.adoptions,
                    self.query.method,
                    rollout.replicates,
                    ctx,
                )
                .map_err(estimation)?;
                let mut effects = vec![fit.overall.clone()];
                effects.extend(fit.units.iter().cloned());
                let placebo =
                    fit.placebo.unwrap_or(antecedent_estimate::synthetic_rollout::PlaceboRecord {
                        requested: rollout.replicates,
                        replicates_ok: 0,
                        replicates_failed: rollout.replicates,
                        critical_value: None,
                    });
                let did = self.query.method == SyntheticPanelMethod::DifferenceInDifferences;
                RolloutReport {
                    design: if did {
                        "multi_unit_synthetic_did"
                    } else {
                        "multi_unit_synthetic_control"
                    },
                    method: if did {
                        "unit_and_time_simplex_weights.never_treated_donors.post_weighted_average"
                    } else {
                        "simplex_donor_weights.never_treated_donors.post_weighted_average"
                    }
                    .to_string(),
                    effects,
                    uncertainty: "placebo_variance_refit_normal_interval_and_per_unit_sup_t_band",
                    replicates: [
                        placebo.requested,
                        placebo.replicates_ok,
                        placebo.replicates_failed,
                    ],
                    critical_value: placebo.critical_value,
                    diagnostics: named(&[
                        ("adopting_units", rollout.adoptions.len() as f64),
                        ("never_treated_donors", fit.donors as f64),
                    ]),
                    notes: vec![Diagnostic::new(
                        "estimate.quasi.synthetic_control.placebo_variance",
                        DiagnosticKind::Scientific,
                        DiagnosticSeverity::Info,
                        "each adopting unit is fit against the never-treated donors at its own \
                         adoption period; the placebo variance reassigns the adoptions to \
                         never-treated units and refits; it assumes the treated units' untreated \
                         errors are exchangeable with the donors'",
                    )],
                }
            }
        };
        Ok(finish_rollout(self, data, self.query.outcome, report))
    }

    /// The Bayesian posterior of the post-weighted average effect of a multi-unit rollout.
    #[allow(
        clippy::too_many_lines,
        reason = "one Bayesian rollout: the posterior fit, its draw summaries and the rollout payload read the same draws"
    )]
    pub(super) fn execute_bayesian_rollout(
        &self,
        data: &TabularData,
        y: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let (Some(config), Some(rollout)) = (self.extra.as_ref(), self.query.rollout.as_ref())
        else {
            return Err(CausalError::Compile {
                message: "the Bayesian synthetic rollout requires its configuration and adoptions"
                    .into(),
            });
        };
        if rollout.interval == SyntheticInterval::Conformal {
            return self.execute_bayesian_predictive(data, y, config, ctx);
        }
        let started = Instant::now();
        let adopting: std::collections::BTreeSet<&str> =
            rollout.adoptions.iter().map(|(unit, _)| unit.as_ref()).collect();
        let mut columns: Vec<(antecedent_prob::PosteriorQuantityKind, Vec<f64>)> = Vec::new();
        let mut overall: Option<Vec<f64>> = None;
        let mut total_posts = 0usize;
        let (mut max_rhat, mut min_ess, mut acceptance) = (0.0_f64, f64::INFINITY, 0.0_f64);
        let mut notes = Vec::new();
        for (index, (unit, adoption)) in rollout.adoptions.iter().enumerate() {
            let keep: Vec<bool> = self
                .query
                .units
                .iter()
                .map(|label| label == unit || !adopting.contains(label.as_ref()))
                .collect();
            let pick_f = |values: &[f64]| -> Vec<f64> {
                values.iter().zip(&keep).filter(|(_, kept)| **kept).map(|(v, _)| *v).collect()
            };
            let units: Vec<String> = self
                .query
                .units
                .iter()
                .zip(&keep)
                .filter(|(_, kept)| **kept)
                .map(|(label, _)| label.to_string())
                .collect();
            let periods: Vec<i64> = self
                .query
                .periods
                .iter()
                .zip(&keep)
                .filter(|(_, kept)| **kept)
                .map(|(period, _)| *period)
                .collect();
            let spec = BayesianSyntheticControlSpec::new(unit, *adoption);
            let seed = antecedent_core::mix64(ctx.rng.master_seed() ^ (index as u64 + 1));
            let fit = fit_bayesian_synthetic_control(
                &pick_f(y),
                &units,
                &periods,
                &spec,
                DrawSpec::new(config.n_draws, seed),
            )?;
            max_rhat = max_rhat.max(fit.max_split_rhat);
            min_ess = min_ess.min(fit.min_bulk_ess);
            acceptance += fit.acceptance_rate / rollout.adoptions.len() as f64;
            let posts = fit.n_post_periods;
            total_posts += posts;
            notes.push(format!(
                "unit={unit} adoption={adoption} posterior_mean={:.6} sd={:.6} post_periods={posts}",
                fit.mean, fit.sd
            ));
            let weighted: Vec<f64> =
                fit.effect_draws.iter().map(|draw| draw * posts as f64).collect();
            overall = Some(match overall {
                None => weighted,
                Some(sum) => sum.iter().zip(&weighted).map(|(a, b)| a + b).collect(),
            });
            columns.push((scalar_quantity(format!("att_unit_{unit}")), fit.effect_draws));
        }
        let overall: Vec<f64> = overall
            .unwrap_or_default()
            .into_iter()
            .map(|sum| sum / total_posts.max(1) as f64)
            .collect();
        let draws = overall.len().max(1) as f64;
        let mean = overall.iter().sum::<f64>() / draws;
        let sd = (overall.iter().map(|draw| (draw - mean).powi(2)).sum::<f64>() / draws).sqrt();
        columns.insert(0, (effect_quantity("att"), overall));
        let design = DesignPosterior {
            columns,
            point: mean,
            point_sd: sd,
            n_obs: self.query.units.len() as u64,
            interval_withheld: false,
            backend: "bayesian_synthetic_control.multi_unit_post_weighted",
            model: "per-unit simplex Dirichlet donor-weight posteriors against the never-treated \
                    donors, averaged draw by draw with post-period weights; the per-unit \
                    posteriors are treated as independent given the shared donors"
                .to_string(),
            assumptions: vec![(
                "independent_per_unit_synthetic_posteriors",
                "each adopting unit's donor-weight posterior is fit separately against the \
                 never-treated donors, and the joint posterior of the average treats them as \
                 independent given those donors"
                    .to_string(),
            )],
            diagnostics: vec![Diagnostic::new(
                "estimate.quasi.bayesian_synthetic_control.multi_unit",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "{} adopting units; worst split R-hat {max_rhat:.4}; smallest bulk ESS \
                     {min_ess:.1}; unlicensed and uncalibrated",
                    rollout.adoptions.len()
                ),
            )],
            notes,
            sampler: Some(SamplerRecord {
                chains: 4,
                max_split_rhat: max_rhat,
                min_bulk_ess: min_ess,
                acceptance_rate: acceptance,
            }),
        };
        finish_design_posterior(
            DesignFinish {
                context: &self.result_context,
                data,
                physical: &self.physical,
                identification: &self.identification,
                estimand: &self.estimand,
                outcome: self.query.outcome,
                config,
                started,
            },
            design,
        )
    }
}

impl super::CheckedSyntheticControlOperation {
    /// The Bayesian counterpart of the conformal interval: the posterior predictive
    /// distribution of the effect at each post period and of their average, from the simplex
    /// donor-weight posterior with its error process continued into the post period.
    ///
    /// It is a different object from the conformal set: a model-based predictive interval for
    /// the counterfactual path under the declared Gaussian error model and Dirichlet prior,
    /// not a finite-sample permutation guarantee.
    #[allow(
        clippy::too_many_lines,
        reason = "one posterior predictive execution: the fit, the per-period effect draws and the payload read the same draws"
    )]
    fn execute_bayesian_predictive(
        &self,
        data: &TabularData,
        y: &[f64],
        config: &super::BayesianConfig,
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        use antecedent_estimate::rollout_resampling::DesignEffectKind;
        let started = Instant::now();
        let units: Vec<String> = self.query.units.iter().map(ToString::to_string).collect();
        let spec = BayesianSyntheticControlSpec::new(
            &self.query.treated_unit,
            self.query.intervention_period,
        );
        let fit = fit_bayesian_synthetic_control(
            y,
            &units,
            &self.query.periods,
            &spec,
            DrawSpec::new(config.n_draws, ctx.rng.master_seed()),
        )?;
        let summarize =
            |draws: &[f64]| -> Option<antecedent_estimate::rollout_resampling::PosteriorQuantity> {
                let wrapped: Vec<Option<Vec<f64>>> =
                    draws.iter().map(|draw| Some(vec![*draw])).collect();
                let count = u32::try_from(draws.len()).unwrap_or(u32::MAX);
                antecedent_estimate::rollout_resampling::summarize_posterior(&wrapped, 1, count)
                    .quantities[0]
            };
        let effect = |kind, draws: &[f64]| {
            let mut effect = antecedent_estimate::rollout_resampling::DesignEffect::point(
                kind,
                draws.iter().sum::<f64>() / draws.len().max(1) as f64,
            );
            if let Some(quantity) = summarize(draws) {
                effect.estimate = quantity.mean;
                effect.interval = Some(quantity.interval);
            }
            effect.unit = Some(self.query.treated_unit.to_string());
            effect.treated = 1;
            effect.comparison = fit.donor_weight_means.len();
            effect
        };
        let mut effects = vec![effect(DesignEffectKind::Overall, &fit.effect_draws)];
        let mut columns: Vec<(antecedent_prob::PosteriorQuantityKind, Vec<f64>)> =
            vec![(effect_quantity("att"), fit.effect_draws.clone())];
        for (period, draws) in fit.post_periods.iter().zip(&fit.period_effect_draws) {
            let mut period_effect = effect(DesignEffectKind::Period, draws);
            period_effect.period = Some(*period);
            effects.push(period_effect);
            columns.push((scalar_quantity(format!("effect_period_{period}")), draws.clone()));
        }
        let point = effects[0].estimate;
        let design = DesignPosterior {
            columns,
            point,
            point_sd: fit.sd,
            n_obs: (fit.n_pre_periods + fit.n_post_periods) as u64,
            interval_withheld: false,
            backend: "bayesian_synthetic_control.posterior_predictive_path",
            model: format!(
                "{}; per-period effects are posterior predictive draws of the counterfactual \
                 path (donor-weighted outcome plus the error process continued from the last \
                 pre-period residual); this is the Bayesian counterpart of the conformal set, \
                 not a permutation guarantee",
                fit.model
            ),
            assumptions: vec![(
                "bayesian_simplex_dirichlet_gaussian_model",
                "pre-period treated outcomes are a simplex-weighted donor combination plus \
                 Gaussian errors under a Dirichlet weight prior and an inverse-gamma variance \
                 prior; the post-period errors continue the same process"
                    .to_string(),
            )],
            diagnostics: vec![Diagnostic::new(
                "estimate.quasi.bayesian_synthetic_control.predictive_path",
                DiagnosticKind::Scientific,
                DiagnosticSeverity::Info,
                format!(
                    "posterior predictive intervals for {} post periods; split R-hat {:.4}; \
                     bulk ESS {:.1}; unlicensed and uncalibrated",
                    fit.n_post_periods, fit.max_split_rhat, fit.min_bulk_ess
                ),
            )],
            notes: vec![format!("pre_fit_rmse={:.6}", fit.pre_fit_rmse)],
            sampler: Some(SamplerRecord {
                chains: 4,
                max_split_rhat: fit.max_split_rhat,
                min_bulk_ess: fit.min_bulk_ess,
                acceptance_rate: fit.acceptance_rate,
            }),
        };
        let draws = u32::try_from(fit.effect_draws.len()).unwrap_or(u32::MAX);
        let parts = RolloutPayloadParts {
            design: "posterior_predictive_synthetic_control",
            method: "simplex_dirichlet_posterior.predictive_counterfactual_path".to_string(),
            effects,
            uncertainty: "bayesian_posterior_predictive_intervals",
            replicates: [draws, draws, 0],
            critical_value: None,
            diagnostics: named(&[
                ("pre_fit_rmse", fit.pre_fit_rmse),
                ("post_periods", fit.n_post_periods as f64),
                ("max_split_rhat", fit.max_split_rhat),
            ]),
        };
        let mut result = finish_design_posterior(
            DesignFinish {
                context: &self.result_context,
                data,
                physical: &self.physical,
                identification: &self.identification,
                estimand: &self.estimand,
                outcome: self.query.outcome,
                config,
                started,
            },
            design,
        )?;
        result.payload = rollout_payload(parts);
        result.treatment = None;
        result.support_status = None;
        Ok(result)
    }
}
