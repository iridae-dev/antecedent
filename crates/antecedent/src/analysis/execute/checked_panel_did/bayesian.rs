//! Bayesian posteriors of the panel difference-in-differences designs.
//!
//! Each design keeps the estimand of its frequentist row (the same group-time or two-period
//! difference in outcome changes) and puts a posterior on it under a stated Gaussian outcome
//! model with a Normal-Inverse-Gamma prior. Parallel untreated trends, no anticipation and the
//! design's sampling assumption stay declared: the prior supplies none of them and the
//! posterior tests none of them.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::bayesian_quasi_common::{
    DesignFinish, DesignPosterior, effect_quantity, finish_design_posterior, scalar_quantity,
};
use super::{
    CausalError, Diagnostic, DiagnosticKind, DiagnosticSeverity, ExecutionContext, Instant,
    StudyResult, TableView, TabularData,
};
use antecedent_core::DidSamplingDesign;
use antecedent_estimate::DrawSpec;
use antecedent_estimate::bayesian_panel_did::{
    BayesianDidEstimate, DidPrior, fit_bayesian_augmented_panel_did,
    fit_bayesian_cross_section_did, fit_bayesian_panel_did,
};
use antecedent_estimate::bayesian_staggered_event_study::{
    BayesianEventTimeEffect, fit_bayesian_event_study,
};

/// Plan id, identifier and estimator a compiled Bayesian `DiD` plan records. The estimator is
/// the closed-set `bayesian.panel_did` for every sampling design; the design is the query's.
pub(in crate::analysis::execute) fn bayesian_panel_did_plan_ids(
    _design: DidSamplingDesign,
) -> (&'static str, &'static str, &'static str) {
    (
        "bayesian.panel_did",
        "quasi.parallel_trends",
        crate::strategy_table::EstimatorId::BayesianPanelDid.as_str(),
    )
}

/// Rows of `values` where `keep` is true, in order.
fn pick<T: Clone>(values: &[T], keep: &[bool]) -> Vec<T> {
    values.iter().zip(keep).filter(|(_, kept)| **kept).map(|(value, _)| value.clone()).collect()
}

/// One continuous column of the table.
fn float_column(
    data: &TabularData,
    variable: antecedent_core::VariableId,
) -> Result<Vec<f64>, CausalError> {
    match data.column(variable).map_err(CausalError::from)? {
        antecedent_data::ColumnView::Float64(column) => Ok(column.values.as_slice().to_vec()),
        _ => Err(CausalError::Unsupported {
            message: "augmented panel DiD requires continuous outcome and nuisance columns",
        }),
    }
}

/// The premise that stays declared beside every Bayesian `DiD` posterior.
fn premise_diagnostic() -> Diagnostic {
    Diagnostic::new(
        "identification.quasi.bayesian.parallel_trends_declared",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        "parallel untreated trends and no anticipation stay declared identifying premises; the \
         prior supplies neither and the posterior neither tests nor licenses them",
    )
}

/// Assemble the posterior of one scalar `DiD` effect.
fn scalar_posterior(
    fit: BayesianDidEstimate,
    effect: &str,
    assumption: (&'static str, &str),
    prior: DidPrior,
) -> DesignPosterior {
    let n_obs = (fit.n_treated + fit.n_comparison) as u64;
    DesignPosterior {
        columns: vec![(effect_quantity(effect), fit.effect_draws)],
        point: fit.mean,
        point_sd: fit.sd,
        n_obs,
        interval_withheld: false,
        backend: "bayesian_did.normal_inverse_gamma",
        model: fit.model.to_string(),
        assumptions: vec![(assumption.0, assumption.1.to_string())],
        diagnostics: vec![premise_diagnostic()],
        notes: vec![
            format!(
                "interval=equal_tailed_95 treated={} comparison={}",
                fit.n_treated, fit.n_comparison
            ),
            format!(
                "prior coefficient_scale={} variance_shape={} variance_scale={}",
                prior.coefficient_scale, prior.variance_shape, prior.variance_scale
            ),
        ],
        sampler: None,
    }
}

impl super::CheckedPanelDidOperation {
    /// Execute the retained design as a Bayesian posterior on complete, finite outcomes.
    pub(super) fn execute_bayesian(
        &self,
        data: &TabularData,
        y: &[f64],
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let Some(config) = self.extra.bayesian.as_ref() else {
            return Err(CausalError::Compile {
                message: "Bayesian panel DiD requires its retained Bayesian configuration".into(),
            });
        };
        if data.storage().weights().is_some() {
            return Err(CausalError::Estimate(
                antecedent_estimate::sampling_weights::unsupported_route(
                    "the Bayesian panel DiD posterior reads no sampling weights. Next attempt: \
                     the frequentist balanced two-period panel DiD, which reads them",
                ),
            ));
        }
        let started = Instant::now();
        let prior = DidPrior { coefficient_scale: config.prior_scale, ..DidPrior::diffuse() };
        let sampling = DrawSpec::new(config.n_draws, ctx.rng.master_seed());
        let mut fit = match self.query.design {
            DidSamplingDesign::BalancedPanel => self.balanced_posterior(y, prior, sampling)?,
            DidSamplingDesign::RepeatedCrossSection => {
                self.cross_section_posterior(y, prior, sampling)?
            }
            DidSamplingDesign::AugmentedPanel => {
                self.augmented_posterior(data, y, prior, sampling)?
            }
            DidSamplingDesign::StaggeredEventStudy | DidSamplingDesign::StaggeredGroupTime => {
                self.staggered_posterior(y, prior, sampling)?
            }
        };
        if self.extra.restricted {
            fit.notes.push(
                "the declared missing-data policy restricted the rows; the posterior describes the retained rows"
                    .to_string(),
            );
        }
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
            fit,
        )
    }

    /// Posterior of the difference in mean subject change scores of a balanced two-period panel.
    fn balanced_posterior(
        &self,
        y: &[f64],
        prior: DidPrior,
        sampling: DrawSpec,
    ) -> Result<DesignPosterior, CausalError> {
        let (changes, _, _) = self.balanced_subject_changes(y)?;
        let treated: Vec<bool> = changes.iter().map(|(group, _, _)| *group).collect();
        let clusters: Vec<&str> = changes.iter().map(|(_, cluster, _)| *cluster).collect();
        let change: Vec<f64> = changes.iter().map(|(_, _, delta)| *delta).collect();
        // The module differences within subject, so the change score is its only input from the
        // two periods.
        let pre = vec![0.0; change.len()];
        let fit = fit_bayesian_panel_did(&pre, &change, &treated, &clusters, prior, sampling)?;
        Ok(scalar_posterior(
            fit,
            "att",
            (
                "bayesian_gaussian_change_score_model",
                "subject change scores are independent Gaussian with a common variance and a \
                 group mean shift, under a Normal-Inverse-Gamma prior",
            ),
            prior,
        ))
    }

    /// Posterior of the interaction in the four-cell repeated cross-section model.
    fn cross_section_posterior(
        &self,
        y: &[f64],
        prior: DidPrior,
        sampling: DrawSpec,
    ) -> Result<DesignPosterior, CausalError> {
        let fit = fit_bayesian_cross_section_did(
            y,
            &self.query.treated,
            &self.query.post,
            &self.query.clusters,
            prior,
            sampling,
        )?;
        Ok(scalar_posterior(
            fit,
            "att",
            (
                "bayesian_gaussian_cell_model",
                "rows are independent Gaussian with a common variance and group, period and \
                 interaction means, under a Normal-Inverse-Gamma prior",
            ),
            prior,
        ))
    }

    /// Posterior of the covariate-adjusted ATT with the supplied columns as covariates.
    fn augmented_posterior(
        &self,
        data: &TabularData,
        y: &[f64],
        prior: DidPrior,
        sampling: DrawSpec,
    ) -> Result<DesignPosterior, CausalError> {
        let Some(nuisance) = self.query.augmented.as_ref() else {
            return Err(CausalError::Compile {
                message: "augmented panel DiD requires the augmented nuisance block".into(),
            });
        };
        let pre = float_column(data, nuisance.outcome_pre)?;
        let covariates = vec![
            float_column(data, nuisance.propensity)?,
            float_column(data, nuisance.untreated_change_prediction)?,
        ];
        let fit = fit_bayesian_augmented_panel_did(
            &pre,
            y,
            &self.query.treated,
            &covariates,
            &self.query.clusters,
            prior,
            sampling,
        )?;
        let mut posterior = scalar_posterior(
            fit,
            "att",
            (
                "bayesian_arm_specific_change_score_regressions",
                "each arm's change score is Gaussian and linear in the supplied propensity and \
                 untreated-change prediction columns, under Normal-Inverse-Gamma priors",
            ),
            prior,
        );
        posterior.notes.push(
            "the supplied propensity and untreated-change prediction columns enter as covariates of the arm regressions; neither is used as an estimated nuisance, and cross-fitting is not verified"
                .to_string(),
        );
        Ok(posterior)
    }

    /// Posteriors of the cohort-by-period contrasts against never-treated subjects; the group-time
    /// design keeps the one selected contrast.
    fn staggered_posterior(
        &self,
        y: &[f64],
        prior: DidPrior,
        sampling: DrawSpec,
    ) -> Result<DesignPosterior, CausalError> {
        let group_time = self
            .query
            .target
            .filter(|_| self.query.design == DidSamplingDesign::StaggeredGroupTime);
        let keep: Vec<bool> = self
            .query
            .cohorts
            .iter()
            .map(|cohort| group_time.is_none_or(|(target, _)| *cohort == 0 || *cohort == target))
            .collect();
        let effects = fit_bayesian_event_study(
            &pick(y, &keep),
            &pick(&self.query.subjects, &keep),
            &pick(&self.query.periods, &keep),
            &pick(&self.query.cohorts, &keep),
            &pick(&self.query.clusters, &keep),
            prior,
            sampling,
        )?;
        match group_time {
            Some((cohort, period)) => group_time_posterior(&effects, cohort, period, prior),
            None => event_study_posterior(&effects, prior),
        }
    }
}

/// The selected cohort-period contrast of a staggered group-time design.
fn group_time_posterior(
    effects: &[BayesianEventTimeEffect],
    cohort: i64,
    period: i64,
    prior: DidPrior,
) -> Result<DesignPosterior, CausalError> {
    let Some(selected) =
        effects.iter().find(|effect| effect.cohort == cohort && effect.period == period)
    else {
        return Err(CausalError::Compile {
            message: format!(
                "the selected cohort {cohort} and period {period} is not a contrast of the panel"
            ),
        });
    };
    let n_obs = (selected.treated_subjects + selected.comparison_subjects) as u64;
    Ok(DesignPosterior {
        columns: vec![(effect_quantity("group_time_att"), selected.effect_draws.clone())],
        point: selected.mean,
        point_sd: selected.sd,
        n_obs,
        interval_withheld: false,
        backend: "bayesian_did.normal_inverse_gamma",
        model: "cohort-specific change score from the period before adoption against never-treated subjects; Gaussian common-variance model; independent subjects".to_string(),
        assumptions: vec![(
            "bayesian_gaussian_change_score_model",
            "the selected contrast's change scores are independent Gaussian with a common \
             variance and a group mean shift, under a Normal-Inverse-Gamma prior"
                .to_string(),
        )],
        diagnostics: vec![premise_diagnostic()],
        notes: vec![
            format!(
                "interval=equal_tailed_95 cohort={cohort} period={period} treated={} comparison={}",
                selected.treated_subjects, selected.comparison_subjects
            ),
            format!(
                "prior coefficient_scale={} variance_shape={} variance_scale={}",
                prior.coefficient_scale, prior.variance_shape, prior.variance_scale
            ),
        ],
        sampler: None,
    })
}

/// Every event-time contrast: post-adoption effects as effect quantities, pre-adoption leads as
/// descriptive scalar quantities.
fn event_study_posterior(
    effects: &[BayesianEventTimeEffect],
    prior: DidPrior,
) -> Result<DesignPosterior, CausalError> {
    let Some(representative) = effects.iter().find(|effect| effect.event_time >= 0) else {
        return Err(CausalError::Compile {
            message: "event study has no post-adoption contrast".into(),
        });
    };
    let column = |effect: &BayesianEventTimeEffect| {
        let name = format!("cohort_{}_period_{}", effect.cohort, effect.period);
        let kind = if effect.event_time >= 0 {
            effect_quantity(format!("event_time_{name}"))
        } else {
            scalar_quantity(format!("lead_{name}"))
        };
        (kind, effect.effect_draws.clone())
    };
    let mut columns = vec![column(representative)];
    columns
        .extend(effects.iter().filter(|effect| !std::ptr::eq(*effect, representative)).map(column));
    let leads: Vec<&BayesianEventTimeEffect> =
        effects.iter().filter(|effect| effect.event_time < 0).collect();
    let covering_zero = leads.iter().filter(|lead| lead.lower <= 0.0 && 0.0 <= lead.upper).count();
    let lead_diagnostic = Diagnostic::new(
        "diagnostic.quasi.bayesian.event_study.leads_descriptive",
        DiagnosticKind::Scientific,
        DiagnosticSeverity::Info,
        format!(
            "{} pre-adoption lead posteriors are descriptive and pointwise, {covering_zero} with a \
             95% interval covering zero; contrasts share never-treated subjects so no joint or \
             simultaneous claim is made, and a lead near zero does not license parallel \
             untreated trends after adoption",
            leads.len()
        ),
    );
    let n_obs = (representative.treated_subjects + representative.comparison_subjects) as u64;
    Ok(DesignPosterior {
        columns,
        point: representative.mean,
        point_sd: representative.sd,
        n_obs,
        interval_withheld: false,
        backend: "bayesian_did.normal_inverse_gamma",
        model: "cohort-specific event-time change scores against never-treated subjects; Gaussian common-variance model per contrast; independent subjects".to_string(),
        assumptions: vec![(
            "bayesian_gaussian_change_score_model",
            "each contrast's change scores are independent Gaussian with a common variance and a \
             group mean shift, under a Normal-Inverse-Gamma prior"
                .to_string(),
        )],
        diagnostics: vec![premise_diagnostic(), lead_diagnostic],
        notes: vec![
            format!(
                "interval=equal_tailed_95 representative_cohort={} representative_period={} contrasts={}",
                representative.cohort,
                representative.period,
                effects.len()
            ),
            format!(
                "prior coefficient_scale={} variance_shape={} variance_scale={}",
                prior.coefficient_scale, prior.variance_shape, prior.variance_scale
            ),
        ],
        sampler: None,
    })
}
