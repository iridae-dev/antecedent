//! Bayesian simplex synthetic control.
//!
//! The estimand is the one of the frequentist row: the average post-intervention gap between the
//! treated unit and its donor-weighted counterfactual. The convex-donor premise, donors
//! unaffected by the intervention and a persisting pre-period relation stay declared; a pre-fit
//! and a donor-range count describe the premise but do not establish it for the post period.
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::super::bayesian_quasi_common::{
    DesignFinish, DesignPosterior, SamplerRecord, effect_quantity, finish_design_posterior,
    unread_option,
};
use super::{
    CausalError, Diagnostic, DiagnosticKind, DiagnosticSeverity, ExecutionContext, StudyResult,
    TabularData,
};
use antecedent_core::SyntheticPanelMethod;
use antecedent_estimate::DrawSpec;
use antecedent_estimate::bayesian_synthetic_control::{
    BayesianSyntheticControlEstimate, BayesianSyntheticControlSpec, fit_bayesian_synthetic_control,
};
use std::time::Instant;

/// Plan id, identifier and estimator a compiled Bayesian synthetic-control plan records.
pub(in crate::analysis::execute) fn bayesian_synthetic_panel_plan_ids()
-> (&'static str, &'static str, &'static str) {
    (
        "bayesian.synthetic_control",
        "quasi.convex_donor_counterfactual",
        crate::strategy_table::EstimatorId::BayesianSyntheticControl.as_str(),
    )
}

/// Refuse a query option the Bayesian posterior does not read, naming the next attempt.
pub(super) fn refuse_unread_options(
    query: &antecedent_core::SyntheticControlQuery,
) -> Result<(), CausalError> {
    let route = "the Bayesian synthetic control";
    if query.method == SyntheticPanelMethod::DifferenceInDifferences {
        return Err(unread_option(
            route,
            "the synthetic difference-in-differences panel method",
            "the frequentist synthetic difference-in-differences route, or the Bayesian \
             simplex synthetic control without the difference-in-differences method",
        ));
    }
    if query.uniform_unit_randomization || query.sharp_null_effect.is_some() {
        return Err(unread_option(
            route,
            "a unit-randomization or sharp-null declaration, which is a design-based test with no \
             posterior",
            "the frequentist exact unit-randomization test for that null, or the Bayesian \
             posterior of the average effect without the declaration",
        ));
    }
    if query.augmentation_ridge.is_some() {
        return Err(unread_option(
            route,
            "a donor ridge outcome-model augmentation",
            "the frequentist augmented synthetic control, or the Bayesian simplex posterior \
             without augmentation",
        ));
    }
    Ok(())
}

/// Diagnostics of the simplex posterior: pre-fit and sampler summary, the declared convex-donor
/// premise, and a warning when the treated outcome leaves the donors' range.
fn synthetic_control_diagnostics(fit: &BayesianSyntheticControlEstimate) -> Vec<Diagnostic> {
    let mut diagnostics = vec![
        Diagnostic::new(
            "estimate.quasi.bayesian_synthetic_control.fit",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            format!(
                "pre-fit RMSE {:.6} at the posterior-mean weights; {} of {} pre-periods have a \
                 treated outcome outside the donors' range; sampler acceptance {:.3}, split \
                 R-hat {:.4}, bulk ESS {:.1}",
                fit.pre_fit_rmse,
                fit.pre_periods_outside_donor_range,
                fit.n_pre_periods,
                fit.acceptance_rate,
                fit.max_split_rhat,
                fit.min_bulk_ess
            ),
        ),
        Diagnostic::new(
            "identification.quasi.bayesian.convex_donor_premise_declared",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Info,
            "the convex-donor counterfactual, donors unaffected by the intervention and a \
             persisting pre-period relation stay declared premises; the prior supplies none \
             and a good pre-fit does not establish them for the post period",
        ),
    ];
    if fit.pre_periods_outside_donor_range > 0 {
        diagnostics.push(Diagnostic::new(
            "estimate.quasi.bayesian_synthetic_control.outside_donor_range",
            DiagnosticKind::Scientific,
            DiagnosticSeverity::Warning,
            format!(
                "the treated outcome lies outside the donors' range in {} pre-periods, where \
                 no convex combination can match it; the convex-donor premise is doubtful",
                fit.pre_periods_outside_donor_range
            ),
        ));
    }
    diagnostics
}

impl super::CheckedSyntheticControlOperation {
    /// Execute the retained design as a Bayesian posterior of the average post-intervention effect.
    pub(super) fn execute_bayesian(
        &self,
        data: &TabularData,
        y: &[f64],
        units: &[String],
        ctx: &ExecutionContext,
    ) -> Result<StudyResult, CausalError> {
        let Some(config) = self.extra.as_ref() else {
            return Err(CausalError::Compile {
                message: "Bayesian synthetic control requires its retained Bayesian configuration"
                    .into(),
            });
        };
        refuse_unread_options(&self.query)?;
        let started = Instant::now();
        let spec = BayesianSyntheticControlSpec::new(
            &self.query.treated_unit,
            self.query.intervention_period,
        );
        let fit = fit_bayesian_synthetic_control(
            y,
            units,
            &self.query.periods,
            &spec,
            DrawSpec::new(config.n_draws, ctx.rng.master_seed()),
        )?;
        let mut notes = vec![
            format!(
                "interval=equal_tailed_95 donors={} pre_periods={} post_periods={}",
                fit.donor_weight_means.len(),
                fit.n_pre_periods,
                fit.n_post_periods
            ),
            format!(
                "declared prior dirichlet_concentration={} variance_shape={} variance_scale={} \
                 ar1_rho={} burn_in={}",
                spec.dirichlet_concentration,
                spec.variance_shape,
                spec.variance_scale,
                spec.ar1_rho,
                spec.burn_in
            ),
        ];
        notes.extend(fit.donor_weight_means.iter().zip(&fit.donor_weight_sds).map(
            |((unit, mean), sd)| format!("donor_weight unit={unit} mean={mean:.6} sd={sd:.6}"),
        ));
        let diagnostics = synthetic_control_diagnostics(&fit);
        let design = DesignPosterior {
            columns: vec![(effect_quantity("att"), fit.effect_draws)],
            point: fit.mean,
            point_sd: fit.sd,
            n_obs: (fit.n_pre_periods + fit.n_post_periods) as u64,
            interval_withheld: false,
            backend: "bayesian_synthetic_control.pairwise_metropolis",
            model: format!(
                "{}; the Dirichlet weight prior and the inverse-gamma variance prior are fixed \
                 and prior_scale is not read; pre-fit errors are declared independent \
                 (AR(1) coefficient 0), so serially correlated pre-fit errors make the \
                 posterior overconfident",
                fit.model
            ),
            assumptions: vec![(
                "bayesian_simplex_dirichlet_gaussian_model",
                "pre-period treated outcomes are a simplex-weighted donor combination plus \
                 independent Gaussian errors under a Dirichlet weight prior and an \
                 inverse-gamma variance prior; the post-period error is continued from the \
                 same process"
                    .to_string(),
            )],
            diagnostics,
            notes,
            sampler: Some(SamplerRecord {
                chains: 4,
                max_split_rhat: fit.max_split_rhat,
                min_bulk_ess: fit.min_bulk_ess,
                acceptance_rate: fit.acceptance_rate,
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
