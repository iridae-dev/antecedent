//! Simulated unobserved common cause refuter.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_data::TableView;
use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte};

use crate::common::{
    RefutationProblem, RefutationReport, check_cancelled, check_replicate_count,
    complete_case_rows, fill_gaussian, float64_full, linear_estimator_no_bootstrap, refit_effect,
    with_replaced_float,
};
use crate::error::ValidationError;
use crate::sensitivity::{residual_sd_pair_on_adjustment, temporal_residual_sd_pair};

/// Perturb treatment and outcome with a shared simulated confounder `U`, refit **without**
/// adding `U` to the adjustment set (it is "unobserved" by construction), and compare.
///
/// Unlike [`crate::rcc::RandomCommonCause`] — which adds an independent covariate *to* the
/// adjustment set to check it doesn't change the estimate — this refuter simulates a
/// confounder the estimator never sees, checking how sensitive the ATE is to a confounder of
/// the configured strength on treatment and outcome.
///
/// Strengths are in units of the *residual* SDs `SD(T | Z)` and `SD(Y | Z)` (`Z` the adjustment
/// set), the variation left for a confounder to explain, not the marginal SDs (which include
/// what `Z` already explains and would understate the confounder's relative size whenever `Z`
/// predicts `T`). With `a = effect_on_treatment`, `b = effect_on_outcome` and `ρ` the
/// partial correlation of `T` and `Y` given `Z`, the standardized slope moves from `ρ` to
/// `(ρ + a·b) / (1 + a²)`, so the expected shift is `(a·b − ρ·a²) / (1 + a²)` of a residual-SD
/// ratio. The verdict compares that shift with the effect itself: the check fails when a
/// confounder of the configured strength moves the estimate by at least as much as the
/// estimate's own magnitude (`std_delta_threshold = 1`).
#[derive(Clone, Debug)]
pub struct UnobservedCommonCause {
    /// Replicate count (fresh `U` draw per replicate).
    pub replicates: u32,
    /// Simulated confounder's linear effect on treatment, in residual treatment-SD units
    /// (`SD(T | Z)`) per unit of the standard-normal confounder.
    pub effect_on_treatment: f64,
    /// Simulated confounder's linear effect on outcome, in residual outcome-SD units
    /// (`SD(Y | Z)`) per unit of the standard-normal confounder.
    pub effect_on_outcome: f64,
    /// Pass if the mean `|refuted_ate - original_ate|`, as a multiple of the published
    /// effect's own magnitude `|original_ate|` (floored at `1e-3` of the residual-SD scale of
    /// an ATE, `SD(Y|Z)/SD(T|Z)` times the treatment contrast, so a null estimate is judged
    /// against a small positive scale), is below this threshold. The default of 1 fails when
    /// the simulated confounder can move the estimate by its whole size.
    pub std_delta_threshold: f64,
    /// Estimator used for refits (bootstrap disabled).
    pub estimator: LinearAdjustmentAte,
}

impl Default for UnobservedCommonCause {
    fn default() -> Self {
        Self::new()
    }
}

impl UnobservedCommonCause {
    /// Defaults: 20 replicates, effect 0.5 sd on both treatment and outcome, threshold 1.0.
    #[must_use]
    pub fn new() -> Self {
        Self {
            replicates: 20,
            effect_on_treatment: 0.5,
            effect_on_outcome: 0.5,
            std_delta_threshold: 1.0,
            estimator: linear_estimator_no_bootstrap(),
        }
    }

    /// Run the unobserved-common-cause refuter.
    ///
    /// # Errors
    ///
    /// Data or estimation failures.
    pub fn refute(
        &self,
        problem: &RefutationProblem<'_>,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<RefutationReport, ValidationError> {
        check_replicate_count(self.replicates)?;
        if !(self.effect_on_treatment.is_finite()
            && self.effect_on_treatment >= 0.0
            && self.effect_on_outcome.is_finite()
            && self.effect_on_outcome >= 0.0)
        {
            return Err(ValidationError::NotApplicable {
                message: "unobserved common cause requires finite, non-negative effect strengths",
            });
        }
        if !problem.estimand.is_adjustment_shaped()
            && problem.estimand.method_kind().ok()
                != Some(antecedent_expr::EstimandMethod::TemporalBackdoorUnfolded)
        {
            return Err(ValidationError::NotApplicable {
                message: "unobserved common cause requires backdoor.adjustment or \
                          temporal.backdoor.unfolded",
            });
        }
        let n = problem.data.row_count();
        let t0 = float64_full(problem.data, problem.treatment())?;
        let y0 = float64_full(problem.data, problem.outcome())?;
        let mut ids = vec![problem.treatment(), problem.outcome()];
        if problem.temporal.is_none() {
            ids.extend_from_slice(&problem.estimand.adjustment_set);
        }
        let (mask, _valid) = complete_case_rows(problem.data, &ids)?;
        let (sd_t, sd_y) = if problem.temporal.is_some() {
            temporal_residual_sd_pair(problem)?
        } else {
            residual_sd_pair_on_adjustment(problem, problem.treatment(), problem.outcome(), &mask)?
        };
        if !(sd_t.is_finite() && sd_t > 0.0 && sd_y.is_finite() && sd_y > 0.0) {
            return Err(ValidationError::NotApplicable {
                message: "unobserved common cause requires positive, finite residual variation \
                          in treatment and outcome",
            });
        }
        let (kt, ky) = (self.effect_on_treatment * sd_t, self.effect_on_outcome * sd_y);
        let mut u = vec![0.0; n];
        let mut sum_delta = 0.0;
        let mut sum_ate = 0.0;
        for r in 0..self.replicates {
            check_cancelled(ctx)?;
            fill_gaussian(&mut u, ctx, 0xA7E0_0006_0000_u64.wrapping_add(u64::from(r)));
            let t: Vec<f64> = t0.iter().zip(&u).map(|(&t, &u)| t + kt * u).collect();
            let y: Vec<f64> = y0.iter().zip(&u).map(|(&y, &u)| y + ky * u).collect();
            let data = with_replaced_float(problem.data, problem.treatment(), Arc::from(t))?;
            let data = with_replaced_float(&data, problem.outcome(), Arc::from(y))?;
            let est = refit_effect(
                problem,
                &data,
                problem.estimand,
                &[],
                &self.estimator,
                workspace,
                ctx,
            )?;
            if !est.ate.is_finite() {
                return Err(ValidationError::estimation_msg(
                    "non-finite simulated-confounder refit effect: the refit did not produce a \
                     usable effect",
                ));
            }
            sum_delta += (est.ate - problem.original.ate).abs();
            sum_ate += est.ate;
        }
        let mean_delta = sum_delta / f64::from(self.replicates);
        let mean_ate = sum_ate / f64::from(self.replicates);
        let (_, _, treatment_delta) = antecedent_estimate::prepare::treatment_contrast(
            &problem.query.active,
            &problem.query.control,
        )?;
        // The scale of an ATE in residual-SD units; the floor keeps a null estimate finite.
        let ate_scale = (sd_y / sd_t) * treatment_delta.abs();
        let std_delta = mean_delta / problem.original.ate.abs().max(1e-3 * ate_scale);
        let passed = std_delta < self.std_delta_threshold;
        Ok(RefutationReport {
            refuter: Arc::from("unobserved.common_cause"),
            original_ate: problem.original.ate,
            refuted_ate: mean_ate,
            comparison: std_delta,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "standardized mean |ΔATE|={std_delta} exceeded threshold {} under simulated \
                     confounding (effect_on_treatment={} sd, effect_on_outcome={} sd)",
                    self.std_delta_threshold, self.effect_on_treatment, self.effect_on_outcome
                )))
            },
            replicates: self.replicates,
        })
    }
}
