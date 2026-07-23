//! Simulated unobserved common cause refuter.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_data::TableView;
use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte};

use crate::common::{
    RefutationProblem, RefutationReport, complete_case_rows, fill_gaussian, float64_full,
    linear_estimator_no_bootstrap, masked_sample_sd, refit_effect, with_replaced_float,
};
use crate::error::ValidationError;

/// Perturb treatment and outcome with a shared simulated confounder `U`, refit **without**
/// adding `U` to the adjustment set (it is "unobserved" by construction), and compare.
///
/// Unlike [`crate::rcc::RandomCommonCause`] — which adds an independent covariate *to* the
/// adjustment set to check it doesn't change the estimate — this refuter simulates a
/// confounder the estimator never sees, checking how sensitive the ATE is to a confounder of
/// the configured strength on treatment and outcome.
#[derive(Clone, Debug)]
pub struct UnobservedCommonCause {
    /// Replicate count (fresh `U` draw per replicate).
    pub replicates: u32,
    /// Simulated confounder's linear effect on treatment, in treatment-sd units per unit
    /// of the standard-normal confounder.
    pub effect_on_treatment: f64,
    /// Simulated confounder's linear effect on outcome, in outcome-sd units per unit of
    /// the standard-normal confounder.
    pub effect_on_outcome: f64,
    /// Pass if the mean `|refuted_ate - original_ate|`, standardized by `sd(Y)/sd(T)`
    /// (the natural scale of an ATE), is below this threshold.
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
        if self.replicates == 0 {
            return Err(ValidationError::NotApplicable {
                message: "unobserved common cause requires replicates > 0",
            });
        }
        if !matches!(
            problem.estimand.method_kind().ok(),
            Some(
                antecedent_expr::EstimandMethod::BackdoorAdjustment
                    | antecedent_expr::EstimandMethod::TemporalBackdoorUnfolded
            )
        ) {
            return Err(ValidationError::NotApplicable {
                message: "unobserved common cause requires backdoor.adjustment or temporal.backdoor.unfolded",
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
        let sd_t = masked_sample_sd(problem.data, problem.treatment(), &mask)?.max(1e-12);
        let sd_y = masked_sample_sd(problem.data, problem.outcome(), &mask)?.max(1e-12);
        let (kt, ky) = (self.effect_on_treatment * sd_t, self.effect_on_outcome * sd_y);
        let mut u = vec![0.0; n];
        let mut sum_delta = 0.0;
        let mut sum_ate = 0.0;
        for r in 0..self.replicates {
            fill_gaussian(&mut u, ctx, 0xA7E0_0006_0000_u64.wrapping_add(u64::from(r)));
            let t: Vec<f64> = t0.iter().zip(&u).map(|(&t, &u)| t + kt * u).collect();
            let y: Vec<f64> = y0.iter().zip(&u).map(|(&y, &u)| y + ky * u).collect();
            let data = with_replaced_float(problem.data, problem.treatment(), Arc::from(t))?;
            let data = with_replaced_float(&data, problem.outcome(), Arc::from(y))?;
            let est = refit_effect(problem, &data, problem.estimand, &[], workspace, ctx)?;
            sum_delta += (est.ate - problem.original.ate).abs();
            sum_ate += est.ate;
        }
        let mean_delta = sum_delta / f64::from(self.replicates);
        let mean_ate = sum_ate / f64::from(self.replicates);
        let std_delta = mean_delta / (sd_y / sd_t);
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
