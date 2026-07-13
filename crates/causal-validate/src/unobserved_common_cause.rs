//! Simulated unobserved common cause refuter.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use causal_core::ExecutionContext;
use causal_data::TableView;
use causal_estimate::{EstimationWorkspace, LinearAdjustmentAte};

use crate::common::{
    RefutationProblem, RefutationReport, fill_gaussian, fit_once, float64_full,
    linear_estimator_no_bootstrap, with_replaced_float,
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
    /// Simulated confounder's linear effect on treatment.
    pub effect_on_treatment: f64,
    /// Simulated confounder's linear effect on outcome.
    pub effect_on_outcome: f64,
    /// Pass if mean `|refuted_ate - original_ate|` is below this threshold.
    pub abs_delta_threshold: f64,
    /// Estimator used for refits (bootstrap disabled).
    pub estimator: LinearAdjustmentAte,
}

impl Default for UnobservedCommonCause {
    fn default() -> Self {
        Self::new()
    }
}

impl UnobservedCommonCause {
    /// Defaults: 20 replicates, effect 0.5 on both treatment and outcome, threshold 1.0.
    #[must_use]
    pub fn new() -> Self {
        Self {
            replicates: 20,
            effect_on_treatment: 0.5,
            effect_on_outcome: 0.5,
            abs_delta_threshold: 1.0,
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
        if &*problem.estimand.method != "backdoor.adjustment" {
            return Err(ValidationError::NotApplicable {
                message: "unobserved common cause requires backdoor.adjustment estimand",
            });
        }
        let n = problem.data.row_count();
        let t0 = float64_full(problem.data, problem.treatment())?;
        let y0 = float64_full(problem.data, problem.outcome())?;
        let mut u = vec![0.0; n];
        let mut sum_delta = 0.0;
        let mut sum_ate = 0.0;
        for r in 0..self.replicates {
            fill_gaussian(&mut u, ctx, 0xA7E0_0006_u64.wrapping_add(u64::from(r)));
            let t: Vec<f64> =
                t0.iter().zip(&u).map(|(&t, &u)| t + self.effect_on_treatment * u).collect();
            let y: Vec<f64> =
                y0.iter().zip(&u).map(|(&y, &u)| y + self.effect_on_outcome * u).collect();
            let data = with_replaced_float(problem.data, problem.treatment(), Arc::from(t))?;
            let data = with_replaced_float(&data, problem.outcome(), Arc::from(y))?;
            let est =
                fit_once(&self.estimator, &data, problem.estimand, problem.query, workspace, ctx)?;
            sum_delta += (est.ate - problem.original.ate).abs();
            sum_ate += est.ate;
        }
        let mean_delta = sum_delta / f64::from(self.replicates);
        let mean_ate = sum_ate / f64::from(self.replicates);
        let passed = mean_delta < self.abs_delta_threshold;
        Ok(RefutationReport {
            refuter: Arc::from("unobserved.common_cause"),
            original_ate: problem.original.ate,
            refuted_ate: mean_ate,
            comparison: mean_delta,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "mean |ΔATE|={mean_delta} exceeded threshold {} under simulated confounding \
                     (effect_on_treatment={}, effect_on_outcome={})",
                    self.abs_delta_threshold, self.effect_on_treatment, self.effect_on_outcome
                )))
            },
            replicates: self.replicates,
        })
    }
}
