//! Placebo treatment refuter.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use causal_core::ExecutionContext;
use causal_data::TableView;
use causal_estimate::{EstimationWorkspace, LinearAdjustmentAte};

use crate::common::{
    RefutationProblem, RefutationReport, fill_gaussian, fit_once, with_replaced_float,
};
use crate::error::ValidationError;

/// Replace treatment with independent noise; expect ATE near zero.
#[derive(Clone, Debug)]
pub struct PlaceboTreatment {
    /// Replicate count (each draw a fresh placebo treatment).
    pub replicates: u32,
    /// Pass if mean `|refuted_ate|` is below this threshold.
    pub abs_ate_threshold: f64,
    /// Estimator used for refits (bootstrap disabled to avoid nested pools).
    pub estimator: LinearAdjustmentAte,
}

impl Default for PlaceboTreatment {
    fn default() -> Self {
        Self::new()
    }
}

impl PlaceboTreatment {
    /// Default: 20 replicates, threshold 0.25.
    #[must_use]
    pub fn new() -> Self {
        let mut estimator = LinearAdjustmentAte::new();
        estimator.bootstrap_replicates = 0;
        Self { replicates: 20, abs_ate_threshold: 0.25, estimator }
    }

    /// Run the placebo refuter.
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
                message: "placebo requires replicates > 0",
            });
        }
        let n = problem.data.row_count();
        let mut placebo = vec![0.0; n];
        let mut sum_abs = 0.0;
        let mut sum_ate = 0.0;
        for r in 0..self.replicates {
            fill_gaussian(&mut placebo, ctx, 0xA7E0_0001_u64.wrapping_add(u64::from(r)));
            let data = with_replaced_float(
                problem.data,
                problem.treatment(),
                Arc::<[f64]>::from(placebo.clone()),
            )?;
            let est =
                fit_once(&self.estimator, &data, problem.estimand, problem.query, workspace, ctx)?;
            sum_abs += est.ate.abs();
            sum_ate += est.ate;
        }
        let mean_abs = sum_abs / f64::from(self.replicates);
        let mean_ate = sum_ate / f64::from(self.replicates);
        let passed = mean_abs < self.abs_ate_threshold;
        Ok(RefutationReport {
            refuter: Arc::from("placebo.treatment"),
            original_ate: problem.original.ate,
            refuted_ate: mean_ate,
            comparison: mean_abs,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "mean |placebo ATE|={mean_abs} exceeded threshold {}",
                    self.abs_ate_threshold
                )))
            },
            replicates: self.replicates,
        })
    }
}
