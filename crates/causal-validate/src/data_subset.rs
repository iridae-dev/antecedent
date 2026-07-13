//! Data-subset refuter.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use causal_core::ExecutionContext;
use causal_estimate::{EstimationWorkspace, LinearAdjustmentAte};

use crate::common::{RefutationProblem, RefutationReport, fit_once, with_row_subset};
use crate::error::ValidationError;

/// Randomly subset rows and re-estimate; expect the ATE to move little.
#[derive(Clone, Debug)]
pub struct DataSubsetRefuter {
    /// Replicate count (fresh subset draw per replicate).
    pub replicates: u32,
    /// Fraction of rows kept per replicate.
    pub subset_fraction: f64,
    /// Pass if mean `|refuted_ate - original_ate|` is below this threshold.
    pub abs_delta_threshold: f64,
    /// Estimator used for refits (bootstrap disabled).
    pub estimator: LinearAdjustmentAte,
}

impl Default for DataSubsetRefuter {
    fn default() -> Self {
        Self::new()
    }
}

impl DataSubsetRefuter {
    /// Defaults: 20 replicates, 80% subset fraction, threshold 0.15.
    #[must_use]
    pub fn new() -> Self {
        let mut estimator = LinearAdjustmentAte::new();
        estimator.bootstrap_replicates = 0;
        Self { replicates: 20, subset_fraction: 0.8, abs_delta_threshold: 0.15, estimator }
    }

    /// Run the data-subset refuter.
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
                message: "data subset requires replicates > 0",
            });
        }
        if !(0.0..1.0).contains(&self.subset_fraction) {
            return Err(ValidationError::NotApplicable {
                message: "data subset requires subset_fraction in (0, 1)",
            });
        }
        let mut sum_delta = 0.0;
        let mut sum_ate = 0.0;
        for r in 0..self.replicates {
            let data = with_row_subset(
                problem.data,
                self.subset_fraction,
                ctx,
                0xA7E0_0007_u64.wrapping_add(u64::from(r)),
            )?;
            let est =
                fit_once(&self.estimator, &data, problem.estimand, problem.query, workspace, ctx)?;
            sum_delta += (est.ate - problem.original.ate).abs();
            sum_ate += est.ate;
        }
        let mean_delta = sum_delta / f64::from(self.replicates);
        let mean_ate = sum_ate / f64::from(self.replicates);
        let passed = mean_delta < self.abs_delta_threshold;
        Ok(RefutationReport {
            refuter: Arc::from("data.subset"),
            original_ate: problem.original.ate,
            refuted_ate: mean_ate,
            comparison: mean_delta,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "mean |ΔATE|={mean_delta} exceeded threshold {} across {}% subsets",
                    self.abs_delta_threshold,
                    self.subset_fraction * 100.0
                )))
            },
            replicates: self.replicates,
        })
    }
}
