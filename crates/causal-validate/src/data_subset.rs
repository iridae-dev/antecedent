//! Data-subset refuter.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use causal_core::ExecutionContext;
use causal_estimate::{EstimationWorkspace, LinearAdjustmentAte};

use crate::common::{
    RefutationProblem, RefutationReport, linear_estimator_no_bootstrap, refit_effect,
    replicate_p_value, with_row_subset,
};
use crate::error::ValidationError;

/// Randomly subset rows and re-estimate; expect the ATE to move little.
#[derive(Clone, Debug)]
pub struct DataSubsetRefuter {
    /// Replicate count (fresh subset draw per replicate).
    pub replicates: u32,
    /// Fraction of rows kept per replicate.
    pub subset_fraction: f64,
    /// Pass if the subset ATE distribution is consistent with the original estimate at
    /// this significance level (two-sided normal test on the replicates, `p >= alpha`).
    pub alpha: f64,
    /// Estimator used for refits (bootstrap disabled).
    pub estimator: LinearAdjustmentAte,
}

impl Default for DataSubsetRefuter {
    fn default() -> Self {
        Self::new()
    }
}

impl DataSubsetRefuter {
    /// Defaults: 20 replicates, 80% subset fraction, significance level 0.05.
    #[must_use]
    pub fn new() -> Self {
        Self {
            replicates: 20,
            subset_fraction: 0.8,
            alpha: 0.05,
            estimator: linear_estimator_no_bootstrap(),
        }
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
        if self.replicates < 2 {
            return Err(ValidationError::NotApplicable {
                message: "data subset requires replicates >= 2",
            });
        }
        if !(self.subset_fraction > 0.0 && self.subset_fraction < 1.0) {
            return Err(ValidationError::NotApplicable {
                message: "data subset requires subset_fraction in (0, 1)",
            });
        }
        let mut ates = Vec::with_capacity(self.replicates as usize);
        for r in 0..self.replicates {
            let data = with_row_subset(
                problem.data,
                self.subset_fraction,
                ctx,
                0xA7E0_0007_0000_u64.wrapping_add(u64::from(r)),
            )?;
            let est = refit_effect(problem, &data, problem.estimand, &[], workspace, ctx)?;
            ates.push(est.ate);
        }
        let mean_ate = ates.iter().sum::<f64>() / f64::from(self.replicates);
        let p_value = replicate_p_value(&ates, problem.original.ate);
        let passed = p_value >= self.alpha;
        Ok(RefutationReport {
            refuter: Arc::from("data.subset"),
            original_ate: problem.original.ate,
            refuted_ate: mean_ate,
            comparison: p_value,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "subset ATE distribution (mean {mean_ate}) is inconsistent with the \
                     original estimate (p={p_value} < alpha={}) across {}% subsets",
                    self.alpha,
                    self.subset_fraction * 100.0
                )))
            },
            replicates: self.replicates,
        })
    }
}
