//! Bootstrap refuter: percentile CI around the original ATE.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]

use std::sync::Arc;

use causal_core::ExecutionContext;
use causal_data::TableView;
use causal_estimate::{EstimationWorkspace, LinearAdjustmentAte};
use causal_kernels::unbiased_index;

use crate::common::{
    RefutationProblem, RefutationReport, complete_case_rows, fit_once,
    linear_estimator_no_bootstrap, with_resampled_rows,
};
use crate::error::ValidationError;

/// IID row bootstrap of the whole `(T, Y, Z…)` design; "passes" if the original point estimate
/// falls inside the percentile confidence interval of the resampled ATEs.
///
/// Each replicate refits with `estimator.bootstrap_replicates = 0` (per [`crate::common::fit_once`])
/// so this never creates a nested bootstrap pool inside the resample loop.
#[derive(Clone, Debug)]
pub struct BootstrapRefute {
    /// Bootstrap replicates.
    pub replicates: u32,
    /// Confidence level for the percentile interval (e.g. 0.95).
    pub ci_level: f64,
    /// Estimator used for refits (bootstrap disabled).
    pub estimator: LinearAdjustmentAte,
}

impl Default for BootstrapRefute {
    fn default() -> Self {
        Self::new()
    }
}

impl BootstrapRefute {
    /// Defaults: 200 replicates, 95% CI.
    #[must_use]
    pub fn new() -> Self {
        Self { replicates: 200, ci_level: 0.95, estimator: linear_estimator_no_bootstrap() }
    }

    /// Run the bootstrap refuter.
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
                message: "bootstrap refutation requires replicates >= 2",
            });
        }
        if !(self.ci_level > 0.0 && self.ci_level < 1.0) {
            return Err(ValidationError::NotApplicable {
                message: "bootstrap refutation requires ci_level in (0, 1)",
            });
        }
        let n = problem.data.row_count();
        let mut resample_ids = vec![problem.treatment(), problem.outcome()];
        resample_ids.extend_from_slice(&problem.estimand.adjustment_set);
        // Resample only complete-case rows so slots that are invalid in the source (whose
        // stored values are sentinels) never enter a replicate as real observations.
        let (keep, valid) = complete_case_rows(problem.data, &resample_ids)?;
        if valid.len() < 2 {
            return Err(ValidationError::NotApplicable {
                message: "bootstrap refutation requires at least 2 complete-case rows",
            });
        }
        let mut rng = ctx.rng.stream(0xA7E0_0009_0000_u64);
        let mut row_idx = vec![0usize; n];
        let mut ates = Vec::with_capacity(self.replicates as usize);
        for _ in 0..self.replicates {
            for slot in &mut row_idx {
                *slot = valid[unbiased_index(&mut rng, valid.len())];
            }
            let data = with_resampled_rows(problem.data, &resample_ids, &row_idx, &keep)?;
            let est =
                fit_once(&self.estimator, &data, problem.estimand, problem.query, workspace, ctx)?;
            ates.push(est.ate);
        }
        ates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let m = ates.len();
        let lo_frac = (1.0 - self.ci_level) / 2.0;
        let hi_frac = 1.0 - lo_frac;
        let lo_idx = ((lo_frac * (m - 1) as f64).round() as usize).min(m - 1);
        let hi_idx = ((hi_frac * (m - 1) as f64).round() as usize).min(m - 1);
        let lo = ates[lo_idx];
        let hi = ates[hi_idx];
        let mean_ate = ates.iter().sum::<f64>() / m as f64;
        let width = hi - lo;
        let passed = problem.original.ate >= lo && problem.original.ate <= hi;
        Ok(RefutationReport {
            refuter: Arc::from("bootstrap.refute"),
            original_ate: problem.original.ate,
            refuted_ate: mean_ate,
            comparison: width,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "original ATE {} outside {}% bootstrap interval [{lo}, {hi}]",
                    problem.original.ate,
                    self.ci_level * 100.0
                )))
            },
            replicates: self.replicates,
        })
    }
}
