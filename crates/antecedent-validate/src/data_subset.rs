//! Data-subset refuter.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte};

use crate::common::{
    RefutationProblem, RefutationReport, check_cancelled, check_replicate_count,
    full_sample_refit_ate, linear_estimator_no_bootstrap, refit_effect, replicate_mean,
    replicate_p_value, with_contiguous_row_window, with_row_subset,
};
use crate::error::ValidationError;

/// Randomly subset rows and re-estimate; expect the ATE to move little.
///
/// Under the OLS / linear-adjustment path this refuter is gated to, subset refits of a
/// correctly specified linear model stay near the full-sample estimate by construction,
/// so the procedure has no power as a falsifier of the causal claim. Reports set
/// `informative: false` (sampling-stability diagnostic only).
#[derive(Clone, Debug)]
pub struct DataSubsetRefuter {
    /// Replicate count (fresh subset draw per replicate).
    pub replicates: u32,
    /// Fraction of rows kept per replicate.
    pub subset_fraction: f64,
    /// Pass if the subset ATE distribution is consistent with the same estimator's
    /// full-sample refit at this significance level (two-sided normal test on the
    /// replicates, `p >= alpha`). A non-finite replicate is an error, not a verdict. A pass
    /// is not an informative falsification under OLS.
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
        check_replicate_count(self.replicates)?;
        if !(self.subset_fraction > 0.0 && self.subset_fraction < 1.0) {
            return Err(ValidationError::NotApplicable {
                message: "data subset requires subset_fraction in (0, 1)",
            });
        }
        // Row-hiding masks are unsafe for a single lag-gathered series: `ensure_unmasked`
        // (crates/antecedent-data/src/sample.rs) rejects them outright because lag gathers
        // index raw row positions, and even a physically-dropped interior hole would quietly
        // change what "lag-1" means across the seam. A contiguous window keeps every retained
        // row adjacent to its true predecessor, so lag semantics survive; see
        // `with_contiguous_row_window` for the full argument. Panel refits are left on the
        // row-mask path (unchanged from before): `PanelSliceTemplate::apply_stacked` slices
        // stacked rows at fixed per-unit offsets computed from the *original* row count, so a
        // window that shortens the stacked table would desynchronize those offsets — panel
        // temporal designs are not part of this fix's required coverage.
        let is_temporal_series = problem.temporal.is_some_and(|t| !t.is_panel());
        // Centre on the same estimator's full-sample refit, not the published number: see
        // `full_sample_refit_ate` (a Bayesian posterior mean is not the least-squares refit).
        let centre = full_sample_refit_ate(problem, &self.estimator, workspace, ctx)?;
        let mut ates = Vec::with_capacity(self.replicates as usize);
        for r in 0..self.replicates {
            check_cancelled(ctx)?;
            let stream = 0xA7E0_0007_0000_u64.wrapping_add(u64::from(r));
            let data = if is_temporal_series {
                with_contiguous_row_window(problem.data, self.subset_fraction, ctx, stream)?
            } else {
                with_row_subset(problem.data, self.subset_fraction, ctx, stream)?
            };
            let est = refit_effect(
                problem,
                &data,
                problem.estimand,
                &[],
                &self.estimator,
                workspace,
                ctx,
            )?;
            ates.push(est.ate);
        }
        let mean_ate = replicate_mean(&ates);
        let p_value = replicate_p_value(&ates, centre)?;
        let passed = p_value >= self.alpha;
        Ok(RefutationReport {
            refuter: Arc::from("data.subset"),
            original_ate: problem.original.ate,
            refuted_ate: mean_ate,
            comparison: p_value,
            // OLS subset stability is not a falsifier of the causal claim.
            informative: false,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "subset ATE distribution (mean {mean_ate}) is inconsistent with the \
                     full-sample refit {centre} (p={p_value} < alpha={}) across {}% subsets",
                    self.alpha,
                    self.subset_fraction * 100.0
                )))
            },
            replicates: self.replicates,
        })
    }
}
