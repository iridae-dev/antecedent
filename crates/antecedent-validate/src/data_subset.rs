//! Data-subset refuter.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte};

use crate::common::{
    RefutationProblem, RefutationReport, check_replicate_count, finish_p_report,
    full_sample_refit_ate, linear_estimator_no_bootstrap, map_problem_replicates,
    refit_effect_shared, with_contiguous_row_window, with_row_subset,
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
    /// Pass if the full-sample refit is a typical draw from the subset-replicate
    /// distribution (`p >= alpha`). The p-value is the two-sided normal tail of that
    /// one value under `N(mean, sd)` of the replicates, not a test of the replicate
    /// mean. A non-finite replicate is an error, not a verdict. A pass is not an
    /// informative falsification under OLS.
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
        let ates = map_problem_replicates(
            ctx,
            problem,
            self.replicates,
            EstimationWorkspace::default,
            |problem, refit, workspace, ctx, r| {
                let stream = crate::streams::replicate_stream(crate::streams::DATA_SUBSET, r);
                let data = if is_temporal_series {
                    with_contiguous_row_window(problem.data, self.subset_fraction, ctx, stream)?
                } else {
                    with_row_subset(problem.data, self.subset_fraction, ctx, stream)?
                };
                let est = refit_effect_shared(
                    problem,
                    refit,
                    &data,
                    problem.estimand,
                    &[],
                    &self.estimator,
                    workspace,
                    ctx,
                )?;
                Ok(est.ate)
            },
        )?;
        // OLS subset stability is not a falsifier of the causal claim.
        finish_p_report(
            "data.subset",
            problem.original.ate,
            &ates,
            centre,
            self.alpha,
            |mean_ate, p_value| {
                format!(
                    "subset null value {centre} is atypical under the replicate distribution \
                     (mean {mean_ate}, p={p_value} < alpha={}) across {}% subsets",
                    self.alpha,
                    self.subset_fraction * 100.0
                )
            },
        )
    }
}
