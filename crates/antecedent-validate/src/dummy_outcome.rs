//! Dummy outcome refuter.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte};

use crate::common::{
    NoiseReplaceSpec, NoiseReplaceTarget, RefutationProblem, RefutationReport,
    linear_estimator_no_bootstrap, noise_replace_refute,
};
use crate::error::ValidationError;

/// Replace the outcome with independent noise; expect ATE near zero.
#[derive(Clone, Debug)]
pub struct DummyOutcome {
    /// Replicate count (each draw a fresh dummy outcome).
    pub replicates: u32,
    /// Pass if zero is a typical draw from the replicate distribution (`p >= alpha`).
    /// The p-value is the two-sided normal tail of that one null value, not a test
    /// of the replicate mean.
    pub alpha: f64,
    /// Estimator used for refits (bootstrap disabled to avoid nested pools).
    pub estimator: LinearAdjustmentAte,
}

impl Default for DummyOutcome {
    fn default() -> Self {
        Self::new()
    }
}

impl DummyOutcome {
    /// Default: 20 replicates, significance level 0.05.
    #[must_use]
    pub fn new() -> Self {
        Self { replicates: 20, alpha: 0.05, estimator: linear_estimator_no_bootstrap() }
    }

    /// Run the dummy-outcome refuter.
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
        noise_replace_refute(
            problem,
            workspace,
            ctx,
            &self.estimator,
            &NoiseReplaceSpec {
                replicates: self.replicates,
                alpha: self.alpha,
                target: NoiseReplaceTarget::Outcome,
                stream_base: crate::streams::DUMMY_OUTCOME,
                refuter_id: "dummy.outcome",
                failure_label: "dummy-outcome",
            },
        )
    }
}
