//! Dummy outcome refuter.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use antecedent_core::ExecutionContext;
use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte};

use crate::common::{
    NoiseReplaceTarget, RefutationProblem, RefutationReport, linear_estimator_no_bootstrap,
    noise_replace_refute,
};
use crate::error::ValidationError;

/// Replace the outcome with independent noise; expect ATE near zero.
#[derive(Clone, Debug)]
pub struct DummyOutcome {
    /// Replicate count (each draw a fresh dummy outcome).
    pub replicates: u32,
    /// Pass if the dummy-outcome ATE distribution is consistent with zero at this
    /// significance level (two-sided normal test on the replicates, `p >= alpha`).
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
            self.replicates,
            self.alpha,
            NoiseReplaceTarget::Outcome,
            0xA7E0_0008_0000,
            "dummy.outcome",
            "dummy-outcome",
        )
    }
}
