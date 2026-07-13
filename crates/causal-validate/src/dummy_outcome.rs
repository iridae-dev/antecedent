//! Dummy outcome refuter.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use causal_core::ExecutionContext;
use causal_estimate::{EstimationWorkspace, LinearAdjustmentAte};

use crate::common::{
    NoiseReplaceConfig, NoiseReplaceTarget, RefutationProblem, RefutationReport,
    linear_estimator_no_bootstrap, noise_replace_refute,
};
use crate::error::ValidationError;

/// Replace the outcome with independent noise; expect ATE near zero.
#[derive(Clone, Debug)]
pub struct DummyOutcome {
    /// Replicate count (each draw a fresh dummy outcome).
    pub replicates: u32,
    /// Pass if mean `|refuted_ate|` is below this threshold.
    pub abs_ate_threshold: f64,
    /// Estimator used for refits (bootstrap disabled to avoid nested pools).
    pub estimator: LinearAdjustmentAte,
}

impl Default for DummyOutcome {
    fn default() -> Self {
        Self::new()
    }
}

impl DummyOutcome {
    /// Default: 20 replicates, threshold 0.25.
    #[must_use]
    pub fn new() -> Self {
        Self { replicates: 20, abs_ate_threshold: 0.25, estimator: linear_estimator_no_bootstrap() }
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
            &NoiseReplaceConfig {
                estimator: &self.estimator,
                replicates: self.replicates,
                abs_ate_threshold: self.abs_ate_threshold,
                target: NoiseReplaceTarget::Outcome,
                stream_base: 0xA7E0_0008,
                refuter_id: "dummy.outcome",
                failure_label: "dummy-outcome",
            },
        )
    }
}
