//! Placebo treatment refuter.
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
        Self { replicates: 20, abs_ate_threshold: 0.25, estimator: linear_estimator_no_bootstrap() }
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
        noise_replace_refute(
            problem,
            workspace,
            ctx,
            &NoiseReplaceConfig {
                estimator: &self.estimator,
                replicates: self.replicates,
                abs_ate_threshold: self.abs_ate_threshold,
                target: NoiseReplaceTarget::Treatment,
                stream_base: 0xA7E0_0001,
                refuter_id: "placebo.treatment",
                failure_label: "placebo",
            },
        )
    }
}
