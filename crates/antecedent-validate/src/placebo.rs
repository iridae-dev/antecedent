//! Placebo treatment refuter.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte};
use antecedent_kernels::shuffle;

use crate::common::{
    NoiseReplaceTarget, RefutationProblem, RefutationReport, float64_full,
    linear_estimator_no_bootstrap, noise_replace_refute, refit_effect, replicate_p_value,
    with_replaced_float,
};
use crate::error::ValidationError;

/// How the placebo treatment column is constructed.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PlaceboMode {
    /// Replace treatment with i.i.d. Gaussian noise (pinned baseline default).
    #[default]
    RandomGaussian,
    /// Permute the observed treatment column (preserves the treatment marginal).
    Permute,
}

/// Replace treatment with independent noise or a permutation; expect ATE near zero.
#[derive(Clone, Debug)]
pub struct PlaceboTreatment {
    /// Replicate count (each draw a fresh placebo treatment).
    pub replicates: u32,
    /// Pass if the placebo ATE distribution is consistent with zero at this significance
    /// level (two-sided normal test on the replicates, `p >= alpha`).
    pub alpha: f64,
    /// Estimator used for refits (bootstrap disabled to avoid nested pools).
    pub estimator: LinearAdjustmentAte,
    /// Placebo construction mode.
    pub mode: PlaceboMode,
}

impl Default for PlaceboTreatment {
    fn default() -> Self {
        Self::new()
    }
}

impl PlaceboTreatment {
    /// Default: 20 replicates, significance level 0.05, Gaussian replacement.
    #[must_use]
    pub fn new() -> Self {
        Self {
            replicates: 20,
            alpha: 0.05,
            estimator: linear_estimator_no_bootstrap(),
            mode: PlaceboMode::RandomGaussian,
        }
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
        match self.mode {
            PlaceboMode::RandomGaussian => noise_replace_refute(
                problem,
                workspace,
                ctx,
                &self.estimator,
                self.replicates,
                self.alpha,
                NoiseReplaceTarget::Treatment,
                0xA7E0_0001_0000,
                "placebo.treatment",
                "placebo",
            ),
            PlaceboMode::Permute => self.refute_permute(problem, workspace, ctx),
        }
    }

    fn refute_permute(
        &self,
        problem: &RefutationProblem<'_>,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<RefutationReport, ValidationError> {
        if self.replicates < 2 {
            return Err(ValidationError::NotApplicable {
                message: "placebo permute refuter requires replicates >= 2",
            });
        }
        let treatment = problem.treatment();
        let factual = float64_full(problem.data, treatment)?;
        let mut ates = Vec::with_capacity(self.replicates as usize);
        for r in 0..self.replicates {
            let mut perm = factual.clone();
            let mut rng = ctx.rng.stream(0xA7E0_0001_1000_u64.wrapping_add(u64::from(r)));
            shuffle(&mut rng, &mut perm);
            let data = with_replaced_float(problem.data, treatment, Arc::from(perm))?;
            let est = refit_effect(problem, &data, problem.estimand, &[], workspace, ctx)?;
            ates.push(est.ate);
        }
        let mean_ate = ates.iter().sum::<f64>() / f64::from(self.replicates);
        let p_value = replicate_p_value(&ates, 0.0);
        let passed = p_value >= self.alpha;
        Ok(RefutationReport {
            refuter: Arc::from("placebo.treatment.permute"),
            original_ate: problem.original.ate,
            refuted_ate: mean_ate,
            comparison: p_value,
            informative: true,
            passed,
            failure_condition: (!passed).then(|| {
                Arc::from(format!(
                    "placebo permute ATE distribution (mean {mean_ate}) is inconsistent with zero \
                     (p={p_value} < alpha={})",
                    self.alpha
                ))
            }),
            replicates: self.replicates,
        })
    }
}
