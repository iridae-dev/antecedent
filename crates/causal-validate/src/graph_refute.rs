//! Graph refutation: sensitivity to dropping one adjustment covariate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::ExecutionContext;
use causal_estimate::{EstimationWorkspace, LinearAdjustmentAte};
use causal_identify::IdentifiedEstimand;

use crate::common::{RefutationProblem, RefutationReport, fit_once};
use crate::error::ValidationError;

/// Drop the first adjustment covariate (a proxy for questioning whether it belongs in the
/// causal graph as a confounder / removing that edge) and re-estimate.
///
/// A large change flags sensitivity to the assumed graph structure. When the adjustment set is
/// empty there is nothing to drop, so the check reports `informative = false` rather than
/// fabricating a comparison.
#[derive(Clone, Debug)]
pub struct GraphRefuter {
    /// Pass if `|refuted_ate - original_ate|` is below this threshold.
    pub abs_delta_threshold: f64,
    /// Estimator used for the refit (bootstrap disabled).
    pub estimator: LinearAdjustmentAte,
}

impl Default for GraphRefuter {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphRefuter {
    /// Default threshold: 0.5.
    #[must_use]
    pub fn new() -> Self {
        let mut estimator = LinearAdjustmentAte::new();
        estimator.bootstrap_replicates = 0;
        Self { abs_delta_threshold: 0.5, estimator }
    }

    /// Run the graph refuter.
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
        if &*problem.estimand.method != "backdoor.adjustment" {
            return Err(ValidationError::NotApplicable {
                message: "graph refutation requires backdoor.adjustment estimand",
            });
        }
        if problem.estimand.adjustment_set.is_empty() {
            return Ok(RefutationReport {
                refuter: Arc::from("graph.refute"),
                original_ate: problem.original.ate,
                refuted_ate: problem.original.ate,
                comparison: 0.0,
                informative: false,
                passed: true,
                failure_condition: None,
                replicates: 0,
            });
        }
        let reduced = drop_first_adjustment(problem.estimand);
        let est =
            fit_once(&self.estimator, problem.data, &reduced, problem.query, workspace, ctx)?;
        let delta = (est.ate - problem.original.ate).abs();
        let passed = delta < self.abs_delta_threshold;
        Ok(RefutationReport {
            refuter: Arc::from("graph.refute"),
            original_ate: problem.original.ate,
            refuted_ate: est.ate,
            comparison: delta,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "|ΔATE|={delta} exceeded threshold {} after dropping adjustment covariate \
                     {:?}",
                    self.abs_delta_threshold, problem.estimand.adjustment_set[0]
                )))
            },
            replicates: 1,
        })
    }
}

fn drop_first_adjustment(base: &IdentifiedEstimand) -> IdentifiedEstimand {
    let zs: Vec<_> = base.adjustment_set.iter().copied().skip(1).collect();
    IdentifiedEstimand {
        method: Arc::clone(&base.method),
        adjustment_set: Arc::from(zs),
        instruments: Arc::clone(&base.instruments),
        mediators: Arc::clone(&base.mediators),
        functional: base.functional,
    }
}
