//! Graph refutation: sensitivity to dropping one adjustment covariate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::ExecutionContext;
use causal_estimate::{EstimationWorkspace, LinearAdjustmentAte};
use causal_identify::IdentifiedEstimand;

use crate::common::{
    RefutationProblem, RefutationReport, complete_case_rows, fit_once,
    linear_estimator_no_bootstrap, masked_sample_sd,
};
use crate::error::ValidationError;

/// Drop the first adjustment covariate (a proxy for questioning whether it belongs in the
/// causal graph as a confounder / removing that edge) and re-estimate.
///
/// A large change flags sensitivity to the assumed graph structure. When the adjustment set is
/// empty there is nothing to drop, so the check reports `informative = false` rather than
/// fabricating a comparison.
#[derive(Clone, Debug)]
pub struct GraphRefuter {
    /// Pass if `|refuted_ate - original_ate| / |original_ate|` is below this threshold
    /// (relative change, so the verdict is invariant to outcome units).
    pub rel_delta_threshold: f64,
    /// Estimator used for the refit (bootstrap disabled).
    pub estimator: LinearAdjustmentAte,
}

impl Default for GraphRefuter {
    fn default() -> Self {
        Self::new()
    }
}

impl GraphRefuter {
    /// Default threshold: 0.5 (the estimate may move by up to half its own magnitude).
    #[must_use]
    pub fn new() -> Self {
        Self { rel_delta_threshold: 0.5, estimator: linear_estimator_no_bootstrap() }
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
        if problem.estimand.method_kind().ok() != Some(causal_expr::EstimandMethod::BackdoorAdjustment)
        {
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
        let est = fit_once(&self.estimator, problem.data, &reduced, problem.query, workspace, ctx)?;
        // Relative change with an sd-based floor on the denominator: a near-zero original
        // estimate that moves materially when a covariate is dropped is graph-sensitive.
        let mut ids = vec![problem.treatment(), problem.outcome()];
        ids.extend_from_slice(&problem.estimand.adjustment_set);
        let (mask, _valid) = complete_case_rows(problem.data, &ids)?;
        let sd_t = masked_sample_sd(problem.data, problem.treatment(), &mask)?.max(1e-12);
        let sd_y = masked_sample_sd(problem.data, problem.outcome(), &mask)?.max(1e-12);
        let floor = 1e-3 * (sd_y / sd_t);
        let delta = (est.ate - problem.original.ate).abs() / problem.original.ate.abs().max(floor);
        let passed = delta < self.rel_delta_threshold;
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
                    "relative |ΔATE|={delta} exceeded threshold {} after dropping adjustment \
                     covariate {:?}",
                    self.rel_delta_threshold, problem.estimand.adjustment_set[0]
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
