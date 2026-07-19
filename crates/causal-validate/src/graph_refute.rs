//! Leave-one-out adjustment-set sensitivity (not structural graph editing).
//!
//! Drops each backdoor adjustment covariate in turn, refits, and reports the
//! worst relative ATE change. This checks sensitivity to the *chosen adjustment
//! set*, not to edge deletions in an underlying DAG (the refutation problem has
//! no graph handle).
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

/// Drop each adjustment covariate once and re-estimate (leave-one-out).
///
/// A large change flags sensitivity to the assumed adjustment set. When the
/// adjustment set is empty there is nothing to drop, so the check reports
/// `informative = false` rather than fabricating a comparison.
///
/// Historically named "graph refuter"; the report id is
/// `adjustment.drop_covariate` to match the actual check.
#[derive(Clone, Debug)]
pub struct GraphRefuter {
    /// Pass if max `|refuted_ate - original_ate| / |original_ate|` is below this
    /// threshold (relative change, so the verdict is invariant to outcome units).
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

    /// Run leave-one-out adjustment-set sensitivity.
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
                message: "adjustment drop-covariate requires backdoor.adjustment estimand",
            });
        }
        if problem.estimand.adjustment_set.is_empty() {
            return Ok(RefutationReport {
                refuter: Arc::from("adjustment.drop_covariate"),
                original_ate: problem.original.ate,
                refuted_ate: problem.original.ate,
                comparison: 0.0,
                informative: false,
                passed: true,
                failure_condition: None,
                replicates: 0,
            });
        }
        let mut ids = vec![problem.treatment(), problem.outcome()];
        ids.extend_from_slice(&problem.estimand.adjustment_set);
        let (mask, _valid) = complete_case_rows(problem.data, &ids)?;
        let sd_t = masked_sample_sd(problem.data, problem.treatment(), &mask)?.max(1e-12);
        let sd_y = masked_sample_sd(problem.data, problem.outcome(), &mask)?.max(1e-12);
        let floor = 1e-3 * (sd_y / sd_t);

        let mut worst_delta = 0.0_f64;
        let mut worst_ate = problem.original.ate;
        let mut worst_dropped = problem.estimand.adjustment_set[0];
        for drop_idx in 0..problem.estimand.adjustment_set.len() {
            let reduced = drop_adjustment_at(problem.estimand, drop_idx);
            let est =
                fit_once(&self.estimator, problem.data, &reduced, problem.query, workspace, ctx)?;
            // Relative change with an sd-based floor on the denominator: a near-zero original
            // estimate that moves materially when a covariate is dropped is set-sensitive.
            let delta =
                (est.ate - problem.original.ate).abs() / problem.original.ate.abs().max(floor);
            if delta >= worst_delta {
                worst_delta = delta;
                worst_ate = est.ate;
                worst_dropped = problem.estimand.adjustment_set[drop_idx];
            }
        }
        let passed = worst_delta < self.rel_delta_threshold;
        Ok(RefutationReport {
            refuter: Arc::from("adjustment.drop_covariate"),
            original_ate: problem.original.ate,
            refuted_ate: worst_ate,
            comparison: worst_delta,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "relative |ΔATE|={worst_delta} exceeded threshold {} after dropping \
                     adjustment covariate {worst_dropped:?} (leave-one-out max)",
                    self.rel_delta_threshold
                )))
            },
            replicates: u32::try_from(problem.estimand.adjustment_set.len()).unwrap_or(u32::MAX),
        })
    }
}

fn drop_adjustment_at(base: &IdentifiedEstimand, drop_idx: usize) -> IdentifiedEstimand {
    let zs: Vec<_> = base
        .adjustment_set
        .iter()
        .copied()
        .enumerate()
        .filter(|(i, _)| *i != drop_idx)
        .map(|(_, z)| z)
        .collect();
    IdentifiedEstimand {
        method: Arc::clone(&base.method),
        adjustment_set: Arc::from(zs),
        instruments: Arc::clone(&base.instruments),
        mediators: Arc::clone(&base.mediators),
        functional: base.functional,
        rd_design: base.rd_design.clone(),
    }
}
