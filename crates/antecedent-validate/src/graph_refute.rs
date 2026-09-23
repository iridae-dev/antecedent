//! Leave-one-out adjustment-set sensitivity (not structural graph editing).
//!
//! Drops each backdoor adjustment covariate in turn, refits, and reports the
//! worst relative ATE change. This checks sensitivity to the *chosen adjustment
//! set*, not to edge deletions in an underlying DAG (the refutation problem has
//! no graph handle).
//!
//! The result is descriptive, not a validation: dropping a genuine confounder
//! *should* move the estimate, so a large change means the covariate is doing its
//! job and a small one means it is irrelevant. Neither says the causal claim is
//! right or wrong, so reports are `informative: false`; `passed` only records
//! whether the estimate stayed within `rel_delta_threshold` of its published value.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte};
use antecedent_identify::IdentifiedEstimand;

use crate::common::{
    RefutationProblem, RefutationReport, check_cancelled, complete_case_rows,
    linear_estimator_no_bootstrap, masked_sample_sd, refit_effect, validity_from_flags,
};
use crate::error::ValidationError;

/// Drop each adjustment covariate once and re-estimate (leave-one-out).
///
/// A large change flags sensitivity to the assumed adjustment set. Every refit uses
/// the rows that are complete for the *full* adjustment set, so a change is due to the
/// specification and not to rows re-entering the sample when a covariate with missing
/// values is dropped. When the adjustment set is empty there is nothing to drop, so
/// the check reports `passed = true` with zero replicates rather than fabricating a
/// comparison.
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
        if !problem.estimand.is_adjustment_shaped() {
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
        let sd_t = masked_sample_sd(problem.data, problem.treatment(), &mask)?;
        let sd_y = masked_sample_sd(problem.data, problem.outcome(), &mask)?;
        if !(sd_t.is_finite() && sd_t > 0.0 && sd_y.is_finite() && sd_y > 0.0) {
            return Err(ValidationError::NotApplicable {
                message: "adjustment drop-covariate requires non-degenerate treatment and \
                          outcome variation on the complete-case rows",
            });
        }
        let floor = 1e-3 * (sd_y / sd_t);
        // Hold the estimation sample fixed across drops. A temporal problem's rows are the
        // lag-aligned design, which the refit rebuilds; it is not a schema drop-covariate
        // target (see `ValidatorId::Graph`), so only the static path fixes a mask.
        let fixed;
        let data = if problem.temporal.is_none() && mask.iter().any(|&keep| !keep) {
            fixed = problem.data.with_analysis_mask(validity_from_flags(&mask)?)?;
            &fixed
        } else {
            problem.data
        };

        let mut worst_delta = 0.0_f64;
        let mut worst_ate = problem.original.ate;
        let mut worst_dropped = problem.estimand.adjustment_set[0];
        for drop_idx in 0..problem.estimand.adjustment_set.len() {
            check_cancelled(ctx)?;
            let reduced = drop_adjustment_at(problem.estimand, drop_idx);
            let est = refit_effect(problem, data, &reduced, &[], &self.estimator, workspace, ctx)?;
            if !est.ate.is_finite() {
                return Err(ValidationError::estimation_msg(
                    "non-finite leave-one-out refit effect: the refit did not produce a usable \
                     effect",
                ));
            }
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
            // Descriptive: a moving estimate is what a real confounder does.
            informative: false,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "relative |ΔATE|={worst_delta} exceeded threshold {} after dropping \
                     adjustment covariate {worst_dropped:?} (leave-one-out max); the estimate \
                     depends on that covariate, which is expected if it is a confounder",
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
    IdentifiedEstimand::new(
        Arc::clone(&base.method),
        Arc::from(zs),
        Arc::clone(&base.instruments),
        Arc::clone(&base.mediators),
        base.functional,
        base.rd_design,
    )
}
