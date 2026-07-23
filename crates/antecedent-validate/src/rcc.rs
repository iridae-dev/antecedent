//! Random common cause refuter.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TableView;
use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte};
use antecedent_identify::IdentifiedEstimand;

use crate::common::{
    RefutationProblem, RefutationReport, fill_gaussian, linear_estimator_no_bootstrap,
    refit_effect, replicate_p_value, with_extra_float,
};
use crate::error::ValidationError;

/// Add an independent noise covariate; expect ATE largely unchanged.
#[derive(Clone, Debug)]
pub struct RandomCommonCause {
    /// Replicate count.
    pub replicates: u32,
    /// Pass if the refit ATE distribution is consistent with the original estimate at
    /// this significance level (two-sided normal test on the replicates, `p >= alpha`).
    pub alpha: f64,
    /// Estimator used for refits (bootstrap disabled).
    pub estimator: LinearAdjustmentAte,
}

impl Default for RandomCommonCause {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomCommonCause {
    /// Default: 20 replicates, significance level 0.05.
    #[must_use]
    pub fn new() -> Self {
        Self { replicates: 20, alpha: 0.05, estimator: linear_estimator_no_bootstrap() }
    }

    /// Run the random-common-cause refuter.
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
        if self.replicates < 2 {
            return Err(ValidationError::NotApplicable {
                message: "random common cause requires replicates >= 2",
            });
        }
        let method = problem.estimand.method_kind().ok();
        let static_ok = method == Some(antecedent_expr::EstimandMethod::BackdoorAdjustment);
        let temporal_ok = method == Some(antecedent_expr::EstimandMethod::TemporalBackdoorUnfolded)
            && problem.temporal.is_some();
        if !static_ok && !temporal_ok {
            return Err(ValidationError::NotApplicable {
                message: "random common cause requires backdoor.adjustment or temporal.backdoor.unfolded",
            });
        }
        let n = problem.data.row_count();
        let mut noise = vec![0.0; n];
        let mut ates = Vec::with_capacity(self.replicates as usize);
        for r in 0..self.replicates {
            fill_gaussian(&mut noise, ctx, 0xA7E0_0002_0000_u64.wrapping_add(u64::from(r)));
            let (data, new_id) = with_extra_float(
                problem.data,
                &format!("__rcc_{r}"),
                Arc::<[f64]>::from(noise.clone()),
            )?;
            let est = if temporal_ok {
                refit_effect(problem, &data, problem.estimand, &[new_id], workspace, ctx)?
            } else {
                let estimand = extend_adjustment(problem.estimand, new_id);
                refit_effect(problem, &data, &estimand, &[], workspace, ctx)?
            };
            ates.push(est.ate);
        }
        let mean_ate = ates.iter().sum::<f64>() / f64::from(self.replicates);
        let p_value = replicate_p_value(&ates, problem.original.ate);
        let passed = p_value >= self.alpha;
        Ok(RefutationReport {
            refuter: Arc::from("random.common_cause"),
            original_ate: problem.original.ate,
            refuted_ate: mean_ate,
            comparison: p_value,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "refit ATE distribution (mean {mean_ate}) is inconsistent with the \
                     original estimate (p={p_value} < alpha={})",
                    self.alpha
                )))
            },
            replicates: self.replicates,
        })
    }
}

fn extend_adjustment(base: &IdentifiedEstimand, extra: VariableId) -> IdentifiedEstimand {
    let mut zs: Vec<VariableId> = base.adjustment_set.to_vec();
    zs.push(extra);
    IdentifiedEstimand::new(
        Arc::clone(&base.method),
        Arc::from(zs),
        Arc::clone(&base.instruments),
        Arc::clone(&base.mediators),
        base.functional,
        base.rd_design,
    )
}
