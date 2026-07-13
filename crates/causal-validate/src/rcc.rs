//! Random common cause refuter.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::sync::Arc;

use causal_core::{ExecutionContext, VariableId};
use causal_data::TableView;
use causal_estimate::{EstimationWorkspace, LinearAdjustmentAte};
use causal_identify::IdentifiedEstimand;

use crate::common::{
    RefutationProblem, RefutationReport, fill_gaussian, fit_once, with_extra_float,
};
use crate::error::ValidationError;

/// Add an independent noise covariate; expect ATE largely unchanged.
#[derive(Clone, Debug)]
pub struct RandomCommonCause {
    /// Replicate count.
    pub replicates: u32,
    /// Pass if mean `|refuted_ate - original_ate|` is below this threshold.
    pub abs_delta_threshold: f64,
    /// Estimator used for refits (bootstrap disabled).
    pub estimator: LinearAdjustmentAte,
}

impl Default for RandomCommonCause {
    fn default() -> Self {
        Self::new()
    }
}

impl RandomCommonCause {
    /// Default: 20 replicates, threshold 0.15.
    #[must_use]
    pub fn new() -> Self {
        let mut estimator = LinearAdjustmentAte::new();
        estimator.bootstrap_replicates = 0;
        Self { replicates: 20, abs_delta_threshold: 0.15, estimator }
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
        if self.replicates == 0 {
            return Err(ValidationError::NotApplicable {
                message: "random common cause requires replicates > 0",
            });
        }
        if &*problem.estimand.method != "backdoor.adjustment" {
            return Err(ValidationError::NotApplicable {
                message: "random common cause requires backdoor.adjustment estimand",
            });
        }
        let n = problem.data.row_count();
        let mut noise = vec![0.0; n];
        let mut sum_delta = 0.0;
        let mut sum_ate = 0.0;
        for r in 0..self.replicates {
            fill_gaussian(&mut noise, ctx, 0xA7E0_0002_u64.wrapping_add(u64::from(r)));
            let (data, new_id) = with_extra_float(
                problem.data,
                &format!("__rcc_{r}"),
                Arc::<[f64]>::from(noise.clone()),
            )?;
            let estimand = extend_adjustment(problem.estimand, new_id);
            let est = fit_once(&self.estimator, &data, &estimand, problem.query, workspace, ctx)?;
            sum_delta += (est.ate - problem.original.ate).abs();
            sum_ate += est.ate;
        }
        let mean_delta = sum_delta / f64::from(self.replicates);
        let mean_ate = sum_ate / f64::from(self.replicates);
        let passed = mean_delta < self.abs_delta_threshold;
        Ok(RefutationReport {
            refuter: Arc::from("random.common_cause"),
            original_ate: problem.original.ate,
            refuted_ate: mean_ate,
            comparison: mean_delta,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "mean |ΔATE|={mean_delta} exceeded threshold {}",
                    self.abs_delta_threshold
                )))
            },
            replicates: self.replicates,
        })
    }
}

fn extend_adjustment(base: &IdentifiedEstimand, extra: VariableId) -> IdentifiedEstimand {
    let mut zs: Vec<VariableId> = base.adjustment_set.to_vec();
    zs.push(extra);
    IdentifiedEstimand {
        method: Arc::clone(&base.method),
        adjustment_set: Arc::from(zs),
        functional: base.functional,
    }
}
