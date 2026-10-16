//! Random common cause refuter.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ExecutionContext, VariableId};
use antecedent_data::TableView;
use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte};
use antecedent_identify::IdentifiedEstimand;

use crate::common::{
    RefutationProblem, RefutationReport, check_replicate_count, fill_gaussian, finish_p_report,
    full_sample_refit_ate, linear_estimator_no_bootstrap, map_problem_replicates,
    refit_effect_shared, with_extra_float,
};
use crate::error::ValidationError;

/// Add an independent noise covariate; expect ATE largely unchanged.
///
/// Under OLS, an independent nuisance covariate is asymptotically orthogonal to the
/// treatment coefficient, so this procedure cannot falsify a wrong causal claim.
/// Reports set `informative: false`.
#[derive(Clone, Debug)]
pub struct RandomCommonCause {
    /// Replicate count.
    pub replicates: u32,
    /// Pass if the full-sample refit is a typical draw from the replicate distribution
    /// (`p >= alpha`). The p-value is the two-sided normal tail of that one value
    /// under `N(mean, sd)` of the replicates, not a test of the replicate mean.
    /// A non-finite replicate is an error, not a verdict. A pass is not an informative
    /// falsification under OLS.
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
        check_replicate_count(self.replicates)?;
        let method = problem.estimand.method_kind().ok();
        let static_ok = problem.estimand.is_adjustment_shaped();
        let temporal_ok = method == Some(antecedent_expr::EstimandMethod::TemporalBackdoorUnfolded)
            && problem.temporal.is_some();
        if !static_ok && !temporal_ok {
            return Err(ValidationError::NotApplicable {
                message: "random common cause requires backdoor.adjustment or \
                          temporal.backdoor.unfolded",
            });
        }
        let n = problem.data.row_count();
        // Centre on the same estimator's full-sample refit, not the published number: see
        // `full_sample_refit_ate` (a Bayesian posterior mean is not the least-squares refit).
        let centre = full_sample_refit_ate(problem, &self.estimator, workspace, ctx)?;
        let ates = map_problem_replicates(
            ctx,
            problem,
            self.replicates,
            || (EstimationWorkspace::default(), vec![0.0; n]),
            |problem, refit, (workspace, noise), ctx, r| {
                fill_gaussian(
                    noise,
                    ctx,
                    crate::streams::RANDOM_COMMON_CAUSE.wrapping_add(u64::from(r)),
                );
                let (data, new_id) = with_extra_float(
                    problem.data,
                    &format!("__rcc_{r}"),
                    Arc::<[f64]>::from(noise.clone()),
                )?;
                let (estimand, extra): (_, &[VariableId]) = if temporal_ok {
                    (problem.estimand.clone(), &[new_id])
                } else {
                    (extend_adjustment(problem.estimand, new_id), &[])
                };
                let est = refit_effect_shared(
                    problem,
                    refit,
                    &data,
                    &estimand,
                    extra,
                    &self.estimator,
                    workspace,
                    ctx,
                )?;
                Ok(est.ate)
            },
        )?;
        // Independent noise in the adjustment set cannot move the OLS ATE
        // enough to falsify a wrong claim.
        finish_p_report(
            "random.common_cause",
            problem.original.ate,
            &ates,
            centre,
            self.alpha,
            |mean_ate, p_value| {
                format!(
                    "refit null value {centre} is atypical under the replicate distribution \
                     (mean {mean_ate}, p={p_value} < alpha={})",
                    self.alpha
                )
            },
        )
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
