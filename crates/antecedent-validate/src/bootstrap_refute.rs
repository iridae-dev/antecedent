//! Bootstrap CI coverage of the original point estimate (not placebo falsification).
//!
//! This check resamples rows, refits the same linear adjustment ATE, and asks whether
//! the *original* point estimate lies inside the percentile confidence interval of the
//! bootstrap ATEs. It is a stability / sampling-variability diagnostic — it does **not**
//! permute treatment, add noise outcomes, or otherwise falsify the causal claim the way
//! placebo / dummy-outcome refuters do.
//!
//! Report id: `bootstrap.ci_coverage`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss, clippy::cast_sign_loss)]

use std::sync::Arc;

use antecedent_core::{ExecutionContext, TargetPopulation};
use antecedent_data::TableView;
use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte};
use antecedent_kernels::unbiased_index;

use crate::common::{
    RefutationProblem, RefutationReport, complete_case_rows, forced_refit_estimator,
    linear_estimator_no_bootstrap, refit_effect, with_resampled_rows,
};
use crate::error::ValidationError;

/// IID row bootstrap of the whole `(T, Y, Z…)` design; "passes" if the original point estimate
/// falls inside the percentile confidence interval of the resampled ATEs.
///
/// Each replicate refits with `estimator.bootstrap_replicates = 0` (per the internal `fit_once` path)
/// so this never creates a nested bootstrap pool inside the resample loop.
#[derive(Clone, Debug)]
pub struct BootstrapRefute {
    /// Bootstrap replicates.
    pub replicates: u32,
    /// Confidence level for the percentile interval (e.g. 0.95).
    pub ci_level: f64,
    /// Estimator used for refits (bootstrap disabled).
    pub estimator: LinearAdjustmentAte,
}

impl Default for BootstrapRefute {
    fn default() -> Self {
        Self::new()
    }
}

impl BootstrapRefute {
    fn refute_composed(
        &self,
        problem: &RefutationProblem<'_>,
        workspace: &mut EstimationWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<RefutationReport, ValidationError> {
        let n = problem.data.row_count();
        let temporal = problem.temporal.ok_or(ValidationError::NotApplicable {
            message: "composed bootstrap requires a temporal context",
        })?;
        let time = temporal.time_index.ok_or(ValidationError::NotApplicable {
            message: "composed bootstrap requires a series time index",
        })?;
        let series =
            antecedent_data::TimeSeriesData::try_new(problem.data.storage().clone(), time.clone())?;
        let length =
            (temporal.indexer.history() as usize + 1).max((n as f64).cbrt().ceil() as usize).min(n);
        let mut rng = ctx.rng.stream(0xA7E0_0009_0000_u64);
        let mut indices = Vec::new();
        let mut ates = Vec::new();
        for _ in 0..self.replicates {
            if ctx.cancellation.is_cancelled() {
                return Err(ValidationError::Cancelled);
            }
            let sampled = antecedent_data::resample_timeseries(
                &series,
                antecedent_data::ResamplingPlan::CircularBlock { length },
                &mut rng,
                &mut indices,
            )?;
            let table = antecedent_data::TabularData::new(sampled.storage().clone());
            ates.push(
                refit_effect(
                    problem,
                    &table,
                    problem.estimand,
                    &[],
                    &self.estimator,
                    workspace,
                    ctx,
                )?
                .ate,
            );
        }
        Ok(coverage_report(problem, ates, self.ci_level, self.replicates))
    }

    /// Defaults: 200 replicates, 95% CI.
    #[must_use]
    pub fn new() -> Self {
        Self { replicates: 200, ci_level: 0.95, estimator: linear_estimator_no_bootstrap() }
    }

    /// Run the bootstrap CI-coverage check.
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
                message: "bootstrap CI coverage requires replicates >= 2",
            });
        }
        if !(self.ci_level > 0.0 && self.ci_level < 1.0) {
            return Err(ValidationError::NotApplicable {
                message: "bootstrap CI coverage requires ci_level in (0, 1)",
            });
        }
        let n = problem.data.row_count();
        let mut resample_ids = vec![problem.treatment(), problem.outcome()];
        // Temporal unfolded adjustment ids are dense node ids, not schema columns.
        if problem.temporal.is_none() {
            resample_ids.extend_from_slice(&problem.estimand.adjustment_set);
        } else {
            // Contemporaneous schema covariates already present in the series/panel table.
            for v in problem.data.schema().variables() {
                let id = v.id;
                if id != problem.treatment() && id != problem.outcome() {
                    resample_ids.push(id);
                }
            }
        }
        if problem.effect_refit.is_some() {
            return self.refute_composed(problem, workspace, ctx);
        }
        // Resample only complete-case rows so slots that are invalid in the source (whose
        // stored values are sentinels) never enter a replicate as real observations.
        let (keep, valid) = complete_case_rows(problem.data, &resample_ids)?;
        if valid.len() < 2 {
            return Err(ValidationError::NotApplicable {
                message: "bootstrap CI coverage requires at least 2 complete-case rows",
            });
        }
        let mut rng = ctx.rng.stream(0xA7E0_0009_0000_u64);
        let mut row_idx = vec![0usize; n];
        let mut ates = Vec::with_capacity(self.replicates as usize);
        // AllObserved static OLS: resampling the prepared design rows is the same
        // complete-case bootstrap as rebuilding a table and calling `prepare` again.
        // ATT/ATC re-filter treatment labels after the table rebuild, and temporal
        // designs need the lag indexer, so those stay on the table path.
        let use_design = problem.temporal.is_none()
            && matches!(problem.query.target_population, TargetPopulation::AllObserved);
        if use_design {
            let estimator = forced_refit_estimator(&self.estimator);
            let prepared = estimator
                .prepare(problem.data, problem.estimand, problem.query)
                .map_err(ValidationError::from)?;
            let selected = prepared.design.row_selection.as_ref();
            if selected.len() == prepared.design.nrows {
                let mut orig_to_design = vec![usize::MAX; n];
                for (k, &orig) in selected.iter().enumerate() {
                    if orig < n {
                        orig_to_design[orig] = k;
                    }
                }
                let n_design = prepared.design.nrows;
                let mut design_idx = vec![0usize; n_design];
                let mut x_boot = vec![0.0; n_design * prepared.design.ncols];
                let mut y_boot = vec![0.0; n_design];
                for _ in 0..self.replicates {
                    for slot in &mut row_idx {
                        *slot = valid[unbiased_index(&mut rng, valid.len())];
                    }
                    for (k, &orig) in selected.iter().enumerate() {
                        if orig >= n {
                            return Err(ValidationError::estimation_msg(
                                "prepared design row_selection is out of range",
                            ));
                        }
                        let src_orig = row_idx[orig];
                        let src_d = orig_to_design[src_orig];
                        if src_d == usize::MAX {
                            return Err(ValidationError::estimation_msg(
                                "bootstrap resample left the prepared design",
                            ));
                        }
                        design_idx[k] = src_d;
                    }
                    ates.push(estimator.ate_on_row_indices_into(
                        &prepared,
                        workspace,
                        &design_idx,
                        &mut x_boot,
                        &mut y_boot,
                    )?);
                }
                return Ok(coverage_report(problem, ates, self.ci_level, self.replicates));
            }
        }
        for _ in 0..self.replicates {
            for slot in &mut row_idx {
                *slot = valid[unbiased_index(&mut rng, valid.len())];
            }
            let data = with_resampled_rows(problem.data, &resample_ids, &row_idx, &keep)?;
            let est = refit_effect(
                problem,
                &data,
                problem.estimand,
                &[],
                &self.estimator,
                workspace,
                ctx,
            )?;
            ates.push(est.ate);
        }
        Ok(coverage_report(problem, ates, self.ci_level, self.replicates))
    }
}

fn coverage_report(
    problem: &RefutationProblem<'_>,
    mut ates: Vec<f64>,
    ci_level: f64,
    replicates: u32,
) -> RefutationReport {
    ates.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let m = ates.len();
    let lo_frac = (1.0 - ci_level) / 2.0;
    let hi_frac = 1.0 - lo_frac;
    let lo_idx = ((lo_frac * (m - 1) as f64).round() as usize).min(m - 1);
    let hi_idx = ((hi_frac * (m - 1) as f64).round() as usize).min(m - 1);
    let lo = ates[lo_idx];
    let hi = ates[hi_idx];
    let mean_ate = ates.iter().sum::<f64>() / m as f64;
    let width = hi - lo;
    let passed = problem.original.ate >= lo && problem.original.ate <= hi;
    RefutationReport {
        refuter: Arc::from("bootstrap.ci_coverage"),
        original_ate: problem.original.ate,
        refuted_ate: mean_ate,
        comparison: width,
        informative: true,
        passed,
        failure_condition: if passed {
            None
        } else {
            Some(Arc::from(format!(
                "original ATE {} outside {}% bootstrap CI [{lo}, {hi}] \
                 (coverage check of the point estimate, not a placebo falsification)",
                problem.original.ate,
                ci_level * 100.0
            )))
        },
        replicates,
    }
}
