//! Placebo treatment refuter.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_estimate::{EstimationWorkspace, LinearAdjustmentAte};
use antecedent_kernels::shuffle;

use crate::common::{
    NoiseReplaceSpec, NoiseReplaceTarget, RefutationProblem, RefutationReport,
    check_replicate_count, complete_case_rows, finish_p_report, float64_full,
    linear_estimator_no_bootstrap, map_problem_replicates, noise_replace_refute,
    refit_effect_shared, with_replaced_float,
};
use crate::error::ValidationError;

/// How the placebo treatment column is constructed.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum PlaceboMode {
    /// Replace treatment with i.i.d. Gaussian noise (the conventional placebo-treatment default).
    #[default]
    RandomGaussian,
    /// Permute the observed treatment column (preserves the treatment marginal).
    Permute,
}

/// Replace treatment with independent noise or a permutation; expect ATE near zero.
///
/// Under the OLS / linear-adjustment estimators this refuter is gated to, the coefficient
/// on an independent placebo treatment is asymptotically zero by construction, so the
/// procedure cannot falsify a wrong causal claim. Reports set `informative: false`.
#[derive(Clone, Debug)]
pub struct PlaceboTreatment {
    /// Replicate count (each draw a fresh placebo treatment).
    pub replicates: u32,
    /// Pass if zero is a typical draw from the replicate distribution at this
    /// significance level (`p >= alpha`). The p-value is the two-sided normal tail
    /// of the null value under `N(mean, sd)` of the replicates — a typicality test
    /// of that one value, not a test of the mean (`sd` is not divided by `√n`).
    /// Non-finite replicates fail closed. A pass is not an informative falsification
    /// under OLS.
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
                &NoiseReplaceSpec {
                    replicates: self.replicates,
                    alpha: self.alpha,
                    target: NoiseReplaceTarget::Treatment,
                    stream_base: crate::streams::PLACEBO_REFIT,
                    refuter_id: "placebo.treatment",
                    failure_label: "placebo",
                },
            ),
            PlaceboMode::Permute => self.refute_permute(problem, ctx),
        }
    }

    fn refute_permute(
        &self,
        problem: &RefutationProblem<'_>,
        ctx: &ExecutionContext,
    ) -> Result<RefutationReport, ValidationError> {
        check_replicate_count(self.replicates)?;
        let treatment = problem.treatment();
        let factual = float64_full(problem.data, treatment)?;
        // Shuffle only the observed analysis population: invalid or masked payloads
        // are not observations and must never enter the placebo marginal.
        let mut ids = vec![treatment];
        if problem.temporal.is_none() {
            ids.push(problem.outcome());
            ids.extend_from_slice(&problem.estimand.adjustment_set);
            ids.extend_from_slice(&problem.query.effect_modifiers);
        }
        let (_, eligible) = complete_case_rows(problem.data, &ids)?;
        let observed: Vec<f64> = eligible.iter().map(|&row| factual[row]).collect();
        let ates = map_problem_replicates(
            ctx,
            problem,
            self.replicates,
            EstimationWorkspace::default,
            |problem, refit, workspace, ctx, r| {
                let mut perm = factual.clone();
                let mut rng =
                    ctx.rng.stream(crate::streams::replicate_stream(crate::streams::PLACEBO_SHUFFLE, r));
                let mut shuffled = observed.clone();
                shuffle(&mut rng, &mut shuffled);
                for (&row, value) in eligible.iter().zip(shuffled) {
                    perm[row] = value;
                }
                let data = with_replaced_float(problem.data, treatment, Arc::from(perm))?;
                let est = refit_effect_shared(
                    problem,
                    refit,
                    &data,
                    problem.estimand,
                    &[],
                    &self.estimator,
                    workspace,
                    ctx,
                )?;
                Ok(est.ate)
            },
        )?;
        // Permuting T under OLS still yields ~0 coefficient when T ⊥ Y | Z in
        // the fitted linear model; this is not a falsifier of the causal claim.
        finish_p_report(
            "placebo.treatment.permute",
            problem.original.ate,
            &ates,
            0.0,
            self.alpha,
            |mean_ate, p_value| {
                format!(
                    "placebo permute null value 0 is atypical under the replicate distribution \
                     (mean {mean_ate}, p={p_value} < alpha={})",
                    self.alpha
                )
            },
        )
    }
}
