//! Overlap / positivity refuter.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use causal_estimate::{OverlapPolicy, OverlapReport};
use causal_stats::GlmOptions;

use crate::common::{RefutationProblem, RefutationReport, diagnostic_overlap_report};
use crate::error::ValidationError;

/// Overlap / positivity assessment.
///
/// **No silent propensity rebuild:** when `problem.original.overlap_report` is `Some` (the
/// original estimate came from a propensity-based estimator), that report is reused verbatim.
/// When it is `None` (the linear-adjustment path, which deliberately skips propensity via
/// [`OverlapPolicy::ExplicitOverride`]), this refuter fits its own diagnostic-only logistic
/// propensity model on the adjustment covariates — explicitly, and only to populate the
/// diagnostics this check needs. That fit never feeds back into the original point estimate.
#[derive(Clone, Debug)]
pub struct OverlapRefuter {
    /// Minimum acceptable margin from the propensity boundary: pass requires propensities in
    /// `[eps, 1 - eps]`.
    pub eps: f64,
    /// Minimum acceptable fraction of effective sample size retained (`ess / n`).
    pub min_ess_fraction: f64,
    /// GLM options used only for the diagnostic-only fit (linear-adjustment path).
    pub glm_options: GlmOptions,
}

impl Default for OverlapRefuter {
    fn default() -> Self {
        Self::new()
    }
}

impl OverlapRefuter {
    /// Defaults: `eps = 0.05`, `min_ess_fraction = 0.5`.
    #[must_use]
    pub fn new() -> Self {
        Self { eps: 0.05, min_ess_fraction: 0.5, glm_options: GlmOptions::default() }
    }

    /// Run the overlap / positivity refuter.
    ///
    /// Complete separation / non-converged diagnostic GLM fits are treated as overlap
    /// failures (extreme propensities), not as hard errors.
    ///
    /// # Errors
    ///
    /// Data failures while building a diagnostic-only propensity fit.
    pub fn refute(
        &self,
        problem: &RefutationProblem<'_>,
    ) -> Result<RefutationReport, ValidationError> {
        let (report, replicates) = match &problem.original.overlap_report {
            Some(r) => (r.clone(), 0),
            None => (self.diagnostic_report(problem)?, 1),
        };
        let nrows = estimation_row_count(problem)? as f64;
        let Some(ess) = report.ess else {
            return Ok(RefutationReport {
                refuter: Arc::from("overlap.assessment"),
                original_ate: problem.original.ate,
                refuted_ate: problem.original.ate,
                comparison: f64::NAN,
                informative: false,
                passed: false,
                failure_condition: Some(Arc::from(
                    "overlap report has no weights; ESS is undefined",
                )),
                replicates,
            });
        };
        let ess_fraction = if nrows > 0.0 { ess / nrows } else { 0.0 };
        let bounds_ok =
            report.propensity_min >= self.eps && report.propensity_max <= 1.0 - self.eps;
        let ess_ok = ess_fraction >= self.min_ess_fraction;
        let passed = bounds_ok && ess_ok;
        let comparison = 1.0 - ess_fraction;
        Ok(RefutationReport {
            refuter: Arc::from("overlap.assessment"),
            original_ate: problem.original.ate,
            refuted_ate: problem.original.ate,
            comparison,
            informative: true,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "propensity range [{}, {}] or ess_fraction={ess_fraction} failed eps={} / \
                     min_ess_fraction={}",
                    report.propensity_min, report.propensity_max, self.eps, self.min_ess_fraction
                )))
            },
            replicates,
        })
    }

    fn diagnostic_report(
        &self,
        problem: &RefutationProblem<'_>,
    ) -> Result<OverlapReport, ValidationError> {
        diagnostic_overlap_report(problem, &self.glm_options, OverlapPolicy::require_diagnostics())
    }
}

fn estimation_row_count(problem: &RefutationProblem<'_>) -> Result<usize, ValidationError> {
    let mut ids = vec![problem.treatment(), problem.outcome()];
    ids.extend_from_slice(&problem.estimand.adjustment_set);
    let mask = problem.data.complete_case_mask(&ids).map_err(ValidationError::from)?;
    Ok(mask.iter().filter(|&&k| k).count())
}
