//! Overlap / positivity refuter.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use std::sync::Arc;

use antecedent_estimate::OverlapPolicy;
use antecedent_stats::{DenseLinearAlgebra, FaerBackend, GlmOptions, LeastSquaresWorkspace};

use crate::common::{RefutationProblem, RefutationReport};
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
    /// Minimum acceptable fraction of effective sample size retained (`ess / n`),
    /// or supported target rows for a continuous-treatment contrast.
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
        let mut local = antecedent_stats::PropensityWorkspace::default();
        self.refute_with_propensity(problem, &mut local)
    }

    /// Like [`Self::refute`], reusing a warmed propensity workspace for diagnostic fits.
    ///
    /// # Errors
    ///
    /// Data failures while building a diagnostic-only propensity fit.
    #[allow(clippy::float_cmp)] // Exact 0/1 values distinguish the binary treatment contract.
    pub fn refute_with_propensity(
        &self,
        problem: &RefutationProblem<'_>,
        propensity: &mut antecedent_stats::PropensityWorkspace,
    ) -> Result<RefutationReport, ValidationError> {
        if let Some(report) = self.continuous_report(problem)? {
            return Ok(report);
        }
        let (report, replicates) = match &problem.original.overlap_report {
            Some(r) => (r.clone(), 0),
            None => (
                crate::common::diagnostic_overlap_report_with(
                    problem,
                    &self.glm_options,
                    OverlapPolicy::require_diagnostics(),
                    propensity,
                )?,
                1,
            ),
        };
        self.binary_report(problem, &report, replicates)
    }

    #[allow(clippy::float_cmp)] // Binary treatment is exactly 0/1.
    pub(crate) fn continuous_report(
        &self,
        problem: &RefutationProblem<'_>,
    ) -> Result<Option<RefutationReport>, ValidationError> {
        if problem.original.overlap_report.is_none() {
            let mut ids = vec![problem.treatment(), problem.outcome()];
            ids.extend_from_slice(&problem.estimand.adjustment_set);
            ids.extend_from_slice(&problem.query.effect_modifiers);
            let mask = problem.data.complete_case_mask(&ids)?;
            let treatment = problem.data.float64_masked(problem.treatment(), &mask)?;
            if treatment.iter().any(|&t| t != 0.0 && t != 1.0) {
                return self.continuous_support(problem, &mask, &treatment).map(Some);
            }
        }
        Ok(None)
    }

    fn binary_report(
        &self,
        problem: &RefutationProblem<'_>,
        report: &antecedent_estimate::OverlapReport,
        replicates: u32,
    ) -> Result<RefutationReport, ValidationError> {
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

    /// Descriptive conditional support under a linear treatment-location model.
    /// Both intervention levels must lie inside the empirical central residual
    /// interval after subtracting each row's fitted treatment mean. This is a
    /// support diagnostic, not a test or proof of conditional positivity.
    fn continuous_support(
        &self,
        problem: &RefutationProblem<'_>,
        mask: &[bool],
        treatment: &[f64],
    ) -> Result<RefutationReport, ValidationError> {
        let level = |intervention: &antecedent_core::Intervention| {
            if let antecedent_core::Intervention::Set { value, .. } = intervention {
                value.as_f64().ok_or(ValidationError::NotApplicable {
                    message: "continuous support requires numeric Set interventions",
                })
            } else {
                Err(ValidationError::NotApplicable {
                    message: "continuous support requires Set interventions",
                })
            }
        };
        let active = level(&problem.query.active)?;
        let control = level(&problem.query.control)?;
        let n = treatment.len();
        if n < 3 {
            return Err(ValidationError::NotApplicable {
                message: "continuous support requires three complete rows",
            });
        }
        let mut covariates = problem.estimand.adjustment_set.to_vec();
        covariates.extend_from_slice(&problem.query.effect_modifiers);
        covariates.sort_unstable();
        covariates.dedup();
        let mut design = vec![1.0; n];
        for &id in &covariates {
            design.extend(problem.data.float64_masked(id, mask)?);
        }
        let fit = FaerBackend.least_squares(
            &design,
            n,
            covariates.len() + 1,
            treatment,
            &mut LeastSquaresWorkspace::default(),
        )?;
        let means: Vec<f64> = (0..n)
            .map(|r| {
                fit.coefficients.iter().enumerate().map(|(c, beta)| beta * design[c * n + r]).sum()
            })
            .collect();
        let mut residuals: Vec<_> =
            treatment.iter().zip(&means).map(|(t, mean)| t - mean).collect();
        residuals.sort_by(f64::total_cmp);
        let quantile = |p: f64| {
            // Select without float-to-index casts; only two quantiles are needed.
            residuals
                .iter()
                .enumerate()
                .find(|(i, _)| *i as f64 >= p * (n - 1) as f64)
                .map_or(residuals[n - 1], |(_, &value)| value)
        };
        let lower = quantile(self.eps.clamp(0.0, 0.5));
        let upper = quantile(1.0 - self.eps.clamp(0.0, 0.5));
        let supported = means
            .iter()
            .filter(|&&mean| {
                [active, control].iter().all(|dose| (lower..=upper).contains(&(dose - mean)))
            })
            .count();
        let fraction = supported as f64 / n as f64;
        let passed = fraction >= self.min_ess_fraction;
        // Same polarity as binary overlap.assessment: comparison is unsupported mass.
        Ok(RefutationReport::new(
            "overlap.continuous_support", problem.original.ate, problem.original.ate,
            1.0 - fraction, true, passed,
            (!passed).then(|| Arc::from(format!(
                "both doses lie inside the empirical conditional residual interval on only {fraction} of target rows (required {}); linear treatment-location support diagnostic",
                self.min_ess_fraction,
            ))), 1,
        ))
    }
}

fn estimation_row_count(problem: &RefutationProblem<'_>) -> Result<usize, ValidationError> {
    let mut ids = vec![problem.treatment(), problem.outcome()];
    ids.extend_from_slice(&problem.estimand.adjustment_set);
    let mask = problem.data.complete_case_mask(&ids).map_err(ValidationError::from)?;
    Ok(mask.iter().filter(|&&k| k).count())
}
