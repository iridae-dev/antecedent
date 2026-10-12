//! Overlap / positivity refuter.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

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
    pub fn refute_with_propensity(
        &self,
        problem: &RefutationProblem<'_>,
        propensity: &mut antecedent_stats::PropensityWorkspace,
    ) -> Result<RefutationReport, ValidationError> {
        if let Some(report) = self.continuous_report(problem)? {
            return Ok(report);
        }
        let (report, replicates) = if let Some(r) = &problem.original.diagnostics.overlap_report {
            (r.clone(), 0)
        } else {
            let fit = crate::common::diagnostic_overlap_report_with(
                problem,
                &self.glm_options,
                OverlapPolicy::require_diagnostics(),
                propensity,
            )?;
            if let Some(defect) = fit.defect {
                return Ok(Self::separated_report(problem, defect));
            }
            (fit.report, 1)
        };
        self.binary_report(problem, &report, replicates)
    }

    /// The overlap failure a separated / non-converged diagnostic propensity fit stands for.
    fn separated_report(problem: &RefutationProblem<'_>, defect: &str) -> RefutationReport {
        RefutationReport::builder("overlap.assessment")
            .original_ate(problem.original.ate)
            .refuted_ate(problem.original.ate)
            // Same polarity as the ESS report: the share of the sample without support.
            .comparison(1.0)
            .informative(true)
            .passed(false)
            .failure_condition(Some(Arc::from(defect)))
            .replicates(1)
            .build()
    }

    #[allow(clippy::float_cmp, reason = "Binary treatment is exactly 0/1")]
    pub(crate) fn continuous_report(
        &self,
        problem: &RefutationProblem<'_>,
    ) -> Result<Option<RefutationReport>, ValidationError> {
        if problem.original.diagnostics.overlap_report.is_none() {
            let ids = problem.estimation_ids(true);
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
            return Ok(RefutationReport::builder("overlap.assessment")
                .original_ate(problem.original.ate)
                .refuted_ate(problem.original.ate)
                .comparison(f64::NAN)
                .informative(false)
                .passed(false)
                .failure_condition(Some(Arc::from(
                    "overlap report has no weights; ESS is undefined",
                )))
                .replicates(replicates)
                .build());
        };
        let ess_fraction = if nrows > 0.0 { ess / nrows } else { 0.0 };
        let bounds_ok =
            report.propensity_min >= self.eps && report.propensity_max <= 1.0 - self.eps;
        let ess_ok = ess_fraction >= self.min_ess_fraction;
        let passed = bounds_ok && ess_ok;
        let comparison = 1.0 - ess_fraction;
        Ok(RefutationReport::builder("overlap.assessment")
            .original_ate(problem.original.ate)
            .refuted_ate(problem.original.ate)
            .comparison(comparison)
            .informative(true)
            .passed(passed)
            .failure_condition(if passed {
                None
            } else {
                Some(Arc::from(format!(
                    "propensity range [{}, {}] or ess_fraction={ess_fraction} failed eps={} / \
                     min_ess_fraction={}",
                    report.propensity_min, report.propensity_max, self.eps, self.min_ess_fraction
                )))
            })
            .replicates(replicates)
            .build())
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
        let mut means = vec![0.0; n];
        crate::common::column_major_dot(&design, n, &fit.coefficients, &mut means);
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
        Ok(RefutationReport::builder("overlap.continuous_support")
            .original_ate(problem.original.ate)
            .refuted_ate(problem.original.ate)
            .comparison(1.0 - fraction)
            .informative(true)
            .passed(passed)
            .failure_condition((!passed).then(|| {
                Arc::from(format!(
                    "both doses lie inside the empirical conditional residual interval on only {fraction} of target rows (required {}); linear treatment-location support diagnostic",
                    self.min_ess_fraction,
                ))
            }))
            .replicates(1)
            .build())
    }
}

fn estimation_row_count(problem: &RefutationProblem<'_>) -> Result<usize, ValidationError> {
    let ids = problem.estimation_ids(true);
    let mask = problem.data.complete_case_mask(&ids).map_err(ValidationError::from)?;
    Ok(mask.iter().filter(|&&k| k).count())
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use antecedent_core::{
        AssumptionSet, AverageEffectQuery, CausalSchemaBuilder, MeasurementSpec, RoleHint,
        SmallRoleSet, ValueType, VariableId,
    };
    use antecedent_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, ValidityBitmap};
    use antecedent_estimate::{EffectEstimate, OverlapPolicy};
    use antecedent_expr::ExprId;
    use antecedent_identify::IdentifiedEstimand;

    use super::*;

    /// A missing effect modifier used to stay in the ESS denominator. The diagnostic
    /// propensity is fit on the modifier-complete rows, so `ess / n` counted rows the
    /// estimate never used and reported half the effective-sample fraction.
    #[test]
    fn missing_modifier_does_not_inflate_the_ess_denominator() {
        let n = 8usize;
        let t: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
        let y: Vec<f64> = t.iter().map(|ti| 1.0 + 2.0 * ti).collect();
        let modifier: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let mut builder = CausalSchemaBuilder::new();
        for (name, role) in [
            ("t", RoleHint::TreatmentCandidate),
            ("y", RoleHint::OutcomeCandidate),
            ("w", RoleHint::Context),
        ] {
            builder
                .add_variable(
                    name,
                    ValueType::Continuous,
                    SmallRoleSet::from_hint(role),
                    None,
                    None,
                    MeasurementSpec::default(),
                )
                .unwrap();
        }
        let valid = ValidityBitmap::all_valid(n);
        let mut modifier_flags = vec![true; n];
        modifier_flags[4..].fill(false);
        let cols = vec![
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(0), Arc::from(t), valid.clone()).unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(VariableId::from_raw(1), Arc::from(y), valid.clone()).unwrap(),
            ),
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(2),
                    Arc::from(modifier),
                    crate::common::validity_from_flags(&modifier_flags).unwrap(),
                )
                .unwrap(),
            ),
        ];
        let data = antecedent_data::TabularData::new(
            OwnedColumnarStorage::try_new(builder.build().unwrap(), cols, None, None).unwrap(),
        );
        let estimand =
            IdentifiedEstimand::backdoor("backdoor.adjustment", Arc::from([]), ExprId::from_raw(0));
        let query =
            AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
                .with_effect_modifiers([VariableId::from_raw(2)]);
        let original =
            EffectEstimate::new(2.0, 0.1, AssumptionSet::new(), OverlapPolicy::ExplicitOverride);
        let problem = RefutationProblem::new(
            &data,
            &estimand,
            &query,
            &original,
            Some("conditional.linear.adjustment"),
            None,
        );
        let report = OverlapRefuter::new().refute(&problem).unwrap();
        // Balanced intercept-only propensity on the 4 complete rows has ESS = 4.
        // Dividing by all 8 rows reported comparison = 1 - 0.5 = 0.5.
        assert!(
            report.comparison < 0.05,
            "ess fraction should use the 4 modifier-complete rows, comparison={}",
            report.comparison
        );
        assert!(report.passed, "{report:?}");
    }
}
