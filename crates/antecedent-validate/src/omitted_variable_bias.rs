//! Omitted-variable-bias sensitivity for linear adjustment: partial R², the robustness
//! value and benchmarking against observed covariates (Cinelli and Hazlett 2020, "Making
//! sense of sensitivity: extending omitted variable bias", JRSS-B 82(1)).
//!
//! For the coefficient `τ̂` of a treatment `D` in the linear regression of `Y` on `D` and
//! covariates `X`, with standard error `se` and residual degrees of freedom `dof`, an omitted
//! confounder `Z` with partial R² `R²_{D~Z|X}` with the treatment and `R²_{Y~Z|D,X}` with the
//! outcome changes the estimate by at most
//!
//! `|bias| = se · sqrt(dof) · sqrt(R²_{Y~Z|D,X} · R²_{D~Z|X} / (1 − R²_{D~Z|X}))`
//!
//! and the standard error becomes
//! `se · sqrt((1 − R²_{Y~Z|D,X}) / (1 − R²_{D~Z|X})) · sqrt(dof / (dof − 1))`.
//!
//! * The partial R² of the treatment with the outcome is `R²_{Y~D|X} = t² / (t² + dof)`; it is
//!   also the share of residual treatment variance an extreme confounder (explaining all
//!   residual outcome variance) must explain to bring the estimate to zero.
//! * The robustness value `RV_q = ½ (sqrt(f_q⁴ + 4 f_q²) − f_q²)`, `f_q = q·|t|/sqrt(dof)`,
//!   is the equal partial R² with treatment and outcome that reduces the estimate by
//!   `100·q`%; `RV_{q,α}` replaces `f_q` by `f_q − |t*_{α, dof−1}| / sqrt(dof − 1)` (zero when
//!   negative) to bring the adjusted interval to include `(1 − q)·τ̂`, with the boundary
//!   solution `(f_q² − f*²) / (1 + f_q²)` when `f_q > 1/f*`.
//! * Benchmarking bounds a confounder `k_D` and `k_Y` times as strong as an observed
//!   covariate `X_j` (partial R²s `R²_{D~X_j|X_{−j}}`, `R²_{Y~X_j|D,X_{−j}}`) by
//!   `R²_{D~Z|X} = k_D · R²_{D~X_j} / (1 − R²_{D~X_j})` and
//!   `R²_{Y~Z|D,X} = ((sqrt(k_Y) + sqrt(r)) / sqrt(1 − r))² · R²_{Y~X_j} / (1 − R²_{Y~X_j})`,
//!   `r = k_D · R²_{D~X_j}² / ((1 − k_D·R²_{D~X_j})(1 − R²_{D~X_j}))`, capped at one.
//!
//! [`ovb_range`] reports the extreme-scenario curve and the robustness values as an
//! [`AssumptionRange`] tied to the target.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace, student_t_ppf};

use crate::ValidationError;
use crate::assumption_range::{AssumptionRange, RangePoint, SensitivityAssumption};
use crate::target_native::RefutationTarget;

/// Summary statistics of a linear-adjustment estimate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LinearAdjustmentSummary {
    /// Treatment coefficient `τ̂`.
    pub estimate: f64,
    /// Its standard error.
    pub standard_error: f64,
    /// Residual degrees of freedom of the outcome regression.
    pub dof: f64,
}

impl LinearAdjustmentSummary {
    /// Validated summary.
    ///
    /// # Errors
    ///
    /// A non-finite estimate, a non-positive standard error, or fewer than two degrees of
    /// freedom.
    pub fn new(estimate: f64, standard_error: f64, dof: f64) -> Result<Self, ValidationError> {
        if !(estimate.is_finite()
            && standard_error.is_finite()
            && standard_error > 0.0
            && dof.is_finite()
            && dof >= 2.0)
        {
            return Err(ValidationError::NotApplicable {
                message: "omitted-variable-bias analysis needs a finite estimate, a positive \
                          standard error and at least two residual degrees of freedom",
            });
        }
        Ok(Self { estimate, standard_error, dof })
    }

    /// `t = τ̂ / se`.
    #[must_use]
    pub fn t_statistic(&self) -> f64 {
        self.estimate / self.standard_error
    }

    /// Partial Cohen's `f = t / sqrt(dof)`.
    #[must_use]
    pub fn partial_f(&self) -> f64 {
        self.t_statistic() / self.dof.sqrt()
    }

    /// `R²_{Y~D|X} = t² / (t² + dof)`.
    #[must_use]
    pub fn partial_r2_treatment(&self) -> f64 {
        partial_r2_from_t(self.t_statistic(), self.dof)
    }

    /// Robustness value `RV_q` (with `alpha = None`) or `RV_{q,α}`.
    #[must_use]
    pub fn robustness_value(&self, q: f64, alpha: Option<f64>) -> f64 {
        let fq = q * self.partial_f().abs();
        let Some(alpha) = alpha else {
            return rv_from_f(fq);
        };
        let fcrit =
            student_t_ppf(1.0 - alpha / 2.0, self.dof - 1.0).abs() / (self.dof - 1.0).sqrt();
        let fqa = fq - fcrit;
        if fqa <= 0.0 {
            0.0
        } else if fq > 1.0 / fcrit {
            (fq * fq - fcrit * fcrit) / (1.0 + fq * fq)
        } else {
            rv_from_f(fqa)
        }
    }

    /// Largest absolute bias of a confounder with partial R²s `r2dz` (treatment) and `r2yz`
    /// (outcome).
    #[must_use]
    pub fn bias_bound(&self, r2dz: f64, r2yz: f64) -> f64 {
        self.standard_error * self.dof.sqrt() * (r2yz * r2dz / (1.0 - r2dz)).sqrt()
    }

    /// Estimate moved toward zero by [`Self::bias_bound`].
    #[must_use]
    pub fn adjusted_estimate(&self, r2dz: f64, r2yz: f64) -> f64 {
        let bias = self.bias_bound(r2dz, r2yz);
        if self.estimate >= 0.0 { self.estimate - bias } else { self.estimate + bias }
    }

    /// Standard error after adjusting for the confounder.
    #[must_use]
    pub fn adjusted_standard_error(&self, r2dz: f64, r2yz: f64) -> f64 {
        self.standard_error
            * ((1.0 - r2yz) / (1.0 - r2dz)).sqrt()
            * (self.dof / (self.dof - 1.0)).sqrt()
    }

    /// Share of residual treatment variance a confounder with outcome partial R² `r2yz`
    /// must explain to reduce the estimate by `100·q`%: `f_q² / (f_q² + R²_{Y~Z|D,X})`.
    #[must_use]
    pub fn break_even_r2dz(&self, r2yz: f64, q: f64) -> f64 {
        let fq2 = (q * self.partial_f()).powi(2);
        fq2 / (fq2 + r2yz)
    }
}

fn rv_from_f(f: f64) -> f64 {
    let f2 = f * f;
    0.5 * ((f2 * f2 + 4.0 * f2).sqrt() - f2)
}

/// `t² / (t² + dof)`.
#[must_use]
pub fn partial_r2_from_t(t: f64, dof: f64) -> f64 {
    t * t / (t * t + dof)
}

/// Partial R²s of a confounder `k_d` and `k_y` times as strong as an observed covariate.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BenchmarkBound {
    /// `k_D`.
    pub kd: f64,
    /// `k_Y`.
    pub ky: f64,
    /// Bound on `R²_{D~Z|X}`.
    pub r2dz: f64,
    /// Bound on `R²_{Y~Z|D,X}` (at most one).
    pub r2yz: f64,
}

/// Confounder bounds from an observed covariate's partial R²s.
///
/// # Errors
///
/// Partial R²s outside `[0, 1)`, a non-positive multiplier, or a multiplier so large that
/// the implied treatment partial R² reaches one.
pub fn benchmark_bound(
    r2dxj: f64,
    r2yxj: f64,
    kd: f64,
    ky: f64,
) -> Result<BenchmarkBound, ValidationError> {
    if !((0.0..1.0).contains(&r2dxj) && (0.0..1.0).contains(&r2yxj) && kd > 0.0 && ky > 0.0) {
        return Err(ValidationError::NotApplicable {
            message: "benchmark partial R2 values must lie in [0, 1) and multipliers must be \
                      positive",
        });
    }
    let r2dz = kd * r2dxj / (1.0 - r2dxj);
    if r2dz >= 1.0 {
        return Err(ValidationError::NotApplicable {
            message: "this treatment multiplier implies a confounder explaining all residual \
                      treatment variance; next attempt: a smaller k_D",
        });
    }
    let r2zxj = kd * r2dxj * r2dxj / ((1.0 - kd * r2dxj) * (1.0 - r2dxj));
    if r2zxj >= 1.0 {
        return Err(ValidationError::NotApplicable {
            message: "this treatment multiplier is incompatible with the benchmark covariate; \
                      next attempt: a smaller k_D",
        });
    }
    let eta = (ky.sqrt() + r2zxj.sqrt()) / (1.0 - r2zxj).sqrt();
    let r2yz = (eta * eta * r2yxj / (1.0 - r2yxj)).min(1.0);
    Ok(BenchmarkBound { kd, ky, r2dz, r2yz })
}

/// Linear adjustment with per-covariate benchmarks, computed from data.
#[derive(Clone, Debug, PartialEq)]
pub struct LinearAdjustmentFit {
    /// Treatment coefficient summary.
    pub summary: LinearAdjustmentSummary,
    /// `R²_{D~X_j|X_{−j}}` per covariate, in input order.
    pub r2_treatment: Vec<f64>,
    /// `R²_{Y~X_j|D,X_{−j}}` per covariate, in input order.
    pub r2_outcome: Vec<f64>,
}

/// Fit `Y ~ 1 + D + X` and the partial R²s of each covariate.
///
/// Partial R²s come from the residual sums of squares of the full and the leave-one-out
/// regressions, `R² = 1 − RSS_full / RSS_reduced`; the treatment standard error from
/// `t² = dof · (RSS_without_D − RSS_full) / RSS_full`.
///
/// # Errors
///
/// Mismatched lengths, too few rows, non-finite values, a rank-deficient design or a
/// treatment that explains nothing.
pub fn linear_adjustment_fit(
    outcome: &[f64],
    treatment: &[f64],
    covariates: &[Vec<f64>],
) -> Result<LinearAdjustmentFit, ValidationError> {
    let n = outcome.len();
    if treatment.len() != n || covariates.iter().any(|c| c.len() != n) {
        return Err(ValidationError::NotApplicable {
            message: "outcome, treatment and covariates must have the same length",
        });
    }
    let p = 2 + covariates.len();
    if n <= p + 1 {
        return Err(ValidationError::NotApplicable {
            message: "omitted-variable-bias analysis needs more rows than regression parameters \
                      plus one",
        });
    }
    if outcome.iter().chain(treatment).chain(covariates.iter().flatten()).any(|v| !v.is_finite()) {
        return Err(ValidationError::NotApplicable {
            message: "omitted-variable-bias analysis needs finite values",
        });
    }
    let ones = vec![1.0; n];
    let mut ws = LeastSquaresWorkspace::default();
    let mut fit = |columns: &[&[f64]], y: &[f64]| -> Result<(Vec<f64>, f64), ValidationError> {
        let mut design = Vec::with_capacity(n * columns.len());
        for column in columns {
            design.extend_from_slice(column);
        }
        let result = FaerBackend
            .least_squares(&design, n, columns.len(), y, &mut ws)
            .map_err(ValidationError::from)?;
        if result.rank < columns.len() {
            return Err(ValidationError::NotApplicable {
                message: "omitted-variable-bias analysis needs a full-rank design",
            });
        }
        Ok((result.coefficients, result.rss))
    };
    let mut full_columns: Vec<&[f64]> = vec![ones.as_slice(), treatment];
    full_columns.extend(covariates.iter().map(Vec::as_slice));
    let (coefficients, rss_full) = fit(&full_columns, outcome)?;
    let mut without_d: Vec<&[f64]> = vec![ones.as_slice()];
    without_d.extend(covariates.iter().map(Vec::as_slice));
    let (_, rss_without_d) = fit(&without_d, outcome)?;
    let dof = (n - p) as f64;
    if !(rss_full > 0.0 && rss_without_d > rss_full) {
        return Err(ValidationError::NotApplicable {
            message: "the treatment explains no residual outcome variance, or the outcome is \
                      fitted exactly",
        });
    }
    let t2 = dof * (rss_without_d - rss_full) / rss_full;
    let estimate = coefficients[1];
    let summary = LinearAdjustmentSummary::new(estimate, estimate.abs() / t2.sqrt(), dof)?;
    let mut r2_treatment = Vec::with_capacity(covariates.len());
    let mut r2_outcome = Vec::with_capacity(covariates.len());
    for j in 0..covariates.len() {
        let others: Vec<&[f64]> = covariates
            .iter()
            .enumerate()
            .filter(|(i, _)| *i != j)
            .map(|(_, c)| c.as_slice())
            .collect();
        let mut d_full: Vec<&[f64]> = vec![ones.as_slice()];
        d_full.extend(covariates.iter().map(Vec::as_slice));
        let mut d_reduced: Vec<&[f64]> = vec![ones.as_slice()];
        d_reduced.extend(others.iter().copied());
        let (_, d_rss_full) = fit(&d_full, treatment)?;
        let (_, d_rss_reduced) = fit(&d_reduced, treatment)?;
        r2_treatment.push((1.0 - d_rss_full / d_rss_reduced).max(0.0));
        let mut y_reduced: Vec<&[f64]> = vec![ones.as_slice(), treatment];
        y_reduced.extend(others.iter().copied());
        let (_, y_rss_reduced) = fit(&y_reduced, outcome)?;
        r2_outcome.push((1.0 - rss_full / y_rss_reduced).max(0.0));
    }
    Ok(LinearAdjustmentFit { summary, r2_treatment, r2_outcome })
}

/// Treatment partial R²s probed by the extreme-scenario curve.
const R2DZ_GRID: [f64; 9] = [0.0, 0.01, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5];

/// The omitted-variable-bias analysis of `summary` as an [`AssumptionRange`] on `target`.
///
/// The tested range is the extreme-scenario curve: a confounder explaining the share
/// `r2yz_extreme` of the residual outcome variance (one is the most extreme) and each probed
/// share of the residual treatment variance; at each the adjusted estimate and the adjusted
/// `1 − alpha` limit nearest zero are reported. The break-even values are `RV_q` (equal
/// partial R²s that reduce the estimate by `100·q`%) and `RV_{q,α}`.
///
/// # Errors
///
/// `q` not positive, `alpha` outside `(0, 1)` or `r2yz_extreme` outside `(0, 1]`.
pub fn ovb_range(
    target: RefutationTarget,
    summary: &LinearAdjustmentSummary,
    q: f64,
    alpha: f64,
    r2yz_extreme: f64,
) -> Result<AssumptionRange, ValidationError> {
    if !(q > 0.0
        && q.is_finite()
        && alpha > 0.0
        && alpha < 1.0
        && r2yz_extreme > 0.0
        && r2yz_extreme <= 1.0)
    {
        return Err(ValidationError::NotApplicable {
            message: "q must be positive, alpha must lie in (0, 1) and the extreme outcome \
                      partial R2 in (0, 1]",
        });
    }
    let z = student_t_ppf(1.0 - alpha / 2.0, summary.dof - 1.0);
    let tested = R2DZ_GRID
        .iter()
        .map(|&r2dz| {
            // An outcome partial R² of one leaves no residual variance; the adjusted standard
            // error is then zero and the limit equals the adjusted estimate.
            let estimate = summary.adjusted_estimate(r2dz, r2yz_extreme);
            let se = summary.adjusted_standard_error(r2dz, r2yz_extreme);
            let limit = if summary.estimate >= 0.0 { estimate - z * se } else { estimate + z * se };
            RangePoint { parameter: r2dz, target: estimate, test: Some(limit) }
        })
        .collect();
    Ok(AssumptionRange::new(
        target,
        SensitivityAssumption::UnmeasuredConfoundingPartialR2,
        "partial_r2",
        0.0,
        tested,
        Some(summary.robustness_value(q, None)),
        Some(summary.robustness_value(q, Some(alpha))),
    ))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Cinelli and Hazlett (2020), Darfur example: the effect of being directly harmed on
    /// peace attitudes, estimate 0.0973 (se 0.0233) on 783 degrees of freedom.
    fn darfur() -> LinearAdjustmentSummary {
        LinearAdjustmentSummary::new(0.097_315_82, 0.023_256_54, 783.0).unwrap()
    }

    #[test]
    fn darfur_partial_r2_and_robustness_values() {
        let s = darfur();
        assert!((s.partial_r2_treatment() - 0.0219).abs() < 5e-5, "{}", s.partial_r2_treatment());
        assert!((s.robustness_value(1.0, None) - 0.1388).abs() < 5e-5);
        assert!((s.robustness_value(1.0, Some(0.05)) - 0.0763).abs() < 5e-5);
    }

    #[test]
    fn darfur_bound_of_a_confounder_like_female() {
        // The published 1x female bound R2dz = 0.0092, R2yz = 0.1246 gives adjusted estimate
        // 0.0752 and adjusted standard error 0.0219.
        let s = darfur();
        assert!((s.adjusted_estimate(0.0092, 0.1246) - 0.0752).abs() < 5e-5);
        assert!((s.adjusted_standard_error(0.0092, 0.1246) - 0.0219).abs() < 5e-5);
    }

    #[test]
    fn robustness_value_reduces_the_estimate_by_q() {
        let s = darfur();
        for q in [0.5, 1.0, 2.0] {
            let rv = s.robustness_value(q, None);
            let adjusted = s.adjusted_estimate(rv, rv);
            assert!((adjusted - (1.0 - q) * s.estimate).abs() < 1e-10, "q {q}: {adjusted}");
        }
    }

    #[test]
    fn extreme_confounder_needs_the_treatment_partial_r2() {
        let s = darfur();
        assert!((s.break_even_r2dz(1.0, 1.0) - s.partial_r2_treatment()).abs() < 1e-12);
        let r2dz = s.break_even_r2dz(0.5, 1.0);
        assert!(s.adjusted_estimate(r2dz, 0.5).abs() < 1e-12);
    }

    #[test]
    fn the_significance_robustness_value_is_continuous_at_the_boundary() {
        // At f_q = 1/f*, both branches equal 1 - f*^2.
        let fcrit: f64 = 0.3;
        let fq = 1.0 / fcrit;
        let interior = rv_from_f(fq - fcrit);
        let boundary = (fq * fq - fcrit * fcrit) / (1.0 + fq * fq);
        assert!((interior - boundary).abs() < 1e-12);
        assert!((boundary - (1.0 - fcrit * fcrit)).abs() < 1e-12);
    }

    #[test]
    #[allow(clippy::float_cmp, reason = "a zero treatment benchmark maps to exactly zero")]
    fn benchmarks_scale_with_the_multipliers() {
        let b = benchmark_bound(0.0, 0.1, 1.0, 1.0).unwrap();
        assert_eq!(b.r2dz, 0.0);
        assert!((b.r2yz - 0.1 / 0.9).abs() < 1e-15);
        let b1 = benchmark_bound(0.01, 0.1, 1.0, 1.0).unwrap();
        let b3 = benchmark_bound(0.01, 0.1, 3.0, 3.0).unwrap();
        assert!((b3.r2dz - 3.0 * b1.r2dz).abs() < 1e-15);
        assert!(b3.r2yz > 2.9 * b1.r2yz);
        assert!(benchmark_bound(0.6, 0.1, 2.0, 1.0).is_err());
    }

    #[test]
    fn data_fit_recovers_the_closed_form_relations() {
        // Deterministic design: y = 1 + 2 d + x1 - 0.5 x2 + e with a structured e.
        let n = 60;
        let mut y = Vec::new();
        let mut d = Vec::new();
        let mut x1 = Vec::new();
        let mut x2 = Vec::new();
        for i in 0..n {
            let a = f64::from(i);
            let xi1 = (a * 0.37).sin();
            let xi2 = (a * 0.11).cos();
            let di = 0.5 * xi1 + (a * 0.71).sin();
            let e = 0.3 * (a * 1.3).sin();
            x1.push(xi1);
            x2.push(xi2);
            d.push(di);
            y.push(1.0 + 2.0 * di + xi1 - 0.5 * xi2 + e);
        }
        let fit = linear_adjustment_fit(&y, &d, &[x1, x2]).unwrap();
        let t = fit.summary.t_statistic();
        assert!((fit.summary.partial_r2_treatment() - t * t / (t * t + 56.0)).abs() < 1e-12);
        assert!(fit.r2_treatment[0] > fit.r2_treatment[1]);
        assert!(fit.r2_outcome.iter().all(|r| (0.0..1.0).contains(r)));
    }
}
