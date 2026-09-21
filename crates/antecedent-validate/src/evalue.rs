//! E-value sensitivity analysis (`VanderWeele` & Ding, 2017).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

use crate::common::{RefutationProblem, RefutationReport, complete_case_rows};
use crate::error::ValidationError;

/// Library-selected pass threshold of E ≥ 2, not a universal robustness criterion.
/// The original E-value formulation reports it as a
/// continuous diagnostic without a pass/fail gate; this library uses 2.0 so `ValidationSuite` verdicts are not
/// vacuous (the formula always yields E ≥ 1, so threshold 1.0 would pass every estimate,
/// including a true null).
pub const DEFAULT_EVALUE_THRESHOLD: f64 = 2.0;

/// Two-sided normal quantile of the 95% confidence interval the CI-limit E-value uses.
const CI_Z: f64 = 1.959_963_984_540_054;

/// E-value: the minimum strength of association, on the risk-ratio scale, that an
/// unmeasured confounder would need with both treatment and outcome to fully explain away
/// the observed effect, **and the same for the confidence-interval limit nearest the null**.
///
/// The reported value ([`RefutationReport::comparison`]) is the smaller of the two, so a
/// statistically null estimate (interval covering the null, E-value 1) cannot pass on the
/// strength of its point estimate alone.
///
/// The effect is put on the risk-ratio scale as follows.
///
/// - **Binary (0/1) outcome.** The estimate is a risk difference; the arm risks come from
///   the linear model's standardized means, `μ_c = ȳ + (ATE/Δ)(control − t̄)` and
///   `μ_a = μ_c + ATE`, and `RR = μ_a / μ_c`. Arm risks outside `(0, 1)` make the risk
///   ratio undefined and the check not applicable.
/// - **Continuous outcome.** The `VanderWeele`/Ding approximate conversion
///   `RR = exp(0.91 d)` of the standardized mean difference `d = ATE / σ`, where `σ` is the
///   *residual* SD of the outcome regression on treatment and the adjustment set (the
///   within-group SD the conversion is defined for), not the marginal SD of `Y` (which also
///   contains treatment- and covariate-explained variance and would halve `d` at
///   `R² = 0.75`).
///
/// then `E = RR + sqrt(RR (RR − 1))` (inverted first if `RR < 1`).
#[derive(Clone, Debug)]
pub struct EValue {
    /// Pass if the computed E-value is at least this large.
    ///
    /// Default [`DEFAULT_EVALUE_THRESHOLD`] (2.0) is a library convention. Override via
    /// [`EValue::with_threshold`] when a different convention is needed; the E-value itself
    /// is always reported in [`RefutationReport::comparison`] regardless of the gate.
    pub threshold: f64,
}

impl Default for EValue {
    fn default() -> Self {
        Self::new()
    }
}

/// Outcome sample the E-value is computed on.
struct EvalueSample {
    outcome: Vec<f64>,
    treatment: Vec<f64>,
    /// Adjustment covariates (column per entry), empty when none are addressable.
    covariates: Vec<Vec<f64>>,
}

impl EValue {
    /// Default threshold [`DEFAULT_EVALUE_THRESHOLD`] (2.0, a library convention).
    #[must_use]
    pub fn new() -> Self {
        Self { threshold: DEFAULT_EVALUE_THRESHOLD }
    }

    /// Explicit pass threshold (for report-only use, set `threshold = 0.0` or ignore
    /// `passed` and read [`RefutationReport::comparison`]).
    #[must_use]
    pub fn with_threshold(threshold: f64) -> Self {
        Self { threshold }
    }

    /// Compute the E-value for `problem.original.ate` and its confidence-interval limit.
    ///
    /// # Errors
    ///
    /// Too few complete rows for the outcome regression, or (binary outcome) a risk ratio that
    /// is undefined.
    pub fn refute(
        &self,
        problem: &RefutationProblem<'_>,
    ) -> Result<RefutationReport, ValidationError> {
        let sample = evalue_sample(problem)?;
        let ate = problem.original.ate;
        let (rr_point, rr_limit) = if is_binary(&sample.outcome) {
            binary_risk_ratios(problem, &sample, ate)?
        } else {
            continuous_risk_ratios(problem, &sample, ate)?
        };
        let e_point = e_value_from_risk_ratio(rr_point);
        let e_limit = rr_limit.map(e_value_from_risk_ratio);
        // `+∞` is a valid E-value (no finite confounder strength suffices, the limit for an outcome
        // with no residual variation); `NaN` is not.
        if e_point.is_nan() || e_limit.is_some_and(f64::is_nan) {
            return Err(ValidationError::estimation_msg("e-value is not a number"));
        }
        // No usable standard error: the interval-limit E-value cannot be computed, so the
        // point E-value is reported but cannot support a pass.
        let (e_value, passed, informative) = match e_limit {
            Some(limit) => {
                let e = e_point.min(limit);
                (e, e >= self.threshold, true)
            }
            None => (e_point, false, false),
        };
        Ok(RefutationReport {
            refuter: Arc::from("sensitivity.evalue"),
            original_ate: ate,
            refuted_ate: ate,
            comparison: e_value,
            informative,
            passed,
            failure_condition: if passed {
                None
            } else {
                Some(Arc::from(match e_limit {
                    Some(limit) => format!(
                        "e-value {e_value} below threshold {} (point estimate {e_point}, \
                         95% confidence-limit {limit})",
                        self.threshold
                    ),
                    None => format!(
                        "e-value of the point estimate is {e_point}, but the estimate has no \
                         usable standard error, so the confidence-limit e-value is undefined \
                         and cannot support a pass"
                    ),
                }))
            },
            replicates: 0,
        })
    }
}

/// Complete-case outcome / treatment / adjustment columns on the rows the estimate used.
fn evalue_sample(problem: &RefutationProblem<'_>) -> Result<EvalueSample, ValidationError> {
    if problem.effect_refit.is_some() {
        // A composed estimator has no single adjustment regression to residualize on.
        let (mask, _) =
            complete_case_rows(problem.data, &[problem.outcome(), problem.treatment()])?;
        return Ok(EvalueSample {
            outcome: problem.data.float64_masked(problem.outcome(), &mask)?,
            treatment: problem.data.float64_masked(problem.treatment(), &mask)?,
            covariates: Vec::new(),
        });
    }
    if problem.temporal.is_some() {
        let prep = crate::common::temporal_diagnostic_design(problem)?;
        let n = prep.design.nrows;
        let covariates = (0..prep.adjustment_set.len())
            .map(|i| prep.design.matrix[(i + 2) * n..(i + 3) * n].to_vec())
            .collect();
        return Ok(EvalueSample {
            outcome: prep.design.outcome.to_vec(),
            treatment: prep.treatment.to_vec(),
            covariates,
        });
    }
    let mut ids = vec![problem.treatment(), problem.outcome()];
    ids.extend_from_slice(&problem.estimand.adjustment_set);
    let (mask, _) = complete_case_rows(problem.data, &ids)?;
    let mut covariates = Vec::with_capacity(problem.estimand.adjustment_set.len());
    for &z in problem.estimand.adjustment_set.iter() {
        covariates.push(problem.data.float64_masked(z, &mask)?);
    }
    Ok(EvalueSample {
        outcome: problem.data.float64_masked(problem.outcome(), &mask)?,
        treatment: problem.data.float64_masked(problem.treatment(), &mask)?,
        covariates,
    })
}

#[allow(clippy::float_cmp)] // Exact 0/1 coding is the binary-outcome contract.
fn is_binary(outcome: &[f64]) -> bool {
    !outcome.is_empty() && outcome.iter().all(|&y| y == 0.0 || y == 1.0)
}

/// Standard error of the estimate: bootstrap when present, else analytic; `None` unless
/// finite and non-negative.
fn usable_se(problem: &RefutationProblem<'_>) -> Option<f64> {
    let se = problem.original.se_bootstrap.unwrap_or(problem.original.se_analytic);
    (se.is_finite() && se >= 0.0).then_some(se)
}

/// Confidence limit nearest the null for estimate `ate` and standard error `se`, or `None`
/// when the interval covers the null (whose E-value is 1 by definition).
fn nearest_limit(ate: f64, se: f64) -> Option<f64> {
    let half = CI_Z * se;
    let limit = if ate >= 0.0 { ate - half } else { ate + half };
    // The limit is on the wrong side of (or at) the null exactly when the interval covers it.
    (limit * ate > 0.0).then_some(limit)
}

/// `(point RR, RR of the nearest confidence limit)` for a continuous outcome.
///
/// The second entry is `Some(1.0)` when the interval covers the null and `None` when the
/// standard error is unusable.
fn continuous_risk_ratios(
    problem: &RefutationProblem<'_>,
    sample: &EvalueSample,
    ate: f64,
) -> Result<(f64, Option<f64>), ValidationError> {
    let Some(sd) = residual_sd(sample)? else {
        // The regression reproduces the outcome exactly: there is no within-group variation left
        // for an unmeasured confounder to act on, so any nonzero effect (and any nonzero
        // confidence limit) needs an unbounded confounder strength. The standardized difference
        // is infinite, and so is the E-value; a zero effect keeps `RR = 1`.
        let infinite_unless_zero = |effect: f64| if effect == 0.0 { 1.0 } else { f64::INFINITY };
        let rr_limit = usable_se(problem).map(|se| match nearest_limit(ate, se) {
            Some(limit) => infinite_unless_zero(limit),
            None => 1.0,
        });
        return Ok((infinite_unless_zero(ate), rr_limit));
    };
    let rr_point = smd_to_risk_ratio(ate / sd);
    let rr_limit = usable_se(problem).map(|se| match nearest_limit(ate, se) {
        Some(limit) => smd_to_risk_ratio(limit / sd),
        None => 1.0,
    });
    Ok((rr_point, rr_limit))
}

/// `VanderWeele`/Ding approximate conversion of a standardized mean difference to a risk ratio.
fn smd_to_risk_ratio(d: f64) -> f64 {
    (0.91 * d.abs()).exp()
}

/// Residual SD `sqrt(RSS / (n − p))` of the outcome regressed on `[1, T, Z…]`; `None` when it is
/// at rounding-error level relative to the outcome's own spread (the regression reproduces the
/// outcome exactly).
fn residual_sd(sample: &EvalueSample) -> Result<Option<f64>, ValidationError> {
    let n = sample.outcome.len();
    let p = 2 + sample.covariates.len();
    if n <= p {
        return Err(ValidationError::NotApplicable {
            message: "e-value requires more complete rows than outcome-regression parameters",
        });
    }
    let mut design = vec![1.0; n];
    design.extend_from_slice(&sample.treatment);
    for column in &sample.covariates {
        design.extend_from_slice(column);
    }
    let fit = FaerBackend
        .least_squares(&design, n, p, &sample.outcome, &mut LeastSquaresWorkspace::default())
        .map_err(ValidationError::from)?;
    #[allow(clippy::cast_precision_loss)]
    let sd = (fit.rss / (n - p) as f64).sqrt();
    // A residual SD at rounding-error level relative to the outcome's own spread is zero for
    // every purpose here: the ratio `ATE / sd` would be an artefact of floating point.
    let scale = crate::common::sample_sd(&sample.outcome);
    if !sd.is_finite() {
        return Err(ValidationError::NotApplicable {
            message: "e-value requires a finite residual standard deviation of the outcome",
        });
    }
    Ok((sd > 1e-9 * scale && sd > 0.0).then_some(sd))
}

/// `(point RR, RR of the nearest confidence limit)` for a binary outcome, from the arm risks
/// implied by the linear model's standardized means.
fn binary_risk_ratios(
    problem: &RefutationProblem<'_>,
    sample: &EvalueSample,
    ate: f64,
) -> Result<(f64, Option<f64>), ValidationError> {
    let (_, control, delta) = antecedent_estimate::prepare::treatment_contrast(
        &problem.query.active,
        &problem.query.control,
    )?;
    #[allow(clippy::cast_precision_loss)]
    let n = sample.outcome.len() as f64;
    let mean_y = sample.outcome.iter().sum::<f64>() / n;
    let mean_t = sample.treatment.iter().sum::<f64>() / n;
    let risk_control = mean_y + (ate / delta) * (control - mean_t);
    let ratio = |shift: f64| -> Result<f64, ValidationError> {
        let risk_active = risk_control + shift;
        if risk_control > 0.0 && risk_control < 1.0 && risk_active > 0.0 && risk_active < 1.0 {
            Ok(risk_active / risk_control)
        } else {
            Err(ValidationError::NotApplicable {
                message: "e-value on a binary outcome requires both arm risks strictly inside \
                          (0, 1); the risk ratio is undefined",
            })
        }
    };
    let rr_point = ratio(ate)?;
    // The limit shifts the *contrast*, holding the control risk: the risk ratio at the nearest
    // confidence limit of the risk difference.
    let rr_limit = match usable_se(problem) {
        None => None,
        Some(se) => Some(match nearest_limit(ate, se) {
            Some(limit) => ratio(limit)?,
            None => 1.0,
        }),
    };
    Ok((rr_point, rr_limit))
}

fn e_value_from_risk_ratio(rr: f64) -> f64 {
    let rr = if rr >= 1.0 { rr } else { 1.0 / rr };
    rr + rr.sqrt() * (rr - 1.0).sqrt()
}

#[cfg(test)]
mod tests {
    use super::{CI_Z, e_value_from_risk_ratio, nearest_limit, smd_to_risk_ratio};

    #[test]
    fn large_finite_risk_ratio_does_not_overflow_intermediate_product() {
        let value = e_value_from_risk_ratio(1e200);
        assert!(value.is_finite());
        assert!((value / 2e200 - 1.0).abs() < 1e-12);
    }

    #[test]
    fn protective_and_harmful_risk_ratios_have_same_evalue() {
        assert!((e_value_from_risk_ratio(0.5) - e_value_from_risk_ratio(2.0)).abs() < 1e-12);
    }

    #[test]
    fn vanderweele_ding_risk_ratio_two_has_evalue_two_plus_root_two() {
        // E(RR = 2) = 2 + sqrt(2 * 1) = 3.4142...
        assert!((e_value_from_risk_ratio(2.0) - (2.0 + 2.0_f64.sqrt())).abs() < 1e-15);
    }

    #[test]
    fn standardized_difference_converts_with_the_091_exponent() {
        assert!((smd_to_risk_ratio(1.0) - 0.91_f64.exp()).abs() < 1e-15);
        assert!((smd_to_risk_ratio(-1.0) - smd_to_risk_ratio(1.0)).abs() < 1e-15);
    }

    #[test]
    fn nearest_limit_is_none_exactly_when_the_interval_covers_the_null() {
        // ate 2, se 1: lower limit 2 - 1.96 = 0.04 > 0, does not cover the null.
        let limit = nearest_limit(2.0, 1.0).unwrap();
        assert!((limit - (2.0 - CI_Z)).abs() < 1e-15);
        // ate 2, se 2: lower limit is negative, the interval covers the null.
        assert!(nearest_limit(2.0, 2.0).is_none());
        // Negative estimates use the upper limit.
        let limit = nearest_limit(-2.0, 1.0).unwrap();
        assert!((limit - (-2.0 + CI_Z)).abs() < 1e-15);
        assert!(nearest_limit(-2.0, 2.0).is_none());
    }
}
