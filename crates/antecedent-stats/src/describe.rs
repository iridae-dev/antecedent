//! Sample mean and standard-error helpers over a slice of per-unit scores.
//!
//! Both use the two-pass shape (`mean`, then `Σ(v − mean)²`), not a streaming
//! update, so values pinned against the call sites they replace do not move.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

/// Sample mean and the standard error of the mean, `sqrt(s² / n)` with
/// `s² = Σ(v − mean)² / (n − 1)`.
///
/// `None` when there are fewer than two values, any value is non-finite, or the
/// standard error overflows to a non-finite number. A standard error of exactly
/// zero (constant scores) is returned as `Some((mean, 0.0))`: callers that must
/// refuse zero variance (a studentization) test `se > 0.0` themselves, because
/// whether a degenerate sample is an error is the caller's policy, not this
/// helper's.
#[must_use]
pub fn mean_and_se(values: &[f64]) -> Option<(f64, f64)> {
    if values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let (mean, variance) = mean_and_sample_variance(values)?;
    let se = (variance / values.len() as f64).sqrt();
    se.is_finite().then_some((mean, se))
}

/// Mean and Bessel-corrected sample variance `Σ(v − mean)² / (n − 1)` by the two-pass formula;
/// `None` for fewer than two values.
///
/// This applies no finiteness policy: a non-finite input gives a non-finite result, so a caller
/// that must refuse it checks first (as [`mean_and_se`] does). The one owner of the sample
/// mean/variance arithmetic; population variance (denominator `n`) is
/// `divergence::mean_var`.
#[must_use]
pub fn mean_and_sample_variance(values: &[f64]) -> Option<(f64, f64)> {
    if values.len() < 2 {
        return None;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    Some((mean, variance))
}

/// Standard error of the mean of independent scores (an influence-function SE):
/// `sqrt(Σ(v − mean)² / (n − 1) / n)`.
///
/// `NaN` for fewer than two values; a `NaN` input gives a `NaN` standard error (a `NaN`
/// variance is never clamped to a confident `0.0`), and an infinite input gives `NaN` or
/// `∞`. The sum of squares is non-negative by construction, so no clamp is applied. See
/// [`mean_and_se`] for the refusing form that returns `None` instead.
#[must_use]
pub fn iid_se(values: &[f64]) -> f64 {
    let Some((_, var)) = mean_and_sample_variance(values) else {
        return f64::NAN;
    };
    (var / values.len() as f64).sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_and_se_matches_hand_value() {
        // Values 1, 2, 3, 4: mean 2.5, s² = 5/3, se = sqrt(5/12).
        let (mean, se) = mean_and_se(&[1.0, 2.0, 3.0, 4.0]).unwrap();
        assert!((mean - 2.5).abs() < 1e-15);
        assert!((se - (5.0_f64 / 12.0).sqrt()).abs() < 1e-15);
        assert!((iid_se(&[1.0, 2.0, 3.0, 4.0]) - se).abs() < 1e-15);
    }

    #[test]
    fn mean_and_se_refuses_short_and_non_finite_but_not_zero_variance() {
        assert!(mean_and_se(&[]).is_none());
        assert!(mean_and_se(&[1.0]).is_none());
        assert!(mean_and_se(&[1.0, f64::NAN]).is_none());
        assert!(mean_and_se(&[1.0, f64::INFINITY]).is_none());
        let (mean, se) = mean_and_se(&[2.0, 2.0, 2.0]).unwrap();
        assert!((mean - 2.0).abs() < f64::EPSILON && se.abs() < f64::MIN_POSITIVE);
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact comparison against exactly representable or bit-identical reference values"
    )]
    fn iid_se_propagates_nan_instead_of_reporting_zero() {
        assert!(iid_se(&[1.0, f64::NAN, 3.0]).is_nan());
        assert!(iid_se(&[f64::NAN, f64::NAN]).is_nan());
        assert!(!iid_se(&[1.0, f64::INFINITY, 3.0]).is_finite());
        // A genuinely constant sample is still an exact zero.
        assert_eq!(iid_se(&[2.0, 2.0, 2.0]), 0.0);
    }

    #[test]
    fn iid_se_is_nan_below_two_values() {
        assert!(iid_se(&[]).is_nan());
        assert!(iid_se(&[3.0]).is_nan());
    }
}
