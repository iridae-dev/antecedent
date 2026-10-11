//! Sample mean and standard-error helpers over a slice of per-unit scores.
//!
//! Both use the two-pass shape (`mean`, then `Σ(v − mean)²`), not a streaming
//! update, so values pinned against the call sites they replace do not move.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

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
    let n = values.len() as f64;
    if values.len() < 2 || values.iter().any(|v| !v.is_finite()) {
        return None;
    }
    let mean = values.iter().sum::<f64>() / n;
    let variance = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    let se = (variance / n).sqrt();
    se.is_finite().then_some((mean, se))
}

/// Standard error of the mean of independent scores (an influence-function SE):
/// `sqrt(max(Σ(v − mean)² / (n − 1) / n, 0))`.
///
/// `NaN` for fewer than two values; non-finite inputs propagate to `NaN` or
/// `∞` rather than being refused (see [`mean_and_se`] for the refusing form).
#[must_use]
pub fn iid_se(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mean = values.iter().sum::<f64>() / nf;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (nf - 1.0);
    (var / nf).max(0.0).sqrt()
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
    fn iid_se_is_nan_below_two_values() {
        assert!(iid_se(&[]).is_nan());
        assert!(iid_se(&[3.0]).is_nan());
    }
}
