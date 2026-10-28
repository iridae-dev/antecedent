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
    let (mean, scaled, scale) = mean_and_scaled_variance(values)?;
    let se = (scaled / values.len() as f64).sqrt() * scale;
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
    let (mean, scaled, scale) = mean_and_scaled_variance(values)?;
    Some((mean, scaled * scale * scale))
}

/// Mean, and the variance as `scaled · scale²` (so a standard error can take its square root
/// before the scale is applied and stay representable); `None` for fewer than two values.
fn mean_and_scaled_variance(values: &[f64]) -> Option<(f64, f64, f64)> {
    if values.len() < 2 {
        return None;
    }
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let (scaled, scale) = centered_variance(values, mean, n);
    Some((mean, scaled, scale))
}

/// Deviations beyond these magnitudes square to a subnormal or overflow, so the sum of squares
/// is formed on deviations divided by their largest magnitude instead.
const SCALED_BELOW: f64 = 1e-140;
const SCALED_ABOVE: f64 = 1e140;

/// `Σ(v − mean)² / (n − 1)` by the corrected two-pass formula
/// `(Σd² − (Σd)²/n) / (n − 1)`, `d = v − mean`.
///
/// The computed mean carries a rounding error `δ` (the sum of `n` values of magnitude `|mean|`
/// is rounded `n` times, and for two values one ulp apart the exact mean is not representable at
/// all); the plain sum of squares is then too large by `n δ²`, which is not small against the
/// variance once `|mean| / sd` approaches `1 / sqrt(n ε)`. The second term removes exactly that
/// excess. For samples whose mean is small against their spread `(Σd)²/n` is far below the last
/// bit of `Σd²` and the result is bit-identical to the plain two-pass sum. The result is
/// clamped at zero (the correction can only round to a tiny negative for a constant sample).
/// Returns the variance divided by `scale²` and the scale.
fn centered_variance(values: &[f64], mean: f64, n: f64) -> (f64, f64) {
    let widest = values.iter().fold(0.0_f64, |m, v| m.max((v - mean).abs()));
    // Scale to keep the squares normal; 1 (no scaling) in the ordinary range.
    let scale =
        if widest > 0.0 && (widest < SCALED_BELOW || widest > SCALED_ABOVE) { widest } else { 1.0 };
    let (mut squares, mut total) = (0.0_f64, 0.0_f64);
    for v in values {
        let d = (v - mean) / scale;
        squares += d * d;
        total += d;
    }
    // Not `f64::max(0.0)`: that would turn a NaN variance into a confident zero.
    let corrected = squares - total * total / n;
    let corrected = if corrected < 0.0 { 0.0 } else { corrected };
    (corrected / (n - 1.0), scale)
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
    let Some((_, scaled, scale)) = mean_and_scaled_variance(values) else {
        return f64::NAN;
    };
    (scaled / values.len() as f64).sqrt() * scale
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
