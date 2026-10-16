//! Small shared numeric primitives: logistic link, softmax, log-weight
//! normalisation, median and midranks.
//!
//! Each function is the single workspace owner of its operation, with the
//! failure semantics documented on the function. They are deliberately free of
//! sanitising: callers that treat non-finite input as "no mass" or "ties" say so
//! at the call site.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::posterior_reduce::quantile_type7_sorted;
use crate::view::ViewError;

/// Logistic function `1 / (1 + e^{-x})`, branching on the sign of `x` so the exponential
/// argument is never positive: `e^{x} / (1 + e^{x})` for `x < 0`, `1 / (1 + e^{-x})`
/// otherwise.
///
/// Never overflows, saturates to exactly `0.0` / `1.0` at `∓∞`, keeps the full relative
/// accuracy of tiny probabilities for very negative `x` (down to the subnormals) and
/// propagates `NaN`. Callers that need a probability bounded away from `{0, 1}` apply their
/// own explicit clip to the result.
#[inline]
#[must_use]
pub fn sigmoid(x: f64) -> f64 {
    if x >= 0.0 {
        1.0 / (1.0 + (-x).exp())
    } else {
        let e = x.exp();
        e / (1.0 + e)
    }
}

/// Arithmetic mean `Σx / n`; `NaN` for an empty slice, and any `NaN` or infinite value
/// propagates (no sanitising). The single plain-mean owner; the masked and weighted forms are
/// in [`crate::scalar`].
#[inline]
#[must_use]
pub fn mean(values: &[f64]) -> f64 {
    if values.is_empty() {
        return f64::NAN;
    }
    values.iter().sum::<f64>() / values.len() as f64
}

/// Dot product `Σ aᵢ bᵢ`, summed left to right with no sanitising (a `NaN` propagates).
///
/// # Panics
///
/// Panics in debug builds if the lengths differ; in release the common prefix is used.
#[inline]
#[must_use]
pub fn dot(a: &[f64], b: &[f64]) -> f64 {
    debug_assert_eq!(a.len(), b.len(), "dot length mismatch");
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}

/// Unbiased sample standard deviation `sqrt(Σ(x − x̄)² / (n − 1))`; `NaN` for fewer than two
/// values, and a `NaN` value gives `NaN`.
#[must_use]
pub fn sample_std(values: &[f64]) -> f64 {
    let n = values.len();
    if n < 2 {
        return f64::NAN;
    }
    let nf = n as f64;
    let mean = values.iter().sum::<f64>() / nf;
    let ss: f64 = values
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum();
    (ss / (nf - 1.0)).sqrt()
}

/// `softplus(x) = ln(1 + e^x)` without overflow: `x + ln_1p(e^{-x})` for `x > 0`,
/// `ln_1p(e^{x})` otherwise. It is the negative log of [`sigmoid`]`(-x)`, so
/// `-ln σ(η)` is `softplus(-η)`.
#[inline]
#[must_use]
pub fn softplus(x: f64) -> f64 {
    if x > 0.0 { x + (-x).exp().ln_1p() } else { x.exp().ln_1p() }
}

/// Max-shifted softmax of the logits in `values`, written back in place.
///
/// The exponentials are taken against the maximum so they cannot overflow. An empty slice is
/// left untouched. Logits may be `-∞` (probability `0`); if no logit is finite, or any is
/// `NaN`, or the maximum is `+∞`, the output is `NaN` throughout, never a silent uniform.
pub fn softmax_in_place(values: &mut [f64]) {
    if values.is_empty() {
        return;
    }
    let max = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut sum = 0.0;
    for v in values.iter_mut() {
        *v = (*v - max).exp();
        sum += *v;
    }
    let inv = 1.0 / sum;
    for v in values.iter_mut() {
        *v *= inv;
    }
}

/// Why [`normalize_log_weights_into`] refused a log-weight vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LogWeightError {
    /// The vector is empty.
    Empty,
    /// A log-weight is `NaN` or `+∞`: an impossible weight, not a zero one.
    Invalid,
    /// Every log-weight is `-∞`: the total mass is zero.
    NoMass,
}

impl core::fmt::Display for LogWeightError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        f.write_str(match self {
            Self::Empty => "no log-weights to normalize",
            Self::Invalid => "log-weights contain NaN or +infinity",
            Self::NoMass => "log-weights carry no finite mass",
        })
    }
}

impl std::error::Error for LogWeightError {}

/// Self-normalise log-weights into `out` (cleared, then one weight per input, summing to 1).
///
/// The single owner of log-weight normalisation, with the `NaN`-rejecting default:
///
/// * `-∞` is a legitimate zero weight and is kept as `0.0`;
/// * `NaN` or `+∞` is [`LogWeightError::Invalid`], never silently dropped;
/// * an empty input is [`LogWeightError::Empty`];
/// * all `-∞` is [`LogWeightError::NoMass`].
///
/// On error `out` is left empty.
///
/// # Errors
///
/// See [`LogWeightError`].
pub fn normalize_log_weights_into(log_w: &[f64], out: &mut Vec<f64>) -> Result<(), LogWeightError> {
    out.clear();
    if log_w.is_empty() {
        return Err(LogWeightError::Empty);
    }
    let mut max = f64::NEG_INFINITY;
    for &lw in log_w {
        if lw.is_nan() || lw == f64::INFINITY {
            return Err(LogWeightError::Invalid);
        }
        max = max.max(lw);
    }
    if max == f64::NEG_INFINITY {
        return Err(LogWeightError::NoMass);
    }
    out.extend(log_w.iter().map(|lw| (lw - max).exp()));
    // The maximum contributes exactly 1, so the sum is at least 1.
    let sum: f64 = out.iter().sum();
    for w in out.iter_mut() {
        *w /= sum;
    }
    Ok(())
}

/// Allocating form of [`normalize_log_weights_into`].
///
/// # Errors
///
/// See [`LogWeightError`].
pub fn normalize_log_weights(log_w: &[f64]) -> Result<Vec<f64>, LogWeightError> {
    let mut out = Vec::with_capacity(log_w.len());
    normalize_log_weights_into(log_w, &mut out)?;
    Ok(out)
}

/// Median of an ascending-sorted sample: the type-7 quantile at `0.5`, so the even-`n` value
/// is `0.5·a + 0.5·b` of the two middle order statistics.
///
/// `NaN` for an empty sample or when the sorted sample carries a `NaN` at either end.
#[inline]
#[must_use]
pub fn median_sorted(sorted: &[f64]) -> f64 {
    quantile_type7_sorted(sorted, 0.5)
}

/// Midranks (1-based; ties share the average of the ranks they span) of `values` into `out`.
///
/// Ties are exact equality. Values are ordered by `f64::total_cmp`, so a `NaN` sorts last and
/// is never equal to anything (each `NaN` gets its own distinct rank); callers that cannot
/// tolerate `NaN` must reject it first.
///
/// On error `out` is left untouched.
///
/// # Errors
///
/// [`ViewError::InvalidShape`] when `out.len() != values.len()`.
#[allow(
    clippy::float_cmp,
    reason = "Rank ties are exact equality, independent of measurement units"
)]
pub fn midranks_into(values: &[f64], out: &mut [f64]) -> Result<(), ViewError> {
    if values.len() != out.len() {
        return Err(ViewError::InvalidShape { message: "midranks_into length mismatch" });
    }
    let n = values.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| values[i].total_cmp(&values[j]));
    let mut i = 0usize;
    while i < n {
        let mut j = i + 1;
        while j < n && values[idx[j]] == values[idx[i]] {
            j += 1;
        }
        // 1-based ranks i+1 ..= j average to (i + 1 + j) / 2.
        let average = (i + 1 + j) as f64 / 2.0;
        for &k in &idx[i..j] {
            out[k] = average;
        }
        i = j;
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact comparison against exactly representable or bit-identical reference values"
    )]
    fn sigmoid_is_stable_and_symmetric() {
        assert_eq!(sigmoid(0.0), 0.5);
        assert_eq!(sigmoid(f64::INFINITY), 1.0);
        assert_eq!(sigmoid(f64::NEG_INFINITY), 0.0);
        assert_eq!(sigmoid(800.0), 1.0);
        assert!(sigmoid(f64::NAN).is_nan());
        // Reference: 1/(1+e^-2) = 0.8807970779778823 (mpmath).
        assert!((sigmoid(2.0) - 0.880_797_077_977_882_3).abs() < 2e-16);
        // Tiny probabilities keep relative accuracy: e^-700 / (1 + e^-700).
        let tiny = sigmoid(-700.0);
        assert!((tiny / (-700.0_f64).exp() - 1.0).abs() < 1e-15);
        for x in [0.3, 1.0, 5.0, 30.0] {
            assert!((sigmoid(x) + sigmoid(-x) - 1.0).abs() < 4e-16, "x={x}");
        }
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact comparison against exactly representable or bit-identical reference values"
    )]
    fn mean_dot_and_sample_std_contracts() {
        assert!(mean(&[]).is_nan());
        assert_eq!(mean(&[1.0, 2.0, 6.0]), 3.0);
        assert!(mean(&[1.0, f64::NAN]).is_nan());
        assert_eq!(dot(&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]), 32.0);
        assert_eq!(dot(&[], &[]), 0.0);
        assert!(dot(&[f64::NAN], &[0.0]).is_nan());
        // 1, 2, 3, 4: s^2 = 5/3.
        assert!((sample_std(&[1.0, 2.0, 3.0, 4.0]) - (5.0_f64 / 3.0).sqrt()).abs() < 1e-15);
        assert!(sample_std(&[1.0]).is_nan());
        assert!(sample_std(&[1.0, f64::NAN]).is_nan());
        assert_eq!(sample_std(&[2.0, 2.0, 2.0]), 0.0);
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact comparison against exactly representable or bit-identical reference values"
    )]
    fn softplus_is_stable() {
        assert!((softplus(0.0) - std::f64::consts::LN_2).abs() < 1e-16);
        assert_eq!(softplus(1000.0), 1000.0);
        assert!((softplus(-800.0) - 0.0).abs() < f64::MIN_POSITIVE);
        // softplus(-x) = -ln sigmoid(x).
        for x in [-30.0, -2.0, 0.5, 8.0] {
            assert!((softplus(-x) + sigmoid(x).ln()).abs() < 1e-14, "x={x}");
        }
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact comparison against exactly representable or bit-identical reference values"
    )]
    fn softmax_in_place_boundaries() {
        let mut v = [1.0, 2.0, 3.0];
        softmax_in_place(&mut v);
        assert!((v.iter().sum::<f64>() - 1.0).abs() < 1e-15);
        assert!((v[2] / v[1] - 1.0_f64.exp()).abs() < 1e-14);
        // Huge logits do not overflow.
        let mut big = [1000.0, 1000.0];
        softmax_in_place(&mut big);
        assert_eq!(big, [0.5, 0.5]);
        // -inf is a zero probability.
        let mut z = [f64::NEG_INFINITY, 0.0];
        softmax_in_place(&mut z);
        assert_eq!(z, [0.0, 1.0]);
        // No finite logit, or a NaN, is NaN rather than uniform.
        let mut none = [f64::NEG_INFINITY, f64::NEG_INFINITY];
        softmax_in_place(&mut none);
        assert!(none.iter().all(|p| p.is_nan()));
        let mut nan = [0.0, f64::NAN];
        softmax_in_place(&mut nan);
        assert!(nan.iter().all(|p| p.is_nan()));
        let mut empty: [f64; 0] = [];
        softmax_in_place(&mut empty);
    }

    #[test]
    fn log_weight_normalisation_contract() {
        let w = normalize_log_weights(&[0.0, 0.0, f64::NEG_INFINITY]).unwrap();
        assert_eq!(w, vec![0.5, 0.5, 0.0]);
        // Shift invariance.
        let a = normalize_log_weights(&[-1000.0, -1001.0]).unwrap();
        let b = normalize_log_weights(&[0.0, -1.0]).unwrap();
        assert!((a[0] - b[0]).abs() < 1e-15 && (a[1] - b[1]).abs() < 1e-15);
        assert_eq!(normalize_log_weights(&[]), Err(LogWeightError::Empty));
        assert_eq!(normalize_log_weights(&[0.0, f64::NAN]), Err(LogWeightError::Invalid));
        assert_eq!(normalize_log_weights(&[f64::INFINITY, 0.0]), Err(LogWeightError::Invalid));
        assert_eq!(normalize_log_weights(&[f64::NEG_INFINITY; 3]), Err(LogWeightError::NoMass));
        let mut out = vec![9.0];
        assert!(normalize_log_weights_into(&[f64::NAN], &mut out).is_err());
        assert!(out.is_empty());
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "exact comparison against exactly representable or bit-identical reference values"
    )]
    fn median_and_midranks() {
        assert!(median_sorted(&[]).is_nan());
        assert_eq!(median_sorted(&[3.0]), 3.0);
        assert_eq!(median_sorted(&[1.0, 2.0, 9.0]), 2.0);
        assert_eq!(median_sorted(&[1.0, 2.0, 4.0, 9.0]), 3.0);
        let mut r = [0.0; 6];
        midranks_into(&[5.0, 1.0, 5.0, 3.0, 5.0, 1.0], &mut r).unwrap();
        // sorted: 1,1 -> 1.5 ; 3 -> 3 ; 5,5,5 -> 5
        assert_eq!(r, [5.0, 1.5, 5.0, 3.0, 5.0, 1.5]);
        let mut e: [f64; 0] = [];
        midranks_into(&[], &mut e).unwrap();
        // -0.0 and 0.0 are tied.
        let mut z = [0.0; 2];
        midranks_into(&[0.0, -0.0], &mut z).unwrap();
        assert_eq!(z, [1.5, 1.5]);
        let mut short = [0.0];
        assert_eq!(
            midranks_into(&[1.0, 2.0], &mut short).unwrap_err(),
            crate::view::ViewError::InvalidShape { message: "midranks_into length mismatch" }
        );
        assert_eq!(short, [0.0]);
    }
}
