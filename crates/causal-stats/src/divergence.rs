//! Divergence and two-sample helpers for mechanism-change detection.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use crate::error::StatsError;

/// Gaussian KL divergence `KL(N(μ0,σ0²) ‖ N(μ1,σ1²))`.
///
/// # Errors
///
/// Non-positive variances.
pub fn gaussian_kl(mu0: f64, var0: f64, mu1: f64, var1: f64) -> Result<f64, StatsError> {
    if var0 <= 0.0 || var1 <= 0.0 {
        return Err(StatsError::Shape { message: "gaussian_kl requires positive variances" });
    }
    Ok(0.5 * ((var1 / var0).ln() + (var0 + (mu0 - mu1).powi(2)) / var1 - 1.0))
}

/// Mean and variance of a slice.
#[must_use]
pub fn mean_var(xs: &[f64]) -> (f64, f64) {
    let n = xs.len().max(1) as f64;
    let mean = xs.iter().sum::<f64>() / n;
    let var = xs.iter().map(|x| (x - mean).powi(2)).sum::<f64>() / n;
    (mean, var)
}

/// Two-sample mean-difference statistic `|mean(a) − mean(b)|` with a pooled-SE
/// z-test p-value approximation (normal).
///
/// Returns `(statistic, p_value)`.
///
/// # Errors
///
/// Empty samples.
pub fn mean_diff_two_sample(a: &[f64], b: &[f64]) -> Result<(f64, f64), StatsError> {
    if a.is_empty() || b.is_empty() {
        return Err(StatsError::Shape {
            message: "mean_diff_two_sample requires non-empty samples",
        });
    }
    let (ma, va) = mean_var(a);
    let (mb, vb) = mean_var(b);
    let se = (va / a.len() as f64 + vb / b.len() as f64).sqrt().max(1e-12);
    let z = (ma - mb).abs() / se;
    let p = erfc_hastings(z / std::f64::consts::SQRT_2);
    Ok(((ma - mb).abs(), p.clamp(0.0, 1.0)))
}

/// Classifier two-sample proxy (1-D score separation); see [`mean_diff_two_sample`].
///
/// # Errors
///
/// Empty samples.
pub fn classifier_two_sample(a: &[f64], b: &[f64]) -> Result<(f64, f64), StatsError> {
    mean_diff_two_sample(a, b)
}

/// Likelihood-ratio style residual comparison via KL between residual Gaussians.
///
/// # Errors
///
/// Empty residuals or non-positive variance.
pub fn residual_likelihood_ratio(
    resid_baseline: &[f64],
    resid_comparison: &[f64],
) -> Result<(f64, f64), StatsError> {
    if resid_baseline.is_empty() || resid_comparison.is_empty() {
        return Err(StatsError::Shape {
            message: "residual_likelihood_ratio requires non-empty residuals",
        });
    }
    let (m0, v0) = mean_var(resid_baseline);
    let (m1, v1) = mean_var(resid_comparison);
    let v0 = v0.max(1e-12);
    let v1 = v1.max(1e-12);
    let kl = gaussian_kl(m0, v0, m1, v1)?;
    let p = erfc_hastings((2.0 * kl).sqrt() / std::f64::consts::SQRT_2).clamp(0.0, 1.0);
    Ok((kl, p))
}

/// Complementary error function (Hastings approximation).
fn erfc_hastings(x: f64) -> f64 {
    let z = x.abs();
    let t = 1.0 / (1.0 + 0.327_591_1 * z);
    let a1 = 0.254_829_592;
    let a2 = -0.284_496_736;
    let a3 = 1.421_413_741;
    let a4 = -1.453_152_027;
    let a5 = 1.061_405_429;
    let erf_c = (-z * z).exp() * (((((a5 * t + a4) * t + a3) * t + a2) * t + a1) * t);
    if x >= 0.0 { erf_c } else { 2.0 - erf_c }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn mean_diff_detects_shift() {
        let a: Vec<f64> = (0..50).map(|i| f64::from(i) * 0.01).collect();
        let b: Vec<f64> = (0..50).map(|i| f64::from(i) * 0.01 + 5.0).collect();
        let (stat, p) = mean_diff_two_sample(&a, &b).unwrap();
        assert!(stat > 4.0);
        assert!(p < 0.01);
    }

    #[test]
    fn gaussian_kl_zero_for_same() {
        assert!(gaussian_kl(0.0, 1.0, 0.0, 1.0).unwrap().abs() < 1e-12);
    }

    #[test]
    fn gaussian_kl_unequal_variances() {
        // KL(N(0,1) ‖ N(0,2)) = ½[ln(2) + ½ − 1] ≈ 0.09657
        let kl = gaussian_kl(0.0, 1.0, 0.0, 2.0).unwrap();
        let expected = 0.5 * (2.0_f64.ln() + 0.5 - 1.0);
        assert!(kl >= 0.0);
        assert!((kl - expected).abs() < 1e-10);
    }

    #[test]
    fn gaussian_kl_non_negative() {
        let cases = [
            (0.0, 1.0, 0.0, 2.0),
            (1.0, 1.0, 0.0, 1.0),
            (-2.0, 0.5, 3.0, 4.0),
            (0.0, 4.0, 0.0, 0.25),
        ];
        for (mu0, var0, mu1, var1) in cases {
            let kl = gaussian_kl(mu0, var0, mu1, var1).unwrap();
            assert!(kl >= -1e-12, "KL({mu0},{var0}‖{mu1},{var1}) = {kl}");
        }
    }
}
