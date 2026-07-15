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

/// Unbiased sample standard deviation; `NaN` if fewer than 2 observations.
#[must_use]
pub fn sample_std(values: &[f64]) -> f64 {
    let n = values.len() as f64;
    if n < 2.0 {
        return f64::NAN;
    }
    let mean = values.iter().sum::<f64>() / n;
    let var = values
        .iter()
        .map(|v| {
            let d = v - mean;
            d * d
        })
        .sum::<f64>()
        / (n - 1.0);
    var.sqrt()
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
    let p = causal_kernels::erfc(z / std::f64::consts::SQRT_2);
    Ok(((ma - mb).abs(), p.clamp(0.0, 1.0)))
}

/// Classifier two-sample test via Mann–Whitney U on 1-D scores (AUC-style).
///
/// Unlike [`mean_diff_two_sample`], this is sensitive to stochastic dominance / shape
/// shifts, not only mean separation. Statistic is `|U / (n_a n_b) − 0.5|` (distance of
/// AUC from chance); p-value uses the normal approximation to U.
///
/// # Errors
///
/// Empty samples.
pub fn classifier_two_sample(a: &[f64], b: &[f64]) -> Result<(f64, f64), StatsError> {
    if a.is_empty() || b.is_empty() {
        return Err(StatsError::Shape {
            message: "classifier_two_sample requires non-empty samples",
        });
    }
    let na = a.len() as f64;
    let nb = b.len() as f64;
    // Rank all observations; average ranks for ties.
    let mut all: Vec<(f64, u8)> = Vec::with_capacity(a.len() + b.len());
    all.extend(a.iter().copied().map(|v| (v, 0)));
    all.extend(b.iter().copied().map(|v| (v, 1)));
    all.sort_by(|x, y| x.0.partial_cmp(&y.0).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; all.len()];
    let mut i = 0;
    while i < all.len() {
        let mut j = i + 1;
        while j < all.len() && all[j].0 == all[i].0 {
            j += 1;
        }
        let avg = (i + j + 1) as f64 / 2.0; // 1-based average rank
        for r in ranks.iter_mut().take(j).skip(i) {
            *r = avg;
        }
        i = j;
    }
    let mut rank_sum_a = 0.0;
    for (k, (_, lab)) in all.iter().enumerate() {
        if *lab == 0 {
            rank_sum_a += ranks[k];
        }
    }
    let u_a = rank_sum_a - na * (na + 1.0) / 2.0;
    let auc = u_a / (na * nb);
    let stat = (auc - 0.5).abs();
    let mu = na * nb / 2.0;
    let sigma = ((na * nb * (na + nb + 1.0)) / 12.0).sqrt().max(1e-12);
    let z = (u_a - mu).abs() / sigma;
    let p = causal_kernels::erfc(z / std::f64::consts::SQRT_2).clamp(0.0, 1.0);
    Ok((stat, p))
}

/// Likelihood-ratio style residual comparison via KL between residual Gaussians.
///
/// Statistic is the KL; the p-value is an asymptotic χ² survival
/// `P(χ²_{df=2} > 2 n KL)` with `n = min(|r₀|, |r₁|)` — a Wilks-style
/// approximation for a two-parameter (mean, variance) Gaussian residual model,
/// not an exact LR test under misspecification.
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
    let n = resid_baseline.len().min(resid_comparison.len()) as f64;
    let lr = (2.0 * n * kl).max(0.0);
    // χ²_2 survival via Q(1, lr/2); df=2 → a = df/2 = 1.
    let p = crate::special::gamma_q(1.0, lr * 0.5).clamp(0.0, 1.0);
    Ok((kl, p))
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

    #[test]
    fn classifier_two_sample_detects_shift() {
        let a: Vec<f64> = (0..40).map(|i| f64::from(i) * 0.01).collect();
        let b: Vec<f64> = (0..40).map(|i| f64::from(i) * 0.01 + 3.0).collect();
        let (stat, p) = classifier_two_sample(&a, &b).unwrap();
        assert!(stat > 0.4, "stat={stat}");
        assert!(p < 0.01, "p={p}");
    }

    #[test]
    fn residual_lr_identical_residuals_have_unit_p() {
        let r: Vec<f64> = (0..40).map(|i| (i as f64) * 0.01 - 0.2).collect();
        let (kl, p) = residual_likelihood_ratio(&r, &r).unwrap();
        assert!(kl.abs() < 1e-12, "kl={kl}");
        assert!((p - 1.0).abs() < 1e-9, "p={p}");
    }

    #[test]
    fn residual_lr_detects_scale_shift() {
        let a: Vec<f64> = (0..80).map(|i| (i as f64) * 0.01).collect();
        let b: Vec<f64> = a.iter().map(|x| x * 3.0).collect();
        let (kl, p) = residual_likelihood_ratio(&a, &b).unwrap();
        assert!(kl > 0.1, "kl={kl}");
        assert!(p < 0.01, "p={p}");
    }
}
