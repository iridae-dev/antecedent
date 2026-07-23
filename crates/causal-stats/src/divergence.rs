//! Divergence and two-sample helpers for mechanism-change detection.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::unnecessary_wraps
)]

use causal_core::CausalRng;

use crate::ci::SignificanceMethod;
use crate::ci::nonparametric_permutation_count;
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

/// Two-sample mean-difference statistic `|mean(a) − mean(b)|` with a Welch-SE
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
    let (ma, _) = mean_var(a);
    let (mb, _) = mean_var(b);
    let sa = sample_std(a);
    let sb = sample_std(b);
    let va = if sa.is_finite() { sa * sa } else { 0.0 };
    let vb = if sb.is_finite() { sb * sb } else { 0.0 };
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
        while j < all.len() && all[j].0.to_bits() == all[i].0.to_bits() {
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
    // Tie correction: Σ(t³ − t) over tied groups of size t.
    let n = na + nb;
    let mut tie_sum = 0.0;
    let mut i = 0usize;
    while i < all.len() {
        let mut j = i + 1;
        while j < all.len() && all[j].0.to_bits() == all[i].0.to_bits() {
            j += 1;
        }
        let t = (j - i) as f64;
        if t > 1.0 {
            tie_sum += t * t * t - t;
        }
        i = j;
    }
    let var_u = (na * nb / 12.0) * ((n + 1.0) - tie_sum / (n * (n - 1.0).max(1.0)));
    let sigma = var_u.max(0.0).sqrt().max(1e-12);
    let z = (u_a - mu).abs() / sigma;
    let p = causal_kernels::erfc(z / std::f64::consts::SQRT_2).clamp(0.0, 1.0);
    Ok((stat, p))
}

/// Two-sample Gaussian likelihood-ratio test on residual segments.
///
/// Statistic is `n ln v̂₀ − n₀ ln v̂₀_seg − n₁ ln v̂₁_seg` (MLE variances),
/// asymptotically `χ²₂` under equal mean and variance (Wilks). Returns
/// `(lr_statistic, p_value)`.
///
/// # Errors
///
/// Empty residuals.
pub fn residual_likelihood_ratio(
    resid_baseline: &[f64],
    resid_comparison: &[f64],
) -> Result<(f64, f64), StatsError> {
    if resid_baseline.is_empty() || resid_comparison.is_empty() {
        return Err(StatsError::Shape {
            message: "residual_likelihood_ratio requires non-empty residuals",
        });
    }
    gaussian_segment_lr(resid_baseline, resid_comparison)
}

/// Biased MMD² with RBF kernel on 1-D samples (Gretton et al.).
///
/// Bandwidth uses the median pairwise-|diff| heuristic on the pooled sample
/// (`γ = 1 / (2 median²)`, floored away from zero). P-value is a permutation
/// null that reshuffles the pooled labels while keeping sample sizes fixed.
///
/// Returns `(mmd², p_value)`.
///
/// # Errors
///
/// Empty samples.
pub fn kernel_two_sample(a: &[f64], b: &[f64], rng_seed: u64) -> Result<(f64, f64), StatsError> {
    if a.is_empty() || b.is_empty() {
        return Err(StatsError::Shape { message: "kernel_two_sample requires non-empty samples" });
    }
    let n_perm = nonparametric_permutation_count(SignificanceMethod::Analytic);
    let gamma = rbf_gamma_median_heuristic(a, b);
    let observed = biased_mmd2(a, b, gamma);
    let mut pooled = Vec::with_capacity(a.len() + b.len());
    pooled.extend_from_slice(a);
    pooled.extend_from_slice(b);
    let na = a.len();
    let mut rng = CausalRng::from_seed(rng_seed);
    let mut ge = 0usize;
    for _ in 0..n_perm {
        fisher_yates_shuffle(&mut pooled, &mut rng);
        let (pa, pb) = pooled.split_at(na);
        let null_stat = biased_mmd2(pa, pb, gamma);
        if null_stat >= observed {
            ge += 1;
        }
    }
    // Add-one smoothing so p ∈ (0, 1].
    let p = ((ge + 1) as f64) / ((n_perm + 1) as f64);
    Ok((observed, p.clamp(0.0, 1.0)))
}

/// Known-split two-segment Gaussian change test on concatenated residuals.
///
/// `series[..split]` is the baseline regime and `series[split..]` the comparison.
/// Statistic is the Gaussian mean+variance likelihood-ratio
/// `n ln(σ₀²) − n₁ ln(σ₁²) − n₂ ln(σ₂²)` (up to additive constants), with a
/// χ²₂ asymptotic p-value.
///
/// Returns `(statistic, p_value)`.
///
/// # Errors
///
/// Empty series, `split` at an endpoint, or non-positive residual variance.
pub fn change_point_known_split(series: &[f64], split: usize) -> Result<(f64, f64), StatsError> {
    if series.len() < 4 || split == 0 || split >= series.len() {
        return Err(StatsError::Shape {
            message: "change_point_known_split requires len≥4 and interior split",
        });
    }
    let left = &series[..split];
    let right = &series[split..];
    if left.len() < 2 || right.len() < 2 {
        return Err(StatsError::Shape { message: "each regime needs ≥2 observations" });
    }
    let (stat, p) = gaussian_segment_lr(left, right)?;
    Ok((stat, p))
}

/// Convenience wrapper: concatenate baseline/comparison residuals and test at the join.
///
/// # Errors
///
/// See [`change_point_known_split`].
pub fn change_point_two_sample(a: &[f64], b: &[f64]) -> Result<(f64, f64), StatsError> {
    if a.is_empty() || b.is_empty() {
        return Err(StatsError::Shape {
            message: "change_point_two_sample requires non-empty samples",
        });
    }
    let mut series = Vec::with_capacity(a.len() + b.len());
    series.extend_from_slice(a);
    series.extend_from_slice(b);
    change_point_known_split(&series, a.len())
}

/// Max-|CUSUM| scan for an unknown change location in a single series.
///
/// Uses the standardized cumulative-sum statistic
/// `max_k |S_k|` with `S_k = Σᵢ₌₁ᵏ (xᵢ − x̄)`, and a permutation null under
/// exchangeability (reshuffles the series).
///
/// Returns `(max_|CUSUM|, p_value)`.
///
/// # Errors
///
/// Series shorter than 4.
pub fn change_point_scan(series: &[f64], rng_seed: u64) -> Result<(f64, f64), StatsError> {
    if series.len() < 4 {
        return Err(StatsError::Shape { message: "change_point_scan requires len≥4" });
    }
    let observed = max_abs_cusum(series);
    let n_perm = nonparametric_permutation_count(SignificanceMethod::Analytic);
    let mut buf = series.to_vec();
    let mut rng = CausalRng::from_seed(rng_seed);
    let mut ge = 0usize;
    for _ in 0..n_perm {
        fisher_yates_shuffle(&mut buf, &mut rng);
        if max_abs_cusum(&buf) >= observed {
            ge += 1;
        }
    }
    let p = ((ge + 1) as f64) / ((n_perm + 1) as f64);
    Ok((observed, p.clamp(0.0, 1.0)))
}

fn rbf_gamma_median_heuristic(a: &[f64], b: &[f64]) -> f64 {
    let mut diffs = Vec::with_capacity(a.len() * b.len().max(1));
    // Subsample pairwise |diffs| on pooled points for O(n²) with small n.
    let mut pooled = Vec::with_capacity(a.len() + b.len());
    pooled.extend_from_slice(a);
    pooled.extend_from_slice(b);
    let n = pooled.len();
    let step = ((n * n) / 2_000).max(1);
    let mut idx = 0usize;
    for i in 0..n {
        for j in (i + 1)..n {
            if idx % step == 0 {
                diffs.push((pooled[i] - pooled[j]).abs());
            }
            idx += 1;
        }
    }
    if diffs.is_empty() {
        return 1.0;
    }
    diffs.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let median = diffs[diffs.len() / 2].max(1e-8);
    1.0 / (2.0 * median * median)
}

fn biased_mmd2(a: &[f64], b: &[f64], gamma: f64) -> f64 {
    let na = a.len() as f64;
    let nb = b.len() as f64;
    let mut kxx = 0.0;
    for i in 0..a.len() {
        for j in 0..a.len() {
            kxx += rbf(a[i], a[j], gamma);
        }
    }
    let mut kyy = 0.0;
    for i in 0..b.len() {
        for j in 0..b.len() {
            kyy += rbf(b[i], b[j], gamma);
        }
    }
    let mut kxy = 0.0;
    for &x in a {
        for &y in b {
            kxy += rbf(x, y, gamma);
        }
    }
    kxx / (na * na) + kyy / (nb * nb) - 2.0 * kxy / (na * nb)
}

#[inline]
fn rbf(x: f64, y: f64, gamma: f64) -> f64 {
    let d = x - y;
    (-gamma * d * d).exp()
}

fn fisher_yates_shuffle(xs: &mut [f64], rng: &mut CausalRng) {
    causal_kernels::shuffle(rng, xs);
}

fn gaussian_segment_lr(left: &[f64], right: &[f64]) -> Result<(f64, f64), StatsError> {
    let n1 = left.len() as f64;
    let n2 = right.len() as f64;
    let n = n1 + n2;
    let sum: f64 = left.iter().chain(right.iter()).sum();
    let mean0 = sum / n;
    let sse0: f64 = left
        .iter()
        .chain(right.iter())
        .map(|x| {
            let d = x - mean0;
            d * d
        })
        .sum();
    let v0 = (sse0 / n).max(1e-12);
    let (_m1, v1) = mean_var(left);
    let (_m2, v2) = mean_var(right);
    let v1 = v1.max(1e-12);
    let v2 = v2.max(1e-12);
    // Gaussian mean+var change: 2(ℓ_alt−ℓ_null) = n ln v0 − n1 ln v1 − n2 ln v2 ~ χ²_2.
    let stat = (n * v0.ln() - n1 * v1.ln() - n2 * v2.ln()).max(0.0);
    let p = crate::special::gamma_q(1.0, stat * 0.5).clamp(0.0, 1.0);
    Ok((stat, p))
}

/// Max absolute CUSUM of demeaned `series` (interior points only).
#[must_use]
pub fn max_abs_cusum(series: &[f64]) -> f64 {
    let (mean, _) = mean_var(series);
    let mut s = 0.0;
    let mut max_abs: f64 = 0.0;
    // Exclude endpoints so a change must be interior.
    for (i, &x) in series.iter().enumerate() {
        s += x - mean;
        if i > 0 && i + 1 < series.len() {
            max_abs = max_abs.max(s.abs());
        }
    }
    max_abs
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
        let r: Vec<f64> = (0..40).map(|i| f64::from(i) * 0.01 - 0.2).collect();
        let (stat, p) = residual_likelihood_ratio(&r, &r).unwrap();
        assert!(stat.abs() < 1e-12, "stat={stat}");
        assert!((p - 1.0).abs() < 1e-9, "p={p}");
    }

    #[test]
    fn residual_lr_detects_scale_shift() {
        let a: Vec<f64> = (0..80).map(|i| f64::from(i) * 0.01).collect();
        let b: Vec<f64> = a.iter().map(|x| x * 3.0).collect();
        let (stat, p) = residual_likelihood_ratio(&a, &b).unwrap();
        assert!(stat > 0.5, "stat={stat}");
        assert!(p < 0.01, "p={p}");
    }

    fn lcg_noise(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((s >> 33) as f64) / ((1u64 << 31) as f64) - 0.5
            })
            .collect()
    }

    #[test]
    fn kernel_two_sample_detects_variance_shift() {
        // Same mean, different scale — mean_diff is weak; MMD should fire.
        let a = lcg_noise(60, 11);
        let b: Vec<f64> = lcg_noise(60, 22).into_iter().map(|x| x * 4.0).collect();
        let (stat, p) = kernel_two_sample(&a, &b, 0x_4E12_A001).unwrap();
        assert!(stat > 0.0, "stat={stat}");
        assert!(p < 0.05, "p={p}");
        let (_md, p_md) = mean_diff_two_sample(&a, &b).unwrap();
        // Document the regime: mean-diff may or may not fire; kernel must.
        let _ = p_md;
    }

    #[test]
    fn kernel_two_sample_null_not_tiny() {
        let a = lcg_noise(50, 1);
        let b = lcg_noise(50, 2);
        let (_stat, p) = kernel_two_sample(&a, &b, 0x_A011_0001).unwrap();
        assert!(p > 0.01, "null p should not be tiny: p={p}");
    }

    #[test]
    fn change_point_two_sample_detects_level_shift() {
        let a: Vec<f64> = (0..40).map(|i| f64::from(i) * 0.01).collect();
        let b: Vec<f64> = (0..40).map(|i| f64::from(i) * 0.01 + 5.0).collect();
        let (stat, p) = change_point_two_sample(&a, &b).unwrap();
        assert!(stat > 0.5, "stat={stat}");
        assert!(p < 0.01, "p={p}");
    }

    #[test]
    fn change_point_scan_detects_mid_series_shift() {
        let mut series: Vec<f64> = (0..80).map(|i| f64::from(i) * 0.01).collect();
        for v in &mut series[40..] {
            *v += 4.0;
        }
        let (stat, p) = change_point_scan(&series, 0x0C05_CA11).unwrap();
        assert!(stat > 10.0, "stat={stat}");
        assert!(p < 0.05, "p={p}");
    }
}
