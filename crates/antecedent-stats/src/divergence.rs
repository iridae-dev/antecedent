//! Divergence and two-sample helpers for mechanism-change detection.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::float_cmp, clippy::cast_possible_truncation, clippy::unnecessary_wraps)]

use antecedent_core::CausalRng;

use crate::error::StatsError;

/// Permutations used by the mechanism-change tests when the caller does not choose: the
/// smallest attainable p-value is `1 / (DEFAULT_MECHANISM_PERMUTATIONS + 1) = 0.001`, low
/// enough to survive Bonferroni / BH over dozens of targets.
pub const DEFAULT_MECHANISM_PERMUTATIONS: usize = 999;

/// Segment size below which the Gaussian likelihood-ratio p-value is calibrated by
/// permutation instead of the `χ²₂` asymptotics. At 5 rows per segment the asymptotic test
/// rejects a true null about 11% of the time at nominal 5%; at 30 it is about 5.6%.
const LR_ASYMPTOTIC_MIN_SEGMENT: usize = 30;

/// Outcome of a Monte-Carlo permutation test, carrying the resolution of its p-value.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PermutationTestResult {
    /// Test statistic on the observed data.
    pub statistic: f64,
    /// Add-one permutation p-value `(1 + #{null ≥ observed}) / (1 + n_permutations)`.
    pub p_value: f64,
    /// Permutations drawn.
    pub n_permutations: usize,
    /// Smallest attainable p-value, `1 / (1 + n_permutations)`. A p-value at this floor
    /// means "no permutation was as extreme", not "exactly zero".
    pub p_floor: f64,
}

impl PermutationTestResult {
    fn new(statistic: f64, exceed: usize, n_permutations: usize) -> Self {
        let denom = (n_permutations + 1) as f64;
        Self {
            statistic,
            p_value: (((exceed + 1) as f64) / denom).clamp(0.0, 1.0),
            n_permutations,
            p_floor: 1.0 / denom,
        }
    }
}

/// Gaussian KL divergence `KL(N(μ0,σ0²) ‖ N(μ1,σ1²))`.
///
/// # Errors
///
/// Nonfinite means or non-positive/nonfinite variances.
pub fn gaussian_kl(mu0: f64, var0: f64, mu1: f64, var1: f64) -> Result<f64, StatsError> {
    if !mu0.is_finite()
        || !mu1.is_finite()
        || !var0.is_finite()
        || !var1.is_finite()
        || var0 <= 0.0
        || var1 <= 0.0
    {
        return Err(StatsError::Shape {
            message: "gaussian_kl requires finite means and positive finite variances",
        });
    }
    let relative_change = (var0 - var1) / var1;
    let variance_term = if relative_change.abs() < 0.5 {
        // log1p avoids cancellation of the log ratio close to equal variances.
        relative_change - relative_change.ln_1p()
    } else {
        var0 / var1 - 1.0 + var1.ln() - var0.ln()
    };
    let standardized_shift = (mu0 - mu1) / var1.sqrt();
    Ok(0.5 * (variance_term + standardized_shift * standardized_shift))
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

/// Linear-interpolation empirical quantile (Hyndman–Fan type 7) of an ascending sample.
///
/// `p` is clamped to `[0, 1]`. `None` for an empty sample or a non-finite `p`, so
/// the caller decides what an unavailable quantile means; no substitute value is
/// invented and non-finite draws are never silently dropped.
#[must_use]
pub fn quantile_type7(sorted: &[f64], p: f64) -> Option<f64> {
    if sorted.is_empty() || !p.is_finite() {
        return None;
    }
    Some(crate::quantile::quantile_sorted(sorted, p, crate::quantile::QuantileRule::Interpolated))
}

/// Two-sample mean-difference statistic `|mean(a) − mean(b)|` with a Welch t-test
/// p-value (Satterthwaite degrees of freedom, two-sided).
///
/// The reference distribution is Student-t, not normal: with the two or three observations
/// per arm this function admits, a normal reference rejects a true null 12–19% of the time
/// at nominal 5%. Two exactly constant samples with different means have zero standard
/// error and return `p = 0`; identical constant samples return `p = 1`.
///
/// Returns `(statistic, p_value)`.
///
/// # Errors
///
/// Fewer than two observations in a sample, or non-finite observations.
pub fn mean_diff_two_sample(a: &[f64], b: &[f64]) -> Result<(f64, f64), StatsError> {
    if a.len() < 2 || b.len() < 2 {
        return Err(StatsError::Shape {
            message: "mean_diff_two_sample requires at least two observations per sample",
        });
    }
    if !a.iter().chain(b).all(|v| v.is_finite()) {
        return Err(StatsError::Shape {
            message: "mean_diff_two_sample requires finite observations",
        });
    }
    let (ma, _) = mean_var(a);
    let (mb, _) = mean_var(b);
    let sa = sample_std(a);
    let sb = sample_std(b);
    let va = sa * sa;
    let vb = sb * sb;
    let (na, nb) = (a.len() as f64, b.len() as f64);
    let (qa, qb) = (va / na, vb / nb);
    let se = (qa + qb).sqrt();
    let difference = (ma - mb).abs();
    let p = if se > 0.0 {
        let t = difference / se;
        // Welch–Satterthwaite: (qa + qb)² / (qa²/(na−1) + qb²/(nb−1)).
        let df = (qa + qb).powi(2) / (qa * qa / (na - 1.0) + qb * qb / (nb - 1.0));
        2.0 * crate::special::student_t_sf(t, df)
    } else if difference == 0.0 {
        1.0
    } else {
        0.0
    };
    Ok((difference, p.clamp(0.0, 1.0)))
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
    if !a.iter().chain(b).all(|v| v.is_finite()) {
        return Err(StatsError::Shape {
            message: "classifier_two_sample requires finite observations",
        });
    }
    let na = a.len() as f64;
    let nb = b.len() as f64;
    // Rank all observations; average ranks for ties.
    let mut all: Vec<(f64, u8)> = Vec::with_capacity(a.len() + b.len());
    all.extend(a.iter().copied().map(|v| (v, 0)));
    all.extend(b.iter().copied().map(|v| (v, 1)));
    all.sort_by(|x, y| x.0.partial_cmp(&y.0).expect("finite values are comparable"));
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
    // Tie correction: Σ(t³ − t) over tied groups of size t.
    let n = na + nb;
    let mut tie_sum = 0.0;
    let mut i = 0usize;
    while i < all.len() {
        let mut j = i + 1;
        while j < all.len() && all[j].0 == all[i].0 {
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
    let p = if z == 0.0 {
        1.0
    } else {
        antecedent_kernels::erfc(z / std::f64::consts::SQRT_2).clamp(0.0, 1.0)
    };
    Ok((stat, p))
}

/// Two-sample Gaussian likelihood-ratio test on residual segments.
///
/// Statistic is `n ln v̂₀ − n₀ ln v̂₀_seg − n₁ ln v̂₁_seg` (MLE variances),
/// asymptotically `χ²₂` under equal mean and variance (Wilks). Returns
/// `(lr_statistic, p_value)`. When both segments have at least 30 rows the p-value is the
/// `χ²₂` tail; below that the asymptotics are anti-conservative (type-I error 33% at 2 v 2,
/// 18% at 3 v 3, 11% at 5 v 5 for a nominal 5%), so the p-value is instead an exact
/// permutation p-value (999 label permutations of the pooled residuals, seeded from the
/// segments' order pattern so the result is reproducible and unit-free). The chi-square
/// calibration assumes positive
/// segment variances: constant pooled data return (0, 1) (nothing to compare), but a
/// segment that is itself exactly constant — including the single-row case, whose
/// sample variance is always exactly zero — is refused rather than reported. The
/// unrestricted Gaussian likelihood diverges as a segment's variance shrinks to zero, so
/// the naive "limiting ratio" is `(∞, p = 0)`: the strongest possible change claim from a
/// segment with no measured variation at all, which the Wilks asymptotics this p-value
/// relies on do not cover.
///
/// # Errors
///
/// Empty residuals, or a segment with fewer than two observations or exactly zero
/// variance.
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
/// (the smallest positive distance is used when ties make the median zero).
/// All-identical samples use bandwidth 1. P-value is a permutation
/// null that reshuffles the pooled labels while keeping sample sizes fixed.
///
/// Returns `(mmd², p_value)` using [`DEFAULT_MECHANISM_PERMUTATIONS`] permutations; see
/// [`kernel_two_sample_with_permutations`] for the permutation count and p-value floor.
///
/// # Errors
///
/// Empty or non-finite samples.
pub fn kernel_two_sample(a: &[f64], b: &[f64], rng_seed: u64) -> Result<(f64, f64), StatsError> {
    let r = kernel_two_sample_with_permutations(a, b, rng_seed, DEFAULT_MECHANISM_PERMUTATIONS)?;
    Ok((r.statistic, r.p_value))
}

/// Pooled sample size up to which the pooled RBF Gram matrix is held in memory (`N² · 8`
/// bytes, ≤ 128 MiB) so each permutation costs additions instead of `N²` exponentials.
const MMD_GRAM_CACHE_MAX_POOLED: usize = 4096;

/// [`kernel_two_sample`] with a caller-chosen permutation count, returning the p-value's
/// resolution.
///
/// The pooled RBF Gram matrix is computed once and each permutation only re-partitions
/// indices, so the cost is one `O(N²)` kernel evaluation plus `O(n_permutations · N²)`
/// additions (kernel values are recomputed per permutation above 4096 pooled rows).
///
/// # Errors
///
/// Empty or non-finite samples, or `n_permutations == 0`.
pub fn kernel_two_sample_with_permutations(
    a: &[f64],
    b: &[f64],
    rng_seed: u64,
    n_permutations: usize,
) -> Result<PermutationTestResult, StatsError> {
    if a.is_empty() || b.is_empty() {
        return Err(StatsError::Shape { message: "kernel_two_sample requires non-empty samples" });
    }
    if a.iter().chain(b).any(|v| !v.is_finite()) {
        return Err(StatsError::Shape {
            message: "kernel_two_sample requires finite observations",
        });
    }
    if n_permutations == 0 {
        return Err(StatsError::Shape {
            message: "permutation tests need at least one permutation",
        });
    }
    let n_perm = n_permutations;
    // Separate, salted stream from the permutation shuffle below so bandwidth selection
    // and the null distribution don't share draws.
    let mut bandwidth_rng = CausalRng::from_seed(rng_seed ^ 0xB4E5_1C7A_9D02_33F1);
    let bandwidth = rbf_bandwidth_median_heuristic(a, b, &mut bandwidth_rng);
    let mut pooled = Vec::with_capacity(a.len() + b.len());
    pooled.extend_from_slice(a);
    pooled.extend_from_slice(b);
    let na = a.len();
    let n = pooled.len();
    let gram = (n <= MMD_GRAM_CACHE_MAX_POOLED).then(|| pooled_rbf_gram(&pooled, bandwidth));
    // Observed and null statistics take the same route so their fp rounding agrees.
    let mut order: Vec<usize> = (0..n).collect();
    let mmd2 = |order: &[usize], values: &[f64]| match &gram {
        Some(k) => mmd2_from_gram(k, n, &order[..na], &order[na..]),
        None => biased_mmd2(&values[..na], &values[na..], bandwidth),
    };
    let observed = mmd2(&order, &pooled);
    let mut rng = CausalRng::from_seed(rng_seed);
    let mut ge = 0usize;
    for _ in 0..n_perm {
        if gram.is_some() {
            fisher_yates_shuffle_index(&mut order, &mut rng);
        } else {
            fisher_yates_shuffle(&mut pooled, &mut rng);
        }
        let null_stat = mmd2(&order, &pooled);
        if null_stat >= observed - MMD_TIE_TOLERANCE {
            ge += 1;
        }
    }
    Ok(PermutationTestResult::new(observed, ge, n_perm))
}

/// A permutation that reproduces the observed partition sums the same kernel values in a
/// different order; treat statistics equal to within rounding as ties, not exceedances.
const MMD_TIE_TOLERANCE: f64 = 1e-12;

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
/// Returns `(max_|CUSUM|, p_value)` using [`DEFAULT_MECHANISM_PERMUTATIONS`] permutations.
///
/// # Errors
///
/// Series shorter than 4.
pub fn change_point_scan(series: &[f64], rng_seed: u64) -> Result<(f64, f64), StatsError> {
    let r = change_point_scan_with_permutations(series, rng_seed, DEFAULT_MECHANISM_PERMUTATIONS)?;
    Ok((r.statistic, r.p_value))
}

/// [`change_point_scan`] with a caller-chosen permutation count, returning the p-value's
/// resolution.
///
/// # Errors
///
/// Series shorter than 4, or `n_permutations == 0`.
pub fn change_point_scan_with_permutations(
    series: &[f64],
    rng_seed: u64,
    n_permutations: usize,
) -> Result<PermutationTestResult, StatsError> {
    if series.len() < 4 {
        return Err(StatsError::Shape { message: "change_point_scan requires len≥4" });
    }
    if n_permutations == 0 {
        return Err(StatsError::Shape {
            message: "permutation tests need at least one permutation",
        });
    }
    let observed = max_abs_cusum(series);
    let n_perm = n_permutations;
    let mut buf = series.to_vec();
    let mut rng = CausalRng::from_seed(rng_seed);
    let mut ge = 0usize;
    for _ in 0..n_perm {
        fisher_yates_shuffle(&mut buf, &mut rng);
        if max_abs_cusum(&buf) >= observed - 1e-12 * (1.0 + observed) {
            ge += 1;
        }
    }
    Ok(PermutationTestResult::new(observed, ge, n_perm))
}

/// Pair count below which the exact median pairwise-|diff| is affordable to compute.
const MEDIAN_HEURISTIC_EXACT_PAIR_CAP: usize = 20_000;
/// Number of pairs drawn when the pool is too large to enumerate exactly.
const MEDIAN_HEURISTIC_SAMPLE_SIZE: usize = 2_000;

/// Bandwidth via the median pairwise-|diff| heuristic on the pooled sample.
///
/// Computes the exact median over all `C(n, 2)` pairs when that's affordable;
/// otherwise draws a seeded random subsample of pairs. The subsample must be
/// unbiased with respect to input ordering: a fixed stride over the row-major
/// `(i, j)` pair enumeration can alias with periodicity in structured / time-ordered
/// input — exactly what `kernel_two_sample` is documented to be used on — and bias the
/// estimated median away from the true one.
fn rbf_bandwidth_median_heuristic(a: &[f64], b: &[f64], rng: &mut CausalRng) -> f64 {
    let mut pooled = Vec::with_capacity(a.len() + b.len());
    pooled.extend_from_slice(a);
    pooled.extend_from_slice(b);
    let n = pooled.len();
    if n < 2 {
        return 1.0;
    }
    let total_pairs = n * (n - 1) / 2;
    let mut diffs = Vec::with_capacity(total_pairs.min(MEDIAN_HEURISTIC_SAMPLE_SIZE.max(1)));
    if total_pairs <= MEDIAN_HEURISTIC_EXACT_PAIR_CAP {
        for i in 0..n {
            for j in (i + 1)..n {
                diffs.push((pooled[i] - pooled[j]).abs());
            }
        }
    } else {
        for _ in 0..MEDIAN_HEURISTIC_SAMPLE_SIZE {
            let i = (rng.next_u64() as usize) % n;
            let mut j = (rng.next_u64() as usize) % n;
            while j == i {
                j = (rng.next_u64() as usize) % n;
            }
            diffs.push((pooled[i] - pooled[j]).abs());
        }
    }
    diffs.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
    let median = diffs[diffs.len() / 2];
    if median > 0.0 { median } else { diffs.into_iter().find(|d| *d > 0.0).unwrap_or(1.0) }
}

fn biased_mmd2(a: &[f64], b: &[f64], bandwidth: f64) -> f64 {
    let na = a.len() as f64;
    let nb = b.len() as f64;
    let mut kxx = 0.0;
    for i in 0..a.len() {
        for j in 0..a.len() {
            kxx += rbf(a[i], a[j], bandwidth);
        }
    }
    let mut kyy = 0.0;
    for i in 0..b.len() {
        for j in 0..b.len() {
            kyy += rbf(b[i], b[j], bandwidth);
        }
    }
    let mut kxy = 0.0;
    for &x in a {
        for &y in b {
            kxy += rbf(x, y, bandwidth);
        }
    }
    kxx / (na * na) + kyy / (nb * nb) - 2.0 * kxy / (na * nb)
}

/// Pooled RBF Gram matrix (row-major `n × n`).
fn pooled_rbf_gram(pooled: &[f64], bandwidth: f64) -> Vec<f64> {
    let n = pooled.len();
    let mut k = vec![0.0; n * n];
    for i in 0..n {
        k[i * n + i] = 1.0;
        for j in (i + 1)..n {
            let v = rbf(pooled[i], pooled[j], bandwidth);
            k[i * n + j] = v;
            k[j * n + i] = v;
        }
    }
    k
}

/// Biased MMD² for the index sets `ia` / `ib` of a precomputed pooled Gram matrix.
fn mmd2_from_gram(k: &[f64], n: usize, ia: &[usize], ib: &[usize]) -> f64 {
    let block_sum = |rows: &[usize], cols: &[usize]| -> f64 {
        let mut s = 0.0;
        for &i in rows {
            let row = &k[i * n..(i + 1) * n];
            for &j in cols {
                s += row[j];
            }
        }
        s
    };
    let (na, nb) = (ia.len() as f64, ib.len() as f64);
    block_sum(ia, ia) / (na * na) + block_sum(ib, ib) / (nb * nb)
        - 2.0 * block_sum(ia, ib) / (na * nb)
}

#[inline]
fn rbf(x: f64, y: f64, bandwidth: f64) -> f64 {
    let d = (x - y) / bandwidth;
    (-0.5 * d * d).exp()
}

fn fisher_yates_shuffle(xs: &mut [f64], rng: &mut CausalRng) {
    antecedent_kernels::shuffle(rng, xs);
}

fn fisher_yates_shuffle_index(xs: &mut [usize], rng: &mut CausalRng) {
    antecedent_kernels::shuffle(rng, xs);
}

/// Whether every observation is bit-identical to the first (a degenerate, zero-variance
/// segment). Checked directly on the raw values rather than by comparing a computed
/// sample variance to `0.0`: summing then dividing by `n` rounds when `n` does not divide
/// evenly, so even genuinely constant input (e.g. three copies of the same value) can
/// come back with a variance that is a tiny nonzero number instead of exact zero.
fn is_constant(xs: &[f64]) -> bool {
    match xs.split_first() {
        Some((first, rest)) => rest.iter().all(|v| v == first),
        None => true,
    }
}

fn gaussian_segment_lr(left: &[f64], right: &[f64]) -> Result<(f64, f64), StatsError> {
    let n1 = left.len() as f64;
    let n2 = right.len() as f64;
    let n = n1 + n2;
    if left.iter().chain(right).any(|v| !v.is_finite()) {
        return Err(StatsError::Shape {
            message: "Gaussian likelihood ratio requires finite observations",
        });
    }
    // A common rescaling cancels from the likelihood ratio, while preventing
    // squaring tiny/huge observations from underflowing/overflowing.
    let scale = left.iter().chain(right).fold(0.0_f64, |acc, v| acc.max(v.abs()));
    if scale == 0.0 {
        return Ok((0.0, 1.0));
    }
    let scaled_left: Vec<_> = left.iter().map(|v| v / scale).collect();
    let scaled_right: Vec<_> = right.iter().map(|v| v / scale).collect();
    let mean0 = scaled_left.iter().chain(&scaled_right).sum::<f64>() / n;
    let v0 = scaled_left.iter().chain(&scaled_right).map(|v| (v - mean0).powi(2)).sum::<f64>() / n;
    let (_, v1) = mean_var(&scaled_left);
    let (_, v2) = mean_var(&scaled_right);
    if v0 == 0.0 {
        return Ok((0.0, 1.0));
    }
    // A segment with fewer than two observations, or whose observations are all
    // identical, has zero sample variance. The unrestricted Gaussian likelihood is then
    // unbounded as its variance shrinks to zero, so "infinite evidence, p = 0" here is a
    // calibration artifact of a degenerate segment, not a finding — refuse it instead of
    // reporting a p-value the Wilks asymptotics were never meant to cover.
    if is_constant(left) || is_constant(right) {
        return Err(StatsError::Shape {
            message: "Gaussian likelihood ratio requires each segment to have measured \
                      variation; a single-row or exactly constant segment cannot support \
                      a calibrated likelihood-ratio p-value",
        });
    }
    // Gaussian mean+var change: 2(ℓ_alt−ℓ_null) = n ln v0 − n1 ln v1 − n2 ln v2 ~ χ²_2.
    let stat = (n * v0.ln() - n1 * v1.ln() - n2 * v2.ln()).max(0.0);
    if left.len().min(right.len()) >= LR_ASYMPTOTIC_MIN_SEGMENT {
        let p = crate::special::gamma_q(1.0, stat * 0.5).clamp(0.0, 1.0);
        return Ok((stat, p));
    }
    // Small segments: the Wilks χ²₂ is far too liberal, so calibrate by permuting the
    // segment labels of the pooled (scaled) residuals. Exact under exchangeability of the
    // two segments, valid at any size. A permuted segment that is constant has an infinite
    // statistic, which counts as at least as extreme as the observed one.
    let mut pooled: Vec<f64> = scaled_left.iter().chain(&scaled_right).copied().collect();
    let mut rng = CausalRng::from_seed(segment_order_seed(&pooled, left.len()));
    let mut exceed = 0usize;
    for _ in 0..DEFAULT_MECHANISM_PERMUTATIONS {
        antecedent_kernels::shuffle(&mut rng, &mut pooled);
        let (_, w1) = mean_var(&pooled[..left.len()]);
        let (_, w2) = mean_var(&pooled[left.len()..]);
        let null_stat = n * v0.ln() - n1 * w1.ln() - n2 * w2.ln();
        if null_stat >= stat - 1e-12 * (1.0 + stat) {
            exceed += 1;
        }
    }
    let p = PermutationTestResult::new(stat, exceed, DEFAULT_MECHANISM_PERMUTATIONS).p_value;
    Ok((stat, p))
}

/// Seed for the small-segment permutation null, derived from which segment each pooled
/// value belongs to in sorted order. That pattern is invariant to any positive rescaling
/// of the residuals, so the p-value stays unit-free, and it is fully determined by the
/// data, so repeated calls agree.
fn segment_order_seed(pooled: &[f64], n_left: usize) -> u64 {
    let mut idx: Vec<usize> = (0..pooled.len()).collect();
    idx.sort_by(|&i, &j| pooled[i].total_cmp(&pooled[j]).then(i.cmp(&j)));
    let mut h: u64 = 0xcbf2_9ce4_8422_2325 ^ (pooled.len() as u64) ^ ((n_left as u64) << 32);
    for i in idx {
        h ^= u64::from(i < n_left) + 1;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
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
    fn type7_quantile_matches_hand_interpolation_and_boundaries() {
        let s = [1.0, 2.0, 4.0, 8.0, 16.0];
        // h = 4p: p = 0.25 -> exactly index 1; p = 0.6 -> 2.4 -> 4 + 0.4*(8-4).
        assert_eq!(quantile_type7(&s, 0.25), Some(2.0));
        assert!((quantile_type7(&s, 0.6).unwrap() - 5.6).abs() < 1e-12);
        assert_eq!(quantile_type7(&s, 0.0), Some(1.0));
        assert_eq!(quantile_type7(&s, 1.0), Some(16.0));
        assert_eq!(quantile_type7(&s, -3.0), Some(1.0));
        assert_eq!(quantile_type7(&s, 7.0), Some(16.0));
        assert_eq!(quantile_type7(&[3.5], 0.9), Some(3.5));
    }

    #[test]
    fn type7_quantile_is_none_when_undefined() {
        assert_eq!(quantile_type7(&[], 0.5), None);
        assert_eq!(quantile_type7(&[1.0, 2.0], f64::NAN), None);
    }

    #[test]
    fn kernel_statistic_keeps_small_unit_distances_and_ties() {
        let a = [0.0, 1.0, 2.0, 3.0];
        let b = [2.0, 4.0, 6.0, 8.0];
        let mut rng = CausalRng::from_seed(1);
        let bandwidth = rbf_bandwidth_median_heuristic(&a, &b, &mut rng);
        let expected = biased_mmd2(&a, &b, bandwidth);
        for scale in [1e-200, 1e-20, 1e200] {
            let scaled_a = a.map(|v| v * scale);
            let scaled_b = b.map(|v| v * scale);
            let bandwidth = rbf_bandwidth_median_heuristic(&scaled_a, &scaled_b, &mut rng);
            let actual = biased_mmd2(&scaled_a, &scaled_b, bandwidth);
            assert!((actual - expected).abs() < 1e-14);
        }
        assert_eq!(kernel_two_sample(&[0.0; 2], &[0.0; 2], 1).unwrap(), (0.0, 1.0));
        assert!(kernel_two_sample(&[f64::NAN], &[0.0], 1).is_err());
    }

    #[test]
    fn gaussian_likelihood_ratio_is_invariant_to_extreme_units() {
        let left = [0.0, 1.0, 2.0, 3.0];
        let right = [2.0, 4.0, 6.0, 8.0];
        let expected = residual_likelihood_ratio(&left, &right).unwrap();
        for scale in [1e-200, 1e-20, 1e200] {
            let actual =
                residual_likelihood_ratio(&left.map(|v| v * scale), &right.map(|v| v * scale))
                    .unwrap();
            assert!((actual.0 - expected.0).abs() < 1e-12);
            assert!((actual.1 - expected.1).abs() < 1e-12);
        }
        assert_eq!(residual_likelihood_ratio(&[1.0; 2], &[1.0; 2]).unwrap(), (0.0, 1.0));
        // Two exactly-constant segments carry no within-segment variation at all; this
        // used to be pinned as the limiting ratio (∞, p = 0.0) — a "certain change" claim
        // manufactured from zero evidence. It is refused instead
        // (`residual_lr_refuses_single_row_or_constant_segments` below).
        assert!(residual_likelihood_ratio(&[1.0; 2], &[2.0; 2]).is_err());
        assert!(residual_likelihood_ratio(&[f64::NAN], &[1.0]).is_err());
    }

    #[test]
    fn gaussian_kl_stays_accurate_for_nearby_and_extreme_variances() {
        let close = gaussian_kl(0.0, 1.0 + 1e-8, 0.0, 1.0).unwrap();
        assert!((close - 2.5e-17).abs() < 1e-24, "{close:e}");
        let extreme = gaussian_kl(0.0, 1e-300, 0.0, 1e300).unwrap();
        assert!(extreme.is_finite());
        for bad in [f64::NAN, f64::INFINITY, -1.0, 0.0] {
            assert!(gaussian_kl(0.0, bad, 0.0, 1.0).is_err());
        }
    }

    #[test]
    fn mean_difference_test_is_invariant_to_small_units() {
        let a = [0.0, 1.0, 2.0, 3.0];
        let b = [1.0, 2.0, 3.0, 4.0];
        let small_a = a.map(|v| v * 1e-20);
        let small_b = b.map(|v| v * 1e-20);
        let (_, expected) = mean_diff_two_sample(&a, &b).unwrap();
        let (_, actual) = mean_diff_two_sample(&small_a, &small_b).unwrap();
        assert!((actual - expected).abs() < 1e-14);
        assert_eq!(mean_diff_two_sample(&[0.0; 2], &[0.0; 2]).unwrap().1, 1.0);
        assert_eq!(mean_diff_two_sample(&[0.0; 2], &[1e-20; 2]).unwrap().1, 0.0);
    }

    #[test]
    fn mean_diff_detects_shift() {
        let a: Vec<f64> = (0..50).map(|i| f64::from(i) * 0.01).collect();
        let b: Vec<f64> = (0..50).map(|i| f64::from(i) * 0.01 + 5.0).collect();
        let (stat, p) = mean_diff_two_sample(&a, &b).unwrap();
        assert!(stat > 4.0);
        assert!(p < 0.01);
    }

    #[test]
    fn mean_diff_rejects_samples_with_undefined_variance() {
        assert!(mean_diff_two_sample(&[0.0], &[1.0]).is_err());
        assert!(mean_diff_two_sample(&[0.0], &[1.0, 2.0]).is_err());
        assert!(mean_diff_two_sample(&[0.0, 1.0], &[2.0]).is_err());
    }

    #[test]
    fn mean_diff_is_a_welch_t_test_with_satterthwaite_df() {
        // Both samples have n = 2 and variance 2, so se² = 1 + 1 and the Satterthwaite df is
        // 2² / (1 + 1) = 2. A Student-t with 2 df has the closed-form two-sided tail
        // 1 − t / √(2 + t²); the old normal reference gave erfc(t/√2) instead.
        let (stat, p) = mean_diff_two_sample(&[0.0, 2.0], &[1.0, 3.0]).unwrap();
        assert!((stat - 1.0).abs() <= 1e-12);
        // t = 1/√2 ⇒ 1 − (1/√2)/√(2.5) = 1 − 1/√5.
        assert!((p - (1.0 - 1.0 / 5.0_f64.sqrt())).abs() <= 1e-9, "p={p}");

        // Larger shift: t = 10/√2, t² = 50 ⇒ p = 1 − √(50/52) = 1 − 5/√26 ≈ 0.0194,
        // against 1.5e-12 from the normal reference.
        let (_, p) = mean_diff_two_sample(&[0.0, 2.0], &[10.0, 12.0]).unwrap();
        assert!((p - (1.0 - 5.0 / 26.0_f64.sqrt())).abs() <= 1e-9, "p={p}");
    }

    #[test]
    fn residual_lr_small_segments_use_a_permutation_null() {
        // Pooled {0,1,2,3}, 2 v 2. The observed split {0,1 | 2,3} attains the maximal
        // statistic, which exactly two of the C(4,2) = 6 label assignments reach ({01|23}
        // and {23|01}) ⇒ exact permutation p = 2/6. The χ²₂ tail of the statistic
        // (4 ln 1.25 + 4 ln 4 ≈ 6.44) is exp(−3.22) ≈ 0.04, i.e. a false "change".
        let (stat, p) = residual_likelihood_ratio(&[0.0, 1.0], &[2.0, 3.0]).unwrap();
        assert!((stat - (4.0 * 1.25_f64.ln() + 4.0 * 4.0_f64.ln())).abs() < 1e-9, "stat={stat}");
        assert!((p - 1.0 / 3.0).abs() < 0.07, "p={p}");
        // Reproducible: the seed is derived from the data.
        assert_eq!(p, residual_likelihood_ratio(&[0.0, 1.0], &[2.0, 3.0]).unwrap().1);
        // Large segments keep the asymptotic tail.
        let a: Vec<f64> = (0..40).map(|i| f64::from(i) * 0.01).collect();
        let b: Vec<f64> = (0..40).map(|i| f64::from(i) * 0.03).collect();
        let (s, p) = residual_likelihood_ratio(&a, &b).unwrap();
        assert!((p - (-0.5 * s).exp()).abs() < 1e-9, "χ²₂ tail is exp(−stat/2): p={p} stat={s}");
    }

    #[test]
    fn permutation_tests_report_count_and_p_value_floor() {
        let a = lcg_noise(20, 1);
        let b: Vec<f64> = lcg_noise(20, 2).into_iter().map(|x| x + 10.0).collect();
        let r = kernel_two_sample_with_permutations(&a, &b, 7, 999).unwrap();
        assert_eq!(r.n_permutations, 999);
        assert!((r.p_floor - 0.001).abs() < 1e-15);
        // A gross shift beats every permutation: p sits exactly at the floor.
        assert!((r.p_value - 0.001).abs() < 1e-15, "p={}", r.p_value);
        let r = kernel_two_sample_with_permutations(&a, &b, 7, 199).unwrap();
        assert!((r.p_floor - 0.005).abs() < 1e-15 && (r.p_value - 0.005).abs() < 1e-15);
        assert!(kernel_two_sample_with_permutations(&a, &b, 7, 0).is_err());
        // The default is no longer the 49-permutation floor of 0.02.
        assert!(kernel_two_sample(&a, &b, 7).unwrap().1 < 0.0011);

        let mut series: Vec<f64> = (0..80).map(|i| f64::from(i) * 0.01).collect();
        for v in &mut series[40..] {
            *v += 4.0;
        }
        let r = change_point_scan_with_permutations(&series, 3, 999).unwrap();
        assert!((r.p_value - 0.001).abs() < 1e-15 && (r.p_floor - 0.001).abs() < 1e-15);
    }

    #[test]
    fn cached_gram_mmd_matches_direct_evaluation() {
        let pooled: Vec<f64> = lcg_noise(30, 5);
        let bandwidth = 0.3;
        let gram = pooled_rbf_gram(&pooled, bandwidth);
        let ia: Vec<usize> = (0..12).collect();
        let ib: Vec<usize> = (12..30).collect();
        let cached = mmd2_from_gram(&gram, 30, &ia, &ib);
        let direct = biased_mmd2(&pooled[..12], &pooled[12..], bandwidth);
        assert!((cached - direct).abs() < 1e-13, "cached={cached} direct={direct}");
        // A permuted partition matches the direct value on the correspondingly permuted data.
        let order: Vec<usize> = (0..30).rev().collect();
        let permuted: Vec<f64> = order.iter().map(|&i| pooled[i]).collect();
        let cached = mmd2_from_gram(&gram, 30, &order[..12], &order[12..]);
        let direct = biased_mmd2(&permuted[..12], &permuted[12..], bandwidth);
        assert!((cached - direct).abs() < 1e-13);
    }

    #[test]
    fn mean_diff_rejects_non_finite_observations() {
        assert!(mean_diff_two_sample(&[0.0, f64::NAN], &[1.0, 2.0]).is_err());
        assert!(mean_diff_two_sample(&[0.0, 1.0], &[2.0, f64::INFINITY]).is_err());
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
    fn classifier_treats_signed_zero_as_a_tie() {
        let (stat, p) = classifier_two_sample(&[0.0], &[-0.0]).unwrap();
        assert_eq!(stat, 0.0);
        assert_eq!(p, 1.0);
    }

    #[test]
    fn classifier_assigns_average_ranks_to_numeric_ties() {
        let (stat, p) = classifier_two_sample(&[1.0, 2.0], &[2.0, 3.0]).unwrap();
        assert!((stat - 0.375).abs() <= 1e-12);
        assert!(p.is_finite() && (0.0..=1.0).contains(&p));
    }

    #[test]
    fn classifier_rejects_non_finite_scores() {
        assert!(classifier_two_sample(&[0.0, f64::NAN], &[1.0]).is_err());
        assert!(classifier_two_sample(&[0.0], &[f64::NEG_INFINITY]).is_err());
    }

    #[test]
    fn residual_lr_refuses_single_row_or_constant_segments() {
        // A single-row segment has zero sample variance by construction (there is
        // nothing to vary against). Before the fix this reached the `v1 == 0.0` branch
        // and returned `(inf, 0.0)` — a definitive "changed" verdict from one
        // observation, with no minimum-segment-size gate anywhere in the call chain.
        assert!(residual_likelihood_ratio(&[1.0], &[2.0, 2.5, 1.8]).is_err());
        assert!(residual_likelihood_ratio(&[1.0, 2.5, 1.8], &[2.0]).is_err());
        // A multi-row but exactly constant segment is the same failure in disguise: no
        // within-segment variation to estimate a variance from.
        assert!(residual_likelihood_ratio(&[1.0, 1.0, 1.0], &[2.0, 2.5, 1.8]).is_err());
        // A genuinely varying segment on both sides is unaffected.
        let (stat, p) = residual_likelihood_ratio(&[1.0, 2.0, 1.5], &[5.0, 6.0, 5.5]).unwrap();
        assert!(stat.is_finite() && stat >= 0.0);
        assert!((0.0..=1.0).contains(&p));
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
    fn median_heuristic_unbiased_under_periodic_structure() {
        // Periodic/time-ordered input: a period-5 sine cycle repeated across a pool
        // large enough (n=300, C(300,2)=44,850 pairs) to force the random-subsample
        // path. A fixed stride over the row-major (i, j) enumeration resonates with
        // this period: at this n the old `idx % step` sampler's median was ~1.618x
        // (the golden ratio, coincidentally) the true pairwise-|diff| median. The
        // seeded subsample must recover the true median regardless of this structure.
        let n = 300usize;
        let period = 5.0;
        let pooled: Vec<f64> =
            (0..n).map(|i| (2.0 * std::f64::consts::PI * i as f64 / period).sin()).collect();
        let (a, b) = pooled.split_at(n / 2);

        let mut exact_diffs = Vec::with_capacity(n * (n - 1) / 2);
        for i in 0..n {
            for j in (i + 1)..n {
                exact_diffs.push((pooled[i] - pooled[j]).abs());
            }
        }
        exact_diffs.sort_by(|x, y| x.partial_cmp(y).unwrap_or(std::cmp::Ordering::Equal));
        let true_median = exact_diffs[exact_diffs.len() / 2];
        let expected_bandwidth = true_median;

        let mut rng = CausalRng::from_seed(0x5EED_C0DE);
        let bandwidth = rbf_bandwidth_median_heuristic(a, b, &mut rng);
        assert!(
            (bandwidth - expected_bandwidth).abs() / expected_bandwidth < 0.05,
            "bandwidth={bandwidth} expected≈{expected_bandwidth} (true_median={true_median})"
        );
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
