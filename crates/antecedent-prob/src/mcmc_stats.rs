//! Rank-normalized / folded R-hat and Geyer bulk / tail ESS for multi-chain MCMC.
//!
//! Chain layout: `samples[(chain * n_draws + draw) * n_params + param]`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]

use antecedent_kernels::norm_inv;

/// Per-parameter MCMC diagnostics (Vehtari / Stan style).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ParameterMcmcDiagnostics {
    /// Rank-normalized split R-hat.
    pub rhat_rank: f64,
    /// Folded rank-normalized split R-hat.
    pub rhat_folded: f64,
    /// `max(rhat_rank, rhat_folded)`.
    pub rhat: f64,
    /// Geyer bulk ESS on rank-normalized draws.
    pub ess_bulk: f64,
    /// Minimum Geyer ESS of the 5% / 95% tail indicators.
    pub ess_tail: f64,
}

/// Diagnostics for every parameter.
#[must_use]
pub fn parameter_mcmc_diagnostics(
    samples: &[f64],
    n_chains: usize,
    n_draws: usize,
    n_params: usize,
) -> Vec<ParameterMcmcDiagnostics> {
    (0..n_params).map(|p| diagnostics_one(samples, n_chains, n_draws, n_params, p)).collect()
}

/// Maximum rank∪folded R-hat across parameters (`∞` if any fails).
#[must_use]
pub fn max_split_rhat(samples: &[f64], n_chains: usize, n_draws: usize, n_params: usize) -> f64 {
    if n_chains < 2 || n_draws < 8 || n_params == 0 {
        return f64::INFINITY;
    }
    let mut max_r = 0.0_f64;
    for d in parameter_mcmc_diagnostics(samples, n_chains, n_draws, n_params) {
        if d.rhat.is_finite() {
            max_r = max_r.max(d.rhat);
        } else {
            return f64::INFINITY;
        }
    }
    max_r
}

/// Minimum bulk ESS across parameters.
#[must_use]
pub fn min_bulk_ess(samples: &[f64], n_chains: usize, n_draws: usize, n_params: usize) -> f64 {
    if n_chains == 0 || n_draws < 8 || n_params == 0 {
        return 0.0;
    }
    let mut min_ess = f64::INFINITY;
    for d in parameter_mcmc_diagnostics(samples, n_chains, n_draws, n_params) {
        min_ess = min_ess.min(d.ess_bulk);
    }
    if min_ess.is_finite() { min_ess } else { 0.0 }
}

/// Minimum tail ESS across parameters.
#[must_use]
pub fn min_tail_ess(samples: &[f64], n_chains: usize, n_draws: usize, n_params: usize) -> f64 {
    if n_chains == 0 || n_draws < 8 || n_params == 0 {
        return 0.0;
    }
    let mut min_ess = f64::INFINITY;
    for d in parameter_mcmc_diagnostics(samples, n_chains, n_draws, n_params) {
        min_ess = min_ess.min(d.ess_tail);
    }
    if min_ess.is_finite() { min_ess } else { 0.0 }
}

/// Whether every chain moved on at least one parameter.
///
/// Movement: `max - min > 1e-12 * (1 + |median|)` over post-warmup draws.
#[must_use]
pub fn all_chains_moved(samples: &[f64], n_chains: usize, n_draws: usize, n_params: usize) -> bool {
    if n_chains == 0 || n_draws == 0 || n_params == 0 {
        return false;
    }
    for c in 0..n_chains {
        let mut moved = false;
        for p in 0..n_params {
            let mut vals = Vec::with_capacity(n_draws);
            for d in 0..n_draws {
                vals.push(sample_at(samples, c, d, n_draws, n_params, p));
            }
            vals.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
            let lo = vals[0];
            let hi = vals[n_draws - 1];
            let med = median_sorted(&vals);
            let range = hi - lo;
            if range > 1e-12 * (1.0 + med.abs()) {
                moved = true;
                break;
            }
        }
        if !moved {
            return false;
        }
    }
    true
}

fn diagnostics_one(
    samples: &[f64],
    n_chains: usize,
    n_draws: usize,
    n_params: usize,
    param: usize,
) -> ParameterMcmcDiagnostics {
    let fail = ParameterMcmcDiagnostics {
        rhat_rank: f64::NAN,
        rhat_folded: f64::NAN,
        rhat: f64::NAN,
        ess_bulk: 0.0,
        ess_tail: 0.0,
    };
    if n_chains < 2 || n_draws < 8 {
        return fail;
    }
    // Drop last draw when odd so halves are equal.
    let n_use = if n_draws % 2 == 0 { n_draws } else { n_draws - 1 };
    let half = n_use / 2;
    if half < 4 {
        return fail;
    }
    let m = n_chains * 2;
    let s = m * half; // total split-chain draws

    let mut raw = vec![0.0; s];
    for c in 0..n_chains {
        for split in 0..2 {
            let seg = c * 2 + split;
            let start = split * half;
            for d in 0..half {
                raw[seg * half + d] = sample_at(samples, c, start + d, n_draws, n_params, param);
            }
        }
    }

    if !has_variation(&raw) {
        return fail;
    }

    let z_rank = rank_normalize(&raw);
    let rhat_rank = split_rhat_on_segments(&z_rank, m, half);
    let folded = fold_about_median(&raw);
    let z_fold = rank_normalize(&folded);
    let rhat_folded = split_rhat_on_segments(&z_fold, m, half);
    let rhat = if rhat_rank.is_finite() && rhat_folded.is_finite() {
        rhat_rank.max(rhat_folded)
    } else {
        f64::NAN
    };

    let ess_bulk = geyer_ess_split(&z_rank, m, half);
    let (q05, q95) = empirical_quantiles(&raw, 0.05, 0.95);
    let mut ind_lo = vec![0.0; s];
    let mut ind_hi = vec![0.0; s];
    for i in 0..s {
        ind_lo[i] = if raw[i] <= q05 { 1.0 } else { 0.0 };
        ind_hi[i] = if raw[i] >= q95 { 1.0 } else { 0.0 };
    }
    let ess_lo = if has_variation(&ind_lo) { geyer_ess_split(&ind_lo, m, half) } else { 0.0 };
    let ess_hi = if has_variation(&ind_hi) { geyer_ess_split(&ind_hi, m, half) } else { 0.0 };
    let ess_tail = ess_lo.min(ess_hi);

    ParameterMcmcDiagnostics { rhat_rank, rhat_folded, rhat, ess_bulk, ess_tail }
}

fn has_variation(x: &[f64]) -> bool {
    if x.is_empty() {
        return false;
    }
    let mut lo = x[0];
    let mut hi = x[0];
    for &v in x {
        if !v.is_finite() {
            return false;
        }
        lo = lo.min(v);
        hi = hi.max(v);
    }
    hi - lo > 1e-12 * (1.0 + lo.abs().max(hi.abs()))
}

fn fold_about_median(x: &[f64]) -> Vec<f64> {
    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let med = median_sorted(&sorted);
    x.iter().map(|&v| (v - med).abs()).collect()
}

fn median_sorted(sorted: &[f64]) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n % 2 == 1 { sorted[n / 2] } else { 0.5 * (sorted[n / 2 - 1] + sorted[n / 2]) }
}

fn empirical_quantiles(x: &[f64], q_lo: f64, q_hi: f64) -> (f64, f64) {
    let mut sorted = x.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    (quantile_sorted(&sorted, q_lo), quantile_sorted(&sorted, q_hi))
}

fn quantile_sorted(sorted: &[f64], q: f64) -> f64 {
    let n = sorted.len();
    if n == 0 {
        return f64::NAN;
    }
    if n == 1 {
        return sorted[0];
    }
    let pos = q * (n - 1) as f64;
    let lo = pos.floor() as usize;
    let hi = pos.ceil() as usize;
    if lo == hi {
        sorted[lo]
    } else {
        let w = pos - lo as f64;
        sorted[lo] * (1.0 - w) + sorted[hi] * w
    }
}

/// Average ranks for ties, then Φ⁻¹((r − 3/8) / (S + 1/4)).
fn rank_normalize(x: &[f64]) -> Vec<f64> {
    let s = x.len();
    let mut idx: Vec<usize> = (0..s).collect();
    idx.sort_by(|&i, &j| x[i].partial_cmp(&x[j]).unwrap_or(std::cmp::Ordering::Equal));
    let mut ranks = vec![0.0; s];
    let mut i = 0;
    while i < s {
        let mut j = i + 1;
        while j < s && x[idx[j]] == x[idx[i]] {
            j += 1;
        }
        // Ranks are 1-based; average for ties.
        let avg = (i + 1 + j) as f64 / 2.0;
        for k in i..j {
            ranks[idx[k]] = avg;
        }
        i = j;
    }
    let denom = s as f64 + 0.25;
    ranks
        .into_iter()
        .map(|r| {
            let p = (r - 0.375) / denom;
            norm_inv(p.clamp(1e-12, 1.0 - 1e-12))
        })
        .collect()
}

fn split_rhat_on_segments(seg_major: &[f64], m: usize, n: usize) -> f64 {
    // Layout: segment-major contiguous blocks of length n.
    let mut means = vec![0.0; m];
    let mut vars = vec![0.0; m];
    let nf = n as f64;
    for seg in 0..m {
        let base = seg * n;
        let mut mean = 0.0;
        for d in 0..n {
            mean += seg_major[base + d];
        }
        mean /= nf;
        means[seg] = mean;
        let mut var = 0.0;
        for d in 0..n {
            let v = seg_major[base + d] - mean;
            var += v * v;
        }
        vars[seg] = var / (nf - 1.0);
    }
    let mut w = 0.0;
    let mut grand = 0.0;
    for i in 0..m {
        w += vars[i];
        grand += means[i];
    }
    w /= m as f64;
    grand /= m as f64;
    let mut b = 0.0;
    for i in 0..m {
        let d = means[i] - grand;
        b += d * d;
    }
    b = nf * b / (m as f64 - 1.0);
    if !(w > 0.0) {
        return if b > 0.0 { f64::INFINITY } else { f64::NAN };
    }
    let var_hat = ((nf - 1.0) / nf) * w + b / nf;
    (var_hat / w).sqrt()
}

/// Geyer IPS + IMS ESS on split-chain segments (segment-major layout).
fn geyer_ess_split(seg_major: &[f64], m: usize, n: usize) -> f64 {
    let s = (m * n) as f64;
    let (var_hat, acov) = split_autocovariances(seg_major, m, n);
    if !(var_hat > 0.0) || acov.is_empty() {
        return 0.0;
    }
    // ρ̂_t = acov[t] / var_hat; Geyer initial positive sequence on pair sums.
    let mut rho_hat = Vec::with_capacity(acov.len());
    for &a in &acov {
        rho_hat.push(a / var_hat);
    }
    // Pair sums P_t = ρ_{2t} + ρ_{2t+1}.
    let max_pairs = (rho_hat.len().saturating_sub(1)) / 2;
    if max_pairs == 0 {
        return s;
    }
    let mut pairs = Vec::with_capacity(max_pairs);
    for t in 0..max_pairs {
        let p = rho_hat[2 * t] + rho_hat[2 * t + 1];
        pairs.push(p);
    }
    // Initial positive sequence: truncate at first non-positive pair.
    let mut truncate = pairs.len();
    for (i, &p) in pairs.iter().enumerate() {
        if p <= 0.0 {
            truncate = i;
            break;
        }
    }
    if truncate == 0 {
        return s;
    }
    pairs.truncate(truncate);
    // Initial monotone sequence: enforce nonincreasing.
    for i in 1..pairs.len() {
        if pairs[i] > pairs[i - 1] {
            pairs[i] = pairs[i - 1];
        }
    }
    // Stan/Vehtari: τ̂ = −1 + 2 Σ P_t' with P_t' = ρ̂_{2t'} + ρ̂_{2t'+1}.
    let mut tau = -1.0;
    for &p in &pairs {
        tau += 2.0 * p;
    }
    tau = tau.max(1.0);
    (s / tau).min(s)
}

fn split_autocovariances(seg_major: &[f64], m: usize, n: usize) -> (f64, Vec<f64>) {
    let nf = n as f64;
    let mut means = vec![0.0; m];
    let mut vars = vec![0.0; m];
    for seg in 0..m {
        let base = seg * n;
        let mut mean = 0.0;
        for d in 0..n {
            mean += seg_major[base + d];
        }
        mean /= nf;
        means[seg] = mean;
        let mut var = 0.0;
        for d in 0..n {
            let v = seg_major[base + d] - mean;
            var += v * v;
        }
        vars[seg] = var / (nf - 1.0);
    }
    let mut w = 0.0;
    let mut grand = 0.0;
    for i in 0..m {
        w += vars[i];
        grand += means[i];
    }
    w /= m as f64;
    grand /= m as f64;
    let mut b = 0.0;
    for i in 0..m {
        let d = means[i] - grand;
        b += d * d;
    }
    b = nf * b / (m as f64 - 1.0);
    let var_hat = ((nf - 1.0) / nf) * w + b / nf;

    // Mean autocovariance across chains at each lag (unbiased within-chain).
    // Stan/Vehtari: Â_0 = var̂⁺, Â_t = var̂⁺ − W + ā_t for t>0, so
    // ρ̂_t = 1 − (W − ā_t)/var̂⁺ (not ā_t/var̂⁺, which ignores between-chain).
    let max_lag = n.saturating_sub(1);
    let mut acov = vec![0.0; max_lag + 1];
    acov[0] = var_hat;
    for lag in 1..=max_lag {
        let mut acc = 0.0;
        for seg in 0..m {
            let base = seg * n;
            let mean = means[seg];
            let mut num = 0.0;
            for d in 0..(n - lag) {
                num += (seg_major[base + d] - mean) * (seg_major[base + d + lag] - mean);
            }
            acc += num / (nf - 1.0);
        }
        let a_bar = acc / m as f64;
        acov[lag] = var_hat - w + a_bar;
    }
    (var_hat, acov)
}

#[inline]
fn sample_at(
    samples: &[f64],
    chain: usize,
    draw: usize,
    n_draws: usize,
    n_params: usize,
    param: usize,
) -> f64 {
    samples[(chain * n_draws + draw) * n_params + param]
}

#[cfg(test)]
mod tests {
    use super::*;

    fn fill_iid_normal(n_chains: usize, n_draws: usize, seed: u64) -> Vec<f64> {
        // Deterministic LCG pseudo-normals via Box-Muller-ish bits.
        let mut samples = vec![0.0; n_chains * n_draws];
        let mut state = seed;
        for v in &mut samples {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let u1 = ((state >> 33) as f64) / ((1u64 << 31) as f64);
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
            let u2 = ((state >> 33) as f64) / ((1u64 << 31) as f64);
            let u1 = u1.clamp(1e-12, 1.0 - 1e-12);
            *v = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        }
        samples
    }

    #[test]
    fn iid_chains_rhat_near_one_ess_near_n() {
        let n_chains = 4;
        let n_draws = 200;
        let n_total = (n_chains * n_draws) as f64;
        let samples = fill_iid_normal(n_chains, n_draws, 7);
        let r = max_split_rhat(&samples, n_chains, n_draws, 1);
        let ess = min_bulk_ess(&samples, n_chains, n_draws, 1);
        let ess_t = min_tail_ess(&samples, n_chains, n_draws, 1);
        assert!(r < 1.05, "rhat={r}");
        // With τ̂ = −1 + 2ΣP, IID should recover nearly full sample size.
        assert!(ess > 0.7 * n_total, "ess={ess} vs N={n_total}");
        assert!(ess_t > 0.4 * n_total, "tail ess={ess_t}");
    }

    fn fill_ar1(n_chains: usize, n_draws: usize, rho: f64, seed: u64) -> Vec<f64> {
        let innov = fill_iid_normal(n_chains, n_draws, seed);
        let mut samples = vec![0.0; n_chains * n_draws];
        let scale = (1.0 - rho * rho).sqrt();
        for c in 0..n_chains {
            let base = c * n_draws;
            samples[base] = innov[base];
            for d in 1..n_draws {
                samples[base + d] = rho * samples[base + d - 1] + scale * innov[base + d];
            }
        }
        samples
    }

    #[test]
    fn ar1_chains_ess_near_analytic() {
        let n_chains = 4;
        let n_draws = 500;
        let rho = 0.5;
        let s = (n_chains * n_draws) as f64;
        let expected = s * (1.0 - rho) / (1.0 + rho);
        let samples = fill_ar1(n_chains, n_draws, rho, 41);
        let ess = min_bulk_ess(&samples, n_chains, n_draws, 1);
        assert!((ess - expected).abs() < 0.35 * expected, "ess={ess} expected≈{expected}");
    }

    #[test]
    fn shifted_means_fail_rank_rhat() {
        let n_chains = 4;
        let n_draws = 100;
        let mut samples = vec![0.0; n_chains * n_draws];
        for c in 0..n_chains {
            for d in 0..n_draws {
                samples[c * n_draws + d] = c as f64 + (d as f64) * 1e-6;
            }
        }
        let d = diagnostics_one(&samples, n_chains, n_draws, 1, 0);
        assert!(d.rhat_rank > 1.1, "rhat_rank={}", d.rhat_rank);
    }

    #[test]
    fn disagreeing_chains_keep_ess_far_below_n() {
        // Near-IID within each chain, but chain means differ: Stan/Vehtari ESS
        // must stay ≪ N (ρ̂_t ≈ 1 − W/var̂⁺ for t>0), not collapse to ~N.
        let n_chains = 4;
        let n_draws = 200;
        let n_total = (n_chains * n_draws) as f64;
        let mut samples = fill_iid_normal(n_chains, n_draws, 31);
        for c in 0..n_chains {
            let shift = (c as f64) * 5.0;
            for d in 0..n_draws {
                samples[c * n_draws + d] += shift;
            }
        }
        let ess = min_bulk_ess(&samples, n_chains, n_draws, 1);
        assert!(
            ess < 0.25 * n_total,
            "disagreeing chains ess={ess} should be ≪ N={n_total}"
        );
    }

    #[test]
    fn equal_means_different_scales_fail_folded_rhat() {
        let n_chains = 4;
        let n_draws = 200;
        let mut samples = vec![0.0; n_chains * n_draws];
        let base = fill_iid_normal(n_chains, n_draws, 11);
        for c in 0..n_chains {
            let scale = if c < 2 { 1.0 } else { 5.0 };
            for d in 0..n_draws {
                samples[c * n_draws + d] = scale * base[c * n_draws + d];
            }
        }
        let d = diagnostics_one(&samples, n_chains, n_draws, 1, 0);
        assert!(d.rhat_folded > 1.05, "rhat_folded={} rank={}", d.rhat_folded, d.rhat_rank);
        assert!(d.rhat >= d.rhat_folded);
    }

    #[test]
    fn constant_chains_fail_not_succeed() {
        let n_chains = 4;
        let n_draws = 40;
        let samples = vec![1.0; n_chains * n_draws];
        let d = diagnostics_one(&samples, n_chains, n_draws, 1, 0);
        assert!(d.rhat.is_nan(), "rhat={}", d.rhat);
        assert_eq!(d.ess_bulk, 0.0);
        assert_eq!(d.ess_tail, 0.0);
        assert!(!all_chains_moved(&samples, n_chains, n_draws, 1));
        assert!(!max_split_rhat(&samples, n_chains, n_draws, 1).is_finite());
        assert_eq!(min_bulk_ess(&samples, n_chains, n_draws, 1), 0.0);
    }

    #[test]
    fn poor_tail_mixing_lowers_tail_ess() {
        // Three chains explore the tails; one chain is stuck near zero for long
        // contiguous blocks so the 5%/95% indicators mix poorly.
        let n_chains = 4;
        let n_draws = 400;
        let mut samples = fill_iid_normal(n_chains, n_draws, 19);
        for d in 0..n_draws {
            let block = d / 50;
            samples[3 * n_draws + d] = if block % 2 == 0 { -4.0 } else { 4.0 };
        }
        let d = diagnostics_one(&samples, n_chains, n_draws, 1, 0);
        let iid = fill_iid_normal(n_chains, n_draws, 23);
        let d_iid = diagnostics_one(&iid, n_chains, n_draws, 1, 0);
        assert!(
            d.ess_tail < d_iid.ess_tail * 0.5,
            "tail ess bad={} good={} bulk_bad={}",
            d.ess_tail,
            d_iid.ess_tail,
            d.ess_bulk
        );
    }
}
