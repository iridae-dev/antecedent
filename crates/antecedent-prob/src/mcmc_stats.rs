//! Rank-normalized / folded R-hat and Geyer bulk / tail ESS for multi-chain MCMC.
//!
//! Chain layout: `samples[(chain * n_draws + draw) * n_params + param]`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_kernels::{median_sorted, midranks_into, norm_inv, quantile_type7_sorted};

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

/// One-pass gate summary: max R̂ plus min bulk/tail ESS.
#[derive(Clone, Copy, Debug)]
pub struct McmcSummary {
    /// Maximum rank∪folded split R̂ across parameters (`∞` if any fails).
    pub max_rhat: f64,
    /// Minimum bulk ESS across parameters.
    pub min_bulk_ess: f64,
    /// Minimum tail ESS across parameters.
    pub min_tail_ess: f64,
}

/// Compute [`max_split_rhat`], [`min_bulk_ess`], and [`min_tail_ess`] from one
/// diagnostics pass. The three standalone reducers each recompute every
/// parameter's diagnostics — publication gates that need all three were paying
/// 3× the (quadratic-in-draws) work for identical values.
#[must_use]
pub fn mcmc_summary(
    samples: &[f64],
    n_chains: usize,
    n_draws: usize,
    n_params: usize,
) -> McmcSummary {
    let rhat_guard = n_chains < 2 || n_draws < 8 || n_params == 0;
    let ess_guard = n_chains == 0 || n_draws < 8 || n_params == 0;
    if rhat_guard && ess_guard {
        return McmcSummary { max_rhat: f64::INFINITY, min_bulk_ess: 0.0, min_tail_ess: 0.0 };
    }
    let diags = parameter_mcmc_diagnostics(samples, n_chains, n_draws, n_params);
    summarize(&diags, rhat_guard, ess_guard)
}

/// [`mcmc_summary`] with the per-parameter diagnostics computed under the context's
/// parallelism. Each parameter's diagnostics depend only on its own column of draws and the
/// reduction runs in parameter order, so the summary is bit-identical to [`mcmc_summary`]
/// for every thread count.
#[must_use]
pub fn mcmc_summary_parallel(
    samples: &[f64],
    n_chains: usize,
    n_draws: usize,
    n_params: usize,
    ctx: &antecedent_core::ExecutionContext,
) -> McmcSummary {
    let rhat_guard = n_chains < 2 || n_draws < 8 || n_params == 0;
    let ess_guard = n_chains == 0 || n_draws < 8 || n_params == 0;
    if rhat_guard && ess_guard {
        return McmcSummary { max_rhat: f64::INFINITY, min_bulk_ess: 0.0, min_tail_ess: 0.0 };
    }
    let diags = ctx.map_indexed(n_params, |p, _| {
        Ok::<_, std::convert::Infallible>(diagnostics_one(samples, n_chains, n_draws, n_params, p))
    });
    let Ok(diags) = diags;
    summarize(&diags, rhat_guard, ess_guard)
}

fn summarize(diags: &[ParameterMcmcDiagnostics], rhat_guard: bool, ess_guard: bool) -> McmcSummary {
    let mut max_rhat = 0.0_f64;
    let mut min_bulk = f64::INFINITY;
    let mut min_tail = f64::INFINITY;
    for d in diags {
        if d.rhat.is_finite() {
            max_rhat = max_rhat.max(d.rhat);
        } else {
            max_rhat = f64::INFINITY;
        }
        min_bulk = min_bulk.min(d.ess_bulk);
        min_tail = min_tail.min(d.ess_tail);
    }
    McmcSummary {
        max_rhat: if rhat_guard { f64::INFINITY } else { max_rhat },
        min_bulk_ess: if ess_guard || !min_bulk.is_finite() { 0.0 } else { min_bulk },
        min_tail_ess: if ess_guard || !min_tail.is_finite() { 0.0 } else { min_tail },
    }
}

/// Maximum rank∪folded R-hat across parameters (`∞` if any fails).
#[must_use]
#[cfg(test)]
pub(crate) fn max_split_rhat(
    samples: &[f64],
    n_chains: usize,
    n_draws: usize,
    n_params: usize,
) -> f64 {
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
/// Movement means at least two distinct finite post-warmup draws. No absolute
/// tolerance is imposed: changing parameter units must not change this check.
/// Malformed layouts or nonfinite draws return false.
#[must_use]
pub fn all_chains_moved(samples: &[f64], n_chains: usize, n_draws: usize, n_params: usize) -> bool {
    if n_chains == 0
        || n_draws == 0
        || n_params == 0
        || !valid_layout(samples, n_chains, n_draws, n_params)
        || samples.iter().any(|v| !v.is_finite())
    {
        return false;
    }
    for c in 0..n_chains {
        let mut moved = false;
        for p in 0..n_params {
            let first = sample_at(samples, c, 0, n_draws, n_params, p);
            #[allow(
                clippy::float_cmp,
                reason = "a chain has moved iff any draw differs bitwise-in-value from its first draw"
            )]
            let differs =
                (1..n_draws).any(|d| sample_at(samples, c, d, n_draws, n_params, p) != first);
            if differs {
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

fn valid_layout(samples: &[f64], n_chains: usize, n_draws: usize, n_params: usize) -> bool {
    n_chains.checked_mul(n_draws).and_then(|n| n.checked_mul(n_params)) == Some(samples.len())
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
    if n_chains < 2 || n_draws < 8 || !valid_layout(samples, n_chains, n_draws, n_params) {
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
        // Both local efficiencies concern the CDF I(theta <= quantile)
        // (Vehtari 2021 eq. 16). >= q95 is only complementary for continuous
        // draws with no ties; it measures a different event at a tied cutoff.
        ind_hi[i] = if raw[i] <= q95 { 1.0 } else { 0.0 };
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
    hi > lo
}

fn fold_about_median(x: &[f64]) -> Vec<f64> {
    let mut sorted = x.to_vec();
    sorted.sort_by(f64::total_cmp);
    let med = median_sorted(&sorted);
    x.iter().map(|&v| (v - med).abs()).collect()
}

fn empirical_quantiles(x: &[f64], q_lo: f64, q_hi: f64) -> (f64, f64) {
    let mut sorted = x.to_vec();
    sorted.sort_by(f64::total_cmp);
    (quantile_type7_sorted(&sorted, q_lo), quantile_type7_sorted(&sorted, q_hi))
}

/// Average ranks for ties, then Φ⁻¹((r − 3/8) / (S + 1/4)).
fn rank_normalize(x: &[f64]) -> Vec<f64> {
    let s = x.len();
    let mut ranks = vec![0.0; s];
    midranks_into(x, &mut ranks).expect("rank buffer is allocated to the sample length");
    let denom = s as f64 + 0.25;
    ranks
        .into_iter()
        .map(|r| {
            let p = (r - 0.375) / denom;
            norm_inv(p.clamp(1e-12, 1.0 - 1e-12))
        })
        .collect()
}

#[allow(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "negated comparison is deliberate: a NaN input must take the rejecting branch, which the un-negated form would let through"
)]
fn split_rhat_on_segments(seg_major: &[f64], m: usize, n: usize) -> f64 {
    let stats = split_segment_stats(seg_major, m, n);
    if !(stats.w > 0.0) {
        return if stats.var_hat > 0.0 { f64::INFINITY } else { f64::NAN };
    }
    (stats.var_hat / stats.w).sqrt()
}

/// Geyer (1992) IPS + IMS ESS on split-chain segments (segment-major layout).
///
/// Autocovariances are computed lazily per lag and stop at the Geyer
/// truncation point (the first non-positive pair sum) — the truncation rule
/// only ever reads lags `≤ 2t+1`, so evaluating all `n−1` lags up front was
/// O(m·n²) work of which O(m·n·τ) is used. Per-lag arithmetic, the
/// truncation test, and monotone clamp agree with the corresponding eager form.
#[allow(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "negated comparison is deliberate: a NaN input must take the rejecting branch, which the un-negated form would let through"
)]
fn geyer_ess_split(seg_major: &[f64], m: usize, n: usize) -> f64 {
    let s = (m * n) as f64;
    let stats = split_segment_stats(seg_major, m, n);
    let var_hat = stats.var_hat;
    if !(var_hat > 0.0) || n == 0 {
        return 0.0;
    }
    // Pair sums P_t = ρ_{2t} + ρ_{2t+1}, ρ̂_t = Â_t / var̂⁺; lags 2t, 2t+1
    // must both exist, i.e. 2t+1 ≤ n−1.
    let max_pairs = n / 2;
    if max_pairs == 0 {
        return s;
    }
    // Stan/Vehtari (2021): τ̂ = −1 + 2 Σ P_t' over the initial positive,
    // monotone-nonincreasing pair sequence.
    let mut tau = -1.0;
    let mut prev_pair = f64::INFINITY;
    let mut any_pair = false;
    for t in 0..max_pairs {
        let a0 =
            if 2 * t == 0 { var_hat } else { lag_autocovariance(seg_major, m, n, 2 * t, &stats) };
        let a1 = lag_autocovariance(seg_major, m, n, 2 * t + 1, &stats);
        let mut p = a0 / var_hat + a1 / var_hat;
        // Initial positive sequence: truncate at first non-positive pair.
        if p <= 0.0 {
            break;
        }
        // Initial monotone sequence: enforce nonincreasing.
        if p > prev_pair {
            p = prev_pair;
        }
        prev_pair = p;
        any_pair = true;
        tau += 2.0 * p;
    }
    if !any_pair {
        return s;
    }
    // Stan/ArviZ floor τ̂ at 1/log10(S), not 1, and do not clamp ESS to S: antithetic
    // (negatively autocorrelated) chains are legitimately super-efficient and can exceed the
    // raw draw count. Flooring at 1 and clamping at S understates ESS — safe for trusting a
    // posterior, but it misreports sampler quality and can spuriously fail the ESS gate.
    tau = tau.max(1.0 / s.log10().max(f64::MIN_POSITIVE));
    s / tau
}

/// Per-segment means/variances plus W and var̂⁺ (shared by R̂ and the lags).
struct SegmentStats {
    means: Vec<f64>,
    w: f64,
    var_hat: f64,
}

#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
fn split_segment_stats(seg_major: &[f64], m: usize, n: usize) -> SegmentStats {
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
    SegmentStats { means, w, var_hat }
}

/// `Â_lag` for one lag > 0 (biased autocovariance divisor N, Vehtari 2021 sec. 3.2:
/// `Â_t = var̂⁺ − W + ā_t`, so `ρ̂_t = 1 − (W − ā_t)/var̂⁺`, not `ā_t/var̂⁺`,
/// which ignores between-chain variance).
fn lag_autocovariance(
    seg_major: &[f64],
    m: usize,
    n: usize,
    lag: usize,
    stats: &SegmentStats,
) -> f64 {
    let nf = n as f64;
    let mut acc = 0.0;
    for seg in 0..m {
        let base = seg * n;
        let mean = stats.means[seg];
        let mut num = 0.0;
        for d in 0..(n - lag) {
            num += (seg_major[base + d] - mean) * (seg_major[base + d + lag] - mean);
        }
        // The autocovariance estimator uses N; the sample variance W above
        // still uses N-1. They have different roles in the combined rho.
        acc += num / nf;
    }
    let a_bar = acc / m as f64;
    stats.var_hat - stats.w + a_bar
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

    #[test]
    fn lag_covariance_uses_the_biased_sample_divisor_from_the_ess_definition() {
        // Two segments with means 1.5 and 2.5: W=5/3, B=2, var+=7/4.
        // Each lag-one numerator is 5/4. Dividing by N=4 gives 5/16,
        // hence A_1 = var+ - W + 5/16 = 19/48 (Vehtari 2021, sec. 3.2,
        // discussion before eq. 10). Using N-1 instead would give 1/2.
        let segments = [0.0, 1.0, 2.0, 3.0, 1.0, 2.0, 3.0, 4.0];
        let stats = split_segment_stats(&segments, 2, 4);
        assert!((lag_autocovariance(&segments, 2, 4, 1, &stats) - 19.0 / 48.0).abs() < 1e-12);
    }

    fn fill_iid_normal(n_chains: usize, n_draws: usize, seed: u64) -> Vec<f64> {
        let mut rng = antecedent_core::CausalRng::from_seed(seed);
        (0..n_chains * n_draws).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect()
    }

    #[test]
    fn parallel_summary_is_bit_identical_to_the_serial_summary() {
        use antecedent_core::{ExecutionContext, NonZeroThreadCount, Parallelism};
        let (chains, draws, params) = (4, 300, 5);
        let samples = fill_iid_normal(chains, draws * params, 3);
        let serial = mcmc_summary(&samples, chains, draws, params);
        let mut ctx = ExecutionContext::for_tests(1)
            .with_admission(antecedent_core::execution::AdmissionPool::fixed(4, u64::MAX));
        ctx.parallelism = Parallelism::bounded(NonZeroThreadCount::new(4).unwrap());
        let parallel = mcmc_summary_parallel(&samples, chains, draws, params, &ctx);
        assert_eq!(serial.max_rhat.to_bits(), parallel.max_rhat.to_bits());
        assert_eq!(serial.min_bulk_ess.to_bits(), parallel.min_bulk_ess.to_bits());
        assert_eq!(serial.min_tail_ess.to_bits(), parallel.min_tail_ess.to_bits());
    }

    #[test]
    fn ess_includes_the_last_available_pair_for_even_segment_lengths() {
        // Two constant segments with disagreeing means have rho_t = 1 at
        // every available lag. Four lags give pairs (0,1), (2,3), hence
        // tau = -1 + 2*(2+2) = 7 and ESS = 8/7.
        let segments = [0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0];
        assert!((geyer_ess_split(&segments, 2, 4) - 8.0 / 7.0).abs() < 1e-12);
    }

    #[test]
    fn diagnostics_are_invariant_to_parameter_units() {
        let samples = fill_iid_normal(4, 200, 7);
        let tiny: Vec<_> = samples.iter().map(|v| v * 1e-20).collect();
        let reference = parameter_mcmc_diagnostics(&samples, 4, 200, 1)[0];
        let scaled = parameter_mcmc_diagnostics(&tiny, 4, 200, 1)[0];
        assert!((reference.rhat - scaled.rhat).abs() < 1e-12);
        assert!((reference.ess_bulk - scaled.ess_bulk).abs() < 1e-12);
        assert!((reference.ess_tail - scaled.ess_tail).abs() < 1e-12);
        assert!(all_chains_moved(&tiny, 4, 200, 1));
    }

    #[test]
    fn malformed_or_nonfinite_chains_fail_closed() {
        for samples in [vec![0.0; 15], vec![f64::NAN; 16]] {
            let result = mcmc_summary(&samples, 2, 8, 1);
            assert!(!result.max_rhat.is_finite());
            assert_eq!(result.min_bulk_ess, 0.0);
            assert!(!all_chains_moved(&samples, 2, 8, 1));
        }
        assert!(!all_chains_moved(&[], usize::MAX, 8, 1));
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
        // Near-IID within each chain, but chain means differ: Stan/Vehtari (2021) ESS
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
        assert!(ess < 0.25 * n_total, "disagreeing chains ess={ess} should be ≪ N={n_total}");
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
