//! Circular-block bootstrap over the lag-aligned rows of one series.
//!
//! Lag-aligned rows of one series are serially dependent (overlapping lag
//! windows, autocorrelated residuals, MA(h−1) errors at horizon h), so iid row
//! resampling understates uncertainty. This bootstrap resamples circular blocks
//! of *consecutive* lag-aligned rows (Künsch 1989's blocks-of-blocks form): every
//! replicate row keeps its own intact lag window, and consecutive rows inside a
//! block keep their joint dependence. Resampling the raw series instead and
//! rebuilding lags would pair, at every block junction, an outcome with lagged
//! regressors from an unrelated block; in the 1.9 calibration that inflated a
//! strong-effect SE by ~10% under iid noise and pushed a nominal-90% temporal
//! mediation Total interval to 0.96 coverage.
//!
//! The block length comes from [`antecedent_data::circular_block_length`]:
//! `max(structural_span, ⌈n^{1/3}⌉)`, capped at `n`; [`dependence_block_length`]
//! lengthens it when an estimating score is persistently dependent.
//!
//! Several designs on one series (the atoms of a temporal class / DBN mixture)
//! share one replicate through [`aligned_block_bootstrap`]: blocks of consecutive
//! series times over the window every design can evaluate, each design refit on
//! its own lag-aligned rows at those times.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::cast_sign_loss)]

use antecedent_core::ExecutionContext;
use antecedent_data::{ResamplingPlan, circular_block_length, fill_resample_indexes};

use crate::util::{BootstrapSeResult, finalize_bootstrap_se_ex};

/// Effective-sample floor for a trusted circular-block interval.
///
/// With the `⌈n^{1/3}⌉` block rule (and the [`fixed_b_scale`] correction),
/// measured coverage of nominal-90% intervals fell below the calibration band
/// only in designs whose [`effective_rows`] sat under this floor (AR(1) ρ = 0.9
/// residuals at n = 60 and 160: Pulse 0.835–0.855, mediation Total
/// 0.845–0.853); every measured design above it stayed in band. The floor is conservative: ρ = 0.5 series at n = 60 also sit below it
/// and measured 0.86–0.91 (see `crates/antecedent/tests/v19_temporal_frequentist.rs`).
/// Results below it carry a `short_series` warning.
pub const MIN_EFFECTIVE_ROWS: f64 = 100.0;

/// Effective rows of an estimating-score series, `n·(1 − r₁)/(1 + r₁)`, with `r₁`
/// its lag-1 autocorrelation floored at zero (the AR(1) variance-inflation
/// equivalent). A constant series returns `n`; non-finite scores return `NaN`.
#[must_use]
pub fn effective_rows(scores: &[f64]) -> f64 {
    let n = scores.len();
    if n < 3 {
        return n as f64;
    }
    let mean = scores.iter().sum::<f64>() / n as f64;
    let (mut gamma0, mut gamma1) = (0.0, 0.0);
    for (t, &s) in scores.iter().enumerate() {
        let d = s - mean;
        gamma0 += d * d;
        if t + 1 < n {
            gamma1 += d * (scores[t + 1] - mean);
        }
    }
    if !gamma0.is_finite() || !gamma1.is_finite() {
        return f64::NAN;
    }
    if gamma0 <= 0.0 {
        return n as f64;
    }
    let r1 = (gamma1 / gamma0).clamp(0.0, 0.99);
    n as f64 * (1.0 - r1) / (1.0 + r1)
}

/// Data-driven circular-block length for the mean of `scores` (Politis & White
/// 2004, with the Patton, Politis & White 2009 correction for the circular
/// bootstrap): `b = (2 Ĝ² / D̂)^{1/3} n^{1/3}`, `D̂ = (4/3) ĝ(0)²`, from flat-top
/// lag-window estimates with the bandwidth picked by the first run of `K_n`
/// insignificant autocorrelations. Capped at `min(3√n, n/3)`; `None` for fewer
/// than 8 scores or a degenerate series.
#[must_use]
pub fn politis_white_block_length(scores: &[f64]) -> Option<usize> {
    let n = scores.len();
    if n < 8 || scores.iter().any(|s| !s.is_finite()) {
        return None;
    }
    let nf = n as f64;
    let mean = scores.iter().sum::<f64>() / nf;
    let centered: Vec<f64> = scores.iter().map(|s| s - mean).collect();
    let autocov = |k: usize| -> f64 {
        centered[..n - k].iter().zip(&centered[k..]).map(|(a, b)| a * b).sum::<f64>() / nf
    };
    let gamma0 = autocov(0);
    if gamma0 <= 0.0 {
        return None;
    }
    let k_n = 5usize.max(nf.log10().sqrt().ceil() as usize);
    let m_max = (nf.sqrt().ceil() as usize + k_n).min(n - 1);
    let threshold = 2.0 * (nf.log10() / nf).sqrt();
    let rho: Vec<f64> = (0..=m_max).map(|k| autocov(k) / gamma0).collect();
    let mut m_hat = m_max.saturating_sub(k_n);
    for m in 0..=m_max.saturating_sub(k_n) {
        if (1..=k_n).all(|j| rho.get(m + j).is_none_or(|r| r.abs() < threshold)) {
            m_hat = m;
            break;
        }
    }
    let big_m = (2 * m_hat).min(m_max).max(1);
    let flat_top = |t: f64| {
        let t = t.abs();
        if t <= 0.5 {
            1.0
        } else if t <= 1.0 {
            2.0 * (1.0 - t)
        } else {
            0.0
        }
    };
    let (mut g0, mut big_g) = (gamma0, 0.0);
    for (k, r) in rho.iter().enumerate().take(big_m + 1).skip(1) {
        let weight = flat_top(k as f64 / big_m as f64) * r * gamma0;
        g0 += 2.0 * weight;
        big_g += 2.0 * k as f64 * weight;
    }
    if g0 <= 0.0 {
        return None;
    }
    let d_cb = 4.0 / 3.0 * g0 * g0;
    let b = (2.0 * big_g * big_g / d_cb).cbrt() * nf.cbrt();
    let cap = (3.0 * nf.sqrt()).min(nf / 3.0).ceil();
    b.is_finite().then(|| b.clamp(1.0, cap.max(1.0)).ceil() as usize)
}

/// Dependence-aware circular-block length for a fixed-b interval over `rows`
/// lag-aligned rows: `max(rule, ⌈b_PW · rows^{1/6}⌉)`, capped at `rows / 3` (and
/// never below the rule), where the rule is [`circular_block_length`] and `b_PW`
/// is the largest [`politis_white_block_length`] over `scores` (each an
/// estimating-score series on the same rows: every atom's influence, and their
/// weighted sum).
///
/// Politis–White is MSE-optimal for the variance estimate (rate `n^{1/3}`). An
/// interval is judged by coverage, and with fixed-b critical values the
/// coverage-error-optimal Bartlett bandwidth grows at `n^{1/2}` (Sun, Phillips &
/// Jin 2008), so the data-driven constant is kept and the rate rescaled by
/// `n^{1/6}`. The per-atom maximum matters: a mixture score can look nearly
/// uncorrelated lag by lag while one atom's score carries slowly decaying
/// dependence (persistent regressor × persistent residual). In the 1.9
/// calibration (AR(1) ρ = 0.9, n = 400) the `⌈n^{1/3}⌉` rule gave SE/SD 0.82–0.93
/// and coverage 0.81–0.85 on multi-atom mixtures; this length restored SE/SD ≈ 1.
/// Weakly dependent scores keep the rule.
#[must_use]
pub fn dependence_block_length(structural_span: usize, rows: usize, scores: &[&[f64]]) -> usize {
    let rule = circular_block_length(structural_span, rows);
    let pw = scores.iter().filter_map(|s| politis_white_block_length(s)).max().unwrap_or(0);
    let testing = (pw as f64 * (rows as f64).powf(1.0 / 6.0)).ceil() as usize;
    rule.max(testing.min(rows / 3)).min(rows.max(1))
}

/// Outcome of a shared circular-block bootstrap of `K` scalar targets.
#[derive(Clone, Debug)]
pub struct RowBlockBootstrap<const K: usize> {
    /// Replicates where every target was finite, in replicate order.
    pub draws: Vec<[f64; K]>,
    /// Replicates evaluated (cancellation stops early).
    pub attempted: u32,
    /// Cooperative cancellation stopped the loop.
    pub cancelled: bool,
    /// Block length in lag-aligned rows.
    pub block_length: usize,
    /// Lag-aligned rows resampled.
    pub rows: usize,
}

impl<const K: usize> RowBlockBootstrap<K> {
    /// Raw replicate SD for target `k` under the shared failure policy (≥ 2
    /// successes and at most half of attempted replicates failed). Every target
    /// shares the same replicates.
    #[must_use]
    pub fn raw_se_result(&self, k: usize) -> BootstrapSeResult {
        let values: Vec<f64> = self.draws.iter().map(|draw| draw[k]).collect();
        finalize_bootstrap_se_ex(&values, self.attempted, self.cancelled, false)
    }

    /// Published SE for target `k`: [`Self::raw_se_result`] scaled by
    /// [`fixed_b_scale`], so that `estimate ± z·SE` carries the fixed-b critical
    /// value for this block length.
    #[must_use]
    pub fn se_result(&self, k: usize) -> BootstrapSeResult {
        let mut result = self.raw_se_result(k);
        let scale = fixed_b_scale(self.block_length, self.rows);
        result.se = result.se.map(|se| se * scale);
        result
    }
}

/// Fixed-b correction for a circular-block SE: `cv(ℓ/n) / z`, with `cv` the
/// Kiefer–Vogelsang (2005) Bartlett-kernel fixed-b critical value for a
/// two-sided 95% test, `cv(b) = 1.96 + 2.9694b + 0.4160b² − 0.5324b³`.
///
/// The circular-block variance of a smooth statistic is asymptotically the
/// Bartlett-kernel long-run variance with bandwidth `ℓ` (Künsch 1989), and a
/// `ℓ = ⌈n^{1/3}⌉` bandwidth leaves both kernel bias and estimation noise that a
/// standard-normal critical value ignores: uncorrected nominal-90% intervals
/// measured 2–7 points low in the 1.9 calibration. Scaling the SE by this ratio
/// makes `estimate ± z·SE` the fixed-b interval; the 95% ratio is used so the
/// correction is not smaller than the 90% one (≈ `1 + 1.33ℓ/n`).
#[must_use]
pub fn fixed_b_scale(block_length: usize, rows: usize) -> f64 {
    if rows == 0 {
        return 1.0;
    }
    let b = (block_length as f64 / rows as f64).clamp(0.0, 1.0);
    (1.96 + 2.9694 * b + 0.4160 * b * b - 0.5324 * b * b * b) / 1.96
}

/// Resample `rows` lag-aligned rows in circular blocks of
/// `circular_block_length(structural_span, rows)` and evaluate `estimate` on each
/// replicate's row map (`row_src[r]` = source row for replicate row `r`).
///
/// `estimate` returns `None` for a replicate that cannot be fit; that replicate
/// counts as failed for every target, so all `K` targets always come from the
/// same replicates. Replicate `r` draws from `ctx.rng.stream(stream_base + r)`.
pub fn row_block_bootstrap<const K: usize>(
    rows: usize,
    structural_span: usize,
    replicates: u32,
    stream_base: u64,
    ctx: &ExecutionContext,
    estimate: impl FnMut(&[usize]) -> Option<[f64; K]>,
) -> RowBlockBootstrap<K> {
    let block_length = circular_block_length(structural_span, rows);
    let (draws, attempted, cancelled) =
        block_replicates(rows, block_length, replicates, stream_base, ctx, estimate);
    RowBlockBootstrap { draws, attempted, cancelled, block_length, rows }
}

/// Outcome of a shared circular-block bootstrap of a run-time number of targets
/// (e.g. every atom of a structural mixture plus derived quantities).
#[derive(Clone, Debug)]
pub struct RowBlockDraws {
    /// Replicates where every target was finite, in replicate order; each has
    /// the same length.
    pub draws: Vec<Vec<f64>>,
    /// Replicates evaluated (cancellation stops early).
    pub attempted: u32,
    /// Cooperative cancellation stopped the loop.
    pub cancelled: bool,
    /// Block length in lag-aligned rows.
    pub block_length: usize,
    /// Lag-aligned rows resampled.
    pub rows: usize,
}

impl RowBlockDraws {
    /// The [`fixed_b_scale`] for this block length and row count.
    #[must_use]
    pub fn fixed_b(&self) -> f64 {
        fixed_b_scale(self.block_length, self.rows)
    }

    /// Replicate values of target `k`.
    #[must_use]
    pub fn column(&self, k: usize) -> Vec<f64> {
        self.draws.iter().map(|draw| draw[k]).collect()
    }

    /// Raw replicate SD of target `k` under the shared failure policy (≥ 2
    /// successes and at most half of attempted replicates failed).
    #[must_use]
    pub fn raw_se_result(&self, k: usize) -> BootstrapSeResult {
        finalize_bootstrap_se_ex(&self.column(k), self.attempted, self.cancelled, false)
    }

    /// Published SE of target `k`: [`Self::raw_se_result`] scaled by [`Self::fixed_b`].
    #[must_use]
    pub fn se_result(&self, k: usize) -> BootstrapSeResult {
        let mut result = self.raw_se_result(k);
        let scale = self.fixed_b();
        result.se = result.se.map(|se| se * scale);
        result
    }
}

/// Lag-aligned rows of one prepared temporal design: design row `r` reads the
/// lag window ending at series time `first_time + r`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct AlignedRows {
    /// Series time of design row 0 (the design's maximal lag, plus any split offset).
    pub first_time: usize,
    /// Design rows.
    pub rows: usize,
}

/// Series times every design can evaluate, as `(start, len)`:
/// `start = max first_time` and `start + len = min (first_time + rows)`.
#[must_use]
pub fn common_time_window(designs: &[AlignedRows]) -> Option<(usize, usize)> {
    let start = designs.iter().map(|d| d.first_time).max()?;
    let end = designs.iter().map(|d| d.first_time + d.rows).min()?;
    (end > start).then(|| (start, end - start))
}

/// Shared circular-block bootstrap over several lag-aligned designs of one series.
///
/// Blocks of consecutive *series times* are resampled over the window where the
/// maximal lag window across all designs is available ([`common_time_window`]).
/// Each replicate hands `estimate` one row map per design (`maps[g][r]` = row of
/// design `g` for replicate row `r`), all pointing at the **same** resampled
/// times. Every design row therefore keeps its intact lag window from the
/// original series, and every design is refit on the same resampled index set.
/// Rebuilding lags on a resampled raw series instead pairs, at every block
/// junction and at the circular wrap, an outcome with regressors from an
/// unrelated block; in the 1.9 calibration that inflated multi-atom SEs 1.35–2.3×.
///
/// `block_length` is in series times (callers use
/// [`antecedent_data::circular_block_length`] of the structural span and the
/// window length); publish SEs through [`RowBlockDraws::se_result`] so they carry
/// the [`fixed_b_scale`] of the resampled window. Returns `None` when the designs
/// share no time.
pub fn aligned_block_bootstrap(
    designs: &[AlignedRows],
    block_length: usize,
    replicates: u32,
    stream_base: u64,
    ctx: &ExecutionContext,
    mut estimate: impl FnMut(&[Vec<usize>]) -> Option<Vec<f64>>,
) -> Option<RowBlockDraws> {
    let (start, len) = common_time_window(designs)?;
    let block_length = block_length.clamp(1, len);
    let mut maps: Vec<Vec<usize>> = designs.iter().map(|_| vec![0usize; len]).collect();
    let (draws, attempted, cancelled) =
        block_replicates(len, block_length, replicates, stream_base, ctx, |src| {
            for (design, map) in designs.iter().zip(maps.iter_mut()) {
                let offset = start - design.first_time;
                for (dst, &j) in map.iter_mut().zip(src) {
                    *dst = offset + j;
                }
            }
            estimate(&maps)
        });
    Some(RowBlockDraws { draws, attempted, cancelled, block_length, rows: len })
}

/// [`row_block_bootstrap`] with a run-time target count and an explicit block length.
pub fn row_block_bootstrap_vec(
    rows: usize,
    block_length: usize,
    replicates: u32,
    stream_base: u64,
    ctx: &ExecutionContext,
    estimate: impl FnMut(&[usize]) -> Option<Vec<f64>>,
) -> RowBlockDraws {
    let block_length = block_length.clamp(1, rows.max(1));
    let (draws, attempted, cancelled) =
        block_replicates(rows, block_length, replicates, stream_base, ctx, estimate);
    RowBlockDraws { draws, attempted, cancelled, block_length, rows }
}

/// The replicate loop shared by every row-block bootstrap: circular blocks of
/// `block_length` over `0..rows`, one RNG stream per replicate, and a replicate
/// with any non-finite target dropped for every target.
fn block_replicates<T: AsRef<[f64]>>(
    rows: usize,
    block_length: usize,
    replicates: u32,
    stream_base: u64,
    ctx: &ExecutionContext,
    mut estimate: impl FnMut(&[usize]) -> Option<T>,
) -> (Vec<T>, u32, bool) {
    let mut draws = Vec::with_capacity(replicates as usize);
    let (mut attempted, mut cancelled) = (0u32, false);
    if rows == 0 {
        return (draws, attempted, cancelled);
    }
    let plan = ResamplingPlan::CircularBlock { length: block_length };
    let mut scratch = Vec::with_capacity(rows);
    let mut row_src = vec![0usize; rows];
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            cancelled = true;
            break;
        }
        attempted += 1;
        let mut rng = ctx.rng.stream(stream_base.wrapping_add(u64::from(replicate)));
        if fill_resample_indexes(plan, rows, &mut rng, &mut scratch).is_err() {
            continue;
        }
        for (dst, &src) in row_src.iter_mut().zip(&scratch) {
            *dst = src as usize;
        }
        if let Some(values) =
            estimate(&row_src).filter(|v| v.as_ref().iter().all(|x| x.is_finite()))
        {
            draws.push(values);
        }
        if let Some(progress) = &ctx.progress {
            progress.report(f64::from(replicate + 1) / f64::from(replicates), "block bootstrap");
        }
    }
    (draws, attempted, cancelled)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shared_replicates_drop_jointly_and_keep_blocks_contiguous() {
        let values: Vec<f64> = (0..64).map(f64::from).collect();
        let ctx = ExecutionContext::for_tests(5);
        let mut calls = 0;
        let boot = row_block_bootstrap::<2>(64, 2, 12, 77, &ctx, |rows| {
            calls += 1;
            // Every row map is a concatenation of circular runs of length 4.
            for block in rows.chunks(4) {
                for pair in block.windows(2) {
                    assert_eq!(pair[1], (pair[0] + 1) % 64, "block broke contiguity");
                }
            }
            let mean = rows.iter().map(|&r| values[r]).sum::<f64>() / 64.0;
            // Every third replicate fails one target: it must be dropped for both.
            Some([mean, if calls % 3 == 0 { f64::NAN } else { 2.0 * mean }])
        });
        assert_eq!(boot.block_length, 4, "⌈64^(1/3)⌉ = 4 dominates a span of 2");
        assert_eq!(boot.attempted, 12);
        assert_eq!(boot.draws.len(), 8);
        let a = boot.se_result(0);
        let b = boot.se_result(1);
        assert_eq!(a.replicates_ok, 8);
        assert_eq!(a.replicates_failed, 4);
        assert!((b.se.unwrap() - 2.0 * a.se.unwrap()).abs() < 1e-9);
    }

    #[test]
    fn aligned_bootstrap_points_every_design_at_the_same_times() {
        // Design A's rows start at time 1 (lag 1), design B's at time 3 (lag 3).
        let designs =
            [AlignedRows { first_time: 1, rows: 99 }, AlignedRows { first_time: 3, rows: 97 }];
        assert_eq!(common_time_window(&designs), Some((3, 97)));
        let ctx = ExecutionContext::for_tests(3);
        let boot = aligned_block_bootstrap(&designs, 5, 6, 11, &ctx, |maps| {
            assert_eq!(maps.len(), 2);
            for (a, b) in maps[0].iter().zip(&maps[1]) {
                // Same series time: row r of A is time r + 1, row r of B is time r + 3.
                assert_eq!(a + 1, b + 3);
                assert!(*a < 99 && *b < 97);
            }
            for block in maps[1].chunks(5) {
                for pair in block.windows(2) {
                    assert_eq!(pair[1], (pair[0] + 1) % 97, "blocks are consecutive times");
                }
            }
            Some(vec![1.0, 2.0])
        })
        .unwrap();
        assert_eq!((boot.rows, boot.block_length, boot.draws.len()), (97, 5, 6));
        assert!((boot.fixed_b() - fixed_b_scale(5, 97)).abs() < 1e-15);
        assert!(
            aligned_block_bootstrap(
                &[AlignedRows { first_time: 0, rows: 3 }, AlignedRows { first_time: 5, rows: 3 }],
                2,
                1,
                0,
                &ctx,
                |_| None,
            )
            .is_none(),
            "disjoint windows share no time"
        );
    }

    #[test]
    fn dependence_block_length_keeps_the_rule_for_weak_and_grows_for_persistent_scores() {
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let iid: Vec<f64> = (0..400).map(|_| uniform()).collect();
        let mut level = 0.0;
        let persistent: Vec<f64> = (0..400)
            .map(|_| {
                level = 0.85 * level + uniform();
                level
            })
            .collect();
        let rule = circular_block_length(2, 400);
        assert_eq!(dependence_block_length(2, 400, &[&iid]), rule);
        let long = dependence_block_length(2, 400, &[&iid, &persistent]);
        assert!(long > rule && long <= 400 / 3, "persistent score must lengthen blocks: {long}");
        assert!(politis_white_block_length(&[1.0; 4]).is_none());
    }

    #[test]
    fn effective_rows_discounts_positive_lag_one_dependence() {
        let alternating: Vec<f64> = (0..100).map(|t| if t % 2 == 0 { 1.0 } else { -1.0 }).collect();
        assert!((effective_rows(&alternating) - 100.0).abs() < 1e-12, "negative r1 is floored");
        let mut state = 0.0_f64;
        let persistent: Vec<f64> = (0..400_u32)
            .map(|t| {
                state = 0.8 * state + f64::from((t * 7919) % 13) - 6.0;
                state
            })
            .collect();
        let n_eff = effective_rows(&persistent);
        assert!(n_eff < 200.0 && n_eff > 10.0, "n_eff={n_eff}");
        assert!((effective_rows(&[0.0; 10]) - 10.0).abs() < 1e-12);
    }

    #[test]
    fn fixed_b_scale_matches_kiefer_vogelsang_polynomial() {
        assert!((fixed_b_scale(6, 159) - 1.057_458).abs() < 1e-5);
        assert!((fixed_b_scale(0, 10) - 1.0).abs() < 1e-12);
        assert!(fixed_b_scale(8, 399) < fixed_b_scale(4, 59));
    }
}
