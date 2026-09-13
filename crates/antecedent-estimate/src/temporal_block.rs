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
//! `max(structural_span, ⌈n^{1/3}⌉)`, capped at `n`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

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
    mut estimate: impl FnMut(&[usize]) -> Option<[f64; K]>,
) -> RowBlockBootstrap<K> {
    let block_length = circular_block_length(structural_span, rows);
    let plan = ResamplingPlan::CircularBlock { length: block_length };
    let mut out = RowBlockBootstrap {
        draws: Vec::with_capacity(replicates as usize),
        attempted: 0,
        cancelled: false,
        block_length,
        rows,
    };
    if rows == 0 {
        return out;
    }
    let mut scratch = Vec::with_capacity(rows);
    let mut row_src = vec![0usize; rows];
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            out.cancelled = true;
            break;
        }
        out.attempted += 1;
        let mut rng = ctx.rng.stream(stream_base.wrapping_add(u64::from(replicate)));
        if fill_resample_indexes(plan, rows, &mut rng, &mut scratch).is_err() {
            continue;
        }
        for (dst, &src) in row_src.iter_mut().zip(&scratch) {
            *dst = src as usize;
        }
        if let Some(values) = estimate(&row_src).filter(|v| v.iter().all(|x| x.is_finite())) {
            out.draws.push(values);
        }
        if let Some(progress) = &ctx.progress {
            progress.report(f64::from(replicate + 1) / f64::from(replicates), "block bootstrap");
        }
    }
    out
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
