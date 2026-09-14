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
//! lengthens it when an estimating score — of the target or of any nuisance
//! coefficient ([`normal_equation_scores`]) — is persistently dependent.
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

/// The circular-block SE families that carry their own short-series threshold.
///
/// Each family resamples lag-aligned rows the same way but estimates a
/// different functional, so the effective-row count below which its interval
/// under-covers differs. [`Self::min_effective_rows`] documents the measured
/// provenance of every threshold.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CircularBlockFamily {
    /// One lag-aligned adjustment regression: `TemporalDag` Pulse and
    /// single-step Sustained (the statistic reads the treatment-coefficient score).
    SingleWindow,
    /// Temporal mediation: Total, Direct and Mediated from one shared replicate
    /// (the statistic is the smallest over the three contrast scores).
    Mediation,
    /// Multi-step Sustained sequential g-computation on one `TemporalDag` (the
    /// statistic reads the contrast's influence).
    Sequential,
    /// Frozen-weight mixtures over several temporal atoms (class envelopes and
    /// DBN posteriors) resampled on one shared replicate; the statistic is the
    /// smallest over every atom's score and the weighted mixture score.
    Mixture,
}

impl CircularBlockFamily {
    /// Threshold on [`score_effective_rows`] of the family's estimating scores
    /// below which the interval carries a `short_series` warning.
    ///
    /// Provenance: `crates/antecedent/tests/v19_short_series_measurement.rs`,
    /// table in `docs/short-series-thresholds.md` (2000 replicates per cell,
    /// nominal 0.90, AR(1) ρ ∈ {0.5, 0.8, 0.9, 0.95}, n from 40 to 1600). Each
    /// family is measured on a short-memory score and on a persistent one
    /// (AR(1) treatment as well as residual), because the effective-row count
    /// alone does not separate them: the short-memory designs cover nominally
    /// at every measured n, the persistent ones fail at the same counts. A
    /// threshold is the smallest multiple of 5 at which every cell covering
    /// below 0.855 (the 400-replicate gate band's lower edge) warns on at least
    /// 90% of its replicates, taking the larger of that sweep and an
    /// independent 1000-replicate replication.
    ///
    /// * `SingleWindow` 45 (sweep 45, replication 45): the AR(1)-treatment
    ///   Pulse covers 0.765–0.853 at n = 40 (ρ ≥ 0.8), n = 60–160 (ρ ≥ 0.9)
    ///   and n = 400 (ρ = 0.95); each warns on ≥ 92% of replicates. The
    ///   MA(3)-treatment Pulse covers 0.868–0.911 everywhere and is quiet from
    ///   n = 160 (≤ 6% warned); at n ≤ 100 it still warns on part of the
    ///   replicates.
    /// * `Mediation` 35 (30, 35): with an AR(1) treatment Total and Direct
    ///   cover 0.755–0.850 at n = 40 (ρ ≥ 0.8), n = 60 (ρ ≥ 0.9) and
    ///   n = 100–160 (ρ = 0.95), each warned on ≥ 97%; the MA(3) design covers
    ///   0.875–0.923 and is quiet from n = 100 (≤ 15%).
    /// * `Sequential` 40 (35, 40): the AR(1)-treatment multi-step Sustained
    ///   covers 0.748–0.855 at n ≤ 60 (ρ ≥ 0.8), n = 100 (ρ = 0.95) and n = 400
    ///   (ρ = 0.95), each warned on ≥ 98%; the confounded-DAG design covers
    ///   0.888–0.919 and is quiet from n = 100 (≤ 29%).
    /// * `Mixture` 155 (155, 155): SE-driven failures (the six-completion
    ///   `TemporalPag` at ρ ≥ 0.9, n ≤ 160: 0.803–0.850) warn from 30. The rest
    ///   is bias-driven: a non-causal completion that omits a persistent
    ///   confounder is biased by 0.2–0.8 of its SD at ρ ≥ 0.9 (`TemporalCpdag`
    ///   0.665–0.850 up to n = 400; DBN 0.823–0.841 at n ≤ 60). Only that
    ///   completion's score shows it, as a weak slowly decaying component the
    ///   block-length reading sees at ~80 effective rows (ρ = 0.95, n = 400,
    ///   warned on 91%); the price is warnings on mixtures that cover nominally
    ///   below about n = 400.
    #[must_use]
    pub const fn min_effective_rows(self) -> f64 {
        match self {
            Self::SingleWindow => 45.0,
            Self::Mediation => 35.0,
            Self::Sequential => 40.0,
            Self::Mixture => 155.0,
        }
    }

    /// Short name used in diagnostics.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::SingleWindow => "single-window adjustment",
            Self::Mediation => "temporal mediation",
            Self::Sequential => "multi-step sequential",
            Self::Mixture => "multi-atom mixture",
        }
    }

    /// Whether `effective_rows` (NaN when the score is unavailable) falls short
    /// of [`Self::min_effective_rows`].
    #[must_use]
    pub fn is_short_series(self, effective_rows: f64) -> bool {
        effective_rows.is_nan() || effective_rows < self.min_effective_rows()
    }
}

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

/// Effective rows of `scores` as seen by a circular block of `block_length`:
/// `n·γ̂₀ / ĝ_b`, where `ĝ_b = γ̂₀ + 2 Σ_{k<b} (1 − k/b) γ̂_k` is the Bartlett
/// long-run variance at the block length — the variance a circular block of
/// that length attributes to the scores' mean. Unlike [`effective_rows`] (an
/// AR(1) reading of the lag-1 autocorrelation) it sees dependence that is weak
/// at lag 1 but slowly decaying. Capped at `n`; a constant series returns `n`;
/// non-finite scores return `NaN`.
#[must_use]
pub fn block_effective_rows(scores: &[f64], block_length: usize) -> f64 {
    let n = scores.len();
    if scores.iter().any(|s| !s.is_finite()) {
        return f64::NAN;
    }
    if n < 3 {
        return n as f64;
    }
    let nf = n as f64;
    let mean = scores.iter().sum::<f64>() / nf;
    let centered: Vec<f64> = scores.iter().map(|s| s - mean).collect();
    let autocov = |k: usize| -> f64 {
        centered[..n - k].iter().zip(&centered[k..]).map(|(a, b)| a * b).sum::<f64>() / nf
    };
    let gamma0 = autocov(0);
    if gamma0 <= 0.0 {
        return nf;
    }
    let b = block_length.clamp(1, n);
    let long_run =
        gamma0 + 2.0 * (1..b).map(|k| (1.0 - k as f64 / b as f64) * autocov(k)).sum::<f64>();
    if long_run <= 0.0 {
        return nf;
    }
    (nf * gamma0 / long_run).min(nf)
}

/// The short-series statistic of a circular-block interval: the smallest, over
/// every estimating-score series in `scores` (one per atom or contrast, plus
/// the weighted mixture score), of the two effective-row readings — the lag-1
/// AR(1) reading [`effective_rows`] and the block-length Bartlett reading
/// [`block_effective_rows`] at `block_length`. `NaN` when no series yields a
/// finite reading.
///
/// The two readings fail differently. An AR(1)-like score (persistent
/// regressor × persistent residual) is caught sharply by `r₁`; a mixture whose
/// weighted score is dominated by a nearly iid atom, while another atom's score
/// carries a weak but slowly decaying component (an omitted persistent
/// confounder in a non-causal completion), looks independent at lag 1 but not
/// to the block.
#[must_use]
pub fn score_effective_rows(scores: &[&[f64]], block_length: usize) -> f64 {
    scores
        .iter()
        .flat_map(|s| [effective_rows(s), block_effective_rows(s, block_length)])
        .filter(|v| v.is_finite())
        .fold(f64::NAN, f64::min)
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
/// estimating-score series on the same rows: every atom's or contrast's
/// influence, their weighted sum, and every fitted regression's
/// [`normal_equation_scores`]).
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
/// The nuisance scores matter for the same reason: a short-memory slope
/// influence over a persistent residual (the intercept's score) left SE/SD at
/// 0.94 on a horizon-2 Pulse until the residual sized the blocks.
/// Weakly dependent scores keep the rule.
#[must_use]
pub fn dependence_block_length(structural_span: usize, rows: usize, scores: &[&[f64]]) -> usize {
    let rule = circular_block_length(structural_span, rows);
    let pw = scores.iter().filter_map(|s| politis_white_block_length(s)).max().unwrap_or(0);
    let testing = (pw as f64 * (rows as f64).powf(1.0 / 6.0)).ceil() as usize;
    rule.max(testing.min(rows / 3)).min(rows.max(1))
}

/// OLS normal-equation scores `x_{tj} · ê_t` of a column-major `rows × cols`
/// design, one series per column (an intercept column contributes the residual
/// series itself); `None` when the least-squares fit fails.
///
/// These are the estimating equations of *every* fitted coefficient, nuisance
/// ones included, for [`dependence_block_length`]. The targeted coefficient's
/// influence can be nearly uncorrelated lag by lag while the residual — the
/// intercept's score — is strongly persistent. The replicate slope then depends
/// on how well a block reproduces the persistent residual level: in the 1.9
/// calibration (AR(1) ρ = 0.9 residual, n = 400, a short-memory regressor) blocks
/// sized on the slope influence alone gave SE/SD 0.94–0.96 and coverage
/// 0.850–0.873.
#[must_use]
pub fn normal_equation_scores(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    outcome: &[f64],
) -> Option<Vec<Vec<f64>>> {
    use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};
    if rows == 0 || cols == 0 || matrix.len() < rows * cols || outcome.len() != rows {
        return None;
    }
    let fit = FaerBackend
        .least_squares(
            &matrix[..rows * cols],
            rows,
            cols,
            outcome,
            &mut LeastSquaresWorkspace::default(),
        )
        .ok()?;
    let residuals = fit.residuals;
    Some(
        (0..cols)
            .map(|j| {
                matrix[j * rows..(j + 1) * rows]
                    .iter()
                    .zip(&residuals)
                    .map(|(x, e)| x * e)
                    .collect()
            })
            .collect(),
    )
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

/// Fixed-b correction for a circular-block SE: `cv_95(ℓ/n) / 1.96`, with `cv_95`
/// the Kiefer–Vogelsang (2005) Bartlett-kernel fixed-b critical value for a
/// two-sided 95% test, `cv(b) = 1.96 + 2.9694b + 0.4160b² − 0.5324b³`.
///
/// The circular-block variance of a smooth statistic is asymptotically the
/// Bartlett-kernel long-run variance with bandwidth `ℓ` (Künsch 1989), and a
/// `ℓ = ⌈n^{1/3}⌉` bandwidth leaves both kernel bias and estimation noise that a
/// standard-normal critical value ignores: uncorrected nominal-90% intervals
/// measured 2–7 points low in the 1.9 calibration. Scaling the SE by this ratio
/// makes `estimate ± z_{0.95}·SE` the 95% fixed-b interval. Published 90%
/// scalar intervals still use this 95% ratio (not `cv_90 / z_{0.90}`), so they
/// are 90% normal intervals around a 95%-inflated SE — conservative relative
/// to the 90% KV polynomial (≈ `1 + 1.33ℓ/n`). The 1.9 gate measured that
/// construction; do not silently swap in the 90% ratio.
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
    fn normal_equation_scores_carry_the_persistent_residual_into_the_block_length() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        let n = 400;
        // Short-memory regressor, strongly persistent residual.
        let x: Vec<f64> = (0..n).map(|_| uniform()).collect();
        let mut level = 0.0;
        let e: Vec<f64> = (0..n)
            .map(|_| {
                level = 0.95 * level + uniform();
                level
            })
            .collect();
        let y: Vec<f64> = x.iter().zip(&e).map(|(x, e)| 0.8 * x + e).collect();
        let mut matrix = vec![1.0; n];
        matrix.extend_from_slice(&x);
        let scores = normal_equation_scores(&matrix, n, 2, &y).unwrap();
        assert_eq!(scores.len(), 2);
        // Intercept column: the residuals themselves, which sum to zero.
        assert!(scores[0].iter().sum::<f64>().abs() < 1e-8);
        // Normal equations: every column's score sums to zero at the OLS fit.
        assert!(scores[1].iter().sum::<f64>().abs() < 1e-8);
        let slope_only = dependence_block_length(2, n, &[&scores[1]]);
        let with_residual = dependence_block_length(2, n, &[&scores[1], &scores[0]]);
        assert!(
            with_residual > slope_only,
            "the persistent residual must lengthen blocks: {slope_only} -> {with_residual}"
        );
        assert!(normal_equation_scores(&matrix, n, 2, &y[..10]).is_none());
    }

    /// Deterministic AR(0.8)-like score of length `n` (the series of
    /// `effective_rows_discounts_positive_lag_one_dependence`).
    fn persistent_score(n: u32) -> Vec<f64> {
        let mut state = 0.0_f64;
        (0..n)
            .map(|t| {
                state = 0.8 * state + f64::from((t * 7919) % 13) - 6.0;
                state
            })
            .collect()
    }

    #[test]
    fn block_effective_rows_sees_slowly_decaying_dependence() {
        // An alternating score (lag-1 correlation −1) plus a weak slow wave: the
        // lag-1 reading sees no positive dependence, the block reading does.
        let n = 400;
        let scores: Vec<f64> = (0..n)
            .map(|t| {
                let wave = (2.0 * std::f64::consts::PI * f64::from(t) / 100.0).sin();
                let alternating = if t % 2 == 0 { 1.0 } else { -1.0 };
                alternating + 0.6 * wave
            })
            .collect();
        let lag_one = effective_rows(&scores);
        let block = block_effective_rows(&scores, 40);
        assert!((lag_one - 400.0).abs() < 1e-9, "negative r1 is floored: {lag_one}");
        assert!(block < 200.0, "the slow wave inflates the block variance: {block}");
        assert!((score_effective_rows(&[&scores], 40) - block).abs() < 1e-12);
        // Uncorrelated-looking and constant series keep n; non-finite scores are skipped.
        assert!((block_effective_rows(&[3.0; 10], 4) - 10.0).abs() < 1e-12);
        assert!(block_effective_rows(&[1.0, f64::NAN, 2.0], 2).is_nan());
        assert!(score_effective_rows(&[&[1.0, f64::NAN, 2.0]], 2).is_nan());
        // A single series' two readings, and the smallest over several series.
        let persistent = persistent_score(400);
        let combined = score_effective_rows(&[&scores, &persistent], 40);
        assert!(
            (combined
                - effective_rows(&persistent)
                    .min(block_effective_rows(&persistent, 40))
                    .min(block))
            .abs()
                < 1e-12
        );
    }

    /// The per-family thresholds on one fixed series whose statistic sits
    /// between them: the families disagree exactly as their thresholds say.
    #[test]
    #[allow(clippy::float_cmp)]
    fn circular_block_family_thresholds_on_a_fixed_series() {
        use CircularBlockFamily::{Mediation, Mixture, Sequential, SingleWindow};
        assert_eq!(SingleWindow.min_effective_rows(), 45.0);
        assert_eq!(Mediation.min_effective_rows(), 35.0);
        assert_eq!(Sequential.min_effective_rows(), 40.0);
        assert_eq!(Mixture.min_effective_rows(), 155.0);
        let scores = persistent_score(135);
        let statistic = score_effective_rows(&[&scores], 30);
        assert!((40.0..45.0).contains(&statistic), "statistic {statistic}");
        assert!(SingleWindow.is_short_series(statistic));
        assert!(!Mediation.is_short_series(statistic));
        assert!(!Sequential.is_short_series(statistic));
        assert!(Mixture.is_short_series(statistic));
        for family in [SingleWindow, Mediation, Sequential, Mixture] {
            assert!(family.is_short_series(f64::NAN), "{family:?}: an unknown statistic warns");
            assert!(!family.is_short_series(family.min_effective_rows()), "{family:?}");
            assert!(family.is_short_series(family.min_effective_rows() - 1e-9), "{family:?}");
        }
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
