//! Circular-block bootstrap over the lag-aligned rows of one series.
//!
//! Lag-aligned rows of one series are serially dependent (overlapping lag
//! windows, autocorrelated residuals, MA(h−1) errors at horizon h), so iid row
//! resampling understates uncertainty. This bootstrap resamples circular blocks
//! of *consecutive* lag-aligned rows (Künsch 1989's blocks-of-blocks form): every
//! replicate row keeps its own intact lag window, and consecutive rows inside a
//! block keep their joint dependence. Resampling the raw series instead and
//! rebuilding lags would pair, at every block junction, an outcome with lagged
//! regressors from an unrelated block; calibration showed that inflated a
//! strong-effect SE by ~10% under iid noise and pushed a nominal-90% temporal
//! mediation Total interval to 0.96 coverage.
//!
//! The block length comes from [`antecedent_data::circular_block_length`]:
//! `max(structural_span, ⌈n^{1/3}⌉)`, capped at `n`; [`dependence_block_length`]
//! lengthens it when an estimating score — of the target or of any nuisance
//! coefficient ([`normal_equation_scores`]) — is persistently dependent.
//!
//! Every published SE is the replicate SD times the circular fixed-b factor
//! ([`circular_fixed_b_scale`]) and the Bartlett kernel-bias factor of the
//! interval's estimating scores ([`kernel_bias_scale`]).
//!
//! Several designs on one series (the atoms of a temporal class / DBN mixture)
//! share one replicate through [`aligned_block_bootstrap`]: blocks of consecutive
//! series times over the window every design can evaluate, each design refit on
//! its own lag-aligned rows at those times.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        clippy::cast_sign_loss,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use antecedent_core::ExecutionContext;
use antecedent_core::StreamDomain;
use antecedent_data::{circular_block_length, fill_circular_block_indexes};

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
    /// Family of a shared mediation bootstrap over `atoms` frozen-weight atoms: one atom is
    /// plain temporal mediation, several are a mixture resampled on one shared replicate.
    #[must_use]
    pub const fn for_mediation_atoms(atoms: usize) -> Self {
        if atoms == 1 { Self::Mediation } else { Self::Mixture }
    }

    /// Threshold on [`score_effective_rows`] of the family's estimating scores
    /// below which the interval carries a `short_series` warning.
    ///
    /// Provenance: `crates/antecedent/tests/v19_short_series_measurement.rs`,
    /// table in `docs/short-series-thresholds.md` (2000 replicates per cell, the
    /// nominal-0.90 interval built from the published SE, AR(1) ρ ∈ {0.5, 0.8, 0.9,
    /// 0.95}, n from 40 to 1600). Each family is measured on a short-memory score and on a persistent one
    /// (AR(1) treatment as well as residual), because the effective-row count
    /// alone does not separate them: the short-memory designs cover nominally
    /// at every measured n, the persistent ones fail at the same counts. A
    /// threshold is the smallest multiple of 5 at which every cell covering
    /// below 0.855 (the 400-replicate gate band's lower edge) warns on at least
    /// 90% of its replicates, taking the larger of that sweep and an
    /// independent 1000-replicate replication.
    ///
    /// Measured on the SE construction before the Bartlett kernel-bias factor
    /// ([`kernel_bias_scale`]; blocks sized on every estimating score,
    /// circular-Bartlett fixed-b factor). That factor only widens intervals;
    /// re-measured with it, every cell below 0.855 still warns on at least 91%
    /// of its replicates and the thresholds stand. The factor has since taken the
    /// Kendall-corrected AR(1) and BIC AR(q) model of [`crate::ar_kernel`], which is
    /// never smaller on the same score; the warning reads effective rows, not the
    /// factor, so the warnings are unchanged and only the coverage of the affected
    /// cells can rise:
    ///
    /// * `SingleWindow` 45 (sweep 45, replication 40): the AR(1)-treatment
    ///   Pulse covers 0.750–0.854 at n = 40–60 (ρ ≥ 0.8), n = 100–160 (ρ ≥ 0.9)
    ///   and n = 400 (ρ = 0.95); each warns on ≥ 91% of replicates. The
    ///   MA(3)-treatment Pulse covers 0.871–0.919 everywhere and is quiet from
    ///   n = 160 (≤ 6% warned); at n ≤ 100 it still warns on part of the
    ///   replicates.
    /// * `Mediation` 40 (35, 40): with an AR(1) treatment Total and Direct
    ///   cover 0.772–0.854 at n = 40 (ρ ≥ 0.8), n = 60–100 (ρ ≥ 0.9) and
    ///   n = 160 (ρ = 0.95), each warned on ≥ 98%; the MA(3) design covers
    ///   0.880–0.928 and is quiet from n = 160 (≤ 2%; ≤ 28% at n = 100).
    /// * `Sequential` 40 (25, 30): the AR(1)-treatment multi-step Sustained
    ///   covers 0.764–0.853 at n = 40 (ρ ≥ 0.8), n = 60–100 (ρ ≥ 0.9) and
    ///   n = 160 (ρ = 0.95), each warned on every replicate; the confounded-DAG
    ///   design covers 0.887–0.926 and is quiet from n = 160 (≤ 1%; ≤ 29% at
    ///   n = 100). The rule now gives 30; the threshold stays at the 40 an
    ///   earlier measurement set, which warns more, not less.
    /// * `Mixture` 155 (155, 90): SE-driven failures (the six-completion
    ///   `TemporalPag` at ρ ≥ 0.9, n ≤ 160: 0.804–0.852; the DBN at ρ = 0.95,
    ///   n ≤ 60: 0.833–0.851) warn from 30. The rest is bias-driven: a
    ///   non-causal completion that omits a persistent confounder is biased by
    ///   0.2–0.8 of its SD at ρ ≥ 0.9 (`TemporalCpdag` 0.675–0.853 up to
    ///   n = 400). Only that completion's score shows it, as a weak slowly
    ///   decaying component the block-length reading sees at ~80 effective rows
    ///   (ρ = 0.95, n = 400, warned on 91%); the price is warnings on mixtures
    ///   that cover nominally below about n = 400.
    #[must_use]
    pub const fn min_effective_rows(self) -> f64 {
        match self {
            Self::SingleWindow => 45.0,
            Self::Mediation | Self::Sequential => 40.0,
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

    /// Stable `snake_case` tag, used in the calibration dependence label
    /// `circular_block:<tag>`.
    #[must_use]
    pub const fn tag(self) -> &'static str {
        match self {
            Self::SingleWindow => "single_window",
            Self::Mediation => "mediation",
            Self::Sequential => "sequential",
            Self::Mixture => "mixture",
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
    let autocov = |k: usize| score_autocovariance(scores, mean, k, nf);
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

/// Lag-`k` sample autocovariance of `scores` about `mean`, divided by `n`.
/// Same products, same left-to-right sum as a centered copy.
fn score_autocovariance(scores: &[f64], mean: f64, k: usize, n: f64) -> f64 {
    scores[..scores.len() - k]
        .iter()
        .zip(&scores[k..])
        .map(|(a, b)| (a - mean) * (b - mean))
        .sum::<f64>()
        / n
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

/// Variance at or below which a score series is treated as degenerate.
///
/// GEMM dust on a numerically exact score (the orthogonal treatment-column
/// score of the noiseless period-4 dose × horizon fixture) has `γ̂_0 ~ 1e-30`
/// on `aarch64` and 0 on `x86_64`. Dividing `ρ̂ = γ̂_k / γ̂_0` then invents an
/// architecture-specific Politis–White length. Real estimating scores sit
/// many orders above this floor.
const SCORE_VARIANCE_FLOOR: f64 = 1e-20;

/// Data-driven circular-block length for the mean of `scores` (Politis & White
/// 2004, with the Patton, Politis & White 2009 correction for the circular
/// bootstrap): `b = (2 Ĝ² / D̂)^{1/3} n^{1/3}`, `D̂ = (4/3) ĝ(0)²`, from flat-top
/// lag-window estimates with the bandwidth picked by the first run of `K_n`
/// insignificant autocorrelations. Capped at `min(3√n, n/3)`; `None` for fewer
/// than 8 scores or a degenerate series.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    reason = "ceil(sqrt(n)) and ceil(sqrt(log10 n)) of a series length are far below usize::MAX; the block length is clamped to [1, cap] with cap at most n"
)]
#[allow(
    clippy::cast_sign_loss,
    reason = "both values are ceilings of square roots of positive reals, hence non-negative; the block length is clamped to at least 1 before the cast"
)]
pub fn politis_white_block_length(scores: &[f64]) -> Option<usize> {
    let n = scores.len();
    if n < 8 || scores.iter().any(|s| !s.is_finite()) {
        return None;
    }
    let nf = n as f64;
    let mean = scores.iter().sum::<f64>() / nf;
    let autocov = |k: usize| score_autocovariance(scores, mean, k, nf);
    let gamma0 = autocov(0);
    if !gamma0.is_finite() || gamma0 <= SCORE_VARIANCE_FLOOR {
        return None;
    }
    let k_n = 5usize.max(nf.log10().sqrt().ceil() as usize);
    let m_max = (nf.sqrt().ceil() as usize + k_n).min(n - 1);
    let threshold = 2.0 * (nf.log10() / nf).sqrt();
    // Autocorrelations are taken lag by lag as the bandwidth search asks for
    // them: the first run of `K_n` insignificant lags usually ends within a few
    // lags of zero, so the `√n + K_n` lags the cap allows (each an O(n) pass)
    // are only visited for a persistent series. Each lag is the same sum in
    // the same order whenever it is taken, so the length is unchanged.
    let mut rho: Vec<f64> = Vec::with_capacity(2 * k_n + 1);
    let rho_through = |rho: &mut Vec<f64>, lag: usize| {
        while rho.len() <= lag.min(m_max) {
            rho.push(autocov(rho.len()) / gamma0);
        }
    };
    let mut m_hat = m_max.saturating_sub(k_n);
    for m in 0..=m_max.saturating_sub(k_n) {
        rho_through(&mut rho, m + k_n);
        if (1..=k_n).all(|j| rho.get(m + j).is_none_or(|r| r.abs() < threshold)) {
            m_hat = m;
            break;
        }
    }
    let big_m = (2 * m_hat).min(m_max).max(1);
    rho_through(&mut rho, big_m);
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
/// dependence (persistent regressor × persistent residual). On AR(1)
/// ρ = 0.9, n = 400 the `⌈n^{1/3}⌉` rule gave SE/SD 0.82–0.93
/// and coverage 0.81–0.85 on multi-atom mixtures; this length restored SE/SD ≈ 1.
/// The nuisance scores matter for the same reason: a short-memory slope
/// influence over a persistent residual (the intercept's score) left SE/SD at
/// 0.94 on a horizon-2 Pulse until the residual sized the blocks.
/// Weakly dependent scores keep the rule.
#[must_use]
pub fn dependence_block_length(structural_span: usize, rows: usize, scores: &[&[f64]]) -> usize {
    let rule = circular_block_length(structural_span, rows);
    rule.max(testing_block_length(rows, scores).min(rows / 3)).min(rows.max(1))
}

/// Uncapped coverage-rate block length `⌈b_PW · rows^{1/6}⌉` of
/// [`dependence_block_length`], with `b_PW` the largest
/// [`politis_white_block_length`] over `scores`; `0` when no series yields one.
/// Callers cap it at `rows / 3` and never go below their own rule.
#[must_use]
#[allow(
    clippy::cast_possible_truncation,
    reason = "the product ceil(pw * rows^(1/6)) is far below usize::MAX for in-memory series"
)]
#[allow(clippy::cast_sign_loss, reason = "both factors are non-negative")]
pub fn testing_block_length(rows: usize, scores: &[&[f64]]) -> usize {
    let pw = scores.iter().filter_map(|s| politis_white_block_length(s)).max().unwrap_or(0);
    (pw as f64 * (rows as f64).powf(1.0 / 6.0)).ceil() as usize
}

/// OLS normal-equation scores `x_{tj} · ê_t` of a column-major `rows × cols`
/// design, one series per column (an intercept column contributes the residual
/// series itself); `None` when the least-squares fit fails.
///
/// These are the estimating equations of *every* fitted coefficient, nuisance
/// ones included, for [`dependence_block_length`]. The targeted coefficient's
/// influence can be nearly uncorrelated lag by lag while the residual — the
/// intercept's score — is strongly persistent. The replicate slope then depends
/// on how well a block reproduces the persistent residual level: on AR(1)
/// ρ = 0.9 residual, n = 400, a short-memory regressor, blocks
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
    Some(normal_equation_scores_of_residuals(matrix, rows, cols, &fit.residuals))
}

/// [`normal_equation_scores`] for a regression whose OLS `residuals` the caller
/// already holds: the score `x_tj · ê_t` of every column, without a second fit.
///
/// `matrix` is column-major `rows × cols` and `residuals` has length `rows`.
#[must_use]
pub fn normal_equation_scores_of_residuals(
    matrix: &[f64],
    rows: usize,
    cols: usize,
    residuals: &[f64],
) -> Vec<Vec<f64>> {
    (0..cols)
        .map(|j| {
            matrix[j * rows..(j + 1) * rows].iter().zip(residuals).map(|(x, e)| x * e).collect()
        })
        .collect()
}

/// Fixed-b correction for a circular-block SE: `cv_95(ℓ/n) / 1.96`, with `cv_95`
/// the fixed-b two-sided 95% critical value of a mean studentized by the
/// *circular* Bartlett long-run variance at bandwidth `ℓ`,
/// `cv(b) = 1.96 + 2.4389b + 3.7072b² − 2.1055b³`.
///
/// The circular-block bootstrap variance of a sample mean is exactly `1/n` times
/// the circular Bartlett estimator `γ̃₀ + 2 Σ_{k<ℓ} (1 − k/ℓ) γ̃_k` built from
/// circular autocovariances `γ̃_k` (when `ℓ` divides `n`), and asymptotically so
/// for a smooth statistic (Künsch 1989). Its fixed-b mean is `(1 − b)σ²`, below
/// the `(1 − b + b²/3)σ²` of the non-circular Bartlett estimator, so the
/// Kiefer–Vogelsang (2005) polynomial of [`fixed_b_scale`] (non-circular) is too
/// small at long blocks: applied to the circular variance it covers 0.942 at
/// `b = 1/3` for a nominal 95% interval.
///
/// Provenance of the polynomial: simulated fixed-b quantiles of `|√n·x̄ / σ̂|`
/// for iid Gaussian series (`n = 2000`, 10⁶ series, `b` on a 0.01 grid),
/// least-squares cubic in `b` with the intercept held at 1.96, fitted on
/// `b ≤ 0.4` (the dependence-aware length stays at or below `n/3`); the fit is
/// within 0.005 of every simulated quantile there. Beyond `b = 0.4` it is an
/// extrapolation. The same simulation reproduces the non-circular KV values to
/// within 0.03.
///
/// What the scale does and does not do: it replaces the standard-normal
/// critical value by the fixed-b one for the resampled block length, so the
/// interval accounts for the randomness and the `(1 − b)` level of the block
/// variance at that length. It does not repair a block that is too short for
/// the dependence (a long-run variance the block misses); that is what
/// [`dependence_block_length`] and the short-series warning are for.
///
/// Scaling the SE by this ratio makes `estimate ± 1.96·SE` the 95% fixed-b
/// interval, which is the level the result publishes
/// (`REPORTED_SE_INTERVAL_LEVEL`). The short-series sweeps
/// ([`CircularBlockFamily::min_effective_rows`]) measured a different, nominal
/// 0.90, interval built from the same published SE: `estimate ± z_{0.90}·SE`
/// with the 95% ratio (not `cv_90 / z_{0.90}`). In the same simulation that
/// construction covers 0.901–0.913 at nominal 0.90 for `b ≤ 1/3` (conservative,
/// increasingly so at long blocks). The thresholds describe where that
/// construction stops covering; do not silently swap in the 90% ratio.
#[must_use]
pub fn circular_fixed_b_scale(block_length: usize, rows: usize) -> f64 {
    if rows == 0 {
        return 1.0;
    }
    let b = (block_length as f64 / rows as f64).clamp(0.0, 1.0);
    (1.96 + 2.4389 * b + 3.7072 * b * b - 2.1055 * b * b * b) / 1.96
}

/// Bartlett kernel-bias factor of a circular-block SE over `block_length`: the largest
/// [`crate::ar_kernel::kernel_bias_factor`] over `scores` (the interval's estimating
/// scores: the target influence(s) and, for a mixture, the weighted mixture score).
/// `1` for no scores, a non-finite or constant series, or fewer than eight rows.
///
/// A circular block of length `ℓ` reproduces the circular Bartlett long-run
/// variance at bandwidth `ℓ`, whose expectation under an AR(1)(ρ) score is
/// `γ₀[(1 + ρ)/(1 − ρ) − (2/ℓ)·ρ/(1 − ρ)² + …]`, below the long-run variance
/// `γ₀(1 + ρ)/(1 − ρ)` by the Bartlett small-bandwidth bias. The fixed-b
/// critical value ([`circular_fixed_b_scale`]) prices the `(1 − b)` level and
/// the randomness of the block variance under short memory, not this bias,
/// which grows with the score's memory relative to the block: 3% of the SE for
/// an AR(1)(0.81) score at `ℓ = 74` (a persistent treatment and residual at
/// ρ = 0.9, n = 400, where the short-series measurement put the interval at
/// 0.87–0.89 for nominal 0.90), under 1% for short-memory scores. The score is
/// modelled by a Kendall-corrected AR(1) and a BIC-selected AR(q ≤ 4)
/// ([`crate::ar_kernel`]); the response bands read the same factor per cell, so a
/// scalar Pulse or Sustained SE and the matching response-cell SE apply one
/// correction to one score. The factor only widens an interval, and the short-series
/// warning, not this factor, is what says the series is too short for the
/// dependence.
#[must_use]
pub fn kernel_bias_scale(scores: &[&[f64]], block_length: usize) -> f64 {
    let l = block_length.max(1);
    scores.iter().map(|s| crate::ar_kernel::kernel_bias_factor(s, l)).fold(1.0, f64::max)
}

/// Kiefer–Vogelsang (2005) fixed-b correction for a *non-circular* Bartlett
/// long-run variance (a Newey–West HAC) at bandwidth `ℓ` over `rows` rows:
/// `cv_95(ℓ/n) / 1.96` with `cv(b) = 1.96 + 2.9694b + 0.4160b² − 0.5324b³`,
/// the two-sided 95% fixed-b critical value.
///
/// Circular-block bootstrap SEs studentize by the circular Bartlett variance
/// instead and take [`circular_fixed_b_scale`]; the two agree to within 1% of
/// the SE for `b ≤ 0.2` and diverge above (4% at `b = 1/3`).
#[must_use]
pub fn fixed_b_scale(block_length: usize, rows: usize) -> f64 {
    if rows == 0 {
        return 1.0;
    }
    let b = (block_length as f64 / rows as f64).clamp(0.0, 1.0);
    (1.96 + 2.9694 * b + 0.4160 * b * b - 0.5324 * b * b * b) / 1.96
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
    /// Bartlett kernel-bias factor of the estimating scores at
    /// [`Self::block_length`] ([`kernel_bias_scale`]); `1` until the caller
    /// sets it from its scores ([`Self::with_kernel_bias`]).
    pub kernel_bias: f64,
}

impl RowBlockDraws {
    /// The [`circular_fixed_b_scale`] for this block length and row count.
    #[must_use]
    pub fn fixed_b(&self) -> f64 {
        circular_fixed_b_scale(self.block_length, self.rows)
    }

    /// Set the kernel-bias factor to [`kernel_bias_scale`] of `scores` at this
    /// block length.
    #[must_use]
    pub fn with_kernel_bias(mut self, scores: &[&[f64]]) -> Self {
        self.kernel_bias = kernel_bias_scale(scores, self.block_length);
        self
    }

    /// Factor applied to every raw replicate SD: [`Self::fixed_b`] times
    /// [`Self::kernel_bias`].
    #[must_use]
    pub fn se_scale(&self) -> f64 {
        self.fixed_b() * self.kernel_bias
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

    /// Published SE of target `k`: [`Self::raw_se_result`] scaled by [`Self::se_scale`].
    #[must_use]
    pub fn se_result(&self, k: usize) -> BootstrapSeResult {
        let mut result = self.raw_se_result(k);
        let scale = self.se_scale();
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
/// unrelated block; that pairing inflated multi-atom SEs 1.35–2.3×.
///
/// `block_length` is in series times (callers pass [`dependence_block_length`]
/// of the structural span over the window's estimating scores, at least
/// [`antecedent_data::circular_block_length`]); publish SEs through
/// [`RowBlockDraws::se_result`] so they carry the [`circular_fixed_b_scale`] of
/// the resampled window. Returns `None` when the designs share no time.
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
    Some(RowBlockDraws { draws, attempted, cancelled, block_length, rows: len, kernel_bias: 1.0 })
}

/// Resample `rows` lag-aligned rows in circular blocks of `block_length` (callers
/// pass [`dependence_block_length`]) and evaluate `estimate` on each replicate's
/// row map (`row_src[r]` = source row for replicate row `r`).
///
/// `estimate` returns `None` for a replicate that cannot be fit; that replicate
/// counts as failed for every target, so all targets always come from the same
/// replicates. Replicate `r` draws from `ctx.rng.stream_for(StreamDomain::TemporalBlock, stream_base ^ r)`.
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
    RowBlockDraws { draws, attempted, cancelled, block_length, rows, kernel_bias: 1.0 }
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
    let mut row_src = vec![0usize; rows];
    let progress = ctx.progress.as_ref();
    for replicate in 0..replicates {
        if ctx.cancellation.is_cancelled() {
            cancelled = true;
            break;
        }
        attempted += 1;
        let mut rng =
            ctx.rng.stream_for(StreamDomain::TemporalBlock, stream_base ^ u64::from(replicate));
        if fill_circular_block_indexes(rows, block_length, &mut rng, &mut row_src).is_err() {
            continue;
        }
        if let Some(values) =
            estimate(&row_src).filter(|v| v.as_ref().iter().all(|x| x.is_finite()))
        {
            draws.push(values);
        }
        if let Some(progress) = progress {
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
        let boot = row_block_bootstrap_vec(64, 4, 12, 77, &ctx, |rows| {
            calls += 1;
            // Every row map is a concatenation of circular runs of length 4.
            for block in rows.chunks(4) {
                for pair in block.windows(2) {
                    assert_eq!(pair[1], (pair[0] + 1) % 64, "block broke contiguity");
                }
            }
            let mean = rows.iter().map(|&r| values[r]).sum::<f64>() / 64.0;
            // Every third replicate fails one target: it must be dropped for both.
            Some(vec![mean, if calls % 3 == 0 { f64::NAN } else { 2.0 * mean }])
        });
        assert_eq!(boot.block_length, 4);
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
        assert!((boot.fixed_b() - circular_fixed_b_scale(5, 97)).abs() < 1e-15);
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
        // Orthogonal treatment-column score plus aarch64-scale GEMM dust must
        // not invent a testing length the exact zeros on x86_64 would skip.
        let dust: Vec<f64> = (0u32..240).map(|i| 1e-15 * (f64::from(i % 4) - 1.0)).collect();
        assert!(politis_white_block_length(&dust).is_none());
        assert!(politis_white_block_length(&[0.0; 240]).is_none());
    }

    /// The lag-by-lag bandwidth search of [`politis_white_block_length`] against
    /// a reference that takes every autocorrelation up to the cap first: the
    /// same sums in the same order, so the lengths are identical for short and
    /// long memory alike.
    #[test]
    fn politis_white_lazy_lags_match_the_eager_scan() {
        fn eager(scores: &[f64]) -> usize {
            let n = scores.len();
            let nf = n as f64;
            let mean = scores.iter().sum::<f64>() / nf;
            let centered: Vec<f64> = scores.iter().map(|s| s - mean).collect();
            let autocov = |k: usize| -> f64 {
                centered[..n - k].iter().zip(&centered[k..]).map(|(a, b)| a * b).sum::<f64>() / nf
            };
            let gamma0 = autocov(0);
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
            let d_cb = 4.0 / 3.0 * g0 * g0;
            let b = (2.0 * big_g * big_g / d_cb).cbrt() * nf.cbrt();
            let cap = (3.0 * nf.sqrt()).min(nf / 3.0).ceil();
            b.clamp(1.0, cap.max(1.0)).ceil() as usize
        }
        let mut state = 0x2545_f491_4f6c_dd1d_u64;
        let mut uniform = || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            (state >> 11) as f64 / (1u64 << 53) as f64 - 0.5
        };
        for n in [8usize, 41, 400, 5000] {
            for rho in [0.0, 0.5, 0.85, 0.97, -0.6] {
                let mut level = 0.0;
                let series: Vec<f64> = (0..n)
                    .map(|_| {
                        level = rho * level + uniform();
                        level
                    })
                    .collect();
                assert_eq!(
                    politis_white_block_length(&series),
                    Some(eager(&series)),
                    "n={n} rho={rho}"
                );
            }
        }
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
    #[allow(
        clippy::float_cmp,
        reason = "the thresholds are exact constants returned unchanged, so exact equality is intended"
    )]
    fn circular_block_family_thresholds_on_a_fixed_series() {
        use CircularBlockFamily::{Mediation, Mixture, Sequential, SingleWindow};
        assert_eq!(SingleWindow.min_effective_rows(), 45.0);
        assert_eq!(Mediation.min_effective_rows(), 40.0);
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
    #[allow(
        clippy::float_cmp,
        reason = "the kernel bias factor is a deterministic function of the same score and lag, so the test asserts bit-identical values"
    )]
    fn kernel_bias_scale_grows_with_score_memory_and_shrinks_with_block_length() {
        // A short-memory (alternating, negative r1) score keeps the factor at 1.
        let alternating: Vec<f64> = (0..200).map(|t| if t % 2 == 0 { 1.0 } else { -1.0 }).collect();
        assert!((kernel_bias_scale(&[&alternating], 10) - 1.0).abs() < 1e-12);
        // Constant, short and non-finite series are ignored.
        assert!(
            (kernel_bias_scale(&[&[2.0; 50], &[1.0, 2.0], &[1.0, f64::NAN, 3.0]], 10) - 1.0).abs()
                < 1e-12
        );
        assert!((kernel_bias_scale(&[], 10) - 1.0).abs() < 1e-12);
        // AR(1)(0.81) score of length 400 (the persistent ρ = 0.9 design's score):
        // about 3% at the production block (74), more at a shorter block.
        let mut state = 0.0_f64;
        let mut lcg = 0x1234_5678_u64;
        let persistent: Vec<f64> = (0..400)
            .map(|_| {
                lcg = lcg.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1);
                let u = (lcg >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                state = 0.81 * state + u;
                state
            })
            .collect();
        let at_74 = kernel_bias_scale(&[&persistent], 74);
        let at_20 = kernel_bias_scale(&[&persistent], 20);
        assert!(at_74 > 1.01 && at_74 < 1.08, "{at_74}");
        assert!(at_20 > at_74, "{at_20} vs {at_74}");
        // One correction for one score: the scalar SE reads exactly the factor the
        // response bands read per cell (Kendall-corrected AR(1) / BIC AR(q)), and a
        // mildly persistent score is not left at the raw-lag-1 reading.
        assert_eq!(at_74, crate::ar_kernel::kernel_bias_factor(&persistent, 74));
        assert_eq!(at_20, crate::kernel_bias_factor(&persistent, 20));
        // The largest over several scores, and the ρ̂ cap keeps it finite.
        assert!((kernel_bias_scale(&[&alternating, &persistent], 74) - at_74).abs() < 1e-12);
        let ramp: Vec<f64> = (0..400).map(f64::from).collect();
        assert!(kernel_bias_scale(&[&ramp], 74).is_finite());
        // Exact value for a known ρ: LRV/Bartlett_ℓ with ρ = 0.5, ℓ = 4.
        let rho: f64 = 0.5;
        let bartlett = 1.0 + 2.0 * (0.75 * rho + 0.5 * rho * rho + 0.25 * rho.powi(3));
        let exact = ((1.0 + rho) / (1.0 - rho) / bartlett).sqrt();
        let draws = RowBlockDraws {
            draws: vec![],
            attempted: 0,
            cancelled: false,
            block_length: 4,
            rows: 100,
            kernel_bias: exact,
        };
        assert!((draws.se_scale() - circular_fixed_b_scale(4, 100) * exact).abs() < 1e-12);
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
    fn circular_fixed_b_scale_matches_the_simulated_critical_values() {
        assert!((circular_fixed_b_scale(6, 159) - 1.049_592).abs() < 1e-5);
        assert!((circular_fixed_b_scale(0, 10) - 1.0).abs() < 1e-12);
        assert!(circular_fixed_b_scale(8, 399) < circular_fixed_b_scale(4, 59));
        // Simulated circular-Bartlett 97.5% quantiles (n = 2000, 10⁶ series).
        for (b, simulated) in [(0.10, 2.2396), (0.20, 2.5791), (0.33, 3.0979), (0.40, 3.3958)] {
            let cv = 1.96 * circular_fixed_b_scale((b * 10_000.0_f64).round() as usize, 10_000);
            assert!((cv - simulated).abs() < 0.006, "b={b}: {cv} vs {simulated}");
        }
        // Above the non-circular Kiefer–Vogelsang value at long blocks.
        assert!(circular_fixed_b_scale(1, 3) > fixed_b_scale(1, 3) + 0.05);
    }

    #[test]
    fn fixed_b_scale_matches_kiefer_vogelsang_polynomial() {
        assert!((fixed_b_scale(6, 159) - 1.057_458).abs() < 1e-5);
        assert!((fixed_b_scale(0, 10) - 1.0).abs() < 1e-12);
        assert!(fixed_b_scale(8, 399) < fixed_b_scale(4, 59));
    }
}
