//! Per-cell dispersion of a temporal response circular-block band: the parametric
//! kernel-bias factor and the short-series reading of each published cell.
//!
//! The circular-block bootstrap variance of a level is (asymptotically) the Bartlett
//! long-run variance of that level's influence series at bandwidth `ℓ`, the block
//! length. The Bartlett kernel misses `O(1/ℓ)` of a persistent influence's long-run
//! variance, a bias the fixed-b critical value does not repair: for an AR(1) ρ = 0.9
//! influence at `ℓ = 13` the kernel keeps 46% of the long-run variance. Lengthening
//! the block does not close the gap either — at `ℓ = n/3` the kernel keeps 82% but
//! the fixed-b interval was measured at 0.930 coverage of nominal 0.95 (n = 160) against
//! 0.954 at `ℓ = ⌈√n⌉` with the correction below — and it degrades the sup-t band,
//! which needs many blocks per replicate.
//!
//! [`kernel_bias_factor`] fits the influence with the same autoregressive family the
//! Bayesian tempering uses (a Kendall-corrected AR(1) and a BIC-selected AR(q ≤ 4)),
//! reads the fraction of the fitted model's long-run variance the Bartlett kernel at
//! `ℓ` keeps, and returns `1/√fraction` (never below 1). A weakly dependent influence
//! keeps a factor within 1% of 1; the factor is model-based, so it does not carry the
//! sampling noise of a wide-bandwidth long-run-variance estimate. What the model does
//! not see, the factor does not correct: a persistent component that is a small share
//! of the influence (an AR(1) ρ = 0.9 residual under omitted iid treatment lags) is
//! invisible to the fit at short `n`.
//!
//! [`influence_effective_rows`] is the short-series statistic of the same influence
//! ([`crate::temporal_block::score_effective_rows`]), read against
//! [`RESPONSE_SHORT_SERIES_ROWS`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use crate::temporal_block::score_effective_rows;

/// Largest absolute AR(1) coefficient the fit reports (keeps the model long-run
/// variance finite).
const MAX_AR1_RHO: f64 = 0.97;

/// Largest autoregressive order the BIC search considers.
const MAX_AR_ORDER: usize = 4;

/// Fewest rows on which an influence model is fitted; shorter series keep factor 1.
const MIN_ROWS: usize = 8;

/// Longest autocorrelation tail summed for the model long-run variance.
const MAX_ACF_LAGS: usize = 20_000;

/// Effective rows of a published cell's influence below which a temporal response band
/// carries `response.temporal.block.short_series`.
///
/// Provenance: `crates/antecedent/tests/v19_temporal_response_calibration.rs`
/// (400 replicates, nominal 0.95), the rule of the scalar families: the smallest
/// multiple of 5 at which every design covering below the gate band warns on at least
/// 90% of its replicates. The shift response under an AR(1) φ = 0.9 treatment reads the
/// treatment mean; its influence reads 4.5 / 7.3 / 11.9 effective rows (10th / 50th /
/// 90th percentile) at n = 100, where the band covers 0.885 pointwise and 0.890
/// simultaneous, so it warns on every replicate here. The same design at n = 160 reads
/// 6.4 / 10.4 / 15.7 rows and covers 0.943 / 0.948 (gated), so it still warns on most
/// replicates; at n = 400 it reads 16.7 / 22.9 / 31.3 rows, covers 0.938 / 0.940 and is
/// nearly quiet. The dose cells of the same design read 22.6 / 32.7 / 45.3 rows at
/// n = 100 and 35.9 / 51.4 / 68.6 at n = 160; every iid / AR(1) ρ = 0.5 cell reads above
/// 100 rows. None of them warns.
pub const RESPONSE_SHORT_SERIES_ROWS: f64 = 15.0;

/// Per-cell dispersion readings of one temporal response band, in the cell layout of
/// the published surface.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct CellDispersion {
    /// Kernel-bias factor of each cell ([`kernel_bias_factor`]), applied to that cell's
    /// replicate deviations on top of the block dispersion inflation.
    pub kernel_factors: Vec<f64>,
    /// Short-series reading of each cell's influence ([`influence_effective_rows`]).
    pub effective_rows: Vec<f64>,
}

impl CellDispersion {
    /// Readings of every cell whose influence series is given, at block length `block`.
    #[must_use]
    pub fn from_influences(influences: &[Vec<f64>], block: usize) -> Self {
        Self {
            kernel_factors: influences.iter().map(|s| kernel_bias_factor(s, block)).collect(),
            effective_rows: influences.iter().map(|s| influence_effective_rows(s, block)).collect(),
        }
    }

    /// One reading for a cell whose influence is not available as a single series but
    /// whose estimating scores are: the largest factor and the fewest effective rows
    /// over the scores (a conservative substitute for the delta-method influence).
    #[must_use]
    pub fn from_scores(scores: &[&[f64]], block: usize) -> Self {
        let factor = scores.iter().map(|s| kernel_bias_factor(s, block)).fold(1.0, f64::max);
        let rows = scores
            .iter()
            .map(|s| influence_effective_rows(s, block))
            .filter(|r| r.is_finite())
            .fold(f64::NAN, f64::min);
        Self { kernel_factors: vec![factor], effective_rows: vec![rows] }
    }

    /// Append another surface's readings (cells of a joint band published together).
    pub fn extend(&mut self, other: Self) {
        self.kernel_factors.extend(other.kernel_factors);
        self.effective_rows.extend(other.effective_rows);
    }

    /// Largest kernel-bias factor over the cells (`1` when there are none).
    #[must_use]
    pub fn max_factor(&self) -> f64 {
        self.kernel_factors.iter().copied().fold(1.0, f64::max)
    }

    /// Fewest effective rows over the cells (`NaN` when no cell has a finite reading).
    #[must_use]
    pub fn min_effective_rows(&self) -> f64 {
        self.effective_rows.iter().copied().filter(|r| r.is_finite()).fold(f64::NAN, f64::min)
    }

    /// Whether any cell's influence falls short of [`RESPONSE_SHORT_SERIES_ROWS`] (an
    /// unreadable influence counts as short).
    #[must_use]
    pub fn is_short_series(&self) -> bool {
        let rows = self.min_effective_rows();
        rows.is_nan() || rows < RESPONSE_SHORT_SERIES_ROWS
    }

    /// Scale each replicate's deviation from `center` by its cell's kernel factor.
    pub fn inflate(&self, center: &[f64], draws: &mut [Vec<f64>]) {
        if self.kernel_factors.len() != center.len() {
            return;
        }
        for draw in draws {
            for ((value, mid), factor) in draw.iter_mut().zip(center).zip(&self.kernel_factors) {
                *value = mid + factor * (*value - mid);
            }
        }
    }
}

/// Short-series reading of one influence series at block length `block`: the smaller
/// of its lag-1 AR(1) and block-length Bartlett effective-row readings.
#[must_use]
pub fn influence_effective_rows(influence: &[f64], block: usize) -> f64 {
    score_effective_rows(&[influence], block)
}

/// Kernel-bias factor of one influence series at block length `block`:
/// `1/√f`, `f` the fraction of the fitted autoregression's long-run variance that the
/// Bartlett kernel at `block` keeps, `f = (1 + 2 Σ_{k<ℓ} (1 − k/ℓ) ρ_k) / (1 + 2 Σ_k ρ_k)`
/// with `ρ_k` the model autocorrelation. Two fits are read and the larger factor kept:
/// a Kendall-corrected AR(1) (`ρ̂ + (1 + 3ρ̂)/n`, capped at 0.97) and, when BIC selects
/// an order of at least 2 (Yule–Walker, `q ≤ 4`), that AR(q). A model with no positive
/// long-run excess, a constant or non-finite series, or fewer than 8 rows gives `1`.
#[must_use]
pub fn kernel_bias_factor(influence: &[f64], block: usize) -> f64 {
    let n = influence.len();
    if n < MIN_ROWS || block == 0 || influence.iter().any(|v| !v.is_finite()) {
        return 1.0;
    }
    let mean = influence.iter().sum::<f64>() / n as f64;
    let s: Vec<f64> = influence.iter().map(|v| v - mean).collect();
    let gamma0 = s.iter().map(|v| v * v).sum::<f64>() / n as f64;
    if !gamma0.is_finite() || gamma0 <= 0.0 {
        return 1.0;
    }
    let mut fraction = 1.0_f64;
    let rho1 = kendall_rho(&s);
    if rho1 > 0.0 {
        fraction = fraction.min(bartlett_capture(&ar1_acf(rho1, block), block));
    }
    let model = bic_autoregression(&s);
    if model.phi.len() >= 2 {
        fraction = fraction.min(bartlett_capture(&model.acf(block), block));
    }
    if fraction.is_finite() && fraction > 0.0 && fraction < 1.0 {
        fraction.sqrt().recip()
    } else {
        1.0
    }
}

/// Fraction of the long-run variance `1 + 2 Σ_k ρ_k` (over the whole `acf`, `acf[0] = 1`)
/// that the Bartlett kernel at `block` keeps; `1` when the model carries no positive
/// long-run excess.
fn bartlett_capture(acf: &[f64], block: usize) -> f64 {
    let full = 1.0 + 2.0 * acf.iter().skip(1).sum::<f64>();
    let kept = 1.0
        + 2.0
            * acf
                .iter()
                .enumerate()
                .skip(1)
                .take(block.saturating_sub(1))
                .map(|(k, rho)| (1.0 - k as f64 / block as f64) * rho)
                .sum::<f64>();
    if !full.is_finite() || !kept.is_finite() || full <= 1.0 || kept <= 0.0 {
        1.0
    } else {
        (kept / full).min(1.0)
    }
}

/// Model autocorrelations `ρ_0 = 1, ρ_1, …` of an AR(1) with coefficient `rho`, summed
/// until the tail is negligible (at least `block` lags).
fn ar1_acf(rho: f64, block: usize) -> Vec<f64> {
    let mut acf = vec![1.0];
    let mut value = 1.0;
    for k in 1..MAX_ACF_LAGS {
        value *= rho;
        if k > block && value.abs() < 1e-12 {
            break;
        }
        acf.push(value);
    }
    acf
}

/// Bias-corrected lag-1 autocorrelation of a centered series (Kendall 1954:
/// `E[ρ̂] ≈ ρ − (1 + 3ρ)/n`), clamped to `±MAX_AR1_RHO`.
fn kendall_rho(s: &[f64]) -> f64 {
    let (num, den) =
        s.windows(2).fold((0.0, 0.0), |(num, den), w| (num + w[1] * w[0], den + w[0] * w[0]));
    let rho_hat = if den > 0.0 { num / den } else { 0.0 };
    (rho_hat + (1.0 + 3.0 * rho_hat) / s.len() as f64).clamp(-MAX_AR1_RHO, MAX_AR1_RHO)
}

/// Yule–Walker autoregression of a centered series at a BIC-selected order.
#[derive(Clone, Debug, Default)]
struct Autoregression {
    /// Coefficients `φ₁ … φ_q` (empty for `q = 0`).
    phi: Vec<f64>,
    /// Sample autocorrelations `r_0 = 1, r_1 … r_q` (the model's own up to lag `q`).
    autocorrelation: Vec<f64>,
}

impl Autoregression {
    /// Model autocorrelations extended by the AR recursion until the tail is negligible
    /// (at least `block` lags).
    fn acf(&self, block: usize) -> Vec<f64> {
        let q = self.phi.len();
        let mut rho = self.autocorrelation.clone();
        for k in (q + 1)..MAX_ACF_LAGS {
            let next: f64 = self.phi.iter().enumerate().map(|(j, f)| f * rho[k - 1 - j]).sum();
            if !next.is_finite() {
                break;
            }
            rho.push(next);
            if k > block && rho[k + 1 - q..=k].iter().all(|r| r.abs() < 1e-12) {
                break;
            }
        }
        rho
    }
}

/// Levinson–Durbin fit of `s` (centered) with `q ≤ MAX_AR_ORDER` minimizing
/// `n ln σ̂²_q + q ln n`. The Toeplitz autocovariance is positive definite, so every
/// fitted model is stationary.
fn bic_autoregression(s: &[f64]) -> Autoregression {
    let n = s.len();
    let max_order = MAX_AR_ORDER.min(n.saturating_sub(2));
    let gamma: Vec<f64> = (0..=max_order)
        .map(|k| s[k..].iter().zip(s).map(|(a, b)| a * b).sum::<f64>() / n as f64)
        .collect();
    if gamma[0] <= 0.0 || !gamma[0].is_finite() {
        return Autoregression::default();
    }
    let nf = n as f64;
    let mut phi: Vec<f64> = Vec::new();
    let mut variance = gamma[0];
    let (mut best_bic, mut best) = (nf * variance.ln(), Vec::new());
    for order in 1..=max_order {
        let acc = gamma[order]
            - phi.iter().enumerate().map(|(j, f)| f * gamma[order - 1 - j]).sum::<f64>();
        let reflection = acc / variance;
        if !reflection.is_finite() || reflection.abs() >= 1.0 {
            break;
        }
        let mut next: Vec<f64> =
            (0..order - 1).map(|j| phi[j] - reflection * phi[order - 2 - j]).collect();
        next.push(reflection);
        phi = next;
        variance *= 1.0 - reflection * reflection;
        if variance <= 0.0 || variance.is_nan() {
            break;
        }
        let bic = nf * variance.ln() + order as f64 * nf.ln();
        if bic < best_bic {
            best_bic = bic;
            best.clone_from(&phi);
        }
    }
    let autocorrelation = gamma[..=best.len()].iter().map(|g| g / gamma[0]).collect();
    Autoregression { phi: best, autocorrelation }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic AR(1) series with Gaussian innovations (Box–Muller on splitmix64).
    fn gaussian_series(n: usize, rho: f64, seed: u64) -> Vec<f64> {
        let mut state = seed;
        let mut next = || {
            state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
            let mut z = state;
            z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
            z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
            ((z ^ (z >> 31)) >> 11) as f64 / (1u64 << 53) as f64
        };
        let mut out = Vec::with_capacity(n);
        let mut previous = 0.0;
        for _ in 0..n {
            let (u, v) = (next().max(1e-12), next());
            let innovation = (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos();
            previous = rho * previous + innovation;
            out.push(previous);
        }
        out
    }

    #[test]
    fn bartlett_capture_matches_the_ar1_closed_form() {
        // AR(1) ρ: full ratio (1+ρ)/(1−ρ); kept 1 + 2 Σ_{k<ℓ} (1 − k/ℓ) ρ^k.
        let (rho, block) = (0.9_f64, 13_usize);
        let full = (1.0 + rho) / (1.0 - rho);
        let kept = 1.0
            + 2.0
                * (1..block)
                    .map(|k| (1.0 - k as f64 / block as f64) * rho.powf(k as f64))
                    .sum::<f64>();
        let acf = ar1_acf(rho, block);
        assert!((bartlett_capture(&acf, block) - kept / full).abs() < 1e-9);
        assert!((kept / full - 0.456).abs() < 0.01, "{}", kept / full);
        // No positive excess: the kernel keeps everything.
        assert!((bartlett_capture(&[1.0, -0.3, 0.09], 5) - 1.0).abs() < 1e-12);
        assert!((bartlett_capture(&[1.0], 5) - 1.0).abs() < 1e-12);
    }

    #[test]
    fn kernel_bias_factor_is_near_one_for_white_noise_and_large_for_persistence() {
        let white = gaussian_series(400, 0.0, 3);
        let f0 = kernel_bias_factor(&white, 20);
        assert!((1.0..1.03).contains(&f0), "white noise factor {f0}");
        let persistent = gaussian_series(400, 0.9, 5);
        let f9 = kernel_bias_factor(&persistent, 20);
        // True capture at ℓ = 20 is 0.58: factor about 1.3.
        assert!((1.15..1.6).contains(&f9), "persistent factor {f9}");
        assert!(kernel_bias_factor(&persistent, 400) < f9, "longer blocks keep more");
        // Negative dependence, constants, short and non-finite series keep 1.
        let alternating: Vec<f64> = (0..100).map(|t| if t % 2 == 0 { 1.0 } else { -1.0 }).collect();
        assert!((kernel_bias_factor(&alternating, 10) - 1.0).abs() < 1e-12);
        assert!((kernel_bias_factor(&[2.0; 50], 5) - 1.0).abs() < 1e-12);
        assert!((kernel_bias_factor(&persistent[..6], 2) - 1.0).abs() < 1e-12);
        assert!(
            (kernel_bias_factor(&[1.0, f64::NAN, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0], 2) - 1.0).abs()
                < 1e-12
        );
    }

    #[test]
    fn bic_autoregression_recovers_an_ar2_and_its_acf_follows_the_recursion() {
        let (a, b) = (0.3_f64, 0.5_f64);
        let mut state = 0.0;
        let mut prev = 0.0;
        let mut white = gaussian_series(4_500, 0.0, 11).into_iter();
        let series: Vec<f64> = (0..4_500)
            .map(|_| {
                let next = a * state + b * prev + white.next().unwrap();
                prev = state;
                state = next;
                next
            })
            .skip(500)
            .collect();
        let mean = series.iter().sum::<f64>() / series.len() as f64;
        let centered: Vec<f64> = series.iter().map(|v| v - mean).collect();
        let fit = bic_autoregression(&centered);
        assert_eq!(fit.phi.len(), 2, "{fit:?}");
        assert!((fit.phi[0] - a).abs() < 0.06 && (fit.phi[1] - b).abs() < 0.06, "{fit:?}");
        let acf = fit.acf(10);
        for k in 3..acf.len() {
            let recursion = fit.phi[0] * acf[k - 1] + fit.phi[1] * acf[k - 2];
            assert!((acf[k] - recursion).abs() < 1e-12);
        }
        // The AR(2) model sees dependence the AR(1) fit under-reads: its factor is larger.
        let ar1_only = bartlett_capture(&ar1_acf(kendall_rho(&centered), 20), 20);
        let ar2 = bartlett_capture(&fit.acf(20), 20);
        assert!(ar2 < ar1_only, "AR(2) capture {ar2} vs AR(1) {ar1_only}");
        assert!(kernel_bias_factor(&centered, 20) >= ar2.sqrt().recip() - 1e-12);
    }

    #[test]
    fn cell_dispersion_reads_every_cell_and_scales_deviations_per_cell() {
        let white = gaussian_series(300, 0.0, 7);
        let persistent = gaussian_series(300, 0.9, 9);
        let cells = CellDispersion::from_influences(&[white.clone(), persistent.clone()], 18);
        assert_eq!(cells.kernel_factors.len(), 2);
        assert!(cells.kernel_factors[0] < 1.03 && cells.kernel_factors[1] > 1.15, "{cells:?}");
        assert!(cells.effective_rows[0] > 150.0 && cells.effective_rows[1] < 60.0, "{cells:?}");
        assert!((cells.max_factor() - cells.kernel_factors[1]).abs() < 1e-15);
        assert!((cells.min_effective_rows() - cells.effective_rows[1]).abs() < 1e-15);
        assert!(cells.is_short_series());
        let quiet = CellDispersion::from_influences(&[white.clone()], 18);
        assert!(!quiet.is_short_series());
        assert!(CellDispersion::default().is_short_series(), "no reading warns");
        let scores = CellDispersion::from_scores(&[&white, &persistent], 18);
        assert_eq!(scores.kernel_factors.len(), 1);
        assert!((scores.kernel_factors[0] - cells.kernel_factors[1]).abs() < 1e-15);
        assert!((scores.effective_rows[0] - cells.effective_rows[1]).abs() < 1e-15);
        let center = [1.0, 2.0];
        let mut draws = vec![vec![2.0, 1.0]];
        CellDispersion { kernel_factors: vec![1.5, 2.0], effective_rows: vec![10.0, 10.0] }
            .inflate(&center, &mut draws);
        assert_eq!(draws[0], vec![2.5, 0.0]);
        // A layout mismatch leaves the draws alone rather than scaling the wrong cells.
        CellDispersion { kernel_factors: vec![3.0], effective_rows: vec![10.0] }
            .inflate(&center, &mut draws);
        assert_eq!(draws[0], vec![2.5, 0.0]);
        let mut joined = quiet;
        joined.extend(scores);
        assert_eq!(joined.kernel_factors.len(), 2);
    }
}
