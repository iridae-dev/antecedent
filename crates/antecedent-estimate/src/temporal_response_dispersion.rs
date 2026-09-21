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
//! [`kernel_bias_factor`] ([`crate::ar_kernel`], the one owner of the autoregressive
//! kernel-bias model the scalar SEs share) reads each cell's influence.
//!
//! [`influence_effective_rows`] is the short-series statistic of the same influence
//! ([`crate::temporal_block::score_effective_rows`]), read against
//! [`RESPONSE_SHORT_SERIES_ROWS`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use crate::ar_kernel::kernel_bias_factor;
use crate::temporal_block::score_effective_rows;

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
