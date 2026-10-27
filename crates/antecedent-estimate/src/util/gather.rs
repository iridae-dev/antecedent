//! Row gathers for resampled designs and the arm moments of a binary indicator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::arm::TreatmentArm;

/// Gather `idx`-selected entries of `src` into `out` (a reused scratch buffer) for one IID
/// bootstrap resample: `out[r] = src[idx[r]]`.
pub(crate) fn gather_bootstrap_vector(out: &mut [f64], src: &[f64], idx: &[usize]) {
    for (r, &i) in idx.iter().enumerate() {
        out[r] = src[i];
    }
}

/// Gather `idx`-selected rows of a column-major matrix `src` of `src_rows` rows into `out`,
/// whose row count is `idx.len()`: `out[c·m + r] = src[c·src_rows + idx[r]]` with
/// `m = idx.len()`. The one column-major row gather: an IID resample as long as the data has
/// `m == src_rows`, a whole-cluster resample has its own length.
pub(crate) fn gather_cols(
    out: &mut [f64],
    src: &[f64],
    src_rows: usize,
    ncols: usize,
    idx: &[usize],
) {
    let m = idx.len();
    for c in 0..ncols {
        let source = &src[c * src_rows..(c + 1) * src_rows];
        let target = &mut out[c * m..(c + 1) * m];
        for (slot, &i) in target.iter_mut().zip(idx) {
            *slot = source[i];
        }
    }
}

/// [`gather_cols`] into a growable buffer, which is cleared and resized to `idx.len() × ncols`.
pub(crate) fn gather_cols_into(
    out: &mut Vec<f64>,
    src: &[f64],
    src_rows: usize,
    ncols: usize,
    idx: &[usize],
) {
    out.clear();
    out.resize(idx.len() * ncols, 0.0);
    gather_cols(out, src, src_rows, ncols, idx);
}

/// How many times each of `n` source rows occurs in the resample `idx`: `counts[i]` is the
/// multiplicity of row `i`. The counts of a resample of `n` draws sum to `n`.
pub(crate) fn resample_counts(counts: &mut Vec<u32>, n: usize, idx: &[usize]) {
    counts.clear();
    counts.resize(n, 0);
    for &i in idx {
        counts[i] += 1;
    }
}

/// The design and outcome of the least-squares fit of a resample, formed from the distinct rows
/// it drew instead of the repeated rows themselves.
///
/// The fit that minimises `Σᵢ cᵢ (yᵢ − xᵢβ)²` over the distinct rows `i` with multiplicity
/// `cᵢ = counts[i] > 0` is the fit of the resample's repeated rows, and it is the ordinary
/// least-squares fit of the rows scaled by `√cᵢ`, so the solve stays on the stable QR path (no
/// normal equations) while roughly a third of the rows of an iid resample drop out. `x_out`
/// (column-major, `rows × ncols`) and `y_out` are overwritten; the rows are in source order. At
/// least `min_rows` rows are returned, padded with zero rows (which add nothing to the fit), so
/// a resample with fewer distinct rows than parameters stays a well-formed, rank-deficient
/// problem. Returns the number of rows written.
#[allow(
    clippy::too_many_arguments,
    reason = "the two output buffers, the design and outcome with its leading dimension, and the multiplicities are independent inputs"
)]
pub(crate) fn gather_count_weighted(
    x_out: &mut Vec<f64>,
    y_out: &mut Vec<f64>,
    src: &[f64],
    outcome: &[f64],
    src_rows: usize,
    ncols: usize,
    counts: &[u32],
    min_rows: usize,
) -> usize {
    let distinct = counts.iter().filter(|&&c| c > 0).count();
    let rows = distinct.max(min_rows);
    x_out.clear();
    x_out.resize(rows * ncols, 0.0);
    y_out.clear();
    y_out.resize(rows, 0.0);
    let mut out = 0;
    for (i, &c) in counts.iter().enumerate() {
        if c == 0 {
            continue;
        }
        let scale = f64::from(c).sqrt();
        y_out[out] = outcome[i] * scale;
        for col in 0..ncols {
            x_out[col * rows + out] = src[col * src_rows + i] * scale;
        }
        out += 1;
    }
    rows
}

/// `idx`-selected entries of `values` into a growable buffer, which is cleared first.
pub(crate) fn gather_into<T: Copy>(out: &mut Vec<T>, values: &[T], idx: &[usize]) {
    out.clear();
    out.extend(idx.iter().map(|&i| values[i]));
}

/// `idx`-selected entries of `values` as a fresh vector.
#[must_use]
pub(crate) fn gather<T: Copy>(values: &[T], idx: &[usize]) -> Vec<T> {
    idx.iter().map(|&i| values[i]).collect()
}

/// Count, mean and centered sum of squares of a variable within the two arms of a binary
/// indicator. Index `0` is the control arm, `1` the treated (or instrument-on) arm; an empty
/// arm has `n == 0` and a zero mean.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct ArmMoments {
    /// Rows in each arm.
    pub n: [usize; 2],
    /// Mean of the variable in each arm.
    pub mean: [f64; 2],
    /// `Σ (v − mean)²` within each arm.
    pub sum_sq: [f64; 2],
}

/// [`ArmMoments`] of `values` by the classified indicator `arms` (two passes, so the centered
/// sum of squares does not cancel against a large mean). The indicator is classified once,
/// after validation, with [`TreatmentArm::classify`].
pub(crate) fn arm_moments(arms: &[TreatmentArm], values: &[f64]) -> ArmMoments {
    let mut n = [0usize; 2];
    let mut total = [0.0; 2];
    for (&arm, &v) in arms.iter().zip(values) {
        let k = usize::from(arm.is_treated());
        n[k] += 1;
        total[k] += v;
    }
    let mean = [
        if n[0] > 0 { total[0] / n[0] as f64 } else { 0.0 },
        if n[1] > 0 { total[1] / n[1] as f64 } else { 0.0 },
    ];
    let mut sum_sq = [0.0; 2];
    for (&arm, &v) in arms.iter().zip(values) {
        let k = usize::from(arm.is_treated());
        let d = v - mean[k];
        sum_sq[k] += d * d;
    }
    ArmMoments { n, mean, sum_sq }
}

/// Single-pass `(min, max)` fold.
///
/// Shared by the static and temporal response estimators so neither carries its own
/// min/max pair (two full-slice folds where one suffices).
pub(crate) fn range(values: &[f64]) -> (f64, f64) {
    values.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)))
}

#[cfg(test)]
mod tests {
    use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

    use super::*;

    #[test]
    fn distinct_rows_scaled_by_multiplicity_fit_like_the_repeated_rows() {
        let n = 12;
        let x: Vec<f64> = (0..n).map(|i| (i as f64 * 0.7).sin() + i as f64 * 0.1).collect();
        let mut design = vec![1.0; n];
        design.extend_from_slice(&x);
        let y: Vec<f64> = (0..n).map(|i| 2.0 - 3.0 * x[i] + (i as f64 * 1.3).cos()).collect();
        let idx = [0, 0, 3, 5, 5, 5, 7, 9, 9, 11, 2, 2];
        let mut workspace = LeastSquaresWorkspace::default();

        let mut repeated_x = vec![0.0; idx.len() * 2];
        let mut repeated_y = vec![0.0; idx.len()];
        gather_cols(&mut repeated_x, &design, n, 2, &idx);
        gather_bootstrap_vector(&mut repeated_y, &y, &idx);
        let repeated = FaerBackend
            .least_squares(&repeated_x, idx.len(), 2, &repeated_y, &mut workspace)
            .unwrap();

        let mut counts = Vec::new();
        resample_counts(&mut counts, n, &idx);
        assert_eq!(counts.iter().sum::<u32>() as usize, idx.len());
        let (mut weighted_x, mut weighted_y) = (Vec::new(), Vec::new());
        let rows =
            gather_count_weighted(&mut weighted_x, &mut weighted_y, &design, &y, n, 2, &counts, 2);
        assert_eq!(rows, counts.iter().filter(|&&c| c > 0).count());
        let weighted =
            FaerBackend.least_squares(&weighted_x, rows, 2, &weighted_y, &mut workspace).unwrap();
        for (a, b) in weighted.coefficients.iter().zip(&repeated.coefficients) {
            assert!((a - b).abs() < 1e-10, "{a} vs {b}");
        }
        assert_eq!(weighted.rank, repeated.rank);
    }

    #[test]
    fn a_resample_with_fewer_distinct_rows_than_columns_is_padded_with_zero_rows() {
        let design = [1.0, 1.0, 1.0, 0.0, 1.0, 2.0];
        let y = [1.0, 2.0, 3.0];
        let mut counts = Vec::new();
        resample_counts(&mut counts, 3, &[1, 1, 1]);
        let (mut x, mut yy) = (Vec::new(), Vec::new());
        let rows = gather_count_weighted(&mut x, &mut yy, &design, &y, 3, 2, &counts, 2);
        assert_eq!(rows, 2);
        assert_eq!(yy.len(), 2);
        assert!((yy[0] - 2.0 * 3.0_f64.sqrt()).abs() < 1e-12 && yy[1] == 0.0);
    }
}
