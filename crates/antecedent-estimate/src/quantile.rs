//! Quantile treatment effects by inverting an estimated interventional CDF.
//!
//! `Q_a(τ) = F_a^{-1}(τ)` with `F_a(c) = P(Y(a) ≤ c) = 1 − P(Y(a) > c)`.
//! The influence function is `φ_Q = −φ_F(q) / f(q)`. Density is a finite
//! difference of `F` on the estimation grid. The path refuses when `τ` is
//! outside the estimated CDF range or the density is too small to invert.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use crate::error::EstimationError;

/// Minimum finite-difference density licensed for inversion.
pub const MIN_QUANTILE_DENSITY: f64 = 1e-3;

/// Invert a monotone CDF on a strictly increasing threshold grid.
///
/// `f_le[j] = P(Y ≤ thresholds[j])`. `phi_f[j]` is the IF of that CDF
/// coordinate (length `n`).
///
/// # Errors
///
/// Empty grid, length mismatch, `τ` outside the estimated range, or density
/// below [`MIN_QUANTILE_DENSITY`].
pub fn invert_cdf_quantile(
    thresholds: &[f64],
    f_le: &[f64],
    phi_f: &[Vec<f64>],
    tau: f64,
) -> Result<(f64, Vec<f64>, f64), EstimationError> {
    if thresholds.len() < 2 || thresholds.len() != f_le.len() || phi_f.len() != thresholds.len() {
        return Err(EstimationError::data_msg(
            "quantile inversion requires a CDF grid of at least two aligned thresholds",
        ));
    }
    let n = phi_f[0].len();
    if n == 0 || phi_f.iter().any(|p| p.len() != n) {
        return Err(EstimationError::data_msg(
            "quantile inversion requires aligned influence columns",
        ));
    }
    if !(tau > 0.0 && tau < 1.0) {
        return Err(EstimationError::unsupported("quantile level must lie in (0, 1)"));
    }
    let Some(hi) = f_le.iter().position(|&f| f >= tau) else {
        return Err(EstimationError::unsupported(
            "quantile inversion refused: τ is above the estimated CDF range",
        ));
    };
    if hi == 0 {
        return Err(EstimationError::unsupported(
            "quantile inversion refused: τ is below the estimated CDF range",
        ));
    }
    let lo = hi - 1;
    let width = thresholds[hi] - thresholds[lo];
    let rise = f_le[hi] - f_le[lo];
    if !(width > 0.0) {
        return Err(EstimationError::data_msg("quantile grid must be strictly increasing"));
    }
    let density = rise / width;
    if !density.is_finite() || density < MIN_QUANTILE_DENSITY {
        return Err(EstimationError::unsupported(
            "quantile inversion refused: estimated density at the quantile is too small",
        ));
    }
    let weight = ((tau - f_le[lo]) / rise).clamp(0.0, 1.0);
    let quantile = thresholds[lo] + weight * width;
    let mut influence = vec![0.0; n];
    for i in 0..n {
        let phi = (1.0 - weight) * phi_f[lo][i] + weight * phi_f[hi][i];
        influence[i] = -phi / density;
    }
    Ok((quantile, influence, density))
}

/// Empirical strictly increasing threshold grid from observed `Y`.
///
/// # Errors
///
/// Too few finite values or a collapsed range.
pub fn empirical_threshold_grid(y: &[f64], points: usize) -> Result<Vec<f64>, EstimationError> {
    let mut v: Vec<f64> = y.iter().copied().filter(|x| x.is_finite()).collect();
    if v.len() < 16 || points < 4 {
        return Err(EstimationError::data_msg(
            "quantile estimation requires at least 16 finite outcomes and 4 grid points",
        ));
    }
    v.sort_by(|a, b| a.total_cmp(b));
    let mut out = Vec::with_capacity(points);
    for i in 1..=points {
        let p = i as f64 / (points + 1) as f64;
        let idx = ((v.len() - 1) as f64 * p).round() as usize;
        out.push(v[idx.min(v.len() - 1)]);
    }
    out.dedup_by(|a, b| (*a - *b).abs() <= 1e-12);
    if out.len() < 4 {
        return Err(EstimationError::data_msg(
            "quantile estimation grid collapsed; outcome has too little spread",
        ));
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn inverts_linear_cdf() {
        let thresholds = [0.0, 1.0, 2.0, 3.0];
        let f_le = [0.0, 1.0 / 3.0, 2.0 / 3.0, 1.0];
        let n = 8usize;
        let phi: Vec<Vec<f64>> =
            (0..4).map(|j| (0..n).map(|i| (i as f64) * 0.01 + j as f64 * 0.0).collect()).collect();
        let (q, infl, density) = invert_cdf_quantile(&thresholds, &f_le, &phi, 0.5).unwrap();
        assert!((q - 1.5).abs() < 1e-12);
        assert!((density - (1.0 / 3.0)).abs() < 1e-12);
        assert_eq!(infl.len(), n);
    }

    #[test]
    fn refuses_tail() {
        let thresholds = [0.0, 1.0];
        let f_le = [0.2, 0.4];
        let phi = vec![vec![0.0; 4], vec![0.0; 4]];
        let err = invert_cdf_quantile(&thresholds, &f_le, &phi, 0.9).unwrap_err();
        assert!(err.to_string().contains("above the estimated CDF"));
    }
}
