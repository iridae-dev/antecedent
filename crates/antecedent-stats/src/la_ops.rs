//! Thin wrappers around the LA operations Antecedent actually uses.
#![allow(dead_code)]
//!
//! Not a generic linear-algebra backend. Callers stay on library-owned
//! column-major slices. `faer` types do not leave this crate.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::StatsError;
use crate::faer_backend::FaerBackend;
use crate::gram::form_xtx;
use crate::linalg::{DenseLinearAlgebra, LeastSquaresFit, LeastSquaresWorkspace};
use crate::regularized::fit_ridge;
use crate::twosls::fit_wls;

/// Column-pivoted QR least squares.
pub(crate) fn least_squares(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    workspace: &mut LeastSquaresWorkspace,
) -> Result<LeastSquaresFit, StatsError> {
    FaerBackend.least_squares(x_colmajor, nrows, ncols, y, workspace)
}

/// Weighted least squares via row-scaling with `sqrt(weight)`.
pub(crate) fn weighted_least_squares(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    weights: &[f64],
    workspace: &mut LeastSquaresWorkspace,
) -> Result<LeastSquaresFit, StatsError> {
    fit_wls(x_colmajor, nrows, ncols, y, weights, &FaerBackend, workspace)
}

/// Ridge: `(XᵀX + λI)β = Xᵀy`.
pub(crate) fn ridge_solve(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    lambda: f64,
    workspace: &mut LeastSquaresWorkspace,
) -> Result<LeastSquaresFit, StatsError> {
    fit_ridge(x_colmajor, nrows, ncols, y, lambda, &FaerBackend, workspace)
}

/// Form the Gram matrix `XᵀX` (row-major `ncols×ncols`).
pub(crate) fn gram(x_colmajor: &[f64], nrows: usize, ncols: usize, xtx: &mut [f64]) {
    form_xtx(x_colmajor, nrows, ncols, xtx);
}

/// Form the cross-product `Xᵀy`.
pub(crate) fn crossprod(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    xty: &mut [f64],
) {
    debug_assert!(y.len() >= nrows);
    debug_assert!(xty.len() >= ncols);
    debug_assert!(x_colmajor.len() >= nrows.saturating_mul(ncols));
    for c in 0..ncols {
        let mut s = 0.0;
        let col = &x_colmajor[c * nrows..(c + 1) * nrows];
        for r in 0..nrows {
            s += col[r] * y[r];
        }
        xty[c] = s;
    }
}

/// Numerical rank from column-pivoted QR, same tolerance as [`FaerBackend`].
pub(crate) fn stable_rank(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
) -> Result<usize, StatsError> {
    if x_colmajor.len() < nrows.saturating_mul(ncols) {
        return Err(StatsError::Shape { message: "X buffer too short" });
    }
    if nrows == 0 || ncols == 0 {
        return Ok(0);
    }
    let scales: Vec<f64> = (0..ncols)
        .map(|c| {
            let scale = x_colmajor[c * nrows..(c + 1) * nrows]
                .iter()
                .copied()
                .map(f64::abs)
                .fold(0.0_f64, f64::max);
            if scale == 0.0 { 1.0 } else { scale }
        })
        .collect();
    let a = faer::Mat::<f64>::from_fn(nrows, ncols, |r, c| x_colmajor[c * nrows + r] / scales[c]);
    let qr = faer::linalg::solvers::ColPivQr::new(a.as_ref());
    let r_factor = qr.thin_R();
    let size = r_factor.nrows().min(r_factor.ncols());
    let mut max_diag = 0.0_f64;
    for i in 0..size {
        max_diag = max_diag.max(r_factor[(i, i)].abs());
    }
    let tol = (nrows as f64).sqrt() * f64::EPSILON * max_diag;
    let mut rank = 0usize;
    for i in 0..size {
        if r_factor[(i, i)].abs() > tol {
            rank += 1;
        }
    }
    Ok(rank)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn line_design(n: usize) -> (Vec<f64>, Vec<f64>) {
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            x[i] = 1.0;
            x[n + i] = i as f64;
            y[i] = 3.0 + 4.0 * (i as f64);
        }
        (x, y)
    }

    #[test]
    fn helpers_recover_known_line() {
        let n = 20usize;
        let (x, y) = line_design(n);
        let mut ws = LeastSquaresWorkspace::default();
        let fit = least_squares(&x, n, 2, &y, &mut ws).unwrap();
        assert!((fit.coefficients[0] - 3.0).abs() < 1e-10);
        assert!((fit.coefficients[1] - 4.0).abs() < 1e-10);
        assert_eq!(stable_rank(&x, n, 2).unwrap(), 2);

        let mut xtx = vec![0.0; 4];
        gram(&x, n, 2, &mut xtx);
        let mut xty = vec![0.0; 2];
        crossprod(&x, n, 2, &y, &mut xty);
        assert!(xtx[0] > 0.0);
        assert!(xty[0] > 0.0);

        let weights = vec![1.0; n];
        let wls = weighted_least_squares(&x, n, 2, &y, &weights, &mut ws).unwrap();
        assert!((wls.coefficients[0] - 3.0).abs() < 1e-10);

        let ridge = ridge_solve(&x, n, 2, &y, 1e-8, &mut ws).unwrap();
        assert!((ridge.coefficients[1] - 4.0).abs() < 1e-6);
    }
}
