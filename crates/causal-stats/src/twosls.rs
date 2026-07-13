//! Weighted least squares and two-stage least squares.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use crate::error::StatsError;
use crate::linalg::{DenseLinearAlgebra, LeastSquaresFit, LeastSquaresWorkspace};

/// Fit weighted least squares by row-scaling with `sqrt(weight)`.
///
/// `weights` length = `nrows`. Zero or negative weights are treated as 0 (row dropped via 0 scale).
///
/// # Errors
///
/// Shape mismatch or backend failure.
pub fn fit_wls(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    weights: &[f64],
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
) -> Result<LeastSquaresFit, StatsError> {
    if y.len() != nrows || weights.len() != nrows {
        return Err(StatsError::Shape { message: "y/weights length != nrows" });
    }
    if x_colmajor.len() < nrows.saturating_mul(ncols) {
        return Err(StatsError::Shape { message: "X buffer too short" });
    }
    let mut x_w = vec![0.0; nrows * ncols];
    let mut y_w = vec![0.0; nrows];
    for r in 0..nrows {
        let w = weights[r].max(0.0).sqrt();
        y_w[r] = y[r] * w;
        for c in 0..ncols {
            x_w[c * nrows + r] = x_colmajor[c * nrows + r] * w;
        }
    }
    backend.least_squares(&x_w, nrows, ncols, &y_w, workspace)
}

/// Result of two-stage least squares.
#[derive(Clone, Debug)]
pub struct TwoSlsFit {
    /// First-stage coefficients (instruments → endogenous).
    pub first_stage: LeastSquaresFit,
    /// Second-stage coefficients (fitted endogenous + covariates → outcome).
    pub second_stage: LeastSquaresFit,
    /// Fitted endogenous values used in stage 2.
    pub fitted_endogenous: Vec<f64>,
}

/// Two-stage least squares.
///
/// Stage 1: `endogenous ~ instruments` (column-major, may include intercept/covariates in Z).
/// Stage 2: `y ~ [fitted_endogenous | exogenous]` where `exogenous` is column-major extras
/// (may be empty / intercept-only).
///
/// Convention: stage-2 design is `[fitted_T | X]` with `1 + x_ncols` columns; the treatment
/// coefficient is `second_stage.coefficients[0]`.
///
/// # Errors
///
/// Shape mismatch or backend failure.
#[allow(clippy::too_many_arguments)]
pub fn fit_2sls(
    instruments_colmajor: &[f64],
    z_nrows: usize,
    z_ncols: usize,
    endogenous: &[f64],
    exogenous_colmajor: &[f64],
    x_ncols: usize,
    y: &[f64],
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
) -> Result<TwoSlsFit, StatsError> {
    if endogenous.len() != z_nrows || y.len() != z_nrows {
        return Err(StatsError::Shape { message: "endogenous/y length != nrows" });
    }
    if exogenous_colmajor.len() < z_nrows.saturating_mul(x_ncols) {
        return Err(StatsError::Shape { message: "exogenous buffer too short" });
    }
    let first_stage =
        backend.least_squares(instruments_colmajor, z_nrows, z_ncols, endogenous, workspace)?;
    let mut fitted = vec![0.0; z_nrows];
    for r in 0..z_nrows {
        let mut pred = 0.0;
        for c in 0..z_ncols {
            pred += instruments_colmajor[c * z_nrows + r] * first_stage.coefficients[c];
        }
        fitted[r] = pred;
    }
    let stage2_ncols = 1 + x_ncols;
    let mut x2 = vec![0.0; z_nrows * stage2_ncols];
    for r in 0..z_nrows {
        x2[r] = fitted[r];
        for c in 0..x_ncols {
            x2[(1 + c) * z_nrows + r] = exogenous_colmajor[c * z_nrows + r];
        }
    }
    let second_stage = backend.least_squares(&x2, z_nrows, stage2_ncols, y, workspace)?;
    Ok(TwoSlsFit { first_stage, second_stage, fitted_endogenous: fitted })
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faer_backend::FaerBackend;

    #[test]
    fn wls_matches_ols_with_unit_weights() {
        let n = 20usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            x[i] = 1.0;
            x[n + i] = i as f64;
            y[i] = 1.0 + 2.0 * (i as f64);
        }
        let w = vec![1.0; n];
        let mut ws = LeastSquaresWorkspace::default();
        let ols = FaerBackend.least_squares(&x, n, 2, &y, &mut ws).unwrap();
        let wls = fit_wls(&x, n, 2, &y, &w, &FaerBackend, &mut ws).unwrap();
        assert!((ols.coefficients[0] - wls.coefficients[0]).abs() < 1e-10);
        assert!((ols.coefficients[1] - wls.coefficients[1]).abs() < 1e-10);
    }

    #[test]
    fn twosls_recovers_just_identified() {
        // Z → T → Y with no confounding on Z→Y; T = Z + e, Y = 2T + u.
        let n = 200usize;
        let mut z = vec![0.0; n * 2];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut x = vec![0.0; n]; // intercept only exogenous
        for i in 0..n {
            let zi = (i as f64) / n as f64 - 0.5;
            z[i] = 1.0;
            z[n + i] = zi;
            t[i] = zi + 0.01 * ((i % 7) as f64 - 3.0);
            y[i] = 2.0 * t[i] + 0.01 * ((i % 5) as f64 - 2.0);
            x[i] = 1.0;
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_2sls(&z, n, 2, &t, &x, 1, &y, &FaerBackend, &mut ws).unwrap();
        assert!((fit.second_stage.coefficients[0] - 2.0).abs() < 0.05);
    }
}
