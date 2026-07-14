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
    /// First-stage coefficients (full instrument set `[Z | X]` → endogenous).
    pub first_stage: LeastSquaresFit,
    /// Second-stage coefficients (fitted endogenous + covariates → outcome).
    pub second_stage: LeastSquaresFit,
    /// Fitted endogenous values used in stage 2.
    pub fitted_endogenous: Vec<f64>,
    /// Structural residual sum of squares `‖y − Tβ̂ − Xγ̂‖²` using the *actual*
    /// endogenous values (not the fitted ones). This is the σ̂² numerator for the
    /// conventional 2SLS analytic standard error.
    pub structural_rss: f64,
}

/// Two-stage least squares.
///
/// Stage 1: `endogenous ~ [instruments | exogenous]` — the full instrument set. Included
/// exogenous regressors instrument themselves, so `exogenous` (column-major, may be
/// empty / intercept-only) is appended to the excluded instruments in the first-stage
/// design. Pass the intercept in exactly one of the two blocks.
/// Stage 2: `y ~ [fitted_endogenous | exogenous]`.
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
    // Full instrument set: excluded instruments plus included exogenous regressors.
    let stage1_ncols = z_ncols + x_ncols;
    let mut x1 = vec![0.0; z_nrows * stage1_ncols];
    x1[..z_nrows * z_ncols].copy_from_slice(&instruments_colmajor[..z_nrows * z_ncols]);
    x1[z_nrows * z_ncols..].copy_from_slice(&exogenous_colmajor[..z_nrows * x_ncols]);
    let first_stage = backend.least_squares(&x1, z_nrows, stage1_ncols, endogenous, workspace)?;
    let mut fitted = vec![0.0; z_nrows];
    for r in 0..z_nrows {
        let mut pred = 0.0;
        for c in 0..stage1_ncols {
            pred += x1[c * z_nrows + r] * first_stage.coefficients[c];
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
    // Structural residuals evaluate the second-stage coefficients at the ACTUAL
    // endogenous values; `second_stage.rss` uses fitted T and is not σ̂².
    let mut structural_rss = 0.0;
    for r in 0..z_nrows {
        let mut pred = second_stage.coefficients[0] * endogenous[r];
        for c in 0..x_ncols {
            pred += exogenous_colmajor[c * z_nrows + r] * second_stage.coefficients[1 + c];
        }
        let e = y[r] - pred;
        structural_rss += e * e;
    }
    Ok(TwoSlsFit { first_stage, second_stage, fitted_endogenous: fitted, structural_rss })
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
        // Instruments carry only the excluded Z column; the intercept lives in the
        // exogenous block (stage 1 uses [Z | 1], stage 2 uses [fitted_T | 1]).
        let n = 200usize;
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut x = vec![0.0; n]; // intercept only exogenous
        for i in 0..n {
            let zi = (i as f64) / n as f64 - 0.5;
            z[i] = zi;
            t[i] = zi + 0.01 * ((i % 7) as f64 - 3.0);
            y[i] = 2.0 * t[i] + 0.01 * ((i % 5) as f64 - 2.0);
            x[i] = 1.0;
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_2sls(&z, n, 1, &t, &x, 1, &y, &FaerBackend, &mut ws).unwrap();
        assert!((fit.second_stage.coefficients[0] - 2.0).abs() < 0.05);
        assert!(fit.structural_rss <= fit.second_stage.rss);
    }

    #[test]
    fn twosls_first_stage_includes_exogenous_regressors() {
        // Y = 2T + 1.5X + u with X correlated with T beyond Z; projecting T on [1, Z]
        // only (the old first stage) is inconsistent here.
        let n = 400usize;
        let mut z = vec![0.0; n];
        let mut t = vec![0.0; n];
        let mut y = vec![0.0; n];
        let mut x = vec![0.0; n * 2];
        for i in 0..n {
            let zi = (i as f64) / n as f64 - 0.5;
            let xi = ((i % 13) as f64 - 6.0) / 6.0;
            let e = 0.01 * ((i % 7) as f64 - 3.0);
            z[i] = zi;
            t[i] = zi + 0.8 * xi + e;
            y[i] = 2.0 * t[i] + 1.5 * xi + 0.01 * ((i % 5) as f64 - 2.0);
            x[i] = 1.0;
            x[n + i] = xi;
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_2sls(&z, n, 1, &t, &x, 2, &y, &FaerBackend, &mut ws).unwrap();
        assert!(
            (fit.second_stage.coefficients[0] - 2.0).abs() < 0.05,
            "beta_T={}",
            fit.second_stage.coefficients[0]
        );
        assert!((fit.second_stage.coefficients[2] - 1.5).abs() < 0.05);
    }
}
