//! Ridge and lasso utilities.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::needless_range_loop, clippy::similar_names)]

use crate::error::StatsError;
use crate::gram::{form_xtx, invert_square};
use crate::linalg::{DenseLinearAlgebra, FitDiagnostics, LeastSquaresFit, LeastSquaresWorkspace};

/// Options for [`fit_lasso`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LassoOptions {
    /// Maximum coordinate-descent iterations.
    pub max_iter: u32,
    /// Coefficient change tolerance.
    pub tol: f64,
}

impl Default for LassoOptions {
    fn default() -> Self {
        Self { max_iter: 1000, tol: 1e-6 }
    }
}

/// Result of a lasso fit.
#[derive(Clone, Debug)]
pub struct LassoFit {
    /// Coefficient vector (original scale, intercept in column 0 when present).
    pub coefficients: Vec<f64>,
    /// Coordinate-descent iterations used.
    pub iterations: u32,
    /// Whether the loop converged.
    pub converged: bool,
    /// Rank / condition / backend / allocation diagnostics.
    pub diagnostics: FitDiagnostics,
}

/// Ridge regression: solve `(XᵀX + λ I)β = Xᵀy`, leaving a constant intercept column unpenalized.
///
/// # Errors
///
/// Shape mismatch or singular penalized Gram matrix.
pub fn fit_ridge(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    lambda: f64,
    _backend: &impl DenseLinearAlgebra,
    _workspace: &mut LeastSquaresWorkspace,
) -> Result<LeastSquaresFit, StatsError> {
    if y.len() != nrows {
        return Err(StatsError::Shape { message: "y length != nrows" });
    }
    if x_colmajor.len() < nrows.saturating_mul(ncols) {
        return Err(StatsError::Shape { message: "X buffer too short" });
    }
    if !(lambda.is_finite() && lambda >= 0.0) {
        return Err(StatsError::Shape { message: "ridge lambda must be finite and ≥ 0" });
    }

    let mut xtx = vec![0.0; ncols * ncols];
    form_xtx(x_colmajor, nrows, ncols, &mut xtx);
    let unpenalize0 = col_is_constant(x_colmajor, nrows, 0);
    for c in 0..ncols {
        if c == 0 && unpenalize0 {
            continue;
        }
        xtx[c * ncols + c] += lambda;
    }
    let mut xty = vec![0.0; ncols];
    for c in 0..ncols {
        let mut s = 0.0;
        for r in 0..nrows {
            s += x_colmajor[c * nrows + r] * y[r];
        }
        xty[c] = s;
    }
    let Some(inv) = invert_square(&xtx, ncols) else {
        return Err(StatsError::Backend("ridge: singular X'X+λI".into()));
    };
    let mut coefficients = vec![0.0; ncols];
    for i in 0..ncols {
        let mut s = 0.0;
        for j in 0..ncols {
            s += inv[i * ncols + j] * xty[j];
        }
        coefficients[i] = s;
    }
    let mut residuals = vec![0.0; nrows];
    let mut rss = 0.0;
    for r in 0..nrows {
        let mut pred = 0.0;
        for c in 0..ncols {
            pred += x_colmajor[c * nrows + r] * coefficients[c];
        }
        let e = y[r] - pred;
        residuals[r] = e;
        rss += e * e;
    }
    Ok(LeastSquaresFit {
        coefficients,
        residuals,
        rank: ncols,
        rss,
        diagnostics: FitDiagnostics::new(ncols, None, "ridge", 0),
    })
}

/// Lasso via coordinate descent on centered predictors (soft-thresholding).
///
/// Column 0 is treated as an unpenalized intercept when constant.
///
/// # Errors
///
/// Shape mismatch or non-finite λ.
pub fn fit_lasso(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    lambda: f64,
    options: &LassoOptions,
) -> Result<LassoFit, StatsError> {
    if y.len() != nrows {
        return Err(StatsError::Shape { message: "y length != nrows" });
    }
    if x_colmajor.len() < nrows.saturating_mul(ncols) {
        return Err(StatsError::Shape { message: "X buffer too short" });
    }
    if !(lambda.is_finite() && lambda >= 0.0) {
        return Err(StatsError::Shape { message: "lasso lambda must be finite and ≥ 0" });
    }
    if nrows == 0 || ncols == 0 {
        return Err(StatsError::Shape { message: "lasso needs positive dimensions" });
    }

    let unpenalize0 = col_is_constant(x_colmajor, nrows, 0);
    let y_bar = y.iter().sum::<f64>() / nrows as f64;

    // Center columns (skip intercept); store means and column norms².
    let mut means = vec![0.0; ncols];
    let mut col_ss = vec![0.0; ncols];
    let mut xc = vec![0.0; nrows * ncols];
    for c in 0..ncols {
        if c == 0 && unpenalize0 {
            for r in 0..nrows {
                xc[c * nrows + r] = 1.0;
            }
            col_ss[c] = nrows as f64;
            continue;
        }
        let mut m = 0.0;
        for r in 0..nrows {
            m += x_colmajor[c * nrows + r];
        }
        m /= nrows as f64;
        means[c] = m;
        let mut ss = 0.0;
        for r in 0..nrows {
            let v = x_colmajor[c * nrows + r] - m;
            xc[c * nrows + r] = v;
            ss += v * v;
        }
        col_ss[c] = ss.max(1e-12);
    }

    let mut beta = vec![0.0; ncols];
    let mut residual: Vec<f64> = y.iter().map(|&yi| yi - y_bar).collect();
    let mut converged = false;
    let mut iterations = 0u32;

    for iter in 1..=options.max_iter {
        iterations = iter;
        let mut max_delta = 0.0_f64;
        for c in 0..ncols {
            if c == 0 && unpenalize0 {
                continue;
            }
            // Add back current coordinate contribution.
            if beta[c] != 0.0 {
                for r in 0..nrows {
                    residual[r] += xc[c * nrows + r] * beta[c];
                }
            }
            let mut rho = 0.0;
            for r in 0..nrows {
                rho += xc[c * nrows + r] * residual[r];
            }
            // Objective ½‖r‖² + λ‖β‖₁ → soft-threshold level λ.
            let new_b = soft_threshold(rho, lambda) / col_ss[c];
            max_delta = max_delta.max((new_b - beta[c]).abs());
            beta[c] = new_b;
            if beta[c] != 0.0 {
                for r in 0..nrows {
                    residual[r] -= xc[c * nrows + r] * beta[c];
                }
            }
        }
        if max_delta < options.tol {
            converged = true;
            break;
        }
    }

    if unpenalize0 {
        let mut intercept = y_bar;
        for c in 1..ncols {
            intercept -= means[c] * beta[c];
        }
        beta[0] = intercept;
    }

    Ok(LassoFit {
        coefficients: beta,
        iterations,
        converged,
        diagnostics: FitDiagnostics::new(ncols, None, "lasso", 0),
    })
}

fn soft_threshold(z: f64, gamma: f64) -> f64 {
    if z > gamma {
        z - gamma
    } else if z < -gamma {
        z + gamma
    } else {
        0.0
    }
}

fn col_is_constant(x_colmajor: &[f64], nrows: usize, col: usize) -> bool {
    if nrows == 0 {
        return true;
    }
    let base = col * nrows;
    let v0 = x_colmajor[base];
    x_colmajor[base..base + nrows].iter().all(|&v| (v - v0).abs() < 1e-12)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faer_backend::FaerBackend;

    #[test]
    fn ridge_shrinks_vs_ols() {
        let n = 40usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = (i as f64) / n as f64;
            x[i] = 1.0;
            x[n + i] = t;
            y[i] = 1.0 + 3.0 * t;
        }
        let mut ws = LeastSquaresWorkspace::default();
        let ols = FaerBackend.least_squares(&x, n, 2, &y, &mut ws).unwrap();
        let ridge = fit_ridge(&x, n, 2, &y, 5.0, &FaerBackend, &mut ws).unwrap();
        assert!(
            ridge.coefficients[1].abs() < ols.coefficients[1].abs(),
            "ridge={} ols={}",
            ridge.coefficients[1],
            ols.coefficients[1]
        );
        assert!(ridge.coefficients[1] > 0.0);
    }

    #[test]
    fn ridge_matches_closed_form_four_row() {
        let x = vec![1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 2.0, 3.0];
        let y = vec![1.0, 4.0, 7.0, 10.0];
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_ridge(&x, 4, 2, &y, 5.0, &FaerBackend, &mut ws).unwrap();
        assert!((fit.coefficients[0] - 3.25).abs() < 1e-9);
        assert!((fit.coefficients[1] - 1.5).abs() < 1e-9);
    }

    #[test]
    fn lasso_zeros_null_column() {
        let n = 60usize;
        let mut x = vec![0.0; n * 3];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = (i as f64) / n as f64;
            x[i] = 1.0;
            x[n + i] = t;
            x[2 * n + i] = ((i * 17) % 7) as f64 / 7.0; // weak noise column
            y[i] = 1.0 + 2.0 * t;
        }
        let fit = fit_lasso(&x, n, 3, &y, 2.0, &LassoOptions::default()).unwrap();
        assert!(fit.converged);
        assert!(fit.coefficients[2].abs() < 0.05, "noise coef={}", fit.coefficients[2]);
        assert!((fit.coefficients[1] - 2.0).abs() < 0.5, "slope={}", fit.coefficients[1]);
    }
}
