//! Ridge and lasso utilities.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::similar_names,
    clippy::too_many_lines
)]

use crate::error::StatsError;
use crate::gram::{form_xtx, invert_square};
use crate::linalg::{DenseLinearAlgebra, FitDiagnostics, LeastSquaresFit, LeastSquaresWorkspace};

/// Minimum positive feature scale accepted after centering / RMS.
const MIN_SCALE: f64 = 1e-12;

/// Options for [`fit_lasso`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LassoOptions {
    /// L1 penalty strength (non-negative).
    pub lambda: f64,
    /// Fit an unpenalized intercept via centering (not taken from an X column).
    pub fit_intercept: bool,
    /// Scale columns by a positive RMS (centered when `fit_intercept`, raw otherwise).
    pub standardize: bool,
    /// Maximum coordinate-descent iterations.
    pub max_iter: u32,
    /// Coefficient change tolerance.
    pub tol: f64,
}

impl Default for LassoOptions {
    fn default() -> Self {
        Self { lambda: 0.0, fit_intercept: true, standardize: false, max_iter: 1000, tol: 1e-6 }
    }
}

/// Result of a lasso fit.
#[derive(Clone, Debug)]
pub struct LassoFit {
    /// Unpenalized intercept (`0` when [`LassoOptions::fit_intercept`] is false).
    pub intercept: f64,
    /// Slope coefficients on the original feature scale (length = `ncols`).
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

/// Lasso via coordinate descent with an explicit intercept / standardization contract.
///
/// Predictors are never treated as an intercept column. When
/// [`LassoOptions::fit_intercept`] is true, `X` and `y` are centered, the intercept is
/// recovered after fitting, and it is never penalized. Optional standardization divides by a
/// positive column scale and coefficients are back-transformed before return.
///
/// # Errors
///
/// Shape mismatch, non-finite λ, or a non-positive feature scale under standardization /
/// centering (e.g. a constant predictor when `fit_intercept` and `standardize` are set).
pub fn fit_lasso(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    options: &LassoOptions,
) -> Result<LassoFit, StatsError> {
    if y.len() != nrows {
        return Err(StatsError::Shape { message: "y length != nrows" });
    }
    if x_colmajor.len() < nrows.saturating_mul(ncols) {
        return Err(StatsError::Shape { message: "X buffer too short" });
    }
    if !(options.lambda.is_finite() && options.lambda >= 0.0) {
        return Err(StatsError::Shape { message: "lasso lambda must be finite and ≥ 0" });
    }
    if nrows == 0 {
        return Err(StatsError::Shape { message: "lasso needs positive dimensions" });
    }

    let n = nrows as f64;
    let y_mean = if options.fit_intercept { y.iter().sum::<f64>() / n } else { 0.0 };

    let mut means = vec![0.0; ncols];
    let mut scales = vec![1.0; ncols];
    let mut xc = vec![0.0; nrows * ncols];
    let mut col_ss = vec![0.0; ncols];

    for c in 0..ncols {
        let base = c * nrows;
        let mean = if options.fit_intercept {
            x_colmajor[base..base + nrows].iter().sum::<f64>() / n
        } else {
            0.0
        };
        means[c] = mean;

        let mut sum_sq = 0.0;
        for r in 0..nrows {
            let v = x_colmajor[base + r] - mean;
            xc[base + r] = v;
            sum_sq += v * v;
        }

        let scale = if options.standardize {
            let s = (sum_sq / n).sqrt();
            if !(s.is_finite() && s > MIN_SCALE) {
                return Err(StatsError::Shape {
                    message: "lasso feature scale must be positive (constant or zero column)",
                });
            }
            s
        } else {
            1.0
        };
        scales[c] = scale;

        if options.standardize {
            for r in 0..nrows {
                xc[base + r] /= scale;
            }
            sum_sq = 0.0;
            for r in 0..nrows {
                let v = xc[base + r];
                sum_sq += v * v;
            }
        }

        // Constant columns after centering (no standardization) stay at zero; CD leaves β=0.
        // A raw zero column with fit_intercept=false cannot identify a slope.
        if sum_sq <= MIN_SCALE {
            if !options.fit_intercept && col_all_near_zero(x_colmajor, nrows, c) {
                return Err(StatsError::Shape {
                    message: "lasso feature scale must be positive (constant or zero column)",
                });
            }
            col_ss[c] = 0.0;
        } else {
            col_ss[c] = sum_sq;
        }
    }

    let mut beta_std = vec![0.0_f64; ncols];
    let mut residual: Vec<f64> = y.iter().map(|&yi| yi - y_mean).collect();
    let mut converged = false;
    let mut iterations = 0u32;
    let lambda = options.lambda;

    for iter in 1..=options.max_iter {
        iterations = iter;
        let mut max_delta = 0.0_f64;
        for c in 0..ncols {
            if col_ss[c] <= MIN_SCALE {
                continue;
            }
            let beta_c = beta_std[c];
            if beta_c.abs() > f64::MIN_POSITIVE {
                for r in 0..nrows {
                    residual[r] += xc[c * nrows + r] * beta_c;
                }
            }
            let mut rho = 0.0_f64;
            for r in 0..nrows {
                rho += xc[c * nrows + r] * residual[r];
            }
            // Objective ½‖r‖² + λ‖β‖₁ → soft-threshold level λ.
            let new_b = soft_threshold(rho, lambda) / col_ss[c];
            max_delta = max_delta.max((new_b - beta_c).abs());
            beta_std[c] = new_b;
            if new_b.abs() > f64::MIN_POSITIVE {
                for r in 0..nrows {
                    residual[r] -= xc[c * nrows + r] * new_b;
                }
            }
        }
        if max_delta < options.tol {
            converged = true;
            break;
        }
    }

    let mut coefficients = vec![0.0; ncols];
    for c in 0..ncols {
        coefficients[c] = beta_std[c] / scales[c];
    }

    let intercept = if options.fit_intercept {
        let mut b0 = y_mean;
        for c in 0..ncols {
            b0 -= means[c] * coefficients[c];
        }
        b0
    } else {
        0.0
    };

    Ok(LassoFit {
        intercept,
        coefficients,
        iterations,
        converged,
        diagnostics: FitDiagnostics::new(ncols, None, "lasso", 0),
    })
}

/// Predict `intercept + Xβ` for a [`LassoFit`].
///
/// # Errors
///
/// Shape mismatch between `fit.coefficients` and `ncols`, or a short `X` buffer.
pub fn predict_lasso(
    fit: &LassoFit,
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
) -> Result<Vec<f64>, StatsError> {
    if fit.coefficients.len() != ncols {
        return Err(StatsError::Shape { message: "lasso coefficient length != ncols" });
    }
    if x_colmajor.len() < nrows.saturating_mul(ncols) {
        return Err(StatsError::Shape { message: "X buffer too short" });
    }
    let mut pred = vec![fit.intercept; nrows];
    for c in 0..ncols {
        let b = fit.coefficients[c];
        if b.abs() <= f64::MIN_POSITIVE {
            continue;
        }
        let base = c * nrows;
        for r in 0..nrows {
            pred[r] += x_colmajor[base + r] * b;
        }
    }
    Ok(pred)
}

/// Compatibility entry point for designs whose first column is an exact all-ones intercept.
///
/// Strips column 0, fits with [`LassoOptions::fit_intercept`] forced on, and returns the
/// intercept in [`LassoFit::intercept`] with slopes for the remaining columns. Non-ones constant
/// first columns are not reinterpreted as intercepts — they go through [`fit_lasso`] and fail
/// zero-scale validation when appropriate.
///
/// # Errors
///
/// Same as [`fit_lasso`], or empty design after stripping a lone ones column.
pub fn fit_lasso_with_ones_column(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    options: &LassoOptions,
) -> Result<LassoFit, StatsError> {
    if ncols == 0 || !first_col_is_exact_ones(x_colmajor, nrows) {
        return fit_lasso(x_colmajor, nrows, ncols, y, options);
    }
    if ncols == 1 {
        // Intercept-only model: mean of y.
        if y.len() != nrows {
            return Err(StatsError::Shape { message: "y length != nrows" });
        }
        if nrows == 0 {
            return Err(StatsError::Shape { message: "lasso needs positive dimensions" });
        }
        let intercept = y.iter().sum::<f64>() / nrows as f64;
        return Ok(LassoFit {
            intercept,
            coefficients: Vec::new(),
            iterations: 0,
            converged: true,
            diagnostics: FitDiagnostics::new(0, None, "lasso", 0),
        });
    }
    let rest = &x_colmajor[nrows..];
    let mut opts = *options;
    opts.fit_intercept = true;
    fit_lasso(rest, nrows, ncols - 1, y, &opts)
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

fn first_col_is_exact_ones(x_colmajor: &[f64], nrows: usize) -> bool {
    // Exact bit pattern required so all-twos (etc.) are never treated as intercept.
    #[allow(clippy::float_cmp)]
    {
        nrows > 0 && x_colmajor.len() >= nrows && x_colmajor[..nrows].iter().all(|&v| v == 1.0)
    }
}

fn col_all_near_zero(x_colmajor: &[f64], nrows: usize, col: usize) -> bool {
    let base = col * nrows;
    x_colmajor[base..base + nrows].iter().all(|&v| v.abs() <= MIN_SCALE)
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
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = (i as f64) / n as f64;
            x[i] = t;
            x[n + i] = ((i * 17) % 7) as f64 / 7.0;
            y[i] = 1.0 + 2.0 * t;
        }
        let fit = fit_lasso(
            &x,
            n,
            2,
            &y,
            &LassoOptions { lambda: 2.0, fit_intercept: true, ..LassoOptions::default() },
        )
        .unwrap();
        assert!(fit.converged);
        assert!(fit.coefficients[1].abs() < 0.05, "noise coef={}", fit.coefficients[1]);
        assert!((fit.coefficients[0] - 2.0).abs() < 0.5, "slope={}", fit.coefficients[0]);
        assert!((fit.intercept - 1.0).abs() < 0.5, "intercept={}", fit.intercept);
    }

    #[test]
    fn lasso_no_intercept_column_recovers_nonzero_response_mean() {
        let x = [1.0_f64, 2.0, 3.0, 4.0];
        let y = [5.0, 7.0, 9.0, 11.0]; // 3 + 2x
        let fit = fit_lasso(
            &x,
            4,
            1,
            &y,
            &LassoOptions {
                lambda: 0.0,
                fit_intercept: true,
                max_iter: 10_000,
                tol: 1e-12,
                ..LassoOptions::default()
            },
        )
        .unwrap();
        assert!((fit.intercept - 3.0).abs() <= 1e-9);
        assert!((fit.coefficients[0] - 2.0).abs() <= 1e-9);
        let pred = predict_lasso(&fit, &x, 4, 1).unwrap();
        for (p, yi) in pred.iter().zip(&y) {
            assert!((p - yi).abs() <= 1e-9);
        }
    }

    #[test]
    fn lasso_legacy_ones_column_matches_explicit_intercept_api() {
        let n = 5usize;
        let mut x_legacy = vec![0.0; n * 2];
        let mut x_explicit = vec![0.0; n];
        let mut y = vec![0.0; n];
        for r in 0..n {
            let t = r as f64;
            x_legacy[r] = 1.0;
            x_legacy[n + r] = t;
            x_explicit[r] = t;
            y[r] = 1.5 + 0.75 * t;
        }
        let opts = LassoOptions {
            lambda: 0.0,
            fit_intercept: true,
            max_iter: 10_000,
            tol: 1e-12,
            ..LassoOptions::default()
        };
        let legacy = fit_lasso_with_ones_column(&x_legacy, n, 2, &y, &opts).unwrap();
        let explicit = fit_lasso(&x_explicit, n, 1, &y, &opts).unwrap();
        assert!((legacy.intercept - explicit.intercept).abs() <= 1e-12);
        assert!((legacy.coefficients[0] - explicit.coefficients[0]).abs() <= 1e-12);
        assert!((legacy.intercept - 1.5).abs() <= 1e-9);
        assert!((legacy.coefficients[0] - 0.75).abs() <= 1e-9);
    }

    #[test]
    fn lasso_all_twos_column_is_not_misreported_as_intercept() {
        let x = [2.0_f64, 2.0];
        let y = [4.0, 4.0];
        // Without intercept: through-origin slope 2 on the all-twos column.
        let no_int = fit_lasso(
            &x,
            2,
            1,
            &y,
            &LassoOptions {
                lambda: 0.0,
                fit_intercept: false,
                max_iter: 10_000,
                tol: 1e-12,
                ..LassoOptions::default()
            },
        )
        .unwrap();
        assert!(no_int.intercept.abs() <= 1e-15);
        assert!((no_int.coefficients[0] - 2.0).abs() <= 1e-12);

        // With intercept: mean absorbed into intercept; all-twos is not an intercept coef.
        let with_int = fit_lasso(
            &x,
            2,
            1,
            &y,
            &LassoOptions {
                lambda: 0.0,
                fit_intercept: true,
                standardize: false,
                max_iter: 10_000,
                tol: 1e-12,
            },
        )
        .unwrap();
        assert!((with_int.intercept - 4.0).abs() <= 1e-12);
        assert!(with_int.coefficients[0].abs() <= 1e-15);

        // Standardized fit must reject the zero-scale centered column.
        let err = fit_lasso(
            &x,
            2,
            1,
            &y,
            &LassoOptions {
                lambda: 0.0,
                fit_intercept: true,
                standardize: true,
                max_iter: 100,
                tol: 1e-8,
            },
        )
        .unwrap_err();
        assert!(matches!(err, StatsError::Shape { .. }));
    }

    #[test]
    fn lasso_standardized_and_raw_agree_after_backtransform() {
        let n = 30usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for r in 0..n {
            let t = r as f64 - 14.5;
            x[r] = t;
            x[n + r] = (r as f64 * 0.4).sin() + 0.1 * t;
            y[r] = 2.0 + 1.3 * x[r] - 0.7 * x[n + r];
        }
        // Same λ on differently scaled features is not equivalent; back-transform
        // identity is checked at λ = 0 (unpenalized centered/scaled OLS path).
        let base = LassoOptions {
            lambda: 0.0,
            fit_intercept: true,
            max_iter: 20_000,
            tol: 1e-12,
            standardize: false,
        };
        let raw = fit_lasso(&x, n, 2, &y, &base).unwrap();
        let std = fit_lasso(&x, n, 2, &y, &LassoOptions { standardize: true, ..base }).unwrap();
        let pred_raw = predict_lasso(&raw, &x, n, 2).unwrap();
        let pred_std = predict_lasso(&std, &x, n, 2).unwrap();
        for (a, b) in pred_raw.iter().zip(&pred_std) {
            assert!((a - b).abs() <= 1e-6, "{a} vs {b}");
        }
    }

    #[test]
    fn lasso_fit_intercept_false_preserves_through_origin() {
        // y = 1.5 x with no intercept term in the DGP.
        let x = [1.0_f64, 2.0, 3.0, 4.0];
        let y = [1.5, 3.0, 4.5, 6.0];
        let fit = fit_lasso(
            &x,
            4,
            1,
            &y,
            &LassoOptions {
                lambda: 0.0,
                fit_intercept: false,
                standardize: false,
                max_iter: 10_000,
                tol: 1e-12,
            },
        )
        .unwrap();
        assert!(fit.intercept.abs() <= 1e-15);
        assert!((fit.coefficients[0] - 1.5).abs() <= 1e-12);
        let pred = predict_lasso(&fit, &x, 4, 1).unwrap();
        for (p, yi) in pred.iter().zip(&y) {
            assert!((p - yi).abs() <= 1e-12);
        }

        // Standardization without intercept must not shift origin.
        let fit_s = fit_lasso(
            &x,
            4,
            1,
            &y,
            &LassoOptions {
                lambda: 0.0,
                fit_intercept: false,
                standardize: true,
                max_iter: 10_000,
                tol: 1e-12,
            },
        )
        .unwrap();
        assert!(fit_s.intercept.abs() <= 1e-15);
        assert!((fit_s.coefficients[0] - 1.5).abs() <= 1e-12);
    }

    #[test]
    fn lasso_without_intercept_matches_closed_form_at_zero_penalty() {
        let fit = fit_lasso(
            &[1.0, 2.0],
            2,
            1,
            &[2.0, 3.0],
            &LassoOptions {
                lambda: 0.0,
                fit_intercept: false,
                max_iter: 10_000,
                tol: 1e-12,
                ..LassoOptions::default()
            },
        )
        .unwrap();
        assert!(fit.intercept.abs() <= 1e-15);
        assert!((fit.coefficients[0] - 1.6).abs() <= 1e-12);
    }

    #[test]
    fn zero_penalty_lasso_matches_ols_slopes_with_intercept() {
        let n = 40usize;
        let options = LassoOptions {
            lambda: 0.0,
            fit_intercept: true,
            max_iter: 10_000,
            tol: 1e-12,
            ..LassoOptions::default()
        };
        let mut x = vec![0.0; n * 2];
        let mut x_ols = vec![0.0; n * 3];
        let mut y = vec![0.0; n];
        for r in 0..n {
            let t = r as f64 - 19.5;
            x[r] = (r as f64 * 0.37).sin() + 0.02 * t;
            x[n + r] = (r as f64 * 0.23).cos() - 0.01 * t;
            x_ols[r] = 1.0;
            x_ols[n + r] = x[r];
            x_ols[2 * n + r] = x[n + r];
            y[r] = 1.7 - 0.8 * x[r] + 0.45 * x[n + r];
        }
        let mut ws = LeastSquaresWorkspace::default();
        let ols = FaerBackend.least_squares(&x_ols, n, 3, &y, &mut ws).unwrap();
        let lasso = fit_lasso(&x, n, 2, &y, &options).unwrap();
        assert!(lasso.converged);
        assert!((lasso.intercept - ols.coefficients[0]).abs() <= 1e-8);
        assert!((lasso.coefficients[0] - ols.coefficients[1]).abs() <= 1e-8);
        assert!((lasso.coefficients[1] - ols.coefficients[2]).abs() <= 1e-8);
    }

    #[test]
    fn lasso_no_intercept_satisfies_coordinate_kkt_conditions() {
        let nrows = 25usize;
        let ncols = 3usize;
        let mut design = vec![0.0; nrows * ncols];
        let mut response = vec![0.0; nrows];
        for row in 0..nrows {
            let t = row as f64 - 12.0;
            design[row] = 0.2 * t + 0.3;
            design[nrows + row] = (row as f64 * 0.7).sin();
            design[2 * nrows + row] = (row as f64 * 0.31).cos() - 0.2;
            response[row] =
                1.4 * design[row] - 0.8 * design[nrows + row] + 0.05 * (row as f64).sin();
        }
        let lambda = 0.4;
        let fit = fit_lasso(
            &design,
            nrows,
            ncols,
            &response,
            &LassoOptions {
                lambda,
                fit_intercept: false,
                max_iter: 10_000,
                tol: 1e-12,
                ..LassoOptions::default()
            },
        )
        .unwrap();
        assert!(fit.converged);
        assert!(fit.intercept.abs() <= 1e-15);
        for col in 0..ncols {
            let grad = (0..nrows)
                .map(|row| {
                    let pred = fit.intercept
                        + (0..ncols)
                            .map(|j| design[j * nrows + row] * fit.coefficients[j])
                            .sum::<f64>();
                    design[col * nrows + row] * (response[row] - pred)
                })
                .sum::<f64>();
            if fit.coefficients[col].abs() > 1e-9 {
                let expected = lambda * fit.coefficients[col].signum();
                assert!((grad - expected).abs() <= 1e-7, "col={col} grad={grad}");
            } else {
                assert!(grad.abs() <= lambda + 1e-7, "col={col} grad={grad}");
            }
        }
    }
}
