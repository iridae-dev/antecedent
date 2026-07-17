//! Huber M-estimation for linear regression (DESIGN.md §11.2).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::needless_range_loop)]

use crate::error::StatsError;
use crate::linalg::{DenseLinearAlgebra, FitDiagnostics, LeastSquaresWorkspace};
use crate::twosls::fit_wls;

/// Options for [`fit_huber_m`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct MEstimateOptions {
    /// Maximum outer IRLS iterations.
    pub max_iter: u32,
    /// Coefficient change tolerance.
    pub tol: f64,
    /// Huber tuning constant (default 1.345 ≈ 95% efficiency under Gaussian).
    pub c: f64,
}

impl Default for MEstimateOptions {
    fn default() -> Self {
        Self { max_iter: 50, tol: 1e-8, c: 1.345 }
    }
}

/// Result of Huber M-estimation.
#[derive(Clone, Debug)]
pub struct MEstimateFit {
    /// Coefficient vector.
    pub coefficients: Vec<f64>,
    /// Robust scale estimate (MAD-based).
    pub scale: f64,
    /// Outer iterations used.
    pub iterations: u32,
    /// Whether the outer loop converged.
    pub converged: bool,
    /// Rank / condition / backend / allocation diagnostics.
    pub diagnostics: FitDiagnostics,
}

/// Fit a Huber M-estimator via IRLS with MAD scale updates.
///
/// # Errors
///
/// Shape mismatch or WLS backend failure.
pub fn fit_huber_m(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    options: &MEstimateOptions,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
) -> Result<MEstimateFit, StatsError> {
    if y.len() != nrows {
        return Err(StatsError::Shape { message: "y length != nrows" });
    }
    if x_colmajor.len() < nrows.saturating_mul(ncols) {
        return Err(StatsError::Shape { message: "X buffer too short" });
    }
    if !(options.c.is_finite() && options.c > 0.0) {
        return Err(StatsError::Shape { message: "Huber c must be finite and > 0" });
    }

    // OLS warm start.
    let ols = backend.least_squares(x_colmajor, nrows, ncols, y, workspace)?;
    let mut beta = ols.coefficients;
    let mut weights = vec![1.0; nrows];
    let mut residuals = vec![0.0; nrows];
    let mut converged = false;
    let mut iterations = 0u32;
    let mut scale = 1.0;

    for iter in 1..=options.max_iter {
        iterations = iter;
        for r in 0..nrows {
            let mut pred = 0.0;
            for c in 0..ncols {
                pred += x_colmajor[c * nrows + r] * beta[c];
            }
            residuals[r] = y[r] - pred;
        }
        scale = mad_scale(&residuals).max(1e-12);
        for r in 0..nrows {
            let u = residuals[r] / scale;
            let au = u.abs();
            weights[r] = if au <= options.c || au < 1e-15 {
                1.0
            } else {
                options.c / au
            };
        }
        let fit = fit_wls(x_colmajor, nrows, ncols, y, &weights, backend, workspace)?;
        let mut max_delta = 0.0_f64;
        for c in 0..ncols {
            max_delta = max_delta.max((fit.coefficients[c] - beta[c]).abs());
            beta[c] = fit.coefficients[c];
        }
        if max_delta < options.tol {
            converged = true;
            break;
        }
    }

    Ok(MEstimateFit {
        coefficients: beta,
        scale,
        iterations,
        converged,
        diagnostics: FitDiagnostics::new(ncols, None, "huber", workspace.grow_count),
    })
}

fn mad_scale(residuals: &[f64]) -> f64 {
    if residuals.is_empty() {
        return 1.0;
    }
    let mut sorted = residuals.to_vec();
    sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mid = sorted.len() / 2;
    let center = if sorted.len() % 2 == 0 {
        0.5 * (sorted[mid - 1] + sorted[mid])
    } else {
        sorted[mid]
    };
    let mut abs_dev: Vec<f64> = residuals.iter().map(|r| (r - center).abs()).collect();
    abs_dev.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    let mad = if abs_dev.len() % 2 == 0 {
        0.5 * (abs_dev[mid - 1] + abs_dev[mid])
    } else {
        abs_dev[mid]
    };
    // Consistency constant for Gaussian MAD → σ.
    1.4826 * mad
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faer_backend::FaerBackend;

    #[test]
    fn huber_matches_ols_on_clean_data() {
        let n = 80usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = (i as f64) / n as f64;
            x[i] = 1.0;
            x[n + i] = t;
            y[i] = 1.0 + 2.0 * t;
        }
        let mut ws = LeastSquaresWorkspace::default();
        let ols = FaerBackend.least_squares(&x, n, 2, &y, &mut ws).unwrap();
        let fit = fit_huber_m(&x, n, 2, &y, &MEstimateOptions::default(), &FaerBackend, &mut ws)
            .unwrap();
        assert!(fit.converged);
        assert!((fit.coefficients[1] - ols.coefficients[1]).abs() < 1e-6);
    }

    #[test]
    fn huber_downweights_outlier_vs_ols() {
        let n = 50usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = (i as f64) / n as f64;
            x[i] = 1.0;
            x[n + i] = t;
            y[i] = 1.0 + 2.0 * t;
        }
        y[n - 1] = 100.0; // plant outlier
        let mut ws = LeastSquaresWorkspace::default();
        let ols = FaerBackend.least_squares(&x, n, 2, &y, &mut ws).unwrap();
        let fit = fit_huber_m(&x, n, 2, &y, &MEstimateOptions::default(), &FaerBackend, &mut ws)
            .unwrap();
        assert!(fit.converged);
        // Huber slope should stay closer to 2 than OLS.
        assert!(
            (fit.coefficients[1] - 2.0).abs() < (ols.coefficients[1] - 2.0).abs(),
            "huber={} ols={}",
            fit.coefficients[1],
            ols.coefficients[1]
        );
    }

    #[test]
    fn mad_scale_centers_before_median_abs_dev() {
        // Residuals with nonzero median: MAD must use median(|r − med(r)|).
        let r = [-2.0_f64, -1.0, 0.0, 1.0, 10.0];
        let scale = mad_scale(&r);
        // med(r)=0, mad=1, scale=1.4826
        assert!((scale - 1.4826).abs() < 1e-12);
        let r2 = [8.0_f64, 9.0, 10.0, 11.0, 20.0]; // shifted by +10
        assert!((mad_scale(&r2) - scale).abs() < 1e-12);
    }
}
