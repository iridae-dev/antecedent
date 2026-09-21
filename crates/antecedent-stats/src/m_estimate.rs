//! Huber (1964) M-estimation for linear regression.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::needless_range_loop)]

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
    /// Robust scale estimate at the final iteration: `1.4826 · MAD`, or the mean-absolute-
    /// deviation fallback when [`Self::scale_fallback`] is set. `0.0` means the residuals
    /// are numerically all equal (an exact fit), for which every weight is 1.
    pub scale: f64,
    /// The MAD was exactly zero (more than half of the residuals coincide, as for a
    /// discrete or zero-inflated outcome), so the scale is `√(π/2) ·` the mean absolute
    /// deviation about the median instead — the same Gaussian-consistent units, positive
    /// whenever any residual differs. A zero MAD would otherwise collapse the scale and
    /// give every non-majority residual weight `≈ c·scale/|r| → 0`.
    pub scale_fallback: bool,
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
    let mut scale_fallback = false;
    // Residuals below this are rounding noise of the fit itself, not spread to be scaled.
    let noise_floor = 1e3 * f64::EPSILON * y.iter().fold(0.0_f64, |m, v| m.max(v.abs()));

    for iter in 1..=options.max_iter {
        iterations = iter;
        for r in 0..nrows {
            let mut pred = 0.0;
            for c in 0..ncols {
                pred += x_colmajor[c * nrows + r] * beta[c];
            }
            residuals[r] = y[r] - pred;
        }
        (scale, scale_fallback) = robust_scale(&residuals, noise_floor);
        for r in 0..nrows {
            weights[r] = if scale > 0.0 {
                let au = (residuals[r] / scale).abs();
                if au <= options.c || au < 1e-15 { 1.0 } else { options.c / au }
            } else {
                1.0
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
        scale_fallback,
        iterations,
        converged,
        diagnostics: FitDiagnostics::new(ncols, None, "huber", workspace.grow_count),
    })
}

/// Robust residual scale and whether the mean-absolute-deviation fallback was used.
///
/// `1.4826 · MAD` when that is above `noise_floor`. If the MAD is zero because more than
/// half of the residuals coincide, `√(π/2)` times the mean absolute deviation about the
/// median (Gaussian-consistent; positive whenever any residual differs). Returns `0.0`
/// when even that is at rounding-noise level: the residuals are all equal.
fn robust_scale(residuals: &[f64], noise_floor: f64) -> (f64, bool) {
    let mad = crate::quantile::mad_sigma(residuals).unwrap_or(1.0);
    if mad > noise_floor {
        return (mad, false);
    }
    let mut sorted = residuals.to_vec();
    sorted.sort_by(f64::total_cmp);
    let center = crate::quantile::median_sorted(&sorted);
    let mean_ad =
        residuals.iter().map(|r| (r - center).abs()).sum::<f64>() / residuals.len().max(1) as f64;
    let fallback = (std::f64::consts::FRAC_PI_2).sqrt() * mean_ad;
    if fallback > noise_floor { (fallback, true) } else { (0.0, false) }
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
        let fit =
            fit_huber_m(&x, n, 2, &y, &MEstimateOptions::default(), &FaerBackend, &mut ws).unwrap();
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
        let fit =
            fit_huber_m(&x, n, 2, &y, &MEstimateOptions::default(), &FaerBackend, &mut ws).unwrap();
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
    fn zero_mad_falls_back_to_mean_absolute_deviation_instead_of_collapsing() {
        // Intercept-only fit of 30 zeros and 10 ones. Every residual of a zero is the same
        // number, so MAD = 0 at every iteration; the old 1e-12 scale gave the ones weight
        // ~1e-12 and the fit collapsed onto the majority (coefficient ≈ 0, "scale" 1e-12).
        // Fallback scale s = √(π/2) · meanAD, meanAD = 10·1/40 = 1/4 (ones sit 1 above the
        // median residual). The fixed point of β = 10 w/(30 + 10 w), w = c s/(1 − β) is
        // β = c s / 3.
        let n = 40usize;
        let x = vec![1.0; n];
        let y: Vec<f64> = (0..n).map(|i| if i < 10 { 1.0 } else { 0.0 }).collect();
        let mut ws = LeastSquaresWorkspace::default();
        let opts = MEstimateOptions::default();
        let fit = fit_huber_m(&x, n, 1, &y, &opts, &FaerBackend, &mut ws).unwrap();
        let s = (std::f64::consts::FRAC_PI_2).sqrt() * 0.25;
        assert!(fit.converged && fit.scale_fallback);
        assert!((fit.scale - s).abs() < 1e-12, "scale={}", fit.scale);
        let expected = opts.c * s / 3.0;
        assert!(
            (fit.coefficients[0] - expected).abs() < 1e-6,
            "{} vs {expected}",
            fit.coefficients[0]
        );
    }

    #[test]
    fn exactly_fitted_data_reports_zero_scale_without_fallback() {
        let n = 20usize;
        let mut x = vec![1.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            x[n + i] = i as f64;
            y[i] = 3.0 + 0.5 * i as f64;
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit =
            fit_huber_m(&x, n, 2, &y, &MEstimateOptions::default(), &FaerBackend, &mut ws).unwrap();
        assert_eq!(fit.scale, 0.0);
        assert!(!fit.scale_fallback && fit.converged);
        assert!(
            (fit.coefficients[0] - 3.0).abs() < 1e-9 && (fit.coefficients[1] - 0.5).abs() < 1e-9
        );
    }

    #[test]
    fn mad_scale_centers_before_median_abs_dev() {
        // Residuals with nonzero median: MAD must use median(|r − med(r)|).
        let r = [-2.0_f64, -1.0, 0.0, 1.0, 10.0];
        let scale = crate::quantile::mad_sigma(&r).unwrap();
        // med(r)=0, mad=1, scale=1.4826
        assert!((scale - 1.4826).abs() < 1e-12);
        let r2 = [8.0_f64, 9.0, 10.0, 11.0, 20.0]; // shifted by +10
        assert!((crate::quantile::mad_sigma(&r2).unwrap() - scale).abs() < 1e-12);
    }
}
