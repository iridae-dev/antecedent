//! GLM fitting: logistic, Gaussian/identity, and Poisson/log IRLS.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::needless_range_loop, clippy::float_cmp)]

use causal_kernels::norm_cdf;

use crate::error::StatsError;
use crate::linalg::{DenseLinearAlgebra, LeastSquaresWorkspace};

/// Family for the GLM path.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum GlmFamily {
    /// Binomial / Bernoulli with logit link.
    BinomialLogit,
    /// Binomial / Bernoulli with probit link.
    BinomialProbit,
    /// Gaussian with identity link (OLS).
    GaussianIdentity,
    /// Poisson with log link.
    PoissonLog,
}

impl GlmFamily {
    /// Mean on the response scale given linear predictor `eta`.
    #[must_use]
    pub fn mean_from_eta(self, eta: f64) -> f64 {
        match self {
            Self::BinomialLogit => 1.0 / (1.0 + (-eta).exp()),
            Self::BinomialProbit => norm_cdf(eta),
            Self::GaussianIdentity => eta,
            Self::PoissonLog => eta.exp(),
        }
    }
}

/// Borrowed column-major design + outcome used by [`fit_glm`].
#[derive(Clone, Copy, Debug)]
pub struct GlmDesignRef<'a> {
    /// Column-major design matrix.
    pub x_colmajor: &'a [f64],
    /// Rows.
    pub nrows: usize,
    /// Columns.
    pub ncols: usize,
    /// Outcome aligned with rows.
    pub y: &'a [f64],
}

/// Fitting options for [`fit_glm`].
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GlmOptions {
    /// Maximum IRLS iterations.
    pub max_iter: u32,
    /// Coefficient change tolerance for convergence.
    pub tol: f64,
}

impl Default for GlmOptions {
    fn default() -> Self {
        Self { max_iter: 50, tol: 1e-8 }
    }
}

impl GlmOptions {
    /// Construct options.
    #[must_use]
    pub const fn new(max_iter: u32, tol: f64) -> Self {
        Self { max_iter, tol }
    }
}

/// Convergence / iteration diagnostics from a GLM fit.
#[derive(Clone, Debug)]
pub struct GlmFit {
    /// Coefficient vector.
    pub coefficients: Vec<f64>,
    /// Iterations used.
    pub iterations: u32,
    /// Whether the IRLS loop converged.
    pub converged: bool,
    /// Whether fitted means hit the soft clamp band (logistic separation signal).
    pub separated: bool,
    /// Final deviance.
    pub deviance: f64,
}

impl GlmFit {
    /// Error if IRLS failed to converge or logistic separation was detected.
    ///
    /// # Errors
    ///
    /// Non-converged or separated fits.
    pub fn require_ok(&self) -> Result<(), StatsError> {
        if !self.converged {
            return Err(StatsError::Backend(
                "GLM IRLS did not converge; refuse propensity/outcome scores".into(),
            ));
        }
        if self.separated {
            return Err(StatsError::Backend(
                "GLM indicates (quasi-)complete separation; refuse propensity/outcome scores"
                    .into(),
            ));
        }
        Ok(())
    }
}

/// Fit a GLM on a compiled column-major design via IRLS + least squares.
///
/// # Errors
///
/// Shape mismatch, invalid outcomes for the family, or linear-algebra failure.
pub fn fit_glm(
    family: GlmFamily,
    design: GlmDesignRef<'_>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<GlmFit, StatsError> {
    match family {
        GlmFamily::BinomialLogit => fit_logistic(design, backend, workspace, options),
        GlmFamily::BinomialProbit => Err(StatsError::Backend(
            "BinomialProbit frequentist IRLS is not implemented; use Bayesian Laplace".into(),
        )),
        GlmFamily::GaussianIdentity => fit_gaussian(design, backend, workspace),
        GlmFamily::PoissonLog => fit_poisson(design, backend, workspace, options),
    }
}

fn fit_gaussian(
    design: GlmDesignRef<'_>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
) -> Result<GlmFit, StatsError> {
    let GlmDesignRef { x_colmajor, nrows, ncols, y } = design;
    if y.len() != nrows {
        return Err(StatsError::Shape { message: "y length != nrows" });
    }
    if x_colmajor.len() < nrows.saturating_mul(ncols) {
        return Err(StatsError::Shape { message: "X buffer too short" });
    }
    let fit = backend.least_squares(x_colmajor, nrows, ncols, y, workspace)?;
    Ok(GlmFit {
        coefficients: fit.coefficients,
        iterations: 1,
        converged: true,
        separated: false,
        deviance: fit.rss,
    })
}

fn fit_poisson(
    design: GlmDesignRef<'_>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<GlmFit, StatsError> {
    let GlmDesignRef { x_colmajor, nrows, ncols, y } = design;
    if y.len() != nrows {
        return Err(StatsError::Shape { message: "y length != nrows" });
    }
    if x_colmajor.len() < nrows.saturating_mul(ncols) {
        return Err(StatsError::Shape { message: "X buffer too short" });
    }
    for &yi in y {
        if !(yi.is_finite() && yi >= 0.0) {
            return Err(StatsError::Shape {
                message: "Poisson GLM requires non-negative outcomes",
            });
        }
    }

    let mut beta = vec![0.0; ncols];
    let mut x_w = vec![0.0; nrows * ncols];
    let mut z = vec![0.0; nrows];
    let mut converged = false;
    let mut iterations = 0u32;
    let mut deviance = f64::INFINITY;

    for iter in 1..=options.max_iter {
        iterations = iter;
        let mut max_delta = 0.0_f64;
        deviance = 0.0;
        for r in 0..nrows {
            // Standard IRLS initialization: start from the data (eta0 = ln(y + 0.5))
            // rather than beta = 0, which diverges for ordinary count magnitudes.
            let eta = if iter == 1 {
                (y[r] + 0.5).ln()
            } else {
                let mut acc = 0.0;
                for c in 0..ncols {
                    acc += x_colmajor[c * nrows + r] * beta[c];
                }
                acc
            };
            let mu = eta.exp().max(1e-12);
            let w = mu.sqrt();
            let yi = y[r];
            z[r] = (eta + (yi - mu) / mu) * w;
            for c in 0..ncols {
                x_w[c * nrows + r] = x_colmajor[c * nrows + r] * w;
            }
            if yi > 0.0 {
                deviance += 2.0 * (yi * (yi / mu).ln() - (yi - mu));
            } else {
                deviance += 2.0 * mu;
            }
        }
        let fit = backend.least_squares(&x_w, nrows, ncols, &z, workspace)?;
        for c in 0..ncols {
            max_delta = max_delta.max((fit.coefficients[c] - beta[c]).abs());
            beta[c] = fit.coefficients[c];
        }
        if max_delta < options.tol {
            converged = true;
            break;
        }
    }

    Ok(GlmFit { coefficients: beta, iterations, converged, separated: false, deviance })
}

fn fit_logistic(
    design: GlmDesignRef<'_>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<GlmFit, StatsError> {
    let GlmDesignRef { x_colmajor, nrows, ncols, y } = design;
    if y.len() != nrows {
        return Err(StatsError::Shape { message: "y length != nrows" });
    }
    if x_colmajor.len() < nrows.saturating_mul(ncols) {
        return Err(StatsError::Shape { message: "X buffer too short" });
    }
    for &yi in y {
        if !(yi == 0.0 || yi == 1.0) {
            return Err(StatsError::Shape { message: "binomial GLM requires 0/1 outcomes" });
        }
    }

    let mut beta = vec![0.0; ncols];
    let mut x_w = vec![0.0; nrows * ncols];
    let mut z = vec![0.0; nrows];
    let mut converged = false;
    let mut separated = false;
    let mut iterations = 0u32;
    let mut deviance = f64::INFINITY;

    for iter in 1..=options.max_iter {
        iterations = iter;
        let mut max_delta = 0.0_f64;
        deviance = 0.0;
        for r in 0..nrows {
            let mut eta = 0.0;
            for c in 0..ncols {
                eta += x_colmajor[c * nrows + r] * beta[c];
            }
            let mu = 1.0 / (1.0 + (-eta).exp());
            // Soft clamp masks the MLE under separation; flag when μ hits the band.
            if mu < 1e-8 || mu > 1.0 - 1e-8 {
                separated = true;
            }
            let mu_clamped = mu.clamp(1e-9, 1.0 - 1e-9);
            let w = (mu_clamped * (1.0 - mu_clamped)).sqrt();
            let yi = y[r];
            z[r] = eta + (yi - mu_clamped) / (mu_clamped * (1.0 - mu_clamped));
            z[r] *= w;
            for c in 0..ncols {
                x_w[c * nrows + r] = x_colmajor[c * nrows + r] * w;
            }
            if yi > 0.0 {
                deviance += -2.0 * mu_clamped.ln();
            } else {
                deviance += -2.0 * (1.0 - mu_clamped).ln();
            }
        }

        let fit = backend.least_squares(&x_w, nrows, ncols, &z, workspace)?;
        for c in 0..ncols {
            max_delta = max_delta.max((fit.coefficients[c] - beta[c]).abs());
            beta[c] = fit.coefficients[c];
        }
        if max_delta < options.tol {
            converged = true;
            break;
        }
    }

    Ok(GlmFit { coefficients: beta, iterations, converged, separated, deviance })
}

#[cfg(test)]
#[allow(clippy::float_cmp)]
mod tests {
    use super::*;
    use crate::faer_backend::FaerBackend;

    #[test]
    fn logistic_separates_simple_signal() {
        let n = 80usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = if i < n / 2 { 0.0 } else { 1.0 };
            x[i] = 1.0;
            x[n + i] = t;
            y[i] = if i % 10 == 0 { 1.0 - t } else { t };
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_glm(
            GlmFamily::BinomialLogit,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
            &FaerBackend,
            &mut ws,
            &GlmOptions::new(100, 1e-6),
        )
        .unwrap();
        assert!(fit.converged, "iters={} deviance={}", fit.iterations, fit.deviance);
        assert!(fit.coefficients[1] > 0.5);
        assert!(!fit.separated);
    }

    #[test]
    fn logistic_flags_complete_separation() {
        let n = 60usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = if i < n / 2 { 0.0 } else { 1.0 };
            x[i] = 1.0;
            x[n + i] = t;
            y[i] = t;
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_glm(
            GlmFamily::BinomialLogit,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
            &FaerBackend,
            &mut ws,
            &GlmOptions::new(100, 1e-6),
        )
        .unwrap();
        assert!(fit.separated);
        assert!(fit.require_ok().is_err());
    }

    #[test]
    fn gaussian_recovers_linear_slope() {
        let n = 100usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = (i % 2) as f64;
            x[i] = 1.0;
            x[n + i] = t;
            y[i] = 1.0 + 2.0 * t;
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_glm(
            GlmFamily::GaussianIdentity,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
            &FaerBackend,
            &mut ws,
            &GlmOptions::default(),
        )
        .unwrap();
        assert!((fit.coefficients[1] - 2.0).abs() < 1e-8);
    }

    #[test]
    fn poisson_recovers_positive_association() {
        let n = 120usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for i in 0..n {
            let t = if i < n / 2 { 0.0 } else { 1.0 };
            x[i] = 1.0;
            x[n + i] = t;
            y[i] = if t < 0.5 { 2.0 } else { 4.0 };
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_glm(
            GlmFamily::PoissonLog,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 2, y: &y },
            &FaerBackend,
            &mut ws,
            &GlmOptions::new(100, 1e-8),
        )
        .unwrap();
        assert!(fit.converged);
        assert!(fit.coefficients[1] > 0.3);
    }

    #[test]
    fn poisson_intercept_only_converges_to_log_mean() {
        let n = 40usize;
        let x = vec![1.0; n];
        let y = vec![100.0; n];
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_glm(
            GlmFamily::PoissonLog,
            GlmDesignRef { x_colmajor: &x, nrows: n, ncols: 1, y: &y },
            &FaerBackend,
            &mut ws,
            &GlmOptions::default(),
        )
        .unwrap();
        assert!(fit.converged, "iters={} deviance={}", fit.iterations, fit.deviance);
        assert!((fit.coefficients[0] - 100.0_f64.ln()).abs() < 1e-6, "b0={}", fit.coefficients[0]);
    }
}
