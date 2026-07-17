//! GLM fitting: logistic, multinomial logit, Gaussian/identity, and Poisson/log IRLS.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::float_cmp
)]

use causal_kernels::norm_cdf;

use crate::error::StatsError;
use crate::gram::invert_square;
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

/// Multinomial logit design: column-major `X` and integer category codes.
#[derive(Clone, Copy, Debug)]
pub struct MultinomialDesignRef<'a> {
    /// Column-major design matrix (typically intercept + covariates).
    pub x_colmajor: &'a [f64],
    /// Rows.
    pub nrows: usize,
    /// Columns.
    pub ncols: usize,
    /// Category index in `0..n_categories` per row.
    pub y_category: &'a [u32],
    /// Number of categories `K ≥ 1`.
    pub n_categories: usize,
}

/// Multinomial logit fit (softmax / baseline-category logits).
///
/// Coefficients are stored row-major as `[K * ncols]` with **reference category 0**
/// pinned to zero (identifying constraint). Categories `1..K-1` hold free MLE logits.
#[derive(Clone, Debug)]
pub struct MultinomialFit {
    /// Row-major `[n_categories * ncols]`; category 0 is all zeros.
    pub coefficients: Vec<f64>,
    /// Fisher-scoring iterations used.
    pub iterations: u32,
    /// Whether the score updates converged.
    pub converged: bool,
    /// Whether fitted probabilities hit the soft clamp band (separation signal).
    pub separated: bool,
    /// Final deviance (`−2` log-likelihood).
    pub deviance: f64,
    /// `K`.
    pub n_categories: usize,
    /// Design width.
    pub ncols: usize,
}

impl MultinomialFit {
    /// Error if scoring failed to converge or separation was detected.
    ///
    /// # Errors
    ///
    /// Non-converged or separated fits.
    pub fn require_ok(&self) -> Result<(), StatsError> {
        if !self.converged {
            return Err(StatsError::Backend(
                "multinomial logit IRLS did not converge".into(),
            ));
        }
        if self.separated {
            return Err(StatsError::Backend(
                "multinomial logit indicates (quasi-)complete separation".into(),
            ));
        }
        Ok(())
    }
}

/// Fit a baseline-category multinomial logit via Fisher scoring.
///
/// For `K = 2` this delegates to binomial logit IRLS (same MLE). For `K > 2` it
/// runs Fisher scoring on the `(K−1)·ncols` free parameters with expected Hessian
/// `Xᵀ W X` from the multinomial covariance.
///
/// # Errors
///
/// Shape mismatch, invalid category codes, singular Fisher information, or
/// linear-algebra failure (binary path).
pub fn fit_multinomial_logit(
    design: MultinomialDesignRef<'_>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<MultinomialFit, StatsError> {
    let MultinomialDesignRef { x_colmajor, nrows, ncols, y_category, n_categories: k } = design;
    if k == 0 {
        return Err(StatsError::Shape { message: "multinomial requires K ≥ 1" });
    }
    if y_category.len() != nrows {
        return Err(StatsError::Shape { message: "y_category length != nrows" });
    }
    if x_colmajor.len() < nrows.saturating_mul(ncols) {
        return Err(StatsError::Shape { message: "X buffer too short" });
    }
    for &yi in y_category {
        if (yi as usize) >= k {
            return Err(StatsError::Shape { message: "y_category out of range" });
        }
    }

    if k == 1 {
        return Ok(MultinomialFit {
            coefficients: vec![0.0; ncols],
            iterations: 0,
            converged: true,
            separated: false,
            deviance: 0.0,
            n_categories: 1,
            ncols,
        });
    }

    if k == 2 {
        return fit_multinomial_binary(design, backend, workspace, options);
    }

    fit_multinomial_fisher(design, options)
}

fn fit_multinomial_binary(
    design: MultinomialDesignRef<'_>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<MultinomialFit, StatsError> {
    let MultinomialDesignRef { x_colmajor, nrows, ncols, y_category, .. } = design;
    let y: Vec<f64> = y_category.iter().map(|&c| f64::from(c)).collect();
    let fit = fit_logistic(
        GlmDesignRef { x_colmajor, nrows, ncols, y: &y },
        backend,
        workspace,
        options,
    )?;
    let mut coefficients = vec![0.0; 2 * ncols];
    coefficients[ncols..].copy_from_slice(&fit.coefficients[..ncols]);
    Ok(MultinomialFit {
        coefficients,
        iterations: fit.iterations,
        converged: fit.converged,
        separated: fit.separated,
        deviance: fit.deviance,
        n_categories: 2,
        ncols,
    })
}

fn fit_multinomial_fisher(
    design: MultinomialDesignRef<'_>,
    options: &GlmOptions,
) -> Result<MultinomialFit, StatsError> {
    let MultinomialDesignRef { x_colmajor, nrows, ncols, y_category, n_categories: k } = design;
    let n_free = k - 1;
    let m = n_free * ncols;
    let mut beta_free = vec![0.0; m];
    let mut h = vec![0.0; m * m];
    let mut score = vec![0.0; m];
    let mut pi = vec![0.0; k];
    let mut eta = vec![0.0; k];
    let mut converged = false;
    let mut separated = false;
    let mut iterations = 0u32;
    let mut deviance = f64::INFINITY;
    let mut prev_deviance = f64::INFINITY;

    for iter in 1..=options.max_iter {
        iterations = iter;
        h.fill(0.0);
        score.fill(0.0);
        deviance = 0.0;

        for r in 0..nrows {
            // η_0 = 0; η_j = x·β_j for j = 1..K-1
            eta[0] = 0.0;
            let mut max_eta = 0.0_f64;
            for j in 1..k {
                let mut acc = 0.0;
                let base = (j - 1) * ncols;
                for c in 0..ncols {
                    acc += x_colmajor[c * nrows + r] * beta_free[base + c];
                }
                eta[j] = acc;
                if acc > max_eta {
                    max_eta = acc;
                }
            }
            let mut zsum = 0.0;
            for j in 0..k {
                let e = (eta[j] - max_eta).exp();
                pi[j] = e;
                zsum += e;
            }
            let inv_z = 1.0 / zsum.max(f64::EPSILON);
            for j in 0..k {
                pi[j] *= inv_z;
                if pi[j] < 1e-8 || pi[j] > 1.0 - 1e-8 {
                    separated = true;
                }
            }
            let yi = y_category[r] as usize;
            deviance += -2.0 * pi[yi].max(1e-300).ln();

            // Score and Fisher information over free categories 1..K-1.
            for j in 1..k {
                let pj = pi[j];
                let yj = if yi == j { 1.0 } else { 0.0 };
                let resid = yj - pj;
                let jb = (j - 1) * ncols;
                for c in 0..ncols {
                    let xc = x_colmajor[c * nrows + r];
                    score[jb + c] += resid * xc;
                }
                for jp in 1..k {
                    let pjp = pi[jp];
                    let w = if j == jp { pj * (1.0 - pj) } else { -pj * pjp };
                    let jpb = (jp - 1) * ncols;
                    for c1 in 0..ncols {
                        let xc1 = x_colmajor[c1 * nrows + r];
                        for c2 in 0..ncols {
                            let xc2 = x_colmajor[c2 * nrows + r];
                            h[(jb + c1) * m + (jpb + c2)] += w * xc1 * xc2;
                        }
                    }
                }
            }
        }

        // Ridge for near-singular Fisher information under quasi-separation.
        for i in 0..m {
            h[i * m + i] += 1e-8;
        }
        let h_inv = invert_square(&h, m).ok_or(StatsError::RankDeficient {
            rank: 0,
            ncols: m,
        })?;
        let mut delta = vec![0.0; m];
        let mut max_delta = 0.0_f64;
        let mut score_norm = 0.0_f64;
        for i in 0..m {
            score_norm = score_norm.max(score[i].abs());
            let mut d = 0.0;
            for j in 0..m {
                d += h_inv[i * m + j] * score[j];
            }
            delta[i] = d;
            max_delta = max_delta.max(d.abs());
        }
        // Damped Newton step when the update is large (common under quasi-separation).
        let scale = if max_delta > 5.0 { 5.0 / max_delta } else { 1.0 };
        for i in 0..m {
            beta_free[i] += scale * delta[i];
        }
        let dev_delta = (prev_deviance - deviance).abs();
        prev_deviance = deviance;
        if max_delta * scale < options.tol
            || score_norm < options.tol
            || (iter > 2 && dev_delta < options.tol * (1.0 + deviance.abs()))
        {
            converged = true;
            break;
        }
    }

    let mut coefficients = vec![0.0; k * ncols];
    for j in 1..k {
        let src = (j - 1) * ncols;
        let dst = j * ncols;
        coefficients[dst..dst + ncols].copy_from_slice(&beta_free[src..src + ncols]);
    }

    Ok(MultinomialFit {
        coefficients,
        iterations,
        converged,
        separated,
        deviance,
        n_categories: k,
        ncols,
    })
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

    #[test]
    fn multinomial_binary_matches_logistic_slope_sign() {
        let n = 80usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0u32; n];
        for i in 0..n {
            let t = if i < n / 2 { 0.0 } else { 1.0 };
            x[i] = 1.0;
            x[n + i] = t;
            y[i] = if i % 10 == 0 { 1 - (t as u32) } else { t as u32 };
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_multinomial_logit(
            MultinomialDesignRef {
                x_colmajor: &x,
                nrows: n,
                ncols: 2,
                y_category: &y,
                n_categories: 2,
            },
            &FaerBackend,
            &mut ws,
            &GlmOptions::new(100, 1e-6),
        )
        .unwrap();
        assert!(fit.converged);
        assert!(!fit.separated);
        assert!(fit.coefficients[..2].iter().all(|&c| c == 0.0));
        assert!(fit.coefficients[3] > 0.5, "slope={}", fit.coefficients[3]);
    }


    #[test]
    fn multinomial_intercept_only_three_class() {
        let n = 90usize;
        let x = vec![1.0; n];
        let mut y = vec![0u32; n];
        for i in 0..n {
            y[i] = (i % 3) as u32;
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_multinomial_logit(
            MultinomialDesignRef {
                x_colmajor: &x,
                nrows: n,
                ncols: 1,
                y_category: &y,
                n_categories: 3,
            },
            &FaerBackend,
            &mut ws,
            &GlmOptions::new(100, 1e-8),
        )
        .unwrap();
        assert!(fit.converged);
        // Equal class sizes → free intercepts ≈ 0.
        assert!(fit.coefficients[1].abs() < 0.2, "{:?}", fit.coefficients);
        assert!(fit.coefficients[2].abs() < 0.2, "{:?}", fit.coefficients);
    }

    #[test]
    fn multinomial_three_class_recovers_parent_signal() {
        // Y ∈ {0,1,2}; when X=0 prefer class 0, when X=1 prefer class 2.
        // Enough class mixing to avoid complete separation.
        let n = 180usize;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0u32; n];
        for i in 0..n {
            let t = if i < n / 2 { 0.0 } else { 1.0 };
            x[i] = 1.0;
            x[n + i] = t;
            y[i] = match (t < 0.5, i % 5) {
                (true, 0) => 1,
                (true, 1) => 2,
                (true, _) => 0,
                (false, 0) => 0,
                (false, 1) => 1,
                (false, _) => 2,
            };
        }
        let mut ws = LeastSquaresWorkspace::default();
        let fit = fit_multinomial_logit(
            MultinomialDesignRef {
                x_colmajor: &x,
                nrows: n,
                ncols: 2,
                y_category: &y,
                n_categories: 3,
            },
            &FaerBackend,
            &mut ws,
            &GlmOptions::new(100, 1e-8),
        )
        .unwrap();
        assert!(fit.converged, "iters={} deviance={}", fit.iterations, fit.deviance);
        assert!(fit.coefficients[..2].iter().all(|&c| c == 0.0));
        // Class 2 vs 0: positive slope on X.
        let slope_class2 = fit.coefficients[2 * 2 + 1];
        assert!(slope_class2 > 0.5, "class2 slope={slope_class2}");
        // Softmax at X=1 should put most mass on class 2.
        let eta1 = fit.coefficients[2] + fit.coefficients[3];
        let eta2 = fit.coefficients[4] + fit.coefficients[5];
        let m = eta1.max(eta2).max(0.0);
        let p0 = (-m).exp();
        let p1 = (eta1 - m).exp();
        let p2 = (eta2 - m).exp();
        let z = p0 + p1 + p2;
        assert!(p2 / z > p0 / z && p2 / z > p1 / z, "p={:?}", [p0 / z, p1 / z, p2 / z]);
    }
}
