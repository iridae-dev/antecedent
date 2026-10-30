//! Small dense solves of the temporal designs: Cholesky factors of symmetric positive-definite
//! matrices of a few dozen coordinates at most, and the penalized logistic fit of the regime
//! survival models (the stats engine with a prior on every coefficient).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{GlmDesignRef, GlmOptions, LeastSquaresWorkspace, fit_logit_laplace};

use crate::error::EstimationError;

/// Lower Cholesky factor (row-major `n x n`) of a symmetric positive-definite matrix, or `None`
/// when it is not positive definite under the shared relative pivot rule.
pub(crate) fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    crate::linalg::SpdFactor::new(a, n).map(crate::linalg::SpdFactor::into_lower)
}

/// Solve `L L^T x = b` with the lower factor `l`.
pub(crate) fn cholesky_solve(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; n];
    let mut x = vec![0.0; n];
    antecedent_kernels::chol_solve_into(l, n, b, &mut y, &mut x);
    x
}

/// Inverse (row-major) of a symmetric positive-definite matrix from its lower factor.
pub(crate) fn cholesky_inverse(l: &[f64], n: usize) -> Vec<f64> {
    antecedent_kernels::chol_inverse(l, n)
}

/// `L z` for the lower factor `l` and a vector `z`.
pub(crate) fn lower_times(l: &[f64], n: usize, z: &[f64]) -> Vec<f64> {
    (0..n).map(|i| (0..=i).map(|k| l[i * n + k] * z[k]).sum()).collect()
}

/// A penalized logistic fit: the posterior mode under a `N(0, 1/precision)` coefficient prior
/// (the maximum-likelihood fit when the precision is zero), with the inverse of the penalized
/// observed information.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct LogisticFit {
    /// Coefficients.
    pub coefficients: Vec<f64>,
    /// Row-major inverse penalized information (the Laplace covariance).
    pub covariance: Vec<f64>,
    /// Lower Cholesky factor of `covariance`.
    pub covariance_factor: Vec<f64>,
}

/// Fit `P(y = 1 | x) = expit(x'beta)` on rows of `x` (row-major `n x p`) with nonnegative case
/// weights, through the stats GLM engine's Laplace logistic fit: the posterior mode under an
/// independent `N(0, 1/prior_precision)` prior on every coefficient and the inverse penalized
/// information beside it.
///
/// # Errors
/// Shape mismatch, an outcome other than 0 or 1, a singular penalized information (separation
/// without a prior), or no convergence in 100 Newton steps.
#[allow(clippy::float_cmp, reason = "binary outcomes are exact codes")]
pub(crate) fn fit_logistic(
    x: &[f64],
    p: usize,
    y: &[f64],
    weights: Option<&[f64]>,
    prior_precision: f64,
) -> Result<LogisticFit, EstimationError> {
    let n = y.len();
    if p == 0 || x.len() != n * p || weights.is_some_and(|w| w.len() != n) {
        return Err(EstimationError::data_msg("logistic design is misaligned"));
    }
    if y.iter().any(|value| *value != 0.0 && *value != 1.0) {
        return Err(EstimationError::data_msg("logistic outcomes must be 0 or 1"));
    }
    // A vanishing precision keeps a well-posed MLE step numerically symmetric positive
    // definite; it is far below any data information and is not a prior.
    let precision = if prior_precision.is_nan() { 1e-10 } else { prior_precision.max(1e-10) };
    let mut x_colmajor = vec![0.0; n * p];
    for (i, row) in x.chunks_exact(p).enumerate() {
        for (column, value) in row.iter().enumerate() {
            x_colmajor[column * n + i] = *value;
        }
    }
    let design = GlmDesignRef { x_colmajor: &x_colmajor, nrows: n, ncols: p, y };
    let mut workspace = LeastSquaresWorkspace::default();
    let laplace = fit_logit_laplace(
        design,
        weights,
        precision,
        &mut workspace,
        &GlmOptions::new(LOGISTIC_MAX_STEPS, LOGISTIC_STEP_TOLERANCE),
    )?;
    let coefficients = laplace.fit.coefficients;
    if !laplace.fit.converged {
        return Err(EstimationError::stats_msg(
            "the logistic fit did not converge in 100 Newton steps (separated covariates)",
        ));
    }
    if coefficients.iter().any(|value| !value.is_finite() || value.abs() > 1e6) {
        return Err(EstimationError::stats_msg("the logistic fit diverged (separated covariates)"));
    }
    let mut covariance = laplace.covariance;
    for a in 0..p {
        for b in (a + 1)..p {
            let mean = 0.5 * (covariance[a * p + b] + covariance[b * p + a]);
            covariance[a * p + b] = mean;
            covariance[b * p + a] = mean;
        }
    }
    let covariance_factor = cholesky(&covariance, p).ok_or_else(|| {
        EstimationError::stats_msg("the logistic covariance is not positive definite")
    })?;
    Ok(LogisticFit { coefficients, covariance, covariance_factor })
}

/// Newton steps of [`fit_logistic`] before it reports no convergence.
const LOGISTIC_MAX_STEPS: u32 = 100;
/// Largest coefficient change below which [`fit_logistic`] has converged.
const LOGISTIC_STEP_TOLERANCE: f64 = 1e-10;

/// A conjugate Gaussian linear regression: the posterior of `(beta, sigma^2)` under the prior
/// `beta | sigma^2 ~ N(0, sigma^2 / precision I)`, `sigma^2 ~ InvGamma(a0, b0)`.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct GaussianRegression {
    /// Posterior mean of the coefficients.
    pub mean: Vec<f64>,
    /// Lower Cholesky factor of the posterior precision-scaled covariance `V_n`.
    pub covariance_factor: Vec<f64>,
    /// Posterior inverse-gamma shape.
    pub shape: f64,
    /// Posterior inverse-gamma rate.
    pub rate: f64,
}

/// Conjugate fit of `y = X beta + e` (row-major `n x p`).
///
/// # Errors
/// Shape mismatch or a singular posterior precision.
pub(crate) fn fit_gaussian(
    x: &[f64],
    p: usize,
    y: &[f64],
    prior_precision: f64,
) -> Result<GaussianRegression, EstimationError> {
    let n = y.len();
    if p == 0 || x.len() != n * p || prior_precision.is_nan() || prior_precision <= 0.0 {
        return Err(EstimationError::data_msg("gaussian regression design is misaligned"));
    }
    let mut precision = vec![0.0; p * p];
    let mut xty = vec![0.0; p];
    for i in 0..n {
        let row = &x[i * p..(i + 1) * p];
        for a in 0..p {
            xty[a] += row[a] * y[i];
            for b in 0..p {
                precision[a * p + b] += row[a] * row[b];
            }
        }
    }
    for a in 0..p {
        precision[a * p + a] += prior_precision;
    }
    let factor = cholesky(&precision, p)
        .ok_or_else(|| EstimationError::stats_msg("the confounder regression is singular"))?;
    let mean = cholesky_solve(&factor, p, &xty);
    // The NIG rate includes the prior penalty as well as residual SSE. This
    // centered form avoids cancellation in y'y - mean'X'y at large offsets.
    let rss = antecedent_kernels::numeric::CompensatedSum::of((0..n).map(|i| {
        let fitted = x[i * p..(i + 1) * p].iter().zip(&mean).map(|(a, b)| a * b).sum::<f64>();
        (y[i] - fitted).powi(2)
    }));
    let penalty = antecedent_kernels::numeric::CompensatedSum::of(
        mean.iter().map(|value| prior_precision * value * value),
    );
    let (a0, b0) = (1.0, 1.0);
    let shape = a0 + 0.5 * n as f64;
    let rate = b0 + 0.5 * (rss + penalty);
    if !rate.is_finite() || rate <= 0.0 {
        return Err(EstimationError::stats_msg(
            "the confounder posterior variance scale is nonpositive or nonfinite",
        ));
    }
    let covariance = cholesky_inverse(&factor, p);
    let covariance_factor = cholesky(&covariance, p)
        .ok_or_else(|| EstimationError::stats_msg("the confounder covariance is singular"))?;
    Ok(GaussianRegression { mean, covariance_factor, shape, rate })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn gaussian_rate_keeps_residual_and_prior_penalty_at_large_offsets() {
        let fit = fit_gaussian(&[1.0; 3], 1, &[1e9 - 1.0, 1e9, 1e9 + 1.0], 1e-18).unwrap();
        // SSE=2, coefficient penalty=1, prior rate=1.
        assert!((fit.rate - 2.5).abs() < 1e-6, "{}", fit.rate);
    }

    #[test]
    fn cholesky_solves_a_small_system() {
        let a = [4.0, 2.0, 0.6, 2.0, 2.0, 0.5, 0.6, 0.5, 3.0];
        let l = cholesky(&a, 3).unwrap();
        let x = cholesky_solve(&l, 3, &[1.0, 2.0, 3.0]);
        for i in 0..3 {
            let row: f64 = (0..3).map(|j| a[i * 3 + j] * x[j]).sum();
            assert!((row - [1.0, 2.0, 3.0][i]).abs() < 1e-12);
        }
        let inverse = cholesky_inverse(&l, 3);
        let product: f64 = (0..3).map(|k| a[k] * inverse[k * 3]).sum();
        assert!((product - 1.0).abs() < 1e-12);
    }

    #[test]
    fn cholesky_acceptance_is_relative_to_the_matrix_scale() {
        // The same well-conditioned matrix in rescaled units is still positive definite.
        let tiny = [4.0e-15, 2.0e-15, 2.0e-15, 5.0e-15];
        assert!(cholesky(&tiny, 2).is_some());
        // A perfectly collinear matrix is singular at any scale.
        assert!(cholesky(&[1.0e-15, 1.0e-15, 1.0e-15, 1.0e-15], 2).is_none());
        assert!(cholesky(&[1.0e9, 1.0e9, 1.0e9, 1.0e9], 2).is_none());
    }

    #[test]
    fn the_logistic_fit_recovers_its_coefficients() {
        let n = 4000;
        let mut x = Vec::with_capacity(2 * n);
        let mut y = Vec::with_capacity(n);
        for i in 0..n {
            let z = (i % 200) as f64 / 100.0 - 1.0;
            let p = antecedent_kernels::sigmoid(-0.5 + 1.2 * z);
            // A deterministic thinning keeps the empirical rate near p in every z cell.
            let u = ((i / 200) as f64 + 0.5) / 20.0;
            x.extend_from_slice(&[1.0, z]);
            y.push(if u < p { 1.0 } else { 0.0 });
        }
        let fit = fit_logistic(&x, 2, &y, None, 0.0).unwrap();
        assert!((fit.coefficients[0] + 0.5).abs() < 0.1, "{:?}", fit.coefficients);
        assert!((fit.coefficients[1] - 1.2).abs() < 0.15, "{:?}", fit.coefficients);
        assert!(fit.covariance[0] > 0.0);
    }
}
