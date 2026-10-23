//! Small dense solves of the temporal designs: Cholesky factors of symmetric positive-definite
//! matrices of a few dozen coordinates at most, and the penalized logistic Newton fit of the
//! regime survival models.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::EstimationError;

/// Lower Cholesky factor (row-major `n x n`) of a symmetric positive-definite matrix, or `None`
/// when a pivot is not positive.
pub(crate) fn cholesky(a: &[f64], n: usize) -> Option<Vec<f64>> {
    if a.len() != n * n {
        return None;
    }
    let scale = (0..n).map(|i| a[i * n + i].abs()).fold(0.0_f64, f64::max).max(1.0);
    let mut l = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut sum = a[i * n + j];
            for k in 0..j {
                sum -= l[i * n + k] * l[j * n + k];
            }
            if i == j {
                if !(sum.is_finite() && sum > 1e-13 * scale) {
                    return None;
                }
                l[i * n + i] = sum.sqrt();
            } else {
                l[i * n + j] = sum / l[j * n + j];
            }
        }
    }
    Some(l)
}

/// Solve `L L^T x = b` with the lower factor `l`.
pub(crate) fn cholesky_solve(l: &[f64], n: usize, b: &[f64]) -> Vec<f64> {
    let mut y = vec![0.0; n];
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[i * n + k] * y[k];
        }
        y[i] = sum / l[i * n + i];
    }
    let mut x = vec![0.0; n];
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[k * n + i] * x[k];
        }
        x[i] = sum / l[i * n + i];
    }
    x
}

/// Inverse (row-major) of a symmetric positive-definite matrix from its lower factor.
pub(crate) fn cholesky_inverse(l: &[f64], n: usize) -> Vec<f64> {
    let mut inverse = vec![0.0; n * n];
    let mut unit = vec![0.0; n];
    for column in 0..n {
        unit.iter_mut().for_each(|value| *value = 0.0);
        unit[column] = 1.0;
        let solved = cholesky_solve(l, n, &unit);
        for row in 0..n {
            inverse[row * n + column] = solved[row];
        }
    }
    inverse
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

/// Fit `P(y = 1 | x) = expit(x'beta)` by Newton steps on rows of `x` (row-major `n x p`) with
/// nonnegative case weights.
///
/// # Errors
/// Shape mismatch, an outcome other than 0 or 1, a singular penalized information (separation
/// without a prior), or no convergence in 100 steps.
#[allow(clippy::float_cmp, reason = "binary outcomes and zero case weights are exact codes")]
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
    // A vanishing ridge keeps a well-posed MLE step numerically symmetric positive definite;
    // it is far below any data information and is not a prior.
    let ridge = prior_precision.max(1e-10);
    let mut beta = vec![0.0; p];
    for _ in 0..100 {
        let mut information = vec![0.0; p * p];
        let mut gradient = vec![0.0; p];
        for i in 0..n {
            let w = weights.map_or(1.0, |weights| weights[i]);
            if w == 0.0 {
                continue;
            }
            let row = &x[i * p..(i + 1) * p];
            let eta: f64 = row.iter().zip(&beta).map(|(a, b)| a * b).sum();
            let mu = antecedent_kernels::sigmoid(eta).clamp(1e-12, 1.0 - 1e-12);
            let v = w * mu * (1.0 - mu);
            for a in 0..p {
                gradient[a] += w * (y[i] - mu) * row[a];
                for b in 0..p {
                    information[a * p + b] += v * row[a] * row[b];
                }
            }
        }
        for a in 0..p {
            gradient[a] -= ridge * beta[a];
            information[a * p + a] += ridge;
        }
        let Some(factor) = cholesky(&information, p) else {
            return Err(EstimationError::stats_msg(
                "the logistic information is singular (separated or collinear covariates)",
            ));
        };
        let step = cholesky_solve(&factor, p, &gradient);
        let largest = step.iter().fold(0.0_f64, |acc, value| acc.max(value.abs()));
        for (coefficient, delta) in beta.iter_mut().zip(&step) {
            *coefficient += delta;
        }
        if !largest.is_finite() || beta.iter().any(|value| value.abs() > 1e6) {
            return Err(EstimationError::stats_msg(
                "the logistic fit diverged (separated covariates)",
            ));
        }
        if largest < 1e-10 {
            let covariance = cholesky_inverse(&factor, p);
            let covariance_factor = cholesky(&covariance, p).ok_or_else(|| {
                EstimationError::stats_msg("the logistic covariance is not positive definite")
            })?;
            return Ok(LogisticFit { coefficients: beta, covariance, covariance_factor });
        }
    }
    Err(EstimationError::stats_msg("the logistic fit did not converge in 100 Newton steps"))
}

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
    let fitted_quadratic: f64 = mean.iter().zip(&xty).map(|(m, v)| m * v).sum();
    let yty: f64 = y.iter().map(|value| value * value).sum();
    let (a0, b0) = (1.0, 1.0);
    let shape = a0 + 0.5 * n as f64;
    let rate = b0 + 0.5 * (yty - fitted_quadratic).max(0.0);
    let covariance = cholesky_inverse(&factor, p);
    let covariance_factor = cholesky(&covariance, p)
        .ok_or_else(|| EstimationError::stats_msg("the confounder covariance is singular"))?;
    Ok(GaussianRegression { mean, covariance_factor, shape, rate })
}

#[cfg(test)]
mod tests {
    use super::*;

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
