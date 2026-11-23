//! Posterior summaries and multiple-imputation combination.

use super::{ErrorVariance, LinearErrorData, LinearMoments, MeasurementErrorError, finite_all};

/// Posterior means of a latent regressor from joint posterior draws.
///
/// `draws` is draw-major: draw `s` occupies `draws[s * n..(s + 1) * n]`.
///
/// # Errors
/// A shape mismatch, no draws or non-finite values refuse.
pub fn posterior_means(
    draws: &[f64],
    n: usize,
    draw_count: usize,
) -> Result<Vec<f64>, MeasurementErrorError> {
    if n == 0 || draw_count == 0 || draws.len() != n.saturating_mul(draw_count) {
        return Err(MeasurementErrorError::Shape("draws do not form draw_count x n"));
    }
    if !finite_all(draws) {
        return Err(MeasurementErrorError::NonFinite("posterior draws"));
    }
    let mut means = vec![0.0; n];
    for s in 0..draw_count {
        for i in 0..n {
            means[i] += draws[s * n + i];
        }
    }
    let inv = 1.0 / draw_count as f64;
    for v in &mut means {
        *v *= inv;
    }
    Ok(means)
}

/// Result of combining a linear outcome fit over posterior imputations.
#[derive(Clone, Debug, PartialEq)]
pub struct CombinedImputations {
    /// Number of draws combined.
    pub draws: usize,
    /// Design width.
    pub dim: usize,
    /// Mean coefficients.
    pub coefficients: Vec<f64>,
    /// Total covariance `W + (1 + 1/S) B`.
    pub covariance: Vec<f64>,
    /// Mean within-imputation covariance `W`.
    pub within: Vec<f64>,
    /// Between-imputation covariance `B`.
    pub between: Vec<f64>,
    /// Coefficients of each imputation's fit, `draws` vectors of length `dim`.
    pub draw_coefficients: Vec<Vec<f64>>,
    /// HC0 covariance of each imputation's fit, `draws` row-major `dim x dim` blocks.
    pub draw_covariances: Vec<Vec<f64>>,
}

/// Rubin's combination of per-imputation coefficients and covariances.
///
/// # Errors
/// Fewer than two imputations or inconsistent shapes refuse.
pub fn rubin_combine(
    coefficients: &[Vec<f64>],
    covariances: &[Vec<f64>],
) -> Result<CombinedImputations, MeasurementErrorError> {
    let draws = coefficients.len();
    if draws < 2 {
        return Err(MeasurementErrorError::InsufficientReplicates(
            "Rubin's rules need at least two imputations",
        ));
    }
    let d = coefficients[0].len();
    if d == 0
        || covariances.len() != draws
        || coefficients.iter().any(|c| c.len() != d || !finite_all(c))
        || covariances.iter().any(|c| c.len() != d * d || !finite_all(c))
    {
        return Err(MeasurementErrorError::Shape("imputation fits differ in shape"));
    }
    let s_f = draws as f64;
    let mut within = vec![0.0; d * d];
    for cov in covariances {
        for (acc, v) in within.iter_mut().zip(cov) {
            *acc += v / s_f;
        }
    }
    let mut mean = vec![0.0; d];
    for beta in coefficients {
        for (acc, v) in mean.iter_mut().zip(beta) {
            *acc += v / s_f;
        }
    }
    let mut between = vec![0.0; d * d];
    for beta in coefficients {
        for a in 0..d {
            for b in 0..d {
                between[a * d + b] += (beta[a] - mean[a]) * (beta[b] - mean[b]) / (s_f - 1.0);
            }
        }
    }
    let covariance = within.iter().zip(&between).map(|(w, b)| w + (1.0 + 1.0 / s_f) * b).collect();
    Ok(CombinedImputations {
        draws,
        dim: d,
        coefficients: mean,
        covariance,
        within,
        between,
        draw_coefficients: coefficients.to_vec(),
        draw_covariances: covariances.to_vec(),
    })
}

/// Combine linear fits over joint posterior draws with Rubin's rules.
///
/// Valid only when the draws are proper imputations conditional on the
/// outcome (a posterior that already conditions on `y`); a posterior of the
/// latent value given the error-prone data alone makes the draws
/// Berkson-type noise and the regression of `y` on them is attenuated, so that
/// case uses the posterior mean instead ([`posterior_means`]).
///
/// # Errors
/// Fewer than two draws, a shape mismatch or a singular fit refuse.
pub fn combine_imputations_linear(
    y: &[f64],
    draws: &[f64],
    draw_count: usize,
    covariates: &[f64],
    covariate_count: usize,
) -> Result<CombinedImputations, MeasurementErrorError> {
    let n = y.len();
    if draw_count < 2 {
        return Err(MeasurementErrorError::InsufficientReplicates(
            "Rubin's rules need at least two imputations",
        ));
    }
    if draws.len() != n.saturating_mul(draw_count) {
        return Err(MeasurementErrorError::Shape("draws do not form draw_count x n"));
    }
    let exact = ErrorVariance::known(0.0, 0.0)?;
    // One coefficient vector per imputation of draws the caller already holds (`draws.len() == n * draw_count`).
    let mut betas = Vec::with_capacity(draw_count);
    let mut covs = Vec::with_capacity(draw_count);
    for s in 0..draw_count {
        let data = LinearErrorData {
            y,
            w: &draws[s * n..(s + 1) * n],
            covariates,
            covariate_count,
            replicate_counts: None,
        };
        let fit = LinearMoments::accumulate(&data, &exact, 0.0)?.solve()?;
        betas.push(fit.naive_coefficients);
        covs.push(fit.naive_covariance);
    }
    rubin_combine(&betas, &covs)
}
