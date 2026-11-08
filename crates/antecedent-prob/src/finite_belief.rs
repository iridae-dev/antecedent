//! Finite Bayesian conditioning with explicit aligned state/mapping slots.
//! These numerical operations confer no scientific, causal or calibration authority.
// SPDX-License-Identifier: MIT OR Apache-2.0
use crate::ProbError;

/// Normalize a finite belief after a declared log likelihood, preserving every state slot.
/// # Errors
/// Empty/ragged inputs, nonfinite/negative mass, invalid likelihood or zero posterior mass.
pub fn condition(prior: &[f64], log_likelihood: &[f64]) -> Result<Vec<f64>, ProbError> {
    if prior.is_empty()
        || prior.len() != log_likelihood.len()
        || prior.iter().any(|weight| !weight.is_finite() || *weight < 0.0)
        || log_likelihood.iter().any(|value| value.is_nan() || *value == f64::INFINITY)
    {
        return Err(invalid("invalid aligned finite belief/update"));
    }
    let log_weights = prior
        .iter()
        .zip(log_likelihood)
        .map(|(weight, likelihood)| weight.ln() + likelihood)
        .collect::<Vec<_>>();
    let maximum = log_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    if !maximum.is_finite() {
        return Err(invalid("experiment has zero mass under every retained state"));
    }
    let mut weights = log_weights.iter().map(|value| (value - maximum).exp()).collect::<Vec<_>>();
    let mass = weights.iter().sum::<f64>();
    for weight in &mut weights {
        *weight /= mass;
    }
    Ok(weights)
}

/// Explicit sample-mean likelihood under a paired affine experimental estimand map.
/// The target functional equals `intercept + slope * experimental_mean` in each slot.
#[derive(Clone, Copy, Debug)]
pub struct AffineGaussianExperiment<'a> {
    /// Actual observed experimental sample mean.
    pub observed_mean: f64,
    /// Known variance of one independent observation.
    pub observation_variance: f64,
    /// Declared independent experimental observation count.
    pub sample_size: u64,
    /// Complete aligned target-functional draws.
    pub target_values: &'a [f64],
    /// Paired mapping intercept for every original source slot.
    pub intercepts: &'a [f64],
    /// Paired nonzero mapping slope for every original source slot.
    pub slopes: &'a [f64],
}
impl AffineGaussianExperiment<'_> {
    /// Condition a prior using this exact likelihood without resampling or dropping slots.
    /// # Errors
    /// Invalid likelihood, missing/failed mapping slots or a zero-mass update.
    pub fn condition(&self, prior: &[f64]) -> Result<Vec<f64>, ProbError> {
        let count = self.target_values.len();
        if count == 0
            || self.intercepts.len() != count
            || self.slopes.len() != count
            || prior.len() != count
            || self.sample_size == 0
            || !self.observed_mean.is_finite()
            || !self.observation_variance.is_finite()
            || self.observation_variance <= 0.0
            || self.target_values.iter().chain(self.intercepts).any(|value| !value.is_finite())
            || self.slopes.iter().any(|value| !value.is_finite() || *value == 0.0)
        {
            return Err(invalid("invalid paired affine Gaussian experiment"));
        }
        let variance = self.observation_variance / self.sample_size as f64;
        if !variance.is_finite() || variance <= 0.0 {
            return Err(invalid("experimental sample-mean variance underflow/overflow"));
        }
        let likelihood = self
            .target_values
            .iter()
            .enumerate()
            .map(|(slot, target)| {
                let predicted = (target - self.intercepts[slot]) / self.slopes[slot];
                if !predicted.is_finite() {
                    return Err(invalid("affine experiment mapping overflow in a retained slot"));
                }
                let deviation = self.observed_mean - predicted;
                let sd = variance.sqrt();
                let standardized = if deviation.is_finite() {
                    deviation / sd
                } else {
                    self.observed_mean / sd - predicted / sd
                };
                Ok(-0.5 * (standardized.powi(2) + std::f64::consts::TAU.ln() + variance.ln()))
            })
            .collect::<Result<Vec<_>, ProbError>>()?;
        condition(prior, &likelihood)
    }
}

/// Explicit joint Gaussian likelihood of several experimental means under paired affine maps.
///
/// Component `k` of the target functional equals `intercepts[k] + slopes[k] * mean[k]` in each
/// retained slot. The observed means share one known sampling covariance, which is the
/// covariance of the mean estimators themselves (not of one observation). Off-diagonal entries
/// are never inferred: a caller who has none must not use this likelihood.
#[derive(Clone, Copy, Debug)]
pub struct JointAffineGaussianExperiment<'a> {
    /// Observed experimental means, one per component.
    pub observed_means: &'a [f64],
    /// Known `K x K` row-major covariance of the mean estimators.
    pub sampling_covariance: &'a [f64],
    /// Complete aligned target-functional draws, one vector per component.
    pub target_values: &'a [Vec<f64>],
    /// Paired mapping intercepts, one vector per component, one entry per slot.
    pub intercepts: &'a [Vec<f64>],
    /// Paired nonzero mapping slopes, one vector per component, one entry per slot.
    pub slopes: &'a [Vec<f64>],
}
impl JointAffineGaussianExperiment<'_> {
    /// Condition a prior on the joint likelihood without resampling or dropping slots.
    /// # Errors
    /// Misaligned or nonfinite inputs, an asymmetric or non-positive-definite covariance,
    /// or a zero-mass update.
    pub fn condition(&self, prior: &[f64]) -> Result<Vec<f64>, ProbError> {
        let k = self.observed_means.len();
        let count = prior.len();
        let aligned = |rows: &[Vec<f64>]| rows.len() == k && rows.iter().all(|r| r.len() == count);
        if k == 0
            || count == 0
            || self.sampling_covariance.len() != k * k
            || !aligned(self.target_values)
            || !aligned(self.intercepts)
            || !aligned(self.slopes)
            || self.observed_means.iter().any(|v| !v.is_finite())
            || self.sampling_covariance.iter().any(|v| !v.is_finite())
            || self.target_values.iter().chain(self.intercepts).flatten().any(|v| !v.is_finite())
            || self.slopes.iter().flatten().any(|v| !v.is_finite() || *v == 0.0)
        {
            return Err(invalid("invalid paired joint affine Gaussian experiment"));
        }
        for i in 0..k {
            for j in 0..i {
                let a = self.sampling_covariance[i * k + j];
                let b = self.sampling_covariance[j * k + i];
                if (a - b).abs() > 1e-9 * (a.abs() + b.abs()).max(f64::MIN_POSITIVE) {
                    return Err(invalid("joint sampling covariance is not symmetric"));
                }
            }
        }
        let mut factor = vec![0.0; k * k];
        antecedent_kernels::cholesky_spd_into(self.sampling_covariance, k, &mut factor)
            .map_err(|_| invalid("joint sampling covariance is not positive definite"))?;
        let log_det = 2.0 * (0..k).map(|i| factor[i * k + i].ln()).sum::<f64>();
        let constant = k as f64 * std::f64::consts::TAU.ln() + log_det;
        let mut residual = vec![0.0; k];
        let mut whitened = vec![0.0; k];
        let likelihood = (0..count)
            .map(|slot| {
                for (c, entry) in residual.iter_mut().enumerate() {
                    let predicted = (self.target_values[c][slot] - self.intercepts[c][slot])
                        / self.slopes[c][slot];
                    *entry = self.observed_means[c] - predicted;
                    if !entry.is_finite() {
                        return Err(invalid("joint affine experiment mapping overflow in a slot"));
                    }
                }
                antecedent_kernels::forward_solve_into(&factor, k, &residual, &mut whitened);
                let quadratic = whitened.iter().map(|v| v * v).sum::<f64>();
                Ok(-0.5 * (quadratic + constant))
            })
            .collect::<Result<Vec<_>, ProbError>>()?;
        condition(prior, &likelihood)
    }
}

/// Project a finite aligned row-major law onto one declared linear functional.
/// # Errors
/// Ragged/empty axes, nonfinite coefficients/draws, or numerical overflow.
pub fn linear_functional_rows(rows: &[f64], coefficients: &[f64]) -> Result<Vec<f64>, ProbError> {
    let width = coefficients.len();
    if width == 0
        || rows.is_empty()
        || !rows.len().is_multiple_of(width)
        || rows.iter().chain(coefficients).any(|value| !value.is_finite())
    {
        return Err(invalid("invalid finite-law functional axes"));
    }
    let result = rows
        .chunks_exact(width)
        .map(|row| row.iter().zip(coefficients).map(|(value, weight)| value * weight).sum::<f64>())
        .collect::<Vec<_>>();
    if result.iter().any(|value| !value.is_finite()) {
        return Err(invalid("finite-law functional overflow"));
    }
    Ok(result)
}
fn invalid(message: &str) -> ProbError {
    ProbError::Numerical { message: message.into() }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn affine_experiment_refuses_overflow_in_any_original_mapping_slot() {
        let experiment = AffineGaussianExperiment {
            observed_mean: 0.0,
            observation_variance: 1.0,
            sample_size: 1,
            target_values: &[0.0, f64::MAX],
            intercepts: &[0.0, -f64::MAX],
            slopes: &[1.0, 1.0],
        };
        assert!(experiment.condition(&[0.5, 0.5]).is_err());
    }
    #[test]
    fn affine_experiment_large_finite_variance_keeps_a_valid_likelihood() {
        let experiment = AffineGaussianExperiment {
            observed_mean: 0.0,
            observation_variance: f64::MAX,
            sample_size: 1,
            target_values: &[0.0, 1.0],
            intercepts: &[0.0, 0.0],
            slopes: &[1.0, 1.0],
        };
        assert_eq!(experiment.condition(&[0.5, 0.5]).unwrap(), vec![0.5, 0.5]);
    }
}
