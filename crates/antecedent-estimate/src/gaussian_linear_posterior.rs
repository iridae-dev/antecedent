//! Conjugate Normal-Inverse-Gamma posterior of a Gaussian linear model from sufficient statistics.
//!
//! # Model
//!
//! `y | beta, sigma2 ~ N(X beta, sigma2 I)` (or `N(X beta, sigma2 Omega)` with a known `Omega`,
//! once the caller has reduced the data to `X' Omega^-1 X`, `X' Omega^-1 y`, `y' Omega^-1 y`),
//! with the prior `beta | sigma2 ~ N(0, sigma2 Lambda0^-1)`, `Lambda0` diagonal and positive, and
//! `sigma2 ~ InvGamma(a0, b0)`. The posterior is again Normal-Inverse-Gamma:
//!
//! ```text
//! Lambda_n = Lambda0 + X' X          m_n = Lambda_n^-1 X' y
//! a_n = a0 + n / 2                   b_n = b0 + (y' y - X' y . m_n) / 2
//! ```
//!
//! and every linear functional `w' beta` has a Student-t marginal posterior with `2 a_n` degrees
//! of freedom, location `w' m_n` and squared scale `(b_n / a_n) w' Lambda_n^-1 w`. These are
//! exact posterior statements under the stated likelihood and prior; none of them is a
//! frequentist interval, and the sandwich covariances of the frequentist routes have no
//! counterpart here (the likelihood is the model-based one).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::reason_code;
use antecedent_stats::{invert_square, student_t_ppf, student_t_sf};

use crate::error::EstimationError;
use crate::util::refuse;

/// Diagonal Gaussian coefficient prior and Inverse-Gamma variance prior.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianLinearPrior {
    /// Prior precision of each coefficient relative to `sigma2`: `beta_j | sigma2 ~ N(0,
    /// sigma2 / precision_j)`. Finite and positive.
    pub coefficient_precision: Vec<f64>,
    /// Inverse-Gamma shape `a0`; finite and positive.
    pub variance_shape: f64,
    /// Inverse-Gamma scale `b0`; finite and positive.
    pub variance_scale: f64,
}

/// Normal-Inverse-Gamma posterior of the coefficients and the error variance.
#[derive(Clone, Debug, PartialEq)]
pub struct GaussianLinearPosterior {
    mean: Vec<f64>,
    /// Row-major `Lambda_n^-1`.
    precision_inverse: Vec<f64>,
    shape: f64,
    scale: f64,
}

/// Student-t marginal posterior of one linear functional of the coefficients.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct StudentPosterior {
    /// Posterior location (the posterior mean when the degrees of freedom exceed one).
    pub location: f64,
    /// Posterior scale.
    pub scale: f64,
    /// Degrees of freedom, `2 a_n`.
    pub degrees_of_freedom: f64,
}

/// Reporting summary of a Student-t marginal posterior.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PosteriorSummary {
    /// Posterior mean (the location).
    pub mean: f64,
    /// Posterior standard deviation; `NaN` when the degrees of freedom are at most two.
    pub standard_deviation: f64,
    /// Equal-tailed credible interval.
    pub interval: [f64; 2],
    /// Posterior probability that the functional is positive.
    pub probability_positive: f64,
    /// Degrees of freedom of the marginal posterior.
    pub degrees_of_freedom: f64,
}

impl StudentPosterior {
    /// Mean, spread, equal-tailed interval at `level` and the sign probability.
    #[must_use]
    pub fn summary(&self, level: f64) -> PosteriorSummary {
        PosteriorSummary {
            mean: self.location,
            standard_deviation: self.standard_deviation(),
            interval: self.equal_tailed_interval(level),
            probability_positive: self.probability_above(0.0),
            degrees_of_freedom: self.degrees_of_freedom,
        }
    }

    /// Posterior standard deviation, `scale sqrt(df / (df - 2))`; `NaN` unless `df > 2`.
    #[must_use]
    pub fn standard_deviation(&self) -> f64 {
        if self.degrees_of_freedom > 2.0 {
            self.scale * (self.degrees_of_freedom / (self.degrees_of_freedom - 2.0)).sqrt()
        } else {
            f64::NAN
        }
    }

    /// Posterior probability that the functional exceeds `threshold`.
    #[must_use]
    pub fn probability_above(&self, threshold: f64) -> f64 {
        student_t_sf((threshold - self.location) / self.scale, self.degrees_of_freedom)
    }

    /// Equal-tailed credible interval of the given level in `(0, 1)`.
    #[must_use]
    pub fn equal_tailed_interval(&self, level: f64) -> [f64; 2] {
        let radius = self.scale * student_t_ppf(0.5 + 0.5 * level, self.degrees_of_freedom);
        [self.location - radius, self.location + radius]
    }
}

fn invalid(detail: &str, message: &str) -> EstimationError {
    refuse(reason_code!("invalid_argument"), detail, message)
}

/// Posterior of a Gaussian linear model from `X' X` (row-major `p x p`), `X' y`, `y' y` and the
/// row count `rows`.
///
/// # Errors
///
/// `invalid_argument` with `gaussian_linear_posterior.invalid_prior` for a non-positive or
/// non-finite precision, shape or scale or a length that is not `p`;
/// `gaussian_linear_posterior.invalid_statistics` for non-finite sufficient statistics or a
/// shape mismatch; `gaussian_linear_posterior.singular_precision` when the posterior precision
/// cannot be inverted.
pub fn conjugate_posterior(
    gram: &[f64],
    xty: &[f64],
    yty: f64,
    rows: usize,
    prior: &GaussianLinearPrior,
) -> Result<GaussianLinearPosterior, EstimationError> {
    let p = xty.len();
    let positive = |v: f64| v.is_finite() && v > 0.0;
    if p == 0
        || prior.coefficient_precision.len() != p
        || !prior.coefficient_precision.iter().all(|v| positive(*v))
        || !positive(prior.variance_shape)
        || !positive(prior.variance_scale)
    {
        return Err(invalid(
            "gaussian_linear_posterior.invalid_prior",
            "the prior needs one positive finite precision per coefficient and a positive \
             finite Inverse-Gamma shape and scale",
        ));
    }
    if gram.len() != p * p
        || rows == 0
        || !yty.is_finite()
        || !gram.iter().chain(xty).all(|v| v.is_finite())
    {
        return Err(invalid(
            "gaussian_linear_posterior.invalid_statistics",
            "the sufficient statistics must be finite with a p x p Gram matrix and at least one row",
        ));
    }
    let mut precision = gram.to_vec();
    for (j, value) in prior.coefficient_precision.iter().enumerate() {
        precision[j * p + j] += value;
    }
    let precision_inverse = invert_square(&precision, p).ok_or_else(|| {
        invalid(
            "gaussian_linear_posterior.singular_precision",
            "the posterior precision matrix is singular",
        )
    })?;
    let mean: Vec<f64> = (0..p)
        .map(|i| (0..p).map(|j| precision_inverse[i * p + j] * xty[j]).sum::<f64>())
        .collect();
    let explained: f64 = mean.iter().zip(xty).map(|(m, c)| m * c).sum();
    let shape = prior.variance_shape + rows as f64 / 2.0;
    let scale = prior.variance_scale + 0.5 * (yty - explained).max(0.0);
    Ok(GaussianLinearPosterior { mean, precision_inverse, shape, scale })
}

impl GaussianLinearPosterior {
    /// Posterior mean of the coefficients.
    #[must_use]
    pub fn mean(&self) -> &[f64] {
        &self.mean
    }

    /// Posterior Inverse-Gamma shape `a_n` and scale `b_n` of `sigma2`.
    #[must_use]
    pub const fn variance_shape_and_scale(&self) -> (f64, f64) {
        (self.shape, self.scale)
    }

    /// Marginal posterior of `w' beta`; `None` when `weights` is not one entry per coefficient or
    /// is not finite.
    #[must_use]
    pub fn linear_functional(&self, weights: &[f64]) -> Option<StudentPosterior> {
        let p = self.mean.len();
        if weights.len() != p || !weights.iter().all(|w| w.is_finite()) {
            return None;
        }
        let location: f64 = weights.iter().zip(&self.mean).map(|(w, m)| w * m).sum();
        let mut quadratic = 0.0;
        for (i, wi) in weights.iter().enumerate() {
            for (j, wj) in weights.iter().enumerate() {
                quadratic += wi * self.precision_inverse[i * p + j] * wj;
            }
        }
        let scale = (self.scale / self.shape * quadratic.max(0.0)).sqrt();
        (scale.is_finite() && scale > 0.0).then_some(StudentPosterior {
            location,
            scale,
            degrees_of_freedom: 2.0 * self.shape,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // One coefficient, y = [1, 2, 3], X = [1, 1, 1]: X'X = 3, X'y = 6, y'y = 14.
    // Prior precision 1, a0 = 2, b0 = 1.
    // Lambda_n = 4, m_n = 6 / 4 = 1.5, a_n = 3.5, b_n = 1 + (14 - 9) / 2 = 3.5.
    #[test]
    fn scalar_posterior_matches_hand_algebra() {
        let prior = GaussianLinearPrior {
            coefficient_precision: vec![1.0],
            variance_shape: 2.0,
            variance_scale: 1.0,
        };
        let posterior = conjugate_posterior(&[3.0], &[6.0], 14.0, 3, &prior).unwrap();
        assert!((posterior.mean()[0] - 1.5).abs() < 1e-12);
        let (shape, scale) = posterior.variance_shape_and_scale();
        assert!((shape - 3.5).abs() < 1e-12 && (scale - 3.5).abs() < 1e-12);
        let t = posterior.linear_functional(&[1.0]).unwrap();
        // scale^2 = (b_n / a_n) / Lambda_n = 1 / 4.
        assert!((t.scale - 0.5).abs() < 1e-12);
        assert!((t.degrees_of_freedom - 7.0).abs() < 1e-12);
        assert!((t.probability_above(1.5) - 0.5).abs() < 1e-12);
        let [lo, hi] = t.equal_tailed_interval(0.9);
        assert!((0.5 * (lo + hi) - 1.5).abs() < 1e-12);
    }

    #[test]
    fn invalid_prior_and_singular_precision_refuse() {
        let bad = GaussianLinearPrior {
            coefficient_precision: vec![0.0],
            variance_shape: 2.0,
            variance_scale: 1.0,
        };
        assert!(conjugate_posterior(&[3.0], &[6.0], 14.0, 3, &bad).is_err());
        let prior = GaussianLinearPrior {
            coefficient_precision: vec![1.0; 2],
            variance_shape: 2.0,
            variance_scale: 1.0,
        };
        assert!(conjugate_posterior(&[1.0], &[6.0, 1.0], 14.0, 3, &prior).is_err());
    }
}
