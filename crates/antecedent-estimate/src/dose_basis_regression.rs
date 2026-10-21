//! Conjugate Gaussian regression of an outcome on a fixed dose basis, with an explicit
//! coefficient prior.
//!
//! The model is `y_i = x_i' beta + e_i`, `e_i ~ N(0, sigma^2)`, independent. The prior is
//! `beta_j | sigma^2 ~ N(0, sigma^2 s_j^2)` with the scales `s_j` supplied by the caller: that
//! scale vector is the smoothing prior of the fitted curve, and a different vector is a
//! different prior. The residual variance is either known or has an `InvGamma(shape, scale)`
//! prior. The fit is the conjugate backend of `antecedent-prob`; this module only wires a
//! design, a prior and draw-major coefficient storage so that every functional of the curve
//! can be evaluated draw by draw on the same coefficient draw.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_prob::{
    BayesDesignRef, BayesFitOptions, GaussianCoefficientPrior, InvGammaPrior, LaplaceWorkspace,
    PriorSet, PriorSpec, fit_conjugate_gaussian,
};

use crate::error::EstimationError;

/// Most basis coefficients one regression may carry.
pub const MAX_BASIS_WIDTH: usize = 64;
/// Fewest posterior draws accepted.
pub const MIN_BASIS_DRAWS: usize = 200;
/// Most posterior draws one request may ask for.
pub const MAX_BASIS_DRAWS: usize = 100_000;

/// Prior of the residual variance `sigma^2`.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResidualPrior {
    /// The residual variance is known and fixed. The posterior of the coefficients is exactly
    /// Gaussian and its covariance is reported.
    KnownVariance(f64),
    /// `sigma^2 ~ InvGamma(shape, scale)`; the coefficient marginal is Student-t.
    InverseGamma {
        /// Shape, positive.
        shape: f64,
        /// Scale, positive.
        scale: f64,
    },
}

/// Posterior of the coefficients of one basis regression.
#[derive(Clone, Debug, PartialEq)]
pub struct BasisPosterior {
    /// Number of basis coefficients.
    pub width: usize,
    /// Posterior mean of the coefficients (given the residual variance when it is known).
    pub mean: Vec<f64>,
    /// Exact posterior covariance of the coefficients, row-major `width x width`, only when
    /// the residual variance is known.
    pub covariance: Option<Vec<f64>>,
    /// Number of draws.
    pub draws: usize,
    /// Draw-major coefficient draws: draw `k` occupies `[k * width, (k + 1) * width)`.
    pub coefficients: Vec<f64>,
}

/// Fit the conjugate regression of `y` on the column-major design and draw the posterior.
///
/// `coefficient_scales` has one positive finite scale per design column.
///
/// # Errors
///
/// A shape mismatch, a non-finite design or outcome, a non-positive prior scale or residual
/// prior, a draw count outside `200..=100000`, or a singular posterior precision.
pub fn fit_basis_regression(
    design_colmajor: &[f64],
    rows: usize,
    y: &[f64],
    coefficient_scales: &[f64],
    residual: ResidualPrior,
    draws: usize,
    seed: u64,
) -> Result<BasisPosterior, EstimationError> {
    let width = coefficient_scales.len();
    if width == 0
        || width > MAX_BASIS_WIDTH
        || rows == 0
        || y.len() != rows
        || design_colmajor.len() != rows * width
        || design_colmajor.iter().chain(y).any(|value| !value.is_finite())
    {
        return Err(EstimationError::data_msg(
            "the basis regression needs a finite design with one column per prior scale and one outcome per row",
        ));
    }
    let prior = checked_basis_prior(coefficient_scales, residual, draws)?;
    let fit = fit_conjugate_gaussian(
        BayesDesignRef {
            x_colmajor: design_colmajor,
            nrows: rows,
            ncols: width,
            y,
            weights: None,
            offsets: None,
        },
        &prior,
        &BayesFitOptions { n_draws: draws, seed, ..BayesFitOptions::default() },
        &mut LaplaceWorkspace::default(),
    )?;
    let mut coefficients = vec![0.0; draws * width];
    for j in 0..width {
        let column = fit.draws.column(j)?;
        for (k, value) in column.iter().enumerate() {
            coefficients[k * width + j] = *value;
        }
    }
    Ok(BasisPosterior { width, mean: fit.map, covariance: fit.cov, draws, coefficients })
}

/// Shared prior/draw validation used by cheap compilation and numerical fitting.
pub(crate) fn checked_basis_prior(
    coefficient_scales: &[f64],
    residual: ResidualPrior,
    draws: usize,
) -> Result<PriorSet, EstimationError> {
    let width = coefficient_scales.len();
    if width == 0 || width > MAX_BASIS_WIDTH {
        return Err(EstimationError::data_msg("basis coefficient width must be between 1 and 64"));
    }
    if coefficient_scales.iter().any(|scale| !scale.is_finite() || *scale <= 0.0) {
        return Err(EstimationError::data_msg(
            "coefficient prior scales must be finite and positive",
        ));
    }
    if !(MIN_BASIS_DRAWS..=MAX_BASIS_DRAWS).contains(&draws) {
        return Err(EstimationError::data_msg("posterior draws must be between 200 and 100000"));
    }
    let residual_spec = match residual {
        ResidualPrior::KnownVariance(variance) => {
            if !variance.is_finite() || variance <= 0.0 {
                return Err(EstimationError::data_msg(
                    "the known residual variance must be finite and positive",
                ));
            }
            PriorSpec::KnownResidualVariance(variance)
        }
        ResidualPrior::InverseGamma { shape, scale } => {
            PriorSpec::ResidualInvGamma(InvGammaPrior { shape, scale })
        }
    };
    let mut prior = PriorSet::new();
    prior.push(PriorSpec::GaussianCoefficients(GaussianCoefficientPrior {
        mean: Arc::from(vec![0.0; width]),
        variance: Arc::from(coefficient_scales.iter().map(|s| s * s).collect::<Vec<_>>()),
    }));
    prior.push(residual_spec);
    Ok(prior)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn known_variance_posterior_matches_the_normal_equations() {
        // Intercept and slope, five rows. The posterior mean and covariance are the
        // textbook ridge solution (V0^-1 + X'X)^-1 X'y and sigma^2 (V0^-1 + X'X)^-1, computed
        // independently with numpy: scales (2, 1), sigma^2 = 0.5.
        let x1 = [-1.0, -0.5, 0.0, 0.5, 1.0];
        let y = [0.1, 0.6, 0.9, 1.7, 2.2];
        let mut design = vec![1.0; 5];
        design.extend(x1);
        let fit = fit_basis_regression(
            &design,
            5,
            &y,
            &[2.0, 1.0],
            ResidualPrior::KnownVariance(0.5),
            400,
            3,
        )
        .unwrap();
        assert!((fit.mean[0] - 1.047_619_047_619_05).abs() < 1e-9, "{:?}", fit.mean);
        assert!((fit.mean[1] - 0.757_142_857_142_857).abs() < 1e-9, "{:?}", fit.mean);
        let cov = fit.covariance.unwrap();
        assert!((cov[0] - 0.095_238_095_238_095_2).abs() < 1e-9, "{cov:?}");
        assert!((cov[3] - 0.142_857_142_857_143).abs() < 1e-9, "{cov:?}");
        assert!(cov[1].abs() < 1e-12 && cov[2].abs() < 1e-12);
        assert_eq!(fit.coefficients.len(), 800);
    }

    #[test]
    fn invalid_priors_and_shapes_are_refused() {
        let design = [1.0, 1.0, 1.0];
        let y = [0.0, 1.0, 2.0];
        let known = ResidualPrior::KnownVariance(1.0);
        assert!(fit_basis_regression(&design, 3, &y, &[0.0], known, 400, 1).is_err());
        assert!(fit_basis_regression(&design, 3, &y, &[1.0, 1.0], known, 400, 1).is_err());
        assert!(fit_basis_regression(&design, 3, &y, &[1.0], known, 10, 1).is_err());
        let variance = ResidualPrior::KnownVariance(-1.0);
        assert!(fit_basis_regression(&design, 3, &y, &[1.0], variance, 400, 1).is_err());
    }
}
