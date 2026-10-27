//! Gaussian (identity-link) GLM fits.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{GlmDesignRef, GlmFit, check_prior_weights, validate_glm_shape};
use crate::error::StatsError;
use crate::linalg::{DenseLinearAlgebra, LeastSquaresWorkspace};

pub(super) fn fit_gaussian(
    design: GlmDesignRef<'_>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
) -> Result<GlmFit, StatsError> {
    validate_glm_shape(design)?;
    let GlmDesignRef { x_colmajor, nrows, ncols, y } = design;
    let fit = backend.least_squares(x_colmajor, nrows, ncols, y, workspace)?;
    Ok(GlmFit {
        coefficients: fit.coefficients,
        iterations: 1,
        converged: true,
        separated: false,
        boundary_saturated: false,
        boundary_margin: None,
        boundary_count: None,
        penalized: false,
        deviance: fit.rss,
        nb_alpha: None,
        diagnostics: fit.diagnostics,
    })
}

pub(super) fn fit_gaussian_weighted(
    design: GlmDesignRef<'_>,
    weights: &[f64],
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
) -> Result<GlmFit, StatsError> {
    check_prior_weights(design, weights)?;
    let GlmDesignRef { x_colmajor, nrows, ncols, y } = design;
    let root: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
    let y_w: Vec<f64> = y.iter().zip(&root).map(|(v, scale)| v * scale).collect();
    let fit = backend.row_scaled_least_squares(x_colmajor, nrows, ncols, &root, &y_w, workspace)?;
    Ok(GlmFit {
        coefficients: fit.coefficients,
        iterations: 1,
        converged: true,
        separated: false,
        boundary_saturated: false,
        boundary_margin: None,
        boundary_count: None,
        penalized: false,
        deviance: fit.rss,
        nb_alpha: None,
        diagnostics: fit.diagnostics,
    })
}
