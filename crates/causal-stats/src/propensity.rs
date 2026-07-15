//! Propensity score fitting via logistic GLM.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use crate::error::StatsError;
use crate::glm::{GlmDesignRef, GlmFamily, GlmFit, GlmOptions, fit_glm};
use crate::linalg::{DenseLinearAlgebra, LeastSquaresWorkspace};

/// Fitted propensity scores retained for diagnostics and resampling.
#[derive(Clone, Debug)]
pub struct PropensityFit {
    /// Logistic coefficients (intercept + covariates).
    pub coefficients: Vec<f64>,
    /// Fitted P(T=1 | X) per row, length `nrows`.
    pub scores: Vec<f64>,
    /// GLM diagnostics.
    pub glm: GlmFit,
}

/// Workspace for repeated propensity fits (bootstrap reuse).
#[derive(Clone, Debug, Default)]
pub struct PropensityWorkspace {
    /// Underlying LS scratch used by IRLS.
    pub ols: LeastSquaresWorkspace,
    /// Scratch for predicted scores.
    pub scores: Vec<f64>,
    /// Number of times [`Self::prepare`] grew the scores buffer.
    pub scores_grow_count: u32,
}

impl PropensityWorkspace {
    /// Ensure capacity for `nrows` scores.
    pub fn prepare(&mut self, nrows: usize) {
        if self.scores.len() < nrows {
            self.scores.resize(nrows, 0.0);
            self.scores_grow_count = self.scores_grow_count.saturating_add(1);
        }
    }
}

/// Fit propensity scores: logistic regression of binary treatment on covariates.
///
/// `x_colmajor` is column-major with an intercept column (typically column 0 all ones)
/// and covariate columns; length ≥ `nrows * ncols`. `treatment` must be 0/1, length `nrows`.
///
/// # Errors
///
/// Shape mismatch, non-binary treatment, or GLM failure.
pub fn fit_propensity(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    treatment: &[f64],
    backend: &impl DenseLinearAlgebra,
    workspace: &mut PropensityWorkspace,
    options: &GlmOptions,
) -> Result<PropensityFit, StatsError> {
    if treatment.len() != nrows {
        return Err(StatsError::Shape { message: "treatment length != nrows" });
    }
    workspace.prepare(nrows);
    let glm = fit_glm(
        GlmFamily::BinomialLogit,
        GlmDesignRef { x_colmajor, nrows, ncols, y: treatment },
        backend,
        &mut workspace.ols,
        options,
    )?;
    glm.require_ok()?;
    let mut scores = vec![0.0; nrows];
    predict_propensity(x_colmajor, nrows, ncols, &glm.coefficients, &mut scores)?;
    Ok(PropensityFit { coefficients: glm.coefficients.clone(), scores, glm })
}

/// Predict propensity scores from coefficients into `out` (length `nrows`).
///
/// # Errors
///
/// Shape mismatch.
pub fn predict_propensity(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    coefficients: &[f64],
    out: &mut [f64],
) -> Result<(), StatsError> {
    if coefficients.len() != ncols {
        return Err(StatsError::Shape { message: "coefficient length != ncols" });
    }
    if out.len() < nrows {
        return Err(StatsError::Shape { message: "output buffer too short" });
    }
    if x_colmajor.len() < nrows.saturating_mul(ncols) {
        return Err(StatsError::Shape { message: "X buffer too short" });
    }
    for r in 0..nrows {
        let mut eta = 0.0;
        for c in 0..ncols {
            eta += x_colmajor[c * nrows + r] * coefficients[c];
        }
        out[r] = (1.0 / (1.0 + (-eta).exp())).clamp(1e-9, 1.0 - 1e-9);
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::faer_backend::FaerBackend;

    #[test]
    fn propensity_separates_treatment() {
        let n = 100usize;
        let mut x = vec![0.0; n * 2];
        let mut t = vec![0.0; n];
        for i in 0..n {
            let z = if i < n / 2 { -1.0 } else { 1.0 };
            x[i] = 1.0;
            x[n + i] = z;
            t[i] = if z > 0.0 { 1.0 } else { 0.0 };
            if i % 20 == 0 {
                t[i] = 1.0 - t[i];
            }
        }
        let mut ws = PropensityWorkspace::default();
        let fit = fit_propensity(&x, n, 2, &t, &FaerBackend, &mut ws, &GlmOptions::new(100, 1e-6))
            .unwrap();
        assert!(fit.glm.converged);
        let mean_treated: f64 = fit
            .scores
            .iter()
            .zip(t.iter())
            .filter(|&(_, &ti)| ti > 0.5)
            .map(|(s, _)| s)
            .sum::<f64>()
            / 50.0;
        let mean_control: f64 = fit
            .scores
            .iter()
            .zip(t.iter())
            .filter(|&(_, &ti)| ti < 0.5)
            .map(|(s, _)| s)
            .sum::<f64>()
            / 50.0;
        assert!(mean_treated > mean_control);
    }

    #[test]
    fn propensity_errors_on_complete_separation() {
        let n = 80usize;
        let mut x = vec![0.0; n * 2];
        let mut t = vec![0.0; n];
        for i in 0..n {
            let z = if i < n / 2 { -1.0 } else { 1.0 };
            x[i] = 1.0;
            x[n + i] = z;
            t[i] = if z > 0.0 { 1.0 } else { 0.0 };
        }
        let mut ws = PropensityWorkspace::default();
        let err = fit_propensity(&x, n, 2, &t, &FaerBackend, &mut ws, &GlmOptions::new(100, 1e-6));
        assert!(err.is_err(), "complete separation must error");
    }
}
