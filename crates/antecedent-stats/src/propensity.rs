//! Propensity score fitting via logistic GLM.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::StatsError;
use crate::glm::{GlmDesignRef, GlmFamily, GlmFit, GlmOptions, fit_glm, fit_logistic_weighted};
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
/// Shape mismatch, non-binary treatment, GLM failure, non-convergence, or separation.
pub fn fit_propensity(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    treatment: &[f64],
    backend: &impl DenseLinearAlgebra,
    workspace: &mut PropensityWorkspace,
    options: &GlmOptions,
) -> Result<PropensityFit, StatsError> {
    let fit = fit_propensity_inner(
        x_colmajor,
        nrows,
        ncols,
        treatment,
        backend,
        workspace,
        &options.without_separation_ridge(),
    )?;
    fit.glm.require_ok()?;
    Ok(fit)
}

/// Fit propensity scores into `workspace.scores` without allocating a per-fit vector.
///
/// Bootstrap replicate loops use this form: scores land in
/// `workspace.scores[..nrows]` (the buffer [`PropensityWorkspace::prepare`]
/// sized), coefficients stay on the returned [`GlmFit`], and no per-replicate
/// heap allocation occurs. Numerically identical to [`fit_propensity`].
///
/// # Errors
///
/// Shape mismatch, non-binary treatment, GLM failure, non-convergence, or separation.
pub fn fit_propensity_in_place(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    treatment: &[f64],
    backend: &impl DenseLinearAlgebra,
    workspace: &mut PropensityWorkspace,
    options: &GlmOptions,
) -> Result<GlmFit, StatsError> {
    let glm = fit_propensity_glm(
        x_colmajor,
        nrows,
        ncols,
        treatment,
        backend,
        workspace,
        &options.without_separation_ridge(),
    )?;
    glm.require_ok()?;
    predict_propensity(x_colmajor, nrows, ncols, &glm.coefficients, &mut workspace.scores)?;
    Ok(glm)
}

/// Fit propensity scores into `workspace.scores` under prior (sampling) weights: the
/// weighted logistic maximum likelihood, whose score equation is `Σ w_i x_i (T_i − e_i) = 0`.
/// Numerically identical to [`fit_propensity_in_place`] when every weight is one.
///
/// # Errors
///
/// Shape mismatch, invalid weights, non-binary treatment, GLM failure, non-convergence, or
/// separation.
#[allow(
    clippy::too_many_arguments,
    reason = "the unweighted entry point's arguments plus the weights"
)]
pub fn fit_propensity_weighted_in_place(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    treatment: &[f64],
    weights: &[f64],
    backend: &impl DenseLinearAlgebra,
    workspace: &mut PropensityWorkspace,
    options: &GlmOptions,
) -> Result<GlmFit, StatsError> {
    if treatment.len() != nrows {
        return Err(StatsError::Shape { message: "treatment length != nrows" });
    }
    workspace.prepare(nrows);
    let glm = fit_logistic_weighted(
        GlmDesignRef { x_colmajor, nrows, ncols, y: treatment },
        weights,
        backend,
        &mut workspace.ols,
        &options.without_separation_ridge(),
    )?;
    glm.require_ok()?;
    predict_propensity(x_colmajor, nrows, ncols, &glm.coefficients, &mut workspace.scores)?;
    Ok(glm)
}

/// [`fit_propensity`] under prior (sampling) weights; scores are returned per row.
///
/// # Errors
///
/// As [`fit_propensity_weighted_in_place`].
#[allow(
    clippy::too_many_arguments,
    reason = "the unweighted entry point's arguments plus the weights"
)]
pub fn fit_propensity_weighted(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    treatment: &[f64],
    weights: &[f64],
    backend: &impl DenseLinearAlgebra,
    workspace: &mut PropensityWorkspace,
    options: &GlmOptions,
) -> Result<PropensityFit, StatsError> {
    let glm = fit_propensity_weighted_in_place(
        x_colmajor, nrows, ncols, treatment, weights, backend, workspace, options,
    )?;
    let scores = workspace.scores[..nrows].to_vec();
    Ok(PropensityFit { coefficients: glm.coefficients.clone(), scores, glm })
}

/// Diagnostic propensity fit that keeps soft-clamped scores under separation / non-convergence.
///
/// Estimation paths must use [`fit_propensity`]. Overlap / positivity diagnostics use this
/// path because complete separation *is* the positivity failure signal — refusing the fit
/// would turn a failed check into a hard error.
///
/// # Errors
///
/// Shape mismatch, non-binary treatment, or GLM linear-algebra failure.
pub fn fit_propensity_diagnostic(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    treatment: &[f64],
    backend: &impl DenseLinearAlgebra,
    workspace: &mut PropensityWorkspace,
    options: &GlmOptions,
) -> Result<PropensityFit, StatsError> {
    fit_propensity_inner(x_colmajor, nrows, ncols, treatment, backend, workspace, options)
}

fn fit_propensity_glm(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    treatment: &[f64],
    backend: &impl DenseLinearAlgebra,
    workspace: &mut PropensityWorkspace,
    options: &GlmOptions,
) -> Result<GlmFit, StatsError> {
    if treatment.len() != nrows {
        return Err(StatsError::Shape { message: "treatment length != nrows" });
    }
    workspace.prepare(nrows);
    fit_glm(
        GlmFamily::BinomialLogit,
        GlmDesignRef { x_colmajor, nrows, ncols, y: treatment },
        backend,
        &mut workspace.ols,
        options,
    )
}

fn fit_propensity_inner(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    treatment: &[f64],
    backend: &impl DenseLinearAlgebra,
    workspace: &mut PropensityWorkspace,
    options: &GlmOptions,
) -> Result<PropensityFit, StatsError> {
    let glm = fit_propensity_glm(x_colmajor, nrows, ncols, treatment, backend, workspace, options)?;
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
        out[r] = antecedent_kernels::sigmoid(eta).clamp(1e-9, 1.0 - 1e-9);
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
        let opts = GlmOptions { ridge_on_separation: None, ..GlmOptions::new(100, 1e-6) };
        let err = fit_propensity(&x, n, 2, &t, &FaerBackend, &mut ws, &opts);
        assert!(err.is_err(), "complete separation must error");
        let diag = fit_propensity_diagnostic(&x, n, 2, &t, &FaerBackend, &mut ws, &opts)
            .expect("diagnostics keep scores under separation");
        let min = diag.scores.iter().copied().fold(f64::INFINITY, f64::min);
        let max = diag.scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        assert!(min < 0.05 && max > 0.95, "min={min} max={max}");
    }
    /// A prior weight of `k` is `k` copies of the row: the weighted fit equals the fit on the
    /// replicated table, and a common rescaling of the weights changes nothing.
    #[test]
    fn integer_prior_weights_equal_replicated_rows() {
        let base_n = 40usize;
        let z: Vec<f64> = (0..base_n).map(|i| (i as f64 * 0.37).sin() * 1.5).collect();
        let t: Vec<f64> = (0..base_n)
            .map(|i| f64::from(u8::from(z[i] + 0.4 * ((i * 7 % 5) as f64 - 2.0) > 0.0)))
            .collect();
        let counts: Vec<usize> = (0..base_n).map(|i| 1 + i % 3).collect();
        let mut x = vec![1.0; base_n];
        x.extend_from_slice(&z);
        let weights: Vec<f64> = counts.iter().map(|&c| c as f64).collect();
        let (mut xr, mut zr, mut tr) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..base_n {
            for _ in 0..counts[i] {
                zr.push(z[i]);
                tr.push(t[i]);
            }
        }
        let big_n = tr.len();
        xr.extend(std::iter::repeat_n(1.0, big_n));
        xr.extend_from_slice(&zr);
        let opts = GlmOptions::new(100, 1e-12);
        let mut ws = PropensityWorkspace::default();
        let weighted =
            fit_propensity_weighted(&x, base_n, 2, &t, &weights, &FaerBackend, &mut ws, &opts)
                .unwrap();
        let replicated = fit_propensity(&xr, big_n, 2, &tr, &FaerBackend, &mut ws, &opts).unwrap();
        let scaled: Vec<f64> = weights.iter().map(|w| w * 1000.0).collect();
        let rescaled =
            fit_propensity_weighted(&x, base_n, 2, &t, &scaled, &FaerBackend, &mut ws, &opts)
                .unwrap();
        for c in 0..2 {
            assert!((weighted.coefficients[c] - replicated.coefficients[c]).abs() < 1e-8);
            assert!((weighted.coefficients[c] - rescaled.coefficients[c]).abs() < 1e-8);
        }
    }
}
