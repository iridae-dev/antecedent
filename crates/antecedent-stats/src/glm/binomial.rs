//! Logistic and probit GLM fits, separation diagnostics and the ridge-penalized binomial fit.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_kernels::{norm_cdf, norm_pdf, sigmoid};
use crate::error::StatsError;
use crate::gram::{column_is_constant, invert_square};
use crate::linalg::{DenseLinearAlgebra, FitDiagnostics, LeastSquaresWorkspace};
use super::{GlmDesignRef, GlmFamily, GlmFit, GlmOptions, MEAN_FLOOR, PROB_CLIP, SATURATION_MARGIN, check_prior_weights, design_column, irls_fit, irls_fit_scaled, linear_predictor_into, validate_glm_shape};

/// Binomial deviance and separation evidence at `β`.
pub(super) struct BinomialDiagnostics {
    pub(super) deviance: f64,
    /// Some fitted probability is outside `[1e-8, 1 − 1e-8]`.
    pub(super) saturated: bool,
    /// Smallest `min(mu, 1 − mu)` over the rows.
    pub(super) min_margin: f64,
    /// Rows with a fitted probability outside `[1e-8, 1 − 1e-8]`.
    pub(super) n_saturated: u64,
    /// `η > 0` for every `y = 1` row and `η < 0` for every `y = 0` row: `β` itself is a
    /// separating direction, which proves complete separation.
    pub(super) perfectly_classified: bool,
}

impl BinomialDiagnostics {
    /// Separation verdict: saturated probabilities that either failed to converge or sit on
    /// a perfectly separating hyperplane. Saturation alone (a strong predictor) is not
    /// separation.
    pub(super) fn separated(&self, converged: bool) -> bool {
        self.saturated && (!converged || self.perfectly_classified)
    }
}

#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
fn binomial_diagnostics_at(
    family: GlmFamily,
    design: GlmDesignRef<'_>,
    beta: &[f64],
) -> BinomialDiagnostics {
    let GlmDesignRef { x_colmajor, nrows, ncols, y } = design;
    let mut deviance = 0.0;
    let mut saturated = false;
    let mut min_margin = f64::INFINITY;
    let mut n_saturated = 0u64;
    let mut perfectly_classified = true;
    let mut eta = vec![0.0; nrows];
    linear_predictor_into(x_colmajor, nrows, ncols, beta, &mut eta);
    for r in 0..nrows {
        let eta_r = eta[r];
        let mu = match family {
            GlmFamily::BinomialLogit => sigmoid(eta_r),
            GlmFamily::BinomialProbit => norm_cdf(eta_r),
            _ => unreachable!("binomial diagnostics require a binomial family"),
        };
        if !(SATURATION_MARGIN..=1.0 - SATURATION_MARGIN).contains(&mu) {
            saturated = true;
            n_saturated += 1;
        }
        min_margin = min_margin.min(mu.min(1.0 - mu));
        if (y[r] > 0.0 && eta_r <= 0.0) || (y[r] <= 0.0 && eta_r >= 0.0) {
            perfectly_classified = false;
        }
        let mu_clamped = mu.clamp(PROB_CLIP, 1.0 - PROB_CLIP);
        if y[r] > 0.0 {
            deviance += -2.0 * mu_clamped.ln();
        } else {
            deviance += -2.0 * (1.0 - mu_clamped).ln();
        }
    }
    BinomialDiagnostics { deviance, saturated, min_margin, n_saturated, perfectly_classified }
}

#[allow(clippy::float_cmp, reason = "binomial outcomes must be exactly the coded values 0 or 1")]
pub(super) fn fit_logistic(
    design: GlmDesignRef<'_>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<GlmFit, StatsError> {
    validate_glm_shape(design)?;
    let GlmDesignRef { y, .. } = design;
    for &yi in y {
        if !(yi == 0.0 || yi == 1.0) {
            return Err(StatsError::Shape { message: "binomial GLM requires 0/1 outcomes" });
        }
    }

    let (beta, iterations, converged) =
        irls_fit(design, backend, workspace, options, false, |eta, yi| {
            let mu = sigmoid(eta);
            let mu_clamped = mu.clamp(PROB_CLIP, 1.0 - PROB_CLIP);
            let w = (mu_clamped * (1.0 - mu_clamped)).sqrt();
            let z = eta + (yi - mu_clamped) / (mu_clamped * (1.0 - mu_clamped));
            (w, z * w)
        })?;

    finish_binomial(
        GlmFamily::BinomialLogit,
        design,
        workspace,
        options,
        beta,
        iterations,
        converged,
    )
}

/// Logistic regression with nonnegative prior (case) weights: the maximizer of
/// `Σ w_i [y_i log μ_i + (1 − y_i) log(1 − μ_i)]`.
///
/// This is the design-consistent fit when `w` are sampling weights: its estimating equation
/// `Σ w_i x_i (y_i − μ_i) = 0` is the weighted-population score. Separation diagnostics
/// are those of the unweighted rows (a zero weight is the caller's to drop beforehand), and
/// the separation ridge refit is not offered: estimation paths refuse every separated fit.
///
/// # Errors
///
/// Shape mismatch, a weight that is negative or not finite, non-0/1 outcomes, or a
/// linear-algebra failure.
#[allow(clippy::float_cmp, reason = "binomial outcomes must be exactly the coded values 0 or 1")]
pub fn fit_logistic_weighted(
    design: GlmDesignRef<'_>,
    weights: &[f64],
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<GlmFit, StatsError> {
    validate_glm_shape(design)?;
    if weights.len() != design.nrows {
        return Err(StatsError::Shape { message: "weights length != nrows" });
    }
    if weights.iter().any(|w| !w.is_finite() || *w < 0.0) {
        return Err(StatsError::Shape { message: "prior weights must be finite and non-negative" });
    }
    if design.y.iter().any(|&yi| !(yi == 0.0 || yi == 1.0)) {
        return Err(StatsError::Shape { message: "binomial GLM requires 0/1 outcomes" });
    }
    let root_weights: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
    let (beta, iterations, converged) = irls_fit_scaled(
        design,
        backend,
        workspace,
        options,
        false,
        Some(&root_weights),
        None,
        |eta, yi| {
            let mu = sigmoid(eta).clamp(PROB_CLIP, 1.0 - PROB_CLIP);
            let w = (mu * (1.0 - mu)).sqrt();
            let z = eta + (yi - mu) / (mu * (1.0 - mu));
            (w, z * w)
        },
    )?;
    finish_binomial(
        GlmFamily::BinomialLogit,
        design,
        workspace,
        &options.without_separation_ridge(),
        beta,
        iterations,
        converged,
    )
}

#[allow(clippy::float_cmp, reason = "binomial outcomes must be exactly the coded values 0 or 1")]
pub(super) fn fit_probit_weighted(
    design: GlmDesignRef<'_>,
    weights: &[f64],
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<GlmFit, StatsError> {
    check_prior_weights(design, weights)?;
    if design.y.iter().any(|&yi| !(yi == 0.0 || yi == 1.0)) {
        return Err(StatsError::Shape { message: "binomial GLM requires 0/1 outcomes" });
    }
    let root: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
    let (beta, iterations, converged) =
        irls_fit_scaled(design, backend, workspace, options, false, Some(&root), None, |eta, yi| {
            let mu = norm_cdf(eta).clamp(PROB_CLIP, 1.0 - PROB_CLIP);
            let phi = norm_pdf(eta).max(MEAN_FLOOR);
            let denom = (mu * (1.0 - mu)).max(MEAN_FLOOR);
            let w = ((phi * phi) / denom).sqrt();
            (w, (eta + (yi - mu) / phi) * w)
        })?;
    finish_binomial(
        GlmFamily::BinomialProbit,
        design,
        workspace,
        &options.without_separation_ridge(),
        beta,
        iterations,
        converged,
    )
}

/// Deviance and separation diagnostics for an unpenalized binomial IRLS result, with the
/// optional ridge refit when the fitted probabilities saturate.
fn finish_binomial(
    family: GlmFamily,
    design: GlmDesignRef<'_>,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
    beta: Vec<f64>,
    iterations: u32,
    converged: bool,
) -> Result<GlmFit, StatsError> {
    // Diagnostics at the returned coefficients (not the pre-update IRLS iterate).
    let diag = binomial_diagnostics_at(family, design, &beta);
    let separated = diag.separated(converged);

    if diag.saturated {
        if let Some(lambda) = options.ridge_on_separation {
            if lambda > 0.0 {
                return fit_binomial_ridge(
                    family,
                    design,
                    workspace,
                    options,
                    lambda,
                    Some(separated),
                    None,
                );
            }
        }
    }

    Ok(GlmFit {
        coefficients: beta,
        iterations,
        converged,
        separated,
        boundary_saturated: diag.saturated,
        boundary_margin: Some(diag.min_margin),
        boundary_count: Some(diag.n_saturated),
        penalized: false,
        deviance: diag.deviance,
        nb_alpha: None,
        diagnostics: FitDiagnostics::new(design.ncols, None, "glm-irls", workspace.grow_count),
    })
}

#[allow(clippy::float_cmp, reason = "binomial outcomes must be exactly the coded values 0 or 1")]
pub(super) fn fit_probit(
    design: GlmDesignRef<'_>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<GlmFit, StatsError> {
    validate_glm_shape(design)?;
    let GlmDesignRef { y, .. } = design;
    for &yi in y {
        if !(yi == 0.0 || yi == 1.0) {
            return Err(StatsError::Shape { message: "binomial GLM requires 0/1 outcomes" });
        }
    }

    let (beta, iterations, converged) =
        irls_fit(design, backend, workspace, options, false, |eta, yi| {
            let mu = norm_cdf(eta);
            let mu_clamped = mu.clamp(PROB_CLIP, 1.0 - PROB_CLIP);
            let phi = norm_pdf(eta).max(MEAN_FLOOR);
            let denom = (mu_clamped * (1.0 - mu_clamped)).max(MEAN_FLOOR);
            let w = ((phi * phi) / denom).sqrt();
            (w, (eta + (yi - mu_clamped) / phi) * w)
        })?;

    finish_binomial(
        GlmFamily::BinomialProbit,
        design,
        workspace,
        options,
        beta,
        iterations,
        converged,
    )
}

/// Weighted normal equations `Xᵀ W X` and `Xᵀ W z` from contiguous columns.
///
/// Entry `(i, j)` is `Σ_r (w_r x_{ri}) x_{rj}` summed in row order, the same
/// products the rank-one row update added. A non-finite weight or entry stays
/// in the matrix; the caller refuses it when the factorization fails.
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
pub(super) fn accumulate_weighted_normal_equations(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    fisher: &[f64],
    working: &[f64],
    xtwx: &mut [f64],
    xtwz: &mut [f64],
) {
    xtwx[..ncols * ncols].fill(0.0);
    xtwz[..ncols].fill(0.0);
    if nrows == 0 || ncols == 0 {
        return;
    }
    for c1 in 0..ncols {
        let col1 = design_column(x_colmajor, nrows, c1);
        let mut grad = 0.0;
        for r in 0..nrows {
            grad += (col1[r] * fisher[r]) * working[r];
        }
        xtwz[c1] = grad;
        for c2 in 0..ncols {
            let col2 = design_column(x_colmajor, nrows, c2);
            let mut acc = 0.0;
            for r in 0..nrows {
                acc += (col1[r] * fisher[r]) * col2[r];
            }
            xtwx[c1 * ncols + c2] = acc;
        }
    }
}

/// Ridge-penalized binomial IRLS: the separation fallback (`fallback_separated` is the
/// unpenalized fit's verdict, which the penalized fit must not clear) or the deliberate
/// always-on ridge of [`fit_glm_ridge`] (`None`: the verdict is taken at the ridge solution).
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
fn fit_binomial_ridge(
    family: GlmFamily,
    design: GlmDesignRef<'_>,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
    lambda: f64,
    fallback_separated: Option<bool>,
    prior: Option<&[f64]>,
) -> Result<GlmFit, StatsError> {
    let GlmDesignRef { x_colmajor, nrows, ncols, y } = design;
    let mut beta = vec![0.0; ncols];
    let mut beta_new = vec![0.0; ncols];
    let mut xtwx = vec![0.0; ncols * ncols];
    let mut xtwz = vec![0.0; ncols];
    let mut eta = vec![0.0; nrows];
    let mut fisher = vec![0.0; nrows];
    let mut working_z = vec![0.0; nrows];
    let mut converged = false;
    let mut iterations = 0u32;
    // The intercept column does not change between iterations.
    let unpenalize0 = ncols > 0 && column_is_constant(x_colmajor, nrows, 0);

    for iter in 1..=options.max_iter {
        iterations = iter;
        let mut max_delta = 0.0_f64;
        linear_predictor_into(x_colmajor, nrows, ncols, &beta, &mut eta);
        for r in 0..nrows {
            let eta_r = eta[r];
            let (w_fisher, working) = match family {
                GlmFamily::BinomialLogit => {
                    let mu = sigmoid(eta_r).clamp(PROB_CLIP, 1.0 - PROB_CLIP);
                    let w = mu * (1.0 - mu);
                    let z = eta_r + (y[r] - mu) / w.max(MEAN_FLOOR);
                    (w, z)
                }
                GlmFamily::BinomialProbit => {
                    let mu = norm_cdf(eta_r).clamp(PROB_CLIP, 1.0 - PROB_CLIP);
                    let phi = norm_pdf(eta_r).max(MEAN_FLOOR);
                    let w = (phi * phi) / (mu * (1.0 - mu)).max(MEAN_FLOOR);
                    let z = eta_r + (y[r] - mu) / phi;
                    (w, z)
                }
                _ => unreachable!("ridge fallback is binomial-only"),
            };
            fisher[r] = w_fisher * prior.map_or(1.0, |m| m[r]);
            working_z[r] = working;
        }
        accumulate_weighted_normal_equations(
            x_colmajor, nrows, ncols, &fisher, &working_z, &mut xtwx, &mut xtwz,
        );
        for c in 0..ncols {
            if c == 0 && unpenalize0 {
                continue;
            }
            xtwx[c * ncols + c] += lambda;
        }
        let Some(inv) = invert_square(&xtwx, ncols) else {
            return Err(StatsError::Backend("ridge binomial: singular X'WX+λI".into()));
        };
        for i in 0..ncols {
            let mut s = 0.0;
            for j in 0..ncols {
                s += inv[i * ncols + j] * xtwz[j];
            }
            beta_new[i] = s;
            max_delta = max_delta.max((s - beta[i]).abs());
        }
        beta.copy_from_slice(&beta_new);
        if max_delta < options.tol {
            converged = true;
            break;
        }
    }

    // A separation fallback keeps the unpenalized fit's verdict: penalized scores are not a
    // substitute for an unpenalized MLE, so estimation paths (`require_ok`) still refuse
    // them. A deliberate ridge fit is judged at its own solution and is `penalized`.
    let diag = binomial_diagnostics_at(family, design, &beta);
    let separated = fallback_separated.unwrap_or_else(|| diag.separated(converged));

    Ok(GlmFit {
        coefficients: beta,
        iterations,
        converged,
        separated,
        boundary_saturated: diag.saturated || fallback_separated.is_some(),
        boundary_margin: Some(diag.min_margin),
        boundary_count: Some(diag.n_saturated),
        penalized: true,
        deviance: diag.deviance,
        nb_alpha: None,
        diagnostics: FitDiagnostics::new(ncols, None, "glm-irls", workspace.grow_count),
    })
}

/// Fit a binomial GLM with an always-active ridge penalty.
///
/// A constant first design column is treated as an intercept and is not
/// penalized. Unlike [`GlmOptions::ridge_on_separation`], this entry point
/// applies `lambda` whether or not the unpenalized fit is separated.
///
/// # Errors
///
/// Invalid shapes, outcomes, family, penalty, or a numerical solver failure.
#[allow(clippy::float_cmp, reason = "binomial outcomes must be exactly the coded values 0 or 1")]
pub fn fit_glm_ridge(
    family: GlmFamily,
    design: GlmDesignRef<'_>,
    _backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
    lambda: f64,
) -> Result<GlmFit, StatsError> {
    antecedent_core::execution_attempt::run_operation(
        antecedent_core::execution_attempt::Operation::GlmFit,
        || {
            if !matches!(family, GlmFamily::BinomialLogit | GlmFamily::BinomialProbit) {
                return Err(StatsError::Shape {
                    message: "ridge GLM currently requires a binomial family",
                });
            }
            validate_glm_shape(design)?;
            if !lambda.is_finite() || lambda <= 0.0 {
                return Err(StatsError::Shape {
                    message: "ridge lambda must be finite and positive",
                });
            }
            if design.y.iter().any(|&yi| yi != 0.0 && yi != 1.0) {
                return Err(StatsError::Shape { message: "binomial GLM requires 0/1 outcomes" });
            }
            fit_binomial_ridge(family, design, workspace, options, lambda, None, None)
        },
    )
}

/// [`fit_glm_ridge`] with nonnegative prior (case) weights: the maximizer of
/// `Σ w_i ℓ_i(β) − ½ λ |β_{1..}|²` (a constant first column is the unpenalized intercept).
///
/// # Errors
///
/// As [`fit_glm_ridge`], or a weights length different from the row count or a weight that is
/// negative or not finite.
#[allow(clippy::float_cmp, reason = "binomial outcomes must be exactly the coded values 0 or 1")]
pub fn fit_glm_ridge_weighted(
    family: GlmFamily,
    design: GlmDesignRef<'_>,
    weights: &[f64],
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
    lambda: f64,
) -> Result<GlmFit, StatsError> {
    check_prior_weights(design, weights)?;
    let _ = backend;
    if !matches!(family, GlmFamily::BinomialLogit | GlmFamily::BinomialProbit) {
        return Err(StatsError::Shape {
            message: "ridge GLM currently requires a binomial family",
        });
    }
    if !lambda.is_finite() || lambda <= 0.0 {
        return Err(StatsError::Shape { message: "ridge lambda must be finite and positive" });
    }
    if design.y.iter().any(|&yi| yi != 0.0 && yi != 1.0) {
        return Err(StatsError::Shape { message: "binomial GLM requires 0/1 outcomes" });
    }
    fit_binomial_ridge(family, design, workspace, options, lambda, None, Some(weights))
}
