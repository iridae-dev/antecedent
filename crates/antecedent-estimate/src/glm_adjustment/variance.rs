//! Extracted stage of the parent estimator module.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

pub(super) fn average_gcomp_for_target(
    diffs: &[f64],
    treatment: &[f64],
    target: &TargetPopulation,
    weights: Option<&[f64]>,
) -> Result<f64, EstimationError> {
    let mut sum = 0.0;
    let mut mass = 0.0;
    let mut count = 0usize;
    for (i, &d) in diffs.iter().enumerate() {
        let include = include_gcomp_row(target, treatment.get(i).copied().unwrap_or(0.0));
        if include {
            let weight = weights.map_or(1.0, |w| w[i]);
            sum += weight * d;
            mass += weight;
            count += 1;
        }
    }
    if count == 0 {
        return Err(EstimationError::data_msg(
            "target population left no rows for GLM g-computation",
        ));
    }
    Ok(if weights.is_some() { sum / mass } else { sum / count as f64 })
}

pub(super) fn include_gcomp_row(target: &TargetPopulation, t: f64) -> bool {
    match target {
        TargetPopulation::Treated => TreatmentArm::is_treated_indicator(t),
        TargetPopulation::Untreated => !TreatmentArm::is_treated_indicator(t),
        _ => true,
    }
}

/// Delta-method standard error for the g-computation ATE, **conditional on the observed
/// covariate rows** (standard g-computation practice).
///
/// With gradient `g = (1/n⋆) Σ_{i ∈ T} [μ'(η_i1)·x_i1 − μ'(η_i0)·x_i0]` over the
/// coefficient vector, where `T` is the target population (all rows, treated, or
/// untreated) and `n⋆ = |T|`. `x_i1`/`x_i0` are row `i` with the treatment column
/// set to `active`/`control`. `Cov(β̂) = φ·(XᵀWX)⁻¹` is still the full-sample
/// inverse Fisher information at the fit (`W = diag(w(η_i))` with
/// Bernoulli/logit / Poisson `w = μ'`, NB2 `w = μ/(1+αμ)`, and probit
/// `w = φ²/(μ(1−μ))`; dispersion `φ = RSS/(n−p)` for Gaussian, the fitted NB2
/// `α` for negative binomial, and `1` otherwise). The SE is `sqrt(gᵀ Cov(β̂) g)`.
/// Returns `NaN` when the information matrix is singular.
pub(super) fn gcomp_delta_method_se(
    contrast: &GcompContrast<'_>,
    coefficients: &[f64],
    deviance: f64,
    nb_alpha: f64,
    target: &GcompTarget<'_>,
) -> f64 {
    let ncols = contrast.ncols;
    let Some(cov_unscaled) = adjustment_glm_covariance(
        contrast.family,
        contrast.x_colmajor,
        contrast.nrows,
        ncols,
        coefficients,
        deviance,
        nb_alpha,
    ) else {
        return f64::NAN;
    };
    let grad = gcomp_gradient(contrast, coefficients, target);

    let mut quad = 0.0;
    for i in 0..ncols {
        for j in 0..ncols {
            quad += grad[i] * cov_unscaled[i * ncols + j] * grad[j];
        }
    }
    quad.max(0.0).sqrt()
}

/// Coefficient Fisher covariance shared by ordinary and retained GLM adjustment.
pub(crate) fn adjustment_glm_covariance(
    family: GlmFamily,
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    coefficients: &[f64],
    deviance: f64,
    nb_alpha: f64,
) -> Option<Vec<f64>> {
    // A saturated Gaussian fit identifies its coefficients, but has no residual
    // degrees of freedom with which to estimate their sampling covariance.
    if family == GlmFamily::GaussianIdentity && nrows <= ncols {
        return None;
    }
    // Fisher information XᵀWX at the fitted coefficients, via a √W-scaled design copy.
    let sqrt_w: Vec<f64> = linear_predictors(x_colmajor, nrows, ncols, coefficients)
        .into_iter()
        .map(|eta| fisher_weight(family, eta, nb_alpha).max(0.0).sqrt())
        .collect();
    let mut x_w = vec![0.0; nrows * ncols];
    for (scaled, column) in
        x_w.chunks_exact_mut(nrows.max(1)).zip(x_colmajor.chunks_exact(nrows.max(1)))
    {
        for ((slot, &x), &w) in scaled.iter_mut().zip(column).zip(&sqrt_w) {
            *slot = x * w;
        }
    }
    let cov_unscaled = crate::util::xtx_inverse(&x_w, nrows, ncols)?;
    let n = nrows as f64;
    let dispersion = match family {
        // For Gaussian/identity the fit's deviance is the RSS.
        GlmFamily::GaussianIdentity => deviance / (n - ncols as f64),
        // NB2 carries α in its Fisher weight μ/(1+αμ), so (XᵀWX)⁻¹ is already the
        // full covariance; like the other fixed-dispersion families it scales by 1.
        GlmFamily::NegativeBinomial
        | GlmFamily::BinomialLogit
        | GlmFamily::BinomialProbit
        | GlmFamily::PoissonLog => 1.0,
    };

    Some(cov_unscaled.into_iter().map(|c| c * dispersion).collect())
}

/// Shared unpenalized adjusted GLM solve, requiring a valid finite likelihood fit.
pub(crate) fn solve_adjustment_glm(
    family: GlmFamily,
    design: GlmDesignRef<'_>,
    backend: FaerBackend,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<antecedent_stats::GlmFit, EstimationError> {
    let fit = fit_glm(family, design, &backend, workspace, &options.without_separation_ridge())
        .map_err(crate::error::EstimationError::from)?;
    fit.require_ok().map_err(crate::error::EstimationError::from)?;
    crate::adjustment_resume::record_model_fit();
    Ok(fit)
}

/// G-computation SE using score-exact GLM sandwich Cov(β̂) then the same
/// mean-contrast gradient as the Fisher delta-method.
///
/// Meat uses score contributions `s_i = u_i x_i` with
/// `u_i = (y_i−μ_i) V(μ_i)⁻¹ μ'(η_i)`; bread is Fisher `(XᵀWX)⁻¹`.
pub(super) fn gcomp_sandwich_se(
    spec: SeSpec<'_>,
    contrast: &GcompContrast<'_>,
    coefficients: &[f64],
    y: &[f64],
    nb_alpha: f64,
    target: &GcompTarget<'_>,
) -> Result<f64, EstimationError> {
    let GcompContrast { family, x_colmajor, nrows, ncols, .. } = *contrast;
    let (mut score_u, mut fisher_w) =
        glm_score_components(family, x_colmajor, nrows, ncols, coefficients, y, nb_alpha);
    if let Some(weights) = target.weights {
        // Weighted-population score `m_i u_i x_i` and information `m_i w_i x_i x_iᵀ`.
        for ((u, w), &m) in score_u.iter_mut().zip(fisher_w.iter_mut()).zip(weights) {
            *u *= m;
            *w *= m;
        }
    }
    let cov = sandwich_cov_matrix(spec, x_colmajor, nrows, ncols, &score_u, &fisher_w)?;
    let Some(cov) = cov else {
        return Ok(gcomp_delta_method_se(contrast, coefficients, 0.0, nb_alpha, target));
    };
    let grad = gcomp_gradient(contrast, coefficients, target);
    let mut quad = 0.0;
    for i in 0..ncols {
        for j in 0..ncols {
            quad += grad[i] * cov[i * ncols + j] * grad[j];
        }
    }
    Ok(quad.max(0.0).sqrt())
}

pub(super) fn sandwich_cov_matrix(
    spec: SeSpec<'_>,
    x: &[f64],
    nrows: usize,
    ncols: usize,
    score_u: &[f64],
    fisher_w: &[f64],
) -> Result<Option<Vec<f64>>, EstimationError> {
    crate::se::with_resolved_sandwich(spec, nrows, |kind| {
        let Some(kind) = kind else {
            return Ok(None);
        };
        Ok(Some(score_coefficient_covariance(x, nrows, ncols, score_u, fisher_w, kind)?))
    })
}

/// Per-row GLM score multiplier `u_i` and Fisher weight `w_i`.
///
/// Score contribution is `s_i = u_i x_i` with
/// `u_i = (y−μ) V(μ)⁻¹ μ'(η)`; `w_i` is the IRLS Fisher weight for bread `XᵀWX`.
/// Both clamps live on [`FamilyKernel`]; this only evaluates that kernel on each row.
pub(super) fn glm_score_components(
    family: GlmFamily,
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    coefficients: &[f64],
    y: &[f64],
    nb_alpha: f64,
) -> (Vec<f64>, Vec<f64>) {
    let mut score_u = vec![0.0; nrows];
    let mut fisher_w = vec![0.0; nrows];
    let kernel = FamilyKernel::new(family);
    let linear = linear_predictors(x_colmajor, nrows, ncols, coefficients);
    for (r, &eta) in linear.iter().enumerate() {
        let (u, w) = kernel.score_at(eta, y[r], nb_alpha);
        score_u[r] = u;
        fisher_w[r] = w;
    }
    (score_u, fisher_w)
}

/// Observed treatment labels and the population a g-computation average is taken over.
pub(super) struct GcompTarget<'a> {
    pub(super) treatment: &'a [f64],
    pub(super) population: &'a TargetPopulation,
    /// Sampling weight of every row (mean one); `None` for equal weights.
    pub(super) weights: Option<&'a [f64]>,
}

pub(super) fn gcomp_gradient(
    contrast: &GcompContrast<'_>,
    coefficients: &[f64],
    target: &GcompTarget<'_>,
) -> Vec<f64> {
    let GcompContrast { family, x_colmajor, nrows, ncols, t_col, active, control } = *contrast;
    let GcompTarget { treatment, population, weights } = *target;
    let mut grad = vec![0.0; ncols];
    let mut mass = 0.0;
    for r in 0..nrows {
        if !include_gcomp_row(population, treatment.get(r).copied().unwrap_or(0.0)) {
            continue;
        }
        let weight = weights.map_or(1.0, |w| w[r]);
        mass += weight;
        let (eta_active, eta_control) = contrast.contrast_row_etas(coefficients, r);
        let d1 = mean_derivative(family, eta_active);
        let d0 = mean_derivative(family, eta_control);
        for c in 0..ncols {
            let (x1, x0) = if c == t_col {
                (active, control)
            } else {
                let val = x_colmajor[c * nrows + r];
                (val, val)
            };
            grad[c] += weight * (d1 * x1 - d0 * x0);
        }
    }
    let n = if mass > 0.0 { mass } else { 1.0 };
    for g in &mut grad {
        *g /= n;
    }
    grad
}
