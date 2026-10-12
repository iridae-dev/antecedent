//! Split out of the mechanism registry. Behavior is unchanged.
use super::{
    data_rss, demean_by_group, fit_linear_gaussian, group_structure_usable, residual_sigma,
};

use std::sync::Arc;

use antecedent_stats::{
    FaerBackend, GlmDesignRef, GlmFamily, GlmOptions, LeastSquaresWorkspace, fit_glm_ridge,
};
#[cfg(feature = "gaussian-process")]
use antecedent_stats::{chol_log_det, chol_solve, cholesky_spd};

use super::super::cv::FitFrame;
use crate::compile::MechanismSlot;
use crate::error::ModelError;

/// Empirical-Bayes hierarchical linear: estimate τ² / λ from OLS; with a `UnitId`
/// column, the slopes come from the within-unit (fixed-effects) regression.
pub(super) fn fit_hierarchical_linear(
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    let ols = fit_linear_gaussian(frame, y, backend, ls_ws, 0.0)?;
    let MechanismSlot::LinearGaussian { coeffs: ols_coeffs, sigma: ols_sigma, .. } = ols else {
        return Err(ModelError::Unsupported { message: "hierarchical base fit failed".into() });
    };
    let p = ols_coeffs.len();
    // Method-of-moments EB: τ² ≈ mean(β̂²) − σ²·mean(diag((X'X)^{-1})) proxy;
    // use simplified τ² = mean(β̂²) clamped, λ = σ² / τ².
    let mean_b2 =
        if p == 0 { 0.0 } else { ols_coeffs.iter().map(|b| b * b).sum::<f64>() / p as f64 };
    let tau2 = (mean_b2 - ols_sigma * ols_sigma / n.max(1) as f64).max(1e-8);
    let mut lambda = (ols_sigma * ols_sigma / tau2).clamp(1e-6, 1e6);

    // Optional UnitId random intercept. The within estimator demeans the outcome *and*
    // every parent by unit: demeaning only `y` regresses it on a parent that still
    // carries its between-unit variance, attenuating the slope by
    // `Var_within(x)/Var(x)`. The between-unit spread of the outcome is not
    // discarded: the residual σ below is measured on the population-intercept mean
    // function, so it carries that spread, exactly as abduction (`y − mean`) and the
    // log-density will see it.
    let grouped = frame.groups.as_deref().filter(|g| group_structure_usable(g));
    let slot = if let Some(groups) = grouped {
        let y_work = demean_by_group(y, groups);
        let within = frame
            .with_parents(frame.parents.iter().map(|c| demean_by_group(&c[..n], groups)).collect());
        // Re-estimate λ on the demeaned series.
        let ols2 = fit_linear_gaussian(&within, &y_work, backend, ls_ws, 0.0)?;
        if let MechanismSlot::LinearGaussian { coeffs, sigma, .. } = ols2 {
            let mean_b2 = if coeffs.is_empty() {
                0.0
            } else {
                coeffs.iter().map(|b| b * b).sum::<f64>() / coeffs.len() as f64
            };
            let tau2 = (mean_b2 - sigma * sigma / n.max(1) as f64).max(1e-8);
            lambda = (sigma * sigma / tau2).clamp(1e-6, 1e6);
        }
        fit_linear_gaussian(&within, &y_work, backend, ls_ws, lambda)?
    } else {
        fit_linear_gaussian(frame, y, backend, ls_ws, lambda)?
    };
    let MechanismSlot::LinearGaussian { intercept, coeffs, sigma } = slot else {
        return Ok(slot);
    };
    let (intercept, sigma) = if grouped.is_some() {
        // Population intercept, then σ on the same mean function every downstream
        // consumer evaluates.
        let mean_y = y.iter().sum::<f64>() / n.max(1) as f64;
        let mut intercept = mean_y;
        for (c, col) in frame.parents.iter().enumerate() {
            intercept -= coeffs[c] * col[..n].iter().sum::<f64>() / n.max(1) as f64;
        }
        let rss = data_rss(frame, y, intercept, &coeffs, None);
        (intercept, residual_sigma(rss, n, 1 + p, y))
    } else {
        (intercept, sigma)
    };
    Ok(MechanismSlot::HierarchicalLinear { intercept, coeffs, sigma, shrinkage: lambda })
}

/// Hierarchical Bernoulli logit with always-on empirical-Bayes ridge.
///
/// EB λ is estimated from linear-probability OLS moments and passed to
/// [`fit_glm_ridge`], which applies the penalty on ordinary (non-separated) data
/// as well as separated cases (MM-014). Intercept is left unpenalized.
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
/// Hierarchical Bernoulli-logit fit.
pub(super) fn fit_hierarchical_glm(
    frame: &FitFrame<'_>,
    y: &[f64],
    backend: FaerBackend,
    ls_ws: &mut LeastSquaresWorkspace,
) -> Result<MechanismSlot, ModelError> {
    let n = y.len();
    #[allow(
        clippy::float_cmp,
        reason = "a binary outcome is coded exactly 0/1, so exact equality is the membership test"
    )]
    let binary = y.iter().all(|&yi| yi == 0.0 || yi == 1.0);
    if !binary {
        return Err(ModelError::Unsupported {
            message: "HierarchicalGlm requires binary {0,1} outcomes".into(),
        });
    }
    // EB λ from linear-probability OLS moments.
    let ols = fit_linear_gaussian(frame, y, backend, ls_ws, 0.0)?;
    let lambda = match &ols {
        MechanismSlot::LinearGaussian { coeffs, sigma, .. } => {
            let p = coeffs.len().max(1);
            let mean_b2 = coeffs.iter().map(|b| b * b).sum::<f64>() / p as f64;
            let tau2 = (mean_b2 - sigma * sigma / n.max(1) as f64).max(1e-8);
            (sigma * sigma / tau2).clamp(1e-4, 1e3)
        }
        _ => 1.0,
    };
    let p = frame.n_parents();
    let ncols = 1 + p;
    let mut x = vec![0.0; n * ncols];
    for r in 0..n {
        x[r] = 1.0;
    }
    for (pi, col) in frame.parents.iter().enumerate() {
        let base = (1 + pi) * n;
        x[base..base + n].copy_from_slice(&col[..n]);
    }
    let opts = GlmOptions::default();
    let fit = fit_glm_ridge(
        GlmFamily::BinomialLogit,
        GlmDesignRef { x_colmajor: &x, nrows: n, ncols, y },
        &backend,
        ls_ws,
        &opts,
        lambda,
    )
    .map_err(|e| ModelError::Numerical { message: e.to_string() })?;
    if !fit.converged {
        return Err(ModelError::Numerical {
            message: "hierarchical GLM logit did not converge".into(),
        });
    }
    // Encode as 2-category Discrete with baseline-category logits (cat0 = 0, cat1 = β).
    let mut logit_coeffs = vec![0.0; 2 * ncols];
    logit_coeffs[ncols..].copy_from_slice(&fit.coefficients[..ncols]);
    #[allow(
        clippy::float_cmp,
        reason = "a binary outcome is coded exactly 0/1, so exact equality is the membership test"
    )]
    let n1 = y.iter().filter(|&&yi| yi == 1.0).count() as f64;
    let p1 = n1 / n.max(1) as f64;
    Ok(MechanismSlot::Discrete {
        support: Arc::from([0.0, 1.0]),
        probs: Arc::from([1.0 - p1, p1]),
        logit_coeffs: Some(Arc::from(logit_coeffs)),
    })
}
