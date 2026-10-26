//! Poisson and negative-binomial (NB2) GLM fits.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::error::StatsError;
use crate::linalg::{DenseLinearAlgebra, FitDiagnostics, LeastSquaresWorkspace};
use super::{GlmDesignRef, GlmFit, GlmOptions, MEAN_FLOOR, NbAlphaPolicy, check_prior_weights, irls_fit_scaled, linear_predictor_into, validate_glm_shape};

/// Poisson deviance contribution of one row at linear predictor `eta` (log link); `+∞` when
/// the mean is not representable.
fn poisson_row_deviance(yi: f64, eta: f64) -> f64 {
    let mu = eta.exp().max(MEAN_FLOOR);
    if !mu.is_finite() {
        return f64::INFINITY;
    }
    if yi > 0.0 { 2.0 * (yi * (yi / mu).ln() - (yi - mu)) } else { 2.0 * mu }
}

/// NB2 deviance contribution of one row at `eta` with `θ = 1/α` (log link); `+∞` when the mean
/// is not representable.
fn negbin_row_deviance(yi: f64, eta: f64, theta: f64) -> f64 {
    let mu = eta.exp().max(MEAN_FLOOR);
    if !mu.is_finite() {
        return f64::INFINITY;
    }
    if yi > 0.0 {
        // `ln1p` keeps the large-`θ` (small `α`) limit accurate: the ratio is within rounding of 1
        // there and its plain logarithm would lose the digits the deviance is made of.
        2.0 * (yi * (yi / mu).ln() + (yi + theta) * ((mu - yi) / (yi + theta)).ln_1p())
    } else {
        2.0 * theta * (mu / theta).ln_1p()
    }
}

/// Prior-weighted deviance as a function of the linear predictor, `Σ mᵣ dᵣ(ηᵣ)`.
fn weighted_deviance<'a>(
    y: &'a [f64],
    prior: Option<&'a [f64]>,
    row: impl Fn(f64, f64) -> f64 + 'a,
) -> impl Fn(&[f64]) -> f64 + 'a {
    move |eta: &[f64]| {
        let mut total = 0.0;
        for (r, (&yi, &e)) in y.iter().zip(eta).enumerate() {
            let m = prior.map_or(1.0, |w| w[r]);
            if m > 0.0 {
                total += m * row(yi, e);
            }
        }
        total
    }
}

/// Poisson deviance at `β` (log link).
fn poisson_deviance_at(design: GlmDesignRef<'_>, beta: &[f64]) -> f64 {
    let GlmDesignRef { x_colmajor, nrows, ncols, y } = design;
    let mut eta = vec![0.0; nrows];
    linear_predictor_into(x_colmajor, nrows, ncols, beta, &mut eta);
    y.iter().zip(&eta).map(|(&yi, &e)| poisson_row_deviance(yi, e)).sum()
}

/// NB2 deviance at `β` with fixed dispersion `α` (log link).
fn negbin_deviance_at(design: GlmDesignRef<'_>, beta: &[f64], alpha: f64) -> f64 {
    let GlmDesignRef { x_colmajor, nrows, ncols, y } = design;
    let theta = 1.0 / alpha;
    let mut eta = vec![0.0; nrows];
    linear_predictor_into(x_colmajor, nrows, ncols, beta, &mut eta);
    y.iter().zip(&eta).map(|(&yi, &e)| negbin_row_deviance(yi, e, theta)).sum()
}

pub(super) fn fit_poisson(
    design: GlmDesignRef<'_>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<GlmFit, StatsError> {
    validate_glm_shape(design)?;
    let GlmDesignRef { ncols, y, .. } = design;
    for &yi in y {
        if !(yi.is_finite() && yi >= 0.0) {
            return Err(StatsError::Shape {
                message: "Poisson GLM requires non-negative outcomes",
            });
        }
    }

    let objective = weighted_deviance(y, None, poisson_row_deviance);
    let (beta, iterations, converged) = irls_fit_scaled(
        design,
        backend,
        workspace,
        options,
        true,
        None,
        Some(&objective),
        |eta, yi| {
            let mu = eta.exp().max(MEAN_FLOOR);
            let w = mu.sqrt();
            (w, (eta + (yi - mu) / mu) * w)
        },
    )?;

    // Diagnostics at the returned coefficients (not the pre-update IRLS iterate).
    let deviance = poisson_deviance_at(design, &beta);

    Ok(GlmFit {
        coefficients: beta,
        iterations,
        converged,
        separated: false,
        boundary_saturated: false,
        boundary_margin: None,
        boundary_count: None,
        penalized: false,
        deviance,
        nb_alpha: None,
        diagnostics: FitDiagnostics::new(ncols, None, "glm-irls", workspace.grow_count),
    })
}

pub(super) fn fit_poisson_weighted(
    design: GlmDesignRef<'_>,
    weights: &[f64],
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<GlmFit, StatsError> {
    check_prior_weights(design, weights)?;
    let GlmDesignRef { ncols, y, .. } = design;
    if y.iter().any(|&yi| !(yi.is_finite() && yi >= 0.0)) {
        return Err(StatsError::Shape { message: "Poisson GLM requires non-negative outcomes" });
    }
    let root: Vec<f64> = weights.iter().map(|w| w.sqrt()).collect();
    let objective = weighted_deviance(y, Some(weights), poisson_row_deviance);
    let (beta, iterations, converged) = irls_fit_scaled(
        design,
        backend,
        workspace,
        options,
        true,
        Some(&root),
        Some(&objective),
        |eta, yi| {
            let mu = eta.exp().max(MEAN_FLOOR);
            let w = mu.sqrt();
            (w, (eta + (yi - mu) / mu) * w)
        },
    )?;
    let deviance = poisson_deviance_at(design, &beta);
    Ok(GlmFit {
        coefficients: beta,
        iterations,
        converged,
        separated: false,
        boundary_saturated: false,
        boundary_margin: None,
        boundary_count: None,
        penalized: false,
        deviance,
        nb_alpha: None,
        diagnostics: FitDiagnostics::new(ncols, None, "glm-irls", workspace.grow_count),
    })
}

/// NB2 IRLS with fixed, `MoM`, or nested-MLE dispersion α (`Var = μ + α μ²`).
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
pub(super) fn fit_negbin(
    design: GlmDesignRef<'_>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<GlmFit, StatsError> {
    fit_negbin_prior(design, None, backend, workspace, options)
}

/// NB2 fit under optional nonnegative prior (case) weights. With weights the mean is the
/// prior-weighted IRLS fit and the dispersion is updated from the weighted data: the weighted
/// Pearson moment `α = max(Σ m_i e_i² − (Σ m_i − p), 0) / Σ m_i μ_i` with
/// `e_i = (y_i − μ_i)/√μ_i`, or the weighted profile likelihood `Σ m_i ℓ_i(θ)` maximized over
/// `θ = 1/α`, per the dispersion policy. Unit weights are the unweighted fit.
#[allow(
    clippy::needless_range_loop,
    clippy::too_many_lines,
    reason = "index-based loops over parallel arrays; one routine for the unweighted and weighted dispersion policies"
)]
pub(super) fn fit_negbin_prior(
    design: GlmDesignRef<'_>,
    prior: Option<&[f64]>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
) -> Result<GlmFit, StatsError> {
    validate_glm_shape(design)?;
    let share = |r: usize| prior.map_or(1.0, |m| m[r]);
    let GlmDesignRef { x_colmajor, nrows, ncols, y } = design;
    for &yi in y {
        if !(yi.is_finite() && yi >= 0.0) {
            return Err(StatsError::Shape {
                message: "negative-binomial GLM requires non-negative outcomes",
            });
        }
    }

    let mut mom_alpha = || -> Result<f64, StatsError> {
        let poisson = match prior {
            Some(weights) => fit_poisson_weighted(design, weights, backend, workspace, options)?,
            None => fit_poisson(design, backend, workspace, options)?,
        };
        let mut pearson_ss = 0.0;
        let mut mass = 0.0;
        let mut sum_mu = 0.0;
        let mut eta = vec![0.0; nrows];
        linear_predictor_into(x_colmajor, nrows, ncols, &poisson.coefficients, &mut eta);
        for r in 0..nrows {
            let mu = eta[r].exp().max(MEAN_FLOOR);
            let e = (y[r] - mu) / mu.sqrt();
            pearson_ss += share(r) * e * e;
            sum_mu += share(r) * mu;
            mass += share(r);
        }
        let df = (mass - ncols as f64).max(1.0);
        let excess = (pearson_ss - df).max(0.0);
        Ok((excess / sum_mu.max(MEAN_FLOOR)).max(1e-8))
    };

    match options.nb_alpha {
        NbAlphaPolicy::Fixed(a) => {
            if !(a.is_finite() && a > 0.0) {
                return Err(StatsError::Shape { message: "nb_alpha Fixed must be finite and > 0" });
            }
            fit_negbin_fixed_alpha(design, prior, backend, workspace, options, a)
        }
        NbAlphaPolicy::MethodOfMoments => {
            let alpha = mom_alpha()?;
            fit_negbin_fixed_alpha(design, prior, backend, workspace, options, alpha)
        }
        NbAlphaPolicy::NestedMle { max_outer, tol_alpha } => {
            let mut alpha = mom_alpha()?;
            let mut fit =
                fit_negbin_fixed_alpha(design, prior, backend, workspace, options, alpha)?;
            let mut outer_ok = false;
            let mut mu = vec![0.0; nrows];
            for _ in 0..max_outer.max(1) {
                linear_predictor_into(x_colmajor, nrows, ncols, &fit.coefficients, &mut mu);
                for eta_r in &mut mu {
                    *eta_r = eta_r.exp().max(MEAN_FLOOR);
                }
                let alpha_new = mle_nb_alpha(y, &mu, prior, alpha);
                let beta_prev = fit.coefficients.clone();
                fit =
                    fit_negbin_fixed_alpha(design, prior, backend, workspace, options, alpha_new)?;
                let mut max_db = 0.0_f64;
                for c in 0..ncols {
                    max_db = max_db.max((fit.coefficients[c] - beta_prev[c]).abs());
                }
                let da = (alpha_new - alpha).abs();
                alpha = alpha_new;
                if da < tol_alpha && (fit.converged || max_db < options.tol) {
                    outer_ok = true;
                    break;
                }
            }
            fit.converged = fit.converged && outer_ok;
            fit.nb_alpha = Some(alpha);
            Ok(fit)
        }
    }
}

/// 1-D MLE for NB2 α given fitted means (digamma/trigamma Newton on θ=1/α, safeguarded).
fn mle_nb_alpha(y: &[f64], mu: &[f64], prior: Option<&[f64]>, alpha0: f64) -> f64 {
    use crate::special::{digamma, trigamma};

    // Work in θ = 1/α (more stable); clamp to a wide but finite range.
    let mut theta = (1.0 / alpha0.max(1e-8)).clamp(1e-4, 1e6);
    for _ in 0..80 {
        let mut score = 0.0;
        let mut hess = 0.0;
        for (r, (&yi, &mui)) in y.iter().zip(mu.iter()).enumerate() {
            let m = prior.map_or(1.0, |w| w[r]);
            let d_theta = digamma(yi + theta) - digamma(theta)
                + (theta / (theta + mui)).ln()
                + (mui - yi) / (theta + mui);
            let d2_theta = trigamma(yi + theta) - trigamma(theta) + 1.0 / theta
                - 1.0 / (theta + mui)
                - (mui - yi) / ((theta + mui) * (theta + mui));
            score += m * d_theta;
            hess += m * d2_theta;
        }
        if !score.is_finite() {
            break;
        }
        // Maximize ⇒ Newton uses −score/hess when hess < 0; otherwise gradient ascent.
        let step = if hess < -1e-12 {
            -score / hess
        } else if score.abs() > 1e-12 {
            0.1 * score.signum() * theta
        } else {
            0.0
        };
        let mut theta_new = (theta + step).clamp(1e-4, 1e6);
        // Damp large relative steps.
        if (theta_new - theta).abs() > 0.5 * theta {
            theta_new = theta + 0.25 * (theta_new - theta);
            theta_new = theta_new.clamp(1e-4, 1e6);
        }
        if (theta_new - theta).abs() < 1e-10 * (1.0 + theta) || score.abs() < 1e-10 {
            theta = theta_new;
            break;
        }
        theta = theta_new;
    }
    (1.0 / theta).clamp(1e-8, 1e4)
}

fn fit_negbin_fixed_alpha(
    design: GlmDesignRef<'_>,
    prior: Option<&[f64]>,
    backend: &impl DenseLinearAlgebra,
    workspace: &mut LeastSquaresWorkspace,
    options: &GlmOptions,
    alpha: f64,
) -> Result<GlmFit, StatsError> {
    let ncols = design.ncols;
    let root: Option<Vec<f64>> = prior.map(|w| w.iter().map(|m| m.sqrt()).collect());
    let theta = 1.0 / alpha;
    let objective =
        weighted_deviance(design.y, prior, move |yi, eta| negbin_row_deviance(yi, eta, theta));
    let (beta, iterations, converged) = irls_fit_scaled(
        design,
        backend,
        workspace,
        options,
        true,
        root.as_deref(),
        Some(&objective),
        |eta, yi| {
            let mu = eta.exp().max(MEAN_FLOOR);
            let var = mu * (1.0 + alpha * mu);
            let w = (mu * mu / var).sqrt();
            (w, (eta + (yi - mu) / mu) * w)
        },
    )?;

    // Diagnostics at the returned coefficients (not the pre-update IRLS iterate).
    let deviance = negbin_deviance_at(design, &beta, alpha);

    Ok(GlmFit {
        coefficients: beta,
        iterations,
        converged,
        separated: false,
        boundary_saturated: false,
        boundary_margin: None,
        boundary_count: None,
        penalized: false,
        deviance,
        nb_alpha: Some(alpha),
        diagnostics: FitDiagnostics::new(ncols, None, "glm-irls", workspace.grow_count),
    })
}
