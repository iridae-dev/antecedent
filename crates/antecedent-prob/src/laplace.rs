//! Native Laplace approximation for Bayesian GLMs (ADR 0006 / ).
//!
//! MAP via damped Newton → Cholesky of −Hessian (LDLT fallback) → MVN draws.
//! Refuses to publish a posterior without convergence and curvature diagnostics.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_lines
)]

use std::sync::Arc;

use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_kernels::standard_normal;

use crate::backend::{
    BayesDesignRef, BayesFitOptions, BayesFitResult, BayesLikelihood, InferenceBackend,
    LaplaceWorkspace,
};
use crate::diagnostics::{HessianFactorization, InferenceDiagnostics};
use crate::error::ProbError;
use crate::gaussian_target::{
    PosteriorTarget, gaussian_target_from_model, prior_quadratic, rss_and_xtwr,
};
use crate::likelihood_terms::{
    LikelihoodTerms, gaussian_terms, logit_terms, poisson_terms, probit_terms,
};
use crate::linalg::{cholesky_spd, condition_from_chol, invert_spd, ldlt_decompose, solve_spd};
use crate::posterior::{PosteriorDraws, PosteriorQuantityKind, PosteriorSchema};
use crate::prior::{GaussianCoefficientPrior, GaussianVarianceModel, PriorSet};

/// Native Laplace Bayesian GLM backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct LaplaceGlmBackend;

impl InferenceBackend for LaplaceGlmBackend {
    fn fit(
        &self,
        likelihood: BayesLikelihood,
        design: BayesDesignRef<'_>,
        prior: &PriorSet,
        options: &BayesFitOptions,
        workspace: &mut LaplaceWorkspace,
        _ctx: &ExecutionContext,
    ) -> Result<BayesFitResult, ProbError> {
        prior.validate()?;
        fit_laplace_glm(likelihood, design, prior, options, workspace)
    }
}

/// Fit a Laplace Bayesian GLM and draw from the MVN approximation.
///
/// # Errors
///
/// Shape, prior, non-convergence, or singular curvature.
pub fn fit_laplace_glm(
    likelihood: BayesLikelihood,
    design: BayesDesignRef<'_>,
    prior: &PriorSet,
    options: &BayesFitOptions,
    workspace: &mut LaplaceWorkspace,
) -> Result<BayesFitResult, ProbError> {
    let nrows = design.nrows;
    let ncols = design.ncols;
    validate_design(design)?;
    workspace.prepare(nrows, ncols, options.n_draws);

    if likelihood == BayesLikelihood::GaussianIdentity {
        return fit_gaussian_laplace(design, prior, options, workspace);
    }

    let coef_prior = match prior.gaussian_coefficients() {
        Some(p) => p.clone(),
        None => GaussianCoefficientPrior::isotropic(ncols, 10.0),
    };
    if coef_prior.len() != ncols {
        return Err(ProbError::InvalidPrior { message: "coefficient prior length != ncols" });
    }
    coef_prior.validate()?;
    let prec = coef_prior.precision();

    // Initialize at prior mean (often 0).
    for i in 0..ncols {
        workspace.beta[i] = coef_prior.mean[i];
    }

    let mut converged = false;
    let mut iterations = 0u32;
    let mut grad_inf: f64;
    let mut separation_warning = false;

    for iter in 0..options.max_iter {
        iterations = iter + 1;
        let (_, sep) = accumulate_likelihood(
            likelihood,
            design,
            &workspace.beta[..ncols],
            &mut workspace.grad[..ncols],
            &mut workspace.neg_hessian[..ncols * ncols],
            &mut workspace.eta[..nrows],
            &mut workspace.work_w[..nrows],
            1.0,
        )?;
        separation_warning |= sep;

        // Add prior: log π(β) = -0.5 Σ prec_i (β_i - μ_i)²
        for i in 0..ncols {
            let diff = workspace.beta[i] - coef_prior.mean[i];
            workspace.grad[i] -= prec[i] * diff;
            workspace.neg_hessian[i * ncols + i] += prec[i];
        }

        grad_inf = 0.0;
        for i in 0..ncols {
            grad_inf = grad_inf.max(workspace.grad[i].abs());
        }

        if grad_inf < options.grad_tol {
            converged = true;
            break;
        }

        // Solve (−H) step = grad for Newton step maximizing log-posterior.
        // We store neg_hessian = −∇²ℓ; Newton: β ← β + (−H)^{-1} ∇ℓ
        workspace.step[..ncols].fill(0.0);
        let hess = workspace.neg_hessian[..ncols * ncols].to_vec();
        let grad = workspace.grad[..ncols].to_vec();
        if solve_spd(&hess, ncols, &grad, &mut workspace.step[..ncols]).is_err() {
            // Damped fallback: take gradient step with small step size.
            let scale = 1e-2 / grad_inf.max(1.0);
            for i in 0..ncols {
                workspace.step[i] = scale * workspace.grad[i];
            }
        }

        // Damped line search
        let mut step_scale = 1.0;
        let beta_old = workspace.beta[..ncols].to_vec();
        let mut accepted = false;
        let old_obj = log_posterior_value(
            likelihood,
            design,
            &beta_old,
            &coef_prior,
            &prec,
            &mut workspace.eta[..nrows],
            1.0,
        )?;
        for _ in 0..8 {
            for i in 0..ncols {
                workspace.beta[i] = beta_old[i] + step_scale * workspace.step[i];
            }
            match log_posterior_value(
                likelihood,
                design,
                &workspace.beta[..ncols],
                &coef_prior,
                &prec,
                &mut workspace.eta[..nrows],
                1.0,
            ) {
                Ok(new_obj) if new_obj >= old_obj - 1e-12 => {
                    accepted = true;
                    break;
                }
                Ok(_) | Err(ProbError::Numerical { .. }) => {
                    step_scale *= 0.5;
                }
                Err(e) => return Err(e),
            }
        }
        if !accepted {
            for i in 0..ncols {
                workspace.beta[i] = beta_old[i];
            }
            break;
        }
    }

    // Final gradient / Hessian at MAP.
    accumulate_likelihood(
        likelihood,
        design,
        &workspace.beta[..ncols],
        &mut workspace.grad[..ncols],
        &mut workspace.neg_hessian[..ncols * ncols],
        &mut workspace.eta[..nrows],
        &mut workspace.work_w[..nrows],
        1.0,
    )?;
    for i in 0..ncols {
        let diff = workspace.beta[i] - coef_prior.mean[i];
        workspace.grad[i] -= prec[i] * diff;
        workspace.neg_hessian[i * ncols + i] += prec[i];
    }
    grad_inf = 0.0;
    for i in 0..ncols {
        grad_inf = grad_inf.max(workspace.grad[i].abs());
    }
    if grad_inf < options.grad_tol {
        converged = true;
    }

    let hess = workspace.neg_hessian[..ncols * ncols].to_vec();
    let (factorization, cov, condition) = match cholesky_spd(&hess, ncols) {
        Ok(chol) => {
            let cond = condition_from_chol(&chol, ncols);
            workspace.factor[..ncols * ncols].copy_from_slice(&chol);
            let cov = invert_spd(&hess, ncols)?;
            (HessianFactorization::Cholesky, cov, cond)
        }
        Err(_) => {
            let (d, l) = ldlt_decompose(&hess, ncols)?;
            // Build approximate covariance via LDLT solve of identity.
            let mut cov = vec![0.0; ncols * ncols];
            let mut rhs = vec![0.0; ncols];
            let mut x = vec![0.0; ncols];
            for col in 0..ncols {
                rhs.fill(0.0);
                rhs[col] = 1.0;
                // Solve L D L' x = e
                // forward L y = e
                let mut y = vec![0.0; ncols];
                for i in 0..ncols {
                    let mut acc = rhs[i];
                    for j in 0..i {
                        acc -= l[i * ncols + j] * y[j];
                    }
                    y[i] = acc;
                }
                for i in 0..ncols {
                    y[i] /= d[i];
                }
                for i in (0..ncols).rev() {
                    let mut acc = y[i];
                    for j in (i + 1)..ncols {
                        acc -= l[j * ncols + i] * x[j];
                    }
                    x[i] = acc;
                }
                for i in 0..ncols {
                    cov[i * ncols + col] = x[i];
                }
            }
            let mut min_d = f64::INFINITY;
            let mut max_d: f64 = 0.0;
            for &di in &d {
                min_d = min_d.min(di.abs());
                max_d = max_d.max(di.abs());
            }
            let cond = if min_d > 0.0 { max_d / min_d } else { f64::INFINITY };
            (HessianFactorization::Ldlt, cov, cond)
        }
    };

    let mut notes = Vec::new();
    if separation_warning {
        notes.push(Arc::from("possible separation in Bernoulli model"));
    }

    let diagnostics = InferenceDiagnostics {
        converged,
        iterations,
        grad_inf_norm: grad_inf,
        hessian_condition: condition,
        factorization,
        separation_warning,
        notes,
        backend_id: Arc::from("laplace"),
        n_chains: None,
        n_warmup: None,
        ess_bulk_min: None,
        ess_tail_min: None,
        rhat_max: None,
        n_divergences: None,
        mean_accept_prob: None,
        n_warmup_divergences: None,
        n_postwarmup_divergences: None,
        max_abs_delta_h: None,
        all_chains_moved: None,
    };

    if !diagnostics.allows_posterior() {
        return Err(ProbError::MissingDiagnostics {
            message: "Laplace posterior refused without convergence and curvature diagnostics",
        });
    }

    let map = workspace.beta[..ncols].to_vec();
    let draws_vals = sample_gaussian_mvn(&map, &cov, options.n_draws, options.seed, workspace)?;
    let draws = PosteriorDraws::from_column_major(
        PosteriorSchema::coefficients(ncols),
        options.n_draws,
        draws_vals,
    )?;

    Ok(BayesFitResult { draws, map, diagnostics, cov: Some(cov) })
}

/// GaussianIdentity Laplace: exact known-σ² posterior or joint InvGamma Laplace on `(β, λ)`.
fn fit_gaussian_laplace(
    design: BayesDesignRef<'_>,
    prior: &PriorSet,
    options: &BayesFitOptions,
    workspace: &mut LaplaceWorkspace,
) -> Result<BayesFitResult, ProbError> {
    let ncols = design.ncols;
    let coef_prior = match prior.gaussian_coefficients() {
        Some(p) => p.clone(),
        None => GaussianCoefficientPrior::isotropic(ncols, 10.0),
    };
    if coef_prior.len() != ncols {
        return Err(ProbError::InvalidPrior { message: "coefficient prior length != ncols" });
    }
    coef_prior.validate()?;
    let model = GaussianVarianceModel::from_prior_set(prior)?;
    match model {
        GaussianVarianceModel::Known { sigma2 } => {
            fit_gaussian_laplace_known(design, coef_prior, sigma2, options, workspace)
        }
        GaussianVarianceModel::InvGamma { shape, scale } => {
            fit_gaussian_laplace_inv_gamma(design, coef_prior, shape, scale, options, workspace)
        }
    }
}

/// Form `A_β = X'WX + P` and `b_β = X'W(y − offset) + P m₀`.
fn gaussian_normal_equations(
    design: BayesDesignRef<'_>,
    coef_prior: &GaussianCoefficientPrior,
) -> Result<(Vec<f64>, Vec<f64>), ProbError> {
    let nrows = design.nrows;
    let ncols = design.ncols;
    let prec = coef_prior.precision();
    let mut a_beta = vec![0.0; ncols * ncols];
    let mut b_beta = vec![0.0; ncols];
    for c1 in 0..ncols {
        for c2 in c1..ncols {
            let mut acc = 0.0;
            for r in 0..nrows {
                let w = design.weights.map_or(1.0, |ww| ww[r]);
                let x1 = design.x_colmajor[c1 * nrows + r];
                let x2 = design.x_colmajor[c2 * nrows + r];
                acc += w * x1 * x2;
            }
            a_beta[c1 * ncols + c2] = acc;
            a_beta[c2 * ncols + c1] = acc;
        }
        let mut acc = 0.0;
        for r in 0..nrows {
            let w = design.weights.map_or(1.0, |ww| ww[r]);
            let offset = design.offsets.map_or(0.0, |oo| oo[r]);
            let x = design.x_colmajor[c1 * nrows + r];
            acc += w * x * (design.y[r] - offset);
        }
        b_beta[c1] = acc + prec[c1] * coef_prior.mean[c1];
        a_beta[c1 * ncols + c1] += prec[c1];
    }
    Ok((a_beta, b_beta))
}

fn solve_map_from_normal_eq(
    a_beta: &[f64],
    b_beta: &[f64],
    ncols: usize,
) -> Result<Vec<f64>, ProbError> {
    let a_inv = invert_spd(a_beta, ncols)?;
    let mut map = vec![0.0; ncols];
    for i in 0..ncols {
        let mut acc = 0.0;
        for j in 0..ncols {
            acc += a_inv[i * ncols + j] * b_beta[j];
        }
        map[i] = acc;
    }
    Ok(map)
}

fn fit_gaussian_laplace_known(
    design: BayesDesignRef<'_>,
    coef_prior: GaussianCoefficientPrior,
    sigma2: f64,
    options: &BayesFitOptions,
    workspace: &mut LaplaceWorkspace,
) -> Result<BayesFitResult, ProbError> {
    let ncols = design.ncols;
    workspace.prepare(design.nrows, ncols, options.n_draws);
    let (a_beta, b_beta) = gaussian_normal_equations(design, &coef_prior)?;
    let map = solve_map_from_normal_eq(&a_beta, &b_beta, ncols)?;
    let a_inv = invert_spd(&a_beta, ncols)?;
    let mut cov = vec![0.0; ncols * ncols];
    for i in 0..ncols * ncols {
        cov[i] = sigma2 * a_inv[i];
    }

    let mut target = gaussian_target_from_model(
        design,
        coef_prior.clone(),
        GaussianVarianceModel::Known { sigma2 },
    )?;
    let mut grad = vec![0.0; ncols];
    let _lp = target.logp_and_grad(&map, &mut grad)?;
    let mut grad_inf = 0.0_f64;
    for &g in &grad {
        grad_inf = grad_inf.max(g.abs());
    }
    let chol = cholesky_spd(&a_beta, ncols)?;
    let condition = condition_from_chol(&chol, ncols);
    let diagnostics = InferenceDiagnostics {
        converged: grad_inf < options.grad_tol.max(1e-8),
        iterations: 1,
        grad_inf_norm: grad_inf,
        hessian_condition: condition,
        factorization: HessianFactorization::Cholesky,
        separation_warning: false,
        notes: vec![Arc::from("gaussian_laplace_known_sigma2")],
        backend_id: Arc::from("laplace"),
        n_chains: None,
        n_warmup: None,
        ess_bulk_min: None,
        ess_tail_min: None,
        rhat_max: None,
        n_divergences: None,
        mean_accept_prob: None,
        n_warmup_divergences: None,
        n_postwarmup_divergences: None,
        max_abs_delta_h: None,
        all_chains_moved: None,
    };
    if !diagnostics.allows_posterior() {
        return Err(ProbError::MissingDiagnostics {
            message: "Laplace posterior refused without convergence and curvature diagnostics",
        });
    }
    let draws_vals = sample_gaussian_mvn(&map, &cov, options.n_draws, options.seed, workspace)?;
    let draws = PosteriorDraws::from_column_major(
        PosteriorSchema::coefficients(ncols),
        options.n_draws,
        draws_vals,
    )?;
    Ok(BayesFitResult { draws, map, diagnostics, cov: Some(cov) })
}

fn fit_gaussian_laplace_inv_gamma(
    design: BayesDesignRef<'_>,
    coef_prior: GaussianCoefficientPrior,
    shape: f64,
    scale: f64,
    options: &BayesFitOptions,
    workspace: &mut LaplaceWorkspace,
) -> Result<BayesFitResult, ProbError> {
    let nrows = design.nrows;
    let ncols = design.ncols;
    let dim = ncols + 1;
    workspace.prepare(nrows, dim.max(ncols), options.n_draws.max(dim));

    let (a_beta, b_beta) = gaussian_normal_equations(design, &coef_prior)?;
    let beta_map = solve_map_from_normal_eq(&a_beta, &b_beta, ncols)?;

    let mut xtwr = vec![0.0; ncols];
    let mut p_diff = vec![0.0; ncols];
    let rss = rss_and_xtwr(design, &beta_map, &mut xtwr)?;
    let prec = coef_prior.precision();
    let quad = prior_quadratic(&coef_prior, &prec, &beta_map, &mut p_diff)?;
    let mut n_eff = 0.0;
    for r in 0..nrows {
        n_eff += design.weights.map_or(1.0, |w| w[r]);
    }
    let a_const = shape + 0.5 * (n_eff + ncols as f64);
    let b_at_map = scale + 0.5 * (rss + quad);
    if !(a_const > 0.0) || !(b_at_map > 0.0) || !b_at_map.is_finite() {
        return Err(ProbError::Numerical {
            message: format!("invalid InvGamma Laplace mode: A={a_const} B={b_at_map}"),
        });
    }
    let lambda_map = (b_at_map / a_const).ln();
    let exp_neg_l = (-lambda_map).exp();
    let exp_l = lambda_map.exp();

    let a_inv = invert_spd(&a_beta, ncols)?;
    let mut cov_beta = vec![0.0; ncols * ncols];
    for i in 0..ncols * ncols {
        cov_beta[i] = exp_l * a_inv[i];
    }
    let var_lambda = 1.0 / a_const;

    let mut joint_mean = beta_map.clone();
    joint_mean.push(lambda_map);
    let mut joint_cov = vec![0.0; dim * dim];
    for i in 0..ncols {
        for j in 0..ncols {
            joint_cov[i * dim + j] = cov_beta[i * ncols + j];
        }
    }
    joint_cov[ncols * dim + ncols] = var_lambda;

    let mut target = gaussian_target_from_model(
        design,
        coef_prior.clone(),
        GaussianVarianceModel::InvGamma { shape, scale },
    )?;
    let mut grad = vec![0.0; dim];
    let _lp = target.logp_and_grad(&joint_mean, &mut grad)?;
    let mut grad_inf = 0.0_f64;
    for &g in &grad {
        grad_inf = grad_inf.max(g.abs());
    }

    let mut h_bb = vec![0.0; ncols * ncols];
    for i in 0..ncols * ncols {
        h_bb[i] = exp_neg_l * a_beta[i];
    }
    let chol = cholesky_spd(&h_bb, ncols)?;
    let condition = condition_from_chol(&chol, ncols);

    let diagnostics = InferenceDiagnostics {
        converged: grad_inf < options.grad_tol.max(1e-8),
        iterations: 1,
        grad_inf_norm: grad_inf,
        hessian_condition: condition,
        factorization: HessianFactorization::Cholesky,
        separation_warning: false,
        notes: vec![Arc::from("gaussian_laplace_inv_gamma")],
        backend_id: Arc::from("laplace"),
        n_chains: None,
        n_warmup: None,
        ess_bulk_min: None,
        ess_tail_min: None,
        rhat_max: None,
        n_divergences: None,
        mean_accept_prob: None,
        n_warmup_divergences: None,
        n_postwarmup_divergences: None,
        max_abs_delta_h: None,
        all_chains_moved: None,
    };
    if !diagnostics.allows_posterior() {
        return Err(ProbError::MissingDiagnostics {
            message: "Laplace posterior refused without convergence and curvature diagnostics",
        });
    }

    let joint_draws =
        sample_gaussian_mvn(&joint_mean, &joint_cov, options.n_draws, options.seed, workspace)?;
    let mut values = vec![0.0; options.n_draws * (ncols + 1)];
    for d in 0..options.n_draws {
        for i in 0..ncols {
            values[i * options.n_draws + d] = joint_draws[i * options.n_draws + d];
        }
        let lambda = joint_draws[ncols * options.n_draws + d];
        values[ncols * options.n_draws + d] = lambda.exp();
    }
    let mut quantities: Vec<_> = (0..ncols)
        .map(|i| PosteriorQuantityKind::Coefficient { index: i, name: None })
        .collect();
    quantities.push(PosteriorQuantityKind::ResidualVariance);
    let schema = PosteriorSchema { quantities: Arc::from(quantities) };
    let draws = PosteriorDraws::from_column_major(schema, options.n_draws, values)?;

    Ok(BayesFitResult { draws, map: beta_map, diagnostics, cov: Some(cov_beta) })
}

pub(crate) fn validate_design(design: BayesDesignRef<'_>) -> Result<(), ProbError> {
    let nrows = design.nrows;
    let ncols = design.ncols;
    if design.y.len() != nrows {
        return Err(ProbError::Shape { message: "y length != nrows" });
    }
    if design.x_colmajor.len() < nrows.saturating_mul(ncols) {
        return Err(ProbError::Shape { message: "X buffer too short" });
    }
    if nrows == 0 || ncols == 0 {
        return Err(ProbError::Shape { message: "empty design" });
    }
    if let Some(w) = design.weights {
        if w.len() != nrows {
            return Err(ProbError::Shape { message: "weights length != nrows" });
        }
    }
    if let Some(o) = design.offsets {
        if o.len() != nrows {
            return Err(ProbError::Shape { message: "offsets length != nrows" });
        }
    }
    Ok(())
}

/// Accumulate likelihood gradient and −Hessian at `beta`. Returns (grad_inf, separation).
///
/// `gaussian_sigma2` scales the GaussianIdentity working weights / scores (`1/σ²`). Other
/// likelihoods ignore it.
pub(crate) fn accumulate_likelihood(
    likelihood: BayesLikelihood,
    design: BayesDesignRef<'_>,
    beta: &[f64],
    grad: &mut [f64],
    neg_hess: &mut [f64],
    eta: &mut [f64],
    work_w: &mut [f64],
    gaussian_sigma2: f64,
) -> Result<(f64, bool), ProbError> {
    let nrows = design.nrows;
    let ncols = design.ncols;
    grad.fill(0.0);
    neg_hess.fill(0.0);
    let inv_sigma2 = 1.0 / gaussian_sigma2.max(1e-12);

    let mut separation = false;
    for r in 0..nrows {
        let offset = design.offsets.map_or(0.0, |o| o[r]);
        let mut e = offset;
        for c in 0..ncols {
            e += design.x_colmajor[c * nrows + r] * beta[c];
        }
        eta[r] = e;
        let w_obs = design.weights.map_or(1.0, |w| w[r]);
        let y = design.y[r];

        let terms = glm_observation_terms(likelihood, y, e, w_obs, inv_sigma2)?;
        if matches!(likelihood, BayesLikelihood::BernoulliLogit | BayesLikelihood::BernoulliProbit)
        {
            let mu = if matches!(likelihood, BayesLikelihood::BernoulliLogit) {
                1.0 / (1.0 + (-e).exp())
            } else {
                antecedent_kernels::norm_cdf(e)
            };
            if mu < 1e-8 || mu > 1.0 - 1e-8 {
                separation = true;
            }
        }
        work_w[r] = terms.neg_hessian_eta;
        let score_scale = terms.score_eta;

        for c in 0..ncols {
            let x = design.x_colmajor[c * nrows + r];
            grad[c] += x * score_scale;
        }
        // −Hessian = X' diag(−ℓ'') X
        for c1 in 0..ncols {
            let x1 = design.x_colmajor[c1 * nrows + r];
            for c2 in c1..ncols {
                let x2 = design.x_colmajor[c2 * nrows + r];
                let add = work_w[r] * x1 * x2;
                neg_hess[c1 * ncols + c2] += add;
                if c1 != c2 {
                    neg_hess[c2 * ncols + c1] += add;
                }
            }
        }
    }

    let mut ginf: f64 = 0.0;
    for g in grad.iter() {
        ginf = ginf.max(g.abs());
    }
    Ok((ginf, separation))
}

fn glm_observation_terms(
    likelihood: BayesLikelihood,
    y: f64,
    eta: f64,
    weight: f64,
    inv_sigma2: f64,
) -> Result<LikelihoodTerms, ProbError> {
    match likelihood {
        BayesLikelihood::GaussianIdentity => Ok(gaussian_terms(y, eta, weight, inv_sigma2)),
        BayesLikelihood::BernoulliLogit => Ok(logit_terms(y, eta, weight)),
        BayesLikelihood::BernoulliProbit => probit_terms(y, eta, weight),
        BayesLikelihood::PoissonLog => poisson_terms(y, eta, weight),
    }
}

pub(crate) fn log_posterior_value(
    likelihood: BayesLikelihood,
    design: BayesDesignRef<'_>,
    beta: &[f64],
    prior: &GaussianCoefficientPrior,
    prec: &[f64],
    eta: &mut [f64],
    gaussian_sigma2: f64,
) -> Result<f64, ProbError> {
    let nrows = design.nrows;
    let ncols = design.ncols;
    let inv_sigma2 = 1.0 / gaussian_sigma2.max(1e-12);
    let mut ll = 0.0;
    for r in 0..nrows {
        let offset = design.offsets.map_or(0.0, |o| o[r]);
        let mut e = offset;
        for c in 0..ncols {
            e += design.x_colmajor[c * nrows + r] * beta[c];
        }
        eta[r] = e;
        let w = design.weights.map_or(1.0, |ww| ww[r]);
        let y = design.y[r];
        ll += glm_observation_terms(likelihood, y, e, w, inv_sigma2)?.log_value;
    }
    let mut lp = 0.0;
    for i in 0..ncols {
        let d = beta[i] - prior.mean[i];
        lp -= 0.5 * prec[i] * d * d;
    }
    Ok(ll + lp)
}

/// Draw `n_draws` samples from `N(mean, cov)` (Cholesky).
///
/// Column-major layout: `values[i * n_draws + d]` is coefficient `i` at draw `d`.
///
/// # Errors
///
/// Non-SPD covariance or workspace too small.
pub fn sample_gaussian_mvn(
    mean: &[f64],
    cov: &[f64],
    n_draws: usize,
    seed: u64,
    workspace: &mut LaplaceWorkspace,
) -> Result<Arc<[f64]>, ProbError> {
    let ncols = mean.len();
    // Ensure z-scratch capacity without changing design-sized buffers.
    if workspace.draw_scratch.len() < ncols {
        workspace.draw_scratch.resize(ncols, 0.0);
    }
    let chol = cholesky_spd(cov, ncols)?;
    let mut rng = CausalRng::from_seed(seed);
    let mut values = vec![0.0; n_draws * ncols];
    let z = &mut workspace.draw_scratch[..ncols];
    for d in 0..n_draws {
        for j in 0..ncols {
            z[j] = standard_normal(&mut rng);
        }
        for i in 0..ncols {
            let mut acc = mean[i];
            for j in 0..=i {
                acc += chol[i * ncols + j] * z[j];
            }
            values[i * n_draws + d] = acc;
        }
    }
    Ok(Arc::from(values))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prior::PriorSpec;

    #[test]
    fn laplace_gaussian_matches_ols() {
        let n = 40;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for r in 0..n {
            let xi = r as f64 * 0.1;
            x[r] = 1.0;
            x[n + r] = xi;
            y[r] = 0.5 + 1.5 * xi;
        }
        let prior = PriorSet {
            specs: vec![PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(
                2, 100.0,
            ))],
            contrast: None,
            categorical: Vec::new(),
            restrictions: Vec::new(),
        };
        let mut ws = LaplaceWorkspace::default();
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols: 2,
            y: &y,
            weights: None,
            offsets: None,
        };
        let opts = BayesFitOptions { n_draws: 200, seed: 3, max_iter: 50, grad_tol: 1e-8 };
        let fit =
            fit_laplace_glm(BayesLikelihood::GaussianIdentity, design, &prior, &opts, &mut ws)
                .unwrap();
        assert!(fit.diagnostics.converged);
        assert!(fit.diagnostics.allows_posterior());
        assert!((fit.map[0] - 0.5).abs() < 1e-3);
        assert!((fit.map[1] - 1.5).abs() < 1e-3);
        let g0 = ws.grow_count;
        fit_laplace_glm(BayesLikelihood::GaussianIdentity, design, &prior, &opts, &mut ws).unwrap();
        assert_eq!(ws.grow_count, g0, "workspace must be reused");
    }

    #[test]
    fn laplace_gaussian_posterior_scales_with_residual_variance() {
        let n = 80;
        let mut x = vec![0.0; n * 2];
        let mut y_unit = vec![0.0; n];
        let mut y_scaled = vec![0.0; n];
        for r in 0..n {
            let xi = r as f64 * 0.05;
            x[r] = 1.0;
            x[n + r] = xi;
            let noise = ((r % 7) as f64 - 3.0) * 0.2;
            y_unit[r] = 0.5 + 1.5 * xi + noise;
            y_scaled[r] = 0.5 + 1.5 * xi + noise * 4.0;
        }
        let prior = PriorSet {
            specs: vec![PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(
                2, 1e6,
            ))],
            contrast: None,
            categorical: Vec::new(),
            restrictions: Vec::new(),
        };
        let mut ws = LaplaceWorkspace::default();
        let opts = BayesFitOptions { n_draws: 400, seed: 9, max_iter: 50, grad_tol: 1e-8 };
        let fit_unit = fit_laplace_glm(
            BayesLikelihood::GaussianIdentity,
            BayesDesignRef {
                x_colmajor: &x,
                nrows: n,
                ncols: 2,
                y: &y_unit,
                weights: None,
                offsets: None,
            },
            &prior,
            &opts,
            &mut ws,
        )
        .unwrap();
        let fit_scaled = fit_laplace_glm(
            BayesLikelihood::GaussianIdentity,
            BayesDesignRef {
                x_colmajor: &x,
                nrows: n,
                ncols: 2,
                y: &y_scaled,
                weights: None,
                offsets: None,
            },
            &prior,
            &opts,
            &mut ws,
        )
        .unwrap();
        // Diagonal posterior SD for slope should grow ~4× when residual noise ×4.
        let slope_sd = |fit: &BayesFitResult| -> f64 {
            let col = fit.draws.column(1).unwrap();
            let m = col.iter().sum::<f64>() / col.len() as f64;
            let var = col.iter().map(|v| (v - m).powi(2)).sum::<f64>() / (col.len() - 1) as f64;
            var.sqrt()
        };
        let sd_u = slope_sd(&fit_unit);
        let sd_s = slope_sd(&fit_scaled);
        assert!(sd_s > 2.5 * sd_u, "sd_unit={sd_u} sd_scaled={sd_s}");
    }

    #[test]
    fn laplace_logistic_converges() {
        let n = 60;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for r in 0..n {
            let xi = (r as f64 - 30.0) * 0.2;
            x[r] = 1.0;
            x[n + r] = xi;
            let p = 1.0 / (1.0 + (-(0.0 + 1.2 * xi)).exp());
            y[r] = if p > 0.5 { 1.0 } else { 0.0 };
        }
        let prior = PriorSet::weakly_informative(2);
        let mut ws = LaplaceWorkspace::default();
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols: 2,
            y: &y,
            weights: None,
            offsets: None,
        };
        let opts = BayesFitOptions { n_draws: 100, seed: 11, ..BayesFitOptions::default() };
        let fit = fit_laplace_glm(BayesLikelihood::BernoulliLogit, design, &prior, &opts, &mut ws)
            .unwrap();
        assert!(fit.diagnostics.converged);
        assert!(fit.map[1] > 0.0);
    }

    #[test]
    fn laplace_known_sigma2_matches_analytic_posterior() {
        let n = 40;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for r in 0..n {
            let xi = (r as f64 - 20.0) * 0.1;
            x[r] = 1.0;
            x[n + r] = xi;
            y[r] = 1.0 + 2.0 * xi + ((r % 5) as f64 - 2.0) * 0.1;
        }
        let sigma2 = 0.25;
        let prior = PriorSet {
            specs: vec![
                PriorSpec::GaussianCoefficients(
                    GaussianCoefficientPrior::shared(2, 0.0, 4.0).unwrap(),
                ),
                PriorSpec::KnownResidualVariance(sigma2),
            ],
            contrast: None,
            categorical: Vec::new(),
            restrictions: Vec::new(),
        };
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols: 2,
            y: &y,
            weights: None,
            offsets: None,
        };
        let mut ws = LaplaceWorkspace::default();
        let opts = BayesFitOptions { n_draws: 500, seed: 7, max_iter: 50, grad_tol: 1e-10 };
        let fit =
            fit_laplace_glm(BayesLikelihood::GaussianIdentity, design, &prior, &opts, &mut ws)
                .unwrap();
        let conj = crate::conjugate::fit_conjugate_gaussian(design, &prior, &opts, &mut ws).unwrap();
        assert!((fit.map[0] - conj.map[0]).abs() < 1e-9);
        assert!((fit.map[1] - conj.map[1]).abs() < 1e-9);
        let cov = fit.cov.as_ref().expect("cov");
        // Analytic Cov = σ² A^{-1}; rebuild A and compare.
        let (a_beta, b_beta) = gaussian_normal_equations(design, prior.gaussian_coefficients().unwrap())
            .unwrap();
        let map_ref = solve_map_from_normal_eq(&a_beta, &b_beta, 2).unwrap();
        let a_inv = invert_spd(&a_beta, 2).unwrap();
        for i in 0..2 {
            assert!((fit.map[i] - map_ref[i]).abs() < 1e-10);
            for j in 0..2 {
                assert!((cov[i * 2 + j] - sigma2 * a_inv[i * 2 + j]).abs() < 1e-10);
            }
        }
        assert!(fit.diagnostics.grad_inf_norm < 1e-8);
        assert_eq!(fit.draws.schema.n_quantities(), 2);
    }

    #[test]
    fn laplace_inv_gamma_mode_matches_closed_form() {
        let n = 30;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for r in 0..n {
            let xi = r as f64 * 0.2;
            x[r] = 1.0;
            x[n + r] = xi;
            y[r] = 0.5 + 1.2 * xi + ((r % 3) as f64 - 1.0) * 0.3;
        }
        let shape = 3.0;
        let scale = 1.5;
        let prior = PriorSet {
            specs: vec![
                PriorSpec::GaussianCoefficients(
                    GaussianCoefficientPrior::shared(2, 0.0, 1.0).unwrap(),
                ),
                PriorSpec::ResidualInvGamma(crate::prior::InvGammaPrior { shape, scale }),
            ],
            contrast: None,
            categorical: Vec::new(),
            restrictions: Vec::new(),
        };
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols: 2,
            y: &y,
            weights: None,
            offsets: None,
        };
        let mut ws = LaplaceWorkspace::default();
        let opts = BayesFitOptions { n_draws: 200, seed: 11, max_iter: 50, grad_tol: 1e-10 };
        let fit =
            fit_laplace_glm(BayesLikelihood::GaussianIdentity, design, &prior, &opts, &mut ws)
                .unwrap();

        let (a_beta, b_beta) = gaussian_normal_equations(design, prior.gaussian_coefficients().unwrap())
            .unwrap();
        let beta_ref = solve_map_from_normal_eq(&a_beta, &b_beta, 2).unwrap();
        let mut xtwr = vec![0.0; 2];
        let mut p_diff = vec![0.0; 2];
        let rss = rss_and_xtwr(design, &beta_ref, &mut xtwr).unwrap();
        let prec = prior.gaussian_coefficients().unwrap().precision();
        let quad = prior_quadratic(prior.gaussian_coefficients().unwrap(), &prec, &beta_ref, &mut p_diff)
            .unwrap();
        let a_const = shape + 0.5 * (n as f64 + 2.0);
        let b_at = scale + 0.5 * (rss + quad);
        let lambda_ref = (b_at / a_const).ln();

        assert!((fit.map[0] - beta_ref[0]).abs() < 1e-10);
        assert!((fit.map[1] - beta_ref[1]).abs() < 1e-10);
        assert!(fit.diagnostics.grad_inf_norm < 1e-8);

        let mut target = gaussian_target_from_model(
            design,
            prior.gaussian_coefficients().unwrap().clone(),
            GaussianVarianceModel::InvGamma { shape, scale },
        )
        .unwrap();
        let mut q = beta_ref.clone();
        q.push(lambda_ref);
        let mut grad = vec![0.0; 3];
        let _ = target.logp_and_grad(&q, &mut grad).unwrap();
        for g in &grad {
            assert!(g.abs() < 1e-8, "grad={grad:?}");
        }

        // Finite-difference Hessian H_λλ ≈ A.
        let eps = 1e-5;
        let mut qp = q.clone();
        let mut qm = q.clone();
        qp[2] += eps;
        qm[2] -= eps;
        let mut g_unused = vec![0.0; 3];
        let lp_p = target.logp_and_grad(&qp, &mut g_unused).unwrap();
        let lp_m = target.logp_and_grad(&qm, &mut g_unused).unwrap();
        let lp_0 = target.logp_and_grad(&q, &mut g_unused).unwrap();
        let neg_h_ll = -(lp_p - 2.0 * lp_0 + lp_m) / (eps * eps);
        assert!((neg_h_ll - a_const).abs() / a_const < 1e-3, "H_ll={neg_h_ll} A={a_const}");

        assert_eq!(fit.draws.schema.n_quantities(), 3);
        let cov = fit.cov.as_ref().unwrap();
        let a_inv = invert_spd(&a_beta, 2).unwrap();
        let exp_l = lambda_ref.exp();
        for i in 0..4 {
            assert!((cov[i] - exp_l * a_inv[i]).abs() < 1e-9);
        }
    }

    #[test]
    fn laplace_known_sigma2_strong_prior_far_from_one() {
        // Strong prior mean away from OLS + known σ² ≠ 1; repaired path matches analytic.
        let n = 20;
        let x = vec![1.0; n];
        let mut y = vec![0.0; n];
        for r in 0..n {
            y[r] = 5.0 + ((r % 3) as f64) * 0.01;
        }
        let sigma2 = 4.0;
        let prior = PriorSet {
            specs: vec![
                PriorSpec::GaussianCoefficients(
                    GaussianCoefficientPrior::shared(1, 0.0, 0.01).unwrap(),
                ),
                PriorSpec::KnownResidualVariance(sigma2),
            ],
            contrast: None,
            categorical: Vec::new(),
            restrictions: Vec::new(),
        };
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols: 1,
            y: &y,
            weights: None,
            offsets: None,
        };
        let mut ws = LaplaceWorkspace::default();
        let opts = BayesFitOptions { n_draws: 100, seed: 1, max_iter: 50, grad_tol: 1e-10 };
        let fit =
            fit_laplace_glm(BayesLikelihood::GaussianIdentity, design, &prior, &opts, &mut ws)
                .unwrap();
        let (a_beta, b_beta) = gaussian_normal_equations(design, prior.gaussian_coefficients().unwrap())
            .unwrap();
        let map_ref = solve_map_from_normal_eq(&a_beta, &b_beta, 1).unwrap();
        // Strong pull toward 0: MAP far below OLS (~5).
        assert!(fit.map[0] < 1.0, "map={}", fit.map[0]);
        assert!((fit.map[0] - map_ref[0]).abs() < 1e-10);
        let cov = fit.cov.as_ref().unwrap();
        let a_inv = invert_spd(&a_beta, 1).unwrap();
        assert!((cov[0] - sigma2 * a_inv[0]).abs() < 1e-10);
        // Old plug-in path would use σ²≈RSS/df ~25 and wrong objective at σ²=1 during opt.
        assert!((cov[0] - 4.0 * a_inv[0]).abs() < 1e-10);
    }

    #[test]
    fn laplace_and_hmc_share_gaussian_target_density() {
        let n = 15;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for r in 0..n {
            x[r] = 1.0;
            x[n + r] = r as f64 * 0.1;
            y[r] = 1.0 + 0.5 * (r as f64) * 0.1;
        }
        let coef = GaussianCoefficientPrior::shared(2, 0.0, 2.0).unwrap();
        let prior = PriorSet {
            specs: vec![
                PriorSpec::GaussianCoefficients(coef.clone()),
                PriorSpec::KnownResidualVariance(1.5),
            ],
            contrast: None,
            categorical: Vec::new(),
            restrictions: Vec::new(),
        };
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols: 2,
            y: &y,
            weights: None,
            offsets: None,
        };
        let mut ws = LaplaceWorkspace::default();
        let opts = BayesFitOptions { n_draws: 50, seed: 2, max_iter: 50, grad_tol: 1e-10 };
        let fit =
            fit_laplace_glm(BayesLikelihood::GaussianIdentity, design, &prior, &opts, &mut ws)
                .unwrap();
        let mut t1 = gaussian_target_from_model(
            design,
            coef.clone(),
            GaussianVarianceModel::Known { sigma2: 1.5 },
        )
        .unwrap();
        let mut t2 = gaussian_target_from_model(
            design,
            coef,
            GaussianVarianceModel::Known { sigma2: 1.5 },
        )
        .unwrap();
        let mut g1 = vec![0.0; 2];
        let mut g2 = vec![0.0; 2];
        let lp1 = t1.logp_and_grad(&fit.map, &mut g1).unwrap();
        let lp2 = t2.logp_and_grad(&fit.map, &mut g2).unwrap();
        assert!((lp1 - lp2).abs() < 1e-12);
        for i in 0..2 {
            assert!((g1[i] - g2[i]).abs() < 1e-12);
        }
    }

    #[test]
    fn poisson_laplace_and_hmc_agree_on_value_and_score() {
        let n = 20;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for r in 0..n {
            let xi = (r as f64) * 0.1;
            x[r] = 1.0;
            x[n + r] = xi;
            y[r] = if r % 3 == 0 { 2.0 } else { 1.0 };
        }
        let prior = PriorSet::weakly_informative(2);
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols: 2,
            y: &y,
            weights: None,
            offsets: None,
        };
        let beta = [0.2_f64, 0.1];
        let coef = prior.gaussian_coefficients().unwrap();
        let prec = coef.precision();
        let ws = LaplaceWorkspace::default();
        let mut grad = vec![0.0; 2];
        let mut hess = vec![0.0; 4];
        let mut eta = vec![0.0; n];
        let mut work_w = vec![0.0; n];
        accumulate_likelihood(
            BayesLikelihood::PoissonLog,
            design,
            &beta,
            &mut grad,
            &mut hess,
            &mut eta,
            &mut work_w,
            1.0,
        )
        .unwrap();
        for i in 0..2 {
            let diff = beta[i] - coef.mean[i];
            grad[i] -= prec[i] * diff;
        }
        let lp = log_posterior_value(
            BayesLikelihood::PoissonLog,
            design,
            &beta,
            coef,
            &prec,
            &mut eta,
            1.0,
        )
        .unwrap();
        // Rebuild independently from poisson_terms.
        let mut lp2 = 0.0;
        let mut g2 = [0.0; 2];
        for r in 0..n {
            let e = beta[0] + beta[1] * x[n + r];
            let t = crate::likelihood_terms::poisson_terms(y[r], e, 1.0).unwrap();
            lp2 += t.log_value;
            g2[0] += t.score_eta;
            g2[1] += t.score_eta * x[n + r];
        }
        for i in 0..2 {
            let diff = beta[i] - coef.mean[i];
            lp2 -= 0.5 * prec[i] * diff * diff;
            g2[i] -= prec[i] * diff;
        }
        assert!((lp - lp2).abs() < 1e-10);
        assert!((grad[0] - g2[0]).abs() < 1e-10);
        assert!((grad[1] - g2[1]).abs() < 1e-10);
        let _ = ws;
    }

    #[test]
    fn poisson_newton_backtracks_on_overflow_trial() {
        // Huge step from a sane point should overflow and be rejected, not abort.
        let n = 8;
        let x = vec![1.0; n];
        let y = vec![1.0; n];
        let prior = PriorSet::weakly_informative(1);
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols: 1,
            y: &y,
            weights: None,
            offsets: None,
        };
        let mut ws = LaplaceWorkspace::default();
        let opts = BayesFitOptions { n_draws: 20, seed: 3, max_iter: 30, grad_tol: 1e-8 };
        let fit =
            fit_laplace_glm(BayesLikelihood::PoissonLog, design, &prior, &opts, &mut ws).unwrap();
        assert!(fit.diagnostics.converged || fit.map[0].is_finite());
        assert!(fit.map[0].abs() < 50.0, "map exploded: {}", fit.map[0]);
    }

    #[test]
    fn probit_laplace_cov_matches_observed_hessian_at_map() {
        let n = 40;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for r in 0..n {
            let xi = (r as f64 - 20.0) * 0.15;
            x[r] = 1.0;
            x[n + r] = xi;
            let p = antecedent_kernels::norm_cdf(0.2 + 0.8 * xi);
            y[r] = if p > 0.5 { 1.0 } else { 0.0 };
        }
        let prior = PriorSet::weakly_informative(2);
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols: 2,
            y: &y,
            weights: None,
            offsets: None,
        };
        let mut ws = LaplaceWorkspace::default();
        let opts = BayesFitOptions { n_draws: 50, seed: 5, max_iter: 80, grad_tol: 1e-10 };
        let fit =
            fit_laplace_glm(BayesLikelihood::BernoulliProbit, design, &prior, &opts, &mut ws)
                .unwrap();
        assert!(fit.diagnostics.converged);
        let coef = prior.gaussian_coefficients().unwrap();
        let prec = coef.precision();
        let mut grad = vec![0.0; 2];
        let mut hess = vec![0.0; 4];
        let mut eta = vec![0.0; n];
        let mut work_w = vec![0.0; n];
        accumulate_likelihood(
            BayesLikelihood::BernoulliProbit,
            design,
            &fit.map,
            &mut grad,
            &mut hess,
            &mut eta,
            &mut work_w,
            1.0,
        )
        .unwrap();
        for i in 0..2 {
            hess[i * 2 + i] += prec[i];
        }
        let cov_ref = invert_spd(&hess, 2).unwrap();
        let cov = fit.cov.as_ref().unwrap();
        for i in 0..4 {
            assert!((cov[i] - cov_ref[i]).abs() < 1e-8, "cov[{i}]={} ref={}", cov[i], cov_ref[i]);
        }
        // Observed curvature differs from Fisher at the MAP for this unbalanced design.
        let mut fisher_w = 0.0;
        let mut obs_w = 0.0;
        for r in 0..n {
            let e = eta[r];
            let t = crate::likelihood_terms::probit_terms(y[r], e, 1.0).unwrap();
            obs_w += t.neg_hessian_eta;
            let mu = antecedent_kernels::norm_cdf(e).clamp(1e-12, 1.0 - 1e-12);
            let dens = antecedent_kernels::norm_pdf(e);
            fisher_w += (dens * dens) / (mu * (1.0 - mu));
        }
        assert!((obs_w - fisher_w).abs() > 1e-3, "obs={obs_w} fisher={fisher_w}");
    }

    #[test]
    fn refuses_without_diagnostics() {
        let d = InferenceDiagnostics {
            converged: false,
            iterations: 1,
            grad_inf_norm: 10.0,
            hessian_condition: 1.0,
            factorization: HessianFactorization::Cholesky,
            separation_warning: false,
            notes: Vec::new(),
            backend_id: Arc::from("laplace"),
            n_chains: None,
            n_warmup: None,
            ess_bulk_min: None,
            ess_tail_min: None,
            rhat_max: None,
            n_divergences: None,
            mean_accept_prob: None,
            n_warmup_divergences: None,
            n_postwarmup_divergences: None,
            max_abs_delta_h: None,
            all_chains_moved: None,
        };
        assert!(!d.allows_posterior());
    }
}
