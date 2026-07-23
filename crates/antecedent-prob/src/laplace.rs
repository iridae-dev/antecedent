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
use antecedent_kernels::{norm_cdf, standard_normal};

use crate::backend::{
    BayesDesignRef, BayesFitOptions, BayesFitResult, BayesLikelihood, InferenceBackend,
    LaplaceWorkspace,
};
use crate::diagnostics::{HessianFactorization, InferenceDiagnostics};
use crate::error::ProbError;
use crate::linalg::{cholesky_spd, condition_from_chol, invert_spd, ldlt_decompose, solve_spd};
use crate::posterior::{PosteriorDraws, PosteriorSchema};
use crate::prior::{GaussianCoefficientPrior, PriorSet};

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
        for _ in 0..8 {
            for i in 0..ncols {
                workspace.beta[i] = beta_old[i] + step_scale * workspace.step[i];
            }
            let new_obj = log_posterior_value(
                likelihood,
                design,
                &workspace.beta[..ncols],
                &coef_prior,
                &prec,
                &mut workspace.eta[..nrows],
                1.0,
            )?;
            let old_obj = log_posterior_value(
                likelihood,
                design,
                &beta_old,
                &coef_prior,
                &prec,
                &mut workspace.eta[..nrows],
                1.0,
            )?;
            if new_obj >= old_obj - 1e-12 {
                accepted = true;
                break;
            }
            step_scale *= 0.5;
        }
        if !accepted {
            for i in 0..ncols {
                workspace.beta[i] = beta_old[i];
            }
            break;
        }
    }

    // Final gradient / Hessian at MAP. For GaussianIdentity, scale by residual σ² so
    // posterior covariance matches the frequentist OLS scale when the prior is weak.
    let gaussian_sigma2 = match likelihood {
        BayesLikelihood::GaussianIdentity => {
            gaussian_residual_sigma2(design, &workspace.beta[..ncols])
        }
        _ => 1.0,
    };
    accumulate_likelihood(
        likelihood,
        design,
        &workspace.beta[..ncols],
        &mut workspace.grad[..ncols],
        &mut workspace.neg_hessian[..ncols * ncols],
        &mut workspace.eta[..nrows],
        &mut workspace.work_w[..nrows],
        gaussian_sigma2,
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
        rhat_max: None,
        n_divergences: None,
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

        let (mu, var_w, sep) = match likelihood {
            BayesLikelihood::GaussianIdentity => {
                // Working weight for −Hessian ≈ X' (w/σ²) X
                (e, inv_sigma2, false)
            }
            BayesLikelihood::BernoulliLogit => {
                let mu = 1.0 / (1.0 + (-e).exp());
                if mu < 1e-8 || mu > 1.0 - 1e-8 {
                    separation = true;
                }
                let v = mu * (1.0 - mu);
                (mu, v.max(1e-12), mu < 1e-8 || mu > 1.0 - 1e-8)
            }
            BayesLikelihood::BernoulliProbit => {
                let mu = norm_cdf(e);
                if mu < 1e-8 || mu > 1.0 - 1e-8 {
                    separation = true;
                }
                let dens = norm_pdf(e);
                // Working weight ≈ φ(η)² / (Φ(1-Φ))
                let v = (dens * dens) / (mu * (1.0 - mu)).max(1e-12);
                (mu, v.max(1e-12), mu < 1e-8 || mu > 1.0 - 1e-8)
            }
            BayesLikelihood::PoissonLog => {
                let mu = e.exp().min(1e6);
                (mu, mu.max(1e-12), false)
            }
        };
        separation |= sep;
        work_w[r] = w_obs * var_w;

        let resid = y - mu;

        // Score contribution: for GLM, ∂ℓ/∂β = X' W_working^{-1/2} stuff;
        // use standard GLM score X'(y−μ) for canonical / working forms.
        let score_scale = match likelihood {
            BayesLikelihood::GaussianIdentity => w_obs * resid * inv_sigma2,
            BayesLikelihood::BernoulliLogit => w_obs * resid,
            BayesLikelihood::BernoulliProbit => {
                // ∂ℓ/∂η = (y-μ) φ / (μ(1-μ))
                let dens = norm_pdf(e);
                w_obs * resid * dens / (mu * (1.0 - mu)).max(1e-12)
            }
            BayesLikelihood::PoissonLog => w_obs * resid,
        };

        for c in 0..ncols {
            let x = design.x_colmajor[c * nrows + r];
            grad[c] += x * score_scale;
        }
        // −Hessian ≈ X' diag(w) X
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

pub(crate) fn gaussian_residual_sigma2(design: BayesDesignRef<'_>, beta: &[f64]) -> f64 {
    let nrows = design.nrows;
    let ncols = design.ncols;
    let mut rss = 0.0;
    let mut wsum = 0.0;
    for r in 0..nrows {
        let offset = design.offsets.map_or(0.0, |o| o[r]);
        let mut pred = offset;
        for c in 0..ncols {
            pred += design.x_colmajor[c * nrows + r] * beta[c];
        }
        let w = design.weights.map_or(1.0, |ww| ww[r]);
        let e = design.y[r] - pred;
        rss += w * e * e;
        wsum += w;
    }
    let df = (wsum - ncols as f64).max(1.0);
    (rss / df).max(1e-12)
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
        ll += w * match likelihood {
            BayesLikelihood::GaussianIdentity => {
                let r = y - e;
                -0.5 * inv_sigma2 * r * r
            }
            BayesLikelihood::BernoulliLogit => {
                // y*η - softplus(η)
                y * e - softplus(e)
            }
            BayesLikelihood::BernoulliProbit => {
                let p = norm_cdf(e).clamp(1e-12, 1.0 - 1e-12);
                y * p.ln() + (1.0 - y) * (1.0 - p).ln()
            }
            BayesLikelihood::PoissonLog => {
                let mu = e.exp();
                y * e - mu
            }
        };
    }
    let mut lp = 0.0;
    for i in 0..ncols {
        let d = beta[i] - prior.mean[i];
        lp -= 0.5 * prec[i] * d * d;
    }
    Ok(ll + lp)
}

fn softplus(x: f64) -> f64 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

fn norm_pdf(x: f64) -> f64 {
    const INV_SQRT_2PI: f64 = 0.398_942_280_401_432_7;
    INV_SQRT_2PI * (-0.5 * x * x).exp()
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
            rhat_max: None,
            n_divergences: None,
        };
        assert!(!d.allows_posterior());
    }
}
