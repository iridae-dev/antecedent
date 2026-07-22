//! Native Hamiltonian Monte Carlo for Bayesian GLMs.
//!
//! Leapfrog HMC with dual-averaging step-size adaptation during warmup.
//! Multi-chain draws are columnar; ESS / R-hat / divergence counts gate publication.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::many_single_char_names,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

use std::sync::Arc;

use causal_core::{CausalRng, ExecutionContext};
use causal_kernels::standard_normal;

use crate::backend::{
    BayesDesignRef, BayesFitOptions, BayesFitResult, BayesLikelihood, InferenceBackend,
    LaplaceWorkspace,
};
use crate::diagnostics::{HessianFactorization, InferenceDiagnostics};
use crate::error::ProbError;
use crate::laplace::{
    accumulate_likelihood, gaussian_residual_sigma2, log_posterior_value, validate_design,
};
use crate::mcmc_stats::{max_split_rhat, min_bulk_ess};
use crate::posterior::{PosteriorDraws, PosteriorSchema};
use crate::prior::{GaussianCoefficientPrior, PriorSet};

/// Default HMC sampler settings.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct HmcOptions {
    /// Number of chains (≥ 2 required for R-hat).
    pub n_chains: usize,
    /// Warmup iterations discarded per chain.
    pub n_warmup: usize,
    /// Leapfrog steps per trajectory.
    pub leapfrog_steps: u32,
    /// Initial leapfrog step size.
    pub step_size: f64,
    /// Dual-averaging target acceptance probability.
    pub target_accept: f64,
    /// Diagonal mass-matrix scale (kinetic energy `½ Σ p² / mass`).
    pub mass: f64,
}

impl Default for HmcOptions {
    fn default() -> Self {
        Self {
            n_chains: 4,
            n_warmup: 200,
            leapfrog_steps: 10,
            step_size: 0.1,
            target_accept: 0.8,
            mass: 1.0,
        }
    }
}

/// Native HMC Bayesian GLM backend.
#[derive(Clone, Copy, Debug, Default)]
pub struct HmcGlmBackend {
    /// Sampler options.
    pub options: HmcOptions,
}

impl HmcGlmBackend {
    /// Construct with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Override sampler options.
    #[must_use]
    pub const fn with_options(mut self, options: HmcOptions) -> Self {
        self.options = options;
        self
    }
}

impl InferenceBackend for HmcGlmBackend {
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
        fit_hmc_glm(likelihood, design, prior, options, self.options, workspace)
    }
}

/// Run multi-chain HMC and return columnar post-warmup draws.
///
/// # Errors
///
/// Shape, prior, or diagnostics gate failures.
pub fn fit_hmc_glm(
    likelihood: BayesLikelihood,
    design: BayesDesignRef<'_>,
    prior: &PriorSet,
    fit_opts: &BayesFitOptions,
    hmc: HmcOptions,
    workspace: &mut LaplaceWorkspace,
) -> Result<BayesFitResult, ProbError> {
    let nrows = design.nrows;
    let ncols = design.ncols;
    validate_design(design)?;
    if hmc.n_chains < 2 {
        return Err(ProbError::Inference {
            message: "HMC requires at least 2 chains for R-hat / ESS",
        });
    }
    if hmc.leapfrog_steps == 0 || !(hmc.step_size > 0.0) || !(hmc.mass > 0.0) {
        return Err(ProbError::Inference { message: "invalid HMC step_size / mass / L" });
    }
    if fit_opts.n_draws == 0 {
        return Err(ProbError::Shape { message: "n_draws must be > 0" });
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

    let n_keep = fit_opts.n_draws;
    let total_draws = n_keep.saturating_mul(hmc.n_chains);
    workspace.prepare(nrows, ncols, total_draws.max(ncols));

    let mut chain_samples = vec![0.0; hmc.n_chains * n_keep * ncols];
    let mut n_divergences = 0u32;
    let mut map = coef_prior.mean.to_vec();
    let mut best_lp = f64::NEG_INFINITY;

    for chain in 0..hmc.n_chains {
        let mut rng = CausalRng::from_seed(
            fit_opts.seed ^ ((chain as u64).wrapping_add(1).wrapping_mul(0xD1B5_4A32_D192_ED03)),
        );
        let mut beta = coef_prior.mean.to_vec();
        let mut step_size = hmc.step_size;
        let mut log_eps_bar = step_size.ln();
        let mut h_bar = 0.0;

        let total_iters = hmc.n_warmup.saturating_add(n_keep);
        let mut kept = 0usize;
        for t in 0..total_iters {
            let gaussian_sigma2 = match likelihood {
                BayesLikelihood::GaussianIdentity => gaussian_residual_sigma2(design, &beta),
                _ => 1.0,
            };
            let lp_old = log_posterior_value(
                likelihood,
                design,
                &beta,
                &coef_prior,
                &prec,
                &mut workspace.eta[..nrows],
                gaussian_sigma2,
            )?;
            let (accepted, divergent, new_beta, lp) = hmc_step(
                likelihood,
                design,
                &coef_prior,
                &prec,
                &beta,
                step_size,
                hmc.leapfrog_steps,
                hmc.mass,
                gaussian_sigma2,
                lp_old,
                workspace,
                &mut rng,
            )?;
            if divergent {
                n_divergences = n_divergences.saturating_add(1);
            }
            if accepted {
                beta = new_beta;
                if lp > best_lp {
                    best_lp = lp;
                    map.copy_from_slice(&beta);
                }
            }

            match t.cmp(&hmc.n_warmup) {
                std::cmp::Ordering::Less => {
                    let accept_prob = if accepted { 1.0 } else { 0.0 };
                    let m = (t + 1) as f64;
                    let eta = 1.0 / (m + 10.0);
                    h_bar = (1.0 - eta) * h_bar + eta * (hmc.target_accept - accept_prob);
                    let log_eps = hmc.step_size.ln() - (m.sqrt() / 0.05) * h_bar;
                    step_size = log_eps.exp().clamp(1e-6, 2.0);
                    let kappa = m.powf(-0.75);
                    log_eps_bar = kappa * log_eps + (1.0 - kappa) * log_eps_bar;
                }
                std::cmp::Ordering::Equal => {
                    step_size = log_eps_bar.exp().clamp(1e-6, 2.0);
                }
                std::cmp::Ordering::Greater => {}
            }

            if t >= hmc.n_warmup {
                let base = (chain * n_keep + kept) * ncols;
                chain_samples[base..base + ncols].copy_from_slice(&beta);
                kept += 1;
            }
        }
    }

    let mut values = vec![0.0; total_draws * ncols];
    for chain in 0..hmc.n_chains {
        for d in 0..n_keep {
            let src = (chain * n_keep + d) * ncols;
            let dest_draw = chain * n_keep + d;
            for q in 0..ncols {
                values[q * total_draws + dest_draw] = chain_samples[src + q];
            }
        }
    }

    let ess_min = min_bulk_ess(&chain_samples, hmc.n_chains, n_keep, ncols);
    let rhat_max = max_split_rhat(&chain_samples, hmc.n_chains, n_keep, ncols);

    let diagnostics = InferenceDiagnostics {
        converged: rhat_max.is_finite() && rhat_max < 1.05 && ess_min.is_finite() && ess_min > 10.0,
        iterations: (hmc.n_warmup + n_keep) as u32,
        grad_inf_norm: 0.0,
        hessian_condition: f64::NAN,
        factorization: HessianFactorization::Mcmc,
        separation_warning: false,
        notes: vec![Arc::from(format!(
            "hmc chains={} warmup={} L={}",
            hmc.n_chains, hmc.n_warmup, hmc.leapfrog_steps
        ))],
        backend_id: Arc::from("hmc"),
        n_chains: Some(hmc.n_chains as u32),
        n_warmup: Some(hmc.n_warmup as u32),
        ess_bulk_min: Some(ess_min),
        rhat_max: Some(rhat_max),
        n_divergences: Some(n_divergences),
    };

    if !diagnostics.allows_posterior() {
        return Err(ProbError::MissingDiagnostics {
            message: "HMC posterior refused without ESS/R-hat/divergence diagnostics",
        });
    }

    let draws = PosteriorDraws::from_column_major(
        PosteriorSchema::coefficients(ncols),
        total_draws,
        values,
    )?;
    Ok(BayesFitResult { draws, map, diagnostics })
}

fn hmc_step(
    likelihood: BayesLikelihood,
    design: BayesDesignRef<'_>,
    coef_prior: &GaussianCoefficientPrior,
    prec: &[f64],
    beta: &[f64],
    step_size: f64,
    leapfrog_steps: u32,
    mass: f64,
    gaussian_sigma2: f64,
    lp_old: f64,
    workspace: &mut LaplaceWorkspace,
    rng: &mut CausalRng,
) -> Result<(bool, bool, Vec<f64>, f64), ProbError> {
    let ncols = beta.len();
    let nrows = design.nrows;
    let mut q = beta.to_vec();
    let mut p = vec![0.0; ncols];
    let mut p0_energy = 0.0;
    for i in 0..ncols {
        p[i] = mass.sqrt() * standard_normal(rng);
        p0_energy += 0.5 * p[i] * p[i] / mass;
    }

    let mut grad = vec![0.0; ncols];
    neg_log_posterior_grad(
        likelihood,
        design,
        coef_prior,
        prec,
        &q,
        gaussian_sigma2,
        &mut grad,
        workspace,
    )?;
    for i in 0..ncols {
        p[i] -= 0.5 * step_size * grad[i];
    }

    let mut divergent = false;
    for step in 0..leapfrog_steps {
        for i in 0..ncols {
            q[i] += step_size * p[i] / mass;
            if !q[i].is_finite() {
                divergent = true;
                break;
            }
        }
        if divergent {
            break;
        }
        neg_log_posterior_grad(
            likelihood,
            design,
            coef_prior,
            prec,
            &q,
            gaussian_sigma2,
            &mut grad,
            workspace,
        )?;
        let last = step + 1 == leapfrog_steps;
        let scale = if last { 0.5 } else { 1.0 };
        for i in 0..ncols {
            p[i] -= scale * step_size * grad[i];
            if !p[i].is_finite() {
                divergent = true;
            }
        }
        if divergent {
            break;
        }
    }

    if divergent {
        return Ok((false, true, beta.to_vec(), lp_old));
    }

    let lp_new = log_posterior_value(
        likelihood,
        design,
        &q,
        coef_prior,
        prec,
        &mut workspace.eta[..nrows],
        gaussian_sigma2,
    )?;
    let mut p_new_energy = 0.0;
    for i in 0..ncols {
        p_new_energy += 0.5 * p[i] * p[i] / mass;
    }

    let h_old = -lp_old + p0_energy;
    let h_new = -lp_new + p_new_energy;
    let log_alpha = (h_old - h_new).min(0.0);
    let accept = rng.next_f64().ln() < log_alpha;
    if accept { Ok((true, false, q, lp_new)) } else { Ok((false, false, beta.to_vec(), lp_old)) }
}

fn neg_log_posterior_grad(
    likelihood: BayesLikelihood,
    design: BayesDesignRef<'_>,
    coef_prior: &GaussianCoefficientPrior,
    prec: &[f64],
    beta: &[f64],
    gaussian_sigma2: f64,
    grad_out: &mut [f64],
    workspace: &mut LaplaceWorkspace,
) -> Result<(), ProbError> {
    let nrows = design.nrows;
    let ncols = beta.len();
    accumulate_likelihood(
        likelihood,
        design,
        beta,
        &mut workspace.grad[..ncols],
        &mut workspace.neg_hessian[..ncols * ncols],
        &mut workspace.eta[..nrows],
        &mut workspace.work_w[..nrows],
        gaussian_sigma2,
    )?;
    for i in 0..ncols {
        let diff = beta[i] - coef_prior.mean[i];
        workspace.grad[i] -= prec[i] * diff;
        grad_out[i] = -workspace.grad[i];
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::prior::PriorSpec;

    #[test]
    fn hmc_gaussian_recovers_slope() {
        let n = 60;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for r in 0..n {
            let xi = r as f64 * 0.05;
            x[r] = 1.0;
            x[n + r] = xi;
            y[r] = 0.5 + 1.5 * xi + ((r % 5) as f64 - 2.0) * 0.05;
        }
        let prior = PriorSet {
            specs: vec![PriorSpec::GaussianCoefficients(GaussianCoefficientPrior::isotropic(
                2, 10.0,
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
        let fit_opts = BayesFitOptions { n_draws: 80, seed: 11, max_iter: 50, grad_tol: 1e-8 };
        let hmc = HmcOptions {
            n_chains: 2,
            n_warmup: 60,
            leapfrog_steps: 8,
            step_size: 0.05,
            target_accept: 0.8,
            mass: 1.0,
        };
        let fit =
            fit_hmc_glm(BayesLikelihood::GaussianIdentity, design, &prior, &fit_opts, hmc, &mut ws)
                .unwrap();
        assert!(fit.diagnostics.allows_posterior());
        assert!(fit.diagnostics.rhat_max.unwrap() < 1.2);
        assert!((fit.map[1] - 1.5).abs() < 0.35, "map slope {}", fit.map[1]);
        assert_eq!(fit.draws.n_draws, 160);
    }
}
