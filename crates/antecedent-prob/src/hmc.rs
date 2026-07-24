//! Native Hamiltonian Monte Carlo for Bayesian GLMs.
//!
//! Leapfrog HMC with dual-averaging step-size adaptation during warmup.
//! Multi-chain draws are columnar; ESS / R-hat / divergence counts gate publication.
//!
//! GaussianIdentity uses a fixed prior-driven target (known σ² or joint
//! `(β, log σ²)` under an InvGamma residual prior). Other likelihoods keep the
//! GLM score path.
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

use antecedent_core::{CausalRng, ExecutionContext};
use antecedent_kernels::standard_normal;

use crate::backend::{
    BayesDesignRef, BayesFitOptions, BayesFitResult, BayesLikelihood, InferenceBackend,
    LaplaceWorkspace,
};
use crate::diagnostics::{HessianFactorization, InferenceDiagnostics};
use crate::error::ProbError;
use crate::gaussian_target::{GaussianTarget, PosteriorTarget, gaussian_target_from_model};
use crate::laplace::{accumulate_likelihood, log_posterior_value, validate_design};
use crate::mcmc_stats::{all_chains_moved, max_split_rhat, min_bulk_ess, min_tail_ess};
use crate::posterior::{PosteriorDraws, PosteriorQuantityKind, PosteriorSchema};
use crate::prior::{GaussianCoefficientPrior, GaussianVarianceModel, PriorSet};

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

/// Result of one HMC transition.
struct HmcStepResult {
    state: Vec<f64>,
    logp: f64,
    pub(crate) accepted: bool,
    accept_prob: f64,
    delta_h: f64,
    pub(crate) divergent: bool,
}

/// Running transition aggregates for publication diagnostics.
#[derive(Clone, Copy, Debug, Default)]
struct TransitionStats {
    n_warmup_divergences: u32,
    n_postwarmup_divergences: u32,
    accept_prob_sum: f64,
    n_transitions: u32,
    max_abs_delta_h: f64,
}

impl TransitionStats {
    fn record(&mut self, step: &HmcStepResult, is_warmup: bool) {
        self.n_transitions = self.n_transitions.saturating_add(1);
        self.accept_prob_sum += step.accept_prob;
        let abs_dh = if step.delta_h.is_finite() { step.delta_h.abs() } else { 1_001.0 };
        self.max_abs_delta_h = self.max_abs_delta_h.max(abs_dh);
        if step.divergent {
            if is_warmup {
                self.n_warmup_divergences = self.n_warmup_divergences.saturating_add(1);
            } else {
                self.n_postwarmup_divergences = self.n_postwarmup_divergences.saturating_add(1);
            }
        }
    }

    fn mean_accept_prob(self) -> f64 {
        if self.n_transitions == 0 {
            return 0.0;
        }
        self.accept_prob_sum / f64::from(self.n_transitions)
    }
}

fn finalize_energy(delta_h: f64) -> (bool, f64) {
    let divergent = !delta_h.is_finite() || delta_h.abs() > 1_000.0;
    let accept_prob = if divergent { 0.0 } else { (-delta_h).min(0.0).exp() };
    (divergent, accept_prob)
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

    if likelihood == BayesLikelihood::GaussianIdentity {
        let model = GaussianVarianceModel::from_prior_set(prior)?;
        let mut target = gaussian_target_from_model(design, coef_prior.clone(), model)?;
        return fit_hmc_gaussian(
            &mut target,
            model,
            nrows,
            ncols,
            fit_opts,
            hmc,
            workspace,
            &coef_prior,
        );
    }

    let prec = coef_prior.precision();
    let n_keep = fit_opts.n_draws;
    let total_draws = n_keep.saturating_mul(hmc.n_chains);
    workspace.prepare(nrows, ncols, total_draws.max(ncols));

    let mut chain_samples = vec![0.0; hmc.n_chains * n_keep * ncols];
    let mut stats = TransitionStats::default();
    let mut map = coef_prior.mean.to_vec();
    let mut best_lp = f64::NEG_INFINITY;

    for chain in 0..hmc.n_chains {
        let mut rng = CausalRng::from_seed(
            fit_opts.seed ^ ((chain as u64).wrapping_add(1).wrapping_mul(0xD1B5_4A32_D192_ED03)),
        );
        let mut beta = coef_prior.mean.to_vec();
        for bi in &mut beta {
            *bi += 0.1 * standard_normal(&mut rng);
        }
        let mut step_size = hmc.step_size;
        let mut log_eps_bar = step_size.ln();
        let mut h_bar = 0.0;

        let total_iters = hmc.n_warmup.saturating_add(n_keep);
        let mut kept = 0usize;
        for t in 0..total_iters {
            let lp_old = log_posterior_value(
                likelihood,
                design,
                &beta,
                &coef_prior,
                &prec,
                &mut workspace.eta[..nrows],
                1.0,
            )?;
            let step = hmc_step_glm(
                likelihood,
                design,
                &coef_prior,
                &prec,
                &beta,
                step_size,
                hmc.leapfrog_steps,
                hmc.mass,
                lp_old,
                workspace,
                &mut rng,
            )?;
            let is_warmup = t < hmc.n_warmup;
            stats.record(&step, is_warmup);
            if step.accepted {
                beta = step.state;
                if step.logp > best_lp {
                    best_lp = step.logp;
                    map.copy_from_slice(&beta);
                }
            }

            match t.cmp(&hmc.n_warmup) {
                std::cmp::Ordering::Less => {
                    dual_average_update(
                        &mut h_bar,
                        &mut log_eps_bar,
                        &mut step_size,
                        hmc,
                        t,
                        step.accept_prob,
                    );
                }
                std::cmp::Ordering::Equal => {
                    step_size = dual_average_finalize(log_eps_bar);
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

    pack_and_gate_hmc(&chain_samples, hmc, n_keep, ncols, ncols, false, map, stats)
}

fn fit_hmc_gaussian(
    target: &mut GaussianTarget<'_>,
    model: GaussianVarianceModel,
    nrows: usize,
    ncols: usize,
    fit_opts: &BayesFitOptions,
    hmc: HmcOptions,
    workspace: &mut LaplaceWorkspace,
    coef_prior: &GaussianCoefficientPrior,
) -> Result<BayesFitResult, ProbError> {
    let dim = target.dim();
    let n_keep = fit_opts.n_draws;
    let total_draws = n_keep.saturating_mul(hmc.n_chains);
    // Ensure grad/step cover unconstrained dim (ncols or ncols+1).
    workspace.prepare(nrows, dim.max(ncols), total_draws.max(dim));

    let mut chain_samples = vec![0.0; hmc.n_chains * n_keep * dim];
    let mut stats = TransitionStats::default();
    let mut map = coef_prior.mean.to_vec();
    let mut best_lp = f64::NEG_INFINITY;

    let include_sigma2 = model.include_sigma2();
    let init_lambda = match model {
        GaussianVarianceModel::Known { .. } => None,
        GaussianVarianceModel::InvGamma { shape, scale } => {
            // Prior mode of σ² under InvGamma is scale/(shape+1); use log of that.
            Some((scale / (shape + 1.0)).ln())
        }
    };

    for chain in 0..hmc.n_chains {
        let mut rng = CausalRng::from_seed(
            fit_opts.seed ^ ((chain as u64).wrapping_add(1).wrapping_mul(0xD1B5_4A32_D192_ED03)),
        );
        let mut q = coef_prior.mean.to_vec();
        for qi in &mut q {
            *qi += 0.1 * standard_normal(&mut rng);
        }
        if let Some(lambda0) = init_lambda {
            q.push(lambda0 + 0.05 * standard_normal(&mut rng));
        }
        debug_assert_eq!(q.len(), dim);

        let mut step_size = hmc.step_size;
        let mut log_eps_bar = step_size.ln();
        let mut h_bar = 0.0;

        let mut grad = vec![0.0; dim];
        let lp_init = target.logp_and_grad(&q, &mut grad)?;
        let mut lp_curr = lp_init;

        let total_iters = hmc.n_warmup.saturating_add(n_keep);
        let mut kept = 0usize;
        for t in 0..total_iters {
            let step = hmc_step_target(
                target,
                &q,
                lp_curr,
                step_size,
                hmc.leapfrog_steps,
                hmc.mass,
                &mut rng,
            )?;
            let is_warmup = t < hmc.n_warmup;
            stats.record(&step, is_warmup);
            if step.accepted {
                q = step.state;
                lp_curr = step.logp;
                if lp_curr > best_lp {
                    best_lp = lp_curr;
                    map.copy_from_slice(&q[..ncols]);
                }
            }

            match t.cmp(&hmc.n_warmup) {
                std::cmp::Ordering::Less => {
                    dual_average_update(
                        &mut h_bar,
                        &mut log_eps_bar,
                        &mut step_size,
                        hmc,
                        t,
                        step.accept_prob,
                    );
                }
                std::cmp::Ordering::Equal => {
                    step_size = dual_average_finalize(log_eps_bar);
                }
                std::cmp::Ordering::Greater => {}
            }

            if t >= hmc.n_warmup {
                let base = (chain * n_keep + kept) * dim;
                chain_samples[base..base + dim].copy_from_slice(&q);
                kept += 1;
            }
        }
    }

    pack_and_gate_hmc(&chain_samples, hmc, n_keep, dim, ncols, include_sigma2, map, stats)
}

fn dual_average_update(
    h_bar: &mut f64,
    log_eps_bar: &mut f64,
    step_size: &mut f64,
    hmc: HmcOptions,
    t: usize,
    accept_prob: f64,
) {
    let m = (t + 1) as f64;
    let eta = 1.0 / (m + 10.0);
    *h_bar = (1.0 - eta) * *h_bar + eta * (hmc.target_accept - accept_prob);
    let log_eps = hmc.step_size.ln() - (m.sqrt() / 0.05) * *h_bar;
    *step_size = log_eps.exp().clamp(1e-6, 0.5);
    let kappa = m.powf(-0.75);
    *log_eps_bar = kappa * log_eps + (1.0 - kappa) * *log_eps_bar;
}

fn dual_average_finalize(log_eps_bar: f64) -> f64 {
    log_eps_bar.exp().clamp(1e-6, 0.5)
}

fn pack_and_gate_hmc(
    chain_samples: &[f64],
    hmc: HmcOptions,
    n_keep: usize,
    state_dim: usize,
    ncols: usize,
    include_sigma2: bool,
    map: Vec<f64>,
    stats: TransitionStats,
) -> Result<BayesFitResult, ProbError> {
    let total_draws = n_keep.saturating_mul(hmc.n_chains);
    let n_out = if include_sigma2 { ncols + 1 } else { ncols };
    let mut values = vec![0.0; total_draws * n_out];

    for chain in 0..hmc.n_chains {
        for d in 0..n_keep {
            let src = (chain * n_keep + d) * state_dim;
            let dest_draw = chain * n_keep + d;
            for q in 0..ncols {
                values[q * total_draws + dest_draw] = chain_samples[src + q];
            }
            if include_sigma2 {
                let lambda = chain_samples[src + ncols];
                values[ncols * total_draws + dest_draw] = lambda.exp();
            }
        }
    }

    // Diagnostics on unconstrained state (includes λ, not skewed σ²).
    let ess_bulk = min_bulk_ess(chain_samples, hmc.n_chains, n_keep, state_dim);
    let ess_tail = min_tail_ess(chain_samples, hmc.n_chains, n_keep, state_dim);
    let rhat_max = max_split_rhat(chain_samples, hmc.n_chains, n_keep, state_dim);
    let moved = all_chains_moved(chain_samples, hmc.n_chains, n_keep, state_dim);
    let mean_accept = stats.mean_accept_prob();
    let max_dh = stats.max_abs_delta_h;

    let mut diagnostics = InferenceDiagnostics {
        converged: false,
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
        ess_bulk_min: Some(ess_bulk),
        ess_tail_min: Some(ess_tail),
        rhat_max: Some(rhat_max),
        n_divergences: Some(stats.n_postwarmup_divergences),
        mean_accept_prob: Some(mean_accept),
        n_warmup_divergences: Some(stats.n_warmup_divergences),
        n_postwarmup_divergences: Some(stats.n_postwarmup_divergences),
        max_abs_delta_h: Some(max_dh),
        all_chains_moved: Some(moved),
    };
    diagnostics.converged = diagnostics.mcmc_publication_ok();

    if !diagnostics.allows_posterior() {
        return Err(ProbError::MissingDiagnostics {
            message: "HMC posterior refused without ESS/R-hat/divergence diagnostics",
        });
    }

    let schema = if include_sigma2 {
        let mut q: Vec<_> = (0..ncols)
            .map(|i| PosteriorQuantityKind::Coefficient { index: i, name: None })
            .collect();
        q.push(PosteriorQuantityKind::ResidualVariance);
        PosteriorSchema { quantities: Arc::from(q) }
    } else {
        PosteriorSchema::coefficients(ncols)
    };

    let draws = PosteriorDraws::from_column_major(schema, total_draws, values)?;
    Ok(BayesFitResult { draws, map, diagnostics, cov: None })
}

fn hmc_step_target(
    target: &mut dyn PosteriorTarget,
    q0: &[f64],
    lp_old: f64,
    step_size: f64,
    leapfrog_steps: u32,
    mass: f64,
    rng: &mut CausalRng,
) -> Result<HmcStepResult, ProbError> {
    let dim = q0.len();
    let mut q = q0.to_vec();
    let mut p = vec![0.0; dim];
    let mut p0_energy = 0.0;
    for i in 0..dim {
        p[i] = mass.sqrt() * standard_normal(rng);
        p0_energy += 0.5 * p[i] * p[i] / mass;
    }

    let mut grad = vec![0.0; dim];
    // ∇U = -∇ log π
    let _ = target.logp_and_grad(&q, &mut grad)?;
    for i in 0..dim {
        p[i] -= 0.5 * step_size * (-grad[i]);
    }

    let mut divergent = false;
    for step in 0..leapfrog_steps {
        for i in 0..dim {
            q[i] += step_size * p[i] / mass;
            if !q[i].is_finite() {
                divergent = true;
                break;
            }
        }
        if divergent {
            break;
        }
        match target.logp_and_grad(&q, &mut grad) {
            Ok(_) => {}
            Err(_) => {
                divergent = true;
                break;
            }
        }
        let last = step + 1 == leapfrog_steps;
        let scale = if last { 0.5 } else { 1.0 };
        for i in 0..dim {
            p[i] -= scale * step_size * (-grad[i]);
            if !p[i].is_finite() {
                divergent = true;
            }
        }
        if divergent {
            break;
        }
    }

    if divergent {
        return Ok(HmcStepResult {
            state: q0.to_vec(),
            logp: lp_old,
            accepted: false,
            accept_prob: 0.0,
            delta_h: 1_001.0,
            divergent: true,
        });
    }

    let lp_new = target.logp_and_grad(&q, &mut grad)?;
    let mut p_new_energy = 0.0;
    for i in 0..dim {
        p_new_energy += 0.5 * p[i] * p[i] / mass;
    }

    let h_old = -lp_old + p0_energy;
    let h_new = -lp_new + p_new_energy;
    let delta_h = h_new - h_old;
    let (divergent, accept_prob) = finalize_energy(delta_h);
    if divergent {
        return Ok(HmcStepResult {
            state: q0.to_vec(),
            logp: lp_old,
            accepted: false,
            accept_prob: 0.0,
            delta_h,
            divergent: true,
        });
    }
    let accepted = rng.next_f64() < accept_prob;
    if accepted {
        Ok(HmcStepResult {
            state: q,
            logp: lp_new,
            accepted: true,
            accept_prob,
            delta_h,
            divergent: false,
        })
    } else {
        Ok(HmcStepResult {
            state: q0.to_vec(),
            logp: lp_old,
            accepted: false,
            accept_prob,
            delta_h,
            divergent: false,
        })
    }
}

fn hmc_step_glm(
    likelihood: BayesLikelihood,
    design: BayesDesignRef<'_>,
    coef_prior: &GaussianCoefficientPrior,
    prec: &[f64],
    beta: &[f64],
    step_size: f64,
    leapfrog_steps: u32,
    mass: f64,
    lp_old: f64,
    workspace: &mut LaplaceWorkspace,
    rng: &mut CausalRng,
) -> Result<HmcStepResult, ProbError> {
    let ncols = beta.len();
    let nrows = design.nrows;
    let mut q = beta.to_vec();
    let mut p = vec![0.0; ncols];
    let mut p0_energy = 0.0;
    for i in 0..ncols {
        p[i] = mass.sqrt() * standard_normal(rng);
        p0_energy += 0.5 * p[i] * p[i] / mass;
    }

    let reject_divergent = || HmcStepResult {
        state: beta.to_vec(),
        logp: lp_old,
        accepted: false,
        accept_prob: 0.0,
        delta_h: 1_001.0,
        divergent: true,
    };

    let mut grad = vec![0.0; ncols];
    match neg_log_posterior_grad(likelihood, design, coef_prior, prec, &q, &mut grad, workspace) {
        Ok(()) => {}
        Err(ProbError::Numerical { .. }) => return Ok(reject_divergent()),
        Err(e) => return Err(e),
    }
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
        match neg_log_posterior_grad(likelihood, design, coef_prior, prec, &q, &mut grad, workspace)
        {
            Ok(()) => {}
            Err(ProbError::Numerical { .. }) => {
                divergent = true;
                break;
            }
            Err(e) => return Err(e),
        }
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
        return Ok(reject_divergent());
    }

    let lp_new = match log_posterior_value(
        likelihood,
        design,
        &q,
        coef_prior,
        prec,
        &mut workspace.eta[..nrows],
        1.0,
    ) {
        Ok(v) => v,
        Err(ProbError::Numerical { .. }) => return Ok(reject_divergent()),
        Err(e) => return Err(e),
    };
    let mut p_new_energy = 0.0;
    for i in 0..ncols {
        p_new_energy += 0.5 * p[i] * p[i] / mass;
    }

    let h_old = -lp_old + p0_energy;
    let h_new = -lp_new + p_new_energy;
    let delta_h = h_new - h_old;
    let (divergent, accept_prob) = finalize_energy(delta_h);
    if divergent {
        return Ok(HmcStepResult {
            state: beta.to_vec(),
            logp: lp_old,
            accepted: false,
            accept_prob: 0.0,
            delta_h,
            divergent: true,
        });
    }
    let accepted = rng.next_f64() < accept_prob;
    if accepted {
        Ok(HmcStepResult {
            state: q,
            logp: lp_new,
            accepted: true,
            accept_prob,
            delta_h,
            divergent: false,
        })
    } else {
        Ok(HmcStepResult {
            state: beta.to_vec(),
            logp: lp_old,
            accepted: false,
            accept_prob,
            delta_h,
            divergent: false,
        })
    }
}

fn neg_log_posterior_grad(
    likelihood: BayesLikelihood,
    design: BayesDesignRef<'_>,
    coef_prior: &GaussianCoefficientPrior,
    prec: &[f64],
    beta: &[f64],
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
        1.0,
    )?;
    for i in 0..ncols {
        let diff = beta[i] - coef_prior.mean[i];
        workspace.grad[i] -= prec[i] * diff;
        grad_out[i] = -workspace.grad[i];
    }
    Ok(())
}

/// Leapfrog integration used by reversibility tests (no Metropolis).
#[cfg(test)]
fn leapfrog_trajectory(
    target: &mut dyn PosteriorTarget,
    q0: &[f64],
    p0: &[f64],
    step_size: f64,
    leapfrog_steps: u32,
    mass: f64,
) -> Result<(Vec<f64>, Vec<f64>), ProbError> {
    let dim = q0.len();
    let mut q = q0.to_vec();
    let mut p = p0.to_vec();
    let mut grad = vec![0.0; dim];
    let _ = target.logp_and_grad(&q, &mut grad)?;
    for i in 0..dim {
        p[i] -= 0.5 * step_size * (-grad[i]);
    }
    for step in 0..leapfrog_steps {
        for i in 0..dim {
            q[i] += step_size * p[i] / mass;
        }
        let _ = target.logp_and_grad(&q, &mut grad)?;
        let last = step + 1 == leapfrog_steps;
        let scale = if last { 0.5 } else { 1.0 };
        for i in 0..dim {
            p[i] -= scale * step_size * (-grad[i]);
        }
    }
    Ok((q, p))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conjugate::fit_conjugate_gaussian;
    use crate::prior::{InvGammaPrior, PriorSpec};

    fn hmc_opts(n_warmup: usize, n_chains: usize) -> HmcOptions {
        HmcOptions {
            n_chains,
            n_warmup,
            leapfrog_steps: 12,
            step_size: 0.08,
            target_accept: 0.8,
            mass: 1.0,
        }
    }

    #[test]
    fn energy_error_above_threshold_is_divergent() {
        let (div, ap) = finalize_energy(1_001.0);
        assert!(div);
        assert_eq!(ap, 0.0);
        let (div2, ap2) = finalize_energy(0.5);
        assert!(!div2);
        assert!((ap2 - (-0.5_f64).exp()).abs() < 1e-12);
        let (div3, ap3) = finalize_energy(f64::NAN);
        assert!(div3);
        assert_eq!(ap3, 0.0);
    }

    #[test]
    fn poisson_overflow_trajectory_is_divergent() {
        let n = 4;
        let x = vec![1.0; n];
        let y = vec![1.0; n];
        let prior = PriorSet::weakly_informative(1);
        let coef = prior.gaussian_coefficients().unwrap().clone();
        let prec = coef.precision();
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols: 1,
            y: &y,
            weights: None,
            offsets: None,
        };
        let mut ws = LaplaceWorkspace::default();
        ws.prepare(n, 1, 8);
        let mut rng = CausalRng::from_seed(9);
        let beta = [0.0_f64];
        let lp_old = log_posterior_value(
            BayesLikelihood::PoissonLog,
            design,
            &beta,
            &coef,
            &prec,
            &mut ws.eta[..n],
            1.0,
        )
        .unwrap();
        let step = hmc_step_glm(
            BayesLikelihood::PoissonLog,
            design,
            &coef,
            &prec,
            &beta,
            50.0,
            20,
            1.0,
            lp_old,
            &mut ws,
            &mut rng,
        )
        .unwrap();
        assert!(step.divergent);
        assert!(!step.accepted);
    }

    #[test]
    fn warmup_divergences_do_not_count_as_postwarmup() {
        let mut stats = TransitionStats::default();
        let warm = HmcStepResult {
            state: vec![0.0],
            logp: 0.0,
            accepted: false,
            accept_prob: 0.0,
            delta_h: 1_001.0,
            divergent: true,
        };
        let ok = HmcStepResult {
            state: vec![0.0],
            logp: 0.0,
            accepted: true,
            accept_prob: 0.7,
            delta_h: 0.1,
            divergent: false,
        };
        stats.record(&warm, true);
        stats.record(&ok, false);
        assert_eq!(stats.n_warmup_divergences, 1);
        assert_eq!(stats.n_postwarmup_divergences, 0);
        assert!((stats.mean_accept_prob() - 0.35).abs() < 1e-12);
    }

    #[test]
    fn postwarmup_divergence_fails_publication_gate() {
        let mut d = InferenceDiagnostics {
            converged: true,
            iterations: 400,
            grad_inf_norm: 0.0,
            hessian_condition: f64::NAN,
            factorization: HessianFactorization::Mcmc,
            separation_warning: false,
            notes: Vec::new(),
            backend_id: Arc::from("hmc"),
            n_chains: Some(4),
            n_warmup: Some(100),
            ess_bulk_min: Some(200.0),
            ess_tail_min: Some(150.0),
            rhat_max: Some(1.0),
            n_divergences: Some(1),
            mean_accept_prob: Some(0.8),
            n_warmup_divergences: Some(0),
            n_postwarmup_divergences: Some(1),
            max_abs_delta_h: Some(0.2),
            all_chains_moved: Some(true),
        };
        assert!(!d.allows_posterior());
        d.n_postwarmup_divergences = Some(0);
        d.n_divergences = Some(0);
        d.converged = d.mcmc_publication_ok();
        assert!(d.allows_posterior());
    }

    #[test]
    fn hmc_gaussian_recovers_slope() {
        let n = 80;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for r in 0..n {
            let xi = (r as f64 - 40.0) * 0.05;
            x[r] = 1.0;
            x[n + r] = xi;
            y[r] = 0.5 + 1.5 * xi + ((r % 5) as f64 - 2.0) * 0.2;
        }
        let prior = PriorSet {
            specs: vec![
                PriorSpec::GaussianCoefficients(
                    GaussianCoefficientPrior::shared(2, 0.0, 25.0).unwrap(),
                ),
                PriorSpec::KnownResidualVariance(1.0),
            ],
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
        let fit_opts = BayesFitOptions { n_draws: 800, seed: 42, max_iter: 50, grad_tol: 1e-8 };
        let hmc = HmcOptions {
            n_chains: 4,
            n_warmup: 800,
            leapfrog_steps: 12,
            step_size: 0.08,
            target_accept: 0.8,
            mass: 1.0,
        };
        let fit =
            fit_hmc_glm(BayesLikelihood::GaussianIdentity, design, &prior, &fit_opts, hmc, &mut ws)
                .expect("hmc fit");
        assert!(fit.diagnostics.allows_posterior());
        assert!(fit.diagnostics.rhat_max.unwrap() <= 1.01);
        assert!(fit.diagnostics.ess_bulk_min.unwrap() >= 100.0);
        assert!(fit.diagnostics.ess_tail_min.unwrap() >= 100.0);
        assert_eq!(fit.diagnostics.n_postwarmup_divergences, Some(0));
        assert_eq!(fit.diagnostics.all_chains_moved, Some(true));
        assert!((fit.map[1] - 1.5).abs() < 0.4, "map slope {}", fit.map[1]);
        assert_eq!(fit.draws.n_draws, 3200);
        assert_eq!(fit.draws.schema.n_quantities(), 2);
    }

    #[test]
    fn hmc_known_sigma2_one_predictor_matches_conjugate() {
        let n = 30;
        let mut x = vec![0.0; n];
        let mut y = vec![0.0; n];
        for r in 0..n {
            x[r] = 1.0;
            y[r] = 2.0 + ((r % 4) as f64 - 1.5) * 0.05;
        }
        let prior = PriorSet {
            specs: vec![
                PriorSpec::GaussianCoefficients(
                    GaussianCoefficientPrior::shared(1, 0.0, 4.0).unwrap(),
                ),
                PriorSpec::KnownResidualVariance(0.16),
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
        let fit_opts = BayesFitOptions { n_draws: 600, seed: 5, max_iter: 50, grad_tol: 1e-8 };
        let conj = fit_conjugate_gaussian(design, &prior, &fit_opts, &mut ws).unwrap();
        let hmc = hmc_opts(600, 4);
        let fit =
            fit_hmc_glm(BayesLikelihood::GaussianIdentity, design, &prior, &fit_opts, hmc, &mut ws)
                .unwrap();
        let n_draws = fit.draws.n_draws;
        let mut mean = 0.0;
        for d in 0..n_draws {
            mean += fit.draws.values[d];
        }
        mean /= n_draws as f64;
        let mut c_mean = 0.0;
        for d in 0..conj.draws.n_draws {
            c_mean += conj.draws.values[d];
        }
        c_mean /= conj.draws.n_draws as f64;
        assert!((mean - c_mean).abs() < 0.12, "mean hmc={mean} conj={c_mean}");
    }

    #[test]
    fn hmc_known_sigma2_matches_conjugate_moments() {
        let n = 40;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for r in 0..n {
            let xi = (r as f64 - 20.0) * 0.1;
            x[r] = 1.0;
            x[n + r] = xi;
            y[r] = 1.0 + 2.0 * xi + ((r % 7) as f64 - 3.0) * 0.02;
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
        let fit_opts = BayesFitOptions { n_draws: 800, seed: 3, max_iter: 50, grad_tol: 1e-8 };
        let conj = fit_conjugate_gaussian(design, &prior, &fit_opts, &mut ws).unwrap();
        let hmc = hmc_opts(800, 4);
        let fit =
            fit_hmc_glm(BayesLikelihood::GaussianIdentity, design, &prior, &fit_opts, hmc, &mut ws)
                .unwrap();

        let n_draws = fit.draws.n_draws;
        for j in 0..2 {
            let mut mean = 0.0;
            let mut m2 = 0.0;
            for d in 0..n_draws {
                let v = fit.draws.values[j * n_draws + d];
                mean += v;
                m2 += v * v;
            }
            mean /= n_draws as f64;
            let var = m2 / n_draws as f64 - mean * mean;
            let mut c_mean = 0.0;
            let mut c_m2 = 0.0;
            for d in 0..conj.draws.n_draws {
                let v = conj.draws.values[j * conj.draws.n_draws + d];
                c_mean += v;
                c_m2 += v * v;
            }
            c_mean /= conj.draws.n_draws as f64;
            let c_var = c_m2 / conj.draws.n_draws as f64 - c_mean * c_mean;
            assert!((mean - c_mean).abs() < 0.15, "coef {j} mean hmc={mean} conj={c_mean}");
            assert!(
                (var - c_var).abs() / c_var.max(1e-6) < 0.5,
                "coef {j} var hmc={var} conj={c_var}"
            );
        }
    }

    #[test]
    fn hmc_nig_matches_conjugate_moments() {
        let n = 50;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for r in 0..n {
            let xi = (r as f64) * 0.08;
            x[r] = 1.0;
            x[n + r] = xi;
            y[r] = 0.5 + 1.2 * xi + ((r % 5) as f64 - 2.0) * 0.1;
        }
        let prior = PriorSet {
            specs: vec![
                PriorSpec::GaussianCoefficients(
                    GaussianCoefficientPrior::shared(2, 0.0, 10.0).unwrap(),
                ),
                PriorSpec::ResidualInvGamma(InvGammaPrior { shape: 2.0, scale: 1.0 }),
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
        let fit_opts = BayesFitOptions { n_draws: 1000, seed: 17, max_iter: 50, grad_tol: 1e-8 };
        let conj = fit_conjugate_gaussian(design, &prior, &fit_opts, &mut ws).unwrap();
        let hmc = HmcOptions {
            n_chains: 4,
            n_warmup: 1000,
            leapfrog_steps: 15,
            step_size: 0.04,
            target_accept: 0.8,
            mass: 1.0,
        };
        let fit =
            fit_hmc_glm(BayesLikelihood::GaussianIdentity, design, &prior, &fit_opts, hmc, &mut ws)
                .unwrap();
        assert_eq!(fit.draws.schema.n_quantities(), 3);
        assert!(matches!(fit.draws.schema.quantities[2], PosteriorQuantityKind::ResidualVariance));

        let n_draws = fit.draws.n_draws;
        for j in 0..2 {
            let mut mean = 0.0;
            for d in 0..n_draws {
                mean += fit.draws.values[j * n_draws + d];
            }
            mean /= n_draws as f64;
            let mut c_mean = 0.0;
            for d in 0..conj.draws.n_draws {
                c_mean += conj.draws.values[j * conj.draws.n_draws + d];
            }
            c_mean /= conj.draws.n_draws as f64;
            assert!((mean - c_mean).abs() < 0.25, "coef {j} mean hmc={mean} conj={c_mean}");
        }
        let mut s_mean = 0.0;
        for d in 0..n_draws {
            s_mean += fit.draws.values[2 * n_draws + d];
        }
        s_mean /= n_draws as f64;
        let mut c_s = 0.0;
        for d in 0..conj.draws.n_draws {
            c_s += conj.draws.values[2 * conj.draws.n_draws + d];
        }
        c_s /= conj.draws.n_draws as f64;
        assert!((s_mean - c_s).abs() / c_s.max(1e-6) < 0.6, "sigma2 mean hmc={s_mean} conj={c_s}");
    }

    #[test]
    fn leapfrog_is_reversible() {
        let n = 8;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for r in 0..n {
            x[r] = 1.0;
            x[n + r] = r as f64;
            y[r] = 1.0 + 0.5 * (r as f64);
        }
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols: 2,
            y: &y,
            weights: None,
            offsets: None,
        };
        let prior = GaussianCoefficientPrior::shared(2, 0.0, 4.0).unwrap();
        let mut target = gaussian_target_from_model(
            design,
            prior,
            GaussianVarianceModel::InvGamma { shape: 2.0, scale: 1.0 },
        )
        .unwrap();
        let q0 = vec![0.5, 0.4, 0.0];
        let p0 = vec![0.3, -0.2, 0.1];
        let step = 0.01;
        let l = 20;
        let mass = 1.0;
        let (q1, p1) = leapfrog_trajectory(&mut target, &q0, &p0, step, l, mass).unwrap();
        let p_neg: Vec<_> = p1.iter().map(|v| -v).collect();
        let (q2, p2) = leapfrog_trajectory(&mut target, &q1, &p_neg, step, l, mass).unwrap();
        for i in 0..3 {
            assert!((q2[i] - q0[i]).abs() < 1e-8, "q[{i}] {} vs {}", q2[i], q0[i]);
            assert!((p2[i] + p0[i]).abs() < 1e-8, "p[{i}] {} vs {}", p2[i], -p0[i]);
        }
    }
}
