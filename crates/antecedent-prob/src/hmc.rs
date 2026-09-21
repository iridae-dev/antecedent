//! Native Hamiltonian Monte Carlo for Bayesian GLMs.
//!
//! Leapfrog HMC with dual-averaging step-size adaptation during warmup.
//! Multi-chain draws are columnar; ESS / R-hat / divergence counts gate publication.
//! Chains are independent (one stream each from the library's stream mixer) and run
//! on the execution context's threads when one is supplied.
//!
//! GaussianIdentity uses a fixed prior-driven target (known σ² or joint
//! `(β, log σ²)` under an InvGamma residual prior). Other likelihoods keep the
//! GLM score path.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::too_many_lines
)]

use std::sync::Arc;

use antecedent_core::{CausalRng, ExecutionContext, RngFactory, StreamDomain};
use antecedent_kernels::standard_normal;

use crate::backend::{
    BayesDesignRef, BayesFitOptions, BayesFitResult, BayesLikelihood, InferenceBackend,
    LaplaceWorkspace,
};
use crate::diagnostics::{HessianFactorization, InferenceDiagnostics};
use crate::error::ProbError;
use crate::gaussian_target::{PosteriorTarget, gaussian_target_from_model};
use crate::likelihood_terms::{accumulate_likelihood, log_posterior_value, validate_design};
use crate::mcmc_stats::{all_chains_moved, mcmc_summary};
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
        ctx: &ExecutionContext,
    ) -> Result<BayesFitResult, ProbError> {
        prior.validate()?;
        fit_hmc_impl(likelihood, design, prior, options, self.options, workspace, Some(ctx))
    }
}

/// Result of one HMC transition.
///
/// Accepted position lives in `LaplaceWorkspace::q`; callers copy it on accept.
struct HmcStepResult {
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

    fn merge(&mut self, other: Self) {
        self.n_warmup_divergences =
            self.n_warmup_divergences.saturating_add(other.n_warmup_divergences);
        self.n_postwarmup_divergences =
            self.n_postwarmup_divergences.saturating_add(other.n_postwarmup_divergences);
        self.accept_prob_sum += other.accept_prob_sum;
        self.n_transitions = self.n_transitions.saturating_add(other.n_transitions);
        self.max_abs_delta_h = self.max_abs_delta_h.max(other.max_abs_delta_h);
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

/// Step-size bounds for the adapted leapfrog step.
const MIN_STEP: f64 = 1e-6;
const MAX_STEP: f64 = 0.5;

/// Post-warmup draws and transition counts from one chain.
struct ChainRun {
    /// `n_keep × state_dim` retained states, draw-major.
    samples: Vec<f64>,
    stats: TransitionStats,
    /// Step size the retained draws were generated with.
    final_step: f64,
}

/// One stream per chain from the library's stream mixer, so chain streams are
/// unrelated offsets rather than ad-hoc xor-derived seeds.
fn chain_rng(seed: u64, chain: usize) -> CausalRng {
    RngFactory::from_seed(seed).stream_for(StreamDomain::Bayesian, chain as u64)
}

/// Run multi-chain HMC and return columnar post-warmup draws.
///
/// Chains run serially on `workspace`; [`HmcGlmBackend`] runs them on the
/// execution context's threads with a workspace per chain and identical draws.
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
    fit_hmc_impl(likelihood, design, prior, fit_opts, hmc, workspace, None)
}

fn fit_hmc_impl(
    likelihood: BayesLikelihood,
    design: BayesDesignRef<'_>,
    prior: &PriorSet,
    fit_opts: &BayesFitOptions,
    hmc: HmcOptions,
    workspace: &mut LaplaceWorkspace,
    ctx: Option<&ExecutionContext>,
) -> Result<BayesFitResult, ProbError> {
    let nrows = design.nrows;
    let ncols = design.ncols;
    validate_design(likelihood, design)?;
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

    let gaussian_model = if likelihood == BayesLikelihood::GaussianIdentity {
        Some(GaussianVarianceModel::from_prior_set(prior)?)
    } else {
        None
    };
    let include_sigma2 = gaussian_model.is_some_and(super::prior::GaussianVarianceModel::include_sigma2);
    let dim = ncols + usize::from(include_sigma2);
    // GLM has no residual σ²; absolute prior precision is V0^{-1} at σ² ≡ 1.
    let prec =
        if gaussian_model.is_none() { coef_prior.absolute_precision(1.0)? } else { Vec::new() };
    let n_keep = fit_opts.n_draws;

    let run = |chain: usize, ws: &mut LaplaceWorkspace| -> Result<ChainRun, ProbError> {
        match gaussian_model {
            Some(model) => run_gaussian_chain(chain, design, &coef_prior, model, fit_opts, hmc, ws),
            None => run_glm_chain(chain, likelihood, design, &coef_prior, &prec, fit_opts, hmc, ws),
        }
    };
    let prepare = |ws: &mut LaplaceWorkspace| ws.prepare(nrows, dim.max(ncols), dim);
    let runs: Vec<ChainRun> = if let Some(ctx) = ctx {
        ctx.map_indexed(hmc.n_chains, |chain, _| {
            let mut ws = LaplaceWorkspace::default();
            prepare(&mut ws);
            run(chain, &mut ws)
        })?
    } else {
        prepare(workspace);
        (0..hmc.n_chains).map(|chain| run(chain, workspace)).collect::<Result<_, _>>()?
    };

    let mut chain_samples = Vec::with_capacity(hmc.n_chains * n_keep * dim);
    let mut stats = TransitionStats::default();
    let (mut step_lo, mut step_hi) = (f64::INFINITY, 0.0_f64);
    for r in runs {
        chain_samples.extend_from_slice(&r.samples);
        stats.merge(r.stats);
        step_lo = step_lo.min(r.final_step);
        step_hi = step_hi.max(r.final_step);
    }

    pack_and_gate_hmc(
        &chain_samples,
        hmc,
        n_keep,
        dim,
        ncols,
        include_sigma2,
        stats,
        (step_lo, step_hi),
    )
}

fn run_glm_chain(
    chain: usize,
    likelihood: BayesLikelihood,
    design: BayesDesignRef<'_>,
    coef_prior: &GaussianCoefficientPrior,
    prec: &[f64],
    fit_opts: &BayesFitOptions,
    hmc: HmcOptions,
    ws: &mut LaplaceWorkspace,
) -> Result<ChainRun, ProbError> {
    let nrows = design.nrows;
    let ncols = design.ncols;
    let n_keep = fit_opts.n_draws;
    let mut rng = chain_rng(fit_opts.seed, chain);
    let mut beta = coef_prior.mean.to_vec();
    for bi in &mut beta {
        *bi += 0.1 * standard_normal(&mut rng);
    }
    let mut step_size = hmc.step_size;
    let mut log_eps_bar = step_size.ln();
    let mut h_bar = 0.0;

    // The current state's log-posterior and ∇U are carried across transitions:
    // after an accept they are exactly the trajectory's final-state values, after
    // a reject they are unchanged, so no transition re-sweeps the data for them.
    let mut lp_curr = log_posterior_value(
        likelihood,
        design,
        &beta,
        coef_prior,
        prec,
        &mut ws.eta[..nrows],
        1.0,
    )?;
    let mut gradu_curr = vec![0.0; ncols];
    neg_log_posterior_grad_slices(
        likelihood,
        design,
        coef_prior,
        prec,
        &beta,
        &mut ws.grad[..ncols],
        &mut ws.neg_hessian[..ncols * ncols],
        &mut ws.eta[..nrows],
        &mut ws.work_w[..nrows],
        &mut gradu_curr,
    )?;

    let mut samples = vec![0.0; n_keep * ncols];
    let mut stats = TransitionStats::default();
    let mut kept = 0usize;
    for t in 0..hmc.n_warmup.saturating_add(n_keep) {
        if t == hmc.n_warmup {
            // Warmup is over before this transition runs: every retained draw
            // uses the averaged step, not the last exploratory one.
            step_size = dual_average_finalize(log_eps_bar);
        }
        let step = hmc_step_glm(
            likelihood,
            design,
            coef_prior,
            prec,
            &beta,
            &gradu_curr,
            step_size,
            hmc.leapfrog_steps,
            hmc.mass,
            lp_curr,
            ws,
            &mut rng,
        )?;
        stats.record(&step, t < hmc.n_warmup);
        if step.accepted {
            beta.copy_from_slice(&ws.q[..ncols]);
            gradu_curr.copy_from_slice(&ws.step[..ncols]);
            lp_curr = step.logp;
        }
        if t < hmc.n_warmup {
            dual_average_update(
                &mut h_bar,
                &mut log_eps_bar,
                &mut step_size,
                hmc,
                t,
                step.accept_prob,
            );
        } else {
            samples[kept * ncols..(kept + 1) * ncols].copy_from_slice(&beta);
            kept += 1;
        }
    }
    Ok(ChainRun { samples, stats, final_step: step_size })
}

fn run_gaussian_chain(
    chain: usize,
    design: BayesDesignRef<'_>,
    coef_prior: &GaussianCoefficientPrior,
    model: GaussianVarianceModel,
    fit_opts: &BayesFitOptions,
    hmc: HmcOptions,
    ws: &mut LaplaceWorkspace,
) -> Result<ChainRun, ProbError> {
    let ncols = design.ncols;
    let mut target = gaussian_target_from_model(design, coef_prior.clone(), model)?;
    let dim = target.dim();
    let n_keep = fit_opts.n_draws;
    let mut rng = chain_rng(fit_opts.seed, chain);
    let mut q = coef_prior.mean.to_vec();
    for qi in &mut q {
        *qi += 0.1 * standard_normal(&mut rng);
    }
    if let GaussianVarianceModel::InvGamma { shape, scale } = model {
        // Prior mode of σ² under InvGamma is scale/(shape+1); use log of that.
        let lambda0 = (scale / (shape + 1.0)).ln();
        q.push(lambda0 + 0.05 * standard_normal(&mut rng));
    }
    debug_assert_eq!(q.len(), dim);
    debug_assert!(dim >= ncols);

    let mut step_size = hmc.step_size;
    let mut log_eps_bar = step_size.ln();
    let mut h_bar = 0.0;

    // Log-posterior and its gradient at the current state, carried across transitions.
    let mut lp_curr = target.logp_and_grad(&q, &mut ws.grad[..dim])?;
    let mut grad_curr = ws.grad[..dim].to_vec();

    let mut samples = vec![0.0; n_keep * dim];
    let mut stats = TransitionStats::default();
    let mut kept = 0usize;
    for t in 0..hmc.n_warmup.saturating_add(n_keep) {
        if t == hmc.n_warmup {
            step_size = dual_average_finalize(log_eps_bar);
        }
        let step = hmc_step_target(
            &mut target,
            &q,
            lp_curr,
            &grad_curr,
            step_size,
            hmc.leapfrog_steps,
            hmc.mass,
            ws,
            &mut rng,
        )?;
        stats.record(&step, t < hmc.n_warmup);
        if step.accepted {
            q.copy_from_slice(&ws.q[..dim]);
            grad_curr.copy_from_slice(&ws.grad[..dim]);
            lp_curr = step.logp;
        }
        if t < hmc.n_warmup {
            dual_average_update(
                &mut h_bar,
                &mut log_eps_bar,
                &mut step_size,
                hmc,
                t,
                step.accept_prob,
            );
        } else {
            samples[kept * dim..(kept + 1) * dim].copy_from_slice(&q);
            kept += 1;
        }
    }
    Ok(ChainRun { samples, stats, final_step: step_size })
}

/// Nesterov dual averaging of the leapfrog step (Hoffman & Gelman 2014, alg. 5).
///
/// Iterates shrink toward `μ = ln(10 ε₀)`; the iterate is clamped to the usable
/// step range *before* it enters the average, so `ε̄` never exceeds a step that
/// was actually tried.
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
    let log_eps = ((10.0 * hmc.step_size).ln() - (m.sqrt() / 0.05) * *h_bar)
        .clamp(MIN_STEP.ln(), MAX_STEP.ln());
    *step_size = log_eps.exp();
    let kappa = m.powf(-0.75);
    *log_eps_bar = kappa * log_eps + (1.0 - kappa) * *log_eps_bar;
}

fn dual_average_finalize(log_eps_bar: f64) -> f64 {
    log_eps_bar.exp().clamp(MIN_STEP, MAX_STEP)
}

fn pack_and_gate_hmc(
    chain_samples: &[f64],
    hmc: HmcOptions,
    n_keep: usize,
    state_dim: usize,
    ncols: usize,
    include_sigma2: bool,
    stats: TransitionStats,
    step_range: (f64, f64),
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

    // Point estimate: posterior mean of the retained draws. The best visited
    // state is a noisy mode estimate (typical-set draws sit ~d/2 nats below the
    // mode) and can come from warmup, so it is not reported as a mode.
    let map: Vec<f64> = (0..ncols)
        .map(|q| {
            values[q * total_draws..(q + 1) * total_draws].iter().sum::<f64>() / total_draws as f64
        })
        .collect();

    // Diagnostics on unconstrained state (includes λ, not skewed σ²).
    let summary = mcmc_summary(chain_samples, hmc.n_chains, n_keep, state_dim);
    let (ess_bulk, ess_tail, rhat_max) =
        (summary.min_bulk_ess, summary.min_tail_ess, summary.max_rhat);
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
            "hmc chains={} warmup={} L={} step_size=[{:.3e}, {:.3e}]",
            hmc.n_chains, hmc.n_warmup, hmc.leapfrog_steps, step_range.0, step_range.1
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
        // The unit-mass sampler's step is bounded; an adapted step pinned at a
        // bound means the posterior scale is outside what it can traverse.
        let at_bound =
            step_range.0 <= MIN_STEP * (1.0 + 1e-9) || step_range.1 >= MAX_STEP * (1.0 - 1e-9);
        let scale_hint = if at_bound {
            format!(
                "; adapted step size reached its bound ({:.3e}..{:.3e}), so the posterior \
                 scale is outside the range this unit-mass sampler can traverse: standardise \
                 the design columns",
                step_range.0, step_range.1
            )
        } else {
            String::new()
        };
        return Err(ProbError::MissingDiagnostics {
            message: format!(
                "HMC posterior refused: rhat={:?} ess_bulk={:?} ess_tail={:?} postwarmup_div={:?} moved={:?} accept={:?}{scale_hint}",
                diagnostics.rhat_max,
                diagnostics.ess_bulk_min,
                diagnostics.ess_tail_min,
                diagnostics.n_postwarmup_divergences,
                diagnostics.all_chains_moved,
                diagnostics.mean_accept_prob
            ),
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
    grad_old: &[f64],
    step_size: f64,
    leapfrog_steps: u32,
    mass: f64,
    workspace: &mut LaplaceWorkspace,
    rng: &mut CausalRng,
) -> Result<HmcStepResult, ProbError> {
    let dim = q0.len();
    workspace.q[..dim].copy_from_slice(q0);
    let mut p0_energy = 0.0;
    for i in 0..dim {
        workspace.p[i] = mass.sqrt() * standard_normal(rng);
        p0_energy += 0.5 * workspace.p[i] * workspace.p[i] / mass;
    }

    // ∇U = -∇ log π at q0, supplied by the caller (carried from the previous
    // transition rather than re-swept).
    for i in 0..dim {
        workspace.p[i] -= 0.5 * step_size * (-grad_old[i]);
    }

    let mut divergent = false;
    // Log-density at the trajectory's final state: the last leapfrog evaluation.
    let mut lp_last = lp_old;
    for step in 0..leapfrog_steps {
        for i in 0..dim {
            workspace.q[i] += step_size * workspace.p[i] / mass;
            if !workspace.q[i].is_finite() {
                divergent = true;
                break;
            }
        }
        if divergent {
            break;
        }
        match target.logp_and_grad(&workspace.q[..dim], &mut workspace.grad[..dim]) {
            Ok(v) => lp_last = v,
            Err(_) => {
                divergent = true;
                break;
            }
        }
        let last = step + 1 == leapfrog_steps;
        let scale = if last { 0.5 } else { 1.0 };
        for i in 0..dim {
            workspace.p[i] -= scale * step_size * (-workspace.grad[i]);
            if !workspace.p[i].is_finite() {
                divergent = true;
            }
        }
        if divergent {
            break;
        }
    }

    if divergent {
        return Ok(HmcStepResult {
            logp: lp_old,
            accepted: false,
            accept_prob: 0.0,
            delta_h: 1_001.0,
            divergent: true,
        });
    }

    let lp_new = lp_last;
    let mut p_new_energy = 0.0;
    for i in 0..dim {
        p_new_energy += 0.5 * workspace.p[i] * workspace.p[i] / mass;
    }

    let h_old = -lp_old + p0_energy;
    let h_new = -lp_new + p_new_energy;
    let delta_h = h_new - h_old;
    let (divergent, accept_prob) = finalize_energy(delta_h);
    if divergent {
        return Ok(HmcStepResult {
            logp: lp_old,
            accepted: false,
            accept_prob: 0.0,
            delta_h,
            divergent: true,
        });
    }
    let accepted = rng.next_f64() < accept_prob;
    if accepted {
        Ok(HmcStepResult { logp: lp_new, accepted: true, accept_prob, delta_h, divergent: false })
    } else {
        Ok(HmcStepResult { logp: lp_old, accepted: false, accept_prob, delta_h, divergent: false })
    }
}

fn hmc_step_glm(
    likelihood: BayesLikelihood,
    design: BayesDesignRef<'_>,
    coef_prior: &GaussianCoefficientPrior,
    prec: &[f64],
    beta: &[f64],
    gradu_old: &[f64],
    step_size: f64,
    leapfrog_steps: u32,
    mass: f64,
    lp_old: f64,
    workspace: &mut LaplaceWorkspace,
    rng: &mut CausalRng,
) -> Result<HmcStepResult, ProbError> {
    let ncols = beta.len();
    let nrows = design.nrows;
    workspace.q[..ncols].copy_from_slice(beta);
    let mut p0_energy = 0.0;
    for i in 0..ncols {
        workspace.p[i] = mass.sqrt() * standard_normal(rng);
        p0_energy += 0.5 * workspace.p[i] * workspace.p[i] / mass;
    }

    let reject_divergent = || HmcStepResult {
        logp: lp_old,
        accepted: false,
        accept_prob: 0.0,
        delta_h: 1_001.0,
        divergent: true,
    };

    // ∇U at the current state comes from the caller. `workspace.step` holds the
    // running ∇U afterwards, disjoint from `workspace.grad` (log-posterior score).
    for i in 0..ncols {
        workspace.p[i] -= 0.5 * step_size * gradu_old[i];
    }

    let mut divergent = false;
    for lf in 0..leapfrog_steps {
        for i in 0..ncols {
            workspace.q[i] += step_size * workspace.p[i] / mass;
            if !workspace.q[i].is_finite() {
                divergent = true;
                break;
            }
        }
        if divergent {
            break;
        }
        match neg_log_posterior_grad_slices(
            likelihood,
            design,
            coef_prior,
            prec,
            &workspace.q[..ncols],
            &mut workspace.grad[..ncols],
            &mut workspace.neg_hessian[..ncols * ncols],
            &mut workspace.eta[..nrows],
            &mut workspace.work_w[..nrows],
            &mut workspace.step[..ncols],
        ) {
            Ok(()) => {}
            Err(ProbError::Numerical { .. }) => {
                divergent = true;
                break;
            }
            Err(e) => return Err(e),
        }
        let last = lf + 1 == leapfrog_steps;
        let scale = if last { 0.5 } else { 1.0 };
        for i in 0..ncols {
            workspace.p[i] -= scale * step_size * workspace.step[i];
            if !workspace.p[i].is_finite() {
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
        &workspace.q[..ncols],
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
        p_new_energy += 0.5 * workspace.p[i] * workspace.p[i] / mass;
    }

    let h_old = -lp_old + p0_energy;
    let h_new = -lp_new + p_new_energy;
    let delta_h = h_new - h_old;
    let (divergent, accept_prob) = finalize_energy(delta_h);
    if divergent {
        return Ok(HmcStepResult {
            logp: lp_old,
            accepted: false,
            accept_prob: 0.0,
            delta_h,
            divergent: true,
        });
    }
    let accepted = rng.next_f64() < accept_prob;
    if accepted {
        Ok(HmcStepResult { logp: lp_new, accepted: true, accept_prob, delta_h, divergent: false })
    } else {
        Ok(HmcStepResult { logp: lp_old, accepted: false, accept_prob, delta_h, divergent: false })
    }
}

fn neg_log_posterior_grad_slices(
    likelihood: BayesLikelihood,
    design: BayesDesignRef<'_>,
    coef_prior: &GaussianCoefficientPrior,
    prec: &[f64],
    beta: &[f64],
    grad_lp: &mut [f64],
    neg_hessian: &mut [f64],
    eta: &mut [f64],
    work_w: &mut [f64],
    grad_out: &mut [f64],
) -> Result<(), ProbError> {
    accumulate_likelihood(likelihood, design, beta, grad_lp, neg_hessian, eta, work_w, 1.0, false)?;
    for i in 0..beta.len() {
        let diff = beta[i] - coef_prior.mean[i];
        grad_lp[i] -= prec[i] * diff;
        grad_out[i] = -grad_lp[i];
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

    /// Exact known-σ² Gaussian posterior `N(m, C)` with prior `N(0, v0·I)`:
    /// `C = (X'X/σ² + I/v0)⁻¹`, `m = C X'y/σ²` (column-major `x`, `p` columns).
    fn exact_known_sigma2_posterior(
        x: &[f64],
        y: &[f64],
        p: usize,
        v0: f64,
        sigma2: f64,
    ) -> (Vec<f64>, Vec<f64>) {
        let n = y.len();
        let mut a = vec![0.0; p * p];
        let mut b = vec![0.0; p];
        for i in 0..p {
            for j in 0..p {
                let mut acc = 0.0;
                for r in 0..n {
                    acc += x[i * n + r] * x[j * n + r];
                }
                a[i * p + j] = acc / sigma2 + if i == j { 1.0 / v0 } else { 0.0 };
            }
            b[i] = (0..n).map(|r| x[i * n + r] * y[r]).sum::<f64>() / sigma2;
        }
        let c = crate::linalg::invert_spd(&a, p).unwrap();
        let m = (0..p).map(|i| (0..p).map(|j| c[i * p + j] * b[j]).sum::<f64>()).collect();
        (m, c)
    }

    /// Sample mean and (population) variance of coefficient column `j`.
    fn column_moments(fit: &BayesFitResult, j: usize) -> (f64, f64) {
        let n = fit.draws.n_draws;
        let col = &fit.draws.values[j * n..(j + 1) * n];
        let mean = col.iter().sum::<f64>() / n as f64;
        let var = col.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / n as f64;
        (mean, var)
    }

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
        // ∇U at eta = 0: Σ(y − e^η) = 0 and the prior mean is 0.
        let gradu = [0.0_f64];
        let step = hmc_step_glm(
            BayesLikelihood::PoissonLog,
            design,
            &coef,
            &prec,
            &beta,
            &gradu,
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
            logp: 0.0,
            accepted: false,
            accept_prob: 0.0,
            delta_h: 1_001.0,
            divergent: true,
        };
        let ok = HmcStepResult {
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
            // Healthy for the 4 chains below: the gate requires ~100 bulk/tail ESS per chain.
            ess_bulk_min: Some(500.0),
            ess_tail_min: Some(450.0),
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
        let fit_opts = BayesFitOptions { n_draws: 2000, seed: 42, max_iter: 50, grad_tol: 1e-8 };
        // Longer schedule + milder steps: R̂≤1.01 is tight on 2-parameter
        // Gaussian HMC and was flaking on Linux CI around 1.012.
        let hmc = HmcOptions {
            n_chains: 4,
            n_warmup: 1500,
            leapfrog_steps: 16,
            step_size: 0.04,
            target_accept: 0.85,
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
        // The reported point estimate is the posterior mean of the retained
        // draws, so it must sit within Monte Carlo error of the exact posterior mean.
        let (m, c) = exact_known_sigma2_posterior(&x, &y, 2, 25.0, 1.0);
        let ess = fit.diagnostics.ess_bulk_min.unwrap();
        for j in 0..2 {
            let tol = 6.0 * (c[j * 2 + j] / ess).sqrt();
            assert!(
                (fit.map[j] - m[j]).abs() < tol,
                "map[{j}] {} vs exact {} tol {tol}",
                fit.map[j],
                m[j]
            );
            let (mean, _) = column_moments(&fit, j);
            assert!((fit.map[j] - mean).abs() < 1e-12, "map is the mean of the retained draws");
        }
        assert_eq!(fit.draws.n_draws, 8000);
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
        let hmc = hmc_opts(600, 4);
        let fit =
            fit_hmc_glm(BayesLikelihood::GaussianIdentity, design, &prior, &fit_opts, hmc, &mut ws)
                .unwrap();
        // Exact posterior: precision n/σ² + 1/v0, mean (Σy/σ²)/precision.
        let (m, c) = exact_known_sigma2_posterior(&x, &y, 1, 4.0, 0.16);
        let (mean, _) = column_moments(&fit, 0);
        let ess = fit.diagnostics.ess_bulk_min.unwrap();
        let tol = 6.0 * (c[0] / ess).sqrt();
        assert!((mean - m[0]).abs() < tol, "mean hmc={mean} exact={} tol={tol}", m[0]);
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
        let hmc = hmc_opts(800, 4);
        let fit =
            fit_hmc_glm(BayesLikelihood::GaussianIdentity, design, &prior, &fit_opts, hmc, &mut ws)
                .unwrap();

        // Exact known-σ² posterior; tolerances are Monte Carlo errors from the
        // reported ESS (mean: sd/√ESS; variance: relative √(2/ESS) for a Gaussian).
        let (m, c) = exact_known_sigma2_posterior(&x, &y, 2, 4.0, sigma2);
        let ess = fit.diagnostics.ess_bulk_min.unwrap();
        for j in 0..2 {
            let (mean, var) = column_moments(&fit, j);
            let sd = c[j * 2 + j].sqrt();
            assert!(
                (mean - m[j]).abs() < 6.0 * sd / ess.sqrt(),
                "coef {j} mean hmc={mean} exact={}",
                m[j]
            );
            let c_var = c[j * 2 + j];
            assert!(
                (var - c_var).abs() / c_var < 6.0 * (2.0 / ess).sqrt(),
                "coef {j} var hmc={var} exact={c_var}"
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

        // HMC mean vs the exact (iid) conjugate draws: the difference has
        // variance sd²/ESS_hmc + sd²/n_conj, so 6 of those SEs bounds it.
        let ess = fit.diagnostics.ess_bulk_min.unwrap();
        let n_draws = fit.draws.n_draws;
        let n_conj = conj.draws.n_draws;
        for j in 0..3 {
            let h = &fit.draws.values[j * n_draws..(j + 1) * n_draws];
            let c = &conj.draws.values[j * n_conj..(j + 1) * n_conj];
            let h_mean = h.iter().sum::<f64>() / n_draws as f64;
            let c_mean = c.iter().sum::<f64>() / n_conj as f64;
            let c_var = c.iter().map(|v| (v - c_mean).powi(2)).sum::<f64>() / (n_conj - 1) as f64;
            let tol = 6.0 * (c_var / ess + c_var / n_conj as f64).sqrt();
            assert!(
                (h_mean - c_mean).abs() < tol,
                "quantity {j} mean hmc={h_mean} conj={c_mean} tol={tol}"
            );
        }
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

    /// Standard-normal target that counts density/gradient sweeps.
    struct CountingTarget {
        calls: usize,
    }

    impl PosteriorTarget for CountingTarget {
        fn dim(&self) -> usize {
            2
        }

        fn logp_and_grad(&mut self, q: &[f64], grad: &mut [f64]) -> Result<f64, ProbError> {
            self.calls += 1;
            for i in 0..2 {
                grad[i] = -q[i];
            }
            Ok(-0.5 * (q[0] * q[0] + q[1] * q[1]))
        }
    }

    #[test]
    fn transition_costs_exactly_one_sweep_per_leapfrog_step() {
        let leapfrog = 7;
        let mut target = CountingTarget { calls: 0 };
        let mut ws = LaplaceWorkspace::default();
        ws.prepare(1, 2, 1);
        let mut rng = CausalRng::from_seed(11);
        let q0 = [0.3, -0.4];
        let lp_old = -0.5 * (q0[0] * q0[0] + q0[1] * q0[1]);
        let grad_old = [-q0[0], -q0[1]];
        let step = hmc_step_target(
            &mut target,
            &q0,
            lp_old,
            &grad_old,
            0.05,
            leapfrog,
            1.0,
            &mut ws,
            &mut rng,
        )
        .unwrap();
        assert_eq!(
            target.calls, leapfrog as usize,
            "gradient at q0 and the final density are reused"
        );
        assert!(step.accepted, "a 0.05-step trajectory on N(0, I) is accepted");
        // The reported density is that of the trajectory's final state.
        let q = &ws.q[..2];
        assert!((step.logp - (-0.5 * (q[0] * q[0] + q[1] * q[1]))).abs() < 1e-12);
        // And the workspace gradient is the score at that state, ready to carry forward.
        assert!((ws.grad[0] + q[0]).abs() < 1e-12 && (ws.grad[1] + q[1]).abs() < 1e-12);
    }

    #[test]
    fn dual_averaging_shrinks_toward_ten_times_initial_step() {
        // With acceptance exactly on target, H̄ stays 0 and the iterate is the
        // shrinkage point μ = ln(10 ε₀) = ln 0.2.
        let hmc = HmcOptions { step_size: 0.02, target_accept: 0.8, ..HmcOptions::default() };
        let (mut h_bar, mut log_eps_bar, mut step) = (0.0, hmc.step_size.ln(), hmc.step_size);
        dual_average_update(&mut h_bar, &mut log_eps_bar, &mut step, hmc, 0, 0.8);
        assert!((step - 0.2).abs() < 1e-12, "step {step}");
        assert!((log_eps_bar.exp() - 0.2).abs() < 1e-12);
    }

    #[test]
    fn averaged_step_never_exceeds_a_step_that_was_tried() {
        // Perfect acceptance drives the raw iterate far above the cap; the
        // averaged log-step must be built from the clamped iterates.
        let hmc = HmcOptions::default();
        let (mut h_bar, mut log_eps_bar, mut step) = (0.0, hmc.step_size.ln(), hmc.step_size);
        for t in 0..400 {
            dual_average_update(&mut h_bar, &mut log_eps_bar, &mut step, hmc, t, 1.0);
            assert!(step <= MAX_STEP + 1e-15);
            assert!(log_eps_bar <= MAX_STEP.ln() + 1e-12, "t={t} log_eps_bar={log_eps_bar}");
        }
        assert!(dual_average_finalize(log_eps_bar) <= MAX_STEP);
    }

    #[test]
    fn chain_streams_come_from_the_stream_mixer() {
        let seed = 91;
        for chain in 0..4 {
            let mut ours = chain_rng(seed, chain);
            let mut mixed =
                RngFactory::from_seed(seed).stream_for(StreamDomain::Bayesian, chain as u64);
            assert_eq!(ours.next_u64(), mixed.next_u64());
        }
        let firsts: Vec<u64> = (0..4).map(|c| chain_rng(seed, c).next_u64()).collect();
        for a in 0..4 {
            for b in (a + 1)..4 {
                assert_ne!(firsts[a], firsts[b]);
            }
        }
    }

    #[test]
    fn threaded_chains_reproduce_serial_draws() {
        let n = 30;
        let x = vec![1.0; n];
        let y: Vec<f64> = (0..n).map(|r| 2.0 + ((r % 4) as f64 - 1.5) * 0.05).collect();
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
        let fit_opts = BayesFitOptions { n_draws: 600, seed: 5, max_iter: 50, grad_tol: 1e-8 };
        let hmc = hmc_opts(600, 4);
        let mut ws = LaplaceWorkspace::default();
        let serial =
            fit_hmc_glm(BayesLikelihood::GaussianIdentity, design, &prior, &fit_opts, hmc, &mut ws)
                .unwrap();
        let ctx = ExecutionContext::production(5, 4);
        let threaded = fit_hmc_impl(
            BayesLikelihood::GaussianIdentity,
            design,
            &prior,
            &fit_opts,
            hmc,
            &mut LaplaceWorkspace::default(),
            Some(&ctx),
        )
        .unwrap();
        assert_eq!(serial.draws.values, threaded.draws.values);
        assert_eq!(serial.map, threaded.map);
    }
}
