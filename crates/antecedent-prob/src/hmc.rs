//! Native Hamiltonian Monte Carlo for Bayesian GLMs.
//!
//! Leapfrog HMC with dual-averaging step-size adaptation during warmup.
//! Multi-chain draws are columnar; ESS / R-hat / divergence counts gate publication.
//! Chains are independent (one stream each from the library's stream mixer) and run
//! on the execution context's threads when one is supplied.
//!
//! `GaussianIdentity` uses a fixed prior-driven target (known σ² or joint
//! `(β, log σ²)` under an `InvGamma` residual prior). Other likelihoods keep the
//! GLM score path.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::diagnostic::{RouteMass, RouteNativeDiagnostic};
use antecedent_core::{CausalRng, ExecutionContext};

use crate::streams::chain_rng;
use antecedent_kernels::{Welford, standard_normal};

use crate::backend::{
    BayesDesignRef, BayesFitOptions, BayesFitResult, BayesLikelihood, InferenceBackend,
    LaplaceWorkspace,
};
use crate::diagnostics::{HessianFactorization, InferenceDiagnostics};
use crate::error::ProbError;
use crate::gaussian_target::{PosteriorTarget, gaussian_target_from_model_with_precision};
use crate::likelihood_terms::{
    LikelihoodScratch, accumulate_likelihood, log_posterior_value, validate_design,
};
use crate::mcmc_stats::{all_chains_moved, mcmc_summary};
use crate::posterior::{PosteriorDraws, PosteriorQuantityKind, PosteriorSchema};
use crate::prior::{
    CoefficientPrecision, GaussianCoefficientPrior, GaussianVarianceModel, PriorSet,
};

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
    /// Initial diagonal mass-matrix scale (kinetic energy `½ Σ p² / mass`); the metric used
    /// until [`Self::adapt_metric`] replaces it.
    pub mass: f64,
    /// Adapt a diagonal mass matrix to the posterior scales during warmup: the per-coordinate
    /// variance of the chain's own states over a slow window (regularised toward a small
    /// floor, as in Stan) becomes the inverse mass, after which the step size is re-adapted.
    /// Without it a leapfrog step small enough for the tightest coordinate crawls along the
    /// loosest one, and badly scaled designs fail the ESS / R-hat gate. Needs
    /// [`MIN_METRIC_WARMUP`] warmup iterations; shorter warmups keep the initial metric. The
    /// estimate is refined over doubling slow windows, as in Stan.
    pub adapt_metric: bool,
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
            adapt_metric: true,
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
    n_transitions: u128,
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
        self.accept_prob_sum / self.n_transitions as f64
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

/// Width of the multiplicative step-size jitter applied to every leapfrog trajectory (Neal
/// 2011, sec. 4.2): a fixed leapfrog step count makes the trajectory length exactly
/// `L * step_size`, and any near-Gaussian mode whose oscillation period is close to that
/// length returns close to its own starting point, so the chain barely moves even though the
/// step is accepted almost every time (high mean acceptance, terrible ESS/R-hat). Each
/// transition instead runs at `step_size * U(1 - JITTER, 1)`, decorrelating the realized
/// trajectory length from any one mode's period without touching the dual-averaged
/// `step_size` that adaptation and the [`MIN_STEP`]/[`MAX_STEP`] bounds track. The jitter only
/// shrinks, never grows, the executed step: dual averaging already tunes `step_size` to the
/// requested acceptance rate, so scaling *up* would raise the divergence rate it targeted
/// against; scaling down still detunes the trajectory length from a resonant period.
const STEP_JITTER: f64 = 0.2;

/// Multiplicative jitter factor in `[1 - STEP_JITTER, 1]`, clamped into the step bounds after
/// scaling.
fn jittered_step(step_size: f64, rng: &mut CausalRng) -> f64 {
    let factor = 1.0 - STEP_JITTER * rng.next_f64();
    (step_size * factor).clamp(MIN_STEP, MAX_STEP)
}

/// Shortest warmup that still leaves a usable slow window for metric adaptation.
pub(crate) const MIN_METRIC_WARMUP: usize = 100;

/// Slow-window schedule of metric adaptation over `n_warmup` iterations, as in Stan: a fast
/// initial buffer, then windows that double in length (each re-estimating the metric), then a
/// fast terminal buffer for the final step-size adaptation. Warmups too short for that fall
/// back to one window between proportional buffers. Returns `[start, end)` per window.
fn metric_windows(n_warmup: usize) -> Vec<(usize, usize)> {
    if n_warmup < MIN_METRIC_WARMUP {
        return Vec::new();
    }
    let (mut init, mut term, mut size) = (75usize, 50usize, 25usize);
    if init + size + term > n_warmup {
        init = n_warmup * 15 / 100;
        term = n_warmup * 10 / 100;
        size = n_warmup - init - term;
    }
    let slow_end = n_warmup - term;
    let mut windows = Vec::new();
    let mut start = init;
    loop {
        let mut end = start + size;
        // A following doubled window that would not fit is absorbed into this one.
        if size > (slow_end - end) / 2 {
            end = slow_end;
        }
        windows.push((start, end));
        if end == slow_end {
            return windows;
        }
        start = end;
        size *= 2;
    }
}

/// Diagonal inverse-mass estimator: per-coordinate variance of the chain's own states over
/// each slow window (Welford accumulation, restarted for every window).
struct DiagonalMetric {
    inv_mass: Vec<f64>,
    windows: Vec<(usize, usize)>,
    cursor: usize,
    /// One unit-weight accumulator per coordinate. Every `observe` pushes each
    /// coordinate, so the masses stay equal to the old shared count.
    acc: Vec<Welford>,
}

impl DiagonalMetric {
    fn new(dim: usize, hmc: HmcOptions) -> Self {
        let windows = if hmc.adapt_metric { metric_windows(hmc.n_warmup) } else { Vec::new() };
        Self {
            inv_mass: vec![1.0 / hmc.mass; dim],
            windows,
            cursor: 0,
            acc: vec![Welford::new(); dim],
        }
    }

    /// Feed the state after warmup transition `t`.
    fn observe(&mut self, t: usize, q: &[f64]) {
        let Some(&(start, end)) = self.windows.get(self.cursor) else { return };
        if t < start || t >= end {
            return;
        }
        for (acc, &v) in self.acc.iter_mut().zip(q) {
            acc.push(v);
        }
    }

    /// After the last transition of a slow window, install its estimate (regularised toward a
    /// small floor, `n/(n+5) var + 1e-3 · 5/(n+5)`, as Stan does) and open the next window.
    /// Returns whether the metric changed, so the caller restarts step-size adaptation.
    fn install_if_window_closed(&mut self, t: usize) -> bool {
        let Some(&(_, end)) = self.windows.get(self.cursor) else { return false };
        if t + 1 != end {
            return false;
        }
        self.cursor += 1;
        let n = self.acc.first().map_or(0.0, Welford::mass);
        let installed = n >= 3.0 && {
            let weight = n / (n + 5.0);
            let proposal: Vec<f64> = self
                .acc
                .iter()
                .map(|acc| weight * (acc.m2() / (n - 1.0)) + (1.0 - weight) * 1e-3)
                .collect();
            let usable = proposal.iter().all(|v| v.is_finite() && *v > 0.0);
            if usable {
                self.inv_mass = proposal.into_iter().map(|v| v.clamp(1e-12, 1e12)).collect();
            }
            usable
        };
        for acc in &mut self.acc {
            *acc = Welford::new();
        }
        installed
    }
}

/// Post-warmup draws and transition counts from one chain.
struct ChainRun {
    /// `n_keep × state_dim` retained states, draw-major.
    samples: Vec<f64>,
    stats: TransitionStats,
    /// Step size the retained draws were generated with.
    final_step: f64,
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
    // GLM targets read V0 at σ² ≡ 1, so an absolute-scale prior is already in
    // their units; only a Gaussian target converts it with its residual variance.
    if likelihood == BayesLikelihood::GaussianIdentity {
        return crate::conjugate::fit_with_absolute_scale_resolved(design, prior, |prior| {
            fit_hmc_resolved(likelihood, design, prior, fit_opts, hmc, workspace, ctx)
        });
    }
    fit_hmc_resolved(likelihood, design, prior, fit_opts, hmc, workspace, ctx)
}

#[allow(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "negated comparison is deliberate: a NaN input must take the rejecting branch, which the un-negated form would let through"
)]
fn fit_hmc_resolved(
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
    if hmc.leapfrog_steps == 0
        || !hmc.step_size.is_finite()
        || !(hmc.step_size > 0.0)
        || !hmc.mass.is_finite()
        || !(hmc.mass > 0.0)
        || !(1.0 / hmc.mass).is_finite()
        || !hmc.target_accept.is_finite()
        || !(hmc.target_accept > 0.0 && hmc.target_accept < 1.0)
    {
        return Err(ProbError::Inference {
            message: "invalid HMC step_size / mass / L / target_accept",
        });
    }
    if fit_opts.n_draws == 0 {
        return Err(ProbError::Shape { message: "n_draws must be > 0" });
    }

    let total_transitions = hmc
        .n_warmup
        .checked_add(fit_opts.n_draws)
        .ok_or(ProbError::Shape { message: "HMC transition count overflow" })?;

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
    let include_sigma2 =
        gaussian_model.is_some_and(super::prior::GaussianVarianceModel::include_sigma2);
    let dim = ncols + usize::from(include_sigma2);
    // Conjugate-scale V0^{-1} (dense under a coefficient correlation). GLM has no
    // residual σ², so its absolute prior precision is V0^{-1} at σ² ≡ 1.
    let prec = prior.coefficient_precision(&coef_prior)?;
    let n_keep = fit_opts.n_draws;
    // Each chain keeps `n_keep * dim` draws, the merged copy holds them again, and the packed
    // output is another block of the same width, all live at the merge.
    let _admitted = crate::admission::admit_cells(
        ctx,
        hmc.n_chains
            .checked_mul(n_keep)
            .and_then(|draws| draws.checked_mul(dim.checked_mul(3)?))
            .ok_or(ProbError::Shape { message: "draw buffer size overflow" })?,
    )?;

    let run = |chain: usize, ws: &mut LaplaceWorkspace| -> Result<ChainRun, ProbError> {
        if let Some(model) = gaussian_model {
            run_gaussian_chain(
                chain,
                CoefficientModel { design, coef_prior: &coef_prior, prec: &prec },
                model,
                fit_opts,
                hmc,
                total_transitions,
                ws,
            )
        } else {
            let target = GlmTarget { likelihood, design, coef_prior: &coef_prior, prec: &prec };
            run_glm_chain(chain, target, fit_opts, hmc, total_transitions, ws)
        }
    };
    let prepare = |ws: &mut LaplaceWorkspace| ws.prepare(nrows, dim.max(ncols), dim);
    let runs: Vec<ChainRun> = if let Some(ctx) = ctx {
        let token = antecedent_core::numerical_admission::NumericalAdmissionToken::capture();
        ctx.map_indexed(hmc.n_chains, |chain, worker_ctx| {
            token.execute(worker_ctx, || {
                let mut admitted =
                    crate::backend::AdmittedLaplaceWorkspace::new(nrows, dim.max(ncols), 1)?;
                run(chain, &mut admitted.workspace)
            })
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
        ChainLayout { n_keep, state_dim: dim, ncols, include_sigma2, total_transitions },
        stats,
        (step_lo, step_hi),
    )
}

/// A posterior the leapfrog integrator can traverse.
///
/// The two implementors differ only in how they evaluate the gradient and the
/// log density, and in which sweep errors mean "this trajectory diverged":
///
/// * the GLM score path sweeps the data once per leapfrog step for `∇U` and once
///   more per transition for the final log posterior (`L + 1` sweeps), and
///   propagates every [`ProbError`] except [`ProbError::Numerical`];
/// * the Gaussian target returns the log density with its gradient (`L` sweeps)
///   and treats any error as a divergent trajectory.
///
/// Both leave the gradient in a [`LaplaceWorkspace`] slot (`step` for GLM, which
/// holds `∇U`; `grad` for Gaussian, which holds the log-posterior score), and the
/// slot is also the convention of the carried gradient between transitions.
trait HmcTarget {
    /// Log density at `q` and, in `carried`, the gradient in this target's slot
    /// convention. Uses `ws` as scratch.
    fn init(
        &mut self,
        q: &[f64],
        ws: &mut LaplaceWorkspace,
        carried: &mut [f64],
    ) -> Result<f64, ProbError>;

    /// Gradient at `ws.q[..dim]`, written to [`Self::slot`]. `Some(logp)` when the
    /// same sweep also yields the log density.
    fn eval_grad(
        &mut self,
        dim: usize,
        ws: &mut LaplaceWorkspace,
    ) -> Result<Option<f64>, ProbError>;

    /// Log density at the trajectory's final state `ws.q[..dim]`; `lp_last` is the
    /// last value [`Self::eval_grad`] returned (or the old density if none).
    fn final_logp(
        &mut self,
        dim: usize,
        ws: &mut LaplaceWorkspace,
        lp_last: f64,
    ) -> Result<f64, ProbError>;

    /// Whether a sweep error means a divergent trajectory (otherwise it aborts).
    fn is_divergent(error: &ProbError) -> bool;

    /// Workspace slice holding this target's gradient after [`Self::eval_grad`].
    fn slot(ws: &LaplaceWorkspace) -> &[f64];

    /// `∇U` component from a value stored in the gradient slot.
    fn grad_u(slot_value: f64) -> f64;
}

/// GLM score-path target (every likelihood but Gaussian identity).
struct GlmTarget<'a> {
    likelihood: BayesLikelihood,
    design: BayesDesignRef<'a>,
    coef_prior: &'a GaussianCoefficientPrior,
    prec: &'a CoefficientPrecision,
}

impl HmcTarget for GlmTarget<'_> {
    fn init(
        &mut self,
        q: &[f64],
        ws: &mut LaplaceWorkspace,
        carried: &mut [f64],
    ) -> Result<f64, ProbError> {
        let nrows = self.design.nrows;
        let ncols = self.design.ncols;
        // The current state's log-posterior and ∇U are carried across transitions:
        // after an accept they are exactly the trajectory's final-state values, after
        // a reject they are unchanged, so no transition re-sweeps the data for them.
        let lp = log_posterior_value(
            self.likelihood,
            self.design,
            q,
            self.coef_prior,
            self.prec,
            &mut ws.eta[..nrows],
            1.0,
        )?;
        self.neg_log_posterior_grad(
            q,
            LikelihoodScratch {
                grad: &mut ws.grad[..ncols],
                neg_hess: &mut ws.neg_hessian[..ncols * ncols],
                eta: &mut ws.eta[..nrows],
                work_w: &mut ws.work_w[..nrows],
            },
            carried,
        )?;
        Ok(lp)
    }

    fn eval_grad(
        &mut self,
        dim: usize,
        ws: &mut LaplaceWorkspace,
    ) -> Result<Option<f64>, ProbError> {
        let nrows = self.design.nrows;
        self.neg_log_posterior_grad(
            &ws.q[..dim],
            LikelihoodScratch {
                grad: &mut ws.grad[..dim],
                neg_hess: &mut ws.neg_hessian[..dim * dim],
                eta: &mut ws.eta[..nrows],
                work_w: &mut ws.work_w[..nrows],
            },
            &mut ws.step[..dim],
        )?;
        Ok(None)
    }

    fn final_logp(
        &mut self,
        dim: usize,
        ws: &mut LaplaceWorkspace,
        _lp_last: f64,
    ) -> Result<f64, ProbError> {
        let nrows = self.design.nrows;
        log_posterior_value(
            self.likelihood,
            self.design,
            &ws.q[..dim],
            self.coef_prior,
            self.prec,
            &mut ws.eta[..nrows],
            1.0,
        )
    }

    fn is_divergent(error: &ProbError) -> bool {
        matches!(error, ProbError::Numerical { .. })
    }

    fn slot(ws: &LaplaceWorkspace) -> &[f64] {
        // `step` holds ∇U afterwards, disjoint from `grad` (log-posterior score).
        &ws.step
    }

    fn grad_u(slot_value: f64) -> f64 {
        slot_value
    }
}

/// Gaussian-identity target behind [`PosteriorTarget`].
struct GaussianRef<'a>(&'a mut dyn PosteriorTarget);

impl HmcTarget for GaussianRef<'_> {
    fn init(
        &mut self,
        q: &[f64],
        ws: &mut LaplaceWorkspace,
        carried: &mut [f64],
    ) -> Result<f64, ProbError> {
        let dim = q.len();
        // Log-posterior and its gradient at the current state, carried across transitions.
        let lp = self.0.logp_and_grad(q, &mut ws.grad[..dim])?;
        carried.copy_from_slice(&ws.grad[..dim]);
        Ok(lp)
    }

    fn eval_grad(
        &mut self,
        dim: usize,
        ws: &mut LaplaceWorkspace,
    ) -> Result<Option<f64>, ProbError> {
        self.0.logp_and_grad(&ws.q[..dim], &mut ws.grad[..dim]).map(Some)
    }

    fn final_logp(
        &mut self,
        _dim: usize,
        _ws: &mut LaplaceWorkspace,
        lp_last: f64,
    ) -> Result<f64, ProbError> {
        // The last leapfrog evaluation already produced the final density.
        Ok(lp_last)
    }

    fn is_divergent(_error: &ProbError) -> bool {
        true
    }

    fn slot(ws: &LaplaceWorkspace) -> &[f64] {
        &ws.grad
    }

    fn grad_u(slot_value: f64) -> f64 {
        -slot_value
    }
}

fn run_glm_chain(
    chain: usize,
    mut target: GlmTarget<'_>,
    fit_opts: &BayesFitOptions,
    hmc: HmcOptions,
    total_transitions: usize,
    ws: &mut LaplaceWorkspace,
) -> Result<ChainRun, ProbError> {
    let mut rng = chain_rng(fit_opts.seed, chain);
    let mut beta = target.coef_prior.mean.to_vec();
    for bi in &mut beta {
        *bi += 0.1 * standard_normal(&mut rng);
    }
    run_chain(&mut target, beta, rng, fit_opts.n_draws, hmc, total_transitions, ws)
}

/// Design, coefficient prior and precision a coefficient posterior is sampled under.
#[derive(Clone, Copy)]
struct CoefficientModel<'a> {
    design: BayesDesignRef<'a>,
    coef_prior: &'a GaussianCoefficientPrior,
    prec: &'a CoefficientPrecision,
}

fn run_gaussian_chain(
    chain: usize,
    coefficients: CoefficientModel<'_>,
    model: GaussianVarianceModel,
    fit_opts: &BayesFitOptions,
    hmc: HmcOptions,
    total_transitions: usize,
    ws: &mut LaplaceWorkspace,
) -> Result<ChainRun, ProbError> {
    let CoefficientModel { design, coef_prior, prec } = coefficients;
    let ncols = design.ncols;
    let mut target =
        gaussian_target_from_model_with_precision(design, coef_prior.clone(), prec.clone(), model)?;
    let dim = target.dim();
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
    run_chain(&mut GaussianRef(&mut target), q, rng, fit_opts.n_draws, hmc, total_transitions, ws)
}

/// One chain: dual-averaged step size, optional windowed diagonal metric, `n_keep`
/// retained states after `hmc.n_warmup` transitions, from the initial state `q`.
fn run_chain<T: HmcTarget>(
    target: &mut T,
    mut q: Vec<f64>,
    mut rng: CausalRng,
    n_keep: usize,
    hmc: HmcOptions,
    total_transitions: usize,
    ws: &mut LaplaceWorkspace,
) -> Result<ChainRun, ProbError> {
    let dim = q.len();
    let mut step_size = hmc.step_size;
    let mut log_eps_bar = step_size.ln();
    let mut h_bar = 0.0;
    let mut metric = DiagonalMetric::new(dim, hmc);
    // Step-size adaptation is anchored at the step it (re)started from, counting from `since`.
    let mut anchor = hmc;
    let mut since = 0usize;

    let mut carried = vec![0.0; dim];
    let mut lp_curr = target.init(&q, ws, &mut carried)?;

    let mut samples = vec![0.0; n_keep * dim];
    let mut stats = TransitionStats::default();
    let mut kept = 0usize;
    for t in 0..total_transitions {
        check_hmc_cancelled()?;
        if t == hmc.n_warmup {
            // Warmup is over before this transition runs: every retained draw
            // uses the averaged step, not the last exploratory one.
            step_size = dual_average_finalize(log_eps_bar);
        }
        let exec_step = jittered_step(step_size, &mut rng);
        let path = Trajectory {
            step_size: exec_step,
            leapfrog_steps: hmc.leapfrog_steps,
            inv_mass: &metric.inv_mass,
        };
        let step = hmc_step(target, &q, &carried, lp_curr, path, ws, &mut rng)?;
        stats.record(&step, t < hmc.n_warmup);
        if step.accepted {
            q.copy_from_slice(&ws.q[..dim]);
            carried.copy_from_slice(&T::slot(ws)[..dim]);
            lp_curr = step.logp;
        }
        if t < hmc.n_warmup {
            dual_average_update(
                &mut h_bar,
                &mut log_eps_bar,
                &mut step_size,
                anchor,
                t - since,
                step.accept_prob,
            );
            metric.observe(t, &q);
            if metric.install_if_window_closed(t) {
                // The trajectory geometry changed: restart step-size adaptation from the
                // current step under the new metric.
                anchor = HmcOptions { step_size, ..hmc };
                since = t + 1;
                (h_bar, log_eps_bar) = (0.0, step_size.ln());
            }
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

/// Shape of the pooled chain samples: retained draws per chain, state width and the
/// coefficient / σ² split.
#[derive(Clone, Copy)]
struct ChainLayout {
    n_keep: usize,
    state_dim: usize,
    ncols: usize,
    include_sigma2: bool,
    total_transitions: usize,
}

fn pack_and_gate_hmc(
    chain_samples: &[f64],
    hmc: HmcOptions,
    layout: ChainLayout,
    stats: TransitionStats,
    step_range: (f64, f64),
) -> Result<BayesFitResult, ProbError> {
    let ChainLayout { n_keep, state_dim, ncols, include_sigma2, total_transitions } = layout;
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
        iterations: u32::try_from(total_transitions).unwrap_or(u32::MAX),
        grad_inf_norm: 0.0,
        hessian_condition: f64::NAN,
        factorization: HessianFactorization::Mcmc,
        separation_warning: false,
        notes: vec![Arc::from(format!(
            "hmc chains={} warmup={} L={} step_size=[{:.3e}, {:.3e}]",
            hmc.n_chains, hmc.n_warmup, hmc.leapfrog_steps, step_range.0, step_range.1
        ))],
        backend_id: Arc::from("hmc"),
        n_chains: Some(u32::try_from(hmc.n_chains).unwrap_or(u32::MAX)),
        n_warmup: Some(u32::try_from(hmc.n_warmup).unwrap_or(u32::MAX)),
        ess_bulk_min: Some(ess_bulk),
        ess_tail_min: Some(ess_tail),
        rhat_max: Some(rhat_max),
        n_divergences: None,
        mean_accept_prob: Some(mean_accept),
        n_warmup_divergences: Some(stats.n_warmup_divergences),
        n_postwarmup_divergences: None,
        max_abs_delta_h: Some(max_dh),
        all_chains_moved: Some(moved),
    };
    diagnostics.record_postwarmup_divergences(stats.n_postwarmup_divergences);
    diagnostics.converged = diagnostics.mcmc_publication_ok();

    if !diagnostics.allows_posterior() {
        // The step is bounded; an adapted step pinned at a bound means the posterior scale is
        // outside what the (diagonal-metric) sampler can traverse.
        let at_bound =
            step_range.0 <= MIN_STEP * (1.0 + 1e-9) || step_range.1 >= MAX_STEP * (1.0 - 1e-9);
        let scale_hint = if at_bound {
            format!(
                "; adapted step size reached its bound ({:.3e}..{:.3e}), so the posterior \
                 scale is outside the range this sampler can traverse even with its diagonal \
                 metric: check the design for near-collinear or wildly scaled columns",
                step_range.0, step_range.1
            )
        } else {
            String::new()
        };
        let summary = format!(
            "HMC posterior refused: rhat={:?} ess_bulk={:?} ess_tail={:?} \
             postwarmup_div={:?} moved={:?} accept={:?}{scale_hint}",
            diagnostics.rhat_max,
            diagnostics.ess_bulk_min,
            diagnostics.ess_tail_min,
            diagnostics.n_postwarmup_divergences,
            diagnostics.all_chains_moved,
            diagnostics.mean_accept_prob
        );
        return Err(diagnostics.publication_error(summary.clone()).unwrap_or(ProbError::Route {
            summary,
            diagnostics: vec![
                RouteNativeDiagnostic::computational_failure(RouteMass::unreported()),
            ],
        }));
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

/// Step size, step count and diagonal inverse mass of one leapfrog trajectory.
#[derive(Clone, Copy)]
struct Trajectory<'a> {
    step_size: f64,
    leapfrog_steps: u32,
    inv_mass: &'a [f64],
}

/// Leapfrog trajectory from `ws.q` with momentum `ws.p`, given the carried gradient
/// at the start. `Some(logp)` is the density of the last evaluated state (`lp_old`
/// when no sweep produced one); `None` means the trajectory diverged.
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
fn check_hmc_cancelled() -> Result<(), ProbError> {
    if antecedent_core::numerical_admission::is_cancelled() {
        return Err(ProbError::Refused {
            code: antecedent_core::reason_code!("transport_budget_cancel"),
            message: "hmc.cancelled: posterior sampling cancelled".into(),
        });
    }
    Ok(())
}

fn leapfrog<T: HmcTarget>(
    target: &mut T,
    dim: usize,
    path: Trajectory<'_>,
    carried: &[f64],
    lp_old: f64,
    ws: &mut LaplaceWorkspace,
) -> Result<Option<f64>, ProbError> {
    let Trajectory { step_size, leapfrog_steps, inv_mass } = path;
    // ∇U at q0 is supplied by the caller (carried from the previous transition
    // rather than re-swept).
    for i in 0..dim {
        ws.p[i] -= 0.5 * step_size * T::grad_u(carried[i]);
    }

    let mut divergent = false;
    // Log-density at the trajectory's final state: the last leapfrog evaluation.
    let mut lp_last = lp_old;
    for step in 0..leapfrog_steps {
        check_hmc_cancelled()?;
        for i in 0..dim {
            ws.q[i] += step_size * ws.p[i] * inv_mass[i];
            if !ws.q[i].is_finite() {
                divergent = true;
                break;
            }
        }
        if divergent {
            break;
        }
        match target.eval_grad(dim, ws) {
            Ok(Some(v)) => lp_last = v,
            Ok(None) => {}
            Err(e) if T::is_divergent(&e) => {
                divergent = true;
                break;
            }
            Err(e) => return Err(e),
        }
        let last = step + 1 == leapfrog_steps;
        let scale = if last { 0.5 } else { 1.0 };
        for i in 0..dim {
            let grad_u = T::grad_u(T::slot(ws)[i]);
            ws.p[i] -= scale * step_size * grad_u;
            if !ws.p[i].is_finite() {
                divergent = true;
            }
        }
        if divergent {
            break;
        }
    }
    Ok(if divergent { None } else { Some(lp_last) })
}

/// One HMC transition from `q0`; `carried` is the gradient at `q0` in the target's
/// slot convention and `lp_old` the density there. The proposal is left in
/// `ws.q` / the target's gradient slot for the caller to copy on accept.
#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
fn hmc_step<T: HmcTarget>(
    target: &mut T,
    q0: &[f64],
    carried: &[f64],
    lp_old: f64,
    path: Trajectory<'_>,
    ws: &mut LaplaceWorkspace,
    rng: &mut CausalRng,
) -> Result<HmcStepResult, ProbError> {
    let inv_mass = path.inv_mass;
    let dim = q0.len();
    ws.q[..dim].copy_from_slice(q0);
    let mut p0_energy = 0.0;
    for i in 0..dim {
        ws.p[i] = standard_normal(rng) / inv_mass[i].sqrt();
        p0_energy += 0.5 * ws.p[i] * ws.p[i] * inv_mass[i];
    }

    let reject_divergent = |delta_h: f64| HmcStepResult {
        logp: lp_old,
        accepted: false,
        accept_prob: 0.0,
        delta_h,
        divergent: true,
    };

    let Some(lp_last) = leapfrog(target, dim, path, carried, lp_old, ws)? else {
        return Ok(reject_divergent(1_001.0));
    };

    let lp_new = match target.final_logp(dim, ws, lp_last) {
        Ok(v) => v,
        Err(e) if T::is_divergent(&e) => return Ok(reject_divergent(1_001.0)),
        Err(e) => return Err(e),
    };
    let mut p_new_energy = 0.0;
    for i in 0..dim {
        p_new_energy += 0.5 * ws.p[i] * ws.p[i] * inv_mass[i];
    }

    let h_old = -lp_old + p0_energy;
    let h_new = -lp_new + p_new_energy;
    let delta_h = h_new - h_old;
    let (divergent, accept_prob) = finalize_energy(delta_h);
    if divergent {
        return Ok(reject_divergent(delta_h));
    }
    let accepted = rng.next_f64() < accept_prob;
    if accepted {
        Ok(HmcStepResult { logp: lp_new, accepted: true, accept_prob, delta_h, divergent: false })
    } else {
        Ok(HmcStepResult { logp: lp_old, accepted: false, accept_prob, delta_h, divergent: false })
    }
}

/// [`hmc_step`] for a [`PosteriorTarget`]; `grad_old` is the log-posterior score at `q0`.
#[cfg(test)]
fn hmc_step_target(
    target: &mut dyn PosteriorTarget,
    q0: &[f64],
    lp_old: f64,
    grad_old: &[f64],
    path: Trajectory<'_>,
    workspace: &mut LaplaceWorkspace,
    rng: &mut CausalRng,
) -> Result<HmcStepResult, ProbError> {
    hmc_step(&mut GaussianRef(target), q0, grad_old, lp_old, path, workspace, rng)
}

/// [`hmc_step`] for the GLM score path; `gradu_old` is `∇U` at `beta`.
#[cfg(test)]
fn hmc_step_glm(
    mut target: GlmTarget<'_>,
    beta: &[f64],
    gradu_old: &[f64],
    path: Trajectory<'_>,
    lp_old: f64,
    workspace: &mut LaplaceWorkspace,
    rng: &mut CausalRng,
) -> Result<HmcStepResult, ProbError> {
    hmc_step(&mut target, beta, gradu_old, lp_old, path, workspace, rng)
}

impl GlmTarget<'_> {
    /// `∇U(beta)` into `grad_out`, using `scratch` for the likelihood sweep.
    fn neg_log_posterior_grad(
        &self,
        beta: &[f64],
        scratch: LikelihoodScratch<'_>,
        grad_out: &mut [f64],
    ) -> Result<(), ProbError> {
        let LikelihoodScratch { grad: grad_lp, neg_hess, eta, work_w } = scratch;
        accumulate_likelihood(
            self.likelihood,
            self.design,
            beta,
            LikelihoodScratch { grad: &mut *grad_lp, neg_hess, eta, work_w },
            1.0,
            false,
        )?;
        self.prec.sub_prior_gradient(beta, &self.coef_prior.mean, grad_lp);
        for i in 0..beta.len() {
            grad_out[i] = -grad_lp[i];
        }
        Ok(())
    }
}

/// Leapfrog integration used by reversibility tests (no Metropolis): the shared
/// [`leapfrog`] at a scalar `mass`.
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
    let mut ws = LaplaceWorkspace::default();
    ws.prepare(0, dim, dim);
    ws.q[..dim].copy_from_slice(q0);
    ws.p[..dim].copy_from_slice(p0);
    let mut target = GaussianRef(target);
    let mut carried = vec![0.0; dim];
    let lp0 = target.init(q0, &mut ws, &mut carried)?;
    let inv_mass = vec![1.0 / mass; dim];
    let path = Trajectory { step_size, leapfrog_steps, inv_mass: &inv_mass };
    match leapfrog(&mut target, dim, path, &carried, lp0, &mut ws)? {
        Some(_) => Ok((ws.q[..dim].to_vec(), ws.p[..dim].to_vec())),
        None => Err(ProbError::Numerical { message: "leapfrog trajectory diverged".into() }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::conjugate::fit_conjugate_gaussian;
    use crate::gaussian_target::gaussian_target_from_model;
    use crate::prior::{InvGammaPrior, PriorSpec};
    use antecedent_core::{RngFactory, StreamDomain};

    /// Exact known-σ² Gaussian posterior `N(m, C)` under the conjugate prior
    /// `β | σ² ~ N(0, σ²·v0·I)` (`v0` is the [`GaussianCoefficientPrior`] scale, whose
    /// *absolute* variance at known σ² is `σ²·v0`, per its doc and [`crate::conjugate`]'s
    /// `posterior_known_sigma2`): `C = (X'X/σ² + I/(v0·σ²))⁻¹`, `m = C X'y/σ²`
    /// (column-major `x`, `p` columns).
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
                a[i * p + j] = acc / sigma2 + if i == j { 1.0 / (v0 * sigma2) } else { 0.0 };
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
            adapt_metric: false,
        }
    }

    #[test]
    fn invalid_hmc_metric_step_and_target_are_refused() {
        let x = [1.0, 1.0];
        let y = [0.0, 1.0];
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: 2,
            ncols: 1,
            y: &y,
            weights: None,
            offsets: None,
        };
        let prior = PriorSet::weakly_informative(1);
        let fit_opts = BayesFitOptions { n_draws: 2, seed: 1, max_iter: 50, grad_tol: 1e-8 };
        for value in [f64::INFINITY, f64::NAN, 0.0, -1.0] {
            for hmc in [
                HmcOptions { step_size: value, ..hmc_opts(1, 2) },
                HmcOptions { mass: value, ..hmc_opts(1, 2) },
            ] {
                assert!(matches!(
                    fit_hmc_glm(
                        BayesLikelihood::BernoulliLogit,
                        design,
                        &prior,
                        &fit_opts,
                        hmc,
                        &mut LaplaceWorkspace::default()
                    ),
                    Err(ProbError::Inference { .. })
                ));
            }
        }
        let hmc = HmcOptions { mass: f64::from_bits(1), ..hmc_opts(1, 2) };
        assert!(matches!(
            fit_hmc_glm(
                BayesLikelihood::BernoulliLogit,
                design,
                &prior,
                &fit_opts,
                hmc,
                &mut LaplaceWorkspace::default()
            ),
            Err(ProbError::Inference { .. })
        ));
        let hmc = HmcOptions { n_warmup: usize::MAX, ..hmc_opts(1, 2) };
        assert!(matches!(
            fit_hmc_glm(
                BayesLikelihood::BernoulliLogit,
                design,
                &prior,
                &fit_opts,
                hmc,
                &mut LaplaceWorkspace::default()
            ),
            Err(ProbError::Shape { message: "HMC transition count overflow" })
        ));
        for target_accept in [f64::NAN, f64::INFINITY, -1.0, 0.0, 1.0, 2.0] {
            let hmc = HmcOptions { target_accept, ..hmc_opts(1, 2) };
            assert!(matches!(
                fit_hmc_glm(
                    BayesLikelihood::BernoulliLogit,
                    design,
                    &prior,
                    &fit_opts,
                    hmc,
                    &mut LaplaceWorkspace::default()
                ),
                Err(ProbError::Inference { .. })
            ));
        }
    }

    #[test]
    fn hmc_worker_inherits_cancellation_and_reports_operational_refusal() {
        let x = [1.0, 1.0];
        let y = [0.0, 1.0];
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: 2,
            ncols: 1,
            y: &y,
            weights: None,
            offsets: None,
        };
        let prior = PriorSet::weakly_informative(1);
        let fit_opts = BayesFitOptions { n_draws: 100, seed: 1, max_iter: 50, grad_tol: 1e-8 };
        let mut ctx = ExecutionContext::for_tests(1);
        ctx.cancellation = antecedent_core::CancellationToken::cancel_after_checks(5);
        let error = fit_hmc_impl(
            BayesLikelihood::BernoulliLogit,
            design,
            &prior,
            &fit_opts,
            hmc_opts(10, 2),
            &mut LaplaceWorkspace::default(),
            Some(&ctx),
        )
        .unwrap_err();
        assert!(
            matches!(error, ProbError::Refused { code, .. } if code == antecedent_core::reason_code!("transport_budget_cancel"))
        );
    }

    #[test]
    fn transition_acceptance_denominator_does_not_saturate_at_u32_max() {
        let mut stats = TransitionStats {
            n_transitions: u32::MAX as u128,
            accept_prob_sum: u32::MAX as f64,
            ..TransitionStats::default()
        };
        stats.record(
            &HmcStepResult {
                logp: 0.0,
                accepted: true,
                accept_prob: 1.0,
                delta_h: 0.0,
                divergent: false,
            },
            false,
        );
        assert_eq!(stats.mean_accept_prob(), 1.0);
    }

    #[test]
    fn hmc_context_accounts_for_each_worker_workspace() {
        let x = [1.0, 1.0];
        let y = [0.0, 1.0];
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: 2,
            ncols: 1,
            y: &y,
            weights: None,
            offsets: None,
        };
        let prior = PriorSet::weakly_informative(1);
        let fit_opts = BayesFitOptions { n_draws: 2, seed: 1, max_iter: 50, grad_tol: 1e-8 };
        // Exactly covers the retained draws, leaving no room for a worker workspace.
        let ctx = ExecutionContext::for_tests(1).with_memory_limit(Some(96));
        let result = fit_hmc_impl(
            BayesLikelihood::BernoulliLogit,
            design,
            &prior,
            &fit_opts,
            hmc_opts(1, 2),
            &mut LaplaceWorkspace::default(),
            Some(&ctx),
        );
        assert!(
            matches!(result, Err(ProbError::Numerical { message }) if message.contains("prob.resource_refused"))
        );
    }

    #[test]
    fn metric_windows_handle_the_largest_warmup_count() {
        let windows = metric_windows(usize::MAX);
        assert_eq!(windows.first().unwrap().0, 75);
        assert_eq!(windows.last().unwrap().1, usize::MAX - 50);
        assert!(windows.windows(2).all(|pair| pair[0].1 == pair[1].0));
        assert!(windows.iter().all(|&(start, end)| start < end));
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
        let prec = CoefficientPrecision::Diagonal(coef.precision());
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
            GlmTarget {
                likelihood: BayesLikelihood::PoissonLog,
                design,
                coef_prior: &coef,
                prec: &prec,
            },
            &beta,
            &gradu,
            Trajectory { step_size: 50.0, leapfrog_steps: 20, inv_mass: &[1.0] },
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
        // Longer schedule + milder steps: R̂≤1.01 is tight on 2-parameter Gaussian HMC and was
        // flaking on Linux CI around 1.012 at step_size 0.04 (rhat=1.0123, ess_bulk=431 here);
        // 0.03 clears the gate with a wide margin (rhat=1.0018, ess_bulk=1836).
        let hmc = HmcOptions {
            n_chains: 4,
            n_warmup: 1500,
            leapfrog_steps: 16,
            step_size: 0.03,
            target_accept: 0.85,
            mass: 1.0,
            adapt_metric: false,
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
        let fit_opts = BayesFitOptions { n_draws: 2000, seed: 17, max_iter: 50, grad_tol: 1e-8 };
        let conj = fit_conjugate_gaussian(design, &prior, &fit_opts, &mut ws).unwrap();
        // 1000 warmup / step 0.04 was refused by the R-hat gate for 10 of 500 seeds, so this
        // passed only on its fixed seed and flipped on Linux CI (rhat=1.0107). The schedule of
        // the sibling test above (1500 warmup, step 0.03, 2000 draws) is refused for none of 500.
        let hmc = HmcOptions {
            n_chains: 4,
            n_warmup: 1500,
            leapfrog_steps: 16,
            step_size: 0.03,
            target_accept: 0.85,
            mass: 1.0,
            adapt_metric: false,
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
            Trajectory { step_size: 0.05, leapfrog_steps: leapfrog, inv_mass: &[1.0, 1.0] },
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

    #[test]
    fn metric_windows_follow_the_stan_schedule() {
        // Too short to adapt.
        assert!(metric_windows(99).is_empty());
        // Short warmups: one window between proportional buffers (15% / 10%).
        assert_eq!(metric_windows(100), vec![(15, 90)]);
        // Stan's 75 / 25-doubling / 50 layout; a doubled window that would not fit is absorbed.
        assert_eq!(metric_windows(200), vec![(75, 100), (100, 150)]);
        assert_eq!(
            metric_windows(1000),
            vec![(75, 100), (100, 150), (150, 250), (250, 450), (450, 950)]
        );
        for n in [100, 137, 200, 350, 1000, 1500] {
            let windows = metric_windows(n);
            assert!(windows.windows(2).all(|w| w[0].1 == w[1].0), "n={n}");
            assert!(windows.iter().all(|(s, e)| s < e), "n={n}");
            assert!(windows.last().unwrap().1 < n, "n={n}");
        }
    }

    #[test]
    fn diagonal_metric_is_the_regularised_window_variance() {
        let hmc = HmcOptions { n_warmup: 100, ..HmcOptions::default() };
        let mut metric = DiagonalMetric::new(2, hmc);
        assert_eq!(metric.inv_mass, vec![1.0, 1.0]);
        let mut installed_at = None;
        for t in 0..100 {
            // Coordinate 0 walks 0, 1, 2, ...; coordinate 1 is ten times that.
            let q = [t as f64, 10.0 * t as f64];
            metric.observe(t, &q);
            if metric.install_if_window_closed(t) {
                installed_at = Some(t);
            }
        }
        // The single window is iterations 15..90: n = 75 consecutive integers, whose sample
        // variance is n(n + 1)/12.
        assert_eq!(installed_at, Some(89));
        let n = 75.0_f64;
        let var0 = n * (n + 1.0) / 12.0;
        let weight = n / (n + 5.0);
        let expect = |var: f64| weight * var + (1.0 - weight) * 1e-3;
        assert!((metric.inv_mass[0] - expect(var0)).abs() < 1e-9 * var0);
        assert!((metric.inv_mass[1] - expect(100.0 * var0)).abs() < 1e-9 * 100.0 * var0);

        // Disabled adaptation and too-short warmups leave the initial metric alone.
        for hmc in [
            HmcOptions { n_warmup: 100, adapt_metric: false, mass: 4.0, ..HmcOptions::default() },
            HmcOptions { n_warmup: 60, mass: 4.0, ..HmcOptions::default() },
        ] {
            let mut metric = DiagonalMetric::new(2, hmc);
            for t in 0..hmc.n_warmup {
                metric.observe(t, &[t as f64, 1.0]);
                assert!(!metric.install_if_window_closed(t));
            }
            assert_eq!(metric.inv_mass, vec![0.25, 0.25]);
        }
    }

    /// A design whose columns differ in scale by 10^2 puts the posterior scales of intercept
    /// and slope ~10^2 apart: a unit-mass leapfrog step small enough for the slope crawls along
    /// the intercept (asserted below: the unadapted sampler fails the gate on this same
    /// design). With the adapted diagonal metric the fit publishes and recovers the exact
    /// known-variance posterior.
    ///
    /// Stan's own regularisation of the windowed variance estimate toward a fixed floor
    /// (`1e-3 · 5/(n+5)`, mirrored in [`DiagonalMetric::install_if_window_closed`]) assumes
    /// coordinates are within a few orders of magnitude of unit scale; a 10^4 column scale
    /// pushes the slope's true posterior variance (~2e-10) far enough below that floor that
    /// the metric estimate never escapes it within a 1500-iteration warmup (confirmed by
    /// running it out to 10000 iterations, where it does converge) — the same bootstrapping
    /// limit a real Stan run would hit on this unstandardised parameterisation. 10^2 is
    /// comfortably inside the range this warmup budget resolves (bulk ESS in the
    /// thousands, R-hat well under the 1.01 gate) while still failing the unadapted,
    /// unit-mass sampler outright.
    #[test]
    fn adapted_metric_samples_a_badly_scaled_design() {
        let n = 60;
        let mut x = vec![0.0; n * 2];
        let mut y = vec![0.0; n];
        for r in 0..n {
            let z = (r as f64 - 30.0) * 0.05;
            x[r] = 1.0;
            x[n + r] = 1.0e2 * z;
            y[r] = 0.5 + 1.5 * z + ((r % 5) as f64 - 2.0) * 0.2;
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
        let design = BayesDesignRef {
            x_colmajor: &x,
            nrows: n,
            ncols: 2,
            y: &y,
            weights: None,
            offsets: None,
        };
        let fit_opts = BayesFitOptions { n_draws: 2000, seed: 42, max_iter: 50, grad_tol: 1e-8 };
        let hmc = HmcOptions {
            n_chains: 4,
            n_warmup: 1500,
            leapfrog_steps: 16,
            step_size: 0.04,
            target_accept: 0.85,
            mass: 1.0,
            adapt_metric: true,
        };
        let mut ws = LaplaceWorkspace::default();
        let unadapted = HmcOptions { adapt_metric: false, ..hmc };
        assert!(
            fit_hmc_glm(
                BayesLikelihood::GaussianIdentity,
                design,
                &prior,
                &fit_opts,
                unadapted,
                &mut ws,
            )
            .is_err(),
            "the unadapted, unit-mass sampler should fail the gate on this design"
        );
        let fit =
            fit_hmc_glm(BayesLikelihood::GaussianIdentity, design, &prior, &fit_opts, hmc, &mut ws)
                .expect("the adapted sampler publishes");
        let (m, c) = exact_known_sigma2_posterior(&x, &y, 2, 25.0, 1.0);
        let ess = fit.diagnostics.ess_bulk_min.unwrap();
        for j in 0..2 {
            let (mean, var) = column_moments(&fit, j);
            let sd = c[j * 2 + j].sqrt();
            assert!((mean - m[j]).abs() < 6.0 * sd / ess.sqrt(), "mean[{j}] {mean} vs {}", m[j]);
            assert!((var.sqrt() / sd - 1.0).abs() < 0.15, "sd[{j}] {} vs {sd}", var.sqrt());
        }
    }
}

#[cfg(test)]
#[path = "hmc_oracle_tests.rs"]
mod oracle_tests;
