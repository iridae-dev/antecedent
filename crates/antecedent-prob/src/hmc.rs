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

use antecedent_core::{CausalRng, ExecutionContext};

use crate::streams::chain_rng;
use antecedent_kernels::standard_normal;

mod admission;
mod metric;
mod output;
mod target;
use admission::{admit_hmc_work, hmc_has_absolute_scale};
use metric::DiagonalMetric;
#[cfg(test)]
use metric::metric_windows;
use output::{ChainLayout, pack_and_gate_hmc};
use target::{GaussianRef, GlmTarget, HmcTarget};

use crate::backend::{
    BayesDesignRef, BayesFitOptions, BayesFitResult, BayesLikelihood, InferenceBackend,
    LaplaceWorkspace,
};
use crate::error::ProbError;
use crate::gaussian_target::{PosteriorTarget, gaussian_target_from_model_with_precision};
use crate::likelihood_terms::validate_design;
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
        antecedent_core::numerical_admission::with_context(ctx, || {
            admit_hmc_work(design, prior, options, self.options, Some(ctx))?;
            {
                let cells = prior
                    .specs
                    .iter()
                    .try_fold(0_usize, |largest, spec| {
                        if let crate::prior::PriorSpec::CoefficientCorrelation(corr) = spec {
                            corr.dim()
                                .checked_mul(corr.dim())
                                .and_then(|n| n.checked_mul(2))
                                .map(|n| n.max(largest))
                        } else {
                            Some(largest)
                        }
                    })
                    .ok_or(ProbError::Admission(
                        antecedent_core::execution::AdmissionError::Overflow,
                    ))?;
                let _prior_validation = crate::admission::admit_cells(Some(ctx), cells)?;
                prior.validate()?;
                check_hmc_cancelled()?;
            }
            fit_hmc_impl(likelihood, design, prior, options, self.options, workspace, Some(ctx))
        })
        .map_err(hmc_cancel_error)
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
    if antecedent_core::numerical_admission::has_context() {
        return fit_hmc_impl(likelihood, design, prior, fit_opts, hmc, workspace, None);
    }
    let ctx = ExecutionContext::production_default(fit_opts.seed)
        .with_memory_limit(Some(2 * 1024 * 1024 * 1024));
    fit_hmc_glm_with_context(likelihood, design, prior, fit_opts, hmc, workspace, &ctx)
}

/// Run serial HMC with an explicit memory, work and cancellation policy.
/// The requested sampling schedule is executed in full or refused.
pub fn fit_hmc_glm_with_context(
    likelihood: BayesLikelihood,
    design: BayesDesignRef<'_>,
    prior: &PriorSet,
    fit_opts: &BayesFitOptions,
    hmc: HmcOptions,
    workspace: &mut LaplaceWorkspace,
    ctx: &ExecutionContext,
) -> Result<BayesFitResult, ProbError> {
    antecedent_core::numerical_admission::with_context(ctx, || {
        fit_hmc_impl(likelihood, design, prior, fit_opts, hmc, workspace, None)
    })
}

fn hmc_cancel_error(error: ProbError) -> ProbError {
    match error {
        ProbError::Admission(antecedent_core::execution::AdmissionError::Cancelled) => {
            ProbError::Cancelled { stage: "hmc.cancelled" }
        }
        other => other,
    }
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
    fit_hmc_impl_inner(likelihood, design, prior, fit_opts, hmc, workspace, ctx)
        .map_err(hmc_cancel_error)
}

fn fit_hmc_impl_inner(
    likelihood: BayesLikelihood,
    design: BayesDesignRef<'_>,
    prior: &PriorSet,
    fit_opts: &BayesFitOptions,
    hmc: HmcOptions,
    workspace: &mut LaplaceWorkspace,
    ctx: Option<&ExecutionContext>,
) -> Result<BayesFitResult, ProbError> {
    // Quote the complete sampler and absolute-scale preprocessing before the
    // Gaussian resolver can allocate or run an OLS fit.
    admit_hmc_work(design, prior, fit_opts, hmc, ctx)?;
    let _prior_preprocessing = if likelihood == BayesLikelihood::GaussianIdentity
        && hmc_has_absolute_scale(prior)
    {
        let p = design.ncols;
        let metadata = prior
            .restrictions
            .iter()
            .try_fold(0_usize, |n, r| n.checked_add(r.id.len()))
            .and_then(|n| n.checked_mul(4))
            .and_then(|n| prior.specs.len().checked_mul(16).and_then(|v| n.checked_add(v)))
            .and_then(|n| prior.restrictions.len().checked_mul(4).and_then(|v| n.checked_add(v)))
            .and_then(|n| n.checked_add(prior.categorical.len()))
            .ok_or(ProbError::Admission(antecedent_core::execution::AdmissionError::Overflow))?;
        let cells = p
            .checked_mul(p)
            .and_then(|sq| sq.checked_mul(6))
            .and_then(|sq| p.checked_mul(8).and_then(|v| sq.checked_add(v)))
            .and_then(|v| v.checked_add(metadata))
            .ok_or(ProbError::Admission(antecedent_core::execution::AdmissionError::Overflow))?;
        crate::admission::admit_cells(ctx, cells)?
    } else {
        None
    };
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
    reason = "non-finite values must take the rejecting branch"
)]
fn validate_hmc_schedule(hmc: HmcOptions, n_draws: usize) -> Result<usize, ProbError> {
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
    if n_draws == 0 {
        return Err(ProbError::Shape { message: "n_draws must be > 0" });
    }
    // Published warmup and divergence counts are u32. Never saturate a caller's schedule.
    if u32::try_from(hmc.n_warmup).is_err()
        || hmc.n_chains.checked_mul(hmc.n_warmup).is_none_or(|n| u32::try_from(n).is_err())
        || hmc.n_chains.checked_mul(n_draws).is_none_or(|n| u32::try_from(n).is_err())
    {
        return Err(ProbError::Shape { message: "HMC diagnostic count exceeds u32" });
    }

    hmc.n_warmup
        .checked_add(n_draws)
        .ok_or(ProbError::Shape { message: "HMC transition count overflow" })
}

#[allow(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "negated comparison is deliberate: a NaN input must take the rejecting branch, which the un-negated form would let through"
)]
#[allow(
    clippy::too_many_lines,
    reason = "the HMC execution keeps admission, adaptation and sampling in one bounded sequence"
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
    let total_transitions = validate_hmc_schedule(hmc, fit_opts.n_draws)?;

    if ctx.is_some_and(|ctx| ctx.cancellation.is_cancelled()) {
        return Err(ProbError::Cancelled { stage: "hmc.cancelled" });
    }
    check_hmc_cancelled()?;

    let _default_prior = if prior.gaussian_coefficients().is_none() {
        Some(crate::admission::admit_cells(
            ctx,
            ncols.checked_mul(2).ok_or(ProbError::Admission(
                antecedent_core::execution::AdmissionError::Overflow,
            ))?,
        )?)
    } else {
        None
    };
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
    let dim = ncols
        .checked_add(usize::from(include_sigma2))
        .ok_or(ProbError::Shape { message: "HMC state dimension overflow" })?;
    // Account for per-chain sampler state, metric and target copies, alongside draws.
    let state_cells = hmc
        .n_chains
        .checked_mul(
            dim.checked_mul(16)
                .and_then(|v| {
                    ncols.checked_mul(ncols).and_then(|sq| v.checked_add(sq.checked_mul(2)?))
                })
                .ok_or(ProbError::Shape { message: "HMC state size overflow" })?,
        )
        .ok_or(ProbError::Shape { message: "HMC state size overflow" })?;
    let diagnostic_cells = hmc
        .n_chains
        .checked_mul(fit_opts.n_draws)
        .and_then(|n| n.checked_mul(20))
        .ok_or(ProbError::Shape { message: "HMC diagnostic size overflow" })?;
    let _state_admitted = crate::admission::admit_cells(
        ctx,
        state_cells
            .checked_add(diagnostic_cells)
            .ok_or(ProbError::Shape { message: "HMC state size overflow" })?,
    )?;
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
        // The direct API's caller owns the workspace after return. Admit its
        // full usable working size for this call before it can grow.
        let cols = dim.max(ncols);
        let cells = cols
            .checked_mul(cols)
            .and_then(|v| v.checked_mul(2))
            .and_then(|v| cols.checked_mul(6).and_then(|c| v.checked_add(c)))
            .and_then(|v| nrows.checked_mul(2).and_then(|r| v.checked_add(r)))
            .ok_or(ProbError::Admission(antecedent_core::execution::AdmissionError::Overflow))?;
        let _workspace_admitted = crate::admission::admit_cells(ctx, cells)?;
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
fn check_hmc_cancelled() -> Result<(), ProbError> {
    if antecedent_core::numerical_admission::is_cancelled() {
        return Err(ProbError::Cancelled { stage: "hmc.cancelled" });
    }
    Ok(())
}

#[allow(
    clippy::needless_range_loop,
    reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
)]
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
mod tests;

#[cfg(test)]
#[path = "hmc_oracle_tests.rs"]
mod oracle_tests;
