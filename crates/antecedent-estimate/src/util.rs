//! Shared estimation helpers (SOLID/DRY).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use antecedent_core::{
    AdaptiveBootstrapBudget, DeclaredSamplingDesign, ExecutionContext, ResamplingRule,
    SamplingDesignError, ScratchLease, ScratchRefusal,
};
use antecedent_data::{DataError, ResamplingPlan, resample::fill_resample_index_replicate};
use antecedent_stats::{
    DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace, StatsError, chol_solve, cholesky_spd,
    form_xtx, invert_square,
};

use crate::adaptive_resample_waves::{DriveError, StopReason, WaveStoppingRule, run_waves_serial};
use crate::bootstrap_wave::{fill_worker_replicate, retained_index_slots};
use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;

/// Map a stats-layer error into [`EstimationError::Stats`]. A refusal record the stats layer
/// carries (a refused GLM fit, a rank-deficient design, too few clusters) stays in the error and
/// is read back as structured fields by [`EstimationError::refusal_fields`]; the text and the
/// variant are unchanged.
#[allow(clippy::needless_pass_by_value, reason = "statsError is small / owned at call sites")]
pub(crate) fn stats_err(e: StatsError) -> EstimationError {
    EstimationError::from(e)
}

/// Solve `A x = b` for symmetric positive-definite row-major `A` (`p × p`).
#[must_use]
pub(crate) fn solve_spd(a: &[f64], b: &[f64], p: usize) -> Option<Vec<f64>> {
    let chol = cholesky_spd(a, p)?;
    chol_solve(&chol, p, b)
}

/// The 64-bit identity word of a finished canonical stream: the first eight bytes of its
/// BLAKE3 digest, little-endian. For identities that are carried as a `u64` field; anything
/// stored as a whole digest uses [`CanonicalWriter::finish`] directly.
#[must_use]
pub(crate) fn canonical_word(writer: &antecedent_core::CanonicalWriter) -> u64 {
    let digest = writer.finish();
    u64::from_le_bytes([
        digest[0], digest[1], digest[2], digest[3], digest[4], digest[5], digest[6], digest[7],
    ])
}

/// `n` draws from `N(mean, variance)` on the seeded direct-draw stream of the shared
/// multivariate sampler; `None` when `variance` is not positive and finite (the sampler's
/// Cholesky refuses it) or the draw fails.
#[must_use]
pub(crate) fn normal_draws(mean: f64, variance: f64, n: usize, seed: u64) -> Option<Vec<f64>> {
    if !(variance.is_finite() && variance > 0.0) {
        return None;
    }
    antecedent_prob::sample_gaussian_mvn(
        &[mean],
        &[variance],
        n,
        seed,
        &mut antecedent_prob::LaplaceWorkspace::default(),
    )
    .ok()
    .map(|draws| draws.to_vec())
}

/// `X'X` (row-major `p × p`) of a column-major `n × p` design.
#[must_use]
pub(crate) fn gram(x: &[f64], n: usize, p: usize) -> Vec<f64> {
    let mut xtx = vec![0.0; p * p];
    form_xtx(x, n, p, &mut xtx);
    xtx
}

/// Require [`OverlapPolicy::ExplicitOverride`] (linear / IV / RD / front-door / GLM paths).
pub(crate) fn require_explicit_override(
    overlap: OverlapPolicy,
    message: &'static str,
) -> Result<(), EstimationError> {
    if overlap != OverlapPolicy::ExplicitOverride {
        return Err(EstimationError::Overlap { message });
    }
    Ok(())
}

/// Refuse [`OverlapPolicy::ExplicitOverride`] (propensity / AIPW paths — positivity mandatory).
pub(crate) fn refuse_explicit_override(
    overlap: OverlapPolicy,
    message: &'static str,
) -> Result<(), EstimationError> {
    if matches!(overlap, OverlapPolicy::ExplicitOverride) {
        return Err(EstimationError::Overlap { message });
    }
    Ok(())
}

/// A structured refusal whose message is `"{detail}: {message}"`: `detail` is the namespaced
/// reason literal a caller keys on, `message` the human text.
pub(crate) fn refuse(code: &'static str, detail: &str, message: &str) -> EstimationError {
    EstimationError::refused(code, format!("{detail}: {message}"))
}

/// Slack of every correlation-matrix factorization, relative to its unit diagonal: a pivot
/// within it of zero is a rank-deficient direction (perfectly collinear scores, an identical
/// estimand), one below it is an indefinite matrix. Also the tolerance of the unit-diagonal
/// and symmetry checks that precede the factorization.
pub(crate) const CORRELATION_PSD_TOLERANCE: f64 = 1e-9;

/// Two-sided normal p-value `2·(1 − Φ(|z|))` of a standardized statistic; `NaN` propagates.
pub(crate) fn normal_two_sided_p(z: f64) -> f64 {
    2.0 * antecedent_kernels::norm_sf(z.abs())
}

/// A row or parameter count as `f64`.
#[allow(clippy::cast_precision_loss, reason = "counts are far below 2^53")]
pub(crate) const fn count(n: usize) -> f64 {
    n as f64
}

pub(crate) use antecedent_kernels::{dot, mean};

/// Unbiased sample standard deviation; `NaN` if fewer than 2 observations.
pub(crate) use antecedent_kernels::sample_std;

/// Monte Carlo critical value of a max-statistic band: the `⌈level·(B+1)⌉`-th
/// smallest of `B` ascending-sorted replicate maxima.
///
/// The `B + 1` rank treats the observed statistic as one more exchangeable draw
/// (Davison & Hinkley 1997, §4.2), so the band is conservative wherever the rank
/// exists; `⌈level·B⌉` sits one rank lower whenever `level·B` is not an integer.
/// When `⌈level·(B+1)⌉ > B` (`B < level/(1−level)`: 19 at 0.95, 99 at 0.99) the
/// exact critical value is `+∞` — the sample maximum only covers `B/(B+1) < level`
/// — so `+∞` is returned rather than a narrower, anti-conservative maximum. Every
/// simultaneous response band (multiplier, Gaussian max-t and replicate sup-t)
/// reads its critical value here. `NaN` for an empty slice.
#[allow(
    clippy::cast_possible_truncation,
    reason = "the rank is at least 1 and the code below rejects any rank above b"
)]
#[allow(clippy::cast_sign_loss, reason = "the rank is clamped to at least 1 before the cast")]
pub(crate) fn monte_carlo_critical(sorted_maxima: &[f64], level: f64) -> f64 {
    let b = sorted_maxima.len();
    if b == 0 {
        return f64::NAN;
    }
    // The tolerance keeps a product that lands one ulp above an exact integer (e.g.
    // `0.95 · 20`) from bumping the rank past `B`.
    let rank = ((level * (b as f64 + 1.0) - 1e-9).ceil().max(1.0)) as usize;
    if rank > b {
        return f64::INFINITY;
    }
    sorted_maxima[rank - 1]
}

/// How many seeded posterior or resampling draws an estimator takes, and from which seed.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DrawSpec {
    /// Number of draws.
    pub draws: usize,
    /// Master seed of the draw stream.
    pub seed: u64,
}

impl DrawSpec {
    /// `draws` draws from `seed`.
    #[must_use]
    pub const fn new(draws: usize, seed: u64) -> Self {
        Self { draws, seed }
    }
}

/// Outcome of an IID bootstrap SE computation with failure accounting.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct BootstrapSeResult {
    /// Sample SD of successful replicate ATEs; `None` if too few survivors or too many failures.
    pub se: Option<f64>,
    /// Replicates that produced a finite estimate.
    pub replicates_ok: u32,
    /// Replicates that soft-failed (skipped) or did not yield an estimate.
    pub replicates_failed: u32,
    /// Cooperative cancellation stopped the loop early.
    pub cancelled: bool,
    /// Adaptive SE early-stop (orthogonal to [`Self::cancelled`]).
    pub early_stopped: bool,
}

impl BootstrapSeResult {
    /// Empty / skipped bootstrap (zero replicates requested).
    #[must_use]
    pub const fn skipped() -> Self {
        Self {
            se: None,
            replicates_ok: 0,
            replicates_failed: 0,
            cancelled: false,
            early_stopped: false,
        }
    }
}

/// Maximum allowed soft-failure fraction before refusing to report a bootstrap
/// summary: more than half the attempted replicates failing means the survivors
/// are not a resample of the estimator.
pub(crate) const BOOTSTRAP_MAX_FAILURE_FRAC: f64 = 0.5;

/// When a set of bootstrap replicates or posterior draws may be summarized.
///
/// One rule owns the decision for every transport interval: the estimator's
/// own failure-fraction refusal, the learned-trial refusal of any failed
/// replicate, and the facade's licensing floor are the same struct with
/// different thresholds, so no path re-derives the arithmetic.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct ReplicatePolicy {
    /// Fewest requested replicates that can earn the summary's label.
    pub min_requested: u32,
    /// Fewest successful replicates a summary may be computed from.
    pub min_ok: u32,
    /// Largest failed share of the attempted replicates.
    pub max_failure_frac: f64,
    /// Stable reason published when the summary is withheld.
    pub reason: &'static str,
}

impl ReplicatePolicy {
    /// The estimator's own rule: at least two survivors and no more than
    /// [`BOOTSTRAP_MAX_FAILURE_FRAC`] failures.
    pub const BOOTSTRAP: Self = Self {
        min_requested: 0,
        min_ok: 2,
        max_failure_frac: BOOTSTRAP_MAX_FAILURE_FRAC,
        reason: "bootstrap_failure_fraction",
    };
    /// The learned-trial rule: any failed replicate withholds the interval.
    pub const STRICT: Self = Self {
        min_requested: 2,
        min_ok: 2,
        max_failure_frac: 0.0,
        reason: "bootstrap_replicate_failure",
    };

    /// A nominal-coverage licensing floor on requested and successful replicates.
    #[must_use]
    pub const fn percentile_floor(min_replicates: u32) -> Self {
        Self {
            min_requested: min_replicates,
            min_ok: min_replicates,
            max_failure_frac: 1.0,
            reason: "insufficient_bootstrap_replicates",
        }
    }

    /// `Ok` when `ok` successful replicates out of `ok + failed` attempted, from
    /// `requested`, may be summarized; otherwise the withholding reason.
    ///
    /// # Errors
    /// The policy's `reason` when the summary must be withheld.
    pub fn decide(self, requested: u32, ok: u32, failed: u32) -> Result<(), &'static str> {
        let attempted = ok.saturating_add(failed);
        let fail_frac = if attempted == 0 { 1.0 } else { f64::from(failed) / f64::from(attempted) };
        if requested < self.min_requested || ok < self.min_ok || fail_frac > self.max_failure_frac {
            Err(self.reason)
        } else {
            Ok(())
        }
    }
}

/// Finalize a bootstrap SE from successful replicate ATEs.
#[cfg(test)]
#[must_use]
pub(crate) fn finalize_bootstrap_se(ates: &[f64], replicates: u32) -> BootstrapSeResult {
    finalize_bootstrap_se_ex(ates, replicates, false, false)
}

/// Finalize bootstrap SE with explicit cancellation / early-stop flags.
///
/// `attempted` counts only evaluated replicates, so unattempted draws are never
/// failures, while genuine failures remain visible even after cancellation.
#[must_use]
pub(crate) fn finalize_bootstrap_se_ex(
    ates: &[f64],
    attempted: u32,
    cancelled: bool,
    early_stopped: bool,
) -> BootstrapSeResult {
    let ok = u32::try_from(ates.len()).unwrap_or(u32::MAX);
    let failed = attempted.saturating_sub(ok);
    let too_few = ates.len() < 2;
    let fail_frac = if attempted == 0 { 0.0 } else { f64::from(failed) / f64::from(attempted) };
    let se = if too_few || fail_frac > BOOTSTRAP_MAX_FAILURE_FRAC {
        None
    } else {
        let s = sample_std(ates);
        if s.is_finite() { Some(s) } else { None }
    };
    BootstrapSeResult { se, replicates_ok: ok, replicates_failed: failed, cancelled, early_stopped }
}

/// Whether an enabled bootstrap budget may stop after `successful` replicates.
///
/// The stop is a replicate floor derived from the Monte Carlo error of a
/// bootstrap SE ([`AdaptiveBootstrapBudget::required_replicates`]); it does not
/// look at the SE trajectory, so the count is pinned by the budget alone.
#[must_use]
pub(crate) fn adaptive_bootstrap_should_stop(
    budget: AdaptiveBootstrapBudget,
    successful: u32,
) -> bool {
    budget.enabled && successful >= budget.required_replicates()
}

/// The bootstrap standard error with no per-worker scratch (see [`bootstrap_se_with_scratch`]).
#[cfg(test)]
pub(crate) fn bootstrap_se(
    replicates: u32,
    ctx: &ExecutionContext,
    stream_base: u64,
    n: usize,
    estimate: impl Fn(&[usize]) -> Result<Option<f64>, EstimationError> + Sync,
) -> Result<BootstrapSeResult, EstimationError> {
    bootstrap_se_with_scratch(replicates, ctx, stream_base, n, || (), |(), idx| estimate(idx))
}

/// IID bootstrap standard error with failure accounting and optional early-stop.
///
/// Each replicate draws `n` row indexes from the resample stream keyed by
/// `derive_stream(stream_base, replicate_id)`, so the draw is fixed by the seed and that id even when
/// evaluation stops early. Workers write those indexes into a reused buffer of length
/// `n`; an `n` by replicates index matrix is not retained.
///
/// `estimate` should return `Ok(Some(ate))` on success, `Ok(None)` for a soft-failed replicate
/// (counted as failed, bootstrap continues), or `Err` to abort the whole bootstrap.
///
/// Cooperative cancellation: when `ctx.cancellation` trips mid-loop, returns a partial result
/// with `cancelled = true` rather than inventing a full run.
///
/// Early-stop: when [`ExecutionContext::adaptive_bootstrap`] is enabled (no constructor
/// enables it) and the successful count reaches the budget's Monte Carlo error floor
/// before the request is exhausted, returns with `early_stopped = true` and the actual
/// count. Otherwise every requested replicate is evaluated.
///
/// Replicate evaluation carries per-worker scratch so it can run
/// under [`ExecutionContext::parallelism`] without a shared `&mut` workspace.
///
/// Retained index memory is `n` times the worker count (one buffer on the serial and
/// adaptive paths). Adaptive early-stop stays serial (the floor is a running success
/// count). When an outer map already saturates the thread budget, `ctx` is serial and
/// this path does not spawn. The worker count is also capped by the admission pool.
/// The workers and index bytes are leased on that pool for the whole run, and refused
/// before any replicate runs when they overflow or no longer fit in the pool.
pub(crate) fn bootstrap_se_with_scratch<S, F>(
    replicates: u32,
    ctx: &ExecutionContext,
    stream_base: u64,
    n: usize,
    make_scratch: impl Fn() -> S + Sync,
    estimate: F,
) -> Result<BootstrapSeResult, EstimationError>
where
    S: Send,
    F: Fn(&mut S, &[usize]) -> Result<Option<f64>, EstimationError> + Sync,
{
    if replicates == 0 || n == 0 {
        return Ok(BootstrapSeResult::skipped());
    }
    if ctx.cancellation.is_cancelled() {
        return Ok(finalize_bootstrap_se_ex(&[], 0, true, false));
    }
    let n_rep = replicates as usize;
    // One length-`n` buffer per worker. Adaptive early-stop does not spawn.
    let Some(scratch) = admit_bootstrap_scratch(ctx, n, n_rep)? else {
        return Ok(finalize_bootstrap_se_ex(&[], 0, true, false));
    };
    let workers = scratch.workers();
    if ctx.cancellation.is_cancelled() {
        return Ok(finalize_bootstrap_se_ex(&[], 0, true, false));
    }
    let rows = ReplicateRows::Rows { n, ctx, stream_base };
    if workers > 1 {
        return evaluate_bootstrap_parallel(ctx, &rows, n_rep, make_scratch, estimate);
    }
    evaluate_bootstrap_serial(ctx, &rows, n_rep, make_scratch, estimate)
}

/// A bootstrap run by the tolerance-based wave procedure: the standard error with its failure
/// accounting, the percentile interval of the successes, and the per-wave record.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct WaveBootstrap {
    /// Standard error and counts, as the fixed-count bootstrap reports them.
    pub se: BootstrapSeResult,
    /// Type-7 percentile interval of the successes at the rule's level.
    pub interval: Option<(f64, f64)>,
    /// One record per completed wave.
    pub waves: Vec<crate::adaptive_resample_waves::WaveRecord>,
    /// Estimated relative Monte Carlo error of the standard error.
    pub se_rel_mc_error: Option<f64>,
    /// Estimated relative Monte Carlo error of the interval half-width.
    pub half_width_rel_mc_error: Option<f64>,
    /// Why the run ended.
    pub stop: StopReason,
    /// Replicates attempted in the completed waves, failures included.
    pub attempted: u32,
}

impl WaveBootstrap {
    fn skipped() -> Self {
        Self {
            se: BootstrapSeResult::skipped(),
            interval: None,
            waves: Vec::new(),
            se_rel_mc_error: None,
            half_width_rel_mc_error: None,
            stop: StopReason::MaxReplicates,
            attempted: 0,
        }
    }

    /// A finished or stopped wave run as the interval procedure reports it.
    pub(crate) fn from_run(run: crate::adaptive_resample_waves::WaveRun) -> Self {
        Self {
            se: finalize_bootstrap_se_ex(
                &run.successes,
                run.attempted,
                run.stop == StopReason::Cancelled,
                run.stop == StopReason::Converged,
            ),
            interval: run.interval,
            waves: run.waves,
            se_rel_mc_error: run.se_rel_mc_error,
            half_width_rel_mc_error: run.half_width_rel_mc_error,
            stop: run.stop,
            attempted: run.attempted,
        }
    }

    fn cancelled() -> Self {
        Self {
            se: finalize_bootstrap_se_ex(&[], 0, true, false),
            stop: StopReason::Cancelled,
            ..Self::skipped()
        }
    }

    /// The record of this run under the `rule` it ran: the interval is withheld exactly when
    /// the standard error is (too few survivors or too many failures), so a summary never
    /// leans on replicates the fixed-count bootstrap would have refused.
    #[must_use]
    pub fn report(&self, rule: WaveStoppingRule) -> ResamplingReport {
        ResamplingReport {
            rule,
            interval: self.interval.filter(|_| self.se.se.is_some()),
            waves: self.waves.clone(),
            attempted: self.attempted,
            stop: self.stop,
            se_rel_mc_error: self.se_rel_mc_error,
            half_width_rel_mc_error: self.half_width_rel_mc_error,
        }
    }
}

/// The complete resampling procedure behind a requested interval: the rule it ran (waves,
/// tolerances, the replicate cap), the percentile interval, every wave's accounting, the
/// Monte Carlo error of the standard error and of the half-width at the stop, and why the run
/// stopped. Coverage of the whole stop-and-interval procedure is a measured property of the
/// rule, not asserted here.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct ResamplingReport {
    /// The declared schedule, tolerances, level and replicate cap.
    pub rule: WaveStoppingRule,
    /// Type-7 percentile interval of the successes at `rule.level`, when it is licensed.
    pub interval: Option<(f64, f64)>,
    /// One record per completed wave (attempts, successes, failures by kind).
    pub waves: Vec<crate::adaptive_resample_waves::WaveRecord>,
    /// Replicates attempted in the completed waves, failures included.
    pub attempted: u32,
    /// Why the run ended: a met tolerance, the replicate cap, or cancellation.
    pub stop: StopReason,
    /// Relative Monte Carlo error of the standard error at the stop.
    pub se_rel_mc_error: Option<f64>,
    /// Relative Monte Carlo error of the interval half-width at the stop.
    pub half_width_rel_mc_error: Option<f64>,
}

/// A requested bootstrap interval: the level and the Monte Carlo tolerances the resampling may
/// stop on. The replicate cap is the estimator's declared replicate count, so the request can
/// shorten the run but never lengthen it.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct IntervalRequest {
    /// Nominal level of the percentile interval, strictly inside `(0, 1)`.
    pub level: f64,
    /// Replicates per wave.
    pub wave_size: u32,
    /// Successes required before a tolerance may stop the run (clamped to the cap).
    pub min_replicates: u32,
    /// Relative Monte Carlo error target of the bootstrap standard error (kurtosis bound).
    pub se_rel_tol: Option<f64>,
    /// Relative Monte Carlo error target of the interval half-width.
    pub half_width_rel_tol: Option<f64>,
}

impl Default for IntervalRequest {
    fn default() -> Self {
        Self {
            level: 0.95,
            wave_size: 25,
            min_replicates: 100,
            se_rel_tol: Some(0.05),
            half_width_rel_tol: Some(0.10),
        }
    }
}

impl IntervalRequest {
    /// The wave rule this request declares under a cap of `replicates` attempts. Failures stay
    /// in the attempt count and never refuse here (the standard error and the interval are
    /// withheld by the fixed-count bootstrap's own failure fraction), and cancellation returns
    /// the completed waves.
    #[must_use]
    pub fn rule(&self, replicates: u32) -> WaveStoppingRule {
        WaveStoppingRule {
            wave_size: self.wave_size.max(1),
            min_replicates: self.min_replicates.min(replicates),
            max_replicates: replicates,
            se_rel_tol: self.se_rel_tol,
            half_width_rel_tol: self.half_width_rel_tol,
            success_floor: None,
            level: self.level,
            max_failure_fraction: 1.0,
            on_cancel: crate::adaptive_resample_waves::CancelPolicy::ReturnPartial,
        }
    }
}

/// A bootstrap standard error and, when an interval was requested, the resampling that
/// produced it.
#[derive(Clone, Debug, PartialEq)]
pub(crate) struct BootstrapOutcome {
    /// Standard error and failure accounting.
    pub(crate) se: BootstrapSeResult,
    /// The complete fit/resample/stop/interval record of a requested interval.
    pub(crate) resampling: Option<ResamplingReport>,
}

/// Bootstrap under the sampling design the labels declare, as a plain fixed-count standard
/// error (no `request`, identical to [`bootstrap_by_design`]) or as the wave procedure that
/// stops on the request's Monte Carlo tolerances and reports its interval and waves.
///
/// # Errors
///
/// The errors of [`bootstrap_by_design`] / [`bootstrap_interval_by_design`].
#[allow(
    clippy::too_many_arguments,
    reason = "the dispatcher of bootstrap_by_design, taking its arguments plus the interval request"
)]
pub(crate) fn bootstrap_outcome_by_design<S, F>(
    request: Option<&IntervalRequest>,
    design: BootstrapDesign<'_>,
    replicates: u32,
    ctx: &ExecutionContext,
    stream_base: u64,
    n: usize,
    make_scratch: impl Fn() -> S + Sync,
    estimate: F,
) -> Result<BootstrapOutcome, EstimationError>
where
    S: Send,
    F: Fn(&mut S, &[usize]) -> Result<Option<f64>, EstimationError> + Sync,
{
    let Some(request) = request else {
        let se =
            bootstrap_by_design(design, replicates, ctx, stream_base, n, make_scratch, estimate)?;
        return Ok(BootstrapOutcome { se, resampling: None });
    };
    let rule = request.rule(replicates);
    let run =
        bootstrap_interval_by_design(design, &rule, ctx, stream_base, n, make_scratch, estimate)?;
    Ok(BootstrapOutcome { resampling: Some(run.report(rule)), se: run.se })
}

/// The wave procedure over the rows `rows` draws: one reused scratch, waves on the caller.
fn interval_over_rows<S, F>(
    rule: &WaveStoppingRule,
    ctx: &ExecutionContext,
    stream_base: u64,
    rows: &ReplicateRows<'_>,
    make_scratch: impl Fn() -> S,
    estimate: F,
) -> Result<WaveBootstrap, EstimationError>
where
    F: Fn(&mut S, &[usize]) -> Result<Option<f64>, EstimationError>,
{
    let mut state = (make_scratch(), Vec::<usize>::new());
    let run = run_waves_serial(rule, ctx, stream_base, &mut state, |(scratch, idx), id, _rng| {
        let replicate = usize::try_from(id)
            .map_err(|_| EstimationError::data_msg("bootstrap replicate id exceeds usize"))?;
        rows.fill(replicate, idx)?;
        estimate(scratch, idx)
    });
    match run {
        Ok(run) => Ok(WaveBootstrap::from_run(run)),
        Err(DriveError::Estimator(error)) => Err(error),
        Err(DriveError::Wave(error)) => Err(EstimationError::data_msg(error.to_string())),
    }
}

/// IID bootstrap that stops on the Monte Carlo tolerances of `rule` and returns the interval.
///
/// The replicate rows are those of [`bootstrap_se_with_scratch`] on the same seed, evaluated on
/// the caller in waves of `rule.wave_size` with one reused scratch. An estimator error aborts.
/// The rule's cap is the replicate request.
///
/// # Errors
/// An unusable rule, an index buffer the pool cannot fund, or an estimator error.
pub fn bootstrap_interval_with_scratch<S, F>(
    rule: &WaveStoppingRule,
    ctx: &ExecutionContext,
    stream_base: u64,
    n: usize,
    make_scratch: impl Fn() -> S,
    estimate: F,
) -> Result<WaveBootstrap, EstimationError>
where
    F: Fn(&mut S, &[usize]) -> Result<Option<f64>, EstimationError>,
{
    if rule.max_replicates == 0 || n == 0 {
        return Ok(WaveBootstrap::skipped());
    }
    if ctx.cancellation.is_cancelled() {
        return Ok(WaveBootstrap::cancelled());
    }
    let Some(_lease) = admit_bootstrap_scratch(ctx, n, 1)? else {
        return Ok(WaveBootstrap::cancelled());
    };
    let rows = ReplicateRows::Rows { n, ctx, stream_base };
    interval_over_rows(rule, ctx, stream_base, &rows, make_scratch, estimate)
}

/// The wave procedure under the sampling design the labels declare, with the same design rules,
/// admission and replicate rows as [`bootstrap_by_design`]: whole clusters, contiguous blocks,
/// or single rows. Waves run on the caller with one reused scratch.
///
/// # Errors
///
/// The design refuses the draw, the pool cannot fund the index buffer, the rule is unusable, or
/// an estimator errs.
pub(crate) fn bootstrap_interval_by_design<S, F>(
    design: BootstrapDesign<'_>,
    rule: &WaveStoppingRule,
    ctx: &ExecutionContext,
    stream_base: u64,
    n: usize,
    make_scratch: impl Fn() -> S,
    estimate: F,
) -> Result<WaveBootstrap, EstimationError>
where
    F: Fn(&mut S, &[usize]) -> Result<Option<f64>, EstimationError>,
{
    let BootstrapDesign { clusters, series_block, declared } = design;
    if let Some(labels) = clusters {
        let grouped = ClusterRows::new(labels)?;
        let declared = declared.resolve(DeclaredSamplingDesign::ClusteredPanel);
        let draw = match declared {
            DeclaredSamplingDesign::IndependentSubjectTrajectories => ResamplingRule::RowSubjectIid,
            DeclaredSamplingDesign::ClusteredPanel
            | DeclaredSamplingDesign::DependentTimeSeries => ResamplingRule::WholeCluster,
        };
        declared.apply_resampling(draw).map_err(design_err)?;
        if rule.max_replicates == 0 || grouped.len() == 0 {
            return Ok(WaveBootstrap::skipped());
        }
        if ctx.cancellation.is_cancelled() {
            return Ok(WaveBootstrap::cancelled());
        }
        let ceiling = grouped.replicate_row_ceiling();
        let Some(_lease) = admit_bootstrap_scratch(ctx, ceiling, 1)? else {
            return Ok(WaveBootstrap::cancelled());
        };
        retained_index_slots(ceiling, 1)?;
        let rows =
            ReplicateRows::Clusters { members: grouped.members(), ceiling, ctx, stream_base };
        return interval_over_rows(rule, ctx, stream_base, &rows, make_scratch, estimate);
    }
    if let Some(block_length) = series_block {
        declared
            .resolve(DeclaredSamplingDesign::DependentTimeSeries)
            .apply_resampling(ResamplingRule::ContiguousBlock)
            .map_err(design_err)?;
        if rule.max_replicates == 0 || n == 0 {
            return Ok(WaveBootstrap::skipped());
        }
        if block_length == 0 || block_length > n {
            return Err(EstimationError::data_msg(
                "contiguous block length must be positive and no longer than the series",
            ));
        }
        if ctx.cancellation.is_cancelled() {
            return Ok(WaveBootstrap::cancelled());
        }
        let Some(_lease) = admit_bootstrap_scratch(ctx, n, 1)? else {
            return Ok(WaveBootstrap::cancelled());
        };
        let rows = ReplicateRows::Blocks { n, block_length, ctx, stream_base };
        return interval_over_rows(rule, ctx, stream_base, &rows, make_scratch, estimate);
    }
    match declared {
        SamplingDeclaration::Undeclared => DeclaredSamplingDesign::IndependentSubjectTrajectories
            .apply_resampling(ResamplingRule::RowSubjectIid),
        SamplingDeclaration::Declared(design) => design.apply_resampling(ResamplingRule::RowIid),
    }
    .map_err(design_err)?;
    bootstrap_interval_with_scratch(rule, ctx, stream_base, n, make_scratch, estimate)
}

fn design_err(error: SamplingDesignError) -> EstimationError {
    EstimationError::Data(DataError::SamplingDesign(error))
}

/// Whether the caller declared the sampling design of the rows being resampled.
///
/// [`Self::Undeclared`] is an explicit choice, not a missing value: the design is then the
/// one the row grouping implies (whole clusters are a clustered panel, blocks are a
/// dependent series, single rows are independent subjects). [`Self::Declared`] holds the
/// design the caller stated, and a draw that design does not allow is refused.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub enum SamplingDeclaration {
    /// No design declared: the grouping implies one.
    #[default]
    Undeclared,
    /// The caller declared this design.
    Declared(DeclaredSamplingDesign),
}

impl SamplingDeclaration {
    /// The declared design, or `implied` when none was declared.
    #[must_use]
    pub const fn resolve(self, implied: DeclaredSamplingDesign) -> DeclaredSamplingDesign {
        match self {
            Self::Undeclared => implied,
            Self::Declared(design) => design,
        }
    }
}

/// Source grouping that determines the resampling unit and dependence preservation.
#[derive(Clone, Copy, Debug)]
pub(crate) struct BootstrapDesign<'a> {
    /// Whole-cluster labels take precedence over a series block declaration.
    pub(crate) clusters: Option<&'a [u32]>,
    /// Contiguous time-series block length when no cluster labels are present.
    pub(crate) series_block: Option<usize>,
    /// The sampling design the caller declared. An undeclared design takes the one the
    /// grouping implies. A declared design that does not allow the draw the grouping
    /// implies refuses it.
    pub(crate) declared: SamplingDeclaration,
}

/// Bootstrap under the sampling design the labels declare.
///
/// Cluster labels draw whole clusters. A time-ordered series with no cluster labels
/// draws contiguous blocks of `series_block`. Otherwise each row is one subject, and
/// the subject rule is the draw.
///
/// # Errors
///
/// The design refuses the draw, or [`bootstrap_se_with_scratch`] fails.
pub(crate) fn bootstrap_by_design<S, F>(
    design: BootstrapDesign<'_>,
    replicates: u32,
    ctx: &ExecutionContext,
    stream_base: u64,
    n: usize,
    make_scratch: impl Fn() -> S + Sync,
    estimate: F,
) -> Result<BootstrapSeResult, EstimationError>
where
    S: Send,
    F: Fn(&mut S, &[usize]) -> Result<Option<f64>, EstimationError> + Sync,
{
    let BootstrapDesign { clusters, series_block, declared } = design;
    if let Some(labels) = clusters {
        let grouped = ClusterRows::new(labels)?;
        return bootstrap_se_clusters_under(
            declared.resolve(DeclaredSamplingDesign::ClusteredPanel),
            replicates,
            ctx,
            stream_base,
            &grouped,
            make_scratch,
            estimate,
        );
    }
    if let Some(block_length) = series_block {
        return bootstrap_se_blocks_with_scratch(
            declared.resolve(DeclaredSamplingDesign::DependentTimeSeries),
            replicates,
            ctx,
            stream_base,
            n,
            block_length,
            make_scratch,
            estimate,
        );
    }
    // No cluster labels and no block: every row is its own draw unit. Undeclared, the data
    // carries one row per subject and the subject rule is that draw. A declared design
    // never allows a row-level draw: a trajectory design needs subject labels, a panel
    // cluster labels, and a series a block length.
    match declared {
        SamplingDeclaration::Undeclared => DeclaredSamplingDesign::IndependentSubjectTrajectories
            .apply_resampling(ResamplingRule::RowSubjectIid),
        SamplingDeclaration::Declared(design) => design.apply_resampling(ResamplingRule::RowIid),
    }
    .map_err(design_err)?;
    bootstrap_se_with_scratch(replicates, ctx, stream_base, n, make_scratch, estimate)
}

#[allow(
    clippy::too_many_arguments,
    reason = "the block path of the dispatcher above, taking its arguments plus the block length"
)]
fn bootstrap_se_blocks_with_scratch<S, F>(
    declared: DeclaredSamplingDesign,
    replicates: u32,
    ctx: &ExecutionContext,
    stream_base: u64,
    n: usize,
    block_length: usize,
    make_scratch: impl Fn() -> S + Sync,
    estimate: F,
) -> Result<BootstrapSeResult, EstimationError>
where
    S: Send,
    F: Fn(&mut S, &[usize]) -> Result<Option<f64>, EstimationError> + Sync,
{
    declared.apply_resampling(ResamplingRule::ContiguousBlock).map_err(design_err)?;
    if replicates == 0 || n == 0 {
        return Ok(BootstrapSeResult::skipped());
    }
    if block_length == 0 || block_length > n {
        return Err(EstimationError::data_msg(
            "contiguous block length must be positive and no longer than the series",
        ));
    }
    if ctx.cancellation.is_cancelled() {
        return Ok(finalize_bootstrap_se_ex(&[], 0, true, false));
    }
    let n_rep = replicates as usize;
    let Some(scratch) = admit_bootstrap_scratch(ctx, n, n_rep)? else {
        return Ok(finalize_bootstrap_se_ex(&[], 0, true, false));
    };
    let workers = scratch.workers();
    if ctx.cancellation.is_cancelled() {
        return Ok(finalize_bootstrap_se_ex(&[], 0, true, false));
    }
    let rows = ReplicateRows::Blocks { n, block_length, ctx, stream_base };
    if workers > 1 {
        return evaluate_bootstrap_parallel(ctx, &rows, n_rep, make_scratch, estimate);
    }
    evaluate_bootstrap_serial(ctx, &rows, n_rep, make_scratch, estimate)
}

/// Reserve the index bytes for one bootstrap.
///
/// The lease holds `n` index slots per worker and no CPU slots. The parallel map
/// leases the workers, so this reservation does not block that acquire. Adaptive
/// early-stop runs on the caller with one buffer. `None` means the study was cancelled.
///
/// # Errors
///
/// An index buffer that overflows or no longer fits in the pool. Nothing is allocated.
fn admit_bootstrap_scratch(
    ctx: &ExecutionContext,
    n: usize,
    n_rep: usize,
) -> Result<Option<ScratchLease>, EstimationError> {
    let per_worker = index_buffer_bytes(retained_index_slots(n, 1)?)?;
    let work_items = if ctx.adaptive_bootstrap.enabled { 1 } else { n_rep };
    match ctx.admit_index_bytes(work_items, per_worker) {
        Ok(lease) => {
            index_buffer_bytes(retained_index_slots(n, lease.workers())?)?;
            Ok(Some(lease))
        }
        Err(ScratchRefusal::Cancelled) => Ok(None),
        Err(ScratchRefusal::PoolShort { .. }) => {
            Err(EstimationError::data_msg("admission pool cannot fund the bootstrap scratch"))
        }
        Err(_) => Err(EstimationError::data_msg(
            "bootstrap index allocation exceeds addressable capacity",
        )),
    }
}

fn index_buffer_bytes(slots: usize) -> Result<u64, EstimationError> {
    u64::try_from(slots)
        .ok()
        .and_then(|slots| slots.checked_mul(u64::try_from(size_of::<u64>()).unwrap_or(8)))
        .ok_or_else(|| {
            EstimationError::data_msg("bootstrap index allocation exceeds addressable capacity")
        })
}

/// Lease `count * width` values of `T` that a draw loop retains, before the buffer is built.
///
/// The lease holds bytes on the pool and no CPU slots. Hold it until the buffer is dropped or
/// handed to an owner that holds its own reservation.
///
/// # Errors
///
/// The study is cancelled, the byte product overflows, or the admission pool cannot fund it.
/// Nothing is allocated.
pub(crate) fn admit_array<T>(
    ctx: &ExecutionContext,
    count: usize,
    width: usize,
) -> Result<antecedent_core::execution::AdmissionLease, EstimationError> {
    let bytes = count
        .checked_mul(width)
        .and_then(|cells| cells.checked_mul(size_of::<T>().max(1)))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| EstimationError::data_msg("draw buffer size overflow"))?;
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => EstimationError::unsupported(
            "execution cancelled before the draw buffers were admitted",
        ),
        other => EstimationError::data_msg(format!(
            "admission pool cannot fund the draw buffers: {other}"
        )),
    })
}

/// [`admit_array`] for a function with no [`ExecutionContext`]: the workflow account bound by
/// [`antecedent_core::numerical_admission`], or `None` when the call is made outside any
/// workflow and nothing is charged.
///
/// # Errors
///
/// As [`admit_array`].
pub(crate) fn admit_bound_array<T>(
    count: usize,
    width: usize,
) -> Result<Option<antecedent_core::execution::AdmissionLease>, EstimationError> {
    let bytes = count
        .checked_mul(width)
        .and_then(|cells| cells.checked_mul(size_of::<T>().max(1)))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| EstimationError::data_msg("draw buffer size overflow"))?;
    antecedent_core::numerical_admission::acquire(bytes).transpose().map_err(|error| {
        EstimationError::data_msg(format!("admission pool cannot fund the draw buffers: {error}"))
    })
}

/// Lease the per-worker gather buffers of one bootstrap, before any replicate runs.
///
/// Each worker holds `rows * cells_per_row` `f64` cells (the resampled design, outcome and
/// working vectors it gathers into). The lease holds those bytes for all workers and no CPU
/// slots, because the parallel map leases the workers itself. Adaptive early-stop runs one
/// worker. A whole-cluster resample may exceed `rows` by the cluster-size slack, which
/// [`bootstrap_by_design`] admits separately as its index lease. Hold the returned lease
/// across the bootstrap call. `None` means the study was cancelled; the bootstrap that
/// follows reports that itself, so a caller may ignore it.
///
/// # Errors
///
/// The byte product overflows or the admission pool cannot fund it. Nothing is allocated.
pub fn admit_gather_scratch(
    ctx: &ExecutionContext,
    rows: usize,
    cells_per_row: usize,
    replicates: u32,
) -> Result<Option<ScratchLease>, EstimationError> {
    if replicates == 0 || rows == 0 {
        return Ok(None);
    }
    let per_worker = rows
        .checked_mul(cells_per_row)
        .and_then(|cells| cells.checked_mul(size_of::<f64>()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| {
            EstimationError::data_msg("bootstrap gather allocation exceeds addressable capacity")
        })?;
    let work_items = if ctx.adaptive_bootstrap.enabled { 1 } else { replicates as usize };
    match ctx.admit_index_bytes(work_items, per_worker) {
        Ok(lease) => Ok(Some(lease)),
        Err(ScratchRefusal::Cancelled) => Ok(None),
        Err(ScratchRefusal::PoolShort { .. }) => Err(EstimationError::data_msg(
            "admission pool cannot fund the bootstrap gather buffers",
        )),
        Err(_) => Err(EstimationError::data_msg(
            "bootstrap gather allocation exceeds addressable capacity",
        )),
    }
}

/// The rows of each cluster, in ascending label order, for whole-cluster resampling.
#[derive(Clone, Debug)]
pub(crate) struct ClusterRows {
    members: Vec<Vec<antecedent_data::GlobalId>>,
    rows: usize,
    largest: usize,
}

impl ClusterRows {
    /// Group row indices by cluster label, clusters in ascending label order.
    ///
    /// That is the order [`antecedent_data`] uses for [`antecedent_data::ResamplingPlan::ClusterBootstrap`].
    /// First appearance is a different order when a later label is smaller, so the same stream
    /// would expand different clusters.
    ///
    /// # Errors
    ///
    /// Fewer than two clusters (a one-cluster resample is the data), or a row that does
    /// not fit in a [`antecedent_data::GlobalId`].
    pub(crate) fn new(labels: &[u32]) -> Result<Self, EstimationError> {
        let rows = labels.len();
        let mut grouped: std::collections::BTreeMap<u32, Vec<antecedent_data::GlobalId>> =
            std::collections::BTreeMap::new();
        for (row, label) in labels.iter().enumerate() {
            let row = antecedent_data::GlobalId::new(u64::try_from(row).map_err(|_| {
                EstimationError::data_msg(
                    "row position does not fit in u64; rows are addressed by GlobalId and ChunkIndex only spans one bounded chunk",
                )
            })?);
            grouped.entry(*label).or_default().push(row);
        }
        if grouped.len() < 2 {
            return Err(crate::se::few_clusters(grouped.len()));
        }
        let members: Vec<Vec<antecedent_data::GlobalId>> = grouped.into_values().collect();
        let largest = members.iter().map(Vec::len).max().unwrap_or(0);
        Ok(Self { members, rows, largest })
    }

    /// Rows in the sample these clusters partition.
    #[cfg(test)]
    pub(crate) fn row_count(&self) -> usize {
        self.rows
    }

    /// Rows one replicate may hold, which admission charges and the draw enforces.
    ///
    /// A replicate is a sum of `groups` independent cluster sizes in `[0, largest]` with mean
    /// `rows`, so by Hoeffding's inequality it exceeds `rows + t` with probability at most
    /// `exp(-2 t² / (groups · largest²))`. With `t = 4 · largest · √groups` that is below
    /// `1e-13`, and the sure ceiling `groups · largest` caps it when there are few clusters.
    pub(crate) fn replicate_row_ceiling(&self) -> usize {
        let groups = self.members.len();
        let sure = groups.saturating_mul(self.largest);
        let slack = groups.isqrt().saturating_add(1).saturating_mul(4).saturating_mul(self.largest);
        sure.min(self.rows.saturating_add(slack))
    }

    /// Number of clusters.
    pub(crate) fn len(&self) -> usize {
        self.members.len()
    }

    /// The rows of each cluster, in row order.
    pub(crate) fn members(&self) -> &[Vec<antecedent_data::GlobalId>] {
        &self.members
    }
}

/// Whole-cluster bootstrap standard error with failure accounting.
///
/// Each replicate draws as many clusters as the data has, with replacement, and presents the
/// union of their rows (clusters in draw order, rows in data order) to `estimate`. A
/// replicate therefore has a *variable* number of rows: a cluster drawn twice contributes
/// twice, one not drawn not at all. This is the resampling a cluster-robust variance
/// assumes, with no row ever split from its cluster; a fixed-length row index would break
/// the clusters or the sample size.
///
/// Otherwise behaves as [`bootstrap_se_with_scratch`]: soft failures, cancellation, and
/// parallel evaluation are the same, and the draws are common-random-number stable under a
/// fixed seed.
///
/// Each worker writes one replicate. Retained index memory is the replicate row ceiling
/// ([`ClusterRows::replicate_row_ceiling`]) times the worker count, admitted before any draw
/// and enforced as each cluster is appended. The groups-by-replicates matrix is not allocated.
///
/// # Errors
///
/// The admission pool cannot fund that worker buffer, the buffer is not addressable,
/// or an `Err` from `estimate`.
#[cfg(test)]
pub(crate) fn bootstrap_se_clusters_with_scratch<S, F>(
    replicates: u32,
    ctx: &ExecutionContext,
    stream_base: u64,
    clusters: &ClusterRows,
    make_scratch: impl Fn() -> S + Sync,
    estimate: F,
) -> Result<BootstrapSeResult, EstimationError>
where
    S: Send,
    F: Fn(&mut S, &[usize]) -> Result<Option<f64>, EstimationError> + Sync,
{
    bootstrap_se_clusters_under(
        DeclaredSamplingDesign::ClusteredPanel,
        replicates,
        ctx,
        stream_base,
        clusters,
        make_scratch,
        estimate,
    )
}

/// Whole-cluster bootstrap under the `declared` sampling design. The design refuses the
/// draw when it does not bind whole-cluster resampling.
///
/// # Errors
///
/// The design refuses the draw, or the failures of [`bootstrap_se_clusters_with_scratch`].
pub(crate) fn bootstrap_se_clusters_under<S, F>(
    declared: DeclaredSamplingDesign,
    replicates: u32,
    ctx: &ExecutionContext,
    stream_base: u64,
    clusters: &ClusterRows,
    make_scratch: impl Fn() -> S + Sync,
    estimate: F,
) -> Result<BootstrapSeResult, EstimationError>
where
    S: Send,
    F: Fn(&mut S, &[usize]) -> Result<Option<f64>, EstimationError> + Sync,
{
    // Under independent subject trajectories the labels are subject ids, and drawing whole
    // subjects is that design's own rule; every other design draws whole clusters.
    let rule = match declared {
        DeclaredSamplingDesign::IndependentSubjectTrajectories => ResamplingRule::RowSubjectIid,
        DeclaredSamplingDesign::ClusteredPanel | DeclaredSamplingDesign::DependentTimeSeries => {
            ResamplingRule::WholeCluster
        }
    };
    declared.apply_resampling(rule).map_err(design_err)?;
    let groups = clusters.len();
    if replicates == 0 || groups == 0 {
        return Ok(BootstrapSeResult::skipped());
    }
    if ctx.cancellation.is_cancelled() {
        return Ok(finalize_bootstrap_se_ex(&[], 0, true, false));
    }
    let n_rep = replicates as usize;
    // One replicate buffer per worker; no groups-by-replicates matrix exists to admit.
    let ceiling = clusters.replicate_row_ceiling();
    let Some(scratch) = admit_bootstrap_scratch(ctx, ceiling, n_rep)? else {
        return Ok(finalize_bootstrap_se_ex(&[], 0, true, false));
    };
    let workers = scratch.workers();
    retained_index_slots(ceiling, workers)?;
    if ctx.cancellation.is_cancelled() {
        return Ok(finalize_bootstrap_se_ex(&[], 0, true, false));
    }
    let rows = ReplicateRows::Clusters { members: clusters.members(), ceiling, ctx, stream_base };
    if workers > 1 {
        return evaluate_bootstrap_parallel(ctx, &rows, n_rep, make_scratch, estimate);
    }
    evaluate_bootstrap_serial(ctx, &rows, n_rep, make_scratch, estimate)
}

/// How one bootstrap replicate's row indices are supplied to the estimator.
enum ReplicateRows<'a> {
    /// `n` iid row draws for replicate `r`, generated from `derive_stream(stream_base, r)`.
    ///
    /// Each worker fills its own length-`n` buffer. The `n` by replicates matrix is not
    /// retained.
    Rows { n: usize, ctx: &'a ExecutionContext, stream_base: u64 },
    /// One whole-cluster replicate, drawn into the caller buffer. The groups-by-replicates
    /// matrix is not retained. Length varies with the clusters drawn.
    Clusters {
        members: &'a [Vec<antecedent_data::GlobalId>],
        ceiling: usize,
        ctx: &'a ExecutionContext,
        stream_base: u64,
    },
    /// Contiguous blocks of `block_length` over `n` series rows.
    Blocks { n: usize, block_length: usize, ctx: &'a ExecutionContext, stream_base: u64 },
}

impl ReplicateRows<'_> {
    fn fill(&self, r: usize, out: &mut Vec<usize>) -> Result<(), EstimationError> {
        out.clear();
        let row = |id: antecedent_data::GlobalId| -> Result<usize, EstimationError> {
            usize::try_from(id.get()).map_err(|_| {
                EstimationError::data_msg(
                    "row position does not fit in u64; rows are addressed by GlobalId and ChunkIndex only spans one bounded chunk",
                )
            })
        };
        match self {
            Self::Rows { n, ctx, stream_base } => {
                let replicate = u64::try_from(r)
                    .map_err(|_| EstimationError::data_msg("bootstrap replicate id exceeds u64"))?;
                fill_worker_replicate(*n, replicate, ctx, *stream_base, out)?;
            }
            Self::Clusters { members, ceiling, ctx, stream_base } => {
                let groups = members.len();
                let replicate = u64::try_from(r)
                    .map_err(|_| EstimationError::data_msg("bootstrap replicate id exceeds u64"))?;
                let mut drawn = Vec::new();
                fill_resample_index_replicate(
                    ResamplingPlan::IidBootstrap,
                    groups,
                    None,
                    ctx,
                    *stream_base,
                    replicate,
                    &mut drawn,
                )?;
                for cluster in &drawn {
                    let slot = row(*cluster)?;
                    let Some(sources) = members.get(slot) else {
                        return Err(EstimationError::data_msg(
                            "cluster draw is outside the observed clusters",
                        ));
                    };
                    if out.len().saturating_add(sources.len()) > *ceiling {
                        return Err(EstimationError::data_msg(
                            "cluster replicate exceeds the admitted index buffer",
                        ));
                    }
                    for src in sources {
                        out.push(row(*src)?);
                    }
                }
            }
            Self::Blocks { n, block_length, ctx, stream_base } => {
                let replicate = u64::try_from(r)
                    .map_err(|_| EstimationError::data_msg("bootstrap replicate id exceeds u64"))?;
                let mut drawn = Vec::new();
                fill_resample_index_replicate(
                    ResamplingPlan::MovingBlock { length: *block_length },
                    *n,
                    None,
                    ctx,
                    *stream_base,
                    replicate,
                    &mut drawn,
                )?;
                for id in drawn {
                    out.push(row(id)?);
                }
            }
        }
        Ok(())
    }
}

/// The enabled adaptive budget, run as a wave procedure: one replicate per wave on the caller,
/// stopping once the budget's success floor is reached. The replicates drawn and the standard
/// error are those of the fixed-count loop on the same seed; a replicate whose estimator errs
/// aborts the bootstrap.
fn evaluate_bootstrap_waves<S, F>(
    ctx: &ExecutionContext,
    rows: &ReplicateRows<'_>,
    n_rep: usize,
    make_scratch: impl Fn() -> S,
    estimate: F,
) -> Result<BootstrapSeResult, EstimationError>
where
    F: Fn(&mut S, &[usize]) -> Result<Option<f64>, EstimationError>,
{
    let max_replicates = u32::try_from(n_rep)
        .map_err(|_| EstimationError::data_msg("bootstrap replicate count exceeds u32"))?;
    let rule = WaveStoppingRule::success_floor_rule(
        max_replicates,
        ctx.adaptive_bootstrap.required_replicates(),
    );
    let mut state = (make_scratch(), Vec::<usize>::new());
    let run = run_waves_serial(&rule, ctx, 0, &mut state, |(scratch, idx), id, _rng| {
        let replicate = usize::try_from(id)
            .map_err(|_| EstimationError::data_msg("bootstrap replicate id exceeds usize"))?;
        rows.fill(replicate, idx)?;
        estimate(scratch, idx)
    });
    match run {
        Ok(run) => Ok(finalize_bootstrap_se_ex(
            &run.successes,
            run.attempted,
            run.stop == StopReason::Cancelled,
            run.stop == StopReason::Converged,
        )),
        Err(DriveError::Estimator(error)) => Err(error),
        Err(DriveError::Wave(error)) => Err(EstimationError::data_msg(error.to_string())),
    }
}

fn evaluate_bootstrap_serial<S, F>(
    ctx: &ExecutionContext,
    rows: &ReplicateRows<'_>,
    n_rep: usize,
    make_scratch: impl Fn() -> S,
    estimate: F,
) -> Result<BootstrapSeResult, EstimationError>
where
    F: Fn(&mut S, &[usize]) -> Result<Option<f64>, EstimationError>,
{
    if ctx.adaptive_bootstrap.enabled {
        return evaluate_bootstrap_waves(ctx, rows, n_rep, make_scratch, estimate);
    }
    let mut scratch = make_scratch();
    let mut ates = Vec::with_capacity(n_rep);
    let mut idx = Vec::new();
    let mut cancelled = false;
    let mut early_stopped = false;
    let mut attempted = 0u32;
    let budget = ctx.adaptive_bootstrap;
    for r in 0..n_rep {
        if ctx.cancellation.is_cancelled() {
            cancelled = true;
            break;
        }
        rows.fill(r, &mut idx)?;
        attempted += 1;
        let estimate = estimate(&mut scratch, &idx)?.filter(|ate| ate.is_finite());
        if let Some(ate) = estimate {
            ates.push(ate);
        }
        if ctx.cancellation.is_cancelled() {
            cancelled = true;
            break;
        }
        if estimate.is_some() && r + 1 < n_rep {
            let ok_count = u32::try_from(ates.len()).unwrap_or(u32::MAX);
            if adaptive_bootstrap_should_stop(budget, ok_count) {
                early_stopped = true;
                if let Some(p) = &ctx.progress {
                    p.report((r + 1) as f64 / n_rep as f64, "bootstrap");
                }
                break;
            }
        }
        if let Some(p) = &ctx.progress {
            p.report((r + 1) as f64 / n_rep as f64, "bootstrap");
        }
    }
    Ok(finalize_bootstrap_se_ex(&ates, attempted, cancelled, early_stopped))
}

fn evaluate_bootstrap_parallel<S, F>(
    ctx: &ExecutionContext,
    rows: &ReplicateRows<'_>,
    n_rep: usize,
    make_scratch: impl Fn() -> S + Sync,
    estimate: F,
) -> Result<BootstrapSeResult, EstimationError>
where
    S: Send,
    F: Fn(&mut S, &[usize]) -> Result<Option<f64>, EstimationError> + Sync,
{
    let finished = std::sync::atomic::AtomicUsize::new(0);
    let first_cancel = std::sync::atomic::AtomicUsize::new(usize::MAX);
    // The workers are leased without waiting: a bootstrap that finds the pool held by
    // another study stays on its caller instead of queueing behind it.
    let values = ctx.map_indexed_with_scratch_if_free(
        n_rep,
        || (make_scratch(), Vec::<usize>::new()),
        |r, inner, (scratch, idx): &mut (S, Vec<usize>)| {
            if inner.cancellation.is_cancelled() {
                first_cancel.fetch_min(r, std::sync::atomic::Ordering::Relaxed);
                return Ok::<Option<f64>, EstimationError>(None);
            }
            rows.fill(r, idx)?;
            let value = estimate(scratch, idx)?.filter(|ate| ate.is_finite());
            let done = finished.fetch_add(1, std::sync::atomic::Ordering::Relaxed) + 1;
            if let Some(progress) = &inner.progress {
                progress.report(done as f64 / n_rep as f64, "bootstrap");
            }
            Ok(value)
        },
    )?;
    let stop = first_cancel.load(std::sync::atomic::Ordering::Relaxed);
    let mut ates = Vec::with_capacity(n_rep);
    let mut attempted = 0u32;
    for (r, value) in values.into_iter().enumerate() {
        if r >= stop {
            break;
        }
        attempted += 1;
        if let Some(ate) = value {
            ates.push(ate);
        }
    }
    let cancelled = stop != usize::MAX || ctx.cancellation.is_cancelled();
    Ok(finalize_bootstrap_se_ex(&ates, attempted, cancelled, false))
}

/// Coefficients and residuals of one `FaerBackend` least-squares solve.
pub(crate) struct LsFit {
    /// Coefficient vector, length `ncols`.
    pub coefficients: Vec<f64>,
    /// Residuals, length `nrows`.
    pub residuals: Vec<f64>,
    /// Numerical rank.
    pub rank: usize,
}

/// Solve `min ||X β − y||` on a column-major design.
///
/// The coefficients, residuals, and rank are the backend fit's fields.
///
/// # Errors
///
/// A stats-layer shape or numerical failure.
pub(crate) fn ls_fit(
    backend: FaerBackend,
    design_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    workspace: &mut LeastSquaresWorkspace,
) -> Result<LsFit, EstimationError> {
    let fit =
        backend.least_squares(design_colmajor, nrows, ncols, y, workspace).map_err(stats_err)?;
    Ok(LsFit { coefficients: fit.coefficients, residuals: fit.residuals, rank: fit.rank })
}

/// OLS residual variance `σ² = RSS / (n − p)` for a fitted coefficient vector.
pub(crate) fn ols_sigma2(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    y: &[f64],
    beta: &[f64],
) -> f64 {
    let mut rss = 0.0;
    for r in 0..nrows {
        let mut pred = 0.0;
        for c in 0..ncols {
            pred += x_colmajor[c * nrows + r] * beta[c];
        }
        let e = y[r] - pred;
        rss += e * e;
    }
    rss / (nrows.saturating_sub(ncols)).max(1) as f64
}

/// Variance of a single OLS coefficient: `σ² · [(XᵀX)⁻¹]_{jj}`.
///
/// Returns `NaN` if `XᵀX` is singular.
pub(crate) fn coefficient_variance(
    x_colmajor: &[f64],
    nrows: usize,
    ncols: usize,
    col: usize,
    sigma2: f64,
) -> f64 {
    let mut xtx = vec![0.0; ncols * ncols];
    form_xtx(x_colmajor, nrows, ncols, &mut xtx);
    let Some(inv) = invert_square(&xtx, ncols) else {
        return f64::NAN;
    };
    sigma2 * inv[col * ncols + col].max(0.0)
}

/// Compute `(XᵀX)⁻¹` for a column-major `nrows × ncols` design, or `None` if singular.
///
/// Shared `form_xtx` → `invert_square` idiom used by every analytic-SE helper that needs the
/// unscaled inverse-information matrix.
pub(crate) fn xtx_inverse(x_colmajor: &[f64], nrows: usize, ncols: usize) -> Option<Vec<f64>> {
    let mut xtx = vec![0.0; ncols * ncols];
    form_xtx(x_colmajor, nrows, ncols, &mut xtx);
    invert_square(&xtx, ncols)
}

/// Invert a small fixed-size matrix with [`invert_square`]'s scale-free rule, or `None` when it
/// is singular or non-finite.
///
/// The one inverse for the Bayesian IV / RD and local-polynomial families, which carry their
/// moment matrices as `[[f64; N]; N]`.
#[must_use]
pub(crate) fn invert_fixed<const N: usize>(a: &[[f64; N]; N]) -> Option<[[f64; N]; N]> {
    let flat: Vec<f64> = a.iter().flatten().copied().collect();
    let inv = invert_square(&flat, N)?;
    Some(std::array::from_fn(|i| std::array::from_fn(|j| inv[i * N + j])))
}

/// [`invert_fixed`] for a runtime-sized square matrix held as rows; `None` when the rows are not
/// square, or the matrix is singular or non-finite.
#[must_use]
pub(crate) fn invert_rows(rows: &[Vec<f64>]) -> Option<Vec<Vec<f64>>> {
    let n = rows.len();
    if rows.iter().any(|r| r.len() != n) {
        return None;
    }
    let flat: Vec<f64> = rows.iter().flatten().copied().collect();
    let inv = invert_square(&flat, n)?;
    Some(inv.chunks(n).map(<[f64]>::to_vec).collect())
}

/// Gather `idx`-selected entries of `src` into `out` (a reused scratch buffer) for one IID
/// bootstrap resample: `out[r] = src[idx[r]]`.
pub(crate) fn gather_bootstrap_vector(out: &mut [f64], src: &[f64], idx: &[usize]) {
    for (r, &i) in idx.iter().enumerate() {
        out[r] = src[i];
    }
}

/// Gather `idx`-selected rows of a column-major `n × ncols` matrix `src` into `out` (a reused
/// scratch buffer) for one IID bootstrap resample: `out[c*n+r] = src[c*n+idx[r]]`.
pub(crate) fn gather_bootstrap_design(
    out: &mut [f64],
    src: &[f64],
    n: usize,
    ncols: usize,
    idx: &[usize],
) {
    for (r, &i) in idx.iter().enumerate() {
        for c in 0..ncols {
            out[c * n + r] = src[c * n + i];
        }
    }
}

/// Gather `idx`-selected rows of a column-major matrix `src` of `src_rows` rows into `out`,
/// whose row count is `idx.len()`: `out[c·m + r] = src[c·src_rows + idx[r]]` with
/// `m = idx.len()`. For a resample as long as the data this is [`gather_bootstrap_design`];
/// a whole-cluster resample has its own length.
pub(crate) fn gather_resampled_design(
    out: &mut [f64],
    src: &[f64],
    src_rows: usize,
    ncols: usize,
    idx: &[usize],
) {
    let m = idx.len();
    for (r, &i) in idx.iter().enumerate() {
        for c in 0..ncols {
            out[c * m + r] = src[c * src_rows + i];
        }
    }
}

/// Delta-method SE for a linear contrast `gᵀ β`: `sqrt(σ² · gᵀ (XᵀX)⁻¹ g)`.
///
/// Returns `NaN` if the Gram matrix is singular or the quadratic form is non-finite.
pub(crate) fn delta_method_se(inv_xtx: &[f64], ncols: usize, g: &[f64], sigma2: f64) -> f64 {
    if g.len() != ncols || inv_xtx.len() != ncols * ncols {
        return f64::NAN;
    }
    // v = inv · g
    let mut v = vec![0.0; ncols];
    for i in 0..ncols {
        let mut s = 0.0;
        for j in 0..ncols {
            s += inv_xtx[i * ncols + j] * g[j];
        }
        v[i] = s;
    }
    let mut q = 0.0;
    for i in 0..ncols {
        q += g[i] * v[i];
    }
    let var = sigma2 * q;
    if !var.is_finite() {
        return f64::NAN;
    }
    var.max(0.0).sqrt()
}

/// Single-pass `(min, max)` fold.
///
/// Shared by the static and temporal response estimators so neither carries its own
/// min/max pair (two full-slice folds where one suffices).
pub(crate) fn range(values: &[f64]) -> (f64, f64) {
    values.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(lo, hi), &v| (lo.min(v), hi.max(v)))
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicU32, AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};

    use antecedent_core::execution::AdmissionPool;
    use antecedent_core::{AdaptiveBootstrapBudget, CancellationToken, ExecutionContext};

    use super::*;

    #[test]
    fn fixed_inverse_judges_singularity_scale_free() {
        // A design in units of 1e-8 has pivots far below the old absolute 1e-12 cutoff but is
        // perfectly conditioned once equilibrated.
        let tiny = [[1e-16, 0.0, 0.0], [0.0, 4e-16, 0.0], [0.0, 0.0, 9e-16]];
        let inv = invert_fixed(&tiny).expect("a scaled diagonal is not singular");
        assert!((inv[1][1] * 4e-16 - 1.0).abs() < 1e-12);
        // A genuinely collinear pair is still refused, at any scale.
        let collinear = [[1.0, 2.0], [2.0, 4.0]];
        assert!(invert_fixed(&collinear).is_none());
        let scaled = [[1e12, 2e12], [2e12, 4e12]];
        assert!(invert_fixed(&scaled).is_none());
        // Non-finite entries are refused rather than propagated.
        assert!(invert_fixed(&[[f64::NAN, 0.0], [0.0, 1.0]]).is_none());
    }

    #[test]
    fn row_inverse_matches_the_fixed_inverse_and_refuses_ragged_input() {
        let rows = vec![vec![1e-16, 0.0], vec![0.0, 4e-16]];
        let inv = invert_rows(&rows).expect("a scaled diagonal is not singular");
        assert!((inv[0][0] * 1e-16 - 1.0).abs() < 1e-12 && inv[0][1] == 0.0);
        assert!(invert_rows(&[vec![1.0, 2.0], vec![2.0, 4.0]]).is_none());
        assert!(invert_rows(&[vec![1.0, 0.0], vec![0.0]]).is_none());
    }

    #[test]
    fn cancelled_bootstrap_retains_actual_failures() {
        let ctx = ExecutionContext::for_tests(7);
        let calls = AtomicU32::new(0);
        let result = bootstrap_se(20, &ctx, 1, 10, |_| {
            let n = calls.fetch_add(1, Ordering::Relaxed) + 1;
            if n == 4 {
                ctx.cancellation.cancel();
            }
            Ok(if n <= 2 { Some(f64::from(n)) } else { None })
        })
        .unwrap();
        assert!(result.cancelled);
        assert!(!result.early_stopped);
        assert_eq!(result.replicates_ok, 2);
        assert_eq!(result.replicates_failed, 2);
        assert!(result.se.is_some());
    }

    #[test]
    fn early_stopped_bootstrap_retains_actual_failures() {
        let mut ctx = ExecutionContext::for_tests(7);
        // ε = 0.5 → floor ⌈1 + 1/(2·0.25)⌉ = 3 successes.
        ctx.adaptive_bootstrap =
            AdaptiveBootstrapBudget { enabled: true, min_replicates: 2, se_rel_epsilon: 0.5 };
        let calls = AtomicU32::new(0);
        let result = bootstrap_se(20, &ctx, 1, 10, |_| {
            let n = calls.fetch_add(1, Ordering::Relaxed) + 1;
            Ok(if n <= 4 { None } else { Some(1.0) })
        })
        .unwrap();
        assert!(result.early_stopped);
        assert_eq!(result.replicates_ok, 3);
        assert_eq!(result.replicates_failed, 4);
        assert!(result.se.is_none(), "majority failure must not be hidden by early stopping");
    }

    #[test]
    fn bootstrap_nonfinite_estimates_are_failures() {
        let ctx = ExecutionContext::for_tests(7);
        let values = [Some(1.0), Some(f64::NAN), Some(f64::INFINITY), Some(2.0)];
        let i = AtomicU32::new(0);
        let result = bootstrap_se(4, &ctx, 1, 5, |_| {
            let value = values[i.fetch_add(1, Ordering::Relaxed) as usize];
            Ok(value)
        })
        .unwrap();
        assert_eq!(result.replicates_ok, 2);
        assert_eq!(result.replicates_failed, 2);
        assert!(result.se.is_some());
    }

    #[test]
    fn bootstrap_cancellation_does_not_hide_unrelated_estimation_error() {
        let ctx = ExecutionContext::for_tests(7);
        let error = EstimationError::data_msg("unrelated malformed data");
        let result = bootstrap_se(4, &ctx, 1, 5, |_| {
            ctx.cancellation.cancel();
            Err(error.clone())
        });
        assert_eq!(result.unwrap_err(), error);
    }

    #[test]
    fn bootstrap_checks_capacity_before_allocation() {
        let ctx = ExecutionContext::for_tests(7);
        let result = bootstrap_se(2, &ctx, 1, usize::MAX, |_| panic!("must not estimate"));
        assert!(result.unwrap_err().to_string().contains("capacity"));
        ctx.cancellation.cancel();
        let cancelled =
            bootstrap_se(2, &ctx, 1, usize::MAX, |_| panic!("must not estimate")).unwrap();
        assert!(cancelled.cancelled);
        assert_eq!(cancelled.replicates_failed, 0);
    }

    #[test]
    fn finalize_refuses_se_when_too_few_or_too_many_failures() {
        assert!(finalize_bootstrap_se(&[], 10).se.is_none());
        assert!(finalize_bootstrap_se(&[1.0], 10).se.is_none());
        let many_fail = finalize_bootstrap_se(&[1.0, 1.1], 10);
        assert_eq!(many_fail.replicates_ok, 2);
        assert_eq!(many_fail.replicates_failed, 8);
        assert!(many_fail.se.is_none(), "80% failure must refuse SE");
        let ok = finalize_bootstrap_se(&[1.0, 1.2, 0.9, 1.1], 4);
        assert!(ok.se.is_some());
        assert_eq!(ok.replicates_failed, 0);
        assert!(!ok.early_stopped);
    }

    #[test]
    fn finalize_early_stop_does_not_count_unattempted_as_failures() {
        let r = finalize_bootstrap_se_ex(&[1.0, 1.1, 0.9, 1.05], 4, false, true);
        assert_eq!(r.replicates_ok, 4);
        assert_eq!(r.replicates_failed, 0);
        assert!(r.early_stopped);
        assert!(r.se.is_some());
    }

    #[test]
    fn adaptive_stop_is_the_monte_carlo_error_floor() {
        // ε = 5% needs 201 successes; the explicit floor of 4 is below that.
        let budget =
            AdaptiveBootstrapBudget { enabled: true, min_replicates: 4, se_rel_epsilon: 0.05 };
        assert!(!adaptive_bootstrap_should_stop(budget, 4));
        assert!(!adaptive_bootstrap_should_stop(budget, 200));
        assert!(adaptive_bootstrap_should_stop(budget, 201));
        // A loose ε defers to the explicit floor.
        let floored =
            AdaptiveBootstrapBudget { enabled: true, min_replicates: 10, se_rel_epsilon: 0.5 };
        assert!(!adaptive_bootstrap_should_stop(floored, 9));
        assert!(adaptive_bootstrap_should_stop(floored, 10));
        assert!(!adaptive_bootstrap_should_stop(AdaptiveBootstrapBudget::disabled(), u32::MAX));
    }

    #[test]
    fn bootstrap_early_stop_count_is_pinned_by_the_budget() {
        let mut ctx = ExecutionContext::for_tests(42);
        // ε = 0.2 → ⌈1 + 1/(2·0.04)⌉ = 14 successes.
        ctx.adaptive_bootstrap =
            AdaptiveBootstrapBudget { enabled: true, min_replicates: 5, se_rel_epsilon: 0.2 };
        // A constant ATE would have tripped the old relative-change rule at
        // once; the floor ignores the trajectory.
        let r1 = bootstrap_se(80, &ctx, 0xABCD, 20, |_| Ok(Some(2.0))).unwrap();
        let r2 = bootstrap_se(80, &ctx, 0xABCD, 20, |_| Ok(Some(2.0))).unwrap();
        assert!(r1.early_stopped);
        assert_eq!(r1.replicates_ok, 14);
        assert_eq!(r1.replicates_ok, r2.replicates_ok);
        assert!(!r1.cancelled);
        // A request at or below the floor is evaluated in full, not "stopped".
        let exact = bootstrap_se(14, &ctx, 0xABCD, 20, |_| Ok(Some(2.0))).unwrap();
        assert!(!exact.early_stopped);
        assert_eq!(exact.replicates_ok, 14);
        let below = bootstrap_se(10, &ctx, 0xABCD, 20, |_| Ok(Some(2.0))).unwrap();
        assert!(!below.early_stopped);
        assert_eq!(below.replicates_ok, 10);
    }

    #[test]
    fn enabled_default_budget_never_stops_below_its_floor() {
        let mut ctx = ExecutionContext::for_tests(42);
        ctx.adaptive_bootstrap = AdaptiveBootstrapBudget::enabled_default();
        let r = bootstrap_se(199, &ctx, 0xABCD, 20, |_| Ok(Some(2.0))).unwrap();
        assert!(!r.early_stopped);
        assert_eq!(r.replicates_ok, 199);
        let r = bootstrap_se(400, &ctx, 0xABCD, 20, |_| Ok(Some(2.0))).unwrap();
        assert!(r.early_stopped);
        assert_eq!(r.replicates_ok, 201);
    }

    #[test]
    fn bootstrap_cancel_and_early_stop_are_independent() {
        let mut ctx = ExecutionContext::for_tests(7);
        ctx.adaptive_bootstrap = AdaptiveBootstrapBudget::disabled();
        let token = CancellationToken::new();
        token.cancel();
        ctx.cancellation = token;
        let r = bootstrap_se(40, &ctx, 0x1111, 10, |_| Ok(Some(1.0))).unwrap();
        assert!(r.cancelled);
        assert!(!r.early_stopped);
        assert_eq!(r.replicates_ok, 0);
    }

    #[test]
    fn monte_carlo_critical_is_infinite_when_the_exact_rank_does_not_exist() {
        let maxima: Vec<f64> = (1..=10).map(f64::from).collect();
        // ⌈0.95·11⌉ = 11 > 10: the sample maximum only covers 10/11 < 0.95.
        assert!(monte_carlo_critical(&maxima, 0.95).is_infinite());
        // B = 19: ⌈0.95·20⌉ = 19 ≤ 19, so the maximum is the exact critical value.
        let nineteen: Vec<f64> = (1..=19).map(f64::from).collect();
        assert!((monte_carlo_critical(&nineteen, 0.95) - 19.0).abs() < 1e-12);
        // B = 99 at 0.99: ⌈0.99·100⌉ = 99.
        let ninety_nine: Vec<f64> = (1..=99).map(f64::from).collect();
        assert!((monte_carlo_critical(&ninety_nine, 0.99) - 99.0).abs() < 1e-12);
        let ninety_eight: Vec<f64> = (1..=98).map(f64::from).collect();
        assert!(monte_carlo_critical(&ninety_eight, 0.99).is_infinite());
        // Interior rank: B = 100, ⌈0.95·101⌉ = 96.
        let hundred: Vec<f64> = (1..=100).map(f64::from).collect();
        assert!((monte_carlo_critical(&hundred, 0.95) - 96.0).abs() < 1e-12);
        assert!(monte_carlo_critical(&[], 0.95).is_nan());
    }
    #[test]
    fn cluster_bootstrap_replicates_are_unions_of_whole_clusters_with_variable_length() {
        // Clusters of 1, 2, 3 and 5 rows, interleaved in the data.
        let labels = [0_u32, 1, 2, 3, 1, 2, 3, 2, 3, 3, 3];
        let clusters = ClusterRows::new(&labels).unwrap();
        assert_eq!(clusters.len(), 4);
        let sizes = [1_usize, 2, 3, 5];
        let lengths = std::sync::Mutex::new(std::collections::BTreeSet::new());
        let ctx = ExecutionContext::for_tests(11);
        let run = || {
            bootstrap_se_clusters_with_scratch(
                200,
                &ctx,
                9,
                &clusters,
                || (),
                |(), rows| {
                    // Every drawn cluster is present in full, as many times as it was drawn.
                    let mut seen = [0_usize; 4];
                    for &row in rows {
                        seen[labels[row] as usize] += 1;
                    }
                    let mut draws = 0;
                    for (label, count) in seen.iter().enumerate() {
                        assert_eq!(count % sizes[label], 0, "cluster {label} was split");
                        draws += count / sizes[label];
                    }
                    assert_eq!(draws, 4, "a replicate draws as many clusters as the data has");
                    lengths.lock().unwrap().insert(rows.len());
                    Ok(Some(rows.len() as f64))
                },
            )
            .unwrap()
        };
        let first = run();
        assert_eq!(first.replicates_ok, 200);
        assert!(lengths.lock().unwrap().len() > 3, "replicate lengths must vary");
        // Same seed, same draws.
        assert_eq!(run().se, first.se);
        // One cluster is not resamplable.
        assert!(ClusterRows::new(&[7, 7, 7]).is_err());
    }

    #[test]
    fn cluster_replicate_ceiling_covers_the_sample_and_is_enforced_when_skewed() {
        let mut labels = vec![0_u32; 50];
        labels.extend((1..=20_u32).flat_map(|label| [label, label]));
        let clusters = ClusterRows::new(&labels).unwrap();
        let ceiling = clusters.replicate_row_ceiling();
        assert!(ceiling >= clusters.row_count());
        assert!(ceiling <= clusters.len() * 50);
        let ctx = ExecutionContext::for_tests(7);
        let tight = ReplicateRows::Clusters {
            members: clusters.members(),
            ceiling: 49,
            ctx: &ctx,
            stream_base: 7,
        };
        assert!(tight.fill(0, &mut Vec::new()).is_err());
        let admitted = ReplicateRows::Clusters {
            members: clusters.members(),
            ceiling,
            ctx: &ctx,
            stream_base: 7,
        };
        let mut out = Vec::new();
        admitted.fill(0, &mut out).unwrap();
        assert!(out.len() <= ceiling);
    }

    #[test]
    fn cluster_members_follow_ascending_labels() {
        let labels = [2_u32, 0, 2, 0];
        let clusters = ClusterRows::new(&labels).unwrap();
        let ids =
            |slot: usize| clusters.members()[slot].iter().map(|id| id.get()).collect::<Vec<_>>();
        assert_eq!(ids(0), vec![1, 3]);
        assert_eq!(ids(1), vec![0, 2]);
        let ctx = ExecutionContext::for_tests(9);
        let stream = 0xA1D5_u64;
        let mut from_rows = Vec::new();
        ReplicateRows::Clusters {
            members: clusters.members(),
            ceiling: clusters.replicate_row_ceiling(),
            ctx: &ctx,
            stream_base: stream,
        }
        .fill(0, &mut from_rows)
        .unwrap();
        let mut from_data = Vec::new();
        fill_resample_index_replicate(
            ResamplingPlan::ClusterBootstrap { cluster: antecedent_core::VariableId::from_raw(0) },
            labels.len(),
            Some(&labels),
            &ctx,
            stream,
            0,
            &mut from_data,
        )
        .unwrap();
        let data_rows: Vec<usize> =
            from_data.iter().map(|id| usize::try_from(id.get()).unwrap()).collect();
        assert_eq!(from_rows, data_rows);
    }

    #[test]
    fn cluster_bootstrap_writes_one_replicate_per_worker() {
        let labels = [0_u32, 1, 2, 3, 1, 2, 3, 2, 3, 3, 3];
        let clusters = ClusterRows::new(&labels).unwrap();
        let groups = clusters.len();
        let replicates = 80u32;
        let workers = 4usize;
        let width = usize::try_from(u64::try_from(size_of::<u64>()).unwrap()).unwrap();
        let matrix_bytes = groups * usize::try_from(replicates).unwrap() * width;
        let retained_bytes = clusters.replicate_row_ceiling() * workers * width;
        assert!(retained_bytes < matrix_bytes);
        let ctx = ExecutionContext::production(11, u32::try_from(workers).unwrap()).with_admission(
            AdmissionPool::fixed(
                u32::try_from(workers).unwrap(),
                u64::try_from(retained_bytes).unwrap(),
            ),
        );
        let calls = AtomicU32::new(0);
        let result = bootstrap_se_clusters_with_scratch(
            replicates,
            &ctx,
            9,
            &clusters,
            || (),
            |(), rows| {
                calls.fetch_add(1, Ordering::Relaxed);
                assert!(!rows.is_empty());
                Ok(Some(rows.len() as f64))
            },
        )
        .unwrap();
        assert_eq!(result.replicates_ok, replicates);
        assert_eq!(calls.load(Ordering::Relaxed), replicates);
        let refused = ExecutionContext::production(11, u32::try_from(workers).unwrap())
            .with_admission(AdmissionPool::fixed(
                u32::try_from(workers).unwrap(),
                u64::try_from(retained_bytes - 1).unwrap(),
            ));
        let refused_calls = AtomicU32::new(0);
        let error = bootstrap_se_clusters_with_scratch(
            replicates,
            &refused,
            9,
            &clusters,
            || (),
            |(), _rows| {
                refused_calls.fetch_add(1, Ordering::Relaxed);
                Ok(Some(1.0_f64))
            },
        )
        .unwrap_err();
        assert_eq!(refused_calls.load(Ordering::Relaxed), 0);
        assert!(error.to_string().contains("cannot fund"), "{error}");
    }

    #[test]
    fn bootstrap_workers_stay_within_the_pool() {
        let shared = AdmissionPool::fixed(2, 1 << 20);
        let ctx = ExecutionContext::production(3, 8).with_admission(shared.clone());
        let barrier = Arc::new(Barrier::new(2));
        let inflight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let n = 5usize;
        let result = bootstrap_se(8, &ctx, 1, n, |idx| {
            assert_eq!(idx.len(), n);
            let now = inflight.fetch_add(1, Ordering::SeqCst).saturating_add(1);
            peak.fetch_max(now, Ordering::SeqCst);
            barrier.wait();
            inflight.fetch_sub(1, Ordering::SeqCst);
            Ok(Some(1.0))
        })
        .unwrap();
        let widest = peak.load(Ordering::SeqCst);
        assert_eq!(widest, 2);
        let width = usize::try_from(u64::try_from(size_of::<u64>()).unwrap()).unwrap();
        let scratch_bytes = n * widest * width;
        assert!(scratch_bytes <= usize::try_from(shared.byte_budget()).unwrap());
        assert_eq!(result.replicates_ok, 8);
        assert_eq!(inflight.load(Ordering::SeqCst), 0);
        assert_eq!(ctx.memory_limit_bytes(), Some(1 << 20));
        assert_eq!(shared.available().threads(), 2);
        assert_eq!(shared.available().bytes(), 1 << 20);
    }

    #[test]
    fn bootstrap_scratch_products_use_checked_arithmetic() {
        let calls = AtomicUsize::new(0);
        let estimate = |_: &[usize]| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(Some(1.0))
        };
        let width = usize::try_from(u64::try_from(size_of::<u64>()).unwrap()).unwrap();
        // One buffer is addressable but two workers' buffers are not.
        let ctx = ExecutionContext::production(1, 2);
        let n = usize::try_from(isize::MAX).unwrap() / width;
        let error = bootstrap_se(4, &ctx, 1, n, estimate).unwrap_err();
        assert!(error.to_string().contains("addressable capacity"), "{error}");
        // The byte count of a single buffer wraps to zero.
        let serial = ExecutionContext::for_tests(1);
        let n = usize::MAX / width + 1;
        let error = bootstrap_se(4, &serial, 1, n, estimate).unwrap_err();
        assert!(error.to_string().contains("addressable capacity"), "{error}");
        // The cluster path refuses an oversized row count before any replicate.
        let _clusters = ClusterRows::new(&[0, 0, 1, 1]).unwrap();
        let error = retained_index_slots(usize::MAX / width + 1, 1).unwrap_err();
        assert!(error.to_string().contains("addressable capacity"), "{error}");
        assert_eq!(calls.load(Ordering::SeqCst), 0);
    }

    #[test]
    fn a_concurrent_bootstrap_finds_the_pool_held_and_stays_on_its_caller() {
        let n = 5usize;
        let shared = AdmissionPool::fixed(2, 1 << 20);
        let ctx = ExecutionContext::production(3, 8).with_admission(shared.clone());
        let ready = Barrier::new(3);
        let done = Barrier::new(3);
        let second_threads = std::sync::Mutex::new(std::collections::HashSet::new());
        std::thread::scope(|scope| {
            let holder = scope.spawn(|| {
                bootstrap_se(2, &ctx, 1, n, |_| {
                    ready.wait();
                    done.wait();
                    Ok(Some(1.0))
                })
                .unwrap();
            });
            // Both holder workers are inside the estimate, so the pool's slots are leased.
            ready.wait();
            assert_eq!(shared.available().threads(), 0);
            let result = bootstrap_se(8, &ctx, 2, n, |_| {
                second_threads.lock().unwrap().insert(std::thread::current().id());
                Ok(Some(1.0))
            })
            .unwrap();
            assert_eq!(result.replicates_ok, 8);
            done.wait();
            holder.join().unwrap();
        });
        assert_eq!(second_threads.lock().unwrap().len(), 1);
        assert_eq!(shared.available().threads(), 2);
        assert_eq!(shared.available().bytes(), 1 << 20);
    }

    #[test]
    fn a_pool_that_exactly_funds_the_bootstrap_scratch_runs_it() {
        let n = 8usize;
        let workers = 2usize;
        let bytes = u64::try_from(n * workers).unwrap() * u64::try_from(size_of::<u64>()).unwrap();
        let shared = AdmissionPool::fixed(u32::try_from(workers).unwrap(), bytes);
        let ctx = ExecutionContext::production(1, u32::try_from(workers).unwrap())
            .with_admission(shared.clone());
        let calls = AtomicUsize::new(0);
        let result = bootstrap_se(4, &ctx, 1, n, |_| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(Some(1.0))
        })
        .unwrap();
        assert_eq!(result.replicates_ok, 4);
        assert_eq!(calls.load(Ordering::SeqCst), 4);
        assert_eq!(ctx.memory_limit_bytes(), Some(bytes));
        assert_eq!(shared.available().bytes(), bytes);
    }

    #[test]
    fn bootstrap_refuses_when_the_pool_cannot_fund_scratch() {
        let n = 8usize;
        let workers = 2usize;
        let bytes = u64::try_from(n * workers).unwrap() * u64::try_from(size_of::<u64>()).unwrap();
        let shared = AdmissionPool::fixed(u32::try_from(workers).unwrap(), bytes - 1);
        let ctx = ExecutionContext::production(1, u32::try_from(workers).unwrap())
            .with_admission(shared.clone());
        let calls = AtomicUsize::new(0);
        let error = bootstrap_se(4, &ctx, 1, n, |_| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(Some(1.0))
        })
        .unwrap_err();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(error.to_string().contains("cannot fund"), "{error}");
        assert_eq!(shared.available().threads(), u32::try_from(workers).unwrap());
        assert_eq!(shared.available().bytes(), bytes - 1);
        assert_eq!(ctx.memory_limit_bytes(), Some(bytes - 1));
    }

    fn design_run(
        design: BootstrapDesign<'_>,
        n: usize,
    ) -> (Result<BootstrapSeResult, EstimationError>, usize) {
        let ctx = ExecutionContext::for_tests(5);
        let calls = AtomicUsize::new(0);
        let result = bootstrap_by_design(
            design,
            8,
            &ctx,
            3,
            n,
            || (),
            |(), _rows| {
                calls.fetch_add(1, Ordering::Relaxed);
                Ok(Some(1.0))
            },
        );
        (result, calls.load(Ordering::Relaxed))
    }

    fn is_design_refusal(result: &Result<BootstrapSeResult, EstimationError>) -> bool {
        matches!(result, Err(EstimationError::Data(DataError::SamplingDesign(_))))
    }

    #[test]
    fn a_row_level_bootstrap_is_refused_under_every_declared_design() {
        for design in DeclaredSamplingDesign::ALL {
            let (result, calls) = design_run(
                BootstrapDesign {
                    clusters: None,
                    series_block: None,
                    declared: SamplingDeclaration::Declared(design),
                },
                12,
            );
            assert!(is_design_refusal(&result), "{design}: row draw was accepted");
            assert_eq!(calls, 0, "{design}: a refused draw evaluated replicates");
        }
    }

    #[test]
    fn an_undeclared_design_takes_the_one_the_grouping_implies() {
        let undeclared = SamplingDeclaration::Undeclared;
        let labels = [0_u32, 0, 1, 1, 2, 2];
        let (rows, _) = design_run(
            BootstrapDesign { clusters: None, series_block: None, declared: undeclared },
            6,
        );
        assert!(rows.is_ok());
        let (clusters, _) = design_run(
            BootstrapDesign { clusters: Some(&labels), series_block: None, declared: undeclared },
            6,
        );
        assert!(clusters.is_ok());
        let (blocks, _) = design_run(
            BootstrapDesign { clusters: None, series_block: Some(2), declared: undeclared },
            6,
        );
        assert!(blocks.is_ok());
    }

    #[test]
    fn a_trajectory_bootstrap_draws_whole_subjects_under_the_trajectory_design() {
        let subjects = [0_u32, 0, 0, 1, 1, 1, 2, 2, 2];
        let ctx = ExecutionContext::for_tests(5);
        let result = bootstrap_by_design(
            BootstrapDesign {
                clusters: Some(&subjects),
                series_block: None,
                declared: SamplingDeclaration::Declared(
                    DeclaredSamplingDesign::IndependentSubjectTrajectories,
                ),
            },
            16,
            &ctx,
            3,
            subjects.len(),
            || (),
            |(), rows| {
                let mut seen = [0_usize; 3];
                for &row in rows {
                    seen[subjects[row] as usize] += 1;
                }
                assert!(seen.iter().all(|count| count % 3 == 0), "a trajectory was split");
                Ok(Some(1.0))
            },
        )
        .expect("whole-subject draws are the trajectory rule");
        assert_eq!(result.replicates_ok, 16);
    }

    #[test]
    fn a_declared_design_refuses_the_grouping_that_contradicts_it() {
        let labels = [0_u32, 0, 1, 1];
        let series = SamplingDeclaration::Declared(DeclaredSamplingDesign::DependentTimeSeries);
        let panel = SamplingDeclaration::Declared(DeclaredSamplingDesign::ClusteredPanel);
        let (clusters_under_series, _) = design_run(
            BootstrapDesign { clusters: Some(&labels), series_block: None, declared: series },
            4,
        );
        assert!(is_design_refusal(&clusters_under_series));
        let (blocks_under_panel, _) = design_run(
            BootstrapDesign { clusters: None, series_block: Some(2), declared: panel },
            4,
        );
        assert!(is_design_refusal(&blocks_under_panel));
        let (blocks_under_series, _) = design_run(
            BootstrapDesign { clusters: None, series_block: Some(2), declared: series },
            4,
        );
        assert!(blocks_under_series.is_ok());
    }
}

/// Prints one wall-clock trial per line. Ignored unless the run asks for it.
#[cfg(test)]
mod bootstrap_clock_trials {
    use std::env;
    use std::time::Instant;

    use antecedent_core::ExecutionContext;

    use super::bootstrap_se_with_scratch;

    fn env_u64(name: &str, default: u64) -> u64 {
        env::var(name).ok().and_then(|value| value.parse().ok()).unwrap_or(default)
    }

    #[test]
    #[ignore = "prints bootstrap scratch wall-clock trials"]
    fn print_bootstrap_scratch_trials() {
        let n = usize::try_from(env_u64("N", 10_000)).unwrap_or(10_000);
        let replicates = u32::try_from(env_u64("REPLICATES", 50)).unwrap_or(50);
        let threads = u32::try_from(env_u64("THREADS", 1)).unwrap_or(1);
        let trials = usize::try_from(env_u64("TRIALS", 11)).unwrap_or(11);
        let ctx = ExecutionContext::production(11, threads);
        let run = || {
            bootstrap_se_with_scratch(
                replicates,
                &ctx,
                9,
                n,
                || (),
                |(), idx| Ok(Some(idx.len() as f64)),
            )
        };
        for _ in 0..2 {
            run().expect("warmup");
        }
        println!("n={n} replicates={replicates} threads={threads} trials={trials}");
        for trial in 0..trials {
            let started = Instant::now();
            let result = run().expect("trial");
            println!(
                "trial={trial} nanos={} replicates_ok={} cancelled={}",
                started.elapsed().as_nanos(),
                result.replicates_ok,
                result.cancelled
            );
        }
    }
}

#[cfg(test)]
mod interval_procedure_tests {
    use std::cell::RefCell;

    use super::*;
    use crate::adaptive_resample_waves::StopReason;
    use crate::statistical_transport::percentile_interval;

    const N: usize = 40;

    fn statistic(rows: &[usize]) -> f64 {
        rows.iter().map(|row| ((row * 37) % 11) as f64).sum::<f64>() / rows.len() as f64
    }

    fn labels() -> Vec<u32> {
        (0..N).map(|row| u32::try_from(row / 4).unwrap()).collect()
    }

    /// Every replicate-row design the dispatcher resolves: single rows, whole clusters and
    /// contiguous blocks.
    fn designs(labels: &[u32]) -> Vec<BootstrapDesign<'_>> {
        vec![
            BootstrapDesign {
                clusters: None,
                series_block: None,
                declared: SamplingDeclaration::Undeclared,
            },
            BootstrapDesign {
                clusters: Some(labels),
                series_block: None,
                declared: SamplingDeclaration::Undeclared,
            },
            BootstrapDesign {
                clusters: None,
                series_block: Some(4),
                declared: SamplingDeclaration::Undeclared,
            },
        ]
    }

    fn rule(wave_size: u32, max_replicates: u32, se_rel_tol: f64) -> WaveStoppingRule {
        IntervalRequest {
            level: 0.9,
            wave_size,
            min_replicates: 0,
            se_rel_tol: Some(se_rel_tol),
            half_width_rel_tol: None,
        }
        .rule(max_replicates)
    }

    #[test]
    fn a_never_met_request_reproduces_the_fixed_count_bootstrap_on_every_design() {
        let ctx = ExecutionContext::for_tests(5);
        let labels = labels();
        for design in designs(&labels) {
            let fixed = bootstrap_by_design(
                design,
                60,
                &ctx,
                21,
                N,
                || (),
                |(), rows| Ok(Some(statistic(rows))),
            )
            .unwrap();
            let seen = RefCell::new(Vec::new());
            let run = bootstrap_interval_by_design(
                design,
                &rule(15, 60, 1e-12),
                &ctx,
                21,
                N,
                || (),
                |(), rows| {
                    let value = statistic(rows);
                    seen.borrow_mut().push(value);
                    Ok(Some(value))
                },
            )
            .unwrap();
            assert_eq!(run.stop, StopReason::MaxReplicates);
            assert_eq!((run.attempted, run.waves.len()), (60, 4));
            assert_eq!(run.se.replicates_ok, fixed.replicates_ok);
            assert_eq!(run.se.se.map(f64::to_bits), fixed.se.map(f64::to_bits));
            // The interval is the type-7 percentile of exactly the successes the fixed count draws.
            assert_eq!(run.interval, percentile_interval(&seen.borrow(), 0.9));
            assert!(run.se_rel_mc_error.is_some());
        }
    }

    #[test]
    fn a_met_tolerance_stops_on_a_wave_boundary_with_the_prefix_of_the_full_run() {
        let ctx = ExecutionContext::for_tests(5);
        let labels = labels();
        for design in designs(&labels) {
            let record = |rule: &WaveStoppingRule| {
                let seen = RefCell::new(Vec::new());
                let run = bootstrap_interval_by_design(
                    design,
                    rule,
                    &ctx,
                    21,
                    N,
                    || (),
                    |(), rows| {
                        let value = statistic(rows);
                        seen.borrow_mut().push(value);
                        Ok(Some(value))
                    },
                )
                .unwrap();
                (run, seen.into_inner())
            };
            let (full, all) = record(&rule(20, 400, 1e-12));
            assert_eq!(full.attempted, 400);
            let mut loose = rule(20, 400, 0.3);
            loose.min_replicates = 40;
            let (stopped, prefix) = record(&loose);
            assert_eq!(stopped.stop, StopReason::Converged);
            assert!(stopped.se.early_stopped);
            assert_eq!((stopped.attempted, stopped.waves.len()), (40, 2));
            assert_eq!(prefix, all[..40]);
            assert!(stopped.se_rel_mc_error.is_some_and(|error| error <= 0.3));
        }
    }

    #[test]
    fn the_dispatcher_without_a_request_is_the_fixed_count_bootstrap_bit_for_bit() {
        let ctx = ExecutionContext::for_tests(5);
        let labels = labels();
        for design in designs(&labels) {
            let fixed = bootstrap_by_design(
                design,
                50,
                &ctx,
                3,
                N,
                || (),
                |(), rows| Ok(Some(statistic(rows))),
            )
            .unwrap();
            let outcome = bootstrap_outcome_by_design(
                None,
                design,
                50,
                &ctx,
                3,
                N,
                || (),
                |(), rows| Ok(Some(statistic(rows))),
            )
            .unwrap();
            assert_eq!(outcome.se, fixed);
            assert!(outcome.resampling.is_none());
            // A request reports the procedure beside the same replicate cap.
            let request = IntervalRequest { min_replicates: 10, ..IntervalRequest::default() };
            let requested = bootstrap_outcome_by_design(
                Some(&request),
                design,
                50,
                &ctx,
                3,
                N,
                || (),
                |(), rows| Ok(Some(statistic(rows))),
            )
            .unwrap();
            let report = requested.resampling.expect("a requested interval is reported");
            assert_eq!(report.rule.max_replicates, 50);
            assert_eq!(report.attempted, requested.se.replicates_ok);
            assert!(report.interval.is_some() && !report.waves.is_empty());
        }
    }

    #[test]
    fn failures_stay_in_the_attempts_and_withhold_the_interval_with_the_standard_error() {
        let ctx = ExecutionContext::for_tests(5);
        let calls = RefCell::new(0u32);
        let run = bootstrap_interval_by_design(
            BootstrapDesign {
                clusters: None,
                series_block: None,
                declared: SamplingDeclaration::Undeclared,
            },
            &rule(10, 40, 1e-12),
            &ctx,
            8,
            N,
            || (),
            |(), rows| {
                *calls.borrow_mut() += 1;
                Ok((*calls.borrow() % 4 == 0).then(|| statistic(rows)))
            },
        )
        .unwrap();
        assert_eq!(run.attempted, 40);
        assert_eq!((run.se.replicates_ok, run.se.replicates_failed), (10, 30));
        assert!(run.waves.iter().all(|wave| wave.declined == 7 || wave.declined == 8));
        // Three quarters failed: the fixed-count bootstrap withholds the SE, and the report
        // withholds the interval it would otherwise compute from the survivors.
        assert!(run.se.se.is_none());
        assert!(run.interval.is_some());
        assert!(run.report(rule(10, 40, 1e-12)).interval.is_none());
    }

    #[test]
    fn cancellation_returns_the_completed_waves_flagged() {
        let ctx = ExecutionContext::for_tests(5);
        ctx.cancellation.cancel();
        let run = bootstrap_interval_by_design(
            BootstrapDesign {
                clusters: None,
                series_block: None,
                declared: SamplingDeclaration::Undeclared,
            },
            &rule(10, 40, 1e-12),
            &ctx,
            8,
            N,
            || (),
            |(), rows| Ok(Some(statistic(rows))),
        )
        .unwrap();
        assert_eq!(run.stop, StopReason::Cancelled);
        assert!(run.se.cancelled && run.interval.is_none() && run.waves.is_empty());
    }
}

/// Shared assertions of the estimator-level interval-procedure tests: a request whose tolerances
/// are never met reproduces the fixed-count bootstrap, and one met early stops on a wave boundary.
#[cfg(test)]
pub(crate) mod interval_test_support {
    use super::IntervalRequest;
    use crate::adaptive_resample_waves::StopReason;
    use crate::effect_estimate::EffectEstimate;

    /// Tolerances no finite run can meet, so the procedure runs to the replicate cap.
    pub(crate) fn never_met(wave_size: u32) -> IntervalRequest {
        IntervalRequest {
            level: 0.95,
            wave_size,
            min_replicates: 0,
            se_rel_tol: Some(1e-12),
            half_width_rel_tol: Some(1e-12),
        }
    }

    /// Tolerances any run meets, so the procedure stops at the first wave that has
    /// `min_replicates` successes.
    pub(crate) fn met_early(wave_size: u32, min_replicates: u32) -> IntervalRequest {
        IntervalRequest {
            level: 0.95,
            wave_size,
            min_replicates,
            se_rel_tol: Some(1e9),
            half_width_rel_tol: Some(1e9),
        }
    }

    /// `never` reproduces `fixed` bit for bit and uses the whole cap in whole waves; `early`
    /// stops before the cap on a wave boundary with at least the requested successes.
    pub(crate) fn assert_procedure(
        fixed: &EffectEstimate,
        never: &EffectEstimate,
        early: &EffectEstimate,
        cap: u32,
        wave: u32,
    ) {
        assert!(fixed.diagnostics.resampling.is_none(), "a fixed count reports no procedure");
        let never_report =
            never.diagnostics.resampling.as_ref().expect("a requested interval is reported");
        assert_eq!(never_report.stop, StopReason::MaxReplicates);
        assert_eq!(never_report.attempted, cap);
        assert_eq!(never_report.waves.len() as u32, cap / wave);
        assert_eq!(
            never.se_bootstrap().map(f64::to_bits),
            fixed.se_bootstrap().map(f64::to_bits),
            "an unmet tolerance is the fixed-count bootstrap"
        );
        let early_report =
            early.diagnostics.resampling.as_ref().expect("a requested interval is reported");
        assert_ne!(early_report.stop, StopReason::MaxReplicates);
        assert!(early_report.attempted < cap, "a met tolerance shortens the run");
        assert_eq!(early_report.attempted % wave, 0, "the run ends on a wave boundary");
        assert!(early.se_bootstrap().is_some_and(f64::is_finite));
    }
}
