//! Bootstrap orchestration: standard errors, wave-based intervals and the sampling design a
//! resample must respect.
//!
//! [`bootstrap_se_with_scratch`] is the iid row draw; [`bootstrap_by_design`] dispatches on the
//! declared grouping (whole clusters, contiguous blocks or single rows) to the loops in
//! [`rows`]; [`bootstrap_interval_by_design`] is the same dispatch run as the Monte Carlo
//! tolerance wave procedure; [`bootstrap_outcome_by_design`] chooses between the two from an
//! optional [`IntervalRequest`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    AdaptiveBootstrapBudget, DeclaredSamplingDesign, ExecutionContext, ResamplingRule,
    SamplingDesignError,
};
use antecedent_data::DataError;

use super::sample_std;
use super::scratch::admit_bootstrap_scratch;
use crate::adaptive_resample_waves::{DriveError, StopReason, WaveStoppingRule, run_waves_serial};
use crate::bootstrap_wave::retained_index_slots;
use crate::error::EstimationError;

pub(crate) mod rows;
#[cfg(test)]
mod tests;

use rows::{
    ClusterRows, ReplicateRows, bootstrap_se_clusters_under, evaluate_bootstrap_parallel,
    evaluate_bootstrap_serial,
};

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
    let mut state = (make_scratch(), Vec::<usize>::new(), Vec::new());
    let run = run_waves_serial(rule, ctx, stream_base, &mut state, |(scratch, idx, drawn), id, _rng| {
        let replicate = usize::try_from(id)
            .map_err(|_| EstimationError::data_msg("bootstrap replicate id exceeds usize"))?;
        rows.fill(replicate, idx, drawn)?;
        estimate(scratch, idx)
    });
    match run {
        Ok(run) => Ok(WaveBootstrap::from_run(run)),
        Err(DriveError::Estimator(error)) => Err(error),
        Err(DriveError::Wave(error)) => Err(EstimationError::Wave(error)),
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
