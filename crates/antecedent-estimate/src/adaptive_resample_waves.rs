//! Deterministic adaptive resampling in waves, with cancellation and failure accounting.
//!
//! Replicates are evaluated in waves of a fixed size. Replicate `i` draws from
//! [`StreamDomain::Resample`] at [`derive_stream`]`(stream_base, i)`, so its draw depends on the
//! seed, the base, and `i` alone: not on the wave size, the thread count, or a retry. The
//! derivation mixes the two words instead of combining them by exclusive-or, so a small constant base and a small
//! id never alias (`0 ^ 1` and `1 ^ 0` share a stream under the XOR form).
//!
//! After each wave the stopping rule reads only the successes drawn so far and decides whether
//! to run another wave. The rule stops when every declared relative tolerance is met, and never
//! before `min_replicates` successes. It always stops at `max_replicates` attempts.
//!
//! * `se_rel_tol` bounds the Monte Carlo error of the bootstrap standard error, estimated from
//!   the sample kurtosis of the successes: `sqrt((kurtosis - 1) / (4 B))`.
//! * `half_width_rel_tol` bounds the Monte Carlo error of the percentile interval half-width,
//!   estimated from order statistics around each endpoint rank (Maritz and Jarrett 1978 style
//!   band of one standard error).
//! * `success_floor` is the replicate floor of a bootstrap standard error, a fixed count of
//!   successes. It is the rule [`WaveStoppingRule::success_floor_rule`] declares.
//!
//! # Production use
//!
//! Every production bootstrap (`util::bootstrap_se_with_scratch` and the cluster and block
//! designs behind `util::bootstrap_by_design`) evaluates an enabled
//! [`antecedent_core::AdaptiveBootstrapBudget`] through [`run_waves_serial`]: the budget's
//! floor is the rule's `success_floor`, one replicate is a wave, the replicate rows are those of
//! the fixed-count loop, failures stay in the attempt count, and an estimator error aborts. The
//! tolerance rules above are used by callers that report an interval.
//!
//! Both are errors of the resampling Monte Carlo, never of the estimator's sampling error.
//! A tolerance on a degenerate sample (zero spread) cannot be assessed and counts as not met.
//!
//! # What the rule preserves and what it does not
//!
//! * Replay: the same seed, base, rule, and estimator give the same replicates, the same stop,
//!   and the same bits, on any thread count.
//! * Prefix: the successes of a stopped run are the first successes of the run with no
//!   tolerance, so stopping early never changes a replicate that was drawn.
//! * Fixed-B equivalence: a rule whose tolerances are never met runs `max_replicates` and
//!   returns exactly the fixed-B result at `B = max_replicates`.
//!
//! Stopping on the width of the interval is optional stopping. The stop time depends on the
//! replicates, so the interval at the stop is not the fixed-B interval at that count and the
//! coverage guarantee of the fixed-B percentile procedure does not carry over unchanged. Only
//! the fixed-B-equivalent branch inherits it. The coverage of the whole procedure (draw, stop,
//! interval) is a property to be measured by a coverage study of this exact rule; this module
//! asserts none. The stop also reduces replicate evaluation only: the caller's full-sample fit
//! is never part of a wave and is neither repeated nor skipped here.
//!
//! Failed replicates (an estimator error, a non-finite value, or a declined replicate) stay in
//! the attempt count and are tallied per wave by kind. The interval and standard error are
//! computed from the successes and reported next to the attempt count. A cumulative failure
//! fraction above `max_failure_fraction` refuses with [`WaveError::FailureFraction`]; nothing
//! is dropped silently.
//!
//! Cancellation is observed before each wave and before each replicate. A wave interrupted
//! part-way is discarded whole, so the replicates returned are always whole waves and a
//! function of the rule alone. [`CancelPolicy`] then either refuses or returns the completed
//! waves flagged [`StopReason::Cancelled`]. A wave's scratch is a local vector that is dropped
//! on every exit.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CausalRng, ExecutionContext, StreamDomain, derive_stream};
use antecedent_data::DataError;
use antecedent_kernels::sample_std;
use thiserror::Error;

use crate::error::EstimationError;
use crate::statistical_transport::percentile_interval;

/// What a cancelled run returns.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CancelPolicy {
    /// Refuse with [`WaveError::Cancelled`]; no partial result is returned.
    Refuse,
    /// Return the completed waves with [`StopReason::Cancelled`] set.
    ReturnPartial,
}

/// Declared wave schedule, stopping tolerances, and failure limit.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct WaveStoppingRule {
    /// Replicates per wave. The last wave is shortened to fit `max_replicates`.
    pub wave_size: u32,
    /// Successes required before a tolerance may stop the run.
    pub min_replicates: u32,
    /// Cap on attempted replicates, failures included.
    pub max_replicates: u32,
    /// Relative Monte Carlo error target of the bootstrap standard error. `None` is not tested.
    pub se_rel_tol: Option<f64>,
    /// Relative Monte Carlo error target of the interval half-width. `None` is not tested.
    pub half_width_rel_tol: Option<f64>,
    /// Successes after which the run may stop: the replicate floor of a bootstrap standard
    /// error. `None` is not tested. Every declared condition must hold at once.
    pub success_floor: Option<u32>,
    /// Nominal level of the percentile interval, strictly inside `(0, 1)`.
    pub level: f64,
    /// Largest tolerated cumulative failed fraction of attempted replicates, in `[0, 1]`.
    pub max_failure_fraction: f64,
    /// Behavior when the cancellation token trips.
    pub on_cancel: CancelPolicy,
}

/// Refusals of the wave procedure.
#[derive(Debug, Error)]
pub enum WaveError {
    /// The declared rule is not usable.
    #[error("invalid adaptive wave rule: {0}")]
    InvalidRule(&'static str),
    /// The cumulative failed fraction exceeded the declared limit after a wave.
    #[error(
        "{failed} of {attempted} bootstrap replicates failed after wave {wave}, above the \
         declared failure fraction {max_fraction}"
    )]
    FailureFraction {
        /// Zero-based wave after which the limit was exceeded.
        wave: u32,
        /// Replicates attempted so far.
        attempted: u32,
        /// Failed replicates so far.
        failed: u32,
        /// Declared limit.
        max_fraction: f64,
    },
    /// Cancelled under [`CancelPolicy::Refuse`].
    #[error(
        "adaptive resampling cancelled after {completed_waves} completed waves \
         ({attempted} attempted replicates)"
    )]
    Cancelled {
        /// Waves that finished before the stop.
        completed_waves: u32,
        /// Replicates in those waves.
        attempted: u32,
    },
}

/// Why the run ended.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StopReason {
    /// Every declared tolerance was met at `min_replicates` successes or more.
    Converged,
    /// `max_replicates` were attempted: the fixed-B-equivalent branch.
    MaxReplicates,
    /// The token tripped; only completed waves are reported.
    Cancelled,
}

/// Replicate counts of one finished wave.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct WaveRecord {
    /// Zero-based wave index.
    pub wave: u32,
    /// Id of the first replicate in the wave.
    pub first_replicate: u32,
    /// Replicates attempted in the wave.
    pub attempted: u32,
    /// Finite successes.
    pub ok: u32,
    /// Replicates whose estimator returned an error.
    pub estimator_errors: u32,
    /// Replicates whose estimator returned a NaN or infinite value.
    pub non_finite: u32,
    /// Replicates whose estimator declined with `Ok(None)`.
    pub declined: u32,
}

impl WaveRecord {
    /// Failed replicates of every kind.
    #[must_use]
    pub fn failed(&self) -> u32 {
        self.estimator_errors + self.non_finite + self.declined
    }
}

/// Finished or stopped run.
#[derive(Clone, Debug, PartialEq)]
pub struct WaveRun {
    /// Finite successes in replicate-id order.
    pub successes: Vec<f64>,
    /// One record per completed wave.
    pub waves: Vec<WaveRecord>,
    /// Replicates attempted in completed waves.
    pub attempted: u32,
    /// Finite successes.
    pub ok: u32,
    /// Failed replicates of every kind.
    pub failed: u32,
    /// Why the run ended.
    pub stop: StopReason,
    /// Message of the lowest-id estimator error, when there was one.
    pub first_error: Option<String>,
    /// Sample standard deviation of the successes (`None` under two successes).
    pub se: Option<f64>,
    /// Type-7 percentile interval of the successes.
    pub interval: Option<(f64, f64)>,
    /// Estimated relative Monte Carlo error of [`Self::se`], when assessable.
    pub se_rel_mc_error: Option<f64>,
    /// Estimated relative Monte Carlo error of the interval half-width, when assessable.
    pub half_width_rel_mc_error: Option<f64>,
}

impl WaveRun {
    /// The run ended before its rule did.
    #[must_use]
    pub fn is_partial(&self) -> bool {
        self.stop == StopReason::Cancelled
    }
}

/// Random stream of replicate `replicate` under `stream_base`.
///
/// The id is mixed with the base by [`derive_stream`], not combined by exclusive-or, so no two small
/// `(base, replicate)` pairs share a stream.
#[must_use]
pub fn wave_replicate_rng(ctx: &ExecutionContext, stream_base: u64, replicate: u64) -> CausalRng {
    ctx.rng.stream_for(StreamDomain::Resample, derive_stream(stream_base, replicate))
}

impl WaveStoppingRule {
    /// Check the declared rule.
    ///
    /// # Errors
    ///
    /// [`WaveError::InvalidRule`] for a zero wave size or cap, a minimum above the cap, a level
    /// outside `(0, 1)`, a non-positive or non-finite tolerance, or a failure fraction outside
    /// `[0, 1]`.
    pub fn validate(&self) -> Result<(), WaveError> {
        let bad = |reason| Err(WaveError::InvalidRule(reason));
        if self.wave_size == 0 || self.max_replicates == 0 {
            return bad("wave size and replicate cap must be positive");
        }
        if self.min_replicates > self.max_replicates {
            return bad("minimum replicates exceed the replicate cap");
        }
        if !(self.level > 0.0 && self.level < 1.0) {
            return bad("interval level must lie strictly inside (0, 1)");
        }
        if !(0.0..=1.0).contains(&self.max_failure_fraction) {
            return bad("failure fraction must lie in [0, 1]");
        }
        if self.success_floor == Some(0) {
            return bad("success floor must be positive");
        }
        let tolerances = [self.se_rel_tol, self.half_width_rel_tol];
        if tolerances.into_iter().flatten().any(|tol| !(tol.is_finite() && tol > 0.0)) {
            return bad("relative tolerances must be positive and finite");
        }
        Ok(())
    }

    /// The rule of a bootstrap standard error that stops at a success floor: one replicate per
    /// wave, stopping as soon as `floor` successes are in hand, never refusing on the failure
    /// fraction (the caller's summary applies its own), and returning completed replicates on
    /// cancellation.
    #[must_use]
    pub const fn success_floor_rule(max_replicates: u32, floor: u32) -> Self {
        Self {
            wave_size: 1,
            min_replicates: 0,
            max_replicates,
            se_rel_tol: None,
            half_width_rel_tol: None,
            success_floor: Some(floor),
            level: 0.95,
            max_failure_fraction: 1.0,
            on_cancel: CancelPolicy::ReturnPartial,
        }
    }

    fn tolerances_met(&self, successes: &[f64]) -> bool {
        if self.se_rel_tol.is_none()
            && self.half_width_rel_tol.is_none()
            && self.success_floor.is_none()
        {
            return false;
        }
        let ok = u32::try_from(successes.len()).unwrap_or(u32::MAX);
        if ok < self.min_replicates.max(2) || self.success_floor.is_some_and(|floor| ok < floor) {
            return false;
        }
        let within = |tol: Option<f64>, error: Option<f64>| match tol {
            None => true,
            Some(tol) => error.is_some_and(|error| error <= tol),
        };
        within(self.se_rel_tol, se_relative_mc_error(successes))
            && within(self.half_width_rel_tol, half_width_relative_mc_error(successes, self.level))
    }
}

fn count_f64(count: usize) -> f64 {
    f64::from(u32::try_from(count).unwrap_or(u32::MAX))
}

/// Relative Monte Carlo error of the sample standard deviation, `sqrt((kurtosis - 1) / (4 B))`.
///
/// Uses the population-moment kurtosis `m4 / m2^2` of the `B` values. `None` under four values
/// or when the spread is zero or non-finite, where the relative error is undefined.
#[must_use]
pub fn se_relative_mc_error(values: &[f64]) -> Option<f64> {
    if values.len() < 4 {
        return None;
    }
    let b = count_f64(values.len());
    let mean = values.iter().sum::<f64>() / b;
    let (m2, m4) = values.iter().fold((0.0, 0.0), |(m2, m4), value| {
        let square = (value - mean) * (value - mean);
        (m2 + square, m4 + square * square)
    });
    let (m2, m4) = (m2 / b, m4 / b);
    if !(m2 > 0.0 && m2.is_finite() && m4.is_finite()) {
        return None;
    }
    let kurtosis = m4 / (m2 * m2);
    Some(((kurtosis - 1.0).max(0.0) / (4.0 * b)).sqrt())
}

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the rank is a clamped non-negative floor or ceil below the value count"
)]
fn clamped_rank(position: f64, last: usize, ceil: bool) -> usize {
    let rounded = if ceil { position.ceil() } else { position.floor() };
    if rounded.is_nan() || rounded <= 0.0 {
        return 0;
    }
    (rounded as usize).min(last)
}

/// Standard error of the type-7 quantile at probability `p`, from the order statistics one
/// binomial standard error either side of its rank.
fn quantile_mc_se(sorted: &[f64], p: f64) -> Option<f64> {
    let last = sorted.len().checked_sub(1)?;
    let b = count_f64(sorted.len());
    let rank = (b - 1.0) * p;
    let band = (b * p * (1.0 - p)).sqrt();
    let low = *sorted.get(clamped_rank(rank - band, last, false))?;
    let high = *sorted.get(clamped_rank(rank + band, last, true))?;
    Some((high - low) / 2.0)
}

/// Relative Monte Carlo error of the percentile-interval half-width at `level`.
///
/// The two endpoint errors are combined as independent. `None` under two values, for an
/// invalid level, or when the half-width is zero or non-finite.
#[must_use]
pub fn half_width_relative_mc_error(values: &[f64], level: f64) -> Option<f64> {
    let (lower, upper) = percentile_interval(values, level)?;
    let half_width = (upper - lower) / 2.0;
    if !(half_width > 0.0 && half_width.is_finite()) {
        return None;
    }
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let tail = (1.0 - level) / 2.0;
    let low = quantile_mc_se(&sorted, tail)?;
    let high = quantile_mc_se(&sorted, 1.0 - tail)?;
    let error = (low * low + high * high).sqrt() / 2.0;
    error.is_finite().then_some(error / half_width)
}

enum Replicate {
    Success(f64),
    Declined,
    NonFinite,
    EstimatorError(String),
}

/// Why a wave did not finish.
enum Halt {
    /// A replicate saw the token trip.
    Cancelled,
    /// An estimator error that the caller's procedure treats as fatal.
    Failed(EstimationError),
}

/// Why a driven run ended in an error.
#[derive(Debug)]
pub(crate) enum DriveError {
    /// The rule refused or the run was cancelled under [`CancelPolicy::Refuse`].
    Wave(WaveError),
    /// A replicate's estimator failed and the caller aborts on that.
    Estimator(EstimationError),
}

fn classify(outcome: Result<Option<f64>, EstimationError>) -> Result<Replicate, Halt> {
    match outcome {
        Err(EstimationError::Data(DataError::Cancelled)) => Err(Halt::Cancelled),
        Err(error) => Ok(Replicate::EstimatorError(error.to_string())),
        Ok(None) => Ok(Replicate::Declined),
        Ok(Some(value)) if value.is_finite() => Ok(Replicate::Success(value)),
        Ok(Some(_)) => Ok(Replicate::NonFinite),
    }
}

/// Totals carried across waves.
#[derive(Default)]
struct Ledger {
    successes: Vec<f64>,
    waves: Vec<WaveRecord>,
    attempted: u32,
    failed: u32,
    first_error: Option<String>,
}

impl Ledger {
    fn absorb(&mut self, wave: u32, first_replicate: u32, outcomes: Vec<Replicate>) {
        let mut record = WaveRecord {
            wave,
            first_replicate,
            attempted: u32::try_from(outcomes.len()).unwrap_or(u32::MAX),
            ok: 0,
            estimator_errors: 0,
            non_finite: 0,
            declined: 0,
        };
        for outcome in outcomes {
            match outcome {
                Replicate::Success(value) => {
                    record.ok += 1;
                    self.successes.push(value);
                }
                Replicate::Declined => record.declined += 1,
                Replicate::NonFinite => record.non_finite += 1,
                Replicate::EstimatorError(message) => {
                    record.estimator_errors += 1;
                    self.first_error.get_or_insert(message);
                }
            }
        }
        self.attempted += record.attempted;
        self.failed += record.failed();
        self.waves.push(record);
    }

    fn finish(self, stop: StopReason, level: f64) -> WaveRun {
        let ok = u32::try_from(self.successes.len()).unwrap_or(u32::MAX);
        let se = Some(sample_std(&self.successes)).filter(|value| value.is_finite());
        let interval = percentile_interval(&self.successes, level);
        WaveRun {
            se_rel_mc_error: se_relative_mc_error(&self.successes),
            half_width_rel_mc_error: half_width_relative_mc_error(&self.successes, level),
            successes: self.successes,
            waves: self.waves,
            attempted: self.attempted,
            ok,
            failed: self.failed,
            stop,
            first_error: self.first_error,
            se,
            interval,
        }
    }
}

fn run_wave(
    ctx: &ExecutionContext,
    stream_base: u64,
    first_replicate: u32,
    count: u32,
    estimate: &(impl Fn(u64, &mut CausalRng) -> Result<Option<f64>, EstimationError> + Sync),
) -> Result<Vec<Replicate>, Halt> {
    let len = usize::try_from(count).map_err(|_| Halt::Cancelled)?;
    ctx.map_indexed(len, |local, inner| {
        if inner.cancellation.is_cancelled() {
            return Err(Halt::Cancelled);
        }
        let local = u32::try_from(local).map_err(|_| Halt::Cancelled)?;
        let id = u64::from(first_replicate) + u64::from(local);
        let mut rng = wave_replicate_rng(inner, stream_base, id);
        classify(estimate(id, &mut rng))
    })
}

fn cancelled(ledger: Ledger, rule: &WaveStoppingRule) -> Result<WaveRun, WaveError> {
    match rule.on_cancel {
        CancelPolicy::Refuse => Err(WaveError::Cancelled {
            completed_waves: u32::try_from(ledger.waves.len()).unwrap_or(u32::MAX),
            attempted: ledger.attempted,
        }),
        CancelPolicy::ReturnPartial => Ok(ledger.finish(StopReason::Cancelled, rule.level)),
    }
}

/// Run waves of keyed replicates until the rule stops, the cap is reached, or the token trips.
///
/// `estimate(id, rng)` evaluates replicate `id` with its own keyed stream. It returns
/// `Ok(Some(value))` for a success, `Ok(None)` to decline, or an error. An error other than
/// [`DataError::Cancelled`] and a non-finite value are failed replicates, counted and never
/// dropped from the denominator. [`DataError::Cancelled`] is a cancellation.
///
/// # Errors
///
/// [`WaveError::InvalidRule`] for an unusable rule, [`WaveError::FailureFraction`] when the
/// cumulative failed fraction exceeds the limit after a wave, and [`WaveError::Cancelled`] when
/// the token trips under [`CancelPolicy::Refuse`].
pub fn run_adaptive_waves(
    rule: &WaveStoppingRule,
    ctx: &ExecutionContext,
    stream_base: u64,
    estimate: impl Fn(u64, &mut CausalRng) -> Result<Option<f64>, EstimationError> + Sync,
) -> Result<WaveRun, WaveError> {
    drive_waves(rule, ctx, |first, count| run_wave(ctx, stream_base, first, count, &estimate))
        .map_err(|error| match error {
            DriveError::Wave(error) => error,
            DriveError::Estimator(_) => {
                WaveError::InvalidRule("an estimator error escaped the failure accounting")
            }
        })
}

/// Run waves on the caller's thread with one reused scratch, aborting on an estimator error.
///
/// This is the production form for a bootstrap standard error whose estimator keeps a per-run
/// scratch (a gather buffer, a workspace) and whose caller treats an estimator error as fatal
/// rather than as a failed replicate. `estimate(scratch, id, rng)` evaluates replicate `id`:
/// `Ok(Some(value))` is a success, `Ok(None)` a declined replicate, a non-finite value a failed
/// one, and any error aborts the whole run and is returned as [`DriveError::Estimator`]. The
/// stopping rule, failure accounting and cancellation are those of [`run_adaptive_waves`]: the
/// token is observed before each wave and before each replicate, and a replicate that trips it
/// is still part of its completed wave.
///
/// # Errors
///
/// The errors of [`run_adaptive_waves`], or the first estimator error.
pub(crate) fn run_waves_serial<S>(
    rule: &WaveStoppingRule,
    ctx: &ExecutionContext,
    stream_base: u64,
    scratch: &mut S,
    mut estimate: impl FnMut(&mut S, u64, &mut CausalRng) -> Result<Option<f64>, EstimationError>,
) -> Result<WaveRun, DriveError> {
    drive_waves(rule, ctx, |first, count| {
        let mut outcomes = Vec::with_capacity(usize::try_from(count).unwrap_or(0));
        for local in 0..count {
            if ctx.cancellation.is_cancelled() {
                return Err(Halt::Cancelled);
            }
            let id = u64::from(first) + u64::from(local);
            let mut rng = wave_replicate_rng(ctx, stream_base, id);
            outcomes.push(match estimate(scratch, id, &mut rng) {
                Err(error) => return Err(Halt::Failed(error)),
                Ok(None) => Replicate::Declined,
                Ok(Some(value)) if value.is_finite() => Replicate::Success(value),
                Ok(Some(_)) => Replicate::NonFinite,
            });
        }
        Ok(outcomes)
    })
}

fn drive_waves(
    rule: &WaveStoppingRule,
    ctx: &ExecutionContext,
    mut run_one: impl FnMut(u32, u32) -> Result<Vec<Replicate>, Halt>,
) -> Result<WaveRun, DriveError> {
    rule.validate().map_err(DriveError::Wave)?;
    let mut ledger = Ledger::default();
    let mut next = 0u32;
    let mut wave = 0u32;
    while next < rule.max_replicates {
        if ctx.cancellation.is_cancelled() {
            return cancelled(ledger, rule).map_err(DriveError::Wave);
        }
        let count = rule.wave_size.min(rule.max_replicates - next);
        let outcomes = match run_one(next, count) {
            Ok(outcomes) => outcomes,
            Err(Halt::Cancelled) => return cancelled(ledger, rule).map_err(DriveError::Wave),
            Err(Halt::Failed(error)) => return Err(DriveError::Estimator(error)),
        };
        ledger.absorb(wave, next, outcomes);
        next += count;
        if let Some(progress) = &ctx.progress {
            progress.report(
                f64::from(next) / f64::from(rule.max_replicates),
                "adaptive resample waves",
            );
        }
        if f64::from(ledger.failed) > rule.max_failure_fraction * f64::from(ledger.attempted) {
            return Err(DriveError::Wave(WaveError::FailureFraction {
                wave,
                attempted: ledger.attempted,
                failed: ledger.failed,
                max_fraction: rule.max_failure_fraction,
            }));
        }
        wave += 1;
        if next < rule.max_replicates && rule.tolerances_met(&ledger.successes) {
            return Ok(ledger.finish(StopReason::Converged, rule.level));
        }
    }
    Ok(ledger.finish(StopReason::MaxReplicates, rule.level))
}

#[cfg(test)]
mod tests {
    use std::collections::HashSet;

    use antecedent_core::{CausalRng, ExecutionContext, StreamDomain, derive_stream};
    use antecedent_data::DataError;
    use antecedent_kernels::{sample_std, unbiased_index};

    use super::{
        CancelPolicy, StopReason, WaveError, WaveStoppingRule, half_width_relative_mc_error,
        run_adaptive_waves, se_relative_mc_error, wave_replicate_rng,
    };
    use crate::error::EstimationError;

    fn rule(wave_size: u32, max_replicates: u32) -> WaveStoppingRule {
        WaveStoppingRule {
            wave_size,
            min_replicates: 0,
            max_replicates,
            se_rel_tol: None,
            half_width_rel_tol: None,
            success_floor: None,
            level: 0.95,
            max_failure_fraction: 0.5,
            on_cancel: CancelPolicy::ReturnPartial,
        }
    }

    /// Mean of `n` rows resampled with replacement from `data`.
    fn bootstrap_mean(data: &[f64], rng: &mut CausalRng) -> f64 {
        let n = data.len();
        let sum: f64 = (0..n).map(|_| data[unbiased_index(rng, n)]).sum();
        sum / n as f64
    }

    fn normal_sample(ctx: &ExecutionContext, n: usize, scale: f64) -> Vec<f64> {
        let mut rng = ctx.rng.stream(0xDA7A);
        (0..n)
            .map(|_| {
                let (u, v) = (1.0 - rng.next_f64(), rng.next_f64());
                scale * (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
            })
            .collect()
    }

    fn bits_eq(left: &[f64], right: &[f64]) {
        assert_eq!(left.len(), right.len());
        for (left, right) in left.iter().zip(right) {
            assert_eq!(left.to_bits(), right.to_bits());
        }
    }

    #[test]
    fn replicate_streams_do_not_alias_across_small_bases() {
        let ctx = ExecutionContext::for_tests(5);
        let first = |base: u64, id: u64| wave_replicate_rng(&ctx, base, id).next_u64();
        assert_ne!(first(0, 1), first(1, 0));
        let mut seen = HashSet::new();
        for base in 0..16u64 {
            for id in 0..16u64 {
                assert!(seen.insert(first(base, id)));
            }
        }
        // The XOR form this replaces would collapse these onto one stream.
        let xor = |base: u64, id: u64| ctx.rng.stream_for(StreamDomain::Resample, base ^ id);
        assert_eq!(xor(0, 1).next_u64(), xor(1, 0).next_u64());
    }

    #[test]
    fn budget_branch_reproduces_the_fixed_b_result_bit_for_bit() {
        let ctx = ExecutionContext::for_tests(11);
        let data = normal_sample(&ctx, 30, 1.0);
        let mut unreachable = rule(16, 100);
        unreachable.se_rel_tol = Some(1e-9);
        unreachable.half_width_rel_tol = Some(1e-9);
        let run = run_adaptive_waves(&unreachable, &ctx, 7, |_, rng| {
            Ok(Some(bootstrap_mean(&data, rng)))
        })
        .unwrap();
        assert_eq!(run.stop, StopReason::MaxReplicates);
        // Fixed-B reference: one straight loop over the keyed streams.
        let fixed: Vec<f64> = (0..100u64)
            .map(|i| {
                let mut rng = ctx.rng.stream_for(StreamDomain::Resample, derive_stream(7, i));
                bootstrap_mean(&data, &mut rng)
            })
            .collect();
        bits_eq(&run.successes, &fixed);
        assert_eq!((run.attempted, run.ok, run.failed), (100, 100, 0));
        assert_eq!(run.waves.len(), 7);
        assert_eq!(run.waves.last().map(|w| w.attempted), Some(4));
    }

    #[test]
    fn replicate_values_are_invariant_to_wave_size_threads_and_retries() {
        let ctx = ExecutionContext::for_tests(21);
        let data = normal_sample(&ctx, 25, 1.0);
        let run = |wave: u32, ctx: &ExecutionContext| {
            run_adaptive_waves(&rule(wave, 120), ctx, 3, |_, rng| {
                Ok(Some(bootstrap_mean(&data, rng)))
            })
            .unwrap()
        };
        let reference = run(120, &ctx);
        for wave in [1, 7, 25, 64] {
            bits_eq(&run(wave, &ctx).successes, &reference.successes);
        }
        let threaded = ExecutionContext::production(21, 4);
        bits_eq(&run(7, &threaded).successes, &reference.successes);
        bits_eq(&run(7, &ctx).successes, &reference.successes);
    }

    #[test]
    fn stopped_run_is_a_prefix_of_the_unstopped_run() {
        let ctx = ExecutionContext::for_tests(31);
        let data = normal_sample(&ctx, 30, 1.0);
        let estimate = |_: u64, rng: &mut CausalRng| Ok(Some(bootstrap_mean(&data, rng)));
        let mut adaptive = rule(50, 2000);
        adaptive.min_replicates = 50;
        adaptive.se_rel_tol = Some(0.08);
        let stopped = run_adaptive_waves(&adaptive, &ctx, 1, estimate).unwrap();
        assert_eq!(stopped.stop, StopReason::Converged);
        assert!(stopped.attempted < 2000);
        let full = run_adaptive_waves(&rule(50, 2000), &ctx, 1, estimate).unwrap();
        bits_eq(&stopped.successes, &full.successes[..stopped.successes.len()]);
        assert!(stopped.se_rel_mc_error.unwrap() <= 0.08);
        // Replay: the same rule stops at the same place with the same bits.
        let again = run_adaptive_waves(&adaptive, &ctx, 1, estimate).unwrap();
        assert_eq!(again, stopped);
    }

    #[test]
    fn failed_replicates_are_counted_per_wave_by_kind() {
        let ctx = ExecutionContext::for_tests(41);
        let run = run_adaptive_waves(&rule(5, 20), &ctx, 0, |id, rng| match id {
            3 => Err(EstimationError::data_msg("injected failure")),
            7 => Ok(None),
            12 => Ok(Some(f64::NAN)),
            13 => Ok(Some(f64::INFINITY)),
            _ => Ok(Some(rng.next_f64())),
        })
        .unwrap();
        let counts: Vec<_> = run
            .waves
            .iter()
            .map(|w| (w.attempted, w.ok, w.estimator_errors, w.non_finite, w.declined))
            .collect();
        assert_eq!(counts, [(5, 4, 1, 0, 0), (5, 4, 0, 0, 1), (5, 3, 0, 2, 0), (5, 5, 0, 0, 0)]);
        assert_eq!((run.attempted, run.ok, run.failed), (20, 16, 4));
        assert_eq!(run.successes.len(), 16);
        assert!(run.successes.iter().all(|v| v.is_finite()));
        assert!(run.first_error.as_deref().unwrap().contains("injected failure"));
        assert_eq!(run.waves[2].first_replicate, 10);
        assert_eq!(run.waves.iter().map(super::WaveRecord::failed).sum::<u32>(), run.failed);
    }

    #[test]
    fn failure_fraction_above_the_limit_refuses_and_at_the_limit_runs() {
        let ctx = ExecutionContext::for_tests(43);
        let even_fails = |id: u64, rng: &mut CausalRng| {
            if id % 2 == 0 { Ok(None) } else { Ok(Some(rng.next_f64())) }
        };
        let mut strict = rule(10, 40);
        strict.max_failure_fraction = 0.25;
        match run_adaptive_waves(&strict, &ctx, 0, even_fails) {
            Err(WaveError::FailureFraction { wave: 0, attempted: 10, failed: 5, .. }) => {}
            other => panic!("expected a failure-fraction refusal, got {other:?}"),
        }
        let mut at_limit = rule(10, 40);
        at_limit.max_failure_fraction = 0.5;
        let run = run_adaptive_waves(&at_limit, &ctx, 0, even_fails).unwrap();
        assert_eq!((run.attempted, run.ok, run.failed), (40, 20, 20));
    }

    #[test]
    fn cancellation_between_waves_returns_whole_waves_or_refuses() {
        let ctx = ExecutionContext::for_tests(47);
        let token = ctx.cancellation.clone();
        // Cancel on the last replicate of the second wave: that wave completes, the next never starts.
        let estimate = |id: u64, rng: &mut CausalRng| {
            if id == 19 {
                token.cancel();
            }
            Ok(Some(rng.next_f64()))
        };
        let run = run_adaptive_waves(&rule(10, 100), &ctx, 0, estimate).unwrap();
        assert_eq!(run.stop, StopReason::Cancelled);
        assert!(run.is_partial());
        assert_eq!((run.attempted, run.ok, run.failed, run.waves.len()), (20, 20, 0, 2));

        let ctx = ExecutionContext::for_tests(47);
        let mut refuse = rule(10, 100);
        refuse.on_cancel = CancelPolicy::Refuse;
        let token = ctx.cancellation.clone();
        let result = run_adaptive_waves(&refuse, &ctx, 0, |id, rng| {
            if id == 19 {
                token.cancel();
            }
            Ok(Some(rng.next_f64()))
        });
        assert!(matches!(result, Err(WaveError::Cancelled { completed_waves: 2, attempted: 20 })));
    }

    #[test]
    fn cancellation_inside_a_wave_discards_that_wave() {
        let ctx = ExecutionContext::for_tests(53);
        let token = ctx.cancellation.clone();
        let run = run_adaptive_waves(&rule(10, 100), &ctx, 0, |id, rng| {
            if id == 25 {
                token.cancel();
            }
            Ok(Some(rng.next_f64()))
        })
        .unwrap();
        assert_eq!(run.stop, StopReason::Cancelled);
        assert_eq!((run.attempted, run.waves.len(), run.successes.len()), (20, 2, 20));
        // The kept replicates equal the same replicates of an uncancelled run.
        let clean =
            run_adaptive_waves(&rule(10, 100), &ExecutionContext::for_tests(53), 0, |_, rng| {
                Ok(Some(rng.next_f64()))
            })
            .unwrap();
        bits_eq(&run.successes, &clean.successes[..20]);

        // A cancellation error raised by the estimator is a cancellation, not a failure.
        let ctx = ExecutionContext::for_tests(53);
        let run = run_adaptive_waves(&rule(10, 100), &ctx, 0, |id, rng| {
            if id == 12 {
                Err(EstimationError::Data(DataError::Cancelled))
            } else {
                Ok(Some(rng.next_f64()))
            }
        })
        .unwrap();
        assert_eq!((run.stop, run.attempted, run.failed), (StopReason::Cancelled, 10, 0));
    }

    #[test]
    fn a_token_cancelled_before_the_start_runs_no_wave() {
        let ctx = ExecutionContext::for_tests(59);
        ctx.cancellation.cancel();
        let run =
            run_adaptive_waves(&rule(10, 100), &ctx, 0, |_, rng| Ok(Some(rng.next_f64()))).unwrap();
        assert_eq!((run.stop, run.attempted, run.waves.len()), (StopReason::Cancelled, 0, 0));
        assert!(run.se.is_none() && run.interval.is_none());
    }

    #[test]
    fn invalid_rules_are_refused() {
        let ctx = ExecutionContext::for_tests(61);
        let est = |_: u64, rng: &mut CausalRng| Ok(Some(rng.next_f64()));
        let mut bad = rule(0, 10);
        assert!(matches!(run_adaptive_waves(&bad, &ctx, 0, est), Err(WaveError::InvalidRule(_))));
        bad = rule(5, 10);
        bad.min_replicates = 11;
        assert!(bad.validate().is_err());
        bad = rule(5, 10);
        bad.level = 1.0;
        assert!(bad.validate().is_err());
        bad = rule(5, 10);
        bad.se_rel_tol = Some(0.0);
        assert!(bad.validate().is_err());
        bad = rule(5, 10);
        bad.max_failure_fraction = 1.5;
        assert!(bad.validate().is_err());
    }

    #[test]
    fn monte_carlo_error_estimates_match_hand_values() {
        // Deviations -1, -1, -1, 3: m2 = 3, m4 = 21, kurtosis 7/3, error sqrt((4/3) / 16).
        let kurt = se_relative_mc_error(&[0.0, 0.0, 0.0, 4.0]).unwrap();
        assert!((kurt - (1.0f64 / 12.0).sqrt()).abs() < 1e-12);
        assert_eq!(se_relative_mc_error(&[2.0; 10]), None);
        assert_eq!(se_relative_mc_error(&[1.0, 2.0, 3.0]), None);
        // Values 0..999: endpoint ranks 24.975 and 974.025 with a band of about 4.94 ranks give
        // order-statistic standard errors of 5 at both ends; the half-width is 474.525.
        let grid: Vec<f64> = (0..1000).map(f64::from).collect();
        let error = half_width_relative_mc_error(&grid, 0.95).unwrap();
        let expected = (50.0f64).sqrt() / 2.0 / 474.525;
        assert!((error - expected).abs() < 1e-9, "{error} vs {expected}");
        assert_eq!(half_width_relative_mc_error(&[3.0; 50], 0.95), None);
    }

    #[test]
    fn reported_se_matches_the_analytic_bootstrap_se_within_tolerance() {
        // Mean of normal data: the bootstrap SE of the mean tends to sqrt(m2 / n) with m2 the
        // plug-in variance, so the reported SE is checked against that closed form. The rule
        // targets 5% Monte Carlo error; each run must land within 20% (four times the target).
        let tolerance = 0.05;
        let mut deviations = Vec::new();
        for seed in 0..20u64 {
            let ctx = ExecutionContext::for_tests(1000 + seed);
            let data = normal_sample(&ctx, 40, 2.0);
            let n = data.len() as f64;
            let mean = data.iter().sum::<f64>() / n;
            let plug_in = data.iter().map(|x| (x - mean) * (x - mean)).sum::<f64>() / n;
            let analytic = (plug_in / n).sqrt();
            let mut adaptive = rule(50, 4000);
            adaptive.min_replicates = 100;
            adaptive.se_rel_tol = Some(tolerance);
            let run = run_adaptive_waves(&adaptive, &ctx, seed, |_, rng| {
                Ok(Some(bootstrap_mean(&data, rng)))
            })
            .unwrap();
            assert_eq!(run.stop, StopReason::Converged);
            assert!(run.attempted >= 100 && run.attempted < 1000, "{}", run.attempted);
            let deviation = run.se.unwrap() / analytic - 1.0;
            assert!(deviation.abs() <= 4.0 * tolerance, "seed {seed}: {deviation}");
            deviations.push(deviation);
        }
        let mean_deviation = deviations.iter().sum::<f64>() / deviations.len() as f64;
        assert!(mean_deviation.abs() <= tolerance, "mean deviation {mean_deviation}");
    }

    /// Slope of a simple regression of `y` on `x` over the listed rows.
    fn slope(x: &[f64], y: &[f64], rows: &[usize]) -> f64 {
        let n = rows.len() as f64;
        let mx = rows.iter().map(|&i| x[i]).sum::<f64>() / n;
        let my = rows.iter().map(|&i| y[i]).sum::<f64>() / n;
        let sxy: f64 = rows.iter().map(|&i| (x[i] - mx) * (y[i] - my)).sum();
        let sxx: f64 = rows.iter().map(|&i| (x[i] - mx) * (x[i] - mx)).sum();
        sxy / sxx
    }

    #[test]
    fn serial_runner_equals_the_parallel_runner_and_reuses_one_scratch() {
        use super::run_waves_serial;
        let ctx = ExecutionContext::for_tests(71);
        let data = normal_sample(&ctx, 25, 1.0);
        let parallel = run_adaptive_waves(&rule(8, 60), &ctx, 5, |_, rng| {
            Ok(Some(bootstrap_mean(&data, rng)))
        })
        .unwrap();
        let mut scratch: Vec<u64> = Vec::new();
        let serial = run_waves_serial(&rule(8, 60), &ctx, 5, &mut scratch, |seen, id, rng| {
            seen.push(id);
            Ok(Some(bootstrap_mean(&data, rng)))
        })
        .unwrap();
        bits_eq(&serial.successes, &parallel.successes);
        assert_eq!(serial.waves, parallel.waves);
        assert_eq!(scratch, (0..60u64).collect::<Vec<_>>(), "one scratch saw every id once");
    }

    #[test]
    fn success_floor_stops_after_exactly_the_floor_and_keeps_failures_in_the_count() {
        use super::run_waves_serial;
        let ctx = ExecutionContext::for_tests(73);
        let floor_rule = WaveStoppingRule::success_floor_rule(100, 17);
        let mut unit = ();
        let stopped = run_waves_serial(&floor_rule, &ctx, 2, &mut unit, |(), id, rng| {
            if id == 2 || id == 5 { Ok(None) } else { Ok(Some(rng.next_f64())) }
        })
        .unwrap();
        assert_eq!(stopped.stop, StopReason::Converged);
        assert_eq!((stopped.ok, stopped.failed, stopped.attempted), (17, 2, 19));
        let full = run_waves_serial(&rule(1, 100), &ctx, 2, &mut unit, |(), id, rng| {
            if id == 2 || id == 5 { Ok(None) } else { Ok(Some(rng.next_f64())) }
        })
        .unwrap();
        bits_eq(&stopped.successes, &full.successes[..17]);
        // A floor above the cap never stops: the fixed-B branch.
        let capped = WaveStoppingRule::success_floor_rule(30, 40);
        let run =
            run_waves_serial(&capped, &ctx, 2, &mut unit, |(), _, rng| Ok(Some(rng.next_f64())))
                .unwrap();
        assert_eq!((run.stop, run.attempted), (StopReason::MaxReplicates, 30));
        let mut zero = WaveStoppingRule::success_floor_rule(30, 1);
        zero.success_floor = Some(0);
        assert!(zero.validate().is_err());
    }

    #[test]
    fn serial_runner_aborts_on_an_estimator_error_and_returns_completed_replicates_on_cancel() {
        use super::{DriveError, run_waves_serial};
        let ctx = ExecutionContext::for_tests(79);
        let mut unit = ();
        let aborted = run_waves_serial(&rule(1, 50), &ctx, 0, &mut unit, |(), id, rng| {
            if id == 4 {
                Err(EstimationError::data_msg("malformed replicate"))
            } else {
                Ok(Some(rng.next_f64()))
            }
        });
        match aborted {
            Err(DriveError::Estimator(error)) => assert!(error.to_string().contains("malformed")),
            other => panic!("expected an estimator abort, got {other:?}"),
        }
        let token = ctx.cancellation.clone();
        let partial = run_waves_serial(&rule(1, 50), &ctx, 0, &mut unit, |(), id, rng| {
            if id == 6 {
                token.cancel();
            }
            Ok(Some(rng.next_f64()))
        })
        .unwrap();
        assert_eq!((partial.stop, partial.attempted, partial.ok), (StopReason::Cancelled, 7, 7));
    }

    #[test]
    fn enabled_budget_bootstraps_run_through_the_wave_procedure() {
        use antecedent_core::AdaptiveBootstrapBudget;

        use crate::bootstrap_wave::fill_worker_replicate;
        use crate::util::bootstrap_se_with_scratch;

        let plain = ExecutionContext::for_tests(7);
        let mut ctx = ExecutionContext::for_tests(7);
        let data = normal_sample(&plain, 30, 1.0);
        let n = data.len();
        let mean_of =
            |rows: &[usize]| rows.iter().map(|&i| data[i]).sum::<f64>() / rows.len() as f64;
        let reference = |count: u32| -> Vec<f64> {
            (0..u64::from(count))
                .map(|id| {
                    let mut rows = Vec::new();
                    fill_worker_replicate(n, id, &plain, 3, &mut rows).unwrap();
                    mean_of(&rows)
                })
                .collect()
        };
        // ε = 0.2 gives the floor ceil(1 + 1 / (2 · 0.04)) = 14 successes.
        let budget =
            AdaptiveBootstrapBudget { enabled: true, min_replicates: 2, se_rel_epsilon: 0.2 };
        assert_eq!(budget.required_replicates(), 14);
        let fixed_se =
            bootstrap_se_with_scratch(60, &ctx, 3, n, || (), |(), rows| Ok(Some(mean_of(rows))))
                .unwrap();
        assert!(!fixed_se.early_stopped);
        assert_eq!(fixed_se.replicates_ok, 60);
        assert_eq!(fixed_se.se.unwrap().to_bits(), sample_std(&reference(60)).to_bits());

        ctx.adaptive_bootstrap = budget;
        let calls = std::sync::atomic::AtomicU32::new(0);
        let stopped = bootstrap_se_with_scratch(
            60,
            &ctx,
            3,
            n,
            || (),
            |(), rows| {
                let call = calls.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                // The first two replicates decline; they stay in the attempt count.
                Ok(if call < 2 { None } else { Some(mean_of(rows)) })
            },
        )
        .unwrap();
        assert!(stopped.early_stopped && !stopped.cancelled);
        assert_eq!((stopped.replicates_ok, stopped.replicates_failed), (14, 2));
        assert_eq!(calls.load(std::sync::atomic::Ordering::Relaxed), 16);
        // The successes are replicates 2..16 of the fixed run, so the SE is their sample SD.
        assert_eq!(stopped.se.unwrap().to_bits(), sample_std(&reference(16)[2..]).to_bits());

        // An estimator error aborts the bootstrap, and a cancelled token is a partial result.
        let aborted = bootstrap_se_with_scratch(
            60,
            &ctx,
            3,
            n,
            || (),
            |(), _| Err(EstimationError::data_msg("estimator refused")),
        );
        assert!(aborted.unwrap_err().to_string().contains("estimator refused"));
        let token = ctx.cancellation.clone();
        let cancelled = bootstrap_se_with_scratch(
            60,
            &ctx,
            3,
            n,
            || (),
            |(), rows| {
                token.cancel();
                Ok(Some(mean_of(rows)))
            },
        )
        .unwrap();
        assert!(cancelled.cancelled && !cancelled.early_stopped);
        assert_eq!(cancelled.replicates_ok, 1);
    }

    #[test]
    fn regression_slope_procedure_fits_resamples_stops_and_reports_the_interval() {
        use crate::bootstrap_wave::fill_worker_replicate;
        use crate::statistical_transport::percentile_interval;

        let ctx = ExecutionContext::for_tests(83);
        let n = 200;
        let x = normal_sample(&ctx, n, 1.0);
        let noise = {
            let mut rng = ctx.rng.stream(0xBEEF);
            (0..n)
                .map(|_| {
                    let (u, v) = (1.0 - rng.next_f64(), rng.next_f64());
                    (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
                })
                .collect::<Vec<f64>>()
        };
        let y: Vec<f64> = x.iter().zip(&noise).map(|(x, e)| 1.0 + 2.0 * x + e).collect();
        // The full-sample fit is made once, outside the waves.
        let everything: Vec<usize> = (0..n).collect();
        let fitted = slope(&x, &y, &everything);
        let mx = x.iter().sum::<f64>() / n as f64;
        let sxx: f64 = x.iter().map(|v| (v - mx) * (v - mx)).sum();
        let residual_variance = {
            let my = y.iter().sum::<f64>() / n as f64;
            let intercept = my - fitted * mx;
            y.iter().zip(&x).map(|(y, x)| (y - intercept - fitted * x).powi(2)).sum::<f64>()
                / (n as f64 - 2.0)
        };
        let analytic_se = (residual_variance / sxx).sqrt();

        let evaluations = std::sync::atomic::AtomicU32::new(0);
        let mut adaptive = rule(50, 4000);
        adaptive.min_replicates = 100;
        adaptive.se_rel_tol = Some(0.05);
        adaptive.half_width_rel_tol = Some(0.25);
        let run = run_adaptive_waves(&adaptive, &ctx, 9, |id, _| {
            evaluations.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
            let mut rows = Vec::new();
            fill_worker_replicate(n, id, &ctx, 9, &mut rows).unwrap();
            Ok(Some(slope(&x, &y, &rows)))
        })
        .unwrap();
        assert_eq!(run.stop, StopReason::Converged);
        assert_eq!(run.failed, 0);
        assert!(run.attempted < 4000, "stopped at {}", run.attempted);
        assert_eq!(
            evaluations.load(std::sync::atomic::Ordering::Relaxed),
            run.attempted,
            "each replicate is one refit; the full-sample fit is not repeated by the waves"
        );
        let (lower, upper) = run.interval.expect("interval");
        assert_eq!((lower, upper), percentile_interval(&run.successes, 0.95).unwrap());
        assert!(lower < fitted && fitted < upper, "{lower} {fitted} {upper}");
        let se = run.se.unwrap();
        assert!((se / analytic_se - 1.0).abs() < 0.25, "bootstrap {se} vs analytic {analytic_se}");
        let half_width = (upper - lower) / 2.0;
        assert!((half_width / (1.96 * analytic_se) - 1.0).abs() < 0.3, "{half_width}");
        assert!(run.se_rel_mc_error.unwrap() <= 0.05);
        assert!(run.half_width_rel_mc_error.unwrap() <= 0.25);
        // The interval at the stop is the percentile interval of the replicates drawn so far,
        // and those are the leading replicates of a run that never stops.
        let never = run_adaptive_waves(&rule(50, run.attempted), &ctx, 9, |id, _| {
            let mut rows = Vec::new();
            fill_worker_replicate(n, id, &ctx, 9, &mut rows).unwrap();
            Ok(Some(slope(&x, &y, &rows)))
        })
        .unwrap();
        bits_eq(&run.successes, &never.successes);
        assert_eq!(run.interval, never.interval);
    }

    #[test]
    fn production_interval_bootstrap_stops_on_tolerance_and_returns_the_interval() {
        use crate::bootstrap_wave::fill_worker_replicate;
        use crate::util::bootstrap_interval_with_scratch;

        let ctx = ExecutionContext::for_tests(91);
        let data = normal_sample(&ctx, 60, 1.0);
        let n = data.len();
        let mean_of =
            |rows: &[usize]| rows.iter().map(|&i| data[i]).sum::<f64>() / rows.len() as f64;
        let mut tolerant = rule(50, 4000);
        tolerant.min_replicates = 100;
        tolerant.se_rel_tol = Some(0.05);
        tolerant.half_width_rel_tol = Some(0.25);
        let run = bootstrap_interval_with_scratch(
            &tolerant,
            &ctx,
            4,
            n,
            || (),
            |(), rows| Ok(Some(mean_of(rows))),
        )
        .unwrap();
        assert!(run.se.early_stopped && !run.se.cancelled);
        let attempted = run.se.replicates_ok + run.se.replicates_failed;
        assert!(attempted >= 100 && attempted < 4000, "{attempted}");
        assert_eq!(run.waves.len() as u32 * 50, attempted);
        // The successes are the leading replicates of the fixed-count bootstrap on this seed.
        let reference: Vec<f64> = (0..u64::from(attempted))
            .map(|id| {
                let mut rows = Vec::new();
                fill_worker_replicate(n, id, &ctx, 4, &mut rows).unwrap();
                mean_of(&rows)
            })
            .collect();
        assert_eq!(run.se.se.unwrap().to_bits(), sample_std(&reference).to_bits());
        let expected = crate::statistical_transport::percentile_interval(&reference, 0.95);
        assert_eq!(run.interval, expected);
        let (lower, upper) = run.interval.unwrap();
        let analytic =
            (data.iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt() / (n as f64).sqrt();
        assert!(lower < upper && ((upper - lower) / 2.0 / (1.96 * analytic) - 1.0).abs() < 0.4);
        assert!(run.se_rel_mc_error.unwrap() <= 0.05);
        assert!(run.half_width_rel_mc_error.unwrap() <= 0.25);
        // Estimator errors abort; an unreachable tolerance runs the whole request.
        let aborted = bootstrap_interval_with_scratch(
            &tolerant,
            &ctx,
            4,
            n,
            || (),
            |(), _| Err(EstimationError::data_msg("estimator refused")),
        );
        assert!(aborted.is_err());
        let mut never = rule(25, 100);
        never.se_rel_tol = Some(1e-9);
        let full = bootstrap_interval_with_scratch(
            &never,
            &ctx,
            4,
            n,
            || (),
            |(), rows| Ok(Some(mean_of(rows))),
        )
        .unwrap();
        assert!(!full.se.early_stopped);
        assert_eq!(full.se.replicates_ok, 100);
    }
}
