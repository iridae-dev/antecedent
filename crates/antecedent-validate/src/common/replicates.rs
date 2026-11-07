//! Refuter replicate admission, execution and report summaries.

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_kernels::erfc;

use super::{
    EffectRefit, PanelSliceTemplate, RefutationProblem, RefutationReport, SharedProblem,
    ValidationError,
};

/// Two-sided p-value of observing `hypothesized` as a *typical single replicate*
/// under a normal fit to `samples` (`z = (hypothesized − mean) / sd`, not `/ (sd/√R)`).
///
/// This is not a test of whether the replicate *mean* is `hypothesized`. Adding `√R`
/// would make any tiny systematic bias refute at large R, which is the wrong operating
/// characteristic for a placebo / dummy-outcome refuter. Degenerate spread compares
/// means directly. When both the observed spread and displacement from the null are
/// within a small relative floating-point resolution envelope, the normal fit cannot
/// distinguish them and returns the non-informative value 1. This is a numerical
/// convention, not a calibrated test or a certified estimator error bound.
///
/// # Errors
///
/// Fewer than two samples, or any non-finite sample or `hypothesized`. A replicate that
/// is `NaN`/`±∞` means the refit itself broke; it must be neither a pass (`p = 1`) nor a
/// refutation (`p = 0`), so the caller sees an operational failure instead of a verdict.
#[allow(clippy::float_cmp, reason = "the check preserves exact boundary and category semantics")]
#[allow(
    clippy::items_after_statements,
    reason = "the local helper stays with the validation calculation"
)]
pub(crate) fn replicate_p_value(
    samples: &[f64],
    hypothesized: f64,
) -> Result<f64, ValidationError> {
    if samples.len() < 2 {
        return Err(ValidationError::NotApplicable {
            message: "replicate p-value requires at least two replicates",
        });
    }
    if !hypothesized.is_finite() || samples.iter().any(|x| !x.is_finite()) {
        return Err(ValidationError::estimation_msg(
            "non-finite replicate estimate: the refit did not produce a usable effect",
        ));
    }
    let anchor = samples[0];
    if samples.iter().all(|value| *value == anchor) {
        return Ok(if hypothesized == anchor { 1.0 } else { 0.0 });
    }
    // Center before scaling so the spread is independent of a common location.
    // If subtraction spans both finite extremes, halve both operands first.
    let half = samples.iter().any(|&value| !(value - anchor).is_finite());
    let centered = |value: f64| {
        if half { value * 0.5 - anchor * 0.5 } else { value - anchor }
    };
    let scale = samples.iter().map(|&value| centered(value).abs()).fold(0.0_f64, f64::max);
    // Independent floating-point refits of an algebraically identical effect may
    // differ by tens of ulps. Fitting a normal law to only those rounding bits
    // falsely turns the refit's numerical jitter into near-certain rejection.
    // Compare relative to the actual values, with no absolute floor: a genuinely
    // different effect near zero or under a change of units remains detectable.
    // This fixed guard is a numerical-resolution convention, not a forward-error
    // certificate for an arbitrary estimator.
    let magnitude = samples.iter().map(|value| value.abs()).fold(hypothesized.abs(), f64::max);
    let delta = centered(hypothesized);
    const RELATIVE_ROUNDOFF: f64 = 64.0 * f64::EPSILON;
    if !half
        && magnitude > 0.0
        && delta.is_finite()
        && scale / magnitude <= RELATIVE_ROUNDOFF
        && delta.abs() / magnitude <= RELATIVE_ROUNDOFF
    {
        return Ok(1.0);
    }
    let normalized = |value: f64| centered(value) / scale;
    let mean = antecedent_kernels::numeric::CompensatedSum::of(
        samples.iter().map(|&value| normalized(value)),
    ) / samples.len() as f64;
    let variance = antecedent_kernels::numeric::CompensatedSum::of(
        samples.iter().map(|&value| (normalized(value) - mean).powi(2)),
    ) / (samples.len() - 1) as f64;
    let sd = variance.sqrt();
    let hypothesis = if delta.is_finite() {
        delta / scale
    } else {
        ((hypothesized * 0.5 - anchor * 0.5) / scale) * 2.0
    };
    let z = (hypothesis - mean) / sd;
    // Finite inputs can be separated by more standard deviations than f64 can
    // represent. Their two-sided normal tail is zero.
    if z.is_infinite() {
        return Ok(0.0);
    }
    if !z.is_finite() {
        return Err(ValidationError::estimation_msg(
            "non-finite replicate z-score: the refit did not produce a usable effect",
        ));
    }
    // `erfc` is guaranteed in [0, 2] for finite input, so `z.abs() >= 0` keeps this
    // in [0, 1] mathematically — clamp anyway since this value is documented as a
    // probability and defensive clamping is cheap.
    Ok(erfc(z.abs() / std::f64::consts::SQRT_2).clamp(0.0, 1.0))
}

/// Validate `replicates >= 2`. There is no upper bound: the result buffers are leased from
/// the admission pool ([`admit_replicate_results`]) and per-replicate streams stay disjoint
/// at any count ([`crate::streams::replicate_stream`]).
pub(crate) fn check_replicate_count(replicates: u32) -> Result<(), ValidationError> {
    if replicates < 2 {
        return Err(ValidationError::NotApplicable {
            message: "replicate-based refuter requires replicates >= 2",
        });
    }
    Ok(())
}

/// Cooperative cancellation poll for replicate loops.
pub(crate) fn check_cancelled(ctx: &ExecutionContext) -> Result<(), ValidationError> {
    if ctx.cancellation.is_cancelled() { Err(ValidationError::Cancelled) } else { Ok(()) }
}

/// Mean of the replicate estimates.
pub(crate) fn replicate_mean(ates: &[f64]) -> f64 {
    antecedent_kernels::mean(ates)
}

/// Significance level of the replicate-stability checks that are not user-configurable
/// (query-native mediation and functional refuters).
pub(crate) const STABILITY_ALPHA: f64 = 0.05;

/// Replicates per query-native stability check (mediation and functional refuters).
pub(crate) const STABILITY_REPLICATES: u32 = 20;

/// Lease `count * width` `f64` values that a check retains, before the buffer is built.
///
/// With a context the lease is taken on its pool. Without one the workflow account bound by
/// [`antecedent_core::numerical_admission`] is charged, and a direct call outside any
/// workflow has no pool and returns `None`. Hold the lease until the buffer is dropped.
///
/// # Errors
///
/// [`ValidationError::Cancelled`] when the study is cancelled, otherwise a refusal when the
/// byte count overflows or the pool cannot fund it. Nothing is allocated.
pub(crate) fn admit_cells(
    ctx: Option<&ExecutionContext>,
    count: usize,
    width: usize,
) -> Result<Option<antecedent_core::execution::AdmissionLease>, ValidationError> {
    let bytes = count
        .checked_mul(width)
        .and_then(|cells| cells.checked_mul(size_of::<f64>()))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| ValidationError::Resource {
            message: "check buffer size overflow".into(),
        })?;
    let outcome = match ctx {
        Some(ctx) => Some(ctx.acquire_admission(0, bytes)),
        None => antecedent_core::numerical_admission::acquire(bytes),
    };
    outcome.transpose().map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => ValidationError::Cancelled,
        other => ValidationError::Resource {
            message: format!("admission pool cannot fund the check buffers: {other}"),
        },
    })
}

/// Lease the result vectors of `replicates` replicate values before any is computed.
///
/// A replicate loop returns `replicates` values of `T`, held once by their block and again
/// by the merged vector. The lease holds those bytes on the pool and no CPU slots, because
/// the parallel map leases its workers itself. Hold the result until the values are returned.
///
/// # Errors
///
/// [`ValidationError::Cancelled`] when the study is cancelled, otherwise a refusal when the
/// byte count overflows or the pool cannot fund it. Nothing is allocated.
pub(crate) fn admit_replicate_results<T>(
    ctx: &ExecutionContext,
    replicates: u32,
) -> Result<antecedent_core::execution::AdmissionLease, ValidationError> {
    let bytes = u64::from(replicates)
        .checked_mul(size_of::<T>().max(1) as u64)
        .and_then(|bytes| bytes.checked_mul(2))
        .ok_or_else(|| ValidationError::Resource {
            message: "replicate result size overflow".into(),
        })?;
    ctx.acquire_admission(0, bytes).map_err(|error| match error {
        antecedent_core::execution::AdmissionError::Cancelled => ValidationError::Cancelled,
        other => ValidationError::Resource {
            message: format!("admission pool cannot fund the replicate results: {other}"),
        },
    })
}

/// Run `one(state, ctx, r)` for `r in 0..replicates` and return the results in replicate order.
///
/// The replicates are split into contiguous blocks, one per worker of `ctx`'s pool; every block
/// builds its own scratch with `init` and the closure receives a serial context. A replicate's
/// randomness must derive from `r` alone (a stream id built from `r`, never a shared stream),
/// so the results are bit-identical for every thread count. Cancellation is polled before each
/// replicate, and the error of the lowest failing replicate is the one returned.
pub(crate) fn map_replicates<S, T: Send>(
    ctx: &ExecutionContext,
    replicates: u32,
    init: impl Fn() -> S + Sync,
    one: impl Fn(&mut S, &ExecutionContext, u32) -> Result<T, ValidationError> + Sync,
) -> Result<Vec<T>, ValidationError> {
    let total = replicates as usize;
    if total == 0 {
        return Ok(Vec::new());
    }
    let _results = admit_replicate_results::<T>(ctx, replicates)?;
    let blocks = (ctx.parallelism.max_threads.get() as usize).clamp(1, total);
    let per_block = total.div_ceil(blocks);
    let results = ctx.map_indexed(total.div_ceil(per_block), |block, inner| {
        let mut state = init();
        let start = block * per_block;
        let end = (start + per_block).min(total);
        let mut out = Vec::with_capacity(end - start);
        for r in start..end {
            check_cancelled(inner)?;
            out.push(one(&mut state, inner, u32::try_from(r).unwrap_or(u32::MAX))?);
        }
        Ok::<_, ValidationError>(out)
    })?;
    Ok(results.into_iter().flatten().collect())
}

/// [`map_replicates`] for a refit loop over `problem`.
///
/// `one` receives the shareable view of the problem and its composed-estimator refit. That
/// refit cannot be shared between threads, so a problem that carries one runs its replicates
/// on the caller's thread; every other problem is shared by the workers.
pub(crate) fn map_problem_replicates<S, T: Send>(
    ctx: &ExecutionContext,
    problem: &RefutationProblem<'_>,
    replicates: u32,
    init: impl Fn() -> S + Sync,
    one: impl Fn(
        &SharedProblem<'_, '_>,
        Option<&dyn EffectRefit>,
        &mut S,
        &ExecutionContext,
        u32,
    ) -> Result<T, ValidationError>
    + Sync,
) -> Result<Vec<T>, ValidationError> {
    let plan = compile_replicate_plan(problem)?;
    let shared = shared_for_replicates(problem, plan.as_ref());
    if let Some(refit) = problem.effect_refit {
        let _results = admit_replicate_results::<T>(ctx, replicates)?;
        let mut state = init();
        let mut out = Vec::with_capacity(replicates as usize);
        for r in 0..replicates {
            check_cancelled(ctx)?;
            out.push(one(&shared, Some(refit), &mut state, ctx, r)?);
        }
        return Ok(out);
    }
    map_replicates(ctx, replicates, init, |state, ctx, r| one(&shared, None, state, ctx, r))
}

/// [`map_replicates`] for replicates whose inputs come from one shared sequential RNG stream.
///
/// `draw(r)` runs on the caller's thread in replicate order and owns that stream, so every
/// replicate sees exactly the draws it saw in a serial loop. The drawn inputs are then
/// computed on the pool, a bounded batch at a time (memory stays proportional to the worker
/// count, not the replicate count). A failing draw is reported after the replicates drawn
/// before it were computed, so a serial loop's error priority is kept.
pub(crate) fn map_drawn_replicates<D: Sync, S, T: Send>(
    ctx: &ExecutionContext,
    replicates: u32,
    mut draw: impl FnMut(u32) -> Result<D, ValidationError>,
    init: impl Fn() -> S + Sync,
    compute: impl Fn(&mut S, &ExecutionContext, u32, &D) -> Result<T, ValidationError> + Sync,
) -> Result<Vec<T>, ValidationError> {
    let workers = ctx.parallelism.max_threads.get() as usize;
    let _results = admit_replicate_results::<T>(ctx, replicates)?;
    let mut out = Vec::with_capacity(replicates as usize);
    if workers <= 1 {
        let mut state = init();
        for r in 0..replicates {
            check_cancelled(ctx)?;
            let drawn = draw(r)?;
            out.push(compute(&mut state, ctx, r, &drawn)?);
        }
        return Ok(out);
    }
    let batch = u32::try_from(workers.saturating_mul(2)).unwrap_or(u32::MAX);
    let mut next = 0u32;
    while next < replicates {
        let end = next.saturating_add(batch).min(replicates);
        let mut drawn = Vec::with_capacity((end - next) as usize);
        let mut failed = None;
        for r in next..end {
            match check_cancelled(ctx).and_then(|()| draw(r)) {
                Ok(d) => drawn.push(d),
                Err(e) => {
                    failed = Some(e);
                    break;
                }
            }
        }
        let count = u32::try_from(drawn.len()).unwrap_or(u32::MAX);
        out.extend(map_replicates(ctx, count, &init, |state, ctx, i| {
            compute(state, ctx, next + i, &drawn[i as usize])
        })?);
        if let Some(e) = failed {
            return Err(e);
        }
        next = end;
    }
    Ok(out)
}

/// [`map_drawn_replicates`] for a refit loop over `problem` (see [`map_problem_replicates`]).
pub(crate) fn map_drawn_problem_replicates<D: Sync, S, T: Send>(
    ctx: &ExecutionContext,
    problem: &RefutationProblem<'_>,
    replicates: u32,
    mut draw: impl FnMut(u32) -> Result<D, ValidationError>,
    init: impl Fn() -> S + Sync,
    compute: impl Fn(
        &SharedProblem<'_, '_>,
        Option<&dyn EffectRefit>,
        &mut S,
        &ExecutionContext,
        u32,
        &D,
    ) -> Result<T, ValidationError>
    + Sync,
) -> Result<Vec<T>, ValidationError> {
    let plan = compile_replicate_plan(problem)?;
    let shared = shared_for_replicates(problem, plan.as_ref());
    if let Some(refit) = problem.effect_refit {
        let _results = admit_replicate_results::<T>(ctx, replicates)?;
        let mut state = init();
        let mut out = Vec::with_capacity(replicates as usize);
        for r in 0..replicates {
            check_cancelled(ctx)?;
            let drawn = draw(r)?;
            out.push(compute(&shared, Some(refit), &mut state, ctx, r, &drawn)?);
        }
        return Ok(out);
    }
    map_drawn_replicates(ctx, replicates, draw, init, |state, ctx, r, drawn| {
        compute(&shared, None, state, ctx, r, drawn)
    })
}

/// Compile one panel slice plan for a replicate loop, when the problem has a
/// panel and no plan yet.
///
/// The mutated table is still sliced and prepared on each replicate; only the
/// offset plan is shared. A plan already on the problem is left in place.
fn compile_replicate_plan<'a>(
    problem: &RefutationProblem<'a>,
) -> Result<Option<PanelSliceTemplate<'a>>, ValidationError> {
    if problem.panel_slices.is_some() {
        return Ok(None);
    }
    let Some(panel) = problem.temporal.and_then(|temporal| temporal.panel) else {
        return Ok(None);
    };
    Ok(Some(PanelSliceTemplate::from_panel(panel, problem.data)?))
}

fn shared_for_replicates<'a, 'plan>(
    problem: &'plan RefutationProblem<'a>,
    plan: Option<&'plan PanelSliceTemplate<'a>>,
) -> SharedProblem<'a, 'plan> {
    let mut shared = problem.shared();
    if let Some(plan) = plan {
        shared.panel_slices = Some(plan);
    }
    shared
}

/// Finish a replicate refuter: mean, p-value of `centre` as a typical single replicate
/// ([`replicate_p_value`]), `passed = p >= alpha`, and the (always non-informative) report.
///
/// `failure` renders the failure condition from `(mean, p)` and runs only on a failure.
pub(crate) fn finish_p_report(
    refuter: &str,
    original_ate: f64,
    ates: &[f64],
    centre: f64,
    alpha: f64,
    failure: impl FnOnce(f64, f64) -> String,
) -> Result<RefutationReport, ValidationError> {
    let mean_ate = replicate_mean(ates);
    let p_value = replicate_p_value(ates, centre)?;
    let passed = p_value >= alpha;
    Ok(RefutationReport::builder(refuter)
        .original_ate(original_ate)
        .refuted_ate(mean_ate)
        .comparison(p_value)
        .informative(false)
        .passed(passed)
        .failure_condition((!passed).then(|| Arc::from(failure(mean_ate, p_value))))
        .replicates(u32::try_from(ates.len()).unwrap_or(u32::MAX))
        .build())
}
