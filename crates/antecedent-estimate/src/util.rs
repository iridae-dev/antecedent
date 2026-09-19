//! Shared estimation helpers (SOLID/DRY).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::similar_names)]

use antecedent_core::{AdaptiveBootstrapBudget, ExecutionContext};
use antecedent_data::{DataError, ResamplingPlan, fill_resample_index_batch};
use antecedent_stats::{StatsError, chol_solve, cholesky_spd, form_xtx, invert_square};

use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;

/// Map a stats-layer error into [`EstimationError::Stats`].
#[allow(clippy::needless_pass_by_value)] // StatsError is small / owned at call sites
pub(crate) fn stats_err(e: StatsError) -> EstimationError {
    EstimationError::from(e)
}

/// Solve `A x = b` for symmetric positive-definite row-major `A` (`p × p`).
#[must_use]
pub(crate) fn solve_spd(a: &[f64], b: &[f64], p: usize) -> Option<Vec<f64>> {
    let chol = cholesky_spd(a, p)?;
    chol_solve(&chol, p, b)
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

/// Unbiased sample standard deviation; `NaN` if fewer than 2 observations.
pub(crate) fn sample_std(values: &[f64]) -> f64 {
    antecedent_stats::sample_std(values)
}

/// Monte Carlo critical value of a max-statistic band: the `⌈level·(B+1)⌉`-th
/// smallest of `B` ascending-sorted replicate maxima (rank clamped to `[1, B]`).
///
/// The `B + 1` rank treats the observed statistic as one more exchangeable draw
/// (Davison & Hinkley 1997, §4.2), so the band is conservative at every `B`;
/// `⌈level·B⌉` sits one rank lower whenever `level·B` is not an integer. Every
/// simultaneous response band (multiplier, Gaussian max-t and replicate sup-t)
/// reads its critical value here. `NaN` for an empty slice.
#[allow(clippy::cast_sign_loss)]
pub(crate) fn monte_carlo_critical(sorted_maxima: &[f64], level: f64) -> f64 {
    let b = sorted_maxima.len();
    if b == 0 {
        return f64::NAN;
    }
    let rank = ((level * (b as f64 + 1.0)).ceil() as usize).clamp(1, b);
    sorted_maxima[rank - 1]
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

/// Maximum allowed soft-failure fraction before refusing to report an SE.
pub(crate) const BOOTSTRAP_MAX_FAILURE_FRAC: f64 = 0.5;

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

/// IID bootstrap standard error with failure accounting and optional early-stop.
///
/// Index plans are produced in one batch under `ctx` via
/// [`fill_resample_index_batch`] for the full requested `replicates` (CRN-stable under
/// fixed seeds even when evaluation stops early).
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
pub(crate) fn bootstrap_se(
    replicates: u32,
    ctx: &ExecutionContext,
    stream_base: u64,
    n: usize,
    estimate: impl Fn(&[usize]) -> Result<Option<f64>, EstimationError> + Sync,
) -> Result<BootstrapSeResult, EstimationError> {
    bootstrap_se_with_scratch(replicates, ctx, stream_base, n, || (), |_, idx| estimate(idx))
}

/// [`bootstrap_se`] with per-worker scratch so replicate evaluation can run
/// under [`ExecutionContext::parallelism`] without a shared `&mut` workspace.
///
/// Adaptive early-stop stays serial (the floor is a running success count).
/// When an outer map already saturates the thread budget, `ctx` is serial and
/// this path does not spawn.
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
    let len = n
        .checked_mul(n_rep)
        .filter(|&len| {
            len.checked_mul(std::mem::size_of::<u32>())
                .is_some_and(|bytes| isize::try_from(bytes).is_ok())
        })
        .ok_or_else(|| {
            EstimationError::data_msg("bootstrap index allocation exceeds addressable capacity")
        })?;
    if u32::try_from(n - 1).is_err() {
        return Err(EstimationError::data_msg("bootstrap rows exceed u32 index capacity"));
    }
    let mut indexes = Vec::new();
    indexes.try_reserve_exact(len).map_err(|e| {
        EstimationError::data_msg(format!("bootstrap index allocation failed: {e}"))
    })?;
    indexes.resize(len, 0u32);
    if let Err(error) = fill_resample_index_batch(
        ResamplingPlan::IidBootstrap,
        n,
        n_rep,
        None,
        ctx,
        stream_base,
        &mut indexes,
    ) {
        // A cancellation can race with the check above. Other data failures must
        // remain errors, even if cancellation happens concurrently with them.
        if ctx.cancellation.is_cancelled()
            && matches!(&error, DataError::InvalidArgument { message } if message == "resampling cancelled")
        {
            return Ok(finalize_bootstrap_se_ex(&[], 0, true, false));
        }
        return Err(error.into());
    }
    if ctx.cancellation.is_cancelled() {
        return Ok(finalize_bootstrap_se_ex(&[], 0, true, false));
    }
    let threads = (ctx.parallelism.max_threads.get() as usize).clamp(1, n_rep);
    let parallel = threads > 1 && !ctx.adaptive_bootstrap.enabled;
    if parallel {
        return evaluate_bootstrap_parallel(ctx, &indexes, n, n_rep, make_scratch, estimate);
    }
    evaluate_bootstrap_serial(ctx, &indexes, n, n_rep, make_scratch, estimate)
}

fn fill_replicate_idx(idx: &mut [usize], indexes: &[u32], n: usize, r: usize) {
    let slice = &indexes[r * n..(r + 1) * n];
    for (dst, &src) in idx.iter_mut().zip(slice.iter()) {
        *dst = src as usize;
    }
}

fn evaluate_bootstrap_serial<S, F>(
    ctx: &ExecutionContext,
    indexes: &[u32],
    n: usize,
    n_rep: usize,
    make_scratch: impl Fn() -> S,
    estimate: F,
) -> Result<BootstrapSeResult, EstimationError>
where
    F: Fn(&mut S, &[usize]) -> Result<Option<f64>, EstimationError>,
{
    let mut scratch = make_scratch();
    let mut ates = Vec::with_capacity(n_rep);
    let mut idx = vec![0usize; n];
    let mut cancelled = false;
    let mut early_stopped = false;
    let mut attempted = 0u32;
    let budget = ctx.adaptive_bootstrap;
    for r in 0..n_rep {
        if ctx.cancellation.is_cancelled() {
            cancelled = true;
            break;
        }
        fill_replicate_idx(&mut idx, indexes, n, r);
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
    indexes: &[u32],
    n: usize,
    n_rep: usize,
    make_scratch: impl Fn() -> S + Sync,
    estimate: F,
) -> Result<BootstrapSeResult, EstimationError>
where
    S: Send,
    F: Fn(&mut S, &[usize]) -> Result<Option<f64>, EstimationError> + Sync,
{
    let threads = (ctx.parallelism.max_threads.get() as usize).clamp(1, n_rep);
    let mut slots: Vec<Option<Result<Option<f64>, EstimationError>>> =
        (0..n_rep).map(|_| None).collect();
    std::thread::scope(|scope| {
        let estimate = &estimate;
        let make_scratch = &make_scratch;
        let indexes = indexes;
        let mut rest = slots.as_mut_slice();
        let mut start = 0usize;
        for t in 0..threads {
            let take = rest.len().div_ceil(threads - t);
            let (mine, next) = rest.split_at_mut(take);
            let begin = start;
            scope.spawn(move || {
                let mut scratch = make_scratch();
                let mut idx = vec![0usize; n];
                for (k, slot) in mine.iter_mut().enumerate() {
                    let r = begin + k;
                    if ctx.cancellation.is_cancelled() {
                        break;
                    }
                    fill_replicate_idx(&mut idx, indexes, n, r);
                    *slot =
                        Some(estimate(&mut scratch, &idx).map(|v| v.filter(|ate| ate.is_finite())));
                }
            });
            rest = next;
            start += take;
            if rest.is_empty() {
                break;
            }
        }
    });
    let mut ates = Vec::with_capacity(n_rep);
    let mut attempted = 0u32;
    let mut cancelled = ctx.cancellation.is_cancelled();
    for slot in slots {
        match slot {
            None => {
                cancelled = true;
                break;
            }
            Some(Err(error)) => return Err(error),
            Some(Ok(value)) => {
                attempted += 1;
                if let Some(ate) = value {
                    ates.push(ate);
                }
            }
        }
    }
    if let Some(p) = &ctx.progress {
        p.report(1.0, "bootstrap");
    }
    Ok(finalize_bootstrap_se_ex(&ates, attempted, cancelled, false))
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
    use super::*;
    use std::sync::atomic::{AtomicU32, Ordering};

    use antecedent_core::{AdaptiveBootstrapBudget, CancellationToken, ExecutionContext};

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
}
