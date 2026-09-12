//! Shared estimation helpers (SOLID/DRY).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::similar_names)]

use antecedent_core::{AdaptiveBootstrapBudget, ExecutionContext};
use antecedent_data::{DataError, ResamplingPlan, fill_resample_index_batch};
use antecedent_stats::{StatsError, form_xtx, invert_square};

use crate::error::EstimationError;
use crate::overlap::OverlapPolicy;

/// Map a stats-layer error into [`EstimationError::Stats`].
#[allow(clippy::needless_pass_by_value)] // StatsError is small / owned at call sites
pub(crate) fn stats_err(e: StatsError) -> EstimationError {
    EstimationError::from(e)
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

/// Relative SE change for adaptive early-stop.
#[must_use]
pub(crate) fn se_relative_change(se_prev: f64, se_new: f64) -> f64 {
    if se_prev == 0.0 {
        if se_new == 0.0 { 0.0 } else { f64::INFINITY }
    } else {
        (se_new / se_prev - 1.0).abs()
    }
}

/// Whether adaptive bootstrap should stop given the current SE trajectory.
#[must_use]
pub(crate) fn adaptive_bootstrap_should_stop(
    budget: AdaptiveBootstrapBudget,
    successful: u32,
    se_prev: Option<f64>,
    se_new: f64,
) -> bool {
    if !budget.enabled || successful < budget.min_replicates.max(2) {
        return false;
    }
    let Some(prev) = se_prev else {
        return false;
    };
    if !prev.is_finite() || !se_new.is_finite() {
        return false;
    }
    se_relative_change(prev, se_new) < budget.se_rel_epsilon
}

/// IID bootstrap standard error with failure accounting and optional adaptive early-stop.
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
/// Adaptive early-stop: when [`ExecutionContext::adaptive_bootstrap`] is enabled and the SE
/// relative change falls below ε after `min_replicates` successes, returns with
/// `early_stopped = true`.
pub(crate) fn bootstrap_se(
    replicates: u32,
    ctx: &ExecutionContext,
    stream_base: u64,
    n: usize,
    mut estimate: impl FnMut(&[usize]) -> Result<Option<f64>, EstimationError>,
) -> Result<BootstrapSeResult, EstimationError> {
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
    let mut ates = Vec::with_capacity(n_rep);
    let mut idx = vec![0usize; n];
    let mut cancelled = false;
    let mut early_stopped = false;
    let mut attempted = 0u32;
    let mut se_prev: Option<f64> = None;
    let budget = ctx.adaptive_bootstrap;
    for r in 0..n_rep {
        if ctx.cancellation.is_cancelled() {
            cancelled = true;
            break;
        }
        let slice = &indexes[r * n..(r + 1) * n];
        for (dst, &src) in idx.iter_mut().zip(slice.iter()) {
            *dst = src as usize;
        }
        attempted += 1;
        let estimate = estimate(&idx)?.filter(|ate| ate.is_finite());
        if let Some(ate) = estimate {
            ates.push(ate);
        }
        if ctx.cancellation.is_cancelled() {
            cancelled = true;
            break;
        }
        if estimate.is_some() && ates.len() >= 2 {
            let se_new = sample_std(&ates);
            let ok_count = u32::try_from(ates.len()).unwrap_or(u32::MAX);
            if adaptive_bootstrap_should_stop(budget, ok_count, se_prev, se_new) {
                early_stopped = true;
                if let Some(p) = &ctx.progress {
                    p.report((r + 1) as f64 / n_rep as f64, "bootstrap");
                }
                break;
            }
            se_prev = Some(se_new);
        }
        if let Some(p) = &ctx.progress {
            p.report((r + 1) as f64 / n_rep as f64, "bootstrap");
        }
    }
    Ok(finalize_bootstrap_se_ex(&ates, attempted, cancelled, early_stopped))
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
    use antecedent_core::{AdaptiveBootstrapBudget, CancellationToken, ExecutionContext};

    #[test]
    fn adaptive_se_change_is_invariant_to_measurement_units() {
        for scale in [1e-200, 1.0, 1e200] {
            assert!((se_relative_change(scale, 1.01 * scale) - 0.01).abs() < 1e-12);
            assert!(se_relative_change(0.0, scale).is_infinite());
            assert!((se_relative_change(scale, 0.0) - 1.0).abs() < 1e-12);
        }
        assert!(se_relative_change(0.0, 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn cancelled_bootstrap_retains_actual_failures() {
        let ctx = ExecutionContext::for_tests(7);
        let mut calls = 0;
        let result = bootstrap_se(20, &ctx, 1, 10, |_| {
            calls += 1;
            if calls == 4 {
                ctx.cancellation.cancel();
            }
            Ok(if calls <= 2 { Some(f64::from(calls)) } else { None })
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
        ctx.adaptive_bootstrap =
            AdaptiveBootstrapBudget { enabled: true, min_replicates: 2, se_rel_epsilon: 0.1 };
        let mut calls = 0;
        let result = bootstrap_se(20, &ctx, 1, 10, |_| {
            calls += 1;
            Ok(if calls <= 4 { None } else { Some(1.0) })
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
        let mut i = 0;
        let result = bootstrap_se(4, &ctx, 1, 5, |_| {
            let value = values[i];
            i += 1;
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
    fn adaptive_stop_requires_min_and_relative_eps() {
        let budget =
            AdaptiveBootstrapBudget { enabled: true, min_replicates: 4, se_rel_epsilon: 0.05 };
        assert!(!adaptive_bootstrap_should_stop(budget, 3, Some(1.0), 1.0));
        assert!(!adaptive_bootstrap_should_stop(budget, 4, None, 1.0));
        assert!(!adaptive_bootstrap_should_stop(budget, 4, Some(1.0), 1.1)); // 10% > 5%
        assert!(adaptive_bootstrap_should_stop(budget, 4, Some(1.0), 1.01)); // 1% < 5%
        assert!(!adaptive_bootstrap_should_stop(
            AdaptiveBootstrapBudget::disabled(),
            100,
            Some(1.0),
            1.0
        ));
    }

    #[test]
    fn bootstrap_adaptive_early_stop_stable_count() {
        let mut ctx = ExecutionContext::for_tests(42);
        ctx.adaptive_bootstrap =
            AdaptiveBootstrapBudget { enabled: true, min_replicates: 5, se_rel_epsilon: 0.02 };
        // Constant ATE → SE → 0 quickly; should early-stop soon after min.
        let r1 = bootstrap_se(80, &ctx, 0xABCD, 20, |_| Ok(Some(2.0))).unwrap();
        let r2 = bootstrap_se(80, &ctx, 0xABCD, 20, |_| Ok(Some(2.0))).unwrap();
        assert!(r1.early_stopped);
        assert_eq!(r1.replicates_ok, r2.replicates_ok);
        assert!(r1.replicates_ok >= 5);
        assert!(r1.replicates_ok < 80);
        assert!(!r1.cancelled);
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
