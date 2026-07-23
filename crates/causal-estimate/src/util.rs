//! Shared estimation helpers (SOLID/DRY).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation, clippy::similar_names)]

use causal_core::{AdaptiveBootstrapBudget, ExecutionContext};
use causal_data::{ResamplingPlan, fill_resample_index_batch};
use causal_stats::{StatsError, form_xtx, invert_square};

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
    causal_stats::sample_std(values)
}

/// Floor for relative SE change denominator.
const SE_REL_EPS_FLOOR: f64 = 1e-12;

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
/// When `cancelled` or `early_stopped`, unattempted replicates are not counted as failures.
#[must_use]
pub(crate) fn finalize_bootstrap_se_ex(
    ates: &[f64],
    replicates: u32,
    cancelled: bool,
    early_stopped: bool,
) -> BootstrapSeResult {
    let ok = u32::try_from(ates.len()).unwrap_or(u32::MAX);
    let partial = cancelled || early_stopped;
    let failed = if partial { 0 } else { replicates.saturating_sub(ok) };
    let too_few = ates.len() < 2;
    let fail_frac =
        if replicates == 0 || partial { 0.0 } else { f64::from(failed) / f64::from(replicates) };
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
    (se_new - se_prev).abs() / se_prev.abs().max(SE_REL_EPS_FLOOR)
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
    let n_rep = replicates as usize;
    let mut indexes = vec![0u32; n * n_rep];
    fill_resample_index_batch(
        ResamplingPlan::IidBootstrap,
        n,
        n_rep,
        None,
        ctx,
        stream_base,
        &mut indexes,
    )
    .map_err(EstimationError::from)?;
    let mut ates = Vec::with_capacity(n_rep);
    let mut idx = vec![0usize; n];
    let mut cancelled = false;
    let mut early_stopped = false;
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
        if let Some(ate) = estimate(&idx)? {
            ates.push(ate);
            if ates.len() >= 2 {
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
        }
        if let Some(p) = &ctx.progress {
            p.report((r + 1) as f64 / n_rep as f64, "bootstrap");
        }
    }
    Ok(finalize_bootstrap_se_ex(&ates, replicates, cancelled, early_stopped))
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

#[cfg(test)]
mod tests {
    use super::*;
    use causal_core::{AdaptiveBootstrapBudget, CancellationToken, ExecutionContext};

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
        let r = finalize_bootstrap_se_ex(&[1.0, 1.1, 0.9, 1.05], 50, false, true);
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
