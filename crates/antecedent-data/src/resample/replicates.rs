//! Replicate and batch resampling: per-replicate streams, weight fills and scratch admission.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use super::indexes::*;

/// Fill `out` with length-`n` Bayesian-bootstrap weights (normalized Exp(1) draws).
///
/// Weights sum to `n` (mean weight 1) so weighted estimators match unweighted
/// scale on the original sample size.
///
/// # Errors
///
/// Zero length, or a non-weight plan.
pub(crate) fn fill_resample_weights(
    plan: ResamplingPlan,
    n: usize,
    rng: &mut CausalRng,
    out: &mut Vec<f64>,
) -> Result<(), DataError> {
    if n == 0 {
        return Err(DataError::InvalidArgument { message: "resample needs n > 0".into() });
    }
    if !matches!(plan, ResamplingPlan::BayesianBootstrap) {
        return Err(DataError::InvalidArgument {
            message: "fill_resample_weights requires BayesianBootstrap".into(),
        });
    }
    // The workspace endpoint policy and mean-one normalization are the kernel's: one uniform
    // per slot in order, floored at `f64::MIN_POSITIVE` (only the exact-zero and `2^-53`
    // uniforms differ from a floor at `f64::EPSILON`), then `wᵢ *= n / Σw`.
    out.clear();
    out.resize(n, 0.0);
    fill_bayesian_bootstrap_weights(rng, out, true);
    // Every raw exponential is strictly positive and finite, so a non-positive or non-finite
    // weight can only come from a zero or non-finite total.
    if !out.iter().all(|w| w.is_finite() && *w > 0.0) {
        return Err(DataError::InvalidArgument {
            message: "BayesianBootstrap weight sum non-positive".into(),
        });
    }
    Ok(())
}

pub(super) fn resample_rng(ctx: &ExecutionContext, stream_base: u64, replicate: u64) -> CausalRng {
    ctx.rng.stream_for(StreamDomain::Resample, derive_stream(stream_base, replicate))
}

pub(super) fn checked_batch_len(n: usize, n_replicates: usize) -> Result<usize, DataError> {
    if n == 0 || n_replicates == 0 {
        return Err(DataError::InvalidArgument {
            message: "batch resample needs n > 0 and n_replicates > 0".into(),
        });
    }
    n.checked_mul(n_replicates).ok_or_else(|| DataError::InvalidArgument {
        message: "batch resample dimensions overflow usize".into(),
    })
}

pub(super) fn check_resampling_cancelled(ctx: &ExecutionContext) -> Result<(), DataError> {
    if ctx.cancellation.is_cancelled() {
        return Err(DataError::Cancelled);
    }
    Ok(())
}

/// Fill `out` with one replicate's row indexes, reusing the buffer.
///
/// The draw is `ctx.rng.stream_for(StreamDomain::Resample, derive_stream(stream_base, replicate))`,
/// the same stream as row `replicate` of [`fill_resample_index_batch`]. `out` is cleared first. Index
/// plans other than cluster bootstrap write `n` ids. A cluster bootstrap writes one copy of
/// each drawn cluster's rows, so unequal clusters change `out.len()` and a cluster drawn `k`
/// times contributes `k` separate copies under its original row ids.
///
/// # Errors
///
/// Cancellation, `n == 0`, a weight plan, a clustered plan without matching `cluster_ids`, or
/// a row id that does not fit in [`GlobalId`].
pub fn fill_resample_index_replicate(
    plan: ResamplingPlan,
    n: usize,
    cluster_ids: Option<&[u32]>,
    ctx: &ExecutionContext,
    stream_base: u64,
    replicate: u64,
    out: &mut Vec<GlobalId>,
) -> Result<(), DataError> {
    check_resampling_cancelled(ctx)?;
    let clusters = prepare_clusters(plan, n, cluster_ids)?;
    let mut rng = resample_rng(ctx, stream_base, replicate);
    fill_indexes_prepared(plan, n, clusters.as_ref(), &mut rng, out)
}

/// Fill `out` with one Bayesian-bootstrap weight replicate, reusing the buffer.
///
/// The draw is `ctx.rng.stream_for(StreamDomain::Resample, derive_stream(stream_base, replicate))`,
/// the same stream as one row of a replicate-major Bayesian-bootstrap weight batch. Weights sum to `n`.
///
/// # Errors
///
/// Cancellation, `n == 0`, a non-weight plan, or a non-positive weight sum.
pub fn fill_resample_weight_replicate(
    plan: ResamplingPlan,
    n: usize,
    ctx: &ExecutionContext,
    stream_base: u64,
    replicate: u64,
    out: &mut Vec<f64>,
) -> Result<(), DataError> {
    check_resampling_cancelled(ctx)?;
    let mut rng = resample_rng(ctx, stream_base, replicate);
    fill_resample_weights(plan, n, &mut rng, out)
}

/// Fill `out` (`len = n * n_replicates`, replicate-major) with index plans under one
/// [`ExecutionContext`].
///
/// Each replicate uses `ctx.rng.stream_for(StreamDomain::Resample, derive_stream(stream_base, replicate_id))`
/// ([`fill_resample_index_replicate`]) so results are independent of scheduling order under
/// [`antecedent_core::Determinism::Strict`]. When `max_threads > 1` and `n_replicates ≥ 2`,
/// fills run in a bounded `std::thread::scope` pool.
///
/// # Errors
///
/// Shape / plan mismatches as in [`fill_resample_indexes_grouped`], dimension overflow,
/// cancellation, or `out` length mismatch.
/// Unequal cluster sizes cannot fit this fixed-width representation and are refused.
pub fn fill_resample_index_batch(
    plan: ResamplingPlan,
    n: usize,
    n_replicates: usize,
    cluster_ids: Option<&[u32]>,
    ctx: &ExecutionContext,
    stream_base: u64,
    out: &mut [GlobalId],
) -> Result<(), DataError> {
    check_resampling_cancelled(ctx)?;
    let expected = checked_batch_len(n, n_replicates)?;
    if out.len() != expected {
        return Err(DataError::LengthMismatch {
            expected,
            actual: out.len(),
            context: "fill_resample_index_batch out",
        });
    }
    if plan.is_weight_plan() {
        return Err(DataError::InvalidArgument {
            message: "BayesianBootstrap yields weights; use fill_resample_weight_batch".into(),
        });
    }
    if matches!(plan, ResamplingPlan::ClusterBootstrap { .. }) {
        let ids = cluster_ids.ok_or_else(|| DataError::InvalidArgument {
            message: "clustered resampling requires cluster_ids".into(),
        })?;
        if ids.len() != n {
            return Err(DataError::LengthMismatch {
                expected: n,
                actual: ids.len(),
                context: "cluster_ids",
            });
        }
        let mut counts = std::collections::BTreeMap::new();
        for &id in ids {
            *counts.entry(id).or_insert(0usize) += 1;
        }
        let mut sizes = counts.values();
        let first = sizes.next();
        if sizes.any(|size| Some(size) != first) {
            return Err(DataError::InvalidArgument {
                message: "unequal cluster sizes cannot be represented by fixed-width batch output"
                    .into(),
            });
        }
    }
    let clusters = prepare_clusters(plan, n, cluster_ids)?;
    fill_replicate_rows(ctx, n, n_replicates, out, |r, scratch| {
        let mut rng = resample_rng(ctx, stream_base, r as u64);
        fill_indexes_prepared(plan, n, clusters.as_ref(), &mut rng, scratch)
    })
}

/// Batch Bayesian-bootstrap weights under one [`ExecutionContext`] (replicate-major).
///
/// # Errors
///
/// Non-weight plan, zero/overflowing sizes, cancellation, or `out` length mismatch.
#[cfg(test)]
pub(crate) fn fill_resample_weight_batch(
    plan: ResamplingPlan,
    n: usize,
    n_replicates: usize,
    ctx: &ExecutionContext,
    stream_base: u64,
    out: &mut [f64],
) -> Result<(), DataError> {
    check_resampling_cancelled(ctx)?;
    let expected = checked_batch_len(n, n_replicates)?;
    if out.len() != expected {
        return Err(DataError::LengthMismatch {
            expected,
            actual: out.len(),
            context: "fill_resample_weight_batch out",
        });
    }
    if !matches!(plan, ResamplingPlan::BayesianBootstrap) {
        return Err(DataError::InvalidArgument {
            message: "fill_resample_weight_batch requires BayesianBootstrap".into(),
        });
    }
    fill_replicate_rows(ctx, n, n_replicates, out, |r, scratch| {
        let mut rng = resample_rng(ctx, stream_base, r as u64);
        fill_resample_weights(plan, n, &mut rng, scratch)
    })
}

/// Run `fill` for every replicate `r` and copy its length-`n` result into row `r` of the
/// replicate-major `out`. Replicates are independent (each derives its own RNG stream), so
/// the output does not depend on the thread count; the first error wins.
///
/// Scratch is one length-`n` buffer per worker. The worker count does not exceed
/// the admission pool. A product that overflows or a pool that cannot fund
/// that scratch returns before `fill` runs and before any scratch buffer is
/// allocated.
pub(super) fn fill_replicate_rows<T: Copy + Send>(
    ctx: &ExecutionContext,
    n: usize,
    n_replicates: usize,
    out: &mut [T],
    fill: impl Fn(usize, &mut Vec<T>) -> Result<(), DataError> + Sync,
) -> Result<(), DataError> {
    let lease = admit_resample_scratch::<T>(ctx, n, n_replicates)?;
    let workers = lease.workers();
    if workers == 1 {
        let mut scratch = Vec::with_capacity(n);
        for r in 0..n_replicates {
            fill(r, &mut scratch)?;
            out[r * n..(r + 1) * n].copy_from_slice(&scratch);
        }
        return Ok(());
    }
    let chunk = n_replicates.div_ceil(workers);
    let err = std::sync::Mutex::new(None::<DataError>);
    std::thread::scope(|scope| {
        let mut rest = out;
        let mut start = 0usize;
        while start < n_replicates {
            let end = (start + chunk).min(n_replicates);
            let (this, next) = rest.split_at_mut((end - start) * n);
            rest = next;
            let err_slot = &err;
            let fill = &fill;
            scope.spawn(move || {
                let mut scratch = Vec::with_capacity(n);
                for (local, r) in (start..end).enumerate() {
                    if let Err(e) = fill(r, &mut scratch) {
                        let mut guard =
                            err_slot.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                        if guard.is_none() {
                            *guard = Some(e);
                        }
                        return;
                    }
                    this[local * n..(local + 1) * n].copy_from_slice(&scratch);
                }
            });
            start = end;
        }
    });
    if let Some(e) = err.into_inner().unwrap_or_else(std::sync::PoisonError::into_inner) {
        return Err(e);
    }
    Ok(())
}

/// Lease the workers and scratch for one batch.
///
/// `workers` length-`n` buffers of `T` are held on the admission pool until the
/// returned lease drops, so concurrent batches share one remainder.
///
/// # Errors
///
/// The product does not fit in `u64`, the admission pool cannot fund it, or
/// the study is cancelled. Nothing is
/// allocated.
pub(super) fn admit_resample_scratch<T>(
    ctx: &ExecutionContext,
    n: usize,
    n_replicates: usize,
) -> Result<ScratchLease, DataError> {
    let per_worker =
        n.checked_mul(size_of::<T>()).and_then(|bytes| u64::try_from(bytes).ok()).ok_or_else(
            || DataError::InvalidArgument { message: "resample scratch size overflow".into() },
        )?;
    ctx.admit_scratch(n_replicates, per_worker).map_err(|refusal| match refusal {
        ScratchRefusal::Overflow => {
            DataError::InvalidArgument { message: "resample scratch size overflow".into() }
        }
        ScratchRefusal::PoolShort { .. } => DataError::InvalidArgument {
            message: "admission pool cannot fund the resample scratch".into(),
        },
        ScratchRefusal::Cancelled => DataError::Cancelled,
        _ => DataError::InvalidArgument { message: refusal.to_string() },
    })
}
