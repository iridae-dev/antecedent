//! Temporal bootstrap / resampling index plans.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::neg_cmp_op_on_partial_ord
)]

use std::sync::Arc;

use antecedent_core::{CausalRng, ExecutionContext, VariableId};
use antecedent_kernels::unbiased_index;

use crate::column::{ColumnView, Float64Column, OwnedColumn};
use crate::dataset::TimeSeriesData;
use crate::error::DataError;
use crate::storage::OwnedColumnarStorage;
use crate::table::TableView;

/// Null / permutation scheme for [`ResamplingPlan::Permutation`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum PermutationScheme {
    /// Full Fisher–Yates (Fisher & Yates 1938; Durstenfeld 1964) shuffle of row indexes `0..n`.
    Full,
    /// Shuffle within each cluster (ids supplied to the fill helper).
    WithinCluster,
}

/// Resampling plan producing row-index or weight replicates.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ResamplingPlan {
    /// IID with-replacement row bootstrap.
    IidBootstrap,
    /// Bayesian bootstrap: Dirichlet(1,…,1) observation weights (Rubin).
    BayesianBootstrap,
    /// Moving-block bootstrap (contiguous blocks).
    MovingBlock {
        /// Block length in rows.
        length: usize,
    },
    /// Circular block bootstrap (wraps the series).
    CircularBlock {
        /// Block length in rows.
        length: usize,
    },
    /// Cluster bootstrap: draw G whole clusters with replacement from G observed clusters.
    /// Replicate row counts vary when cluster sizes differ (ids via grouped fill).
    ///
    /// `cluster` names the variable whose labels are supplied as `cluster_ids`
    /// to the fill helpers (labels remain caller-resolved).
    ClusterBootstrap {
        /// Variable that holds cluster membership labels.
        cluster: VariableId,
    },
    /// Stationary block bootstrap (Politis & Romano 1994 geometric lengths).
    StationaryBlock {
        /// Expected block length (mean of geometric distribution).
        expected_length: f64,
    },
    /// Permutation (shuffle) index plan.
    Permutation(PermutationScheme),
}

impl ResamplingPlan {
    /// Whether this plan yields observation weights rather than row indexes.
    #[must_use]
    pub const fn is_weight_plan(self) -> bool {
        matches!(self, Self::BayesianBootstrap)
    }

    /// Whether this plan requires per-row cluster labels.
    #[must_use]
    pub const fn needs_clusters(self) -> bool {
        matches!(
            self,
            Self::ClusterBootstrap { .. } | Self::Permutation(PermutationScheme::WithinCluster)
        )
    }

    /// Cluster variable recorded on [`ResamplingPlan::ClusterBootstrap`], if any.
    #[must_use]
    pub const fn cluster_variable(self) -> Option<VariableId> {
        match self {
            Self::ClusterBootstrap { cluster } => Some(cluster),
            _ => None,
        }
    }
}

/// Fill `out` with a length-`n` row-index plan under `plan`.
///
/// Plans that need cluster labels ([`ResamplingPlan::ClusterBootstrap`],
/// [`PermutationScheme::WithinCluster`]) must use
/// [`fill_resample_indexes_grouped`] instead.
///
/// # Errors
///
/// Zero series length, zero block length, weight-only plan, or a clustered plan.
pub fn fill_resample_indexes(
    plan: ResamplingPlan,
    n: usize,
    rng: &mut CausalRng,
    out: &mut Vec<u32>,
) -> Result<(), DataError> {
    if plan.needs_clusters() {
        return Err(DataError::InvalidArgument {
            message: "clustered plan requires fill_resample_indexes_grouped".into(),
        });
    }
    fill_resample_indexes_grouped(plan, n, None, rng, out)
}

/// Fill `out` with a row-index plan, optionally using `cluster_ids`.
///
/// Cluster bootstrap draws as many clusters as were observed, so unequal cluster
/// sizes produce variable-length output. All other index plans return `n` rows.
///
/// # Errors
///
/// Shape / plan mismatches as in [`fill_resample_indexes`], plus missing or
/// wrong-length cluster ids when required.
///
/// # Panics
///
/// Propagates panic from [`unbiased_index`] if a zero-length draw is requested
/// after the `n > 0` guard (should be unreachable).
pub fn fill_resample_indexes_grouped(
    plan: ResamplingPlan,
    n: usize,
    cluster_ids: Option<&[u32]>,
    rng: &mut CausalRng,
    out: &mut Vec<u32>,
) -> Result<(), DataError> {
    if n == 0 {
        return Err(DataError::InvalidArgument { message: "resample needs n > 0".into() });
    }
    if plan.is_weight_plan() {
        return Err(DataError::InvalidArgument {
            message: "BayesianBootstrap yields weights; use fill_resample_weights".into(),
        });
    }
    if plan.needs_clusters() {
        let Some(ids) = cluster_ids else {
            return Err(DataError::InvalidArgument {
                message: "clustered resampling requires cluster_ids".into(),
            });
        };
        if ids.len() != n {
            return Err(DataError::LengthMismatch {
                expected: n,
                actual: ids.len(),
                context: "cluster_ids",
            });
        }
    }
    out.clear();
    out.reserve(n);
    match plan {
        ResamplingPlan::IidBootstrap => {
            for _ in 0..n {
                out.push(unbiased_index(rng, n) as u32);
            }
        }
        ResamplingPlan::BayesianBootstrap => unreachable!("checked above"),
        ResamplingPlan::MovingBlock { length } | ResamplingPlan::CircularBlock { length } => {
            fill_block(n, length, matches!(plan, ResamplingPlan::CircularBlock { .. }), rng, out)?;
        }
        ResamplingPlan::StationaryBlock { expected_length } => {
            fill_stationary(n, expected_length, rng, out)?;
        }
        ResamplingPlan::ClusterBootstrap { .. } => {
            fill_cluster_bootstrap(n, cluster_ids.unwrap(), rng, out)?;
        }
        ResamplingPlan::Permutation(PermutationScheme::Full) => {
            out.extend(0..n as u32);
            // Fisher–Yates
            for i in (1..n).rev() {
                let j = unbiased_index(rng, i + 1);
                out.swap(i, j);
            }
        }
        ResamplingPlan::Permutation(PermutationScheme::WithinCluster) => {
            fill_within_cluster_permutation(n, cluster_ids.unwrap(), rng, out);
        }
    }
    Ok(())
}

fn fill_block(
    n: usize,
    length: usize,
    circular: bool,
    rng: &mut CausalRng,
    out: &mut Vec<u32>,
) -> Result<(), DataError> {
    if length == 0 {
        return Err(DataError::InvalidArgument { message: "block length must be > 0".into() });
    }
    if length > n {
        return Err(DataError::InvalidArgument {
            message: format!("block length {length} exceeds series length {n}"),
        });
    }
    let n_starts = if circular { n } else { n.saturating_sub(length).saturating_add(1) };
    while out.len() < n {
        let start = unbiased_index(rng, n_starts);
        for k in 0..length {
            if out.len() >= n {
                break;
            }
            let idx = if circular { (start + k) % n } else { start + k };
            out.push(idx as u32);
        }
    }
    out.truncate(n);
    Ok(())
}

fn fill_stationary(
    n: usize,
    expected_length: f64,
    rng: &mut CausalRng,
    out: &mut Vec<u32>,
) -> Result<(), DataError> {
    if !(expected_length.is_finite() && expected_length >= 1.0) {
        return Err(DataError::InvalidArgument {
            message: "stationary expected_length must be finite and ≥ 1".into(),
        });
    }
    // Geometric block length with mean L: P(L=k) = p(1-p)^{k-1}, p = 1/L.
    let p = 1.0 / expected_length;
    while out.len() < n {
        let mut len = 1usize;
        while rng.next_f64() > p {
            len += 1;
            if len > n {
                break;
            }
        }
        len = len.min(n);
        let start = unbiased_index(rng, n);
        for k in 0..len {
            if out.len() >= n {
                break;
            }
            out.push(((start + k) % n) as u32);
        }
    }
    out.truncate(n);
    Ok(())
}

fn fill_cluster_bootstrap(
    n: usize,
    cluster_ids: &[u32],
    rng: &mut CausalRng,
    out: &mut Vec<u32>,
) -> Result<(), DataError> {
    // Build cluster → row list.
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| cluster_ids[i]);
    let mut clusters: Vec<Vec<u32>> = Vec::new();
    let mut idx = 0usize;
    while idx < n {
        let g = cluster_ids[order[idx]];
        let mut members = Vec::new();
        while idx < n && cluster_ids[order[idx]] == g {
            members.push(order[idx] as u32);
            idx += 1;
        }
        clusters.push(members);
    }
    if clusters.is_empty() {
        return Err(DataError::InvalidArgument { message: "no clusters".into() });
    }
    // G IID draws from the empirical cluster distribution. Conditioning on
    // exactly n rows changes that distribution when clusters have unequal sizes.
    for _ in 0..clusters.len() {
        let c = unbiased_index(rng, clusters.len());
        out.extend_from_slice(&clusters[c]);
    }
    Ok(())
}

fn fill_within_cluster_permutation(
    n: usize,
    cluster_ids: &[u32],
    rng: &mut CausalRng,
    out: &mut Vec<u32>,
) {
    out.extend(0..n as u32);
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by_key(|&i| cluster_ids[i]);
    let mut idx = 0usize;
    while idx < n {
        let g = cluster_ids[order[idx]];
        let start = idx;
        while idx < n && cluster_ids[order[idx]] == g {
            idx += 1;
        }
        // Fisher–Yates on out[order[start..idx]] positions — shuffle the member rows
        // among the slots that belong to this cluster in the original order.
        let members: Vec<usize> = order[start..idx].to_vec();
        let mut vals: Vec<u32> = members.iter().map(|&i| i as u32).collect();
        for i in (1..vals.len()).rev() {
            let j = unbiased_index(rng, i + 1);
            vals.swap(i, j);
        }
        for (k, &pos) in members.iter().enumerate() {
            out[pos] = vals[k];
        }
    }
}

/// Fill `out` with length-`n` Bayesian-bootstrap weights (normalized Exp(1) draws).
///
/// Weights sum to `n` (mean weight 1) so weighted estimators match unweighted
/// scale on the original sample size.
///
/// # Errors
///
/// Zero length, or a non-weight plan.
pub fn fill_resample_weights(
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
    out.clear();
    out.reserve(n);
    let mut sum = 0.0;
    for _ in 0..n {
        // Exp(1) via -ln(U); Dirichlet(1..1) = normalized exponentials.
        let u = rng.next_f64().max(f64::EPSILON);
        let e = -u.ln();
        out.push(e);
        sum += e;
    }
    if !(sum > 0.0) {
        return Err(DataError::InvalidArgument {
            message: "BayesianBootstrap weight sum non-positive".into(),
        });
    }
    let scale = n as f64 / sum;
    for w in out.iter_mut() {
        *w *= scale;
    }
    Ok(())
}

fn checked_batch_len(n: usize, n_replicates: usize) -> Result<usize, DataError> {
    if n == 0 || n_replicates == 0 {
        return Err(DataError::InvalidArgument {
            message: "batch resample needs n > 0 and n_replicates > 0".into(),
        });
    }
    n.checked_mul(n_replicates).ok_or_else(|| DataError::InvalidArgument {
        message: "batch resample dimensions overflow usize".into(),
    })
}

fn check_resampling_cancelled(ctx: &ExecutionContext) -> Result<(), DataError> {
    if ctx.cancellation.is_cancelled() {
        return Err(DataError::InvalidArgument { message: "resampling cancelled".into() });
    }
    Ok(())
}

fn check_allocation_len<T>(len: usize) -> Result<(), DataError> {
    if len.checked_mul(std::mem::size_of::<T>()).is_none_or(|bytes| bytes > isize::MAX as usize) {
        return Err(DataError::InvalidArgument {
            message: "resampling allocation size exceeds addressable capacity".into(),
        });
    }
    Ok(())
}

/// Replicate-major row indexes with offsets for variable-sized replicates.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct RaggedResampleIndexBatch {
    /// Concatenated row indexes for all replicates.
    pub indexes: Vec<u32>,
    /// Replicate r occupies `indexes[offsets[r]..offsets[r + 1]]`.
    /// Contains `n_replicates + 1` entries, beginning with zero.
    pub offsets: Vec<usize>,
}

/// Generate index replicates, including unequal-sized whole-cluster bootstrap draws.
///
/// Uses the same independent replicate RNG streams as [`fill_resample_index_batch`].
/// This allocation-returning interface retains variable row counts without padding,
/// truncation, or conditioning the bootstrap on a fixed total number of rows.
///
/// # Errors
///
/// Invalid/overflowing counts, cancellation, allocation failure, missing cluster ids,
/// or a weight-only plan.
pub fn resample_index_batch_ragged(
    plan: ResamplingPlan,
    n: usize,
    n_replicates: usize,
    cluster_ids: Option<&[u32]>,
    ctx: &ExecutionContext,
    stream_base: u64,
) -> Result<RaggedResampleIndexBatch, DataError> {
    check_resampling_cancelled(ctx)?;
    let nominal_len = checked_batch_len(n, n_replicates)?;
    check_allocation_len::<u32>(nominal_len)?;
    let offset_len = n_replicates.checked_add(1).ok_or_else(|| DataError::InvalidArgument {
        message: "ragged resample offset count overflows usize".into(),
    })?;
    check_allocation_len::<usize>(offset_len)?;
    let mut indexes = Vec::new();
    let mut offsets = vec![0];
    let mut scratch = Vec::new();
    scratch.try_reserve(n).map_err(|error| DataError::InvalidArgument {
        message: format!("resampling scratch allocation failed: {error}"),
    })?;
    for replicate in 0..n_replicates {
        check_resampling_cancelled(ctx)?;
        let mut rng = ctx.rng.stream(stream_base ^ replicate as u64);
        fill_resample_indexes_grouped(plan, n, cluster_ids, &mut rng, &mut scratch)?;
        indexes.try_reserve(scratch.len()).map_err(|error| DataError::InvalidArgument {
            message: format!("ragged resampling allocation failed: {error}"),
        })?;
        offsets.try_reserve(1).map_err(|error| DataError::InvalidArgument {
            message: format!("ragged offset allocation failed: {error}"),
        })?;
        indexes.extend_from_slice(&scratch);
        offsets.push(indexes.len());
    }
    Ok(RaggedResampleIndexBatch { indexes, offsets })
}

/// Fill `out` (`len = n * n_replicates`, replicate-major) with index plans under one
/// [`ExecutionContext`].
///
/// Each replicate uses `ctx.rng.stream(stream_base ^ replicate_id)` so results are
/// independent of scheduling order under [`antecedent_core::Determinism::Strict`].
/// When `max_threads > 1` and `n_replicates ≥ 2`, fills run in a bounded
/// `std::thread::scope` pool.
///
/// # Errors
///
/// Shape / plan mismatches as in [`fill_resample_indexes_grouped`], dimension overflow,
/// cancellation, or `out` length mismatch.
/// Unequal cluster sizes cannot fit this fixed-width representation; use
/// [`resample_index_batch_ragged`] for that case.
pub fn fill_resample_index_batch(
    plan: ResamplingPlan,
    n: usize,
    n_replicates: usize,
    cluster_ids: Option<&[u32]>,
    ctx: &ExecutionContext,
    stream_base: u64,
    out: &mut [u32],
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
                message: "unequal cluster sizes require resample_index_batch_ragged; fixed-width output cannot represent the draws".into(),
            });
        }
    }
    let threads = ctx.parallelism.max_threads.get().max(1) as usize;
    if threads == 1 || n_replicates < 2 {
        let mut scratch = Vec::with_capacity(n);
        for r in 0..n_replicates {
            let mut rng = ctx.rng.stream(stream_base ^ r as u64);
            fill_resample_indexes_grouped(plan, n, cluster_ids, &mut rng, &mut scratch)?;
            out[r * n..(r + 1) * n].copy_from_slice(&scratch);
        }
        return Ok(());
    }
    let chunk = n_replicates.div_ceil(threads);
    let cluster_owned: Option<Vec<u32>> = cluster_ids.map(<[u32]>::to_vec);
    let err = std::sync::Mutex::new(None::<DataError>);
    std::thread::scope(|scope| {
        let mut rest = out;
        let mut start = 0usize;
        while start < n_replicates {
            let end = (start + chunk).min(n_replicates);
            let (this, next) = rest.split_at_mut((end - start) * n);
            rest = next;
            let cluster_ref = cluster_owned.as_deref();
            let err_slot = &err;
            let rng_factory = &ctx.rng;
            scope.spawn(move || {
                let mut scratch = Vec::with_capacity(n);
                for (local, r) in (start..end).enumerate() {
                    let mut rng = rng_factory.stream(stream_base ^ r as u64);
                    if let Err(e) =
                        fill_resample_indexes_grouped(plan, n, cluster_ref, &mut rng, &mut scratch)
                    {
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

/// Batch Bayesian-bootstrap weights under one [`ExecutionContext`] (replicate-major).
///
/// # Errors
///
/// Non-weight plan, zero/overflowing sizes, cancellation, or `out` length mismatch.
pub fn fill_resample_weight_batch(
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
    let threads = ctx.parallelism.max_threads.get().max(1) as usize;
    if threads == 1 || n_replicates < 2 {
        let mut scratch = Vec::with_capacity(n);
        for r in 0..n_replicates {
            let mut rng = ctx.rng.stream(stream_base ^ r as u64);
            fill_resample_weights(plan, n, &mut rng, &mut scratch)?;
            out[r * n..(r + 1) * n].copy_from_slice(&scratch);
        }
        return Ok(());
    }
    let chunk = n_replicates.div_ceil(threads);
    let err = std::sync::Mutex::new(None::<DataError>);
    std::thread::scope(|scope| {
        let mut rest = out;
        let mut start = 0usize;
        while start < n_replicates {
            let end = (start + chunk).min(n_replicates);
            let (this, next) = rest.split_at_mut((end - start) * n);
            rest = next;
            let err_slot = &err;
            let rng_factory = &ctx.rng;
            scope.spawn(move || {
                let mut scratch = Vec::with_capacity(n);
                for (local, r) in (start..end).enumerate() {
                    let mut rng = rng_factory.stream(stream_base ^ r as u64);
                    if let Err(e) = fill_resample_weights(plan, n, &mut rng, &mut scratch) {
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

/// Apply a resampling plan to produce a new float64 time series.
///
/// Index plans gather rows. [`ResamplingPlan::BayesianBootstrap`] keeps row
/// order and replaces storage weights with a fresh weight plan (multiplied by
/// any existing weights). Clustered plans require [`resample_timeseries_grouped`].
///
/// # Errors
///
/// Non-float columns or construction failures.
pub fn resample_timeseries(
    data: &TimeSeriesData,
    plan: ResamplingPlan,
    rng: &mut CausalRng,
    index_scratch: &mut Vec<u32>,
) -> Result<TimeSeriesData, DataError> {
    if plan.needs_clusters() {
        return Err(DataError::InvalidArgument {
            message: "clustered plan requires resample_timeseries_grouped".into(),
        });
    }
    resample_timeseries_grouped(data, plan, None, rng, index_scratch)
}

/// Like [`resample_timeseries`], with optional cluster labels for clustered plans.
///
/// # Errors
///
/// Non-float columns, clustered plan without ids, or construction failures.
pub fn resample_timeseries_grouped(
    data: &TimeSeriesData,
    plan: ResamplingPlan,
    cluster_ids: Option<&[u32]>,
    rng: &mut CausalRng,
    index_scratch: &mut Vec<u32>,
) -> Result<TimeSeriesData, DataError> {
    let n = data.row_count();
    if plan.is_weight_plan() {
        let mut weights = Vec::new();
        fill_resample_weights(plan, n, rng, &mut weights)?;
        return apply_weight_plan(data, &weights);
    }
    fill_resample_indexes_grouped(plan, n, cluster_ids, rng, index_scratch)?;
    apply_row_map(data, index_scratch)
}

fn apply_weight_plan(data: &TimeSeriesData, weights: &[f64]) -> Result<TimeSeriesData, DataError> {
    let n = data.row_count();
    if weights.len() != n {
        return Err(DataError::LengthMismatch {
            expected: n,
            actual: weights.len(),
            context: "BayesianBootstrap weights",
        });
    }
    let schema = data.schema().clone();
    let mut cols = Vec::with_capacity(schema.len());
    for v in schema.variables() {
        let ColumnView::Float64(src) = data.column(v.id)? else {
            return Err(DataError::TypeMismatch { id: v.id, expected: "float64" });
        };
        cols.push(OwnedColumn::Float64(Float64Column::new(
            v.id,
            src.values.clone(),
            src.validity.clone(),
        )?));
    }
    let analysis_mask = data.storage().analysis_mask().cloned();
    let combined: Arc<[f64]> = match data.storage().weights() {
        Some(existing) => {
            Arc::from(existing.iter().zip(weights.iter()).map(|(a, b)| a * b).collect::<Vec<_>>())
        }
        None => Arc::from(weights.to_vec()),
    };
    let storage = OwnedColumnarStorage::try_new(schema, cols, analysis_mask, Some(combined))?;
    TimeSeriesData::try_new(storage, data.time_index().clone())
}

fn apply_row_map(data: &TimeSeriesData, row_map: &[u32]) -> Result<TimeSeriesData, DataError> {
    let n = row_map.len();
    let schema = data.schema().clone();
    let mut cols = Vec::with_capacity(schema.len());
    for v in schema.variables() {
        let ColumnView::Float64(src) = data.column(v.id)? else {
            return Err(DataError::TypeMismatch { id: v.id, expected: "float64" });
        };
        let values: Vec<f64> = row_map.iter().map(|&r| src.values[r as usize]).collect();
        // A replicate slot inherits the source row's validity: values under
        // null slots must not resurface as valid observations.
        let validity = src.validity.gather(row_map)?;
        cols.push(OwnedColumn::Float64(Float64Column::new(v.id, Arc::from(values), validity)?));
    }
    // Analysis mask and weights follow the same row map as the values.
    let analysis_mask = data.storage().analysis_mask().map(|m| m.gather(row_map)).transpose()?;
    let weights = data
        .storage()
        .weights()
        .map(|w| Arc::from(row_map.iter().map(|&r| w[r as usize]).collect::<Vec<f64>>()));
    let storage = OwnedColumnarStorage::try_new(schema, cols, analysis_mask, weights)?;
    let mut time_index = data.time_index().clone();
    time_index.length = n;
    TimeSeriesData::try_new(storage, time_index)
}

#[cfg(test)]
#[allow(clippy::cast_precision_loss)]
mod tests {
    use antecedent_core::{
        CausalSchemaBuilder, MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
    };

    use super::*;
    use crate::column::ValidityBitmap;
    use crate::temporal::{SamplingRegularity, TimeIndex};
    use crate::testing::float_series;

    /// One-variable series of length 4 with row 2 invalid, a mask hiding row 1,
    /// and per-row weights.
    fn series_with_missing() -> TimeSeriesData {
        let mut b = CausalSchemaBuilder::new();
        b.add_variable(
            "v0".to_owned(),
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
        let schema = b.build().unwrap();
        // Row 2 invalid (LSB-first bits: rows 0,1,3 set).
        let validity = ValidityBitmap::from_bytes(vec![0b1011u8], 4).unwrap();
        let col = OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(vec![10.0, 11.0, 0.0, 13.0]),
                validity,
            )
            .unwrap(),
        );
        // Analysis mask hides row 1 (LSB-first bits: rows 0,2,3 set).
        let mask = ValidityBitmap::from_bytes(vec![0b1101u8], 4).unwrap();
        let weights: Arc<[f64]> = Arc::from(vec![1.0, 2.0, 3.0, 4.0]);
        let storage =
            OwnedColumnarStorage::try_new(schema, vec![col], Some(mask), Some(weights)).unwrap();
        TimeSeriesData::try_new(
            storage,
            TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: 4 },
        )
        .unwrap()
    }

    #[test]
    fn row_map_gathers_validity_mask_and_weights() {
        let data = series_with_missing();
        let out = apply_row_map(&data, &[2, 0, 2, 1]).unwrap();
        let ColumnView::Float64(col) = out.column(VariableId::from_raw(0)).unwrap() else {
            panic!("expected float64 column");
        };
        // Replicate slots sourced from row 2 must stay invalid.
        assert!(!col.validity.is_valid(0));
        assert!(col.validity.is_valid(1));
        assert!(!col.validity.is_valid(2));
        assert!(col.validity.is_valid(3));
        // No invalid source value appears as a valid replicate value.
        for i in 0..4 {
            if col.validity.is_valid(i) {
                assert!(
                    (col.values[i] - 10.0).abs() < 1e-12 || (col.values[i] - 11.0).abs() < 1e-12
                );
            }
        }
        // Analysis mask follows the row map (source row 1 was hidden).
        let mask = out.storage().analysis_mask().unwrap();
        assert!(mask.is_valid(0));
        assert!(mask.is_valid(1));
        assert!(mask.is_valid(2));
        assert!(!mask.is_valid(3));
        // Weights follow the row map.
        assert_eq!(out.storage().weights().unwrap(), &[3.0, 1.0, 3.0, 2.0]);
    }

    #[test]
    fn bayesian_bootstrap_weights_sum_to_n() {
        let mut rng = CausalRng::from_seed(99);
        let mut w = Vec::new();
        fill_resample_weights(ResamplingPlan::BayesianBootstrap, 50, &mut rng, &mut w).unwrap();
        assert_eq!(w.len(), 50);
        let sum: f64 = w.iter().sum();
        assert!((sum - 50.0).abs() < 1e-9);
        assert!(w.iter().all(|&x| x > 0.0));
    }

    #[test]
    fn bayesian_bootstrap_rejects_index_fill() {
        let mut rng = CausalRng::from_seed(1);
        let mut idx = Vec::new();
        assert!(
            fill_resample_indexes(ResamplingPlan::BayesianBootstrap, 10, &mut rng, &mut idx)
                .is_err()
        );
    }

    #[test]
    fn moving_block_preserves_length() {
        let data = float_series(100, 1);
        let mut rng = CausalRng::from_seed(1);
        let mut idx = Vec::new();
        let out = resample_timeseries(
            &data,
            ResamplingPlan::MovingBlock { length: 10 },
            &mut rng,
            &mut idx,
        )
        .unwrap();
        assert_eq!(out.row_count(), 100);
        assert_eq!(idx.len(), 100);
    }

    #[test]
    fn permutation_full_is_bijection() {
        let mut rng = CausalRng::from_seed(3);
        let mut idx = Vec::new();
        fill_resample_indexes(
            ResamplingPlan::Permutation(PermutationScheme::Full),
            20,
            &mut rng,
            &mut idx,
        )
        .unwrap();
        let mut sorted = idx.clone();
        sorted.sort_unstable();
        assert_eq!(sorted, (0..20u32).collect::<Vec<_>>());
    }

    #[test]
    fn cluster_bootstrap_keeps_members_contiguous_from_same_cluster() {
        let n = 12usize;
        let clusters: Vec<u32> = (0..n as u32).map(|i| i / 3).collect();
        let mut rng = CausalRng::from_seed(5);
        let mut idx = Vec::new();
        fill_resample_indexes_grouped(
            ResamplingPlan::ClusterBootstrap { cluster: VariableId::from_raw(0) },
            n,
            Some(&clusters),
            &mut rng,
            &mut idx,
        )
        .unwrap();
        assert_eq!(idx.len(), n);
        // Every index must come from some original cluster; consecutive triples
        // from the same draw share a cluster id when length allows.
        for &i in &idx {
            assert!((i as usize) < n);
        }
    }

    #[test]
    fn stationary_block_preserves_length() {
        let mut rng = CausalRng::from_seed(2);
        let mut idx = Vec::new();
        fill_resample_indexes(
            ResamplingPlan::StationaryBlock { expected_length: 5.0 },
            80,
            &mut rng,
            &mut idx,
        )
        .unwrap();
        assert_eq!(idx.len(), 80);
    }

    #[test]
    fn index_batch_shape_and_strict_determinism() {
        use antecedent_core::ExecutionContext;
        let ctx = ExecutionContext::for_tests(42);
        let n = 20usize;
        let b = 8usize;
        let mut out_a = vec![0u32; n * b];
        let mut out_b = vec![0u32; n * b];
        fill_resample_index_batch(
            ResamplingPlan::IidBootstrap,
            n,
            b,
            None,
            &ctx,
            0xDEAD_u64,
            &mut out_a,
        )
        .unwrap();
        fill_resample_index_batch(
            ResamplingPlan::IidBootstrap,
            n,
            b,
            None,
            &ctx,
            0xDEAD_u64,
            &mut out_b,
        )
        .unwrap();
        assert_eq!(out_a, out_b);
        assert!(out_a.iter().all(|&i| (i as usize) < n));
    }

    #[test]
    fn weight_batch_rows_sum_to_n() {
        use antecedent_core::ExecutionContext;
        let ctx = ExecutionContext::for_tests(7);
        let n = 15usize;
        let b = 4usize;
        let mut out = vec![0.0; n * b];
        fill_resample_weight_batch(
            ResamplingPlan::BayesianBootstrap,
            n,
            b,
            &ctx,
            0xBEEF_u64,
            &mut out,
        )
        .unwrap();
        for r in 0..b {
            let sum: f64 = out[r * n..(r + 1) * n].iter().sum();
            assert!((sum - n as f64).abs() < 1e-9);
        }
    }
    #[test]
    fn unequal_cluster_bootstrap_draws_g_whole_clusters_without_conditioning_on_rows() {
        let plan = ResamplingPlan::ClusterBootstrap { cluster: VariableId::from_raw(0) };
        let ids = [0, 1, 1, 1];
        let context = ExecutionContext::for_tests(42);
        let batch = resample_index_batch_ragged(plan, 4, 4000, Some(&ids), &context, 11).unwrap();
        let mut lengths = [0usize; 3];
        for bounds in batch.offsets.windows(2) {
            let rows = &batch.indexes[bounds[0]..bounds[1]];
            lengths[(rows.len() - 2) / 2] += 1;
            let mut cursor = 0;
            for _ in 0..2 {
                if rows[cursor] == 0 {
                    cursor += 1;
                } else {
                    assert_eq!(&rows[cursor..cursor + 3], &[1, 2, 3]);
                    cursor += 3;
                }
            }
            assert_eq!(cursor, rows.len(), "each replicate must contain exactly G=2 draws");
        }
        // Two independent cluster draws give row-count probabilities 1/4, 1/2, 1/4.
        for (count, target) in lengths.into_iter().zip([0.25, 0.5, 0.25]) {
            assert!((count as f64 / 4000.0 - target).abs() < 0.04);
        }
        let mut fixed = vec![0; 8];
        assert!(
            fill_resample_index_batch(plan, 4, 2, Some(&ids), &context, 11, &mut fixed).is_err()
        );
    }

    #[test]
    fn ragged_and_fixed_batches_agree_when_clusters_have_equal_sizes() {
        let plan = ResamplingPlan::ClusterBootstrap { cluster: VariableId::from_raw(0) };
        let ids = [0, 0, 1, 1];
        let context = ExecutionContext::for_tests(7);
        let ragged = resample_index_batch_ragged(plan, 4, 5, Some(&ids), &context, 13).unwrap();
        let mut fixed = vec![0; 20];
        fill_resample_index_batch(plan, 4, 5, Some(&ids), &context, 13, &mut fixed).unwrap();
        assert_eq!(ragged.indexes, fixed);
        assert_eq!(ragged.offsets, [0, 4, 8, 12, 16, 20]);
    }

    #[test]
    fn unequal_cluster_timeseries_updates_length_and_gathers_metadata() {
        let data = series_with_missing();
        let plan = ResamplingPlan::ClusterBootstrap { cluster: VariableId::from_raw(0) };
        let ids = [0, 1, 1, 1];
        let mut rng = CausalRng::from_seed(9);
        let mut rows = Vec::new();
        let mut saw_variable_length = false;
        for _ in 0..20 {
            let replicate =
                resample_timeseries_grouped(&data, plan, Some(&ids), &mut rng, &mut rows).unwrap();
            saw_variable_length |= rows.len() != data.row_count();
            assert_eq!(replicate.row_count(), rows.len());
            assert_eq!(replicate.time_index().length, rows.len());
            let column = replicate.column(VariableId::from_raw(0)).unwrap();
            let source = data.column(VariableId::from_raw(0)).unwrap();
            for (row, &origin) in rows.iter().enumerate() {
                assert_eq!(
                    column.validity().is_valid(row),
                    source.validity().is_valid(origin as usize)
                );
                assert_eq!(
                    replicate.storage().analysis_mask().unwrap().is_valid(row),
                    data.storage().analysis_mask().unwrap().is_valid(origin as usize)
                );
                assert!(
                    (replicate.storage().weights().unwrap()[row]
                        - data.storage().weights().unwrap()[origin as usize])
                        .abs()
                        < 1e-12
                );
            }
        }
        assert!(saw_variable_length);
    }
    #[test]
    fn batch_dimension_overflow_is_an_error_before_allocation() {
        let context = ExecutionContext::for_tests(1);
        let index_error = fill_resample_index_batch(
            ResamplingPlan::IidBootstrap,
            usize::MAX,
            2,
            None,
            &context,
            0,
            &mut [],
        )
        .unwrap_err();
        let weight_error = fill_resample_weight_batch(
            ResamplingPlan::BayesianBootstrap,
            usize::MAX,
            2,
            &context,
            0,
            &mut [],
        )
        .unwrap_err();
        let ragged_error = resample_index_batch_ragged(
            ResamplingPlan::IidBootstrap,
            usize::MAX,
            2,
            None,
            &context,
            0,
        )
        .unwrap_err();
        for error in [index_error, weight_error, ragged_error] {
            assert!(error.to_string().contains("overflow"));
        }
        // Element count fits usize but its byte capacity does not fit a Vec.
        assert!(
            resample_index_batch_ragged(
                ResamplingPlan::IidBootstrap,
                usize::MAX,
                1,
                None,
                &context,
                0,
            )
            .unwrap_err()
            .to_string()
            .contains("capacity")
        );
        assert!(
            resample_index_batch_ragged(
                ResamplingPlan::IidBootstrap,
                1,
                (isize::MAX as usize / std::mem::size_of::<usize>()) + 1,
                None,
                &context,
                0,
            )
            .unwrap_err()
            .to_string()
            .contains("capacity")
        );
    }

    #[test]
    fn cancelled_batches_return_before_mutating_output() {
        let context = ExecutionContext::for_tests(1);
        context.cancellation.cancel();
        let mut indexes = [7];
        let mut weights = [0.7];
        assert!(
            fill_resample_index_batch(
                ResamplingPlan::IidBootstrap,
                1,
                1,
                None,
                &context,
                0,
                &mut indexes,
            )
            .unwrap_err()
            .to_string()
            .contains("cancelled")
        );
        assert!(
            fill_resample_weight_batch(
                ResamplingPlan::BayesianBootstrap,
                1,
                1,
                &context,
                0,
                &mut weights,
            )
            .unwrap_err()
            .to_string()
            .contains("cancelled")
        );
        assert!(
            resample_index_batch_ragged(
                ResamplingPlan::IidBootstrap,
                usize::MAX,
                1,
                None,
                &context,
                0,
            )
            .unwrap_err()
            .to_string()
            .contains("cancelled")
        );
        assert_eq!(indexes, [7]);
        assert!((weights[0] - 0.7).abs() < 1e-12);
    }
}
