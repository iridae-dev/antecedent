//! Temporal bootstrap / resampling index plans.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![cfg_attr(
    test,
    allow(
        clippy::cast_possible_truncation,
        reason = "test fixtures compare exact constants and index with small literals"
    )
)]

use std::sync::Arc;

use antecedent_core::{
    CausalRng, DeclaredSamplingDesign, ExecutionContext, ResamplingRule, SamplingDesignDeclaration,
    SamplingDesignError, SplittingRule, StreamDomain, VariableId,
};
use antecedent_kernels::unbiased_index;

use crate::buffer::F64Buffer;
use crate::column::{ColumnView, Float64Column, OwnedColumn};
use crate::dataset::TimeSeriesData;
use crate::error::DataError;
use crate::global_index::GlobalId;
use crate::split::TemporalRandomPolicy;
use crate::storage::OwnedColumnarStorage;
use crate::table::TableView;

fn row_id(index: usize) -> Result<GlobalId, DataError> {
    u64::try_from(index).map(GlobalId::new).map_err(|_| DataError::InvalidArgument {
        message: "address the row by GlobalId and keep ChunkIndex inside one bounded chunk. \
                  A u32 backend feature flag is not that attempt"
            .into(),
    })
}

fn row_pos(id: GlobalId) -> Result<usize, DataError> {
    usize::try_from(id.get()).map_err(|_| DataError::InvalidArgument {
        message: "address the row by GlobalId and keep ChunkIndex inside one bounded chunk. \
                  A u32 backend feature flag is not that attempt"
            .into(),
    })
}

/// Null / permutation scheme for [`ResamplingPlan::Permutation`].
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
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
    /// A cluster drawn `k` times contributes its rows `k` times under one original
    /// label: the replicate is a row-index list, so a cluster-aware estimator (cluster-robust
    /// variance, cluster fixed effects) must treat the repeats as distinct clusters
    /// rather than merge them by label.
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
    out: &mut Vec<GlobalId>,
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
pub(crate) fn fill_resample_indexes_grouped(
    plan: ResamplingPlan,
    n: usize,
    cluster_ids: Option<&[u32]>,
    rng: &mut CausalRng,
    out: &mut Vec<GlobalId>,
) -> Result<(), DataError> {
    let clusters = prepare_clusters(plan, n, cluster_ids)?;
    fill_indexes_prepared(plan, n, clusters.as_ref(), rng, out)
}

/// Draw one replicate for a declared sampling design.
///
/// The design selects the draw. [`SamplingDesignDeclaration::time_column_present`]
/// is recorded and is not that selection. A trajectory design refuses a row-IID
/// draw. A panel draws whole clusters. A series draws a contiguous block on the
/// interior left after the embargo on each side.
///
/// # Errors
///
/// A row-IID attempt on a trajectory, an embargo on a design with no block
/// boundary, a panel without cluster labels, a block that does not fit after
/// the embargo, or a row id that does not fit in [`GlobalId`].
pub fn fill_declared_resample(
    declaration: SamplingDesignDeclaration,
    n: usize,
    cluster_ids: Option<&[u32]>,
    block_length: usize,
    embargo: u32,
    rng: &mut CausalRng,
    out: &mut Vec<GlobalId>,
) -> Result<(), DataError> {
    let design = design_independent_of_time_column(declaration);
    match design.rules().resampling() {
        ResamplingRule::RowSubjectIid | ResamplingRule::RowIid => Err(refuse_row_iid(design)),
        ResamplingRule::WholeCluster => {
            draw_whole_clusters(design, n, cluster_ids, embargo, rng, out)
        }
        ResamplingRule::ContiguousBlock => {
            draw_contiguous_block(design, n, block_length, embargo, rng, out)
        }
    }
}

fn design_independent_of_time_column(
    declaration: SamplingDesignDeclaration,
) -> DeclaredSamplingDesign {
    let with_column = SamplingDesignDeclaration::new(declaration.design(), true);
    let without_column = SamplingDesignDeclaration::new(declaration.design(), false);
    let selected = if declaration.time_column_present() { with_column } else { without_column };
    debug_assert_eq!(with_column.design(), without_column.design());
    selected.design()
}

fn refuse_row_iid(design: DeclaredSamplingDesign) -> DataError {
    match design.apply_resampling(ResamplingRule::RowIid) {
        Err(error) => declared_design_error(error),
        Ok(_) => DataError::InvalidArgument {
            message: "a trajectory design refuses a row-IID draw".into(),
        },
    }
}

fn draw_whole_clusters(
    design: DeclaredSamplingDesign,
    n: usize,
    cluster_ids: Option<&[u32]>,
    embargo: u32,
    rng: &mut CausalRng,
    out: &mut Vec<GlobalId>,
) -> Result<(), DataError> {
    design.apply_resampling(ResamplingRule::WholeCluster).map_err(declared_design_error)?;
    design
        .apply_splitting_with_embargo(SplittingRule::WholeCluster, embargo)
        .map_err(declared_design_error)?;
    if n == 0 {
        return Err(DataError::InvalidArgument { message: "resample needs n > 0".into() });
    }
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
    let clusters = ClusterIndex::build(ids)?;
    out.clear();
    fill_cluster_bootstrap(&clusters, rng, out)
}

fn draw_contiguous_block(
    design: DeclaredSamplingDesign,
    n: usize,
    block_length: usize,
    embargo: u32,
    rng: &mut CausalRng,
    out: &mut Vec<GlobalId>,
) -> Result<(), DataError> {
    design.apply_resampling(ResamplingRule::ContiguousBlock).map_err(declared_design_error)?;
    let split = design
        .apply_splitting_with_embargo(SplittingRule::ContiguousBlock, embargo)
        .map_err(declared_design_error)?;
    fill_embargoed_contiguous_block(n, block_length, split.embargo(), rng, out)
}

fn declared_design_error(error: SamplingDesignError) -> DataError {
    DataError::InvalidArgument { message: error.to_string() }
}

fn fill_embargoed_contiguous_block(
    n: usize,
    length: usize,
    embargo: u32,
    rng: &mut CausalRng,
    out: &mut Vec<GlobalId>,
) -> Result<(), DataError> {
    if n == 0 {
        return Err(DataError::InvalidArgument { message: "resample needs n > 0".into() });
    }
    let embargo = usize::try_from(embargo).map_err(|_| DataError::InvalidArgument {
        message: "embargo does not fit in usize".into(),
    })?;
    let margin = embargo.checked_mul(2).ok_or_else(|| DataError::InvalidArgument {
        message: "embargo does not fit in the series".into(),
    })?;
    if margin >= n {
        return Err(DataError::InvalidArgument {
            message: "embargo leaves no observation to draw".into(),
        });
    }
    let span = n - margin;
    if length == 0 || length > span {
        return Err(DataError::InvalidArgument {
            message: format!(
                "block length {length} exceeds the series length {span} left after the embargo"
            ),
        });
    }
    let mut indexes = vec![0usize; span];
    fill_block_slice(span, length, false, rng, &mut indexes, Ok)?;
    out.clear();
    out.reserve(span);
    for index in indexes {
        out.push(row_id(embargo + index)?);
    }
    Ok(())
}

/// Validate the plan / shape and, for clustered plans, build the replicate-invariant
/// cluster index once.
fn prepare_clusters(
    plan: ResamplingPlan,
    n: usize,
    cluster_ids: Option<&[u32]>,
) -> Result<Option<ClusterIndex>, DataError> {
    if n == 0 {
        return Err(DataError::InvalidArgument { message: "resample needs n > 0".into() });
    }
    if plan.is_weight_plan() {
        return Err(DataError::InvalidArgument {
            message: "BayesianBootstrap yields weights; use fill_resample_weights".into(),
        });
    }
    if !plan.needs_clusters() {
        return Ok(None);
    }
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
    Ok(Some(ClusterIndex::build(ids)?))
}

/// Cluster membership in CSR form: rows sorted (stably) by cluster label, with
/// `offsets[c]..offsets[c + 1]` delimiting cluster `c` (labels ascending).
struct ClusterIndex {
    members: Vec<GlobalId>,
    offsets: Vec<usize>,
}

impl ClusterIndex {
    fn build(cluster_ids: &[u32]) -> Result<Self, DataError> {
        let n = cluster_ids.len();
        let mut keyed = Vec::with_capacity(n);
        for (index, &label) in cluster_ids.iter().enumerate() {
            keyed.push((label, row_id(index)?));
        }
        keyed.sort_by_key(|(label, _)| *label);
        let mut offsets = vec![0usize];
        for k in 1..n {
            if keyed[k].0 != keyed[k - 1].0 {
                offsets.push(k);
            }
        }
        offsets.push(n);
        let members = keyed.into_iter().map(|(_, id)| id).collect();
        Ok(Self { members, offsets })
    }

    fn n_clusters(&self) -> usize {
        self.offsets.len() - 1
    }

    fn cluster(&self, c: usize) -> &[GlobalId] {
        &self.members[self.offsets[c]..self.offsets[c + 1]]
    }
}

fn fill_indexes_prepared(
    plan: ResamplingPlan,
    n: usize,
    clusters: Option<&ClusterIndex>,
    rng: &mut CausalRng,
    out: &mut Vec<GlobalId>,
) -> Result<(), DataError> {
    out.clear();
    out.reserve(n);
    match plan {
        ResamplingPlan::IidBootstrap => {
            for _ in 0..n {
                out.push(row_id(unbiased_index(rng, n))?);
            }
        }
        ResamplingPlan::BayesianBootstrap => {
            return Err(DataError::InvalidArgument {
                message: "BayesianBootstrap yields weights; use fill_resample_weights".into(),
            });
        }
        ResamplingPlan::MovingBlock { length } | ResamplingPlan::CircularBlock { length } => {
            let mut indexes = vec![0usize; n];
            fill_block_slice(
                n,
                length,
                matches!(plan, ResamplingPlan::CircularBlock { .. }),
                rng,
                &mut indexes,
                Ok,
            )?;
            for index in indexes {
                out.push(row_id(index)?);
            }
        }
        ResamplingPlan::StationaryBlock { expected_length } => {
            fill_stationary(n, expected_length, rng, out)?;
        }
        ResamplingPlan::ClusterBootstrap { .. } => {
            let Some(clusters) = clusters else {
                return Err(DataError::InvalidArgument {
                    message: "clustered resampling requires cluster_ids".into(),
                });
            };
            fill_cluster_bootstrap(clusters, rng, out)?;
        }
        ResamplingPlan::Permutation(PermutationScheme::Full) => {
            for index in 0..n {
                out.push(row_id(index)?);
            }
            // Fisher–Yates
            for i in (1..n).rev() {
                let j = unbiased_index(rng, i + 1);
                out.swap(i, j);
            }
        }
        ResamplingPlan::Permutation(PermutationScheme::WithinCluster) => {
            let Some(clusters) = clusters else {
                return Err(DataError::InvalidArgument {
                    message: "clustered resampling requires cluster_ids".into(),
                });
            };
            fill_within_cluster_permutation(n, clusters, rng, out)?;
        }
    }
    Ok(())
}

/// Fill `out` (`len = n`) with circular-block row indexes.
///
/// Same RNG stream and index law as [`fill_resample_indexes`] with
/// [`ResamplingPlan::CircularBlock`].
///
/// # Errors
///
/// Zero series length, zero block length, `length > n`, or `out.len() != n`.
pub fn fill_circular_block_indexes(
    n: usize,
    length: usize,
    rng: &mut CausalRng,
    out: &mut [usize],
) -> Result<(), DataError> {
    if n == 0 {
        return Err(DataError::InvalidArgument { message: "resample needs n > 0".into() });
    }
    if out.len() != n {
        return Err(DataError::LengthMismatch {
            expected: n,
            actual: out.len(),
            context: "fill_circular_block_indexes out",
        });
    }
    fill_block_slice(n, length, true, rng, out, Ok)
}

fn fill_block_slice<T: Copy>(
    n: usize,
    length: usize,
    circular: bool,
    rng: &mut CausalRng,
    out: &mut [T],
    cast: impl Fn(usize) -> Result<T, DataError>,
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
    let mut written = 0;
    while written < n {
        let start = unbiased_index(rng, n_starts);
        let take = length.min(n - written);
        for k in 0..take {
            let idx = if circular { (start + k) % n } else { start + k };
            out[written + k] = cast(idx)?;
        }
        written += take;
    }
    Ok(())
}

fn fill_stationary(
    n: usize,
    expected_length: f64,
    rng: &mut CausalRng,
    out: &mut Vec<GlobalId>,
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
            out.push(row_id((start + k) % n)?);
        }
    }
    out.truncate(n);
    Ok(())
}

fn fill_cluster_bootstrap(
    clusters: &ClusterIndex,
    rng: &mut CausalRng,
    out: &mut Vec<GlobalId>,
) -> Result<(), DataError> {
    let g = clusters.n_clusters();
    if g == 0 {
        return Err(DataError::InvalidArgument { message: "no clusters".into() });
    }
    // G IID draws from the empirical cluster distribution. Conditioning on
    // exactly n rows changes that distribution when clusters have unequal sizes.
    for _ in 0..g {
        let c = unbiased_index(rng, g);
        out.extend_from_slice(clusters.cluster(c));
    }
    Ok(())
}

fn fill_within_cluster_permutation(
    n: usize,
    clusters: &ClusterIndex,
    rng: &mut CausalRng,
    out: &mut Vec<GlobalId>,
) -> Result<(), DataError> {
    for index in 0..n {
        out.push(row_id(index)?);
    }
    // Shuffle each cluster's member rows among the slots that cluster occupies.
    let mut vals: Vec<GlobalId> = Vec::new();
    for c in 0..clusters.n_clusters() {
        let members = clusters.cluster(c);
        vals.clear();
        vals.extend_from_slice(members);
        for i in (1..vals.len()).rev() {
            let j = unbiased_index(rng, i + 1);
            vals.swap(i, j);
        }
        for (k, &pos) in members.iter().enumerate() {
            out[row_pos(pos)?] = vals[k];
        }
    }
    Ok(())
}

/// Fill `out` with length-`n` Bayesian-bootstrap weights (normalized Exp(1) draws).
///
/// Weights sum to `n` (mean weight 1) so weighted estimators match unweighted
/// scale on the original sample size.
///
/// # Errors
///
/// Zero length, or a non-weight plan.
#[allow(
    clippy::neg_cmp_op_on_partial_ord,
    reason = "negated comparison is deliberate: a NaN input must take the rejecting branch, which the un-negated form would let through"
)]
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

fn resample_rng(ctx: &ExecutionContext, stream_base: u64, replicate: u64) -> CausalRng {
    ctx.rng.stream_for(StreamDomain::Resample, stream_base ^ replicate)
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
        return Err(DataError::Cancelled);
    }
    Ok(())
}

/// Fill `out` with one replicate's row indexes, reusing the buffer.
///
/// The draw is `ctx.rng.stream_for(StreamDomain::Resample, stream_base ^ replicate)`, the same
/// stream as row `replicate` of [`fill_resample_index_batch`]. `out` is cleared first. Index
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
/// The draw is `ctx.rng.stream_for(StreamDomain::Resample, stream_base ^ replicate)`, the same
/// stream as one row of a replicate-major Bayesian-bootstrap weight batch. Weights sum to `n`.
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
/// Each replicate uses `ctx.rng.stream_for(StreamDomain::Resample, stream_base ^ replicate_id)`
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
/// the admission pool. A product that overflows, a byte count above the hard
/// limit, or a pool that cannot fund that scratch returns before `fill` runs
/// and before any scratch buffer is allocated. [`ExecutionContext::memory`] is
/// not decremented.
fn fill_replicate_rows<T: Copy + Send>(
    ctx: &ExecutionContext,
    n: usize,
    n_replicates: usize,
    out: &mut [T],
    fill: impl Fn(usize, &mut Vec<T>) -> Result<(), DataError> + Sync,
) -> Result<(), DataError> {
    let workers = resample_workers(ctx, n_replicates);
    admit_resample_scratch::<T>(ctx, n, workers)?;
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

/// Workers for one batch: the context's thread count, the replicate count, and
/// the admission pool's free slots, whichever is smallest. One stays on the caller.
fn resample_workers(ctx: &ExecutionContext, n_replicates: usize) -> usize {
    if n_replicates < 2 {
        return 1;
    }
    let requested = usize::try_from(ctx.parallelism.max_threads.get()).unwrap_or(usize::MAX);
    let requested = requested.clamp(1, n_replicates);
    if requested <= 1 {
        return 1;
    }
    let budget = usize::try_from(ctx.admission().thread_budget()).unwrap_or(usize::MAX);
    let free = usize::try_from(ctx.admission().available().threads()).unwrap_or(usize::MAX);
    let cap = budget.min(free);
    if cap <= 1 { 1 } else { requested.min(cap) }
}

/// Bytes of `workers` length-`n` scratch buffers of `T`.
///
/// # Errors
///
/// The product does not fit in `u64`, the hard byte limit is lower, or the
/// admission pool's free bytes cannot fund it. Nothing is allocated.
fn admit_resample_scratch<T>(
    ctx: &ExecutionContext,
    n: usize,
    workers: usize,
) -> Result<(), DataError> {
    let width = size_of::<T>();
    let bytes = n
        .checked_mul(workers)
        .and_then(|slots| slots.checked_mul(width))
        .and_then(|bytes| u64::try_from(bytes).ok())
        .ok_or_else(|| DataError::InvalidArgument {
            message: "resample scratch size overflow".into(),
        })?;
    if let Some(limit) = ctx.memory.hard_limit_bytes {
        if bytes > limit {
            return Err(DataError::InvalidArgument {
                message: "resample scratch exceeds the hard byte limit".into(),
            });
        }
    }
    if bytes > ctx.admission().available().bytes() {
        return Err(DataError::InvalidArgument {
            message: "admission pool cannot fund the resample scratch".into(),
        });
    }
    Ok(())
}

/// Smallest integer `k` with `k³ ≥ n` (`0` for `n = 0`), without floating-point rounding.
#[must_use]
fn integer_cube_root_ceil(n: usize) -> usize {
    if n <= 1 {
        return n;
    }
    let (mut low, mut high) = (1usize, n);
    while low < high {
        let middle = low + (high - low) / 2;
        if middle.saturating_mul(middle).saturating_mul(middle) >= n {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    low
}

/// Block length shared by every temporal circular-block bootstrap:
/// `max(structural_span, ⌈n^{1/3}⌉)`, capped at `n` (and at least 1).
///
/// `structural_span` is the number of consecutive time slices one estimating
/// row reads (the unfolded window, `history + horizon`, or `max_lag + 1` for a
/// lagged design), so a block never splits a single row's lag window. The
/// `⌈n^{1/3}⌉` floor is the MSE-rate block growth for the variance of a smooth
/// mean (Hall, Horowitz & Jing 1995); it is a rule of thumb, not a data-driven
/// choice, and it under-represents long-memory dependence at small `n`.
#[must_use]
pub fn circular_block_length(structural_span: usize, n: usize) -> usize {
    structural_span.max(integer_cube_root_ceil(n)).min(n).max(1)
}

/// Apply a resampling plan to produce a new float64 time series.
///
/// Index plans gather rows. [`ResamplingPlan::BayesianBootstrap`] keeps row
/// order and replaces storage weights with a fresh weight plan (multiplied by
/// any existing weights). Clustered plans require [`resample_timeseries_grouped`].
///
/// Serially dependent data must use a block plan: the IID row bootstrap and the
/// Bayesian bootstrap (IID Dirichlet weights) ignore autocorrelation and understate
/// uncertainty, so they are refused here. [`resample_timeseries_with_policy`] is the
/// explicit opt-in.
///
/// # Errors
///
/// Non-float columns, an IID plan on time-series data, or construction failures.
pub fn resample_timeseries(
    data: &TimeSeriesData,
    plan: ResamplingPlan,
    rng: &mut CausalRng,
    index_scratch: &mut Vec<GlobalId>,
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
pub(crate) fn resample_timeseries_grouped(
    data: &TimeSeriesData,
    plan: ResamplingPlan,
    cluster_ids: Option<&[u32]>,
    rng: &mut CausalRng,
    index_scratch: &mut Vec<GlobalId>,
) -> Result<TimeSeriesData, DataError> {
    resample_timeseries_with_policy(
        data,
        plan,
        cluster_ids,
        rng,
        index_scratch,
        TemporalRandomPolicy::Refuse,
    )
}

/// [`resample_timeseries_grouped`] with an explicit [`TemporalRandomPolicy`].
///
/// [`TemporalRandomPolicy::Allow`] admits [`ResamplingPlan::IidBootstrap`] and
/// [`ResamplingPlan::BayesianBootstrap`], whose replicates carry no serial dependence.
///
/// # Errors
///
/// As [`resample_timeseries_grouped`]; an IID plan under [`TemporalRandomPolicy::Refuse`].
pub(crate) fn resample_timeseries_with_policy(
    data: &TimeSeriesData,
    plan: ResamplingPlan,
    cluster_ids: Option<&[u32]>,
    rng: &mut CausalRng,
    index_scratch: &mut Vec<GlobalId>,
    policy: TemporalRandomPolicy,
) -> Result<TimeSeriesData, DataError> {
    if matches!(plan, ResamplingPlan::IidBootstrap | ResamplingPlan::BayesianBootstrap) {
        crate::split::ensure_random_allowed_on_temporal(policy).map_err(|_| {
            DataError::InvalidArgument {
                message: "IID bootstrap refused on time-series data (ignores serial dependence); use a block plan or pass TemporalRandomPolicy::Allow".into(),
            }
        })?;
    }
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

fn apply_row_map(data: &TimeSeriesData, row_map: &[GlobalId]) -> Result<TimeSeriesData, DataError> {
    let n = row_map.len();
    let positions = row_map.iter().copied().map(row_pos).collect::<Result<Vec<_>, _>>()?;
    let schema = data.schema().clone();
    let mut cols = Vec::with_capacity(schema.len());
    for v in schema.variables() {
        let ColumnView::Float64(src) = data.column(v.id)? else {
            return Err(DataError::TypeMismatch { id: v.id, expected: "float64" });
        };
        let values: Vec<f64> = positions.iter().map(|&r| src.values[r]).collect();
        // A replicate slot inherits the source row's validity: values under
        // null slots must not resurface as valid observations.
        let validity = src.validity.gather_rows(&positions)?;
        // Rows gathered from a NaN-free column are NaN-free: skip the rescan.
        let mut values = F64Buffer::owned(Arc::<[f64]>::from(values));
        if src.values.is_nan_free() {
            values.mark_nan_free();
        }
        cols.push(OwnedColumn::Float64(Float64Column::new(v.id, values, validity)?));
    }
    // Analysis mask and weights follow the same row map as the values.
    let analysis_mask =
        data.storage().analysis_mask().map(|m| m.gather_rows(&positions)).transpose()?;
    let weights = data
        .storage()
        .weights()
        .map(|w| Arc::from(positions.iter().map(|&r| w[r]).collect::<Vec<f64>>()));
    let storage = OwnedColumnarStorage::try_new(schema, cols, analysis_mask, weights)?;
    let mut time_index = data.time_index().clone();
    time_index.length = n;
    TimeSeriesData::try_new(storage, time_index)
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Barrier};

    use antecedent_core::execution::AdmissionPool;
    use antecedent_core::{
        CausalSchemaBuilder, ExecutionContext, MeasurementSpec, MemoryBudget, RoleHint,
        SmallRoleSet, ValueType, VariableId,
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
        let map = [GlobalId::new(2), GlobalId::new(0), GlobalId::new(2), GlobalId::new(1)];
        let out = apply_row_map(&data, &map).unwrap();
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
    fn iid_plans_are_refused_on_time_series_without_opt_in() {
        let data = float_series(30, 1);
        let mut idx = Vec::new();
        for plan in [ResamplingPlan::IidBootstrap, ResamplingPlan::BayesianBootstrap] {
            let mut rng = CausalRng::from_seed(3);
            assert!(resample_timeseries(&data, plan, &mut rng, &mut idx).is_err());
            let out = resample_timeseries_with_policy(
                &data,
                plan,
                None,
                &mut rng,
                &mut idx,
                TemporalRandomPolicy::Allow,
            )
            .unwrap();
            assert_eq!(out.row_count(), 30);
        }
    }

    #[test]
    fn cluster_index_bootstrap_emits_whole_clusters_and_permutation_stays_within() {
        // label 0 -> rows {1,4}, label 1 -> {3}, label 2 -> {0,2}.
        let ids = [2u32, 0, 2, 1, 0];
        let mut rng = CausalRng::from_seed(5);
        let mut out = Vec::new();
        let allowed: [Vec<GlobalId>; 3] = [
            vec![GlobalId::new(1), GlobalId::new(4)],
            vec![GlobalId::new(3)],
            vec![GlobalId::new(0), GlobalId::new(2)],
        ];
        for _ in 0..50 {
            let plan = ResamplingPlan::ClusterBootstrap { cluster: VariableId::from_raw(0) };
            fill_resample_indexes_grouped(plan, 5, Some(&ids), &mut rng, &mut out).unwrap();
            let mut rest = out.as_slice();
            let mut clusters = 0;
            while !rest.is_empty() {
                let hit = allowed.iter().find(|c| rest.starts_with(c)).expect("whole cluster");
                rest = &rest[hit.len()..];
                clusters += 1;
            }
            assert_eq!(clusters, 3);
            let plan = ResamplingPlan::Permutation(PermutationScheme::WithinCluster);
            fill_resample_indexes_grouped(plan, 5, Some(&ids), &mut rng, &mut out).unwrap();
            for (pos, &row) in out.iter().enumerate() {
                let index = usize::try_from(row.get()).expect("row");
                assert_eq!(ids[pos], ids[index]);
            }
            let mut sorted = out.clone();
            sorted.sort_unstable();
            assert_eq!(
                sorted,
                [
                    GlobalId::new(0),
                    GlobalId::new(1),
                    GlobalId::new(2),
                    GlobalId::new(3),
                    GlobalId::new(4),
                ]
            );
        }
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

    /// A moving-block bootstrap of 100 rows in blocks of 10 must be ten runs of ten
    /// consecutive source rows (serial dependence kept inside each block), and the
    /// same seed must replay the same indexes.
    #[test]
    fn moving_block_indexes_are_consecutive_runs_and_replay() {
        let data = float_series(100, 1);
        let draw = |seed: u64| {
            let mut rng = CausalRng::from_seed(seed);
            let mut idx = Vec::new();
            resample_timeseries(
                &data,
                ResamplingPlan::MovingBlock { length: 10 },
                &mut rng,
                &mut idx,
            )
            .unwrap();
            idx
        };
        let idx = draw(5);
        assert_eq!(idx.len(), 100);
        assert!(idx.iter().all(|i| i.get() < 100));
        for block in idx.chunks(10) {
            for w in block.windows(2) {
                assert_eq!(w[1].get(), w[0].get() + 1, "block is not a consecutive run: {block:?}");
            }
        }
        assert_eq!(idx, draw(5));
    }

    #[test]
    fn circular_block_usize_fill_matches_u32_stream() {
        let mut rng_u32 = CausalRng::from_seed(17);
        let mut rng_usize = CausalRng::from_seed(17);
        let mut as_u32 = Vec::new();
        fill_resample_indexes(
            ResamplingPlan::CircularBlock { length: 7 },
            64,
            &mut rng_u32,
            &mut as_u32,
        )
        .unwrap();
        let mut as_usize = vec![0usize; 64];
        fill_circular_block_indexes(64, 7, &mut rng_usize, &mut as_usize).unwrap();
        assert_eq!(
            as_u32.iter().map(|i| usize::try_from(i.get()).expect("row")).collect::<Vec<_>>(),
            as_usize
        );
        assert_eq!(rng_u32.state(), rng_usize.state());
    }

    #[test]
    fn circular_block_length_rule() {
        assert_eq!(integer_cube_root_ceil(0), 0);
        assert_eq!(integer_cube_root_ceil(1), 1);
        assert_eq!(integer_cube_root_ceil(64), 4);
        assert_eq!(integer_cube_root_ceil(65), 5);
        assert_eq!(integer_cube_root_ceil(1000), 10);
        // ⌈160^{1/3}⌉ = 6 dominates a two-slice window; a wide window dominates.
        assert_eq!(circular_block_length(2, 160), 6);
        assert_eq!(circular_block_length(9, 160), 9);
        // Capped at n, never zero.
        assert_eq!(circular_block_length(50, 12), 12);
        assert_eq!(circular_block_length(0, 0), 1);
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
        assert_eq!(
            sorted.iter().map(|id| id.get()).collect::<Vec<_>>(),
            (0..20u64).collect::<Vec<_>>()
        );
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
            assert!(i.get() < u64::try_from(n).expect("n"));
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
        let ctx = ExecutionContext::for_tests(42);
        let n = 20usize;
        let b = 8usize;
        let mut out_a = vec![GlobalId::new(0); n * b];
        let mut out_b = vec![GlobalId::new(0); n * b];
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
        assert!(out_a.iter().all(|i| i.get() < u64::try_from(n).expect("n")));
    }

    #[test]
    fn weight_batch_rows_sum_to_n() {
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
        let mut rng = CausalRng::from_seed(11);
        let mut rows = Vec::new();
        let mut lengths = [0usize; 3];
        for _ in 0..4000 {
            fill_resample_indexes_grouped(plan, 4, Some(&ids), &mut rng, &mut rows).unwrap();
            lengths[(rows.len() - 2) / 2] += 1;
            let mut cursor = 0;
            for _ in 0..2 {
                if rows[cursor].get() == 0 {
                    cursor += 1;
                } else {
                    let got: Vec<u64> =
                        rows[cursor..cursor + 3].iter().map(|id| id.get()).collect();
                    assert_eq!(got, [1, 2, 3]);
                    cursor += 3;
                }
            }
            assert_eq!(cursor, rows.len(), "each replicate must contain exactly G=2 draws");
        }
        // Two independent cluster draws give row-count probabilities 1/4, 1/2, 1/4.
        for (count, target) in lengths.into_iter().zip([0.25, 0.5, 0.25]) {
            assert!((count as f64 / 4000.0 - target).abs() < 0.04);
        }
        let mut fixed = vec![GlobalId::new(0); 8];
        assert!(
            fill_resample_index_batch(plan, 4, 2, Some(&ids), &context, 11, &mut fixed).is_err()
        );
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
                    source.validity().is_valid(usize::try_from(origin.get()).expect("row"))
                );
                assert_eq!(
                    replicate.storage().analysis_mask().unwrap().is_valid(row),
                    data.storage()
                        .analysis_mask()
                        .unwrap()
                        .is_valid(usize::try_from(origin.get()).expect("row"))
                );
                assert!(
                    (replicate.storage().weights().unwrap()[row]
                        - data.storage().weights().unwrap()
                            [usize::try_from(origin.get()).expect("row")])
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
        for error in [index_error, weight_error] {
            assert!(error.to_string().contains("overflow"));
        }
    }

    #[test]
    fn resample_workers_stay_within_the_pool() {
        let n = 4usize;
        let replicates = 8usize;
        let scratch_bytes = u64::try_from(n * 2 * size_of::<u64>()).unwrap();
        let shared = AdmissionPool::fixed(2, scratch_bytes);
        let mut ctx = ExecutionContext::production(1, 8).with_admission(shared.clone());
        ctx.memory = MemoryBudget { soft_limit_bytes: Some(1), hard_limit_bytes: None };
        let barrier = Arc::new(Barrier::new(2));
        let inflight = Arc::new(AtomicUsize::new(0));
        let peak = Arc::new(AtomicUsize::new(0));
        let mut out = vec![0u64; n * replicates];
        fill_replicate_rows(&ctx, n, replicates, &mut out, |_, scratch| {
            assert!(scratch.len() <= n);
            let now = inflight.fetch_add(1, Ordering::SeqCst).saturating_add(1);
            peak.fetch_max(now, Ordering::SeqCst);
            barrier.wait();
            inflight.fetch_sub(1, Ordering::SeqCst);
            scratch.clear();
            scratch.resize(n, 0);
            Ok(())
        })
        .unwrap();
        assert_eq!(peak.load(Ordering::SeqCst), 2);
        assert!(n * 2 * size_of::<u64>() <= usize::try_from(shared.byte_budget()).unwrap());
        assert_eq!(inflight.load(Ordering::SeqCst), 0);
        assert_eq!(ctx.memory.soft_limit_bytes, Some(1));
        assert!(ctx.memory.hard_limit_bytes.is_none());
        assert_eq!(shared.available().threads(), 2);
        assert_eq!(shared.available().bytes(), scratch_bytes);
    }

    #[test]
    fn resample_scratch_above_the_hard_limit_is_refused_before_work() {
        let n = 4usize;
        let replicates = 2usize;
        let bytes = u64::try_from(n * size_of::<GlobalId>()).unwrap();
        let mut ctx = ExecutionContext::for_tests(1);
        ctx.memory = MemoryBudget { soft_limit_bytes: Some(1), hard_limit_bytes: Some(bytes - 1) };
        let mut out = vec![GlobalId::new(9); n * replicates];
        let error = fill_resample_index_batch(
            ResamplingPlan::IidBootstrap,
            n,
            replicates,
            None,
            &ctx,
            0,
            &mut out,
        )
        .unwrap_err();
        assert!(error.to_string().contains("hard byte limit"), "{error}");
        assert!(out.iter().all(|id| id.get() == 9));
        assert_eq!(ctx.memory.soft_limit_bytes, Some(1));
        assert_eq!(ctx.memory.hard_limit_bytes, Some(bytes - 1));
    }

    #[test]
    fn resample_scratch_overflow_is_refused_before_allocation() {
        let ctx = ExecutionContext::for_tests(1);
        let width = size_of::<u64>();
        let n = (usize::MAX / width) + 1;
        assert_eq!(n.wrapping_mul(width), 0);
        let calls = AtomicUsize::new(0);
        let mut out = [0u64; 0];
        let error = fill_replicate_rows(&ctx, n, 1, &mut out, |_, _| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .unwrap_err();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(error.to_string().contains("overflow"), "{error}");
        assert_eq!(ctx.memory, MemoryBudget::unlimited());
    }

    #[test]
    fn resample_refuses_when_the_pool_cannot_fund_scratch() {
        let n = 4usize;
        let replicates = 8usize;
        let bytes = u64::try_from(n * 2 * size_of::<u64>()).unwrap();
        let shared = AdmissionPool::fixed(2, bytes - 1);
        let ctx = ExecutionContext::production(1, 8).with_admission(shared.clone());
        let calls = AtomicUsize::new(0);
        let mut out = vec![7u64; n * replicates];
        let error = fill_replicate_rows(&ctx, n, replicates, &mut out, |_, _| {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(())
        })
        .unwrap_err();
        assert_eq!(calls.load(Ordering::SeqCst), 0);
        assert!(error.to_string().contains("cannot fund"), "{error}");
        assert!(out.iter().all(|value| *value == 7));
        assert_eq!(shared.available().threads(), 2);
        assert_eq!(shared.available().bytes(), bytes - 1);
        assert!(ctx.memory.hard_limit_bytes.is_none());
    }

    #[test]
    fn cancelled_batches_return_before_mutating_output() {
        let context = ExecutionContext::for_tests(1);
        context.cancellation.cancel();
        let mut indexes = [GlobalId::new(7)];
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
                == DataError::Cancelled
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
                == DataError::Cancelled
        );
        assert_eq!(indexes, [GlobalId::new(7)]);
        assert!((weights[0] - 0.7).abs() < 1e-12);
    }

    /// One reused buffer, filled per replicate, matches the replicate-major batch row.
    fn assert_reused_indexes_match_batch(
        plan: ResamplingPlan,
        n: usize,
        replicates: usize,
        cluster_ids: Option<&[u32]>,
        seed: u64,
        stream_base: u64,
        threads: u32,
    ) {
        let ctx = if threads == 1 {
            ExecutionContext::for_tests(seed)
        } else {
            ExecutionContext::production(seed, threads)
        };
        let mut batch = vec![GlobalId::new(0); n * replicates];
        fill_resample_index_batch(plan, n, replicates, cluster_ids, &ctx, stream_base, &mut batch)
            .unwrap();
        let mut reused = Vec::new();
        let mut held = 0usize;
        for replicate in 0..replicates {
            fill_resample_index_replicate(
                plan,
                n,
                cluster_ids,
                &ctx,
                stream_base,
                replicate as u64,
                &mut reused,
            )
            .unwrap();
            let start = replicate * n;
            assert_eq!(reused.as_slice(), &batch[start..start + n]);
            if replicate == 0 {
                held = reused.capacity();
            } else {
                assert_eq!(reused.capacity(), held);
            }
        }
    }

    #[test]
    fn bounded_iid_indexes_match_the_batch() {
        assert_reused_indexes_match_batch(ResamplingPlan::IidBootstrap, 20, 8, None, 42, 0xDEAD, 1);
        assert_reused_indexes_match_batch(ResamplingPlan::IidBootstrap, 20, 8, None, 42, 0xDEAD, 4);
    }

    #[test]
    fn bounded_block_indexes_match_the_batch() {
        let plans = [
            ResamplingPlan::MovingBlock { length: 3 },
            ResamplingPlan::CircularBlock { length: 4 },
            ResamplingPlan::StationaryBlock { expected_length: 2.5 },
        ];
        for plan in plans {
            assert_reused_indexes_match_batch(plan, 24, 6, None, 5, 0xB10C, 1);
        }
    }

    #[test]
    fn bounded_permutation_indexes_match_the_batch() {
        assert_reused_indexes_match_batch(
            ResamplingPlan::Permutation(PermutationScheme::Full),
            16,
            5,
            None,
            3,
            0x5EED,
            1,
        );
        let ids = [0u32, 0, 1, 1, 1];
        assert_reused_indexes_match_batch(
            ResamplingPlan::Permutation(PermutationScheme::WithinCluster),
            5,
            7,
            Some(&ids),
            3,
            0x5EED,
            1,
        );
    }

    #[test]
    fn bounded_bayesian_weights_match_the_batch() {
        let ctx = ExecutionContext::for_tests(7);
        let n = 15usize;
        let replicates = 4usize;
        let stream_base = 0xBEEF_u64;
        let mut batch = vec![0.0; n * replicates];
        fill_resample_weight_batch(
            ResamplingPlan::BayesianBootstrap,
            n,
            replicates,
            &ctx,
            stream_base,
            &mut batch,
        )
        .unwrap();
        let mut reused = Vec::new();
        let mut held = 0usize;
        for replicate in 0..replicates {
            fill_resample_weight_replicate(
                ResamplingPlan::BayesianBootstrap,
                n,
                &ctx,
                stream_base,
                replicate as u64,
                &mut reused,
            )
            .unwrap();
            let row = &batch[replicate * n..(replicate + 1) * n];
            assert_eq!(reused.len(), n);
            for (got, expect) in reused.iter().zip(row) {
                assert!((got - expect).abs() < 1e-12);
            }
            if replicate == 0 {
                held = reused.capacity();
            } else {
                assert_eq!(reused.capacity(), held);
            }
        }
    }

    #[test]
    fn bounded_equal_cluster_indexes_match_the_batch() {
        let ids = [0u32, 0, 1, 1, 2, 2];
        let plan = ResamplingPlan::ClusterBootstrap { cluster: VariableId::from_raw(0) };
        assert_reused_indexes_match_batch(plan, ids.len(), 9, Some(&ids), 11, 0xC1, 1);
    }

    #[test]
    fn bounded_cluster_replicate_keeps_variable_length_and_repeated_copies() {
        let plan = ResamplingPlan::ClusterBootstrap { cluster: VariableId::from_raw(0) };
        let ids = [0u32, 1, 1, 1];
        let ctx = ExecutionContext::for_tests(42);
        let mut reused = Vec::new();
        let mut saw_repeat = false;
        let mut saw_variable = false;
        for replicate in 0..64u64 {
            fill_resample_index_replicate(plan, 4, Some(&ids), &ctx, 11, replicate, &mut reused)
                .unwrap();
            let mut cursor = 0;
            let (mut copies0, mut copies1) = (0usize, 0usize);
            while cursor < reused.len() {
                if reused[cursor].get() == 0 {
                    copies0 += 1;
                    cursor += 1;
                } else {
                    let got: Vec<u64> =
                        reused[cursor..cursor + 3].iter().map(|id| id.get()).collect();
                    assert_eq!(got, [1, 2, 3]);
                    copies1 += 1;
                    cursor += 3;
                }
            }
            assert_eq!(copies0 + copies1, 2, "a replicate draws G whole clusters");
            saw_repeat |= copies0 == 2 || copies1 == 2;
            saw_variable |= reused.len() != 4;
        }
        assert!(saw_repeat, "a repeated cluster must stay two copies");
        assert!(saw_variable, "unequal clusters change the replicate length");
    }

    fn declaration(design: DeclaredSamplingDesign, time_column: bool) -> SamplingDesignDeclaration {
        SamplingDesignDeclaration::new(design, time_column)
    }

    #[test]
    fn trajectory_design_refuses_a_row_bootstrap() {
        for time_column in [false, true] {
            let mut rng = CausalRng::from_seed(9);
            let mut out = vec![GlobalId::new(7)];
            let error = fill_declared_resample(
                declaration(DeclaredSamplingDesign::IndependentSubjectTrajectories, time_column),
                6,
                None,
                2,
                0,
                &mut rng,
                &mut out,
            )
            .expect_err("a trajectory cannot take a row bootstrap");
            let message = error.to_string();
            assert!(message.contains("independent_subject_trajectories"), "{message}");
            assert!(message.contains("row_iid"), "{message}");
            assert!(message.contains("row_subject_iid"), "{message}");
            assert_eq!(out, vec![GlobalId::new(7)]);
        }
    }

    #[test]
    fn panel_design_draws_whole_clusters() {
        let labels = [0_u32, 0, 1, 1, 1, 2];
        let members: [Vec<u64>; 3] = [vec![0, 1], vec![2, 3, 4], vec![5]];
        let mut with_time = Vec::new();
        let mut without_time = Vec::new();
        fill_declared_resample(
            declaration(DeclaredSamplingDesign::ClusteredPanel, true),
            labels.len(),
            Some(&labels),
            1,
            0,
            &mut CausalRng::from_seed(5),
            &mut with_time,
        )
        .expect("a panel draws clusters");
        fill_declared_resample(
            declaration(DeclaredSamplingDesign::ClusteredPanel, false),
            labels.len(),
            Some(&labels),
            1,
            0,
            &mut CausalRng::from_seed(5),
            &mut without_time,
        )
        .expect("a time column does not change the panel draw");
        assert_eq!(with_time, without_time);
        let flat: Vec<u64> = with_time.iter().map(|id| id.get()).collect();
        let mut cursor = 0usize;
        let mut draws = 0usize;
        while cursor < flat.len() {
            let matched = members.iter().find(|member| flat[cursor..].starts_with(member));
            let Some(member) = matched else {
                panic!("a panel row was split from its cluster at {cursor}: {flat:?}");
            };
            cursor += member.len();
            draws += 1;
        }
        assert_eq!(draws, members.len());
        let embargoed = fill_declared_resample(
            declaration(DeclaredSamplingDesign::ClusteredPanel, true),
            labels.len(),
            Some(&labels),
            1,
            2,
            &mut CausalRng::from_seed(5),
            &mut Vec::new(),
        )
        .expect_err("a panel has no block boundary for an embargo");
        assert!(embargoed.to_string().contains("embargo"), "{embargoed}");
    }

    #[test]
    fn series_design_draws_a_contiguous_block_and_honors_the_embargo() {
        let n = 10usize;
        let length = 3usize;
        let mut block = Vec::new();
        let mut moving = Vec::new();
        fill_declared_resample(
            declaration(DeclaredSamplingDesign::DependentTimeSeries, false),
            n,
            None,
            length,
            0,
            &mut CausalRng::from_seed(4),
            &mut block,
        )
        .expect("a series draws a contiguous block");
        fill_resample_indexes(
            ResamplingPlan::MovingBlock { length },
            n,
            &mut CausalRng::from_seed(4),
            &mut moving,
        )
        .expect("moving block");
        assert_eq!(block, moving);
        let embargo = 2u32;
        let bound = DeclaredSamplingDesign::DependentTimeSeries
            .apply_splitting_with_embargo(SplittingRule::ContiguousBlock, embargo)
            .expect("a series takes an embargo");
        assert_eq!(bound.embargo(), embargo);
        let mut with_time = Vec::new();
        let mut without_time = Vec::new();
        fill_declared_resample(
            declaration(DeclaredSamplingDesign::DependentTimeSeries, true),
            n,
            None,
            length,
            embargo,
            &mut CausalRng::from_seed(8),
            &mut with_time,
        )
        .expect("embargoed block");
        fill_declared_resample(
            declaration(DeclaredSamplingDesign::DependentTimeSeries, false),
            n,
            None,
            length,
            embargo,
            &mut CausalRng::from_seed(8),
            &mut without_time,
        )
        .expect("a time column does not change the series draw");
        assert_eq!(with_time, without_time);
        let margin = usize::try_from(embargo).unwrap();
        assert_eq!(with_time.len(), n - margin * 2);
        for id in &with_time {
            let row = usize::try_from(id.get()).unwrap();
            assert!(row >= margin && row < n - margin, "embargoed row {row} was drawn");
        }
        for window in with_time.chunks(length) {
            let rows: Vec<usize> =
                window.iter().map(|id| usize::try_from(id.get()).unwrap()).collect();
            for pair in rows.windows(2) {
                assert_eq!(pair[1], pair[0] + 1);
            }
        }
    }
}
