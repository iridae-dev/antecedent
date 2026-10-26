//! Index-plan drawing: block, cluster, stationary and declared-design fills.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;

pub(super) fn design_independent_of_time_column(
    declaration: SamplingDesignDeclaration,
) -> DeclaredSamplingDesign {
    let with_column = SamplingDesignDeclaration::new(declaration.design(), true);
    let without_column = SamplingDesignDeclaration::new(declaration.design(), false);
    let selected = if declaration.time_column_present() { with_column } else { without_column };
    debug_assert_eq!(with_column.design(), without_column.design());
    selected.design()
}

pub(super) fn refuse_row_iid(design: DeclaredSamplingDesign) -> DataError {
    match design.apply_resampling(ResamplingRule::RowIid) {
        Err(error) => declared_design_error(error),
        Ok(_) => DataError::InvalidArgument {
            message: "a trajectory design refuses a row-IID draw".into(),
        },
    }
}

#[allow(
    clippy::too_many_arguments,
    reason = "private helper shared by the two grouped draws; each argument is a distinct input of the one draw"
)]
pub(super) fn draw_grouped_units(
    design: DeclaredSamplingDesign,
    resampling: ResamplingRule,
    splitting: SplittingRule,
    n: usize,
    unit_ids: Option<&[u32]>,
    embargo: u32,
    missing_units_are_a_row_bootstrap: bool,
    rng: &mut CausalRng,
    out: &mut Vec<GlobalId>,
) -> Result<(), DataError> {
    design.apply_resampling(resampling).map_err(declared_design_error)?;
    design.apply_splitting_with_embargo(splitting, embargo).map_err(declared_design_error)?;
    if n == 0 {
        return Err(DataError::InvalidArgument { message: "resample needs n > 0".into() });
    }
    let Some(ids) = unit_ids else {
        if missing_units_are_a_row_bootstrap {
            return Err(refuse_row_iid(design));
        }
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

pub(super) fn draw_contiguous_block(
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

pub(super) fn declared_design_error(error: SamplingDesignError) -> DataError {
    DataError::SamplingDesign(error)
}

pub(super) fn fill_embargoed_contiguous_block(
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
pub(super) fn prepare_clusters(
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
pub(super) struct ClusterIndex {
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

pub(super) fn fill_indexes_prepared(
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

pub(super) fn fill_block_slice<T: Copy>(
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

#[allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_precision_loss,
    reason = "the block length is a non-negative integer-valued f64 below the series length"
)]
pub(super) fn fill_stationary(
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
    // Geometric block length with mean L: P(L=k) = p(1-p)^{k-1}, p = 1/L, drawn by inverting
    // the survival function P(L > k) = (1-p)^k: one uniform per block, and none at all when
    // p = 1 (every block is a single row).
    let p = 1.0 / expected_length;
    let log_continue = (-p).ln_1p();
    while out.len() < n {
        let len = if p >= 1.0 {
            1
        } else {
            // `1 - U` lies in (0, 1], so its logarithm is finite. L = 1 + floor(ln(1-U) / ln(1-p)).
            let extra = ((1.0 - rng.next_f64()).ln() / log_continue).floor();
            if extra >= (n - 1) as f64 {
                n
            } else {
                // `extra` is a non-negative integer-valued f64 below `n`.
                extra as usize + 1
            }
        };
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

pub(super) fn fill_cluster_bootstrap(
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

pub(super) fn fill_within_cluster_permutation(
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

/// Smallest integer `k` with `k³ ≥ n` (`0` for `n = 0`), without floating-point rounding.
#[must_use]
pub(super) fn integer_cube_root_ceil(n: usize) -> usize {
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
