//! Exact nearest-neighbor matching index.
//!
//! Every query is answered exactly, with the tie rule of a linear scan over the donors in
//! index order: the strictly smallest distance wins and, among equal distances, the lowest
//! donor position. One-dimensional absolute-difference indexes search a sorted copy of the
//! donor values; Euclidean indexes search a balanced k-d tree whose pruning bound is
//! computed with the same floating-point operations as the distance itself, so a pruned
//! subtree can never hold a donor the scan would have chosen. Resource use is bounded by
//! the [`ExecutionContext`]: memory against its hard limit, work through cancellation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use antecedent_core::ExecutionContext;
use antecedent_core::execution::{AdmissionError, AdmissionLease};

use crate::error::StatsError;

/// Donors at or below which a k-d tree node is a leaf scanned directly.
const KD_LEAF: usize = 16;

/// Queries one worker evaluates between cancellation checks.
const QUERY_CHUNK: usize = 1024;

/// Distance metric for matching.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MatchingDistance {
    /// Euclidean distance on feature rows.
    Euclidean,
    /// Absolute difference on a single scalar (e.g. propensity).
    Absolute,
}

/// How an index answers nearest-neighbour queries.
#[derive(Clone, Debug)]
enum Layout {
    /// Distinct donor values ascending, each with the lowest donor position holding it
    /// (`Absolute`, one dimension).
    Sorted { values: Vec<f64>, first_position: Vec<usize> },
    /// Balanced k-d tree over the donor rows (`Euclidean`).
    Tree(KdTree),
}

/// Reusable nearest-neighbor index over control (or donor) rows.
#[derive(Clone, Debug)]
pub struct MatchingIndex {
    /// Feature dimension.
    pub dim: usize,
    /// Row-major donor features: `n_donors * dim`.
    features: Vec<f64>,
    /// Original row indices for each donor.
    donor_rows: Vec<usize>,
    /// Metric.
    distance: MatchingDistance,
    /// Search structure over `features`.
    layout: Layout,
}

impl MatchingIndex {
    /// Build an exact index from donor feature rows.
    ///
    /// `features_rowmajor` length must be `donor_rows.len() * dim`. The index keeps a copy
    /// of the features and an `O(n)` search structure, so memory is `O(n * dim)`.
    ///
    /// # Errors
    ///
    /// Shape mismatch, a non-finite donor feature (a NaN distance compares false against
    /// everything, so such a donor could never be matched and would silently corrupt every
    /// nearest-neighbour scan), cancellation, or a footprint above the context's hard
    /// memory limit.
    pub fn exact_within(
        features_rowmajor: &[f64],
        dim: usize,
        donor_rows: &[usize],
        distance: MatchingDistance,
        ctx: &ExecutionContext,
    ) -> Result<Self, StatsError> {
        let n = donor_rows.len();
        if dim == 0 {
            return Err(StatsError::Shape { message: "matching dim must be > 0" });
        }
        if features_rowmajor.len() != n.saturating_mul(dim) {
            return Err(StatsError::Shape { message: "features length != n_donors * dim" });
        }
        if distance == MatchingDistance::Absolute && dim != 1 {
            return Err(StatsError::Shape { message: "Absolute distance requires dim == 1" });
        }
        if features_rowmajor.iter().any(|v| !v.is_finite()) {
            return Err(StatsError::Shape { message: "matching features must be finite" });
        }
        let _admitted = check_budget(ctx, "matching index", index_bytes(n, dim, distance))?;
        let layout = match distance {
            MatchingDistance::Absolute => sorted_layout(features_rowmajor),
            MatchingDistance::Euclidean => {
                Layout::Tree(KdTree::build(features_rowmajor, n, dim, ctx)?)
            }
        };
        Ok(Self {
            dim,
            features: features_rowmajor.to_vec(),
            donor_rows: donor_rows.to_vec(),
            distance,
            layout,
        })
    }

    /// Whether this index retains these exact feature bits.
    ///
    /// Callers may use this to verify a compact hash key before reusing the index. The
    /// retained donor features make the check allocation-free and prevent a hash collision
    /// from replaying nearest-neighbor results from different donor features.
    #[must_use]
    pub fn matches_features(&self, features_rowmajor: &[f64]) -> bool {
        self.features.len() == features_rowmajor.len()
            && self
                .features
                .iter()
                .zip(features_rowmajor)
                .all(|(stored, candidate)| stored.to_bits() == candidate.to_bits())
    }

    /// Number of donors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.donor_rows.len()
    }

    /// Whether the index is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.donor_rows.is_empty()
    }

    /// Estimated retained memory for this index (bytes).
    #[must_use]
    pub fn retained_memory_bytes(&self) -> u64 {
        let f = self.features.capacity() * core::mem::size_of::<f64>();
        let d = self.donor_rows.capacity() * core::mem::size_of::<usize>();
        let layout = match &self.layout {
            Layout::Sorted { values, first_position } => {
                values.capacity() * core::mem::size_of::<f64>()
                    + first_position.capacity() * core::mem::size_of::<usize>()
            }
            Layout::Tree(tree) => tree.bytes(),
        };
        u64::try_from(f + d + layout).unwrap_or(u64::MAX)
    }

    /// Find the nearest donor to `query` (length `dim`).
    ///
    /// Returns `(donor_original_row, distance)`. If `caliper` is `Some(c)`, returns
    /// `None` when the nearest distance exceeds `c`. Returns `None` as well when no donor
    /// is at a finite distance (the squared differences overflowed): there is no nearest
    /// donor to report. Equal distances resolve to the donor listed first.
    ///
    /// # Errors
    ///
    /// Query length mismatch, empty index, or a non-finite query.
    pub fn nearest(
        &self,
        query: &[f64],
        caliper: Option<f64>,
    ) -> Result<Option<(usize, f64)>, StatsError> {
        if query.len() != self.dim {
            return Err(StatsError::Shape { message: "query length != dim" });
        }
        if self.donor_rows.is_empty() {
            return Err(StatsError::Shape { message: "empty matching index" });
        }
        if query.iter().any(|v| !v.is_finite()) {
            return Err(StatsError::Shape { message: "matching query must be finite" });
        }
        let (best_i, best_d) = match &self.layout {
            Layout::Sorted { values, first_position } => {
                nearest_sorted(values, first_position, query[0])
            }
            Layout::Tree(tree) => {
                let mut found = Nearest::new(1);
                tree.search(&self.features, self.dim, query, usize::MAX, &mut found);
                found.best().unwrap_or((usize::MAX, f64::INFINITY))
            }
        };
        if !best_d.is_finite() {
            return Ok(None);
        }
        if let Some(c) = caliper {
            if best_d > c {
                return Ok(None);
            }
        }
        Ok(Some((self.donor_rows[best_i], best_d)))
    }

    /// The `k` nearest donors to `query` by Euclidean distance, nearest first, as
    /// `(donor_original_row, distance)` pairs in `out` (cleared first). Equal distances
    /// resolve to the donor listed first, so the answer is the `k` smallest
    /// `(distance, position)` pairs. Fewer than `k` pairs come back when the index holds
    /// fewer donors. Memory is `O(k)`.
    ///
    /// # Errors
    ///
    /// Query length mismatch, a non-finite query, or an index that is not Euclidean.
    pub fn nearest_k(
        &self,
        query: &[f64],
        k: usize,
        out: &mut Vec<(usize, f64)>,
    ) -> Result<(), StatsError> {
        let Layout::Tree(tree) = &self.layout else {
            return Err(StatsError::Shape { message: "nearest_k requires a Euclidean index" });
        };
        if query.len() != self.dim {
            return Err(StatsError::Shape { message: "query length != dim" });
        }
        if query.iter().any(|v| !v.is_finite()) {
            return Err(StatsError::Shape { message: "matching query must be finite" });
        }
        let mut found = Nearest::new(k.min(self.donor_rows.len()));
        tree.search(&self.features, self.dim, query, usize::MAX, &mut found);
        out.clear();
        out.extend(found.into_sorted().into_iter().map(|(pos, d)| (self.donor_rows[pos], d)));
        Ok(())
    }

    /// `k`-th self-distances of a matrix against itself without building an index.
    ///
    /// Equivalent to `MatchingIndex::exact_within(features, dim, &(0..n).collect(), distance, ctx)`
    /// followed by [`Self::kth_self_distances_of_donors_within`] and equal, bit for bit, to the
    /// `k`-th smallest of the `n - 1` row distances of an exhaustive scan. Each row is
    /// excluded from its own neighbour set by index; other rows at distance zero
    /// (duplicates) count. A k-d tree is built per call and each query keeps only its `k`
    /// best candidates, so memory is `O(n + k)` per worker beyond `out`, and time is
    /// `O(n log n)` for low-dimensional features (it degrades toward `O(n^2 dim)` only as
    /// the dimension grows). Permutation-null loops that vary one column of
    /// `features_rowmajor` between calls use this form; the joint distance changes with
    /// every permuted column, so no structure survives from one call to the next.
    ///
    /// # Errors
    ///
    /// Shape mismatch, `k == 0`, `n <= k`, non-finite features, cancellation, or a
    /// footprint above the context's hard memory limit.
    pub fn kth_self_distances_within(
        features_rowmajor: &[f64],
        n: usize,
        dim: usize,
        distance: MatchingDistance,
        k: usize,
        out: &mut [f64],
        ctx: &ExecutionContext,
    ) -> Result<(), StatsError> {
        if dim == 0 {
            return Err(StatsError::Shape { message: "matching dim must be > 0" });
        }
        if distance == MatchingDistance::Absolute && dim != 1 {
            return Err(StatsError::Shape { message: "Absolute distance requires dim == 1" });
        }
        if k == 0 {
            return Err(StatsError::Shape { message: "k must be > 0" });
        }
        if features_rowmajor.len() != n.saturating_mul(dim) {
            return Err(StatsError::Shape { message: "features length != n * dim" });
        }
        if out.len() < n {
            return Err(StatsError::Shape { message: "output too short" });
        }
        if n <= k {
            return Err(StatsError::Shape { message: "not enough donors for k" });
        }
        if features_rowmajor.iter().any(|v| !v.is_finite()) {
            return Err(StatsError::Shape { message: "matching features must be finite" });
        }
        let _admitted =
            check_budget(ctx, "k-th self-distance search", self_distance_bytes(n, k, ctx))?;
        let tree = KdTree::build(features_rowmajor, n, dim, ctx)?;
        tree.kth_self_distances(features_rowmajor, dim, distance, k, &mut out[..n], ctx)
    }

    /// Distance to the `k`-th nearest donor for each query row (row-major queries), where
    /// the queries are **external** to the donor set: no donor is skipped, so a query that
    /// coincides with a donor value matches it at distance zero.
    ///
    /// For queries that *are* the donor rows use [`Self::kth_self_distances_of_donors_within`].
    ///
    /// # Errors
    ///
    /// Shape mismatch, `k == 0`, too few donors, or a non-finite query.
    #[cfg(test)]
    pub fn kth_distances(
        &self,
        queries_rowmajor: &[f64],
        n_queries: usize,
        k: usize,
        out: &mut [f64],
    ) -> Result<(), StatsError> {
        if k == 0 {
            return Err(StatsError::Shape { message: "k must be > 0" });
        }
        if queries_rowmajor.len() != n_queries.saturating_mul(self.dim) {
            return Err(StatsError::Shape { message: "queries length != n_queries * dim" });
        }
        if out.len() < n_queries {
            return Err(StatsError::Shape { message: "output too short" });
        }
        let n_donors = self.donor_rows.len();
        if n_donors < k {
            return Err(StatsError::Shape { message: "not enough donors for k" });
        }
        if queries_rowmajor.iter().any(|v| !v.is_finite()) {
            return Err(StatsError::Shape { message: "matching query must be finite" });
        }
        let mut dists = vec![0.0; n_donors];
        for q in 0..n_queries {
            let query = &queries_rowmajor[q * self.dim..(q + 1) * self.dim];
            fill_distances(&mut dists, query, &self.features, self.dim, self.distance);
            let (_, kth, _) = dists.select_nth_unstable_by(k - 1, f64::total_cmp);
            out[q] = *kth;
        }
        Ok(())
    }

    /// Distance to the `k`-th nearest *other* donor for every donor row: donor `i` is
    /// excluded from its own neighbour set by index (never by a distance threshold), while
    /// other donors at distance zero count as neighbours.
    ///
    /// # Errors
    ///
    /// `k == 0`, `len() <= k`, `out` shorter than the donor count, cancellation, or a
    /// footprint above the context's hard memory limit.
    pub fn kth_self_distances_of_donors_within(
        &self,
        k: usize,
        out: &mut [f64],
        ctx: &ExecutionContext,
    ) -> Result<(), StatsError> {
        if k == 0 {
            return Err(StatsError::Shape { message: "k must be > 0" });
        }
        let n = self.donor_rows.len();
        if out.len() < n {
            return Err(StatsError::Shape { message: "output too short" });
        }
        if n <= k {
            return Err(StatsError::Shape { message: "not enough donors for k" });
        }
        match &self.layout {
            Layout::Tree(tree) => {
                let _admitted =
                    check_budget(ctx, "k-th self-distance search", self_distance_bytes(n, k, ctx))?;
                tree.kth_self_distances(
                    &self.features,
                    self.dim,
                    self.distance,
                    k,
                    &mut out[..n],
                    ctx,
                )
            }
            Layout::Sorted { .. } => Self::kth_self_distances_within(
                &self.features,
                n,
                self.dim,
                self.distance,
                k,
                out,
                ctx,
            ),
        }
    }

    /// Match every query row to its nearest donor.
    ///
    /// `queries_rowmajor` length = `n_queries * dim`.
    ///
    /// # Errors
    ///
    /// Shape mismatch, or cancellation.
    pub fn match_all_within(
        &self,
        queries_rowmajor: &[f64],
        n_queries: usize,
        caliper: Option<f64>,
        out_donor_row: &mut [usize],
        out_distance: &mut [f64],
        ctx: &ExecutionContext,
    ) -> Result<u32, StatsError> {
        if queries_rowmajor.len() != n_queries.saturating_mul(self.dim) {
            return Err(StatsError::Shape { message: "queries length != n_queries * dim" });
        }
        if out_donor_row.len() < n_queries || out_distance.len() < n_queries {
            return Err(StatsError::Shape { message: "output buffers too short" });
        }
        let mut matched = 0u32;
        for q in 0..n_queries {
            if q % QUERY_CHUNK == 0 && ctx.cancellation.is_cancelled() {
                return Err(StatsError::Cancelled);
            }
            let query = &queries_rowmajor[q * self.dim..(q + 1) * self.dim];
            if let Some((row, d)) = self.nearest(query, caliper)? {
                out_donor_row[q] = row;
                out_distance[q] = d;
                matched = matched.saturating_add(1);
            } else {
                out_donor_row[q] = usize::MAX;
                out_distance[q] = f64::INFINITY;
            }
        }
        Ok(matched)
    }
}

/// Soft upper bound the brute-force matching path had on its donor count.
#[deprecated(
    since = "2.4.1",
    note = "no longer enforced: the index scales with the donor count, bounded by the execution context's memory limit"
)]
pub const EXACT_MATCHING_ROW_LIMIT: usize = 10_000;

/// Entry points of the earlier signatures, which carried no execution context. Each runs the
/// `_within` form under a context with no memory limit and no cancellation.
impl MatchingIndex {
    /// [`Self::exact_within`] without an execution context.
    ///
    /// # Errors
    ///
    /// As [`Self::exact_within`], except no memory limit applies.
    pub fn exact(
        features_rowmajor: &[f64],
        dim: usize,
        donor_rows: &[usize],
        distance: MatchingDistance,
    ) -> Result<Self, StatsError> {
        Self::exact_within(features_rowmajor, dim, donor_rows, distance, &unbounded_context())
    }

    /// [`Self::kth_self_distances_within`] without an execution context.
    ///
    /// `dist_scratch` is the earlier per-query distance buffer; the search no longer needs one
    /// and leaves it untouched.
    ///
    /// # Errors
    ///
    /// As [`Self::kth_self_distances_within`], except no memory limit applies.
    #[allow(
        unused_variables,
        clippy::ptr_arg,
        reason = "the released signature, kept as it was; the search needs no scratch"
    )]
    pub fn kth_self_distances(
        features_rowmajor: &[f64],
        n: usize,
        dim: usize,
        distance: MatchingDistance,
        k: usize,
        out: &mut [f64],
        dist_scratch: &mut Vec<f64>,
    ) -> Result<(), StatsError> {
        Self::kth_self_distances_within(
            features_rowmajor,
            n,
            dim,
            distance,
            k,
            out,
            &unbounded_context(),
        )
    }

    /// [`Self::kth_self_distances_of_donors_within`] without an execution context.
    ///
    /// # Errors
    ///
    /// As [`Self::kth_self_distances_of_donors_within`], except no memory limit applies.
    pub fn kth_self_distances_of_donors(
        &self,
        k: usize,
        out: &mut [f64],
    ) -> Result<(), StatsError> {
        self.kth_self_distances_of_donors_within(k, out, &unbounded_context())
    }

    /// [`Self::match_all_within`] without an execution context.
    ///
    /// # Errors
    ///
    /// Shape mismatch.
    pub fn match_all(
        &self,
        queries_rowmajor: &[f64],
        n_queries: usize,
        caliper: Option<f64>,
        out_donor_row: &mut [usize],
        out_distance: &mut [f64],
    ) -> Result<u32, StatsError> {
        self.match_all_within(
            queries_rowmajor,
            n_queries,
            caliper,
            out_donor_row,
            out_distance,
            &unbounded_context(),
        )
    }
}

/// The context the earlier context-free entry points run under: no memory limit, not cancelled.
fn unbounded_context() -> ExecutionContext {
    ExecutionContext::production(0, 1)
}

/// Lease `bytes` on the admission pool, or refuse when the context is cancelled, when `bytes`
/// is `None` because the size math overflowed, or when the pool cannot fund it.
///
/// `None` is not zero and is not allocated. The returned lease holds the bytes until it is
/// dropped; keep it for as long as the planned buffers live.
pub(crate) fn check_budget(
    ctx: &ExecutionContext,
    what: &'static str,
    bytes: Option<u64>,
) -> Result<AdmissionLease, StatsError> {
    if ctx.cancellation.is_cancelled() {
        return Err(StatsError::Cancelled);
    }
    let Some(required_bytes) = bytes else {
        return Err(StatsError::ResourceLimit {
            message: what,
            required_bytes: u64::MAX,
            limit_bytes: ctx.memory_limit_bytes().unwrap_or(u64::MAX),
        });
    };
    let limit_bytes = ctx.memory_limit_bytes().unwrap_or(u64::MAX);
    if required_bytes > limit_bytes {
        return Err(StatsError::ResourceLimit { message: what, required_bytes, limit_bytes });
    }
    ctx.acquire_admission(0, required_bytes).map_err(|error| match error {
        AdmissionError::Cancelled => StatsError::Cancelled,
        _ => StatsError::ResourceLimit {
            message: what,
            required_bytes,
            limit_bytes: ctx.admission().available().bytes(),
        },
    })
}

/// Bytes a freshly built index over `n` donors of `dim` features retains.
fn index_bytes(n: usize, dim: usize, distance: MatchingDistance) -> Option<u64> {
    let word = core::mem::size_of::<f64>();
    let rows = n.checked_mul(dim)?.checked_mul(word)?.checked_add(n.checked_mul(word)?)?;
    let layout = match distance {
        MatchingDistance::Absolute => n.checked_mul(2 * word)?,
        MatchingDistance::Euclidean => KdTree::bytes_for(n)?,
    };
    u64::try_from(rows.checked_add(layout)?).ok()
}

/// Bytes the transient tree and per-worker candidate sets of a self-distance search use.
fn self_distance_bytes(n: usize, k: usize, ctx: &ExecutionContext) -> Option<u64> {
    let workers = usize::try_from(ctx.parallelism.max_threads.get()).ok()?;
    let candidates = k
        .checked_add(1)?
        .checked_mul(core::mem::size_of::<Entry>())?
        .checked_add(QUERY_CHUNK * core::mem::size_of::<f64>())?
        .checked_mul(workers)?;
    u64::try_from(KdTree::bytes_for(n)?.checked_add(candidates)?).ok()
}

/// Whether two finite values compare equal (`-0.0 == 0.0`).
fn same_value(a: f64, b: f64) -> bool {
    a.partial_cmp(&b) == Some(Ordering::Equal)
}

/// Sorted distinct values of a one-dimensional feature column, each with the lowest donor
/// position that holds it. Positions within a value run are ascending, so the first of a
/// run is that lowest position.
fn sorted_layout(features: &[f64]) -> Layout {
    let mut order: Vec<usize> = (0..features.len()).collect();
    order.sort_by(|&a, &b| {
        features[a].partial_cmp(&features[b]).unwrap_or(Ordering::Equal).then(a.cmp(&b))
    });
    let mut values: Vec<f64> = Vec::new();
    let mut first_position: Vec<usize> = Vec::new();
    for pos in order {
        if values.last().is_none_or(|&last| !same_value(last, features[pos])) {
            values.push(features[pos]);
            first_position.push(pos);
        }
    }
    Layout::Sorted { values, first_position }
}

/// Nearest value to `q` among ascending distinct `values`, as `(donor position, distance)`.
///
/// Floating-point subtraction is monotone, so the distance `|q - v|` is non-decreasing as
/// `v` moves away from `q` on either side: the minimum is at one of the two neighbours of
/// `q`, and every value tied with it (equal rounded distance, possibly several distinct
/// values) is contiguous around them. The lowest position among those ties is the donor a
/// linear scan returns. `(usize::MAX, inf)` when no value is at a finite distance.
fn nearest_sorted(values: &[f64], first_position: &[usize], q: f64) -> (usize, f64) {
    let split = values.partition_point(|&v| v < q);
    let left = split.checked_sub(1).map(|i| (q - values[i]).abs());
    let right = values.get(split).map(|&v| (q - v).abs());
    let best = match (left, right) {
        (Some(l), Some(r)) => l.min(r),
        (Some(d), None) | (None, Some(d)) => d,
        (None, None) => return (usize::MAX, f64::INFINITY),
    };
    if !best.is_finite() {
        return (usize::MAX, f64::INFINITY);
    }
    let mut position = usize::MAX;
    for i in (0..split).rev() {
        if !((q - values[i]).abs()).total_cmp(&best).is_eq() {
            break;
        }
        position = position.min(first_position[i]);
    }
    for i in split..values.len() {
        if !((q - values[i]).abs()).total_cmp(&best).is_eq() {
            break;
        }
        position = position.min(first_position[i]);
    }
    (position, best)
}

/// A candidate neighbour ordered by `(distance, position)`.
#[derive(Clone, Copy, Debug)]
struct Entry {
    distance: f64,
    position: usize,
}

impl PartialEq for Entry {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other) == Ordering::Equal
    }
}

impl Eq for Entry {}

impl PartialOrd for Entry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Entry {
    fn cmp(&self, other: &Self) -> Ordering {
        self.distance.total_cmp(&other.distance).then(self.position.cmp(&other.position))
    }
}

/// The `k` smallest `(distance, position)` candidates seen so far.
struct Nearest {
    k: usize,
    heap: BinaryHeap<Entry>,
}

impl Nearest {
    fn new(k: usize) -> Self {
        Self { k, heap: BinaryHeap::with_capacity(k.saturating_add(1)) }
    }

    fn offer(&mut self, distance: f64, position: usize) {
        let entry = Entry { distance, position };
        if self.heap.len() < self.k {
            self.heap.push(entry);
        } else if let Some(mut worst) = self.heap.peek_mut() {
            if entry < *worst {
                *worst = entry;
            }
        }
    }

    /// Distance of the current `k`-th candidate once `k` have been seen: a subtree whose
    /// every distance exceeds it cannot hold a better `(distance, position)` pair.
    fn bound(&self) -> Option<f64> {
        if self.k == 0 || self.heap.len() < self.k {
            return None;
        }
        self.heap.peek().map(|worst| worst.distance)
    }

    /// The single best candidate, as `(position, distance)`.
    fn best(&self) -> Option<(usize, f64)> {
        self.heap.iter().min().map(|e| (e.position, e.distance))
    }

    /// Candidates nearest first, as `(position, distance)`.
    fn into_sorted(self) -> Vec<(usize, f64)> {
        self.heap.into_sorted_vec().into_iter().map(|e| (e.position, e.distance)).collect()
    }
}

/// Balanced k-d tree over donor positions, stored implicitly.
///
/// Node `i` has children `2i + 1` and `2i + 2`; a node covering `perm[lo..hi]` splits at
/// `mid = lo + (hi - lo) / 2`, with every position in `perm[lo..mid]` at or below the split
/// value on the node's axis and every position in `perm[mid..hi]` at or above it. Ranges of
/// at most [`KD_LEAF`] positions are leaves.
#[derive(Clone, Debug)]
struct KdTree {
    /// Donor positions in tree order.
    perm: Vec<usize>,
    /// Split axis of each internal node.
    axis: Vec<u32>,
    /// Split value of each internal node.
    split: Vec<f64>,
}

impl KdTree {
    /// Heap-ordered node slots a tree over `n` positions needs.
    fn node_slots(n: usize) -> usize {
        let mut size = n;
        let mut levels = 0u32;
        while size > KD_LEAF {
            size = size.div_ceil(2);
            levels += 1;
        }
        1usize << levels
    }

    /// Bytes of a tree over `n` positions.
    fn bytes_for(n: usize) -> Option<usize> {
        let per_slot = core::mem::size_of::<u32>() + core::mem::size_of::<f64>();
        n.checked_mul(core::mem::size_of::<usize>())?
            .checked_add(Self::node_slots(n).checked_mul(per_slot)?)
    }

    fn bytes(&self) -> usize {
        self.perm.capacity() * core::mem::size_of::<usize>()
            + self.axis.capacity() * core::mem::size_of::<u32>()
            + self.split.capacity() * core::mem::size_of::<f64>()
    }

    /// Build the tree over the `n` rows of row-major `features`.
    fn build(
        features: &[f64],
        n: usize,
        dim: usize,
        ctx: &ExecutionContext,
    ) -> Result<Self, StatsError> {
        let slots = Self::node_slots(n);
        let mut tree =
            Self { perm: (0..n).collect(), axis: vec![0; slots], split: vec![0.0; slots] };
        tree.split_node(features, dim, 0, 0, n, ctx)?;
        Ok(tree)
    }

    fn split_node(
        &mut self,
        features: &[f64],
        dim: usize,
        node: usize,
        lo: usize,
        hi: usize,
        ctx: &ExecutionContext,
    ) -> Result<(), StatsError> {
        if hi - lo <= KD_LEAF {
            return Ok(());
        }
        if hi - lo >= 4 * QUERY_CHUNK && ctx.cancellation.is_cancelled() {
            return Err(StatsError::Cancelled);
        }
        let axis = widest_axis(features, dim, &self.perm[lo..hi]);
        let mid = lo + (hi - lo) / 2;
        self.perm[lo..hi].select_nth_unstable_by(mid - lo, |&a, &b| {
            features[a * dim + axis].total_cmp(&features[b * dim + axis])
        });
        // `axis < dim <= u32::MAX` for any matrix that fits in memory.
        self.axis[node] = u32::try_from(axis).unwrap_or(0);
        self.split[node] = features[self.perm[mid] * dim + axis];
        self.split_node(features, dim, 2 * node + 1, lo, mid, ctx)?;
        self.split_node(features, dim, 2 * node + 2, mid, hi, ctx)
    }

    /// Offer every position (except `exclude`) that can still improve `found`.
    fn search(
        &self,
        features: &[f64],
        dim: usize,
        query: &[f64],
        exclude: usize,
        found: &mut Nearest,
    ) {
        self.search_node(features, dim, query, exclude, found, 0, 0, self.perm.len());
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the recursion threads the search state and the node's position range explicitly to stay allocation-free"
    )]
    fn search_node(
        &self,
        features: &[f64],
        dim: usize,
        query: &[f64],
        exclude: usize,
        found: &mut Nearest,
        node: usize,
        lo: usize,
        hi: usize,
    ) {
        if hi - lo <= KD_LEAF {
            for &pos in &self.perm[lo..hi] {
                if pos != exclude {
                    found.offer(euclidean(query, &features[pos * dim..(pos + 1) * dim]), pos);
                }
            }
            return;
        }
        let mid = lo + (hi - lo) / 2;
        let axis = self.axis[node] as usize;
        let diff = query[axis] - self.split[node];
        let (near, far) = if diff <= 0.0 {
            ((2 * node + 1, lo, mid), (2 * node + 2, mid, hi))
        } else {
            ((2 * node + 2, mid, hi), (2 * node + 1, lo, mid))
        };
        self.search_node(features, dim, query, exclude, found, near.0, near.1, near.2);
        // The bound is computed with the distance's own operations (`d * d`, then `sqrt`):
        // every operation is monotone, so no donor across the plane is nearer than it, and
        // a donor tied with the current worst is still visited for the position tie-break.
        let plane = (diff * diff).sqrt();
        if found.bound().is_none_or(|worst| plane <= worst) {
            self.search_node(features, dim, query, exclude, found, far.0, far.1, far.2);
        }
    }

    /// `k`-th smallest distance from every row to the other rows, written to `out`.
    ///
    /// Queries run in chunks on the context's pool; each query's answer depends only on the
    /// features, so the output does not depend on the thread count.
    fn kth_self_distances(
        &self,
        features: &[f64],
        dim: usize,
        distance: MatchingDistance,
        k: usize,
        out: &mut [f64],
        ctx: &ExecutionContext,
    ) -> Result<(), StatsError> {
        let n = out.len();
        let chunks = n.div_ceil(QUERY_CHUNK);
        let parts = ctx.map_indexed(chunks, |chunk, inner| {
            if inner.cancellation.is_cancelled() {
                return Err(StatsError::Cancelled);
            }
            let start = chunk * QUERY_CHUNK;
            let end = (start + QUERY_CHUNK).min(n);
            let mut found = Nearest::new(k);
            let mut part = Vec::with_capacity(end - start);
            for q in start..end {
                found.heap.clear();
                let query = &features[q * dim..(q + 1) * dim];
                match distance {
                    MatchingDistance::Euclidean => {
                        self.search(features, dim, query, q, &mut found);
                    }
                    MatchingDistance::Absolute => {
                        self.search_absolute(features, query[0], q, &mut found);
                    }
                }
                part.push(found.heap.peek().map_or(f64::INFINITY, |worst| worst.distance));
            }
            Ok(part)
        })?;
        for (chunk, part) in parts.into_iter().enumerate() {
            out[chunk * QUERY_CHUNK..][..part.len()].copy_from_slice(&part);
        }
        Ok(())
    }

    /// One-dimensional variant of [`Self::search`] under the absolute-difference metric.
    fn search_absolute(&self, features: &[f64], query: f64, exclude: usize, found: &mut Nearest) {
        self.search_absolute_node(features, query, exclude, found, 0, 0, self.perm.len());
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the recursion threads the search state and the node's position range explicitly to stay allocation-free"
    )]
    fn search_absolute_node(
        &self,
        features: &[f64],
        query: f64,
        exclude: usize,
        found: &mut Nearest,
        node: usize,
        lo: usize,
        hi: usize,
    ) {
        if hi - lo <= KD_LEAF {
            for &pos in &self.perm[lo..hi] {
                if pos != exclude {
                    found.offer((query - features[pos]).abs(), pos);
                }
            }
            return;
        }
        let mid = lo + (hi - lo) / 2;
        let diff = query - self.split[node];
        let (near, far) = if diff <= 0.0 {
            ((2 * node + 1, lo, mid), (2 * node + 2, mid, hi))
        } else {
            ((2 * node + 2, mid, hi), (2 * node + 1, lo, mid))
        };
        self.search_absolute_node(features, query, exclude, found, near.0, near.1, near.2);
        if found.bound().is_none_or(|worst| diff.abs() <= worst) {
            self.search_absolute_node(features, query, exclude, found, far.0, far.1, far.2);
        }
    }
}

/// Axis with the widest coordinate spread over `positions` (the lowest axis on ties).
fn widest_axis(features: &[f64], dim: usize, positions: &[usize]) -> usize {
    let mut best_axis = 0usize;
    let mut best_spread = -1.0_f64;
    for axis in 0..dim {
        let mut lo = f64::INFINITY;
        let mut hi = f64::NEG_INFINITY;
        for &pos in positions {
            let v = features[pos * dim + axis];
            lo = lo.min(v);
            hi = hi.max(v);
        }
        let spread = hi - lo;
        if spread > best_spread {
            best_spread = spread;
            best_axis = axis;
        }
    }
    best_axis
}

/// Distances from `query` to every row of row-major `features`.
#[cfg(test)]
fn fill_distances(
    dists: &mut [f64],
    query: &[f64],
    features: &[f64],
    dim: usize,
    distance: MatchingDistance,
) {
    for (i, d) in dists.iter_mut().enumerate() {
        let row = &features[i * dim..(i + 1) * dim];
        *d = match distance {
            MatchingDistance::Euclidean => euclidean(query, row),
            MatchingDistance::Absolute => (query[0] - row[0]).abs(),
        };
    }
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

/// Scalar reference: nearest Euclidean neighbor among `donors` (row-major).
#[must_use]
pub fn nearest_euclidean_scalar(
    query: &[f64],
    donors_rowmajor: &[f64],
    n_donors: usize,
    dim: usize,
) -> Option<(usize, f64)> {
    if n_donors == 0 || query.len() != dim {
        return None;
    }
    let mut best_i = 0usize;
    let mut best_d = f64::INFINITY;
    for i in 0..n_donors {
        let row = &donors_rowmajor[i * dim..(i + 1) * dim];
        let d = euclidean(query, row);
        if d < best_d {
            best_d = d;
            best_i = i;
        }
    }
    Some((best_i, best_d))
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_core::StreamDomain;

    fn ctx() -> ExecutionContext {
        ExecutionContext::for_tests(0x3A7C)
    }

    /// `k`-th smallest (1-based) among all entries except `self_index`: the exhaustive
    /// reference the tree search must reproduce bit for bit.
    fn kth_excluding_self(dists: &mut [f64], self_index: usize, k: usize) -> f64 {
        dists[self_index] = f64::NEG_INFINITY;
        let (_, kth, _) = dists.select_nth_unstable_by(k, f64::total_cmp);
        *kth
    }

    /// Exhaustive `k`-th self-distances over all rows.
    fn scan_kth_self(
        features: &[f64],
        n: usize,
        dim: usize,
        distance: MatchingDistance,
        k: usize,
        queries: impl Iterator<Item = usize>,
    ) -> Vec<(usize, f64)> {
        let mut dists = vec![0.0; n];
        queries
            .map(|q| {
                let query = &features[q * dim..(q + 1) * dim];
                fill_distances(&mut dists, query, features, dim, distance);
                (q, kth_excluding_self(&mut dists, q, k))
            })
            .collect()
    }

    /// The linear scan the sorted and tree searches replace: strict `<`, so the lowest
    /// donor position wins a tie; a donor at a non-finite distance is never chosen.
    fn scan_nearest(
        index: &MatchingIndex,
        query: &[f64],
        caliper: Option<f64>,
    ) -> Option<(usize, f64)> {
        let mut best_i = 0usize;
        let mut best_d = f64::INFINITY;
        for (i, _) in index.donor_rows.iter().enumerate() {
            let row = &index.features[i * index.dim..(i + 1) * index.dim];
            let d = match index.distance {
                MatchingDistance::Euclidean => euclidean(query, row),
                MatchingDistance::Absolute => (query[0] - row[0]).abs(),
            };
            if d < best_d {
                best_d = d;
                best_i = i;
            }
        }
        if !best_d.is_finite() {
            return None;
        }
        if let Some(c) = caliper {
            if best_d > c {
                return None;
            }
        }
        Some((index.donor_rows[best_i], best_d))
    }

    fn bits(found: Option<(usize, f64)>) -> Option<(usize, u64)> {
        found.map(|(row, d)| (row, d.to_bits()))
    }

    /// A value drawn from a coarse lattice (many exact ties), with signed zeros and a
    /// continuous component so both tied and untied queries occur.
    fn lattice_value(rng: &mut antecedent_core::CausalRng) -> f64 {
        match rng.next_u64() % 5 {
            0 => -0.0,
            1 => 0.0,
            2 | 3 => (rng.next_u64() % 9) as f64 * 0.25 - 1.0,
            _ => rng.next_f64() * 2.0 - 1.0,
        }
    }

    #[test]
    fn finds_nearest_euclidean() {
        let donors = [0.0, 0.0, 1.0, 1.0, 5.0, 5.0];
        let idx = MatchingIndex::exact_within(
            &donors,
            2,
            &[10, 20, 30],
            MatchingDistance::Euclidean,
            &ctx(),
        )
        .unwrap();
        let (row, d) = idx.nearest(&[0.1, 0.1], None).unwrap().unwrap();
        assert_eq!(row, 10);
        assert!(d < 0.2);
    }

    #[test]
    fn caliper_rejects_far_matches() {
        let donors = [0.0, 10.0];
        let idx =
            MatchingIndex::exact_within(&donors, 1, &[0, 1], MatchingDistance::Absolute, &ctx())
                .unwrap();
        assert!(idx.nearest(&[0.05], Some(0.1)).unwrap().is_some());
        assert!(idx.nearest(&[5.0], Some(0.1)).unwrap().is_none());
    }

    #[test]
    fn sorted_and_tree_nearest_reproduce_the_linear_scan_bit_for_bit() {
        // Random donors on a coarse lattice (ties, duplicates, signed zeros) and queries on
        // the same lattice (exact hits and midpoints between donors), with and without a
        // caliper, across sizes that make the tree one leaf, shallow and several levels deep.
        let ctx = ctx();
        let mut rng = ctx.rng.stream_for(StreamDomain::Test, 0x11);
        for (case, n) in [1usize, 2, 3, 15, 16, 17, 33, 100, 257, 700].into_iter().enumerate() {
            for (dim, distance) in [
                (1, MatchingDistance::Absolute),
                (1, MatchingDistance::Euclidean),
                (2, MatchingDistance::Euclidean),
                (3, MatchingDistance::Euclidean),
                (5, MatchingDistance::Euclidean),
            ] {
                let donors: Vec<f64> = (0..n * dim).map(|_| lattice_value(&mut rng)).collect();
                let rows: Vec<usize> = (0..n).map(|i| 1000 + 3 * i).collect();
                let idx = MatchingIndex::exact_within(&donors, dim, &rows, distance, &ctx).unwrap();
                for _ in 0..150 {
                    let query: Vec<f64> = (0..dim).map(|_| lattice_value(&mut rng)).collect();
                    for caliper in [None, Some(0.0), Some(0.3)] {
                        assert_eq!(
                            bits(idx.nearest(&query, caliper).unwrap()),
                            bits(scan_nearest(&idx, &query, caliper)),
                            "case {case} n={n} dim={dim} {distance:?} query={query:?}"
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn nearest_resolves_equal_distances_to_the_lowest_donor_position() {
        // Donors 1.0 (pos 0), -1.0 (pos 1), 1.0 (pos 2), -1.0 (pos 3): the query 0.0 is
        // equidistant from all four, and a linear scan keeps the first.
        let idx = MatchingIndex::exact_within(
            &[1.0, -1.0, 1.0, -1.0],
            1,
            &[40, 41, 42, 43],
            MatchingDistance::Absolute,
            &ctx(),
        )
        .unwrap();
        assert_eq!(idx.nearest(&[0.0], None).unwrap(), Some((40, 1.0)));
        // Same donors listed with the negative value first.
        let idx = MatchingIndex::exact_within(
            &[-1.0, 1.0, -1.0, 1.0],
            1,
            &[40, 41, 42, 43],
            MatchingDistance::Absolute,
            &ctx(),
        )
        .unwrap();
        assert_eq!(idx.nearest(&[0.0], None).unwrap(), Some((40, 1.0)));
        // A signed zero is the same value: position decides, not the sort order of -0.0.
        let idx = MatchingIndex::exact_within(
            &[0.0, -0.0],
            1,
            &[7, 8],
            MatchingDistance::Absolute,
            &ctx(),
        )
        .unwrap();
        assert_eq!(idx.nearest(&[0.0], None).unwrap(), Some((7, 0.0)));
    }

    #[test]
    fn nearest_k_returns_the_k_smallest_distance_position_pairs() {
        let ctx = ctx();
        let mut rng = ctx.rng.stream_for(StreamDomain::Test, 0x12);
        let (n, dim) = (300usize, 3usize);
        let donors: Vec<f64> = (0..n * dim).map(|_| lattice_value(&mut rng)).collect();
        let rows: Vec<usize> = (0..n).map(|i| 500 + i).collect();
        let idx =
            MatchingIndex::exact_within(&donors, dim, &rows, MatchingDistance::Euclidean, &ctx)
                .unwrap();
        let mut out = Vec::new();
        for _ in 0..40 {
            let query: Vec<f64> = (0..dim).map(|_| lattice_value(&mut rng)).collect();
            for k in [1usize, 5, 40, 300, 400] {
                idx.nearest_k(&query, k, &mut out).unwrap();
                let mut all: Vec<(f64, usize)> = (0..n)
                    .map(|i| (euclidean(&query, &donors[i * dim..(i + 1) * dim]), i))
                    .collect();
                all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(&b.1)));
                let expected: Vec<(usize, u64)> =
                    all.iter().take(k).map(|&(d, i)| (rows[i], d.to_bits())).collect();
                let got: Vec<(usize, u64)> = out.iter().map(|&(r, d)| (r, d.to_bits())).collect();
                assert_eq!(got, expected, "k={k}");
            }
        }
        let absolute =
            MatchingIndex::exact_within(&[0.0, 1.0], 1, &[0, 1], MatchingDistance::Absolute, &ctx)
                .unwrap();
        assert!(absolute.nearest_k(&[0.0], 1, &mut out).is_err());
    }

    #[test]
    fn kth_self_distances_reproduce_the_exhaustive_scan_bit_for_bit() {
        // Duplicated rows and coarse-lattice ties exercise the exclude-by-index rule, and a
        // multi-threaded context must give the same bits as a serial one.
        let serial = ctx();
        let mut threaded = ExecutionContext::for_tests(0x3A7C);
        threaded.parallelism = antecedent_core::Parallelism::bounded(
            antecedent_core::NonZeroThreadCount::new(4).unwrap(),
        );
        let mut rng = serial.rng.stream_for(StreamDomain::Test, 0x13);
        for (n, dim, distance) in [
            (40usize, 3usize, MatchingDistance::Euclidean),
            (700, 2, MatchingDistance::Euclidean),
            (1500, 4, MatchingDistance::Euclidean),
            (900, 1, MatchingDistance::Absolute),
        ] {
            let mut feats: Vec<f64> = (0..n * dim).map(|_| lattice_value(&mut rng)).collect();
            // Make a block of exact duplicates.
            for r in 1..n.min(20) {
                let (head, tail) = feats.split_at_mut(r * dim);
                tail[..dim].copy_from_slice(&head[..dim]);
            }
            for k in [1usize, 3, 7] {
                let mut via_tree = vec![0.0; n];
                MatchingIndex::kth_self_distances_within(
                    &feats,
                    n,
                    dim,
                    distance,
                    k,
                    &mut via_tree,
                    &serial,
                )
                .unwrap();
                let mut via_pool = vec![0.0; n];
                MatchingIndex::kth_self_distances_within(
                    &feats,
                    n,
                    dim,
                    distance,
                    k,
                    &mut via_pool,
                    &threaded,
                )
                .unwrap();
                let donors: Vec<usize> = (0..n).collect();
                let idx =
                    MatchingIndex::exact_within(&feats, dim, &donors, distance, &serial).unwrap();
                let mut via_index = vec![0.0; n];
                idx.kth_self_distances_of_donors_within(k, &mut via_index, &serial).unwrap();
                for (q, expected) in scan_kth_self(&feats, n, dim, distance, k, 0..n) {
                    let want = expected.to_bits();
                    assert_eq!(via_tree[q].to_bits(), want, "n={n} dim={dim} k={k} query {q}");
                    assert_eq!(via_pool[q].to_bits(), want, "pool n={n} dim={dim} k={k} query {q}");
                    assert_eq!(
                        via_index[q].to_bits(),
                        want,
                        "index n={n} dim={dim} k={k} query {q}"
                    );
                }
            }
        }
    }

    #[test]
    fn resource_limits_and_cancellation_refuse_instead_of_building() {
        let donors = vec![0.5; 4000];
        let rows: Vec<usize> = (0..2000).collect();
        let mut limited = ctx();
        limited.set_memory_limit(Some(1024));
        for distance in [MatchingDistance::Euclidean, MatchingDistance::Absolute] {
            let dim = if distance == MatchingDistance::Absolute { 1 } else { 2 };
            let feats = &donors[..rows.len() * dim];
            assert!(matches!(
                MatchingIndex::exact_within(feats, dim, &rows, distance, &limited),
                Err(StatsError::ResourceLimit { .. })
            ));
        }
        let mut out = vec![0.0; 2000];
        assert!(matches!(
            MatchingIndex::kth_self_distances_within(
                &donors,
                2000,
                2,
                MatchingDistance::Euclidean,
                3,
                &mut out,
                &limited
            ),
            Err(StatsError::ResourceLimit { .. })
        ));
        let cancelled = ctx();
        cancelled.cancellation.cancel();
        assert!(matches!(
            MatchingIndex::exact_within(&donors, 2, &rows, MatchingDistance::Euclidean, &cancelled),
            Err(StatsError::Cancelled)
        ));
        let idx =
            MatchingIndex::exact_within(&donors, 2, &rows, MatchingDistance::Euclidean, &ctx())
                .unwrap();
        let mut donor_rows = [0usize; 1];
        let mut dist = [0.0; 1];
        assert!(matches!(
            idx.match_all_within(&[0.5, 0.5], 1, None, &mut donor_rows, &mut dist, &cancelled),
            Err(StatsError::Cancelled)
        ));
    }

    #[test]
    fn unrepresentable_index_bytes_are_refused_before_allocation() {
        assert_eq!(usize::MAX.wrapping_mul(usize::MAX), 1);
        assert!(index_bytes(usize::MAX, usize::MAX, MatchingDistance::Euclidean).is_none());
        let mut open = ctx();
        open.set_memory_limit(None);
        let error = check_budget(&open, "matching index", None).unwrap_err();
        assert!(matches!(
            error,
            StatsError::ResourceLimit { required_bytes: u64::MAX, limit_bytes: u64::MAX, .. }
        ));
        assert!(open.memory_limit_bytes().is_none());
    }

    #[test]
    fn self_distance_workers_cannot_exceed_the_hard_limit() {
        let n = 32usize;
        let k = 3usize;
        let dim = 2usize;
        let one = ctx();
        let mut many = ExecutionContext::production(1, 8);
        let serial_bytes = self_distance_bytes(n, k, &one).unwrap();
        let parallel_bytes = self_distance_bytes(n, k, &many).unwrap();
        assert!(parallel_bytes > serial_bytes);
        assert!(parallel_bytes <= serial_bytes.saturating_mul(8));
        many.set_memory_limit(Some(serial_bytes));
        let mut out = vec![1.0; n];
        let feats = vec![0.5; n * dim];
        let error = MatchingIndex::kth_self_distances_within(
            &feats,
            n,
            dim,
            MatchingDistance::Euclidean,
            k,
            &mut out,
            &many,
        )
        .unwrap_err();
        assert!(matches!(error, StatsError::ResourceLimit { .. }));
        assert!(out.iter().all(|value| value.total_cmp(&1.0).is_eq()));
        assert_eq!(many.memory_limit_bytes(), Some(serial_bytes));
    }

    #[test]
    fn nan_and_infinite_features_are_refused_not_matched() {
        let ctx = ctx();
        assert!(
            MatchingIndex::exact_within(
                &[0.0, f64::NAN],
                1,
                &[0, 1],
                MatchingDistance::Absolute,
                &ctx
            )
            .is_err()
        );
        assert!(
            MatchingIndex::exact_within(
                &[f64::INFINITY, 1.0],
                1,
                &[0, 1],
                MatchingDistance::Absolute,
                &ctx
            )
            .is_err()
        );
        let idx =
            MatchingIndex::exact_within(&[0.0, 1.0], 1, &[7, 8], MatchingDistance::Absolute, &ctx)
                .unwrap();
        // A NaN query used to compare false against every donor and match donor 0 at +inf.
        assert!(idx.nearest(&[f64::NAN], None).is_err());
        let mut rows = [0usize; 1];
        let mut dist = [0.0; 1];
        assert!(idx.match_all_within(&[f64::NAN], 1, None, &mut rows, &mut dist, &ctx).is_err());
        // Squared differences overflow: no donor is at a finite distance, so no match.
        let far = MatchingIndex::exact_within(&[1e200], 1, &[3], MatchingDistance::Euclidean, &ctx)
            .unwrap();
        assert_eq!(far.nearest(&[-1e200], None).unwrap(), None);
        // A finite donor beyond an overflowing one is still found, as the scan finds it.
        let mixed = MatchingIndex::exact_within(
            &[1e200, 0.0, 1e200],
            1,
            &[3, 4, 5],
            MatchingDistance::Euclidean,
            &ctx,
        )
        .unwrap();
        assert_eq!(mixed.nearest(&[-1e200], None).unwrap(), None);
        assert_eq!(bits(mixed.nearest(&[-3.0], None).unwrap()), bits(Some((4, 3.0))));
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "the expected values are small integers or dyadic fractions reproduced exactly in f64"
    )]
    fn external_query_equal_to_a_donor_matches_it_at_distance_zero() {
        // Queries that are not the donor set must not have "their" zero-distance donor
        // skipped: the 1st neighbour of a query sitting on donor value 1.0 is at 0, not 1.
        let idx = MatchingIndex::exact_within(
            &[0.0, 1.0, 2.0, 3.0],
            1,
            &[0, 1, 2, 3],
            MatchingDistance::Absolute,
            &ctx(),
        )
        .unwrap();
        let mut out = [f64::NAN; 1];
        idx.kth_distances(&[1.0], 1, 1, &mut out).unwrap();
        assert_eq!(out[0], 0.0);
        idx.kth_distances(&[1.0], 1, 2, &mut out).unwrap();
        assert_eq!(out[0], 1.0);
        // Unit-free: the old absolute 1e-15 test misclassified small-unit queries.
        let tiny = MatchingIndex::exact_within(
            &[0.0, 1e-20],
            1,
            &[0, 1],
            MatchingDistance::Absolute,
            &ctx(),
        )
        .unwrap();
        tiny.kth_distances(&[3e-21], 1, 1, &mut out).unwrap();
        assert!((out[0] - 3e-21).abs() < 1e-33, "{}", out[0]);
    }

    #[test]
    #[allow(
        clippy::float_cmp,
        reason = "the expected values are small integers or dyadic fractions reproduced exactly in f64"
    )]
    fn self_distances_exclude_by_index_and_count_duplicates() {
        // Donors 0, 1, 1, 3. For donor 1 (index 1): others are 0, 1, 3 at distances 1, 0, 2,
        // so its 1st nearest other is the duplicate at 0 and its 2nd is at 1.
        let idx = MatchingIndex::exact_within(
            &[0.0, 1.0, 1.0, 3.0],
            1,
            &[0, 1, 2, 3],
            MatchingDistance::Absolute,
            &ctx(),
        )
        .unwrap();
        let mut k1 = [0.0; 4];
        idx.kth_self_distances_of_donors_within(1, &mut k1, &ctx()).unwrap();
        assert_eq!(k1, [1.0, 0.0, 0.0, 2.0]);
        let mut k2 = [0.0; 4];
        idx.kth_self_distances_of_donors_within(2, &mut k2, &ctx()).unwrap();
        assert_eq!(k2, [1.0, 1.0, 1.0, 2.0]);
    }

    #[test]
    fn differential_vs_scalar() {
        let donors = [0.0, 0.0, 2.0, 0.0, 0.0, 3.0];
        let query = [0.1, 0.0];
        let idx = MatchingIndex::exact_within(
            &donors,
            2,
            &[0, 1, 2],
            MatchingDistance::Euclidean,
            &ctx(),
        )
        .unwrap();
        let (row, d) = idx.nearest(&query, None).unwrap().unwrap();
        let (si, sd) = nearest_euclidean_scalar(&query, &donors, 3, 2).unwrap();
        assert_eq!(row, si);
        assert!((d - sd).abs() < 1e-12);
    }

    /// Continuous features of `n` rows (distinct with probability one).
    fn continuous(rng: &mut antecedent_core::CausalRng, len: usize) -> Vec<f64> {
        (0..len).map(|_| rng.next_f64() * 4.0 - 2.0).collect()
    }

    /// Match and self-distance workloads on `n` donors, checked against the exhaustive scan
    /// on a sample of rows (the scan is `O(n)` per row).
    fn exercise_at_scale(n: usize, sampled: usize) {
        let ctx = ExecutionContext::production(0x5CA1E, 4);
        let mut rng = ctx.rng.stream_for(StreamDomain::Test, 0x14);
        let step = n / sampled;

        // One dimension: sorted search.
        let donors = continuous(&mut rng, n);
        let rows: Vec<usize> = (0..n).collect();
        let index =
            MatchingIndex::exact_within(&donors, 1, &rows, MatchingDistance::Absolute, &ctx)
                .unwrap();
        let queries = continuous(&mut rng, n);
        let mut matched_rows = vec![0usize; n];
        let mut matched_d = vec![0.0; n];
        let matched = index
            .match_all_within(&queries, n, Some(0.5), &mut matched_rows, &mut matched_d, &ctx)
            .unwrap();
        assert!(matched > 0);
        for q in (0..n).step_by(step) {
            let expect = scan_nearest(&index, &queries[q..=q], Some(0.5));
            let got = (matched_rows[q] != usize::MAX).then(|| (matched_rows[q], matched_d[q]));
            assert_eq!(bits(got), bits(expect), "1-d query {q}");
        }

        // Four dimensions: k-d tree match and k-th self-distances.
        let dim = 4usize;
        let donors = continuous(&mut rng, n * dim);
        let index =
            MatchingIndex::exact_within(&donors, dim, &rows, MatchingDistance::Euclidean, &ctx)
                .unwrap();
        let queries = continuous(&mut rng, n * dim);
        let mut matched_rows = vec![0usize; n];
        let mut matched_d = vec![0.0; n];
        index.match_all_within(&queries, n, None, &mut matched_rows, &mut matched_d, &ctx).unwrap();
        for q in (0..n).step_by(step) {
            let expect = scan_nearest(&index, &queries[q * dim..(q + 1) * dim], None);
            assert_eq!(bits(Some((matched_rows[q], matched_d[q]))), bits(expect), "4-d query {q}");
        }
        let mut kth = vec![0.0; n];
        MatchingIndex::kth_self_distances_within(
            &donors,
            n,
            dim,
            MatchingDistance::Euclidean,
            5,
            &mut kth,
            &ctx,
        )
        .unwrap();
        for (q, expected) in
            scan_kth_self(&donors, n, dim, MatchingDistance::Euclidean, 5, (0..n).step_by(step))
        {
            assert_eq!(kth[q].to_bits(), expected.to_bits(), "k-th self-distance {q}");
        }
    }

    #[test]
    fn workloads_beyond_the_former_ten_thousand_donor_limit() {
        exercise_at_scale(20_000, 100);
    }

    #[test]
    fn one_hundred_thousand_donor_acceptance() {
        let started = std::time::Instant::now();
        exercise_at_scale(100_000, 200);
        eprintln!("100k-donor matching acceptance took {:?}", started.elapsed());
    }
}
