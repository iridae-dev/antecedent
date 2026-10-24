//! Exact multivariate nearest-neighbour search over a bounding-box tree.
//!
//! This is a second exact spatial index for Euclidean donor rows, built beside
//! [`MatchingIndex`](crate::matching::MatchingIndex) so the two can be measured against each
//! other. [`MatchingIndex`](crate::matching::MatchingIndex) prunes with the distance to one
//! splitting plane per node. This tree stores the axis-aligned bounding box of every node and
//! prunes with the distance to the whole box, which stays tight as the dimension grows and
//! when the donors lie on a lower-dimensional surface.
//!
//! Answers are the ones of a linear scan over the donors in order: the strictly smallest
//! distance wins and, among equal distances, the lowest donor position. A subtree is skipped
//! only when its box is strictly farther than the current `k`-th candidate, and the box
//! distance is computed with the same floating-point operations as the donor distance
//! (per-axis gap, square, left-to-right sum, square root). Every one of those operations is
//! monotone, so no donor inside a skipped box can be nearer than, or tied with, a kept
//! candidate.
//!
//! Memory is leased from the [`ExecutionContext`] before anything is allocated, so a donor set
//! that does not fit is refused by the caller's budget exactly as the other index is.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use antecedent_core::ExecutionContext;
use antecedent_core::execution::AdmissionLease;

use crate::error::StatsError;
use crate::matching::check_budget;

/// Donors at or below which a node is a leaf scanned directly.
const BOX_LEAF: usize = 16;

/// Queries answered between cancellation checks.
const QUERY_CHUNK: usize = 1024;

/// Reusable exact Euclidean index over donor feature rows, pruned by node bounding boxes.
#[derive(Debug)]
pub struct BoxTreeIndex {
    dim: usize,
    /// Row-major donor features: `n_donors * dim`.
    features: Vec<f64>,
    /// Donor positions in tree order.
    perm: Vec<usize>,
    /// Lower corner of every node's bounding box, `dim` values per heap-ordered node.
    low: Vec<f64>,
    /// Upper corner of every node's bounding box, laid out like `low`.
    high: Vec<f64>,
    /// Holds the bytes of this index on the caller's admission pool until the index drops.
    _lease: AdmissionLease,
}

impl BoxTreeIndex {
    /// Bytes an index over `n` donors of `dim` features retains, or `None` when that size
    /// does not fit in `u64`. This is the figure the build leases before it allocates.
    #[must_use]
    pub fn quote_bytes(n: usize, dim: usize) -> Option<u64> {
        let word = core::mem::size_of::<f64>();
        let features = n.checked_mul(dim)?.checked_mul(word)?;
        let perm = n.checked_mul(core::mem::size_of::<usize>())?;
        let boxes = heap_slots(n)?.checked_mul(dim)?.checked_mul(2)?.checked_mul(word)?;
        u64::try_from(features.checked_add(perm)?.checked_add(boxes)?).ok()
    }

    /// Build the index over the `n = features_rowmajor.len() / dim` donor rows.
    ///
    /// # Errors
    ///
    /// Shape mismatch, a non-finite donor feature, cancellation, or a footprint above the
    /// context's memory limit or its free admission bytes.
    pub fn exact_within(
        features_rowmajor: &[f64],
        dim: usize,
        ctx: &ExecutionContext,
    ) -> Result<Self, StatsError> {
        if dim == 0 {
            return Err(StatsError::Shape { message: "matching dim must be > 0" });
        }
        if features_rowmajor.len() % dim != 0 {
            return Err(StatsError::Shape { message: "features length != n_donors * dim" });
        }
        if features_rowmajor.iter().any(|v| !v.is_finite()) {
            return Err(StatsError::Shape { message: "matching features must be finite" });
        }
        let n = features_rowmajor.len() / dim;
        let lease = check_budget(ctx, "bounding-box matching index", Self::quote_bytes(n, dim))?;
        let slots = heap_slots(n).ok_or(StatsError::Shape { message: "donor count overflows" })?;
        let mut index = Self {
            dim,
            features: features_rowmajor.to_vec(),
            perm: (0..n).collect(),
            low: vec![0.0; slots * dim],
            high: vec![0.0; slots * dim],
            _lease: lease,
        };
        if n > 0 {
            index.build_node(0, 0, n, ctx)?;
        }
        Ok(index)
    }

    /// Number of donors.
    #[must_use]
    pub fn len(&self) -> usize {
        self.perm.len()
    }

    /// Whether the index holds no donors.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.perm.is_empty()
    }

    /// Feature dimension.
    #[must_use]
    pub const fn dim(&self) -> usize {
        self.dim
    }

    /// Bytes this index retains.
    #[must_use]
    pub fn retained_memory_bytes(&self) -> u64 {
        let word = core::mem::size_of::<f64>();
        let bytes = (self.features.capacity() + self.low.capacity() + self.high.capacity()) * word
            + self.perm.capacity() * core::mem::size_of::<usize>();
        u64::try_from(bytes).unwrap_or(u64::MAX)
    }

    fn row(&self, position: usize) -> &[f64] {
        &self.features[position * self.dim..(position + 1) * self.dim]
    }

    fn build_node(
        &mut self,
        node: usize,
        lo: usize,
        hi: usize,
        ctx: &ExecutionContext,
    ) -> Result<(), StatsError> {
        let dim = self.dim;
        let mut low = vec![f64::INFINITY; dim];
        let mut high = vec![f64::NEG_INFINITY; dim];
        for &position in &self.perm[lo..hi] {
            let row = &self.features[position * dim..(position + 1) * dim];
            for axis in 0..dim {
                low[axis] = low[axis].min(row[axis]);
                high[axis] = high[axis].max(row[axis]);
            }
        }
        self.low[node * dim..(node + 1) * dim].copy_from_slice(&low);
        self.high[node * dim..(node + 1) * dim].copy_from_slice(&high);
        if hi - lo <= BOX_LEAF {
            return Ok(());
        }
        if hi - lo >= 4 * QUERY_CHUNK && ctx.cancellation.is_cancelled() {
            return Err(StatsError::Cancelled);
        }
        let mut axis = 0;
        let mut widest = f64::NEG_INFINITY;
        for candidate in 0..dim {
            let extent = high[candidate] - low[candidate];
            if extent > widest {
                widest = extent;
                axis = candidate;
            }
        }
        let mid = lo + (hi - lo) / 2;
        let features = &self.features;
        self.perm[lo..hi].select_nth_unstable_by(mid - lo, |&a, &b| {
            features[a * dim + axis].total_cmp(&features[b * dim + axis])
        });
        self.build_node(2 * node + 1, lo, mid, ctx)?;
        self.build_node(2 * node + 2, mid, hi, ctx)
    }

    /// Distance from `query` to the box of `node`, by the donor distance's own operations.
    fn box_distance(&self, query: &[f64], node: usize) -> f64 {
        let dim = self.dim;
        let low = &self.low[node * dim..(node + 1) * dim];
        let high = &self.high[node * dim..(node + 1) * dim];
        query
            .iter()
            .zip(low.iter().zip(high))
            .map(|(&q, (&l, &h))| {
                let gap = if q < l {
                    l - q
                } else if q > h {
                    q - h
                } else {
                    0.0
                };
                gap * gap
            })
            .sum::<f64>()
            .sqrt()
    }

    fn search_node(&self, query: &[f64], found: &mut Nearest, node: usize, lo: usize, hi: usize) {
        if hi - lo <= BOX_LEAF {
            for &position in &self.perm[lo..hi] {
                found.offer(euclidean(query, self.row(position)), position);
            }
            return;
        }
        let mid = lo + (hi - lo) / 2;
        let (left, right) = (2 * node + 1, 2 * node + 2);
        let (left_bound, right_bound) =
            (self.box_distance(query, left), self.box_distance(query, right));
        let ((first, first_range, first_bound), (second, second_range, second_bound)) =
            if left_bound <= right_bound {
                ((left, (lo, mid), left_bound), (right, (mid, hi), right_bound))
            } else {
                ((right, (mid, hi), right_bound), (left, (lo, mid), left_bound))
            };
        if found.bound().is_none_or(|worst| first_bound <= worst) {
            self.search_node(query, found, first, first_range.0, first_range.1);
        }
        if found.bound().is_none_or(|worst| second_bound <= worst) {
            self.search_node(query, found, second, second_range.0, second_range.1);
        }
    }

    fn check_query(&self, query: &[f64]) -> Result<(), StatsError> {
        if query.len() != self.dim {
            return Err(StatsError::Shape { message: "query length != dim" });
        }
        if query.iter().any(|v| !v.is_finite()) {
            return Err(StatsError::Shape { message: "matching query must be finite" });
        }
        Ok(())
    }

    /// The `k` nearest donors to `query`, nearest first, as `(donor position, distance)` in
    /// `out` (cleared first). Equal distances resolve to the lower position. Fewer than `k`
    /// pairs come back when the index holds fewer donors. Memory is `O(k)`.
    ///
    /// # Errors
    ///
    /// Query length mismatch or a non-finite query.
    pub fn nearest_k(
        &self,
        query: &[f64],
        k: usize,
        out: &mut Vec<(usize, f64)>,
    ) -> Result<(), StatsError> {
        self.check_query(query)?;
        let mut found = Nearest::new(k.min(self.perm.len()));
        if !self.perm.is_empty() && found.k > 0 {
            self.search_node(query, &mut found, 0, 0, self.perm.len());
        }
        out.clear();
        out.extend(found.into_sorted());
        Ok(())
    }

    /// Nearest donor to `query` as `(donor position, distance)`, or `None` when the index is
    /// empty, the nearest distance is not finite, or it exceeds `caliper`.
    ///
    /// # Errors
    ///
    /// Query length mismatch or a non-finite query.
    pub fn nearest(
        &self,
        query: &[f64],
        caliper: Option<f64>,
    ) -> Result<Option<(usize, f64)>, StatsError> {
        self.check_query(query)?;
        if self.perm.is_empty() {
            return Ok(None);
        }
        let mut found = Nearest::new(1);
        self.search_node(query, &mut found, 0, 0, self.perm.len());
        let Some((position, distance)) = found.best() else {
            return Ok(None);
        };
        if !distance.is_finite() || caliper.is_some_and(|limit| distance > limit) {
            return Ok(None);
        }
        Ok(Some((position, distance)))
    }

    /// Match every query row (row-major) to its nearest donor.
    ///
    /// `out_donor_row[q]` is `usize::MAX` and `out_distance[q]` is `∞` when no donor is within
    /// the caliper. Returns the number of matched queries.
    ///
    /// # Errors
    ///
    /// Shape mismatch, a non-finite query, or cancellation.
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
        for (q, query) in queries_rowmajor.chunks_exact(self.dim).enumerate() {
            if q % QUERY_CHUNK == 0 && ctx.cancellation.is_cancelled() {
                return Err(StatsError::Cancelled);
            }
            if let Some((row, distance)) = self.nearest(query, caliper)? {
                out_donor_row[q] = row;
                out_distance[q] = distance;
                matched = matched.saturating_add(1);
            } else {
                out_donor_row[q] = usize::MAX;
                out_distance[q] = f64::INFINITY;
            }
        }
        Ok(matched)
    }
}

/// Heap-ordered node slots of a balanced tree over `n` donors: `2^(levels + 1) - 1`, where
/// `levels` halvings (rounding up) bring a span to at most [`BOX_LEAF`] donors.
fn heap_slots(n: usize) -> Option<usize> {
    let mut size = n;
    let mut levels = 0u32;
    while size > BOX_LEAF {
        size = size.div_ceil(2);
        levels += 1;
    }
    1usize.checked_shl(levels + 1)?.checked_sub(1)
}

fn euclidean(a: &[f64], b: &[f64]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let d = x - y;
            d * d
        })
        .sum::<f64>()
        .sqrt()
}

/// A candidate ordered by `(distance, position)`.
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

    /// Distance of the current `k`-th candidate once `k` have been seen.
    fn bound(&self) -> Option<f64> {
        if self.k == 0 || self.heap.len() < self.k {
            return None;
        }
        self.heap.peek().map(|worst| worst.distance)
    }

    fn best(&self) -> Option<(usize, f64)> {
        self.heap.iter().min().map(|entry| (entry.position, entry.distance))
    }

    fn into_sorted(self) -> Vec<(usize, f64)> {
        self.heap.into_sorted_vec().into_iter().map(|e| (e.position, e.distance)).collect()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::matching::{MatchingDistance, MatchingIndex};

    /// Linear scan with the documented tie rule, the reference every answer is checked against.
    fn scan(features: &[f64], dim: usize, query: &[f64], k: usize) -> Vec<(usize, f64)> {
        let mut all: Vec<(usize, f64)> = features
            .chunks_exact(dim)
            .enumerate()
            .map(|(position, row)| (position, euclidean(query, row)))
            .collect();
        all.sort_by(|a, b| a.1.total_cmp(&b.1).then(a.0.cmp(&b.0)));
        all.truncate(k);
        all
    }

    fn lattice(n: usize, dim: usize, levels: u64, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n * dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 33) % levels) as f64
            })
            .collect()
    }

    fn spread(n: usize, dim: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n * dim)
            .map(|_| {
                state = state
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((state >> 11) as f64) / ((1u64 << 53) as f64)
            })
            .collect()
    }

    fn assert_same(got: &[(usize, f64)], want: &[(usize, f64)]) {
        assert_eq!(got.len(), want.len());
        for (g, w) in got.iter().zip(want) {
            assert_eq!(g.0, w.0);
            assert_eq!(g.1.to_bits(), w.1.to_bits());
        }
    }

    #[test]
    fn answers_equal_the_linear_scan_including_ties_at_every_dimension() {
        let ctx = ExecutionContext::for_tests(1);
        for dim in [1usize, 2, 3, 5, 8, 12] {
            // A coarse lattice makes many donors tie at the same distance.
            for (name, donors) in [
                ("lattice", lattice(400, dim, 4, 17 + dim as u64)),
                ("continuous", spread(400, dim, 29 + dim as u64)),
            ] {
                let index = BoxTreeIndex::exact_within(&donors, dim, &ctx).unwrap();
                assert_eq!(index.len(), 400);
                let queries: Vec<f64> =
                    if name == "lattice" { lattice(60, dim, 6, 91) } else { spread(60, dim, 91) };
                let mut out = Vec::new();
                for query in queries.chunks_exact(dim).chain(donors.chunks_exact(dim).take(25)) {
                    for k in [1usize, 3, 7, 40] {
                        index.nearest_k(query, k, &mut out).unwrap();
                        assert_same(&out, &scan(&donors, dim, query, k));
                    }
                }
            }
        }
    }

    #[test]
    fn nearest_and_match_all_agree_with_the_tree_index_under_calipers() {
        let ctx = ExecutionContext::for_tests(1);
        for dim in [2usize, 4, 9] {
            let donors = lattice(300, dim, 5, 3);
            let queries = lattice(120, dim, 7, 5);
            let ids: Vec<usize> = (0..300).collect();
            let kd =
                MatchingIndex::exact_within(&donors, dim, &ids, MatchingDistance::Euclidean, &ctx)
                    .unwrap();
            let boxed = BoxTreeIndex::exact_within(&donors, dim, &ctx).unwrap();
            for caliper in [None, Some(0.0), Some(1.0), Some(2.5)] {
                let (mut kd_rows, mut kd_dist) = (vec![0usize; 120], vec![0.0; 120]);
                let (mut box_rows, mut box_dist) = (vec![0usize; 120], vec![0.0; 120]);
                let kd_matched = kd
                    .match_all_within(&queries, 120, caliper, &mut kd_rows, &mut kd_dist, &ctx)
                    .unwrap();
                let box_matched = boxed
                    .match_all_within(&queries, 120, caliper, &mut box_rows, &mut box_dist, &ctx)
                    .unwrap();
                assert_eq!(kd_matched, box_matched);
                assert_eq!(kd_rows, box_rows);
                for (a, b) in kd_dist.iter().zip(&box_dist) {
                    assert_eq!(a.to_bits(), b.to_bits());
                }
            }
        }
    }

    #[test]
    fn duplicates_resolve_to_the_lowest_position_and_edge_inputs_are_refused() {
        let ctx = ExecutionContext::for_tests(1);
        let donors = vec![2.0; 50 * 3];
        let index = BoxTreeIndex::exact_within(&donors, 3, &ctx).unwrap();
        assert_eq!(index.nearest(&[2.0, 2.0, 2.0], None).unwrap(), Some((0, 0.0)));
        let mut out = Vec::new();
        index.nearest_k(&[0.0, 0.0, 0.0], 3, &mut out).unwrap();
        assert_eq!(out.iter().map(|p| p.0).collect::<Vec<_>>(), vec![0, 1, 2]);
        index.nearest_k(&[0.0, 0.0, 0.0], 0, &mut out).unwrap();
        assert!(out.is_empty());
        assert!(index.nearest(&[1.0, 2.0], None).is_err());
        assert!(index.nearest(&[f64::NAN, 0.0, 0.0], None).is_err());
        assert!(BoxTreeIndex::exact_within(&[1.0, f64::INFINITY], 2, &ctx).is_err());
        assert!(BoxTreeIndex::exact_within(&[1.0, 2.0, 3.0], 2, &ctx).is_err());
        assert!(BoxTreeIndex::exact_within(&[1.0], 0, &ctx).is_err());
        let empty = BoxTreeIndex::exact_within(&[], 2, &ctx).unwrap();
        assert!(empty.is_empty());
        assert_eq!(empty.nearest(&[0.0, 0.0], None).unwrap(), None);
    }

    #[test]
    fn index_memory_is_quoted_leased_before_allocation_and_returned_on_drop() {
        let (n, dim) = (1_000usize, 6usize);
        let donors = spread(n, dim, 7);
        let quote = BoxTreeIndex::quote_bytes(n, dim).unwrap();

        let mut refused = ExecutionContext::for_tests(1);
        refused.set_memory_limit(Some(quote - 1));
        match BoxTreeIndex::exact_within(&donors, dim, &refused) {
            Err(StatsError::ResourceLimit { required_bytes, limit_bytes, .. }) => {
                assert_eq!((required_bytes, limit_bytes), (quote, quote - 1));
            }
            other => panic!("expected a resource refusal, got {other:?}"),
        }
        assert_eq!(refused.admission().available().bytes(), quote - 1);

        let mut exact = ExecutionContext::for_tests(1);
        exact.set_memory_limit(Some(quote));
        let index = BoxTreeIndex::exact_within(&donors, dim, &exact).unwrap();
        assert_eq!(exact.admission().available().bytes(), 0);
        assert!(index.retained_memory_bytes() <= quote);
        assert!(index.retained_memory_bytes() >= quote / 2);
        drop(index);
        assert_eq!(exact.admission().available().bytes(), quote);
    }

    #[test]
    fn a_far_cluster_box_is_strictly_farther_than_the_answer_so_it_is_never_searched() {
        let dim = 4usize;
        let mut donors = spread(500, dim, 11);
        donors.extend(spread(500, dim, 12).iter().map(|v| v + 1_000.0));
        let ctx = ExecutionContext::for_tests(1);
        let index = BoxTreeIndex::exact_within(&donors, dim, &ctx).unwrap();
        let query = [0.5; 4];
        let mut found = Nearest::new(1);
        index.search_node(&query, &mut found, 0, 0, index.len());
        let (position, distance) = found.best().unwrap();
        let want = scan(&donors, dim, &query, 1)[0];
        assert_eq!((position, distance.to_bits()), (want.0, want.1.to_bits()));
        // The median split puts one cluster under each child of the root.
        assert!(index.box_distance(&query, 1) <= distance);
        assert!(index.box_distance(&query, 2) > distance);
    }
}
