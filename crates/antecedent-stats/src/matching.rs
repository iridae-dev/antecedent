//! Exact nearest-neighbor matching index (small-n path).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use crate::error::StatsError;

/// Soft upper bound for the exact (brute-force) matching path.
pub const EXACT_MATCHING_ROW_LIMIT: usize = 10_000;

/// Distance metric for matching.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
pub enum MatchingDistance {
    /// Euclidean distance on feature rows.
    Euclidean,
    /// Absolute difference on a single scalar (e.g. propensity).
    Absolute,
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
}

impl MatchingIndex {
    /// Build an exact index from donor feature rows.
    ///
    /// `features_rowmajor` length must be `donor_rows.len() * dim`.
    ///
    /// # Errors
    ///
    /// Shape mismatch, donor count exceeding [`EXACT_MATCHING_ROW_LIMIT`], or a non-finite
    /// donor feature (a NaN distance compares false against everything, so such a donor
    /// could never be matched and would silently corrupt every nearest-neighbour scan).
    pub fn exact(
        features_rowmajor: &[f64],
        dim: usize,
        donor_rows: &[usize],
        distance: MatchingDistance,
    ) -> Result<Self, StatsError> {
        let n = donor_rows.len();
        if n > EXACT_MATCHING_ROW_LIMIT {
            return Err(StatsError::Shape {
                message: "donor count exceeds exact matching row limit",
            });
        }
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
        Ok(Self {
            dim,
            features: features_rowmajor.to_vec(),
            donor_rows: donor_rows.to_vec(),
            distance,
        })
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
        u64::try_from(f + d).unwrap_or(u64::MAX)
    }

    /// Find the nearest donor to `query` (length `dim`).
    ///
    /// Returns `(donor_original_row, distance)`. If `caliper` is `Some(c)`, returns
    /// `None` when the nearest distance exceeds `c`. Returns `None` as well when no donor
    /// is at a finite distance (the squared differences overflowed): there is no nearest
    /// donor to report.
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
        let mut best_i = 0usize;
        let mut best_d = f64::INFINITY;
        for (i, _) in self.donor_rows.iter().enumerate() {
            let row = &self.features[i * self.dim..(i + 1) * self.dim];
            let d = match self.distance {
                MatchingDistance::Euclidean => euclidean(query, row),
                MatchingDistance::Absolute => (query[0] - row[0]).abs(),
            };
            if d < best_d {
                best_d = d;
                best_i = i;
            }
        }
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

    /// `k`-th self-distances of a matrix against itself without building an index.
    ///
    /// Equivalent to `MatchingIndex::exact(features, dim, &(0..n).collect(), distance)`
    /// followed by [`Self::kth_self_distances_of_donors`] — the loop body is identical, so
    /// results match bit for bit — but with no feature/donor copies. Each row is excluded
    /// from its own neighbour set by index; other rows at distance zero (duplicates) count. `dist_scratch` is a reused per-query distance buffer. Permutation-null
    /// loops that vary one column of `features_rowmajor` between calls use this form.
    ///
    /// # Errors
    ///
    /// Shape mismatch, `k == 0`, `n <= k`, non-finite features, or `n` over
    /// [`EXACT_MATCHING_ROW_LIMIT`].
    pub fn kth_self_distances(
        features_rowmajor: &[f64],
        n: usize,
        dim: usize,
        distance: MatchingDistance,
        k: usize,
        out: &mut [f64],
        dist_scratch: &mut Vec<f64>,
    ) -> Result<(), StatsError> {
        if n > EXACT_MATCHING_ROW_LIMIT {
            return Err(StatsError::Shape {
                message: "donor count exceeds exact matching row limit",
            });
        }
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
        if dist_scratch.len() < n {
            dist_scratch.resize(n, 0.0);
        }
        let dists = &mut dist_scratch[..n];
        for q in 0..n {
            let query = &features_rowmajor[q * dim..(q + 1) * dim];
            fill_distances(dists, query, features_rowmajor, dim, distance);
            out[q] = kth_excluding_self(dists, q, k);
        }
        Ok(())
    }

    /// Distance to the `k`-th nearest donor for each query row (row-major queries), where
    /// the queries are **external** to the donor set: no donor is skipped, so a query that
    /// coincides with a donor value matches it at distance zero.
    ///
    /// For queries that *are* the donor rows use [`Self::kth_self_distances_of_donors`].
    ///
    /// # Errors
    ///
    /// Shape mismatch, `k == 0`, too few donors, or a non-finite query.
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
    /// `k == 0`, `len() <= k`, or `out` shorter than the donor count.
    pub fn kth_self_distances_of_donors(
        &self,
        k: usize,
        out: &mut [f64],
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
        let mut dists = vec![0.0; n];
        for q in 0..n {
            let query = &self.features[q * self.dim..(q + 1) * self.dim];
            fill_distances(&mut dists, query, &self.features, self.dim, self.distance);
            out[q] = kth_excluding_self(&mut dists, q, k);
        }
        Ok(())
    }

    /// Match every query row to its nearest donor.
    ///
    /// `queries_rowmajor` length = `n_queries * dim`.
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
        if queries_rowmajor.len() != n_queries.saturating_mul(self.dim) {
            return Err(StatsError::Shape { message: "queries length != n_queries * dim" });
        }
        if out_donor_row.len() < n_queries || out_distance.len() < n_queries {
            return Err(StatsError::Shape { message: "output buffers too short" });
        }
        let mut matched = 0u32;
        for q in 0..n_queries {
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

/// Distances from `query` to every row of row-major `features`.
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

/// `k`-th smallest distance (1-based) among all entries except `self_index`.
fn kth_excluding_self(dists: &mut [f64], self_index: usize, k: usize) -> f64 {
    // Parking the excluded entry below every real distance makes 0-based rank `k` among the
    // rest exactly the `k`-th nearest other donor.
    dists[self_index] = f64::NEG_INFINITY;
    let (_, kth, _) = dists.select_nth_unstable_by(k, f64::total_cmp);
    *kth
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

    #[test]
    fn finds_nearest_euclidean() {
        let donors = [0.0, 0.0, 1.0, 1.0, 5.0, 5.0];
        let idx =
            MatchingIndex::exact(&donors, 2, &[10, 20, 30], MatchingDistance::Euclidean).unwrap();
        let (row, d) = idx.nearest(&[0.1, 0.1], None).unwrap().unwrap();
        assert_eq!(row, 10);
        assert!(d < 0.2);
    }

    #[test]
    fn caliper_rejects_far_matches() {
        let donors = [0.0, 10.0];
        let idx = MatchingIndex::exact(&donors, 1, &[0, 1], MatchingDistance::Absolute).unwrap();
        assert!(idx.nearest(&[0.05], Some(0.1)).unwrap().is_some());
        assert!(idx.nearest(&[5.0], Some(0.1)).unwrap().is_none());
    }

    #[test]
    fn kth_self_distances_matches_index_path_bit_for_bit() {
        // The kNN permutation null uses the index-free form; it must reproduce
        // MatchingIndex::exact + kth_distances exactly.
        let n = 40usize;
        let dim = 3usize;
        let feats: Vec<f64> =
            (0..n * dim).map(|i| ((i * 17 + 3) % 29) as f64 * 0.37 - 4.0).collect();
        let donors: Vec<usize> = (0..n).collect();
        let idx = MatchingIndex::exact(&feats, dim, &donors, MatchingDistance::Euclidean).unwrap();
        for k in [1usize, 3, 7] {
            let mut via_index = vec![0.0; n];
            idx.kth_self_distances_of_donors(k, &mut via_index).unwrap();
            let mut direct = vec![0.0; n];
            let mut scratch = Vec::new();
            MatchingIndex::kth_self_distances(
                &feats,
                n,
                dim,
                MatchingDistance::Euclidean,
                k,
                &mut direct,
                &mut scratch,
            )
            .unwrap();
            for (q, (a, b)) in via_index.iter().zip(&direct).enumerate() {
                assert!(a.to_bits() == b.to_bits(), "k={k} query {q}: {a:?} vs {b:?}");
            }
        }
    }

    #[test]
    fn nan_and_infinite_features_are_refused_not_matched() {
        assert!(
            MatchingIndex::exact(&[0.0, f64::NAN], 1, &[0, 1], MatchingDistance::Absolute).is_err()
        );
        assert!(
            MatchingIndex::exact(&[f64::INFINITY, 1.0], 1, &[0, 1], MatchingDistance::Absolute)
                .is_err()
        );
        let idx =
            MatchingIndex::exact(&[0.0, 1.0], 1, &[7, 8], MatchingDistance::Absolute).unwrap();
        // A NaN query used to compare false against every donor and match donor 0 at +inf.
        assert!(idx.nearest(&[f64::NAN], None).is_err());
        let mut rows = [0usize; 1];
        let mut dist = [0.0; 1];
        assert!(idx.match_all(&[f64::NAN], 1, None, &mut rows, &mut dist).is_err());
        // Squared differences overflow: no donor is at a finite distance, so no match.
        let far = MatchingIndex::exact(&[1e200], 1, &[3], MatchingDistance::Euclidean).unwrap();
        assert_eq!(far.nearest(&[-1e200], None).unwrap(), None);
    }

    #[test]
    fn external_query_equal_to_a_donor_matches_it_at_distance_zero() {
        // Queries that are not the donor set must not have "their" zero-distance donor
        // skipped: the 1st neighbour of a query sitting on donor value 1.0 is at 0, not 1.
        let idx = MatchingIndex::exact(
            &[0.0, 1.0, 2.0, 3.0],
            1,
            &[0, 1, 2, 3],
            MatchingDistance::Absolute,
        )
        .unwrap();
        let mut out = [f64::NAN; 1];
        idx.kth_distances(&[1.0], 1, 1, &mut out).unwrap();
        assert_eq!(out[0], 0.0);
        idx.kth_distances(&[1.0], 1, 2, &mut out).unwrap();
        assert_eq!(out[0], 1.0);
        // Unit-free: the old absolute 1e-15 test misclassified small-unit queries.
        let tiny =
            MatchingIndex::exact(&[0.0, 1e-20], 1, &[0, 1], MatchingDistance::Absolute).unwrap();
        tiny.kth_distances(&[3e-21], 1, 1, &mut out).unwrap();
        assert!((out[0] - 3e-21).abs() < 1e-33, "{}", out[0]);
    }

    #[test]
    fn self_distances_exclude_by_index_and_count_duplicates() {
        // Donors 0, 1, 1, 3. For donor 1 (index 1): others are 0, 1, 3 at distances 1, 0, 2,
        // so its 1st nearest other is the duplicate at 0 and its 2nd is at 1.
        let idx = MatchingIndex::exact(
            &[0.0, 1.0, 1.0, 3.0],
            1,
            &[0, 1, 2, 3],
            MatchingDistance::Absolute,
        )
        .unwrap();
        let mut k1 = [0.0; 4];
        idx.kth_self_distances_of_donors(1, &mut k1).unwrap();
        assert_eq!(k1, [1.0, 0.0, 0.0, 2.0]);
        let mut k2 = [0.0; 4];
        idx.kth_self_distances_of_donors(2, &mut k2).unwrap();
        assert_eq!(k2, [1.0, 1.0, 1.0, 2.0]);
    }

    #[test]
    fn differential_vs_scalar() {
        let donors = [0.0, 0.0, 2.0, 0.0, 0.0, 3.0];
        let query = [0.1, 0.0];
        let idx =
            MatchingIndex::exact(&donors, 2, &[0, 1, 2], MatchingDistance::Euclidean).unwrap();
        let (row, d) = idx.nearest(&query, None).unwrap().unwrap();
        let (si, sd) = nearest_euclidean_scalar(&query, &donors, 3, 2).unwrap();
        assert_eq!(row, si);
        assert!((d - sd).abs() < 1e-12);
    }
}
