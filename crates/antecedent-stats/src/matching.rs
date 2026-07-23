//! Exact nearest-neighbor matching index (small-n path).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::cast_possible_truncation)]

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
    /// Shape mismatch or donor count exceeding [`EXACT_MATCHING_ROW_LIMIT`].
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
    /// `None` when the nearest distance exceeds `c`.
    ///
    /// # Errors
    ///
    /// Query length mismatch or empty index.
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
        if let Some(c) = caliper {
            if best_d > c {
                return Ok(None);
            }
        }
        Ok(Some((self.donor_rows[best_i], best_d)))
    }

    /// Mean distance to the `k`-th nearest donor for each query row (row-major queries).
    ///
    /// # Errors
    ///
    /// Shape mismatch or `k == 0`.
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
        if n_donors <= k {
            return Err(StatsError::Shape { message: "not enough donors for k" });
        }
        let mut dists = vec![0.0; n_donors];
        for q in 0..n_queries {
            let query = &queries_rowmajor[q * self.dim..(q + 1) * self.dim];
            for (i, _) in self.donor_rows.iter().enumerate() {
                let row = &self.features[i * self.dim..(i + 1) * self.dim];
                dists[i] = match self.distance {
                    MatchingDistance::Euclidean => euclidean(query, row),
                    MatchingDistance::Absolute => (query[0] - row[0]).abs(),
                };
            }
            // Skip self-match at distance 0 when query is a donor; k-th among others ≈ index k
            // when self is present as exact 0.
            let min = dists.iter().copied().fold(f64::INFINITY, f64::min);
            let idx = (if min < 1e-15 { k } else { k - 1 }).min(dists.len() - 1);
            let (_, kth, _) = dists.select_nth_unstable_by(idx, |a, b| {
                a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal)
            });
            out[q] = *kth;
        }
        Ok(())
    }
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
