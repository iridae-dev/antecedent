//! kNN distance-dependence conditional independence tests (plain and mixed-data).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::ci::block_shuffle::{add_one_pvalue, permutation_null_exceedances, query_stream_salt};
use crate::ci::gsquared::ensure_permutation_support;
use crate::ci::types::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependenceTest,
    KnnDependenceWorkspace, PreparedCiTest, nonparametric_permutation_count,
    reject_unsupported_block_size, requested_block_size,
};
use crate::error::StatsError;
use crate::matching::{MatchingDistance, MatchingIndex};
use antecedent_core::{ExecutionContext, StreamDomain};
use std::collections::HashMap;

/// kNN distance dependence CI (permutation null).
///
/// **Not** KSG/CMIknn: the statistic is −(mean k-th NN distance) in the joint
/// `(X,Y,Z)` space — a generic dependence proxy for permutation testing.
/// Factory id: `knn_dependence`.
#[derive(Clone, Debug)]
pub struct KnnDependence {
    /// Neighbors.
    pub k: usize,
}

impl Default for KnnDependence {
    fn default() -> Self {
        Self::new(5)
    }
}

impl KnnDependence {
    /// Construct with neighbor count `k`.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self { k: k.max(1) }
    }
}

impl ConditionalIndependenceTest for KnnDependence {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        let request = &prepared.bind_request(request);
        let n = request.columns.first().map_or(0, |c| c.len());
        if n < self.k + 2 {
            return Err(StatsError::Shape { message: "n too small for kNN dependence" });
        }
        if workspace.knn.perm.len() != n {
            workspace.knn.perm = (0..n).collect();
            workspace.knn.index_generation = workspace.knn.index_generation.saturating_add(1);
            workspace.knn.last_n = n;
        }
        if workspace.block_perm.len() != n {
            workspace.block_perm = workspace.knn.perm.clone();
        }
        let n_perm = nonparametric_permutation_count(request.significance);
        let block_size = requested_block_size(request.significance);
        // Null-loop scratch, reused across replicates and queries: the permuted
        // feature matrix differs from the primary index's only in the Y column,
        // so each replicate rewrites that column and computes k-th self-distances
        // directly — no per-replicate index copy and no `n`-length distance buffer.
        // The joint distance changes with every permuted column, so nothing but the
        // feature matrix carries over between replicates.
        let mut null_feats: Vec<f64> = Vec::new();
        let mut null_dists: Vec<f64> = Vec::new();
        let mut results = Vec::with_capacity(request.queries.len());
        for q in request.queries {
            let z = &request.z_flat[q.z_start..q.z_start + q.z_len];
            // Blocking is well defined only for an empty conditioning set, where
            // `z_permutation_strata` degenerates to one stratum holding every row in original time
            // order. With conditioning the strata are Z-level or local-neighbourhood groups
            // scattered across time, so preserving `Y|Z` and preserving serial dependence cannot
            // both hold — that case is rejected rather than silently downgraded to an
            // exchangeable null. Checked per query because `z` varies across a batch.
            if block_size > 1 && !z.is_empty() {
                reject_unsupported_block_size(request.significance, "KnnDependence")?;
            }
            let dim = 2 + z.len();
            ensure_knn_index(request.columns, q.x, q.y, z, n, dim, &mut workspace.knn, ctx)?;
            let builds_before = workspace.knn.index_builds;
            let stat = knn_stat_from_index(&mut workspace.knn, self.k, ctx)?;
            // Null: permute Y within local Z neighbourhoods (or exact Z-level strata when
            // Z is discrete) so the Y–Z link is preserved under H0. A full unconditional
            // shuffle — or coarse tercile bins on continuous Z — inflates type-I error
            // whenever Y depends on Z. See `z_permutation_strata`.
            let strata = z_permutation_strata(request.columns, z, n, self.k)?;
            ensure_permutation_support(&strata, request.columns[q.x], request.columns[q.y])?;
            let mut y_perm = request.columns[q.y].to_vec();
            let mut rng = ctx.rng.stream_for(
                StreamDomain::StatsCi,
                0xC11_u64 ^ query_stream_salt(&[q.x], &[q.y], z),
            );
            // The primary index's feature matrix has exactly the x/y/z layout the
            // null needs; clone it once per query and rewrite the Y column per
            // replicate.
            null_feats.clear();
            null_feats.extend_from_slice(&workspace.knn.features);
            if null_dists.len() < n {
                null_dists.resize(n, 0.0);
            }
            let null_ge = permutation_null_exceedances(
                n_perm,
                &mut y_perm,
                &strata,
                block_size,
                &mut rng,
                |permuted| {
                    for (r, &value) in permuted.iter().enumerate() {
                        null_feats[r * dim + 1] = value;
                    }
                    MatchingIndex::kth_self_distances_within(
                        &null_feats[..n * dim],
                        n,
                        dim,
                        MatchingDistance::Euclidean,
                        self.k,
                        &mut null_dists,
                        ctx,
                    )?;
                    let null = -null_dists[..n].iter().sum::<f64>() / n as f64;
                    Ok::<_, StatsError>(null >= stat)
                },
            )?;
            debug_assert_eq!(workspace.knn.index_builds, builds_before);
            let p = add_one_pvalue(null_ge, n_perm);
            // df is not defined for this distance statistic; leave 0 rather than claim n.
            results.push(CiResult { statistic: stat, p_value: p, df: 0.0, ci: None });
        }
        Ok(CiBatchResult { results })
    }

    fn attainable_p(&self) -> crate::ci::types::AttainableP {
        crate::ci::types::AttainableP::NonparametricPermutation
    }
}

/// Content identity of the columns feeding a kNN index.
///
/// Hashes query indexes and the full contents of each involved column. Pointers
/// are not identity: allocator reuse or an unsampled mutation must miss.
fn knn_input_fingerprint(columns: &[&[f64]], x: usize, y: usize, z: &[usize], n: usize) -> u64 {
    let mut h = antecedent_core::Fnv1a64::new();
    let mut mix = |v: u64| {
        h.fold(v);
    };
    mix(x as u64);
    mix(y as u64);
    mix(z.len() as u64);
    mix(n as u64);
    for &zc in z {
        mix(zc as u64);
    }
    for &c in [x, y].iter().chain(z.iter()) {
        let col = columns[c];
        mix(col.len() as u64);
        for &v in col.iter().take(n) {
            mix(v.to_bits());
        }
    }
    h.finish()
}

#[allow(
    clippy::too_many_arguments,
    reason = "the index key (columns, query, n, dim), the cache it fills and the execution context are all distinct inputs"
)]
fn ensure_knn_index(
    columns: &[&[f64]],
    x: usize,
    y: usize,
    z: &[usize],
    n: usize,
    dim: usize,
    knn: &mut KnnDependenceWorkspace,
    ctx: &ExecutionContext,
) -> Result<(), StatsError> {
    let fingerprint = knn_input_fingerprint(columns, x, y, z, n);
    let need_rebuild = knn.index.is_none()
        || knn.last_dim != dim
        || knn.last_n != n
        || knn.last_fingerprint != fingerprint;
    if !need_rebuild {
        return Ok(());
    }
    let mut feats = vec![0.0; n * dim];
    for r in 0..n {
        feats[r * dim] = columns[x][r];
        feats[r * dim + 1] = columns[y][r];
        for (j, &zc) in z.iter().enumerate() {
            feats[r * dim + 2 + j] = columns[zc][r];
        }
    }
    let donors: Vec<usize> = (0..n).collect();
    let idx = MatchingIndex::exact_within(&feats, dim, &donors, MatchingDistance::Euclidean, ctx)?;
    knn.features = feats;
    knn.index = Some(idx);
    knn.last_dim = dim;
    knn.last_n = n;
    knn.last_fingerprint = fingerprint;
    knn.index_generation = knn.index_generation.saturating_add(1);
    knn.index_builds = knn.index_builds.saturating_add(1);
    Ok(())
}

fn knn_stat_from_index(
    knn: &mut KnnDependenceWorkspace,
    k: usize,
    ctx: &ExecutionContext,
) -> Result<f64, StatsError> {
    let n = knn.last_n;
    if knn.distances.len() < n {
        knn.distances.resize(n, 0.0);
    } else {
        knn.distances.truncate(n);
    }
    let idx = knn.index.as_ref().ok_or(StatsError::Shape { message: "missing kNN index" })?;
    idx.kth_self_distances_of_donors_within(k, &mut knn.distances, ctx)?;
    let mean = knn.distances.iter().sum::<f64>() / n as f64;
    Ok(-mean)
}

/// Strata for the kNN conditional permutation null.
///
/// - Empty `Z`: one stratum (every row), so exchangeable / block shuffles are well defined.
/// - Discrete `Z` (few distinct joint levels): exact level strata.
/// - Continuous univariate `Z`: adjacent pairs after sorting Z, so permuting Y within a pair
///   preserves local Y–Z dependence under H0.
/// - Continuous multivariate `Z`: refused ([`StatsError::Unsupported`]). No row grouping keeps
///   the null honest: ordering by the norm of Z pairs `z = (2, 0)` with `z = (-2, 0)`, and even
///   true nearest-neighbour pairs sit `O(n^{-1/|Z|})` apart, which for `|Z| >= 2` shrinks no
///   faster than the statistic's own noise, so every swap blurs Y–Z and Type I stays inflated.
///
/// Non-finite Z values are refused: rank/sort paths must not coerce NaN into a finite stratum.
pub(super) fn z_permutation_strata(
    columns: &[&[f64]],
    z: &[usize],
    n: usize,
    _k: usize,
) -> Result<Vec<Vec<usize>>, StatsError> {
    if z.is_empty() {
        return Ok(vec![(0..n).collect()]);
    }
    ensure_finite_z(columns, z, n)?;
    let keys = joint_z_keys(columns, z, n);
    let mut distinct = keys.clone();
    distinct.sort_unstable();
    distinct.dedup();
    // Few distinct levels → exact strata (permuting within a level keeps Y|Z).
    // Many distinct levels (continuous or high-cardinality) → local neighbourhoods.
    let discrete_cap = (n / 4).max(8);
    if distinct.len() <= discrete_cap {
        let mut map: HashMap<u64, Vec<usize>> = HashMap::new();
        for (r, key) in keys.iter().enumerate() {
            map.entry(*key).or_default().push(r);
        }
        let mut sorted_keys: Vec<u64> = map.keys().copied().collect();
        sorted_keys.sort_unstable();
        return Ok(sorted_keys.into_iter().filter_map(|key| map.remove(&key)).collect());
    }
    if z.len() >= 2 {
        return Err(StatsError::Unsupported {
            message: "KnnDependence cannot build a valid conditional permutation null for a \
                      continuous conditioning set of two or more variables; use ParCorr, GPDC, \
                      or discretised Z",
        });
    }
    // Pairwise (size-2) windows: larger local groups re-break Y–Z and inflate type I.
    Ok(local_z_neighbourhood_strata(columns[z[0]], n, 2))
}

fn ensure_finite_z(columns: &[&[f64]], z: &[usize], n: usize) -> Result<(), StatsError> {
    for &zc in z {
        let col = columns
            .get(zc)
            .ok_or(StatsError::Shape { message: "Z column index out of range for kNN strata" })?;
        if col.len() < n || col.iter().take(n).any(|v| !v.is_finite()) {
            return Err(StatsError::Shape {
                message: "non-finite Z in conditional permutation strata",
            });
        }
    }
    Ok(())
}

fn joint_z_keys(columns: &[&[f64]], z: &[usize], n: usize) -> Vec<u64> {
    let mut keys = vec![antecedent_core::Fnv1a64::new(); n];
    for &zc in z {
        let col = columns[zc];
        for (r, key) in keys.iter_mut().enumerate() {
            key.fold(col[r].to_bits());
        }
    }
    keys.iter().map(antecedent_core::Fnv1a64::finish).collect()
}

/// Sort rows by univariate Z, then cut into contiguous windows of size `neighbourhood`.
fn local_z_neighbourhood_strata(col: &[f64], n: usize, neighbourhood: usize) -> Vec<Vec<usize>> {
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|&a, &b| col[a].total_cmp(&col[b]).then_with(|| a.cmp(&b)));
    let m = neighbourhood.clamp(2, n);
    let mut strata = Vec::new();
    let mut start = 0usize;
    while start < n {
        let mut end = (start + m).min(n);
        if end < n && n - end < m {
            // Absorb a tiny remainder into the last full window.
            end = n;
        }
        strata.push(order[start..end].to_vec());
        start = end;
    }
    strata
}

/// Former name retained as a thin wrapper for call sites / docs that still say "coarse".
#[cfg(test)]
pub(super) fn coarse_z_strata(
    columns: &[&[f64]],
    z: &[usize],
    n: usize,
) -> Result<Vec<Vec<usize>>, StatsError> {
    z_permutation_strata(columns, z, n, 5)
}

/// Mixed-data kNN distance dependence: ranks discrete-looking columns then runs [`KnnDependence`].
#[derive(Clone, Debug, Default)]
pub struct MixedKnnDependence {
    inner: KnnDependence,
}

impl MixedKnnDependence {
    /// Construct.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self { inner: KnnDependence::new(k) }
    }
}

impl ConditionalIndependenceTest for MixedKnnDependence {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        let request = &prepared.bind_request(request);
        let n = request.columns.first().map_or(0, |c| c.len());
        for col in request.columns {
            if col.iter().any(|v| !v.is_finite()) {
                return Err(StatsError::Shape {
                    message: "non-finite value in MixedKnnDependence rank path",
                });
            }
        }
        let mut owned: Vec<Vec<f64>> = request.columns.iter().map(|c| c.to_vec()).collect();
        for col in &mut owned {
            if looks_discrete(col) {
                let ranked = col.clone();
                crate::ci::parcorr_variants::rank_column(&ranked, col)?;
            }
        }
        let refs: Vec<&[f64]> = owned.iter().map(std::vec::Vec::as_slice).collect();
        let ranked_req = CiBatchRequest {
            columns: &refs,
            queries: request.queries,
            z_flat: request.z_flat,
            significance: request.significance,
            confidence: request.confidence,
        };
        let _ = n;
        self.inner.test_batch(prepared, &ranked_req, workspace, ctx)
    }

    fn attainable_p(&self) -> crate::ci::types::AttainableP {
        crate::ci::types::AttainableP::NonparametricPermutation
    }
}

#[allow(
    clippy::cast_possible_truncation,
    reason = "rounding to i64 only feeds a distinct-level count for a discreteness heuristic; astronomically large values saturate and merely merge"
)]
fn looks_discrete(col: &[f64]) -> bool {
    if col.is_empty() {
        return false;
    }
    let mut uniq = col.iter().map(|v| v.round() as i64).collect::<Vec<_>>();
    uniq.sort_unstable();
    uniq.dedup();
    let integerish = col.iter().all(|v| (v - v.round()).abs() < 1e-9);
    integerish && uniq.len() <= col.len().saturating_div(4).max(8)
}
