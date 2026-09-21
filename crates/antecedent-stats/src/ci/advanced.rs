//! Oracle, kNN distance-dependence, symbolic CMI, and GPDC CI tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::doc_markdown,
    clippy::trivially_copy_pass_by_ref
)]

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_core::{ExecutionContext, KernelPolicy, StreamDomain};
use antecedent_kernels::{shuffle, unbiased_index};

use super::block_shuffle::block_permute_contiguous;
use super::types::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependenceTest,
    KnnDependenceWorkspace, PreparedCiTest, nonparametric_permutation_count,
    reject_unsupported_block_size, requested_block_size,
};
use crate::error::StatsError;
use crate::matching::{MatchingDistance, MatchingIndex};

/// Oracle CI: independence decided by an explicit forbidden-edge set (synthetic/conformance).
#[derive(Clone, Debug, Default)]
pub struct OracleCi {
    /// Pairs `(min,max)` column indexes that are dependent (edge present in true graph).
    pub dependent_pairs: Arc<[(usize, usize)]>,
}

impl OracleCi {
    /// Construct.
    #[must_use]
    pub fn new(dependent_pairs: impl Into<Arc<[(usize, usize)]>>) -> Self {
        Self { dependent_pairs: dependent_pairs.into() }
    }

    fn is_dependent(&self, x: usize, y: usize) -> bool {
        let (a, b) = if x <= y { (x, y) } else { (y, x) };
        self.dependent_pairs.iter().any(|&(u, v)| u == a && v == b)
    }
}

impl ConditionalIndependenceTest for OracleCi {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        _workspace: &mut CiWorkspace,
        _ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        let request = &prepared.bind_request(request);
        let mut results = Vec::with_capacity(request.queries.len());
        for q in request.queries {
            let dep = self.is_dependent(q.x, q.y);
            results.push(CiResult {
                statistic: if dep { 1.0 } else { 0.0 },
                p_value: if dep { 0.0 } else { 1.0 },
                df: 0.0,
                ci: None,
            });
        }
        Ok(CiBatchResult { results })
    }
}

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
        // directly — no per-replicate index construction or feature copies.
        let mut null_feats: Vec<f64> = Vec::new();
        let mut null_dists: Vec<f64> = Vec::new();
        let mut null_dist_scratch: Vec<f64> = Vec::new();
        let mut results = Vec::with_capacity(request.queries.len());
        for (qi, q) in request.queries.iter().enumerate() {
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
            ensure_knn_index(request.columns, q.x, q.y, z, n, dim, &mut workspace.knn)?;
            let builds_before = workspace.knn.index_builds;
            let stat = knn_stat_from_index(&mut workspace.knn, self.k)?;
            // Null: permute Y within local Z neighbourhoods (or exact Z-level strata when
            // Z is discrete) so the Y–Z link is preserved under H0. A full unconditional
            // shuffle — or coarse tercile bins on continuous Z — inflates type-I error
            // whenever Y depends on Z. See `z_permutation_strata`.
            let strata = z_permutation_strata(request.columns, z, n, self.k)?;
            let mut y_perm = request.columns[q.y].to_vec();
            let mut rng = ctx.rng.stream_for(StreamDomain::StatsCi, 0xC11_u64 ^ qi as u64);
            let mut null_ge = 0u32;
            // The primary index's feature matrix has exactly the x/y/z layout the
            // null needs; clone it once per query and rewrite the Y column per
            // replicate.
            null_feats.clear();
            null_feats.extend_from_slice(&workspace.knn.features);
            if null_dists.len() < n {
                null_dists.resize(n, 0.0);
            }
            for _ in 0..n_perm {
                if block_size > 1 {
                    block_permute_contiguous(&mut y_perm, block_size, &mut rng);
                } else {
                    for rows in &strata {
                        for i in (1..rows.len()).rev() {
                            let j = unbiased_index(&mut rng, i + 1);
                            y_perm.swap(rows[i], rows[j]);
                        }
                    }
                }
                for (r, &value) in y_perm.iter().enumerate() {
                    null_feats[r * dim + 1] = value;
                }
                MatchingIndex::kth_self_distances(
                    &null_feats[..n * dim],
                    n,
                    dim,
                    MatchingDistance::Euclidean,
                    self.k,
                    &mut null_dists,
                    &mut null_dist_scratch,
                )?;
                let null = -null_dists[..n].iter().sum::<f64>() / n as f64;
                if null >= stat {
                    null_ge = null_ge.saturating_add(1);
                }
            }
            debug_assert_eq!(workspace.knn.index_builds, builds_before);
            let p = (1.0 + f64::from(null_ge)) / (1.0 + n_perm as f64);
            // df is not defined for this distance statistic; leave 0 rather than claim n.
            results.push(CiResult { statistic: stat, p_value: p, df: 0.0, ci: None });
        }
        Ok(CiBatchResult { results })
    }
}

/// Content identity of the columns feeding a kNN index.
///
/// Hashes query indexes and the full contents of each involved column. Pointers
/// are not identity: allocator reuse or an unsampled mutation must miss.
fn knn_input_fingerprint(columns: &[&[f64]], x: usize, y: usize, z: &[usize], n: usize) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(0x0100_0000_01b3);
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
    h
}

fn ensure_knn_index(
    columns: &[&[f64]],
    x: usize,
    y: usize,
    z: &[usize],
    n: usize,
    dim: usize,
    knn: &mut KnnDependenceWorkspace,
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
    let idx = MatchingIndex::exact(&feats, dim, &donors, MatchingDistance::Euclidean)
        .map_err(|e| StatsError::Backend(e.to_string()))?;
    knn.features = feats;
    knn.index = Some(idx);
    knn.last_dim = dim;
    knn.last_n = n;
    knn.last_fingerprint = fingerprint;
    knn.index_generation = knn.index_generation.saturating_add(1);
    knn.index_builds = knn.index_builds.saturating_add(1);
    Ok(())
}

fn knn_stat_from_index(knn: &mut KnnDependenceWorkspace, k: usize) -> Result<f64, StatsError> {
    let n = knn.last_n;
    if knn.distances.len() < n {
        knn.distances.resize(n, 0.0);
    } else {
        knn.distances.truncate(n);
    }
    let idx = knn.index.as_ref().ok_or(StatsError::Shape { message: "missing kNN index" })?;
    idx.kth_self_distances_of_donors(k, &mut knn.distances)?;
    let mean = knn.distances.iter().sum::<f64>() / n as f64;
    Ok(-mean)
}

/// Strata for the kNN conditional permutation null.
///
/// - Empty `Z`: one stratum (every row), so exchangeable / block shuffles are well defined.
/// - Discrete `Z` (few distinct joint levels): exact level strata.
/// - Continuous `Z`: contiguous windows after ordering in standardised Z-space so that
///   permuting Y within a window preserves local Y–Z dependence under H0.
///
/// Non-finite Z values are refused: rank/sort paths must not coerce NaN into a finite stratum.
fn z_permutation_strata(
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
    // Pairwise (size-2) windows: larger local groups re-break Y–Z and inflate type I.
    local_z_neighbourhood_strata(columns, z, n, 2)
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
    let mut keys = vec![0xcbf2_9ce4_8422_2325_u64; n];
    for &zc in z {
        let col = columns[zc];
        for (r, key) in keys.iter_mut().enumerate() {
            *key ^= col[r].to_bits();
            *key = key.wrapping_mul(0x0100_0000_01b3);
        }
    }
    keys
}

/// Order rows along a 1-D embedding of standardised Z, then cut into contiguous
/// windows of size `neighbourhood`. For univariate Z this is an ordinary sort;
/// for multivariate Z the embedding is distance from the origin in Z-space
/// (with the first coordinate as a tie-break), which keeps nearby Z together.
fn local_z_neighbourhood_strata(
    columns: &[&[f64]],
    z: &[usize],
    n: usize,
    neighbourhood: usize,
) -> Result<Vec<Vec<usize>>, StatsError> {
    let zdim = z.len();
    let mut feats = vec![0.0; n * zdim];
    for (j, &zc) in z.iter().enumerate() {
        let col = columns[zc];
        let mean = col.iter().take(n).sum::<f64>() / n as f64;
        let var = col
            .iter()
            .take(n)
            .map(|v| {
                let d = v - mean;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let sd = var.sqrt();
        if sd.is_finite() && sd > 0.0 {
            for r in 0..n {
                feats[r * zdim + j] = (col[r] - mean) / sd;
            }
        }
    }
    let mut order: Vec<usize> = (0..n).collect();
    if zdim == 1 {
        order.sort_by(|&a, &b| {
            feats[a]
                .partial_cmp(&feats[b])
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| a.cmp(&b))
        });
    } else {
        order.sort_by(|&a, &b| {
            let da: f64 = (0..zdim).map(|j| feats[a * zdim + j].powi(2)).sum();
            let db: f64 = (0..zdim).map(|j| feats[b * zdim + j].powi(2)).sum();
            da.partial_cmp(&db)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then_with(|| {
                    feats[a * zdim]
                        .partial_cmp(&feats[b * zdim])
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .then_with(|| a.cmp(&b))
        });
    }
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
    Ok(strata)
}

/// Former name retained as a thin wrapper for call sites / docs that still say "coarse".
#[cfg(test)]
fn coarse_z_strata(
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
                super::parcorr_variants::rank_column(&ranked, col)?;
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
}

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

/// Symbolic CMI on already-binned/ordinal integer codes (G²-style on symbol pairs).
#[derive(Clone, Debug, Default)]
pub struct SymbolicCmi;

impl SymbolicCmi {
    /// Construct.
    #[must_use]
    pub fn new() -> Self {
        Self
    }
}

impl ConditionalIndependenceTest for SymbolicCmi {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        _workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        let request = &prepared.bind_request(request);
        let mut results = Vec::with_capacity(request.queries.len());
        for (qi, q) in request.queries.iter().enumerate() {
            let n = request.columns[q.x].len();
            let z = &request.z_flat[q.z_start..q.z_start + q.z_len];
            let mi = conditional_symbolic_mi(request.columns, q.x, q.y, z, n);
            // Permutation p-value on Y, shuffled within Z strata so the Y–Z link is
            // preserved under H0 (an unconditional shuffle inflates type-I error when
            // Y depends on Z).
            let strata = symbol_strata_sorted(request.columns, z, n);
            let mut y_perm = request.columns[q.y].to_vec();
            let mut rng = ctx.rng.stream_for(StreamDomain::StatsCi, 0x51C_u64 ^ qi as u64);
            // As for `KnnDependence`: blocking is well defined only when the conditioning set
            // is empty, where `symbol_strata_sorted` yields a single time-ordered stratum. With
            // conditioning, the strata are Z-symbol hashes scattered across time and blocking is
            // structurally impossible, so the request is refused.
            let block_size = requested_block_size(request.significance);
            if block_size > 1 && !z.is_empty() {
                reject_unsupported_block_size(request.significance, "SymbolicCmi")?;
            }
            let n_perm = nonparametric_permutation_count(request.significance);
            let mut null_ge = 0u32;
            for _ in 0..n_perm {
                if block_size > 1 {
                    block_permute_contiguous(&mut y_perm, block_size, &mut rng);
                } else {
                    for rows in &strata {
                        for i in (1..rows.len()).rev() {
                            let j = unbiased_index(&mut rng, i + 1);
                            y_perm.swap(rows[i], rows[j]);
                        }
                    }
                }
                let mut cols: Vec<&[f64]> = request.columns.to_vec();
                cols[q.y] = &y_perm;
                let null = conditional_symbolic_mi(&cols, q.x, q.y, z, n);
                if null >= mi {
                    null_ge = null_ge.saturating_add(1);
                }
            }
            let p = (1.0 + f64::from(null_ge)) / (1.0 + n_perm as f64);
            results.push(CiResult { statistic: mi, p_value: p, df: 0.0, ci: None });
        }
        Ok(CiBatchResult { results })
    }
}

/// Rows grouped by exact Z symbol key. Deterministic order (sorted keys) so the seeded
/// RNG stream used by permutation nulls stays reproducible.
fn symbol_strata_sorted(columns: &[&[f64]], z: &[usize], n: usize) -> Vec<Vec<usize>> {
    let mut strata: HashMap<u64, Vec<usize>> = HashMap::new();
    for r in 0..n {
        let key = if z.is_empty() {
            0u64
        } else {
            let mut h = 0xcbf2_9ce4_8422_2325_u64;
            for &zc in z {
                let v = columns[zc][r].round() as i32;
                h ^= u64::from(v as u32);
                h = h.wrapping_mul(0x0100_0000_01b3);
            }
            h
        };
        strata.entry(key).or_default().push(r);
    }
    let mut keys: Vec<u64> = strata.keys().copied().collect();
    keys.sort_unstable();
    keys.into_iter().filter_map(|k| strata.remove(&k)).collect()
}

fn conditional_symbolic_mi(columns: &[&[f64]], x: usize, y: usize, z: &[usize], n: usize) -> f64 {
    // Stratify by Z symbols; average stratum MI(X;Y|Z=z).
    let strata = symbol_strata_sorted(columns, z, n);
    let mut mi = 0.0;
    let mut weight = 0.0;
    for rows in &strata {
        if rows.len() < 2 {
            continue;
        }
        let w = rows.len() as f64;
        mi += w * symbolic_mi_on_rows(columns, x, y, rows);
        weight += w;
    }
    if weight > 0.0 { mi / weight } else { 0.0 }
}

fn symbolic_mi_on_rows(columns: &[&[f64]], x: usize, y: usize, rows: &[usize]) -> f64 {
    let mut joint: HashMap<(i32, i32), f64> = HashMap::new();
    let mut mx: HashMap<i32, f64> = HashMap::new();
    let mut my: HashMap<i32, f64> = HashMap::new();
    let nf = rows.len() as f64;
    for &r in rows {
        let a = columns[x][r].round() as i32;
        let b = columns[y][r].round() as i32;
        *joint.entry((a, b)).or_default() += 1.0;
        *mx.entry(a).or_default() += 1.0;
        *my.entry(b).or_default() += 1.0;
    }
    let mut mi = 0.0;
    for ((a, b), c) in &joint {
        let pxy = c / nf;
        let px = mx[a] / nf;
        let py = my[b] / nf;
        if pxy > 0.0 && px > 0.0 && py > 0.0 {
            mi += pxy * (pxy / (px * py)).ln();
        }
    }
    mi
}

/// Native GPDC: RBF-GP residualization (ridge) + distance-correlation on residuals.
///
/// Residualization centers the response, factors `K+λI` once with Cholesky, and
/// predicts with the unregularized kernel (MM-008). The earlier Jacobi-on-raw-`y`
/// path rejected conditional nulls and missed two-conditioner alternatives against
/// the pinned advanced-CI oracle.
#[derive(Clone, Debug)]
pub struct Gpdc {
    /// RBF length scale.
    pub length_scale: f64,
    /// Ridge.
    pub ridge: f64,
}

impl Default for Gpdc {
    fn default() -> Self {
        Self::new()
    }
}

impl Gpdc {
    /// Construct with defaults.
    #[must_use]
    pub fn new() -> Self {
        Self { length_scale: 1.0, ridge: 1e-3 }
    }
}

impl ConditionalIndependenceTest for Gpdc {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        _workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        if !self.length_scale.is_finite() || self.length_scale <= 0.0 {
            return Err(StatsError::Shape {
                message: "GPDC length scale must be finite and positive",
            });
        }
        if !self.ridge.is_finite() || self.ridge <= 0.0 {
            return Err(StatsError::Shape { message: "GPDC ridge must be finite and positive" });
        }
        let request = &prepared.bind_request(request);
        let n = request.columns.first().map_or(0, |c| c.len());
        if n == 0 {
            return Err(StatsError::Shape { message: "no columns" });
        }
        // Unlike KnnDependence / SymbolicCmi, GPDC residualizes X and Y on Z through the GP
        // regression *before* permuting (see `gp_residual`), so by the time the null is built
        // there is nothing left to stratify — Z's influence is already removed from both
        // residual series. That makes a contiguous-block permutation of the Y residual a
        // direct, valid substitution for the exchangeable shuffle, the same architecture
        // ParCorr's block-shuffle path already uses. So `block_size` is honoured, not rejected.
        let block_size = requested_block_size(request.significance);
        let n_perm = nonparametric_permutation_count(request.significance);
        let policy = &ctx.kernel_policy;
        // The X-side centered distance matrix is invariant under Y permutations;
        // prepare each side once per query and recompute only the Y side per
        // replicate, into buffers reused across queries and replicates.
        let mut x_side = CenteredDistances::default();
        let mut y_side = CenteredDistances::default();
        let mut center_row = Vec::new();
        let mut center_col = Vec::new();
        let mut results = Vec::with_capacity(request.queries.len());
        for (qi, q) in request.queries.iter().enumerate() {
            let z = &request.z_flat[q.z_start..q.z_start + q.z_len];
            let rx = gp_residual(request.columns[q.x], request.columns, z, self)?;
            let ry = gp_residual(request.columns[q.y], request.columns, z, self)?;
            x_side.prepare(policy, &rx, &mut center_row, &mut center_col);
            y_side.prepare(policy, &ry, &mut center_row, &mut center_col);
            let dcor = dcor_from_sides(&x_side, &y_side);
            // Permutation null: permute the Y residuals (Z influence already removed) and
            // recompute dCor; add-one p-value keeps it in (0, 1]. `block_size > 1` permutes
            // contiguous blocks so the residual's serial dependence survives into the null;
            // otherwise it is an ordinary exchangeable shuffle.
            let mut ry_perm = ry.clone();
            let mut rng = ctx.rng.stream_for(StreamDomain::StatsCi, 0x69DC_u64 ^ qi as u64);
            let mut null_ge = 0u32;
            for _ in 0..n_perm {
                if block_size > 1 {
                    block_permute_contiguous(&mut ry_perm, block_size, &mut rng);
                } else {
                    shuffle(&mut rng, &mut ry_perm);
                }
                y_side.prepare(policy, &ry_perm, &mut center_row, &mut center_col);
                if dcor_from_sides(&x_side, &y_side) >= dcor {
                    null_ge = null_ge.saturating_add(1);
                }
            }
            let p = (1.0 + f64::from(null_ge)) / (1.0 + n_perm as f64);
            results.push(CiResult { statistic: dcor, p_value: p, df: 0.0, ci: None });
        }
        Ok(CiBatchResult { results })
    }
}

/// One series' double-centered pairwise-distance matrix and its distance variance.
///
/// [`dcor_from_sides`] over two prepared sides matches [`distance_correlation`]
/// bit for bit: each accumulator runs over the same indices in the same order,
/// only split across calls.
#[derive(Default)]
struct CenteredDistances {
    a: Vec<f64>,
    n: usize,
    dvar: f64,
}

impl CenteredDistances {
    fn prepare(
        &mut self,
        policy: &KernelPolicy,
        series: &[f64],
        row: &mut Vec<f64>,
        col: &mut Vec<f64>,
    ) {
        let n = series.len();
        self.n = n;
        if n < 2 {
            self.a.clear();
            self.dvar = 0.0;
            return;
        }
        self.a.resize(n * n, 0.0);
        antecedent_kernels::pairwise_l1_fill(policy, series, &mut self.a);
        double_center_inplace_with(&mut self.a, n, row, col);
        let mut dvar = 0.0;
        for &v in &self.a {
            dvar += v * v;
        }
        self.dvar = dvar / (n * n) as f64;
    }
}

fn dcor_from_sides(x: &CenteredDistances, y: &CenteredDistances) -> f64 {
    let n = x.n;
    if n < 2 || y.n != n {
        return 0.0;
    }
    let mut dcov2 = 0.0;
    for (&ax, &ay) in x.a.iter().zip(&y.a) {
        dcov2 += ax * ay;
    }
    dcov2 /= (n * n) as f64;
    if x.dvar <= 0.0 || y.dvar <= 0.0 {
        return 0.0;
    }
    // Székely et al. (2007) dCor: sqrt(dCov² / sqrt(dVarX · dVarY)).
    (dcov2.max(0.0) / (x.dvar * y.dvar).sqrt()).sqrt()
}

fn gp_residual(
    y: &[f64],
    columns: &[&[f64]],
    z: &[usize],
    gp: &Gpdc,
) -> Result<Vec<f64>, StatsError> {
    let n = y.len();
    if y.iter().any(|v| !v.is_finite()) {
        return Err(StatsError::Shape { message: "non-finite response in GPDC residualization" });
    }
    let mean = y.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = y.iter().map(|value| value - mean).collect();
    if z.is_empty() {
        return Ok(centered);
    }
    // Standardise each Z column before the fixed RBF length scale so conditioning
    // does not silently fail when Z is far from unit scale (stats-ci-4).
    let zdim = z.len();
    let mut z_std = vec![0.0; n * zdim];
    for (j, &zc) in z.iter().enumerate() {
        let col = columns
            .get(zc)
            .ok_or(StatsError::Shape { message: "Z column index out of range for GPDC" })?;
        if col.len() < n || col.iter().take(n).any(|v| !v.is_finite()) {
            return Err(StatsError::Shape { message: "non-finite Z in GPDC residualization" });
        }
        let mean_z = col.iter().take(n).sum::<f64>() / n as f64;
        let var_z = col
            .iter()
            .take(n)
            .map(|v| {
                let d = v - mean_z;
                d * d
            })
            .sum::<f64>()
            / n as f64;
        let sd = var_z.sqrt();
        if sd.is_finite() && sd > 0.0 {
            for r in 0..n {
                z_std[r * zdim + j] = (col[r] - mean_z) / sd;
            }
        }
    }
    // Build Gram on standardised Z (sum of RBF over Z dims) and solve (K+λI)α = y.
    let mut k = vec![0.0; n * n];
    let ls2 = gp.length_scale * gp.length_scale;
    for i in 0..n {
        for j in 0..=i {
            let mut d2 = 0.0;
            for d in 0..zdim {
                let diff = z_std[i * zdim + d] - z_std[j * zdim + d];
                d2 += diff * diff;
            }
            let kij = (-0.5 * d2 / ls2).exp();
            k[i * n + j] = kij;
            k[j * n + i] = kij;
        }
        k[i * n + i] += gp.ridge;
    }
    let chol = crate::gram::cholesky_spd(&k, n)
        .ok_or_else(|| StatsError::Backend("GPDC kernel factorization failed".into()))?;
    let alpha = crate::gram::chol_solve(&chol, n, &centered)
        .ok_or_else(|| StatsError::Backend("GPDC kernel solve failed".into()))?;
    let mut pred = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            pred[i] += k[i * n + j] * alpha[j];
        }
        // MM-008: `α` solves (K+λI)α = y_c; mean prediction uses Kα = y_c − λα.
        pred[i] -= gp.ridge * alpha[i];
    }
    Ok((0..n).map(|i| centered[i] - pred[i]).collect())
}

/// Reference implementation retained for the differential test of
/// [`dcor_from_sides`]; production paths use the side-cached form.
#[cfg(test)]
fn distance_correlation(policy: &KernelPolicy, x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mut ax = vec![0.0; n * n];
    let mut ay = vec![0.0; n * n];
    antecedent_kernels::pairwise_l1_fill(policy, x, &mut ax);
    antecedent_kernels::pairwise_l1_fill(policy, y, &mut ay);
    double_center_inplace(&mut ax, n);
    double_center_inplace(&mut ay, n);
    let mut dcov2 = 0.0;
    let mut dvarx = 0.0;
    let mut dvary = 0.0;
    for i in 0..n * n {
        dcov2 += ax[i] * ay[i];
        dvarx += ax[i] * ax[i];
        dvary += ay[i] * ay[i];
    }
    dcov2 /= (n * n) as f64;
    dvarx /= (n * n) as f64;
    dvary /= (n * n) as f64;
    if dvarx <= 0.0 || dvary <= 0.0 {
        return 0.0;
    }
    // Székely et al. (2007) dCor: sqrt(dCov² / sqrt(dVarX · dVarY)).
    (dcov2.max(0.0) / (dvarx * dvary).sqrt()).sqrt()
}

#[cfg(test)]
fn double_center_inplace(a: &mut [f64], n: usize) {
    let mut row = Vec::new();
    let mut col = Vec::new();
    double_center_inplace_with(a, n, &mut row, &mut col);
}

/// [`double_center_inplace`] with caller-owned row/column scratch for replicate loops.
fn double_center_inplace_with(a: &mut [f64], n: usize, row: &mut Vec<f64>, col: &mut Vec<f64>) {
    row.clear();
    row.resize(n, 0.0);
    col.clear();
    col.resize(n, 0.0);
    let mut mean = 0.0;
    for i in 0..n {
        for j in 0..n {
            row[i] += a[i * n + j];
            col[j] += a[i * n + j];
            mean += a[i * n + j];
        }
    }
    for i in 0..n {
        row[i] /= n as f64;
        col[i] /= n as f64;
    }
    mean /= (n * n) as f64;
    for i in 0..n {
        for j in 0..n {
            a[i * n + j] = a[i * n + j] - row[i] - col[j] + mean;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ci::types::{
        CiBatchRequest, CiQuery, CiWorkspace, ConfidenceMethod, SignificanceMethod,
    };

    /// The GPDC null recomputes only the Y-side centered distance matrix per
    /// replicate; the split accumulation must match the monolithic
    /// `distance_correlation` bit for bit.
    #[test]
    fn dcor_from_sides_matches_monolithic_distance_correlation() {
        let policy = KernelPolicy::default_policy();
        let x: Vec<f64> = (0..64).map(|i| ((i * 13 + 5) % 31) as f64 * 0.17 - 2.0).collect();
        let y: Vec<f64> = (0..64).map(|i| ((i * 7 + 11) % 23) as f64 * 0.31 - 3.0).collect();
        let mut xs = CenteredDistances::default();
        let mut ys = CenteredDistances::default();
        let (mut row, mut col) = (Vec::new(), Vec::new());
        xs.prepare(&policy, &x, &mut row, &mut col);
        ys.prepare(&policy, &y, &mut row, &mut col);
        let split = dcor_from_sides(&xs, &ys);
        let mono = distance_correlation(&policy, &x, &y);
        assert!(split.to_bits() == mono.to_bits(), "split={split:?} mono={mono:?}");
    }

    /// A test that cannot honour `block_size` must refuse it rather than discard it — but these
    /// three draw that line in different places, and this pins all three.
    ///
    /// `KnnDependence` and `SymbolicCmi` build their null as an exchange within Z strata
    /// (`z_permutation_strata` / `symbol_strata_sorted`). **With a conditioning set** those strata are
    /// Z-level or local-neighbourhood groups whose members are scattered across time: preserving `Y|Z` needs
    /// permutation within scattered index sets, preserving serial dependence needs contiguous
    /// runs, and the two cannot both hold. A caller asking for `block_size = 20` to preserve
    /// 20-step serial dependence must not silently receive an ordinary within-stratum exchange,
    /// which under-disperses the null and inflates Type I error for autocorrelated data — the one
    /// case the parameter exists for. So that request is an error.
    ///
    /// **With an empty conditioning set** the conflict disappears: both stratifiers degenerate to
    /// a single stratum holding every row in original time order, so a contiguous-block
    /// permutation is exactly as well defined as it is for `ParCorr`, and both tests honour it.
    /// That is not a corner case — PC1's first level (`cond_size = 0`) tests unconditionally, and
    /// `GSquared` already drew the line here.
    ///
    /// `Gpdc` honours `block_size` in both shapes because it residualizes X and Y on Z via GP
    /// regression *before* permuting (`gp_residual`), leaving nothing to stratify.
    ///
    /// `block_size = 1` requests no blocking and is always accepted.
    #[test]
    fn block_preserving_requests_honoured_or_refused_per_conditioning_set() {
        static UNCONDITIONAL: [CiQuery; 1] = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        static CONDITIONAL: [CiQuery; 1] = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];

        let x: Vec<f64> = (0..60).map(|i| (i as f64 * 0.3).sin()).collect();
        let y: Vec<f64> = (0..60).map(|i| (i as f64 * 0.3).cos()).collect();
        let z: Vec<f64> = (0..60).map(|i| (i as f64 * 0.11).sin()).collect();
        let cols: [&[f64]; 3] = [&x, &y, &z];
        let z_flat = [2usize];
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);

        let req_for = |queries: &'static [CiQuery], block_size: usize| CiBatchRequest {
            columns: &cols,
            queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::BlockShuffle { replicates: 19, block_size },
            confidence: ConfidenceMethod::default(),
        };
        let conditional: &'static [CiQuery] = &CONDITIONAL;
        let unconditional: &'static [CiQuery] = &UNCONDITIONAL;

        for block_size in [5usize, 20] {
            // Conditioned: the stratifying tests must refuse; Gpdc must not.
            let req = req_for(conditional, block_size);
            assert!(
                KnnDependence::new(3).test_batch_adhoc(&req, &mut ws, &ctx).is_err(),
                "KnnDependence accepted block_size={block_size} with a conditioning set, \
                 which it cannot honour"
            );
            assert!(
                SymbolicCmi::new().test_batch_adhoc(&req, &mut ws, &ctx).is_err(),
                "SymbolicCmi accepted block_size={block_size} with a conditioning set, \
                 which it cannot honour"
            );
            assert!(
                Gpdc::new().test_batch_adhoc(&req, &mut ws, &ctx).is_ok(),
                "Gpdc refused block_size={block_size}, which it honours by residualizing"
            );

            // Unconditional: a single time-ordered stratum, so all three honour blocking.
            let req = req_for(unconditional, block_size);
            assert!(
                KnnDependence::new(3).test_batch_adhoc(&req, &mut ws, &ctx).is_ok(),
                "KnnDependence refused block_size={block_size} with an empty conditioning set, \
                 where a contiguous-block permutation is well defined"
            );
            assert!(
                SymbolicCmi::new().test_batch_adhoc(&req, &mut ws, &ctx).is_ok(),
                "SymbolicCmi refused block_size={block_size} with an empty conditioning set, \
                 where a contiguous-block permutation is well defined"
            );
            assert!(Gpdc::new().test_batch_adhoc(&req, &mut ws, &ctx).is_ok());
        }

        // block_size = 1 imposes no blocking, so it must still run in both shapes.
        for queries in [conditional, unconditional] {
            let req = req_for(queries, 1);
            assert!(KnnDependence::new(3).test_batch_adhoc(&req, &mut ws, &ctx).is_ok());
            assert!(SymbolicCmi::new().test_batch_adhoc(&req, &mut ws, &ctx).is_ok());
            assert!(Gpdc::new().test_batch_adhoc(&req, &mut ws, &ctx).is_ok());
        }
    }

    #[test]
    fn oracle_marks_dependence() {
        let oracle = OracleCi::new([(0usize, 1usize)]);
        let x = [1.0, 2.0, 3.0];
        let y = [1.0, 2.0, 3.0];
        let cols: [&[f64]; 2] = [&x, &y];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let out = oracle.test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        assert!((out.results[0].p_value - 0.0).abs() < f64::EPSILON);
    }

    #[test]
    fn symbolic_mi_positive_on_copy() {
        let x: Vec<f64> = (0..100).map(|i| f64::from(u32::try_from(i % 4).unwrap_or(0))).collect();
        let y = x.clone();
        let cols: [&[f64]; 2] = [&x, &y];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let out = SymbolicCmi::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        assert!(out.results[0].statistic > 0.5);
    }

    #[test]
    fn gpdc_runs() {
        let n = 40usize;
        let z: Vec<f64> = (0..n).map(|i| i as f64 / n as f64).collect();
        let x: Vec<f64> = z.iter().map(|v| v + 0.01).collect();
        let y: Vec<f64> = z.iter().map(|v| 2.0 * v).collect();
        let cols: [&[f64]; 3] = [&x, &y, &z];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
        let z_flat = [2usize];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let out = Gpdc::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        assert!(out.results[0].statistic.is_finite());
        assert!((0.0..=1.0).contains(&out.results[0].p_value));
    }

    fn lcg_noise(n: usize, seed: u64) -> Vec<f64> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6_364_136_223_846_793_005)
                    .wrapping_add(1_442_695_040_888_963_407);
                ((s >> 33) as f64) / ((1u64 << 31) as f64) - 0.5
            })
            .collect()
    }

    #[test]
    fn dcor_self_is_one_and_scale_invariant() {
        let policy = KernelPolicy::default_policy();
        let x: Vec<f64> = (0..50).map(|i| (i as f64 * 0.7).sin() + 0.1 * i as f64).collect();
        let d = distance_correlation(&policy, &x, &x);
        assert!((d - 1.0).abs() < 1e-9, "dcor(x,x)={d}");
        let y: Vec<f64> = (0..50).map(|i| f64::from(((i * 13 + 5) % 17) as u32)).collect();
        let d1 = distance_correlation(&policy, &x, &y);
        let xs: Vec<f64> = x.iter().map(|v| 3.5 * v).collect();
        let ys: Vec<f64> = y.iter().map(|v| 3.5 * v).collect();
        let d2 = distance_correlation(&policy, &xs, &ys);
        assert!((d1 - d2).abs() < 1e-9, "scale dependence: {d1} vs {d2}");
    }

    #[test]
    fn dcor_independent_small() {
        let policy = KernelPolicy::default_policy();
        let x = lcg_noise(200, 1);
        let y = lcg_noise(200, 2);
        let d = distance_correlation(&policy, &x, &y);
        assert!(d < 0.3, "dcor of independent noise = {d}");
    }

    #[test]
    fn gpdc_permutation_pvalue_separates_dependence() {
        let n = 60usize;
        let x = lcg_noise(n, 3);
        let y_dep = x.clone();
        let y_ind = lcg_noise(n, 4);
        let cols: [&[f64]; 3] = [&x, &y_dep, &y_ind];
        let queries = [
            CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 },
            CiQuery { x: 0, y: 2, z_start: 0, z_len: 0 },
        ];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(11);
        let out = Gpdc::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        assert!(out.results[0].p_value < 0.05, "dependent p={}", out.results[0].p_value);
        assert!(out.results[1].p_value > 0.1, "independent p={}", out.results[1].p_value);
    }

    #[test]
    fn knn_rebuilds_index_for_different_pairs_in_one_batch() {
        let n = 60usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
        let y_tight: Vec<f64> = x.iter().map(|v| v + 0.001).collect();
        let y_spread: Vec<f64> = (0..n).map(|i| ((i * 37 + 11) % 60) as f64).collect();
        let cols: [&[f64]; 3] = [&x, &y_tight, &y_spread];
        let queries = [
            CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 },
            CiQuery { x: 0, y: 2, z_start: 0, z_len: 0 },
        ];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(12);
        let out = KnnDependence::new(3).test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        let s1 = out.results[0].statistic;
        let s2 = out.results[1].statistic;
        assert!(
            (s1 - s2).abs() > 1e-6,
            "same n/dim pairs must not share a cached index: {s1} vs {s2}"
        );
        assert!(s1 > s2, "tight pair should have smaller kth distances: {s1} vs {s2}");
    }

    #[test]
    fn symbolic_null_preserves_yz_dependence() {
        // X ⊥ Y | Z with Y = Z (maximal Y–Z dependence): within-stratum permutation
        // leaves Y unchanged, so the p-value must be large, not systematically tiny.
        let n = 200usize;
        let z: Vec<f64> = (0..n).map(|i| (i % 4) as f64).collect();
        let y = z.clone();
        let x: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 5) as f64).collect();
        let cols: [&[f64]; 3] = [&x, &y, &z];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
        let z_flat = [2usize];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(13);
        let out = SymbolicCmi::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        assert!(out.results[0].p_value > 0.5, "p={}", out.results[0].p_value);
    }

    #[test]
    fn knn_null_preserves_yz_dependence() {
        // X ⊥ Y | Z with Y strongly driven by Z (three well-separated Z levels): the
        // within-strata null must not report systematically tiny p-values.
        let n = 90usize;
        let z: Vec<f64> = (0..n).map(|i| (i % 3) as f64 * 5.0).collect();
        let noise = lcg_noise(n, 6);
        let y: Vec<f64> = z.iter().zip(&noise).map(|(v, e)| v + 0.1 * e).collect();
        let x = lcg_noise(n, 5);
        let cols: [&[f64]; 3] = [&x, &y, &z];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
        let z_flat = [2usize];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(14);
        let out = KnnDependence::new(3).test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        assert!(out.results[0].p_value > 0.05, "p={}", out.results[0].p_value);
    }

    /// stats-ci-1: continuous Z with Y←Z must not yield Type I ≈ 1 under X ⊥ Y | Z.
    #[test]
    fn knn_continuous_z_null_type_i_not_near_one() {
        let n = 120usize;
        let trials = 200u32;
        let alpha = 0.05;
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(0xC11_C0u64);
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
        let z_flat = [2usize];
        let ci = KnnDependence::new(5);
        let mut rejects = 0u32;
        for t in 0..trials {
            let mut rng = ctx.rng.stream_for(StreamDomain::StatsCi, 0xC11_u64 ^ u64::from(t));
            let z: Vec<f64> = (0..n)
                .map(|_| {
                    let u = (rng.next_u64() as f64) / (u64::MAX as f64);
                    // Box–Muller half: approximate N(0,1)
                    let v = (rng.next_u64() as f64) / (u64::MAX as f64);
                    let r = (-2.0 * (u.max(1e-12)).ln()).sqrt();
                    r * (2.0 * std::f64::consts::PI * v).cos()
                })
                .collect();
            let y: Vec<f64> = z
                .iter()
                .map(|&zi| {
                    let e = (rng.next_u64() as f64) / (u64::MAX as f64) - 0.5;
                    zi + 0.35 * e
                })
                .collect();
            let x: Vec<f64> =
                (0..n).map(|_| (rng.next_u64() as f64) / (u64::MAX as f64) - 0.5).collect();
            let cols: [&[f64]; 3] = [&x, &y, &z];
            let req = CiBatchRequest {
                columns: &cols,
                queries: &queries,
                z_flat: &z_flat,
                significance: SignificanceMethod::BlockShuffle { replicates: 49, block_size: 1 },
                confidence: ConfidenceMethod::None,
            };
            let out = ci.test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
            if out.results[0].p_value < alpha {
                rejects += 1;
            }
        }
        let rate = f64::from(rejects) / f64::from(trials);
        // Coarse tercile nulls reject ≈ 1.0 here; local neighbourhoods must stay far below that.
        assert!(
            rate < 0.25,
            "continuous-Z kNN type I near 1 (or badly inflated): {rate} ({rejects}/{trials})"
        );
    }

    /// stats-ci-4: standardised Z lets GPDC condition when Z is scaled by ~100.
    #[test]
    fn gpdc_scaled_z_null_and_power() {
        let n = 80usize;
        let trials = 80u32;
        let alpha = 0.05;
        let scale = 100.0;
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(0x69DC_5Cu64);
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
        let z_flat = [2usize];
        let ci = Gpdc::new();
        let mut null_rej = 0u32;
        let mut alt_rej = 0u32;
        for t in 0..trials {
            let noise_z = lcg_noise(n, 100 + u64::from(t));
            let noise_x = lcg_noise(n, 200 + u64::from(t));
            let noise_y = lcg_noise(n, 300 + u64::from(t));
            let z: Vec<f64> = noise_z.iter().map(|e| scale * e).collect();
            // Null: X ⊥ Y | Z with both driven by Z.
            let x_null: Vec<f64> = z.iter().zip(&noise_x).map(|(zi, e)| 0.01 * zi + e).collect();
            let y_null: Vec<f64> = z.iter().zip(&noise_y).map(|(zi, e)| 0.02 * zi + e).collect();
            let cols_null: [&[f64]; 3] = [&x_null, &y_null, &z];
            let req = CiBatchRequest {
                columns: &cols_null,
                queries: &queries,
                z_flat: &z_flat,
                significance: SignificanceMethod::BlockShuffle { replicates: 49, block_size: 1 },
                confidence: ConfidenceMethod::None,
            };
            let out = ci.test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
            if out.results[0].p_value < alpha {
                null_rej += 1;
            }
            // Alt: residual X–Y dependence after conditioning on Z.
            let x_alt = x_null.clone();
            let y_alt: Vec<f64> = x_alt.iter().zip(&y_null).map(|(xi, yi)| xi + yi).collect();
            let cols_alt: [&[f64]; 3] = [&x_alt, &y_alt, &z];
            let req_alt = CiBatchRequest {
                columns: &cols_alt,
                queries: &queries,
                z_flat: &z_flat,
                significance: SignificanceMethod::BlockShuffle { replicates: 49, block_size: 1 },
                confidence: ConfidenceMethod::None,
            };
            let out_alt = ci.test_batch_adhoc(&req_alt, &mut ws, &ctx).unwrap();
            if out_alt.results[0].p_value < alpha {
                alt_rej += 1;
            }
        }
        let type_i = f64::from(null_rej) / f64::from(trials);
        let power = f64::from(alt_rej) / f64::from(trials);
        assert!(
            type_i < 0.25,
            "scaled-Z GPDC null rejects near 1 without standardisation: {type_i}"
        );
        assert!(power > 0.5, "scaled-Z GPDC lost power after standardisation: {power}");
    }

    /// stats-ci-5: non-finite Z must not become a finite stratum / rank result.
    #[test]
    fn nonfinite_z_and_rank_path_return_errors() {
        let n = 30usize;
        let mut z: Vec<f64> = (0..n).map(|i| i as f64).collect();
        z[7] = f64::NAN;
        let x = lcg_noise(n, 1);
        let y = lcg_noise(n, 2);
        let cols: [&[f64]; 3] = [&x, &y, &z];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
        let z_flat = [2usize];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        assert!(
            KnnDependence::new(3).test_batch_adhoc(&req, &mut ws, &ctx).is_err(),
            "KnnDependence must refuse non-finite Z"
        );
        assert!(
            coarse_z_strata(&cols, &[2], n).is_err(),
            "coarse_z_strata / z_permutation_strata must refuse non-finite Z"
        );

        let mut x_nan = x.clone();
        x_nan[3] = f64::INFINITY;
        let z_ok: Vec<f64> = (0..n).map(|i| (i % 4) as f64).collect();
        let cols_rank: [&[f64]; 3] = [&x_nan, &y, &z_ok];
        let req_rank = CiBatchRequest {
            columns: &cols_rank,
            queries: &queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::default(),
        };
        assert!(
            MixedKnnDependence::new(3).test_batch_adhoc(&req_rank, &mut ws, &ctx).is_err(),
            "MixedKnnDependence rank path must refuse non-finite input"
        );
    }
}
