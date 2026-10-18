//! Oracle, kNN distance-dependence, symbolic CMI, and GPDC CI tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, HashMap};
use std::sync::Arc;

use antecedent_core::{ExecutionContext, StreamDomain};
use antecedent_kernels::{shuffle, unbiased_index};

use super::block_shuffle::{block_permute_contiguous, query_stream_salt};
use super::gsquared::{
    discrete_strata, encode_categories, ensure_finite_categories, ensure_permutation_support,
};
use super::residualize::residual_is_uninformative;
use super::types::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependenceTest,
    KnnDependenceWorkspace, PreparedCiTest, nonparametric_permutation_count,
    reject_unsupported_block_size, requested_block_size,
};
use crate::dcor::distance_correlation;
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

    fn attainable_p(&self) -> super::types::AttainableP {
        super::types::AttainableP::NonparametricPermutation
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

    fn attainable_p(&self) -> super::types::AttainableP {
        super::types::AttainableP::NonparametricPermutation
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

/// Symbolic CMI on already-binned symbols (G²-style on symbol pairs); each distinct finite
/// value is a symbol.
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
    #[allow(
        clippy::cast_possible_truncation,
        reason = "y_perm holds exact f64 images of the i32 codes from encode_categories"
    )]
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
        for q in request.queries {
            let n = request.columns[q.x].len();
            let z = &request.z_flat[q.z_start..q.z_start + q.z_len];
            // NaN / ±inf would round to a finite symbol; refuse them.
            ensure_finite_categories(
                request.columns,
                [q.x, q.y].into_iter().chain(z.iter().copied()),
            )?;
            let xi = encode_categories(request.columns[q.x])?;
            let yi = encode_categories(request.columns[q.y])?;
            // Permutation p-value on Y, shuffled within Z strata so the Y–Z link is
            // preserved under H0 (an unconditional shuffle inflates type-I error when
            // Y depends on Z). Strata are invariant under the Y-only permutation.
            let strata = discrete_strata(request.columns, z, n)?;
            ensure_permutation_support(&strata, &xi, &yi)?;
            let mi = conditional_symbolic_mi(&xi, &yi, &strata);
            // Permute exact f64 images of the Y codes so the block permutation can move them.
            let mut y_perm: Vec<f64> = yi.iter().copied().map(f64::from).collect();
            let mut yi_perm = yi.clone();
            let mut rng = ctx.rng.stream_for(
                StreamDomain::StatsCi,
                0x51C_u64 ^ query_stream_salt(&[q.x], &[q.y], z),
            );
            // As for `KnnDependence`: blocking is well defined only when the conditioning set
            // is empty, where `discrete_strata` yields a single time-ordered stratum. With
            // conditioning, the strata are Z-symbol groups scattered across time and blocking
            // is structurally impossible, so the request is refused.
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
                for (code, value) in yi_perm.iter_mut().zip(&y_perm) {
                    *code = *value as i32;
                }
                let null = conditional_symbolic_mi(&xi, &yi_perm, &strata);
                if null >= mi {
                    null_ge = null_ge.saturating_add(1);
                }
            }
            let p = (1.0 + f64::from(null_ge)) / (1.0 + n_perm as f64);
            results.push(CiResult { statistic: mi, p_value: p, df: 0.0, ci: None });
        }
        Ok(CiBatchResult { results })
    }

    fn attainable_p(&self) -> super::types::AttainableP {
        super::types::AttainableP::NonparametricPermutation
    }
}

/// Stratum-size-weighted mean of the per-stratum `MI(X;Y | Z = z)` over strata with at least
/// two rows. [`ensure_permutation_support`] has already guaranteed such a stratum exists.
fn conditional_symbolic_mi(xi: &[i32], yi: &[i32], strata: &[Vec<usize>]) -> f64 {
    let mut mi = 0.0;
    let mut weight = 0.0;
    for rows in strata {
        if rows.len() < 2 {
            continue;
        }
        let w = rows.len() as f64;
        mi += w * symbolic_mi_on_rows(xi, yi, rows);
        weight += w;
    }
    if weight > 0.0 { mi / weight } else { 0.0 }
}

fn symbolic_mi_on_rows(xi: &[i32], yi: &[i32], rows: &[usize]) -> f64 {
    // Ordered maps: the float sum below must not depend on the process hash seed, or the
    // `null >= mi` comparison of two mathematically equal statistics flips between runs.
    let mut joint: BTreeMap<(i32, i32), f64> = BTreeMap::new();
    let mut mx: BTreeMap<i32, f64> = BTreeMap::new();
    let mut my: BTreeMap<i32, f64> = BTreeMap::new();
    let nf = rows.len() as f64;
    for &r in rows {
        let (a, b) = (xi[r], yi[r]);
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

/// Rows above which GPDC refuses a *conditional* test: it factorizes a dense `n × n` kernel
/// (memory `O(n²)`, Cholesky `O(n³)`), so a few thousand rows already take hundreds of MB. The
/// distance correlation itself is `O(n log n)` and never limits the unconditional test.
pub(crate) const GPDC_ROW_LIMIT: usize = 4_000;

/// Conditioning sets whose kernel factorization one batch keeps for reuse.
const GPDC_FACTOR_CACHE: usize = 2;

/// Native GPDC: RBF-GP residualization (ridge) + distance-correlation on residuals.
///
/// Residualization centers the response and factors `K+λI` once per distinct Z set with
/// Cholesky (X and Y are two right-hand sides of the same factorization, and queries of one
/// batch sharing a Z set reuse it). The GP mean prediction is `Kα` with `(K+λI)α = y_c`, so the
/// residual is exactly `y_c − Kα = λα` (MM-008). The earlier Jacobi-on-raw-`y` path rejected
/// conditional nulls and missed two-conditioner alternatives against the pinned advanced-CI
/// oracle. Conditional inputs above [`GPDC_ROW_LIMIT`] rows are refused; [`GpdcNystrom`] is the
/// separately identified construction for them.
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
        // direct, valid substitution for the exchangeable shuffle: the same residual-null
        // architecture the ParCorr family uses. So `block_size` is honoured, not rejected.
        let block_size = requested_block_size(request.significance);
        let n_perm = nonparametric_permutation_count(request.significance);
        let mut factors: Vec<(Vec<usize>, Vec<f64>)> = Vec::new();
        let mut results = Vec::with_capacity(request.queries.len());
        for q in request.queries {
            let z = &request.z_flat[q.z_start..q.z_start + q.z_len];
            let factor: Option<&[f64]> = if z.is_empty() {
                None
            } else {
                if n > GPDC_ROW_LIMIT {
                    return Err(StatsError::Unsupported {
                        message: "GPDC conditions through a dense n x n kernel (O(n^2) memory, \
                                  O(n^3) factorization) and refuses more than 4000 rows with a \
                                  non-empty conditioning set; use GpdcNystrom (`gpdc_nystrom`), \
                                  which conditions through a declared-rank low-rank kernel",
                    });
                }
                let at = if let Some(at) = factors.iter().position(|(key, _)| key.as_slice() == z) {
                    at
                } else {
                    let chol = gp_factor(request.columns, z, n, self)?;
                    if factors.len() >= GPDC_FACTOR_CACHE {
                        factors.remove(0);
                    }
                    factors.push((z.to_vec(), chol));
                    factors.len() - 1
                };
                Some(factors[at].1.as_slice())
            };
            let rx = gp_residual(request.columns[q.x], factor, self.ridge)?;
            let ry = gp_residual(request.columns[q.y], factor, self.ridge)?;
            // A constant series, or one the GP explains completely from Z, has no distance
            // variance left: dCor is undefined, and reporting 0 (p = 1) would be an
            // independence verdict manufactured by the absence of information.
            ensure_residual_information(request.columns[q.x], &rx)?;
            ensure_residual_information(request.columns[q.y], &ry)?;
            results.push(residual_dcor_permutation_result(
                &rx,
                &ry,
                ResidualNull {
                    salt: 0x69DC_u64 ^ query_stream_salt(&[q.x], &[q.y], z),
                    n_perm,
                    block_size,
                },
                ctx,
            ));
        }
        Ok(CiBatchResult { results })
    }

    fn attainable_p(&self) -> super::types::AttainableP {
        super::types::AttainableP::NonparametricPermutation
    }
}

/// How a GPDC-family test permutes its residuals.
#[derive(Clone, Copy)]
struct ResidualNull {
    /// Stream salt of the query (distinguishes the test family and the query).
    salt: u64,
    /// Permutation replicates.
    n_perm: usize,
    /// Contiguous block length (1: exchangeable shuffle).
    block_size: usize,
}

/// Distance correlation of two Z-free residual series with its permutation p-value.
///
/// The distance correlation is `O(n log n)` time and `O(n)` memory (sorting / Fenwick
/// identities): neither the statistic nor any permutation replicate builds an `n x n`
/// distance matrix. The null permutes the Y residuals (Z influence already removed) and
/// recomputes dCor; the add-one p-value keeps it in `(0, 1]`. `block_size > 1` permutes
/// contiguous blocks so the residual's serial dependence survives into the null; otherwise
/// it is an ordinary exchangeable shuffle.
fn residual_dcor_permutation_result(
    rx: &[f64],
    ry: &[f64],
    null: ResidualNull,
    ctx: &ExecutionContext,
) -> CiResult {
    let dcor = distance_correlation(rx, ry);
    let mut ry_perm = ry.to_vec();
    let mut rng = ctx.rng.stream_for(StreamDomain::StatsCi, null.salt);
    let mut null_ge = 0u32;
    for _ in 0..null.n_perm {
        if null.block_size > 1 {
            block_permute_contiguous(&mut ry_perm, null.block_size, &mut rng);
        } else {
            shuffle(&mut rng, &mut ry_perm);
        }
        if distance_correlation(rx, &ry_perm) >= dcor {
            null_ge = null_ge.saturating_add(1);
        }
    }
    let p = (1.0 + f64::from(null_ge)) / (1.0 + null.n_perm as f64);
    CiResult { statistic: dcor, p_value: p, df: 0.0, ci: None }
}

/// Refuse a residual carrying no information about its column.
fn ensure_residual_information(raw: &[f64], resid: &[f64]) -> Result<(), StatsError> {
    if residual_is_uninformative(raw, resid) {
        return Err(StatsError::Unsupported {
            message: "GPDC: a series is constant or fully explained by the conditioning set, so \
                      its distance correlation is undefined (not zero)",
        });
    }
    Ok(())
}

/// Z columns standardised to zero mean and unit SD (row-major `n × |Z|`; a constant column
/// becomes zeros), so the fixed RBF length scale means the same thing at any scale of Z.
fn standardized_z(columns: &[&[f64]], z: &[usize], n: usize) -> Result<Vec<f64>, StatsError> {
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
    Ok(z_std)
}

/// Cholesky factor of `K + λI` on the standardised Z columns (row-major `n × n`).
fn gp_factor(columns: &[&[f64]], z: &[usize], n: usize, gp: &Gpdc) -> Result<Vec<f64>, StatsError> {
    let zdim = z.len();
    let z_std = standardized_z(columns, z, n)?;
    // Gram on standardised Z (sum of RBF over Z dims) plus the ridge.
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
    crate::gram::cholesky_spd(&k, n)
        .ok_or_else(|| StatsError::Backend("GPDC kernel factorization failed".into()))
}

/// GP residual of `y` given the Cholesky `factor` of `K + λI` (`None`: empty Z).
///
/// With `(K+λI)α = y_c` the mean prediction is `Kα = y_c − λα` (MM-008), so the residual is
/// `y_c − Kα = λα` — one triangular solve, no `O(n²)` matrix-vector product.
fn gp_residual(y: &[f64], factor: Option<&[f64]>, ridge: f64) -> Result<Vec<f64>, StatsError> {
    let n = y.len();
    if y.iter().any(|v| !v.is_finite()) {
        return Err(StatsError::Shape { message: "non-finite response in GPDC residualization" });
    }
    let mean = y.iter().sum::<f64>() / n as f64;
    let centered: Vec<f64> = y.iter().map(|value| value - mean).collect();
    let Some(chol) = factor else {
        return Ok(centered);
    };
    let alpha = crate::gram::chol_solve(chol, n, &centered)
        .ok_or_else(|| StatsError::Backend("GPDC kernel solve failed".into()))?;
    Ok(alpha.into_iter().map(|a| ridge * a).collect())
}

/// Largest Nyström rank [`GpdcNystrom`] accepts: the `m × m` normal equations are accumulated
/// per row block, so memory grows as `m²` times the block count.
pub const GPDC_NYSTROM_MAX_INDUCING_POINTS: usize = 1_024;

/// Row blocks the Nyström accumulations are split into at most. The split depends only on
/// the row count, so the sums (and every result) are identical at any thread count.
const NYSTROM_MAX_ROW_BLOCKS: usize = 32;

/// Rows per Nyström row block, at least this many.
const NYSTROM_MIN_BLOCK_ROWS: usize = 2_048;

/// Relative diagonal jitter of the Nyström normal equations: the RBF kernel matrix of
/// inducing points is numerically rank deficient, and without it the factorization of
/// `K_mn K_nm + λ K_mm` would fail on exactly the smooth conditioning sets GPDC is for.
const NYSTROM_JITTER: f64 = 1e-10;

/// Scalable GPDC: Nyström (subset-of-regressors) RBF-GP residualization of X and Y on Z
/// followed by the same distance correlation and permutation null as [`Gpdc`].
///
/// This is a separate construction from [`Gpdc`], not an approximation switch: it matches
/// `Gpdc` in kernel, standardisation and ridge, but replaces the full-rank predictor
/// `K (K + λI)⁻¹ y` by the rank-`m` one `K_nm (K_mn K_nm + λ K_mm + εI)⁻¹ K_mn y`, where the
/// `m = inducing_points` inducing rows are a seeded draw without replacement from the
/// sample (`inducing_seed`, independent of the execution seed, so the residual series is a
/// function of the data and these two fields alone) and `ε` is [`NYSTROM_JITTER`] times the
/// mean diagonal. Normal equations are accumulated in row blocks and X and Y are two
/// right-hand sides of one factorization, so time is `O(n m²)` for the fit and `O(n m)` per
/// series, and memory is `O(n + m²)` — never `n × n` — which makes 100 000 rows with
/// `m = 256` a matter of seconds, where [`Gpdc`] refuses more than 4000 rows. With
/// `inducing_points >= n` every row is an inducing point.
///
/// The rank limits what Z structure is removed, not which null is used: a Z effect that the
/// `m`-point kernel cannot represent leaves Z-dependence in both residuals and inflates the
/// rejection rate, which is why the construction has its own Type I / power calibration.
/// Factory id: `gpdc_nystrom`.
#[derive(Clone, Debug)]
pub struct GpdcNystrom {
    /// RBF length scale (on standardised Z).
    pub length_scale: f64,
    /// Ridge.
    pub ridge: f64,
    /// Nyström rank `m`: inducing rows drawn from the sample (at most
    /// [`GPDC_NYSTROM_MAX_INDUCING_POINTS`]).
    pub inducing_points: usize,
    /// Seed of the inducing-row draw.
    pub inducing_seed: u64,
}

impl Default for GpdcNystrom {
    fn default() -> Self {
        Self::new()
    }
}

impl GpdcNystrom {
    /// Construct with defaults: `Gpdc`'s length scale and ridge, 256 inducing points.
    #[must_use]
    pub fn new() -> Self {
        Self {
            length_scale: 1.0,
            ridge: 1e-3,
            inducing_points: 256,
            inducing_seed: 0x4E59_5354_0001,
        }
    }
}

impl ConditionalIndependenceTest for GpdcNystrom {
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
        if self.inducing_points == 0 || self.inducing_points > GPDC_NYSTROM_MAX_INDUCING_POINTS {
            return Err(StatsError::Shape {
                message: "GPDC Nystrom inducing_points must be between 1 and 1024",
            });
        }
        let request = &prepared.bind_request(request);
        let n = request.columns.first().map_or(0, |c| c.len());
        if n == 0 {
            return Err(StatsError::Shape { message: "no columns" });
        }
        // As for `Gpdc`, both series are residualized on Z before permuting, so a
        // contiguous-block permutation of the Y residual is a valid null.
        let block_size = requested_block_size(request.significance);
        let n_perm = nonparametric_permutation_count(request.significance);
        let mut fits: Vec<(Vec<usize>, NystromFit)> = Vec::new();
        let mut results = Vec::with_capacity(request.queries.len());
        for q in request.queries {
            let z = &request.z_flat[q.z_start..q.z_start + q.z_len];
            let fit: Option<&NystromFit> = if z.is_empty() {
                None
            } else {
                let at = if let Some(at) = fits.iter().position(|(key, _)| key.as_slice() == z) {
                    at
                } else {
                    let fit = nystrom_fit(request.columns, z, n, self, ctx)?;
                    if fits.len() >= GPDC_FACTOR_CACHE {
                        fits.remove(0);
                    }
                    fits.push((z.to_vec(), fit));
                    fits.len() - 1
                };
                Some(&fits[at].1)
            };
            let [rx, ry] =
                nystrom_residuals(fit, [request.columns[q.x], request.columns[q.y]], ctx)?;
            ensure_residual_information(request.columns[q.x], &rx)?;
            ensure_residual_information(request.columns[q.y], &ry)?;
            results.push(residual_dcor_permutation_result(
                &rx,
                &ry,
                ResidualNull {
                    salt: 0x69DD_u64 ^ query_stream_salt(&[q.x], &[q.y], z),
                    n_perm,
                    block_size,
                },
                ctx,
            ));
        }
        Ok(CiBatchResult { results })
    }

    fn attainable_p(&self) -> super::types::AttainableP {
        super::types::AttainableP::NonparametricPermutation
    }
}

/// The Nyström fit for one conditioning set: standardised Z, inducing rows and the Cholesky
/// factor of `K_mn K_nm + λ K_mm + εI`.
#[derive(Debug)]
struct NystromFit {
    z_std: Vec<f64>,
    zdim: usize,
    inducing: Vec<f64>,
    rank: usize,
    ls2: f64,
    chol: Vec<f64>,
}

impl NystromFit {
    /// Kernel row of one standardised Z row against the inducing rows.
    fn kernel_row(&self, row: usize, out: &mut [f64]) {
        rbf_row(
            &self.z_std[row * self.zdim..(row + 1) * self.zdim],
            &self.inducing,
            self.zdim,
            self.ls2,
            out,
        );
    }
}

/// `exp(-d² / (2 ℓ²))` between `point` and each row of row-major `centres`.
fn rbf_row(point: &[f64], centres: &[f64], zdim: usize, ls2: f64, out: &mut [f64]) {
    for (slot, centre) in out.iter_mut().zip(centres.chunks_exact(zdim)) {
        let mut d2 = 0.0;
        for (a, b) in point.iter().zip(centre) {
            let diff = a - b;
            d2 += diff * diff;
        }
        *slot = (-0.5 * d2 / ls2).exp();
    }
}

/// `(rows per block, block count)` for the Nyström accumulations over `n` rows.
fn nystrom_blocks(n: usize) -> (usize, usize) {
    let block_rows = n.div_ceil(NYSTROM_MAX_ROW_BLOCKS).max(NYSTROM_MIN_BLOCK_ROWS);
    (block_rows, n.div_ceil(block_rows))
}

/// Bytes a Nyström fit over `n` rows, `zdim` conditioning columns and rank `m` plans to hold.
fn nystrom_bytes(n: usize, zdim: usize, m: usize) -> Option<u64> {
    let (_, blocks) = nystrom_blocks(n);
    let per_row = zdim.checked_add(9)?;
    let words = n
        .checked_mul(per_row)?
        .checked_add(blocks.checked_add(3)?.checked_mul(m.checked_mul(m)?)?)?
        .checked_add(m.checked_mul(zdim)?)?;
    u64::try_from(words.checked_mul(core::mem::size_of::<f64>())?).ok()
}

/// The `m` inducing rows (ascending): a seeded draw without replacement from `n` rows,
/// keyed by the conditioning set so different sets draw independent rows.
fn nystrom_inducing_rows(n: usize, m: usize, seed: u64, z: &[usize]) -> Vec<usize> {
    let mut rng = antecedent_core::RngFactory::from_seed(seed)
        .stream_for(StreamDomain::StatsCi, query_stream_salt(&[], &[], z));
    let mut rows: Vec<usize> = (0..n).collect();
    for i in 0..m {
        let j = i + unbiased_index(&mut rng, n - i);
        rows.swap(i, j);
    }
    rows.truncate(m);
    rows.sort_unstable();
    rows
}

/// Fit the Nyström predictor for conditioning set `z`.
fn nystrom_fit(
    columns: &[&[f64]],
    z: &[usize],
    n: usize,
    gp: &GpdcNystrom,
    ctx: &ExecutionContext,
) -> Result<NystromFit, StatsError> {
    let m = gp.inducing_points.min(n);
    let zdim = z.len();
    crate::matching::check_budget(ctx, "GPDC Nystrom fit", nystrom_bytes(n, zdim, m))?;
    let z_std = standardized_z(columns, z, n)?;
    let rows = nystrom_inducing_rows(n, m, gp.inducing_seed, z);
    let mut inducing = Vec::with_capacity(m * zdim);
    for &r in &rows {
        inducing.extend_from_slice(&z_std[r * zdim..(r + 1) * zdim]);
    }
    let ls2 = gp.length_scale * gp.length_scale;
    let mut fit = NystromFit { z_std, zdim, inducing, rank: m, ls2, chol: Vec::new() };

    // K_mn K_nm, lower triangle, summed over row blocks in block order.
    let (block_rows, blocks) = nystrom_blocks(n);
    let partials = ctx.map_indexed(blocks, |block, inner| {
        if inner.cancellation.is_cancelled() {
            return Err(StatsError::Cancelled);
        }
        let mut acc = vec![0.0; m * m];
        let mut k = vec![0.0; m];
        for row in block * block_rows..((block + 1) * block_rows).min(n) {
            fit.kernel_row(row, &mut k);
            for (j, &kj) in k.iter().enumerate() {
                for (slot, kl) in acc[j * m..=j * m + j].iter_mut().zip(&k) {
                    *slot += kj * kl;
                }
            }
        }
        Ok(acc)
    })?;
    let mut normal = vec![0.0; m * m];
    for partial in &partials {
        for (slot, v) in normal.iter_mut().zip(partial) {
            *slot += v;
        }
    }
    drop(partials);
    // + λ K_mm, mirrored to the full symmetric matrix, + jitter on the diagonal.
    let mut k_row = vec![0.0; m];
    for j in 0..m {
        rbf_row(&fit.inducing[j * zdim..(j + 1) * zdim], &fit.inducing, zdim, ls2, &mut k_row);
        for l in 0..=j {
            normal[j * m + l] += gp.ridge * k_row[l];
        }
    }
    let mean_diagonal = (0..m).map(|j| normal[j * m + j]).sum::<f64>() / m as f64;
    for j in 0..m {
        normal[j * m + j] += NYSTROM_JITTER * mean_diagonal;
        for l in 0..j {
            normal[l * m + j] = normal[j * m + l];
        }
    }
    fit.chol = crate::gram::cholesky_spd(&normal, m).ok_or_else(|| {
        StatsError::Backend("GPDC Nystrom normal equations not positive definite".into())
    })?;
    Ok(fit)
}

/// Nyström residuals of two series given the fit (`None`: empty Z, only centred).
fn nystrom_residuals(
    fit: Option<&NystromFit>,
    targets: [&[f64]; 2],
    ctx: &ExecutionContext,
) -> Result<[Vec<f64>; 2], StatsError> {
    let n = targets[0].len();
    let mut centered: [Vec<f64>; 2] = [Vec::new(), Vec::new()];
    for (slot, y) in centered.iter_mut().zip(targets) {
        if y.len() != n || y.iter().any(|v| !v.is_finite()) {
            return Err(StatsError::Shape {
                message: "non-finite response in GPDC residualization",
            });
        }
        let mean = y.iter().sum::<f64>() / n as f64;
        *slot = y.iter().map(|value| value - mean).collect();
    }
    let Some(fit) = fit else {
        return Ok(centered);
    };
    let m = fit.rank;
    let (block_rows, blocks) = nystrom_blocks(n);
    // K_mn y for both series.
    let moments = ctx.map_indexed(blocks, |block, inner| {
        if inner.cancellation.is_cancelled() {
            return Err(StatsError::Cancelled);
        }
        let mut sums = [vec![0.0; m], vec![0.0; m]];
        let mut k = vec![0.0; m];
        for row in block * block_rows..((block + 1) * block_rows).min(n) {
            fit.kernel_row(row, &mut k);
            for (sum, y) in sums.iter_mut().zip(&centered) {
                for (slot, kl) in sum.iter_mut().zip(&k) {
                    *slot += kl * y[row];
                }
            }
        }
        Ok(sums)
    })?;
    let mut coefficients: [Vec<f64>; 2] = [vec![0.0; m], vec![0.0; m]];
    for sums in &moments {
        for (total, sum) in coefficients.iter_mut().zip(sums) {
            for (slot, v) in total.iter_mut().zip(sum) {
                *slot += v;
            }
        }
    }
    for coefficient in &mut coefficients {
        *coefficient = crate::gram::chol_solve(&fit.chol, m, coefficient)
            .ok_or_else(|| StatsError::Backend("GPDC Nystrom solve failed".into()))?;
    }
    // Residual = centred y - K_nm c.
    let parts = ctx.map_indexed(blocks, |block, inner| {
        if inner.cancellation.is_cancelled() {
            return Err(StatsError::Cancelled);
        }
        let range = block * block_rows..((block + 1) * block_rows).min(n);
        let mut out = [Vec::with_capacity(range.len()), Vec::with_capacity(range.len())];
        let mut k = vec![0.0; m];
        for row in range {
            fit.kernel_row(row, &mut k);
            for ((part, y), c) in out.iter_mut().zip(&centered).zip(&coefficients) {
                let fitted: f64 = k.iter().zip(c).map(|(kl, cl)| kl * cl).sum();
                part.push(y[row] - fitted);
            }
        }
        Ok(out)
    })?;
    let mut residuals: [Vec<f64>; 2] = [Vec::with_capacity(n), Vec::with_capacity(n)];
    for part in parts {
        for (total, piece) in residuals.iter_mut().zip(part) {
            total.extend(piece);
        }
    }
    Ok(residuals)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ci::types::{
        CiBatchRequest, CiQuery, CiWorkspace, ConfidenceMethod, SignificanceMethod,
    };

    /// A test that cannot honour `block_size` must refuse it rather than discard it — but these
    /// three draw that line in different places, and this pins all three.
    ///
    /// `KnnDependence` and `SymbolicCmi` build their null as an exchange within Z strata
    /// (`z_permutation_strata` / `discrete_strata`). **With a conditioning set** those strata are
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

        // Rounded so symbols recur: distinct continuous values would each be a singleton
        // category, leaving `SymbolicCmi` no permutation support.
        let x: Vec<f64> = (0..60).map(|i| (f64::from(i) * 0.3).sin().round()).collect();
        let y: Vec<f64> = (0..60).map(|i| (f64::from(i) * 0.3).cos().round()).collect();
        let z: Vec<f64> = (0..60).map(|i| (f64::from(i) * 0.11).sin().round()).collect();
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

    /// Non-integer symbols are labels: shifting every symbol by 0.3 (and conditioning on a
    /// fractional Z) yields the same statistic and permutation p-value as the integer coding.
    #[test]
    fn symbolic_treats_non_integer_values_as_symbols() {
        let n = 90usize;
        let z: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        let x: Vec<f64> = (0..n).map(|i| ((i / 3) % 3) as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| ((i / 3 + i / 9) % 3) as f64).collect();
        let shift = |v: &[f64]| v.iter().map(|a| a * 1.5 + 0.3).collect::<Vec<f64>>();
        let (xs, ys, zs) = (shift(&x), shift(&y), shift(&z));
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
        let run = |cols: [&[f64]; 3]| {
            let req = CiBatchRequest {
                columns: &cols,
                queries: &queries,
                z_flat: &[2],
                significance: SignificanceMethod::BlockShuffle { replicates: 49, block_size: 1 },
                confidence: ConfidenceMethod::default(),
            };
            let mut ws = CiWorkspace::default();
            let ctx = ExecutionContext::for_tests(4);
            SymbolicCmi::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap().results[0]
        };
        let coded = run([&x, &y, &z]);
        let labelled = run([&xs, &ys, &zs]);
        assert!(coded.statistic > 0.0);
        assert_eq!(coded.statistic.to_bits(), labelled.statistic.to_bits());
        assert_eq!(coded.p_value.to_bits(), labelled.p_value.to_bits());
    }

    /// When every X symbol occurs once, every rearrangement of Y is a relabelled copy of the
    /// observed table, so the null never moves and the add-one p-value is exactly 1. That is a
    /// valid, conservative p-value (the data cannot reject independence), not an error.
    #[test]
    fn symbolic_null_that_cannot_move_gives_conservative_p_of_one() {
        let n = 30usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        let cols: [&[f64]; 2] = [&x, &y];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::BlockShuffle { replicates: 19, block_size: 1 },
            confidence: ConfidenceMethod::default(),
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(3);
        let out = SymbolicCmi::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        assert_eq!(out.results[0].p_value.to_bits(), 1.0_f64.to_bits());
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

    use crate::test_util::centered_uniform as lcg_noise;

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
        // The dependent pair (y = x) has dCor 1, which no permutation reaches: the add-one p is
        // at its floor 1/50. The independent pair is decided on the statistic, not on a
        // Monte Carlo p-value that is uniform under H0 and would trip a fixed cut-off by chance.
        assert!(out.results[0].p_value < 0.05, "dependent p={}", out.results[0].p_value);
        assert!(
            out.results[0].statistic > out.results[1].statistic + 0.3,
            "dCor dependent={} independent={}",
            out.results[0].statistic,
            out.results[1].statistic
        );
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

    /// Y = Z exactly: within every Z stratum Y is constant, so no permutation can change the
    /// statistic and the add-one p-value is 1 by construction. That used to be reported as
    /// "independent" (p = 1); it is no information, and is refused.
    #[test]
    fn symbolic_refuses_y_determined_by_z() {
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
        let err = SymbolicCmi::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap_err();
        assert!(matches!(err, StatsError::Unsupported { .. }), "{err:?}");
    }

    /// Within-stratum Y variation, X independent of Y given Z: the null has support, and the
    /// p-value is an honest large one, not the degenerate p = 1.
    #[test]
    fn symbolic_null_preserves_yz_dependence() {
        let n = 240usize;
        let z: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        // Y = Z plus an independent binary wobble; X independent of both given Z.
        let y: Vec<f64> = (0..n).map(|i| (i % 3) as f64 * 2.0 + ((i / 3) % 2) as f64).collect();
        let x: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 5) as f64).collect();
        let cols: [&[f64]; 3] = [&x, &y, &z];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
        let z_flat = [2usize];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::BlockShuffle { replicates: 199, block_size: 1 },
            confidence: ConfidenceMethod::default(),
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(13);
        let out = SymbolicCmi::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        assert!(out.results[0].p_value > 0.05, "p={}", out.results[0].p_value);
    }

    #[test]
    fn symbolic_refuses_non_finite_symbols() {
        let n = 40usize;
        let mut x: Vec<f64> = (0..n).map(|i| (i % 3) as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| ((i / 3) % 3) as f64).collect();
        x[5] = f64::NAN;
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
        let ctx = ExecutionContext::for_tests(2);
        assert!(SymbolicCmi::new().test_batch_adhoc(&req, &mut ws, &ctx).is_err());
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
        // Under H0 the p-value is uniform, so any single seed has a ~5% chance of a small p.
        // A null that ignored Y|Z would put every seed near the floor; require most seeds to
        // sit above 0.05.
        let mut ws = CiWorkspace::default();
        let ps: Vec<f64> = (14..20u64)
            .map(|seed| {
                let ctx = ExecutionContext::for_tests(seed);
                KnnDependence::new(3).test_batch_adhoc(&req, &mut ws, &ctx).unwrap().results[0]
                    .p_value
            })
            .collect();
        let above = ps.iter().filter(|&&p| p > 0.05).count();
        assert!(above >= 4, "kNN conditional null reports tiny p-values: {ps:?}");
    }

    /// stats-ci-1: continuous Z with Y←Z must not yield Type I ≈ 1 under X ⊥ Y | Z.
    #[test]
    fn knn_continuous_z_null_type_i_not_near_one() {
        let n = 120usize;
        let trials = 200u32;
        let alpha = 0.05;
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(0x000C_11C0_u64);
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

    /// Bivariate Z with X and Y both driven by the first Z coordinate, so X ⊥ Y | Z. Continuous
    /// bivariate Z used to be grouped by the norm of Z, pairing z = (2, 0) with z = (-2, 0) and
    /// rejecting every trial; it is now refused. Discrete bivariate Z keeps exact level strata
    /// and a controlled Type I rate.
    #[test]
    fn knn_bivariate_z_conditional_null() {
        let n = 100usize;
        let trials = 60u32;
        let alpha = 0.05;
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(0x2D_2D_u64);
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 2 }];
        let z_flat = [2usize, 3];
        let ci = KnnDependence::new(5);
        let mut rejects = 0u32;
        for t in 0..trials {
            let mut rng = ctx.rng.stream_for(StreamDomain::StatsCi, 0x2D2_u64 ^ u64::from(t));
            let mut normal = || {
                let u = (rng.next_u64() as f64) / (u64::MAX as f64);
                let v = (rng.next_u64() as f64) / (u64::MAX as f64);
                (-2.0 * u.max(1e-12).ln()).sqrt() * (2.0 * std::f64::consts::PI * v).cos()
            };
            let z1: Vec<f64> = (0..n).map(|_| normal()).collect();
            let z2: Vec<f64> = (0..n).map(|_| normal()).collect();
            let x: Vec<f64> = z1.iter().map(|&a| a + 0.35 * normal()).collect();
            let y: Vec<f64> = z1.iter().map(|&a| a + 0.35 * normal()).collect();
            let cols: [&[f64]; 4] = [&x, &y, &z1, &z2];
            let req = CiBatchRequest {
                columns: &cols,
                queries: &queries,
                z_flat: &z_flat,
                significance: SignificanceMethod::BlockShuffle { replicates: 49, block_size: 1 },
                confidence: ConfidenceMethod::None,
            };
            let err = ci.test_batch_adhoc(&req, &mut ws, &ctx).unwrap_err();
            assert!(matches!(err, StatsError::Unsupported { .. }), "{err:?}");

            let d1: Vec<f64> = z1.iter().map(|v| v.round().clamp(-1.0, 1.0)).collect();
            let d2: Vec<f64> = z2.iter().map(|v| f64::from(u8::from(*v > 0.0))).collect();
            let x: Vec<f64> = d1.iter().map(|&a| 2.0 * a + 0.35 * normal()).collect();
            let y: Vec<f64> = d1.iter().map(|&a| 2.0 * a + 0.35 * normal()).collect();
            let cols: [&[f64]; 4] = [&x, &y, &d1, &d2];
            let req = CiBatchRequest { columns: &cols, ..req };
            if ci.test_batch_adhoc(&req, &mut ws, &ctx).unwrap().results[0].p_value < alpha {
                rejects += 1;
            }
        }
        let rate = f64::from(rejects) / f64::from(trials);
        // alpha + 3 binomial SE at 60 trials is about 0.135.
        assert!(
            rate < 0.14,
            "discrete bivariate-Z kNN type I inflated: {rate} ({rejects}/{trials})"
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
        let ctx = ExecutionContext::for_tests(0x0069_DC5C_u64);
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

    /// The permutation stream is keyed by the query, not its position in the batch, for every
    /// nonparametric test.
    #[test]
    fn nonparametric_p_values_are_independent_of_batch_position() {
        let n = 60usize;
        let x: Vec<f64> = (0..n).map(|i| ((i * 7 + 3) % 5) as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| ((i * 3 + 1) % 4) as f64).collect();
        let w: Vec<f64> = (0..n).map(|i| ((i * 5 + 2) % 6) as f64).collect();
        let cols: [&[f64]; 3] = [&x, &y, &w];
        let target = CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 };
        let other = CiQuery { x: 0, y: 2, z_start: 0, z_len: 0 };
        let sig = SignificanceMethod::BlockShuffle { replicates: 49, block_size: 1 };
        let tests: [(&str, &dyn ConditionalIndependenceTest); 3] = [
            ("knn", &KnnDependence::new(3)),
            ("symbolic", &SymbolicCmi::new()),
            ("gpdc", &Gpdc::new()),
        ];
        for (name, ci) in tests {
            let run = |queries: &[CiQuery]| {
                let req = CiBatchRequest {
                    columns: &cols,
                    queries,
                    z_flat: &[],
                    significance: sig,
                    confidence: ConfidenceMethod::None,
                };
                let mut ws = CiWorkspace::default();
                let ctx = ExecutionContext::for_tests(77);
                ci.test_batch_adhoc(&req, &mut ws, &ctx).unwrap()
            };
            let alone = run(&[target]);
            let second = run(&[other, target]);
            assert_eq!(
                alone.results[0].p_value.to_bits(),
                second.results[1].p_value.to_bits(),
                "{name}: p depends on batch position"
            );
        }
    }

    #[test]
    fn permutation_tests_report_their_resolution() {
        let analytic = SignificanceMethod::Analytic;
        let b99 = SignificanceMethod::BlockShuffle { replicates: 99, block_size: 1 };
        for ci in [
            &KnnDependence::new(3) as &dyn ConditionalIndependenceTest,
            &MixedKnnDependence::new(3),
            &SymbolicCmi::new(),
            &Gpdc::new(),
        ] {
            assert!((ci.min_attainable_p(analytic) - 0.02).abs() < 1e-15);
            assert!((ci.min_attainable_p(b99) - 0.01).abs() < 1e-15);
        }
        assert!(crate::ci::ensure_alpha_resolvable(0.02, 0.01).is_err());
        assert!(crate::ci::ensure_alpha_resolvable(0.02, 0.05).is_ok());
    }

    #[test]
    fn knn_refuses_constant_series() {
        let n = 30usize;
        let x = lcg_noise(n, 8);
        let y = vec![3.0; n];
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
        let ctx = ExecutionContext::for_tests(3);
        let err = KnnDependence::new(3).test_batch_adhoc(&req, &mut ws, &ctx).unwrap_err();
        assert!(matches!(err, StatsError::Unsupported { .. }), "{err:?}");
    }

    #[test]
    fn gpdc_refuses_constant_series_and_oversized_inputs() {
        let n = 30usize;
        let x = lcg_noise(n, 9);
        let y = vec![3.0; n];
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
        let ctx = ExecutionContext::for_tests(3);
        let err = Gpdc::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap_err();
        assert!(matches!(err, StatsError::Unsupported { .. }), "{err:?}");

        // An unconditional test never builds an n x n matrix (distance correlation is
        // O(n log n) in time and O(n) in memory), so it runs above the limit; a conditional
        // one factorizes a dense kernel and is refused.
        let big = GPDC_ROW_LIMIT + 1;
        let bx = lcg_noise(big, 5);
        let by = lcg_noise(big, 6);
        let bz = lcg_noise(big, 7);
        let big_cols: [&[f64]; 3] = [&bx, &by, &bz];
        let unconditional = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let big_req = CiBatchRequest {
            columns: &big_cols,
            queries: &unconditional,
            z_flat: &[],
            significance: SignificanceMethod::BlockShuffle { replicates: 9, block_size: 1 },
            confidence: ConfidenceMethod::default(),
        };
        let done = Gpdc::new().test_batch_adhoc(&big_req, &mut ws, &ctx).unwrap();
        let dcor = crate::dcor::distance_correlation(
            &big_cols[0]
                .iter()
                .map(|v| v - bx.iter().sum::<f64>() / big as f64)
                .collect::<Vec<_>>(),
            &big_cols[1]
                .iter()
                .map(|v| v - by.iter().sum::<f64>() / big as f64)
                .collect::<Vec<_>>(),
        );
        assert!((done.results[0].statistic - dcor).abs() < 1e-12);
        let conditional = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
        let big_req = CiBatchRequest { queries: &conditional, z_flat: &[2], ..big_req };
        let err = Gpdc::new().test_batch_adhoc(&big_req, &mut ws, &ctx).unwrap_err();
        assert!(matches!(err, StatsError::Unsupported { .. }), "{err:?}");
    }

    /// The residual is `y_c - K a` with `(K + lambda I) a = y_c` solved by an independent dense
    /// Gauss-Jordan; the implementation returns `lambda a` from the Cholesky solve.
    #[test]
    fn gpdc_residual_matches_dense_gp_regression() {
        let n = 12usize;
        let z: Vec<f64> = (0..n).map(|i| 0.4 * i as f64 + (i as f64 * 0.7).sin()).collect();
        let y: Vec<f64> = (0..n).map(|i| (z[i] * 0.5).sin() + 0.1 * ((i * 5 % 7) as f64)).collect();
        let gp = Gpdc::new();
        let cols: [&[f64]; 2] = [&y, &z];
        let chol = gp_factor(&cols, &[1], n, &gp).unwrap();
        let got = gp_residual(&y, Some(&chol), gp.ridge).unwrap();

        let mz = z.iter().sum::<f64>() / n as f64;
        let sd = (z.iter().map(|v| (v - mz) * (v - mz)).sum::<f64>() / n as f64).sqrt();
        let zs: Vec<f64> = z.iter().map(|v| (v - mz) / sd).collect();
        let my = y.iter().sum::<f64>() / n as f64;
        let yc: Vec<f64> = y.iter().map(|v| v - my).collect();
        let kmat = |i: usize, j: usize| (-0.5 * (zs[i] - zs[j]) * (zs[i] - zs[j])).exp();
        // Augmented [K + lambda I | y_c], Gauss-Jordan with partial pivoting.
        let mut a = vec![vec![0.0; n + 1]; n];
        for i in 0..n {
            for j in 0..n {
                a[i][j] = kmat(i, j) + if i == j { gp.ridge } else { 0.0 };
            }
            a[i][n] = yc[i];
        }
        for c in 0..n {
            let piv = (c..n).max_by(|&p, &q| a[p][c].abs().total_cmp(&a[q][c].abs())).unwrap();
            a.swap(c, piv);
            let d = a[c][c];
            for j in 0..=n {
                a[c][j] /= d;
            }
            for r in 0..n {
                if r != c {
                    let f = a[r][c];
                    for j in 0..=n {
                        let v = a[c][j];
                        a[r][j] -= f * v;
                    }
                }
            }
        }
        let alpha: Vec<f64> = (0..n).map(|i| a[i][n]).collect();
        for i in 0..n {
            let pred: f64 = (0..n).map(|j| kmat(i, j) * alpha[j]).sum();
            let want = yc[i] - pred;
            assert!((got[i] - want).abs() < 1e-8, "{i}: {} vs {want}", got[i]);
        }
    }

    /// Columns `[x, y, z1, z2]` with a smooth nonlinear confounder `(z1, z2)` and independent
    /// unit noises `e1`, `e2`: `x = 1.5 sin(1.5 z1) + 0.8 z2 + e1`,
    /// `y = 1.5 tanh(1.5 z1) - 0.8 z2 + e2 + y_on_x * x`. With `y_on_x = 0`, `X ⊥ Y | Z`.
    fn nystrom_sample(n: usize, seed: u64, y_on_x: f64) -> (Vec<Vec<f64>>, Vec<f64>, Vec<f64>) {
        let mut rng = ExecutionContext::for_tests(seed).rng.stream_for(StreamDomain::Test, 0x6E59);
        let z1: Vec<f64> = (0..n).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect();
        let z2: Vec<f64> = (0..n).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect();
        let e1: Vec<f64> = (0..n).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect();
        let e2: Vec<f64> = (0..n).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect();
        let x: Vec<f64> = (0..n).map(|i| 1.5 * (1.5 * z1[i]).sin() + 0.8 * z2[i] + e1[i]).collect();
        let y: Vec<f64> = (0..n)
            .map(|i| 1.5 * (1.5 * z1[i]).tanh() - 0.8 * z2[i] + e2[i] + y_on_x * x[i])
            .collect();
        (vec![x, y, z1, z2], e1, e2)
    }

    fn pearson(a: &[f64], b: &[f64]) -> f64 {
        let n = a.len() as f64;
        let (ma, mb) = (a.iter().sum::<f64>() / n, b.iter().sum::<f64>() / n);
        let (mut sab, mut saa, mut sbb) = (0.0, 0.0, 0.0);
        for (x, y) in a.iter().zip(b) {
            sab += (x - ma) * (y - mb);
            saa += (x - ma) * (x - ma);
            sbb += (y - mb) * (y - mb);
        }
        sab / (saa * sbb).sqrt()
    }

    /// Independent dense computation of the rank-`m` predictor on the same inducing rows: an
    /// explicit `n x m` kernel matrix, naive normal equations and a dense solve, with none of
    /// the row-block accumulation of the implementation.
    fn dense_nystrom_residual(
        columns: &[&[f64]],
        z: &[usize],
        y: &[f64],
        gp: &GpdcNystrom,
    ) -> Vec<f64> {
        let n = y.len();
        let m = gp.inducing_points.min(n);
        let z_std = standardized_z(columns, z, n).unwrap();
        let zdim = z.len();
        let rows = nystrom_inducing_rows(n, m, gp.inducing_seed, z);
        let ls2 = gp.length_scale * gp.length_scale;
        let kernel = |a: usize, b: usize| {
            let d2: f64 = (0..zdim)
                .map(|d| {
                    let diff = z_std[a * zdim + d] - z_std[b * zdim + d];
                    diff * diff
                })
                .sum();
            (-0.5 * d2 / ls2).exp()
        };
        let knm: Vec<Vec<f64>> =
            (0..n).map(|i| rows.iter().map(|&r| kernel(i, r)).collect()).collect();
        let mut normal = vec![0.0; m * m];
        for j in 0..m {
            for l in 0..m {
                let mut acc: f64 = knm.iter().map(|row| row[j] * row[l]).sum();
                acc += gp.ridge * kernel(rows[j], rows[l]);
                normal[j * m + l] = acc;
            }
        }
        let mean_diagonal = (0..m).map(|j| normal[j * m + j]).sum::<f64>() / m as f64;
        for j in 0..m {
            normal[j * m + j] += NYSTROM_JITTER * mean_diagonal;
        }
        let mean = y.iter().sum::<f64>() / n as f64;
        let centered: Vec<f64> = y.iter().map(|v| v - mean).collect();
        let rhs: Vec<f64> =
            (0..m).map(|j| knm.iter().zip(&centered).map(|(row, yc)| row[j] * yc).sum()).collect();
        let chol = crate::gram::cholesky_spd(&normal, m).unwrap();
        let c = crate::gram::chol_solve(&chol, m, &rhs).unwrap();
        (0..n)
            .map(|i| centered[i] - knm[i].iter().zip(&c).map(|(k, ci)| k * ci).sum::<f64>())
            .collect()
    }

    #[test]
    fn gpdc_nystrom_residuals_match_a_dense_reference_beyond_the_exact_limit() {
        let n = 6_000usize;
        assert!(n > GPDC_ROW_LIMIT);
        let (data, _, _) = nystrom_sample(n, 21, 0.0);
        let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
        let gp = GpdcNystrom { inducing_points: 24, ..GpdcNystrom::new() };
        let ctx = ExecutionContext::for_tests(5);
        let fit = nystrom_fit(&cols, &[2, 3], n, &gp, &ctx).unwrap();
        let [rx, ry] = nystrom_residuals(Some(&fit), [cols[0], cols[1]], &ctx).unwrap();
        for (got, target) in [(&rx, cols[0]), (&ry, cols[1])] {
            let want = dense_nystrom_residual(&cols, &[2, 3], target, &gp);
            let worst = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).fold(0.0, f64::max);
            assert!(worst < 1e-5, "worst residual gap {worst}");
        }
    }

    #[test]
    fn gpdc_nystrom_residual_recovers_the_noise_of_a_nonlinear_confounder() {
        let n = 6_000usize;
        let (data, e1, e2) = nystrom_sample(n, 22, 0.0);
        let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
        let ctx = ExecutionContext::for_tests(6);
        let fit = nystrom_fit(&cols, &[2, 3], n, &GpdcNystrom::new(), &ctx).unwrap();
        let [rx, ry] = nystrom_residuals(Some(&fit), [cols[0], cols[1]], &ctx).unwrap();
        // The truth is known: what is left after removing the smooth Z effect is the noise.
        assert!(pearson(&rx, &e1) > 0.99, "x residual vs noise {}", pearson(&rx, &e1));
        assert!(pearson(&ry, &e2) > 0.99, "y residual vs noise {}", pearson(&ry, &e2));
        // Removing Z must remove the dependence it induced between X and Y.
        let raw = crate::dcor::distance_correlation(cols[0], cols[1]);
        let cleaned = crate::dcor::distance_correlation(&rx, &ry);
        assert!(raw > 0.08 && cleaned < 0.05, "dCor before {raw}, after {cleaned}");
    }

    #[test]
    fn gpdc_nystrom_agrees_with_the_exact_gp_where_both_run() {
        let n = 1_200usize;
        let (data, _, _) = nystrom_sample(n, 23, 0.0);
        let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
        let ctx = ExecutionContext::for_tests(7);
        let exact_factor = gp_factor(&cols, &[2, 3], n, &Gpdc::new()).unwrap();
        let exact = gp_residual(cols[0], Some(&exact_factor), Gpdc::new().ridge).unwrap();
        let fit = nystrom_fit(&cols, &[2, 3], n, &GpdcNystrom::new(), &ctx).unwrap();
        let [low_rank, _] = nystrom_residuals(Some(&fit), [cols[0], cols[1]], &ctx).unwrap();
        assert!(pearson(&exact, &low_rank) > 0.95, "{}", pearson(&exact, &low_rank));
    }

    #[test]
    fn gpdc_nystrom_is_identical_at_any_thread_count() {
        let n = 5_000usize;
        let (data, _, _) = nystrom_sample(n, 24, 0.0);
        let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
        let serial = ExecutionContext::for_tests(8);
        let mut threaded = ExecutionContext::for_tests(8);
        threaded.parallelism = antecedent_core::Parallelism::bounded(
            antecedent_core::NonZeroThreadCount::new(4).unwrap(),
        );
        let run = |ctx: &ExecutionContext| {
            let fit = nystrom_fit(&cols, &[2, 3], n, &GpdcNystrom::new(), ctx).unwrap();
            nystrom_residuals(Some(&fit), [cols[0], cols[1]], ctx).unwrap()
        };
        let (a, b) = (run(&serial), run(&threaded));
        for (u, v) in a.iter().zip(&b) {
            assert!(u.iter().zip(v).all(|(p, q)| p.to_bits() == q.to_bits()));
        }
    }

    #[test]
    fn gpdc_nystrom_separates_dependence_given_the_confounder_beyond_the_exact_limit() {
        let n = 6_000usize;
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 2 }];
        let z_flat = [2usize, 3];
        let ctx = ExecutionContext::for_tests(9);
        let run = |y_on_x: f64| {
            let (data, _, _) = nystrom_sample(n, 25, y_on_x);
            let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
            let req = CiBatchRequest {
                columns: &cols,
                queries: &queries,
                z_flat: &z_flat,
                significance: SignificanceMethod::BlockShuffle { replicates: 49, block_size: 1 },
                confidence: ConfidenceMethod::None,
            };
            let mut ws = CiWorkspace::default();
            GpdcNystrom::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap().results[0]
        };
        let dependent = run(0.4);
        assert!(dependent.p_value <= 0.02 + 1e-12, "dependent p={}", dependent.p_value);
        assert!(dependent.statistic > 0.15, "dCor {}", dependent.statistic);
        let independent = run(0.0);
        assert!(independent.statistic < 0.05, "dCor {}", independent.statistic);
        assert!(independent.statistic + 0.1 < dependent.statistic);
    }

    #[test]
    fn gpdc_nystrom_refuses_invalid_rank_and_budget_instead_of_degrading() {
        let n = 300usize;
        let (data, _, _) = nystrom_sample(n, 26, 0.0);
        let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 1 }];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[2],
            significance: SignificanceMethod::BlockShuffle { replicates: 9, block_size: 1 },
            confidence: ConfidenceMethod::None,
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(10);
        for inducing_points in [0, GPDC_NYSTROM_MAX_INDUCING_POINTS + 1] {
            let gp = GpdcNystrom { inducing_points, ..GpdcNystrom::new() };
            assert!(matches!(
                gp.test_batch_adhoc(&req, &mut ws, &ctx),
                Err(StatsError::Shape { .. })
            ));
        }
        let mut limited = ExecutionContext::for_tests(10);
        limited.memory =
            antecedent_core::MemoryBudget { soft_limit_bytes: None, hard_limit_bytes: Some(1_024) };
        assert!(matches!(
            GpdcNystrom::new().test_batch_adhoc(&req, &mut ws, &limited),
            Err(StatsError::ResourceLimit { .. })
        ));
        let cancelled = ExecutionContext::for_tests(10);
        cancelled.cancellation.cancel();
        assert!(matches!(
            GpdcNystrom::new().test_batch_adhoc(&req, &mut ws, &cancelled),
            Err(StatsError::Cancelled)
        ));
        // More inducing points than rows: every row is one.
        let all = GpdcNystrom { inducing_points: 1_024, ..GpdcNystrom::new() };
        assert!(all.test_batch_adhoc(&req, &mut ws, &ctx).is_ok());
    }

    #[test]
    #[ignore = "acceptance workload: 100k rows; run with --ignored in a release build"]
    fn gpdc_nystrom_one_hundred_thousand_row_acceptance() {
        let n = 100_000usize;
        let (data, e1, _) = nystrom_sample(n, 27, 0.4);
        let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 2 }];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[2, 3],
            significance: SignificanceMethod::BlockShuffle { replicates: 49, block_size: 1 },
            confidence: ConfidenceMethod::None,
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::production(11, 4);
        let started = std::time::Instant::now();
        let out = GpdcNystrom::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        eprintln!("100k-row GPDC Nystrom took {:?}", started.elapsed());
        assert!(out.results[0].p_value <= 0.02 + 1e-12, "p={}", out.results[0].p_value);
        let fit = nystrom_fit(&cols, &[2, 3], n, &GpdcNystrom::new(), &ctx).unwrap();
        let [rx, _] = nystrom_residuals(Some(&fit), [cols[0], cols[1]], &ctx).unwrap();
        assert!(pearson(&rx, &e1) > 0.99);
    }

    /// At 2000 rows the exact GPDC test keeps its construction: the statistic and p-value are,
    /// bit for bit, those of the dense kernel factorization, GP residuals and residual
    /// permutation null it has always run, and the low-rank test is a different construction.
    #[test]
    fn exact_conditional_gpdc_retained_at_small_sizes() {
        let n = 2_000usize;
        assert!(n <= GPDC_ROW_LIMIT);
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 2 }];
        let z_flat = [2usize, 3];
        let significance = SignificanceMethod::BlockShuffle { replicates: 49, block_size: 1 };
        for (y_on_x, label) in [(0.0, "independent"), (0.3, "dependent")] {
            let (data, _, _) = nystrom_sample(n, 6_075_547, y_on_x);
            let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
            let req = CiBatchRequest {
                columns: &cols,
                queries: &queries,
                z_flat: &z_flat,
                significance,
                confidence: ConfidenceMethod::None,
            };
            let ctx = ExecutionContext::for_tests(12);
            let mut ws = CiWorkspace::default();
            let got = Gpdc::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap().results[0];

            let gp = Gpdc::new();
            let factor = gp_factor(&cols, &z_flat, n, &gp).unwrap();
            let rx = gp_residual(cols[0], Some(&factor), gp.ridge).unwrap();
            let ry = gp_residual(cols[1], Some(&factor), gp.ridge).unwrap();
            let want = residual_dcor_permutation_result(
                &rx,
                &ry,
                ResidualNull {
                    salt: 0x69DC_u64 ^ query_stream_salt(&[0], &[1], &z_flat),
                    n_perm: nonparametric_permutation_count(significance),
                    block_size: 1,
                },
                &ctx,
            );
            assert_eq!(got.statistic.to_bits(), want.statistic.to_bits(), "{label} statistic");
            assert_eq!(got.p_value.to_bits(), want.p_value.to_bits(), "{label} p-value");

            let low_rank =
                GpdcNystrom::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap().results[0];
            assert_ne!(low_rank.statistic.to_bits(), got.statistic.to_bits(), "{label}");
        }
        // The exact test's boundary is unchanged and names the low-rank construction.
        let (data, _, _) = nystrom_sample(GPDC_ROW_LIMIT + 1, 6_075_547, 0.0);
        let cols: Vec<&[f64]> = data.iter().map(Vec::as_slice).collect();
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &z_flat,
            significance,
            confidence: ConfidenceMethod::None,
        };
        let refused = Gpdc::new()
            .test_batch_adhoc(&req, &mut CiWorkspace::default(), &ExecutionContext::for_tests(12))
            .unwrap_err();
        assert!(refused.to_string().contains("gpdc_nystrom"), "{refused}");
    }

    #[test]
    fn knn_dependence_runs_beyond_the_former_ten_thousand_row_limit() {
        let n = 12_000usize;
        let mut rng = ExecutionContext::for_tests(31).rng.stream_for(StreamDomain::Test, 0x4B4E);
        let x: Vec<f64> = (0..n).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect();
        let y_dep: Vec<f64> =
            x.iter().map(|v| v + 0.3 * antecedent_kernels::standard_normal(&mut rng)).collect();
        let y_ind: Vec<f64> =
            (0..n).map(|_| antecedent_kernels::standard_normal(&mut rng)).collect();
        let cols: [&[f64]; 3] = [&x, &y_dep, &y_ind];
        let queries = [
            CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 },
            CiQuery { x: 0, y: 2, z_start: 0, z_len: 0 },
        ];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::BlockShuffle { replicates: 9, block_size: 1 },
            confidence: ConfidenceMethod::None,
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(32);
        let out = KnnDependence::new(5).test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        // Every null replicate is less dependent than the observed pair: the add-one floor.
        assert!((out.results[0].p_value - 0.1).abs() < 1e-12, "{}", out.results[0].p_value);
        assert!(out.results[0].statistic > out.results[1].statistic);
    }
}
