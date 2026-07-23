//! Oracle, kNN distance-dependence, symbolic CMI, and GPDC CI tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    clippy::cast_lossless,
    clippy::needless_range_loop,
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::doc_markdown,
    clippy::trivially_copy_pass_by_ref
)]

use std::collections::HashMap;
use std::sync::Arc;

use causal_core::{ExecutionContext, KernelPolicy};
use causal_kernels::{shuffle, unbiased_index};

use super::types::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependenceTest,
    KnnDependenceWorkspace, PreparedCiTest, nonparametric_permutation_count,
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
        let mut results = Vec::with_capacity(request.queries.len());
        for (qi, q) in request.queries.iter().enumerate() {
            let z = &request.z_flat[q.z_start..q.z_start + q.z_len];
            let dim = 2 + z.len();
            ensure_knn_index(request.columns, q.x, q.y, z, n, dim, &mut workspace.knn)?;
            let builds_before = workspace.knn.index_builds;
            let stat = knn_stat_from_index(&mut workspace.knn, self.k)?;
            // Null: permute Y within coarse Z strata so the Y–Z link is preserved under
            // H0 (a full unconditional shuffle would inflate type-I error when Y depends
            // on Z). See `coarse_z_strata`.
            let strata = coarse_z_strata(request.columns, z, n);
            let mut y_perm = request.columns[q.y].to_vec();
            let mut rng = ctx.rng.stream(0xC11_u64.wrapping_add(qi as u64));
            let mut null_ge = 0u32;
            for _ in 0..n_perm {
                for rows in &strata {
                    for i in (1..rows.len()).rev() {
                        let j = unbiased_index(&mut rng, i + 1);
                        y_perm.swap(rows[i], rows[j]);
                    }
                }
                let mut cols: Vec<&[f64]> = request.columns.to_vec();
                cols[q.y] = &y_perm;
                let null = knn_mi_proxy_ephemeral(&cols, q.x, q.y, z, self.k)?;
                if null >= stat {
                    null_ge = null_ge.saturating_add(1);
                }
            }
            // Restore primary index after nulls.
            ensure_knn_index(request.columns, q.x, q.y, z, n, dim, &mut workspace.knn)?;
            debug_assert_eq!(workspace.knn.index_builds, builds_before);
            let p = (1.0 + f64::from(null_ge)) / (1.0 + n_perm as f64);
            // df is not defined for this distance statistic; leave 0 rather than claim n.
            results.push(CiResult { statistic: stat, p_value: p, df: 0.0, ci: None });
        }
        Ok(CiBatchResult { results })
    }
}

/// Fingerprint of the columns feeding a kNN index: query indexes plus pointer, length,
/// and sampled contents of each involved column. Content samples guard against pointer
/// reuse after frees between batches.
fn knn_input_fingerprint(columns: &[&[f64]], x: usize, y: usize, z: &[usize], n: usize) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325_u64;
    let mut mix = |v: u64| {
        h ^= v;
        h = h.wrapping_mul(0x0100_0000_01b3);
    };
    mix(x as u64);
    mix(y as u64);
    mix(z.len() as u64);
    for &zc in z {
        mix(zc as u64);
    }
    for &c in [x, y].iter().chain(z.iter()) {
        let col = columns[c];
        mix(col.as_ptr() as u64);
        mix(col.len() as u64);
        if n > 0 {
            mix(col[0].to_bits());
            mix(col[n / 2].to_bits());
            mix(col[n - 1].to_bits());
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
    idx.kth_distances(&knn.features, n, k, &mut knn.distances)?;
    let mean = knn.distances.iter().sum::<f64>() / n as f64;
    Ok(-mean)
}

fn knn_mi_proxy_ephemeral(
    columns: &[&[f64]],
    x: usize,
    y: usize,
    z: &[usize],
    k: usize,
) -> Result<f64, StatsError> {
    let n = columns[x].len();
    let dim = 2 + z.len();
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
    let mut dists = vec![0.0; n];
    idx.kth_distances(&feats, n, k, &mut dists)?;
    Ok(-dists.iter().sum::<f64>() / n as f64)
}

/// Coarse Z strata for permutation nulls: each Z column is binned into terciles
/// (rank-based) and rows are grouped by the joint bin key. Permuting Y within these
/// strata approximately preserves the Y–Z dependence under H0 — a documented coarse
/// approximation to a full within-neighborhood conditional permutation scheme.
/// Deterministic order (sorted keys) keeps the seeded RNG stream reproducible.
fn coarse_z_strata(columns: &[&[f64]], z: &[usize], n: usize) -> Vec<Vec<usize>> {
    const BINS: usize = 3;
    if z.is_empty() {
        return vec![(0..n).collect()];
    }
    let mut keys = vec![0u64; n];
    for &zc in z {
        let col = columns[zc];
        let mut idx: Vec<usize> = (0..n).collect();
        idx.sort_by(|&a, &b| col[a].partial_cmp(&col[b]).unwrap_or(std::cmp::Ordering::Equal));
        for (rank, &r) in idx.iter().enumerate() {
            let bin = rank * BINS / n;
            keys[r] = keys[r].wrapping_mul(31).wrapping_add(bin as u64 + 1);
        }
    }
    let mut map: HashMap<u64, Vec<usize>> = HashMap::new();
    for r in 0..n {
        map.entry(keys[r]).or_default().push(r);
    }
    let mut sorted_keys: Vec<u64> = map.keys().copied().collect();
    sorted_keys.sort_unstable();
    sorted_keys.into_iter().filter_map(|k| map.remove(&k)).collect()
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
        let mut owned: Vec<Vec<f64>> = request.columns.iter().map(|c| c.to_vec()).collect();
        for col in &mut owned {
            if looks_discrete(col) {
                let ranked = col.clone();
                super::parcorr_variants::rank_column(&ranked, col);
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
            let mut rng = ctx.rng.stream(0x51C_u64.wrapping_add(qi as u64));
            let n_perm = nonparametric_permutation_count(request.significance);
            let mut null_ge = 0u32;
            for _ in 0..n_perm {
                for rows in &strata {
                    for i in (1..rows.len()).rev() {
                        let j = unbiased_index(&mut rng, i + 1);
                        y_perm.swap(rows[i], rows[j]);
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
        let request = &prepared.bind_request(request);
        let n = request.columns.first().map_or(0, |c| c.len());
        if n == 0 {
            return Err(StatsError::Shape { message: "no columns" });
        }
        let n_perm = nonparametric_permutation_count(request.significance);
        let policy = &ctx.kernel_policy;
        let mut results = Vec::with_capacity(request.queries.len());
        for (qi, q) in request.queries.iter().enumerate() {
            let z = &request.z_flat[q.z_start..q.z_start + q.z_len];
            let rx = gp_residual(request.columns[q.x], request.columns, z, self);
            let ry = gp_residual(request.columns[q.y], request.columns, z, self);
            let dcor = distance_correlation(policy, &rx, &ry);
            // Permutation null: shuffle the Y residuals (Z influence already removed)
            // and recompute dCor; add-one p-value keeps it in (0, 1].
            let mut ry_perm = ry.clone();
            let mut rng = ctx.rng.stream(0x69DC_u64.wrapping_add(qi as u64));
            let mut null_ge = 0u32;
            for _ in 0..n_perm {
                shuffle(&mut rng, &mut ry_perm);
                if distance_correlation(policy, &rx, &ry_perm) >= dcor {
                    null_ge = null_ge.saturating_add(1);
                }
            }
            let p = (1.0 + f64::from(null_ge)) / (1.0 + n_perm as f64);
            results.push(CiResult { statistic: dcor, p_value: p, df: 0.0, ci: None });
        }
        Ok(CiBatchResult { results })
    }
}

fn gp_residual(y: &[f64], columns: &[&[f64]], z: &[usize], gp: &Gpdc) -> Vec<f64> {
    let n = y.len();
    if z.is_empty() {
        let mean = y.iter().sum::<f64>() / n as f64;
        return y.iter().map(|v| v - mean).collect();
    }
    // Build Gram on Z (sum of RBF over Z dims) and solve (K+λI)α = y.
    let mut k = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..=i {
            let mut d2 = 0.0;
            for &zc in z {
                let d = columns[zc][i] - columns[zc][j];
                d2 += d * d;
            }
            let kij = (-0.5 * d2 / (gp.length_scale * gp.length_scale)).exp();
            k[i * n + j] = kij;
            k[j * n + i] = kij;
        }
        k[i * n + i] += gp.ridge;
    }
    // Simple Gauss-Seidel / Jacobi for α
    let mut alpha = vec![0.0; n];
    for _ in 0..40 {
        for i in 0..n {
            let mut s = y[i];
            for j in 0..n {
                if i != j {
                    s -= k[i * n + j] * alpha[j];
                }
            }
            alpha[i] = s / k[i * n + i];
        }
    }
    let mut pred = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            pred[i] += k[i * n + j] * alpha[j];
        }
        // remove ridge contribution approx by using original K without ridge on predict —
        // residual = y - K_unreg α; use y - pred + ridge*α as correction
        pred[i] -= gp.ridge * alpha[i];
    }
    (0..n).map(|i| y[i] - pred[i]).collect()
}

fn distance_correlation(policy: &KernelPolicy, x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mut ax = vec![0.0; n * n];
    let mut ay = vec![0.0; n * n];
    causal_kernels::pairwise_l1_fill(policy, x, &mut ax);
    causal_kernels::pairwise_l1_fill(policy, y, &mut ay);
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
    // Székely dCor: sqrt(dCov² / sqrt(dVarX · dVarY)).
    (dcov2.max(0.0) / (dvarx * dvary).sqrt()).sqrt()
}

fn double_center_inplace(a: &mut [f64], n: usize) {
    let mut row = vec![0.0; n];
    let mut col = vec![0.0; n];
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
}
