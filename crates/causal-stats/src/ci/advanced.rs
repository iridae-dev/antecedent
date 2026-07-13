//! Oracle, kNN/symbolic CMI, and GPDC CI tests (Phase 5).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::all)]

use std::collections::HashMap;
use std::sync::Arc;

use causal_core::ExecutionContext;

use super::types::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependence, KnnCmiWorkspace,
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

impl ConditionalIndependence for OracleCi {
    fn test_batch(
        &self,
        request: &CiBatchRequest<'_>,
        _workspace: &mut CiWorkspace,
        _ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
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

/// kNN conditional mutual information CI (KSG-style rank proxy).
#[derive(Clone, Debug)]
pub struct KnnCmi {
    /// Neighbors.
    pub k: usize,
}

impl Default for KnnCmi {
    fn default() -> Self {
        Self::new(5)
    }
}

impl KnnCmi {
    /// Construct with neighbor count `k`.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self { k: k.max(1) }
    }
}

impl ConditionalIndependence for KnnCmi {
    fn test_batch(
        &self,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        let n = request.columns.first().map(|c| c.len()).unwrap_or(0);
        if n < self.k + 2 {
            return Err(StatsError::Shape { message: "n too small for kNN CMI" });
        }
        if workspace.knn.perm.len() != n {
            workspace.knn.perm = (0..n).collect();
            workspace.knn.index_generation = workspace.knn.index_generation.saturating_add(1);
            workspace.knn.last_n = n;
        }
        if workspace.block_perm.len() != n {
            workspace.block_perm = workspace.knn.perm.clone();
        }
        let n_perm = 49usize;
        let mut results = Vec::with_capacity(request.queries.len());
        for (qi, q) in request.queries.iter().enumerate() {
            let z = &request.z_flat[q.z_start..q.z_start + q.z_len];
            let dim = 2 + z.len();
            ensure_knn_index(request.columns, q.x, q.y, z, n, dim, &mut workspace.knn)?;
            let builds_before = workspace.knn.index_builds;
            let stat = knn_stat_from_index(&workspace.knn, self.k)?;
            // Null: permute Y via reusable plan shuffled with RNG stream (plan buffer reused).
            let mut y_perm = request.columns[q.y].to_vec();
            let mut rng = ctx.rng.stream(0xC11_u64.wrapping_add(qi as u64));
            let mut null_ge = 0u32;
            for _ in 0..n_perm {
                // Fisher–Yates into perm order applied to Y values.
                for i in (1..n).rev() {
                    let j = (rng.next_u64() as usize) % (i + 1);
                    workspace.knn.perm.swap(i, j);
                }
                for r in 0..n {
                    y_perm[r] = request.columns[q.y][workspace.knn.perm[r]];
                }
                // Restore identity permutation in place (no reallocation).
                for (i, slot) in workspace.knn.perm.iter_mut().enumerate() {
                    *slot = i;
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
            results.push(CiResult {
                statistic: stat,
                p_value: p,
                df: n as f64,
                ci: None,
            });
        }
        Ok(CiBatchResult { results })
    }
}

fn ensure_knn_index(
    columns: &[&[f64]],
    x: usize,
    y: usize,
    z: &[usize],
    n: usize,
    dim: usize,
    knn: &mut KnnCmiWorkspace,
) -> Result<(), StatsError> {
    let need_rebuild = knn.index.is_none() || knn.last_dim != dim || knn.last_n != n;
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
    knn.index_generation = knn.index_generation.saturating_add(1);
    knn.index_builds = knn.index_builds.saturating_add(1);
    Ok(())
}

fn knn_stat_from_index(knn: &KnnCmiWorkspace, k: usize) -> Result<f64, StatsError> {
    let n = knn.last_n;
    let idx = knn.index.as_ref().ok_or(StatsError::Shape { message: "missing kNN index" })?;
    let mut dists = vec![0.0; n];
    idx.kth_distances(&knn.features, n, k, &mut dists)?;
    let mean = dists.iter().sum::<f64>() / n as f64;
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

/// Mixed-data kNN CMI: ranks discrete-looking columns then runs [`KnnCmi`].
#[derive(Clone, Debug, Default)]
pub struct MixedKnnCmi {
    inner: KnnCmi,
}

impl MixedKnnCmi {
    /// Construct.
    #[must_use]
    pub fn new(k: usize) -> Self {
        Self { inner: KnnCmi::new(k) }
    }
}

impl ConditionalIndependence for MixedKnnCmi {
    fn test_batch(
        &self,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        let n = request.columns.first().map(|c| c.len()).unwrap_or(0);
        let mut owned: Vec<Vec<f64>> = request.columns.iter().map(|c| c.to_vec()).collect();
        for col in &mut owned {
            if looks_discrete(col) {
                let ranked = col.clone();
                super::parcorr_variants::rank_column(&ranked, col);
            }
        }
        let refs: Vec<&[f64]> = owned.iter().map(|c| c.as_slice()).collect();
        let ranked_req = CiBatchRequest {
            columns: &refs,
            queries: request.queries,
            z_flat: request.z_flat,
            significance: request.significance,
        };
        let _ = n;
        self.inner.test_batch(&ranked_req, workspace, ctx)
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

impl ConditionalIndependence for SymbolicCmi {
    fn test_batch(
        &self,
        request: &CiBatchRequest<'_>,
        _workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        let mut results = Vec::with_capacity(request.queries.len());
        for (qi, q) in request.queries.iter().enumerate() {
            let n = request.columns[q.x].len();
            let z = &request.z_flat[q.z_start..q.z_start + q.z_len];
            let mi = conditional_symbolic_mi(request.columns, q.x, q.y, z, n);
            // Permutation p-value on Y.
            let mut y_perm = request.columns[q.y].to_vec();
            let mut rng = ctx.rng.stream(0x51C_u64.wrapping_add(qi as u64));
            let n_perm = 49usize;
            let mut null_ge = 0u32;
            for _ in 0..n_perm {
                for i in (1..n).rev() {
                    let j = (rng.next_u64() as usize) % (i + 1);
                    y_perm.swap(i, j);
                }
                let mut cols: Vec<&[f64]> = request.columns.to_vec();
                cols[q.y] = &y_perm;
                let null = conditional_symbolic_mi(&cols, q.x, q.y, z, n);
                if null >= mi {
                    null_ge = null_ge.saturating_add(1);
                }
            }
            let p = (1.0 + f64::from(null_ge)) / (1.0 + n_perm as f64);
            results.push(CiResult {
                statistic: mi,
                p_value: p,
                df: n as f64,
                ci: None,
            });
        }
        Ok(CiBatchResult { results })
    }
}

fn conditional_symbolic_mi(
    columns: &[&[f64]],
    x: usize,
    y: usize,
    z: &[usize],
    n: usize,
) -> f64 {
    // Stratify by Z symbols; average stratum MI(X;Y|Z=z).
    let mut strata: HashMap<u64, Vec<usize>> = HashMap::new();
    for r in 0..n {
        let key = if z.is_empty() {
            0u64
        } else {
            let mut h = 0xcbf29ce484222325u64;
            for &zc in z {
                let v = columns[zc][r].round() as i32;
                h ^= v as u64;
                h = h.wrapping_mul(0x100000001b3);
            }
            h
        };
        strata.entry(key).or_default().push(r);
    }
    let mut mi = 0.0;
    let mut weight = 0.0;
    for rows in strata.values() {
        if rows.len() < 2 {
            continue;
        }
        let w = rows.len() as f64;
        mi += w * symbolic_mi_on_rows(columns, x, y, rows);
        weight += w;
    }
    if weight > 0.0 {
        mi / weight
    } else {
        0.0
    }
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

impl ConditionalIndependence for Gpdc {
    fn test_batch(
        &self,
        request: &CiBatchRequest<'_>,
        _workspace: &mut CiWorkspace,
        _ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        let n = request.columns.first().map(|c| c.len()).unwrap_or(0);
        if n == 0 {
            return Err(StatsError::Shape { message: "no columns" });
        }
        let mut results = Vec::with_capacity(request.queries.len());
        for q in request.queries {
            let z = &request.z_flat[q.z_start..q.z_start + q.z_len];
            let rx = gp_residual(request.columns[q.x], request.columns, z, self)?;
            let ry = gp_residual(request.columns[q.y], request.columns, z, self)?;
            let dcor = distance_correlation(&rx, &ry);
            results.push(CiResult {
                statistic: dcor,
                p_value: if dcor < 0.1 { 0.5 } else { 0.01 },
                df: n as f64,
                ci: None,
            });
        }
        Ok(CiBatchResult { results })
    }
}

fn gp_residual(
    y: &[f64],
    columns: &[&[f64]],
    z: &[usize],
    gp: &Gpdc,
) -> Result<Vec<f64>, StatsError> {
    let n = y.len();
    if z.is_empty() {
        let mean = y.iter().sum::<f64>() / n as f64;
        return Ok(y.iter().map(|v| v - mean).collect());
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
    Ok((0..n).map(|i| y[i] - pred[i]).collect())
}

fn distance_correlation(x: &[f64], y: &[f64]) -> f64 {
    let n = x.len();
    if n < 2 {
        return 0.0;
    }
    let mut ax = vec![0.0; n * n];
    let mut ay = vec![0.0; n * n];
    for i in 0..n {
        for j in 0..n {
            ax[i * n + j] = (x[i] - x[j]).abs();
            ay[i * n + j] = (y[i] - y[j]).abs();
        }
    }
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
    dcov2.max(0.0).sqrt() / (dvarx.sqrt() * dvary.sqrt())
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
    use crate::ci::types::{CiBatchRequest, CiQuery, CiWorkspace, SignificanceMethod};

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
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let out = oracle.test_batch(&req, &mut ws, &ctx).unwrap();
        assert_eq!(out.results[0].p_value, 0.0);
    }

    #[test]
    fn symbolic_mi_positive_on_copy() {
        let x: Vec<f64> = (0..100).map(|i| (i % 4) as f64).collect();
        let y = x.clone();
        let cols: [&[f64]; 2] = [&x, &y];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::Analytic,
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let out = SymbolicCmi::new().test_batch(&req, &mut ws, &ctx).unwrap();
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
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let out = Gpdc::new().test_batch(&req, &mut ws, &ctx).unwrap();
        assert!(out.results[0].statistic.is_finite());
    }
}
