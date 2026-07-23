//! Robust, weighted, and multivariate partial-correlation CI tests .
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::doc_markdown,
    clippy::too_many_arguments,
    clippy::similar_names,
    clippy::many_single_char_names,
    clippy::trivially_copy_pass_by_ref
)]

use antecedent_core::{ExecutionContext, KernelPolicy};
use antecedent_kernels::{sanitize_weight, weighted_mean};

use super::parcorr::PartialCorrelation;
use super::types::{
    CiBatchRequest, CiBatchResult, CiQuery, CiResult, CiWorkspace, ConditionalIndependenceTest,
    ConfidenceMethod, PreparedCiTest, SignificanceMethod,
};
use crate::error::StatsError;
use crate::gram::invert_square;

pub(crate) fn rank_column(col: &[f64], out: &mut [f64]) {
    let n = col.len();
    let mut idx: Vec<usize> = (0..n).collect();
    idx.sort_by(|&i, &j| col[i].partial_cmp(&col[j]).unwrap_or(std::cmp::Ordering::Equal));
    let mut i = 0usize;
    while i < n {
        let mut j = i;
        while j + 1 < n && (col[idx[j + 1]] - col[idx[i]]).abs() < 1e-15 {
            j += 1;
        }
        let first = (i + 1) as f64;
        let last = (j + 1) as f64;
        let avg_rank = (first + last) / 2.0;
        for k in i..=j {
            out[idx[k]] = avg_rank;
        }
        i = j + 1;
    }
}

/// Robust (nonparanormal / rank-based) partial correlation.
#[derive(Clone, Debug, Default)]
pub struct RobustPartialCorrelation {
    inner: PartialCorrelation,
}

impl RobustPartialCorrelation {
    /// Construct.
    #[must_use]
    pub fn new() -> Self {
        Self { inner: PartialCorrelation::new() }
    }
}

impl ConditionalIndependenceTest for RobustPartialCorrelation {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        let request = &prepared.bind_request(request);
        let n = request.nrows()?;
        let mut ranked: Vec<Vec<f64>> = request.columns.iter().map(|_| vec![0.0; n]).collect();
        for (c, col) in request.columns.iter().enumerate() {
            rank_column(col, &mut ranked[c]);
        }
        let refs: Vec<&[f64]> = ranked.iter().map(Vec::as_slice).collect();
        let req = CiBatchRequest {
            columns: &refs,
            queries: request.queries,
            z_flat: request.z_flat,
            significance: request.significance,
            confidence: request.confidence,
        };
        self.inner.test_batch(prepared, &req, workspace, ctx)
    }
}

/// Weighted partial correlation: weighted least-squares residualization on `[1 | Z]`
/// followed by weighted Pearson correlation of the residuals.
///
/// Row scaling by `sqrt(w)` alone is *not* used because the downstream kernel regresses
/// on an unscaled intercept and centers with unweighted means, which is invalid for
/// heterogeneous weights.
#[derive(Clone, Debug)]
pub struct WeightedPartialCorrelation {
    /// Per-row weights. May be longer than a batch's row count: lagged discovery frames
    /// drop leading rows, so the *last* `nrows` weights are used (frame row `i` observes
    /// series time `offset + i`, which suffix alignment matches).
    pub weights: Vec<f64>,
}

impl WeightedPartialCorrelation {
    /// Construct with positive weights.
    #[must_use]
    pub fn new(weights: Vec<f64>) -> Self {
        Self { weights }
    }
}

impl ConditionalIndependenceTest for WeightedPartialCorrelation {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        _workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        let request = &prepared.bind_request(request);
        let n = request.nrows()?;
        if self.weights.len() < n {
            return Err(StatsError::Shape { message: "weights length < nrows" });
        }
        let weights = &self.weights[self.weights.len() - n..];
        let policy = &ctx.kernel_policy;
        let mut results = Vec::with_capacity(request.queries.len());
        for (qi, q) in request.queries.iter().enumerate() {
            let z = &request.z_flat[q.z_start..q.z_start + q.z_len];
            let r = weighted_parcorr_stat(request.columns, q.x, q.y, z, weights, n, policy)?;
            let df = (n as f64) - 2.0 - (q.z_len as f64);
            let result = match request.significance {
                SignificanceMethod::Analytic => {
                    if df <= 0.0 {
                        return Err(StatsError::Shape { message: "non-positive residual df" });
                    }
                    let p = crate::ci::analytic::analytic_parcorr_pvalue(r, df);
                    let ci = match request.confidence {
                        ConfidenceMethod::None => None,
                        ConfidenceMethod::Analytic { level } => {
                            Some(crate::ci::analytic::analytic_parcorr_ci(r, df, level))
                        }
                    };
                    CiResult { statistic: r, p_value: p, df, ci }
                }
                SignificanceMethod::BlockShuffle { replicates, block_size } => {
                    if block_size == 0 || replicates == 0 {
                        return Err(StatsError::Shape {
                            message: "block shuffle needs positive block_size and replicates",
                        });
                    }
                    let p = weighted_block_shuffle_pvalue(
                        request.columns,
                        *q,
                        z,
                        weights,
                        r,
                        replicates,
                        block_size,
                        ctx,
                        qi as u64,
                        policy,
                    )?;
                    CiResult { statistic: r, p_value: p, df, ci: None }
                }
            };
            results.push(result);
        }
        Ok(CiBatchResult { results })
    }
}

/// Weighted partial correlation of `columns[x]` and `columns[y]` given `z`.
fn weighted_parcorr_stat(
    columns: &[&[f64]],
    x: usize,
    y: usize,
    z: &[usize],
    weights: &[f64],
    n: usize,
    policy: &KernelPolicy,
) -> Result<f64, StatsError> {
    let ex = weighted_residuals(columns[x], columns, z, weights, n)?;
    let ey = weighted_residuals(columns[y], columns, z, weights, n)?;
    weighted_pearson(policy, &ex, &ey, weights)
        .ok_or(StatsError::Shape { message: "degenerate weighted correlation" })
}

/// Residuals of `target` after weighted least squares on `[1 | Z]`.
fn weighted_residuals(
    target: &[f64],
    columns: &[&[f64]],
    z: &[usize],
    weights: &[f64],
    n: usize,
) -> Result<Vec<f64>, StatsError> {
    let q = 1 + z.len();
    let mut g = vec![0.0; q * q];
    let mut rhs = vec![0.0; q];
    let mut d = vec![0.0; q];
    for r in 0..n {
        let w = sanitize_weight(weights[r]);
        d[0] = 1.0;
        for (j, &zc) in z.iter().enumerate() {
            d[j + 1] = columns[zc][r];
        }
        for i in 0..q {
            rhs[i] += w * d[i] * target[r];
            for j in 0..q {
                g[i * q + j] += w * d[i] * d[j];
            }
        }
    }
    let g_inv = invert_square(&g, q)
        .ok_or(StatsError::Shape { message: "singular Z design in multivariate ParCorr" })?;
    let mut beta = vec![0.0; q];
    for i in 0..q {
        for j in 0..q {
            beta[i] += g_inv[i * q + j] * rhs[j];
        }
    }
    let mut out = vec![0.0; n];
    for r in 0..n {
        let mut pred = beta[0];
        for (j, &zc) in z.iter().enumerate() {
            pred += beta[j + 1] * columns[zc][r];
        }
        out[r] = target[r] - pred;
    }
    Ok(out)
}

/// Weighted Pearson correlation with weighted centering.
fn weighted_pearson(policy: &KernelPolicy, x: &[f64], y: &[f64], weights: &[f64]) -> Option<f64> {
    let n = x.len();
    let mx = weighted_mean(policy, x, weights)?;
    let my = weighted_mean(policy, y, weights)?;
    let mut cxx = 0.0;
    let mut cyy = 0.0;
    let mut cxy = 0.0;
    for i in 0..n {
        let w = sanitize_weight(weights[i]);
        let dx = x[i] - mx;
        let dy = y[i] - my;
        cxx += w * dx * dx;
        cyy += w * dy * dy;
        cxy += w * dx * dy;
    }
    let denom = (cxx * cyy).sqrt();
    if denom <= f64::EPSILON {
        return None;
    }
    Some((cxy / denom).clamp(-1.0, 1.0))
}

/// Block-shuffle null for the weighted statistic (X shuffled by blocks; weights stay
/// aligned with rows).
#[allow(clippy::too_many_arguments)]
fn weighted_block_shuffle_pvalue(
    columns: &[&[f64]],
    q: CiQuery,
    z: &[usize],
    weights: &[f64],
    observed: f64,
    replicates: u32,
    block_size: usize,
    ctx: &ExecutionContext,
    stream_salt: u64,
    policy: &KernelPolicy,
) -> Result<f64, StatsError> {
    let n = columns[q.x].len();
    let x = columns[q.x];
    let n_blocks = n.div_ceil(block_size);
    let mut block_perm: Vec<usize> = (0..n_blocks).collect();
    let mut shuffled = vec![0.0; n];
    let mut rng = ctx.rng.stream(0x77C1_u64.wrapping_add(stream_salt));
    let mut extreme = 0u32;
    let abs_obs = observed.abs();
    for _ in 0..replicates {
        for i in (1..n_blocks).rev() {
            let j = (rng.next_u64() as usize) % (i + 1);
            block_perm.swap(i, j);
        }
        let mut dst = 0usize;
        for &b in &block_perm {
            let start = b * block_size;
            let end = (start + block_size).min(n);
            let len = end - start;
            shuffled[dst..dst + len].copy_from_slice(&x[start..end]);
            dst += len;
        }
        let mut cols: Vec<&[f64]> = columns.to_vec();
        cols[q.x] = &shuffled;
        let r = weighted_parcorr_stat(&cols, q.x, q.y, z, weights, n, policy)?;
        if r.abs() >= abs_obs {
            extreme += 1;
        }
    }
    Ok((f64::from(extreme) + 1.0) / (f64::from(replicates) + 1.0))
}

/// Multivariate partial correlation via block residualization and first canonical
/// correlation.
///
/// Each column of X and Y is residualized against Z by OLS; the leading canonical
/// correlation between residual blocks is the dependence statistic. When both blocks
/// are scalar this reduces to ordinary partial correlation.
#[derive(Clone, Debug, Default)]
pub struct MultivariatePartialCorrelation {
    inner: PartialCorrelation,
}

impl MultivariatePartialCorrelation {
    /// Construct.
    #[must_use]
    pub fn new() -> Self {
        Self { inner: PartialCorrelation::new() }
    }

    /// Test independence of two multivariate blocks given Z columns.
    ///
    /// `x_cols` / `y_cols` are indexes into `columns`; Z via `z_flat`.
    ///
    /// # Errors
    ///
    /// Shape failures.
    pub fn test_blocks(
        &self,
        columns: &[&[f64]],
        x_cols: &[usize],
        y_cols: &[usize],
        z_flat: &[usize],
        significance: SignificanceMethod,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiResult, StatsError> {
        if x_cols.is_empty() || y_cols.is_empty() {
            return Err(StatsError::Shape { message: "empty X or Y block" });
        }
        // Scalar path: exact partial correlation.
        if x_cols.len() == 1 && y_cols.len() == 1 {
            let n = columns[0].len();
            let mut owned: Vec<Vec<f64>> = Vec::with_capacity(2 + z_flat.len());
            owned.push(columns[x_cols[0]].to_vec());
            owned.push(columns[y_cols[0]].to_vec());
            for &z in z_flat {
                if columns[z].len() != n {
                    return Err(StatsError::Shape { message: "column length mismatch" });
                }
                owned.push(columns[z].to_vec());
            }
            let refs: Vec<&[f64]> = owned.iter().map(Vec::as_slice).collect();
            let z_idx: Vec<usize> = (2..2 + z_flat.len()).collect();
            return self.inner.test_one(&refs, &z_idx, significance, workspace, ctx);
        }

        let n = columns[0].len();
        for &c in x_cols.iter().chain(y_cols.iter()).chain(z_flat.iter()) {
            if columns[c].len() != n {
                return Err(StatsError::Shape { message: "column length mismatch" });
            }
        }

        let rx = residualize_block(columns, x_cols, z_flat, n)?;
        let ry = residualize_block(columns, y_cols, z_flat, n)?;
        let px = x_cols.len();
        let py = y_cols.len();
        let rho = first_canonical_correlation(&rx, &ry, n, px, py)?;

        let df = (n as f64) - 2.0 - (z_flat.len() as f64) - ((px + py) as f64 - 2.0).max(0.0);
        match significance {
            SignificanceMethod::Analytic => {
                if df <= 0.0 {
                    return Err(StatsError::Shape { message: "non-positive residual df" });
                }
                // Bartlett approximation on first canonical correlation (Fisher-z of ρ).
                let p = crate::ci::analytic::analytic_parcorr_pvalue(rho, df);
                Ok(CiResult {
                    statistic: rho,
                    p_value: p,
                    df,
                    ci: Some(crate::ci::analytic::analytic_parcorr_ci(rho, df, 0.95)),
                })
            }
            SignificanceMethod::BlockShuffle { .. } => {
                // Delegate shuffle significance through scalar ParCorr on projected residuals
                // of the leading CCA directions when requested.
                let (ux, uy) = leading_cca_scores(&rx, &ry, n, px, py)?;
                let mut owned = vec![ux, uy];
                for &z in z_flat {
                    owned.push(columns[z].to_vec());
                }
                // Residuals already orthogonal to Z; test without Z for shuffle path.
                let refs: Vec<&[f64]> = owned.iter().map(Vec::as_slice).collect();
                self.inner.test_one(&refs, &[], significance, workspace, ctx)
            }
        }
    }
}

impl ConditionalIndependenceTest for MultivariatePartialCorrelation {
    fn test_batch(
        &self,
        prepared: &PreparedCiTest,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        prepared.ensure_compatible(request)?;
        let request = &prepared.bind_request(request);
        // Scalar queries: exact ParCorr. Block queries go through test_blocks via pairwise wrapper.
        self.inner.test_batch(prepared, request, workspace, ctx)
    }
}

/// Residualize each column in `idxs` against the Z design (intercept + Z columns).
fn residualize_block(
    columns: &[&[f64]],
    idxs: &[usize],
    z_flat: &[usize],
    n: usize,
) -> Result<Vec<f64>, StatsError> {
    let p = idxs.len();
    let q = z_flat.len() + 1; // intercept
    let mut design = vec![0.0; n * q];
    for r in 0..n {
        design[r] = 1.0; // col-major: column 0
    }
    for (j, &z) in z_flat.iter().enumerate() {
        for r in 0..n {
            design[(j + 1) * n + r] = columns[z][r];
        }
    }
    // Gram matrix G = D'D (q × q) and its inverse via Gauss-Jordan.
    let mut g = vec![0.0; q * q];
    for i in 0..q {
        for j in 0..q {
            let mut s = 0.0;
            for r in 0..n {
                s += design[i * n + r] * design[j * n + r];
            }
            g[i * q + j] = s;
        }
    }
    let g_inv = invert_square(&g, q)
        .ok_or(StatsError::Shape { message: "singular Z design in multivariate ParCorr" })?;
    let mut out = vec![0.0; n * p];
    for (k, &c) in idxs.iter().enumerate() {
        // beta = G^{-1} D' y
        let mut dty = vec![0.0; q];
        for i in 0..q {
            let mut s = 0.0;
            for r in 0..n {
                s += design[i * n + r] * columns[c][r];
            }
            dty[i] = s;
        }
        let mut beta = vec![0.0; q];
        for i in 0..q {
            for j in 0..q {
                beta[i] += g_inv[i * q + j] * dty[j];
            }
        }
        for r in 0..n {
            let mut pred = 0.0;
            for i in 0..q {
                pred += design[i * n + r] * beta[i];
            }
            out[k * n + r] = columns[c][r] - pred;
        }
    }
    Ok(out)
}

/// First canonical correlation between residual blocks (col-major `n×px`, `n×py`).
fn first_canonical_correlation(
    rx: &[f64],
    ry: &[f64],
    n: usize,
    px: usize,
    py: usize,
) -> Result<f64, StatsError> {
    let (rho, _, _) = cca_leading(rx, ry, n, px, py)?;
    Ok(rho)
}

fn leading_cca_scores(
    rx: &[f64],
    ry: &[f64],
    n: usize,
    px: usize,
    py: usize,
) -> Result<(Vec<f64>, Vec<f64>), StatsError> {
    let (_, ax, ay) = cca_leading(rx, ry, n, px, py)?;
    let mut ux = vec![0.0; n];
    let mut uy = vec![0.0; n];
    for r in 0..n {
        for j in 0..px {
            ux[r] += rx[j * n + r] * ax[j];
        }
        for j in 0..py {
            uy[r] += ry[j * n + r] * ay[j];
        }
    }
    Ok((ux, uy))
}

/// Return (ρ, a_x, a_y) for the leading canonical pair via power iteration on
/// Cxx^{-1} Cxy Cyy^{-1} Cyx.
#[allow(clippy::too_many_lines)]
fn cca_leading(
    rx: &[f64],
    ry: &[f64],
    n: usize,
    px: usize,
    py: usize,
) -> Result<(f64, Vec<f64>, Vec<f64>), StatsError> {
    let denom = (n.saturating_sub(1)).max(1) as f64;
    let mut cxx = vec![0.0; px * px];
    let mut cyy = vec![0.0; py * py];
    let mut cxy = vec![0.0; px * py];
    for i in 0..px {
        for j in 0..px {
            let mut s = 0.0;
            for r in 0..n {
                s += rx[i * n + r] * rx[j * n + r];
            }
            cxx[i * px + j] = s / denom;
        }
    }
    for i in 0..py {
        for j in 0..py {
            let mut s = 0.0;
            for r in 0..n {
                s += ry[i * n + r] * ry[j * n + r];
            }
            cyy[i * py + j] = s / denom;
        }
    }
    for i in 0..px {
        for j in 0..py {
            let mut s = 0.0;
            for r in 0..n {
                s += rx[i * n + r] * ry[j * n + r];
            }
            cxy[i * py + j] = s / denom;
        }
    }
    // Regularize diagonals slightly for numerical stability.
    for i in 0..px {
        cxx[i * px + i] += 1e-8;
    }
    for i in 0..py {
        cyy[i * py + i] += 1e-8;
    }
    let cxx_inv = invert_square(&cxx, px)
        .ok_or(StatsError::Shape { message: "singular Z design in multivariate ParCorr" })?;
    let cyy_inv = invert_square(&cyy, py)
        .ok_or(StatsError::Shape { message: "singular Z design in multivariate ParCorr" })?;

    // M = Cxx^{-1} Cxy Cyy^{-1} Cyx (px × px)
    // temp = Cxy Cyy^{-1} (px × py)
    let mut temp = vec![0.0; px * py];
    for i in 0..px {
        for j in 0..py {
            let mut s = 0.0;
            for k in 0..py {
                s += cxy[i * py + k] * cyy_inv[k * py + j];
            }
            temp[i * py + j] = s;
        }
    }
    // Cyx = Cxy'
    // temp2 = temp * Cyx = temp * Cxy' (px × px)
    let mut temp2 = vec![0.0; px * px];
    for i in 0..px {
        for j in 0..px {
            let mut s = 0.0;
            for k in 0..py {
                s += temp[i * py + k] * cxy[j * py + k];
            }
            temp2[i * px + j] = s;
        }
    }
    // M = Cxx^{-1} temp2
    let mut m = vec![0.0; px * px];
    for i in 0..px {
        for j in 0..px {
            let mut s = 0.0;
            for k in 0..px {
                s += cxx_inv[i * px + k] * temp2[k * px + j];
            }
            m[i * px + j] = s;
        }
    }

    // Power iteration for leading eigenvector of M.
    let mut a = vec![1.0 / (px as f64).sqrt(); px];
    let mut lambda = 0.0;
    for _ in 0..64 {
        let mut w = vec![0.0; px];
        for i in 0..px {
            for j in 0..px {
                w[i] += m[i * px + j] * a[j];
            }
        }
        lambda = w.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-15);
        for i in 0..px {
            a[i] = w[i] / lambda;
        }
    }
    // ρ = sqrt(λ) where λ is the eigenvalue of the CCA matrix (canonical correlation²).
    let rho = lambda.sqrt().clamp(0.0, 1.0 - 1e-12);

    // b ∝ Cyy^{-1} Cyx a
    let mut cyx_a = vec![0.0; py];
    for j in 0..py {
        let mut s = 0.0;
        for i in 0..px {
            s += cxy[i * py + j] * a[i];
        }
        cyx_a[j] = s;
    }
    let mut b = vec![0.0; py];
    for i in 0..py {
        for j in 0..py {
            b[i] += cyy_inv[i * py + j] * cyx_a[j];
        }
    }
    let bn = b.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-15);
    for v in &mut b {
        *v /= bn;
    }
    Ok((rho, a, b))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn robust_detects_monotonic_dependence() {
        let n = 200usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = x.iter().map(|&v| v.powi(3)).collect();
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
        let out = RobustPartialCorrelation::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        assert!(out.results[0].p_value < 1e-3);
    }

    #[test]
    fn weighted_unit_matches_parcorr() {
        let n = 100usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| 2.0 * i as f64).collect();
        let w = vec![1.0; n];
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
        let a = PartialCorrelation::new().test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        let b = WeightedPartialCorrelation::new(w).test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        assert!((a.results[0].statistic - b.results[0].statistic).abs() < 1e-9);
    }

    #[test]
    fn weighted_independent_nonzero_means_near_zero() {
        // Independent columns with large common offsets and heterogeneous weights must
        // not produce spurious correlation (regression test for the sqrt-w scaling bug).
        let n = 200usize;
        let x: Vec<f64> = (0..n).map(|i| 10.0 + ((i * 37 + 11) % 17) as f64 * 0.01).collect();
        let y: Vec<f64> = (0..n).map(|i| 10.0 + ((i * 53 + 5) % 19) as f64 * 0.01).collect();
        let w: Vec<f64> = (0..n).map(|i| 0.1 + ((i * 29 + 3) % 23) as f64 * 0.5).collect();
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
        let ctx = ExecutionContext::for_tests(7);
        let out = WeightedPartialCorrelation::new(w).test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        assert!(
            out.results[0].statistic.abs() < 0.2,
            "spurious weighted correlation: {}",
            out.results[0].statistic
        );
        assert!(out.results[0].p_value > 0.01, "p={}", out.results[0].p_value);
    }

    #[test]
    fn weighted_nonfinite_weights_match_zeroed_weights() {
        let n = 80usize;
        let x: Vec<f64> = (0..n).map(|i| i as f64).collect();
        let y: Vec<f64> = (0..n).map(|i| 2.0 * i as f64 + 1.0).collect();
        let mut w_dirty = vec![1.0; n];
        let mut w_clean = vec![1.0; n];
        for i in (0..n).step_by(7) {
            w_dirty[i] = f64::NAN;
            w_clean[i] = 0.0;
        }
        for i in (3..n).step_by(11) {
            w_dirty[i] = f64::NEG_INFINITY;
            w_clean[i] = 0.0;
        }
        for i in (5..n).step_by(13) {
            w_dirty[i] = -2.0;
            w_clean[i] = 0.0;
        }
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
        let ctx = ExecutionContext::for_tests(8);
        let dirty =
            WeightedPartialCorrelation::new(w_dirty).test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        let clean =
            WeightedPartialCorrelation::new(w_clean).test_batch_adhoc(&req, &mut ws, &ctx).unwrap();
        assert!(
            (dirty.results[0].statistic - clean.results[0].statistic).abs() < 1e-12,
            "dirty={} clean={}",
            dirty.results[0].statistic,
            clean.results[0].statistic
        );
        assert!(dirty.results[0].statistic.is_finite());
    }

    #[test]
    fn multivariate_scalar_matches_parcorr() {
        let n = 150usize;
        let x: Vec<f64> = (0..n).map(|i| (i as f64) * 0.01).collect();
        let y: Vec<f64> = x.iter().map(|&v| 0.8 * v + 0.1).collect();
        let z: Vec<f64> = (0..n).map(|i| (i as f64).sin()).collect();
        let cols: [&[f64]; 3] = [&x, &y, &z];
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(3);
        let a = PartialCorrelation::new()
            .test_one(&cols, &[2], SignificanceMethod::Analytic, &mut ws, &ctx)
            .unwrap();
        let b = MultivariatePartialCorrelation::new()
            .test_blocks(&cols, &[0], &[1], &[2], SignificanceMethod::Analytic, &mut ws, &ctx)
            .unwrap();
        assert!((a.statistic - b.statistic).abs() < 1e-8);
    }

    #[test]
    fn multivariate_block_detects_shared_latent() {
        let n = 300usize;
        let mut latent = vec![0.0; n];
        let mut x1 = vec![0.0; n];
        let mut x2 = vec![0.0; n];
        let mut y1 = vec![0.0; n];
        let mut y2 = vec![0.0; n];
        for i in 0..n {
            let t = i as f64 * 0.05;
            latent[i] = t.sin();
            x1[i] = latent[i] + 0.05 * (i as f64).cos();
            x2[i] = 0.7 * latent[i] + 0.05 * (i as f64).sin();
            y1[i] = 0.9 * latent[i] + 0.05 * ((i + 3) as f64).cos();
            y2[i] = 0.6 * latent[i] + 0.05 * ((i + 7) as f64).sin();
        }
        let cols: [&[f64]; 4] = [&x1, &x2, &y1, &y2];
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(4);
        let out = MultivariatePartialCorrelation::new()
            .test_blocks(&cols, &[0, 1], &[2, 3], &[], SignificanceMethod::Analytic, &mut ws, &ctx)
            .unwrap();
        assert!(out.p_value < 1e-3, "p={}, r={}", out.p_value, out.statistic);
        assert!(out.statistic.abs() > 0.5);
    }
}
