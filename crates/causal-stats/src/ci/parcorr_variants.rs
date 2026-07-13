//! Robust, weighted, and multivariate partial-correlation CI tests (Phase 5).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::all)]
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::needless_range_loop,
    clippy::doc_markdown,
    clippy::too_many_arguments,
    clippy::similar_names
)]

use causal_core::ExecutionContext;

use super::parcorr::PartialCorrelation;
use super::types::{
    CiBatchRequest, CiBatchResult, CiResult, CiWorkspace, ConditionalIndependence,
    SignificanceMethod,
};

#[cfg(test)]
use super::types::CiQuery;
use crate::error::StatsError;

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

impl ConditionalIndependence for RobustPartialCorrelation {
    fn test_batch(
        &self,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
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
        };
        self.inner.test_batch(&req, workspace, ctx)
    }
}

/// Weighted partial correlation via row reweighting (sqrt-weight scaling).
#[derive(Clone, Debug)]
pub struct WeightedPartialCorrelation {
    inner: PartialCorrelation,
    /// Per-row weights (length = n).
    pub weights: Vec<f64>,
}

impl WeightedPartialCorrelation {
    /// Construct with positive weights.
    #[must_use]
    pub fn new(weights: Vec<f64>) -> Self {
        Self { inner: PartialCorrelation::new(), weights }
    }
}

impl ConditionalIndependence for WeightedPartialCorrelation {
    fn test_batch(
        &self,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        let n = request.nrows()?;
        if self.weights.len() != n {
            return Err(StatsError::Shape { message: "weights length != nrows" });
        }
        let mut scaled: Vec<Vec<f64>> = request.columns.iter().map(|_| vec![0.0; n]).collect();
        for r in 0..n {
            let w = self.weights[r].max(0.0).sqrt();
            for c in 0..request.columns.len() {
                scaled[c][r] = request.columns[c][r] * w;
            }
        }
        let refs: Vec<&[f64]> = scaled.iter().map(Vec::as_slice).collect();
        let req = CiBatchRequest {
            columns: &refs,
            queries: request.queries,
            z_flat: request.z_flat,
            significance: request.significance,
        };
        self.inner.test_batch(&req, workspace, ctx)
    }
}

/// Multivariate partial correlation: residualize vector blocks via first principal
/// direction of each block, then scalar ParCorr (Phase 5 practical approximation).
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
        let n = columns[0].len();
        let px = project_first_pc(columns, x_cols, n)?;
        let py = project_first_pc(columns, y_cols, n)?;
        let mut owned: Vec<Vec<f64>> = Vec::with_capacity(2 + z_flat.len());
        owned.push(px);
        owned.push(py);
        for &z in z_flat {
            owned.push(columns[z].to_vec());
        }
        let refs: Vec<&[f64]> = owned.iter().map(Vec::as_slice).collect();
        let z_idx: Vec<usize> = (2..2 + z_flat.len()).collect();
        self.inner.test_one(&refs, &z_idx, significance, workspace, ctx)
    }
}

impl ConditionalIndependence for MultivariatePartialCorrelation {
    fn test_batch(
        &self,
        request: &CiBatchRequest<'_>,
        workspace: &mut CiWorkspace,
        ctx: &ExecutionContext,
    ) -> Result<CiBatchResult, StatsError> {
        // Scalar path: same as ParCorr when X/Y are single columns.
        self.inner.test_batch(request, workspace, ctx)
    }
}

fn project_first_pc(columns: &[&[f64]], idxs: &[usize], n: usize) -> Result<Vec<f64>, StatsError> {
    if idxs.len() == 1 {
        return Ok(columns[idxs[0]].to_vec());
    }
    // Mean-center and take equal-weight sum as a cheap first-PC surrogate.
    let mut out = vec![0.0; n];
    for &c in idxs {
        if columns[c].len() != n {
            return Err(StatsError::Shape { message: "column length mismatch" });
        }
        let mean = columns[c].iter().sum::<f64>() / n as f64;
        for r in 0..n {
            out[r] += columns[c][r] - mean;
        }
    }
    let norm = out.iter().map(|v| v * v).sum::<f64>().sqrt().max(1e-12);
    for v in &mut out {
        *v /= norm;
    }
    Ok(out)
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
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(1);
        let out = RobustPartialCorrelation::new().test_batch(&req, &mut ws, &ctx).unwrap();
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
        };
        let mut ws = CiWorkspace::default();
        let ctx = ExecutionContext::for_tests(2);
        let a = PartialCorrelation::new().test_batch(&req, &mut ws, &ctx).unwrap();
        let b = WeightedPartialCorrelation::new(w).test_batch(&req, &mut ws, &ctx).unwrap();
        assert!((a.results[0].statistic - b.results[0].statistic).abs() < 1e-9);
    }
}
