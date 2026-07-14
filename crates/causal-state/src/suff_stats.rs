//! Incremental sufficient statistics (DESIGN.md §20).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::similar_names)] // xtx / xty

use causal_stats::{accumulate_xtx_xty_row, invert_square};

use crate::error::StateError;
use crate::retention::RetentionPolicy;

/// Linear OLS sufficient statistics (`XᵀX`, `Xᵀy`, `n`).
#[derive(Clone, Debug, PartialEq)]
pub struct LinearOlsSuffStats {
    /// Number of columns `p`.
    pub ncols: usize,
    /// Row-major `p×p` Gram.
    pub xtx: Vec<f64>,
    /// `Xᵀy` length `p`.
    pub xty: Vec<f64>,
    /// Sample count.
    pub n: u64,
    /// Sum of squared responses (for residual variance).
    pub yty: f64,
    /// Retention declaration.
    pub retention: RetentionPolicy,
}

impl LinearOlsSuffStats {
    /// Empty stats for `ncols` predictors.
    #[must_use]
    pub fn new(ncols: usize) -> Self {
        Self {
            ncols,
            xtx: vec![0.0; ncols * ncols],
            xty: vec![0.0; ncols],
            n: 0,
            yty: 0.0,
            retention: RetentionPolicy::SufficientStatisticsOnly,
        }
    }

    /// Append one design row and response.
    ///
    /// # Errors
    ///
    /// Row length mismatch.
    pub fn append_row(&mut self, row: &[f64], y: f64) -> Result<(), StateError> {
        if row.len() != self.ncols {
            return Err(StateError::Shape(format!(
                "row len {} != ncols {}",
                row.len(),
                self.ncols
            )));
        }
        accumulate_xtx_xty_row(row, y, &mut self.xtx, &mut self.xty);
        self.yty += y * y;
        self.n = self.n.saturating_add(1);
        Ok(())
    }

    /// Append a batch of rows (row-major `n×p`) and responses.
    ///
    /// # Errors
    ///
    /// Shape mismatch.
    pub fn append_batch(&mut self, rows_rowmajor: &[f64], y: &[f64]) -> Result<(), StateError> {
        if self.ncols == 0 {
            return Err(StateError::Shape("ncols is 0".into()));
        }
        if rows_rowmajor.len() % self.ncols != 0 {
            return Err(StateError::Shape("rows not multiple of ncols".into()));
        }
        let n = rows_rowmajor.len() / self.ncols;
        if y.len() != n {
            return Err(StateError::Shape("y length mismatch".into()));
        }
        for i in 0..n {
            let row = &rows_rowmajor[i * self.ncols..(i + 1) * self.ncols];
            self.append_row(row, y[i])?;
        }
        Ok(())
    }

    /// Solve OLS coefficients `β = (XᵀX)⁻¹ Xᵀy`.
    ///
    /// # Errors
    ///
    /// Singular Gram or empty data.
    pub fn solve_beta(&self) -> Result<Vec<f64>, StateError> {
        if self.n == 0 {
            return Err(StateError::Numerical("no observations".into()));
        }
        let inv = invert_square(&self.xtx, self.ncols)
            .ok_or_else(|| StateError::Numerical("singular XtX".into()))?;
        let mut beta = vec![0.0; self.ncols];
        for i in 0..self.ncols {
            let mut s = 0.0;
            for j in 0..self.ncols {
                s += inv[i * self.ncols + j] * self.xty[j];
            }
            beta[i] = s;
        }
        Ok(beta)
    }

    /// Residual variance estimate `σ² = (yty − βᵀXᵀy) / (n − p)` when `n > p`.
    #[must_use]
    pub fn residual_variance(&self, beta: &[f64]) -> Option<f64> {
        if beta.len() != self.ncols || self.n as usize <= self.ncols {
            return None;
        }
        let mut bxty = 0.0;
        for i in 0..self.ncols {
            bxty += beta[i] * self.xty[i];
        }
        let sse = (self.yty - bxty).max(0.0);
        Some(sse / (self.n as f64 - self.ncols as f64))
    }
}

/// Streaming mean / covariance (Welford / pairwise updates).
#[derive(Clone, Debug, PartialEq)]
pub struct StreamingCovariance {
    /// Dimension.
    pub dim: usize,
    /// Observation count.
    pub n: u64,
    /// Running mean.
    pub mean: Vec<f64>,
    /// Upper-triangular packed? — store full `dim×dim` unnormalized scatter `M2`.
    pub m2: Vec<f64>,
    /// Retention.
    pub retention: RetentionPolicy,
}

impl StreamingCovariance {
    /// Empty streaming covariance for `dim`.
    #[must_use]
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            n: 0,
            mean: vec![0.0; dim],
            m2: vec![0.0; dim * dim],
            retention: RetentionPolicy::SufficientStatisticsOnly,
        }
    }

    /// Append one observation.
    ///
    /// # Errors
    ///
    /// Dimension mismatch.
    pub fn append(&mut self, x: &[f64]) -> Result<(), StateError> {
        if x.len() != self.dim {
            return Err(StateError::Shape(format!("cov dim {} != {}", x.len(), self.dim)));
        }
        self.n = self.n.saturating_add(1);
        let n = self.n as f64;
        let mut delta = vec![0.0; self.dim];
        for i in 0..self.dim {
            delta[i] = x[i] - self.mean[i];
            self.mean[i] += delta[i] / n;
        }
        for i in 0..self.dim {
            let di = x[i] - self.mean[i];
            for j in 0..self.dim {
                self.m2[i * self.dim + j] += delta[i] * di;
            }
        }
        Ok(())
    }

    /// Sample covariance matrix (`n − 1` denominator); `None` if `n < 2`.
    #[must_use]
    pub fn sample_covariance(&self) -> Option<Vec<f64>> {
        if self.n < 2 {
            return None;
        }
        let denom = (self.n - 1) as f64;
        Some(self.m2.iter().map(|v| v / denom).collect())
    }
}

/// Cached lagged sample-index key (semantic; no borrowed buffers).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct LagIndexCacheKey {
    /// Data version.
    pub data_version: u64,
    /// Max lag.
    pub max_lag: u32,
    /// Variable set fingerprint.
    pub var_fingerprint: u64,
}

/// Lag-index cache entry metadata (values reconstructed by callers).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct LagIndexCacheEntry {
    /// Key.
    pub key: LagIndexCacheKey,
    /// Approximate retained bytes.
    pub bytes: u64,
    /// Retention.
    pub retention: RetentionPolicy,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn incremental_ols_matches_full_batch() {
        let rows = [
            1.0, 0.0, //
            1.0, 1.0, //
            1.0, 2.0, //
            1.0, 3.0,
        ];
        let y = [1.0, 3.0, 5.0, 7.0];
        let mut full = LinearOlsSuffStats::new(2);
        full.append_batch(&rows, &y).expect("batch");
        let mut inc = LinearOlsSuffStats::new(2);
        for i in 0..4 {
            inc.append_row(&rows[i * 2..(i + 1) * 2], y[i]).expect("row");
        }
        assert_eq!(full.n, inc.n);
        for i in 0..4 {
            assert!((full.xtx[i] - inc.xtx[i]).abs() < 1e-12);
        }
        let b_full = full.solve_beta().expect("beta");
        let b_inc = inc.solve_beta().expect("beta");
        assert!((b_full[0] - b_inc[0]).abs() < 1e-10);
        assert!((b_full[1] - b_inc[1]).abs() < 1e-10);
        // y ≈ 1 + 2x
        assert!((b_inc[0] - 1.0).abs() < 1e-8);
        assert!((b_inc[1] - 2.0).abs() < 1e-8);
    }

    #[test]
    fn streaming_cov_matches_batch() {
        let data = [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]];
        let mut s = StreamingCovariance::new(2);
        for row in &data {
            s.append(row).expect("append");
        }
        let cov = s.sample_covariance().expect("cov");
        // Manual: means (3,4); deviations (-2,-2),(0,0),(2,2); M2 var = 8 → s²=4
        assert!((cov[0] - 4.0).abs() < 1e-10);
        assert!((cov[1] - 4.0).abs() < 1e-10);
        assert!((cov[3] - 4.0).abs() < 1e-10);
    }
}
