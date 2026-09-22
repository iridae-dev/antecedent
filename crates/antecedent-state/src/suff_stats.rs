//! Incremental sufficient statistics.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::similar_names)] // xtx / xty

use antecedent_stats::{accumulate_xtx_xty_row, invert_square};

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
    /// Welford scatter of `[x | y]` for a translation-stable residual scale.
    joint: StreamingCovariance,
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
            joint: StreamingCovariance::new(ncols.saturating_add(1)),
            retention: RetentionPolicy::SufficientStatisticsOnly,
        }
    }

    /// Append one design row and response.
    ///
    /// The row is validated before anything is touched: a rejected row leaves the statistics
    /// exactly as they were, so a single bad observation cannot poison the accumulated
    /// Gram, the Welford scatter and the count out of step with each other.
    ///
    /// # Errors
    ///
    /// Row length mismatch, or a non-finite entry in the row or the response.
    pub fn append_row(&mut self, row: &[f64], y: f64) -> Result<(), StateError> {
        if row.len() != self.ncols {
            return Err(StateError::Shape(format!(
                "row len {} != ncols {}",
                row.len(),
                self.ncols
            )));
        }
        if !y.is_finite() || row.iter().any(|v| !v.is_finite()) {
            return Err(StateError::Numerical("design row and response must be finite".into()));
        }
        let mut joint_row = Vec::with_capacity(self.ncols.saturating_add(1));
        joint_row.extend_from_slice(row);
        joint_row.push(y);
        // Fallible step first (count overflow); everything after it cannot fail.
        self.joint.append(&joint_row)?;
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

    /// Residual variance estimate `σ² = SSE / (n − p)` when `n > p`.
    ///
    /// `SSE = Σ (y − xᵀβ)²` for *any* `β` and any design (with or without a constant
    /// column). It is assembled from the mean-centred Welford scatter, which is stable under
    /// a large common offset but only equals `SSE` when the residuals average to zero:
    /// `Σ r² = Σ (r − r̄)² + n r̄²` with `r̄ = ȳ − x̄ᵀβ`. For OLS on a design containing an
    /// intercept `r̄ = 0` and the second term vanishes; through-origin designs and
    /// externally supplied coefficients need it. `r̄` is taken as zero when it is below the
    /// round-off of its own evaluation, so a huge offset does not turn rounding noise into
    /// variance.
    #[must_use]
    pub fn residual_variance(&self, beta: &[f64]) -> Option<f64> {
        if beta.len() != self.ncols || usize::try_from(self.n).is_ok_and(|n| n <= self.ncols) {
            return None;
        }
        if self.joint.n != self.n || self.joint.dim != self.ncols.saturating_add(1) {
            return None;
        }
        let p = self.ncols;
        let dim = p + 1;
        let s = &self.joint.m2;
        let mut sse = s[p * dim + p];
        let mut bsy = 0.0;
        let mut bsb = 0.0;
        for i in 0..p {
            bsy += beta[i] * s[i * dim + p];
            for j in 0..p {
                bsb += beta[i] * s[i * dim + j] * beta[j];
            }
        }
        sse = sse - 2.0 * bsy + bsb;
        if !sse.is_finite() {
            return None;
        }
        // Mean residual r̄ = ȳ − x̄ᵀβ and the magnitude of the terms it is a difference of.
        let means = &self.joint.mean;
        let mut r_bar = means[p];
        let mut r_scale = means[p].abs();
        for i in 0..p {
            let term = beta[i] * means[i];
            r_bar -= term;
            r_scale += term.abs();
        }
        if r_bar.abs() > 8.0 * f64::EPSILON * r_scale {
            sse += self.n as f64 * r_bar * r_bar;
        }
        if sse < 0.0 {
            let scale = s[p * dim + p].abs().max(1.0);
            if sse.abs() <= 1e-12 * scale {
                sse = 0.0;
            } else {
                return None;
            }
        }
        Some(sse / (self.n as f64 - p as f64))
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
        if x.iter().any(|value| !value.is_finite()) {
            return Err(StateError::Numerical("covariance observations must be finite".into()));
        }
        let next_n = self
            .n
            .checked_add(1)
            .ok_or_else(|| StateError::Numerical("covariance count overflow".into()))?;
        if self.n == 0 {
            self.mean.copy_from_slice(x);
            self.n = next_n;
            return Ok(());
        }
        // Welford's rank-one scatter update uses the old mean on both axes.
        // Updating means afterward eliminates two scratch allocations per row
        // and makes symmetry exact rather than relying on rounded delta2 values.
        let correction = self.n as f64 / next_n as f64;
        for i in 0..self.dim {
            let delta_i = x[i] - self.mean[i];
            for j in i..self.dim {
                let value =
                    self.m2[i * self.dim + j] + delta_i * (x[j] - self.mean[j]) * correction;
                self.m2[i * self.dim + j] = value;
                self.m2[j * self.dim + i] = value;
            }
        }
        for (mean, value) in self.mean.iter_mut().zip(x) {
            *mean += (value - *mean) / next_n as f64;
        }
        self.n = next_n;
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
    fn streaming_covariance_rejects_nonfinite_rows_without_poisoning_state() {
        let mut stats = StreamingCovariance::new(2);
        stats.append(&[1.0, 2.0]).unwrap();
        let before = stats.clone();
        assert!(stats.append(&[3.0, f64::NAN]).is_err());
        assert_eq!(stats, before);
        stats.append(&[3.0, 4.0]).unwrap();
        let covariance = stats.sample_covariance().unwrap();
        assert!(covariance.iter().all(|value| (*value - 2.0).abs() < 1e-12));
    }

    /// Brute-force `Σ (y − xᵀβ)² / (n − p)`, independent of the accumulated scatter.
    fn brute_force_sigma2(rows: &[f64], y: &[f64], p: usize, beta: &[f64]) -> f64 {
        let n = y.len();
        let sse: f64 = (0..n)
            .map(|i| {
                let pred: f64 = (0..p).map(|j| rows[i * p + j] * beta[j]).sum();
                (y[i] - pred) * (y[i] - pred)
            })
            .sum();
        sse / (n - p) as f64
    }

    /// Through-origin fit (no constant column): the residuals do not average to zero, so the
    /// centred scatter alone understates σ². x = [1,2,3], y = [11,12,13]: β = 74/14,
    /// Σr² = 300/7, σ² = 150/7 (the centred scatter gives 36.7/2 = 18.4).
    #[test]
    fn residual_variance_is_exact_without_an_intercept_column() {
        let rows = [1.0, 2.0, 3.0];
        let y = [11.0, 12.0, 13.0];
        let mut stats = LinearOlsSuffStats::new(1);
        stats.append_batch(&rows, &y).unwrap();
        let beta = stats.solve_beta().unwrap();
        assert!((beta[0] - 74.0 / 14.0).abs() < 1e-12);
        let var = stats.residual_variance(&beta).expect("variance");
        assert!((var - 150.0 / 7.0).abs() < 1e-10, "got {var}");
        assert!((var - brute_force_sigma2(&rows, &y, 1, &beta)).abs() < 1e-10);

        // x = [1,2,3], y = [2,3,5]: β = 23/14, Σr² = 38 − 23²/14 = 3/14, σ² = 3/28.
        let y2 = [2.0, 3.0, 5.0];
        let mut s2 = LinearOlsSuffStats::new(1);
        s2.append_batch(&rows, &y2).unwrap();
        let b2 = s2.solve_beta().unwrap();
        let v2 = s2.residual_variance(&b2).expect("variance");
        assert!((v2 - 3.0 / 28.0).abs() < 1e-12, "got {v2}");
    }

    /// An externally supplied β that is not the OLS solution: y = 1 + 2x fitted exactly by
    /// (1, 2); the wrong intercept (0, 2) leaves a constant residual 1, so Σr² = 4 and
    /// σ² = 4 / (4 − 2) = 2. A centred scatter sees a constant and reports 0.
    #[test]
    fn residual_variance_is_exact_for_a_non_ols_beta() {
        let rows = [1.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0, 3.0];
        let y = [1.0, 3.0, 5.0, 7.0];
        let mut stats = LinearOlsSuffStats::new(2);
        stats.append_batch(&rows, &y).unwrap();
        let wrong = [0.0, 2.0];
        let var = stats.residual_variance(&wrong).expect("variance");
        assert!((var - 2.0).abs() < 1e-12, "got {var}");
        assert!((var - brute_force_sigma2(&rows, &y, 2, &wrong)).abs() < 1e-12);
        // The true OLS β still gives (numerically) zero residual variance.
        let ols = stats.solve_beta().unwrap();
        assert!(stats.residual_variance(&ols).expect("variance").abs() < 1e-12);
    }

    /// A non-finite row is refused before any state changes; later valid rows still
    /// accumulate exactly as if the bad row had never been offered.
    #[test]
    fn nonfinite_row_is_rejected_atomically() {
        let mut stats = LinearOlsSuffStats::new(2);
        stats.append_row(&[1.0, 0.0], 1.0).unwrap();
        stats.append_row(&[1.0, 1.0], 3.0).unwrap();
        let before = stats.clone();
        assert!(stats.append_row(&[1.0, f64::NAN], 2.0).is_err());
        assert!(stats.append_row(&[1.0, 2.0], f64::INFINITY).is_err());
        assert_eq!(stats, before);

        let mut clean = LinearOlsSuffStats::new(2);
        clean.append_row(&[1.0, 0.0], 1.0).unwrap();
        clean.append_row(&[1.0, 1.0], 3.0).unwrap();
        stats.append_row(&[1.0, 2.0], 5.0).unwrap();
        clean.append_row(&[1.0, 2.0], 5.0).unwrap();
        assert_eq!(stats, clean);
        let beta = stats.solve_beta().unwrap();
        assert!((beta[0] - 1.0).abs() < 1e-10 && (beta[1] - 2.0).abs() < 1e-10);
    }

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
    fn incremental_ols_residual_variance_survives_large_offset() {
        let offset = 1e12;
        let eps = [-2.0, -1.0, 0.0, 1.0, 2.0];
        let y: Vec<f64> = eps.iter().map(|e| offset + e).collect();
        let rows = vec![1.0; 5];
        let mut stats = LinearOlsSuffStats::new(1);
        stats.append_batch(&rows, &y).unwrap();
        let beta = stats.solve_beta().unwrap();
        let var = stats.residual_variance(&beta).expect("residual variance");
        assert!((var - 2.5).abs() < 1e-9, "got {var}");

        let shifted: Vec<f64> = y.iter().map(|yi| yi + 1e9).collect();
        let mut shifted_stats = LinearOlsSuffStats::new(1);
        shifted_stats.append_batch(&rows, &shifted).unwrap();
        let shifted_var = shifted_stats.residual_variance(&shifted_stats.solve_beta().unwrap());
        assert!((shifted_var.unwrap() - var).abs() < 1e-9);
    }

    #[test]
    fn streaming_cov_matches_batch() {
        // Distinct variances, partial correlation: x2 = 0.5*x1 + noise
        let data: Vec<[f64; 2]> = (0..20)
            .map(|i| {
                let x1 = f64::from(i) - 9.5;
                let x2 = 0.5 * x1 + f64::from(i % 3) - 1.0;
                [x1, x2 * 2.0]
            })
            .collect();
        let mut s = StreamingCovariance::new(2);
        for row in &data {
            s.append(row).expect("append");
        }
        let cov = s.sample_covariance().expect("cov");

        let n = data.len() as f64;
        let mut mean = [0.0, 0.0];
        for row in &data {
            mean[0] += row[0];
            mean[1] += row[1];
        }
        mean[0] /= n;
        mean[1] /= n;
        let mut batch = [0.0; 4];
        for row in &data {
            let d0 = row[0] - mean[0];
            let d1 = row[1] - mean[1];
            batch[0] += d0 * d0;
            batch[1] += d0 * d1;
            batch[2] += d1 * d0;
            batch[3] += d1 * d1;
        }
        let denom = n - 1.0;
        for v in &mut batch {
            *v /= denom;
        }

        for i in 0..4 {
            assert!((cov[i] - batch[i]).abs() < 1e-10, "cov[{i}]={} batch={}", cov[i], batch[i]);
        }
        assert!((cov[1] - cov[2]).abs() < 1e-12, "covariance must be symmetric");
        // Off-diagonal must be non-trivial (catches the old delta[i]*di bug)
        assert!(cov[1].abs() > 1.0);
    }
}
