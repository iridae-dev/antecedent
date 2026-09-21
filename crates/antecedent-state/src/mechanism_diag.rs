//! Rolling linear-Gaussian mechanism diagnostics under `CausalState`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_lossless, clippy::cast_possible_truncation, clippy::implicit_hasher)]

use std::collections::VecDeque;

use antecedent_core::{CacheBudget, StateVersion};
use antecedent_stats::max_abs_cusum;

use crate::error::StateError;
use crate::retention::RetentionPolicy;
use crate::suff_stats::LinearOlsSuffStats;

/// Rebuild Gram from the ring every this many appends to limit subtract drift.
const REBUILD_EVERY: u64 = 64;

/// Rolling window diagnostics for a linear-Gaussian mechanism.
///
/// Maintains a bounded design/response ring and OLS sufficient statistics for the
/// current window. Eviction of a slot affects performance only; reconstruct by
/// replaying the retained window (or re-appending from raw history).
#[derive(Clone, Debug)]
pub struct RollingMechanismDiagnostics {
    /// State version stamp (caller-maintained).
    pub state_version: StateVersion,
    /// Data version stamp (caller-maintained).
    pub data_version: u64,
    /// Retention declaration (`BoundedWindow`).
    pub retention: RetentionPolicy,
    /// Maximum rows retained.
    pub window: usize,
    /// Observations currently in the window.
    pub n: u64,
    /// Latest OLS coefficients (empty until first successful refresh).
    pub beta: Vec<f64>,
    /// Residual sum of squares on the window.
    pub residual_sse: f64,
    /// Residual variance when identifiable.
    pub residual_var: Option<f64>,
    /// Mean absolute residual on the window.
    pub mean_abs_residual: f64,
    /// Max-|CUSUM| of window residuals (None if window shorter than 4).
    pub max_abs_cusum: Option<f64>,
    /// Approximate retained bytes right now.
    pub bytes: u64,
    /// Bytes charged to the cache budget by [`insert_mechanism_diag`]: the slot's
    /// full-window worst case, so growth after insertion cannot exceed what was reserved.
    charged_bytes: u64,
    ols: LinearOlsSuffStats,
    ring: VecDeque<(Vec<f64>, f64)>,
    appends_since_rebuild: u64,
}

impl RollingMechanismDiagnostics {
    /// Empty diagnostics for `ncols` predictors and a trailing window of `window` rows.
    ///
    /// # Errors
    ///
    /// Zero window or zero ncols.
    pub fn new(ncols: usize, window: usize) -> Result<Self, StateError> {
        if ncols == 0 {
            return Err(StateError::Shape("ncols is 0".into()));
        }
        if window == 0 {
            return Err(StateError::Shape("window is 0".into()));
        }
        Ok(Self {
            state_version: StateVersion::ZERO,
            data_version: 0,
            retention: RetentionPolicy::BoundedWindow { max_rows: window as u64 },
            window,
            n: 0,
            beta: Vec::new(),
            residual_sse: 0.0,
            residual_var: None,
            mean_abs_residual: 0.0,
            max_abs_cusum: None,
            bytes: 0,
            charged_bytes: 0,
            ols: LinearOlsSuffStats::new(ncols),
            ring: VecDeque::with_capacity(window),
            appends_since_rebuild: 0,
        })
    }

    /// Predictor dimension.
    #[must_use]
    pub fn ncols(&self) -> usize {
        self.ols.ncols
    }

    /// Append one design row and response; drop the oldest row when over window.
    ///
    /// # Errors
    ///
    /// Row length mismatch, or a non-finite entry (checked before the window is touched).
    pub fn append_row(&mut self, row: &[f64], y: f64) -> Result<(), StateError> {
        if row.len() != self.ols.ncols {
            return Err(StateError::Shape(format!(
                "row len {} != ncols {}",
                row.len(),
                self.ols.ncols
            )));
        }
        if !y.is_finite() || row.iter().any(|v| !v.is_finite()) {
            return Err(StateError::Numerical("design row and response must be finite".into()));
        }
        while self.ring.len() >= self.window {
            self.evict_oldest()?;
        }
        self.ols.append_row(row, y)?;
        self.ring.push_back((row.to_vec(), y));
        self.n = self.ring.len() as u64;
        self.appends_since_rebuild = self.appends_since_rebuild.saturating_add(1);
        if self.appends_since_rebuild >= REBUILD_EVERY {
            self.rebuild_ols_from_ring()?;
        }
        self.bytes = self.bytes_estimate();
        Ok(())
    }

    /// Append a batch of row-major `n×p` rows and responses.
    ///
    /// # Errors
    ///
    /// Shape mismatch.
    pub fn append_batch(&mut self, rows_rowmajor: &[f64], y: &[f64]) -> Result<(), StateError> {
        let p = self.ols.ncols;
        if rows_rowmajor.len() % p != 0 {
            return Err(StateError::Shape("rows not multiple of ncols".into()));
        }
        let n = rows_rowmajor.len() / p;
        if y.len() != n {
            return Err(StateError::Shape("y length mismatch".into()));
        }
        for i in 0..n {
            self.append_row(&rows_rowmajor[i * p..(i + 1) * p], y[i])?;
        }
        Ok(())
    }

    /// Recompute β, SSE, σ², mean |e|, and max-|CUSUM| from the current window.
    ///
    /// # Errors
    ///
    /// Singular Gram or empty window.
    pub fn refresh_summaries(&mut self) -> Result<(), StateError> {
        if self.ring.is_empty() {
            return Err(StateError::Numerical("no observations".into()));
        }
        let beta = self.ols.solve_beta()?;
        let mut sse = 0.0;
        let mut mean_abs = 0.0;
        let mut residuals = Vec::with_capacity(self.ring.len());
        for (row, y) in &self.ring {
            let mut pred = 0.0;
            for (j, &b) in beta.iter().enumerate() {
                pred += b * row[j];
            }
            let e = y - pred;
            sse += e * e;
            mean_abs += e.abs();
            residuals.push(e);
        }
        let n = self.ring.len() as f64;
        let p = self.ols.ncols as f64;
        self.beta = beta;
        self.residual_sse = sse;
        // Window σ² is SSE/(n−p) from the same residuals as residual_sse.
        // IncrementalOLS Welford scatter is append-only and refuses after eviction.
        self.residual_var = if n > p { Some(sse / (n - p)) } else { None };
        self.mean_abs_residual = mean_abs / n;
        self.max_abs_cusum =
            if residuals.len() >= 4 { Some(max_abs_cusum(&residuals)) } else { None };
        self.bytes = self.bytes_estimate();
        Ok(())
    }

    /// Approximate bytes retained by this slot.
    #[must_use]
    pub fn bytes_estimate(&self) -> u64 {
        self.bytes_for(self.ring.len(), self.beta.len())
    }

    /// Bytes the slot occupies once its window is full and `β` is populated: the most it can
    /// ever hold, and what [`insert_mechanism_diag`] reserves.
    #[must_use]
    pub fn worst_case_bytes(&self) -> u64 {
        self.bytes_for(self.window, self.ols.ncols)
    }

    fn bytes_for(&self, ring_rows: usize, beta_len: usize) -> u64 {
        let p = self.ols.ncols;
        let ring_bytes = ring_rows * (p * 8 + 8);
        let ols_bytes = p * p * 8 + p * 8 + 64;
        let beta_bytes = beta_len * 8;
        (ring_bytes + ols_bytes + beta_bytes) as u64
    }

    fn evict_oldest(&mut self) -> Result<(), StateError> {
        let Some((row, y)) = self.ring.pop_front() else {
            return Ok(());
        };
        subtract_row(&mut self.ols, &row, y)?;
        self.n = self.ring.len() as u64;
        Ok(())
    }

    fn rebuild_ols_from_ring(&mut self) -> Result<(), StateError> {
        let mut ols = LinearOlsSuffStats::new(self.ols.ncols);
        ols.retention = RetentionPolicy::BoundedWindow { max_rows: self.window as u64 };
        for (row, y) in &self.ring {
            ols.append_row(row, *y)?;
        }
        self.ols = ols;
        self.appends_since_rebuild = 0;
        Ok(())
    }
}

fn subtract_row(ols: &mut LinearOlsSuffStats, row: &[f64], y: f64) -> Result<(), StateError> {
    if row.len() != ols.ncols {
        return Err(StateError::Shape(format!("row len {} != ncols {}", row.len(), ols.ncols)));
    }
    if ols.n == 0 {
        return Err(StateError::Numerical("cannot subtract from empty OLS".into()));
    }
    for i in 0..ols.ncols {
        for j in 0..ols.ncols {
            ols.xtx[i * ols.ncols + j] -= row[i] * row[j];
        }
        ols.xty[i] -= row[i] * y;
    }
    ols.yty -= y * y;
    ols.n = ols.n.saturating_sub(1);
    Ok(())
}

/// Insert a diagnostics slot into a map while respecting [`CacheBudget`].
///
/// The slot is charged its full-window worst case ([`RollingMechanismDiagnostics::worst_case_bytes`]),
/// not its current size: the ring keeps growing after insertion, and what is refunded on
/// eviction must be exactly what was charged.
///
/// # Errors
///
/// Budget exceeded (same refuse policy as [`crate::store::ResultStore`]).
pub fn insert_mechanism_diag(
    map: &mut std::collections::HashMap<std::sync::Arc<str>, RollingMechanismDiagnostics>,
    key: std::sync::Arc<str>,
    mut diag: RollingMechanismDiagnostics,
    budget: &mut CacheBudget,
) -> Result<(), StateError> {
    let old_charged = map.get(&key).map_or(0, |d| d.charged_bytes);
    let charge = diag.worst_case_bytes();
    let net = charge.saturating_sub(old_charged);
    if !budget.can_admit(net) {
        return Err(StateError::CacheBudget { need: net, remaining: budget.remaining() });
    }
    budget.used_bytes = budget.used_bytes.saturating_sub(old_charged).saturating_add(charge);
    diag.charged_bytes = charge;
    map.insert(key, diag);
    Ok(())
}

/// Remove a diagnostics slot and free exactly the budget it was charged.
pub fn evict_mechanism_diag(
    map: &mut std::collections::HashMap<std::sync::Arc<str>, RollingMechanismDiagnostics>,
    key: &str,
    budget: &mut CacheBudget,
) -> Option<RollingMechanismDiagnostics> {
    let removed = map.remove(key)?;
    budget.used_bytes = budget.used_bytes.saturating_sub(removed.charged_bytes);
    Some(removed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rolling_window_matches_batch_on_last_w() {
        let p = 2usize;
        let w = 5usize;
        let mut roll = RollingMechanismDiagnostics::new(p, w).unwrap();
        let mut all_rows = Vec::new();
        let mut all_y = Vec::new();
        for i in 0..12 {
            let row = [1.0, i as f64];
            let y = 2.0 + 3.0 * i as f64;
            roll.append_row(&row, y).unwrap();
            all_rows.extend_from_slice(&row);
            all_y.push(y);
        }
        roll.refresh_summaries().unwrap();

        let start = all_y.len() - w;
        let mut batch = LinearOlsSuffStats::new(p);
        batch.append_batch(&all_rows[start * p..], &all_y[start..]).unwrap();
        let beta_b = batch.solve_beta().unwrap();
        assert_eq!(roll.n, w as u64);
        assert!((roll.beta[0] - beta_b[0]).abs() < 1e-8);
        assert!((roll.beta[1] - beta_b[1]).abs() < 1e-8);
        let sse_b = {
            let mut s = 0.0;
            for i in 0..w {
                let row = &all_rows[(start + i) * p..(start + i + 1) * p];
                let pred = beta_b[0] * row[0] + beta_b[1] * row[1];
                let e = all_y[start + i] - pred;
                s += e * e;
            }
            s
        };
        assert!((roll.residual_sse - sse_b).abs() < 1e-8);
        let var = roll.residual_var.expect("residual variance after eviction");
        assert!((var - sse_b / (w as f64 - p as f64)).abs() < 1e-8);
    }

    /// A slot is charged its full-window worst case at insert, keeps fitting under that
    /// charge as its ring fills, and gives back exactly what it took on eviction — the budget
    /// returns to zero instead of leaking or draining someone else's share.
    #[test]
    fn budget_charge_covers_growth_and_is_refunded_exactly() {
        use std::collections::HashMap;
        use std::sync::Arc;

        let (p, w) = (2usize, 8usize);
        let mut diag = RollingMechanismDiagnostics::new(p, w).unwrap();
        let mut map: HashMap<Arc<str>, RollingMechanismDiagnostics> = HashMap::new();
        let mut budget = CacheBudget::unlimited();
        let key: Arc<str> = Arc::from("m");
        insert_mechanism_diag(&mut map, Arc::clone(&key), diag.clone(), &mut budget).unwrap();
        // window·(p·8 + 8) ring + (p²·8 + p·8 + 64) Gram + p·8 beta.
        let worst = (w * (p * 8 + 8) + (p * p * 8 + p * 8 + 64) + p * 8) as u64;
        assert_eq!(budget.used_bytes, worst);
        assert_eq!(diag.worst_case_bytes(), worst);

        for i in 0..20 {
            diag.append_row(&[1.0, f64::from(i)], 2.0 + 3.0 * f64::from(i)).unwrap();
        }
        diag.refresh_summaries().unwrap();
        assert!(diag.bytes <= worst, "grown slot {} exceeds its charge {worst}", diag.bytes);
        // Re-inserting the grown slot charges the same amount (no double count).
        insert_mechanism_diag(&mut map, Arc::clone(&key), diag, &mut budget).unwrap();
        assert_eq!(budget.used_bytes, worst);

        assert!(evict_mechanism_diag(&mut map, "m", &mut budget).is_some());
        assert_eq!(budget.used_bytes, 0);
    }

    /// A non-finite row is refused before the window is evicted or the fit is touched.
    #[test]
    fn nonfinite_row_leaves_window_untouched() {
        let mut roll = RollingMechanismDiagnostics::new(2, 3).unwrap();
        for i in 0..3 {
            roll.append_row(&[1.0, f64::from(i)], f64::from(i)).unwrap();
        }
        let before_n = roll.n;
        assert!(roll.append_row(&[1.0, f64::NAN], 1.0).is_err());
        assert_eq!(roll.n, before_n, "the oldest row must not be evicted for a rejected row");
        roll.refresh_summaries().unwrap();
        assert!((roll.beta[1] - 1.0).abs() < 1e-10, "β={:?}", roll.beta);
    }

    #[test]
    fn empty_refresh_errors() {
        let mut roll = RollingMechanismDiagnostics::new(2, 4).unwrap();
        assert!(roll.refresh_summaries().is_err());
    }

    #[test]
    fn shape_mismatch_errors() {
        let mut roll = RollingMechanismDiagnostics::new(2, 4).unwrap();
        assert!(roll.append_row(&[1.0], 0.0).is_err());
    }
}
