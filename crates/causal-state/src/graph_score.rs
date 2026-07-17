//! Incremental graph-score caches (DESIGN.md §20 / §13.8).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use causal_stats::{accumulate_xtx_xty_row, invert_square};

use crate::error::StateError;
use crate::retention::RetentionPolicy;

/// Score family for local graph scores.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum GraphScoreFamily {
    /// Gaussian BIC local score (intercept + linear parents, σ² MLE = SSE/n).
    GaussianBic,
}

/// Semantic cache key for a graph-score table (no pointer identity).
#[derive(Clone, Debug, Eq, PartialEq, Hash)]
pub struct GraphScoreCacheKey {
    /// Data catalog version.
    pub data_version: u64,
    /// Score family.
    pub family: GraphScoreFamily,
    /// Variable-set fingerprint.
    pub var_fingerprint: u64,
    /// Penalty / mechanism fingerprint (e.g. BIC sample-size encoding).
    pub penalty_fingerprint: u64,
}

/// Parent-set edit used for incremental delta scoring.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ParentSetOp {
    /// Replace the parent set of `node` with `parents` (sorted ascending on insert).
    SetParents {
        /// Target node index.
        node: u32,
        /// New parent indices (need not be sorted; cached sorted).
        parents: Arc<[u32]>,
    },
}

/// Column-major tabular view used to compute local Gaussian BIC scores.
#[derive(Clone, Debug)]
pub struct GraphScoreData {
    /// Number of rows.
    pub n_rows: usize,
    /// Number of variables (columns).
    pub n_vars: usize,
    /// Column-major `n_vars × n_rows` values.
    pub columns: Arc<[f64]>,
}

impl GraphScoreData {
    /// Build from column-major storage (`columns.len() == n_vars * n_rows`).
    ///
    /// # Errors
    ///
    /// Shape mismatch.
    pub fn new(n_rows: usize, n_vars: usize, columns: Arc<[f64]>) -> Result<Self, StateError> {
        if n_rows == 0 || n_vars == 0 {
            return Err(StateError::Shape("GraphScoreData requires n_rows,n_vars ≥ 1".into()));
        }
        if columns.len() != n_rows.saturating_mul(n_vars) {
            return Err(StateError::Shape(format!(
                "columns len {} != n_rows {} * n_vars {}",
                columns.len(),
                n_rows,
                n_vars
            )));
        }
        Ok(Self { n_rows, n_vars, columns })
    }

    fn col(&self, j: usize) -> &[f64] {
        let start = j * self.n_rows;
        &self.columns[start..start + self.n_rows]
    }
}

/// Local-score cache keyed by `(node, sorted parent set)`.
#[derive(Clone, Debug)]
pub struct LocalScoreCache {
    /// Cache identity.
    pub key: GraphScoreCacheKey,
    /// Node → (parent-set key → local score).
    entries: HashMap<u32, HashMap<Arc<[u32]>, f64>>,
    /// Current parent sets per node (graph state).
    parents: HashMap<u32, Arc<[u32]>>,
    /// Approximate retained bytes.
    pub bytes: u64,
    /// Retention policy.
    pub retention: RetentionPolicy,
}

impl LocalScoreCache {
    /// Empty cache for `key`.
    #[must_use]
    pub fn new(key: GraphScoreCacheKey) -> Self {
        Self {
            key,
            entries: HashMap::new(),
            parents: HashMap::new(),
            bytes: 0,
            retention: RetentionPolicy::SufficientStatisticsOnly,
        }
    }

    /// Clear all cached local scores and parent assignments.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.parents.clear();
        self.bytes = 0;
    }

    /// Invalidate cached scores for one node.
    pub fn invalidate_node(&mut self, node: u32) {
        self.entries.remove(&node);
    }

    /// Current parent set for `node` (empty if unset).
    #[must_use]
    pub fn parents_of(&self, node: u32) -> Arc<[u32]> {
        self.parents.get(&node).cloned().unwrap_or_else(|| Arc::from([]))
    }

    /// Full graph score = sum of local scores over nodes `0..n_vars`.
    ///
    /// # Errors
    ///
    /// Score computation failures.
    pub fn score_graph(&mut self, data: &GraphScoreData) -> Result<f64, StateError> {
        let mut total = 0.0;
        for node in 0..data.n_vars as u32 {
            total += self.local_score(data, node, &self.parents_of(node))?;
        }
        Ok(total)
    }

    /// Apply a parent-set op and return `(delta, new_total)` vs previous total.
    ///
    /// # Errors
    ///
    /// Unknown node / score failure.
    pub fn delta_score(
        &mut self,
        data: &GraphScoreData,
        op: ParentSetOp,
    ) -> Result<(f64, f64), StateError> {
        let ParentSetOp::SetParents { node, parents } = op;
        if node as usize >= data.n_vars {
            return Err(StateError::Shape(format!("node {node} out of range")));
        }
        let sorted = sorted_parents(&parents, node)?;
        let old_parents = self.parents_of(node);
        let old_local = self.local_score(data, node, &old_parents)?;
        let new_local = self.local_score(data, node, &sorted)?;
        self.parents.insert(node, Arc::clone(&sorted));
        let delta = new_local - old_local;
        // Recompute total from current parent map (small n_vars in unit/conformance).
        let new_total = self.score_graph(data)?;
        Ok((delta, new_total))
    }

    /// Local score for `(node | parents)`, cached.
    ///
    /// # Errors
    ///
    /// Numerical / shape failures.
    pub fn local_score(
        &mut self,
        data: &GraphScoreData,
        node: u32,
        parents: &Arc<[u32]>,
    ) -> Result<f64, StateError> {
        if let Some(s) = self.entries.get(&node).and_then(|m| m.get(parents)).copied() {
            return Ok(s);
        }
        let s = match self.key.family {
            GraphScoreFamily::GaussianBic => gaussian_bic_local(data, node, parents)?,
        };
        self.entries.entry(node).or_default().insert(Arc::clone(parents), s);
        self.bytes = self.bytes.saturating_add(32 + 8 * parents.len() as u64);
        Ok(s)
    }
}

fn sorted_parents(parents: &[u32], node: u32) -> Result<Arc<[u32]>, StateError> {
    let mut v: Vec<u32> = parents.to_vec();
    v.sort_unstable();
    v.dedup();
    if v.iter().any(|&p| p == node) {
        return Err(StateError::Shape(format!("node {node} cannot be its own parent")));
    }
    Ok(Arc::from(v))
}

/// Gaussian BIC local score (higher is better):
/// `-n/2 · (1 + ln(2π) + ln(σ²)) − (k/2) · ln(n)` with `k = |Pa| + 1` (intercept)
/// and `σ² = SSE / n` from OLS of the node on intercept + parents.
fn gaussian_bic_local(
    data: &GraphScoreData,
    node: u32,
    parents: &[u32],
) -> Result<f64, StateError> {
    let n = data.n_rows;
    if n < 2 {
        return Err(StateError::Numerical("need n≥2 for BIC".into()));
    }
    let k = parents.len() + 1; // intercept
    let y = data.col(node as usize);
    let mut xtx = vec![0.0; k * k];
    let mut xty = vec![0.0; k];
    let mut row = vec![0.0; k];
    row[0] = 1.0;
    for r in 0..n {
        for (j, &p) in parents.iter().enumerate() {
            if p as usize >= data.n_vars {
                return Err(StateError::Shape(format!("parent {p} out of range")));
            }
            row[j + 1] = data.col(p as usize)[r];
        }
        accumulate_xtx_xty_row(&row, y[r], &mut xtx, &mut xty);
    }
    let inv = invert_square(&xtx, k)
        .ok_or_else(|| StateError::Numerical("singular parent Gram in BIC".into()))?;
    let mut beta = vec![0.0; k];
    for i in 0..k {
        let mut s = 0.0;
        for j in 0..k {
            s += inv[i * k + j] * xty[j];
        }
        beta[i] = s;
    }
    let mut sse = 0.0;
    for r in 0..n {
        let mut pred = beta[0];
        for (j, &p) in parents.iter().enumerate() {
            pred += beta[j + 1] * data.col(p as usize)[r];
        }
        let e = y[r] - pred;
        sse += e * e;
    }
    let sigma2 = (sse / n as f64).max(1e-12);
    let n_f = n as f64;
    let k_f = k as f64;
    Ok(-0.5 * n_f * (1.0 + (2.0 * std::f64::consts::PI).ln() + sigma2.ln())
        - 0.5 * k_f * n_f.ln())
}

/// Rebuild full graph score without using the cache (acceptance oracle).
///
/// # Errors
///
/// Score failures.
pub fn full_graph_score(
    data: &GraphScoreData,
    family: GraphScoreFamily,
    parents: &HashMap<u32, Arc<[u32]>>,
) -> Result<f64, StateError> {
    let mut total = 0.0;
    for node in 0..data.n_vars as u32 {
        let pa = parents.get(&node).cloned().unwrap_or_else(|| Arc::from([]));
        total += match family {
            GraphScoreFamily::GaussianBic => gaussian_bic_local(data, node, &pa)?,
        };
    }
    Ok(total)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn chain_data() -> GraphScoreData {
        // x0 ~ N, x1 = 2*x0 + noise, x2 = x1 + noise — synthetic linear chain.
        let n = 40usize;
        let mut cols = vec![0.0; 3 * n];
        for i in 0..n {
            let x0 = (i as f64) * 0.1 - 2.0;
            let x1 = 2.0 * x0 + 0.01 * ((i % 3) as f64 - 1.0);
            let x2 = x1 + 0.01 * ((i as f64 * 0.3).sin());
            cols[i] = x0;
            cols[n + i] = x1;
            cols[2 * n + i] = x2;
        }
        GraphScoreData::new(n, 3, Arc::from(cols)).unwrap()
    }

    #[test]
    fn delta_score_matches_full_recompute() {
        let data = chain_data();
        let key = GraphScoreCacheKey {
            data_version: 1,
            family: GraphScoreFamily::GaussianBic,
            var_fingerprint: 3,
            penalty_fingerprint: data.n_rows as u64,
        };
        let mut cache = LocalScoreCache::new(key);
        let empty = Arc::from([]);
        for node in 0..3u32 {
            cache.parents.insert(node, Arc::clone(&empty));
        }
        let s0 = cache.score_graph(&data).unwrap();
        let full0 = full_graph_score(&data, GraphScoreFamily::GaussianBic, &cache.parents).unwrap();
        assert!((s0 - full0).abs() < 1e-10);

        let (delta, new_total) = cache
            .delta_score(
                &data,
                ParentSetOp::SetParents { node: 1, parents: Arc::from([0u32]) },
            )
            .unwrap();
        let full1 = full_graph_score(&data, GraphScoreFamily::GaussianBic, &cache.parents).unwrap();
        assert!((new_total - full1).abs() < 1e-10, "inc={new_total} full={full1}");
        assert!((new_total - (s0 + delta)).abs() < 1e-10);
        // Adding the true parent of x1 should improve the score.
        assert!(delta > 0.0, "delta={delta}");
    }
}
