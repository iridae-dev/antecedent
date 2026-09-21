//! Incremental graph-score caches.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::implicit_hasher, clippy::similar_names)]

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_stats::{DenseLinearAlgebra, FaerBackend, LeastSquaresWorkspace};

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
    /// Content fingerprint (shape and every value's bits), fixed at construction.
    fingerprint: u64,
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
        let fingerprint = content_fingerprint(n_rows, n_vars, &columns);
        Ok(Self { n_rows, n_vars, columns, fingerprint })
    }

    /// Fingerprint of the shape and values; equal data always has equal fingerprints, and a
    /// [`LocalScoreCache`] uses it to refuse scoring different data.
    #[must_use]
    pub const fn fingerprint(&self) -> u64 {
        self.fingerprint
    }

    fn col(&self, j: usize) -> &[f64] {
        let start = j * self.n_rows;
        &self.columns[start..start + self.n_rows]
    }
}

/// FNV-1a over the shape and the bit pattern of every value.
fn content_fingerprint(n_rows: usize, n_vars: usize, columns: &[f64]) -> u64 {
    const PRIME: u64 = 0x0000_0100_0000_01B3;
    let mut h: u64 = 0xCBF2_9CE4_8422_2325;
    let mut mix = |word: u64| {
        for byte in word.to_le_bytes() {
            h ^= u64::from(byte);
            h = h.wrapping_mul(PRIME);
        }
    };
    mix(n_rows as u64);
    mix(n_vars as u64);
    for v in columns {
        mix(v.to_bits());
    }
    h
}

/// Local-score cache keyed by `(node, sorted parent set)`.
///
/// The cache binds to the first [`GraphScoreData`] it scores (by content fingerprint) and
/// refuses any other data until [`Self::clear`]: a `(node, parents)` key says nothing about
/// which rows the score was computed on.
#[derive(Clone, Debug)]
pub struct LocalScoreCache {
    /// Cache identity.
    pub key: GraphScoreCacheKey,
    /// Node → (parent-set key → local score).
    entries: HashMap<u32, HashMap<Arc<[u32]>, f64>>,
    /// Current parent sets per node (graph state).
    parents: HashMap<u32, Arc<[u32]>>,
    /// Fingerprint of the data the cached scores belong to.
    bound_data: Option<u64>,
    /// Sum of local scores under the current parent sets, maintained incrementally.
    running_total: Option<f64>,
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
            bound_data: None,
            running_total: None,
            bytes: 0,
            retention: RetentionPolicy::SufficientStatisticsOnly,
        }
    }

    /// Clear all cached local scores and parent assignments, and unbind from the data.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.parents.clear();
        self.bound_data = None;
        self.running_total = None;
        self.bytes = 0;
    }

    /// Invalidate cached scores for one node.
    pub fn invalidate_node(&mut self, node: u32) {
        self.entries.remove(&node);
        self.running_total = None;
    }

    /// Bind to `data` on first use; refuse different data afterwards.
    fn bind(&mut self, data: &GraphScoreData) -> Result<(), StateError> {
        match self.bound_data {
            None => {
                self.bound_data = Some(data.fingerprint);
                Ok(())
            }
            Some(bound) if bound == data.fingerprint => Ok(()),
            Some(_) => Err(StateError::StaleCache(
                "local scores were computed on different data; call clear() first".into(),
            )),
        }
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
        self.running_total = Some(total);
        Ok(total)
    }

    /// Apply a parent-set op and return `(delta, new_total)` vs previous total.
    ///
    /// Atomic: both local scores are computed before anything is committed, so an error
    /// leaves the parent assignment and the running total untouched. The new total is the
    /// old total plus the local delta, not a rescore of the whole graph.
    ///
    /// # Errors
    ///
    /// Unknown node / score failure / data other than the cache's.
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
        let old_total = match self.running_total {
            Some(total) => total,
            None => self.score_graph(data)?,
        };
        let delta = new_local - old_local;
        let new_total = old_total + delta;
        // Commit: nothing above can fail past this point.
        self.parents.insert(node, sorted);
        self.running_total = Some(new_total);
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
        self.bind(data)?;
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
    let mut x = vec![0.0; n.saturating_mul(k)];
    x[..n].fill(1.0);
    for (j, &p) in parents.iter().enumerate() {
        if p as usize >= data.n_vars {
            return Err(StateError::Shape(format!("parent {p} out of range")));
        }
        x[(j + 1) * n..(j + 2) * n].copy_from_slice(data.col(p as usize));
    }
    let fit = FaerBackend
        .least_squares(&x, n, k, y, &mut LeastSquaresWorkspace::default())
        .map_err(|e| StateError::Numerical(format!("Gaussian BIC local fit: {e}")))?;
    bic_from_rss(fit.rss, n, k)
}

/// Gaussian BIC from the residual sum of squares of a `k`-parameter fit on `n` rows.
///
/// A zero RSS is a degenerate fit (a duplicated or deterministic column): the likelihood is
/// unbounded, so there is no finite score to report. `+∞` would make the graph total `+∞`
/// and the next edit's delta `∞ − ∞ = NaN`, and a search would treat the first such parent
/// set as unbeatable; it is an error instead.
fn bic_from_rss(rss: f64, n: usize, k: usize) -> Result<f64, StateError> {
    let sigma2 = rss / n as f64;
    if !sigma2.is_finite() {
        return Err(StateError::Numerical("non-finite Gaussian BIC residual variance".into()));
    }
    if sigma2 <= 0.0 {
        return Err(StateError::Numerical(
            "degenerate Gaussian BIC fit (zero residual variance)".into(),
        ));
    }
    let n_f = n as f64;
    let k_f = k as f64;
    Ok(-0.5 * n_f * (1.0 + (2.0 * std::f64::consts::PI).ln() + sigma2.ln()) - 0.5 * k_f * n_f.ln())
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

    fn fresh_cache(data: &GraphScoreData) -> LocalScoreCache {
        LocalScoreCache::new(GraphScoreCacheKey {
            data_version: 1,
            family: GraphScoreFamily::GaussianBic,
            var_fingerprint: 3,
            penalty_fingerprint: data.n_rows as u64,
        })
    }

    /// A cache filled from one dataset must refuse another (the `(node, parents)` key says
    /// nothing about which rows the score was computed on), accept equal data, and accept
    /// new data after `clear()`.
    #[test]
    fn cache_refuses_different_data() {
        let a = chain_data();
        let mut cache = fresh_cache(&a);
        let parents: Arc<[u32]> = Arc::from([0u32]);
        let s = cache.local_score(&a, 1, &parents).unwrap();

        let mut cols = a.columns.to_vec();
        cols[a.n_rows + 3] += 1.0; // perturb one value of x1
        let b = GraphScoreData::new(a.n_rows, a.n_vars, Arc::from(cols)).unwrap();
        assert_ne!(a.fingerprint(), b.fingerprint());
        let err = cache.local_score(&b, 1, &parents).unwrap_err();
        assert!(matches!(err, StateError::StaleCache(_)), "{err:?}");

        // Rebuilt-equal data is the same data.
        let same = GraphScoreData::new(a.n_rows, a.n_vars, Arc::clone(&a.columns)).unwrap();
        assert_eq!(cache.local_score(&same, 1, &parents).unwrap(), s);

        // After clear() the cache rebinds and the new data is scored on its own values.
        cache.clear();
        let fresh = cache.local_score(&b, 1, &parents).unwrap();
        assert_eq!(fresh, gaussian_bic_local(&b, 1, &parents).unwrap());
        assert_ne!(fresh, s);
    }

    /// A rejected edit leaves the graph state and the running total exactly as they were.
    #[test]
    fn failed_delta_is_atomic() {
        let data = chain_data();
        let mut cache = fresh_cache(&data);
        let total0 = cache.score_graph(&data).unwrap();
        let err = cache
            .delta_score(&data, ParentSetOp::SetParents { node: 1, parents: Arc::from([0u32, 9]) })
            .unwrap_err();
        assert!(matches!(err, StateError::Shape(_)), "{err:?}");
        assert!(cache.parents_of(1).is_empty());
        // The next valid edit's total is built on the untouched previous total.
        let (delta, total) = cache
            .delta_score(&data, ParentSetOp::SetParents { node: 1, parents: Arc::from([0u32]) })
            .unwrap();
        assert!((total - (total0 + delta)).abs() < 1e-12);
    }

    /// The running total after a chain of edits equals a from-scratch rescore.
    #[test]
    fn running_total_tracks_full_rescore_across_edits() {
        let data = chain_data();
        let mut cache = fresh_cache(&data);
        cache.score_graph(&data).unwrap();
        let mut last = 0.0;
        for (node, pa) in [(1u32, vec![0u32]), (2, vec![1]), (2, vec![0, 1]), (1, vec![])] {
            let (_, total) = cache
                .delta_score(&data, ParentSetOp::SetParents { node, parents: Arc::from(pa) })
                .unwrap();
            last = total;
            let full =
                full_graph_score(&data, GraphScoreFamily::GaussianBic, &cache.parents).unwrap();
            assert!((total - full).abs() < 1e-9, "running={total} full={full}");
        }
        assert!(last.is_finite());
    }

    /// A zero-RSS fit has no finite Gaussian likelihood: error, never `+∞`. The finite branch
    /// is pinned to the definition `−n/2 (1 + ln 2π + ln σ²) − k/2 ln n` at σ² = 0.1.
    #[test]
    fn degenerate_fit_is_an_error_not_infinity() {
        let err = bic_from_rss(0.0, 10, 2).unwrap_err();
        assert!(matches!(err, StateError::Numerical(_)), "{err:?}");
        let got = bic_from_rss(1.0, 10, 2).unwrap();
        let expected =
            -5.0 * (1.0 + (2.0 * std::f64::consts::PI).ln() + 0.1_f64.ln()) - 1.0 * 10.0_f64.ln();
        assert!((got - expected).abs() < 1e-12, "got={got} expected={expected}");
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
            .delta_score(&data, ParentSetOp::SetParents { node: 1, parents: Arc::from([0u32]) })
            .unwrap();
        let full1 = full_graph_score(&data, GraphScoreFamily::GaussianBic, &cache.parents).unwrap();
        assert!((new_total - full1).abs() < 1e-10, "inc={new_total} full={full1}");
        assert!((new_total - (s0 + delta)).abs() < 1e-10);
        // Adding the true parent of x1 should improve the score.
        assert!(delta > 0.0, "delta={delta}");
    }

    /// Independent reference for the Gaussian BIC of a simple linear regression.
    ///
    /// Deliberately avoids the Gram-accumulate / `invert_square` path the
    /// implementation uses, so this is a genuine cross-check rather than a
    /// restatement: OLS via the two-variable closed form, SSE by direct
    /// residual accumulation, then the score assembled from the definition
    /// `l_max - (k/2) ln n` with `l_max = -n/2 (ln 2pi + ln sigma2 + 1)`.
    fn reference_bic_one_parent(x: &[f64], y: &[f64]) -> f64 {
        let n = x.len();
        let n_f = n as f64;
        let xbar = x.iter().sum::<f64>() / n_f;
        let ybar = y.iter().sum::<f64>() / n_f;
        let sxy: f64 = x.iter().zip(y).map(|(xi, yi)| (xi - xbar) * (yi - ybar)).sum();
        let sxx: f64 = x.iter().map(|xi| (xi - xbar) * (xi - xbar)).sum();
        let slope = sxy / sxx;
        let intercept = ybar - slope * xbar;
        let sse: f64 = x
            .iter()
            .zip(y)
            .map(|(xi, yi)| {
                let e = yi - (intercept + slope * xi);
                e * e
            })
            .sum();
        let sigma2 = sse / n_f;
        let k_f = 2.0; // slope + intercept
        -0.5 * n_f * (1.0 + (2.0 * std::f64::consts::PI).ln() + sigma2.ln()) - 0.5 * k_f * n_f.ln()
    }

    #[test]
    fn gaussian_bic_matches_closed_form_regression() {
        let data = chain_data();
        let n = data.n_rows;
        let x: Vec<f64> = data.col(0).to_vec();
        let y: Vec<f64> = data.col(1).to_vec();

        let got = gaussian_bic_local(&data, 1, &[0u32]).unwrap();
        let expected = reference_bic_one_parent(&x, &y);
        assert!((got - expected).abs() < 1e-9, "one-parent BIC: got {got}, closed form {expected}");

        // Intercept-only: sigma2 is the population variance of y and k = 1.
        let ybar = y.iter().sum::<f64>() / n as f64;
        let sigma2 = y.iter().map(|yi| (yi - ybar) * (yi - ybar)).sum::<f64>() / n as f64;
        let n_f = n as f64;
        let expected_empty =
            -0.5 * n_f * (1.0 + (2.0 * std::f64::consts::PI).ln() + sigma2.ln()) - 0.5 * n_f.ln();
        let got_empty = gaussian_bic_local(&data, 1, &[]).unwrap();
        assert!(
            (got_empty - expected_empty).abs() < 1e-9,
            "intercept-only BIC: got {got_empty}, closed form {expected_empty}"
        );
    }

    #[test]
    fn gaussian_bic_loglik_is_monotone_under_nesting() {
        // Backing the penalty out of the score recovers the maximized
        // log-likelihood. Enlarging a parent set can only improve in-sample
        // fit, so that quantity must be non-decreasing -- a wrong parameter
        // count `k` breaks this because the penalty subtracted would not match
        // the penalty implied by the parent set.
        //
        // The exact value of `k` is pinned separately, and non-tautologically,
        // by `gaussian_bic_matches_closed_form_regression` (|Pa| = 0 and 1).
        let data = chain_data();
        let n_f = data.n_rows as f64;
        let loglik = |parents: &[u32]| {
            let k = parents.len() as f64 + 1.0;
            gaussian_bic_local(&data, 2, parents).unwrap() + 0.5 * k * n_f.ln()
        };
        let nested = loglik(&[1u32]);
        let full = loglik(&[0u32, 1u32]);
        assert!(full >= nested - 1e-9, "log-likelihood fell on nesting: {nested} -> {full}");
        assert!(loglik(&[1u32]) >= loglik(&[]) - 1e-9);
    }

    #[test]
    fn gaussian_bic_score_difference_is_invariant_to_outcome_units() {
        let base = chain_data();
        let n = base.n_rows;
        let diff = |scale: f64| {
            let mut cols = base.columns.to_vec();
            for i in 0..n {
                cols[n + i] *= scale;
            }
            let data = GraphScoreData::new(n, 3, Arc::from(cols)).unwrap();
            let edge = gaussian_bic_local(&data, 1, &[0u32]).unwrap();
            let empty = gaussian_bic_local(&data, 1, &[]).unwrap();
            edge - empty
        };
        let d1 = diff(1.0);
        for scale in [1e-8, 1e8] {
            let ds = diff(scale);
            assert!(
                (ds - d1).abs() < 1e-6,
                "score(X→Y)-score(∅) changed under Y*={scale}: {d1} vs {ds}"
            );
        }
    }

    #[test]
    fn gaussian_bic_score_difference_is_invariant_to_predictor_units() {
        let base = chain_data();
        let n = base.n_rows;
        let diff = |scale: f64| {
            let mut cols = base.columns.to_vec();
            for i in 0..n {
                cols[i] *= scale;
            }
            let data = GraphScoreData::new(n, 3, Arc::from(cols)).unwrap();
            let edge = gaussian_bic_local(&data, 1, &[0u32]).unwrap();
            let empty = gaussian_bic_local(&data, 1, &[]).unwrap();
            edge - empty
        };
        let d1 = diff(1.0);
        for scale in [1e-8, 1e8] {
            let ds = diff(scale);
            assert!(
                (ds - d1).abs() < 1e-4,
                "score(X→Y)-score(∅) changed under X*={scale}: {d1} vs {ds}"
            );
        }
    }
}
