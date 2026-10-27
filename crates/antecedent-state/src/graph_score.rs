//! Incremental graph-score caches.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;
use std::sync::Arc;

use antecedent_core::Fnv1a64;
use antecedent_kernels::{chol_solve_into, cholesky_spd_into, equilibrate_into, symmetric_scales};

use crate::error::StateError;

/// Score family for local graph scores.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash)]
#[non_exhaustive]
pub enum GraphScoreFamily {
    /// Gaussian BIC local score (intercept + linear parents, σ² MLE = SSE/n).
    GaussianBic,
    /// Laplace-noise BIC local score: intercept + linear parents fitted by least absolute
    /// deviations, scale `b` the mean absolute residual, `k = |Pa| + 2` parameters. Unlike the
    /// Gaussian score it is not invariant within a Markov equivalence class when the noise is
    /// not Gaussian, so a posterior under it can orient edges the Gaussian posterior cannot.
    LaplaceBic,
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
    columns: Arc<[f64]>,
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
    let mut h = Fnv1a64::new();
    h.word(n_rows as u64);
    h.word(n_vars as u64);
    for v in columns {
        h.word(v.to_bits());
    }
    h.finish()
}

/// Local-score cache keyed by `(node, sorted parent set)`.
///
/// The cache binds to the first [`GraphScoreData`] it scores (by content fingerprint) and
/// refuses any other data until [`Self::clear`]: a `(node, parents)` key says nothing about
/// which rows the score was computed on.
#[derive(Clone, Debug)]
pub struct LocalScoreCache {
    /// Cache identity.
    key: GraphScoreCacheKey,
    /// Node → (parent-set key → local score).
    entries: HashMap<u32, HashMap<Arc<[u32]>, f64>>,
    /// Current parent sets per node (graph state).
    parents: HashMap<u32, Arc<[u32]>>,
    /// Fingerprint of the data the cached scores belong to.
    bound_data: Option<u64>,
    /// Gram of the mean-centered columns, built once for the bound data.
    gram: Option<Vec<f64>>,
    /// Sum of local scores under the current parent sets, maintained incrementally.
    running_total: Option<f64>,
    /// Approximate retained bytes.
    bytes: u64,
    /// Bytes the cache may retain; once exceeded the score entries (recomputable on demand)
    /// are dropped. The Gram is retained for the life of the bound data and is never evicted.
    byte_budget: u64,
}

/// Default retained-bytes budget of a [`LocalScoreCache`].
pub const DEFAULT_LOCAL_SCORE_CACHE_BYTES: u64 = 1 << 30;

impl LocalScoreCache {
    /// Empty cache for `key`.
    #[must_use]
    pub fn new(key: GraphScoreCacheKey) -> Self {
        Self {
            key,
            entries: HashMap::new(),
            parents: HashMap::new(),
            bound_data: None,
            gram: None,
            running_total: None,
            bytes: 0,
            byte_budget: DEFAULT_LOCAL_SCORE_CACHE_BYTES,
        }
    }

    /// Replace the retained-bytes budget (default [`DEFAULT_LOCAL_SCORE_CACHE_BYTES`]).
    ///
    /// When caching a further score would exceed the budget the cached scores are dropped
    /// and recomputed on demand; results are unchanged, only the memory held is bounded.
    #[must_use]
    pub fn with_byte_budget(mut self, bytes: u64) -> Self {
        self.byte_budget = bytes;
        self
    }

    /// Approximate bytes currently retained.
    #[must_use]
    pub const fn retained_bytes(&self) -> u64 {
        self.bytes
    }

    /// Clear all cached local scores and parent assignments, and unbind from the data.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.parents.clear();
        self.bound_data = None;
        self.gram = None;
        self.running_total = None;
        self.bytes = 0;
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
        #[allow(
            clippy::cast_possible_truncation,
            reason = "n_vars counts graph nodes, which are addressed by u32 node ids"
        )]
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
            GraphScoreFamily::GaussianBic => {
                self.ensure_gram(data);
                let Some(gram) = self.gram.as_deref() else {
                    return Err(StateError::Numerical("Gaussian BIC Gram was not stored".into()));
                };
                gaussian_bic_from_gram(data, gram, node, parents)?
            }
            GraphScoreFamily::LaplaceBic => laplace_bic_local(data, node, parents)?,
        };
        let cost = 32 + 8 * parents.len() as u64;
        if self.bytes.saturating_add(cost) > self.byte_budget {
            self.entries.clear();
            self.bytes =
                self.gram.as_ref().map_or(0, |g| (g.len() * core::mem::size_of::<f64>()) as u64);
        }
        self.entries.entry(node).or_default().insert(Arc::clone(parents), s);
        self.bytes = self.bytes.saturating_add(cost);
        Ok(s)
    }

    fn ensure_gram(&mut self, data: &GraphScoreData) {
        if self.gram.is_some() {
            return;
        }
        let gram = gaussian_design_gram(data);
        self.bytes = self.bytes.saturating_add((gram.len() * core::mem::size_of::<f64>()) as u64);
        self.gram = Some(gram);
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
#[cfg(test)]
fn gaussian_bic_local(
    data: &GraphScoreData,
    node: u32,
    parents: &[u32],
) -> Result<f64, StateError> {
    gaussian_bic_from_gram(data, &gaussian_design_gram(data), node, parents)
}

/// Neumaier-compensated sum: the rounding error of every partial sum is carried in a
/// separate term, so long sums of mixed magnitudes keep their low-order digits.
fn compensated_sum(values: impl Iterator<Item = f64>) -> f64 {
    let mut sum = 0.0_f64;
    let mut carry = 0.0_f64;
    for v in values {
        let t = sum + v;
        if sum.abs() >= v.abs() {
            carry += (sum - t) + v;
        } else {
            carry += (v - t) + sum;
        }
        sum = t;
    }
    sum + carry
}

/// Column `values` minus its compensated mean, with a second pass removing the rounding
/// residue left in the mean of the first centering.
#[allow(clippy::cast_precision_loss, reason = "row counts enter a mean as floats")]
fn centered_column(values: &[f64]) -> Vec<f64> {
    let n = values.len() as f64;
    let mean = compensated_sum(values.iter().copied()) / n;
    let mut centered: Vec<f64> = values.iter().map(|v| v - mean).collect();
    let residue = compensated_sum(centered.iter().copied()) / n;
    if residue.is_finite() && residue.abs() > 0.0 {
        for v in &mut centered {
            *v -= residue;
        }
    }
    centered
}

/// Row-major `n_vars × n_vars` Gram of the mean-centered columns, accumulated with
/// compensated sums.
///
/// The Gaussian BIC regresses a node on an intercept and its parents; the residual sum of
/// squares of that fit is the Schur complement of the *centered* cross-product matrix, which
/// is offset invariant. Forming it from the uncentered Gram of `[1 | X | y]` would subtract
/// two quantities of size `n · offset²` and lose every digit when the data sit far from zero.
#[allow(
    clippy::many_single_char_names,
    reason = "j and k index the two columns of the cross-product block"
)]
fn gaussian_design_gram(data: &GraphScoreData) -> Vec<f64> {
    let m = data.n_vars;
    let centered: Vec<Vec<f64>> = (0..m).map(|j| centered_column(data.col(j))).collect();
    let mut g = vec![0.0; m * m];
    for j in 0..m {
        for k in j..m {
            let cross = compensated_sum(centered[j].iter().zip(&centered[k]).map(|(a, b)| a * b));
            g[j * m + k] = cross;
            g[k * m + j] = cross;
        }
    }
    g
}

fn gaussian_bic_from_gram(
    data: &GraphScoreData,
    gram: &[f64],
    node: u32,
    parents: &[u32],
) -> Result<f64, StateError> {
    let n = data.n_rows;
    if n < 2 {
        return Err(StateError::Numerical("need n≥2 for BIC".into()));
    }
    if node as usize >= data.n_vars {
        return Err(StateError::Shape(format!("node {node} out of range")));
    }
    let m = data.n_vars;
    let mut cols = Vec::with_capacity(parents.len() + 1);
    for &p in parents {
        if p as usize >= data.n_vars {
            return Err(StateError::Shape(format!("parent {p} out of range")));
        }
        cols.push(p as usize);
    }
    cols.push(node as usize);
    bic_from_rss(rss_from_joint_gram(gram, m, &cols)?, n, parents.len() + 1)
}

/// Upper bound on iteratively reweighted least squares sweeps of the least-absolute-deviations
/// fit; the iteration stops earlier as soon as the objective stops improving.
const LAD_MAX_SWEEPS: usize = 500;

/// Relative change of the absolute-deviation objective below which the fit has converged.
const LAD_OBJECTIVE_TOLERANCE: f64 = 1e-12;

/// Residual floor, relative to the outcome's scale, bounding the reweighting `1 / |r|`.
const LAD_RESIDUAL_FLOOR: f64 = 1e-8;

/// Solve the symmetric positive-definite system `a x = b` (`k × k`, row-major) through the
/// shared equilibrated Cholesky; `None` when it is singular to working precision.
fn solve_spd(a: &[f64], b: &[f64], k: usize) -> Option<Vec<f64>> {
    let scales = symmetric_scales(a, k);
    if scales.len() != k {
        return None;
    }
    let mut eq = vec![0.0; k * k];
    equilibrate_into(a, &scales, k, &mut eq);
    let mut factor = vec![0.0; k * k];
    cholesky_spd_into(&eq, k, &mut factor).ok()?;
    // A = S E S with E = L Lᵀ, so A x = b  ⇔  E (S x) = S⁻¹ b.
    let scaled: Vec<f64> = b.iter().zip(&scales).map(|(v, s)| v / s).collect();
    let mut work = vec![0.0; k];
    let mut y = vec![0.0; k];
    chol_solve_into(&factor, k, &scaled, &mut work, &mut y);
    let x: Vec<f64> = y.iter().zip(&scales).map(|(v, s)| v / s).collect();
    x.iter().all(|v| v.is_finite()).then_some(x)
}

/// Laplace-noise BIC local score (higher is better):
/// `-n (ln(2 b) + 1) - (k / 2) ln(n)` with `b` the mean absolute residual of the least
/// absolute deviations fit of the node on intercept + parents and `k = |Pa| + 2`.
///
/// The fit is iteratively reweighted least squares on mean-centered columns (the residuals
/// are unchanged by centering when an intercept is fitted). The reweighting floor scales with
/// the outcome, the weighted normal equations go through the shared equilibrated Cholesky,
/// and the iteration stops once the objective stops improving.
#[allow(
    clippy::cast_precision_loss,
    clippy::needless_range_loop,
    clippy::many_single_char_names,
    reason = "row counts enter a likelihood as floats; the loops index the design by row and column"
)]
fn laplace_bic_local(data: &GraphScoreData, node: u32, parents: &[u32]) -> Result<f64, StateError> {
    let n = data.n_rows;
    if n < 2 {
        return Err(StateError::Numerical("need n≥2 for BIC".into()));
    }
    if node as usize >= data.n_vars {
        return Err(StateError::Shape(format!("node {node} out of range")));
    }
    if parents.iter().any(|&p| p as usize >= data.n_vars) {
        return Err(StateError::Shape("parent out of range".into()));
    }
    let k = parents.len() + 1;
    let y = centered_column(data.col(node as usize));
    let outcome_scale = compensated_sum(y.iter().copied().map(f64::abs)) / n as f64;
    if !(outcome_scale.is_finite() && outcome_scale > 0.0) {
        return Err(StateError::DegenerateScore("Laplace BIC fit (zero residual scale)".into()));
    }
    let floor = LAD_RESIDUAL_FLOOR * outcome_scale;
    // Design columns: the intercept, then each centered parent.
    let mut design: Vec<Vec<f64>> = Vec::with_capacity(k);
    design.push(vec![1.0; n]);
    for &p in parents {
        design.push(centered_column(data.col(p as usize)));
    }
    let mut weights = vec![1.0; n];
    let mut residuals = vec![0.0; n];
    let mut previous = f64::INFINITY;
    let mut objective = f64::INFINITY;
    for _ in 0..LAD_MAX_SWEEPS {
        let mut gram = vec![0.0; k * k];
        let mut rhs = vec![0.0; k];
        for i in 0..k {
            rhs[i] = compensated_sum((0..n).map(|r| weights[r] * design[i][r] * y[r]));
            for j in i..k {
                let v = compensated_sum((0..n).map(|r| weights[r] * design[i][r] * design[j][r]));
                gram[i * k + j] = v;
                gram[j * k + i] = v;
            }
        }
        let Some(next) = solve_spd(&gram, &rhs, k) else {
            return Err(StateError::Numerical(
                "Laplace BIC local fit: rank-deficient parents".into(),
            ));
        };
        let beta = next;
        for row in 0..n {
            let fitted: f64 = (0..k).map(|c| beta[c] * design[c][row]).sum();
            residuals[row] = y[row] - fitted;
            weights[row] = 1.0 / residuals[row].abs().max(floor);
        }
        objective = compensated_sum(residuals.iter().copied().map(f64::abs));
        if !objective.is_finite() {
            return Err(StateError::Numerical("non-finite Laplace BIC residuals".into()));
        }
        // IRLS minimizes the L1 objective monotonically up to rounding; stop on stagnation.
        if previous.is_finite()
            && (previous - objective).abs() <= LAD_OBJECTIVE_TOLERANCE * previous
        {
            break;
        }
        previous = objective;
    }
    let scale = objective / n as f64;
    if !(scale.is_finite() && scale > 1e-12 * outcome_scale) {
        return Err(StateError::DegenerateScore("Laplace BIC fit (zero residual scale)".into()));
    }
    let nf = n as f64;
    Ok(-nf * ((2.0 * scale).ln() + 1.0) - 0.5 * (k as f64 + 1.0) * nf.ln())
}

fn rss_from_joint_gram(gram: &[f64], m: usize, cols: &[usize]) -> Result<f64, StateError> {
    let q = cols.len();
    if q < 1 || gram.len() < m.saturating_mul(m) {
        return Err(StateError::Shape("Gaussian BIC Gram is shorter than the design".into()));
    }
    let mut sub = vec![0.0; q * q];
    for (i, &ci) in cols.iter().enumerate() {
        for (j, &cj) in cols.iter().enumerate() {
            sub[i * q + j] = gram[ci * m + cj];
        }
    }
    let p = q - 1;
    if p > 0 && !equilibrated_spd(&sub, q, p) {
        return Err(StateError::Numerical("Gaussian BIC local fit: rank-deficient parents".into()));
    }
    let scales = symmetric_scales(&sub, q);
    if scales.len() != q {
        return Err(StateError::Numerical("Gaussian BIC equilibration failed".into()));
    }
    let mut eq = vec![0.0; q * q];
    equilibrate_into(&sub, &scales, q, &mut eq);
    let mut factor = vec![0.0; q * q];
    if cholesky_spd_into(&eq, q, &mut factor).is_err() {
        return Err(StateError::DegenerateScore(
            "Gaussian BIC fit (zero residual variance)".into(),
        ));
    }
    let pivot = factor[(q - 1) * q + (q - 1)];
    let d_y = scales[q - 1];
    Ok(pivot * pivot * d_y * d_y)
}

fn equilibrated_spd(joint: &[f64], q: usize, p: usize) -> bool {
    let mut parent = vec![0.0; p * p];
    for i in 0..p {
        for j in 0..p {
            parent[i * p + j] = joint[i * q + j];
        }
    }
    let scales = symmetric_scales(&parent, p);
    if scales.len() != p {
        return false;
    }
    let mut eq = vec![0.0; p * p];
    equilibrate_into(&parent, &scales, p, &mut eq);
    let mut factor = vec![0.0; p * p];
    cholesky_spd_into(&eq, p, &mut factor).is_ok()
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
        return Err(StateError::DegenerateScore(
            "Gaussian BIC fit (zero residual variance)".into(),
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
#[allow(
    clippy::implicit_hasher,
    reason = "concrete map keyed by the default hasher; these caches are never built with a custom one"
)]
pub fn full_graph_score(
    data: &GraphScoreData,
    family: GraphScoreFamily,
    parents: &HashMap<u32, Arc<[u32]>>,
) -> Result<f64, StateError> {
    let gram = match family {
        GraphScoreFamily::GaussianBic => gaussian_design_gram(data),
        GraphScoreFamily::LaplaceBic => Vec::new(),
    };
    let mut total = 0.0;
    #[allow(
        clippy::cast_possible_truncation,
        reason = "n_vars counts graph nodes, which are addressed by u32 node ids"
    )]
    for node in 0..data.n_vars as u32 {
        let pa = parents.get(&node).cloned().unwrap_or_else(|| Arc::from([]));
        total += match family {
            GraphScoreFamily::GaussianBic => gaussian_bic_from_gram(data, &gram, node, &pa)?,
            GraphScoreFamily::LaplaceBic => laplace_bic_local(data, node, &pa)?,
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

    /// Deterministic Laplace draws by the inverse CDF of a splitmix64 stream.
    fn laplace_stream(n: usize, seed: u64) -> Vec<f64> {
        let mut state = seed;
        (0..n)
            .map(|_| {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let u = (z >> 11) as f64 / (1u64 << 53) as f64 - 0.5;
                -u.signum() * (1.0 - 2.0 * u.abs()).max(1e-12).ln()
            })
            .collect()
    }

    #[test]
    fn the_laplace_score_prefers_the_causal_direction_under_laplace_noise() {
        let n = 3000;
        let x = laplace_stream(n, 11);
        let e = laplace_stream(n, 29);
        let y: Vec<f64> = x.iter().zip(&e).map(|(x, e)| 1.5 * x + e).collect();
        let mut cols = x;
        cols.extend(y);
        let data = GraphScoreData::new(n, 2, Arc::from(cols)).unwrap();
        let score = |node: u32, parents: &[u32]| laplace_bic_local(&data, node, parents).unwrap();
        let causal = score(0, &[]) + score(1, &[0]);
        let anticausal = score(1, &[]) + score(0, &[1]);
        assert!(causal > anticausal, "causal {causal} anticausal {anticausal}");
        // The Gaussian score cannot separate the two directions.
        let gaussian =
            |node: u32, parents: &[u32]| gaussian_bic_local(&data, node, parents).unwrap();
        let gap = (gaussian(0, &[]) + gaussian(1, &[0])) - (gaussian(1, &[]) + gaussian(0, &[1]));
        assert!(gap.abs() < 1e-6, "gaussian gap {gap}");
        // A perfect fit has no residual scale and is refused, not scored.
        let twin: Vec<f64> = data.col(0).iter().chain(data.col(0)).copied().collect();
        let twin = GraphScoreData::new(n, 2, Arc::from(twin)).unwrap();
        assert!(laplace_bic_local(&twin, 1, &[0]).is_err());
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
    #[allow(
        clippy::float_cmp,
        reason = "a cache hit must return the stored score bit for bit, and a recompute on other data must differ exactly"
    )]
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
        assert!(matches!(err, StateError::DegenerateScore(_)), "{err:?}");
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
    #[allow(
        clippy::needless_range_loop,
        reason = "index-based loop over parallel arrays; the subscript also names the matrix coordinate"
    )]
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
    /// A large common offset must not erase the residual: the score difference of an edge is
    /// the same with and without a 1e6 shift of every column.
    #[test]
    fn gaussian_bic_is_offset_invariant() {
        let base = chain_data();
        let n = base.n_rows;
        let shifted_cols: Vec<f64> = base.columns.iter().map(|v| v + 1.0e6).collect();
        let shifted = GraphScoreData::new(n, 3, Arc::from(shifted_cols)).unwrap();
        for (node, parents) in [(1u32, vec![0u32]), (2, vec![1]), (1, vec![])] {
            let a = gaussian_bic_local(&base, node, &parents).unwrap();
            let b = gaussian_bic_local(&shifted, node, &parents).unwrap();
            assert!((a - b).abs() < 1e-5, "offset changed the score: {a} vs {b}");
        }
    }

    /// The Laplace fit is offset invariant as well and still refuses a perfect fit.
    #[test]
    fn laplace_bic_is_offset_invariant() {
        let n = 400;
        let x = laplace_stream(n, 5);
        let e = laplace_stream(n, 6);
        let y: Vec<f64> = x.iter().zip(&e).map(|(x, e)| 0.5 * x + e).collect();
        let build = |offset: f64| {
            let mut cols: Vec<f64> = x.iter().map(|v| v + offset).collect();
            cols.extend(y.iter().map(|v| v + offset));
            GraphScoreData::new(n, 2, Arc::from(cols)).unwrap()
        };
        let a = laplace_bic_local(&build(0.0), 1, &[0]).unwrap();
        let b = laplace_bic_local(&build(1.0e6), 1, &[0]).unwrap();
        assert!((a - b).abs() < 1e-3, "offset changed the Laplace score: {a} vs {b}");
    }

    /// A tiny byte budget evicts cached scores without changing any returned score.
    #[test]
    fn cache_budget_bounds_retained_entries() {
        let data = chain_data();
        let mut cache = fresh_cache(&data).with_byte_budget(1);
        let mut reference = fresh_cache(&data);
        for parents in [vec![], vec![0u32], vec![0, 1]] {
            let pa: Arc<[u32]> = Arc::from(parents);
            let got = cache.local_score(&data, 2, &pa).unwrap();
            let expect = reference.local_score(&data, 2, &pa).unwrap();
            assert!((got - expect).abs() < 1e-12);
        }
        let entries: usize = cache.entries.values().map(HashMap::len).sum();
        assert!(entries <= 1, "budget kept {entries} entries");
    }
}
