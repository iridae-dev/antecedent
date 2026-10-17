//! Shared DAG scoring for Bayesian discovery (exact + MCMC).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::VariableId;
use antecedent_data::TabularData;
use antecedent_state::{GraphScoreCacheKey, GraphScoreData, GraphScoreFamily, LocalScoreCache};

use crate::adjacency_mask::AdjacencyMask;
use crate::adjacency_mask::{fill_parents, parent_bits, set_edge};
use crate::error::DiscoveryError;
use crate::graph_posterior::{GraphPrior, GraphPriorTable, edge_required};
use crate::pc::collect_float_columns;

/// Version stamped on every discovery [`GraphScoreCacheKey`].
///
/// These searches score the columns they were handed, not a data-catalog snapshot.
/// One literal keeps the copied `1`s from drifting apart and splitting the cache.
pub(crate) const GRAPH_SCORE_DATA_VERSION: u64 = 1;

/// Cache key for a local-score table over `n_vars` columns and `n_rows` rows.
///
/// Fingerprints use the same `u64` conversion every caller used (`n as u64` where `n`
/// fits, which every `usize` does).
pub(crate) fn graph_score_cache_key(
    family: GraphScoreFamily,
    n_vars: usize,
    n_rows: usize,
) -> GraphScoreCacheKey {
    GraphScoreCacheKey {
        data_version: GRAPH_SCORE_DATA_VERSION,
        family,
        var_fingerprint: u64::try_from(n_vars).unwrap_or(u64::MAX),
        penalty_fingerprint: u64::try_from(n_rows).unwrap_or(u64::MAX),
    }
}

/// Build column-major [`GraphScoreData`] from tabular float columns.
pub(crate) fn tabular_score_data(
    data: &TabularData,
    variables: &[VariableId],
) -> Result<GraphScoreData, DiscoveryError> {
    let col_owned = collect_float_columns(data, variables)?;
    let n_rows = col_owned[0].len();
    if n_rows < 2 {
        return Err(DiscoveryError::stats_msg("insufficient rows for graph score"));
    }
    for c in &col_owned {
        if c.len() != n_rows {
            return Err(DiscoveryError::data_msg("column length mismatch"));
        }
    }
    let n_vars = variables.len();
    let mut flat = Vec::with_capacity(n_vars.saturating_mul(n_rows));
    for c in &col_owned {
        flat.extend_from_slice(c.as_ref());
    }
    Ok(GraphScoreData::new(n_rows, n_vars, Arc::from(flat))?)
}

/// Score one DAG: BIC + log prior (`None` if invalid).
///
/// Uses [`LocalScoreCache::local_score`] only (does not mutate cached parent sets),
/// so MCMC proposals can reject without restoring graph state.
pub(crate) fn score_dag_mask(
    mask: AdjacencyMask,
    n: usize,
    data: &GraphScoreData,
    cache: &mut LocalScoreCache,
    prior: &GraphPrior,
    variables: &[VariableId],
) -> Option<f64> {
    let mut parents = Vec::new();
    let table = GraphPriorTable::new(n, prior, variables);
    score_dag_mask_with_parents(mask, n, data, cache, &table, &mut parents)
}

/// [`score_dag_mask`] against a prior resolved once, reusing `parents` for every node's
/// parent list.
///
/// A sampler scores one mask per step; resolving the constraints per step would dominate the
/// run on a dozen variables. The stored local score is still keyed by a fresh `Arc` of those
/// indexes, in the same increasing order [`crate::graph_posterior::parents_of`] produces.
pub(crate) fn score_dag_mask_with_parents(
    mask: AdjacencyMask,
    n: usize,
    data: &GraphScoreData,
    cache: &mut LocalScoreCache,
    prior: &GraphPriorTable,
    parents: &mut Vec<u32>,
) -> Option<f64> {
    let lp = prior.log_prior(mask)?;
    let mut total = lp;
    for node in 0..n {
        fill_parents(mask, n, node, parents);
        let pa = Arc::<[u32]>::from(parents.as_slice());
        let s = cache.local_score(data, crate::indexing::dense_u32(node), &pa).ok()?;
        if !s.is_finite() {
            return None;
        }
        total += s;
    }
    Some(total)
}

/// The log target of one sampler chain: BIC + log prior of a graph, and of a single node's
/// parent set.
///
/// A sampler asks for the whole graph once per proposal, and for single nodes when it resamples
/// parent sets exactly. The target is a trait so a test can run the same kernels on a target
/// with the data removed.
pub(crate) trait ChainTarget {
    /// Log target of `mask`; `None` when it is invalid (constraints, cycle, no finite score).
    fn score(&mut self, mask: AdjacencyMask) -> Option<f64>;
    /// Log target of `node` having parent set `parents`; `None` when that is invalid. The sum
    /// over a valid graph's nodes is [`Self::score`].
    fn node_score(&mut self, node: usize, parents: u16) -> Option<f64>;
}

/// [`ChainTarget`] scoring Gaussian BIC + prior, with its own local-score cache and parent-list
/// scratch.
pub(crate) struct ScoredTarget<'a> {
    n: usize,
    data: &'a GraphScoreData,
    cache: LocalScoreCache,
    prior: &'a GraphPriorTable,
    parents: Vec<u32>,
}

impl<'a> ScoredTarget<'a> {
    pub(crate) fn new(
        n: usize,
        data: &'a GraphScoreData,
        family: GraphScoreFamily,
        prior: &'a GraphPriorTable,
    ) -> Self {
        Self {
            n,
            data,
            cache: LocalScoreCache::new(graph_score_cache_key(family, n, data.n_rows)),
            prior,
            parents: Vec::with_capacity(n),
        }
    }
}

impl ChainTarget for ScoredTarget<'_> {
    fn score(&mut self, mask: AdjacencyMask) -> Option<f64> {
        score_dag_mask_with_parents(
            mask,
            self.n,
            self.data,
            &mut self.cache,
            self.prior,
            &mut self.parents,
        )
    }

    fn node_score(&mut self, node: usize, parents: u16) -> Option<f64> {
        let prior = self.prior.node_log_prior(node, parents)?;
        let set: Arc<[u32]> = (0..self.n)
            .filter(|&p| (parents >> p) & 1 == 1)
            .map(crate::indexing::dense_u32)
            .collect();
        let local =
            self.cache.local_score(self.data, crate::indexing::dense_u32(node), &set).ok()?;
        local.is_finite().then_some(prior + local)
    }
}

/// A [`ChainTarget`] several replicas of one chain share, so they share its score cache.
pub(crate) struct SharedTarget<T>(pub(crate) std::rc::Rc<std::cell::RefCell<T>>);

impl<T> Clone for SharedTarget<T> {
    fn clone(&self) -> Self {
        Self(std::rc::Rc::clone(&self.0))
    }
}

impl<T: ChainTarget> ChainTarget for SharedTarget<T> {
    fn score(&mut self, mask: AdjacencyMask) -> Option<f64> {
        self.0.borrow_mut().score(mask)
    }

    fn node_score(&mut self, node: usize, parents: u16) -> Option<f64> {
        self.0.borrow_mut().node_score(node, parents)
    }
}

/// Scores every DAG of an exhaustive enumeration from a table of local scores.
///
/// Over `n` labelled nodes there are only `n * 2^(n-1)` distinct `(node, parent set)` local
/// scores, so each is computed once up front and a mask is then scored by summation, in node
/// order, exactly as [`score_dag_mask`] does.
pub(crate) struct DagScorer {
    n: usize,
    prior: GraphPriorTable,
    /// `local[node << n | parents]`: `None` when the score is not finite or not computable.
    local: Vec<Option<f64>>,
}

impl DagScorer {
    pub(crate) fn new(
        n: usize,
        data: &GraphScoreData,
        family: GraphScoreFamily,
        prior: &GraphPrior,
        variables: &[VariableId],
    ) -> Self {
        let mut cache = LocalScoreCache::new(graph_score_cache_key(family, n, data.n_rows));
        let mut local = vec![None; n << n];
        for node in 0..n {
            for parents in 0..(1usize << n) {
                if parents & (1 << node) != 0 {
                    continue;
                }
                let pa: Arc<[u32]> = (0..n)
                    .filter(|&p| parents & (1 << p) != 0)
                    .map(|p| u32::try_from(p).unwrap_or(u32::MAX))
                    .collect();
                local[(node << n) | parents] = u32::try_from(node)
                    .ok()
                    .and_then(|node| cache.local_score(data, node, &pa).ok())
                    .filter(|s| s.is_finite());
            }
        }
        Self { n, prior: GraphPriorTable::new(n, prior, variables), local }
    }

    /// BIC + log prior of `mask` (`None` if invalid), bit-identical to [`score_dag_mask`].
    pub(crate) fn score(&self, mask: AdjacencyMask) -> Option<f64> {
        let n = self.n;
        let mut total = self.prior.log_prior(mask)?;
        for node in 0..n {
            let parents = usize::from(parent_bits(mask, n, node));
            total += self.local[(node << n) | parents]?;
        }
        Some(total)
    }
}

/// Start mask samplers inside the constrained support rather than hoping that
/// one-edge proposals can cross a region of zero prior probability.
pub(crate) fn initial_scored_mask(
    n: usize,
    data: &GraphScoreData,
    family: GraphScoreFamily,
    prior: &GraphPrior,
    variables: &[VariableId],
) -> Result<(AdjacencyMask, f64), DiscoveryError> {
    let mut mask = AdjacencyMask::EMPTY;
    for i in 0..n {
        for j in 0..n {
            if i != j && edge_required(&prior.constraints, variables, i, j) {
                mask = set_edge(mask, n, i, j, true);
            }
        }
    }
    let mut cache = LocalScoreCache::new(graph_score_cache_key(family, n, data.n_rows));
    let score = score_dag_mask(mask, n, data, &mut cache, prior, variables)
        .filter(|s| s.is_finite())
        .ok_or_else(|| {
            DiscoveryError::unsupported(
                "required-edge graph is cyclic, violates constraints, or has no finite score",
            )
        })?;
    Ok((mask, score))
}
