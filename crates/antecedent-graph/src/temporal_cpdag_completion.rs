//! Streamed [`TemporalCpdag`] → [`TemporalDag`] MEC completions.
//!
//! Same orientation rule as [`crate::cpdag_completion`]: undirected edges are
//! oriented without new unshielded colliders. Conflict marks are refused.
//! Future→past orientations are rejected by [`TemporalCpdag::orient_undirected`].
//!
//! Search orients one undirected edge at a time and rejects a partial assignment
//! as soon as it creates a cycle, a future→past edge, or an unshielded collider
//! absent from the CPDAG. No edge-count ceiling applies: a search budget bounds the
//! visited states, and a stop is typed and counted by `hit_cap`. The search itself is
//! shared with the static sampler (`crate::mec_search`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, SearchLimits, SearchReceipt, SearchStop};

use crate::completion::default_completion_search;
use crate::cpdag::TemporalCpdag;
use crate::error::GraphError;
use crate::mec_search::{self, MecSearch};
use crate::temporal::TemporalDag;
#[cfg(test)]
use crate::types::DenseNodeId;

/// One [`TemporalDag`] completion of a [`TemporalCpdag`].
#[derive(Clone, Debug)]
pub struct TemporalCpdagCompletion {
    /// Completed temporal DAG.
    pub graph: TemporalDag,
    /// Index among valid yields.
    pub index: usize,
}

/// Streams [`TemporalCpdag`] → [`TemporalDag`] completions with a hard cap.
#[derive(Clone, Debug)]
pub struct TemporalCpdagCompletionSampler {
    search: MecSearch<TemporalCpdag>,
}

impl TemporalCpdagCompletionSampler {
    /// Build a sampler that yields at most `max_completions` valid MEC [`TemporalDag`]s.
    ///
    /// The visited states are bounded by [`crate::DEFAULT_COMPLETION_OPERATIONS`]; use
    /// [`Self::with_budget`] to choose the limits, memory cap and cancellation.
    ///
    /// # Errors
    ///
    /// Conflict edges present.
    pub fn new(cpdag: TemporalCpdag, max_completions: usize) -> Result<Self, GraphError> {
        let (limits, ctx) = default_completion_search();
        Self::with_budget(cpdag, max_completions, limits, &ctx)
    }

    /// [`Self::new`] under an explicit search budget; see
    /// [`CpdagCompletionSampler::with_budget`](crate::CpdagCompletionSampler::with_budget).
    ///
    /// # Errors
    ///
    /// Conflict edges present.
    pub fn with_budget(
        cpdag: TemporalCpdag,
        max_completions: usize,
        limits: SearchLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, GraphError> {
        let search = MecSearch::new(
            cpdag,
            max_completions,
            limits,
            ctx,
            "TemporalCpdagCompletionSampler refuses conflict (x-x) edges",
        )?;
        Ok(Self { search })
    }

    /// Hard cap on yielded valid completions.
    #[must_use]
    pub fn max_completions(&self) -> usize {
        self.search.max_completions
    }

    /// Number of undirected edges being oriented.
    #[must_use]
    pub fn n_undirected(&self) -> usize {
        self.search.n_undirected()
    }

    /// Whether the stream ended before the search finished, because the retention cap was
    /// reached or the search budget stopped it ([`Self::search_stop`]).
    #[must_use]
    pub fn hit_cap(&self) -> bool {
        self.search.hit_cap()
    }

    /// The bound that stopped the search, if a budget rather than the retention cap ended
    /// the stream.
    #[must_use]
    pub fn search_stop(&self) -> Option<SearchStop> {
        self.search.search_stop()
    }

    /// The limits in force, the states visited and the orientations left pending, once a
    /// budget has stopped the search.
    #[must_use]
    pub fn search_receipt(&self) -> Option<SearchReceipt> {
        self.search.search_receipt()
    }
}

/// Whether `dag` is a Markov-equivalence member of `cpdag`.
#[must_use]
pub(crate) fn is_temporal_mec_member(cpdag: &TemporalCpdag, dag: &TemporalDag) -> bool {
    mec_search::is_mec_member(cpdag, dag, cpdag.nodes() == dag.nodes())
}

impl Iterator for TemporalCpdagCompletionSampler {
    type Item = TemporalCpdagCompletion;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(oriented) = self.search.next_oriented() {
            let dag = oriented.try_into_temporal_dag().ok()?;
            if !is_temporal_mec_member(&self.search.base, &dag) {
                continue;
            }
            let index = self.search.record_yield();
            return Some(TemporalCpdagCompletion { graph: dag, index });
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use antecedent_core::{Lag, VariableId};

    use super::*;

    fn lagged(g: &mut TemporalCpdag, var: u32, lag: u32) -> DenseNodeId {
        g.add_lagged(VariableId::from_raw(var), Lag::from_raw(lag)).unwrap()
    }

    #[test]
    fn fully_oriented_yields_one() {
        let mut g = TemporalCpdag::empty();
        let a = lagged(&mut g, 0, 1);
        let b = lagged(&mut g, 1, 0);
        g.insert_directed(a, b).unwrap();
        let collected: Vec<_> = TemporalCpdagCompletionSampler::new(g, 8).unwrap().collect();
        assert_eq!(collected.len(), 1);
    }

    #[test]
    fn contemporaneous_undirected_has_two_mec_members() {
        let mut g = TemporalCpdag::empty();
        let z = lagged(&mut g, 0, 1);
        let t = lagged(&mut g, 1, 1);
        let y = lagged(&mut g, 2, 0);
        g.insert_directed(z, y).unwrap();
        g.insert_directed(t, y).unwrap();
        g.insert_undirected(z, t).unwrap();
        let collected: Vec<_> =
            TemporalCpdagCompletionSampler::new(g.clone(), 8).unwrap().collect();
        assert_eq!(collected.len(), 2);
        for completion in &collected {
            assert!(is_temporal_mec_member(&g, &completion.graph));
        }
    }

    #[test]
    fn refuses_conflict() {
        let mut g = TemporalCpdag::empty();
        let a = lagged(&mut g, 0, 1);
        let b = lagged(&mut g, 1, 0);
        g.insert_undirected(a, b).unwrap();
        g.mark_conflict(a, b).unwrap();
        assert!(TemporalCpdagCompletionSampler::new(g, 4).is_err());
    }

    #[test]
    fn long_undirected_chain_completes_by_pruning() {
        let mut g = TemporalCpdag::empty();
        let mut nodes = Vec::with_capacity(26);
        for i in 0..26u32 {
            nodes.push(lagged(&mut g, i, 0));
        }
        for w in nodes.windows(2) {
            g.insert_undirected(w[0], w[1]).unwrap();
        }
        assert_eq!(g.undirected_edge_count(), 25);
        let start = Instant::now();
        let collected: Vec<_> =
            TemporalCpdagCompletionSampler::new(g.clone(), 64).unwrap().collect();
        assert!(
            start.elapsed().as_secs_f64() < 1.0,
            "25-edge temporal chain must finish by pruning, took {:?}",
            start.elapsed()
        );
        assert_eq!(collected.len(), 26);
        for completion in &collected {
            assert!(is_temporal_mec_member(&g, &completion.graph));
        }
    }

    #[test]
    fn small_completable_temporal_cpdag_still_completes() {
        let mut g = TemporalCpdag::empty();
        let a = lagged(&mut g, 0, 0);
        let b = lagged(&mut g, 1, 0);
        g.insert_undirected(a, b).unwrap();
        let collected: Vec<_> =
            TemporalCpdagCompletionSampler::new(g.clone(), 8).unwrap().collect();
        assert_eq!(collected.len(), 2);
        for completion in &collected {
            assert!(is_temporal_mec_member(&g, &completion.graph));
        }
    }

    fn contemporaneous_path(n: u32) -> (TemporalCpdag, Vec<DenseNodeId>) {
        let mut g = TemporalCpdag::empty();
        let nodes: Vec<_> = (0..n).map(|i| lagged(&mut g, i, 0)).collect();
        for w in nodes.windows(2) {
            g.insert_undirected(w[0], w[1]).unwrap();
        }
        (g, nodes)
    }

    #[test]
    fn a_seventy_one_edge_path_completes_with_one_dag_per_source() {
        // An undirected path on n nodes has n members (one per choice of source).
        let n = 72u32;
        let (g, nodes) = contemporaneous_path(n);
        assert_eq!(g.undirected_edge_count(), 71);
        let mut sampler = TemporalCpdagCompletionSampler::new(g.clone(), 128).unwrap();
        let collected: Vec<_> = sampler.by_ref().collect();
        assert!(!sampler.hit_cap());
        assert_eq!(sampler.search_stop(), None);
        assert_eq!(collected.len(), n as usize);
        let mut sources = std::collections::BTreeSet::new();
        for completion in &collected {
            assert!(is_temporal_mec_member(&g, &completion.graph));
            let source: Vec<usize> = nodes
                .iter()
                .enumerate()
                .filter(|(_, v)| completion.graph.parents(**v).is_empty())
                .map(|(i, _)| i)
                .collect();
            assert_eq!(source.len(), 1);
            sources.insert(source[0]);
        }
        assert_eq!(sources.len(), n as usize);
    }

    #[test]
    fn an_exhausted_budget_or_cancellation_is_a_typed_partial_result() {
        // Six contemporaneous nodes, all pairs undirected: 6! members.
        let mut g = TemporalCpdag::empty();
        let nodes: Vec<_> = (0..6u32).map(|i| lagged(&mut g, i, 0)).collect();
        for (i, &a) in nodes.iter().enumerate() {
            for &b in &nodes[i + 1..] {
                g.insert_undirected(a, b).unwrap();
            }
        }
        let ctx = ExecutionContext::for_tests(0);
        let limits = SearchLimits { operations: 100, depth: usize::MAX };
        let mut sampler =
            TemporalCpdagCompletionSampler::with_budget(g.clone(), usize::MAX, limits, &ctx)
                .unwrap();
        let yielded = sampler.by_ref().count();
        assert!(yielded < 720);
        assert_eq!(sampler.search_stop(), Some(SearchStop::Operations));
        assert!(sampler.hit_cap());
        assert_eq!(sampler.search_receipt().unwrap().operations_consumed, Some(100));

        let ctx = ExecutionContext::for_tests(0);
        ctx.cancellation.cancel();
        let limits = SearchLimits { operations: 1_000_000, depth: usize::MAX };
        let mut sampler =
            TemporalCpdagCompletionSampler::with_budget(g, usize::MAX, limits, &ctx).unwrap();
        assert!(sampler.next().is_none());
        assert_eq!(sampler.search_stop(), Some(SearchStop::Cancelled));
    }
}
