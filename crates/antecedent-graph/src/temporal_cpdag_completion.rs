//! Streamed [`TemporalCpdag`] → [`TemporalDag`] MEC completions.
//!
//! Same orientation rule as [`crate::cpdag_completion`]: undirected edges are
//! oriented without new unshielded colliders. Conflict marks are refused.
//! Future→past orientations are rejected by [`TemporalCpdag::orient_undirected`].
//!
//! Search orients one undirected edge at a time and rejects a partial assignment
//! as soon as it creates a cycle, a future→past edge, or an unshielded collider
//! absent from the CPDAG. A hard ceiling ([`MAX_TEMPORAL_UNDIRECTED_EDGES`])
//! still refuses pathological unconstrained instances. The search itself is shared
//! with the static sampler (`crate::mec_search`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::cpdag::TemporalCpdag;
use crate::error::GraphError;
use crate::mec_search::{self, MecSearch};
use crate::temporal::TemporalDag;
#[cfg(test)]
use crate::types::DenseNodeId;

/// Hard ceiling on undirected edges accepted by the temporal completion search.
///
/// Matches [`crate::cpdag_completion::MAX_UNDIRECTED_EDGES`]: above the ~25-edge
/// hang threshold of a blind mask scan; pruning handles sparse classes below it.
pub const MAX_TEMPORAL_UNDIRECTED_EDGES: usize = 64;

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
    /// # Errors
    ///
    /// Conflict edges, or more than [`MAX_TEMPORAL_UNDIRECTED_EDGES`] undirected edges.
    pub fn new(cpdag: TemporalCpdag, max_completions: usize) -> Result<Self, GraphError> {
        let search = MecSearch::new(
            cpdag,
            max_completions,
            MAX_TEMPORAL_UNDIRECTED_EDGES,
            "TemporalCpdagCompletionSampler refuses conflict (x-x) edges",
            "TemporalCpdagCompletionSampler supports at most 64 undirected edges",
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

    /// Whether the retention cap stopped the stream before the search finished.
    #[must_use]
    pub fn hit_cap(&self) -> bool {
        self.search.hit_cap()
    }
}

/// Whether `dag` is a Markov-equivalence member of `cpdag`.
#[must_use]
pub fn is_temporal_mec_member(cpdag: &TemporalCpdag, dag: &TemporalDag) -> bool {
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

    #[test]
    fn refuses_above_hard_undirected_ceiling() {
        let mut g = TemporalCpdag::empty();
        let n = MAX_TEMPORAL_UNDIRECTED_EDGES + 2;
        let mut nodes = Vec::with_capacity(n);
        for i in 0..n {
            nodes.push(lagged(&mut g, u32::try_from(i).unwrap(), 0));
        }
        for w in nodes.windows(2) {
            g.insert_undirected(w[0], w[1]).unwrap();
        }
        assert_eq!(g.undirected_edge_count(), MAX_TEMPORAL_UNDIRECTED_EDGES + 1);
        let err = TemporalCpdagCompletionSampler::new(g, 4).unwrap_err();
        assert!(matches!(err, GraphError::InvalidEndpoints { .. }));
    }
}
