//! Streamed / bounded CPDAG MEC completion sampling.
//!
//! Yields DAG members of the Markov equivalence class of a static [`Cpdag`]:
//! orientations of undirected (Tail–Tail) edges that remain acyclic and do not
//! introduce new unshielded colliders. Completions are never retained without
//! bound (`max_completions` caps **valid** yields). Conflict (`x-x`) edges are
//! refused at construction.
//!
//! Search orients one undirected edge at a time and rejects a partial assignment
//! as soon as it creates a directed cycle or an unshielded collider absent from
//! the CPDAG — so sparse classes (e.g. long undirected chains) finish by pruning
//! instead of scanning `2^k` full masks. A hard ceiling
//! ([`MAX_UNDIRECTED_EDGES`]) still refuses pathological unconstrained instances.
//! The search itself is shared with the temporal sampler (`crate::mec_search`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::cpdag::Cpdag;
use crate::dag::Dag;
use crate::error::GraphError;
use crate::mec_search::{self, MecSearch};
#[cfg(test)]
use crate::types::DenseNodeId;

/// Hard ceiling on undirected edges accepted by the completion search.
///
/// Partial-assignment pruning handles sparse classes (chains, trees) well above
/// the ~25-edge hang of a blind mask scan. This ceiling only stops fully
/// unconstrained `2^k` blow-ups; it sits above that practical hang threshold.
pub const MAX_UNDIRECTED_EDGES: usize = 64;

/// One DAG completion of a CPDAG (MEC member).
#[derive(Clone, Debug)]
pub struct CpdagCompletion {
    /// Completed DAG.
    pub graph: Dag,
    /// Index of this completion in the stream (0-based among valid yields).
    pub index: usize,
}

/// Streams CPDAG → DAG completions with a hard cap (no unbounded retain).
#[derive(Clone, Debug)]
pub struct CpdagCompletionSampler {
    search: MecSearch<Cpdag>,
}

impl CpdagCompletionSampler {
    /// Build a sampler that yields at most `max_completions` **valid** MEC DAGs.
    ///
    /// # Errors
    ///
    /// Conflict edges present, or more than [`MAX_UNDIRECTED_EDGES`] undirected edges.
    pub fn new(cpdag: Cpdag, max_completions: usize) -> Result<Self, GraphError> {
        let search = MecSearch::new(
            cpdag,
            max_completions,
            MAX_UNDIRECTED_EDGES,
            "CpdagCompletionSampler refuses conflict (x-x) edges",
            "CpdagCompletionSampler supports at most 64 undirected edges",
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
    /// Local validity still holds for yielded DAGs; the unexplored remainder is
    /// not certified empty of further MEC members.
    #[must_use]
    pub fn hit_cap(&self) -> bool {
        self.search.hit_cap()
    }
}

/// Whether `dag` is a Markov-equivalence member of `cpdag` (same skeleton, same
/// unshielded colliders, all compelled directed edges of the CPDAG present).
#[must_use]
pub fn is_mec_member(cpdag: &Cpdag, dag: &Dag) -> bool {
    mec_search::is_mec_member(cpdag, dag, cpdag.node_count() == dag.node_count())
}

impl Iterator for CpdagCompletionSampler {
    type Item = CpdagCompletion;

    fn next(&mut self) -> Option<Self::Item> {
        while let Some(oriented) = self.search.next_oriented() {
            let dag = oriented.try_into_dag().ok()?;
            if !is_mec_member(&self.search.base, &dag) {
                continue;
            }
            let index = self.search.record_yield();
            return Some(CpdagCompletion { graph: dag, index });
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use std::time::Instant;

    use super::*;
    use crate::cpdag::Cpdag;

    #[test]
    fn fully_oriented_yields_one() {
        let mut g = Cpdag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let collected: Vec<_> = CpdagCompletionSampler::new(g, 10).unwrap().collect();
        assert_eq!(collected.len(), 1);
        assert!(
            collected[0]
                .graph
                .children(DenseNodeId::from_raw(0))
                .contains(&DenseNodeId::from_raw(1))
        );
    }

    #[test]
    fn chain_undirected_has_three_mec_dags() {
        // A — B — C: MEC has A→B→C, A←B←C, and A←B→C — not A→B←C (new v-structure).
        let mut g = Cpdag::with_variables(3);
        let a = DenseNodeId::from_raw(0);
        let b = DenseNodeId::from_raw(1);
        let c = DenseNodeId::from_raw(2);
        g.insert_undirected(a, b).unwrap();
        g.insert_undirected(b, c).unwrap();
        let collected: Vec<_> = CpdagCompletionSampler::new(g.clone(), 16).unwrap().collect();
        assert_eq!(g.undirected_edge_count(), 2);
        assert_eq!(collected.len(), 3, "expected 3 MEC DAGs, got {}", collected.len());
        for c in &collected {
            assert!(is_mec_member(&g, &c.graph));
        }
    }

    #[test]
    fn respects_max_completions_bound() {
        let mut g = Cpdag::with_variables(3);
        g.insert_undirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.insert_undirected(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let mut sampler = CpdagCompletionSampler::new(g, 2).unwrap();
        let collected: Vec<_> = sampler.by_ref().collect();
        assert_eq!(collected.len(), 2);
        assert!(sampler.hit_cap());
    }

    #[test]
    fn refuses_conflict_edges() {
        let mut g = Cpdag::with_variables(2);
        g.insert_undirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.mark_conflict(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        assert!(CpdagCompletionSampler::new(g, 4).is_err());
    }

    #[test]
    fn compelled_collider_preserved() {
        // A → B ← C with A—C absent: classic v-structure CPDAG (A—C may be absent).
        let mut g = Cpdag::with_variables(3);
        let a = DenseNodeId::from_raw(0);
        let b = DenseNodeId::from_raw(1);
        let c = DenseNodeId::from_raw(2);
        g.insert_directed(a, b).unwrap();
        g.insert_directed(c, b).unwrap();
        let collected: Vec<_> = CpdagCompletionSampler::new(g.clone(), 8).unwrap().collect();
        assert_eq!(collected.len(), 1);
        assert!(is_mec_member(&g, &collected[0].graph));
    }

    #[test]
    fn long_undirected_chain_completes_by_pruning() {
        // Path of 26 nodes / 25 undirected edges: MEC has 26 DAGs (one source each).
        // Blind 2^25 would hang; collider pruning must finish promptly.
        let n = 26u32;
        let mut g = Cpdag::with_variables(n);
        for i in 0..(n - 1) {
            g.insert_undirected(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 1)).unwrap();
        }
        assert_eq!(g.undirected_edge_count(), 25);
        let start = Instant::now();
        let collected: Vec<_> = CpdagCompletionSampler::new(g.clone(), 64).unwrap().collect();
        assert!(
            start.elapsed().as_secs_f64() < 1.0,
            "25-edge chain must finish by pruning, took {:?}",
            start.elapsed()
        );
        assert_eq!(collected.len(), 26);
        for c in &collected {
            assert!(is_mec_member(&g, &c.graph));
        }
    }

    #[test]
    fn small_completable_cpdag_still_completes() {
        let mut g = Cpdag::with_variables(2);
        g.insert_undirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let collected: Vec<_> = CpdagCompletionSampler::new(g.clone(), 8).unwrap().collect();
        assert_eq!(collected.len(), 2);
        for c in &collected {
            assert!(is_mec_member(&g, &c.graph));
        }
    }

    #[test]
    fn refuses_above_hard_undirected_ceiling() {
        let n = u32::try_from(MAX_UNDIRECTED_EDGES + 2).unwrap();
        let mut g = Cpdag::with_variables(n);
        for i in 0..(n - 1) {
            g.insert_undirected(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 1)).unwrap();
        }
        assert_eq!(g.undirected_edge_count(), MAX_UNDIRECTED_EDGES + 1);
        let err = CpdagCompletionSampler::new(g, 4).unwrap_err();
        assert!(matches!(err, GraphError::InvalidEndpoints { .. }));
    }
}
