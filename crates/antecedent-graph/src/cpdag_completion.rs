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
//! instead of scanning `2^k` full masks, however many undirected edges they carry.
//! No edge-count ceiling applies: a [`SearchBudget`](antecedent_core::SearchBudget)
//! (operations, memory, cancellation) bounds the visited states, and a stop is typed
//! ([`CpdagCompletionSampler::search_stop`]) and counted by [`CpdagCompletionSampler::hit_cap`].
//! The search itself is shared with the temporal sampler (`crate::mec_search`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{ExecutionContext, SearchLimits, SearchReceipt, SearchStop};

use crate::completion::default_completion_search;
use crate::cpdag::Cpdag;
use crate::dag::Dag;
use crate::error::GraphError;
use crate::mec_search::{self, MecSearch};
#[cfg(test)]
use crate::types::DenseNodeId;

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
    /// The visited states are bounded by [`crate::DEFAULT_COMPLETION_OPERATIONS`]; use
    /// [`Self::with_budget`] to choose the limits, memory cap and cancellation.
    ///
    /// # Errors
    ///
    /// Conflict edges present.
    pub fn new(cpdag: Cpdag, max_completions: usize) -> Result<Self, GraphError> {
        let (limits, ctx) = default_completion_search();
        Self::with_budget(cpdag, max_completions, limits, &ctx)
    }

    /// [`Self::new`] under an explicit search budget: `limits.operations` bounds the partial
    /// orientations visited, and `ctx` supplies cancellation and the hard memory limit. A stop
    /// ends the stream; see [`Self::search_stop`].
    ///
    /// # Errors
    ///
    /// Conflict edges present.
    pub fn with_budget(
        cpdag: Cpdag,
        max_completions: usize,
        limits: SearchLimits,
        ctx: &ExecutionContext,
    ) -> Result<Self, GraphError> {
        let search = MecSearch::new(
            cpdag,
            max_completions,
            limits,
            ctx,
            "CpdagCompletionSampler refuses conflict (x-x) edges",
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
    /// Local validity still holds for yielded DAGs; the unexplored remainder is
    /// not certified empty of further MEC members, so no class-wide claim is earned.
    #[must_use]
    pub fn hit_cap(&self) -> bool {
        self.search.hit_cap()
    }

    /// The bound that stopped the search (operations, depth, memory or cancellation), if a
    /// budget rather than the retention cap ended the stream.
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

    /// An undirected tree on `n` nodes (node `i > 0` hangs off `parent_of(i)`).
    fn undirected_tree(n: u32, parent_of: impl Fn(u32) -> u32) -> Cpdag {
        let mut g = Cpdag::with_variables(n);
        for i in 1..n {
            g.insert_undirected(DenseNodeId::from_raw(parent_of(i)), DenseNodeId::from_raw(i))
                .unwrap();
        }
        g
    }

    /// The class of an undirected tree on `n` nodes is exactly its `n` rootings: one DAG per
    /// choice of the single source, every edge pointing away from it.
    fn assert_one_dag_per_rooting(g: &Cpdag, n: u32, completions: &[CpdagCompletion]) {
        assert_eq!(completions.len(), n as usize, "an undirected tree has one member per node");
        let skeleton: Vec<(u32, u32)> = g.edges().iter().map(|e| (e.a.raw(), e.b.raw())).collect();
        let mut roots = std::collections::BTreeSet::new();
        for c in completions {
            let sources: Vec<u32> =
                (0..n).filter(|&v| c.graph.parents(DenseNodeId::from_raw(v)).is_empty()).collect();
            assert_eq!(sources.len(), 1, "a rooted orientation has a single source");
            let root = sources[0];
            assert!(roots.insert(root), "each root is realised once");
            // Breadth-first distance from the root over the skeleton.
            let mut dist = vec![u32::MAX; n as usize];
            dist[root as usize] = 0;
            let mut queue = std::collections::VecDeque::from([root]);
            while let Some(u) = queue.pop_front() {
                for &(a, b) in &skeleton {
                    let next = if a == u {
                        b
                    } else if b == u {
                        a
                    } else {
                        continue;
                    };
                    if dist[next as usize] == u32::MAX {
                        dist[next as usize] = dist[u as usize] + 1;
                        queue.push_back(next);
                    }
                }
            }
            for &(a, b) in &skeleton {
                let a_to_b =
                    c.graph.children(DenseNodeId::from_raw(a)).contains(&DenseNodeId::from_raw(b));
                assert_eq!(a_to_b, dist[a as usize] < dist[b as usize]);
            }
            assert!(is_mec_member(g, &c.graph));
        }
        assert_eq!(roots.len(), n as usize);
    }

    #[test]
    fn a_seventy_one_edge_path_completes_with_one_dag_per_source() {
        let n = 72u32;
        let g = undirected_tree(n, |i| i - 1);
        assert_eq!(g.undirected_edge_count(), 71);
        let mut sampler = CpdagCompletionSampler::new(g.clone(), 128).unwrap();
        let collected: Vec<_> = sampler.by_ref().collect();
        assert_eq!(sampler.n_undirected(), 71);
        assert!(!sampler.hit_cap());
        assert_eq!(sampler.search_stop(), None);
        assert_one_dag_per_rooting(&g, n, &collected);
    }

    #[test]
    fn a_seventy_two_edge_class_completes_and_its_membership_audit_rejects_a_collider() {
        // 72 undirected edges, past the old 64-edge ceiling: one DAG per source, every one a
        // member of the class, found without any bound being reached.
        let n = 73u32;
        let g = undirected_tree(n, |i| i - 1);
        assert_eq!(g.undirected_edge_count(), 72);
        let mut sampler = CpdagCompletionSampler::new(g.clone(), 128).unwrap();
        let collected: Vec<_> = sampler.by_ref().collect();
        assert_eq!(sampler.n_undirected(), 72);
        assert!(!sampler.hit_cap());
        assert_eq!(sampler.search_stop(), None);
        assert!(sampler.search_receipt().is_none());
        assert_one_dag_per_rooting(&g, n, &collected);
        // Membership is audited, not assumed: two sources make a collider the class does not
        // have, and the path read in one direction with one edge flipped is exactly that.
        let mut collider = Dag::with_variables(n);
        for i in 1..n {
            let (from, to) = if i == 40 { (i, i - 1) } else { (i - 1, i) };
            collider
                .insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
                .unwrap();
        }
        assert!(!is_mec_member(&g, &collider));
    }

    #[test]
    fn search_receipts_report_the_visited_state_work_and_memory_controls() {
        // K8 has 8! = 40320 members; the receipt of a starved search states every control in
        // force and what it consumed.
        let g = complete_undirected(8);
        let mut ctx = ExecutionContext::for_tests(0);
        ctx.set_memory_limit(Some(64 * 1024 * 1024));
        let limits = SearchLimits { operations: 500, depth: 40 };
        let mut sampler =
            CpdagCompletionSampler::with_budget(g.clone(), usize::MAX, limits, &ctx).unwrap();
        let _ = sampler.by_ref().count();
        let receipt = sampler.search_receipt().expect("the work limit stops the search");
        assert_eq!(receipt.stop, SearchStop::Operations);
        assert_eq!(receipt.operations_limit, 500, "work control");
        assert_eq!(receipt.operations_consumed, Some(500), "visited states");
        assert_eq!(receipt.depth_limit, 40);
        assert!(receipt.depth_reached.is_some_and(|depth| depth <= 40));
        assert_eq!(receipt.memory_limit_bytes, Some(64 * 1024 * 1024), "memory control");
        assert!(receipt.explored[0].contains("500 orientation states visited"));
        assert!(!receipt.unevaluated.is_empty());

        // A memory limit that cannot hold one partial orientation reports itself as the stop.
        let mut tight = ExecutionContext::for_tests(0);
        tight.set_memory_limit(Some(1));
        let limits = SearchLimits { operations: 1_000_000, depth: usize::MAX };
        let mut sampler =
            CpdagCompletionSampler::with_budget(g, usize::MAX, limits, &tight).unwrap();
        assert!(sampler.next().is_none());
        let receipt = sampler.search_receipt().expect("the memory limit stops the search");
        assert_eq!(receipt.stop, SearchStop::Memory);
        assert_eq!(receipt.memory_limit_bytes, Some(1));
        assert_eq!(receipt.operations_limit, 1_000_000);
    }

    #[test]
    fn partial_orientation_storage_is_leased_before_it_is_allocated() {
        use antecedent_core::execution::AdmissionPool;
        let g = complete_undirected(6);
        let limits = SearchLimits { operations: 1_000_000, depth: usize::MAX };
        // A pool with one byte free funds no block of states: the search stops on memory
        // before it stacks anything, yields nothing, and leases nothing.
        let pool = AdmissionPool::fixed(1, 1 << 20);
        let ctx = ExecutionContext::for_tests(0).with_admission(pool.clone());
        let held = ctx.acquire_admission(0, (1 << 20) - 1).unwrap();
        let mut sampler = CpdagCompletionSampler::with_budget(g.clone(), 64, limits, &ctx).unwrap();
        assert!(sampler.next().is_none());
        assert_eq!(sampler.search_stop(), Some(SearchStop::Memory));
        assert_eq!(pool.available().bytes(), 1);
        drop(sampler);
        drop(held);
        assert_eq!(pool.available().bytes(), 1 << 20);
        // The same class under an ample pool completes, and the leases return with the sampler.
        let ample = AdmissionPool::fixed(1, 1 << 30);
        let ctx = ExecutionContext::for_tests(0).with_admission(ample.clone());
        let mut sampler = CpdagCompletionSampler::with_budget(g, 1_000, limits, &ctx).unwrap();
        assert_eq!(sampler.by_ref().count(), 720);
        assert_eq!(sampler.search_stop(), None);
        assert!(ample.available().bytes() < 1 << 30);
        drop(sampler);
        assert_eq!(ample.available().bytes(), 1 << 30);
    }

    #[test]
    fn cancellation_is_honoured_within_one_state_expansion() {
        let g = complete_undirected(8);
        let limits = SearchLimits { operations: 1_000_000, depth: usize::MAX };

        // Cancelled before the first state: nothing is expanded or yielded.
        let ctx = ExecutionContext::for_tests(0);
        ctx.cancellation.cancel();
        let mut sampler =
            CpdagCompletionSampler::with_budget(g.clone(), usize::MAX, limits, &ctx).unwrap();
        assert!(sampler.next().is_none());
        let receipt = sampler.search_receipt().expect("cancellation is a typed stop");
        assert_eq!(receipt.stop, SearchStop::Cancelled);
        assert_eq!(receipt.operations_consumed, None, "no state was expanded");

        // Cancelled after a fixed number of checks: the search charges one check per state, so
        // it has expanded at most that many states, and a stopped search expands no more.
        let mut ctx = ExecutionContext::for_tests(0);
        ctx.cancellation = antecedent_core::CancellationToken::cancel_after_checks(25);
        let mut sampler = CpdagCompletionSampler::with_budget(g, usize::MAX, limits, &ctx).unwrap();
        let yielded = sampler.by_ref().count();
        assert!(yielded < 40_320);
        let receipt = sampler.search_receipt().expect("cancellation is a typed stop");
        assert_eq!(receipt.stop, SearchStop::Cancelled);
        let consumed = receipt.operations_consumed.expect("states were expanded before the stop");
        assert!(consumed <= 25, "{consumed} states expanded under 25 cancellation checks");
        assert!(sampler.next().is_none(), "a cancelled search is sticky");
        assert_eq!(sampler.search_receipt().unwrap().operations_consumed, Some(consumed));
        assert!(sampler.hit_cap(), "a cancelled search is not read as the whole class");
    }

    #[test]
    fn an_eighty_node_binary_tree_completes_with_one_dag_per_rooting() {
        let n = 80u32;
        let g = undirected_tree(n, |i| (i - 1) / 2);
        assert_eq!(g.undirected_edge_count(), 79);
        let mut sampler = CpdagCompletionSampler::new(g.clone(), 128).unwrap();
        let collected: Vec<_> = sampler.by_ref().collect();
        assert!(!sampler.hit_cap());
        assert_one_dag_per_rooting(&g, n, &collected);
    }

    #[test]
    fn disjoint_edges_beyond_the_old_ceiling_stream_up_to_the_retention_cap() {
        // 72 disjoint undirected edges: 2^72 members, far past any enumeration, but the
        // first ones are reached depth-first and the cap is reported, not an error.
        let mut g = Cpdag::with_variables(144);
        for i in 0..72 {
            g.insert_undirected(DenseNodeId::from_raw(2 * i), DenseNodeId::from_raw(2 * i + 1))
                .unwrap();
        }
        let mut sampler = CpdagCompletionSampler::new(g.clone(), 3).unwrap();
        assert_eq!(sampler.n_undirected(), 72);
        let collected: Vec<_> = sampler.by_ref().collect();
        assert_eq!(collected.len(), 3);
        assert!(sampler.hit_cap());
        assert_eq!(sampler.search_stop(), None, "the retention cap is not a budget stop");
        for c in &collected {
            assert!(is_mec_member(&g, &c.graph));
        }
    }

    fn complete_undirected(n: u32) -> Cpdag {
        let mut g = Cpdag::with_variables(n);
        for a in 0..n {
            for b in (a + 1)..n {
                g.insert_undirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
            }
        }
        g
    }

    #[test]
    fn an_exhausted_operation_budget_is_a_typed_partial_result() {
        // K8 has 8! = 40320 members; 500 visited states cannot enumerate them.
        let g = complete_undirected(8);
        let ctx = ExecutionContext::for_tests(0);
        let limits = SearchLimits { operations: 500, depth: usize::MAX };
        let mut sampler =
            CpdagCompletionSampler::with_budget(g.clone(), usize::MAX, limits, &ctx).unwrap();
        let collected: Vec<_> = sampler.by_ref().collect();
        assert!(collected.len() < 40_320);
        assert_eq!(sampler.search_stop(), Some(SearchStop::Operations));
        assert!(sampler.hit_cap(), "a stopped search must not be read as the whole class");
        let receipt = sampler.search_receipt().unwrap();
        assert_eq!(receipt.stop, SearchStop::Operations);
        assert_eq!(receipt.operations_consumed, Some(500));
        assert!(receipt.explored[0].contains("500 orientation states"));
        assert!(!receipt.unevaluated.is_empty());
        for c in &collected {
            assert!(is_mec_member(&g, &c.graph), "yielded members stay locally valid");
        }
        assert!(sampler.next().is_none(), "a budget stop is sticky");
    }

    #[test]
    fn a_budget_large_enough_completes_the_same_class() {
        // K6: 6! = 720 members.
        let g = complete_undirected(6);
        let ctx = ExecutionContext::for_tests(0);
        let limits = SearchLimits { operations: 1_000_000, depth: usize::MAX };
        let mut sampler = CpdagCompletionSampler::with_budget(g, usize::MAX, limits, &ctx).unwrap();
        assert_eq!(sampler.by_ref().count(), 720);
        assert!(!sampler.hit_cap());
        assert_eq!(sampler.search_stop(), None);
        assert!(sampler.search_receipt().is_none());
    }

    #[test]
    fn cancellation_stops_the_search_before_and_during_enumeration() {
        let g = complete_undirected(8);
        let limits = SearchLimits { operations: 1_000_000, depth: usize::MAX };

        let ctx = ExecutionContext::for_tests(0);
        ctx.cancellation.cancel();
        let mut sampler =
            CpdagCompletionSampler::with_budget(g.clone(), usize::MAX, limits, &ctx).unwrap();
        assert!(sampler.next().is_none());
        assert_eq!(sampler.search_stop(), Some(SearchStop::Cancelled));
        assert!(sampler.hit_cap());

        let mut ctx = ExecutionContext::for_tests(0);
        ctx.cancellation = antecedent_core::CancellationToken::cancel_after_checks(200);
        let mut sampler = CpdagCompletionSampler::with_budget(g, usize::MAX, limits, &ctx).unwrap();
        let yielded = sampler.by_ref().count();
        assert!(yielded < 40_320);
        assert_eq!(sampler.search_stop(), Some(SearchStop::Cancelled));
    }

    #[test]
    fn a_memory_limit_below_one_partial_orientation_stops_the_search() {
        let g = complete_undirected(4);
        let mut ctx = ExecutionContext::for_tests(0);
        ctx.set_memory_limit(Some(1));
        let limits = SearchLimits { operations: 1_000_000, depth: usize::MAX };
        let mut sampler = CpdagCompletionSampler::with_budget(g, usize::MAX, limits, &ctx).unwrap();
        assert!(sampler.next().is_none());
        assert_eq!(sampler.search_stop(), Some(SearchStop::Memory));
    }
}
