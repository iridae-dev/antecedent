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
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::cpdag::Cpdag;
use crate::dag::Dag;
use crate::error::GraphError;
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
    base: Cpdag,
    /// Undirected edges `(a, b)` with `a.raw() <= b.raw()`.
    undirected: Vec<(DenseNodeId, DenseNodeId)>,
    /// Unshielded colliders already present in `base` (sorted).
    allowed_colliders: Vec<(u32, u32, u32)>,
    max_completions: usize,
    next_index: usize,
    /// DFS stack: partial orientation and index of the next edge to orient.
    stack: Vec<(Cpdag, usize)>,
}

impl CpdagCompletionSampler {
    /// Build a sampler that yields at most `max_completions` **valid** MEC DAGs.
    ///
    /// # Errors
    ///
    /// Conflict edges present, or more than [`MAX_UNDIRECTED_EDGES`] undirected edges.
    pub fn new(cpdag: Cpdag, max_completions: usize) -> Result<Self, GraphError> {
        if cpdag.conflict_edge_count() > 0 {
            return Err(GraphError::InvalidEndpoints {
                message: "CpdagCompletionSampler refuses conflict (x-x) edges",
            });
        }
        let mut undirected = Vec::new();
        for e in cpdag.edges() {
            if e.is_undirected() {
                let (a, b) = if e.a.raw() <= e.b.raw() { (e.a, e.b) } else { (e.b, e.a) };
                undirected.push((a, b));
            }
        }
        undirected.sort_by_key(|(a, b)| (a.raw(), b.raw()));
        undirected.dedup();
        if undirected.len() > MAX_UNDIRECTED_EDGES {
            return Err(GraphError::InvalidEndpoints {
                message: "CpdagCompletionSampler supports at most 64 undirected edges",
            });
        }
        let allowed_colliders = unshielded_colliders_cpdag(&cpdag);
        let start = cpdag.clone();
        Ok(Self {
            base: cpdag,
            undirected,
            allowed_colliders,
            max_completions,
            next_index: 0,
            stack: vec![(start, 0)],
        })
    }

    /// Hard cap on yielded valid completions.
    #[must_use]
    pub fn max_completions(&self) -> usize {
        self.max_completions
    }

    /// Number of undirected edges being oriented.
    #[must_use]
    pub fn n_undirected(&self) -> usize {
        self.undirected.len()
    }

    /// Whether the retention cap stopped the stream before the search finished.
    /// Local validity still holds for yielded DAGs; the unexplored remainder is
    /// not certified empty of further MEC members.
    #[must_use]
    pub fn hit_cap(&self) -> bool {
        self.next_index >= self.max_completions && !self.stack.is_empty()
    }
}

/// Whether `dag` is a Markov-equivalence member of `cpdag` (same skeleton, same
/// unshielded colliders, all compelled directed edges of the CPDAG present).
#[must_use]
pub fn is_mec_member(cpdag: &Cpdag, dag: &Dag) -> bool {
    if cpdag.node_count() != dag.node_count() {
        return false;
    }
    // Compelled directed edges must appear.
    for e in cpdag.edges() {
        if let Some((from, to)) = e.parent_child() {
            if !dag.children(from).contains(&to) {
                return false;
            }
        } else if e.is_undirected() {
            let a = e.a;
            let b = e.b;
            let ab = dag.children(a).contains(&b);
            let ba = dag.children(b).contains(&a);
            if ab == ba {
                // missing or both — not a simple orientation
                return false;
            }
        } else if e.is_conflict() {
            return false;
        }
    }
    // Skeleton: every DAG edge must exist in the CPDAG (any mark).
    for e in dag.edges() {
        if let Some((from, to)) = e.parent_child() {
            if !cpdag.has_edge(from, to) {
                return false;
            }
        }
    }
    // Unshielded colliders must match.
    let cpdag_colliders = unshielded_colliders_cpdag(cpdag);
    let dag_colliders = unshielded_colliders_dag(dag);
    cpdag_colliders == dag_colliders
}

fn unshielded_colliders_cpdag(g: &Cpdag) -> Vec<(u32, u32, u32)> {
    let n = g.node_count();
    let mut out = Vec::new();
    for center_idx in 0..n {
        let Ok(center_raw) = u32::try_from(center_idx) else {
            break;
        };
        let center = DenseNodeId::from_raw(center_raw);
        let parents = g.parents(center);
        for left_i in 0..parents.len() {
            for right_i in (left_i + 1)..parents.len() {
                let left_parent = parents[left_i];
                let right_parent = parents[right_i];
                if !g.has_edge(left_parent, right_parent) {
                    let (lo, hi) = if left_parent.raw() <= right_parent.raw() {
                        (left_parent.raw(), right_parent.raw())
                    } else {
                        (right_parent.raw(), left_parent.raw())
                    };
                    out.push((lo, center.raw(), hi));
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

fn unshielded_colliders_dag(g: &Dag) -> Vec<(u32, u32, u32)> {
    let n = g.node_count();
    let mut out = Vec::new();
    for center_idx in 0..n {
        let Ok(center_raw) = u32::try_from(center_idx) else {
            break;
        };
        let center = DenseNodeId::from_raw(center_raw);
        let parents = g.parents(center);
        for left_i in 0..parents.len() {
            for right_i in (left_i + 1)..parents.len() {
                let left_parent = parents[left_i];
                let right_parent = parents[right_i];
                let adjacent = g.children(left_parent).contains(&right_parent)
                    || g.children(right_parent).contains(&left_parent);
                if !adjacent {
                    let (lo, hi) = if left_parent.raw() <= right_parent.raw() {
                        (left_parent.raw(), right_parent.raw())
                    } else {
                        (right_parent.raw(), left_parent.raw())
                    };
                    out.push((lo, center.raw(), hi));
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// True when `center` already has an unshielded collider not present in `allowed`.
fn has_forbidden_collider_at(
    allowed: &[(u32, u32, u32)],
    g: &Cpdag,
    center: DenseNodeId,
) -> bool {
    let parents = g.parents(center);
    for left_i in 0..parents.len() {
        for right_i in (left_i + 1)..parents.len() {
            let left_parent = parents[left_i];
            let right_parent = parents[right_i];
            if g.has_edge(left_parent, right_parent) {
                continue;
            }
            let (lo, hi) = if left_parent.raw() <= right_parent.raw() {
                (left_parent.raw(), right_parent.raw())
            } else {
                (right_parent.raw(), left_parent.raw())
            };
            let trip = (lo, center.raw(), hi);
            if allowed.binary_search(&trip).is_err() {
                return true;
            }
        }
    }
    false
}

impl Iterator for CpdagCompletionSampler {
    type Item = CpdagCompletion;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index >= self.max_completions {
            return None;
        }
        while let Some((g, edge_i)) = self.stack.pop() {
            if edge_i == self.undirected.len() {
                let dag = g.try_into_dag().ok()?;
                if !is_mec_member(&self.base, &dag) {
                    continue;
                }
                let index = self.next_index;
                self.next_index += 1;
                return Some(CpdagCompletion { graph: dag, index });
            }
            let (a, b) = self.undirected[edge_i];
            // Push reverse first so a→b is explored first (LIFO).
            for (from, to) in [(b, a), (a, b)] {
                let mut next_g = g.clone();
                if next_g.orient_undirected(from, to).is_err() {
                    continue; // cycle (or temporal future→past on TemporalCpdag)
                }
                // New arrow into `to` can only create colliders centered at `to`.
                if has_forbidden_collider_at(&self.allowed_colliders, &next_g, to) {
                    continue;
                }
                self.stack.push((next_g, edge_i + 1));
            }
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
