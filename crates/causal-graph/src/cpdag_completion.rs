//! Streamed / bounded CPDAG MEC completion sampling.
//!
//! Yields DAG members of the Markov equivalence class of a static [`Cpdag`]:
//! orientations of undirected (Tail–Tail) edges that remain acyclic and do not
//! introduce new unshielded colliders. Completions are never retained without
//! bound (`max_completions` caps **valid** yields). Conflict (`x-x`) edges are
//! refused at construction.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::cpdag::Cpdag;
use crate::dag::Dag;
use crate::error::GraphError;
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
    base: Cpdag,
    /// Undirected edges `(a, b)` with `a.raw() < b.raw()`.
    undirected: Vec<(DenseNodeId, DenseNodeId)>,
    max_completions: usize,
    next_index: usize,
    /// Bitmask: bit i = 0 → orient a→b, bit i = 1 → orient b→a.
    assign: u64,
}

impl CpdagCompletionSampler {
    /// Build a sampler that yields at most `max_completions` **valid** MEC DAGs.
    ///
    /// # Errors
    ///
    /// Conflict edges present, or more than 63 undirected edges (mask capacity).
    pub fn new(cpdag: Cpdag, max_completions: usize) -> Result<Self, GraphError> {
        if cpdag.conflict_edge_count() > 0 {
            return Err(GraphError::InvalidEndpoints {
                message: "CpdagCompletionSampler refuses conflict (x-x) edges",
            });
        }
        let mut undirected = Vec::new();
        for e in cpdag.edges() {
            if e.is_undirected() {
                let (a, b) = if e.a.raw() <= e.b.raw() {
                    (e.a, e.b)
                } else {
                    (e.b, e.a)
                };
                undirected.push((a, b));
            }
        }
        undirected.sort_by_key(|(a, b)| (a.raw(), b.raw()));
        undirected.dedup();
        if undirected.len() > 63 {
            return Err(GraphError::InvalidEndpoints {
                message: "too many undirected edges for CpdagCompletionSampler mask",
            });
        }
        Ok(Self {
            base: cpdag,
            undirected,
            max_completions,
            next_index: 0,
            assign: 0,
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

    fn build_completion(&self, mask: u64) -> Option<Dag> {
        let mut g = self.base.clone();
        for (i, &(a, b)) in self.undirected.iter().enumerate() {
            let reverse = ((mask >> i) & 1) == 1;
            let (from, to) = if reverse { (b, a) } else { (a, b) };
            if g.orient_undirected(from, to).is_err() {
                return None;
            }
        }
        let dag = g.try_into_dag().ok()?;
        if is_mec_member(&self.base, &dag) {
            Some(dag)
        } else {
            None
        }
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
    for zi in 0..n {
        let z = DenseNodeId::from_raw(zi as u32);
        let parents = g.parents(z);
        for i in 0..parents.len() {
            for j in (i + 1)..parents.len() {
                let p = parents[i];
                let q = parents[j];
                if !g.has_edge(p, q) {
                    let (a, b) = if p.raw() <= q.raw() {
                        (p.raw(), q.raw())
                    } else {
                        (q.raw(), p.raw())
                    };
                    out.push((a, z.raw(), b));
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
    for zi in 0..n {
        let z = DenseNodeId::from_raw(zi as u32);
        let parents = g.parents(z);
        for i in 0..parents.len() {
            for j in (i + 1)..parents.len() {
                let p = parents[i];
                let q = parents[j];
                let adjacent = g.children(p).contains(&q) || g.children(q).contains(&p);
                if !adjacent {
                    let (a, b) = if p.raw() <= q.raw() {
                        (p.raw(), q.raw())
                    } else {
                        (q.raw(), p.raw())
                    };
                    out.push((a, z.raw(), b));
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

impl Iterator for CpdagCompletionSampler {
    type Item = CpdagCompletion;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index >= self.max_completions {
            return None;
        }
        let n = self.undirected.len();
        let total = if n == 0 { 1u64 } else { 1u64 << n };
        while self.assign < total {
            let mask = self.assign;
            self.assign += 1;
            if let Some(graph) = self.build_completion(mask) {
                let index = self.next_index;
                self.next_index += 1;
                return Some(CpdagCompletion { graph, index });
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::cpdag::Cpdag;

    #[test]
    fn fully_oriented_yields_one() {
        let mut g = Cpdag::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let collected: Vec<_> = CpdagCompletionSampler::new(g, 10).unwrap().collect();
        assert_eq!(collected.len(), 1);
        assert!(collected[0].graph.children(DenseNodeId::from_raw(0)).contains(&DenseNodeId::from_raw(1)));
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
        let collected: Vec<_> = CpdagCompletionSampler::new(g, 2).unwrap().collect();
        assert_eq!(collected.len(), 2);
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
}
