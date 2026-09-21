//! Streamed [`TemporalCpdag`] → [`TemporalDag`] MEC completions.
//!
//! Same orientation rule as [`crate::cpdag_completion`]: undirected edges are
//! oriented without new unshielded colliders. Conflict marks are refused.
//! Future→past orientations are rejected by [`TemporalCpdag::orient_undirected`].
//!
//! Search orients one undirected edge at a time and rejects a partial assignment
//! as soon as it creates a cycle, a future→past edge, or an unshielded collider
//! absent from the CPDAG. A hard ceiling ([`MAX_TEMPORAL_UNDIRECTED_EDGES`])
//! still refuses pathological unconstrained instances.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::cpdag::TemporalCpdag;
use crate::error::GraphError;
use crate::temporal::TemporalDag;
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
    base: TemporalCpdag,
    undirected: Vec<(DenseNodeId, DenseNodeId)>,
    allowed_colliders: Vec<(u32, u32, u32)>,
    max_completions: usize,
    next_index: usize,
    stack: Vec<(TemporalCpdag, usize)>,
}

impl TemporalCpdagCompletionSampler {
    /// Build a sampler that yields at most `max_completions` valid MEC [`TemporalDag`]s.
    ///
    /// # Errors
    ///
    /// Conflict edges, or more than [`MAX_TEMPORAL_UNDIRECTED_EDGES`] undirected edges.
    pub fn new(cpdag: TemporalCpdag, max_completions: usize) -> Result<Self, GraphError> {
        if cpdag.conflict_edge_count() > 0 {
            return Err(GraphError::InvalidEndpoints {
                message: "TemporalCpdagCompletionSampler refuses conflict (x-x) edges",
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
        if undirected.len() > MAX_TEMPORAL_UNDIRECTED_EDGES {
            return Err(GraphError::InvalidEndpoints {
                message: "TemporalCpdagCompletionSampler supports at most 64 undirected edges",
            });
        }
        let allowed_colliders = unshielded_colliders_temporal_cpdag(&cpdag);
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
    #[must_use]
    pub fn hit_cap(&self) -> bool {
        self.next_index >= self.max_completions && !self.stack.is_empty()
    }
}

/// Whether `dag` is a Markov-equivalence member of `cpdag`.
#[must_use]
pub fn is_temporal_mec_member(cpdag: &TemporalCpdag, dag: &TemporalDag) -> bool {
    if cpdag.nodes() != dag.nodes() {
        return false;
    }
    for e in cpdag.edges() {
        if let Some((from, to)) = e.parent_child() {
            if !dag.children(from).contains(&to) {
                return false;
            }
        } else if e.is_undirected() {
            let ab = dag.children(e.a).contains(&e.b);
            let ba = dag.children(e.b).contains(&e.a);
            if ab == ba {
                return false;
            }
        } else if e.is_conflict() {
            return false;
        }
    }
    for e in dag.edges() {
        if let Some((from, to)) = e.parent_child() {
            if !cpdag.has_edge(from, to) {
                return false;
            }
        }
    }
    unshielded_colliders_temporal_cpdag(cpdag) == unshielded_colliders_temporal_dag(dag)
}

fn unshielded_colliders_temporal_cpdag(g: &TemporalCpdag) -> Vec<(u32, u32, u32)> {
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

fn unshielded_colliders_temporal_dag(g: &TemporalDag) -> Vec<(u32, u32, u32)> {
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

fn has_forbidden_collider_at(
    allowed: &[(u32, u32, u32)],
    g: &TemporalCpdag,
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

impl Iterator for TemporalCpdagCompletionSampler {
    type Item = TemporalCpdagCompletion;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index >= self.max_completions {
            return None;
        }
        while let Some((g, edge_i)) = self.stack.pop() {
            if edge_i == self.undirected.len() {
                let dag = g.try_into_temporal_dag().ok()?;
                if !is_temporal_mec_member(&self.base, &dag) {
                    continue;
                }
                let index = self.next_index;
                self.next_index += 1;
                return Some(TemporalCpdagCompletion { graph: dag, index });
            }
            let (a, b) = self.undirected[edge_i];
            for (from, to) in [(b, a), (a, b)] {
                let mut next_g = g.clone();
                if next_g.orient_undirected(from, to).is_err() {
                    continue;
                }
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
