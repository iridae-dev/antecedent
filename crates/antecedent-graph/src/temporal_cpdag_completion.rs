//! Streamed [`TemporalCpdag`] → [`TemporalDag`] MEC completions.
//!
//! Same orientation rule as [`crate::cpdag_completion`]: undirected edges are
//! oriented without new unshielded colliders. Conflict marks are refused.
//! Future→past orientations are rejected by [`TemporalCpdag::orient_undirected`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::cpdag::TemporalCpdag;
use crate::error::GraphError;
use crate::temporal::TemporalDag;
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
    base: TemporalCpdag,
    undirected: Vec<(DenseNodeId, DenseNodeId)>,
    max_completions: usize,
    next_index: usize,
    assign: u64,
}

impl TemporalCpdagCompletionSampler {
    /// Build a sampler that yields at most `max_completions` valid MEC [`TemporalDag`]s.
    ///
    /// # Errors
    ///
    /// Conflict edges, or more than 63 undirected edges.
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
        if undirected.len() > 63 {
            return Err(GraphError::InvalidEndpoints {
                message: "too many undirected edges for TemporalCpdagCompletionSampler mask",
            });
        }
        Ok(Self { base: cpdag, undirected, max_completions, next_index: 0, assign: 0 })
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

    /// Whether the retention cap stopped the stream before every mask was examined.
    #[must_use]
    pub fn hit_cap(&self) -> bool {
        self.next_index >= self.max_completions && self.assign < self.total_masks()
    }

    fn total_masks(&self) -> u64 {
        let n = self.undirected.len();
        if n == 0 { 1 } else { 1u64 << n }
    }

    fn build_completion(&self, mask: u64) -> Option<TemporalDag> {
        let mut g = self.base.clone();
        for (i, &(a, b)) in self.undirected.iter().enumerate() {
            let reverse = ((mask >> i) & 1) == 1;
            let (from, to) = if reverse { (b, a) } else { (a, b) };
            if g.orient_undirected(from, to).is_err() {
                return None;
            }
        }
        let dag = g.try_into_temporal_dag().ok()?;
        if is_temporal_mec_member(&self.base, &dag) { Some(dag) } else { None }
    }
}

/// Whether `dag` is a Markov-equivalence member of `cpdag`.
#[must_use]
pub fn is_temporal_mec_member(cpdag: &TemporalCpdag, dag: &TemporalDag) -> bool {
    if cpdag.node_count() != dag.node_count() {
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

impl Iterator for TemporalCpdagCompletionSampler {
    type Item = TemporalCpdagCompletion;

    fn next(&mut self) -> Option<Self::Item> {
        if self.next_index >= self.max_completions {
            return None;
        }
        let total = self.total_masks();
        while self.assign < total {
            let mask = self.assign;
            self.assign += 1;
            if let Some(graph) = self.build_completion(mask) {
                let index = self.next_index;
                self.next_index += 1;
                return Some(TemporalCpdagCompletion { graph, index });
            }
        }
        None
    }
}

#[cfg(test)]
mod tests {
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
}
