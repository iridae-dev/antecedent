//! Orientation search shared by the static and temporal CPDAG completion samplers.
//!
//! A CPDAG's Markov equivalence class is the set of DAGs with the same skeleton and the same
//! unshielded colliders whose orientations extend the CPDAG's compelled edges. The search
//! orients one undirected edge at a time and prunes a partial assignment as soon as it closes
//! a directed cycle (or, on temporal graphs, a future→past edge — refused by
//! `orient_undirected`) or creates an unshielded collider the CPDAG does not carry.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::cpdag::{Cpdag, TemporalCpdag};
use crate::dag::Dag;
use crate::error::GraphError;
use crate::temporal::TemporalDag;
use crate::types::{DenseNodeId, MarkedEdge};

/// The CPDAG operations the search needs.
pub(crate) trait MecCpdag: Clone {
    fn node_count(&self) -> usize;
    fn parents(&self, id: DenseNodeId) -> Vec<DenseNodeId>;
    fn has_edge(&self, a: DenseNodeId, b: DenseNodeId) -> bool;
    fn edges(&self) -> Vec<MarkedEdge>;
    fn conflict_edge_count(&self) -> usize;
    fn orient_undirected(&mut self, from: DenseNodeId, to: DenseNodeId) -> Result<(), GraphError>;
}

/// The DAG operations the membership test needs.
pub(crate) trait MecDag {
    fn node_count(&self) -> usize;
    fn parents(&self, id: DenseNodeId) -> &[DenseNodeId];
    fn children(&self, id: DenseNodeId) -> &[DenseNodeId];
    fn edges(&self) -> Vec<MarkedEdge>;
}

macro_rules! forward_mec_cpdag {
    ($ty:ty) => {
        impl MecCpdag for $ty {
            fn node_count(&self) -> usize {
                <$ty>::node_count(self)
            }
            fn parents(&self, id: DenseNodeId) -> Vec<DenseNodeId> {
                <$ty>::parents(self, id)
            }
            fn has_edge(&self, a: DenseNodeId, b: DenseNodeId) -> bool {
                <$ty>::has_edge(self, a, b)
            }
            fn edges(&self) -> Vec<MarkedEdge> {
                <$ty>::edges(self)
            }
            fn conflict_edge_count(&self) -> usize {
                <$ty>::conflict_edge_count(self)
            }
            fn orient_undirected(
                &mut self,
                from: DenseNodeId,
                to: DenseNodeId,
            ) -> Result<(), GraphError> {
                <$ty>::orient_undirected(self, from, to)
            }
        }
    };
}

macro_rules! forward_mec_dag {
    ($ty:ty) => {
        impl MecDag for $ty {
            fn node_count(&self) -> usize {
                <$ty>::node_count(self)
            }
            fn parents(&self, id: DenseNodeId) -> &[DenseNodeId] {
                <$ty>::parents(self, id)
            }
            fn children(&self, id: DenseNodeId) -> &[DenseNodeId] {
                <$ty>::children(self, id)
            }
            fn edges(&self) -> Vec<MarkedEdge> {
                <$ty>::edges(self).collect()
            }
        }
    };
}

forward_mec_cpdag!(Cpdag);
forward_mec_cpdag!(TemporalCpdag);
forward_mec_dag!(Dag);
forward_mec_dag!(TemporalDag);

fn sorted_pair(a: DenseNodeId, b: DenseNodeId) -> (u32, u32) {
    if a.raw() <= b.raw() { (a.raw(), b.raw()) } else { (b.raw(), a.raw()) }
}

/// Unshielded colliders `(low parent, centre, high parent)` of a CPDAG, sorted.
pub(crate) fn unshielded_colliders_cpdag<G: MecCpdag>(g: &G) -> Vec<(u32, u32, u32)> {
    let mut out = Vec::new();
    for center_idx in 0..g.node_count() {
        let Ok(center_raw) = u32::try_from(center_idx) else {
            break;
        };
        let center = DenseNodeId::from_raw(center_raw);
        let parents = g.parents(center);
        for left_i in 0..parents.len() {
            for right_i in (left_i + 1)..parents.len() {
                let (left, right) = (parents[left_i], parents[right_i]);
                if !g.has_edge(left, right) {
                    let (lo, hi) = sorted_pair(left, right);
                    out.push((lo, center.raw(), hi));
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Unshielded colliders of a DAG, sorted.
fn unshielded_colliders_dag<D: MecDag>(g: &D) -> Vec<(u32, u32, u32)> {
    let mut out = Vec::new();
    for center_idx in 0..g.node_count() {
        let Ok(center_raw) = u32::try_from(center_idx) else {
            break;
        };
        let center = DenseNodeId::from_raw(center_raw);
        let parents = g.parents(center);
        for left_i in 0..parents.len() {
            for right_i in (left_i + 1)..parents.len() {
                let (left, right) = (parents[left_i], parents[right_i]);
                let adjacent =
                    g.children(left).contains(&right) || g.children(right).contains(&left);
                if !adjacent {
                    let (lo, hi) = sorted_pair(left, right);
                    out.push((lo, center.raw(), hi));
                }
            }
        }
    }
    out.sort_unstable();
    out.dedup();
    out
}

/// Whether `dag` is a Markov-equivalence member of `cpdag`: every compelled edge present,
/// every undirected edge oriented exactly one way, no edge outside the skeleton, and the
/// same unshielded colliders. Node identity (`same_nodes`) is the caller's check.
pub(crate) fn is_mec_member<G: MecCpdag, D: MecDag>(cpdag: &G, dag: &D, same_nodes: bool) -> bool {
    if !same_nodes {
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
                // missing or both — not a simple orientation
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
    unshielded_colliders_cpdag(cpdag) == unshielded_colliders_dag(dag)
}

/// True when `center` already has an unshielded collider not present in `allowed` (sorted).
fn has_forbidden_collider_at<G: MecCpdag>(
    allowed: &[(u32, u32, u32)],
    g: &G,
    center: DenseNodeId,
) -> bool {
    let parents = g.parents(center);
    for left_i in 0..parents.len() {
        for right_i in (left_i + 1)..parents.len() {
            let (left, right) = (parents[left_i], parents[right_i]);
            if g.has_edge(left, right) {
                continue;
            }
            let (lo, hi) = sorted_pair(left, right);
            if allowed.binary_search(&(lo, center.raw(), hi)).is_err() {
                return true;
            }
        }
    }
    false
}

/// Depth-first orientation search over the undirected edges of a CPDAG.
#[derive(Clone, Debug)]
pub(crate) struct MecSearch<G: MecCpdag> {
    pub(crate) base: G,
    /// Undirected edges `(a, b)` with `a.raw() <= b.raw()`.
    undirected: Vec<(DenseNodeId, DenseNodeId)>,
    /// Unshielded colliders already present in `base` (sorted).
    allowed_colliders: Vec<(u32, u32, u32)>,
    pub(crate) max_completions: usize,
    pub(crate) next_index: usize,
    /// Partial orientation and index of the next edge to orient.
    stack: Vec<(G, usize)>,
}

impl<G: MecCpdag> MecSearch<G> {
    /// # Errors
    ///
    /// `conflict_message` when the CPDAG carries conflict (`x-x`) edges; `ceiling_message`
    /// when it has more than `ceiling` undirected edges.
    pub(crate) fn new(
        cpdag: G,
        max_completions: usize,
        ceiling: usize,
        conflict_message: &'static str,
        ceiling_message: &'static str,
    ) -> Result<Self, GraphError> {
        if cpdag.conflict_edge_count() > 0 {
            return Err(GraphError::InvalidEndpoints { message: conflict_message });
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
        if undirected.len() > ceiling {
            return Err(GraphError::InvalidEndpoints { message: ceiling_message });
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

    pub(crate) fn n_undirected(&self) -> usize {
        self.undirected.len()
    }

    /// Whether the retention cap stopped the stream before the search finished.
    pub(crate) fn hit_cap(&self) -> bool {
        self.next_index >= self.max_completions && !self.stack.is_empty()
    }

    /// The next fully oriented CPDAG (every undirected edge oriented, no forbidden collider).
    pub(crate) fn next_oriented(&mut self) -> Option<G> {
        if self.next_index >= self.max_completions {
            return None;
        }
        while let Some((g, edge_i)) = self.stack.pop() {
            if edge_i == self.undirected.len() {
                return Some(g);
            }
            let (a, b) = self.undirected[edge_i];
            // Push reverse first so a→b is explored first (LIFO).
            for (from, to) in [(b, a), (a, b)] {
                let mut next_g = g.clone();
                if next_g.orient_undirected(from, to).is_err() {
                    continue; // cycle (or temporal future→past)
                }
                // A new arrow into `to` can only create colliders centred at `to`.
                if has_forbidden_collider_at(&self.allowed_colliders, &next_g, to) {
                    continue;
                }
                self.stack.push((next_g, edge_i + 1));
            }
        }
        None
    }

    /// Record one yielded completion and return its stream index.
    pub(crate) fn record_yield(&mut self) -> usize {
        let index = self.next_index;
        self.next_index += 1;
        index
    }
}
