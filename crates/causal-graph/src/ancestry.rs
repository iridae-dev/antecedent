//! Directed ancestry, descendants, and intervention mutilation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::dag::Dag;
use crate::error::GraphError;
use crate::overlay::GraphOverlay;
use crate::types::DenseNodeId;
use crate::workspace::{BitSet, GraphWorkspace};

impl Dag {
    /// Collect all ancestors of `nodes` (including `nodes` themselves) into `out`.
    pub fn ancestors_of(&self, nodes: &[DenseNodeId], out: &mut BitSet, ws: &mut GraphWorkspace) {
        self.ancestors_of_with(nodes, out, ws, None);
    }

    /// Ancestors under an optional edge-visibility overlay.
    pub(crate) fn ancestors_of_with(
        &self,
        nodes: &[DenseNodeId],
        out: &mut BitSet,
        ws: &mut GraphWorkspace,
        overlay: Option<&GraphOverlay>,
    ) {
        let n = self.node_count();
        out.resize(n);
        out.clear();
        ws.prepare(n);
        for &v in nodes {
            if v.as_usize() >= n {
                continue;
            }
            if !out.contains(v) {
                out.insert(v);
                ws.frontier.push(v);
            }
        }
        while let Some(u) = ws.frontier.pop() {
            for &p in self.parents(u) {
                if let Some(ov) = overlay {
                    if !ov.edge_visible(p, u) {
                        continue;
                    }
                }
                if !out.contains(p) {
                    out.insert(p);
                    ws.frontier.push(p);
                }
            }
        }
    }

    /// Collect all descendants of `nodes` (including `nodes`) into `out`.
    pub fn descendants_of(&self, nodes: &[DenseNodeId], out: &mut BitSet, ws: &mut GraphWorkspace) {
        self.descendants_of_with(nodes, out, ws, None);
    }

    /// Descendants under an optional edge-visibility overlay.
    pub(crate) fn descendants_of_with(
        &self,
        nodes: &[DenseNodeId],
        out: &mut BitSet,
        ws: &mut GraphWorkspace,
        overlay: Option<&GraphOverlay>,
    ) {
        let n = self.node_count();
        out.resize(n);
        out.clear();
        ws.prepare(n);
        for &v in nodes {
            if v.as_usize() >= n {
                continue;
            }
            if !out.contains(v) {
                out.insert(v);
                ws.frontier.push(v);
            }
        }
        while let Some(u) = ws.frontier.pop() {
            for &c in self.children(u) {
                if let Some(ov) = overlay {
                    if !ov.edge_visible(u, c) {
                        continue;
                    }
                }
                if !out.contains(c) {
                    out.insert(c);
                    ws.frontier.push(c);
                }
            }
        }
    }

    /// Whether `anc` is an ancestor of `desc` (or equal).
    #[must_use]
    pub fn is_ancestor(&self, anc: DenseNodeId, desc: DenseNodeId) -> bool {
        self.reaches(anc, desc)
    }

    /// Markov blanket of `node`: parents ∪ children ∪ spouses (co-parents of
    /// children). Does not include `node` itself.
    ///
    /// # Errors
    ///
    /// Unknown node id.
    pub fn markov_blanket(&self, node: DenseNodeId, out: &mut BitSet) -> Result<(), GraphError> {
        self.validate_node_pub(node)?;
        let n = self.node_count();
        out.resize(n);
        out.clear();
        for &p in self.parents(node) {
            out.insert(p);
        }
        for &c in self.children(node) {
            out.insert(c);
            for &spouse in self.parents(c) {
                if spouse != node {
                    out.insert(spouse);
                }
            }
        }
        Ok(())
    }

    /// Sorted Markov blanket of `node` (excluding `node`).
    ///
    /// # Errors
    ///
    /// Unknown node id.
    pub fn markov_blanket_nodes(&self, node: DenseNodeId) -> Result<Vec<DenseNodeId>, GraphError> {
        let mut bits = BitSet::with_len(self.node_count());
        self.markov_blanket(node, &mut bits)?;
        Ok((0..self.node_count())
            .map(|i| DenseNodeId::from_raw(u32::try_from(i).expect("node fit")))
            .filter(|&id| bits.contains(id))
            .collect())
    }

    /// Mutilate the graph under intervention: remove all edges into each
    /// intervened node. Returns a new DAG (nodes preserved).
    ///
    /// Prefer [`Dag::view`] with [`GraphOverlay::do_intervention`] on hot paths
    /// to avoid cloning adjacency.
    ///
    /// # Errors
    ///
    /// Unknown node ids.
    pub fn mutilate(&self, intervened: &[DenseNodeId]) -> Result<Dag, GraphError> {
        for &v in intervened {
            self.validate_node_pub(v)?;
        }
        let overlay = GraphOverlay::do_intervention(self.node_count(), intervened);
        self.view(&overlay).materialize()
    }

    pub(crate) fn validate_node_pub(&self, id: DenseNodeId) -> Result<(), GraphError> {
        if id.as_usize() >= self.node_count() {
            Err(GraphError::UnknownNode { id: id.raw() })
        } else {
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn markov_blanket_includes_parents_children_spouses() {
        // A → T ← B, T → Y ← C  ⇒  MB(T) = {A, B, Y, C}
        let mut graph = Dag::with_variables(5);
        let parent_a = DenseNodeId::from_raw(0);
        let parent_b = DenseNodeId::from_raw(1);
        let treatment = DenseNodeId::from_raw(2);
        let outcome = DenseNodeId::from_raw(3);
        let spouse_c = DenseNodeId::from_raw(4);
        graph.insert_directed(parent_a, treatment).unwrap();
        graph.insert_directed(parent_b, treatment).unwrap();
        graph.insert_directed(treatment, outcome).unwrap();
        graph.insert_directed(spouse_c, outcome).unwrap();

        let mb = graph.markov_blanket_nodes(treatment).unwrap();
        assert_eq!(mb, vec![parent_a, parent_b, outcome, spouse_c]);
        assert!(!mb.contains(&treatment));
    }

    #[test]
    fn markov_blanket_of_root_includes_child_and_spouse() {
        let mut graph = Dag::with_variables(3);
        let parent_a = DenseNodeId::from_raw(0);
        let parent_b = DenseNodeId::from_raw(1);
        let outcome = DenseNodeId::from_raw(2);
        graph.insert_directed(parent_a, outcome).unwrap();
        graph.insert_directed(parent_b, outcome).unwrap();
        assert_eq!(graph.markov_blanket_nodes(outcome).unwrap(), vec![parent_a, parent_b]);
        assert_eq!(graph.markov_blanket_nodes(parent_a).unwrap(), vec![parent_b, outcome]);
    }
}
