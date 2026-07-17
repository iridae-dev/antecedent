//! Directed ancestry, descendants, and intervention mutilation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::dag::Dag;
use crate::error::GraphError;
use crate::types::DenseNodeId;
use crate::workspace::{BitSet, GraphWorkspace};

impl Dag {
    /// Collect all ancestors of `nodes` (including `nodes` themselves) into `out`.
    pub fn ancestors_of(&self, nodes: &[DenseNodeId], out: &mut BitSet, ws: &mut GraphWorkspace) {
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
                if !out.contains(p) {
                    out.insert(p);
                    ws.frontier.push(p);
                }
            }
        }
    }

    /// Collect all descendants of `nodes` (including `nodes`) into `out`.
    pub fn descendants_of(&self, nodes: &[DenseNodeId], out: &mut BitSet, ws: &mut GraphWorkspace) {
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
    pub fn markov_blanket(
        &self,
        node: DenseNodeId,
        out: &mut BitSet,
    ) -> Result<(), GraphError> {
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
    /// # Errors
    ///
    /// Unknown node ids.
    pub fn mutilate(&self, intervened: &[DenseNodeId]) -> Result<Dag, GraphError> {
        for &v in intervened {
            self.validate_node_pub(v)?;
        }
        let mut out = Dag::with_variables(
            u32::try_from(self.node_count()).map_err(|_| GraphError::TooManyNodes)?,
        );
        // Copy only non-incoming-to-intervened edges. The source is a valid DAG
        // and removing edges cannot create cycles, so skip per-edge validation.
        let mut blocked = BitSet::with_len(self.node_count());
        for &v in intervened {
            blocked.insert(v);
        }
        for i in 0..self.node_count() {
            let from = DenseNodeId::from_raw(u32::try_from(i).expect("node fit"));
            for &to in self.children(from) {
                if blocked.contains(to) {
                    continue;
                }
                out.insert_directed_unchecked(from, to);
            }
        }
        Ok(out)
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
        let mut g = Dag::with_variables(5);
        let a = DenseNodeId::from_raw(0);
        let b = DenseNodeId::from_raw(1);
        let t = DenseNodeId::from_raw(2);
        let y = DenseNodeId::from_raw(3);
        let c = DenseNodeId::from_raw(4);
        g.insert_directed(a, t).unwrap();
        g.insert_directed(b, t).unwrap();
        g.insert_directed(t, y).unwrap();
        g.insert_directed(c, y).unwrap();

        let mb = g.markov_blanket_nodes(t).unwrap();
        assert_eq!(mb, vec![a, b, y, c]);
        assert!(!mb.contains(&t));
    }

    #[test]
    fn markov_blanket_of_root_includes_child_and_spouse() {
        let mut g = Dag::with_variables(3);
        let a = DenseNodeId::from_raw(0);
        let b = DenseNodeId::from_raw(1);
        let y = DenseNodeId::from_raw(2);
        g.insert_directed(a, y).unwrap();
        g.insert_directed(b, y).unwrap();
        assert_eq!(g.markov_blanket_nodes(y).unwrap(), vec![a, b]);
        assert_eq!(g.markov_blanket_nodes(a).unwrap(), vec![b, y]);
    }
}
