//! Acyclic directed mixed graphs (ADMGs): directed + bidirected edges (DESIGN.md §6.2).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::VariableId;

use crate::algo::{bfs_reaches, is_dag};
use crate::error::GraphError;
use crate::types::{DenseNodeId, MarkedEdge, NodeRef};
use crate::workspace::{BitSet, GraphWorkspace};

/// ADMG: directed edges and bidirected (latent-confounder) edges; no directed cycles.
#[derive(Clone, Debug)]
pub struct Admg {
    nodes: Vec<NodeRef>,
    /// Outgoing children (directed).
    children: Vec<Vec<DenseNodeId>>,
    /// Incoming parents (directed).
    parents: Vec<Vec<DenseNodeId>>,
    /// Bidirected neighbors (symmetric adjacency).
    bidirected: Vec<Vec<DenseNodeId>>,
    insert_ws: GraphWorkspace,
}

impl Admg {
    /// Empty ADMG.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            children: Vec::new(),
            parents: Vec::new(),
            bidirected: Vec::new(),
            insert_ws: GraphWorkspace::default(),
        }
    }

    /// One static node per variable `0..n`.
    #[must_use]
    pub fn with_variables(n: u32) -> Self {
        let mut g = Self::empty();
        for i in 0..n {
            let _ = g.add_node(NodeRef::Static(VariableId::from_raw(i)));
        }
        g
    }

    /// Number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.nodes.is_empty()
    }

    /// Node refs in dense order.
    #[must_use]
    pub fn nodes(&self) -> &[NodeRef] {
        &self.nodes
    }

    /// Add a static node.
    ///
    /// # Errors
    ///
    /// Non-static node or capacity overflow.
    pub fn add_node(&mut self, node: NodeRef) -> Result<DenseNodeId, GraphError> {
        if !matches!(node, NodeRef::Static(_)) {
            return Err(GraphError::InvalidEndpoints { message: "Admg accepts only Static nodes" });
        }
        let id = u32::try_from(self.nodes.len()).map_err(|_| GraphError::TooManyNodes)?;
        self.nodes.push(node);
        self.children.push(Vec::new());
        self.parents.push(Vec::new());
        self.bidirected.push(Vec::new());
        Ok(DenseNodeId::from_raw(id))
    }

    fn validate_node(&self, id: DenseNodeId) -> Result<(), GraphError> {
        if id.as_usize() >= self.node_count() {
            return Err(GraphError::UnknownNode { id: id.raw() });
        }
        Ok(())
    }

    /// Validate node for public APIs.
    pub(crate) fn validate_node_pub(&self, id: DenseNodeId) -> Result<(), GraphError> {
        self.validate_node(id)
    }

    /// Insert directed edge `from -> to` if it preserves directed acyclicity.
    ///
    /// # Errors
    ///
    /// Unknown nodes, duplicates, or directed cycles.
    pub fn insert_directed(
        &mut self,
        from: DenseNodeId,
        to: DenseNodeId,
    ) -> Result<(), GraphError> {
        self.validate_node(from)?;
        self.validate_node(to)?;
        if from == to {
            return Err(GraphError::InvalidEndpoints {
                message: "Admg rejects directed self-loops",
            });
        }
        if self.children[from.as_usize()].contains(&to) {
            return Err(GraphError::DuplicateEdge { from: from.raw(), to: to.raw() });
        }
        let mut ws = core::mem::take(&mut self.insert_ws);
        let cycle = self.reaches_with(to, from, &mut ws);
        self.insert_ws = ws;
        if cycle {
            return Err(GraphError::Cycle { from: from.raw(), to: to.raw() });
        }
        self.children[from.as_usize()].push(to);
        self.parents[to.as_usize()].push(from);
        Ok(())
    }

    /// Insert bidirected edge `a ↔ b` (latent confounder).
    ///
    /// # Errors
    ///
    /// Unknown nodes, self-loop, or duplicate.
    pub fn insert_bidirected(&mut self, a: DenseNodeId, b: DenseNodeId) -> Result<(), GraphError> {
        self.validate_node(a)?;
        self.validate_node(b)?;
        if a == b {
            return Err(GraphError::InvalidEndpoints {
                message: "Admg rejects bidirected self-loops",
            });
        }
        if self.bidirected[a.as_usize()].contains(&b) {
            return Err(GraphError::DuplicateEdge { from: a.raw(), to: b.raw() });
        }
        self.bidirected[a.as_usize()].push(b);
        self.bidirected[b.as_usize()].push(a);
        Ok(())
    }

    /// Insert a marked edge if ADMG-legal.
    ///
    /// # Errors
    ///
    /// Illegal marks or insert failures.
    pub fn insert_marked(&mut self, edge: MarkedEdge) -> Result<(), GraphError> {
        if !edge.is_admg_legal() {
            return Err(GraphError::InvalidEndpoints {
                message: "Admg allows only directed or bidirected edges",
            });
        }
        if edge.is_bidirected() {
            self.insert_bidirected(edge.a, edge.b)
        } else if let Some((p, c)) = edge.parent_child() {
            self.insert_directed(p, c)
        } else {
            Err(GraphError::InvalidEndpoints { message: "unrecognized ADMG edge" })
        }
    }

    /// Children of `id` (directed).
    #[must_use]
    pub fn children(&self, id: DenseNodeId) -> &[DenseNodeId] {
        &self.children[id.as_usize()]
    }

    /// Parents of `id` (directed).
    #[must_use]
    pub fn parents(&self, id: DenseNodeId) -> &[DenseNodeId] {
        &self.parents[id.as_usize()]
    }

    /// Bidirected neighbors of `id`.
    #[must_use]
    pub fn bidirected_neighbors(&self, id: DenseNodeId) -> &[DenseNodeId] {
        &self.bidirected[id.as_usize()]
    }

    /// Local Markov blanket of `node` on an ADMG: directed blanket (parents ∪
    /// children ∪ spouses) plus bidirected neighbors and their parents.
    ///
    /// This is the adjacency-style blanket used for local conditioning; it is
    /// not a complete m-separation closure over inducing paths.
    ///
    /// # Errors
    ///
    /// Unknown node id.
    pub fn markov_blanket(
        &self,
        node: DenseNodeId,
        out: &mut BitSet,
    ) -> Result<(), GraphError> {
        self.validate_node(node)?;
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
        for &b in self.bidirected_neighbors(node) {
            out.insert(b);
            for &p in self.parents(b) {
                if p != node {
                    out.insert(p);
                }
            }
        }
        Ok(())
    }

    /// Sorted local Markov blanket of `node` (excluding `node`).
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

    /// Whether `from` reaches `to` via directed edges.
    #[must_use]
    pub fn reaches(&self, from: DenseNodeId, to: DenseNodeId) -> bool {
        if from == to {
            return true;
        }
        let mut ws = GraphWorkspace::default();
        self.reaches_with(from, to, &mut ws)
    }

    /// Directed reachability with reusable workspace.
    pub fn reaches_with(
        &self,
        from: DenseNodeId,
        to: DenseNodeId,
        ws: &mut GraphWorkspace,
    ) -> bool {
        bfs_reaches(&self.children, from, to, ws)
    }

    /// Connected components under bidirected edges (districts).
    ///
    /// Returns a district id per dense node (`0..n_districts-1`).
    #[must_use]
    pub fn districts(&self) -> Vec<u32> {
        let n = self.node_count();
        let mut label = vec![u32::MAX; n];
        let mut next = 0u32;
        let mut stack = Vec::new();
        for i in 0..n {
            if label[i] != u32::MAX {
                continue;
            }
            let root = DenseNodeId::from_raw(u32::try_from(i).expect("node fit"));
            label[i] = next;
            stack.push(root);
            while let Some(u) = stack.pop() {
                for &v in self.bidirected_neighbors(u) {
                    let vi = v.as_usize();
                    if label[vi] == u32::MAX {
                        label[vi] = next;
                        stack.push(v);
                    }
                }
            }
            next += 1;
        }
        label
    }

    /// Number of districts.
    #[must_use]
    pub fn district_count(&self) -> usize {
        self.districts().into_iter().max().map_or(0, |m| m as usize + 1)
    }

    /// Validate invariants.
    ///
    /// # Errors
    ///
    /// Directed cycle detected.
    pub fn validate(&self) -> Result<(), GraphError> {
        if !is_dag(&self.parents, &self.children) {
            return Err(GraphError::Cycle { from: 0, to: 0 });
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::{Endpoint, MiddleMark};

    #[test]
    fn districts_split_on_bidirected() {
        let mut g = Admg::with_variables(4);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.insert_bidirected(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let d = g.districts();
        assert_eq!(d[1], d[2]);
        assert_ne!(d[0], d[1]);
        assert_ne!(d[3], d[1]);
    }

    #[test]
    fn rejects_directed_cycle() {
        let mut g = Admg::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        assert!(g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(0)).is_err());
    }

    #[test]
    fn rejects_circle_marks() {
        let mut g = Admg::with_variables(2);
        let e = MarkedEdge {
            a: DenseNodeId::from_raw(0),
            b: DenseNodeId::from_raw(1),
            at_a: Endpoint::Circle,
            at_b: Endpoint::Arrow,
            middle: MiddleMark::Empty,
        };
        assert!(g.insert_marked(e).is_err());
    }

    #[test]
    fn markov_blanket_includes_bidirected_neighbors() {
        // A → T ↔ U ← B  ⇒  MB(T) = {A, U, B}
        let mut g = Admg::with_variables(4);
        let a = DenseNodeId::from_raw(0);
        let t = DenseNodeId::from_raw(1);
        let u = DenseNodeId::from_raw(2);
        let b = DenseNodeId::from_raw(3);
        g.insert_directed(a, t).unwrap();
        g.insert_bidirected(t, u).unwrap();
        g.insert_directed(b, u).unwrap();
        assert_eq!(g.markov_blanket_nodes(t).unwrap(), vec![a, u, b]);
    }
}
