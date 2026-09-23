//! Acyclic directed mixed graphs (ADMGs): directed + bidirected edges.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::VariableId;

use crate::algo::{bfs_reaches, is_dag, kahn_order};
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

    /// Schema-aligned ADMG with named directed edges (`VariableId` raw == dense id).
    ///
    /// # Errors
    ///
    /// Unknown names, duplicate edges, or directed cycles.
    pub fn from_named_edges(
        schema: &antecedent_core::CausalSchema,
        edges: &[(&str, &str)],
    ) -> Result<Self, GraphError> {
        let n = crate::named::schema_node_count(schema)?;
        let mut g = Self::with_variables(n);
        for &(from_name, to_name) in edges {
            let (from, to) = crate::named::resolve_named_edge(schema, from_name, to_name)?;
            g.insert_directed(from, to)?;
        }
        Ok(g)
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
        if !node.is_static_graph_node() {
            return Err(GraphError::InvalidEndpoints {
                message: "Admg accepts only Static or Unfolded nodes",
            });
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

    /// Children of `id` (directed; empty for an id outside the graph).
    #[must_use]
    pub fn children(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.children.get(id.as_usize()).map_or(&[], Vec::as_slice)
    }

    /// Parents of `id` (directed; empty for an id outside the graph).
    #[must_use]
    pub fn parents(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.parents.get(id.as_usize()).map_or(&[], Vec::as_slice)
    }

    /// Bidirected neighbors of `id` (empty for an id outside the graph).
    #[must_use]
    pub fn bidirected_neighbors(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.bidirected.get(id.as_usize()).map_or(&[], Vec::as_slice)
    }

    /// Whether any bidirected edge is present.
    #[must_use]
    pub fn has_bidirected(&self) -> bool {
        self.bidirected.iter().any(|neighbors| !neighbors.is_empty())
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

    /// Collect directed descendants of `nodes` (including `nodes`) into `out`.
    ///
    /// Walks [`Self::children`] only. Bidirected edges are ignored.
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

    /// Connected components under bidirected edges (districts).
    ///
    /// Returns a district id per dense node (`0..n_districts-1`).
    #[must_use]
    pub fn districts(&self) -> Vec<u32> {
        let mut label = Vec::new();
        self.districts_into(None, &mut label, &mut Vec::new());
        label
    }

    /// Districts of the subgraph induced on `within`: bidirected-connected components using
    /// only nodes of `within`. Returns a district id per dense node; nodes outside `within`
    /// carry `u32::MAX`.
    #[must_use]
    pub fn districts_within(&self, within: &BitSet) -> Vec<u32> {
        let mut label = Vec::new();
        self.districts_into(Some(within), &mut label, &mut Vec::new());
        label
    }

    /// Allocation-reusing core of [`Self::districts`] / [`Self::districts_within`]; returns
    /// the number of districts.
    pub(crate) fn districts_into(
        &self,
        within: Option<&BitSet>,
        label: &mut Vec<u32>,
        stack: &mut Vec<DenseNodeId>,
    ) -> u32 {
        let n = self.node_count();
        let inside = |v: DenseNodeId| within.is_none_or(|w| w.contains(v));
        label.clear();
        label.resize(n, u32::MAX);
        stack.clear();
        let mut next = 0u32;
        for i in 0..n {
            let root = DenseNodeId::from_raw(u32::try_from(i).expect("node fit"));
            if label[i] != u32::MAX || !inside(root) {
                continue;
            }
            label[i] = next;
            stack.push(root);
            while let Some(u) = stack.pop() {
                for &v in self.bidirected_neighbors(u) {
                    let vi = v.as_usize();
                    if label[vi] == u32::MAX && inside(v) {
                        label[vi] = next;
                        stack.push(v);
                    }
                }
            }
            next += 1;
        }
        next
    }

    /// Topological order of the directed edges (Kahn, stack discipline).
    ///
    /// `None` only if a directed cycle slipped in; insertion refuses cycles.
    #[must_use]
    pub fn topological_order(&self) -> Option<Vec<DenseNodeId>> {
        kahn_order(&self.parents, &self.children)
    }

    /// Districts (bidirected-connected components) of the subgraph induced by `nodes`.
    ///
    /// Unlike [`Self::districts`], only edges with both endpoints in `nodes` connect, so
    /// nodes that a bidirected edge joins through a removed node fall apart. Components come
    /// out in ascending order of their smallest member.
    #[must_use]
    pub fn district_components_within(&self, nodes: &BitSet) -> Vec<BitSet> {
        let n = self.node_count();
        let mut label = Vec::new();
        let count = self.districts_into(Some(nodes), &mut label, &mut Vec::new());
        // Labels are handed out in ascending order of each component's smallest member.
        let mut comps = vec![BitSet::with_len(n); count as usize];
        for id in nodes.to_dense_ids() {
            comps[label[id.as_usize()] as usize].insert(id);
        }
        comps
    }

    /// Directed ancestors of `seeds` (seeds included) inside `active`.
    ///
    /// Seeds outside `active` are ignored. With `bar_x` set, the nodes in it have no
    /// incoming edges, which is the ancestry of the graph with all edges into `bar_x`
    /// removed.
    #[must_use]
    pub fn ancestors_within(
        &self,
        seeds: &BitSet,
        active: &BitSet,
        bar_x: Option<&BitSet>,
        ws: &mut GraphWorkspace,
    ) -> BitSet {
        let n = self.node_count();
        let mut out = BitSet::with_len(n);
        ws.prepare(n);
        for id in seeds.to_dense_ids() {
            if !active.contains(id) {
                continue;
            }
            if !out.contains(id) {
                out.insert(id);
                ws.frontier.push(id);
            }
        }
        while let Some(u) = ws.frontier.pop() {
            if bar_x.is_some_and(|bx| bx.contains(u)) {
                continue;
            }
            for &p in self.parents(u) {
                if !active.contains(p) || out.contains(p) {
                    continue;
                }
                out.insert(p);
                ws.frontier.push(p);
            }
        }
        out
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
        assert!(g.has_bidirected());
        let dag_shaped = Admg::with_variables(2);
        assert!(!dag_shaped.has_bidirected());
    }

    fn set(n: usize, members: &[u32]) -> BitSet {
        let mut b = BitSet::with_len(n);
        for &m in members {
            b.insert(DenseNodeId::from_raw(m));
        }
        b
    }

    #[test]
    fn district_components_within_only_join_through_present_nodes() {
        // 0 <-> 1 <-> 2: dropping 1 separates 0 from 2.
        let mut g = Admg::with_variables(3);
        g.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.insert_bidirected(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let all = g.district_components_within(&set(3, &[0, 1, 2]));
        assert_eq!(all.len(), 1);
        assert!(all[0].equal_set(&set(3, &[0, 1, 2])));
        let split = g.district_components_within(&set(3, &[0, 2]));
        assert_eq!(split.len(), 2);
        assert!(split[0].equal_set(&set(3, &[0])));
        assert!(split[1].equal_set(&set(3, &[2])));
    }

    #[test]
    fn ancestors_within_respects_active_set_and_bar_x() {
        // 0 -> 1 -> 2 -> 3.
        let mut g = Admg::with_variables(4);
        for (a, b) in [(0, 1), (1, 2), (2, 3)] {
            g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let mut ws = GraphWorkspace::default();
        let seeds = set(4, &[3]);
        let full = set(4, &[0, 1, 2, 3]);
        assert!(g.ancestors_within(&seeds, &full, None, &mut ws).equal_set(&full));
        // Node 1 is absent, so 0 is no longer reachable from 3.
        let holed = set(4, &[0, 2, 3]);
        assert!(g.ancestors_within(&seeds, &holed, None, &mut ws).equal_set(&set(4, &[2, 3])));
        // Edges into 2 removed: ancestry stops at 2.
        let bar = set(4, &[2]);
        assert!(g.ancestors_within(&seeds, &full, Some(&bar), &mut ws).equal_set(&set(4, &[2, 3])));
    }

    #[test]
    fn topological_order_places_parents_first() {
        let mut g = Admg::with_variables(4);
        for (a, b) in [(2, 0), (0, 1), (2, 3)] {
            g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        let order = g.topological_order().unwrap();
        let pos = |v: u32| order.iter().position(|&d| d.raw() == v).unwrap();
        assert_eq!(order.len(), 4);
        assert!(pos(2) < pos(0) && pos(0) < pos(1) && pos(2) < pos(3));
    }

    #[test]
    fn rejects_directed_cycle() {
        let mut g = Admg::with_variables(2);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        assert!(g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(0)).is_err());
    }

    #[test]
    fn descendants_follow_directed_edges_only() {
        let mut g = Admg::with_variables(3);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        let mut out = BitSet::default();
        let mut ws = GraphWorkspace::default();
        g.descendants_of(&[DenseNodeId::from_raw(0)], &mut out, &mut ws);
        assert!(out.contains(DenseNodeId::from_raw(0)));
        assert!(out.contains(DenseNodeId::from_raw(1)));
        assert!(
            !out.contains(DenseNodeId::from_raw(2)),
            "bidirected neighbor must not become a directed descendant"
        );
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
}
