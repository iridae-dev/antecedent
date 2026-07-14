//! Indexed DAG storage with acyclicity validation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::VariableId;

use crate::error::GraphError;
use crate::types::{DenseNodeId, MarkedEdge, NodeRef};
use crate::workspace::GraphWorkspace;

/// Static directed acyclic graph over variables.
#[derive(Clone, Debug)]
pub struct Dag {
    nodes: Vec<NodeRef>,
    /// Outgoing children per node.
    children: Vec<Vec<DenseNodeId>>,
    /// Incoming parents per node.
    parents: Vec<Vec<DenseNodeId>>,
    /// Reused by insertion-time acyclicity checks to avoid per-insert allocation.
    insert_ws: GraphWorkspace,
}

impl Dag {
    /// Empty DAG.
    #[must_use]
    pub fn empty() -> Self {
        Self {
            nodes: Vec::new(),
            children: Vec::new(),
            parents: Vec::new(),
            insert_ws: GraphWorkspace::default(),
        }
    }

    /// Build a DAG with one static node per variable `0..n`.
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

    /// Add a static node; returns its dense id.
    ///
    /// # Errors
    ///
    /// [`GraphError::TooManyNodes`] on overflow.
    pub fn add_node(&mut self, node: NodeRef) -> Result<DenseNodeId, GraphError> {
        if !matches!(node, NodeRef::Static(_)) {
            return Err(GraphError::InvalidEndpoints { message: "Dag accepts only Static nodes" });
        }
        let id = u32::try_from(self.nodes.len()).map_err(|_| GraphError::TooManyNodes)?;
        self.nodes.push(node);
        self.children.push(Vec::new());
        self.parents.push(Vec::new());
        Ok(DenseNodeId::from_raw(id))
    }

    /// Insert a directed edge `from -> to` if it preserves acyclicity.
    ///
    /// # Errors
    ///
    /// Unknown nodes, duplicates, or cycles.
    pub fn insert_directed(
        &mut self,
        from: DenseNodeId,
        to: DenseNodeId,
    ) -> Result<(), GraphError> {
        self.validate_node(from)?;
        self.validate_node(to)?;
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

    /// Push an edge without duplicate/acyclicity checks; the caller guarantees
    /// both nodes exist and the edge preserves invariants.
    pub(crate) fn insert_directed_unchecked(&mut self, from: DenseNodeId, to: DenseNodeId) {
        self.children[from.as_usize()].push(to);
        self.parents[to.as_usize()].push(from);
    }

    /// Remove a directed edge if present.
    pub fn remove_directed(&mut self, from: DenseNodeId, to: DenseNodeId) {
        if from.as_usize() >= self.node_count() || to.as_usize() >= self.node_count() {
            return;
        }
        self.children[from.as_usize()].retain(|c| *c != to);
        self.parents[to.as_usize()].retain(|p| *p != from);
    }

    /// Children of `id`.
    #[must_use]
    pub fn children(&self, id: DenseNodeId) -> &[DenseNodeId] {
        &self.children[id.as_usize()]
    }

    /// Parents of `id`.
    #[must_use]
    pub fn parents(&self, id: DenseNodeId) -> &[DenseNodeId] {
        &self.parents[id.as_usize()]
    }

    /// Whether `from` can reach `to` via directed edges.
    #[must_use]
    pub fn reaches(&self, from: DenseNodeId, to: DenseNodeId) -> bool {
        if from == to {
            return true;
        }
        let mut ws = GraphWorkspace::default();
        self.reaches_with(from, to, &mut ws)
    }

    /// Reachability using a reusable workspace.
    pub fn reaches_with(
        &self,
        from: DenseNodeId,
        to: DenseNodeId,
        ws: &mut GraphWorkspace,
    ) -> bool {
        if from == to {
            return true;
        }
        ws.prepare(self.node_count());
        ws.frontier.push(from);
        ws.visited.insert(from);
        while let Some(n) = ws.frontier.pop() {
            for &c in self.children(n) {
                if c == to {
                    return true;
                }
                if !ws.visited.contains(c) {
                    ws.visited.insert(c);
                    ws.frontier.push(c);
                }
            }
        }
        false
    }

    /// Topological order (Kahn). Returns `None` if a cycle slipped in.
    #[must_use]
    pub fn topological_order(&self) -> Option<Vec<DenseNodeId>> {
        let n = self.node_count();
        let mut indeg = vec![0u32; n];
        for (i, parents) in self.parents.iter().enumerate() {
            indeg[i] = u32::try_from(parents.len()).ok()?;
        }
        let mut q: Vec<DenseNodeId> = indeg
            .iter()
            .enumerate()
            .filter(|&(_, &d)| d == 0)
            .map(|(i, _)| DenseNodeId::from_raw(u32::try_from(i).expect("node fit")))
            .collect();
        let mut order = Vec::with_capacity(n);
        while let Some(u) = q.pop() {
            order.push(u);
            for &v in self.children(u) {
                indeg[v.as_usize()] -= 1;
                if indeg[v.as_usize()] == 0 {
                    q.push(v);
                }
            }
        }
        (order.len() == n).then_some(order)
    }

    /// Validate graph invariants.
    ///
    /// # Errors
    ///
    /// Cycle detected.
    pub fn validate(&self) -> Result<(), GraphError> {
        if self.topological_order().is_none() {
            return Err(GraphError::Cycle { from: 0, to: 0 });
        }
        Ok(())
    }

    fn validate_node(&self, id: DenseNodeId) -> Result<(), GraphError> {
        if id.as_usize() >= self.node_count() {
            Err(GraphError::UnknownNode { id: id.raw() })
        } else {
            Ok(())
        }
    }

    /// Iterate directed edges as marked edges.
    pub fn edges(&self) -> impl Iterator<Item = MarkedEdge> + '_ {
        self.children.iter().enumerate().flat_map(|(i, kids)| {
            let from = DenseNodeId::from_raw(u32::try_from(i).expect("node fit"));
            kids.iter().map(move |&to| MarkedEdge::directed(from, to))
        })
    }

    /// Enumerate simple directed paths from `from` to `to` (inclusive endpoints).
    ///
    /// Bounded by `max_paths` and `max_len` (number of nodes on the path).
    ///
    /// # Errors
    ///
    /// Unknown nodes.
    pub fn directed_paths(
        &self,
        from: DenseNodeId,
        to: DenseNodeId,
        max_paths: usize,
        max_len: usize,
    ) -> Result<Vec<Vec<DenseNodeId>>, GraphError> {
        self.validate_node(from)?;
        self.validate_node(to)?;
        let mut out = Vec::new();
        if max_paths == 0 || max_len == 0 {
            return Ok(out);
        }
        let mut stack = vec![vec![from]];
        while let Some(path) = stack.pop() {
            if out.len() >= max_paths {
                break;
            }
            let last = *path.last().expect("nonempty");
            if path.len() > 1 && last == to {
                out.push(path);
                continue;
            }
            if last == to && path.len() == 1 {
                out.push(path);
                continue;
            }
            if path.len() >= max_len {
                continue;
            }
            for &c in self.children(last) {
                if path.contains(&c) {
                    continue;
                }
                let mut next = path.clone();
                next.push(c);
                stack.push(next);
            }
        }
        Ok(out)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_cycles() {
        let mut g = Dag::with_variables(3);
        let a = DenseNodeId::from_raw(0);
        let b = DenseNodeId::from_raw(1);
        let c = DenseNodeId::from_raw(2);
        g.insert_directed(a, b).unwrap();
        g.insert_directed(b, c).unwrap();
        assert!(matches!(g.insert_directed(c, a), Err(GraphError::Cycle { .. })));
    }

    #[test]
    fn topological_order_respects_edges() {
        let mut g = Dag::with_variables(3);
        g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        let order = g.topological_order().unwrap();
        let pos = |id: u32| order.iter().position(|n| n.raw() == id).unwrap();
        assert!(pos(0) < pos(1) && pos(1) < pos(2));
    }

    #[test]
    fn traversal_workspace_reuses_frontier_capacity() {
        let mut dag = Dag::with_variables(1_000);
        for i in 0..999 {
            dag.insert_directed(DenseNodeId::from_raw(i), DenseNodeId::from_raw(i + 1)).unwrap();
        }
        let mut ws = GraphWorkspace::default();
        assert!(dag.reaches_with(DenseNodeId::from_raw(0), DenseNodeId::from_raw(999), &mut ws));
        let ptr = ws.frontier.as_ptr();
        let cap = ws.frontier.capacity();
        for _ in 0..50 {
            assert!(dag.reaches_with(
                DenseNodeId::from_raw(0),
                DenseNodeId::from_raw(999),
                &mut ws
            ));
            assert_eq!(ws.frontier.as_ptr(), ptr);
            assert_eq!(ws.frontier.capacity(), cap);
        }
    }
}
