//! Indexed DAG storage with acyclicity validation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::VariableId;

use crate::directed::DirectedCore;
use crate::error::GraphError;
use crate::types::{DenseNodeId, MarkedEdge, NodeRef};
use crate::workspace::GraphWorkspace;

/// Static directed acyclic graph over variables.
///
/// The node table and directed edges are the shared directed core.
/// This type adds the static-node check.
#[derive(Clone, Debug)]
pub struct Dag {
    directed: DirectedCore,
}

impl PartialEq for Dag {
    fn eq(&self, other: &Self) -> bool {
        self.directed.same_directed(&other.directed)
    }
}

impl Eq for Dag {}

impl Dag {
    /// Empty DAG.
    #[must_use]
    pub fn empty() -> Self {
        Self { directed: DirectedCore::new() }
    }

    /// One [`NodeRef::Unfolded`] node per slot of `indexer`, in dense order.
    ///
    /// # Errors
    ///
    /// Window larger than the node-id space.
    pub fn from_unfolding(indexer: &antecedent_core::TemporalIndexer) -> Result<Self, GraphError> {
        let mut g = Self::empty();
        for dense in 0..indexer.dense_len() {
            let key = indexer
                .key_of(u32::try_from(dense).map_err(|_| GraphError::TooManyNodes)?)
                .map_err(|_| GraphError::TooManyNodes)?;
            g.add_node(NodeRef::Unfolded { variable: key.variable, offset: key.offset })?;
        }
        Ok(g)
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

    /// Copy nodes `0..n` and the directed edges `keep` accepts.
    ///
    /// # Errors
    ///
    /// Node capacity, or an insertion the source graph itself would reject.
    pub fn copy_filtered(
        &self,
        mut keep: impl FnMut(DenseNodeId, DenseNodeId) -> bool,
    ) -> Result<Self, GraphError> {
        let n = u32::try_from(self.node_count()).map_err(|_| GraphError::TooManyNodes)?;
        let mut out = Self::with_variables(n);
        for edge in self.edges() {
            let Some((from, to)) = edge.parent_child() else {
                continue;
            };
            if keep(from, to) {
                // A subset of a DAG's edges is acyclic, and each source edge is copied once.
                out.directed.push_unchecked(from, to);
            }
        }
        Ok(out)
    }

    /// Build a DAG with one static node per schema variable (`VariableId` raw == dense id),
    /// then insert directed edges named by schema variable names.
    ///
    /// # Errors
    ///
    /// Unknown names, duplicate edges, or cycles.
    pub fn from_named_edges(
        schema: &antecedent_core::CausalSchema,
        edges: &[(&str, &str)],
    ) -> Result<Self, GraphError> {
        let n = crate::named::schema_node_count(schema)?;
        let mut g = Self::with_variables(n);
        g.insert_resolved_directed(schema, edges)?;
        Ok(g)
    }

    /// Insert already-resolved directed edges.
    ///
    /// When every edge is new and the result is acyclic, they are appended after
    /// one acyclicity check. Otherwise each edge goes through [`Self::insert_directed`],
    /// so a duplicate or a cycle leaves the same prefix and the same error.
    pub(crate) fn insert_directed_edges(
        &mut self,
        edges: &[(DenseNodeId, DenseNodeId)],
    ) -> Result<(), GraphError> {
        if self.directed.accepts_new_edges(edges) {
            self.directed.push_edges(edges);
            return Ok(());
        }
        for &(from, to) in edges {
            self.insert_directed(from, to)?;
        }
        Ok(())
    }

    /// Resolve names in order, then insert. A later unknown name is reported only
    /// after earlier edges have been inserted, which is the one-edge loop.
    fn insert_resolved_directed(
        &mut self,
        schema: &antecedent_core::CausalSchema,
        edges: &[(&str, &str)],
    ) -> Result<(), GraphError> {
        let mut pending = Vec::with_capacity(edges.len());
        for &(from_name, to_name) in edges {
            match crate::named::resolve_named_edge(schema, from_name, to_name) {
                Ok(pair) => pending.push(pair),
                Err(err) => {
                    self.insert_directed_edges(&pending)?;
                    return Err(err);
                }
            }
        }
        self.insert_directed_edges(&pending)
    }

    /// Number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.directed.node_count()
    }

    /// Whether empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.directed.is_empty()
    }

    /// Node refs in dense order.
    #[must_use]
    pub fn nodes(&self) -> &[NodeRef] {
        self.directed.nodes()
    }

    /// Add a static node; returns its dense id.
    ///
    /// # Errors
    ///
    /// [`GraphError::TooManyNodes`] on overflow.
    pub fn add_node(&mut self, node: NodeRef) -> Result<DenseNodeId, GraphError> {
        if !node.is_static_graph_node() {
            return Err(GraphError::InvalidEndpoints {
                message: "Dag accepts only Static or Unfolded nodes",
            });
        }
        self.directed.push_labeled(node)
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
        self.directed.try_insert(from, to)
    }

    /// Push an edge without duplicate/acyclicity checks; the caller guarantees
    /// both nodes exist and the edge preserves invariants.
    pub(crate) fn insert_directed_unchecked(&mut self, from: DenseNodeId, to: DenseNodeId) {
        self.directed.push_unchecked(from, to);
    }

    /// Node labels. A copy shares the allocation until one side adds a node.
    #[cfg(test)]
    pub(crate) fn shared_nodes(&self) -> Arc<Vec<NodeRef>> {
        self.directed.shared_nodes()
    }

    /// Shared directed tables. Cloning bumps the adjacency refcount.
    pub(crate) fn directed_core(&self) -> &DirectedCore {
        &self.directed
    }

    /// Mutable directed tables. Unfolding uses this after the template proved acyclicity.
    pub(crate) fn directed_mut(&mut self) -> &mut DirectedCore {
        &mut self.directed
    }

    /// A DAG over an already-validated directed core.
    pub(crate) fn from_directed(directed: DirectedCore) -> Self {
        Self { directed }
    }

    /// Directed parent lists, one per node, for separation.
    pub(crate) fn parent_rows(&self) -> &[Vec<DenseNodeId>] {
        self.directed.parent_adj()
    }

    /// Directed child lists, one per node, for separation.
    pub(crate) fn child_rows(&self) -> &[Vec<DenseNodeId>] {
        self.directed.child_adj()
    }

    /// Children of `id` (empty for an id outside the graph).
    #[must_use]
    pub fn children(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.directed.children_of(id)
    }

    /// Parents of `id` (empty for an id outside the graph).
    #[must_use]
    pub fn parents(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.directed.parents_of(id)
    }

    /// Whether `from` can reach `to` via directed edges.
    #[must_use]
    pub fn reaches(&self, from: DenseNodeId, to: DenseNodeId) -> bool {
        self.directed.reaches(from, to)
    }

    /// Reachability using a reusable workspace.
    pub fn reaches_with(
        &self,
        from: DenseNodeId,
        to: DenseNodeId,
        ws: &mut GraphWorkspace,
    ) -> bool {
        self.directed.reaches_with(from, to, ws)
    }

    /// Topological order (Kahn). Returns `None` if a cycle slipped in.
    #[must_use]
    pub fn topological_order(&self) -> Option<Vec<DenseNodeId>> {
        self.directed.topological_order()
    }

    /// Validate graph invariants.
    ///
    /// # Errors
    ///
    /// Cycle detected.
    pub fn validate(&self) -> Result<(), GraphError> {
        if !crate::algo::is_dag(self.directed.parent_adj(), self.directed.child_adj()) {
            return Err(GraphError::Cycle { from: 0, to: 0 });
        }
        Ok(())
    }

    /// Nodes that reach `to` (itself excluded), as a membership mask.
    fn ancestors_mask(&self, to: DenseNodeId) -> Vec<bool> {
        let mut seen = vec![false; self.node_count()];
        let mut queue = vec![to];
        while let Some(v) = queue.pop() {
            for &p in self.parents(v) {
                if !seen[p.as_usize()] {
                    seen[p.as_usize()] = true;
                    queue.push(p);
                }
            }
        }
        seen
    }

    /// Node count of the longest simple directed path starting at `from` that never
    /// enters `avoid`.
    fn longest_path_nodes_avoiding(&self, from: DenseNodeId, avoid: DenseNodeId) -> usize {
        // 0 = unvisited, 1 = on the DFS stack, 2 = finished.
        let mut state = vec![0u8; self.node_count()];
        let mut best = vec![0usize; self.node_count()];
        let mut stack = vec![(from, 0usize)];
        state[from.as_usize()] = 1;
        while let Some(&(v, next)) = stack.last() {
            let kids = self.children(v);
            if let Some(&c) = kids.get(next) {
                stack.last_mut().expect("nonempty").1 += 1;
                if c != avoid && state[c.as_usize()] == 0 {
                    state[c.as_usize()] = 1;
                    stack.push((c, 0));
                }
            } else {
                let deepest = kids
                    .iter()
                    .filter(|&&c| c != avoid && state[c.as_usize()] == 2)
                    .map(|c| best[c.as_usize()])
                    .max()
                    .unwrap_or(0);
                best[v.as_usize()] = deepest + 1;
                state[v.as_usize()] = 2;
                stack.pop();
            }
        }
        best[from.as_usize()]
    }

    fn validate_node(&self, id: DenseNodeId) -> Result<(), GraphError> {
        if id.as_usize() >= self.node_count() {
            Err(GraphError::UnknownNode { id: id.raw() })
        } else {
            Ok(())
        }
    }

    /// Every dense node id in ascending order.
    #[must_use]
    pub fn node_ids(&self) -> impl DoubleEndedIterator<Item = DenseNodeId> + ExactSizeIterator {
        DenseNodeId::all(self.node_count())
    }

    /// Iterate directed edges as marked edges.
    pub fn edges(&self) -> impl Iterator<Item = MarkedEdge> + '_ {
        self.directed.marked_edges()
    }

    /// Directed edges as raw `(parent, child)` dense-id pairs in ascending order.
    #[must_use]
    pub fn directed_edge_pairs_sorted(&self) -> Vec<(u32, u32)> {
        self.directed.edge_pairs_sorted()
    }

    /// Enumerate simple directed paths from `from` to `to` (inclusive endpoints).
    ///
    /// Bounded by `max_paths` and `max_len` (number of nodes on the path).
    ///
    /// The returned set is silently truncated when either bound binds. Callers whose
    /// correctness depends on seeing *every* path — e.g. a recanting-witness test, which
    /// concludes "no witness exists" from the absence of one — must use
    /// [`Self::directed_paths_with_budget`] and fail closed on truncation instead.
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
        self.directed_paths_with_budget(from, to, max_paths, max_len).map(|(paths, _)| paths)
    }

    /// [`Self::directed_paths`] plus a flag reporting whether enumeration was cut short.
    ///
    /// The flag is `true` when `max_paths` stopped the search with candidates still
    /// pending, or when `max_len` pruned a partial path that had not yet reached `to`.
    /// It is deliberately conservative: `true` means "the path set may be incomplete",
    /// never "it is definitely incomplete".
    ///
    /// # Errors
    ///
    /// Unknown nodes.
    pub fn directed_paths_with_budget(
        &self,
        from: DenseNodeId,
        to: DenseNodeId,
        max_paths: usize,
        max_len: usize,
    ) -> Result<(Vec<Vec<DenseNodeId>>, bool), GraphError> {
        self.validate_node(from)?;
        self.validate_node(to)?;
        let mut out = Vec::new();
        if max_paths == 0 || max_len == 0 {
            // A zero budget cannot certify that no path exists.
            return Ok((out, true));
        }
        if from == to {
            out.push(vec![from]);
            return Ok((out, false));
        }
        if max_len == 1 {
            // The start node alone already fills the length budget.
            return Ok((out, true));
        }
        // Only `An(to)` can complete a path; branches outside it are never entered. The
        // search order, the `max_paths` flag and the `max_len` flag are those of the
        // unpruned depth-first enumeration: a skipped branch still counts as a pending
        // sibling, and `max_len` is flagged when any simple path of `max_len` nodes avoids
        // `to`, which one longest-path pass over the nodes reachable from `from` decides.
        let live = self.ancestors_mask(to);
        let mut truncated = self.longest_path_nodes_avoiding(from, to) >= max_len;
        // Every live partial path extends to a completed one unless `max_len` cuts it, so a
        // search that finishes without any cut visits at most `paths * max_len` nodes.
        let visit_budget = max_paths.saturating_mul(max_len).saturating_mul(16);
        let mut visits = 0usize;
        let mut path = vec![from];
        let mut on_path = vec![false; self.node_count()];
        on_path[from.as_usize()] = true;
        // Remaining children of each path node, consumed from the back (the unpruned
        // search pops its stack last-pushed first).
        let mut cursor = vec![self.children(from).len()];
        'dfs: while let Some(&last) = path.last() {
            let kids = self.children(last);
            let top = cursor.len() - 1;
            if cursor[top] == 0 {
                on_path[last.as_usize()] = false;
                path.pop();
                cursor.pop();
                continue;
            }
            cursor[top] -= 1;
            let c = kids[cursor[top]];
            if out.len() >= max_paths {
                truncated = true;
                break 'dfs;
            }
            if on_path[c.as_usize()] {
                continue;
            }
            if c == to {
                let mut done = path.clone();
                done.push(c);
                out.push(done);
                continue;
            }
            if !live[c.as_usize()] || path.len() + 1 >= max_len {
                continue;
            }
            visits += 1;
            if visits > visit_budget {
                truncated = true;
                break 'dfs;
            }
            on_path[c.as_usize()] = true;
            path.push(c);
            cursor.push(self.children(c).len());
        }
        Ok((out, truncated))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn directed_edge_pairs_are_sorted_raw_ids() {
        let mut g = Dag::with_variables(4);
        for (u, v) in [(2, 3), (0, 3), (0, 1), (1, 2)] {
            g.insert_directed(DenseNodeId::from_raw(u), DenseNodeId::from_raw(v)).unwrap();
        }
        assert_eq!(g.directed_edge_pairs_sorted(), vec![(0, 1), (0, 3), (1, 2), (2, 3)]);
        assert!(Dag::with_variables(2).directed_edge_pairs_sorted().is_empty());
    }

    #[test]
    fn directed_paths_reports_max_paths_truncation() {
        // t→c→y, t→w→b→y, t→w→a→y : three directed paths.
        let mut g = Dag::with_variables(6);
        for (u, v) in [(0, 4), (4, 5), (0, 1), (1, 3), (3, 5), (1, 2), (2, 5)] {
            g.insert_directed(DenseNodeId::from_raw(u), DenseNodeId::from_raw(v)).unwrap();
        }
        let (t, y) = (DenseNodeId::from_raw(0), DenseNodeId::from_raw(5));

        let (all, truncated) = g.directed_paths_with_budget(t, y, 64, 16).unwrap();
        assert_eq!(all.len(), 3);
        assert!(!truncated, "a budget that comfortably fits every path must not report truncation");

        for cap in 1..=2 {
            let (paths, truncated) = g.directed_paths_with_budget(t, y, cap, 16).unwrap();
            assert_eq!(paths.len(), cap);
            assert!(truncated, "max_paths={cap} dropped paths but reported none");
        }
    }

    #[test]
    fn directed_paths_reports_max_len_truncation() {
        // Single path 0→1→2→3 needs 4 nodes; max_len=3 prunes it before it reaches the target.
        let mut g = Dag::with_variables(4);
        for (u, v) in [(0, 1), (1, 2), (2, 3)] {
            g.insert_directed(DenseNodeId::from_raw(u), DenseNodeId::from_raw(v)).unwrap();
        }
        let (t, y) = (DenseNodeId::from_raw(0), DenseNodeId::from_raw(3));
        let (paths, truncated) = g.directed_paths_with_budget(t, y, 64, 3).unwrap();
        assert!(paths.is_empty());
        assert!(truncated, "max_len pruned the only path but reported no truncation");

        let (paths, truncated) = g.directed_paths_with_budget(t, y, 64, 4).unwrap();
        assert_eq!(paths.len(), 1);
        assert!(!truncated);
    }

    /// The unpruned enumeration this method replaced, kept as an oracle.
    fn reference_paths(
        g: &Dag,
        from: DenseNodeId,
        to: DenseNodeId,
        max_paths: usize,
        max_len: usize,
    ) -> (Vec<Vec<DenseNodeId>>, bool) {
        let mut out = Vec::new();
        if max_paths == 0 || max_len == 0 {
            return (out, true);
        }
        let mut truncated = false;
        let mut stack = vec![vec![from]];
        while let Some(path) = stack.pop() {
            if out.len() >= max_paths {
                truncated = true;
                break;
            }
            let last = *path.last().unwrap();
            if last == to {
                out.push(path);
                continue;
            }
            if path.len() >= max_len {
                truncated = true;
                continue;
            }
            for &c in g.children(last) {
                if !path.contains(&c) {
                    let mut next = path.clone();
                    next.push(c);
                    stack.push(next);
                }
            }
        }
        (out, truncated)
    }

    #[test]
    fn directed_paths_match_the_unpruned_enumeration() {
        let mut state = 0x9E37_79B9_7F4A_7C15_u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };
        for _ in 0..300 {
            let n = 3 + (next() % 7) as u32;
            let mut g = Dag::with_variables(n);
            for u in 0..n {
                for v in (u + 1)..n {
                    if next() % 100 < 40 {
                        g.insert_directed(DenseNodeId::from_raw(u), DenseNodeId::from_raw(v))
                            .unwrap();
                    }
                }
            }
            for _ in 0..6 {
                let from = DenseNodeId::from_raw(u32::try_from(next() % u64::from(n)).unwrap());
                let to = DenseNodeId::from_raw(u32::try_from(next() % u64::from(n)).unwrap());
                let max_paths = (next() % 6) as usize;
                let max_len = (next() % 8) as usize;
                assert_eq!(
                    g.directed_paths_with_budget(from, to, max_paths, max_len).unwrap(),
                    reference_paths(&g, from, to, max_paths, max_len),
                    "n={n} from={from:?} to={to:?} max_paths={max_paths} max_len={max_len}"
                );
            }
        }
    }

    #[test]
    fn rejects_cycles() {
        let mut g = Dag::with_variables(3);
        let a = DenseNodeId::from_raw(0);
        let b = DenseNodeId::from_raw(1);
        let c = DenseNodeId::from_raw(2);
        g.insert_directed(a, b).unwrap();
        g.insert_directed(b, c).unwrap();
        let err = g.insert_directed(c, a).unwrap_err();
        assert!(matches!(err, GraphError::Cycle { .. }));
        let text = err.to_string();
        assert!(text.contains("condensation"), "{text}");
        assert!(text.contains("sigma_separates"), "{text}");
        assert!(g.children(c).is_empty(), "the closing edge was not inserted");
        assert!(g.topological_order().is_some(), "the chain that remains still has an order");
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

    #[test]
    fn batch_insert_matches_one_edge_at_a_time() {
        let cases: &[&[(u32, u32)]] = &[
            &[(0, 1), (1, 2), (0, 2)],
            &[(0, 1), (1, 2), (2, 0)],
            &[(0, 1), (0, 1)],
            &[(0, 0)],
            &[],
            &[(0, 1), (2, 3), (1, 2)],
        ];
        for edges in cases {
            let mut one = Dag::with_variables(4);
            let one_err = edges.iter().try_for_each(|&(from, to)| {
                one.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
            });
            let mut batch = Dag::with_variables(4);
            let batch_edges: Vec<_> = edges
                .iter()
                .map(|&(from, to)| (DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)))
                .collect();
            let batch_err = batch.insert_directed_edges(&batch_edges);
            assert_eq!(one_err, batch_err, "edges {edges:?}");
            assert_eq!(one, batch, "edges {edges:?}");
        }
    }
}

/// Review-required static DAG artifact (`DirectLiNGAM` and other full-DAG discovery).
#[derive(Clone, Debug)]
pub struct DagReview {
    /// Proposed discovery DAG.
    pub graph: Dag,
    /// Directed edges awaiting explicit acceptance `(from, to)` as [`VariableId`]s.
    pub pending_edges: Arc<[(VariableId, VariableId)]>,
    /// Algorithm id that produced the proposal.
    pub algorithm: Arc<str>,
}

impl DagReview {
    /// Construct a review listing all current edges as pending.
    #[must_use]
    pub fn from_dag(graph: Dag, algorithm: impl Into<Arc<str>>) -> Self {
        let mut pending = Vec::new();
        for e in graph.edges() {
            if let Some((from, to)) = e.parent_child() {
                if let (Some(fv), Some(tv)) =
                    (variable_id_of(&graph, from), variable_id_of(&graph, to))
                {
                    pending.push((fv, tv));
                }
            }
        }
        Self { graph, pending_edges: Arc::from(pending), algorithm: algorithm.into() }
    }

    /// Accept a pending directed edge (no-op if absent).
    #[must_use]
    pub fn accept_edge(mut self, from: VariableId, to: VariableId) -> Self {
        self.pending_edges = crate::types::without_pending(&self.pending_edges, (from, to));
        self
    }

    /// Accept all remaining pending edges.
    #[must_use]
    pub fn accept_all(mut self) -> Self {
        self.pending_edges = Arc::from([]);
        self
    }

    /// Whether all pending edges have been accepted.
    #[must_use]
    pub fn is_complete(&self) -> bool {
        self.pending_edges.is_empty()
    }

    /// Borrow the accepted DAG when review is complete.
    ///
    /// # Errors
    ///
    /// Incomplete review.
    pub fn try_into_dag(self) -> Result<Dag, GraphError> {
        if !self.is_complete() {
            return Err(GraphError::InvalidEndpoints {
                message: "cannot finish DagReview while pending edges remain",
            });
        }
        Ok(self.graph)
    }
}

fn variable_id_of(dag: &Dag, id: DenseNodeId) -> Option<VariableId> {
    match dag.nodes().get(id.as_usize()) {
        Some(NodeRef::Static(v)) => Some(*v),
        _ => None,
    }
}
