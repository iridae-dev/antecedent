//! Acyclic directed mixed graphs (ADMGs): directed + bidirected edges.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::VariableId;

use crate::algo::{closure_into, components_by, is_dag};
use crate::directed::{DirectedCore, same_rows_as_sets, sorted_row_pairs};
use crate::error::GraphError;
use crate::types::{DenseNodeId, MarkedEdge, NodeRef};
use crate::workspace::{BitSet, GraphWorkspace};

/// How [`Admg::copy_filtered`] labels the copied nodes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CopyLabels {
    /// Keep each source node's label.
    Preserve,
    /// Static labels `0..n`, plus `extra` trailing static nodes.
    StaticIndex {
        /// Trailing static nodes beyond the source node count.
        extra: u32,
    },
}

/// ADMG: directed edges and bidirected (latent-confounder) edges; no directed cycles.
///
/// Directed nodes and edges are the shared directed core a [`Dag`](crate::Dag)
/// and a [`TemporalDag`](crate::TemporalDag) also wrap. This type adds the
/// bidirected adjacency and the ADMG endpoint checks.
#[derive(Clone, Debug)]
pub struct Admg {
    directed: DirectedCore,
    /// Bidirected neighbors (symmetric adjacency).
    bidirected: Vec<Vec<DenseNodeId>>,
}

impl PartialEq for Admg {
    fn eq(&self, other: &Self) -> bool {
        self.directed.same_directed(&other.directed)
            && same_rows_as_sets(&self.bidirected, &other.bidirected)
    }
}

impl Eq for Admg {}

impl Admg {
    /// Empty ADMG.
    #[must_use]
    pub fn empty() -> Self {
        Self { directed: DirectedCore::new(), bidirected: Vec::new() }
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

    /// Copy nodes and the edges the predicates keep.
    ///
    /// `keep_bidirected` is the bidirected rule. Callers that drop latent
    /// arrows at an intervention pass a predicate that rejects those endpoints.
    /// Callers that leave confounding in place pass a predicate that accepts
    /// every pair. The two are not interchangeable.
    ///
    /// # Errors
    ///
    /// Node capacity, or an insertion the source graph itself would reject.
    pub fn copy_filtered(
        &self,
        labels: CopyLabels,
        mut keep_directed: impl FnMut(DenseNodeId, DenseNodeId) -> bool,
        mut keep_bidirected: impl FnMut(DenseNodeId, DenseNodeId) -> bool,
    ) -> Result<Self, GraphError> {
        let mut out = match labels {
            CopyLabels::Preserve => {
                let mut out = Self::empty();
                for node in self.nodes() {
                    out.add_node(*node)?;
                }
                out
            }
            CopyLabels::StaticIndex { extra } => {
                let n = u32::try_from(self.node_count()).map_err(|_| GraphError::TooManyNodes)?;
                let total = n.checked_add(extra).ok_or(GraphError::TooManyNodes)?;
                Self::with_variables(total)
            }
        };
        for i in 0..self.node_count() {
            let from = DenseNodeId::from_index(i);
            for &to in self.children(from) {
                if keep_directed(from, to) {
                    // Dropping directed edges of an acyclic graph cannot create a cycle.
                    out.directed.push_unchecked(from, to);
                }
            }
            for &to in self.bidirected_neighbors(from) {
                if to.raw() > from.raw() && keep_bidirected(from, to) {
                    out.insert_bidirected(from, to)?;
                }
            }
        }
        Ok(out)
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
        let mut pending = Vec::with_capacity(edges.len());
        for &(from_name, to_name) in edges {
            match crate::named::resolve_named_edge(schema, from_name, to_name) {
                Ok(pair) => pending.push(pair),
                Err(err) => {
                    g.insert_directed_edges(&pending)?;
                    return Err(err);
                }
            }
        }
        g.insert_directed_edges(&pending)?;
        Ok(g)
    }

    /// Insert directed edges, using one acyclicity check when every edge is new.
    ///
    /// A self-loop, duplicate, or cycle takes [`Self::insert_directed`] instead,
    /// so the error and the edges kept before it match inserting one at a time.
    /// An ADMG self-loop is [`GraphError::InvalidEndpoints`], not a cycle report.
    pub(crate) fn insert_directed_edges(
        &mut self,
        edges: &[(DenseNodeId, DenseNodeId)],
    ) -> Result<(), GraphError> {
        if edges.iter().all(|&(from, to)| from != to) && self.directed.accepts_new_edges(edges) {
            self.directed.push_edges(edges);
            return Ok(());
        }
        for &(from, to) in edges {
            self.insert_directed(from, to)?;
        }
        Ok(())
    }

    /// The ADMG with `dag`'s nodes (labels preserved) and directed edges, no bidirected edges.
    ///
    /// Node labels and directed adjacency are shared with `dag` until either
    /// graph adds a node or a directed edge. The directed part is checked with
    /// [`Self::validate`]; a DAG from the checked constructors passes that check
    /// and the returned edges are the DAG's edges.
    ///
    /// # Errors
    ///
    /// Node capacity, or a directed cycle.
    pub fn from_dag(dag: &crate::dag::Dag) -> Result<Self, GraphError> {
        // The DAG already accepted every directed edge. Sharing the node labels
        // and the adjacency is a snapshot: a later `add_node` on either graph
        // copies its own labels. [`Self::validate`] still checks the directed
        // part; [`crate::dag::Dag`] construction is what accepted the edges.
        let admg = Self {
            directed: dag.directed_core().clone(),
            bidirected: vec![Vec::new(); dag.node_count()],
        };
        admg.validate()?;
        Ok(admg)
    }

    /// The DAG of this ADMG's directed edges over the same nodes (labels preserved);
    /// bidirected edges are dropped.
    ///
    /// # Errors
    ///
    /// Node capacity (the directed part of an ADMG is acyclic by construction).
    pub fn directed_part(&self) -> Result<crate::dag::Dag, GraphError> {
        Ok(crate::dag::Dag::from_directed(self.directed.clone()))
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
        let id = self.directed.push_labeled(node)?;
        self.bidirected.push(Vec::new());
        Ok(id)
    }

    pub(crate) fn validate_node(&self, id: DenseNodeId) -> Result<(), GraphError> {
        if id.as_usize() >= self.node_count() {
            return Err(GraphError::UnknownNode { id: id.raw() });
        }
        Ok(())
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
        self.directed.try_insert(from, to)
    }

    pub(crate) fn insert_directed_temporal(
        &mut self,
        from: DenseNodeId,
        to: DenseNodeId,
        ctx: &antecedent_core::ExecutionContext,
    ) -> Result<(), GraphError> {
        self.validate_node(from)?;
        self.validate_node(to)?;
        if from == to {
            return Err(GraphError::InvalidEndpoints {
                message: "Admg rejects directed self-loops",
            });
        }
        self.directed.try_insert_temporal(from, to, ctx)
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

    /// Directed parent lists, one per node, for separation.
    pub(crate) fn parent_rows(&self) -> &[Vec<DenseNodeId>] {
        self.directed.parent_adj()
    }

    /// Directed child lists, one per node, for separation.
    pub(crate) fn child_rows(&self) -> &[Vec<DenseNodeId>] {
        self.directed.child_adj()
    }

    /// Bidirected neighbor lists, one per node, for separation.
    pub(crate) fn bidirected_rows(&self) -> &[Vec<DenseNodeId>] {
        &self.bidirected
    }

    /// Children of `id` (directed; empty for an id outside the graph).
    #[must_use]
    pub fn children(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.directed.children_of(id)
    }

    /// Parents of `id` (directed; empty for an id outside the graph).
    #[must_use]
    pub fn parents(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.directed.parents_of(id)
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
        self.directed.reaches(from, to)
    }

    /// Directed reachability with reusable workspace.
    pub fn reaches_with(
        &self,
        from: DenseNodeId,
        to: DenseNodeId,
        ws: &mut GraphWorkspace,
    ) -> bool {
        self.directed.reaches_with(from, to, ws)
    }

    /// Collect directed descendants of `nodes` (including `nodes`) into `out`.
    ///
    /// Walks [`Self::children`] only. Bidirected edges are ignored.
    pub fn descendants_of(&self, nodes: &[DenseNodeId], out: &mut BitSet, ws: &mut GraphWorkspace) {
        closure_into(
            self.node_count(),
            |u| self.children(u),
            nodes.iter().copied(),
            out,
            ws,
            |_, _| true,
        );
    }

    /// Collect all ancestors of `nodes` (including `nodes`) into `out`.
    ///
    /// Walks [`Self::parents`] only. Bidirected edges are ignored.
    pub fn ancestors_of(&self, nodes: &[DenseNodeId], out: &mut BitSet, ws: &mut GraphWorkspace) {
        closure_into(
            self.node_count(),
            |u| self.parents(u),
            nodes.iter().copied(),
            out,
            ws,
            |_, _| true,
        );
    }

    /// Descendants of `seeds` that stay inside `active`, with nodes of `cut` reached but not
    /// expanded (the paths through a cut node end there). Seeds outside `active` are dropped.
    #[must_use]
    pub fn descendants_within(
        &self,
        seeds: &BitSet,
        active: &BitSet,
        cut: Option<&BitSet>,
        ws: &mut GraphWorkspace,
    ) -> BitSet {
        let mut out = BitSet::with_len(self.node_count());
        closure_into(
            self.node_count(),
            |u| self.children(u),
            seeds.iter().filter(|&id| active.contains(id)),
            &mut out,
            ws,
            |u, c| !cut.is_some_and(|cut| cut.contains(u)) && active.contains(c),
        );
        out
    }

    /// Directed edges as raw `(parent, child)` dense-id pairs in ascending order.
    #[must_use]
    pub fn directed_edge_pairs_sorted(&self) -> Vec<(u32, u32)> {
        self.directed.edge_pairs_sorted()
    }

    /// Bidirected edges as raw `(lower, higher)` dense-id pairs in ascending order.
    #[must_use]
    pub fn bidirected_edge_pairs_sorted(&self) -> Vec<(u32, u32)> {
        sorted_row_pairs(&self.bidirected, |from, to| to > from)
    }

    /// The variable a static node stands for, or `None` for a non-static or unknown node.
    #[must_use]
    pub fn static_variable(&self, id: DenseNodeId) -> Option<VariableId> {
        match self.nodes().get(id.as_usize()) {
            Some(NodeRef::Static(v)) => Some(*v),
            _ => None,
        }
    }

    /// The dense id of the static node standing for `variable`, if present.
    #[must_use]
    pub fn dense_of_static(&self, variable: VariableId) -> Option<DenseNodeId> {
        self.nodes()
            .iter()
            .position(|node| *node == NodeRef::Static(variable))
            .map(DenseNodeId::from_index)
    }

    /// Directed edges as `(parent, child)` variable pairs in ascending order, or `None`
    /// when any endpoint is not a static node (edges between unfolded nodes have no
    /// variable pair).
    #[must_use]
    pub fn static_directed_edges_sorted(&self) -> Option<Vec<(VariableId, VariableId)>> {
        let mut pairs = Vec::new();
        for (from, kids) in DenseNodeId::all(self.node_count()).zip(self.directed.child_adj()) {
            let from_var = self.static_variable(from)?;
            for &to in kids {
                pairs.push((from_var, self.static_variable(to)?));
            }
        }
        pairs.sort_unstable();
        Some(pairs)
    }

    /// Every dense node id in ascending order.
    #[must_use]
    pub fn node_ids(&self) -> impl DoubleEndedIterator<Item = DenseNodeId> + ExactSizeIterator {
        DenseNodeId::all(self.node_count())
    }

    /// `An(v) \ {v}` for every node `v`, indexed by dense id: the nodes with a directed path
    /// into `v` (the graph is acyclic, so `v` is never its own ancestor).
    ///
    /// One topological sweep, `O(n * (n + e) / 64)`; the owner for callers that need the
    /// ancestry of every node rather than of a seed set.
    #[must_use]
    pub fn strict_ancestor_sets(&self) -> Vec<BitSet> {
        let n = self.node_count();
        let mut ancestors: Vec<BitSet> = (0..n).map(|_| BitSet::with_len(n)).collect();
        let mut pending: Vec<usize> = self.directed.parent_adj().iter().map(Vec::len).collect();
        let mut ready: Vec<DenseNodeId> =
            self.node_ids().filter(|v| pending[v.as_usize()] == 0).collect();
        // One scratch copy of the finished node's ancestors, reused for every node.
        let mut done = BitSet::with_len(n);
        while let Some(v) = ready.pop() {
            done.clear();
            done.union_with(&ancestors[v.as_usize()]);
            for &c in self.children(v) {
                let ci = c.as_usize();
                ancestors[ci].union_with(&done);
                ancestors[ci].insert(v);
                pending[ci] -= 1;
                if pending[ci] == 0 {
                    ready.push(c);
                }
            }
        }
        ancestors
    }

    /// The bidirected-connected closure of `seeds` inside `active`: every active node a
    /// chain of bidirected edges through active nodes joins to a seed (seeds outside
    /// `active` are dropped). The district of each seed in the subgraph induced by `active`.
    #[must_use]
    pub fn district_closure_within(
        &self,
        seeds: &BitSet,
        active: &BitSet,
        ws: &mut GraphWorkspace,
    ) -> BitSet {
        let mut out = BitSet::with_len(self.node_count());
        closure_into(
            self.node_count(),
            |u| self.bidirected_neighbors(u),
            seeds.iter().filter(|&id| active.contains(id)),
            &mut out,
            ws,
            |_, w| active.contains(w),
        );
        out
    }

    /// Weakly connected components: nodes joined by any chain of directed or bidirected
    /// edges, ignoring direction. Each component ascends; components are ordered by their
    /// smallest member.
    #[must_use]
    pub fn weak_components(&self) -> Vec<Vec<DenseNodeId>> {
        components_by(self.node_count(), |i| {
            let id = DenseNodeId::from_index(i);
            self.children(id)
                .iter()
                .chain(self.parents(id))
                .chain(self.bidirected_neighbors(id))
                .map(|v| v.as_usize())
        })
        .into_iter()
        .map(|c| c.into_iter().map(DenseNodeId::from_index).collect())
        .collect()
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

    /// Allocation-reusing core of [`Self::districts`]; returns
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
            let root = DenseNodeId::from_index(i);
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
        self.directed.topological_order()
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
        let mut out = BitSet::with_len(self.node_count());
        closure_into(
            self.node_count(),
            |u| self.parents(u),
            seeds.iter().filter(|&id| active.contains(id)),
            &mut out,
            ws,
            |u, p| !bar_x.is_some_and(|bx| bx.contains(u)) && active.contains(p),
        );
        out
    }

    /// Number of districts.
    #[must_use]
    pub fn district_count(&self) -> usize {
        self.districts_into(None, &mut Vec::new(), &mut Vec::new()) as usize
    }

    /// Validate invariants.
    ///
    /// # Errors
    ///
    /// Directed cycle detected.
    pub fn validate(&self) -> Result<(), GraphError> {
        if !is_dag(self.directed.parent_adj(), self.directed.child_adj()) {
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

    #[test]
    fn from_dag_shares_labels_until_a_node_is_added_and_still_validates() {
        use crate::dag::Dag;
        use crate::types::NodeRef;
        use antecedent_core::VariableId;

        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let admg = Admg::from_dag(&dag).unwrap();
        assert!(std::sync::Arc::ptr_eq(&dag.shared_nodes(), &admg.directed.shared_nodes()));
        let mut looped = Admg::with_variables(2);
        let loop_err = looped.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(0));
        let mut batched = Admg::with_variables(2);
        let batch_err =
            batched.insert_directed_edges(&[(DenseNodeId::from_raw(0), DenseNodeId::from_raw(0))]);
        assert_eq!(loop_err, batch_err);
        assert!(matches!(batch_err, Err(GraphError::InvalidEndpoints { .. })));
        assert_eq!(looped, batched);
        assert_eq!(admg.children(DenseNodeId::from_raw(0)), &[DenseNodeId::from_raw(1)]);
        assert!(admg.validate().is_ok());
        dag.add_node(NodeRef::Static(VariableId::from_raw(9))).unwrap();
        assert_eq!(dag.node_count(), 3);
        assert_eq!(admg.node_count(), 2);
        assert_eq!(admg.nodes()[0], NodeRef::Static(VariableId::from_raw(0)));
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
    fn directed_edge_pairs_are_sorted_raw_ids() {
        let mut g = Admg::with_variables(4);
        for (a, b) in [(2, 3), (0, 3), (0, 1), (1, 2)] {
            g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        g.insert_bidirected(DenseNodeId::from_raw(1), DenseNodeId::from_raw(3)).unwrap();
        assert_eq!(g.directed_edge_pairs_sorted(), vec![(0, 1), (0, 3), (1, 2), (2, 3)]);
    }

    #[test]
    fn directed_edges_and_shared_errors_match_temporal_dag() {
        use crate::temporal::TemporalDag;
        use antecedent_core::{Lag, VariableId};

        let edges = [(0, 1), (1, 2), (0, 2)];
        let mut mixed = Admg::with_variables(3);
        let mut temporal = TemporalDag::empty();
        for i in 0..3 {
            temporal.add_lagged(VariableId::from_raw(i), Lag::from_raw(1)).unwrap();
        }
        for (from, to) in edges {
            let (from, to) = (DenseNodeId::from_raw(from), DenseNodeId::from_raw(to));
            mixed.insert_directed(from, to).unwrap();
            temporal.insert_directed(from, to).unwrap();
        }
        assert_eq!(mixed.directed_edge_pairs_sorted(), temporal.directed_edge_pairs_sorted());
        assert_eq!(mixed.directed_edge_pairs_sorted(), vec![(0, 1), (0, 2), (1, 2)]);
        assert!(mixed.reaches(DenseNodeId::from_raw(0), DenseNodeId::from_raw(0)));
        assert!(temporal.reaches(DenseNodeId::from_raw(0), DenseNodeId::from_raw(0)));
        assert_eq!(
            mixed.reaches(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)),
            temporal.reaches(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2))
        );

        let unknown = DenseNodeId::from_raw(9);
        assert_eq!(
            mixed.insert_directed(unknown, DenseNodeId::from_raw(0)),
            temporal.insert_directed(unknown, DenseNodeId::from_raw(0))
        );
        let dup = (DenseNodeId::from_raw(0), DenseNodeId::from_raw(1));
        assert_eq!(mixed.insert_directed(dup.0, dup.1), temporal.insert_directed(dup.0, dup.1));
        let cycle = (DenseNodeId::from_raw(2), DenseNodeId::from_raw(0));
        assert_eq!(
            mixed.insert_directed(cycle.0, cycle.1),
            temporal.insert_directed(cycle.0, cycle.1)
        );
        // A self-loop is a different error on each type and stays that way.
        let self_loop = DenseNodeId::from_raw(0);
        assert_ne!(
            mixed.insert_directed(self_loop, self_loop),
            temporal.insert_directed(self_loop, self_loop)
        );
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
    fn ancestor_and_descendant_closures_follow_directed_edges_only() {
        // 0 -> 1 -> 2 -> 3, 4 <-> 2 (bidirected edges never carry ancestry).
        let mut g = Admg::with_variables(5);
        for (a, b) in [(0, 1), (1, 2), (2, 3)] {
            g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
        }
        g.insert_bidirected(DenseNodeId::from_raw(4), DenseNodeId::from_raw(2)).unwrap();
        let mut ws = GraphWorkspace::default();
        let mut out = BitSet::with_len(0);
        g.ancestors_of(&[DenseNodeId::from_raw(2)], &mut out, &mut ws);
        assert!(out.equal_set(&set(5, &[0, 1, 2])));
        g.descendants_of(&[DenseNodeId::from_raw(1)], &mut out, &mut ws);
        assert!(out.equal_set(&set(5, &[1, 2, 3])));
        // A cut node is reached but not expanded; seeds outside `active` are dropped.
        let full = BitSet::full(5);
        let seeds = set(5, &[0]);
        let cut = set(5, &[2]);
        assert!(
            g.descendants_within(&seeds, &full, Some(&cut), &mut ws).equal_set(&set(5, &[0, 1, 2]))
        );
        let holed = set(5, &[1, 2, 3]);
        assert!(g.descendants_within(&seeds, &holed, None, &mut ws).equal_set(&set(5, &[])));
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

    /// 0 -> 1 -> 3, 2 -> 3, 0 <-> 2, 4 isolated, 5 <-> 6.
    fn mixed() -> Admg {
        let id = DenseNodeId::from_raw;
        let mut g = Admg::with_variables(7);
        for (a, b) in [(0, 1), (1, 3), (2, 3)] {
            g.insert_directed(id(a), id(b)).unwrap();
        }
        g.insert_bidirected(id(2), id(0)).unwrap();
        g.insert_bidirected(id(5), id(6)).unwrap();
        g
    }

    #[test]
    fn strict_ancestor_sets_match_the_seeded_closure_minus_the_seed() {
        let g = mixed();
        let sets = g.strict_ancestor_sets();
        let mut ws = GraphWorkspace::default();
        for v in g.node_ids() {
            let mut inclusive = BitSet::default();
            g.ancestors_of(&[v], &mut inclusive, &mut ws);
            inclusive.remove(v);
            assert_eq!(sets[v.as_usize()], inclusive, "node {v:?}");
        }
    }

    #[test]
    fn weak_components_and_district_closure_follow_the_right_edge_kinds() {
        let g = mixed();
        let raw = |c: &Vec<DenseNodeId>| c.iter().map(|v| v.raw()).collect::<Vec<_>>();
        let components = g.weak_components();
        assert_eq!(
            components.iter().map(raw).collect::<Vec<_>>(),
            vec![vec![0, 1, 2, 3], vec![4], vec![5, 6]]
        );
        let mut ws = GraphWorkspace::default();
        let active = BitSet::full(7);
        assert_eq!(
            g.district_closure_within(&set(7, &[0]), &active, &mut ws).to_dense_ids(),
            vec![DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)]
        );
        // A seed outside the active set is dropped; an inactive node is never crossed.
        let mut without_two = BitSet::full(7);
        without_two.remove(DenseNodeId::from_raw(2));
        assert_eq!(
            g.district_closure_within(&set(7, &[0, 2]), &without_two, &mut ws).to_dense_ids(),
            vec![DenseNodeId::from_raw(0)]
        );
    }

    #[test]
    fn edge_lists_and_dag_round_trip() {
        let g = mixed();
        assert_eq!(g.directed_edge_pairs_sorted(), vec![(0, 1), (1, 3), (2, 3)]);
        assert_eq!(g.bidirected_edge_pairs_sorted(), vec![(0, 2), (5, 6)]);
        let variable = |raw| antecedent_core::VariableId::from_raw(raw);
        assert_eq!(
            g.static_directed_edges_sorted(),
            Some(vec![
                (variable(0), variable(1)),
                (variable(1), variable(3)),
                (variable(2), variable(3))
            ])
        );
        assert_eq!(g.static_variable(DenseNodeId::from_raw(4)), Some(variable(4)));
        assert_eq!(g.dense_of_static(variable(6)), Some(DenseNodeId::from_raw(6)));
        assert_eq!(g.dense_of_static(variable(9)), None);
        let dag = g.directed_part().unwrap();
        assert_eq!(dag.directed_edge_pairs_sorted(), g.directed_edge_pairs_sorted());
        let back = Admg::from_dag(&dag).unwrap();
        assert_eq!(back.directed_edge_pairs_sorted(), g.directed_edge_pairs_sorted());
        assert!(!back.has_bidirected());
        assert_eq!(back.nodes(), g.nodes());
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
