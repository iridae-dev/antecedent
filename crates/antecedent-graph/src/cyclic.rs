//! Cyclic directed and mixed graphs.
//!
//! [`CyclicDigraph`] stores directed edges that may form cycles. A reciprocal
//! pair is two directed edges. [`CyclicMixedGraph`] adds symmetric bidirected
//! edges for latent confounding. A bidirected edge is not stored as a directed
//! edge, and a reciprocal pair is not stored as a bidirected edge.
//!
//! These types are not a [`Dag`] or an [`Admg`]. Those constructors still
//! reject directed cycles. Contemporaneous self-loops are rejected here too.
//!
//! [`CyclicDigraph::condensation`] schedules directed strongly connected
//! components, sources before sinks. [`CyclicMixedGraph::condensation`] reads
//! the same directed edges. A bidirected edge is latent confounding and does
//! not join components.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::VariableId;

use crate::algo::insert_sorted;
use crate::directed::{DirectedStore, sorted_row_pairs};
use crate::error::GraphError;
use crate::scc::Condensation;
use crate::types::{DenseNodeId, MarkedEdge, NodeRef};

/// Directed graph that allows cycles.
///
/// Reciprocal edges `from -> to` and `to -> from` are two directed edges.
/// Self-loops are rejected ([`GraphError::ContemporaneousSelfEdge`]): an
/// initial linear family has a zero diagonal. This is not a [`Dag`].
/// [`Dag::insert_directed`] still returns [`GraphError::Cycle`] when an edge
/// would close a directed cycle.
///
/// Dense ids follow node-insertion order. Inserting the same [`NodeRef`]
/// labels in another order assigns different dense ids, so the graphs are not
/// equal. Prefer [`Self::with_variables`] or [`Self::from_named_edges`] when
/// schema variable `i` should be dense id `i`.
///
/// Child and parent lists are kept in ascending dense-id order. [`Eq`] is that
/// edge set plus the node labels, independent of edge insertion order.
/// [`Self::directed_edge_pairs_sorted`] is the canonical edge list.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CyclicDigraph {
    store: DirectedStore,
}

impl CyclicDigraph {
    /// Empty graph.
    #[must_use]
    pub const fn empty() -> Self {
        Self { store: DirectedStore::empty() }
    }

    /// One static node per variable `0..n`, in that dense order.
    #[must_use]
    pub fn with_variables(n: u32) -> Self {
        let mut graph = Self::empty();
        for i in 0..n {
            graph
                .add_node(NodeRef::Static(VariableId::from_raw(i)))
                .expect("variable count fits in a dense id");
        }
        graph
    }

    /// Static nodes for `schema`, then named directed edges.
    ///
    /// Names resolve through the schema (`VariableId` raw == dense id). Every
    /// edge is checked before any edge is inserted, so a failure returns no graph.
    ///
    /// # Errors
    ///
    /// Unknown names, unknown nodes, duplicate edges, or self-loops. A directed
    /// cycle is not an error.
    pub fn from_named_edges(
        schema: &antecedent_core::CausalSchema,
        edges: &[(&str, &str)],
    ) -> Result<Self, GraphError> {
        let n = crate::named::schema_node_count(schema)?;
        let pending = named_directed_edges(schema, n, edges)?;
        let mut graph = Self::with_variables(n);
        for (from, to) in pending {
            graph.insert_directed(from, to)?;
        }
        Ok(graph)
    }

    /// Number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.store.node_count()
    }

    /// Whether the node table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.store.is_empty()
    }

    /// Node labels in dense order.
    #[must_use]
    pub fn nodes(&self) -> &[NodeRef] {
        self.store.nodes()
    }

    /// Every dense node id in ascending order.
    #[must_use]
    pub fn node_ids(&self) -> impl DoubleEndedIterator<Item = DenseNodeId> + ExactSizeIterator {
        DenseNodeId::all(self.node_count())
    }

    /// Add a static or unfolded node. Returns its dense id.
    ///
    /// # Errors
    ///
    /// The node is not a static graph node, or the dense id would not fit in `u32`.
    pub fn add_node(&mut self, node: NodeRef) -> Result<DenseNodeId, GraphError> {
        if !node.is_static_graph_node() {
            return Err(GraphError::InvalidEndpoints {
                message: "CyclicDigraph accepts only Static or Unfolded nodes",
            });
        }
        self.store.push_labeled(node)
    }

    /// Insert directed edge `from -> to`.
    ///
    /// The reciprocal edge may be inserted separately. Nothing in this method
    /// searches for a cycle.
    ///
    /// # Errors
    ///
    /// Unknown nodes, a self-loop, or a duplicate directed edge.
    pub fn insert_directed(
        &mut self,
        from: DenseNodeId,
        to: DenseNodeId,
    ) -> Result<(), GraphError> {
        DirectedStore::known(self.node_count(), from)?;
        DirectedStore::known(self.node_count(), to)?;
        reject_self_loop(self.nodes(), from, to)?;
        self.store.link(from, to)
    }

    /// Directed children of `id` (empty when `id` is outside the graph).
    #[must_use]
    pub fn children(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.store.children_of(id)
    }

    /// Directed parents of `id` (empty when `id` is outside the graph).
    #[must_use]
    pub fn parents(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.store.parents_of(id)
    }

    /// Directed successors of `id`.
    ///
    /// This is the child list. The iterator does not record visits, so a walk
    /// that follows it may return to the same node when a directed cycle is
    /// present.
    #[must_use = "the successor walk is the only cycle-safe traversal"]
    pub fn successors(&self, id: DenseNodeId) -> impl Iterator<Item = DenseNodeId> + '_ {
        self.children(id).iter().copied()
    }

    /// Directed edges as marked edges, parent then child, in dense order.
    pub fn edges(&self) -> impl Iterator<Item = MarkedEdge> + '_ {
        self.store.child_rows().iter().enumerate().flat_map(|(i, kids)| {
            let from = DenseNodeId::from_index(i);
            kids.iter().copied().map(move |to| MarkedEdge::directed(from, to))
        })
    }

    /// Directed edges as `(parent, child)` dense-id pairs in ascending order.
    ///
    /// Swapping insertion order does not change this list. Together with
    /// [`Eq`], it identifies the directed edge set.
    #[must_use]
    pub fn directed_edge_pairs_sorted(&self) -> Vec<(u32, u32)> {
        self.store.edge_pairs_sorted()
    }

    /// Check endpoints, self-loops, duplicate edges, and the parent index.
    ///
    /// Directed cycles are legal.
    ///
    /// # Errors
    ///
    /// An unknown endpoint, a self-loop, a duplicate edge, or adjacency that
    /// does not match the node table.
    pub fn validate(&self) -> Result<(), GraphError> {
        validate_directed(self)
    }

    /// Directed strongly connected components, sources before sinks.
    ///
    /// See [`Condensation`]. In an acyclic graph each node is its own component
    /// and the schedule is the minimum-dense-id topological order.
    #[must_use = "the condensation is the directed solve-block schedule"]
    pub fn condensation(&self) -> Condensation {
        crate::scc::decompose(self.store.child_rows(), self.store.parent_rows())
    }
}

/// Directed cyclic graph plus symmetric bidirected edges.
///
/// Bidirected edges record latent confounding. Inserting `a <-> b` does not
/// insert `a -> b` or `b -> a`. Directed cycles, including a reciprocal pair,
/// stay on the directed side and are not rewritten as one bidirected edge.
///
/// Node order is the same rule as [`CyclicDigraph`]: dense ids follow insertion,
/// and [`Self::with_variables`] / [`Self::from_named_edges`] keep schema order.
/// [`Eq`] compares node labels, the directed edge set, and the bidirected edge
/// set, independent of edge insertion order.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CyclicMixedGraph {
    directed: CyclicDigraph,
    bidirected: Vec<Vec<DenseNodeId>>,
    revision: Revision,
}

/// Identity of one graph state, for caches keyed on "this exact graph".
///
/// The empty graph carries a fixed token; every mutation (and every graph built from outside
/// parts) draws a process-unique one. A clone shares its source's token until either side
/// mutates, so equal tokens imply identical content. The token is not part of graph equality:
/// two graphs built in different ways compare equal while their tokens differ.
#[derive(Clone, Copy, Debug)]
struct Revision(u64);

impl Revision {
    const EMPTY: Self = Self(0);

    fn fresh() -> Self {
        static NEXT: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);
        Self(NEXT.fetch_add(1, std::sync::atomic::Ordering::Relaxed))
    }
}

impl PartialEq for Revision {
    fn eq(&self, _: &Self) -> bool {
        true
    }
}

impl Eq for Revision {}

impl CyclicMixedGraph {
    /// Empty graph.
    #[must_use]
    pub const fn empty() -> Self {
        Self {
            directed: CyclicDigraph::empty(),
            bidirected: Vec::new(),
            revision: Revision::EMPTY,
        }
    }

    /// One static node per variable `0..n`, in that dense order.
    #[must_use]
    pub fn with_variables(n: u32) -> Self {
        let mut graph = Self::empty();
        for i in 0..n {
            graph
                .add_node(NodeRef::Static(VariableId::from_raw(i)))
                .expect("variable count fits in a dense id");
        }
        graph
    }

    /// Static nodes for `schema`, then named directed edges. No bidirected edges.
    ///
    /// Every edge is checked before any edge is inserted, so a failure returns
    /// no graph.
    ///
    /// # Errors
    ///
    /// Unknown names, unknown nodes, duplicate directed edges, or self-loops.
    /// A directed cycle is not an error.
    pub fn from_named_edges(
        schema: &antecedent_core::CausalSchema,
        edges: &[(&str, &str)],
    ) -> Result<Self, GraphError> {
        let n = crate::named::schema_node_count(schema)?;
        let pending = named_directed_edges(schema, n, edges)?;
        let mut graph = Self::with_variables(n);
        for (from, to) in pending {
            graph.insert_directed(from, to)?;
        }
        Ok(graph)
    }

    /// Directed edges of `graph`, and no bidirected edges.
    ///
    /// Cycles are preserved. This does not go through [`Admg`].
    #[must_use]
    pub fn from_cyclic_digraph(graph: CyclicDigraph) -> Self {
        let bidirected = vec![Vec::new(); graph.node_count()];
        Self { directed: graph, bidirected, revision: Revision::fresh() }
    }

    /// Token identifying this exact graph state: equal tokens imply identical content, and any
    /// change to the graph draws a new one. A cache keyed on it needs no structural comparison.
    pub(crate) fn revision(&self) -> u64 {
        self.revision.0
    }

    /// Number of nodes.
    #[must_use]
    pub fn node_count(&self) -> usize {
        self.directed.node_count()
    }

    /// Whether the node table is empty.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.directed.is_empty()
    }

    /// Node labels in dense order.
    #[must_use]
    pub fn nodes(&self) -> &[NodeRef] {
        self.directed.nodes()
    }

    /// Every dense node id in ascending order.
    #[must_use]
    pub fn node_ids(&self) -> impl DoubleEndedIterator<Item = DenseNodeId> + ExactSizeIterator {
        self.directed.node_ids()
    }

    /// Add a static or unfolded node. Returns its dense id.
    ///
    /// # Errors
    ///
    /// The node is not a static graph node, or the dense id would not fit in `u32`.
    pub fn add_node(&mut self, node: NodeRef) -> Result<DenseNodeId, GraphError> {
        if !node.is_static_graph_node() {
            return Err(GraphError::InvalidEndpoints {
                message: "CyclicMixedGraph accepts only Static or Unfolded nodes",
            });
        }
        let id = self.directed.add_node(node)?;
        self.bidirected.push(Vec::new());
        self.revision = Revision::fresh();
        Ok(id)
    }

    /// Insert directed edge `from -> to`. Cycles are kept.
    ///
    /// # Errors
    ///
    /// Unknown nodes, a self-loop, or a duplicate directed edge.
    pub fn insert_directed(
        &mut self,
        from: DenseNodeId,
        to: DenseNodeId,
    ) -> Result<(), GraphError> {
        self.revision = Revision::fresh();
        self.directed.insert_directed(from, to)
    }

    /// Insert bidirected edge `a <-> b`.
    ///
    /// The edge is stored symmetrically. It does not add a directed edge.
    ///
    /// # Errors
    ///
    /// Unknown nodes, a self-loop, or a duplicate bidirected edge.
    pub fn insert_bidirected(&mut self, a: DenseNodeId, b: DenseNodeId) -> Result<(), GraphError> {
        DirectedStore::known(self.node_count(), a)?;
        DirectedStore::known(self.node_count(), b)?;
        reject_self_loop(self.nodes(), a, b)?;
        self.revision = Revision::fresh();
        if !insert_sorted(&mut self.bidirected[a.as_usize()], b) {
            return Err(GraphError::DuplicateEdge { from: a.raw(), to: b.raw() });
        }
        if !insert_sorted(&mut self.bidirected[b.as_usize()], a) {
            let row = &mut self.bidirected[a.as_usize()];
            let at = row.binary_search(&b).expect("bidirected edge was just inserted");
            row.remove(at);
            return Err(GraphError::DuplicateEdge { from: a.raw(), to: b.raw() });
        }
        Ok(())
    }

    /// Directed children of `id` (empty when `id` is outside the graph).
    #[must_use]
    pub fn children(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.directed.children(id)
    }

    /// Directed parents of `id` (empty when `id` is outside the graph).
    #[must_use]
    pub fn parents(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.directed.parents(id)
    }

    /// Directed successors of `id`.
    ///
    /// The iterator does not record visits. A walk that follows it may return
    /// to the same node when the directed edges contain a cycle. Bidirected
    /// neighbors are not successors.
    #[must_use = "the successor walk is the only cycle-safe traversal"]
    pub fn successors(&self, id: DenseNodeId) -> impl Iterator<Item = DenseNodeId> + '_ {
        self.directed.successors(id)
    }

    /// Bidirected neighbors of `id` (empty when `id` is outside the graph).
    ///
    /// These are not directed children. `a <-> b` does not appear here as a
    /// stand-in for `a -> b`.
    #[must_use]
    pub fn bidirected_neighbors(&self, id: DenseNodeId) -> &[DenseNodeId] {
        self.bidirected.get(id.as_usize()).map_or(&[], Vec::as_slice)
    }

    /// Directed edges as marked edges. Bidirected edges are omitted.
    pub fn edges(&self) -> impl Iterator<Item = MarkedEdge> + '_ {
        self.directed.edges()
    }

    /// Directed edges as `(parent, child)` dense-id pairs in ascending order.
    #[must_use]
    pub fn directed_edge_pairs_sorted(&self) -> Vec<(u32, u32)> {
        self.directed.directed_edge_pairs_sorted()
    }

    /// Bidirected edges as `(lower, higher)` dense-id pairs in ascending order.
    ///
    /// Each unordered pair appears once. Swapping insertion order does not
    /// change this list. Together with [`Eq`], it identifies the bidirected
    /// edge set.
    #[must_use]
    pub fn bidirected_edge_pairs_sorted(&self) -> Vec<(u32, u32)> {
        bidirected_pairs(&self.bidirected)
    }

    /// Check directed and bidirected adjacency.
    ///
    /// Directed cycles are legal. Bidirected edges must be symmetric and free
    /// of self-loops and duplicates.
    ///
    /// # Errors
    ///
    /// An unknown endpoint, a self-loop, a duplicate edge, a missing symmetric
    /// bidirected endpoint, or adjacency that does not match the node table.
    pub fn validate(&self) -> Result<(), GraphError> {
        self.directed.validate()?;
        validate_bidirected(self.nodes(), &self.bidirected)
    }

    /// Directed strongly connected components of this mixed graph.
    ///
    /// Bidirected edges are ignored. A graph whose only edge is bidirected has
    /// one singleton component per node, in dense-id order. See [`Condensation`].
    #[must_use = "the condensation is the directed solve-block schedule"]
    pub fn condensation(&self) -> Condensation {
        self.directed.condensation()
    }
}

fn reject_self_loop(
    nodes: &[NodeRef],
    from: DenseNodeId,
    to: DenseNodeId,
) -> Result<(), GraphError> {
    if from != to {
        return Ok(());
    }
    Err(GraphError::ContemporaneousSelfEdge { variable: nodes[from.as_usize()].variable() })
}

/// Resolve and check every named edge before any of them is inserted.
fn named_directed_edges(
    schema: &antecedent_core::CausalSchema,
    node_count: u32,
    edges: &[(&str, &str)],
) -> Result<Vec<(DenseNodeId, DenseNodeId)>, GraphError> {
    let limit = node_count as usize;
    let mut pending = Vec::with_capacity(edges.len());
    let mut seen = std::collections::HashSet::with_capacity(edges.len());
    for &(from_name, to_name) in edges {
        let (from, to) = crate::named::resolve_named_edge(schema, from_name, to_name)?;
        if from.as_usize() >= limit {
            return Err(GraphError::UnknownNode { id: from.raw() });
        }
        if to.as_usize() >= limit {
            return Err(GraphError::UnknownNode { id: to.raw() });
        }
        if from == to {
            return Err(GraphError::ContemporaneousSelfEdge {
                variable: VariableId::from_raw(from.raw()),
            });
        }
        if !seen.insert((from, to)) {
            return Err(GraphError::DuplicateEdge { from: from.raw(), to: to.raw() });
        }
        pending.push((from, to));
    }
    Ok(pending)
}

fn bidirected_pairs(bidirected: &[Vec<DenseNodeId>]) -> Vec<(u32, u32)> {
    sorted_row_pairs(bidirected, |from, to| to > from)
}

fn validate_directed(graph: &CyclicDigraph) -> Result<(), GraphError> {
    let nodes = graph.nodes();
    let children = graph.store.child_rows();
    let parents = graph.store.parent_rows();
    if children.len() != nodes.len() || parents.len() != nodes.len() {
        return Err(GraphError::InvalidEndpoints {
            message: "directed adjacency does not cover every node",
        });
    }
    let mut child_edges = 0usize;
    for (i, kids) in children.iter().enumerate() {
        let from = DenseNodeId::from_index(i);
        for (k, &to) in kids.iter().enumerate() {
            DirectedStore::known(nodes.len(), to)?;
            reject_self_loop(nodes, from, to)?;
            if kids[..k].contains(&to) {
                return Err(GraphError::DuplicateEdge { from: from.raw(), to: to.raw() });
            }
            if !parents.get(to.as_usize()).is_some_and(|preds| preds.contains(&from)) {
                return Err(GraphError::InvalidEndpoints {
                    message: "directed edge is missing its parent entry",
                });
            }
            child_edges += 1;
        }
    }
    let mut parent_edges = 0usize;
    for (i, preds) in parents.iter().enumerate() {
        let to = DenseNodeId::from_index(i);
        for &from in preds {
            DirectedStore::known(nodes.len(), from)?;
            reject_self_loop(nodes, from, to)?;
            parent_edges += 1;
        }
    }
    if child_edges != parent_edges {
        return Err(GraphError::InvalidEndpoints {
            message: "directed parent lists do not match children",
        });
    }
    Ok(())
}

fn validate_bidirected(
    nodes: &[NodeRef],
    bidirected: &[Vec<DenseNodeId>],
) -> Result<(), GraphError> {
    if bidirected.len() != nodes.len() {
        return Err(GraphError::InvalidEndpoints {
            message: "bidirected adjacency does not cover every node",
        });
    }
    for (i, neighbors) in bidirected.iter().enumerate() {
        let a = DenseNodeId::from_index(i);
        for (k, &b) in neighbors.iter().enumerate() {
            DirectedStore::known(nodes.len(), b)?;
            reject_self_loop(nodes, a, b)?;
            if neighbors[..k].contains(&b) {
                return Err(GraphError::DuplicateEdge { from: a.raw(), to: b.raw() });
            }
            let symmetric = bidirected.get(b.as_usize()).is_some_and(|row| row.contains(&a));
            if !symmetric {
                return Err(GraphError::InvalidEndpoints {
                    message: "bidirected edge is missing its symmetric endpoint",
                });
            }
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use antecedent_core::{CausalSchemaBuilder, Lag, VariableId};

    use super::*;
    use crate::admg::Admg;
    use crate::dag::Dag;

    fn id(raw: u32) -> DenseNodeId {
        DenseNodeId::from_raw(raw)
    }

    fn ab_schema() -> antecedent_core::CausalSchema {
        CausalSchemaBuilder::new()
            .continuous("a")
            .finish()
            .continuous("b")
            .finish()
            .build()
            .unwrap()
    }

    #[test]
    fn cycle_is_legal_while_dag_and_admg_still_reject_it() {
        let (a, b, c) = (id(0), id(1), id(2));
        let mut cyclic = CyclicDigraph::with_variables(2);
        cyclic.insert_directed(a, b).unwrap();
        cyclic.insert_directed(b, a).unwrap();
        assert!(cyclic.validate().is_ok());
        assert_eq!(cyclic.directed_edge_pairs_sorted(), vec![(0, 1), (1, 0)]);
        assert_eq!(cyclic.children(a), &[b]);
        assert_eq!(cyclic.parents(a), &[b]);
        let marked: Vec<_> = cyclic.edges().collect();
        assert_eq!(marked.len(), 2);
        assert!(marked.iter().all(|edge| edge.parent_child().is_some() && !edge.is_bidirected()));

        let mut dag = Dag::with_variables(2);
        dag.insert_directed(a, b).unwrap();
        assert_eq!(dag.insert_directed(b, a), Err(GraphError::Cycle { from: 1, to: 0 }));
        assert_eq!(dag.directed_edge_pairs_sorted(), vec![(0, 1)]);
        assert!(dag.validate().is_ok());

        let mut admg = Admg::with_variables(2);
        admg.insert_directed(a, b).unwrap();
        assert_eq!(admg.insert_directed(b, a), Err(GraphError::Cycle { from: 1, to: 0 }));
        assert_eq!(admg.directed_edge_pairs_sorted(), vec![(0, 1)]);
        assert!(admg.bidirected_edge_pairs_sorted().is_empty());
        assert!(admg.validate().is_ok());

        let mut triangle = CyclicDigraph::with_variables(3);
        for (from, to) in [(0, 1), (1, 2), (2, 0)] {
            triangle.insert_directed(id(from), id(to)).unwrap();
        }
        assert!(triangle.validate().is_ok());
        assert_eq!(triangle.directed_edge_pairs_sorted(), vec![(0, 1), (1, 2), (2, 0)]);
        let mut dag3 = Dag::with_variables(3);
        dag3.insert_directed(a, b).unwrap();
        dag3.insert_directed(b, c).unwrap();
        assert_eq!(dag3.insert_directed(c, a), Err(GraphError::Cycle { from: 2, to: 0 }));

        let mixed = CyclicMixedGraph::from_cyclic_digraph(cyclic);
        assert_eq!(mixed.directed_edge_pairs_sorted(), vec![(0, 1), (1, 0)]);
        assert!(mixed.bidirected_edge_pairs_sorted().is_empty());
        assert!(mixed.bidirected_neighbors(a).is_empty());
        assert!(mixed.validate().is_ok());
        assert!(CyclicDigraph::empty().validate().is_ok());
        assert!(CyclicMixedGraph::empty().validate().is_ok());
    }

    #[test]
    fn reciprocal_edges_differ_from_bidirected_and_one_way() {
        let (a, b) = (id(0), id(1));
        let mut one_way = CyclicMixedGraph::with_variables(2);
        one_way.insert_directed(a, b).unwrap();
        let mut reciprocal = CyclicMixedGraph::with_variables(2);
        reciprocal.insert_directed(a, b).unwrap();
        reciprocal.insert_directed(b, a).unwrap();
        let mut confounded = CyclicMixedGraph::with_variables(2);
        confounded.insert_bidirected(a, b).unwrap();

        assert_ne!(one_way, reciprocal);
        assert_ne!(reciprocal, confounded);
        assert_ne!(one_way, confounded);
        assert_eq!(one_way.directed_edge_pairs_sorted(), vec![(0, 1)]);
        assert!(one_way.bidirected_edge_pairs_sorted().is_empty());
        assert_eq!(reciprocal.directed_edge_pairs_sorted(), vec![(0, 1), (1, 0)]);
        assert!(reciprocal.bidirected_edge_pairs_sorted().is_empty());
        assert!(reciprocal.bidirected_neighbors(a).is_empty());
        assert!(confounded.directed_edge_pairs_sorted().is_empty());
        assert!(confounded.children(a).is_empty());
        assert!(confounded.parents(b).is_empty());
        assert!(confounded.successors(a).next().is_none());
        assert_eq!(confounded.bidirected_neighbors(a), &[b]);
        assert_eq!(confounded.bidirected_neighbors(b), &[a]);
        assert_eq!(confounded.bidirected_edge_pairs_sorted(), vec![(0, 1)]);
        assert!(confounded.edges().next().is_none());

        let mut both = reciprocal.clone();
        both.insert_bidirected(b, a).unwrap();
        assert_eq!(both.directed_edge_pairs_sorted(), vec![(0, 1), (1, 0)]);
        assert_eq!(both.bidirected_edge_pairs_sorted(), vec![(0, 1)]);
        assert_ne!(both, reciprocal);
        assert_ne!(both, confounded);
        assert_eq!(both.children(a), &[b]);
        assert_eq!(both.bidirected_neighbors(a), &[b]);
    }

    #[test]
    fn canonical_lists_and_equality_ignore_insertion_order() {
        let mut first = CyclicMixedGraph::with_variables(3);
        first.insert_directed(id(2), id(0)).unwrap();
        first.insert_directed(id(0), id(1)).unwrap();
        first.insert_directed(id(0), id(2)).unwrap();
        first.insert_bidirected(id(2), id(1)).unwrap();
        first.insert_bidirected(id(0), id(2)).unwrap();

        let mut second = CyclicMixedGraph::with_variables(3);
        second.insert_directed(id(0), id(2)).unwrap();
        second.insert_directed(id(0), id(1)).unwrap();
        second.insert_directed(id(2), id(0)).unwrap();
        second.insert_bidirected(id(0), id(2)).unwrap();
        second.insert_bidirected(id(1), id(2)).unwrap();

        assert_eq!(first, second);
        assert_eq!(first.directed_edge_pairs_sorted(), second.directed_edge_pairs_sorted());
        assert_eq!(first.directed_edge_pairs_sorted(), vec![(0, 1), (0, 2), (2, 0)]);
        assert_eq!(first.bidirected_edge_pairs_sorted(), second.bidirected_edge_pairs_sorted());
        assert_eq!(first.bidirected_edge_pairs_sorted(), vec![(0, 2), (1, 2)]);
        assert_eq!(first.children(id(0)), &[id(1), id(2)]);
        assert_eq!(second.children(id(0)), &[id(1), id(2)]);
    }

    #[test]
    fn self_loop_is_a_contemporaneous_self_edge() {
        let mut graph = CyclicDigraph::with_variables(1);
        let node = id(0);
        assert_eq!(
            graph.insert_directed(node, node),
            Err(GraphError::ContemporaneousSelfEdge { variable: VariableId::from_raw(0) }),
        );
        assert!(graph.directed_edge_pairs_sorted().is_empty());
        assert!(graph.validate().is_ok());

        let mut unfolded = CyclicDigraph::empty();
        let slot = unfolded
            .add_node(NodeRef::Unfolded { variable: VariableId::from_raw(3), offset: 0 })
            .unwrap();
        assert_eq!(
            unfolded.insert_directed(slot, slot),
            Err(GraphError::ContemporaneousSelfEdge { variable: VariableId::from_raw(3) }),
        );

        let mut mixed = CyclicMixedGraph::with_variables(1);
        assert_eq!(
            mixed.insert_bidirected(node, node),
            Err(GraphError::ContemporaneousSelfEdge { variable: VariableId::from_raw(0) }),
        );
        assert!(mixed.bidirected_edge_pairs_sorted().is_empty());
    }

    #[test]
    fn unknown_duplicate_and_missing_endpoints_are_rejected() {
        let (a, b) = (id(0), id(1));
        let mut graph = CyclicDigraph::with_variables(2);
        graph.insert_directed(a, b).unwrap();
        assert_eq!(graph.insert_directed(a, b), Err(GraphError::DuplicateEdge { from: 0, to: 1 }),);
        assert_eq!(graph.insert_directed(id(9), a), Err(GraphError::UnknownNode { id: 9 }),);
        assert_eq!(graph.insert_directed(a, id(4)), Err(GraphError::UnknownNode { id: 4 }),);
        assert_eq!(
            graph.add_node(NodeRef::Lagged {
                variable: VariableId::from_raw(0),
                lag: Lag::CONTEMPORANEOUS,
            }),
            Err(GraphError::InvalidEndpoints {
                message: "CyclicDigraph accepts only Static or Unfolded nodes",
            }),
        );

        let mut mixed = CyclicMixedGraph::with_variables(2);
        mixed.insert_bidirected(a, b).unwrap();
        assert_eq!(
            mixed.insert_bidirected(b, a),
            Err(GraphError::DuplicateEdge { from: 1, to: 0 }),
        );
        assert_eq!(mixed.insert_bidirected(id(3), a), Err(GraphError::UnknownNode { id: 3 }),);
        assert!(mixed.directed_edge_pairs_sorted().is_empty());
        assert_eq!(
            mixed.add_node(NodeRef::Context {
                variable: VariableId::from_raw(1),
                environment: None,
            }),
            Err(GraphError::InvalidEndpoints {
                message: "CyclicMixedGraph accepts only Static or Unfolded nodes",
            }),
        );
    }

    #[test]
    fn successors_may_revisit_a_node() {
        let (a, b) = (id(0), id(1));
        let mut graph = CyclicDigraph::with_variables(2);
        graph.insert_directed(a, b).unwrap();
        graph.insert_directed(b, a).unwrap();
        let mut at = a;
        let mut visits = [0u8; 2];
        for _ in 0..4 {
            visits[at.as_usize()] += 1;
            at = graph.successors(at).next().expect("directed cycle");
        }
        assert_eq!(visits, [2, 2]);
        assert_eq!(at, a);

        let mut fan = CyclicMixedGraph::with_variables(3);
        fan.insert_directed(a, id(2)).unwrap();
        fan.insert_directed(a, b).unwrap();
        fan.insert_bidirected(b, id(2)).unwrap();
        assert_eq!(fan.successors(a).collect::<Vec<_>>(), vec![b, id(2)]);
        assert!(fan.successors(id(9)).next().is_none());
        assert!(CyclicDigraph::empty().successors(a).next().is_none());
    }

    #[test]
    fn named_edges_follow_schema_order() {
        let schema = ab_schema();
        let named = CyclicDigraph::from_named_edges(&schema, &[("b", "a"), ("a", "b")]).unwrap();
        let mut manual = CyclicDigraph::with_variables(2);
        manual.insert_directed(id(0), id(1)).unwrap();
        manual.insert_directed(id(1), id(0)).unwrap();
        assert_eq!(named, manual);
        assert_eq!(named.nodes()[0], NodeRef::Static(VariableId::from_raw(0)));
        assert_eq!(named.directed_edge_pairs_sorted(), vec![(0, 1), (1, 0)]);

        let mixed = CyclicMixedGraph::from_named_edges(&schema, &[("a", "b"), ("b", "a")]).unwrap();
        assert_eq!(mixed.directed_edge_pairs_sorted(), named.directed_edge_pairs_sorted());
        assert!(mixed.bidirected_edge_pairs_sorted().is_empty());
        assert!(mixed.bidirected_neighbors(id(0)).is_empty());

        assert_eq!(
            CyclicDigraph::from_named_edges(&schema, &[("a", "a"), ("missing", "b")]).unwrap_err(),
            GraphError::ContemporaneousSelfEdge { variable: VariableId::from_raw(0) },
        );
        assert_eq!(
            CyclicMixedGraph::from_named_edges(&schema, &[("a", "missing")]).unwrap_err(),
            GraphError::UnknownVariableName { name: "missing".to_owned() },
        );
    }

    #[test]
    fn named_edges_fail_before_any_edge_is_inserted() {
        let schema = ab_schema();
        assert_eq!(
            CyclicDigraph::from_named_edges(&schema, &[("a", "b"), ("b", "b")]).unwrap_err(),
            GraphError::ContemporaneousSelfEdge { variable: VariableId::from_raw(1) },
        );
        assert_eq!(
            CyclicDigraph::from_named_edges(&schema, &[("a", "b"), ("a", "b")]).unwrap_err(),
            GraphError::DuplicateEdge { from: 0, to: 1 },
        );
        assert_eq!(
            CyclicDigraph::from_named_edges(&schema, &[("a", "b"), ("missing", "a")]).unwrap_err(),
            GraphError::UnknownVariableName { name: "missing".to_owned() },
        );
        assert_eq!(
            CyclicMixedGraph::from_named_edges(&schema, &[("a", "b"), ("nope", "b")]).unwrap_err(),
            GraphError::UnknownVariableName { name: "nope".to_owned() },
        );
        let kept = CyclicDigraph::from_named_edges(&schema, &[("a", "b")]).unwrap();
        assert_eq!(kept.directed_edge_pairs_sorted(), vec![(0, 1)]);
    }

    #[test]
    fn reordered_nodes_are_a_different_label_assignment() {
        let mut schema_order = CyclicDigraph::empty();
        schema_order.add_node(NodeRef::Static(VariableId::from_raw(0))).unwrap();
        schema_order.add_node(NodeRef::Static(VariableId::from_raw(1))).unwrap();
        let mut swapped = CyclicDigraph::empty();
        swapped.add_node(NodeRef::Static(VariableId::from_raw(1))).unwrap();
        swapped.add_node(NodeRef::Static(VariableId::from_raw(0))).unwrap();
        assert_ne!(schema_order, swapped);
        assert_eq!(schema_order.nodes()[0], NodeRef::Static(VariableId::from_raw(0)));
        assert_eq!(swapped.nodes()[0], NodeRef::Static(VariableId::from_raw(1)));
        assert_eq!(schema_order, CyclicDigraph::with_variables(2));
        assert_eq!(
            CyclicMixedGraph::with_variables(2).nodes(),
            CyclicDigraph::with_variables(2).nodes(),
        );
    }
}
