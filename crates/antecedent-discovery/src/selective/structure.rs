//! A discovered structure in a neutral, content-addressed form.
//!
//! Static and temporal discovery return different graph classes. For identity, comparison
//! and refusal text they are all the same thing here: a set of nodes keyed by what they stand
//! for (never by dense index) and a set of marked edges. The identity does not depend on node
//! insertion order or edge storage order.
//!
//! An equivalence class stays an equivalence class: an edge whose endpoint marks are not
//! both resolved is reported as unresolved and no function in this module chooses a member.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{CanonicalWriter, NodeRef, VariableId};
use antecedent_graph::{
    Cpdag, Dag, DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag, TemporalCpdag, TemporalDag,
    TemporalPag,
};

use super::identity::Digest;
use antecedent_core::named_variants;

const STRUCTURE_TAG: &str = "antecedent.selective_discovery.structure.v1";

/// The class of the discovered structure.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum StructureClass {
    /// Completed partially directed acyclic graph over static variables.
    Cpdag,
    /// Partial ancestral graph over static variables.
    Pag,
    /// Lagged DAG (lagged-only parent search).
    TemporalDag,
    /// Temporal CPDAG (contemporaneous links oriented up to equivalence).
    TemporalCpdag,
    /// Temporal partial ancestral graph (latent confounding allowed).
    TemporalPag,
}

named_variants! {
    StructureClass {
        Cpdag => "cpdag",
        Pag => "pag",
        TemporalDag => "temporal_dag",
        TemporalCpdag => "temporal_cpdag",
        TemporalPag => "temporal_pag",
    }
}

impl StructureClass {
    const fn tag(self) -> u64 {
        match self {
            Self::Cpdag => 0,
            Self::Pag => 1,
            Self::TemporalDag => 2,
            Self::TemporalCpdag => 3,
            Self::TemporalPag => 4,
        }
    }

    /// Whether a graph of this class can still hold more than one member of its equivalence
    /// class (an undirected edge or a circle endpoint).
    #[must_use]
    pub const fn is_equivalence_class(self) -> bool {
        !matches!(self, Self::TemporalDag)
    }
}

/// What a node stands for, independent of dense index.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StructureNode {
    /// `0` static, `1` lagged, `2` unfolded slot, `3` context.
    pub kind: u8,
    /// Variable id.
    pub variable: u32,
    /// Lag magnitude, signed offset or environment id (`-1` for none).
    pub aux: i64,
}

impl StructureNode {
    /// The node a graph node reference stands for.
    #[must_use]
    pub fn from_node_ref(node: NodeRef) -> Self {
        match node {
            NodeRef::Static(variable) => Self { kind: 0, variable: variable.raw(), aux: 0 },
            NodeRef::Lagged { variable, lag } => {
                Self { kind: 1, variable: variable.raw(), aux: i64::from(lag.raw()) }
            }
            NodeRef::Unfolded { variable, offset } => {
                Self { kind: 2, variable: variable.raw(), aux: i64::from(offset) }
            }
            NodeRef::Context { variable, environment } => Self {
                kind: 3,
                variable: variable.raw(),
                aux: environment.map_or(-1, |e| i64::from(e.raw())),
            },
        }
    }

    /// The variable this node is an instance of.
    #[must_use]
    pub const fn variable_id(self) -> VariableId {
        VariableId::from_raw(self.variable)
    }

    /// Readable label such as `v2` or `v2@-1`.
    #[must_use]
    pub fn label(self) -> String {
        match self.kind {
            0 => format!("v{}", self.variable),
            1 => format!("v{}@-{}", self.variable, self.aux),
            2 => format!("v{}@{}", self.variable, self.aux),
            _ => format!("v{}#{}", self.variable, self.aux),
        }
    }
}

/// Endpoint mark of an edge at one node.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum Mark {
    /// Tail.
    Tail,
    /// Arrowhead.
    Arrow,
    /// Circle: not determined by the equivalence class.
    Circle,
    /// Conflicting orientation evidence.
    Conflict,
}

named_variants! {
    Mark {
        Tail => "tail",
        Arrow => "arrow",
        Circle => "circle",
        Conflict => "conflict",
    }
}

impl Mark {
    const fn from_endpoint(endpoint: Endpoint) -> Self {
        match endpoint {
            Endpoint::Tail => Self::Tail,
            Endpoint::Arrow => Self::Arrow,
            Endpoint::Circle => Self::Circle,
            Endpoint::Conflict => Self::Conflict,
        }
    }

    const fn tag(self) -> u64 {
        match self {
            Self::Tail => 0,
            Self::Arrow => 1,
            Self::Circle => 2,
            Self::Conflict => 3,
        }
    }
}

const fn middle_tag(middle: MiddleMark) -> u64 {
    match middle {
        MiddleMark::Unknown => 0,
        MiddleMark::Left => 1,
        MiddleMark::Right => 2,
        MiddleMark::Both => 3,
        MiddleMark::Empty => 4,
    }
}

/// One marked edge between two nodes, endpoints in canonical (ascending) order.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct StructureEdge {
    /// First endpoint (the smaller node).
    pub a: StructureNode,
    /// Second endpoint.
    pub b: StructureNode,
    /// Mark at `a`.
    pub at_a: Mark,
    /// Mark at `b`.
    pub at_b: Mark,
    /// Middle-mark tag of the search state (`4` = definite adjacency).
    pub middle: u8,
}

impl StructureEdge {
    /// Whether both endpoint marks are a resolved tail or arrowhead.
    #[must_use]
    pub const fn is_resolved(&self) -> bool {
        matches!(self.at_a, Mark::Tail | Mark::Arrow)
            && matches!(self.at_b, Mark::Tail | Mark::Arrow)
    }

    /// Whether the edge is a definite directed edge (tail at one end, arrowhead at the other).
    #[must_use]
    pub const fn is_directed(&self) -> bool {
        matches!((self.at_a, self.at_b), (Mark::Tail, Mark::Arrow) | (Mark::Arrow, Mark::Tail))
    }

    /// Why the edge is not resolved, or `None` when it is.
    ///
    /// An undirected edge may still be reversible within the class; a circle endpoint is not
    /// determined; a conflict records incompatible orientation evidence; an edge whose search
    /// state is not a definite adjacency (`middle != 4`) was not settled by the search.
    #[must_use]
    pub const fn unresolved_reason(&self) -> Option<&'static str> {
        if matches!(self.at_a, Mark::Conflict) || matches!(self.at_b, Mark::Conflict) {
            return Some("orientation_conflict");
        }
        if matches!(self.at_a, Mark::Circle) || matches!(self.at_b, Mark::Circle) {
            return Some("circle_endpoint");
        }
        if matches!((self.at_a, self.at_b), (Mark::Tail, Mark::Tail)) {
            return Some("undirected_in_equivalence_class");
        }
        if matches!((self.at_a, self.at_b), (Mark::Arrow, Mark::Arrow)) {
            return Some("bidirected_latent_confounding");
        }
        if self.middle != 4 {
            return Some("search_state_not_definite");
        }
        None
    }
}

/// One unresolved edge with the reason it is unresolved.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct UnresolvedEdge {
    /// The edge.
    pub edge: StructureEdge,
    /// Why it is unresolved (`snake_case`).
    pub reason: &'static str,
}

/// A discovered structure: class, nodes and marked edges in canonical order.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct DiscoveredStructure {
    class: StructureClass,
    nodes: Vec<StructureNode>,
    edges: Vec<StructureEdge>,
}

fn canonical_edge(
    nodes: &[NodeRef],
    a: DenseNodeId,
    b: DenseNodeId,
    at_a: Endpoint,
    at_b: Endpoint,
    middle: MiddleMark,
) -> Option<StructureEdge> {
    let na = StructureNode::from_node_ref(*nodes.get(a.as_usize())?);
    let nb = StructureNode::from_node_ref(*nodes.get(b.as_usize())?);
    let (first, second, mark_first, mark_second) = if na <= nb {
        (na, nb, Mark::from_endpoint(at_a), Mark::from_endpoint(at_b))
    } else {
        (nb, na, Mark::from_endpoint(at_b), Mark::from_endpoint(at_a))
    };
    Some(StructureEdge {
        a: first,
        b: second,
        at_a: mark_first,
        at_b: mark_second,
        middle: u8::try_from(middle_tag(middle)).unwrap_or(u8::MAX),
    })
}

fn collect_edges(
    nodes: &[NodeRef],
    edges: impl IntoIterator<Item = MarkedEdge>,
) -> Vec<StructureEdge> {
    let mut out: Vec<StructureEdge> = edges
        .into_iter()
        .filter_map(|e| canonical_edge(nodes, e.a, e.b, e.at_a, e.at_b, e.middle))
        .collect();
    out.sort_unstable();
    out.dedup();
    out
}

impl DiscoveredStructure {
    fn finish(class: StructureClass, nodes: &[NodeRef], edges: Vec<StructureEdge>) -> Self {
        let mut keys: Vec<StructureNode> =
            nodes.iter().map(|n| StructureNode::from_node_ref(*n)).collect();
        keys.sort_unstable();
        keys.dedup();
        Self { class, nodes: keys, edges }
    }

    /// From a static CPDAG.
    #[must_use]
    pub fn from_cpdag(graph: &Cpdag) -> Self {
        let edges = collect_edges(graph.nodes(), graph.edges());
        Self::finish(StructureClass::Cpdag, graph.nodes(), edges)
    }

    /// From a static DAG, held as the CPDAG whose every edge is directed: a DAG search names
    /// one graph, and this form tabulates its edges with the classes' canonical identity.
    #[must_use]
    pub fn from_dag(graph: &Dag) -> Self {
        Self::from_cpdag(&Cpdag::from_dag(graph))
    }

    /// From a static PAG.
    #[must_use]
    pub fn from_pag(graph: &Pag) -> Self {
        let nodes = graph.nodes();
        let mut collected = Vec::new();
        for i in 0..graph.node_count() {
            for j in (i + 1)..graph.node_count() {
                if let Some(edge) =
                    graph.edge_between(DenseNodeId::from_index(i), DenseNodeId::from_index(j))
                {
                    collected.push(edge);
                }
            }
        }
        Self::finish(StructureClass::Pag, nodes, collect_edges(nodes, collected))
    }

    /// From a lagged DAG.
    #[must_use]
    pub fn from_temporal_dag(graph: &TemporalDag) -> Self {
        let edges = collect_edges(graph.nodes(), graph.edges());
        Self::finish(StructureClass::TemporalDag, graph.nodes(), edges)
    }

    /// From a temporal CPDAG.
    #[must_use]
    pub fn from_temporal_cpdag(graph: &TemporalCpdag) -> Self {
        let edges = collect_edges(graph.nodes(), graph.edges());
        Self::finish(StructureClass::TemporalCpdag, graph.nodes(), edges)
    }

    /// From a temporal PAG.
    #[must_use]
    pub fn from_temporal_pag(graph: &TemporalPag) -> Self {
        let nodes = graph.nodes();
        let mut collected = Vec::new();
        for i in 0..graph.node_count() {
            for j in (i + 1)..graph.node_count() {
                if let Some(edge) =
                    graph.edge_between(DenseNodeId::from_index(i), DenseNodeId::from_index(j))
                {
                    collected.push(edge);
                }
            }
        }
        Self::finish(StructureClass::TemporalPag, nodes, collect_edges(nodes, collected))
    }

    /// Rebuild from parts already in canonical order (an artifact consumer).
    ///
    /// Returns `None` when nodes or edges are not strictly increasing, or an edge's endpoints
    /// are out of order or not nodes of the structure.
    #[must_use]
    pub fn from_parts(
        class: StructureClass,
        nodes: Vec<StructureNode>,
        edges: Vec<StructureEdge>,
    ) -> Option<Self> {
        if nodes.windows(2).any(|w| w[0] >= w[1]) || edges.windows(2).any(|w| w[0] >= w[1]) {
            return None;
        }
        let known = |n: &StructureNode| nodes.binary_search(n).is_ok();
        if edges.iter().any(|e| e.a >= e.b || !known(&e.a) || !known(&e.b)) {
            return None;
        }
        Some(Self { class, nodes, edges })
    }

    /// The class.
    #[must_use]
    pub const fn class(&self) -> StructureClass {
        self.class
    }

    /// Nodes in canonical order.
    #[must_use]
    pub fn nodes(&self) -> &[StructureNode] {
        &self.nodes
    }

    /// Edges in canonical order.
    #[must_use]
    pub fn edges(&self) -> &[StructureEdge] {
        &self.edges
    }

    /// Every edge that is not a definite directed edge of a resolved class member, with the
    /// reason. Empty only when every edge is directed and its search state definite.
    #[must_use]
    pub fn unresolved_edges(&self) -> Vec<UnresolvedEdge> {
        self.edges
            .iter()
            .filter_map(|edge| {
                edge.unresolved_reason().map(|reason| UnresolvedEdge { edge: *edge, reason })
            })
            .collect()
    }

    /// Whether the structure is a single graph: no unresolved edge. A fully directed CPDAG
    /// is a single DAG, but still a *discovered* one: this says nothing about whether the
    /// assumptions under which it was discovered hold.
    #[must_use]
    pub fn is_fully_directed(&self) -> bool {
        self.edges.iter().all(StructureEdge::is_directed)
    }

    /// Content identity: class, nodes and edges (marks and middle state), order independent.
    #[must_use]
    pub fn identity(&self) -> Digest {
        let mut writer = CanonicalWriter::new(STRUCTURE_TAG);
        writer.word(super::identity::SELECTIVE_IDENTITY_VERSION).word(self.class.tag());
        writer.word(self.nodes.len() as u64);
        for node in &self.nodes {
            write_node(&mut writer, *node);
        }
        writer.word(self.edges.len() as u64);
        for edge in &self.edges {
            write_node(&mut writer, edge.a);
            write_node(&mut writer, edge.b);
            writer.word(edge.at_a.tag()).word(edge.at_b.tag()).word(u64::from(edge.middle));
        }
        writer.finish()
    }
}

fn write_node(writer: &mut CanonicalWriter, node: StructureNode) {
    writer
        .word(u64::from(node.kind))
        .word(u64::from(node.variable))
        .word(u64::from_le_bytes(node.aux.to_le_bytes()));
}

#[cfg(test)]
mod tests {
    use super::*;
    use antecedent_graph::Pag;

    fn node(v: u32) -> NodeRef {
        NodeRef::Static(VariableId::from_raw(v))
    }

    #[test]
    fn identity_ignores_insertion_order_and_sees_marks() {
        let mut forward = Cpdag::empty();
        let mut reverse = Cpdag::empty();
        let ids_f: Vec<_> = (0..3).map(|v| forward.add_node(node(v)).unwrap()).collect();
        let ids_r: Vec<_> = (0..3).rev().map(|v| reverse.add_node(node(v)).unwrap()).collect();
        forward.insert_directed(ids_f[0], ids_f[1]).unwrap();
        forward.insert_undirected(ids_f[1], ids_f[2]).unwrap();
        // `ids_r` holds nodes 2, 1, 0 in that dense order.
        reverse.insert_undirected(ids_r[1], ids_r[0]).unwrap();
        reverse.insert_directed(ids_r[2], ids_r[1]).unwrap();
        let a = DiscoveredStructure::from_cpdag(&forward);
        let b = DiscoveredStructure::from_cpdag(&reverse);
        assert_eq!(a.identity(), b.identity());
        assert_eq!(a, b);
        assert_eq!(a.unresolved_edges().len(), 1);
        assert_eq!(a.unresolved_edges()[0].reason, "undirected_in_equivalence_class");
        assert!(!a.is_fully_directed());

        let mut flipped = Cpdag::empty();
        let ids: Vec<_> = (0..3).map(|v| flipped.add_node(node(v)).unwrap()).collect();
        flipped.insert_directed(ids[1], ids[0]).unwrap();
        flipped.insert_undirected(ids[1], ids[2]).unwrap();
        assert_ne!(a.identity(), DiscoveredStructure::from_cpdag(&flipped).identity());
    }

    #[test]
    fn circle_endpoints_are_unresolved_in_a_pag() {
        let mut pag = Pag::with_variables(2);
        pag.insert_circle_circle(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let structure = DiscoveredStructure::from_pag(&pag);
        let unresolved = structure.unresolved_edges();
        assert_eq!(unresolved.len(), 1);
        assert_eq!(unresolved[0].reason, "circle_endpoint");
        assert_eq!(structure.class(), StructureClass::Pag);
    }

    #[test]
    fn from_parts_refuses_non_canonical_input() {
        let a = StructureNode { kind: 0, variable: 0, aux: 0 };
        let b = StructureNode { kind: 0, variable: 1, aux: 0 };
        let edge = StructureEdge { a, b, at_a: Mark::Tail, at_b: Mark::Arrow, middle: 4 };
        assert!(
            DiscoveredStructure::from_parts(StructureClass::Cpdag, vec![a, b], vec![edge])
                .is_some()
        );
        assert!(
            DiscoveredStructure::from_parts(StructureClass::Cpdag, vec![b, a], vec![edge])
                .is_none()
        );
        let reversed = StructureEdge { a: b, b: a, ..edge };
        assert!(
            DiscoveredStructure::from_parts(StructureClass::Cpdag, vec![a, b], vec![reversed])
                .is_none()
        );
    }
}
