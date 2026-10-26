//! σ-separation for directed mixed graphs that may contain cycles.
//!
//! The engine follows Forré and Mooij by d-separating the acyclification
//! (Bongers, Forré, Peters, and Mooij, Proposition A.19). Edges in that
//! acyclification are not original direct mechanisms: an outside parent of a
//! strongly connected component is copied onto every member, and a component
//! of size at least two becomes a bidirected clique.
//!
//! [`sigma_walk_oracle`] is a different procedure. It enumerates simple paths
//! on the original graph and applies the σ-blocking rules (Definition A.16).
//! Lemma A.17 is why simple paths are enough. The walk is capped at
//! [`WALK_ORACLE_NODE_BOUND`] nodes.
//!
//! On an acyclic graph with distinct endpoints outside the conditioning set,
//! σ-separation agrees with ordinary d-separation. A query whose endpoint lies
//! in the conditioning set is σ-blocked at that endpoint (Definition A.16,
//! clause 1). [`crate::Dag::is_d_separated`] instead treats that query as
//! ill-posed and reports it as not separated.
//!
//! Ordinary d-separation is a different criterion. It is not declared valid
//! for every cyclic graph, and it is not discarded: linear and discrete cyclic
//! families can still satisfy it (Bongers et al., Theorem A.7). A
//! [`SigmaSeparationStatus::NotSeparated`] answer means this procedure did not
//! separate the endpoints. It is not a proof of dependence. σ-faithfulness and
//! simplicity of the SCM are not assumed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::algo::{closure_into, insert_sorted};
use crate::cyclic::CyclicMixedGraph;
use crate::dsep::{BayesBallGraph, DSeparationWorkspace, bayes_ball_reaches};
use crate::error::GraphError;
use crate::types::DenseNodeId;
use crate::workspace::{BitSet, GraphWorkspace};

/// Largest node count of the exhaustive directed-graph comparison in the tests.
///
/// Mixed graphs, which also vary bidirected edges, are exhausted at
/// `EXHAUSTIVE_MIXED_NODE_BOUND`.
#[cfg(test)]
const EXHAUSTIVE_NODE_BOUND: usize = WALK_ORACLE_NODE_BOUND;

/// Node count of the exhaustive directed-and-bidirected comparison in the tests.
#[cfg(test)]
const EXHAUSTIVE_MIXED_NODE_BOUND: usize = 3;

/// Maximum nodes [`sigma_walk_oracle`] will accept.
///
/// The walk is exponential in the number of nodes. [`sigma_separates`] has no
/// such cap.
pub const WALK_ORACLE_NODE_BOUND: usize = 4;

/// Restrictions carried by every σ-separation answer.
///
/// The fields are the theorem's hypotheses, not a solvability check. Read
/// [`Self::reading`] before treating the status as a conditional independence.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub struct SigmaSeparationAssumptions {
    /// Full statement of what this procedure does and does not claim.
    pub reading: &'static str,
    /// The observational Markov reading needs unique solvability of each
    /// strongly connected component (Theorem A.21). This query does not check it.
    pub observational_markov_requires_unique_solvability_per_component: bool,
    /// A simple SCM is not assumed, so interventional and counterfactual
    /// Markov properties (Corollary A.22) are not claimed.
    pub simple_scm_assumed: bool,
    /// σ-faithfulness (Definition A.23) is not assumed. Separation is not a
    /// list of every independence in a distribution.
    pub sigma_faithfulness_assumed: bool,
    /// What this answer says about ordinary d-separation once a directed cycle
    /// is present.
    pub ordinary_d_separation_on_cycles: OrdinaryDSeparationOnCycles,
}

impl SigmaSeparationAssumptions {
    /// Prose form of [`Self::PROCEDURE`].
    pub const READING: &'static str = concat!(
        "Graphical σ-separation on a directed mixed graph ",
        "(Forré and Mooij 2017; Bongers, Forré, Peters, and Mooij, Definition A.16). ",
        "The observational general directed global Markov property needs unique solvability with ",
        "respect to each strongly connected component; this query does not check solvability. ",
        "A simple SCM is not assumed, so interventional and counterfactual Markov properties ",
        "are not claimed. σ-faithfulness is not assumed. ",
        "Failure to separate is not a proof of dependence. ",
        "Ordinary d-separation is neither affirmed nor rejected on cyclic graphs; ",
        "it can still hold for restricted families such as linear or discrete models.",
    );

    /// Assumptions of [`sigma_separates`] and [`sigma_walk_oracle`].
    pub const PROCEDURE: Self = Self {
        reading: Self::READING,
        observational_markov_requires_unique_solvability_per_component: true,
        simple_scm_assumed: false,
        sigma_faithfulness_assumed: false,
        ordinary_d_separation_on_cycles: OrdinaryDSeparationOnCycles::NotSettled,
    };
}

/// Ordinary d-separation is left alone on cyclic graphs.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum OrdinaryDSeparationOnCycles {
    /// This answer does not say that d-separation is valid, or invalid, once
    /// a directed cycle is present.
    NotSettled,
}

/// Whether this procedure σ-separated the endpoints.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum SigmaSeparationStatus {
    /// Every path between the endpoints is σ-blocked by the conditioning set.
    Separated,
    /// This procedure did not σ-separate the endpoints.
    ///
    /// That is not a proof of dependence. σ-faithfulness is not assumed, so an
    /// open path does not imply a dependence in any distribution.
    NotSeparated,
}

impl SigmaSeparationStatus {
    /// Whether the status is [`Self::Separated`].
    #[must_use]
    pub const fn is_separated(self) -> bool {
        matches!(self, Self::Separated)
    }
}

/// One σ-separation answer, with the restrictions that were assumed.
#[derive(Clone, Debug, Eq, PartialEq)]
#[must_use = "read assumptions; NotSeparated is not a proof of dependence"]
#[non_exhaustive]
pub struct SigmaSeparation {
    /// First endpoint.
    pub x: DenseNodeId,
    /// Second endpoint.
    pub y: DenseNodeId,
    /// Conditioning set, sorted and with duplicates removed.
    pub conditioning: Vec<DenseNodeId>,
    /// Graphical status of this procedure.
    pub status: SigmaSeparationStatus,
    /// Restrictions a consumer has to read before using [`Self::status`].
    pub assumptions: SigmaSeparationAssumptions,
}

impl SigmaSeparation {
    /// Whether this procedure σ-separated [`Self::x`] from [`Self::y`].
    ///
    /// `false` means not separated by this procedure. It is not a proof of dependence.
    #[must_use]
    pub const fn separated(&self) -> bool {
        self.status.is_separated()
    }
}

/// Scratch for repeated [`sigma_separates`] queries.
///
/// The adjacency stored here is the acyclification used as d-separation fuel.
/// Those edges are not original direct mechanisms and are not exposed.
/// Condensation and that adjacency are kept when the next query is on a graph
/// with the same node count and the same directed and bidirected edges.
#[derive(Clone, Debug, Default)]
pub struct SigmaSeparationWorkspace {
    dsep: DSeparationWorkspace,
    parents: Vec<Vec<DenseNodeId>>,
    children: Vec<Vec<DenseNodeId>>,
    bidirected: Vec<Vec<DenseNodeId>>,
    cache: Option<CachedAcyclification>,
    rebuilds: u64,
}

/// Condensation and acyclification identity for one graph.
#[derive(Clone, Debug, PartialEq, Eq)]
struct CachedAcyclification {
    nodes: usize,
    /// [`CyclicMixedGraph::revision`] of the graph the acyclification was built from: equal
    /// tokens mean the same graph, so reuse costs no edge-list comparison.
    revision: u64,
    component_of: Vec<usize>,
    components: Vec<Vec<DenseNodeId>>,
}

impl SigmaSeparationWorkspace {
    /// Empty scratch buffers.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }
}

/// Whether `x` is σ-separated from `y` given `conditioning` in `graph`.
///
/// The decision is d-separation in the acyclification. A directed edge there
/// need not be an original mechanism. [`SigmaSeparationStatus::NotSeparated`]
/// means this procedure did not separate the endpoints.
///
/// # Errors
///
/// `x`, `y`, or a conditioning node is outside the graph.
pub fn sigma_separates(
    graph: &CyclicMixedGraph,
    x: DenseNodeId,
    y: DenseNodeId,
    conditioning: &[DenseNodeId],
    ws: &mut SigmaSeparationWorkspace,
) -> Result<SigmaSeparation, GraphError> {
    let ends = prepare(graph.node_count(), x, y, conditioning)?;
    let status = match trivial_status(&ends) {
        Some(status) => status,
        None => status_from_connection(acyclification_connects(graph, &ends, ws)),
    };
    Ok(finish(ends, status))
}

/// σ-separation by enumerating simple paths on the original graph.
///
/// This is the check for [`sigma_separates`], not a second copy of the
/// acyclification. Graphs with more than [`WALK_ORACLE_NODE_BOUND`] nodes are
/// refused.
///
/// # Errors
///
/// A node is outside the graph, or the graph is larger than the walk bound.
pub fn sigma_walk_oracle(
    graph: &CyclicMixedGraph,
    x: DenseNodeId,
    y: DenseNodeId,
    conditioning: &[DenseNodeId],
) -> Result<SigmaSeparation, GraphError> {
    if graph.node_count() > WALK_ORACLE_NODE_BOUND {
        return Err(GraphError::NodeBoundExceeded {
            nodes: graph.node_count(),
            bound: WALK_ORACLE_NODE_BOUND,
        });
    }
    let ends = prepare(graph.node_count(), x, y, conditioning)?;
    let status = match trivial_status(&ends) {
        Some(status) => status,
        None => status_from_connection(walk_connects(graph, &ends)),
    };
    Ok(finish(ends, status))
}

struct QueryEnds {
    x: DenseNodeId,
    y: DenseNodeId,
    conditioning: Vec<DenseNodeId>,
    conditioned: BitSet,
}

fn prepare(
    n: usize,
    x: DenseNodeId,
    y: DenseNodeId,
    conditioning: &[DenseNodeId],
) -> Result<QueryEnds, GraphError> {
    known(n, x)?;
    known(n, y)?;
    let mut set = Vec::new();
    let mut conditioned = BitSet::with_len(n);
    for &node in conditioning {
        known(n, node)?;
        if conditioned.contains(node) {
            continue;
        }
        conditioned.insert(node);
        set.push(node);
    }
    set.sort_unstable();
    Ok(QueryEnds { x, y, conditioning: set, conditioned })
}

const fn known(n: usize, id: DenseNodeId) -> Result<(), GraphError> {
    if id.as_usize() >= n { Err(GraphError::UnknownNode { id: id.raw() }) } else { Ok(()) }
}

/// Definition A.16, clause 1, and the trivial one-node walk.
fn trivial_status(ends: &QueryEnds) -> Option<SigmaSeparationStatus> {
    if ends.conditioned.contains(ends.x) || ends.conditioned.contains(ends.y) {
        return Some(SigmaSeparationStatus::Separated);
    }
    if ends.x == ends.y {
        return Some(SigmaSeparationStatus::NotSeparated);
    }
    None
}

const fn status_from_connection(connected: bool) -> SigmaSeparationStatus {
    if connected { SigmaSeparationStatus::NotSeparated } else { SigmaSeparationStatus::Separated }
}

fn finish(ends: QueryEnds, status: SigmaSeparationStatus) -> SigmaSeparation {
    SigmaSeparation {
        x: ends.x,
        y: ends.y,
        conditioning: ends.conditioning,
        status,
        assumptions: SigmaSeparationAssumptions::PROCEDURE,
    }
}

fn acyclification_connects(
    graph: &CyclicMixedGraph,
    ends: &QueryEnds,
    ws: &mut SigmaSeparationWorkspace,
) -> bool {
    if !ws.same_graph(graph) {
        ws.rebuild_acyclification(graph);
    }
    bayes_ball_reaches(
        &mut ws.dsep,
        BayesBallGraph {
            parents: &ws.parents,
            children: &ws.children,
            bidirected: &ws.bidirected,
            overlay: None,
        },
        &[ends.x],
        &[ends.y],
        &ends.conditioning,
    )
}

impl SigmaSeparationWorkspace {
    fn same_graph(&self, graph: &CyclicMixedGraph) -> bool {
        let Some(cache) = &self.cache else {
            return false;
        };
        cache.nodes == graph.node_count()
            && condensation_covers(cache)
            && cache.revision == graph.revision()
    }

    fn rebuild_acyclification(&mut self, graph: &CyclicMixedGraph) {
        self.rebuilds += 1;
        let n = graph.node_count();
        reset_rows(&mut self.parents, n);
        reset_rows(&mut self.children, n);
        reset_rows(&mut self.bidirected, n);
        let condensation = graph.condensation();
        let components = condensation.components().to_vec();
        let mut component_of = vec![usize::MAX; n];
        for (index, members) in components.iter().enumerate() {
            for &node in members {
                component_of[node.as_usize()] = index;
            }
        }
        fill_directed(graph, &component_of, &components, &mut self.children, &mut self.parents);
        fill_bidirected(graph, &component_of, &components, &mut self.bidirected);
        self.cache = Some(CachedAcyclification {
            nodes: n,
            revision: graph.revision(),
            component_of,
            components,
        });
    }
}

fn condensation_covers(cache: &CachedAcyclification) -> bool {
    cache.component_of.len() == cache.nodes
        && cache.components.iter().map(Vec::len).sum::<usize>() == cache.nodes
}

fn reset_rows(rows: &mut Vec<Vec<DenseNodeId>>, n: usize) {
    rows.resize(n, Vec::new());
    for row in rows {
        row.clear();
    }
}

fn component_parts(graph: &CyclicMixedGraph) -> (Vec<usize>, Vec<Vec<DenseNodeId>>) {
    let n = graph.node_count();
    let condensation = graph.condensation();
    let components = condensation.components().to_vec();
    let mut component_of = vec![usize::MAX; n];
    for (index, members) in components.iter().enumerate() {
        for &node in members {
            component_of[node.as_usize()] = index;
        }
    }
    debug_assert!(component_of.iter().all(|index| *index != usize::MAX));
    (component_of, components)
}

/// Directed edges `parent → member` for every outside parent of `member`'s component.
fn fill_directed(
    graph: &CyclicMixedGraph,
    component_of: &[usize],
    components: &[Vec<DenseNodeId>],
    children: &mut [Vec<DenseNodeId>],
    parents: &mut [Vec<DenseNodeId>],
) {
    let mut outside = Vec::new();
    for (index, members) in components.iter().enumerate() {
        outside.clear();
        for &member in members {
            for &parent in graph.parents(member) {
                if component_of[parent.as_usize()] != index {
                    insert_sorted(&mut outside, parent);
                }
            }
        }
        for &member in members {
            for &parent in &outside {
                link_directed(children, parents, parent, member);
            }
        }
    }
}

/// Bidirected clique inside each component, and a complete bipartite join across
/// components that the original graph already joins by a bidirected edge.
fn fill_bidirected(
    graph: &CyclicMixedGraph,
    component_of: &[usize],
    components: &[Vec<DenseNodeId>],
    rows: &mut [Vec<DenseNodeId>],
) {
    let mut links = vec![Vec::<usize>::new(); components.len()];
    for (index, members) in components.iter().enumerate() {
        for (offset, &left) in members.iter().enumerate() {
            for &right in &members[offset + 1..] {
                link_bidirected(rows, left, right);
            }
        }
        for &member in members {
            for &other in graph.bidirected_neighbors(member) {
                let other_component = component_of[other.as_usize()];
                if other_component != index {
                    insert_sorted(&mut links[index], other_component);
                }
            }
        }
    }
    for (left_component, neighbors) in links.iter().enumerate() {
        for &right_component in neighbors {
            if right_component <= left_component {
                continue;
            }
            for &left in &components[left_component] {
                for &right in &components[right_component] {
                    link_bidirected(rows, left, right);
                }
            }
        }
    }
}

fn link_directed(
    children: &mut [Vec<DenseNodeId>],
    parents: &mut [Vec<DenseNodeId>],
    from: DenseNodeId,
    to: DenseNodeId,
) {
    insert_sorted(&mut children[from.as_usize()], to);
    insert_sorted(&mut parents[to.as_usize()], from);
}

fn link_bidirected(rows: &mut [Vec<DenseNodeId>], a: DenseNodeId, b: DenseNodeId) {
    insert_sorted(&mut rows[a.as_usize()], b);
    insert_sorted(&mut rows[b.as_usize()], a);
}

fn walk_connects(graph: &CyclicMixedGraph, ends: &QueryEnds) -> bool {
    let (component_of, _) = component_parts(graph);
    let mut ancestral = BitSet::with_len(graph.node_count());
    let mut scratch = GraphWorkspace::default();
    closure_into(
        graph.node_count(),
        |node| graph.parents(node),
        ends.conditioning.iter().copied(),
        &mut ancestral,
        &mut scratch,
        |_from, _to| true,
    );
    open_simple_path(graph, ends, &component_of, &ancestral)
}

#[derive(Clone, Copy)]
struct Frame {
    node: DenseNodeId,
    arrival_head: bool,
    next_index: usize,
}

#[derive(Clone, Copy)]
struct Departure {
    next: DenseNodeId,
    /// Arrowhead at the node being left.
    head_at_source: bool,
    /// The step follows a directed edge from the node being left toward [`Self::next`].
    forward: bool,
    /// Arrowhead at [`Self::next`].
    head_at_next: bool,
}

fn open_simple_path(
    graph: &CyclicMixedGraph,
    ends: &QueryEnds,
    component_of: &[usize],
    ancestral: &BitSet,
) -> bool {
    let mut on_path = BitSet::with_len(graph.node_count());
    on_path.insert(ends.x);
    let mut stack = vec![Frame { node: ends.x, arrival_head: false, next_index: 0 }];
    while let Some(frame) = stack.last().copied() {
        if frame.next_index == departure_count(graph, frame.node) {
            on_path.remove(frame.node);
            stack.pop();
            continue;
        }
        stack.last_mut().expect("the frame being expanded is on the stack").next_index =
            frame.next_index + 1;
        let step = departure_at(graph, frame.node, frame.next_index);
        if on_path.contains(step.next) {
            continue;
        }
        if stack.len() >= 2 {
            let prev = stack[stack.len() - 2].node;
            if departure_blocks(
                component_of,
                ancestral,
                &ends.conditioned,
                prev,
                frame.node,
                frame.arrival_head,
                step,
            ) {
                continue;
            }
        }
        if step.next == ends.y {
            return true;
        }
        on_path.insert(step.next);
        stack.push(Frame { node: step.next, arrival_head: step.head_at_next, next_index: 0 });
    }
    false
}

fn departure_count(graph: &CyclicMixedGraph, node: DenseNodeId) -> usize {
    graph.children(node).len() + graph.parents(node).len() + graph.bidirected_neighbors(node).len()
}

fn departure_at(graph: &CyclicMixedGraph, node: DenseNodeId, index: usize) -> Departure {
    let children = graph.children(node);
    if let Some(&next) = children.get(index) {
        return Departure { next, head_at_source: false, forward: true, head_at_next: true };
    }
    let index = index - children.len();
    let parents = graph.parents(node);
    if let Some(&next) = parents.get(index) {
        return Departure { next, head_at_source: true, forward: false, head_at_next: false };
    }
    let index = index - parents.len();
    let next = graph
        .bidirected_neighbors(node)
        .get(index)
        .copied()
        .expect("departure index is inside the node's degree");
    Departure { next, head_at_source: true, forward: false, head_at_next: true }
}

/// Definition A.16, clauses 2 and 3, at a non-endpoint.
fn departure_blocks(
    component_of: &[usize],
    ancestral: &BitSet,
    conditioning: &BitSet,
    prev: DenseNodeId,
    middle: DenseNodeId,
    arrival_head: bool,
    step: Departure,
) -> bool {
    let collider = arrival_head && step.head_at_source;
    if collider {
        return !ancestral.contains(middle);
    }
    if !conditioning.contains(middle) {
        return false;
    }
    let middle_component = component_of[middle.as_usize()];
    let left_points_out = !arrival_head && component_of[prev.as_usize()] != middle_component;
    let right_points_out = step.forward && component_of[step.next.as_usize()] != middle_component;
    left_points_out || right_points_out
}

#[cfg(test)]
mod tests {
    use super::{
        EXHAUSTIVE_MIXED_NODE_BOUND, EXHAUSTIVE_NODE_BOUND, OrdinaryDSeparationOnCycles,
        SigmaSeparationAssumptions, SigmaSeparationStatus, SigmaSeparationWorkspace,
        WALK_ORACLE_NODE_BOUND, sigma_separates, sigma_walk_oracle,
    };
    use crate::cyclic::CyclicMixedGraph;
    use crate::dag::Dag;
    use crate::dsep::DSeparationWorkspace;
    use crate::error::GraphError;
    use crate::types::DenseNodeId;

    fn n(raw: u32) -> DenseNodeId {
        DenseNodeId::from_raw(raw)
    }

    fn mixed(nodes: u32, directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> CyclicMixedGraph {
        let mut graph = CyclicMixedGraph::with_variables(nodes);
        for &(from, to) in directed {
            graph.insert_directed(n(from), n(to)).unwrap();
        }
        for &(a, b) in bidirected {
            graph.insert_bidirected(n(a), n(b)).unwrap();
        }
        graph
    }

    fn assert_both(graph: &CyclicMixedGraph, x: u32, y: u32, z: &[u32], separated: bool) {
        let conditioning: Vec<DenseNodeId> = z.iter().copied().map(n).collect();
        let mut ws = SigmaSeparationWorkspace::new();
        let engine = sigma_separates(graph, n(x), n(y), &conditioning, &mut ws).unwrap();
        let oracle = sigma_walk_oracle(graph, n(x), n(y), &conditioning).unwrap();
        assert_eq!(engine, oracle);
        assert_eq!(engine.separated(), separated);
        assert_eq!(engine.assumptions, SigmaSeparationAssumptions::PROCEDURE);
    }

    #[test]
    fn acyclic_chain_fork_collider_and_conditioned_collider_match_d_separation() {
        let cases = [
            (3, &[(0, 1), (1, 2)][..], 0, 2, &[][..], false),
            (3, &[(0, 1), (1, 2)][..], 0, 2, &[1][..], true),
            (3, &[(1, 0), (1, 2)][..], 0, 2, &[][..], false),
            (3, &[(1, 0), (1, 2)][..], 0, 2, &[1][..], true),
            (3, &[(0, 1), (2, 1)][..], 0, 2, &[][..], true),
            (3, &[(0, 1), (2, 1)][..], 0, 2, &[1][..], false),
            (4, &[(0, 1), (2, 1), (1, 3)][..], 0, 2, &[][..], true),
            (4, &[(0, 1), (2, 1), (1, 3)][..], 0, 2, &[1][..], false),
            (4, &[(0, 1), (2, 1), (1, 3)][..], 0, 2, &[3][..], false),
            (4, &[(0, 1), (2, 1), (1, 3)][..], 0, 2, &[1, 3][..], false),
        ];
        let mut ws = SigmaSeparationWorkspace::new();
        let mut dws = DSeparationWorkspace::default();
        for (nodes, directed, x, y, z, separated) in cases {
            let graph = mixed(nodes, directed, &[]);
            let dag = dag_of(&graph);
            let conditioning: Vec<DenseNodeId> = z.iter().copied().map(n).collect();
            for (left, right) in [(x, y), (y, x)] {
                let engine =
                    sigma_separates(&graph, n(left), n(right), &conditioning, &mut ws).unwrap();
                let oracle = sigma_walk_oracle(&graph, n(left), n(right), &conditioning).unwrap();
                assert_eq!(engine.status, oracle.status);
                assert_eq!(engine.separated(), separated);
                let d_separated =
                    dag.is_d_separated(n(left), n(right), &conditioning, &mut dws).unwrap();
                assert_eq!(engine.separated(), d_separated);
            }
        }
    }

    #[test]
    fn engine_matches_walk_oracle_on_every_mixed_graph_of_three_nodes() {
        assert_eq!(EXHAUSTIVE_MIXED_NODE_BOUND, 3);
        exhaust(EXHAUSTIVE_MIXED_NODE_BOUND, true, true);
    }

    #[test]
    fn engine_matches_walk_oracle_on_every_directed_graph_of_four_nodes() {
        assert_eq!(EXHAUSTIVE_NODE_BOUND, 4);
        exhaust(EXHAUSTIVE_NODE_BOUND, false, false);
    }

    #[test]
    fn forre_mooij_cycle_stays_sigma_open_given_both_cycle_nodes() {
        // X1 → X3 ⇄ X4 ← X2, with X1 = 0, X2 = 1, X3 = 2, X4 = 3.
        let graph = mixed(4, &[(0, 2), (3, 2), (1, 3), (2, 3)], &[]);
        assert_both(&graph, 0, 1, &[2, 3], false);
        assert_both(&graph, 1, 0, &[2, 3], false);
        assert_both(&graph, 0, 1, &[], true);
    }

    #[test]
    fn noncollider_inside_a_strongly_connected_component_does_not_block() {
        let graph = mixed(3, &[(0, 1), (1, 2), (2, 0)], &[]);
        assert_both(&graph, 0, 2, &[1], false);
    }

    #[test]
    fn noncollider_blocks_when_it_points_out_of_its_component() {
        let graph = mixed(3, &[(0, 1), (1, 0), (1, 2)], &[]);
        assert_both(&graph, 0, 2, &[1], true);
        assert_both(&graph, 0, 2, &[], false);
    }

    #[test]
    fn endpoint_in_the_conditioning_set_blocks() {
        let graph = mixed(3, &[(0, 1), (1, 2)], &[]);
        assert_both(&graph, 0, 2, &[2], true);
        assert_both(&graph, 0, 2, &[0], true);
    }

    #[test]
    fn same_node_follows_the_trivial_walk() {
        let graph = mixed(2, &[(0, 1)], &[]);
        assert_both(&graph, 0, 0, &[], false);
        assert_both(&graph, 0, 0, &[0], true);
    }

    #[test]
    fn assumptions_do_not_claim_simple_scm_or_sigma_faithfulness() {
        let graph = mixed(2, &[(0, 1)], &[]);
        let mut ws = SigmaSeparationWorkspace::new();
        let answer = sigma_separates(&graph, n(0), n(1), &[], &mut ws).unwrap();
        assert_eq!(answer.assumptions, SigmaSeparationAssumptions::PROCEDURE);
        assert!(answer.assumptions.observational_markov_requires_unique_solvability_per_component);
        assert!(!answer.assumptions.simple_scm_assumed);
        assert!(!answer.assumptions.sigma_faithfulness_assumed);
        assert_eq!(
            answer.assumptions.ordinary_d_separation_on_cycles,
            OrdinaryDSeparationOnCycles::NotSettled,
        );
        assert_eq!(answer.assumptions.reading, SigmaSeparationAssumptions::READING);
    }

    #[test]
    fn not_separated_is_not_a_proof_of_dependence() {
        let graph = mixed(4, &[(0, 2), (3, 2), (1, 3), (2, 3)], &[]);
        let mut ws = SigmaSeparationWorkspace::new();
        let answer = sigma_separates(&graph, n(0), n(1), &[n(2), n(3)], &mut ws).unwrap();
        assert_eq!(answer.status, SigmaSeparationStatus::NotSeparated);
        assert!(!answer.separated());
        assert!(answer.assumptions.reading.contains("not a proof of dependence"));
        assert!(!answer.assumptions.sigma_faithfulness_assumed);
    }

    #[test]
    fn walk_oracle_refuses_graphs_past_the_node_bound() {
        assert_eq!(WALK_ORACLE_NODE_BOUND, EXHAUSTIVE_NODE_BOUND);
        let nodes = u32::try_from(WALK_ORACLE_NODE_BOUND).unwrap() + 1;
        let graph = mixed(nodes, &[], &[]);
        let error = sigma_walk_oracle(&graph, n(0), n(1), &[]).unwrap_err();
        assert_eq!(error, GraphError::NodeBoundExceeded { nodes: 5, bound: 4 },);
    }

    #[test]
    fn engine_matches_d_separation_on_a_five_node_chain() {
        let graph = mixed(5, &[(0, 1), (1, 2), (2, 3), (3, 4)], &[]);
        let dag = dag_of(&graph);
        let mut ws = SigmaSeparationWorkspace::new();
        let mut dws = DSeparationWorkspace::default();
        for z in [&[][..], &[2][..]] {
            let conditioning: Vec<DenseNodeId> = z.iter().copied().map(n).collect();
            let engine = sigma_separates(&graph, n(0), n(4), &conditioning, &mut ws).unwrap();
            let d_separated = dag.is_d_separated(n(0), n(4), &conditioning, &mut dws).unwrap();
            assert_eq!(engine.separated(), d_separated);
            assert_eq!(engine.separated(), z == [2u32].as_slice());
        }
        assert!(sigma_walk_oracle(&graph, n(0), n(4), &[]).is_err());
    }

    #[test]
    fn bidirected_chain_is_blocked_until_the_collider_is_conditioned() {
        let graph = mixed(3, &[], &[(0, 1), (1, 2)]);
        assert_both(&graph, 0, 2, &[], true);
        assert_both(&graph, 0, 2, &[1], false);
        assert_both(&graph, 0, 1, &[], false);
    }

    #[test]
    fn direct_latent_edge_bypasses_a_conditioned_noncollider() {
        let open = mixed(3, &[(0, 1), (1, 2)], &[(0, 2)]);
        let blocked = mixed(3, &[(0, 1), (1, 2)], &[]);
        assert_both(&open, 0, 2, &[1], false);
        assert_both(&blocked, 0, 2, &[1], true);
    }

    #[test]
    fn unknown_node_is_rejected() {
        let graph = mixed(2, &[(0, 1)], &[]);
        let mut ws = SigmaSeparationWorkspace::new();
        assert_eq!(
            sigma_separates(&graph, n(0), n(4), &[], &mut ws).unwrap_err(),
            GraphError::UnknownNode { id: 4 },
        );
        assert_eq!(
            sigma_walk_oracle(&graph, n(0), n(1), &[n(3)]).unwrap_err(),
            GraphError::UnknownNode { id: 3 },
        );
    }

    #[test]
    fn repeated_queries_reuse_one_graph_and_a_changed_graph_rebuilds() {
        let mut graph = mixed(4, &[(0, 1), (1, 2), (2, 0), (2, 3)], &[(0, 3)]);
        let mut ws = SigmaSeparationWorkspace::new();
        let first = sigma_separates(&graph, n(0), n(3), &[], &mut ws).unwrap();
        assert_eq!(ws.rebuilds, 1);
        let again = sigma_separates(&graph, n(0), n(3), &[n(1)], &mut ws).unwrap();
        assert_eq!(ws.rebuilds, 1);
        assert_eq!(first.assumptions, again.assumptions);
        let oracle = sigma_walk_oracle(&graph, n(0), n(3), &[n(1)]).unwrap();
        assert_eq!(again, oracle);

        graph.insert_directed(n(3), n(0)).unwrap();
        let changed = sigma_separates(&graph, n(0), n(3), &[], &mut ws).unwrap();
        assert_eq!(ws.rebuilds, 2);
        assert_eq!(changed, sigma_walk_oracle(&graph, n(0), n(3), &[]).unwrap());

        let other = mixed(3, &[(0, 1), (1, 2)], &[]);
        let blocked = sigma_separates(&other, n(0), n(2), &[n(1)], &mut ws).unwrap();
        assert_eq!(ws.rebuilds, 3);
        assert!(blocked.separated());
        assert_eq!(blocked, sigma_walk_oracle(&other, n(0), n(2), &[n(1)]).unwrap());
        let reused = sigma_separates(&other, n(0), n(2), &[], &mut ws).unwrap();
        assert_eq!(ws.rebuilds, 3);
        assert!(!reused.separated());
    }

    fn exhaust(nodes: usize, bidirected: bool, include_endpoints: bool) {
        let n = u32::try_from(nodes).unwrap();
        let directed = directed_pairs(n);
        let bidirected_edges = if bidirected { undirected_pairs(n) } else { Vec::new() };
        let mut ws = SigmaSeparationWorkspace::new();
        let mut dws = DSeparationWorkspace::default();
        for dir_mask in 0..mask_limit(directed.len()) {
            for bi_mask in 0..mask_limit(bidirected_edges.len()) {
                let graph = graph_from_masks(n, &directed, dir_mask, &bidirected_edges, bi_mask);
                let acyclic =
                    graph.condensation().components().iter().all(|members| members.len() == 1);
                let latent = graph.node_ids().any(|id| !graph.bidirected_neighbors(id).is_empty());
                let dag = if acyclic && !latent { Some(dag_of(&graph)) } else { None };
                compare_graph(&graph, dag.as_ref(), include_endpoints, &mut ws, &mut dws);
            }
        }
    }

    fn compare_graph(
        graph: &CyclicMixedGraph,
        dag: Option<&Dag>,
        include_endpoints: bool,
        ws: &mut SigmaSeparationWorkspace,
        dws: &mut DSeparationWorkspace,
    ) {
        let nodes: Vec<DenseNodeId> =
            (0..graph.node_count()).map(DenseNodeId::from_index).collect();
        let mut pool = Vec::new();
        let mut chosen = Vec::new();
        for (index, &x) in nodes.iter().enumerate() {
            for &y in &nodes[index + 1..] {
                pool.clear();
                for &node in &nodes {
                    if include_endpoints || (node != x && node != y) {
                        pool.push(node);
                    }
                }
                let subset_count = mask_limit(pool.len());
                for mask in 0..subset_count {
                    chosen.clear();
                    for (bit, &node) in pool.iter().enumerate() {
                        if mask_has(mask, bit) {
                            chosen.push(node);
                        }
                    }
                    let engine = sigma_separates(graph, x, y, &chosen, ws).unwrap();
                    let oracle = sigma_walk_oracle(graph, x, y, &chosen).unwrap();
                    assert_eq!(
                        engine,
                        oracle,
                        "directed {:?} bidirected {:?} x {} y {} z {:?}",
                        graph.directed_edge_pairs_sorted(),
                        graph.bidirected_edge_pairs_sorted(),
                        x.raw(),
                        y.raw(),
                        chosen,
                    );
                    let Some(dag) = dag else {
                        continue;
                    };
                    if chosen.contains(&x) || chosen.contains(&y) {
                        continue;
                    }
                    let d_separated = dag.is_d_separated(x, y, &chosen, dws).unwrap();
                    assert_eq!(
                        engine.separated(),
                        d_separated,
                        "directed {:?} x {} y {} z {:?}",
                        graph.directed_edge_pairs_sorted(),
                        x.raw(),
                        y.raw(),
                        chosen,
                    );
                }
            }
        }
    }

    fn dag_of(graph: &CyclicMixedGraph) -> Dag {
        let nodes = u32::try_from(graph.node_count()).unwrap();
        let mut dag = Dag::with_variables(nodes);
        for (from, to) in graph.directed_edge_pairs_sorted() {
            dag.insert_directed(n(from), n(to)).unwrap();
        }
        dag
    }

    fn directed_pairs(nodes: u32) -> Vec<(u32, u32)> {
        let mut edges = Vec::new();
        for from in 0..nodes {
            for to in 0..nodes {
                if from != to {
                    edges.push((from, to));
                }
            }
        }
        edges
    }

    fn undirected_pairs(nodes: u32) -> Vec<(u32, u32)> {
        let mut edges = Vec::new();
        for left in 0..nodes {
            for right in (left + 1)..nodes {
                edges.push((left, right));
            }
        }
        edges
    }

    fn graph_from_masks(
        nodes: u32,
        directed: &[(u32, u32)],
        dir_mask: u32,
        bidirected: &[(u32, u32)],
        bi_mask: u32,
    ) -> CyclicMixedGraph {
        let mut graph = CyclicMixedGraph::with_variables(nodes);
        for (index, &(from, to)) in directed.iter().enumerate() {
            if mask_has(dir_mask, index) {
                graph.insert_directed(n(from), n(to)).unwrap();
            }
        }
        for (index, &(left, right)) in bidirected.iter().enumerate() {
            if mask_has(bi_mask, index) {
                graph.insert_bidirected(n(left), n(right)).unwrap();
            }
        }
        graph
    }

    fn mask_limit(bits: usize) -> u32 {
        let width = u32::try_from(bits).unwrap();
        assert!(width < u32::BITS, "mask fits in u32");
        1u32 << width
    }

    fn mask_has(mask: u32, index: usize) -> bool {
        let shift = u32::try_from(index).unwrap();
        mask & (1u32 << shift) != 0
    }
}
