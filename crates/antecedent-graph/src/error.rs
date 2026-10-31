//! Graph construction and validation errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{Lag, VariableId};
use thiserror::Error;

/// Graph-layer errors.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum GraphError {
    /// Unknown dense node.
    #[error("unknown dense node {id}")]
    UnknownNode {
        /// Dense id.
        id: u32,
    },
    /// Unknown variable name at an API boundary.
    #[error("unknown variable name '{name}'")]
    UnknownVariableName {
        /// Requested name.
        name: String,
    },
    /// Edge would introduce a directed cycle.
    ///
    /// The edge is not inserted. The next attempt for the feedback structure is
    /// [`CyclicDigraph::condensation`](crate::CyclicDigraph::condensation) or
    /// [`CyclicMixedGraph::condensation`](crate::CyclicMixedGraph::condensation).
    /// A conditional-independence question on that graph uses
    /// [`sigma_separates`](crate::sigma_separates).
    #[error(
        "edge {from}->{to} would create a cycle; next attempt: \
         CyclicDigraph::condensation or CyclicMixedGraph::condensation for the feedback \
         structure, or sigma_separates on a CyclicMixedGraph for a conditional-independence \
         question. The edge was not inserted"
    )]
    Cycle {
        /// Source dense id.
        from: u32,
        /// Target dense id.
        to: u32,
    },
    /// Invalid endpoint combination for this graph class.
    #[error("invalid endpoints: {message}")]
    InvalidEndpoints {
        /// Explanation.
        message: &'static str,
    },
    /// A routine with a fixed node bound was given a larger graph.
    ///
    /// This is a size cap, not a problem with any endpoint.
    #[error("graph has {nodes} nodes; this routine accepts at most {bound}")]
    NodeBoundExceeded {
        /// Node count of the graph that was offered.
        nodes: usize,
        /// Largest node count the routine accepts.
        bound: usize,
    },
    /// Contemporaneous self-edge is invalid.
    #[error("contemporaneous self-edge on {variable}")]
    ContemporaneousSelfEdge {
        /// Variable.
        variable: VariableId,
    },
    /// Duplicate edge.
    #[error("duplicate edge {from}->{to}")]
    DuplicateEdge {
        /// From.
        from: u32,
        /// To.
        to: u32,
    },
    /// Lagged self-edge with lag 0.
    #[error("invalid lag {lag}")]
    InvalidLag {
        /// Lag value.
        lag: Lag,
    },
    /// Edge points from the future into the past (source lag nearer the present
    /// than target lag).
    #[error("edge {from}->{to} points from the future ({from_lag}) into the past ({to_lag})")]
    FutureToPast {
        /// Source dense id.
        from: u32,
        /// Target dense id.
        to: u32,
        /// Source lag.
        from_lag: Lag,
        /// Target lag.
        to_lag: Lag,
    },
    /// A directed edge into a context node.
    ///
    /// A context is an exogenous regime or environment indicator: it may cause system
    /// variables but is never caused by them, so no system node may be the parent of a
    /// context node. The edge is not inserted and no undirected edge is oriented.
    #[error(
        "edge {from}->{to} points into a context node; a context is exogenous and may only \
         cause system nodes. The edge was not inserted"
    )]
    ContextCaused {
        /// Source dense id (a system node).
        from: u32,
        /// Target dense id (the context node).
        to: u32,
    },
    /// A context node that a [`TemporalDag`](crate::TemporalDag) cannot hold.
    ///
    /// A `TemporalDag` holds lagged nodes only. An untagged context converts to the
    /// contemporaneous slot of its variable; a context that carries an environment tag, or
    /// whose variable already occupies that slot, has no faithful lagged representation.
    #[error("context node {node} cannot be represented in a TemporalDag: {reason}")]
    ContextNotRepresentable {
        /// Dense id of the context node.
        node: u32,
        /// Why the node has no lagged representation.
        reason: &'static str,
    },
    /// Temporal graph materialization was cancelled or exceeded its caller's admission.
    #[error(transparent)]
    TemporalMaterialization(#[from] antecedent_core::TemporalMaterializationError),
    /// Node capacity exceeded.
    #[error("too many nodes")]
    TooManyNodes,
    /// Invalid selection-diagram metadata.
    #[error("invalid selection diagram: {message}")]
    InvalidSelectionDiagram {
        /// Explanation.
        message: String,
    },
    /// Bounded path search hit `max_paths` or `max_len` before exploring all candidates.
    ///
    /// Returned when m-separation would otherwise conclude "separated" after an incomplete
    /// search (an unexplored active path may still exist). Finding an active path remains
    /// conclusive even under truncation.
    #[error("path search budget exhausted (max_paths={max_paths}, max_len={max_len})")]
    SearchBudgetExhausted {
        /// Path-count budget.
        max_paths: usize,
        /// Path-length budget.
        max_len: usize,
    },
    /// PAG m-separation holds in some members of the equivalence class and fails in
    /// others, or the class could not be enumerated; neither "separated" nor
    /// "connected" is a statement about every member.
    #[error("m-separation is not determined for every MAG in the PAG's class")]
    SeparationUndetermined,
}

/// A DAG-only operation that refused a directed cycle.
///
/// The refusal is one failed attempt. It does not say that a cyclic graph has
/// no conditional independence and no feedback structure.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum DagOperation {
    /// A topological order of the directed edges.
    TopologicalOrder,
    /// Coercion of directed edges into a [`crate::Dag`].
    DagProjection,
    /// An adjustment set that assumes acyclicity.
    Adjustment,
    /// An identification rule that assumes acyclicity.
    Identification,
    /// Sampling or structural evaluation along a topological order.
    Simulation,
    /// Noise inference along a topological order.
    Abduction,
    /// A discovery search whose state is a DAG.
    Discovery,
    /// A graph posterior whose atoms are read as DAGs.
    GraphPosterior,
}

impl DagOperation {
    /// Stable name of the refused operation.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::TopologicalOrder => "topological order",
            Self::DagProjection => "DAG projection",
            Self::Adjustment => "adjustment",
            Self::Identification => "identification",
            Self::Simulation => "simulation",
            Self::Abduction => "abduction",
            Self::Discovery => "discovery",
            Self::GraphPosterior => "graph posterior",
        }
    }
}

impl std::fmt::Display for DagOperation {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str(self.as_str())
    }
}

/// What to try after a DAG operation refuses a directed cycle.
///
/// [`Self::SigmaSeparation`] is the licensed conditional-independence query.
/// [`Self::Condensation`] is the licensed reading of feedback structure.
/// [`Self::Unlicensed`] names the consumer that has no cyclic license yet and
/// the attempt that consumer still owes.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
#[non_exhaustive]
pub enum CyclicNextAttempt {
    /// `sigma_separates` on a [`crate::CyclicMixedGraph`].
    ///
    /// [`SigmaSeparationStatus::NotSeparated`](crate::SigmaSeparationStatus::NotSeparated)
    /// is not a proof of dependence.
    SigmaSeparation,
    /// [`crate::CyclicDigraph::condensation`] or
    /// [`crate::CyclicMixedGraph::condensation`].
    ///
    /// The schedule is the feedback structure. It is not an identification,
    /// a sample, or an abduction.
    Condensation,
    /// This consumer has no cyclic license. `attempt` is the design to try.
    Unlicensed {
        /// The DAG consumer that refused.
        consumer: &'static str,
        /// The next design, named explicitly.
        attempt: &'static str,
    },
}

impl std::fmt::Display for CyclicNextAttempt {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::SigmaSeparation => f.write_str(
                "sigma_separates on a CyclicMixedGraph for the conditional-independence question",
            ),
            Self::Condensation => f.write_str(
                "CyclicDigraph::condensation or CyclicMixedGraph::condensation for the feedback \
                 structure",
            ),
            Self::Unlicensed { consumer, attempt } => {
                write!(f, "{consumer} has no cyclic license; next attempt: {attempt}")
            }
        }
    }
}

/// A DAG operation refused directed edges that contain a cycle.
///
/// Callers keep the acyclic algorithm for an edge list this check accepts.
/// A refusal is not a topological order and not a projection that drops the cycle.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Error)]
#[error("{operation} refused a directed cycle; next attempt: {next_attempt}")]
pub struct CyclicDagRefusal {
    /// The operation that did not run.
    pub operation: DagOperation,
    /// The legal next attempt for the same request.
    pub next_attempt: CyclicNextAttempt,
}

impl CyclicDagRefusal {
    /// Refusal of `operation`, naming `next_attempt`.
    ///
    /// [`Display`](std::fmt::Display) is the prose a DAG consumer returns. Copy
    /// that text from here rather than writing another sentence.
    #[must_use]
    pub const fn new(operation: DagOperation, next_attempt: CyclicNextAttempt) -> Self {
        Self { operation, next_attempt }
    }

    /// Refusal of `operation` that points at condensation for the feedback structure.
    #[must_use]
    pub const fn condensation(operation: DagOperation) -> Self {
        Self::new(operation, CyclicNextAttempt::Condensation)
    }

    /// Refusal of `operation` by a `consumer` that has no cyclic license.
    ///
    /// `attempt` names the next design for the same request.
    #[must_use]
    pub const fn unlicensed(
        operation: DagOperation,
        consumer: &'static str,
        attempt: &'static str,
    ) -> Self {
        Self::new(operation, CyclicNextAttempt::Unlicensed { consumer, attempt })
    }
}

/// Whether `edges` contain a directed cycle on nodes `0..node_count`.
///
/// A self-loop is a cycle. An endpoint at or past `node_count` is ignored here;
/// the caller reports that endpoint on its own. This check does not return an
/// order. An acyclic list is then passed to the caller's existing algorithm.
#[must_use = "a directed cycle must be refused before a DAG algorithm runs"]
pub fn directed_edges_contain_cycle(node_count: usize, edges: &[(u32, u32)]) -> bool {
    if node_count == 0 {
        return false;
    }
    // Kahn's algorithm over a compressed-row adjacency built once: O(V + E).
    let mut indegree = vec![0usize; node_count];
    let mut start = vec![0usize; node_count + 1];
    for &(from, to) in edges {
        let from_i = from as usize;
        let to_i = to as usize;
        if from_i >= node_count || to_i >= node_count {
            continue;
        }
        if from == to {
            return true;
        }
        indegree[to_i] += 1;
        start[from_i + 1] += 1;
    }
    for node in 0..node_count {
        start[node + 1] += start[node];
    }
    let mut fill = start.clone();
    let mut targets = vec![0usize; start[node_count]];
    for &(from, to) in edges {
        let from_i = from as usize;
        let to_i = to as usize;
        if from_i >= node_count || to_i >= node_count {
            continue;
        }
        targets[fill[from_i]] = to_i;
        fill[from_i] += 1;
    }
    let mut ready: Vec<usize> =
        indegree.iter().enumerate().filter(|(_, degree)| **degree == 0).map(|(i, _)| i).collect();
    let mut placed = 0usize;
    while let Some(node) = ready.pop() {
        placed += 1;
        for &to in &targets[start[node]..start[node + 1]] {
            indegree[to] -= 1;
            if indegree[to] == 0 {
                ready.push(to);
            }
        }
    }
    placed != node_count
}

/// Whether `edges` contain a directed cycle when the ids are not `0..n`.
///
/// Each distinct endpoint is a node. The ids are remapped onto `0..k` and
/// [`directed_edges_contain_cycle`] decides. A self-loop is a cycle. An empty
/// list is acyclic. This check does not return an order.
#[must_use = "a directed cycle must be refused before a DAG algorithm runs"]
pub fn sparse_directed_edges_contain_cycle(edges: &[(u32, u32)]) -> bool {
    if edges.iter().any(|&(from, to)| from == to) {
        return true;
    }
    if edges.is_empty() {
        return false;
    }
    let mut ids = Vec::with_capacity(edges.len().saturating_mul(2));
    for &(from, to) in edges {
        ids.push(from);
        ids.push(to);
    }
    ids.sort_unstable();
    ids.dedup();
    let mut dense = Vec::with_capacity(edges.len());
    for &(from, to) in edges {
        let from_at = ids.binary_search(&from).expect("endpoint was collected");
        let to_at = ids.binary_search(&to).expect("endpoint was collected");
        let from_id = u32::try_from(from_at).expect("distinct endpoint count fits u32");
        let to_id = u32::try_from(to_at).expect("distinct endpoint count fits u32");
        dense.push((from_id, to_id));
    }
    directed_edges_contain_cycle(ids.len(), &dense)
}

/// Refuse `operation` when `edges` contain a directed cycle.
///
/// `Ok(())` means the caller's existing acyclic algorithm may run. `Err` means
/// that algorithm must not run.
///
/// # Errors
///
/// [`CyclicDagRefusal`] when the directed edges contain a cycle.
#[must_use = "the refusal is the failed DAG attempt"]
pub fn refuse_directed_cycle(
    operation: DagOperation,
    next_attempt: CyclicNextAttempt,
    node_count: usize,
    edges: &[(u32, u32)],
) -> Result<(), CyclicDagRefusal> {
    if directed_edges_contain_cycle(node_count, edges) {
        Err(CyclicDagRefusal::new(operation, next_attempt))
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{
        CyclicDagRefusal, CyclicNextAttempt, DagOperation, directed_edges_contain_cycle,
        refuse_directed_cycle, sparse_directed_edges_contain_cycle,
    };

    /// Cycle test by repeated reachability: a cycle exists exactly when some node reaches
    /// itself through at least one edge. Quadratic and independent of the Kahn pass.
    fn brute_force_has_cycle(node_count: usize, edges: &[(u32, u32)]) -> bool {
        let edges: Vec<(usize, usize)> = edges
            .iter()
            .map(|&(from, to)| (from as usize, to as usize))
            .filter(|&(from, to)| from < node_count && to < node_count)
            .collect();
        for start in 0..node_count {
            let mut seen = vec![false; node_count];
            let mut stack = vec![start];
            while let Some(node) = stack.pop() {
                for &(from, to) in &edges {
                    if from != node {
                        continue;
                    }
                    if to == start {
                        return true;
                    }
                    if !seen[to] {
                        seen[to] = true;
                        stack.push(to);
                    }
                }
            }
        }
        false
    }

    #[test]
    fn kahn_agrees_with_reachability_on_every_small_graph() {
        // Every edge subset of the 16 ordered pairs on four nodes, self-loops included,
        // plus a duplicated edge and an out-of-range endpoint in each case.
        let pairs: Vec<(u32, u32)> =
            (0..4).flat_map(|from| (0..4).map(move |to| (from, to))).collect();
        for mask in 0u32..(1 << pairs.len()) {
            let mut edges: Vec<(u32, u32)> = pairs
                .iter()
                .enumerate()
                .filter(|(bit, _)| mask & (1 << bit) != 0)
                .map(|(_, pair)| *pair)
                .collect();
            for variant in 0..3 {
                match variant {
                    1 => {
                        if let Some(&first) = edges.first() {
                            edges.push(first);
                        }
                    }
                    2 => edges.push((0, 9)),
                    _ => {}
                }
                assert_eq!(
                    directed_edges_contain_cycle(4, &edges),
                    brute_force_has_cycle(4, &edges),
                    "edges {edges:?}"
                );
            }
        }
    }

    #[test]
    fn a_long_path_and_a_closed_long_path_are_decided_in_linear_time() {
        let n = 200_000u32;
        let mut edges: Vec<(u32, u32)> = (0..n - 1).map(|i| (i, i + 1)).collect();
        assert!(!directed_edges_contain_cycle(n as usize, &edges));
        edges.push((n - 1, 0));
        assert!(directed_edges_contain_cycle(n as usize, &edges));
    }

    #[test]
    fn sparse_ids_remap_onto_the_dense_cycle_test() {
        assert!(!sparse_directed_edges_contain_cycle(&[]));
        assert!(!sparse_directed_edges_contain_cycle(&[(10, 20), (20, 40)]));
        assert!(!sparse_directed_edges_contain_cycle(&[(4, 8), (4, 8)]));
        assert!(sparse_directed_edges_contain_cycle(&[(7, 7)]));
        assert!(sparse_directed_edges_contain_cycle(&[(5, 9), (9, 5)]));
        assert!(sparse_directed_edges_contain_cycle(&[(1, 1), (1, 2)]));
        let sparse = [(100, 300), (300, 200), (200, 100)];
        assert!(sparse_directed_edges_contain_cycle(&sparse));
        assert!(!directed_edges_contain_cycle(3, &[(0, 1), (1, 2)]));
        let refusal =
            CyclicDagRefusal::new(DagOperation::Discovery, CyclicNextAttempt::Condensation);
        let text = refusal.to_string();
        assert!(text.contains("discovery"), "{text}");
        assert!(text.contains("condensation"), "{text}");
        assert!(text.contains("CyclicDigraph"), "{text}");
    }

    #[test]
    fn refusal_constructors_match_the_explicit_forms() {
        assert_eq!(
            CyclicDagRefusal::condensation(DagOperation::Discovery),
            CyclicDagRefusal::new(DagOperation::Discovery, CyclicNextAttempt::Condensation),
        );
        assert_eq!(
            CyclicDagRefusal::unlicensed(DagOperation::Simulation, "example", "try this"),
            CyclicDagRefusal::new(
                DagOperation::Simulation,
                CyclicNextAttempt::Unlicensed { consumer: "example", attempt: "try this" },
            ),
        );
    }

    #[test]
    fn two_cycle_is_refused_and_a_chain_is_accepted() {
        let refusal = refuse_directed_cycle(
            DagOperation::TopologicalOrder,
            CyclicNextAttempt::Condensation,
            2,
            &[(0, 1), (1, 0)],
        )
        .unwrap_err();
        let text = refusal.to_string();
        assert!(text.contains("topological order"), "{text}");
        assert!(text.contains("condensation"), "{text}");
        assert!(text.contains("CyclicDigraph"), "{text}");
        assert!(
            refuse_directed_cycle(
                DagOperation::TopologicalOrder,
                CyclicNextAttempt::Condensation,
                3,
                &[(0, 1), (1, 2)],
            )
            .is_ok()
        );
    }

    #[test]
    fn a_self_loop_is_a_cycle_and_an_unlicensed_consumer_names_its_attempt() {
        let refusal = refuse_directed_cycle(
            DagOperation::Identification,
            CyclicNextAttempt::Unlicensed {
                consumer: "example",
                attempt: "hold the edges as a CyclicDigraph",
            },
            1,
            &[(0, 0)],
        )
        .unwrap_err();
        let text = refusal.to_string();
        assert!(text.contains("example has no cyclic license"), "{text}");
        assert!(text.contains("CyclicDigraph"), "{text}");
    }
}
