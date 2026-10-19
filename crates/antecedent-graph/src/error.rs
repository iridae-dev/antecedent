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
    let mut indegree = vec![0usize; node_count];
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
    }
    let mut ready: Vec<usize> =
        indegree.iter().enumerate().filter(|(_, degree)| **degree == 0).map(|(i, _)| i).collect();
    let mut placed = 0usize;
    while let Some(node) = ready.pop() {
        placed += 1;
        for &(from, to) in edges {
            if from as usize == node && (to as usize) < node_count && from != to {
                let left = &mut indegree[to as usize];
                *left -= 1;
                if *left == 0 {
                    ready.push(to as usize);
                }
            }
        }
    }
    placed != node_count
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
        Err(CyclicDagRefusal { operation, next_attempt })
    } else {
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::{CyclicNextAttempt, DagOperation, refuse_directed_cycle};

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
