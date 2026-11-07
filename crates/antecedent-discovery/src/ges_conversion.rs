// SPDX-License-Identifier: MIT OR Apache-2.0
//! Construct essential graphs from DAGs and orientation knowledge.

use antecedent_graph::{Cpdag, Dag, DenseNodeId};

use super::DagEdgeHelpers;
use crate::error::DiscoveryError;
use crate::orientation::{
    MeekR1, MeekR2, MeekR3, MeekR4, OrientationRule, OrientationState,
    run_static_orientation_to_fixed_point,
};

/// Essential graph: skeleton + DAG v-structures + Meek R1–R4.
pub(crate) fn dag_to_cpdag(dag: &Dag) -> Result<Cpdag, DiscoveryError> {
    dag_to_cpdag_with_known(dag, |_, _| false)
}

/// Essential graph under known orientations: the edges `known(from, to)` names start directed
/// (time order, for an unrolled window), then v-structures and Meek R1–R4 orient the rest.
pub(crate) fn dag_to_cpdag_with_known(
    dag: &Dag,
    known: impl Fn(DenseNodeId, DenseNodeId) -> bool,
) -> Result<Cpdag, DiscoveryError> {
    let mut cpdag = Cpdag::empty();
    for node in dag.nodes() {
        cpdag.add_node(*node)?;
    }
    for e in dag.edges() {
        if let Some((from, to)) = e.parent_child() {
            if !cpdag.has_edge(from, to) {
                if known(from, to) {
                    cpdag.insert_directed(from, to)?;
                } else {
                    cpdag.insert_undirected(from, to)?;
                }
            }
        }
    }
    // Orient unshielded colliders present in the DAG.
    let n = dag.node_count();
    for zi in 0..n {
        let z = DenseNodeId::from_raw(crate::indexing::dense_u32(zi));
        let parents = dag.parents(z);
        for i in 0..parents.len() {
            for j in i + 1..parents.len() {
                let p = parents[i];
                let q = parents[j];
                if !dag.has_edge(p, q) && !dag.has_edge(q, p) {
                    // p → z ← q
                    if cpdag
                        .edge_between(p, z)
                        .is_some_and(antecedent_graph::MarkedEdge::is_undirected)
                    {
                        cpdag.orient_undirected(p, z)?;
                    }
                    if cpdag
                        .edge_between(q, z)
                        .is_some_and(antecedent_graph::MarkedEdge::is_undirected)
                    {
                        cpdag.orient_undirected(q, z)?;
                    }
                }
            }
        }
    }
    let mut state = OrientationState::default();
    let rules: [&dyn OrientationRule<antecedent_graph::Cpdag>; 4] =
        [&MeekR1, &MeekR2, &MeekR3, &MeekR4];
    let _ = run_static_orientation_to_fixed_point(&mut cpdag, &rules, &mut state)?;
    Ok(cpdag)
}
