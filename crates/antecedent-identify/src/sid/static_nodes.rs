//! Static-node variable lists shared by the transport searches.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::VariableId;
use antecedent_graph::{Admg, NodeRef};

/// Variable of every dense node, in dense order.
///
/// `Err(())` when any node is not [`NodeRef::Static`]. Callers attach their own
/// refusal text; the node walk is the same.
pub(super) fn static_variables(graph: &Admg) -> Result<Vec<VariableId>, ()> {
    let mut variables = Vec::with_capacity(graph.node_count());
    for node in graph.nodes() {
        match node {
            NodeRef::Static(variable) => variables.push(*variable),
            _ => return Err(()),
        }
    }
    Ok(variables)
}
