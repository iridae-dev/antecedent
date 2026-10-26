//! The one variable <-> dense-node mapping for identification graphs.
//!
//! Node order is a property of the graph, never assumed to equal a variable's raw id. A
//! `Static` node carries its variable; an `Unfolded` window slot is addressed by its own dense
//! position (see [`node_variable_id`]); other node kinds carry no identification variable.
//! The first node carrying a variable wins, so a lookup agrees with a front-to-back scan.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashMap;

use antecedent_core::VariableId;
use antecedent_graph::{Dag, DenseNodeId, NodeRef};

use crate::error::IdentificationError;
use crate::prepared::node_variable_id;

/// Indexed variable <-> dense-node mapping built once per graph.
#[derive(Clone, Debug, Default)]
pub(crate) struct NodeMap {
    to_dense: HashMap<VariableId, DenseNodeId>,
    to_variable: Vec<Option<VariableId>>,
}

impl NodeMap {
    /// Index `nodes`.
    pub(crate) fn new(nodes: &[NodeRef]) -> Self {
        let mut to_dense = HashMap::with_capacity(nodes.len());
        let mut to_variable = Vec::with_capacity(nodes.len());
        for index in 0..nodes.len() {
            let variable = node_variable_id(nodes, index);
            if let Some(variable) = variable {
                to_dense.entry(variable).or_insert_with(|| DenseNodeId::from_index(index));
            }
            to_variable.push(variable);
        }
        Self { to_dense, to_variable }
    }

    /// Index the nodes of `dag`.
    pub(crate) fn from_dag(dag: &Dag) -> Self {
        Self::new(dag.nodes())
    }

    /// Dense node carrying `id`.
    ///
    /// # Errors
    ///
    /// No node carries `id`.
    pub(crate) fn dense(&self, id: VariableId) -> Result<DenseNodeId, IdentificationError> {
        self.to_dense.get(&id).copied().ok_or(IdentificationError::UnknownVariable { id })
    }

    /// Variable carried by dense node `id`.
    ///
    /// # Errors
    ///
    /// The node is out of range or carries no identification variable.
    pub(crate) fn variable(&self, id: DenseNodeId) -> Result<VariableId, IdentificationError> {
        self.to_variable
            .get(id.as_usize())
            .copied()
            .flatten()
            .ok_or_else(|| IdentificationError::UnknownDenseNode { id: id.raw() })
    }
}

/// Dense node at position `index` of a graph built by `with_variables`, whose node `i` is
/// labelled with raw variable id `i`.
///
/// # Panics
///
/// `index` exceeds `u32`; a graph position always fits.
pub(crate) fn dense_at(index: usize) -> DenseNodeId {
    DenseNodeId::from_index(index)
}

/// Variable labelling position `index` of a graph built by `with_variables`.
///
/// # Panics
///
/// `index` exceeds `u32`; a graph position always fits.
pub(crate) fn variable_at(index: usize) -> VariableId {
    VariableId::from_raw(u32::try_from(index).expect("graph position fits u32"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn first_node_wins_and_unknowns_are_errors() {
        let nodes = [
            NodeRef::Static(VariableId::from_raw(7)),
            NodeRef::Static(VariableId::from_raw(3)),
            NodeRef::Static(VariableId::from_raw(7)),
        ];
        let map = NodeMap::new(&nodes);
        assert_eq!(map.dense(VariableId::from_raw(7)).unwrap(), DenseNodeId::from_raw(0));
        assert_eq!(map.dense(VariableId::from_raw(3)).unwrap(), DenseNodeId::from_raw(1));
        assert!(map.dense(VariableId::from_raw(0)).is_err());
        assert_eq!(map.variable(DenseNodeId::from_raw(1)).unwrap(), VariableId::from_raw(3));
        assert!(map.variable(DenseNodeId::from_raw(9)).is_err());
    }
}
