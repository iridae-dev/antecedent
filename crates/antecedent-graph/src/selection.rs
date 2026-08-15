//! Selection diagrams for structural transportability.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;
use std::sync::Arc;

use antecedent_core::VariableId;

use crate::{Admg, GraphError};

/// A source/target selection diagram.
///
/// `selection_targets` are causal variables whose generating mechanisms may differ between the
/// source and target populations. Selection nodes are represented extensionally by their target;
/// they are not ordinary observed variables and cannot accidentally enter an adjustment set.
#[derive(Clone, Debug)]
pub struct SelectionDiagram {
    causal_graph: Admg,
    selection_targets: Arc<[VariableId]>,
}

impl SelectionDiagram {
    /// Build and validate a selection diagram over a semi-Markovian causal graph.
    pub fn try_new(
        causal_graph: Admg,
        selection_targets: impl Into<Arc<[VariableId]>>,
    ) -> Result<Self, GraphError> {
        let selection_targets = selection_targets.into();
        let mut seen = BTreeSet::new();
        for target in selection_targets.iter().copied() {
            if target.as_usize() >= causal_graph.node_count() {
                return Err(GraphError::UnknownNode { id: target.raw() });
            }
            if !seen.insert(target.raw()) {
                return Err(GraphError::InvalidSelectionDiagram {
                    message: "selection targets must be unique".into(),
                });
            }
        }
        Ok(Self { causal_graph, selection_targets })
    }

    /// Borrow the causal ADMG shared by source and target populations.
    #[must_use]
    pub const fn causal_graph(&self) -> &Admg {
        &self.causal_graph
    }

    /// Variables whose mechanisms are allowed to differ between populations.
    #[must_use]
    pub fn selection_targets(&self) -> &[VariableId] {
        &self.selection_targets
    }

    /// Whether a variable's mechanism is marked as population-specific.
    #[must_use]
    pub fn mechanism_may_differ(&self, variable: VariableId) -> bool {
        self.selection_targets.contains(&variable)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_targets_are_not_graph_nodes() {
        let graph = Admg::with_variables(3);
        let diagram = SelectionDiagram::try_new(graph, [VariableId::from_raw(1)]).unwrap();
        assert_eq!(diagram.causal_graph().node_count(), 3);
        assert!(diagram.mechanism_may_differ(VariableId::from_raw(1)));
    }
}
