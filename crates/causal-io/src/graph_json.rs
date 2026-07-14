//! JSON DAG interchange mirroring [`DagWire`] (Phase 12).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_graph::Dag;
use serde::{Deserialize, Serialize};

use crate::convert::{dag_from_wire, dag_to_wire};
use crate::error::IoError;
use crate::wire::{DagWire, SchemaWire};

/// JSON document for a static DAG, optionally with variable names.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DagJson {
    /// Dense node count.
    pub node_count: u32,
    /// Directed edges `(from, to)`.
    pub edges: Vec<(u32, u32)>,
    /// Optional variable names in dense id order.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub variable_names: Option<Vec<String>>,
}

impl From<&DagWire> for DagJson {
    fn from(wire: &DagWire) -> Self {
        Self {
            node_count: wire.node_count,
            edges: wire.edges.clone(),
            variable_names: None,
        }
    }
}

impl DagJson {
    /// Attach schema names when lengths match.
    #[must_use]
    pub fn with_schema(mut self, schema: &SchemaWire) -> Self {
        if schema.variable_names.len() == self.node_count as usize {
            self.variable_names = Some(schema.variable_names.clone());
        }
        self
    }

    /// Convert to wire form (names are not part of [`DagWire`]).
    #[must_use]
    pub fn to_wire(&self) -> DagWire {
        DagWire { node_count: self.node_count, edges: self.edges.clone() }
    }
}

/// Parse a JSON DAG document into a [`Dag`].
///
/// # Errors
///
/// JSON parse errors or invalid DAG structure.
pub fn dag_from_json(json: &str) -> Result<Dag, IoError> {
    let doc: DagJson =
        serde_json::from_str(json).map_err(|e| IoError::Convert(format!("json: {e}")))?;
    if let Some(names) = &doc.variable_names {
        if names.len() != doc.node_count as usize {
            return Err(IoError::Convert(
                "variable_names length must equal node_count".into(),
            ));
        }
    }
    dag_from_wire(&doc.to_wire())
}

/// Serialize a [`Dag`] to JSON.
///
/// # Errors
///
/// Wire conversion or JSON encode failures.
pub fn dag_to_json(dag: &Dag, names: Option<&[String]>) -> Result<String, IoError> {
    let wire = dag_to_wire(dag)?;
    let mut doc = DagJson::from(&wire);
    if let Some(n) = names {
        if n.len() == wire.node_count as usize {
            doc.variable_names = Some(n.to_vec());
        }
    }
    serde_json::to_string_pretty(&doc).map_err(|e| IoError::Convert(format!("json: {e}")))
}

/// Parse JSON into [`DagJson`] without building a [`Dag`].
///
/// # Errors
///
/// JSON parse errors.
pub fn dag_json_from_str(json: &str) -> Result<DagJson, IoError> {
    serde_json::from_str(json).map_err(|e| IoError::Convert(format!("json: {e}")))
}

#[cfg(test)]
mod tests {
    use causal_graph::DenseNodeId;

    use super::*;

    #[test]
    fn json_round_trip_with_names() {
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let names = vec!["x".into(), "y".into()];
        let s = dag_to_json(&dag, Some(&names)).unwrap();
        let back = dag_from_json(&s).unwrap();
        assert_eq!(back.node_count(), 2);
        assert!(back.reaches(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)));
        let doc = dag_json_from_str(&s).unwrap();
        assert_eq!(doc.variable_names.as_deref(), Some(names.as_slice()));
    }
}
