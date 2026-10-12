//! JSON DAG interchange mirroring [`DagWire`] .
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_graph::Dag;
use serde::{Deserialize, Serialize};

use crate::convert::{check_graph_budget, dag_from_wire, dag_to_wire};
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
        Self { node_count: wire.node_count, edges: wire.edges.clone(), variable_names: None }
    }
}

pub(crate) fn json_from_str<T: for<'de> Deserialize<'de>>(json: &str) -> Result<T, IoError> {
    serde_json::from_str(json).map_err(|error| IoError::Convert(format!("json: {error}")))
}

pub(crate) fn json_to_pretty<T: Serialize>(doc: &T) -> Result<String, IoError> {
    serde_json::to_string_pretty(doc).map_err(|error| IoError::Convert(format!("json: {error}")))
}

pub(crate) fn check_optional_names(
    names: Option<&[String]>,
    node_count: u32,
) -> Result<(), IoError> {
    if let Some(names) = names {
        if names.len() != node_count as usize {
            return Err(IoError::Convert(format!(
                "variable_names length {} must equal node_count {node_count}",
                names.len()
            )));
        }
    }
    Ok(())
}

/// Public parse entry point that keeps its name and drops node labels.
macro_rules! without_names {
    (
        $(#[$meta:meta])*
        pub fn $name:ident($input:ident: &str) -> $Graph:ty = $with_names:path
    ) => {
        $(#[$meta])*
        pub fn $name($input: &str) -> Result<$Graph, IoError> {
            $with_names($input).map(|(graph, _names)| graph)
        }
    };
}
pub(crate) use without_names;

impl DagJson {
    /// Attach schema names; errors if length mismatches `node_count`.
    ///
    /// # Errors
    ///
    /// Name count ≠ node count.
    pub fn with_schema(mut self, schema: &SchemaWire) -> Result<Self, IoError> {
        let names = schema.variable_names();
        if names.len() != self.node_count as usize {
            return Err(IoError::Convert(format!(
                "variable_names length {} must equal node_count {}",
                names.len(),
                self.node_count
            )));
        }
        self.variable_names = Some(names);
        Ok(self)
    }

    /// Convert to wire form (names are not part of [`DagWire`]).
    #[must_use]
    pub fn to_wire(&self) -> DagWire {
        DagWire { node_count: self.node_count, edges: self.edges.clone() }
    }
}

without_names! {
    /// Parse a JSON DAG document into a [`Dag`].
    ///
    /// # Errors
    ///
    /// JSON parse errors or invalid DAG structure.
    pub fn dag_from_json(json: &str) -> Dag = dag_with_names_from_json
}

/// Parse a JSON DAG document into a [`Dag`] plus its node names.
///
/// When the document has no `variable_names`, dense-index strings (`"0"`,
/// `"1"`, ...) are returned instead, since the document carries no distinct
/// name information in that case.
///
/// # Errors
///
/// JSON parse errors, invalid DAG structure, or a `variable_names` length
/// mismatch.
pub fn dag_with_names_from_json(json: &str) -> Result<(Dag, Vec<String>), IoError> {
    let doc: DagJson = json_from_str(json)?;
    check_graph_budget(doc.node_count, doc.edges.len())?;
    let names = match &doc.variable_names {
        Some(names) => {
            if names.len() != doc.node_count as usize {
                return Err(IoError::Convert("variable_names length must equal node_count".into()));
            }
            names.clone()
        }
        None => (0..doc.node_count).map(|i| i.to_string()).collect(),
    };
    let dag = dag_from_wire(&doc.to_wire())?;
    Ok((dag, names))
}

/// Serialize a [`Dag`] to JSON.
///
/// # Errors
///
/// Wire conversion or JSON encode failures.
pub fn dag_to_json(dag: &Dag, names: Option<&[String]>) -> Result<String, IoError> {
    let wire = dag_to_wire(dag)?;
    check_optional_names(names, wire.node_count)?;
    let mut doc = DagJson::from(&wire);
    doc.variable_names = names.map(<[String]>::to_vec);
    json_to_pretty(&doc)
}

/// Parse JSON into [`DagJson`] without building a [`Dag`].
///
/// # Errors
///
/// JSON parse errors.
pub fn dag_json_from_str(json: &str) -> Result<DagJson, IoError> {
    json_from_str(json)
}

#[cfg(test)]
mod tests {
    use antecedent_graph::DenseNodeId;

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
        assert_eq!(
            s,
            "{\n  \"node_count\": 2,\n  \"edges\": [\n    [\n      0,\n      1\n    ]\n  ],\n  \"variable_names\": [\n    \"x\",\n    \"y\"\n  ]\n}"
        );
        assert!(!s.contains("\"kind\""));
    }

    #[test]
    fn with_names_round_trip_preserves_labels() {
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let names = vec!["x".to_string(), "y".to_string()];
        let s = dag_to_json(&dag, Some(&names)).unwrap();
        let (back, back_names) = dag_with_names_from_json(&s).unwrap();
        assert_eq!(back.node_count(), 2);
        assert_eq!(back_names, names);
    }

    #[test]
    fn with_names_nameless_falls_back_to_dense_index() {
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let s = dag_to_json(&dag, None).unwrap();
        let (_back, names) = dag_with_names_from_json(&s).unwrap();
        assert_eq!(names, vec!["0".to_string(), "1".to_string()]);
    }

    #[test]
    fn huge_node_count_is_refused_before_allocation() {
        let err = dag_from_json(r#"{"node_count":4294967295,"edges":[]}"#).unwrap_err();
        let msg = err.to_string();
        assert!(msg.contains("node_count"), "{msg}");
    }
}
