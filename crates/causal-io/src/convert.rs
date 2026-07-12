//! Convert between domain types and wire types.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::CausalSchema;
use causal_graph::{Dag, DenseNodeId};

use crate::error::IoError;
use crate::wire::{DagWire, SchemaWire};

/// Encode a schema to wire form.
#[must_use]
pub fn schema_to_wire(schema: &CausalSchema) -> SchemaWire {
    SchemaWire { variable_names: schema.variables().iter().map(|v| v.name.to_string()).collect() }
}

/// Encode a DAG to wire form (static variable nodes only).
///
/// # Errors
///
/// Non-static nodes.
pub fn dag_to_wire(dag: &Dag) -> Result<DagWire, IoError> {
    let node_count = u32::try_from(dag.node_count()).map_err(|_| IoError::TooLarge)?;
    let mut edges = Vec::new();
    for e in dag.edges() {
        let (from, to) = e
            .parent_child()
            .ok_or_else(|| IoError::Convert("non-directed edge in DAG wire encoding".into()))?;
        edges.push((from.raw(), to.raw()));
    }
    Ok(DagWire { node_count, edges })
}

/// Decode a DAG from wire form.
///
/// # Errors
///
/// Invalid edges / cycles.
pub fn dag_from_wire(wire: &DagWire) -> Result<Dag, IoError> {
    let mut dag = Dag::with_variables(wire.node_count);
    for &(from, to) in &wire.edges {
        dag.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
            .map_err(|e| IoError::Convert(e.to_string()))?;
    }
    dag.validate().map_err(|e| IoError::Convert(e.to_string()))?;
    Ok(dag)
}

/// CBOR-encode a value to bytes.
///
/// # Errors
///
/// CBOR failure.
pub fn to_cbor<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, IoError> {
    let mut buf = Vec::new();
    ciborium::into_writer(value, &mut buf).map_err(|e| IoError::Cbor(e.to_string()))?;
    Ok(buf)
}

/// CBOR-decode bytes.
///
/// # Errors
///
/// CBOR failure.
pub fn from_cbor<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, IoError> {
    ciborium::from_reader(bytes).map_err(|e| IoError::Cbor(e.to_string()))
}
