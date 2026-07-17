//! Convert between domain types and wire types.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::num::NonZeroU32;
use std::sync::Arc;

use causal_core::{
    CausalSchema, CausalSchemaBuilder, MeasurementSpec, ScalarType, SmallRoleSet, ValueType,
    VariableId,
};
use causal_graph::{Dag, DenseNodeId};

use crate::error::IoError;
use crate::wire::{
    DagWire, MeasurementSpecWire, ScalarTypeWire, SchemaWire, SchemaWireV01, ValueTypeWire,
    VariableSchemaWire,
};

/// Encode a schema to full wire form (format ≥ 0.2).
#[must_use]
pub fn schema_to_wire(schema: &CausalSchema) -> SchemaWire {
    SchemaWire {
        variables: schema
            .variables()
            .iter()
            .map(|v| VariableSchemaWire {
                id: v.id.raw(),
                name: v.name.to_string(),
                value_type: value_type_to_wire(&v.value_type),
                role_bits: v.role_hints.bits(),
                unit: v.unit.as_ref().map(|u| u.to_string()),
                category_domain: v.category_domain.map(|d| d.raw()),
                measurement: MeasurementSpecWire {
                    description: v.measurement.description.as_ref().map(|d| d.to_string()),
                    noisy: v.measurement.noisy,
                },
            })
            .collect(),
    }
}

/// Decode full schema wire.
///
/// # Errors
///
/// Invalid value types or schema builder failures.
pub fn schema_from_wire(wire: &SchemaWire) -> Result<CausalSchema, IoError> {
    let mut b = CausalSchemaBuilder::new();
    for (i, v) in wire.variables.iter().enumerate() {
        if v.id as usize != i {
            return Err(IoError::Convert(format!(
                "schema wire id {} must equal dense index {i}",
                v.id
            )));
        }
        b.add_variable(
            Arc::<str>::from(v.name.as_str()),
            value_type_from_wire(&v.value_type)?,
            SmallRoleSet::from_bits_truncate(v.role_bits),
            v.unit.as_ref().map(|u| Arc::<str>::from(u.as_str())),
            v.category_domain.map(causal_core::CategoryDomainId::from_raw),
            MeasurementSpec {
                description: v.measurement.description.as_ref().map(|d| Arc::<str>::from(d.as_str())),
                noisy: v.measurement.noisy,
            },
        )
        .map_err(|e| IoError::Convert(e.to_string()))?;
    }
    b.build().map_err(|e| IoError::Convert(e.to_string()))
}

/// Migrate format 0.1 skinny schema to full wire with Continuous defaults.
#[must_use]
pub fn schema_wire_from_v01(v01: &SchemaWireV01) -> SchemaWire {
    SchemaWire {
        variables: v01
            .variable_names
            .iter()
            .enumerate()
            .map(|(i, name)| VariableSchemaWire {
                id: u32::try_from(i).unwrap_or(u32::MAX),
                name: name.clone(),
                value_type: ValueTypeWire::Continuous,
                role_bits: 0,
                unit: None,
                category_domain: None,
                measurement: MeasurementSpecWire { description: None, noisy: false },
            })
            .collect(),
    }
}

/// Try decode schema section bytes as 0.2 then 0.1.
///
/// # Errors
///
/// CBOR failure for both layouts.
pub fn schema_wire_from_cbor_bytes(bytes: &[u8]) -> Result<SchemaWire, IoError> {
    if let Ok(w) = crate::convert::from_cbor::<SchemaWire>(bytes) {
        if !w.variables.is_empty() {
            return Ok(w);
        }
    }
    let v01: SchemaWireV01 = crate::convert::from_cbor(bytes)?;
    Ok(schema_wire_from_v01(&v01))
}

fn value_type_to_wire(v: &ValueType) -> ValueTypeWire {
    match v {
        ValueType::Continuous => ValueTypeWire::Continuous,
        ValueType::Count => ValueTypeWire::Count,
        ValueType::Binary => ValueTypeWire::Binary,
        ValueType::Categorical => ValueTypeWire::Categorical,
        ValueType::Ordinal => ValueTypeWire::Ordinal,
        ValueType::Vector { width, element } => ValueTypeWire::Vector {
            width: width.get(),
            element: match element {
                ScalarType::Float64 => ScalarTypeWire::Float64,
                ScalarType::Float32 => ScalarTypeWire::Float32,
                ScalarType::Int64 => ScalarTypeWire::Int64,
                ScalarType::Int32 => ScalarTypeWire::Int32,
            },
        },
    }
}

fn value_type_from_wire(v: &ValueTypeWire) -> Result<ValueType, IoError> {
    Ok(match v {
        ValueTypeWire::Continuous => ValueType::Continuous,
        ValueTypeWire::Count => ValueType::Count,
        ValueTypeWire::Binary => ValueType::Binary,
        ValueTypeWire::Categorical => ValueType::Categorical,
        ValueTypeWire::Ordinal => ValueType::Ordinal,
        ValueTypeWire::Vector { width, element } => {
            let width = NonZeroU32::new(*width)
                .ok_or_else(|| IoError::Convert("vector width must be non-zero".into()))?;
            let element = match element {
                ScalarTypeWire::Float64 => ScalarType::Float64,
                ScalarTypeWire::Float32 => ScalarType::Float32,
                ScalarTypeWire::Int64 => ScalarType::Int64,
                ScalarTypeWire::Int32 => ScalarType::Int32,
            };
            ValueType::Vector { width, element }
        }
    })
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

/// Helper: dense variable id list.
#[must_use]
pub fn vars_to_raw(vars: &[VariableId]) -> Vec<u32> {
    vars.iter().map(|v| v.raw()).collect()
}

/// Helper: dense variable ids from raw.
#[must_use]
pub fn vars_from_raw(raw: &[u32]) -> Arc<[VariableId]> {
    raw.iter().copied().map(VariableId::from_raw).collect::<Vec<_>>().into()
}
