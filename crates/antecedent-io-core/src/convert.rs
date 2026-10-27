//! Convert between domain types and wire types.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::num::NonZeroU32;
use std::sync::Arc;

use antecedent_core::{
    CausalSchema, CausalSchemaBuilder, MeasurementSpec, ScalarType, SmallRoleSet, ValueType,
    VariableId,
};
use antecedent_graph::{Admg, Cpdag, Dag, DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag};

use crate::error::IoError;
use crate::wire::{
    AdmgWire, CpdagWire, DagWire, EndpointWire, GraphCpdagWire, MarkedEdgeWire,
    MeasurementSpecWire, PagWire, ScalarTypeWire, SchemaWire, SchemaWireV01, ValueTypeWire,
    VariableSchemaWire,
};

/// Largest `node_count` accepted from an untrusted graph document.
///
/// A wire or JSON `node_count` is an allocation multiplier: the loader builds
/// one node (and, when names are omitted, one `String`) per count before it
/// has seen that many edges. `u32::MAX` would ask for billions of allocations
/// from a few dozen bytes of input.
pub const MAX_GRAPH_NODES: u32 = 100_000;

/// Largest edge list accepted from an untrusted graph document.
pub const MAX_GRAPH_EDGES: usize = 1_000_000;

/// Bounds on a graph document read from untrusted input. The defaults are
/// [`MAX_GRAPH_NODES`] and [`MAX_GRAPH_EDGES`]; the `*_with_limits` decoders take a caller's.
/// Exceeding a bound refuses the document.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GraphLimits {
    /// Largest `node_count`.
    pub max_nodes: u32,
    /// Largest edge list.
    pub max_edges: usize,
}

impl Default for GraphLimits {
    fn default() -> Self {
        Self { max_nodes: MAX_GRAPH_NODES, max_edges: MAX_GRAPH_EDGES }
    }
}

/// Refuse a graph document whose node or edge count exceeds `limits`.
pub(crate) fn check_graph_budget_within(
    node_count: u32,
    edge_count: usize,
    limits: &GraphLimits,
) -> Result<(), IoError> {
    if node_count > limits.max_nodes {
        return Err(IoError::Convert(format!(
            "node_count {node_count} exceeds the maximum of {}",
            limits.max_nodes
        )));
    }
    if edge_count > limits.max_edges {
        return Err(IoError::Convert(format!(
            "edge count {edge_count} exceeds the maximum of {}",
            limits.max_edges
        )));
    }
    Ok(())
}

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
                unit: v.unit.as_ref().map(ToString::to_string),
                category_domain: v.category_domain.map(antecedent_core::CategoryDomainId::raw),
                measurement: MeasurementSpecWire {
                    description: v.measurement.description.as_ref().map(ToString::to_string),
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
            v.category_domain.map(antecedent_core::CategoryDomainId::from_raw),
            MeasurementSpec {
                description: v
                    .measurement
                    .description
                    .as_ref()
                    .map(|d| Arc::<str>::from(d.as_str())),
                noisy: v.measurement.noisy,
            },
        )
        .map_err(|e| IoError::Convert(e.to_string()))?;
    }
    b.build().map_err(|e| IoError::Convert(e.to_string()))
}

/// Migrate format 0.1 skinny schema to full wire. Format 0.1 stored names only, so every
/// variable is [`ValueTypeWire::Unspecified`]: nothing is known about its measurement scale.
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
                value_type: ValueTypeWire::Unspecified,
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
        ValueType::Unspecified => ValueTypeWire::Unspecified,
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
        ValueTypeWire::Unspecified => ValueType::Unspecified,
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
    dag_from_wire_with_limits(wire, &GraphLimits::default())
}

/// [`dag_from_wire`] under the caller's graph bounds.
///
/// # Errors
///
/// As [`dag_from_wire`]; a bound of `limits` exceeded is refused.
pub fn dag_from_wire_with_limits(wire: &DagWire, limits: &GraphLimits) -> Result<Dag, IoError> {
    check_graph_budget_within(wire.node_count, wire.edges.len(), limits)?;
    let mut dag = Dag::with_variables(wire.node_count);
    for &(from, to) in &wire.edges {
        dag.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
            .map_err(|e| IoError::Convert(e.to_string()))?;
    }
    dag.validate().map_err(|e| IoError::Convert(e.to_string()))?;
    Ok(dag)
}

pub(crate) fn endpoint_to_wire(e: Endpoint) -> EndpointWire {
    match e {
        Endpoint::Tail => EndpointWire::Tail,
        Endpoint::Arrow => EndpointWire::Arrow,
        Endpoint::Circle => EndpointWire::Circle,
        Endpoint::Conflict => EndpointWire::Conflict,
    }
}

fn endpoint_from_wire(e: EndpointWire) -> Endpoint {
    match e {
        EndpointWire::Tail => Endpoint::Tail,
        EndpointWire::Arrow => Endpoint::Arrow,
        EndpointWire::Circle => Endpoint::Circle,
        EndpointWire::Conflict => Endpoint::Conflict,
    }
}

/// Encode a PAG to wire form.
///
/// # Errors
///
/// Non-static nodes (should not occur for [`Pag::with_variables`]).
pub fn pag_to_wire(pag: &Pag) -> Result<PagWire, IoError> {
    let node_count = u32::try_from(pag.node_count()).map_err(|_| IoError::TooLarge)?;
    let mut edges = Vec::new();
    for i in 0..pag.node_count() {
        let a = DenseNodeId::from_raw(u32::try_from(i).map_err(|_| IoError::TooLarge)?);
        for (b, at_a, at_b) in pag.neighbors(a) {
            if a.raw() < b.raw() {
                edges.push(MarkedEdgeWire {
                    a: a.raw(),
                    b: b.raw(),
                    at_a: endpoint_to_wire(at_a),
                    at_b: endpoint_to_wire(at_b),
                });
            }
        }
    }
    Ok(PagWire { node_count, edges })
}

/// Decode a PAG from wire form.
///
/// # Errors
///
/// Invalid marks / edges.
pub fn pag_from_wire(wire: &PagWire) -> Result<Pag, IoError> {
    pag_from_wire_with_limits(wire, &GraphLimits::default())
}

/// [`pag_from_wire`] under the caller's graph bounds.
///
/// # Errors
///
/// As [`pag_from_wire`]; a bound of `limits` exceeded is refused.
pub fn pag_from_wire_with_limits(wire: &PagWire, limits: &GraphLimits) -> Result<Pag, IoError> {
    check_graph_budget_within(wire.node_count, wire.edges.len(), limits)?;
    let mut pag = Pag::with_variables(wire.node_count);
    for e in &wire.edges {
        let edge = MarkedEdge {
            a: DenseNodeId::from_raw(e.a),
            b: DenseNodeId::from_raw(e.b),
            at_a: endpoint_from_wire(e.at_a),
            at_b: endpoint_from_wire(e.at_b),
            middle: MiddleMark::Empty,
        };
        pag.insert_marked(edge).map_err(|err| IoError::Convert(err.to_string()))?;
    }
    Ok(pag)
}

/// Encode a CPDAG to wire form.
///
/// # Errors
///
/// Unexpected edge marks.
pub fn cpdag_to_wire(cpdag: &Cpdag) -> Result<CpdagWire, IoError> {
    let node_count = u32::try_from(cpdag.node_count()).map_err(|_| IoError::TooLarge)?;
    let mut directed = Vec::new();
    let mut undirected = Vec::new();
    for e in cpdag.edges() {
        if let Some((from, to)) = e.parent_child() {
            directed.push((from.raw(), to.raw()));
        } else if e.is_undirected() {
            let (a, b) = if e.a.raw() <= e.b.raw() {
                (e.a.raw(), e.b.raw())
            } else {
                (e.b.raw(), e.a.raw())
            };
            undirected.push((a, b));
        } else {
            return Err(IoError::Convert(format!(
                "CPDAG wire encoding rejects non-directed/undirected edge {:?}",
                (e.at_a, e.at_b)
            )));
        }
    }
    Ok(GraphCpdagWire { node_count, directed, undirected })
}

/// Decode a CPDAG from wire form.
///
/// # Errors
///
/// Invalid edges / cycles.
pub fn cpdag_from_wire(wire: &CpdagWire) -> Result<Cpdag, IoError> {
    cpdag_from_wire_with_limits(wire, &GraphLimits::default())
}

/// [`cpdag_from_wire`] under the caller's graph bounds.
///
/// # Errors
///
/// As [`cpdag_from_wire`]; a bound of `limits` exceeded is refused.
pub fn cpdag_from_wire_with_limits(
    wire: &CpdagWire,
    limits: &GraphLimits,
) -> Result<Cpdag, IoError> {
    let edges = wire.directed.len().saturating_add(wire.undirected.len());
    check_graph_budget_within(wire.node_count, edges, limits)?;
    let mut g = Cpdag::with_variables(wire.node_count);
    for &(from, to) in &wire.directed {
        g.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
            .map_err(|e| IoError::Convert(e.to_string()))?;
    }
    for &(a, b) in &wire.undirected {
        g.insert_undirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
            .map_err(|e| IoError::Convert(e.to_string()))?;
    }
    Ok(g)
}

/// Encode an ADMG to wire form.
///
/// # Errors
///
/// Capacity overflow.
pub fn admg_to_wire(admg: &Admg) -> Result<AdmgWire, IoError> {
    let node_count = u32::try_from(admg.node_count()).map_err(|_| IoError::TooLarge)?;
    let mut directed = Vec::new();
    let mut bidirected = Vec::new();
    for i in 0..admg.node_count() {
        let from = DenseNodeId::from_raw(u32::try_from(i).map_err(|_| IoError::TooLarge)?);
        for &to in admg.children(from) {
            directed.push((from.raw(), to.raw()));
        }
        for &nbr in admg.bidirected_neighbors(from) {
            if from.raw() < nbr.raw() {
                bidirected.push((from.raw(), nbr.raw()));
            }
        }
    }
    Ok(AdmgWire { node_count, directed, bidirected })
}

/// Decode an ADMG from wire form.
///
/// # Errors
///
/// Invalid edges / cycles.
pub fn admg_from_wire(wire: &AdmgWire) -> Result<Admg, IoError> {
    admg_from_wire_with_limits(wire, &GraphLimits::default())
}

/// [`admg_from_wire`] under the caller's graph bounds.
///
/// # Errors
///
/// As [`admg_from_wire`]; a bound of `limits` exceeded is refused.
pub fn admg_from_wire_with_limits(wire: &AdmgWire, limits: &GraphLimits) -> Result<Admg, IoError> {
    let edges = wire.directed.len().saturating_add(wire.bidirected.len());
    check_graph_budget_within(wire.node_count, edges, limits)?;
    let mut g = Admg::with_variables(wire.node_count);
    for &(from, to) in &wire.directed {
        g.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to))
            .map_err(|e| IoError::Convert(e.to_string()))?;
    }
    for &(a, b) in &wire.bidirected {
        g.insert_bidirected(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b))
            .map_err(|e| IoError::Convert(e.to_string()))?;
    }
    g.validate().map_err(|e| IoError::Convert(e.to_string()))?;
    Ok(g)
}

/// CBOR-encode a value to bytes.
///
/// # Errors
///
/// CBOR failure.
pub fn to_cbor<T: serde::Serialize>(value: &T) -> Result<Vec<u8>, IoError> {
    let mut buf = Vec::new();
    ciborium::into_writer(value, &mut buf).map_err(IoError::cbor_encode)?;
    Ok(buf)
}

/// CBOR-decode bytes.
///
/// # Errors
///
/// CBOR failure.
pub fn from_cbor<T: serde::de::DeserializeOwned>(bytes: &[u8]) -> Result<T, IoError> {
    let mut reader = bytes;
    let value = ciborium::from_reader(&mut reader).map_err(IoError::cbor_decode)?;
    // A section is one CBOR item. Bytes after it would ride under the section checksum
    // while no consumer reads them.
    if reader.is_empty() {
        Ok(value)
    } else {
        Err(IoError::Cbor(crate::error::CborFailure::new(
            crate::error::CborFault::TrailingBytes,
            format!("{} trailing bytes after the CBOR item", reader.len()),
        )))
    }
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

/// A unit enum with `rename_all = "snake_case"` encodes as the same CBOR text
/// as the historical string, and that text decodes back to the variant.
#[cfg(test)]
pub(crate) fn assert_cbor_text_code<T>(text: &str, value: &T)
where
    T: serde::Serialize + serde::de::DeserializeOwned + PartialEq + std::fmt::Debug,
{
    #[derive(serde::Serialize)]
    struct AsText<'a> {
        code: &'a str,
    }
    #[derive(serde::Serialize)]
    struct AsEnum<'a, T> {
        code: &'a T,
    }
    #[derive(serde::Deserialize, Debug)]
    struct Owned<T> {
        code: T,
    }
    let text_bytes = to_cbor(&AsText { code: text }).expect("text cbor");
    let enum_bytes = to_cbor(&AsEnum { code: value }).expect("enum cbor");
    assert_eq!(text_bytes, enum_bytes, "{text}");
    let back: Owned<T> = from_cbor(&text_bytes)
        .unwrap_or_else(|error| panic!("historical text `{text}` did not decode: {error}"));
    assert_eq!(&back.code, value);
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn from_cbor_rejects_bytes_after_the_first_item() {
        let mut bytes = to_cbor(&vec![1u32, 2, 3]).unwrap();
        assert_eq!(from_cbor::<Vec<u32>>(&bytes).unwrap(), vec![1, 2, 3]);
        bytes.extend_from_slice(&to_cbor(&99u32).unwrap());
        let error = from_cbor::<Vec<u32>>(&bytes).unwrap_err().to_string();
        assert!(error.contains("trailing bytes"), "{error}");
    }
}
