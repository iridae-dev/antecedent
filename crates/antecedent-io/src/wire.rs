//! Wire types for durable artifacts (not internal Rust structs).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use serde::{Deserialize, Serialize};

/// Format version pair.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct FormatVersion {
    /// Major.
    pub major: u16,
    /// Minor.
    pub minor: u16,
}

/// Semantic library version.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SemanticVersion {
    /// Major.
    pub major: u32,
    /// Minor.
    pub minor: u32,
    /// Patch.
    pub patch: u32,
}

impl SemanticVersion {
    /// Parse crate version string `x.y.z`.
    ///
    /// # Errors
    ///
    /// Non-semver `major.minor.patch` with three unsigned integers.
    pub fn from_crate_version(v: &str) -> Result<Self, crate::error::IoError> {
        let mut parts = v.split('.');
        let parse = |s: Option<&str>| -> Result<u32, crate::error::IoError> {
            s.and_then(|p| p.parse().ok()).ok_or_else(|| {
                crate::error::IoError::Convert(format!(
                    "invalid semantic version {v:?}; expected major.minor.patch"
                ))
            })
        };
        let major = parse(parts.next())?;
        let minor = parse(parts.next())?;
        let patch = parse(parts.next())?;
        if parts.next().is_some() {
            return Err(crate::error::IoError::Convert(format!(
                "invalid semantic version {v:?}; expected major.minor.patch"
            )));
        }
        Ok(Self { major, minor, patch })
    }
}

/// Artifact kind tag.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    /// Schema + graph bundle.
    SchemaGraph,
    /// Identification/estimation analysis trace.
    AnalysisTrace,
    /// Columnar causal posterior.
    CausalPosterior,
    /// Fitted / compiled model plus optional analysis sections.
    ModelBundle,
    /// Other / future.
    Other(String),
}

/// Provenance summary on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ProvenanceWire {
    /// Free-form notes / operation id.
    pub note: String,
}

/// Section table entry.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SectionDescriptor {
    /// Stable section id.
    pub id: String,
    /// Content type.
    pub content_type: String,
    /// Encoding version.
    pub encoding_version: u16,
    /// Required for readers.
    pub required: bool,
    /// Optional compression algorithm name.
    pub compression: Option<String>,
    /// Compressed size.
    pub compressed_size: u64,
    /// Uncompressed size.
    pub uncompressed_size: u64,
    /// BLAKE3 checksum of the stored (on-wire) payload bytes.
    pub blake3: [u8; 32],
    /// Logical schema id.
    pub logical_schema: String,
}

/// Format 0.1 skinny schema wire (names only) — used by migration.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SchemaWireV01 {
    /// Variable names in dense id order.
    pub variable_names: Vec<String>,
}

/// Scalar element type on the wire.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ScalarTypeWire {
    /// f64.
    Float64,
    /// f32.
    Float32,
    /// i64.
    Int64,
    /// i32.
    Int32,
}

/// Value type on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ValueTypeWire {
    /// Continuous.
    Continuous,
    /// Count.
    Count,
    /// Binary.
    Binary,
    /// Categorical.
    Categorical,
    /// Ordinal.
    Ordinal,
    /// Fixed-width vector.
    Vector {
        /// Width.
        width: u32,
        /// Element type.
        element: ScalarTypeWire,
    },
}

/// Measurement metadata on the wire.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MeasurementSpecWire {
    /// Optional description.
    pub description: Option<String>,
    /// Noisy measurement flag.
    pub noisy: bool,
}

/// One variable on the wire (format ≥ 0.2).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct VariableSchemaWire {
    /// Dense variable id.
    pub id: u32,
    /// Name.
    pub name: String,
    /// Value type.
    pub value_type: ValueTypeWire,
    /// Role-hint bit mask ([`antecedent_core::SmallRoleSet::bits`]).
    pub role_bits: u16,
    /// Optional unit.
    pub unit: Option<String>,
    /// Optional category domain id.
    pub category_domain: Option<u32>,
    /// Measurement.
    pub measurement: MeasurementSpecWire,
}

/// Full schema wire (format ≥ 0.2).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SchemaWire {
    /// Variables in dense id order.
    pub variables: Vec<VariableSchemaWire>,
}

impl SchemaWire {
    /// Skinny name list (API convenience / migration helper).
    #[must_use]
    pub fn variable_names(&self) -> Vec<String> {
        self.variables.iter().map(|v| v.name.clone()).collect()
    }
}

/// Wire DAG: directed edges by dense variable index.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DagWire {
    /// Node count (static variables).
    pub node_count: u32,
    /// Directed edges `(from, to)`.
    pub edges: Vec<(u32, u32)>,
}

/// Endpoint mark on the wire (`tail` / `arrow` / `circle` / `conflict`).
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum EndpointWire {
    /// Tail.
    Tail,
    /// Arrow head.
    Arrow,
    /// Circle (PAG).
    Circle,
    /// Conflict.
    Conflict,
}

/// Marked edge on the wire (PAG / shared marked interchange).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct MarkedEdgeWire {
    /// Endpoint A dense index.
    pub a: u32,
    /// Endpoint B dense index.
    pub b: u32,
    /// Mark at A.
    pub at_a: EndpointWire,
    /// Mark at B.
    pub at_b: EndpointWire,
}

/// Wire PAG: marked edges by dense variable index.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct PagWire {
    /// Node count (static variables).
    pub node_count: u32,
    /// Marked edges (each undirected pair once).
    pub edges: Vec<MarkedEdgeWire>,
}

/// Wire CPDAG: directed + undirected edges.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CpdagWire {
    /// Node count (static variables).
    pub node_count: u32,
    /// Directed edges `(from, to)`.
    pub directed: Vec<(u32, u32)>,
    /// Undirected edges `(a, b)` with `a < b` preferred.
    pub undirected: Vec<(u32, u32)>,
}

/// Wire ADMG: directed + bidirected edges.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AdmgWire {
    /// Node count (static variables).
    pub node_count: u32,
    /// Directed edges `(from, to)`.
    pub directed: Vec<(u32, u32)>,
    /// Bidirected edges `(a, b)` with `a < b` preferred.
    pub bidirected: Vec<(u32, u32)>,
}
