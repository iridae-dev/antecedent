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
    /// From crate version string `x.y.z`.
    #[must_use]
    pub fn from_crate_version(v: &str) -> Self {
        let mut parts = v.split('.');
        let major = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let minor = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        let patch = parts.next().and_then(|s| s.parse().ok()).unwrap_or(0);
        Self { major, minor, patch }
    }
}

/// Artifact kind tag.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum ArtifactKind {
    /// Schema + graph bundle (Phase 0).
    SchemaGraph,
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
    /// BLAKE3 checksum of the stored payload bytes.
    pub blake3: [u8; 32],
    /// Logical schema id.
    pub logical_schema: String,
}

/// Wire schema: ordered variable names and raw ids.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SchemaWire {
    /// Variable names in dense id order.
    pub variable_names: Vec<String>,
}

/// Wire DAG: directed edges by dense variable index.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct DagWire {
    /// Node count (static variables).
    pub node_count: u32,
    /// Directed edges `(from, to)`.
    pub edges: Vec<(u32, u32)>,
}
