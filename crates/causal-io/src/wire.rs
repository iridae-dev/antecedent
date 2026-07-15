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
    /// Schema + graph bundle .
    SchemaGraph,
    /// Identification/estimation analysis trace .
    AnalysisTrace,
    /// Columnar causal posterior .
    CausalPosterior,
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
