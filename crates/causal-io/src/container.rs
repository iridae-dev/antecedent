//! Versioned sectioned artifact container (DESIGN.md §24, ADR 0002).
//!
//! Layout:
//! ```text
//! magic (8) | container_version (u32 LE) | manifest_len (u32 LE)
//! | canonical CBOR manifest | section payloads...
//! ```
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::{Read, Write};

use serde::{Deserialize, Serialize};

use crate::error::IoError;
use crate::wire::{
    ArtifactKind, FormatVersion, ProvenanceWire, SectionDescriptor, SemanticVersion,
};

/// Magic bytes: `CAUSAL\0\0`.
pub const MAGIC: &[u8; 8] = b"CAUSAL\0\0";

/// Current container format version.
pub const CONTAINER_VERSION: u32 = 1;

/// Artifact manifest (canonical CBOR).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ArtifactManifest {
    /// Format version of this artifact.
    pub format_version: FormatVersion,
    /// Minimum reader version.
    pub minimum_reader_version: FormatVersion,
    /// Artifact kind.
    pub artifact_kind: ArtifactKind,
    /// Producing library version.
    pub library_version: SemanticVersion,
    /// Artifact id.
    pub artifact_id: String,
    /// Section table.
    pub sections: Vec<SectionDescriptor>,
    /// Provenance summary.
    pub provenance: ProvenanceWire,
}

/// One section's payload bytes with checksum verification on read.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct SectionBytes {
    /// Section id matching the manifest.
    pub id: String,
    /// Raw payload.
    pub data: Vec<u8>,
}

/// Encoded artifact ready for storage/transport.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct EncodedArtifact {
    /// Manifest.
    pub manifest: ArtifactManifest,
    /// Section payloads in manifest order.
    pub sections: Vec<SectionBytes>,
}

impl EncodedArtifact {
    /// Serialize to the sectioned container format.
    ///
    /// # Errors
    ///
    /// CBOR or IO failures; section/manifest inconsistency.
    pub fn write_to<W: Write>(&self, mut w: W) -> Result<(), IoError> {
        if self.manifest.sections.len() != self.sections.len() {
            return Err(IoError::ManifestMismatch {
                message: "section count != manifest section count",
            });
        }
        for (desc, sec) in self.manifest.sections.iter().zip(self.sections.iter()) {
            if desc.id != sec.id {
                return Err(IoError::ManifestMismatch { message: "section id mismatch" });
            }
            let hash = blake3::hash(&sec.data);
            if hash.as_bytes() != &desc.blake3 {
                return Err(IoError::ChecksumMismatch { section: sec.id.clone() });
            }
        }

        w.write_all(MAGIC)?;
        w.write_all(&CONTAINER_VERSION.to_le_bytes())?;
        let mut manifest_buf = Vec::new();
        ciborium::into_writer(&self.manifest, &mut manifest_buf)
            .map_err(|e| IoError::Cbor(e.to_string()))?;
        let manifest_len = u32::try_from(manifest_buf.len()).map_err(|_| IoError::TooLarge)?;
        w.write_all(&manifest_len.to_le_bytes())?;
        w.write_all(&manifest_buf)?;
        for sec in &self.sections {
            let len = u32::try_from(sec.data.len()).map_err(|_| IoError::TooLarge)?;
            w.write_all(&len.to_le_bytes())?;
            w.write_all(&sec.data)?;
        }
        Ok(())
    }

    /// Decode a container.
    ///
    /// # Errors
    ///
    /// Bad magic, version, CBOR, length, or checksum.
    pub fn read_from<R: Read>(mut r: R) -> Result<Self, IoError> {
        let mut magic = [0u8; 8];
        r.read_exact(&mut magic)?;
        if &magic != MAGIC {
            return Err(IoError::BadMagic);
        }
        let mut ver_buf = [0u8; 4];
        r.read_exact(&mut ver_buf)?;
        let version = u32::from_le_bytes(ver_buf);
        if version != CONTAINER_VERSION {
            return Err(IoError::UnsupportedVersion { version });
        }
        r.read_exact(&mut ver_buf)?;
        let manifest_len =
            usize::try_from(u32::from_le_bytes(ver_buf)).map_err(|_| IoError::TooLarge)?;
        let mut manifest_buf = vec![0u8; manifest_len];
        r.read_exact(&mut manifest_buf)?;
        let manifest: ArtifactManifest = ciborium::from_reader(manifest_buf.as_slice())
            .map_err(|e| IoError::Cbor(e.to_string()))?;

        let mut sections = Vec::with_capacity(manifest.sections.len());
        for desc in &manifest.sections {
            r.read_exact(&mut ver_buf)?;
            let len =
                usize::try_from(u32::from_le_bytes(ver_buf)).map_err(|_| IoError::TooLarge)?;
            let mut data = vec![0u8; len];
            r.read_exact(&mut data)?;
            let hash = blake3::hash(&data);
            if hash.as_bytes() != &desc.blake3 {
                return Err(IoError::ChecksumMismatch { section: desc.id.clone() });
            }
            let expected =
                usize::try_from(desc.uncompressed_size).map_err(|_| IoError::TooLarge)?;
            if data.len() != expected {
                return Err(IoError::ManifestMismatch { message: "section size mismatch" });
            }
            sections.push(SectionBytes { id: desc.id.clone(), data });
        }
        Ok(Self { manifest, sections })
    }
}

/// Build a section descriptor for uncompressed CBOR payload.
#[must_use]
pub fn section_descriptor(
    id: impl Into<String>,
    content_type: impl Into<String>,
    data: &[u8],
) -> SectionDescriptor {
    let hash = blake3::hash(data);
    let mut blake3_bytes = [0u8; 32];
    blake3_bytes.copy_from_slice(hash.as_bytes());
    SectionDescriptor {
        id: id.into(),
        content_type: content_type.into(),
        encoding_version: 1,
        required: true,
        compression: None,
        compressed_size: u64::try_from(data.len()).unwrap_or(u64::MAX),
        uncompressed_size: u64::try_from(data.len()).unwrap_or(u64::MAX),
        blake3: blake3_bytes,
        logical_schema: "cbor.v1".into(),
    }
}
