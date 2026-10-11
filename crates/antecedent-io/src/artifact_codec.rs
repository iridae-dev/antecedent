//! Shared sectioned-artifact codec.
//!
//! One container writer and reader for the fixed one-, two- and three-section
//! artifacts, and one CBOR version gate. Callers keep their own error type and
//! map [`CodecError`] into it, so the on-wire bytes and the refusal text stay
//! those of the artifact that owns the payload.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::Cursor;

use serde::de::DeserializeOwned;
use serde::Deserialize;

use crate::container::{ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor};
use crate::error::IoError;
use crate::reader::ArtifactReader;
use crate::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};

/// CBOR section content type.
pub const CONTENT_CBOR: &str = "application/cbor";
/// Opaque section content type.
pub const CONTENT_BYTES: &str = "application/octet-stream";

/// Manifest fields shared by every fixed-layout artifact container.
pub struct ArtifactContainer<'a> {
    /// `ArtifactKind::Other` label.
    pub kind: &'a str,
    /// Provenance note stored in the manifest.
    pub note: &'a str,
    /// Largest accepted file, and the largest sum of declared section sizes.
    pub max_bytes: usize,
    /// Manifest artifact id. Blank ids are refused.
    pub artifact_id: &'a str,
    /// Producing crate's `CARGO_PKG_VERSION`.
    pub library_version: &'a str,
}

/// One section to write, in manifest order.
pub struct OutSection<'a> {
    /// Section id.
    pub id: &'a str,
    /// Content type stored on the descriptor (`application/cbor` or
    /// `application/octet-stream`).
    pub content_type: &'a str,
    /// Logical section bytes.
    pub bytes: &'a [u8],
}

/// Failure of the shared container codec, before an artifact maps it onto its
/// own refusal.
#[derive(Debug)]
pub enum CodecError {
    /// `artifact_id` is empty or only whitespace.
    MissingArtifactId,
    /// CBOR, manifest or write failure while encoding.
    Encode(IoError),
    /// The bytes are not a readable container.
    Decode(IoError),
    /// The file itself is over the artifact's byte bound.
    FileTooLarge,
    /// The declared section sizes sum to more than the artifact's byte bound.
    SectionsTooLarge,
    /// Kind or section ids are not this artifact's layout.
    Layout,
}

impl CodecError {
    /// Map onto [`IoError`] using `layout` as the convert message for a layout
    /// mismatch. File and section bounds are both [`IoError::TooLarge`].
    pub fn into_io(self, layout: &'static str) -> IoError {
        match self {
            Self::MissingArtifactId => IoError::Convert("missing artifact id".into()),
            Self::Encode(error) | Self::Decode(error) => error,
            Self::FileTooLarge | Self::SectionsTooLarge => IoError::TooLarge,
            Self::Layout => IoError::Convert(layout.into()),
        }
    }

    /// Map onto the family of checksummed artifacts whose container refusals
    /// share one wording: a blank id or an encode failure is `Encode`, a short
    /// or corrupt container is `Decode`, either size bound is
    /// `LimitsExceeded("artifact bytes")`, and a foreign layout is
    /// `Malformed("unsupported container layout")`.
    pub fn into_check(self) -> ArtifactCheckError {
        match self {
            Self::MissingArtifactId => ArtifactCheckError::Encode("missing artifact id".into()),
            Self::Encode(error) => ArtifactCheckError::Encode(error.to_string()),
            Self::Decode(error) => ArtifactCheckError::Decode(error.to_string()),
            Self::FileTooLarge | Self::SectionsTooLarge => {
                ArtifactCheckError::LimitsExceeded("artifact bytes")
            }
            Self::Layout => ArtifactCheckError::Malformed("unsupported container layout".into()),
        }
    }
}

/// Container-level check shared by the sectioned artifacts.
///
/// Artifact-specific enums keep their own `Display` text and any extra
/// variants (a replay mismatch, a core refusal). They convert this value for
/// the checks the container itself can make.
#[derive(Clone, Debug, PartialEq, Eq, thiserror::Error)]
#[non_exhaustive]
pub enum ArtifactCheckError {
    /// The bytes do not decode as a container.
    #[error("{0}")]
    Decode(String),
    /// The container could not be encoded.
    #[error("{0}")]
    Encode(String),
    /// A file or section bound was exceeded.
    #[error("consumer limit exceeded: {0}")]
    LimitsExceeded(&'static str),
    /// The layout or a stored field is not well formed.
    #[error("malformed artifact: {0}")]
    Malformed(String),
    /// Another major version, refused before the payload is interpreted.
    #[error("unsupported artifact version {version}")]
    UnsupportedVersion {
        /// Stored version.
        version: u32,
    },
    /// A feature marker or claim that is not this format's.
    #[error("unsupported semantics: {0}")]
    UnsupportedSemantics(&'static str),
}

/// Integer width of a CBOR `version` field. Only `u16` and `u32` are accepted,
/// which is what the artifact readers peek.
pub trait VersionWord: Copy + Eq + DeserializeOwned + private::Sealed {}

impl VersionWord for u16 {}
impl VersionWord for u32 {}

mod private {
    pub trait Sealed {}
    impl Sealed for u16 {}
    impl Sealed for u32 {}
}

/// Write `sections` in order as one checksummed container.
///
/// # Errors
/// A blank id, an encoding failure, or a file over `max_bytes`.
pub fn encode_container(
    header: &ArtifactContainer<'_>,
    sections: &[OutSection<'_>],
) -> Result<Vec<u8>, CodecError> {
    if header.artifact_id.trim().is_empty() {
        return Err(CodecError::MissingArtifactId);
    }
    let encoded = EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: crate::migrate::STABLE_FORMAT,
            minimum_reader_version: crate::migrate::STABLE_FORMAT,
            artifact_kind: ArtifactKind::Other(header.kind.into()),
            library_version: SemanticVersion::from_crate_version(header.library_version)
                .map_err(CodecError::Encode)?,
            artifact_id: header.artifact_id.into(),
            sections: sections
                .iter()
                .map(|section| section_descriptor(section.id, section.content_type, section.bytes))
                .collect(),
            provenance: ProvenanceWire { note: header.note.into() },
        },
        sections: sections
            .iter()
            .map(|section| SectionBytes::new(section.id, section.bytes.to_vec()))
            .collect(),
    };
    let mut bytes = Vec::new();
    encoded.write_to(&mut bytes).map_err(CodecError::Encode)?;
    if bytes.len() > header.max_bytes {
        return Err(CodecError::FileTooLarge);
    }
    Ok(bytes)
}

/// Read a container whose kind and section ids match `section_ids` in order.
///
/// The file length is refused before the reader allocates. A version check, when
/// the artifact has one, is the caller's [`decode_with_version`] on the returned
/// section bytes, so a foreign version is still refused before that section is
/// interpreted as its payload type.
///
/// # Errors
/// Oversize, truncated, corrupt, or differently laid out containers.
pub fn decode_container(
    kind: &str,
    max_bytes: usize,
    section_ids: &[&str],
    bytes: &[u8],
) -> Result<Vec<Vec<u8>>, CodecError> {
    if bytes.len() > max_bytes {
        return Err(CodecError::FileTooLarge);
    }
    let mut reader = ArtifactReader::open_seek(Cursor::new(bytes)).map_err(CodecError::Decode)?;
    let manifest = reader.manifest();
    let layout_ok = manifest.artifact_kind == ArtifactKind::Other(kind.into())
        && manifest.sections.len() == section_ids.len()
        && manifest.sections.iter().zip(section_ids).all(|(section, id)| section.id == *id);
    if !layout_ok {
        return Err(CodecError::Layout);
    }
    let declared = manifest
        .sections
        .iter()
        .try_fold(0_u64, |total, section| total.checked_add(section.uncompressed_size));
    let bound = u64::try_from(max_bytes).unwrap_or(u64::MAX);
    if declared.is_none_or(|total| total > bound) {
        return Err(CodecError::SectionsTooLarge);
    }
    let mut payloads = Vec::with_capacity(section_ids.len());
    for id in section_ids {
        let section = reader.load_section(id).map_err(CodecError::Decode)?;
        payloads.push(section.as_bytes().to_vec());
    }
    Ok(payloads)
}

/// Read the `version` field of one CBOR item.
///
/// The rest of the item is not interpreted as a payload type. Trailing bytes
/// after the item are refused, matching [`crate::from_cbor`].
///
/// # Errors
/// The bytes are not one CBOR item, or they have no integer `version` of type `V`.
pub fn peek_version<V: VersionWord>(bytes: &[u8]) -> Result<V, IoError> {
    version_of(&cbor_value(bytes)?)
}

/// Parse one CBOR item, run `gate` on its `version`, and decode `T` from that
/// same value only when the gate accepts it.
///
/// A rejected version never deserializes `T`, so a future layout is refused as
/// an unsupported version rather than as a field error.
///
/// # Errors
/// `gate`'s error, or a CBOR failure (including a `version` that does not fit `V`).
pub fn decode_with_version<T, V, E>(
    bytes: &[u8],
    gate: impl FnOnce(V) -> Result<(), E>,
) -> Result<T, E>
where
    T: DeserializeOwned,
    V: VersionWord,
    E: From<IoError>,
{
    let value = cbor_value(bytes)?;
    let version = version_of::<V>(&value)?;
    gate(version)?;
    value.deserialized().map_err(|error| IoError::Cbor(error.to_string()).into())
}

fn cbor_value(bytes: &[u8]) -> Result<ciborium::Value, IoError> {
    let mut reader = bytes;
    let value = ciborium::from_reader(&mut reader)
        .map_err(|error| IoError::Cbor(error.to_string()))?;
    if reader.is_empty() {
        Ok(value)
    } else {
        Err(IoError::Cbor(format!("{} trailing bytes after the CBOR item", reader.len())))
    }
}

fn version_of<V: VersionWord>(value: &ciborium::Value) -> Result<V, IoError> {
    #[derive(Deserialize)]
    struct Peek<V> {
        version: V,
    }
    let peek: Peek<V> = value.deserialized().map_err(|error| IoError::Cbor(error.to_string()))?;
    Ok(peek.version)
}

/// Map a [`CodecError`] onto an artifact error that has the family variants
/// `Decode`, `Encode`, `LimitsExceeded`, `Malformed`, `UnsupportedVersion { version: u16 }`
/// and `UnsupportedSemantics`.
#[macro_export]
macro_rules! map_artifact_check {
    ($err:expr, $ty:ident) => {{
        match $crate::artifact_codec::CodecError::into_check($err) {
            $crate::artifact_codec::ArtifactCheckError::Decode(message) => $ty::Decode(message),
            $crate::artifact_codec::ArtifactCheckError::Encode(message) => $ty::Encode(message),
            $crate::artifact_codec::ArtifactCheckError::LimitsExceeded(what) => {
                $ty::LimitsExceeded(what)
            }
            $crate::artifact_codec::ArtifactCheckError::Malformed(message) => {
                $ty::Malformed(message)
            }
            $crate::artifact_codec::ArtifactCheckError::UnsupportedVersion { version } => {
                match u16::try_from(version) {
                    Ok(version) => $ty::UnsupportedVersion { version },
                    Err(_) => $ty::Decode(format!("unsupported artifact version {version}")),
                }
            }
            $crate::artifact_codec::ArtifactCheckError::UnsupportedSemantics(detail) => {
                $ty::UnsupportedSemantics(detail)
            }
        }
    }};
}
