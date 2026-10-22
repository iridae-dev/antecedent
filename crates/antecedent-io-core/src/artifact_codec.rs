//! Shared sectioned-artifact codec.
//!
//! One container writer and reader for the fixed one-, two- and three-section
//! artifacts, and one CBOR version gate. Callers keep their own error type and
//! map [`CodecError`] into it, so the on-wire bytes and the refusal text stay
//! those of the artifact that owns the payload.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::io::Cursor;

use serde::Deserialize;
use serde::Serialize;
use serde::de::DeserializeOwned;

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
    #[must_use]
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
    #[must_use]
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

/// The header, digest and limit refusals every sectioned checksummed artifact shares, as one
/// generated enum.
///
/// The seven shared variants (`Decode`, `UnsupportedVersion`, `UnsupportedSemantics`,
/// `LimitsExceeded`, `Malformed`, `IdentityMismatch`, `Encode`), their `Display` texts, the
/// `From<IoError>` decode mapping and the `Error` impl are generated from the artifact's
/// `noun` and the spelling of its identity-mismatch detail. What differs between artifacts is
/// declared beside them: extra `variants` and the `display` arms that render them, written as
/// the arms of a `match self` with the formatter bound to the identifier named in
/// `display(..)`. `refusal()` is not generated; each artifact keeps the one that matches the
/// refusals it registers.
///
/// Derives are the caller's: put `#[derive(Clone, Debug, PartialEq)]` (and `Eq` where every
/// field allows it) among the attributes.
#[macro_export]
macro_rules! sectioned_artifact_enum {
    (
        $(#[$meta:meta])*
        $name:ident {
            noun: $noun:literal,
            identity: $identity:expr,
            variants {
                $(
                    $(#[$vmeta:meta])*
                    $variant:ident
                    $(($($tuple:ty),+ $(,)?))?
                    $({ $($(#[$fmeta:meta])* $field:ident : $fty:ty),+ $(,)? })?
                ),* $(,)?
            }
            display($f:ident) { $($arms:tt)* }
        }
    ) => {
        $(#[$meta])*
        #[non_exhaustive]
        pub enum $name {
            /// The bytes do not decode as this format.
            Decode(String),
            /// Another major version, refused before the payload is interpreted.
            UnsupportedVersion {
                /// The stored version.
                version: u16,
            },
            /// A feature marker, claim, calibration coordinate or caveat that is not this format's.
            UnsupportedSemantics(&'static str),
            /// A collection exceeds the format's bound.
            LimitsExceeded(&'static str),
            /// The stored declarations are not a well-formed request.
            Malformed(String),
            /// A stored or retained identity field differs from the one the inputs produce.
            IdentityMismatch {
                /// Which identity field differs.
                field: &'static str,
            },
            /// The artifact could not be encoded.
            Encode(String),
            $(
                $(#[$vmeta])*
                $variant
                $(($($tuple),+))?
                $({ $($(#[$fmeta])* $field : $fty),+ })?,
            )*
        }

        impl std::fmt::Display for $name {
            fn fmt(&self, $f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                match self {
                    Self::Decode(message) => {
                        write!($f, "{noun} artifact does not decode: {message}", noun = $noun)
                    }
                    Self::UnsupportedVersion { version } => {
                        write!($f, "unsupported {noun} artifact version {version}", noun = $noun)
                    }
                    Self::UnsupportedSemantics(detail) => {
                        write!($f, "unsupported {noun} semantics: {detail}", noun = $noun)
                    }
                    Self::LimitsExceeded(what) => {
                        write!($f, "{noun} consumer limit exceeded: {what}", noun = $noun)
                    }
                    Self::Malformed(message) => {
                        write!($f, "malformed {noun} artifact: {message}", noun = $noun)
                    }
                    Self::IdentityMismatch { field } => {
                        write!($f, "{}: {field} changed", $identity)
                    }
                    Self::Encode(message) => {
                        write!($f, "{noun} artifact does not encode: {message}", noun = $noun)
                    }
                    $($arms)*
                }
            }
        }

        impl std::error::Error for $name {}

        impl From<$crate::IoError> for $name {
            fn from(error: $crate::IoError) -> Self {
                Self::Decode(error.to_string())
            }
        }
    };
}

/// One refusal type for a sectioned checksummed artifact whose replay is a core estimator
/// run.
///
/// Built on [`sectioned_artifact_enum!`]: it adds the `ResultMismatch` and `Refused`
/// variants, the registered `refusal()` and the conversion from an estimator refusal.
/// String const generics are not available, so each artifact expands this once. `$noun` is
/// the display name, `$contract` the refusal prefix, `$replay` the noun in a result mismatch
/// ("result" or "record"), and `$report` whether a result mismatch is a registered refusal.
#[macro_export]
macro_rules! sectioned_artifact_error {
    (
        $(#[$meta:meta])*
        $name:ident,
        $noun:literal,
        $contract:literal,
        $replay:literal,
        $report:literal
    ) => {
        $crate::sectioned_artifact_enum! {
            $(#[$meta])*
            #[derive(Clone, Debug, PartialEq)]
            $name {
                noun: $noun,
                identity: concat!($contract, ".wrong_contract"),
                variants {
                    /// A stored result differs from the recomputed one.
                    ResultMismatch(&'static str),
                    /// The core estimator refused. The message begins with its detail.
                    Refused {
                        /// Registered reason code.
                        code: &'static str,
                        /// `detail: explanation`.
                        message: String,
                    },
                }
                display(f) {
                    Self::ResultMismatch(what) => write!(
                        f,
                        "stored {noun} {replay} does not replay: {what}",
                        noun = $noun,
                        replay = $replay,
                    ),
                    Self::Refused { message, .. } => f.write_str(message),
                }
            }
        }

        impl $name {
            /// The registered refusal this error carries: `(code, detail, explanation)`.
            ///
            /// Present for a core refusal, a changed identity and, when the artifact
            /// reports a replay mismatch, a result that does not replay. Absent for
            /// corruption, an unsupported version, a limit and an encoding failure.
            #[must_use]
            pub fn refusal(&self) -> Option<(&'static str, String, String)> {
                match self {
                    Self::Refused { code, message } => {
                        let (detail, text) =
                            message.split_once(": ").unwrap_or((message.as_str(), ""));
                        Some((*code, detail.to_owned(), text.to_owned()))
                    }
                    Self::IdentityMismatch { field } => {
                        Some(antecedent_core::refusal::changed_identity(
                            &format!("{contract}.wrong_contract", contract = $contract),
                            field,
                        ))
                    }
                    Self::ResultMismatch(what) => {
                        if $report {
                            Some(antecedent_core::refusal::replay_mismatch(
                                &format!(
                                    "{contract}.report_replay_mismatch",
                                    contract = $contract
                                ),
                                format!("the stored {what} does not replay"),
                            ))
                        } else {
                            None
                        }
                    }
                    _ => None,
                }
            }
        }

        impl From<antecedent_estimate::EstimationError> for $name {
            fn from(error: antecedent_estimate::EstimationError) -> Self {
                match error {
                    antecedent_estimate::EstimationError::Refused { code, message }
                    | antecedent_estimate::EstimationError::RefusedWithFields {
                        code,
                        message,
                        ..
                    } => Self::Refused { code, message },
                    other => Self::Malformed(other.to_string()),
                }
            }
        }
    };
}

/// Exact-law replay bounds.
///
/// The one definition of the bounds a consumer imposes on an exact-law replay: the
/// evaluation budget, the support the replayed laws may materialize, and the law count
/// and law size. `ZTransportConsumeLimits` is this type; the artifacts that bound more
/// (a search budget, a request count) carry these four fields flat and convert with
/// `From`, so the checks below are the only interpretation of them.
#[derive(Clone, Copy, Debug)]
pub struct ConsumeLimits {
    /// Exact-evaluation operation and depth limits of the replay.
    pub evaluation: antecedent_expr::ExactEvaluationLimits,
    /// Largest support the replayed laws may materialize.
    pub max_support_rows: usize,
    /// Most laws an artifact may carry.
    pub max_laws: usize,
    /// Most cells one stored law may carry.
    pub max_law_cells: usize,
}

impl Default for ConsumeLimits {
    fn default() -> Self {
        Self {
            evaluation: antecedent_expr::ExactEvaluationLimits::default(),
            max_support_rows: 1_000_000,
            max_laws: 256,
            max_law_cells: 1_000_000,
        }
    }
}

impl ConsumeLimits {
    /// The identification limits the replay verifies a proof under.
    #[must_use]
    pub const fn identification(&self) -> antecedent_identify::SidLimits {
        antecedent_identify::SidLimits {
            steps: self.evaluation.operations,
            depth: self.evaluation.depth,
        }
    }

    /// The stored evaluation budget, support or law count that exceeds these
    /// bounds: `operation limit`, `depth limit`, `support rows` or `law count`.
    #[must_use]
    pub fn budget(
        &self,
        operation_limit: usize,
        depth_limit: usize,
        stored_support_rows: usize,
        law_count: usize,
    ) -> Option<&'static str> {
        if operation_limit > self.evaluation.operations {
            Some("operation limit")
        } else if depth_limit > self.evaluation.depth {
            Some("depth limit")
        } else {
            stored_law_bounds(stored_support_rows, self.max_support_rows, law_count, self.max_laws)
        }
    }

    /// `law cells` when a stored law's probabilities, or its empirical counts
    /// when `empirical` is set, exceed [`Self::max_law_cells`].
    #[must_use]
    pub fn law_cells(
        &self,
        laws: &[crate::exact_law_wire::ExactLawWire],
        empirical: bool,
    ) -> Option<&'static str> {
        let over = laws.iter().any(|law| {
            law.probabilities.len() > self.max_law_cells
                || (empirical
                    && law
                        .empirical_counts
                        .as_ref()
                        .is_some_and(|counts| counts.len() > self.max_law_cells))
        });
        if over { Some("law cells") } else { None }
    }
}

/// `support rows` or `law count` when a stored support size or law count
/// exceeds the consumer's bound, support first.
///
/// For the replay checks whose evaluation labels differ from
/// [`ConsumeLimits::budget`] and that carry no law-cell bound.
#[must_use]
pub const fn stored_law_bounds(
    stored_support_rows: usize,
    max_support_rows: usize,
    law_count: usize,
    max_laws: usize,
) -> Option<&'static str> {
    if stored_support_rows > max_support_rows {
        Some("support rows")
    } else if law_count > max_laws {
        Some("law count")
    } else {
        None
    }
}

/// `search operation limit` or `search depth limit` when the search budget a producer
/// recorded exceeds the consumer's, operations first.
#[must_use]
pub const fn search_bounds(
    stored_operations: usize,
    stored_depth: usize,
    limit: antecedent_core::SearchLimits,
) -> Option<&'static str> {
    if stored_operations > limit.operations {
        Some("search operation limit")
    } else if stored_depth > limit.depth {
        Some("search depth limit")
    } else {
        None
    }
}

/// Count words enter the canonical stream as `u64`.
///
/// The word is [`crate::error::lossless_u64`]: a `usize` that does not fit in
/// `u64` does not compile, so a count is never saturated to `u64::MAX`.
// doc(hidden): implemented by antecedent-io artifact metadata.
#[doc(hidden)]
pub trait CanonicalSize {
    /// Write `value` as one canonical word.
    fn size(&mut self, value: usize) -> &mut Self;
}

impl CanonicalSize for antecedent_core::CanonicalWriter {
    fn size(&mut self, value: usize) -> &mut Self {
        self.word(crate::error::lossless_u64(value))
    }
}

/// Hex digest of one canonical value.
///
/// The domain and the value are the digest inputs. This only shares the
/// encode-and-hex step.
///
/// # Errors
/// Canonical encoding failure.
pub fn digest_hex<T: Serialize>(
    domain: antecedent_core::IdentityDomain,
    value: &T,
) -> Result<String, IoError> {
    Ok(crate::identity::digest_wire(domain, value)?.to_hex())
}

/// Refuse unless `stored` is the digest `expected` produced.
///
/// An encoding failure is `on_encode`. A digest that differs is `on_mismatch`.
/// The bytes that were hashed are whatever `expected` already hashed.
pub fn require_digest<E>(
    stored: &str,
    expected: Result<String, IoError>,
    on_encode: impl FnOnce(IoError) -> E,
    on_mismatch: impl FnOnce() -> E,
) -> Result<(), E> {
    match expected {
        Ok(value) if value == stored => Ok(()),
        Ok(_) => Err(on_mismatch()),
        Err(error) => Err(on_encode(error)),
    }
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
    if crate::error::declared_exceeds_usize_limit(declared, max_bytes) {
        return Err(CodecError::SectionsTooLarge);
    }
    let mut payloads = Vec::with_capacity(section_ids.len());
    for id in section_ids {
        let section = reader.load_section(id).map_err(CodecError::Decode)?;
        payloads.push(section.as_bytes().to_vec());
    }
    Ok(payloads)
}

/// The `ArtifactKind::Other` label of a container, or `None` for bytes that are
/// not a readable container or whose kind is a built-in one.
///
/// Reads the manifest only: nothing is checksummed or decoded, so a caller
/// still routes to the kind's own decoder, which refuses a corrupt body.
#[must_use]
pub fn container_kind(bytes: &[u8]) -> Option<String> {
    let reader = ArtifactReader::open_seek(Cursor::new(bytes)).ok()?;
    match &reader.manifest().artifact_kind {
        ArtifactKind::Other(kind) => Some(kind.clone()),
        _ => None,
    }
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
    let value =
        ciborium::from_reader(&mut reader).map_err(|error| IoError::Cbor(error.to_string()))?;
    if reader.is_empty() {
        Ok(value)
    } else {
        Err(IoError::Cbor(format!("{} trailing bytes after the CBOR item", reader.len())))
    }
}

/// Decode little-endian IEEE-754 `f64` words from a section payload.
///
/// Each value is copied out of its 8-byte word. A trailing partial word is
/// not a value; callers that require an exact multiple of eight bytes check
/// the length themselves.
#[must_use]
pub(crate) fn f64s_from_le_bytes(bytes: &[u8]) -> Vec<f64> {
    let mut values = Vec::with_capacity(bytes.len() / 8);
    for chunk in bytes.chunks_exact(8) {
        let mut word = [0_u8; 8];
        word.copy_from_slice(chunk);
        values.push(f64::from_le_bytes(word));
    }
    values
}

/// Encode `f64` values as little-endian IEEE-754 words.
///
/// Bit patterns are preserved, including the sign of zero and NaN payloads.
#[must_use]
pub(crate) fn f64s_to_le_bytes(values: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(8));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Next `len` values of a decoded real section.
///
/// The caller has already checked that the section is long enough. A short
/// slice panics, the same as indexing it.
pub(crate) fn split_reals<'a>(values: &mut &'a [f64], len: usize) -> &'a [f64] {
    let (head, tail) = values.split_at(len);
    *values = tail;
    head
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
            // `ArtifactCheckError` is non-exhaustive and this expands in other crates.
            #[allow(unreachable_patterns, reason = "ArtifactCheckError is non-exhaustive and this macro expands in other crates")]
            other => $ty::Malformed(other.to_string()),
        }
    }};
}

#[cfg(test)]
mod le_real_tests {
    use super::{f64s_from_le_bytes, f64s_to_le_bytes};

    #[test]
    fn le_real_bytes_round_trip_keeps_bits() {
        let values =
            [0.0, -0.0, 1.0, -1.5, f64::MIN_POSITIVE, f64::from_bits(0x7ff8_0000_0000_0001)];
        let bytes = f64s_to_le_bytes(&values);
        let direct: Vec<u8> = values.iter().flat_map(|value| value.to_le_bytes()).collect();
        assert_eq!(bytes, direct);
        let decoded = f64s_from_le_bytes(&bytes);
        let bits = |sample: &[f64]| sample.iter().map(|value| value.to_bits()).collect::<Vec<_>>();
        assert_eq!(bits(&decoded), bits(&values));
        let mut trailing = bytes;
        trailing.push(0x7f);
        assert_eq!(f64s_from_le_bytes(&trailing).len(), values.len());
    }
}
