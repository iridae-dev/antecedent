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
            Self::Decode(error) if error.is_container_fault() => {
                ArtifactCheckError::Container(error)
            }
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
    /// The container itself is damaged or over a bound (a checksum, a framing, a size or a
    /// compression fault); the typed cause is kept so a caller can match it.
    #[error("{0}")]
    Container(IoError),
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
/// The eight shared variants (`Decode`, `Container`, `UnsupportedVersion`,
/// `UnsupportedSemantics`, `LimitsExceeded`, `Malformed`, `IdentityMismatch`, `Encode`), their
/// `Display` texts, the `From<IoError>` decode mapping (a fault of the container itself keeps
/// its typed [`IoError`] in `Container`) and the `Error` impl are generated from the
/// artifact's `noun` and the spelling of its identity-mismatch detail. What differs between artifacts is
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
            /// The container is damaged or over a bound; the typed cause is kept.
            Container($crate::IoError),
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
                    Self::Container(error) => {
                        write!($f, "{noun} artifact does not decode: {error}", noun = $noun)
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
                if error.is_container_fault() {
                    Self::Container(error)
                } else {
                    Self::Decode(error.to_string())
                }
            }
        }

        impl $crate::artifact_codec::ArtifactFamilyError for $name {
            fn from_check(check: $crate::artifact_codec::ArtifactCheckError) -> Self {
                match check {
                    $crate::artifact_codec::ArtifactCheckError::Decode(message) => {
                        Self::Decode(message)
                    }
                    $crate::artifact_codec::ArtifactCheckError::Container(error) => {
                        Self::Container(error)
                    }
                    $crate::artifact_codec::ArtifactCheckError::Encode(message) => {
                        Self::Encode(message)
                    }
                    $crate::artifact_codec::ArtifactCheckError::LimitsExceeded(what) => {
                        Self::LimitsExceeded(what)
                    }
                    $crate::artifact_codec::ArtifactCheckError::Malformed(message) => {
                        Self::Malformed(message)
                    }
                    $crate::artifact_codec::ArtifactCheckError::UnsupportedVersion { version } => {
                        match u16::try_from(version) {
                            Ok(version) => Self::UnsupportedVersion { version },
                            Err(_) => {
                                Self::Decode(format!("unsupported artifact version {version}"))
                            }
                        }
                    }
                    $crate::artifact_codec::ArtifactCheckError::UnsupportedSemantics(detail) => {
                        Self::UnsupportedSemantics(detail)
                    }
                    // `ArtifactCheckError` is non-exhaustive and this expands in other crates.
                    #[allow(
                        unreachable_patterns,
                        reason = "ArtifactCheckError is non-exhaustive and this macro expands in other crates"
                    )]
                    other => Self::Malformed(other.to_string()),
                }
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
                    /// A registered refusal: the core estimator's, or one the artifact states.
                    Refused {
                        /// Registered reason code.
                        code: &'static str,
                        /// Namespaced detail identifier.
                        detail: String,
                        /// What was refused and the next attempt; empty when the refusal
                        /// carries only its detail.
                        explanation: String,
                    },
                }
                display(f) {
                    Self::ResultMismatch(what) => write!(
                        f,
                        "stored {noun} {replay} does not replay: {what}",
                        noun = $noun,
                        replay = $replay,
                    ),
                    Self::Refused { detail, explanation, .. } if explanation.is_empty() => {
                        f.write_str(detail)
                    }
                    Self::Refused { detail, explanation, .. } => {
                        write!(f, "{detail}: {explanation}")
                    }
                }
            }
        }

        impl $name {
            /// A registered refusal the artifact states itself, with its detail identifier and
            /// explanation kept apart.
            #[must_use]
            pub fn refused(
                code: &'static str,
                detail: impl Into<String>,
                explanation: impl Into<String>,
            ) -> Self {
                Self::Refused { code, detail: detail.into(), explanation: explanation.into() }
            }

            /// The registered refusal this error carries: `(code, detail, explanation)`.
            ///
            /// Present for a core refusal, a changed identity and, when the artifact
            /// reports a replay mismatch, a result that does not replay. Absent for
            /// corruption, an unsupported version, a limit and an encoding failure.
            #[must_use]
            pub fn refusal(&self) -> Option<(&'static str, String, String)> {
                match self {
                    Self::Refused { code, detail, explanation } => {
                        Some((*code, detail.clone(), explanation.clone()))
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
                    } => match message.detail() {
                        Some(detail) => Self::Refused {
                            code,
                            detail: detail.to_owned(),
                            explanation: message.explanation().to_owned(),
                        },
                        None => Self::Refused {
                            code,
                            detail: message.explanation().to_owned(),
                            explanation: String::new(),
                        },
                    },
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

/// An artifact error that has the container-level family variants (`Decode`, `Encode`,
/// `LimitsExceeded`, `Malformed`, `UnsupportedVersion { version: u16 }` and
/// `UnsupportedSemantics`) and decodes from an [`IoError`]. [`sectioned_artifact_enum!`]
/// implements it, which is what lets the drivers below serve every sectioned artifact.
pub trait ArtifactFamilyError: Sized + From<IoError> {
    /// The artifact's own spelling of a container-level check.
    fn from_check(check: ArtifactCheckError) -> Self;
}

/// `UnsupportedSemantics(what)` unless `holds`: the marker checks of a consumed artifact (its
/// feature marker, claim, calibration and caveats are the ones this format defines).
///
/// # Errors
/// The artifact's `UnsupportedSemantics(what)`.
pub fn require_semantics<E: ArtifactFamilyError>(holds: bool, what: &'static str) -> Result<(), E> {
    if holds { Ok(()) } else { Err(E::from_check(ArtifactCheckError::UnsupportedSemantics(what))) }
}

/// [`require_semantics`] over an ordered list of `(holds, what)` markers: the first that does not
/// hold is the refusal. Every marker is evaluated before the first is checked, so a marker is a
/// plain comparison and not a check with a side effect.
///
/// # Errors
/// The artifact's `UnsupportedSemantics(what)` of the first marker that does not hold.
pub fn require_markers<E: ArtifactFamilyError>(markers: &[(bool, &'static str)]) -> Result<(), E> {
    markers.iter().try_for_each(|&(holds, what)| require_semantics::<E>(holds, what))
}

/// Write a CBOR metadata section followed by `payloads` as one checksummed container.
///
/// A blank artifact id is refused before the metadata is encoded; a file over the header's
/// `max_bytes` is `LimitsExceeded("artifact bytes")`.
///
/// # Errors
/// The artifact's `Encode`, `LimitsExceeded` or `Malformed` for the container-level failures.
pub fn encode_sectioned<E: ArtifactFamilyError, M: Serialize>(
    header: &ArtifactContainer<'_>,
    meta_section: &str,
    meta: &M,
    payloads: &[OutSection<'_>],
) -> Result<Vec<u8>, E> {
    encode_sectioned_admitting(header, meta_section, meta, payloads, |_| Ok(()))
}

/// [`encode_sectioned`] with an admission check on the uncompressed bytes (the encoded metadata
/// plus every payload), run after the metadata encodes and before the container is built, so an
/// artifact that would exceed its export bound refuses without being assembled.
///
/// # Errors
/// As [`encode_sectioned`], and `admit`'s refusal.
pub fn encode_sectioned_admitting<E: ArtifactFamilyError, M: Serialize>(
    header: &ArtifactContainer<'_>,
    meta_section: &str,
    meta: &M,
    payloads: &[OutSection<'_>],
    admit: impl FnOnce(usize) -> Result<(), E>,
) -> Result<Vec<u8>, E> {
    if header.artifact_id.trim().is_empty() {
        return Err(E::from_check(ArtifactCheckError::Encode("missing artifact id".into())));
    }
    let meta_bytes = crate::convert::to_cbor(meta)
        .map_err(|error| E::from_check(ArtifactCheckError::Encode(error.to_string())))?;
    let uncompressed = payloads
        .iter()
        .fold(meta_bytes.len(), |total, section| total.saturating_add(section.bytes.len()));
    admit(uncompressed)?;
    let mut sections = Vec::with_capacity(payloads.len() + 1);
    sections.push(OutSection { id: meta_section, content_type: CONTENT_CBOR, bytes: &meta_bytes });
    sections.extend(payloads.iter().map(|section| OutSection {
        id: section.id,
        content_type: section.content_type,
        bytes: section.bytes,
    }));
    encode_container(header, &sections).map_err(|error| E::from_check(error.into_check()))
}

/// Read a container of a CBOR metadata section followed by `payload_sections`, refusing another
/// version before the metadata is interpreted as `M`.
///
/// Returns the metadata and the payload sections in order.
///
/// # Errors
/// The artifact's `Decode`, `LimitsExceeded`, `Malformed` or `UnsupportedVersion`.
pub fn decode_sectioned<E: ArtifactFamilyError, M: DeserializeOwned>(
    kind: &str,
    max_bytes: usize,
    meta_section: &str,
    payload_sections: &[&str],
    supported_version: u16,
    bytes: &[u8],
) -> Result<(M, Vec<Vec<u8>>), E> {
    let mut ids = Vec::with_capacity(payload_sections.len() + 1);
    ids.push(meta_section);
    ids.extend_from_slice(payload_sections);
    let mut parts = decode_container(kind, max_bytes, &ids, bytes)
        .map_err(|error| E::from_check(error.into_check()))?;
    if parts.len() != ids.len() {
        return Err(E::from_check(ArtifactCheckError::Malformed(
            "unsupported container layout".into(),
        )));
    }
    let meta_bytes = parts.remove(0);
    let meta = decode_with_version(&meta_bytes, |version: u16| {
        if version == supported_version {
            Ok(())
        } else {
            Err(E::from_check(ArtifactCheckError::UnsupportedVersion { version: version.into() }))
        }
    })?;
    Ok((meta, parts))
}

/// Read a container laid out as the metadata section followed by one of several alternative
/// `payload_layouts`, tried in order, for a format that grew an optional trailing section.
///
/// The first layout the container matches is read; when none matches, the first layout's refusal
/// is returned. The metadata is gated on `accepts_version` before it is interpreted as `M`.
/// Returns the metadata, the index of the matched layout and its payload sections.
///
/// # Errors
/// The artifact's `Decode`, `LimitsExceeded`, `Malformed` or `UnsupportedVersion`.
pub fn decode_sectioned_layouts<E: ArtifactFamilyError, M: DeserializeOwned>(
    kind: &str,
    max_bytes: usize,
    meta_section: &str,
    payload_layouts: &[&[&str]],
    accepts_version: impl Fn(u16) -> bool,
    bytes: &[u8],
) -> Result<(M, usize, Vec<Vec<u8>>), E> {
    let mut first_refusal = None;
    let mut matched = None;
    for (index, payload_sections) in payload_layouts.iter().enumerate() {
        let mut ids = Vec::with_capacity(payload_sections.len() + 1);
        ids.push(meta_section);
        ids.extend_from_slice(payload_sections);
        match decode_container(kind, max_bytes, &ids, bytes) {
            Ok(parts) => {
                matched = Some((index, ids.len(), parts));
                break;
            }
            Err(error) => {
                first_refusal.get_or_insert(error);
            }
        }
    }
    let Some((index, expected_sections, mut parts)) = matched else {
        let refusal = first_refusal.map_or_else(
            || ArtifactCheckError::Malformed("unsupported container layout".into()),
            CodecError::into_check,
        );
        return Err(E::from_check(refusal));
    };
    if parts.len() != expected_sections {
        return Err(E::from_check(ArtifactCheckError::Malformed(
            "unsupported container layout".into(),
        )));
    }
    let meta_bytes = parts.remove(0);
    let meta = decode_with_version(&meta_bytes, |version: u16| {
        if accepts_version(version) {
            Ok(())
        } else {
            Err(E::from_check(ArtifactCheckError::UnsupportedVersion { version: version.into() }))
        }
    })?;
    Ok((meta, index, parts))
}

/// [`encode_sectioned`] for an artifact whose error is [`IoError`]: a CBOR metadata section
/// followed by `payloads`, with a container-level failure mapped by [`CodecError::into_io`] using
/// `layout` as the convert message of a foreign layout.
///
/// A blank artifact id is refused before the metadata is encoded.
///
/// # Errors
/// A blank id, an encoding failure, or a file over the header's `max_bytes`.
pub fn encode_sectioned_io<M: Serialize>(
    header: &ArtifactContainer<'_>,
    meta_section: &str,
    meta: &M,
    payloads: &[OutSection<'_>],
    layout: &'static str,
) -> Result<Vec<u8>, IoError> {
    if header.artifact_id.trim().is_empty() {
        return Err(CodecError::MissingArtifactId.into_io(layout));
    }
    let meta_bytes = crate::convert::to_cbor(meta)?;
    let mut sections = Vec::with_capacity(payloads.len() + 1);
    sections.push(OutSection { id: meta_section, content_type: CONTENT_CBOR, bytes: &meta_bytes });
    sections.extend(payloads.iter().map(|section| OutSection {
        id: section.id,
        content_type: section.content_type,
        bytes: section.bytes,
    }));
    encode_container(header, &sections).map_err(|error| error.into_io(layout))
}

/// Read a container of a metadata section followed by `payload_sections` for an artifact whose
/// error is [`IoError`], returning the metadata bytes and the payload sections in order. The
/// metadata is not interpreted; [`decode_sectioned_io`] and [`decode_sectioned_io_versioned`]
/// add the CBOR decode.
///
/// A file over `max_bytes` is [`IoError::TooLarge`]; a foreign layout is the convert message
/// `layout`.
///
/// # Errors
/// Oversize, truncated, corrupt, or differently laid out containers.
pub fn decode_sectioned_io_raw<const N: usize>(
    kind: &str,
    max_bytes: usize,
    meta_section: &str,
    payload_sections: &[&str; N],
    layout: &'static str,
    bytes: &[u8],
) -> Result<(Vec<u8>, [Vec<u8>; N]), IoError> {
    let mut ids = Vec::with_capacity(N + 1);
    ids.push(meta_section);
    ids.extend_from_slice(payload_sections);
    let mut parts =
        decode_container(kind, max_bytes, &ids, bytes).map_err(|error| error.into_io(layout))?;
    let meta = parts.remove(0);
    let payloads = <[Vec<u8>; N]>::try_from(parts).map_err(|_| IoError::Convert(layout.into()))?;
    Ok((meta, payloads))
}

/// [`decode_sectioned_io_raw`] with the metadata decoded as CBOR, for a format that carries no
/// version word.
///
/// # Errors
/// As [`decode_sectioned_io_raw`], and a metadata section that is not an `M`.
pub fn decode_sectioned_io<M: DeserializeOwned, const N: usize>(
    kind: &str,
    max_bytes: usize,
    meta_section: &str,
    payload_sections: &[&str; N],
    layout: &'static str,
    bytes: &[u8],
) -> Result<(M, [Vec<u8>; N]), IoError> {
    let (meta_bytes, payloads) =
        decode_sectioned_io_raw(kind, max_bytes, meta_section, payload_sections, layout, bytes)?;
    Ok((crate::convert::from_cbor(&meta_bytes)?, payloads))
}

/// [`decode_sectioned_io_raw`] with the metadata gated on its `version` word before it is
/// interpreted as `M`, so another version is [`IoError::UnsupportedVersion`] and not a shape
/// error.
///
/// # Errors
/// As [`decode_sectioned_io_raw`], an unsupported version, and a metadata section that is not an
/// `M`.
pub fn decode_sectioned_io_versioned<M: DeserializeOwned, const N: usize>(
    kind: &str,
    max_bytes: usize,
    meta_section: &str,
    payload_sections: &[&str; N],
    layout: &'static str,
    supported_version: u16,
    bytes: &[u8],
) -> Result<(M, [Vec<u8>; N]), IoError> {
    let (meta_bytes, payloads) =
        decode_sectioned_io_raw(kind, max_bytes, meta_section, payload_sections, layout, bytes)?;
    let meta = decode_with_version(&meta_bytes, |version: u16| {
        if version == supported_version {
            Ok(())
        } else {
            Err(IoError::UnsupportedVersion { version: u32::from(version) })
        }
    })?;
    Ok((meta, payloads))
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
    value.deserialized().map_err(|error| IoError::cbor_decode(error).into())
}

/// One CBOR item, trailing bytes refused (the shared single-item decoder).
fn cbor_value(bytes: &[u8]) -> Result<ciborium::Value, IoError> {
    crate::convert::from_cbor(bytes)
}

/// Decode little-endian IEEE-754 `f64` words from a section payload.
///
/// Each value is copied out of its 8-byte word. A trailing partial word is
/// not a value; callers that require an exact multiple of eight bytes check
/// the length themselves.
#[must_use]
#[doc(hidden)]
pub fn f64s_from_le_bytes(bytes: &[u8]) -> Vec<f64> {
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
#[doc(hidden)]
pub fn f64s_to_le_bytes(values: &[f64]) -> Vec<u8> {
    let mut bytes = Vec::with_capacity(values.len().saturating_mul(8));
    for value in values {
        bytes.extend_from_slice(&value.to_le_bytes());
    }
    bytes
}

/// Encode `u32` values as little-endian words.
#[must_use]
pub(crate) fn u32s_to_le_bytes(values: &[u32]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

/// Encode `u64` values as little-endian words.
#[must_use]
pub(crate) fn u64s_to_le_bytes(values: &[u64]) -> Vec<u8> {
    values.iter().flat_map(|value| value.to_le_bytes()).collect()
}

/// Next `len` values of a decoded real section.
///
/// The caller has already checked that the section is long enough. A short
/// slice panics, the same as indexing it.
#[doc(hidden)]
pub fn split_reals<'a>(values: &mut &'a [f64], len: usize) -> &'a [f64] {
    let (head, tail) = values.split_at(len);
    *values = tail;
    head
}

fn version_of<V: VersionWord>(value: &ciborium::Value) -> Result<V, IoError> {
    #[derive(Deserialize)]
    struct Peek<V> {
        version: V,
    }
    let peek: Peek<V> = value.deserialized().map_err(IoError::cbor_decode)?;
    Ok(peek.version)
}

/// The first field of two identity records that differs, as the name the artifact reports.
///
/// The fields are compared in the order they are listed, which is the order the artifact's
/// contract names them; `None` when every listed field is equal. Each field type needs
/// `PartialEq`.
#[macro_export]
macro_rules! first_identity_difference {
    ($stored:expr, $other:expr, { $($field:ident => $name:literal),+ $(,)? }) => {{
        let (stored, other) = (&$stored, &$other);
        $( if stored.$field != other.$field { Some($name) } else )+ { None }
    }};
}

/// Map a [`CodecError`] onto an artifact error that implements [`ArtifactFamilyError`] (every
/// error generated by [`sectioned_artifact_enum!`]).
#[macro_export]
macro_rules! map_artifact_check {
    ($err:expr, $ty:ident) => {{
        <$ty as $crate::artifact_codec::ArtifactFamilyError>::from_check(
            $crate::artifact_codec::CodecError::into_check($err),
        )
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

#[cfg(test)]
mod sectioned_io_tests {
    use super::{
        ArtifactContainer, CONTENT_BYTES, OutSection, decode_sectioned_io, decode_sectioned_io_raw,
        decode_sectioned_io_versioned, encode_sectioned_io,
    };
    use crate::error::IoError;
    use serde::{Deserialize, Serialize};

    #[derive(Debug, PartialEq, Serialize, Deserialize)]
    struct Meta {
        version: u16,
        label: String,
    }

    const KIND: &str = "sectioned_io_test";
    const LAYOUT: &str = "unsupported test layout";

    fn header(artifact_id: &str) -> ArtifactContainer<'_> {
        ArtifactContainer {
            kind: KIND,
            note: "test",
            max_bytes: 1 << 16,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        }
    }

    fn encode(version: u16) -> Vec<u8> {
        encode_sectioned_io(
            &header("test:1"),
            "test.meta",
            &Meta { version, label: "a".into() },
            &[OutSection { id: "test.payload", content_type: CONTENT_BYTES, bytes: b"xyz" }],
            LAYOUT,
        )
        .unwrap()
    }

    #[test]
    fn round_trip_returns_metadata_and_payloads_in_order() {
        let bytes = encode(1);
        let (meta, [payload]): (Meta, _) =
            decode_sectioned_io(KIND, 1 << 16, "test.meta", &["test.payload"], LAYOUT, &bytes)
                .unwrap();
        assert_eq!(meta, Meta { version: 1, label: "a".into() });
        assert_eq!(payload, b"xyz");
    }

    #[test]
    fn blank_artifact_id_is_refused_before_encoding() {
        let result = encode_sectioned_io(
            &header("  "),
            "test.meta",
            &Meta { version: 1, label: String::new() },
            &[],
            LAYOUT,
        );
        assert!(
            matches!(result, Err(IoError::Convert(message)) if message == "missing artifact id")
        );
    }

    #[test]
    fn a_foreign_layout_is_the_callers_convert_message() {
        let bytes = encode(1);
        let wrong = decode_sectioned_io_raw(KIND, 1 << 16, "test.meta", &[], LAYOUT, &bytes);
        assert!(matches!(wrong, Err(IoError::Convert(message)) if message == LAYOUT));
        let oversize =
            decode_sectioned_io_raw(KIND, 4, "test.meta", &["test.payload"], LAYOUT, &bytes);
        assert!(matches!(oversize, Err(IoError::TooLarge)));
    }

    #[test]
    fn another_version_is_refused_before_the_metadata_is_interpreted() {
        let bytes = encode(2);
        let result: Result<(Meta, [Vec<u8>; 1]), IoError> = decode_sectioned_io_versioned(
            KIND,
            1 << 16,
            "test.meta",
            &["test.payload"],
            LAYOUT,
            1,
            &bytes,
        );
        assert!(matches!(result, Err(IoError::UnsupportedVersion { version: 2 })));
    }
}

#[cfg(test)]
mod identity_difference_tests {
    struct Identity {
        snapshot: String,
        design: String,
        digest: String,
    }

    fn identity(snapshot: &str, design: &str, digest: &str) -> Identity {
        Identity { snapshot: snapshot.into(), design: design.into(), digest: digest.into() }
    }

    fn diff(stored: &Identity, other: &Identity) -> Option<&'static str> {
        crate::first_identity_difference!(stored, other, {
            snapshot => "row_snapshot",
            design => "design",
            digest => "identity",
        })
    }

    #[test]
    fn the_first_differing_field_in_listed_order_is_named() {
        let base = identity("s", "d", "h");
        assert_eq!(diff(&base, &identity("s", "d", "h")), None);
        assert_eq!(diff(&base, &identity("s", "d", "x")), Some("identity"));
        assert_eq!(diff(&base, &identity("s", "x", "x")), Some("design"));
        assert_eq!(diff(&base, &identity("x", "x", "x")), Some("row_snapshot"));
    }
}
