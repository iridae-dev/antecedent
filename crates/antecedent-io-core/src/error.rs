//! IO errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use thiserror::Error;

/// Artifact IO errors.
#[derive(Clone, Debug, Eq, PartialEq, Error)]
#[non_exhaustive]
pub enum IoError {
    /// Bad magic bytes.
    #[error("bad artifact magic")]
    BadMagic,
    /// Unsupported container version.
    #[error("unsupported container version {version}")]
    UnsupportedVersion {
        /// Observed version.
        version: u32,
    },
    /// Unsupported artifact format version (major.minor).
    #[error("unsupported artifact format {major}.{minor}")]
    UnsupportedFormat {
        /// Major.
        major: u16,
        /// Minor.
        minor: u16,
    },
    /// CBOR encode/decode failure.
    #[error("cbor error: {0}")]
    Cbor(String),
    /// Checksum mismatch.
    #[error("checksum mismatch for section `{section}`")]
    ChecksumMismatch {
        /// Section id.
        section: String,
    },
    /// Manifest/payload inconsistency.
    #[error("manifest mismatch: {message}")]
    ManifestMismatch {
        /// Explanation.
        message: &'static str,
    },
    /// Payload too large for u32 length prefix.
    #[error("payload too large")]
    TooLarge,
    /// Underlying IO.
    #[error("io error: {0}")]
    Io(String),
    /// Graph/schema conversion.
    #[error("convert error: {0}")]
    Convert(String),
    /// Unknown or unsupported section compression algorithm.
    #[error("unsupported section compression `{algo}`")]
    UnsupportedCompression {
        /// Algorithm name from the manifest.
        algo: String,
    },
    /// Section decompression failure.
    #[error("decompress section `{section}`: {message}")]
    Decompress {
        /// Section id.
        section: String,
        /// Explanation.
        message: String,
    },
    /// Requested a mapped logical view of a compressed section.
    #[error("section `{section}` is compressed; mapped views require uncompressed sections")]
    MappedCompressed {
        /// Section id.
        section: String,
    },
    /// A refusal carrying a registered runtime reason code
    /// (`parity/reason_codes.toml`), rendered `reason=<code>: <message>` like
    /// every other boundary. An estimation refusal crossing an artifact or
    /// prepared-study boundary keeps its code here instead of flattening to a
    /// conversion message.
    #[error("{}{code}: {message}", antecedent_core::reason_code::PREFIX)]
    Refused {
        /// Registered reason code.
        code: &'static str,
        /// What was refused.
        message: String,
    },
    /// A z-transport artifact failed one of its typed consumer checks.
    #[error("z-transport artifact: {0}")]
    ZTransport(#[from] crate::transport_errors::ZTransportArtifactError),
    /// A multi-source mz-transport artifact failed one of its typed consumer checks.
    #[error("mz-transport artifact: {0}")]
    MzTransport(#[from] crate::transport_errors::MzTransportArtifactError),
    /// A mixed-source proof-search artifact failed one of its typed consumer checks.
    #[error("mixed-source artifact: {0}")]
    MixedSource(#[from] crate::transport_errors::MixedSourceArtifactError),
    /// An ADMG conditional transport artifact failed one of its typed consumer checks.
    #[error("admg conditional transport artifact: {0}")]
    AdmgConditional(#[from] crate::transport_errors::AdmgConditionalArtifactError),
    /// A transport scenario artifact failed one of its typed consumer checks.
    #[error("transport scenario artifact: {0}")]
    TransportScenario(#[from] crate::transport_errors::TransportScenarioArtifactError),
    /// A temporal sequence artifact failed one of its typed consumer checks.
    #[error("temporal transport artifact: {0}")]
    TemporalTransport(#[from] crate::transport_errors::TemporalTransportArtifactError),
    /// A learned continuous-outcome transport artifact failed one of its typed consumer checks.
    #[error("learned continuous transport artifact: {0}")]
    LearnedContinuous(#[from] crate::transport_errors::LearnedContinuousArtifactError),
    /// A smoothed dose-response transport artifact failed one of its typed consumer checks.
    #[error("smoothed dose transport artifact: {0}")]
    SmoothedDose(#[from] crate::transport_errors::SmoothedDoseArtifactError),
}

impl IoError {
    /// The registered reason code this error carries, when it has one: a coded
    /// refusal, or an mz-transport artifact failure the registry assigns a code.
    #[must_use]
    pub fn reason_code(&self) -> Option<&'static str> {
        match self {
            Self::Refused { code, .. } => Some(code),
            Self::MzTransport(inner) => inner.refusal().map(|(code, _)| code),
            Self::AdmgConditional(inner) => Some(inner.refusal().0),
            Self::SmoothedDose(inner) => inner.reason_code(),
            _ => None,
        }
    }
}

impl From<antecedent_identify::IdentificationError> for IoError {
    fn from(error: antecedent_identify::IdentificationError) -> Self {
        match error {
            antecedent_identify::IdentificationError::Cancelled
            | antecedent_identify::IdentificationError::Budget { .. } => Self::Refused {
                code: antecedent_core::reason_code!("transport_budget_cancel"),
                message: error.to_string(),
            },
            other => Self::Convert(other.to_string()),
        }
    }
}

/// Wrap any displayable failure as [`IoError::Convert`].
// doc(hidden): shared by antecedent-io conversions.
#[doc(hidden)]
pub fn convert_err(error: impl std::fmt::Display) -> IoError {
    IoError::Convert(error.to_string())
}

/// A `usize` count on the wire. Every supported target has `usize` no wider
/// than `u64`, so a value that fits in memory fits in the word. A conversion
/// that does not is [`IoError::TooLarge`], never a saturated `u64::MAX`.
#[inline]
pub(crate) fn fit_u64(value: usize) -> Result<u64, IoError> {
    u64::try_from(value).map_err(|_| IoError::TooLarge)
}

/// `usize` as `u64` for builders that do not return [`Result`].
///
/// `usize::BITS <= 64` is a compile-time requirement, so the word is the one
/// [`fit_u64`] would return. A wider `usize` does not compile: it is not
/// saturated to `u64::MAX` and not truncated.
#[inline]
#[allow(
    clippy::cast_possible_truncation,
    reason = "usize::BITS <= 64 is asserted; the cast cannot truncate on a target this crate builds for"
)]
pub(crate) const fn lossless_u64(value: usize) -> u64 {
    const {
        assert!(usize::BITS <= 64, "refusing to narrow a usize that does not fit in u64");
    }
    value as u64
}

/// Whether a declared `u64` total exceeds a `usize` byte limit.
///
/// A limit that does not fit in `u64` is larger than every declared total, so
/// the comparison admits it. Truncating the limit would refuse a total the
/// caller allowed.
#[inline]
pub(crate) fn declared_exceeds_usize_limit(declared: Option<u64>, limit: usize) -> bool {
    match u64::try_from(limit) {
        Ok(bound) => declared.is_none_or(|total| total > bound),
        Err(_) => false,
    }
}

impl From<std::io::Error> for IoError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}

#[cfg(test)]
mod count_tests {
    use super::{IoError, declared_exceeds_usize_limit, fit_u64, lossless_u64};

    fn fit_unsigned(value: u128) -> Result<u64, IoError> {
        u64::try_from(value).map_err(|_| IoError::TooLarge)
    }

    #[test]
    fn counts_above_u64_max_are_refused() {
        assert_eq!(fit_unsigned(0).unwrap(), 0);
        assert_eq!(fit_unsigned(u128::from(u64::MAX)).unwrap(), u64::MAX);
        assert_eq!(fit_unsigned(u128::from(u64::MAX) + 1), Err(IoError::TooLarge));
        assert_eq!(fit_u64(0).unwrap(), 0);
        assert_eq!(fit_u64(usize::MAX).unwrap(), u64::try_from(usize::MAX).unwrap());
    }

    #[test]
    fn lossless_word_matches_the_exact_conversion() {
        for value in [0usize, 1, 42, usize::MAX / 2, usize::MAX] {
            assert_eq!(lossless_u64(value), u64::try_from(value).unwrap());
            assert_eq!(fit_u64(value).unwrap(), lossless_u64(value));
        }
    }

    #[test]
    fn a_limit_wider_than_u64_does_not_truncate_to_a_tighter_bound() {
        assert!(!declared_exceeds_usize_limit(Some(1), usize::MAX));
        assert!(declared_exceeds_usize_limit(Some(8), 7));
        assert!(!declared_exceeds_usize_limit(Some(7), 7));
        assert!(declared_exceeds_usize_limit(None, usize::MAX));
    }
}
