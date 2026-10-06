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
    ZTransport(#[from] crate::z_transport_artifact::ZTransportArtifactError),
    /// A multi-source mz-transport artifact failed one of its typed consumer checks.
    #[error("mz-transport artifact: {0}")]
    MzTransport(#[from] crate::mz_transport_artifact::MzTransportArtifactError),
    /// A mixed-source proof-search artifact failed one of its typed consumer checks.
    #[error("mixed-source artifact: {0}")]
    MixedSource(#[from] crate::mixed_source_artifact::MixedSourceArtifactError),
    /// An ADMG conditional transport artifact failed one of its typed consumer checks.
    #[error("admg conditional transport artifact: {0}")]
    AdmgConditional(
        #[from] crate::admg_conditional_transport_artifact::AdmgConditionalArtifactError,
    ),
    /// A transport scenario artifact failed one of its typed consumer checks.
    #[error("transport scenario artifact: {0}")]
    TransportScenario(#[from] crate::transport_scenario_artifact::TransportScenarioArtifactError),
    /// A temporal sequence artifact failed one of its typed consumer checks.
    #[error("temporal transport artifact: {0}")]
    TemporalTransport(#[from] crate::temporal_transport_artifact::TemporalTransportArtifactError),
    /// A learned continuous-outcome transport artifact failed one of its typed consumer checks.
    #[error("learned continuous transport artifact: {0}")]
    LearnedContinuous(#[from] crate::learned_continuous_artifact::LearnedContinuousArtifactError),
    /// A smoothed dose-response transport artifact failed one of its typed consumer checks.
    #[error("smoothed dose transport artifact: {0}")]
    SmoothedDose(#[from] crate::smoothed_dose_artifact::SmoothedDoseArtifactError),
}

impl IoError {
    /// The registered reason code this error carries, when it has one: a coded
    /// refusal, or an mz-transport artifact failure the X1 record assigns a code.
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

impl From<antecedent_estimate::EstimationError> for IoError {
    fn from(error: antecedent_estimate::EstimationError) -> Self {
        match error {
            antecedent_estimate::EstimationError::Refused { code, message } => {
                Self::Refused { code, message }
            }
            other => Self::Convert(other.to_string()),
        }
    }
}

/// Wrap any displayable failure as [`IoError::Convert`].
pub(crate) fn convert_err(error: impl std::fmt::Display) -> IoError {
    IoError::Convert(error.to_string())
}

impl From<std::io::Error> for IoError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}
