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
}

impl From<std::io::Error> for IoError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}
