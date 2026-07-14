//! IO errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use core::fmt;

/// Artifact IO errors.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum IoError {
    /// Bad magic bytes.
    BadMagic,
    /// Unsupported container version.
    UnsupportedVersion {
        /// Observed version.
        version: u32,
    },
    /// Unsupported artifact format version (major.minor).
    UnsupportedFormat {
        /// Major.
        major: u16,
        /// Minor.
        minor: u16,
    },
    /// CBOR encode/decode failure.
    Cbor(String),
    /// Checksum mismatch.
    ChecksumMismatch {
        /// Section id.
        section: String,
    },
    /// Manifest/payload inconsistency.
    ManifestMismatch {
        /// Explanation.
        message: &'static str,
    },
    /// Payload too large for u32 length prefix.
    TooLarge,
    /// Underlying IO.
    Io(String),
    /// Graph/schema conversion.
    Convert(String),
}

impl fmt::Display for IoError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::BadMagic => write!(f, "bad artifact magic"),
            Self::UnsupportedVersion { version } => {
                write!(f, "unsupported container version {version}")
            }
            Self::UnsupportedFormat { major, minor } => {
                write!(f, "unsupported artifact format {major}.{minor}")
            }
            Self::Cbor(msg) => write!(f, "cbor error: {msg}"),
            Self::ChecksumMismatch { section } => {
                write!(f, "checksum mismatch for section `{section}`")
            }
            Self::ManifestMismatch { message } => write!(f, "manifest mismatch: {message}"),
            Self::TooLarge => write!(f, "payload too large"),
            Self::Io(msg) => write!(f, "io error: {msg}"),
            Self::Convert(msg) => write!(f, "convert error: {msg}"),
        }
    }
}

impl std::error::Error for IoError {}

impl From<std::io::Error> for IoError {
    fn from(value: std::io::Error) -> Self {
        Self::Io(value.to_string())
    }
}
