//! Discovery errors.
use core::fmt;

/// Discovery failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DiscoveryError {
    /// Data / sample preparation.
    Data(String),
    /// Stats / CI failure.
    Stats(String),
    /// Unsupported configuration.
    Unsupported {
        /// Message.
        message: &'static str,
    },
}

impl fmt::Display for DiscoveryError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Data(m) | Self::Stats(m) => write!(f, "{m}"),
            Self::Unsupported { message } => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for DiscoveryError {}
