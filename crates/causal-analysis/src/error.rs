//! Facade errors.
use core::fmt;

/// Analysis pipeline failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum AnalysisError {
    /// Identification failed.
    Identify(String),
    /// Estimation failed.
    Estimate(String),
    /// Validation / refutation failed.
    Validate(String),
    /// Query unsupported in Phase 1.
    Unsupported {
        /// Message.
        message: &'static str,
    },
    /// Missing required builder input.
    Missing {
        /// Field name.
        field: &'static str,
    },
}

impl fmt::Display for AnalysisError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Identify(m) | Self::Estimate(m) | Self::Validate(m) => write!(f, "{m}"),
            Self::Unsupported { message } => write!(f, "{message}"),
            Self::Missing { field } => write!(f, "missing required field: {field}"),
        }
    }
}

impl std::error::Error for AnalysisError {}
