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
    /// Logical / physical plan compilation failed.
    Compile {
        /// Message.
        message: String,
    },
    /// Memory or other resource refusal.
    Resource {
        /// Message.
        message: String,
    },
    /// Graph review incomplete.
    ReviewRequired {
        /// Message.
        message: String,
    },
    /// Query / feature unsupported.
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
            Self::Compile { message }
            | Self::Resource { message }
            | Self::ReviewRequired { message } => write!(f, "{message}"),
            Self::Unsupported { message } => write!(f, "{message}"),
            Self::Missing { field } => write!(f, "missing required field: {field}"),
        }
    }
}

impl std::error::Error for AnalysisError {}
