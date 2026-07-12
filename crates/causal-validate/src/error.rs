//! Validation errors.
use core::fmt;

/// Validation / refutation failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum ValidationError {
    /// Data transformation failed.
    Data(String),
    /// Estimation failed inside a refuter.
    Estimation(String),
    /// Refuter not applicable to the problem.
    NotApplicable {
        /// Reason.
        message: &'static str,
    },
}

impl fmt::Display for ValidationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Data(m) | Self::Estimation(m) => write!(f, "{m}"),
            Self::NotApplicable { message } => write!(f, "{message}"),
        }
    }
}

impl std::error::Error for ValidationError {}
