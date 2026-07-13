//! Estimation errors.
use core::fmt;

/// Estimation failures.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum EstimationError {
    /// Data/schema issue.
    Data(String),
    /// Stats backend.
    Stats(String),
    /// Missing overlap override when required.
    Overlap {
        /// Message.
        message: &'static str,
    },
    /// Incompatible estimand.
    IncompatibleEstimand {
        /// Message.
        message: &'static str,
    },
    /// Query options unsupported by this estimator.
    UnsupportedQuery(String),
}

impl fmt::Display for EstimationError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::Data(m) | Self::Stats(m) | Self::UnsupportedQuery(m) => write!(f, "{m}"),
            Self::Overlap { message } | Self::IncompatibleEstimand { message } => {
                write!(f, "{message}")
            }
        }
    }
}

impl std::error::Error for EstimationError {}
