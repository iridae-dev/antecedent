//! Facade errors (DESIGN.md §22 `CausalError` shape).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_attribution::AttributionError;
use causal_counterfactual::CounterfactualError;
use causal_discovery::DiscoveryError;
use causal_estimate::EstimationError;
use causal_identify::IdentificationError;
use causal_io::IoError;
use causal_model::ModelError;
use causal_validate::ValidationError;
use thiserror::Error;

/// Analysis pipeline failures — structured sum over domain errors.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum AnalysisError {
    /// Identification failed.
    #[error(transparent)]
    Identify(#[from] IdentificationError),
    /// Estimation failed.
    #[error(transparent)]
    Estimate(#[from] EstimationError),
    /// Validation / refutation failed.
    #[error(transparent)]
    Validate(#[from] ValidationError),
    /// Discovery failed.
    #[error(transparent)]
    Discovery(#[from] DiscoveryError),
    /// Structural / probabilistic model failure.
    #[error(transparent)]
    Model(#[from] ModelError),
    /// Counterfactual evaluation failed.
    #[error(transparent)]
    Counterfactual(#[from] CounterfactualError),
    /// Attribution failed.
    #[error(transparent)]
    Attribution(#[from] AttributionError),
    /// Artifact serialization / deserialization.
    #[error(transparent)]
    Serialization(#[from] IoError),
    /// Logical / physical plan compilation failed.
    #[error("{message}")]
    Compile {
        /// Message.
        message: String,
    },
    /// Memory or other resource refusal.
    #[error("{message}")]
    Resource {
        /// Message.
        message: String,
    },
    /// Graph review incomplete.
    #[error("{message}")]
    ReviewRequired {
        /// Message.
        message: String,
    },
    /// Query / feature unsupported.
    #[error("{message}")]
    Unsupported {
        /// Message.
        message: &'static str,
    },
    /// Missing required builder input.
    #[error("missing required field: {field}")]
    Missing {
        /// Field name.
        field: &'static str,
    },
}

/// Alias matching DESIGN.md §22 naming.
pub type CausalError = AnalysisError;
