//! Facade errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_attribution::AttributionError;
use causal_core::SchemaError;
use causal_counterfactual::CounterfactualError;
use causal_data::DataError;
use causal_design::DesignError;
use causal_discovery::DiscoveryError;
use causal_estimate::EstimationError;
use causal_graph::GraphError;
use causal_identify::IdentificationError;
use causal_io::IoError;
use causal_model::ModelError;
use causal_state::StateError;
use causal_validate::ValidationError;
use thiserror::Error;

/// Pipeline and facade failures — structured sum over domain errors.
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum CausalError {
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
    /// Tabular / time-series data construction or lookup.
    #[error(transparent)]
    Data(#[from] DataError),
    /// Graph construction or validation.
    #[error(transparent)]
    Graph(#[from] GraphError),
    /// Experiment / measurement design evaluation.
    #[error(transparent)]
    Design(#[from] DesignError),
    /// Incremental causal-state update.
    #[error(transparent)]
    State(#[from] StateError),
    /// Schema construction or name lookup at an API boundary.
    #[error(transparent)]
    Schema(#[from] SchemaError),
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
    /// Graph review incomplete (structured for facade UX).
    #[error("{message}")]
    ReviewRequired {
        /// Review kind: `temporal_pag`, `temporal_cpdag`, `static_pag`, `static_cpdag`, `temporal_dag`, `generic`.
        kind: String,
        /// Discovery / supply algorithm id when known.
        algorithm: Option<String>,
        /// Count of pending / ambiguous marks blocking estimation.
        pending_edge_count: usize,
        /// Human-readable message.
        message: String,
        /// Next-step hint for callers.
        hint: String,
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
    /// Cooperative cancellation before a usable point estimate was available.
    #[error("cancelled during {stage}")]
    Cancelled {
        /// Pipeline stage where cancellation was observed.
        stage: &'static str,
    },
}

impl CausalError {
    /// Build a structured review-required error.
    #[must_use]
    pub fn review_required(
        kind: impl Into<String>,
        algorithm: Option<impl Into<String>>,
        pending_edge_count: usize,
        message: impl Into<String>,
        hint: impl Into<String>,
    ) -> Self {
        Self::ReviewRequired {
            kind: kind.into(),
            algorithm: algorithm.map(Into::into),
            pending_edge_count,
            message: message.into(),
            hint: hint.into(),
        }
    }

    /// Convenience when only a message is available (generic review).
    #[must_use]
    pub fn review_required_msg(message: impl Into<String>) -> Self {
        let message = message.into();
        Self::review_required(
            "generic",
            None::<String>,
            0,
            message,
            "complete graph review (finish_*_review) or supply a fully oriented graph",
        )
    }
}

/// Deprecated alias for [`CausalError`].
#[deprecated(note = "renamed to CausalError")]
pub type AnalysisError = CausalError;
