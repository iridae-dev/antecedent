//! Attribution errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use causal_core::VariableId;
use thiserror::Error;

/// Attribution errors (DESIGN.md §17).
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum AttributionError {
    /// Query / component / allocation combination not supported on this path.
    #[error("{message}")]
    Unsupported {
        /// Explanation.
        message: &'static str,
    },
    /// Required variable absent from model or data.
    #[error("{kind} {id} missing")]
    MissingVariable {
        /// Role (`outcome`, `source`, `target`, …).
        kind: &'static str,
        /// Variable id.
        id: VariableId,
    },
    /// Required model artifact missing (gather plan, edge coeff, …).
    #[error("{0}")]
    MissingArtifact(&'static str),
    /// Empty or out-of-range population / contribution inputs.
    #[error("{message}")]
    InvalidInput {
        /// Explanation.
        message: &'static str,
    },
    /// Hard size limit exceeded.
    #[error("{kind} count {requested} exceeds max={max}")]
    SizeLimit {
        /// What was limited (units, components, …).
        kind: &'static str,
        /// Requested size.
        requested: usize,
        /// Configured maximum.
        max: usize,
    },
    /// Exact Shapley rejected without override.
    #[error(
        "exact Shapley rejected for {n_components} components (limit {max}); enable allow_exact_override or use approximation"
    )]
    ExactShapleyRejected {
        /// Component count.
        n_components: usize,
        /// Configured max.
        max: usize,
    },
    /// Cache policy / budget failure.
    #[error("cache error: {message}")]
    Cache {
        /// Context.
        message: String,
    },
    /// Compute budget exhausted.
    #[error("compute budget exhausted: {message}")]
    Budget {
        /// Context.
        message: String,
    },
    /// Ad-hoc detail that does not fit a structured variant (prefer structured).
    #[error("{0}")]
    Message(String),
    /// Passthrough from causal-model.
    #[error(transparent)]
    Model(#[from] causal_model::ModelError),
    /// Passthrough from causal-data.
    #[error(transparent)]
    Data(#[from] causal_data::DataError),
    /// Passthrough from causal-core query validation.
    #[error(transparent)]
    Query(#[from] causal_core::QueryError),
    /// Passthrough from counterfactual engine.
    #[error(transparent)]
    Counterfactual(#[from] causal_counterfactual::CounterfactualError),
    /// Passthrough from causal-stats.
    #[error(transparent)]
    Stats(#[from] causal_stats::StatsError),
    /// Passthrough from causal-graph.
    #[error(transparent)]
    Graph(#[from] causal_graph::GraphError),
    /// Passthrough from causal-prob.
    #[error(transparent)]
    Prob(#[from] causal_prob::ProbError),
}

impl AttributionError {
    /// Unsupported path / combination.
    #[must_use]
    pub const fn unsupported(message: &'static str) -> Self {
        Self::Unsupported { message }
    }

    /// Missing variable by role.
    #[must_use]
    pub const fn missing_var(kind: &'static str, id: VariableId) -> Self {
        Self::MissingVariable { kind, id }
    }

    /// Invalid empty / out-of-range input.
    #[must_use]
    pub const fn invalid_input(message: &'static str) -> Self {
        Self::InvalidInput { message }
    }
}
