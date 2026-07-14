//! Attribution errors.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use thiserror::Error;

/// Attribution errors (DESIGN.md §17).
#[derive(Clone, Debug, Error, Eq, PartialEq)]
pub enum AttributionError {
    /// Model / data / query message.
    #[error("{0}")]
    Message(String),
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
