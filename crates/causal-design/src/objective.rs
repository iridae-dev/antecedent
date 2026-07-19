//! Design objectives.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use causal_core::{ModelId, QueryId};

use crate::decision::DecisionProblemId;

/// Objective maximized (or regret minimized) by candidate ranking.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DesignObjective {
    /// Reduce entropy of the discrete graph posterior.
    ReduceGraphEntropy,
    /// Increase identification probability for a registered query.
    IncreaseIdentificationProbability {
        /// Query handle.
        query: QueryId,
    },
    /// Reduce posterior width of an effect estimate for a query.
    ReduceEffectPosteriorWidth {
        /// Query handle.
        query: QueryId,
    },
    /// Reduce decision regret for a registered decision problem.
    ReduceDecisionRegret {
        /// Decision problem handle.
        decision: DecisionProblemId,
    },
    /// Distinguish among registered models (expected log-score gap).
    DistinguishModels {
        /// Models to separate.
        models: Arc<[ModelId]>,
    },
}
