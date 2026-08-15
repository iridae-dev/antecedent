//! Design objectives.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ModelId, QueryId};

use crate::decision::DecisionProblemId;

/// Objective maximized (or regret minimized) by candidate ranking.
///
/// Public names are historical. [`Self::implemented_functional`] is the mathematics
/// actually scored; `ReduceGraphEntropy` is not expected information gain under a
/// likelihood, `ReduceEffectPosteriorWidth` is OLS Gram SE reduction, and
/// `IncreaseIdentificationProbability` applies static unlock lists.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum DesignObjective {
    /// Heuristic entropy drop under a discrete observation channel
    /// `reliability = 1 − exp(−c · k)` in the number of measured variables.
    /// Not EIG under `p(y | G, design)`.
    ReduceGraphEntropy,
    /// Static unlock-list identified-mass gain, not `P(identify | data, design)`.
    IncreaseIdentificationProbability {
        /// Query handle.
        query: QueryId,
    },
    /// OLS Gram SE reduction with fixed σ² (classical design optimality), not Bayesian
    /// posterior width.
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

impl DesignObjective {
    /// Mathematics actually scored for this objective (not the historical public name).
    #[must_use]
    pub const fn implemented_functional(&self) -> &'static str {
        match self {
            Self::ReduceGraphEntropy => "heuristic_graph_channel_entropy",
            Self::IncreaseIdentificationProbability { .. } => "static_unlock_identified_mass",
            Self::ReduceEffectPosteriorWidth { .. } => "ols_gram_se_reduction",
            Self::ReduceDecisionRegret { .. } => "monte_carlo_decision_regret",
            Self::DistinguishModels { .. } => "expected_logscore_gap",
        }
    }

    /// Whether the scored functional is the decision-theoretic object the public name suggests.
    #[must_use]
    pub const fn is_exact_information_functional(&self) -> bool {
        matches!(self, Self::ReduceDecisionRegret { .. } | Self::DistinguishModels { .. })
    }
}
