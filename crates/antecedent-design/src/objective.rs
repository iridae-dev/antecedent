//! Design objectives.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ModelId, QueryId};

use crate::decision::DecisionProblemId;

/// Objective maximized by candidate ranking.
///
/// Public names are historical. [`Self::implemented_functional`] is the mathematics
/// actually scored; `ReduceGraphEntropy` is not expected information gain under a
/// likelihood, `ReduceEffectPosteriorWidth` is OLS Gram SE reduction,
/// `IncreaseIdentificationProbability` applies static unlock lists, and
/// `DistinguishModels` is a reliability-scaled log-likelihood gap.
/// `ReduceDecisionRegret` is the decision-theoretic object its name states.
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
    /// Expected reduction in decision regret from the candidate's data: the
    /// preposterior expected value of sample information
    /// `EVSI = E_y[max_{a∈F} E[U | y]] − max_{a∈F} E[U]` for a registered decision
    /// problem, prior and signal (Raiffa & Schlaifer 1961, ch. 1 and 4; see
    /// [`crate::preposterior`]). Exact for finite-support signals and the conjugate
    /// normal model, a Monte Carlo estimate otherwise
    /// ([`crate::ScoreEvaluation`]).
    ReduceDecisionRegret {
        /// Decision problem handle.
        decision: DecisionProblemId,
    },
    /// Reliability heuristic times the absolute pairwise log-likelihood gap at one
    /// sampled posterior draw (averaged over ranker draws). Not the Box–Hill expected
    /// divergence between models.
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
            Self::ReduceDecisionRegret { .. } => {
                "preposterior_expected_value_of_sample_information"
            }
            Self::DistinguishModels { .. } => "heuristic_reliability_scaled_loglik_gap",
        }
    }

    /// Whether the scored functional is the decision-theoretic object the public name
    /// states, rather than a heuristic proxy for it.
    ///
    /// Only `ReduceDecisionRegret` qualifies. Whether a given score is that
    /// functional's exact value or a Monte Carlo estimate of it is reported per
    /// candidate by [`crate::RankedCandidate::evaluation`].
    #[must_use]
    pub const fn is_exact_information_functional(&self) -> bool {
        matches!(self, Self::ReduceDecisionRegret { .. })
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn only_decision_regret_claims_the_named_functional() {
        let objectives = [
            DesignObjective::ReduceGraphEntropy,
            DesignObjective::IncreaseIdentificationProbability { query: QueryId::from_raw(0) },
            DesignObjective::ReduceEffectPosteriorWidth { query: QueryId::from_raw(0) },
            DesignObjective::DistinguishModels {
                models: Arc::from([ModelId::from_raw(0), ModelId::from_raw(1)]),
            },
        ];
        for objective in &objectives {
            assert!(!objective.is_exact_information_functional(), "{objective:?}");
        }
        let regret =
            DesignObjective::ReduceDecisionRegret { decision: DecisionProblemId::from_raw(0) };
        assert!(regret.is_exact_information_functional());
        assert_eq!(
            regret.implemented_functional(),
            "preposterior_expected_value_of_sample_information"
        );
    }
}
