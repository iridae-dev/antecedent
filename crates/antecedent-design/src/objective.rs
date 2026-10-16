//! Design objectives.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{ModelId, QueryId};

use crate::decision::DecisionProblemId;
use crate::result::ScoreEvaluation;

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

/// Which scoring channel an objective uses. Callers dispatch on this instead of
/// matching [`DesignObjective`].
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ObjectiveChannel<'a> {
    /// Heuristic graph-channel entropy.
    GraphEntropy,
    /// Static unlock-list identified-mass gain for `query`.
    Identification {
        /// Query handle.
        query: QueryId,
    },
    /// OLS Gram standard-error reduction for `query`.
    EffectWidth {
        /// Query handle.
        query: QueryId,
    },
    /// Reliability-scaled log-likelihood gap over `models`.
    ModelDistinction {
        /// Models to separate.
        models: &'a [ModelId],
    },
    /// Preposterior expected value of sample information for `decision`.
    Decision {
        /// Decision problem handle.
        decision: DecisionProblemId,
    },
}

/// The scoring questions a design objective answers without exposing its match arms.
pub trait ObjectiveScorer {
    /// Channel the ranker, the facade and the Python binding dispatch on.
    fn channel(&self) -> ObjectiveChannel<'_>;

    /// Whether a non-decision score is an exact functional or a Monte Carlo estimate.
    fn static_evaluation(&self) -> ScoreEvaluation;
}

impl ObjectiveScorer for DesignObjective {
    fn channel(&self) -> ObjectiveChannel<'_> {
        match self {
            Self::ReduceGraphEntropy => ObjectiveChannel::GraphEntropy,
            Self::IncreaseIdentificationProbability { query } => {
                ObjectiveChannel::Identification { query: *query }
            }
            Self::ReduceEffectPosteriorWidth { query } => {
                ObjectiveChannel::EffectWidth { query: *query }
            }
            Self::DistinguishModels { models } => {
                ObjectiveChannel::ModelDistinction { models: models.as_ref() }
            }
            Self::ReduceDecisionRegret { decision } => {
                ObjectiveChannel::Decision { decision: *decision }
            }
        }
    }

    fn static_evaluation(&self) -> ScoreEvaluation {
        match self.channel() {
            ObjectiveChannel::GraphEntropy
            | ObjectiveChannel::Identification { .. }
            | ObjectiveChannel::EffectWidth { .. } => ScoreEvaluation::Exact,
            ObjectiveChannel::ModelDistinction { .. } | ObjectiveChannel::Decision { .. } => {
                ScoreEvaluation::MonteCarlo
            }
        }
    }
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

    /// Build an objective from the Python / facade label.
    ///
    /// `eig` is the historical alias for heuristic graph-channel entropy.
    ///
    /// # Errors
    /// An unknown label, or a label that is missing the id it requires.
    pub fn from_label(
        kind: &str,
        query: Option<QueryId>,
        models: Option<&[ModelId]>,
        decision: Option<DecisionProblemId>,
    ) -> Result<Self, String> {
        match kind {
            "reduce_graph_entropy" | "eig" => Ok(Self::ReduceGraphEntropy),
            "increase_identification_probability" | "id_probability" => query
                .map(|query| Self::IncreaseIdentificationProbability { query })
                .ok_or_else(|| "increase_identification_probability requires query_id".to_owned()),
            "reduce_effect_posterior_width" | "effect_width" => query
                .map(|query| Self::ReduceEffectPosteriorWidth { query })
                .ok_or_else(|| "reduce_effect_posterior_width requires query_id".to_owned()),
            "reduce_decision_regret" | "decision_regret" => decision
                .map(|decision| Self::ReduceDecisionRegret { decision })
                .ok_or_else(|| "reduce_decision_regret requires decision_id".to_owned()),
            "distinguish_models" => {
                let models =
                    models.ok_or_else(|| "distinguish_models requires model_ids".to_owned())?;
                if models.len() < 2 {
                    return Err("distinguish_models needs ≥2 model_ids".to_owned());
                }
                Ok(Self::DistinguishModels { models: Arc::from(models) })
            }
            other => Err(format!("unknown objective `{other}`")),
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
