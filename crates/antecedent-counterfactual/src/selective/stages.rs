//! Stage vocabulary, observed work counters and semantic stream identities.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::CanonicalWriter;

/// A stage of cross-world execution, in dependency order.
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub enum ExecutionStage {
    /// The mechanisms the worlds are evaluated under.
    MechanismFitting,
    /// The exogenous posterior given each unit's factual evidence.
    FactualAbduction,
    /// The shared exogenous states every world reads.
    ExogenousDraws,
    /// Each named world evaluated under the shared exogenous states.
    ActionPrediction,
    /// The requested quantity from the coupled worlds.
    SummaryDecision,
}

impl ExecutionStage {
    /// Every stage in dependency order.
    pub const ALL: [ExecutionStage; 5] = [
        Self::MechanismFitting,
        Self::FactualAbduction,
        Self::ExogenousDraws,
        Self::ActionPrediction,
        Self::SummaryDecision,
    ];

    /// Stable `snake_case` name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::MechanismFitting => "mechanism_fitting",
            Self::FactualAbduction => "factual_abduction",
            Self::ExogenousDraws => "exogenous_draws",
            Self::ActionPrediction => "action_prediction",
            Self::SummaryDecision => "summary_decision",
        }
    }

    /// The canonical dependency names the stage's identity covers.
    #[must_use]
    pub const fn dependencies(self) -> &'static [&'static str] {
        match self {
            Self::MechanismFitting => &["population", "history", "graph", "mechanism_parameters"],
            Self::FactualAbduction => {
                &["mechanism_fitting", "factual_evidence", "noise_law", "inference_method"]
            }
            Self::ExogenousDraws => &["factual_abduction", "stream_coordinates"],
            Self::ActionPrediction => &["exogenous_draws", "world_definitions", "routes"],
            Self::SummaryDecision => &["action_prediction", "quantity", "coupling_declaration"],
        }
    }
}

/// How a stage's output was obtained in one execution.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum StageStatus {
    /// Supplied by declaration; nothing was fitted.
    Declared,
    /// Computed in this execution.
    Computed,
    /// Taken from a retained output whose identity matched and whose content
    /// digest was re-validated.
    Reused,
}

/// One stage's identity and status in one execution.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct StageReceipt {
    /// The stage.
    pub stage: ExecutionStage,
    /// Canonical dependency digest (64 lowercase hex characters).
    pub identity: String,
    /// How the output was obtained.
    pub status: StageStatus,
}

/// Work observed at actual algorithm boundaries.
///
/// A reused abduction increments only `abductions_reused`; a cache hit is
/// never counted as work.
#[derive(Clone, Copy, Debug, Default, Eq, PartialEq)]
pub struct WorkCounters {
    /// Mechanisms fitted from data (a declared mechanism is not a fit).
    pub mechanism_fits: u64,
    /// Units abducted (an exogenous posterior built from evidence).
    pub abductions: u64,
    /// Joint exogenous states enumerated by abduction.
    pub abduction_states_enumerated: u64,
    /// Units whose retained abduction was reused.
    pub abductions_reused: u64,
    /// Random exogenous or posterior draws taken (zero for exact enumeration).
    pub shared_draws: u64,
    /// `(unit, world)` evaluations completed.
    pub world_predictions: u64,
    /// Posterior states evaluated across all worlds.
    pub posterior_states_predicted: u64,
    /// Summaries computed.
    pub summaries: u64,
}

impl WorkCounters {
    /// The work done since `earlier` (saturating field by field).
    #[must_use]
    pub fn since(self, earlier: Self) -> Self {
        Self {
            mechanism_fits: self.mechanism_fits.saturating_sub(earlier.mechanism_fits),
            abductions: self.abductions.saturating_sub(earlier.abductions),
            abduction_states_enumerated: self
                .abduction_states_enumerated
                .saturating_sub(earlier.abduction_states_enumerated),
            abductions_reused: self.abductions_reused.saturating_sub(earlier.abductions_reused),
            shared_draws: self.shared_draws.saturating_sub(earlier.shared_draws),
            world_predictions: self.world_predictions.saturating_sub(earlier.world_predictions),
            posterior_states_predicted: self
                .posterior_states_predicted
                .saturating_sub(earlier.posterior_states_predicted),
            summaries: self.summaries.saturating_sub(earlier.summaries),
        }
    }
}

/// Purpose coordinate of the exogenous posterior state of a unit's node.
pub const PURPOSE_POSTERIOR_STATE: &str = "exogenous_posterior_state";

/// Stream identity by semantic coordinates `(unit, node, purpose)`.
///
/// The world is deliberately **not** a coordinate: every world of a unit reads
/// the same exogenous term, which is what couples them. Nothing else enters, so
/// another node's size, another world's definition or the number of worlds
/// never shifts this identity. Fold and replicate coordinates do not exist for
/// exact enumeration; a sampled route would add them as further coordinates.
#[must_use]
pub fn stream_identity(unit: &str, node: &str, purpose: &str) -> String {
    let mut w = CanonicalWriter::new("antecedent.counterfactual.selective.stream.v1");
    w.text(unit).text(node).text(purpose);
    w.finish_hex()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn the_stream_identity_is_unit_node_purpose_and_nothing_else() {
        let a = stream_identity("u1", "m", PURPOSE_POSTERIOR_STATE);
        assert_eq!(a, stream_identity("u1", "m", PURPOSE_POSTERIOR_STATE));
        assert_ne!(a, stream_identity("u2", "m", PURPOSE_POSTERIOR_STATE));
        assert_ne!(a, stream_identity("u1", "y", PURPOSE_POSTERIOR_STATE));
        assert_ne!(a, stream_identity("u1", "m", "other"));
        // Length-delimited: a boundary shift between coordinates changes it.
        assert_ne!(stream_identity("u1m", "", "p"), stream_identity("u1", "m", "p"));
    }

    #[test]
    fn work_since_is_a_saturating_difference() {
        let before = WorkCounters { abductions: 2, ..WorkCounters::default() };
        let after = WorkCounters { abductions: 5, world_predictions: 4, ..before };
        let delta = after.since(before);
        assert_eq!((delta.abductions, delta.world_predictions), (3, 4));
        assert_eq!(before.since(after).abductions, 0);
    }
}
