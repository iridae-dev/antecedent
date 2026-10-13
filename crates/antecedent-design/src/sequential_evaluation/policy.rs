//! Observed histories and caller-supplied contingent policies.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;
use std::fmt;

use antecedent_core::CanonicalWriter;

use super::{
    MAX_LABEL_BYTES, MAX_POLICY_ENTRIES, PROBABILITY_TOLERANCE,
    SEQUENTIAL_EVALUATION_IDENTITY_VERSION, SequentialResult,
};

/// One completed decision cycle: the action taken and the observation it
/// produced (the symbol index within the action's observation channel; `0` for
/// an action with no channel).
#[derive(Clone, Copy, Debug, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct HistoryStep {
    /// Action index.
    pub action: usize,
    /// Observation symbol index.
    pub observation: usize,
}

/// An observed history: the information set a decision may depend on.
#[derive(Clone, Debug, Default, Eq, Hash, Ord, PartialEq, PartialOrd)]
pub struct HistoryKey(Vec<HistoryStep>);

impl HistoryKey {
    /// The empty history, at the first decision time.
    #[must_use]
    pub const fn root() -> Self {
        Self(Vec::new())
    }

    /// A history from its completed cycles.
    #[must_use]
    pub fn from_steps(steps: impl IntoIterator<Item = HistoryStep>) -> Self {
        Self(steps.into_iter().collect())
    }

    /// The completed cycles, oldest first.
    #[must_use]
    pub fn steps(&self) -> &[HistoryStep] {
        &self.0
    }

    /// Number of completed cycles, which is also the decision time.
    #[must_use]
    pub fn len(&self) -> usize {
        self.0.len()
    }

    /// Whether this is the empty history.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.0.is_empty()
    }

    /// This history followed by one more completed cycle.
    #[must_use]
    pub fn extended(&self, action: usize, observation: usize) -> Self {
        let mut steps = self.0.clone();
        steps.push(HistoryStep { action, observation });
        Self(steps)
    }

    pub(crate) fn write(&self, writer: &mut CanonicalWriter) {
        writer.word(self.0.len() as u64);
        for step in &self.0 {
            writer.word(step.action as u64).word(step.observation as u64);
        }
    }
}

impl fmt::Display for HistoryKey {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.0.is_empty() {
            return f.write_str("<root>");
        }
        for (index, step) in self.0.iter().enumerate() {
            if index > 0 {
                f.write_str("/")?;
            }
            write!(f, "a{}:o{}", step.action, step.observation)?;
        }
        Ok(())
    }
}

/// A caller-supplied contingent policy: a table from observed histories to a
/// distribution over actions. It is data, versioned by a caller label, and
/// never changes during an evaluation.
#[derive(Clone, Debug, PartialEq)]
pub struct ContingentPolicy {
    version: String,
    decisions: BTreeMap<HistoryKey, Vec<(usize, f64)>>,
}

impl ContingentPolicy {
    /// An empty policy with a version label.
    #[must_use]
    pub fn new(version: impl Into<String>) -> Self {
        Self { version: version.into(), decisions: BTreeMap::new() }
    }

    /// The policy version label.
    #[must_use]
    pub fn version(&self) -> &str {
        &self.version
    }

    /// Take `action` for certain in `history`.
    #[must_use]
    pub fn decide(mut self, history: HistoryKey, action: usize) -> Self {
        self.decisions.insert(history, vec![(action, 1.0)]);
        self
    }

    /// Take actions at random in `history` with the given probabilities.
    ///
    /// # Errors
    /// An empty, non-finite, negative or non-normalized distribution, or an
    /// action named twice.
    pub fn decide_stochastic(
        mut self,
        history: HistoryKey,
        distribution: Vec<(usize, f64)>,
    ) -> SequentialResult<Self> {
        let mut entries: Vec<(usize, f64)> =
            distribution.into_iter().filter(|(_, p)| *p != 0.0).collect();
        if entries.is_empty() {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.policy_distribution",
                format!(
                    "the decision for history {history} has no action with positive probability"
                ),
                "give the history at least one action with positive probability",
            ));
        }
        entries.sort_by_key(|(action, _)| *action);
        if entries.windows(2).any(|pair| pair[0].0 == pair[1].0) {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.policy_distribution",
                format!("the decision for history {history} names an action twice"),
                "list each action once",
            ));
        }
        if entries.iter().any(|(_, p)| !(p.is_finite() && *p > 0.0 && *p <= 1.0)) {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.policy_distribution",
                format!("the decision for history {history} has a probability outside (0, 1]"),
                "use probabilities in (0, 1] that sum to one",
            ));
        }
        let total: f64 = entries.iter().map(|(_, p)| *p).sum();
        if (total - 1.0).abs() > PROBABILITY_TOLERANCE {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.policy_distribution",
                format!("the decision for history {history} sums to {total}, not 1"),
                "normalize the action probabilities",
            ));
        }
        self.decisions.insert(history, entries);
        Ok(self)
    }

    /// The action distribution declared for `history`, sorted by action.
    #[must_use]
    pub fn distribution(&self, history: &HistoryKey) -> Option<&[(usize, f64)]> {
        self.decisions.get(history).map(Vec::as_slice)
    }

    /// Probability the policy takes `action` in `history`; `None` when the
    /// history has no entry.
    #[must_use]
    pub fn probability(&self, history: &HistoryKey, action: usize) -> Option<f64> {
        self.distribution(history)
            .map(|entries| entries.iter().find(|(a, _)| *a == action).map_or(0.0, |(_, p)| *p))
    }

    /// Every declared decision, in canonical history order.
    pub fn entries(&self) -> impl Iterator<Item = (&HistoryKey, &[(usize, f64)])> {
        self.decisions.iter().map(|(key, dist)| (key, dist.as_slice()))
    }

    /// Number of declared histories.
    #[must_use]
    pub fn len(&self) -> usize {
        self.decisions.len()
    }

    /// Whether the policy declares no history.
    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.decisions.is_empty()
    }

    /// Check the table's own bounds. Whether it covers every reachable
    /// history is decided against a model, during evaluation.
    ///
    /// # Errors
    /// A blank or oversized version, or more entries than the bound.
    pub fn validate(&self) -> SequentialResult<()> {
        if self.version.trim().is_empty() || self.version.len() > MAX_LABEL_BYTES {
            return Err(refusal!(
                "invalid_argument",
                "sequential_evaluation.policy_version",
                "a policy needs a non-blank version label of bounded length",
                "label the policy so its receipts name the version evaluated",
            ));
        }
        if self.decisions.len() > MAX_POLICY_ENTRIES {
            return Err(super::limit_exceeded("the policy table", MAX_POLICY_ENTRIES as u64));
        }
        Ok(())
    }

    /// Canonical identity: version, then every decision in history order with
    /// action probabilities as bit patterns.
    #[must_use]
    pub fn identity_hex(&self) -> String {
        let mut writer = CanonicalWriter::new("antecedent.sequential_evaluation.policy");
        writer.word(SEQUENTIAL_EVALUATION_IDENTITY_VERSION).text(&self.version);
        writer.word(self.decisions.len() as u64);
        for (history, dist) in &self.decisions {
            history.write(&mut writer);
            writer.word(dist.len() as u64);
            for (action, probability) in dist {
                writer.word(*action as u64).real(*probability);
            }
        }
        writer.finish_hex()
    }
}

/// What a caller offers as the policy of an evaluation.
///
/// Only a fixed [`ContingentPolicy`] is evaluated. The other variants exist so
/// the closed routes are reachable, executed refusals rather than absences.
#[derive(Clone, Debug, PartialEq)]
pub enum PolicySpec {
    /// A fixed contingent policy: evaluated.
    Contingent(ContingentPolicy),
    /// A static table of per-option scores with weights. It has no
    /// information sets, histories or observations, so it is not a sequential
    /// policy; reweighting scores is not sequential evaluation.
    ReweightedStaticScores {
        /// Per-option scores.
        scores: Vec<f64>,
        /// Per-option weights.
        weights: Vec<f64>,
    },
    /// A policy that is learned or changed from the data being evaluated.
    /// Learning is not evaluation.
    LearnedFromEvaluationData {
        /// Label of the learning rule.
        rule: String,
    },
}

impl PolicySpec {
    /// The fixed policy to evaluate.
    ///
    /// # Errors
    /// A static score table or a learning rule: the typed route refusal.
    pub fn into_contingent(self) -> SequentialResult<ContingentPolicy> {
        match self {
            Self::Contingent(policy) => Ok(policy),
            Self::ReweightedStaticScores { scores, weights } => Err(refusal!(
                "route_not_supported",
                "sequential_evaluation.static_score_table",
                format!(
                    "a table of {} scores with {} weights has no information sets, observation histories or decision times, so reweighting it is not sequential evaluation",
                    scores.len(),
                    weights.len()
                ),
                "supply a ContingentPolicy: a table from observed histories to actions over the declared horizon",
            )),
            Self::LearnedFromEvaluationData { rule } => Err(refusal!(
                "route_not_supported",
                "sequential_evaluation.policy_change_refused",
                format!(
                    "the policy is produced by the learning rule `{rule}` from the data under evaluation; changing a policy inside an evaluation is policy learning, not re-evaluation of a fixed policy"
                ),
                "fix the policy as a new version and evaluate it, against fresh or held-out observations when its value is to be estimated from data",
            )),
        }
    }
}
