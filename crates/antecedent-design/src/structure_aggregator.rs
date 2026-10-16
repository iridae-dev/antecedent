//! One aggregation of a structural policy over structures.
//!
//! `decision_structural`, `decision_robustness` and `decision_adapters` share
//! this sign table, policy dispatch and refusal text. A larger criterion value
//! is better except for loss, regret and minimax over an identified set.
//! `MaximinOverStructures` and `MinimaxOverIdentifiedSet` force a worst case;
//! otherwise the declared policy decides. Unresolved structures and unresolved
//! probability mass are reported with the text below.

use crate::decision_contract::{DecisionCriterion, StructuralPolicy};

/// How structures are combined, after the criterion override.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub(crate) enum AggregateKind {
    /// An action must be uniquely best in every structure.
    Invariant,
    /// Rank by the worst case over structures.
    WorstCase,
    /// Weight by genuine structure probabilities.
    Bayes,
    /// Return each structure's answer without choosing.
    Report,
}

/// What aggregation can claim, in action-index form where a choice is made.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum AggregateDecision {
    /// Per-structure answers only.
    Report,
    /// The existing refusal text; the mass or structure is not dropped.
    Insufficient(&'static str),
    /// No candidate remains.
    NoCandidate,
    /// A support rule removed the structures or the mass the rule would have used.
    SupportWithheld,
    /// The structures disagree; each structure's leaders are listed.
    Disagree(Vec<(String, Vec<String>)>),
    /// A unique or tied choice.
    Choice(AggregateChoice),
}

/// A choice aggregation can name.
#[derive(Clone, Debug, PartialEq)]
pub(crate) enum AggregateChoice {
    /// Uniquely best in every structure that was checked.
    Invariant(String),
    /// Best worst case.
    WorstCase(String),
    /// Best probability-weighted value over the evaluated mass.
    Bayes {
        /// Chosen action.
        action: String,
        /// Probability mass that was evaluated.
        evaluated_mass: f64,
    },
    /// Several actions share the best score.
    Tied(Vec<String>),
}

/// Facts the three callers already computed. Scores stay at the call site so a
/// zero-probability exclusion and a support filter keep their existing meaning.
pub(crate) struct AggregateFacts<'a> {
    /// At least one structure was evaluated.
    pub any_evaluated: bool,
    /// Every structure was evaluated.
    pub all_resolved: bool,
    /// Candidate action indexes.
    pub candidates: &'a [usize],
    /// Action identities in declaration order.
    pub action_ids: &'a [&'a str],
    /// Per-structure leaders, used by the invariance policy.
    pub leaders: &'a [(String, Vec<String>)],
    /// Probability mass of unidentified structures; zero when no probabilities exist.
    pub unidentified_mass: f64,
    /// Probability mass of unevaluated structures, including unassigned mass.
    pub unevaluated_mass: f64,
    /// Probability mass of evaluated structures.
    pub evaluated_mass: f64,
    /// Evaluated probability mass a support rule removed. Zero when support is not applied.
    pub dropped_mass: f64,
}

/// No structure was evaluated.
pub(crate) const NO_STRUCTURE_EVALUATED: &str = "no structure was evaluated";

/// Invariance cannot be checked while a structure is unidentified or unevaluated.
pub(crate) const INVARIANCE_UNCHECKED: &str =
    "an unidentified or unevaluated structure leaves invariance unchecked";

/// A worst case needs every structure.
pub(crate) const WORST_CASE_NEEDS_EVERY_STRUCTURE: &str =
    "a worst case over structures needs every structure evaluated";

/// Unresolved probability mass can change the Bayes action.
pub(crate) const UNRESOLVED_MASS_CAN_CHANGE_BAYES: &str =
    "positive unresolved probability mass can change the Bayes action";

/// Bayes has no evaluated structure with positive probability.
pub(crate) const NO_EVALUATED_POSITIVE_PROBABILITY: &str =
    "no evaluated structure has positive probability";

/// Sign and policy that combine per-structure values.
#[derive(Clone, Copy, Debug, PartialEq)]
pub(crate) struct StructureAggregator {
    kind: AggregateKind,
    sign: f64,
    criterion: DecisionCriterion,
    policy: StructuralPolicy,
}

impl StructureAggregator {
    /// Sign and policy for `criterion` and `policy`.
    ///
    /// `+1` when a larger criterion value is better, `-1` when smaller is.
    /// Loss, regret and minimax-over-an-identified-set read smaller as better.
    #[must_use]
    pub(crate) fn new(criterion: DecisionCriterion, policy: StructuralPolicy) -> Self {
        let sign = match criterion {
            DecisionCriterion::PosteriorExpectedLoss
            | DecisionCriterion::ExpectedRegret
            | DecisionCriterion::Regret
            | DecisionCriterion::MinimaxOverIdentifiedSet => -1.0,
            _ => 1.0,
        };
        let kind = match criterion {
            DecisionCriterion::MaximinOverStructures
            | DecisionCriterion::MinimaxOverIdentifiedSet => AggregateKind::WorstCase,
            _ => match policy {
                StructuralPolicy::RequireInvariantBestAction => AggregateKind::Invariant,
                StructuralPolicy::Maximin => AggregateKind::WorstCase,
                StructuralPolicy::BayesOverStructures => AggregateKind::Bayes,
                StructuralPolicy::ReportOnly => AggregateKind::Report,
            },
        };
        Self { kind, sign, criterion, policy }
    }

    /// How structures are combined.
    #[must_use]
    pub(crate) const fn kind(self) -> AggregateKind {
        self.kind
    }

    /// `+1` when larger is better, `-1` when smaller is.
    #[must_use]
    pub(crate) const fn sign(self) -> f64 {
        self.sign
    }

    /// Identified-set endpoints in larger-is-better order.
    ///
    /// Loss and minimax negate and swap the interval. Regret and expected
    /// regret stay in the supplied units: their score is a regret, not a
    /// flipped endpoint.
    #[must_use]
    pub(crate) fn orient_interval(self, lower: f64, upper: f64) -> (f64, f64) {
        if self.sign < 0.0
            && !matches!(
                self.criterion,
                DecisionCriterion::Regret | DecisionCriterion::ExpectedRegret
            )
        {
            (-upper, -lower)
        } else {
            (lower, upper)
        }
    }

    /// `Maximin` under `Regret` picks the smallest maximum regret.
    ///
    /// This is the existing identified-set case. It is not a refusal, and it
    /// does not apply to the criteria that already force a worst case.
    #[must_use]
    pub(crate) fn scores_minimax_regret(self) -> bool {
        self.policy == StructuralPolicy::Maximin && self.criterion == DecisionCriterion::Regret
    }

    /// Combine `facts` under this policy.
    ///
    /// `worst_score` and `bayes_score` are consulted only for those policies.
    /// Empty candidate sets and a report policy return before either score.
    pub(crate) fn decide(
        &self,
        facts: &AggregateFacts<'_>,
        worst_score: impl Fn(usize) -> f64,
        bayes_score: impl Fn(usize) -> f64,
    ) -> AggregateDecision {
        if self.kind == AggregateKind::Report {
            return AggregateDecision::Report;
        }
        if !facts.any_evaluated {
            return AggregateDecision::Insufficient(NO_STRUCTURE_EVALUATED);
        }
        if facts.candidates.is_empty() {
            return AggregateDecision::NoCandidate;
        }
        match self.kind {
            AggregateKind::Invariant => Self::invariant(facts),
            AggregateKind::WorstCase => Self::worst_case(facts, worst_score),
            AggregateKind::Bayes => Self::bayes(facts, bayes_score),
            AggregateKind::Report => AggregateDecision::Report,
        }
    }

    fn invariant(facts: &AggregateFacts<'_>) -> AggregateDecision {
        if !facts.all_resolved {
            return AggregateDecision::Insufficient(INVARIANCE_UNCHECKED);
        }
        let winner = facts.leaders.first().and_then(|entry| {
            let first = &entry.1;
            let unique = first.len() == 1
                && facts.leaders.iter().all(|row| &row.1 == first)
                && facts.candidates.iter().any(|&i| facts.action_ids[i] == first[0]);
            unique.then(|| first[0].clone())
        });
        match winner {
            Some(id) => AggregateDecision::Choice(AggregateChoice::Invariant(id)),
            None => AggregateDecision::Disagree(facts.leaders.to_vec()),
        }
    }

    fn worst_case(
        facts: &AggregateFacts<'_>,
        worst_score: impl Fn(usize) -> f64,
    ) -> AggregateDecision {
        if !facts.all_resolved {
            return AggregateDecision::Insufficient(WORST_CASE_NEEDS_EVERY_STRUCTURE);
        }
        Self::scored(facts, worst_score, AggregateChoice::WorstCase)
    }

    fn bayes(facts: &AggregateFacts<'_>, bayes_score: impl Fn(usize) -> f64) -> AggregateDecision {
        if facts.unidentified_mass > 0.0 || facts.unevaluated_mass > 0.0 {
            return AggregateDecision::Insufficient(UNRESOLVED_MASS_CAN_CHANGE_BAYES);
        }
        if facts.evaluated_mass <= 0.0 {
            return AggregateDecision::Insufficient(NO_EVALUATED_POSITIVE_PROBABILITY);
        }
        if facts.dropped_mass > 0.0 {
            return AggregateDecision::SupportWithheld;
        }
        let evaluated_mass = facts.evaluated_mass;
        Self::scored(facts, bayes_score, |action| AggregateChoice::Bayes { action, evaluated_mass })
    }

    fn scored(
        facts: &AggregateFacts<'_>,
        score: impl Fn(usize) -> f64,
        chosen: impl Fn(String) -> AggregateChoice,
    ) -> AggregateDecision {
        match unique_best(facts.candidates, score) {
            Ok(index) => AggregateDecision::Choice(chosen(facts.action_ids[index].to_owned())),
            Err(tied) if tied.is_empty() => AggregateDecision::NoCandidate,
            Err(tied) => AggregateDecision::Choice(AggregateChoice::Tied(
                tied.into_iter().map(|index| facts.action_ids[index].to_owned()).collect(),
            )),
        }
    }
}

/// The unique maximizer, or every index that shares the best score.
fn unique_best(candidates: &[usize], score: impl Fn(usize) -> f64) -> Result<usize, Vec<usize>> {
    let mut best = f64::NEG_INFINITY;
    let mut tied = Vec::new();
    for &index in candidates {
        match score(index).total_cmp(&best) {
            std::cmp::Ordering::Greater => {
                best = score(index);
                tied.clear();
                tied.push(index);
            }
            std::cmp::Ordering::Equal => tied.push(index),
            std::cmp::Ordering::Less => {}
        }
    }
    match tied.as_slice() {
        [only] => Ok(*only),
        _ => Err(tied),
    }
}
