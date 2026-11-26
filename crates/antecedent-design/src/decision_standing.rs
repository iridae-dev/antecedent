//! One reading of every decision result: the action it chooses, how firmly it stands behind
//! that choice, and a plain statement.
//!
//! Each decision result keeps its own verdict vocabulary. [`DecisionOutcome`] reads any of them
//! through the same three questions. [`Standing`] is the shared answer to "what kind of claim is
//! this": a value read off the verdict by an exhaustive match, so a new verdict variant cannot be
//! added without saying what it stands for.

// SPDX-License-Identifier: MIT OR Apache-2.0

use crate::decision_eval::{DecisionResult, Verdict};
use crate::decision_robustness::{RobustDecisionResult, RobustVerdict};
use crate::decision_structural::{StructuralDecisionResult, StructuralVerdict};
use crate::preference_certified::{
    CertifiedPreferenceResult, CertifiedVerdict, StructuralCertifiedResult,
    StructuralCertifiedVerdict,
};
use crate::preference_decision::{
    NO_ADMISSIBLE_SENTENCE, PreferenceDecisionResult, PreferenceVerdict, REPORT_ONLY_SENTENCE,
    StructuralPreferenceResult, StructuralPreferenceVerdict, join_ids, measure_choice_sentence,
    minimax_regret_sentence, plain_number, preferred_everywhere, worst_case_sentence,
};
use crate::preference_route_choice::{PreferenceEvaluation, StructuralPreferenceEvaluation};

/// What kind of claim a decision result makes about its choice.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Standing {
    /// One action is best everywhere the result looks: under the supplied law, in every
    /// structure, or at every admissible trade-off value. A tie that leaves one action never
    /// beaten is included.
    Best,
    /// Several actions share the best score; declaration order is not a choice.
    Tied,
    /// A declared policy picks the action (worst case, minimax regret, a supplied measure or
    /// structure probabilities); the action is not best everywhere.
    PolicyChoice,
    /// The best action depends on structure, support rules or trade-off values.
    Depends,
    /// The analysis did not settle the choice within its budget.
    Undetermined,
    /// Hard constraints or declared rules remove every action.
    NoAdmissibleAction,
    /// Everything was computed and no choice was requested.
    ReportOnly,
    /// A claim would need evidence that is missing or unsupported.
    InsufficientEvidence,
    /// The statement is an estimate over sampled members, not an exact ranking.
    SampledEstimate,
}

impl Standing {
    /// Stable `snake_case` id of the standing.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Best => "best",
            Self::Tied => "tied",
            Self::PolicyChoice => "policy_choice",
            Self::Depends => "depends",
            Self::Undetermined => "undetermined",
            Self::NoAdmissibleAction => "no_admissible_action",
            Self::ReportOnly => "report_only",
            Self::InsufficientEvidence => "insufficient_evidence",
            Self::SampledEstimate => "sampled_estimate",
        }
    }
}

/// A decision result read through its choice, its standing and its statement.
pub trait DecisionOutcome {
    /// The action the result chooses; `None` when it chooses none (a tie, a dependence on
    /// structure or trade-offs, an unsettled analysis, a report-only result).
    fn choice(&self) -> Option<&str>;

    /// The kind of claim the result makes.
    fn standing(&self) -> Standing;

    /// The result in one or two plain sentences, built only from its own facts.
    fn statement(&self) -> String;
}

// ---------------------------------------------------------------------------------------------
// Ordinary decisions
// ---------------------------------------------------------------------------------------------

fn ordinary_standing(verdict: &Verdict) -> Standing {
    match verdict {
        Verdict::UniquelyOptimal(_) => Standing::Best,
        Verdict::Indistinguishable(_) => Standing::Tied,
        Verdict::NoAdmissibleAction => Standing::NoAdmissibleAction,
    }
}

impl DecisionOutcome for DecisionResult {
    fn choice(&self) -> Option<&str> {
        match &self.verdict {
            Verdict::UniquelyOptimal(action) => Some(action.as_str()),
            Verdict::Indistinguishable(_) | Verdict::NoAdmissibleAction => None,
        }
    }

    fn standing(&self) -> Standing {
        ordinary_standing(&self.verdict)
    }

    fn statement(&self) -> String {
        match &self.verdict {
            Verdict::UniquelyOptimal(action) => {
                format!("Action {action} ranks first on {}.", self.criterion_label)
            }
            Verdict::Indistinguishable(actions) => format!(
                "Actions {} have equal values on {}; no action is chosen.",
                join_ids(actions),
                self.criterion_label
            ),
            Verdict::NoAdmissibleAction => NO_ADMISSIBLE_SENTENCE.to_owned(),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Ordinary decisions across structures
// ---------------------------------------------------------------------------------------------

fn structural_standing(verdict: &StructuralVerdict) -> Standing {
    match verdict {
        StructuralVerdict::InvariantBest(_) => Standing::Best,
        StructuralVerdict::NoInvariantBest(_) => Standing::Depends,
        StructuralVerdict::WorstCaseChoice(_) | StructuralVerdict::BayesChoice { .. } => {
            Standing::PolicyChoice
        }
        StructuralVerdict::Tied(_) => Standing::Tied,
        StructuralVerdict::ReportOnly => Standing::ReportOnly,
        StructuralVerdict::InsufficientScience(_) => Standing::InsufficientEvidence,
        StructuralVerdict::NoAdmissibleAction => Standing::NoAdmissibleAction,
    }
}

fn leaders_sentence(leaders: &[(String, Vec<String>)]) -> String {
    let parts: Vec<String> = leaders
        .iter()
        .map(|(atom, actions)| format!("{atom} favours {}", join_ids(actions)))
        .collect();
    join_ids(&parts)
}

impl DecisionOutcome for StructuralDecisionResult {
    fn choice(&self) -> Option<&str> {
        match &self.verdict {
            StructuralVerdict::InvariantBest(action)
            | StructuralVerdict::WorstCaseChoice(action)
            | StructuralVerdict::BayesChoice { action, .. } => Some(action.as_str()),
            StructuralVerdict::NoInvariantBest(_)
            | StructuralVerdict::Tied(_)
            | StructuralVerdict::ReportOnly
            | StructuralVerdict::InsufficientScience(_)
            | StructuralVerdict::NoAdmissibleAction => None,
        }
    }

    fn standing(&self) -> Standing {
        structural_standing(&self.verdict)
    }

    fn statement(&self) -> String {
        match &self.verdict {
            StructuralVerdict::InvariantBest(action) => {
                format!("Action {action} is uniquely best in every evaluated structure.")
            }
            StructuralVerdict::NoInvariantBest(leaders) => format!(
                "No action is uniquely best in every evaluated structure: {}.",
                leaders_sentence(leaders)
            ),
            StructuralVerdict::WorstCaseChoice(action) => format!(
                "Action {action} has the best worst-case value over the evaluated structures."
            ),
            StructuralVerdict::Tied(actions) => {
                format!("Actions {} share the best score; no action is chosen.", join_ids(actions))
            }
            StructuralVerdict::BayesChoice { action, evaluated_mass } => format!(
                "Action {action} has the best probability-weighted value over the evaluated \
                 structures, which carry a mass of {}.",
                plain_number(*evaluated_mass)
            ),
            StructuralVerdict::ReportOnly => {
                "No action is chosen: the result of each structure is reported only.".to_owned()
            }
            StructuralVerdict::InsufficientScience(reason) => {
                format!("No action is chosen across structures: {reason}.")
            }
            StructuralVerdict::NoAdmissibleAction => NO_ADMISSIBLE_SENTENCE.to_owned(),
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Robustness of an ordinary decision
// ---------------------------------------------------------------------------------------------

fn robust_standing(verdict: &RobustVerdict) -> Standing {
    match verdict {
        RobustVerdict::StructurallyRobust(_) | RobustVerdict::SupportRobust(_) => Standing::Best,
        RobustVerdict::SupportDependent { .. } | RobustVerdict::GraphDependentChoice(_) => {
            Standing::Depends
        }
        RobustVerdict::UnsupportedExtrapolation | RobustVerdict::InsufficientClaims(_) => {
            Standing::InsufficientEvidence
        }
        RobustVerdict::NoAdmissibleAction => Standing::NoAdmissibleAction,
        RobustVerdict::WorstCaseChoice(_) | RobustVerdict::BayesChoice { .. } => {
            Standing::PolicyChoice
        }
        RobustVerdict::Tied(_) => Standing::Tied,
        RobustVerdict::ReportOnly => Standing::ReportOnly,
    }
}

impl DecisionOutcome for RobustDecisionResult {
    fn choice(&self) -> Option<&str> {
        match &self.verdict {
            RobustVerdict::StructurallyRobust(action)
            | RobustVerdict::SupportRobust(action)
            | RobustVerdict::WorstCaseChoice(action)
            | RobustVerdict::BayesChoice { action, .. }
            | RobustVerdict::SupportDependent { supported_choice: action, .. } => {
                Some(action.as_str())
            }
            RobustVerdict::GraphDependentChoice(_)
            | RobustVerdict::UnsupportedExtrapolation
            | RobustVerdict::InsufficientClaims(_)
            | RobustVerdict::NoAdmissibleAction
            | RobustVerdict::Tied(_)
            | RobustVerdict::ReportOnly => None,
        }
    }

    fn standing(&self) -> Standing {
        robust_standing(&self.verdict)
    }

    fn statement(&self) -> String {
        match &self.verdict {
            RobustVerdict::StructurallyRobust(action) => format!(
                "Action {action} is uniquely best in every structure, with every input supported \
                 as the rules require."
            ),
            RobustVerdict::SupportRobust(action) => format!(
                "Action {action} is uniquely best in every structure that meets the support \
                 rules, and stays best when the rules are ignored."
            ),
            RobustVerdict::SupportDependent { supported_choice, unrestricted_choice } => {
                let unrestricted = unrestricted_choice.as_ref().map_or_else(
                    || "no action is chosen".to_owned(),
                    |action| format!("action {action} is chosen"),
                );
                format!(
                    "Action {supported_choice} is best among the structures that meet the \
                     support rules; with the rules ignored, {unrestricted}."
                )
            }
            RobustVerdict::GraphDependentChoice(leaders) => format!(
                "Structures that meet the support rules disagree on the best action: {}.",
                leaders_sentence(leaders)
            ),
            RobustVerdict::UnsupportedExtrapolation => {
                "No structure supports any admissible action.".to_owned()
            }
            RobustVerdict::InsufficientClaims(reason) => {
                format!("The claims cannot answer the contract: {reason}.")
            }
            RobustVerdict::NoAdmissibleAction => NO_ADMISSIBLE_SENTENCE.to_owned(),
            RobustVerdict::WorstCaseChoice(action) => {
                format!("Action {action} has the best worst-case value over the structures.")
            }
            RobustVerdict::Tied(actions) => {
                format!("Actions {} share the best score; no action is chosen.", join_ids(actions))
            }
            RobustVerdict::BayesChoice { action, evaluated_mass } => format!(
                "Action {action} has the best probability-weighted value over the evaluated \
                 structures, which carry a mass of {}.",
                plain_number(*evaluated_mass)
            ),
            RobustVerdict::ReportOnly => {
                "No action is chosen: the result of each structure is reported only.".to_owned()
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Decisions across a preference set: exact route
// ---------------------------------------------------------------------------------------------

fn exact_choice(verdict: &PreferenceVerdict) -> Option<&str> {
    match verdict {
        PreferenceVerdict::NecessarilyBest(action)
        | PreferenceVerdict::NeverBeaten { action, .. }
        | PreferenceVerdict::WorstCaseChoice(action)
        | PreferenceVerdict::MinimaxRegretChoice(action)
        | PreferenceVerdict::MeasureChoice { action, .. } => Some(action.as_str()),
        PreferenceVerdict::PreferenceDependent { .. }
        | PreferenceVerdict::Tied(_)
        | PreferenceVerdict::ReportOnly
        | PreferenceVerdict::NoAdmissibleAction => None,
    }
}

fn exact_standing(verdict: &PreferenceVerdict) -> Standing {
    match verdict {
        PreferenceVerdict::NecessarilyBest(_) | PreferenceVerdict::NeverBeaten { .. } => {
            Standing::Best
        }
        PreferenceVerdict::PreferenceDependent { .. } => Standing::Depends,
        PreferenceVerdict::WorstCaseChoice(_)
        | PreferenceVerdict::MinimaxRegretChoice(_)
        | PreferenceVerdict::MeasureChoice { .. } => Standing::PolicyChoice,
        PreferenceVerdict::Tied(_) => Standing::Tied,
        PreferenceVerdict::ReportOnly => Standing::ReportOnly,
        PreferenceVerdict::NoAdmissibleAction => Standing::NoAdmissibleAction,
    }
}

impl DecisionOutcome for PreferenceDecisionResult {
    fn choice(&self) -> Option<&str> {
        exact_choice(&self.verdict)
    }

    fn standing(&self) -> Standing {
        exact_standing(&self.verdict)
    }

    fn statement(&self) -> String {
        PreferenceDecisionResult::statement(self)
    }
}

fn structural_exact_choice(verdict: &StructuralPreferenceVerdict) -> Option<&str> {
    match verdict {
        StructuralPreferenceVerdict::Combined(inner) => exact_choice(inner),
        StructuralPreferenceVerdict::StructuresDisagree(_)
        | StructuralPreferenceVerdict::InsufficientScience(_)
        | StructuralPreferenceVerdict::ReportOnly => None,
    }
}

fn structural_exact_standing(verdict: &StructuralPreferenceVerdict) -> Standing {
    match verdict {
        StructuralPreferenceVerdict::Combined(inner) => exact_standing(inner),
        StructuralPreferenceVerdict::StructuresDisagree(_) => Standing::Depends,
        StructuralPreferenceVerdict::InsufficientScience(_) => Standing::InsufficientEvidence,
        StructuralPreferenceVerdict::ReportOnly => Standing::ReportOnly,
    }
}

impl DecisionOutcome for StructuralPreferenceResult {
    fn choice(&self) -> Option<&str> {
        structural_exact_choice(&self.verdict)
    }

    fn standing(&self) -> Standing {
        structural_exact_standing(&self.verdict)
    }

    fn statement(&self) -> String {
        StructuralPreferenceResult::statement(self)
    }
}

// ---------------------------------------------------------------------------------------------
// Decisions across a preference set: certified route
// ---------------------------------------------------------------------------------------------

fn certified_choice(verdict: &CertifiedVerdict) -> Option<&str> {
    match verdict {
        CertifiedVerdict::NecessarilyBest(action)
        | CertifiedVerdict::WorstCaseChoice(action)
        | CertifiedVerdict::MinimaxRegretChoice(action)
        | CertifiedVerdict::MeasureChoice { action, .. } => Some(action.as_str()),
        CertifiedVerdict::PreferenceDependent { .. }
        | CertifiedVerdict::Undetermined { .. }
        | CertifiedVerdict::ReportOnly
        | CertifiedVerdict::NoAdmissibleAction => None,
    }
}

fn certified_standing(verdict: &CertifiedVerdict) -> Standing {
    match verdict {
        CertifiedVerdict::NecessarilyBest(_) => Standing::Best,
        CertifiedVerdict::PreferenceDependent { .. } => Standing::Depends,
        CertifiedVerdict::WorstCaseChoice(_)
        | CertifiedVerdict::MinimaxRegretChoice(_)
        | CertifiedVerdict::MeasureChoice { .. } => Standing::PolicyChoice,
        CertifiedVerdict::Undetermined { .. } => Standing::Undetermined,
        CertifiedVerdict::ReportOnly => Standing::ReportOnly,
        CertifiedVerdict::NoAdmissibleAction => Standing::NoAdmissibleAction,
    }
}

/// The sentence of a certified verdict; `scope` is appended to the claims that hold over the
/// whole set (for example the structures they hold across).
fn certified_sentence(verdict: &CertifiedVerdict, scope: &str) -> String {
    match verdict {
        CertifiedVerdict::NecessarilyBest(action) => preferred_everywhere(action, true, scope),
        CertifiedVerdict::PreferenceDependent { certified_somewhere } => format!(
            "The preferred action depends on the declared trade-offs: {} {} certified preferred \
             on part of the admissible set{scope}.",
            join_ids(certified_somewhere),
            if certified_somewhere.len() == 1 { "is" } else { "are each" }
        ),
        CertifiedVerdict::WorstCaseChoice(action) => worst_case_sentence(action, scope),
        CertifiedVerdict::MinimaxRegretChoice(action) => minimax_regret_sentence(action, scope),
        CertifiedVerdict::MeasureChoice { action, share } => {
            measure_choice_sentence(action, *share)
        }
        CertifiedVerdict::Undetermined { candidates, undecided_fraction } => format!(
            "The certified subdivision does not settle the choice among {}; a volume fraction of \
             {} of the admissible set is left undecided.",
            join_ids(candidates),
            plain_number(*undecided_fraction)
        ),
        CertifiedVerdict::ReportOnly => REPORT_ONLY_SENTENCE.to_owned(),
        CertifiedVerdict::NoAdmissibleAction => NO_ADMISSIBLE_SENTENCE.to_owned(),
    }
}

impl DecisionOutcome for CertifiedPreferenceResult {
    fn choice(&self) -> Option<&str> {
        certified_choice(&self.verdict)
    }

    fn standing(&self) -> Standing {
        certified_standing(&self.verdict)
    }

    fn statement(&self) -> String {
        PreferenceEvaluation::Certified(Box::new(self.clone())).statement()
    }
}

fn structural_certified_choice(verdict: &StructuralCertifiedVerdict) -> Option<&str> {
    match verdict {
        StructuralCertifiedVerdict::Combined(inner) => certified_choice(inner),
        StructuralCertifiedVerdict::StructuresDisagree(_)
        | StructuralCertifiedVerdict::InsufficientScience(_)
        | StructuralCertifiedVerdict::ReportOnly => None,
    }
}

fn structural_certified_standing(verdict: &StructuralCertifiedVerdict) -> Standing {
    match verdict {
        StructuralCertifiedVerdict::Combined(inner) => certified_standing(inner),
        StructuralCertifiedVerdict::StructuresDisagree(_) => Standing::Depends,
        StructuralCertifiedVerdict::InsufficientScience(_) => Standing::InsufficientEvidence,
        StructuralCertifiedVerdict::ReportOnly => Standing::ReportOnly,
    }
}

impl DecisionOutcome for StructuralCertifiedResult {
    fn choice(&self) -> Option<&str> {
        structural_certified_choice(&self.verdict)
    }

    fn standing(&self) -> Standing {
        structural_certified_standing(&self.verdict)
    }

    fn statement(&self) -> String {
        match &self.verdict {
            StructuralCertifiedVerdict::Combined(inner) => {
                certified_sentence(inner, " and every evaluated structure")
            }
            StructuralCertifiedVerdict::StructuresDisagree(choices) => {
                let atoms: Vec<String> = choices.iter().map(|c| c.atom.clone()).collect();
                format!(
                    "The evaluated structures {} give different verdicts, so no action is \
                     preferred across structures.",
                    join_ids(&atoms)
                )
            }
            StructuralCertifiedVerdict::InsufficientScience(reason) => {
                format!("No action is chosen across structures: {reason}.")
            }
            StructuralCertifiedVerdict::ReportOnly => {
                "No action is chosen: the result of each structure is reported only.".to_owned()
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Decisions across a preference set: whichever route answered
// ---------------------------------------------------------------------------------------------

impl DecisionOutcome for PreferenceEvaluation {
    fn choice(&self) -> Option<&str> {
        match self {
            Self::Exact(result) => exact_choice(&result.verdict),
            Self::Certified(result) => certified_choice(&result.verdict),
        }
    }

    fn standing(&self) -> Standing {
        match self {
            Self::Exact(result) => exact_standing(&result.verdict),
            Self::Certified(result) => certified_standing(&result.verdict),
        }
    }

    fn statement(&self) -> String {
        PreferenceEvaluation::statement(self)
    }
}

impl DecisionOutcome for StructuralPreferenceEvaluation {
    fn choice(&self) -> Option<&str> {
        match self {
            Self::Exact(result) => structural_exact_choice(&result.verdict),
            Self::Certified(result) => structural_certified_choice(&result.verdict),
        }
    }

    fn standing(&self) -> Standing {
        match self {
            Self::Exact(result) => structural_exact_standing(&result.verdict),
            Self::Certified(result) => structural_certified_standing(&result.verdict),
        }
    }

    fn statement(&self) -> String {
        match self {
            Self::Exact(result) => result.as_ref().statement(),
            Self::Certified(result) => DecisionOutcome::statement(result.as_ref()),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decision_contract::DecisionCriterion;
    use crate::decision_eval::SourceReceipt;
    use crate::preference_certified::AtomVerdict;
    use crate::preference_decision::AtomChoice;

    fn ids(names: &[&str]) -> Vec<String> {
        names.iter().map(|n| (*n).to_owned()).collect()
    }

    fn ordinary(verdict: Verdict) -> DecisionResult {
        DecisionResult {
            method_provenance: None,
            contract_identity: "c".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            criterion_label: "posterior expected utility".into(),
            actions: Vec::new(),
            verdict,
            evpi: None,
            n_draws: 0,
            effective_draws: 0.0,
            source: SourceReceipt {
                provider_id: String::new(),
                snapshot_id: String::new(),
                rng_id: String::new(),
                causal_contract_id: String::new(),
            },
            assumptions: Vec::new(),
        }
    }

    #[test]
    fn standing_ids_are_distinct_snake_case() {
        let all = [
            Standing::Best,
            Standing::Tied,
            Standing::PolicyChoice,
            Standing::Depends,
            Standing::Undetermined,
            Standing::NoAdmissibleAction,
            Standing::ReportOnly,
            Standing::InsufficientEvidence,
            Standing::SampledEstimate,
        ];
        let mut seen: Vec<&str> = all.iter().map(|s| s.as_str()).collect();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), all.len());
        assert!(
            seen.iter().all(|id| id.chars().all(|c| c.is_ascii_lowercase() || c == '_')),
            "{seen:?}"
        );
    }

    #[test]
    fn an_ordinary_result_chooses_only_a_unique_leader() {
        let best = ordinary(Verdict::UniquelyOptimal("treat".into()));
        assert_eq!(best.choice(), Some("treat"));
        assert_eq!(best.standing(), Standing::Best);
        assert_eq!(best.statement(), "Action treat ranks first on posterior expected utility.");

        let tied = ordinary(Verdict::Indistinguishable(ids(&["a", "b"])));
        assert_eq!(tied.choice(), None);
        assert_eq!(tied.standing(), Standing::Tied);
        assert!(tied.statement().contains("a and b"));

        let none = ordinary(Verdict::NoAdmissibleAction);
        assert_eq!(none.choice(), None);
        assert_eq!(none.standing(), Standing::NoAdmissibleAction);
        assert_eq!(none.statement(), NO_ADMISSIBLE_SENTENCE);
    }

    #[test]
    fn robust_verdicts_map_to_their_standing() {
        let cases = [
            (RobustVerdict::StructurallyRobust("a".into()), Standing::Best),
            (RobustVerdict::SupportRobust("a".into()), Standing::Best),
            (
                RobustVerdict::SupportDependent {
                    supported_choice: "a".into(),
                    unrestricted_choice: None,
                },
                Standing::Depends,
            ),
            (RobustVerdict::GraphDependentChoice(Vec::new()), Standing::Depends),
            (RobustVerdict::UnsupportedExtrapolation, Standing::InsufficientEvidence),
            (RobustVerdict::InsufficientClaims("x".into()), Standing::InsufficientEvidence),
            (RobustVerdict::NoAdmissibleAction, Standing::NoAdmissibleAction),
            (RobustVerdict::WorstCaseChoice("a".into()), Standing::PolicyChoice),
            (RobustVerdict::Tied(ids(&["a", "b"])), Standing::Tied),
            (
                RobustVerdict::BayesChoice { action: "a".into(), evaluated_mass: 1.0 },
                Standing::PolicyChoice,
            ),
            (RobustVerdict::ReportOnly, Standing::ReportOnly),
        ];
        for (verdict, standing) in cases {
            assert_eq!(robust_standing(&verdict), standing, "{verdict:?}");
        }
    }

    #[test]
    fn structural_verdicts_map_to_their_standing() {
        let cases = [
            (StructuralVerdict::InvariantBest("a".into()), Standing::Best),
            (StructuralVerdict::NoInvariantBest(Vec::new()), Standing::Depends),
            (StructuralVerdict::WorstCaseChoice("a".into()), Standing::PolicyChoice),
            (StructuralVerdict::Tied(ids(&["a", "b"])), Standing::Tied),
            (
                StructuralVerdict::BayesChoice { action: "a".into(), evaluated_mass: 0.5 },
                Standing::PolicyChoice,
            ),
            (StructuralVerdict::ReportOnly, Standing::ReportOnly),
            (StructuralVerdict::InsufficientScience("x".into()), Standing::InsufficientEvidence),
            (StructuralVerdict::NoAdmissibleAction, Standing::NoAdmissibleAction),
        ];
        for (verdict, standing) in cases {
            assert_eq!(structural_standing(&verdict), standing, "{verdict:?}");
        }
    }

    #[test]
    fn exact_preference_verdicts_map_to_standing_and_choice() {
        let cases = [
            (PreferenceVerdict::NecessarilyBest("a".into()), Standing::Best, Some("a")),
            (
                PreferenceVerdict::NeverBeaten { action: "a".into(), ties_with: ids(&["b"]) },
                Standing::Best,
                Some("a"),
            ),
            (
                PreferenceVerdict::PreferenceDependent { possibly_best: ids(&["a", "b"]) },
                Standing::Depends,
                None,
            ),
            (PreferenceVerdict::WorstCaseChoice("a".into()), Standing::PolicyChoice, Some("a")),
            (PreferenceVerdict::MinimaxRegretChoice("b".into()), Standing::PolicyChoice, Some("b")),
            (
                PreferenceVerdict::MeasureChoice { action: "b".into(), share: 0.75 },
                Standing::PolicyChoice,
                Some("b"),
            ),
            (PreferenceVerdict::Tied(ids(&["a", "b"])), Standing::Tied, None),
            (PreferenceVerdict::ReportOnly, Standing::ReportOnly, None),
            (PreferenceVerdict::NoAdmissibleAction, Standing::NoAdmissibleAction, None),
        ];
        for (verdict, standing, choice) in cases {
            assert_eq!(exact_standing(&verdict), standing, "{verdict:?}");
            assert_eq!(exact_choice(&verdict), choice, "{verdict:?}");
        }
    }

    #[test]
    fn certified_verdicts_map_to_standing_and_choice() {
        let cases = [
            (CertifiedVerdict::NecessarilyBest("a".into()), Standing::Best, Some("a")),
            (
                CertifiedVerdict::PreferenceDependent { certified_somewhere: ids(&["a", "b"]) },
                Standing::Depends,
                None,
            ),
            (CertifiedVerdict::WorstCaseChoice("a".into()), Standing::PolicyChoice, Some("a")),
            (CertifiedVerdict::MinimaxRegretChoice("a".into()), Standing::PolicyChoice, Some("a")),
            (
                CertifiedVerdict::MeasureChoice { action: "a".into(), share: 1.0 },
                Standing::PolicyChoice,
                Some("a"),
            ),
            (
                CertifiedVerdict::Undetermined {
                    candidates: ids(&["a", "b"]),
                    undecided_fraction: 0.25,
                },
                Standing::Undetermined,
                None,
            ),
            (CertifiedVerdict::ReportOnly, Standing::ReportOnly, None),
            (CertifiedVerdict::NoAdmissibleAction, Standing::NoAdmissibleAction, None),
        ];
        for (verdict, standing, choice) in cases {
            assert_eq!(certified_standing(&verdict), standing, "{verdict:?}");
            assert_eq!(certified_choice(&verdict), choice, "{verdict:?}");
        }
    }

    #[test]
    fn structural_preference_verdicts_follow_the_inner_verdict() {
        let combined =
            StructuralPreferenceVerdict::Combined(PreferenceVerdict::NecessarilyBest("a".into()));
        assert_eq!(structural_exact_standing(&combined), Standing::Best);
        assert_eq!(structural_exact_choice(&combined), Some("a"));

        let disagree = StructuralPreferenceVerdict::StructuresDisagree(vec![AtomChoice {
            atom: "g1".into(),
            verdict: PreferenceVerdict::ReportOnly,
        }]);
        assert_eq!(structural_exact_standing(&disagree), Standing::Depends);
        assert_eq!(structural_exact_choice(&disagree), None);

        let missing = StructuralPreferenceVerdict::InsufficientScience("g2".into());
        assert_eq!(structural_exact_standing(&missing), Standing::InsufficientEvidence);
        assert_eq!(
            structural_exact_standing(&StructuralPreferenceVerdict::ReportOnly),
            Standing::ReportOnly
        );

        let certified = StructuralCertifiedVerdict::Combined(CertifiedVerdict::Undetermined {
            candidates: ids(&["a"]),
            undecided_fraction: 0.5,
        });
        assert_eq!(structural_certified_standing(&certified), Standing::Undetermined);
        assert_eq!(structural_certified_choice(&certified), None);
        let disagree = StructuralCertifiedVerdict::StructuresDisagree(vec![AtomVerdict {
            atom: "g1".into(),
            verdict: CertifiedVerdict::ReportOnly,
        }]);
        assert_eq!(structural_certified_standing(&disagree), Standing::Depends);
        assert_eq!(
            structural_certified_standing(&StructuralCertifiedVerdict::InsufficientScience(
                "g".into()
            )),
            Standing::InsufficientEvidence
        );
    }

    #[test]
    fn certified_sentences_name_the_scope() {
        let sentence = certified_sentence(
            &CertifiedVerdict::NecessarilyBest("a".into()),
            " and every evaluated structure",
        );
        assert!(sentence.starts_with("Action a is preferred"), "{sentence}");
        assert!(sentence.contains("every evaluated structure"), "{sentence}");
        let undetermined = certified_sentence(
            &CertifiedVerdict::Undetermined {
                candidates: ids(&["a", "b"]),
                undecided_fraction: 0.25,
            },
            "",
        );
        assert!(undetermined.contains("a and b") && undetermined.contains("0.25"));
    }
}
