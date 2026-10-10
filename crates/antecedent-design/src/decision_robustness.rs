//! Robustness of a decision across structures, support and claims.
//!
//! [`evaluate_structural`] answers what each structure says. This module reads
//! those per-structure results together with the contract's admissibility rules
//! and the support each structure's evidence has, and names which of these
//! states the decision is in:
//!
//! - structurally robust: one action is uniquely best in every structure;
//! - support robust: the same action wins with and without the structures that
//!   fall short of the required empirical support;
//! - support dependent: the choice changes if support rules are ignored;
//! - graph dependent: structures that meet the support rules disagree;
//! - unsupported extrapolation: no structure supports any admissible action;
//! - insufficient claims: the claims cannot answer the contract (unresolved
//!   structures, unresolved probability mass, or the wrong uncertainty kind);
//! - no admissible action: hard constraints or declared rules remove every action.
//!
//! Support rules and declared exclusions only remove actions in the structure
//! that violates them; they are never a penalty. Unidentified and unevaluated
//! mass is read from the structural result and never renormalized. The
//! per-structure outcomes are consumed as computed; nothing is re-evaluated.

use antecedent_core::{ExternalRefusal, SupportStatus};

use crate::decision_contract::{
    AdmissibilityError, AdmissibleDecisionContract, DecisionCriterion, StructuralPolicy,
    UncertaintyKind, UncertaintyRequirement,
};
use crate::decision_structural::{
    AtomStatus, StructuralAtom, StructuralDecisionResult, StructuralError, evaluate_structural,
};

/// Support status of one input of one action under one structure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct InputSupport {
    /// Action that reads the input.
    pub action_id: String,
    /// Input position within the action.
    pub input: usize,
    /// Empirical support status of the input under this structure.
    pub status: SupportStatus,
}

/// Empirical support a structure's evidence has, retained beyond whether the
/// structure was evaluated.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct AtomSupport {
    /// Status of every input without its own entry.
    pub overall: SupportStatus,
    /// Per-input statuses.
    pub per_input: Vec<InputSupport>,
}

impl AtomSupport {
    /// Every input empirically supported.
    #[must_use]
    pub const fn supported() -> Self {
        Self { overall: SupportStatus::Supported, per_input: Vec::new() }
    }

    /// Support was not assessed; a rule that needs support treats it as missing
    /// evidence.
    #[must_use]
    pub const fn unassessed() -> Self {
        Self { overall: SupportStatus::MissingEvidence, per_input: Vec::new() }
    }

    /// One uniform status for every input.
    #[must_use]
    pub const fn uniform(status: SupportStatus) -> Self {
        Self { overall: status, per_input: Vec::new() }
    }

    /// Status of `action_id`'s input.
    #[must_use]
    pub fn status_of(&self, action_id: &str, input: usize) -> SupportStatus {
        self.per_input
            .iter()
            .find(|entry| entry.action_id == action_id && entry.input == input)
            .map_or(self.overall, |entry| entry.status)
    }
}

/// What the supplied claims carry beyond their draws.
#[derive(Clone, Debug, PartialEq)]
pub struct ClaimProfile {
    /// Uncertainty representation the claims carry.
    pub uncertainty: UncertaintyKind,
    /// Support per structure identity; a structure without an entry is
    /// unassessed.
    pub support: Vec<(String, AtomSupport)>,
}

impl ClaimProfile {
    fn support_of(&self, atom_id: &str) -> Option<&AtomSupport> {
        self.support.iter().find(|(id, _)| id == atom_id).map(|(_, support)| support)
    }
}

/// An input that fell short of a support rule under one structure.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SupportShortfall {
    /// Structure identity.
    pub atom: String,
    /// Action whose input fell short.
    pub action: String,
    /// Input position.
    pub input: usize,
    /// Status the structure reported.
    pub status: SupportStatus,
    /// Weakest status the rules allow.
    pub weakest_allowed: SupportStatus,
}

/// The robustness state of a decision.
#[derive(Clone, Debug, PartialEq)]
pub enum RobustVerdict {
    /// Uniquely best in every structure, with every input supported as required.
    StructurallyRobust(String),
    /// Uniquely best in every structure that meets the support rules, and the
    /// choice is unchanged when the rules are ignored.
    SupportRobust(String),
    /// The supported structures choose one action; ignoring the support rules
    /// does not (or chooses another).
    SupportDependent {
        /// Choice among the supported structures.
        supported_choice: String,
        /// Choice with the support rules ignored, when there is one.
        unrestricted_choice: Option<String>,
    },
    /// Structures that meet the support rules disagree on the best action; each
    /// structure's leaders are listed.
    GraphDependentChoice(Vec<(String, Vec<String>)>),
    /// No structure supports any admissible action.
    UnsupportedExtrapolation,
    /// The claims cannot answer the contract.
    InsufficientClaims(String),
    /// Hard constraints or declared rules remove every action.
    NoAdmissibleAction,
    /// Best worst case over structures.
    WorstCaseChoice(String),
    /// Best probability-weighted value; the unresolved mass is reported on the
    /// result, not assumed.
    BayesChoice {
        /// Chosen action.
        action: String,
        /// Probability mass that was evaluated.
        evaluated_mass: f64,
    },
    /// Per-structure answers only.
    ReportOnly,
}

impl RobustVerdict {
    /// The action the verdict selects, when it selects one.
    #[must_use]
    pub fn selected_action(&self) -> Option<&str> {
        match self {
            Self::StructurallyRobust(id)
            | Self::SupportRobust(id)
            | Self::WorstCaseChoice(id)
            | Self::BayesChoice { action: id, .. } => Some(id.as_str()),
            _ => None,
        }
    }
}

/// One action across structures and rules.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionRobustness {
    /// Semantic action identity.
    pub id: String,
    /// Declared reason for removing the action, when removed.
    pub declared_exclusion: Option<String>,
    /// Criterion range over evaluated structures where hard constraints admit it.
    pub range: Option<(f64, f64)>,
    /// Criterion range over the structures where it also meets the support rules.
    pub supported_range: Option<(f64, f64)>,
    /// Structures where only a support rule removes it.
    pub unsupported_in: Vec<String>,
}

/// A robustness assessment.
#[derive(Clone, Debug, PartialEq)]
pub struct RobustDecisionResult {
    /// Identity of the contract with its admissibility rules.
    pub contract_identity: String,
    /// Identity of the base decision contract.
    pub base_contract_identity: String,
    /// The robustness state.
    pub verdict: RobustVerdict,
    /// Per-action view in declaration order.
    pub actions: Vec<ActionRobustness>,
    /// Every input that fell short of a support rule.
    pub shortfalls: Vec<SupportShortfall>,
    /// Evaluated structures that support no admissible action.
    pub unsupported_atoms: Vec<String>,
    /// Probability mass of the unsupported structures, when probabilities exist.
    pub unsupported_mass: Option<f64>,
    /// Unidentified mass from the structural result.
    pub unidentified_mass: Option<f64>,
    /// Unevaluated mass from the structural result.
    pub unevaluated_mass: Option<f64>,
    /// Evaluated mass from the structural result.
    pub evaluated_mass: Option<f64>,
    /// Uncertainty representation the contract requires.
    pub uncertainty_required: UncertaintyRequirement,
    /// Uncertainty representation the claims carry.
    pub uncertainty_supplied: UncertaintyKind,
    /// Assumptions the assessment stands on.
    pub assumptions: Vec<String>,
}

/// Why a robustness assessment refused.
#[derive(Clone, Debug, PartialEq)]
pub enum RobustnessError {
    /// The contract or its rules are invalid.
    Admissibility(AdmissibilityError),
    /// The structural evaluation refused.
    Structural(StructuralError),
    /// The structural result was not computed from this contract.
    ContractMismatch,
    /// The claim profile names a structure twice or one that was not evaluated
    /// against.
    InvalidProfile(&'static str),
}

impl RobustnessError {
    /// Structured refusal with a registered code and a namespaced detail.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let build = |code, detail: &str, expected: Option<&str>| ExternalRefusal {
            code,
            stage: "robustness",
            detail: detail.to_owned(),
            offending: None,
            expected: expected.map(str::to_owned),
            supplied: None,
            capability: None,
            remedy: None,
        };
        match self {
            Self::Admissibility(error) => {
                let mut value = error.to_refusal();
                value.stage = "robustness";
                value
            }
            Self::Structural(error) => {
                let mut value = error.to_refusal();
                value.stage = "robustness";
                value
            }
            Self::ContractMismatch => ExternalRefusal {
                remedy: Some("assess the structural result against the contract that produced it"),
                ..build(
                    antecedent_core::reason_code!("decision_contract_unsatisfied"),
                    "decision_robustness.contract_mismatch",
                    None,
                )
            },
            Self::InvalidProfile(why) => build(
                antecedent_core::reason_code!("invalid_argument"),
                "decision_robustness.invalid_profile",
                Some(*why),
            ),
        }
    }
}

/// How structures are aggregated, after the criterion override.
#[derive(Clone, Copy, PartialEq)]
enum Agg {
    Invariant,
    WorstCase,
    Bayes,
    Report,
}

#[derive(Clone, Debug, PartialEq)]
enum Choice {
    Invariant(String),
    WorstCase(String),
    Bayes(String, f64),
}

impl Choice {
    fn action(&self) -> &str {
        match self {
            Self::Invariant(id) | Self::WorstCase(id) | Self::Bayes(id, _) => id,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
enum Outcome {
    Report,
    Insufficient(String),
    NoCandidate,
    Unsupported,
    Disagree(Vec<(String, Vec<String>)>),
    Choice(Choice),
}

struct AtomView {
    value: Vec<f64>,
    hard: Vec<bool>,
    support: Vec<bool>,
}

struct AtomRow {
    id: String,
    probability: Option<f64>,
    view: Option<AtomView>,
}

struct Ctx<'a> {
    action_ids: Vec<&'a str>,
    declared: Vec<bool>,
    atoms: Vec<AtomRow>,
    agg: Agg,
    sign: f64,
    all_resolved: bool,
    unidentified_mass: Option<f64>,
    unevaluated_mass: Option<f64>,
    evaluated_mass: Option<f64>,
}

fn same(a: f64, b: f64) -> bool {
    a.total_cmp(&b).is_eq()
}

impl Ctx<'_> {
    fn n(&self) -> usize {
        self.action_ids.len()
    }

    fn eligible(&self, view: &AtomView, action: usize, supported: bool) -> bool {
        self.declared[action] && view.hard[action] && (!supported || view.support[action])
    }

    /// Evaluated structures that still have an eligible action in this view.
    fn retained(&self, supported: bool) -> Vec<usize> {
        (0..self.atoms.len())
            .filter(|k| {
                self.atoms[*k].view.as_ref().is_some_and(|view| {
                    !supported || (0..self.n()).any(|i| self.eligible(view, i, true))
                })
            })
            .collect()
    }

    fn counts(&self, atom: usize) -> bool {
        self.agg != Agg::Bayes || self.atoms[atom].probability.is_some_and(|p| p > 0.0)
    }

    /// Actions eligible in every retained structure that counts.
    fn candidates(&self, supported: bool, retained: &[usize]) -> Vec<usize> {
        (0..self.n())
            .filter(|i| {
                self.declared[*i]
                    && retained.iter().all(|k| {
                        !self.counts(*k)
                            || self.atoms[*k]
                                .view
                                .as_ref()
                                .is_some_and(|view| self.eligible(view, *i, supported))
                    })
            })
            .collect()
    }

    /// Actions tied for best in one structure; empty when none is eligible.
    fn leaders(&self, view: &AtomView, supported: bool) -> Vec<usize> {
        let eligible: Vec<usize> =
            (0..self.n()).filter(|i| self.eligible(view, *i, supported)).collect();
        let score = |i: usize| self.sign * view.value[i];
        let Some(best) =
            eligible.iter().copied().max_by(|a, b| score(*a).total_cmp(&score(*b)).then(b.cmp(a)))
        else {
            return Vec::new();
        };
        eligible.into_iter().filter(|i| same(score(*i), score(best))).collect()
    }

    fn resolve(&self, supported: bool) -> Outcome {
        if self.agg == Agg::Report {
            return Outcome::Report;
        }
        let unrestricted = self.retained(false);
        if unrestricted.is_empty() {
            return Outcome::Insufficient("no structure was evaluated".into());
        }
        let retained = self.retained(supported);
        if retained.is_empty() {
            let supportable = unrestricted.iter().any(|k| {
                self.atoms[*k]
                    .view
                    .as_ref()
                    .is_some_and(|view| (0..self.n()).any(|i| self.eligible(view, i, false)))
            });
            return if supported && supportable {
                Outcome::Unsupported
            } else {
                Outcome::NoCandidate
            };
        }
        let candidates = self.candidates(supported, &retained);
        if candidates.is_empty() {
            return if supported && !self.candidates(false, &unrestricted).is_empty() {
                Outcome::Unsupported
            } else {
                Outcome::NoCandidate
            };
        }
        match self.agg {
            Agg::Invariant => self.invariant(supported, &retained, &candidates),
            Agg::WorstCase => self.worst_case(&retained, &candidates),
            Agg::Bayes => self.bayes(supported, &retained, &unrestricted, &candidates),
            Agg::Report => Outcome::Report,
        }
    }

    fn invariant(&self, supported: bool, retained: &[usize], candidates: &[usize]) -> Outcome {
        if !self.all_resolved {
            return Outcome::Insufficient(
                "an unidentified or unevaluated structure leaves invariance unchecked".into(),
            );
        }
        let leaders: Vec<(String, Vec<String>)> = retained
            .iter()
            .filter_map(|k| {
                let row = &self.atoms[*k];
                let view = row.view.as_ref()?;
                let ids = self
                    .leaders(view, supported)
                    .into_iter()
                    .map(|i| self.action_ids[i].to_owned())
                    .collect();
                Some((row.id.clone(), ids))
            })
            .collect();
        let winner = leaders.first().map(|entry| entry.1.clone()).filter(|first| {
            first.len() == 1
                && leaders.iter().all(|entry| &entry.1 == first)
                && candidates.iter().any(|i| self.action_ids[*i] == first[0])
        });
        match winner {
            Some(first) => Outcome::Choice(Choice::Invariant(first[0].clone())),
            None => Outcome::Disagree(leaders),
        }
    }

    fn worst_case(&self, retained: &[usize], candidates: &[usize]) -> Outcome {
        if !self.all_resolved {
            return Outcome::Insufficient(
                "a worst case over structures needs every structure evaluated".into(),
            );
        }
        let worst = |i: usize| -> f64 {
            retained
                .iter()
                .filter_map(|k| self.atoms[*k].view.as_ref())
                .map(|view| self.sign * view.value[i])
                .fold(f64::INFINITY, f64::min)
        };
        let best = candidates
            .iter()
            .copied()
            .max_by(|a, b| worst(*a).total_cmp(&worst(*b)).then(b.cmp(a)));
        best.map_or(Outcome::NoCandidate, |i| {
            Outcome::Choice(Choice::WorstCase(self.action_ids[i].to_owned()))
        })
    }

    fn bayes(
        &self,
        supported: bool,
        retained: &[usize],
        unrestricted: &[usize],
        candidates: &[usize],
    ) -> Outcome {
        if self.unidentified_mass.unwrap_or(0.0) > 0.0 || self.unevaluated_mass.unwrap_or(0.0) > 0.0
        {
            return Outcome::Insufficient(
                "positive unresolved probability mass can change the Bayes action".into(),
            );
        }
        let evaluated = self.evaluated_mass.unwrap_or(0.0);
        if evaluated <= 0.0 {
            return Outcome::Insufficient("no evaluated structure has positive probability".into());
        }
        let dropped: f64 = unrestricted
            .iter()
            .filter(|k| supported && !retained.contains(k))
            .filter_map(|k| self.atoms[*k].probability)
            .sum();
        if dropped > 0.0 {
            return Outcome::Unsupported;
        }
        let score = |i: usize| -> f64 {
            let weighted: f64 = retained
                .iter()
                .filter_map(|k| {
                    let row = &self.atoms[*k];
                    Some(row.probability? * row.view.as_ref()?.value[i])
                })
                .sum();
            self.sign * weighted
        };
        let best = candidates
            .iter()
            .copied()
            .max_by(|a, b| score(*a).total_cmp(&score(*b)).then(b.cmp(a)));
        best.map_or(Outcome::NoCandidate, |i| {
            Outcome::Choice(Choice::Bayes(self.action_ids[i].to_owned(), evaluated))
        })
    }
}

fn aggregation(criterion: DecisionCriterion, policy: StructuralPolicy) -> (Agg, f64) {
    let sign = match criterion {
        DecisionCriterion::PosteriorExpectedLoss
        | DecisionCriterion::ExpectedRegret
        | DecisionCriterion::Regret
        | DecisionCriterion::MinimaxOverIdentifiedSet => -1.0,
        _ => 1.0,
    };
    let agg = match criterion {
        DecisionCriterion::MaximinOverStructures | DecisionCriterion::MinimaxOverIdentifiedSet => {
            Agg::WorstCase
        }
        _ => match policy {
            StructuralPolicy::RequireInvariantBestAction => Agg::Invariant,
            StructuralPolicy::Maximin => Agg::WorstCase,
            StructuralPolicy::BayesOverStructures => Agg::Bayes,
            StructuralPolicy::ReportOnly => Agg::Report,
        },
    };
    (agg, sign)
}

fn check_profile(
    structural: &StructuralDecisionResult,
    profile: &ClaimProfile,
) -> Result<(), RobustnessError> {
    let mut seen = std::collections::HashSet::new();
    for (id, _) in &profile.support {
        if !structural.atoms.iter().any(|atom| &atom.id == id) {
            return Err(RobustnessError::InvalidProfile("support names an unknown structure"));
        }
        if !seen.insert(id.as_str()) {
            return Err(RobustnessError::InvalidProfile("support names a structure twice"));
        }
    }
    Ok(())
}

fn build_ctx<'a>(
    contract: &'a AdmissibleDecisionContract,
    structural: &StructuralDecisionResult,
    profile: &ClaimProfile,
) -> Result<Ctx<'a>, RobustnessError> {
    let actions = &contract.contract.actions;
    let action_ids: Vec<&str> = actions.iter().map(|a| a.id.as_str()).collect();
    let declared: Vec<bool> =
        action_ids.iter().map(|id| contract.rules.exclusion_reason(id).is_none()).collect();
    let unassessed = AtomSupport::unassessed();
    let mut atoms = Vec::with_capacity(structural.atoms.len());
    for summary in &structural.atoms {
        let view = match &summary.status {
            AtomStatus::Evaluated(result) => {
                if result.actions.len() != actions.len()
                    || result.actions.iter().zip(actions).any(|(o, a)| o.id != a.id)
                {
                    return Err(RobustnessError::ContractMismatch);
                }
                let support = profile.support_of(&summary.id).unwrap_or(&unassessed);
                Some(AtomView {
                    value: result.actions.iter().map(|o| o.value).collect(),
                    hard: result.actions.iter().map(|o| o.admissible).collect(),
                    support: actions
                        .iter()
                        .map(|a| {
                            (0..a.inputs.len()).all(|j| {
                                contract.rules.support_met(&a.id, j, support.status_of(&a.id, j))
                            })
                        })
                        .collect(),
                })
            }
            _ => None,
        };
        atoms.push(AtomRow { id: summary.id.clone(), probability: summary.probability, view });
    }
    let (agg, sign) = aggregation(contract.contract.criterion, contract.contract.structural_policy);
    Ok(Ctx {
        action_ids,
        declared,
        agg,
        sign,
        all_resolved: atoms.iter().all(|row| row.view.is_some()),
        atoms,
        unidentified_mass: structural.unidentified_mass,
        unevaluated_mass: structural.unevaluated_mass,
        evaluated_mass: structural.evaluated_mass,
    })
}

fn assemble(all: &Outcome, supported: Outcome, shortfall: bool) -> RobustVerdict {
    match supported {
        Outcome::Report => RobustVerdict::ReportOnly,
        Outcome::Insufficient(why) => RobustVerdict::InsufficientClaims(why),
        Outcome::NoCandidate => RobustVerdict::NoAdmissibleAction,
        Outcome::Unsupported => RobustVerdict::UnsupportedExtrapolation,
        Outcome::Disagree(leaders) => RobustVerdict::GraphDependentChoice(leaders),
        Outcome::Choice(choice) => {
            let unrestricted = match all {
                Outcome::Choice(other) => Some(other.action()),
                _ => None,
            };
            if unrestricted != Some(choice.action()) {
                return RobustVerdict::SupportDependent {
                    supported_choice: choice.action().to_owned(),
                    unrestricted_choice: unrestricted.map(str::to_owned),
                };
            }
            match choice {
                Choice::Invariant(id) if shortfall => RobustVerdict::SupportRobust(id),
                Choice::Invariant(id) => RobustVerdict::StructurallyRobust(id),
                Choice::WorstCase(id) => RobustVerdict::WorstCaseChoice(id),
                Choice::Bayes(action, evaluated_mass) => {
                    RobustVerdict::BayesChoice { action, evaluated_mass }
                }
            }
        }
    }
}

fn shortfalls_of(
    contract: &AdmissibleDecisionContract,
    ctx: &Ctx<'_>,
    profile: &ClaimProfile,
) -> Vec<SupportShortfall> {
    let unassessed = AtomSupport::unassessed();
    let mut out = Vec::new();
    for row in &ctx.atoms {
        let Some(view) = &row.view else {
            continue;
        };
        let support = profile.support_of(&row.id).unwrap_or(&unassessed);
        for (i, action) in contract.contract.actions.iter().enumerate() {
            if !ctx.declared[i] || !view.hard[i] {
                continue;
            }
            for input in 0..action.inputs.len() {
                let status = support.status_of(&action.id, input);
                let Some(weakest) = contract.rules.weakest_allowed(&action.id, input) else {
                    continue;
                };
                if status.severity() > weakest.severity() {
                    out.push(SupportShortfall {
                        atom: row.id.clone(),
                        action: action.id.clone(),
                        input,
                        status,
                        weakest_allowed: weakest,
                    });
                }
            }
        }
    }
    out
}

fn range_of(values: impl Iterator<Item = f64>) -> Option<(f64, f64)> {
    values.fold(None, |range, v| match range {
        None => Some((v, v)),
        Some((lo, hi)) => Some((lo.min(v), hi.max(v))),
    })
}

/// Assess the robustness of a structural decision under admissibility rules.
///
/// # Errors
/// Refuses an invalid contract or rule set, a structural result computed from a
/// different contract, and a profile naming an unknown or repeated structure.
pub fn assess_robustness(
    contract: &AdmissibleDecisionContract,
    structural: &StructuralDecisionResult,
    profile: &ClaimProfile,
) -> Result<RobustDecisionResult, RobustnessError> {
    let identity = contract.identity().map_err(RobustnessError::Admissibility)?;
    let base = contract
        .contract
        .identity()
        .map_err(|error| RobustnessError::Admissibility(AdmissibilityError::Contract(error)))?;
    if base != structural.contract_identity {
        return Err(RobustnessError::ContractMismatch);
    }
    check_profile(structural, profile)?;
    let ctx = build_ctx(contract, structural, profile)?;
    let shortfalls = shortfalls_of(contract, &ctx, profile);

    let all = ctx.resolve(false);
    let supported = ctx.resolve(true);
    let mut verdict = assemble(&all, supported, !shortfalls.is_empty());
    let required = contract.rules.uncertainty;
    if !required.satisfied_by(profile.uncertainty) {
        verdict = RobustVerdict::InsufficientClaims(format!(
            "the decision requires {} uncertainty but the claims carry {}",
            required.name(),
            profile.uncertainty.name()
        ));
    }

    let retained = ctx.retained(true);
    let unsupported_atoms: Vec<String> = ctx
        .retained(false)
        .into_iter()
        .filter(|k| !retained.contains(k))
        .map(|k| ctx.atoms[k].id.clone())
        .collect();
    let has_probabilities = structural.evaluated_mass.is_some();
    let unsupported_mass = has_probabilities.then(|| {
        ctx.atoms
            .iter()
            .filter(|row| unsupported_atoms.contains(&row.id))
            .filter_map(|row| row.probability)
            .sum::<f64>()
    });
    let actions = contract
        .contract
        .actions
        .iter()
        .enumerate()
        .map(|(i, action)| {
            let evaluated = ctx.atoms.iter().filter_map(|row| Some((row, row.view.as_ref()?)));
            ActionRobustness {
                id: action.id.clone(),
                declared_exclusion: contract.rules.exclusion_reason(&action.id).map(str::to_owned),
                range: range_of(
                    evaluated.clone().filter(|(_, v)| v.hard[i]).map(|(_, v)| v.value[i]),
                ),
                supported_range: range_of(
                    evaluated
                        .clone()
                        .filter(|(_, v)| v.hard[i] && v.support[i])
                        .map(|(_, v)| v.value[i]),
                ),
                unsupported_in: evaluated
                    .filter(|(_, v)| v.hard[i] && !v.support[i])
                    .map(|(row, _)| row.id.clone())
                    .collect(),
            }
        })
        .collect();
    Ok(RobustDecisionResult {
        contract_identity: identity,
        base_contract_identity: base,
        verdict,
        actions,
        shortfalls,
        unsupported_atoms,
        unsupported_mass,
        unidentified_mass: structural.unidentified_mass,
        unevaluated_mass: structural.unevaluated_mass,
        evaluated_mass: structural.evaluated_mass,
        uncertainty_required: required,
        uncertainty_supplied: profile.uncertainty,
        assumptions: vec![
            "each structure's draws answer the contract under that structure alone".into(),
            "support rules and declared exclusions only remove actions; they never enter a utility"
                .into(),
            "unidentified and unevaluated mass is reported and never renormalized away".into(),
            "a structural envelope is not a probability law; completion counts are not probabilities"
                .into(),
        ],
    })
}

/// Evaluate `atoms` under the base contract and assess robustness in one call.
///
/// # Errors
/// Any refusal of [`evaluate_structural`] or [`assess_robustness`].
pub fn evaluate_robust(
    contract: &AdmissibleDecisionContract,
    atoms: &[StructuralAtom],
    profile: &ClaimProfile,
) -> Result<(StructuralDecisionResult, RobustDecisionResult), RobustnessError> {
    contract.validate().map_err(RobustnessError::Admissibility)?;
    let structural =
        evaluate_structural(&contract.contract, atoms).map_err(RobustnessError::Structural)?;
    let robust = assess_robustness(contract, &structural, profile)?;
    Ok((structural, robust))
}
