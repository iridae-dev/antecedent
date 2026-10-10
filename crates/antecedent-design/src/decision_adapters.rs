//! Adapters from claim types to decision inputs under a declared policy.
//!
//! The 2.2 claims a decision can read are point claims, graph-dependent claims
//! (one answer per graph, no probability), weighted graph-posterior atoms,
//! finite supplied scenarios and identified sets. The first four become
//! [`StructuralAtom`]s for [`evaluate_structural`]; each atom keeps the
//! empirical support its evidence has, not only whether it was evaluated. An
//! identified set is not an atom: it gives each action an interval of utility,
//! and [`evaluate_identified_sets`] reads those intervals directly (partial
//! identification), reporting dominance, possibly and necessarily optimal
//! actions, and conflicting leaders.
//!
//! Completion counts are never probabilities: `BayesOverStructures` refuses
//! them, and every other policy drops them from the weights. Unidentified and
//! unevaluated mass is passed through and never renormalized. A hard
//! constraint, a declared exclusion or a support rule removes an action; none
//! of them becomes a penalty.

use std::collections::HashSet;

use antecedent_core::{ExternalRefusal, IdentifiedSet, SupportStatus};

use crate::decision_contract::{
    AdmissibilityError, AdmissibleDecisionContract, DecisionAction, DecisionContract,
    DecisionCriterion, StructuralPolicy, UncertaintyKind, UtilityExpr,
};
use crate::decision_robustness::{
    AtomSupport, ClaimProfile, RobustDecisionResult, RobustnessError, evaluate_robust,
};
use crate::decision_structural::{AtomEvidence, StructuralAtom, StructuralDecisionResult};

/// What a claim says about its own probability.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum ClaimProbability {
    /// No probability is claimed.
    Unspecified,
    /// A genuine probability of this structure (a posterior or declared prior).
    Genuine(f64),
    /// How many completions or members produced this structure. Never a
    /// probability.
    CompletionCount(u64),
}

/// One structure's claim.
#[derive(Clone, Debug)]
pub struct SuppliedClaim {
    /// Stable structure identity (a graph, scenario or claim identity).
    pub id: String,
    /// The claim's probability, if any.
    pub probability: ClaimProbability,
    /// Evidence under this structure.
    pub evidence: AtomEvidence,
    /// Empirical support of the evidence, retained per input.
    pub support: AtomSupport,
}

/// Which claim type was adapted.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ClaimKind {
    /// One point claim.
    Point,
    /// One answer per graph, without probabilities.
    GraphDependent,
    /// Atoms weighted by graph-posterior probabilities.
    WeightedGraphPosterior,
    /// A finite set of supplied scenarios, optionally with declared weights.
    FiniteScenarios,
}

/// Claims adapted to structural atoms under a declared policy.
#[derive(Clone, Debug)]
pub struct AdaptedClaims {
    /// Claim type that was adapted.
    pub kind: ClaimKind,
    /// Policy the claims were validated against.
    pub policy: StructuralPolicy,
    /// Atoms for [`evaluate_structural`].
    pub atoms: Vec<StructuralAtom>,
    /// Uncertainty kind and per-atom support.
    pub profile: ClaimProfile,
    /// Completion counts that were supplied, retained and never used as weights.
    pub completion_counts: Vec<(String, u64)>,
}

/// An adapted decision: the per-structure result and its robustness state.
#[derive(Clone, Debug, PartialEq)]
pub struct AdaptedDecision {
    /// Claim type that was adapted.
    pub kind: ClaimKind,
    /// Per-structure results.
    pub structural: StructuralDecisionResult,
    /// Robustness over structures, support and claims.
    pub robust: RobustDecisionResult,
    /// Completion counts that were supplied and not used as weights.
    pub completion_counts: Vec<(String, u64)>,
}

/// Why an adapter refused.
#[derive(Clone, Debug, PartialEq)]
pub enum AdapterError {
    /// The claims are malformed.
    InvalidClaims(&'static str),
    /// `BayesOverStructures` was asked to weight by completion counts.
    CompletionCountNotProbability(String),
    /// `BayesOverStructures` has no genuine probabilities.
    ProbabilitiesRequired,
    /// A graph-dependent claim carries a probability.
    GraphClaimsCarryNoProbability(String),
    /// The adapted claims were validated against another policy.
    PolicyMismatch,
    /// An interval is not finite or has `lower > upper`.
    InvalidInterval(String),
    /// A utility interval names an action the contract does not declare.
    UnknownAction(String),
    /// Two utility intervals name the same action.
    DuplicateAction(String),
    /// A declared action has no utility interval.
    MissingAction(String),
    /// An identified set carries no probability law to weight by.
    BayesOverIdentifiedSet,
    /// The criterion needs a distribution an identified set does not supply.
    CriterionNotLicensedForSet,
    /// The contract or its rules are invalid.
    Admissibility(AdmissibilityError),
    /// The robustness assessment refused.
    Robustness(RobustnessError),
}

impl AdapterError {
    /// Structured refusal with a registered code and a namespaced detail.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = antecedent_core::reason_code!("invalid_argument");
        let unsatisfied = antecedent_core::reason_code!("decision_contract_unsatisfied");
        let build = |code, detail: &str, offending: Option<String>, remedy| ExternalRefusal {
            code,
            stage: "adapt",
            detail: detail.to_owned(),
            offending,
            expected: None,
            supplied: None,
            capability: None,
            remedy,
        };
        match self {
            Self::InvalidClaims(why) => ExternalRefusal {
                expected: Some((*why).to_owned()),
                ..build(invalid, "decision_adapters.invalid_claims", None, None)
            },
            Self::CompletionCountNotProbability(id) => build(
                unsatisfied,
                "decision_adapters.completion_count_not_probability",
                Some(id.clone()),
                Some("supply genuine structure probabilities, not completion counts"),
            ),
            Self::ProbabilitiesRequired => build(
                unsatisfied,
                "decision_adapters.probabilities_required",
                None,
                Some("supply genuine structure probabilities for Bayes over structures"),
            ),
            Self::GraphClaimsCarryNoProbability(id) => build(
                invalid,
                "decision_adapters.graph_claims_carry_no_probability",
                Some(id.clone()),
                Some("adapt probability-weighted graphs as weighted graph-posterior atoms"),
            ),
            Self::PolicyMismatch => build(
                unsatisfied,
                "decision_adapters.policy_mismatch",
                None,
                Some("adapt the claims under the contract's declared structural policy"),
            ),
            Self::InvalidInterval(id) => {
                build(invalid, "decision_adapters.invalid_interval", Some(id.clone()), None)
            }
            Self::UnknownAction(id) => {
                build(invalid, "decision_adapters.unknown_action", Some(id.clone()), None)
            }
            Self::DuplicateAction(id) => {
                build(invalid, "decision_adapters.duplicate_action", Some(id.clone()), None)
            }
            Self::MissingAction(id) => {
                build(invalid, "decision_adapters.missing_action", Some(id.clone()), None)
            }
            Self::BayesOverIdentifiedSet => build(
                unsatisfied,
                "decision_adapters.bayes_over_identified_set",
                None,
                Some(
                    "an identified set has no probability law; use a worst-case or invariance policy",
                ),
            ),
            Self::CriterionNotLicensedForSet => build(
                unsatisfied,
                "decision_adapters.criterion_not_licensed_for_set",
                None,
                Some(
                    "an identified set supports expected-utility, loss, worst-case and minimax regret criteria only",
                ),
            ),
            Self::Admissibility(error) => {
                let mut value = error.to_refusal();
                value.stage = "adapt";
                value
            }
            Self::Robustness(error) => {
                let mut value = error.to_refusal();
                value.stage = "adapt";
                value
            }
        }
    }
}

fn adapt(
    kind: ClaimKind,
    claims: Vec<SuppliedClaim>,
    policy: StructuralPolicy,
) -> Result<AdaptedClaims, AdapterError> {
    if claims.is_empty() {
        return Err(AdapterError::InvalidClaims("at least one claim is required"));
    }
    let mut ids = HashSet::new();
    if claims.iter().any(|c| c.id.trim().is_empty() || !ids.insert(c.id.as_str())) {
        return Err(AdapterError::InvalidClaims("claim identities are distinct and non-blank"));
    }
    if kind == ClaimKind::GraphDependent {
        let weighted =
            claims.iter().find(|c| matches!(c.probability, ClaimProbability::Genuine(_)));
        if let Some(claim) = weighted {
            return Err(AdapterError::GraphClaimsCarryNoProbability(claim.id.clone()));
        }
    }
    let completion_counts: Vec<(String, u64)> = claims
        .iter()
        .filter_map(|c| match c.probability {
            ClaimProbability::CompletionCount(n) => Some((c.id.clone(), n)),
            _ => None,
        })
        .collect();
    if let (StructuralPolicy::BayesOverStructures, Some((id, _))) =
        (policy, completion_counts.first())
    {
        return Err(AdapterError::CompletionCountNotProbability(id.clone()));
    }
    let genuine =
        claims.iter().filter(|c| matches!(c.probability, ClaimProbability::Genuine(_))).count();
    if genuine != 0 && genuine != claims.len() {
        return Err(AdapterError::InvalidClaims(
            "probabilities are given for every claim or for none",
        ));
    }
    if policy == StructuralPolicy::BayesOverStructures && genuine == 0 {
        return Err(AdapterError::ProbabilitiesRequired);
    }
    let uncertainty = match kind {
        ClaimKind::Point => UncertaintyKind::Point,
        ClaimKind::WeightedGraphPosterior if genuine == claims.len() => UncertaintyKind::Credible,
        _ => UncertaintyKind::StructuralEnvelope,
    };
    let support = claims.iter().map(|c| (c.id.clone(), c.support.clone())).collect();
    let atoms = claims
        .into_iter()
        .map(|c| StructuralAtom {
            id: c.id,
            probability: match c.probability {
                ClaimProbability::Genuine(p) => Some(p),
                _ => None,
            },
            evidence: c.evidence,
        })
        .collect();
    Ok(AdaptedClaims {
        kind,
        policy,
        atoms,
        profile: ClaimProfile { uncertainty, support },
        completion_counts,
    })
}

/// Adapt one point claim (a single exact law or answer).
///
/// # Errors
/// A blank identity, or `BayesOverStructures`, which needs genuine structure
/// probabilities a single claim does not supply.
pub fn adapt_point_claim(
    id: &str,
    evidence: AtomEvidence,
    support: AtomSupport,
    policy: StructuralPolicy,
) -> Result<AdaptedClaims, AdapterError> {
    let claim = SuppliedClaim {
        id: id.to_owned(),
        probability: ClaimProbability::Unspecified,
        evidence,
        support,
    };
    adapt(ClaimKind::Point, vec![claim], policy)
}

/// Adapt graph-dependent claims: one answer per graph, no probabilities.
///
/// # Errors
/// Empty, repeated or blank identities; a genuine probability on any graph; and
/// `BayesOverStructures`, which needs probabilities graph-dependent claims do not
/// have (completion counts are named in the refusal).
pub fn adapt_graph_dependent_claims(
    claims: Vec<SuppliedClaim>,
    policy: StructuralPolicy,
) -> Result<AdaptedClaims, AdapterError> {
    adapt(ClaimKind::GraphDependent, claims, policy)
}

/// Adapt weighted graph-posterior atoms. With genuine probabilities for every
/// atom the claims are credible; without them they are a structural envelope.
///
/// # Errors
/// Empty, repeated or blank identities; probabilities for some atoms only;
/// completion counts or missing probabilities under `BayesOverStructures`.
pub fn adapt_weighted_graph_atoms(
    claims: Vec<SuppliedClaim>,
    policy: StructuralPolicy,
) -> Result<AdaptedClaims, AdapterError> {
    adapt(ClaimKind::WeightedGraphPosterior, claims, policy)
}

/// Adapt a finite set of supplied scenarios. Declared weights are weights the
/// caller asserts as probabilities; scenarios are always read as a structural
/// envelope, never as a posterior.
///
/// # Errors
/// As [`adapt_weighted_graph_atoms`].
pub fn adapt_finite_scenarios(
    claims: Vec<SuppliedClaim>,
    policy: StructuralPolicy,
) -> Result<AdaptedClaims, AdapterError> {
    adapt(ClaimKind::FiniteScenarios, claims, policy)
}

/// Evaluate adapted claims under the contract and assess their robustness.
///
/// # Errors
/// A policy that differs from the one the claims were adapted under, and any
/// refusal of the structural evaluation or the robustness assessment.
pub fn evaluate_adapted(
    contract: &AdmissibleDecisionContract,
    adapted: &AdaptedClaims,
) -> Result<AdaptedDecision, AdapterError> {
    if contract.contract.structural_policy != adapted.policy {
        return Err(AdapterError::PolicyMismatch);
    }
    let (structural, robust) = evaluate_robust(contract, &adapted.atoms, &adapted.profile)
        .map_err(AdapterError::Robustness)?;
    Ok(AdaptedDecision {
        kind: adapted.kind,
        structural,
        robust,
        completion_counts: adapted.completion_counts.clone(),
    })
}

/// An identified set of one action's utility: every value in the interval is
/// compatible with the evidence.
#[derive(Clone, Debug, PartialEq)]
pub struct IdentifiedUtility {
    /// Semantic action identity.
    pub action_id: String,
    /// Interval of the criterion value, in the contract's utility units.
    pub utility: IdentifiedSet<f64>,
    /// Hard constraints the action violates; any one excludes it.
    pub hard_exclusions: Vec<String>,
    /// Empirical support of the action's inputs.
    pub support: AtomSupport,
}

/// One action's interval and what partial identification says of it.
#[derive(Clone, Debug, PartialEq)]
pub struct IdentifiedActionRange {
    /// Semantic action identity.
    pub id: String,
    /// Supplied interval `(lower, upper)`.
    pub utility: (f64, f64),
    /// Hard constraints that exclude it.
    pub hard_exclusions: Vec<String>,
    /// Declared reason for removing it.
    pub declared_exclusion: Option<String>,
    /// Inputs below the required support, with their status.
    pub support_shortfalls: Vec<(usize, SupportStatus)>,
    /// Whether no constraint, declaration or support rule removes it.
    pub eligible: bool,
    /// Eligible actions whose whole interval beats this one's.
    pub dominated_by: Vec<String>,
    /// Not dominated: it can be best for some value in the sets.
    pub possibly_optimal: bool,
    /// Beats every other eligible action for every value in the sets.
    pub necessarily_optimal: bool,
    /// Largest regret over the sets, treating each action's utility as free to
    /// vary independently within its interval; an upper bound when the unknowns
    /// are coupled. `None` when the action is not eligible.
    pub max_regret: Option<f64>,
}

/// What partial identification can claim.
#[derive(Clone, Debug, PartialEq)]
pub enum IdentifiedVerdict {
    /// One action beats every other for every value in the sets.
    NecessarilyBest(String),
    /// No action beats every other everywhere; these remain possibly optimal.
    NoNecessarilyBest {
        /// Actions that are not dominated.
        possibly_optimal: Vec<String>,
    },
    /// Best worst case over the sets.
    WorstCaseChoice(String),
    /// Smallest maximum regret.
    MinimaxRegretChoice(String),
    /// The rule leaves these actions tied.
    Tied(Vec<String>),
    /// Intervals only.
    ReportOnly,
    /// Every action that passes constraints and declarations lacks support.
    UnsupportedExtrapolation,
    /// The claims cannot answer the contract.
    InsufficientClaims(String),
    /// Constraints or declarations remove every action.
    NoAdmissibleAction,
}

/// A decision over per-action identified sets.
#[derive(Clone, Debug, PartialEq)]
pub struct IdentifiedSetDecision {
    /// Identity of the contract with its admissibility rules.
    pub contract_identity: String,
    /// Policy that combined the sets.
    pub policy: StructuralPolicy,
    /// Per-action view in declaration order.
    pub actions: Vec<IdentifiedActionRange>,
    /// Eligible action with the highest lower bound, when unique.
    pub lower_leader: Option<String>,
    /// Eligible action with the highest upper bound, when unique.
    pub upper_leader: Option<String>,
    /// The lower-bound and upper-bound leaders differ.
    pub conflicting_leaders: bool,
    /// The verdict.
    pub verdict: IdentifiedVerdict,
}

fn enclose(expr: &UtilityExpr, inputs: &[IdentifiedSet<f64>]) -> Result<(f64, f64), AdapterError> {
    let both =
        |a: &UtilityExpr, b: &UtilityExpr| -> Result<((f64, f64), (f64, f64)), AdapterError> {
            Ok((enclose(a, inputs)?, enclose(b, inputs)?))
        };
    Ok(match expr {
        UtilityExpr::Const(value) => (*value, *value),
        UtilityExpr::Input(k) => {
            let set = inputs
                .get(*k)
                .ok_or(AdapterError::InvalidClaims("one interval per action input"))?;
            (set.lower, set.upper)
        }
        UtilityExpr::Neg(a) => {
            let (lo, hi) = enclose(a, inputs)?;
            (-hi, -lo)
        }
        UtilityExpr::Add(a, b) => {
            let ((alo, ahi), (blo, bhi)) = both(a, b)?;
            (alo + blo, ahi + bhi)
        }
        UtilityExpr::Sub(a, b) => {
            let ((alo, ahi), (blo, bhi)) = both(a, b)?;
            (alo - bhi, ahi - blo)
        }
        UtilityExpr::Mul(a, b) => {
            let ((alo, ahi), (blo, bhi)) = both(a, b)?;
            let products = [alo * blo, alo * bhi, ahi * blo, ahi * bhi];
            (
                products.iter().copied().fold(f64::INFINITY, f64::min),
                products.iter().copied().fold(f64::NEG_INFINITY, f64::max),
            )
        }
        UtilityExpr::Min(a, b) => {
            let ((alo, ahi), (blo, bhi)) = both(a, b)?;
            (alo.min(blo), ahi.min(bhi))
        }
        UtilityExpr::Max(a, b) => {
            let ((alo, ahi), (blo, bhi)) = both(a, b)?;
            (alo.max(blo), ahi.max(bhi))
        }
    })
}

/// Interval of an action's utility given an identified set for each input.
///
/// This is an outer enclosure by interval arithmetic: it contains every value
/// the utility can take, and is sharp when each input appears once in the
/// expression. An input that appears twice may widen it.
///
/// # Errors
/// An unknown action, a count of intervals different from the action's inputs,
/// an invalid contract, or an interval that is not finite.
pub fn utility_interval(
    contract: &DecisionContract,
    action_id: &str,
    inputs: &[IdentifiedSet<f64>],
) -> Result<IdentifiedSet<f64>, AdapterError> {
    contract
        .validate()
        .map_err(|error| AdapterError::Admissibility(AdmissibilityError::Contract(error)))?;
    let action: &DecisionAction = contract
        .actions
        .iter()
        .find(|a| a.id == action_id)
        .ok_or_else(|| AdapterError::UnknownAction(action_id.to_owned()))?;
    if inputs.len() != action.inputs.len()
        || inputs.iter().any(|s| !s.lower.is_finite() || !s.upper.is_finite() || s.lower > s.upper)
    {
        return Err(AdapterError::InvalidInterval(action_id.to_owned()));
    }
    let (lo, hi) = enclose(&action.utility, inputs)?;
    IdentifiedSet::try_new(lo, hi).map_err(|_| AdapterError::InvalidInterval(action_id.to_owned()))
}

fn check_coverage<'a>(
    actions: &[DecisionAction],
    utilities: &'a [IdentifiedUtility],
) -> Result<Vec<&'a IdentifiedUtility>, AdapterError> {
    let mut seen = HashSet::new();
    for utility in utilities {
        if !actions.iter().any(|a| a.id == utility.action_id) {
            return Err(AdapterError::UnknownAction(utility.action_id.clone()));
        }
        if !seen.insert(utility.action_id.as_str()) {
            return Err(AdapterError::DuplicateAction(utility.action_id.clone()));
        }
        let set = &utility.utility;
        if !set.lower.is_finite() || !set.upper.is_finite() || set.lower > set.upper {
            return Err(AdapterError::InvalidInterval(utility.action_id.clone()));
        }
    }
    actions
        .iter()
        .map(|a| {
            utilities
                .iter()
                .find(|u| u.action_id == a.id)
                .ok_or_else(|| AdapterError::MissingAction(a.id.clone()))
        })
        .collect()
}

struct Gate {
    hard: bool,
    declared: bool,
    shortfalls: Vec<(usize, SupportStatus)>,
}

impl Gate {
    fn open(&self) -> bool {
        self.hard && self.declared && self.shortfalls.is_empty()
    }
}

struct Analysis {
    dominated_by: Vec<String>,
    possibly: bool,
    necessarily: bool,
    max_regret: Option<f64>,
}

/// Dominance and regret over the eligible actions, in value orientation.
fn analyse(eligible: &[usize], oriented: &[(f64, f64)], ids: &[&str]) -> Vec<Analysis> {
    (0..oriented.len())
        .map(|a| {
            if !eligible.contains(&a) {
                return Analysis {
                    dominated_by: Vec::new(),
                    possibly: false,
                    necessarily: false,
                    max_regret: None,
                };
            }
            let others = || eligible.iter().copied().filter(move |b| *b != a);
            let dominated_by: Vec<String> = others()
                .filter(|b| oriented[*b].0 > oriented[a].1)
                .map(|b| ids[b].to_owned())
                .collect();
            let regret = others().map(|b| oriented[b].1 - oriented[a].0).fold(0.0_f64, f64::max);
            Analysis {
                possibly: dominated_by.is_empty(),
                necessarily: others().all(|b| oriented[a].0 > oriented[b].1),
                dominated_by,
                max_regret: Some(regret),
            }
        })
        .collect()
}

/// The unique maximizer of `key` over `eligible`, or the tied set.
fn unique_best(eligible: &[usize], key: impl Fn(usize) -> f64) -> Result<usize, Vec<usize>> {
    let best = eligible.iter().copied().map(&key).fold(f64::NEG_INFINITY, f64::max);
    let tied: Vec<usize> =
        eligible.iter().copied().filter(|i| key(*i).total_cmp(&best).is_eq()).collect();
    match tied.as_slice() {
        [only] => Ok(*only),
        _ => Err(tied),
    }
}

fn pick(
    result: Result<usize, Vec<usize>>,
    ids: &[&str],
    chosen: fn(String) -> IdentifiedVerdict,
) -> IdentifiedVerdict {
    match result {
        Ok(i) => chosen(ids[i].to_owned()),
        Err(tied) => IdentifiedVerdict::Tied(tied.into_iter().map(|i| ids[i].to_owned()).collect()),
    }
}

/// The verdict over a nonempty eligible set.
fn verdict_over(
    base: &DecisionContract,
    worst_case_criterion: bool,
    eligible: &[usize],
    oriented: &[(f64, f64)],
    analysis: &[Analysis],
    ids: &[&str],
) -> IdentifiedVerdict {
    let worst_case =
        || pick(unique_best(eligible, |i| oriented[i].0), ids, IdentifiedVerdict::WorstCaseChoice);
    if worst_case_criterion {
        return worst_case();
    }
    match base.structural_policy {
        StructuralPolicy::RequireInvariantBestAction => {
            match eligible.iter().copied().find(|i| analysis[*i].necessarily) {
                Some(i) => IdentifiedVerdict::NecessarilyBest(ids[i].to_owned()),
                None => IdentifiedVerdict::NoNecessarilyBest {
                    possibly_optimal: eligible
                        .iter()
                        .filter(|i| analysis[**i].possibly)
                        .map(|i| ids[*i].to_owned())
                        .collect(),
                },
            }
        }
        StructuralPolicy::Maximin if base.criterion == DecisionCriterion::Regret => pick(
            unique_best(eligible, |i| -analysis[i].max_regret.unwrap_or(f64::INFINITY)),
            ids,
            IdentifiedVerdict::MinimaxRegretChoice,
        ),
        StructuralPolicy::Maximin => worst_case(),
        StructuralPolicy::ReportOnly | StructuralPolicy::BayesOverStructures => {
            IdentifiedVerdict::ReportOnly
        }
    }
}

/// Decide over per-action identified sets of utility.
///
/// Intervals are read in the contract's units. `PosteriorExpectedLoss` and
/// `MinimaxOverIdentifiedSet` read them as losses. Under
/// `RequireInvariantBestAction` an action is chosen only if its interval beats
/// every other eligible action's interval entirely (strictly); otherwise the
/// possibly optimal actions are listed. Under `Maximin`, or for the
/// `MaximinOverStructures` and `MinimaxOverIdentifiedSet` criteria, the best
/// worst case wins; the `Regret` criterion under `Maximin` picks the smallest
/// maximum regret. `ReportOnly` returns the intervals.
///
/// # Errors
/// An invalid contract, a utility interval for an unknown, repeated or
/// missing action, a non-finite or inverted interval, `BayesOverStructures`
/// (an identified set has no probability law) and criteria that need a
/// distribution (`ThresholdProbability`, `Quantile`, `ExpectedRegret`).
#[allow(clippy::too_many_lines)]
pub fn evaluate_identified_sets(
    contract: &AdmissibleDecisionContract,
    utilities: &[IdentifiedUtility],
) -> Result<IdentifiedSetDecision, AdapterError> {
    let identity = contract.identity().map_err(AdapterError::Admissibility)?;
    let base = &contract.contract;
    if matches!(
        base.criterion,
        DecisionCriterion::ThresholdProbability { .. }
            | DecisionCriterion::Quantile { .. }
            | DecisionCriterion::ExpectedRegret
    ) {
        return Err(AdapterError::CriterionNotLicensedForSet);
    }
    let worst_case_criterion = matches!(
        base.criterion,
        DecisionCriterion::MaximinOverStructures | DecisionCriterion::MinimaxOverIdentifiedSet
    );
    if base.structural_policy == StructuralPolicy::BayesOverStructures && !worst_case_criterion {
        return Err(AdapterError::BayesOverIdentifiedSet);
    }
    let supplied = check_coverage(&base.actions, utilities)?;
    let loss = matches!(
        base.criterion,
        DecisionCriterion::PosteriorExpectedLoss | DecisionCriterion::MinimaxOverIdentifiedSet
    );
    let oriented: Vec<(f64, f64)> = supplied
        .iter()
        .map(|u| {
            let set = &u.utility;
            if loss { (-set.upper, -set.lower) } else { (set.lower, set.upper) }
        })
        .collect();
    let gates: Vec<Gate> = base
        .actions
        .iter()
        .zip(&supplied)
        .map(|(action, claim)| Gate {
            hard: claim.hard_exclusions.is_empty(),
            declared: contract.rules.exclusion_reason(&action.id).is_none(),
            shortfalls: (0..action.inputs.len())
                .filter_map(|j| {
                    let status = claim.support.status_of(&action.id, j);
                    (!contract.rules.support_met(&action.id, j, status)).then_some((j, status))
                })
                .collect(),
        })
        .collect();
    let ids: Vec<&str> = base.actions.iter().map(|a| a.id.as_str()).collect();
    let eligible: Vec<usize> = (0..ids.len()).filter(|i| gates[*i].open()).collect();
    let analysis = analyse(&eligible, &oriented, &ids);

    let lead = |key: &dyn Fn(usize) -> f64| -> Option<String> {
        unique_best(&eligible, key).ok().map(|i| ids[i].to_owned())
    };
    let lower_leader = if eligible.is_empty() { None } else { lead(&|i| oriented[i].0) };
    let upper_leader = if eligible.is_empty() { None } else { lead(&|i| oriented[i].1) };
    let conflicting_leaders =
        matches!((&lower_leader, &upper_leader), (Some(a), Some(b)) if a != b);

    let support_only = eligible.is_empty() && gates.iter().any(|g| g.hard && g.declared);
    let verdict = if !contract.rules.uncertainty.satisfied_by(UncertaintyKind::StructuralEnvelope) {
        IdentifiedVerdict::InsufficientClaims(format!(
            "the decision requires {} uncertainty but an identified set is a structural envelope",
            contract.rules.uncertainty.name()
        ))
    } else if eligible.is_empty() {
        if support_only {
            IdentifiedVerdict::UnsupportedExtrapolation
        } else {
            IdentifiedVerdict::NoAdmissibleAction
        }
    } else {
        verdict_over(base, worst_case_criterion, &eligible, &oriented, &analysis, &ids)
    };

    let actions = base
        .actions
        .iter()
        .enumerate()
        .map(|(i, action)| IdentifiedActionRange {
            id: action.id.clone(),
            utility: (supplied[i].utility.lower, supplied[i].utility.upper),
            hard_exclusions: supplied[i].hard_exclusions.clone(),
            declared_exclusion: contract.rules.exclusion_reason(&action.id).map(str::to_owned),
            support_shortfalls: gates[i].shortfalls.clone(),
            eligible: gates[i].open(),
            dominated_by: analysis[i].dominated_by.clone(),
            possibly_optimal: analysis[i].possibly,
            necessarily_optimal: analysis[i].necessarily,
            max_regret: analysis[i].max_regret,
        })
        .collect();
    Ok(IdentifiedSetDecision {
        contract_identity: identity,
        policy: base.structural_policy,
        actions,
        lower_leader,
        upper_leader,
        conflicting_leaders,
        verdict,
    })
}
