//! Decisions under structural uncertainty.
//!
//! Each structure (a graph, a completion, a supplied scenario) is an atom that
//! carries its own evidence and, only when genuine, a probability. Every atom is
//! evaluated on its own draws; the contract's [`StructuralPolicy`] then decides
//! whether and how atoms are combined:
//!
//! - `RequireInvariantBestAction` names an action only if it is uniquely best in
//!   every structure;
//! - `Maximin` (and the `MaximinOverStructures` / `MinimaxOverIdentifiedSet`
//!   criteria) rank by the worst case over structures;
//! - `BayesOverStructures` weights by supplied structure probabilities, which a
//!   completion count is not;
//! - `ReportOnly` returns each structure's answer without choosing.
//!
//! Unidentified and unevaluated mass is retained and never renormalized away.
//! A hard constraint that fails in any structure with mass excludes the action.

use antecedent_io_core::distribution_artifact::DistributionArtifact;

use crate::decision_contract::{
    DecisionContract, DecisionContractError, DecisionCriterion, StructuralPolicy,
};
use crate::decision_eval::{DecisionEvalError, DecisionResult, Verdict, evaluate_contract};
use crate::structure_aggregator::{
    AggregateChoice, AggregateDecision, AggregateFacts, AggregateKind, StructureAggregator,
};

/// What is known about one structure.
#[derive(Clone, Debug)]
pub enum AtomEvidence {
    /// Joint draws under this structure.
    Evaluated(Box<DistributionArtifact>),
    /// The quantity is not identified under this structure.
    Unidentified,
    /// The structure was not evaluated (budget, missing evidence, truncation).
    Unevaluated(String),
}

/// One structure with an optional genuine probability.
#[derive(Clone, Debug)]
pub struct StructuralAtom {
    /// Stable structure identity.
    pub id: String,
    /// Probability of this structure. Supply it for every atom or for none;
    /// completion counts are not probabilities.
    pub probability: Option<f64>,
    /// Evidence under this structure.
    pub evidence: AtomEvidence,
}

/// What became of an atom.
#[derive(Clone, Debug, PartialEq)]
pub enum AtomStatus {
    /// Evaluated; carries the per-structure decision.
    Evaluated(Box<DecisionResult>),
    /// Not identified.
    Unidentified,
    /// Not evaluated, with the registered reason code or the supplied reason.
    Unevaluated(String),
}

/// One structure in the result.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomSummary {
    /// Structure identity.
    pub id: String,
    /// Supplied probability.
    pub probability: Option<f64>,
    /// Outcome.
    pub status: AtomStatus,
}

/// One action across structures.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionStructure {
    /// Semantic action identity.
    pub id: String,
    /// Criterion value per atom, in atom order; `None` where the atom was not
    /// evaluated or the action was excluded there.
    pub per_atom: Vec<Option<f64>>,
    /// Lowest and highest criterion value over evaluated atoms where admissible.
    pub range: Option<(f64, f64)>,
    /// `sum p_k * value_k` over evaluated atoms; present only with genuine
    /// probabilities, and not renormalized by the evaluated mass.
    pub weighted_value: Option<f64>,
    /// Probability mass of the structures in which this action leads.
    pub mass_where_best: Option<f64>,
    /// Structures in which a hard constraint excluded it.
    pub excluded_in: Vec<String>,
}

/// What the structural analysis can claim.
#[derive(Clone, Debug, PartialEq)]
pub enum StructuralVerdict {
    /// Uniquely best in every evaluated structure.
    InvariantBest(String),
    /// No action is uniquely best everywhere; each structure's leader is listed.
    NoInvariantBest(Vec<(String, Vec<String>)>),
    /// Best worst case over structures.
    WorstCaseChoice(String),
    /// Several actions share the best worst-case or Bayes score. Declaration order
    /// is not a choice.
    Tied(Vec<String>),
    /// Best probability-weighted value over the evaluated mass.
    BayesChoice {
        /// Chosen action.
        action: String,
        /// Probability mass that was evaluated; the rest is reported, not assumed.
        evaluated_mass: f64,
    },
    /// Per-structure answers only.
    ReportOnly,
    /// A claim would need evidence that is missing.
    InsufficientScience(String),
    /// Every action is excluded in some structure.
    NoAdmissibleAction,
}

/// Combined structural result.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralDecisionResult {
    /// Identity of the evaluated contract.
    pub contract_identity: String,
    /// Policy that combined the atoms.
    pub policy: StructuralPolicy,
    /// Per-structure outcomes.
    pub atoms: Vec<AtomSummary>,
    /// Per-action view across structures, in declaration order.
    pub actions: Vec<ActionStructure>,
    /// Probability mass of unidentified structures, when probabilities exist.
    pub unidentified_mass: Option<f64>,
    /// Probability mass of unevaluated structures, when probabilities exist.
    pub unevaluated_mass: Option<f64>,
    /// Probability mass of evaluated structures, when probabilities exist.
    pub evaluated_mass: Option<f64>,
    /// The verdict.
    pub verdict: StructuralVerdict,
}

/// Why a structural evaluation refused.
#[derive(Clone, Debug, PartialEq)]
pub enum StructuralError {
    /// The contract is invalid.
    Contract(DecisionContractError),
    /// No structures were supplied, or identities repeat or are blank.
    InvalidAtoms,
    /// Probabilities are given for some atoms but not all, are not finite or
    /// positive, or sum to more than one.
    InvalidProbabilities,
    /// `BayesOverStructures` needs genuine structure probabilities.
    ProbabilitiesRequired,
}

impl StructuralError {
    /// Registered runtime reason code.
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        match self {
            Self::Contract(_) | Self::InvalidProbabilities | Self::ProbabilitiesRequired => {
                antecedent_core::reason_code!("decision_contract_unsatisfied")
            }
            Self::InvalidAtoms => antecedent_core::reason_code!("invalid_argument"),
        }
    }
}

/// The contract each atom is scored with.
///
/// `MaximinOverStructures` scores each atom by expected utility and
/// `MinimaxOverIdentifiedSet` by expected loss. The structural policy that
/// combines those scores is [`StructureAggregator`].
fn per_atom_contract(contract: &DecisionContract) -> DecisionContract {
    let mut inner = contract.clone();
    match contract.criterion {
        DecisionCriterion::MaximinOverStructures => {
            inner.criterion = DecisionCriterion::PosteriorExpectedUtility;
        }
        DecisionCriterion::MinimaxOverIdentifiedSet => {
            // Utilities are read as losses; each member is scored by expected loss.
            inner.criterion = DecisionCriterion::PosteriorExpectedLoss;
        }
        _ => {}
    }
    inner
}

fn validate_atoms(atoms: &[StructuralAtom]) -> Result<bool, StructuralError> {
    if atoms.is_empty() {
        return Err(StructuralError::InvalidAtoms);
    }
    let mut ids = std::collections::HashSet::new();
    if atoms.iter().any(|a| a.id.trim().is_empty() || !ids.insert(a.id.as_str())) {
        return Err(StructuralError::InvalidAtoms);
    }
    let with = atoms.iter().filter(|a| a.probability.is_some()).count();
    if with == 0 {
        return Ok(false);
    }
    if with != atoms.len() {
        return Err(StructuralError::InvalidProbabilities);
    }
    let total: f64 = atoms.iter().filter_map(|a| a.probability).sum();
    if atoms.iter().any(|a| a.probability.is_some_and(|p| !p.is_finite() || p < 0.0))
        || total > 1.0 + 1e-9
    {
        return Err(StructuralError::InvalidProbabilities);
    }
    Ok(true)
}

/// Evaluate `contract` over structures and combine under its policy.
///
/// # Errors
/// Refuses an invalid contract, malformed atoms or probabilities, and Bayes
/// weighting without genuine probabilities.
#[allow(
    clippy::too_many_lines,
    reason = "one contract evaluated in order: bind, score each action under each structure, aggregate"
)]
pub fn evaluate_structural(
    contract: &DecisionContract,
    atoms: &[StructuralAtom],
) -> Result<StructuralDecisionResult, StructuralError> {
    let identity = contract.identity().map_err(StructuralError::Contract)?;
    let has_probabilities = validate_atoms(atoms)?;
    let aggregator = StructureAggregator::new(contract.criterion, contract.structural_policy);
    let inner = per_atom_contract(contract);
    if aggregator.kind() == AggregateKind::Bayes && !has_probabilities {
        return Err(StructuralError::ProbabilitiesRequired);
    }
    if aggregator.kind() == AggregateKind::Bayes
        && matches!(inner.criterion, DecisionCriterion::Quantile { .. } | DecisionCriterion::Regret)
    {
        // Quantiles and suprema of a mixture are not weighted averages of the
        // component functionals. No mixture-law evaluator is licensed here.
        return Err(StructuralError::Contract(DecisionContractError::InvalidParameter(
            "structural_criterion",
        )));
    }
    let sign = aggregator.sign();

    let mut summaries = Vec::with_capacity(atoms.len());
    for atom in atoms {
        let status = match &atom.evidence {
            AtomEvidence::Unidentified => AtomStatus::Unidentified,
            AtomEvidence::Unevaluated(reason) => AtomStatus::Unevaluated(reason.clone()),
            AtomEvidence::Evaluated(source) => match evaluate_contract(&inner, source) {
                Ok(result) => AtomStatus::Evaluated(Box::new(result)),
                Err(DecisionEvalError::Contract(error)) => {
                    return Err(StructuralError::Contract(error));
                }
                // A structure whose evidence cannot answer the contract keeps its
                // mass as unevaluated, with the registered reason.
                Err(other) => AtomStatus::Unevaluated(other.reason_code().to_owned()),
            },
        };
        summaries.push(AtomSummary { id: atom.id.clone(), probability: atom.probability, status });
    }

    let mass = |wanted: fn(&AtomStatus) -> bool| -> Option<f64> {
        has_probabilities.then(|| {
            summaries.iter().filter(|s| wanted(&s.status)).filter_map(|s| s.probability).sum()
        })
    };
    let evaluated_mass = mass(|s| matches!(s, AtomStatus::Evaluated(_)));
    let unidentified_mass = mass(|s| matches!(s, AtomStatus::Unidentified));
    let unassigned_mass = if has_probabilities {
        (1.0 - summaries.iter().filter_map(|s| s.probability).sum::<f64>()).max(0.0)
    } else {
        0.0
    };
    let unevaluated_mass =
        mass(|s| matches!(s, AtomStatus::Unevaluated(_))).map(|m| m + unassigned_mass);

    let n_actions = contract.actions.len();
    let mut actions: Vec<ActionStructure> = contract
        .actions
        .iter()
        .map(|a| ActionStructure {
            id: a.id.clone(),
            per_atom: vec![None; summaries.len()],
            range: None,
            weighted_value: None,
            mass_where_best: has_probabilities.then_some(0.0),
            excluded_in: Vec::new(),
        })
        .collect();
    let mut leaders: Vec<(String, Vec<String>)> = Vec::new();
    for (k, summary) in summaries.iter().enumerate() {
        let AtomStatus::Evaluated(result) = &summary.status else {
            continue;
        };
        for (i, outcome) in result.actions.iter().enumerate() {
            if outcome.admissible {
                actions[i].per_atom[k] = Some(outcome.value);
            } else if aggregator.kind() != AggregateKind::Bayes
                || summary.probability.is_some_and(|p| p > 0.0)
            {
                actions[i].excluded_in.push(summary.id.clone());
            }
        }
        match &result.verdict {
            Verdict::UniquelyOptimal(id) => {
                leaders.push((summary.id.clone(), vec![id.clone()]));
                if let (Some(slot), Some(p)) = (
                    actions
                        .iter_mut()
                        .find(|a| &a.id == id)
                        .and_then(|a| a.mass_where_best.as_mut()),
                    summary.probability,
                ) {
                    *slot += p;
                }
            }
            Verdict::Indistinguishable(ids) => {
                leaders.push((summary.id.clone(), ids.clone()));
            }
            Verdict::NoAdmissibleAction => {
                leaders.push((summary.id.clone(), Vec::new()));
            }
        }
    }
    for action in &mut actions {
        let values: Vec<f64> = action.per_atom.iter().flatten().copied().collect();
        if !values.is_empty() {
            let lo = values.iter().copied().fold(f64::INFINITY, f64::min);
            let hi = values.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            action.range = Some((lo, hi));
        }
        if has_probabilities {
            let weighted: f64 = summaries
                .iter()
                .zip(&action.per_atom)
                .filter_map(|(s, v)| Some(s.probability? * (*v)?))
                .sum();
            action.weighted_value = (!values.is_empty()).then_some(weighted);
        }
    }

    let any_evaluated = summaries.iter().any(|s| matches!(s.status, AtomStatus::Evaluated(_)));
    let all_evaluated = summaries.iter().all(|s| matches!(s.status, AtomStatus::Evaluated(_)));
    // An action is a candidate only if no structure excludes it.
    let candidates: Vec<usize> =
        (0..n_actions).filter(|i| actions[*i].excluded_in.is_empty()).collect();
    let action_ids: Vec<&str> = actions.iter().map(|action| action.id.as_str()).collect();
    let worst_score = |i: usize| -> f64 {
        actions[i].per_atom.iter().flatten().map(|v| sign * v).fold(f64::INFINITY, f64::min)
    };
    let bayes_score = |i: usize| sign * actions[i].weighted_value.unwrap_or(f64::NEG_INFINITY);
    let verdict = structural_verdict(aggregator.decide(
        &AggregateFacts {
            any_evaluated,
            all_resolved: all_evaluated,
            candidates: &candidates,
            action_ids: &action_ids,
            leaders: &leaders,
            unidentified_mass: unidentified_mass.unwrap_or(0.0),
            unevaluated_mass: unevaluated_mass.unwrap_or(0.0),
            evaluated_mass: evaluated_mass.unwrap_or(0.0),
            dropped_mass: 0.0,
        },
        worst_score,
        bayes_score,
    ));
    Ok(StructuralDecisionResult {
        contract_identity: identity,
        policy: contract.structural_policy,
        atoms: summaries,
        actions,
        unidentified_mass,
        unevaluated_mass,
        evaluated_mass,
        verdict,
    })
}

fn structural_verdict(decision: AggregateDecision) -> StructuralVerdict {
    match decision {
        AggregateDecision::Report => StructuralVerdict::ReportOnly,
        AggregateDecision::Insufficient(why) => StructuralVerdict::InsufficientScience(why.into()),
        // Support is not a structural filter; dropped mass is zero on this path.
        AggregateDecision::NoCandidate | AggregateDecision::SupportWithheld => {
            StructuralVerdict::NoAdmissibleAction
        }
        AggregateDecision::Disagree(leaders) => StructuralVerdict::NoInvariantBest(leaders),
        AggregateDecision::Choice(AggregateChoice::Invariant(id)) => {
            StructuralVerdict::InvariantBest(id)
        }
        AggregateDecision::Choice(AggregateChoice::WorstCase(id)) => {
            StructuralVerdict::WorstCaseChoice(id)
        }
        AggregateDecision::Choice(AggregateChoice::Bayes { action, evaluated_mass }) => {
            StructuralVerdict::BayesChoice { action, evaluated_mass }
        }
        AggregateDecision::Choice(AggregateChoice::Tied(ids)) => StructuralVerdict::Tied(ids),
    }
}
