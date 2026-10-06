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

use antecedent_io::distribution_artifact::DistributionArtifact;

use crate::decision_contract::{
    DecisionContract, DecisionContractError, DecisionCriterion, StructuralPolicy,
};
use crate::decision_eval::{DecisionEvalError, DecisionResult, Verdict, evaluate_contract};

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

/// How atoms are aggregated, after the criterion override.
#[derive(Clone, Copy, PartialEq)]
enum Aggregate {
    Invariant,
    WorstCase,
    Bayes,
    Report,
}

/// `+1` when a larger criterion value is better, `-1` when smaller is.
fn orientation(criterion: DecisionCriterion) -> f64 {
    match criterion {
        DecisionCriterion::PosteriorExpectedLoss
        | DecisionCriterion::ExpectedRegret
        | DecisionCriterion::Regret
        | DecisionCriterion::MinimaxOverIdentifiedSet => -1.0,
        _ => 1.0,
    }
}

fn per_atom_contract(contract: &DecisionContract) -> (DecisionContract, Aggregate) {
    let mut inner = contract.clone();
    let aggregate = match contract.criterion {
        DecisionCriterion::MaximinOverStructures => {
            inner.criterion = DecisionCriterion::PosteriorExpectedUtility;
            Aggregate::WorstCase
        }
        DecisionCriterion::MinimaxOverIdentifiedSet => {
            // Utilities are read as losses; each member is scored by expected loss.
            inner.criterion = DecisionCriterion::PosteriorExpectedLoss;
            Aggregate::WorstCase
        }
        _ => match contract.structural_policy {
            StructuralPolicy::RequireInvariantBestAction => Aggregate::Invariant,
            StructuralPolicy::Maximin => Aggregate::WorstCase,
            StructuralPolicy::BayesOverStructures => Aggregate::Bayes,
            StructuralPolicy::ReportOnly => Aggregate::Report,
        },
    };
    (inner, aggregate)
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
#[allow(clippy::too_many_lines)]
pub fn evaluate_structural(
    contract: &DecisionContract,
    atoms: &[StructuralAtom],
) -> Result<StructuralDecisionResult, StructuralError> {
    let identity = contract.identity().map_err(StructuralError::Contract)?;
    let has_probabilities = validate_atoms(atoms)?;
    let (inner, aggregate) = per_atom_contract(contract);
    if aggregate == Aggregate::Bayes && !has_probabilities {
        return Err(StructuralError::ProbabilitiesRequired);
    }
    let sign = orientation(inner.criterion);

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
    let unevaluated_mass = mass(|s| matches!(s, AtomStatus::Unevaluated(_)));

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
    let mut every_atom_unique = true;
    for (k, summary) in summaries.iter().enumerate() {
        let AtomStatus::Evaluated(result) = &summary.status else {
            continue;
        };
        for (i, outcome) in result.actions.iter().enumerate() {
            if outcome.admissible {
                actions[i].per_atom[k] = Some(outcome.value);
            } else {
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
                every_atom_unique = false;
                leaders.push((summary.id.clone(), ids.clone()));
            }
            Verdict::NoAdmissibleAction => {
                every_atom_unique = false;
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
    let verdict = if aggregate == Aggregate::Report {
        StructuralVerdict::ReportOnly
    } else if !any_evaluated {
        StructuralVerdict::InsufficientScience("no structure was evaluated".into())
    } else if candidates.is_empty() {
        StructuralVerdict::NoAdmissibleAction
    } else {
        match aggregate {
            Aggregate::Invariant => {
                if !all_evaluated {
                    StructuralVerdict::InsufficientScience(
                        "an unidentified or unevaluated structure leaves invariance unchecked"
                            .into(),
                    )
                } else if every_atom_unique
                    && leaders.windows(2).all(|pair| pair[0].1 == pair[1].1)
                    && candidates.iter().any(|i| actions[*i].id == leaders[0].1[0])
                {
                    StructuralVerdict::InvariantBest(leaders[0].1[0].clone())
                } else {
                    StructuralVerdict::NoInvariantBest(leaders)
                }
            }
            Aggregate::WorstCase => {
                if all_evaluated {
                    let worst = |i: usize| -> f64 {
                        actions[i]
                            .per_atom
                            .iter()
                            .flatten()
                            .map(|v| sign * v)
                            .fold(f64::INFINITY, f64::min)
                    };
                    let best = candidates
                        .iter()
                        .copied()
                        .max_by(|a, b| worst(*a).total_cmp(&worst(*b)).then(b.cmp(a)))
                        .expect("candidates are nonempty");
                    StructuralVerdict::WorstCaseChoice(actions[best].id.clone())
                } else {
                    StructuralVerdict::InsufficientScience(
                        "a worst case over structures needs every structure evaluated".into(),
                    )
                }
            }
            Aggregate::Bayes => {
                let evaluated = evaluated_mass.unwrap_or(0.0);
                if evaluated <= 0.0 {
                    StructuralVerdict::InsufficientScience(
                        "no evaluated structure has positive probability".into(),
                    )
                } else {
                    let score =
                        |i: usize| sign * actions[i].weighted_value.unwrap_or(f64::NEG_INFINITY);
                    let best = candidates
                        .iter()
                        .copied()
                        .max_by(|a, b| score(*a).total_cmp(&score(*b)).then(b.cmp(a)))
                        .expect("candidates are nonempty");
                    StructuralVerdict::BayesChoice {
                        action: actions[best].id.clone(),
                        evaluated_mass: evaluated,
                    }
                }
            }
            Aggregate::Report => StructuralVerdict::ReportOnly,
        }
    };
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
