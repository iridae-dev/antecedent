//! Evaluate a [`DecisionContract`] on an aligned joint distribution artifact.
//!
//! Every action's utility is computed per draw from the artifact's aligned
//! rows, so a nonlinear utility over several quantities sees genuine joint
//! realizations. Hard constraints exclude actions before the criterion ranks
//! the rest. The result distinguishes a uniquely best action from actions the
//! draws cannot tell apart, and from no admissible action; it never turns a
//! hard constraint into a penalty.
//!
//! Minimax over an identified set and maximin over structures need structure
//! inputs a single draw source does not carry; they refuse here rather than
//! run a generic scorer.

use antecedent_contract::ScientificQuantity;
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DrawAlignment,
};

use crate::decision_contract::{
    DecisionContract, DecisionContractError, DecisionCriterion, DecisionFunctional, SourceMode,
    SourceRepresentation, Tail,
};

/// Why an evaluation refused.
#[derive(Clone, Debug, PartialEq)]
pub enum DecisionEvalError {
    /// The contract is invalid.
    Contract(DecisionContractError),
    /// Rows are not paired joint realizations, so cross-quantity integration and
    /// state-by-state comparison are invalid.
    JointLawRequired {
        /// First action whose utility reads more than one input, when there is one.
        action: Option<String>,
        /// Alignment the source supplied, as its wire name.
        supplied_alignment: &'static str,
    },
    /// An action input has no matching coordinate in the source.
    QuantityNotFound {
        /// Action whose input is missing.
        action: String,
        /// Input position.
        input: usize,
    },
    /// A source coordinate is masked as unsupported.
    UnsupportedCoordinate {
        /// Action that reads it.
        action: String,
        /// Input position.
        input: usize,
    },
    /// The source's distribution meaning cannot answer an outcome-law input.
    MeaningMismatch {
        /// Action that reads it.
        action: String,
        /// Input position.
        input: usize,
    },
    /// The criterion needs structure or identified-set inputs this source lacks.
    StructureInputsRequired(&'static str),
    /// A utility expression produced a non-finite value.
    NonFiniteUtility {
        /// Action whose utility failed.
        action: String,
    },
    /// A mean-only source cannot answer this contract (a nonlinear utility, a
    /// hard constraint or a criterion that needs more than means).
    MeanSourceInsufficient {
        /// Representations that would have sufficed.
        needed: Vec<SourceRepresentation>,
    },
}

impl DecisionEvalError {
    /// Registered runtime reason code for this refusal.
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        match self {
            Self::Contract(_)
            | Self::NonFiniteUtility { .. }
            | Self::MeanSourceInsufficient { .. } => {
                antecedent_core::reason_code!("decision_contract_unsatisfied")
            }
            Self::JointLawRequired { .. } => antecedent_core::reason_code!("joint_law_required"),
            Self::QuantityNotFound { .. } | Self::UnsupportedCoordinate { .. } => {
                antecedent_core::reason_code!("quantity_semantics_mismatch")
            }
            Self::MeaningMismatch { .. } => {
                antecedent_core::reason_code!("distribution_meaning_mismatch")
            }
            Self::StructureInputsRequired(_) => {
                antecedent_core::reason_code!("route_not_supported")
            }
        }
    }
}

/// A hard constraint an action failed.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstraintExclusion {
    /// Constraint identity.
    pub constraint_id: String,
    /// Weighted probability that the constraint held.
    pub probability: f64,
    /// Probability the contract required.
    pub required: f64,
}

/// One action's outcome.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionOutcome {
    /// Semantic action identity.
    pub id: String,
    /// Whether every hard constraint held.
    pub admissible: bool,
    /// Constraints it failed; empty when admissible.
    pub exclusions: Vec<ConstraintExclusion>,
    /// Weighted expected utility.
    pub expected_utility: f64,
    /// The criterion's value for this action, in the criterion's own units.
    pub value: f64,
    /// Zero for an exact finite-law mean/probability; otherwise unavailable.
    /// Sampled artifacts do not declare independent draws or calibrated MC error.
    pub standard_error: Option<f64>,
    /// Expected regret against the best admissible action per draw; `None` when
    /// the action is excluded.
    pub expected_regret: Option<f64>,
    /// Largest per-draw regret; `None` when the action is excluded.
    pub max_regret: Option<f64>,
}

/// What the evaluation can claim about the choice.
#[derive(Clone, Debug, PartialEq)]
pub enum Verdict {
    /// One action has the best value under the supplied finite law. For sampled
    /// laws this is a point ranking, with no statistical optimality guarantee.
    UniquelyOptimal(String),
    /// These actions have equal finite-law values; no sampling comparison is claimed.
    Indistinguishable(Vec<String>),
    /// Every action violated a hard constraint.
    NoAdmissibleAction,
}

/// Where the draws came from, copied from the artifact identity.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct SourceReceipt {
    /// Provider object.
    pub provider_id: String,
    /// Data snapshot or exact-law identity.
    pub snapshot_id: String,
    /// RNG algorithm, seed and stream.
    pub rng_id: String,
    /// Causal-contract identity the draws were bound to.
    pub causal_contract_id: String,
}

/// Decision result with the evidence behind it.
#[derive(Clone, Debug, PartialEq)]
pub struct DecisionResult {
    /// Identity of the contract that was evaluated.
    pub contract_identity: String,
    /// Criterion that ranked the actions.
    pub criterion: DecisionCriterion,
    /// Per-action outcomes in declaration order.
    pub actions: Vec<ActionOutcome>,
    /// The verdict.
    pub verdict: Verdict,
    /// Expected value of perfect information over the admissible actions, when
    /// any action is admissible: `E[max_a U] - max_a E[U]`.
    pub evpi: Option<f64>,
    /// Number of draws.
    pub n_draws: usize,
    /// Kish effective sample size of the weights.
    pub effective_draws: f64,
    /// Source lineage.
    pub source: SourceReceipt,
    /// Assumptions the result stands on.
    pub assumptions: Vec<String>,
}

struct Table {
    /// `utilities[a][i]` for action `a`, draw `i`.
    utilities: Vec<Vec<f64>>,
    /// Normalized weights.
    weights: Vec<f64>,
}

/// Each source coordinate, parsed once. A wire that does not convert is skipped,
/// matching the old per-lookup `try_from` filter.
fn parsed_quantities(source: &DistributionArtifact) -> Vec<Option<ScientificQuantity>> {
    source
        .quantities()
        .iter()
        .map(|candidate| ScientificQuantity::try_from(candidate.clone()).ok())
        .collect()
}

fn locate(
    parsed: &[Option<ScientificQuantity>],
    action: &str,
    position: usize,
    wanted: &ScientificQuantity,
) -> Result<usize, DecisionEvalError> {
    parsed
        .iter()
        .position(|quantity| {
            quantity
                .as_ref()
                .is_some_and(|quantity| quantity.require_same_coordinate(wanted).is_ok())
        })
        .ok_or_else(|| DecisionEvalError::QuantityNotFound {
            action: action.to_owned(),
            input: position,
        })
}

/// Utilities of the contract's actions (or of action `only` alone) on `source`.
fn build_table(
    contract: &DecisionContract,
    source: &DistributionArtifact,
    only: Option<usize>,
) -> Result<Table, DecisionEvalError> {
    build_table_with(contract, source, only, &parsed_quantities(source))
}

fn build_table_with(
    contract: &DecisionContract,
    source: &DistributionArtifact,
    only: Option<usize>,
    parsed: &[Option<ScientificQuantity>],
) -> Result<Table, DecisionEvalError> {
    let alignment = source.metadata().identity.alignment;
    // The joint-law propagation rule decides whether the rows are paired.
    let rule_alignment = match alignment {
        DrawAlignment::Joint => antecedent_contract::InputAlignment::Aligned {
            alignment_id: "declared_pairing".to_owned(),
        },
        DrawAlignment::IndependentMarginals => {
            antecedent_contract::InputAlignment::IndependentMarginals
        }
    };
    if antecedent_contract::check_rows_pairable(&rule_alignment).is_err() {
        return Err(DecisionEvalError::JointLawRequired {
            action: contract
                .actions
                .iter()
                .find(|a| a.utility.inputs_used().len() > 1)
                .map(|a| a.id.clone()),
            supplied_alignment: match alignment {
                DrawAlignment::Joint => "joint",
                DrawAlignment::IndependentMarginals => "independent_marginals",
            },
        });
    }
    let meaning = antecedent_contract::DistributionMeaning::from(source.semantic());
    let [n_draws, width] = source.shape();
    let draws = source.draws();
    let mask = source.metadata().supported.as_deref();
    let mut utilities = Vec::with_capacity(contract.actions.len());
    for (action_index, action) in contract.actions.iter().enumerate() {
        if only.is_some_and(|index| index != action_index) {
            continue;
        }
        let mut columns = Vec::with_capacity(action.inputs.len());
        for (position, quantity) in action.inputs.iter().enumerate() {
            let column = locate(parsed, &action.id, position, quantity)?;
            if mask.is_some_and(|m| !m[column]) {
                return Err(DecisionEvalError::UnsupportedCoordinate {
                    action: action.id.clone(),
                    input: position,
                });
            }
            if quantity.functional_id == "outcome"
                && !meaning.answers_interventional_outcome_threshold()
            {
                return Err(DecisionEvalError::MeaningMismatch {
                    action: action.id.clone(),
                    input: position,
                });
            }
            columns.push(column);
        }
        let mut row_values = vec![0.0; columns.len()];
        let mut per_draw = Vec::with_capacity(n_draws);
        for draw in 0..n_draws {
            for (slot, column) in columns.iter().enumerate() {
                row_values[slot] = draws[draw * width + column];
            }
            let value =
                action.utility.evaluate(&row_values).map_err(DecisionEvalError::Contract)?;
            if !value.is_finite() {
                return Err(DecisionEvalError::NonFiniteUtility { action: action.id.clone() });
            }
            per_draw.push(value);
        }
        utilities.push(per_draw);
    }
    let raw: Vec<f64> = source.metadata().weights.clone().unwrap_or_else(|| vec![1.0; n_draws]);
    let total: f64 = raw.iter().sum();
    Ok(Table { utilities, weights: raw.into_iter().map(|w| w / total).collect() })
}

/// Expectation `Σ wᵢ vᵢ` under already-normalised weights; zero-weight draws contribute
/// nothing (the kernel reduction sanitises weights, utilities are validated finite).
///
/// # Errors
///
/// The kernel's length mismatch, carried as [`DecisionEvalError::Contract`] with the
/// same `weighted_sum length mismatch` text.
fn weighted_mean(values: &[f64], weights: &[f64]) -> Result<f64, DecisionEvalError> {
    antecedent_kernels::scalar::weighted_sum(values, weights).map_err(|error| {
        let message = match error {
            antecedent_kernels::ViewError::InvalidShape { message } => message,
            _ => "weighted_sum length mismatch",
        };
        DecisionEvalError::Contract(DecisionContractError::InvalidParameter(message))
    })
}

fn positive_mass_max(values: &[f64], weights: &[f64]) -> f64 {
    values.iter().zip(weights).filter(|(_, w)| **w > 0.0).map(|(v, _)| *v).fold(0.0, f64::max)
}

/// Evaluate `contract` on `source`.
///
/// # Errors
/// Refuses an invalid contract, non-joint draws, a missing or masked
/// coordinate, a source meaning that cannot answer an outcome-law input, and
/// criteria that need structure inputs.
#[allow(clippy::float_cmp, reason = "Ties of the finite-law criterion values")]
#[allow(
    clippy::too_many_lines,
    reason = "one contract evaluated in order: bind, score each action, aggregate, assemble the record"
)]
pub fn evaluate_contract(
    contract: &DecisionContract,
    source: &DistributionArtifact,
) -> Result<DecisionResult, DecisionEvalError> {
    let identity = contract.identity().map_err(DecisionEvalError::Contract)?;
    if matches!(
        contract.criterion,
        DecisionCriterion::MinimaxOverIdentifiedSet | DecisionCriterion::MaximinOverStructures
    ) {
        return Err(DecisionEvalError::StructureInputsRequired(
            "needs identified-set or structure inputs, not one draw source",
        ));
    }
    let parsed = parsed_quantities(source);
    let table = build_table_with(contract, source, None, &parsed)?;
    let n = source.n_draws();
    // An exact finite law has no sampling error: its draws are the distribution.
    let exact = source.metadata().calibration == DistributionCalibration::Exact;
    let weights = &table.weights;

    // Hard constraints exclude actions; they never enter the utility.
    let mut exclusions: Vec<Vec<ConstraintExclusion>> = vec![Vec::new(); contract.actions.len()];
    for constraint in &contract.constraints {
        for (index, action) in contract.actions.iter().enumerate() {
            if !constraint.applies_to.is_empty() && !constraint.applies_to.contains(&action.id) {
                continue;
            }
            let columns: Vec<usize> = action
                .inputs
                .iter()
                .enumerate()
                .map(|(p, q)| locate(&parsed, &action.id, p, q))
                .collect::<Result<_, _>>()?;
            let [_, width] = source.shape();
            let mut held = 0.0;
            let mut positive_violation = false;
            let mut row = vec![0.0; columns.len()];
            for (draw, weight) in weights.iter().enumerate() {
                for (slot, column) in columns.iter().enumerate() {
                    row[slot] = source.draws()[draw * width + column];
                }
                let value = constraint.expr.evaluate(&row).map_err(DecisionEvalError::Contract)?;
                if !value.is_finite() {
                    return Err(DecisionEvalError::NonFiniteUtility { action: action.id.clone() });
                }
                if value <= constraint.bound {
                    held += weight;
                } else if *weight > 0.0 {
                    positive_violation = true;
                }
            }
            let fails = if constraint.min_probability == 1.0 {
                positive_violation
            } else {
                held + MASS_TOLERANCE * held.max(constraint.min_probability)
                    < constraint.min_probability
            };
            if fails {
                exclusions[index].push(ConstraintExclusion {
                    constraint_id: constraint.id.clone(),
                    probability: held,
                    required: constraint.min_probability,
                });
            }
        }
    }
    let admissible: Vec<usize> =
        (0..contract.actions.len()).filter(|i| exclusions[*i].is_empty()).collect();

    // Regret is measured against the best admissible action in each draw.
    let loss = contract.criterion == DecisionCriterion::PosteriorExpectedLoss;
    let sign = if loss { -1.0 } else { 1.0 };
    let best_per_draw: Vec<f64> = (0..n)
        .map(|i| {
            admissible
                .iter()
                .map(|a| sign * table.utilities[*a][i])
                .fold(f64::NEG_INFINITY, f64::max)
        })
        .collect();
    let regret_of = |a: usize| -> Vec<f64> {
        (0..n).map(|i| best_per_draw[i] - sign * table.utilities[a][i]).collect()
    };

    let mut outcomes = Vec::with_capacity(contract.actions.len());
    for (index, action) in contract.actions.iter().enumerate() {
        let utility = &table.utilities[index];
        let is_admissible = exclusions[index].is_empty();
        let regret = if is_admissible { regret_of(index) } else { Vec::new() };
        let expected_utility = weighted_mean(utility, weights)?;
        let (value, standard_error) = match contract.criterion {
            DecisionCriterion::PosteriorExpectedUtility
            | DecisionCriterion::PosteriorExpectedLoss => (expected_utility, exact.then_some(0.0)),
            DecisionCriterion::ThresholdProbability { threshold } => {
                let indicator: Vec<f64> =
                    utility.iter().map(|u| f64::from(u32::from(*u >= threshold))).collect();
                (weighted_mean(&indicator, weights)?, exact.then_some(0.0))
            }
            DecisionCriterion::Quantile { p } => (
                left_inverse(utility, weights, p).ok_or_else(|| {
                    DecisionEvalError::NonFiniteUtility { action: action.id.clone() }
                })?,
                None,
            ),
            DecisionCriterion::ExpectedRegret if is_admissible => {
                (weighted_mean(&regret, weights)?, exact.then_some(0.0))
            }
            DecisionCriterion::Regret if is_admissible => {
                (positive_mass_max(&regret, weights), None)
            }
            _ => (f64::NAN, None),
        };
        if !expected_utility.is_finite()
            || (is_admissible
                && (!value.is_finite()
                    || regret.iter().zip(weights).any(|(r, w)| *w > 0.0 && !r.is_finite())))
        {
            return Err(DecisionEvalError::NonFiniteUtility { action: action.id.clone() });
        }
        outcomes.push(ActionOutcome {
            id: action.id.clone(),
            admissible: is_admissible,
            exclusions: std::mem::take(&mut exclusions[index]),
            expected_utility,
            value,
            standard_error,
            expected_regret: if is_admissible {
                Some(weighted_mean(&regret, weights)?)
            } else {
                None
            },
            max_regret: is_admissible.then(|| positive_mass_max(&regret, weights)),
        });
    }

    // Rank admissible actions by the criterion. `larger` ranks higher.
    let larger = |a: &ActionOutcome, b: &ActionOutcome| -> std::cmp::Ordering {
        let key = |x: &ActionOutcome| match contract.criterion {
            DecisionCriterion::PosteriorExpectedLoss
            | DecisionCriterion::Regret
            | DecisionCriterion::ExpectedRegret => -x.value,
            _ => x.value,
        };
        key(a).total_cmp(&key(b))
    };
    let leader = admissible
        .iter()
        .copied()
        .max_by(|a, b| larger(&outcomes[*a], &outcomes[*b]).then(b.cmp(a)));
    let verdict = match leader {
        None => Verdict::NoAdmissibleAction,
        Some(best) => {
            let mut tied = vec![contract.actions[best].id.clone()];
            for other in admissible.iter().copied().filter(|a| *a != best) {
                // These are functionals of the supplied finite law. A sampled
                // artifact carries no draw-dependence or calibrated ranking contract.
                let indistinct = outcomes[best].value == outcomes[other].value;
                if indistinct {
                    tied.push(contract.actions[other].id.clone());
                }
            }
            if tied.len() == 1 {
                Verdict::UniquelyOptimal(tied.remove(0))
            } else {
                Verdict::Indistinguishable(tied)
            }
        }
    };
    let evpi = if admissible.is_empty() {
        None
    } else {
        let perfect = weighted_mean(&best_per_draw, weights)?;
        let mut best_mean = f64::NEG_INFINITY;
        for action in &admissible {
            best_mean = best_mean.max(sign * weighted_mean(&table.utilities[*action], weights)?);
        }
        Some(perfect - best_mean)
    };
    if evpi.is_some_and(|value| !value.is_finite()) {
        return Err(DecisionEvalError::NonFiniteUtility {
            action: contract.actions[admissible[0]].id.clone(),
        });
    }
    let identity_meta = &source.metadata().identity;
    Ok(DecisionResult {
        contract_identity: identity,
        criterion: contract.criterion,
        actions: outcomes,
        verdict,
        evpi,
        n_draws: n,
        effective_draws: antecedent_kernels::kish_ess(weights),
        source: SourceReceipt {
            provider_id: identity_meta.provider_id.clone(),
            snapshot_id: identity_meta.snapshot_id.clone(),
            rng_id: identity_meta.rng_id.clone(),
            causal_contract_id: identity_meta.causal_contract_id.clone(),
        },
        assumptions: vec![
            "draws are aligned joint realizations of every input quantity".into(),
            "hard constraints exclude actions and never enter the utility".into(),
            if exact {
                "the source is an exact finite law, so values carry no sampling error".to_owned()
            } else {
                "point ranking of the supplied empirical law only; Monte Carlo error and ranking guarantees are not licensed".to_owned()
            },
        ],
    })
}

/// A functional of one action's utility, with how it was obtained.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct FunctionalValue {
    /// The functional's value, in the units of the utility (a probability for
    /// [`DecisionFunctional::Probability`], the squared utility unit for
    /// [`DecisionFunctional::Variance`]).
    pub value: f64,
    /// `Some(0.0)` for an exact finite law; `None` for sampled laws, whose
    /// draw-dependence and Monte Carlo error are not certified by the artifact.
    pub standard_error: Option<f64>,
    /// The source representation the functional was answered from.
    pub source_mode: SourceMode,
}

/// Relative roundoff tolerance used when comparing cumulative weights with a level `p`.
const MASS_TOLERANCE: f64 = 8.0 * f64::EPSILON;

/// Left inverse `inf { x : F(x) >= p }` of the weighted empirical CDF.
fn left_inverse(values: &[f64], weights: &[f64], p: f64) -> Option<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut cumulative = 0.0;
    for &index in &order {
        if weights[index] == 0.0 {
            continue;
        }
        cumulative += weights[index];
        if cumulative >= p || (cumulative - p).abs() <= MASS_TOLERANCE * cumulative.max(p) {
            return Some(values[index]);
        }
    }
    order.iter().rev().find(|&&i| weights[i] > 0.0).map(|&index| values[index])
}

/// Mean of the lowest or highest probability mass `p`, splitting the boundary
/// atom fractionally.
fn fractional_tail_mean(values: &[f64], weights: &[f64], p: f64, tail: Tail) -> f64 {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    if tail == Tail::Upper {
        order.reverse();
    }
    let mut accumulated = 0.0;
    let mut total = 0.0;
    for &index in &order {
        let remaining = p - accumulated;
        if remaining <= 0.0 {
            break;
        }
        let taken = weights[index].min(remaining);
        total += taken * values[index];
        accumulated += taken;
    }
    total / p
}

/// Compute `functional` of the named action's utility on an aligned joint
/// distribution artifact.
///
/// The utility is evaluated per aligned draw exactly as in [`evaluate_contract`]
/// (joint alignment, coordinates located by [`ScientificQuantity`] coordinate
/// identity, distribution meaning checked, weights normalized to sum one). The
/// functional is then taken of the resulting finite weighted law:
///
/// - `Expectation` and `ExpectedUtility`: the weighted mean `sum w_i x_i`.
/// - `Variance`: the weighted population variance `sum w_i (x_i - mean)^2` of
///   the finite law, not an unbiased sample variance.
/// - `Probability { threshold, tail }`: `Lower` is `P(x <= threshold)` and
///   `Upper` is `P(x >= threshold)`; both include the atom at the threshold.
/// - `Quantile { p }`: the left inverse `F^-1(p) = inf { x : F(x) >= p }` of the
///   weighted empirical CDF, the smallest value whose cumulative weight reaches
///   `p`. Cumulative weights are compared to `p` with a relative roundoff tolerance of
///   eight machine epsilons so a level that falls exactly on an atom boundary selects that atom.
/// - `TailExpectation { p, tail }`: the fractional-boundary tail mean. Draws are
///   sorted ascending for `Lower` and descending for `Upper`; weights accumulate
///   until mass `p` is reached, taking only the needed fraction of the boundary
///   atom, and the weighted sum is divided by `p`. This is the explicitly
///   selected tail rule.
///
/// `standard_error` is `Some(0.0)` for an exact finite law; Monte Carlo
/// error is withheld for sampled laws (`None`).
///
/// # Errors
/// Refuses an invalid contract, an unknown action
/// (`Contract(UnknownAction)`), a level `p` outside `(0, 1)` or a non-finite
/// threshold (`Contract(InvalidParameter("functional"))`), non-joint draws, a
/// missing or masked coordinate, a source meaning that cannot answer an
/// outcome-law input, and a non-finite utility.
pub fn evaluate_functional(
    contract: &DecisionContract,
    action_id: &str,
    functional: DecisionFunctional,
    source: &DistributionArtifact,
) -> Result<FunctionalValue, DecisionEvalError> {
    let index = resolve_action(contract, action_id)?;
    check_functional(functional)?;
    ActionLaw::from_table(index, build_table(contract, source, None)?, source)
        .functional(contract, functional)
}

/// Validate `contract` and find `action_id` in it.
fn resolve_action(
    contract: &DecisionContract,
    action_id: &str,
) -> Result<usize, DecisionEvalError> {
    contract.validate().map_err(DecisionEvalError::Contract)?;
    contract.actions.iter().position(|a| a.id == action_id).ok_or_else(|| {
        DecisionEvalError::Contract(DecisionContractError::UnknownAction(action_id.to_owned()))
    })
}

/// Refuse a functional with an out-of-range level or a non-finite threshold.
fn check_functional(functional: DecisionFunctional) -> Result<(), DecisionEvalError> {
    let level_ok = |p: f64| p.is_finite() && p > 0.0 && p < 1.0;
    let valid = match functional {
        DecisionFunctional::Expectation
        | DecisionFunctional::ExpectedUtility
        | DecisionFunctional::Variance => true,
        DecisionFunctional::Probability { threshold, .. } => threshold.is_finite(),
        DecisionFunctional::Quantile { p } | DecisionFunctional::TailExpectation { p, .. } => {
            level_ok(p)
        }
    };
    if valid {
        Ok(())
    } else {
        Err(DecisionEvalError::Contract(DecisionContractError::InvalidParameter("functional")))
    }
}

/// One action's finite utility law on one source: the per-draw utilities and normalized
/// weights that several functionals can be taken of without re-binding the draws.
pub(crate) struct ActionLaw {
    index: usize,
    utility: Vec<f64>,
    weights: Vec<f64>,
    exact: bool,
}

impl ActionLaw {
    fn from_table(index: usize, mut table: Table, source: &DistributionArtifact) -> Self {
        let slot = if table.utilities.len() == 1 { 0 } else { index };
        Self {
            index,
            utility: table.utilities.swap_remove(slot),
            weights: table.weights,
            exact: source.metadata().calibration == DistributionCalibration::Exact,
        }
    }

    /// The law of `action_id`'s utility. Only that action's column is evaluated, so a
    /// refusal belonging to another action of `contract` is not raised here.
    ///
    /// # Errors
    /// The refusals [`evaluate_functional`] raises before the functional is taken.
    pub(crate) fn of_action(
        contract: &DecisionContract,
        action_id: &str,
        source: &DistributionArtifact,
    ) -> Result<Self, DecisionEvalError> {
        let index = resolve_action(contract, action_id)?;
        Ok(Self::from_table(index, build_table(contract, source, Some(index))?, source))
    }

    /// `functional` of this law, as [`evaluate_functional`] defines it.
    ///
    /// # Errors
    /// The functional's source requirement is unmet, or its value is not finite.
    pub(crate) fn functional(
        &self,
        contract: &DecisionContract,
        functional: DecisionFunctional,
    ) -> Result<FunctionalValue, DecisionEvalError> {
        let action = &contract.actions[self.index];
        let utility = &self.utility;
        let weights = &self.weights;
        let source_mode = functional
            .requirement(&action.utility)
            .check(&[SourceRepresentation::JointDraws])
            .map_err(DecisionEvalError::Contract)?;
        let mean = weighted_mean(utility, weights)?;
        let (value, sampled_error) = match functional {
            DecisionFunctional::Expectation | DecisionFunctional::ExpectedUtility => {
                (mean, self.exact.then_some(0.0))
            }
            DecisionFunctional::Variance => {
                let variance: f64 = utility
                    .iter()
                    .zip(weights)
                    .filter(|(_, w)| **w > 0.0)
                    .map(|(v, w)| w * (v - mean) * (v - mean))
                    .sum();
                (variance, None)
            }
            DecisionFunctional::Probability { threshold, tail } => {
                let mass: f64 = utility
                    .iter()
                    .zip(weights)
                    .filter(|(v, _)| match tail {
                        Tail::Lower => **v <= threshold,
                        Tail::Upper => **v >= threshold,
                    })
                    .map(|(_, w)| w)
                    .sum();
                (mass, None)
            }
            DecisionFunctional::Quantile { p } => (
                left_inverse(utility, weights, p).ok_or_else(|| {
                    DecisionEvalError::NonFiniteUtility { action: action.id.clone() }
                })?,
                None,
            ),
            DecisionFunctional::TailExpectation { p, tail } => {
                (fractional_tail_mean(utility, weights, p, tail), None)
            }
        };
        if !value.is_finite() {
            return Err(DecisionEvalError::NonFiniteUtility { action: action.id.clone() });
        }
        let standard_error = if self.exact { Some(0.0) } else { sampled_error };
        Ok(FunctionalValue { value, standard_error, source_mode })
    }
}

/// A source that supplies only one mean per coordinate, such as an external
/// response grid. A mean carries no distribution and no pairing, so it can answer
/// only an expectation of an affine utility.
#[derive(Clone, Debug, PartialEq)]
pub struct MeanSource {
    /// Coordinates, one per mean, in the same order as `means`.
    pub coordinates: Vec<ScientificQuantity>,
    /// Mean of each coordinate.
    pub means: Vec<f64>,
    /// Provider object that supplied the means.
    pub provider_id: String,
    /// Data snapshot or exact-request identity the means were computed from.
    pub snapshot_id: String,
    /// Causal-contract identity the means were bound to.
    pub causal_contract_id: String,
    /// RNG identity; there is no sampling in a mean grid, so this names that.
    pub rng_id: String,
}

fn mean_insufficient(needed: Vec<SourceRepresentation>) -> DecisionEvalError {
    DecisionEvalError::MeanSourceInsufficient { needed }
}

/// Evaluate `contract` on a [`MeanSource`].
///
/// Exact only for affine utilities, whose expectation is the utility of the
/// means. The result is the expected utility at the supplied means, which may be
/// posterior means or point estimates; the criterion name does not assert a
/// posterior. No sampling error is claimed (`standard_error` is `None`), regret and
/// the value of perfect information are unavailable, and the verdict is
/// indistinguishable only on exactly equal values.
///
/// # Errors
/// Refuses an invalid contract, a criterion other than expected utility or
/// expected loss, any hard constraint, a nonlinear utility, an unknown
/// coordinate, a coordinate named twice (the first would silently win), an input whose functional is `outcome` (a mean grid is not an
/// outcome law), and a non-finite utility.
#[allow(
    clippy::too_many_lines,
    reason = "one contract evaluated in order: bind, score each action, aggregate, assemble the record"
)]
pub fn evaluate_contract_on_means(
    contract: &DecisionContract,
    source: &MeanSource,
) -> Result<DecisionResult, DecisionEvalError> {
    let identity = contract.identity().map_err(DecisionEvalError::Contract)?;
    if source.coordinates.len() != source.means.len() || source.means.iter().any(|m| !m.is_finite())
    {
        return Err(DecisionEvalError::Contract(DecisionContractError::InvalidDeclaration(
            "a mean source has one finite mean per coordinate",
        )));
    }
    for (index, coordinate) in source.coordinates.iter().enumerate() {
        if source.coordinates[..index]
            .iter()
            .any(|earlier| earlier.require_same_coordinate(coordinate).is_ok())
        {
            return Err(DecisionEvalError::Contract(DecisionContractError::InvalidDeclaration(
                "a mean source names each coordinate once",
            )));
        }
    }
    if !matches!(
        contract.criterion,
        DecisionCriterion::PosteriorExpectedUtility | DecisionCriterion::PosteriorExpectedLoss
    ) {
        return Err(mean_insufficient(vec![SourceRepresentation::JointDraws]));
    }
    if !contract.constraints.is_empty() {
        // A mean cannot answer a probability constraint.
        return Err(mean_insufficient(vec![
            SourceRepresentation::JointDraws,
            SourceRepresentation::MarginalDraws,
        ]));
    }
    let mut outcomes = Vec::with_capacity(contract.actions.len());
    for action in &contract.actions {
        DecisionFunctional::ExpectedUtility
            .requirement(&action.utility)
            .check(&[SourceRepresentation::Mean])
            .map_err(|error| match error {
                DecisionContractError::MissingSource { needed } => mean_insufficient(needed),
                other => DecisionEvalError::Contract(other),
            })?;
        if !action.utility.is_affine() {
            return Err(mean_insufficient(vec![SourceRepresentation::JointDraws]));
        }
        let mut inputs = Vec::with_capacity(action.inputs.len());
        for (position, wanted) in action.inputs.iter().enumerate() {
            let column = source
                .coordinates
                .iter()
                .position(|candidate| candidate.require_same_coordinate(wanted).is_ok())
                .ok_or_else(|| DecisionEvalError::QuantityNotFound {
                    action: action.id.clone(),
                    input: position,
                })?;
            if wanted.functional_id == "outcome" {
                return Err(DecisionEvalError::MeaningMismatch {
                    action: action.id.clone(),
                    input: position,
                });
            }
            inputs.push(source.means[column]);
        }
        let expected_utility =
            action.utility.evaluate(&inputs).map_err(DecisionEvalError::Contract)?;
        if !expected_utility.is_finite() {
            return Err(DecisionEvalError::NonFiniteUtility { action: action.id.clone() });
        }
        outcomes.push(ActionOutcome {
            id: action.id.clone(),
            admissible: true,
            exclusions: Vec::new(),
            expected_utility,
            value: expected_utility,
            standard_error: None,
            expected_regret: None,
            max_regret: None,
        });
    }
    let loss = contract.criterion == DecisionCriterion::PosteriorExpectedLoss;
    let key = |outcome: &ActionOutcome| if loss { -outcome.value } else { outcome.value };
    let best = (0..outcomes.len())
        .max_by(|a, b| key(&outcomes[*a]).total_cmp(&key(&outcomes[*b])).then(b.cmp(a)))
        .unwrap_or(0);
    let mut tied = vec![outcomes[best].id.clone()];
    for (other, outcome) in outcomes.iter().enumerate() {
        // Total equality of the values: `-0.0` and `0.0` are the same utility.
        #[allow(clippy::float_cmp, reason = "exact equality is the declared tie rule")]
        let equal = outcome.value == outcomes[best].value;
        if other != best && equal {
            tied.push(outcome.id.clone());
        }
    }
    let verdict = if tied.len() == 1 {
        Verdict::UniquelyOptimal(tied.remove(0))
    } else {
        Verdict::Indistinguishable(tied)
    };
    Ok(DecisionResult {
        contract_identity: identity,
        criterion: contract.criterion,
        actions: outcomes,
        verdict,
        evpi: None,
        n_draws: 0,
        effective_draws: 0.0,
        source: SourceReceipt {
            provider_id: source.provider_id.clone(),
            snapshot_id: source.snapshot_id.clone(),
            rng_id: source.rng_id.clone(),
            causal_contract_id: source.causal_contract_id.clone(),
        },
        assumptions: vec![
            "the source supplies means only; regret, value of perfect information and hard \
             constraints are unavailable"
                .into(),
            "the result is exact only because every utility is affine in its inputs".into(),
            "the means are taken as supplied (posterior means or point estimates); the criterion \
             name does not assert a posterior"
                .into(),
            "no sampling error is claimed; indistinguishable means exactly equal values".into(),
        ],
    })
}

#[cfg(test)]
mod tests {
    use antecedent_contract::QuantityRole;
    use antecedent_io_core::distribution_artifact::{
        DistributionIdentity, DistributionMetadata, DistributionProvenance, DistributionTrust,
    };
    use antecedent_io_core::quantity_wire::DistributionMeaningWire;

    use super::*;
    use crate::decision_contract::{
        ActionKind, DecisionAction, StructuralPolicy, Tail, UtilityExpr,
    };

    fn quantity(variable: &str) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: variable.into(),
            variable_name: variable.into(),
            role: QuantityRole::Outcome,
            units: "units".into(),
            population_id: "target".into(),
            regime_id: "do(a=1)".into(),
            horizon: 0,
            functional_id: "outcome".into(),
            conditioning: vec![],
            transform_id: "identity".into(),
        }
    }

    /// Two actions on a weighted, sampled five-draw law.
    fn fixture() -> (DecisionContract, DistributionArtifact) {
        let columns = [quantity("p"), quantity("q")];
        let identity = DistributionIdentity::new(
            DistributionMeaningWire::InterventionalPredictive,
            &columns,
            DrawAlignment::Joint,
            DistributionProvenance {
                source_id: "s".into(),
                provider_id: "p".into(),
                rng_id: "r".into(),
                snapshot_id: "n".into(),
                causal_contract_id: "c".into(),
            },
        )
        .unwrap();
        let draws = vec![1.0, 4.0, 3.0, 1.0, 2.0, 2.0, 0.5, 6.0, 3.0, 3.0];
        let artifact = DistributionArtifact::new(
            DistributionMetadata {
                version: 1,
                identity,
                axes: ["draw".into(), "quantity".into()],
                shape: [5, 2],
                weights: Some(vec![1.0, 2.0, 0.0, 3.0, 4.0]),
                supported: None,
                calibration: DistributionCalibration::Unmeasured,
                trust: DistributionTrust::Unverified,
                legacy_posterior: None,
                legacy_bindings: None,
            },
            draws,
        )
        .unwrap();
        let action = |id: &str, column: usize| DecisionAction {
            id: id.into(),
            kind: ActionKind::Intervention,
            inputs: vec![columns[column].clone()],
            utility: UtilityExpr::Input(0),
        };
        let contract = DecisionContract {
            actions: vec![action("a", 0), action("b", 1)],
            utility_units: "units".into(),
            criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: vec![],
            target_population: "target".into(),
            horizon: 0,
            structural_policy: StructuralPolicy::ReportOnly,
        };
        (contract, artifact)
    }

    /// A law built once answers every functional bit-for-bit as the one-shot engine does.
    #[test]
    fn action_law_matches_evaluate_functional_for_every_functional() {
        let (contract, artifact) = fixture();
        let functionals = [
            DecisionFunctional::Expectation,
            DecisionFunctional::ExpectedUtility,
            DecisionFunctional::Variance,
            DecisionFunctional::Probability { threshold: 2.0, tail: Tail::Lower },
            DecisionFunctional::Probability { threshold: 2.0, tail: Tail::Upper },
            DecisionFunctional::Quantile { p: 0.35 },
            DecisionFunctional::TailExpectation { p: 0.4, tail: Tail::Lower },
            DecisionFunctional::TailExpectation { p: 0.4, tail: Tail::Upper },
        ];
        for id in ["a", "b"] {
            let law = ActionLaw::of_action(&contract, id, &artifact).unwrap();
            for functional in functionals {
                let once = evaluate_functional(&contract, id, functional, &artifact).unwrap();
                let hoisted = law.functional(&contract, functional).unwrap();
                assert_eq!(hoisted.value.to_bits(), once.value.to_bits(), "{id} {functional:?}");
                assert_eq!(hoisted.standard_error, once.standard_error);
                assert_eq!(hoisted.source_mode, once.source_mode);
            }
        }
    }
}
