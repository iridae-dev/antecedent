//! Generalized finite inverse decision query (F7).
//!
//! The question is "which of these declared actions satisfy these constraints?"
//! over a finite, ordered action grid. Three constraint kinds run through the
//! same typed functional engine ([`evaluate_functional`]) that evaluates every
//! other decision functional, on a forward claim that actually supplies the law
//! the functional needs:
//!
//! - target mean: `E[U_a] >= target` or `<= target` ([`DecisionFunctional::ExpectedUtility`]);
//! - target quantile: `Q_p(U_a) >= target` or `<= target`, with `Q_p` the engine's
//!   left inverse `inf { x : F(x) >= p }` of the weighted CDF (not an
//!   interpolated or midpoint quantile);
//! - probability threshold: `P(U_a <= t) >= q` (lower tail) or `P(U_a >= t) >= q`
//!   (upper tail), both including the atom at `t`, as the engine defines them.
//!
//! No quantile or probability is computed in this module: it only compares the
//! engine's value with the target, within a declared non-negative `tolerance`.
//!
//! # Conventions
//!
//! - **Action grid and order.** The caller declares the grid as ordered action
//!   ids of the contract. The order is the multiple-action tie rule's order, not
//!   an id sort. The grid is finite and enumerated; nothing is interpolated.
//! - **Constraints** are a conjunction. An action is `Infeasible` as soon as one
//!   constraint definitely fails, `Unsupported` or `Unevaluated` when none fails
//!   but one cannot be answered, otherwise `Feasible`.
//! - **Multiple feasible actions.** [`SelectionRule::FirstInGridOrder`] selects
//!   the earliest feasible action in the declared order, [`SelectionRule::LastInGridOrder`]
//!   the latest, and [`SelectionRule::RequireUnique`] selects only when exactly
//!   one action is feasible. The full feasible list is always reported. A
//!   selection is `certified` only when every action the rule passes over is
//!   definitely `Infeasible`, never merely unevaluated or unsupported.
//! - **Law requirements.** A nonlinear utility, a quantile or a probability
//!   needs aligned joint draws. Independent marginals and mean-only claims
//!   refuse with the engine's own refusal; a mean never yields an outcome
//!   probability or quantile.
//! - **Uncertainty.** Point feasibility is a statement about the supplied
//!   forward law only. Interval-region, identified-set, all-scenario and
//!   posterior-probability feasibility are separate fields and are `None` unless
//!   that kind of evidence was supplied; none is derived from another. No
//!   coverage or calibration claim is made here.
//! - **Scenarios.** Structural scenarios are [`StructuralAtom`]s. A conflicting
//!   answer across scenarios is `StructurallyAmbiguous`; an unidentified
//!   scenario is `Unidentified`; an unevaluated one is `Unevaluated`. Their
//!   probability mass is retained and never renormalized away.
//! - **Continuous domains.** A grid sampled from a continuous action space can
//!   find a feasible point, but that is never reported as a global feasibility
//!   claim; continuous optimization is separately scoped.
//!
//! The 2.2 finite-enumeration baseline stays available as
//! [`finite_enumeration_baseline`], independent of the engine.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::HashSet;

use antecedent_core::{ExternalRefusal, ScientificQuantity};
use antecedent_io::distribution_artifact::DistributionArtifact;

use crate::decision_contract::{
    DecisionAction, DecisionContract, DecisionContractError, DecisionCriterion, DecisionFunctional,
    SourceRepresentation, Tail,
};
use crate::decision_eval::{
    DecisionEvalError, MeanSource, SourceReceipt, evaluate_contract_on_means, evaluate_functional,
};
use crate::decision_structural::{AtomEvidence, StructuralAtom};

/// Most actions one query may enumerate.
pub const MAX_INVERSE_GRID: usize = 4096;

/// What a result is and is not, carried on every report.
pub const INVERSE_QUERY_SCOPE_NOTE: &str = "Feasibility is relative to the declared finite action grid and the supplied forward claims; an action outside the grid is unresolved, not excluded. Point, interval-region, identified-set, all-scenario and posterior-probability feasibility are separate fields and none is derived from another. No coverage or calibration claim is made, and one found point is never a global feasibility claim over a continuous domain.";

const BUDGET_REASON: &str = "inverse_query.budget_exhausted";
const STAGE: &str = "inverse_query";

/// Which side of the target is the goal.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum Comparison {
    /// The functional value is at least the target.
    AtLeast,
    /// The functional value is at most the target.
    AtMost,
}

impl Comparison {
    /// Stable `snake_case` spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::AtLeast => "at_least",
            Self::AtMost => "at_most",
        }
    }

    fn holds(self, value: f64, target: f64, tolerance: f64) -> bool {
        match self {
            Self::AtLeast => value >= target - tolerance,
            Self::AtMost => value <= target + tolerance,
        }
    }
}

/// One typed constraint on an action's forward utility law.
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum InverseConstraint {
    /// `E[U_a]` against `target`.
    TargetMean {
        /// Target for the mean.
        target: f64,
        /// Side of the target that is feasible.
        comparison: Comparison,
    },
    /// The left-inverse `p`-quantile of `U_a` against `target`.
    TargetQuantile {
        /// Level in `(0, 1)`.
        p: f64,
        /// Target for the quantile.
        target: f64,
        /// Side of the target that is feasible.
        comparison: Comparison,
    },
    /// `P(U_a tail outcome_threshold)` against `probability`.
    ProbabilityThreshold {
        /// Outcome threshold `t`.
        outcome_threshold: f64,
        /// `Lower` is `P(U <= t)`, `Upper` is `P(U >= t)`.
        tail: Tail,
        /// Probability target in `[0, 1]`.
        probability: f64,
        /// Side of the probability target that is feasible.
        comparison: Comparison,
    },
}

impl InverseConstraint {
    /// The shared-engine functional this constraint evaluates.
    #[must_use]
    pub fn functional(&self) -> DecisionFunctional {
        match *self {
            Self::TargetMean { .. } => DecisionFunctional::ExpectedUtility,
            Self::TargetQuantile { p, .. } => DecisionFunctional::Quantile { p },
            Self::ProbabilityThreshold { outcome_threshold, tail, .. } => {
                DecisionFunctional::Probability { threshold: outcome_threshold, tail }
            }
        }
    }

    fn target(&self) -> (f64, Comparison) {
        match *self {
            Self::TargetMean { target, comparison }
            | Self::TargetQuantile { target, comparison, .. } => (target, comparison),
            Self::ProbabilityThreshold { probability, comparison, .. } => (probability, comparison),
        }
    }

    fn valid(&self) -> bool {
        match *self {
            Self::TargetMean { target, .. } => target.is_finite(),
            Self::TargetQuantile { p, target, .. } => {
                p.is_finite() && p > 0.0 && p < 1.0 && target.is_finite()
            }
            Self::ProbabilityThreshold { outcome_threshold, probability, .. } => {
                outcome_threshold.is_finite()
                    && probability.is_finite()
                    && (0.0..=1.0).contains(&probability)
            }
        }
    }
}

/// Feasibility classification. `StructurallyAmbiguous` and `Unidentified` apply
/// to scenario, set and region fields; a single supplied law yields the other four.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum FeasibilityStatus {
    /// Every constraint definitely holds.
    Feasible,
    /// At least one constraint definitely fails.
    Infeasible,
    /// A needed coordinate is masked as unsupported.
    Unsupported,
    /// The claim was not evaluated (absent coordinate, budget, missing evidence).
    Unevaluated,
    /// Scenarios, set members or region endpoints give conflicting answers.
    StructurallyAmbiguous,
    /// A scenario does not identify the quantity.
    Unidentified,
}

impl FeasibilityStatus {
    /// Stable `snake_case` spelling.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Feasible => "feasible",
            Self::Infeasible => "infeasible",
            Self::Unsupported => "unsupported",
            Self::Unevaluated => "unevaluated",
            Self::StructurallyAmbiguous => "structurally_ambiguous",
            Self::Unidentified => "unidentified",
        }
    }
}

/// How several feasible actions are resolved.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SelectionRule {
    /// The earliest point-feasible action in the declared grid order.
    FirstInGridOrder,
    /// The latest point-feasible action in the declared grid order.
    LastInGridOrder,
    /// Select only when exactly one action is point-feasible.
    RequireUnique,
}

/// Whether the grid is a finite enumeration or a sample of a continuous domain.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum GridScope {
    /// The declared finite set is the whole question.
    FiniteEnumeration,
    /// Points sampled from a continuous action space; no global claim is licensed.
    ContinuousSample,
}

/// A forward claim that supplies the law (or only the means) of every action's inputs.
#[derive(Clone, Debug)]
pub enum ForwardClaim {
    /// An aligned joint distribution artifact.
    Law(Box<DistributionArtifact>),
    /// Means only: answers an affine target mean and nothing else.
    Means(MeanSource),
}

/// Bounding claims of a published interval region. The endpoints are laws the
/// caller asserts bound the functional; that assertion is not verified here.
#[derive(Clone, Debug)]
pub struct IntervalRegion {
    /// Lower-endpoint claim.
    pub lower: ForwardClaim,
    /// Upper-endpoint claim.
    pub upper: ForwardClaim,
    /// Declared: the endpoint laws bound the functional over the whole region.
    /// Without it a feasible or infeasible endpoint answer is `Unevaluated`.
    pub endpoints_bound_functional: bool,
}

/// Member laws of an identified set.
#[derive(Clone, Debug)]
pub struct IdentifiedSet {
    /// Member laws, by stable identity. Probabilities, if present, are ignored.
    pub members: Vec<StructuralAtom>,
    /// Declared: the members enumerate the whole identified set.
    /// Without it a uniform answer is `Unevaluated`.
    pub exhaustive: bool,
}

/// Forward evidence, one optional slot per feasibility field.
#[derive(Clone, Debug, Default)]
pub struct ForwardEvidence {
    /// A single forward claim, answering the `point` field.
    pub point: Option<ForwardClaim>,
    /// A published interval region, answering the `interval_region` field.
    pub interval_region: Option<IntervalRegion>,
    /// An identified set, answering the `identified_set` field.
    pub identified_set: Option<IdentifiedSet>,
    /// Declared structural scenarios, answering `all_scenario` and, when every
    /// scenario carries a genuine probability, `posterior_probability`.
    pub scenarios: Option<Vec<StructuralAtom>>,
}

/// A typed inverse decision query.
#[derive(Clone, Debug)]
pub struct InverseQuery {
    /// Contract that declares the actions, their input quantities and utilities.
    /// Its criterion and hard constraints are not used by this query.
    pub contract: DecisionContract,
    /// Ordered action ids: the declared grid and the tie-rule order.
    pub grid_order: Vec<String>,
    /// Finite enumeration or a sample of a continuous domain.
    pub grid_scope: GridScope,
    /// Constraints, all of which must hold.
    pub constraints: Vec<InverseConstraint>,
    /// Multiple-action rule.
    pub selection: SelectionRule,
    /// Non-negative slack on each comparison, for roundoff in exact laws.
    pub tolerance: f64,
    /// Most (action, source, constraint) functional evaluations; remaining
    /// candidates stay `Unevaluated`. `None` is unbounded.
    pub max_evaluations: Option<usize>,
}

/// One constraint's value for one action and claim.
#[derive(Clone, Debug, PartialEq)]
pub struct ConstraintValue {
    /// Index into the query's constraints.
    pub constraint: usize,
    /// The engine's functional value, when it was answered.
    pub value: Option<f64>,
    /// `Some(0.0)` for an exact finite law; `None` when uncertified.
    pub standard_error: Option<f64>,
    /// This constraint's own status.
    pub status: FeasibilityStatus,
    /// Detail of why it was not answered, when it was not.
    pub reason: Option<String>,
}

/// One scenario or set member's status for one action.
#[derive(Clone, Debug, PartialEq)]
pub struct MemberStatus {
    /// Member identity.
    pub id: String,
    /// Its conjunction status.
    pub status: FeasibilityStatus,
    /// Why it was not answered, when it was not.
    pub reason: Option<String>,
}

/// Probability mass over scenarios for one action. Never renormalized.
#[derive(Clone, Debug, PartialEq)]
pub struct PosteriorFeasibility {
    /// Mass of scenarios in which the action is feasible.
    pub feasible_mass: f64,
    /// Mass of scenarios in which it is infeasible.
    pub infeasible_mass: f64,
    /// Everything else: unsupported, unevaluated and unidentified scenarios and
    /// mass the supplied probabilities do not assign.
    pub unresolved_mass: f64,
}

/// One action across every feasibility field.
#[derive(Clone, Debug, PartialEq)]
pub struct ActionReport {
    /// Semantic action identity.
    pub id: String,
    /// Position in the declared grid order.
    pub position: usize,
    /// Feasibility on the single forward claim.
    pub point: Option<FeasibilityStatus>,
    /// Per-constraint values behind `point`.
    pub point_values: Vec<ConstraintValue>,
    /// Feasibility over a published interval region.
    pub interval_region: Option<FeasibilityStatus>,
    /// Feasibility over an identified set.
    pub identified_set: Option<FeasibilityStatus>,
    /// Per-member statuses behind `identified_set`.
    pub identified_set_members: Vec<MemberStatus>,
    /// Feasibility across every declared scenario.
    pub all_scenario: Option<FeasibilityStatus>,
    /// Per-scenario statuses behind `all_scenario`.
    pub scenario_members: Vec<MemberStatus>,
    /// Posterior mass of feasibility; only with a genuine probability per scenario.
    pub posterior_probability: Option<PosteriorFeasibility>,
}

/// What selection returned.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum SelectionOutcome {
    /// One action was selected under the rule.
    Selected,
    /// No action is point-feasible.
    NoFeasibleAction,
    /// `RequireUnique` and more than one action is point-feasible.
    MultipleFeasible,
    /// No point claim was supplied, so nothing is selected.
    NoPointClaim,
}

/// What the point field says about the existence of a feasible action.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ExistenceClaim {
    /// A feasible action was found; a witness, not a global claim.
    FoundFeasibleAction,
    /// Every action of a finite enumeration is decided and none is feasible.
    NoFeasibleActionInDeclaredSet,
    /// Nothing can be said (unresolved actions, a continuous sample, no claim).
    Undetermined,
}

/// Result of an inverse query.
// Independent public report flags; grouping them would change the public surface.
#[allow(clippy::struct_excessive_bools)]
#[derive(Clone, Debug, PartialEq)]
pub struct InverseResult {
    /// Identity of the contract that declared the actions.
    pub contract_identity: String,
    /// The declared grid order.
    pub grid_order: Vec<String>,
    /// The declared grid scope.
    pub grid_scope: GridScope,
    /// The constraints evaluated.
    pub constraints: Vec<InverseConstraint>,
    /// Per-action reports in grid order.
    pub actions: Vec<ActionReport>,
    /// Point-feasible action ids in grid order.
    pub feasible_actions: Vec<String>,
    /// The selected action, when the rule selects one.
    pub selected: Option<String>,
    /// How selection ended.
    pub selection: SelectionOutcome,
    /// Whether every action the rule passed over is definitely infeasible.
    pub selection_certified: bool,
    /// Whether every grid action has a decided point status.
    pub grid_fully_decided: bool,
    /// Existence claim from the point field.
    pub existence: ExistenceClaim,
    /// `true` only for a finite enumeration whose every action is decided; always
    /// `false` for a continuous sample.
    pub exhaustive_over_declared_set: bool,
    /// Functional evaluations charged.
    pub evaluations_used: usize,
    /// Whether the evaluation limit stopped the search.
    pub budget_exhausted: bool,
    /// Lineage of the point claim, when there is one.
    pub point_source: Option<SourceReceipt>,
    /// What this result is and is not.
    pub scope_note: &'static str,
}

/// Why an inverse query refused.
#[derive(Clone, Debug, PartialEq)]
pub enum InverseQueryError {
    /// The grid is empty, too large, repeats an action or names an undeclared one.
    InvalidGrid(&'static str),
    /// No constraint was declared.
    NoConstraints,
    /// A constraint, level, threshold or tolerance is invalid.
    InvalidParameter(&'static str),
    /// No forward claim of any kind was supplied.
    NoForwardEvidence,
    /// Scenario or member identities are blank or repeat, or there are none.
    InvalidScenarios,
    /// Scenario probabilities are partial, not finite, negative or sum above one.
    InvalidProbabilities,
    /// The shared functional engine refused the forward claim.
    Engine(DecisionEvalError),
}

fn refusal(code: &'static str, detail: &str, offending: Option<String>) -> ExternalRefusal {
    ExternalRefusal {
        code,
        stage: STAGE,
        detail: detail.to_owned(),
        offending,
        expected: None,
        supplied: None,
        capability: None,
        remedy: None,
    }
}

impl InverseQueryError {
    /// Registered runtime reason code.
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        match self {
            Self::Engine(error) => error.reason_code(),
            Self::InvalidGrid(_)
            | Self::NoConstraints
            | Self::InvalidParameter(_)
            | Self::NoForwardEvidence
            | Self::InvalidScenarios
            | Self::InvalidProbabilities => antecedent_core::reason_code!("invalid_argument"),
        }
    }

    /// Structured refusal. Engine refusals keep the engine's own detail (for
    /// example `decision_evaluation.joint_law_required`); the rest are
    /// `inverse_query.*`.
    #[must_use]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = antecedent_core::reason_code!("invalid_argument");
        match self {
            Self::Engine(error) => {
                let mut value = error.to_refusal();
                value.stage = STAGE;
                value
            }
            Self::InvalidGrid(why) => {
                refusal(invalid, "inverse_query.invalid_grid", Some((*why).to_owned()))
            }
            Self::NoConstraints => refusal(invalid, "inverse_query.no_constraints", None),
            Self::InvalidParameter(name) => {
                refusal(invalid, "inverse_query.invalid_parameter", Some((*name).to_owned()))
            }
            Self::NoForwardEvidence => refusal(invalid, "inverse_query.no_forward_evidence", None),
            Self::InvalidScenarios => refusal(invalid, "inverse_query.invalid_scenarios", None),
            Self::InvalidProbabilities => {
                refusal(invalid, "inverse_query.invalid_probabilities", None)
            }
        }
    }
}

/// A forward point of the finite-enumeration baseline.
#[derive(Clone, Debug, PartialEq)]
pub struct EnumeratedForwardValue {
    /// Action identity.
    pub id: String,
    /// The forward mean; `None` when the action was not evaluated.
    pub value: Option<f64>,
    /// Whether the forward route reports the action inside its support.
    pub supported: bool,
}

/// The 2.2 finite-enumeration baseline: classify each enumerated action by its
/// forward mean against `target`, within `tolerance`. An action off the
/// evaluated grid is `Unevaluated`, never interpolated; an unsupported one is
/// `Unsupported`. It uses no distribution engine, so it is the independent
/// reference the generalized query must agree with on target-mean cases.
///
/// # Errors
/// A non-finite target, tolerance or forward value, or a negative tolerance.
pub fn finite_enumeration_baseline(
    points: &[EnumeratedForwardValue],
    target: f64,
    comparison: Comparison,
    tolerance: f64,
) -> Result<Vec<(String, FeasibilityStatus)>, InverseQueryError> {
    if !target.is_finite() || !tolerance.is_finite() || tolerance < 0.0 {
        return Err(InverseQueryError::InvalidParameter("baseline_target"));
    }
    let mut classes = Vec::with_capacity(points.len());
    for point in points {
        let status = match (point.supported, point.value) {
            (false, _) => FeasibilityStatus::Unsupported,
            (true, None) => FeasibilityStatus::Unevaluated,
            (true, Some(value)) if !value.is_finite() => {
                return Err(InverseQueryError::InvalidParameter("baseline_value"));
            }
            (true, Some(value)) => {
                if comparison.holds(value, target, tolerance) {
                    FeasibilityStatus::Feasible
                } else {
                    FeasibilityStatus::Infeasible
                }
            }
        };
        classes.push((point.id.clone(), status));
    }
    Ok(classes)
}

#[derive(Clone, Copy)]
enum Source<'a> {
    Law(&'a DistributionArtifact),
    Means(&'a MeanSource),
}

impl Source<'_> {
    /// Classify an action whose coordinates the source cannot answer, before the
    /// engine is asked. Returns the status and the engine's own detail.
    fn screen(self, action: &DecisionAction) -> Option<(FeasibilityStatus, &'static str)> {
        let mut missing = false;
        let mut masked = false;
        for wanted in &action.inputs {
            match self {
                Self::Law(artifact) => {
                    let found = artifact.quantities().iter().position(|candidate| {
                        ScientificQuantity::try_from(candidate.clone())
                            .is_ok_and(|q| q.require_same_coordinate(wanted).is_ok())
                    });
                    match found {
                        None => missing = true,
                        Some(column) => {
                            let mask = artifact.metadata().supported.as_deref();
                            if mask.is_some_and(|m| m.get(column).is_some_and(|ok| !*ok)) {
                                masked = true;
                            }
                        }
                    }
                }
                Self::Means(means) => {
                    if !means.coordinates.iter().any(|c| c.require_same_coordinate(wanted).is_ok())
                    {
                        missing = true;
                    }
                }
            }
        }
        if masked {
            Some((FeasibilityStatus::Unsupported, "decision_evaluation.unsupported_coordinate"))
        } else if missing {
            Some((FeasibilityStatus::Unevaluated, "decision_evaluation.quantity_not_found"))
        } else {
            None
        }
    }
}

impl ForwardClaim {
    fn source(&self) -> Source<'_> {
        match self {
            Self::Law(artifact) => Source::Law(artifact),
            Self::Means(means) => Source::Means(means),
        }
    }

    fn receipt(&self) -> SourceReceipt {
        match self {
            Self::Law(artifact) => {
                let identity = &artifact.metadata().identity;
                SourceReceipt {
                    provider_id: identity.provider_id.clone(),
                    snapshot_id: identity.snapshot_id.clone(),
                    rng_id: identity.rng_id.clone(),
                    causal_contract_id: identity.causal_contract_id.clone(),
                }
            }
            Self::Means(means) => SourceReceipt {
                provider_id: means.provider_id.clone(),
                snapshot_id: means.snapshot_id.clone(),
                rng_id: means.rng_id.clone(),
                causal_contract_id: means.causal_contract_id.clone(),
            },
        }
    }
}

struct Meter {
    used: usize,
    limit: Option<usize>,
    exhausted: bool,
}

impl Meter {
    fn charge(&mut self, amount: usize) -> bool {
        if self.exhausted
            || self.limit.is_some_and(|limit| self.used.saturating_add(amount) > limit)
        {
            self.exhausted = true;
            return false;
        }
        self.used += amount;
        true
    }
}

struct Member {
    status: FeasibilityStatus,
    values: Vec<ConstraintValue>,
    reason: Option<String>,
}

/// A two-action contract holding the queried action and a padded copy, so the
/// engine sees only this action's coordinates and a missing or masked
/// coordinate of another grid action cannot refuse this one.
fn single_action_contract(
    contract: &DecisionContract,
    action: &DecisionAction,
) -> DecisionContract {
    let mut padded = action.clone();
    padded.id = format!("{}#pad", action.id);
    DecisionContract {
        actions: vec![action.clone(), padded],
        utility_units: contract.utility_units.clone(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: Vec::new(),
        target_population: contract.target_population.clone(),
        horizon: contract.horizon,
        structural_policy: contract.structural_policy,
    }
}

fn mean_value(
    contract: &DecisionContract,
    action_id: &str,
    functional: DecisionFunctional,
    means: &MeanSource,
) -> Result<(f64, Option<f64>), DecisionEvalError> {
    let unknown =
        || DecisionEvalError::Contract(DecisionContractError::UnknownAction(action_id.to_owned()));
    let action = contract.actions.iter().find(|a| a.id == action_id).ok_or_else(unknown)?;
    functional.requirement(&action.utility).check(&[SourceRepresentation::Mean]).map_err(
        |error| match error {
            DecisionContractError::MissingSource { needed } => {
                DecisionEvalError::MeanSourceInsufficient { needed }
            }
            other => DecisionEvalError::Contract(other),
        },
    )?;
    let result = evaluate_contract_on_means(contract, means)?;
    let outcome = result.actions.iter().find(|o| o.id == action_id).ok_or_else(unknown)?;
    Ok((outcome.expected_utility, outcome.standard_error))
}

fn engine_value(
    contract: &DecisionContract,
    action_id: &str,
    functional: DecisionFunctional,
    source: Source<'_>,
) -> Result<(f64, Option<f64>), DecisionEvalError> {
    match source {
        Source::Law(artifact) => evaluate_functional(contract, action_id, functional, artifact)
            .map(|value| (value.value, value.standard_error)),
        Source::Means(means) => mean_value(contract, action_id, functional, means),
    }
}

fn unresolved(
    index: usize,
    status: FeasibilityStatus,
    error: &DecisionEvalError,
) -> ConstraintValue {
    ConstraintValue {
        constraint: index,
        value: None,
        standard_error: None,
        status,
        reason: Some(error.to_refusal().detail),
    }
}

fn constraint_value(
    tolerance: f64,
    contract: &DecisionContract,
    source: Source<'_>,
    action_id: &str,
    index: usize,
    constraint: &InverseConstraint,
) -> Result<ConstraintValue, InverseQueryError> {
    match engine_value(contract, action_id, constraint.functional(), source) {
        Ok((value, standard_error)) => {
            let (target, comparison) = constraint.target();
            let status = if comparison.holds(value, target, tolerance) {
                FeasibilityStatus::Feasible
            } else {
                FeasibilityStatus::Infeasible
            };
            Ok(ConstraintValue {
                constraint: index,
                value: Some(value),
                standard_error,
                status,
                reason: None,
            })
        }
        Err(error) => {
            let status = match &error {
                DecisionEvalError::UnsupportedCoordinate { .. } => {
                    Some(FeasibilityStatus::Unsupported)
                }
                DecisionEvalError::QuantityNotFound { .. } => Some(FeasibilityStatus::Unevaluated),
                _ => None,
            };
            match status {
                Some(status) => Ok(unresolved(index, status, &error)),
                None => Err(InverseQueryError::Engine(error)),
            }
        }
    }
}

/// Conjunction over one member's constraints.
fn conjunction(values: &[ConstraintValue]) -> FeasibilityStatus {
    let has = |wanted: FeasibilityStatus| values.iter().any(|v| v.status == wanted);
    if has(FeasibilityStatus::Infeasible) {
        FeasibilityStatus::Infeasible
    } else if has(FeasibilityStatus::Unsupported) {
        FeasibilityStatus::Unsupported
    } else if has(FeasibilityStatus::Unevaluated) || values.is_empty() {
        FeasibilityStatus::Unevaluated
    } else {
        FeasibilityStatus::Feasible
    }
}

/// Evaluate every constraint of `query` for `action_id` on one source.
///
/// `lenient` (scenario and set members) keeps a member whose evidence cannot
/// answer the contract as `Unevaluated` with the engine's detail; a contract
/// error still refuses. A point or endpoint claim is strict and refuses.
fn evaluate_member(
    query: &InverseQuery,
    source: Source<'_>,
    action_id: &str,
    lenient: bool,
    meter: &mut Meter,
) -> Result<Member, InverseQueryError> {
    let Some(action) = query.contract.actions.iter().find(|a| a.id == action_id) else {
        return Err(InverseQueryError::InvalidGrid("a grid action is not declared"));
    };
    if !meter.charge(query.constraints.len()) {
        return Ok(Member {
            status: FeasibilityStatus::Unevaluated,
            values: Vec::new(),
            reason: Some(BUDGET_REASON.to_owned()),
        });
    }
    if let Some((status, reason)) = source.screen(action) {
        return Ok(Member { status, values: Vec::new(), reason: Some(reason.to_owned()) });
    }
    let single = single_action_contract(&query.contract, action);
    let mut values = Vec::with_capacity(query.constraints.len());
    for (index, constraint) in query.constraints.iter().enumerate() {
        match constraint_value(query.tolerance, &single, source, action_id, index, constraint) {
            Ok(value) => values.push(value),
            Err(InverseQueryError::Engine(error))
                if lenient && !matches!(error, DecisionEvalError::Contract(_)) =>
            {
                values.push(unresolved(index, FeasibilityStatus::Unevaluated, &error));
            }
            Err(error) => return Err(error),
        }
    }
    let reason = values.iter().find_map(|v| v.reason.clone());
    Ok(Member { status: conjunction(&values), values, reason })
}

/// Combine member statuses into one set-wide status.
///
/// A mix of feasible and infeasible members is `StructurallyAmbiguous`. Any
/// unidentified, then unevaluated, then unsupported member stops a uniform
/// claim. Only a uniform feasible or infeasible set reports that status.
fn aggregate_members(statuses: &[FeasibilityStatus]) -> FeasibilityStatus {
    let count = |wanted: FeasibilityStatus| statuses.iter().filter(|s| **s == wanted).count();
    let feasible = count(FeasibilityStatus::Feasible);
    let infeasible = count(FeasibilityStatus::Infeasible);
    if count(FeasibilityStatus::StructurallyAmbiguous) > 0 || (feasible > 0 && infeasible > 0) {
        FeasibilityStatus::StructurallyAmbiguous
    } else if count(FeasibilityStatus::Unidentified) > 0 {
        FeasibilityStatus::Unidentified
    } else if count(FeasibilityStatus::Unevaluated) > 0 {
        FeasibilityStatus::Unevaluated
    } else if count(FeasibilityStatus::Unsupported) > 0 {
        FeasibilityStatus::Unsupported
    } else if feasible > 0 {
        FeasibilityStatus::Feasible
    } else if infeasible > 0 {
        FeasibilityStatus::Infeasible
    } else {
        FeasibilityStatus::Unevaluated
    }
}

/// A uniform answer over a set that was not declared complete is unevaluated.
fn require_declared_complete(status: FeasibilityStatus, complete: bool) -> FeasibilityStatus {
    if !complete && matches!(status, FeasibilityStatus::Feasible | FeasibilityStatus::Infeasible) {
        FeasibilityStatus::Unevaluated
    } else {
        status
    }
}

fn atom_members(
    query: &InverseQuery,
    atoms: &[StructuralAtom],
    action_id: &str,
    meter: &mut Meter,
) -> Result<Vec<MemberStatus>, InverseQueryError> {
    let mut members = Vec::with_capacity(atoms.len());
    for atom in atoms {
        let (status, reason) = match &atom.evidence {
            AtomEvidence::Unidentified => (FeasibilityStatus::Unidentified, None),
            AtomEvidence::Unevaluated(reason) => {
                (FeasibilityStatus::Unevaluated, Some(reason.clone()))
            }
            AtomEvidence::Evaluated(artifact) => {
                let member = evaluate_member(query, Source::Law(artifact), action_id, true, meter)?;
                (member.status, member.reason)
            }
        };
        members.push(MemberStatus { id: atom.id.clone(), status, reason });
    }
    Ok(members)
}

fn statuses(members: &[MemberStatus]) -> Vec<FeasibilityStatus> {
    members.iter().map(|m| m.status).collect()
}

fn posterior(atoms: &[StructuralAtom], members: &[MemberStatus]) -> Option<PosteriorFeasibility> {
    if atoms.iter().any(|a| a.probability.is_none()) {
        return None;
    }
    let mass = |wanted: FeasibilityStatus| -> f64 {
        atoms
            .iter()
            .zip(members)
            .filter(|(_, member)| member.status == wanted)
            .filter_map(|(atom, _)| atom.probability)
            .sum()
    };
    let feasible_mass = mass(FeasibilityStatus::Feasible);
    let infeasible_mass = mass(FeasibilityStatus::Infeasible);
    Some(PosteriorFeasibility {
        feasible_mass,
        infeasible_mass,
        unresolved_mass: (1.0 - feasible_mass - infeasible_mass).max(0.0),
    })
}

fn report_action(
    query: &InverseQuery,
    evidence: &ForwardEvidence,
    id: &str,
    position: usize,
    meter: &mut Meter,
) -> Result<ActionReport, InverseQueryError> {
    let mut report = ActionReport {
        id: id.to_owned(),
        position,
        point: None,
        point_values: Vec::new(),
        interval_region: None,
        identified_set: None,
        identified_set_members: Vec::new(),
        all_scenario: None,
        scenario_members: Vec::new(),
        posterior_probability: None,
    };
    if let Some(claim) = &evidence.point {
        let member = evaluate_member(query, claim.source(), id, false, meter)?;
        report.point = Some(member.status);
        report.point_values = member.values;
    }
    if let Some(region) = &evidence.interval_region {
        let lower = evaluate_member(query, region.lower.source(), id, false, meter)?;
        let upper = evaluate_member(query, region.upper.source(), id, false, meter)?;
        let status = aggregate_members(&[lower.status, upper.status]);
        report.interval_region =
            Some(require_declared_complete(status, region.endpoints_bound_functional));
    }
    if let Some(set) = &evidence.identified_set {
        let members = atom_members(query, &set.members, id, meter)?;
        let status = aggregate_members(&statuses(&members));
        report.identified_set = Some(require_declared_complete(status, set.exhaustive));
        report.identified_set_members = members;
    }
    if let Some(atoms) = &evidence.scenarios {
        let members = atom_members(query, atoms, id, meter)?;
        report.all_scenario = Some(aggregate_members(&statuses(&members)));
        report.posterior_probability = posterior(atoms, &members);
        report.scenario_members = members;
    }
    Ok(report)
}

fn validate_grid(query: &InverseQuery) -> Result<(), InverseQueryError> {
    let grid = &query.grid_order;
    if grid.is_empty() {
        return Err(InverseQueryError::InvalidGrid("the action grid is empty"));
    }
    if grid.len() > MAX_INVERSE_GRID {
        return Err(InverseQueryError::InvalidGrid("the action grid exceeds the bound"));
    }
    let mut seen = HashSet::new();
    for id in grid {
        if !seen.insert(id.as_str()) {
            return Err(InverseQueryError::InvalidGrid("the action grid repeats an action"));
        }
        if !query.contract.actions.iter().any(|a| &a.id == id) {
            return Err(InverseQueryError::InvalidGrid("a grid action is not declared"));
        }
    }
    Ok(())
}

fn validate_atom_ids(atoms: &[StructuralAtom]) -> Result<(), InverseQueryError> {
    let mut ids = HashSet::new();
    if atoms.is_empty()
        || atoms.iter().any(|a| a.id.trim().is_empty() || !ids.insert(a.id.as_str()))
    {
        return Err(InverseQueryError::InvalidScenarios);
    }
    Ok(())
}

fn validate_probabilities(atoms: &[StructuralAtom]) -> Result<(), InverseQueryError> {
    let with = atoms.iter().filter(|a| a.probability.is_some()).count();
    if with == 0 {
        return Ok(());
    }
    let total: f64 = atoms.iter().filter_map(|a| a.probability).sum();
    if with != atoms.len()
        || atoms.iter().any(|a| a.probability.is_some_and(|p| !p.is_finite() || p < 0.0))
        || total > 1.0 + 1e-9
    {
        return Err(InverseQueryError::InvalidProbabilities);
    }
    Ok(())
}

fn validate_query(
    query: &InverseQuery,
    evidence: &ForwardEvidence,
) -> Result<String, InverseQueryError> {
    let identity = query
        .contract
        .identity()
        .map_err(|error| InverseQueryError::Engine(DecisionEvalError::Contract(error)))?;
    validate_grid(query)?;
    if query.constraints.is_empty() {
        return Err(InverseQueryError::NoConstraints);
    }
    if query.constraints.iter().any(|c| !c.valid()) {
        return Err(InverseQueryError::InvalidParameter("constraint"));
    }
    if !query.tolerance.is_finite() || query.tolerance < 0.0 {
        return Err(InverseQueryError::InvalidParameter("tolerance"));
    }
    if evidence.point.is_none()
        && evidence.interval_region.is_none()
        && evidence.identified_set.is_none()
        && evidence.scenarios.is_none()
    {
        return Err(InverseQueryError::NoForwardEvidence);
    }
    if let Some(set) = &evidence.identified_set {
        validate_atom_ids(&set.members)?;
    }
    if let Some(atoms) = &evidence.scenarios {
        validate_atom_ids(atoms)?;
        validate_probabilities(atoms)?;
    }
    Ok(identity)
}

struct Selection {
    outcome: SelectionOutcome,
    selected: Option<usize>,
    certified: bool,
}

fn select(rule: SelectionRule, actions: &[ActionReport], has_point: bool) -> Selection {
    let none = |outcome| Selection { outcome, selected: None, certified: false };
    if !has_point {
        return none(SelectionOutcome::NoPointClaim);
    }
    let feasible: Vec<usize> = actions
        .iter()
        .enumerate()
        .filter(|(_, a)| a.point == Some(FeasibilityStatus::Feasible))
        .map(|(index, _)| index)
        .collect();
    let (Some(first), Some(last)) = (feasible.first().copied(), feasible.last().copied()) else {
        return none(SelectionOutcome::NoFeasibleAction);
    };
    let pick = match rule {
        SelectionRule::FirstInGridOrder => first,
        SelectionRule::LastInGridOrder => last,
        SelectionRule::RequireUnique => {
            if feasible.len() > 1 {
                return none(SelectionOutcome::MultipleFeasible);
            }
            first
        }
    };
    let infeasible = |index: usize| actions[index].point == Some(FeasibilityStatus::Infeasible);
    let certified = match rule {
        SelectionRule::FirstInGridOrder => (0..pick).all(infeasible),
        SelectionRule::LastInGridOrder => (pick + 1..actions.len()).all(infeasible),
        SelectionRule::RequireUnique => {
            (0..actions.len()).filter(|index| *index != pick).all(infeasible)
        }
    };
    Selection { outcome: SelectionOutcome::Selected, selected: Some(pick), certified }
}

/// Evaluate an inverse query on forward evidence.
///
/// Each (action, claim, constraint) is answered by the shared functional
/// engine; the module only compares its value with the target. Per-action
/// refusals of unsupported or absent coordinates become that action's
/// `Unsupported` or `Unevaluated` status; a joint-law requirement the claim
/// cannot meet, a mean-only claim asked for a quantile or probability, an
/// outcome-law meaning mismatch and an invalid contract refuse the query.
///
/// # Errors
/// Refuses an invalid grid, constraint, tolerance or scenario declaration, a
/// query with no forward evidence, and any engine refusal on a point or
/// interval-endpoint claim (see [`InverseQueryError::Engine`]).
pub fn evaluate_inverse_query(
    query: &InverseQuery,
    evidence: &ForwardEvidence,
) -> Result<InverseResult, InverseQueryError> {
    let contract_identity = validate_query(query, evidence)?;
    let mut meter = Meter { used: 0, limit: query.max_evaluations, exhausted: false };
    let mut actions = Vec::with_capacity(query.grid_order.len());
    for (position, id) in query.grid_order.iter().enumerate() {
        actions.push(report_action(query, evidence, id, position, &mut meter)?);
    }
    let has_point = evidence.point.is_some();
    let feasible_actions: Vec<String> = actions
        .iter()
        .filter(|a| a.point == Some(FeasibilityStatus::Feasible))
        .map(|a| a.id.clone())
        .collect();
    let decided = has_point
        && actions.iter().all(|a| {
            matches!(a.point, Some(FeasibilityStatus::Feasible | FeasibilityStatus::Infeasible))
        });
    let exhaustive = decided && query.grid_scope == GridScope::FiniteEnumeration;
    let existence = if !feasible_actions.is_empty() {
        ExistenceClaim::FoundFeasibleAction
    } else if exhaustive {
        ExistenceClaim::NoFeasibleActionInDeclaredSet
    } else {
        ExistenceClaim::Undetermined
    };
    let selection = select(query.selection, &actions, has_point);
    Ok(InverseResult {
        contract_identity,
        grid_order: query.grid_order.clone(),
        grid_scope: query.grid_scope,
        constraints: query.constraints.clone(),
        selected: selection.selected.map(|index| actions[index].id.clone()),
        selection: selection.outcome,
        selection_certified: selection.certified,
        grid_fully_decided: decided,
        existence,
        exhaustive_over_declared_set: exhaustive,
        evaluations_used: meter.used,
        budget_exhausted: meter.exhausted,
        point_source: evidence.point.as_ref().map(ForwardClaim::receipt),
        feasible_actions,
        actions,
        scope_note: INVERSE_QUERY_SCOPE_NOTE,
    })
}
