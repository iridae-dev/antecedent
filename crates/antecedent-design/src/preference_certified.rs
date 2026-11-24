//! Certified verdicts across a preference set by interval subdivision.
//!
//! When an action's value is not affine in the preference parameters (a discount factor that
//! is squared, the location of a breakpoint, a risk attitude inside an exponential), or the
//! criterion is not an expectation, the verdict across the admissible set cannot be read from
//! linear programs. This module covers the declared parameter box with sub-boxes and bounds
//! every action's criterion on each sub-box with outward-rounded interval arithmetic.
//!
//! Per-draw attribute values are read once, through [`PreferenceContract::probe_contract`]
//! and [`utility_draws`]; a fixed input is a constant. On a box, each action's value on each
//! draw is enclosed by evaluating the model expression with the attributes at their drawn
//! values and every parameter ranging over its side of the box. Every floating operation moves
//! the lower end of its result down and the upper end up by one representable step (two for
//! `exp` and `ln`), so each enclosure contains the exact real range. When the model has a
//! linear form, the parameter-free parts are enclosed once per draw and the value on a box is
//! the affine combination of those parts with the parameter intervals; expectations and
//! expectation margins then combine the expected parts first, which removes the dependency
//! between actions that share a parameter. Correctness does not rest on this: without a linear
//! form the natural interval extension of the expression is used.
//!
//! Criterion bounds on a box follow the decision layer's definitions exactly:
//! - expectation: the weighted mean of the per-draw enclosures; margins between two actions
//!   are bounded from the per-draw enclosure of the difference;
//! - threshold probability `P(U >= t)`: `[P(lo >= t), P(hi >= t)]`;
//! - quantile: `[Q_p(lo), Q_p(hi)]` with the left inverse of the weighted empirical CDF and the
//!   same mass tolerance as [`crate::decision_eval::evaluate_functional`];
//! - regret (largest per-draw regret over positive-weight draws): the per-draw enclosure of
//!   `max(0, max_b U_b - U_a)`, then its maximum;
//! - expected regret: ranked through expected-utility margins (the best-action term is common
//!   to every action), its own value enclosed as the weighted mean of per-draw regret.
//!
//! Weighted sums are widened further by `2 (n + 1) eps` times the sum of magnitudes, so they also
//! contain any floating summation order. A box is certified for an action only when its
//! enclosures prove that action better than every admissible rival by more than the tie
//! tolerance everywhere in the box; a box is never labelled from a point evaluation.
//!
//! Before any box is examined, the linear program of the exact route checks that the declared
//! ranges and linear relations admit a point; relations that conflict only jointly by more than
//! its feasibility tolerance are refused as an empty set rather than kept as boundary boxes. A
//! gap narrower than that tolerance passes this check, so it proves nothing about membership:
//! see the witness rule below. An equality relation is solved for one of its parameters (taken in order of relation
//! id; the pivot is the parameter of positive width with the largest `|coefficient| * width`
//! that no earlier solved equality reads): that parameter is not subdivided, its interval on a
//! box is the outward-rounded enclosure of its solution, and its declared range is checked as two
//! relations, so a box can lie wholly inside an equality.
//!
//! Relations between parameters are checked per box with the same arithmetic: a box whose
//! relation left side is proven to violate some relation everywhere is dropped; a box on which
//! some relation is neither proven satisfied nor proven violated is kept and flagged as touching
//! the relation boundary. A value relation (a recorded choice under a value with shape
//! parameters: the expected value of the chosen option minus that of the rejected one, at least
//! minus its tolerance) is checked the same way, its gap enclosed by evaluating the model
//! expression on the recorded rows of both options. That enclosure need not be tight, so a box on
//! which a value relation is undecided is split further even when an action is certified on it.
//!
//! What a choosing verdict guarantees: a necessarily-best, worst-case or minimax-regret choice
//! is made only when the set is proven to hold a point, by a witness: a kept box wholly inside
//! every relation, or a point (the set check's point, then the centre of each kept box) inside
//! the subdivided box at which every relation is proven satisfied with no tolerance. At a point,
//! linear relations are evaluated in interval arithmetic whose ends move outward only when an
//! operation is inexact (its exact rounding error, recovered by error-free transformations, is
//! non-zero), so touching bounds such as `t >= 0.5` and `t <= 0.5` are proven at `t = 0.5`; a
//! solved parameter is the enclosure of its solution; value relations use their outward-rounded
//! gap enclosure. Without a witness the verdict is undetermined: neither a choice nor a claim
//! that the set is empty, since a set left with only boundary boxes may be empty or may be a thin
//! set whose points the arithmetic cannot exhibit. A set is refused as empty only when the linear
//! program finds it infeasible or every box is proven outside some relation. Fractions are volume fractions of the kept boxes
//! relative to their total volume, measured relative to the declared ranges, with a fixed or
//! solved parameter contributing no dimension; a boundary box counts in full, although part of
//! it may lie outside the set. Splitting is deterministic: the undecided box of largest volume
//! (ties: the lexicographically smallest lower corner) is split at the midpoint of the parameter
//! whose width is largest relative to its declared range, until every box is decided, the box
//! budget is spent, or every width is at most the minimum fraction of its declared range.
//!
//! Hard constraints never read parameters. Their exclusions are read once: from the ordinary
//! contract bound at the midpoint of the declared ranges and evaluated by
//! [`evaluate_contract`] when the bound value can be lowered, and otherwise (an `Exp` or `Ln`
//! wraps a quantity) by evaluating each constraint on the drawn attributes with the same rule,
//! `P(expr <= bound) >= min_probability`. A result over the preference set is an assumption
//! range, never a probability, unless a measure over the set is supplied.

// SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BinaryHeap;

use antecedent_contract::ExternalRefusal;
use antecedent_core::ExecutionContext;
use antecedent_io_core::distribution_artifact::DistributionArtifact;
use antecedent_stats::linear_program::LpOptions;

use crate::decision_contract::{DecisionCriterion, StructuralPolicy};
use crate::decision_eval::{
    ConstraintExclusion, DecisionEvalError, SourceReceipt, constraint_fails, criterion_value,
    evaluate_contract, left_inverse, utility_draws,
};
use crate::decision_structural::{AtomEvidence, StructuralAtom};
use crate::preference_contract::{
    ActionInput, PreferenceContract, PreferenceContractError, PreferencePolicy,
};
use crate::preference_decision::{
    MeasureOrigin, PreferenceDecisionError, PreferenceMeasure, linear_set_point,
};
use crate::preference_enclosure::{
    EncloseFailure, Fit, Interval, Queued, affine, enclose, lexicographic, linear_relation_fit,
    linear_relations_hold, measure_box, split_box, value_relation_fit, weighted_sum,
};
use crate::preference_model::{
    LinearForm, LinearRelation, PreferenceModelError, PreferenceSet, RelationOrigin, RelationSense,
    ValueModel,
};
use crate::preference_refusal::{EVALUATE, STRUCTURAL, refusal};
use crate::signal::at_stage;
use crate::structure_aggregator::{
    NO_STRUCTURE_EVALUATED, StructureMasses, StructureState, missing_evidence, policy_kind,
    probability_sum, structure_masses,
};

mod scoring;
mod subdivision;
#[allow(clippy::wildcard_imports, reason = "the parts of this module")]
use scoring::*;
#[allow(clippy::wildcard_imports, reason = "the parts of this module")]
use subdivision::*;

/// Distance from the declared set within which a measure point is accepted.
const MEASURE_SET_TOLERANCE: f64 = 1e-9;

/// Limits and tolerance of the certified subdivision.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct CertifiedOptions {
    /// Largest number of boxes examined, the declared box and dropped boxes included; at
    /// least one.
    pub max_boxes: usize,
    /// A box is not split once every parameter's width is at most this fraction of its
    /// declared range; finite, in `[0, 1]`.
    pub min_width_fraction: f64,
    /// A box is certified for an action only when the action's criterion is proven better
    /// than every admissible rival's by more than this; finite and non-negative. Also the
    /// margin by which a worst-case, regret or measure-weighted value must lead. Every lead is
    /// read from an outward-rounded enclosure (of a box, or of a measure point with degenerate
    /// parameter intervals), so this is the whole margin.
    pub tie_tolerance: f64,
}

impl Default for CertifiedOptions {
    fn default() -> Self {
        Self { max_boxes: 4096, min_width_fraction: 1e-6, tie_tolerance: 1e-9 }
    }
}

/// What a kept box proves.
#[derive(Clone, Debug, PartialEq)]
pub enum RegionStatus {
    /// The action is certified best everywhere in the box.
    Best(String),
    /// No action is certified; the candidates are the admissible actions not proven worse
    /// than some rival on the box, in declaration order.
    Undecided {
        /// Actions still possibly best on the box.
        candidates: Vec<String>,
    },
}

/// One kept box of parameter values.
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedRegion {
    /// Lower corner, one value per parameter.
    pub lower: Vec<f64>,
    /// Upper corner, one value per parameter.
    pub upper: Vec<f64>,
    /// What the box proves.
    pub status: RegionStatus,
    /// Some relation is neither proven satisfied nor proven violated on the box, so part of
    /// the box may lie outside the set.
    pub touches_relation_boundary: bool,
}

/// One action across the preference set.
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedActionOutcome {
    /// Semantic action identity.
    pub id: String,
    /// Whether every hard constraint held (in every evaluated structure).
    pub admissible: bool,
    /// Constraints it failed, in constraint order (structure order first when several
    /// structures are combined); empty when admissible.
    pub exclusions: Vec<ConstraintExclusion>,
    /// Volume fraction of the kept boxes on which it is certified best.
    pub best_fraction: f64,
    /// Enclosure of its criterion value over every kept box, in the criterion's own
    /// orientation. `None` when excluded or when no boxes were examined.
    pub value_bounds: Option<(f64, f64)>,
    /// Enclosure of its worst criterion value over the set, in the criterion's own orientation
    /// (the largest value for a loss or regret criterion). The end on the worse side is the
    /// extreme bound over every kept box; the other end is the best bound over boxes wholly
    /// inside the relations and over the points at their centers (the worst value is no worse
    /// than the value at any admissible point), or the extreme bound over every kept box when no
    /// box lies wholly inside. `None` when excluded or when no boxes were examined.
    pub worst_case_bounds: Option<(f64, f64)>,
    /// Enclosure of its worst-case regret over the set (the largest amount by which a rival's
    /// expected criterion exceeds its own, floored at zero), for expectation criteria only.
    /// The upper end is taken over every kept box, the lower end over boxes wholly inside the
    /// relations and the points at their centers.
    pub worst_case_regret_bounds: Option<(f64, f64)>,
    /// Measure-weighted criterion value, in the criterion's own orientation; only under the
    /// `Weighted` policy.
    pub measure_value: Option<f64>,
    /// Weight of measure points at which it leads every admissible rival by more than the tie
    /// tolerance; only under the `Weighted` policy.
    pub measure_share: Option<f64>,
}

/// What the certified analysis can claim.
#[derive(Clone, Debug, PartialEq)]
pub enum CertifiedVerdict {
    /// Every kept box is certified for this action: it is necessarily best.
    NecessarilyBest(String),
    /// At least two actions are each certified best on some box wholly inside the relations,
    /// in declaration order; the choice depends on preferences.
    PreferenceDependent {
        /// Actions certified best somewhere in the set.
        certified_somewhere: Vec<String>,
    },
    /// The worst-value enclosure of this action lies above every rival's by more than the tie
    /// tolerance.
    WorstCaseChoice(String),
    /// The worst-case regret enclosure of this action lies below every rival's by more than
    /// the tie tolerance.
    MinimaxRegretChoice(String),
    /// The largest measure-weighted value, with the weight of measure points where it leads.
    MeasureChoice {
        /// Chosen action.
        action: String,
        /// Weight of the measure points at which it leads.
        share: f64,
    },
    /// The enclosures do not settle the policy.
    Undetermined {
        /// Actions that remain possible choices, in declaration order.
        candidates: Vec<String>,
        /// Volume fraction of kept boxes left undecided.
        undecided_fraction: f64,
    },
    /// Everything computed; no choice requested.
    ReportOnly,
    /// Every action is excluded by a hard constraint.
    NoAdmissibleAction,
}

/// Result of the certified subdivision on one source or one combination of structures.
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedPreferenceResult {
    /// Identity of the evaluated preference contract.
    pub contract_identity: String,
    /// Preference policy applied.
    pub policy: PreferencePolicy,
    /// Parameter ids, in index order.
    pub parameter_ids: Vec<String>,
    /// Per-action outcomes, in declaration order.
    pub actions: Vec<CertifiedActionOutcome>,
    /// Kept boxes, in lexicographic order of `lower` (then `upper`); empty under the
    /// `Weighted` policy, which evaluates measure points instead.
    pub regions: Vec<CertifiedRegion>,
    /// Volume fraction of kept boxes left undecided.
    pub undecided_fraction: f64,
    /// Boxes examined, dropped boxes included.
    pub boxes_examined: usize,
    /// The verdict.
    pub verdict: CertifiedVerdict,
    /// Parameter assumption ids and, when a measure is read, its origin; sorted, distinct.
    pub assumptions: Vec<String>,
    /// Source lineage; for a combination of structures, the first evaluated structure's.
    pub source: SourceReceipt,
    /// Number of draws read (summed over combined structures).
    pub n_draws: usize,
    /// The tie tolerance the verdict applied: the declared one on boxes, whose enclosures are
    /// outward-rounded; under the `Weighted` policy, the declared one plus the largest width of a
    /// measure-weighted margin enclosure, the numerical resolution of the measure-point
    /// evaluation.
    pub effective_tie_tolerance: f64,
}

/// What became of one structure.
#[derive(Clone, Debug, PartialEq)]
pub enum CertifiedAtomStatus {
    /// Evaluated on its own.
    Evaluated(Box<CertifiedPreferenceResult>),
    /// The quantity is not identified under this structure.
    Unidentified,
    /// Not evaluated, with the supplied reason or the registered reason code of the source's
    /// refusal.
    Unevaluated(String),
}

/// One structure in a structural result.
#[derive(Clone, Debug, PartialEq)]
pub struct CertifiedAtomSummary {
    /// Structure identity.
    pub id: String,
    /// Supplied probability.
    pub probability: Option<f64>,
    /// Outcome.
    pub status: CertifiedAtomStatus,
}

/// One structure's own verdict.
#[derive(Clone, Debug, PartialEq)]
pub struct AtomVerdict {
    /// Structure identity.
    pub atom: String,
    /// Its verdict.
    pub verdict: CertifiedVerdict,
}

/// What the analysis across structures and preferences can claim.
#[derive(Clone, Debug, PartialEq)]
pub enum StructuralCertifiedVerdict {
    /// The joint verdict: the combined result's verdict, or the verdict every structure
    /// agrees on.
    Combined(CertifiedVerdict),
    /// Structures give different choices; each structure's verdict is listed.
    StructuresDisagree(Vec<AtomVerdict>),
    /// The claim needs evidence that is missing.
    InsufficientScience(String),
    /// Per-structure results only.
    ReportOnly,
}

/// Certified analysis across structures and preferences.
#[derive(Clone, Debug, PartialEq)]
pub struct StructuralCertifiedResult {
    /// Identity of the evaluated preference contract.
    pub contract_identity: String,
    /// Structural policy applied.
    pub structural_policy: StructuralPolicy,
    /// Preference policy applied.
    pub preference_policy: PreferencePolicy,
    /// Per-structure outcomes, in atom order.
    pub atoms: Vec<CertifiedAtomSummary>,
    /// The joint result over structures and parameters: worst case over structures for
    /// `Maximin`, the probability-weighted mixture for `BayesOverStructures` (not
    /// renormalized by the evaluated mass); `None` for `RequireInvariantBestAction` (each
    /// structure's own result is in `atoms`), `ReportOnly`, or when no structure was
    /// evaluated.
    pub combined: Option<CertifiedPreferenceResult>,
    /// Probability mass of unidentified structures, when probabilities exist.
    pub unidentified_mass: Option<f64>,
    /// Probability mass of unevaluated structures (and unassigned mass), when probabilities
    /// exist.
    pub unevaluated_mass: Option<f64>,
    /// Probability mass of evaluated structures, when probabilities exist.
    pub evaluated_mass: Option<f64>,
    /// The verdict.
    pub verdict: StructuralCertifiedVerdict,
}

/// Why a certified evaluation is refused.
#[derive(Clone, Debug, PartialEq, thiserror::Error)]
pub enum PreferenceCertifiedError {
    /// The preference contract, or a contract derived from it, is refused.
    #[error("preference contract: {0}")]
    Contract(PreferenceContractError),
    /// The model could not be evaluated at a measure point.
    #[error("value model: {0}")]
    Model(PreferenceModelError),
    /// The source cannot answer the contract.
    #[error("source: {0}")]
    Eval(DecisionEvalError),
    /// The criterion needs identified-set or structure inputs.
    #[error("the criterion needs identified-set or structure inputs, not one draw source")]
    StructureInputsRequired,
    /// An option is outside its range.
    #[error("option {option} is outside its range")]
    InvalidOptions {
        /// Name of the option.
        option: &'static str,
    },
    /// `MinimaxRegret` on this route needs an expectation criterion.
    #[error("minimax regret on the certified route needs an expectation criterion")]
    MinimaxRegretNeedsExpectation,
    /// The `Weighted` policy needs a measure over the set.
    #[error("the weighted policy needs a measure over the preference set")]
    MeasureRequired,
    /// The measure has no points.
    #[error("the measure has no points")]
    MeasureEmpty,
    /// The measure marks a parameter index that does not exist as unresolved.
    #[error("the measure names unknown parameter {index}")]
    MeasureUnknownParameter {
        /// The unknown index.
        index: usize,
    },
    /// The measure says nothing about some parameters, so measure-weighted values are not
    /// defined.
    #[error("the measure leaves parameters {parameters:?} unresolved")]
    MeasureIncomplete {
        /// Ids of the unresolved parameters, in index order.
        parameters: Vec<String>,
    },
    /// A measure point does not have one value per parameter.
    #[error("measure point {index} has {found} values for {expected} parameters")]
    MeasurePointDimension {
        /// Position of the point.
        index: usize,
        /// The set's dimension.
        expected: usize,
        /// Values given.
        found: usize,
    },
    /// A measure point has a non-finite value.
    #[error("measure point {index} is not finite")]
    MeasureNonFinitePoint {
        /// Position of the point.
        index: usize,
    },
    /// A measure point lies outside the preference set.
    #[error("measure point {index} lies outside the preference set")]
    MeasurePointOutsideSet {
        /// Position of the point.
        index: usize,
    },
    /// The measure weights do not give one finite non-negative weight per point with a
    /// positive total.
    #[error("the measure weights are invalid")]
    MeasureWeightsInvalid,
    /// The measure labels do not give one label per point.
    #[error("the measure labels do not match the points")]
    MeasureLabelsInvalid,
    /// The relations exclude every box of the declared ranges.
    #[error("the relations exclude every parameter value of the declared ranges")]
    EmptyPreferenceSet,
    /// The linear program that checks whether the declared ranges and linear relations admit a
    /// point refused.
    #[error("set check: {0}")]
    SetProgram(PreferenceDecisionError),
    /// A logarithm's argument reaches zero or below on a box: the model is not defined there.
    #[error("the logarithm in the value of {action} is undefined on part of the box")]
    LogarithmDomain {
        /// Action whose value was being enclosed.
        action: String,
        /// Lower corner of the box.
        lower: Vec<f64>,
        /// Upper corner of the box.
        upper: Vec<f64>,
    },
    /// An enclosure has a non-finite bound on a box.
    #[error("the enclosure for {subject} is not finite on the box")]
    NonFiniteEnclosure {
        /// Action or relation whose enclosure failed.
        subject: String,
        /// Lower corner of the box.
        lower: Vec<f64>,
        /// Upper corner of the box.
        upper: Vec<f64>,
    },
    /// No structures, or a blank or repeated structure id.
    #[error("structures are missing, blank or repeated")]
    InvalidAtoms,
    /// Probabilities are given for some structures only, or are not finite and non-negative,
    /// or sum to more than one.
    #[error("structure probabilities are invalid")]
    InvalidProbabilities,
    /// `BayesOverStructures` needs genuine structure probabilities.
    #[error("weighting structures needs a probability for every structure")]
    ProbabilitiesRequired,
    /// A quantile or largest regret of a mixture is not a weighted average of the structures'
    /// values.
    #[error("the {criterion} of a mixture of structures is not supported")]
    MixtureCriterionUnsupported {
        /// The criterion.
        criterion: &'static str,
    },
    /// The execution context was cancelled between boxes, measure points or structures.
    #[error("the certified evaluation was cancelled")]
    Cancelled,
}

impl PreferenceCertifiedError {
    /// Stable detail id, prefixed `preference_certified.`; a contract, model, source or set-check
    /// refusal keeps its own detail, the detail the returned refusal carries.
    #[must_use]
    pub fn detail(&self) -> &'static str {
        use antecedent_core::reason_details;
        match self {
            Self::Contract(inner) => inner.detail(),
            Self::Model(inner) => inner.detail(),
            Self::Eval(inner) => inner.detail(),
            Self::StructureInputsRequired => {
                reason_details::preference_certified::STRUCTURE_INPUTS_REQUIRED
            }
            Self::InvalidOptions { .. } => reason_details::preference_certified::OPTIONS_INVALID,
            Self::MinimaxRegretNeedsExpectation => {
                reason_details::preference_certified::MINIMAX_REGRET_NEEDS_EXPECTATION
            }
            Self::MeasureRequired => reason_details::preference_certified::MEASURE_REQUIRED,
            Self::MeasureEmpty => reason_details::preference_certified::MEASURE_EMPTY,
            Self::MeasureUnknownParameter { .. } => {
                reason_details::preference_certified::MEASURE_UNKNOWN_PARAMETER
            }
            Self::MeasureIncomplete { .. } => {
                reason_details::preference_certified::MEASURE_INCOMPLETE
            }
            Self::MeasurePointDimension { .. } => {
                reason_details::preference_certified::MEASURE_POINT_DIMENSION
            }
            Self::MeasureNonFinitePoint { .. } => {
                reason_details::preference_certified::MEASURE_POINT_NON_FINITE
            }
            Self::MeasurePointOutsideSet { .. } => {
                reason_details::preference_certified::MEASURE_POINT_OUTSIDE_SET
            }
            Self::MeasureWeightsInvalid => {
                reason_details::preference_certified::MEASURE_WEIGHTS_INVALID
            }
            Self::MeasureLabelsInvalid => {
                reason_details::preference_certified::MEASURE_LABELS_INVALID
            }
            Self::EmptyPreferenceSet => reason_details::preference_certified::PREFERENCE_SET_EMPTY,
            Self::SetProgram(inner) => inner.detail(),
            Self::LogarithmDomain { .. } => reason_details::preference_certified::LOGARITHM_DOMAIN,
            Self::NonFiniteEnclosure { .. } => {
                reason_details::preference_certified::ENCLOSURE_NON_FINITE
            }
            Self::InvalidAtoms => reason_details::preference_certified::ATOMS_INVALID,
            Self::InvalidProbabilities => {
                reason_details::preference_certified::ATOM_PROBABILITIES_INVALID
            }
            Self::ProbabilitiesRequired => {
                reason_details::preference_certified::ATOM_PROBABILITIES_REQUIRED
            }
            Self::MixtureCriterionUnsupported { .. } => {
                reason_details::preference_certified::MIXTURE_CRITERION_UNSUPPORTED
            }
            Self::Cancelled => reason_details::preference_certified::CANCELLED,
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------------------------

/// The validated contract, options and measure shared by every run.
struct Setup<'c> {
    contract: &'c PreferenceContract,
    identity: String,
    measure: Option<&'c PreferenceMeasure>,
    options: CertifiedOptions,
    /// A point of the declared ranges and linear relations.
    interior: Vec<f64>,
    ctx: Option<&'c ExecutionContext>,
}

/// `Err(Cancelled)` when a context is given and its token is set.
fn observe(ctx: Option<&ExecutionContext>) -> Result<(), PreferenceCertifiedError> {
    if ctx.is_some_and(|ctx| ctx.cancellation.is_cancelled()) {
        return Err(PreferenceCertifiedError::Cancelled);
    }
    Ok(())
}

fn validate_options(options: CertifiedOptions) -> Result<(), PreferenceCertifiedError> {
    let invalid = |option| Err(PreferenceCertifiedError::InvalidOptions { option });
    if options.max_boxes == 0 {
        return invalid("max_boxes");
    }
    let width = options.min_width_fraction;
    let width_valid = width.is_finite() && (0.0..=1.0).contains(&width);
    if !width_valid {
        return invalid("min_width_fraction");
    }
    let tie = options.tie_tolerance;
    let tie_valid = tie.is_finite() && tie >= 0.0;
    if !tie_valid {
        return invalid("tie_tolerance");
    }
    Ok(())
}

fn validate_measure(
    measure: &PreferenceMeasure,
    model: &ValueModel,
) -> Result<(), PreferenceCertifiedError> {
    use PreferenceCertifiedError as E;
    let set = &model.preferences;
    if measure.points.is_empty() {
        return Err(E::MeasureEmpty);
    }
    if let Some(index) = measure.unresolved.iter().copied().find(|j| *j >= set.dimension()) {
        return Err(E::MeasureUnknownParameter { index });
    }
    if !measure.unresolved.is_empty() {
        let mut indices = measure.unresolved.clone();
        indices.sort_unstable();
        indices.dedup();
        return Err(E::MeasureIncomplete {
            parameters: indices.into_iter().map(|j| set.parameters[j].id.clone()).collect(),
        });
    }
    for (index, point) in measure.points.iter().enumerate() {
        if point.len() != set.dimension() {
            return Err(E::MeasurePointDimension {
                index,
                expected: set.dimension(),
                found: point.len(),
            });
        }
        if point.iter().any(|value| !value.is_finite()) {
            return Err(E::MeasureNonFinitePoint { index });
        }
        if !model.contains(point, MEASURE_SET_TOLERANCE) {
            return Err(E::MeasurePointOutsideSet { index });
        }
    }
    if let Some(weights) = &measure.weights {
        let valid = weights.len() == measure.points.len()
            && weights.iter().all(|w| w.is_finite() && *w >= 0.0)
            && weights.iter().sum::<f64>() > 0.0;
        if !valid {
            return Err(E::MeasureWeightsInvalid);
        }
    }
    if measure.labels.as_ref().is_some_and(|labels| labels.len() != measure.points.len()) {
        return Err(E::MeasureLabelsInvalid);
    }
    Ok(())
}

impl<'c> Setup<'c> {
    fn new(
        contract: &'c PreferenceContract,
        measure: Option<&'c PreferenceMeasure>,
        options: CertifiedOptions,
        ctx: Option<&'c ExecutionContext>,
    ) -> Result<Self, PreferenceCertifiedError> {
        use PreferenceCertifiedError as E;
        contract.validate().map_err(E::Contract)?;
        if matches!(
            contract.criterion,
            DecisionCriterion::MinimaxOverIdentifiedSet | DecisionCriterion::MaximinOverStructures
        ) {
            return Err(E::StructureInputsRequired);
        }
        validate_options(options)?;
        if contract.preference_policy == PreferencePolicy::MinimaxRegret
            && !is_expectation(contract.criterion)
        {
            return Err(E::MinimaxRegretNeedsExpectation);
        }
        // Relations that conflict only jointly, or leave a gap narrower than the smallest box,
        // never prove a box outside: the linear program decides whether any point remains.
        let interior = linear_set_point(&contract.model.preferences, LpOptions::default())
            .map_err(E::SetProgram)?
            .ok_or(E::EmptyPreferenceSet)?;
        let measure = if contract.preference_policy == PreferencePolicy::Weighted {
            let measure = measure.ok_or(E::MeasureRequired)?;
            validate_measure(measure, &contract.model)?;
            Some(measure)
        } else {
            None
        };
        let identity = contract.identity().map_err(E::Contract)?;
        Ok(Self { contract, identity, measure, options, interior, ctx })
    }

    fn assumptions(&self) -> Vec<String> {
        let mut out: Vec<String> = self
            .contract
            .model
            .preferences
            .parameters
            .iter()
            .map(|p| p.assumption_id.clone())
            .collect();
        if let Some(measure) = self.measure {
            out.push(match &measure.origin {
                MeasureOrigin::Declared { assumption_id } => assumption_id.clone(),
                MeasureOrigin::ChoiceEstimate { estimate_identity } => estimate_identity.clone(),
            });
        }
        out.sort();
        out.dedup();
        out
    }

    /// Evaluate over `sources` (non-empty) combined as stated.
    fn run(
        &self,
        sources: &[SourceData<'_>],
        combination: Combination,
    ) -> Result<CertifiedPreferenceResult, PreferenceCertifiedError> {
        let contract = self.contract;
        let n = contract.actions.len();
        // Under Bayes weighting a structure without probability excludes no action, as for the
        // ordinary contract.
        let excluding: Vec<&SourceData<'_>> = match &combination {
            Combination::Bayes(probabilities) => sources
                .iter()
                .zip(probabilities)
                .filter(|(_, p)| **p > 0.0)
                .map(|(s, _)| s)
                .collect(),
            Combination::Single | Combination::Worst => sources.iter().collect(),
        };
        let admissible: Vec<bool> =
            (0..n).map(|a| excluding.iter().all(|s| s.exclusions[a].is_empty())).collect();
        let mut actions: Vec<CertifiedActionOutcome> = contract
            .actions
            .iter()
            .enumerate()
            .map(|(a, action)| CertifiedActionOutcome {
                id: action.id.clone(),
                admissible: admissible[a],
                exclusions: excluding
                    .iter()
                    .flat_map(|s| s.exclusions[a].iter().cloned())
                    .collect(),
                best_fraction: 0.0,
                value_bounds: None,
                worst_case_bounds: None,
                worst_case_regret_bounds: None,
                measure_value: None,
                measure_share: None,
            })
            .collect();
        let plan = relation_plan(&contract.model.preferences);
        let engine = Engine {
            contract,
            sources,
            combination,
            admissible,
            options: self.options,
            declared: plan.subdivided,
            ranges: contract.model.preferences.bounds(),
            relations: plan.relations,
            eliminations: plan.eliminations,
            interior: &self.interior,
            ctx: self.ctx,
        };
        let sign = orientation(contract.criterion);
        let any_admissible = engine.admissible.iter().any(|a| *a);
        let outcome = match self.measure {
            _ if !any_admissible => Subdivided {
                regions: Vec::new(),
                undecided_fraction: 0.0,
                boxes_examined: 0,
                verdict: CertifiedVerdict::NoAdmissibleAction,
                effective_tie_tolerance: self.options.tie_tolerance,
            },
            Some(measure) => {
                let summary = weighted_policy(&engine, measure)?;
                for (a, outcome) in actions.iter_mut().enumerate() {
                    outcome.measure_value = summary.value[a].map(|v| sign * v);
                    outcome.measure_share = summary.share[a];
                }
                Subdivided {
                    regions: Vec::new(),
                    undecided_fraction: 0.0,
                    boxes_examined: 0,
                    verdict: summary.verdict,
                    effective_tie_tolerance: summary.effective_tie_tolerance,
                }
            }
            None => subdivided(&engine, &mut actions)?,
        };
        Ok(CertifiedPreferenceResult {
            contract_identity: self.identity.clone(),
            policy: contract.preference_policy,
            parameter_ids: contract
                .model
                .preferences
                .parameters
                .iter()
                .map(|p| p.id.clone())
                .collect(),
            actions,
            regions: outcome.regions,
            undecided_fraction: outcome.undecided_fraction,
            boxes_examined: outcome.boxes_examined,
            verdict: outcome.verdict,
            assumptions: self.assumptions(),
            source: sources.first().map_or_else(
                || SourceReceipt {
                    provider_id: String::new(),
                    snapshot_id: String::new(),
                    rng_id: String::new(),
                    causal_contract_id: String::new(),
                },
                |s| s.receipt.clone(),
            ),
            n_draws: sources.iter().map(|s| s.weights.len()).sum(),
            effective_tie_tolerance: outcome.effective_tie_tolerance,
        })
    }
}

/// Decide a preference contract across its preference set by certified subdivision on one
/// aligned joint source.
///
/// The criterion may be `PosteriorExpectedUtility`, `PosteriorExpectedLoss`, `ExpectedRegret`,
/// `ThresholdProbability`, `Quantile` or `Regret`. Policies:
/// - `RequireInvariantBestAction`: `NecessarilyBest` when every kept box is certified for the
///   same action; `PreferenceDependent` when at least two actions are each certified on some
///   box wholly inside the relations; otherwise `Undetermined`. With value relations a choosing
///   verdict also needs the set shown to hold a point (see the module documentation).
/// - `Maximin`: compares the enclosures of each action's worst value over the set and chooses
///   only when one lies above every rival's by more than the tie tolerance.
/// - `MinimaxRegret`: expectation criteria only; compares worst-case regret enclosures built
///   from the pairwise margin enclosures.
/// - `Weighted`: needs a measure with every parameter resolved; each measure point is evaluated
///   exactly, through [`PreferenceContract::bind`] and [`evaluate_contract`] when the bound value
///   can be lowered and otherwise by evaluating the model on each draw with the same criterion
///   rules, for the reported values; no boxes are examined. Leads are read from enclosures, not
///   from those floating values, since a value that cancels large terms can lead by pure
///   rounding: each point is also evaluated in outward-rounded interval arithmetic with every
///   parameter a degenerate interval, and a point counts toward an action's share when the
///   lower end of its margin enclosure over every admissible rival exceeds the tie tolerance.
///   The measure-weighted margin enclosures (outward-rounded weighted sums of the point
///   enclosures) decide the choice the same way.
/// - `ReportOnly`: everything computed, verdict `ReportOnly`.
///
/// A measure is read only under the `Weighted` policy. Excluded actions take part in no
/// comparison.
///
/// # Errors
/// An invalid contract; a criterion that needs structure inputs; options out of range;
/// `MinimaxRegret` with a criterion that is not an expectation; declared ranges and linear
/// relations that admit no point, or a refusal of the linear program that checks them; under
/// `Weighted`, a missing, incomplete or invalid measure; a source that cannot answer the
/// contract; relations that exclude every box; a logarithm undefined or an enclosure non-finite
/// on an examined box; a model value that cannot be evaluated at a measure point.
pub fn evaluate_preference_certified(
    contract: &PreferenceContract,
    source: &DistributionArtifact,
    measure: Option<&PreferenceMeasure>,
    options: CertifiedOptions,
) -> Result<CertifiedPreferenceResult, PreferenceCertifiedError> {
    certified(contract, source, measure, options, None)
}

/// [`evaluate_preference_certified`] under an execution context: the context's cancellation
/// token is read before every examined box and every measure point.
///
/// # Errors
/// As [`evaluate_preference_certified`]; [`PreferenceCertifiedError::Cancelled`] once the token
/// is set.
pub fn evaluate_preference_certified_with_context(
    contract: &PreferenceContract,
    source: &DistributionArtifact,
    measure: Option<&PreferenceMeasure>,
    options: CertifiedOptions,
    ctx: &ExecutionContext,
) -> Result<CertifiedPreferenceResult, PreferenceCertifiedError> {
    certified(contract, source, measure, options, Some(ctx))
}

pub(crate) fn certified(
    contract: &PreferenceContract,
    source: &DistributionArtifact,
    measure: Option<&PreferenceMeasure>,
    options: CertifiedOptions,
    ctx: Option<&ExecutionContext>,
) -> Result<CertifiedPreferenceResult, PreferenceCertifiedError> {
    let setup = Setup::new(contract, measure, options, ctx)?;
    let data = prepare_source(contract, source)?;
    setup.run(std::slice::from_ref(&data), Combination::Single)
}

fn validate_atoms(atoms: &[StructuralAtom]) -> Result<bool, PreferenceCertifiedError> {
    use PreferenceCertifiedError as E;
    let mut seen = std::collections::BTreeSet::new();
    if atoms.is_empty()
        || atoms.iter().any(|a| a.id.trim().is_empty() || !seen.insert(a.id.as_str()))
    {
        return Err(E::InvalidAtoms);
    }
    let with = atoms.iter().filter(|a| a.probability.is_some()).count();
    if with == 0 {
        return Ok(false);
    }
    let invalid = with != atoms.len()
        || atoms.iter().any(|a| a.probability.is_some_and(|p| !p.is_finite() || p < 0.0))
        || probability_sum(atoms.iter().filter_map(|a| a.probability)) > 1.0 + 1e-9;
    if invalid { Err(E::InvalidProbabilities) } else { Ok(true) }
}

/// The action a verdict chooses, when it chooses one.
fn chosen(verdict: &CertifiedVerdict) -> Option<&str> {
    match verdict {
        CertifiedVerdict::NecessarilyBest(a)
        | CertifiedVerdict::WorstCaseChoice(a)
        | CertifiedVerdict::MinimaxRegretChoice(a)
        | CertifiedVerdict::MeasureChoice { action: a, .. } => Some(a.as_str()),
        _ => None,
    }
}

fn invariant_over_atoms(
    summaries: &[CertifiedAtomSummary],
    missing: Option<&'static str>,
) -> StructuralCertifiedVerdict {
    let verdicts: Vec<AtomVerdict> = summaries
        .iter()
        .filter_map(|s| match &s.status {
            CertifiedAtomStatus::Evaluated(result) => {
                Some(AtomVerdict { atom: s.id.clone(), verdict: result.verdict.clone() })
            }
            _ => None,
        })
        .collect();
    let Some(first) = verdicts.first() else {
        return StructuralCertifiedVerdict::InsufficientScience(NO_STRUCTURE_EVALUATED.into());
    };
    if let Some(why) = missing {
        return StructuralCertifiedVerdict::InsufficientScience(why.into());
    }
    let identical = verdicts.iter().all(|v| v.verdict == first.verdict);
    let same_choice = chosen(&first.verdict)
        .is_some_and(|action| verdicts.iter().all(|v| chosen(&v.verdict) == Some(action)));
    if !(identical || same_choice) {
        return StructuralCertifiedVerdict::StructuresDisagree(verdicts);
    }
    // Measure choices of one action agree with the smallest share, as on the exact route.
    let smallest = verdicts
        .iter()
        .map(|v| match &v.verdict {
            CertifiedVerdict::MeasureChoice { share, .. } => Some(*share),
            _ => None,
        })
        .collect::<Option<Vec<f64>>>()
        .and_then(|shares| shares.into_iter().reduce(f64::min));
    match (&first.verdict, smallest) {
        (CertifiedVerdict::MeasureChoice { action, .. }, Some(share)) => {
            StructuralCertifiedVerdict::Combined(CertifiedVerdict::MeasureChoice {
                action: action.clone(),
                share,
            })
        }
        _ => StructuralCertifiedVerdict::Combined(first.verdict.clone()),
    }
}

/// Per-structure summaries, the evaluated sources in atom order and their probabilities.
type EvaluatedAtoms<'s> = (Vec<CertifiedAtomSummary>, Vec<SourceData<'s>>, Vec<f64>);

/// Decide each structure on its own. A structure whose source cannot answer the contract is
/// kept as unevaluated with the registered reason code.
fn evaluate_atoms<'s>(
    setup: &Setup<'_>,
    atoms: &'s [StructuralAtom],
) -> Result<EvaluatedAtoms<'s>, PreferenceCertifiedError> {
    let mut summaries = Vec::with_capacity(atoms.len());
    let mut sources = Vec::new();
    let mut probabilities = Vec::new();
    for atom in atoms {
        observe(setup.ctx)?;
        let status = match &atom.evidence {
            AtomEvidence::Unidentified => CertifiedAtomStatus::Unidentified,
            AtomEvidence::Unevaluated(reason) => CertifiedAtomStatus::Unevaluated(reason.clone()),
            AtomEvidence::Evaluated(artifact) => match prepare_source(setup.contract, artifact) {
                Ok(data) => {
                    let result = setup.run(std::slice::from_ref(&data), Combination::Single)?;
                    sources.push(data);
                    probabilities.push(atom.probability.unwrap_or(0.0));
                    CertifiedAtomStatus::Evaluated(Box::new(result))
                }
                Err(PreferenceCertifiedError::Eval(error))
                    if !matches!(error, DecisionEvalError::Contract(_)) =>
                {
                    CertifiedAtomStatus::Unevaluated(error.reason_code().to_owned())
                }
                Err(other) => return Err(other),
            },
        };
        summaries.push(CertifiedAtomSummary {
            id: atom.id.clone(),
            probability: atom.probability,
            status,
        });
    }
    Ok((summaries, sources, probabilities))
}

/// Decide a preference contract across structures and its preference set.
///
/// Every evaluated structure is first decided on its own (reported in `atoms`); a structure
/// whose source cannot answer the contract is kept as unevaluated with the registered reason
/// code. The contract's structural policy then combines them:
/// - `RequireInvariantBestAction`: every structure must be evaluated and every structure's
///   verdict must be the same or choose the same action (measure choices of one action combine
///   with the smallest share); otherwise the structures' verdicts are listed as a disagreement,
///   or the claim is insufficient science.
/// - `Maximin`: one joint subdivision whose score on a box is the worst over structures of each
///   structure's enclosure; a choice needs every structure evaluated.
/// - `BayesOverStructures`: needs a probability on every structure; one joint subdivision whose
///   score and margins are the probability-weighted sums of the structures' enclosures, not
///   renormalized; a quantile or largest-regret criterion refuses; a choice needs no unresolved
///   mass.
/// - `ReportOnly`: per-structure results only.
///
/// An action excluded by a hard constraint in any evaluated structure is excluded overall;
/// under `BayesOverStructures` only a structure with positive probability excludes, as for the
/// ordinary contract.
///
/// # Errors
/// Every refusal of [`evaluate_preference_certified`] other than a source that cannot answer
/// the contract; no structures or blank or repeated ids; invalid or missing probabilities; a
/// mixture criterion that is not a weighted average.
pub fn evaluate_preference_certified_structural(
    contract: &PreferenceContract,
    atoms: &[StructuralAtom],
    measure: Option<&PreferenceMeasure>,
    options: CertifiedOptions,
) -> Result<StructuralCertifiedResult, PreferenceCertifiedError> {
    certified_structural(contract, atoms, measure, options, None)
}

/// [`evaluate_preference_certified_structural`] under an execution context: the context's
/// cancellation token is read before every structure, every examined box and every measure
/// point.
///
/// # Errors
/// As [`evaluate_preference_certified_structural`]; [`PreferenceCertifiedError::Cancelled`]
/// once the token is set.
pub fn evaluate_preference_certified_structural_with_context(
    contract: &PreferenceContract,
    atoms: &[StructuralAtom],
    measure: Option<&PreferenceMeasure>,
    options: CertifiedOptions,
    ctx: &ExecutionContext,
) -> Result<StructuralCertifiedResult, PreferenceCertifiedError> {
    certified_structural(contract, atoms, measure, options, Some(ctx))
}

pub(crate) fn certified_structural(
    contract: &PreferenceContract,
    atoms: &[StructuralAtom],
    measure: Option<&PreferenceMeasure>,
    options: CertifiedOptions,
    ctx: Option<&ExecutionContext>,
) -> Result<StructuralCertifiedResult, PreferenceCertifiedError> {
    use PreferenceCertifiedError as E;
    let setup = Setup::new(contract, measure, options, ctx)?;
    let has_probabilities = validate_atoms(atoms)?;
    let policy = contract.structural_policy;
    if policy == StructuralPolicy::BayesOverStructures {
        if !has_probabilities {
            return Err(E::ProbabilitiesRequired);
        }
        match contract.criterion {
            DecisionCriterion::Quantile { .. } => {
                return Err(E::MixtureCriterionUnsupported { criterion: "quantile" });
            }
            DecisionCriterion::Regret => {
                return Err(E::MixtureCriterionUnsupported { criterion: "regret" });
            }
            _ => {}
        }
    }
    let (summaries, sources, probabilities) = evaluate_atoms(&setup, atoms)?;
    let states: Vec<(Option<f64>, StructureState)> = summaries
        .iter()
        .map(|s| {
            let state = match s.status {
                CertifiedAtomStatus::Evaluated(_) => StructureState::Evaluated,
                CertifiedAtomStatus::Unidentified => StructureState::Unidentified,
                CertifiedAtomStatus::Unevaluated(_) => StructureState::Unevaluated,
            };
            (s.probability, state)
        })
        .collect();
    let StructureMasses {
        evaluated: evaluated_mass,
        unidentified: unidentified_mass,
        unevaluated: unevaluated_mass,
    } = structure_masses(has_probabilities, &states);
    let all_evaluated = sources.len() == atoms.len();
    let missing = missing_evidence(
        policy_kind(policy),
        all_evaluated,
        unidentified_mass.unwrap_or(0.0),
        unevaluated_mass.unwrap_or(0.0),
        evaluated_mass.unwrap_or(0.0),
    );
    let insufficient = |why: &str| StructuralCertifiedVerdict::InsufficientScience(why.into());
    let (combined, verdict) = match policy {
        StructuralPolicy::ReportOnly => (None, StructuralCertifiedVerdict::ReportOnly),
        StructuralPolicy::RequireInvariantBestAction => {
            (None, invariant_over_atoms(&summaries, missing))
        }
        _ if sources.is_empty() => (None, insufficient(NO_STRUCTURE_EVALUATED)),
        StructuralPolicy::Maximin => {
            let combined = setup.run(&sources, Combination::Worst)?;
            let verdict = match missing {
                Some(why) => insufficient(why),
                None => StructuralCertifiedVerdict::Combined(combined.verdict.clone()),
            };
            (Some(combined), verdict)
        }
        StructuralPolicy::BayesOverStructures => {
            let combined = setup.run(&sources, Combination::Bayes(probabilities))?;
            let verdict = match missing {
                Some(why) => insufficient(why),
                None => StructuralCertifiedVerdict::Combined(combined.verdict.clone()),
            };
            (Some(combined), verdict)
        }
    };
    Ok(StructuralCertifiedResult {
        contract_identity: setup.identity.clone(),
        structural_policy: policy,
        preference_policy: contract.preference_policy,
        atoms: summaries,
        combined,
        unidentified_mass,
        unevaluated_mass,
        evaluated_mass,
        verdict,
    })
}

impl PreferenceCertifiedError {
    /// Structured refusal for a failure of the certified evaluation across a preference set.
    ///
    /// A contract, value-model, source or set-check refusal keeps its own refusal.
    #[must_use]
    #[allow(clippy::too_many_lines, reason = "one arm per certified-evaluation failure")]
    pub fn to_refusal(&self) -> ExternalRefusal {
        let invalid = |stage, remedy| {
            refusal(antecedent_core::reason_code!("invalid_argument"), stage, self.detail())
                .remedy(remedy)
        };
        let unsatisfied = |stage, remedy| {
            refusal(
                antecedent_core::reason_code!("decision_contract_unsatisfied"),
                stage,
                self.detail(),
            )
            .remedy(remedy)
        };
        let unsupported = |stage, remedy| {
            refusal(antecedent_core::reason_code!("route_not_supported"), stage, self.detail())
                .remedy(remedy)
        };
        let corners = |lower: &[f64], upper: &[f64]| format!("{lower:?}..{upper:?}");
        match self {
            Self::Contract(error) => at_stage(error.to_refusal(), EVALUATE),
            Self::Model(error) => at_stage(error.to_refusal(), EVALUATE),
            Self::Eval(error) => error.to_refusal(),
            Self::StructureInputsRequired => unsupported(
                EVALUATE,
                "evaluate across structures with evaluate_preference_certified_structural",
            )
            .build(),
            Self::InvalidOptions { option } => {
                invalid(EVALUATE, "supply the named option inside its documented range")
                    .offending(*option)
                    .build()
            }
            Self::MinimaxRegretNeedsExpectation => unsupported(
                EVALUATE,
                "use an expectation criterion, or choose another preference policy",
            )
            .build(),
            Self::MeasureRequired => unsatisfied(
                EVALUATE,
                "supply a measure over the preference set, or choose another preference policy",
            )
            .build(),
            Self::MeasureEmpty => invalid(EVALUATE, "supply at least one measure point").build(),
            Self::MeasureUnknownParameter { index } => {
                invalid(EVALUATE, "name only declared parameter positions as unresolved")
                    .offending(index.to_string())
                    .build()
            }
            Self::MeasureIncomplete { parameters } => {
                unsatisfied(EVALUATE, "supply a measure that resolves every parameter")
                    .offending(parameters.join(","))
                    .build()
            }
            Self::MeasurePointDimension { index, expected, found } => {
                invalid(EVALUATE, "supply one value per preference parameter")
                    .offending(index.to_string())
                    .expected(expected.to_string())
                    .supplied(found.to_string())
                    .build()
            }
            Self::MeasureNonFinitePoint { index } => {
                invalid(EVALUATE, "supply finite measure points")
                    .offending(index.to_string())
                    .build()
            }
            Self::MeasurePointOutsideSet { index } => {
                invalid(EVALUATE, "supply measure points inside the declared ranges and relations")
                    .offending(index.to_string())
                    .build()
            }
            Self::MeasureWeightsInvalid => invalid(
                EVALUATE,
                "supply one finite, non-negative weight per measure point with a positive total",
            )
            .build(),
            Self::MeasureLabelsInvalid => {
                invalid(EVALUATE, "supply one label per measure point").build()
            }
            Self::EmptyPreferenceSet => {
                invalid(EVALUATE, "widen the declared ranges or remove a relation").build()
            }
            Self::SetProgram(error) => at_stage(error.to_refusal(), EVALUATE),
            Self::LogarithmDomain { action, lower, upper } => invalid(
                EVALUATE,
                "keep the logarithm's argument positive: narrow the ranges or change the value",
            )
            .offending(action.clone())
            .supplied(corners(lower, upper))
            .build(),
            Self::NonFiniteEnclosure { subject, lower, upper } => invalid(
                EVALUATE,
                "narrow the parameter ranges, or change the value so it stays finite",
            )
            .offending(subject.clone())
            .supplied(corners(lower, upper))
            .build(),
            Self::InvalidAtoms => {
                invalid(STRUCTURAL, "supply at least one structure with a distinct, non-blank id")
                    .build()
            }
            Self::InvalidProbabilities => unsatisfied(
                STRUCTURAL,
                "give each structure a finite, non-negative probability summing to at most one",
            )
            .build(),
            Self::ProbabilitiesRequired => unsatisfied(
                STRUCTURAL,
                "supply genuine structure probabilities, or choose another structural policy",
            )
            .build(),
            Self::MixtureCriterionUnsupported { criterion } => unsupported(
                STRUCTURAL,
                "use an expectation criterion, or choose another structural policy",
            )
            .offending(*criterion)
            .build(),
            Self::Cancelled => refusal(
                antecedent_core::reason_code!("cancelled_no_claim"),
                EVALUATE,
                self.detail(),
            )
            .build(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::preference_model::PreferenceExpr;

    /// A deterministic generator for sampled points.
    struct Lcg(u64);

    impl Lcg {
        fn unit(&mut self) -> f64 {
            self.0 = self
                .0
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            f64::from(u32::try_from(self.0 >> 40).unwrap()) / f64::from(1_u32 << 24)
        }

        fn between(&mut self, lo: f64, hi: f64) -> f64 {
            lo + (hi - lo) * self.unit()
        }
    }

    fn contains(interval: Interval, value: f64) -> bool {
        interval.lo <= value && value <= interval.hi
    }

    fn sample(interval: Interval, rng: &mut Lcg) -> f64 {
        rng.between(interval.lo, interval.hi)
    }

    #[test]
    fn arithmetic_encloses_sampled_points() {
        let mut rng = Lcg(7);
        for _ in 0..2000 {
            let a = Interval { lo: rng.between(-5.0, 1.0), hi: 0.0 };
            let a = Interval { lo: a.lo, hi: a.lo + rng.between(0.0, 6.0) };
            let b = Interval { lo: rng.between(-3.0, 3.0), hi: 0.0 };
            let b = Interval { lo: b.lo, hi: b.lo + rng.between(0.0, 4.0) };
            let x = sample(a, &mut rng);
            let y = sample(b, &mut rng);
            assert!(contains(a.plus(b).unwrap(), x + y));
            assert!(contains(a.minus(b).unwrap(), x - y));
            assert!(contains(a.times(b).unwrap(), x * y));
            assert!(contains(a.min(b), x.min(y)));
            assert!(contains(a.max(b), x.max(y)));
            assert!(contains(a.negated(), -x));
            assert!(contains(a.exp().unwrap(), x.exp()));
            let positive = Interval { lo: a.lo.abs() + 0.01, hi: a.hi.abs() + a.lo.abs() + 0.02 };
            let z = sample(positive, &mut rng);
            assert!(contains(positive.ln().unwrap(), z.ln()));
        }
    }

    #[test]
    fn rounding_moves_each_end_outward() {
        let third = Interval::point(1.0).times(Interval::point(1.0 / 3.0)).unwrap();
        assert!(third.lo < 1.0 / 3.0 && third.hi > 1.0 / 3.0);
        let sum = Interval::point(0.1).plus(Interval::point(0.2)).unwrap();
        assert!(sum.lo < 0.1 + 0.2 && sum.hi > 0.1 + 0.2);
    }

    #[test]
    fn logarithm_of_an_interval_reaching_zero_is_refused() {
        assert_eq!(Interval { lo: 0.0, hi: 1.0 }.ln(), Err(EncloseFailure::LnDomain));
        assert_eq!(Interval { lo: -1.0, hi: 1.0 }.ln(), Err(EncloseFailure::LnDomain));
        assert_eq!(Interval::point(800.0).exp(), Err(EncloseFailure::NonFinite));
        assert_eq!(
            Interval::point(f64::MAX).plus(Interval::point(f64::MAX)),
            Err(EncloseFailure::NonFinite)
        );
    }

    #[test]
    fn expression_enclosures_contain_point_values_over_the_box() {
        use PreferenceExpr as P;
        // x0 * d + max(x1 - k, 0) * d * d - exp(-r * x0) + ln(1 + k)
        let expr = P::sum(
            P::sum(
                P::product(P::Attribute(0), P::Parameter(0)),
                P::product(
                    P::maximum(P::difference(P::Attribute(1), P::Parameter(1)), P::Const(0.0)),
                    P::product(P::Parameter(0), P::Parameter(0)),
                ),
            ),
            P::difference(
                P::ln(P::sum(P::Const(1.0), P::Parameter(1))),
                P::exp(P::negate(P::product(P::Parameter(2), P::Attribute(0)))),
            ),
        );
        let mut rng = Lcg(11);
        for _ in 0..300 {
            let attributes = [rng.between(-3.0, 3.0), rng.between(-3.0, 3.0)];
            let lower = [rng.between(0.2, 1.0), rng.between(0.0, 2.0), rng.between(0.0, 1.5)];
            let upper = [lower[0] + 0.3, lower[1] + 0.5, lower[2] + 0.2];
            let params: Vec<Interval> =
                lower.iter().zip(&upper).map(|(l, u)| Interval { lo: *l, hi: *u }).collect();
            let enclosure = enclose(&expr, &attributes, &params).unwrap();
            for _ in 0..20 {
                let point: Vec<f64> = params.iter().map(|p| sample(*p, &mut rng)).collect();
                let value = expr.evaluate(&attributes, &point).unwrap();
                assert!(contains(enclosure, value), "{value} outside {enclosure:?}");
            }
        }
    }

    #[test]
    fn weighted_sums_and_affine_forms_enclose_exact_values() {
        let values = [Interval::point(0.1), Interval { lo: -2.0, hi: 3.0 }, Interval::point(7.3)];
        let weights = [0.2, 0.0, 0.8];
        let sum = weighted_sum(weights.iter().copied().zip(values)).unwrap();
        assert!(contains(sum, 0.2 * 0.1 + 0.8 * 7.3));
        let constant = Interval::point(1.0);
        let coefficients = [Interval::point(2.0), Interval::point(0.0), Interval::point(-1.0)];
        let params = [
            Interval { lo: 0.0, hi: 1.0 },
            Interval { lo: -9.0, hi: 9.0 },
            Interval { lo: 1.0, hi: 2.0 },
        ];
        let range = affine(constant, &coefficients, &params).unwrap();
        // Exact range of 1 + 2a - c over the box: [-1, 2].
        assert!(contains(range, -1.0) && contains(range, 2.0));
        assert!(range.lo > -1.0 - 1e-12 && range.hi < 2.0 + 1e-12);
    }

    #[test]
    fn the_quantile_rule_is_monotone_in_the_values() {
        let weights = [0.25, 0.25, 0.5];
        let mut rng = Lcg(3);
        for _ in 0..500 {
            let values: Vec<f64> = (0..3).map(|_| rng.between(-2.0, 2.0)).collect();
            let lower: Vec<f64> = values.iter().map(|v| v - rng.between(0.0, 1.0)).collect();
            for p in [0.1, 0.25, 0.5, 0.75, 0.9] {
                let low = left_inverse(&lower, &weights, p).unwrap();
                let at = left_inverse(&values, &weights, p).unwrap();
                assert!(low <= at);
            }
        }
        // A level on an atom boundary selects that atom.
        assert!((left_inverse(&[1.0, 2.0, 3.0], &weights, 0.5).unwrap() - 2.0).abs() < 1e-15);
    }

    #[test]
    fn queue_order_is_volume_then_lower_corner() {
        let mut queue = BinaryHeap::new();
        queue.push(Queued { volume: 0.25, lower: vec![0.5], slot: 0 });
        queue.push(Queued { volume: 0.5, lower: vec![0.5], slot: 1 });
        queue.push(Queued { volume: 0.25, lower: vec![0.0], slot: 2 });
        let order: Vec<usize> = std::iter::from_fn(|| queue.pop().map(|q| q.slot)).collect();
        assert_eq!(order, vec![1, 2, 0]);
    }
}
