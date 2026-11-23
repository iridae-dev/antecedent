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
//! Relations between parameters are checked per box with the same arithmetic: a box whose
//! relation left side is proven to violate some relation everywhere is dropped; a box on which
//! some relation is neither proven satisfied nor proven violated is kept and flagged as touching
//! the relation boundary. Fractions are volume fractions of the kept boxes relative to their
//! total volume, measured relative to the declared ranges, with a fixed parameter (zero width)
//! contributing no dimension; a boundary box counts in full, although part of it may lie
//! outside the set. Splitting is deterministic: the undecided box of largest volume (ties: the
//! lexicographically smallest lower corner) is split at the midpoint of the parameter whose
//! width is largest relative to its declared range, until every box is decided, the box budget
//! is spent, or every width is at most the minimum fraction of its declared range.
//!
//! Hard constraints never read parameters. Their exclusions are read once: from the ordinary
//! contract bound at the midpoint of the declared ranges and evaluated by
//! [`evaluate_contract`] when the bound value can be lowered, and otherwise (an `Exp` or `Ln`
//! wraps a quantity) by evaluating each constraint on the drawn attributes with the same rule,
//! `P(expr <= bound) >= min_probability`. A result over the preference set is an assumption
//! range, never a probability, unless a measure over the set is supplied.

// SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Ordering;
use std::collections::BinaryHeap;

use antecedent_io_core::distribution_artifact::DistributionArtifact;

use crate::decision_contract::{DecisionCriterion, HardConstraint, StructuralPolicy};
use crate::decision_eval::{
    ConstraintExclusion, DecisionEvalError, SourceReceipt, constraint_fails, evaluate_contract,
    utility_draws,
};
use crate::decision_structural::{AtomEvidence, StructuralAtom};
use crate::preference_contract::{
    ActionInput, PreferenceAction, PreferenceContract, PreferenceContractError, PreferencePolicy,
};
use crate::preference_decision::{MeasureOrigin, PreferenceMeasure};
use crate::preference_model::{
    LinearForm, PreferenceExpr, PreferenceModelError, PreferenceSet, RelationSense,
};
use crate::structure_aggregator::{
    INVARIANCE_UNCHECKED, NO_EVALUATED_POSITIVE_PROBABILITY, NO_STRUCTURE_EVALUATED,
    UNRESOLVED_MASS_CAN_CHANGE_BAYES, WORST_CASE_NEEDS_EVERY_STRUCTURE, probability_sum,
    unassigned_probability,
};

/// Distance from the declared set within which a measure point is accepted.
const MEASURE_SET_TOLERANCE: f64 = 1e-9;

/// Relative roundoff tolerance the decision layer uses when comparing cumulative weights with
/// a quantile level.
const MASS_TOLERANCE: f64 = 8.0 * f64::EPSILON;

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
    /// margin by which a worst-case, regret or measure-weighted value must lead.
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
}

impl PreferenceCertifiedError {
    /// Stable detail id, prefixed `preference_certified.`; a contract or model refusal keeps its
    /// own detail.
    #[must_use]
    pub fn detail(&self) -> &'static str {
        match self {
            Self::Contract(inner) => inner.detail(),
            Self::Model(inner) => inner.detail(),
            Self::Eval(_) => "preference_certified.source_refused",
            Self::StructureInputsRequired => "preference_certified.structure_inputs_required",
            Self::InvalidOptions { .. } => "preference_certified.options_invalid",
            Self::MinimaxRegretNeedsExpectation => {
                "preference_certified.minimax_regret_needs_expectation"
            }
            Self::MeasureRequired => "preference_certified.measure_required",
            Self::MeasureEmpty => "preference_certified.measure_empty",
            Self::MeasureUnknownParameter { .. } => {
                "preference_certified.measure_unknown_parameter"
            }
            Self::MeasureIncomplete { .. } => "preference_certified.measure_incomplete",
            Self::MeasurePointDimension { .. } => "preference_certified.measure_point_dimension",
            Self::MeasureNonFinitePoint { .. } => "preference_certified.measure_point_non_finite",
            Self::MeasurePointOutsideSet { .. } => "preference_certified.measure_point_outside_set",
            Self::MeasureWeightsInvalid => "preference_certified.measure_weights_invalid",
            Self::MeasureLabelsInvalid => "preference_certified.measure_labels_invalid",
            Self::EmptyPreferenceSet => "preference_certified.preference_set_empty",
            Self::LogarithmDomain { .. } => "preference_certified.logarithm_domain",
            Self::NonFiniteEnclosure { .. } => "preference_certified.enclosure_non_finite",
            Self::InvalidAtoms => "preference_certified.atoms_invalid",
            Self::InvalidProbabilities => "preference_certified.atom_probabilities_invalid",
            Self::ProbabilitiesRequired => "preference_certified.atom_probabilities_required",
            Self::MixtureCriterionUnsupported { .. } => {
                "preference_certified.mixture_criterion_unsupported"
            }
        }
    }
}

// ---------------------------------------------------------------------------------------------
// Interval arithmetic
// ---------------------------------------------------------------------------------------------

/// Why an enclosure could not be formed.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum EncloseFailure {
    /// A logarithm's argument reaches zero or below.
    LnDomain,
    /// A bound is not finite (or an index is unknown, which a validated contract excludes).
    NonFinite,
}

/// A closed interval `[lo, hi]` of reals with finite ends.
#[derive(Clone, Copy, Debug, PartialEq)]
struct Interval {
    lo: f64,
    hi: f64,
}

impl Interval {
    /// The degenerate interval of an exactly known value.
    const fn point(value: f64) -> Self {
        Self { lo: value, hi: value }
    }

    /// Finite ends only.
    fn checked(lo: f64, hi: f64) -> Result<Self, EncloseFailure> {
        if lo.is_finite() && hi.is_finite() && lo <= hi {
            Ok(Self { lo, hi })
        } else {
            Err(EncloseFailure::NonFinite)
        }
    }

    /// Ends computed with round-to-nearest, widened outward by one representable step.
    fn rounded(lo: f64, hi: f64) -> Result<Self, EncloseFailure> {
        Self::checked(lo.next_down(), hi.next_up())
    }

    fn plus(self, other: Self) -> Result<Self, EncloseFailure> {
        Self::rounded(self.lo + other.lo, self.hi + other.hi)
    }

    fn minus(self, other: Self) -> Result<Self, EncloseFailure> {
        Self::rounded(self.lo - other.hi, self.hi - other.lo)
    }

    const fn negated(self) -> Self {
        Self { lo: -self.hi, hi: -self.lo }
    }

    fn times(self, other: Self) -> Result<Self, EncloseFailure> {
        let products =
            [self.lo * other.lo, self.lo * other.hi, self.hi * other.lo, self.hi * other.hi];
        if products.iter().any(|value| value.is_nan()) {
            return Err(EncloseFailure::NonFinite);
        }
        let lo = products.iter().copied().fold(f64::INFINITY, f64::min);
        let hi = products.iter().copied().fold(f64::NEG_INFINITY, f64::max);
        Self::rounded(lo, hi)
    }

    fn scale(self, factor: f64) -> Result<Self, EncloseFailure> {
        self.times(Self::point(factor))
    }

    /// Pointwise minimum; exact, since `min` is monotone in both arguments.
    const fn min(self, other: Self) -> Self {
        Self { lo: self.lo.min(other.lo), hi: self.hi.min(other.hi) }
    }

    /// Pointwise maximum; exact, since `max` is monotone in both arguments.
    const fn max(self, other: Self) -> Self {
        Self { lo: self.lo.max(other.lo), hi: self.hi.max(other.hi) }
    }

    /// `exp` is increasing; the library function is not correctly rounded, so each end moves
    /// two steps. The lower end is never below zero.
    fn exp(self) -> Result<Self, EncloseFailure> {
        let lo = self.lo.exp().next_down().next_down().max(0.0);
        let hi = self.hi.exp().next_up().next_up();
        Self::checked(lo, hi)
    }

    /// `ln` is increasing on `(0, inf)`; an argument reaching zero or below is outside the
    /// domain.
    fn ln(self) -> Result<Self, EncloseFailure> {
        if self.lo <= 0.0 {
            return Err(EncloseFailure::LnDomain);
        }
        Self::checked(self.lo.ln().next_down().next_down(), self.hi.ln().next_up().next_up())
    }

    /// Exactly zero at both ends.
    fn is_zero(self) -> bool {
        self.lo >= 0.0 && self.hi <= 0.0
    }
}

/// Enclose `expr` with the attributes at the given values and every parameter ranging over its
/// interval.
fn enclose(
    expr: &PreferenceExpr,
    attributes: &[f64],
    parameters: &[Interval],
) -> Result<Interval, EncloseFailure> {
    use PreferenceExpr as P;
    let go = |inner: &PreferenceExpr| enclose(inner, attributes, parameters);
    let value = match expr {
        P::Const(value) => Interval::point(*value),
        P::Attribute(k) => Interval::point(*attributes.get(*k).ok_or(EncloseFailure::NonFinite)?),
        P::Parameter(j) => *parameters.get(*j).ok_or(EncloseFailure::NonFinite)?,
        P::Neg(a) => go(a)?.negated(),
        P::Add(a, b) => go(a)?.plus(go(b)?)?,
        P::Sub(a, b) => go(a)?.minus(go(b)?)?,
        P::Mul(a, b) => go(a)?.times(go(b)?)?,
        P::Min(a, b) => go(a)?.min(go(b)?),
        P::Max(a, b) => go(a)?.max(go(b)?),
        P::Exp(a) => go(a)?.exp()?,
        P::Ln(a) => go(a)?.ln()?,
    };
    Interval::checked(value.lo, value.hi)
}

/// Enclosure of `sum w_i x_i` over terms of positive weight, widened by `2 (n + 1) eps` times
/// the sum of magnitudes so that any floating summation order is also contained.
fn weighted_sum(
    terms: impl IntoIterator<Item = (f64, Interval)>,
) -> Result<Interval, EncloseFailure> {
    let mut total = Interval::point(0.0);
    let mut magnitude = 0.0_f64;
    let mut count = 0.0_f64;
    for (weight, value) in terms {
        if weight > 0.0 {
            total = total.plus(value.scale(weight)?)?;
            magnitude += weight * value.lo.abs().max(value.hi.abs());
            count += 1.0;
        }
    }
    let slack = 2.0 * (count + 1.0) * f64::EPSILON * magnitude;
    Interval::rounded(total.lo - slack, total.hi + slack)
}

/// `constant + sum_j parameters_j * coefficients_j`, skipping exactly-zero coefficients.
fn affine(
    constant: Interval,
    coefficients: &[Interval],
    parameters: &[Interval],
) -> Result<Interval, EncloseFailure> {
    let mut total = constant;
    for (coefficient, parameter) in coefficients.iter().zip(parameters) {
        if !coefficient.is_zero() {
            total = total.plus(parameter.times(*coefficient)?)?;
        }
    }
    Ok(total)
}

/// Left inverse `inf { x : F(x) >= p }` of the weighted empirical CDF, with the decision
/// layer's mass tolerance.
fn left_inverse(values: &[f64], weights: &[f64], p: f64) -> Option<f64> {
    let mut order: Vec<usize> = (0..values.len()).collect();
    order.sort_by(|&a, &b| values[a].total_cmp(&values[b]));
    let mut cumulative = 0.0;
    for &index in &order {
        if weights[index] <= 0.0 {
            continue;
        }
        cumulative += weights[index];
        if cumulative >= p || (cumulative - p).abs() <= MASS_TOLERANCE * cumulative.max(p) {
            return Some(values[index]);
        }
    }
    order.iter().rev().find(|&&i| weights[i] > 0.0).map(|&index| values[index])
}

// ---------------------------------------------------------------------------------------------
// Criterion conventions
// ---------------------------------------------------------------------------------------------

/// `+1` when larger criterion values are better, `-1` when smaller are.
const fn orientation(criterion: DecisionCriterion) -> f64 {
    match criterion {
        DecisionCriterion::PosteriorExpectedLoss
        | DecisionCriterion::ExpectedRegret
        | DecisionCriterion::Regret => -1.0,
        _ => 1.0,
    }
}

/// Criteria ranked through expected-utility margins.
const fn is_expectation(criterion: DecisionCriterion) -> bool {
    matches!(
        criterion,
        DecisionCriterion::PosteriorExpectedUtility
            | DecisionCriterion::PosteriorExpectedLoss
            | DecisionCriterion::ExpectedRegret
    )
}

/// An interval in the criterion's own orientation from one in the larger-is-better
/// orientation.
fn oriented(sign: f64, score: (f64, f64)) -> (f64, f64) {
    if sign > 0.0 { score } else { (-score.1, -score.0) }
}

// ---------------------------------------------------------------------------------------------
// Sources
// ---------------------------------------------------------------------------------------------

/// Parameter-free parts of the linear form, enclosed per draw and in expectation.
struct LinearData {
    /// `fixed[a][i]`.
    fixed: Vec<Vec<Interval>>,
    /// `terms[a][i][j]`.
    terms: Vec<Vec<Vec<Interval>>>,
    /// Expected fixed part per action.
    mean_fixed: Vec<Interval>,
    /// Expected term per action and parameter.
    mean_terms: Vec<Vec<Interval>>,
}

impl LinearData {
    fn draw_value(
        &self,
        a: usize,
        i: usize,
        params: &[Interval],
    ) -> Result<Interval, EncloseFailure> {
        affine(self.fixed[a][i], &self.terms[a][i], params)
    }

    fn draw_difference(
        &self,
        b: usize,
        a: usize,
        i: usize,
        params: &[Interval],
    ) -> Result<Interval, EncloseFailure> {
        let constant = self.fixed[b][i].minus(self.fixed[a][i])?;
        let coefficients = differences(&self.terms[b][i], &self.terms[a][i])?;
        affine(constant, &coefficients, params)
    }

    fn mean(&self, a: usize, params: &[Interval]) -> Result<Interval, EncloseFailure> {
        affine(self.mean_fixed[a], &self.mean_terms[a], params)
    }

    fn mean_difference(
        &self,
        a: usize,
        b: usize,
        params: &[Interval],
    ) -> Result<Interval, EncloseFailure> {
        let constant = self.mean_fixed[a].minus(self.mean_fixed[b])?;
        let coefficients = differences(&self.mean_terms[a], &self.mean_terms[b])?;
        affine(constant, &coefficients, params)
    }
}

fn differences(left: &[Interval], right: &[Interval]) -> Result<Vec<Interval>, EncloseFailure> {
    left.iter()
        .zip(right)
        .map(
            |(l, r)| {
                if l.is_zero() && r.is_zero() { Ok(Interval::point(0.0)) } else { l.minus(*r) }
            },
        )
        .collect()
}

/// One source's draws, read once.
struct SourceData<'s> {
    artifact: &'s DistributionArtifact,
    /// `attributes[a][i][k]`: attribute `k` of action `a` on draw `i`.
    attributes: Vec<Vec<Vec<f64>>>,
    /// Normalized draw weights.
    weights: Vec<f64>,
    /// Hard-constraint exclusions per action.
    exclusions: Vec<Vec<ConstraintExclusion>>,
    linear: Option<LinearData>,
    receipt: SourceReceipt,
}

/// Read every action's attributes per draw, the constraint exclusions and, when the model has
/// a linear form, the enclosed parts.
fn prepare_source<'s>(
    contract: &PreferenceContract,
    artifact: &'s DistributionArtifact,
) -> Result<SourceData<'s>, PreferenceCertifiedError> {
    use PreferenceCertifiedError as E;
    let probe = contract.probe_contract().map_err(E::Contract)?;
    let draws = utility_draws(&probe.contract, artifact).map_err(E::Eval)?;
    let n_draws = draws.weights.len();
    let mut attributes: Vec<Vec<Vec<f64>>> = Vec::with_capacity(contract.actions.len());
    for (a, action) in contract.actions.iter().enumerate() {
        let mut columns = Vec::with_capacity(action.inputs.len());
        for (k, input) in action.inputs.iter().enumerate() {
            let column = match (input, probe.probe_index[a][k]) {
                (ActionInput::Fixed { value, .. }, _) => vec![*value; n_draws],
                (ActionInput::Quantity(_), Some(p)) => draws.utilities[p].clone(),
                (ActionInput::Quantity(_), None) => {
                    return Err(E::Eval(DecisionEvalError::QuantityNotFound {
                        action: action.id.clone(),
                        input: k,
                    }));
                }
            };
            columns.push(column);
        }
        let rows: Vec<Vec<f64>> =
            (0..n_draws).map(|i| columns.iter().map(|c| c[i]).collect()).collect();
        attributes.push(rows);
    }
    let identity = &artifact.metadata().identity;
    let receipt = SourceReceipt {
        provider_id: identity.provider_id.clone(),
        snapshot_id: identity.snapshot_id.clone(),
        rng_id: identity.rng_id.clone(),
        causal_contract_id: identity.causal_contract_id.clone(),
    };
    let exclusions = read_exclusions(contract, artifact, &attributes, &draws.weights)?;
    let linear = match contract.model.linear_form() {
        Some(form) => Some(linear_data(contract, &form, &attributes, &draws.weights)?),
        None => None,
    };
    Ok(SourceData { artifact, attributes, weights: draws.weights, exclusions, linear, receipt })
}

/// Enclose the parts of the linear form on every draw and in expectation.
fn linear_data(
    contract: &PreferenceContract,
    form: &LinearForm,
    attributes: &[Vec<Vec<f64>>],
    weights: &[f64],
) -> Result<LinearData, PreferenceCertifiedError> {
    let bounds = contract.model.preferences.bounds();
    let lower: Vec<f64> = bounds.iter().map(|b| b.0).collect();
    let upper: Vec<f64> = bounds.iter().map(|b| b.1).collect();
    let mut data = LinearData {
        fixed: Vec::new(),
        terms: Vec::new(),
        mean_fixed: Vec::new(),
        mean_terms: Vec::new(),
    };
    for (action, rows) in contract.actions.iter().zip(attributes) {
        let fail = |failure| box_refusal(failure, &action.id, &lower, &upper);
        let mut fixed = Vec::with_capacity(rows.len());
        let mut terms = Vec::with_capacity(rows.len());
        for row in rows {
            fixed.push(enclose(&form.fixed, row, &[]).map_err(fail)?);
            terms.push(
                form.terms
                    .iter()
                    .map(|term| enclose(term, row, &[]))
                    .collect::<Result<Vec<_>, _>>()
                    .map_err(fail)?,
            );
        }
        let mean_fixed =
            weighted_sum(weights.iter().copied().zip(fixed.iter().copied())).map_err(fail)?;
        let mean_terms = (0..form.terms.len())
            .map(|j| weighted_sum(weights.iter().copied().zip(terms.iter().map(|t| t[j]))))
            .collect::<Result<Vec<_>, _>>()
            .map_err(fail)?;
        data.fixed.push(fixed);
        data.terms.push(terms);
        data.mean_fixed.push(mean_fixed);
        data.mean_terms.push(mean_terms);
    }
    Ok(data)
}

fn box_refusal(
    failure: EncloseFailure,
    subject: &str,
    lower: &[f64],
    upper: &[f64],
) -> PreferenceCertifiedError {
    match failure {
        EncloseFailure::LnDomain => PreferenceCertifiedError::LogarithmDomain {
            action: subject.to_owned(),
            lower: lower.to_vec(),
            upper: upper.to_vec(),
        },
        EncloseFailure::NonFinite => PreferenceCertifiedError::NonFiniteEnclosure {
            subject: subject.to_owned(),
            lower: lower.to_vec(),
            upper: upper.to_vec(),
        },
    }
}

/// Exclusions per action. Constraints never read parameters, so the ordinary contract bound at
/// the declared midpoint gives them when its value can be lowered; otherwise each constraint is
/// evaluated on the drawn attributes.
fn read_exclusions(
    contract: &PreferenceContract,
    artifact: &DistributionArtifact,
    attributes: &[Vec<Vec<f64>>],
    weights: &[f64],
) -> Result<Vec<Vec<ConstraintExclusion>>, PreferenceCertifiedError> {
    use PreferenceCertifiedError as E;
    if contract.constraints.is_empty() {
        return Ok(vec![Vec::new(); contract.actions.len()]);
    }
    match contract.bind(&contract.model.preferences.midpoint()) {
        Ok(mut bound) => {
            bound.criterion = DecisionCriterion::PosteriorExpectedUtility;
            let result = evaluate_contract(&bound, artifact).map_err(E::Eval)?;
            Ok(result.actions.into_iter().map(|outcome| outcome.exclusions).collect())
        }
        Err(PreferenceContractError::NotLowerable { .. }) => {
            own_exclusions(contract, attributes, weights)
        }
        Err(other) => Err(E::Contract(other)),
    }
}

/// Constraint exclusions evaluated on the drawn attributes, named as the bound contract would
/// name them: a constraint whose rewrite differs between the actions it applies to becomes
/// `"{constraint}@{action}"`.
fn own_exclusions(
    contract: &PreferenceContract,
    attributes: &[Vec<Vec<f64>>],
    weights: &[f64],
) -> Result<Vec<Vec<ConstraintExclusion>>, PreferenceCertifiedError> {
    let mut exclusions = vec![Vec::new(); contract.actions.len()];
    for constraint in &contract.constraints {
        let applies: Vec<usize> = (0..contract.actions.len())
            .filter(|a| {
                constraint.applies_to.is_empty()
                    || constraint.applies_to.contains(&contract.actions[*a].id)
            })
            .collect();
        let shared = applies.windows(2).all(|pair| {
            same_rewrite(constraint, &contract.actions[pair[0]], &contract.actions[pair[1]])
        });
        for a in applies {
            let action = &contract.actions[a];
            let mut held = 0.0;
            let mut positive_violation = false;
            for (row, weight) in attributes[a].iter().zip(weights) {
                let value = constraint.expr.evaluate(row).map_err(|error| {
                    PreferenceCertifiedError::Eval(DecisionEvalError::Contract(error))
                })?;
                if !value.is_finite() {
                    return Err(PreferenceCertifiedError::Eval(
                        DecisionEvalError::NonFiniteUtility { action: action.id.clone() },
                    ));
                }
                if value <= constraint.bound {
                    held += weight;
                } else if *weight > 0.0 {
                    positive_violation = true;
                }
            }
            if constraint_fails(constraint.min_probability, held, positive_violation) {
                exclusions[a].push(ConstraintExclusion {
                    constraint_id: if shared {
                        constraint.id.clone()
                    } else {
                        format!("{}@{}", constraint.id, action.id)
                    },
                    probability: held,
                    required: constraint.min_probability,
                });
            }
        }
    }
    Ok(exclusions)
}

/// What attribute `k` of an action becomes in a rewritten constraint: its quantity input
/// number, or its fixed value.
enum Rewritten {
    Input(usize),
    Fixed(f64),
}

fn rewrite_of(action: &PreferenceAction, k: usize) -> Option<Rewritten> {
    match action.inputs.get(k)? {
        ActionInput::Fixed { value, .. } => Some(Rewritten::Fixed(*value)),
        ActionInput::Quantity(_) => Some(Rewritten::Input(
            action.inputs[..k].iter().filter(|i| matches!(i, ActionInput::Quantity(_))).count(),
        )),
    }
}

/// Whether the constraint rewrites to the same expression for both actions. Equal constants
/// give equal rewritten expressions, `0.0` and `-0.0` included.
#[allow(clippy::float_cmp, reason = "the rewrite compares constants by value")]
fn same_rewrite(
    constraint: &HardConstraint,
    first: &PreferenceAction,
    second: &PreferenceAction,
) -> bool {
    constraint.expr.inputs_used().into_iter().all(|k| {
        match (rewrite_of(first, k), rewrite_of(second, k)) {
            (Some(Rewritten::Input(x)), Some(Rewritten::Input(y))) => x == y,
            (Some(Rewritten::Fixed(x)), Some(Rewritten::Fixed(y))) => x == y,
            _ => false,
        }
    })
}

// ---------------------------------------------------------------------------------------------
// Box evaluation
// ---------------------------------------------------------------------------------------------

/// How per-structure enclosures combine.
enum Combination {
    /// One source.
    Single,
    /// The worst over structures.
    Worst,
    /// The probability-weighted sum over structures, not renormalized.
    Bayes(Vec<f64>),
}

/// Score enclosures (larger is better) and pairwise margins on one box.
#[derive(Clone, Debug)]
struct BoxEval {
    /// `scores[a]`; `None` for an excluded action.
    scores: Vec<Option<Interval>>,
    /// `margins[a][b]` encloses `score_a - score_b`; `None` unless both are admissible and
    /// distinct.
    margins: Vec<Vec<Option<Interval>>>,
}

/// One examined box that was kept.
struct Examined {
    lower: Vec<f64>,
    upper: Vec<f64>,
    boundary: bool,
    eval: BoxEval,
    best: Option<usize>,
    candidates: Vec<usize>,
    volume: f64,
}

/// A box waiting to be split. Larger volume first, then the smaller lower corner.
struct Queued {
    volume: f64,
    lower: Vec<f64>,
    slot: usize,
}

fn lexicographic(left: &[f64], right: &[f64]) -> Ordering {
    left.iter()
        .zip(right)
        .map(|(l, r)| l.total_cmp(r))
        .find(|order| order.is_ne())
        .unwrap_or(Ordering::Equal)
}

impl Ord for Queued {
    fn cmp(&self, other: &Self) -> Ordering {
        self.volume
            .total_cmp(&other.volume)
            .then_with(|| lexicographic(&other.lower, &self.lower))
            .then_with(|| other.slot.cmp(&self.slot))
    }
}

impl PartialOrd for Queued {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

impl PartialEq for Queued {
    fn eq(&self, other: &Self) -> bool {
        self.cmp(other).is_eq()
    }
}

impl Eq for Queued {}

/// Where a box sits relative to the relations.
enum Fit {
    Inside,
    Boundary,
    Outside,
}

struct Engine<'a> {
    contract: &'a PreferenceContract,
    sources: &'a [SourceData<'a>],
    combination: Combination,
    admissible: Vec<bool>,
    options: CertifiedOptions,
    /// Declared `(lower, upper)` per parameter.
    declared: Vec<(f64, f64)>,
}

impl Engine<'_> {
    fn criterion(&self) -> DecisionCriterion {
        self.contract.criterion
    }

    fn admissible_indices(&self) -> Vec<usize> {
        (0..self.admissible.len()).filter(|a| self.admissible[*a]).collect()
    }

    fn relation_fit(
        &self,
        params: &[Interval],
        lower: &[f64],
        upper: &[f64],
    ) -> Result<Fit, PreferenceCertifiedError> {
        let mut boundary = false;
        for relation in &self.contract.model.preferences.relations {
            let lhs = relation
                .terms
                .iter()
                .try_fold(Interval::point(0.0), |sum, (j, coefficient)| {
                    let parameter = params.get(*j).copied().ok_or(EncloseFailure::NonFinite)?;
                    sum.plus(parameter.scale(*coefficient)?)
                })
                .map_err(|failure| box_refusal(failure, &relation.id, lower, upper))?;
            let bound = relation.bound;
            let (satisfied, violated) = match relation.sense {
                RelationSense::AtLeast => (lhs.lo >= bound, lhs.hi < bound),
                RelationSense::AtMost => (lhs.hi <= bound, lhs.lo > bound),
                RelationSense::Equal => {
                    (lhs.lo >= bound && lhs.hi <= bound, lhs.hi < bound || lhs.lo > bound)
                }
            };
            if violated {
                return Ok(Fit::Outside);
            }
            boundary |= !satisfied;
        }
        Ok(if boundary { Fit::Boundary } else { Fit::Inside })
    }

    /// Per-draw value enclosures of action `a` on the box.
    fn draw_utilities(
        &self,
        source: &SourceData<'_>,
        a: usize,
        params: &[Interval],
    ) -> Result<Vec<Interval>, EncloseFailure> {
        (0..source.weights.len())
            .map(|i| match &source.linear {
                Some(linear) => linear.draw_value(a, i, params),
                None => enclose(&self.contract.model.value, &source.attributes[a][i], params),
            })
            .collect()
    }

    /// Enclosure of `U_b(i) - U_a(i)`.
    fn draw_difference(
        source: &SourceData<'_>,
        utilities: &[Option<Vec<Interval>>],
        b: usize,
        a: usize,
        i: usize,
        params: &[Interval],
    ) -> Result<Interval, EncloseFailure> {
        match &source.linear {
            Some(linear) => linear.draw_difference(b, a, i, params),
            None => draws_of(utilities, b)?[i].minus(draws_of(utilities, a)?[i]),
        }
    }

    /// Per-draw regret `max(0, max_b U_b - U_a)` of action `a` over admissible rivals.
    fn regrets(
        &self,
        source: &SourceData<'_>,
        utilities: &[Option<Vec<Interval>>],
        a: usize,
        params: &[Interval],
    ) -> Result<Vec<Interval>, EncloseFailure> {
        let rivals: Vec<usize> =
            self.admissible_indices().into_iter().filter(|b| *b != a).collect();
        (0..source.weights.len())
            .map(|i| {
                rivals.iter().try_fold(Interval::point(0.0), |regret, b| {
                    Ok(regret.max(Self::draw_difference(source, utilities, *b, a, i, params)?))
                })
            })
            .collect()
    }

    /// Score enclosure (larger is better) of action `a` on one source.
    fn score(
        &self,
        source: &SourceData<'_>,
        utilities: &[Option<Vec<Interval>>],
        a: usize,
        params: &[Interval],
    ) -> Result<Interval, EncloseFailure> {
        let weights = &source.weights;
        match self.criterion() {
            DecisionCriterion::PosteriorExpectedUtility
            | DecisionCriterion::PosteriorExpectedLoss => {
                let mean = match &source.linear {
                    Some(linear) => linear.mean(a, params)?,
                    None => weighted_sum(
                        weights.iter().copied().zip(draws_of(utilities, a)?.iter().copied()),
                    )?,
                };
                Ok(if orientation(self.criterion()) > 0.0 { mean } else { mean.negated() })
            }
            DecisionCriterion::ThresholdProbability { threshold } => {
                let draws = draws_of(utilities, a)?;
                let mass = |reaches: &dyn Fn(&Interval) -> bool| {
                    weighted_sum(
                        weights
                            .iter()
                            .zip(draws)
                            .filter(|(_, value)| reaches(value))
                            .map(|(w, _)| (*w, Interval::point(1.0))),
                    )
                };
                let low = mass(&|value| value.lo >= threshold)?;
                let high = mass(&|value| value.hi >= threshold)?;
                Interval::checked(low.lo, high.hi)
            }
            DecisionCriterion::Quantile { p } => {
                let draws = draws_of(utilities, a)?;
                let lows: Vec<f64> = draws.iter().map(|value| value.lo).collect();
                let highs: Vec<f64> = draws.iter().map(|value| value.hi).collect();
                let low = left_inverse(&lows, weights, p).ok_or(EncloseFailure::NonFinite)?;
                let high = left_inverse(&highs, weights, p).ok_or(EncloseFailure::NonFinite)?;
                Interval::checked(low, high)
            }
            DecisionCriterion::Regret => {
                let regrets = self.regrets(source, utilities, a, params)?;
                let largest = regrets
                    .iter()
                    .zip(weights)
                    .filter(|(_, w)| **w > 0.0)
                    .fold(Interval::point(0.0), |top, (regret, _)| top.max(*regret));
                Ok(largest.negated())
            }
            DecisionCriterion::ExpectedRegret => {
                let regrets = self.regrets(source, utilities, a, params)?;
                Ok(weighted_sum(weights.iter().copied().zip(regrets))?.negated())
            }
            DecisionCriterion::MinimaxOverIdentifiedSet
            | DecisionCriterion::MaximinOverStructures => Err(EncloseFailure::NonFinite),
        }
    }

    /// Margin enclosure `score_a - score_b` on one source.
    fn margin(
        &self,
        source: &SourceData<'_>,
        utilities: &[Option<Vec<Interval>>],
        scores: &[Option<Interval>],
        a: usize,
        b: usize,
        params: &[Interval],
    ) -> Result<Interval, EncloseFailure> {
        if is_expectation(self.criterion()) {
            let difference = if let Some(linear) = &source.linear {
                linear.mean_difference(a, b, params)?
            } else {
                let left = draws_of(utilities, a)?;
                let right = draws_of(utilities, b)?;
                let per_draw = left
                    .iter()
                    .zip(right)
                    .map(|(l, r)| l.minus(*r))
                    .collect::<Result<Vec<_>, _>>()?;
                weighted_sum(source.weights.iter().copied().zip(per_draw))?
            };
            let loss = self.criterion() == DecisionCriterion::PosteriorExpectedLoss;
            Ok(if loss { difference.negated() } else { difference })
        } else {
            let left = scores[a].ok_or(EncloseFailure::NonFinite)?;
            let right = scores[b].ok_or(EncloseFailure::NonFinite)?;
            left.minus(right)
        }
    }

    fn source_box(
        &self,
        source: &SourceData<'_>,
        params: &[Interval],
        lower: &[f64],
        upper: &[f64],
    ) -> Result<BoxEval, PreferenceCertifiedError> {
        let n = self.contract.actions.len();
        let admissible = self.admissible_indices();
        let fail =
            |failure, a: usize| box_refusal(failure, &self.contract.actions[a].id, lower, upper);
        let means_only = source.linear.is_some()
            && matches!(
                self.criterion(),
                DecisionCriterion::PosteriorExpectedUtility
                    | DecisionCriterion::PosteriorExpectedLoss
            );
        let mut utilities: Vec<Option<Vec<Interval>>> = vec![None; n];
        if !means_only {
            for &a in &admissible {
                utilities[a] =
                    Some(self.draw_utilities(source, a, params).map_err(|f| fail(f, a))?);
            }
        }
        let mut scores = vec![None; n];
        for &a in &admissible {
            scores[a] = Some(self.score(source, &utilities, a, params).map_err(|f| fail(f, a))?);
        }
        let mut margins = vec![vec![None; n]; n];
        for &a in &admissible {
            for &b in admissible.iter().filter(|b| **b != a) {
                margins[a][b] = Some(
                    self.margin(source, &utilities, &scores, a, b, params)
                        .map_err(|f| fail(f, a))?,
                );
            }
        }
        Ok(BoxEval { scores, margins })
    }

    fn evaluate_box(
        &self,
        params: &[Interval],
        lower: &[f64],
        upper: &[f64],
    ) -> Result<BoxEval, PreferenceCertifiedError> {
        let mut per_source = self
            .sources
            .iter()
            .map(|source| self.source_box(source, params, lower, upper))
            .collect::<Result<Vec<_>, _>>()?;
        let n = self.contract.actions.len();
        let fail =
            |failure, a: usize| box_refusal(failure, &self.contract.actions[a].id, lower, upper);
        match &self.combination {
            // A run always has at least one source.
            Combination::Single => per_source.pop().ok_or(PreferenceCertifiedError::InvalidAtoms),
            Combination::Worst => {
                let scores: Vec<Option<Interval>> = (0..n)
                    .map(|a| per_source.iter().filter_map(|e| e.scores[a]).reduce(Interval::min))
                    .collect();
                let mut margins = vec![vec![None; n]; n];
                for (a, row) in margins.iter_mut().enumerate() {
                    for (b, slot) in row.iter_mut().enumerate() {
                        if let (Some(left), Some(right)) = (scores[a], scores[b])
                            && a != b
                        {
                            *slot = Some(left.minus(right).map_err(|f| fail(f, a))?);
                        }
                    }
                }
                Ok(BoxEval { scores, margins })
            }
            Combination::Bayes(probabilities) => {
                let mix = |pick: &dyn Fn(&BoxEval) -> Option<Interval>, a: usize| {
                    let parts: Option<Vec<Interval>> = per_source.iter().map(pick).collect();
                    parts
                        .map(|parts| weighted_sum(probabilities.iter().copied().zip(parts)))
                        .transpose()
                        .map_err(|f| fail(f, a))
                };
                let mut scores = Vec::with_capacity(n);
                for a in 0..n {
                    scores.push(mix(&|e: &BoxEval| e.scores[a], a)?);
                }
                let mut margins = vec![vec![None; n]; n];
                for (a, row) in margins.iter_mut().enumerate() {
                    for (b, slot) in row.iter_mut().enumerate() {
                        *slot = mix(&|e: &BoxEval| e.margins[a][b], a)?;
                    }
                }
                Ok(BoxEval { scores, margins })
            }
        }
    }

    /// The action certified on the box, if any, and the actions not proven worse.
    fn classify(&self, eval: &BoxEval) -> (Option<usize>, Vec<usize>) {
        let admissible = self.admissible_indices();
        let tie = self.options.tie_tolerance;
        let beats = |a: usize, b: usize| eval.margins[a][b].is_some_and(|m| m.lo > tie);
        let best =
            admissible.iter().copied().find(|&a| admissible.iter().all(|&b| b == a || beats(a, b)));
        let candidates = match best {
            Some(a) => vec![a],
            None => admissible
                .iter()
                .copied()
                .filter(|&a| !admissible.iter().any(|&b| b != a && beats(b, a)))
                .collect(),
        };
        (best, candidates)
    }

    /// Volume relative to the declared box and the largest relative width, over parameters
    /// of positive declared width.
    fn measure_box(&self, lower: &[f64], upper: &[f64]) -> (f64, f64) {
        let mut volume = 1.0;
        let mut widest = 0.0_f64;
        for ((l, u), (dl, du)) in lower.iter().zip(upper).zip(&self.declared) {
            let range = du - dl;
            if range > 0.0 {
                let fraction = (u - l) / range;
                volume *= fraction;
                widest = widest.max(fraction);
            }
        }
        (volume, widest)
    }

    fn examine(
        &self,
        lower: Vec<f64>,
        upper: Vec<f64>,
    ) -> Result<Option<Examined>, PreferenceCertifiedError> {
        let params: Vec<Interval> =
            lower.iter().zip(&upper).map(|(l, u)| Interval { lo: *l, hi: *u }).collect();
        let boundary = match self.relation_fit(&params, &lower, &upper)? {
            Fit::Outside => return Ok(None),
            Fit::Inside => false,
            Fit::Boundary => true,
        };
        let eval = self.evaluate_box(&params, &lower, &upper)?;
        let (best, candidates) = self.classify(&eval);
        let (volume, _) = self.measure_box(&lower, &upper);
        Ok(Some(Examined { lower, upper, boundary, eval, best, candidates, volume }))
    }

    /// The two halves of a box, or `None` when it is not to be split.
    fn split(&self, item: &Examined) -> Option<[(Vec<f64>, Vec<f64>); 2]> {
        let (_, widest) = self.measure_box(&item.lower, &item.upper);
        if widest <= self.options.min_width_fraction {
            return None;
        }
        let mut chosen: Option<(usize, f64)> = None;
        for (j, ((l, u), (dl, du))) in
            item.lower.iter().zip(&item.upper).zip(&self.declared).enumerate()
        {
            let range = du - dl;
            if range > 0.0 {
                let fraction = (u - l) / range;
                if chosen.is_none_or(|(_, best)| fraction > best) {
                    chosen = Some((j, fraction));
                }
            }
        }
        let (j, _) = chosen?;
        let (l, u) = (item.lower[j], item.upper[j]);
        let mid = 0.5 * l + 0.5 * u;
        let interior = l < mid && mid < u;
        if !interior {
            return None;
        }
        let mut left_upper = item.upper.clone();
        left_upper[j] = mid;
        let mut right_lower = item.lower.clone();
        right_lower[j] = mid;
        Some([(item.lower.clone(), left_upper), (right_lower, item.upper.clone())])
    }

    fn place(
        &self,
        item: Examined,
        kept: &mut Vec<Examined>,
        slots: &mut Vec<Option<Examined>>,
        queue: &mut BinaryHeap<Queued>,
    ) {
        if item.best.is_some() || self.split(&item).is_none() {
            kept.push(item);
        } else {
            queue.push(Queued {
                volume: item.volume,
                lower: item.lower.clone(),
                slot: slots.len(),
            });
            slots.push(Some(item));
        }
    }

    /// Cover the declared box; returns the kept boxes and the number examined.
    fn subdivide(&self) -> Result<(Vec<Examined>, usize), PreferenceCertifiedError> {
        let lower: Vec<f64> = self.declared.iter().map(|b| b.0).collect();
        let upper: Vec<f64> = self.declared.iter().map(|b| b.1).collect();
        let mut kept = Vec::new();
        let mut slots = Vec::new();
        let mut queue = BinaryHeap::new();
        let mut examined = 1;
        if let Some(root) = self.examine(lower, upper)? {
            self.place(root, &mut kept, &mut slots, &mut queue);
        }
        while let Some(entry) = queue.pop() {
            let Some(item) = slots[entry.slot].take() else {
                continue;
            };
            let halves =
                if examined + 2 > self.options.max_boxes { None } else { self.split(&item) };
            let Some(halves) = halves else {
                kept.push(item);
                continue;
            };
            examined += 2;
            for (child_lower, child_upper) in halves {
                if let Some(child) = self.examine(child_lower, child_upper)? {
                    self.place(child, &mut kept, &mut slots, &mut queue);
                }
            }
        }
        if kept.is_empty() {
            return Err(PreferenceCertifiedError::EmptyPreferenceSet);
        }
        Ok((kept, examined))
    }

    /// Exact score (larger is better) of every admissible action at one parameter point.
    fn point_scores(&self, point: &[f64]) -> Result<Vec<Option<f64>>, PreferenceCertifiedError> {
        let per_source = self
            .sources
            .iter()
            .map(|source| self.source_point_scores(source, point))
            .collect::<Result<Vec<_>, _>>()?;
        let n = self.contract.actions.len();
        Ok((0..n)
            .map(|a| {
                let values: Option<Vec<f64>> = per_source.iter().map(|s| s[a]).collect();
                let values = values?;
                match &self.combination {
                    Combination::Single => values.first().copied(),
                    Combination::Worst => values.into_iter().reduce(f64::min),
                    Combination::Bayes(probabilities) => {
                        Some(probabilities.iter().zip(values).map(|(p, v)| p * v).sum())
                    }
                }
            })
            .collect())
    }

    /// Exact scores on one source: the bound ordinary contract evaluated by the decision layer
    /// when the value can be lowered and the source excludes the same actions as the combined
    /// analysis; otherwise the model evaluated on each draw with the same criterion rules.
    fn source_point_scores(
        &self,
        source: &SourceData<'_>,
        point: &[f64],
    ) -> Result<Vec<Option<f64>>, PreferenceCertifiedError> {
        use PreferenceCertifiedError as E;
        let sign = orientation(self.criterion());
        let same_admissible =
            source.exclusions.iter().map(Vec::is_empty).eq(self.admissible.iter().copied());
        if same_admissible {
            match self.contract.bind(point) {
                Ok(bound) => {
                    let result = evaluate_contract(&bound, source.artifact).map_err(E::Eval)?;
                    return Ok(result
                        .actions
                        .iter()
                        .zip(&self.admissible)
                        .map(|(outcome, admissible)| admissible.then_some(sign * outcome.value))
                        .collect());
                }
                Err(PreferenceContractError::NotLowerable { .. }) => {}
                Err(other) => return Err(E::Contract(other)),
            }
        }
        self.direct_point_scores(source, point)
    }

    fn direct_point_scores(
        &self,
        source: &SourceData<'_>,
        point: &[f64],
    ) -> Result<Vec<Option<f64>>, PreferenceCertifiedError> {
        let n = self.contract.actions.len();
        let admissible = self.admissible_indices();
        let weights = &source.weights;
        let mut utilities: Vec<Vec<f64>> = vec![Vec::new(); n];
        for &a in &admissible {
            utilities[a] = source.attributes[a]
                .iter()
                .map(|row| self.contract.model.value.evaluate(row, point))
                .collect::<Result<Vec<_>, _>>()
                .map_err(PreferenceCertifiedError::Model)?;
        }
        let best: Vec<f64> = (0..weights.len())
            .map(|i| admissible.iter().map(|a| utilities[*a][i]).fold(f64::NEG_INFINITY, f64::max))
            .collect();
        let mean = |values: &[f64]| -> f64 {
            values.iter().zip(weights).filter(|(_, w)| **w > 0.0).map(|(v, w)| v * w).sum()
        };
        let mut scores = vec![None; n];
        for &a in &admissible {
            let utility = &utilities[a];
            let regret: Vec<f64> = utility.iter().zip(&best).map(|(u, top)| top - u).collect();
            let score = match self.criterion() {
                DecisionCriterion::PosteriorExpectedUtility => mean(utility),
                DecisionCriterion::PosteriorExpectedLoss => -mean(utility),
                DecisionCriterion::ThresholdProbability { threshold } => {
                    let reached: Vec<f64> =
                        utility.iter().map(|u| f64::from(u8::from(*u >= threshold))).collect();
                    mean(&reached)
                }
                DecisionCriterion::Quantile { p } => left_inverse(utility, weights, p)
                    .ok_or(PreferenceCertifiedError::Model(PreferenceModelError::NonFiniteValue))?,
                DecisionCriterion::Regret => -regret
                    .iter()
                    .zip(weights)
                    .filter(|(_, w)| **w > 0.0)
                    .fold(0.0, |top: f64, (r, _)| top.max(*r)),
                DecisionCriterion::ExpectedRegret => -mean(&regret),
                DecisionCriterion::MinimaxOverIdentifiedSet
                | DecisionCriterion::MaximinOverStructures => {
                    return Err(PreferenceCertifiedError::StructureInputsRequired);
                }
            };
            if !score.is_finite() {
                return Err(PreferenceCertifiedError::Model(PreferenceModelError::NonFiniteValue));
            }
            scores[a] = Some(score);
        }
        Ok(scores)
    }
}

fn draws_of(utilities: &[Option<Vec<Interval>>], a: usize) -> Result<&[Interval], EncloseFailure> {
    utilities.get(a).and_then(Option::as_deref).ok_or(EncloseFailure::NonFinite)
}

// ---------------------------------------------------------------------------------------------
// Aggregation over kept boxes
// ---------------------------------------------------------------------------------------------

/// Fractions and enclosures over the kept boxes, scores in the larger-is-better orientation.
struct Tally {
    best_fraction: Vec<f64>,
    undecided_fraction: f64,
    value: Vec<Option<(f64, f64)>>,
    worst: Vec<Option<(f64, f64)>>,
    regret: Vec<Option<(f64, f64)>>,
}

/// Fractions and enclosures over the kept boxes.
///
/// A worst value over the set is at most the value at any point of the set, so the upper end
/// of a worst-value enclosure also uses the enclosure at the center of each box wholly inside
/// the relations; likewise a worst-case regret is at least the margin at such a center.
fn tally(engine: &Engine<'_>, kept: &[Examined]) -> Result<Tally, PreferenceCertifiedError> {
    let n = engine.contract.actions.len();
    let total: f64 = kept.iter().map(|item| item.volume).sum();
    let fraction = |volume: f64| if total > 0.0 { volume / total } else { 0.0 };
    let mut best_fraction = vec![0.0; n];
    let mut undecided = 0.0;
    for item in kept {
        match item.best {
            Some(a) => best_fraction[a] += item.volume,
            None => undecided += item.volume,
        }
    }
    let best_fraction = best_fraction.into_iter().map(fraction).collect();
    let mut centers = Vec::new();
    for item in kept.iter().filter(|item| !item.boundary) {
        let center: Vec<f64> =
            item.lower.iter().zip(&item.upper).map(|(l, u)| 0.5 * l + 0.5 * u).collect();
        let params: Vec<Interval> = center.iter().map(|c| Interval::point(*c)).collect();
        centers.push(engine.evaluate_box(&params, &center, &center)?);
    }
    let admissible = engine.admissible_indices();
    let mut value = vec![None; n];
    let mut worst = vec![None; n];
    let mut regret = vec![None; n];
    for &a in &admissible {
        let scores: Vec<Interval> = kept.iter().filter_map(|item| item.eval.scores[a]).collect();
        if scores.is_empty() {
            continue;
        }
        let lo = scores.iter().map(|s| s.lo).fold(f64::INFINITY, f64::min);
        let hi = scores.iter().map(|s| s.hi).fold(f64::NEG_INFINITY, f64::max);
        let inside_hi = kept
            .iter()
            .filter(|item| !item.boundary)
            .filter_map(|item| item.eval.scores[a])
            .chain(centers.iter().filter_map(|center| center.scores[a]))
            .map(|s| s.hi)
            .fold(f64::INFINITY, f64::min);
        value[a] = Some((lo, hi));
        worst[a] = Some((lo, if inside_hi.is_finite() { inside_hi } else { hi }));
        if is_expectation(engine.criterion()) {
            let rivals = || admissible.iter().copied().filter(move |b| *b != a);
            let upper = kept
                .iter()
                .flat_map(|item| rivals().filter_map(move |b| item.eval.margins[b][a]))
                .fold(0.0_f64, |top, m| top.max(m.hi));
            let lower = kept
                .iter()
                .filter(|item| !item.boundary)
                .map(|item| &item.eval)
                .chain(&centers)
                .flat_map(|eval| rivals().filter_map(move |b| eval.margins[b][a]))
                .fold(0.0_f64, |top, m| top.max(m.lo));
            regret[a] = Some((lower, upper));
        }
    }
    Ok(Tally { best_fraction, undecided_fraction: fraction(undecided), value, worst, regret })
}

/// The action whose enclosure (larger is better) lies above every other's by more than
/// `tie`, or the actions not proven below another.
fn dominant(enclosures: &[Option<(f64, f64)>], tie: f64) -> Result<usize, Vec<usize>> {
    let present: Vec<usize> = (0..enclosures.len()).filter(|a| enclosures[*a].is_some()).collect();
    let above = |a: usize, b: usize| match (enclosures[a], enclosures[b]) {
        (Some(x), Some(y)) => x.0 > y.1 + tie,
        _ => false,
    };
    if let Some(a) =
        present.iter().copied().find(|&a| present.iter().all(|&b| b == a || above(a, b)))
    {
        return Ok(a);
    }
    Err(present
        .iter()
        .copied()
        .filter(|&a| !present.iter().any(|&b| b != a && above(b, a)))
        .collect())
}

fn ids(contract: &PreferenceContract, indices: impl IntoIterator<Item = usize>) -> Vec<String> {
    indices.into_iter().map(|a| contract.actions[a].id.clone()).collect()
}

fn subdivision_verdict(engine: &Engine<'_>, kept: &[Examined], tally: &Tally) -> CertifiedVerdict {
    let contract = engine.contract;
    let tie = engine.options.tie_tolerance;
    let undetermined = |candidates: Vec<usize>| CertifiedVerdict::Undetermined {
        candidates: ids(contract, candidates),
        undecided_fraction: tally.undecided_fraction,
    };
    match contract.preference_policy {
        PreferencePolicy::ReportOnly | PreferencePolicy::Weighted => CertifiedVerdict::ReportOnly,
        PreferencePolicy::RequireInvariantBestAction => {
            if let Some(first) = kept.first().and_then(|item| item.best)
                && kept.iter().all(|item| item.best == Some(first))
            {
                return CertifiedVerdict::NecessarilyBest(contract.actions[first].id.clone());
            }
            let n = contract.actions.len();
            let mut certified = vec![false; n];
            let mut possible = vec![false; n];
            for item in kept {
                if let (Some(a), false) = (item.best, item.boundary) {
                    certified[a] = true;
                }
                for a in &item.candidates {
                    possible[*a] = true;
                }
            }
            if certified.iter().filter(|c| **c).count() >= 2 {
                CertifiedVerdict::PreferenceDependent {
                    certified_somewhere: ids(contract, (0..n).filter(|a| certified[*a])),
                }
            } else {
                undetermined((0..n).filter(|a| possible[*a]).collect())
            }
        }
        PreferencePolicy::Maximin => match dominant(&tally.worst, tie) {
            Ok(a) => CertifiedVerdict::WorstCaseChoice(contract.actions[a].id.clone()),
            Err(candidates) => undetermined(candidates),
        },
        PreferencePolicy::MinimaxRegret => {
            let negated: Vec<Option<(f64, f64)>> =
                tally.regret.iter().map(|r| r.map(|(lo, hi)| (-hi, -lo))).collect();
            match dominant(&negated, tie) {
                Ok(a) => CertifiedVerdict::MinimaxRegretChoice(contract.actions[a].id.clone()),
                Err(candidates) => undetermined(candidates),
            }
        }
    }
}

/// Measure-weighted values and shares, scores in the larger-is-better orientation.
struct MeasureSummary {
    value: Vec<Option<f64>>,
    share: Vec<Option<f64>>,
    verdict: CertifiedVerdict,
}

fn measure_weights(measure: &PreferenceMeasure) -> Vec<f64> {
    let raw = measure.weights.clone().unwrap_or_else(|| vec![1.0; measure.points.len()]);
    let total: f64 = raw.iter().sum();
    raw.into_iter().map(|w| w / total).collect()
}

fn weighted_policy(
    engine: &Engine<'_>,
    measure: &PreferenceMeasure,
) -> Result<MeasureSummary, PreferenceCertifiedError> {
    let n = engine.contract.actions.len();
    let tie = engine.options.tie_tolerance;
    let admissible = engine.admissible_indices();
    let mut value: Vec<Option<f64>> = (0..n).map(|a| engine.admissible[a].then_some(0.0)).collect();
    let mut share = value.clone();
    for (point, weight) in measure.points.iter().zip(measure_weights(measure)) {
        let scores = engine.point_scores(point)?;
        for &a in &admissible {
            if let (Some(total), Some(score)) = (value[a].as_mut(), scores[a]) {
                *total += weight * score;
            }
        }
        let leads = |a: usize| {
            admissible.iter().all(|&b| {
                b == a || matches!((scores[a], scores[b]), (Some(x), Some(y)) if x > y + tie)
            })
        };
        if let Some(leader) = admissible.iter().copied().find(|&a| leads(a))
            && let Some(slot) = share[leader].as_mut()
        {
            *slot += weight;
        }
    }
    let point_enclosures: Vec<Option<(f64, f64)>> =
        value.iter().map(|v| v.map(|x| (x, x))).collect();
    let verdict = match dominant(&point_enclosures, tie) {
        Ok(a) => CertifiedVerdict::MeasureChoice {
            action: engine.contract.actions[a].id.clone(),
            share: share[a].unwrap_or(0.0),
        },
        Err(candidates) => CertifiedVerdict::Undetermined {
            candidates: ids(engine.contract, candidates),
            undecided_fraction: 0.0,
        },
    };
    Ok(MeasureSummary { value, share, verdict })
}

// ---------------------------------------------------------------------------------------------
// Entry points
// ---------------------------------------------------------------------------------------------

/// What a run reports besides the per-action table.
struct Subdivided {
    regions: Vec<CertifiedRegion>,
    undecided_fraction: f64,
    boxes_examined: usize,
    verdict: CertifiedVerdict,
}

/// Subdivide, fill the per-action enclosures and fractions, and decide.
fn subdivided(
    engine: &Engine<'_>,
    actions: &mut [CertifiedActionOutcome],
) -> Result<Subdivided, PreferenceCertifiedError> {
    let contract = engine.contract;
    let sign = orientation(contract.criterion);
    let (mut kept, boxes_examined) = engine.subdivide()?;
    let tally = tally(engine, &kept)?;
    for (a, outcome) in actions.iter_mut().enumerate() {
        outcome.best_fraction = tally.best_fraction[a];
        outcome.value_bounds = tally.value[a].map(|v| oriented(sign, v));
        outcome.worst_case_bounds = tally.worst[a].map(|v| oriented(sign, v));
        outcome.worst_case_regret_bounds = tally.regret[a];
    }
    let verdict = subdivision_verdict(engine, &kept, &tally);
    kept.sort_by(|x, y| {
        lexicographic(&x.lower, &y.lower).then_with(|| lexicographic(&x.upper, &y.upper))
    });
    let regions = kept
        .into_iter()
        .map(|item| CertifiedRegion {
            status: match item.best {
                Some(a) => RegionStatus::Best(contract.actions[a].id.clone()),
                None => RegionStatus::Undecided { candidates: ids(contract, item.candidates) },
            },
            lower: item.lower,
            upper: item.upper,
            touches_relation_boundary: item.boundary,
        })
        .collect();
    Ok(Subdivided {
        regions,
        undecided_fraction: tally.undecided_fraction,
        boxes_examined,
        verdict,
    })
}

/// The validated contract, options and measure shared by every run.
struct Setup<'c> {
    contract: &'c PreferenceContract,
    identity: String,
    measure: Option<&'c PreferenceMeasure>,
    options: CertifiedOptions,
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
    set: &PreferenceSet,
) -> Result<(), PreferenceCertifiedError> {
    use PreferenceCertifiedError as E;
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
        if !set.contains(point, MEASURE_SET_TOLERANCE) {
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
        let measure = if contract.preference_policy == PreferencePolicy::Weighted {
            let measure = measure.ok_or(E::MeasureRequired)?;
            validate_measure(measure, &contract.model.preferences)?;
            Some(measure)
        } else {
            None
        };
        let identity = contract.identity().map_err(E::Contract)?;
        Ok(Self { contract, identity, measure, options })
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
        let admissible: Vec<bool> =
            (0..n).map(|a| sources.iter().all(|s| s.exclusions[a].is_empty())).collect();
        let mut actions: Vec<CertifiedActionOutcome> = contract
            .actions
            .iter()
            .enumerate()
            .map(|(a, action)| CertifiedActionOutcome {
                id: action.id.clone(),
                admissible: admissible[a],
                exclusions: sources.iter().flat_map(|s| s.exclusions[a].iter().cloned()).collect(),
                best_fraction: 0.0,
                value_bounds: None,
                worst_case_bounds: None,
                worst_case_regret_bounds: None,
                measure_value: None,
                measure_share: None,
            })
            .collect();
        let engine = Engine {
            contract,
            sources,
            combination,
            admissible,
            options: self.options,
            declared: contract.model.preferences.bounds(),
        };
        let sign = orientation(contract.criterion);
        let any_admissible = engine.admissible.iter().any(|a| *a);
        let outcome = match self.measure {
            _ if !any_admissible => Subdivided {
                regions: Vec::new(),
                undecided_fraction: 0.0,
                boxes_examined: 0,
                verdict: CertifiedVerdict::NoAdmissibleAction,
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
///   box wholly inside the relations; otherwise `Undetermined`.
/// - `Maximin`: compares the enclosures of each action's worst value over the set and chooses
///   only when one lies above every rival's by more than the tie tolerance.
/// - `MinimaxRegret`: expectation criteria only; compares worst-case regret enclosures built
///   from the pairwise margin enclosures.
/// - `Weighted`: needs a measure with every parameter resolved; each measure point is evaluated
///   exactly, through [`PreferenceContract::bind`] and [`evaluate_contract`] when the bound value
///   can be lowered and otherwise by evaluating the model on each draw with the same criterion
///   rules; no boxes are examined. A point counts toward an action's share when the action leads
///   every admissible rival there by more than the tie tolerance.
/// - `ReportOnly`: everything computed, verdict `ReportOnly`.
///
/// A measure is read only under the `Weighted` policy. Excluded actions take part in no
/// comparison.
///
/// # Errors
/// An invalid contract; a criterion that needs structure inputs; options out of range;
/// `MinimaxRegret` with a criterion that is not an expectation; under `Weighted`, a missing,
/// incomplete or invalid measure; a source that cannot answer the contract; relations that
/// exclude every box; a logarithm undefined or an enclosure non-finite on an examined box; a
/// model value that cannot be evaluated at a measure point.
pub fn evaluate_preference_certified(
    contract: &PreferenceContract,
    source: &DistributionArtifact,
    measure: Option<&PreferenceMeasure>,
    options: CertifiedOptions,
) -> Result<CertifiedPreferenceResult, PreferenceCertifiedError> {
    let setup = Setup::new(contract, measure, options)?;
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
    all_evaluated: bool,
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
    if !all_evaluated {
        return StructuralCertifiedVerdict::InsufficientScience(INVARIANCE_UNCHECKED.into());
    }
    let identical = verdicts.iter().all(|v| v.verdict == first.verdict);
    let same_choice = chosen(&first.verdict)
        .is_some_and(|action| verdicts.iter().all(|v| chosen(&v.verdict) == Some(action)));
    if identical || same_choice {
        StructuralCertifiedVerdict::Combined(first.verdict.clone())
    } else {
        StructuralCertifiedVerdict::StructuresDisagree(verdicts)
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
///   verdict must be the same or choose the same action; otherwise the structures' verdicts are
///   listed as a disagreement, or the claim is insufficient science.
/// - `Maximin`: one joint subdivision whose score on a box is the worst over structures of each
///   structure's enclosure; a choice needs every structure evaluated.
/// - `BayesOverStructures`: needs a probability on every structure; one joint subdivision whose
///   score and margins are the probability-weighted sums of the structures' enclosures, not
///   renormalized; a quantile or largest-regret criterion refuses; a choice needs no unresolved
///   mass.
/// - `ReportOnly`: per-structure results only.
///
/// An action excluded by a hard constraint in any evaluated structure is excluded overall.
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
    use PreferenceCertifiedError as E;
    let setup = Setup::new(contract, measure, options)?;
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
    let mass = |wanted: fn(&CertifiedAtomStatus) -> bool| -> Option<f64> {
        has_probabilities.then(|| {
            probability_sum(
                summaries.iter().filter(|s| wanted(&s.status)).filter_map(|s| s.probability),
            )
        })
    };
    let evaluated_mass = mass(|s| matches!(s, CertifiedAtomStatus::Evaluated(_)));
    let unidentified_mass = mass(|s| matches!(s, CertifiedAtomStatus::Unidentified));
    let unassigned = if has_probabilities {
        unassigned_probability(summaries.iter().filter_map(|s| s.probability))
    } else {
        0.0
    };
    let unevaluated_mass =
        mass(|s| matches!(s, CertifiedAtomStatus::Unevaluated(_))).map(|m| m + unassigned);
    let all_evaluated = sources.len() == atoms.len();
    let insufficient = |why: &str| StructuralCertifiedVerdict::InsufficientScience(why.into());
    let (combined, verdict) = match policy {
        StructuralPolicy::ReportOnly => (None, StructuralCertifiedVerdict::ReportOnly),
        StructuralPolicy::RequireInvariantBestAction => {
            (None, invariant_over_atoms(&summaries, all_evaluated))
        }
        _ if sources.is_empty() => (None, insufficient(NO_STRUCTURE_EVALUATED)),
        StructuralPolicy::Maximin => {
            let combined = setup.run(&sources, Combination::Worst)?;
            let verdict = if all_evaluated {
                StructuralCertifiedVerdict::Combined(combined.verdict.clone())
            } else {
                insufficient(WORST_CASE_NEEDS_EVERY_STRUCTURE)
            };
            (Some(combined), verdict)
        }
        StructuralPolicy::BayesOverStructures => {
            let combined = setup.run(&sources, Combination::Bayes(probabilities))?;
            let unresolved = unidentified_mass.unwrap_or(0.0) + unevaluated_mass.unwrap_or(0.0);
            let verdict = if unresolved > 0.0 {
                insufficient(UNRESOLVED_MASS_CAN_CHANGE_BAYES)
            } else if evaluated_mass.unwrap_or(0.0) <= 0.0 {
                insufficient(NO_EVALUATED_POSITIVE_PROBABILITY)
            } else {
                StructuralCertifiedVerdict::Combined(combined.verdict.clone())
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

#[cfg(test)]
mod tests {
    use super::*;

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
