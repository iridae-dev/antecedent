//! F17 `SensitivityArtifact` (`sensitivity_decision_v1`).
//!
//! A composition-ready record of an assumption-sensitivity surface: one declared
//! assumption coordinate on a finite grid, one or more surface quantities with an
//! assumption range (`lower`, `upper`) at every grid point, the actions whose
//! utilities read those quantities, the per-point coordinate support, the
//! uncertainty relationship and the provenance of the numbers.
//!
//! Three kinds of uncertainty stay in three distinct fields and are never merged:
//!
//! * the **assumption range** (`lower`, `upper` of every surface quantity) is what the
//!   quantity may be if the assumption holds anywhere inside the declared range; it is
//!   not a probability and not a confidence interval;
//! * an **identified bound** is a separate, optional bound from partial identification;
//! * a **sampling interval** is a separate, optional interval for estimation noise.
//!   Composition of a sampling interval with the assumption range is refused unless the
//!   method appears in [`LICENSED_SAMPLING_COMPOSITIONS`] (empty: none is licensed).
//!
//! Layout of the bounded sectioned container:
//!
//! * `sensitivity_decision.meta` (CBOR): version, feature, claim, coordinate, per-point
//!   support, quantity identities, action utilities, uncertainty relationship, provenance,
//!   the identity digests (premises and data digest kept separate) and the stored decision
//!   outcome;
//! * `sensitivity_decision.numbers` (little-endian `f64`): the grid, then per quantity the
//!   lower and upper values, then the sampling interval's endpoints when one is reported.
//!
//! A consumer trusts none of the stored results. It rebuilds the artifact from the stored
//! declarations and numbers, recomputes the identity digests and the decision outcome
//! (invariant action, assumption-dependent switch with its tipping coordinate, or no robust
//! action) and refuses unless every stored value matches, and, when the caller retained an
//! identity of its own, unless that identity matches too. A resealed change of a surface
//! value, action, coordinate, unit or uncertainty field is refused.
//!
//! Refusal details use the frozen record's namespace `sensitivity_decision_composition`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::cmp::Ordering;
use std::collections::{HashMap, HashSet};
use std::io::Cursor;

use antecedent_core::ScientificQuantity;
// Re-exported so a design-level consumer can adapt a 2.2 result without depending on
// the validation crate directly.
pub use antecedent_validate::{
    AxisTipping, JointFactor, JointFactorBound, JointMechanismSensitivityResult,
    JointSensitivityLimits, JointSensitivityRange, JointSensitivityReceipt,
    JointSensitivityUncertainty, TippingBracket, TippingStatus,
};
use serde::{Deserialize, Serialize};

use crate::container::{ArtifactManifest, EncodedArtifact, SectionBytes, section_descriptor};
use crate::convert::{from_cbor, to_cbor};
use crate::error::IoError;
use crate::quantity_wire::ScientificQuantityWire;
use crate::reader::ArtifactReader;
use crate::wire::{ArtifactKind, ProvenanceWire, SemanticVersion};

/// The artifact major version this reader writes and accepts.
pub const SENSITIVITY_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const SENSITIVITY_ARTIFACT_FEATURE: &str = "sensitivity_decision_v1";
/// The only claim a surface range carries.
pub const SENSITIVITY_INFERENCE_CLAIM: &str = "assumption_range";
/// Most bytes an artifact may occupy, enforced on export and on consumption.
pub const MAX_SENSITIVITY_ARTIFACT_BYTES: usize = 8 * 1024 * 1024;
/// Most grid points of the assumption coordinate.
pub const MAX_SENSITIVITY_GRID_POINTS: usize = 1024;
/// Most surface quantities.
pub const MAX_SENSITIVITY_QUANTITIES: usize = 64;
/// Most actions.
pub const MAX_SENSITIVITY_ACTIONS: usize = 64;
/// Most surface quantities whose assumption range is not a single point. Every grid point
/// is evaluated at the `2^r` vertices of the box these quantities span.
pub const MAX_SENSITIVITY_RANGED_QUANTITIES: usize = 3;
/// Deepest utility expression.
pub const MAX_SENSITIVITY_UTILITY_DEPTH: usize = 32;
/// Sampling-composition methods licensed for an assumption range. None is licensed, so a
/// sampling interval is always reported next to the range and never inside it.
pub const LICENSED_SAMPLING_COMPOSITIONS: &[&str] = &[];
/// What the surface range means.
pub const ASSUMPTION_RANGE_STATEMENT: &str = "assumption range over the declared coordinate set; not a probability, not a confidence interval and not a sampling interval";

const ARTIFACT_KIND: &str = "sensitivity_decision_v1";
const META_SECTION: &str = "sensitivity_decision.meta";
const NUMBERS_SECTION: &str = "sensitivity_decision.numbers";
const JOINT_SOURCE_KIND: &str = "joint_mechanism_sensitivity_2_2";

fn invalid_surface(text: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("invalid_argument"),
        message: format!("sensitivity_decision_composition.invalid_surface: {text}"),
    }
}

fn wrong_contract(text: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("decision_contract_unsatisfied"),
        message: format!("sensitivity_decision_composition.wrong_contract: {text}"),
    }
}

fn unsupported_coordinate(text: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("quantity_semantics_mismatch"),
        message: format!("sensitivity_decision_composition.unsupported_coordinate: {text}"),
    }
}

fn composition_not_licensed(text: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("cell_not_licensed"),
        message: format!("sensitivity_decision_composition.composition_not_licensed: {text}"),
    }
}

fn bounds_exceeded(text: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("route_not_supported"),
        message: format!("sensitivity_decision_composition.bounds_exceeded: {text}"),
    }
}

fn unsupported_adaptation(text: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("route_not_supported"),
        message: format!("sensitivity_decision_composition.unsupported_adaptation: {text}"),
    }
}

fn blank(value: &str) -> bool {
    value.trim().is_empty()
}

/// Exact float equality (ties of finite utilities).
fn same(left: f64, right: f64) -> bool {
    left.partial_cmp(&right) == Some(Ordering::Equal)
}

/// The declared assumption coordinate (for example `gamma`).
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssumptionCoordinate {
    /// Stable coordinate identity.
    pub id: String,
    /// Scale of the coordinate (`contamination_fraction`, `log_odds_ratio`, ...).
    pub scale: String,
    /// Units of the coordinate.
    pub units: String,
    /// Smallest value of the declared assumption range.
    pub minimum: f64,
    /// Largest value of the declared assumption range.
    pub maximum: f64,
}

/// Whether a grid point of the coordinate can be used.
#[derive(Clone, Copy, Debug, Eq, PartialEq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PointSupport {
    /// The surface was evaluated and may be consumed here.
    Supported,
    /// The coordinate is outside what the producer supports; consumption refuses.
    Unsupported,
    /// The producer did not evaluate this point (budget, cancellation); the decision
    /// over the range is then unresolved, never assumed.
    Unevaluated,
}

impl PointSupport {
    /// Stable wire name.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Supported => "supported",
            Self::Unsupported => "unsupported",
            Self::Unevaluated => "unevaluated",
        }
    }
}

/// A closed utility expression over named surface quantities.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum UtilityTerm {
    /// A constant.
    Const(f64),
    /// The surface quantity with this `variable_id`.
    Quantity(String),
    /// Negation.
    Neg(Box<UtilityTerm>),
    /// Sum.
    Add(Box<UtilityTerm>, Box<UtilityTerm>),
    /// Difference.
    Sub(Box<UtilityTerm>, Box<UtilityTerm>),
    /// Product.
    Mul(Box<UtilityTerm>, Box<UtilityTerm>),
    /// Pointwise minimum.
    Min(Box<UtilityTerm>, Box<UtilityTerm>),
    /// Pointwise maximum.
    Max(Box<UtilityTerm>, Box<UtilityTerm>),
}

impl UtilityTerm {
    /// The surface quantity `variable_id`.
    #[must_use]
    pub fn quantity(variable_id: &str) -> Self {
        Self::Quantity(variable_id.to_owned())
    }

    /// Build `left + right`.
    #[must_use]
    pub fn sum(left: Self, right: Self) -> Self {
        Self::Add(Box::new(left), Box::new(right))
    }

    /// Build `left - right`.
    #[must_use]
    pub fn difference(left: Self, right: Self) -> Self {
        Self::Sub(Box::new(left), Box::new(right))
    }

    /// Build `left * right`.
    #[must_use]
    pub fn product(left: Self, right: Self) -> Self {
        Self::Mul(Box::new(left), Box::new(right))
    }

    fn depth(&self) -> usize {
        match self {
            Self::Const(_) | Self::Quantity(_) => 1,
            Self::Neg(a) => 1 + a.depth(),
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b) => 1 + a.depth().max(b.depth()),
        }
    }

    fn all_finite(&self) -> bool {
        match self {
            Self::Const(value) => value.is_finite(),
            Self::Quantity(_) => true,
            Self::Neg(a) => a.all_finite(),
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b) => a.all_finite() && b.all_finite(),
        }
    }

    fn collect(&self, into: &mut Vec<String>) {
        match self {
            Self::Const(_) => {}
            Self::Quantity(id) => into.push(id.clone()),
            Self::Neg(a) => a.collect(into),
            Self::Add(a, b)
            | Self::Sub(a, b)
            | Self::Mul(a, b)
            | Self::Min(a, b)
            | Self::Max(a, b) => {
                a.collect(into);
                b.collect(into);
            }
        }
    }

    /// Variable ids the expression reads, ascending and distinct.
    #[must_use]
    pub fn references(&self) -> Vec<String> {
        let mut found = Vec::new();
        self.collect(&mut found);
        found.sort();
        found.dedup();
        found
    }

    /// Whether the expression is multilinear in its quantities (no minimum or maximum,
    /// no quantity multiplied by itself). The difference of two multilinear utilities
    /// attains its extremes at the vertices of a box, so the vertex scenarios then
    /// certify the whole assumption range at a grid point.
    #[must_use]
    pub fn is_multilinear(&self) -> bool {
        match self {
            Self::Const(_) | Self::Quantity(_) => true,
            Self::Neg(a) => a.is_multilinear(),
            Self::Add(a, b) | Self::Sub(a, b) => a.is_multilinear() && b.is_multilinear(),
            Self::Mul(a, b) => {
                let right = b.references();
                a.is_multilinear()
                    && b.is_multilinear()
                    && !a.references().iter().any(|id| right.contains(id))
            }
            Self::Min(..) | Self::Max(..) => false,
        }
    }

    /// Evaluate with each quantity's value in `values`; `None` for an unknown quantity.
    #[must_use]
    pub fn evaluate(&self, values: &HashMap<&str, f64>) -> Option<f64> {
        Some(match self {
            Self::Const(value) => *value,
            Self::Quantity(id) => *values.get(id.as_str())?,
            Self::Neg(a) => -a.evaluate(values)?,
            Self::Add(a, b) => a.evaluate(values)? + b.evaluate(values)?,
            Self::Sub(a, b) => a.evaluate(values)? - b.evaluate(values)?,
            Self::Mul(a, b) => a.evaluate(values)? * b.evaluate(values)?,
            Self::Min(a, b) => a.evaluate(values)?.min(b.evaluate(values)?),
            Self::Max(a, b) => a.evaluate(values)?.max(b.evaluate(values)?),
        })
    }

    /// Units of the expression: `None` for a pure constant, which adopts any units.
    fn units(&self, by_id: &HashMap<&str, &str>) -> Result<Option<String>, String> {
        match self {
            Self::Const(_) => Ok(None),
            Self::Quantity(id) => by_id
                .get(id.as_str())
                .map(|units| Some((*units).to_owned()))
                .ok_or_else(|| format!("unknown quantity `{id}`")),
            Self::Neg(a) => a.units(by_id),
            Self::Add(a, b) | Self::Sub(a, b) | Self::Min(a, b) | Self::Max(a, b) => {
                merge_units(a.units(by_id)?, b.units(by_id)?)
            }
            Self::Mul(a, b) => Ok(match (a.units(by_id)?, b.units(by_id)?) {
                (Some(x), Some(y)) => Some(format!("{x}*{y}")),
                (Some(x), None) | (None, Some(x)) => Some(x),
                (None, None) => None,
            }),
        }
    }
}

fn merge_units(left: Option<String>, right: Option<String>) -> Result<Option<String>, String> {
    match (left, right) {
        (Some(x), Some(y)) if x != y => Err(format!("`{x}` and `{y}`")),
        (Some(x), _) | (None, Some(x)) => Ok(Some(x)),
        (None, None) => Ok(None),
    }
}

/// One action and the utility it reads from the surface quantities.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct ActionUtility {
    /// Stable semantic action identity.
    pub id: String,
    /// Utility over the surface quantities.
    pub utility: UtilityTerm,
}

/// One surface quantity: its scientific coordinate and its range at every grid point.
#[derive(Clone, Debug, PartialEq)]
pub struct SurfaceQuantity {
    /// Scientific coordinate (units, population, regime, horizon, functional).
    pub quantity: ScientificQuantityWire,
    /// Lower end of the assumption range at each grid point.
    pub lower: Vec<f64>,
    /// Upper end of the assumption range at each grid point; equal to `lower` for a
    /// point surface.
    pub upper: Vec<f64>,
}

impl SurfaceQuantity {
    /// Whether the assumption range is not a single point somewhere on the grid.
    #[must_use]
    pub fn is_ranged(&self) -> bool {
        self.lower.iter().zip(&self.upper).any(|(low, high)| !same(*low, *high))
    }
}

/// What the surface range means; always an assumption range.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct AssumptionRangeStatement {
    /// Always [`SENSITIVITY_INFERENCE_CLAIM`].
    pub kind: String,
    /// Human-readable interpretation.
    pub interpretation: String,
}

/// A separate identified (partial-identification) bound of one surface quantity.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct IdentifiedBound {
    /// `variable_id` of the bounded quantity.
    pub quantity: String,
    /// Lower bound.
    pub lower: f64,
    /// Upper bound.
    pub upper: f64,
    /// Where the bound comes from.
    pub source: String,
}

/// A separate sampling interval of one surface quantity at every grid point.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SamplingInterval {
    /// `variable_id` of the quantity the interval describes.
    pub quantity: String,
    /// Nominal level in `(0, 1)`.
    pub level: f64,
    /// Interval method.
    pub method: String,
    /// Licensed composition method with the assumption range; refused unless listed in
    /// [`LICENSED_SAMPLING_COMPOSITIONS`].
    pub composed: Option<String>,
    /// Lower endpoint at each grid point (stored in the numbers section).
    #[serde(skip)]
    pub lower: Vec<f64>,
    /// Upper endpoint at each grid point (stored in the numbers section).
    #[serde(skip)]
    pub upper: Vec<f64>,
}

/// Status of the sampling uncertainty of the surface.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SamplingStatus {
    /// No sampling interval is reported, with the registered reason code and detail.
    Withheld {
        /// Registered reason code.
        reason_code: String,
        /// Namespaced detail.
        detail: String,
    },
    /// A sampling interval is reported next to, never inside, the assumption range.
    Reported(SamplingInterval),
}

/// The three uncertainty kinds, kept distinct.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct UncertaintyRelationship {
    /// What the surface `lower`/`upper` mean.
    pub assumption_range: AssumptionRangeStatement,
    /// Separate identified bound, when one exists.
    pub identified_bound: Option<IdentifiedBound>,
    /// Separate sampling interval or its withheld status.
    pub sampling: SamplingStatus,
}

/// A tipping result of the producing sensitivity analysis, retained as provenance.
/// It is reported, not recomputed: the decision tipping is recomputed from the surface.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SourceTipping {
    /// The perturbed factor.
    pub factor: String,
    /// How the producer's search ended.
    pub status: String,
    /// Lower end of the producer's certified bracket.
    pub lower: Option<f64>,
    /// Upper end of the producer's certified bracket.
    pub upper: Option<f64>,
    /// The producer's closed-form tipping fraction, when it has one.
    pub analytic: Option<f64>,
}

/// Provenance of the numbers.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SensitivityProvenance {
    /// Producer kind (`joint_mechanism_sensitivity_2_2` or a supplied surface).
    pub source_kind: String,
    /// Identity of the checked derivation the numbers came from.
    pub query_binding: String,
    /// Data or provider snapshot behind the numbers.
    pub provider_snapshot: String,
    /// Source regime identity.
    pub source_regime: String,
    /// Producer's method statement.
    pub method: String,
    /// Independently checked causal-contract identity.
    pub causal_contract_id: String,
    /// Decision threshold the producer searched, when it had one.
    pub decision_threshold: Option<f64>,
    /// The producer's own tipping results, reported as provenance.
    pub source_tipping: Vec<SourceTipping>,
}

/// Identity digests; premises and numbers are kept apart.
#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SensitivityIdentity {
    /// BLAKE3 of the canonical declarations (coordinate, support, quantities, actions,
    /// uncertainty kinds, provenance).
    pub premises_digest: String,
    /// BLAKE3 of the numerical section.
    pub data_digest: String,
    /// BLAKE3 of both digests.
    pub digest: String,
}

/// A change of leader between grid points.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct DecisionSwitch {
    /// Leading actions before the switch.
    pub from: Vec<String>,
    /// Leading actions after the switch.
    pub to: Vec<String>,
    /// Lower end of the coordinate bracket containing the switch.
    pub lower: f64,
    /// Upper end of the coordinate bracket containing the switch.
    pub upper: f64,
    /// Whether the switch is an exact tie at a grid coordinate (`lower` equals `upper`).
    pub exact: bool,
    /// Linear interpolation of the crossing between the two bracketing grid points. It
    /// is exact only if the utility difference is linear in the coordinate between them;
    /// present only for a point surface.
    pub interpolated: Option<f64>,
}

/// Why no action is robust.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum NoRobustReason {
    /// At these coordinates the best action depends on where in the assumption range
    /// the surface lies.
    MixedWithinRange {
        /// The coordinates.
        coordinates: Vec<f64>,
    },
    /// The leader changes more than once along the coordinate.
    MultipleSwitches,
    /// Actions tie at these coordinates without a change of leader.
    TieWithoutSwitch {
        /// The coordinates.
        coordinates: Vec<f64>,
    },
}

/// What the decision over the assumption range can claim.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SensitivityOutcome {
    /// One action is uniquely best at every grid point under every range vertex.
    InvariantAction {
        /// The action.
        action: String,
    },
    /// The unique leader changes exactly once along the coordinate.
    AssumptionDependent {
        /// The switch and its tipping coordinate.
        switch: DecisionSwitch,
    },
    /// No action is robust over the range.
    NoRobustAction {
        /// Why.
        reason: NoRobustReason,
        /// Switches found, when the leader moves along the coordinate.
        switches: Vec<DecisionSwitch>,
    },
    /// A grid point was not evaluated; invariance is not checked there.
    Unresolved {
        /// The first unevaluated coordinate.
        coordinate: f64,
    },
}

/// What is known at one grid point for classification.
#[derive(Clone, Debug, PartialEq)]
pub enum PointInput {
    /// The point was not evaluated.
    Unevaluated,
    /// The leading actions under each range-vertex scenario at this point.
    Scenarios {
        /// Leading actions (ids) of each scenario; one id is a unique leader.
        leaders: Vec<Vec<String>>,
        /// Utility of each action; present when the point has a single scenario.
        values: Option<Vec<(String, f64)>>,
    },
}

/// One grid point handed to [`classify_decision_points`].
#[derive(Clone, Debug, PartialEq)]
pub struct PointLeaders {
    /// Coordinate value.
    pub coordinate: f64,
    /// Evidence at the point.
    pub input: PointInput,
}

#[derive(Clone, Debug, PartialEq)]
enum PointState {
    Unique(String),
    Tie(Vec<String>),
    Mixed,
}

struct Run {
    state: PointState,
    first: usize,
    last: usize,
}

fn state_of(leaders: &[Vec<String>]) -> PointState {
    let normalized: Vec<Vec<String>> = leaders
        .iter()
        .map(|ids| {
            let mut sorted = ids.clone();
            sorted.sort();
            sorted
        })
        .collect();
    let Some(first) = normalized.first() else {
        return PointState::Mixed;
    };
    if normalized.iter().any(|ids| ids != first) {
        return PointState::Mixed;
    }
    match first.len() {
        0 => PointState::Mixed,
        1 => PointState::Unique(first[0].clone()),
        _ => PointState::Tie(first.clone()),
    }
}

fn runs_of(states: Vec<PointState>) -> Vec<Run> {
    let mut runs: Vec<Run> = Vec::new();
    for (index, state) in states.into_iter().enumerate() {
        if runs.last().is_some_and(|run| run.state == state) {
            if let Some(run) = runs.last_mut() {
                run.last = index;
            }
        } else {
            runs.push(Run { state, first: index, last: index });
        }
    }
    runs
}

fn leaders_of_state(state: &PointState) -> Vec<String> {
    match state {
        PointState::Unique(id) => vec![id.clone()],
        PointState::Tie(ids) => ids.clone(),
        PointState::Mixed => Vec::new(),
    }
}

fn interpolate(points: &[PointLeaders], i: usize, j: usize, a: &str, b: &str) -> Option<f64> {
    let value = |point: &PointLeaders, id: &str| -> Option<f64> {
        match &point.input {
            PointInput::Scenarios { values: Some(values), .. } => {
                values.iter().find(|(name, _)| name == id).map(|(_, x)| *x)
            }
            _ => None,
        }
    };
    let (left, right) = (&points[i], &points[j]);
    let before = value(left, a)? - value(left, b)?;
    let after = value(right, a)? - value(right, b)?;
    if before <= 0.0 || after >= 0.0 {
        return None;
    }
    Some(left.coordinate + before / (before - after) * (right.coordinate - left.coordinate))
}

fn switch_between(current: &Run, next: &Run, points: &[PointLeaders]) -> DecisionSwitch {
    let at = |index: usize| points[index].coordinate;
    let (lower, upper, exact) = match (&current.state, &next.state) {
        (_, PointState::Tie(_)) => (at(next.first), at(next.last), next.first == next.last),
        (PointState::Tie(_), _) => {
            (at(current.first), at(current.last), current.first == current.last)
        }
        _ => (at(current.last), at(next.first), false),
    };
    let interpolated = match (&current.state, &next.state) {
        (PointState::Unique(a), PointState::Unique(b)) => {
            interpolate(points, current.last, next.first, a, b)
        }
        _ => None,
    };
    DecisionSwitch {
        from: leaders_of_state(&current.state),
        to: leaders_of_state(&next.state),
        lower,
        upper,
        exact,
        interpolated,
    }
}

/// Switches along the coordinate and the coordinates where a tie is touched and left
/// again without a change of leader.
fn switches_of(runs: &[Run], points: &[PointLeaders]) -> (Vec<DecisionSwitch>, Vec<f64>) {
    let mut switches = Vec::new();
    let mut touched = Vec::new();
    let mut index = 0;
    while index + 1 < runs.len() {
        if let (PointState::Unique(before), PointState::Tie(_), Some(after)) =
            (&runs[index].state, &runs[index + 1].state, runs.get(index + 2))
        {
            if let PointState::Unique(beyond) = &after.state {
                let tie = &runs[index + 1];
                if before == beyond {
                    touched.extend((tie.first..=tie.last).map(|k| points[k].coordinate));
                } else {
                    switches.push(DecisionSwitch {
                        from: vec![before.clone()],
                        to: vec![beyond.clone()],
                        lower: points[tie.first].coordinate,
                        upper: points[tie.last].coordinate,
                        exact: tie.first == tie.last,
                        interpolated: None,
                    });
                }
                index += 2;
                continue;
            }
        }
        switches.push(switch_between(&runs[index], &runs[index + 1], points));
        index += 1;
    }
    (switches, touched)
}

/// Classify the decision over an ordered grid of coordinates.
///
/// `points` must be in ascending coordinate order. A point is `Unique` when every range
/// scenario has the same single leader, `Tie` when every scenario has the same set of
/// tied leaders and `Mixed` otherwise. All points `Unique` with one action is an
/// invariant action; no `Mixed` point and exactly one change of leader (an exact tie
/// bridging two leaders counts as the change) is an assumption-dependent switch with its
/// tipping coordinate; everything else has no robust action.
#[must_use]
pub fn classify_decision_points(points: &[PointLeaders]) -> SensitivityOutcome {
    if let Some(point) = points.iter().find(|p| matches!(p.input, PointInput::Unevaluated)) {
        return SensitivityOutcome::Unresolved { coordinate: point.coordinate };
    }
    let states: Vec<PointState> = points
        .iter()
        .map(|point| match &point.input {
            PointInput::Scenarios { leaders, .. } => state_of(leaders),
            PointInput::Unevaluated => PointState::Mixed,
        })
        .collect();
    if let Some(PointState::Unique(first)) = states.first() {
        if states.iter().all(|state| matches!(state, PointState::Unique(id) if id == first)) {
            return SensitivityOutcome::InvariantAction { action: first.clone() };
        }
    }
    let coordinates_where = |wanted: fn(&PointState) -> bool| -> Vec<f64> {
        points.iter().zip(&states).filter(|(_, s)| wanted(s)).map(|(p, _)| p.coordinate).collect()
    };
    let mixed = coordinates_where(|s| matches!(s, PointState::Mixed));
    if !mixed.is_empty() {
        return SensitivityOutcome::NoRobustAction {
            reason: NoRobustReason::MixedWithinRange { coordinates: mixed },
            switches: Vec::new(),
        };
    }
    let tied = coordinates_where(|s| matches!(s, PointState::Tie(_)));
    let runs = runs_of(states);
    let (mut switches, touched) = switches_of(&runs, points);
    if !touched.is_empty() {
        let reason = NoRobustReason::TieWithoutSwitch { coordinates: touched };
        return SensitivityOutcome::NoRobustAction { reason, switches };
    }
    match switches.len() {
        0 => SensitivityOutcome::NoRobustAction {
            reason: NoRobustReason::TieWithoutSwitch { coordinates: tied },
            switches,
        },
        1 => SensitivityOutcome::AssumptionDependent { switch: switches.remove(0) },
        _ => SensitivityOutcome::NoRobustAction {
            reason: NoRobustReason::MultipleSwitches,
            switches,
        },
    }
}

/// The declarations and numbers an artifact is built from. Grid points, quantities and
/// actions may be given in any order; [`SensitivityArtifact::new`] canonicalizes them, so
/// the identity does not depend on declaration order.
#[derive(Clone, Debug, PartialEq)]
pub struct SensitivityParts {
    /// The assumption coordinate.
    pub coordinate: AssumptionCoordinate,
    /// Grid points of the coordinate (finite, distinct, inside the declared range).
    pub grid: Vec<f64>,
    /// Support status of each grid point, in the order of `grid`.
    pub support: Vec<PointSupport>,
    /// Surface quantities with unique `variable_id`s.
    pub quantities: Vec<SurfaceQuantity>,
    /// Actions (at least two) reading the surface quantities.
    pub actions: Vec<ActionUtility>,
    /// The three uncertainty kinds.
    pub uncertainty: UncertaintyRelationship,
    /// Provenance of the numbers.
    pub provenance: SensitivityProvenance,
}

/// Stored metadata; the numbers are a separate `f64` section.
#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct SensitivityMeta {
    /// Metadata format version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// Always [`SENSITIVITY_INFERENCE_CLAIM`].
    pub inference_claim: String,
    /// The assumption coordinate.
    pub coordinate: AssumptionCoordinate,
    /// Support status of each grid point.
    pub support: Vec<PointSupport>,
    /// Quantity coordinates in canonical (`variable_id`) order.
    pub quantities: Vec<ScientificQuantityWire>,
    /// Actions in canonical (`id`) order.
    pub actions: Vec<ActionUtility>,
    /// The three uncertainty kinds.
    pub uncertainty: UncertaintyRelationship,
    /// Provenance.
    pub provenance: SensitivityProvenance,
    /// Identity digests.
    pub identity: SensitivityIdentity,
    /// Decision outcome over the whole grid; absent when the grid has an unsupported point.
    pub outcome: Option<SensitivityOutcome>,
}

/// A validated, canonical sensitivity artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct SensitivityArtifact {
    parts: SensitivityParts,
    identity: SensitivityIdentity,
    outcome: Option<SensitivityOutcome>,
}

fn canonicalize(mut parts: SensitivityParts) -> Result<SensitivityParts, IoError> {
    let n = parts.grid.len();
    if n == 0
        || n > MAX_SENSITIVITY_GRID_POINTS
        || parts.quantities.is_empty()
        || parts.quantities.len() > MAX_SENSITIVITY_QUANTITIES
        || parts.actions.len() > MAX_SENSITIVITY_ACTIONS
    {
        return Err(bounds_exceeded("grid points, quantities or actions are outside the bounds"));
    }
    let sampled = match &parts.uncertainty.sampling {
        SamplingStatus::Reported(interval) => Some((interval.lower.len(), interval.upper.len())),
        SamplingStatus::Withheld { .. } => None,
    };
    if parts.support.len() != n
        || parts.quantities.iter().any(|q| q.lower.len() != n || q.upper.len() != n)
        || sampled.is_some_and(|(low, high)| low != n || high != n)
    {
        return Err(invalid_surface("grid, support and surface lengths differ"));
    }
    if parts.grid.iter().any(|g| !g.is_finite()) {
        return Err(invalid_surface("grid points must be finite"));
    }
    let mut order: Vec<usize> = (0..n).collect();
    order.sort_by(|a, b| parts.grid[*a].total_cmp(&parts.grid[*b]));
    let pick = |values: &[f64]| -> Vec<f64> { order.iter().map(|i| values[*i]).collect() };
    parts.grid = pick(&parts.grid);
    parts.support = order.iter().map(|i| parts.support[*i]).collect();
    for quantity in &mut parts.quantities {
        quantity.lower = pick(&quantity.lower);
        quantity.upper = pick(&quantity.upper);
    }
    if let SamplingStatus::Reported(interval) = &mut parts.uncertainty.sampling {
        interval.lower = pick(&interval.lower);
        interval.upper = pick(&interval.upper);
    }
    parts.quantities.sort_by(|a, b| a.quantity.variable_id.cmp(&b.quantity.variable_id));
    parts.actions.sort_by(|a, b| a.id.cmp(&b.id));
    Ok(parts)
}

fn validate_coordinate(parts: &SensitivityParts) -> Result<(), IoError> {
    let c = &parts.coordinate;
    if blank(&c.id) || blank(&c.scale) || blank(&c.units) {
        return Err(invalid_surface("the coordinate needs an id, a scale and units"));
    }
    if !c.minimum.is_finite() || !c.maximum.is_finite() || c.minimum > c.maximum {
        return Err(invalid_surface("the declared assumption range must be finite and ordered"));
    }
    if parts.grid.iter().any(|g| *g < c.minimum || *g > c.maximum) {
        return Err(unsupported_coordinate(
            "a grid point lies outside the declared assumption range",
        ));
    }
    if parts.grid.windows(2).any(|w| w[0].total_cmp(&w[1]) != Ordering::Less) {
        return Err(invalid_surface("grid points must be distinct"));
    }
    Ok(())
}

fn validate_quantities(parts: &SensitivityParts) -> Result<(), IoError> {
    let mut ids = HashSet::new();
    for surface in &parts.quantities {
        ScientificQuantity::try_from(surface.quantity.clone())
            .map_err(|error| invalid_surface(&format!("quantity coordinate: {error}")))?;
        if !ids.insert(surface.quantity.variable_id.as_str()) {
            return Err(invalid_surface("two surface quantities share a variable id"));
        }
        let bad = surface
            .lower
            .iter()
            .zip(&surface.upper)
            .any(|(low, high)| !low.is_finite() || !high.is_finite() || low > high);
        if bad {
            return Err(invalid_surface("surface values must be finite with lower <= upper"));
        }
    }
    let first = &parts.quantities[0].quantity;
    if parts.quantities.iter().any(|q| {
        q.quantity.population_id != first.population_id || q.quantity.horizon != first.horizon
    }) {
        return Err(wrong_contract("surface quantities mix populations or horizons"));
    }
    if parts.quantities.iter().filter(|q| q.is_ranged()).count() > MAX_SENSITIVITY_RANGED_QUANTITIES
    {
        return Err(bounds_exceeded("too many surface quantities carry an assumption range"));
    }
    Ok(())
}

fn validate_actions(parts: &SensitivityParts) -> Result<(), IoError> {
    if parts.actions.len() < 2 {
        return Err(wrong_contract("a decision compares at least two actions"));
    }
    let known: HashSet<&str> =
        parts.quantities.iter().map(|q| q.quantity.variable_id.as_str()).collect();
    let by_id: HashMap<&str, &str> = parts
        .quantities
        .iter()
        .map(|q| (q.quantity.variable_id.as_str(), q.quantity.units.as_str()))
        .collect();
    let mut ids = HashSet::new();
    let mut shared: Option<String> = None;
    for action in &parts.actions {
        if blank(&action.id) || !ids.insert(action.id.as_str()) {
            return Err(wrong_contract("action ids must be nonblank and unique"));
        }
        if action.utility.depth() > MAX_SENSITIVITY_UTILITY_DEPTH || !action.utility.all_finite() {
            return Err(invalid_surface("utility expression is too deep or not finite"));
        }
        if action.utility.references().iter().any(|id| !known.contains(id.as_str())) {
            return Err(wrong_contract("an action reads a quantity the surface does not carry"));
        }
        let units = action.utility.units(&by_id).map_err(|message| {
            wrong_contract(&format!("mixed estimands in action `{}`: {message}", action.id))
        })?;
        shared = merge_units(shared, units).map_err(|message| {
            wrong_contract(&format!("actions compare unlike units: {message}"))
        })?;
    }
    Ok(())
}

fn validate_uncertainty(parts: &SensitivityParts) -> Result<(), IoError> {
    let u = &parts.uncertainty;
    let known: HashSet<&str> =
        parts.quantities.iter().map(|q| q.quantity.variable_id.as_str()).collect();
    if u.assumption_range.kind != SENSITIVITY_INFERENCE_CLAIM
        || blank(&u.assumption_range.interpretation)
    {
        return Err(wrong_contract("the surface range must be declared an assumption range"));
    }
    if let Some(bound) = &u.identified_bound {
        if !known.contains(bound.quantity.as_str())
            || !bound.lower.is_finite()
            || !bound.upper.is_finite()
            || bound.lower > bound.upper
            || blank(&bound.source)
        {
            return Err(invalid_surface("identified bound is malformed"));
        }
    }
    match &u.sampling {
        SamplingStatus::Withheld { reason_code, detail } => {
            if blank(reason_code) || blank(detail) {
                return Err(invalid_surface(
                    "a withheld interval needs its reason code and detail",
                ));
            }
        }
        SamplingStatus::Reported(interval) => {
            let finite = interval
                .lower
                .iter()
                .zip(&interval.upper)
                .all(|(low, high)| low.is_finite() && high.is_finite() && low <= high);
            if !known.contains(interval.quantity.as_str())
                || interval.level <= 0.0
                || interval.level >= 1.0
                || !interval.level.is_finite()
                || blank(&interval.method)
                || !finite
            {
                return Err(invalid_surface("sampling interval is malformed"));
            }
            if let Some(method) = &interval.composed {
                if !LICENSED_SAMPLING_COMPOSITIONS.contains(&method.as_str()) {
                    return Err(composition_not_licensed(&format!(
                        "composing a sampling interval with an assumption range by `{method}` is not licensed"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn validate(parts: &SensitivityParts) -> Result<(), IoError> {
    validate_coordinate(parts)?;
    validate_quantities(parts)?;
    validate_actions(parts)?;
    validate_uncertainty(parts)?;
    let p = &parts.provenance;
    if blank(&p.source_kind)
        || blank(&p.causal_contract_id)
        || p.decision_threshold.is_some_and(|t| !t.is_finite())
    {
        return Err(invalid_surface("provenance needs a source kind and a causal contract id"));
    }
    Ok(())
}

fn push_f64s(out: &mut Vec<u8>, values: &[f64]) {
    for value in values {
        out.extend_from_slice(&value.to_le_bytes());
    }
}

fn numbers_of(parts: &SensitivityParts) -> Vec<u8> {
    let mut out = Vec::new();
    push_f64s(&mut out, &parts.grid);
    for quantity in &parts.quantities {
        push_f64s(&mut out, &quantity.lower);
        push_f64s(&mut out, &quantity.upper);
    }
    if let SamplingStatus::Reported(interval) = &parts.uncertainty.sampling {
        push_f64s(&mut out, &interval.lower);
        push_f64s(&mut out, &interval.upper);
    }
    out
}

#[derive(Serialize)]
struct Premises<'a> {
    version: u16,
    feature: &'a str,
    inference_claim: &'a str,
    coordinate: &'a AssumptionCoordinate,
    support: &'a [PointSupport],
    quantities: Vec<&'a ScientificQuantityWire>,
    actions: &'a [ActionUtility],
    uncertainty: &'a UncertaintyRelationship,
    provenance: &'a SensitivityProvenance,
}

fn tagged_digest(tag: &str, bytes: &[u8]) -> String {
    let mut hasher = blake3::Hasher::new();
    hasher.update(tag.as_bytes());
    hasher.update(&(bytes.len() as u64).to_le_bytes());
    hasher.update(bytes);
    hasher.finalize().to_hex().to_string()
}

fn identity_of(parts: &SensitivityParts) -> Result<SensitivityIdentity, IoError> {
    let premises = to_cbor(&Premises {
        version: SENSITIVITY_ARTIFACT_VERSION,
        feature: SENSITIVITY_ARTIFACT_FEATURE,
        inference_claim: SENSITIVITY_INFERENCE_CLAIM,
        coordinate: &parts.coordinate,
        support: &parts.support,
        quantities: parts.quantities.iter().map(|q| &q.quantity).collect(),
        actions: &parts.actions,
        uncertainty: &parts.uncertainty,
        provenance: &parts.provenance,
    })?;
    let premises_digest = tagged_digest("antecedent.sensitivity_decision.premises.v1", &premises);
    let data_digest = tagged_digest("antecedent.sensitivity_decision.data.v1", &numbers_of(parts));
    let mut whole = blake3::Hasher::new();
    whole.update(b"antecedent.sensitivity_decision.identity.v1");
    whole.update(premises_digest.as_bytes());
    whole.update(data_digest.as_bytes());
    Ok(SensitivityIdentity {
        premises_digest,
        data_digest,
        digest: whole.finalize().to_hex().to_string(),
    })
}

/// Encode a metadata section and a numbers section as a checksummed container.
///
/// Hidden: the producer path is [`SensitivityArtifact::to_bytes`]; tests use this to build
/// deliberately inconsistent or resealed artifacts.
///
/// # Errors
/// An empty id, an encoding failure or an oversized payload.
#[doc(hidden)]
pub fn encode_parts(
    meta: &SensitivityMeta,
    numbers: &[u8],
    artifact_id: &str,
) -> Result<Vec<u8>, IoError> {
    if blank(artifact_id) {
        return Err(IoError::Convert("missing artifact id".into()));
    }
    let meta_bytes = to_cbor(meta)?;
    let encoded = EncodedArtifact {
        manifest: ArtifactManifest {
            format_version: crate::migrate::STABLE_FORMAT,
            minimum_reader_version: crate::migrate::STABLE_FORMAT,
            artifact_kind: ArtifactKind::Other(ARTIFACT_KIND.into()),
            library_version: SemanticVersion::from_crate_version(env!("CARGO_PKG_VERSION"))?,
            artifact_id: artifact_id.into(),
            sections: vec![
                section_descriptor(META_SECTION, "application/cbor", &meta_bytes),
                section_descriptor(NUMBERS_SECTION, "application/octet-stream", numbers),
            ],
            provenance: ProvenanceWire { note: "sensitivity_decision_assumption_range".into() },
        },
        sections: vec![
            SectionBytes::new(META_SECTION, meta_bytes),
            SectionBytes::new(NUMBERS_SECTION, numbers.to_vec()),
        ],
    };
    let mut bytes = Vec::new();
    encoded.write_to(&mut bytes)?;
    if bytes.len() > MAX_SENSITIVITY_ARTIFACT_BYTES {
        return Err(IoError::TooLarge);
    }
    Ok(bytes)
}

#[derive(Deserialize)]
struct VersionPeek {
    version: u16,
}

/// Decode the two sections of a container, refusing another version before the metadata
/// is interpreted. No semantic validation happens here.
///
/// Hidden: see [`encode_parts`].
///
/// # Errors
/// Oversized, truncated, corrupt, differently laid out or other-version artifacts.
#[doc(hidden)]
pub fn decode_parts(bytes: &[u8]) -> Result<(SensitivityMeta, Vec<u8>), IoError> {
    if bytes.len() > MAX_SENSITIVITY_ARTIFACT_BYTES {
        return Err(IoError::TooLarge);
    }
    let mut reader = ArtifactReader::open_seek(Cursor::new(bytes))?;
    let manifest = reader.manifest();
    if manifest.artifact_kind != ArtifactKind::Other(ARTIFACT_KIND.into())
        || manifest.sections.len() != 2
        || manifest.sections[0].id != META_SECTION
        || manifest.sections[1].id != NUMBERS_SECTION
    {
        return Err(IoError::Convert("unsupported sensitivity artifact layout".into()));
    }
    let declared = manifest
        .sections
        .iter()
        .try_fold(0_u64, |total, section| total.checked_add(section.uncompressed_size));
    if declared.is_none_or(|total| total > MAX_SENSITIVITY_ARTIFACT_BYTES as u64) {
        return Err(IoError::TooLarge);
    }
    let meta_section = reader.load_section(META_SECTION)?;
    let peek: VersionPeek = from_cbor(meta_section.as_bytes())?;
    if peek.version != SENSITIVITY_ARTIFACT_VERSION {
        return Err(IoError::UnsupportedVersion { version: u32::from(peek.version) });
    }
    let meta: SensitivityMeta = from_cbor(meta_section.as_bytes())?;
    let numbers = reader.load_section(NUMBERS_SECTION)?;
    Ok((meta, numbers.as_bytes().to_vec()))
}

fn parts_from(meta: &SensitivityMeta, numbers: &[u8]) -> Result<SensitivityParts, IoError> {
    let n = meta.support.len();
    let q = meta.quantities.len();
    if n == 0 || n > MAX_SENSITIVITY_GRID_POINTS || q == 0 || q > MAX_SENSITIVITY_QUANTITIES {
        return Err(bounds_exceeded("grid points or quantities are outside the bounds"));
    }
    let reported = matches!(meta.uncertainty.sampling, SamplingStatus::Reported(_));
    let count = n + 2 * n * q + if reported { 2 * n } else { 0 };
    if numbers.len() != count * 8 {
        return Err(invalid_surface("numbers section length differs from the declared shape"));
    }
    let mut values = numbers.chunks_exact(8).map(|chunk| {
        let mut word = [0_u8; 8];
        word.copy_from_slice(chunk);
        f64::from_le_bytes(word)
    });
    let mut take = |len: usize| values.by_ref().take(len).collect::<Vec<f64>>();
    let grid = take(n);
    let quantities = meta
        .quantities
        .iter()
        .map(|quantity| {
            let lower = take(n);
            let upper = take(n);
            SurfaceQuantity { quantity: quantity.clone(), lower, upper }
        })
        .collect();
    let mut uncertainty = meta.uncertainty.clone();
    if let SamplingStatus::Reported(interval) = &mut uncertainty.sampling {
        interval.lower = take(n);
        interval.upper = take(n);
    }
    Ok(SensitivityParts {
        coordinate: meta.coordinate.clone(),
        grid,
        support: meta.support.clone(),
        quantities,
        actions: meta.actions.clone(),
        uncertainty,
        provenance: meta.provenance.clone(),
    })
}

impl SensitivityArtifact {
    /// Canonicalize, validate and seal `parts` with their identity and decision outcome.
    ///
    /// # Errors
    /// Bounds, malformed surfaces (`invalid_surface`), mixed estimands or unlike units
    /// (`wrong_contract`), a grid point outside the declared range
    /// (`unsupported_coordinate`) and a sampling composition that is not licensed
    /// (`composition_not_licensed`).
    pub fn new(parts: SensitivityParts) -> Result<Self, IoError> {
        let parts = canonicalize(parts)?;
        validate(&parts)?;
        let identity = identity_of(&parts)?;
        let mut artifact = Self { parts, identity, outcome: None };
        artifact.outcome = if artifact.parts.support.contains(&PointSupport::Unsupported) {
            None
        } else {
            Some(artifact.decision_outcome(None)?)
        };
        Ok(artifact)
    }

    /// Adapt a 2.2 joint mechanism sensitivity result
    /// ([`JointMechanismSensitivityResult`]) into a surface of the effect over the
    /// assumption coordinate.
    ///
    /// With one declared factor the range is exactly linear in the contamination
    /// fraction (`baseline + t (extremum - baseline)` for `t = fraction / bound`), so the
    /// surface is filled on `grid_points` equally spaced fractions in `[0, bound]` with
    /// the exact endpoints. With two factors only the unperturbed point and the declared
    /// box are exact, so the surface is the two points `0` and `1` of the box scale and
    /// `grid_points` must be 2. The 2.2 withheld sampling status is carried unchanged:
    /// the range stays an assumption range.
    ///
    /// # Errors
    /// A result that is not an assumption range, a degenerate or unsupported factor
    /// set, a `grid_points` the result cannot fill exactly, or any
    /// [`Self::new`] refusal.
    pub fn from_joint_result(
        result: &JointMechanismSensitivityResult,
        effect: &ScientificQuantity,
        grid_points: usize,
        actions: Vec<ActionUtility>,
        causal_contract_id: &str,
    ) -> Result<Self, IoError> {
        if result.inference_claim != SENSITIVITY_INFERENCE_CLAIM {
            return Err(wrong_contract("only an assumption-range result can be adapted"));
        }
        effect.validate().map_err(|_| invalid_surface("effect quantity coordinate is invalid"))?;
        let (coordinate, grid, lower, upper) = surface_of(result, grid_points)?;
        let n = grid.len();
        Self::new(SensitivityParts {
            coordinate,
            grid,
            support: vec![PointSupport::Supported; n],
            quantities: vec![SurfaceQuantity {
                quantity: ScientificQuantityWire::from(effect),
                lower,
                upper,
            }],
            actions,
            uncertainty: UncertaintyRelationship {
                assumption_range: AssumptionRangeStatement {
                    kind: SENSITIVITY_INFERENCE_CLAIM.into(),
                    interpretation: result.interpretation.into(),
                },
                identified_bound: None,
                sampling: SamplingStatus::Withheld {
                    reason_code: result.uncertainty.reason_code.into(),
                    detail: result.uncertainty.detail.into(),
                },
            },
            provenance: SensitivityProvenance {
                source_kind: JOINT_SOURCE_KIND.into(),
                query_binding: result.query_binding.clone(),
                provider_snapshot: result.provider_snapshot.clone(),
                source_regime: format!("regime:{}", result.source_regime.raw()),
                method: result.receipt.method.into(),
                causal_contract_id: causal_contract_id.into(),
                decision_threshold: result.decision_threshold,
                source_tipping: result
                    .axis_tipping
                    .iter()
                    .map(|axis| SourceTipping {
                        factor: axis.factor.name().into(),
                        status: axis.status.name().into(),
                        lower: axis.bracket.map(|b| b.lower),
                        upper: axis.bracket.map(|b| b.upper),
                        analytic: axis.analytic,
                    })
                    .collect(),
            },
        })
    }

    /// The canonical declarations and numbers.
    #[must_use]
    pub fn parts(&self) -> &SensitivityParts {
        &self.parts
    }

    /// The identity digests.
    #[must_use]
    pub fn identity(&self) -> &SensitivityIdentity {
        &self.identity
    }

    /// The decision outcome over the whole grid; `None` when a grid point is unsupported.
    #[must_use]
    pub fn outcome(&self) -> Option<&SensitivityOutcome> {
        self.outcome.as_ref()
    }

    /// The assumption coordinate.
    #[must_use]
    pub fn coordinate(&self) -> &AssumptionCoordinate {
        &self.parts.coordinate
    }

    /// Grid points in ascending order.
    #[must_use]
    pub fn grid(&self) -> &[f64] {
        &self.parts.grid
    }

    /// Support status per grid point.
    #[must_use]
    pub fn support(&self) -> &[PointSupport] {
        &self.parts.support
    }

    /// Surface quantities in canonical order.
    #[must_use]
    pub fn quantities(&self) -> &[SurfaceQuantity] {
        &self.parts.quantities
    }

    /// Actions in canonical order.
    #[must_use]
    pub fn actions(&self) -> &[ActionUtility] {
        &self.parts.actions
    }

    /// The three uncertainty kinds.
    #[must_use]
    pub fn uncertainty(&self) -> &UncertaintyRelationship {
        &self.parts.uncertainty
    }

    /// Provenance.
    #[must_use]
    pub fn provenance(&self) -> &SensitivityProvenance {
        &self.parts.provenance
    }

    /// The units every action's utility is stated in, when any action reads a quantity.
    #[must_use]
    pub fn utility_units(&self) -> Option<String> {
        let by_id: HashMap<&str, &str> = self
            .parts
            .quantities
            .iter()
            .map(|q| (q.quantity.variable_id.as_str(), q.quantity.units.as_str()))
            .collect();
        self.parts.actions.iter().find_map(|action| action.utility.units(&by_id).ok().flatten())
    }

    /// Indices (canonical quantity order) of the quantities with an assumption range.
    #[must_use]
    pub fn ranged_quantities(&self) -> Vec<usize> {
        (0..self.parts.quantities.len()).filter(|i| self.parts.quantities[*i].is_ranged()).collect()
    }

    /// Number of range-vertex scenarios per grid point (`2^r`).
    #[must_use]
    pub fn scenario_count(&self) -> usize {
        1 << self.ranged_quantities().len()
    }

    /// Quantity values (canonical order) at a grid point under one range-vertex scenario:
    /// bit `b` of `mask` selects the upper end of the `b`-th ranged quantity.
    #[must_use]
    pub fn scenario_values(&self, point: usize, mask: usize) -> Vec<f64> {
        let ranged = self.ranged_quantities();
        self.parts
            .quantities
            .iter()
            .enumerate()
            .map(|(index, quantity)| {
                let upper = ranged
                    .iter()
                    .position(|r| *r == index)
                    .is_some_and(|bit| (mask >> bit) & 1 == 1);
                if upper { quantity.upper[point] } else { quantity.lower[point] }
            })
            .collect()
    }

    /// Utility of every action (canonical order) at a grid point under one scenario.
    ///
    /// # Errors
    /// A non-finite utility refuses.
    pub fn action_utilities(&self, point: usize, mask: usize) -> Result<Vec<f64>, IoError> {
        let values = self.scenario_values(point, mask);
        let by_id: HashMap<&str, f64> = self
            .parts
            .quantities
            .iter()
            .map(|q| q.quantity.variable_id.as_str())
            .zip(values.iter().copied())
            .collect();
        self.parts
            .actions
            .iter()
            .map(|action| {
                action.utility.evaluate(&by_id).filter(|v| v.is_finite()).ok_or_else(|| {
                    invalid_surface(&format!("utility of action `{}` is not finite", action.id))
                })
            })
            .collect()
    }

    /// Grid indices inside `range` (all points when `None`).
    ///
    /// # Errors
    /// A malformed or empty range refuses (`invalid_surface`); an unsupported grid point
    /// inside the range refuses (`unsupported_coordinate`).
    pub fn considered_points(&self, range: Option<(f64, f64)>) -> Result<Vec<usize>, IoError> {
        let grid = &self.parts.grid;
        let points: Vec<usize> = match range {
            None => (0..grid.len()).collect(),
            Some((low, high)) => {
                if !low.is_finite() || !high.is_finite() || low > high {
                    return Err(invalid_surface("the evaluated range must be finite and ordered"));
                }
                (0..grid.len()).filter(|k| grid[*k] >= low && grid[*k] <= high).collect()
            }
        };
        if points.is_empty() {
            return Err(invalid_surface("the evaluated range contains no grid point"));
        }
        if let Some(bad) =
            points.iter().find(|k| self.parts.support[**k] == PointSupport::Unsupported)
        {
            return Err(unsupported_coordinate(&format!(
                "assumption coordinate {} is unsupported",
                grid[*bad]
            )));
        }
        Ok(points)
    }

    fn point_input(&self, point: usize, scenarios: usize) -> Result<PointInput, IoError> {
        if self.parts.support[point] == PointSupport::Unevaluated {
            return Ok(PointInput::Unevaluated);
        }
        let mut leaders = Vec::with_capacity(scenarios);
        let mut values = None;
        for mask in 0..scenarios {
            let utilities = self.action_utilities(point, mask)?;
            leaders.push(leaders_of(&self.parts.actions, &utilities));
            if scenarios == 1 {
                values =
                    Some(self.parts.actions.iter().map(|a| a.id.clone()).zip(utilities).collect());
            }
        }
        Ok(PointInput::Scenarios { leaders, values })
    }

    /// Recompute the decision over `range` from the stored surface values and the
    /// declared action utilities: every grid point is evaluated at every range-vertex
    /// scenario and the leaders are classified along the coordinate.
    ///
    /// # Errors
    /// See [`Self::considered_points`] and [`Self::action_utilities`].
    pub fn decision_outcome(
        &self,
        range: Option<(f64, f64)>,
    ) -> Result<SensitivityOutcome, IoError> {
        let scenarios = self.scenario_count();
        let mut inputs = Vec::new();
        for point in self.considered_points(range)? {
            inputs.push(PointLeaders {
                coordinate: self.parts.grid[point],
                input: self.point_input(point, scenarios)?,
            });
        }
        Ok(classify_decision_points(&inputs))
    }

    fn meta(&self) -> SensitivityMeta {
        SensitivityMeta {
            version: SENSITIVITY_ARTIFACT_VERSION,
            feature: SENSITIVITY_ARTIFACT_FEATURE.into(),
            inference_claim: SENSITIVITY_INFERENCE_CLAIM.into(),
            coordinate: self.parts.coordinate.clone(),
            support: self.parts.support.clone(),
            quantities: self.parts.quantities.iter().map(|q| q.quantity.clone()).collect(),
            actions: self.parts.actions.clone(),
            uncertainty: self.parts.uncertainty.clone(),
            provenance: self.parts.provenance.clone(),
            identity: self.identity.clone(),
            outcome: self.outcome.clone(),
        }
    }

    /// Serialize through the checksummed sectioned container.
    ///
    /// # Errors
    /// An empty id, an encoding failure or an oversized payload.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, IoError> {
        encode_parts(&self.meta(), &numbers_of(&self.parts), artifact_id)
    }

    /// Consume an artifact by recomputation.
    ///
    /// The declarations and numbers are rebuilt and validated, the identity digests and
    /// the decision outcome are recomputed and must equal the stored ones, and, when the
    /// caller retained an `expected` identity of its own, that identity must match as
    /// well, so a resealed and self-consistent semantic change is refused.
    ///
    /// # Errors
    /// Corruption, another major version (`UnsupportedVersion`), any validation refusal of
    /// [`Self::new`], a changed identity or a stored outcome that does not replay
    /// (`wrong_contract`).
    pub fn from_bytes(
        bytes: &[u8],
        expected: Option<&SensitivityIdentity>,
    ) -> Result<Self, IoError> {
        let (meta, numbers) = decode_parts(bytes)?;
        if meta.feature != SENSITIVITY_ARTIFACT_FEATURE
            || meta.inference_claim != SENSITIVITY_INFERENCE_CLAIM
        {
            return Err(wrong_contract("feature marker or inference claim is not this format's"));
        }
        let artifact = Self::new(parts_from(&meta, &numbers)?)?;
        if let Some(field) = identity_diff(&meta.identity, &artifact.identity) {
            return Err(wrong_contract(&format!("stored {field} differs from the recomputed one")));
        }
        if meta.outcome != artifact.outcome {
            return Err(wrong_contract("stored decision outcome does not replay from the surface"));
        }
        if let Some(retained) = expected {
            if let Some(field) = identity_diff(retained, &artifact.identity) {
                return Err(wrong_contract(&format!(
                    "{field} differs from the consumer's retained identity"
                )));
            }
        }
        Ok(artifact)
    }
}

fn identity_diff(left: &SensitivityIdentity, right: &SensitivityIdentity) -> Option<&'static str> {
    if left.premises_digest != right.premises_digest {
        Some("premises digest")
    } else if left.data_digest != right.data_digest {
        Some("data digest")
    } else if left.digest != right.digest {
        Some("identity digest")
    } else {
        None
    }
}

/// Sorted ids of the actions whose utility equals the best.
fn leaders_of(actions: &[ActionUtility], utilities: &[f64]) -> Vec<String> {
    let best = utilities.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut ids: Vec<String> = actions
        .iter()
        .zip(utilities)
        .filter(|(_, utility)| same(**utility, best))
        .map(|(action, _)| action.id.clone())
        .collect();
    ids.sort();
    ids
}

type Surface = (AssumptionCoordinate, Vec<f64>, Vec<f64>, Vec<f64>);

/// Coordinate, grid and range surface of a 2.2 result, exact at every grid point.
fn surface_of(
    result: &JointMechanismSensitivityResult,
    grid_points: usize,
) -> Result<Surface, IoError> {
    let (baseline, minimum, maximum) =
        (result.baseline, result.range.minimum, result.range.maximum);
    match result.factors.as_slice() {
        [bound] => {
            let extent = bound.max_fraction;
            if extent <= 0.0 || !extent.is_finite() {
                return Err(unsupported_adaptation("the declared contamination bound is empty"));
            }
            if !(2..=MAX_SENSITIVITY_GRID_POINTS).contains(&grid_points) {
                return Err(bounds_exceeded("grid points must be between 2 and 1024"));
            }
            let last = grid_points - 1;
            let grid: Vec<f64> = (0..grid_points)
                .map(|j| if j == last { extent } else { extent * j as f64 / last as f64 })
                .collect();
            let blend = |end: f64| -> Vec<f64> {
                grid.iter()
                    .enumerate()
                    .map(|(j, g)| match j {
                        0 => baseline,
                        _ if j == last => end,
                        _ => baseline + (end - baseline) * (g / extent),
                    })
                    .collect()
            };
            let (lower, upper) = (blend(minimum), blend(maximum));
            let coordinate = AssumptionCoordinate {
                id: bound.factor.name().into(),
                scale: "contamination_fraction".into(),
                units: "fraction".into(),
                minimum: 0.0,
                maximum: extent,
            };
            Ok((coordinate, grid, lower, upper))
        }
        [_, _] => {
            if grid_points != 2 {
                return Err(unsupported_adaptation(
                    "a two-factor range is evaluated only at the unperturbed point and the declared box; it is not interpolated",
                ));
            }
            let coordinate = AssumptionCoordinate {
                id: "box_scale".into(),
                scale: "fraction_of_declared_box".into(),
                units: "fraction".into(),
                minimum: 0.0,
                maximum: 1.0,
            };
            Ok((coordinate, vec![0.0, 1.0], vec![baseline, minimum], vec![baseline, maximum]))
        }
        _ => Err(unsupported_adaptation("one or two declared factors are adapted")),
    }
}
