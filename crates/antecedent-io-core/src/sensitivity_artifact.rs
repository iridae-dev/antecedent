//! `SensitivityArtifact` (`sensitivity_decision_v1`).
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
//!   method appears in [`SUPPORTED_SAMPLING_COMPOSITIONS`] (empty: none is supported).
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

mod classification;
mod utility;

pub use utility::UtilityTerm;
use utility::merge_units;

pub use classification::{
    DecisionSwitch, NoRobustReason, PointInput, PointLeaders, SensitivityOutcome,
    classify_decision_points,
};

use std::cmp::Ordering;
use std::collections::{BTreeSet, HashMap, HashSet};

use antecedent_contract::ScientificQuantity;
use antecedent_core::is_blank;
use serde::{Deserialize, Serialize};

use crate::convert::to_cbor;
use crate::error::IoError;
use crate::quantity_wire::ScientificQuantityWire;

/// The artifact major version this reader writes and accepts.
pub const SENSITIVITY_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const SENSITIVITY_ARTIFACT_FEATURE: &str = "sensitivity_decision_v1";
/// The only claim a surface range carries.
pub const SENSITIVITY_INFERENCE_CLAIM: &str = "assumption_range";
/// Identity of the reported assumption range within
/// [`SensitivityArtifact::provenance_chain`].
pub const SENSITIVITY_CLAIM_LINK_ID: &str = "sensitivity_claim";
/// Most bytes an artifact may occupy, enforced on export and on consumption.
pub const MAX_SENSITIVITY_ARTIFACT_BYTES: usize = 8 * 1024 * 1024;
/// Most grid points of the assumption coordinate.
pub const MAX_SENSITIVITY_GRID_POINTS: usize = 1024;
/// Most surface quantities.
pub const MAX_SENSITIVITY_QUANTITIES: usize = 64;
/// Most actions.
pub const MAX_SENSITIVITY_ACTIONS: usize = 64;
/// Most utility evaluations a surface may need: grid points times the `2^r` range vertices of
/// the `r` quantities whose assumption range is not a single point, times actions. Every grid
/// point is evaluated at every vertex of the box the ranged quantities span, for every action,
/// by the producer and again by every consumer.
pub const MAX_SENSITIVITY_SCENARIO_EVALUATIONS: usize = 1 << 24;
/// Deepest utility expression.
pub const MAX_SENSITIVITY_UTILITY_DEPTH: usize = 32;
/// Most nodes in one utility expression. Depth alone does not bound a tree: a binary tree of
/// depth `d` may hold `2^d` nodes, and every node is evaluated at every grid point, range
/// vertex and action.
pub const MAX_SENSITIVITY_UTILITY_NODES: usize = 1024;
/// Most utility node evaluations a surface may need: the scenario evaluations of
/// [`MAX_SENSITIVITY_SCENARIO_EVALUATIONS`] weighted by the node count of each action's
/// utility, that is grid points times range vertices times the nodes of every action.
pub const MAX_SENSITIVITY_SCENARIO_NODE_EVALUATIONS: usize = 1 << 28;

/// Intake bounds for a sensitivity artifact. The defaults are the `MAX_SENSITIVITY_*`
/// constants; a consumer with a larger budget passes its own to
/// [`SensitivityArtifact::new_within`] and [`SensitivityArtifact::from_bytes_within`].
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SensitivityLimits {
    /// Most container bytes.
    pub max_bytes: usize,
    /// Most grid points of the assumption coordinate.
    pub max_grid_points: usize,
    /// Most surface quantities.
    pub max_quantities: usize,
    /// Most actions.
    pub max_actions: usize,
    /// Most utility evaluations the range vertices of the surface may need.
    pub max_scenario_evaluations: usize,
    /// Deepest utility expression.
    pub max_utility_depth: usize,
    /// Most nodes in one utility expression.
    pub max_utility_nodes: usize,
    /// Most utility node evaluations the range vertices of the surface may need, counting
    /// every node of every action's utility at every grid point and vertex.
    pub max_scenario_node_evaluations: usize,
}

impl Default for SensitivityLimits {
    fn default() -> Self {
        Self {
            max_bytes: MAX_SENSITIVITY_ARTIFACT_BYTES,
            max_grid_points: MAX_SENSITIVITY_GRID_POINTS,
            max_quantities: MAX_SENSITIVITY_QUANTITIES,
            max_actions: MAX_SENSITIVITY_ACTIONS,
            max_scenario_evaluations: MAX_SENSITIVITY_SCENARIO_EVALUATIONS,
            max_utility_depth: MAX_SENSITIVITY_UTILITY_DEPTH,
            max_utility_nodes: MAX_SENSITIVITY_UTILITY_NODES,
            max_scenario_node_evaluations: MAX_SENSITIVITY_SCENARIO_NODE_EVALUATIONS,
        }
    }
}

/// Sampling-composition methods supported for an assumption range. None is supported, so a
/// sampling interval is always reported next to the range and never inside it.
pub const SUPPORTED_SAMPLING_COMPOSITIONS: &[&str] = &[];
/// What the surface range means.
pub const ASSUMPTION_RANGE_STATEMENT: &str = "assumption range over the declared coordinate set; not a probability, not a confidence interval and not a sampling interval";

const ARTIFACT_KIND: &str = crate::artifact_kinds::SENSITIVITY_DECISION;
const META_SECTION: &str = "sensitivity_decision.meta";
const NUMBERS_SECTION: &str = "sensitivity_decision.numbers";
const JOINT_SOURCE_KIND: &str = crate::artifact_kinds::JOINT_MECHANISM_SENSITIVITY;

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

fn composition_not_supported(text: &str) -> IoError {
    IoError::Refused {
        code: antecedent_core::reason_code!("route_not_supported"),
        message: format!("sensitivity_decision_composition.composition_not_supported: {text}"),
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
    /// Supported composition method with the assumption range; refused unless listed in
    /// [`SUPPORTED_SAMPLING_COMPOSITIONS`].
    pub composed: Option<String>,
    /// Calibration annotation of this interval, resolved from its construction basis.
    /// It annotates the interval and never withholds it.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration: Option<crate::contract_section::CalibrationSlotWire>,
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
    Reported(Box<SamplingInterval>),
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
    /// Producer kind (`joint_mechanism_sensitivity` or a supplied surface; artifacts sealed with the earlier
    /// `joint_mechanism_sensitivity_2_2` spelling carry that string and keep verifying).
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

fn canonicalize(
    mut parts: SensitivityParts,
    limits: &SensitivityLimits,
) -> Result<SensitivityParts, IoError> {
    let n = parts.grid.len();
    if n == 0
        || n > limits.max_grid_points
        || parts.quantities.is_empty()
        || parts.quantities.len() > limits.max_quantities
        || parts.actions.len() > limits.max_actions
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
    if is_blank(&c.id) || is_blank(&c.scale) || is_blank(&c.units) {
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
    Ok(())
}

/// Refuse a surface whose vertex enumeration exceeds the limits' scenario evaluations
/// ([`MAX_SENSITIVITY_SCENARIO_EVALUATIONS`] by default) or, weighted by the nodes of each
/// action's utility, its node evaluations ([`MAX_SENSITIVITY_SCENARIO_NODE_EVALUATIONS`]).
fn validate_scenario_work(
    parts: &SensitivityParts,
    limits: &SensitivityLimits,
) -> Result<(), IoError> {
    let ranged = parts.quantities.iter().filter(|q| q.is_ranged()).count();
    let scenarios = u32::try_from(ranged).ok().and_then(|r| 1_usize.checked_shl(r));
    let points = scenarios.and_then(|vertices| vertices.checked_mul(parts.grid.len()));
    let evaluations = points.and_then(|points| points.checked_mul(parts.actions.len()));
    if evaluations.is_none_or(|work| work > limits.max_scenario_evaluations) {
        return Err(bounds_exceeded(
            "the range vertices of the surface need more utility evaluations than the bound",
        ));
    }
    // Each evaluation walks the action's whole expression, so the work is weighted by the
    // nodes of every action's utility, not by the action count alone.
    let nodes = parts
        .actions
        .iter()
        .try_fold(0_usize, |total, action| total.checked_add(action.utility.node_count()));
    let node_evaluations = points.zip(nodes).and_then(|(points, nodes)| points.checked_mul(nodes));
    if node_evaluations.is_none_or(|work| work > limits.max_scenario_node_evaluations) {
        return Err(bounds_exceeded(
            "the utility expressions of the surface need more node evaluations than the bound",
        ));
    }
    Ok(())
}

fn validate_actions(parts: &SensitivityParts, limits: &SensitivityLimits) -> Result<(), IoError> {
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
        if is_blank(&action.id) || !ids.insert(action.id.as_str()) {
            return Err(wrong_contract("action ids must be nonblank and unique"));
        }
        if action.utility.depth() > limits.max_utility_depth || !action.utility.all_finite() {
            return Err(invalid_surface("utility expression is too deep or not finite"));
        }
        if action.utility.node_count() > limits.max_utility_nodes {
            return Err(bounds_exceeded("a utility expression has more nodes than the bound"));
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
        || is_blank(&u.assumption_range.interpretation)
    {
        return Err(wrong_contract("the surface range must be declared an assumption range"));
    }
    if let Some(bound) = &u.identified_bound {
        if !known.contains(bound.quantity.as_str())
            || !bound.lower.is_finite()
            || !bound.upper.is_finite()
            || bound.lower > bound.upper
            || is_blank(&bound.source)
        {
            return Err(invalid_surface("identified bound is malformed"));
        }
    }
    match &u.sampling {
        SamplingStatus::Withheld { reason_code, detail } => {
            if is_blank(reason_code) || is_blank(detail) {
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
                || is_blank(&interval.method)
                || !finite
            {
                return Err(invalid_surface("sampling interval is malformed"));
            }
            if let Some(method) = &interval.composed {
                if !SUPPORTED_SAMPLING_COMPOSITIONS.contains(&method.as_str()) {
                    return Err(composition_not_supported(&format!(
                        "composing a sampling interval with an assumption range by `{method}` is not supported"
                    )));
                }
            }
        }
    }
    Ok(())
}

fn validate(parts: &SensitivityParts, limits: &SensitivityLimits) -> Result<(), IoError> {
    validate_coordinate(parts)?;
    validate_quantities(parts)?;
    validate_actions(parts, limits)?;
    validate_scenario_work(parts, limits)?;
    validate_uncertainty(parts)?;
    let p = &parts.provenance;
    if is_blank(&p.source_kind)
        || is_blank(&p.causal_contract_id)
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
    antecedent_core::CanonicalWriter::digest_bytes_hex(tag, bytes)
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
    let whole =
        antecedent_core::CanonicalWriter::new("antecedent.sensitivity_decision.identity.v1")
            .text(&premises_digest)
            .text(&data_digest)
            .finish();
    Ok(SensitivityIdentity {
        premises_digest,
        data_digest,
        digest: antecedent_core::to_hex(&whole),
    })
}

/// Encode a metadata section and a numbers section as a checksummed container.
///
/// Hidden: the producer path is [`SensitivityArtifact::to_bytes`]; tests use this to build
/// deliberately inconsistent or resealed artifacts.
///
/// # Errors
/// An empty id, an encoding failure or an oversized payload.
#[cfg(feature = "test-util")]
#[doc(hidden)]
pub fn encode_parts(
    meta: &SensitivityMeta,
    numbers: &[u8],
    artifact_id: &str,
) -> Result<Vec<u8>, IoError> {
    encode_parts_impl(meta, numbers, artifact_id, MAX_SENSITIVITY_ARTIFACT_BYTES)
}

fn encode_parts_impl(
    meta: &SensitivityMeta,
    numbers: &[u8],
    artifact_id: &str,
    max_bytes: usize,
) -> Result<Vec<u8>, IoError> {
    crate::artifact_codec::encode_sectioned_io(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "sensitivity_decision_assumption_range",
            max_bytes,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        META_SECTION,
        meta,
        &[crate::OutSection {
            id: NUMBERS_SECTION,
            content_type: crate::CONTENT_BYTES,
            bytes: numbers,
        }],
        "unsupported sensitivity artifact layout",
    )
}

/// Decode the two sections of a container, refusing another version before the metadata
/// is interpreted. No semantic validation happens here.
///
/// Hidden: see [`encode_parts`].
///
/// # Errors
/// Oversized, truncated, corrupt, differently laid out or other-version artifacts.
#[cfg(feature = "test-util")]
#[doc(hidden)]
pub fn decode_parts(bytes: &[u8]) -> Result<(SensitivityMeta, Vec<u8>), IoError> {
    decode_parts_impl(bytes, MAX_SENSITIVITY_ARTIFACT_BYTES)
}

fn decode_parts_impl(
    bytes: &[u8],
    max_bytes: usize,
) -> Result<(SensitivityMeta, Vec<u8>), IoError> {
    let (meta, [payload]) = crate::artifact_codec::decode_sectioned_io_versioned(
        ARTIFACT_KIND,
        max_bytes,
        META_SECTION,
        &[NUMBERS_SECTION],
        "unsupported sensitivity artifact layout",
        SENSITIVITY_ARTIFACT_VERSION,
        bytes,
    )?;
    Ok((meta, payload))
}

fn parts_from(
    meta: &SensitivityMeta,
    numbers: &[u8],
    limits: &SensitivityLimits,
) -> Result<SensitivityParts, IoError> {
    let n = meta.support.len();
    let q = meta.quantities.len();
    if n == 0 || n > limits.max_grid_points || q == 0 || q > limits.max_quantities {
        return Err(bounds_exceeded("grid points or quantities are outside the bounds"));
    }
    let reported = matches!(meta.uncertainty.sampling, SamplingStatus::Reported(_));
    let count = n + 2 * n * q + if reported { 2 * n } else { 0 };
    if numbers.len() != count * 8 {
        return Err(invalid_surface("numbers section length differs from the declared shape"));
    }
    let decoded = crate::artifact_codec::f64s_from_le_bytes(numbers);
    let mut rest = decoded.as_slice();
    let mut take = |len: usize| crate::artifact_codec::split_reals(&mut rest, len).to_vec();
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

/// One declared contamination factor of a [`JointSensitivityView`].
#[derive(Clone, Debug, PartialEq)]
pub struct JointFactorView {
    /// Stable factor name (the coordinate id of a one-factor surface).
    pub factor: String,
    /// Largest contamination fraction, in `[0, 1]`.
    pub max_fraction: f64,
}

/// The endpoint-bootstrap replicates of a joint mechanism sensitivity result: for each
/// resampled table, the range's baseline (zero deviation), minimum and maximum over the declared
/// box. The sampling interval at every grid point is derived from them.
#[derive(Clone, Debug, PartialEq)]
pub struct JointSamplingView {
    /// Nominal level in `(0, 1)`.
    pub level: f64,
    /// Interval method name.
    pub method: String,
    /// Calibration annotation resolved from the construction basis; it never withholds.
    pub calibration: Option<crate::contract_section::CalibrationSlotWire>,
    /// Range baseline of each replicate.
    pub baselines: Vec<f64>,
    /// Range minimum over the full box of each replicate.
    pub minima: Vec<f64>,
    /// Range maximum over the full box of each replicate.
    pub maxima: Vec<f64>,
}

/// Linear-interpolation (type-7) quantile of an ascending slice.
#[allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "the position is within the slice length"
)]
fn type7_quantile(sorted: &[f64], probability: f64) -> f64 {
    let position = probability.clamp(0.0, 1.0) * (sorted.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    sorted[lower] + (position - lower as f64) * (sorted[upper] - sorted[lower])
}

impl JointSamplingView {
    /// The equal-tailed percentile interval of the replicate range scaled to `scale` of the
    /// declared box: `[q(zero + scale (min - zero)), q(zero + scale (max - zero))]`. At scale 0
    /// it is the ordinary percentile interval of the baseline; at scale 1 the conservative
    /// endpoint interval. The range is exactly linear in the scale, so this is exact for a
    /// one-factor box at every grid point and at the two endpoints of a two-factor box.
    fn interval_at(&self, scale: f64) -> Option<(f64, f64)> {
        let n = self.baselines.len();
        if n < 2 || self.minima.len() != n || self.maxima.len() != n {
            return None;
        }
        let at = |zero: f64, end: f64| -> f64 {
            if scale == 0.0 {
                zero
            } else if scale.to_bits() == 1.0_f64.to_bits() {
                end
            } else {
                zero + scale * (end - zero)
            }
        };
        let lows: Vec<f64> = (0..n).map(|i| at(self.baselines[i], self.minima[i])).collect();
        let highs: Vec<f64> = (0..n).map(|i| at(self.baselines[i], self.maxima[i])).collect();
        if lows.iter().chain(&highs).any(|v| !v.is_finite()) {
            return None;
        }
        let (mut lows, mut highs) = (lows, highs);
        lows.sort_by(f64::total_cmp);
        highs.sort_by(f64::total_cmp);
        let tail = (1.0 - self.level) / 2.0;
        Some((type7_quantile(&lows, tail), type7_quantile(&highs, 1.0 - tail)))
    }
}

/// A joint mechanism sensitivity result reduced to what the surface adapter
/// reads.
///
/// The validator owns the result type and converts it with `From`, so this
/// crate can sit below the validator and below the estimators.
#[derive(Clone, Debug, PartialEq)]
pub struct JointSensitivityView {
    /// The inference claim; only an assumption range is adaptable.
    pub inference_claim: String,
    /// Interpretation statement carried into the artifact.
    pub interpretation: String,
    /// The endpoint-bootstrap replicates, when the result carries its sampling interval.
    pub sampling: Option<JointSamplingView>,
    /// Registered reason code reported when the result carries no sampling interval (none was
    /// requested for this result).
    pub withheld_reason_code: String,
    /// Detail reported when the result carries no sampling interval.
    pub withheld_detail: String,
    /// Stable identity of the checked derivation's inputs.
    pub query_binding: String,
    /// Provider snapshot used by the formula.
    pub provider_snapshot: String,
    /// Raw id of the source-law regime.
    pub source_regime: u32,
    /// Optimization method.
    pub method: String,
    /// Decision threshold, when one was declared.
    pub decision_threshold: Option<f64>,
    /// Response under the unmodified source factors.
    pub baseline: f64,
    /// Minimum response over the declared box.
    pub minimum: f64,
    /// Maximum response over the declared box.
    pub maximum: f64,
    /// Declared factors in canonical order.
    pub factors: Vec<JointFactorView>,
    /// Axis tipping points, one per declared factor (empty without a threshold).
    pub axis_tipping: Vec<SourceTipping>,
}

impl From<&Self> for JointSensitivityView {
    fn from(view: &Self) -> Self {
        view.clone()
    }
}

impl SensitivityArtifact {
    /// Canonicalize, validate and seal `parts` with their identity and decision outcome.
    ///
    /// # Errors
    /// Bounds, malformed surfaces (`invalid_surface`), mixed estimands or unlike units
    /// (`wrong_contract`), a grid point outside the declared range
    /// (`unsupported_coordinate`) and a sampling composition that is not supported
    /// (`composition_not_supported`).
    pub fn new(parts: SensitivityParts) -> Result<Self, IoError> {
        Self::new_within(parts, &SensitivityLimits::default())
    }

    /// [`Self::new`] under the caller's intake bounds instead of the `MAX_SENSITIVITY_*`
    /// defaults.
    ///
    /// # Errors
    /// As [`Self::new`]; a bound of `limits` exceeded is `bounds_exceeded`.
    pub fn new_within(
        parts: SensitivityParts,
        limits: &SensitivityLimits,
    ) -> Result<Self, IoError> {
        let parts = canonicalize(parts, limits)?;
        validate(&parts, limits)?;
        let identity = identity_of(&parts)?;
        let mut artifact = Self { parts, identity, outcome: None };
        artifact.outcome = if artifact.parts.support.contains(&PointSupport::Unsupported) {
            None
        } else {
            Some(artifact.decision_outcome(None)?)
        };
        Ok(artifact)
    }

    /// Adapt a joint mechanism sensitivity result (a [`JointSensitivityView`], or the
    /// validator's `JointMechanismSensitivityResult`, which converts into one) into a
    /// surface of the effect over the assumption coordinate.
    ///
    /// With one declared factor the range is exactly linear in the contamination
    /// fraction (`baseline + t (extremum - baseline)` for `t = fraction / bound`), so the
    /// surface is filled on `grid_points` equally spaced fractions in `[0, bound]` with
    /// the exact endpoints. With two factors only the unperturbed point and the declared
    /// box are exact, so the surface is the two points `0` and `1` of the box scale and
    /// `grid_points` must be 2. When the result carries its endpoint-bootstrap replicates, the
    /// separate sampling interval is reported at every grid point next to, never inside, the
    /// range, with its calibration annotation; otherwise the not-requested status is carried.
    /// The range stays an assumption range.
    ///
    /// # Errors
    /// A result that is not an assumption range, a degenerate or unsupported factor
    /// set, a `grid_points` the result cannot fill exactly, or any
    /// [`Self::new`] refusal.
    pub fn from_joint_result<R>(
        result: &R,
        effect: &ScientificQuantity,
        grid_points: usize,
        actions: Vec<ActionUtility>,
        causal_contract_id: &str,
    ) -> Result<Self, IoError>
    where
        for<'a> JointSensitivityView: From<&'a R>,
    {
        let result = JointSensitivityView::from(result);
        if result.inference_claim != SENSITIVITY_INFERENCE_CLAIM {
            return Err(wrong_contract("only an assumption-range result can be adapted"));
        }
        effect.validate().map_err(|_| invalid_surface("effect quantity coordinate is invalid"))?;
        let (coordinate, grid, lower, upper) = surface_of(&result, grid_points)?;
        let n = grid.len();
        let effect_wire = ScientificQuantityWire::from(effect);
        let sampling = match &result.sampling {
            Some(view) => {
                if !(view.level > 0.0 && view.level < 1.0) || coordinate.maximum <= 0.0 {
                    return Err(invalid_surface("sampling interval level is malformed"));
                }
                let mut sampling_lower = Vec::with_capacity(n);
                let mut sampling_upper = Vec::with_capacity(n);
                for (j, g) in grid.iter().enumerate() {
                    let scale = if j == 0 {
                        0.0
                    } else if j + 1 == n {
                        1.0
                    } else {
                        g / coordinate.maximum
                    };
                    let (low, high) = view.interval_at(scale).ok_or_else(|| {
                        invalid_surface("the endpoint bootstrap replicates give no interval")
                    })?;
                    sampling_lower.push(low);
                    sampling_upper.push(high);
                }
                SamplingStatus::Reported(Box::new(SamplingInterval {
                    quantity: effect_wire.variable_id.clone(),
                    level: view.level,
                    method: view.method.clone(),
                    composed: None,
                    calibration: view.calibration.clone(),
                    lower: sampling_lower,
                    upper: sampling_upper,
                }))
            }
            None => SamplingStatus::Withheld {
                reason_code: result.withheld_reason_code.clone(),
                detail: result.withheld_detail.clone(),
            },
        };
        Self::new(SensitivityParts {
            coordinate,
            grid,
            support: vec![PointSupport::Supported; n],
            quantities: vec![SurfaceQuantity { quantity: effect_wire, lower, upper }],
            actions,
            uncertainty: UncertaintyRelationship {
                assumption_range: AssumptionRangeStatement {
                    kind: SENSITIVITY_INFERENCE_CLAIM.into(),
                    interpretation: result.interpretation,
                },
                identified_bound: None,
                sampling,
            },
            provenance: SensitivityProvenance {
                source_kind: JOINT_SOURCE_KIND.into(),
                query_binding: result.query_binding,
                provider_snapshot: result.provider_snapshot,
                source_regime: format!("regime:{}", result.source_regime),
                method: result.method,
                causal_contract_id: causal_contract_id.into(),
                decision_threshold: result.decision_threshold,
                source_tipping: result.axis_tipping,
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

    /// Queryable lineage of the sensitivity input: the checked causal contract and the
    /// data or provider snapshot behind the numbers, then the input itself (named by the
    /// artifact identity digest, so changing any premise or number changes its digest),
    /// then the `sensitivity_claim` that reports the assumption range. It is derived
    /// from the retained identity and provenance and is not stored in the artifact bytes.
    ///
    /// # Errors
    /// A blank causal-contract or snapshot identity refuses (`InvalidLink`).
    pub fn provenance_chain(&self) -> Result<antecedent_contract::ProvenanceChain, IoError> {
        use antecedent_contract::{CompositionLink, CompositionStage, ProvenanceChain};
        let provenance = &self.parts.provenance;
        let contract = format!("causal_contract:{}", provenance.causal_contract_id);
        let snapshot = format!("snapshot:{}", provenance.provider_snapshot);
        let input = format!("sensitivity_input:{}", self.identity.digest);
        let link = |id: &str, stage: CompositionStage, parents: &[&str]| CompositionLink {
            id: id.to_owned(),
            stage,
            parents: parents.iter().map(|p| (*p).to_owned()).collect(),
            declared_parent_digests: None,
        };
        ProvenanceChain::new(vec![
            link(&contract, CompositionStage::CausalContract, &[]),
            link(&snapshot, CompositionStage::Data, &[]),
            link(
                &input,
                CompositionStage::SensitivityInput,
                &[contract.as_str(), snapshot.as_str()],
            ),
            link(SENSITIVITY_CLAIM_LINK_ID, CompositionStage::Claim, &[input.as_str()]),
        ])
        .map_err(|error| {
            let refusal = error.to_refusal();
            IoError::Convert(format!("sensitivity_artifact.lineage: {}", refusal.detail))
        })
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
            let active: BTreeSet<&str> = self
                .parts
                .quantities
                .iter()
                .filter(|q| !same(q.lower[point], q.upper[point]))
                .map(|q| q.quantity.variable_id.as_str())
                .collect();
            if self
                .parts
                .actions
                .iter()
                .any(|action| action.utility.multilinear_over(&active).is_none())
            {
                return Err(invalid_surface("utility must be multilinear in ranged quantities"));
            }
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
        self.to_bytes_within(artifact_id, MAX_SENSITIVITY_ARTIFACT_BYTES)
    }

    /// [`Self::to_bytes`] under the caller's byte bound instead of
    /// [`MAX_SENSITIVITY_ARTIFACT_BYTES`].
    ///
    /// # Errors
    /// As [`Self::to_bytes`]; a payload over `max_bytes` is refused.
    pub fn to_bytes_within(&self, artifact_id: &str, max_bytes: usize) -> Result<Vec<u8>, IoError> {
        encode_parts_impl(&self.meta(), &numbers_of(&self.parts), artifact_id, max_bytes)
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
        Self::from_bytes_within(bytes, expected, &SensitivityLimits::default())
    }

    /// [`Self::from_bytes`] under the caller's intake bounds instead of the
    /// `MAX_SENSITIVITY_*` defaults.
    ///
    /// # Errors
    /// As [`Self::from_bytes`]; a bound of `limits` exceeded is refused.
    pub fn from_bytes_within(
        bytes: &[u8],
        expected: Option<&SensitivityIdentity>,
        limits: &SensitivityLimits,
    ) -> Result<Self, IoError> {
        let (meta, numbers) = decode_parts_impl(bytes, limits.max_bytes)?;
        if meta.feature != SENSITIVITY_ARTIFACT_FEATURE
            || meta.inference_claim != SENSITIVITY_INFERENCE_CLAIM
        {
            return Err(wrong_contract("feature marker or inference claim is not this format's"));
        }
        let artifact = Self::new_within(parts_from(&meta, &numbers, limits)?, limits)?;
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

/// Coordinate, grid and range surface of a joint mechanism sensitivity result, exact at every grid point.
fn surface_of(result: &JointSensitivityView, grid_points: usize) -> Result<Surface, IoError> {
    let (baseline, minimum, maximum) = (result.baseline, result.minimum, result.maximum);
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
                id: bound.factor.clone(),
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

#[cfg(test)]
mod multilinear_tests {
    use super::UtilityTerm;

    fn q(id: &str) -> UtilityTerm {
        UtilityTerm::Quantity(id.into())
    }

    #[test]
    fn multilinear_rejects_a_repeated_factor_and_a_min() {
        assert!(UtilityTerm::product(q("x"), q("y")).is_multilinear());
        assert!(UtilityTerm::sum(q("x"), q("x")).is_multilinear());
        assert!(!UtilityTerm::product(q("x"), q("x")).is_multilinear());
        assert!(!UtilityTerm::product(q("x"), UtilityTerm::sum(q("y"), q("x"))).is_multilinear());
        assert!(!UtilityTerm::Min(Box::new(q("x")), Box::new(q("y"))).is_multilinear());
        assert!(UtilityTerm::product(UtilityTerm::Const(2.0), q("x")).is_multilinear());
    }
}
