//! F17: consuming an assumption-sensitivity surface in a durable decision.
//!
//! A [`SensitivityArtifact`] (defined with its portable wire in
//! `antecedent_io::sensitivity_artifact`, where the 2.2 result is adapted) carries a
//! surface of one or more quantities over a declared assumption coordinate, with an
//! assumption range at every grid point. This module maps that surface into a
//! [`DecisionContract`] evaluation by treating every pair of a grid point and a
//! range-vertex scenario as one structural atom of [`evaluate_structural`]:
//!
//! * the atom's law is the exact one-row finite law "if this assumption coordinate
//!   holds and the range sits at this vertex, the quantities equal these values";
//! * assumption ranges are never probabilities: atoms carry a probability only when the
//!   caller declares genuine weights for a point surface, and a range with weights is
//!   refused;
//! * a sampling interval is never composed into the range. It is reported next to the
//!   decision ([`SamplingReport::SeparateNotComposed`]) and a request to compose it is
//!   refused unless the method is licensed
//!   ([`LICENSED_SAMPLING_COMPOSITIONS`], empty today);
//! * an unsupported grid point refuses the evaluation; an unevaluated one leaves the
//!   atom unevaluated, so the decision is reported unresolved instead of invariant.
//!
//! The decision is classified along the coordinate as an invariant action, an
//! assumption-dependent switch with its tipping coordinate, or no robust action
//! ([`SensitivityOutcome`]). The classification is also recomputable from the artifact
//! alone ([`SensitivityArtifact::decision_outcome`]); the two paths agree.
//!
//! Evaluation happens only at the declared grid points. Between grid points nothing is
//! evaluated; the interpolated crossing of a switch is labelled as an interpolation.

use antecedent_core::ScientificQuantity;
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::error::IoError;
use antecedent_io::quantity_wire::DistributionMeaningWire;
pub use antecedent_io::sensitivity_artifact::{
    ASSUMPTION_RANGE_STATEMENT, ActionUtility, AssumptionCoordinate, AssumptionRangeStatement,
    DecisionSwitch, IdentifiedBound, LICENSED_SAMPLING_COMPOSITIONS, NoRobustReason, PointSupport,
    SamplingInterval, SamplingStatus, SensitivityArtifact, SensitivityIdentity, SensitivityOutcome,
    SensitivityParts, SensitivityProvenance, SourceTipping, SurfaceQuantity,
    UncertaintyRelationship, UtilityTerm,
};
use antecedent_io::sensitivity_artifact::{PointInput, PointLeaders, classify_decision_points};

use crate::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use crate::decision_eval::Verdict;
use crate::decision_structural::{
    AtomEvidence, AtomStatus, StructuralAtom, StructuralDecisionResult, StructuralError,
    evaluate_structural,
};

/// What a sensitivity decision result can and cannot claim.
pub const SENSITIVITY_DECISION_INTERPRETATION: &str = "decision over the declared assumption range at the declared grid points only; the range is not a probability law and not a sampling interval, and nothing between grid points is evaluated";

/// How the caller restricts and weights the evaluation.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct SensitivityDecisionSpec {
    /// Inclusive coordinate sub-range to evaluate; the whole grid when `None`.
    pub range: Option<(f64, f64)>,
    /// Declared genuine probabilities, one per evaluated grid point in ascending
    /// coordinate order. Allowed only for a point surface (no assumption range);
    /// required by `BayesOverStructures`.
    pub weights: Option<Vec<f64>>,
    /// A requested composition of the sampling interval with the assumption range;
    /// refused unless the method is in [`LICENSED_SAMPLING_COMPOSITIONS`].
    pub sampling_composition: Option<String>,
}

/// What the evaluated scenarios cover at each grid point.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ScenarioCoverage {
    /// No quantity carries an assumption range; every point is a single scenario.
    PointSurface,
    /// Ranged quantities are evaluated at every vertex of their box and every utility is
    /// multilinear, so the vertices certify the whole range at each grid point.
    VertexCertified,
    /// Ranged quantities are evaluated at their vertices only and some utility is not
    /// multilinear (a minimum or maximum), so interior range values are not certified.
    VerticesOnly,
}

/// The sampling uncertainty of the surface, kept apart from the assumption range.
#[derive(Clone, Debug, PartialEq)]
pub enum SamplingReport {
    /// No sampling interval exists; the registered reason and detail say why.
    Withheld {
        /// Registered reason code.
        reason_code: String,
        /// Namespaced detail.
        detail: String,
    },
    /// A sampling interval exists and is reported next to the decision, flagged as not
    /// composed with the assumption range.
    SeparateNotComposed {
        /// `variable_id` of the quantity the interval describes.
        quantity: String,
        /// Nominal level.
        level: f64,
        /// Interval method.
        method: String,
        /// Evaluated coordinates.
        coordinates: Vec<f64>,
        /// Lower endpoint at each evaluated coordinate.
        lower: Vec<f64>,
        /// Upper endpoint at each evaluated coordinate.
        upper: Vec<f64>,
    },
}

/// A decision under an assumption-sensitivity surface.
#[derive(Clone, Debug, PartialEq)]
pub struct SensitivityDecisionResult {
    /// Identity digest of the artifact the decision was computed from.
    pub artifact_identity: String,
    /// Evaluated coordinates in ascending order.
    pub coordinates: Vec<f64>,
    /// The structural evaluation over the grid-point by range-vertex atoms.
    pub structural: StructuralDecisionResult,
    /// Invariant action, assumption-dependent switch or no robust action.
    pub outcome: SensitivityOutcome,
    /// What the scenarios certify.
    pub coverage: ScenarioCoverage,
    /// Sampling uncertainty, reported separately.
    pub sampling: SamplingReport,
    /// [`SENSITIVITY_DECISION_INTERPRETATION`].
    pub interpretation: &'static str,
}

/// A reason-coded refusal of the sensitivity decision composition.
#[derive(Clone, Debug, PartialEq)]
pub struct SensitivityDecisionError {
    /// Registered reason code.
    pub code: &'static str,
    /// Namespaced `sensitivity_decision_composition.*` detail.
    pub detail: &'static str,
    /// Human-readable context.
    pub message: String,
}

impl SensitivityDecisionError {
    /// Registered reason code.
    #[must_use]
    pub const fn reason_code(&self) -> &'static str {
        self.code
    }

    /// Namespaced detail.
    #[must_use]
    pub const fn detail(&self) -> &'static str {
        self.detail
    }
}

impl std::fmt::Display for SensitivityDecisionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}: {}", self.detail, self.message)
    }
}

impl std::error::Error for SensitivityDecisionError {}

fn wrong_contract(message: impl Into<String>) -> SensitivityDecisionError {
    SensitivityDecisionError {
        code: antecedent_core::reason_code!("decision_contract_unsatisfied"),
        detail: "sensitivity_decision_composition.wrong_contract",
        message: message.into(),
    }
}

fn not_licensed(message: impl Into<String>) -> SensitivityDecisionError {
    SensitivityDecisionError {
        code: antecedent_core::reason_code!("cell_not_licensed"),
        detail: "sensitivity_decision_composition.composition_not_licensed",
        message: message.into(),
    }
}

fn invalid_surface(message: impl Into<String>) -> SensitivityDecisionError {
    SensitivityDecisionError {
        code: antecedent_core::reason_code!("invalid_argument"),
        detail: "sensitivity_decision_composition.invalid_surface",
        message: message.into(),
    }
}

/// Map an artifact refusal to its registered code with a static detail.
fn from_io(error: &IoError) -> SensitivityDecisionError {
    let IoError::Refused { code, message } = error else {
        return invalid_surface(error.to_string());
    };
    let detail = match message.split_once(": ").map_or("", |(detail, _)| detail) {
        "sensitivity_decision_composition.unsupported_coordinate" => {
            "sensitivity_decision_composition.unsupported_coordinate"
        }
        "sensitivity_decision_composition.composition_not_licensed" => {
            "sensitivity_decision_composition.composition_not_licensed"
        }
        "sensitivity_decision_composition.bounds_exceeded" => {
            "sensitivity_decision_composition.bounds_exceeded"
        }
        "sensitivity_decision_composition.unsupported_adaptation" => {
            "sensitivity_decision_composition.unsupported_adaptation"
        }
        "sensitivity_decision_composition.invalid_surface" => {
            "sensitivity_decision_composition.invalid_surface"
        }
        _ => "sensitivity_decision_composition.wrong_contract",
    };
    SensitivityDecisionError { code, detail, message: message.clone() }
}

fn from_structural(error: &StructuralError) -> SensitivityDecisionError {
    SensitivityDecisionError {
        code: error.reason_code(),
        detail: "sensitivity_decision_composition.wrong_contract",
        message: format!("{error:?}"),
    }
}

fn expr_of(term: &UtilityTerm, refs: &[String]) -> UtilityExpr {
    let both = |a: &UtilityTerm, b: &UtilityTerm| (expr_of(a, refs), expr_of(b, refs));
    match term {
        UtilityTerm::Const(value) => UtilityExpr::Const(*value),
        UtilityTerm::Quantity(id) => {
            UtilityExpr::Input(refs.iter().position(|r| r == id).unwrap_or(0))
        }
        UtilityTerm::Neg(a) => UtilityExpr::Neg(Box::new(expr_of(a, refs))),
        UtilityTerm::Add(a, b) => {
            let (x, y) = both(a, b);
            UtilityExpr::sum(x, y)
        }
        UtilityTerm::Sub(a, b) => {
            let (x, y) = both(a, b);
            UtilityExpr::difference(x, y)
        }
        UtilityTerm::Mul(a, b) => {
            let (x, y) = both(a, b);
            UtilityExpr::product(x, y)
        }
        UtilityTerm::Min(a, b) => {
            let (x, y) = both(a, b);
            UtilityExpr::minimum(x, y)
        }
        UtilityTerm::Max(a, b) => {
            let (x, y) = both(a, b);
            UtilityExpr::maximum(x, y)
        }
    }
}

fn expr_multilinear(expr: &UtilityExpr) -> bool {
    match expr {
        UtilityExpr::Const(_) | UtilityExpr::Input(_) => true,
        UtilityExpr::Neg(a) => expr_multilinear(a),
        UtilityExpr::Add(a, b) | UtilityExpr::Sub(a, b) => {
            expr_multilinear(a) && expr_multilinear(b)
        }
        UtilityExpr::Mul(a, b) => {
            let right = b.inputs_used();
            expr_multilinear(a)
                && expr_multilinear(b)
                && !a.inputs_used().iter().any(|k| right.contains(k))
        }
        UtilityExpr::Min(..) | UtilityExpr::Max(..) => false,
    }
}

/// Build the [`DecisionContract`] the artifact's own declared actions define.
///
/// Each action reads exactly the surface quantities its utility names, under the
/// artifact's population and horizon. An action whose utility is a pure constant gets a
/// declared zero-weight read of the first quantity, because a contract action must read
/// an input; the read adds exactly zero. The criterion is posterior expected utility and
/// there are no hard constraints; the structural policy is the caller's.
///
/// # Errors
/// An artifact whose actions leave the utility units undetermined refuses.
pub fn contract_from_artifact(
    artifact: &SensitivityArtifact,
    policy: StructuralPolicy,
) -> Result<DecisionContract, SensitivityDecisionError> {
    let units = artifact.utility_units().ok_or_else(|| {
        wrong_contract("no action reads a quantity, so the utility units are undetermined")
    })?;
    let first = artifact
        .quantities()
        .first()
        .ok_or_else(|| invalid_surface("the artifact carries no surface quantity"))?;
    let scientific = |id: &str| -> Result<ScientificQuantity, SensitivityDecisionError> {
        let surface = artifact
            .quantities()
            .iter()
            .find(|q| q.quantity.variable_id == id)
            .ok_or_else(|| wrong_contract(format!("surface carries no quantity `{id}`")))?;
        ScientificQuantity::try_from(surface.quantity.clone())
            .map_err(|error| invalid_surface(format!("quantity coordinate: {error}")))
    };
    let mut actions = Vec::with_capacity(artifact.actions().len());
    for action in artifact.actions() {
        let mut refs = action.utility.references();
        let constant = refs.is_empty();
        if constant {
            refs.push(first.quantity.variable_id.clone());
        }
        let inputs = refs.iter().map(|id| scientific(id)).collect::<Result<Vec<_>, _>>()?;
        let mut utility = expr_of(&action.utility, &refs);
        if constant {
            utility = UtilityExpr::sum(
                utility,
                UtilityExpr::product(UtilityExpr::Const(0.0), UtilityExpr::Input(0)),
            );
        }
        actions.push(DecisionAction {
            id: action.id.clone(),
            kind: ActionKind::Intervention,
            inputs,
            utility,
        });
    }
    Ok(DecisionContract {
        actions,
        utility_units: units,
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: Vec::new(),
        target_population: first.quantity.population_id.clone(),
        horizon: first.quantity.horizon,
        structural_policy: policy,
    })
}

/// Refuse a contract the surface cannot answer: an unsupported criterion, unlike units
/// (mixed estimands) or an input the surface does not carry.
fn check_contract(
    contract: &DecisionContract,
    artifact: &SensitivityArtifact,
) -> Result<(), SensitivityDecisionError> {
    contract
        .validate()
        .map_err(|error| wrong_contract(format!("invalid decision contract: {error:?}")))?;
    if !matches!(
        contract.criterion,
        DecisionCriterion::PosteriorExpectedUtility | DecisionCriterion::MaximinOverStructures
    ) {
        return Err(wrong_contract(
            "surface scenarios are exact point laws; only expected-utility and maximin criteria are defined on them",
        ));
    }
    if artifact.utility_units().is_some_and(|units| units != contract.utility_units) {
        return Err(wrong_contract("the contract's utility units differ from the surface's"));
    }
    let surface = artifact
        .quantities()
        .iter()
        .map(|q| ScientificQuantity::try_from(q.quantity.clone()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| invalid_surface(format!("quantity coordinate: {error}")))?;
    for action in &contract.actions {
        for input in &action.inputs {
            if !surface.iter().any(|q| q.require_same_coordinate(input).is_ok()) {
                return Err(wrong_contract(format!(
                    "action `{}` reads a quantity the surface does not carry",
                    action.id
                )));
            }
        }
    }
    Ok(())
}

/// One-row exact law of the surface quantities at a grid point under one scenario.
fn scenario_law(
    artifact: &SensitivityArtifact,
    columns: &[ScientificQuantity],
    point: usize,
    mask: usize,
) -> Result<DistributionArtifact, SensitivityDecisionError> {
    let provenance = DistributionProvenance {
        source_id: format!("sensitivity:{}", artifact.identity().digest),
        provider_id: "sensitivity_decision_composition".into(),
        rng_id: "deterministic_exact".into(),
        snapshot_id: format!("{}[{point}]#{mask}", artifact.coordinate().id),
        causal_contract_id: artifact.provenance().causal_contract_id.clone(),
    };
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        columns,
        DrawAlignment::Joint,
        provenance,
    )
    .map_err(|error| from_io(&error))?;
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [1, columns.len()],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        artifact.scenario_values(point, mask),
    )
    .map_err(|error| from_io(&error))
}

fn build_atoms(
    artifact: &SensitivityArtifact,
    points: &[usize],
    probabilities: &[Option<f64>],
) -> Result<Vec<StructuralAtom>, SensitivityDecisionError> {
    let columns = artifact
        .quantities()
        .iter()
        .map(|q| ScientificQuantity::try_from(q.quantity.clone()))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|error| invalid_surface(format!("quantity coordinate: {error}")))?;
    let scenarios = artifact.scenario_count();
    let mut atoms = Vec::with_capacity(points.len() * scenarios);
    for (position, point) in points.iter().copied().enumerate() {
        for mask in 0..scenarios {
            let evidence = if artifact.support()[point] == PointSupport::Unevaluated {
                AtomEvidence::Unevaluated(antecedent_core::reason_code!("not_executed").into())
            } else {
                AtomEvidence::Evaluated(Box::new(scenario_law(artifact, &columns, point, mask)?))
            };
            atoms.push(StructuralAtom {
                id: format!("{}[{point}]#{mask}", artifact.coordinate().id),
                probability: probabilities[position],
                evidence,
            });
        }
    }
    Ok(atoms)
}

fn leaders_of_verdict(verdict: &Verdict) -> Vec<String> {
    match verdict {
        Verdict::UniquelyOptimal(id) => vec![id.clone()],
        Verdict::Indistinguishable(ids) => ids.clone(),
        Verdict::NoAdmissibleAction => Vec::new(),
    }
}

/// Group the per-atom leaders by grid point for the shared classification.
fn leaders_by_point(
    structural: &StructuralDecisionResult,
    artifact: &SensitivityArtifact,
    points: &[usize],
) -> Vec<PointLeaders> {
    let scenarios = artifact.scenario_count();
    points
        .iter()
        .copied()
        .enumerate()
        .map(|(position, point)| {
            let atoms = &structural.atoms[position * scenarios..(position + 1) * scenarios];
            let mut leaders = Vec::with_capacity(scenarios);
            let mut values = None;
            let mut evaluated = true;
            for atom in atoms {
                if let AtomStatus::Evaluated(result) = &atom.status {
                    leaders.push(leaders_of_verdict(&result.verdict));
                    if scenarios == 1 {
                        values = Some(
                            result
                                .actions
                                .iter()
                                .map(|a| (a.id.clone(), a.expected_utility))
                                .collect(),
                        );
                    }
                } else {
                    evaluated = false;
                }
            }
            let input = if evaluated {
                PointInput::Scenarios { leaders, values }
            } else {
                PointInput::Unevaluated
            };
            PointLeaders { coordinate: artifact.grid()[point], input }
        })
        .collect()
}

fn coverage_of(contract: &DecisionContract, artifact: &SensitivityArtifact) -> ScenarioCoverage {
    if artifact.ranged_quantities().is_empty() {
        ScenarioCoverage::PointSurface
    } else if contract.actions.iter().all(|a| expr_multilinear(&a.utility)) {
        ScenarioCoverage::VertexCertified
    } else {
        ScenarioCoverage::VerticesOnly
    }
}

fn sampling_report(artifact: &SensitivityArtifact, points: &[usize]) -> SamplingReport {
    match &artifact.uncertainty().sampling {
        SamplingStatus::Withheld { reason_code, detail } => {
            SamplingReport::Withheld { reason_code: reason_code.clone(), detail: detail.clone() }
        }
        SamplingStatus::Reported(interval) => SamplingReport::SeparateNotComposed {
            quantity: interval.quantity.clone(),
            level: interval.level,
            method: interval.method.clone(),
            coordinates: points.iter().map(|k| artifact.grid()[*k]).collect(),
            lower: points.iter().map(|k| interval.lower[*k]).collect(),
            upper: points.iter().map(|k| interval.upper[*k]).collect(),
        },
    }
}

/// Probability per evaluated grid point: declared weights only, and only for a point
/// surface. An assumption range never becomes a probability.
fn point_probabilities(
    spec: &SensitivityDecisionSpec,
    artifact: &SensitivityArtifact,
    n_points: usize,
) -> Result<Vec<Option<f64>>, SensitivityDecisionError> {
    let Some(weights) = &spec.weights else {
        return Ok(vec![None; n_points]);
    };
    if artifact.scenario_count() > 1 {
        return Err(wrong_contract(
            "an assumption range is not a probability law; declared weights apply only to a point surface",
        ));
    }
    if weights.len() != n_points {
        return Err(wrong_contract("declare exactly one weight per evaluated grid point"));
    }
    Ok(weights.iter().map(|w| Some(*w)).collect())
}

/// Evaluate `contract` over the surface of `artifact`.
///
/// Every evaluated grid point and range-vertex scenario becomes one structural atom of
/// [`evaluate_structural`], so the contract's [`StructuralPolicy`] decides how the
/// assumption scenarios are combined (invariant best action, worst case, or Bayes over
/// declared genuine weights). The result also classifies the decision along the
/// coordinate (see [`SensitivityOutcome`]).
///
/// # Errors
/// Refuses (a) a requested sampling composition that is not licensed
/// (`composition_not_licensed`, code `cell_not_licensed`); (b) an invalid contract, an
/// unsupported criterion, unlike units or an input the surface does not carry, weights on
/// a ranged surface, a weight count that does not match the grid, and `Bayes` without
/// genuine weights (`wrong_contract`, code `decision_contract_unsatisfied`); (c) an
/// unsupported grid point inside the evaluated range (`unsupported_coordinate`, code
/// `quantity_semantics_mismatch`); (d) a malformed or empty range (`invalid_surface`).
pub fn evaluate_sensitivity_decision(
    contract: &DecisionContract,
    artifact: &SensitivityArtifact,
    spec: &SensitivityDecisionSpec,
) -> Result<SensitivityDecisionResult, SensitivityDecisionError> {
    if let Some(method) = &spec.sampling_composition {
        return Err(not_licensed(format!(
            "composing a sampling interval with an assumption range by `{method}` is not licensed; licensed methods: {LICENSED_SAMPLING_COMPOSITIONS:?}"
        )));
    }
    check_contract(contract, artifact)?;
    let points = artifact.considered_points(spec.range).map_err(|error| from_io(&error))?;
    let probabilities = point_probabilities(spec, artifact, points.len())?;
    let atoms = build_atoms(artifact, &points, &probabilities)?;
    let structural = evaluate_structural(contract, &atoms).map_err(|e| from_structural(&e))?;
    let outcome = classify_decision_points(&leaders_by_point(&structural, artifact, &points));
    Ok(SensitivityDecisionResult {
        artifact_identity: artifact.identity().digest.clone(),
        coordinates: points.iter().map(|k| artifact.grid()[*k]).collect(),
        structural,
        outcome,
        coverage: coverage_of(contract, artifact),
        sampling: sampling_report(artifact, &points),
        interpretation: SENSITIVITY_DECISION_INTERPRETATION,
    })
}
