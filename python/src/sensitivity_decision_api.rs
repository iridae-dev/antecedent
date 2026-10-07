//! Bounded Python bridge for F17: an assumption-sensitivity surface consumed by a
//! decision contract.
//!
//! Python builds the declarations (a surface, a contract, a coordinate sub-range);
//! the artifact, its identity digests, the decision, the classification and every
//! refusal are Rust's. A refusal comes back as structured JSON for the Python layer
//! to raise as its own exception type. An assumption range is never a probability
//! and never a sampling interval; composing the two is refused natively.

use antecedent::analysis::sensitivity_decision::{
    ActionUtility, AssumptionCoordinate, AssumptionRangeStatement, IdentifiedBound,
    MAX_SENSITIVITY_ARTIFACT_BYTES, PointSupport, SENSITIVITY_INFERENCE_CLAIM, SamplingInterval,
    SamplingReport, SamplingStatus, ScenarioCoverage, SensitivityArtifact,
    SensitivityDecisionError, SensitivityDecisionResult, SensitivityDecisionSpec,
    SensitivityIdentity, SensitivityParts, SensitivityProvenance, SurfaceQuantity,
    UncertaintyRelationship, contract_from_artifact, decide,
};
use antecedent_core::{ScientificQuantity, reason_code};
use antecedent_design::decision_artifact::{contract_from_json_refusal, contract_to_json};
use antecedent_design::decision_contract::StructuralPolicy;
use antecedent_design::decision_eval::Verdict;
use antecedent_design::decision_structural::{AtomStatus, StructuralVerdict};
use antecedent_io::IoError;
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_validate::{
    JointDeviationSpec, JointFactor, JointFactorBound, JointSensitivityLimits,
};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::Deserialize;
use serde_json::{Value, json};

use crate::CausalSerializationError;
use crate::transport_z_api::PreparedZTransportStage;

const STAGE: &str = "sensitivity_decision";

fn serialization(error: impl std::fmt::Display) -> PyErr {
    CausalSerializationError::new_err(error.to_string())
}

fn check_size(text: &str) -> PyResult<()> {
    if text.len() > MAX_SENSITIVITY_ARTIFACT_BYTES {
        return Err(PyValueError::new_err("sensitivity declaration is too large"));
    }
    Ok(())
}

fn refusal_json(code: &str, detail: &str, message: &str) -> String {
    json!({
        "code": code,
        "stage": STAGE,
        "detail": detail,
        "offending": Value::Null,
        "message": message,
        "expected": Value::Null,
        "supplied": Value::Null,
        "remedy": Value::Null,
    })
    .to_string()
}

fn io_refusal(error: &IoError) -> String {
    match error {
        IoError::Refused { code, message } => {
            let (detail, text) = message
                .split_once(": ")
                .unwrap_or(("sensitivity_decision_composition.wrong_contract", message.as_str()));
            refusal_json(code, detail, text)
        }
        other => refusal_json(
            reason_code!("invalid_argument"),
            "sensitivity_decision_composition.invalid_artifact",
            &other.to_string(),
        ),
    }
}

fn decision_refusal(error: &SensitivityDecisionError) -> String {
    refusal_json(error.code, error.detail, &error.message)
}

fn declaration_refusal(message: &str) -> String {
    refusal_json(
        reason_code!("invalid_argument"),
        "sensitivity_decision_composition.invalid_surface",
        message,
    )
}

// ---------------------------------------------------------------------------
// Input wires
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SurfaceQuantityIn {
    quantity: ScientificQuantityWire,
    lower: Vec<f64>,
    upper: Vec<f64>,
}

#[derive(Deserialize)]
#[serde(rename_all = "snake_case")]
enum SamplingIn {
    Withheld {
        reason_code: String,
        detail: String,
    },
    Reported {
        quantity: String,
        level: f64,
        method: String,
        composed: Option<String>,
        lower: Vec<f64>,
        upper: Vec<f64>,
    },
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct UncertaintyIn {
    assumption_range: AssumptionRangeStatement,
    identified_bound: Option<IdentifiedBound>,
    sampling: SamplingIn,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SurfaceIn {
    coordinate: AssumptionCoordinate,
    grid: Vec<f64>,
    support: Vec<PointSupport>,
    quantities: Vec<SurfaceQuantityIn>,
    actions: Vec<ActionUtility>,
    uncertainty: UncertaintyIn,
    provenance: SensitivityProvenance,
}

impl SurfaceIn {
    fn into_parts(self) -> SensitivityParts {
        let sampling = match self.uncertainty.sampling {
            SamplingIn::Withheld { reason_code, detail } => {
                SamplingStatus::Withheld { reason_code, detail }
            }
            SamplingIn::Reported { quantity, level, method, composed, lower, upper } => {
                SamplingStatus::Reported(SamplingInterval {
                    quantity,
                    level,
                    method,
                    composed,
                    lower,
                    upper,
                })
            }
        };
        SensitivityParts {
            coordinate: self.coordinate,
            grid: self.grid,
            support: self.support,
            quantities: self
                .quantities
                .into_iter()
                .map(|q| SurfaceQuantity { quantity: q.quantity, lower: q.lower, upper: q.upper })
                .collect(),
            actions: self.actions,
            uncertainty: UncertaintyRelationship {
                assumption_range: self.uncertainty.assumption_range,
                identified_bound: self.uncertainty.identified_bound,
                sampling,
            },
            provenance: self.provenance,
        }
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct SpecIn {
    range: Option<(f64, f64)>,
    weights: Option<Vec<f64>>,
    sampling_composition: Option<String>,
}

// ---------------------------------------------------------------------------
// Output wires
// ---------------------------------------------------------------------------

fn sampling_status_json(status: &SamplingStatus) -> Value {
    match status {
        SamplingStatus::Withheld { reason_code, detail } => {
            json!({"withheld": {"reason_code": reason_code, "detail": detail}})
        }
        SamplingStatus::Reported(interval) => json!({"reported": {
            "quantity": interval.quantity,
            "level": interval.level,
            "method": interval.method,
            "composed": interval.composed,
            "lower": interval.lower,
            "upper": interval.upper,
        }}),
    }
}

fn summary_json(artifact: &SensitivityArtifact) -> Value {
    let quantities: Vec<Value> = artifact
        .quantities()
        .iter()
        .map(|q| json!({"quantity": q.quantity, "lower": q.lower, "upper": q.upper}))
        .collect();
    let support: Vec<&str> = artifact.support().iter().map(|s| s.name()).collect();
    json!({
        "inference_claim": SENSITIVITY_INFERENCE_CLAIM,
        "coordinate": artifact.coordinate(),
        "grid": artifact.grid(),
        "support": support,
        "quantities": quantities,
        "actions": artifact.actions(),
        "utility_units": artifact.utility_units(),
        "scenario_count": artifact.scenario_count(),
        "uncertainty": {
            "assumption_range": artifact.uncertainty().assumption_range,
            "identified_bound": artifact.uncertainty().identified_bound,
            "sampling": sampling_status_json(&artifact.uncertainty().sampling),
        },
        "provenance": artifact.provenance(),
        "identity": artifact.identity(),
        "outcome": artifact.outcome(),
    })
}

const fn policy_name(policy: StructuralPolicy) -> &'static str {
    match policy {
        StructuralPolicy::RequireInvariantBestAction => "require_invariant_best_action",
        StructuralPolicy::Maximin => "maximin",
        StructuralPolicy::BayesOverStructures => "bayes_over_structures",
        StructuralPolicy::ReportOnly => "report_only",
    }
}

fn policy_from(name: &str) -> PyResult<StructuralPolicy> {
    Ok(match name {
        "require_invariant_best_action" => StructuralPolicy::RequireInvariantBestAction,
        "maximin" => StructuralPolicy::Maximin,
        "bayes_over_structures" => StructuralPolicy::BayesOverStructures,
        "report_only" => StructuralPolicy::ReportOnly,
        other => return Err(PyValueError::new_err(format!("unknown structural policy `{other}`"))),
    })
}

fn leaders(verdict: &Verdict) -> Vec<String> {
    match verdict {
        Verdict::UniquelyOptimal(id) => vec![id.clone()],
        Verdict::Indistinguishable(ids) => ids.clone(),
        Verdict::NoAdmissibleAction => Vec::new(),
    }
}

fn verdict_json(verdict: &StructuralVerdict) -> Value {
    match verdict {
        StructuralVerdict::InvariantBest(action) => {
            json!({"kind": "invariant_best", "action": action})
        }
        StructuralVerdict::NoInvariantBest(by_atom) => json!({
            "kind": "no_invariant_best",
            "leaders": by_atom
                .iter()
                .map(|(atom, actions)| json!({"atom": atom, "actions": actions}))
                .collect::<Vec<_>>(),
        }),
        StructuralVerdict::WorstCaseChoice(action) => {
            json!({"kind": "worst_case_choice", "action": action})
        }
        StructuralVerdict::BayesChoice { action, evaluated_mass } => {
            json!({"kind": "bayes_choice", "action": action, "evaluated_mass": evaluated_mass})
        }
        StructuralVerdict::ReportOnly => json!({"kind": "report_only"}),
        StructuralVerdict::InsufficientScience(why) => {
            json!({"kind": "insufficient_science", "reason": why})
        }
        StructuralVerdict::NoAdmissibleAction => json!({"kind": "no_admissible_action"}),
    }
}

fn atom_json(atom: &antecedent_design::decision_structural::AtomSummary) -> Value {
    match &atom.status {
        AtomStatus::Evaluated(result) => json!({
            "id": atom.id,
            "probability": atom.probability,
            "status": "evaluated",
            "leaders": leaders(&result.verdict),
            "values": result
                .actions
                .iter()
                .map(|a| json!({
                    "id": a.id,
                    "expected_utility": a.expected_utility,
                    "value": a.value,
                    "admissible": a.admissible,
                }))
                .collect::<Vec<_>>(),
        }),
        AtomStatus::Unidentified => {
            json!({"id": atom.id, "probability": atom.probability, "status": "unidentified"})
        }
        AtomStatus::Unevaluated(reason) => json!({
            "id": atom.id,
            "probability": atom.probability,
            "status": "unevaluated",
            "reason": reason,
        }),
    }
}

const fn coverage_name(coverage: ScenarioCoverage) -> &'static str {
    match coverage {
        ScenarioCoverage::PointSurface => "point_surface",
        ScenarioCoverage::VertexCertified => "vertex_certified",
        ScenarioCoverage::VerticesOnly => "vertices_only",
    }
}

fn sampling_report_json(report: &SamplingReport) -> Value {
    match report {
        SamplingReport::Withheld { reason_code, detail } => {
            json!({"withheld": {"reason_code": reason_code, "detail": detail}})
        }
        SamplingReport::SeparateNotComposed {
            quantity,
            level,
            method,
            coordinates,
            lower,
            upper,
        } => json!({"separate_not_composed": {
            "quantity": quantity,
            "level": level,
            "method": method,
            "coordinates": coordinates,
            "lower": lower,
            "upper": upper,
        }}),
    }
}

fn decision_json(result: &SensitivityDecisionResult) -> Value {
    let structural = &result.structural;
    json!({
        "artifact_identity": result.artifact_identity,
        "coordinates": result.coordinates,
        "outcome": result.outcome,
        "coverage": coverage_name(result.coverage),
        "sampling": sampling_report_json(&result.sampling),
        "interpretation": result.interpretation,
        "structural": {
            "contract_identity": structural.contract_identity,
            "policy": policy_name(structural.policy),
            "verdict": verdict_json(&structural.verdict),
            "atoms": structural.atoms.iter().map(atom_json).collect::<Vec<_>>(),
            "actions": structural
                .actions
                .iter()
                .map(|a| json!({
                    "id": a.id,
                    "per_atom": a.per_atom,
                    "range": a.range,
                    "weighted_value": a.weighted_value,
                    "mass_where_best": a.mass_where_best,
                    "excluded_in": a.excluded_in,
                }))
                .collect::<Vec<_>>(),
            "unidentified_mass": structural.unidentified_mass,
            "unevaluated_mass": structural.unevaluated_mass,
            "evaluated_mass": structural.evaluated_mass,
        },
    })
}

// ---------------------------------------------------------------------------
// Artifact class
// ---------------------------------------------------------------------------

#[pyclass(name = "SensitivityArtifact", skip_from_py_object)]
pub(crate) struct PySensitivityArtifact {
    artifact: SensitivityArtifact,
}

#[pymethods]
impl PySensitivityArtifact {
    /// The canonical declarations, numbers, identity and stored outcome as JSON.
    #[getter]
    fn summary_json(&self) -> String {
        summary_json(&self.artifact).to_string()
    }

    /// Serialize through the bounded sectioned container.
    fn export<'py>(&self, py: Python<'py>, artifact_id: &str) -> PyResult<Bound<'py, PyBytes>> {
        let bytes = self.artifact.to_bytes(artifact_id).map_err(serialization)?;
        Ok(PyBytes::new(py, &bytes))
    }

    /// Consume an artifact by recomputation; `expected_identity_json` is an identity
    /// the consumer retained independently of the bytes. Returns the artifact or a
    /// structured refusal.
    #[staticmethod]
    #[pyo3(signature = (data, expected_identity_json=None))]
    fn consume(
        data: &[u8],
        expected_identity_json: Option<&str>,
    ) -> PyResult<(Option<Self>, Option<String>)> {
        let expected: Option<SensitivityIdentity> = match expected_identity_json {
            Some(text) => {
                check_size(text)?;
                match serde_json::from_str(text) {
                    Ok(identity) => Some(identity),
                    Err(error) => {
                        return Ok((None, Some(declaration_refusal(&error.to_string()))));
                    }
                }
            }
            None => None,
        };
        match SensitivityArtifact::from_bytes(data, expected.as_ref()) {
            Ok(artifact) => Ok((Some(Self { artifact }), None)),
            Err(error) => Ok((None, Some(io_refusal(&error)))),
        }
    }

    /// Recompute the decision over `range` from the surface and the declared action
    /// utilities alone. Returns the outcome JSON or a structured refusal.
    #[pyo3(signature = (range=None))]
    fn outcome_json(&self, range: Option<(f64, f64)>) -> (Option<String>, Option<String>) {
        match self.artifact.decision_outcome(range) {
            Ok(outcome) => (serde_json::to_string(&outcome).ok(), None),
            Err(error) => (None, Some(io_refusal(&error))),
        }
    }
}

/// Build an artifact from a declared surface.
#[pyfunction]
fn sensitivity_artifact_from_surface(
    surface_json: &str,
) -> PyResult<(Option<PySensitivityArtifact>, Option<String>)> {
    check_size(surface_json)?;
    let surface: SurfaceIn = match serde_json::from_str(surface_json) {
        Ok(surface) => surface,
        Err(error) => return Ok((None, Some(declaration_refusal(&error.to_string())))),
    };
    match SensitivityArtifact::new(surface.into_parts()) {
        Ok(artifact) => Ok((Some(PySensitivityArtifact { artifact }), None)),
        Err(error) => Ok((None, Some(io_refusal(&error)))),
    }
}

fn parse_factors(factors: Vec<(String, f64)>) -> Result<Vec<JointFactorBound>, String> {
    factors
        .into_iter()
        .map(|(name, max_fraction)| {
            JointFactor::from_name(&name)
                .map(|factor| JointFactorBound { factor, max_fraction })
                .ok_or_else(|| format!("joint_sensitivity.unknown_factor: {name:?}"))
        })
        .collect()
}

/// Run the 2.2 joint mechanism sensitivity of a prepared z stage and adapt its
/// assumption range into a composition-ready artifact.
#[pyfunction]
#[pyo3(signature=(stage, factors, effect_json, grid_points, actions_json, causal_contract_id, *, decision_threshold=None, total_budget=None, tolerance=1e-9, frontier_points=17, max_operations=100_000, max_depth=64, max_memory_bytes=67_108_864, memory_bytes=None, cancel=None))]
#[allow(clippy::too_many_arguments)]
fn sensitivity_artifact_from_z_joint(
    py: Python<'_>,
    stage: PyRef<'_, PreparedZTransportStage>,
    factors: Vec<(String, f64)>,
    effect_json: &str,
    grid_points: usize,
    actions_json: &str,
    causal_contract_id: &str,
    decision_threshold: Option<f64>,
    total_budget: Option<f64>,
    tolerance: f64,
    frontier_points: usize,
    max_operations: usize,
    max_depth: usize,
    max_memory_bytes: u64,
    memory_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(Option<PySensitivityArtifact>, Option<String>)> {
    check_size(effect_json)?;
    check_size(actions_json)?;
    let factors = match parse_factors(factors) {
        Ok(factors) => factors,
        Err(message) => return Ok((None, Some(declaration_refusal(&message)))),
    };
    let effect = match serde_json::from_str::<ScientificQuantityWire>(effect_json)
        .map_err(|e| e.to_string())
        .and_then(|wire| ScientificQuantity::try_from(wire).map_err(String::from))
    {
        Ok(effect) => effect,
        Err(message) => return Ok((None, Some(declaration_refusal(&message)))),
    };
    let actions: Vec<ActionUtility> = match serde_json::from_str(actions_json) {
        Ok(actions) => actions,
        Err(error) => return Ok((None, Some(declaration_refusal(&error.to_string())))),
    };
    let spec = JointDeviationSpec {
        factors,
        total_budget,
        decision_threshold,
        tolerance,
        frontier_points,
        limits: JointSensitivityLimits {
            operations: max_operations,
            depth: max_depth,
            memory_bytes: max_memory_bytes,
        },
    };
    let contract_id = causal_contract_id.to_owned();
    let ctx = stage.ctx(memory_bytes, cancel);
    let prepared = stage.prepared().clone();
    drop(stage);
    let outcome = crate::detach_catch(py, move || {
        Ok(prepared.sensitivity_artifact(&spec, &effect, grid_points, actions, &contract_id, &ctx))
    })?;
    match outcome {
        Ok(artifact) => Ok((Some(PySensitivityArtifact { artifact }), None)),
        Err(error) => Ok((None, Some(io_refusal(&error)))),
    }
}

/// The decision contract an artifact's own declared actions define.
#[pyfunction]
fn sensitivity_contract(
    artifact: &PySensitivityArtifact,
    policy: &str,
) -> PyResult<(Option<String>, Option<String>)> {
    let policy = policy_from(policy)?;
    match contract_from_artifact(&artifact.artifact, policy) {
        Ok(contract) => Ok((Some(contract_to_json(&contract).map_err(serialization)?), None)),
        Err(error) => Ok((None, Some(decision_refusal(&error)))),
    }
}

/// Evaluate a contract over the surface; returns the result JSON or a refusal.
#[pyfunction]
fn sensitivity_decide(
    contract_json: &str,
    artifact: &PySensitivityArtifact,
    spec_json: &str,
) -> PyResult<(Option<String>, Option<String>)> {
    check_size(contract_json)?;
    check_size(spec_json)?;
    let contract = match contract_from_json_refusal(contract_json) {
        Ok(contract) => contract,
        Err(refusal) => {
            return Ok((
                None,
                Some(refusal_json(
                    refusal.code,
                    &refusal.detail,
                    refusal.offending.as_deref().unwrap_or(""),
                )),
            ));
        }
    };
    let spec: SpecIn = match serde_json::from_str(spec_json) {
        Ok(spec) => spec,
        Err(error) => return Ok((None, Some(declaration_refusal(&error.to_string())))),
    };
    let spec = SensitivityDecisionSpec {
        range: spec.range,
        weights: spec.weights,
        sampling_composition: spec.sampling_composition,
    };
    match decide(&contract, &artifact.artifact, &spec) {
        Ok(result) => Ok((Some(decision_json(&result).to_string()), None)),
        Err(error) => Ok((None, Some(decision_refusal(&error)))),
    }
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PySensitivityArtifact>()?;
    m.add_function(wrap_pyfunction!(sensitivity_artifact_from_surface, m)?)?;
    m.add_function(wrap_pyfunction!(sensitivity_artifact_from_z_joint, m)?)?;
    m.add_function(wrap_pyfunction!(sensitivity_contract, m)?)?;
    m.add_function(wrap_pyfunction!(sensitivity_decide, m)?)
}
