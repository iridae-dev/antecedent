//! Python bridge for composition-aware uncertainty.
//!
//! Python hands over draws (or artifacts), declarations and numbers; the propagation rule
//! (which dependence information a composition needs), the arithmetic, the artifact and every
//! refusal are Rust's. A refusal comes back as structured JSON for the Python layer to raise as
//! a typed exception. Nothing here pairs draws by index: rows are paired only under one
//! alignment identity, and a dependence envelope is never a point.

use antecedent::analysis::joint_law_propagation::{
    CorrelationRangeSummary, EnvelopeQueryWire, JointLawPropagationRefusal, LinearDependence,
    MixtureComponent, MomentInput, PropagationClaim, PropagationDraws, ValueFunctional,
    alignment_identity, consume_joint_law_propagation_artifact, propagate,
    propagate_aligned_functional, propagate_aligned_linear, propagate_correlation_range,
    propagate_dependence_envelope, propagate_linear_moments, propagate_mixture,
};
use antecedent_contract::{
    InputAlignment, InputRole, PropagationDependence, PropagationInput, PropagationOperation,
    PropagationRequest, RequestedReading,
};
use antecedent_io::joint_law_propagation_artifact::{PropagationRequestWire, float_bits};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::{Value as Json, json};

use crate::PyCancellationToken;
use crate::distribution_api::PyJointDistributionArtifact;
use crate::transport_common::execution_context;

type Refused<T> = (Option<T>, Option<String>);
/// Exported artifact bytes and the claim summary JSON.
type ClaimParts = (Py<PyBytes>, String);

const STAGE: &str = "propagate";

fn refusal_json(refusal: &JointLawPropagationRefusal) -> String {
    match refusal.refusals.first() {
        Some(first) => json!({
            "code": first.code,
            "stage": first.stage,
            "detail": first.detail,
            "message": refusal.to_string(),
            "offending": first.offending,
            "expected": first.expected,
            "supplied": first.supplied,
            "remedy": first.remedy,
        })
        .to_string(),
        None => invalid_json("refused".to_owned()),
    }
}

fn invalid_json(message: String) -> String {
    json!({
        "code": antecedent_core::reason_code!("invalid_argument"),
        "stage": STAGE,
        "detail": "joint_law_propagation.invalid_request",
        "message": message,
        "offending": Json::Null,
        "remedy": Json::Null,
    })
    .to_string()
}

fn operation(name: &str) -> Result<PropagationOperation, String> {
    PropagationOperation::from_name(name)
        .ok_or_else(|| invalid_json(format!("unknown operation `{name}`")))
}

fn role(name: &str) -> Result<InputRole, String> {
    InputRole::from_name(name).ok_or_else(|| invalid_json(format!("unknown input role `{name}`")))
}

fn claim_parts(py: Python<'_>, claim: &PropagationClaim) -> Refused<ClaimParts> {
    let bytes = match claim.export() {
        Ok(bytes) => bytes,
        Err(refusal) => return (None, Some(refusal_json(&refusal))),
    };
    let values: Vec<Json> = claim.values().into_iter().map(|(n, v)| json!([n, v])).collect();
    let summary = json!({
        "identity": claim.identity(),
        "operation": claim.operation(),
        "route": claim.route(),
        "label": claim.label(),
        "is_envelope": claim.is_envelope(),
        "alignment_id": claim.alignment_id(),
        "n_rows": claim.n_rows(),
        "values": values,
    });
    (Some((PyBytes::new(py, &bytes).unbind(), summary.to_string())), None)
}

fn finish(
    py: Python<'_>,
    outcome: Result<PropagationClaim, JointLawPropagationRefusal>,
) -> Refused<ClaimParts> {
    match outcome {
        Ok(claim) => claim_parts(py, &claim),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    }
}

macro_rules! tri {
    ($e:expr) => {
        match $e {
            Ok(value) => value,
            Err(refusal) => return (None, Some(refusal)),
        }
    };
}

/// Alignment identity and evidence identity of one artifact column, as JSON.
#[pyfunction]
fn joint_law_column(
    input_id: &str,
    artifact: &PyJointDistributionArtifact,
    column: usize,
) -> Refused<String> {
    match PropagationDraws::from_artifact_column(input_id, artifact.inner(), column) {
        Ok(_) => {}
        Err(refusal) => return (None, Some(refusal_json(&refusal))),
    }
    let identity = artifact.inner().metadata().identity.clone();
    let aligned = identity.alignment == antecedent_io::distribution_artifact::DrawAlignment::Joint;
    let alignment_id = if aligned { alignment_identity(artifact.inner()).ok() } else { None };
    (
        Some(
            json!({
                "evidence_id": identity.snapshot_id,
                "alignment": if aligned { "joint" } else { "independent_marginals" },
                "alignment_id": alignment_id,
            })
            .to_string(),
        ),
        None,
    )
}

/// `E[g(X, Y)]` of a closed functional under the aligned law of two artifact columns.
#[pyfunction]
#[pyo3(signature = (operation_name, x_id, x, x_column, y_id, y, y_column, functional, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "two named columns plus the declaration")]
fn joint_law_aligned_functional(
    py: Python<'_>,
    operation_name: &str,
    x_id: &str,
    x: &PyJointDistributionArtifact,
    x_column: usize,
    y_id: &str,
    y: &PyJointDistributionArtifact,
    y_column: usize,
    functional: &str,
    cancel: Option<PyCancellationToken>,
) -> Refused<ClaimParts> {
    let op = tri!(operation(operation_name));
    let Some(functional) = ValueFunctional::from_name(functional) else {
        return (None, Some(invalid_json(format!("unknown functional `{functional}`"))));
    };
    let dx = match PropagationDraws::from_artifact_column(x_id, x.inner(), x_column) {
        Ok(v) => v,
        Err(r) => return (None, Some(refusal_json(&r))),
    };
    let dy = match PropagationDraws::from_artifact_column(y_id, y.inner(), y_column) {
        Ok(v) => v,
        Err(r) => return (None, Some(refusal_json(&r))),
    };
    let ctx = execution_context(0, None, cancel);
    finish(py, propagate_aligned_functional(op, &dx, &dy, functional, &ctx))
}

/// Mean and variance of `a X + b Y` from two aligned artifact columns.
#[pyfunction]
#[pyo3(signature = (operation_name, a, b, x_id, x, x_column, y_id, y, y_column, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "two named columns plus the declaration")]
fn joint_law_aligned_linear(
    py: Python<'_>,
    operation_name: &str,
    a: f64,
    b: f64,
    x_id: &str,
    x: &PyJointDistributionArtifact,
    x_column: usize,
    y_id: &str,
    y: &PyJointDistributionArtifact,
    y_column: usize,
    cancel: Option<PyCancellationToken>,
) -> Refused<ClaimParts> {
    let op = tri!(operation(operation_name));
    let dx = match PropagationDraws::from_artifact_column(x_id, x.inner(), x_column) {
        Ok(v) => v,
        Err(r) => return (None, Some(refusal_json(&r))),
    };
    let dy = match PropagationDraws::from_artifact_column(y_id, y.inner(), y_column) {
        Ok(v) => v,
        Err(r) => return (None, Some(refusal_json(&r))),
    };
    let ctx = execution_context(0, None, cancel);
    finish(py, propagate_aligned_linear(op, (a, b), &dx, &dy, &ctx))
}

/// Linear combination from means and a covariance. `dependence` is
/// `declared_independent`, `covariance` (with `covariance_id`) or `unknown`.
#[pyfunction]
#[pyo3(signature = (operation_name, inputs_json, coefficients, covariance, dependence, covariance_id=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "the declaration is several separately named parts")]
fn joint_law_linear_moments(
    py: Python<'_>,
    operation_name: &str,
    inputs_json: &str,
    coefficients: Vec<f64>,
    covariance: Vec<f64>,
    dependence: &str,
    covariance_id: Option<String>,
    cancel: Option<PyCancellationToken>,
) -> Refused<ClaimParts> {
    let op = tri!(operation(operation_name));
    let rows: Vec<Json> =
        tri!(serde_json::from_str(inputs_json).map_err(|e| invalid_json(format!("inputs: {e}"))));
    let mut inputs = Vec::with_capacity(rows.len());
    for row in &rows {
        let text = |key: &str| row.get(key).and_then(Json::as_str).unwrap_or("").to_owned();
        let role_name = row.get("role").and_then(Json::as_str).unwrap_or("estimate");
        inputs.push(MomentInput {
            id: text("id"),
            evidence_id: text("evidence_id"),
            role: tri!(role(role_name)),
            mean: row.get("mean").and_then(Json::as_f64).unwrap_or(f64::NAN),
        });
    }
    let ctx = execution_context(0, None, cancel);
    let outcome = match dependence {
        "declared_independent" => propagate_linear_moments(
            op,
            &inputs,
            &coefficients,
            &covariance,
            &LinearDependence::DeclaredIndependent,
            &ctx,
        ),
        "covariance" => propagate_linear_moments(
            op,
            &inputs,
            &coefficients,
            &covariance,
            &LinearDependence::Covariance { id: covariance_id.unwrap_or_default() },
            &ctx,
        ),
        "unknown" => {
            let declared = PropagationRequest {
                operation: op,
                inputs: inputs
                    .iter()
                    .map(|m| PropagationInput {
                        id: m.id.clone(),
                        evidence_id: m.evidence_id.clone(),
                        role: m.role,
                        alignment: InputAlignment::MomentsOnly,
                    })
                    .collect(),
                dependence: PropagationDependence::Unknown,
                reading: RequestedReading::Point,
            };
            let means: Vec<f64> = inputs.iter().map(|m| m.mean).collect();
            propagate(
                &declared,
                PropagationRequestWire::LinearCombination {
                    coefficients: float_bits(&coefficients),
                    means: float_bits(&means),
                    covariance: float_bits(&covariance),
                },
                &ctx,
            )
        }
        other => return (None, Some(invalid_json(format!("unknown dependence `{other}`")))),
    };
    finish(py, outcome)
}

/// Mixture with known weights. `components_json`: `[{id, evidence_id, weight, mean, variance}]`.
#[pyfunction]
#[pyo3(signature = (components_json, cancel=None))]
fn joint_law_mixture(
    py: Python<'_>,
    components_json: &str,
    cancel: Option<PyCancellationToken>,
) -> Refused<ClaimParts> {
    let rows: Vec<Json> = tri!(
        serde_json::from_str(components_json).map_err(|e| invalid_json(format!("components: {e}")))
    );
    let number = |row: &Json, key: &str| row.get(key).and_then(Json::as_f64).unwrap_or(f64::NAN);
    let components: Vec<MixtureComponent> = rows
        .iter()
        .map(|row| MixtureComponent {
            id: row.get("id").and_then(Json::as_str).unwrap_or("").to_owned(),
            evidence_id: row.get("evidence_id").and_then(Json::as_str).unwrap_or("").to_owned(),
            weight: number(row, "weight"),
            mean: number(row, "mean"),
            variance: number(row, "variance"),
        })
        .collect();
    let ctx = execution_context(0, None, cancel);
    finish(py, propagate_mixture(&components, &ctx))
}

/// A mixture whose weights are not declared: always refused by the rule (the bridge for
/// the negative case, so Python never fabricates a refusal).
#[pyfunction]
fn joint_law_mixture_undeclared(
    py: Python<'_>,
    ids_json: &str,
    weights: Vec<f64>,
    means: Vec<f64>,
    variances: Vec<f64>,
) -> Refused<ClaimParts> {
    let ids: Vec<String> =
        tri!(serde_json::from_str(ids_json).map_err(|e| invalid_json(format!("ids: {e}"))));
    let declared = PropagationRequest {
        operation: PropagationOperation::Mixture,
        inputs: ids
            .iter()
            .map(|id| PropagationInput {
                id: id.clone(),
                evidence_id: format!("evidence:{id}"),
                role: InputRole::Estimate,
                alignment: InputAlignment::MomentsOnly,
            })
            .collect(),
        dependence: PropagationDependence::Unknown,
        reading: RequestedReading::Point,
    };
    let ctx = execution_context(0, None, None);
    finish(
        py,
        propagate(
            &declared,
            PropagationRequestWire::Mixture {
                weights: float_bits(&weights),
                means: float_bits(&means),
                variances: float_bits(&variances),
            },
            &ctx,
        ),
    )
}

fn query(text: &str) -> Result<EnvelopeQueryWire, String> {
    let value: Json =
        serde_json::from_str(text).map_err(|e| invalid_json(format!("query: {e}")))?;
    let number = |key: &str| {
        value
            .get(key)
            .and_then(Json::as_f64)
            .map(f64::to_bits)
            .ok_or_else(|| invalid_json(format!("query needs `{key}`")))
    };
    match value.get("kind").and_then(Json::as_str) {
        Some("sum_cdf") => Ok(EnvelopeQueryWire::SumCdf { threshold: number("threshold")? }),
        Some("difference_cdf") => {
            Ok(EnvelopeQueryWire::DifferenceCdf { threshold: number("threshold")? })
        }
        Some("product_moment") => Ok(EnvelopeQueryWire::ProductMoment),
        Some("covariance") => Ok(EnvelopeQueryWire::Covariance),
        Some("linear_variance") => {
            Ok(EnvelopeQueryWire::LinearVariance { a: number("a")?, b: number("b")? })
        }
        Some("functional") => Ok(EnvelopeQueryWire::Functional {
            functional: value
                .get("functional")
                .and_then(Json::as_str)
                .unwrap_or_default()
                .to_owned(),
        }),
        _ => Err(invalid_json("unknown envelope query kind".to_owned())),
    }
}

/// Structural envelope of a functional of two unpaired marginals (draw arrays).
#[pyfunction]
#[pyo3(signature = (operation_name, query_json, x_id, x_evidence, x_values, y_id, y_evidence, y_values, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "two named marginals plus the declaration")]
fn joint_law_envelope(
    py: Python<'_>,
    operation_name: &str,
    query_json: &str,
    x_id: &str,
    x_evidence: &str,
    x_values: Vec<f64>,
    y_id: &str,
    y_evidence: &str,
    y_values: Vec<f64>,
    cancel: Option<PyCancellationToken>,
) -> Refused<ClaimParts> {
    let op = tri!(operation(operation_name));
    let q = tri!(query(query_json));
    let x = PropagationDraws::independent_marginal(x_id, x_evidence, x_values);
    let y = PropagationDraws::independent_marginal(y_id, y_evidence, y_values);
    let ctx = execution_context(0, None, cancel);
    finish(py, propagate_dependence_envelope(op, q, &x, &y, &ctx))
}

/// Ranges over a declared correlation range.
#[pyfunction]
#[pyo3(signature = (operation_name, x_id, x_evidence, y_id, y_evidence, x_mean, x_sd, y_mean, y_sd, a, b, rho_lo, rho_hi, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "two summaries plus the declared range")]
fn joint_law_correlation_range(
    py: Python<'_>,
    operation_name: &str,
    x_id: &str,
    x_evidence: &str,
    y_id: &str,
    y_evidence: &str,
    x_mean: f64,
    x_sd: f64,
    y_mean: f64,
    y_sd: f64,
    a: f64,
    b: f64,
    rho_lo: f64,
    rho_hi: f64,
    cancel: Option<PyCancellationToken>,
) -> Refused<ClaimParts> {
    let op = tri!(operation(operation_name));
    let ctx = execution_context(0, None, cancel);
    finish(
        py,
        propagate_correlation_range(
            op,
            (x_id, x_evidence),
            (y_id, y_evidence),
            CorrelationRangeSummary { x_mean, x_sd, y_mean, y_sd, a, b },
            rho_lo,
            rho_hi,
            &ctx,
        ),
    )
}

/// Verify a sealed artifact and replay it to identical bits; returns the summary JSON.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity=None, cancel=None))]
fn joint_law_consume(
    artifact: &[u8],
    expected_identity: Option<&str>,
    cancel: Option<PyCancellationToken>,
) -> Refused<String> {
    let ctx = execution_context(0, None, cancel);
    match consume_joint_law_propagation_artifact(artifact, expected_identity, &ctx) {
        Ok(consumed) => {
            let values: Vec<Json> = consumed
                .result
                .iter()
                .filter(|(name, _)| name != "n_rows")
                .map(|(name, bits)| json!([name, f64::from_bits(*bits)]))
                .collect();
            let n_rows = consumed.result.iter().find(|(n, _)| n == "n_rows").map(|(_, v)| *v);
            (
                Some(
                    json!({
                        "identity": consumed.identity,
                        "operation": consumed.operation,
                        "route": consumed.route,
                        "label": consumed.label,
                        "is_envelope": consumed.is_envelope(),
                        "alignment_id": consumed.alignment_id,
                        "n_rows": n_rows,
                        "values": values,
                    })
                    .to_string(),
                ),
                None,
            )
        }
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    }
}

/// The refusal of an interval or posterior reading of a composed quantity (always
/// `cell_not_licensed`).
#[pyfunction]
fn joint_law_closed_reading(reading: &str) -> String {
    let Some(parsed) = RequestedReading::from_name(reading) else {
        return invalid_json(format!("unknown reading `{reading}`"));
    };
    let aligned = |id: &str| PropagationInput {
        id: id.to_owned(),
        evidence_id: format!("evidence:{id}"),
        role: InputRole::Estimate,
        alignment: InputAlignment::Aligned { alignment_id: "closed_reading".to_owned() },
    };
    let request = PropagationRequest {
        operation: PropagationOperation::Sum,
        inputs: vec![aligned("x"), aligned("y")],
        dependence: PropagationDependence::AlignedJointLaw,
        reading: parsed,
    };
    match antecedent_contract::check_propagation(&request) {
        Err(refusal) => refusal_json(&JointLawPropagationRefusal { refusals: vec![refusal] }),
        Ok(_) => invalid_json("the reading is not a closed reading".to_owned()),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(joint_law_column, module)?)?;
    module.add_function(wrap_pyfunction!(joint_law_aligned_functional, module)?)?;
    module.add_function(wrap_pyfunction!(joint_law_aligned_linear, module)?)?;
    module.add_function(wrap_pyfunction!(joint_law_linear_moments, module)?)?;
    module.add_function(wrap_pyfunction!(joint_law_mixture, module)?)?;
    module.add_function(wrap_pyfunction!(joint_law_mixture_undeclared, module)?)?;
    module.add_function(wrap_pyfunction!(joint_law_envelope, module)?)?;
    module.add_function(wrap_pyfunction!(joint_law_correlation_range, module)?)?;
    module.add_function(wrap_pyfunction!(joint_law_consume, module)?)?;
    module.add_function(wrap_pyfunction!(joint_law_closed_reading, module)?)?;
    Ok(())
}
