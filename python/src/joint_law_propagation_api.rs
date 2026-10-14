//! Python bridge for composition-aware uncertainty.
//!
//! Python hands over draws (or artifacts), declarations and numbers; the propagation rule
//! (which dependence information a composition needs), the arithmetic, the artifact and every
//! refusal are Rust's. A refusal comes back as structured JSON for the Python layer to raise as
//! a typed exception. Nothing here pairs draws by index: rows are paired only under one
//! alignment identity, and a dependence envelope is never a point.

use antecedent::analysis::joint_law_propagation::{
    CorrelationRangeSummary, EnvelopeQueryWire, ExceedanceQuery, JointLawPropagationRefusal,
    LinearDependence, MixtureComponent, MixtureDraws, MomentInput, PropagationClaim,
    PropagationDraws, ValueFunctional, alignment_identity, claim_from_artifact,
    consume_joint_law_propagation_artifact, propagate, propagate_aligned_functional,
    propagate_aligned_linear, propagate_aligned_linear_spread, propagate_composed_exceedance,
    propagate_correlation_range, propagate_dependence_envelope, propagate_linear_moments,
    propagate_linear_moments_spread, propagate_mixture, propagate_mixture_exceedance,
    propagate_spread, propagate_transported_spread_aligned,
};
use antecedent_contract::{
    CoverageRecord, InputAlignment, InputRole, PropagationDependence, PropagationInput,
    PropagationOperation, PropagationRequest, RequestedReading,
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
        "verdict": claim.verdict(),
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
    linear_moments_call(
        py,
        RequestedReading::Point,
        operation_name,
        inputs_json,
        coefficients,
        covariance,
        dependence,
        covariance_id,
        cancel,
    )
}

/// The spread (variance, labelled `uncalibrated_spread`) of a linear combination from means
/// and a covariance. With `operation_name` `transported_composition` the inputs are the pieces
/// of a transported claim (roles `source_sampling`, `target_sampling`, `transport_functional`).
#[pyfunction]
#[pyo3(signature = (operation_name, inputs_json, coefficients, covariance, dependence, covariance_id=None, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "the declaration is several separately named parts")]
fn joint_law_linear_moments_spread(
    py: Python<'_>,
    operation_name: &str,
    inputs_json: &str,
    coefficients: Vec<f64>,
    covariance: Vec<f64>,
    dependence: &str,
    covariance_id: Option<String>,
    cancel: Option<PyCancellationToken>,
) -> Refused<ClaimParts> {
    linear_moments_call(
        py,
        RequestedReading::Spread,
        operation_name,
        inputs_json,
        coefficients,
        covariance,
        dependence,
        covariance_id,
        cancel,
    )
}

#[allow(clippy::too_many_arguments, reason = "the declaration is several separately named parts")]
fn linear_moments_call(
    py: Python<'_>,
    reading: RequestedReading,
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
    let spread = reading == RequestedReading::Spread;
    let ctx = execution_context(0, None, cancel);
    let run = |linear: &LinearDependence| {
        if spread {
            propagate_linear_moments_spread(op, &inputs, &coefficients, &covariance, linear, &ctx)
        } else {
            propagate_linear_moments(op, &inputs, &coefficients, &covariance, linear, &ctx)
        }
    };
    let outcome = match dependence {
        "declared_independent" => run(&LinearDependence::DeclaredIndependent),
        "covariance" => {
            run(&LinearDependence::Covariance { id: covariance_id.clone().unwrap_or_default() })
        }
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
                reading,
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
                .filter(|(name, _)| name != "n_rows" && name != "verdict_code")
                .map(|(name, bits)| json!([name, f64::from_bits(*bits)]))
                .collect();
            let n_rows = consumed.result.iter().find(|(n, _)| n == "n_rows").map(|(_, v)| *v);
            let verdict =
                consumed.result.iter().find(|(n, _)| n == "verdict_code").and_then(|(_, code)| {
                    match code {
                        0 => Some("act"),
                        1 => Some("withhold"),
                        2 => Some("unresolved"),
                        _ => None,
                    }
                });
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
                        "verdict": verdict,
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

/// Pull the two artifact columns of an aligned request.
fn two_columns(
    x_id: &str,
    x: &PyJointDistributionArtifact,
    x_column: usize,
    y_id: &str,
    y: &PyJointDistributionArtifact,
    y_column: usize,
) -> Result<(PropagationDraws, PropagationDraws), String> {
    let dx = PropagationDraws::from_artifact_column(x_id, x.inner(), x_column)
        .map_err(|r| refusal_json(&r))?;
    let dy = PropagationDraws::from_artifact_column(y_id, y.inner(), y_column)
        .map_err(|r| refusal_json(&r))?;
    Ok((dx, dy))
}

fn functional_by_name(name: &str) -> Result<ValueFunctional, String> {
    ValueFunctional::from_name(name)
        .ok_or_else(|| invalid_json(format!("unknown functional `{name}`")))
}

/// The spread of `g(X, Y)` under the aligned law of two artifact columns.
#[pyfunction]
#[pyo3(signature = (operation_name, x_id, x, x_column, y_id, y, y_column, functional, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "two named columns plus the declaration")]
fn joint_law_spread(
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
    let functional = tri!(functional_by_name(functional));
    let (dx, dy) = tri!(two_columns(x_id, x, x_column, y_id, y, y_column));
    let ctx = execution_context(0, None, cancel);
    finish(py, propagate_spread(op, &dx, &dy, functional, &ctx))
}

/// The spread of `a X + b Y` from two aligned artifact columns.
#[pyfunction]
#[pyo3(signature = (operation_name, a, b, x_id, x, x_column, y_id, y, y_column, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "two named columns plus the declaration")]
fn joint_law_aligned_linear_spread(
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
    let (dx, dy) = tri!(two_columns(x_id, x, x_column, y_id, y, y_column));
    let ctx = execution_context(0, None, cancel);
    finish(py, propagate_aligned_linear_spread(op, (a, b), &dx, &dy, &ctx))
}

/// `P(g(X, Y) > threshold)` from aligned artifact columns, with its Monte Carlo band and verdict.
#[pyfunction]
#[pyo3(signature = (operation_name, x_id, x, x_column, y_id, y, y_column, functional, threshold, cutoff, band_level, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "two named columns plus the declaration and query")]
fn joint_law_composed_exceedance(
    py: Python<'_>,
    operation_name: &str,
    x_id: &str,
    x: &PyJointDistributionArtifact,
    x_column: usize,
    y_id: &str,
    y: &PyJointDistributionArtifact,
    y_column: usize,
    functional: &str,
    threshold: f64,
    cutoff: f64,
    band_level: f64,
    cancel: Option<PyCancellationToken>,
) -> Refused<ClaimParts> {
    let op = tri!(operation(operation_name));
    let functional = tri!(functional_by_name(functional));
    let (dx, dy) = tri!(two_columns(x_id, x, x_column, y_id, y, y_column));
    let ctx = execution_context(0, None, cancel);
    finish(
        py,
        propagate_composed_exceedance(
            op,
            &dx,
            &dy,
            functional,
            ExceedanceQuery { threshold, cutoff, band_level },
            &ctx,
        ),
    )
}

/// `P(X > threshold)` of a known-weight mixture of draw sets. `components_json`:
/// `[{id, evidence_id, weight, draws}]`.
#[pyfunction]
#[pyo3(signature = (components_json, threshold, cancel=None))]
fn joint_law_mixture_exceedance(
    py: Python<'_>,
    components_json: &str,
    threshold: f64,
    cancel: Option<PyCancellationToken>,
) -> Refused<ClaimParts> {
    let rows: Vec<Json> = tri!(
        serde_json::from_str(components_json).map_err(|e| invalid_json(format!("components: {e}")))
    );
    let components: Vec<MixtureDraws> = rows
        .iter()
        .map(|row| MixtureDraws {
            id: row.get("id").and_then(Json::as_str).unwrap_or("").to_owned(),
            evidence_id: row.get("evidence_id").and_then(Json::as_str).unwrap_or("").to_owned(),
            weight: row.get("weight").and_then(Json::as_f64).unwrap_or(f64::NAN),
            draws: row
                .get("draws")
                .and_then(Json::as_array)
                .map(|a| a.iter().map(|v| v.as_f64().unwrap_or(f64::NAN)).collect())
                .unwrap_or_default(),
        })
        .collect();
    let ctx = execution_context(0, None, cancel);
    finish(py, propagate_mixture_exceedance(&components, threshold, &ctx))
}

/// The spread of a transported claim from the aligned draws of its three pieces (columns of
/// artifacts; one alignment identity for all three).
#[pyfunction]
#[pyo3(signature = (source_id, source, source_column, target_id, target, target_column, functional_id, functional, functional_column, coefficients, cancel=None))]
#[allow(clippy::too_many_arguments, reason = "three named columns plus coefficients")]
fn joint_law_transported_spread_aligned(
    py: Python<'_>,
    source_id: &str,
    source: &PyJointDistributionArtifact,
    source_column: usize,
    target_id: &str,
    target: &PyJointDistributionArtifact,
    target_column: usize,
    functional_id: &str,
    functional: &PyJointDistributionArtifact,
    functional_column: usize,
    coefficients: Vec<f64>,
    cancel: Option<PyCancellationToken>,
) -> Refused<ClaimParts> {
    let coefficients: [f64; 3] = tri!(
        coefficients
            .try_into()
            .map_err(|_| invalid_json("give one coefficient per piece (three)".to_owned()))
    );
    let (ds, dt) =
        tri!(two_columns(source_id, source, source_column, target_id, target, target_column));
    let df = match PropagationDraws::from_artifact_column(
        functional_id,
        functional.inner(),
        functional_column,
    ) {
        Ok(v) => v,
        Err(r) => return (None, Some(refusal_json(&r))),
    };
    let ctx = execution_context(0, None, cancel);
    finish(py, propagate_transported_spread_aligned(&ds, &dt, &df, coefficients, &ctx))
}

/// The interval reading of a sealed claim: the artifact is verified and replayed, then matched
/// against `record_json` (`{record_id, coordinate_id, construction, nominal_level, measured}`,
/// or `null` for no record). Returns the licence JSON or the typed refusal.
#[pyfunction]
#[pyo3(signature = (artifact, record_json, construction, cancel=None))]
fn joint_law_check_composed_interval(
    artifact: &[u8],
    record_json: Option<&str>,
    construction: &str,
    cancel: Option<PyCancellationToken>,
) -> Refused<String> {
    let record = match record_json {
        None => None,
        Some(text) => {
            let value: Json =
                tri!(serde_json::from_str(text).map_err(|e| invalid_json(format!("record: {e}"))));
            let string = |key: &str| value.get(key).and_then(Json::as_str).unwrap_or("").to_owned();
            Some(CoverageRecord {
                record_id: string("record_id"),
                coordinate_id: string("coordinate_id"),
                construction: string("construction"),
                nominal_level: value
                    .get("nominal_level")
                    .and_then(Json::as_f64)
                    .unwrap_or(f64::NAN),
                measured: value.get("measured").and_then(Json::as_bool).unwrap_or(false),
            })
        }
    };
    let ctx = execution_context(0, None, cancel);
    let claim = match claim_from_artifact(artifact, None, &ctx) {
        Ok(claim) => claim,
        Err(refusal) => return (None, Some(refusal_json(&refusal))),
    };
    match claim.check_composed_interval(record.as_ref(), construction) {
        Ok(licence) => (
            Some(
                json!({
                    "coordinate_id": licence.coordinate_id(),
                    "record_id": licence.record_id(),
                    "construction": licence.construction(),
                    "nominal_level": licence.nominal_level(),
                    "label": licence.label(),
                    "lower": licence.bounds().map(|b| b.0),
                    "upper": licence.bounds().map(|b| b.1),
                })
                .to_string(),
            ),
            None,
        ),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
    }
}

/// The composed coordinate a coverage record for a sealed claim must be measured at.
#[pyfunction]
fn joint_law_coordinate_id(artifact: &[u8]) -> Refused<String> {
    let ctx = execution_context(0, None, None);
    match claim_from_artifact(artifact, None, &ctx).and_then(|claim| claim.coordinate_id()) {
        Ok(id) => (Some(id), None),
        Err(refusal) => (None, Some(refusal_json(&refusal))),
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
    module.add_function(wrap_pyfunction!(joint_law_linear_moments_spread, module)?)?;
    module.add_function(wrap_pyfunction!(joint_law_spread, module)?)?;
    module.add_function(wrap_pyfunction!(joint_law_aligned_linear_spread, module)?)?;
    module.add_function(wrap_pyfunction!(joint_law_composed_exceedance, module)?)?;
    module.add_function(wrap_pyfunction!(joint_law_mixture_exceedance, module)?)?;
    module.add_function(wrap_pyfunction!(joint_law_transported_spread_aligned, module)?)?;
    module.add_function(wrap_pyfunction!(joint_law_check_composed_interval, module)?)?;
    module.add_function(wrap_pyfunction!(joint_law_coordinate_id, module)?)?;
    Ok(())
}
