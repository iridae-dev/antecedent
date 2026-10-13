//! Python bridge for observation-conditioned scientific queries.
//!
//! Python states the request (the outcome, recorded values, interventions, an
//! optional measured indicator value with its measurement model, or a declared
//! Bayes model); every check, probability, identity, artifact and refusal is
//! Rust's. A refusal comes back as structured JSON for the Python layer to
//! raise as a typed exception. A conditioning claim and an interventional claim
//! are different queries here exactly as they are in Rust.

use antecedent::analysis::observation_evidence::{
    required_bayes_factor, required_observations_binary_signal, required_observations_conjugate,
};
use antecedent::analysis::observation_posterior::{
    FiniteBayesInput, conjugate_posterior, finite_posterior,
};
use antecedent::analysis::observation_query::{
    Direction, ObservationClaim, ObservationQueryRefusal, ObservationRequest,
    ObservationResultWire, consume_observation_query_artifact,
};
use antecedent_contract::{
    ExternalRefusal, InferenceBasis, MeasuredIndicator, ObservationOperation, ObservedSlot,
    SlotProvenance, SlotValue, calibration_refusal, interval_refusal,
};
use antecedent_core::measurement_model::{MeasurementKind, RepresentationVersion};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::{Map, Value as Json, json};

use crate::measurement_model_api::{
    list, member, object, opt_text, parse, parse_model, present, real, small, text,
};
use crate::mechanism_composition_api::PreparedCompositionHandle;
use crate::transport_common::execution_context;

type Refused<T> = (Option<T>, Option<String>);
type Claimed = (String, Py<PyBytes>);

fn one_refusal(refusal: &ExternalRefusal) -> Json {
    json!({
        "code": refusal.code,
        "stage": refusal.stage,
        "detail": refusal.detail,
        "message": "",
        "offending": refusal.offending,
        "expected": refusal.expected,
        "supplied": refusal.supplied,
        "remedy": refusal.remedy,
    })
}

fn refusals_json(code: &str, refusals: &[ExternalRefusal]) -> String {
    let all: Vec<Json> = refusals.iter().map(one_refusal).collect();
    let mut first = all.first().cloned().unwrap_or_else(|| {
        json!({
            "code": code,
            "stage": "observe",
            "detail": "observation_query.request_unsupported",
            "message": "observation query refused",
            "offending": Json::Null,
            "expected": Json::Null,
            "supplied": Json::Null,
            "remedy": Json::Null,
        })
    });
    first["refusals"] = Json::Array(all);
    first.to_string()
}

fn refused<T>(refusal: &ObservationQueryRefusal) -> Refused<T> {
    (None, Some(refusals_json(refusal.code(), &refusal.refusals)))
}

fn bits_levels(values: &[(String, u64)]) -> Json {
    Json::Array(values.iter().map(|(l, p)| json!([l, f64::from_bits(*p)])).collect())
}

fn result_json(result: &ObservationResultWire) -> Json {
    match result {
        ObservationResultWire::Conditional {
            operation,
            evidence_probability_bits,
            levels,
            construct_posterior,
        } => json!({
            "kind": "conditional",
            "operation": operation,
            "evidence_probability": f64::from_bits(*evidence_probability_bits),
            "levels": bits_levels(levels),
            "construct_posterior": construct_posterior.as_deref().map(bits_levels),
        }),
        ObservationResultWire::Interventional { operation, levels } => json!({
            "kind": "interventional",
            "operation": operation,
            "levels": bits_levels(levels),
        }),
        ObservationResultWire::Paired {
            conditioning_evidence_bits,
            conditioning,
            intervention,
            max_abs_difference_bits,
        } => json!({
            "kind": "paired",
            "conditioning_evidence": f64::from_bits(*conditioning_evidence_bits),
            "conditioning": bits_levels(conditioning),
            "intervention": bits_levels(intervention),
            "max_abs_difference": f64::from_bits(*max_abs_difference_bits),
        }),
        ObservationResultWire::ThresholdCrossing { crossing_probability_bits, rows } => json!({
            "kind": "threshold_crossing",
            "crossing_probability": f64::from_bits(*crossing_probability_bits),
            "rows": rows.iter().map(|(l, p, c, x)| json!({
                "level": l,
                "probability": f64::from_bits(*p),
                "conditional": f64::from_bits(*c),
                "crosses": x,
            })).collect::<Vec<_>>(),
        }),
        ObservationResultWire::FinitePosterior {
            posterior,
            log_evidence_bits,
            evidence_probability_bits,
            predictive,
        } => json!({
            "kind": "finite_posterior",
            "posterior": bits_levels(posterior),
            "log_evidence": f64::from_bits(*log_evidence_bits),
            "evidence_probability": f64::from_bits(*evidence_probability_bits),
            "predictive": bits_levels(predictive),
        }),
        ObservationResultWire::ConjugatePosterior {
            alpha_bits,
            beta_bits,
            predictive_success_bits,
        } => json!({
            "kind": "conjugate_posterior",
            "alpha": f64::from_bits(*alpha_bits),
            "beta": f64::from_bits(*beta_bits),
            "predictive_success": f64::from_bits(*predictive_success_bits),
        }),
        ObservationResultWire::RequiredBayesFactor { factor_bits, already_satisfied } => json!({
            "kind": "required_bayes_factor",
            "factor": f64::from_bits(*factor_bits),
            "already_satisfied": already_satisfied,
        }),
        ObservationResultWire::RequiredObservations {
            observations,
            value_bits,
            already_satisfied,
            bayes_factor_bits,
        } => json!({
            "kind": "required_observations",
            "observations": observations,
            "value": f64::from_bits(*value_bits),
            "already_satisfied": already_satisfied,
            "bayes_factor": bayes_factor_bits.map(f64::from_bits),
        }),
    }
}

fn claim_payload(py: Python<'_>, claim: &ObservationClaim) -> Refused<Claimed> {
    match claim.export() {
        Ok(bytes) => (
            Some((
                json!({
                    "identity": claim.identity(),
                    "operation": claim.operation(),
                    "inference_basis": claim.inference_basis(),
                    "result": result_json(claim.result()),
                })
                .to_string(),
                PyBytes::new(py, &bytes).unbind(),
            )),
            None,
        ),
        Err(refusal) => refused(&refusal),
    }
}

fn claimed(
    py: Python<'_>,
    outcome: Result<ObservationClaim, ObservationQueryRefusal>,
) -> Refused<Claimed> {
    match outcome {
        Ok(claim) => claim_payload(py, &claim),
        Err(refusal) => refused(&refusal),
    }
}

// ---------------------------------------------------------------- parsing

fn parse_basis(name: &str) -> PyResult<InferenceBasis> {
    match name {
        "declared_model" => Ok(InferenceBasis::DeclaredModel),
        "sample_fitted" => Ok(InferenceBasis::SampleFittedPointOnly),
        other => Err(crate::value_err(format!("unknown inference basis `{other}`"))),
    }
}

fn parse_direction(name: &str) -> PyResult<Direction> {
    Direction::from_name(name)
        .ok_or_else(|| crate::value_err(format!("unknown direction `{name}`")))
}

fn parse_slot(value: &Json) -> PyResult<ObservedSlot> {
    let map = object(value, "conditioning slot")?;
    let provenance = match text(map, "provenance")?.as_str() {
        "recorded" => SlotProvenance::Recorded,
        "assigned" => SlotProvenance::Assigned,
        "hypothetical" => SlotProvenance::Hypothetical,
        other => return Err(crate::value_err(format!("unknown slot provenance `{other}`"))),
    };
    let value = match present(map, "measured") {
        None => SlotValue::Direct,
        Some(measured) => {
            let m = object(measured, "measured indicator")?;
            let kind_name = text(m, "kind")?;
            let kind = MeasurementKind::from_name(&kind_name).ok_or_else(|| {
                crate::value_err(format!("unknown measurement kind `{kind_name}`"))
            })?;
            let model = match present(m, "model") {
                None => None,
                Some(model) => Some(parse_model(model)?),
            };
            let representation = object(member(m, "representation")?, "representation")?;
            SlotValue::Measured(MeasuredIndicator {
                indicator_id: text(m, "indicator_id")?,
                kind,
                model,
                positive_level: opt_text(m, "positive_level")?,
                representation: RepresentationVersion {
                    coordinate_system_id: text(representation, "coordinate_system_id")?,
                    version: small(representation, "version")?,
                },
            })
        }
    };
    Ok(ObservedSlot {
        variable_id: text(map, "variable")?,
        level: text(map, "level")?,
        provenance,
        value,
    })
}

fn parse_slots(map: &Map<String, Json>, key: &str) -> PyResult<Vec<ObservedSlot>> {
    list(map, key)?.iter().map(parse_slot).collect()
}

fn parse_pairs(map: &Map<String, Json>, key: &str) -> PyResult<Vec<(String, String)>> {
    list(map, key)?
        .iter()
        .map(|entry| {
            let pair = entry.as_array().filter(|p| p.len() == 2 && p.iter().all(Json::is_string));
            pair.map(|p| {
                (
                    p[0].as_str().unwrap_or_default().to_owned(),
                    p[1].as_str().unwrap_or_default().to_owned(),
                )
            })
            .ok_or_else(|| crate::value_err(format!("`{key}` must hold [name, level] pairs")))
        })
        .collect()
}

fn parse_request(value: &Json) -> PyResult<ObservationRequest> {
    let map = object(value, "observation request")?;
    let operation = match text(map, "operation")?.as_str() {
        "conditioning" => ObservationOperation::Conditioning,
        "intervention" => ObservationOperation::Intervention,
        other => return Err(crate::value_err(format!("unknown operation `{other}`"))),
    };
    Ok(ObservationRequest {
        operation,
        outcome: text(map, "outcome")?,
        conditioning: parse_slots(map, "conditioning")?,
        interventions: parse_pairs(map, "interventions")?,
    })
}

// ---------------------------------------------------------------- entry points

/// Answer one conditioning or interventional query on a prepared composition.
#[pyfunction]
#[pyo3(signature = (handle, request_json, *, cancel=None))]
fn observation_query_observe(
    py: Python<'_>,
    handle: PyRef<'_, PreparedCompositionHandle>,
    request_json: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused<Claimed>> {
    let request = parse_request(&parse(request_json, "observation request")?)?;
    let inner = handle.inner.clone();
    let outcome = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        Ok(inner.observe(&request, &ctx))
    })?;
    Ok(claimed(py, outcome))
}

/// Both laws of one assignment from one model.
#[pyfunction]
#[pyo3(signature = (handle, outcome, assignment_json, *, cancel=None))]
fn observation_query_pair(
    py: Python<'_>,
    handle: PyRef<'_, PreparedCompositionHandle>,
    outcome: &str,
    assignment_json: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused<Claimed>> {
    let wrapped = json!({ "assignment": parse(assignment_json, "assignment")? });
    let assignment = parse_pairs(object(&wrapped, "assignment")?, "assignment")?;
    let inner = handle.inner.clone();
    let outcome = outcome.to_owned();
    let result = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        Ok(inner.condition_versus_intervene(&outcome, &assignment, &ctx))
    })?;
    Ok(claimed(py, result))
}

/// Which observed levels of a signal cross a threshold.
#[pyfunction]
#[pyo3(signature = (handle, request_json, *, cancel=None))]
fn observation_query_threshold(
    py: Python<'_>,
    handle: PyRef<'_, PreparedCompositionHandle>,
    request_json: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused<Claimed>> {
    let value = parse(request_json, "threshold request")?;
    let map = object(&value, "threshold request")?;
    let outcome = text(map, "outcome")?;
    let outcome_level = text(map, "outcome_level")?;
    let signal = text(map, "signal")?;
    let background = parse_slots(map, "background")?;
    let threshold = real(map, "threshold")?;
    let direction = parse_direction(&text(map, "direction")?)?;
    let inner = handle.inner.clone();
    let result = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        Ok(inner.threshold_crossing(
            &outcome,
            &outcome_level,
            &signal,
            &background,
            threshold,
            direction,
            &ctx,
        ))
    })?;
    Ok(claimed(py, result))
}

/// Posterior of a declared finite hypothesis space.
#[pyfunction]
#[pyo3(signature = (model_json, observations, basis, *, cancel=None))]
fn observation_query_finite_posterior(
    py: Python<'_>,
    model_json: &str,
    observations: Vec<String>,
    basis: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused<Claimed>> {
    let value = parse(model_json, "bayes model")?;
    let map = object(&value, "bayes model")?;
    let model = FiniteBayesInput {
        hypotheses: crate::measurement_model_api::texts(map, "hypotheses")?,
        prior: crate::measurement_model_api::reals(map, "prior")?,
        observation_levels: crate::measurement_model_api::texts(map, "observation_levels")?,
        likelihood: crate::measurement_model_api::reals(map, "likelihood")?,
    };
    let basis = parse_basis(basis)?;
    let result = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        Ok(finite_posterior(&model, &observations, basis, &ctx))
    })?;
    Ok(claimed(py, result))
}

/// Beta-Bernoulli posterior and predictive success probability.
#[pyfunction]
#[pyo3(signature = (alpha, beta, successes, failures, basis, *, cancel=None))]
fn observation_query_conjugate_posterior(
    py: Python<'_>,
    alpha: f64,
    beta: f64,
    successes: u64,
    failures: u64,
    basis: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused<Claimed>> {
    let basis = parse_basis(basis)?;
    let result = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        Ok(conjugate_posterior(alpha, beta, successes, failures, basis, &ctx))
    })?;
    Ok(claimed(py, result))
}

/// The Bayes factor that moves a probability to a level.
#[pyfunction]
#[pyo3(signature = (prior, level, direction, *, cancel=None))]
fn observation_query_required_bayes_factor(
    py: Python<'_>,
    prior: f64,
    level: f64,
    direction: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused<Claimed>> {
    let direction = parse_direction(direction)?;
    let result = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        Ok(required_bayes_factor(prior, level, direction, &ctx))
    })?;
    Ok(claimed(py, result))
}

/// The fewest readings of a binary signal that reach a level.
#[pyfunction]
#[pyo3(signature = (prior, level, direction, rate_given_hypothesis, rate_given_complement, max_observations, basis, *, cancel=None))]
#[allow(
    clippy::too_many_arguments,
    reason = "the Python signature is the declared observation-query argument list, one parameter per declared quantity"
)]
fn observation_query_required_observations_binary(
    py: Python<'_>,
    prior: f64,
    level: f64,
    direction: &str,
    rate_given_hypothesis: f64,
    rate_given_complement: f64,
    max_observations: u64,
    basis: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused<Claimed>> {
    let direction = parse_direction(direction)?;
    let basis = parse_basis(basis)?;
    let result = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        Ok(required_observations_binary_signal(
            prior,
            level,
            direction,
            rate_given_hypothesis,
            rate_given_complement,
            max_observations,
            basis,
            &ctx,
        ))
    })?;
    Ok(claimed(py, result))
}

/// The fewest consecutive successes or failures that reach a predictive level.
#[pyfunction]
#[pyo3(signature = (alpha, beta, level, direction, max_observations, basis, *, cancel=None))]
#[allow(
    clippy::too_many_arguments,
    reason = "the Python signature is the declared observation-query argument list, one parameter per declared quantity"
)]
fn observation_query_required_observations_conjugate(
    py: Python<'_>,
    alpha: f64,
    beta: f64,
    level: f64,
    direction: &str,
    max_observations: u64,
    basis: &str,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused<Claimed>> {
    let direction = parse_direction(direction)?;
    let basis = parse_basis(basis)?;
    let result = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        Ok(required_observations_conjugate(
            alpha,
            beta,
            level,
            direction,
            max_observations,
            basis,
            &ctx,
        ))
    })?;
    Ok(claimed(py, result))
}

/// The closed routes: `interval` or `calibrated_probability`. Always the
/// typed refusal.
#[pyfunction]
fn observation_query_closed(route: &str) -> PyResult<String> {
    let refusal = match route {
        "interval" => interval_refusal(),
        "calibrated_probability" => calibration_refusal(),
        other => return Err(crate::value_err(format!("unknown closed route `{other}`"))),
    };
    Ok(refusals_json(refusal.code, &[refusal]))
}

/// Independently consume an observation-query artifact.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity=None, *, cancel=None))]
fn observation_query_consume(
    py: Python<'_>,
    artifact: &[u8],
    expected_identity: Option<&str>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<Refused<String>> {
    let bytes = artifact.to_vec();
    let expected = expected_identity.map(str::to_owned);
    let outcome = crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        Ok(consume_observation_query_artifact(&bytes, expected.as_deref(), &ctx))
    })?;
    Ok(match outcome {
        Ok(consumed) => (
            Some(
                json!({
                    "identity": consumed.identity,
                    "operation": consumed.operation,
                    "inference_basis": consumed.inference_basis,
                    "result": result_json(&consumed.result),
                })
                .to_string(),
            ),
            None,
        ),
        Err(refusal) => refused(&refusal),
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_function(wrap_pyfunction!(observation_query_observe, module)?)?;
    module.add_function(wrap_pyfunction!(observation_query_pair, module)?)?;
    module.add_function(wrap_pyfunction!(observation_query_threshold, module)?)?;
    module.add_function(wrap_pyfunction!(observation_query_finite_posterior, module)?)?;
    module.add_function(wrap_pyfunction!(observation_query_conjugate_posterior, module)?)?;
    module.add_function(wrap_pyfunction!(observation_query_required_bayes_factor, module)?)?;
    module
        .add_function(wrap_pyfunction!(observation_query_required_observations_binary, module)?)?;
    module.add_function(wrap_pyfunction!(
        observation_query_required_observations_conjugate,
        module
    )?)?;
    module.add_function(wrap_pyfunction!(observation_query_closed, module)?)?;
    module.add_function(wrap_pyfunction!(observation_query_consume, module)?)?;
    Ok(())
}
