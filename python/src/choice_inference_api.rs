//! Python bridge for inference from repeated categorical choices.
//!
//! Python declares the retained choice study (frame, respondents, retained sources, looks and
//! stopping rule) as the wire declaration, and asks for one route at a time. The contract checks,
//! the evidence ledger's overlap judgement, the paradigm gate, every number, the exported
//! artifacts and every refusal are Rust's; a refusal comes back as structured JSON for the Python
//! layer to raise as a typed exception. Every result is point-only and calibration is unmeasured.

use antecedent::analysis::choice_inference::{
    ChoiceInferenceRefusal, ChoiceInferenceSession, ChoiceKernel, ChoiceRequest, StudyCandidate,
    consume_choice_inference_result_artifact,
};
use antecedent_io::choice_inference_result_artifact::{
    ChoiceDeclarationWire, ChoiceInferenceResultArtifact, ChoiceSpecWire, declaration_from_wire,
    spec_from_wire,
};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::{Value as Json, json};

use crate::adaptive_inference_api::status_json;
use crate::measurement_model_api::{list, member, object, parse, real, reals, small, text};
use crate::transport_common::execution_context;

type Refused<T> = (Option<T>, Option<String>);

fn refusal_value(refusal: &ChoiceInferenceRefusal) -> Json {
    json!({
        "code": refusal.code,
        "stage": "choice_inference",
        "detail": refusal.detail,
        "message": refusal.message,
        "offending": Json::Null,
        "remedy": Json::Null,
    })
}

fn refused<T>(refusal: &ChoiceInferenceRefusal) -> Refused<T> {
    (None, Some(refusal_value(refusal).to_string()))
}

fn request_of(value: &Json) -> PyResult<ChoiceRequest> {
    let wire: ChoiceSpecWire = serde_json::from_value(value.clone())
        .map_err(|error| crate::value_err(format!("invalid choice request: {error}")))?;
    spec_from_wire(&wire).map_err(|error| crate::value_err(error.to_string()))
}

fn kernel_of(value: &Json) -> PyResult<ChoiceKernel> {
    let map = object(value, "kernel")?;
    Ok(match text(map, "kind")?.as_str() {
        "fixed" => ChoiceKernel::Fixed { theta: reals(map, "shares")? },
        "prior_predictive" => ChoiceKernel::PriorPredictive { alpha: reals(map, "alpha")? },
        other => return Err(crate::value_err(format!("unknown kernel `{other}`"))),
    })
}

fn candidates_of(value: &Json) -> PyResult<Vec<StudyCandidate>> {
    let map = object(value, "ranking")?;
    list(map, "candidates")?
        .iter()
        .map(|c| {
            let m = object(c, "candidate")?;
            Ok(StudyCandidate {
                label: text(m, "label")?,
                observations: small(m, "observations")?,
                cost: real(m, "cost")?,
            })
        })
        .collect()
}

/// A prepared choice-study analysis.
#[pyclass(name = "ChoiceInferenceHandle", skip_from_py_object)]
pub(crate) struct ChoiceInferenceHandle {
    inner: ChoiceInferenceSession,
}

/// A consumed choice inference result.
#[pyclass(name = "ConsumedChoiceResultHandle", skip_from_py_object)]
pub(crate) struct ConsumedChoiceResultHandle {
    summary: String,
}

#[pymethods]
impl ConsumedChoiceResultHandle {
    /// Identity, procedure, route, method and values, as JSON.
    fn summary_json(&self) -> String {
        self.summary.clone()
    }
}

#[pymethods]
impl ChoiceInferenceHandle {
    /// Identity (hex) of the validated contract.
    fn contract_identity(&self) -> String {
        self.inner.contract_identity()
    }

    /// Procedure cell and its matrix status, as JSON.
    fn summary_json(&self) -> String {
        json!({
            "identity": self.inner.contract_identity(),
            "procedure": self.inner.contract().cell().identifier(),
            "status": status_json(&self.inner.cell_status()),
        })
        .to_string()
    }

    /// Run the route a request names.
    #[pyo3(signature = (request_json, cancel=None))]
    fn analyse(
        &self,
        request_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<String>> {
        let request = request_of(&parse(request_json, "choice request")?)?;
        let ctx = execution_context(0, None, cancel);
        Ok(match self.inner.analyse(&request, &ctx) {
            Ok(outcome) => (
                Some(
                    json!({
                        "paradigm": outcome.result.paradigm.as_str(),
                        "procedure": outcome.procedure.identifier(),
                        "method": outcome.result.method,
                        "values": outcome.result.values,
                        "closed_guarantee": outcome.result.closed_guarantee,
                        "contract_identity": outcome.contract_identity,
                        "look": outcome.request.look(),
                    })
                    .to_string(),
                ),
                None,
            ),
            Err(refusal) => refused(&refusal),
        })
    }

    /// Exact operating characteristics of the contract's sequential procedure under a kernel.
    #[pyo3(signature = (kernel_json, cancel=None))]
    fn operating_characteristics(
        &self,
        kernel_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<String>> {
        let kernel = kernel_of(&parse(kernel_json, "choice kernel")?)?;
        let ctx = execution_context(0, None, cancel);
        Ok(match self.inner.operating_characteristics(&kernel, &ctx) {
            Ok(oc) => (
                Some(
                    json!({
                        "selection": oc.selection,
                        "undecided": oc.undecided,
                        "expected_enrolled": oc.expected_enrolled,
                        "rule_stop": oc.rule_stop,
                        "horizon_probability": oc.horizon_probability,
                        "work_units": oc.work_units,
                        "statement": oc.statement,
                    })
                    .to_string(),
                ),
                None,
            ),
            Err(refusal) => refused(&refusal),
        })
    }

    /// Rank candidate studies by EVSI about the pooled posterior at a look.
    #[pyo3(signature = (ranking_json, cancel=None))]
    fn rank_studies(
        &self,
        ranking_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<String>> {
        let value = parse(ranking_json, "choice ranking")?;
        let map = object(&value, "ranking")?;
        let look = small(map, "look")? as usize;
        let values = reals(map, "values")?;
        let candidates = candidates_of(&value)?;
        let seed = member(map, "seed")?
            .as_u64()
            .ok_or_else(|| crate::value_err("`seed` must be a non-negative integer"))?;
        let ctx = execution_context(seed, None, cancel);
        Ok(match self.inner.rank_studies(look, &values, &candidates, seed, &ctx) {
            Ok(rows) => (
                Some(
                    Json::Array(
                        rows.iter()
                            .map(|r| {
                                json!({
                                    "label": r.candidate.label,
                                    "observations": r.candidate.observations,
                                    "cost": r.candidate.cost,
                                    "evsi": r.evsi,
                                    "standard_error": r.standard_error,
                                    "net_value": r.net_value,
                                    "rank_uncertain": r.rank_uncertain,
                                    "method": match r.method {
                                        antecedent::analysis::choice_inference::EvsiMethod::Exact { compositions } =>
                                            json!({"kind": "exact", "compositions": compositions}),
                                        antecedent::analysis::choice_inference::EvsiMethod::MonteCarlo { draws } =>
                                            json!({"kind": "monte_carlo", "draws": draws}),
                                    },
                                })
                            })
                            .collect(),
                    )
                    .to_string(),
                ),
                None,
            ),
            Err(refusal) => refused(&refusal),
        })
    }

    /// An interval for the whole procedure: refused.
    fn interval(&self) -> Option<String> {
        self.inner.interval().err().map(|r| refusal_value(&r).to_string())
    }

    /// The interval of one result of a measured choice cell.
    #[pyo3(signature = (request_json, cancel=None))]
    fn measured_interval(
        &self,
        request_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<(Option<crate::measured_inference_api::NativeMeasuredInference>, Option<String>)>
    {
        let request = request_of(&parse(request_json, "choice request")?)?;
        let ctx = execution_context(0, None, cancel);
        Ok(match self.inner.measured_interval(&request, "measured-choice-interval", &ctx) {
            Ok(measured) => (
                Some(crate::measured_inference_api::NativeMeasuredInference::from_io(measured)?),
                None,
            ),
            Err(refusal) => (None, Some(refusal_value(&refusal).to_string())),
        })
    }

    /// Export a result bound to its contract; returns the bytes and the artifact identity.
    #[pyo3(signature = (request_json, artifact_id, cancel=None))]
    fn export_result(
        &self,
        py: Python<'_>,
        request_json: &str,
        artifact_id: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Refused<(Py<PyBytes>, String)>> {
        let request = request_of(&parse(request_json, "choice request")?)?;
        let ctx = execution_context(0, None, cancel);
        Ok(match self.inner.export_result(&request, artifact_id, &ctx) {
            Ok(bytes) => {
                let identity = ChoiceInferenceResultArtifact::decode_unchecked(&bytes)
                    .map_err(|error| crate::serialization_err(error.to_string()))?
                    .identity;
                (Some((PyBytes::new(py, &bytes).unbind(), identity)), None)
            }
            Err(refusal) => refused(&refusal),
        })
    }
}

/// Validate a retained choice study and prepare the analysis.
#[pyfunction]
fn choice_inference_prepare(declaration_json: &str) -> PyResult<Refused<ChoiceInferenceHandle>> {
    let value = parse(declaration_json, "choice study")?;
    let wire: ChoiceDeclarationWire = serde_json::from_value(value)
        .map_err(|error| crate::value_err(format!("invalid choice study declaration: {error}")))?;
    let declaration = declaration_from_wire(&wire).map_err(|e| crate::value_err(e.to_string()))?;
    Ok(match ChoiceInferenceSession::prepare(declaration) {
        Ok(inner) => (Some(ChoiceInferenceHandle { inner }), None),
        Err(refusals) => {
            let mut first = refusals.first().map_or_else(|| json!({}), refusal_value);
            first["refusals"] = Json::Array(refusals.iter().map(refusal_value).collect());
            (None, Some(first.to_string()))
        }
    })
}

/// Consume a choice inference result artifact by replay and recomputation.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity=None, cancel=None))]
fn choice_inference_result_consume(
    artifact: &[u8],
    expected_identity: Option<&str>,
    cancel: Option<crate::PyCancellationToken>,
) -> Refused<ConsumedChoiceResultHandle> {
    let ctx = execution_context(0, None, cancel);
    match consume_choice_inference_result_artifact(artifact, expected_identity, &ctx) {
        Ok(consumed) => {
            let result = consumed.result();
            let summary = json!({
                "identity": consumed.identity(),
                "procedure": consumed.procedure().identifier(),
                "paradigm": result.paradigm.as_str(),
                "method": result.method,
                "values": result.values,
                "closed_guarantee": result.closed_guarantee,
                "contract_identity": consumed.contract().identity().to_hex(),
            });
            (Some(ConsumedChoiceResultHandle { summary: summary.to_string() }), None)
        }
        Err(refusal) => refused(&refusal),
    }
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<ChoiceInferenceHandle>()?;
    module.add_class::<ConsumedChoiceResultHandle>()?;
    module.add_function(wrap_pyfunction!(choice_inference_prepare, module)?)?;
    module.add_function(wrap_pyfunction!(choice_inference_result_consume, module)?)?;
    Ok(())
}
