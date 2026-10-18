//! Bounded Python bridge for nonlinear continuous-mediator mediation.
//!
//! Python supplies the analysis columns, the declared premises and the configuration; the
//! estimand check, the premise refusals, the Gauss-Hermite integration with its `n` versus `2n`
//! error, the seeded bootstrap, the identity digests, the artifact and every refusal are Rust's.
//! A refusal comes back as structured JSON for the Python layer to raise as its own exception
//! type; corruption and unknown versions raise `CausalSerializationError`.

use crate::bridge::{artifact_refusal_json, check_declared_size};
use antecedent::analysis::nonlinear_mediation::{
    MediationConfigWire, MediationDataWire, MediationPremisesWire, NonlinearMediation,
    NonlinearMediationArtifactError, NonlinearMediationIdentity, NonlinearMediationRequestWire,
};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::{detach_catch, value_err, with_reason_code};

/// Largest premises / configuration / identity JSON accepted.
const MAX_JSON_BYTES: usize = crate::bridge::MAX_CONFIG_DECLARATION_BYTES;

type EvaluatePayload = (Option<String>, Option<Vec<u8>>, Option<String>);
type ConsumePayload = (Option<String>, Option<String>);

fn refusal_json(error: &NonlinearMediationArtifactError, stage: &'static str) -> Option<String> {
    let offending = match error {
        NonlinearMediationArtifactError::IdentityMismatch { field } => Some(*field),
        _ => None,
    };
    Some(artifact_refusal_json(stage, error.refusal()?, offending))
}

/// A non-refusal failure: a bound is a value error, everything else is a serialization error.
fn failure(error: &NonlinearMediationArtifactError) -> PyErr {
    match error {
        NonlinearMediationArtifactError::LimitsExceeded(what) => with_reason_code(
            value_err(format!("nonlinear_mediation.limits_exceeded: {what}")),
            antecedent_core::reason_code!("invalid_argument"),
        ),
        other => crate::serialization_err(other.to_string()),
    }
}

fn check_size(text: &str) -> PyResult<()> {
    check_declared_size(text, MAX_JSON_BYTES, "nonlinear mediation")
}

fn parse<T: serde::de::DeserializeOwned>(text: &str, what: &str) -> PyResult<T> {
    check_size(text)?;
    serde_json::from_str(text).map_err(|e| {
        with_reason_code(
            value_err(format!("nonlinear_mediation.invalid_request: {what}: {e}")),
            antecedent_core::reason_code!("invalid_argument"),
        )
    })
}

/// The dataset a digest-bound artifact is consumed against: treatment, mediator, outcome,
/// covariate names and covariate columns.
type MediationColumns<'py> = (
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    Vec<String>,
    Vec<PyReadonlyArray1<'py, f64>>,
);

/// Estimate the natural effects on declared columns:
/// `(report json, artifact bytes, refusal json)`; an export over its byte budget returns the report
/// and the refusal with no artifact.
#[pyfunction]
#[pyo3(signature = (
    treatment,
    mediator,
    outcome,
    covariates,
    covariate_names,
    premises_json,
    config_json,
    artifact_id,
    *,
    digest_bound=false,
))]
#[allow(
    clippy::too_many_arguments,
    reason = "the declared columns and declarations are the request"
)]
fn evaluate_nonlinear_mediation(
    py: Python<'_>,
    treatment: PyReadonlyArray1<'_, f64>,
    mediator: PyReadonlyArray1<'_, f64>,
    outcome: PyReadonlyArray1<'_, f64>,
    covariates: Vec<PyReadonlyArray1<'_, f64>>,
    covariate_names: Vec<String>,
    premises_json: &str,
    config_json: &str,
    artifact_id: &str,
    digest_bound: bool,
) -> PyResult<EvaluatePayload> {
    let premises: MediationPremisesWire = parse(premises_json, "premises")?;
    let config: MediationConfigWire = parse(config_json, "config")?;
    let request = NonlinearMediationRequestWire {
        premises,
        config,
        data: MediationDataWire {
            treatment: treatment.as_array().to_vec(),
            mediator: mediator.as_array().to_vec(),
            outcome: outcome.as_array().to_vec(),
            covariate_names,
            covariates: covariates.iter().map(|c| c.as_array().to_vec()).collect(),
        },
    };
    let artifact_id = artifact_id.to_owned();
    detach_catch(py, move || match NonlinearMediation::estimate(&request) {
        Ok(done) => {
            let report = serde_json::to_string(done.report())
                .map_err(|e| crate::serialization_err(e.to_string()))?;
            // An export over its byte budget refuses on its own: the fit is complete and its
            // report returns with the refusal in place of the artifact.
            let exported = if digest_bound {
                done.export_digest_bound(&artifact_id)
            } else {
                done.export(&artifact_id)
            };
            match exported {
                Ok(bytes) => Ok((Some(report), Some(bytes), None)),
                Err(error) => match refusal_json(&error, "export") {
                    Some(json) => Ok((Some(report), None, Some(json))),
                    None => Err(failure(&error)),
                },
            }
        }
        Err(error) => match refusal_json(&error, "estimate") {
            Some(json) => Ok((None, None, Some(json))),
            None => Err(failure(&error)),
        },
    })
}

/// Consume an artifact by re-estimation: `(report json, refusal json)`.
///
/// `expected_identity_json` is an identity the caller retained independently; a changed
/// premise, configuration, dataset or result is refused against it.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity_json=None, *, memory_limit_bytes=None, data=None))]
fn consume_nonlinear_mediation_artifact(
    py: Python<'_>,
    artifact: Vec<u8>,
    expected_identity_json: Option<String>,
    memory_limit_bytes: Option<u64>,
    data: Option<MediationColumns<'_>>,
) -> PyResult<ConsumePayload> {
    let expected: Option<NonlinearMediationIdentity> = match expected_identity_json {
        Some(text) => Some(parse(&text, "identity")?),
        None => None,
    };
    let data =
        data.map(|(treatment, mediator, outcome, covariate_names, covariates)| MediationDataWire {
            treatment: treatment.as_array().to_vec(),
            mediator: mediator.as_array().to_vec(),
            outcome: outcome.as_array().to_vec(),
            covariate_names,
            covariates: covariates.iter().map(|c| c.as_array().to_vec()).collect(),
        });
    let ctx = crate::measured_inference_api::context(0, memory_limit_bytes, None);
    detach_catch(py, move || {
        match NonlinearMediation::consume_with(&artifact, expected.as_ref(), data.as_ref(), &ctx) {
            Ok(done) => {
                let report = serde_json::to_string(done.report())
                    .map_err(|e| crate::serialization_err(e.to_string()))?;
                Ok((Some(report), None))
            }
            Err(error) => match refusal_json(&error, "consume") {
                Some(json) => Ok((None, Some(json))),
                None => Err(failure(&error)),
            },
        }
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(evaluate_nonlinear_mediation, m)?)?;
    m.add_function(wrap_pyfunction!(consume_nonlinear_mediation_artifact, m)?)
}
