//! Bounded Python bridge for latent-class (finite mixture) regime effects.
//!
//! Python supplies the analysis columns and the declared configuration (class count, seed,
//! randomization declaration); the EM fit, the canonical class order, the label-aligned
//! bootstrap, the identity digests, the artifact and every refusal are Rust's. A refusal comes
//! back as structured JSON for the Python layer to raise as its own exception type; corruption
//! and unknown versions raise `CausalSerializationError`.

use crate::bridge::{artifact_refusal_json, check_declared_size};
use antecedent::analysis::latent_class_effects::{
    LatentClassArtifactError, LatentClassConfigWire, LatentClassDataWire, LatentClassEffects,
    LatentClassIdentity, LatentClassRequestWire,
};
use numpy::PyReadonlyArray1;
use pyo3::prelude::*;

use crate::{detach_catch, value_err, with_reason_code};

/// Largest configuration / identity JSON accepted.
const MAX_JSON_BYTES: usize = crate::bridge::MAX_CONFIG_DECLARATION_BYTES;

type EvaluatePayload = (Option<String>, Option<Vec<u8>>, Option<String>);
type ConsumePayload = (Option<String>, Option<String>);

fn refusal_json(error: &LatentClassArtifactError, stage: &'static str) -> Option<String> {
    let offending = match error {
        LatentClassArtifactError::IdentityMismatch { field } => Some(*field),
        _ => None,
    };
    Some(artifact_refusal_json(stage, error.refusal()?, offending))
}

/// A non-refusal failure: a bound is a value error, everything else is a serialization error.
fn failure(error: &LatentClassArtifactError) -> PyErr {
    match error {
        LatentClassArtifactError::LimitsExceeded(what) => with_reason_code(
            value_err(format!("latent_class.limits_exceeded: {what}")),
            antecedent_core::reason_code!("invalid_argument"),
        ),
        other => crate::serialization_err(other.to_string()),
    }
}

fn parse<T: serde::de::DeserializeOwned>(text: &str, what: &str) -> PyResult<T> {
    check_declared_size(text, MAX_JSON_BYTES, "latent class")?;
    serde_json::from_str(text).map_err(|e| {
        with_reason_code(
            value_err(format!("latent_class.invalid_request: {what}: {e}")),
            antecedent_core::reason_code!("invalid_argument"),
        )
    })
}

/// The dataset a digest-bound artifact is consumed against: outcome, treatment, covariate names
/// and covariate columns.
type LatentClassColumns<'py> = (
    PyReadonlyArray1<'py, f64>,
    PyReadonlyArray1<'py, f64>,
    Vec<String>,
    Vec<PyReadonlyArray1<'py, f64>>,
);

/// Fit the mixture on declared columns: `(report json, artifact bytes, refusal json)`; an export
/// over its byte budget returns the report and the refusal with no artifact.
#[pyfunction]
#[pyo3(signature = (outcome, treatment, covariates, covariate_names, config_json, artifact_id, *, digest_bound=false))]
fn evaluate_latent_class_effects(
    py: Python<'_>,
    outcome: PyReadonlyArray1<'_, f64>,
    treatment: PyReadonlyArray1<'_, f64>,
    covariates: Vec<PyReadonlyArray1<'_, f64>>,
    covariate_names: Vec<String>,
    config_json: &str,
    artifact_id: &str,
    digest_bound: bool,
) -> PyResult<EvaluatePayload> {
    let config: LatentClassConfigWire = parse(config_json, "config")?;
    let request = LatentClassRequestWire {
        config,
        data: LatentClassDataWire {
            outcome: outcome.as_array().to_vec(),
            treatment: treatment.as_array().to_vec(),
            covariate_names,
            covariates: covariates.iter().map(|c| c.as_array().to_vec()).collect(),
        },
    };
    let artifact_id = artifact_id.to_owned();
    detach_catch(py, move || match LatentClassEffects::fit(&request) {
        Ok(done) => {
            let report = serde_json::to_string(&done.report())
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
        Err(error) => match refusal_json(&error, "fit") {
            Some(json) => Ok((None, None, Some(json))),
            None => Err(failure(&error)),
        },
    })
}

/// Consume an artifact by refitting: `(report json, refusal json)`.
///
/// `expected_identity_json` is an identity the caller retained independently; a changed
/// configuration, dataset or result is refused against it.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity_json=None, *, memory_limit_bytes=None, data=None))]
fn consume_latent_class_artifact(
    py: Python<'_>,
    artifact: Vec<u8>,
    expected_identity_json: Option<String>,
    memory_limit_bytes: Option<u64>,
    data: Option<LatentClassColumns<'_>>,
) -> PyResult<ConsumePayload> {
    let expected: Option<LatentClassIdentity> = match expected_identity_json {
        Some(text) => Some(parse(&text, "identity")?),
        None => None,
    };
    let data = data.map(|(outcome, treatment, covariate_names, covariates)| LatentClassDataWire {
        outcome: outcome.as_array().to_vec(),
        treatment: treatment.as_array().to_vec(),
        covariate_names,
        covariates: covariates.iter().map(|c| c.as_array().to_vec()).collect(),
    });
    let ctx = crate::measured_inference_api::context(0, memory_limit_bytes, None);
    detach_catch(py, move || {
        match LatentClassEffects::consume_with(&artifact, expected.as_ref(), data.as_ref(), &ctx) {
            Ok(done) => {
                let report = serde_json::to_string(&done.report())
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
    m.add_function(wrap_pyfunction!(evaluate_latent_class_effects, m)?)?;
    m.add_function(wrap_pyfunction!(consume_latent_class_artifact, m)?)
}
