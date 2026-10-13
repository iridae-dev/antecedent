//! Bounded Python bridge for the cohort-mean treatment-contrast decision source.
//!
//! Python declares a cohort (named feature rows, identities, weights, population) and the
//! bytes of a retained model: an exported analysis result or a compact export. Every number,
//! the support check, the identity, the artifact and every refusal are Rust's; Python never
//! supplies a prediction. A refusal comes back as structured JSON for the Python layer to
//! raise as its own exception type.
//!
//! The source answers the expectation of an affine utility of one cohort-mean contrast and
//! nothing else. The model-based standard error of a linear compact model is a labelled,
//! uncalibrated diagnostic and never an interval.

use antecedent::analysis::cohort_decision::{
    CohortBinding, CohortDecisionSource, CohortExpectation, CohortOptions, CohortRefusal,
    CovariateShiftDeclaration, FittedEffectModel, LawRequest, cohort_forward_claim,
    cohort_mean_source,
};
use antecedent_core::ExecutionContext;
use antecedent_io::quantity_wire::ScientificQuantityWire;
use pyo3::prelude::*;

use crate::bridge::{RefusalJson, check_declared_size};
use crate::{detach_catch, value_err_code};

const MAX_JSON_BYTES: usize = crate::bridge::MAX_DECLARED_ARTIFACT_BYTES;

type BuildPayload = (Option<String>, Option<Vec<u8>>, Option<String>);
type ConsumePayload = (Option<String>, Option<String>);

fn invalid_argument(message: String) -> PyErr {
    value_err_code(antecedent_core::reason_code!("invalid_argument"), message)
}

fn refusal_json(error: &CohortRefusal, stage: &'static str) -> String {
    RefusalJson::new(error.code, stage, error.detail, &error.to_string())
        .offending(Some(&error.subject))
        .remedy(Some(error.remedy))
        .json()
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ShiftJson {
    source_population: String,
    target_population: String,
    invariant_modifiers: Vec<String>,
}

#[derive(serde::Deserialize, Default)]
#[serde(deny_unknown_fields, default)]
struct OptionsJson {
    shift: Option<ShiftJson>,
    expected_contrast_regime: Option<String>,
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct ExpectationJson {
    identity: String,
    #[serde(default)]
    parent_claim: Option<String>,
    #[serde(default)]
    population_id: Option<String>,
    #[serde(default)]
    outcome_units: Option<String>,
    #[serde(default)]
    contrast_regime: Option<String>,
    #[serde(default)]
    cohort_digest: Option<String>,
}

/// Which retained model the bytes are.
#[derive(serde::Deserialize)]
#[serde(rename_all = "snake_case")]
enum ModelKind {
    AnalysisResult,
    CompactExport,
}

fn check_size(text: &str, what: &str) -> PyResult<()> {
    check_declared_size(text, MAX_JSON_BYTES, what)
}

fn parse_cohort(text: &str) -> PyResult<Result<CohortBinding, CohortRefusal>> {
    check_size(text, "cohort")?;
    let cohort: CohortBinding = serde_json::from_str(text)
        .map_err(|e| invalid_argument(format!("cohort_decision.invalid_request: {e}")))?;
    Ok(cohort.validate().map(|()| cohort))
}

fn options_of(text: &str) -> PyResult<CohortOptions> {
    check_size(text, "cohort options")?;
    let parsed: OptionsJson = serde_json::from_str(text)
        .map_err(|e| invalid_argument(format!("cohort_decision.invalid_request: {e}")))?;
    Ok(CohortOptions {
        shift: parsed.shift.map(|s| {
            let modifiers: Vec<&str> = s.invariant_modifiers.iter().map(String::as_str).collect();
            CovariateShiftDeclaration::new(&s.source_population, &s.target_population, &modifiers)
        }),
        expected_contrast_regime: parsed.expected_contrast_regime,
    })
}

fn summary(source: &CohortDecisionSource) -> serde_json::Value {
    let mean = cohort_mean_source(source);
    let quantity = ScientificQuantityWire::from(source.quantity());
    let standard_error = source.uncalibrated_standard_error().map(|d| {
        serde_json::json!({
            "value": d.value, "basis": d.basis, "scope": d.scope, "calibration": d.calibration,
        })
    });
    serde_json::json!({
        "identity": source.identity(),
        "model_identity": source.model().identity(),
        "parent_claim": source.model().parent().claim_id,
        "is_native": source.is_native(),
        "provider_id": source.provider_id(),
        "outcome_units": source.outcome_units(),
        "cohort_digest": source.cohort().digest(),
        "population_id": source.cohort().population_id(),
        "feature_names": source.model().feature_names(),
        "mean_contrast": source.mean_contrast(),
        "row_contrasts": source.row_contrasts(),
        "quantity": quantity,
        "mean_source": {
            "coordinates": [quantity],
            "means": mean.means,
            "provider_id": mean.provider_id,
            "snapshot_id": mean.snapshot_id,
            "causal_contract_id": mean.causal_contract_id,
            "rng_id": mean.rng_id,
        },
        "support": source.support(),
        "support_status": source.support_status().as_str(),
        "uncalibrated_standard_error": standard_error,
        "shift": source.shift(),
        "assumptions": source.assumptions(),
    })
}

/// Bind a cohort to a retained model and issue the source:
/// `(summary json, artifact bytes, refusal json)`.
///
/// `model_kind` is `analysis_result` (the exported parent result) or `compact_export`;
/// `expected_model` is the claim (analysis result) or export identity the caller retained.
#[pyfunction]
#[pyo3(signature = (model_kind, model_bytes, expected_model, cohort_json, outcome_units, options_json, artifact_id))]
#[allow(
    clippy::too_many_arguments,
    reason = "one flat native entry mirroring the Rust facade call"
)]
fn build_cohort_decision_source(
    py: Python<'_>,
    model_kind: &str,
    model_bytes: Vec<u8>,
    expected_model: Option<String>,
    cohort_json: &str,
    outcome_units: &str,
    options_json: &str,
    artifact_id: &str,
) -> PyResult<BuildPayload> {
    let kind: ModelKind = serde_json::from_value(serde_json::Value::String(model_kind.to_owned()))
        .map_err(|_| invalid_argument("cohort_decision.invalid_request: model_kind".to_owned()))?;
    let cohort = match parse_cohort(cohort_json)? {
        Ok(cohort) => cohort,
        Err(error) => return Ok((None, None, Some(refusal_json(&error, "bind")))),
    };
    let options = options_of(options_json)?;
    let units = outcome_units.to_owned();
    let artifact_id = artifact_id.to_owned();
    detach_catch(py, move || {
        let ctx = ExecutionContext::production_default(0);
        let model = match kind {
            ModelKind::AnalysisResult => match &expected_model {
                Some(claim) => FittedEffectModel::load_expecting(&model_bytes, claim),
                None => FittedEffectModel::load(&model_bytes),
            },
            ModelKind::CompactExport => {
                use antecedent::analysis::compact_export::{CompactExport, ExportLimits};
                let Some(identity) = &expected_model else {
                    return Ok((
                        None,
                        None,
                        Some(refusal_json(
                            &CohortRefusal {
                                code: antecedent_core::reason_code!("invalid_argument"),
                                detail: "cohort_decision.fitted_model_invalid",
                                subject: "expected_identity".to_owned(),
                                remedy: "retain the compact export identity and supply it",
                            },
                            "bind",
                        )),
                    ));
                };
                match CompactExport::consume(&model_bytes, &ExportLimits::default(), identity) {
                    Ok(export) => FittedEffectModel::from_compact_export(&export),
                    Err(error) => Err(CohortRefusal {
                        code: error.code,
                        detail: "cohort_decision.fitted_model_invalid",
                        subject: format!("{}: {}", error.detail, error.subject),
                        remedy: "re-export the model from its producer",
                    }),
                }
            }
        };
        let built = model
            .and_then(|model| model.decision_source(&cohort, &units, &options, &ctx))
            .and_then(|source| {
                let bytes = source.to_bytes(&artifact_id)?;
                Ok((source, bytes))
            });
        match built {
            Ok((source, bytes)) => Ok((Some(summary(&source).to_string()), Some(bytes), None)),
            Err(error) => Ok((None, None, Some(refusal_json(&error, "bind")))),
        }
    })
}

/// Independently replay an artifact under the retained expectation:
/// `(summary json, refusal json)`.
#[pyfunction]
fn consume_cohort_decision_source(
    py: Python<'_>,
    artifact: Vec<u8>,
    expectation_json: &str,
) -> PyResult<ConsumePayload> {
    check_size(expectation_json, "cohort expectation")?;
    let parsed: ExpectationJson = serde_json::from_str(expectation_json)
        .map_err(|e| invalid_argument(format!("cohort_decision.invalid_request: {e}")))?;
    let expectation = CohortExpectation {
        identity: parsed.identity,
        parent_claim: parsed.parent_claim,
        population_id: parsed.population_id,
        outcome_units: parsed.outcome_units,
        contrast_regime: parsed.contrast_regime,
        cohort_digest: parsed.cohort_digest,
    };
    detach_catch(py, move || {
        match antecedent::analysis::cohort_decision::consume_cohort_decision_source(
            &artifact,
            &expectation,
        ) {
            Ok(source) => Ok((Some(summary(&source).to_string()), None)),
            Err(error) => Ok((None, Some(refusal_json(&error, "consume")))),
        }
    })
}

/// The cohort digest and validation of a declared cohort: `(digest, refusal json)`.
#[pyfunction]
fn validate_cohort_binding(cohort_json: &str) -> PyResult<ConsumePayload> {
    Ok(match parse_cohort(cohort_json)? {
        Ok(cohort) => (Some(cohort.digest()), None),
        Err(error) => (None, Some(refusal_json(&error, "bind"))),
    })
}

/// Normalize raw weights explicitly: `(weights json, refusal json)`.
#[pyfunction]
fn normalize_cohort_weights(weights: Vec<f64>) -> ConsumePayload {
    match CohortBinding::normalize_weights(&weights) {
        Ok(w) => (Some(serde_json::json!(w).to_string()), None),
        Err(error) => (None, Some(refusal_json(&error, "bind"))),
    }
}

/// The typed refusal of a request the cohort mean cannot answer, with its remedy.
/// `kind` is `probability`, `quantile`, `nonlinear_utility`, `joint_law` or `interval`.
#[pyfunction]
fn cohort_decision_law_refusal(kind: &str) -> PyResult<String> {
    let request = match kind {
        "probability" => LawRequest::Probability,
        "quantile" => LawRequest::Quantile,
        "nonlinear_utility" => LawRequest::NonlinearUtility,
        "joint_law" => LawRequest::JointLaw,
        "interval" => LawRequest::Interval,
        other => {
            return Err(invalid_argument(format!("cohort_decision.invalid_request: kind {other}")));
        }
    };
    let mut refusal = request.refusal();
    kind.clone_into(&mut refusal.subject);
    Ok(refusal_json(&refusal, "evaluate"))
}

/// Whether a replayed source may serve as an inverse-query forward claim:
/// `(mean source json, refusal json)`. A source built under a declared shift is refused.
#[pyfunction]
fn cohort_decision_forward_claim(
    py: Python<'_>,
    artifact: Vec<u8>,
    expectation_json: &str,
) -> PyResult<ConsumePayload> {
    check_size(expectation_json, "cohort expectation")?;
    let parsed: ExpectationJson = serde_json::from_str(expectation_json)
        .map_err(|e| invalid_argument(format!("cohort_decision.invalid_request: {e}")))?;
    let expectation = CohortExpectation {
        identity: parsed.identity,
        parent_claim: parsed.parent_claim,
        population_id: parsed.population_id,
        outcome_units: parsed.outcome_units,
        contrast_regime: parsed.contrast_regime,
        cohort_digest: parsed.cohort_digest,
    };
    detach_catch(py, move || {
        let source = match antecedent::analysis::cohort_decision::consume_cohort_decision_source(
            &artifact,
            &expectation,
        ) {
            Ok(source) => source,
            Err(error) => return Ok((None, Some(refusal_json(&error, "consume")))),
        };
        match cohort_forward_claim(&source) {
            Ok(_) => Ok((Some(summary(&source)["mean_source"].to_string()), None)),
            Err(error) => Ok((None, Some(refusal_json(&error, "evaluate")))),
        }
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(build_cohort_decision_source, m)?)?;
    m.add_function(wrap_pyfunction!(consume_cohort_decision_source, m)?)?;
    m.add_function(wrap_pyfunction!(validate_cohort_binding, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_cohort_weights, m)?)?;
    m.add_function(wrap_pyfunction!(cohort_decision_law_refusal, m)?)?;
    m.add_function(wrap_pyfunction!(cohort_decision_forward_claim, m)?)
}
