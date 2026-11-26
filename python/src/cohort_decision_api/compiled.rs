//! The compiled measured interval of a declared cohort's mean conditional effect.
//!
//! `cohort_decision_compiled_interval` consumes the cohort source and decodes the native fit
//! exactly as `cohort_decision_interval` does, binds them and the level into the
//! `CohortMeanInterval` program family and executes it. The family runs the same measured
//! consumer as the direct interval, so the numbers and refusals are identical.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::cohort_decision::{CohortDecisionSource, CohortExpectation};
use antecedent::Study;
use pyo3::prelude::*;

use super::{
    check_size, invalid_argument, native_fit_of, native_fit_refusal, refusal_json, ExpectationJson,
    NativeFitJson,
};
use crate::bridge::RefusalJson;
use crate::detach_catch;

/// The compiled interval of a cohort source: the measured result for a source built from a
/// native fit whose rows are retained, a typed refusal otherwise:
/// `(measured inference, refusal json)`.
#[pyfunction]
#[pyo3(signature = (artifact, expectation_json, fit_json, level))]
pub(super) fn cohort_decision_compiled_interval(
    py: Python<'_>,
    artifact: &[u8],
    expectation_json: &str,
    fit_json: Option<String>,
    level: f64,
) -> PyResult<(Option<crate::measured_inference_api::NativeMeasuredInference>, Option<String>)> {
    check_size(expectation_json, "cohort expectation")?;
    let parsed: ExpectationJson = serde_json::from_str(expectation_json)
        .map_err(|e| invalid_argument(format!("cohort_decision.invalid_request: {e}")))?;
    let fit: Option<NativeFitJson> =
        match &fit_json {
            Some(text) => {
                check_size(text, "native fit")?;
                Some(serde_json::from_str(text).map_err(|e| {
                    invalid_argument(format!("cohort_decision.invalid_request: {e}"))
                })?)
            }
            None => None,
        };
    let expectation = CohortExpectation {
        identity: parsed.identity,
        parent_claim: parsed.parent_claim,
        population_id: parsed.population_id,
        outcome_units: parsed.outcome_units,
        contrast_regime: parsed.contrast_regime,
        cohort_digest: parsed.cohort_digest,
    };
    let ctx = crate::execution_api::context(0);
    detach_catch(py, move || {
        let source = match CohortDecisionSource::consume_with(artifact, &expectation, &ctx) {
            Ok(source) => source,
            Err(error) => return Ok((None, Some(refusal_json(&error, "consume")))),
        };
        let Some(fit) = fit else {
            return Ok((None, Some(refusal_json(&source.interval_refusal(), "evaluate"))));
        };
        let native = match native_fit_of(&fit) {
            Ok(native) => native,
            Err(message) => return Ok((None, Some(native_fit_refusal(&message)))),
        };
        let compiled = Study::compile_cohort_interval(native, source, level)
            .map_err(|error| crate::value_err(error.to_string()))?;
        match compiled.execute(&ctx) {
            Ok(run) => Ok((
                Some(crate::measured_inference_api::NativeMeasuredInference::from_io(
                    run.measured,
                )?),
                None,
            )),
            Err(antecedent::io::IoError::Refused { code, message }) => {
                Ok((None, Some(RefusalJson::from_coded_message(code, "evaluate", &message).json())))
            }
            Err(other) => Err(crate::transport_common::error(other)),
        }
    })
}
