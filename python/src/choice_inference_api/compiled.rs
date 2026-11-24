//! The compiled measured interval of one result of a choice study.
//!
//! `ChoiceInferenceHandle.compiled_interval` parses the request exactly as `measured_interval`
//! does, binds the session, the request and the same artifact id into the
//! `ChoiceResultInterval` program family and executes it. The family seals, replays and
//! measures the result with the same consumer as the direct interval, so the numbers and
//! refusals are identical.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::Study;
use pyo3::prelude::*;

use super::{ChoiceInferenceHandle, refusal_value, request_of};
use crate::measurement_model_api::parse;
use crate::transport_common::execution_context;

/// The artifact id the direct `measured_interval` seals the result under.
const ARTIFACT_ID: &str = "measured-choice-interval";

#[pymethods]
impl ChoiceInferenceHandle {
    /// The interval of one result of a measured choice cell, through the compiled
    /// `ChoiceResultInterval` program.
    #[pyo3(signature = (request_json, cancel=None))]
    fn compiled_interval(
        &self,
        py: Python<'_>,
        request_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<(Option<crate::measured_inference_api::NativeMeasuredInference>, Option<String>)>
    {
        let request = request_of(&parse(request_json, "choice request")?)?;
        let ctx = execution_context(0, None, cancel);
        let result = crate::detach_catch(py, || {
            let compiled = Study::compile_choice_interval(&self.inner, request, ARTIFACT_ID)
                .map_err(|error| crate::value_err(error.to_string()))?;
            Ok(compiled.execute(&ctx))
        })?;
        Ok(match result {
            Ok(run) => (
                Some(crate::measured_inference_api::NativeMeasuredInference::from_io(
                    run.measured,
                )?),
                None,
            ),
            Err(refusal) => (None, Some(refusal_value(&refusal).to_string())),
        })
    }
}
