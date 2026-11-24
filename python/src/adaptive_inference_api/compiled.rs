//! The compiled measured interval of one result of an adaptive experiment.
//!
//! `AdaptiveInferenceHandle.compiled_interval` parses the request exactly as
//! `measured_interval` does, binds the session, the request and the same artifact id into the
//! `AdaptiveResultInterval` program family and executes it. The family seals, replays and
//! measures the result with the same consumer as the direct interval, so the numbers and
//! refusals are identical.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::Study;
use pyo3::prelude::*;

use super::{AdaptiveInferenceHandle, parse_request, refusal_value};
use crate::measurement_model_api::parse;
use crate::transport_common::execution_context;

/// The artifact id the direct `measured_interval` seals the result under.
const ARTIFACT_ID: &str = "measured-adaptive-interval";

#[pymethods]
impl AdaptiveInferenceHandle {
    /// The interval of one result of a measured procedure cell, through the compiled
    /// `AdaptiveResultInterval` program.
    #[pyo3(signature = (request_json, cancel=None))]
    fn compiled_interval(
        &self,
        py: Python<'_>,
        request_json: &str,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<(Option<crate::measured_inference_api::NativeMeasuredInference>, Option<String>)>
    {
        crate::catch_ffi(|| {
            let request = parse_request(&parse(request_json, "adaptive inference request")?)?;
            let ctx = execution_context(0, None, cancel);
            let outcome = py.detach(|| {
                Study::compile_adaptive_interval(&self.inner, request, ARTIFACT_ID)
                    .map(|compiled| compiled.execute(&ctx))
                    .map_err(|error| error.to_string())
            });
            let outcome = outcome.map_err(crate::value_err)?;
            Ok(match outcome {
                Ok(run) => (
                    Some(crate::measured_inference_api::NativeMeasuredInference::from_io(
                        run.measured,
                    )?),
                    None,
                ),
                Err(refusal) => (None, Some(refusal_value(&refusal).to_string())),
            })
        })
    }
}
