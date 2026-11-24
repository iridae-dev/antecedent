//! The compiled posterior interval of a count-fitted observation probability.
//!
//! `count_fitted_observation_compiled_interval` decodes the retained source exactly as
//! `count_fitted_observation_interval` does, binds it and the level into the
//! `CountFittedInterval` program family and executes it. The family runs the same measured
//! consumer as the direct interval, so the numbers are identical.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::Study;
use pyo3::prelude::*;

use super::{count_fitted_of, execution_context, refused};

/// The compiled measured posterior interval or the typed refusal:
/// `(measured inference, refusal json)`.
#[pyfunction]
#[pyo3(signature = (bytes, level, *, cancel=None))]
pub(super) fn count_fitted_observation_compiled_interval(
    py: Python<'_>,
    bytes: &[u8],
    level: f64,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(Option<crate::measured_inference_api::NativeMeasuredInference>, Option<String>)> {
    let observation = match count_fitted_of(bytes)? {
        Ok(observation) => observation,
        Err(refusal) => return Ok((None, refused::<()>(&refusal).1)),
    };
    crate::detach_catch(py, move || {
        let ctx = execution_context(0, None, cancel);
        let compiled = Study::compile_count_fitted_interval(observation, level)
            .map_err(|error| crate::value_err(error.to_string()))?;
        match compiled.execute(&ctx) {
            Ok(run) => Ok((
                Some(crate::measured_inference_api::NativeMeasuredInference::from_io(
                    run.measured,
                )?),
                None,
            )),
            Err(refusal) => Ok((None, refused::<()>(&refusal).1)),
        }
    })
}
