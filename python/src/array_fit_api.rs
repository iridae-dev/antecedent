//! Python bridge for the retained fits of arrays and prepared designs.
//!
//! The report is the stored result with the request digest; the artifact is the binary replay
//! artifact whose consumer refits from the arrays it holds. Every estimate, refusal and
//! recomputation is Rust's: a refusal raises the typed exception with its reason code, and a
//! corrupt or edited artifact raises the same.

use antecedent::analysis::array_fits::{ArrayFit, ArrayFitConsumeLimits};
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::bridge::{MAX_DECLARED_ARTIFACT_BYTES, check_declared_len};
use crate::{detach_catch, py_err};

/// The report of a fit as JSON: its result and the request digest.
pub(crate) fn report_json(fit: &ArrayFit) -> PyResult<String> {
    serde_json::to_string(&serde_json::json!({
        "result": fit.result(),
        "request_digest": fit.digest(),
    }))
    .map_err(|e| crate::serialization_err(e.to_string()))
}

/// The report and the binary artifact of a fit.
pub(crate) fn report_and_artifact(
    py: Python<'_>,
    fit: &ArrayFit,
) -> PyResult<(String, Py<PyBytes>)> {
    let bytes = fit.export().map_err(|e| py_err(antecedent::CausalError::from(e)))?;
    Ok((report_json(fit)?, PyBytes::new(py, &bytes).unbind()))
}

/// Consume a retained artifact by refitting from its arrays: `(report json, estimator id)`.
#[pyfunction]
#[pyo3(signature = (artifact, *, max_numbers=4_000_000, max_draws=200_000, max_columns=1_024))]
fn consume_array_fit_artifact(
    py: Python<'_>,
    artifact: Vec<u8>,
    max_numbers: usize,
    max_draws: usize,
    max_columns: usize,
) -> PyResult<(String, String)> {
    check_declared_len(artifact.len(), MAX_DECLARED_ARTIFACT_BYTES, "array fit artifact")?;
    let limits = ArrayFitConsumeLimits { max_numbers, max_draws, max_columns };
    detach_catch(py, move || {
        let fit = ArrayFit::consume(&artifact, limits)
            .map_err(|e| py_err(antecedent::CausalError::from(e)))?;
        Ok((report_json(&fit)?, fit.estimator_id().to_owned()))
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(consume_array_fit_artifact, m)?)
}
