//! Immutable native nuisance-score owners and independent replay consumption.

use crate::bridge::{MAX_DECLARED_ARTIFACT_BYTES, check_declared_len};
use crate::{detach_catch, py_err};
use antecedent::analysis::array_fits::{ArrayFit, ArrayFitConsumeLimits, ArrayFitError};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use std::sync::Arc;

/// Admit the real numerical work and preserve operational failures across the
/// legacy array-fit receipt layer, without changing its scientific encoding.
pub(crate) fn execute_score(
    seed: u64,
    threads: Option<u32>,
    stage: &'static str,
    fit: impl FnOnce(&antecedent_core::ExecutionContext) -> Result<ArrayFit, antecedent::CausalError>,
) -> PyResult<ArrayFit> {
    let ctx = crate::py_execution_context(seed, crate::resolve_user_threads(threads));
    let check_cancel = || {
        if ctx.cancellation.is_cancelled() {
            Err(py_err(antecedent::CausalError::Cancelled { stage }))
        } else {
            Ok(())
        }
    };
    check_cancel()?;
    if let Some(progress) = &ctx.progress {
        progress.report(0.0, stage);
    }
    check_cancel()?;
    let (result, failure) =
        antecedent_core::numerical_admission::with_context_and_failure(&ctx, || fit(&ctx));
    check_cancel()?;
    if let (true, Some(error)) = (result.is_err(), failure) {
        return Err(match error {
            antecedent_core::execution::AdmissionError::Cancelled => {
                py_err(antecedent::CausalError::Cancelled { stage })
            }
            error => crate::CausalResourceError::new_err(error.to_string()),
        });
    }
    let fit = result.map_err(py_err)?;
    if let Some(progress) = &ctx.progress {
        progress.report(1.0, stage);
    }
    check_cancel()?;
    Ok(fit)
}

/// An owner of verified native request/result arrays; never a reconstructed live study.
#[pyclass(name = "_ArrayFit", frozen, skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct PyArrayFit {
    inner: Arc<ArrayFit>,
    consumed: bool,
    identity_bound: bool,
    original_encoding: Option<Arc<Py<PyBytes>>>,
}

impl PyArrayFit {
    pub(crate) fn produced(fit: ArrayFit) -> Self {
        Self {
            inner: Arc::new(fit),
            consumed: false,
            identity_bound: false,
            original_encoding: None,
        }
    }
}

fn page(offset: usize, limit: usize, total: usize) -> PyResult<std::ops::Range<usize>> {
    if limit == 0 || limit > 256 || offset > total {
        return Err(crate::value_err("page requires 0 <= offset <= total and 1 <= limit <= 256"));
    }
    Ok(offset..offset.saturating_add(limit).min(total))
}

#[pymethods]
impl PyArrayFit {
    fn metadata_json(&self) -> PyResult<String> {
        let result = self.inner.result();
        serde_json::to_string(&serde_json::json!({
            "estimator_id": result.estimator_id, "target": result.target,
            "uncertainty": result.uncertainty, "licensing": result.licensing,
            "calibration": result.calibration, "request_digest": self.inner.digest(),
            "value_count": result.values.len(), "assumption_count": result.assumptions.len(),
            "diagnostic_count": result.diagnostics.len(), "consumed": self.consumed,
            "identity_bound": self.identity_bound,
        }))
        .map_err(|error| crate::serialization_err(error.to_string()))
    }

    fn value(&self, name: &str) -> Option<f64> {
        self.inner.value(name)
    }

    #[pyo3(signature = (*, offset=0, limit=64))]
    fn values_page(&self, offset: usize, limit: usize) -> PyResult<Vec<(String, f64)>> {
        let values = &self.inner.result().values;
        Ok(values[page(offset, limit, values.len())?]
            .iter()
            .map(|value| (value.name.clone(), value.value))
            .collect())
    }

    #[pyo3(signature = (*, offset=0, limit=64))]
    fn assumptions_page(&self, offset: usize, limit: usize) -> PyResult<Vec<String>> {
        let values = &self.inner.result().assumptions;
        Ok(values[page(offset, limit, values.len())?].to_vec())
    }

    #[pyo3(signature = (*, offset=0, limit=64))]
    fn diagnostics_page(&self, offset: usize, limit: usize) -> PyResult<Vec<String>> {
        let values = &self.inner.result().diagnostics;
        Ok(values[page(offset, limit, values.len())?].to_vec())
    }

    fn export_bytes(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        // No fit or replay: serialize only at this explicit boundary and reuse the encoding.
        if let Some(original) = &self.original_encoding {
            return Ok(original.clone_ref(py));
        }
        let inner = Arc::clone(&self.inner);
        // Heavy native serialization runs detached from the GIL, under the caller's
        // execution scope, so SIGINT/cancellation and concurrent exports stay responsive.
        let encoded = detach_catch(py, move || {
            let ctx = crate::py_execution_context(0, 1);
            if ctx.cancellation.is_cancelled() {
                return Err(py_err(antecedent::CausalError::Cancelled {
                    stage: "array_fit_export",
                }));
            }
            inner
                .encoded_with_context(&ctx)
                .map_err(|error| py_err(antecedent::CausalError::from(error)))
        })?;
        Ok(PyBytes::new(py, encoded.bytes()).unbind())
    }

    fn encoding_retained(&self) -> bool {
        self.original_encoding.is_some() || self.inner.has_encoding()
    }
}

fn consumer_error(error: ArrayFitError) -> antecedent::CausalError {
    match error {
        error @ ArrayFitError::LimitsExceeded(_) => {
            antecedent::CausalError::Resource { message: error.to_string() }
        }
        error @ (ArrayFitError::Admission(_) | ArrayFitError::Refused { .. }) => error.into(),
        error => antecedent::CausalError::Serialization(antecedent_io::IoError::Convert(
            error.to_string(),
        )),
    }
}

/// Select this consumer by array_fit_replay_v1; replay the native recipe exactly once.
#[pyfunction]
#[pyo3(signature = (artifact, *, expected_identity=None, max_numbers=4_000_000, max_draws=200_000, max_columns=1_024, threads=None))]
fn consume_array_fit_artifact(
    py: Python<'_>,
    artifact: Py<PyBytes>,
    expected_identity: Option<String>,
    max_numbers: usize,
    max_draws: usize,
    max_columns: usize,
    threads: Option<u32>,
) -> PyResult<PyArrayFit> {
    let bytes = artifact.bind(py).as_bytes();
    check_declared_len(bytes.len(), MAX_DECLARED_ARTIFACT_BYTES, "array fit artifact")?;
    if expected_identity.as_ref().is_some_and(|digest| {
        digest.len() != 64
            || !digest.bytes().all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    }) {
        return Err(crate::value_err(
            "expected_identity must be 64 lowercase hexadecimal characters",
        ));
    }
    let limits = ArrayFitConsumeLimits { max_numbers, max_draws, max_columns };
    let identity_bound = expected_identity.is_some();
    let fit = detach_catch(py, move || {
        execute_score(0, threads, "array_fit_consume", |_| {
            ArrayFit::consume_expected(bytes, limits, expected_identity.as_deref())
                .map_err(consumer_error)
        })
    })?;
    Ok(PyArrayFit {
        inner: Arc::new(fit),
        consumed: true,
        identity_bound,
        original_encoding: Some(Arc::new(artifact)),
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyArrayFit>()?;
    m.add_function(wrap_pyfunction!(consume_array_fit_artifact, m)?)
}
