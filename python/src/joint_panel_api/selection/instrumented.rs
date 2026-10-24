//! Native continuous selection from independent original IV observations.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::error;
use crate::joint_panel_api::context;
use antecedent::analysis::joint_panel::selection::instrumented::{
    InstrumentedSelectionRequest, InstrumentedSelectionResult, replay_instrumented_selection,
};
use pyo3::{prelude::*, types::PyBytes};

fn declaration(json: &str) -> PyResult<InstrumentedSelectionRequest> {
    crate::bridge::check_declared_size(
        json,
        crate::bridge::MAX_SPEC_DECLARATION_BYTES,
        "instrumented program selection",
    )?;
    serde_json::from_str(json).map_err(|e| crate::value_err(e.to_string()))
}
#[pyclass(name = "_InstrumentedProgramSelection", frozen)]
struct PyInstrumentedSelection {
    inner: InstrumentedSelectionResult,
    request: InstrumentedSelectionRequest,
}
#[pymethods]
impl PyInstrumentedSelection {
    fn report(&self) -> String {
        self.inner.report().to_string()
    }
    fn export(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        Ok(PyBytes::new(py, &self.inner.export().map_err(error)?).unbind())
    }
    #[pyo3(signature=(bytes,*,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn verify_current(
        &self,
        py: Python<'_>,
        bytes: &[u8],
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<()> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        crate::detach_catch(py, || {
            replay_instrumented_selection(bytes, Some(&self.request), &ctx)
                .map(|_| ())
                .map_err(error)
        })
    }
}
#[pyfunction]
#[pyo3(signature=(declaration_json,*,repetitions=199,tie_tolerance=0.0,seed=0,threads=1,max_bytes=None,cancel=None))]
fn select_instrumented_program(
    py: Python<'_>,
    declaration_json: &str,
    repetitions: usize,
    tie_tolerance: f64,
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyInstrumentedSelection> {
    let ctx = context(seed, threads, max_bytes, cancel)?;
    if ctx.cancellation.is_cancelled() {
        return Err(crate::CausalCancelledError::new_err("instrumented selection cancelled"));
    }
    let _input = ctx
        .acquire_admission(0, (declaration_json.len() as u64).saturating_mul(16))
        .map_err(|e| crate::CausalResourceError::new_err(e.to_string()))?;
    let request = declaration(declaration_json)?;
    let inner = crate::detach_catch(py, || {
        request.execute(repetitions, tie_tolerance, &ctx).map_err(error)
    })?;
    Ok(PyInstrumentedSelection { inner, request })
}
#[pyfunction]
#[pyo3(signature=(bytes,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn consume_instrumented_selection(
    py: Python<'_>,
    bytes: &[u8],
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let ctx = context(seed, threads, max_bytes, cancel)?;
    crate::detach_catch(py, || {
        replay_instrumented_selection(bytes, None, &ctx)
            .map(|result| {
                serde_json::json!({"report": result.report(), "grants_current_authority":false})
                    .to_string()
            })
            .map_err(error)
    })
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyInstrumentedSelection>()?;
    module.add_function(wrap_pyfunction!(select_instrumented_program, module)?)?;
    module.add_function(wrap_pyfunction!(consume_instrumented_selection, module)?)?;
    Ok(())
}
