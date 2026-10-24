//! Python inputs execute the native complete randomized selection procedure.
// SPDX-License-Identifier: MIT OR Apache-2.0
use super::{context, engine};
use antecedent::analysis::joint_panel::selection::{
    PreparedRandomizedSelection, RandomizedSelectionPolicy, RandomizedSelectionResult,
    replay_randomized_selection,
};
use antecedent_design::program_search::ProgramSearchError;
use antecedent_estimate::joint_design::randomized::{JointRandomizedDesign, JointRandomizedUnit};
use pyo3::{prelude::*, types::PyBytes};
mod instrumented;

fn error(e: ProgramSearchError) -> PyErr {
    match e {
        ProgramSearchError::Cancelled => crate::CausalCancelledError::new_err(e.to_string()),
        ProgramSearchError::Resource(_) => crate::CausalResourceError::new_err(e.to_string()),
        ProgramSearchError::Invalid(_) => crate::value_err(e.to_string()),
    }
}
#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct Declaration {
    design: JointRandomizedDesign,
    training: Vec<JointRandomizedUnit>,
    validation: Vec<JointRandomizedUnit>,
    policy: RandomizedSelectionPolicy,
}
#[pyclass(name = "_PreparedRandomizedProgramSelection", frozen)]
pub(crate) struct PyPreparedRandomizedSelection {
    inner: PreparedRandomizedSelection,
}
#[pymethods]
impl PyPreparedRandomizedSelection {
    #[pyo3(signature=(*,repetitions=199,tie_tolerance=0.0,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn execute(
        &self,
        py: Python<'_>,
        repetitions: usize,
        tie_tolerance: f64,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<PyRandomizedSelection> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let inner = crate::detach_catch(py, || {
            self.inner.execute(repetitions, tie_tolerance, &ctx).map_err(error)
        })?;
        Ok(PyRandomizedSelection { inner, prepared: self.inner.clone() })
    }
    #[pyo3(signature=(*,repetitions=199,level=0.95,seed=0,threads=1,max_bytes=None,cancel=None))]
    fn simultaneous_intervals(
        &self,
        py: Python<'_>,
        repetitions: usize,
        level: f64,
        seed: u64,
        threads: u32,
        max_bytes: Option<u64>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<String> {
        let ctx = context(seed, threads, max_bytes, cancel)?;
        let r = crate::detach_catch(py, || {
            self.inner.simultaneous_intervals(repetitions, level, &ctx).map_err(error)
        })?;
        Ok(serde_json::json!({"ids":r.ids,"intervals":r.intervals,"level":r.level,
            "critical_value":r.critical_value,"roots":r.roots,"selected_id":r.selected_id,
            "selected_index":r.selected_index,"calibration":"unmeasured",
            "uncertainty":"fixed_set_joint_complete_procedure_bootstrap_max_statistic"})
        .to_string())
    }
}
#[pyclass(name = "_RandomizedProgramSelection", frozen)]
pub(crate) struct PyRandomizedSelection {
    inner: RandomizedSelectionResult,
    prepared: PreparedRandomizedSelection,
}
#[pymethods]
impl PyRandomizedSelection {
    fn report(&self) -> String {
        self.inner.report().to_string()
    }
    fn export(&self, py: Python<'_>) -> PyResult<Py<PyBytes>> {
        let bytes = self.inner.export().map_err(error)?;
        Ok(PyBytes::new(py, &bytes).unbind())
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
            self.prepared.consume_current(bytes, &ctx).map(|_| ()).map_err(error)
        })
    }
}
#[pyfunction]
#[pyo3(signature=(declaration_json,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn prepare_randomized_program_selection(
    py: Python<'_>,
    declaration_json: &str,
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<PyPreparedRandomizedSelection> {
    crate::bridge::check_declared_size(
        declaration_json,
        crate::bridge::MAX_SPEC_DECLARATION_BYTES,
        "randomized program selection",
    )?;
    let ctx = context(seed, threads, max_bytes, cancel)?;
    if ctx.cancellation.is_cancelled() {
        return Err(crate::CausalCancelledError::new_err("randomized selection cancelled"));
    }
    let _input = ctx
        .acquire_admission(0, (declaration_json.len() as u64).saturating_mul(8))
        .map_err(|e| super::panel_error(engine::PanelError::Resource(e.to_string()).into()))?;
    let d: Declaration =
        serde_json::from_str(declaration_json).map_err(|e| crate::value_err(e.to_string()))?;
    crate::detach_catch(py, || {
        PreparedRandomizedSelection::compile(d.design, d.training, d.validation, d.policy)
            .map(|inner| PyPreparedRandomizedSelection { inner })
            .map_err(error)
    })
}
#[pyfunction]
#[pyo3(signature=(bytes,*,seed=0,threads=1,max_bytes=None,cancel=None))]
fn consume_randomized_program_selection(
    py: Python<'_>,
    bytes: &[u8],
    seed: u64,
    threads: u32,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let ctx = context(seed, threads, max_bytes, cancel)?;
    crate::detach_catch(py, || {
        replay_randomized_selection(bytes, &ctx)
            .map(|r| {
                serde_json::json!({"report":r.report(),"grants_current_authority":false})
                    .to_string()
            })
            .map_err(error)
    })
}
pub(super) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    instrumented::register(module)?;
    module.add_class::<PyPreparedRandomizedSelection>()?;
    module.add_class::<PyRandomizedSelection>()?;
    module.add_function(wrap_pyfunction!(prepare_randomized_program_selection, module)?)?;
    module.add_function(wrap_pyfunction!(consume_randomized_program_selection, module)?)?;
    Ok(())
}
