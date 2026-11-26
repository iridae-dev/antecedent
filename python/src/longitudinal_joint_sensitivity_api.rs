//! Native bridge for the compiled two-decision sensitivity study.
use antecedent::analysis::longitudinal_joint_sensitivity::{
    CompiledLongitudinalJointSensitivity, JointSensitivityError, LongitudinalSensitivityArtifact,
    LongitudinalSensitivitySpec, LongitudinalSubject, compile_longitudinal_joint_sensitivity,
    consume_longitudinal_joint_sensitivity,
};
use antecedent::reason::reason_code;
use antecedent::reason::reason_details::longitudinal_joint_sensitivity as details;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::PyCancellationToken;
use crate::bridge::RefusalJson;
use crate::chain_decision_api::context;

type Refused<T> = (Option<T>, Option<String>);
type RawSubject = (u64, u8, [u8; 2], u8, Option<f64>);

fn refusal(error: &JointSensitivityError) -> String {
    let (code, detail) = match error {
        JointSensitivityError::Positivity | JointSensitivityError::RefitPositivity(_) => {
            (reason_code!("positivity_violated"), details::POSITIVITY)
        }
        JointSensitivityError::IncoherentBox => {
            (reason_code!("invalid_argument"), details::INCOHERENT_BOX)
        }
        JointSensitivityError::Cancelled => {
            (reason_code!("cancelled_no_claim"), details::CANCELLED)
        }
        JointSensitivityError::Resource(_) => {
            (reason_code!("route_not_supported"), details::RESOURCES)
        }
        JointSensitivityError::Identity => {
            (reason_code!("invalid_argument"), details::IDENTITY)
        }
        _ => {
            (reason_code!("invalid_argument"), details::INVALID_DECLARATION)
        }
    };
    RefusalJson::new(code, "longitudinal_joint_sensitivity", detail, &error.to_string()).json()
}

fn checked<Input, Output>(
    result: Result<Input, JointSensitivityError>,
    wrap: impl FnOnce(Input) -> Output,
) -> Refused<Output> {
    match result {
        Ok(value) => (Some(wrap(value)), None),
        Err(error) => (None, Some(refusal(&error))),
    }
}

#[pyclass(name = "LongitudinalJointSensitivityPlanHandle", skip_from_py_object)]
pub(crate) struct PyLongitudinalJointSensitivityPlan {
    plan: CompiledLongitudinalJointSensitivity,
}

#[pyclass(name = "LongitudinalJointSensitivityResultHandle", skip_from_py_object)]
pub(crate) struct PyLongitudinalJointSensitivityResult {
    result: LongitudinalSensitivityArtifact,
}

#[pymethods]
impl PyLongitudinalJointSensitivityPlan {
    #[getter]
    fn identity(&self) -> String {
        self.plan.identity().to_hex()
    }

    fn plan_view_json(&self) -> String {
        self.plan.plan_view().to_json().to_string()
    }

    fn unit_support(&self) -> Option<String> {
        crate::errors::unit_support_json(&self.plan.plan_view())
    }

    #[pyo3(signature = (*, seed=1, threads=None, cancel=None, memory_bytes=None))]
    fn execute(
        &self,
        py: Python<'_>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<PyCancellationToken>,
        memory_bytes: Option<u64>,
    ) -> PyResult<Refused<PyLongitudinalJointSensitivityResult>> {
        let ctx = context(seed, threads, cancel, memory_bytes);
        let plan = &self.plan;
        let result = crate::detach_catch(py, move || Ok(plan.execute(&ctx)))?;
        Ok(checked(result, |result| PyLongitudinalJointSensitivityResult { result }))
    }
}

#[pymethods]
impl PyLongitudinalJointSensitivityResult {
    #[getter]
    fn identity(&self) -> String {
        self.result.identity().to_hex()
    }

    #[getter]
    fn program_identity(&self) -> &str {
        self.result.program_identity()
    }

    #[getter]
    fn calibration_status(&self) -> &'static str {
        self.result.calibration_status()
    }

    fn calibration_json(&self) -> PyResult<String> {
        serde_json::to_string(&self.result.calibration()).map_err(crate::value_err_display)
    }

    fn report_json(&self) -> PyResult<String> {
        serde_json::to_string(self.result.report()).map_err(crate::value_err_display)
    }

    fn export<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyBytes>> {
        self.result.to_bytes().map(|data| PyBytes::new(py, &data)).map_err(crate::value_err_display)
    }
}

#[pyfunction]
fn longitudinal_joint_sensitivity_compile(
    py: Python<'_>,
    declaration: &str,
    rows: Vec<RawSubject>,
) -> PyResult<Refused<PyLongitudinalJointSensitivityPlan>> {
    crate::bridge::check_declared_size(declaration, 8192, "joint sensitivity declaration")?;
    let spec: LongitudinalSensitivitySpec =
        serde_json::from_str(declaration).map_err(crate::value_err_display)?;
    let subjects = rows
        .into_iter()
        .map(|(subject, baseline, treatment, intermediate, outcome)| LongitudinalSubject {
            subject,
            baseline,
            treatment,
            intermediate,
            outcome,
        })
        .collect::<Vec<_>>();
    let result = crate::detach_catch(py, move || {
        let ctx = context(1, None, None, None);
        Ok(compile_longitudinal_joint_sensitivity(spec, subjects, &ctx))
    })?;
    Ok(checked(result, |plan| PyLongitudinalJointSensitivityPlan { plan }))
}

#[pyfunction]
#[pyo3(signature = (data, expected, expected_program, *, seed=1, threads=None, cancel=None, memory_bytes=None))]
#[allow(clippy::too_many_arguments, reason = "retained identity pair and execution controls")]
fn longitudinal_joint_sensitivity_consume(
    py: Python<'_>,
    data: &[u8],
    expected: &str,
    expected_program: &str,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> PyResult<Refused<PyLongitudinalJointSensitivityResult>> {
    let expected = crate::digest::digest(expected, "longitudinal sensitivity result")?;
    let expected_program =
        crate::digest::digest(expected_program, "longitudinal sensitivity program")?;
    let ctx = context(seed, threads, cancel, memory_bytes);
    let result = crate::detach_catch(py, move || {
        Ok(consume_longitudinal_joint_sensitivity(data, expected, expected_program, &ctx))
    })?;
    Ok(checked(result, |result| PyLongitudinalJointSensitivityResult { result }))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyLongitudinalJointSensitivityPlan>()?;
    module.add_class::<PyLongitudinalJointSensitivityResult>()?;
    module.add_function(wrap_pyfunction!(longitudinal_joint_sensitivity_compile, module)?)?;
    module.add_function(wrap_pyfunction!(longitudinal_joint_sensitivity_consume, module)?)
}
