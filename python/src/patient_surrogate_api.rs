//! Typed Python FFI for checked patient-level surrogate identification.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::Study;
use antecedent::analysis::kernel_programs::patient_surrogate::{
    CompiledPatientSurrogate, PatientSurrogateError as E, PatientSurrogateProgramError as P,
    PatientSurrogateRequest, PatientSurrogateResult, PrincipalSurrogateError,
    consume_patient_surrogate,
};
use antecedent::reason::reason_code;
use antecedent::reason::reason_details::patient_surrogate as details;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde_json::json;

use crate::PyCancellationToken;
use crate::bridge::RefusalJson;
use crate::chain_decision_api::context;

type Checked<T> = (Option<T>, Option<String>);

fn refusal(code: &str, detail: &str, error: impl ToString) -> String {
    RefusalJson::new(code, "patient_surrogate", detail, &error.to_string()).json()
}

fn scientific(error: &E) -> String {
    let (code, detail) = match error {
        E::Cancelled => (reason_code!("cancelled_no_claim"), details::CANCELLED),
        E::Resource => (reason_code!("route_not_supported"), details::RESOURCE),
        E::Identification(PrincipalSurrogateError::Monotonicity) => {
            (reason_code!("invalid_argument"), details::MONOTONICITY)
        }
        E::Identification(PrincipalSurrogateError::EmptyImproverStratum) => {
            (reason_code!("invalid_argument"), details::EMPTY_STRATUM)
        }
        E::DecisionReading | E::PointNotIdentified => {
            (reason_code!("route_not_supported"), details::UNSUPPORTED_READING)
        }
        E::Artifact(_) | E::ReplayMismatch => {
            (reason_code!("invalid_argument"), details::REPLAY)
        }
        _ => (reason_code!("invalid_argument"), details::DECLARATION),
    };
    refusal(code, detail, error)
}

fn program(error: &P) -> String {
    match error {
        P::Scientific(error) => scientific(error),
        P::Support(error) => refusal(error.code(), details::NOT_SUPPORTED, error),
        P::Compile(error) => {
            refusal(reason_code!("invalid_argument"), details::DECLARATION, error)
        }
    }
}

fn checked<T, E>(value: Result<T, E>, refusal: impl Fn(&E) -> String) -> Checked<T> {
    match value {
        Ok(value) => (Some(value), None),
        Err(error) => (None, Some(refusal(&error))),
    }
}

#[pyclass(name = "PatientSurrogatePlanHandle", skip_from_py_object)]
pub(crate) struct PyPatientSurrogatePlan {
    plan: CompiledPatientSurrogate,
}

#[pyclass(name = "PatientSurrogateResultHandle", skip_from_py_object)]
pub(crate) struct PyPatientSurrogateResult {
    result: PatientSurrogateResult,
}

#[pymethods]
impl PyPatientSurrogatePlan {
    #[getter]
    fn program_identity(&self) -> String {
        self.plan.lowered().identity().to_hex()
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
    ) -> PyResult<Checked<PyPatientSurrogateResult>> {
        let plan = &self.plan;
        let ctx = context(seed, threads, cancel, memory_bytes);
        let result = crate::detach_catch(py, move || Ok(plan.execute(&ctx)))?;
        Ok(checked(result.map(|result| PyPatientSurrogateResult { result }), program))
    }

    #[pyo3(signature = (artifact_id, *, seed=1, threads=None, cancel=None, memory_bytes=None))]
    fn export<'py>(
        &self,
        py: Python<'py>,
        artifact_id: &str,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<PyCancellationToken>,
        memory_bytes: Option<u64>,
    ) -> PyResult<Checked<(Bound<'py, PyBytes>, String)>> {
        let plan = &self.plan;
        let ctx = context(seed, threads, cancel, memory_bytes);
        let result = crate::detach_catch(py, move || Ok(plan.export_with_identity(artifact_id, &ctx)))?;
        Ok(checked(result.map(|(bytes, identity)| (PyBytes::new(py, &bytes), identity.to_hex())), program))
    }
}

#[pymethods]
impl PyPatientSurrogateResult {
    #[getter]
    fn identity(&self) -> String {
        self.result.identity().to_hex()
    }

    #[getter]
    fn source_identity(&self) -> String {
        self.result.source_identity().to_hex()
    }

    #[getter]
    fn reading(&self) -> &'static str {
        self.result.reading().as_str()
    }

    #[getter]
    fn bounds(&self) -> (f64, f64) {
        let [lo, hi] = self.result.bounds().endpoints();
        (lo, hi)
    }

    #[getter]
    fn exact_bounds(&self) -> (u64, u64, u64) {
        let bounds = self.result.bounds();
        (bounds.lower_numerator, bounds.upper_numerator, bounds.improver_numerator)
    }

    #[getter]
    fn improver_probability(&self) -> f64 {
        self.result.bounds().improver_probability()
    }

    fn quantity_json(&self) -> String {
        let wire =
            antecedent::io::quantity_wire::ScientificQuantityWire::from(self.result.quantity());
        json!(wire).to_string()
    }

    fn point(&self) -> Checked<f64> {
        checked(self.result.point(), scientific)
    }

    fn decide(&self, numerator: u64, denominator: u64) -> Checked<String> {
        checked(
            self.result.decide(numerator, denominator).map(|decision| {
                json!({
                    "reading": decision.reading,
                    "identity": decision.identity.to_hex(),
                    "source_identity": decision.source.to_hex(),
                })
                .to_string()
            }),
            scientific,
        )
    }
}

#[pyfunction]
#[pyo3(signature = (request, *, seed=1, threads=None, cancel=None, memory_bytes=None))]
fn patient_surrogate_compile(
    py: Python<'_>,
    request: &str,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> PyResult<Checked<PyPatientSurrogatePlan>> {
    crate::bridge::check_declared_size(request, 16 * 1024 * 1024, "patient surrogate declaration")?;
    let ctx = context(seed, threads, cancel, memory_bytes);
    crate::detach_catch(py, move || {
        let request: PatientSurrogateRequest = match serde_json::from_str(request) {
            Ok(value) => value,
            Err(error) => {
                return Ok((
                    None,
                    Some(refusal(
                        reason_code!("invalid_argument"),
                        details::DECLARATION,
                        error,
                    )),
                ));
            }
        };
        Ok(checked(
            Study::compile_patient_surrogate(request, &ctx)
                .map(|plan| PyPatientSurrogatePlan { plan }),
            program,
        ))
    })
}

#[pyfunction]
#[pyo3(signature = (data, expected_program, expected_result, *, seed=1, threads=None, cancel=None, memory_bytes=None))]
#[allow(clippy::too_many_arguments, reason = "expected identities and execution controls")]
fn patient_surrogate_consume(
    py: Python<'_>,
    data: &[u8],
    expected_program: &str,
    expected_result: &str,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> PyResult<Checked<PyPatientSurrogateResult>> {
    let expected_program = crate::digest::digest(expected_program, "expected_program")?;
    let expected_result = crate::digest::digest(expected_result, "expected_result")?;
    let ctx = context(seed, threads, cancel, memory_bytes);
    let result = crate::detach_catch(py, move || {
        Ok(consume_patient_surrogate(data, expected_program, expected_result, &ctx))
    })?;
    Ok(checked(result.map(|result| PyPatientSurrogateResult { result }), program))
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyPatientSurrogatePlan>()?;
    module.add_class::<PyPatientSurrogateResult>()?;
    module.add_function(wrap_pyfunction!(patient_surrogate_compile, module)?)?;
    module.add_function(wrap_pyfunction!(patient_surrogate_consume, module)?)
}
