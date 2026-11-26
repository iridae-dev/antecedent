//! Owned, inspectable native cohort fit plans for Python.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::Study;
use antecedent::analysis::compiled_native_cohort_fit::CompiledNativeCohortFit;
use antecedent::estimate::EstimationError;
use antecedent::execution::AdmissionRefusal;
use antecedent::reason::{reason_code, reason_details};
use pyo3::prelude::*;

use super::{NativeFitJson, check_size, fit_quantities, invalid_argument, native_fit_payload};
use crate::PyCancellationToken;
use crate::bridge::RefusalJson;
use crate::chain_decision_api::context;

type Checked<T> = (Option<T>, Option<String>);
type FitResult = (String, Vec<u8>, String);

fn refusal_json(code: &str, stage: &str, message: &str) -> String {
    RefusalJson::new(code, stage, reason_details::cohort_decision::NATIVE_FIT_REFUSED, message)
        .json()
}

fn resource_refusal(error: AdmissionRefusal, stage: &str) -> String {
    let code = if error.is_cancelled() {
        reason_code!("cancelled_no_claim")
    } else {
        reason_code!("execution_resource_limit")
    };
    refusal_json(code, stage, &error.to_string())
}

fn refusal(error: &EstimationError, stage: &str) -> String {
    if let EstimationError::Admission(admission) = error {
        return resource_refusal((*admission).into(), stage);
    }
    refusal_json(reason_code!("invalid_argument"), stage, &error.to_string())
}

#[pyclass(name = "NativeCohortFitPlanHandle", skip_from_py_object)]
struct PyNativeCohortFitPlan {
    plan: CompiledNativeCohortFit<'static>,
}

#[pymethods]
impl PyNativeCohortFitPlan {
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
    ) -> PyResult<Checked<FitResult>> {
        let ctx = context(seed, threads, cancel, memory_bytes);
        crate::detach_catch(py, || match self.plan.execute(&ctx) {
            Ok(run) => {
                let (summary, artifact) = native_fit_payload(&run.fit)?;
                Ok((Some((summary, artifact, run.lowered_program_identity.to_hex())), None))
            }
            Err(error) => Ok((None, Some(refusal(&error, "execute")))),
        })
    }
}

#[pyfunction]
#[pyo3(signature = (fit_json, *, seed=1, threads=None, cancel=None, memory_bytes=None))]
fn compile_native_cohort_fit(
    py: Python<'_>,
    fit_json: &str,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> PyResult<Checked<PyNativeCohortFitPlan>> {
    check_size(fit_json, "native fit")?;
    let ctx = context(seed, threads, cancel, memory_bytes);
    crate::detach_catch(py, || {
        let _intake = match ctx.acquire_admission(0, (fit_json.len() as u64).saturating_mul(4)) {
            Ok(lease) => lease,
            Err(error) => return Ok((None, Some(resource_refusal(error.into(), "compile")))),
        };
        let parsed: NativeFitJson = serde_json::from_str(fit_json).map_err(|error| {
            invalid_argument(format!("cohort_decision.invalid_request: {error}"))
        })?;
        let (response, quantities) = match fit_quantities(&parsed) {
            Ok(quantities) => quantities,
            Err(message) => return Ok((None, Some(super::native_fit_refusal(&message)))),
        };
        let plan = Study::compile_owned_native_cohort_fit(
            parsed.outcome,
            parsed.treatment,
            parsed.features,
            response,
            quantities,
        );
        if ctx.cancellation.is_cancelled() {
            return Ok((None, Some(resource_refusal(AdmissionRefusal::Cancelled, "compile"))));
        }
        Ok(match plan {
            Ok(plan) => (Some(PyNativeCohortFitPlan { plan }), None),
            Err(error) => (None, Some(refusal(&error, "compile"))),
        })
    })
}

pub(super) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyNativeCohortFitPlan>()?;
    m.add_function(wrap_pyfunction!(compile_native_cohort_fit, m)?)
}
