//! Python bindings for compiled intervention schedule decisions.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::composition::BundleLimits;
use antecedent::scientific::SemanticDigest;
use pyo3::prelude::*;
use pyo3::types::PyBytes;

use crate::PyCancellationToken;
use crate::chain_decision_api::context;

fn integrated_schedule_error(error: &antecedent::design::ScheduleBundleError) -> String {
    use antecedent::design::{ScheduleBundleError as B, ScheduleDecisionError as D};
    use antecedent::reason::reason_code;
    let (code, detail) = match error {
        B::Procedure(D::MissingJointEvidence) | B::MissingParent => (
            reason_code!("route_not_supported"),
            antecedent::reason::reason_details::integrated_schedule::MISSING_JOINT_EVIDENCE,
        ),
        B::Procedure(D::InadequateOverlap) => (
            reason_code!("route_not_supported"),
            antecedent::reason::reason_details::integrated_schedule::INADEQUATE_OVERLAP,
        ),
        B::Procedure(D::InvalidDeclaration) => (
            reason_code!("invalid_argument"),
            antecedent::reason::reason_details::integrated_schedule::INVALID_DECLARATION,
        ),
        B::Procedure(D::Cancelled) => (
            reason_code!("cancelled_no_claim"),
            antecedent::reason::reason_details::integrated_schedule::CANCELLED,
        ),
        B::Resource(_) => (
            reason_code!("route_not_supported"),
            antecedent::reason::reason_details::integrated_schedule::RESOURCE_REFUSED,
        ),
        B::Unsupported(_) => (
            reason_code!("route_not_supported"),
            antecedent::reason::reason_details::integrated_schedule::UNSUPPORTED,
        ),
        B::ChangedProgram => (
            reason_code!("invalid_argument"),
            antecedent::reason::reason_details::integrated_schedule::CHANGED_PROGRAM,
        ),
        B::Bundle(_) | B::Io(_) => (
            reason_code!("invalid_argument"),
            antecedent::reason::reason_details::integrated_schedule::ARTIFACT_REFUSED,
        ),
    };
    crate::bridge::RefusalJson::new(code, "integrated_schedule", detail, &error.to_string()).json()
}

fn integrated_decision_json(
    decision: &antecedent::design::IntegratedScheduleDecision,
) -> PyResult<String> {
    let mut result = serde_json::to_value(decision).map_err(crate::value_err_display)?;
    result["calibration"] =
        serde_json::to_value(decision.calibration()).map_err(crate::value_err_display)?;
    Ok(result.to_string())
}

#[pyclass(name = "IntegratedSchedulePlanHandle", skip_from_py_object)]
pub(crate) struct PyIntegratedSchedulePlan {
    plan: antecedent::design::CompiledIntegratedScheduleDecision,
}

type ScheduleResult<T> = (Option<T>, Option<String>);
type ScheduleArtifact<'py> = (Bound<'py, PyBytes>, String, String);

#[pymethods]
impl PyIntegratedSchedulePlan {
    #[getter]
    fn program_identity(&self) -> String {
        self.plan.lowered().identity().to_hex()
    }

    #[getter]
    fn child_program_identities(&self) -> [String; 3] {
        self.plan.child_program_identities().map(|id| SemanticDigest::from_bytes(id).to_hex())
    }

    fn plan_view_json(&self) -> String {
        self.plan.plan_view().to_json().to_string()
    }

    fn unit_support(&self) -> Option<String> {
        crate::errors::unit_support_json(&self.plan.plan_view())
    }

    #[pyo3(signature = (*, seed=1, threads=None, cancel=None, memory_bytes=None))]
    fn execute<'py>(
        &self,
        py: Python<'py>,
        seed: u64,
        threads: Option<u32>,
        cancel: Option<PyCancellationToken>,
        memory_bytes: Option<u64>,
    ) -> PyResult<ScheduleResult<ScheduleArtifact<'py>>> {
        let ctx = context(seed, threads, cancel, memory_bytes);
        let plan = &self.plan;
        let result = crate::detach_catch(py, move || {
            let result = match plan.execute(&ctx) {
                Ok(value) => value,
                Err(error) => return Ok(Err(integrated_schedule_error(&error))),
            };
            let bytes = match result
                .bundle
                .to_bytes("integrated-schedule-decision", &BundleLimits::default())
            {
                Ok(bytes) => bytes,
                Err(error) => return Ok(Err(integrated_schedule_error(&error.into()))),
            };
            Ok(Ok((
                bytes,
                result.bundle.identity().to_owned(),
                integrated_decision_json(&result.decision)?,
            )))
        })?;
        Ok(match result {
            Ok((bytes, identity, decision)) => {
                (Some((PyBytes::new(py, &bytes), identity, decision)), None)
            }
            Err(error) => (None, Some(error)),
        })
    }
}

#[pyfunction]
#[pyo3(signature = (source_json, target, declaration_json, *, seed=1, threads=None, cancel=None, memory_bytes=None))]
#[allow(clippy::too_many_arguments)]
fn integrated_schedule_decision_compile(
    py: Python<'_>,
    source_json: &str,
    target: Vec<bool>,
    declaration_json: &str,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> PyResult<ScheduleResult<PyIntegratedSchedulePlan>> {
    use antecedent::design::{
        ScheduleBundleError as B, ScheduleDecisionDeclaration, ScheduleDecisionError as D,
        ScheduleSourceSubject, compile_integrated_schedule_decision,
    };
    let refused = |error| Ok((None, Some(integrated_schedule_error(&error))));
    if source_json.len() > 80 * 1024 * 1024 || declaration_json.len() > 16 * 1024 {
        return refused(B::Procedure(D::InvalidDeclaration));
    }
    let ctx = context(seed, threads, cancel, memory_bytes);
    crate::detach_catch(py, move || {
        let _lease = match ctx.acquire_admission(0, source_json.len() as u64) {
            Ok(lease) => lease,
            Err(_) if ctx.cancellation.is_cancelled() => {
                return refused(B::Procedure(D::Cancelled));
            }
            Err(error) => return refused(B::Resource(error.to_string())),
        };
        let invalid = |error: serde_json::Error| {
            crate::bridge::RefusalJson::new(
                antecedent::reason::reason_code!("invalid_argument"),
                "integrated_schedule",
                antecedent::reason::reason_details::integrated_schedule::INVALID_DECLARATION,
                &error.to_string(),
            )
            .json()
        };
        let source: Vec<ScheduleSourceSubject> = match serde_json::from_str(source_json) {
            Ok(value) => value,
            Err(error) => return Ok((None, Some(invalid(error)))),
        };
        let declaration: ScheduleDecisionDeclaration = match serde_json::from_str(declaration_json)
        {
            Ok(value) => value,
            Err(error) => return Ok((None, Some(invalid(error)))),
        };
        match compile_integrated_schedule_decision(source, target, declaration, &ctx) {
            Ok(plan) => Ok((Some(PyIntegratedSchedulePlan { plan }), None)),
            Err(error) => refused(error),
        }
    })
}

#[pyfunction]
#[pyo3(signature = (bytes, expected_identity, *, seed=1, threads=None, cancel=None, memory_bytes=None))]
#[allow(clippy::too_many_arguments)]
fn integrated_schedule_decision_consume(
    py: Python<'_>,
    bytes: &[u8],
    expected_identity: &str,
    seed: u64,
    threads: Option<u32>,
    cancel: Option<PyCancellationToken>,
    memory_bytes: Option<u64>,
) -> PyResult<ScheduleResult<String>> {
    let ctx = context(seed, threads, cancel, memory_bytes);
    crate::detach_catch(py, || {
        match antecedent::design::consume_schedule_decision_bundle_with_context(
            bytes,
            expected_identity,
            &ctx,
        ) {
            Ok(decision) => Ok((Some(integrated_decision_json(&decision)?), None)),
            Err(_) if ctx.cancellation.is_cancelled() => Ok((
                None,
                Some(integrated_schedule_error(
                    &antecedent::design::ScheduleDecisionError::Cancelled.into(),
                )),
            )),
            Err(error) => Ok((None, Some(integrated_schedule_error(&error)))),
        }
    })
}

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyIntegratedSchedulePlan>()?;
    module.add_function(wrap_pyfunction!(integrated_schedule_decision_compile, module)?)?;
    module.add_function(wrap_pyfunction!(integrated_schedule_decision_consume, module)?)?;
    Ok(())
}
