// SPDX-License-Identifier: MIT OR Apache-2.0
//! Registered external cyclic mechanisms: Python callable registry bridge.
//!
//! A Python implementation is a process-local object; it is never serialized. The portable
//! record carries the implementation identity and content digest only. Replay takes a
//! caller-supplied resolver callable that maps a recorded specification (a dict) to an
//! implementation object, and the native layer verifies the object against the record
//! before running it.
use antecedent_core::ExecutionContext;
use antecedent_estimate::external_cyclic_fit::{fit_external_feedback, ExternalFitPolicy};
use antecedent_io::external_cyclic_artifact::{
    consume_external_cyclic_artifact, inspect_external_cyclic_artifact, ExternalCyclicRequest,
    MAX_EXTERNAL_CYCLIC_ARTIFACT_BYTES,
};
use antecedent_model::continuous_cyclic::{OdePolicy, TimedClamp};
use antecedent_model::external_mechanism::{
    ExternalFeedbackSystem, ExternalFitParameter, ExternalMechanismImplementation,
    ExternalMechanismSpec, ExternalOdeSystem, TrustedMechanismResolver,
};
use antecedent_model::nonlinear_equilibrium::{EquilibriumError, RegimeRow};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyBytesMethods};
use std::sync::Arc;

fn execution_context(
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> ExecutionContext {
    let mut ctx = crate::py_execution_context_cancel(0, 1, cancel);
    if max_bytes.is_some() {
        ctx.set_memory_limit(max_bytes);
    }
    ctx
}

fn operation_error(error: impl std::error::Error + 'static, ctx: &ExecutionContext) -> PyErr {
    let failure = antecedent_core::CallbackFailure::in_chain(&error).cloned();
    let converted = classified(&error, ctx);
    match &failure {
        Some(failure) => crate::interrupt::with_callback_failure(converted, failure),
        None => converted,
    }
}

fn classified(error: &(dyn std::error::Error + 'static), ctx: &ExecutionContext) -> PyErr {
    if ctx.cancellation.is_cancelled() {
        return crate::CausalCancelledError::new_err(error.to_string());
    }
    let mut cause: &(dyn std::error::Error + 'static) = error;
    loop {
        if cause.is::<antecedent_core::execution::AdmissionError>()
            || matches!(
                cause.downcast_ref::<EquilibriumError>(),
                Some(EquilibriumError::Resource(_))
            )
        {
            return crate::CausalResourceError::new_err(error.to_string());
        }
        match cause.source() {
            Some(source) => cause = source,
            // A typed refusal's message opens with its registered `reason=<code>:`.
            None => return crate::value_err(error.to_string()),
        }
    }
}

fn input_admission(
    json: &str,
    ctx: &ExecutionContext,
) -> PyResult<antecedent_core::execution::AdmissionLease> {
    if json.len() > MAX_EXTERNAL_CYCLIC_ARTIFACT_BYTES {
        return Err(crate::value_err("external cyclic input byte limit"));
    }
    if ctx.cancellation.is_cancelled() {
        return Err(crate::CausalCancelledError::new_err("cancelled"));
    }
    ctx.acquire_admission(0, (json.len() as u64).saturating_mul(4))
        .map_err(|e| operation_error(e, ctx))
}

/// A Python object realizing a registered mechanism. The object stays in the process.
struct PythonImplementation {
    object: Py<PyAny>,
    spec: ExternalMechanismSpec,
    content: Vec<u8>,
}
impl PythonImplementation {
    fn from_object(object: &Bound<'_, PyAny>) -> PyResult<Self> {
        let spec_json: String = object.getattr("spec_json")?.extract()?;
        let spec: ExternalMechanismSpec =
            serde_json::from_str(&spec_json).map_err(|e| crate::value_err(e.to_string()))?;
        let content = object.getattr("content")?.cast::<PyBytes>()?.as_bytes().to_vec();
        Ok(Self { object: object.clone().unbind(), spec, content })
    }
    fn call(
        &self,
        method: &str,
        inputs: &[f64],
        parameters: &[f64],
    ) -> Result<Option<Vec<f64>>, EquilibriumError> {
        Python::attach(|py| {
            let result = self
                .object
                .bind(py)
                .call_method1(method, (inputs.to_vec(), parameters.to_vec()))?;
            if result.is_none() {
                return Ok(None);
            }
            result.extract::<Vec<f64>>().map(Some)
        })
        .map_err(|error| EquilibriumError::CallbackFailed {
            mechanism: self.spec.implementation_id.clone(),
            failure: crate::interrupt::callback_failure("external mechanism raised", error),
        })
    }
}
impl ExternalMechanismImplementation for PythonImplementation {
    fn spec(&self) -> ExternalMechanismSpec {
        self.spec.clone()
    }
    fn content(&self) -> &[u8] {
        &self.content
    }
    fn value(&self, inputs: &[f64], parameters: &[f64]) -> Result<f64, EquilibriumError> {
        Python::attach(|py| {
            self.object
                .bind(py)
                .call_method1("value", (inputs.to_vec(), parameters.to_vec()))?
                .extract::<f64>()
        })
        .map_err(|error| EquilibriumError::CallbackFailed {
            mechanism: self.spec.implementation_id.clone(),
            failure: crate::interrupt::callback_failure("external mechanism raised", error),
        })
    }
    fn input_gradient(
        &self,
        inputs: &[f64],
        parameters: &[f64],
    ) -> Result<Option<Vec<f64>>, EquilibriumError> {
        self.call("input_gradient", inputs, parameters)
    }
    fn parameter_gradient(
        &self,
        inputs: &[f64],
        parameters: &[f64],
    ) -> Result<Option<Vec<f64>>, EquilibriumError> {
        self.call("parameter_gradient", inputs, parameters)
    }
    fn input_gradient_parameter_derivative(
        &self,
        inputs: &[f64],
        parameters: &[f64],
    ) -> Result<Option<Vec<Vec<f64>>>, EquilibriumError> {
        Python::attach(|py| {
            let result = self.object.bind(py).call_method1(
                "input_gradient_parameter_derivative",
                (inputs.to_vec(), parameters.to_vec()),
            )?;
            if result.is_none() {
                return Ok(None);
            }
            result.extract::<Vec<Vec<f64>>>().map(Some)
        })
        .map_err(|error| EquilibriumError::CallbackFailed {
            mechanism: self.spec.implementation_id.clone(),
            failure: crate::interrupt::callback_failure("external mechanism raised", error),
        })
    }
}

/// The caller's trusted resolver callable: `resolver(spec: dict) -> implementation | None`.
struct PythonResolver {
    resolver: Py<PyAny>,
}
impl TrustedMechanismResolver for PythonResolver {
    fn resolve(
        &self,
        recorded: &ExternalMechanismSpec,
    ) -> Option<Arc<dyn ExternalMechanismImplementation>> {
        self.resolve_checked(recorded).ok().flatten()
    }
    /// A resolver that raises, or returns an object that is not an implementation, is a
    /// resolver failure; only a `None` answer means the implementation is unavailable.
    fn resolve_checked(
        &self,
        recorded: &ExternalMechanismSpec,
    ) -> Result<Option<Arc<dyn ExternalMechanismImplementation>>, String> {
        Python::attach(|py| -> PyResult<Option<Arc<dyn ExternalMechanismImplementation>>> {
            let text = serde_json::to_string(recorded)
                .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
            let spec = py.import("json")?.call_method1("loads", (text,))?;
            let found = self.resolver.bind(py).call1((spec,))?;
            if found.is_none() {
                return Ok(None);
            }
            let implementation = PythonImplementation::from_object(&found)?;
            Ok(Some(Arc::new(implementation) as Arc<dyn ExternalMechanismImplementation>))
        })
        .map_err(|error| error.to_string())
    }
}

fn resolver_of(resolver: Option<Py<PyAny>>) -> Option<PythonResolver> {
    resolver.map(|resolver| PythonResolver { resolver })
}

/// BLAKE3 digest of implementation content, as registered in a specification.
#[pyfunction]
fn external_content_digest(content: &[u8]) -> String {
    antecedent_core::blake3_hex(content)
}

/// Validate a specification and return its identity.
#[pyfunction]
fn external_spec_identity(spec_json: &str) -> PyResult<String> {
    let spec: ExternalMechanismSpec =
        serde_json::from_str(spec_json).map_err(|e| crate::value_err(e.to_string()))?;
    spec.validate().map_err(|e| crate::value_err(e.to_string()))?;
    Ok(spec.identity())
}

/// Identity of a retained request.
#[pyfunction]
fn external_request_identity(request_json: &str) -> PyResult<String> {
    let request: ExternalCyclicRequest =
        serde_json::from_str(request_json).map_err(|e| crate::value_err(e.to_string()))?;
    request.identity().map_err(|e| crate::value_err(e.to_string()))
}

/// Verify, fit and evaluate a request; the report as JSON.
#[pyfunction]
#[pyo3(signature = (request_json, resolver=None, *, max_bytes=None, cancel=None))]
fn external_execute(
    py: Python<'_>,
    request_json: &str,
    resolver: Option<Py<PyAny>>,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let ctx = execution_context(max_bytes, cancel);
    let _input = input_admission(request_json, &ctx)?;
    let request: ExternalCyclicRequest =
        serde_json::from_str(request_json).map_err(|e| operation_error(e, &ctx))?;
    let resolver = resolver_of(resolver);
    crate::detach_catch(py, move || {
        let report = request
            .execute(resolver.as_ref().map(|r| r as &dyn TrustedMechanismResolver), &ctx)
            .map_err(|e| operation_error(e, &ctx))?;
        serde_json::to_string(&report).map_err(|e| crate::serialization_err(e.to_string()))
    })
}

/// Execute and encode the portable artifact; `(report_json, artifact_bytes)`.
#[pyfunction]
#[pyo3(signature = (request_json, artifact_id, resolver=None, *, max_bytes=None, cancel=None))]
fn external_export(
    py: Python<'_>,
    request_json: &str,
    artifact_id: &str,
    resolver: Option<Py<PyAny>>,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<(String, Py<PyBytes>)> {
    let ctx = execution_context(max_bytes, cancel);
    let _input = input_admission(request_json, &ctx)?;
    let request: ExternalCyclicRequest =
        serde_json::from_str(request_json).map_err(|e| operation_error(e, &ctx))?;
    let resolver = resolver_of(resolver);
    let artifact_id = artifact_id.to_owned();
    let (report, bytes) = crate::detach_catch(py, move || {
        let (report, bytes) = request
            .export(
                &artifact_id,
                resolver.as_ref().map(|r| r as &dyn TrustedMechanismResolver),
                &ctx,
            )
            .map_err(|e| operation_error(e, &ctx))?;
        let report =
            serde_json::to_string(&report).map_err(|e| crate::serialization_err(e.to_string()))?;
        Ok((report, bytes))
    })?;
    Ok((report, PyBytes::new(py, &bytes).unbind()))
}

/// Independently replay an artifact against its retained identity; the report as JSON.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity, resolver=None, *, max_bytes=None, cancel=None))]
fn external_consume(
    py: Python<'_>,
    artifact: &[u8],
    expected_identity: &str,
    resolver: Option<Py<PyAny>>,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let ctx = execution_context(max_bytes, cancel);
    let artifact = artifact.to_vec();
    let expected_identity = expected_identity.to_owned();
    let resolver = resolver_of(resolver);
    crate::detach_catch(py, move || {
        let report = consume_external_cyclic_artifact(
            &artifact,
            &expected_identity,
            resolver.as_ref().map(|r| r as &dyn TrustedMechanismResolver),
            &ctx,
        )
        .map_err(|e| operation_error(e, &ctx))?;
        serde_json::to_string(&report).map_err(|e| crate::serialization_err(e.to_string()))
    })
}

/// Recorded mechanism identities of an artifact, without a resolver or any execution.
#[pyfunction]
fn external_inspect(py: Python<'_>, artifact: &[u8]) -> PyResult<String> {
    let artifact = artifact.to_vec();
    crate::detach_catch(py, move || {
        let inspection = inspect_external_cyclic_artifact(&artifact)
            .map_err(|e| crate::value_err(e.to_string()))?;
        serde_json::to_string(&inspection).map_err(|e| crate::serialization_err(e.to_string()))
    })
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct FitRequest {
    system: ExternalFeedbackSystem,
    parameters: Vec<ExternalFitParameter>,
    rows: Vec<RegimeRow>,
    policy: ExternalFitPolicy,
}

/// Fit the declared scalars of a system; the fit as JSON.
#[pyfunction]
#[pyo3(signature = (request_json, resolver=None, *, max_bytes=None, cancel=None))]
fn external_fit(
    py: Python<'_>,
    request_json: &str,
    resolver: Option<Py<PyAny>>,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let ctx = execution_context(max_bytes, cancel);
    let _input = input_admission(request_json, &ctx)?;
    let request: FitRequest =
        serde_json::from_str(request_json).map_err(|e| operation_error(e, &ctx))?;
    let resolver = resolver_of(resolver);
    crate::detach_catch(py, move || {
        let fit = fit_external_feedback(
            &request.system,
            resolver.as_ref().map(|r| r as &dyn TrustedMechanismResolver),
            &request.parameters,
            &request.rows,
            request.policy,
            &ctx,
        )
        .map_err(|e| operation_error(e, &ctx))?;
        serde_json::to_string(&fit).map_err(|e| crate::serialization_err(e.to_string()))
    })
}

#[derive(serde::Deserialize)]
#[serde(deny_unknown_fields)]
struct OdeRequest {
    system: ExternalOdeSystem,
    initial: Vec<f64>,
    times: Vec<f64>,
    schedule: Vec<TimedClamp>,
    policy: OdePolicy,
}

/// Integrate an ODE with registered external drift mechanisms; the trajectory as JSON.
#[pyfunction]
#[pyo3(signature = (request_json, resolver=None, *, max_bytes=None, cancel=None))]
fn external_ode(
    py: Python<'_>,
    request_json: &str,
    resolver: Option<Py<PyAny>>,
    max_bytes: Option<u64>,
    cancel: Option<crate::PyCancellationToken>,
) -> PyResult<String> {
    let ctx = execution_context(max_bytes, cancel);
    let _input = input_admission(request_json, &ctx)?;
    let request: OdeRequest =
        serde_json::from_str(request_json).map_err(|e| operation_error(e, &ctx))?;
    let resolver = resolver_of(resolver);
    crate::detach_catch(py, move || {
        let resolved = request
            .system
            .resolve(resolver.as_ref().map(|r| r as &dyn TrustedMechanismResolver))
            .map_err(|e| operation_error(e, &ctx))?;
        let trajectory = resolved
            .trajectory(&request.initial, &request.times, &request.schedule, request.policy, &ctx)
            .map_err(|e| operation_error(e, &ctx))?;
        serde_json::to_string(&trajectory).map_err(|e| crate::serialization_err(e.to_string()))
    })
}

pub(crate) fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(external_content_digest, m)?)?;
    m.add_function(wrap_pyfunction!(external_spec_identity, m)?)?;
    m.add_function(wrap_pyfunction!(external_request_identity, m)?)?;
    m.add_function(wrap_pyfunction!(external_execute, m)?)?;
    m.add_function(wrap_pyfunction!(external_export, m)?)?;
    m.add_function(wrap_pyfunction!(external_consume, m)?)?;
    m.add_function(wrap_pyfunction!(external_inspect, m)?)?;
    m.add_function(wrap_pyfunction!(external_fit, m)?)?;
    m.add_function(wrap_pyfunction!(external_ode, m)?)?;
    Ok(())
}
