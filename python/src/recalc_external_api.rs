//! Bounded executing external mean callbacks, retaining attested output only.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use crate::program_claims_api::{ClaimWire, claim_from, parse_binding};
use crate::recalc_api::{context, parse_json, plan_refusal_json, refusal_json};
use crate::recalc_bounds::check_columns;
use antecedent::analysis::recalc_external::{
    CallbackContext, CallbackFailure, CallbackPolicy, ExternalCallbackError,
    ExternalCallbackRequest, ExternalCallbackSession, ExternalMeanProvider, ProviderDescriptor,
};
use antecedent_core::recalc::Branch;
use antecedent_core::{ExternalResponse, ProviderObjectIdentity};
use antecedent_io::external_binding_wire::{
    ContractWire, ProviderWire, RefusalWire, ResponseWire, core_contract, response_from_wire,
};
use antecedent_io::external_claim_artifact::ExternalClaimArtifact;
use numpy::{PyArray1, PyReadonlyArray1};
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyString};
use pyo3::{PyTraverseError, PyVisit};
use serde::Deserialize;

fn invalid(detail: &str, message: impl AsRef<str>) -> PyErr {
    crate::with_reason_code(
        crate::value_err(format!("{detail}: {}", message.as_ref())),
        antecedent_core::reason_code!("invalid_argument"),
    )
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct DescriptorWire {
    provider: ProviderWire,
    environment_id: String,
    policy: String,
    idempotency_supported: bool,
}
fn descriptor(wire: DescriptorWire) -> PyResult<ProviderDescriptor> {
    let policy = match wire.policy.as_str() {
        "deterministic" => CallbackPolicy::Deterministic,
        "seeded" => CallbackPolicy::Seeded,
        "stateful" => CallbackPolicy::Stateful,
        "side_effecting" => CallbackPolicy::SideEffecting,
        "unknown" => CallbackPolicy::Unknown,
        _ => return Err(invalid("external_recalc.invalid_descriptor", "unknown callback policy")),
    };
    Ok(ProviderDescriptor {
        identity: ProviderObjectIdentity {
            provider_id: wire.provider.provider_id,
            object_id: wire.provider.object_id,
            version_id: wire.provider.version_id,
            snapshot_id: wire.provider.snapshot_id,
            request_id: wire.provider.request_id,
        },
        environment_id: wire.environment_id,
        policy,
        idempotency_supported: wire.idempotency_supported,
    })
}
#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestWire {
    program: serde_json::Value,
    claim: ClaimWire,
    contract: ContractWire,
    descriptor: DescriptorWire,
    model_parameters: Vec<(String, f64)>,
    seed: u64,
    branch: u8,
    idempotency_key: Option<String>,
}
pub(crate) fn request(
    names: Vec<String>,
    columns: &[PyReadonlyArray1<'_, f64>],
    text: &str,
) -> PyResult<ExternalCallbackRequest> {
    check_columns(&names, columns)?;
    if names.len() != columns.len() {
        return Err(invalid("external_recalc.invalid_request", "column names and arrays differ"));
    }
    let wire: RequestWire = parse_json(text, "external callback")?;
    let program = parse_binding(&wire.program.to_string())?;
    let claim = claim_from(wire.claim)
        .map_err(|error| invalid("external_recalc.invalid_request", format!("{error:?}")))?;
    let contract = core_contract(&wire.contract)
        .map_err(|error| invalid("external_recalc.invalid_request", format!("{error:?}")))?;
    let branch = Branch::new(wire.branch)
        .ok_or_else(|| invalid("external_recalc.invalid_request", "branch exceeds bound"))?;
    Ok(ExternalCallbackRequest {
        columns: names
            .into_iter()
            .zip(columns.iter().map(|array| array.as_array().to_vec()))
            .collect(),
        program,
        claim,
        contract,
        descriptor: descriptor(wire.descriptor)?,
        model_parameters: wire.model_parameters,
        seed: wire.seed,
        branch,
        idempotency_key: wire.idempotency_key,
    })
}
pub(crate) fn error(error: &ExternalCallbackError, branch: Branch) -> String {
    let stage = format!("provider_request.{}", branch.index());
    let mut body = match error {
        ExternalCallbackError::Binding(refusal)
        | ExternalCallbackError::BindingAttempt { refusal, .. } => {
            serde_json::to_value(RefusalWire::from((**refusal).clone()))
                .expect("refusal serialization")
        }
        ExternalCallbackError::Refused(plan) => return plan_refusal_json(plan),
        _ => {
            let detail = match error {
                ExternalCallbackError::Request(detail)
                | ExternalCallbackError::Attempt { detail, .. } => *detail,
                _ => "external_recalc.artifact_invalid",
            };
            let code = error.refusal_code();
            serde_json::from_str(&refusal_json(
                code,
                &stage,
                detail,
                None,
                None,
                &error.to_string(),
                None,
            ))
            .expect("refusal JSON")
        }
    };
    if let ExternalCallbackError::Attempt { report, .. }
    | ExternalCallbackError::BindingAttempt { report, .. } = error
    {
        body["attempt"] = serde_json::json!({"request_digest":report.request_digest.to_hex(),"invocations":report.invocations});
    }
    body.to_string()
}
pub(crate) struct PythonProvider {
    descriptor: ProviderDescriptor,
    callback: Py<PyAny>,
}
impl ExternalMeanProvider for PythonProvider {
    fn descriptor(&self) -> &ProviderDescriptor {
        &self.descriptor
    }
    fn invoke(
        &mut self,
        request: &ExternalCallbackRequest,
        ctx: &CallbackContext,
    ) -> Result<ExternalResponse, CallbackFailure> {
        Python::attach(|py| -> PyResult<ExternalResponse> {
            let payload = PyDict::new(py);
            let data = PyDict::new(py);
            for (name, values) in &request.columns {
                let array = PyArray1::from_vec(py, values.clone());
                array.call_method1("setflags", (false,))?;
                data.set_item(name, array)?;
            }
            let params = PyDict::new(py);
            for (name, value) in &request.model_parameters {
                params.set_item(name, value)?;
            }
            payload.set_item("data", data)?;
            payload.set_item("model_parameters", params)?;
            payload.set_item("doses", &request.program.dose_grid)?;
            payload.set_item("seed", request.seed)?;
            payload.set_item("branch", request.branch.index())?;
            payload.set_item("idempotency_key", &request.idempotency_key)?;
            payload.set_item("graph_id", &request.contract.graph_id)?;
            let quantities = serde_json::to_string(
                &request
                    .contract
                    .estimand
                    .iter()
                    .map(antecedent_io::quantity_wire::ScientificQuantityWire::from)
                    .collect::<Vec<_>>(),
            )
            .map_err(|cause| invalid("external_recalc.invalid_request", cause.to_string()))?;
            payload
                .set_item("quantities", py.import("json")?.call_method1("loads", (quantities,))?)?;
            payload.set_item(
                "cancellation",
                Py::new(py, crate::PyCancellationToken { inner: ctx.cancellation.clone() })?,
            )?;
            let output = self.callback.bind(py).call1((payload,))?;
            let text = output.cast::<PyString>()?;
            if text.len()? > 1024 * 1024 {
                return Err(invalid(
                    "external_recalc.limits_exceeded",
                    "callback output exceeds byte bound",
                ));
            }
            let json = text.to_str()?;
            if json.len() > 1024 * 1024 {
                return Err(invalid(
                    "external_recalc.limits_exceeded",
                    "callback output exceeds byte bound",
                ));
            }
            let response: ResponseWire = parse_json(json, "callback output")?;
            if response.values.len() > 1024 || response.quantities.len() > 1024 {
                return Err(invalid(
                    "external_recalc.limits_exceeded",
                    "callback values/coordinates exceed bound",
                ));
            }
            response_from_wire(&response)
                .map_err(|cause| invalid("external_recalc.invalid_output", format!("{cause:?}")))
        })
        .map_err(|cause| CallbackFailure { message: cause.to_string() })
    }
}
#[pyclass(name = "ExternalCallbackProviderHandle", skip_from_py_object)]
pub(crate) struct PyCallbackProvider {
    descriptor: ProviderDescriptor,
    callback: Option<Py<PyAny>>,
}
impl PyCallbackProvider {
    pub(crate) fn clone_actual_provider(&self, py: Python<'_>) -> PyResult<PythonProvider> {
        Ok(PythonProvider {
            descriptor: self.descriptor.clone(),
            callback: self
                .callback
                .as_ref()
                .ok_or_else(|| {
                    invalid("external_recalc.provider_unavailable", "callback was cleared")
                })?
                .clone_ref(py),
        })
    }
}
#[pymethods]
impl PyCallbackProvider {
    #[new]
    fn new(py: Python<'_>, text: &str, callback: Py<PyAny>) -> PyResult<Self> {
        if !callback.bind(py).is_callable() {
            return Err(invalid("external_recalc.invalid_descriptor", "callback must be callable"));
        }
        Ok(Self {
            descriptor: descriptor(parse_json(text, "callback descriptor")?)?,
            callback: Some(callback),
        })
    }
    fn __traverse__(&self, visit: PyVisit<'_>) -> Result<(), PyTraverseError> {
        if let Some(callback) = &self.callback {
            visit.call(callback)?;
        }
        Ok(())
    }
    fn __clear__(&mut self) {
        self.callback = None;
    }
}
#[pyclass(name = "ExternalCallbackSessionHandle", skip_from_py_object)]
pub(crate) struct PyCallbackSession {
    inner: ExternalCallbackSession,
}
type Payload = (
    Option<String>,
    Option<Vec<u8>>,
    Option<crate::external_api::PyExternalClaimArtifact>,
    Option<String>,
);
#[pymethods]
impl PyCallbackSession {
    #[new]
    fn new() -> Self {
        Self { inner: ExternalCallbackSession::new() }
    }
    fn is_live(&self) -> bool {
        self.inner.issued_claim().is_some()
    }
    fn plan(
        &self,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        text: &str,
        supplied_provider: bool,
    ) -> PyResult<(Option<String>, Option<String>)> {
        let req = request(names, &columns, text)?;
        let plan = self.inner.plan_with_provider(&req, supplied_provider);
        Ok((
            Some(
                serde_json::to_string(&antecedent_io::recalc_receipt_artifact::plan_to_wire(&plan))
                    .map_err(|cause| {
                        invalid("external_recalc.invalid_request", cause.to_string())
                    })?,
            ),
            None,
        ))
    }
    #[pyo3(signature=(names,columns,text,provider=None,*,threads=None,cancel=None))]
    fn execute(
        &mut self,
        py: Python<'_>,
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        text: &str,
        provider: Option<PyRef<'_, PyCallbackProvider>>,
        threads: Option<u32>,
        cancel: Option<crate::PyCancellationToken>,
    ) -> PyResult<Payload> {
        let req = request(names, &columns, text)?;
        let mut actual = provider.map(|provider| provider.clone_actual_provider(py)).transpose()?;
        let mut ctx = context(req.seed, threads);
        if let Some(cancel) = cancel {
            ctx.cancellation = cancel.inner;
        }
        let mut session = std::mem::take(&mut self.inner);
        let ran = py.detach(|| {
            session.execute(&req, actual.as_mut().map(|p| p as &mut dyn ExternalMeanProvider), &ctx)
        });
        self.inner = session;
        match ran {
            Err(cause) => Ok((None, None, None, Some(error(&cause, req.branch)))),
            Ok(out) => {
                let (wire, receipt, refusal) = crate::recalc_family_api::payload(
                    &out.previous,
                    &out.requested,
                    &out.capabilities,
                    &out.plan,
                    &out.receipt,
                    serde_json::json!({}),
                )?;
                let artifact =
                    ExternalClaimArtifact::from_bound_claim(&out.claim, &req.program.contract_id)
                        .map_err(|cause| {
                        invalid("external_recalc.artifact_invalid", cause.to_string())
                    })?;
                Ok((
                    wire,
                    receipt,
                    Some(crate::external_api::PyExternalClaimArtifact { artifact }),
                    refusal,
                ))
            }
        }
    }
    fn export_output(&self) -> (Option<Vec<u8>>, Option<String>) {
        match self.inner.export_output() {
            Ok(bytes) => (Some(bytes), None),
            Err(cause) => (None, Some(error(&cause, Branch::new(0).expect("zero branch")))),
        }
    }
    #[staticmethod]
    fn resume(
        data: &[u8],
        names: Vec<String>,
        columns: Vec<PyReadonlyArray1<'_, f64>>,
        text: &str,
    ) -> PyResult<(Option<Self>, Option<String>)> {
        if data.len() > 16 * 1024 * 1024 {
            return Err(invalid(
                "external_recalc.limits_exceeded",
                "callback artifact exceeds byte bound",
            ));
        }
        let req = request(names, &columns, text)?;
        match ExternalCallbackSession::resume(data, &req) {
            Ok(inner) => Ok((Some(Self { inner }), None)),
            Err(cause) => Ok((None, Some(error(&cause, req.branch)))),
        }
    }
}
pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<PyCallbackProvider>()?;
    module.add_class::<PyCallbackSession>()
}
