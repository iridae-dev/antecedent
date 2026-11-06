//! Python bridge for arbitrary law, posterior and nested callbacks.
//!
//! Python declares callbacks, requests and policies as JSON; the validation, the
//! negotiation, the reuse plan, the output check, the identities and the
//! artifact are Rust's. A provider is a Python callable the caller passes in: it
//! is called only from `execute` and `resume`, with the GIL held, as
//! `invoke(query_json, context_json, token) -> output_json`. Floating-point
//! output crosses as IEEE bit patterns (an integer array), so a NaN reaches the
//! Rust output check instead of failing JSON decoding. Nothing is pickled and
//! no artifact field can name a callable.

mod transport;
mod wire;

use std::collections::BTreeMap;
use wire::*;

use antecedent::analysis::evidence_ledger::{SourceId, TransportIdentity};
use antecedent::analysis::interface_contract::{
    CalibrationStatus, ContractOperation, EvidenceBasis, SupportDomain, UncertaintyMeaning,
};
use antecedent::analysis::predictive_assessment::{ForecastLaw, LawSource};
use antecedent::analysis::provider_callbacks::{
    CallbackConnectionError, CallbackContext, CallbackDeclaration, CallbackDependency,
    CallbackExecution, CallbackExecutionError, CallbackExecutionPolicy, CallbackFailure,
    CallbackInput, CallbackKind, CallbackOperation, CallbackOutcome, CallbackOutput, CallbackPhase,
    CallbackPlan, CallbackQuery, CallbackRegistrationError, CallbackRequest,
    CallbackResultArtifact, CallbackResultRecord, CallbackResume, CallbackSession, CallbackTrust,
    DependencyDeclaration, DependencyKind, FnCallback, InputSupplier, ProviderCallback,
    ReplayContract, RetryContract, UpdateDeclaration, ValidatedOutput, check_claim_to_callback,
    export_callback_result, forecast_law_from_sample, output_values_digest, predictive_law_source,
    refuse_identification_evidence, refuse_interval, refuse_joint_law, refuse_native_trust,
    refuse_posterior_guarantee, verification_holds,
};
use antecedent::analysis::transport_handoff::{
    TransportHandoffRefusal, TransportedClaimOffer, TransportedInputPolicy, consume_for_callback,
    hand_to_callback,
};
use antecedent::contract::evidence_ledger::parse_digest_hex;
use antecedent::contract::provider_transport::{
    AdmittedProviderSource, ProviderTransportSource, SourceTrustRequirement,
    connect_bound_provider_source,
};
use antecedent::contract::{
    ExternalRefusal, ExternalTrustState, ProviderObjectIdentity, VerificationProbe,
    probe_from_name, verify_external_object,
};
use antecedent::execution::CancellationToken;
use antecedent::io::external_claim_artifact::VerificationProbeWire;
use antecedent::io::interface_contract_artifact::contract_to_wire;
use antecedent::io::quantity_wire::ScientificQuantityWire;
use antecedent::prelude::ExecutionContext;
use antecedent::scientific::ScientificQuantity;
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};
use serde_json::{Value as Json, json};

use crate::bridge::RefusalJson;
use crate::evidence_ledger_api::EvidenceLedgerHandle;
use crate::interface_contract_api::{connection_json, contract_of};
use crate::measurement_model_api::parse;
use crate::provider_api::PyProviderVerificationReceipt;

/// An exported record `(bytes, identity)` or the refusal JSON.
type ExportedRecord = (Option<(Py<PyBytes>, String)>, Option<String>);

fn refusal_json(refusal: &ExternalRefusal) -> Json {
    json!({
        "code": refusal.code,
        "stage": refusal.stage,
        "detail": refusal.detail,
        "offending": refusal.offending,
        "expected": refusal.expected,
        "supplied": refusal.supplied,
        "remedy": refusal.remedy,
        "message": refusal.remedy.unwrap_or_default(),
    })
}

fn error_json(error: &CallbackExecutionError) -> String {
    json!({
        "refusal": refusal_json(&error.refusal),
        "invocations": error.invocations,
        "request_attempts": error.request_attempts,
    })
    .to_string()
}

fn standing_name(execution: &CallbackExecution) -> &'static str {
    execution.standing.as_str()
}

fn reason_name(reason: &antecedent::analysis::provider_callbacks::RecomputeReason) -> String {
    use antecedent::analysis::provider_callbacks::RecomputeReason as R;
    match reason {
        R::NoPriorResult => "no_prior_result".into(),
        R::OpaqueDependencies => "opaque_dependencies".into(),
        R::DeclarationChanged => "declaration_changed".into(),
        R::DependencyChanged { name, .. } => format!("dependency_changed:{name}"),
        R::RequestChanged => "request_changed".into(),
        R::HiddenStateUnobservable => "hidden_state_unobservable".into(),
        R::StateChanged => "state_changed".into(),
        R::StateUnknown => "state_unknown".into(),
    }
}

// -- the provider ------------------------------------------------------------------------------

/// A cooperative cancellation token shared with the callable that runs.
#[pyclass(name = "CallbackToken", skip_from_py_object)]
#[derive(Clone)]
pub(crate) struct CallbackToken {
    inner: CancellationToken,
}

#[pymethods]
impl CallbackToken {
    #[new]
    fn new() -> Self {
        Self { inner: CancellationToken::new() }
    }

    /// Request cancellation.
    fn cancel(&self) {
        self.inner.cancel();
    }

    /// Whether cancellation was requested.
    fn is_cancelled(&self) -> bool {
        self.inner.is_cancelled()
    }
}

/// A provider that answers with a Python callable: the library's closure adapter around
/// [`invoke_python`], reporting the state identity through the caller's state callable.
type PyProvider = FnCallback<
    Box<
        dyn FnMut(
            &CallbackQuery,
            &mut CallbackContext<'_>,
        ) -> Result<CallbackOutput, CallbackFailure>,
    >,
>;

/// The first non-`Exception` raised by a provider callable (an interrupt), held until the
/// native call returns so it can be raised in place of the refusal it was reported as.
type Interrupt = std::sync::Arc<std::sync::Mutex<Option<PyErr>>>;

fn raise_interrupt(interrupt: &Interrupt) -> PyResult<()> {
    match interrupt.lock().unwrap_or_else(std::sync::PoisonError::into_inner).take() {
        Some(error) => Err(error),
        None => Ok(()),
    }
}

fn invoke_python(
    invoke: &Py<PyAny>,
    token: &CallbackToken,
    interrupt: &Interrupt,
    query: &CallbackQuery,
    ctx: &mut CallbackContext<'_>,
) -> Result<CallbackOutput, CallbackFailure> {
    let inputs: BTreeMap<&String, Json> =
        ctx.inputs.iter().map(|(name, value)| (name, output_json(value))).collect();
    let context = json!({
        "stream_seed": ctx.rng.as_ref().map(antecedent::execution::CausalRng::state),
        "attempt": ctx.attempt,
        "idempotency_key": ctx.idempotency_key,
        "inputs": inputs,
    });
    let query_text = serde_json::to_string(&query_json(query))
        .map_err(|error| CallbackFailure::new(error.to_string()))?;
    let token = token.clone();
    // The caller's thread holds the GIL; attach is re-entrant.
    Python::attach(|py| {
        let called = invoke
            .bind(py)
            .call1((query_text, context.to_string(), token.clone()))
            .map_err(|error| {
                if error.is_instance_of::<pyo3::exceptions::PyException>(py) {
                    let text = format!("{}", error.value(py));
                    return CallbackFailure::new(text.chars().take(512).collect::<String>());
                }
                let mut first = interrupt.lock().unwrap_or_else(std::sync::PoisonError::into_inner);
                if first.is_none() {
                    *first = Some(error);
                }
                token.inner.cancel();
                CallbackFailure::new("provider callable was interrupted")
            })?;
        let text: String = called
            .extract()
            .map_err(|_| CallbackFailure::new("the callable must return a JSON string"))?;
        if text.len() > 64 * 1024 * 1024 {
            return Err(CallbackFailure::new("the callable's answer is over the bound"));
        }
        let parsed: OutputBitsJson = serde_json::from_str(&text)
            .map_err(|error| CallbackFailure::new(format!("unparseable answer: {error}")))?;
        output_of(&parsed).map_err(CallbackFailure::new)
    })
}

#[allow(
    clippy::type_complexity,
    reason = "the callback type preserves its context and error contract"
)]
fn provider_of(
    identity_json: &str,
    invoke: Py<PyAny>,
    state: Option<Py<PyAny>>,
    token: &CallbackToken,
    interrupt: &Interrupt,
) -> PyResult<PyProvider> {
    let value = parse(identity_json, "provider identity")?;
    let identity: IdentityJson =
        serde_json::from_value(value).map_err(|error| bad("provider identity", error))?;
    let token = token.clone();
    let interrupt = Interrupt::clone(interrupt);
    let call: Box<
        dyn FnMut(
            &CallbackQuery,
            &mut CallbackContext<'_>,
        ) -> Result<CallbackOutput, CallbackFailure>,
    > = Box::new(move |query, ctx| invoke_python(&invoke, &token, &interrupt, query, ctx));
    let provider = FnCallback::new(
        ProviderObjectIdentity {
            provider_id: identity.provider_id,
            object_id: identity.object_id,
            version_id: identity.version_id,
            snapshot_id: identity.snapshot_id,
            request_id: identity.request_id,
        },
        call,
    );
    Ok(match state {
        None => provider,
        Some(state) => provider.with_state_identity(move || {
            Python::attach(|py| {
                state
                    .bind(py)
                    .call0()
                    .ok()
                    .and_then(|value| value.extract::<Option<String>>().ok())
                    .flatten()
            })
        }),
    })
}

fn context_of(token: &CallbackToken) -> ExecutionContext {
    let mut ctx = crate::execution_api::context(0);
    ctx.cancellation = token.inner.clone();
    ctx
}

fn inputs_of(session: &CallbackSession, text: &str) -> PyResult<BTreeMap<String, ValidatedOutput>> {
    let value = parse(text, "callback inputs")?;
    let map = value.as_object().ok_or_else(|| bad("callback inputs", "an object"))?;
    let mut out = BTreeMap::new();
    for (name, index) in map {
        let index = index
            .as_u64()
            .and_then(|i| usize::try_from(i).ok())
            .ok_or_else(|| bad("callback inputs", "a history index"))?;
        let record = session
            .history()
            .get(index)
            .ok_or_else(|| bad("callback inputs", "history index out of range"))?;
        match &record.outcome {
            CallbackOutcome::Computed { output, .. } => {
                out.insert(name.clone(), output.clone());
            }
            CallbackOutcome::Unavailable { .. } => {
                return Err(bad("callback inputs", "an unavailable stage has no output"));
            }
        }
    }
    Ok(out)
}

fn history_index(session: &CallbackSession, execution: &CallbackExecution) -> Option<usize> {
    session.history().iter().rposition(|record| {
        record.declaration.canonical_identity().to_hex() == execution.declaration_identity
            && record.request.identity(&record.declaration).to_hex() == execution.request_identity
            && matches!(record.outcome, CallbackOutcome::Computed { .. })
    })
}

fn execution_json(session: &CallbackSession, execution: &CallbackExecution) -> String {
    json!({
        "output": output_json(&execution.output),
        "output_digest": output_values_digest(&execution.output.output().values()),
        "invocations": execution.invocations,
        "reused": execution.reused,
        "recomputed_because": execution.recomputed_because.as_ref().map(reason_name),
        "standing": standing_name(execution),
        "declaration_identity": execution.declaration_identity,
        "request_identity": execution.request_identity,
        "state_identity": execution.state_identity,
        "history_index": history_index(session, execution),
    })
    .to_string()
}

fn record_json(record: &CallbackResultRecord) -> Json {
    let (stage, values, invocations, state, reason) = match &record.outcome {
        CallbackOutcome::Computed { output, invocations, state_identity } => {
            ("computed", output.output().values(), *invocations, state_identity.clone(), None)
        }
        CallbackOutcome::Unavailable { reason } => {
            ("unavailable", Vec::new(), 0, None, Some(reason.clone()))
        }
    };
    json!({
        "declaration": declaration_json(&record.declaration),
        "request": {
            "query": query_json(&record.request.query),
            "seed": record.request.seed,
            "input_digests": record.request.input_digests,
            "idempotency_key": record.request.idempotency_key,
        },
        "declaration_identity": record.declaration.canonical_identity().to_hex(),
        "request_identity": record.request.identity(&record.declaration).to_hex(),
        "stage": stage,
        "reason": reason,
        "values": values,
        "invocations": invocations,
        "state_identity": state,
        "output_digest": output_values_digest(&match &record.outcome {
            CallbackOutcome::Computed { output, .. } => output.output().values(),
            CallbackOutcome::Unavailable { .. } => Vec::new(),
        }),
    })
}

type Checked = (Option<String>, Option<String>);

// -- the session -------------------------------------------------------------------------------

/// The execution state of one process: history and invocation counts.
#[pyclass(name = "CallbackSessionHandle", skip_from_py_object)]
pub(crate) struct CallbackSessionHandle {
    inner: CallbackSession,
}

#[pymethods]
impl CallbackSessionHandle {
    #[new]
    fn new() -> Self {
        Self { inner: CallbackSession::new() }
    }

    /// Provider invocations so far, failed attempts included.
    fn invocations(&self) -> u64 {
        self.inner.invocations()
    }

    /// State-identity probes so far; a probe is not an invocation.
    fn state_probes(&self) -> u64 {
        self.inner.state_probes()
    }

    /// Invocations of one request identity (hex).
    fn request_attempts(&self, request_identity: &str) -> u64 {
        self.inner.request_attempts(request_identity)
    }

    /// Number of records in the history.
    fn history_len(&self) -> usize {
        self.inner.history().len()
    }

    /// One history record as JSON.
    fn record_json(&self, index: usize) -> PyResult<String> {
        crate::catch_ffi(|| {
            self.inner
                .history()
                .get(index)
                .map(|record| record_json(record).to_string())
                .ok_or_else(|| crate::value_err("history index out of range"))
        })
    }

    /// History indices of results that consumed the callback with this
    /// declaration identity, transitively.
    fn dependents_of(&self, identity: &str) -> Vec<usize> {
        self.inner.dependents_of(identity)
    }

    /// Decide reuse from declarations and stored records alone. Takes no
    /// provider, so it cannot call one.
    #[pyo3(signature = (declaration_json, request_json, current_state=None))]
    fn plan(
        &self,
        declaration_json: &str,
        request_json: &str,
        current_state: Option<&str>,
    ) -> PyResult<Checked> {
        crate::catch_ffi(|| {
            let declaration = declaration_of(declaration_json)?;
            let request = request_of(request_json)?;
            Ok(match self.inner.plan(&declaration, &request, current_state) {
                Ok(CallbackPlan::Reuse) => (Some(json!({"plan": "reuse"}).to_string()), None),
                Ok(CallbackPlan::Recompute(reason)) => (
                    Some(json!({"plan": "recompute", "reason": reason_name(&reason)}).to_string()),
                    None,
                ),
                Ok(CallbackPlan::Refuse(refusal)) => (
                    Some(json!({"plan": "refuse", "refusal": refusal_json(&refusal)}).to_string()),
                    None,
                ),
                Err(refusal) => (None, Some(refusal_json(&refusal).to_string())),
            })
        })
    }

    /// Execute a request. `invoke` and `state` are the caller's Python
    /// callables; they run only here, with the GIL held.
    #[pyo3(signature = (identity_json, invoke, state, declaration_json, request_json, inputs_json, policy_json, token))]
    #[allow(
        clippy::too_many_arguments,
        reason = "one coordinate per declared input of an execution"
    )]
    fn execute(
        &mut self,
        identity_json: &str,
        invoke: Py<PyAny>,
        state: Option<Py<PyAny>>,
        declaration_json: &str,
        request_json: &str,
        inputs_json: &str,
        policy_json: &str,
        token: &CallbackToken,
    ) -> PyResult<Checked> {
        crate::catch_ffi(|| {
            let declaration = declaration_of(declaration_json)?;
            let request = request_of(request_json)?;
            let policy = policy_of(policy_json)?;
            let inputs = inputs_of(&self.inner, inputs_json)?;
            let interrupt = Interrupt::default();
            let mut provider = provider_of(identity_json, invoke, state, token, &interrupt)?;
            let ctx = context_of(token);
            let outcome =
                self.inner.execute(&mut provider, &declaration, &request, &inputs, &policy, &ctx);
            raise_interrupt(&interrupt)?;
            Ok(match outcome {
                Ok(execution) => (Some(execution_json(&self.inner, &execution)), None),
                Err(error) => (None, Some(error_json(&error))),
            })
        })
    }

    /// Check the connection of the declared callback to a consumer through the interface
    /// contract, then execute. A mismatched population, regime, support or operation is
    /// refused before the provider is invoked. The payload is the connection and the
    /// execution; a connection refusal lists every defect.
    #[pyo3(signature = (identity_json, invoke, state, declaration_json, consumer_json, consumed_port, request_json, inputs_json, policy_json, token))]
    #[allow(
        clippy::too_many_arguments,
        reason = "one coordinate per declared input of an execution and one for the consumer"
    )]
    fn execute_for_consumer(
        &mut self,
        identity_json: &str,
        invoke: Py<PyAny>,
        state: Option<Py<PyAny>>,
        declaration_json: &str,
        consumer_json: &str,
        consumed_port: &str,
        request_json: &str,
        inputs_json: &str,
        policy_json: &str,
        token: &CallbackToken,
    ) -> PyResult<Checked> {
        crate::catch_ffi(|| {
            let declaration = declaration_of(declaration_json)?;
            let consumer = contract_of(consumer_json, "consumer contract")?;
            let request = request_of(request_json)?;
            let policy = policy_of(policy_json)?;
            let inputs = inputs_of(&self.inner, inputs_json)?;
            let interrupt = Interrupt::default();
            let mut provider = provider_of(identity_json, invoke, state, token, &interrupt)?;
            let ctx = context_of(token);
            let outcome = self.inner.execute_for_consumer(
                &mut provider,
                &declaration,
                &consumer,
                consumed_port,
                &request,
                &inputs,
                &policy,
                &ctx,
            );
            raise_interrupt(&interrupt)?;
            Ok(
            match outcome {
                Ok((connection, execution)) => (
                    Some(
                        json!({
                            "connection": serde_json::from_str::<Json>(&connection_json(&connection))
                                .unwrap_or(Json::Null),
                            "execution": serde_json::from_str::<Json>(&execution_json(&self.inner, &execution))
                                .unwrap_or(Json::Null),
                        })
                        .to_string(),
                    ),
                    None,
                ),
                Err(CallbackConnectionError::Connection(refusals)) => (
                    None,
                    Some(
                        json!({"connection_refusals": refusals.iter().map(refusal_json).collect::<Vec<_>>()})
                            .to_string(),
                    ),
                ),
                Err(CallbackConnectionError::Execution(error)) => (None, Some(error_json(&error))),
            },
        )
        })
    }

    /// Resume a stored result in this process. With no provider the stage is
    /// recorded unavailable.
    #[pyo3(signature = (artifact, expected_identity, identity_json, invoke, state, inputs_json, policy_json, token))]
    #[allow(clippy::too_many_arguments, reason = "one coordinate per declared input of a resume")]
    fn resume(
        &mut self,
        artifact: &[u8],
        expected_identity: Option<&str>,
        identity_json: Option<&str>,
        invoke: Option<Py<PyAny>>,
        state: Option<Py<PyAny>>,
        inputs_json: &str,
        policy_json: &str,
        token: &CallbackToken,
    ) -> PyResult<Checked> {
        crate::catch_ffi(|| {
            let policy = policy_of(policy_json)?;
            let inputs = inputs_of(&self.inner, inputs_json)?;
            let ctx = context_of(token);
            let interrupt = Interrupt::default();
            let mut provider = match (identity_json, invoke) {
                (Some(identity), Some(invoke)) => {
                    Some(provider_of(identity, invoke, state, token, &interrupt)?)
                }
                (None, None) => None,
                _ => return Err(bad("provider", "supply both the identity and the callable")),
            };
            let resumed = self.inner.resume(
                artifact,
                expected_identity,
                provider.as_mut().map(|p| p as &mut dyn ProviderCallback),
                &inputs,
                &policy,
                &ctx,
            );
            raise_interrupt(&interrupt)?;
            Ok(match resumed {
            Ok(CallbackResume::Answered(execution)) => (
                Some(
                    json!({
                        "resume": "answered",
                        "execution": serde_json::from_str::<Json>(&execution_json(&self.inner, &execution)).unwrap_or(Json::Null),
                    })
                    .to_string(),
                ),
                None,
            ),
            Ok(CallbackResume::Unavailable { record, refusal }) => (
                Some(
                    json!({"resume": "unavailable", "record": record, "refusal": refusal_json(&refusal)})
                        .to_string(),
                ),
                None,
            ),
            Err(error) => (None, Some(error_json(&error))),
        })
        })
    }

    /// The empirical-sample forecast law of the accepted sample output at a history index, as the
    /// predictive-assessment law JSON, or the typed refusal when the output is not a sample.
    fn forecast_law(&self, index: usize) -> PyResult<Checked> {
        crate::catch_ffi(|| {
            let record = self
                .inner
                .history()
                .get(index)
                .ok_or_else(|| crate::value_err("history index out of range"))?;
            let output = match &record.outcome {
                CallbackOutcome::Computed { output, .. } => output,
                CallbackOutcome::Unavailable { .. } => {
                    return Err(bad("forecast law", "an unavailable stage has no output"));
                }
            };
            Ok(match forecast_law_from_sample(output) {
                Ok(ForecastLaw::EmpiricalSample { draws }) => {
                    (Some(json!({"kind": "empirical_sample", "draws": draws}).to_string()), None)
                }
                Ok(_) => return Err(bad("forecast law", "a sample output lifts to a sample law")),
                Err(refusal) => (None, Some(refusal_json(&refusal).to_string())),
            })
        })
    }

    /// Export one history record as `callback_result_v1`.
    fn export_record(
        &self,
        py: Python<'_>,
        index: usize,
        artifact_id: &str,
    ) -> PyResult<ExportedRecord> {
        crate::catch_ffi(|| {
            let record = self
                .inner
                .history()
                .get(index)
                .ok_or_else(|| crate::value_err("history index out of range"))?;
            Ok(match export_callback_result(record, artifact_id) {
                Ok(bytes) => {
                    let identity = CallbackResultArtifact::from_bytes(&bytes, None)
                        .map(|a| a.identity().to_owned())
                        .unwrap_or_default();
                    (Some((PyBytes::new(py, &bytes).unbind(), identity)), None)
                }
                Err(error) => (None, Some(artifact_error_json(&error))),
            })
        })
    }
}

fn artifact_error_json(
    error: &antecedent::analysis::provider_callbacks::CallbackResultArtifactError,
) -> String {
    match error.refusal() {
        Some((code, detail, message)) => {
            json!({"code": code, "stage": "consume", "detail": detail, "message": message,
                   "offending": null, "expected": null, "supplied": null, "remedy": message})
            .to_string()
        }
        None => json!({
            "code": antecedent::reason::reason_code!("invalid_argument"),
            "stage": "consume",
            "detail": antecedent::reason::reason_details::callback_protocol::ARTIFACT_INVALID,
            "message": error.to_string(),
            "offending": null, "expected": null, "supplied": null,
            "remedy": "supply an intact callback_result_v1 artifact",
        })
        .to_string(),
    }
}

/// The canonical identity of a declaration, or its refusal. Runs nothing.
#[pyfunction]
fn callback_declaration_identity(declaration_json: &str) -> PyResult<Checked> {
    crate::catch_ffi(|| {
        let declaration = declaration_of(declaration_json)?;
        Ok(match declaration.validate() {
        Ok(()) => (
            Some(
                json!({
                    "identity": declaration.canonical_identity().to_hex(),
                    "effective_evidence": evidence_json(&declaration.effective_evidence()),
                    "contract_identity": declaration.interface_contract().canonical_identity().to_hex(),
                    "standing_if_attested": antecedent::analysis::provider_callbacks::standing_label(&declaration, false).as_str(),
                })
                .to_string(),
            ),
            None,
        ),
        Err(refusal) => (None, Some(refusal_json(&refusal).to_string())),
    })
    })
}

/// The digest (hex) of output values, for binding a nested callback's input.
#[pyfunction]
fn callback_values_digest(values: Vec<f64>) -> String {
    output_values_digest(&values)
}

/// Consume a `callback_result_v1` artifact: every identity recomputed, every
/// bound checked. Nothing is executed.
#[pyfunction]
#[pyo3(signature = (artifact, expected_identity=None))]
fn callback_result_consume(artifact: &[u8], expected_identity: Option<&str>) -> Checked {
    match CallbackResultArtifact::from_bytes(artifact, expected_identity) {
        Ok(done) => {
            let mut value = record_json(done.record());
            value["identity"] = json!(done.identity());
            (Some(value.to_string()), None)
        }
        Err(error) => (None, Some(artifact_error_json(&error))),
    }
}

/// Read the stored metadata of a result without replaying it.
#[pyfunction]
fn callback_result_inspect(artifact: &[u8]) -> Checked {
    match antecedent::analysis::inspect_callback_result(artifact) {
        Ok(meta) => (Some(serde_json::to_value(&meta).unwrap_or(Json::Null).to_string()), None),
        Err(error) => (None, Some(artifact_error_json(&error))),
    }
}

/// The refusal of a route that stays closed: `interval`, `posterior_guarantee`,
/// `native_trust`, `identification_evidence` or `joint_law`.
#[pyfunction]
fn callback_closed_refusal(route: &str) -> PyResult<String> {
    crate::catch_ffi(|| {
        let refusal = match route {
            "interval" => refuse_interval(),
            "posterior_guarantee" => refuse_posterior_guarantee(),
            "native_trust" => refuse_native_trust(),
            "identification_evidence" => refuse_identification_evidence(),
            "joint_law" => refuse_joint_law(),
            other => return Err(bad("unsupported route", other)),
        };
        Ok(refusal_json(&refusal).to_string())
    })
}

/// Verify the exact provider object of a declared callback against independently computed
/// probes. The library issues the receipt (or the structured refusal JSON): whether the probes
/// pass, what the receipt covers and its digest are decided natively. Runs nothing.
#[pyfunction]
fn callback_verify_provider(
    py: Python<'_>,
    declaration_json: &str,
    probes_json: &str,
) -> PyResult<(Option<Py<PyProviderVerificationReceipt>>, Option<String>)> {
    crate::catch_ffi(|| {
        let declaration = declaration_of(declaration_json)?;
        let wires: Vec<VerificationProbeWire> =
            serde_json::from_value(parse(probes_json, "verification probes")?)
                .map_err(|error| bad("verification probes", error))?;
        let mut probes = Vec::with_capacity(wires.len());
        for wire in &wires {
            let kind = probe_from_name(&wire.kind).ok_or_else(|| {
                bad("verification probe", format!("unknown kind `{}`", wire.kind))
            })?;
            probes.push(VerificationProbe {
                kind,
                observed: wire.observed,
                expected: wire.expected,
                tolerance: wire.tolerance,
            });
        }
        Ok(match verify_external_object(&declaration.provider_object(), &probes) {
            Ok(inner) => (Some(Py::new(py, PyProviderVerificationReceipt { inner })?), None),
            Err(error) => (None, Some(refusal_json(&error.to_refusal()).to_string())),
        })
    })
}

/// The predictive-assessment law source of a declared callback, as the law-source JSON, or the
/// refusal of a malformed declaration or a diagnostic standing. A verified-extension claim
/// stands only when `receipt` is the library-issued verification the declaration names for the
/// exact provider object. Runs nothing.
#[pyfunction]
#[pyo3(signature = (declaration_json, receipt=None))]
fn callback_predictive_law_source(
    declaration_json: &str,
    receipt: Option<PyRef<'_, PyProviderVerificationReceipt>>,
) -> PyResult<Checked> {
    crate::catch_ffi(|| {
        let declaration = declaration_of(declaration_json)?;
        let verification_confirmed =
            receipt.is_some_and(|receipt| verification_holds(&declaration, &receipt.inner));
        Ok(match predictive_law_source(&declaration, verification_confirmed) {
            Ok(LawSource::Provider { identity, trust }) => (
                Some(
                    json!({
                        "class": "provider",
                        "identity": {
                            "provider_id": identity.provider_id,
                            "object_id": identity.object_id,
                            "version_id": identity.version_id,
                            "snapshot_id": identity.snapshot_id,
                            "request_id": identity.request_id,
                        },
                        "trust": trust.name(),
                    })
                    .to_string(),
                ),
                None,
            ),
            Ok(_) => return Err(bad("law source", "a callback declares a provider law source")),
            Err(refusal) => (None, Some(refusal_json(&refusal).to_string())),
        })
    })
}

/// Check a producer's claim against one input of a declared callback through the interface
/// contract. The connection JSON, or a JSON array of every refusal.
#[pyfunction]
fn callback_check_claim_connection(
    producer_json: &str,
    produced_port: &str,
    callback_json: &str,
    input: &str,
) -> PyResult<Checked> {
    crate::catch_ffi(|| {
        let producer = contract_of(producer_json, "producer contract")?;
        let declaration = declaration_of(callback_json)?;
        Ok(match check_claim_to_callback(&producer, produced_port, &declaration, input) {
            Ok(connection) => (Some(connection_json(&connection)), None),
            Err(refusals) => {
                (None, Some(Json::Array(refusals.iter().map(refusal_json).collect()).to_string()))
            }
        })
    })
}

fn inner_sources_of(text: &str) -> PyResult<BTreeMap<String, SourceId>> {
    let value = parse(text, "inner sources")?;
    let map = value.as_object().ok_or_else(|| bad("inner sources", "an object"))?;
    let mut out = BTreeMap::new();
    for (identity, source) in map {
        let id = source
            .as_str()
            .and_then(SourceId::from_hex)
            .ok_or_else(|| bad("inner sources", "a source is not a digest"))?;
        out.insert(identity.clone(), id);
    }
    Ok(out)
}

fn registration_error_json(error: &CallbackRegistrationError) -> String {
    match error {
        CallbackRegistrationError::Declaration(refusal) => refusal_json(refusal).to_string(),
        CallbackRegistrationError::Ledger(error) => json!({
            "code": error.code,
            "stage": "evidence_ledger",
            "detail": error.detail,
            "message": error.message,
            "offending": error.sources.first().map(SourceId::to_hex),
            "remedy": Json::Null,
            "sources": error.sources.iter().map(SourceId::to_hex).collect::<Vec<_>>(),
        })
        .to_string(),
    }
}

/// Register a declared callback and its evidence in an evidence ledger session. The new source
/// identity (hex), or the refusal; a refusal leaves the ledger unchanged.
#[pyfunction]
fn callback_register_source(
    mut ledger: PyRefMut<'_, EvidenceLedgerHandle>,
    declaration_json: &str,
    inner_sources_json: &str,
) -> PyResult<Refused<String>> {
    crate::catch_ffi(|| {
        let declaration = declaration_of(declaration_json)?;
        let inner = inner_sources_of(inner_sources_json)?;
        Ok(match ledger.session_mut().register_callback(&declaration, &inner) {
            Ok(id) => (Some(id.to_hex()), None),
            Err(error) => (None, Some(registration_error_json(&error))),
        })
    })
}

/// Supersede a registered callback source with its replacement. The replacement's source
/// identity (hex), or the refusal.
#[pyfunction]
fn callback_supersede_source(
    mut ledger: PyRefMut<'_, EvidenceLedgerHandle>,
    old: &str,
    replacement_json: &str,
    inner_sources_json: &str,
) -> PyResult<Refused<String>> {
    crate::catch_ffi(|| {
        let old = SourceId::from_hex(old).ok_or_else(|| bad("source", "not a digest"))?;
        let replacement = declaration_of(replacement_json)?;
        let inner = inner_sources_of(inner_sources_json)?;
        Ok(match ledger.session_mut().supersede_callback(old, &replacement, &inner) {
            Ok(id) => (Some(id.to_hex()), None),
            Err(error) => (None, Some(registration_error_json(&error))),
        })
    })
}

type Refused<T> = (Option<T>, Option<String>);

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CallbackToken>()?;
    module.add_class::<CallbackSessionHandle>()?;
    module.add_function(wrap_pyfunction!(callback_declaration_identity, module)?)?;
    module.add_function(wrap_pyfunction!(callback_values_digest, module)?)?;
    module.add_function(wrap_pyfunction!(callback_result_consume, module)?)?;
    module.add_function(wrap_pyfunction!(callback_result_inspect, module)?)?;
    module.add_function(wrap_pyfunction!(callback_closed_refusal, module)?)?;
    module.add_function(wrap_pyfunction!(callback_verify_provider, module)?)?;
    module.add_function(wrap_pyfunction!(callback_predictive_law_source, module)?)?;
    module.add_function(wrap_pyfunction!(callback_check_claim_connection, module)?)?;
    module.add_function(wrap_pyfunction!(callback_register_source, module)?)?;
    module.add_function(wrap_pyfunction!(callback_supersede_source, module)?)?;
    transport::register(module)?;
    Ok(())
}
