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

use std::collections::BTreeMap;

use antecedent::analysis::provider_callbacks::{
    CallbackContext, CallbackDeclaration, CallbackDependency, CallbackExecution,
    CallbackExecutionError, CallbackExecutionPolicy, CallbackFailure, CallbackInput, CallbackKind,
    CallbackOperation, CallbackOutcome, CallbackOutput, CallbackPhase, CallbackPlan, CallbackQuery,
    CallbackRegistrationError, CallbackRequest, CallbackResultArtifact, CallbackResultRecord,
    CallbackResume, CallbackSession, CallbackTrust, DependencyDeclaration, DependencyKind,
    InputSupplier, ProviderCallback, ReplayContract, RetryContract, UpdateDeclaration,
    ValidatedOutput, check_claim_to_callback, forecast_law_from_sample, output_values_digest,
    predictive_law_source, refuse_identification_licence, refuse_interval, refuse_joint_law,
    refuse_native_trust, refuse_posterior_guarantee,
};
use antecedent::analysis::transport_handoff::{
    TransportHandoffRefusal, TransportedClaimOffer, TransportedInputPolicy, consume_for_callback,
    hand_to_callback,
};
use antecedent_contract::evidence_ledger::{SourceId, TransportIdentity, parse_digest_hex};
use antecedent_contract::interface_contract::{
    CalibrationStatus, ContractOperation, EvidenceBasis, SupportDomain, UncertaintyMeaning,
};
use antecedent_contract::{ExternalRefusal, ProviderObjectIdentity, ScientificQuantity};
use antecedent_core::{CancellationToken, ExecutionContext};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_validate::predictive_assessment::{ForecastLaw, LawSource};
use pyo3::prelude::*;
use pyo3::types::PyBytes;
use serde::{Deserialize, Serialize};
use serde_json::{Value as Json, json};

use crate::bridge::RefusalJson;
use crate::evidence_ledger_api::EvidenceLedgerHandle;
use crate::interface_contract_api::{connection_json, contract_of};
use crate::measurement_model_api::parse;

/// An exported record `(bytes, identity)` or the refusal JSON.
type ExportedRecord = (Option<(Py<PyBytes>, String)>, Option<String>);

// -- JSON shapes -------------------------------------------------------------------------------

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
#[allow(
    clippy::struct_field_names,
    reason = "the JSON field names are the serialized identity names of a provider object"
)]
struct IdentityJson {
    provider_id: String,
    object_id: String,
    version_id: String,
    snapshot_id: String,
    request_id: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SupportJson {
    kind: String,
    lower: Option<f64>,
    upper: Option<f64>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct EvidenceJson {
    declared: bool,
    sources: Vec<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DependencyJson {
    kind: String,
    name: String,
    version: String,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DependenciesJson {
    status: String,
    items: Vec<DependencyJson>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct ReplayJson {
    kind: String,
    stream_id: Option<String>,
    exposes_state_identity: Option<bool>,
    retry: Option<String>,
    max_attempts: Option<u32>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct TrustJson {
    level: String,
    attestor: Option<String>,
    receipt_digest: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct UpdateJson {
    candidate_id: String,
    prior_id: String,
    observation: ScientificQuantityWire,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct InputJson {
    name: String,
    quantity: ScientificQuantityWire,
    semantic_identity: String,
    support: SupportJson,
    operation: String,
    uncertainty: Option<String>,
    supplier: String,
    callback_identity: Option<String>,
    evidence: EvidenceJson,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct DeclarationJson {
    component: String,
    kind: String,
    identity: IdentityJson,
    quantity: ScientificQuantityWire,
    semantic_identity: String,
    meaning: String,
    support: SupportJson,
    operations: Vec<String>,
    update: Option<UpdateJson>,
    inputs: Vec<InputJson>,
    dependencies: DependenciesJson,
    replay: ReplayJson,
    trust: TrustJson,
    evidence: EvidenceJson,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct QueryJson {
    operation: String,
    draws: Option<u32>,
    points: Vec<f64>,
    observation: Option<f64>,
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct RequestJson {
    query: QueryJson,
    seed: u64,
    input_digests: BTreeMap<String, String>,
    idempotency_key: Option<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct PolicyJson {
    max_invocations: u64,
    max_output_values: usize,
    wall_limit_ms: Option<u64>,
    allow_side_effects: bool,
    phase: String,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OutputBitsJson {
    operation: String,
    values_bits: Vec<u64>,
}

// -- conversions -------------------------------------------------------------------------------

fn bad(what: &str, detail: impl std::fmt::Display) -> PyErr {
    crate::value_err(format!("invalid {what}: {detail}"))
}

fn evidence_of(json: &EvidenceJson) -> PyResult<EvidenceBasis> {
    if !json.declared {
        if json.sources.is_empty() {
            return Ok(EvidenceBasis::Undeclared);
        }
        return Err(bad("evidence", "undeclared evidence lists sources"));
    }
    let mut set = std::collections::BTreeSet::new();
    for hex in &json.sources {
        set.insert(
            SourceId::from_hex(hex).ok_or_else(|| bad("evidence", "a source is not a digest"))?,
        );
    }
    Ok(EvidenceBasis::Declared(set))
}

fn evidence_json(evidence: &EvidenceBasis) -> EvidenceJson {
    match evidence {
        EvidenceBasis::Undeclared => EvidenceJson { declared: false, sources: Vec::new() },
        EvidenceBasis::Declared(sources) => {
            EvidenceJson { declared: true, sources: sources.iter().map(SourceId::to_hex).collect() }
        }
    }
}

fn support_of(json: &SupportJson) -> PyResult<SupportDomain> {
    match (json.kind.as_str(), json.lower, json.upper) {
        ("real", None, None) => Ok(SupportDomain::Real),
        ("non_negative", None, None) => Ok(SupportDomain::NonNegative),
        ("binary", None, None) => Ok(SupportDomain::Binary),
        ("interval", Some(lower), Some(upper)) => Ok(SupportDomain::interval(lower, upper)),
        _ => Err(bad("support", "a numeric support is real, non_negative, binary or an interval")),
    }
}

fn support_json(support: &SupportDomain) -> SupportJson {
    match support {
        SupportDomain::Interval { lower_bits, upper_bits } => SupportJson {
            kind: "interval".into(),
            lower: Some(f64::from_bits(*lower_bits)),
            upper: Some(f64::from_bits(*upper_bits)),
        },
        other => SupportJson { kind: other.kind().to_owned(), lower: None, upper: None },
    }
}

fn quantity_of(wire: &ScientificQuantityWire) -> PyResult<ScientificQuantity> {
    ScientificQuantity::try_from(wire.clone()).map_err(|reason| bad("quantity", reason))
}

fn operation_of(name: &str) -> PyResult<CallbackOperation> {
    CallbackOperation::from_name(name).ok_or_else(|| bad("operation", name))
}

#[allow(
    clippy::too_many_lines,
    reason = "one flat conversion of every declared field between the JSON and typed forms"
)]
fn declaration_of(text: &str) -> PyResult<CallbackDeclaration> {
    let value = parse(text, "callback declaration")?;
    let json: DeclarationJson =
        serde_json::from_value(value).map_err(|error| bad("callback declaration", error))?;
    let kind = CallbackKind::from_name(&json.kind).ok_or_else(|| bad("kind", &json.kind))?;
    let meaning = UncertaintyMeaning::from_name(&json.meaning)
        .ok_or_else(|| bad("meaning", &json.meaning))?
        .meaning;
    let mut operations = std::collections::BTreeSet::new();
    for name in &json.operations {
        operations.insert(operation_of(name)?);
    }
    let update = match &json.update {
        None => None,
        Some(update) => Some(UpdateDeclaration {
            candidate_id: update.candidate_id.clone(),
            prior_id: update.prior_id.clone(),
            observation: quantity_of(&update.observation)?,
        }),
    };
    let mut inputs = Vec::with_capacity(json.inputs.len());
    for input in &json.inputs {
        let supplier = match (input.supplier.as_str(), &input.callback_identity) {
            ("external", None) => InputSupplier::External,
            ("callback", Some(identity)) => InputSupplier::Callback {
                identity: identity.clone(),
                evidence: evidence_of(&input.evidence)?,
            },
            _ => return Err(bad("input supplier", &input.supplier)),
        };
        let uncertainty = match &input.uncertainty {
            None => None,
            Some(name) => Some(
                UncertaintyMeaning::from_name(name)
                    .ok_or_else(|| bad("input meaning", name))?
                    .meaning,
            ),
        };
        inputs.push(CallbackInput {
            name: input.name.clone(),
            quantity: quantity_of(&input.quantity)?,
            semantic_identity: input.semantic_identity.clone(),
            support: support_of(&input.support)?,
            operation: operation_of(&input.operation)?,
            uncertainty,
            supplier,
        });
    }
    let dependencies = match json.dependencies.status.as_str() {
        "complete" => {
            let mut items = Vec::new();
            for item in &json.dependencies.items {
                let kind = DependencyKind::from_name(&item.kind)
                    .ok_or_else(|| bad("dependency kind", &item.kind))?;
                items.push(CallbackDependency::new(kind, item.name.clone(), item.version.clone()));
            }
            DependencyDeclaration::Complete(items)
        }
        "opaque" => DependencyDeclaration::Opaque,
        "undeclared" => DependencyDeclaration::Undeclared,
        other => return Err(bad("dependency status", other)),
    };
    let replay = match (json.replay.kind.as_str(), &json.replay) {
        ("deterministic_pure", _) => ReplayContract::DeterministicPure,
        ("seeded_stochastic", r) => {
            ReplayContract::SeededStochastic { stream_id: r.stream_id.clone().unwrap_or_default() }
        }
        ("stateful", r) => ReplayContract::Stateful {
            exposes_state_identity: r.exposes_state_identity.unwrap_or(false),
        },
        ("side_effecting", r) => ReplayContract::SideEffecting {
            retry: match (r.retry.as_deref(), r.max_attempts) {
                (Some("idempotent_key"), Some(max_attempts)) => {
                    RetryContract::IdempotentKey { max_attempts }
                }
                (Some("no_retry") | None, _) => RetryContract::NoRetry,
                _ => return Err(bad("retry contract", "unknown")),
            },
        },
        (other, _) => return Err(bad("replay contract", other)),
    };
    let trust = match json.trust.level.as_str() {
        "unverified" => CallbackTrust::Unverified,
        "externally_attested" => CallbackTrust::ExternallyAttested {
            attestor: json.trust.attestor.clone().unwrap_or_default(),
        },
        "verified_extension" => CallbackTrust::ExactRequestVerified {
            receipt_digest: json.trust.receipt_digest.clone().unwrap_or_default(),
        },
        // Metadata never confers native licence; there is no such level to build.
        other => return Err(bad("trust level", other)),
    };
    Ok(CallbackDeclaration {
        component: json.component.clone(),
        kind,
        identity: ProviderObjectIdentity {
            provider_id: json.identity.provider_id.clone(),
            object_id: json.identity.object_id.clone(),
            version_id: json.identity.version_id.clone(),
            snapshot_id: json.identity.snapshot_id.clone(),
            request_id: json.identity.request_id.clone(),
        },
        quantity: quantity_of(&json.quantity)?,
        semantic_identity: json.semantic_identity.clone(),
        meaning,
        support: support_of(&json.support)?,
        operations,
        update,
        inputs,
        dependencies,
        replay,
        trust,
        evidence: evidence_of(&json.evidence)?,
    })
}

#[allow(
    clippy::too_many_lines,
    reason = "one flat conversion of every declared field between the JSON and typed forms"
)]
fn declaration_json(declaration: &CallbackDeclaration) -> DeclarationJson {
    let dependencies = match &declaration.dependencies {
        DependencyDeclaration::Complete(items) => DependenciesJson {
            status: "complete".into(),
            items: items
                .iter()
                .map(|d| DependencyJson {
                    kind: d.kind.as_str().into(),
                    name: d.name.clone(),
                    version: d.version.clone(),
                })
                .collect(),
        },
        DependencyDeclaration::Opaque => {
            DependenciesJson { status: "opaque".into(), items: Vec::new() }
        }
        DependencyDeclaration::Undeclared => {
            DependenciesJson { status: "undeclared".into(), items: Vec::new() }
        }
    };
    let replay = match &declaration.replay {
        ReplayContract::DeterministicPure => ReplayJson {
            kind: "deterministic_pure".into(),
            stream_id: None,
            exposes_state_identity: None,
            retry: None,
            max_attempts: None,
        },
        ReplayContract::SeededStochastic { stream_id } => ReplayJson {
            kind: "seeded_stochastic".into(),
            stream_id: Some(stream_id.clone()),
            exposes_state_identity: None,
            retry: None,
            max_attempts: None,
        },
        ReplayContract::Stateful { exposes_state_identity } => ReplayJson {
            kind: "stateful".into(),
            stream_id: None,
            exposes_state_identity: Some(*exposes_state_identity),
            retry: None,
            max_attempts: None,
        },
        ReplayContract::SideEffecting { retry } => ReplayJson {
            kind: "side_effecting".into(),
            stream_id: None,
            exposes_state_identity: None,
            retry: Some(
                match retry {
                    RetryContract::NoRetry => "no_retry",
                    RetryContract::IdempotentKey { .. } => "idempotent_key",
                }
                .into(),
            ),
            max_attempts: match retry {
                RetryContract::NoRetry => None,
                RetryContract::IdempotentKey { max_attempts } => Some(*max_attempts),
            },
        },
    };
    let trust = match &declaration.trust {
        CallbackTrust::Unverified => {
            TrustJson { level: "unverified".into(), attestor: None, receipt_digest: None }
        }
        CallbackTrust::ExternallyAttested { attestor } => TrustJson {
            level: "externally_attested".into(),
            attestor: Some(attestor.clone()),
            receipt_digest: None,
        },
        CallbackTrust::ExactRequestVerified { receipt_digest } => TrustJson {
            level: "verified_extension".into(),
            attestor: None,
            receipt_digest: Some(receipt_digest.clone()),
        },
    };
    DeclarationJson {
        component: declaration.component.clone(),
        kind: declaration.kind.as_str().into(),
        identity: IdentityJson {
            provider_id: declaration.identity.provider_id.clone(),
            object_id: declaration.identity.object_id.clone(),
            version_id: declaration.identity.version_id.clone(),
            snapshot_id: declaration.identity.snapshot_id.clone(),
            request_id: declaration.identity.request_id.clone(),
        },
        quantity: ScientificQuantityWire::from(&declaration.quantity),
        semantic_identity: declaration.semantic_identity.clone(),
        meaning: UncertaintyMeaning::new(declaration.meaning).name().into(),
        support: support_json(&declaration.support),
        operations: declaration.operations.iter().map(|o| o.as_str().to_owned()).collect(),
        update: declaration.update.as_ref().map(|u| UpdateJson {
            candidate_id: u.candidate_id.clone(),
            prior_id: u.prior_id.clone(),
            observation: ScientificQuantityWire::from(&u.observation),
        }),
        inputs: declaration
            .inputs
            .iter()
            .map(|input| {
                let (supplier, callback_identity, evidence) = match &input.supplier {
                    InputSupplier::External => {
                        ("external", None, EvidenceJson { declared: false, sources: Vec::new() })
                    }
                    InputSupplier::Callback { identity, evidence } => {
                        ("callback", Some(identity.clone()), evidence_json(evidence))
                    }
                };
                InputJson {
                    name: input.name.clone(),
                    quantity: ScientificQuantityWire::from(&input.quantity),
                    semantic_identity: input.semantic_identity.clone(),
                    support: support_json(&input.support),
                    operation: input.operation.as_str().into(),
                    uncertainty: input
                        .uncertainty
                        .map(|m| UncertaintyMeaning::new(m).name().into()),
                    supplier: supplier.into(),
                    callback_identity,
                    evidence,
                }
            })
            .collect(),
        dependencies,
        replay,
        trust,
        evidence: evidence_json(&declaration.evidence),
    }
}

fn query_of(json: &QueryJson) -> PyResult<CallbackQuery> {
    let operation = operation_of(&json.operation)?;
    Ok(match operation {
        CallbackOperation::Mean => CallbackQuery::Mean,
        CallbackOperation::Sample => {
            CallbackQuery::Sample { draws: json.draws.ok_or_else(|| bad("query", "draws"))? }
        }
        CallbackOperation::Cdf => CallbackQuery::Cdf { thresholds: json.points.clone() },
        CallbackOperation::Quantile => CallbackQuery::Quantile { levels: json.points.clone() },
        CallbackOperation::PosteriorUpdate => CallbackQuery::PosteriorUpdate {
            observation: json.observation.ok_or_else(|| bad("query", "observation"))?,
            draws: json.draws.unwrap_or(0),
        },
    })
}

fn query_json(query: &CallbackQuery) -> QueryJson {
    let blank = |operation: &str| QueryJson {
        operation: operation.into(),
        draws: None,
        points: Vec::new(),
        observation: None,
    };
    match query {
        CallbackQuery::Mean => blank("mean"),
        CallbackQuery::Sample { draws } => QueryJson { draws: Some(*draws), ..blank("sample") },
        CallbackQuery::Cdf { thresholds } => {
            QueryJson { points: thresholds.clone(), ..blank("cdf") }
        }
        CallbackQuery::Quantile { levels } => {
            QueryJson { points: levels.clone(), ..blank("quantile") }
        }
        CallbackQuery::PosteriorUpdate { observation, draws } => QueryJson {
            draws: Some(*draws),
            observation: Some(*observation),
            ..blank("posterior_update")
        },
    }
}

fn request_of(text: &str) -> PyResult<CallbackRequest> {
    let value = parse(text, "callback request")?;
    let json: RequestJson =
        serde_json::from_value(value).map_err(|error| bad("callback request", error))?;
    Ok(CallbackRequest {
        query: query_of(&json.query)?,
        seed: json.seed,
        input_digests: json.input_digests,
        idempotency_key: json.idempotency_key,
    })
}

fn policy_of(text: &str) -> PyResult<CallbackExecutionPolicy> {
    let value = parse(text, "execution policy")?;
    let json: PolicyJson =
        serde_json::from_value(value).map_err(|error| bad("execution policy", error))?;
    let phase = match json.phase.as_str() {
        "planning" => CallbackPhase::Planning,
        "verification" => CallbackPhase::Verification,
        "execution" => CallbackPhase::Execution,
        other => return Err(bad("phase", other)),
    };
    Ok(CallbackExecutionPolicy {
        max_invocations: json.max_invocations,
        max_output_values: json.max_output_values,
        wall_limit_ms: json.wall_limit_ms,
        allow_side_effects: json.allow_side_effects,
        phase,
    })
}

fn output_json(output: &ValidatedOutput) -> Json {
    json!({
        "operation": output.output().operation().as_str(),
        "values": output.output().values(),
    })
}

fn output_of(parsed: &OutputBitsJson) -> Result<CallbackOutput, String> {
    // The output is built for the operation it names; a mismatch with the query is refused by
    // the Rust output check, not here.
    let values: Vec<f64> = parsed.values_bits.iter().map(|b| f64::from_bits(*b)).collect();
    let operation = CallbackOperation::from_name(&parsed.operation)
        .ok_or_else(|| "the output names an unknown operation".to_owned())?;
    Ok(match operation {
        CallbackOperation::Mean => match values.as_slice() {
            [value] => CallbackOutput::Mean { value: *value },
            _ => return Err("a mean is one value".into()),
        },
        CallbackOperation::Sample => CallbackOutput::Sample { draws: values },
        CallbackOperation::Cdf => CallbackOutput::Cdf { values },
        CallbackOperation::Quantile => CallbackOutput::Quantile { values },
        CallbackOperation::PosteriorUpdate => match values.split_first() {
            Some((mean, draws)) => {
                CallbackOutput::PosteriorUpdate { mean: *mean, draws: draws.to_vec() }
            }
            None => return Err("a posterior update has at least its mean".into()),
        },
    })
}

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

struct PyProvider {
    identity: ProviderObjectIdentity,
    invoke: Py<PyAny>,
    state: Option<Py<PyAny>>,
    token: CallbackToken,
}

impl ProviderCallback for PyProvider {
    fn identity(&self) -> &ProviderObjectIdentity {
        &self.identity
    }

    fn invoke(
        &mut self,
        query: &CallbackQuery,
        ctx: &mut CallbackContext<'_>,
    ) -> Result<CallbackOutput, CallbackFailure> {
        let inputs: BTreeMap<&String, Json> =
            ctx.inputs.iter().map(|(name, value)| (name, output_json(value))).collect();
        let context = json!({
            "stream_seed": ctx.rng.as_ref().map(antecedent_core::CausalRng::state),
            "attempt": ctx.attempt,
            "idempotency_key": ctx.idempotency_key,
            "inputs": inputs,
        });
        let query_text = serde_json::to_string(&query_json(query))
            .map_err(|error| CallbackFailure::new(error.to_string()))?;
        let token = self.token.clone();
        // The caller's thread holds the GIL; attach is re-entrant.
        Python::attach(|py| {
            let called = self
                .invoke
                .bind(py)
                .call1((query_text, context.to_string(), token))
                .map_err(|error| CallbackFailure::new(format!("{}", error.value(py))))?;
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

    fn state_identity(&self) -> Option<String> {
        let state = self.state.as_ref()?;
        Python::attach(|py| {
            state
                .bind(py)
                .call0()
                .ok()
                .and_then(|value| value.extract::<Option<String>>().ok())
                .flatten()
        })
    }
}

fn provider_of(
    identity_json: &str,
    invoke: Py<PyAny>,
    state: Option<Py<PyAny>>,
    token: &CallbackToken,
) -> PyResult<PyProvider> {
    let value = parse(identity_json, "provider identity")?;
    let identity: IdentityJson =
        serde_json::from_value(value).map_err(|error| bad("provider identity", error))?;
    Ok(PyProvider {
        identity: ProviderObjectIdentity {
            provider_id: identity.provider_id,
            object_id: identity.object_id,
            version_id: identity.version_id,
            snapshot_id: identity.snapshot_id,
            request_id: identity.request_id,
        },
        invoke,
        state,
        token: token.clone(),
    })
}

fn context_of(token: &CallbackToken) -> ExecutionContext {
    let mut ctx = ExecutionContext::for_tests(0);
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
        self.inner
            .history()
            .get(index)
            .map(|record| record_json(record).to_string())
            .ok_or_else(|| crate::value_err("history index out of range"))
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
        let declaration = declaration_of(declaration_json)?;
        let request = request_of(request_json)?;
        let policy = policy_of(policy_json)?;
        let inputs = inputs_of(&self.inner, inputs_json)?;
        let mut provider = provider_of(identity_json, invoke, state, token)?;
        let ctx = context_of(token);
        Ok(
            match self.inner.execute(&mut provider, &declaration, &request, &inputs, &policy, &ctx)
            {
                Ok(execution) => (Some(execution_json(&self.inner, &execution)), None),
                Err(error) => (None, Some(error_json(&error))),
            },
        )
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
        let policy = policy_of(policy_json)?;
        let inputs = inputs_of(&self.inner, inputs_json)?;
        let ctx = context_of(token);
        let mut provider = match (identity_json, invoke) {
            (Some(identity), Some(invoke)) => Some(provider_of(identity, invoke, state, token)?),
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
    }

    /// The empirical-sample forecast law of the accepted sample output at a history index, as the
    /// predictive-assessment law JSON, or the typed refusal when the output is not a sample.
    fn forecast_law(&self, index: usize) -> PyResult<Checked> {
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
    }

    /// Export one history record as `callback_result_v1`.
    fn export_record(
        &self,
        py: Python<'_>,
        index: usize,
        artifact_id: &str,
    ) -> PyResult<ExportedRecord> {
        let record = self
            .inner
            .history()
            .get(index)
            .ok_or_else(|| crate::value_err("history index out of range"))?;
        Ok(match CallbackResultArtifact::to_bytes(record, artifact_id) {
            Ok(bytes) => {
                let identity = CallbackResultArtifact::from_bytes(&bytes, None)
                    .map(|a| a.identity().to_owned())
                    .unwrap_or_default();
                (Some((PyBytes::new(py, &bytes).unbind(), identity)), None)
            }
            Err(error) => (None, Some(artifact_error_json(&error))),
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
            "code": antecedent_core::reason_code!("invalid_argument"),
            "stage": "consume",
            "detail": "callback_protocol.artifact_invalid",
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
}

/// The request identity (hex) against a declaration.
#[pyfunction]
fn callback_request_identity(declaration_json: &str, request_json: &str) -> PyResult<String> {
    let declaration = declaration_of(declaration_json)?;
    let request = request_of(request_json)?;
    Ok(request.identity(&declaration).to_hex())
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
    match CallbackResultArtifact::inspect(artifact) {
        Ok(meta) => (Some(serde_json::to_value(&meta).unwrap_or(Json::Null).to_string()), None),
        Err(error) => (None, Some(artifact_error_json(&error))),
    }
}

/// The refusal of a route that stays closed: `interval`, `posterior_guarantee`,
/// `native_trust`, `identification_licence` or `joint_law`.
#[pyfunction]
fn callback_closed_refusal(route: &str) -> PyResult<String> {
    let refusal = match route {
        "interval" => refuse_interval(),
        "posterior_guarantee" => refuse_posterior_guarantee(),
        "native_trust" => refuse_native_trust(),
        "identification_licence" => refuse_identification_licence(),
        "joint_law" => refuse_joint_law(),
        other => return Err(bad("closed route", other)),
    };
    Ok(refusal_json(&refusal).to_string())
}

/// The predictive-assessment law source of a declared callback, as the law-source JSON, or the
/// refusal of a malformed declaration or a diagnostic standing. Runs nothing.
#[pyfunction]
fn callback_predictive_law_source(
    declaration_json: &str,
    verification_confirmed: bool,
) -> PyResult<Checked> {
    let declaration = declaration_of(declaration_json)?;
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
    let producer = contract_of(producer_json, "producer contract")?;
    let declaration = declaration_of(callback_json)?;
    Ok(match check_claim_to_callback(&producer, produced_port, &declaration, input) {
        Ok(connection) => (Some(connection_json(&connection)), None),
        Err(refusals) => {
            (None, Some(Json::Array(refusals.iter().map(refusal_json).collect()).to_string()))
        }
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
    let declaration = declaration_of(declaration_json)?;
    let inner = inner_sources_of(inner_sources_json)?;
    Ok(match ledger.session_mut().register_callback(&declaration, &inner) {
        Ok(id) => (Some(id.to_hex()), None),
        Err(error) => (None, Some(registration_error_json(&error))),
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
    let old = SourceId::from_hex(old).ok_or_else(|| bad("source", "not a digest"))?;
    let replacement = declaration_of(replacement_json)?;
    let inner = inner_sources_of(inner_sources_json)?;
    Ok(match ledger.session_mut().supersede_callback(old, &replacement, &inner) {
        Ok(id) => (Some(id.to_hex()), None),
        Err(error) => (None, Some(registration_error_json(&error))),
    })
}

// -- transported claim handoff -----------------------------------------------------------------

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct TransportIdentityJson {
    source_populations: Vec<String>,
    target_population: String,
    selection_diagram: String,
    assumptions: Vec<String>,
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct OfferJson {
    component: String,
    port: String,
    quantity: ScientificQuantityWire,
    semantic_identity: String,
    support: SupportJson,
    operations: Vec<String>,
    uncertainty: Option<String>,
    transport: TransportIdentityJson,
    evidence: Vec<String>,
}

fn offer_of(text: &str) -> PyResult<TransportedClaimOffer> {
    let value = parse(text, "transported claim offer")?;
    let json: OfferJson =
        serde_json::from_value(value).map_err(|error| bad("transported claim offer", error))?;
    let mut operations = std::collections::BTreeSet::new();
    for name in &json.operations {
        operations
            .insert(ContractOperation::from_name(name).ok_or_else(|| bad("operation", name))?);
    }
    let uncertainty = match &json.uncertainty {
        None => None,
        Some(name) => {
            Some(UncertaintyMeaning::from_name(name).ok_or_else(|| bad("meaning", name))?.meaning)
        }
    };
    let selection_diagram = parse_digest_hex(&json.transport.selection_diagram)
        .ok_or_else(|| bad("selection diagram", "not a 64 hexadecimal digest"))?;
    let mut evidence = std::collections::BTreeSet::new();
    for hex in &json.evidence {
        evidence.insert(
            SourceId::from_hex(hex).ok_or_else(|| bad("evidence", "a source is not a digest"))?,
        );
    }
    Ok(TransportedClaimOffer {
        component: json.component,
        port: json.port,
        quantity: quantity_of(&json.quantity)?,
        semantic_identity: json.semantic_identity,
        support: support_of(&json.support)?,
        operations,
        uncertainty,
        transport: TransportIdentity {
            source_populations: json.transport.source_populations,
            target_population: json.transport.target_population,
            selection_diagram,
            assumptions: json.transport.assumptions,
        },
        evidence,
    })
}

fn handoff_summary(
    identity: &str,
    transport: &TransportIdentity,
    calibration: &CalibrationStatus,
    standing: &str,
) -> Json {
    let calibration = match calibration {
        CalibrationStatus::None => "none",
        CalibrationStatus::PointOnly => "point_only",
        CalibrationStatus::Unmeasured => "unmeasured",
        CalibrationStatus::Measured { .. } => "measured",
        CalibrationStatus::Inherited { .. } => "inherited",
    };
    json!({
        "identity": identity,
        "standing": standing,
        "native_target_data": false,
        "transport": {
            "source_populations": transport.source_populations,
            "target_population": transport.target_population,
            "selection_diagram": transport.selection_diagram.to_hex(),
            "assumptions": transport.assumptions,
        },
        "calibration": calibration,
    })
}

fn handoff_refusal_json(error: &TransportHandoffRefusal) -> String {
    RefusalJson::new(error.code, "transport_handoff", &error.detail, &error.message).json()
}

/// Seal a transported claim to one input of a callback and verify it against the callback's
/// declaration: `((bytes, identity, summary json), refusal json)`.
#[pyfunction]
fn transport_handoff_seal(
    py: Python<'_>,
    offer_json: &str,
    callback_json: &str,
    input: &str,
    accepts_transported: bool,
    artifact_id: &str,
) -> PyResult<Refused<(Py<PyBytes>, String, String)>> {
    let offer = offer_of(offer_json)?;
    let callback = declaration_of(callback_json)?;
    let policy = if accepts_transported {
        TransportedInputPolicy::AcceptsTransported
    } else {
        TransportedInputPolicy::NativeTargetOnly
    };
    let handoff = match hand_to_callback(&offer, &callback, input, policy) {
        Ok(handoff) => handoff,
        Err(error) => return Ok((None, Some(handoff_refusal_json(&error)))),
    };
    let bytes = match handoff.to_bytes(artifact_id) {
        Ok(bytes) => bytes,
        Err(error) => return Ok((None, Some(handoff_refusal_json(&error)))),
    };
    let calibration = handoff.artifact().handoff().connection().calibration.clone();
    let summary = handoff_summary(
        handoff.identity(),
        handoff.transport(),
        &calibration,
        handoff.artifact().handoff().standing(),
    );
    Ok((
        Some((
            PyBytes::new(py, &bytes).unbind(),
            handoff.identity().to_owned(),
            summary.to_string(),
        )),
        None,
    ))
}

/// Consume a handoff's bytes and verify them against the callback declaration the consumer
/// holds: `(summary json, refusal json)`.
#[pyfunction]
#[pyo3(signature = (artifact, callback_json, expected_identity=None))]
fn transport_handoff_consume(
    artifact: &[u8],
    callback_json: &str,
    expected_identity: Option<&str>,
) -> PyResult<Refused<String>> {
    let callback = declaration_of(callback_json)?;
    Ok(match consume_for_callback(artifact, expected_identity, &callback) {
        Ok(consumed) => {
            let summary = handoff_summary(
                consumed.identity(),
                consumed.transport(),
                &consumed.handoff().connection().calibration,
                consumed.handoff().standing(),
            );
            (Some(summary.to_string()), None)
        }
        Err(error) => (None, Some(handoff_refusal_json(&error))),
    })
}

type Refused<T> = (Option<T>, Option<String>);

pub(crate) fn register(module: &Bound<'_, PyModule>) -> PyResult<()> {
    module.add_class::<CallbackToken>()?;
    module.add_class::<CallbackSessionHandle>()?;
    module.add_function(wrap_pyfunction!(callback_declaration_identity, module)?)?;
    module.add_function(wrap_pyfunction!(callback_request_identity, module)?)?;
    module.add_function(wrap_pyfunction!(callback_values_digest, module)?)?;
    module.add_function(wrap_pyfunction!(callback_result_consume, module)?)?;
    module.add_function(wrap_pyfunction!(callback_result_inspect, module)?)?;
    module.add_function(wrap_pyfunction!(callback_closed_refusal, module)?)?;
    module.add_function(wrap_pyfunction!(callback_predictive_law_source, module)?)?;
    module.add_function(wrap_pyfunction!(callback_check_claim_connection, module)?)?;
    module.add_function(wrap_pyfunction!(callback_register_source, module)?)?;
    module.add_function(wrap_pyfunction!(callback_supersede_source, module)?)?;
    module.add_function(wrap_pyfunction!(transport_handoff_seal, module)?)?;
    module.add_function(wrap_pyfunction!(transport_handoff_consume, module)?)?;
    Ok(())
}
