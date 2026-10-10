//! Executed, attested finite mean callbacks under the original shared planner.
//! Planning never invokes a provider. Metadata establishes no native causal proof or coupling.
//! SPDX-License-Identifier: MIT OR Apache-2.0
use super::recalc_receipt::{Counter, RecalcReceipt, ReceiptRecorder};
use antecedent_core::recalc::{
    Boundary, Branch, RecalcCapabilities, RecalcPlan, RequestSupport, ResumeContext,
    RetargetSupport, Stage, StageIdentities, StageIdentity, StageStatus,
};
use antecedent_core::{
    BoundExternalClaim, CheckedCausalContract, ExecutionContext, ExternalProgramClaim,
    ExternalResponse, ExternalResult, ExternalTrustState, ExternalUncertaintyMeaning,
    ProgramBinding, ProviderObjectIdentity,
};
use antecedent_io::external_callback_artifact::ExternalCallbackArtifact;
use antecedent_io::external_claim_artifact::ExternalClaimArtifact;
use std::collections::{BTreeMap, BTreeSet};
use std::sync::Arc;

/// Declared replay premise. These labels do not prove callback purity.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum CallbackPolicy {
    /// Complete declared dependencies, no effects and no RNG.
    Deterministic,
    /// Complete declared dependencies and the supplied seed, no effects.
    Seeded,
    /// Changing hidden state: invoke on every execution; portable replay is refused.
    Stateful,
    /// Effects may occur even when the call fails. Retry needs an explicit idempotency key.
    SideEffecting,
    /// Dependency completeness or effects are unknown; unavailable in this bounded route.
    Unknown,
}
impl CallbackPolicy {
    /// Durable replay label.
    #[must_use]
    pub const fn label(self) -> &'static str {
        match self {
            Self::Deterministic => "deterministic",
            Self::Seeded => "seeded",
            Self::Stateful => "stateful",
            Self::SideEffecting => "side_effecting",
            Self::Unknown => "unknown",
        }
    }
}
/// Exact implementation, input snapshot, environment and replay declarations.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ProviderDescriptor {
    /// Original external specification's provider execution identity.
    pub identity: ProviderObjectIdentity,
    /// Complete declared environment dependency identity.
    pub environment_id: String,
    /// Declared dependency/replay policy.
    pub policy: CallbackPolicy,
    /// Provider-declared support for retrying the same explicit idempotency key.
    pub idempotency_supported: bool,
}
impl ProviderDescriptor {
    /// Stable complete descriptor digest.
    #[must_use]
    pub fn digest(&self) -> StageIdentity {
        text_identity("callback.descriptor.v1", &format!("{self:?}"))
    }
}
/// Complete bounded callback inputs and the existing scientific program contract.
#[derive(Clone, Debug)]
pub struct ExternalCallbackRequest {
    /// Shared planner branch, in `0..MAX_EXTERNAL_BRANCHES`.
    pub branch: Branch,
    /// Existing full scientific question and explicit action grid.
    pub program: ProgramBinding,
    /// Original external specification's claim about that question.
    pub claim: ExternalProgramClaim,
    /// Original checked contract premises; this adapter performs no causal identification.
    pub contract: CheckedCausalContract,
    /// Required actual provider implementation/environment.
    pub descriptor: ProviderDescriptor,
    /// Actual named raw numeric payload, all rows included in identity.
    pub columns: Vec<(String, Vec<f64>)>,
    /// Declared finite model parameters, in canonical supplied order.
    pub model_parameters: Vec<(String, f64)>,
    /// Actual callback RNG seed, including deterministic callbacks' declared input identity.
    pub seed: u64,
    /// Explicit provider-supported idempotency key, passed unchanged to the callback.
    pub idempotency_key: Option<String>,
}
/// Foreign execution failure; no claim authority is issued from it.
#[derive(Clone, Debug)]
pub struct CallbackFailure {
    /// Bounded supplier diagnostic.
    pub message: String,
}
/// Narrow callback context: all numerical RNG inputs are request-bound.
/// Cancellation may stop execution; host kernel/thread/adaptive settings are not callback inputs.
#[derive(Clone)]
pub struct CallbackContext {
    /// RNG factory constructed from the exact request seed.
    pub rng: antecedent_core::RngFactory,
    /// Cooperative cancellation token; no hard preemption is promised.
    pub cancellation: antecedent_core::CancellationToken,
}
/// Provider implementation supplied in this process, never serialized.
pub trait ExternalMeanProvider {
    /// Actual implementation descriptor. Execution checks it; planning does not call this method.
    fn descriptor(&self) -> &ProviderDescriptor;
    /// One foreign invocation. Implementations may poll cooperative cancellation.
    /// # Errors
    /// Provider-defined execution failure; no successful claim is issued from it.
    fn invoke(
        &mut self,
        request: &ExternalCallbackRequest,
        ctx: &CallbackContext,
    ) -> Result<ExternalResponse, CallbackFailure>;
}
/// Attempt evidence survives failure without granting scientific output authority.
#[derive(Clone, Debug)]
pub struct InvocationAttempt {
    /// Exact attempted request.
    pub request_digest: StageIdentity,
    /// Actual callback entries, including failed calls.
    pub invocations: u64,
}
/// Typed executor refusal, with an optional non-authoritative attempt report.
#[derive(Debug)]
pub enum ExternalCallbackError {
    /// Preflight or output validation refusal.
    Request(&'static str),
    /// Original scientific binding refusal.
    Binding(Box<antecedent_core::ExternalRefusal>),
    /// Original binding refusal after a counted callback invocation.
    BindingAttempt {
        /// Original typed scientific refusal.
        refusal: Box<antecedent_core::ExternalRefusal>,
        /// Actual callback entries before the refusal.
        report: InvocationAttempt,
    },
    /// Invocation failed, was cancelled, or returned an invalid output.
    Attempt {
        /// Stable executor refusal detail.
        detail: &'static str,
        /// Actual callback entries before failure.
        report: InvocationAttempt,
        /// Bounded supplier or executor diagnostic.
        message: String,
    },
    /// Original shared planner refusal.
    Refused(Box<RecalcPlan>),
    /// Receipt or original portable artifact encoding error.
    Artifact(String),
}
impl ExternalCallbackError {
    /// Actual registered refusal code, preserving original binding and planner reasons.
    #[must_use]
    pub fn refusal_code(&self) -> &'static str {
        match self {
            Self::Binding(refusal) | Self::BindingAttempt { refusal, .. } => refusal.code,
            Self::Refused(plan) => plan
                .entries()
                .iter()
                .find_map(|entry| {
                    if let StageStatus::Refused { reason } = entry.status {
                        Some(reason.reason_code())
                    } else {
                        None
                    }
                })
                .unwrap_or(antecedent_core::reason_code!("route_not_supported")),
            Self::Request(detail) | Self::Attempt { detail, .. } => match *detail {
                "recalc.cancelled" => antecedent_core::reason_code!("cancelled_no_claim"),
                "recalc.unavailable_provider" => {
                    antecedent_core::reason_code!("score_table_unavailable")
                }
                "external_recalc.policy_unsupported"
                | "external_recalc.quantity_unsupported"
                | "external_recalc.replay_unsupported"
                | "external_recalc.retry_unsafe" => {
                    antecedent_core::reason_code!("route_not_supported")
                }
                _ => antecedent_core::reason_code!("invalid_argument"),
            },
            Self::Artifact(_) => antecedent_core::reason_code!("invalid_argument"),
        }
    }
}
impl std::fmt::Display for ExternalCallbackError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
impl std::error::Error for ExternalCallbackError {}
/// Successfully issued attested mean response and actual branch work.
#[derive(Clone, Debug)]
pub struct ExternalCallbackOutcome {
    /// Original shared dependency plan.
    pub plan: RecalcPlan,
    /// Actual successful branch execution receipt.
    pub receipt: RecalcReceipt,
    /// Original checked external claim, remaining externally attested.
    pub claim: Arc<BoundExternalClaim>,
    /// Actual previous identities used by this execution, for independent receipt sealing.
    pub previous: StageIdentities,
    /// Actual requested identities used by this execution.
    pub requested: StageIdentities,
    /// Actual provider boundary and request capabilities used by this execution.
    pub capabilities: RecalcCapabilities,
}
#[derive(Clone)]
struct LiveOutput {
    claim: Arc<BoundExternalClaim>,
    artifact: Arc<ExternalCallbackArtifact>,
}
/// Transactional output retention and a separate non-retry ledger for attempted side effects.
#[derive(Clone)]
pub struct ExternalCallbackSession {
    previous: StageIdentities,
    live: Option<LiveOutput>,
    portable: Option<Arc<ExternalCallbackArtifact>>,
    attempts: BTreeSet<StageIdentity>,
    idempotency_bindings: BTreeMap<StageIdentity, StageIdentity>,
    epoch: u64,
}
impl Default for ExternalCallbackSession {
    fn default() -> Self {
        Self::new()
    }
}
thread_local! { static INVOCATIONS: std::cell::Cell<u64> = const { std::cell::Cell::new(0) }; }
/// Observe actual callback entries; nested observers see the same actual calls.
pub fn count_external_invocations<T>(work: impl FnOnce() -> T) -> (T, u64) {
    let before = INVOCATIONS.get();
    let result = work();
    (result, INVOCATIONS.get().saturating_sub(before))
}
fn text_identity(domain: &str, text: &str) -> StageIdentity {
    StageIdentity::of(domain, &[text.as_bytes()])
}
pub(crate) fn quantity_bounded(quantity: &antecedent_core::ScientificQuantity) -> bool {
    quantity.conditioning.len() <= 64
        && [
            &quantity.variable_id,
            &quantity.variable_name,
            &quantity.units,
            &quantity.population_id,
            &quantity.regime_id,
            &quantity.functional_id,
            &quantity.transform_id,
        ]
        .into_iter()
        .all(|text| !text.is_empty() && text.len() <= 256)
        && quantity.conditioning.iter().all(|condition| {
            !condition.variable_id.is_empty()
                && condition.variable_id.len() <= 256
                && !condition.value_id.is_empty()
                && condition.value_id.len() <= 256
        })
}
fn response_bounded(response: &ExternalResponse) -> bool {
    let antecedent_core::ExternalScientificObject::Law(law) = &response.header.object else {
        return false;
    };
    response.header.graph_id.len() <= 256
        && response.header.quantities.len() <= 64
        && response.header.quantities.iter().all(quantity_bounded)
        && law.quantities.len() <= 64
        && law.quantities.iter().all(quantity_bounded)
        && law.capabilities.len() <= 32
        && law.capabilities.contains(&antecedent_core::ExternalCapability::Mean)
        && response.header.evidence_ids.len() <= 128
        && response.header.assumption_ids.len() <= 128
        && response
            .header
            .evidence_ids
            .iter()
            .chain(&response.header.assumption_ids)
            .all(|text| !text.is_empty() && text.len() <= 256)
}
impl ExternalCallbackRequest {
    /// Full input digest. Validate bounded request before using this method on untrusted inputs.
    #[must_use]
    pub fn digest(&self) -> StageIdentity {
        let mut parts = vec![
            format!(
                "{:?}:{:?}:{:?}:{:?}:{:?}",
                self.program,
                self.claim,
                self.contract,
                self.descriptor,
                (self.seed, &self.idempotency_key)
            )
            .into_bytes(),
        ];
        for (name, values) in &self.columns {
            parts.push(name.as_bytes().to_vec());
            let mut hash = blake3::Hasher::new_derive_key("antecedent.callback.column.v1");
            hash.update(&u64::try_from(values.len()).unwrap_or(u64::MAX).to_le_bytes());
            for value in values {
                hash.update(&value.to_bits().to_le_bytes());
            }
            parts.push(hash.finalize().as_bytes().to_vec());
        }
        for (name, value) in &self.model_parameters {
            parts.push(name.as_bytes().to_vec());
            parts.push(value.to_bits().to_le_bytes().to_vec());
        }
        StageIdentity::of(
            "callback.request.v1",
            &parts.iter().map(Vec::as_slice).collect::<Vec<_>>(),
        )
    }
    pub(crate) fn validate(&self) -> Result<(), ExternalCallbackError> {
        if self.branch.index() >= antecedent_core::recalc::MAX_EXTERNAL_BRANCHES
            || self.descriptor.policy == CallbackPolicy::Unknown
        {
            return Err(ExternalCallbackError::Request("external_recalc.policy_unsupported"));
        }
        self.metadata_guard()?;
        if self.idempotency_key.is_some() && !self.descriptor.idempotency_supported {
            return Err(ExternalCallbackError::Request("external_recalc.idempotency_unsupported"));
        }
        let rows = self.columns.first().map_or(0, |(_, values)| values.len());
        if self.columns.len() > 64
            || rows > 100_000
            || rows.saturating_mul(self.columns.len()) > 1_000_000
            || self.columns.iter().any(|(name, values)| {
                name.is_empty()
                    || name.len() > 256
                    || values.len() != rows
                    || values.iter().any(|v| !v.is_finite())
            })
            || self.model_parameters.len() > 128
            || self
                .model_parameters
                .iter()
                .any(|(name, value)| name.is_empty() || name.len() > 256 || !value.is_finite())
            || self.idempotency_key.as_ref().is_some_and(|key| key.is_empty() || key.len() > 256)
        {
            return Err(ExternalCallbackError::Request("external_recalc.invalid_request"));
        }
        let column_names = self.columns.iter().map(|(name, _)| name).collect::<BTreeSet<_>>();
        let parameter_names =
            self.model_parameters.iter().map(|(name, _)| name).collect::<BTreeSet<_>>();
        if column_names.len() != self.columns.len()
            || parameter_names.len() != self.model_parameters.len()
        {
            return Err(ExternalCallbackError::Request("external_recalc.invalid_request"));
        }
        antecedent_core::check_external_against_program(&self.program, &self.claim)
            .map_err(|error| ExternalCallbackError::Binding(Box::new(error)))?;
        if self.program.functional_id != "mean" || self.program.transform_id != "identity" {
            return Err(ExternalCallbackError::Request("external_recalc.quantity_unsupported"));
        }
        if self.contract.graph_id != self.program.graph_id
            || self.contract.estimand != self.program.expected_quantities()
        {
            return Err(ExternalCallbackError::Request("external_recalc.contract_mismatch"));
        }
        Ok(())
    }
    fn metadata_guard(&self) -> Result<(), ExternalCallbackError> {
        let bounded = |text: &String| !text.is_empty() && text.len() <= 256;
        let p = &self.program;
        let c = &self.claim;
        let d = &self.descriptor;
        if p.dose_grid.len() > 64
            || c.doses.len() > 64
            || c.quantities.len() > 64
            || self.contract.estimand.len() > 64
            || self.contract.required_evidence_ids.len() > 128
            || self.contract.accepted_meanings.len() > 32
            || self.contract.required_assumption_ids.len() > 128
            || !self.contract.equivalences.is_empty()
            || [
                &p.graph_id,
                &p.contract_id,
                &p.treatment_id,
                &p.outcome_id,
                &p.population_id,
                &p.intervention_kind,
                &p.dose_units,
                &p.outcome_units,
                &p.functional_id,
                &p.transform_id,
                &c.contract_id,
                &c.graph_id,
                &c.declared_identity,
                &c.treatment_id,
                &c.outcome_id,
                &c.population_id,
                &c.dose_units,
                &d.identity.provider_id,
                &d.identity.object_id,
                &d.identity.version_id,
                &d.identity.snapshot_id,
                &d.identity.request_id,
                &d.environment_id,
                &self.contract.graph_id,
            ]
            .into_iter()
            .any(|text| !bounded(text))
            || self
                .contract
                .required_evidence_ids
                .iter()
                .chain(&self.contract.required_assumption_ids)
                .any(|text| !bounded(text))
            || self.claim.quantities.iter().chain(&self.contract.estimand).any(|quantity| {
                quantity.conditioning.len() > 64
                    || [
                        &quantity.variable_id,
                        &quantity.variable_name,
                        &quantity.units,
                        &quantity.population_id,
                        &quantity.regime_id,
                        &quantity.functional_id,
                        &quantity.transform_id,
                    ]
                    .into_iter()
                    .any(|text| !bounded(text))
                    || quantity.conditioning.iter().any(|condition| {
                        !bounded(&condition.variable_id) || !bounded(&condition.value_id)
                    })
            })
        {
            return Err(ExternalCallbackError::Request("external_recalc.invalid_request"));
        }
        Ok(())
    }
    pub(crate) fn identities(&self, epoch: u64) -> StageIdentities {
        let mut ids = StageIdentities::new();
        ids.set(Stage::Graph, text_identity("callback.graph.v1", &self.program.graph_id));
        ids.set(Stage::Query, text_identity("callback.program.v1", &self.program.identity()));
        ids.set(Stage::TreatmentGrid, text_identity("callback.grid.v1", &self.program.identity()));
        ids.set(Stage::ExternalStudy(self.branch), self.digest());
        ids.set(
            Stage::ProviderRequest(self.branch),
            text_identity(
                "callback.operation.v1",
                &format!("{}:{epoch}", self.descriptor.policy.label()),
            ),
        );
        ids
    }
}
impl ExternalCallbackSession {
    /// Empty process-local executor; no provider or claim is assumed.
    #[must_use]
    pub fn new() -> Self {
        Self {
            previous: StageIdentities::new(),
            live: None,
            portable: None,
            attempts: BTreeSet::new(),
            idempotency_bindings: BTreeMap::new(),
            epoch: 0,
        }
    }
    /// Actual issued claim, absent until successful native transaction finalization.
    #[must_use]
    pub fn issued_claim(&self) -> Option<&BoundExternalClaim> {
        self.live.as_ref().map(|live| live.claim.as_ref())
    }
    /// Last successful shared stage identities.
    #[must_use]
    pub const fn identities(&self) -> &StageIdentities {
        &self.previous
    }
    pub(crate) fn plan_inputs(
        &self,
        request: &ExternalCallbackRequest,
        supplied: bool,
    ) -> (StageIdentities, StageIdentities, RecalcCapabilities) {
        let supported = request.validate().is_ok();
        let epoch = if matches!(
            request.descriptor.policy,
            CallbackPolicy::Stateful | CallbackPolicy::SideEffecting
        ) {
            self.epoch
        } else {
            0
        };
        let mut previous = self.previous.clone();
        if self.portable.is_some() {
            previous.remove(Stage::ProviderRequest(request.branch));
        }
        let requested = if supported {
            request.identities(epoch)
        } else {
            let mut ids = self.previous.clone();
            ids.set(Stage::TreatmentGrid, text_identity("callback.invalid.v1", "unsupported"));
            ids.set(
                Stage::ProviderRequest(request.branch),
                text_identity("callback.invalid.v1", "unsupported"),
            );
            ids
        };
        (
            previous,
            requested,
            RecalcCapabilities {
                retarget: RetargetSupport::NotDeclared,
                request: if supported {
                    RequestSupport::OnGrid
                } else {
                    RequestSupport::Unsupported { licensed_route: None }
                },
                boundary: if self.portable.is_some() {
                    Boundary::FreshProcess(ResumeContext {
                        supplied_provider: supplied,
                        ..ResumeContext::default()
                    })
                } else {
                    Boundary::InProcess
                },
            },
        )
    }
    /// Pure planning over declared inputs. No provider method, callback or side effect is invoked.
    #[must_use]
    pub fn plan(&self, request: &ExternalCallbackRequest) -> RecalcPlan {
        self.plan_with_provider(request, false)
    }
    /// Pure planning with a declared provider availability boundary; execution checks the actual provider.
    #[must_use]
    pub fn plan_with_provider(
        &self,
        request: &ExternalCallbackRequest,
        supplied: bool,
    ) -> RecalcPlan {
        let (previous, requested, capabilities) = self.plan_inputs(request, supplied);
        RecalcPlan::plan(&previous, &requested, &capabilities)
    }
    pub(crate) fn absorb_attempts(&mut self, other: &Self) {
        self.attempts.extend(other.attempts.iter().copied());
        self.idempotency_bindings
            .extend(other.idempotency_bindings.iter().map(|(key, request)| (*key, *request)));
        self.epoch = self.epoch.max(other.epoch);
    }
    pub(crate) fn force_recompute(&mut self, branch: Branch) {
        self.previous.remove(Stage::ProviderRequest(branch));
    }
    /// Export original attested output and complete execution identities, never the callback.
    /// # Errors
    /// No successfully issued output or bounded artifact validation failure.
    pub fn export_output(&self) -> Result<Vec<u8>, ExternalCallbackError> {
        self.live
            .as_ref()
            .ok_or(ExternalCallbackError::Request("external_recalc.output_unavailable"))?
            .artifact
            .export()
            .map_err(|error| ExternalCallbackError::Artifact(error.to_string()))
    }
    /// Independently validate portable output; a supplied provider must still replay it before issue.
    /// # Errors
    /// Original artifact/identity refusal or non-replayable stateful/effectful policy.
    pub fn resume(
        bytes: &[u8],
        request: &ExternalCallbackRequest,
    ) -> Result<Self, ExternalCallbackError> {
        request.validate()?;
        if !matches!(
            request.descriptor.policy,
            CallbackPolicy::Deterministic | CallbackPolicy::Seeded
        ) {
            return Err(ExternalCallbackError::Request("external_recalc.replay_unsupported"));
        }
        let artifact = ExternalCallbackArtifact::consume(
            bytes,
            request.digest().as_bytes(),
            request.descriptor.digest().as_bytes(),
        )
        .map_err(|error| ExternalCallbackError::Artifact(error.to_string()))?;
        if artifact.policy != request.descriptor.policy.label() {
            return Err(ExternalCallbackError::Request("external_recalc.replay_mismatch"));
        }
        let mut session = Self::new();
        session.previous = request.identities(0);
        session.portable = Some(Arc::new(artifact));
        Ok(session)
    }
    /// Invoke or reuse the original checked attested mean provider, transactionally.
    /// No automatic retry is performed, and cancellation is cooperative, not hard preemption.
    /// # Errors
    /// Invalid request/provider, unsafe retry, original scientific binding, cancellation or replay failure.
    pub fn execute(
        &mut self,
        request: &ExternalCallbackRequest,
        provider: Option<&mut dyn ExternalMeanProvider>,
        ctx: &ExecutionContext,
    ) -> Result<ExternalCallbackOutcome, ExternalCallbackError> {
        request.validate()?;
        if ctx.cancellation.is_cancelled() {
            return Err(ExternalCallbackError::Request("recalc.cancelled"));
        }
        if provider.as_ref().is_some_and(|provider| provider.descriptor() != &request.descriptor) {
            return Err(ExternalCallbackError::Request("external_recalc.provider_mismatch"));
        }
        let (previous, requested, capabilities) = self.plan_inputs(request, provider.is_some());
        let plan = RecalcPlan::plan(&previous, &requested, &capabilities);
        if !plan.is_executable() {
            return Err(ExternalCallbackError::Refused(Box::new(plan)));
        }
        let stage = Stage::ProviderRequest(request.branch);
        let changed = matches!(plan.status(stage), Some(StageStatus::Recomputed { .. }));
        let mut recorder = ReceiptRecorder::new();
        if !changed {
            let claim = Arc::clone(
                &self
                    .live
                    .as_ref()
                    .ok_or(ExternalCallbackError::Request("external_recalc.output_unavailable"))?
                    .claim,
            );
            let receipt = recorder
                .finish(&plan)
                .map_err(|error| ExternalCallbackError::Artifact(error.to_string()))?;
            return Ok(ExternalCallbackOutcome {
                plan,
                receipt,
                claim,
                previous,
                requested,
                capabilities,
            });
        }
        let provider =
            provider.ok_or(ExternalCallbackError::Request("recalc.unavailable_provider"))?;
        let values =
            request.columns.iter().fold(0_usize, |n, (_, values)| n.saturating_add(values.len()));
        let bytes =
            u64::try_from(values).unwrap_or(u64::MAX).saturating_mul(16).saturating_add(4_000_000);
        if ctx
            .memory
            .hard_limit_bytes
            .into_iter()
            .chain(ctx.memory.soft_limit_bytes)
            .any(|limit| bytes > limit)
        {
            return Err(ExternalCallbackError::Request("recalc.memory_budget_exceeded"));
        }
        if provider.descriptor() != &request.descriptor {
            return Err(ExternalCallbackError::Request("external_recalc.provider_mismatch"));
        }
        let digest = request.digest();
        let key_binding = request.idempotency_key.as_ref().map(|key| {
            StageIdentity::of(
                "callback.idempotency.v1",
                &[request.descriptor.digest().as_bytes(), key.as_bytes()],
            )
        });
        if request.descriptor.policy == CallbackPolicy::SideEffecting
            && key_binding.is_some_and(|key| {
                self.idempotency_bindings.get(&key).is_some_and(|prior| *prior != digest)
            })
        {
            return Err(ExternalCallbackError::Request("external_recalc.idempotency_conflict"));
        }
        if request.descriptor.policy == CallbackPolicy::SideEffecting
            && self.attempts.contains(&digest)
            && request.idempotency_key.is_none()
        {
            return Err(ExternalCallbackError::Request("external_recalc.retry_unsafe"));
        }
        if request.descriptor.policy == CallbackPolicy::SideEffecting
            && self.attempts.len() >= 256
            && !self.attempts.contains(&digest)
        {
            return Err(ExternalCallbackError::Request("external_recalc.attempt_limit"));
        }
        if request.descriptor.policy == CallbackPolicy::SideEffecting {
            self.attempts.insert(digest);
            if let Some(key) = key_binding {
                self.idempotency_bindings.insert(key, digest);
            }
        }
        self.epoch = self.epoch.saturating_add(1);
        INVOCATIONS.set(INVOCATIONS.get().saturating_add(1));
        let report = InvocationAttempt { request_digest: digest, invocations: 1 };
        let callback_context = CallbackContext {
            rng: antecedent_core::RngFactory::from_seed(request.seed),
            cancellation: ctx.cancellation.clone(),
        };
        let mut response = antecedent_core::execution_attempt::run_operation(
            antecedent_core::execution_attempt::Operation::ProviderInvocation,
            || provider.invoke(request, &callback_context),
        )
        .map_err(|failure| ExternalCallbackError::Attempt {
            detail: "external_recalc.callback_failed",
            report: report.clone(),
            message: failure.message.chars().take(1024).collect(),
        })?;
        if ctx.cancellation.is_cancelled() {
            return Err(ExternalCallbackError::Attempt {
                detail: "recalc.cancelled",
                report,
                message: "cooperative cancellation after callback".into(),
            });
        }
        if !response_bounded(&response)
            || response.values.len() != request.program.dose_grid.len()
            || response.values.iter().any(|value| !value.is_finite())
            || response.header.object.identity() != &request.descriptor.identity
            || response.uncertainty != ExternalUncertaintyMeaning::None
        {
            return Err(ExternalCallbackError::Attempt {
                detail: "external_recalc.invalid_output",
                report,
                message: "response shape, finite means, provider identity or uncertainty mismatch"
                    .into(),
            });
        }
        response.header.trust = ExternalTrustState::ExternallyAttested {
            attestor: request.descriptor.identity.provider_id.clone(),
        };
        let bound = antecedent_core::bind_external_result_to_program(
            &request.program,
            &request.claim,
            &request.contract,
            &ExternalResult::Response(response),
        )
        .map_err(|error| ExternalCallbackError::BindingAttempt {
            refusal: Box::new(error),
            report: report.clone(),
        })?;
        let claim = Arc::new(bound.claim().clone());
        let scientific =
            ExternalClaimArtifact::from_bound_claim(&claim, &request.program.contract_id)
                .map_err(|error| ExternalCallbackError::Artifact(error.to_string()))?;
        let artifact = ExternalCallbackArtifact {
            request_digest: *digest.as_bytes(),
            provider_digest: *request.descriptor.digest().as_bytes(),
            policy: request.descriptor.policy.label().into(),
            identity: scientific.metadata().identity.clone(),
            artifact: scientific
                .to_bytes("executed-attested-callback")
                .map_err(|error| ExternalCallbackError::Artifact(error.to_string()))?,
        };
        if let Some(portable) = &self.portable {
            if artifact.identity != portable.identity
                || scientific.values()
                    != ExternalClaimArtifact::from_bytes(&portable.artifact, &portable.identity)
                        .map_err(|error| ExternalCallbackError::Artifact(error.to_string()))?
                        .values()
            {
                return Err(ExternalCallbackError::Attempt {
                    detail: "external_recalc.replay_mismatch",
                    report,
                    message: "provided callback did not reproduce the portable attested output"
                        .into(),
                });
            }
        }
        recorder.record(Counter::ExternalInvocation(request.branch), 1);
        let receipt = recorder
            .finish(&plan)
            .map_err(|error| ExternalCallbackError::Artifact(error.to_string()))?;
        self.previous = requested.clone();
        self.live = Some(LiveOutput { claim: Arc::clone(&claim), artifact: Arc::new(artifact) });
        self.portable = None;
        Ok(ExternalCallbackOutcome { plan, receipt, claim, previous, requested, capabilities })
    }
}
