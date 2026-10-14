//! Arbitrary law, posterior and nested callbacks under the one provider
//! contract: declare, plan, execute, reuse, register evidence, export and
//! resume.
//!
//! A callback is a function this library does not own. This module is the
//! executor of the callback protocol in `antecedent-contract`; it adds nothing
//! to what a callback may claim.
//!
//! 1. **Declare** a [`CallbackDeclaration`]: the operations the callback
//!    answers (mean, sample, distribution function, quantile function and
//!    posterior update are negotiated separately; one never implies another),
//!    its quantity and meaning, provider object and request identity, data,
//!    model, file, remote and environment dependencies, randomness, state and
//!    side-effect contract, support and the trust it claims. The declaration is
//!    an instance of the one interface-contract schema and bridges to the one
//!    provider contract; there is no second registry.
//! 2. **Plan** ([`CallbackSession::plan`]): reads declarations and stored
//!    records only. It takes no provider, so it cannot invoke one.
//! 3. **Execute** ([`CallbackSession::execute`]): negotiates the operation,
//!    checks the provider is the declared object, reuses a stored result when
//!    the replay contract and every identity match, otherwise invokes the
//!    provider under an explicit [`CallbackExecutionPolicy`] with invocation
//!    counts, cooperative cancellation and checked output validation.
//! 4. **Register** callbacks and their evidence in the one evidence ledger
//!    ([`register_callback_source`]) so nested callbacks' overlap is computed.
//! 5. **Export** a result ([`export_callback_result`]) as a portable,
//!    schema-checked artifact and **resume** it ([`CallbackSession::resume`]):
//!    a fresh process needs a supplied compatible provider, or the stage is
//!    recorded `unavailable` and the historical result stays inspectable.
//!
//! # Limits, stated plainly
//!
//! * The wall-clock bound is enforced after the call returns: an in-process
//!   closure cannot be preempted, so a late result is discarded, not
//!   interrupted. A hard limit needs an isolated runtime, which is a separate,
//!   declared execution policy and is not provided here.
//! * Cancellation is cooperative: it is observed before an invocation, passed
//!   to the callback and observed after it.
//! * A result is `diagnostic` unless the callback is deterministic or seeded,
//!   has complete dependencies and is attested or verified; even then it is an
//!   external claim. Metadata never mints native trust, a joint law or an
//!   identification licence, and no interval or posterior guarantee is
//!   carried: those routes refuse with `cell_not_licensed`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::{BTreeMap, BTreeSet};
use std::time::Instant;

pub use antecedent_contract::callback_protocol::{
    CALLBACK_OUTPUT_PORT, CallbackDeclaration, CallbackDependency, CallbackInput, CallbackKind,
    CallbackOperation, CallbackOutput, CallbackPhase, CallbackPlan, CallbackQuery, CallbackRequest,
    CallbackTrust, DependencyDeclaration, DependencyKind, InputSupplier, MAX_CALLBACK_DRAWS,
    MAX_CALLBACK_POINTS, RecomputeReason, ReplayContract, ResultStanding, RetryContract,
    UpdateDeclaration, ValidatedOutput, semantic_stream, standing_label, standing_of,
};
use antecedent_contract::callback_protocol::{
    StoredCallback, callback_source_declaration, plan_reuse, refusal, validate_output,
};
use antecedent_contract::evidence_ledger::{EvidenceLedger, LedgerError, SourceId};
use antecedent_contract::interface_contract::{Connection, InterfaceContract, check_connection};
use antecedent_contract::{
    ExternalCapabilityRequest, ExternalRefusal, ExternalScientificObject, ProviderObjectIdentity,
    ProviderVerificationReceipt,
};
use antecedent_core::{CancellationToken, CausalRng, ExecutionContext};
pub use antecedent_io::callback_result_artifact::{
    CallbackOutcome, CallbackResultArtifact, CallbackResultArtifactError, CallbackResultMeta,
    CallbackResultRecord, MAX_CALLBACK_RESULT_ARTIFACT_BYTES, output_values_digest,
};
use antecedent_validate::predictive_assessment::{
    ForecastLaw, LawSource, ProviderIdentity, ProviderTrustLabel,
};

/// A provider's own report that a call failed. It carries no scientific
/// authority and no value.
#[derive(Clone, Debug)]
pub struct CallbackFailure {
    /// Bounded supplier diagnostic.
    pub message: String,
}

impl CallbackFailure {
    /// A failure with a diagnostic.
    #[must_use]
    pub fn new(message: impl Into<String>) -> Self {
        Self { message: message.into() }
    }
}

/// What one invocation is given. The RNG exists only for a seeded callback and
/// is the stream of its semantic stream identity, fresh for every invocation.
pub struct CallbackContext<'a> {
    /// The stream of a seeded callback; `None` for every other contract.
    pub rng: Option<CausalRng>,
    /// Cooperative cancellation token.
    pub cancellation: &'a CancellationToken,
    /// Which invocation of this request this is, from one.
    pub attempt: u64,
    /// The idempotency key of the request.
    pub idempotency_key: Option<&'a str>,
    /// Accepted outputs of the callbacks this one consumes, by input name.
    pub inputs: &'a BTreeMap<String, ValidatedOutput>,
}

/// A provider implementation supplied in this process. It is never serialized.
pub trait ProviderCallback {
    /// The exact provider object this implementation is. Execution checks it
    /// against the declaration; planning never calls the provider.
    fn identity(&self) -> &ProviderObjectIdentity;

    /// One invocation.
    ///
    /// # Errors
    /// A provider-defined failure; no claim is issued from it.
    fn invoke(
        &mut self,
        query: &CallbackQuery,
        ctx: &mut CallbackContext<'_>,
    ) -> Result<CallbackOutput, CallbackFailure>;

    /// The state identity the provider reports through the protocol (a version
    /// counter, a weights digest). A stateful callback that reports none
    /// cannot be shown current. Asking is a read, not an invocation.
    fn state_identity(&self) -> Option<String> {
        None
    }
}

/// A [`ProviderCallback`] from a closure.
pub struct FnCallback<F> {
    identity: ProviderObjectIdentity,
    call: F,
}

impl<F> FnCallback<F>
where
    F: FnMut(&CallbackQuery, &mut CallbackContext<'_>) -> Result<CallbackOutput, CallbackFailure>,
{
    /// A provider that answers with `call`.
    #[must_use]
    pub fn new(identity: ProviderObjectIdentity, call: F) -> Self {
        Self { identity, call }
    }
}

impl<F> ProviderCallback for FnCallback<F>
where
    F: FnMut(&CallbackQuery, &mut CallbackContext<'_>) -> Result<CallbackOutput, CallbackFailure>,
{
    fn identity(&self) -> &ProviderObjectIdentity {
        &self.identity
    }

    fn invoke(
        &mut self,
        query: &CallbackQuery,
        ctx: &mut CallbackContext<'_>,
    ) -> Result<CallbackOutput, CallbackFailure> {
        (self.call)(query, ctx)
    }
}

/// The explicit execution policy of one call.
#[derive(Clone, Debug)]
pub struct CallbackExecutionPolicy {
    /// Most provider invocations of this call. Zero refuses before any
    /// invocation. Retries are never automatic: a call invokes at most once.
    pub max_invocations: u64,
    /// Most values an accepted output may hold.
    pub max_output_values: usize,
    /// Wall-clock bound in milliseconds. Enforced after the call returns: the
    /// result is discarded; the call is not interrupted.
    pub wall_limit_ms: Option<u64>,
    /// Whether a side-effecting callback may be invoked by this call.
    pub allow_side_effects: bool,
    /// Where in a run the call is made.
    pub phase: CallbackPhase,
}

impl Default for CallbackExecutionPolicy {
    fn default() -> Self {
        Self {
            max_invocations: 1,
            max_output_values: MAX_CALLBACK_DRAWS as usize + 1,
            wall_limit_ms: None,
            allow_side_effects: false,
            phase: CallbackPhase::Execution,
        }
    }
}

/// An accepted answer.
#[derive(Clone, Debug)]
pub struct CallbackExecution {
    /// The accepted output.
    pub output: ValidatedOutput,
    /// Provider invocations this call made (zero when reused).
    pub invocations: u64,
    /// Whether a stored result answered the request.
    pub reused: bool,
    /// Why it was recomputed, when it was.
    pub recomputed_because: Option<RecomputeReason>,
    /// Diagnostic or licensed external; never native.
    pub standing: ResultStanding,
    /// Declaration identity (hex).
    pub declaration_identity: String,
    /// Request identity (hex).
    pub request_identity: String,
    /// The state identity the provider reported after the call.
    pub state_identity: Option<String>,
}

/// A refused or failed execution, with what actually happened.
#[derive(Clone, Debug)]
pub struct CallbackExecutionError {
    /// The typed refusal, with a registered reason code and a remedy.
    pub refusal: ExternalRefusal,
    /// Provider invocations this call made before it ended.
    pub invocations: u64,
    /// Provider invocations of this request so far, this call included.
    pub request_attempts: u64,
}

impl std::fmt::Display for CallbackExecutionError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}: {} ({} invocation(s) this call)",
            antecedent_core::reason_code::PREFIX,
            self.refusal.code,
            self.refusal.detail,
            self.invocations
        )
    }
}

impl std::error::Error for CallbackExecutionError {}

impl CallbackExecutionError {
    fn before(refusal: ExternalRefusal, attempts: u64) -> Self {
        Self { refusal, invocations: 0, request_attempts: attempts }
    }
}

/// A refusal at the connection to a consumer, or of the execution after it.
#[derive(Clone, Debug)]
pub enum CallbackConnectionError {
    /// The declared quantity is not what the consumer needs; the provider was
    /// never invoked.
    Connection(Vec<ExternalRefusal>),
    /// The connection held and the execution was refused or failed.
    Execution(CallbackExecutionError),
}

/// The outcome of resuming a stored result.
#[derive(Clone, Debug)]
pub enum CallbackResume {
    /// The stored result stands: reused or recomputed under a supplied
    /// compatible provider.
    Answered(CallbackExecution),
    /// No compatible provider was supplied: the stage is recorded unavailable
    /// and the historical result stays inspectable.
    Unavailable {
        /// Index of the unavailable record in the session history.
        record: usize,
        /// The refusal that names what would license execution.
        refusal: ExternalRefusal,
    },
}

/// The slot of a request: component, query and seed. Excludes the declaration
/// version and the input digests so a changed dependency or input is seen as a
/// change of an answer, not as a different question.
fn slot_of(declaration: &CallbackDeclaration, request: &CallbackRequest) -> String {
    let mut writer =
        antecedent_core::CanonicalWriter::new("antecedent.callback_protocol.session_slot");
    writer.text(&declaration.component);
    request.query.write_canonical(&mut writer);
    writer.word(request.seed);
    writer.finish_hex()
}

/// Execution state of one process: the append-only history of results and the
/// count of provider invocations per request.
#[derive(Clone, Debug, Default)]
pub struct CallbackSession {
    history: Vec<CallbackResultRecord>,
    latest: BTreeMap<String, usize>,
    attempts: BTreeMap<String, u64>,
    invocations: u64,
    state_probes: u64,
}

impl CallbackSession {
    /// An empty session.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Every record, in the order produced; superseded results stay.
    #[must_use]
    pub fn history(&self) -> &[CallbackResultRecord] {
        &self.history
    }

    /// Provider invocations this session has made, failed attempts included.
    #[must_use]
    pub const fn invocations(&self) -> u64 {
        self.invocations
    }

    /// State-identity probes this session has made. A probe reads the state a
    /// stateful provider reports; it is not an invocation.
    #[must_use]
    pub const fn state_probes(&self) -> u64 {
        self.state_probes
    }

    /// Provider invocations of one request identity (hex).
    #[must_use]
    pub fn request_attempts(&self, request_identity: &str) -> u64 {
        self.attempts.get(request_identity).copied().unwrap_or(0)
    }

    fn stored_for(
        &self,
        declaration: &CallbackDeclaration,
        request: &CallbackRequest,
    ) -> Option<(usize, StoredCallback)> {
        let index = *self.latest.get(&slot_of(declaration, request))?;
        let record = self.history.get(index)?;
        let CallbackOutcome::Computed { state_identity, .. } = &record.outcome else {
            return None;
        };
        Some((
            index,
            StoredCallback {
                declaration_identity: record.declaration.canonical_identity(),
                request_identity: record.request.identity(&record.declaration),
                dependencies: record
                    .declaration
                    .dependencies
                    .complete()
                    .map(<[CallbackDependency]>::to_vec)
                    .unwrap_or_default(),
                state_identity: state_identity.clone(),
            },
        ))
    }

    /// Decide, from declarations and stored records alone, whether the stored
    /// result answers `request`. Takes no provider and never invokes one.
    ///
    /// `current_state` is the state identity of the provider now, when the
    /// caller holds it.
    ///
    /// # Errors
    /// A malformed declaration or query.
    #[allow(
        clippy::result_large_err,
        reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
    )]
    pub fn plan(
        &self,
        declaration: &CallbackDeclaration,
        request: &CallbackRequest,
        current_state: Option<&str>,
    ) -> Result<CallbackPlan, ExternalRefusal> {
        declaration.validate()?;
        request.query.validate()?;
        let stored = self.stored_for(declaration, request).map(|(_, stored)| stored);
        Ok(plan_reuse(declaration, request, stored.as_ref(), current_state))
    }

    /// Records whose declaration consumes the callback with `identity` (hex),
    /// directly or through other callbacks: the results invalidated when that
    /// callback's closure state, weights, files or remote version change.
    #[must_use]
    pub fn dependents_of(&self, identity: &str) -> Vec<usize> {
        let mut affected: BTreeSet<usize> = BTreeSet::new();
        let mut frontier: Vec<String> = vec![identity.to_owned()];
        while let Some(current) = frontier.pop() {
            for (index, record) in self.history.iter().enumerate() {
                let consumes = record.declaration.inputs.iter().any(|input| {
                    matches!(&input.supplier, InputSupplier::Callback { identity: inner, .. } if *inner == current)
                });
                if consumes && affected.insert(index) {
                    frontier.push(record.declaration.canonical_identity().to_hex());
                }
            }
        }
        affected.into_iter().collect()
    }

    #[allow(
        clippy::result_large_err,
        reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
    )]
    fn check_inputs(
        declaration: &CallbackDeclaration,
        request: &CallbackRequest,
        inputs: &BTreeMap<String, ValidatedOutput>,
    ) -> Result<(), ExternalRefusal> {
        for input in &declaration.inputs {
            let Some(value) = inputs.get(&input.name) else {
                return Err(refusal::request_mismatch(&input.name));
            };
            if value.output().operation() != input.operation {
                return Err(refusal::request_mismatch(&input.name));
            }
            let digest = output_values_digest(&value.output().values());
            if request.input_digests.get(&input.name) != Some(&digest) {
                return Err(refusal::request_mismatch(&input.name));
            }
        }
        if inputs.len() != declaration.inputs.len()
            || request.input_digests.len() != declaration.inputs.len()
        {
            return Err(refusal::request_mismatch("inputs"));
        }
        Ok(())
    }

    /// Answer `request` from the stored result or by invoking `provider`.
    ///
    /// The order of checks, each before any invocation: the declaration, the
    /// phase, negotiation of exactly the requested operation through the one
    /// provider contract, the provider's identity, the inputs, the reuse plan,
    /// the side-effect policy, the invocation bound and cancellation. After an
    /// invocation: cancellation, the wall-clock bound and the checked output.
    ///
    /// # Errors
    /// A [`CallbackExecutionError`] carrying the typed refusal and the actual
    /// invocation counts. A failed call is not retried.
    #[allow(
        clippy::too_many_lines,
        reason = "one linear sequence of checks, each before the invocation it guards"
    )]
    #[allow(
        clippy::result_large_err,
        reason = "a refusal is the cold path of a once-per-call check; boxing the error would change the public error type"
    )]
    pub fn execute(
        &mut self,
        provider: &mut dyn ProviderCallback,
        declaration: &CallbackDeclaration,
        request: &CallbackRequest,
        inputs: &BTreeMap<String, ValidatedOutput>,
        policy: &CallbackExecutionPolicy,
        ctx: &ExecutionContext,
    ) -> Result<CallbackExecution, CallbackExecutionError> {
        let request_hex = request.identity(declaration).to_hex();
        let attempts = self.request_attempts(&request_hex);
        let fail = |refusal: ExternalRefusal| CallbackExecutionError::before(refusal, attempts);
        declaration.validate().map_err(fail)?;
        request.query.validate().map_err(fail)?;
        match policy.phase {
            CallbackPhase::Planning => {
                return Err(fail(refusal::invocation_outside_execution(policy.phase.as_str())));
            }
            CallbackPhase::Verification => {
                if matches!(
                    declaration.replay,
                    ReplayContract::SideEffecting { .. } | ReplayContract::Stateful { .. }
                ) {
                    return Err(fail(refusal::invocation_outside_execution(policy.phase.as_str())));
                }
            }
            CallbackPhase::Execution => {}
        }
        let operation = request.query.operation();
        let object = if operation == CallbackOperation::PosteriorUpdate {
            declaration.update_object()
        } else {
            Some(declaration.provider_object())
        };
        let negotiated = object
            .ok_or_else(|| {
                refusal::operation_not_declared(operation.capability(), "posterior_update")
            })
            .and_then(|object: ExternalScientificObject| {
                if !declaration.operations.contains(&operation) {
                    return Err(refusal::operation_not_declared(
                        operation.capability(),
                        &declaration
                            .operations
                            .iter()
                            .copied()
                            .map(CallbackOperation::as_str)
                            .collect::<Vec<_>>()
                            .join(","),
                    ));
                }
                object
                    .negotiate(&ExternalCapabilityRequest {
                        expected_identity: declaration.identity.clone(),
                        operation: operation.capability(),
                    })
                    .map_err(|_| {
                        refusal::operation_not_declared(operation.capability(), "contract")
                    })
            })
            .map_err(fail)?;
        if provider.identity() != negotiated.identity() {
            let expected = format!(
                "{}/{}/{}",
                declaration.identity.object_id,
                declaration.identity.version_id,
                declaration.identity.snapshot_id
            );
            let supplied = format!(
                "{}/{}/{}",
                provider.identity().object_id,
                provider.identity().version_id,
                provider.identity().snapshot_id
            );
            return Err(fail(refusal::provider_identity_mismatch(&expected, &supplied)));
        }
        Self::check_inputs(declaration, request, inputs).map_err(fail)?;
        let exposes_state =
            matches!(declaration.replay, ReplayContract::Stateful { exposes_state_identity: true });
        let current_state = if exposes_state {
            self.state_probes += 1;
            provider.state_identity()
        } else {
            None
        };
        let stored = self.stored_for(declaration, request);
        let plan = plan_reuse(
            declaration,
            request,
            stored.as_ref().map(|(_, stored)| stored),
            current_state.as_deref(),
        );
        let reason = match plan {
            CallbackPlan::Reuse => {
                if let Some((index, _)) = stored {
                    if let CallbackOutcome::Computed { output, state_identity, .. } =
                        &self.history[index].outcome
                    {
                        // A cached output was checked against the policy of
                        // its original execution. The current caller may
                        // impose a tighter bound, which reuse must respect.
                        if output.output().value_count() > policy.max_output_values {
                            return Err(fail(refusal::output_too_large(policy.max_output_values)));
                        }
                        return Ok(CallbackExecution {
                            output: output.clone(),
                            invocations: 0,
                            reused: true,
                            recomputed_because: None,
                            standing: standing_label(declaration, false),
                            declaration_identity: declaration.canonical_identity().to_hex(),
                            request_identity: request_hex,
                            state_identity: state_identity.clone(),
                        });
                    }
                }
                RecomputeReason::NoPriorResult
            }
            CallbackPlan::Refuse(refused) => return Err(fail(refused)),
            CallbackPlan::Recompute(reason) => reason,
        };
        Self::guard_effects(declaration, request, policy, attempts).map_err(fail)?;
        if policy.max_invocations == 0 {
            return Err(fail(refusal::invocation_limit(0)));
        }
        if attempts == u64::MAX {
            return Err(fail(refusal::invocation_limit(u64::MAX)));
        }
        if ctx.cancellation.is_cancelled() {
            return Err(fail(refusal::cancelled("before_invocation")));
        }
        self.invoke_and_accept(
            provider,
            declaration,
            request,
            inputs,
            policy,
            ctx,
            request_hex,
            reason,
        )
    }

    #[allow(
        clippy::result_large_err,
        reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
    )]
    fn guard_effects(
        declaration: &CallbackDeclaration,
        request: &CallbackRequest,
        policy: &CallbackExecutionPolicy,
        attempts: u64,
    ) -> Result<(), ExternalRefusal> {
        let ReplayContract::SideEffecting { retry } = &declaration.replay else {
            return Ok(());
        };
        if !policy.allow_side_effects {
            return Err(refusal::side_effects_not_allowed());
        }
        if attempts == 0 {
            return Ok(());
        }
        // An earlier attempt of this exact request exists: only a declared
        // idempotent contract, with the same key, within its bound, may repeat it.
        match retry {
            RetryContract::IdempotentKey { max_attempts }
                if request.idempotency_key.is_some() && attempts < u64::from(*max_attempts) =>
            {
                Ok(())
            }
            _ => Err(refusal::retry_outside_policy(attempts)),
        }
    }

    #[allow(
        clippy::too_many_arguments,
        reason = "the second half of execute, split to keep each function one sequence of checks"
    )]
    #[allow(
        clippy::result_large_err,
        reason = "a refusal is the cold path of a once-per-call check; boxing the error would change the public error type"
    )]
    fn invoke_and_accept(
        &mut self,
        provider: &mut dyn ProviderCallback,
        declaration: &CallbackDeclaration,
        request: &CallbackRequest,
        inputs: &BTreeMap<String, ValidatedOutput>,
        policy: &CallbackExecutionPolicy,
        ctx: &ExecutionContext,
        request_hex: String,
        reason: RecomputeReason,
    ) -> Result<CallbackExecution, CallbackExecutionError> {
        let rng = match &declaration.replay {
            ReplayContract::SeededStochastic { stream_id } => {
                Some(semantic_stream(request.seed, stream_id))
            }
            _ => None,
        };
        *self.attempts.entry(request_hex.clone()).or_insert(0) += 1;
        self.invocations += 1;
        let attempt = self.request_attempts(&request_hex);
        let mut callback_ctx = CallbackContext {
            rng,
            cancellation: &ctx.cancellation,
            attempt,
            idempotency_key: request.idempotency_key.as_deref(),
            inputs,
        };
        let started = Instant::now();
        let outcome = provider.invoke(&request.query, &mut callback_ctx);
        let elapsed = started.elapsed();
        let error = |refusal: ExternalRefusal| CallbackExecutionError {
            refusal,
            invocations: 1,
            request_attempts: attempt,
        };
        let output =
            outcome.map_err(|failure| error(refusal::provider_failed(&failure.message)))?;
        if ctx.cancellation.is_cancelled() {
            return Err(error(refusal::cancelled("after_invocation")));
        }
        if let Some(limit) = policy.wall_limit_ms {
            if elapsed.as_millis() > u128::from(limit) {
                return Err(error(refusal::wall_limit_exceeded(limit, elapsed.as_millis())));
            }
        }
        let accepted =
            validate_output(&request.query, output, &declaration.support, policy.max_output_values)
                .map_err(error)?;
        let state_identity = if matches!(
            declaration.replay,
            ReplayContract::Stateful { exposes_state_identity: true }
        ) {
            self.state_probes += 1;
            provider.state_identity()
        } else {
            None
        };
        let slot = slot_of(declaration, request);
        self.history.push(CallbackResultRecord {
            declaration: declaration.clone(),
            request: request.clone(),
            outcome: CallbackOutcome::Computed {
                output: accepted.clone(),
                invocations: attempt,
                state_identity: state_identity.clone(),
            },
        });
        self.latest.insert(slot, self.history.len() - 1);
        Ok(CallbackExecution {
            output: accepted,
            invocations: 1,
            reused: false,
            recomputed_because: Some(reason),
            standing: standing_label(declaration, false),
            declaration_identity: declaration.canonical_identity().to_hex(),
            request_identity: request_hex,
            state_identity,
        })
    }

    /// Check the connection to a consumer through the interface contract, then
    /// execute. A mismatched population, regime, support or operation is
    /// refused before the provider is invoked.
    ///
    /// # Errors
    /// [`CallbackConnectionError::Connection`] when the declared quantity is
    /// not the consumer's, or [`CallbackConnectionError::Execution`].
    #[allow(
        clippy::too_many_arguments,
        reason = "the consumer contract and port are two further coordinates of one connection"
    )]
    #[allow(
        clippy::result_large_err,
        reason = "a refusal is the cold path of a once-per-call check; boxing the error would change the public error type"
    )]
    pub fn execute_for_consumer(
        &mut self,
        provider: &mut dyn ProviderCallback,
        declaration: &CallbackDeclaration,
        consumer: &InterfaceContract,
        consumed_port: &str,
        request: &CallbackRequest,
        inputs: &BTreeMap<String, ValidatedOutput>,
        policy: &CallbackExecutionPolicy,
        ctx: &ExecutionContext,
    ) -> Result<(Connection, CallbackExecution), CallbackConnectionError> {
        let connection = check_callback_connection(declaration, consumer, consumed_port)
            .map_err(CallbackConnectionError::Connection)?;
        let execution = self
            .execute(provider, declaration, request, inputs, policy, ctx)
            .map_err(CallbackConnectionError::Execution)?;
        Ok((connection, execution))
    }

    #[allow(
        clippy::result_large_err,
        reason = "a refusal is the cold path of a once-per-call check; boxing the error would change the public error type"
    )]
    /// Resume a stored result from its portable artifact in this process.
    ///
    /// The artifact is consumed first (every identity recomputed, every bound
    /// applied). The historical result is kept in the session history either
    /// way. With a compatible `provider` the request is answered by reuse or
    /// recomputation under the replay contract; with none the stage is recorded
    /// as unavailable. A provider that is not the declared object is refused:
    /// declare the replacement and execute it as a new callback.
    ///
    /// # Errors
    /// An artifact the consumer refuses, or the refusal of the execution.
    pub fn resume(
        &mut self,
        bytes: &[u8],
        expected_identity: Option<&str>,
        provider: Option<&mut dyn ProviderCallback>,
        inputs: &BTreeMap<String, ValidatedOutput>,
        policy: &CallbackExecutionPolicy,
        ctx: &ExecutionContext,
    ) -> Result<CallbackResume, CallbackExecutionError> {
        let artifact =
            CallbackResultArtifact::from_bytes(bytes, expected_identity).map_err(|error| {
                let (code, detail, message) = error.refusal().unwrap_or_else(|| {
                    (
                        antecedent_core::reason_code!("invalid_argument"),
                        "callback_protocol.artifact_invalid".to_owned(),
                        error.to_string(),
                    )
                });
                CallbackExecutionError::before(
                    ExternalRefusal::new(code, "consume", detail).with_offending(message),
                    0,
                )
            })?;
        let record = artifact.record().clone();
        // The historical result is imported as it was; it is inspectable and a
        // later identical request may reuse it under the replay contract.
        let slot = slot_of(&record.declaration, &record.request);
        let request_hex = record.request.identity(&record.declaration).to_hex();
        if let CallbackOutcome::Computed { invocations, .. } = &record.outcome {
            self.attempts
                .entry(request_hex)
                .and_modify(|attempts| *attempts = (*attempts).max(*invocations))
                .or_insert(*invocations);
        }
        let imported_computed = matches!(record.outcome, CallbackOutcome::Computed { .. });
        self.history.push(record.clone());
        if imported_computed {
            self.latest.insert(slot, self.history.len() - 1);
        }
        let Some(provider) = provider else {
            let refusal = refusal::provider_unavailable(&format!(
                "{}/{}/{}/{}",
                record.declaration.identity.provider_id,
                record.declaration.identity.object_id,
                record.declaration.identity.version_id,
                record.declaration.identity.snapshot_id
            ));
            self.history.push(CallbackResultRecord {
                declaration: record.declaration.clone(),
                request: record.request.clone(),
                outcome: CallbackOutcome::Unavailable { reason: refusal.detail.clone() },
            });
            return Ok(CallbackResume::Unavailable { record: self.history.len() - 1, refusal });
        };
        let execution =
            self.execute(provider, &record.declaration, &record.request, inputs, policy, ctx)?;
        Ok(CallbackResume::Answered(execution))
    }
}

/// Check the connection of a callback's output to a consumer's port through
/// the one interface contract.
///
/// # Errors
/// Every defect, each a registered reason code with an
/// `interface_contract.*` detail and the declaration that would license it.
pub fn check_callback_connection(
    declaration: &CallbackDeclaration,
    consumer: &InterfaceContract,
    consumed_port: &str,
) -> Result<Connection, Vec<ExternalRefusal>> {
    check_connection(
        &declaration.interface_contract(),
        CALLBACK_OUTPUT_PORT,
        consumer,
        consumed_port,
    )
}

/// Check the connection of a producer's claim (a transported claim, an
/// observed summary) to one input of a callback through the one interface
/// contract.
///
/// # Errors
/// As [`check_callback_connection`].
pub fn check_claim_to_callback(
    producer: &InterfaceContract,
    produced_port: &str,
    declaration: &CallbackDeclaration,
    input: &str,
) -> Result<Connection, Vec<ExternalRefusal>> {
    check_connection(producer, produced_port, &declaration.interface_contract(), input)
}

/// Refusal of an interval from a callback law.
#[must_use]
pub fn refuse_interval() -> ExternalRefusal {
    refusal::interval_not_licensed()
}

/// Refusal of a posterior or probability guarantee from a callback.
#[must_use]
pub fn refuse_posterior_guarantee() -> ExternalRefusal {
    refusal::posterior_guarantee_not_licensed()
}

/// Refusal of native trust conferred by callback metadata.
#[must_use]
pub fn refuse_native_trust() -> ExternalRefusal {
    refusal::native_trust_not_conferred()
}

/// Refusal of an identification licence conferred by a callback.
#[must_use]
pub fn refuse_identification_licence() -> ExternalRefusal {
    refusal::identification_not_conferred()
}

/// Refusal of a joint law conferred by independently produced callback draws.
#[must_use]
pub fn refuse_joint_law() -> ExternalRefusal {
    refusal::joint_law_not_conferred()
}

/// A failure to register a callback in the evidence ledger.
#[derive(Debug)]
pub enum CallbackRegistrationError {
    /// The declaration or a nested input is not registrable.
    Declaration(ExternalRefusal),
    /// The ledger refused.
    Ledger(LedgerError),
}

#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would change the public error type"
)]
/// Register a callback and its evidence in the one evidence ledger, so a
/// nested callback's overlap with its inner callbacks and with other callbacks
/// is computed rather than asserted.
///
/// `inner_sources` maps the declaration identity (hex) of each inner callback
/// to the ledger source it was registered as.
///
/// # Errors
/// An invalid declaration, an inner callback that is not registered, or a
/// ledger refusal.
pub fn register_callback_source(
    ledger: &mut EvidenceLedger,
    declaration: &CallbackDeclaration,
    inner_sources: &BTreeMap<String, SourceId>,
) -> Result<SourceId, CallbackRegistrationError> {
    declaration.validate().map_err(CallbackRegistrationError::Declaration)?;
    let source = callback_source_declaration(declaration, inner_sources)
        .map_err(CallbackRegistrationError::Declaration)?;
    ledger.add_source(source).map_err(CallbackRegistrationError::Ledger)
}

#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would change the public error type"
)]
/// Supersede a registered callback with its replacement (a new provider
/// version, a changed dependency): the ledger revalidates results that rest on
/// the old source.
///
/// # Errors
/// As [`register_callback_source`].
pub fn supersede_callback_source(
    ledger: &mut EvidenceLedger,
    old: SourceId,
    replacement: &CallbackDeclaration,
    inner_sources: &BTreeMap<String, SourceId>,
) -> Result<SourceId, CallbackRegistrationError> {
    replacement.validate().map_err(CallbackRegistrationError::Declaration)?;
    let source = callback_source_declaration(replacement, inner_sources)
        .map_err(CallbackRegistrationError::Declaration)?;
    let applied =
        ledger.supersede_source(old, source).map_err(CallbackRegistrationError::Ledger)?;
    applied.effect.created_source.ok_or_else(|| {
        CallbackRegistrationError::Declaration(refusal::request_mismatch("supersede"))
    })
}

/// Whether `receipt` is the verification the declaration names for the exact
/// provider object; only then does a verified standing hold.
#[must_use]
pub fn verification_holds(
    declaration: &CallbackDeclaration,
    receipt: &ProviderVerificationReceipt,
) -> bool {
    declaration.confirms_verification(receipt)
}

/// Serialize a result record as a portable `callback_result_v1` artifact. The
/// artifact is a declaration and an output, not a promise a closure runs
/// elsewhere.
///
/// # Errors
/// A record the protocol refuses, or an encoding failure.
pub fn export_callback_result(
    record: &CallbackResultRecord,
    artifact_id: &str,
) -> Result<Vec<u8>, CallbackResultArtifactError> {
    CallbackResultArtifact::to_bytes(record, artifact_id)
}

/// Consume a callback result artifact: rebuild and validate the declaration,
/// request and output and recompute every identity against the retained one.
/// Nothing is executed.
///
/// # Errors
/// As [`CallbackResultArtifact::from_bytes`].
pub fn consume_callback_result(
    bytes: &[u8],
    expected_identity: Option<&str>,
) -> Result<CallbackResultArtifact, CallbackResultArtifactError> {
    CallbackResultArtifact::from_bytes(bytes, expected_identity)
}

/// Read a result's stored metadata without replaying it.
///
/// # Errors
/// Corruption, another version or a malformed container.
pub fn inspect_callback_result(
    bytes: &[u8],
) -> Result<CallbackResultMeta, CallbackResultArtifactError> {
    CallbackResultArtifact::inspect(bytes)
}

/// The predictive-assessment source of a callback law.
///
/// A callback is consumed by the assessment as a provider-supplied law, never
/// as an anonymous closure: the declaration supplies the provider object
/// identity and the trust label. A diagnostic callback (opaque dependencies,
/// stateful, side-effecting, or unverified) is refused, because its scores
/// could not be reproduced by anyone holding the declaration.
///
/// # Errors
/// A malformed declaration, or a diagnostic standing.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
pub fn predictive_law_source(
    declaration: &CallbackDeclaration,
    verification_confirmed: bool,
) -> Result<LawSource, ExternalRefusal> {
    declaration.validate()?;
    standing_of(declaration, verification_confirmed)?;
    let trust = match (&declaration.trust, verification_confirmed) {
        (CallbackTrust::ExactRequestVerified { .. }, true) => {
            ProviderTrustLabel::ExactRequestVerified
        }
        _ => ProviderTrustLabel::ExternallyAttested,
    };
    Ok(LawSource::Provider {
        identity: ProviderIdentity {
            provider_id: declaration.identity.provider_id.clone(),
            object_id: declaration.identity.object_id.clone(),
            version_id: declaration.identity.version_id.clone(),
            snapshot_id: declaration.identity.snapshot_id.clone(),
            request_id: declaration.identity.request_id.clone(),
        },
        trust,
    })
}

/// The empirical-sample forecast law of an accepted sample output. The draws
/// are scored as the empirical distribution they are; nothing is inferred
/// about the callback's law beyond them.
///
/// # Errors
/// An output that is not a sample.
#[allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]
pub fn forecast_law_from_sample(output: &ValidatedOutput) -> Result<ForecastLaw, ExternalRefusal> {
    match output.output() {
        CallbackOutput::Sample { draws } => {
            Ok(ForecastLaw::EmpiricalSample { draws: draws.clone() })
        }
        _ => Err(refusal::request_mismatch("sample")),
    }
}
