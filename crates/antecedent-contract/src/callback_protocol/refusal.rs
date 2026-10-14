//! Typed refusals of the callback protocol: a registered reason code, a
//! `callback_protocol.*` detail and the declaration or evidence that would
//! license the request.
//!
//! Every detail literal is written here, beside the reason code it is emitted
//! under, as one named constructor per refusal. The rest of the module tree
//! calls the constructors and never spells a detail.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use crate::{ExternalCapability, ExternalRefusal};

/// A refusal of a callback declaration, request, plan or output.
pub type CallbackRefusal = ExternalRefusal;

const STAGE_DECLARE: &str = "declare";
const STAGE_NEGOTIATE: &str = "negotiate";
const STAGE_PLAN: &str = "plan";
const STAGE_EXECUTE: &str = "execute";
const STAGE_VALIDATE: &str = "validate";
const STAGE_CLOSED: &str = "uncertainty";

fn invalid(detail: &str, remedy: &'static str) -> CallbackRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("callback_protocol_invalid"),
        STAGE_DECLARE,
        detail,
    )
    .with_remedy(remedy)
}

fn reuse(detail: &str, remedy: &'static str) -> CallbackRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("callback_reuse_ineligible"),
        STAGE_PLAN,
        detail,
    )
    .with_remedy(remedy)
}

fn effect(detail: &str, remedy: &'static str) -> CallbackRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("callback_side_effect_forbidden"),
        STAGE_EXECUTE,
        detail,
    )
    .with_remedy(remedy)
}

fn output(detail: &str, remedy: &'static str) -> CallbackRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("callback_output_invalid"),
        STAGE_VALIDATE,
        detail,
    )
    .with_remedy(remedy)
}

// -- malformed declarations (callback_protocol_invalid) -----------------------------------------

/// A required identity or name is blank or too long.
#[must_use]
pub fn blank_field(field: &str) -> CallbackRefusal {
    invalid(
        "callback_protocol.blank_field",
        "name every provider, object, version, snapshot, request, dependency and input identity, in at most the declared number of bytes",
    )
    .with_offending(field)
}

/// A declaration list is longer than the bound.
#[must_use]
pub fn too_many_items(what: &str) -> CallbackRefusal {
    invalid(
        "callback_protocol.too_many_items",
        "declare at most the bounded number of dependencies, inputs and operations",
    )
    .with_offending(what)
}

/// The output quantity is not a well-formed scientific quantity.
#[must_use]
pub fn invalid_quantity(what: &str) -> CallbackRefusal {
    invalid(
        "callback_protocol.invalid_quantity",
        "name the variable, units, population, regime, functional and transform of the quantity, with canonical conditioning",
    )
    .with_offending(what)
}

/// The support domain is not well formed.
#[must_use]
pub fn invalid_support(what: &str) -> CallbackRefusal {
    invalid(
        "callback_protocol.invalid_support",
        "declare a support domain with finite ordered bounds or sorted, distinct, non-blank levels",
    )
    .with_offending(what)
}

/// No operation is declared.
#[must_use]
pub fn no_operations() -> CallbackRefusal {
    invalid(
        "callback_protocol.no_operations",
        "declare at least one operation (mean, sample, cdf, quantile or posterior update) the callback answers; none implies another",
    )
}

/// The declared operation is not one this kind of callback answers.
#[must_use]
pub fn operation_not_for_kind(operation: &str, kind: &str) -> CallbackRefusal {
    invalid(
        "callback_protocol.operation_not_for_kind",
        "declare only operations the callback kind answers: a posterior callback answers mean, sample and posterior update; a law or nested callback answers mean, sample, cdf and quantile",
    )
    .with_offending(operation)
    .with_supplied(kind)
}

/// The distribution meaning does not belong to the callback kind.
#[must_use]
pub fn meaning_not_for_kind(meaning: &str, kind: &str) -> CallbackRefusal {
    invalid(
        "callback_protocol.meaning_not_for_kind",
        "declare a posterior meaning for a posterior callback and a non-posterior meaning for a law callback",
    )
    .with_offending(meaning)
    .with_supplied(kind)
}

/// The posterior-update declaration and the update operation disagree.
#[must_use]
pub fn update_declaration_mismatch() -> CallbackRefusal {
    invalid(
        "callback_protocol.update_declaration_mismatch",
        "declare the candidate, prior and possible observation exactly when the callback answers a posterior update",
    )
}

/// Two inputs, dependencies or operations share a name.
#[must_use]
pub fn duplicate_name(what: &str) -> CallbackRefusal {
    invalid("callback_protocol.duplicate_name", "give every input and dependency its own name")
        .with_offending(what)
}

/// The replay contract is incomplete: a seeded callback without a semantic
/// stream identity, an idempotent retry without an attempt bound.
#[must_use]
pub fn replay_contract_incomplete(detail: &str) -> CallbackRefusal {
    invalid(
        "callback_protocol.replay_contract_incomplete",
        "declare the semantic stream identity of a seeded callback and a positive attempt bound for an idempotent retry",
    )
    .with_offending(detail)
}

/// A nested callback names no callback input, or a non-nested callback does.
#[must_use]
pub fn nested_inputs_mismatch() -> CallbackRefusal {
    invalid(
        "callback_protocol.nested_inputs_mismatch",
        "a nested callback declares at least one input supplied by another callback, and only a nested callback does",
    )
}

/// A trust level that metadata cannot confer.
#[must_use]
pub fn native_trust_claimed() -> CallbackRefusal {
    invalid(
        "callback_protocol.native_trust_claimed",
        "a callback is external: declare it unverified, externally attested or exact-request verified; native licence is never conferred by metadata",
    )
}

/// A trust declaration with a blank attestor or receipt digest.
#[must_use]
pub fn trust_incomplete() -> CallbackRefusal {
    invalid(
        "callback_protocol.trust_incomplete",
        "name the attestor of an attested callback and the verification receipt digest of a verified one",
    )
}

// -- negotiation -------------------------------------------------------------------------------

/// The requested operation is not declared; no other operation substitutes.
#[must_use]
pub fn operation_not_declared(operation: ExternalCapability, declared: &str) -> CallbackRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("external_capability_missing"),
        STAGE_NEGOTIATE,
        "callback_protocol.operation_not_declared",
    )
    .with_capability(operation)
    .with_supplied(declared)
    .with_remedy(
        "declare the exact operation on the callback; a mean does not supply a law, draws do not supply a distribution function or quantile, and none supplies a posterior update",
    )
}

/// The provider presented for execution is not the declared provider object.
#[must_use]
pub fn provider_identity_mismatch(expected: &str, supplied: &str) -> CallbackRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("external_binding_mismatch"),
        STAGE_NEGOTIATE,
        "callback_protocol.provider_identity_mismatch",
    )
    .with_expected(expected)
    .with_supplied(supplied)
    .with_remedy(
        "declare the replacement provider (new object, version, snapshot and request identity) and verify it; a replaced provider is a different callback",
    )
}

/// The request names an operation or input that disagrees with the protocol.
#[must_use]
pub fn request_mismatch(detail: &str) -> CallbackRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("external_binding_mismatch"),
        STAGE_NEGOTIATE,
        "callback_protocol.request_mismatch",
    )
    .with_offending(detail)
    .with_remedy("supply every declared input, once, and a query of a declared operation")
}

/// A nested input's callback is not registered in the evidence ledger, so its
/// overlap with the other inputs cannot be computed.
#[must_use]
pub fn nested_input_unregistered(input: &str) -> CallbackRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("sampling_dependence_unknown"),
        STAGE_NEGOTIATE,
        "callback_protocol.nested_input_unregistered",
    )
    .with_offending(input)
    .with_remedy(
        "register the inner callback and its evidence in the session's evidence ledger before executing a callback that combines it with another callback's output",
    )
}

/// Two nested inputs rest on observations in common, so combining them counts
/// the same evidence twice.
#[must_use]
pub fn nested_inputs_share_evidence(first: &str, second: &str, shared: u64) -> CallbackRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("evidence_double_counted"),
        STAGE_NEGOTIATE,
        "callback_protocol.nested_inputs_share_evidence",
    )
    .with_offending(format!("{first}, {second}"))
    .with_supplied(format!("{shared} shared observation(s)"))
    .with_remedy(
        "feed the callback inputs that rest on disjoint evidence, or use one input that already carries the dependence between the two",
    )
}

/// The ledger cannot decide whether two nested inputs share observations.
#[must_use]
pub fn nested_input_overlap_unknown(first: &str, second: &str) -> CallbackRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("sampling_dependence_unknown"),
        STAGE_NEGOTIATE,
        "callback_protocol.nested_input_overlap_unknown",
    )
    .with_offending(format!("{first}, {second}"))
    .with_remedy(
        "declare the observations behind each inner callback's evidence so the ledger can compute the overlap; unknown overlap is never independence",
    )
}

// -- reuse and replay (callback_reuse_ineligible) ------------------------------------------------

/// A dependency is opaque or undeclared, so no stable identity exists.
#[must_use]
pub fn opaque_dependencies() -> CallbackRefusal {
    reuse(
        "callback_protocol.opaque_dependencies",
        "declare every data, model, file, remote and environment dependency with a version or digest; an opaque or undeclared dependency makes a prior result ineligible for reuse",
    )
}

/// A stateful callback exposes no state identity.
#[must_use]
pub fn hidden_state_unobservable() -> CallbackRefusal {
    reuse(
        "callback_protocol.hidden_state_unobservable",
        "expose a state identity through the protocol (a version counter or weights digest) so a changed state is detected, or recompute on every use",
    )
}

/// A side-effecting callback has no idempotency contract, so a stored result
/// is not reusable and a new invocation is not allowed.
#[must_use]
pub fn no_idempotency_contract() -> CallbackRefusal {
    reuse(
        "callback_protocol.no_idempotency_contract",
        "declare an idempotent retry contract and supply the idempotency key; without one the effects of a repeated call are unknown and the callback is not run again",
    )
}

/// A diagnostic result cannot back a licensed claim.
#[must_use]
pub fn diagnostic_only(reason: &str) -> CallbackRefusal {
    reuse(
        "callback_protocol.diagnostic_only",
        "declare complete dependencies and a deterministic or seeded replay contract and attest or verify the provider; a stateful or side-effecting result stays diagnostic",
    )
    .with_offending(reason)
}

// -- side effects (callback_side_effect_forbidden) -----------------------------------------------

/// A side-effecting callback must not be invoked while planning or verifying.
#[must_use]
pub fn invocation_outside_execution(phase: &str) -> CallbackRefusal {
    effect(
        "callback_protocol.invocation_outside_execution",
        "planning and verification read declarations and stored records only; run the callback through an explicit execution",
    )
    .with_offending(phase)
}

/// The execution policy does not allow side effects.
#[must_use]
pub fn side_effects_not_allowed() -> CallbackRefusal {
    effect(
        "callback_protocol.side_effects_not_allowed",
        "opt in to side-effecting execution in the execution policy for this one request",
    )
}

/// A retry is outside the declared policy.
#[must_use]
pub fn retry_outside_policy(attempts: u64) -> CallbackRefusal {
    effect(
        "callback_protocol.retry_outside_policy",
        "a side-effecting callback is retried only under a declared idempotent contract with the same key and within its attempt bound",
    )
    .with_supplied(attempts.to_string())
}

// -- output (callback_output_invalid) ------------------------------------------------------------

/// The output is not the shape of the query.
#[must_use]
pub fn output_shape(detail: &str) -> CallbackRefusal {
    output(
        "callback_protocol.output_shape",
        "return exactly the values the query asks for, in order",
    )
    .with_offending(detail)
}

/// A value is not finite.
#[must_use]
pub fn output_not_finite(detail: &str) -> CallbackRefusal {
    output("callback_protocol.output_not_finite", "return finite values only")
        .with_offending(detail)
}

/// A value lies outside the declared support.
#[must_use]
pub fn output_outside_support(detail: &str, support: &str) -> CallbackRefusal {
    output(
        "callback_protocol.output_outside_support",
        "return values inside the declared support, or declare the support the law actually has",
    )
    .with_offending(detail)
    .with_expected(support)
}

/// A distribution function or quantile function is not a law.
#[must_use]
pub fn output_not_a_law(detail: &str) -> CallbackRefusal {
    output(
        "callback_protocol.output_not_a_law",
        "a distribution function has values in [0, 1] that do not decrease in the threshold; a quantile function does not decrease in the level",
    )
    .with_offending(detail)
}

/// The query itself is malformed.
#[must_use]
pub fn query_invalid(detail: &str) -> CallbackRefusal {
    output(
        "callback_protocol.query_invalid",
        "ask for a bounded number of draws, finite increasing thresholds and quantile levels strictly inside (0, 1)",
    )
    .with_offending(detail)
}

/// More output than the policy bound.
#[must_use]
pub fn output_too_large(limit: usize) -> CallbackRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("callback_limit_exceeded"),
        STAGE_VALIDATE,
        "callback_protocol.output_too_large",
    )
    .with_supplied(limit.to_string())
    .with_remedy("raise the declared output bound or ask for fewer values")
}

// -- resources -----------------------------------------------------------------------------------

/// The invocation bound was reached.
#[must_use]
pub fn invocation_limit(limit: u64) -> CallbackRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("callback_limit_exceeded"),
        STAGE_EXECUTE,
        "callback_protocol.invocation_limit",
    )
    .with_supplied(limit.to_string())
    .with_remedy("raise the invocation bound of the execution policy")
}

/// The callback returned after the declared wall-clock bound. The in-process
/// executor cannot preempt a closure: the late result is discarded.
#[must_use]
pub fn wall_limit_exceeded(limit_ms: u64, observed_ms: u128) -> CallbackRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("callback_limit_exceeded"),
        STAGE_EXECUTE,
        "callback_protocol.wall_limit_exceeded",
    )
    .with_expected(limit_ms.to_string())
    .with_supplied(observed_ms.to_string())
    .with_remedy(
        "the in-process executor enforces this bound after the call returns and discards the result; a hard preemptible limit needs a declared isolated runtime",
    )
}

/// Cancellation was observed.
#[must_use]
pub fn cancelled(boundary: &str) -> CallbackRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("cancelled_no_claim"),
        STAGE_EXECUTE,
        "callback_protocol.cancelled",
    )
    .with_offending(boundary)
    .with_remedy("rerun without cancellation; the cancellation is not a verdict on the callback")
}

/// The provider reported a failure.
#[must_use]
pub fn provider_failed(message: &str) -> CallbackRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("callback_provider_unavailable"),
        STAGE_EXECUTE,
        "callback_protocol.provider_failed",
    )
    .with_supplied(message)
    .with_remedy("repair the provider and execute again; no claim is issued from a failed call")
}

/// No compatible provider is available in this process.
#[must_use]
pub fn provider_unavailable(identity: &str) -> CallbackRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("callback_provider_unavailable"),
        STAGE_PLAN,
        "callback_protocol.provider_unavailable",
    )
    .with_expected(identity)
    .with_remedy(
        "supply a compatible provider of this exact object, version, snapshot and request, or a declared portable runtime with resolved dependencies; the historical result stays inspectable",
    )
}

// -- closed routes (cell_not_licensed, joint_law_required) -----------------------------------------

/// An interval from a callback law is closed.
#[must_use]
pub fn interval_not_licensed() -> CallbackRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("cell_not_licensed"),
        STAGE_CLOSED,
        "callback_protocol.interval_not_licensed",
    )
    .with_remedy(
        "a callback supplies values, not coverage: an interval needs a whole-method coverage record measured at this coordinate",
    )
}

/// A posterior or probability guarantee from a callback is closed.
#[must_use]
pub fn posterior_guarantee_not_licensed() -> CallbackRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("cell_not_licensed"),
        STAGE_CLOSED,
        "callback_protocol.posterior_guarantee_not_licensed",
    )
    .with_remedy(
        "a callback posterior is the provider's object; a posterior or probability guarantee needs a measured coverage record for the whole method",
    )
}

/// Native trust cannot be conferred by a callback declaration.
#[must_use]
pub fn native_trust_not_conferred() -> CallbackRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("cell_not_licensed"),
        STAGE_CLOSED,
        "callback_protocol.native_trust_not_conferred",
    )
    .with_remedy(
        "metadata never confers native licence; implement and license the method natively, or keep the result an external claim",
    )
}

/// A callback cannot mint an identification licence.
#[must_use]
pub fn identification_not_conferred() -> CallbackRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("cell_not_licensed"),
        STAGE_CLOSED,
        "callback_protocol.identification_not_conferred",
    )
    .with_remedy(
        "identification is established from a graph and its assumptions; bind the callback output to a checked causal contract instead",
    )
}

/// A callback cannot mint a joint law.
#[must_use]
pub fn joint_law_not_conferred() -> CallbackRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("joint_law_required"),
        STAGE_CLOSED,
        "callback_protocol.joint_law_not_conferred",
    )
    .with_remedy(
        "declare an aligned joint law or a covariance with an alignment identity every input shares; independently produced callback draws are never paired by index",
    )
}
