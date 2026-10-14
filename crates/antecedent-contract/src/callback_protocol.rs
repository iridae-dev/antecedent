//! The callback adapter protocol: how an arbitrary law, posterior or nested
//! callback is declared, negotiated, planned and its output accepted, without
//! executing anything.
//!
//! A callback is a function the library does not own. This module is the one
//! place that says what it may claim and when a stored answer of it may be
//! reused:
//!
//! * [`CallbackDeclaration`] states the operation set (mean, sample,
//!   distribution function, quantile function and posterior update are
//!   negotiated separately: one never implies another), the input and output
//!   schema, the estimand, quantity and meaning of the law, the provider
//!   object and request identity, the data, model, file, remote and
//!   environment dependencies, the randomness, state and side-effect contract,
//!   the support and the trust the provider claims. It is an *instance of the
//!   one interface-contract schema*
//!   ([`CallbackDeclaration::interface_contract`]) and bridges to the one
//!   provider contract ([`CallbackDeclaration::provider_object`]); there is no
//!   second format and no second registry.
//! * [`plan_reuse`] decides, from identities alone, whether a stored result
//!   answers a request, must be recomputed, or the request is refused.
//!   Opaque or undeclared dependencies make reuse ineligible; a stateful
//!   callback is reused only while the state identity it reports through the
//!   protocol is unchanged; a side-effecting callback is reused or retried only
//!   under a declared idempotent contract. Nothing in this module can invoke a
//!   provider, so planning and verification never run one.
//! * [`validate_output`] accepts a returned value only if it is a well-formed
//!   answer of the declared quantity and support; a malformed law is refused.
//! * [`standing_of`] separates a diagnostic result from an external claim a
//!   licensed route may consume. Neither is native; metadata cannot mint
//!   native trust, a joint law or an identification licence, and no interval or
//!   posterior guarantee is carried.
//! * [`callback_source_declaration`] registers a callback and its evidence in
//!   the one evidence ledger so nested callbacks' overlap is computed.
//!
//! The executor, with its invocation counts, cancellation and resource policy,
//! is in the analysis facade; the portable result is an artifact of
//! `antecedent-io-core`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]

mod declaration;
mod ledger;
mod plan;
mod query;
/// Typed refusals: one named constructor per registered detail.
pub mod refusal;

pub use declaration::{
    CALLBACK_OUTPUT_PORT, CALLBACK_PROTOCOL_VERSION, CallbackDeclaration, CallbackDependency,
    CallbackInput, CallbackKind, CallbackOperation, CallbackTrust, DependencyDeclaration,
    DependencyKind, InputSupplier, MAX_CALLBACK_DEPENDENCIES, MAX_CALLBACK_INPUTS,
    MAX_CALLBACK_TEXT_BYTES, ReplayContract, RetryContract, UpdateDeclaration,
};
pub use ledger::{callback_source_declaration, callback_source_label};
pub use plan::{
    CallbackPhase, CallbackPlan, CallbackRequest, RecomputeReason, ResultStanding, StoredCallback,
    hidden_state_refusal, plan_reuse, semantic_stream, standing_label, standing_of,
};
pub use query::{
    CallbackOutput, CallbackQuery, MAX_CALLBACK_DRAWS, MAX_CALLBACK_POINTS, ValidatedOutput,
    validate_output,
};
pub use refusal::CallbackRefusal;
