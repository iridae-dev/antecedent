//! Request identity, semantic stream identities, reuse planning and result
//! labels.
//!
//! Planning and verification live here and take no provider: nothing in this
//! module can invoke a callback. A plan says whether a stored result may be
//! reused for a request, must be recomputed, or the request is refused, from
//! declarations, identities and a stored record alone.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::identity::{CanonicalWriter, SemanticDigest};
use antecedent_core::{CausalRng, RngFactory};

use super::declaration::{
    CallbackDeclaration, CallbackDependency, CallbackTrust, DependencyDeclaration, ReplayContract,
    RetryContract,
};
use super::query::CallbackQuery;
use super::refusal::{
    CallbackRefusal, diagnostic_only, hidden_state_unobservable, no_idempotency_contract,
    opaque_dependencies,
};

const TAG_REQUEST: &str = "antecedent.callback_protocol.request";
const TAG_STREAM: &str = "antecedent.callback_protocol.stream";

/// Where in a run a callback is asked about.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum CallbackPhase {
    /// Building a plan: reads declarations only.
    Planning,
    /// Replaying or checking a stored result: reads records only.
    Verification,
    /// An explicit execution: the only phase that may invoke a provider.
    Execution,
}

impl CallbackPhase {
    /// Stable `snake_case` name.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Planning => "planning",
            Self::Verification => "verification",
            Self::Execution => "execution",
        }
    }

    /// Whether a provider may be invoked in this phase.
    #[must_use]
    pub const fn may_invoke(self) -> bool {
        matches!(self, Self::Execution)
    }
}

/// One callback request: the query, the seed, the supplied input digests and
/// the idempotency key.
#[derive(Clone, Debug, PartialEq)]
pub struct CallbackRequest {
    /// The question.
    pub query: CallbackQuery,
    /// Master seed of the callback's stochastic stream. Part of the request
    /// identity unless the callback is deterministic.
    pub seed: u64,
    /// Digest (hex) of the value supplied for each input, by input name.
    pub input_digests: BTreeMap<String, String>,
    /// Idempotency key a side-effecting callback may be retried under.
    pub idempotency_key: Option<String>,
}

impl CallbackRequest {
    /// A request with no inputs and no key.
    #[must_use]
    pub fn new(query: CallbackQuery, seed: u64) -> Self {
        Self { query, seed, input_digests: BTreeMap::new(), idempotency_key: None }
    }

    /// The identity of this request against `declaration`: the declaration
    /// identity, the query as bit patterns, the seed (not for a deterministic
    /// callback, which has none), the input digests by name and the key.
    #[must_use]
    pub fn identity(&self, declaration: &CallbackDeclaration) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(TAG_REQUEST);
        writer.bytes(declaration.canonical_identity().as_bytes());
        self.query.write_canonical(&mut writer);
        match declaration.replay {
            ReplayContract::DeterministicPure => {
                writer.word(0);
            }
            _ => {
                writer.word(1).word(self.seed);
            }
        }
        writer.word(self.input_digests.len() as u64);
        for (name, digest) in &self.input_digests {
            writer.text(name).text(digest);
        }
        writer.opt_text(self.idempotency_key.as_deref());
        SemanticDigest::from_bytes(writer.finish())
    }
}

/// The stream a seeded callback draws from, by semantic identity.
///
/// The stream depends only on the master seed and the stream identity text, not
/// on how many calls came before or in what order, so replay and a different
/// scheduling give the same draws and two stream identities give independent
/// streams.
#[must_use]
pub fn semantic_stream(master_seed: u64, stream_id: &str) -> CausalRng {
    let digest = CanonicalWriter::digest_bytes(TAG_STREAM, stream_id.as_bytes());
    let mut word = [0u8; 8];
    word.copy_from_slice(&digest[..8]);
    RngFactory::from_seed(master_seed).stream(u64::from_le_bytes(word))
}

/// A stored result as planning sees it: identities and snapshots only.
#[derive(Clone, Debug, PartialEq)]
pub struct StoredCallback {
    /// Identity of the declaration the result was produced under.
    pub declaration_identity: SemanticDigest,
    /// Identity of the request it answered.
    pub request_identity: SemanticDigest,
    /// The dependency versions in force when it was produced (empty when the
    /// declaration was not complete).
    pub dependencies: Vec<CallbackDependency>,
    /// The state identity the provider reported after producing it.
    pub state_identity: Option<String>,
}

/// Why a stored result is not reused.
#[derive(Clone, Debug, Eq, PartialEq)]
pub enum RecomputeReason {
    /// There is no stored result.
    NoPriorResult,
    /// A dependency is opaque or undeclared: no stable identity.
    OpaqueDependencies,
    /// The declaration changed in a way no dependency version explains (the
    /// provider, snapshot, request or contract).
    DeclarationChanged,
    /// A declared dependency changed version; dependents are invalidated.
    DependencyChanged {
        /// The changed dependency's name.
        name: String,
        /// Version stored with the result.
        was: String,
        /// Version declared now.
        now: String,
    },
    /// The request differs.
    RequestChanged,
    /// A stateful callback exposes no state identity, so a stored result cannot
    /// be shown current.
    HiddenStateUnobservable,
    /// The reported state identity is not the stored one.
    StateChanged,
    /// The provider did not report a state identity when asked.
    StateUnknown,
}

/// What planning decides for one request. Planning never invokes.
#[derive(Clone, Debug, PartialEq)]
pub enum CallbackPlan {
    /// The stored result answers the request; no invocation.
    Reuse,
    /// The callback must be invoked again, under the policy of the phase that
    /// asks.
    Recompute(RecomputeReason),
    /// The request is refused before any invocation.
    Refuse(CallbackRefusal),
}

impl CallbackPlan {
    /// Whether the stored result is reused.
    #[must_use]
    pub const fn is_reuse(&self) -> bool {
        matches!(self, Self::Reuse)
    }
}

fn dependency_change(
    declaration: &CallbackDeclaration,
    stored: &StoredCallback,
) -> Option<RecomputeReason> {
    let DependencyDeclaration::Complete(now) = &declaration.dependencies else {
        return None;
    };
    for dependency in now {
        let before = stored
            .dependencies
            .iter()
            .find(|old| old.kind == dependency.kind && old.name == dependency.name);
        match before {
            Some(old) if old.version == dependency.version => {}
            Some(old) => {
                return Some(RecomputeReason::DependencyChanged {
                    name: dependency.name.clone(),
                    was: old.version.clone(),
                    now: dependency.version.clone(),
                });
            }
            None => {
                return Some(RecomputeReason::DependencyChanged {
                    name: dependency.name.clone(),
                    was: String::new(),
                    now: dependency.version.clone(),
                });
            }
        }
    }
    None
}

/// Decide whether `stored` answers `request`, from identities alone.
///
/// `current_state` is the state identity the provider reports now (or `None`
/// when it has not been, or cannot be, asked). The decision, in order:
///
/// 1. an opaque or undeclared dependency set gives no stable identity: a
///    side-effecting callback is refused (it is never run to find out), any
///    other recomputes;
/// 2. no stored result, a changed declaration, a changed dependency version or
///    a changed request recompute (a side-effecting callback with no
///    idempotency contract is refused instead);
/// 3. otherwise the replay contract decides: deterministic and seeded results
///    are reused, a stateful one only when it exposes a state identity and it
///    is unchanged, a side-effecting one only under an idempotent contract
///    with the same key.
#[must_use]
pub fn plan_reuse(
    declaration: &CallbackDeclaration,
    request: &CallbackRequest,
    stored: Option<&StoredCallback>,
    current_state: Option<&str>,
) -> CallbackPlan {
    let side_effecting = declaration.replay.has_side_effects();
    if !declaration.dependencies_complete() {
        return if side_effecting {
            CallbackPlan::Refuse(opaque_dependencies())
        } else {
            CallbackPlan::Recompute(RecomputeReason::OpaqueDependencies)
        };
    }
    let idempotent = matches!(
        declaration.replay,
        ReplayContract::SideEffecting { retry: RetryContract::IdempotentKey { .. } }
    );
    let Some(stored) = stored else {
        return CallbackPlan::Recompute(RecomputeReason::NoPriorResult);
    };
    if stored.declaration_identity != declaration.canonical_identity() {
        let reason =
            dependency_change(declaration, stored).unwrap_or(RecomputeReason::DeclarationChanged);
        return if side_effecting && !idempotent {
            CallbackPlan::Refuse(no_idempotency_contract())
        } else {
            CallbackPlan::Recompute(reason)
        };
    }
    if stored.request_identity != request.identity(declaration) {
        return if side_effecting && !idempotent {
            CallbackPlan::Refuse(no_idempotency_contract())
        } else {
            CallbackPlan::Recompute(RecomputeReason::RequestChanged)
        };
    }
    match &declaration.replay {
        ReplayContract::DeterministicPure | ReplayContract::SeededStochastic { .. } => {
            CallbackPlan::Reuse
        }
        ReplayContract::Stateful { exposes_state_identity } => {
            if !exposes_state_identity {
                return CallbackPlan::Recompute(RecomputeReason::HiddenStateUnobservable);
            }
            match (current_state, stored.state_identity.as_deref()) {
                (Some(now), Some(then)) if now == then => CallbackPlan::Reuse,
                (Some(_), _) => CallbackPlan::Recompute(RecomputeReason::StateChanged),
                (None, _) => CallbackPlan::Recompute(RecomputeReason::StateUnknown),
            }
        }
        ReplayContract::SideEffecting { retry } => match retry {
            RetryContract::IdempotentKey { .. } if request.idempotency_key.is_some() => {
                CallbackPlan::Reuse
            }
            _ => CallbackPlan::Refuse(no_idempotency_contract()),
        },
    }
}

/// The refusal a stateful callback without a state identity raises when asked
/// for a replayable claim; planning's [`RecomputeReason::HiddenStateUnobservable`]
/// as a typed refusal.
#[must_use]
pub fn hidden_state_refusal() -> CallbackRefusal {
    hidden_state_unobservable()
}

/// Whether a result stands as a diagnostic or as an external claim that may
/// feed a licensed route. Neither is native.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub enum ResultStanding {
    /// Inspectable; no claim may be built on it.
    Diagnostic,
    /// An external claim: replayable by contract, attested or verified. It
    /// carries no interval, posterior guarantee, joint law or identification.
    LicensedExternal,
}

impl ResultStanding {
    /// Stable `snake_case` label.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Diagnostic => "diagnostic",
            Self::LicensedExternal => "licensed_external",
        }
    }
}

/// The standing of a result of `declaration`.
///
/// `verification_confirmed` is true only when the caller holds the
/// verification receipt the declaration names and it covers the exact object
/// ([`CallbackDeclaration::confirms_verification`]); a declared verification
/// without it counts as no verification.
///
/// # Errors
/// Returns the reason the result is diagnostic.
pub fn standing_of(
    declaration: &CallbackDeclaration,
    verification_confirmed: bool,
) -> Result<ResultStanding, CallbackRefusal> {
    if !declaration.dependencies_complete() {
        return Err(diagnostic_only("opaque_dependencies"));
    }
    match &declaration.replay {
        ReplayContract::DeterministicPure | ReplayContract::SeededStochastic { .. } => {}
        ReplayContract::Stateful { .. } => return Err(diagnostic_only("stateful")),
        ReplayContract::SideEffecting { .. } => return Err(diagnostic_only("side_effecting")),
    }
    match &declaration.trust {
        CallbackTrust::Unverified => Err(diagnostic_only("unverified")),
        CallbackTrust::ExternallyAttested { .. } => Ok(ResultStanding::LicensedExternal),
        CallbackTrust::ExactRequestVerified { .. } => {
            if verification_confirmed {
                Ok(ResultStanding::LicensedExternal)
            } else {
                Err(diagnostic_only("verification_not_confirmed"))
            }
        }
    }
}

/// The standing of a result, with the reason it is diagnostic folded into
/// [`ResultStanding::Diagnostic`].
#[must_use]
pub fn standing_label(
    declaration: &CallbackDeclaration,
    verification_confirmed: bool,
) -> ResultStanding {
    standing_of(declaration, verification_confirmed).unwrap_or(ResultStanding::Diagnostic)
}
