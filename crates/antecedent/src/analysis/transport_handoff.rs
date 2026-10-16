//! Handing a transported claim to a declared callback, verified in one call.
//!
//! A claim carried to a target population (source populations, target population,
//! selection diagram, assumptions) may be consumed by an external callback only
//! under a policy that accepts transported claims, and only at an input the
//! callback declares. [`hand_to_callback`] does both halves of that check so a
//! consumer need not remember the second one:
//!
//! 1. it seals the offer to the callback's declared input
//!    (`TransportHandoffArtifact::seal_for_callback`: every library validation of
//!    the offer, the callback's interface contract and the connection applies,
//!    and a policy of `native_target_only` is refused), and
//! 2. it immediately verifies the sealed handoff against the callback declaration
//!    (`verify_against_callback`: the recorded declaration identity equals the
//!    declaration's and the connection re-derives for that declaration).
//!
//! [`consume_for_callback`] is the consumer's counterpart: it re-derives the
//! handoff from bytes (offer, consumer contract, connection, every identity) and
//! then verifies it against the declaration the consumer holds, so a handoff
//! sealed for another callback, or an input another callback supplies, is refused
//! at consumption, never later.
//!
//! The record is a declaration. It does not prove the populations are the ones
//! the data came from, that the selection diagram holds, or that the callback
//! will use the claim as declared; it never presents the claim as native target
//! data and carries no interval, posterior or calibration statement of its own.
//! Nothing is invoked.
//!
//! # Lifecycle
//!
//! A handoff is a record of one connection. A refreshed offer (new outcome-only
//! data of the source) is a new offer: **recompute** the offer's identity and
//! the handoff identity; **keep** the callback declaration and the consumed
//! input; **refuse** an offer whose quantity is not in the target population of
//! its transport identity, and a changed callback declaration (the recorded
//! declaration identity no longer matches).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

pub use antecedent_contract::callback_protocol::CallbackDeclaration;
pub use antecedent_contract::provider_transport::{
    TransportedClaimOffer, TransportedHandoff, TransportedInputPolicy,
};
pub use antecedent_io::transport_handoff_artifact::{
    MAX_TRANSPORT_HANDOFF_ARTIFACT_BYTES, TRANSPORT_HANDOFF_ARTIFACT_FEATURE,
    TRANSPORT_HANDOFF_ARTIFACT_VERSION, TransportHandoffArtifact, TransportHandoffArtifactError,
    TransportHandoffMeta,
};

/// One refusal as plain data.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportHandoffRefusal {
    /// Registered runtime reason code.
    pub code: &'static str,
    /// Namespaced `provider_transport.*` or `interface_contract.*` detail.
    pub detail: String,
    /// What would make the handoff well formed.
    pub message: String,
}

impl std::fmt::Display for TransportHandoffRefusal {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "{}{}: {}: {}",
            antecedent_core::reason_code::PREFIX,
            self.code,
            self.detail,
            self.message
        )
    }
}

impl std::error::Error for TransportHandoffRefusal {}

impl From<TransportHandoffArtifactError> for TransportHandoffRefusal {
    fn from(error: TransportHandoffArtifactError) -> Self {
        match error.refusal() {
            Some((code, detail, message)) => Self { code, detail, message },
            None => Self {
                code: antecedent_core::reason_code!("invalid_argument"),
                detail: "provider_transport.artifact_invalid".to_owned(),
                message: error.to_string(),
            },
        }
    }
}

/// A sealed handoff to a callback input that has been verified against the
/// callback's declaration.
#[derive(Clone, Debug)]
pub struct CallbackHandoff {
    artifact: TransportHandoffArtifact,
    meta: TransportHandoffMeta,
}

impl CallbackHandoff {
    /// The validated handoff.
    #[must_use]
    pub const fn artifact(&self) -> &TransportHandoffArtifact {
        &self.artifact
    }

    /// The transport identity that travels with the claim.
    #[must_use]
    pub fn transport(&self) -> &antecedent_contract::evidence_ledger::TransportIdentity {
        self.artifact.transport()
    }

    /// The artifact identity (hex); retain it independently of the bytes.
    #[must_use]
    pub fn identity(&self) -> &str {
        self.artifact.identity()
    }

    /// Always false: the record never presents the claim as native target data.
    #[must_use]
    pub const fn is_native_target_data(&self) -> bool {
        false
    }

    /// Serialize through the checksummed container.
    ///
    /// # Errors
    /// An empty id, an encoding failure or an oversized payload.
    pub fn to_bytes(&self, artifact_id: &str) -> Result<Vec<u8>, TransportHandoffRefusal> {
        TransportHandoffArtifact::meta_to_bytes(&self.meta, artifact_id).map_err(Into::into)
    }
}

/// Seal `offer` to `input` of `callback` and verify the result against the
/// callback's declaration.
///
/// # Errors
/// A malformed offer or callback, a policy of `native_target_only`, a quantity
/// outside the target population, an input another callback supplies or a
/// connection defect, each with its registered code and detail.
pub fn hand_to_callback(
    offer: &TransportedClaimOffer,
    callback: &CallbackDeclaration,
    input: &str,
    policy: TransportedInputPolicy,
) -> Result<CallbackHandoff, TransportHandoffRefusal> {
    let (artifact, meta) =
        TransportHandoffArtifact::seal_for_callback(offer, callback, input, policy)?;
    artifact.verify_against_callback(callback)?;
    Ok(CallbackHandoff { artifact, meta })
}

/// Consume a handoff's bytes and verify them against the callback declaration
/// the consumer holds.
///
/// `expected_identity` is an artifact identity retained independently of the
/// bytes; when given, a consistently resealed handoff of another claim is
/// refused.
///
/// # Errors
/// Corruption, another version, a declaration the library refuses, a stored
/// identity that does not match, a handoff sealed for another callback
/// (`provider_transport.identity_mismatch`) or a connection that no longer holds.
pub fn consume_for_callback(
    bytes: &[u8],
    expected_identity: Option<&str>,
    callback: &CallbackDeclaration,
) -> Result<TransportHandoffArtifact, TransportHandoffRefusal> {
    let artifact = TransportHandoffArtifact::from_bytes(bytes, expected_identity)?;
    artifact.verify_against_callback(callback)?;
    Ok(artifact)
}
