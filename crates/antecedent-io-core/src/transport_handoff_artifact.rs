//! Portable record of a transported claim handed to an external consumer
//! (`transport_handoff_v1`).
//!
//! A bounded, checksummed container holding the claim as the interface-contract
//! port it was offered as, the transport identity it was carried under
//! (source populations, target population, selection diagram identity,
//! assumptions), the consumer's contract and consumed port, the policy the
//! consumer accepted it under, and the identities that bind them. When the
//! consumer is a declared callback its declaration identity is recorded too.
//!
//! # Provider trust
//!
//! An offer built from a provider's output may record which of the two external
//! trust levels stood behind it, `externally_attested` or `verified_extension`
//! (exact-request verified), so a reader of the artifact alone can tell them apart
//! without the evidence ledger. The label is part of the artifact identity. It is a
//! declaration: the receipt that would confirm a verified label does not travel in
//! the artifact, so the label is never read as native licence and never raises the
//! claim's standing. `native_licensed` and `unverified` are not accepted.
//!
//! # Refresh
//!
//! A handoff is a record of one claim at one moment. New data at a source population
//! produces a new offer: *recompute* the offer, handoff and artifact identities; *keep*
//! the consumer's contract and the policy; *refuse* a stored artifact whose identities
//! no longer re-derive. Nothing is refreshed in place.
//!
//! # What a consumer does
//!
//! It trusts none of the stored identities. It rebuilds the offer and the
//! consumer contract (every library validation applies), re-derives the offer's
//! port from the offer so a relabelled calibration, provenance or evidence is
//! refused, re-runs the connection decision and recomputes the handoff and
//! artifact identities. A stored policy of `native_target_only` is refused, as
//! is a claim whose quantity is not in the target population of its transport
//! identity. Nothing is invoked: the record never executes the consumer.
//!
//! # What this is not
//!
//! The record is a declaration. It does not prove that the populations are the
//! ones the data came from, that the selection diagram holds, or that the
//! consumer will use the claim as declared. It never presents the claim as
//! native target data and it carries no interval, posterior or calibration
//! statement of its own. A consumer that holds the callback declaration can
//! additionally call [`TransportHandoffArtifact::verify_against_callback`].
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::callback_protocol::CallbackDeclaration;
use antecedent_contract::evidence_ledger::{TransportIdentity, parse_digest_hex};
use antecedent_contract::interface_contract::InterfaceContract;
use antecedent_contract::provider_transport::{
    TransportedClaimOffer, TransportedHandoff, TransportedInputPolicy, hand_to_callback,
    hand_to_consumer,
};
use antecedent_contract::{ExternalRefusal, ProviderTrust};
use antecedent_core::refusal::changed_identity;
use antecedent_core::{CanonicalWriter, SemanticDigest};
use serde::{Deserialize, Serialize};

use crate::convert::to_cbor;
use crate::interface_contract_artifact::{
    ContractWire, InterfaceContractArtifactError, contract_from_wire, contract_to_wire,
};

/// The artifact major version this reader writes and accepts.
pub const TRANSPORT_HANDOFF_ARTIFACT_VERSION: u16 = 1;
/// The feature marker of the accepted format.
pub const TRANSPORT_HANDOFF_ARTIFACT_FEATURE: &str = "transport_handoff_v1";
/// Most bytes a transport handoff artifact may occupy.
pub const MAX_TRANSPORT_HANDOFF_ARTIFACT_BYTES: usize = 8 * 1024 * 1024;
/// Domain tag of the artifact identity stream.
pub const TRANSPORT_HANDOFF_ARTIFACT_TAG: &str = "antecedent.provider_transport.artifact";

const ARTIFACT_KIND: &str = "transport_handoff_v1";
const META_SECTION: &str = "transport_handoff_meta";

crate::sectioned_artifact_enum! {
    /// Why a transport handoff artifact was refused or could not be produced.
    #[derive(Clone, Debug, PartialEq, Eq)]
    TransportHandoffArtifactError {
        noun: "transport handoff",
        identity: "provider_transport.identity_mismatch",
        variants {
            /// The library refused the stored declaration.
            Refused {
                /// Registered reason code.
                code: &'static str,
                /// Namespaced `provider_transport.*` or `interface_contract.*` detail.
                detail: String,
                /// The declaration that would make the handoff well formed.
                message: String,
            },
        }
        display(f) {
            Self::Refused { detail, message, .. } => write!(f, "{detail}: {message}"),
        }
    }
}

impl TransportHandoffArtifactError {
    /// The registered refusal this error carries: `(code, detail, explanation)`.
    ///
    /// Present for a refusal of the stored declaration and for a changed
    /// identity; absent for corruption, unsupported versions or features,
    /// malformed values, limits and encoding failures.
    #[must_use]
    pub fn refusal(&self) -> Option<(&'static str, String, String)> {
        match self {
            Self::Refused { code, detail, message } => {
                Some((*code, detail.clone(), message.clone()))
            }
            Self::IdentityMismatch { field } => {
                Some(changed_identity("provider_transport.identity_mismatch", field))
            }
            _ => None,
        }
    }
}

type Failure = TransportHandoffArtifactError;

fn malformed(text: impl Into<String>) -> Failure {
    Failure::Malformed(text.into())
}

fn refused(refusal: ExternalRefusal) -> Failure {
    Failure::Refused {
        code: refusal.code,
        detail: refusal.detail,
        message: refusal.remedy.unwrap_or("the transport handoff is not well formed").to_owned(),
    }
}

fn first_refusal(refusals: Vec<ExternalRefusal>) -> Failure {
    refusals.into_iter().next().map_or_else(|| malformed("a refusal without a refusal"), refused)
}

fn from_contract_error(error: InterfaceContractArtifactError) -> Failure {
    match error {
        InterfaceContractArtifactError::Refused { code, detail, message } => {
            Failure::Refused { code, detail, message }
        }
        InterfaceContractArtifactError::UnsupportedSemantics(what) => {
            Failure::UnsupportedSemantics(what)
        }
        InterfaceContractArtifactError::LimitsExceeded(what) => Failure::LimitsExceeded(what),
        other => malformed(other.to_string()),
    }
}

// -- wire ---------------------------------------------------------------------------------------

/// A transport identity on the wire: sets sorted and distinct.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportIdentityWire {
    /// Source populations, sorted and distinct.
    pub source_populations: Vec<String>,
    /// The population the claim was carried to.
    pub target_population: String,
    /// Selection diagram identity (lowercase hex).
    pub selection_diagram: String,
    /// Declared assumptions, sorted and distinct.
    pub assumptions: Vec<String>,
}

/// Stored metadata of a transport handoff artifact.
#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TransportHandoffMeta {
    /// Metadata format version.
    pub version: u16,
    /// Feature marker.
    pub feature: String,
    /// Features a reader must understand. This version defines none; a reader
    /// refuses any entry.
    pub required_features: Vec<String>,
    /// The offered claim as the one produced port of this contract.
    pub offer_contract: ContractWire,
    /// Name of the offered port.
    pub offer_port: String,
    /// What the claim was carried under.
    pub transport: TransportIdentityWire,
    /// The consumer's contract, as declared.
    pub consumer_contract: ContractWire,
    /// The consumed port the claim was handed to.
    pub consumed_port: String,
    /// `accepts_transported`; `native_target_only` is refused on consumption.
    pub policy: String,
    /// Declaration identity (hex) of the consuming callback, when it is one.
    pub callback_identity: Option<String>,
    /// Trust label of the provider behind the claim, when recorded:
    /// `externally_attested` or `verified_extension`. Part of the identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub provider_trust: Option<String>,
    /// Handoff identity (hex), re-derived on consumption.
    pub handoff_identity: String,
    /// BLAKE3 identity (hex) of the artifact: version, feature, required
    /// features, handoff identity and callback identity.
    pub identity: String,
}

fn transport_to_wire(transport: &TransportIdentity) -> TransportIdentityWire {
    let mut populations = transport.source_populations.clone();
    populations.sort_unstable();
    populations.dedup();
    let mut assumptions = transport.assumptions.clone();
    assumptions.sort_unstable();
    assumptions.dedup();
    TransportIdentityWire {
        source_populations: populations,
        target_population: transport.target_population.clone(),
        selection_diagram: transport.selection_diagram.to_hex(),
        assumptions,
    }
}

fn strictly_increasing(values: &[String], what: &'static str) -> Result<(), Failure> {
    if crate::interface_contract_artifact::is_strictly_increasing(values) {
        Ok(())
    } else {
        Err(malformed(format!("{what} are not sorted and distinct")))
    }
}

/// Whether `text` is a 64-digit lowercase hex digest (checked on bytes, no allocation).
fn is_hex_digest(text: &str) -> bool {
    text.len() == 64 && text.bytes().all(|b| matches!(b, b'0'..=b'9' | b'a'..=b'f'))
}

fn transport_from_wire(wire: &TransportIdentityWire) -> Result<TransportIdentity, Failure> {
    strictly_increasing(&wire.source_populations, "source populations")?;
    strictly_increasing(&wire.assumptions, "assumptions")?;
    let selection_diagram = parse_digest_hex(&wire.selection_diagram)
        .filter(|digest| digest.to_hex() == wire.selection_diagram)
        .ok_or_else(|| malformed("selection diagram identity is not a lowercase digest"))?;
    Ok(TransportIdentity {
        source_populations: wire.source_populations.clone(),
        target_population: wire.target_population.clone(),
        selection_diagram,
        assumptions: wire.assumptions.clone(),
    })
}

// -- identity and consumption ---------------------------------------------------------------------

/// Read a stored provider trust label: only the two external levels, in their
/// canonical spelling.
fn parse_provider_trust(label: &str) -> Result<ProviderTrust, Failure> {
    match ProviderTrust::from_wire(label) {
        Some(trust @ (ProviderTrust::ExternallyAttested | ProviderTrust::ExactRequestVerified))
            if trust.as_wire() == label =>
        {
            Ok(trust)
        }
        _ => Err(Failure::UnsupportedSemantics("provider trust label")),
    }
}

fn compute_identity(meta: &TransportHandoffMeta) -> String {
    let mut writer = CanonicalWriter::new(TRANSPORT_HANDOFF_ARTIFACT_TAG);
    writer
        .word(u64::from(meta.version))
        .text(&meta.feature)
        .texts(&meta.required_features)
        .text(&meta.handoff_identity)
        .opt_text(meta.callback_identity.as_deref());
    // Written only when recorded, so an artifact without a label keeps its identity.
    if let Some(trust) = &meta.provider_trust {
        writer.text("provider_trust").text(trust);
    }
    writer.finish_hex()
}

/// The fully re-derived parts of a handoff artifact.
struct Derived {
    offer: TransportedClaimOffer,
    consumer: InterfaceContract,
    handoff: TransportedHandoff,
}

fn derive(meta: &TransportHandoffMeta) -> Result<Derived, Failure> {
    let offer_contract = contract_from_wire(&meta.offer_contract).map_err(from_contract_error)?;
    let [port] = offer_contract.produces.as_slice() else {
        return Err(malformed("the offer contract holds exactly one produced port"));
    };
    if !offer_contract.consumes.is_empty() || port.name != meta.offer_port {
        return Err(malformed("the offer contract holds only the offered port"));
    }
    let transport = transport_from_wire(&meta.transport)?;
    let offer = TransportedClaimOffer::from_port(&offer_contract.component, port, transport)
        .map_err(refused)?;
    // The stored port must be the one the offer projects to: a relabelled
    // calibration, provenance stage or evidence is a different declaration.
    if offer.interface_contract().canonical_identity() != offer_contract.canonical_identity() {
        return Err(Failure::IdentityMismatch { field: "offer_contract" });
    }
    let consumer = contract_from_wire(&meta.consumer_contract).map_err(from_contract_error)?;
    consumer.validate().map_err(refused)?;
    let policy = TransportedInputPolicy::from_name(&meta.policy)
        .ok_or(Failure::UnsupportedSemantics("input policy"))?;
    let handoff =
        hand_to_consumer(&offer, &consumer, &meta.consumed_port, policy).map_err(first_refusal)?;
    Ok(Derived { offer, consumer, handoff })
}

fn check(
    meta: &TransportHandoffMeta,
    expected_identity: Option<&str>,
) -> Result<TransportHandoffArtifact, Failure> {
    if meta.version != TRANSPORT_HANDOFF_ARTIFACT_VERSION {
        return Err(Failure::UnsupportedVersion { version: meta.version });
    }
    if meta.feature != TRANSPORT_HANDOFF_ARTIFACT_FEATURE {
        return Err(Failure::UnsupportedSemantics("feature marker"));
    }
    if !meta.required_features.is_empty() {
        return Err(Failure::UnsupportedSemantics("required feature"));
    }
    if meta.callback_identity.as_deref().is_some_and(|identity| !is_hex_digest(identity)) {
        return Err(malformed("callback identity is not a lowercase digest"));
    }
    if let Some(label) = &meta.provider_trust {
        parse_provider_trust(label)?;
    }
    let derived = derive(meta)?;
    if meta.handoff_identity != derived.handoff.identity().to_hex() {
        return Err(Failure::IdentityMismatch { field: "handoff_identity" });
    }
    let identity = compute_identity(meta);
    if meta.identity != identity {
        return Err(Failure::IdentityMismatch { field: "identity" });
    }
    if expected_identity.is_some_and(|expected| expected != identity) {
        return Err(Failure::IdentityMismatch { field: "retained_identity" });
    }
    Ok(TransportHandoffArtifact {
        offer: derived.offer,
        consumer: derived.consumer,
        consumed_port: meta.consumed_port.clone(),
        callback_identity: meta.callback_identity.clone(),
        provider_trust: meta.provider_trust.as_deref().map(parse_provider_trust).transpose()?,
        handoff: derived.handoff,
        identity,
    })
}

fn encode_meta(meta: &TransportHandoffMeta, artifact_id: &str) -> Result<Vec<u8>, Failure> {
    if artifact_id.trim().is_empty() {
        return Err(Failure::Encode("missing artifact id".into()));
    }
    let meta_bytes = to_cbor(meta).map_err(|error| Failure::Encode(error.to_string()))?;
    crate::encode_container(
        &crate::ArtifactContainer {
            kind: ARTIFACT_KIND,
            note: "transport_handoff",
            max_bytes: MAX_TRANSPORT_HANDOFF_ARTIFACT_BYTES,
            artifact_id,
            library_version: env!("CARGO_PKG_VERSION"),
        },
        &[crate::OutSection {
            id: META_SECTION,
            content_type: crate::CONTENT_CBOR,
            bytes: &meta_bytes,
        }],
    )
    .map_err(|error| crate::map_artifact_check!(error, TransportHandoffArtifactError))
}

fn decode_meta(bytes: &[u8]) -> Result<TransportHandoffMeta, Failure> {
    let parts = crate::decode_container(
        ARTIFACT_KIND,
        MAX_TRANSPORT_HANDOFF_ARTIFACT_BYTES,
        &[META_SECTION],
        bytes,
    )
    .map_err(|error| crate::map_artifact_check!(error, TransportHandoffArtifactError))?;
    let Ok([meta_bytes]) = <[Vec<u8>; 1]>::try_from(parts) else {
        return Err(Failure::Malformed("unsupported container layout".into()));
    };
    crate::decode_with_version(&meta_bytes, |version: u16| {
        if version == TRANSPORT_HANDOFF_ARTIFACT_VERSION {
            Ok(())
        } else {
            Err(Failure::UnsupportedVersion { version })
        }
    })
}

// -- public type ----------------------------------------------------------------------------------

/// A validated transport handoff and the identities that bind it.
#[derive(Clone, Debug)]
pub struct TransportHandoffArtifact {
    offer: TransportedClaimOffer,
    consumer: InterfaceContract,
    consumed_port: String,
    callback_identity: Option<String>,
    provider_trust: Option<ProviderTrust>,
    handoff: TransportedHandoff,
    identity: String,
}

impl TransportHandoffArtifact {
    fn build_meta(
        offer: &TransportedClaimOffer,
        consumer: &InterfaceContract,
        consumed_port: &str,
        policy: TransportedInputPolicy,
        callback_identity: Option<String>,
        provider_trust: Option<ProviderTrust>,
        handoff: &TransportedHandoff,
    ) -> TransportHandoffMeta {
        let mut meta = TransportHandoffMeta {
            version: TRANSPORT_HANDOFF_ARTIFACT_VERSION,
            feature: TRANSPORT_HANDOFF_ARTIFACT_FEATURE.to_owned(),
            required_features: Vec::new(),
            offer_contract: contract_to_wire(&offer.interface_contract()),
            offer_port: offer.port.clone(),
            transport: transport_to_wire(&offer.transport),
            consumer_contract: contract_to_wire(consumer),
            consumed_port: consumed_port.to_owned(),
            policy: policy.as_str().to_owned(),
            callback_identity,
            provider_trust: provider_trust.map(|trust| trust.as_wire().to_owned()),
            handoff_identity: handoff.identity().to_hex(),
            identity: String::new(),
        };
        meta.identity = compute_identity(&meta);
        meta
    }

    /// Seal a handoff to a consumer's port. The result passes the checks a
    /// consumer applies, so a handoff the library refuses does not seal.
    ///
    /// # Errors
    /// A malformed offer or consumer, a consumer that needs native target
    /// data, a connection defect, or an encoding failure.
    pub fn seal(
        offer: &TransportedClaimOffer,
        consumer: &InterfaceContract,
        consumed_port: &str,
        policy: TransportedInputPolicy,
    ) -> Result<(Self, TransportHandoffMeta), Failure> {
        let handoff =
            hand_to_consumer(offer, consumer, consumed_port, policy).map_err(first_refusal)?;
        let meta = Self::build_meta(offer, consumer, consumed_port, policy, None, None, &handoff);
        let artifact = check(&meta, None)?;
        Ok((artifact, meta))
    }

    /// Seal a handoff and record the provider trust level behind the claim.
    ///
    /// # Errors
    /// As [`Self::seal`], and a trust level other than `ExternallyAttested`
    /// or `ExactRequestVerified`.
    pub fn seal_with_provider_trust(
        offer: &TransportedClaimOffer,
        consumer: &InterfaceContract,
        consumed_port: &str,
        policy: TransportedInputPolicy,
        provider_trust: ProviderTrust,
    ) -> Result<(Self, TransportHandoffMeta), Failure> {
        if !matches!(
            provider_trust,
            ProviderTrust::ExternallyAttested | ProviderTrust::ExactRequestVerified
        ) {
            return Err(Failure::UnsupportedSemantics("provider trust label"));
        }
        let handoff =
            hand_to_consumer(offer, consumer, consumed_port, policy).map_err(first_refusal)?;
        let meta = Self::build_meta(
            offer,
            consumer,
            consumed_port,
            policy,
            None,
            Some(provider_trust),
            &handoff,
        );
        let artifact = check(&meta, None)?;
        Ok((artifact, meta))
    }

    /// Seal a handoff to an input of a declared callback; the declaration's
    /// identity is recorded.
    ///
    /// # Errors
    /// As [`Self::seal`], and an input another callback supplies.
    pub fn seal_for_callback(
        offer: &TransportedClaimOffer,
        callback: &CallbackDeclaration,
        input: &str,
        policy: TransportedInputPolicy,
    ) -> Result<(Self, TransportHandoffMeta), Failure> {
        let handoff = hand_to_callback(offer, callback, input, policy).map_err(first_refusal)?;
        let meta = Self::build_meta(
            offer,
            &callback.interface_contract(),
            input,
            policy,
            Some(callback.canonical_identity().to_hex()),
            None,
            &handoff,
        );
        let artifact = check(&meta, None)?;
        Ok((artifact, meta))
    }

    /// Serialize a sealed handoff through the checksummed container.
    ///
    /// # Errors
    /// A handoff that does not seal, an encoding failure or an oversized payload.
    pub fn to_bytes(
        offer: &TransportedClaimOffer,
        consumer: &InterfaceContract,
        consumed_port: &str,
        policy: TransportedInputPolicy,
        artifact_id: &str,
    ) -> Result<Vec<u8>, Failure> {
        let (_, meta) = Self::seal(offer, consumer, consumed_port, policy)?;
        encode_meta(&meta, artifact_id)
    }

    /// Serialize a sealed handoff that records its provider trust level.
    ///
    /// # Errors
    /// As [`Self::seal_with_provider_trust`], an encoding failure or an
    /// oversized payload.
    pub fn to_bytes_with_provider_trust(
        offer: &TransportedClaimOffer,
        consumer: &InterfaceContract,
        consumed_port: &str,
        policy: TransportedInputPolicy,
        provider_trust: ProviderTrust,
        artifact_id: &str,
    ) -> Result<Vec<u8>, Failure> {
        let (_, meta) =
            Self::seal_with_provider_trust(offer, consumer, consumed_port, policy, provider_trust)?;
        encode_meta(&meta, artifact_id)
    }

    /// Consume an artifact: rebuild the offer and the consumer, re-derive the
    /// connection and recompute every identity.
    ///
    /// `expected_identity` is an artifact identity (hex) the consumer retained
    /// independently; when given, a consistently resealed artifact of another
    /// handoff is refused.
    ///
    /// # Errors
    /// Corruption, another major version, an unknown or required feature, a
    /// bound exceeded, a declaration the library refuses, a policy that needs
    /// native target data, or a stored identity that does not match.
    pub fn from_bytes(bytes: &[u8], expected_identity: Option<&str>) -> Result<Self, Failure> {
        check(&decode_meta(bytes)?, expected_identity)
    }

    /// Read the stored metadata without replaying it.
    ///
    /// # Errors
    /// Corruption, another major version or a malformed container.
    pub fn inspect(bytes: &[u8]) -> Result<TransportHandoffMeta, Failure> {
        decode_meta(bytes)
    }

    /// Check the handoff against a callback declaration the consumer holds:
    /// the recorded declaration identity equals the declaration's, and the
    /// connection holds for that declaration, so an input another callback
    /// supplies is refused here.
    ///
    /// # Errors
    /// `provider_transport.identity_mismatch` for another declaration, or the
    /// refusal of [`hand_to_callback`].
    pub fn verify_against_callback(&self, callback: &CallbackDeclaration) -> Result<(), Failure> {
        let supplied = callback.canonical_identity().to_hex();
        if self.callback_identity.as_deref() != Some(supplied.as_str()) {
            return Err(Failure::IdentityMismatch { field: "callback_identity" });
        }
        let rederived =
            hand_to_callback(&self.offer, callback, &self.consumed_port, self.handoff.policy())
                .map_err(first_refusal)?;
        if rederived.identity() != self.handoff.identity() {
            return Err(Failure::IdentityMismatch { field: "handoff_identity" });
        }
        Ok(())
    }

    /// The offered claim.
    #[must_use]
    pub const fn offer(&self) -> &TransportedClaimOffer {
        &self.offer
    }

    /// The consumer's contract.
    #[must_use]
    pub const fn consumer(&self) -> &InterfaceContract {
        &self.consumer
    }

    /// The consumed port.
    #[must_use]
    pub fn consumed_port(&self) -> &str {
        &self.consumed_port
    }

    /// The re-derived handoff.
    #[must_use]
    pub const fn handoff(&self) -> &TransportedHandoff {
        &self.handoff
    }

    /// The provider trust level recorded with the claim, when one was: always
    /// an external level (attested or exact-request verified), never native.
    #[must_use]
    pub const fn provider_trust(&self) -> Option<ProviderTrust> {
        self.provider_trust
    }

    /// The handoff identity (hex).
    #[must_use]
    pub fn handoff_identity(&self) -> String {
        self.handoff.identity().to_hex()
    }

    /// The artifact identity (hex).
    #[must_use]
    pub fn identity(&self) -> &str {
        &self.identity
    }

    /// Always false: the record never presents the claim as native target data.
    #[must_use]
    pub const fn is_native_target_data(&self) -> bool {
        false
    }

    /// The transport identity that travels with the claim.
    #[must_use]
    pub fn transport(&self) -> &TransportIdentity {
        &self.offer.transport
    }

    /// The digest of a selection diagram identity as stored, for tests.
    #[doc(hidden)]
    #[must_use]
    pub fn selection_diagram(&self) -> SemanticDigest {
        self.offer.transport.selection_diagram
    }

    /// Validate metadata as a consumer would, for tests that reseal edits.
    ///
    /// # Errors
    /// The same refusals as [`Self::from_bytes`].
    #[doc(hidden)]
    pub fn from_meta(
        meta: &TransportHandoffMeta,
        expected_identity: Option<&str>,
    ) -> Result<Self, Failure> {
        check(meta, expected_identity)
    }

    /// Recompute the identity of `meta` after an edit, for tests that reseal.
    #[doc(hidden)]
    pub fn reseal_identity(meta: &mut TransportHandoffMeta) {
        meta.identity = compute_identity(meta);
    }

    /// Serialize metadata as it is, for tests that write an edited layout.
    ///
    /// # Errors
    /// An encoding failure.
    #[doc(hidden)]
    pub fn meta_to_bytes(
        meta: &TransportHandoffMeta,
        artifact_id: &str,
    ) -> Result<Vec<u8>, Failure> {
        encode_meta(meta, artifact_id)
    }
}
