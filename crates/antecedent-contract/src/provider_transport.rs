//! External providers and consumers of transported evidence, through the one
//! interface contract.
//!
//! Two routes, each decided from declarations before any number is computed
//! and each with a typed refusal:
//!
//! * **A provider's output as source evidence for transport.**
//!   [`ProviderTransportSource`] declares one provider object at an exact
//!   request, the trust it holds ([`ExternalTrustState`]: attestation and
//!   exact-request verification stay apart), the source population and regime
//!   of the quantity it answers, the invariance and selection assumptions it
//!   relies on, and the evidence it rests on. [`ProviderTransportSource::admit`]
//!   refuses a missing population or regime, a quantity the provider object
//!   does not declare, a native licence, a verification receipt that does not
//!   cover the object, and a provider that is only attested where the route
//!   requires exact-request verification. An admitted source is an instance of
//!   the interface contract (a produced port that supplies the
//!   `external_provider` provenance stage) and an `external_claim` source in
//!   the one evidence ledger. Neither makes it native target data.
//! * **A transported claim handed to an external consumer.**
//!   [`TransportedClaimOffer`] is a claim about the *target* population that
//!   carries its [`TransportIdentity`] (source populations, selection
//!   diagram, assumptions). [`hand_to_consumer`] and [`hand_to_callback`] connect
//!   it to a consumed port, requiring the transformation provenance stage on
//!   the consumer's side, so the consumer receives the claim together with its
//!   identity. A consumer that accepts only native target data refuses it,
//!   and an accepted handoff stays a transported external claim: it is never
//!   presented as native target data and its calibration label is never
//!   raised.
//!
//! Nothing here executes a provider, a callback or any declared code.
//! Declarations are data; the decision is about what they state.
//!
//! # Refresh
//!
//! A source or offer is an immutable declaration. New data at a provider's
//! population is a new declaration: *recompute* the admitted source (identity,
//! evidence and trust follow the new declaration) and every handoff built on
//! it; *keep* the transport identity and the consumer's port when the
//! population, regime, quantity and assumptions are unchanged; *refuse* a
//! source whose population is not a source population of the transport
//! (`provider_transport.population_not_a_source`), an assumption the transport
//! does not carry, and data of another quantity or arm. Nothing is refreshed
//! in place: a refused refresh leaves the previous admission untouched.
//!
//! # What is and is not decided
//!
//! A declared population is the producer's declaration, not proof that the
//! data came from it; verification receipts cover provider behaviour at exact
//! requests, not the truth of an invariance assumption. A handoff licenses no
//! interval, posterior or calibration claim.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::result_large_err,
    reason = "a refusal is the cold path of a once-per-call check; boxing the error would not pay"
)]

use std::collections::BTreeSet;

use antecedent_core::identity::{CanonicalWriter, SemanticDigest};

use crate::callback_protocol::{CallbackDeclaration, InputSupplier};
use crate::composition_provenance::CompositionStage;
use crate::evidence_ledger::{Lineage, SourceDeclaration, SourceId, SourceKind, TransportIdentity};
use crate::interface_contract::{
    CalibrationStatus, Connection, ConsumedPort, ContractOperation, DependenceRequirement,
    EvidenceBasis, InterfaceContract, InterfaceRefusal, ProducedPort, SupportDomain,
    check_connection, operations_of_capabilities,
};
use crate::{
    DistributionMeaning, ExternalPosteriorKind, ExternalRefusal, ExternalScientificObject,
    ExternalTrustState, ProviderTrust, ScientificQuantity,
};
use antecedent_core::named_variants;

/// Version word written into every identity of this module. Bump only when a
/// canonical encoding below changes.
pub const PROVIDER_TRANSPORT_VERSION: u64 = 1;
/// Most assumptions, populations or evidence sources one declaration may name.
pub const MAX_PROVIDER_TRANSPORT_ITEMS: usize = 256;
/// Longest identity, in bytes.
pub const MAX_PROVIDER_TRANSPORT_TEXT_BYTES: usize = 4096;
/// Port name of the quantity an admitted provider source produces.
pub const PROVIDER_SOURCE_PORT: &str = "provider_source";
/// Port name of the quantity a transport route consumes from a provider.
pub const TRANSPORT_SOURCE_PORT: &str = "transport_source";

const TAG_SOURCE: &str = "antecedent.provider_transport.source";
const TAG_OFFER: &str = "antecedent.provider_transport.offer";
const TAG_HANDOFF: &str = "antecedent.provider_transport.handoff";
const STAGE_DECLARE: &str = "declare";
const STAGE_BIND: &str = "bind";
const STAGE_VERIFY: &str = "verify";

// -- refusals ---------------------------------------------------------------------------------------

fn declare(detail: &str, remedy: &'static str) -> ExternalRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("interface_contract_invalid"),
        STAGE_DECLARE,
        detail,
    )
    .with_remedy(remedy)
}

fn untrusted(detail: &str, remedy: &'static str) -> ExternalRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("external_verification_failed"),
        STAGE_VERIFY,
        detail,
    )
    .with_remedy(remedy)
}

fn not_bound(detail: &str, remedy: &'static str) -> ExternalRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("external_binding_mismatch"),
        STAGE_BIND,
        detail,
    )
    .with_remedy(remedy)
}

fn quantity_mismatch(detail: &str, remedy: &'static str) -> ExternalRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("interface_contract_quantity_mismatch"),
        STAGE_BIND,
        detail,
    )
    .with_remedy(remedy)
}

fn assumption_mismatch(detail: &str, remedy: &'static str) -> ExternalRefusal {
    ExternalRefusal::new(
        antecedent_core::reason_code!("interface_contract_assumption_mismatch"),
        STAGE_BIND,
        detail,
    )
    .with_remedy(remedy)
}

fn text_ok(text: &str) -> bool {
    !text.trim().is_empty() && text.len() <= MAX_PROVIDER_TRANSPORT_TEXT_BYTES
}

fn write_support(writer: &mut CanonicalWriter, support: &SupportDomain) {
    match support {
        SupportDomain::Real => {
            writer.word(1);
        }
        SupportDomain::NonNegative => {
            writer.word(2);
        }
        SupportDomain::Binary => {
            writer.word(3);
        }
        SupportDomain::Interval { lower_bits, upper_bits } => {
            writer.word(4).word(*lower_bits).word(*upper_bits);
        }
        SupportDomain::Categorical { levels } => {
            writer.word(5).word(levels.len() as u64);
            for level in levels {
                writer.text(level);
            }
        }
    }
}

fn write_evidence(writer: &mut CanonicalWriter, evidence: &EvidenceBasis) {
    match evidence {
        EvidenceBasis::Undeclared => {
            writer.word(0);
        }
        EvidenceBasis::Declared(sources) => {
            writer.word(1).word(sources.len() as u64);
            for source in sources {
                writer.text(&source.to_hex());
            }
        }
    }
}

fn write_operations(writer: &mut CanonicalWriter, operations: &BTreeSet<ContractOperation>) {
    writer.word(operations.len() as u64);
    for operation in operations {
        writer.word(operation.tag());
    }
}

// -- trust ------------------------------------------------------------------------------------------

/// What trust a transport route requires of a provider's output.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum SourceTrustRequirement {
    /// A supplier attestation is enough. The source stays externally
    /// attested; nothing is verified here.
    AttestationAccepted,
    /// The provider object must hold an exact-request verification receipt
    /// that covers it.
    ExactRequestVerified,
}

named_variants! {
    SourceTrustRequirement {
        AttestationAccepted => "attestation_accepted",
        ExactRequestVerified => "exact_request_verified",
    }
}

// -- provider output as source evidence ---------------------------------------------------------------

/// A provider's output offered as source evidence for a transport route.
#[derive(Clone, Debug, PartialEq)]
pub struct ProviderTransportSource {
    /// Component name in the interface contract.
    pub component: String,
    /// The provider object at its exact request.
    pub object: ExternalScientificObject,
    /// The trust the object holds.
    pub trust: ExternalTrustState,
    /// The coordinate the output answers, including the source population and
    /// regime it was established in. It must be one the object declares.
    pub quantity: ScientificQuantity,
    /// The estimand or semantic identity the value answers.
    pub semantic_identity: String,
    /// The values the quantity takes.
    pub support: SupportDomain,
    /// Invariance and selection assumptions the output relies on.
    pub assumptions: Vec<String>,
    /// The evidence sources (ledger identities) the output rests on.
    pub evidence: EvidenceBasis,
    /// What the route requires of the provider's trust.
    pub requirement: SourceTrustRequirement,
}

fn object_quantities(object: &ExternalScientificObject) -> Vec<&ScientificQuantity> {
    match object {
        ExternalScientificObject::Law(value) => value.quantities.iter().collect(),
        ExternalScientificObject::Posterior(value) => value.quantities.iter().collect(),
        _ => Vec::new(),
    }
}

/// The meaning of the provider's randomness, for the objects this route
/// admits: a law, and a parameter or causal-functional posterior.
fn object_meaning(object: &ExternalScientificObject) -> Option<DistributionMeaning> {
    match object {
        ExternalScientificObject::Law(value) => Some(value.meaning),
        ExternalScientificObject::Posterior(value) => match value.kind {
            ExternalPosteriorKind::Parameter => Some(DistributionMeaning::ParameterPosterior),
            ExternalPosteriorKind::CausalFunctional => {
                Some(DistributionMeaning::CausalFunctionalPosterior)
            }
            ExternalPosteriorKind::Functional | ExternalPosteriorKind::StructuralModel => None,
        },
        _ => None,
    }
}

impl ProviderTransportSource {
    /// Check the declaration and, only if it holds, admit the source.
    ///
    /// # Errors
    /// The first defect, in a fixed order, as a registered refusal with a
    /// `provider_transport.*` detail and the declaration that would license
    /// the route: a missing population or regime
    /// (`population_missing`, `regime_missing`), an incomplete quantity,
    /// a provider object this route does not admit or that is ill formed,
    /// missing assumptions, a quantity the object does not declare, a native
    /// licence, a receipt that does not cover the object, or attestation where
    /// verification is required (`verification_required`).
    pub fn admit(&self) -> Result<AdmittedProviderSource, ExternalRefusal> {
        if !text_ok(&self.component) || !text_ok(&self.semantic_identity) {
            return Err(declare(
                "provider_transport.blank_field",
                "name the component and the semantic identity of the quantity",
            ));
        }
        if self.quantity.population_id.trim().is_empty() {
            return Err(declare(
                "provider_transport.population_missing",
                "declare the population the provider's output was established in",
            )
            .with_offending("quantity.population_id"));
        }
        if self.quantity.regime_id.trim().is_empty() {
            return Err(declare(
                "provider_transport.regime_missing",
                "declare the intervention regime (`observational` is explicit) the provider's output answers",
            )
            .with_offending("quantity.regime_id"));
        }
        if self.quantity.validate().is_err() || !self.support.is_well_formed() {
            return Err(declare(
                "provider_transport.quantity_invalid",
                "name the variable, units, functional and transform of the quantity, with canonical conditioning and a well formed support",
            ));
        }
        if object_meaning(&self.object).is_none() {
            return Err(declare(
                "provider_transport.object_unsupported",
                "offer a law, a parameter posterior or a causal-functional posterior; other provider objects are not source evidence of this route",
            ));
        }
        if self.object.validate().is_err() {
            return Err(declare(
                "provider_transport.object_invalid",
                "declare a complete provider object with distinct quantities and operations",
            ));
        }
        if self.assumptions.is_empty()
            || self.assumptions.len() > MAX_PROVIDER_TRANSPORT_ITEMS
            || self.assumptions.iter().any(|a| !text_ok(a))
        {
            return Err(declare(
                "provider_transport.assumptions_missing",
                "name the invariance and selection assumptions the provider's output relies on",
            ));
        }
        if !object_quantities(&self.object)
            .iter()
            .any(|declared| declared.require_same_coordinate(&self.quantity).is_ok())
        {
            return Err(not_bound(
                "provider_transport.quantity_not_declared",
                "offer a quantity, population and regime that the provider object declares",
            )
            .with_supplied(self.quantity.canonical_identity()));
        }
        match &self.trust {
            ExternalTrustState::NativeLicensed => {
                return Err(declare(
                    "provider_transport.native_trust_claimed",
                    "a provider is external: declare it attested or exact-request verified; native licence is never conferred by metadata",
                ));
            }
            ExternalTrustState::ExternallyAttested { attestor } => {
                if !text_ok(attestor) {
                    return Err(declare(
                        "provider_transport.trust_incomplete",
                        "name the attestor of an attested provider",
                    ));
                }
                if self.requirement == SourceTrustRequirement::ExactRequestVerified {
                    return Err(untrusted(
                        "provider_transport.verification_required",
                        "supply an exact-request verification receipt that covers this provider object; an attestation does not stand in for it",
                    )
                    .with_expected("verified_extension")
                    .with_supplied("externally_attested"));
                }
            }
            ExternalTrustState::ExactRequestVerified(_) => {
                if !self.trust.verifies(&self.object) {
                    return Err(untrusted(
                        "provider_transport.receipt_does_not_cover",
                        "supply the receipt issued for this exact provider object, version, snapshot and request",
                    ));
                }
            }
        }
        Ok(AdmittedProviderSource { source: self.clone() })
    }
}

/// A provider source that passed [`ProviderTransportSource::admit`].
#[derive(Clone, Debug, PartialEq)]
pub struct AdmittedProviderSource {
    source: ProviderTransportSource,
}

impl AdmittedProviderSource {
    /// The admitted declaration.
    #[must_use]
    pub const fn source(&self) -> &ProviderTransportSource {
        &self.source
    }

    /// The trust level the provider holds. Never native.
    #[must_use]
    pub fn trust(&self) -> ProviderTrust {
        ProviderTrust::from_external_state(&self.source.trust)
    }

    /// Always false: a provider's output is never native target data.
    #[must_use]
    pub const fn is_native_target_data(&self) -> bool {
        false
    }

    /// Population the output was established in.
    #[must_use]
    pub fn population(&self) -> &str {
        &self.source.quantity.population_id
    }

    /// Regime the output answers.
    #[must_use]
    pub fn regime(&self) -> &str {
        &self.source.quantity.regime_id
    }

    /// The operations the provider object declares, as contract operations.
    #[must_use]
    pub fn operations(&self) -> BTreeSet<ContractOperation> {
        operations_of_capabilities(self.source.object.capabilities())
    }

    /// Canonical identity: component, provider object and request, quantity,
    /// support, sorted assumptions, evidence, trust and requirement. A changed
    /// request, population, regime, assumption or trust level changes it.
    #[must_use]
    pub fn canonical_identity(&self) -> SemanticDigest {
        let source = &self.source;
        let mut writer = CanonicalWriter::new(TAG_SOURCE);
        writer.word(PROVIDER_TRANSPORT_VERSION).text(&source.component);
        let identity = source.object.identity();
        for field in [
            &identity.provider_id,
            &identity.object_id,
            &identity.version_id,
            &identity.snapshot_id,
            &identity.request_id,
        ] {
            writer.text(field);
        }
        source.quantity.write_canonical(&mut writer);
        writer.text(&source.semantic_identity);
        write_support(&mut writer, &source.support);
        write_operations(&mut writer, &self.operations());
        let mut assumptions = source.assumptions.clone();
        assumptions.sort_unstable();
        assumptions.dedup();
        writer.texts(&assumptions);
        write_evidence(&mut writer, &source.evidence);
        match &source.trust {
            ExternalTrustState::ExternallyAttested { attestor } => {
                writer.word(1).text(attestor);
            }
            ExternalTrustState::ExactRequestVerified(receipt) => {
                writer.word(2).text(&receipt.digest());
            }
            ExternalTrustState::NativeLicensed => {
                writer.word(0);
            }
        }
        writer.text(source.requirement.as_str());
        SemanticDigest::from_bytes(writer.finish())
    }

    /// The source as one instance of the interface-contract schema: a produced
    /// port that supplies the `external_provider` provenance stage. The port
    /// is point-only for a mean alone and unmeasured otherwise; it is never
    /// measured, so no interval or calibration is carried.
    #[must_use]
    pub fn interface_contract(&self) -> InterfaceContract {
        let source = &self.source;
        let operations = self.operations();
        let carries = operations.iter().copied().any(ContractOperation::carries_uncertainty);
        let mut port = ProducedPort::new(
            PROVIDER_SOURCE_PORT,
            source.quantity.clone(),
            source.semantic_identity.clone(),
            source.support.clone(),
            operations,
        )
        .with_provenance([CompositionStage::ExternalProvider.as_str()]);
        if carries {
            if let Some(meaning) = object_meaning(&source.object) {
                port = port.with_uncertainty(meaning);
            }
            port = port.with_calibration(CalibrationStatus::Unmeasured);
        } else {
            port = port.with_calibration(CalibrationStatus::PointOnly);
        }
        if let EvidenceBasis::Declared(sources) = &source.evidence {
            port = port.with_evidence(sources.iter().copied());
        }
        InterfaceContract {
            component: source.component.clone(),
            produces: vec![port],
            consumes: Vec::new(),
        }
    }

    /// The evidence-ledger declaration: an `external_claim` source whose
    /// lineage is the evidence it rests on. Declared-empty evidence is a
    /// source with no data origin; undeclared evidence overlaps with unknown
    /// dependence, never independence. It carries no transport identity: it is
    /// source evidence, not a transported claim.
    #[must_use]
    pub fn ledger_declaration(&self) -> SourceDeclaration {
        let label = format!("provider_source:{}", self.canonical_identity().to_hex());
        let declaration = SourceDeclaration::new(label, SourceKind::ExternalClaim);
        match &self.source.evidence {
            EvidenceBasis::Undeclared => declaration,
            EvidenceBasis::Declared(sources) if sources.is_empty() => {
                declaration.with_observations(std::iter::empty())
            }
            EvidenceBasis::Declared(sources) => {
                declaration.with_lineage(Lineage::DerivedFromUsedData {
                    inputs: sources.iter().copied().collect(),
                })
            }
        }
    }

    /// Check that this source belongs to `transport`: its population is one
    /// of the declared source populations and every assumption it relies on
    /// is carried by the transport identity.
    ///
    /// # Errors
    /// `provider_transport.population_not_a_source` or
    /// `provider_transport.assumptions_not_carried`.
    pub fn require_bound_to(&self, transport: &TransportIdentity) -> Result<(), ExternalRefusal> {
        if !transport.source_populations.iter().any(|p| p == self.population()) {
            return Err(quantity_mismatch(
                "provider_transport.population_not_a_source",
                "use a provider whose population is a source population of the selection diagram, or declare the transport from that population",
            )
            .with_expected(transport.source_populations.join(","))
            .with_supplied(self.population()));
        }
        if let Some(missing) =
            self.source.assumptions.iter().find(|a| !transport.assumptions.contains(a))
        {
            return Err(assumption_mismatch(
                "provider_transport.assumptions_not_carried",
                "carry every assumption the provider's output relies on in the transport identity",
            )
            .with_offending(missing.clone()));
        }
        Ok(())
    }
}

/// The interface contract of a transport route's source-evidence consumer:
/// one consumed port that requires the declared evidence and the
/// `external_provider` provenance stage.
#[must_use]
pub fn transport_source_consumer(
    component: impl Into<String>,
    quantity: ScientificQuantity,
    semantic_identity: impl Into<String>,
    support: SupportDomain,
    operations: impl IntoIterator<Item = ContractOperation>,
) -> InterfaceContract {
    let port =
        ConsumedPort::new(TRANSPORT_SOURCE_PORT, quantity, semantic_identity, support, operations)
            .with_dependence(DependenceRequirement::DeclaredEvidence)
            .with_provenance([CompositionStage::ExternalProvider.as_str()]);
    InterfaceContract { component: component.into(), produces: Vec::new(), consumes: vec![port] }
}

/// Connect an admitted provider source to a transport route's consumer.
///
/// # Errors
/// Every defect of the connection, as [`check_connection`] reports them (a
/// source population or regime that is not the one the route needs, an
/// operation the provider does not offer, undeclared evidence, ...).
pub fn connect_provider_source(
    source: &AdmittedProviderSource,
    consumer: &InterfaceContract,
    consumed_port: &str,
) -> Result<Connection, Vec<InterfaceRefusal>> {
    check_connection(&source.interface_contract(), PROVIDER_SOURCE_PORT, consumer, consumed_port)
}

/// Connect an admitted provider source to a transport route's consumer,
/// after checking the source belongs to the transport the route carries.
///
/// The provider's population must be a source population of `transport` and
/// every assumption it relies on must be carried by it; otherwise the
/// connection is not attempted.
///
/// # Errors
/// `provider_transport.population_not_a_source`,
/// `provider_transport.assumptions_not_carried`, then every defect of
/// [`connect_provider_source`].
pub fn connect_bound_provider_source(
    source: &AdmittedProviderSource,
    transport: &TransportIdentity,
    consumer: &InterfaceContract,
    consumed_port: &str,
) -> Result<Connection, Vec<InterfaceRefusal>> {
    source.require_bound_to(transport).map_err(|refusal| vec![refusal])?;
    connect_provider_source(source, consumer, consumed_port)
}

// -- a transported claim handed to an external consumer -------------------------------------------------

/// What an external consumer accepts of a claim about its population.
#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum TransportedInputPolicy {
    /// Native target data only; a transported claim is refused.
    NativeTargetOnly,
    /// A transported claim is accepted together with its transport identity.
    AcceptsTransported,
}

named_variants! {
    TransportedInputPolicy {
        NativeTargetOnly => "native_target_only",
        AcceptsTransported => "accepts_transported",
    }
}

/// A claim about the target population that was carried there by transport.
#[derive(Clone, Debug, PartialEq)]
pub struct TransportedClaimOffer {
    /// Component name in the interface contract.
    pub component: String,
    /// Produced port name.
    pub port: String,
    /// The coordinate of the claim, in the target population.
    pub quantity: ScientificQuantity,
    /// The estimand or semantic identity the value answers.
    pub semantic_identity: String,
    /// The values the quantity takes.
    pub support: SupportDomain,
    /// The operations offered.
    pub operations: BTreeSet<ContractOperation>,
    /// What the offered law, draws or posterior means, when one is offered.
    pub uncertainty: Option<DistributionMeaning>,
    /// Source populations, target population, selection diagram and
    /// assumptions the claim was carried under.
    pub transport: TransportIdentity,
    /// The evidence sources (ledger identities) the transported claim rests
    /// on. Never empty: a transported claim is derived from something.
    pub evidence: BTreeSet<SourceId>,
}

impl TransportedClaimOffer {
    /// Check the offer is a well formed transported claim.
    ///
    /// # Errors
    /// `provider_transport.transport_identity_incomplete` when the identity
    /// names no source population, no assumption, a blank target, or a
    /// target that is also a source; `provider_transport.claim_not_in_target`
    /// when the quantity is not in the target population;
    /// `provider_transport.evidence_missing` for an offer with no evidence;
    /// and the interface-contract refusal of the projected port.
    pub fn validate(&self) -> Result<(), ExternalRefusal> {
        let transport = &self.transport;
        if !text_ok(&self.component)
            || !text_ok(&self.port)
            || !text_ok(&self.semantic_identity)
            || !text_ok(&transport.target_population)
            || transport.source_populations.is_empty()
            || transport.source_populations.len() > MAX_PROVIDER_TRANSPORT_ITEMS
            || transport.source_populations.iter().any(|p| !text_ok(p))
            || transport.assumptions.is_empty()
            || transport.assumptions.len() > MAX_PROVIDER_TRANSPORT_ITEMS
            || transport.assumptions.iter().any(|a| !text_ok(a))
            || transport.source_populations.contains(&transport.target_population)
        {
            return Err(declare(
                "provider_transport.transport_identity_incomplete",
                "name the source populations (none of them the target), the target population, the selection diagram and the assumptions the claim was carried under",
            ));
        }
        if self.quantity.population_id != transport.target_population {
            return Err(quantity_mismatch(
                "provider_transport.claim_not_in_target",
                "state the transported claim in the target population of its transport identity",
            )
            .with_expected(transport.target_population.clone())
            .with_supplied(self.quantity.population_id.clone()));
        }
        if self.evidence.is_empty() || self.evidence.len() > MAX_PROVIDER_TRANSPORT_ITEMS {
            return Err(declare(
                "provider_transport.evidence_missing",
                "name the ledger sources the transported claim was derived from",
            ));
        }
        self.interface_contract().validate()
    }

    /// The offer as one instance of the interface-contract schema: a produced
    /// port that supplies the `transformation` and `claim` provenance stages.
    /// It is point-only for a mean alone and unmeasured otherwise; transport
    /// never produces a measured calibration label.
    #[must_use]
    pub fn interface_contract(&self) -> InterfaceContract {
        let carries = self.operations.iter().copied().any(ContractOperation::carries_uncertainty);
        let mut port = ProducedPort::new(
            self.port.clone(),
            self.quantity.clone(),
            self.semantic_identity.clone(),
            self.support.clone(),
            self.operations.iter().copied(),
        )
        .with_provenance([
            CompositionStage::Transformation.as_str(),
            CompositionStage::Claim.as_str(),
        ])
        .with_evidence(self.evidence.iter().copied());
        if let Some(meaning) = self.uncertainty {
            port = port.with_uncertainty(meaning);
        }
        port = port.with_calibration(if carries {
            CalibrationStatus::Unmeasured
        } else {
            CalibrationStatus::PointOnly
        });
        InterfaceContract {
            component: self.component.clone(),
            produces: vec![port],
            consumes: Vec::new(),
        }
    }

    /// Rebuild an offer from a produced port and its transport identity.
    ///
    /// # Errors
    /// A port whose evidence is undeclared, or any refusal of
    /// [`Self::validate`].
    pub fn from_port(
        component: &str,
        port: &ProducedPort,
        transport: TransportIdentity,
    ) -> Result<Self, ExternalRefusal> {
        let EvidenceBasis::Declared(evidence) = &port.evidence else {
            return Err(declare(
                "provider_transport.evidence_missing",
                "name the ledger sources the transported claim was derived from",
            ));
        };
        let offer = Self {
            component: component.to_owned(),
            port: port.name.clone(),
            quantity: port.quantity.clone(),
            semantic_identity: port.semantic_identity.clone(),
            support: port.support.clone(),
            operations: port.operations.clone(),
            uncertainty: port.uncertainty.map(|u| u.meaning),
            transport,
            evidence: evidence.clone(),
        };
        offer.validate()?;
        Ok(offer)
    }

    /// Canonical identity: the port declaration and the transport identity.
    #[must_use]
    pub fn canonical_identity(&self) -> SemanticDigest {
        let mut writer = CanonicalWriter::new(TAG_OFFER);
        writer
            .word(PROVIDER_TRANSPORT_VERSION)
            .text(&self.component)
            .text(&self.port)
            .text(&self.semantic_identity);
        self.quantity.write_canonical(&mut writer);
        write_support(&mut writer, &self.support);
        write_operations(&mut writer, &self.operations);
        writer.opt_text(self.uncertainty.map(crate::quantity::meaning_tag));
        writer.text(&self.transport.digest().to_hex());
        writer.word(self.evidence.len() as u64);
        for source in &self.evidence {
            writer.text(&source.to_hex());
        }
        SemanticDigest::from_bytes(writer.finish())
    }

    /// The ledger declaration of the claim: an `external_claim` source with
    /// its transport identity, derived from the evidence it names. A source
    /// that carries a transport identity is never native target data.
    #[must_use]
    pub fn ledger_declaration(&self, label: impl Into<String>) -> SourceDeclaration {
        SourceDeclaration::new(label, SourceKind::ExternalClaim)
            .with_lineage(Lineage::DerivedFromUsedData {
                inputs: self.evidence.iter().copied().collect(),
            })
            .with_transport(self.transport.clone())
    }
}

/// An accepted handoff of a transported claim to one consumed port.
///
/// It records the connection and the transport identity that travels with the
/// claim. The claim stays a transported external claim.
#[derive(Clone, Debug, Eq, PartialEq)]
pub struct TransportedHandoff {
    connection: Connection,
    transport: TransportIdentity,
    policy: TransportedInputPolicy,
    identity: SemanticDigest,
}

impl TransportedHandoff {
    /// Stable label of what the consumer receives.
    pub const STANDING: &'static str = "transported_claim";

    /// The accepted connection (its calibration label is the producer's).
    #[must_use]
    pub const fn connection(&self) -> &Connection {
        &self.connection
    }

    /// The transport identity that travels with the claim.
    #[must_use]
    pub const fn transport(&self) -> &TransportIdentity {
        &self.transport
    }

    /// The policy the consumer accepted the claim under.
    #[must_use]
    pub const fn policy(&self) -> TransportedInputPolicy {
        self.policy
    }

    /// What the consumer receives: always `transported_claim`.
    #[must_use]
    pub const fn standing(&self) -> &'static str {
        Self::STANDING
    }

    /// Always false: the consumer is never given native target data.
    #[must_use]
    pub const fn is_native_target_data(&self) -> bool {
        false
    }

    /// Identity binding the connection, the offer and the transport identity.
    #[must_use]
    pub const fn identity(&self) -> SemanticDigest {
        self.identity
    }
}

/// Hand a transported claim to a consumed port of an external consumer.
///
/// The consumer's port additionally requires the `transformation` provenance
/// stage, which the offer supplies, so the claim arrives with its transport
/// identity rather than as an anonymous value.
///
/// # Errors
/// A malformed offer, a consumer that accepts native target data only
/// (`provider_transport.native_target_required`), and every defect of the
/// connection (population, regime, units, operations, uncertainty meaning,
/// calibration the consumer relies on, ...).
pub fn hand_to_consumer(
    offer: &TransportedClaimOffer,
    consumer: &InterfaceContract,
    consumed_port: &str,
    policy: TransportedInputPolicy,
) -> Result<TransportedHandoff, Vec<InterfaceRefusal>> {
    offer.validate().map_err(|refusal| vec![refusal])?;
    if policy == TransportedInputPolicy::NativeTargetOnly {
        return Err(vec![
            assumption_mismatch(
                "provider_transport.native_target_required",
                "supply native target data, or declare the consumer's input as accepting a transported claim with its transport identity",
            )
            .with_offending(consumed_port)
            .with_expected("native_target_data")
            .with_supplied(TransportedHandoff::STANDING),
        ]);
    }
    let mut required = consumer.clone();
    if let Some(port) = required.consumes.iter_mut().find(|port| port.name == consumed_port) {
        port.provenance.insert(CompositionStage::Transformation.as_str().to_owned());
    }
    let producer = offer.interface_contract();
    let connection = check_connection(&producer, &offer.port, &required, consumed_port)?;
    let mut writer = CanonicalWriter::new(TAG_HANDOFF);
    writer
        .word(PROVIDER_TRANSPORT_VERSION)
        .text(&connection.identity.to_hex())
        .text(&offer.canonical_identity().to_hex())
        .text(&offer.transport.digest().to_hex())
        .text(policy.as_str());
    Ok(TransportedHandoff {
        connection,
        transport: offer.transport.clone(),
        policy,
        identity: SemanticDigest::from_bytes(writer.finish()),
    })
}

/// Hand a transported claim to an input of a declared callback.
///
/// The input must be supplied from outside the callback graph; the
/// callback's declaration, projected through the interface contract, is the
/// consumer. Nothing is invoked.
///
/// # Errors
/// A malformed callback declaration, an input another callback supplies
/// (`provider_transport.input_not_external`), and every refusal of
/// [`hand_to_consumer`].
pub fn hand_to_callback(
    offer: &TransportedClaimOffer,
    callback: &CallbackDeclaration,
    input: &str,
    policy: TransportedInputPolicy,
) -> Result<TransportedHandoff, Vec<InterfaceRefusal>> {
    callback.validate().map_err(|refusal| vec![refusal])?;
    if callback
        .inputs
        .iter()
        .any(|candidate| candidate.name == input && candidate.supplier != InputSupplier::External)
    {
        return Err(vec![
            assumption_mismatch(
                "provider_transport.input_not_external",
                "hand a transported claim to an input supplied from outside the callback graph",
            )
            .with_offending(input),
        ]);
    }
    hand_to_consumer(offer, &callback.interface_contract(), input, policy)
}

/// Hand a transported claim to a consumed port together with the provider
/// sources it was derived from.
///
/// Every provider source must belong to the offer's transport identity (its
/// population is one of the source populations and its assumptions are
/// carried); a provider of another population cannot stand behind the claim.
///
/// # Errors
/// `provider_transport.population_not_a_source`,
/// `provider_transport.assumptions_not_carried`, then every refusal of
/// [`hand_to_consumer`].
pub fn hand_to_consumer_from_sources(
    offer: &TransportedClaimOffer,
    sources: &[&AdmittedProviderSource],
    consumer: &InterfaceContract,
    consumed_port: &str,
    policy: TransportedInputPolicy,
) -> Result<TransportedHandoff, Vec<InterfaceRefusal>> {
    offer.validate().map_err(|refusal| vec![refusal])?;
    for source in sources {
        source.require_bound_to(&offer.transport).map_err(|refusal| vec![refusal])?;
    }
    hand_to_consumer(offer, consumer, consumed_port, policy)
}
