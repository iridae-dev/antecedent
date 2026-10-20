//! The derived claim of a bridge: its own identity, assumptions, provenance
//! and receipt, with the source claim left separately inspectable.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::assumption::{AssumptionSet, AssumptionSource, AssumptionStatus};
use antecedent_core::identity::{CanonicalWriter, ContractIdentities, SemanticDigest};

use super::contract::{BridgeAssumption, BridgeContract};
use super::kind::{BridgeKind, ValueKind};
use super::mapping::{MappingKind, TrialLevelFit};
use super::refusal::{
    BridgeRefusal, interval_not_licensed, provenance_failed, quantity_mismatch, source_not_point,
    units_mismatch, wrong_value_kind,
};
use crate::claim::{
    ClaimDomains, ClaimEnvelope, ClaimFreshness, ClaimKind, DomainStatus, HandoffReceipt,
};
use crate::obligation::{ObligationKind, ObligationRecord, ObligationScope};
use crate::provenance::{ProvenanceGraph, ProvenanceNode};
use crate::quantity::{QuantityMismatch, ScientificQuantity};
use crate::reasoning::{
    AssumptionSlot, ReasoningView, SlotAvailability, UncertaintyComponent, UncertaintySlot,
    UncertaintySource,
};

/// Version word written into every bridged claim identity.
pub const BRIDGE_CLAIM_IDENTITY_VERSION: u64 = 1;

const TAG_CLAIM: &str = "antecedent.scientific_bridge.claim";
const TAG_TARGET: &str = "antecedent.scientific_bridge.target";

/// The claim being bridged and the quantity it is about.
///
/// A claim envelope does not carry its quantity, so the producer states it
/// here; the contract's source quantity must be the same coordinate, and the
/// claim's units must agree with it.
#[derive(Clone, Copy, Debug)]
pub struct BridgeSource<'a> {
    /// The source claim, left unchanged.
    pub claim: &'a ClaimEnvelope,
    /// The quantity the source claim is about.
    pub quantity: &'a ScientificQuantity,
}

/// A claim derived across a bridge.
///
/// The source claim is not part of it and is not modified: [`Self::source_claim_id`]
/// names it, and [`Self::claim`] is a different claim with a different
/// identity, a derivation provenance chain and the assumptions the bridge
/// states. It never reports identification, support or uncertainty the source
/// does not establish for the target quantity.
#[derive(Clone, Debug, PartialEq)]
#[non_exhaustive]
pub struct BridgedClaim {
    /// The derived claim envelope.
    pub claim: ClaimEnvelope,
    /// The contract the claim was derived under.
    pub contract: BridgeContract,
    /// Identity of the contract.
    pub contract_identity: SemanticDigest,
    /// Identity of the source claim.
    pub source_claim_id: SemanticDigest,
    /// The source value.
    pub source_value: f64,
    /// The bridged value.
    pub value: f64,
    /// The kind of object the value is.
    pub value_kind: ValueKind,
    /// The mapping that carried the value.
    pub mapping: MappingKind,
    /// The trial-level fit, when the mapping fitted one.
    pub fit: Option<TrialLevelFit>,
    /// Whether the bridge carries evidence established elsewhere than the
    /// claim's population or regime, under a declared premise.
    pub transported: bool,
    /// Ancestry: the source claim and this derivation.
    pub provenance: ProvenanceGraph,
    /// Derivation receipt: what was retained, omitted and left unresolved.
    pub receipt: HandoffReceipt,
}

/// The identity of the claim a contract derives from a source claim.
///
/// It binds the contract identity (which binds both quantities, the mapping
/// and every declared assumption) and the source claim identity.
#[must_use]
pub fn bridged_claim_id(contract: SemanticDigest, source_claim: SemanticDigest) -> SemanticDigest {
    let mut out = CanonicalWriter::new(TAG_CLAIM);
    out.word(BRIDGE_CLAIM_IDENTITY_VERSION)
        .bytes(contract.as_bytes())
        .bytes(source_claim.as_bytes());
    SemanticDigest::from_bytes(out.finish())
}

const fn mismatch_name(mismatch: QuantityMismatch) -> &'static str {
    match mismatch {
        QuantityMismatch::InvalidCoordinate => "invalid_coordinate",
        QuantityMismatch::Variable => "variable",
        QuantityMismatch::Units => "units",
        QuantityMismatch::Population => "population",
        QuantityMismatch::Regime => "regime",
        QuantityMismatch::Horizon => "horizon",
        QuantityMismatch::Functional => "functional",
        QuantityMismatch::Conditioning => "conditioning",
        QuantityMismatch::Transform => "transform",
    }
}

fn is_alignment_evidence(item: &str) -> bool {
    item.starts_with("alignment:")
        || ["joint_draws", "covariance", "replicate_ids"]
            .iter()
            .any(|kind| item == *kind || item.starts_with(&format!("{kind}:")))
}

fn node(id: SemanticDigest, operation: &str, parents: &[Arc<str>]) -> ProvenanceNode {
    ProvenanceNode {
        artifact_id: Arc::from(id.to_hex()),
        operation: Arc::from(operation),
        parents: parents.iter().cloned().collect(),
        assumptions: AssumptionSet::new(),
        library_version: Arc::from(antecedent_core::VERSION),
        config_digest: None,
    }
}

fn obligation(assumption: &BridgeAssumption) -> ObligationRecord {
    let mut record = ObligationRecord::new(
        format!("bridge:{}", assumption.id),
        ObligationScope::Program,
        AssumptionSource::UserDeclared,
        ObligationKind::UserAssertion,
        AssumptionStatus::Declared,
        assumption.statement.as_str(),
    );
    record.evidence = assumption.evidence.as_deref().map(Arc::from);
    record
}

/// Derive the claim a contract licenses from a source claim.
///
/// Nothing is derived unless the contract licenses its bridge
/// ([`BridgeContract::check_licensing`]) and the source claim is a finite
/// point value about the contract's source quantity.
///
/// # Errors
/// The first failed check, as a coded refusal: a contract that does not
/// license its bridge; a source claim about another quantity or in other units;
/// a source that is not a finite point value; a value outside the mapping's
/// validated or attainable range.
pub fn derive_bridged_claim(
    contract: &BridgeContract,
    source: &BridgeSource<'_>,
) -> Result<BridgedClaim, BridgeRefusal> {
    let licensing = contract.check_licensing()?;
    if let Err(mismatch) = source.quantity.require_same_coordinate(&contract.source.quantity) {
        return Err(quantity_mismatch(
            mismatch_name(mismatch),
            contract.source.quantity.canonical_identity(),
            source.quantity.canonical_identity(),
        ));
    }
    if let Some(units) = &source.claim.outcome_units {
        if &**units != source.quantity.units.as_str() {
            return Err(units_mismatch(source.quantity.units.clone(), units.to_string()));
        }
    }
    if source.claim.kind != ClaimKind::Point {
        return Err(source_not_point(source.claim.kind.as_str()));
    }
    let Some(source_value) = source.claim.value.filter(|value| value.is_finite()) else {
        return Err(source_not_point("no_finite_value"));
    };
    let applied = contract.mapping.apply(source_value, licensing.direction)?;

    let contract_identity = contract.identity();
    let source_claim_id = source.claim.claim_id;
    let claim_id = bridged_claim_id(contract_identity, source_claim_id);
    let identities = bridged_identities(contract, contract_identity, &source.claim.identities);
    let reasoning = bridged_reasoning(contract);
    let domains = ClaimDomains::new(
        DomainStatus::Unknown,
        DomainStatus::Unknown,
        source.claim.domains.evaluated,
    );
    let mut evidence: Vec<Arc<str>> =
        source.claim.evidence.iter().filter(|item| !is_alignment_evidence(item)).cloned().collect();
    evidence.push(Arc::from(format!("bridge:{}", contract_identity.to_hex())));
    evidence.push(Arc::from(format!("bridge_source:{}", source_claim_id.to_hex())));
    evidence.push(Arc::from(format!("bridge_kind:{}", contract.kind.as_str())));
    if let Some(uncertainty) = &contract.mapping_uncertainty {
        evidence.push(Arc::from(format!("mapping_uncertainty:{}", uncertainty.id)));
    }
    if let Some(dependence) = &contract.source_dependence {
        let label = match dependence {
            super::contract::SourceMappingDependence::Independent { evidence: name } => {
                format!("source_dependence:independent:{name}")
            }
            super::contract::SourceMappingDependence::SlopeCovariance {
                evidence: name, ..
            } => {
                format!("source_dependence:covariance:{name}")
            }
        };
        evidence.push(Arc::from(label));
    }
    if let Some(bridge) = &contract.estimand_bridge {
        evidence.push(Arc::from(format!("estimand_bridge:{}", bridge.id)));
    }
    let envelope = ClaimEnvelope::new(
        claim_id,
        identities,
        ClaimKind::Point,
        Some(applied.value),
        Some(Arc::from(contract.target.quantity.units.as_str())),
        reasoning,
        domains,
        source.claim.execution,
        evidence,
    );

    let mut provenance = ProvenanceGraph::new();
    let source_hex: Arc<str> = Arc::from(source_claim_id.to_hex());
    let operation = format!("scientific_bridge:{}", contract.kind.as_str());
    let record_failed = |_| provenance_failed();
    provenance.try_push(node(source_claim_id, "source_claim", &[])).map_err(record_failed)?;
    provenance
        .try_push(node(claim_id, &operation, std::slice::from_ref(&source_hex)))
        .map_err(record_failed)?;

    let receipt = bridge_receipt(contract, source_claim_id, claim_id, operation);
    Ok(BridgedClaim {
        claim: envelope,
        contract: contract.clone(),
        contract_identity,
        source_claim_id,
        source_value,
        value: applied.value,
        value_kind: contract.target_value,
        mapping: contract.mapping.kind(),
        fit: applied.fit,
        transported: licensing.transported,
        provenance,
        receipt,
    })
}

/// The derivation receipt: what is retained, what the bridge did not
/// establish, which declared assumptions stay unresolved, and what the derived
/// claim cannot offer.
fn bridge_receipt(
    contract: &BridgeContract,
    source_claim_id: SemanticDigest,
    claim_id: SemanticDigest,
    operation: String,
) -> HandoffReceipt {
    let mut unresolved: Vec<Arc<str>> = contract
        .assumptions
        .iter()
        .map(|assumption| Arc::<str>::from(format!("assumption:{}", assumption.id)))
        .collect();
    if let Some(premise) = &contract.transport {
        unresolved.push(Arc::from(format!("transportability:{}", premise.id)));
    }
    let mut retained =
        vec![Arc::from("claim.envelope"), Arc::from("parents"), Arc::from("provenance")];
    if contract.mapping_uncertainty.is_some() {
        retained.push(Arc::from("mapping_uncertainty"));
    }
    if contract.source_dependence.is_some() {
        retained.push(Arc::from("source_dependence"));
    }
    if contract.estimand_bridge.is_some() {
        retained.push(Arc::from("estimand_bridge"));
    }
    HandoffReceipt::new(
        source_claim_id,
        Some(claim_id),
        "scientific_bridge",
        operation,
        retained,
        [Arc::from("identification"), Arc::from("support"), Arc::from("uncertainty")],
        unresolved,
        [
            Arc::from("interval"),
            Arc::from("posterior"),
            Arc::from("joint_law"),
            Arc::from("accept_as_identified"),
        ],
    )
}

/// The identity layers of a bridged claim.
///
/// The target layer is the bridged question (the target quantity and class
/// under this contract). The identification, observation, inference binding
/// and data snapshot are the source's: the bridge adds assumptions, not data,
/// so evidence sharing with the source stays detectable. The program identity
/// is unavailable, as for any claim whose products were not prepared.
fn bridged_identities(
    contract: &BridgeContract,
    contract_identity: SemanticDigest,
    source: &ContractIdentities,
) -> ContractIdentities {
    let mut target = CanonicalWriter::new(TAG_TARGET);
    target.word(BRIDGE_CLAIM_IDENTITY_VERSION).word(contract.target.class.tag());
    contract.target.quantity.write_canonical(&mut target);
    target.bytes(contract_identity.as_bytes());
    ContractIdentities::new(
        SemanticDigest::from_bytes(target.finish()),
        source.identification,
        source.identification_product,
        None,
        source.inference_binding,
        source.observation,
        source.data_snapshot,
    )
}

/// Reasoning slots of a bridged claim: nothing is established for the target
/// quantity beyond the declared assumptions, which are listed as unresolved.
fn bridged_reasoning(contract: &BridgeContract) -> ReasoningView {
    let mut obligations: Vec<ObligationRecord> =
        contract.assumptions.iter().map(obligation).collect();
    if let Some(premise) = &contract.transport {
        let mut record = ObligationRecord::new(
            format!("bridge_transport:{}", premise.id),
            ObligationScope::Program,
            AssumptionSource::UserDeclared,
            ObligationKind::UserAssertion,
            AssumptionStatus::Declared,
            premise.statement.as_str(),
        );
        record.evidence = premise.evidence.as_deref().map(Arc::from);
        obligations.push(record);
    }
    ReasoningView::new(
        SlotAvailability::unavailable("bridge_assumptions_declared"),
        SlotAvailability::unavailable("bridge_support_not_established"),
        bridged_uncertainty(contract),
        SlotAvailability::Available(AssumptionSlot::new(obligations)),
    )
}

/// Mapping uncertainty and source dependence stay on the claim when the
/// contract declares them. The mapping variance is the sampling variance of the
/// estimated intercept and slope, so it is labelled sampling, not parameter. Neither declaration is invented when it is absent,
/// and an absent dependence is recorded as omitted rather than as independence.
fn bridged_uncertainty(contract: &BridgeContract) -> SlotAvailability<UncertaintySlot> {
    if contract.mapping_uncertainty.is_none() && contract.source_dependence.is_none() {
        return SlotAvailability::unavailable("bridge_uncertainty_not_propagated");
    }
    let mapping = match &contract.mapping_uncertainty {
        Some(uncertainty) => UncertaintyComponent::new(
            UncertaintySource::Sampling,
            format!("mapping_variance:{}", uncertainty.id),
            false,
        ),
        None => UncertaintyComponent::new(UncertaintySource::Sampling, "mapping_variance", true),
    };
    SlotAvailability::Available(UncertaintySlot::new([
        mapping,
        UncertaintyComponent::new(
            UncertaintySource::Sampling,
            "source_dependence",
            contract.source_dependence.is_none(),
        ),
    ]))
}

impl BridgedClaim {
    /// The bridge kind.
    #[must_use]
    pub const fn kind(&self) -> BridgeKind {
        self.contract.kind
    }

    /// The declared assumptions.
    #[must_use]
    pub fn assumptions(&self) -> &[BridgeAssumption] {
        &self.contract.assumptions
    }

    /// Check the claim carries the kind of object a consumer requires.
    ///
    /// A bridged claim is a mean. A consumer that needs a predictive law, a
    /// posterior, an identified set or a joint outcome law is refused: no
    /// bridge relaxes the rule that these are not interchangeable.
    ///
    /// # Errors
    /// `bridge_substitution_refused` naming the held and requested kinds.
    pub fn require_value_kind(&self, requested: ValueKind) -> Result<(), BridgeRefusal> {
        if self.value_kind == requested {
            Ok(())
        } else {
            Err(wrong_value_kind(requested, self.value_kind))
        }
    }

    /// An unscoped request for an interval, posterior or joint law of a bridged
    /// claim. The claim itself carries one mean and no uncertainty; the interval of
    /// the bridged value is built from the source claim's declared standard error and
    /// the mapping's own estimation uncertainty by the layer that holds the coverage
    /// records (`antecedent_io::bridge_interval`).
    ///
    /// # Errors
    /// Always `cell_not_licensed` with `scientific_bridge.interval_not_licensed`.
    pub fn request_interval(&self) -> Result<(), BridgeRefusal> {
        Err(interval_not_licensed())
    }

    /// Freshness of the bridged claim given the freshness of its source.
    ///
    /// A bridged claim rests on its source: a withdrawn or superseded source
    /// makes it inapplicable, and the receiver's snapshot is the source's.
    /// The bridged claim was valid for its snapshot exactly when its source was
    /// (`historically_valid` is the source's); retirement only changes applicability.
    #[must_use]
    pub fn freshness(&self, source: &ClaimFreshness) -> ClaimFreshness {
        let retired = source.withdrawn || source.superseded_by.is_some();
        ClaimFreshness {
            historically_valid: source.historically_valid,
            currently_applicable: if retired { Some(false) } else { source.currently_applicable },
            locally_ready: source.locally_ready,
            superseded_by: None,
            withdrawn: false,
            as_of_snapshot: source.as_of_snapshot,
        }
    }
}
