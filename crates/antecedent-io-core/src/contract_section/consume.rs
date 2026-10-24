//! Contract seal, claim acceptance and host projection.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::verify::verify_contract_against_body;

use std::sync::Arc;

use antecedent_contract::{
    AcceptanceReport, ConsumerProfile, HandoffReceipt, IDENTIFICATION_NOT_CLAIMED,
};
use antecedent_core::SemanticDigest;
use serde::{Deserialize, Serialize};

use super::{
    AnalysisResultConsumption, AnalysisResultContractWire, CONTRACT_SECTION,
    CONTRACT_SECTION_FORMAT, ClaimSectionWire, ContractIdentitiesWire,
};
use crate::contract_reason::ContractReason;
use crate::contract_wire::{
    IdentificationSlotWire, ReasoningSectionWire, SlotSectionWire, SupportSlotWire,
};
use crate::identity::{ClaimIdentityWire, IdentificationProductWire, claim_digest, payload_digest};
use crate::{
    AnalysisResultHeader, AnalysisResultWire, IoError, RefutationReportWire,
    StructuralWeightBasisWire, to_cbor,
};

/// Acceptance and handoff operation stored as the historical `snake_case` text.
///
/// [`AcceptanceReport`] keeps `Arc<str>` because that type lives in
/// `antecedent-contract`. These variants are the only texts this crate writes.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub(crate) enum AcceptanceOp {
    /// Store the artifact bytes.
    Store,
    /// Forward the artifact bytes.
    Forward,
    /// Read the result body.
    ReadBody,
    /// Verify the contract section.
    VerifyContract,
    /// Inspect the bound claim.
    InspectClaim,
    /// Retain the artifact bytes on a restricted handoff.
    Bytes,
    /// Operation dropped when a claim is not accepted.
    AcceptAsClaim,
}

impl AcceptanceOp {
    pub(crate) const fn as_str(self) -> &'static str {
        match self {
            Self::Store => "store",
            Self::Forward => "forward",
            Self::ReadBody => "read_body",
            Self::VerifyContract => "verify_contract",
            Self::InspectClaim => "inspect_claim",
            Self::Bytes => "bytes",
            Self::AcceptAsClaim => "accept_as_claim",
        }
    }

    fn label(self) -> Arc<str> {
        Arc::from(self.as_str())
    }
}

#[derive(Serialize)]
pub(in crate::contract_section) struct SealWire<'a> {
    format: u16,
    identities: &'a ContractIdentitiesWire,
    reasoning: &'a ReasoningSectionWire,
    graph_class: &'a str,
    structure_source: &'a str,
    identifier: Option<&'a str>,
    estimator: Option<&'a str>,
}

/// Seal binding every advertised identity, the four reasoning slots, and the
/// section's audit fields.
///
/// Recomputed on consume. A claim id covers the seal, so no reported slot,
/// domain, graph class, identifier, or estimator is outside a digest.
///
/// # Errors
///
/// CBOR encode failure.
pub fn contract_seal(
    identities: &ContractIdentitiesWire,
    reasoning: &ReasoningSectionWire,
    graph_class: &str,
    structure_source: &str,
    identifier: Option<&str>,
    estimator: Option<&str>,
) -> Result<[u8; 32], IoError> {
    let bytes = to_cbor(&SealWire {
        format: CONTRACT_SECTION_FORMAT,
        identities,
        reasoning,
        graph_class,
        structure_source,
        identifier,
        estimator,
    })?;
    Ok(payload_digest("analysis_result.contract.seal", &bytes))
}

pub(in crate::contract_section) fn contract_seal_of(
    contract: &AnalysisResultContractWire,
) -> Result<[u8; 32], IoError> {
    contract_seal(
        &contract.identities,
        &contract.reasoning,
        &contract.graph_class,
        &contract.structure_source,
        contract.identifier.as_deref(),
        contract.estimator.as_deref(),
    )
}

/// Digest of an executed result body, bound into the claim id.
///
/// Covers every body field: scalar, standard error, assumptions, diagnostics,
/// refutations, response, posterior draws, mediation grid, structural atoms,
/// and the identification certificate. The native execution observation is a
/// record of how one run did its work (a fit, or a reuse of a frozen fit), not
/// part of what the result claims, so the claim identity leaves it out: a
/// retained study that re-estimates reproduces its own claim.
///
/// # Errors
///
/// CBOR encode failure.
pub fn result_digest(body: &AnalysisResultWire) -> Result<[u8; 32], IoError> {
    if body.native_execution.is_none() {
        return crate::identity::payload_digest_of("analysis_result.body", body);
    }
    let mut claimed = body.clone();
    claimed.native_execution = None;
    crate::identity::payload_digest_of("analysis_result.body", &claimed)
}

/// Claim kind implied by a result body and its identification slot.
///
/// The single rule shared by the producer and the independent consumer.
#[must_use]
pub fn claim_kind_name(
    body: &AnalysisResultWire,
    identification: Option<&IdentificationSlotWire>,
) -> &'static str {
    if body.policy_value.is_some() {
        return "policy_value";
    }
    if body.response.is_some() {
        return "response";
    }
    // A description of the observed law is associational whatever its identification slot
    // says: no slot turns an association section into a causal claim.
    if body.association.is_some() {
        return "associational";
    }
    // An identified set is never a point, whatever status the identification slot carries.
    if body.iv_bounds.is_some() {
        return "bounds";
    }
    let Some(slot) = identification else {
        return "point";
    };
    match slot.status.as_str() {
        "partially_identified" => return "bounds",
        "nonparametrically_identified"
        | "identified_under_parametric_restrictions"
        | "identified_under_prior_restrictions"
        | "graph_dependent" => {}
        // `not_identified`, `not_certified`, `proven_non_transportable`, or any spelling this reader
        // does not know: never a point.
        _ => return "incomplete",
    }
    // Any mass short of full identification, or a weighted mixture, is not a point.
    if slot.identified_mass < 1.0 || slot.weight_basis.is_some() {
        if body.structural_response.as_ref().is_some_and(|mixture| mixture.identified_set.is_some())
        {
            return "bounds";
        }
        return "mixture";
    }
    "point"
}

/// Whether an identification section states that identification is not claimed.
///
/// A noncausal claim has nothing to identify, so its identification slot is
/// unavailable for the stable reason [`IDENTIFICATION_NOT_CLAIMED`]. Any other
/// unavailable reason, or a present slot, is an ordinary causal claim.
#[must_use]
pub fn identification_not_claimed(section: &SlotSectionWire<IdentificationSlotWire>) -> bool {
    section.value.is_none() && section.unavailable.as_deref() == Some(IDENTIFICATION_NOT_CLAIMED)
}

/// [`claim_kind_name`] over the whole identification section.
///
/// A section that states identification is not claimed implies an
/// `associational` claim whatever the body carries; every other section reads
/// as [`claim_kind_name`] does. The producer and the independent consumer both
/// call this one rule.
#[must_use]
pub fn claim_kind_name_of_section(
    body: &AnalysisResultWire,
    identification: &SlotSectionWire<IdentificationSlotWire>,
) -> &'static str {
    if identification_not_claimed(identification) {
        return "associational";
    }
    claim_kind_name(body, identification.value.as_ref())
}

/// Refuter ids whose outcome speaks to empirical support (overlap / positivity).
pub(in crate::contract_section) fn is_support_refuter(refuter: &str) -> bool {
    refuter.starts_with("overlap.") || refuter == "positivity"
}

/// Empirical support label from executed refutations.
///
/// `None` when no support check ran; `supported` when every support check
/// passed; `failed:<refuter>` naming the first failed check otherwise. Other
/// refuters (placebo, sensitivity, ...) do not speak to support.
#[must_use]
pub fn support_empirical(refutations: &[RefutationReportWire]) -> Option<String> {
    let mut checks = refutations.iter().filter(|report| is_support_refuter(&report.refuter));
    let first = checks.next()?;
    std::iter::once(first).chain(checks).find(|report| !report.passed).map_or_else(
        || Some("supported".into()),
        |failed| Some(format!("failed:{}", failed.refuter)),
    )
}

/// Identification / support / evaluated domain statuses of a claim.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct ClaimDomainsWire {
    /// Identification domain status.
    pub identification: String,
    /// Support domain status.
    pub support: String,
    /// Evaluated domain status.
    pub evaluated: String,
}

/// Claim domains implied by the reasoning slots.
///
/// Identified (and evaluated) only for a full-mass, nonparametric or
/// parametric identification. Supported only when the matrix cell is
/// licensed and every executed support check passed; a failed check is
/// `contradicted`; a refused or not-applicable cell is `outside_scope`.
#[must_use]
pub fn claim_domains(
    support: Option<&SupportSlotWire>,
    identification: Option<&IdentificationSlotWire>,
) -> ClaimDomainsWire {
    let identified = identification.is_some_and(|slot| {
        slot.unidentified_mass == 0.0
            && slot.unevaluable_mass == 0.0
            && slot.incomplete_search_mass == 0.0
            && matches!(
                slot.status.as_str(),
                "nonparametrically_identified" | "identified_under_parametric_restrictions"
            )
    });
    let support_domain = match support {
        Some(slot) if matches!(slot.matrix_status.as_str(), "refused" | "not_applicable") => {
            "outside_scope"
        }
        Some(slot) if slot.empirical.starts_with("failed:") => "contradicted",
        Some(slot) if slot.matrix_status == "licensed" && slot.empirical == "supported" => {
            "supported"
        }
        _ => "unknown",
    };
    ClaimDomainsWire {
        identification: if identified { "identified" } else { "unknown" }.into(),
        support: support_domain.into(),
        evaluated: if identified { "evaluated" } else { "unknown" }.into(),
    }
}

/// [`claim_domains`] over the whole identification section.
///
/// A noncausal claim reports identification as `not_claimed`, never as a
/// failed or unknown identification, and is evaluated: the association it
/// states was computed from the data the result carries. Its support domain is
/// read exactly as a causal claim's.
#[must_use]
pub fn claim_domains_of_section(
    support: Option<&SupportSlotWire>,
    identification: &SlotSectionWire<IdentificationSlotWire>,
) -> ClaimDomainsWire {
    let mut domains = claim_domains(support, identification.value.as_ref());
    if identification_not_claimed(identification) {
        domains.identification = "not_claimed".into();
        domains.evaluated = "evaluated".into();
    }
    domains
}

/// Wire spelling of a structural weight basis.
#[must_use]
pub const fn weight_basis_name(basis: StructuralWeightBasisWire) -> &'static str {
    match basis {
        StructuralWeightBasisWire::PosteriorProbability => "posterior_probability",
        StructuralWeightBasisWire::CompletionEnumeration => "completion_enumeration",
        StructuralWeightBasisWire::CallerSuppliedClassPrior => "caller_supplied_class_prior",
    }
}

/// Mass-conservation check for structural mixtures.
///
/// # Errors
///
/// Any mass outside `[0, 1]`, a non-finite value, or a sum that is not 1
/// within `1e-12`.
pub fn validate_mixture_masses(
    identified: f64,
    unidentified: f64,
    unevaluable: f64,
    incomplete_search: f64,
) -> Result<(), IoError> {
    validate_mixture_masses_within(
        [identified, unidentified, unevaluable, incomplete_search],
        1e-12,
    )
}

/// [`validate_mixture_masses`] with an explicit sum tolerance.
pub(crate) fn validate_mixture_masses_within(
    masses: [f64; 4],
    tolerance: f64,
) -> Result<(), IoError> {
    if masses.iter().any(|mass| !mass.is_finite() || *mass < 0.0 || *mass > 1.0) {
        return Err(IoError::Convert("mixture masses must be finite and inside [0, 1]".into()));
    }
    let sum: f64 = masses.iter().sum();
    if (sum - 1.0).abs() > tolerance {
        return Err(IoError::Convert(format!("mixture masses must sum to 1 (got {sum})")));
    }
    Ok(())
}

pub(in crate::contract_section) fn require_nested_digest(
    unresolved: &mut Vec<ContractReason>,
    label: ContractReason,
    nested: Option<[u8; 32]>,
    advertised: Option<&[u8; 32]>,
) {
    let agrees = match (nested, advertised) {
        (Some(got), Some(want)) => got == *want,
        (None, None) => true,
        (None, Some(_)) | (Some(_), None) => false,
    };
    if !agrees {
        unresolved.push(label);
    }
}

pub(in crate::contract_section) fn claim_id_matches(
    contract: &AnalysisResultContractWire,
    claim: &ClaimSectionWire,
    body: &AnalysisResultWire,
) -> bool {
    let Ok(result) = result_digest(body) else {
        return false;
    };
    matches!(
        claim_digest(&ClaimIdentityWire::new(contract.seal, claim, result)),
        Ok(id) if id.as_bytes() == &claim.claim_id
    )
}

pub(in crate::contract_section) fn claim_value_matches_body(
    claim: &ClaimSectionWire,
    body: &AnalysisResultWire,
) -> bool {
    match (claim.value_bits, body.estimate) {
        (Some(bits), Some(estimate)) if bits == estimate.to_bits() => true,
        (None, None) => true,
        _ => false,
    }
}

pub(in crate::contract_section) fn identification_product_matches_body(
    product: &IdentificationProductWire,
    body: &AnalysisResultWire,
) -> bool {
    product.status == body.identification.status
        && product.derivation_rules.iter().map(String::as_str).eq(body
            .identification
            .derivation
            .iter()
            .map(|step| step.rule.as_str()))
        && product.required_assumptions == body.identification.required_assumptions
        && cbor_eq(&product.estimands, &body.identification.estimands)
        && cbor_eq(&product.arena, &body.identification.arena)
}

pub(in crate::contract_section) fn cbor_eq<T: Serialize>(left: &T, right: &T) -> bool {
    match (crate::to_cbor(left), crate::to_cbor(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

pub(in crate::contract_section) fn require_present<T>(
    unresolved: &mut Vec<ContractReason>,
    label: ContractReason,
    payload: Option<&T>,
) {
    if payload.is_none() {
        unresolved.push(label);
    }
}

pub(in crate::contract_section) fn require_slot<T>(
    unresolved: &mut Vec<ContractReason>,
    reason: ContractReason,
    slot: &SlotSectionWire<T>,
) {
    if validate_slot(reason.as_str(), slot).is_err() {
        unresolved.push(reason);
    }
}

pub(in crate::contract_section) fn require_payload_digest<T>(
    unresolved: &mut Vec<ContractReason>,
    label: ContractReason,
    payload: Option<&T>,
    advertised: Option<&[u8; 32]>,
    digest: impl FnOnce(&T) -> Result<SemanticDigest, IoError>,
) {
    let Some(payload) = payload else {
        return;
    };
    match (digest(payload), advertised) {
        (Ok(got), Some(want)) if got.as_bytes() == want => {}
        _ => unresolved.push(label),
    }
}

/// Consume analysis-result bytes without caller context.
///
/// Decode failures are errors. Missing or unverifiable contracts are
/// successful consumptions whose [`AcceptanceReport`] refuses verification.
///
/// # Errors
///
/// Wrong artifact kind, missing header/body, or invalid CBOR.
pub fn consume_analysis_result(bytes: &[u8]) -> Result<AnalysisResultConsumption, IoError> {
    let (artifact, header, body) = crate::decode_analysis_result_artifact(bytes)?;
    let artifact_digest = payload_digest("analysis_result.bytes", bytes);
    let contract = match super::decode_analysis_result_contract(&artifact) {
        Ok(contract) => contract,
        Err(err) => {
            return Ok(AnalysisResultConsumption {
                header,
                body,
                contract: None,
                acceptance: AcceptanceReport::new(
                    true,
                    false,
                    [Arc::from(CONTRACT_SECTION)],
                    storage_ops(),
                    Some(Arc::from(format!("invalid_contract_section:{err}"))),
                ),
                artifact_digest,
            });
        }
    };
    let acceptance = accept_contract(&header, &body, contract.as_ref());
    Ok(AnalysisResultConsumption { header, body, contract, acceptance, artifact_digest })
}

pub(in crate::contract_section) fn accept_contract(
    header: &AnalysisResultHeader,
    body: &AnalysisResultWire,
    contract: Option<&AnalysisResultContractWire>,
) -> AcceptanceReport {
    let Some(contract) = contract else {
        return AcceptanceReport::new(
            true,
            false,
            [Arc::from(CONTRACT_SECTION)],
            storage_ops(),
            Some(Arc::from("missing_contract_section")),
        );
    };
    if contract.format != CONTRACT_SECTION_FORMAT {
        return AcceptanceReport::new(
            false,
            false,
            [Arc::from("contract_format")],
            storage_ops(),
            Some(Arc::from("unknown_contract_format")),
        );
    }
    let unresolved = verify_contract_against_body(header, body, contract);
    let verified = unresolved.is_empty();
    let claim_present = contract.claim.is_some();
    let operations: Arc<[Arc<str>]> = match (verified, claim_present) {
        (true, true) => Arc::from([
            AcceptanceOp::Store.label(),
            AcceptanceOp::Forward.label(),
            AcceptanceOp::ReadBody.label(),
            AcceptanceOp::VerifyContract.label(),
            AcceptanceOp::InspectClaim.label(),
        ]),
        (true, false) => Arc::from([
            AcceptanceOp::Store.label(),
            AcceptanceOp::Forward.label(),
            AcceptanceOp::ReadBody.label(),
            AcceptanceOp::VerifyContract.label(),
        ]),
        (false, _) => storage_ops(),
    };
    let restriction = if !verified {
        Some(Arc::from("unverified_contract_references"))
    } else if !claim_present {
        Some(Arc::from("verified_program_without_claim"))
    } else {
        None
    };
    AcceptanceReport::new(true, verified, unresolved, operations, restriction)
        .with_claim(claim_present)
}

pub(in crate::contract_section) fn validate_slot<T>(
    name: &str,
    slot: &SlotSectionWire<T>,
) -> Result<(), IoError> {
    match (&slot.value, &slot.unavailable) {
        (Some(_), None) | (None, Some(_)) => Ok(()),
        (Some(_), Some(_)) => {
            Err(IoError::Convert(format!("{name} slot cannot be both available and unavailable")))
        }
        (None, None) => Err(IoError::Convert(format!(
            "{name} slot must be available or explicitly unavailable"
        ))),
    }
}

pub(in crate::contract_section) fn storage_ops() -> Arc<[Arc<str>]> {
    Arc::from([
        AcceptanceOp::Store.label(),
        AcceptanceOp::Forward.label(),
        AcceptanceOp::ReadBody.label(),
    ])
}

/// Accept / verify a claim from bytes under a consumer profile.
///
/// Reuses [`consume_analysis_result`] / [`verify_contract_against_body`]. Unknown
/// required features refuse claim acceptance; storage/forwarding remains allowed.
///
/// # Errors
///
/// Wrong artifact kind, missing header/body, or invalid CBOR.
pub fn accept_claim(
    bytes: &[u8],
    profile: &ConsumerProfile,
) -> Result<(AnalysisResultConsumption, HandoffReceipt), IoError> {
    let mut consumed = consume_analysis_result(bytes)?;
    let claim_id = receipt_input(&consumed);
    if !profile.understands(AcceptanceOp::VerifyContract.as_str())
        || !profile.understands(AcceptanceOp::InspectClaim.as_str())
    {
        consumed.acceptance = AcceptanceReport::opaque_storage();
        return Ok((consumed, HandoffReceipt::opaque_forward(claim_id, profile.id.clone())));
    }
    if !consumed.acceptance.recognized {
        return Ok((consumed, HandoffReceipt::opaque_forward(claim_id, profile.id.clone())));
    }
    if let Some(missing) = unknown_required_features(&consumed, profile) {
        consumed.acceptance = AcceptanceReport::new(
            false,
            false,
            [Arc::from(missing)],
            storage_ops(),
            Some(Arc::from("unknown_required_feature")),
        );
        return Ok((consumed, HandoffReceipt::opaque_forward(claim_id, profile.id.clone())));
    }
    let handoff = if consumed.acceptance.accepts_as_claim() {
        HandoffReceipt::lossless(claim_id, profile.id.clone())
    } else {
        let mut omitted: Vec<Arc<str>> = Vec::new();
        if !consumed.acceptance.unresolved.is_empty() {
            omitted.push(Arc::from("verified_references"));
        }
        if !consumed.acceptance.claim_present {
            omitted.push(Arc::from("claim"));
        }
        HandoffReceipt::new(
            claim_id,
            None,
            profile.id.clone(),
            "restricted_accept",
            [AcceptanceOp::Bytes.label(), AcceptanceOp::ReadBody.label()],
            omitted,
            consumed.acceptance.unresolved.clone(),
            [AcceptanceOp::AcceptAsClaim.label()],
        )
    };
    let allowed: Vec<Arc<str>> = consumed
        .acceptance
        .supported_operations
        .iter()
        .filter(|op| profile.understands(op))
        .cloned()
        .collect();
    consumed.acceptance.supported_operations = Arc::from(allowed);
    Ok((consumed, handoff))
}

pub(in crate::contract_section) fn unknown_required_features(
    consumed: &AnalysisResultConsumption,
    profile: &ConsumerProfile,
) -> Option<String> {
    if !profile.understands("contract.v1") {
        return Some("contract.v1".into());
    }
    let contract = consumed.contract.as_ref()?;
    if let Some(claim) = &contract.claim {
        let feature = format!("claim.{}", claim.kind);
        if !profile.understands(&feature) {
            return Some(feature);
        }
    }
    if !profile.understands("reasoning.four_slots") {
        return Some("reasoning.four_slots".into());
    }
    None
}

/// Receipt input: the claim id when the artifact carries a claim, otherwise
/// the digest of the received bytes. A zero id is never synthesized.
pub(in crate::contract_section) fn receipt_input(
    consumed: &AnalysisResultConsumption,
) -> SemanticDigest {
    consumed
        .contract
        .as_ref()
        .and_then(|contract| contract.claim.as_ref())
        .map_or(SemanticDigest::from_bytes(consumed.artifact_digest), |claim| {
            SemanticDigest::from_bytes(claim.claim_id)
        })
}

/// JSON-compatible host projection. `null`, `0`, `unknown`, and `unsupported` stay distinct.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ClaimHostProjection {
    /// Claim digest hex, when a claim section exists.
    pub claim_id: Option<String>,
    /// Claim kind, when present.
    pub kind: Option<String>,
    /// Point bits as a number, explicit absence as JSON `null`, else a distinct token.
    pub value: serde_json::Value,
    /// Identification domain status, or JSON `null` when no claim exists.
    pub identification_domain: serde_json::Value,
    /// Support domain status.
    pub support_domain: serde_json::Value,
    /// Evaluated domain status.
    pub evaluated_domain: serde_json::Value,
    /// Whether required semantics were recognized.
    pub recognized: bool,
    /// Whether the receiver accepts a usable causal claim.
    pub accepts_as_claim: bool,
}

/// Project a consumed artifact for host JSON. Inspection does not fetch or run callbacks.
#[must_use]
pub fn project_claim_host(consumed: &AnalysisResultConsumption) -> ClaimHostProjection {
    let Some(contract) = consumed.contract.as_ref() else {
        return unsupported_host(consumed, false);
    };
    let Some(claim) = contract.claim.as_ref() else {
        return unsupported_host(consumed, consumed.acceptance.accepts_as_claim());
    };
    let value = match (claim.kind.as_str(), claim.value_bits) {
        ("point", Some(bits)) => serde_json::Value::from(f64::from_bits(bits)),
        ("point", None) => serde_json::Value::Null,
        ("bounds" | "incomplete", _) => serde_json::Value::String("unbounded".into()),
        ("refusal", _) => serde_json::Value::String("unsupported".into()),
        (_, _) => serde_json::Value::String("unknown".into()),
    };
    // Domains are re-derived from rehashed reasoning slots; when references do
    // not verify, the stored strings are the sender's say-so and are not shown.
    let domain = |stored: &str| {
        if consumed.acceptance.verified_references {
            serde_json::Value::String(stored.to_owned())
        } else {
            serde_json::Value::Null
        }
    };
    ClaimHostProjection {
        claim_id: Some(antecedent_core::to_hex(&claim.claim_id)),
        kind: Some(claim.kind.clone()),
        value,
        identification_domain: domain(&claim.identification_domain),
        support_domain: domain(&claim.support_domain),
        evaluated_domain: domain(&claim.evaluated_domain),
        recognized: consumed.acceptance.recognized,
        accepts_as_claim: consumed.acceptance.accepts_as_claim(),
    }
}

pub(in crate::contract_section) fn unsupported_host(
    consumed: &AnalysisResultConsumption,
    accepts: bool,
) -> ClaimHostProjection {
    ClaimHostProjection {
        claim_id: None,
        kind: None,
        value: serde_json::Value::String("unsupported".into()),
        identification_domain: serde_json::Value::Null,
        support_domain: serde_json::Value::Null,
        evaluated_domain: serde_json::Value::Null,
        recognized: consumed.acceptance.recognized,
        accepts_as_claim: accepts,
    }
}

/// Lossy scalar view plus a handoff that cannot impersonate the complete claim.
#[must_use]
pub fn project_lossy_scalar(
    consumed: &AnalysisResultConsumption,
) -> (ClaimHostProjection, HandoffReceipt) {
    let mut view = project_claim_host(consumed);
    view.identification_domain = serde_json::Value::Null;
    view.support_domain = serde_json::Value::Null;
    view.evaluated_domain = serde_json::Value::Null;
    view.accepts_as_claim = false;
    let claim_id = receipt_input(consumed);
    (view, HandoffReceipt::lossy_scalar(claim_id, "restricted"))
}
