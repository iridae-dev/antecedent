//! Versioned `analysis_result.contract` section and independent consumer.
//!
//! The section is optional on the existing composite container. Old artifacts
//! remain readable through [`crate::decode_analysis_result_artifact`] but cannot
//! be silently promoted to verified programs. The consumer takes only bytes —
//! no originating `Study` or caller context.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::AcceptanceReport;
use antecedent_core::ContractIdentities;
use serde::{Deserialize, Serialize};

pub use crate::contract_reason::ContractReason;
pub use crate::contract_wire::{
    AssumptionSlotWire, CheckedAipwLoweringWire, CheckedAipwRowsWire, CheckedIvLoweringWire,
    IdentificationSlotWire, ObligationSectionWire, ReasoningSectionWire, SlotSectionWire,
    SupportSlotWire, UncertaintyComponentWire, UncertaintySlotWire, checked_aipw_rows_digest,
};

use crate::identity::{
    DataSnapshotIdentityWire, IdentificationIdentityWire, IdentificationProductWire,
    InferenceBindingWire, ObservationIdentityWire, ProgramIdentityWire, TargetIdentityWire,
};
use crate::identity::{ScoreReuseIdentityWire, TargetWeightsIdentityWire};
use crate::{
    AnalysisResultHeader, AnalysisResultWire, EncodedArtifact, ExecutionIdentityWire, IoError,
    from_cbor, query_wire::causal_query_from_wire,
};

/// Section id on the composite `analysis_result` artifact.
pub const CONTRACT_SECTION: &str = "analysis_result.contract";

/// Contract-section payload format. Bump only when the wire shape changes.
///
/// Format 2 adds the contract seal, the identification slot's weight basis,
/// and the version-2 identity payloads.
pub const CONTRACT_SECTION_FORMAT: u16 = 2;

/// Domain-separated identity bytes stored beside the rehashable target payload.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(deny_unknown_fields)]
pub struct ContractIdentitiesWire {
    /// Target digest.
    pub target: [u8; 32],
    /// Identification-premises digest.
    pub identification: [u8; 32],
    /// Identification-product digest, when prepared.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identification_product: Option<[u8; 32]>,
    /// Program digest.
    pub program: [u8; 32],
    /// Inference-binding digest.
    pub inference_binding: [u8; 32],
    /// Observation-contract digest.
    pub observation: [u8; 32],
    /// Data-snapshot digest.
    pub data_snapshot: [u8; 32],
    /// Execution digest, when the artifact records an execution.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<[u8; 32]>,
    /// Score-reuse digest of the score table the exporting handle holds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_reuse: Option<[u8; 32]>,
    /// Target-weights digest of a row-weight retarget. The target population
    /// references it, and the seal covers it like every other identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_weights: Option<[u8; 32]>,
    /// Identity of the checked AIPW row binding, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checked_aipw_rows: Option<[u8; 32]>,
}

/// Row weights of a retarget, carried with the identity they bind so a
/// consumer can re-derive it and confirm the population the answer is about.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct TargetWeightsSectionWire {
    /// Rehashable target-weights identity.
    pub identity: TargetWeightsIdentityWire,
    /// Weight values in score-table row order.
    pub values: Vec<f64>,
}

impl TryFrom<&ContractIdentities> for ContractIdentitiesWire {
    type Error = crate::IoError;

    /// A portable section advertises a program, so a structural inspection
    /// (which has none) cannot be encoded as one.
    fn try_from(identities: &ContractIdentities) -> Result<Self, Self::Error> {
        let program = identities.program.ok_or(crate::IoError::ManifestMismatch {
            message: "a structural inspection has no program identity to encode",
        })?;
        Ok(Self {
            target: *identities.target.as_bytes(),
            identification: *identities.identification.as_bytes(),
            identification_product: identities
                .identification_product
                .map(|digest| *digest.as_bytes()),
            program: *program.as_bytes(),
            inference_binding: *identities.inference_binding.as_bytes(),
            observation: *identities.observation.as_bytes(),
            data_snapshot: *identities.data_snapshot.as_bytes(),
            execution: None,
            score_reuse: None,
            target_weights: None,
            checked_aipw_rows: None,
        })
    }
}

/// Portable claim fields stored on the contract section.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct ClaimSectionWire {
    /// Claim digest.
    pub claim_id: [u8; 32],
    /// Claim kind.
    pub kind: String,
    /// Point value bits, when present.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub value_bits: Option<u64>,
    /// Execution digest, when an execution exists.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<[u8; 32]>,
    /// Identification domain status.
    pub identification_domain: String,
    /// Support domain status.
    pub support_domain: String,
    /// Evaluated domain status.
    pub evaluated_domain: String,
    /// Calibration slot computed from coverage records.
    pub calibration: CalibrationSlotWire,
    /// Caller-attested evidence (custom validators or an external estimate).
    #[serde(default)]
    pub attested: Vec<AttestedEvidenceWire>,
}

/// Calibration status of a reported interval.
///
/// CBOR is the historical text: `calibrated`, `scope_not_assessed`, or
/// `unavailable`.
#[derive(Clone, Copy, Debug, Serialize, Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum CalibrationStatus {
    /// The governing record attests the reported interval.
    Calibrated,
    /// A record matches the construction but not the measured scope.
    ScopeNotAssessed,
    /// No attesting record covers this interval.
    Unavailable,
}

impl CalibrationStatus {
    /// Historical wire text.
    #[must_use]
    pub const fn as_str(self) -> &'static str {
        match self {
            Self::Calibrated => "calibrated",
            Self::ScopeNotAssessed => "scope_not_assessed",
            Self::Unavailable => "unavailable",
        }
    }
}

impl std::fmt::Display for CalibrationStatus {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        formatter.write_str(self.as_str())
    }
}

impl PartialEq<&str> for CalibrationStatus {
    fn eq(&self, other: &&str) -> bool {
        self.as_str() == *other
    }
}

impl PartialEq<CalibrationStatus> for &str {
    fn eq(&self, other: &CalibrationStatus) -> bool {
        *self == other.as_str()
    }
}

impl From<&str> for CalibrationStatus {
    fn from(value: &str) -> Self {
        match value {
            "calibrated" => Self::Calibrated,
            "scope_not_assessed" => Self::ScopeNotAssessed,
            "unavailable" => Self::Unavailable,
            other => panic!("calibration status `{other}` is not a historical code"),
        }
    }
}

/// Calibration status of a reported interval, bound into the claim.
///
/// Computed by [`crate::calibration::calibration_slot`] from [`Self::basis`];
/// the consumer re-derives it from the same basis.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct CalibrationSlotWire {
    /// `calibrated` | `scope_not_assessed` | `unavailable`.
    pub status: CalibrationStatus,
    /// Governing coverage record id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub record_id: Option<String>,
    /// Reason code when not calibrated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Smallest row count the governing record measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_n: Option<u64>,
    /// Record dependence label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_dependence: Option<String>,
    /// Commit the governing record was measured at.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_sha: Option<String>,
    /// Largest row count the governing record measured.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_n_max: Option<u64>,
    /// Nominal level of the governing record.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub nominal: Option<f64>,
    /// Coverage the governing record observed (a boundary record's measured
    /// under-coverage is reported here, not hidden).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observed: Option<f64>,
    /// Match key and scope facts the slot was computed from.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub basis: Option<crate::calibration::CalibrationBasisWire>,
    /// Slots of further intervals the execution reported beside the primary
    /// one (an identified-set interval, a simultaneous band).
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub secondary: Vec<CalibrationSlotWire>,
}

/// Caller-attested custom-validator or external-estimate evidence.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AttestedEvidenceWire {
    /// Validator name, or the caller-supplied learner label.
    pub name: String,
    /// Evidence kind (`custom_validator` or `external_estimate`).
    pub kind: String,
    /// Whether the validator passed.
    pub passed: bool,
    /// Refuted ATE, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refuted_ate: Option<f64>,
    /// Comparison value, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub comparison: Option<f64>,
    /// Whether the result is informative.
    pub informative: bool,
    /// Failure condition, when reported.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub failure_condition: Option<String>,
    /// Always false — attested evidence is not re-verifiable.
    pub reverifiable: bool,
    /// Digest of the caller-supplied learner config, when this is an external estimate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub config_digest: Option<String>,
    /// Digest of the caller-supplied estimate payload, when this is an external estimate.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub payload_digest: Option<String>,
}

impl CalibrationSlotWire {
    /// Unavailable slot with a reason code.
    #[must_use]
    pub fn unavailable(reason: impl Into<String>) -> Self {
        Self {
            status: CalibrationStatus::Unavailable,
            record_id: None,
            reason: Some(reason.into()),
            scope_n: None,
            scope_dependence: None,
            calibration_sha: None,
            scope_n_max: None,
            nominal: None,
            observed: None,
            basis: None,
            secondary: Vec::new(),
        }
    }

    /// Slot governed by `record` with `status`.
    #[must_use]
    pub fn from_record(
        record: &crate::coverage_records_data::CoverageRecord,
        status: CalibrationStatus,
    ) -> Self {
        let (n_min, n_max) = crate::calibration::measured_range(record);
        Self {
            status,
            record_id: Some(record.id.into()),
            reason: None,
            scope_n: Some(n_min),
            scope_dependence: Some(record.dependence.into()),
            calibration_sha: Some(record.calibration_sha.into()),
            scope_n_max: Some(n_max),
            nominal: Some(record.nominal),
            observed: Some(record.observed),
            basis: None,
            secondary: Vec::new(),
        }
    }

    /// Attach a reason code.
    #[must_use]
    pub fn with_reason(mut self, reason: &str) -> Self {
        self.reason = Some(reason.into());
        self
    }

    /// Attach the match basis the slot was computed from.
    #[must_use]
    pub fn with_basis(mut self, basis: &crate::calibration::CalibrationBasisWire) -> Self {
        self.basis = Some(basis.clone());
        self
    }
}

/// Versioned contract companion for an `analysis_result` artifact.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
#[serde(deny_unknown_fields)]
pub struct AnalysisResultContractWire {
    /// Section format.
    pub format: u16,
    /// Advertised identities.
    pub identities: ContractIdentitiesWire,
    /// [`contract_seal`] over the identities, the four reasoning slots, and
    /// the audit fields (`graph_class`, `structure_source`, `identifier`,
    /// `estimator`). A claim's id covers the seal.
    pub seal: [u8; 32],
    /// Rehashable target payload; binds the body query to the advertised target.
    pub target: TargetIdentityWire,
    /// Four reasoning slots.
    pub reasoning: ReasoningSectionWire,
    /// Graph class.
    pub graph_class: String,
    /// How structure was supplied.
    pub structure_source: String,
    /// Identifier, when selected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identifier: Option<String>,
    /// Estimator, when selected.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub estimator: Option<String>,
    /// Execution claim, when the artifact records one.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub claim: Option<ClaimSectionWire>,
    /// Rehashable identification premises (graph + observation). Missing means attested-only.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identification: Option<IdentificationIdentityWire>,
    /// Rehashable identification product (expr arena + estimands).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub identification_product: Option<IdentificationProductWire>,
    /// Rehashable program payload.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub program: Option<ProgramIdentityWire>,
    /// Rehashable inference binding.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inference_binding: Option<InferenceBindingWire>,
    /// Rehashable observation contract.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub observation: Option<ObservationIdentityWire>,
    /// Rehashable data-snapshot identity (content digests, not raw rows).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub data_snapshot: Option<DataSnapshotIdentityWire>,
    /// Rehashable execution lineage, when a claim was produced.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub execution: Option<ExecutionIdentityWire>,
    /// Rehashable score-reuse identity.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub score_reuse: Option<ScoreReuseIdentityWire>,
    /// Row weights and their identity, for a row-weight retarget.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub target_weights: Option<TargetWeightsSectionWire>,
    /// Data-dependent row binding for checked AIPW, when retained.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub checked_aipw_rows: Option<CheckedAipwRowsWire>,
}

/// Independent consumption of a composite analysis-result artifact.
#[derive(Clone, Debug, PartialEq)]
pub struct AnalysisResultConsumption {
    /// Decoded header.
    pub header: AnalysisResultHeader,
    /// Decoded body. Always available when decode succeeds.
    pub body: AnalysisResultWire,
    /// Contract section, when present and well-formed enough to decode.
    pub contract: Option<AnalysisResultContractWire>,
    /// Acceptance without originating-process context.
    pub acceptance: AcceptanceReport,
    /// BLAKE3 of the received artifact bytes. Receipts for an artifact without
    /// a claim reference these bytes, never a synthesized claim id.
    pub artifact_digest: [u8; 32],
}

/// Decode the optional contract section. Missing is `Ok(None)`.
///
/// # Errors
///
/// Present but invalid CBOR.
pub fn decode_analysis_result_contract(
    artifact: &EncodedArtifact,
) -> Result<Option<AnalysisResultContractWire>, IoError> {
    let Some(section) = artifact.sections.iter().find(|section| section.id == CONTRACT_SECTION)
    else {
        return Ok(None);
    };
    Ok(Some(from_cbor(&section.data)?))
}

/// Validate a contract section in isolation (producer-side well-formedness).
///
/// Rehashes every stored payload against its advertised digest. Missing
/// payloads are allowed here so old writers can still encode; independent
/// consume treats them as unresolved.
///
/// # Errors
///
/// Unsupported format, malformed slots, or identity/payload disagreement.
pub fn validate_contract_section(contract: &AnalysisResultContractWire) -> Result<(), IoError> {
    if contract.format != CONTRACT_SECTION_FORMAT {
        return Err(IoError::Convert(format!(
            "unsupported `{CONTRACT_SECTION}` format {}",
            contract.format
        )));
    }
    let unresolved =
        verify::producer_encoding_unresolved(contract, verify::verify_stored_payloads(contract));
    if !unresolved.is_empty() {
        return Err(IoError::Convert(format!(
            "contract payloads do not rehash: {}",
            unresolved.iter().map(|reason| reason.as_str()).collect::<Vec<_>>().join(",")
        )));
    }
    causal_query_from_wire(&contract.target.query)?;
    Ok(())
}

mod consume;
pub mod cyclic;
mod replay;
mod verify;

pub use cyclic::{
    CYCLIC_CONTRACT_FORMAT, CYCLIC_CONTRACT_SECTION, CyclicContractPayloads, CyclicContractWire,
    CyclicDataSnapshotWire, CyclicIdentificationWire, CyclicInferenceWire, CyclicLayerDigests,
    CyclicObservationWire, CyclicProgramWire, CyclicRecoveryWire, CyclicSolutionWire,
    CyclicTargetWire, CyclicUnresolvedWire, cyclic_layer_digests, cyclic_request_identity,
    seal_cyclic_contract, verify_cyclic_contract_against_body,
};

pub(crate) use consume::validate_mixture_masses_within;
pub use consume::{
    ClaimDomainsWire, ClaimHostProjection, OVERLAP_LIMITED_DIAGNOSTIC, accept_claim,
    claim_domain_statuses, claim_domain_statuses_of_section, claim_domains,
    claim_domains_of_section, claim_kind, claim_kind_name, claim_kind_name_of_section,
    claim_kind_of_section, consume_analysis_result, consume_analysis_result_within, contract_seal,
    identification_not_claimed, project_claim_host, project_lossy_scalar, result_digest,
    support_empirical, validate_mixture_masses, weight_basis_name,
};
pub(crate) use verify::verify_contract_for_encoding;
pub use verify::{verify_contract_against_body, verify_contract_against_body_within};

#[cfg(test)]
mod tests;
