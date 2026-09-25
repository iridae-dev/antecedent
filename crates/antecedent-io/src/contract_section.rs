//! Versioned `analysis_result.contract` section and independent consumer.
//!
//! The section is optional on the existing composite container. Old artifacts
//! remain readable through [`crate::decode_analysis_result_artifact`] but cannot
//! be silently promoted to verified programs. The consumer takes only bytes —
//! no originating `Study` or caller context.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AcceptanceReport, ConsumerProfile, ContractIdentities, HandoffReceipt, IdentityDomain,
    SemanticDigest,
};
use serde::{Deserialize, Serialize};

use crate::identity::{
    ClaimIdentityWire, DataSnapshotIdentityWire, IdentificationIdentityWire,
    IdentificationProductWire, InferenceBindingWire, ObservationIdentityWire, ProgramIdentityWire,
    TargetIdentityWire, claim_digest, data_snapshot_digest, digest_wire, execution_digest,
    identification_digest, identification_product_digest_wire, inference_binding_digest,
    payload_digest, program_digest,
};
use crate::identity::{
    PayloadDigestWire, ScoreReuseIdentityWire, TargetWeightsIdentityWire, score_reuse_digest,
    target_weights_digest,
};
use crate::query_wire::TargetPopulationWire;
use crate::{
    AnalysisResultHeader, AnalysisResultWire, CausalQueryWire, EncodedArtifact,
    ExecutionIdentityWire, IoError, RefutationReportWire, StructuralWeightBasisWire, from_cbor,
    query_wire::causal_query_from_wire, to_cbor,
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

/// Semantic choices retained by the checked AIPW lowering.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckedAipwLoweringWire {
    /// Lowering payload format.
    pub format: u16,
    /// Selected causal functional expression id.
    pub functional: u32,
    /// Treatment variable id.
    pub treatment: u32,
    /// Outcome variable id.
    pub outcome: u32,
    /// Adjustment variable ids in selected-estimand order.
    pub adjustment: Vec<u32>,
    /// Target population name (`all_observed` for this checked route).
    pub population: String,
    /// Licensed procedure (`cross_fitted_logistic_ols`).
    pub procedure: String,
    /// Cross-fitting folds.
    pub folds: u64,
    /// Analytic standard-error kind.
    pub se_kind: String,
    /// Lag for lagged analytic standard-error methods, when applicable.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub se_lag: Option<u64>,
    /// Bootstrap replicate count (zero means disabled).
    pub bootstrap_replicates: u32,
}

/// Data-dependent complete-case rows bound to a checked AIPW lowering.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckedAipwRowsWire {
    /// Row-binding payload format.
    pub format: u16,
    /// Source row ids retained by the checked estimator, in source order.
    pub rows: Vec<u32>,
    /// Data snapshot identity these rows came from.
    pub data_snapshot: [u8; 32],
}

/// Semantic roles, procedure, weak-instrument policy, and complete-case scope
/// retained by a checked prepared IV operation.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CheckedIvLoweringWire {
    /// Lowering payload format.
    pub format: u16,
    /// Selected causal Wald-ratio functional expression id.
    pub functional: u32,
    /// Endogenous treatment variable id.
    pub treatment: u32,
    /// Outcome variable id.
    pub outcome: u32,
    /// Single instrument variable id.
    pub instrument: u32,
    /// Exogenous adjustment variable ids in selected-estimand order.
    pub adjustment: Vec<u32>,
    /// Queried active treatment value bits.
    pub active_bits: u64,
    /// Queried control treatment value bits.
    pub control_bits: u64,
    /// Instrument's active arm value bits.
    pub instrument_active_bits: u64,
    /// Instrument's control arm value bits.
    pub instrument_control_bits: u64,
    /// `wald` or `two_stage_least_squares`.
    pub procedure: String,
    /// Analytic standard-error method selected by the prepared operation.
    pub se_kind: String,
    /// Weak-instrument confidence-set route, including an explicit withheld state.
    pub weak_instrument_uncertainty: String,
    /// Number of complete-case observations in the prepared design.
    pub complete_case_rows: u64,
}

/// Hash a checked AIPW row binding for inclusion in contract identities.
///
/// # Errors
///
/// CBOR encoding failure.
pub fn checked_aipw_rows_digest(rows: &CheckedAipwRowsWire) -> Result<[u8; 32], IoError> {
    Ok(payload_digest("analysis_result.checked_aipw_rows", &to_cbor(rows)?))
}

/// Optional slot: exactly one of `value` or `unavailable` is required.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct SlotSectionWire<T> {
    /// Present value.
    pub value: Option<T>,
    /// Stable unavailable reason.
    pub unavailable: Option<String>,
}

/// Compact identification slot.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct IdentificationSlotWire {
    /// Identification status.
    pub status: String,
    /// Identified mass.
    pub identified_mass: f64,
    /// Unidentified mass.
    pub unidentified_mass: f64,
    /// Unevaluable mass.
    pub unevaluable_mass: f64,
    /// Incomplete-search mass.
    pub incomplete_search_mass: f64,
    /// Whether reported mass covers the full class.
    pub full_mass_scope: bool,
    /// Search was capped before a determination.
    pub search_capped: bool,
    /// What the masses measure (`posterior_probability`,
    /// `completion_enumeration`, `caller_supplied_class_prior`). Enumeration
    /// weights are not probabilities; absent for a single identified atom.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub weight_basis: Option<String>,
}

/// Compact support slot.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct SupportSlotWire {
    /// Matrix evidence status.
    pub matrix_status: String,
    /// Matrix coordinate, when classified.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub matrix_coordinate: Option<String>,
    /// Empirical support, or `unavailable:<reason>`.
    pub empirical: String,
}

/// One uncertainty component.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct UncertaintyComponentWire {
    /// Source name.
    pub source: String,
    /// Reported target.
    pub target: String,
    /// Omitted / unresolved.
    pub omitted: bool,
}

/// Compact uncertainty slot.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct UncertaintySlotWire {
    /// Declared components.
    pub components: Vec<UncertaintyComponentWire>,
}

/// Compact obligation record (full assumption encodings stay on the body).
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct ObligationSectionWire {
    /// Stable obligation id.
    pub id: String,
    /// Scope name.
    pub scope: String,
    /// Kind name.
    pub kind: String,
    /// Assumption status.
    pub status: String,
}

/// Compact assumption slot.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct AssumptionSlotWire {
    /// Scoped obligations.
    pub obligations: Vec<ObligationSectionWire>,
}

/// Four reasoning slots.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct ReasoningSectionWire {
    /// Identification.
    pub identification: SlotSectionWire<IdentificationSlotWire>,
    /// Support.
    pub support: SlotSectionWire<SupportSlotWire>,
    /// Uncertainty.
    pub uncertainty: SlotSectionWire<UncertaintySlotWire>,
    /// Assumptions.
    pub assumptions: SlotSectionWire<AssumptionSlotWire>,
}

/// Portable claim fields stored on the contract section.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
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

/// Calibration status of a reported interval, bound into the claim.
///
/// Computed by [`crate::calibration::calibration_slot`] from [`Self::basis`];
/// the consumer re-derives it from the same basis.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct CalibrationSlotWire {
    /// `calibrated` | `scope_not_assessed` | `unavailable`.
    pub status: String,
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
            status: "unavailable".into(),
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
        status: &str,
    ) -> Self {
        let (n_min, n_max) = crate::calibration::measured_range(record);
        Self {
            status: status.into(),
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
    let unresolved = producer_encoding_unresolved(contract, verify_stored_payloads(contract));
    if !unresolved.is_empty() {
        return Err(IoError::Convert(format!(
            "contract payloads do not rehash: {}",
            unresolved.join(",")
        )));
    }
    causal_query_from_wire(&contract.target.query)?;
    Ok(())
}

/// Shared producer/consumer check: stored payloads, body query, and claim.
///
/// Missing rehashable payloads are unresolved — attested digests are not enough.
#[must_use]
pub fn verify_contract_against_body(
    header: &AnalysisResultHeader,
    body: &AnalysisResultWire,
    contract: &AnalysisResultContractWire,
) -> Vec<Arc<str>> {
    let mut unresolved = verify_stored_payloads(contract);
    if contract.program.as_ref().is_some_and(|program| {
        [
            program.functional_program.is_some(),
            program.checked_aipw_lowering.is_some(),
            program.checked_frontdoor_lowering.is_some(),
            program.checked_iv_lowering.is_some(),
            program.checked_linear_adjustment_lowering.is_some(),
            program.checked_functional_response_grid.is_some(),
            program.checked_nested_counterfactual.is_some(),
        ]
        .into_iter()
        .filter(|present| *present)
        .count()
            > 1
    }) {
        unresolved.push(Arc::from("program.checked_payload_conflict"));
    }
    if let Some(program) = contract.program.as_ref() {
        let estimator_matches = |expected: &[&str]| {
            contract
                .estimator
                .as_deref()
                .into_iter()
                .chain(program.commitments.resolved_estimator.as_deref())
                .all(|name| expected.contains(&name))
                && (contract.estimator.is_some()
                    || program.commitments.resolved_estimator.is_some())
        };
        let average_effect = matches!(contract.target.query, CausalQueryWire::AverageEffect { .. });
        if program.functional_program.is_some()
            && !((estimator_matches(&["functional.effect"])
                && matches!(
                    contract.target.query,
                    CausalQueryWire::AverageEffect { .. } | CausalQueryWire::PathSpecific(_)
                ))
                || (estimator_matches(&["functional.distribution"])
                    && matches!(contract.target.query, CausalQueryWire::Distribution(_))))
        {
            unresolved.push(Arc::from("program.functional_program.target"));
        }
        for (present, matches, reason) in [
            (
                program.checked_aipw_lowering.is_some(),
                average_effect && estimator_matches(&["aipw"]),
                "program.checked_aipw_lowering.target",
            ),
            (
                program.checked_frontdoor_lowering.is_some(),
                average_effect && estimator_matches(&["frontdoor.linear_two_stage"]),
                "program.checked_frontdoor_lowering.target",
            ),
            (
                program.checked_iv_lowering.is_some(),
                average_effect && estimator_matches(&["iv.wald", "iv.2sls"]),
                "program.checked_iv_lowering.target",
            ),
            (
                program.checked_linear_adjustment_lowering.is_some(),
                average_effect && estimator_matches(&["linear.adjustment.ate"]),
                "program.checked_linear_adjustment_lowering.target",
            ),
            (
                program.checked_functional_response_grid.is_some(),
                is_response_functional_effect_contract(contract),
                "program.checked_functional_response_grid.target",
            ),
            (
                program.checked_nested_counterfactual.is_some(),
                matches!(contract.target.query, CausalQueryWire::NestedCounterfactual { .. })
                    && estimator_matches(&["mediation.linear"]),
                "program.checked_nested_counterfactual.target",
            ),
        ] {
            if present && !matches {
                unresolved.push(Arc::from(reason));
            }
        }
    }
    if contract.target.query != body.query {
        unresolved.push(Arc::from("body.query"));
    }
    // These prepared routes execute from retained model or estimator operations,
    // but their current result wires do not carry a portable operation and its
    // fitted data dependencies. Keep the artifact readable and name the missing
    // dependency instead of silently treating a checked in-process run as
    // independently replayable.
    let resolved_estimator = contract
        .program
        .as_ref()
        .and_then(|program| program.commitments.resolved_estimator.as_deref())
        .or(contract.estimator.as_deref());
    match resolved_estimator {
        Some("bayesian.basis.gcomp")
            if matches!(contract.target.query, CausalQueryWire::AverageEffect { .. }) =>
        {
            unresolved.push(Arc::from("dependencies.checked_bayesian_basis_ate_operation"));
        }
        Some("bayesian.basis.gcomp")
            if matches!(contract.target.query, CausalQueryWire::ConditionalEffect { .. }) =>
        {
            unresolved.push(Arc::from("dependencies.checked_bayesian_basis_cate_operation"));
        }
        Some("bayesian.robust_ate") => {
            unresolved.push(Arc::from("dependencies.checked_bayesian_robust_ate_operation"));
        }
        Some("glm.adjustment") => {
            unresolved.push(Arc::from("dependencies.checked_glm_operation"));
        }
        Some("rd.sharp") => {
            unresolved.push(Arc::from("dependencies.checked_rd_operation"));
        }
        Some("propensity.weighting" | "propensity.matching") => {
            unresolved.push(Arc::from("dependencies.checked_propensity_operation"));
        }
        Some("response.kennedy_dr" | "response.riesz_ade" | "response.gam_derivative")
            if matches!(&contract.target.query, CausalQueryWire::Response(query)
                if matches!(query.functional,
                    crate::ResponseFunctionalWire::PointDerivative { .. }
                    | crate::ResponseFunctionalWire::AverageDerivative { .. }
                    | crate::ResponseFunctionalWire::DirectionalDerivative { .. }
                    | crate::ResponseFunctionalWire::Jacobian { .. })) =>
        {
            unresolved.push(Arc::from("dependencies.checked_derivative_response_operation"));
        }
        Some("response.kennedy_dr")
            if matches!(&contract.target.query, CausalQueryWire::Response(query)
                if matches!(query.functional, crate::ResponseFunctionalWire::MeanCurve { .. })) =>
        {
            unresolved.push(Arc::from("dependencies.checked_response_grid_operation"));
        }
        Some("response.intervention_gcomp")
            if matches!(&contract.target.query, CausalQueryWire::Response(query)
                if matches!(query.functional, crate::ResponseFunctionalWire::InterventionResponse { .. })) =>
        {
            unresolved.push(Arc::from("dependencies.checked_intervention_response_operation"));
        }
        Some("interference.ht_hajek" | "interference.bayesian_gaussian")
            if matches!(contract.target.query, CausalQueryWire::Interference { .. }) =>
        {
            unresolved.push(Arc::from("dependencies.checked_interference_operation"));
        }
        Some("cell.aipw")
            if matches!(&contract.target.query, CausalQueryWire::Response(query)
                if matches!(query.functional, crate::ResponseFunctionalWire::InterventionResponse { .. })) =>
        {
            unresolved.push(Arc::from("dependencies.checked_intervention_response_operation"));
        }
        Some("gcm.fit" | "gcm.fit.bayesian")
            if matches!(contract.target.query, CausalQueryWire::Counterfactual { .. }) =>
        {
            unresolved.push(Arc::from("dependencies.fitted_counterfactual_mechanisms"));
        }
        _ => {}
    }
    if resolved_estimator == Some("linear.adjustment.ate")
        && body.identification.status == "graph_dependent"
        && body.structural_response.is_some()
    {
        unresolved.push(Arc::from("dependencies.checked_unknown_tiered_average_operation"));
    }
    if contract.structure_source == "graph_posterior"
        && matches!(contract.target.query, CausalQueryWire::AverageEffect { .. })
        && resolved_estimator == Some("linear.adjustment.ate")
        && contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference.eq_ignore_ascii_case("frequentist")
        })
    {
        unresolved.push(Arc::from(
            if matches!(contract.graph_class.as_str(), "Cpdag" | "Pag") {
                "dependencies.checked_class_graph_posterior_effect_operation"
            } else {
                "dependencies.checked_graph_posterior_effect_operation"
            },
        ));
    }
    if contract.structure_source == "graph_posterior"
        && contract.graph_class == "Dag"
        && matches!(contract.target.query, CausalQueryWire::AverageEffect { .. })
        && resolved_estimator == Some("bayesian.gcomp")
        && contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference.eq_ignore_ascii_case("bayesian")
        })
    {
        unresolved.push(Arc::from(
            "dependencies.checked_bayesian_graph_posterior_ate_operation",
        ));
    }
    if matches!(contract.graph_class.as_str(), "Cpdag" | "Pag")
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::AverageEffect { .. })
        && resolved_estimator == Some("linear.adjustment.ate")
        && contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference.eq_ignore_ascii_case("frequentist")
        })
    {
        unresolved.push(Arc::from("dependencies.checked_static_class_effect_operation"));
    }
    if contract.structure_source == "graph_posterior"
        && contract.graph_class == "Admg"
        && matches!(contract.target.query, CausalQueryWire::AverageEffect { .. })
        && resolved_estimator == Some("functional.effect")
    {
        unresolved.push(Arc::from("dependencies.checked_graph_posterior_effect_operation"));
    }
    if contract.structure_source == "graph_posterior"
        && contract.graph_class == "Admg"
        && matches!(contract.target.query, CausalQueryWire::Response(_))
        && resolved_estimator == Some("functional.effect")
    {
        unresolved.push(Arc::from("dependencies.checked_admg_graph_posterior_response_operation"));
    }
    if matches!(contract.graph_class.as_str(), "Dag" | "Cpdag" | "Pag")
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::ConditionalEffect { .. })
        && resolved_estimator == Some("conditional.linear.adjustment")
        && contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference.eq_ignore_ascii_case("frequentist")
        })
    {
        unresolved.push(Arc::from("dependencies.checked_conditional_effect_operation"));
    }
    if contract.graph_class == "Dag"
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::Mediation { .. })
        && resolved_estimator == Some("mediation.linear")
        && contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference.eq_ignore_ascii_case("frequentist")
        })
    {
        unresolved.push(Arc::from("dependencies.checked_mediation_operation"));
    }
    if contract.graph_class == "Dag"
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::Mediation { .. })
        && resolved_estimator == Some("mediation.linear")
        && contract
            .program
            .as_ref()
            .is_some_and(|program| program.commitments.inference.eq_ignore_ascii_case("bayesian"))
    {
        unresolved.push(Arc::from("dependencies.checked_bayesian_mediation_operation"));
    }
    if contract.graph_class == "Dag"
        && contract.structure_source == "explicit"
        && matches!(
            contract.target.query,
            CausalQueryWire::AnomalyAttribution { .. } | CausalQueryWire::ChangeAttribution { .. }
        )
        && ((resolved_estimator == Some("gcm.fit")
            && contract.program.as_ref().is_some_and(|program| {
                program.commitments.inference.eq_ignore_ascii_case("frequentist")
            }))
            || (matches!(contract.target.query, CausalQueryWire::AnomalyAttribution { .. })
                && resolved_estimator == Some("gcm.fit.bayesian")
                && contract.program.as_ref().is_some_and(|program| {
                    program.commitments.inference.eq_ignore_ascii_case("bayesian")
                }))
            || (matches!(contract.target.query, CausalQueryWire::ChangeAttribution { .. })
                && resolved_estimator == Some("gcm.attribution.bayesian")
                && contract.program.as_ref().is_some_and(|program| {
                    program.commitments.inference.eq_ignore_ascii_case("bayesian")
                })))
    {
        unresolved.push(Arc::from("dependencies.checked_attribution_operation"));
    }
    if contract.graph_class == "Dag"
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::ConditionalEffect { .. })
        && resolved_estimator == Some("conditional.bayesian")
        && contract
            .program
            .as_ref()
            .is_some_and(|program| program.commitments.inference.eq_ignore_ascii_case("bayesian"))
    {
        unresolved.push(Arc::from("dependencies.checked_bayesian_conditional_operation"));
    }
    if contract.graph_class == "Dag"
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::AverageEffect { .. })
        && resolved_estimator == Some("bayesian.gcomp")
        && contract
            .program
            .as_ref()
            .is_some_and(|program| program.commitments.inference.eq_ignore_ascii_case("bayesian"))
    {
        unresolved.push(Arc::from("dependencies.checked_bayesian_dag_ate_operation"));
    }
    if contract.graph_class == "TemporalDag"
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(&contract.target.query, CausalQueryWire::Response(query)
            if matches!(query.functional, crate::ResponseFunctionalWire::MeanCurve { .. })
                && matches!(query.observation, crate::ObservationSpecWire::Complete))
        && resolved_estimator == Some("temporal.response.gcomp")
        && contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference.eq_ignore_ascii_case("frequentist")
        })
    {
        unresolved.push(Arc::from("dependencies.checked_temporal_response_operation"));
    }
    if contract.graph_class == "TemporalDag"
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::TemporalEffect { .. })
        && contract.data_snapshot.as_ref().is_some_and(|snapshot| snapshot.modality == "series")
        && matches!(
            resolved_estimator,
            Some("temporal.linear.adjustment" | "temporal.sequential.gcomp")
        )
        && contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference.eq_ignore_ascii_case("frequentist")
        })
    {
        unresolved.push(Arc::from("dependencies.checked_temporal_dag_effect_operation"));
    }
    if contract.graph_class == "TemporalDag"
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::TemporalEffect { .. })
        && contract.data_snapshot.as_ref().is_some_and(|snapshot| snapshot.modality == "series")
        && resolved_estimator == Some("bayesian.temporal.gcomp")
        && contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference.eq_ignore_ascii_case("bayesian")
        })
    {
        unresolved.push(Arc::from("dependencies.checked_bayesian_temporal_dag_effect_operation"));
    }
    if contract.graph_class == "TemporalDag"
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::TemporalEffect { .. })
        && contract.data_snapshot.as_ref().is_some_and(|snapshot| snapshot.modality == "series")
        && resolved_estimator == Some("temporal.sequential.gcomp")
        && contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference.eq_ignore_ascii_case("bayesian")
        })
    {
        unresolved.push(Arc::from("dependencies.checked_bayesian_temporal_dag_effect_operation"));
    }
    if contract.graph_class == "TemporalDag"
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::Mediation { .. })
        && contract.data_snapshot.as_ref().is_some_and(|snapshot| snapshot.modality == "series")
        && matches!(resolved_estimator, Some("temporal.mediation" | "temporal.mediation.bayesian"))
    {
        unresolved.push(Arc::from("dependencies.checked_temporal_mediation_operation"));
    }
    if matches!(contract.graph_class.as_str(), "TemporalCpdag" | "TemporalPag")
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, CausalQueryWire::TemporalEffect { .. })
        && contract.data_snapshot.as_ref().is_some_and(|snapshot| snapshot.modality == "series")
        && matches!(
            resolved_estimator,
            Some("temporal.linear.adjustment" | "temporal.sequential.gcomp")
        )
        && contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference.eq_ignore_ascii_case("frequentist")
        })
    {
        unresolved.push(Arc::from("dependencies.checked_temporal_class_effect_operation"));
    }
    if contract.estimator.as_deref() == Some("functional.distribution")
        && body.interventional_distribution.is_none()
    {
        // Pre-atom artifacts remain decodable, but cannot independently verify
        // the distribution that produced the scalar summary.
        unresolved.push(Arc::from("body.interventional_distribution"));
    }
    let factor_laws = contract
        .data_snapshot
        .as_ref()
        .and_then(|snapshot| snapshot.distribution_factor_laws.as_ref());
    let posterior_distribution = contract.estimator.as_deref() == Some("functional.distribution")
        && (contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference == "bayesian" || program.commitments.prior_required
        }) || contract
            .inference_binding
            .as_ref()
            .is_some_and(|binding| binding.inference == "bayesian" || binding.bayesian.is_some()));
    let functional_effect = is_scalar_functional_effect_contract(contract);
    let graph_posterior_functional_effect = contract.structure_source == "graph_posterior"
        && contract.graph_class == "Admg"
        && contract
            .program
            .as_ref()
            .and_then(|program| program.commitments.resolved_estimator.as_deref())
            .or(contract.estimator.as_deref())
            == Some("functional.effect");
    let posterior_functional_effect = functional_effect
        && (contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference == "bayesian" || program.commitments.prior_required
        }) || contract
            .inference_binding
            .as_ref()
            .is_some_and(|binding| binding.inference == "bayesian" || binding.bayesian.is_some()));
    if posterior_distribution {
        // Posterior atoms require joint per-draw factor tables and draw
        // identity. Empirical marginal laws cannot replay their weights.
        unresolved.push(Arc::from("dependencies.distribution_posterior_factor_draws"));
    }
    if contract.estimator.as_deref() == Some("functional.distribution") && factor_laws.is_none() {
        // The checked expression proves what to evaluate, while the data snapshot
        // only carries partition digests. Without portable factor laws a detached
        // consumer cannot recompute the atom probabilities.
        unresolved.push(Arc::from("dependencies.distribution_factor_laws"));
    }
    if functional_effect
        && factor_laws.is_none()
        && !posterior_functional_effect
        && !graph_posterior_functional_effect
    {
        unresolved.push(Arc::from("dependencies.functional_effect_factor_laws"));
    }
    if posterior_functional_effect && !graph_posterior_functional_effect {
        // The scalar provider law is enough for a frequentist point replay, but Bayesian
        // results require the shared row weights for every posterior draw.
        unresolved.push(Arc::from("dependencies.functional_effect_posterior_draws"));
    }
    if is_response_functional_effect_contract(contract) {
        let graph_posterior = contract.structure_source == "graph_posterior"
            && contract.graph_class == "Admg"
            && contract
                .program
                .as_ref()
                .and_then(|program| program.commitments.resolved_estimator.as_deref())
                .or(contract.estimator.as_deref())
                == Some("functional.effect");
        let posterior = contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference == "bayesian" || program.commitments.prior_required
        }) || contract
            .inference_binding
            .as_ref()
            .is_some_and(|binding| binding.inference == "bayesian" || binding.bayesian.is_some());
        if graph_posterior {
            // The atom-wise checked execution and frozen graph mass are the
            // replay dependency; one scalar program cannot represent the family.
        } else if posterior {
            unresolved.push(Arc::from("dependencies.functional_effect_response_posterior_draws"));
        } else {
            match (
                contract
                    .program
                    .as_ref()
                    .and_then(|program| program.checked_functional_response_grid.as_ref()),
                contract
                    .data_snapshot
                    .as_ref()
                    .and_then(|snapshot| snapshot.distribution_factor_laws.as_ref()),
                body.response.as_ref(),
            ) {
                (Some(grid), Some(laws), Some(response)) => {
                    if let Err(reason) = replay_functional_response_grid(
                        &contract.target.query,
                        grid,
                        &response.estimate,
                        laws,
                    ) {
                        unresolved.push(Arc::from(reason));
                    }
                }
                (None, _, _) => {
                    unresolved.push(Arc::from("program.functional_effect_response_grid"))
                }
                (_, None, _) => unresolved
                    .push(Arc::from("dependencies.functional_effect_response_factor_laws")),
                (_, _, None) => unresolved.push(Arc::from("body.response")),
            }
        }
    }
    if body.identification.query != body.query {
        unresolved.push(Arc::from("body.identification.query"));
    }
    if contract.estimator.as_deref() == Some("functional.distribution") && !posterior_distribution {
        match (
            body.interventional_distribution.as_ref(),
            contract.program.as_ref().and_then(|program| program.functional_program.as_ref()),
            contract
                .data_snapshot
                .as_ref()
                .and_then(|snapshot| snapshot.distribution_factor_laws.as_ref()),
        ) {
            (Some(result), Some(program), Some(laws)) => {
                if let Err(reason) = crate::distribution_replay::replay_distribution_atoms(
                    &body.query,
                    result,
                    program,
                    laws,
                ) {
                    unresolved.push(Arc::from(reason));
                }
            }
            _ => {
                if !unresolved
                    .iter()
                    .any(|reason| reason.as_ref() == "dependencies.distribution_factor_laws")
                {
                    unresolved.push(Arc::from("dependencies.distribution_factor_laws"));
                }
            }
        }
    }
    if functional_effect && !posterior_functional_effect && !graph_posterior_functional_effect {
        match (
            contract.program.as_ref().and_then(|program| program.functional_program.as_ref()),
            contract
                .data_snapshot
                .as_ref()
                .and_then(|snapshot| snapshot.distribution_factor_laws.as_ref()),
        ) {
            (Some(program), Some(laws)) => {
                if let Err(reason) = crate::distribution_replay::replay_functional_scalar(
                    &body.query,
                    body.estimate,
                    program,
                    laws,
                ) {
                    unresolved.push(Arc::from(reason));
                }
            }
            (_, None) => {
                if !unresolved
                    .iter()
                    .any(|reason| reason.as_ref() == "dependencies.functional_effect_factor_laws")
                {
                    unresolved.push(Arc::from("dependencies.functional_effect_factor_laws"));
                }
            }
            (None, Some(_)) => unresolved.push(Arc::from("program.functional_program")),
        }
    }
    if matches!(contract.target.query, crate::CausalQueryWire::NestedCounterfactual { .. }) {
        if let Err(reason) = verify_nested_counterfactual_result(contract, body) {
            unresolved.push(Arc::from(reason));
        }
    }
    if contract.target.schema.variable_names() != header.variable_names {
        unresolved.push(Arc::from("header.variable_names"));
    }
    if let Some(slot) = contract.reasoning.identification.value.as_ref() {
        if slot.status != body.identification.status {
            unresolved.push(Arc::from("identification.status"));
        }
        if validate_mixture_masses(
            slot.identified_mass,
            slot.unidentified_mass,
            slot.unevaluable_mass,
            slot.incomplete_search_mass,
        )
        .is_err()
        {
            unresolved.push(Arc::from("reasoning.identification.masses"));
        }
        if let Some(structural) = &body.structural_response {
            let masses_match = (slot.identified_mass - structural.identified_mass).abs() <= 1e-12
                && (slot.unidentified_mass - structural.unidentified_mass).abs() <= 1e-12
                && (slot.unevaluable_mass - structural.unevaluable_mass).abs() <= 1e-12
                && (slot.incomplete_search_mass - structural.subsampled_out_mass).abs() <= 1e-12;
            if !masses_match {
                unresolved.push(Arc::from("reasoning.identification.masses"));
            }
            if slot.weight_basis.as_deref() != Some(weight_basis_name(structural.weight_basis)) {
                unresolved.push(Arc::from("reasoning.identification.weight_basis"));
            }
        }
    }
    if let Some(product) = &contract.identification_product {
        if !identification_product_matches_body(product, body) {
            unresolved.push(Arc::from("body.identification_product"));
        }
    } else if contract.identities.identification_product.is_some() {
        unresolved.push(Arc::from("identities.identification_product"));
    }
    if contract.estimator.as_deref() == Some("frontdoor.linear_two_stage")
        && !body.assumptions.iter().any(|assumption| {
            matches!(&assumption.assumption,
                crate::trace::AssumptionTagWire::ParametricRestriction { id, .. }
                    if id == "frontdoor.linear_path_product")
        })
    {
        unresolved.push(Arc::from("body.frontdoor_assumption"));
    }
    require_present(&mut unresolved, "identities.identification", contract.identification.as_ref());
    require_present(&mut unresolved, "identities.program", contract.program.as_ref());
    require_present(
        &mut unresolved,
        "identities.inference_binding",
        contract.inference_binding.as_ref(),
    );
    require_present(&mut unresolved, "identities.observation", contract.observation.as_ref());
    require_present(&mut unresolved, "identities.data_snapshot", contract.data_snapshot.as_ref());
    if let Some(claim) = &contract.claim {
        if !claim_id_matches(contract, claim, body) {
            unresolved.push(Arc::from("claim.id"));
        }
        if claim.kind == "point" && !claim_value_matches_body(claim, body) {
            unresolved.push(Arc::from("claim.value"));
        }
        if claim.kind != "point" && claim.value_bits.is_some() {
            unresolved.push(Arc::from("claim.value"));
        }
        let identification = contract.reasoning.identification.value.as_ref();
        if claim.kind != claim_kind_name(body, identification) {
            unresolved.push(Arc::from("claim.kind"));
        }
        let support = contract.reasoning.support.value.as_ref();
        if let Some(slot) = support {
            let empirical = support_empirical(&body.refutations)
                .unwrap_or_else(|| "unavailable:not_evaluated".into());
            if slot.empirical != empirical {
                unresolved.push(Arc::from("reasoning.support.empirical"));
            }
        }
        let domains = claim_domains(support, identification);
        if claim.identification_domain != domains.identification
            || claim.support_domain != domains.support
            || claim.evaluated_domain != domains.evaluated
        {
            unresolved.push(Arc::from("claim.domains"));
        }
        if claim.execution.is_some() && contract.execution.is_none() {
            unresolved.push(Arc::from("identities.execution"));
        }
        verify_claim_calibration(contract, claim, &mut unresolved);
    }
    unresolved
}

/// Producer validation permits readable exports with explicitly unavailable
/// replay dependencies. Independent consumption receives the unfiltered
/// refusals from [`verify_contract_against_body`]. A malformed present payload
/// remains fatal.
pub(crate) fn verify_contract_for_encoding(
    header: &AnalysisResultHeader,
    body: &AnalysisResultWire,
    contract: &AnalysisResultContractWire,
) -> Vec<Arc<str>> {
    producer_encoding_unresolved(contract, verify_contract_against_body(header, body, contract))
}

fn producer_encoding_unresolved(
    contract: &AnalysisResultContractWire,
    mut unresolved: Vec<Arc<str>>,
) -> Vec<Arc<str>> {
    unresolved.retain(|key| {
        !matches!(
            key.as_ref(),
            "dependencies.checked_glm_operation"
                | "dependencies.checked_rd_operation"
                | "dependencies.checked_propensity_operation"
                | "dependencies.checked_derivative_response_operation"
                | "dependencies.checked_response_grid_operation"
                | "dependencies.checked_intervention_response_operation"
                | "dependencies.fitted_counterfactual_mechanisms"
                | "dependencies.checked_graph_posterior_effect_operation"
                | "dependencies.checked_class_graph_posterior_effect_operation"
                | "dependencies.checked_static_class_effect_operation"
                | "dependencies.checked_admg_graph_posterior_response_operation"
                | "dependencies.checked_conditional_effect_operation"
                | "dependencies.checked_mediation_operation"
                | "dependencies.checked_bayesian_mediation_operation"
                | "dependencies.checked_attribution_operation"
                | "dependencies.checked_bayesian_dag_ate_operation"
                | "dependencies.checked_bayesian_basis_ate_operation"
                | "dependencies.checked_bayesian_basis_cate_operation"
                | "dependencies.checked_bayesian_robust_ate_operation"
                | "dependencies.checked_bayesian_conditional_operation"
                | "dependencies.checked_temporal_dag_effect_operation"
                | "dependencies.checked_bayesian_temporal_dag_effect_operation"
                | "dependencies.checked_temporal_class_effect_operation"
                | "dependencies.checked_temporal_response_operation"
                | "dependencies.checked_temporal_mediation_operation"
                | "dependencies.checked_interference_operation"
                | "dependencies.checked_unknown_tiered_average_operation"
                | "dependencies.checked_bayesian_graph_posterior_ate_operation"
        )
    });
    // Preserve portable structural artifacts while making the missing replay
    // input explicit to independent consumers.
    if contract.estimator.as_deref() == Some("functional.distribution") {
        unresolved.retain(|key| key.as_ref() != "dependencies.distribution_factor_laws");
        unresolved.retain(|key| key.as_ref() != "dependencies.distribution_posterior_factor_draws");
    }
    if is_scalar_functional_effect_contract(contract) {
        unresolved.retain(|key| key.as_ref() != "dependencies.functional_effect_factor_laws");
        unresolved.retain(|key| key.as_ref() != "dependencies.functional_effect_posterior_draws");
    }
    if is_response_functional_effect_contract(contract) {
        unresolved.retain(|key| key.as_ref() != "program.functional_effect_response_grid");
        unresolved
            .retain(|key| key.as_ref() != "dependencies.functional_effect_response_factor_laws");
        unresolved.retain(|key| {
            key.as_ref() != "dependencies.functional_effect_response_posterior_draws"
        });
    }
    if contract
        .program
        .as_ref()
        .is_some_and(|program| program.checked_linear_adjustment_lowering.is_some())
    {
        unresolved.retain(|key| key.as_ref() != "dependencies.linear_fit_sufficient_statistics");
    }
    if contract.estimator.as_deref() == Some("aipw") && contract.program.is_some() {
        if contract.program.as_ref().is_some_and(|program| program.checked_aipw_lowering.is_none())
        {
            unresolved.retain(|key| key.as_ref() != "program.checked_aipw_lowering");
        }
        if contract.checked_aipw_rows.is_none() && contract.identities.checked_aipw_rows.is_none() {
            unresolved.retain(|key| key.as_ref() != "checked_aipw.row_binding");
        }
    }
    let is_iv = matches!(contract.estimator.as_deref(), Some("iv.wald" | "iv.2sls"))
        || contract.program.as_ref().is_some_and(|program| {
            matches!(program.commitments.resolved_estimator.as_deref(), Some("iv.wald" | "iv.2sls"))
        });
    if is_iv
        && contract.program.as_ref().is_some_and(|program| program.checked_iv_lowering.is_none())
    {
        unresolved.retain(|key| key.as_ref() != "program.checked_iv_lowering");
    }
    unresolved
}

/// Re-derive the claim's calibration slot from the basis it carries and check
/// that basis against the payloads this contract rehashes.
fn verify_claim_calibration(
    contract: &AnalysisResultContractWire,
    claim: &ClaimSectionWire,
    unresolved: &mut Vec<Arc<str>>,
) {
    let slot = &claim.calibration;
    let statuses_ok = std::iter::once(slot).chain(slot.secondary.iter()).all(|slot| {
        matches!(slot.status.as_str(), "calibrated" | "scope_not_assessed" | "unavailable")
    });
    if !statuses_ok || crate::calibration::rederive_calibration(slot) != *slot {
        unresolved.push(Arc::from("claim.calibration"));
    }
    let consistent = std::iter::once(slot)
        .chain(slot.secondary.iter())
        .filter_map(|slot| slot.basis.as_ref())
        .all(|basis| calibration_basis_matches_contract(contract, basis));
    if !consistent {
        unresolved.push(Arc::from("claim.calibration.basis"));
    }
}

/// The basis fields a consumer can recompute from rehashed payloads: the
/// support coordinate (query, graph axis, structure, inference), the resolved
/// estimator, the snapshot row count and modality, and the identification
/// masses.
fn calibration_basis_matches_contract(
    contract: &AnalysisResultContractWire,
    basis: &crate::calibration::CalibrationBasisWire,
) -> bool {
    let key = &basis.key;
    let coordinate_ok = contract
        .reasoning
        .support
        .value
        .as_ref()
        .and_then(|slot| slot.matrix_coordinate.as_deref())
        .is_some_and(|coordinate| {
            let mut parts = coordinate.split(':');
            let (Some(query), Some(graph), Some(structure), Some(inference)) =
                (parts.next(), parts.next(), parts.next(), parts.next())
            else {
                return false;
            };
            let structure =
                if structure == "graph_posterior" { "graph_posterior" } else { "fixed" };
            query == key.query
                && graph == key.graph_class
                && structure == key.structure
                && inference == key.inference
        });
    let program_ok = contract.program.as_ref().is_some_and(|program| {
        let commitments = &program.commitments;
        commitments.inference.eq_ignore_ascii_case(&key.inference)
            && commitments
                .resolved_estimator
                .as_deref()
                .is_none_or(|resolved| resolved == key.estimator)
    });
    let snapshot_ok = contract.data_snapshot.as_ref().is_some_and(|snapshot| {
        basis.scope.row_count <= snapshot.row_count
            && snapshot.modality == key.modality
            && match snapshot.modality.as_str() {
                "panel" => key.dependence == "panel_cluster",
                "tabular" => key.dependence == "iid",
                _ => key.dependence != "panel_cluster",
            }
    });
    let identification_ok = contract.reasoning.identification.value.as_ref().is_none_or(|slot| {
        let unidentified =
            slot.unidentified_mass + slot.unevaluable_mass + slot.incomplete_search_mass;
        (unidentified - basis.scope.unidentified_mass).abs() <= 1e-9
    });
    coordinate_ok && program_ok && snapshot_ok && identification_ok
}

fn verify_stored_payloads(contract: &AnalysisResultContractWire) -> Vec<Arc<str>> {
    let mut unresolved = Vec::new();
    require_slot(&mut unresolved, "identification", &contract.reasoning.identification);
    require_slot(&mut unresolved, "support", &contract.reasoning.support);
    require_slot(&mut unresolved, "uncertainty", &contract.reasoning.uncertainty);
    require_slot(&mut unresolved, "assumptions", &contract.reasoning.assumptions);
    match digest_wire(IdentityDomain::Target, &contract.target) {
        Ok(target) if target.as_bytes() == &contract.identities.target => {}
        Ok(_) => unresolved.push(Arc::from("identities.target")),
        Err(_) => unresolved.push(Arc::from("target_payload")),
    }
    require_payload_digest(
        &mut unresolved,
        "identities.identification",
        contract.identification.as_ref(),
        Some(&contract.identities.identification),
        identification_digest,
    );
    require_payload_digest(
        &mut unresolved,
        "identities.identification_product",
        contract.identification_product.as_ref(),
        contract.identities.identification_product.as_ref(),
        identification_product_digest_wire,
    );
    require_payload_digest(
        &mut unresolved,
        "identities.program",
        contract.program.as_ref(),
        Some(&contract.identities.program),
        program_digest,
    );
    let graph_posterior_functional_effect = contract.structure_source == "graph_posterior"
        && contract.graph_class == "Admg"
        && contract
            .program
            .as_ref()
            .and_then(|program| program.commitments.resolved_estimator.as_deref())
            .or(contract.estimator.as_deref())
            == Some("functional.effect");
    let functional = contract
        .program
        .as_ref()
        .and_then(|item| item.functional_program.as_ref())
        .filter(|_| !graph_posterior_functional_effect);
    let posterior_distribution = contract.estimator.as_deref() == Some("functional.distribution")
        && (contract.program.as_ref().is_some_and(|program| {
            program.commitments.inference == "bayesian" || program.commitments.prior_required
        }) || contract
            .inference_binding
            .as_ref()
            .is_some_and(|binding| binding.inference == "bayesian" || binding.bayesian.is_some()));
    if contract.estimator.as_deref() == Some("functional.distribution")
        && functional.is_none()
        && !posterior_distribution
    {
        // Older artifacts stay readable but cannot advertise a verified
        // executable distribution without the program that was evaluated.
        unresolved.push(Arc::from("program.functional_program"));
    }
    if is_scalar_functional_effect_contract(contract)
        && functional.is_none()
        && !graph_posterior_functional_effect
    {
        unresolved.push(Arc::from("program.functional_program"));
    }
    if let Some(program) = functional.filter(|_| {
        !posterior_distribution
            && (contract.estimator.as_deref() == Some("functional.distribution")
                || is_scalar_functional_effect_contract(contract))
    }) {
        if crate::functional_program_from_wire(program, antecedent_expr::ProgramLimits::default())
            .is_err()
        {
            unresolved.push(Arc::from("program.functional_program"));
        }
        let expected_variables: Vec<_> = contract
            .target
            .schema
            .variables
            .iter()
            .map(|variable| (variable.id, variable.name.clone()))
            .collect();
        let matches_identification =
            contract.identification_product.as_ref().is_some_and(|product| {
                program.arena == product.arena
                    && program.source == program.executable
                    && product
                        .estimands
                        .iter()
                        .any(|estimand| estimand.functional == program.source)
            });
        let distribution_binding = contract.estimator.as_deref() == Some("functional.distribution")
            && matches!(contract.target.query, crate::CausalQueryWire::Distribution(_));
        let effect_method = match &contract.target.query {
            crate::CausalQueryWire::AverageEffect { .. } => Some("general.id"),
            crate::CausalQueryWire::PathSpecific(_) => Some("path_specific.natural"),
            _ => None,
        };
        let effect_binding = is_scalar_functional_effect_contract(contract)
            && effect_method.is_some_and(|method| {
                contract.identification_product.as_ref().is_some_and(|product| {
                    product.estimands.iter().any(|estimand| {
                        estimand.functional == program.source && estimand.method == method
                    })
                })
            });
        if (!distribution_binding && !effect_binding)
            || program.variables != expected_variables
            || !matches_identification
        {
            unresolved.push(Arc::from("program.functional_binding"));
        }
    }
    let frontdoor =
        contract.program.as_ref().and_then(|item| item.checked_frontdoor_lowering.as_ref());
    if (contract.estimator.as_deref() == Some("frontdoor.linear_two_stage")
        || contract
            .program
            .as_ref()
            .and_then(|item| item.commitments.resolved_estimator.as_deref())
            == Some("frontdoor.linear_two_stage"))
        && frontdoor.is_none()
    {
        unresolved.push(Arc::from("program.checked_frontdoor_lowering"));
    }
    if let Some(frontdoor) = frontdoor {
        if !verify_frontdoor_lowering(contract, frontdoor) {
            unresolved.push(Arc::from("program.frontdoor_binding"));
        }
    }
    let checked_iv = contract.program.as_ref().and_then(|item| item.checked_iv_lowering.as_ref());
    let is_checked_iv = matches!(contract.estimator.as_deref(), Some("iv.wald" | "iv.2sls"))
        || contract.program.as_ref().is_some_and(|item| {
            matches!(item.commitments.resolved_estimator.as_deref(), Some("iv.wald" | "iv.2sls"))
        });
    if is_checked_iv && checked_iv.is_none() {
        unresolved.push(Arc::from("program.checked_iv_lowering"));
    }
    if let Some(iv) = checked_iv {
        if !verify_checked_iv(contract, iv) {
            unresolved.push(Arc::from("program.checked_iv_binding"));
        }
    }
    verify_checked_linear_adjustment(contract, &mut unresolved);
    if contract
        .program
        .as_ref()
        .is_some_and(|program| program.checked_linear_adjustment_lowering.is_some())
    {
        // This payload retains the checked design roles and selected
        // procedure, but the artifact has no rows or replay sufficient
        // statistics for recomputing the reported numeric estimate.
        unresolved.push(Arc::from("dependencies.linear_fit_sufficient_statistics"));
    }
    verify_checked_aipw(contract, &mut unresolved);
    require_payload_digest(
        &mut unresolved,
        "identities.inference_binding",
        contract.inference_binding.as_ref(),
        Some(&contract.identities.inference_binding),
        inference_binding_digest,
    );
    require_payload_digest(
        &mut unresolved,
        "identities.observation",
        contract.observation.as_ref(),
        Some(&contract.identities.observation),
        |observation| digest_wire(IdentityDomain::Observation, observation),
    );
    require_payload_digest(
        &mut unresolved,
        "identities.data_snapshot",
        contract.data_snapshot.as_ref(),
        Some(&contract.identities.data_snapshot),
        data_snapshot_digest,
    );
    require_payload_digest(
        &mut unresolved,
        "identities.execution",
        contract.execution.as_ref(),
        contract.identities.execution.as_ref(),
        execution_digest,
    );
    require_nested_digest(
        &mut unresolved,
        "identities.execution",
        contract.identities.execution,
        contract.claim.as_ref().and_then(|claim| claim.execution.as_ref()),
    );
    require_nested_digest(
        &mut unresolved,
        "program.target",
        contract.program.as_ref().map(|item| item.target),
        Some(&contract.identities.target),
    );
    require_nested_digest(
        &mut unresolved,
        "program.identification",
        contract.program.as_ref().map(|item| item.identification),
        Some(&contract.identities.identification),
    );
    require_nested_digest(
        &mut unresolved,
        "program.identification_product",
        contract.program.as_ref().and_then(|item| item.identification_product),
        contract.identities.identification_product.as_ref(),
    );
    require_nested_digest(
        &mut unresolved,
        "data_snapshot.observation",
        contract.data_snapshot.as_ref().map(|item| item.observation),
        Some(&contract.identities.observation),
    );
    verify_checked_aipw_rows(contract, &mut unresolved);
    verify_layer_links(contract, &mut unresolved);
    match contract_seal_of(contract) {
        Ok(seal) if seal == contract.seal => {}
        _ => unresolved.push(Arc::from("contract.seal")),
    }
    unresolved
}

fn is_functional_effect_estimator_contract(contract: &AnalysisResultContractWire) -> bool {
    contract.estimator.as_deref() == Some("functional.effect")
        || contract.program.as_ref().is_some_and(|program| {
            program.commitments.resolved_estimator.as_deref() == Some("functional.effect")
        })
}

fn verify_nested_counterfactual_result(
    contract: &AnalysisResultContractWire,
    body: &AnalysisResultWire,
) -> Result<(), &'static str> {
    let crate::CausalQueryWire::NestedCounterfactual {
        treatment,
        mediator,
        outcome,
        control_bits,
        active_bits,
    } = contract.target.query
    else {
        return Err("program.checked_nested_counterfactual.target");
    };
    let operation = contract
        .program
        .as_ref()
        .and_then(|program| program.checked_nested_counterfactual.as_ref())
        .ok_or("program.checked_nested_counterfactual")?;
    if contract.graph_class != "Dag"
        || contract.structure_source != "explicit"
        || !matches!(contract.identifier.as_deref(), None | Some("path_specific.natural"))
        || !matches!(contract.estimator.as_deref(), None | Some("mediation.linear"))
        || contract
            .program
            .as_ref()
            .and_then(|program| program.commitments.resolved_estimator.as_deref())
            != Some("mediation.linear")
        || contract.inference_binding.as_ref().map(|binding| binding.inference.as_str())
            != Some("frequentist")
    {
        return Err("program.checked_nested_counterfactual.scope");
    }
    if operation.format != 1
        || operation.treatment != treatment
        || operation.mediator != mediator
        || operation.outcome != outcome
        || operation.control_bits != control_bits
        || operation.active_bits != active_bits
        || operation.model != "linear_gaussian"
        || operation.procedure != "natural_direct_shared_exogenous"
        || !f64::from_bits(active_bits).is_finite()
        || !f64::from_bits(control_bits).is_finite()
    {
        return Err("program.checked_nested_counterfactual.binding");
    }
    let graph = contract
        .identification
        .as_ref()
        .map(|identity| &identity.graph)
        .ok_or("identities.identification")?;
    let crate::GraphIdentityWire::Dag(dag) = graph else {
        return Err("program.checked_nested_counterfactual.graph");
    };
    let mut edges = dag.edges.clone();
    edges.sort_unstable();
    let mut expected_edges = vec![(treatment, mediator), (treatment, outcome), (mediator, outcome)];
    expected_edges.sort_unstable();
    if dag.node_count != 3 || edges != expected_edges {
        return Err("program.checked_nested_counterfactual.graph");
    }
    if !body.identification.estimands.iter().any(|estimand| {
        estimand.method == "path_specific.natural" && estimand.mediators == [mediator]
    }) {
        return Err("program.checked_nested_counterfactual.identification");
    }
    let snapshot =
        contract.data_snapshot.as_ref().ok_or("dependencies.nested_counterfactual_fit_moments")?;
    let fit = snapshot
        .nested_counterfactual_fit
        .as_ref()
        .ok_or("dependencies.nested_counterfactual_fit_moments")?;
    if fit.format != 1
        || fit.complete_case_rows < 4
        || fit.complete_case_rows > snapshot.row_count
        || fit.gram.iter().chain(&fit.outcome_cross).any(|value| !value.is_finite())
        || (fit.gram[0] - fit.complete_case_rows as f64).abs() > 1e-9
        || (0..3).any(|row| {
            (0..3).any(|column| {
                (fit.gram[row * 3 + column] - fit.gram[column * 3 + row]).abs() > 1e-9
            })
        })
    {
        return Err("dependencies.nested_counterfactual_fit_moments");
    }
    let beta = solve_nested_outcome_moments(fit.gram, fit.outcome_cross)
        .ok_or("dependencies.nested_counterfactual_fit_rank")?;
    let expected = beta[1] * (f64::from_bits(active_bits) - f64::from_bits(control_bits));
    let actual = body.estimate.ok_or("body.nested_counterfactual_estimate")?;
    if !expected.is_finite()
        || !actual.is_finite()
        || (actual - expected).abs() > 1e-8 * expected.abs().max(1.0)
    {
        return Err("body.nested_counterfactual_estimate");
    }
    Ok(())
}

fn solve_nested_outcome_moments(gram: [f64; 9], cross: [f64; 3]) -> Option<[f64; 3]> {
    let mut augmented = [[0.0_f64; 4]; 3];
    let scale = gram.iter().map(|value| value.abs()).fold(0.0_f64, f64::max);
    if !scale.is_finite() || scale == 0.0 {
        return None;
    }
    for row in 0..3 {
        augmented[row][..3].copy_from_slice(&gram[row * 3..row * 3 + 3]);
        augmented[row][3] = cross[row];
    }
    for pivot in 0..3 {
        let best = (pivot..3).max_by(|left, right| {
            augmented[*left][pivot].abs().total_cmp(&augmented[*right][pivot].abs())
        })?;
        if augmented[best][pivot].abs() <= 1e-12 * scale {
            return None;
        }
        augmented.swap(pivot, best);
        let divisor = augmented[pivot][pivot];
        for column in pivot..4 {
            augmented[pivot][column] /= divisor;
        }
        for row in 0..3 {
            if row == pivot {
                continue;
            }
            let factor = augmented[row][pivot];
            for column in pivot..4 {
                augmented[row][column] -= factor * augmented[pivot][column];
            }
        }
    }
    Some([augmented[0][3], augmented[1][3], augmented[2][3]])
}

fn is_scalar_functional_effect_contract(contract: &AnalysisResultContractWire) -> bool {
    is_functional_effect_estimator_contract(contract)
        && matches!(
            contract.target.query,
            crate::CausalQueryWire::AverageEffect { .. } | crate::CausalQueryWire::PathSpecific(_)
        )
}

fn is_response_functional_effect_contract(contract: &AnalysisResultContractWire) -> bool {
    is_functional_effect_estimator_contract(contract)
        && matches!(contract.target.query, crate::CausalQueryWire::Response(_))
}

fn replay_functional_response_grid(
    target: &CausalQueryWire,
    family: &crate::CheckedFunctionalResponseGridWire,
    reported: &crate::ResponseIdentificationWire,
    laws: &crate::DistributionFactorLawsWire,
) -> Result<(), &'static str> {
    use crate::{GridSpecWire, ResponseFunctionalWire, ResponseValueWire};
    let CausalQueryWire::Response(target) = target else {
        return Err("program.functional_effect_response_grid.target");
    };
    let ResponseFunctionalWire::MeanCurve { outcome, treatment } = &target.functional else {
        return Err("program.functional_effect_response_grid.target_kind");
    };
    if family.format != 1 || family.outcome != *outcome || family.treatment != treatment.variable {
        return Err("program.functional_effect_response_grid.target_binding");
    }
    let expected_grid = match &treatment.grid {
        GridSpecWire::Values(values) => values.clone(),
        GridSpecWire::Linspace { start, end, points } => {
            let points = usize::try_from(*points).map_err(|_| "functional_effect.response_grid")?;
            if points == 0 || points > 100_000 {
                return Err("functional_effect.response_grid_resource_limit");
            }
            if points == 1 {
                vec![*start]
            } else {
                (0..points)
                    .map(|index| start + (end - start) * index as f64 / (points - 1) as f64)
                    .collect()
            }
        }
    };
    if expected_grid.len() != family.members.len() || expected_grid.is_empty() {
        return Err("program.functional_effect_response_grid.member_count");
    }
    let (reported_grid, means) = match reported {
        crate::ResponseIdentificationWire::PointIdentified(ResponseValueWire::Surface {
            grid,
            dimension: 1,
            mean,
        }) => (grid, mean),
        _ => return Err("functional_effect.response_result_shape"),
    };
    if reported_grid.len() != expected_grid.len()
        || means.len() != expected_grid.len()
        || reported_grid
            .iter()
            .zip(&expected_grid)
            .any(|(left, right)| left.to_bits() != right.to_bits())
    {
        return Err("functional_effect.response_grid_result_mismatch");
    }
    if laws.format != 1
        || laws.provider != "empirical_table"
        || laws.missing_row_policy != "joint_complete_case"
        || laws.complete_case_rows == 0
        || laws.source_rows < laws.complete_case_rows
    {
        return Err("functional_effect.provider_provenance");
    }
    for ((member, expected), reported) in family.members.iter().zip(expected_grid).zip(means) {
        if member.grid_value_bits != expected.to_bits() {
            return Err("program.functional_effect_response_grid.member_value");
        }
        let CausalQueryWire::Response(query) = &member.query else {
            return Err("program.functional_effect_response_grid.member_query_kind");
        };
        let ResponseFunctionalWire::InterventionResponse { outcome: member_outcome, interventions } =
            &query.functional
        else {
            return Err("program.functional_effect_response_grid.member_functional");
        };
        if member_outcome != outcome
            || interventions.len() != 1
            || !matches!(interventions.first(), Some(crate::InterventionWire::Set { variable, value })
                if *variable == treatment.variable && value.to_value() == antecedent_core::Value::Float64(expected))
        {
            return Err("program.functional_effect_response_grid.member_intervention");
        }
        if member.identification.estimands.iter().all(|estimand| {
            estimand.method != "general.id" || estimand.functional != member.program.source
        }) || member.program.source != member.program.executable
        {
            return Err("program.functional_effect_response_grid.member_program");
        }
        crate::distribution_replay::replay_functional_scalar(
            &member.query,
            Some(*reported),
            &member.program,
            laws,
        )?;
    }
    Ok(())
}

fn verify_checked_linear_adjustment(
    contract: &AnalysisResultContractWire,
    unresolved: &mut Vec<Arc<str>>,
) {
    let lowering = contract
        .program
        .as_ref()
        .and_then(|program| program.checked_linear_adjustment_lowering.as_ref());
    // The checked linear-adjustment lowering currently covers the supplied
    // or accepted DAG ATE. Other graph classes may select the same numerical
    // estimator inside an envelope, but they do not have this lowering yet.
    let is_linear = contract.graph_class == "Dag"
        && matches!(contract.structure_source.as_str(), "explicit" | "accepted")
        && matches!(contract.target.query, crate::CausalQueryWire::AverageEffect { .. })
        && (contract.estimator.as_deref() == Some("linear.adjustment.ate")
            || contract.program.as_ref().is_some_and(|program| {
                program.commitments.resolved_estimator.as_deref() == Some("linear.adjustment.ate")
            }));
    if is_linear && lowering.is_none() {
        unresolved.push(Arc::from("program.checked_linear_adjustment_lowering"));
        return;
    }
    let Some(lowering) = lowering else { return };
    if !verify_linear_adjustment_lowering(contract, lowering) {
        unresolved.push(Arc::from("program.checked_linear_adjustment_binding"));
    }
}

fn verify_linear_adjustment_lowering(
    contract: &AnalysisResultContractWire,
    lowering: &crate::CheckedLinearAdjustmentLoweringWire,
) -> bool {
    let (Some(product), Some(program), Some(snapshot), Some(binding)) = (
        contract.identification_product.as_ref(),
        contract.program.as_ref(),
        contract.data_snapshot.as_ref(),
        contract.inference_binding.as_ref(),
    ) else {
        return false;
    };
    if lowering.format != 1
        || lowering.backend != "faer"
        || lowering.population != TargetPopulationWire::AllObserved
        || lowering.complete_case_rows == 0
        || lowering.complete_case_rows > snapshot.row_count
        || lowering.design_columns.first().map(String::as_str) != Some("intercept")
        || lowering.design_columns.get(1).map(String::as_str) != Some("treatment")
        || lowering.design_columns.len() != lowering.adjustment.len() + 2
        || lowering
            .design_columns
            .iter()
            .skip(2)
            .zip(&lowering.adjustment)
            .any(|(role, id)| role != &format!("covariate:{id}"))
        || !valid_linear_fit_kind(&lowering.fit_kind)
        || !matches!(
            lowering.se_kind.as_str(),
            "homoskedastic"
                | "hc0"
                | "hc1"
                | "hc2"
                | "hc3"
                | "cluster"
                | "multiway"
                | "newey_west"
                | "panel_cluster_hac"
        )
        || matches!(lowering.se_kind.as_str(), "newey_west" | "panel_cluster_hac")
            != lowering.se_lag.is_some()
    {
        return false;
    }
    let Some(estimand) = product.estimands.iter().find(|estimand| {
        estimand.functional == lowering.functional
            && estimand.method == "backdoor.adjustment"
            && estimand.adjustment_set == lowering.adjustment
    }) else {
        return false;
    };
    let target_matches = match &contract.target.query {
        crate::CausalQueryWire::AverageEffect {
            treatment,
            outcome,
            active,
            control,
            outcome_functional,
            target_population,
            effect_modifiers,
        } => {
            *treatment == lowering.treatment
                && *outcome == lowering.outcome
                && intervention_bits(active, *treatment) == Some(lowering.active_bits)
                && intervention_bits(control, *treatment) == Some(lowering.control_bits)
                && matches!(outcome_functional, crate::query_wire::OutcomeFunctionalWire::Mean)
                && target_population == &lowering.population
                && effect_modifiers.is_empty()
        }
        _ => false,
    };
    if !product.estimands.iter().any(|candidate| candidate.functional == lowering.functional) {
        return false;
    }
    let mut arena = match crate::expr_arena_from_wire(&lowering.arena) {
        Ok(arena) => arena,
        Err(_) => return false,
    };
    let treatment = antecedent_core::VariableId::from_raw(lowering.treatment);
    let outcome = antecedent_core::VariableId::from_raw(lowering.outcome);
    let adjustment = lowering
        .adjustment
        .iter()
        .copied()
        .map(antecedent_core::VariableId::from_raw)
        .collect::<Vec<_>>();
    let expected_source = arena.backdoor_ate(
        treatment,
        outcome,
        &adjustment,
        antecedent_core::Value::Float64(f64::from_bits(lowering.active_bits)),
        antecedent_core::Value::Float64(f64::from_bits(lowering.control_bits)),
    );
    let z = arena.intern_var_set(adjustment.iter().copied());
    let y = arena.intern_var_set([outcome]);
    let tz = arena.intern_var_set(std::iter::once(treatment).chain(adjustment.iter().copied()));
    let empty = arena.empty_var_set();
    let no_intervention = arena.empty_intervention_set();
    let marginal_z = arena.intern_distribution(
        z,
        empty,
        no_intervention,
        antecedent_expr::DomainRef::Observational,
    );
    let outcome_given_tz = arena.intern_distribution(
        y,
        tz,
        no_intervention,
        antecedent_expr::DomainRef::Observational,
    );
    let factors = arena.intern_list([outcome_given_tz, marginal_z]);
    let product_expr = arena.intern(antecedent_expr::ExprNode::Product(factors));
    let averaged =
        arena.intern(antecedent_expr::ExprNode::SumOut { variables: z, expr: product_expr });
    let expected_executable = arena.intern(antecedent_expr::ExprNode::Expectation {
        function: antecedent_expr::OutcomeExprId::identity(outcome),
        distribution: averaged,
    });
    let expected_arena = crate::expr_arena_to_wire(&arena).ok();
    let commitments_match = program.commitments.resolved_estimator.as_deref()
        == Some("linear.adjustment.ate")
        && program.commitments.inference == "frequentist"
        && !program.commitments.prior_required
        && program.commitments.se_kind.as_deref() == Some(lowering.se_kind.as_str())
        && program.commitments.interval_method == lowering.interval_method
        && lowering.interval_method
            == if lowering.bootstrap_replicates > 0 { "bootstrap_se" } else { "analytic_se" }
        && binding.inference == "frequentist"
        && binding.bayesian.is_none()
        && binding.bootstrap_replicates == lowering.bootstrap_replicates;
    let binding_matches = match binding.estimator_spec.as_ref() {
        Some(crate::EstimatorSpecWire::Default(name)) => name == "linear.adjustment.ate",
        Some(crate::EstimatorSpecWire::LinearAdjustmentAte(config)) => {
            config.bootstrap_replicates == lowering.bootstrap_replicates
                && config.se_kind.as_deref().unwrap_or("homoskedastic") == lowering.se_kind
                && config.backend == lowering.backend
                && config.fit_kind.as_ref().is_none_or(|(family, parameter)| {
                    let expected = match lowering.fit_kind.split_once(':') {
                        Some((kind, bits)) => (kind, u64::from_str_radix(bits, 16).ok()),
                        None => (lowering.fit_kind.as_str(), None),
                    };
                    family == expected.0 && *parameter == expected.1
                })
        }
        None => true,
        _ => false,
    };
    target_matches
        && estimand.adjustment_set == lowering.adjustment
        && expected_source.raw() == lowering.functional
        && expected_executable.raw() == lowering.executable
        && expected_arena.as_ref() == Some(&lowering.arena)
        && commitments_match
        && binding_matches
}

fn valid_linear_fit_kind(fit_kind: &str) -> bool {
    match fit_kind {
        "ols" => true,
        _ => fit_kind.split_once(':').is_some_and(|(kind, bits)| {
            let Ok(bits) = u64::from_str_radix(bits, 16) else { return false };
            let value = f64::from_bits(bits);
            value.is_finite() && value > 0.0 && matches!(kind, "ridge" | "lasso" | "huber")
        }),
    }
}

fn verify_checked_iv(
    contract: &AnalysisResultContractWire,
    lowering: &crate::CheckedIvLoweringWire,
) -> bool {
    let (Some(product), Some(program), Some(snapshot)) = (
        contract.identification_product.as_ref(),
        contract.program.as_ref(),
        contract.data_snapshot.as_ref(),
    ) else {
        return false;
    };
    let Some(_estimand) = product.estimands.iter().find(|estimand| {
        estimand.functional == lowering.functional
            && estimand.method == "iv"
            && estimand.instruments == [lowering.instrument]
            && estimand.adjustment_set == lowering.adjustment
    }) else {
        return false;
    };
    let (query_treatment, query_outcome, active, control, mean_outcome) =
        match &contract.target.query {
            crate::CausalQueryWire::AverageEffect {
                treatment,
                outcome,
                active,
                control,
                outcome_functional,
                ..
            } => (
                *treatment,
                *outcome,
                active,
                control,
                matches!(outcome_functional, crate::query_wire::OutcomeFunctionalWire::Mean),
            ),
            _ => return false,
        };
    let expected_procedure = match contract.estimator.as_deref() {
        Some("iv.wald") => "wald",
        Some("iv.2sls") => "two_stage_least_squares",
        _ => return false,
    };
    let expected_weak_uncertainty = match lowering.se_kind.as_str() {
        "homoskedastic" => "anderson_rubin_if_weak",
        "hc0" | "hc1" | "hc2" | "hc3" | "cluster" | "multiway" | "newey_west"
        | "panel_cluster_hac" => "withheld_non_homoskedastic",
        _ => return false,
    };
    let mut arena = match crate::expr_arena_from_wire(&product.arena) {
        Ok(arena) => arena,
        Err(_) => return false,
    };
    let Ok(expected_functional) = arena.iv_wald(
        antecedent_core::VariableId::from_raw(lowering.treatment),
        antecedent_core::VariableId::from_raw(lowering.outcome),
        &[antecedent_core::VariableId::from_raw(lowering.instrument)],
        &antecedent_core::Value::Float64(f64::from_bits(lowering.active_bits)),
        &antecedent_core::Value::Float64(f64::from_bits(lowering.control_bits)),
    ) else {
        return false;
    };
    let expected_arena = crate::expr_arena_to_wire(&arena).ok();
    lowering.format == 1
        && lowering.procedure == expected_procedure
        && lowering.treatment == query_treatment
        && lowering.outcome == query_outcome
        && intervention_bits(active, query_treatment) == Some(lowering.active_bits)
        && intervention_bits(control, query_treatment) == Some(lowering.control_bits)
        && mean_outcome
        && lowering.instrument_active_bits == 1.0_f64.to_bits()
        && lowering.instrument_control_bits == 0.0_f64.to_bits()
        && lowering.complete_case_rows > 0
        && lowering.complete_case_rows <= snapshot.row_count
        && program.commitments.resolved_estimator.as_deref()
            == Some(expected_procedure_id(expected_procedure))
        && program.commitments.se_kind.as_deref() == Some(lowering.se_kind.as_str())
        && lowering.weak_instrument_uncertainty == expected_weak_uncertainty
        && expected_functional.raw() == lowering.functional
        && expected_arena.as_ref() == Some(&product.arena)
}

fn expected_procedure_id(procedure: &str) -> &'static str {
    match procedure {
        "wald" => "iv.wald",
        _ => "iv.2sls",
    }
}

fn verify_frontdoor_lowering(
    contract: &AnalysisResultContractWire,
    lowering: &crate::CheckedFrontDoorLoweringWire,
) -> bool {
    let Some(product) = contract.identification_product.as_ref() else { return false };
    let Some(program) = contract.program.as_ref() else { return false };
    let Some(estimand) = product.estimands.iter().find(|estimand| {
        estimand.functional == lowering.functional
            && estimand.mediators == lowering.mediators
            && estimand.method == "frontdoor"
    }) else {
        return false;
    };
    let target_matches = match &contract.target.query {
        crate::CausalQueryWire::AverageEffect {
            treatment,
            outcome,
            active,
            control,
            outcome_functional,
            ..
        } => {
            *treatment == lowering.treatment
                && *outcome == lowering.outcome
                && intervention_bits(active, *treatment) == Some(lowering.active_bits)
                && intervention_bits(control, *treatment) == Some(lowering.control_bits)
                && matches!(outcome_functional, crate::query_wire::OutcomeFunctionalWire::Mean)
        }
        _ => false,
    };
    let expected_variables: Vec<_> = contract
        .target
        .schema
        .variables
        .iter()
        .map(|variable| (variable.id, variable.name.clone()))
        .collect();
    let program_valid = crate::functional_program_from_wire(
        &crate::FunctionalProgramWire {
            arena: lowering.arena.clone(),
            source: lowering.functional,
            executable: lowering.executable,
            variables: expected_variables,
        },
        antecedent_expr::ProgramLimits::default(),
    )
    .is_ok();
    let lowering_matches_frontdoor = (|| {
        let mut arena = crate::expr_arena_from_wire(&product.arena).ok()?;
        let treatment = antecedent_core::VariableId::from_raw(lowering.treatment);
        let outcome = antecedent_core::VariableId::from_raw(lowering.outcome);
        let mediators = lowering
            .mediators
            .iter()
            .copied()
            .map(antecedent_core::VariableId::from_raw)
            .collect::<Vec<_>>();
        let source = arena.frontdoor_ate(
            treatment,
            outcome,
            &mediators,
            antecedent_core::Value::Float64(f64::from_bits(lowering.active_bits)),
            antecedent_core::Value::Float64(f64::from_bits(lowering.control_bits)),
        );
        let executable = frontdoor_observational_root(&mut arena, treatment, outcome, &mediators);
        let expected_arena = crate::expr_arena_to_wire(&arena).ok()?;
        Some(
            source.raw() == lowering.functional
                && executable.raw() == lowering.executable
                && expected_arena == lowering.arena,
        )
    })()
    .unwrap_or(false);
    let uncertainty_matches = lowering.uncertainty
        == format!(
            "{}:{}",
            program.commitments.interval_method,
            program.commitments.se_kind.as_deref().unwrap_or("unspecified")
        );
    let overlap_matches = contract.inference_binding.as_ref().is_some_and(|binding| {
        if let Some(overlap) = binding.overlap_policy.as_deref() {
            return overlap == lowering.overlap;
        }
        match binding.estimator_spec.as_ref() {
            Some(crate::EstimatorSpecWire::FrontDoorTwoStage(config)) => {
                overlap_policy_tag(&config.overlap) == lowering.overlap
            }
            Some(crate::EstimatorSpecWire::Default(id)) if id == "frontdoor.linear_two_stage" => {
                lowering.overlap == "explicit_override"
            }
            _ => false,
        }
    });
    target_matches
        && program_valid
        && lowering_matches_frontdoor
        && lowering.format == 1
        && lowering.procedure == "linear_path_product"
        && contract.estimator.as_deref() == Some("frontdoor.linear_two_stage")
        && program.commitments.resolved_estimator.as_deref() == Some("frontdoor.linear_two_stage")
        && uncertainty_matches
        && overlap_matches
        && lowering.complete_case_rows > 0
        && !lowering.mediators.is_empty()
        && lowering.treatment != lowering.outcome
        && lowering.mediators.iter().all(|id| *id != lowering.treatment && *id != lowering.outcome)
        && estimand.method == "frontdoor"
}

fn frontdoor_observational_root(
    arena: &mut antecedent_expr::CausalExprArena,
    treatment: antecedent_core::VariableId,
    outcome: antecedent_core::VariableId,
    mediators: &[antecedent_core::VariableId],
) -> antecedent_expr::ExprId {
    use antecedent_expr::{DomainRef, ExprNode, OutcomeExprId};

    let m = arena.intern_var_set(mediators.iter().copied());
    let y = arena.intern_var_set([outcome]);
    let t = arena.intern_var_set([treatment]);
    let m_and_t = arena.intern_var_set(mediators.iter().copied().chain([treatment]));
    let empty = arena.empty_var_set();
    let no_bindings = arena.empty_intervention_set();
    let m_given_t = arena.intern_distribution(m, t, no_bindings, DomainRef::Observational);
    let y_given_m_t = arena.intern_distribution(y, m_and_t, no_bindings, DomainRef::Observational);
    let t_marginal = arena.intern_distribution(t, empty, no_bindings, DomainRef::Observational);
    let inner_factors = arena.intern_list([y_given_m_t, t_marginal]);
    let inner_product = arena.intern(ExprNode::Product(inner_factors));
    let inner_sum = arena.intern(ExprNode::SumOut { variables: t, expr: inner_product });
    let outer_factors = arena.intern_list([m_given_t, inner_sum]);
    let outer_product = arena.intern(ExprNode::Product(outer_factors));
    let outer_sum = arena.intern(ExprNode::SumOut { variables: m, expr: outer_product });
    arena.intern(ExprNode::Expectation {
        function: OutcomeExprId::identity(outcome),
        distribution: outer_sum,
    })
}

fn intervention_bits(wire: &crate::query_wire::InterventionWire, variable: u32) -> Option<u64> {
    match wire {
        crate::query_wire::InterventionWire::Set { variable: target, value }
            if *target == variable =>
        {
            match value {
                crate::query_wire::ValueWire::Float64(value) => Some(value.to_bits()),
                crate::query_wire::ValueWire::Int64(value) => Some((*value as f64).to_bits()),
                _ => None,
            }
        }
        _ => None,
    }
}

fn overlap_policy_tag(policy: &crate::OverlapPolicyWire) -> String {
    match policy {
        crate::OverlapPolicyWire::ExplicitOverride => "explicit_override".into(),
        crate::OverlapPolicyWire::RequireDiagnostics { clip_bits, trim_bits } => format!(
            "require_diagnostics:clip={}:trim={}",
            clip_bits.map_or_else(|| "none".into(), |bits| format!("{bits:016x}")),
            trim_bits.map_or_else(|| "none".into(), |bits| format!("{bits:016x}")),
        ),
    }
}

/// Cross-layer references: every layer must describe the same question,
/// observation contract, graph class, schema, and inference family.
fn verify_layer_links(contract: &AnalysisResultContractWire, unresolved: &mut Vec<Arc<str>>) {
    let target = &contract.target;
    if let Some(identification) = &contract.identification {
        match digest_wire(IdentityDomain::Target, &target.question()) {
            Ok(question) if question.as_bytes() == &identification.target_question => {}
            _ => unresolved.push(Arc::from("identification.target_question")),
        }
        if identification.observation != contract.identities.observation {
            unresolved.push(Arc::from("identification.observation"));
        }
        if identification.graph_class != contract.graph_class {
            unresolved.push(Arc::from("identification.graph_class"));
        }
        if identification.structure_source != contract.structure_source {
            unresolved.push(Arc::from("identification.structure_source"));
        }
        if identification
            .schema_names
            .as_ref()
            .is_some_and(|names| *names != target.schema.variable_names())
        {
            unresolved.push(Arc::from("identification.schema_names"));
        }
        let depends_on: &[u32] = match target.query.target_population() {
            Some(TargetPopulationWire::CustomDistribution { depends_on, .. }) => depends_on,
            _ => &[],
        };
        if identification.population_depends_on != depends_on {
            unresolved.push(Arc::from("identification.population_depends_on"));
        }
    }
    verify_score_reuse_link(contract, unresolved);
    verify_target_weights_link(contract, unresolved);
    if let Some(observation) = &contract.observation {
        if observation.schema != target.schema {
            unresolved.push(Arc::from("observation.schema"));
        }
    }
    if let Some(program) = &contract.program {
        let commitments = &program.commitments;
        if commitments.identifier != contract.identifier {
            unresolved.push(Arc::from("program.commitments.identifier"));
        }
        if commitments.estimator != contract.estimator {
            unresolved.push(Arc::from("program.commitments.estimator"));
        }
        if let Some(binding) = &contract.inference_binding {
            if commitments.inference != binding.inference {
                unresolved.push(Arc::from("inference_binding.inference"));
            }
            if commitments.validation_suite != binding.validation_suite {
                unresolved.push(Arc::from("inference_binding.validation_suite"));
            }
            if binding.bayesian.is_some() != (binding.inference == "bayesian") {
                unresolved.push(Arc::from("inference_binding.bayesian"));
            }
            if commitments.prior_required != binding.bayesian.is_some() {
                unresolved.push(Arc::from("program.commitments.prior_required"));
            }
        }
    }
}

fn verify_checked_aipw(contract: &AnalysisResultContractWire, unresolved: &mut Vec<Arc<str>>) {
    let lowering =
        contract.program.as_ref().and_then(|program| program.checked_aipw_lowering.as_ref());
    if contract.estimator.as_deref() == Some("aipw") && lowering.is_none() {
        unresolved.push(Arc::from("program.checked_aipw_lowering"));
        return;
    }
    let Some(lowering) = lowering else { return };
    let valid_format = lowering.format == 1;
    let valid_method = matches!(
        lowering.se_kind.as_str(),
        "homoskedastic"
            | "hc0"
            | "hc1"
            | "hc2"
            | "hc3"
            | "cluster"
            | "multiway"
            | "newey_west"
            | "panel_cluster_hac"
    );
    let lag_valid = matches!(lowering.se_kind.as_str(), "newey_west" | "panel_cluster_hac")
        == lowering.se_lag.is_some();
    if !valid_format || !valid_method || !lag_valid || lowering.folds != 5 {
        unresolved.push(Arc::from("program.checked_aipw_lowering"));
    }
    if contract.estimator.as_deref() != Some("aipw") {
        unresolved.push(Arc::from("program.checked_aipw_binding"));
    }
    let commitments = contract.program.as_ref().map(|program| &program.commitments);
    if commitments.is_none_or(|commitments| {
        commitments.inference != "frequentist"
            || commitments.prior_required
            || commitments.se_kind.as_deref() != Some(lowering.se_kind.as_str())
            || commitments.interval_method
                != if lowering.bootstrap_replicates > 0 { "bootstrap_se" } else { "analytic_se" }
    }) {
        unresolved.push(Arc::from("program.checked_aipw_uncertainty"));
    }
    if contract.inference_binding.as_ref().is_none_or(|binding| {
        let choices_match = match binding.estimator_spec.as_ref() {
            Some(crate::EstimatorSpecWire::Aipw(config)) => {
                config.bootstrap_replicates == lowering.bootstrap_replicates
                    && config.se_kind.as_deref() == Some(lowering.se_kind.as_str())
                    && matches!(
                        config.overlap,
                        crate::OverlapPolicyWire::RequireDiagnostics { trim_bits: None, .. }
                    )
            }
            Some(crate::EstimatorSpecWire::Default(estimator)) if estimator == "aipw" => {
                lowering.se_kind == "homoskedastic"
                    && binding.bootstrap_replicates == lowering.bootstrap_replicates
            }
            None => binding.bootstrap_replicates == lowering.bootstrap_replicates,
            _ => false,
        };
        binding.inference != "frequentist" || binding.bayesian.is_some() || !choices_match
    }) {
        unresolved.push(Arc::from("program.checked_aipw_uncertainty"));
    }
    let query_matches = match &contract.target.query {
        crate::CausalQueryWire::AverageEffect {
            treatment,
            outcome,
            control,
            active,
            target_population: crate::TargetPopulationWire::AllObserved,
            outcome_functional: crate::query_wire::OutcomeFunctionalWire::Mean,
            effect_modifiers,
        } => {
            *treatment == lowering.treatment
                && *outcome == lowering.outcome
                && effect_modifiers.is_empty()
                && is_set_value(control, lowering.treatment, 0.0)
                && is_set_value(active, lowering.treatment, 1.0)
                && lowering.population == "all_observed"
        }
        _ => false,
    };
    let estimand_matches = contract.identification_product.as_ref().is_some_and(|product| {
        product.arena.nodes.get(lowering.functional as usize).is_some()
            && product.estimands.iter().any(|estimand| {
                estimand.functional == lowering.functional
                    && estimand.adjustment_set == lowering.adjustment
                    && estimand.instruments.is_empty()
                    && estimand.mediators.is_empty()
            })
    });
    let adjustments_unique =
        lowering.adjustment.iter().copied().collect::<std::collections::BTreeSet<_>>().len()
            == lowering.adjustment.len()
            && !lowering.adjustment.contains(&lowering.treatment)
            && !lowering.adjustment.contains(&lowering.outcome);
    if !query_matches
        || !estimand_matches
        || !adjustments_unique
        || lowering.procedure != "cross_fitted_logistic_ols"
    {
        unresolved.push(Arc::from("program.checked_aipw_binding"));
    }
}

fn is_set_value(intervention: &crate::InterventionWire, variable: u32, value: f64) -> bool {
    matches!(intervention,
        crate::InterventionWire::Set { variable: actual, value: crate::ValueWire::Float64(actual_value) }
            if *actual == variable && actual_value.to_bits() == value.to_bits())
}

fn verify_checked_aipw_rows(contract: &AnalysisResultContractWire, unresolved: &mut Vec<Arc<str>>) {
    let row_binding = contract.checked_aipw_rows.as_ref();
    if row_binding.is_some() != contract.identities.checked_aipw_rows.is_some() {
        unresolved.push(Arc::from("checked_aipw.row_binding"));
        return;
    }
    let Some(row_binding) = row_binding else {
        if contract.estimator.as_deref() == Some("aipw")
            || contract
                .program
                .as_ref()
                .is_some_and(|program| program.checked_aipw_lowering.is_some())
        {
            unresolved.push(Arc::from("checked_aipw.row_binding"));
        }
        return;
    };
    let digest_matches = checked_aipw_rows_digest(row_binding)
        .is_ok_and(|digest| Some(digest) == contract.identities.checked_aipw_rows);
    let snapshot_matches = row_binding.data_snapshot == contract.identities.data_snapshot
        && contract.data_snapshot.as_ref().is_some_and(|snapshot| {
            row_binding.rows.len() as u64 <= snapshot.row_count
                && row_binding.rows.iter().all(|row| u64::from(*row) < snapshot.row_count)
        });
    let ordered_unique =
        !row_binding.rows.is_empty() && row_binding.rows.windows(2).all(|pair| pair[0] < pair[1]);
    if row_binding.format != 1
        || !digest_matches
        || !snapshot_matches
        || !ordered_unique
        || contract.estimator.as_deref() != Some("aipw")
    {
        unresolved.push(Arc::from("checked_aipw.row_binding"));
    }
}

/// Score-reuse layer: present with its digest, fitted on this snapshot, and
/// keyed by this identification.
fn verify_score_reuse_link(contract: &AnalysisResultContractWire, unresolved: &mut Vec<Arc<str>>) {
    if contract.score_reuse.is_some() != contract.identities.score_reuse.is_some() {
        unresolved.push(Arc::from("identities.score_reuse"));
        return;
    }
    let Some(score) = &contract.score_reuse else {
        return;
    };
    require_payload_digest(
        unresolved,
        "identities.score_reuse",
        Some(score),
        contract.identities.score_reuse.as_ref(),
        score_reuse_digest,
    );
    require_nested_digest(
        unresolved,
        "score_reuse.data_snapshot",
        Some(score.data_snapshot),
        Some(&contract.identities.data_snapshot),
    );
    require_nested_digest(
        unresolved,
        "score_reuse.identification",
        score.identification,
        Some(&contract.identities.identification),
    );
}

/// Row-weight layer: re-derive the target-weights identity from the weights the
/// section carries, and check that the target population, the snapshot and the
/// score table all name it.
fn verify_target_weights_link(
    contract: &AnalysisResultContractWire,
    unresolved: &mut Vec<Arc<str>>,
) {
    let row_weights = match contract.target.query.target_population() {
        Some(TargetPopulationWire::RowWeights { weights, depends_on }) => {
            Some((*weights, depends_on.clone()))
        }
        _ => None,
    };
    if contract.target_weights.is_some() != contract.identities.target_weights.is_some() {
        unresolved.push(Arc::from("identities.target_weights"));
        return;
    }
    let Some(section) = &contract.target_weights else {
        if row_weights.is_some() {
            unresolved.push(Arc::from("target.population"));
        }
        return;
    };
    let identity = &section.identity;
    require_payload_digest(
        unresolved,
        "identities.target_weights",
        Some(identity),
        contract.identities.target_weights.as_ref(),
        target_weights_digest,
    );
    let rows = PayloadDigestWire::f64s("target_weights.rows", &section.values);
    let values_ok = rows.len == identity.row_count
        && rows.digest == identity.weights
        && section.values.iter().all(|weight| weight.is_finite() && *weight >= 0.0);
    if !values_ok {
        unresolved.push(Arc::from("target_weights.values"));
    }
    require_nested_digest(
        unresolved,
        "target_weights.data_snapshot",
        Some(identity.data_snapshot),
        Some(&contract.identities.data_snapshot),
    );
    require_nested_digest(
        unresolved,
        "target_weights.score_reuse",
        Some(identity.score_reuse),
        contract.identities.score_reuse.as_ref(),
    );
    if contract.score_reuse.as_ref().is_some_and(|score| score.row_index.len != identity.row_count)
    {
        unresolved.push(Arc::from("target_weights.row_count"));
    }
    match row_weights {
        Some((weights, depends_on))
            if Some(weights) == contract.identities.target_weights
                && depends_on == identity.depends_on => {}
        _ => unresolved.push(Arc::from("target.population")),
    }
}

#[derive(Serialize)]
struct SealWire<'a> {
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

fn contract_seal_of(contract: &AnalysisResultContractWire) -> Result<[u8; 32], IoError> {
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
/// and the identification certificate.
///
/// # Errors
///
/// CBOR encode failure.
pub fn result_digest(body: &AnalysisResultWire) -> Result<[u8; 32], IoError> {
    Ok(payload_digest("analysis_result.body", &to_cbor(body)?))
}

/// Claim kind implied by a result body and its identification slot.
///
/// The single rule shared by the producer and the independent consumer.
#[must_use]
pub fn claim_kind_name(
    body: &AnalysisResultWire,
    identification: Option<&IdentificationSlotWire>,
) -> &'static str {
    if body.response.is_some() {
        return "response";
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

/// Refuter ids whose outcome speaks to empirical support (overlap / positivity).
fn is_support_refuter(refuter: &str) -> bool {
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
    let masses = [identified, unidentified, unevaluable, incomplete_search];
    if masses.iter().any(|mass| !mass.is_finite() || *mass < 0.0 || *mass > 1.0) {
        return Err(IoError::Convert("mixture masses must be finite and inside [0, 1]".into()));
    }
    let sum = identified + unidentified + unevaluable + incomplete_search;
    if (sum - 1.0).abs() > 1e-12 {
        return Err(IoError::Convert(format!("mixture masses must sum to 1 (got {sum})")));
    }
    Ok(())
}

fn require_nested_digest(
    unresolved: &mut Vec<Arc<str>>,
    label: &'static str,
    nested: Option<[u8; 32]>,
    advertised: Option<&[u8; 32]>,
) {
    let agrees = match (nested, advertised) {
        (Some(got), Some(want)) => got == *want,
        (None, None) => true,
        (None, Some(_)) | (Some(_), None) => false,
    };
    if !agrees {
        unresolved.push(Arc::from(label));
    }
}

fn claim_id_matches(
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

fn claim_value_matches_body(claim: &ClaimSectionWire, body: &AnalysisResultWire) -> bool {
    match (claim.value_bits, body.estimate) {
        (Some(bits), Some(estimate)) if bits == estimate.to_bits() => true,
        (None, None) => true,
        _ => false,
    }
}

fn identification_product_matches_body(
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

fn cbor_eq<T: Serialize>(left: &T, right: &T) -> bool {
    match (crate::to_cbor(left), crate::to_cbor(right)) {
        (Ok(left), Ok(right)) => left == right,
        _ => false,
    }
}

fn require_present<T>(unresolved: &mut Vec<Arc<str>>, label: &'static str, payload: Option<&T>) {
    if payload.is_none() {
        unresolved.push(Arc::from(label));
    }
}

fn require_slot<T>(unresolved: &mut Vec<Arc<str>>, name: &'static str, slot: &SlotSectionWire<T>) {
    if validate_slot(name, slot).is_err() {
        unresolved.push(Arc::from(format!("reasoning.{name}")));
    }
}

fn require_payload_digest<T>(
    unresolved: &mut Vec<Arc<str>>,
    label: &'static str,
    payload: Option<&T>,
    advertised: Option<&[u8; 32]>,
    digest: impl FnOnce(&T) -> Result<SemanticDigest, IoError>,
) {
    let Some(payload) = payload else {
        return;
    };
    match (digest(payload), advertised) {
        (Ok(got), Some(want)) if got.as_bytes() == want => {}
        _ => unresolved.push(Arc::from(label)),
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
    let contract = match decode_analysis_result_contract(&artifact) {
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

fn accept_contract(
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
            Arc::from("store"),
            Arc::from("forward"),
            Arc::from("read_body"),
            Arc::from("verify_contract"),
            Arc::from("inspect_claim"),
        ]),
        (true, false) => Arc::from([
            Arc::from("store"),
            Arc::from("forward"),
            Arc::from("read_body"),
            Arc::from("verify_contract"),
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

fn validate_slot<T>(name: &str, slot: &SlotSectionWire<T>) -> Result<(), IoError> {
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

fn storage_ops() -> Arc<[Arc<str>]> {
    Arc::from([Arc::from("store"), Arc::from("forward"), Arc::from("read_body")])
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
    if !profile.understands("verify_contract") || !profile.understands("inspect_claim") {
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
            [Arc::from("bytes"), Arc::from("read_body")],
            omitted,
            consumed.acceptance.unresolved.clone(),
            [Arc::from("accept_as_claim")],
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

fn unknown_required_features(
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
fn receipt_input(consumed: &AnalysisResultConsumption) -> SemanticDigest {
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
        claim_id: Some(digest_hex(&claim.claim_id)),
        kind: Some(claim.kind.clone()),
        value,
        identification_domain: domain(&claim.identification_domain),
        support_domain: domain(&claim.support_domain),
        evaluated_domain: domain(&claim.evaluated_domain),
        recognized: consumed.acceptance.recognized,
        accepts_as_claim: consumed.acceptance.accepts_as_claim(),
    }
}

fn unsupported_host(consumed: &AnalysisResultConsumption, accepts: bool) -> ClaimHostProjection {
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

/// Hex-encode a digest for host/Python reports.
#[must_use]
pub fn digest_hex(bytes: &[u8; 32]) -> String {
    SemanticDigest::from_bytes(*bytes).to_hex()
}

#[cfg(test)]
mod tests {
    use antecedent_core::{
        AverageEffectQuery, CausalQuery, CausalSchemaBuilder, IDENTITY_FORMAT, MeasurementSpec,
        RoleHint, SmallRoleSet, ValueType, VariableId,
    };

    use super::*;
    use crate::expr_wire::ExprArenaWire;
    use crate::query_wire::causal_query_to_wire;
    use crate::{
        CausalQueryWire, IdentificationResultWire, encode_analysis_result_artifact,
        encode_analysis_result_artifact_with_contract, schema_to_wire, to_cbor,
    };

    #[test]
    fn checked_response_grid_rejects_reordered_member_values() {
        use crate::{
            CheckedFunctionalResponseGridWire, CheckedFunctionalResponseMemberWire,
            FunctionalProgramWire, IdentificationProductWire, ResponseIdentificationWire,
            ResponseValueWire,
        };
        use antecedent_core::{ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery};

        let query = CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome: VariableId::from_raw(1),
            treatment: ContinuousDomain::new(
                VariableId::from_raw(0),
                GridSpec::Values(vec![0.0, 1.0].into()),
            ),
        }));
        let target = causal_query_to_wire(&query).unwrap();
        let empty_arena = ExprArenaWire {
            derivations: Vec::new(),
            var_sets: Vec::new(),
            interventions: Vec::new(),
            lists: Vec::new(),
            nodes: Vec::new(),
        };
        let member = CheckedFunctionalResponseMemberWire {
            grid_value_bits: 1.0f64.to_bits(), // Tampered order: this should be 0.0.
            query: CausalQueryWire::Response(
                crate::response_query_to_wire(&ResponseQuery::new(
                    ResponseFunctional::InterventionResponse {
                        outcome: VariableId::from_raw(1),
                        interventions: vec![antecedent_core::Intervention::set(
                            VariableId::from_raw(0),
                            antecedent_core::Value::f64(0.0),
                        )]
                        .into(),
                    },
                ))
                .unwrap(),
            ),
            identification: IdentificationProductWire {
                format: IDENTITY_FORMAT,
                status: "identified".into(),
                estimands: Vec::new(),
                arena: empty_arena.clone(),
                derivation_rules: Vec::new(),
                required_assumptions: Vec::new(),
                hedge: false,
                search_capped: false,
                envelope: None,
            },
            program: FunctionalProgramWire {
                arena: empty_arena,
                source: 0,
                executable: 0,
                variables: Vec::new(),
            },
        };
        let family = CheckedFunctionalResponseGridWire {
            format: 1,
            treatment: 0,
            outcome: 1,
            members: vec![member.clone(), member],
        };
        let reported = ResponseIdentificationWire::PointIdentified(ResponseValueWire::Surface {
            grid: vec![0.0, 1.0],
            dimension: 1,
            mean: vec![0.2, 0.8],
        });
        let laws = crate::DistributionFactorLawsWire {
            format: 1,
            provider: "empirical_table".into(),
            source_rows: 2,
            complete_case_rows: 2,
            missing_row_policy: "joint_complete_case".into(),
            domains: Vec::new(),
            requirements: Vec::new(),
            factors: Vec::new(),
        };
        assert_eq!(
            replay_functional_response_grid(&target, &family, &reported, &laws),
            Err("program.functional_effect_response_grid.member_value")
        );
    }

    fn schema_and_query() -> (antecedent_core::CausalSchema, CausalQuery) {
        let mut builder = CausalSchemaBuilder::new();
        builder
            .add_variable(
                "t",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        builder
            .add_variable(
                "y",
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
        let schema = builder.build().unwrap();
        let query = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
        ));
        (schema, query)
    }

    fn fixture_body() -> (AnalysisResultWire, TargetIdentityWire, Vec<String>) {
        let (schema, query) = schema_and_query();
        let query_wire = causal_query_to_wire(&query).unwrap();
        let body = AnalysisResultWire {
            query: query_wire.clone(),
            identification: IdentificationResultWire {
                status: "nonparametrically_identified".into(),
                query: query_wire.clone(),
                estimands: Vec::new(),
                arena: ExprArenaWire {
                    derivations: Vec::new(),
                    var_sets: Vec::new(),
                    interventions: Vec::new(),
                    lists: Vec::new(),
                    nodes: Vec::new(),
                },
                derivation: Vec::new(),
                required_assumptions: Vec::new(),
                diagnostics: Vec::new(),
                candidates_examined: 0,
                sets_returned: 0,
                hedge: None,
            },
            identification_variables: None,
            temporal_identification: Vec::new(),
            estimate: Some(2.0),
            interventional_distribution: None,
            standard_error: Some(0.1),
            interval_lower: None,
            interval_upper: None,
            assumptions: Vec::new(),
            diagnostics: Vec::new(),
            refutations: Vec::new(),
            response: None,
            posterior_artifact: None,
            mediation_grid: None,
            structural_response: None,
            unit_effects: None,
            cate: None,
            fitted_effect: None,
            cate_se: None,
            cate_leaf_dispersion: None,
            outcome_oof_r2: None,
            treatment_oof_logloss: None,
            crossfit_folds: None,
            crossfit_seed: None,
            learner_provenance: Vec::new(),
        };
        let target = TargetIdentityWire {
            format: IDENTITY_FORMAT,
            schema: schema_to_wire(&schema),
            query: query_wire,
        };
        (body, target, schema.variables().iter().map(|v| v.name.to_string()).collect())
    }

    #[test]
    fn distribution_consumer_names_missing_factor_laws_as_replay_dependency() {
        let (body, target, names) = fixture_body();
        let mut contract = contract_for(target, &body);
        contract.estimator = Some("functional.distribution".into());
        let unresolved = verify_contract_against_body(
            &AnalysisResultHeader { variable_names: names },
            &body,
            &contract,
        );
        assert!(
            unresolved
                .iter()
                .any(|reason| { reason.as_ref() == "dependencies.distribution_factor_laws" })
        );
    }

    #[allow(clippy::too_many_lines)]
    fn contract_for(
        target: TargetIdentityWire,
        body: &AnalysisResultWire,
    ) -> AnalysisResultContractWire {
        use antecedent_graph::{Dag, DenseNodeId};

        use crate::identity::{
            InferentialCommitmentsWire, ObservationIdentityWire, ObservationOptionsWire,
            dag_identity, data_snapshot_digest, execution_digest, identification_digest,
            identification_product_digest_wire, inference_binding_digest, program_digest,
        };

        let target_digest = digest_wire(IdentityDomain::Target, &target).unwrap();
        let question_digest = digest_wire(IdentityDomain::Target, &target.question()).unwrap();
        let observation = ObservationIdentityWire {
            format: IDENTITY_FORMAT,
            schema: target.schema.clone(),
            observation: Vec::new(),
        };
        let observation_digest = digest_wire(IdentityDomain::Observation, &observation).unwrap();
        let mut dag = Dag::with_variables(2);
        dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        let identification = IdentificationIdentityWire {
            format: IDENTITY_FORMAT,
            target_question: *question_digest.as_bytes(),
            population_depends_on: Vec::new(),
            rd_config: None,
            class_prior: None,
            transport: None,
            graph_class: "Dag".into(),
            structure_source: "explicit".into(),
            accepted_version: 0,
            algorithm_id: None,
            schema_names: Some(target.schema.variable_names()),
            graph: dag_identity(&dag).unwrap(),
            observation: *observation_digest.as_bytes(),
        };
        let identification_digest = identification_digest(&identification).unwrap();
        let identification_product = IdentificationProductWire {
            format: IDENTITY_FORMAT,
            status: body.identification.status.clone(),
            estimands: body.identification.estimands.clone(),
            arena: body.identification.arena.clone(),
            derivation_rules: body
                .identification
                .derivation
                .iter()
                .map(|step| step.rule.clone())
                .collect(),
            required_assumptions: body.identification.required_assumptions.clone(),
            hedge: false,
            search_capped: false,
            envelope: None,
        };
        let product_digest = identification_product_digest_wire(&identification_product).unwrap();
        let commitments = InferentialCommitmentsWire {
            format: IDENTITY_FORMAT,
            estimator: Some("g_computation".into()),
            resolved_estimator: Some("g_computation".into()),
            identifier: Some("backdoor".into()),
            inference: "frequentist".into(),
            validation_suite: None,
            interval_method: "analytic_se".into(),
            se_kind: Some("hc1".into()),
            prior_required: false,
        };
        let program = ProgramIdentityWire {
            format: IDENTITY_FORMAT,
            target: *target_digest.as_bytes(),
            identification: *identification_digest.as_bytes(),
            identification_product: Some(*product_digest.as_bytes()),
            completion_budget: None,
            commitments,
            functional_program: None,
            checked_aipw_lowering: None,
            checked_frontdoor_lowering: None,
            checked_iv_lowering: None,
            checked_linear_adjustment_lowering: None,
            checked_functional_response_grid: None,
            checked_nested_counterfactual: None,
        };
        let program_digest = program_digest(&program).unwrap();
        let inference_binding = InferenceBindingWire {
            format: IDENTITY_FORMAT,
            inference: "frequentist".into(),
            bootstrap_replicates: 0,
            bayesian: None,
            validation_suite: None,
            overlap_policy: None,
            estimator_spec: None,
            response_options: None,
            observation_options: ObservationOptionsWire {
                selected_correction: "aipw".into(),
                observation_probability_floor_bits: 0.01f64.to_bits(),
                censoring_survival_floor_bits: 0.01f64.to_bits(),
                crossfit_folds: 5,
            },
            split: None,
        };
        let inference_digest = inference_binding_digest(&inference_binding).unwrap();
        let data_snapshot = DataSnapshotIdentityWire {
            format: IDENTITY_FORMAT,
            observation: *observation_digest.as_bytes(),
            modality: "tabular".into(),
            regularity: None,
            row_count: 8,
            unit_count: None,
            partitions: Vec::new(),
            interference: None,
            distribution_factor_laws: None,
            nested_counterfactual_fit: None,
        };
        let snapshot_digest = data_snapshot_digest(&data_snapshot).unwrap();
        let execution = crate::execution_identity_from_context(
            &antecedent_core::ExecutionContext::for_tests(1),
        );
        let execution_digest = execution_digest(&execution).unwrap();
        let identities = ContractIdentitiesWire {
            target: *target_digest.as_bytes(),
            identification: *identification_digest.as_bytes(),
            identification_product: Some(*product_digest.as_bytes()),
            program: *program_digest.as_bytes(),
            inference_binding: *inference_digest.as_bytes(),
            observation: *observation_digest.as_bytes(),
            data_snapshot: *snapshot_digest.as_bytes(),
            execution: Some(*execution_digest.as_bytes()),
            score_reuse: None,
            target_weights: None,
            checked_aipw_rows: None,
        };
        let mut contract = AnalysisResultContractWire {
            format: CONTRACT_SECTION_FORMAT,
            identities,
            seal: [0; 32],
            target,
            reasoning: ReasoningSectionWire {
                identification: SlotSectionWire {
                    value: Some(IdentificationSlotWire {
                        status: "nonparametrically_identified".into(),
                        identified_mass: 1.0,
                        unidentified_mass: 0.0,
                        unevaluable_mass: 0.0,
                        incomplete_search_mass: 0.0,
                        full_mass_scope: true,
                        search_capped: false,
                        weight_basis: None,
                    }),
                    unavailable: None,
                },
                support: SlotSectionWire {
                    value: Some(SupportSlotWire {
                        matrix_status: "licensed".into(),
                        matrix_coordinate: Some(
                            "AverageEffect:Dag:explicit:Frequentist:none".into(),
                        ),
                        empirical: "unavailable:not_evaluated".into(),
                    }),
                    unavailable: None,
                },
                uncertainty: SlotSectionWire {
                    value: None,
                    unavailable: Some("execution_specific".into()),
                },
                assumptions: SlotSectionWire {
                    value: Some(AssumptionSlotWire { obligations: Vec::new() }),
                    unavailable: None,
                },
            },
            graph_class: "Dag".into(),
            structure_source: "explicit".into(),
            identifier: Some("backdoor".into()),
            estimator: Some("g_computation".into()),
            claim: None,
            identification: Some(identification),
            identification_product: Some(identification_product),
            program: Some(program),
            inference_binding: Some(inference_binding),
            observation: Some(observation),
            data_snapshot: Some(data_snapshot),
            execution: Some(execution),
            score_reuse: None,
            target_weights: None,
            checked_aipw_rows: None,
        };
        let claim = ClaimSectionWire {
            claim_id: [0; 32],
            kind: "point".into(),
            value_bits: body.estimate.map(f64::to_bits),
            execution: Some(*execution_digest.as_bytes()),
            identification_domain: "identified".into(),
            support_domain: "unknown".into(),
            evaluated_domain: "evaluated".into(),
            calibration: crate::calibration::calibration_slots(&[
                crate::calibration::CalibrationBasisWire {
                    key: crate::calibration::CalibrationKeyWire {
                        query: "AverageEffect".into(),
                        graph_class: "Dag".into(),
                        structure: "fixed".into(),
                        modality: "tabular".into(),
                        inference: "Frequentist".into(),
                        estimator: "g_computation".into(),
                        interval_method: "analytic_se".into(),
                        se_kind: "hc1".into(),
                        dependence: "iid".into(),
                        posterior: String::new(),
                        functional: "all_observed.mean".into(),
                        level: 0.95,
                        identification: "point".into(),
                    },
                    scope: crate::calibration::CalibrationScopeWire {
                        row_count: 8,
                        replicates_ok: None,
                        posterior_draws: None,
                        unidentified_mass: 0.0,
                    },
                },
            ]),
            attested: Vec::new(),
        };
        contract.claim = Some(claim);
        seal_claim(&mut contract, body);
        contract
    }

    /// Recompute the seal and claim id after a deliberate fixture edit.
    fn seal_claim(contract: &mut AnalysisResultContractWire, body: &AnalysisResultWire) {
        contract.seal = super::contract_seal_of(contract).unwrap();
        let Some(claim) = contract.claim.clone() else {
            return;
        };
        let result = result_digest(body).unwrap();
        let claim_id = *claim_digest(&ClaimIdentityWire::new(contract.seal, &claim, result))
            .unwrap()
            .as_bytes();
        if let Some(claim) = contract.claim.as_mut() {
            claim.claim_id = claim_id;
        }
    }

    fn to_bytes(
        body: &AnalysisResultWire,
        names: Vec<String>,
        contract: Option<&AnalysisResultContractWire>,
    ) -> Vec<u8> {
        let artifact =
            encode_analysis_result_artifact_with_contract(body, names, "ate", contract).unwrap();
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).unwrap();
        bytes
    }

    fn replace_contract_section(
        body: &AnalysisResultWire,
        names: Vec<String>,
        contract: &AnalysisResultContractWire,
    ) -> Vec<u8> {
        let mut artifact = encode_analysis_result_artifact(body, names, "mutated").unwrap();
        let (descriptor, section) = crate::pack_section_shared(
            CONTRACT_SECTION,
            "application/cbor",
            to_cbor(contract).unwrap().into(),
            crate::CompressPolicy::Auto,
        );
        artifact.manifest.sections.push(descriptor);
        artifact.sections.push(section);
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).unwrap();
        bytes
    }

    #[test]
    fn old_artifact_remains_readable_but_is_not_verified() {
        let (body, _, names) = fixture_body();
        let artifact = encode_analysis_result_artifact(&body, names, "legacy").unwrap();
        let mut bytes = Vec::new();
        artifact.write_to(&mut bytes).unwrap();
        let (_, _, decoded) = crate::decode_analysis_result_artifact(&bytes).unwrap();
        assert_eq!(decoded, body);
        let consumed = consume_analysis_result(&bytes).unwrap();
        assert!(consumed.contract.is_none());
        assert!(!consumed.acceptance.accepts_as_verified_program());
        assert_eq!(consumed.acceptance.restriction.as_deref(), Some("missing_contract_section"));
        assert!(consumed.acceptance.supported_operations.iter().any(|op| &**op == "read_body"));
    }

    #[test]
    fn contracted_artifact_is_accepted_from_bytes_alone() {
        let (body, target, names) = fixture_body();
        let contract = contract_for(target, &body);
        let consumed = consume_analysis_result(&to_bytes(&body, names, Some(&contract))).unwrap();
        assert!(consumed.acceptance.accepts_as_verified_program());
        assert_eq!(consumed.contract.as_ref().map(|c| c.graph_class.as_str()), Some("Dag"));
        assert_eq!(consumed.body.estimate, Some(2.0));
    }

    #[test]
    fn nested_identification_target_must_match_advertised_target() {
        let (body, target, names) = fixture_body();
        let mut contract = contract_for(target, &body);
        let identification = contract.identification.as_mut().expect("identification");
        identification.target_question = [0u8; 32];
        let consumed =
            consume_analysis_result(&replace_contract_section(&body, names, &contract)).unwrap();
        assert!(!consumed.acceptance.accepts_as_verified_program());
        assert!(
            consumed
                .acceptance
                .unresolved
                .iter()
                .any(|item| &**item == "identities.identification"),
            "{:?}",
            consumed.acceptance.unresolved
        );
    }

    #[test]
    fn invented_reasoning_masses_are_not_verified() {
        let (body, target, names) = fixture_body();
        let mut contract = contract_for(target, &body);
        let slot = contract.reasoning.identification.value.as_mut().expect("slot");
        slot.identified_mass = 1.0;
        slot.unidentified_mass = 0.3;
        let consumed =
            consume_analysis_result(&replace_contract_section(&body, names, &contract)).unwrap();
        assert!(!consumed.acceptance.accepts_as_verified_program());
        assert!(
            consumed
                .acceptance
                .unresolved
                .iter()
                .any(|item| &**item == "reasoning.identification.masses"),
            "{:?}",
            consumed.acceptance.unresolved
        );
    }

    #[test]
    fn tampered_target_identity_is_not_promoted() {
        let (body, target, names) = fixture_body();
        let mut contract = contract_for(target, &body);
        contract.identities.target[0] ^= 0xff;
        let consumed =
            consume_analysis_result(&replace_contract_section(&body, names, &contract)).unwrap();
        assert!(!consumed.acceptance.accepts_as_verified_program());
        assert!(consumed.acceptance.unresolved.iter().any(|item| &**item == "identities.target"));
    }

    #[test]
    fn body_query_must_match_contract_target() {
        let (mut body, target, names) = fixture_body();
        let contract = contract_for(target, &body);
        if let CausalQueryWire::AverageEffect { outcome, .. } = &mut body.query {
            *outcome = 0;
        }
        if let CausalQueryWire::AverageEffect { outcome, .. } = &mut body.identification.query {
            *outcome = 0;
        }
        let consumed =
            consume_analysis_result(&replace_contract_section(&body, names, &contract)).unwrap();
        assert!(!consumed.acceptance.accepts_as_verified_program());
        assert!(consumed.acceptance.unresolved.iter().any(|item| &**item == "body.query"));
    }

    #[test]
    fn attested_digests_without_payloads_are_not_verified() {
        let (body, target, names) = fixture_body();
        let mut contract = contract_for(target, &body);
        contract.identification = None;
        contract.identification_product = None;
        contract.program = None;
        contract.inference_binding = None;
        contract.observation = None;
        contract.data_snapshot = None;
        contract.execution = None;
        let consumed =
            consume_analysis_result(&replace_contract_section(&body, names, &contract)).unwrap();
        assert!(!consumed.acceptance.accepts_as_verified_program());
        assert!(consumed.acceptance.unresolved.iter().any(|item| item.starts_with("identities.")));
    }

    #[test]
    fn tampered_identification_product_is_not_promoted() {
        let (body, target, names) = fixture_body();
        let mut contract = contract_for(target, &body);
        if let Some(digest) = contract.identities.identification_product.as_mut() {
            digest[0] ^= 0xff;
        }
        let consumed =
            consume_analysis_result(&replace_contract_section(&body, names, &contract)).unwrap();
        assert!(!consumed.acceptance.accepts_as_verified_program());
        assert!(
            consumed
                .acceptance
                .unresolved
                .iter()
                .any(|item| &**item == "identities.identification_product")
        );
    }

    #[test]
    fn unknown_contract_format_needs_a_newer_reader() {
        let (body, target, names) = fixture_body();
        let mut contract = contract_for(target, &body);
        contract.format = 99;
        assert!(validate_contract_section(&contract).is_err());
        let consumed =
            consume_analysis_result(&replace_contract_section(&body, names, &contract)).unwrap();
        assert!(!consumed.acceptance.recognized);
        assert_eq!(consumed.acceptance.restriction.as_deref(), Some("unknown_contract_format"));
    }

    #[test]
    fn claim_host_projection_keeps_null_zero_unknown_distinct() {
        let (mut body, target, names) = fixture_body();
        body.estimate = Some(0.0);
        let mut zero = contract_for(target.clone(), &body);
        if let Some(claim) = zero.claim.as_mut() {
            claim.value_bits = Some(0.0f64.to_bits());
        }
        seal_claim(&mut zero, &body);
        let consumed_zero =
            consume_analysis_result(&to_bytes(&body, names.clone(), Some(&zero))).unwrap();
        let proj_zero = project_claim_host(&consumed_zero);
        assert_eq!(proj_zero.value, serde_json::Value::from(0.0));
        assert_ne!(proj_zero.value, serde_json::Value::Null);
        assert_ne!(proj_zero.value, serde_json::Value::String("unknown".into()));

        body.estimate = None;
        let mut absent = contract_for(target.clone(), &body);
        if let Some(claim) = absent.claim.as_mut() {
            claim.value_bits = None;
        }
        seal_claim(&mut absent, &body);
        let consumed_absent =
            consume_analysis_result(&to_bytes(&body, names.clone(), Some(&absent))).unwrap();
        let proj_absent = project_claim_host(&consumed_absent);
        assert_eq!(proj_absent.value, serde_json::Value::Null);

        let consumed_old = consume_analysis_result(&{
            let artifact = encode_analysis_result_artifact(&body, names, "legacy").unwrap();
            let mut bytes = Vec::new();
            artifact.write_to(&mut bytes).unwrap();
            bytes
        })
        .unwrap();
        let proj_old = project_claim_host(&consumed_old);
        assert_eq!(proj_old.value, serde_json::Value::String("unsupported".into()));
        assert_ne!(proj_old.value, serde_json::Value::Null);
        assert_ne!(proj_old.value, serde_json::Value::from(0.0));
    }

    #[test]
    fn verified_program_without_claim_is_not_an_accepted_claim() {
        let (body, target, names) = fixture_body();
        let mut contract = contract_for(target, &body);
        contract.claim = None;
        contract.execution = None;
        contract.identities.execution = None;
        seal_claim(&mut contract, &body);
        let bytes = to_bytes(&body, names, Some(&contract));
        let consumed = consume_analysis_result(&bytes).unwrap();
        assert!(consumed.acceptance.accepts_as_verified_program());
        assert!(!consumed.acceptance.accepts_as_claim());
        assert!(
            !consumed.acceptance.supported_operations.iter().any(|op| &**op == "inspect_claim")
        );
        let (accepted, receipt) =
            accept_claim(&bytes, &antecedent_core::ConsumerProfile::full()).unwrap();
        assert!(!accepted.acceptance.accepts_as_claim());
        assert_ne!(receipt.input.as_bytes(), &[0u8; 32]);
        assert_eq!(receipt.input.as_bytes(), &payload_digest("analysis_result.bytes", &bytes));
        assert_eq!(&*receipt.rule, "restricted_accept");
        assert!(receipt.omitted.iter().any(|field| &**field == "claim"));
        assert!(!receipt.equivalent_claim());
        let host = project_claim_host(&accepted);
        assert!(host.claim_id.is_none());
        assert!(!host.accepts_as_claim);
    }

    #[test]
    fn accept_claim_unknown_feature_is_storage_not_acceptance() {
        let (body, target, names) = fixture_body();
        let mut contract = contract_for(target, &body);
        // A posterior-weighted mixture: the kind follows from the weight basis.
        if let Some(slot) = contract.reasoning.identification.value.as_mut() {
            slot.weight_basis = Some("posterior_probability".into());
        }
        if let Some(claim) = contract.claim.as_mut() {
            claim.kind = "mixture".into();
            claim.value_bits = None;
        }
        seal_claim(&mut contract, &body);
        let bytes = to_bytes(&body, names, Some(&contract));
        let (full, lossless) =
            accept_claim(&bytes, &antecedent_core::ConsumerProfile::full()).unwrap();
        assert!(full.acceptance.accepts_as_claim());
        assert!(lossless.equivalent_claim());
        let (restricted, receipt) =
            accept_claim(&bytes, &antecedent_core::ConsumerProfile::restricted()).unwrap();
        assert!(!restricted.acceptance.accepts_as_claim());
        assert_eq!(restricted.acceptance.restriction.as_deref(), Some("unknown_required_feature"));
        assert!(restricted.acceptance.supported_operations.iter().any(|op| &**op == "forward"));
        assert!(!receipt.equivalent_claim());
        let (_forward, forwarded) =
            accept_claim(&bytes, &antecedent_core::ConsumerProfile::forwarding()).unwrap();
        let chained = receipt.chain(&forwarded);
        assert!(!chained.equivalent_claim());
        assert!(chained.omitted.iter().any(|field| &**field == "required_semantics"));
    }

    fn unresolved_after(
        body: &AnalysisResultWire,
        names: &[String],
        contract: &AnalysisResultContractWire,
    ) -> Vec<String> {
        let consumed =
            consume_analysis_result(&replace_contract_section(body, names.to_vec(), contract))
                .unwrap();
        assert!(!consumed.acceptance.accepts_as_claim(), "tampered artifact was accepted");
        consumed.acceptance.unresolved.iter().map(ToString::to_string).collect()
    }

    fn refutation(refuter: &str, passed: bool) -> RefutationReportWire {
        RefutationReportWire {
            refuter: refuter.into(),
            original_ate: 2.0,
            refuted_ate: 2.0,
            comparison: 0.0,
            informative: true,
            passed,
            failure_condition: None,
            replicates: 0,
        }
    }

    #[test]
    fn untouched_fixture_verifies() {
        let (body, target, names) = fixture_body();
        let contract = contract_for(target, &body);
        let consumed =
            consume_analysis_result(&replace_contract_section(&body, names, &contract)).unwrap();
        assert!(consumed.acceptance.accepts_as_claim(), "{:?}", consumed.acceptance.unresolved);
    }

    #[test]
    fn rehashed_contract_with_two_checked_execution_products_is_refused() {
        let (body, target, names) = fixture_body();
        let mut contract = contract_for(target, &body);
        let program = contract.program.as_mut().unwrap();
        program.checked_functional_response_grid = Some(crate::CheckedFunctionalResponseGridWire {
            format: 1,
            treatment: 0,
            outcome: 1,
            members: Vec::new(),
        });
        program.checked_nested_counterfactual = Some(crate::CheckedNestedCounterfactualWire {
            format: 1,
            treatment: 0,
            mediator: 1,
            outcome: 2,
            control_bits: 0.0f64.to_bits(),
            active_bits: 1.0f64.to_bits(),
            model: "linear_gaussian".into(),
            procedure: "shared_exogenous".into(),
        });
        contract.identities.program =
            *program_digest(contract.program.as_ref().unwrap()).unwrap().as_bytes();
        seal_claim(&mut contract, &body);

        let consumed =
            consume_analysis_result(&replace_contract_section(&body, names, &contract)).unwrap();
        assert!(
            consumed
                .acceptance
                .unresolved
                .iter()
                .any(|reason| reason.as_ref() == "program.checked_payload_conflict")
        );
    }

    #[test]
    fn rehashed_contract_with_one_checked_product_for_wrong_target_is_refused() {
        let (body, target, names) = fixture_body();
        let mut contract = contract_for(target, &body);
        let program = contract.program.as_mut().unwrap();
        program.checked_nested_counterfactual = Some(crate::CheckedNestedCounterfactualWire {
            format: 1,
            treatment: 0,
            mediator: 1,
            outcome: 2,
            control_bits: 0.0f64.to_bits(),
            active_bits: 1.0f64.to_bits(),
            model: "linear_gaussian".into(),
            procedure: "shared_exogenous".into(),
        });
        contract.identities.program =
            *program_digest(contract.program.as_ref().unwrap()).unwrap().as_bytes();
        seal_claim(&mut contract, &body);

        let consumed =
            consume_analysis_result(&replace_contract_section(&body, names, &contract)).unwrap();
        assert!(
            consumed
                .acceptance
                .unresolved
                .iter()
                .any(|reason| reason.as_ref() == "program.checked_nested_counterfactual.target")
        );
    }

    #[test]
    fn legacy_aipw_contract_is_readable_with_executable_dependency_refusal() {
        let (body, target, names) = fixture_body();
        let mut contract = contract_for(target, &body);
        contract.estimator = Some("aipw".into());
        let program = contract.program.as_mut().unwrap();
        program.commitments.estimator = Some("aipw".into());
        program.commitments.resolved_estimator = Some("aipw".into());
        contract.claim = None;
        contract.execution = None;
        contract.identities.execution = None;
        contract.identities.program =
            *program_digest(contract.program.as_ref().unwrap()).unwrap().as_bytes();
        seal_claim(&mut contract, &body);

        let bytes = to_bytes(&body, names, Some(&contract));
        let consumed = consume_analysis_result(&bytes).unwrap();
        assert_eq!(consumed.body, body);
        assert!(
            consumed
                .acceptance
                .unresolved
                .iter()
                .any(|key| { key.as_ref() == "program.checked_aipw_lowering" })
        );
        assert!(
            consumed
                .acceptance
                .unresolved
                .iter()
                .any(|key| { key.as_ref() == "checked_aipw.row_binding" })
        );
    }

    #[test]
    fn legacy_iv_contract_is_readable_with_precise_checked_lowering_refusal() {
        let (body, target, names) = fixture_body();
        let mut contract = contract_for(target, &body);
        contract.estimator = Some("iv.wald".into());
        let program = contract.program.as_mut().unwrap();
        program.commitments.estimator = Some("iv.wald".into());
        program.commitments.resolved_estimator = Some("iv.wald".into());
        contract.claim = None;
        contract.execution = None;
        contract.identities.execution = None;
        contract.identities.program =
            *program_digest(contract.program.as_ref().unwrap()).unwrap().as_bytes();
        seal_claim(&mut contract, &body);

        let consumed = consume_analysis_result(&to_bytes(&body, names, Some(&contract))).unwrap();
        assert!(
            consumed
                .acceptance
                .unresolved
                .iter()
                .any(|key| { key.as_ref() == "program.checked_iv_lowering" })
        );
        assert!(!consumed.acceptance.accepts_as_verified_program());
    }

    #[test]
    fn rehashed_checked_aipw_with_invalid_fold_count_is_refused() {
        let (body, target, names) = fixture_body();
        let mut contract = contract_for(target, &body);
        contract.estimator = Some("aipw".into());
        let program = contract.program.as_mut().unwrap();
        program.commitments.estimator = Some("aipw".into());
        program.commitments.resolved_estimator = Some("aipw".into());
        program.checked_aipw_lowering = Some(CheckedAipwLoweringWire {
            format: 1,
            functional: 0,
            treatment: 0,
            outcome: 1,
            adjustment: Vec::new(),
            population: "all_observed".into(),
            procedure: "cross_fitted_logistic_ols".into(),
            folds: 0,
            se_kind: "hc1".into(),
            se_lag: None,
            bootstrap_replicates: 0,
        });
        contract.identities.program =
            *program_digest(contract.program.as_ref().unwrap()).unwrap().as_bytes();
        seal_claim(&mut contract, &body);

        assert!(
            encode_analysis_result_artifact_with_contract(
                &body,
                names.clone(),
                "tampered-aipw-lowering",
                Some(&contract),
            )
            .is_err()
        );
        let consumed =
            consume_analysis_result(&replace_contract_section(&body, names, &contract)).unwrap();
        assert!(
            consumed
                .acceptance
                .unresolved
                .iter()
                .any(|key| { key.as_ref() == "program.checked_aipw_lowering" })
        );
    }

    #[test]
    fn checked_aipw_rows_require_the_sealed_snapshot_digest() {
        let (body, target, names) = fixture_body();
        let mut contract = contract_for(target, &body);
        contract.estimator = Some("aipw".into());
        let program = contract.program.as_mut().unwrap();
        program.commitments.estimator = Some("aipw".into());
        program.commitments.resolved_estimator = Some("aipw".into());
        program.checked_aipw_lowering = Some(CheckedAipwLoweringWire {
            format: 1,
            functional: 0,
            treatment: 0,
            outcome: 1,
            adjustment: Vec::new(),
            population: "all_observed".into(),
            procedure: "cross_fitted_logistic_ols".into(),
            folds: 5,
            se_kind: "hc1".into(),
            se_lag: None,
            bootstrap_replicates: 0,
        });
        contract.identities.program =
            *program_digest(contract.program.as_ref().unwrap()).unwrap().as_bytes();
        let row_binding = CheckedAipwRowsWire {
            format: 1,
            rows: vec![0, 2, 4, 6],
            data_snapshot: contract.identities.data_snapshot,
        };
        contract.checked_aipw_rows = Some(row_binding);
        contract.identities.checked_aipw_rows = Some([0; 32]);
        seal_claim(&mut contract, &body);

        assert!(
            encode_analysis_result_artifact_with_contract(
                &body,
                names.clone(),
                "tampered-aipw-rows",
                Some(&contract),
            )
            .is_err()
        );
        let consumed =
            consume_analysis_result(&replace_contract_section(&body, names, &contract)).unwrap();
        assert!(
            consumed
                .acceptance
                .unresolved
                .iter()
                .any(|key| { key.as_ref() == "checked_aipw.row_binding" })
        );
    }

    #[test]
    fn prior_requirement_is_bound_to_bayesian_inference_dependencies() {
        let (body, target, names) = fixture_body();
        let mut contract = contract_for(target, &body);
        contract.program.as_mut().unwrap().commitments.prior_required = true;
        seal_claim(&mut contract, &body);

        let unresolved = unresolved_after(&body, &names, &contract);
        assert!(
            unresolved.iter().any(|item| item == "program.commitments.prior_required"),
            "{unresolved:?}"
        );
    }

    #[test]
    fn older_linear_frontdoor_program_is_readable_but_unverified() {
        let (body, target, names) = fixture_body();
        let mut contract = contract_for(target, &body);
        contract.estimator = Some("frontdoor.linear_two_stage".into());
        contract.program.as_mut().unwrap().commitments.estimator =
            Some("frontdoor.linear_two_stage".into());
        contract.program.as_mut().unwrap().commitments.resolved_estimator =
            Some("frontdoor.linear_two_stage".into());
        contract.program.as_mut().unwrap().checked_frontdoor_lowering = None;
        // Recompute the program identity and seal to isolate the missing
        // semantic payload from ordinary integrity failures.
        let program = contract.program.as_ref().unwrap();
        contract.identities.program = *program_digest(program).unwrap().as_bytes();
        contract.seal = contract_seal(
            &contract.identities,
            &contract.reasoning,
            &contract.graph_class,
            &contract.structure_source,
            contract.identifier.as_deref(),
            contract.estimator.as_deref(),
        )
        .unwrap();
        let unresolved = unresolved_after(&body, &names, &contract);
        assert!(
            unresolved.iter().any(|item| item == "program.checked_frontdoor_lowering"),
            "{unresolved:?}"
        );
    }

    #[test]
    fn tampered_claim_fields_fail_the_claim_id() {
        let (body, target, names) = fixture_body();
        let original = contract_for(target, &body);
        let edits: [fn(&mut ClaimSectionWire); 5] = [
            |claim| claim.support_domain = "supported".into(),
            |claim| claim.identification_domain = "unknown".into(),
            |claim| claim.calibration = CalibrationSlotWire::unavailable("forged"),
            |claim| claim.value_bits = Some(3.0f64.to_bits()),
            |claim| claim.execution = Some([5; 32]),
        ];
        for edit in edits {
            let mut contract = original.clone();
            edit(contract.claim.as_mut().unwrap());
            let unresolved = unresolved_after(&body, &names, &contract);
            assert!(unresolved.iter().any(|item| item == "claim.id"), "{unresolved:?}");
        }
    }

    #[test]
    fn tampered_audit_fields_and_slots_fail_the_seal() {
        let (body, target, names) = fixture_body();
        let original = contract_for(target, &body);
        let edits: [fn(&mut AnalysisResultContractWire); 7] = [
            |contract| contract.graph_class = "Pag".into(),
            |contract| contract.structure_source = "accepted".into(),
            |contract| contract.estimator = Some("propensity_weighting".into()),
            |contract| contract.identifier = Some("frontdoor".into()),
            |contract| {
                contract.reasoning.support.value.as_mut().unwrap().empirical = "supported".into();
            },
            |contract| {
                contract.reasoning.assumptions.value.as_mut().unwrap().obligations.push(
                    ObligationSectionWire {
                        id: "assumption.0".into(),
                        scope: "program".into(),
                        kind: "user_assertion".into(),
                        status: "declared".into(),
                    },
                );
            },
            |contract| {
                contract.reasoning.uncertainty = SlotSectionWire {
                    value: Some(UncertaintySlotWire { components: Vec::new() }),
                    unavailable: None,
                };
            },
        ];
        for edit in edits {
            let mut contract = original.clone();
            edit(&mut contract);
            let unresolved = unresolved_after(&body, &names, &contract);
            assert!(unresolved.iter().any(|item| item == "contract.seal"), "{unresolved:?}");
        }
    }

    #[test]
    fn tampered_body_fails_the_claim_id() {
        let (body, target, names) = fixture_body();
        let contract = contract_for(target, &body);
        let edits: [fn(&mut AnalysisResultWire); 4] = [
            |body| body.standard_error = body.standard_error.map(|se| se / 100.0),
            |body| body.refutations.push(refutation("placebo.treatment.permute", true)),
            |body| {
                body.diagnostics.push(crate::DiagnosticWire {
                    code: "forged".into(),
                    kind: "execution".into(),
                    severity: "info".into(),
                    message: "forged".into(),
                    artifact_id: None,
                    fields: Vec::new(),
                });
            },
            |body| body.identification.candidates_examined += 1,
        ];
        for edit in edits {
            let mut tampered = body.clone();
            edit(&mut tampered);
            let unresolved = unresolved_after(&tampered, &names, &contract);
            assert!(unresolved.iter().any(|item| item == "claim.id"), "{unresolved:?}");
        }
    }

    #[test]
    fn forged_premises_with_a_rehashed_chain_are_unresolved() {
        use crate::identity::{identification_digest, program_digest};
        let (body, target, names) = fixture_body();
        let mut contract = contract_for(target, &body);
        let identification = contract.identification.as_mut().unwrap();
        identification.target_question = [7; 32];
        identification.observation = [9; 32];
        let digest = *identification_digest(identification).unwrap().as_bytes();
        contract.identities.identification = digest;
        let program = contract.program.as_mut().unwrap();
        program.identification = digest;
        contract.identities.program = *program_digest(program).unwrap().as_bytes();
        seal_claim(&mut contract, &body);
        let unresolved = unresolved_after(&body, &names, &contract);
        for link in ["identification.target_question", "identification.observation"] {
            assert!(unresolved.iter().any(|item| item == link), "{link}: {unresolved:?}");
        }
    }

    #[test]
    fn resealed_domains_kind_and_support_are_rederived() {
        let (mut body, target, names) = fixture_body();
        body.refutations.push(refutation("overlap.assessment", false));
        let mut contract = contract_for(target, &body);
        contract.reasoning.support.value.as_mut().unwrap().empirical = "supported".into();
        let claim = contract.claim.as_mut().unwrap();
        claim.support_domain = "supported".into();
        claim.kind = "bounds".into();
        claim.value_bits = None;
        seal_claim(&mut contract, &body);
        // Resealing hides nothing: the support label and the claim kind are
        // re-derived from the body the artifact carries.
        let unresolved = unresolved_after(&body, &names, &contract);
        for item in ["reasoning.support.empirical", "claim.kind"] {
            assert!(unresolved.iter().any(|entry| entry == item), "{item}: {unresolved:?}");
        }

        // Domains alone, with the slots left honest, are re-derived too.
        let (body, target, names) = fixture_body();
        let mut contract = contract_for(target, &body);
        contract.claim.as_mut().unwrap().support_domain = "supported".into();
        seal_claim(&mut contract, &body);
        let unresolved = unresolved_after(&body, &names, &contract);
        assert!(unresolved.iter().any(|entry| entry == "claim.domains"), "{unresolved:?}");
    }

    #[test]
    fn support_empirical_distinguishes_pass_failure_and_absence() {
        assert_eq!(support_empirical(&[refutation("placebo.treatment.permute", false)]), None);
        assert_eq!(
            support_empirical(&[refutation("overlap.assessment", true)]).as_deref(),
            Some("supported")
        );
        let failed = [refutation("overlap.assessment", true), refutation("overlap.rule", false)];
        assert_eq!(support_empirical(&failed).as_deref(), Some("failed:overlap.rule"));
        let support = |empirical: &str| SupportSlotWire {
            matrix_status: "licensed".into(),
            matrix_coordinate: None,
            empirical: empirical.into(),
        };
        assert_eq!(claim_domains(Some(&support("supported")), None).support, "supported");
        assert_eq!(
            claim_domains(Some(&support("failed:overlap.rule")), None).support,
            "contradicted"
        );
        assert_eq!(
            claim_domains(Some(&support("unavailable:not_evaluated")), None).support,
            "unknown"
        );
    }

    #[test]
    fn partially_identified_single_atom_is_bounds() {
        let (body, _, _) = fixture_body();
        let slot = IdentificationSlotWire {
            status: "partially_identified".into(),
            identified_mass: 1.0,
            unidentified_mass: 0.0,
            unevaluable_mass: 0.0,
            incomplete_search_mass: 0.0,
            full_mass_scope: true,
            search_capped: false,
            weight_basis: None,
        };
        assert_eq!(claim_kind_name(&body, Some(&slot)), "bounds");
    }

    #[test]
    fn claim_kind_is_never_point_short_of_full_identified_mass_or_for_unknown_status() {
        let (body, _, _) = fixture_body();
        let slot = |status: &str, identified: f64, unevaluable: f64, incomplete: f64| {
            IdentificationSlotWire {
                status: status.into(),
                identified_mass: identified,
                unidentified_mass: 0.0,
                unevaluable_mass: unevaluable,
                incomplete_search_mass: incomplete,
                full_mass_scope: true,
                search_capped: false,
                weight_basis: None,
            }
        };
        let identified = "nonparametrically_identified";
        assert_eq!(claim_kind_name(&body, Some(&slot(identified, 1.0, 0.0, 0.0))), "point");
        assert_eq!(claim_kind_name(&body, Some(&slot(identified, 0.6, 0.0, 0.4))), "mixture");
        assert_eq!(claim_kind_name(&body, Some(&slot(identified, 0.75, 0.25, 0.0))), "mixture");
        for status in ["not_certified", "proven_non_transportable", "surprise"] {
            assert_eq!(claim_kind_name(&body, Some(&slot(status, 1.0, 0.0, 0.0))), "incomplete");
        }
    }

    #[test]
    fn mixture_masses_must_sum_to_one() {
        validate_mixture_masses(0.5, 0.3, 0.2, 0.0).unwrap();
        assert!(validate_mixture_masses(0.5, 0.3, 0.1, 0.0).is_err());
        assert!(validate_mixture_masses(1.2, 0.0, 0.0, 0.0).is_err());
        assert!(validate_mixture_masses(f64::NAN, 0.0, 0.0, 0.0).is_err());
    }
}
