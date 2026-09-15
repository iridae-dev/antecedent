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
    program_digest, validate_mixture_masses,
};
use crate::{
    AnalysisResultHeader, AnalysisResultWire, EncodedArtifact, ExecutionIdentityWire, IoError,
    from_cbor, query_wire::causal_query_from_wire,
};

/// Section id on the composite `analysis_result` artifact.
pub const CONTRACT_SECTION: &str = "analysis_result.contract";

/// Contract-section payload format. Bump only when the wire shape changes.
pub const CONTRACT_SECTION_FORMAT: u16 = 1;

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
}

impl From<&ContractIdentities> for ContractIdentitiesWire {
    fn from(identities: &ContractIdentities) -> Self {
        Self {
            target: *identities.target.as_bytes(),
            identification: *identities.identification.as_bytes(),
            identification_product: identities
                .identification_product
                .map(|digest| *digest.as_bytes()),
            program: *identities.program.as_bytes(),
            inference_binding: *identities.inference_binding.as_bytes(),
            observation: *identities.observation.as_bytes(),
            data_snapshot: *identities.data_snapshot.as_bytes(),
        }
    }
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
    /// Caller-attested evidence (custom validators).
    #[serde(default)]
    pub attested: Vec<AttestedEvidenceWire>,
}

/// Calibration status bound into the claim.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq, Eq)]
pub struct CalibrationSlotWire {
    /// `calibrated` | `scope_not_assessed` | `unavailable`.
    pub status: String,
    /// Matching coverage record id.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub record_id: Option<String>,
    /// Reason code when not calibrated.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<String>,
    /// Record `n`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_n: Option<u64>,
    /// Record dependence label.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope_dependence: Option<String>,
    /// Record SHA.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub calibration_sha: Option<String>,
}

/// Caller-attested custom-validator evidence.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AttestedEvidenceWire {
    /// Validator name.
    pub name: String,
    /// Evidence kind (`custom_validator`).
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
    /// Always false at 1.10.0 — attested evidence is not re-verifiable.
    pub reverifiable: bool,
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
        }
    }

    /// Bind a coverage record onto an execution.
    #[must_use]
    pub fn from_record(
        record_id: impl Into<String>,
        n: u64,
        dependence: impl Into<String>,
        calibration_sha: impl Into<String>,
        boundary: bool,
        in_scope: bool,
    ) -> Self {
        let (status, reason) = if boundary {
            ("scope_not_assessed", Some("boundary_record"))
        } else if in_scope {
            ("calibrated", None)
        } else {
            ("scope_not_assessed", None)
        };
        Self {
            status: status.into(),
            record_id: Some(record_id.into()),
            reason: reason.map(str::to_string),
            scope_n: Some(n),
            scope_dependence: Some(dependence.into()),
            calibration_sha: Some(calibration_sha.into()),
        }
    }
}

/// Versioned contract companion for an `analysis_result` artifact.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub struct AnalysisResultContractWire {
    /// Section format.
    pub format: u16,
    /// Advertised identities.
    pub identities: ContractIdentitiesWire,
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
    let unresolved = verify_stored_payloads(contract);
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
    if contract.target.query != body.query {
        unresolved.push(Arc::from("body.query"));
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
        }
    }
    if let Some(product) = &contract.identification_product {
        if !identification_product_matches_body(product, body) {
            unresolved.push(Arc::from("body.identification_product"));
        }
    } else if contract.identities.identification_product.is_some() {
        unresolved.push(Arc::from("identities.identification_product"));
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
        if claim.kind == "point" && !claim_value_matches_body(claim, body) {
            unresolved.push(Arc::from("claim.value"));
        }
        if claim.execution.is_some() && contract.execution.is_none() {
            unresolved.push(Arc::from("identities.execution"));
        }
        if !matches!(
            claim.calibration.status.as_str(),
            "calibrated" | "scope_not_assessed" | "unavailable"
        ) {
            unresolved.push(Arc::from("claim.calibration"));
        }
        if rederive_calibration_slot(contract) != claim.calibration {
            unresolved.push(Arc::from("claim.calibration"));
        }
    }
    unresolved
}

fn rederive_calibration_slot(contract: &AnalysisResultContractWire) -> CalibrationSlotWire {
    let Some(program) = contract.program.as_ref() else {
        return CalibrationSlotWire::unavailable("estimator_grid_not_measured");
    };
    let commitments = &program.commitments;
    let query = crate::query_wire::query_axis_name(&contract.target.query, &contract.graph_class)
        .unwrap_or("Unknown");
    let inference = if commitments.inference.eq_ignore_ascii_case("bayesian") {
        "Bayesian"
    } else {
        "Frequentist"
    };
    let estimator = if inference == "Bayesian" {
        ""
    } else {
        commitments.resolved_estimator.as_deref().or(commitments.estimator.as_deref()).unwrap_or("")
    };
    let interval_method = if inference == "Bayesian" {
        "posterior_quantile"
    } else {
        commitments.interval_method.as_str()
    };
    let se_kind = commitments.se_kind.as_deref().unwrap_or("");
    let row_count = contract.data_snapshot.as_ref().map(|snap| snap.row_count).unwrap_or(0);
    let dependence = if estimator == "circular_block" {
        "circular_block"
    } else if contract.data_snapshot.as_ref().is_some_and(|snap| snap.modality == "panel") {
        "panel_cluster"
    } else {
        "iid"
    };
    if let Some(record) = coverage_records().into_iter().find(|row| {
        row.query == query
            && row.graph_class == contract.graph_class
            && row.inference == inference
            && row.estimator == estimator
            && row.interval_method == interval_method
            && row.se_kind == se_kind
    }) {
        let in_scope = row_count >= record.n && dependence == record.dependence;
        return CalibrationSlotWire::from_record(
            record.id,
            record.n,
            record.dependence,
            record.calibration_sha,
            record.boundary,
            in_scope,
        );
    }
    CalibrationSlotWire::unavailable(if interval_method == "none" {
        "no_interval_reported"
    } else {
        "estimator_grid_not_measured"
    })
}

struct CoverageRow {
    id: String,
    query: String,
    graph_class: String,
    inference: String,
    estimator: String,
    interval_method: String,
    se_kind: String,
    n: u64,
    dependence: String,
    boundary: bool,
    calibration_sha: String,
}

fn coverage_records() -> Vec<CoverageRow> {
    let mut rows = Vec::new();
    let mut current: Option<CoverageRow> = None;
    let flush = |rows: &mut Vec<CoverageRow>, current: &mut Option<CoverageRow>| {
        if let Some(row) = current.take() {
            if !row.id.is_empty() {
                rows.push(row);
            }
        }
    };
    for line in include_str!("../../../parity/coverage_records.toml").lines() {
        let line = line.trim();
        if line == "[[record]]" {
            flush(&mut rows, &mut current);
            current = Some(CoverageRow {
                id: String::new(),
                query: String::new(),
                graph_class: String::new(),
                inference: String::new(),
                estimator: String::new(),
                interval_method: String::new(),
                se_kind: String::new(),
                n: 0,
                dependence: String::new(),
                boundary: false,
                calibration_sha: String::new(),
            });
            continue;
        }
        let Some(row) = current.as_mut() else { continue };
        let Some((key, value)) = line.split_once('=') else { continue };
        let key = key.trim();
        let value = value.trim().trim_matches('"');
        match key {
            "id" => row.id = value.to_string(),
            "query" => row.query = value.to_string(),
            "graph_class" => row.graph_class = value.to_string(),
            "inference" => row.inference = value.to_string(),
            "estimator" => row.estimator = value.to_string(),
            "interval_method" => row.interval_method = value.to_string(),
            "se_kind" => row.se_kind = value.to_string(),
            "n" => row.n = value.parse().unwrap_or(0),
            "dependence" => row.dependence = value.to_string(),
            "boundary" => row.boundary = value == "true",
            "calibration_sha" => row.calibration_sha = value.to_string(),
            _ => {}
        }
    }
    flush(&mut rows, &mut current);
    rows
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
        contract.claim.as_ref().and_then(|claim| claim.execution.as_ref()),
        execution_digest,
    );
    if let Some(claim) = &contract.claim {
        if !claim_id_matches(contract, claim) {
            unresolved.push(Arc::from("claim.id"));
        }
    }
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
        "inference_binding.program",
        contract.inference_binding.as_ref().map(|item| item.program),
        Some(&contract.identities.program),
    );
    require_nested_digest(
        &mut unresolved,
        "data_snapshot.observation",
        contract.data_snapshot.as_ref().map(|item| item.observation),
        Some(&contract.identities.observation),
    );
    unresolved
}

fn require_nested_digest(
    unresolved: &mut Vec<Arc<str>>,
    label: &'static str,
    nested: Option<[u8; 32]>,
    advertised: Option<&[u8; 32]>,
) {
    match (nested, advertised) {
        (Some(got), Some(want)) if got == *want => {}
        (None, None) => {}
        (None, Some(_)) | (Some(_), None) | (Some(_), Some(_)) => {
            unresolved.push(Arc::from(label));
        }
    }
}

fn claim_identity_wire(
    contract: &AnalysisResultContractWire,
    claim: &ClaimSectionWire,
) -> ClaimIdentityWire {
    let calibration = crate::to_cbor(&claim.calibration)
        .ok()
        .map(|bytes| crate::hash_payload(&bytes))
        .unwrap_or([0; 32]);
    let attested = if claim.attested.is_empty() {
        [0; 32]
    } else {
        crate::to_cbor(&claim.attested)
            .ok()
            .map(|bytes| crate::hash_payload(&bytes))
            .unwrap_or([0; 32])
    };
    ClaimIdentityWire::new(
        contract.identities.program,
        contract.identities.target,
        claim.kind.clone(),
        claim.value_bits,
        claim.execution,
        calibration,
        attested,
    )
}

fn claim_id_matches(contract: &AnalysisResultContractWire, claim: &ClaimSectionWire) -> bool {
    matches!(
        claim_digest(&claim_identity_wire(contract, claim)),
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
        && product.derivation == body.identification.derivation
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
            });
        }
    };
    let acceptance = accept_contract(&header, &body, contract.as_ref());
    Ok(AnalysisResultConsumption { header, body, contract, acceptance })
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
    AcceptanceReport::new(
        true,
        verified,
        unresolved,
        if verified {
            Arc::from([
                Arc::from("store"),
                Arc::from("forward"),
                Arc::from("read_body"),
                Arc::from("verify_contract"),
                Arc::from("inspect_claim"),
            ])
        } else {
            storage_ops()
        },
        (!verified).then(|| Arc::from("unverified_contract_references")),
    )
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
    let claim_id = consumption_claim_id(&consumed);
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
        HandoffReceipt::new(
            claim_id,
            None,
            profile.id.clone(),
            "restricted_accept",
            [Arc::from("bytes"), Arc::from("read_body")],
            if consumed.acceptance.unresolved.is_empty() {
                Vec::new()
            } else {
                vec![Arc::from("verified_references")]
            },
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

fn consumption_claim_id(consumed: &AnalysisResultConsumption) -> SemanticDigest {
    consumed
        .contract
        .as_ref()
        .and_then(|contract| contract.claim.as_ref())
        .map_or(SemanticDigest::from_bytes([0; 32]), |claim| {
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
    ClaimHostProjection {
        claim_id: Some(digest_hex(&claim.claim_id)),
        kind: Some(claim.kind.clone()),
        value,
        identification_domain: serde_json::Value::String(claim.identification_domain.clone()),
        support_domain: serde_json::Value::String(claim.support_domain.clone()),
        evaluated_domain: serde_json::Value::String(claim.evaluated_domain.clone()),
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
    let claim_id = consumption_claim_id(consumed);
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
            },
            identification_variables: None,
            temporal_identification: Vec::new(),
            estimate: Some(2.0),
            standard_error: Some(0.1),
            assumptions: Vec::new(),
            diagnostics: Vec::new(),
            refutations: Vec::new(),
            response: None,
            posterior_artifact: None,
            mediation_grid: None,
            structural_response: None,
        };
        let target = TargetIdentityWire {
            format: IDENTITY_FORMAT,
            schema: schema_to_wire(&schema),
            query: query_wire,
        };
        (body, target, schema.variables().iter().map(|v| v.name.to_string()).collect())
    }

    #[allow(clippy::too_many_lines)]
    fn contract_for(
        target: TargetIdentityWire,
        body: &AnalysisResultWire,
    ) -> AnalysisResultContractWire {
        use antecedent_graph::{Dag, DenseNodeId};

        use crate::identity::{
            InferentialCommitmentsWire, ObservationIdentityWire, dag_identity,
            data_snapshot_digest, execution_digest, identification_digest,
            identification_product_digest_wire, inference_binding_digest, program_digest,
        };

        let target_digest = digest_wire(IdentityDomain::Target, &target).unwrap();
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
            target_question: *target_digest.as_bytes(),
            population_depends_on: Vec::new(),
            rd_config: None,
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
            derivation: body.identification.derivation.clone(),
            required_assumptions: body.identification.required_assumptions.clone(),
            hedge: false,
            search_capped: false,
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
            identification: *identification_digest.as_bytes(),
            identification_product: Some(*product_digest.as_bytes()),
            commitments: commitments.clone(),
        };
        let program_digest = program_digest(&program).unwrap();
        let inference_binding = InferenceBindingWire {
            format: IDENTITY_FORMAT,
            program: *program_digest.as_bytes(),
            inference: "frequentist".into(),
            bootstrap_replicates: 0,
            n_draws: None,
            prior_scale: None,
            prior_mapping: None,
            validation_suite: None,
            overlap_policy: None,
            estimator_spec: None,
            response_options: None,
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
        };
        let snapshot_digest = data_snapshot_digest(&data_snapshot).unwrap();
        let execution = crate::ExecutionIdentityWire {
            format: IDENTITY_FORMAT,
            library_version: antecedent_core::VERSION.into(),
            seed: 1,
            threads: 1,
            backend: "scalar".into(),
        };
        let execution_digest = execution_digest(&execution).unwrap();
        let identities = ContractIdentitiesWire {
            target: *target_digest.as_bytes(),
            identification: *identification_digest.as_bytes(),
            identification_product: Some(*product_digest.as_bytes()),
            program: *program_digest.as_bytes(),
            inference_binding: *inference_digest.as_bytes(),
            observation: *observation_digest.as_bytes(),
            data_snapshot: *snapshot_digest.as_bytes(),
        };
        let mut contract = AnalysisResultContractWire {
            format: CONTRACT_SECTION_FORMAT,
            identities,
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
        };
        let mut claim = ClaimSectionWire {
            claim_id: [0; 32],
            kind: "point".into(),
            value_bits: Some(2.0f64.to_bits()),
            execution: Some(*execution_digest.as_bytes()),
            identification_domain: "identified".into(),
            support_domain: "unknown".into(),
            evaluated_domain: "evaluated".into(),
            calibration: super::rederive_calibration_slot(&contract),
            attested: Vec::new(),
        };
        contract.claim = Some(claim);
        seal_claim(&mut contract);
        contract
    }

    fn seal_claim(contract: &mut AnalysisResultContractWire) {
        let Some(claim) = contract.claim.clone() else {
            return;
        };
        let claim_id = *claim_digest(&super::claim_identity_wire(contract, &claim)).unwrap().as_bytes();
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
        seal_claim(&mut zero);
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
        seal_claim(&mut absent);
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
    fn accept_claim_unknown_feature_is_storage_not_acceptance() {
        let (body, target, names) = fixture_body();
        let mut contract = contract_for(target, &body);
        if let Some(claim) = contract.claim.as_mut() {
            claim.kind = "mixture".into();
        }
        seal_claim(&mut contract);
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
}
