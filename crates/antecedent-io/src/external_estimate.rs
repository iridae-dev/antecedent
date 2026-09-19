//! Caller-attested external-estimate claims.
//!
//! Antecedent certifies identification. The fitted payload is hashed and stored
//! as attested evidence; it is not re-run, not an `EconML` wrapper, and not a
//! native calibrated point claim.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalSchemaBuilder, IDENTITY_FORMAT, IdentityDomain,
    MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
};

use crate::analysis_result_artifact::{AnalysisResultWire, encode_analysis_result_artifact};
use crate::analysis_wire::IdentificationResultWire;
use crate::container::{CompressPolicy, pack_section_shared};
use crate::contract_section::{
    AnalysisResultContractWire, AttestedEvidenceWire, CONTRACT_SECTION, CONTRACT_SECTION_FORMAT,
    CalibrationSlotWire, ClaimSectionWire, ContractIdentitiesWire, IdentificationSlotWire,
    ReasoningSectionWire, SlotSectionWire, SupportSlotWire, contract_seal, digest_hex,
    result_digest,
};
use crate::convert::schema_to_wire;
use crate::error::IoError;
use crate::expr_wire::ExprArenaWire;
use crate::identity::{
    ClaimIdentityWire, IdentificationProductWire, ObservationIdentityWire, TargetIdentityWire,
    claim_digest,
};
use crate::payload_digest;
use crate::query_wire::causal_query_to_wire;
use crate::to_cbor;

/// Inputs for an attested external-estimate receipt.
#[derive(Clone, Debug)]
pub struct ExternalEstimateAttach {
    /// Caller-supplied learner label. Hashed, not interpreted.
    pub learner: String,
    /// Canonical learner-config bytes. Hashed, not interpreted.
    pub config: Vec<u8>,
    /// Effect (and optional interval) bytes. Hashed, not interpreted.
    pub payload: Vec<u8>,
    /// Schema names in receipt order.
    pub names: Vec<String>,
    /// Certified treatment name.
    pub treatment: String,
    /// Certified outcome name.
    pub outcome: String,
    /// Certified adjustment names.
    pub confounders: Vec<String>,
    /// Identifier id (`backdoor.adjustment`, …).
    pub identifier: String,
    /// Parent identification status, `snake_case` or `PascalCase`.
    pub status: String,
    /// Parent identification digest, when the parent already had one.
    pub identification: Option<[u8; 32]>,
    /// Parent or column-derived data-snapshot digest.
    pub data_snapshot: [u8; 32],
    /// Scalar ATE the caller labelled as such; omitted for arrays / unlabelled payloads.
    pub scalar_value: Option<f64>,
}

/// Encode a contracted `analysis_result` whose claim is an attested external estimate.
///
/// # Errors
///
/// Unknown treatment/outcome names, invalid schema, or encode failure.
pub fn encode_external_estimate_claim(attach: &ExternalEstimateAttach) -> Result<Vec<u8>, IoError> {
    let evidence = attested_external_estimate(attach);
    let (body, target, names) = receipt_body(attach)?;
    let mut contract = receipt_contract(attach, &body, target, &evidence)?;
    seal_claim(&mut contract, &body)?;
    write_receipt(&body, names, &contract)
}

/// Hash learner / config / payload onto an `external_estimate` evidence row.
#[must_use]
pub fn attested_external_estimate(attach: &ExternalEstimateAttach) -> AttestedEvidenceWire {
    AttestedEvidenceWire {
        name: attach.learner.clone(),
        kind: "external_estimate".into(),
        passed: true,
        refuted_ate: None,
        comparison: None,
        informative: true,
        failure_condition: None,
        reverifiable: false,
        config_digest: Some(digest_hex(&payload_digest(
            "external_estimate.config",
            &attach.config,
        ))),
        payload_digest: Some(digest_hex(&payload_digest(
            "external_estimate.payload",
            &attach.payload,
        ))),
    }
}

fn receipt_body(
    attach: &ExternalEstimateAttach,
) -> Result<(AnalysisResultWire, TargetIdentityWire, Vec<String>), IoError> {
    let mut builder = CausalSchemaBuilder::new();
    for name in &attach.names {
        let hint = if *name == attach.treatment {
            RoleHint::TreatmentCandidate
        } else if *name == attach.outcome {
            RoleHint::OutcomeCandidate
        } else {
            RoleHint::Context
        };
        builder
            .add_variable(
                name.as_str(),
                ValueType::Continuous,
                SmallRoleSet::from_hint(hint),
                None,
                None,
                MeasurementSpec::default(),
            )
            .map_err(|err| IoError::Convert(err.to_string()))?;
    }
    let schema = builder.build().map_err(|err| IoError::Convert(err.to_string()))?;
    let treatment = index_of(&attach.names, &attach.treatment)?;
    let outcome = index_of(&attach.names, &attach.outcome)?;
    let query = CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(
        VariableId::from_raw(treatment),
        VariableId::from_raw(outcome),
    ));
    let query_wire = causal_query_to_wire(&query)?;
    let status = wire_status(&attach.status);
    let body = AnalysisResultWire {
        query: query_wire.clone(),
        identification: IdentificationResultWire {
            status: status.clone(),
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
        estimate: attach.scalar_value,
        standard_error: None,
        assumptions: Vec::new(),
        diagnostics: Vec::new(),
        refutations: Vec::new(),
        response: None,
        posterior_artifact: None,
        mediation_grid: None,
        structural_response: None,
        unit_effects: None,
    };
    let target = TargetIdentityWire {
        format: IDENTITY_FORMAT,
        schema: schema_to_wire(&schema),
        query: query_wire,
    };
    Ok((body, target, attach.names.clone()))
}

#[allow(clippy::too_many_lines)]
fn receipt_contract(
    attach: &ExternalEstimateAttach,
    body: &AnalysisResultWire,
    target: TargetIdentityWire,
    evidence: &AttestedEvidenceWire,
) -> Result<AnalysisResultContractWire, IoError> {
    let target_digest = crate::digest_wire(IdentityDomain::Target, &target)?;
    let observation = ObservationIdentityWire {
        format: IDENTITY_FORMAT,
        schema: target.schema.clone(),
        observation: Vec::new(),
    };
    let observation_digest = crate::digest_wire(IdentityDomain::Observation, &observation)?;
    let identification = attach.identification.unwrap_or_else(|| {
        payload_digest(
            "external_estimate.identification",
            format!(
                "{}|{}|{}|{}|{}",
                attach.status,
                attach.identifier,
                attach.treatment,
                attach.outcome,
                attach.confounders.join(",")
            )
            .as_bytes(),
        )
    });
    let identification_product = IdentificationProductWire {
        format: IDENTITY_FORMAT,
        status: body.identification.status.clone(),
        estimands: body.identification.estimands.clone(),
        arena: body.identification.arena.clone(),
        derivation_rules: Vec::new(),
        required_assumptions: Vec::new(),
        hedge: false,
        search_capped: false,
        envelope: None,
    };
    let kind = if attach.scalar_value.is_some() { "point" } else { "incomplete" };
    let reasoning = ReasoningSectionWire {
        identification: SlotSectionWire {
            value: Some(IdentificationSlotWire {
                status: body.identification.status.clone(),
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
                matrix_status: "unknown".into(),
                matrix_coordinate: None,
                empirical: "unavailable:not_evaluated".into(),
            }),
            unavailable: None,
        },
        uncertainty: SlotSectionWire {
            value: None,
            unavailable: Some("attested_not_reverifiable".into()),
        },
        assumptions: SlotSectionWire {
            value: Some(crate::AssumptionSlotWire { obligations: Vec::new() }),
            unavailable: None,
        },
    };
    let identities = ContractIdentitiesWire {
        target: *target_digest.as_bytes(),
        identification,
        identification_product: None,
        program: [0; 32],
        inference_binding: [0; 32],
        observation: *observation_digest.as_bytes(),
        data_snapshot: attach.data_snapshot,
        execution: None,
        score_reuse: None,
        target_weights: None,
    };
    let seal = contract_seal(
        &identities,
        &reasoning,
        "unknown",
        "explicit",
        Some(attach.identifier.as_str()),
        None,
    )?;
    Ok(AnalysisResultContractWire {
        format: CONTRACT_SECTION_FORMAT,
        identities,
        seal,
        target,
        reasoning,
        graph_class: "unknown".into(),
        structure_source: "explicit".into(),
        identifier: Some(attach.identifier.clone()),
        estimator: None,
        claim: Some(ClaimSectionWire {
            claim_id: [0; 32],
            kind: kind.into(),
            value_bits: attach.scalar_value.filter(|value| value.is_finite()).map(f64::to_bits),
            execution: None,
            identification_domain: "identified".into(),
            support_domain: "unknown".into(),
            evaluated_domain: "unknown".into(),
            calibration: CalibrationSlotWire::unavailable(antecedent_core::reason_code!(
                "attested_not_reverifiable"
            )),
            attested: vec![evidence.clone()],
        }),
        identification: None,
        identification_product: Some(identification_product),
        program: None,
        inference_binding: None,
        observation: Some(observation),
        data_snapshot: None,
        execution: None,
        score_reuse: None,
        target_weights: None,
    })
}

fn seal_claim(
    contract: &mut AnalysisResultContractWire,
    body: &AnalysisResultWire,
) -> Result<(), IoError> {
    contract.seal = contract_seal(
        &contract.identities,
        &contract.reasoning,
        &contract.graph_class,
        &contract.structure_source,
        contract.identifier.as_deref(),
        contract.estimator.as_deref(),
    )?;
    let Some(claim) = contract.claim.clone() else {
        return Ok(());
    };
    let result = result_digest(body)?;
    let claim_id =
        *claim_digest(&ClaimIdentityWire::new(contract.seal, &claim, result))?.as_bytes();
    if let Some(claim) = contract.claim.as_mut() {
        claim.claim_id = claim_id;
    }
    Ok(())
}

fn write_receipt(
    body: &AnalysisResultWire,
    names: Vec<String>,
    contract: &AnalysisResultContractWire,
) -> Result<Vec<u8>, IoError> {
    // Body must be a well-formed analysis_result. The contract is an attested
    // receipt: consume will not treat missing program payloads as verified.
    let mut artifact = encode_analysis_result_artifact(body, names, "external-estimate")?;
    validate_contract_section_format(contract)?;
    let (descriptor, section) = pack_section_shared(
        CONTRACT_SECTION,
        "application/cbor",
        to_cbor(contract)?.into(),
        CompressPolicy::Auto,
    );
    artifact.manifest.sections.push(descriptor);
    artifact.sections.push(section);
    let mut bytes = Vec::new();
    artifact.write_to(&mut bytes)?;
    Ok(bytes)
}

fn validate_contract_section_format(contract: &AnalysisResultContractWire) -> Result<(), IoError> {
    if contract.format != CONTRACT_SECTION_FORMAT {
        return Err(IoError::Convert(format!(
            "unsupported `{CONTRACT_SECTION}` format {}",
            contract.format
        )));
    }
    if contract
        .claim
        .as_ref()
        .is_some_and(|claim| claim.attested.iter().any(|item| item.reverifiable))
    {
        return Err(IoError::Convert("attested evidence must not be marked reverifiable".into()));
    }
    Ok(())
}

fn index_of(names: &[String], name: &str) -> Result<u32, IoError> {
    names
        .iter()
        .position(|item| item == name)
        .map(|idx| u32::try_from(idx).expect("schema index fits u32"))
        .ok_or_else(|| IoError::Convert(format!("unknown variable {name}")))
}

fn wire_status(status: &str) -> String {
    if status.contains('_') {
        return status.to_ascii_lowercase();
    }
    let mut out = String::new();
    for (idx, ch) in status.chars().enumerate() {
        if ch.is_uppercase() && idx > 0 {
            out.push('_');
        }
        out.extend(ch.to_lowercase());
    }
    out
}

/// Parse a 64-digit hex digest.
///
/// # Errors
///
/// Wrong length or non-hex input.
pub fn parse_digest_hex(hex: &str) -> Result<[u8; 32], IoError> {
    if hex.len() != 64 {
        return Err(IoError::Convert("digest hex must be 64 characters".into()));
    }
    let mut out = [0u8; 32];
    for (idx, chunk) in hex.as_bytes().chunks(2).enumerate() {
        let text = std::str::from_utf8(chunk).map_err(|err| IoError::Convert(err.to_string()))?;
        out[idx] = u8::from_str_radix(text, 16).map_err(|err| IoError::Convert(err.to_string()))?;
    }
    Ok(out)
}

/// Decode an external-estimate receipt and return its claim section.
///
/// # Errors
///
/// Decode failure or a missing claim.
pub fn decode_external_estimate_claim(
    bytes: &[u8],
) -> Result<(AnalysisResultWire, AnalysisResultContractWire, ClaimSectionWire), IoError> {
    let (artifact, _, body) = crate::decode_analysis_result_artifact(bytes)?;
    let contract = crate::decode_analysis_result_contract(&artifact)?.ok_or_else(|| {
        IoError::Convert("external estimate receipt is missing a contract".into())
    })?;
    let claim = contract
        .claim
        .clone()
        .ok_or_else(|| IoError::Convert("external estimate receipt is missing a claim".into()))?;
    Ok((body, contract, claim))
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::from_cbor;

    fn attach_with_payload(payload: &[u8]) -> ExternalEstimateAttach {
        ExternalEstimateAttach {
            learner: "econml.dml.CausalForestDML".into(),
            config: br#"{"n_estimators":100}"#.to_vec(),
            payload: payload.to_vec(),
            names: vec!["t".into(), "y".into(), "z".into()],
            treatment: "t".into(),
            outcome: "y".into(),
            confounders: vec!["z".into()],
            identifier: "backdoor.adjustment".into(),
            status: "NonparametricallyIdentified".into(),
            identification: None,
            data_snapshot: [7; 32],
            scalar_value: None,
        }
    }

    #[test]
    fn external_estimate_wire_keeps_kind_and_is_not_reverifiable() {
        let evidence = attested_external_estimate(&attach_with_payload(&[1, 2, 3]));
        assert_eq!(evidence.kind, "external_estimate");
        assert!(!evidence.reverifiable);
        assert_eq!(evidence.name, "econml.dml.CausalForestDML");
        assert!(evidence.config_digest.is_some());
        assert!(evidence.payload_digest.is_some());
        let encoded = to_cbor(&evidence).unwrap();
        let decoded: AttestedEvidenceWire = from_cbor(&encoded).unwrap();
        assert_eq!(decoded, evidence);
    }

    #[test]
    fn claim_id_moves_when_payload_bytes_change() {
        let first =
            encode_external_estimate_claim(&attach_with_payload(&[1.0f64.to_le_bytes()].concat()))
                .unwrap();
        let second =
            encode_external_estimate_claim(&attach_with_payload(&[2.0f64.to_le_bytes()].concat()))
                .unwrap();
        let (_, _, first_claim) = decode_external_estimate_claim(&first).unwrap();
        let (_, _, second_claim) = decode_external_estimate_claim(&second).unwrap();
        assert_eq!(first_claim.attested[0].kind, "external_estimate");
        assert!(!first_claim.attested[0].reverifiable);
        assert_eq!(first_claim.calibration.reason.as_deref(), Some("attested_not_reverifiable"));
        assert_eq!(first_claim.identification_domain, "identified");
        assert_eq!(first_claim.support_domain, "unknown");
        assert_eq!(first_claim.evaluated_domain, "unknown");
        assert!(first_claim.value_bits.is_none());
        assert_ne!(first_claim.claim_id, second_claim.claim_id);
        assert_ne!(first_claim.attested[0].payload_digest, second_claim.attested[0].payload_digest);
    }

    #[test]
    fn receipt_does_not_advertise_an_econml_estimator() {
        let bytes = encode_external_estimate_claim(&attach_with_payload(&[0])).unwrap();
        let (_, contract, _) = decode_external_estimate_claim(&bytes).unwrap();
        assert!(contract.estimator.is_none());
        assert!(contract.score_reuse.is_none());
        assert_eq!(contract.identities.data_snapshot, [7; 32]);
    }
}
