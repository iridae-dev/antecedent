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
use crate::analysis_wire::{DiagnosticWire, IdentificationResultWire};
use crate::container::{CompressPolicy, pack_section_shared};
use crate::contract_section::{
    AnalysisResultContractWire, AttestedEvidenceWire, CONTRACT_SECTION, CONTRACT_SECTION_FORMAT,
    CalibrationSlotWire, ClaimSectionWire, ContractIdentitiesWire, IdentificationSlotWire,
    ReasoningSectionWire, SlotSectionWire, SupportSlotWire, claim_domains, contract_seal,
    digest_hex, result_digest,
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
    /// Value type of each name in `names`, when the caller declared them (`continuous`,
    /// `binary`, `count`). Empty means none were declared: every variable is then recorded as
    /// [`ValueType::Unspecified`] rather than guessed. Otherwise the length must match `names`.
    pub value_types: Vec<ValueType>,
    /// `(control, active)` treatment levels the estimate contrasts, when the caller stated
    /// them. `None` records the conventional 0 -> 1 as a named placeholder, not a claim.
    pub contrast: Option<(f64, f64)>,
    /// Effect-modifier names the estimate is conditional on (empty for a marginal effect).
    pub modifiers: Vec<String>,
}

/// Encode a contracted `analysis_result` whose claim is an attested external estimate.
///
/// Only a fully identified parent (`nonparametrically_identified` or
/// `identified_under_parametric_restrictions`) can carry an attested estimate: a partially
/// identified, graph-dependent or refused parent licenses no point value to attach to.
///
/// # Errors
///
/// A parent status that is not fully identified, unknown treatment/outcome names, invalid
/// schema, or encode failure.
pub fn encode_external_estimate_claim(attach: &ExternalEstimateAttach) -> Result<Vec<u8>, IoError> {
    require_identified_status(&attach.status)?;
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

fn require_identified_status(status: &str) -> Result<(), IoError> {
    match wire_status(status).as_str() {
        "nonparametrically_identified" | "identified_under_parametric_restrictions" => Ok(()),
        other => Err(IoError::Convert(format!(
            "an external estimate can only be attached to a fully identified result (got `{other}`)"
        ))),
    }
}

fn receipt_body(
    attach: &ExternalEstimateAttach,
) -> Result<(AnalysisResultWire, TargetIdentityWire, Vec<String>), IoError> {
    if !attach.value_types.is_empty() && attach.value_types.len() != attach.names.len() {
        return Err(IoError::Convert(
            "external estimate value types must be empty or one per schema name".into(),
        ));
    }
    let mut builder = CausalSchemaBuilder::new();
    for (position, name) in attach.names.iter().enumerate() {
        let value_type =
            attach.value_types.get(position).cloned().unwrap_or(ValueType::Unspecified);
        if value_type.requires_category_domain() {
            return Err(IoError::Convert(format!(
                "external estimate variable `{name}` is categorical or ordinal, which needs a \
                 category domain the receipt does not carry"
            )));
        }
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
                value_type,
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
    let treatment_id = VariableId::from_raw(treatment);
    let outcome_id = VariableId::from_raw(outcome);
    let (control, active) = attach.contrast.unwrap_or((0.0, 1.0));
    let modifiers = attach
        .modifiers
        .iter()
        .map(|name| index_of(&attach.names, name).map(VariableId::from_raw))
        .collect::<Result<Vec<_>, _>>()?;
    let query = CausalQuery::AverageEffect(
        AverageEffectQuery::with_levels(treatment_id, outcome_id, control, active)
            .with_effect_modifiers(modifiers),
    );
    let query_wire = causal_query_to_wire(&query)?;
    let status = wire_status(&attach.status);
    let body = AnalysisResultWire {
        query: query_wire.clone(),
        identification: IdentificationResultWire {
            status: status.clone(),
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
        estimate: attach.scalar_value,
        standard_error: None,
        interval_lower: None,
        interval_upper: None,
        assumptions: Vec::new(),
        diagnostics: vec![target_attestation(attach)],
        refutations: Vec::new(),
        response: None,
        posterior_artifact: None,
        mediation_grid: None,
        structural_response: None,
        unit_effects: None,
        cate: None,
        fitted_effect: None,
        cate_se: None,
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
    Ok((body, target, attach.names.clone()))
}

/// What the receipt states about its target, and what it does not. The learner's own target is
/// never observed: the query is exactly what the caller declared (levels, modifiers), and a
/// contrast the caller did not state is recorded as the conventional 0 -> 1 placeholder, named
/// as such. Variable types not declared are `Unspecified`, never assumed continuous.
fn target_attestation(attach: &ExternalEstimateAttach) -> DiagnosticWire {
    let contrast = match attach.contrast {
        Some((control, active)) => format!("do({control}) versus do({active}) as declared"),
        None => "no contrast declared (do(0) versus do(1) recorded as a placeholder)".to_string(),
    };
    let types = if attach.value_types.is_empty() {
        "variable value types were not declared and are recorded as unspecified"
    } else {
        "variable value types are as declared by the caller"
    };
    let (labelled, message) = if attach.scalar_value.is_some() {
        (
            "average_effect",
            format!(
                "target attested by the caller as a scalar average effect; {contrast}; {types}"
            ),
        )
    } else {
        (
            "unlabelled",
            format!(
                "the estimate was not labelled as a scalar average effect and may be a curve, \
                 heterogeneous-effect array or multi-valued contrast; {contrast}; {types}; the \
                 recorded query is not what the learner estimated"
            ),
        )
    };
    let contrast_field = if attach.contrast.is_some() { "declared" } else { "placeholder" };
    DiagnosticWire {
        code: "external_estimate.target_attested".into(),
        kind: "scientific".into(),
        severity: "info".into(),
        message,
        artifact_id: None,
        fields: vec![
            ("target".into(), labelled.into()),
            ("contrast".into(), contrast_field.into()),
        ],
    }
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
    // The parent is fully identified (checked at entry), so the slot is full-mass; the claim
    // domains are derived from the slots by the same rule the consumer re-derives them with.
    let identification_slot = IdentificationSlotWire {
        status: body.identification.status.clone(),
        identified_mass: 1.0,
        unidentified_mass: 0.0,
        unevaluable_mass: 0.0,
        incomplete_search_mass: 0.0,
        full_mass_scope: true,
        search_capped: false,
        weight_basis: None,
    };
    let support_slot = SupportSlotWire {
        matrix_status: "unknown".into(),
        matrix_coordinate: None,
        empirical: "unavailable:not_evaluated".into(),
    };
    let domains = claim_domains(Some(&support_slot), Some(&identification_slot));
    let reasoning = ReasoningSectionWire {
        identification: SlotSectionWire { value: Some(identification_slot), unavailable: None },
        support: SlotSectionWire { value: Some(support_slot), unavailable: None },
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
            identification_domain: domains.identification,
            support_domain: domains.support,
            evaluated_domain: domains.evaluated,
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
    crate::analysis_wire::identification_status_from_any(status).map_or_else(
        || status.to_ascii_lowercase(),
        |known| crate::analysis_wire::identification_status_snake(known).into(),
    )
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
            value_types: Vec::new(),
            contrast: None,
            modifiers: Vec::new(),
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
        assert_eq!(first_claim.evaluated_domain, "evaluated");
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

    #[test]
    fn receipt_refuses_parents_that_are_not_fully_identified() {
        for status in [
            "PartiallyIdentified",
            "not_identified",
            "GraphDependent",
            "IdentifiedUnderPriorRestrictions",
            "bogus",
        ] {
            let mut attach = attach_with_payload(&[0]);
            attach.status = status.into();
            attach.scalar_value = Some(1.2);
            let error = encode_external_estimate_claim(&attach).unwrap_err().to_string();
            assert!(error.contains("fully identified"), "{status}: {error}");
        }
        for status in ["NonparametricallyIdentified", "identified_under_parametric_restrictions"] {
            let mut attach = attach_with_payload(&[0]);
            attach.status = status.into();
            assert!(encode_external_estimate_claim(&attach).is_ok(), "{status}");
        }
    }

    #[test]
    fn receipt_domains_are_the_ones_the_consumer_derives_from_its_slots() {
        let bytes = encode_external_estimate_claim(&attach_with_payload(&[0])).unwrap();
        let (_, contract, claim) = decode_external_estimate_claim(&bytes).unwrap();
        let derived = claim_domains(
            contract.reasoning.support.value.as_ref(),
            contract.reasoning.identification.value.as_ref(),
        );
        assert_eq!(claim.identification_domain, derived.identification);
        assert_eq!(claim.support_domain, derived.support);
        assert_eq!(claim.evaluated_domain, derived.evaluated);
    }

    #[test]
    fn receipt_states_that_its_target_and_value_types_are_attested() {
        let unlabelled = encode_external_estimate_claim(&attach_with_payload(&[0])).unwrap();
        let (body, _, _) = decode_external_estimate_claim(&unlabelled).unwrap();
        let diagnostic = body
            .diagnostics
            .iter()
            .find(|d| d.code == "external_estimate.target_attested")
            .expect("target attestation");
        assert_eq!(
            diagnostic.fields,
            vec![
                ("target".to_string(), "unlabelled".to_string()),
                ("contrast".to_string(), "placeholder".to_string()),
            ]
        );
        let mut labelled = attach_with_payload(&[0]);
        labelled.scalar_value = Some(1.2);
        let (body, _, _) =
            decode_external_estimate_claim(&encode_external_estimate_claim(&labelled).unwrap())
                .unwrap();
        assert_eq!(
            body.diagnostics[0].fields,
            vec![
                ("target".to_string(), "average_effect".to_string()),
                ("contrast".to_string(), "placeholder".to_string()),
            ]
        );
    }

    #[test]
    fn receipt_carries_declared_types_contrast_and_modifiers() {
        use crate::wire::ValueTypeWire;
        let mut attach = attach_with_payload(&[0]);
        attach.scalar_value = Some(0.4);
        attach.value_types = vec![ValueType::Binary, ValueType::Continuous, ValueType::Count];
        attach.contrast = Some((2.0, 5.0));
        attach.modifiers = vec!["z".into()];
        let (body, contract, _) =
            decode_external_estimate_claim(&encode_external_estimate_claim(&attach).unwrap())
                .unwrap();
        let types: Vec<_> =
            contract.target.schema.variables.iter().map(|v| v.value_type.clone()).collect();
        assert_eq!(
            types,
            vec![ValueTypeWire::Binary, ValueTypeWire::Continuous, ValueTypeWire::Count]
        );
        let expected = causal_query_to_wire(&CausalQuery::AverageEffect(
            AverageEffectQuery::with_levels(
                VariableId::from_raw(0),
                VariableId::from_raw(1),
                2.0,
                5.0,
            )
            .with_effect_modifiers(vec![VariableId::from_raw(2)]),
        ))
        .unwrap();
        assert_eq!(body.query, expected);
        assert_eq!(contract.target.query, expected);
        assert_eq!(body.diagnostics[0].fields[1].1, "declared");
    }

    #[test]
    fn undeclared_types_are_unspecified_not_continuous() {
        use crate::wire::ValueTypeWire;
        let bytes = encode_external_estimate_claim(&attach_with_payload(&[0])).unwrap();
        let (_, contract, _) = decode_external_estimate_claim(&bytes).unwrap();
        assert!(
            contract
                .target
                .schema
                .variables
                .iter()
                .all(|v| v.value_type == ValueTypeWire::Unspecified)
        );
    }

    #[test]
    fn mismatched_or_domainless_types_are_refused() {
        let mut attach = attach_with_payload(&[0]);
        attach.value_types = vec![ValueType::Binary];
        assert!(encode_external_estimate_claim(&attach).is_err());
        attach.value_types = vec![ValueType::Categorical, ValueType::Continuous, ValueType::Count];
        assert!(encode_external_estimate_claim(&attach).is_err());
    }

    #[test]
    fn host_projection_withholds_domains_of_an_unverified_receipt() {
        let mut attach = attach_with_payload(&[0]);
        attach.scalar_value = Some(1.2);
        let bytes = encode_external_estimate_claim(&attach).unwrap();
        let consumed = crate::consume_analysis_result(&bytes).unwrap();
        assert!(!consumed.acceptance.verified_references);
        let host = crate::project_claim_host(&consumed);
        assert_eq!(host.identification_domain, serde_json::Value::Null);
        assert_eq!(host.support_domain, serde_json::Value::Null);
        assert_eq!(host.evaluated_domain, serde_json::Value::Null);
    }
}
