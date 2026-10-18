//! Contract-section tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::ContractReason;
use super::consume::contract_seal_of;
use super::replay::{
    apply_moment_inverse, invert_linear_moments, replay_functional_response_grid,
    verify_nested_counterfactual_result,
};
use super::verify::verify_claim_calibration;
use crate::identity::{
    ClaimIdentityWire, claim_digest, digest_wire, payload_digest, program_digest,
};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalSchemaBuilder, IDENTITY_FORMAT, IdentityDomain,
    MeasurementSpec, RoleHint, SmallRoleSet, ValueType, VariableId,
};

use super::*;

use crate::expr_wire::ExprArenaWire;
use crate::query_wire::causal_query_to_wire;
use crate::{
    CausalQueryWire, IdentificationResultWire, RefutationReportWire,
    encode_analysis_result_artifact, encode_analysis_result_artifact_with_contract, schema_to_wire,
    to_cbor,
};

#[test]
fn cbor_eq_follows_encoded_bits_not_partial_eq() {
    assert_eq!(0.0, -0.0);
    assert!(!super::consume::cbor_eq(&0.0_f64, &-0.0_f64));
    let nan = f64::from_bits(0x7ff8_0000_0000_0001);
    assert_ne!(nan, nan);
    assert!(super::consume::cbor_eq(&nan, &nan));
}

#[test]
fn acceptance_ops_keep_historical_cbor_text() {
    for op in [
        super::consume::AcceptanceOp::Store,
        super::consume::AcceptanceOp::Forward,
        super::consume::AcceptanceOp::ReadBody,
        super::consume::AcceptanceOp::VerifyContract,
        super::consume::AcceptanceOp::InspectClaim,
        super::consume::AcceptanceOp::Bytes,
        super::consume::AcceptanceOp::AcceptAsClaim,
    ] {
        crate::convert::assert_cbor_text_code(op.as_str(), &op);
    }
}

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
        Err(ContractReason::ProgramFunctionalEffectResponseGridMemberValue)
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
        policy_value: None,
        continuous_dose_response: None,
        panel_did: None,
        synthetic_control: None,
        synthetic_did: None,
        local_polynomial_ratio: None,
        randomized_effect: None,
        survival: None,
        longitudinal_regime: None,
        interference_inference: None,
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

#[allow(
    clippy::too_many_lines,
    reason = "one fixture exercised end to end; later assertions read locals built above them"
)]
fn contract_for(
    target: TargetIdentityWire,
    body: &AnalysisResultWire,
) -> AnalysisResultContractWire {
    use antecedent_graph::{Dag, DenseNodeId};

    use crate::identity::{
        InferentialCommitmentsWire, ObservationIdentityWire, ObservationOptionsWire, dag_identity,
        data_snapshot_digest, execution_digest, identification_digest,
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
        participants: None,
        missing: None,
        nested_counterfactual_fit: None,
        linear_fit_moments: None,
    };
    let snapshot_digest = data_snapshot_digest(&data_snapshot).unwrap();
    let execution =
        crate::execution_identity_from_context(&antecedent_core::ExecutionContext::for_tests(1));
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
                    matrix_coordinate: Some("AverageEffect:Dag:explicit:Frequentist:none".into()),
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
    contract.seal = contract_seal_of(contract).unwrap();
    let Some(claim) = contract.claim.clone() else {
        return;
    };
    let result = result_digest(body).unwrap();
    let claim_id =
        *claim_digest(&ClaimIdentityWire::new(contract.seal, &claim, result)).unwrap().as_bytes();
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
        consumed.acceptance.unresolved.iter().any(|item| &**item == "identities.identification"),
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
    body.standard_error = None;
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
    assert!(!consumed.acceptance.supported_operations.iter().any(|op| &**op == "inspect_claim"));
    let (accepted, receipt) =
        accept_claim(&bytes, &antecedent_contract::ConsumerProfile::full()).unwrap();
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
        accept_claim(&bytes, &antecedent_contract::ConsumerProfile::full()).unwrap();
    assert!(full.acceptance.accepts_as_claim());
    assert!(lossless.equivalent_claim());
    let (restricted, receipt) =
        accept_claim(&bytes, &antecedent_contract::ConsumerProfile::restricted()).unwrap();
    assert!(!restricted.acceptance.accepts_as_claim());
    assert_eq!(restricted.acceptance.restriction.as_deref(), Some("unknown_required_feature"));
    assert!(restricted.acceptance.supported_operations.iter().any(|op| &**op == "forward"));
    assert!(!receipt.equivalent_claim());
    let (_forward, forwarded) =
        accept_claim(&bytes, &antecedent_contract::ConsumerProfile::forwarding()).unwrap();
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
        consume_analysis_result(&replace_contract_section(body, names.to_vec(), contract)).unwrap();
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
fn resolved_only_aipw_contract_requires_checked_lowering_and_rows() {
    let (body, target, names) = fixture_body();
    let mut contract = contract_for(target, &body);
    contract.estimator = None;
    let program = contract.program.as_mut().unwrap();
    program.commitments.estimator = None;
    program.commitments.resolved_estimator = Some("aipw".into());
    contract.claim = None;
    contract.execution = None;
    contract.identities.execution = None;
    contract.identities.program =
        *program_digest(contract.program.as_ref().unwrap()).unwrap().as_bytes();
    seal_claim(&mut contract, &body);

    let bytes = to_bytes(&body, names, Some(&contract));
    let consumed = consume_analysis_result(&bytes).unwrap();
    assert!(!consumed.acceptance.accepts_as_verified_program());
    for key in ["program.checked_aipw_lowering", "checked_aipw.row_binding"] {
        assert!(
            consumed.acceptance.unresolved.iter().any(|reason| reason.as_ref() == key),
            "{key} missing from {:?}",
            consumed.acceptance.unresolved
        );
    }
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
        trim_bits: None,
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

/// The trim rule selects the procedure: a trimmed lowering must run no
/// cross-fit folds and carry its trim, and the cross-fitted one neither.
#[test]
fn checked_aipw_lowering_procedure_must_agree_with_its_folds_and_trim() {
    let cases = [
        ("trimmed_logistic_ols", 0, Some(0.02f64.to_bits()), true),
        ("trimmed_logistic_ols", 5, Some(0.02f64.to_bits()), false),
        ("trimmed_logistic_ols", 0, None, false),
        ("cross_fitted_logistic_ols", 5, None, true),
        ("cross_fitted_logistic_ols", 5, Some(0.02f64.to_bits()), false),
        ("cross_fitted_logistic_ols", 0, None, false),
        ("full_sample_logistic_ols", 0, None, false),
    ];
    for (procedure, folds, trim_bits, consistent) in cases {
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
            procedure: procedure.into(),
            folds,
            se_kind: "hc1".into(),
            se_lag: None,
            bootstrap_replicates: 0,
            trim_bits,
        });
        contract.identities.program =
            *program_digest(contract.program.as_ref().unwrap()).unwrap().as_bytes();
        seal_claim(&mut contract, &body);
        let consumed =
            consume_analysis_result(&replace_contract_section(&body, names.clone(), &contract))
                .unwrap();
        assert_eq!(
            consumed
                .acceptance
                .unresolved
                .iter()
                .any(|key| key.as_ref() == "program.checked_aipw_lowering"),
            !consistent,
            "{procedure} folds={folds} trim={trim_bits:?}: {:?}",
            consumed.acceptance.unresolved
        );
    }
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
        trim_bits: None,
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
fn off_axis_claim_cannot_self_license_calibration() {
    let (body, target, _) = fixture_body();
    let mut contract = contract_for(target, &body);
    let support = contract.reasoning.support.value.as_mut().unwrap();
    support.matrix_status = "off_axis".into();
    support.matrix_coordinate = None;
    contract.claim.as_mut().unwrap().calibration.status = "calibrated".into();
    let mut unresolved = Vec::new();
    verify_claim_calibration(&contract, contract.claim.as_ref().unwrap(), &mut unresolved);
    assert!(unresolved.iter().any(|reason| reason.as_ref() == "claim.calibration.off_axis"));
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
    assert_eq!(claim_domains(Some(&support("failed:overlap.rule")), None).support, "contradicted");
    assert_eq!(claim_domains(Some(&support("unavailable:not_evaluated")), None).support, "unknown");
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
    let slot =
        |status: &str, identified: f64, unevaluable: f64, incomplete: f64| IdentificationSlotWire {
            status: status.into(),
            identified_mass: identified,
            unidentified_mass: 0.0,
            unevaluable_mass: unevaluable,
            incomplete_search_mass: incomplete,
            full_mass_scope: true,
            search_capped: false,
            weight_basis: None,
        };
    let identified = "nonparametrically_identified";
    assert_eq!(claim_kind_name(&body, Some(&slot(identified, 1.0, 0.0, 0.0))), "point");
    assert_eq!(claim_kind_name(&body, Some(&slot(identified, 0.6, 0.0, 0.4))), "mixture");
    assert_eq!(claim_kind_name(&body, Some(&slot(identified, 0.75, 0.25, 0.0))), "mixture");
    for status in ["not_certified", "proven_non_transportable", "surprise"] {
        assert_eq!(claim_kind_name(&body, Some(&slot(status, 1.0, 0.0, 0.0))), "incomplete");
    }
}

fn orthogonal_direct_fit() -> crate::NestedCounterfactualFitWire {
    crate::NestedCounterfactualFitWire {
        format: 1,
        complete_case_rows: 4,
        gram: [4.0, 0.0, 0.0, 0.0, 4.0, 0.0, 0.0, 0.0, 4.0],
        outcome_cross: [0.0, 8.0, 0.0],
    }
}

fn nested_query_wire() -> CausalQueryWire {
    causal_query_to_wire(&CausalQuery::NestedCounterfactual(
        antecedent_core::NestedCounterfactualQuery::with_levels(
            VariableId::from_raw(0),
            VariableId::from_raw(1),
            VariableId::from_raw(2),
            0.0,
            1.0,
        )
        .unwrap(),
    ))
    .unwrap()
}

#[test]
fn bayesian_nested_point_must_match_the_solved_natural_direct_effect() {
    let bind = |estimate: f64| {
        let (mut body, target, _) = fixture_body();
        body.estimate = Some(estimate);
        body.standard_error = None;
        body.interval_lower = None;
        body.interval_upper = None;
        body.posterior_artifact = None;
        let query = nested_query_wire();
        body.query = query.clone();
        body.identification.query = query.clone();
        body.identification.estimands = vec![crate::IdentifiedEstimandWire {
            method: "path_specific.natural".into(),
            adjustment_set: Vec::new(),
            instruments: Vec::new(),
            mediators: vec![1],
            functional: 0,
            rd_design: None,
        }];
        let mut contract = contract_for(target, &body);
        contract.identifier = Some("path_specific.natural".into());
        contract.estimator = Some("mediation.linear".into());
        contract.target.query = query;
        contract.inference_binding.as_mut().unwrap().inference = "bayesian".into();
        let program = contract.program.as_mut().unwrap();
        program.commitments.resolved_estimator = Some("mediation.linear".into());
        program.commitments.inference = "bayesian".into();
        program.checked_nested_counterfactual = Some(crate::CheckedNestedCounterfactualWire {
            format: 1,
            treatment: 0,
            mediator: 1,
            outcome: 2,
            control_bits: 0.0f64.to_bits(),
            active_bits: 1.0f64.to_bits(),
            model: "linear_gaussian".into(),
            procedure: "natural_direct_shared_exogenous".into(),
        });
        if let crate::GraphIdentityWire::Dag(dag) =
            &mut contract.identification.as_mut().unwrap().graph
        {
            dag.node_count = 3;
            dag.edges = vec![(0, 1), (0, 2), (1, 2)];
        }
        contract.data_snapshot.as_mut().unwrap().nested_counterfactual_fit =
            Some(orthogonal_direct_fit());
        (body, contract)
    };
    let (matched, contract) = bind(2.0);
    assert_eq!(verify_nested_counterfactual_result(&contract, &matched), Ok(()));
    let (wrong, contract) = bind(99.0);
    assert_eq!(
        verify_nested_counterfactual_result(&contract, &wrong),
        Err(ContractReason::BodyNestedCounterfactualEstimate)
    );
}

#[test]
fn temporal_nested_point_must_match_the_solved_natural_direct_effect() {
    let (mut body, target, names) = fixture_body();
    body.standard_error = None;
    body.interval_lower = None;
    body.interval_upper = None;
    body.posterior_artifact = None;
    body.estimate = Some(99.0);
    let query = nested_query_wire();
    body.query = query.clone();
    let mut contract = contract_for(target, &body);
    contract.graph_class = "TemporalDag".into();
    contract.target.query = query;
    contract.data_snapshot.as_mut().unwrap().nested_counterfactual_fit =
        Some(orthogonal_direct_fit());
    let header = AnalysisResultHeader { variable_names: names };
    let wrong = verify_contract_against_body(&header, &body, &contract);
    assert!(
        wrong.iter().any(|reason| reason.as_ref() == "body.temporal_nested_point"),
        "{wrong:?}"
    );
    body.estimate = Some(2.0);
    let matched = verify_contract_against_body(&header, &body, &contract);
    assert!(
        !matched.iter().any(|reason| reason.as_ref() == "body.temporal_nested_point"),
        "{matched:?}"
    );
    contract.data_snapshot.as_mut().unwrap().nested_counterfactual_fit = None;
    let unbound = verify_contract_against_body(&header, &body, &contract);
    assert!(
        unbound
            .iter()
            .any(|reason| reason.as_ref() == "dependencies.nested_counterfactual_fit_moments"),
        "{unbound:?}"
    );
}

#[test]
fn mixture_masses_must_sum_to_one() {
    validate_mixture_masses(0.5, 0.3, 0.2, 0.0).unwrap();
    assert!(validate_mixture_masses(0.5, 0.3, 0.1, 0.0).is_err());
    assert!(validate_mixture_masses(1.2, 0.0, 0.0, 0.0).is_err());
    assert!(validate_mixture_masses(f64::NAN, 0.0, 0.0, 0.0).is_err());
}

#[test]
fn moment_replay_inverse_is_scale_free_and_refuses_collinear_moments() {
    // The former `pivot <= 1e-12 * max|entry|` rule called a design with one column in
    // units of 1e6 singular (the small pivot is 1e-12 of the largest entry); the
    // equilibrated inverse recovers it, and a genuinely collinear Gram is still refused.
    let gram = [4.0, 0.0, 0.0, 0.0, 1e-12, 0.0, 0.0, 0.0, 9.0e12];
    let inverse = invert_linear_moments(&gram, 3).expect("a scaled diagonal is invertible");
    let beta = apply_moment_inverse(&inverse, &[4.0, 2e-12, 9.0e12], 3).unwrap();
    for (got, want) in beta.iter().zip([1.0, 2.0, 1.0]) {
        assert!((got - want).abs() < 1e-9, "{beta:?}");
    }
    let collinear = [1.0, 2.0, 0.0, 2.0, 4.0, 0.0, 0.0, 0.0, 1.0];
    assert!(invert_linear_moments(&collinear, 3).is_none());
    assert!(invert_linear_moments(&[0.0; 9], 3).is_none());
    let mut non_finite = gram;
    non_finite[0] = f64::NAN;
    assert!(invert_linear_moments(&non_finite, 3).is_none());
}
