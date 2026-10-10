//! C3 facade: a composed native-plus-external decision is built from real
//! artifacts, exported, and consumed independently. Hand-enumerated oracles:
//!
//! * joint decision over four equally likely rows: `risky = E[p * q] = 2`,
//!   `safe = max(3, 0) = 3`;
//! * point-only decision over `E[Y | do(a)] = 1 + 2a`: `wait` (reads `a = 0`) is
//!   `2 * 1 - 1 = 1`, `treat` (reads `a = 2`) is `2 * 5 - 1 = 9`.

use antecedent::analysis::composition::{
    BundleBuilder, BundleError, BundleStage, ClaimLabel, ConsumedBundle, EvidenceRelationship,
    FACT_LAW, FACT_REQUIRES_LAW, FACT_TRUST, LAW_JOINT, LAW_MEAN_ONLY, NodeKind, NodeStatus,
    ProviderOrData, SuppliedSources, TRUST_ATTESTED, consume, export_bundle, mean_result_to_bytes,
    mean_source_of,
};
use antecedent_core::{
    CheckedCausalContract, DistributionMeaning, ExternalCapability, ExternalResponse,
    ExternalResult, ExternalResultHeader, ExternalScientificObject, ExternalTrustState,
    ExternalUncertaintyMeaning, IdentificationStatus, LawProviderContract, ProviderObjectIdentity,
    QuantityRole, ScientificQuantity, SupportStatus, bind_external_result,
};
use antecedent_design::decision_artifact::{DecisionContractArtifact, DecisionResultArtifact};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, HardConstraint,
    StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::{evaluate_contract, evaluate_contract_on_means};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::external_claim_artifact::ExternalClaimArtifact;
use antecedent_io::quantity_wire::DistributionMeaningWire;

fn close(a: f64, b: f64) {
    assert!((a - b).abs() < 1e-9, "{a} != {b}");
}

fn outcome(variable: &str, regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn columns() -> Vec<ScientificQuantity> {
    vec![outcome("p", "do(a=1)"), outcome("q", "do(a=1)"), outcome("safe", "do(a=0)")]
}

fn source() -> DistributionArtifact {
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &columns(),
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "enumerated".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "enumeration-1".into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .unwrap();
    let p = [1.0, 3.0, 2.0, 0.0];
    let q = [4.0, 0.0, 2.0, 6.0];
    let mut draws = Vec::new();
    for i in 0..4 {
        draws.extend([p[i], q[i], 3.0]);
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [4, 3],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .unwrap()
}

fn contract() -> DecisionContract {
    let cols = columns();
    DecisionContract {
        actions: vec![
            DecisionAction {
                id: "risky".into(),
                kind: ActionKind::Intervention,
                inputs: vec![cols[0].clone(), cols[1].clone()],
                utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
            },
            DecisionAction {
                id: "safe".into(),
                kind: ActionKind::Policy,
                inputs: vec![cols[2].clone()],
                utility: UtilityExpr::maximum(UtilityExpr::Input(0), UtilityExpr::Const(0.0)),
            },
        ],
        utility_units: "units".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![HardConstraint {
            id: "q-cap".into(),
            expr: UtilityExpr::Input(1),
            bound: 5.0,
            min_probability: 0.75,
            units: "units".into(),
            applies_to: vec!["risky".into()],
        }],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

struct Native {
    contract: Vec<u8>,
    distribution: Vec<u8>,
    result: Vec<u8>,
}

fn native() -> Native {
    let c = contract();
    let s = source();
    let result = evaluate_contract(&c, &s).unwrap();
    Native {
        contract: DecisionContractArtifact::new(c).unwrap().to_bytes("contract").unwrap(),
        distribution: s.to_bytes("distribution").unwrap(),
        result: DecisionResultArtifact::new(result, &s).to_bytes("result").unwrap(),
    }
}

fn claim_quantity(dose: u32) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "schema:y".into(),
        variable_name: "Y".into(),
        role: QuantityRole::Outcome,
        units: "mmHg".into(),
        population_id: "target".into(),
        regime_id: format!("do(a={dose})"),
        horizon: 0,
        functional_id: "mean".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn claim(request: &str, evidence: &str) -> ExternalClaimArtifact {
    let quantities: Vec<ScientificQuantity> = (0..3).map(claim_quantity).collect();
    let object = ExternalScientificObject::Law(LawProviderContract {
        identity: ProviderObjectIdentity {
            provider_id: "lab".into(),
            object_id: "curve".into(),
            version_id: "v3".into(),
            snapshot_id: "snap-9".into(),
            request_id: request.into(),
        },
        quantities: quantities.clone(),
        meaning: DistributionMeaning::InterventionalPredictive,
        capabilities: vec![ExternalCapability::Mean],
    });
    let contract = CheckedCausalContract {
        graph_id: "graph-1".into(),
        identification: IdentificationStatus::NonparametricallyIdentified,
        estimand: quantities.clone(),
        accepted_meanings: vec![DistributionMeaning::InterventionalPredictive],
        required_evidence_ids: vec![evidence.into()],
        required_assumption_ids: vec!["ignorability".into()],
        equivalences: vec![],
    };
    let response = ExternalResponse {
        header: ExternalResultHeader {
            object,
            graph_id: "graph-1".into(),
            quantities,
            evidence_ids: vec![evidence.into()],
            assumption_ids: vec!["ignorability".into()],
            trust: ExternalTrustState::ExternallyAttested { attestor: "lab".into() },
        },
        values: vec![1.0, 3.0, 5.0],
        uncertainty: ExternalUncertaintyMeaning::None,
        point_support: Some(vec![
            SupportStatus::Supported,
            SupportStatus::Supported,
            SupportStatus::Supported,
        ]),
    };
    let bound = bind_external_result(&contract, &ExternalResult::Response(response)).unwrap();
    ExternalClaimArtifact::from_bound_claim(&bound, "checked-contract").unwrap()
}

fn mean_contract() -> DecisionContract {
    let affine = || {
        UtilityExpr::difference(
            UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(2.0)),
            UtilityExpr::Const(1.0),
        )
    };
    let action = |id: &str, dose: u32| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![claim_quantity(dose)],
        utility: affine(),
    };
    DecisionContract {
        actions: vec![action("wait", 0), action("treat", 2)],
        utility_units: "utility".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn stage_of(error: &BundleError) -> BundleStage {
    error.stage().expect("a structured refusal")
}

fn failed_stage(consumed: &ConsumedBundle, id: &str) -> BundleStage {
    match &consumed.node(id).unwrap().status {
        NodeStatus::Failed { stage, .. } => *stage,
        other => panic!("node `{id}` is {other:?}"),
    }
}

fn native_builder(n: &Native, reverse: bool) -> BundleBuilder {
    let mut builder = BundleBuilder::new();
    let mut parts: Vec<(&str, &[u8])> =
        vec![("contract", &n.contract), ("distribution", &n.distribution), ("result", &n.result)];
    if reverse {
        parts.reverse();
    }
    for (id, bytes) in parts {
        builder.add_detected_artifact(Some(id), bytes).unwrap();
    }
    builder.connect("distribution", "result").unwrap();
    builder.connect("contract", "result").unwrap();
    builder
}

#[test]
fn c3_facade_a_built_bundle_is_consumed_independently_and_reports_the_decision() {
    let n = native();
    let bundle = native_builder(&n, false).build().unwrap();
    let bytes = export_bundle(&bundle, "composed").unwrap();

    let consumed = consume(&bytes, bundle.identity(), &SuppliedSources::default()).unwrap();
    consumed.require_verified().unwrap();
    close(consumed.value("result", "risky.expected_utility").unwrap(), 2.0);
    close(consumed.value("result", "safe.expected_utility").unwrap(), 3.0);
    assert_eq!(consumed.node("result").unwrap().claim_label, Some(ClaimLabel::JointDraw));
    assert_eq!(
        consumed.node("distribution").unwrap().facts.get(FACT_LAW).map(String::as_str),
        Some(LAW_JOINT)
    );
    assert_eq!(
        consumed.node("result").unwrap().facts.get(FACT_REQUIRES_LAW).map(String::as_str),
        Some(LAW_JOINT)
    );

    // Insertion order does not change the identity or the bytes.
    let reversed = native_builder(&n, true).build().unwrap();
    assert_eq!(reversed.identity(), bundle.identity());
    assert_eq!(export_bundle(&reversed, "composed").unwrap(), bytes);
}

#[test]
fn c3_facade_derived_ids_are_stable_and_the_builder_refuses_malformed_graphs() {
    let n = native();
    let mut builder = BundleBuilder::new();
    let id = builder.add_detected_artifact(None, &n.distribution).unwrap();
    assert!(id.starts_with("distribution:"), "{id}");
    let mut again = BundleBuilder::new();
    assert_eq!(again.add_detected_artifact(None, &n.distribution).unwrap(), id);

    // The same id twice, an unknown or identical edge end, a repeated edge.
    let repeat = builder.add_detected_artifact(Some(&id), &n.contract).unwrap_err();
    assert_eq!(stage_of(&repeat), BundleStage::EdgeDigestMismatch);
    assert_eq!(
        stage_of(&builder.connect(&id, "missing").unwrap_err()),
        BundleStage::EdgeDigestMismatch
    );
    assert_eq!(stage_of(&builder.connect(&id, &id).unwrap_err()), BundleStage::EdgeDigestMismatch);
    builder.add_detected_artifact(Some("contract"), &n.contract).unwrap();
    builder.connect(&id, "contract").unwrap();
    assert_eq!(
        stage_of(&builder.connect(&id, "contract").unwrap_err()),
        BundleStage::EdgeDigestMismatch
    );

    // Bytes that are not the stated kind of artifact refuse at the builder.
    let wrong = builder.add_artifact(Some("bad"), NodeKind::DecisionContract, &n.distribution);
    assert_eq!(stage_of(&wrong.unwrap_err()), BundleStage::SwappedEvidence);
    let unknown = builder.add_detected_artifact(Some("junk"), b"not a container").unwrap_err();
    assert_eq!(stage_of(&unknown), BundleStage::UnknownNodeKind);
    assert_eq!(builder.node_ids().len(), 2);

    // A cycle is refused when the bundle is built.
    builder.connect("contract", &id).unwrap();
    assert_eq!(stage_of(&builder.build().unwrap_err()), BundleStage::EdgeDigestMismatch);
}

#[test]
fn c3_facade_external_mean_decision_is_point_only_attested_and_a_reference_needs_its_source() {
    let attested = claim("req-1", "factor:z");
    let source_means = mean_source_of(&attested).unwrap();
    let result = evaluate_contract_on_means(&mean_contract(), &source_means).unwrap();
    let result_bytes = mean_result_to_bytes(&result, &source_means, "mean-result").unwrap();

    let mut builder = BundleBuilder::new();
    builder.add_detected_artifact(Some("claim"), &attested.to_bytes("claim").unwrap()).unwrap();
    builder
        .add_detected_artifact(
            Some("contract"),
            &DecisionContractArtifact::new(mean_contract()).unwrap().to_bytes("c").unwrap(),
        )
        .unwrap();
    builder.add_detected_artifact(Some("result"), &result_bytes).unwrap();
    // The native fit the claim is bound to is held elsewhere: a reference.
    builder
        .add_reference(
            "causal",
            NodeKind::CausalContract,
            "program-1",
            ProviderOrData::Data { snapshot_id: "graph-1".into(), digest: "g1".into() },
        )
        .unwrap();
    builder.declare_inspected("causal", "identified_effect", 2.0).unwrap();
    assert_eq!(
        stage_of(&builder.declare_fact("claim", "k", "v").unwrap_err()),
        BundleStage::EdgeDigestMismatch
    );
    for (from, to) in [("claim", "result"), ("contract", "result"), ("causal", "claim")] {
        builder.connect(from, to).unwrap();
    }
    let bundle = builder.build().unwrap();
    let bytes = export_bundle(&bundle, "mean").unwrap();

    let without = consume(&bytes, bundle.identity(), &SuppliedSources::default()).unwrap();
    assert!(matches!(
        without.node("causal").unwrap().status,
        NodeStatus::ReferenceUnresolved { requires: ProviderOrData::Data { .. } }
    ));
    // The inspected decision survives the unavailable reference.
    assert_eq!(without.node("result").unwrap().status, NodeStatus::Verified);
    close(without.value("result", "treat.expected_utility").unwrap(), 9.0);
    close(without.value("result", "wait.expected_utility").unwrap(), 1.0);
    assert_eq!(
        without.node("causal").unwrap().inspected,
        vec![("identified_effect".to_owned(), 2.0)]
    );
    assert_eq!(
        stage_of(&without.require_verified().unwrap_err()),
        BundleStage::CallbackUnavailable
    );
    assert_eq!(without.node("result").unwrap().claim_label, Some(ClaimLabel::PointOnlyAttested));
    assert_eq!(
        without.node("claim").unwrap().facts.get(FACT_LAW).map(String::as_str),
        Some(LAW_MEAN_ONLY)
    );
    assert_eq!(
        without.node("claim").unwrap().facts.get(FACT_TRUST).map(String::as_str),
        Some(TRUST_ATTESTED)
    );

    let supplied = SuppliedSources::default().with_data("graph-1", "g1");
    let with = consume(&bytes, bundle.identity(), &supplied).unwrap();
    with.require_verified().unwrap();

    // Supplied data that is not the referenced snapshot fails the reference.
    let other = SuppliedSources::default().with_data("graph-1", "g2");
    let changed = consume(&bytes, bundle.identity(), &other).unwrap();
    assert_eq!(failed_stage(&changed, "causal"), BundleStage::GraphOrSnapshotMismatch);
}

#[test]
fn c3_facade_refusals_name_their_stage_and_a_retained_identity_is_required() {
    let n = native();
    let bundle = native_builder(&n, false).build().unwrap();
    let bytes = export_bundle(&bundle, "composed").unwrap();

    let wrong = consume(&bytes, &"0".repeat(64), &SuppliedSources::default()).unwrap_err();
    assert_eq!(stage_of(&wrong), BundleStage::ExpectedIdentityMismatch);
    let refusal = wrong.to_refusal().unwrap();
    assert_eq!(refusal.detail, "composition_bundle.expected_identity_mismatch");

    // A byte flipped in the container is a container refusal, not a verified bundle.
    let mut flipped = bytes.clone();
    let middle = flipped.len() / 2;
    flipped[middle] ^= 0xff;
    assert!(consume(&flipped, bundle.identity(), &SuppliedSources::default()).is_err());

    // A reference that must answer one exact request refuses another.
    let mut builder = BundleBuilder::new();
    builder
        .add_reference(
            "provider",
            NodeKind::ExternalClaim,
            "claim-id",
            ProviderOrData::Provider {
                provider_id: "lab".into(),
                snapshot_id: "snap-9".into(),
                request_fingerprint: "fp-1".into(),
            },
        )
        .unwrap();
    builder.declare_fact("provider", FACT_LAW, LAW_MEAN_ONLY).unwrap();
    let bundle = builder.build().unwrap();
    let bytes = export_bundle(&bundle, "provider").unwrap();
    let changed = consume(
        &bytes,
        bundle.identity(),
        &SuppliedSources::default().with_provider("lab", "snap-9", "fp-CHANGED"),
    )
    .unwrap();
    assert_eq!(failed_stage(&changed, "provider"), BundleStage::ProviderRequestChanged);
    let refusal = changed.require_verified().unwrap_err().to_refusal().unwrap();
    assert_eq!(refusal.detail, "composition_bundle.provider_request_changed");
}

#[test]
fn c3_facade_evidence_relationships_are_checked_against_the_studies_they_name() {
    let a = claim("req-a", "factor:z");
    let build = |second: &ExternalClaimArtifact| {
        let mut builder = BundleBuilder::new();
        builder.add_detected_artifact(Some("a"), &a.to_bytes("a").unwrap()).unwrap();
        builder.add_detected_artifact(Some("b"), &second.to_bytes("b").unwrap()).unwrap();
        let id = builder.relate("b", "a", EvidenceRelationship::Independent).unwrap();
        assert_eq!(id, "relation:a:b");
        assert!(builder.relate("a", "b", EvidenceRelationship::SharedData).is_err());
        let bundle = builder.build().unwrap();
        let bytes = export_bundle(&bundle, "studies").unwrap();
        consume(&bytes, bundle.identity(), &SuppliedSources::default()).unwrap()
    };
    build(&claim("req-b", "factor:w")).require_verified().unwrap();
    let overlapping = build(&claim("req-b", "factor:z"));
    assert_eq!(failed_stage(&overlapping, "relation:a:b"), BundleStage::SwappedEvidence);
}
