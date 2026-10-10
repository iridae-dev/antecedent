//! Actual original-source execution and affine utility, with independent SCM truth.
use antecedent::analysis::composition::{
    BundleBuilder, BundleLimits, BundleStage, NodeStatus, SuppliedSources, standard_consumer,
};
use antecedent::analysis::source_evidence::{SourceEvidence, SourceResolution};
use antecedent::analysis::source_projection::source_resolved_consumer;
use antecedent::{PreparedStudy, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseCoordinateLabels,
    ResponseFunctional, ResponseQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_design::decision_artifact::contract_to_json;
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::source_projection_artifact::{
    ProjectionSource, SourceProjection, SourceProjectionArtifact,
};
use antecedent_graph::{Dag, DenseNodeId};

#[derive(serde::Serialize, serde::Deserialize)]
struct Mutant {
    version: u16,
    source: ProjectionSource,
    projection: SourceProjection,
    report: antecedent_design::source_projection_artifact::ProjectionReport,
    identity: String,
}

fn pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/composition/source_projection/expected.json"
    ))
    .unwrap()
}
fn prepared(shift: f64, seed: u64) -> (PreparedStudy, ExecutionContext, TabularData) {
    let mut a = Vec::new();
    let mut x = Vec::new();
    let mut y = Vec::new();
    for i in 0_u32..80 {
        for dose in 0_u32..4 {
            let covariate = -1. + 2. * f64::from(i) / 79.;
            a.push(f64::from(dose));
            x.push(covariate);
            y.push(1. + shift + 2. * f64::from(dose) + 0.2 * covariate);
        }
    }
    let data = TabularData::from_f64_columns([
        ("x", x.as_slice()),
        ("a", a.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap();
    let mut graph = Dag::with_variables(3);
    for (from, to) in [(0, 1), (0, 2), (1, 2)] {
        graph.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }
    let ctx = ExecutionContext::for_tests(seed);
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(2),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(1),
            GridSpec::Values(vec![1., 2.].into()),
        ),
    });
    let prepared = Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    (prepared, ctx, data)
}
fn source() -> (SourceEvidence, PreparedStudy, ExecutionContext) {
    let (prepared, ctx, data) = prepared(0., 51);
    let result = prepared.estimate(&data, &ctx).unwrap();
    let bytes = prepared
        .encode_contracted_result_with_quantity_labels(
            &result,
            "native-source-evidence",
            &ctx,
            Some(&ResponseCoordinateLabels {
                outcome_units: "mmHg",
                population_id: "target",
                transform_id: "identity",
            }),
        )
        .unwrap();
    (SourceEvidence::from_result(&bytes).unwrap(), prepared, ctx)
}
fn contract(evidence: &SourceEvidence) -> DecisionContract {
    DecisionContract {
        actions: vec![
            DecisionAction {
                id: "A".into(),
                kind: ActionKind::Intervention,
                inputs: vec![evidence.coordinates()[0].clone()],
                utility: UtilityExpr::difference(
                    UtilityExpr::product(UtilityExpr::Const(2.), UtilityExpr::Input(0)),
                    UtilityExpr::Const(1.),
                ),
            },
            DecisionAction {
                id: "B".into(),
                kind: ActionKind::Intervention,
                inputs: vec![evidence.coordinates()[1].clone()],
                utility: UtilityExpr::product(UtilityExpr::Const(0.5), UtilityExpr::Input(0)),
            },
        ],
        utility_units: pin()["utility_units"].as_str().unwrap().into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}
fn project(evidence: &SourceEvidence, projection: SourceProjection) -> SourceProjectionArtifact {
    SourceProjectionArtifact::produce(
        ProjectionSource::NativeResponse(evidence.original_bytes().to_vec()),
        projection,
    )
    .unwrap()
}
fn bundle(evidence: &SourceEvidence) -> antecedent_design::composition_bundle::CompositionBundle {
    let mut builder = BundleBuilder::new();
    for (name, projection) in [
        ("causal", SourceProjection::CausalContract),
        ("coordinates", SourceProjection::QuantityCoordinates),
        (
            "utility",
            SourceProjection::AffineDecision {
                contract: contract_to_json(&contract(evidence)).unwrap(),
            },
        ),
    ] {
        let artifact = project(evidence, projection);
        builder
            .add_artifact(Some(name), artifact.node_kind(), &artifact.export().unwrap())
            .unwrap();
    }
    builder.connect("causal", "coordinates").unwrap();
    builder.connect("coordinates", "utility").unwrap();
    builder.build().unwrap()
}
#[test]
fn source_projection_original_native_mean_and_affine_utility_match_independent_truth() {
    let (evidence, _, _) = source();
    let artifact = project(
        &evidence,
        SourceProjection::AffineDecision {
            contract: contract_to_json(&contract(&evidence)).unwrap(),
        },
    );
    let truth = pin();
    for (i, id) in ["A", "B"].iter().enumerate() {
        let got = artifact.report().output["actions"][i]["expected_utility"].as_f64().unwrap();
        assert!(
            (got - truth["utilities"][id].as_f64().unwrap()).abs()
                < truth["native_tolerance"].as_f64().unwrap() * 2.
        );
        assert!(artifact.report().output["actions"][i]["standard_error"].is_null());
    }
    assert_eq!(artifact.report().output["calibration"], truth["calibration"]);
    assert_eq!(artifact.report().unresolved, vec!["dependencies.checked_response_grid_operation"]);
    let bytes = artifact.export().unwrap();
    let (replayed, fits) = antecedent_learn::fit_counts::observe_resolved_fits(|| {
        SourceProjectionArtifact::consume(&bytes, artifact.identity()).unwrap()
    });
    assert_eq!(fits, 0);
    assert_eq!(replayed.report(), artifact.report());
}
#[test]
fn source_projection_bundle_requires_actual_checked_execution_and_reuses_one_opaque_receipt() {
    let (evidence, prepared, ctx) = source();
    let bundle = bundle(&evidence);
    let limits = BundleLimits::default();
    let bytes = bundle.to_bytes("source-projection-bundle", &limits).unwrap();
    let imported = standard_consumer()
        .consume(&bytes, &limits, bundle.identity(), &SuppliedSources::default())
        .unwrap();
    assert!(!imported.all_verified());
    assert!(matches!(
        imported.nodes()[0].status,
        NodeStatus::Failed { stage: BundleStage::CallbackUnavailable, .. }
    ));
    let (receipt, fits) = antecedent_learn::fit_counts::observe_resolved_fits(|| {
        SourceResolution::execute(&evidence, &prepared, &ctx).unwrap()
    });
    assert_eq!(fits, 0, "this original local-polynomial response does not execute a learner fit");
    let consumer = source_resolved_consumer(vec![receipt]).unwrap();
    let (consumed, fits) = antecedent_learn::fit_counts::observe_resolved_fits(|| {
        consumer.consume(&bytes, &limits, bundle.identity(), &SuppliedSources::default()).unwrap()
    });
    consumed.require_verified().unwrap();
    assert_eq!(
        fits, 0,
        "projection consumers reuse one actual execution receipt across three nodes"
    );
    assert!((consumed.value("utility", "A.expected_utility").unwrap() - 5.).abs() < 2e-4);
    for node in consumed.nodes() {
        assert_eq!(node.facts["trust"], "unverified");
        assert_eq!(node.facts["native_execution_authority_issued"], "false");
    }
}
#[test]
fn source_projection_changed_data_seed_source_and_unlicensed_mean_functionals_refuse() {
    let (evidence, _, _) = source();
    for (shift, seed) in [(1., 51), (0., 52)] {
        let (changed, ctx, _) = prepared(shift, seed);
        assert_eq!(
            SourceResolution::execute(&evidence, &changed, &ctx).unwrap_err().detail,
            "source_evidence.source_binding_mismatch"
        );
    }
    let mut nonlinear = contract(&evidence);
    nonlinear.actions[0].utility =
        UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(0));
    assert!(
        SourceProjectionArtifact::produce(
            ProjectionSource::NativeResponse(evidence.original_bytes().to_vec()),
            SourceProjection::AffineDecision { contract: contract_to_json(&nonlinear).unwrap() }
        )
        .unwrap_err()
        .to_string()
        .contains("source_projection.affine_contract_refused")
    );
    let mut wrong = contract(&evidence);
    wrong.actions[0].inputs[0].units = "unconverted_units".into();
    assert!(
        SourceProjectionArtifact::produce(
            ProjectionSource::NativeResponse(evidence.original_bytes().to_vec()),
            SourceProjection::AffineDecision { contract: contract_to_json(&wrong).unwrap() }
        )
        .unwrap_err()
        .to_string()
        .contains("source_projection.coordinate_mismatch")
    );
}
#[test]
fn source_projection_original_artifact_replay_refuses_resealed_numerical_changes() {
    let (evidence, _, _) = source();
    let artifact = project(
        &evidence,
        SourceProjection::AffineDecision {
            contract: contract_to_json(&contract(&evidence)).unwrap(),
        },
    );
    for field in ["value", "coordinate", "snapshot", "source", "unresolved"] {
        let original = artifact.export().unwrap();
        let mut mutant: Mutant = antecedent_io::convert::from_cbor(
            original.strip_prefix(b"ANTE-SOURCE-PROJECTION-1\0").unwrap(),
        )
        .unwrap();
        match field {
            "value" => mutant.report.output["actions"][0]["expected_utility"] = 999.into(),
            "coordinate" => mutant.report.quantities[0].units = "changed-unit".into(),
            "snapshot" => mutant.report.snapshot = "changed-snapshot".into(),
            "source" => mutant.report.source_digest = "changed-source".into(),
            "unresolved" => mutant.report.unresolved.clear(),
            _ => unreachable!(),
        }
        mutant.identity = blake3::hash(
            &antecedent_io::convert::to_cbor(&(
                mutant.version,
                &mutant.source,
                &mutant.projection,
                &mutant.report,
            ))
            .unwrap(),
        )
        .to_hex()
        .to_string();
        let mut resealed = b"ANTE-SOURCE-PROJECTION-1\0".to_vec();
        resealed.extend(antecedent_io::convert::to_cbor(&mutant).unwrap());
        assert!(
            SourceProjectionArtifact::consume(&resealed, &mutant.identity)
                .unwrap_err()
                .to_string()
                .contains("source_projection.replay_mismatch"),
            "{field}"
        );
    }
    let mut bytes = artifact.export().unwrap();
    bytes.pop();
    assert!(SourceProjectionArtifact::consume(&bytes, artifact.identity()).is_err());
    assert!(
        SourceProjectionArtifact::consume(
            &artifact.export().unwrap(),
            "different_retained_identity"
        )
        .is_err()
    );
    let (_, actual, ctx) = source();
    let receipt = SourceResolution::execute(&evidence, &actual, &ctx).unwrap();
    let imported = source_resolved_consumer(vec![receipt])
        .unwrap()
        .consume(
            &bundle(&evidence).to_bytes("replay", &BundleLimits::default()).unwrap(),
            &BundleLimits::default(),
            bundle(&evidence).identity(),
            &SuppliedSources::default(),
        )
        .unwrap();
    imported.require_verified().unwrap();
}

fn external(evidence: &SourceEvidence, means: Vec<f64>) -> Vec<u8> {
    use antecedent_core::{
        CheckedCausalContract, DistributionMeaning, ExternalCapability, ExternalResponse,
        ExternalResult, ExternalResultHeader, ExternalScientificObject, ExternalTrustState,
        ExternalUncertaintyMeaning, IdentificationStatus, LawProviderContract,
        ProviderObjectIdentity, SupportStatus, bind_external_result,
    };
    let quantities = evidence.coordinates().to_vec();
    let object = ExternalScientificObject::Law(LawProviderContract {
        identity: ProviderObjectIdentity {
            provider_id: "lab".into(),
            object_id: "mean-grid".into(),
            version_id: "1".into(),
            snapshot_id: "raw-grid-1".into(),
            request_id: "grid-request-1".into(),
        },
        quantities: quantities.clone(),
        meaning: DistributionMeaning::InterventionalPredictive,
        capabilities: vec![ExternalCapability::Mean],
    });
    let contract = CheckedCausalContract {
        graph_id: "original-graph".into(),
        identification: IdentificationStatus::NonparametricallyIdentified,
        estimand: quantities.clone(),
        accepted_meanings: vec![DistributionMeaning::InterventionalPredictive],
        required_evidence_ids: vec![],
        required_assumption_ids: vec![],
        equivalences: vec![],
    };
    let response = ExternalResponse {
        header: ExternalResultHeader {
            object,
            graph_id: "original-graph".into(),
            quantities,
            evidence_ids: vec![],
            assumption_ids: vec![],
            trust: ExternalTrustState::ExternallyAttested { attestor: "external-lab".into() },
        },
        values: means,
        uncertainty: ExternalUncertaintyMeaning::None,
        point_support: Some(vec![SupportStatus::Supported; 2]),
    };
    let bound = bind_external_result(&contract, &ExternalResult::Response(response)).unwrap();
    antecedent_io::external_claim_artifact::ExternalClaimArtifact::from_bound_claim(
        &bound,
        "original-external-contract",
    )
    .unwrap()
    .to_bytes("external-source-grid")
    .unwrap()
}
#[test]
fn source_projection_external_attestation_and_exact_coordinates_bind_actual_mean_utilities() {
    let (evidence, _, _) = source();
    let original = external(&evidence, vec![3., 5.]);
    let mut builder = BundleBuilder::new();
    for (id, projection) in [
        ("attestation", SourceProjection::Attestation),
        ("coordinates", SourceProjection::QuantityCoordinates),
        (
            "utility",
            SourceProjection::AffineDecision {
                contract: contract_to_json(&contract(&evidence)).unwrap(),
            },
        ),
    ] {
        let artifact = SourceProjectionArtifact::produce(
            ProjectionSource::ExternalClaim(original.clone()),
            projection,
        )
        .unwrap();
        assert_eq!(artifact.report().trust, "externally_attested");
        assert!(artifact.report().unresolved.is_empty());
        builder.add_artifact(Some(id), artifact.node_kind(), &artifact.export().unwrap()).unwrap();
    }
    builder.connect("attestation", "coordinates").unwrap();
    builder.connect("coordinates", "utility").unwrap();
    let bundle = builder.build().unwrap();
    let consumed = standard_consumer()
        .consume(
            &bundle.to_bytes("external-projection-bundle", &BundleLimits::default()).unwrap(),
            &BundleLimits::default(),
            bundle.identity(),
            &SuppliedSources::default(),
        )
        .unwrap();
    consumed.require_verified().unwrap();
    assert!(
        (consumed.value("utility", "A.expected_utility").unwrap()
            - pin()["utilities"]["A"].as_f64().unwrap())
        .abs()
            < 1e-12
    );
    assert!(
        (consumed.value("utility", "B.expected_utility").unwrap()
            - pin()["utilities"]["B"].as_f64().unwrap())
        .abs()
            < 1e-12
    );
    for node in consumed.nodes() {
        assert_eq!(node.facts["trust"], "externally_attested");
        assert_eq!(node.facts["source_authentication_issued"], "false");
    }
    // A same-provider, same-snapshot replacement has a genuinely different full source.
    let changed = external(&evidence, vec![30., 50.]);
    let attestation = SourceProjectionArtifact::produce(
        ProjectionSource::ExternalClaim(original),
        SourceProjection::Attestation,
    )
    .unwrap();
    let utility = SourceProjectionArtifact::produce(
        ProjectionSource::ExternalClaim(changed),
        SourceProjection::AffineDecision {
            contract: contract_to_json(&contract(&evidence)).unwrap(),
        },
    )
    .unwrap();
    let mut builder = BundleBuilder::new();
    builder
        .add_artifact(Some("attestation"), attestation.node_kind(), &attestation.export().unwrap())
        .unwrap();
    builder.add_artifact(Some("utility"), utility.node_kind(), &utility.export().unwrap()).unwrap();
    builder.connect("attestation", "utility").unwrap();
    let substituted = builder.build().unwrap();
    let consumed = standard_consumer()
        .consume(
            &substituted.to_bytes("changed-source", &BundleLimits::default()).unwrap(),
            &BundleLimits::default(),
            substituted.identity(),
            &SuppliedSources::default(),
        )
        .unwrap();
    assert!(consumed.nodes().iter().any(|node|matches!(&node.status,NodeStatus::Failed{stage:BundleStage::SwappedEvidence,reason} if reason=="composition_bundle.swapped_evidence")));
}

#[test]
fn selected_affine_functional_replays_full_original_request_and_refuses_unsupported_action() {
    use antecedent_io::external_claim_artifact::{ExternalClaimArtifact, ExternalClaimIdentity};
    let (evidence, _, _) = source();
    let original = external(&evidence, vec![3., 5.]);
    let expected: ExternalClaimIdentity = serde_json::from_value(
        SourceEvidence::from_external(&original).unwrap().summary()["source_identities"].clone(),
    )
    .unwrap();
    let claim = ExternalClaimArtifact::from_bytes(&original, &expected).unwrap();
    let mut metadata = claim.metadata().clone();
    metadata.identity.point_status[1] = "weak_overlap".into();
    let weak = ExternalClaimArtifact::new(metadata, claim.values().to_vec())
        .unwrap()
        .to_bytes("original-weak-source")
        .unwrap();
    let declaration = contract_to_json(&contract(&evidence)).unwrap();
    let selected = SourceProjectionArtifact::produce(
        ProjectionSource::ExternalClaim(weak.clone()),
        SourceProjection::AffineFunctional { contract: declaration.clone(), action_id: "A".into() },
    )
    .unwrap();
    assert_eq!(selected.report().output["value"], 5.);
    assert!(selected.report().output["standard_error"].is_null());
    assert_eq!(selected.report().output["action_id"], "A");
    assert_eq!(selected.report().trust, "externally_attested");
    let imported =
        SourceProjectionArtifact::consume(&selected.export().unwrap(), selected.identity())
            .unwrap();
    assert_eq!(imported.report(), selected.report());
    assert_eq!(imported.original_bytes(), weak);
    assert_eq!(
        imported.projection(),
        &SourceProjection::AffineFunctional {
            contract: declaration.clone(),
            action_id: "A".into()
        }
    );
    assert!(
        SourceProjectionArtifact::produce(
            ProjectionSource::ExternalClaim(weak.clone()),
            SourceProjection::AffineFunctional {
                contract: declaration.clone(),
                action_id: "x".repeat(257)
            }
        )
        .unwrap_err()
        .to_string()
        .contains("source_projection.limits_exceeded")
    );
    for projection in [
        SourceProjection::AffineDecision { contract: declaration.clone() },
        SourceProjection::AffineFunctional { contract: declaration, action_id: "B".into() },
    ] {
        assert!(
            SourceProjectionArtifact::produce(
                ProjectionSource::ExternalClaim(weak.clone()),
                projection
            )
            .unwrap_err()
            .to_string()
            .contains("source_projection.unsupported_coordinate")
        );
    }
}
