//! Actual response diagnostics survive scientific coordinate/action mapping and native replay.
use antecedent::analysis::source_evidence::SourceEvidence;
use antecedent::{RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseCoordinateLabels,
    ResponseFunctional, ResponseQuery, VariableId,
};
use antecedent_data::TabularData;
use antecedent_design::decision_artifact::contract_to_json;
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::MeanSource;
use antecedent_design::inverse_query::{
    Comparison, ForwardClaim, ForwardEvidence, GridScope, InverseConstraint, InverseQuery,
    SelectionRule,
};
use antecedent_design::inverse_query_artifact::InverseQueryArtifact;
use antecedent_graph::{Dag, DenseNodeId};

fn source() -> SourceEvidence {
    source_with(0., 51)
}
fn source_with(shift: f64, seed: u64) -> SourceEvidence {
    produce_source(shift, seed).0
}
fn produce_source(
    shift: f64,
    seed: u64,
) -> (SourceEvidence, antecedent::PreparedStudy, ExecutionContext) {
    let treatment: Vec<_> = (0_u32..240).map(|i| (f64::from(i) / 17.).sin()).collect();
    let outcome: Vec<_> = treatment
        .iter()
        .enumerate()
        .map(|(i, t)| {
            1. + shift + 2. * t + 0.03 * (f64::from(u32::try_from(i).unwrap()) / 13.).sin()
        })
        .collect();
    let data =
        TabularData::from_f64_columns([("t", treatment.as_slice()), ("y", outcome.as_slice())])
            .unwrap();
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let ctx = ExecutionContext::for_tests(seed);
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(vec![-0.5, 0.5].into()),
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
    let result = prepared.estimate(&data, &ctx).unwrap();
    let bytes = prepared
        .encode_contracted_result_with_quantity_labels(
            &result,
            "source-evidence-test",
            &ctx,
            Some(&ResponseCoordinateLabels {
                outcome_units: "score",
                population_id: "target",
                transform_id: "identity",
            }),
        )
        .unwrap();
    let consumed = antecedent_io::consume_analysis_result(&bytes).unwrap();
    assert_eq!(
        consumed.acceptance.unresolved.iter().map(AsRef::as_ref).collect::<Vec<_>>(),
        vec!["dependencies.checked_response_grid_operation"]
    );
    (SourceEvidence::from_result(&bytes).unwrap(), prepared, ctx)
}
fn contract(source: &SourceEvidence) -> DecisionContract {
    DecisionContract {
        actions: source
            .coordinates()
            .iter()
            .enumerate()
            .map(|(i, quantity)| DecisionAction {
                id: format!("a{i}"),
                kind: ActionKind::Intervention,
                inputs: vec![quantity.clone()],
                utility: UtilityExpr::Input(0),
            })
            .collect(),
        utility_units: "score".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}
#[test]
fn original_diagnostic_scopes_and_semantic_coordinate_values_survive_consumption() {
    let source = source();
    let summary = source.summary();
    assert!(!summary["diagnostics"].as_array().unwrap().is_empty());
    for (index, quantity) in source.coordinates().iter().enumerate() {
        let diagnostics = source.diagnostics_at(quantity).unwrap();
        for (original, local) in
            summary["diagnostics"].as_array().unwrap().iter().zip(diagnostics.as_array().unwrap())
        {
            match original["scope"].as_str().unwrap_or("global") {
                "per_coordinate" => assert_eq!(local["local_value"], original["values"][index]),
                "global" => {
                    assert!(local["local_value"].is_null());
                    assert_eq!(local["global_values"], original["values"]);
                }
                "inapplicable" => assert!(local["local_value"].is_null()),
                other => panic!("unexpected actual scope {other}"),
            }
        }
    }
    assert_eq!(SourceEvidence::consume(&source.export().unwrap()).unwrap().summary(), summary);
    let mut wrong = source.coordinates()[0].clone();
    wrong.units = "another-unit".into();
    assert_eq!(
        source.diagnostics_at(&wrong).unwrap_err().detail,
        "source_evidence.coordinate_mismatch"
    );
}
#[test]
fn actual_action_contributors_preserve_original_diagnostics_without_synthesized_local_values() {
    let source = source();
    let mut contract = contract(&source);
    contract.actions[0].inputs.push(source.coordinates()[1].clone());
    contract.actions[0].utility =
        UtilityExpr::Sub(Box::new(UtilityExpr::Input(0)), Box::new(UtilityExpr::Input(1)));
    let projected = source.project(&contract_to_json(&contract).unwrap(), &["a0".into()]).unwrap();
    assert_eq!(
        projected.summary()["action_contributors"][0]["source_coordinates"]
            .as_array()
            .unwrap()
            .len(),
        2
    );
    assert_eq!(projected.summary()["diagnostics"], source.summary()["diagnostics"]);
    assert_eq!(
        SourceEvidence::consume(&projected.export().unwrap()).unwrap().summary(),
        projected.summary()
    );
    assert_eq!(
        source
            .project(&contract_to_json(&contract).unwrap(), &["missing".into()])
            .unwrap_err()
            .detail,
        "source_evidence.action_mismatch"
    );
}
#[test]
fn native_inverse_artifact_replays_original_source_diagnostics_and_rejects_changed_mean_binding() {
    use antecedent_io::response_wire::{ResponseIdentificationWire, ResponseValueWire};
    let source = source();
    let contract = contract(&source);
    let original = antecedent_io::consume_analysis_result(source.original_bytes()).unwrap();
    let ResponseIdentificationWire::PointIdentified(ResponseValueWire::Surface { mean, .. }) =
        &original.body.response.as_ref().unwrap().estimate
    else {
        panic!("actual mean curve")
    };
    let snapshot = original.contract.as_ref().unwrap().identities.data_snapshot.iter().fold(
        String::new(),
        |mut hex, byte| {
            use std::fmt::Write;
            write!(hex, "{byte:02x}").unwrap();
            hex
        },
    );
    let means = MeanSource {
        coordinates: source.coordinates().to_vec(),
        means: mean.clone(),
        provider_id: "native-source".into(),
        snapshot_id: snapshot,
        causal_contract_id: "source-contract".into(),
        rng_id: "native:seed:51".into(),
    };
    let query = InverseQuery {
        contract: contract.clone(),
        grid_order: vec!["a0".into(), "a1".into()],
        grid_scope: GridScope::FiniteEnumeration,
        constraints: vec![InverseConstraint::TargetMean {
            target: 0.5,
            comparison: Comparison::AtLeast,
        }],
        selection: SelectionRule::FirstInGridOrder,
        tolerance: 0.,
        max_evaluations: Some(16),
    };
    let mapped = source.project(&contract_to_json(&contract).unwrap(), &query.grid_order).unwrap();
    let artifact = InverseQueryArtifact::new(
        query.clone(),
        ForwardEvidence {
            point: Some(ForwardClaim::Means(means.clone())),
            ..ForwardEvidence::default()
        },
    )
    .unwrap()
    .with_source_evidence(vec![mapped.clone()])
    .unwrap();
    let consumed = InverseQueryArtifact::from_bytes(
        &artifact.to_bytes("sourced-inverse").unwrap(),
        Some(artifact.identity()),
    )
    .unwrap();
    assert_eq!(consumed.source_evidence()[0].summary(), mapped.summary());
    assert_eq!(consumed.result_wire(), artifact.result_wire());
    let mut wrong = means;
    wrong.means[0] += 1.;
    let changed = InverseQueryArtifact::new(
        query,
        ForwardEvidence { point: Some(ForwardClaim::Means(wrong)), ..ForwardEvidence::default() },
    )
    .unwrap();
    let antecedent_io::IoError::Refused { code, message } =
        changed.with_source_evidence(vec![mapped]).unwrap_err()
    else {
        panic!("typed original forward binding refusal");
    };
    assert_eq!(code, "invalid_argument");
    assert_eq!(
        message,
        "source_evidence.forward_binding_mismatch: original source evidence failed validation"
    );
}

#[test]
fn full_original_source_comparison_rejects_changed_science_and_seed() {
    let original = source();
    original.require_same_source(&source()).unwrap();
    for changed in [source_with(0.25, 51), source_with(0., 52)] {
        assert_eq!(
            original.require_same_source(&changed).unwrap_err().detail,
            "source_evidence.source_binding_mismatch"
        );
    }
}

#[test]
fn opaque_resolution_executes_original_preparation_and_rejects_changed_science() {
    use antecedent::analysis::source_evidence::SourceResolution;
    let (evidence, prepared, ctx) = produce_source(0., 51);
    let resolution = SourceResolution::execute(&evidence, &prepared, &ctx).unwrap();
    assert_eq!(
        resolution.source_artifact_digest(),
        evidence.summary()["source_artifact_digest"].as_str().unwrap()
    );
    assert_eq!(
        resolution.resolved_dependencies().iter().map(AsRef::as_ref).collect::<Vec<_>>(),
        vec!["dependencies.checked_response_grid_operation"]
    );
    let (_, changed, _) = produce_source(0.25, 51);
    assert_eq!(
        SourceResolution::execute(&evidence, &changed, &ctx).unwrap_err().detail,
        "source_evidence.source_binding_mismatch"
    );
}

#[test]
fn bounded_evidence_refuses_oversized_logical_sections_before_decompression() {
    let original = source();
    let bytes = original.original_bytes();
    let mut container =
        antecedent_io::EncodedArtifact::read_from(std::io::Cursor::new(bytes)).unwrap();
    container.manifest.sections[0].uncompressed_size = 17 * 1024 * 1024;
    let manifest = antecedent_io::to_cbor(&container.manifest).unwrap();
    let old_len = u32::from_le_bytes(bytes[12..16].try_into().unwrap());
    let mut changed = bytes[..12].to_vec();
    changed.extend(u32::try_from(manifest.len()).unwrap().to_le_bytes());
    changed.extend(manifest);
    changed.extend(&bytes[16 + usize::try_from(old_len).unwrap()..]);
    let error = SourceEvidence::from_result(&changed).unwrap_err();
    assert_eq!((error.code, error.detail), ("invalid_argument", "source_evidence.limits_exceeded"));
}

#[test]
fn original_native_source_chain_is_typed_and_independently_rederived() {
    let source = source();
    let chain = source.provenance_chain().unwrap();
    let loaded = SourceEvidence::consume(&source.export().unwrap()).unwrap();
    assert_eq!(chain, loaded.provenance_chain().unwrap());
    assert_eq!(source.unresolved(), loaded.unresolved());
    assert_eq!(chain.links().last().unwrap().id, "claim");
    assert!(chain.links().iter().any(|link| link.stage == antecedent_core::CompositionStage::Data));
    assert!(
        chain
            .links()
            .iter()
            .any(|link| link.stage == antecedent_core::CompositionStage::CausalContract)
    );
    assert!(
        chain.links().iter().any(|link| link.stage == antecedent_core::CompositionStage::Evidence)
    );
}

#[test]
fn original_external_attestation_retains_exact_support_request_and_source_without_native_authority()
{
    use antecedent_io::external_claim_artifact::{ExternalClaimArtifact, ExternalClaimIdentity};
    let bytes = include_bytes!("../../../conformance/cross_surface/rust_external_claim.bin");
    let expected: ExternalClaimIdentity = serde_json::from_str(include_str!(
        "../../../conformance/cross_surface/rust_external_claim.identity.json"
    ))
    .unwrap();
    let original = ExternalClaimArtifact::from_bytes(bytes, &expected).unwrap();
    let evidence = SourceEvidence::from_external(bytes).unwrap();
    assert!(!evidence.is_native_source());
    assert!(evidence.original_consumption().is_err());
    assert!(evidence.unresolved().is_empty());
    let summary = evidence.summary();
    assert_eq!(summary["source_kind"], "external_bound_claim");
    assert_eq!(summary["source_identities"], serde_json::to_value(&expected).unwrap());
    assert_eq!(summary["point_status"], serde_json::to_value(&expected.point_status).unwrap());
    assert_eq!(
        summary["diagnostic_availability"],
        "not_retained_by_original_external_claim_format"
    );
    assert_eq!(summary["native_authority_issued"], false);
    assert_eq!(summary["calibration_license_issued"], false);
    assert_eq!(evidence.diagnostics_at(&evidence.coordinates()[0]).unwrap(), serde_json::json!([]));
    let means = antecedent_design::composition_verifiers::mean_source_of(&original).unwrap();
    evidence.require_mean_source(&means).unwrap();
    let mut changed = means.clone();
    changed.means[0] += 1.;
    assert_eq!(
        evidence.require_mean_source(&changed).unwrap_err().detail,
        "source_evidence.point_binding_mismatch"
    );
    let declaration = contract(&evidence);
    let projected =
        evidence.project(&contract_to_json(&declaration).unwrap(), &["a1".into()]).unwrap();
    let loaded = SourceEvidence::consume(&projected.export().unwrap()).unwrap();
    assert_eq!(loaded.summary(), projected.summary());
    assert_eq!(loaded.original_bytes(), bytes);
    assert_eq!(
        loaded.summary()["action_contributors"][0]["source_coordinates"][0],
        serde_json::to_value(&expected.quantities[1]).unwrap()
    );
}
