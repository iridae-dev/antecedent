//! Selected pooled results expire with their own study evidence.

use antecedent::StudyBuilder;
use antecedent::analysis::composition::{
    BundleBuilder, BundleStage, CompositionBundle, NodeKind, NodeStatus, SuppliedSources, consume,
    export_bundle,
};
use antecedent::analysis::composition_ledger_decision::{
    LedgerPoolSelection, compile_ledger_pooled_decision_bundle_recomputation,
    consume_portable_ledger_pooled_decision_bundle_recomputation, ledger_pooled_decision_bytes,
    select_ledger_pooled_result,
};
use antecedent::analysis::evidence_ledger::{
    EvidenceLedgerSession, LedgerExport, PoolingDeclaration, PoolingRequest, SourceRevision,
    StudySummary,
};
use antecedent_contract::evidence_ledger::{
    ObservationId, ResultId, SourceDeclaration, SourceId, SourceKind,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::ExecutionContext;
use antecedent_design::decision_artifact::DecisionContractArtifact;
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};

const LEDGER_BYTES: &[u8] =
    include_bytes!("../../../spec/conformance/evidence_ledger__pooled.artifact");
const LEDGER_ID: &str = "b36d763a3eb209258f2c4fe925b31cbbf0dc65f028aa0a58b75cf2173e19c3c2";
const POINT: &str = "8f93fa7182ab27a50a9b0988634e7289d3c0a778ab67288ef681a207929060ca";
const RANGE: &str = "b3ea42845c01939f6437498e593b572288ffd6177e6832566a52b3478ea6dde5";

fn quantity() -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "pooled_effect".into(),
        variable_name: "pooled_effect".into(),
        role: QuantityRole::Outcome,
        units: "unit".into(),
        population_id: "target".into(),
        regime_id: "contrast:treated-control".into(),
        horizon: 0,
        functional_id: "mean_difference".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn contract() -> DecisionContract {
    let action = |id: &str, utility| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![quantity()],
        utility,
    };
    DecisionContract {
        actions: vec![
            action(
                "treat",
                UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Const(1.25)),
            ),
            action("wait", UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(0.0))),
        ],
        utility_units: "utility".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn graph(
    ledger: &[u8],
    first: &LedgerPoolSelection,
    other: Option<&LedgerPoolSelection>,
) -> CompositionBundle {
    let mut builder = BundleBuilder::new();
    builder.add_artifact(Some("ledger"), NodeKind::EvidenceLedger, ledger).unwrap();
    let contract = contract();
    let artifact = DecisionContractArtifact::new(contract.clone()).unwrap();
    builder
        .add_artifact(
            Some("contract"),
            NodeKind::DecisionContract,
            &artifact.to_bytes("contract").unwrap(),
        )
        .unwrap();
    for (selection, selected, terminal) in
        [(Some(first), "selection", "decision"), (other, "other-selection", "other-decision")]
    {
        let Some(selection) = selection else { continue };
        builder
            .add_artifact(
                Some(selected),
                NodeKind::LedgerPooledResult,
                &selection.to_bytes().unwrap(),
            )
            .unwrap();
        let bytes = ledger_pooled_decision_bytes(&contract, selection, "decision").unwrap();
        builder.add_artifact(Some(terminal), NodeKind::DecisionResult, &bytes).unwrap();
        builder.connect("ledger", selected).unwrap();
        builder.connect(selected, terminal).unwrap();
        builder.connect("contract", terminal).unwrap();
    }
    builder.build().unwrap()
}

fn summary(source: SourceId, estimate: f64, variance: f64) -> StudySummary {
    StudySummary { source, estimate, variance }
}

fn observation(label: &str, rows: &[u8]) -> SourceDeclaration {
    SourceDeclaration::new(label, SourceKind::ObservationSet)
        .with_observations(rows.iter().map(|row| ObservationId::from_key("pooled", &[*row])))
}

fn pooled(
    ledger: &mut EvidenceLedgerSession,
    label: &str,
    studies: [StudySummary; 2],
    replaces: Option<ResultId>,
    ctx: &ExecutionContext,
) -> ResultId {
    ledger
        .pool_summaries(
            PoolingRequest { label: label.into(), summaries: studies.to_vec(), replaces },
            PoolingDeclaration::Independent,
            ctx,
        )
        .unwrap()
        .result
}

fn select(ledger: &LedgerExport, result: ResultId, ctx: &ExecutionContext) -> LedgerPoolSelection {
    select_ledger_pooled_result(&ledger.bytes, &ledger.identity, result, quantity(), ctx).unwrap()
}

#[test]
fn inherited_ledger_replays_a_selected_point_and_refuses_an_assumption_range() {
    let ctx = ExecutionContext::for_tests(12);
    let result = ResultId::from_hex(POINT).unwrap();
    let selection =
        select_ledger_pooled_result(LEDGER_BYTES, LEDGER_ID, result, quantity(), &ctx).unwrap();
    // Inverse-variance weights are 4/7, 2/7 and 1/7 for estimates 0, 2 and 6.
    assert!((selection.estimate() - 10.0 / 7.0).abs() < 1e-12);
    let bundle = graph(LEDGER_BYTES, &selection, None);
    let raw = consume(
        &export_bundle(&bundle, "ledger-point").unwrap(),
        bundle.identity(),
        &SuppliedSources::default(),
    )
    .unwrap();
    assert!(raw.all_replayed());
    assert!((raw.value("decision", "treat.expected_utility").unwrap() - 5.0 / 28.0).abs() < 1e-12);
    let plan = compile_ledger_pooled_decision_bundle_recomputation(bundle, "decision").unwrap();
    assert_eq!(plan.plan_view().route().as_str(), "LedgerPooledDecisionBundleRecomputation");
    let receipt = plan.execute(&ctx).unwrap();
    let portable = plan.export_portable(&receipt, &ctx).unwrap();
    assert_eq!(
        consume_portable_ledger_pooled_decision_bundle_recomputation(
            &portable,
            &receipt.bundle_identity,
            plan.lowered().identity(),
            &ExecutionContext::for_tests(13),
        )
        .unwrap(),
        receipt
    );
    let range = select_ledger_pooled_result(
        LEDGER_BYTES,
        LEDGER_ID,
        ResultId::from_hex(RANGE).unwrap(),
        quantity(),
        &ctx,
    )
    .unwrap_err();
    assert!(matches!(
        range,
        antecedent::analysis::composition_decision_terminal::DecisionBundleRecomputeError::Bundle(
            antecedent::analysis::composition::BundleError::Refused {
                stage: BundleStage::UnsupportedLaw,
                ..
            }
        )
    ));

    let mut reference = BundleBuilder::new();
    reference.add_producer_reference("ledger", "ledger", LEDGER_BYTES).unwrap();
    reference
        .add_artifact(
            Some("selection"),
            NodeKind::LedgerPooledResult,
            &selection.to_bytes().unwrap(),
        )
        .unwrap();
    reference.connect("ledger", "selection").unwrap();
    let graph = reference.build().unwrap();
    let wire = export_bundle(&graph, "ledger-provider-reference").unwrap();
    let missing = consume(&wire, graph.identity(), &SuppliedSources::default()).unwrap();
    assert!(!missing.node_replayed("selection"));
    let supplied = SuppliedSources::default().with_producer("ledger", LEDGER_BYTES.to_vec());
    let resolved = consume(&wire, graph.identity(), &supplied).unwrap();
    resolved.require_verified().unwrap();
    assert!(resolved.node_replayed("selection"));
    assert!(!resolved.node_replayed("ledger"));
}

#[test]
fn changed_studies_expire_their_decision_without_invalidating_the_other_pool() {
    let ctx = ExecutionContext::for_tests(14);
    let mut session = StudyBuilder::evidence_ledger();
    let a = session.add_source(observation("a", &[0])).unwrap();
    let b = session.add_source(observation("b", &[1])).unwrap();
    let c = session.add_source(observation("c", &[2])).unwrap();
    let d = session.add_source(observation("d", &[3])).unwrap();
    let main =
        pooled(&mut session, "main", [summary(a, 1.0, 1.0), summary(b, 3.0, 4.0)], None, &ctx);
    let other =
        pooled(&mut session, "other", [summary(c, 2.0, 1.0), summary(d, 4.0, 1.0)], None, &ctx);
    let original = session.export("original").unwrap();
    let first = select(&original, main, &ctx);
    let independent = select(&original, other, &ctx);
    assert!((first.estimate() - 1.4).abs() < 1e-12);
    let before = graph(&original.bytes, &first, Some(&independent));

    let active = session
        .revise_source(SourceRevision::Update {
            target: a,
            replacement: observation("a-revised", &[0, 4]),
        })
        .unwrap()
        .created_source
        .unwrap();
    let update = pooled(
        &mut session,
        "updated",
        [summary(active, 0.0, 1.0), summary(b, 3.0, 4.0)],
        Some(main),
        &ctx,
    );
    let revised = session.export("revised").unwrap();
    let invalid = graph(&revised.bytes, &first, Some(&independent));
    let consumed = consume(
        &export_bundle(&invalid, "retired").unwrap(),
        invalid.identity(),
        &SuppliedSources::default(),
    )
    .unwrap();
    assert!(!consumed.node_replayed("selection"));
    assert!(!consumed.node_replayed("decision"));
    assert!(consumed.node_replayed("other-decision"));
    let independent_plan =
        compile_ledger_pooled_decision_bundle_recomputation(invalid, "other-decision").unwrap();
    independent_plan.execute(&ctx).unwrap();
    assert_eq!(select(&revised, other, &ctx).identity(), independent.identity());
    let replacement = select(&revised, update, &ctx);
    assert!((replacement.estimate() - 0.6).abs() < 1e-12);
    let refreshed = graph(&revised.bytes, &replacement, Some(&independent));
    let result_identity = |bundle: &CompositionBundle| {
        bundle.nodes().iter().find(|node| node.id == "other-decision").unwrap().identity.clone()
    };
    assert_eq!(result_identity(&before), result_identity(&refreshed));
    let live = compile_ledger_pooled_decision_bundle_recomputation(refreshed, "decision")
        .unwrap()
        .execute(&ctx)
        .unwrap();
    assert_ne!(live.bundle_identity, before.identity());

    session
        .revise_source(SourceRevision::Withdraw { target: active, reason: "withdrawn".into() })
        .unwrap();
    let withdrawn = session.export("withdrawn").unwrap();
    let graph = graph(&withdrawn.bytes, &replacement, Some(&independent));
    let replay = consume(
        &export_bundle(&graph, "withdrawn").unwrap(),
        graph.identity(),
        &SuppliedSources::default(),
    )
    .unwrap();
    assert!(!replay.node_replayed("decision"));
    assert!(replay.node_replayed("other-decision"));
}

#[test]
fn ledger_binding_refuses_swapped_study_summaries_even_when_the_mean_is_identical() {
    let ctx = ExecutionContext::for_tests(15);
    let produce = |left, right| {
        let mut ledger = StudyBuilder::evidence_ledger();
        let a = ledger.add_source(observation("a", &[0])).unwrap();
        let b = ledger.add_source(observation("b", &[1])).unwrap();
        let result = pooled(
            &mut ledger,
            "same pool operation",
            [summary(a, left, 1.0), summary(b, right, 1.0)],
            None,
            &ctx,
        );
        let artifact = ledger.export("same artifact label").unwrap();
        let selection = select(&artifact, result, &ctx);
        (artifact, selection)
    };
    let (original, retained) = produce(1.0, 3.0);
    let (updated, changed) = produce(3.0, 1.0);
    assert_eq!(retained.result_id(), changed.result_id());
    assert_eq!(retained.operation_id(), changed.operation_id());
    assert_eq!(retained.estimate().to_bits(), changed.estimate().to_bits());
    assert_ne!(retained.identity(), changed.identity());
    assert_ne!(original.identity, updated.identity);
    for (ledger, selected) in [(&original, &retained), (&updated, &changed)] {
        let matched = graph(&ledger.bytes, selected, None);
        let bytes = export_bundle(&matched, "matched").unwrap();
        let replay = consume(&bytes, matched.identity(), &SuppliedSources::default()).unwrap();
        assert!(replay.all_replayed());
        assert_eq!(replay.value("selection", "pooled_estimate"), Some(2.0));
    }
    let selected_from_wrong_evidence = graph(&updated.bytes, &retained, None);
    let bytes = export_bundle(&selected_from_wrong_evidence, "crossed").unwrap();
    let replay =
        consume(&bytes, selected_from_wrong_evidence.identity(), &SuppliedSources::default())
            .unwrap();
    assert!(replay.node_replayed("ledger"));
    assert!(matches!(
        replay.node("selection").unwrap().status,
        NodeStatus::Failed { stage: BundleStage::SwappedEvidence, .. }
    ));
    assert!(!replay.node_replayed("decision"));
    let plan = compile_ledger_pooled_decision_bundle_recomputation(
        selected_from_wrong_evidence,
        "decision",
    )
    .unwrap();
    assert!(plan.execute(&ctx).is_err());
}
