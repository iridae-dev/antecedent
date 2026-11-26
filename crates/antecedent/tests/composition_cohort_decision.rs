//! Cohort bundle replay against the inherited three-row compact-model fixture.

use antecedent::analysis::cohort_decision::{
    CohortBinding, CohortDecisionSource, CohortExpectation, CohortOptions, evaluate_cohort_decision,
};
use antecedent::analysis::composition::{
    BundleBuilder, BundleLimits, BundleNode, BundleStage, CompositionBundle, NodeKind, NodeStatus,
    SuppliedSources, consume, export_bundle,
};
use antecedent::analysis::composition_cohort_decision::{
    cohort_decision_bytes, compile_cohort_decision_bundle_recomputation,
    consume_portable_cohort_decision_bundle_recomputation,
};
use antecedent::analysis::composition_decision_terminal::DecisionBundleRecomputeError;
use antecedent_core::execution::AdmissionPool;
use antecedent_core::{ExecutionContext, SupportStatus};
use antecedent_design::cohort_decision_input::cohort_mean_binding;
use antecedent_design::composition_boundary::{SupportPolicy, SupportedVerdict};
use antecedent_design::decision_artifact::{DecisionContractArtifact, bound_mean_result_to_bytes};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::Verdict;

const SOURCE_BYTES: &[u8] =
    include_bytes!("../../antecedent-io-core/tests/fixtures/cohort_decision_source_v1.bin");
const SOURCE_IDENTITY: &str = "6fd024b0c15c6416187582f952b3c6c0cdb1ed8ebca01b8851209a50eb82140d";

fn fixture() -> CohortDecisionSource {
    CohortDecisionSource::consume(SOURCE_BYTES, &CohortExpectation::for_identity(SOURCE_IDENTITY))
        .unwrap()
}

fn contract(source: &CohortDecisionSource) -> DecisionContract {
    let action = |id: &str, utility| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs: vec![source.quantity().clone()],
        utility,
    };
    DecisionContract {
        actions: vec![
            action(
                "treat",
                UtilityExpr::difference(
                    UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(4.0)),
                    UtilityExpr::Const(8.0),
                ),
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

fn bundle(source: &CohortDecisionSource, source_bytes: &[u8]) -> CompositionBundle {
    let mut builder = BundleBuilder::new();
    let contract = contract(source);
    let terminal = cohort_decision_bytes(&contract, source, "decision").unwrap();
    let declaration = DecisionContractArtifact::new(contract).unwrap();
    builder.add_cohort_decision_source(Some("cohort"), source_bytes).unwrap();
    builder
        .add_artifact(
            Some("contract"),
            NodeKind::DecisionContract,
            &declaration.to_bytes("contract").unwrap(),
        )
        .unwrap();
    builder.add_artifact(Some("decision"), NodeKind::DecisionResult, &terminal).unwrap();
    builder.connect("cohort", "decision").unwrap();
    builder.connect("contract", "decision").unwrap();
    builder.build().unwrap()
}

#[test]
fn compact_prediction_replays_into_a_supported_cohort_and_names_missing_refit_inputs() {
    use antecedent::analysis::composition::require_compact_model_refit_inputs;
    use antecedent_contract::{QuantityRole, ScientificQuantity};
    use antecedent_io_core::cohort_decision::RetainedEffectModel;
    use antecedent_io_core::compact_export::{
        CompactExport, ExportInput, ExportQuery, ExportSpec, InputSupport, TermSpec,
    };
    let observed = fixture();
    let ctx = ExecutionContext::for_tests(91);
    let quantity =
        |id: &str, units: &str, role, regime: &str, functional: &str| ScientificQuantity {
            variable_id: id.into(),
            variable_name: id.into(),
            role,
            units: units.into(),
            population_id: observed.cohort().population_id().into(),
            regime_id: regime.into(),
            horizon: 0,
            functional_id: functional.into(),
            conditioning: Vec::new(),
            transform_id: "identity".into(),
        };
    let spec = ExportSpec {
        response: (&quantity(
            "y",
            "mmHg",
            QuantityRole::Outcome,
            "contrast:do(t=1)-do(t=0)",
            "conditional_mean_difference",
        ))
        .into(),
        inputs: vec![ExportInput {
            quantity: (&quantity("z", "1", QuantityRole::Covariate, "observational", "feature"))
                .into(),
            support: InputSupport::Range { lo: -1.0, hi: 1.0 },
        }],
        terms: vec![TermSpec::Intercept, TermSpec::Linear { quantity: "z".into() }],
        coefficients: vec![2.0, 0.5],
        covariance: vec![0.04, -0.01, -0.01, 0.09],
        mask: Vec::new(),
    };
    let export = CompactExport::build(spec).unwrap();
    assert!(
        (export.evaluate(&ExportQuery::new().with_number("z", 0.5)).unwrap().point - 2.25).abs()
            < 1e-12
    );
    assert!(export.evaluate(&ExportQuery::new().with_number("z", 1.5)).is_err());
    let model_bytes = export.to_bytes("model").unwrap();
    assert_eq!(
        require_compact_model_refit_inputs(&model_bytes).unwrap_err().stage,
        BundleStage::CallbackUnavailable,
    );
    let model = RetainedEffectModel::from_compact_export(&export).unwrap();
    let cohort = CohortDecisionSource::build(
        &model,
        observed.cohort(),
        "mmHg",
        &CohortOptions::default(),
        &ctx,
    )
    .unwrap();
    let terminal_contract = contract(&cohort);
    let terminal = cohort_decision_bytes(&terminal_contract, &cohort, "cohort-decision").unwrap();
    let declaration =
        DecisionContractArtifact::new(terminal_contract).unwrap().to_bytes("contract").unwrap();
    let mut builder = BundleBuilder::new();
    builder.add_detected_artifact(Some("model"), &model_bytes).unwrap();
    builder.add_detected_artifact(Some("cohort"), &cohort.to_bytes("cohort").unwrap()).unwrap();
    builder.add_detected_artifact(Some("decision"), &terminal).unwrap();
    builder.add_detected_artifact(Some("contract"), &declaration).unwrap();
    builder.connect("model", "cohort").unwrap();
    builder.connect("cohort", "decision").unwrap();
    builder.connect("contract", "decision").unwrap();
    let graph = builder.build().unwrap();
    let bytes = export_bundle(&graph, "compact-cohort").unwrap();
    let replayed = consume(&bytes, graph.identity(), &SuppliedSources::default()).unwrap();
    let mean = replayed.value("cohort", "mean_contrast").unwrap();
    assert!((mean - 2.225).abs() < 1e-12);
    assert!((replayed.value("decision", "treat.expected_utility").unwrap() - 0.9).abs() < 1e-12);
    let receipt = compile_cohort_decision_bundle_recomputation(graph, "decision")
        .unwrap()
        .execute(&ctx)
        .unwrap();
    assert!(receipt.dependencies.iter().any(|dependency| dependency.node_id == "model"));
}

fn revised(source: &CohortDecisionSource, ctx: &ExecutionContext) -> CohortDecisionSource {
    let old = source.cohort();
    let cohort = CohortBinding::new(
        old.snapshot_id(),
        old.population_id(),
        old.row_ids().to_vec(),
        old.features().to_vec(),
        old.columns().to_vec(),
        vec![0.8, 0.1, 0.1],
    )
    .unwrap();
    CohortDecisionSource::build(
        source.model(),
        &cohort,
        source.outcome_units(),
        &CohortOptions::default(),
        ctx,
    )
    .unwrap()
}

#[test]
fn inherited_cohort_bundle_replays_affine_decision_and_reweights() {
    let original = fixture();
    let ctx = ExecutionContext::for_tests(3);
    let updated = revised(&original, &ctx);
    let bytes = updated.to_bytes("reweighted-cohort").unwrap();
    let mut receipts = Vec::new();
    // tau(z) = 2 + z/2, evaluated at z = (-1, 1/2, 1).
    for (source, retained, mean, utility, selected) in [
        (&original, SOURCE_BYTES, 2.225, 0.9, "treat"),
        (&updated, bytes.as_slice(), 1.675, -1.3, "wait"),
    ] {
        assert_eq!(source.support_status(), SupportStatus::Supported);
        let direct =
            evaluate_cohort_decision(&contract(source), source, &SupportPolicy::require_all())
                .unwrap();
        assert_eq!(
            direct.verdict,
            SupportedVerdict::Compared(Verdict::UniquelyOptimal(selected.into()))
        );
        assert!(
            direct.evpi.is_none() && direct.outcomes.iter().all(|a| a.standard_error.is_none())
        );
        assert!((direct.outcomes[0].expected_utility - utility).abs() < 1e-12);
        let graph = bundle(source, retained);
        let graph_bytes = export_bundle(&graph, "cohort-decision").unwrap();
        let consumed =
            consume(&graph_bytes, graph.identity(), &SuppliedSources::default()).unwrap();
        assert!(consumed.all_replayed());
        assert!((consumed.value("cohort", "mean_contrast").unwrap() - mean).abs() < 1e-12);
        assert!(
            (consumed.value("decision", "treat.expected_utility").unwrap() - utility).abs() < 1e-12
        );
        let plan = compile_cohort_decision_bundle_recomputation(graph, "decision").unwrap();
        assert_eq!(plan.plan_view().route().as_str(), "CohortDecisionBundleRecomputation");
        let run = plan.execute(&ctx).unwrap();
        let artifact = plan.export_portable(&run, &ctx).unwrap();
        let verified = consume_portable_cohort_decision_bundle_recomputation(
            &artifact,
            &run.bundle_identity,
            plan.lowered().identity(),
            &ExecutionContext::for_tests(91),
        )
        .unwrap();
        assert_eq!(verified, run);
        receipts.push(run);
    }
    assert_ne!(receipts[0].lowered_program_identity, receipts[1].lowered_program_identity);
    assert_ne!(receipts[0].decision_fingerprint, receipts[1].decision_fingerprint);
}

#[test]
fn cohort_bundle_refuses_swapped_source_or_law_demand() {
    let original = fixture();
    let ctx = ExecutionContext::for_tests(3);
    let graph = bundle(&original, SOURCE_BYTES);
    let replacement = revised(&original, &ctx);
    let replacement_bytes = replacement.to_bytes("changed").unwrap();
    let edges = [("cohort".into(), "decision".into()), ("contract".into(), "decision".into())];
    let mut nodes = graph.nodes().to_vec();
    let cohort = nodes.iter_mut().find(|node| node.id == "cohort").unwrap();
    *cohort = BundleNode::embedded(
        "cohort",
        NodeKind::CohortDecisionSource,
        SOURCE_IDENTITY,
        replacement_bytes,
    );
    // Reseal the graph while retaining the original source-node identity.
    let changed = CompositionBundle::new(nodes, &edges, &BundleLimits::default()).unwrap();
    let bytes = export_bundle(&changed, "changed").unwrap();
    assert_eq!(
        consume(&bytes, graph.identity(), &SuppliedSources::default()).unwrap_err().stage(),
        Some(BundleStage::ExpectedIdentityMismatch),
    );
    let consumed = consume(&bytes, changed.identity(), &SuppliedSources::default()).unwrap();
    assert!(matches!(
        consumed.node("cohort").unwrap().status,
        NodeStatus::Failed { stage: BundleStage::SwappedEvidence, .. }
    ));
    assert!(
        compile_cohort_decision_bundle_recomputation(changed, "decision")
            .unwrap()
            .execute(&ctx)
            .is_err()
    );
    let no_edges =
        CompositionBundle::new(graph.nodes().to_vec(), &[], &BundleLimits::default()).unwrap();
    assert!(compile_cohort_decision_bundle_recomputation(no_edges, "decision").is_err());
    let mut nonlinear = contract(&original);
    nonlinear.actions[0].utility =
        UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(0));
    assert_eq!(
        failure_stage(cohort_decision_bytes(&nonlinear, &original, "nonlinear").unwrap_err()),
        Some(BundleStage::UnsupportedLaw),
    );
}

fn failure_stage(error: DecisionBundleRecomputeError) -> Option<BundleStage> {
    match error {
        DecisionBundleRecomputeError::Bundle(error) => error.stage(),
        _ => None,
    }
}

#[test]
fn cohort_bundle_recomputes_an_edited_terminal_and_obeys_execution_limits() {
    let source = fixture();
    let graph = bundle(&source, SOURCE_BYTES);
    let plan = compile_cohort_decision_bundle_recomputation(graph.clone(), "decision").unwrap();
    let limited = ExecutionContext::for_tests(3).with_admission(AdmissionPool::fixed(1, 1));
    assert_eq!(
        failure_stage(plan.execute(&limited).unwrap_err()),
        Some(BundleStage::ResourceRefused)
    );
    let cancelled = ExecutionContext::for_tests(3);
    cancelled.cancellation.cancel();
    assert_eq!(
        failure_stage(plan.execute(&cancelled).unwrap_err()),
        Some(BundleStage::ExecutionCancelled)
    );

    let mut direct =
        evaluate_cohort_decision(&contract(&source), &source, &SupportPolicy::require_all())
            .unwrap();
    let mut edited = direct.results.pop().unwrap();
    edited.actions[0].expected_utility += 1.0;
    let result =
        bound_mean_result_to_bytes(&edited, &cohort_mean_binding(&source), "edited").unwrap();
    let mut builder = BundleBuilder::new();
    let source_node = graph.nodes().iter().find(|node| node.id == "cohort").unwrap();
    let contract_node = graph.nodes().iter().find(|node| node.id == "contract").unwrap();
    for node in [source_node, contract_node] {
        let antecedent::analysis::composition::NodeSource::Embedded { bytes } = &node.source else {
            panic!("test graph embeds each parent");
        };
        builder.add_artifact(Some(&node.id), node.kind, bytes).unwrap();
    }
    builder.add_artifact(Some("decision"), NodeKind::DecisionResult, &result).unwrap();
    builder.connect("cohort", "decision").unwrap();
    builder.connect("contract", "decision").unwrap();
    let changed =
        compile_cohort_decision_bundle_recomputation(builder.build().unwrap(), "decision").unwrap();
    assert_eq!(
        failure_stage(changed.execute(&ExecutionContext::for_tests(3)).unwrap_err()),
        Some(BundleStage::TamperedQuantity),
    );
}

#[test]
fn compact_refit_checks_original_training_and_replays_the_named_bundle_model() {
    use antecedent::analysis::cohort_decision::{NativeCohortFit, fit_cohort_effect};
    use antecedent::analysis::composition::{refit_compact_bundle_model, refit_compact_model};
    use antecedent_contract::QuantityRole;
    use antecedent_estimate::cohort_linear_effect::{FIT_PROCEDURE_ID, FIT_TRAINING_VARIABLE};
    use antecedent_io_core::compact_export::{
        CompactExport, ExportQuery, ExportSpec, InputSupport, MaskCondition, MaskRegion,
    };

    let mut response = fixture().quantity().clone();
    response.functional_id = "conditional_mean_difference".into();
    let mut modifier = response.clone();
    modifier.variable_id = "z".into();
    modifier.variable_name = "z".into();
    modifier.role = QuantityRole::Covariate;
    modifier.units = "1".into();
    modifier.regime_id = "observed".into();
    modifier.functional_id = "mean".into();
    let feature = [-1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0, -1.0, 0.0, 1.0];
    let outcome = [0.0, 1.0, 2.0, 2.0, 3.0, 4.0, 2.0, 3.5, 5.0, 4.0, 5.5, 7.0];
    let treatment = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let fit =
        fit_cohort_effect(&outcome, &treatment, &[&feature], &response.into(), &[modifier.into()])
            .unwrap();
    let model = fit.export();
    let bytes = model.to_bytes("retained-native-model").unwrap();
    let ctx = ExecutionContext::for_tests(9);
    let checked =
        refit_compact_model(&bytes, model.identity(), fit.rows().clone(), FIT_PROCEDURE_ID, &ctx)
            .unwrap();
    assert_eq!(checked.export().body(), model.body());
    let query = ExportQuery::new().with_number("z", 0.5);
    assert!((checked.export().evaluate(&query).unwrap().point - 2.75).abs() < 1e-12);

    let mut bundle = BundleBuilder::new();
    bundle.add_detected_artifact(Some("native-model"), &bytes).unwrap();
    let bundle = bundle.build().unwrap();
    let bundle_bytes = export_bundle(&bundle, "native-model-bundle").unwrap();
    let from_bundle = refit_compact_bundle_model(
        &bundle_bytes,
        bundle.identity(),
        "native-model",
        fit.rows().clone(),
        FIT_PROCEDURE_ID,
        &ctx,
    )
    .unwrap();
    assert_eq!(from_bundle.export().body(), model.body());

    let mut revised = fit.rows().clone();
    revised.outcome.swap(0, 3); // same features and treatment, but not the original row snapshot
    assert!(matches!(
        refit_compact_model(&bytes, model.identity(), revised, FIT_PROCEDURE_ID, &ctx),
        Err(antecedent::analysis::composition::BundleError::Refused {
            stage: BundleStage::GraphOrSnapshotMismatch,
            ..
        })
    ));
    assert!(matches!(
        refit_compact_model(&bytes, model.identity(), fit.rows().clone(), "another_learner", &ctx),
        Err(antecedent::analysis::composition::BundleError::Refused {
            stage: BundleStage::ProviderRequestChanged,
            ..
        })
    ));

    let mut legacy_response = model.body().response.clone();
    legacy_response.conditioning.retain(|condition| condition.variable_id != FIT_TRAINING_VARIABLE);
    let as_spec = |response, mask| ExportSpec {
        response,
        inputs: model.body().inputs.clone(),
        terms: model.body().terms.clone(),
        coefficients: model.body().coefficients.clone(),
        covariance: model.body().covariance.clone(),
        mask,
    };
    let legacy = CompactExport::build(as_spec(legacy_response, vec![])).unwrap();
    // Historical source-bearing cohort readers still reproduce their original compact identities.
    assert_eq!(
        NativeCohortFit::replay(fit.rows().clone(), &legacy).unwrap().export().identity(),
        legacy.identity(),
    );
    assert!(matches!(
        refit_compact_model(
            &legacy.to_bytes("legacy-model").unwrap(),
            legacy.identity(),
            fit.rows().clone(),
            FIT_PROCEDURE_ID,
            &ctx
        ),
        Err(antecedent::analysis::composition::BundleError::Refused {
            stage: BundleStage::CallbackUnavailable,
            ..
        })
    ));
    let with_mask = CompactExport::build(as_spec(
        model.body().response.clone(),
        vec![MaskRegion {
            id: "restricted".into(),
            conditions: vec![MaskCondition {
                quantity: "z".into(),
                within: InputSupport::Range { lo: -0.1, hi: 0.1 },
            }],
        }],
    ))
    .unwrap();
    assert!(matches!(
        refit_compact_model(
            &with_mask.to_bytes("changed-mask").unwrap(),
            with_mask.identity(),
            fit.rows().clone(),
            FIT_PROCEDURE_ID,
            &ctx
        ),
        Err(antecedent::analysis::composition::BundleError::Refused {
            stage: BundleStage::TamperedQuantity,
            ..
        })
    ));
}
