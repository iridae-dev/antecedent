//! Source-only whole-route replay for an embedded joint transport fit.
use super::*;
use crate::analysis::joint_transport_effect_decision::{
    JointTransportEffectDecisionRequest, compile_joint_transport_effect_decision,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{
    ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery, TransportQuery, VariableId,
};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_estimate::joint_bayesian_transport::{
    DataIdentity, GaussianPrior, JointPriors, JointTransportModel, JointTransportOptions,
    PriorProvenance, SourceData, SourceDependence, SourceSharing, TargetData, TransportGraphClass,
    VaryingBlock,
};
use antecedent_graph::SelectionDiagram;

#[test]
#[allow(clippy::too_many_lines, reason = "one linear sequence of steps that reads top to bottom")]
fn joint_transport_fit_node_refits_and_matches_live_chain() {
    let treatment = VariableId::from_raw(0);
    let outcome = VariableId::from_raw(2);
    let mut graph = antecedent_graph::Admg::with_variables(3);
    let source_node = graph.dense_of_static(treatment).unwrap();
    let target_node = graph.dense_of_static(outcome).unwrap();
    graph.insert_directed(source_node, target_node).unwrap();
    let diagram = SelectionDiagram::try_new(graph, [VariableId::from_raw(1)]).unwrap();
    let query = TransportQuery::new(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome,
            treatment: ContinuousDomain::new(treatment, GridSpec::Values(Arc::from([0.0, 1.0]))),
        }),
        "trial",
        "target",
        [treatment],
    );
    let quantity = ScientificQuantity {
        variable_id: "2".into(),
        variable_name: "outcome effect".into(),
        role: QuantityRole::Outcome,
        units: "score".into(),
        population_id: "target".into(),
        regime_id: "do(0=1)-do(0=0)".into(),
        horizon: 0,
        functional_id: "average_effect".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    };
    let decision = DecisionContract {
        actions: vec![
            DecisionAction {
                id: "treat".into(),
                kind: ActionKind::External,
                inputs: vec![quantity.clone()],
                utility: UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Const(1.0)),
            },
            DecisionAction {
                id: "hold".into(),
                kind: ActionKind::External,
                inputs: vec![quantity.clone()],
                utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(0.0)),
            },
        ],
        utility_units: "score".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: Vec::new(),
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
    };
    let rows = 40;
    let source = SourceData {
        id: "trial-a".into(),
        identity: DataIdentity {
            snapshot_digest: "trial-snapshot".into(),
            datum_ids: (0..rows).map(|i| format!("trial-{i}")).collect(),
        },
        treatment: (0..rows).map(|i| i % 2 == 0).collect(),
        outcome: (0..rows).map(|i| 3.0 + if i % 2 == 0 { 2.0 } else { 0.0 }).collect(),
        covariates: Vec::new(),
        noise_variance: 0.1,
    };
    let target = TargetData {
        identity: DataIdentity {
            snapshot_digest: "target-snapshot".into(),
            datum_ids: (0..rows).map(|i| format!("target-{i}")).collect(),
        },
        rows,
        covariates: Vec::new(),
    };
    let model = JointTransportModel {
        graph: TransportGraphClass::FixedDag,
        features: Vec::new(),
        varying: VaryingBlock::Intercept,
        sharing: SourceSharing::IndependentVaryingBlocks,
        dependence: SourceDependence::IndependentSamples,
        priors: JointPriors {
            invariant: GaussianPrior::isotropic(1, 0.0, 10.0, PriorProvenance::Declared),
            varying: GaussianPrior::isotropic(1, 0.0, 10.0, PriorProvenance::Declared),
        },
        max_unsupported_mass: 0.0,
        conflict_z_threshold: 3.0,
    };
    let ctx = ExecutionContext::for_tests(91);
    let plan = compile_joint_transport_effect_decision(JointTransportEffectDecisionRequest {
        diagram,
        query,
        model,
        sources: vec![source],
        target,
        options: JointTransportOptions { draws: 1024, seed: 91 },
        quantity,
        decision,
    })
    .unwrap();
    let result = plan.execute(&ctx).expect("fit target effect");
    assert_eq!(result.calibration, "not_measured");
    let artifact = plan.export(&result, "joint-transport", &ctx).expect("portable source fit");
    let mut corrupted = artifact.clone();
    corrupted[0] ^= 1;
    assert!(
        BundleBuilder::new()
            .add_joint_transport_effect_fit("fit", &corrupted, plan.lowered().identity(), &ctx,)
            .is_err()
    );
    let mut builder = BundleBuilder::new();
    builder
        .add_joint_transport_effect_fit("fit", &artifact, plan.lowered().identity(), &ctx)
        .unwrap();
    builder
        .add_artifact(
            Some("terminal"),
            NodeKind::PropagationChain,
            &result.decision.chain.export().unwrap(),
        )
        .unwrap();
    assert!(
        compile_joint_transport_effect_node_bundle_recomputation(
            builder.build().unwrap(),
            "fit",
            "terminal",
            &ctx,
        )
        .is_err()
    );
    builder.connect("fit", "terminal").unwrap();
    let bundle = builder.build().unwrap();
    let bundle_bytes = export_bundle(&bundle, "joint-transport-bundle").unwrap();
    let consumed = native_fit_bundle_consumer(&ExecutionContext::for_tests(19))
        .consume(
            &bundle_bytes,
            &BundleLimits::default(),
            bundle.identity(),
            &SuppliedSources::default(),
        )
        .unwrap();
    assert!(consumed.node_replayed("fit"));
    assert!(consumed.all_replayed());
    let bundle_plan = compile_joint_transport_effect_node_bundle_recomputation(
        bundle.clone(),
        "fit",
        "terminal",
        &ctx,
    )
    .unwrap();
    assert_eq!(
        bundle_plan.plan_view().route().as_str(),
        "JointTransportEffectNodeBundleRecomputation"
    );
    assert!(matches!(
        bundle_plan.lowered().statistical().realization(),
        Realization::NestedFit { law: NestedFitLaw::Posterior, .. }
    ));
    let receipt = bundle_plan.execute(&ctx).expect("refit graph and evaluate chain");
    assert!(receipt.bundle.dependencies.iter().any(|node| node.node_id == "fit"));
    let portable = bundle_plan.export_portable(&receipt, &ctx).unwrap();
    assert_eq!(
        consume_portable_joint_transport_effect_node_bundle_recomputation(
            &portable,
            bundle.identity(),
            bundle_plan.lowered().identity(),
            &ctx,
        )
        .unwrap(),
        receipt
    );
    assert!(
        consume_portable_joint_transport_effect_node_bundle_recomputation(
            &portable,
            "wrong",
            bundle_plan.lowered().identity(),
            &ctx,
        )
        .is_err()
    );
}
