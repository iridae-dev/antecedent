//! Source-only tests for raw-evidence predictive fit and decision bundles.
use super::*;
use crate::analysis::joint_transport_predictive_decision::{
    JointTransportPredictiveDecisionRequest, compile_joint_transport_predictive_decision,
};
use crate::analysis::propagation_chain_decision::{
    ChainDecisionResult, compile_action_world_chain_decision,
};
use crate::analysis::trial_ipw_predictive_decision::{
    TrialIpwPredictiveDecisionRequest, TrialIpwRows, compile_trial_ipw_predictive_decision,
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

fn fixture() -> (SelectionDiagram, TransportQuery, [ScientificQuantity; 2], DecisionContract) {
    let treatment = VariableId::from_raw(0);
    let outcome = VariableId::from_raw(2);
    let mut graph = antecedent_graph::Admg::with_variables(3);
    let a = graph.dense_of_static(treatment).unwrap();
    let y = graph.dense_of_static(outcome).unwrap();
    graph.insert_directed(a, y).unwrap();
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
    let world = |i| ScientificQuantity {
        variable_id: "2".into(),
        variable_name: "outcome".into(),
        role: QuantityRole::Outcome,
        units: "score".into(),
        population_id: "target".into(),
        regime_id: format!("do(0={i})"),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    };
    let worlds = [world(0), world(1)];
    let decision = DecisionContract {
        actions: vec![
            DecisionAction {
                id: "hold".into(),
                kind: ActionKind::Intervention,
                inputs: vec![worlds[0].clone()],
                utility: UtilityExpr::Input(0),
            },
            DecisionAction {
                id: "treat".into(),
                kind: ActionKind::Intervention,
                inputs: vec![worlds[1].clone()],
                utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(0)),
            },
        ],
        utility_units: "score".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: Vec::new(),
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
    };
    (diagram, query, worlds, decision)
}
fn check(
    kind: PredictiveFitKind,
    producer: &[u8],
    producer_program: SemanticDigest,
    result: &ChainDecisionResult,
    decision: &DecisionContract,
    ctx: &ExecutionContext,
) {
    let terminal =
        compile_action_world_chain_decision(result.chain.record().clone(), decision.clone(), ctx)
            .unwrap();
    let terminal_bytes = terminal.export(result, "decision").unwrap();
    let mut builder = BundleBuilder::new();
    match kind {
        PredictiveFitKind::Joint => {
            builder.add_joint_transport_predictive_fit("fit", producer, producer_program, ctx)
        }
        PredictiveFitKind::Trial => {
            builder.add_trial_ipw_predictive_fit("fit", producer, producer_program, ctx)
        }
        PredictiveFitKind::Cyclic => {
            builder.add_cyclic_target_feedback_fit("fit", producer, producer_program, ctx)
        }
        PredictiveFitKind::Graph => {
            builder.add_graph_imputation_predictive_fit("fit", producer, producer_program, ctx)
        }
        PredictiveFitKind::Bridge => {
            builder.add_assessed_bridge_posterior_fit("fit", producer, producer_program, ctx)
        }
    }
    .unwrap();
    builder
        .add_artifact(Some("chain"), NodeKind::PropagationChain, &result.chain.export().unwrap())
        .unwrap();
    builder.add_predictive_chain_decision("decision", &terminal_bytes, ctx).unwrap();
    assert!(
        compile_predictive_fit_decision_bundle(
            builder.build().unwrap(),
            "fit",
            "chain",
            "decision",
            ctx,
            kind
        )
        .is_err()
    );
    builder.connect("fit", "chain").unwrap();
    builder.connect("chain", "decision").unwrap();
    let bundle = builder.build().unwrap();
    let plan = compile_predictive_fit_decision_bundle(
        bundle.clone(),
        "fit",
        "chain",
        "decision",
        ctx,
        kind,
    )
    .unwrap();
    let expected_route = match kind {
        PredictiveFitKind::Joint => "JointTransportPredictiveNodeDecisionBundleRecomputation",
        PredictiveFitKind::Trial => "TrialIpwPredictiveNodeDecisionBundleRecomputation",
        PredictiveFitKind::Cyclic => "CyclicTargetFeedbackNodeDecisionBundleRecomputation",
        PredictiveFitKind::Graph => "GraphImputationPredictiveNodeDecisionBundleRecomputation",
        PredictiveFitKind::Bridge => "AssessedBridgePosteriorNodeDecisionBundleRecomputation",
    };
    assert_eq!(plan.plan_view().route().as_str(), expected_route);
    let receipt = plan.execute(ctx).unwrap();
    assert_eq!(receipt.chain_identity, result.chain.identity());
    assert!(receipt.dependencies.iter().any(|node| node.node_id == "fit"));
    let portable = plan.export_portable(&receipt, ctx).unwrap();
    assert_eq!(
        consume_portable_predictive_fit_decision_bundle(
            &portable,
            bundle.identity(),
            plan.lowered().identity(),
            ctx
        )
        .unwrap(),
        receipt
    );
    assert!(
        consume_portable_predictive_fit_decision_bundle(
            &portable,
            "wrong",
            plan.lowered().identity(),
            ctx
        )
        .is_err()
    );
    let mut corrupted = producer.to_vec();
    corrupted[0] ^= 1;
    let mut refused = BundleBuilder::new();
    assert!(match kind { PredictiveFitKind::Joint => refused.add_joint_transport_predictive_fit("fit", &corrupted, producer_program, ctx),
        PredictiveFitKind::Trial => refused.add_trial_ipw_predictive_fit("fit", &corrupted, producer_program, ctx),
        PredictiveFitKind::Cyclic => refused.add_cyclic_target_feedback_fit("fit", &corrupted, producer_program, ctx),
        PredictiveFitKind::Graph => refused.add_graph_imputation_predictive_fit("fit", &corrupted, producer_program, ctx),
        PredictiveFitKind::Bridge => refused.add_assessed_bridge_posterior_fit("fit", &corrupted, producer_program, ctx) }.is_err());
}

#[test]
fn trial_predictive_fit_chain_and_decision_replay() {
    let (diagram, query, worlds, decision) = fixture();
    let rows = TrialIpwRows {
        unit_ids: (0..80).map(|i| format!("unit-{i}")).collect(),
        outcomes: (0..80).map(|i| if i < 40 { f64::from(i % 4) } else { 0.0 }).collect(),
        treatment: (0..80).map(|i| i % 2 == 0).collect(),
        trial: (0..80).map(|i| i < 40).collect(),
        selection_probability: vec![0.5; 80],
        treatment_probability: vec![0.5; 80],
        draws: 100,
        seed: 31,
    };
    let ctx = ExecutionContext::for_tests(31);
    let request = TrialIpwPredictiveDecisionRequest {
        diagram,
        query,
        rows,
        worlds,
        decision: decision.clone(),
        independent_potential_outcomes_given_mass: true,
    };
    let plan = compile_trial_ipw_predictive_decision(request.clone()).unwrap();
    let result = plan.execute(&ctx).unwrap();
    let artifact = plan.export(&result, "trial", &ctx).unwrap();
    check(
        PredictiveFitKind::Trial,
        &artifact,
        plan.lowered().identity(),
        &result.decision,
        &decision,
        &ctx,
    );
    let mut changed = request;
    changed.rows.outcomes[0] += 10.0;
    let foreign = compile_trial_ipw_predictive_decision(changed).unwrap();
    let foreign_result = foreign.execute(&ctx).unwrap();
    let foreign_artifact = foreign.export(&foreign_result, "foreign", &ctx).unwrap();
    let terminal =
        compile_action_world_chain_decision(result.decision.chain.record().clone(), decision, &ctx)
            .unwrap();
    let mut builder = BundleBuilder::new();
    builder
        .add_trial_ipw_predictive_fit("fit", &foreign_artifact, foreign.lowered().identity(), &ctx)
        .unwrap();
    builder
        .add_artifact(
            Some("chain"),
            NodeKind::PropagationChain,
            &result.decision.chain.export().unwrap(),
        )
        .unwrap();
    builder
        .add_predictive_chain_decision(
            "decision",
            &terminal.export(&result.decision, "decision").unwrap(),
            &ctx,
        )
        .unwrap();
    builder.connect("fit", "chain").unwrap();
    builder.connect("chain", "decision").unwrap();
    let mixed = compile_trial_ipw_predictive_node_decision_bundle(
        builder.build().unwrap(),
        "fit",
        "chain",
        "decision",
        &ctx,
    )
    .unwrap();
    assert!(mixed.execute(&ctx).is_err());
}

#[test]
fn joint_predictive_fit_chain_and_decision_replay() {
    let (diagram, query, worlds, decision) = fixture();
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
        sharing: SourceSharing::SharedVaryingBlock,
        dependence: SourceDependence::IndependentSamples,
        priors: JointPriors {
            invariant: GaussianPrior::isotropic(1, 0.0, 10.0, PriorProvenance::Declared),
            varying: GaussianPrior::isotropic(1, 0.0, 10.0, PriorProvenance::Declared),
        },
        max_unsupported_mass: 0.0,
        conflict_z_threshold: 3.0,
    };
    let ctx = ExecutionContext::for_tests(92);
    let plan =
        compile_joint_transport_predictive_decision(JointTransportPredictiveDecisionRequest {
            diagram,
            query,
            model,
            sources: vec![source],
            target,
            options: JointTransportOptions { draws: 1024, seed: 92 },
            target_noise_variance: 0.1,
            target_noise_correlation: 0.0,
            worlds,
            decision: decision.clone(),
        })
        .unwrap();
    let result = plan.execute(&ctx).unwrap();
    let artifact = plan.export(&result, "joint", &ctx).unwrap();
    check(
        PredictiveFitKind::Joint,
        &artifact,
        plan.lowered().identity(),
        &result.decision,
        &decision,
        &ctx,
    );
}

#[test]
fn graph_predictive_fit_chain_and_decision_replay() {
    use crate::analysis::graph_imputation::{CompiledGraphImputation, GraphImputationSpec};
    use crate::analysis::graph_imputation_effect_decision::compile_graph_imputation_bernoulli_predictive_world_decision;
    use antecedent_core::graph_imputation_contract::GraphSupport;
    use antecedent_data::{ImputationSpec, TabularData};
    use antecedent_io::measured_inference::graph_candidate::TransportStandardization;
    let t: Vec<_> = (0..96).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
    let x: Vec<_> = (0..96).map(|i| f64::from(u8::from(i % 4 < 2))).collect();
    let y: Vec<_> = (0..96)
        .map(|i| {
            if i % 13 == 0 {
                f64::NAN
            } else {
                f64::from(u8::from((i * 7 + i / 3) % 11 < 4 + usize::from(i % 2 == 0)))
            }
        })
        .collect();
    let ctx = ExecutionContext::for_tests(41);
    let table = TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("x", x.as_slice()),
    ])
    .unwrap();
    let producer = CompiledGraphImputation::compile(
        table,
        &GraphImputationSpec {
            treatment: VariableId::from_raw(0),
            outcome: VariableId::from_raw(1),
            support: GraphSupport::TreatmentRootPreTreatmentCovariates,
            imputation: ImputationSpec::new().with_imputations(2).with_iterations(2).with_seed(17),
            domains: Arc::from(vec![1, 1, 1]),
            posterior_draws: 100,
            prior_scale: 1.0,
            execution_seed: 19,
        },
        &ctx,
    )
    .unwrap();
    let target = TransportStandardization {
        columns: vec![(2, x)],
        selection_targets: vec![2],
        source_population: "source".into(),
        target_population: "target".into(),
        source_sample: producer.source_identity().to_hex(),
        target_sample: "target-sample".into(),
        source_unit_ids: (0..96).map(|i| format!("source-{i}")).collect(),
        target_unit_ids: (0..96).map(|i| format!("target-{i}")).collect(),
    };
    let quantities = [0, 1].map(|a| ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "outcome".into(),
        role: QuantityRole::Outcome,
        units: "outcome".into(),
        population_id: "target".into(),
        regime_id: format!("do(t={a})"),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    });
    let decision = DecisionContract {
        actions: quantities
            .iter()
            .enumerate()
            .map(|(i, q)| DecisionAction {
                id: format!("action-{i}"),
                kind: ActionKind::Intervention,
                inputs: vec![q.clone()],
                utility: UtilityExpr::Input(0),
            })
            .collect(),
        utility_units: "outcome".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: Vec::new(),
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
    };
    let plan = compile_graph_imputation_bernoulli_predictive_world_decision(
        producer,
        target,
        quantities,
        decision.clone(),
    )
    .unwrap();
    let result = plan.execute(&ctx).unwrap();
    let artifact = plan.export(&result, "graph", &ctx).unwrap();
    check(
        PredictiveFitKind::Graph,
        &artifact,
        plan.lowered().identity(),
        &result.decision,
        &decision,
        &ctx,
    );
}

#[test]
fn cyclic_target_feedback_fit_chain_and_decision_replay() {
    use crate::analysis::cyclic_target_feedback_decision::{
        compile_cyclic_target_feedback_decision, tests::fixture,
    };
    let ctx = ExecutionContext::for_tests(47);
    let (request, decision) = fixture();
    let plan = compile_cyclic_target_feedback_decision(request, decision.clone(), &ctx).unwrap();
    let result = plan.execute(&ctx).unwrap();
    let artifact = plan.export(&result, "target-feedback", &ctx).unwrap();
    check(
        PredictiveFitKind::Cyclic,
        &artifact,
        plan.lowered().identity(),
        &result.decision,
        &decision,
        &ctx,
    );
}

/// `bundle` with each node and edge endpoint named `from` renamed to `to`; node bytes and
/// declared identities are kept.
fn site_renamed(bundle: &CompositionBundle, renames: &[(&str, &str)]) -> CompositionBundle {
    let rename = |id: &str| {
        renames
            .iter()
            .find(|(from, _)| *from == id)
            .map_or_else(|| id.to_owned(), |(_, to)| (*to).to_owned())
    };
    let nodes = bundle
        .nodes()
        .iter()
        .map(|node| {
            let mut node = node.clone();
            node.id = rename(node.id.as_str());
            node
        })
        .collect();
    let edges: Vec<(String, String)> = bundle
        .edges()
        .iter()
        .map(|edge| (rename(edge.from.as_str()), rename(edge.to.as_str())))
        .collect();
    CompositionBundle::new(nodes, &edges, &BundleLimits::default()).expect("renamed bundle")
}

#[test]
fn site_composition_predictive_fit_decision_question_binds_node_names_and_graph() {
    use crate::identity_laws::all_distinct;
    let (diagram, query, worlds, decision) = fixture();
    let rows = TrialIpwRows {
        unit_ids: (0..80).map(|i| format!("unit-{i}")).collect(),
        outcomes: (0..80).map(|i| if i < 40 { f64::from(i % 4) } else { 0.0 }).collect(),
        treatment: (0..80).map(|i| i % 2 == 0).collect(),
        trial: (0..80).map(|i| i < 40).collect(),
        selection_probability: vec![0.5; 80],
        treatment_probability: vec![0.5; 80],
        draws: 100,
        seed: 31,
    };
    let ctx = ExecutionContext::for_tests(31);
    let request = TrialIpwPredictiveDecisionRequest {
        diagram,
        query,
        rows,
        worlds,
        decision: decision.clone(),
        independent_potential_outcomes_given_mass: true,
    };
    let plan = compile_trial_ipw_predictive_decision(request).unwrap();
    let result = plan.execute(&ctx).unwrap();
    let artifact = plan.export(&result, "trial", &ctx).unwrap();
    let terminal =
        compile_action_world_chain_decision(result.decision.chain.record().clone(), decision, &ctx)
            .unwrap();
    let mut builder = BundleBuilder::new();
    builder
        .add_trial_ipw_predictive_fit("fit", &artifact, plan.lowered().identity(), &ctx)
        .unwrap();
    builder
        .add_artifact(
            Some("chain"),
            NodeKind::PropagationChain,
            &result.decision.chain.export().unwrap(),
        )
        .unwrap();
    builder
        .add_predictive_chain_decision(
            "decision",
            &terminal.export(&result.decision, "decision").unwrap(),
            &ctx,
        )
        .unwrap();
    builder.connect("fit", "chain").unwrap();
    builder.connect("chain", "decision").unwrap();
    let bundle = builder.build().unwrap();
    let identity = |bundle: CompositionBundle, names: [&str; 3]| {
        let [fit_id, chain_id, decision_id] = names;
        compile_predictive_fit_decision_bundle(
            bundle,
            fit_id,
            chain_id,
            decision_id,
            &ctx,
            PredictiveFitKind::Trial,
        )
        .unwrap()
        .lowered()
        .identity()
    };
    all_distinct(&[
        identity(bundle.clone(), ["fit", "chain", "decision"]),
        identity(site_renamed(&bundle, &[("fit", "fit-b")]), ["fit-b", "chain", "decision"]),
        identity(site_renamed(&bundle, &[("chain", "chain-b")]), ["fit", "chain-b", "decision"]),
        identity(
            site_renamed(&bundle, &[("decision", "decision-b")]),
            ["fit", "chain", "decision-b"],
        ),
    ]);
}
