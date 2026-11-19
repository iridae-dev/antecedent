use super::*;
use crate::analysis::cyclic_action_world_decision::CyclicDecisionOutput;
use crate::analysis::cyclic_posterior_action_decision::{
    CyclicPosteriorDecisionRequest, compile_cyclic_posterior_action_decision,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_estimate::cyclic_likelihood::CyclicParameter;
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_model::nonlinear_equilibrium::{
    Activation, Clamp, Equation, EquilibriumModel, RegimeRow, SolvePolicy,
};

fn quantity(regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "wishlists".into(),
        variable_name: "wishlists".into(),
        role: QuantityRole::Outcome,
        units: "count".into(),
        population_id: "game".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn request() -> CyclicPosteriorDecisionRequest {
    let training = (0..260)
        .map(|i| {
            let ad = f64::from(i % 5) * 0.4;
            let ew = (f64::from(i) * 0.19).sin() * 0.2;
            let et = (f64::from(i) * 0.23).cos() * 0.2;
            let wishlists = (0.2 * ad + ew + 0.3 * et) / (1.0 - 0.06);
            let traffic = 0.2 * wishlists + et;
            RegimeRow {
                values: vec![wishlists, traffic, ad],
                clamps: vec![Clamp { variable: 2, value: ad }],
            }
        })
        .collect();
    CyclicPosteriorDecisionRequest {
        model: EquilibriumModel {
            equations: vec![
                Equation {
                    id: "wishlists".into(),
                    intercept: 0.0,
                    weights: vec![0.0, 0.3, 0.2],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: Vec::new(),
                },
                Equation {
                    id: "traffic".into(),
                    intercept: 0.0,
                    weights: vec![0.2, 0.0, 0.0],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: Vec::new(),
                },
                Equation {
                    id: "ads".into(),
                    intercept: 0.0,
                    weights: vec![0.0, 0.0, 0.0],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: Vec::new(),
                },
            ],
            noise_covariance: vec![0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.04],
        },
        parameters: vec![CyclicParameter::Weight { equation: 0, parent: 1 }],
        training,
        prior_sd: 1.0,
        chains: 2,
        warmup: 300,
        draws_per_chain: 300,
        proposal_scale: 0.08,
        worlds: vec![
            vec![Clamp { variable: 2, value: 0.0 }],
            vec![Clamp { variable: 2, value: 1.0 }],
        ],
        initial: vec![0.0; 3],
        solve_policy: SolvePolicy::default(),
    }
}

fn outputs() -> Vec<CyclicDecisionOutput> {
    ["ordinary", "promoted"]
        .iter()
        .enumerate()
        .map(|(world, regime)| CyclicDecisionOutput {
            world,
            equation: 0,
            quantity: ScientificQuantityWire::from(&quantity(regime)),
        })
        .collect()
}

fn contract() -> DecisionContract {
    DecisionContract {
        actions: ["ordinary", "promoted"]
            .iter()
            .map(|regime| DecisionAction {
                id: (*regime).into(),
                kind: ActionKind::Intervention,
                inputs: vec![quantity(regime)],
                utility: UtilityExpr::Input(0),
            })
            .collect(),
        utility_units: "wishlist_count".into(),
        criterion: DecisionCriterion::ExpectedRegret,
        constraints: Vec::new(),
        target_population: "game".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
    }
}

#[test]
#[ignore = "source-only posterior method; scientific record has not been earned"]
fn cyclic_fit_reproduces_the_bundle_terminal_from_raw_evidence() {
    let ctx = ExecutionContext::for_tests(47);
    let producer = compile_cyclic_posterior_action_decision(request(), outputs(), contract(), &ctx)
        .expect("compile producer");
    let result = producer.execute(&ctx).expect("fit producer");
    let artifact = producer.export(&result, "producer", &ctx).expect("export producer");
    let mut builder = BundleBuilder::new();
    builder
        .add_artifact(
            Some("terminal"),
            NodeKind::PropagationChain,
            &result.decision.chain.export().expect("export chain"),
        )
        .expect("add chain");
    let bundle = builder.build().expect("build bundle");
    let bundle_id = bundle.identity().to_owned();
    let plan = compile_cyclic_posterior_bundle_recomputation(
        bundle,
        "terminal",
        artifact,
        producer.lowered().identity(),
        &ctx,
    )
    .expect("compile bundle");
    assert_eq!(plan.plan_view().route().as_str(), "CyclicPosteriorBundleRecomputation");
    let receipt = plan.execute(&ctx).expect("refit and execute");
    let portable = plan.export_portable(&receipt, &ctx).expect("export bundle");
    assert_eq!(
        consume_portable_cyclic_posterior_bundle_recomputation(
            &portable,
            &bundle_id,
            plan.lowered().identity(),
            &ctx,
        )
        .expect("independent replay"),
        receipt
    );
    assert!(
        consume_portable_cyclic_posterior_bundle_recomputation(
            &portable,
            "wrong",
            plan.lowered().identity(),
            &ctx,
        )
        .is_err()
    );
    let mut wire: PortableCyclicPosteriorBundleWire = antecedent_io::from_cbor(&portable).unwrap();
    wire.producer_artifact[0] ^= 1;
    let changed = antecedent_io::to_cbor(&wire).unwrap();
    assert!(
        consume_portable_cyclic_posterior_bundle_recomputation(
            &changed,
            &bundle_id,
            plan.lowered().identity(),
            &ctx,
        )
        .is_err()
    );
}

#[test]
#[ignore = "source-only cyclic posterior fit; scientific record has not been earned"]
fn embedded_fit_node_is_refitted_by_the_bundle_consumer() {
    let ctx = ExecutionContext::for_tests(47);
    let producer = compile_cyclic_posterior_action_decision(request(), outputs(), contract(), &ctx)
        .expect("compile producer");
    let result = producer.execute(&ctx).expect("fit producer");
    let artifact = producer.export(&result, "producer", &ctx).expect("export producer");
    let mut corrupted = artifact.clone();
    corrupted[0] ^= 1;
    assert!(
        BundleBuilder::new()
            .add_cyclic_posterior_fit("fit", &corrupted, producer.lowered().identity(), &ctx,)
            .is_err()
    );
    let mut builder = BundleBuilder::new();
    builder
        .add_cyclic_posterior_fit("fit", &artifact, producer.lowered().identity(), &ctx)
        .expect("add replayable fit");
    builder
        .add_artifact(
            Some("terminal"),
            NodeKind::PropagationChain,
            &result.decision.chain.export().expect("export chain"),
        )
        .expect("add terminal");
    let disconnected = builder.build().unwrap();
    assert!(
        compile_cyclic_posterior_node_bundle_recomputation(disconnected, "fit", "terminal", &ctx,)
            .is_err()
    );
    builder.connect("fit", "terminal").unwrap();
    let bundle = builder.build().unwrap();
    let bundle_bytes = export_bundle(&bundle, "fit-bundle").unwrap();
    let consumed = cyclic_posterior_bundle_consumer(&ctx)
        .consume(
            &bundle_bytes,
            &BundleLimits::default(),
            bundle.identity(),
            &SuppliedSources::default(),
        )
        .unwrap();
    assert!(consumed.node_replayed("fit"));
    assert!(consumed.all_replayed());
    let plan =
        compile_cyclic_posterior_node_bundle_recomputation(bundle.clone(), "fit", "terminal", &ctx)
            .expect("compile fit dependency");
    assert_ne!(plan.lowered().identity(), plan.inner.lowered().identity());
    assert!(matches!(
        plan.lowered().statistical().realization(),
        Realization::NestedFit { law: NestedFitLaw::Predictive, .. }
    ));
    let receipt = plan.execute(&ctx).expect("refit node and recompute chain");
    assert!(receipt.bundle.dependencies.iter().any(|dependency| dependency.node_id == "fit"));
    let portable = plan.export_portable(&receipt, &ctx).expect("portable node bundle");
    assert_eq!(
        consume_portable_cyclic_posterior_node_bundle_recomputation(
            &portable,
            bundle.identity(),
            plan.lowered().identity(),
            &ctx,
        )
        .expect("portable refit"),
        receipt
    );
    assert!(
        consume_portable_cyclic_posterior_node_bundle_recomputation(
            &portable,
            "wrong",
            plan.lowered().identity(),
            &ctx,
        )
        .is_err()
    );
}

#[test]
#[ignore = "source-only trial IPW bundle; no sampling or decision reliability is earned"]
#[allow(clippy::too_many_lines, reason = "one linear sequence of steps that reads top to bottom")]
fn embedded_trial_fit_is_refitted_and_matches_the_live_chain() {
    use crate::analysis::trial_ipw_transport_decision::{
        TrialIpwRows, TrialIpwTransportDecisionRequest, compile_trial_ipw_transport_decision,
    };
    use antecedent_core::{
        ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery, TransportQuery, VariableId,
    };
    use antecedent_graph::SelectionDiagram;
    let treatment = VariableId::from_raw(0);
    let outcome = VariableId::from_raw(2);
    let mut graph = antecedent_graph::Admg::with_variables(3);
    let source_node = graph.dense_of_static(treatment).unwrap();
    let outcome_node = graph.dense_of_static(outcome).unwrap();
    graph.insert_directed(source_node, outcome_node).unwrap();
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
        variable_name: "effect".into(),
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
                utility: UtilityExpr::Input(0),
            },
            DecisionAction {
                id: "hold".into(),
                kind: ActionKind::External,
                inputs: vec![quantity.clone()],
                utility: UtilityExpr::Const(0.0),
            },
        ],
        utility_units: "score".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: Vec::new(),
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
    };
    let rows = TrialIpwRows {
        unit_ids: (0..80).map(|i| format!("unit-{i}")).collect(),
        outcomes: (0..80).map(|i| if i < 40 && i % 2 == 0 { 1.0 } else { 0.0 }).collect(),
        treatment: (0..80).map(|i| i % 2 == 0).collect(),
        trial: (0..80).map(|i| i < 40).collect(),
        selection_probability: vec![0.5; 80],
        treatment_probability: vec![0.5; 80],
        draws: 100,
        seed: 31,
    };
    let request = TrialIpwTransportDecisionRequest { diagram, query, rows, quantity, decision };
    let ctx = ExecutionContext::for_tests(31);
    let producer = compile_trial_ipw_transport_decision(request).expect("compile trial");
    let result = producer.execute(&ctx).expect("fit trial");
    assert_eq!(result.calibration, "not_measured");
    let artifact = producer.export(&result, "trial", &ctx).expect("export trial");
    let mut corrupted = artifact.clone();
    corrupted[0] ^= 1;
    assert!(
        BundleBuilder::new()
            .add_trial_ipw_transport_fit("fit", &corrupted, producer.lowered().identity(), &ctx,)
            .is_err()
    );
    let mut builder = BundleBuilder::new();
    builder
        .add_trial_ipw_transport_fit("fit", &artifact, producer.lowered().identity(), &ctx)
        .expect("add refitted node");
    builder
        .add_artifact(
            Some("terminal"),
            NodeKind::PropagationChain,
            &result.decision.chain.export().unwrap(),
        )
        .unwrap();
    assert!(
        compile_trial_ipw_node_bundle_recomputation(
            builder.build().unwrap(),
            "fit",
            "terminal",
            &ctx,
        )
        .is_err()
    );
    builder.connect("fit", "terminal").unwrap();
    let bundle = builder.build().unwrap();
    let encoded = export_bundle(&bundle, "trial-bundle").unwrap();
    let consumed = trial_ipw_bundle_consumer(&ctx)
        .consume(&encoded, &BundleLimits::default(), bundle.identity(), &SuppliedSources::default())
        .unwrap();
    assert!(consumed.node_replayed("fit"));
    assert!(consumed.all_replayed());
    let plan = compile_trial_ipw_node_bundle_recomputation(bundle.clone(), "fit", "terminal", &ctx)
        .expect("compile trial bundle");
    assert_eq!(plan.plan_view().route().as_str(), "TrialIpwNodeBundleRecomputation");
    assert!(matches!(
        plan.lowered().statistical().realization(),
        Realization::NestedFit { law: NestedFitLaw::Posterior, .. }
    ));
    let receipt = plan.execute(&ctx).expect("refit source and execute chain");
    assert!(receipt.bundle.dependencies.iter().any(|node| node.node_id == "fit"));
    let portable = plan.export_portable(&receipt, &ctx).unwrap();
    assert_eq!(
        consume_portable_trial_ipw_node_bundle_recomputation(
            &portable,
            bundle.identity(),
            plan.lowered().identity(),
            &ctx,
        )
        .unwrap(),
        receipt
    );
    assert!(
        consume_portable_trial_ipw_node_bundle_recomputation(
            &portable,
            "wrong",
            plan.lowered().identity(),
            &ctx,
        )
        .is_err()
    );
}

#[test]
#[ignore = "source-only graph transport fit bundle; no whole-route calibration is earned"]
#[allow(clippy::too_many_lines, reason = "one linear sequence of steps that reads top to bottom")]
fn embedded_graph_transport_fit_replays_raw_evidence_and_chain() {
    use crate::analysis::graph_imputation::{CompiledGraphImputation, GraphImputationSpec};
    use crate::analysis::graph_imputation_transport_decision::compile_graph_imputation_transport_decision;
    use antecedent_core::{VariableId, graph_imputation_contract::GraphSupport};
    use antecedent_data::{ImputationSpec, TabularData};
    use antecedent_io::measured_inference::graph_candidate::TransportStandardization;
    let mut treatment = Vec::new();
    let mut outcome = Vec::new();
    let mut covariate = Vec::new();
    for row in 0..80 {
        let x = f64::from(u8::from(row % 2 == 0));
        let a = f64::from(u8::from(row % 4 == 0 || row % 4 == 3));
        treatment.push(a);
        outcome.push(if row % 13 == 0 {
            f64::NAN
        } else {
            0.2 * x + a * (0.5 + 0.5 * x) + 0.1 * f64::from(row % 5)
        });
        covariate.push(x);
    }
    let table = TabularData::from_f64_columns([
        ("t", treatment.as_slice()),
        ("y", outcome.as_slice()),
        ("x", covariate.as_slice()),
    ])
    .unwrap();
    let ctx = ExecutionContext::for_tests(19);
    let producer = CompiledGraphImputation::compile(
        table,
        &GraphImputationSpec {
            treatment: VariableId::from_raw(0),
            outcome: VariableId::from_raw(1),
            support: GraphSupport::TreatmentRootPreTreatmentCovariates,
            imputation: ImputationSpec::new().with_imputations(2).with_iterations(2).with_seed(17),
            domains: Arc::<[u8]>::from(vec![1, 0, 1]),
            posterior_draws: 100,
            prior_scale: 1.0,
            execution_seed: 19,
        },
        &ctx,
    )
    .unwrap();
    let target = TransportStandardization {
        columns: vec![(2, (0..80).map(|row| f64::from(u8::from(row % 5 != 0))).collect())],
        selection_targets: vec![2],
        source_population: "source".into(),
        target_population: "target".into(),
        source_sample: producer.source_identity().to_hex(),
        target_sample: "independent-target-sample".into(),
        source_unit_ids: (0..80).map(|row| format!("source-{row}")).collect(),
        target_unit_ids: (0..80).map(|row| format!("target-{row}")).collect(),
    };
    let quantity = ScientificQuantity {
        variable_id: "y".into(),
        variable_name: "target effect".into(),
        role: QuantityRole::Outcome,
        units: "score".into(),
        population_id: "target".into(),
        regime_id: "do(t=1)-do(t=0)".into(),
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
                utility: UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Const(0.2)),
            },
            DecisionAction {
                id: "hold".into(),
                kind: ActionKind::External,
                inputs: vec![quantity.clone()],
                utility: UtilityExpr::Const(0.0),
            },
        ],
        utility_units: "score".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: Vec::new(),
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
    };
    let producer =
        compile_graph_imputation_transport_decision(producer, target, quantity, decision)
            .expect("compile graph transport");
    let result = producer.execute(&ctx).expect("fit graph transport");
    assert_eq!(result.calibration, "not_measured");
    let artifact = producer.export(&result, "graph-transport", &ctx).unwrap();
    let mut corrupted = artifact.clone();
    corrupted[0] ^= 1;
    assert!(
        BundleBuilder::new()
            .add_graph_imputation_transport_fit(
                "fit",
                &corrupted,
                producer.lowered().identity(),
                &ctx,
            )
            .is_err()
    );
    let mut builder = BundleBuilder::new();
    builder
        .add_graph_imputation_transport_fit("fit", &artifact, producer.lowered().identity(), &ctx)
        .expect("verify graph transport producer");
    builder
        .add_artifact(
            Some("terminal"),
            NodeKind::PropagationChain,
            &result.decision.chain.export().unwrap(),
        )
        .unwrap();
    assert!(
        compile_graph_imputation_transport_node_bundle_recomputation(
            builder.build().unwrap(),
            "fit",
            "terminal",
            &ctx,
        )
        .is_err()
    );
    builder.connect("fit", "terminal").unwrap();
    let bundle = builder.build().unwrap();
    let encoded = export_bundle(&bundle, "graph-transport-bundle").unwrap();
    let consumed = native_fit_bundle_consumer(&ctx)
        .consume(&encoded, &BundleLimits::default(), bundle.identity(), &SuppliedSources::default())
        .unwrap();
    assert!(consumed.node_replayed("fit"));
    assert!(consumed.all_replayed());
    let plan = compile_graph_imputation_transport_node_bundle_recomputation(
        bundle.clone(),
        "fit",
        "terminal",
        &ctx,
    )
    .expect("compile graph transport bundle");
    assert_eq!(
        plan.plan_view().route().as_str(),
        "GraphImputationTransportNodeBundleRecomputation"
    );
    assert!(matches!(
        plan.lowered().statistical().realization(),
        Realization::NestedFit { law: NestedFitLaw::Posterior, .. }
    ));
    let receipt = plan.execute(&ctx).expect("refit graph and replay chain");
    let portable = plan.export_portable(&receipt, &ctx).unwrap();
    assert_eq!(
        consume_portable_graph_imputation_transport_node_bundle_recomputation(
            &portable,
            bundle.identity(),
            plan.lowered().identity(),
            &ctx,
        )
        .unwrap(),
        receipt
    );
    assert!(
        consume_portable_graph_imputation_transport_node_bundle_recomputation(
            &portable,
            "wrong",
            plan.lowered().identity(),
            &ctx,
        )
        .is_err()
    );
}

#[test]
#[ignore = "source-only mixed-seed posterior replay; no scientific record is earned"]
fn native_fit_consumer_replays_each_bound_seed_and_preserves_cancellation() {
    let mut builder = BundleBuilder::new();
    let mut artifacts = Vec::new();
    for seed in [47_u64, 48] {
        let ctx = ExecutionContext::for_tests(seed);
        let plan = compile_cyclic_posterior_action_decision(request(), outputs(), contract(), &ctx)
            .unwrap();
        let result = plan.execute(&ctx).unwrap();
        let artifact = plan.export(&result, "mixed-seed", &ctx).unwrap();
        builder
            .add_cyclic_posterior_fit(
                &format!("fit-{seed}"),
                &artifact,
                plan.lowered().identity(),
                &ctx,
            )
            .unwrap();
        artifacts.push((artifact, plan.lowered().identity()));
    }
    let bundle = builder.build().unwrap();
    let bytes = export_bundle(&bundle, "mixed-seed-bundle").unwrap();
    let unrelated_ctx = ExecutionContext::for_tests(99);
    let consumed = native_fit_bundle_consumer(&unrelated_ctx)
        .consume(&bytes, &BundleLimits::default(), bundle.identity(), &SuppliedSources::default())
        .unwrap();
    assert!(consumed.node_replayed("fit-47"));
    assert!(consumed.node_replayed("fit-48"));
    // A changed bound seed must still fail the producer's own seed check.
    let node = &builder.nodes[0];
    let NodeSource::Embedded { bytes } = &node.source else { panic!("embedded fit") };
    let mut wire: CyclicPosteriorFitWire = antecedent_io::from_cbor(bytes).unwrap();
    wire.replay_seed = Some(48);
    let changed = antecedent_io::to_cbor(&wire).unwrap();
    assert!(
        CyclicPosteriorFitVerifier { ctx: unrelated_ctx.clone() }
            .verify("fit", &changed, &[])
            .is_err()
    );
    // A legacy node has no seed field and remains tied to the caller context.
    let legacy = antecedent_io::to_cbor(&CyclicPosteriorFitWire {
        version: 1,
        required_feature: CYCLIC_POSTERIOR_FIT_FEATURE.into(),
        producer_program: *artifacts[0].1.as_bytes(),
        artifact: artifacts[0].0.clone(),
        replay_seed: None,
    })
    .unwrap();
    assert!(
        CyclicPosteriorFitVerifier { ctx: ExecutionContext::for_tests(47) }
            .verify("legacy", &legacy, &[])
            .is_ok()
    );
    assert!(
        CyclicPosteriorFitVerifier { ctx: unrelated_ctx.clone() }
            .verify("legacy", &legacy, &[])
            .is_err()
    );
    let cancelled = ExecutionContext::for_tests(99);
    cancelled.cancellation.cancel();
    let consumed = native_fit_bundle_consumer(&cancelled)
        .consume(bytes, &BundleLimits::default(), bundle.identity(), &SuppliedSources::default())
        .unwrap();
    assert!(matches!(
        &consumed.node("fit-47").unwrap().status,
        NodeStatus::Failed { stage: BundleStage::ExecutionCancelled, .. }
    ));
}
