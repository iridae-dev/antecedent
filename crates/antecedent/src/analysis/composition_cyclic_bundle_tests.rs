use super::*;
use crate::analysis::cyclic_posterior_action_decision::{
    compile_cyclic_posterior_action_decision, CyclicPosteriorDecisionRequest,
};
use crate::analysis::cyclic_action_world_decision::CyclicDecisionOutput;
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_estimate::cyclic_likelihood::CyclicParameter;
use antecedent_model::nonlinear_equilibrium::{
    Activation, Clamp, Equation, EquilibriumModel, RegimeRow, SolvePolicy,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;

    fn quantity(regime: &str) -> ScientificQuantity {
        ScientificQuantity {
            variable_id: "wishlists".into(), variable_name: "wishlists".into(),
            role: QuantityRole::Outcome, units: "count".into(),
            population_id: "game".into(), regime_id: regime.into(),
            horizon: 0, functional_id: "outcome".into(),
            conditioning: Vec::new(), transform_id: "identity".into(),
        }
    }

    fn request() -> CyclicPosteriorDecisionRequest {
        let training = (0..260).map(|i| {
            let ad = (i % 5) as f64 * 0.4;
            let ew = (i as f64 * 0.19).sin() * 0.2;
            let et = (i as f64 * 0.23).cos() * 0.2;
            let wishlists = (0.2 * ad + ew + 0.3 * et) / (1.0 - 0.06);
            let traffic = 0.2 * wishlists + et;
            RegimeRow {
                values: vec![wishlists, traffic, ad],
                clamps: vec![Clamp { variable: 2, value: ad }],
            }
        }).collect();
        CyclicPosteriorDecisionRequest {
            model: EquilibriumModel {
                equations: vec![
                    Equation { id: "wishlists".into(), intercept: 0.0,
                        weights: vec![0.0, 0.3, 0.2], activation: Activation::Identity,
                        expression: None, parameters: Vec::new() },
                    Equation { id: "traffic".into(), intercept: 0.0,
                        weights: vec![0.2, 0.0, 0.0], activation: Activation::Identity,
                        expression: None, parameters: Vec::new() },
                    Equation { id: "ads".into(), intercept: 0.0,
                        weights: vec![0.0, 0.0, 0.0], activation: Activation::Identity,
                        expression: None, parameters: Vec::new() },
                ],
                noise_covariance: vec![0.04, 0.0, 0.0,
                    0.0, 0.04, 0.0, 0.0, 0.0, 0.04],
            },
            parameters: vec![CyclicParameter::Weight { equation: 0, parent: 1 }],
            training, prior_sd: 1.0,
            chains: 2, warmup: 300, draws_per_chain: 300,
            proposal_scale: 0.08,
            worlds: vec![
                vec![Clamp { variable: 2, value: 0.0 }],
                vec![Clamp { variable: 2, value: 1.0 }],
            ],
            initial: vec![0.0; 3], solve_policy: SolvePolicy::default(),
        }
    }

    fn outputs() -> Vec<CyclicDecisionOutput> {
        ["ordinary", "promoted"].iter().enumerate().map(|(world, regime)| {
            CyclicDecisionOutput {
                world, equation: 0,
                quantity: ScientificQuantityWire::from(&quantity(regime)),
            }
        }).collect()
    }

    fn contract() -> DecisionContract {
        DecisionContract {
            actions: ["ordinary", "promoted"].iter().map(|regime| DecisionAction {
                id: (*regime).into(), kind: ActionKind::Intervention,
                inputs: vec![quantity(regime)], utility: UtilityExpr::Input(0),
            }).collect(),
            utility_units: "wishlist_count".into(),
            criterion: DecisionCriterion::ExpectedRegret,
            constraints: Vec::new(), target_population: "game".into(),
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
    builder.add_artifact(Some("terminal"), NodeKind::PropagationChain,
        &result.decision.chain.export().expect("export chain")).expect("add chain");
    let bundle = builder.build().expect("build bundle");
    let bundle_id = bundle.identity().to_owned();
    let plan = compile_cyclic_posterior_bundle_recomputation(
        bundle, "terminal", artifact, producer.lowered().identity(), &ctx,
    ).expect("compile bundle");
    assert_eq!(plan.plan_view().route().as_str(), "CyclicPosteriorBundleRecomputation");
    let receipt = plan.execute(&ctx).expect("refit and execute");
    let portable = plan.export_portable(&receipt, &ctx).expect("export bundle");
    assert_eq!(consume_portable_cyclic_posterior_bundle_recomputation(
        &portable, &bundle_id, plan.lowered().identity(), &ctx,
    ).expect("independent replay"), receipt);
    assert!(consume_portable_cyclic_posterior_bundle_recomputation(
        &portable, "wrong", plan.lowered().identity(), &ctx,
    ).is_err());
    let mut wire: PortableCyclicPosteriorBundleWire = antecedent_io::from_cbor(&portable).unwrap();
    wire.producer_artifact[0] ^= 1;
    let changed = antecedent_io::to_cbor(&wire).unwrap();
    assert!(consume_portable_cyclic_posterior_bundle_recomputation(
        &changed, &bundle_id, plan.lowered().identity(), &ctx,
    ).is_err());
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
    assert!(BundleBuilder::new().add_cyclic_posterior_fit(
        "fit", &corrupted, producer.lowered().identity(), &ctx,
    ).is_err());
    let mut builder = BundleBuilder::new();
    builder.add_cyclic_posterior_fit("fit", &artifact, producer.lowered().identity(), &ctx)
        .expect("add replayable fit");
    builder.add_artifact(Some("terminal"), NodeKind::PropagationChain,
        &result.decision.chain.export().expect("export chain")).expect("add terminal");
    let disconnected = builder.build().unwrap();
    assert!(compile_cyclic_posterior_node_bundle_recomputation(
        disconnected, "fit", "terminal", &ctx,
    ).is_err());
    builder.connect("fit", "terminal").unwrap();
    let bundle = builder.build().unwrap();
    let bundle_bytes = export_bundle(&bundle, "fit-bundle").unwrap();
    let consumed = cyclic_posterior_bundle_consumer(&ctx).consume(
        &bundle_bytes, &BundleLimits::default(), bundle.identity(), &SuppliedSources::default(),
    ).unwrap();
    assert!(consumed.node_replayed("fit"));
    assert!(consumed.all_replayed());
    let plan = compile_cyclic_posterior_node_bundle_recomputation(bundle.clone(), "fit", "terminal", &ctx)
        .expect("compile fit dependency");
    assert_ne!(plan.lowered().identity(), plan.inner.lowered().identity());
    let receipt = plan.execute(&ctx).expect("refit node and recompute chain");
    assert!(receipt.bundle.dependencies.iter().any(|dependency| dependency.node_id == "fit"));
    let portable = plan.export_portable(&receipt, &ctx).expect("portable node bundle");
    assert_eq!(consume_portable_cyclic_posterior_node_bundle_recomputation(
        &portable, bundle.identity(), plan.lowered().identity(), &ctx,
    ).expect("portable refit"), receipt);
    assert!(consume_portable_cyclic_posterior_node_bundle_recomputation(
        &portable, "wrong", plan.lowered().identity(), &ctx,
    ).is_err());
}

#[test]
#[ignore = "source-only trial IPW bundle; no sampling or decision reliability is earned"]
fn embedded_trial_fit_is_refitted_and_matches_the_live_chain() {
    use antecedent_core::{ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery, TransportQuery, VariableId};
    use antecedent_graph::SelectionDiagram;
    use crate::analysis::trial_ipw_transport_decision::{
        compile_trial_ipw_transport_decision, TrialIpwRows, TrialIpwTransportDecisionRequest,
    };
        let treatment = VariableId::from_raw(0);
        let outcome = VariableId::from_raw(2);
        let mut graph = antecedent_graph::Admg::with_variables(3);
        let source_node = graph.dense_of_static(treatment).unwrap();
        let outcome_node = graph.dense_of_static(outcome).unwrap();
        graph.insert_directed(source_node, outcome_node).unwrap();
        let diagram = SelectionDiagram::try_new(graph, [VariableId::from_raw(1)]).unwrap();
        let query = TransportQuery::new(ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome, treatment: ContinuousDomain::new(treatment,
                GridSpec::Values(Arc::from([0.0, 1.0]))),
        }), "trial", "target", [treatment]);
        let quantity = ScientificQuantity { variable_id: "2".into(), variable_name: "effect".into(),
            role: QuantityRole::Outcome, units: "score".into(), population_id: "target".into(),
            regime_id: "do(0=1)-do(0=0)".into(), horizon: 0,
            functional_id: "average_effect".into(), conditioning: Vec::new(), transform_id: "identity".into() };
        let decision = DecisionContract { actions: vec![
            DecisionAction { id: "treat".into(), kind: ActionKind::External, inputs: vec![quantity.clone()],
                utility: UtilityExpr::Input(0) },
            DecisionAction { id: "hold".into(), kind: ActionKind::External, inputs: vec![quantity.clone()],
                utility: UtilityExpr::Const(0.0) }],
            utility_units: "score".into(), criterion: DecisionCriterion::PosteriorExpectedUtility,
            constraints: Vec::new(), target_population: "target".into(), horizon: 0,
            structural_policy: StructuralPolicy::RequireInvariantBestAction };
        let rows = TrialIpwRows {
            unit_ids: (0..80).map(|i| format!("unit-{i}")).collect(),
            outcomes: (0..80).map(|i| if i < 40 && i % 2 == 0 { 1.0 } else { 0.0 }).collect(),
            treatment: (0..80).map(|i| i % 2 == 0).collect(),
            trial: (0..80).map(|i| i < 40).collect(),
            selection_probability: vec![0.5; 80], treatment_probability: vec![0.5; 80],
            draws: 100, seed: 31,
        };
        let request = TrialIpwTransportDecisionRequest { diagram, query, rows, quantity, decision };
    let ctx = ExecutionContext::for_tests(31);
    let producer = compile_trial_ipw_transport_decision(request).expect("compile trial");
    let result = producer.execute(&ctx).expect("fit trial");
    assert_eq!(result.calibration, "not_measured");
    let artifact = producer.export(&result, "trial", &ctx).expect("export trial");
    let mut corrupted = artifact.clone();
    corrupted[0] ^= 1;
    assert!(BundleBuilder::new().add_trial_ipw_transport_fit(
        "fit", &corrupted, producer.lowered().identity(), &ctx,
    ).is_err());
    let mut builder = BundleBuilder::new();
    builder.add_trial_ipw_transport_fit("fit", &artifact, producer.lowered().identity(), &ctx)
        .expect("add refitted node");
    builder.add_artifact(Some("terminal"), NodeKind::PropagationChain,
        &result.decision.chain.export().unwrap()).unwrap();
    assert!(compile_trial_ipw_node_bundle_recomputation(
        builder.build().unwrap(), "fit", "terminal", &ctx,
    ).is_err());
    builder.connect("fit", "terminal").unwrap();
    let bundle = builder.build().unwrap();
    let encoded = export_bundle(&bundle, "trial-bundle").unwrap();
    let consumed = trial_ipw_bundle_consumer(&ctx).consume(
        &encoded, &BundleLimits::default(), bundle.identity(), &SuppliedSources::default(),
    ).unwrap();
    assert!(consumed.node_replayed("fit"));
    assert!(consumed.all_replayed());
    let plan = compile_trial_ipw_node_bundle_recomputation(bundle.clone(), "fit", "terminal", &ctx)
        .expect("compile trial bundle");
    assert_eq!(plan.plan_view().route().as_str(), "TrialIpwNodeBundleRecomputation");
    let receipt = plan.execute(&ctx).expect("refit source and execute chain");
    assert!(receipt.bundle.dependencies.iter().any(|node| node.node_id == "fit"));
    let portable = plan.export_portable(&receipt, &ctx).unwrap();
    assert_eq!(consume_portable_trial_ipw_node_bundle_recomputation(
        &portable, bundle.identity(), plan.lowered().identity(), &ctx,
    ).unwrap(), receipt);
    assert!(consume_portable_trial_ipw_node_bundle_recomputation(
        &portable, "wrong", plan.lowered().identity(), &ctx,
    ).is_err());
}
