//! Source-only exercise of a nonpredictive parameter-posterior decision bundle.
use antecedent::analysis::composition::{BundleBuilder, NodeKind};
use antecedent::analysis::composition_decision_terminal::{
    compile_posterior_decision_bundle_recomputation,
    consume_portable_posterior_decision_bundle_recomputation,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::ExecutionContext;
use antecedent_design::decision_artifact::{DecisionContractArtifact, DecisionResultArtifact};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::evaluate_contract;
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::quantity_wire::DistributionMeaningWire;

fn quantity() -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "coefficient".into(),
        variable_name: "coefficient".into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: "observational".into(),
        horizon: 0,
        functional_id: "coefficient".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn source() -> DistributionArtifact {
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::ParameterPosterior,
        &[quantity()],
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "declared".into(),
            provider_id: "fixture".into(),
            rng_id: "fixture".into(),
            snapshot_id: "one".into(),
            causal_contract_id: "model".into(),
        },
    )
    .unwrap();
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [3, 1],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::PointOnly,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        vec![1.0, 2.0, 3.0],
    )
    .unwrap()
}

fn contract() -> DecisionContract {
    DecisionContract {
        actions: vec![
            DecisionAction {
                id: "use".into(),
                kind: ActionKind::Policy,
                inputs: vec![quantity()],
                utility: UtilityExpr::Input(0),
            },
            DecisionAction {
                id: "baseline".into(),
                kind: ActionKind::Policy,
                inputs: vec![quantity()],
                utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(0.0)),
            },
        ],
        utility_units: "units".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

#[test]
fn parameter_posterior_decision_replays_without_becoming_predictive() {
    let ctx = ExecutionContext::for_tests(77);
    let source = source();
    let contract = contract();
    let result = evaluate_contract(&contract, &source).unwrap();
    let mut builder = BundleBuilder::new();
    builder
        .add_artifact(
            Some("contract"),
            NodeKind::DecisionContract,
            &DecisionContractArtifact::new(contract.clone()).unwrap().to_bytes("contract").unwrap(),
        )
        .unwrap();
    builder
        .add_artifact(Some("source"), NodeKind::Distribution, &source.to_bytes("source").unwrap())
        .unwrap();
    builder
        .add_artifact(
            Some("result"),
            NodeKind::DecisionResult,
            &DecisionResultArtifact::new(result, &source).to_bytes("result").unwrap(),
        )
        .unwrap();
    builder.connect("contract", "result").unwrap();
    builder.connect("source", "result").unwrap();
    let bundle = builder.build().unwrap();
    let plan = compile_posterior_decision_bundle_recomputation(
        bundle.clone(),
        "result",
        contract.clone(),
        source.clone(),
    )
    .unwrap();
    assert_eq!(plan.plan_view().route().as_str(), "PosteriorDecisionBundleRecomputation");
    let unit = plan.plan_view().unit_support().expect("the route is bound to a unit");
    assert_eq!(unit.unit(), "composition.bundle");
    assert!(unit.supported());
    let receipt = plan.execute(&ctx).unwrap();
    let portable = plan.export_portable(&receipt, &ctx).unwrap();
    assert_eq!(
        consume_portable_posterior_decision_bundle_recomputation(
            &portable,
            bundle.identity(),
            plan.lowered().identity(),
            &ctx,
        )
        .unwrap(),
        receipt
    );
    assert!(
        consume_portable_posterior_decision_bundle_recomputation(
            &portable,
            "wrong",
            plan.lowered().identity(),
            &ctx,
        )
        .is_err()
    );
    let mut outcome_contract = contract;
    outcome_contract.actions[0].inputs[0].functional_id = "outcome".into();
    assert!(compile_posterior_decision_bundle_recomputation(
        bundle, "result", outcome_contract, source,
    ).is_err());
}
