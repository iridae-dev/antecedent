//! Compiler routing of a retained predictive law through a multi-action
//! nonlinear utility, and refusal of posterior parameters as outcome risk.

use antecedent_contract::propagation_chain::{ChainReading, LawMeaning, StageKind, TableConstruction};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{ExecutionContext, SemanticDigest};
use antecedent_core::SupportStatus;
use antecedent_design::composition_boundary::{SupportPolicy, UnsupportedActionPolicy};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_transported_source::{
    TransportDeclaration, TransportedResponse, TransportedRoute,
};
use antecedent_io::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity,
    DistributionMetadata, DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io::propagation_chain_record::{
    ChainCoefficientWire, ChainJointOutputWire, ChainOpWire, ChainSourceWire, ChainStageWire,
};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_io::quantity_wire::DistributionMeaningWire;

use super::propagation_chain::PropagationChain;
use super::propagation_chain_decision::{
    compile_action_world_chain_decision, compile_chain_decision,
    compile_transported_law_chain_decision, consume_chain_decision,
};

fn outcome() -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "steam:wishlists".into(),
        variable_name: "wishlists".into(),
        role: QuantityRole::Outcome,
        units: "count".into(),
        population_id: "game-a".into(),
        regime_id: "declared-traffic".into(),
        horizon: 7,
        functional_id: "outcome".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn draft(meaning: LawMeaning, reading: ChainReading) -> antecedent_io::propagation_chain_record::PropagationChainRecordWire {
    PropagationChain::draft(
        TableConstruction::JointDraws,
        reading,
        "wishlists",
        vec![ChainSourceWire::draws("incoming", meaning, "joint:game-a", &[1.0, 2.0, 3.0, 4.0])],
        vec![ChainStageWire::new(
            "wishlists",
            StageKind::DerivedQuantity,
            &["incoming"],
            ChainOpWire::affine(ChainCoefficientWire::fixed(1.0), ChainCoefficientWire::fixed(0.0)),
        )],
    )
    .with_output_quantity(ScientificQuantityWire::from(&outcome()))
}

fn decision() -> DecisionContract {
    let input = outcome();
    DecisionContract {
        actions: vec![
            DecisionAction {
                id: "promote".into(),
                kind: ActionKind::External,
                inputs: vec![input.clone()],
                utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(0)),
            },
            DecisionAction {
                id: "hold".into(),
                kind: ActionKind::External,
                inputs: vec![input],
                utility: UtilityExpr::Input(0),
            },
        ],
        utility_units: "score".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: Vec::new(),
        target_population: "game-a".into(),
        horizon: 7,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
    }
}

fn joint_draft() -> antecedent_io::propagation_chain_record::PropagationChainRecordWire {
    let mut sales = outcome();
    sales.variable_id = "steam:sales".into();
    sales.variable_name = "sales".into();
    PropagationChain::draft(
        TableConstruction::JointDraws,
        ChainReading::Predictive,
        "wishlists",
        vec![
            ChainSourceWire::draws("traffic", LawMeaning::Predictive, "joint:game-a", &[1.0, 2.0, 3.0, 4.0]),
            ChainSourceWire::draws("conversion", LawMeaning::Predictive, "joint:game-a", &[2.0, 2.0, 2.0, 2.0]),
        ],
        vec![
            ChainStageWire::new("wishlists", StageKind::DerivedQuantity, &["traffic"],
                ChainOpWire::affine(ChainCoefficientWire::fixed(1.0), ChainCoefficientWire::fixed(0.0))),
            ChainStageWire::new("sales", StageKind::DerivedQuantity, &["conversion"],
                ChainOpWire::affine(ChainCoefficientWire::fixed(1.0), ChainCoefficientWire::fixed(0.0))),
        ],
    )
    .with_output_quantity(ScientificQuantityWire::from(&outcome()))
    .with_joint_outputs(vec![
        ChainJointOutputWire { stage: "wishlists".into(), quantity: ScientificQuantityWire::from(&outcome()) },
        ChainJointOutputWire { stage: "sales".into(), quantity: ScientificQuantityWire::from(&sales) },
    ])
}

#[test]
fn joint_chain_law_keeps_cross_quantity_rows_for_nonlinear_utility() {
    let mut contract = decision();
    let mut sales = outcome();
    sales.variable_id = "steam:sales".into();
    sales.variable_name = "sales".into();
    contract.actions[0].inputs.push(sales);
    contract.actions[0].utility = UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1));
    let ctx = ExecutionContext::for_tests(13);
    let plan = compile_chain_decision(joint_draft(), contract, &ctx).expect("joint output compiler");
    let result = plan.execute(&ctx).expect("joint utility");
    let (columns, weights) = result.chain.joint_output_law().expect("retained joint rows");
    assert_eq!(columns.len(), 2);
    assert_eq!(weights, None);
    let mut tampered = result.chain.record().clone();
    tampered.joint_output_draws.as_mut().expect("retained joint law")[0][0] = 99.0_f64.to_bits();
    assert!(tampered.consume(None, &ctx).is_err());
    assert!(result.decision.actions[0].expected_utility > result.decision.actions[1].expected_utility);
    let bytes = plan.export(&result, "joint-wishlists").expect("export");
    let replay = consume_chain_decision(&bytes, Some(result.lowered_program_identity), &ctx)
        .expect("independent joint replay");
    assert_eq!(replay.decision, result.decision);
}

#[test]
fn action_specific_worlds_use_only_the_declared_joint_coupling() {
    let mut draft = joint_draft();
    draft.joint_outputs[1].quantity.regime_id = "alternative-traffic".into();
    for source in &mut draft.sources {
        source.factors.push(antecedent_io::propagation_chain_record::ChainFactorWire::new(
            antecedent_contract::dependence::SharedFactorKind::Disturbance,
            "traffic-world-shock",
        ));
    }
    let mut contract = decision();
    let mut alternative = outcome();
    alternative.variable_id = "steam:sales".into();
    alternative.variable_name = "sales".into();
    alternative.regime_id = "alternative-traffic".into();
    contract.actions[0].utility = UtilityExpr::Input(0);
    contract.actions[0].kind = ActionKind::Intervention;
    contract.actions[1].kind = ActionKind::Intervention;
    contract.actions[1].inputs = vec![alternative];
    contract.criterion = DecisionCriterion::ExpectedRegret;
    let ctx = ExecutionContext::for_tests(13);
    let plan = compile_action_world_chain_decision(draft.clone(), contract.clone(), &ctx)
        .expect("declared joint world rows");
    let result = plan.execute(&ctx).expect("paired regret evaluation");
    assert_eq!(result.decision.n_draws, 4);
    let bytes = plan.export(&result, "paired-worlds").expect("paired artifact");
    assert_eq!(consume_chain_decision(&bytes, Some(result.lowered_program_identity), &ctx)
        .expect("independent paired replay").decision, result.decision);
    draft.sources[1].factors.clear();
    assert!(compile_action_world_chain_decision(draft, contract, &ctx).is_err());
}

#[test]
fn predictive_chain_decision_compiles_and_scores_nonlinear_actions() {
    let ctx = ExecutionContext::for_tests(13);
    let plan = compile_chain_decision(draft(LawMeaning::Predictive, ChainReading::Predictive), decision(), &ctx)
        .expect("one compiled chain and decision");
    let lowered_decision = plan.lowered().evidence().decision.as_ref().expect("decision layer");
    assert_eq!(lowered_decision.actions(), 2);
    let result = plan.execute(&ctx).expect("decision on retained law");
    assert_eq!(result.chain.output_law().map(|(draws, _)| draws.len()), Some(4));
    assert!(result.decision.actions[0].expected_utility > result.decision.actions[1].expected_utility);
    assert_eq!(result.lowered_program_identity, result.chain.lowered_program_identity());
    let bytes = plan.export(&result, "wishlists-decision").expect("versioned artifact");
    let consumed = consume_chain_decision(&bytes, Some(result.lowered_program_identity), &ctx)
        .expect("independent compiler replay");
    assert_eq!(consumed.decision, result.decision);
    assert!(consume_chain_decision(
        &bytes,
        Some(SemanticDigest::from_bytes([0; 32])),
        &ctx,
    ).is_err());
}

#[test]
fn posterior_parameters_do_not_answer_an_outcome_law_decision() {
    let ctx = ExecutionContext::for_tests(13);
    let refused = compile_chain_decision(draft(LawMeaning::Posterior, ChainReading::Posterior), decision(), &ctx)
        .expect_err("a posterior is not a predictive outcome law");
    assert!(refused.to_string().contains("outcome risk requires a predictive law"));
}

fn transported_response(weights: Option<Vec<f64>>) -> TransportedResponse {
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &[outcome()],
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "transported-wishlists".into(),
            provider_id: "declared-transport".into(),
            rng_id: "aligned-transport".into(),
            snapshot_id: "snapshot-1".into(),
            causal_contract_id: "transport-contract".into(),
        },
    ).expect("law identity");
    let law = DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [4, 1],
            weights,
            supported: None,
            calibration: DistributionCalibration::PointOnly,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        vec![1.0, 2.0, 3.0, 4.0],
    ).expect("aligned law");
    TransportedResponse::declared_law(law, &TransportDeclaration {
        route: TransportedRoute::Declared,
        source_identity: "source-trial".into(),
        source_populations: vec!["trial".into()],
        target_population: "game-a".into(),
        transport_identities: vec!["selection-diagram".into()],
        assumptions: vec!["transport mechanism declared".into()],
        support: SupportStatus::Extrapolative,
        uncalibrated_standard_error: None,
        provider_id: "declared-transport".into(),
        snapshot_id: "snapshot-1".into(),
        causal_contract_id: "transport-contract".into(),
    }).expect("transport receipt")
}

#[test]
fn transported_law_decision_requires_accepted_support_and_replays() {
    let response = transported_response(None);
    let ctx = ExecutionContext::for_tests(17);
    assert!(compile_transported_law_chain_decision(
        &response, decision(), &SupportPolicy::require_all(), &ctx,
    ).is_err());
    let accepting = SupportPolicy {
        unsupported: UnsupportedActionPolicy::RequireAllActions,
        weakest_support: SupportStatus::Extrapolative,
    };
    let plan = compile_transported_law_chain_decision(
        &response, decision(), &accepting, &ctx,
    ).expect("accepted transported law compiler");
    let result = plan.execute(&ctx).expect("decision execution");
    assert!(result.decision.actions[0].expected_utility > result.decision.actions[1].expected_utility);
    let bytes = plan.export(&result, "transported-wishlists-decision").expect("artifact");
    let replay = consume_chain_decision(&bytes, Some(result.lowered_program_identity), &ctx)
        .expect("independent replay");
    assert_eq!(replay.decision, result.decision);
}

#[test]
fn zero_mass_transport_rows_keep_exact_joint_weights_through_the_compiler() {
    let response = transported_response(Some(vec![0.0, 1.0, 3.0, 0.0]));
    let ctx = ExecutionContext::for_tests(19);
    let accepting = SupportPolicy {
        unsupported: UnsupportedActionPolicy::RequireAllActions,
        weakest_support: SupportStatus::Extrapolative,
    };
    let plan = compile_transported_law_chain_decision(
        &response, decision(), &accepting, &ctx,
    ).expect("weighted transported law compiler");
    let result = plan.execute(&ctx).expect("weighted decision execution");
    let (draws, weights) = result.chain.output_law()
        .expect("retained aligned law");
    assert_eq!(draws.len(), 2);
    let normalized: Vec<f64> = weights.expect("joint row weights")
        .iter().copied().map(f64::from_bits).collect();
    assert!((normalized[0] - 0.25).abs() < 1e-12);
    assert!((normalized[1] - 0.75).abs() < 1e-12);
    assert!((result.decision.actions[0].expected_utility - 7.75).abs() < 1e-12);
    assert!((result.decision.actions[1].expected_utility - 2.75).abs() < 1e-12);
    let bytes = plan.export(&result, "weighted-transport-decision").unwrap();
    let replay = consume_chain_decision(&bytes, Some(result.lowered_program_identity), &ctx)
        .expect("weighted chain and decision replay");
    assert_eq!(replay.decision, result.decision);
    assert_eq!(replay.chain.output_law(), result.chain.output_law());
}

#[test]
fn weighted_transport_rows_are_shared_across_two_decision_coordinates() {
    let mut sales = outcome();
    sales.variable_id = "steam:sales".into();
    sales.variable_name = "sales".into();
    let quantities = [outcome(), sales.clone()];
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &quantities, DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "transported-wishlists-sales".into(),
            provider_id: "declared-transport".into(),
            rng_id: "aligned-transport".into(),
            snapshot_id: "snapshot-2".into(),
            causal_contract_id: "transport-contract".into(),
        },
    ).unwrap();
    let law = DistributionArtifact::new(
        DistributionMetadata {
            version: 1, identity,
            axes: ["draw".into(), "quantity".into()], shape: [4, 2],
            weights: Some(vec![0.0, 1.0, 3.0, 0.0]), supported: None,
            calibration: DistributionCalibration::NotMeasured,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None, legacy_bindings: None,
        },
        vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0],
    ).unwrap();
    let response = TransportedResponse::declared_law(law, &TransportDeclaration {
        route: TransportedRoute::Declared,
        source_identity: "source-trial".into(),
        source_populations: vec!["trial".into()],
        target_population: "game-a".into(),
        transport_identities: vec!["selection-diagram".into()],
        assumptions: vec!["transport mechanism declared".into()],
        support: SupportStatus::Extrapolative,
        uncalibrated_standard_error: None,
        provider_id: "declared-transport".into(),
        snapshot_id: "snapshot-2".into(),
        causal_contract_id: "transport-contract".into(),
    }).unwrap();
    let mut contract = decision();
    contract.actions[0].inputs.push(sales);
    contract.actions[0].utility = UtilityExpr::product(
        UtilityExpr::Input(0), UtilityExpr::Input(1),
    );
    let ctx = ExecutionContext::for_tests(23);
    let accepting = SupportPolicy {
        unsupported: UnsupportedActionPolicy::RequireAllActions,
        weakest_support: SupportStatus::Extrapolative,
    };
    let plan = compile_transported_law_chain_decision(
        &response, contract, &accepting, &ctx,
    ).unwrap();
    let result = plan.execute(&ctx).unwrap();
    let (columns, weights) = result.chain.joint_output_law().unwrap();
    assert_eq!(columns.len(), 2);
    assert_eq!(columns[0].len(), 2);
    let normalized: Vec<f64> = weights.unwrap().iter().copied()
        .map(f64::from_bits).collect();
    assert!((normalized[0] - 0.25).abs() < 1e-12);
    assert!((normalized[1] - 0.75).abs() < 1e-12);
    assert!((result.decision.actions[0].expected_utility - 77.5).abs() < 1e-12);
    assert!((result.decision.actions[1].expected_utility - 2.75).abs() < 1e-12);
    let bytes = plan.export(&result, "weighted-joint-transport").unwrap();
    let replay = consume_chain_decision(&bytes, Some(result.lowered_program_identity), &ctx)
        .unwrap();
    assert_eq!(replay.decision, result.decision);
    assert_eq!(replay.chain.joint_output_law(), result.chain.joint_output_law());
}
