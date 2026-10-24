//! Composition bundle nodes for an observation query, a scientific bridge and a
//! propagation chain.
//!
//! A bridge of draws publishes the bridged aligned table as a source, so a
//! decision computed over it rests on the bridge node as on a distribution, and a
//! decision computed over a bridge of another seed is refused at a named stage. An
//! observation query and a propagation chain are replayed by their own consumers
//! and publish what they establish; neither claims more than its record holds.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::bridge::{
    AssumptionKind, BridgeAssumption, BridgeContract, BridgeEndpoint, BridgeKind, BridgeMapping,
    BridgeScope, MappingUncertainty, QuantityClass, SourceMappingDependence, ValueKind,
};
use antecedent_contract::propagation_chain::{ChainReading, StageKind, TableConstruction};
use antecedent_contract::{InferenceBasis, QuantityRole, ScientificQuantity};
use antecedent_core::ExecutionContext;
use antecedent_design::composition_bundle::{
    BundleLimits, BundleNode, BundleStage, CompositionBundle, ConsumedBundle, FACT_LAW, FACT_TRUST,
    LAW_JOINT, NodeKind, NodeStatus, SuppliedSources,
};
use antecedent_design::composition_verifiers::{
    describe_artifact, detect_node_kind, standard_consumer,
};
use antecedent_design::decision_artifact::{DecisionContractArtifact, DecisionResultArtifact};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, HardConstraint,
    StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::evaluate_contract;
use antecedent_io_core::bridge_artifact::ScientificBridgeDrawsArtifact;
use antecedent_io_core::bridge_distribution::bridge_draws;
use antecedent_io_core::distribution_artifact::{
    DistributionArtifact, DistributionCalibration, DistributionIdentity, DistributionMetadata,
    DistributionProvenance, DistributionTrust, DrawAlignment,
};
use antecedent_io_core::observation_query_artifact::{
    ObservationQueryArtifactWire, ObservationQueryWire,
};
use antecedent_io_core::propagation_chain_record::{
    ChainCoefficientWire, ChainOpWire, ChainSourceWire, ChainStageWire, PropagationChainRecordWire,
};
use antecedent_io_core::quantity_wire::DistributionMeaningWire;

fn outcome(variable: &str, regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: "units".into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

/// Four equally likely rows of the endpoint `p`, the cost `q` and a safe action.
fn source_table() -> DistributionArtifact {
    let columns = [outcome("p", "do(a=1)"), outcome("q", "do(a=1)"), outcome("safe", "do(a=0)")];
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::InterventionalPredictive,
        &columns,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "enumerated".into(),
            provider_id: "exact-law".into(),
            rng_id: "deterministic_exact".into(),
            snapshot_id: "enumeration-1".into(),
            causal_contract_id: "checked-contract".into(),
        },
    )
    .unwrap();
    let p = [1.0, 3.0, 2.0, 0.0];
    let q = [4.0, 0.0, 2.0, 6.0];
    let mut draws = Vec::new();
    for i in 0..4 {
        draws.extend([p[i], q[i], 3.0]);
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [4, 3],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::Exact,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .unwrap()
}

/// A declared utility turns the endpoint `p` into `value = 10 p`, with an uncertain slope
/// whose draws depend on the seed.
fn utility_bridge() -> BridgeContract {
    BridgeContract {
        kind: BridgeKind::ScientificDecisionUtility,
        source: BridgeEndpoint {
            quantity: outcome("p", "do(a=1)"),
            class: QuantityClass::ScientificEndpoint,
        },
        target: BridgeEndpoint {
            quantity: outcome("value", "do(a=1)"),
            class: QuantityClass::DecisionUtility,
        },
        source_value: ValueKind::PredictiveLaw,
        target_value: ValueKind::PredictiveLaw,
        evidence_scope: BridgeScope { population_id: "target".into(), regime_id: "do(a=1)".into() },
        mapping: BridgeMapping::Affine { intercept: 0.0, slope: 10.0 },
        assumptions: vec![BridgeAssumption {
            id: "utility".into(),
            kind: AssumptionKind::UtilityMapping,
            statement: "each unit of the endpoint is worth ten units of value".into(),
            evidence: Some("utility:function-1".into()),
        }],
        transport: None,
        mapping_uncertainty: Some(MappingUncertainty {
            id: "utility-fit".into(),
            intercept_variance: 0.0,
            slope_variance: 0.25,
            intercept_slope_covariance: 0.0,
        }),
        source_dependence: Some(SourceMappingDependence::Independent {
            evidence: "utility-elicitation-disjoint".into(),
        }),
        estimand_bridge: None,
    }
}

fn decision() -> DecisionContract {
    DecisionContract {
        actions: vec![
            DecisionAction {
                id: "risky".into(),
                kind: ActionKind::Intervention,
                inputs: vec![outcome("value", "do(a=1)"), outcome("q", "do(a=1)")],
                utility: UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Input(1)),
            },
            DecisionAction {
                id: "safe".into(),
                kind: ActionKind::Policy,
                inputs: vec![outcome("safe", "do(a=0)")],
                utility: UtilityExpr::maximum(UtilityExpr::Input(0), UtilityExpr::Const(0.0)),
            },
        ],
        utility_units: "units".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![HardConstraint {
            id: "q-cap".into(),
            expr: UtilityExpr::Input(1),
            bound: 5.0,
            min_probability: 0.75,
            units: "units".into(),
            applies_to: vec!["risky".into()],
        }],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn embed(id: &str, bytes: &[u8]) -> BundleNode {
    let kind = detect_node_kind(bytes).expect("a bundle artifact kind");
    let identity = describe_artifact(kind, bytes).expect("the artifact decodes").identity;
    BundleNode::embedded(id, kind, &identity, bytes.to_vec())
}

fn edges(items: &[(&str, &str)]) -> Vec<(String, String)> {
    items.iter().map(|(a, b)| ((*a).to_owned(), (*b).to_owned())).collect()
}

fn run(nodes: Vec<BundleNode>, dependencies: &[(&str, &str)]) -> ConsumedBundle {
    let bundle =
        CompositionBundle::new(nodes, &edges(dependencies), &BundleLimits::default()).unwrap();
    let bytes = bundle.to_bytes("bundle", &BundleLimits::default()).unwrap();
    standard_consumer()
        .consume(&bytes, &BundleLimits::default(), bundle.identity(), &SuppliedSources::default())
        .unwrap()
}

fn failed_stage(consumed: &ConsumedBundle, id: &str) -> BundleStage {
    match &consumed.node(id).unwrap().status {
        NodeStatus::Failed { stage, .. } => *stage,
        other => panic!("node `{id}` is {other:?}"),
    }
}

struct Chain {
    distribution: Vec<u8>,
    bridge: Vec<u8>,
    contract: Vec<u8>,
    result: Vec<u8>,
}

/// The decision over the table a bridge of the given seed produced.
fn chain(seed: u64) -> Chain {
    let source = source_table();
    let bridged = bridge_draws(&utility_bridge(), &source, None, seed).expect("bridge");
    let contract = decision();
    let result = evaluate_contract(&contract, bridged.distribution()).expect("decision");
    Chain {
        distribution: source.to_bytes("distribution").unwrap(),
        bridge: ScientificBridgeDrawsArtifact::to_bytes(&bridged, "bridge").unwrap(),
        contract: DecisionContractArtifact::new(contract).unwrap().to_bytes("contract").unwrap(),
        result: DecisionResultArtifact::new(result, bridged.distribution())
            .to_bytes("result")
            .unwrap(),
    }
}

const EDGES: [(&str, &str); 3] =
    [("distribution", "bridge"), ("bridge", "result"), ("contract", "result")];

fn nodes(chain: &Chain) -> Vec<BundleNode> {
    vec![
        embed("distribution", &chain.distribution),
        embed("bridge", &chain.bridge),
        embed("contract", &chain.contract),
        embed("result", &chain.result),
    ]
}

#[test]
fn a_decision_over_a_bridge_of_draws_rests_on_the_bridge_as_on_a_distribution() {
    let chain = chain(3);
    assert_eq!(detect_node_kind(&chain.bridge), Some(NodeKind::ScientificBridge));
    let consumed = run(nodes(&chain), &EDGES);
    consumed.require_verified().unwrap();
    let bridge = consumed.node("bridge").unwrap();
    // The bridged table is aligned joint draws, and nothing is calibrated or licensed.
    assert_eq!(bridge.facts.get(FACT_LAW).map(String::as_str), Some(LAW_JOINT));
    assert_eq!(bridge.facts.get(FACT_TRUST).map(String::as_str), Some("unverified"));
    assert_eq!(consumed.value("bridge", "n_draws"), Some(4.0));
}

#[test]
fn a_decision_over_another_realization_of_the_bridge_is_refused() {
    let honest = chain(3);
    let other = chain(4);
    // The decision was computed over the draws of seed 3, but the bridge node is seed 4.
    let swapped = Chain {
        distribution: honest.distribution.clone(),
        bridge: other.bridge,
        contract: honest.contract.clone(),
        result: honest.result.clone(),
    };
    let consumed = run(nodes(&swapped), &EDGES);
    assert_eq!(consumed.node("bridge").unwrap().status, NodeStatus::Verified);
    assert_eq!(failed_stage(&consumed, "result"), BundleStage::TamperedQuantity);

    // A bridge of another source table is not the bridge of the distribution upstream of it.
    let mut shifted = source_table();
    let mut metadata = shifted.metadata().clone();
    metadata.identity.snapshot_id = "enumeration-2".into();
    let mut draws = shifted.draws().to_vec();
    draws[0] += 1.0;
    shifted = DistributionArtifact::new(metadata, draws).unwrap();
    let foreign = Chain {
        distribution: shifted.to_bytes("distribution").unwrap(),
        bridge: honest.bridge.clone(),
        contract: honest.contract.clone(),
        result: honest.result.clone(),
    };
    let consumed = run(nodes(&foreign), &EDGES);
    assert_eq!(failed_stage(&consumed, "bridge"), BundleStage::TamperedQuantity);
}

#[test]
fn an_observation_query_and_a_propagation_chain_are_replayed_and_publish_what_they_establish() {
    let ctx = ExecutionContext::for_tests(9);
    let query = ObservationQueryArtifactWire::seal(
        None,
        ObservationQueryWire::ConjugatePosterior {
            alpha_bits: 1.0_f64.to_bits(),
            beta_bits: 1.0_f64.to_bits(),
            successes: 7,
            failures: 3,
        },
        InferenceBasis::DeclaredModel,
        &ctx,
    )
    .unwrap();
    let query_bytes = query.export().unwrap();
    assert_eq!(detect_node_kind(&query_bytes), Some(NodeKind::ObservationQuery));
    let described = describe_artifact(NodeKind::ObservationQuery, &query_bytes).unwrap();
    assert_eq!(described.identity, query.identity);
    // A declared-model query answers a probability, not a law of draws.
    assert!(!described.facts.contains_key(FACT_LAW));

    let draws = [0.2, 0.4, 0.6, 0.8, 1.0, 1.2];
    let record = PropagationChainRecordWire::seal(
        PropagationChainRecordWire::draft(
            TableConstruction::JointDraws,
            ChainReading::Spread,
            "mapped",
            vec![ChainSourceWire::draws(
                "effect",
                antecedent_contract::propagation_chain::LawMeaning::Posterior,
                "table-1",
                &draws,
            )],
            vec![ChainStageWire::new(
                "mapped",
                StageKind::MeasurementMapping,
                &["effect"],
                ChainOpWire::affine(
                    ChainCoefficientWire::fixed(2.0),
                    ChainCoefficientWire::fixed(0.5),
                ),
            )],
        ),
        &ctx,
    )
    .unwrap();
    let chain_bytes = record.export().unwrap();
    assert_eq!(detect_node_kind(&chain_bytes), Some(NodeKind::PropagationChain));
    let described = describe_artifact(NodeKind::PropagationChain, &chain_bytes).unwrap();
    assert_eq!(described.identity, record.identity);
    // Mean of 2 x + 0.5 over the six draws: 2 * 0.7 + 0.5 = 1.9.
    let mean = described
        .values
        .iter()
        .find(|(name, _)| name == "answer.mean")
        .map(|(_, value)| *value)
        .unwrap();
    assert!((mean - 1.9).abs() < 1e-12, "{mean}");
    // The record keeps summaries and not the output column: it is no source of draws.
    assert!(!described.facts.contains_key(FACT_LAW));

    let consumed = run(vec![embed("query", &query_bytes), embed("chain", &chain_bytes)], &[]);
    consumed.require_verified().unwrap();

    // A resealed-looking edit of either record fails its own replay at a named stage.
    let mut edited = chain_bytes.clone();
    let middle = edited.len() / 2;
    edited[middle] ^= 0xff;
    assert!(describe_artifact(NodeKind::PropagationChain, &edited).is_err());
    let mut edited = query_bytes;
    let middle = edited.len() / 2;
    edited[middle] ^= 0xff;
    assert!(describe_artifact(NodeKind::ObservationQuery, &edited).is_err());
}
