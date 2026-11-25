//! Site tests of the bundle recomputation question digests (`formal/identity_sites.toml`): a
//! changed part of each question (graph, terminal or producer name, producer program, producer
//! artifact, chain, contract, source law, seed) changes the compiled program identity.
//! Compilation reads a fit node's declaration but never refits it, so the fit nodes here are
//! declarations around opaque producer bytes.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use super::*;
use crate::analysis::propagation_chain::PropagationChain;
use crate::identity_laws::all_distinct;
use antecedent_contract::bridge::{
    AssumptionKind, BridgeAssumption, BridgeEndpoint, BridgeKind, BridgeMapping, BridgeScope,
    QuantityClass,
};
use antecedent_contract::propagation_chain::{
    ChainReading, LawMeaning, StageKind, TableConstruction,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_io::distribution_artifact::{
    DistributionCalibration, DistributionIdentity, DistributionMetadata, DistributionProvenance,
    DistributionTrust, DrawAlignment,
};
use antecedent_io::propagation_chain_record::{
    ChainCoefficientWire, ChainOpWire, ChainSourceWire, ChainStageWire,
};

/// A compiled chain and its sealed export.
fn chain(slope: f64, ctx: &ExecutionContext) -> (PropagationChainPlan, Vec<u8>) {
    let draft = PropagationChain::draft(
        TableConstruction::JointDraws,
        ChainReading::Spread,
        "output",
        vec![ChainSourceWire::draws(
            "source",
            LawMeaning::Sampling,
            "aligned-bundle",
            &[1.0, 2.0, 3.0],
        )],
        vec![ChainStageWire::new(
            "output",
            StageKind::DerivedQuantity,
            &["source"],
            ChainOpWire::affine(
                ChainCoefficientWire::fixed(slope),
                ChainCoefficientWire::fixed(0.0),
            ),
        )],
    );
    let plan = compile_propagation_chain(draft, ctx).expect("chain compiler");
    let bytes = plan.execute(ctx).expect("chain execution").export().expect("chain bytes");
    (plan, bytes)
}

fn graph(nodes: Vec<BundleNode>, edges: &[(&str, &str)]) -> CompositionBundle {
    let edges: Vec<(String, String)> =
        edges.iter().map(|(from, to)| ((*from).to_owned(), (*to).to_owned())).collect();
    CompositionBundle::new(nodes, &edges, &BundleLimits::default()).expect("bundle")
}

fn embedded(id: &str, kind: NodeKind, bytes: Vec<u8>) -> BundleNode {
    BundleNode::embedded(id, kind, &format!("{id}-identity"), bytes)
}

#[test]
fn site_composition_chain_bundle_question_binds_graph_terminal_and_chain() {
    let ctx = ExecutionContext::for_tests(23);
    let (first, bytes) = chain(2.0, &ctx);
    let (second, _) = chain(3.0, &ctx);
    let both = graph(
        vec![
            embedded("a", NodeKind::PropagationChain, bytes.clone()),
            embedded("b", NodeKind::PropagationChain, bytes.clone()),
        ],
        &[],
    );
    let one = graph(vec![embedded("a", NodeKind::PropagationChain, bytes)], &[]);
    let identity = |bundle: &CompositionBundle, terminal: &str, chain: &PropagationChainPlan| {
        compile_chain_bundle_recomputation(bundle.clone(), terminal, chain)
            .expect("bundle compiler")
            .lowered()
            .identity()
    };
    all_distinct(&[
        identity(&both, "a", &first),
        identity(&both, "b", &first),
        identity(&both, "a", &second),
        identity(&one, "a", &first),
    ]);
}

fn bridge_quantity(variable: &str, units: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: variable.into(),
        variable_name: variable.into(),
        role: QuantityRole::Outcome,
        units: units.into(),
        population_id: "adults".into(),
        regime_id: "do(t=1)".into(),
        horizon: 0,
        functional_id: "average_effect".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn bridge_source(shift: f64) -> DistributionArtifact {
    let quantities =
        [bridge_quantity("trial:biomarker", "mmol_per_l"), bridge_quantity("trial:cost", "usd")];
    let identity = DistributionIdentity::new(
        DistributionMeaningWire::PosteriorPredictive,
        &quantities,
        DrawAlignment::Joint,
        DistributionProvenance {
            source_id: "study-1".into(),
            provider_id: "posterior.provider".into(),
            rng_id: "seed:11".into(),
            snapshot_id: "snapshot-1".into(),
            causal_contract_id: "contract-1".into(),
        },
    )
    .expect("identity");
    let mut draws = Vec::new();
    for row in 0..16_u32 {
        draws.push(shift + f64::from(row) / 16.0);
        draws.push(0.2 + f64::from(row % 4) / 8.0);
    }
    DistributionArtifact::new(
        DistributionMetadata {
            version: 1,
            identity,
            axes: ["draw".into(), "quantity".into()],
            shape: [16, 2],
            weights: None,
            supported: None,
            calibration: DistributionCalibration::NotMeasured,
            trust: DistributionTrust::Unverified,
            legacy_posterior: None,
            legacy_bindings: None,
        },
        draws,
    )
    .expect("table")
}

fn bridge_contract(slope: f64) -> BridgeContract {
    BridgeContract {
        kind: BridgeKind::SurrogateTerminalOutcome,
        source: BridgeEndpoint {
            quantity: bridge_quantity("trial:biomarker", "mmol_per_l"),
            class: QuantityClass::SurrogateEndpoint,
        },
        target: BridgeEndpoint {
            quantity: bridge_quantity("trial:survival", "log_hazard"),
            class: QuantityClass::TerminalOutcome,
        },
        source_value: ValueKind::PredictiveLaw,
        target_value: ValueKind::PredictiveLaw,
        evidence_scope: BridgeScope { population_id: "adults".into(), regime_id: "do(t=1)".into() },
        mapping: BridgeMapping::Affine { intercept: 0.5, slope },
        assumptions: vec![BridgeAssumption {
            id: "validation".into(),
            kind: AssumptionKind::SurrogateValidation,
            statement: "the surrogate is validated at the trial level".into(),
            evidence: Some("meta-analysis:ckd-2024".into()),
        }],
        transport: None,
        mapping_uncertainty: None,
        source_dependence: None,
        estimand_bridge: None,
    }
}

#[test]
fn site_composition_bridge_bundle_question_binds_graph_terminal_source_contract_and_seed() {
    let identity = |terminal: &str, bytes: Vec<u8>, slope: f64, shift: f64, seed: u64| {
        let bundle = graph(vec![embedded(terminal, NodeKind::ScientificBridge, bytes)], &[]);
        compile_bridge_bundle_recomputation(
            bundle,
            terminal,
            bridge_contract(slope),
            bridge_source(shift),
            seed,
        )
        .expect("bridge bundle compiler")
        .lowered()
        .identity()
    };
    all_distinct(&[
        identity("terminal", vec![1], 2.0, 1.0, 41),
        identity("terminal-b", vec![1], 2.0, 1.0, 41),
        identity("terminal", vec![2], 2.0, 1.0, 41),
        identity("terminal", vec![1], 3.0, 1.0, 41),
        identity("terminal", vec![1], 2.0, 1.5, 41),
        identity("terminal", vec![1], 2.0, 1.0, 42),
    ]);
}

#[test]
fn site_composition_cyclic_posterior_bundle_question_binds_chain_program_and_artifact() {
    let ctx = ExecutionContext::for_tests(47);
    let (_, bytes) = chain(2.0, &ctx);
    let (_, other_bytes) = chain(3.0, &ctx);
    let terminal = graph(vec![embedded("terminal", NodeKind::PropagationChain, bytes)], &[]);
    let other = graph(vec![embedded("terminal", NodeKind::PropagationChain, other_bytes)], &[]);
    let identity = |bundle: &CompositionBundle, artifact: &[u8], program: u8| {
        compile_cyclic_posterior_bundle_recomputation(
            bundle.clone(),
            "terminal",
            artifact.to_vec(),
            SemanticDigest::from_bytes([program; 32]),
            &ctx,
        )
        .expect("compiles")
        .lowered()
        .identity()
    };
    all_distinct(&[
        identity(&terminal, b"producer", 1),
        identity(&terminal, b"producer", 2),
        identity(&terminal, b"other-producer", 1),
        identity(&other, b"producer", 1),
    ]);
}

/// A producer node `fit` of `kind`, directly upstream of the embedded chain `terminal`.
fn fit_graph(
    fit: &str,
    kind: NodeKind,
    fit_bytes: Vec<u8>,
    chain_bytes: Vec<u8>,
) -> CompositionBundle {
    graph(
        vec![
            embedded(fit, kind, fit_bytes),
            embedded("terminal", NodeKind::PropagationChain, chain_bytes),
        ],
        &[(fit, "terminal")],
    )
}

/// Every fit-node family's question over: the base, a renamed producer node, another producer
/// program, another producer artifact and another terminal chain. All five must differ.
fn fit_node_family(
    kind: NodeKind,
    wire: impl Fn(u8, &[u8]) -> Vec<u8>,
    identity: impl Fn(CompositionBundle, &str) -> SemanticDigest,
) {
    let ctx = ExecutionContext::for_tests(23);
    let (_, bytes) = chain(2.0, &ctx);
    let (_, other) = chain(3.0, &ctx);
    all_distinct(&[
        identity(fit_graph("fit", kind, wire(1, b"producer"), bytes.clone()), "fit"),
        identity(fit_graph("fit-b", kind, wire(1, b"producer"), bytes.clone()), "fit-b"),
        identity(fit_graph("fit", kind, wire(2, b"producer"), bytes.clone()), "fit"),
        identity(fit_graph("fit", kind, wire(1, b"other-producer"), bytes), "fit"),
        identity(fit_graph("fit", kind, wire(1, b"producer"), other), "fit"),
    ]);
}

#[test]
fn site_composition_cyclic_posterior_node_bundle_question_binds_its_parts() {
    let ctx = ExecutionContext::for_tests(47);
    fit_node_family(
        NodeKind::CyclicPosteriorFit,
        |program, artifact| {
            antecedent_io::to_cbor(&CyclicPosteriorFitWire {
                version: 2,
                required_feature: CYCLIC_POSTERIOR_FIT_FEATURE_V2.into(),
                producer_program: [program; 32],
                artifact: artifact.to_vec(),
                replay_seed: Some(5),
            })
            .expect("fit wire")
        },
        |bundle, fit| {
            compile_cyclic_posterior_node_bundle_recomputation(bundle, fit, "terminal", &ctx)
                .expect("compiles")
                .lowered()
                .identity()
        },
    );
}

#[test]
fn site_composition_trial_ipw_node_bundle_question_binds_its_parts() {
    let ctx = ExecutionContext::for_tests(31);
    fit_node_family(
        NodeKind::TrialIpwTransportFit,
        |program, artifact| {
            antecedent_io::to_cbor(&TrialIpwFitWire {
                version: 2,
                required_feature: TRIAL_IPW_FIT_FEATURE_V2.into(),
                producer_program: [program; 32],
                artifact: artifact.to_vec(),
                replay_seed: Some(5),
            })
            .expect("fit wire")
        },
        |bundle, fit| {
            compile_trial_ipw_node_bundle_recomputation(bundle, fit, "terminal", &ctx)
                .expect("compiles")
                .lowered()
                .identity()
        },
    );
}

#[test]
fn site_composition_graph_imputation_transport_node_bundle_question_binds_its_parts() {
    let ctx = ExecutionContext::for_tests(37);
    fit_node_family(
        NodeKind::GraphImputationTransportFit,
        |program, artifact| {
            antecedent_io::to_cbor(&GraphImputationTransportFitWire {
                version: 2,
                required_feature: GRAPH_IMPUTATION_TRANSPORT_FIT_FEATURE_V2.into(),
                producer_program: [program; 32],
                artifact: artifact.to_vec(),
                replay_seed: Some(5),
            })
            .expect("fit wire")
        },
        |bundle, fit| {
            compile_graph_imputation_transport_node_bundle_recomputation(
                bundle, fit, "terminal", &ctx,
            )
            .expect("compiles")
            .lowered()
            .identity()
        },
    );
}

#[test]
fn site_composition_joint_transport_effect_node_bundle_question_binds_its_parts() {
    let ctx = ExecutionContext::for_tests(41);
    fit_node_family(
        NodeKind::JointTransportEffectFit,
        |program, artifact| {
            antecedent_io::to_cbor(&JointTransportEffectFitWire {
                version: 2,
                required_feature: JOINT_TRANSPORT_EFFECT_FIT_FEATURE_V2.into(),
                producer_program: [program; 32],
                artifact: artifact.to_vec(),
                replay_seed: Some(5),
            })
            .expect("fit wire")
        },
        |bundle, fit| {
            compile_joint_transport_effect_node_bundle_recomputation(bundle, fit, "terminal", &ctx)
                .expect("compiles")
                .lowered()
                .identity()
        },
    );
}
