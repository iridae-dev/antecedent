//! Source evidence over an actual native analysis, with the evidence it carries derived by hand.
//!
//! The data are exactly linear, y = 1 + 2 t, so the native mean curve at t = -0.5 and t = 0.5 is
//! 1 + 2 * (-0.5) = 0 and 1 + 2 * 0.5 = 2. Call the two source coordinates c0 (t = -0.5) and
//! c1 (t = 0.5). Three actions are projected onto the source:
//!
//! - `a0`: inputs [c0, c1], utility x0 - x1 = 0 - 2 = -2; contributors c0 then c1.
//! - `a1`: inputs [c1, c0], utility x1 - x0 = 0 - 2 = -2; contributors in input order: c1 then c0.
//! - `a2`: inputs [c1], utility 2 * x0 = 4; contributors c1 alone.
//!
//! So the contributor counts are 2, 2 and 1 and `a2` is uniquely optimal. The provenance chain
//! behind the claim has the links causal contract, snapshot, native program, source artifact and
//! (once projected) decision contract and source projection: 5 links before projection and 7
//! after, with the projection depending on exactly the source artifact and the decision contract.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeSet;

use antecedent::analysis::source_evidence::SourceEvidence;
use antecedent::{RefuteSuite, Study};
use antecedent_contract::{CompositionStage, ResponseCoordinateLabels, ScientificQuantity};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional, ResponseQuery,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_design::decision_artifact::contract_to_json;
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::{
    MeanEstimate, MeanSource, Verdict, evaluate_contract_on_means,
};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_io::response_wire::{ResponseIdentificationWire, ResponseValueWire};
use serde_json::Value;

fn source() -> SourceEvidence {
    let treatment: Vec<f64> = (0_u32..240).map(|i| (f64::from(i) / 17.).sin()).collect();
    let outcome: Vec<f64> = treatment.iter().map(|t| 1. + 2. * t).collect();
    let data =
        TabularData::from_f64_columns([("t", treatment.as_slice()), ("y", outcome.as_slice())])
            .unwrap();
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let ctx = ExecutionContext::for_tests(51);
    let query = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(vec![-0.5, 0.5].into()),
        ),
    });
    let prepared = Study::tabular(data.clone())
        .graph(graph)
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let bytes = prepared
        .encode_contracted_result_with_quantity_labels(
            &result,
            "source-evidence-exact",
            &ctx,
            Some(&ResponseCoordinateLabels {
                outcome_units: "score",
                population_id: "target",
                transform_id: "identity",
            }),
        )
        .unwrap();
    SourceEvidence::from_result(&bytes).unwrap()
}

fn contract(c: &[ScientificQuantity]) -> DecisionContract {
    let action = |id: &str, inputs: Vec<ScientificQuantity>, utility: UtilityExpr| DecisionAction {
        id: id.into(),
        kind: ActionKind::Intervention,
        inputs,
        utility,
    };
    DecisionContract {
        actions: vec![
            action(
                "a0",
                vec![c[0].clone(), c[1].clone()],
                UtilityExpr::difference(UtilityExpr::Input(0), UtilityExpr::Input(1)),
            ),
            action(
                "a1",
                vec![c[1].clone(), c[0].clone()],
                UtilityExpr::difference(UtilityExpr::Input(1), UtilityExpr::Input(0)),
            ),
            action(
                "a2",
                vec![c[1].clone()],
                UtilityExpr::product(UtilityExpr::Input(0), UtilityExpr::Const(2.0)),
            ),
        ],
        utility_units: "score".into(),
        criterion: DecisionCriterion::PosteriorExpectedUtility,
        constraints: vec![],
        target_population: "target".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::ReportOnly,
    }
}

fn wire(q: &ScientificQuantity) -> Value {
    serde_json::to_value(ScientificQuantityWire::from(q)).unwrap()
}

fn near(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

#[test]
fn the_native_means_and_projected_action_utilities_follow_the_hand_derivation() {
    let source = source();
    let c = source.coordinates().to_vec();
    assert_eq!(c.len(), 2);
    let original = antecedent_io::consume_analysis_result(source.original_bytes()).unwrap();
    let ResponseIdentificationWire::PointIdentified(ResponseValueWire::Surface { mean, .. }) =
        &original.body.response.as_ref().unwrap().estimate
    else {
        panic!("a mean curve");
    };
    assert!(near(mean[0], 0.0), "{}", mean[0]);
    assert!(near(mean[1], 2.0), "{}", mean[1]);

    let contract = contract(&c);
    let snapshot =
        antecedent_core::to_hex(&original.contract.as_ref().unwrap().identities.data_snapshot);
    let means = MeanSource {
        coordinates: c.clone(),
        means: mean.clone(),
        provider_id: "native-source".into(),
        snapshot_id: snapshot.clone(),
        causal_contract_id: "source-contract".into(),
        rng_id: "native:seed:51".into(),
        estimate: MeanEstimate::Frequentist,
    };
    let result = evaluate_contract_on_means(&contract, &means).unwrap();
    for (action, want) in result.actions.iter().zip([-2.0, -2.0, 4.0]) {
        assert!(near(action.expected_utility, want), "{} {}", action.id, action.expected_utility);
    }
    assert_eq!(result.verdict, Verdict::UniquelyOptimal("a2".into()));

    // The source vouches for exactly these means, and for no other bits.
    let projected = source
        .project(&contract_to_json(&contract).unwrap(), &["a0".into(), "a1".into(), "a2".into()])
        .unwrap();
    projected.require_mean_source(&means).unwrap();
    projected.require_quantity_binding(&c, &snapshot).unwrap();
    let mut shifted = means.clone();
    shifted.means[1] += 1e-9;
    assert_eq!(
        projected.require_mean_source(&shifted).unwrap_err().detail,
        antecedent_core::reason_details::source_evidence::POINT_BINDING_MISMATCH
    );
}

#[test]
fn contributors_are_the_coordinates_each_utility_reads_in_input_order() {
    let source = source();
    let c = source.coordinates().to_vec();
    let projected = source
        .project(
            &contract_to_json(&contract(&c)).unwrap(),
            &["a0".into(), "a1".into(), "a2".into()],
        )
        .unwrap();
    let summary = projected.summary();
    let contributors = summary["action_contributors"].as_array().unwrap();
    let got: Vec<(String, Vec<Value>)> = contributors
        .iter()
        .map(|entry| {
            (
                entry["action_id"].as_str().unwrap().to_owned(),
                entry["source_coordinates"].as_array().unwrap().clone(),
            )
        })
        .collect();
    assert_eq!(
        got,
        vec![
            ("a0".to_owned(), vec![wire(&c[0]), wire(&c[1])]),
            ("a1".to_owned(), vec![wire(&c[1]), wire(&c[0])]),
            ("a2".to_owned(), vec![wire(&c[1])]),
        ]
    );
}

#[test]
fn the_provenance_chain_has_the_hand_listed_links_before_and_after_projection() {
    let source = source();
    let c = source.coordinates().to_vec();
    let before = source.provenance_chain().unwrap();
    assert_eq!(before.links().len(), 5);
    let stages: BTreeSet<CompositionStage> =
        before.stages_behind("claim").unwrap().into_iter().collect();
    assert_eq!(
        stages,
        BTreeSet::from([
            CompositionStage::Claim,
            CompositionStage::Evidence,
            CompositionStage::CausalContract,
            CompositionStage::Data
        ])
    );

    let projected =
        source.project(&contract_to_json(&contract(&c)).unwrap(), &["a0".into()]).unwrap();
    let after = projected.provenance_chain().unwrap();
    assert_eq!(after.links().len(), 7);
    let stages: BTreeSet<CompositionStage> =
        after.stages_behind("claim").unwrap().into_iter().collect();
    assert_eq!(
        stages,
        BTreeSet::from([
            CompositionStage::Claim,
            CompositionStage::Transformation,
            CompositionStage::DecisionContract,
            CompositionStage::Evidence,
            CompositionStage::CausalContract,
            CompositionStage::Data
        ])
    );
    let projection =
        after.links().iter().find(|l| l.stage == CompositionStage::Transformation).unwrap();
    assert_eq!(projection.parents.len(), 2);
    let parent_stages: BTreeSet<CompositionStage> = projection
        .parents
        .iter()
        .map(|p| after.links().iter().find(|l| &l.id == p).unwrap().stage)
        .collect();
    assert_eq!(
        parent_stages,
        BTreeSet::from([CompositionStage::Evidence, CompositionStage::DecisionContract])
    );
}
