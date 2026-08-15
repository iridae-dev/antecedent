//! Consumes the frozen causaleffect 1.3.15 transport-subset oracle.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    ContinuousDomain, GridSpec, ResponseFunctional, ResponseQuery, TransportQuery, VariableId,
};
use antecedent_graph::{Admg, DenseNodeId, SelectionDiagram};
use antecedent_identify::{TransportFormula, TransportIdentification, TransportIdentifier};

fn dense(id: VariableId) -> DenseNodeId {
    DenseNodeId::from_raw(id.raw())
}

fn mean_curve_query(treatment: VariableId, outcome: VariableId) -> TransportQuery {
    TransportQuery::new(
        ResponseQuery::new(ResponseFunctional::MeanCurve {
            outcome,
            treatment: ContinuousDomain::new(
                treatment,
                GridSpec::Values(Arc::from([0.0_f64, 1.0])),
            ),
        }),
        "source",
        "target",
        [treatment],
    )
}

#[test]
fn matches_frozen_causaleffect_supported_sid_subset() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/response/causaleffect_transport_subset/expected.json"
    ))
    .unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    let oracle_outputs = &fixture["reference"]["outputs"];
    assert_eq!(cases.len(), 2);

    for case in cases {
        let id = case["id"].as_str().unwrap();
        let ante = &case["antecedent"];
        let nodes = ante["nodes"].as_array().unwrap();
        let mut graph = Admg::with_variables(u32::try_from(nodes.len()).unwrap());
        for edge in ante["directed"].as_array().unwrap() {
            let from = nodes.iter().position(|n| n.as_str() == edge[0].as_str()).unwrap();
            let to = nodes.iter().position(|n| n.as_str() == edge[1].as_str()).unwrap();
            graph
                .insert_directed(
                    dense(VariableId::from_raw(u32::try_from(from).unwrap())),
                    dense(VariableId::from_raw(u32::try_from(to).unwrap())),
                )
                .unwrap();
        }
        let selection: Vec<VariableId> = ante["selection_targets"]
            .as_array()
            .unwrap()
            .iter()
            .map(|name| {
                let idx = nodes.iter().position(|n| n.as_str() == name.as_str()).unwrap();
                VariableId::from_raw(u32::try_from(idx).unwrap())
            })
            .collect();
        let treatment_name = ante["treatment"].as_str().unwrap();
        let outcome_name = ante["outcome"].as_str().unwrap();
        let treatment = VariableId::from_raw(
            u32::try_from(nodes.iter().position(|n| n.as_str() == Some(treatment_name)).unwrap())
                .unwrap(),
        );
        let outcome = VariableId::from_raw(
            u32::try_from(nodes.iter().position(|n| n.as_str() == Some(outcome_name)).unwrap())
                .unwrap(),
        );
        let diagram = SelectionDiagram::try_new(graph, selection).unwrap();
        let result = TransportIdentifier::new()
            .identify(&diagram, &mean_curve_query(treatment, outcome))
            .unwrap();

        let expected = &case["expected"];
        assert_eq!(
            expected["transportable"].as_bool().unwrap(),
            matches!(result, TransportIdentification::Transportable { .. }),
            "{id}"
        );
        match result {
            TransportIdentification::Transportable { formula, certificate } => {
                assert_eq!(certificate.rule.as_ref(), expected["rule"].as_str().unwrap(), "{id}");
                match (expected["formula_kind"].as_str().unwrap(), formula) {
                    ("direct", TransportFormula::Direct(_)) => {}
                    ("standardize", TransportFormula::Standardize { over, .. }) => {
                        let expected_over: Vec<_> = expected["over"]
                            .as_array()
                            .unwrap()
                            .iter()
                            .map(|v| v.as_str().unwrap())
                            .collect();
                        let got: Vec<_> = over
                            .iter()
                            .map(|vid| nodes[vid.as_usize()].as_str().unwrap())
                            .collect();
                        assert_eq!(got, expected_over, "{id}");
                    }
                    (kind, other) => panic!("{id}: unexpected formula {other:?} for kind {kind}"),
                }
            }
            TransportIdentification::NotCertified(cert) => {
                panic!("{id}: expected transportable, got NotCertified({})", cert.reason);
            }
        }
        assert_eq!(
            case["oracle"]["expression"].as_str().unwrap(),
            oracle_outputs[id].as_str().unwrap(),
            "{id}: fixture oracle expression drifted from reference.outputs"
        );
    }
}

/// The fixture above only exercises the frozen causaleffect oracle, which we only ever
/// invoked for cases inside Antecedent's certified sID subset (see the crate README).
/// It therefore never checks the fail-closed `NotCertified` path, which is the entire
/// point of the conservative design in `TransportIdentifier::identify`.
///
/// This test is NOT oracle-backed: fabricating a causaleffect expression for a case we
/// never ran through the R package would misrepresent a black-box parity claim. Instead
/// it constructs, directly in Rust, a selection diagram with a two-node c-component
/// (`Z <-> Y` with `Z` also a directed parent of `Y`) whose relevant selection node `Z`
/// reaches the outcome `Y`. This lies outside every implemented sound rule:
/// - not `transport.sid.direct` (selection reaches the outcome);
/// - not `transport.sid.standardize` (the standardizer `Z` has a bidirected neighbor,
///   so it is not an exogenous singleton district);
/// - not `transport.sid.s_admissible` (Z and Y share a district, so no subset of
///   standardizers m-separates the outcome from the selection node);
/// - not `transport.sid.singleton_c_components` (the district containing Z and Y has
///   two nodes, so `district_count < node_count`).
///
/// See `conformance/response/causaleffect_transport_subset/README.md` for what is and
/// is not oracle-backed in this file.
#[test]
fn multinode_c_component_outside_certified_subset_is_not_certified() {
    let nodes = ["X", "Z", "Y"];
    let x = VariableId::from_raw(0);
    let z = VariableId::from_raw(1);
    let y = VariableId::from_raw(2);

    let mut graph = Admg::with_variables(u32::try_from(nodes.len()).unwrap());
    graph.insert_directed(dense(x), dense(y)).unwrap();
    graph.insert_directed(dense(z), dense(y)).unwrap();
    graph.insert_bidirected(dense(z), dense(y)).unwrap();

    let diagram = SelectionDiagram::try_new(graph, [z]).unwrap();
    let result = TransportIdentifier::new().identify(&diagram, &mean_curve_query(x, y)).unwrap();

    match result {
        TransportIdentification::NotCertified(cert) => {
            assert_eq!(
                &*cert.reason, "transport.sid.multinode_c_component_not_implemented",
                "expected the general multi-node c-component refusal, got {}",
                cert.reason
            );
        }
        TransportIdentification::Transportable { .. } => {
            panic!(
                "diagram was constructed to fall outside every certified rule; \
                 a Transportable result here would be an unsound (over-)claim"
            );
        }
    }
}
