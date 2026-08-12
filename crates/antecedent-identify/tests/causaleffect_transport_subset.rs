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
