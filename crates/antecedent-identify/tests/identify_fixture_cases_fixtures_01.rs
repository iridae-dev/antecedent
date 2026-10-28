//! Every case of the frozen `id_hedge` and `efficient_adjustment` fixtures, compared with the
//! product.
//!
//! Each fixture case is an edge list over named nodes. The fixtures' expected values are
//! re-derived by their retained references (finite structural-causal-model enumeration for the
//! identification status, exhaustive valid-set enumeration with exact asymptotic variances for
//! the efficient adjustment set); these tests read every case from the fixture, build the graph
//! from its edge strings, run the shipped identifier and compare with the stored value.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::{
    AverageEffectQuery, CausalQuery, Intervention, InterventionalDistributionQuery, Value,
    VariableId,
};
use antecedent_graph::{Admg, Dag, DenseNodeId};
use antecedent_identify::{
    EfficientBackdoorIdentifier, IdIdentifier, IdentificationStatus, IdentificationWorkspace,
};

fn strings(value: &serde_json::Value) -> Vec<&str> {
    value.as_array().unwrap().iter().map(|e| e.as_str().unwrap()).collect()
}

/// Nodes in sorted-name order so that the dense index is reproducible from the fixture alone.
fn index_of(edges: &[&str]) -> BTreeMap<String, u32> {
    let mut names: Vec<String> = edges
        .iter()
        .flat_map(|e| e.split("<->").flat_map(|s| s.split("->")).map(str::to_owned))
        .collect();
    names.sort();
    names.dedup();
    names.into_iter().enumerate().map(|(i, n)| (n, u32::try_from(i).unwrap())).collect()
}

#[test]
fn id_hedge_fixture_every_case_status_and_hedge_certificate_match() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/identify/id_hedge/expected.json"
    ))
    .unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    let id = IdIdentifier::new();
    for case in cases {
        let name = case["id"].as_str().unwrap();
        let edges = strings(&case["edges"]);
        let ids = index_of(&edges);
        let mut graph = Admg::with_variables(u32::try_from(ids.len()).unwrap());
        for edge in &edges {
            if let Some((a, b)) = edge.split_once("<->") {
                graph
                    .insert_bidirected(DenseNodeId::from_raw(ids[a]), DenseNodeId::from_raw(ids[b]))
                    .unwrap();
            } else {
                let (a, b) = edge.split_once("->").unwrap();
                graph
                    .insert_directed(DenseNodeId::from_raw(ids[a]), DenseNodeId::from_raw(ids[b]))
                    .unwrap();
            }
        }
        let prepared = id.prepare(&graph).unwrap();
        let query = CausalQuery::Distribution(InterventionalDistributionQuery::new(
            VariableId::from_raw(ids["Y"]),
            vec![Intervention::set(VariableId::from_raw(ids["T"]), Value::f64(1.0))],
        ));
        let result = id.identify(&prepared, &query, &mut IdentificationWorkspace::default()).unwrap();
        let expected = case["status"].as_str().unwrap();
        match expected {
            "identified" => assert_eq!(
                result.status,
                IdentificationStatus::NonparametricallyIdentified,
                "{name}"
            ),
            "not_identified" => {
                assert_eq!(result.status, IdentificationStatus::NotIdentified, "{name}");
                assert_eq!(case["certificate"].as_str(), Some("hedge"), "{name}");
                assert!(
                    result.diagnostics.iter().any(|d| d.code.as_ref() == "identify.hedge"),
                    "{name}: a hedge certificate is reported"
                );
            }
            other => panic!("{name}: unreadable status {other}"),
        }
    }
}

#[test]
fn efficient_adjustment_fixture_every_case_reports_the_stored_optimal_set() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/identify/efficient_adjustment/expected.json"
    ))
    .unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    for case in cases {
        let name = case["id"].as_str().unwrap();
        let edges = strings(&case["edges"]);
        let ids = index_of(&edges);
        let mut dag = Dag::with_variables(u32::try_from(ids.len()).unwrap());
        for edge in &edges {
            let (a, b) = edge.split_once("->").unwrap();
            dag.insert_directed(DenseNodeId::from_raw(ids[a]), DenseNodeId::from_raw(ids[b]))
                .unwrap();
        }
        let treatment = case["treatment"].as_str().unwrap_or("T");
        let outcome = case["outcome"].as_str().unwrap_or("Y");
        let id = EfficientBackdoorIdentifier::new();
        let prepared = id.prepare(&dag).unwrap();
        let query = CausalQuery::average_effect(AverageEffectQuery::binary_ate(
            VariableId::from_raw(ids[treatment]),
            VariableId::from_raw(ids[outcome]),
        ));
        let result =
            id.identify(&prepared, &query, &mut IdentificationWorkspace::default()).unwrap();
        assert_eq!(result.status, IdentificationStatus::NonparametricallyIdentified, "{name}");
        assert_eq!(result.estimands.len(), 1, "{name}");
        let mut reported: Vec<u32> =
            result.estimands[0].adjustment_set.iter().map(|v| v.raw()).collect();
        reported.sort_unstable();
        let mut expected: Vec<u32> =
            strings(&case["optimal_set"]).iter().map(|n| ids[*n]).collect();
        expected.sort_unstable();
        assert_eq!(reported, expected, "{name}: optimal adjustment set");
    }
}
