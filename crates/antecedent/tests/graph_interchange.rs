//! graph interchange conformance (DOT/JSON/GML/NetworkX).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::PathBuf;

use antecedent::io::{
    dag_from_dot, dag_from_gml, dag_from_json, dag_from_networkx_node_link, dag_to_dot, dag_to_gml,
    dag_to_json, dag_to_networkx_node_link,
};
use causal_graph::DenseNodeId;
use serde_json::Value;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/interchange/graph_dot_json")
}

fn gml_fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/interchange/graph_gml_networkx")
}

#[test]
fn conformance_dot_json_round_trip() {
    let raw = fs::read_to_string(fixture_dir().join("expected.json")).unwrap();
    let v: Value = serde_json::from_str(&raw).unwrap();
    let dot = v["dot"].as_str().unwrap();
    let expected_n = usize::try_from(v["expected_node_count"].as_u64().unwrap()).unwrap();
    let dag = dag_from_dot(dot).unwrap();
    assert_eq!(dag.node_count(), expected_n);
    assert!(dag.reaches(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)));

    let json = serde_json::to_string(&v["json"]).unwrap();
    let dag_j = dag_from_json(&json).unwrap();
    assert_eq!(dag_j.node_count(), expected_n);
    assert!(dag_j.reaches(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)));

    let names = vec!["a".into(), "b".into(), "c".into()];
    let out_dot = dag_to_dot(&dag, Some(&names)).unwrap();
    let again = dag_from_dot(&out_dot).unwrap();
    assert_eq!(again.node_count(), expected_n);

    let out_json = dag_to_json(&dag, Some(&names)).unwrap();
    let again_j = dag_from_json(&out_json).unwrap();
    assert_eq!(again_j.node_count(), expected_n);
}

#[test]
fn conformance_gml_networkx_round_trip() {
    let raw = fs::read_to_string(gml_fixture_dir().join("expected.json")).unwrap();
    let v: Value = serde_json::from_str(&raw).unwrap();
    let gml = v["gml"].as_str().unwrap();
    let expected_n = usize::try_from(v["expected_node_count"].as_u64().unwrap()).unwrap();
    let dag = dag_from_gml(gml).unwrap();
    assert_eq!(dag.node_count(), expected_n);

    let nx = serde_json::to_string(&v["networkx_node_link"]).unwrap();
    let dag_nx = dag_from_networkx_node_link(&nx).unwrap();
    assert_eq!(dag_nx.node_count(), expected_n);

    let names = vec!["Z".into(), "X".into(), "Y".into()];
    let out_gml = dag_to_gml(&dag, Some(&names)).unwrap();
    assert_eq!(dag_from_gml(&out_gml).unwrap().node_count(), expected_n);
    let out_nx = dag_to_networkx_node_link(&dag, Some(&names)).unwrap();
    assert_eq!(dag_from_networkx_node_link(&out_nx).unwrap().node_count(), expected_n);
}
