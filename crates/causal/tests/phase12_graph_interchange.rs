//! Phase 12 graph interchange conformance.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::fs;
use std::path::PathBuf;

use causal::{dag_from_dot, dag_from_json, dag_to_dot, dag_to_json};
use causal_graph::DenseNodeId;
use serde_json::Value;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/phase12/graph_dot_json")
}

#[test]
fn conformance_dot_json_round_trip() {
    let raw = fs::read_to_string(fixture_dir().join("expected.json")).unwrap();
    let v: Value = serde_json::from_str(&raw).unwrap();
    let dot = v["dot"].as_str().unwrap();
    let expected_n = v["expected_node_count"].as_u64().unwrap() as usize;
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
