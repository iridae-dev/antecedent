//! Every case of `conformance/graph/latent_projection`, compared as a complete edge set.
//!
//! The fixture lists the projection of each latent DAG over the observed variables (a directed
//! edge for a latent-only directed path, a bidirected edge for a latent common cause). The
//! projected graph's nodes are the observed variables in the listed order; the comparison is on
//! every ordered observed pair, so no extra and no missing edge passes.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_graph::{Dag, DenseNodeId, latent_project};

fn names(value: &serde_json::Value) -> Vec<&str> {
    value.as_array().unwrap().iter().map(|e| e.as_str().unwrap()).collect()
}

#[test]
fn latent_projection_claims_02_every_fixture_case_matches_the_full_edge_set() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/graph/latent_projection/expected.json"
    ))
    .unwrap();
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 3);
    for case in cases {
        let id = case["id"].as_str().unwrap();
        let mut ids: BTreeMap<&str, u32> = BTreeMap::new();
        let dag_edges: Vec<(&str, &str)> =
            names(&case["dag"]).iter().map(|e| e.split_once("->").unwrap()).collect();
        for (a, b) in &dag_edges {
            for name in [a, b] {
                let next = u32::try_from(ids.len()).unwrap();
                ids.entry(name).or_insert(next);
            }
        }
        let mut dag = Dag::with_variables(u32::try_from(ids.len()).unwrap());
        for (a, b) in &dag_edges {
            dag.insert_directed(DenseNodeId::from_raw(ids[a]), DenseNodeId::from_raw(ids[b]))
                .unwrap();
        }
        let observed = names(&case["observed"]);
        let observed_ids: Vec<DenseNodeId> =
            observed.iter().map(|name| DenseNodeId::from_raw(ids[name])).collect();
        let projected = latent_project(&dag, &observed_ids).unwrap();
        assert_eq!(projected.node_count(), observed.len(), "{id}: observed nodes only");

        let mut directed = Vec::new();
        let mut bidirected = Vec::new();
        for edge in names(&case["projection"]) {
            if let Some((a, b)) = edge.split_once("<->") {
                bidirected.push((a, b));
            } else {
                directed.push(edge.split_once("->").unwrap());
            }
        }
        let position = |name: &str| observed.iter().position(|o| *o == name).unwrap();
        for (i, from) in observed.iter().enumerate() {
            for (j, to) in observed.iter().enumerate() {
                let node = |k: usize| DenseNodeId::from_raw(u32::try_from(k).unwrap());
                let has_directed = projected.children(node(i)).contains(&node(j));
                assert_eq!(
                    has_directed,
                    directed.contains(&(*from, *to)),
                    "{id}: directed {from}->{to}"
                );
                let has_bidirected = projected.bidirected_neighbors(node(i)).contains(&node(j));
                assert_eq!(
                    has_bidirected,
                    bidirected
                        .iter()
                        .any(|&(a, b)| (a == *from && b == *to) || (a == *to && b == *from)),
                    "{id}: bidirected {from}<->{to}"
                );
            }
        }
        // Names resolve to observed positions, so the declared edges are all checked above.
        for (a, b) in directed.iter().chain(&bidirected) {
            assert!(position(a) != position(b), "{id}");
        }
    }
}
