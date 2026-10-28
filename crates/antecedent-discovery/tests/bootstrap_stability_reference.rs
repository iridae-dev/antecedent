//! Selection-frequency tabulation against definition-level exact counting.
//!
//! `oracle_data/bootstrap_stability_reference.py` (frozen in the JSON beside it) counts, with
//! rational arithmetic, how often each node pair is adjacent, carries each pair of endpoint
//! marks and how often each whole structure is returned, over seeded random structures with
//! skewed and fully tied counts, a reference no resample returned, empty structures and the
//! acceptance boundary (at least two successes and at least half of the requested). The product
//! must reproduce every frequency, the reference frequency, the failed mass and the refusals.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::collections::BTreeMap;

use antecedent_core::{NodeRef, VariableId};
use antecedent_discovery::StructureStability;
use antecedent_discovery::selective::{DiscoveredStructure, Mark, StructureNode};
use antecedent_graph::{DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag};
use serde_json::Value;

fn endpoint(name: &str) -> Endpoint {
    match name {
        "tail" => Endpoint::Tail,
        "arrow" => Endpoint::Arrow,
        "circle" => Endpoint::Circle,
        other => panic!("unknown mark {other}"),
    }
}

fn mark(name: &str) -> Mark {
    match name {
        "tail" => Mark::Tail,
        "arrow" => Mark::Arrow,
        "circle" => Mark::Circle,
        other => panic!("unknown mark {other}"),
    }
}

fn node(v: u64) -> StructureNode {
    StructureNode::from_node_ref(NodeRef::Static(VariableId::from_raw(u32::try_from(v).unwrap())))
}

fn structure(edges: &Value) -> DiscoveredStructure {
    let mut graph = Pag::empty();
    let ids: Vec<DenseNodeId> = (0..4_u32)
        .map(|v| graph.add_node(NodeRef::Static(VariableId::from_raw(v))).unwrap())
        .collect();
    for e in edges.as_array().unwrap() {
        let a = ids[e[0].as_u64().unwrap() as usize];
        let b = ids[e[1].as_u64().unwrap() as usize];
        graph
            .insert_marked(MarkedEdge {
                a,
                b,
                at_a: endpoint(e[2].as_str().unwrap()),
                at_b: endpoint(e[3].as_str().unwrap()),
                middle: MiddleMark::Empty,
            })
            .unwrap();
    }
    DiscoveredStructure::from_pag(&graph)
}

fn fraction(v: &Value) -> f64 {
    v[0].as_f64().unwrap() / v[1].as_f64().unwrap()
}

type Signature = Vec<(u64, u64, Mark, Mark)>;

fn signature_of(edges: &Value) -> Signature {
    let mut out: Signature = edges
        .as_array()
        .unwrap()
        .iter()
        .map(|e| {
            (
                e[0].as_u64().unwrap(),
                e[1].as_u64().unwrap(),
                mark(e[2].as_str().unwrap()),
                mark(e[3].as_str().unwrap()),
            )
        })
        .collect();
    out.sort();
    out
}

#[test]
fn tabulated_frequencies_equal_exact_counting_in_every_regime() {
    let cases: Vec<Value> = serde_json::from_str::<Value>(include_str!(
        "oracle_data/bootstrap_stability_reference.json"
    ))
    .unwrap()
    .as_array()
    .unwrap()
    .clone();
    assert_eq!(cases.len(), 11);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let requested = u32::try_from(case["requested"].as_u64().unwrap()).unwrap();
        let reference = structure(&case["reference"]);
        let resamples: Vec<DiscoveredStructure> =
            case["resamples"].as_array().unwrap().iter().map(structure).collect();
        let result = StructureStability::tabulate(reference, &resamples, requested);
        if !case["ok"].as_bool().unwrap() {
            assert!(result.is_err(), "{name}: the tabulation must be refused");
            continue;
        }
        let table = result.unwrap_or_else(|e| panic!("{name}: {e}"));
        let succeeded = u32::try_from(case["succeeded"].as_u64().unwrap()).unwrap();
        assert_eq!(
            (table.requested, table.succeeded, table.failed),
            (requested, succeeded, requested - succeeded),
            "{name}: counts"
        );
        assert!(
            (table.reference_frequency - fraction(&case["reference_frequency"])).abs() < 1e-14,
            "{name}: reference frequency"
        );

        let expected_edges = case["edges"].as_array().unwrap();
        assert_eq!(table.edges.len(), expected_edges.len(), "{name}: node pairs");
        for (got, want) in table.edges.iter().zip(expected_edges) {
            let (a, b) = (want["a"].as_u64().unwrap(), want["b"].as_u64().unwrap());
            assert_eq!((got.a, got.b), (node(a), node(b)), "{name}: pair order");
            assert!(
                (got.frequency - fraction(&want["frequency"])).abs() < 1e-14,
                "{name}: adjacency {a}-{b}"
            );
            assert_eq!(got.in_reference, want["in_reference"].as_bool().unwrap(), "{name}");
            let mut wanted: Vec<(Mark, Mark, f64)> = want["marks"]
                .as_array()
                .unwrap()
                .iter()
                .map(|m| {
                    (mark(m[0].as_str().unwrap()), mark(m[1].as_str().unwrap()), fraction(&m[2]))
                })
                .collect();
            // Most frequent first, ties by the mark pair order.
            wanted.sort_by(|x, y| y.2.total_cmp(&x.2).then((x.0, x.1).cmp(&(y.0, y.1))));
            assert_eq!(got.marks.len(), wanted.len(), "{name}: marks on {a}-{b}");
            for (g, w) in got.marks.iter().zip(&wanted) {
                assert_eq!((g.at_a, g.at_b), (w.0, w.1), "{name}: mark order on {a}-{b}");
                assert!((g.frequency - w.2).abs() < 1e-14, "{name}: mark frequency on {a}-{b}");
            }
        }

        let mut expected: BTreeMap<Signature, u64> = BTreeMap::new();
        for s in case["structures"].as_array().unwrap() {
            expected.insert(signature_of(&s["edges"]), s["count"].as_u64().unwrap());
        }
        assert_eq!(table.structures.len(), expected.len(), "{name}: distinct structures");
        let mut previous = u32::MAX;
        for selection in &table.structures {
            let sig: Signature = selection
                .structure
                .edges()
                .iter()
                .map(|e| (u64::from(e.a.variable), u64::from(e.b.variable), e.at_a, e.at_b))
                .collect();
            let mut sig = sig;
            sig.sort();
            assert_eq!(expected.get(&sig).copied(), Some(u64::from(selection.count)), "{name}");
            assert!(selection.count <= previous, "{name}: most frequent first");
            previous = selection.count;
            assert!(
                (selection.frequency - f64::from(selection.count) / f64::from(succeeded)).abs()
                    < 1e-14
            );
        }
        let mass: f64 = table.structures.iter().map(|s| s.frequency).sum();
        assert!((mass - 1.0).abs() < 1e-12, "{name}: structure mass");
    }
}
