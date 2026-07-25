//! Frozen causal-learn FCI/GES and Python DirectLiNGAM parity matrix.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::collections::{BTreeMap, BTreeSet};
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_discovery::{DirectLingam, DiscoveryConstraints, DiscoveryWorkspace, Fci, Ges};
use antecedent_graph::{DenseNodeId, Endpoint};
use serde_json::Value as JsonValue;

type MarkedKey = (String, String, String, String);

fn fixture() -> JsonValue {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/discovery/static_oracle_matrix/expected.json");
    serde_json::from_str(&fs::read_to_string(path).expect("static-discovery fixture"))
        .expect("parse static-discovery fixture")
}

fn endpoint(value: Endpoint) -> String {
    match value {
        Endpoint::Tail => "TAIL",
        Endpoint::Arrow => "ARROW",
        Endpoint::Circle => "CIRCLE",
        Endpoint::Conflict => "CONFLICT",
    }
    .to_owned()
}

fn canonical_marked(a: &str, at_a: String, b: &str, at_b: String) -> MarkedKey {
    if a <= b {
        (a.to_owned(), at_a, b.to_owned(), at_b)
    } else {
        (b.to_owned(), at_b, a.to_owned(), at_a)
    }
}

fn reference_edges(value: &JsonValue) -> BTreeSet<MarkedKey> {
    value["edges"]
        .as_array()
        .unwrap()
        .iter()
        .map(|edge| {
            canonical_marked(
                edge["node1"].as_str().unwrap(),
                edge["endpoint1"].as_str().unwrap().to_owned(),
                edge["node2"].as_str().unwrap(),
                edge["endpoint2"].as_str().unwrap().to_owned(),
            )
        })
        .collect()
}

fn case_data(
    case: &JsonValue,
    permutation: &[usize],
) -> (TabularData, Vec<VariableId>, Vec<String>) {
    let base_names: Vec<&str> =
        case["var_names"].as_array().unwrap().iter().map(|v| v.as_str().unwrap()).collect();
    let names: Vec<String> = permutation.iter().map(|&i| base_names[i].to_owned()).collect();
    let rows = case["data"].as_array().unwrap();
    let n = rows.len();
    let mut builder = CausalSchemaBuilder::new();
    for name in &names {
        builder
            .add_variable(
                name.as_str(),
                ValueType::Continuous,
                SmallRoleSet::from_hint(RoleHint::Context),
                None,
                None,
                MeasurementSpec::default(),
            )
            .unwrap();
    }
    let schema = builder.build().unwrap();
    let columns: Vec<OwnedColumn> = permutation
        .iter()
        .enumerate()
        .map(|(new_index, &old_index)| {
            let values: Vec<f64> = rows
                .iter()
                .map(|row| row.as_array().unwrap()[old_index].as_f64().unwrap())
                .collect();
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(new_index as u32),
                    Arc::from(values),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    let data = TabularData::new(storage);
    let variables = (0..names.len()).map(|i| VariableId::from_raw(i as u32)).collect();
    (data, variables, names)
}

fn pag_edges(graph: &antecedent_graph::Pag, names: &[String]) -> BTreeSet<MarkedKey> {
    let mut out = BTreeSet::new();
    for a_raw in 0..graph.node_count() {
        for b_raw in a_raw + 1..graph.node_count() {
            let a = DenseNodeId::from_raw(a_raw as u32);
            let b = DenseNodeId::from_raw(b_raw as u32);
            if let Some(edge) = graph.edge_between(a, b) {
                out.insert(canonical_marked(
                    &names[a_raw],
                    endpoint(edge.at_a),
                    &names[b_raw],
                    endpoint(edge.at_b),
                ));
            }
        }
    }
    out
}

fn cpdag_edges(graph: &antecedent_graph::Cpdag, names: &[String]) -> BTreeSet<MarkedKey> {
    graph
        .edges()
        .iter()
        .map(|edge| {
            canonical_marked(
                &names[edge.a.as_usize()],
                endpoint(edge.at_a),
                &names[edge.b.as_usize()],
                endpoint(edge.at_b),
            )
        })
        .collect()
}

#[test]
fn fci_and_ges_match_causal_learn_across_motifs_and_all_permutations() {
    let fixture = fixture();
    for case in fixture["static_cases"].as_array().unwrap() {
        let mut ges_chain_disagreements = 0usize;
        for reference in case["reference_by_permutation"].as_array().unwrap() {
            let permutation: Vec<usize> = reference["permutation"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let (data, variables, names) = case_data(case, &permutation);
            let constraints =
                DiscoveryConstraints { alpha: 0.01, max_cond_size: 3, ..Default::default() };
            let mut workspace = DiscoveryWorkspace::default();
            let fci = Fci::new().with_fdr(false).with_constraints(constraints.clone());
            let fci_result = fci
                .run(&data, &variables, &mut workspace, &ExecutionContext::for_tests(0xFC1))
                .unwrap();
            let native_fci = pag_edges(&fci_result.evidence.graph, &names);
            let oracle_fci = reference_edges(&reference["fci"]);
            assert_eq!(native_fci, oracle_fci, "FCI {} permutation {permutation:?}", case["name"]);

            let mut workspace = DiscoveryWorkspace::default();
            let ges = Ges::new().with_fdr(false).with_constraints(constraints);
            let ges_result = ges
                .run(&data, &variables, &mut workspace, &ExecutionContext::for_tests(0x6E5))
                .unwrap();
            let native_ges = cpdag_edges(&ges_result.evidence.graph, &names);
            let oracle_ges = reference_edges(&reference["ges"]);
            if case["name"] == "chain" && native_ges != oracle_ges {
                ges_chain_disagreements += 1;
                let extras: BTreeSet<_> = native_ges.difference(&oracle_ges).cloned().collect();
                assert_eq!(
                    extras,
                    BTreeSet::from([(
                        "x".to_owned(),
                        "TAIL".to_owned(),
                        "y".to_owned(),
                        "TAIL".to_owned()
                    )]),
                    "GES chain disagreement should be the recorded spurious endpoint edge"
                );
            } else {
                assert_eq!(
                    native_ges, oracle_ges,
                    "GES {} permutation {permutation:?}",
                    case["name"]
                );
            }
        }
        if case["name"] == "chain" {
            assert!(
                ges_chain_disagreements >= 1,
                "at least one GES chain permutation must expose MM-010"
            );
        }
    }
}

#[test]
fn direct_lingam_matches_external_orders_edges_and_coefficients() {
    let fixture = fixture();
    let atol = fixture["tolerances"]["lingam_coefficient_atol"].as_f64().unwrap();
    let rtol = fixture["tolerances"]["lingam_coefficient_rtol"].as_f64().unwrap();
    for case in fixture["lingam_cases"].as_array().unwrap() {
        for reference in case["reference_by_permutation"].as_array().unwrap() {
            let permutation: Vec<usize> = reference["permutation"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_u64().unwrap() as usize)
                .collect();
            let (data, variables, names) = case_data(case, &permutation);
            let mut workspace = DiscoveryWorkspace::default();
            let result = DirectLingam::new()
                .with_prune_threshold(0.05)
                .run(&data, &variables, &mut workspace, &ExecutionContext::for_tests(0x11_64))
                .unwrap();
            let native: BTreeMap<(String, String), f64> = result
                .evidence
                .edge_evidence
                .iter()
                .map(|edge| {
                    (
                        (
                            names[edge.link.source.raw() as usize].clone(),
                            names[edge.link.target.raw() as usize].clone(),
                        ),
                        edge.statistic.unwrap(),
                    )
                })
                .collect();
            let oracle: BTreeMap<(String, String), f64> = reference["coefficients"]
                .as_array()
                .unwrap()
                .iter()
                .map(|edge| {
                    (
                        (
                            edge["source"].as_str().unwrap().to_owned(),
                            edge["target"].as_str().unwrap().to_owned(),
                        ),
                        edge["coefficient"].as_f64().unwrap(),
                    )
                })
                .collect();
            assert_eq!(
                native.keys().collect::<Vec<_>>(),
                oracle.keys().collect::<Vec<_>>(),
                "LiNGAM {} permutation {permutation:?} edge set",
                case["name"]
            );
            for (edge, expected) in oracle {
                let actual = native[&edge];
                assert!(
                    (actual - expected).abs() <= atol + rtol * expected.abs(),
                    "LiNGAM {} permutation {permutation:?} {edge:?}: {actual} != {expected}",
                    case["name"]
                );
            }
        }
    }
}

#[test]
fn static_discovery_fixture_records_exact_pins_and_no_generator() {
    let fixture = fixture();
    let oracle = &fixture["oracle"];
    assert_eq!(oracle["projects"]["fci_ges"]["package"]["version"], "0.1.4.3");
    assert_eq!(oracle["projects"]["direct_lingam"]["package"]["version"], "1.9.1");
    assert_eq!(oracle["generation_location"], "temporary external harness; not retained");
    for package in ["numpy", "scipy", "scikit-learn", "pandas"] {
        assert_eq!(oracle["packages"][package]["metadata_sha256"].as_str().unwrap().len(), 64);
    }
}
