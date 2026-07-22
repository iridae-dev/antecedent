//! NOTEARS linear-SEM chain conformance (`RequiredDirectedEdges`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::cast_precision_loss)]

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use causal_core::{
    CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    VariableId,
};
use causal_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TableView, TabularData, ValidityBitmap,
};
use causal_discovery::{DiscoveryWorkspace, Notears};
use causal_graph::{DenseNodeId, MarkedEdge};
use serde_json::Value as JsonValue;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/discovery/notears_chain")
}

fn load_expected() -> JsonValue {
    let raw = fs::read_to_string(fixture_dir().join("expected.json")).expect("expected.json");
    serde_json::from_str(&raw).expect("parse expected.json")
}

fn load_tabular(expected: &JsonValue) -> (TabularData, Vec<VariableId>, Vec<String>) {
    let names: Vec<String> = expected["variables"]
        .as_array()
        .expect("variables")
        .iter()
        .map(|v| v.as_str().unwrap().to_string())
        .collect();
    let csv = fs::read_to_string(fixture_dir().join("data.csv")).expect("data.csv");
    let mut cols: Vec<Vec<f64>> = names.iter().map(|_| Vec::new()).collect();
    for (i, line) in csv.lines().enumerate() {
        if i == 0 {
            continue;
        }
        for (j, part) in line.split(',').enumerate() {
            cols[j].push(part.parse().unwrap());
        }
    }
    let n = cols[0].len();
    assert_eq!(n, expected["n"].as_u64().unwrap() as usize);

    let mut b = CausalSchemaBuilder::new();
    for name in &names {
        b.add_variable(
            name.as_str(),
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let owned: Vec<OwnedColumn> = cols
        .into_iter()
        .enumerate()
        .map(|(i, data)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(i as u32),
                    Arc::from(data),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, owned, None, None).unwrap();
    let data = TabularData::new(storage);
    let vars: Vec<_> = data.schema().variables().iter().map(|v| v.id).collect();
    (data, vars, names)
}

fn name_index(names: &[String], name: &str) -> usize {
    names.iter().position(|n| n == name).unwrap_or_else(|| panic!("unknown var {name}"))
}

#[test]
fn discovery_notears_chain_required_directed_edges() {
    let expected = load_expected();
    assert_eq!(expected["tolerance_class"].as_str().unwrap(), "RequiredDirectedEdges");
    assert_eq!(expected["algorithm_id"].as_str().unwrap(), "notears");

    let (data, vars, names) = load_tabular(&expected);
    let cfg = &expected["notears"];
    let alg = Notears::new()
        .with_lambda(cfg["lambda"].as_f64().unwrap())
        .with_threshold(cfg["threshold"].as_f64().unwrap())
        .with_standardize(cfg["standardize"].as_bool().unwrap());

    let mut ws = DiscoveryWorkspace::default();
    let ctx = ExecutionContext::for_tests(1);
    let result = alg.run(&data, &vars, &mut ws, &ctx).expect("NOTEARS run");
    assert_eq!(result.discovery.algorithm.id.as_ref(), "notears");

    let g = &result.discovery.evidence.graph;
    let recovered: BTreeSet<(u32, u32)> =
        g.edges().filter_map(MarkedEdge::parent_child).map(|(a, b)| (a.raw(), b.raw())).collect();

    let mut true_edges = BTreeSet::new();
    for e in expected["true_directed_edges"].as_array().unwrap() {
        let s = name_index(&names, e["source"].as_str().unwrap()) as u32;
        let t = name_index(&names, e["target"].as_str().unwrap()) as u32;
        true_edges.insert((s, t));
        let from = DenseNodeId::from_raw(s);
        let to = DenseNodeId::from_raw(t);
        assert!(
            g.children(from).contains(&to),
            "missing required edge {s}→{t}; recovered={recovered:?}"
        );
    }

    let false_positives: BTreeSet<_> = recovered.difference(&true_edges).copied().collect();
    let max_fp = expected["max_false_positive_edges"].as_u64().unwrap() as usize;
    assert!(
        false_positives.len() <= max_fp,
        "too many false-positive edges: {false_positives:?} (max={max_fp}); recovered={recovered:?}"
    );

    // Soft weights for true edges should be non-trivial after standardize + threshold.
    let d = result.dim;
    for &(s, t) in &true_edges {
        let w = result.weights[s as usize * d + t as usize];
        assert!(
            w.abs() >= cfg["threshold"].as_f64().unwrap() * 0.5,
            "soft weight for {s}→{t} too small: {w}"
        );
    }
}
