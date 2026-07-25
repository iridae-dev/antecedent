//! Frozen Tigramite LPCMCI motif-matrix conformance.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::collections::BTreeSet;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, Lag, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, SamplingRegularity, TimeIndex,
    TimeSeriesData, ValidityBitmap,
};
use antecedent_discovery::{DiscoveryConstraints, DiscoveryWorkspace, Lpcmci, TemporalConstraints};
use antecedent_graph::{DenseNodeId, NodeRef};
use serde_json::Value as JsonValue;

type LinkKey = (String, u32, String);

fn fixture() -> JsonValue {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/discovery/lpcmci_matrix/expected.json");
    serde_json::from_str(&fs::read_to_string(path).expect("LPCMCI fixture"))
        .expect("parse LPCMCI fixture")
}

fn series(case: &JsonValue) -> (TimeSeriesData, Vec<VariableId>, Vec<String>) {
    let names: Vec<String> = case["var_names"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_str().unwrap().to_owned())
        .collect();
    let rows = case["data"].as_array().unwrap();
    let n = rows.len();
    let mut schema = CausalSchemaBuilder::new();
    for name in &names {
        schema
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
    let schema = schema.build().unwrap();
    let mut columns = Vec::new();
    for column in 0..names.len() {
        let values: Vec<f64> =
            rows.iter().map(|row| row.as_array().unwrap()[column].as_f64().unwrap()).collect();
        columns.push(OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(column as u32),
                Arc::from(values),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ));
    }
    let storage = OwnedColumnarStorage::try_new(schema, columns, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let variables = (0..names.len()).map(|i| VariableId::from_raw(i as u32)).collect();
    (data, variables, names)
}

fn canonical(source: &str, lag: u32, target: &str) -> LinkKey {
    if lag == 0 && source > target {
        (target.to_owned(), lag, source.to_owned())
    } else {
        (source.to_owned(), lag, target.to_owned())
    }
}

fn reference_links(case: &JsonValue) -> BTreeSet<LinkKey> {
    case["reference"]["links"]
        .as_array()
        .unwrap()
        .iter()
        .map(|link| {
            canonical(
                link["source"].as_str().unwrap(),
                link["source_lag"].as_u64().unwrap() as u32,
                link["target"].as_str().unwrap(),
            )
        })
        .collect()
}

#[test]
fn lpcmci_matches_tigramite_motif_matrix_and_relabeling() {
    let fixture = fixture();
    let mut normalized_lag_chain_native = None;
    for case in fixture["cases"].as_array().unwrap() {
        let (data, variables, names) = series(case);
        let params = &case["parameters"];
        let algorithm = Lpcmci::new().with_fdr(false).with_constraints(DiscoveryConstraints {
            temporal: TemporalConstraints {
                min_lag: Lag::from_raw(params["tau_min"].as_u64().unwrap() as u32),
                max_lag: Lag::from_raw(params["tau_max"].as_u64().unwrap() as u32),
            },
            alpha: params["pc_alpha"].as_f64().unwrap(),
            max_cond_size: 3,
            ..DiscoveryConstraints::default()
        });
        let mut workspace = DiscoveryWorkspace::default();
        let result = algorithm
            .run(&data, &variables, &mut workspace, &ExecutionContext::for_tests(0x1C_C0))
            .unwrap();
        let graph = &result.evidence.graph;
        let mut native = BTreeSet::new();
        for a_raw in 0..graph.node_count() {
            let a = DenseNodeId::from_raw(a_raw as u32);
            for b_raw in a_raw + 1..graph.node_count() {
                let b = DenseNodeId::from_raw(b_raw as u32);
                if graph.edge_between(a, b).is_none() {
                    continue;
                }
                let NodeRef::Lagged { variable: av, lag: al } = graph.nodes()[a_raw] else {
                    unreachable!()
                };
                let NodeRef::Lagged { variable: bv, lag: bl } = graph.nodes()[b_raw] else {
                    unreachable!()
                };
                if al == bl {
                    native.insert(canonical(
                        &names[av.raw() as usize],
                        0,
                        &names[bv.raw() as usize],
                    ));
                } else if al.raw() > bl.raw() {
                    native.insert(canonical(
                        &names[av.raw() as usize],
                        al.raw() - bl.raw(),
                        &names[bv.raw() as usize],
                    ));
                } else {
                    native.insert(canonical(
                        &names[bv.raw() as usize],
                        bl.raw() - al.raw(),
                        &names[av.raw() as usize],
                    ));
                }
            }
        }
        let reference = reference_links(case);
        let name = case["name"].as_str().unwrap();
        assert_eq!(native, reference, "{name} canonical skeleton parity");
        if name == "lag_chain" {
            normalized_lag_chain_native = Some(native.clone());
        } else if name == "lag_chain_relabelled" {
            assert_eq!(
                Some(&native),
                normalized_lag_chain_native.as_ref(),
                "normalized native skeleton must be invariant to column order"
            );
        }
        assert_eq!(result.algorithm.id.as_ref(), "lpcmci");
        assert!(result.review.graph.node_count() >= variables.len());
    }
}

#[test]
fn lpcmci_fixture_records_pins_and_no_generator() {
    let fixture = fixture();
    let oracle = &fixture["oracle"];
    assert_eq!(oracle["packages"]["tigramite"]["version"], "5.2.1.30");
    assert_eq!(oracle["packages"]["numpy"]["version"], "1.23.5");
    assert_eq!(oracle["generation_location"], "temporary external harness; not retained");
    assert!(oracle["api"].as_str().unwrap().contains("run_lpcmci"));
    for package in ["tigramite", "numpy", "scipy"] {
        assert_eq!(oracle["packages"][package]["metadata_sha256"].as_str().unwrap().len(), 64);
    }
}
