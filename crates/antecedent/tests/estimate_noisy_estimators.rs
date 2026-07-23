//! estimate-parity noisy multi-estimator conformance (`val` + `se` vs pinned references).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]

use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use antecedent::CausalAnalysis;
use antecedent_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, TargetPopulation, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_estimate::OverlapPolicy;
use antecedent_graph::{Dag, DenseNodeId};
use serde_json::Value as JsonValue;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/estimate/noisy_estimators")
}

fn load_expected() -> JsonValue {
    let raw = fs::read_to_string(fixture_dir().join("expected.json")).expect("expected.json");
    serde_json::from_str(&raw).expect("parse expected.json")
}

fn close(a: f64, b: f64, atol: f64, rtol: f64) -> bool {
    (a - b).abs() <= atol + rtol * b.abs()
}

fn load_scenario(expected: &JsonValue, scenario: &str) -> (TabularData, Dag, AverageEffectQuery) {
    let sc = &expected["scenarios"][scenario];
    let csv_name = sc["csv"].as_str().unwrap();
    let csv = fs::read_to_string(fixture_dir().join(csv_name)).expect("csv");
    let cols_meta: Vec<&str> =
        sc["columns"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).collect();
    let mut columns: HashMap<String, Vec<f64>> =
        cols_meta.iter().map(|c| ((*c).to_string(), Vec::new())).collect();
    for (i, line) in csv.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let parts: Vec<&str> = line.split(',').collect();
        for (j, name) in cols_meta.iter().enumerate() {
            columns.get_mut(*name).unwrap().push(parts[j].parse().unwrap());
        }
    }
    let n = columns[cols_meta[0]].len();
    assert_eq!(n, expected["n"].as_u64().unwrap() as usize);

    let mut b = CausalSchemaBuilder::new();
    for name in &cols_meta {
        let role = if *name == sc["treatment"].as_str().unwrap() {
            RoleHint::TreatmentCandidate
        } else if *name == sc["outcome"].as_str().unwrap() {
            RoleHint::OutcomeCandidate
        } else {
            RoleHint::Context
        };
        b.add_variable(
            *name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(role),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let owned: Vec<OwnedColumn> = cols_meta
        .iter()
        .enumerate()
        .map(|(i, name)| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(i as u32),
                    Arc::from(columns[*name].clone()),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, owned, None, None).unwrap();

    let name_to_id: HashMap<&str, u32> =
        cols_meta.iter().enumerate().map(|(i, n)| (*n, i as u32)).collect();
    let mut dag = Dag::with_variables(cols_meta.len() as u32);
    for edge in sc["edges"].as_array().unwrap() {
        let from = name_to_id[edge[0].as_str().unwrap()];
        let to = name_to_id[edge[1].as_str().unwrap()];
        dag.insert_directed(DenseNodeId::from_raw(from), DenseNodeId::from_raw(to)).unwrap();
    }

    let t = VariableId::from_raw(name_to_id[sc["treatment"].as_str().unwrap()]);
    let y = VariableId::from_raw(name_to_id[sc["outcome"].as_str().unwrap()]);
    let query = if scenario == "backdoor" {
        AverageEffectQuery::binary_ate(t, y)
    } else {
        AverageEffectQuery::with_levels(t, y, 0.0, 1.0)
    };
    (TabularData::new(storage), dag, query)
}

fn run_method(
    expected: &JsonValue,
    method_id: &str,
    data: TabularData,
    graph: Dag,
    mut query: AverageEffectQuery,
    seed: u64,
) {
    let map = &expected["estimator_map"][method_id];
    let identifier = map["identifier"].as_str().unwrap();
    let estimator = map["estimator"].as_str().unwrap();
    if map["target_population"].as_str() == Some("att") {
        query = query.with_target_population(TargetPopulation::Treated);
    }

    let clip = expected["clip"].as_f64().unwrap_or(0.01);
    let analysis = CausalAnalysis::builder()
        .data(data)
        .graph(graph)
        .query(query)
        .identifier(identifier)
        .estimator(estimator)
        .overlap_policy(OverlapPolicy::RequireDiagnostics { clip: Some(clip), trim: None })
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(seed);
    let result = analysis.run(&ctx).unwrap();

    let atol_val = expected["atol_val"].as_f64().unwrap();
    let rtol_val = expected["rtol_val"].as_f64().unwrap();
    let atol_se = expected["atol_se"].as_f64().unwrap();
    let rtol_se = expected["rtol_se"].as_f64().unwrap();

    let assert_true = map["assert_against"].as_str() == Some("true_ate");
    let ref_val = if assert_true {
        expected["true_ate"].as_f64().unwrap()
    } else {
        expected["methods"][method_id]["val"].as_f64().expect("method val")
    };
    assert!(
        close(result.estimate.ate, ref_val, atol_val, rtol_val),
        "{method_id}: ate={} vs ref={ref_val} (atol={atol_val} rtol={rtol_val})",
        result.estimate.ate
    );

    let se = result.estimate.se_analytic;
    assert!(se.is_finite() && se >= 0.0, "{method_id}: bad se_analytic={se}");
    if !assert_true {
        if let Some(ref_se) = expected["methods"][method_id]["se"].as_f64() {
            assert!(
                close(se, ref_se, atol_se, rtol_se),
                "{method_id}: se={se} vs ref_se={ref_se} (atol={atol_se} rtol={rtol_se})"
            );
        }
    }
}

#[test]
fn estimate_noisy_estimators_val_and_se() {
    let expected = load_expected();
    assert_eq!(expected["tolerance_class"].as_str().unwrap(), "StableFloat");
    assert!(expected["reference"]["available"].as_bool().unwrap_or(false));

    let (data, graph, query) = load_scenario(&expected, "backdoor");
    for (i, method) in ["linear_regression", "ipw_ate", "ipw_att", "aipw"].iter().enumerate() {
        run_method(&expected, method, data.clone(), graph.clone(), query.clone(), 100 + i as u64);
    }

    let (data, graph, query) = load_scenario(&expected, "iv");
    run_method(&expected, "iv_2sls", data, graph, query, 200);

    let (data, graph, query) = load_scenario(&expected, "frontdoor");
    run_method(&expected, "frontdoor", data, graph, query, 300);
}
