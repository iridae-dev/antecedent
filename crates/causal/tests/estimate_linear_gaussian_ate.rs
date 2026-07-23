//! estimate-parity linear-Gaussian ATE conformance (`StableFloat`).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::many_single_char_names
)]

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use causal::{
    CausalAnalysis,
    RefuteSuite,
};
use causal_core::{
    AverageEffectQuery, CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint,
    SmallRoleSet, ToleranceClass, ValueType, VariableId,
};
use causal_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap};
use causal_graph::{Dag, DenseNodeId};
use serde_json::Value as JsonValue;

fn fixture_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/estimate/linear_gaussian_ate")
}

fn load_expected() -> JsonValue {
    let raw = fs::read_to_string(fixture_dir().join("expected.json")).expect("expected.json");
    serde_json::from_str(&raw).expect("parse expected.json")
}

fn load_csv(expected: &JsonValue) -> (TabularData, Dag, AverageEffectQuery) {
    let csv = fs::read_to_string(fixture_dir().join("data.csv")).expect("data.csv");
    let mut t = Vec::new();
    let mut y = Vec::new();
    let mut z = Vec::new();
    for (i, line) in csv.lines().enumerate() {
        if i == 0 {
            continue;
        }
        let mut parts = line.split(',');
        t.push(parts.next().unwrap().parse::<f64>().unwrap());
        y.push(parts.next().unwrap().parse::<f64>().unwrap());
        z.push(parts.next().unwrap().parse::<f64>().unwrap());
    }
    let n = t.len();
    let expected_n = expected["n"].as_u64().expect("n") as usize;
    assert_eq!(n, expected_n, "csv row count vs expected.n");

    let treatment = expected["treatment"].as_str().unwrap();
    let outcome = expected["outcome"].as_str().unwrap();
    assert_eq!(treatment, "t");
    assert_eq!(outcome, "y");

    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "t",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "y",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "z",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::Context),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(z), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let mut dag = Dag::with_variables(3);
    for edge in expected["edges"].as_array().expect("edges") {
        let from = edge[0].as_str().unwrap();
        let to = edge[1].as_str().unwrap();
        let from_id = match from {
            "t" => 0u32,
            "y" => 1,
            "z" => 2,
            other => panic!("unknown edge endpoint {other}"),
        };
        let to_id = match to {
            "t" => 0u32,
            "y" => 1,
            "z" => 2,
            other => panic!("unknown edge endpoint {other}"),
        };
        dag.insert_directed(DenseNodeId::from_raw(from_id), DenseNodeId::from_raw(to_id)).unwrap();
    }
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (TabularData::new(storage), dag, query)
}

#[test]
fn estimate_linear_gaussian_ate_stable_float() {
    let expected = load_expected();
    assert_eq!(expected["tolerance_class"].as_str().unwrap(), "StableFloat");
    let true_ate = expected["true_ate"].as_f64().unwrap();
    let reference_ate = expected["reference_ate"].as_f64().unwrap();
    let adjustment = expected["adjustment_set"].as_array().unwrap();
    assert_eq!(adjustment.len(), 1);
    assert_eq!(adjustment[0].as_str().unwrap(), "z");

    let (data, graph, query) = load_csv(&expected);
    let analysis = CausalAnalysis::builder()
        .data(data)
        .graph(graph)
        .query(query)
        .refute(RefuteSuite::PlaceboAndRcc)
        .bootstrap_replicates(20)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(42);
    let result = analysis.run(&ctx).unwrap();

    assert!(
        ToleranceClass::StableFloat.close(result.estimate.ate, true_ate),
        "ate={} vs true {}",
        result.estimate.ate,
        true_ate
    );
    assert!(
        ToleranceClass::StableFloat.close(result.estimate.ate, reference_ate),
        "ate={} vs estimate-parity/reference {}",
        result.estimate.ate,
        reference_ate
    );
    // When the fixture was generated against pinned estimate-parity, also require the
    // black-box estimate field itself (guards generator / reference_ate drift).
    let dw = &expected["reference"];
    if dw["available"].as_bool() == Some(true) {
        let bb = dw["outputs"]["estimate"].as_f64().expect("estimate.estimate");
        assert!(
            ToleranceClass::StableFloat.close(result.estimate.ate, bb),
            "ate={} vs estimate-parity black-box {}",
            result.estimate.ate,
            bb
        );
        assert!(
            ToleranceClass::StableFloat.close(bb, true_ate),
            "estimate-parity black-box {bb} diverged from analytic true_ate {true_ate}"
        );
    }
    assert_eq!(result.estimand.adjustment_set.as_ref(), &[VariableId::from_raw(2)]);
    assert_eq!(result.refutations.len(), 2);
    assert!(result.refutations.iter().all(|r| r.passed));
    assert!(!result.estimate.assumptions.is_empty());
    assert!(!result.identification.derivation.steps.is_empty());

    let trace = result.analysis_trace_wire();
    assert!(!trace.assumptions.is_empty());
    assert!(!trace.derivation.is_empty());
}
