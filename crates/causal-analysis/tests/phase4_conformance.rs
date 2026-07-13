//! Phase 4 conformance: propensity IPW, IV/2SLS, front-door two-stage.
//!
//! Fixtures under `conformance/phase4/*` are clean-room synthetic SCMs generated inline
//! (deterministic from a fixed seed) — independent of any `DoWhy` install or CSV fixture. Each
//! test checks `|estimate.ate - expected.true_effect| < expected.tolerance`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use causal_analysis::CausalAnalysis;
use causal_core::{
    AverageEffectQuery, CausalRng, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, ValueType, VariableId, SmallRoleSet,
};
use causal_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap};
use causal_graph::{Dag, DenseNodeId};
use serde_json::Value as JsonValue;

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/phase4").join(name)
}

fn load_expected(name: &str) -> JsonValue {
    let raw = fs::read_to_string(fixture_dir(name).join("expected.json")).expect("expected.json");
    serde_json::from_str(&raw).expect("parse expected.json")
}

fn standard_normal(rng: &mut CausalRng) -> f64 {
    let u1 = rng.next_f64().max(1e-12);
    let u2 = rng.next_f64();
    (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos()
}

/// Build `TabularData` from `(name, role, column)` triples; variable ids follow slice order.
fn tabular_data(vars: &[(&str, RoleHint, Vec<f64>)]) -> TabularData {
    let n = vars[0].2.len();
    let mut b = CausalSchemaBuilder::new();
    for (name, role, _) in vars {
        b.add_variable(
            *name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(*role),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let cols: Vec<OwnedColumn> = vars
        .iter()
        .enumerate()
        .map(|(i, (_, _, data))| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(u32::try_from(i).unwrap()),
                    Arc::from(data.clone()),
                    ValidityBitmap::all_valid(n),
                )
                .unwrap(),
            )
        })
        .collect();
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    TabularData::new(storage)
}

fn assert_recovers(result: &causal_analysis::CausalAnalysisResult, expected: &JsonValue) {
    let true_effect = expected["true_effect"].as_f64().unwrap();
    let tolerance = expected["tolerance"].as_f64().unwrap();
    assert!(
        (result.estimate.ate - true_effect).abs() < tolerance,
        "ate={} expected true_effect={} tolerance={}",
        result.estimate.ate,
        true_effect,
        tolerance
    );
    assert_eq!(result.logical_plan.identifier.as_deref(), expected["identifier"].as_str());
    assert_eq!(result.logical_plan.estimator.as_deref(), expected["estimator"].as_str());
}

/// `Z ~ N(0,1)` confounder; `T ~ Bernoulli(sigmoid(-0.4 + 0.9 Z))`; `Y = 2T + Z + noise`.
/// True ATE = 2; a naive unadjusted contrast is biased by `Z`, exercising IPW.
fn propensity_ipw_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x5051_u64);
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let zi = standard_normal(&mut rng);
        let logit = -0.4 + 0.9 * zi;
        let p = 1.0 / (1.0 + (-logit).exp());
        let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
        let noise = standard_normal(&mut rng) * 0.4;
        z[i] = zi;
        t[i] = ti;
        y[i] = 2.0 * ti + zi + noise;
    }
    let data = tabular_data(&[
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y),
        ("z", RoleHint::Context, z),
    ]);
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap(); // z -> t
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap(); // z -> y
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap(); // t -> y
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (data, dag, query)
}

#[test]
fn phase4_propensity_ipw_recovers_ate() {
    let expected = load_expected("propensity_ipw");
    let (data, graph, query) = propensity_ipw_scm(1200, 3);
    let analysis = CausalAnalysis::builder()
        .data(data)
        .graph(graph)
        .query(query)
        .identifier(expected["identifier"].as_str().unwrap())
        .estimator(expected["estimator"].as_str().unwrap())
        .bootstrap_replicates(30)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(9);
    let result = analysis.run(&ctx).unwrap();
    assert_recovers(&result, &expected);
    assert!(result.estimate.overlap_report.is_some(), "propensity.weighting must report overlap");
}

/// Binary instrument `Z`; unobserved confounder `U` (absent from the graph) with
/// `T = 0.6 Z + U + noise`, `Y = 2T + U + noise`. True structural effect = 2.
fn iv_2sls_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x5052_u64);
    let mut z = vec![0.0; n];
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let zi = (i % 2) as f64;
        let u = standard_normal(&mut rng);
        let ti = 0.6 * zi + u + 0.1 * standard_normal(&mut rng);
        let yi = 2.0 * ti + u + 0.1 * standard_normal(&mut rng);
        z[i] = zi;
        t[i] = ti;
        y[i] = yi;
    }
    let data = tabular_data(&[
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y),
        ("z", RoleHint::Context, z),
    ]);
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap(); // z -> t
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap(); // t -> y
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0);
    (data, dag, query)
}

#[test]
fn phase4_iv_2sls_recovers_structural_effect() {
    let expected = load_expected("iv_2sls");
    let (data, graph, query) = iv_2sls_scm(4000, 5);
    let analysis = CausalAnalysis::builder()
        .data(data)
        .graph(graph)
        .query(query)
        .identifier(expected["identifier"].as_str().unwrap())
        .estimator(expected["estimator"].as_str().unwrap())
        .bootstrap_replicates(30)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(21);
    let result = analysis.run(&ctx).unwrap();
    assert_recovers(&result, &expected);
}

/// `U -> T -> M -> Y` with `U -> Y` (no direct `T -> Y` edge; `U` unmeasured, absent from the
/// graph). `M = T + noise`, `Y = 2M + U + noise`. True mediated effect = `1 * 2 = 2`.
fn frontdoor_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng = ExecutionContext::for_tests(seed).rng.stream(0x5053_u64);
    let mut t = vec![0.0; n];
    let mut m = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        let u = standard_normal(&mut rng);
        let ti = u + 0.1 * standard_normal(&mut rng);
        let mi = ti + 0.1 * standard_normal(&mut rng);
        let yi = 2.0 * mi + u + 0.1 * standard_normal(&mut rng);
        t[i] = ti;
        m[i] = mi;
        y[i] = yi;
    }
    let data = tabular_data(&[
        ("t", RoleHint::TreatmentCandidate, t),
        ("y", RoleHint::OutcomeCandidate, y),
        ("m", RoleHint::Context, m),
    ]);
    let mut dag = Dag::with_variables(3);
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap(); // t -> m
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap(); // m -> y
    let query =
        AverageEffectQuery::with_levels(VariableId::from_raw(0), VariableId::from_raw(1), 0.0, 1.0);
    (data, dag, query)
}

#[test]
fn phase4_frontdoor_two_stage_recovers_mediated_effect() {
    let expected = load_expected("frontdoor");
    let (data, graph, query) = frontdoor_scm(4000, 1);
    let analysis = CausalAnalysis::builder()
        .data(data)
        .graph(graph)
        .query(query)
        .identifier(expected["identifier"].as_str().unwrap())
        .estimator(expected["estimator"].as_str().unwrap())
        .bootstrap_replicates(30)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(41);
    let result = analysis.run(&ctx).unwrap();
    assert_recovers(&result, &expected);
}
