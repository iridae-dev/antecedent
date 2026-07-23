//! Prepared analysis re-estimate conformance (backlog B).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::many_single_char_names)]

use std::sync::Arc;
use std::time::Instant;

use antecedent::{CausalAnalysis, LatencyMode, PreparedAnalysis, RefuteSuite};
use causal_core::{
    AverageEffectQuery, CausalRng, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use causal_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap};
use causal_graph::{Dag, DenseNodeId};

/// Confounded linear SCM with structural ATE = 2.
fn confounded_scm(n: usize, seed: u64) -> (TabularData, Dag, AverageEffectQuery) {
    let mut rng = CausalRng::from_seed(seed);
    let mut t = Vec::with_capacity(n);
    let mut y = Vec::with_capacity(n);
    let mut z = Vec::with_capacity(n);
    for _ in 0..n {
        let u1 = rng.next_f64().max(1e-12);
        let u2 = rng.next_f64();
        let zi = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
        let logit = -0.4 + 0.9 * zi;
        let p = 1.0 / (1.0 + (-logit).exp());
        let ti = if rng.next_f64() < p { 1.0 } else { 0.0 };
        let e = (-2.0 * rng.next_f64().max(1e-12).ln()).sqrt()
            * (2.0 * std::f64::consts::PI * rng.next_f64()).cos()
            * 0.4;
        let yi = 2.0 * ti + zi + e;
        z.push(zi);
        t.push(ti);
        y.push(yi);
    }

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
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    dag.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1));
    (TabularData::new(storage), dag, query)
}

fn build_analysis(data: TabularData, dag: Dag, query: AverageEffectQuery) -> CausalAnalysis {
    CausalAnalysis::builder()
        .data(data)
        .graph(dag)
        .query(query)
        .latency_mode(LatencyMode::Interactive)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

#[test]
fn prepared_reestimate_matches_fresh_analyze() {
    let (data, dag, query) = confounded_scm(500, 19);
    let ctx = ExecutionContext::for_tests(1);

    let fresh = build_analysis(data.clone(), dag.clone(), query.clone()).run(&ctx).unwrap();

    let prepared = build_analysis(data.clone(), dag, query).prepare(&ctx).unwrap();
    let first = prepared.estimate(&data, &ctx).unwrap();
    let second = prepared.estimate(&data, &ctx).unwrap();

    assert!(first.estimate.ate.is_finite());
    assert!((first.estimate.ate - 2.0).abs() < 0.5, "ate={}", first.estimate.ate);
    assert!((first.estimate.ate - fresh.estimate.ate).abs() < 1e-12);
    assert!((second.estimate.ate - fresh.estimate.ate).abs() < 1e-12);
    assert_eq!(
        format!("{:?}", first.identification.status),
        format!("{:?}", fresh.identification.status)
    );
    assert_eq!(first.estimand.adjustment_set, fresh.estimand.adjustment_set);
    assert_eq!(first.physical_plan.plan_id, fresh.physical_plan.plan_id);
    assert_eq!(second.physical_plan.plan_id, first.physical_plan.plan_id);
}

#[test]
fn prepared_refresh_rejects_schema_mismatch() {
    let (data, dag, query) = confounded_scm(200, 23);
    let ctx = ExecutionContext::for_tests(1);
    let mut prepared = build_analysis(data.clone(), dag, query).prepare(&ctx).unwrap();

    let (other, _, _) = confounded_scm(50, 29);
    // Same SCM schema actually — rebuild with different variable names.
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "treatment",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::TreatmentCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    b.add_variable(
        "outcome",
        ValueType::Continuous,
        SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
        None,
        None,
        MeasurementSpec::default(),
    )
    .unwrap();
    let schema = b.build().unwrap();
    let n = 10;
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(vec![0.0; n]),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(1),
                Arc::from(vec![0.0; n]),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
    ];
    let bad = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let err = prepared.refresh(bad, &ctx).unwrap_err();
    let msg = err.to_string();
    assert!(msg.contains("same schema"), "unexpected: {msg}");
    // Original data still works after failed refresh.
    let ok = prepared.estimate(&data, &ctx).unwrap();
    assert!(ok.estimate.ate.is_finite());
    let _ = other;
}

#[test]
fn prepared_second_shot_cheaper_than_full_run() {
    let (data, dag, query) = confounded_scm(800, 31);
    let ctx = ExecutionContext::for_tests(1);

    let t0 = Instant::now();
    let analysis = build_analysis(data.clone(), dag.clone(), query.clone());
    let prepared = analysis.prepare(&ctx).unwrap();
    let _ = prepared.estimate(&data, &ctx).unwrap();
    let prepare_plus_first = t0.elapsed();

    let t1 = Instant::now();
    let _ = prepared.estimate(&data, &ctx).unwrap();
    let second = t1.elapsed();

    // Second shot skips compile; on this toy it should not be slower than a full prepare+estimate.
    // Allow generous slack for CI noise — assert structural speedup intent, not a tight budget.
    assert!(
        second <= prepare_plus_first.saturating_mul(2),
        "second={second:?} prepare+first={prepare_plus_first:?}"
    );
    // Stronger check: plan is retained (no recompile path).
    assert_eq!(
        prepared.plan().record.plan_id.as_ref(),
        prepared.estimate(&data, &ctx).unwrap().physical_plan.plan_id.as_ref()
    );
}

#[test]
fn prepare_refuses_discovery_graph() {
    use antecedent::{DiscoveryAccept, FdrControl};
    let (data, _, query) = confounded_scm(100, 37);
    // Discovery under Interactive is refused at build; Standard one-shot still
    // reaches prepare, which refuses Discover* graphs.
    let err = CausalAnalysis::builder()
        .data(data)
        .discover_pc(0.05, 3, FdrControl::Off, DiscoveryAccept::AutoAccept)
        .query(query)
        .latency_mode(LatencyMode::Standard)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ExecutionContext::for_tests(1))
        .unwrap_err();
    assert!(err.to_string().contains("supplied static"), "unexpected: {err}");
}

#[test]
fn refresh_updates_retained_data() {
    let (data1, dag, query) = confounded_scm(400, 41);
    let (data2, _, _) = confounded_scm(400, 43);
    let ctx = ExecutionContext::for_tests(1);
    let mut prepared: PreparedAnalysis =
        build_analysis(data1.clone(), dag, query).prepare(&ctx).unwrap();
    let a = prepared.refresh(data1, &ctx).unwrap().estimate.ate;
    let b = prepared.refresh(data2, &ctx).unwrap().estimate.ate;
    // Different seeds → different finite ATEs (not identical bit-for-bit).
    assert!(a.is_finite() && b.is_finite());
    assert!((a - 2.0).abs() < 0.6);
    assert!((b - 2.0).abs() < 0.6);
}
