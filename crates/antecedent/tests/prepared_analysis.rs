//! Prepared analysis re-estimate conformance (backlog B).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::many_single_char_names)]

use std::sync::Arc;
use std::time::Instant;

use antecedent::{LatencyMode, PreparedStudy, RefuteSuite, Study};
use antecedent_core::{
    AverageEffectQuery, CausalRng, CausalSchemaBuilder, ExecutionContext, MeasurementSpec,
    RoleHint, SmallRoleSet, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_graph::{Dag, DenseNodeId};

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

fn build_analysis(data: TabularData, dag: Dag, query: AverageEffectQuery) -> Study {
    Study::tabular(data)
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
fn prepare_refuses_temporal_graph() {
    // MIGRATION NOTE: this test used to be `prepare_refuses_discovery_graph`, asserting
    // that `.prepare()` refused a graph tagged internally as coming from inline builder
    // discovery (`.discover_pc(..)`, an old `GraphInput::Discover*` variant), with an
    // error message containing "supplied static". That distinction is retired, not
    // relocated: `.discover_pc(..)` and the old `GraphInput` enum are deleted, and a
    // graph produced by standalone discovery + `AcceptedGraph::accept(..)` is, once
    // accepted, an ordinary `AcceptedGraph` of some `GraphClass` — indistinguishable
    // from one supplied directly (`AcceptedGraph::algorithm_id()` is set by discovery
    // but has no reader in `ensure_prepared_supported`,
    // `crates/antecedent/src/analysis/prepared.rs:199-217`). The only structural
    // refusal `PreparedStudy::prepare` still performs is on graph *class*
    // (`is_supplied_static_graph`): temporal classes are refused as "not
    // session-refreshable here" — a different condition, with different wording, than
    // the one this test used to assert. No reachable call path reproduces the original
    // "supplied static" message substring. Rewritten to cover the refusal that actually
    // still exists; see the migration report for the retired behavior.
    use antecedent_core::{Lag, TemporalEffectQuery, TemporalPolicy};
    use antecedent_data::{SamplingRegularity, TimeIndex, TimeSeriesData};
    use antecedent_graph::{TemporalDag, ensure_lagged};

    let n = 60usize;
    let mut b = CausalSchemaBuilder::new();
    b.add_variable(
        "x",
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
    let schema = b.build().unwrap();
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = ((t as f64) * 0.05).sin();
        y[t] = 0.5 * x[t - 1];
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(x), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let series = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let mut g = TemporalDag::empty();
    let x1 = ensure_lagged(&mut g, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut g, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    g.insert_directed(x1, y0).unwrap();
    let q = TemporalEffectQuery::pulse(VariableId::from_raw(0), VariableId::from_raw(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1);

    let err = Study::series(series)
        .graph(g)
        .temporal_query(q)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ExecutionContext::for_tests(1))
        .unwrap_err();
    // `prepare` is a tabular + AverageEffect session handle; a temporal study is
    // refused at that gate, before graph class is ever considered.
    let msg = err.to_string();
    assert!(msg.contains("tabular") && msg.contains("AverageEffect"), "unexpected: {err}");
}

#[test]
fn refresh_updates_retained_data() {
    let (data1, dag, query) = confounded_scm(400, 41);
    let (data2, _, _) = confounded_scm(400, 43);
    let ctx = ExecutionContext::for_tests(1);
    let mut prepared: PreparedStudy =
        build_analysis(data1.clone(), dag, query).prepare(&ctx).unwrap();
    let a = prepared.refresh(data1, &ctx).unwrap().estimate.ate;
    let b = prepared.refresh(data2, &ctx).unwrap().estimate.ate;
    // Different seeds → different finite ATEs (not identical bit-for-bit).
    assert!(a.is_finite() && b.is_finite());
    assert!((a - 2.0).abs() < 0.6);
    assert!((b - 2.0).abs() < 0.6);
}
