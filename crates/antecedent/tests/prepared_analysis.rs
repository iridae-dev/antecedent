//! Prepared analysis re-estimate conformance (backlog B).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::float_cmp, clippy::many_single_char_names)]

use std::sync::Arc;
use std::time::Instant;

use antecedent::{
    AcceptedGraph, BayesianConfig, EstimatorId, IdentifierId, InferenceMode, LatencyMode,
    PreparedStudy, RefuteSuite, Study,
};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, CausalRng, CausalSchemaBuilder, ConditionalEffectQuery,
    ExecutionContext, Intervention, InterventionalDistributionQuery, MeasurementSpec,
    PathSpecificEffectQuery, RoleHint, SmallRoleSet, Value, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_graph::{Admg, Dag, DenseNodeId};

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

    // Prepared clicks reuse prepare-time identification (observable, and only
    // there): fresh runs must not carry the marker.
    for result in [&first, &second] {
        assert!(
            result.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"),
            "prepared estimate missing exec.identify.cached diagnostic"
        );
    }
    assert!(fresh.diagnostics.iter().all(|d| d.code.as_ref() != "exec.identify.cached"));
}

#[test]
fn prepared_accepted_graph_and_cheap_validation_reuse_identification() {
    let (data, dag, query) = confounded_scm(400, 47);
    let ctx = ExecutionContext::for_tests(1);
    let analysis = Study::tabular(data.clone())
        .graph(AcceptedGraph::from(dag))
        .query(query)
        .latency_mode(LatencyMode::Interactive)
        .refute(RefuteSuite::Cheap)
        .build()
        .unwrap();
    let fresh = analysis.run(&ctx).unwrap();
    let prepared = analysis.prepare(&ctx).unwrap();
    let click = prepared.estimate(&data, &ctx).unwrap();
    assert!((click.estimate.ate - fresh.estimate.ate).abs() < 1e-12);
    assert!(click.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"));
    assert!(!click.refutations.is_empty(), "cheap suite must still run on a prepared click");
    assert_eq!(analysis.structure_source(), antecedent::StructureSource::Accepted);
}

#[test]
fn supplied_dag_is_explicit_structure_source() {
    let (data, dag, query) = confounded_scm(80, 11);
    let analysis = build_analysis(data, dag, query);
    assert_eq!(analysis.structure_source(), antecedent::StructureSource::Explicit);
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
    // Stronger check: plan is retained (no recompile path) and identification
    // is served from the prepare-time cache rather than re-derived per click.
    let third = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(prepared.plan().record.plan_id.as_ref(), third.physical_plan.plan_id.as_ref());
    assert!(
        third.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"),
        "prepared second shot did not hit the identification cache"
    );
}

fn build_bayesian_analysis(data: TabularData, dag: Dag, query: AverageEffectQuery) -> Study {
    Study::tabular(data)
        .graph(dag)
        .query(query)
        .latency_mode(LatencyMode::Interactive)
        .refute(RefuteSuite::None)
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .build()
        .unwrap()
}

#[test]
fn prepared_bayesian_reestimate_matches_fresh_analyze() {
    let (data, dag, query) = confounded_scm(300, 53);
    let ctx = ExecutionContext::for_tests(1);

    let fresh =
        build_bayesian_analysis(data.clone(), dag.clone(), query.clone()).run(&ctx).unwrap();
    let prepared = build_bayesian_analysis(data.clone(), dag, query).prepare(&ctx).unwrap();
    let first = prepared.estimate(&data, &ctx).unwrap();

    assert!(first.estimate.ate.is_finite());
    // Bit-identical ATE and posterior summaries between the prepared click and
    // a fresh, un-prepared run: identification caching must not perturb the
    // downstream Bayesian fit (same prep, prior, draws, seed).
    assert_eq!(first.estimate.ate.to_bits(), fresh.estimate.ate.to_bits());
    assert_eq!(first.estimand.adjustment_set, fresh.estimand.adjustment_set);
    assert_eq!(
        format!("{:?}", first.identification.status),
        format!("{:?}", fresh.identification.status)
    );

    let first_post = first.posterior.as_ref().expect("prepared run must carry a posterior");
    let fresh_post = fresh.posterior.as_ref().expect("fresh run must carry a posterior");
    assert_eq!(first_post.summaries.n_draws, fresh_post.summaries.n_draws);
    let first_bits: Vec<u64> = first_post.summaries.mean.iter().map(|v| v.to_bits()).collect();
    let fresh_bits: Vec<u64> = fresh_post.summaries.mean.iter().map(|v| v.to_bits()).collect();
    assert_eq!(first_bits, fresh_bits);
    let first_sd_bits: Vec<u64> = first_post.summaries.sd.iter().map(|v| v.to_bits()).collect();
    let fresh_sd_bits: Vec<u64> = fresh_post.summaries.sd.iter().map(|v| v.to_bits()).collect();
    assert_eq!(first_sd_bits, fresh_sd_bits);

    // Only the prepared click carries the cache marker.
    assert!(
        first.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"),
        "prepared Bayesian estimate missing exec.identify.cached diagnostic"
    );
    assert!(fresh.diagnostics.iter().all(|d| d.code.as_ref() != "exec.identify.cached"));
}

#[test]
fn prepared_bayesian_two_clicks_agree_exactly() {
    let (data, dag, query) = confounded_scm(250, 61);
    let ctx = ExecutionContext::for_tests(1);
    let prepared = build_bayesian_analysis(data.clone(), dag, query).prepare(&ctx).unwrap();

    let first = prepared.estimate(&data, &ctx).unwrap();
    let second = prepared.estimate(&data, &ctx).unwrap();

    assert_eq!(first.estimate.ate.to_bits(), second.estimate.ate.to_bits());
    let first_post = first.posterior.as_ref().expect("posterior");
    let second_post = second.posterior.as_ref().expect("posterior");
    let first_bits: Vec<u64> = first_post.summaries.mean.iter().map(|v| v.to_bits()).collect();
    let second_bits: Vec<u64> = second_post.summaries.mean.iter().map(|v| v.to_bits()).collect();
    assert_eq!(first_bits, second_bits);
    for result in [&first, &second] {
        assert!(
            result.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"),
            "prepared Bayesian click missing exec.identify.cached diagnostic"
        );
    }
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

// -- ConditionalEffect / PathSpecific / InterventionalDistribution staging -----------------

/// Deterministic 3-column continuous fixture (no RNG): `t -> y <- w` interaction table.
fn conditional_effect_fixture() -> (TabularData, Dag, ConditionalEffectQuery) {
    let n = 120usize;
    let mut b = CausalSchemaBuilder::new();
    for name in ["t", "y", "w"] {
        b.add_variable(
            name,
            antecedent_core::ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let t: Vec<f64> = (0..n).map(|i| if i % 2 == 0 { 0.0 } else { 1.0 }).collect();
    let w: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    let y: Vec<f64> =
        t.iter().zip(w.iter()).map(|(&ti, &wi)| 1.0 + 2.0 * ti + 0.5 * ti * wi).collect();
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
            Float64Column::new(VariableId::from_raw(2), Arc::from(w), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut g = Dag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    let inner = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_effect_modifiers([VariableId::from_raw(2)]);
    let query = ConditionalEffectQuery::try_new(inner).unwrap();
    (data, g, query)
}

/// Deterministic discrete chain `t -> m -> y` (no direct edge): all-path natural effect.
fn path_specific_fixture() -> (TabularData, Dag, PathSpecificEffectQuery) {
    let mut b = CausalSchemaBuilder::new();
    for name in ["t", "m", "y"] {
        b.add_variable(
            name,
            antecedent_core::ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let mut t_vals = Vec::new();
    let mut m_vals = Vec::new();
    let mut y_vals = Vec::new();
    for t in [0.0, 1.0] {
        for _ in 0..50 {
            t_vals.push(t);
            m_vals.push(t);
            y_vals.push(t);
        }
    }
    let n = t_vals.len();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(t_vals),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(1),
                Arc::from(m_vals),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(2),
                Arc::from(y_vals),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut g = Dag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let query = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2))
        .with_path_nodes([VariableId::from_raw(1)]);
    (data, g, query)
}

/// Deterministic confounded `z -> t -> y <- z` count table (8 cells): interventional
/// distribution `P(y | do(t=1))`.
fn distribution_fixture() -> (TabularData, Dag, InterventionalDistributionQuery) {
    let mut b = CausalSchemaBuilder::new();
    for name in ["t", "y", "z"] {
        b.add_variable(
            name,
            antecedent_core::ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let combos = [
        (0.0, 0.0, 0.0, 21),
        (0.0, 0.0, 1.0, 9),
        (0.0, 1.0, 0.0, 4),
        (0.0, 1.0, 1.0, 16),
        (1.0, 0.0, 0.0, 12),
        (1.0, 0.0, 1.0, 3),
        (1.0, 1.0, 0.0, 14),
        (1.0, 1.0, 1.0, 21),
    ];
    let mut t_vals = Vec::new();
    let mut y_vals = Vec::new();
    let mut z_vals = Vec::new();
    for (z, t, y, count) in combos {
        for _ in 0..count {
            z_vals.push(z);
            t_vals.push(t);
            y_vals.push(y);
        }
    }
    let n = t_vals.len();
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(0),
                Arc::from(t_vals),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(1),
                Arc::from(y_vals),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(
                VariableId::from_raw(2),
                Arc::from(z_vals),
                ValidityBitmap::all_valid(n),
            )
            .unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut g = Dag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let query = InterventionalDistributionQuery::new(
        VariableId::from_raw(1),
        [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
    );
    (data, g, query)
}

fn assert_cached_only_on_prepared(
    fresh: &antecedent::StudyResult,
    prepared_clicks: &[&antecedent::StudyResult],
) {
    assert!(fresh.diagnostics.iter().all(|d| d.code.as_ref() != "exec.identify.cached"));
    for click in prepared_clicks {
        assert!(
            click.diagnostics.iter().any(|d| d.code.as_ref() == "exec.identify.cached"),
            "prepared click missing exec.identify.cached diagnostic"
        );
    }
}

#[test]
fn prepared_conditional_effect_reestimate_matches_fresh() {
    let (data, dag, query) = conditional_effect_fixture();
    let ctx = ExecutionContext::for_tests(1);
    let build = |data: TabularData, dag: Dag, query: ConditionalEffectQuery| {
        Study::tabular(data)
            .graph(dag)
            .query(CausalQuery::ConditionalEffect(query))
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
    };
    let fresh = build(data.clone(), dag.clone(), query.clone()).run(&ctx).unwrap();
    let prepared = build(data.clone(), dag, query).prepare(&ctx).unwrap();
    let first = prepared.estimate(&data, &ctx).unwrap();
    let second = prepared.estimate(&data, &ctx).unwrap();

    assert!(first.estimate.ate.is_finite());
    assert_eq!(first.estimate.ate.to_bits(), fresh.estimate.ate.to_bits());
    assert_eq!(second.estimate.ate.to_bits(), fresh.estimate.ate.to_bits());
    assert_eq!(first.estimand.adjustment_set, fresh.estimand.adjustment_set);
    assert_cached_only_on_prepared(&fresh, &[&first, &second]);
}

#[test]
fn prepared_path_specific_reestimate_matches_fresh() {
    let (data, dag, query) = path_specific_fixture();
    let ctx = ExecutionContext::for_tests(1);
    let build = |data: TabularData, dag: Dag, query: PathSpecificEffectQuery| {
        Study::tabular(data)
            .graph(dag)
            .query(CausalQuery::PathSpecific(query))
            .identifier(IdentifierId::PathSpecificNatural)
            .estimator(EstimatorId::FunctionalEffect)
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
    };
    let fresh = build(data.clone(), dag.clone(), query.clone()).run(&ctx).unwrap();
    let prepared = build(data.clone(), dag, query).prepare(&ctx).unwrap();
    let first = prepared.estimate(&data, &ctx).unwrap();
    let second = prepared.estimate(&data, &ctx).unwrap();

    assert!(first.estimate.ate.is_finite());
    assert_eq!(first.estimate.ate.to_bits(), fresh.estimate.ate.to_bits());
    assert_eq!(second.estimate.ate.to_bits(), fresh.estimate.ate.to_bits());
    assert_eq!(first.estimand.method.as_ref(), fresh.estimand.method.as_ref());
    assert_cached_only_on_prepared(&fresh, &[&first, &second]);
}

#[test]
fn prepared_distribution_reestimate_matches_fresh() {
    let (data, dag, query) = distribution_fixture();
    let ctx = ExecutionContext::for_tests(1);
    let build = |data: TabularData, dag: Dag, query: InterventionalDistributionQuery| {
        Study::tabular(data)
            .graph(dag)
            .query(CausalQuery::Distribution(query))
            .identifier(IdentifierId::GeneralId)
            .estimator(EstimatorId::FunctionalDistribution)
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
    };
    let fresh = build(data.clone(), dag.clone(), query.clone()).run(&ctx).unwrap();
    let prepared = build(data.clone(), dag, query).prepare(&ctx).unwrap();
    let first = prepared.estimate(&data, &ctx).unwrap();
    let second = prepared.estimate(&data, &ctx).unwrap();

    assert!(first.estimate.ate.is_finite());
    let first_dist = first.distribution.as_ref().expect("prepared distribution payload");
    let fresh_dist = fresh.distribution.as_ref().expect("fresh distribution payload");
    assert_eq!(first_dist.mean.to_bits(), fresh_dist.mean.to_bits());
    let second_dist = second.distribution.as_ref().expect("second distribution payload");
    assert_eq!(second_dist.mean.to_bits(), fresh_dist.mean.to_bits());
    assert_eq!(first.estimand.method.as_ref(), fresh.estimand.method.as_ref());
    assert_cached_only_on_prepared(&fresh, &[&first, &second]);
}

/// Non-Dag explicit structure (a bidirected-free ADMG, class `Admg`) now refuses these
/// three query kinds at `.build()` itself: `parity/support_closed.toml` closes
/// `ConditionalEffect` / `PathSpecificEffect` / `InterventionalDistribution` on Cpdag / Admg /
/// Pag (they are staged Dag-only), so the matrix catches the mismatch before `.prepare()`
/// would otherwise have to.
#[test]
fn prepare_refuses_conditional_path_distribution_on_non_dag_graph() {
    let mut admg = Admg::with_variables(3);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    admg.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();

    let (data, _dag, cq) = conditional_effect_fixture();
    let err = Study::tabular(data)
        .graph(admg.clone())
        .query(CausalQuery::ConditionalEffect(cq))
        .refute(RefuteSuite::None)
        .build()
        .unwrap_err();
    assert!(err.to_string().contains("Dag"), "{err}");

    let mut admg2 = Admg::with_variables(3);
    admg2.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    admg2.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let (data, _dag, pq) = path_specific_fixture();
    let err = Study::tabular(data)
        .graph(admg2)
        .query(CausalQuery::PathSpecific(pq))
        .identifier(IdentifierId::PathSpecificNatural)
        .estimator(EstimatorId::FunctionalEffect)
        .refute(RefuteSuite::None)
        .build()
        .unwrap_err();
    assert!(err.to_string().contains("Dag"), "{err}");

    let mut admg3 = Admg::with_variables(3);
    admg3.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(0)).unwrap();
    admg3.insert_directed(DenseNodeId::from_raw(2), DenseNodeId::from_raw(1)).unwrap();
    admg3.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let (data, _dag, distq) = distribution_fixture();
    let err = Study::tabular(data)
        .graph(admg3)
        .query(CausalQuery::Distribution(distq))
        .identifier(IdentifierId::GeneralId)
        .estimator(EstimatorId::FunctionalDistribution)
        .refute(RefuteSuite::None)
        .build()
        .unwrap_err();
    assert!(err.to_string().contains("Dag"), "{err}");
    let _ = admg;
}

/// `PreparedStudy::refute` is licensed for `AverageEffect` only; a prepared
/// `ConditionalEffect` handle must refuse cleanly (stable `Support` id), not panic.
#[test]
fn refute_on_prepared_conditional_effect_refuses_cleanly() {
    let (data, dag, query) = conditional_effect_fixture();
    let ctx = ExecutionContext::for_tests(1);
    let prepared = Study::tabular(data.clone())
        .graph(dag)
        .query(CausalQuery::ConditionalEffect(query))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let first = prepared.estimate(&data, &ctx).unwrap();
    let err = prepared.refute(&first, &data, RefuteSuite::Cheap, &ctx).unwrap_err();
    match err {
        antecedent::CausalError::Support { id, message } => {
            assert_eq!(id, antecedent::SupportRefusal::Refused);
            assert!(message.contains("AverageEffect"), "{message}");
        }
        other => panic!("expected Support::Refused, got {other:?}"),
    }
}
