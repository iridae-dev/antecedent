//! Phase 9 conformance pins: load every `conformance/phase9/*/expected.json`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss)]

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use causal::{
    ConditionalLinearAdjustment, JpcmciPlus, Rpcmci, TemporalLinearPredictor,
    TemporalMediationEstimator, two_regime_half_split,
};
use causal_core::{
    CausalSchemaBuilder, ConditionalEffectQuery, ExecutionContext, Lag, MeasurementSpec,
    MediationContrast, MediationQuery, RoleHint, SmallRoleSet, Value, ValueType, VariableId,
};
use causal_data::{
    Float64Column, LaggedColumn, MultiEnvironmentData, OwnedColumn, OwnedColumnarStorage,
    SamplingRegularity, TableView, TabularData, TimeIndex, TimeSeriesData, ValidityBitmap,
};
use causal_discovery::{DiscoveryConstraints, DiscoveryWorkspace, PcmciPlus, TemporalConstraints};
use causal_expr::{CausalExprArena, IdentifiedEstimand};
use serde_json::Value as JsonValue;

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../conformance/phase9").join(name)
}

fn load_expected(name: &str) -> JsonValue {
    let raw = fs::read_to_string(fixture_dir(name).join("expected.json")).expect("expected.json");
    serde_json::from_str(&raw).expect("parse expected.json")
}

fn toy_env(n: usize, seed: f64) -> TimeSeriesData {
    let mut b = CausalSchemaBuilder::new();
    for name in ["x", "y"] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = 0.5 * x[t - 1] + 0.1 * ((t as f64) + seed).sin();
        y[t] = 0.7 * x[t] + 0.2 * y[t - 1] + 0.05 * ((t as f64) + seed).cos();
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
    TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap()
}

#[test]
fn jpcmci_plus_two_env_pin() {
    let expected = load_expected("jpcmci_plus_two_env");
    let multi =
        MultiEnvironmentData::try_new(Arc::from([toy_env(160, 0.0), toy_env(160, 1.0)])).unwrap();
    assert!(multi.env_count() >= expected["min_envs"].as_u64().unwrap() as usize);
    let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
    let alg = JpcmciPlus::new().with_fdr(false).with_constraints(DiscoveryConstraints {
        temporal: TemporalConstraints { max_lag: Lag::from_raw(1), min_lag: Lag::CONTEMPORANEOUS },
        alpha: 0.25,
        max_cond_size: 2,
        ..DiscoveryConstraints::default()
    });
    let mut ws = DiscoveryWorkspace::default();
    let result = alg.run(&multi, &vars, &mut ws, &ExecutionContext::for_tests(1)).unwrap();
    assert_eq!(result.algorithm.id.as_ref(), expected["algorithm_id"].as_str().unwrap());
    assert!(result.evidence.graph.node_count() >= expected["min_nodes"].as_u64().unwrap() as usize);
    assert!(result.diagnostics.iter().any(|d| d.code.as_ref() == "jpcmci_plus.multi_env_plan"));
}

#[test]
fn rpcmci_two_regime_pin() {
    let expected = load_expected("rpcmci_two_regime");
    let data = toy_env(200, 0.0);
    let vars = [VariableId::from_raw(0), VariableId::from_raw(1)];
    let assign = two_regime_half_split(data.row_count());
    let alg = Rpcmci::new().with_min_regime_len(40).with_pcmci_plus(
        PcmciPlus::new().with_fdr(false).with_constraints(DiscoveryConstraints {
            temporal: TemporalConstraints {
                max_lag: Lag::from_raw(1),
                min_lag: Lag::CONTEMPORANEOUS,
            },
            alpha: 0.25,
            max_cond_size: 2,
            ..DiscoveryConstraints::default()
        }),
    );
    let mut ws = DiscoveryWorkspace::default();
    let result = alg.run(&data, &vars, &assign, &mut ws, &ExecutionContext::for_tests(2)).unwrap();
    assert_eq!(result.algorithm.id.as_ref(), expected["algorithm_id"].as_str().unwrap());
    assert_eq!(result.graphs.len(), expected["n_regimes"].as_u64().unwrap() as usize);
}

#[test]
fn temporal_mediation_numeric_pin() {
    let expected = load_expected("temporal_mediation");
    let mediated_min = expected["mediated_min"].as_f64().unwrap();
    let decomp_tol = expected["decomposition_tol"].as_f64().unwrap();
    let n = 300usize;
    let mut b = CausalSchemaBuilder::new();
    for name in ["t", "m", "y"] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let mut t = vec![0.0; n];
    let mut m = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 1..n {
        t[i] = 0.3 * t[i - 1] + 0.1 * (i as f64).sin();
        m[i] = 0.8 * t[i - 1] + 0.05 * (i as f64).cos();
        y[i] = 0.5 * m[i] + 0.2 * t[i - 1] + 0.02 * (i as f64).sin();
    }
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(m), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(y), ValidityBitmap::all_valid(n))
                .unwrap(),
        ),
    ];
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let q = MediationQuery::binary(
        VariableId::from_raw(0),
        VariableId::from_raw(2),
        [VariableId::from_raw(1)],
        MediationContrast::Mediated,
    );
    let mut arena = CausalExprArena::new();
    let functional =
        arena.frontdoor_ate(q.treatment, q.outcome, &q.mediators, Value::f64(1.0), Value::f64(0.0));
    let estimand = IdentifiedEstimand::frontdoor(
        "temporal_mediation.mediated",
        Arc::clone(&q.mediators),
        functional,
    );
    let est = TemporalMediationEstimator::new()
        .estimate(&data, &estimand, &q, &ExecutionContext::for_tests(3))
        .unwrap();
    assert!(est.mediated.unwrap() > mediated_min);
    assert!((est.total.unwrap() - est.direct.unwrap() - est.mediated.unwrap()).abs() < decomp_tol);
}

#[test]
fn conditional_effect_pin() {
    let expected = load_expected("conditional_effect");
    let ate_target = expected["ate_target"].as_f64().unwrap();
    let ate_tol = expected["ate_tol"].as_f64().unwrap();
    let n = 200usize;
    let mut b = CausalSchemaBuilder::new();
    for name in ["t", "y", "w"] {
        b.add_variable(
            name,
            ValueType::Continuous,
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
    let storage = OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap();
    let data = TabularData::new(storage);
    let q = causal_core::AverageEffectQuery::binary_ate(
        VariableId::from_raw(0),
        VariableId::from_raw(1),
    )
    .with_effect_modifiers([VariableId::from_raw(2)]);
    let cq = ConditionalEffectQuery::try_new(q).unwrap();
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from([]),
        causal_expr::ExprId::from_raw(0),
    );
    let est = ConditionalLinearAdjustment::new()
        .estimate(&data, &estimand, &cq, &ExecutionContext::for_tests(4))
        .unwrap();
    assert!((est.ate - ate_target).abs() < ate_tol);
}

#[test]
fn prediction_smoke_pin() {
    let expected = load_expected("prediction_smoke");
    let target = expected["mean_prediction_target"].as_f64().unwrap();
    let tol = expected["tol"].as_f64().unwrap();
    let n = 80usize;
    let mut b = CausalSchemaBuilder::new();
    for name in ["x", "y"] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let mut x = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 1..n {
        x[t] = 0.5 * x[t - 1] + 0.1;
        y[t] = 2.0 * x[t - 1] + 0.01;
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
    let data = TimeSeriesData::try_new(
        storage,
        TimeIndex { regularity: SamplingRegularity::Regular { interval_ns: 1 }, length: n },
    )
    .unwrap();
    let pred = TemporalLinearPredictor::fit(
        &data,
        VariableId::from_raw(1),
        [LaggedColumn { variable: VariableId::from_raw(0), lag: Lag::from_raw(1) }],
    )
    .unwrap();
    let yhat = pred.predict_intervened(&data, VariableId::from_raw(0), 1.0).unwrap();
    let mean: f64 = yhat.iter().sum::<f64>() / yhat.len() as f64;
    assert!((mean - target).abs() < tol);
}
