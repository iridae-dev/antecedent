//! Phase 7 GCM conformance: fit/intervene, anomaly, counterfactual ITE.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, clippy::many_single_char_names)]

use std::fs;
use std::path::PathBuf;
use std::sync::Arc;

use causal::{
    anomaly_attribution, counterfactual_ite, fit_gcm, sample_do, arrow_strengths,
    streaming_matches_retained, CounterfactualEngine, CounterfactualWorld, KdeDoSampler,
    McmcDoSampler, MechanismWorkspace, WeightingDoSampler,
};
use causal_core::{
    CausalRng, CausalSchemaBuilder, ExecutionContext, Intervention, MeasurementSpec, RoleHint,
    SmallRoleSet, Value, ValueType, VariableId,
};
use causal_data::{Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap};
use causal_graph::{Dag, DenseNodeId};
use serde_json::Value as JsonValue;

fn fixture_dir(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/phase7")
        .join(name)
}

fn load_expected(name: &str) -> JsonValue {
    let raw = fs::read_to_string(fixture_dir(name).join("expected.json")).expect("expected.json");
    serde_json::from_str(&raw).expect("parse expected.json")
}

fn chain_data(n: usize, plant_outlier: bool) -> (TabularData, Dag) {
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
    let xv: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    let mut yv: Vec<f64> = xv.iter().map(|x| 1.0 + 2.0 * x).collect();
    if plant_outlier {
        yv[n - 1] = 100.0;
    }
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(xv), validity.clone()).unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(yv), validity).unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut g = Dag::with_variables(2);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))
        .unwrap();
    (data, g)
}

#[test]
fn gcm_fit_intervene() {
    let expected = load_expected("gcm_fit_intervene");
    let true_mean = expected["true_interventional_mean_y"].as_f64().unwrap();
    let tol = expected["tolerance"].as_f64().unwrap();
    let (data, g) = chain_data(40, false);
    let fitted = fit_gcm(g, &data).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let mut rng = CausalRng::from_seed(7);
    let batch = sample_do(
        &fitted.model,
        &[Intervention::set(VariableId::from_raw(0), Value::f64(3.0))],
        200,
        &mut rng,
        &ctx,
    )
    .unwrap();
    let y = batch.column(1).unwrap();
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    assert!((mean - true_mean).abs() < tol, "mean={mean} true={true_mean}");
}

#[test]
fn gcm_anomaly() {
    let expected = load_expected("gcm_anomaly");
    let idx = expected["planted_outlier_index"].as_u64().unwrap() as usize;
    let (data, g) = chain_data(30, true);
    let fitted = fit_gcm(g, &data).unwrap();
    let scores = anomaly_attribution(&fitted.model, &data, [VariableId::from_raw(1)], 100).unwrap();
    assert!(scores[0].scores[idx] > scores[0].scores[0]);
    let arrows = arrow_strengths(&fitted.model).unwrap();
    assert!(arrows.iter().any(|a| a.strength > 0.5));
}

#[test]
fn gcm_cf_ite() {
    let expected = load_expected("gcm_cf_ite");
    let true_ite = expected["true_mean_ite"].as_f64().unwrap();
    let tol = expected["tolerance"].as_f64().unwrap();
    let (data, g) = chain_data(50, false);
    let fitted = fit_gcm(g, &data).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let ite = counterfactual_ite(
        fitted.model.clone(),
        &data,
        VariableId::from_raw(0),
        VariableId::from_raw(1),
        1.0,
        0.0,
        &ctx,
    )
    .unwrap();
    assert!(
        (ite.mean_ite - true_ite).abs() < tol,
        "mean_ite={} true={true_ite}",
        ite.mean_ite
    );
    assert_eq!(format!("{:?}", ite.noise_inference), "Invertible");

    let engine = CounterfactualEngine::new(fitted.model);
    let exo = engine.abduct(&data, false).unwrap();
    let mut ws = MechanismWorkspace::default();
    let worlds = [CounterfactualWorld {
        unit_rows: None,
        interventions: Arc::from([Intervention::set(
            VariableId::from_raw(0),
            Value::f64(1.0),
        )]),
    }];
    let res = engine
        .predict(&exo, &worlds, &[VariableId::from_raw(1)], false, &mut ws, &ctx)
        .unwrap();
    assert!(streaming_matches_retained(&res, 0, DenseNodeId::from_raw(1)));
    assert!(expected["streaming_equiv_retained"].as_bool().unwrap());
}

fn binary_treatment_scm(n: usize) -> (causal::CompiledCausalModel, TabularData) {
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
    let schema = b.build().unwrap();
    let mut t = vec![0.0; n];
    let mut y = vec![0.0; n];
    for i in 0..n {
        t[i] = if i % 2 == 0 { 1.0 } else { 0.0 };
        y[i] = 2.0 * t[i];
    }
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(t), validity.clone()).unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(y), validity).unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut g = Dag::with_variables(2);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1))
        .unwrap();
    let fitted = fit_gcm(g, &data).unwrap();
    (fitted.model, data)
}

#[test]
fn do_sampling_weighting() {
    let expected = load_expected("do_sampling_weighting");
    let true_mean = expected["expected_treated_mean"].as_f64().unwrap();
    let (model, data) = binary_treatment_scm(80);
    let ctx = ExecutionContext::for_tests(1);
    let sampler = WeightingDoSampler::new(VariableId::from_raw(0), VariableId::from_raw(1));
    let res = sampler.estimate(&model, &data, 1.0, &ctx).unwrap();
    let mean = WeightingDoSampler::weighted_mean(&res);
    assert!((mean - true_mean).abs() < 1e-9, "mean={mean} expected={true_mean}");
}

#[test]
fn do_sampling_kde() {
    let expected = load_expected("do_sampling_kde");
    let n_draws = expected["n_draws"].as_u64().unwrap() as usize;
    let (model, _) = binary_treatment_scm(80);
    let ctx = ExecutionContext::for_tests(1);
    let mut rng = CausalRng::from_seed(3);
    let mut ws = MechanismWorkspace::default();
    let iv = [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))];
    let kde = KdeDoSampler::new(VariableId::from_raw(1))
        .sample(&model, &iv, n_draws, &mut rng, &mut ws, &ctx)
        .unwrap();
    assert_eq!(kde.values.len(), n_draws);
    if expected["require_finite_draws"].as_bool().unwrap() {
        assert!(kde.values.iter().all(|v| v.is_finite()));
    }
}

#[test]
fn do_sampling_mcmc() {
    let expected = load_expected("do_sampling_mcmc");
    let n_samples = expected["n_samples"].as_u64().unwrap() as usize;
    let (model, _) = binary_treatment_scm(80);
    let ctx = ExecutionContext::for_tests(1);
    let mut rng = CausalRng::from_seed(3);
    let mut ws = MechanismWorkspace::default();
    let iv = [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))];
    let mcmc = McmcDoSampler::new(VariableId::from_raw(1))
        .sample(&model, &iv, n_samples, &mut rng, &mut ws, &ctx)
        .unwrap();
    assert_eq!(mcmc.values.len(), n_samples);
    if expected["require_positive_accept_rate"].as_bool().unwrap() {
        assert!(mcmc.accept_rate.unwrap() > 0.0);
    }
}
