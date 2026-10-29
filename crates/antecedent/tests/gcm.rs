//! GCM conformance: fit/intervene, anomaly, counterfactual ITE.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

use std::path::PathBuf;
use std::sync::Arc;

use antecedent::gcm::{
    AbductionMissingPolicy, CounterfactualEngine, CounterfactualWorld, KdeDoSampler, McmcDoSampler,
    MechanismWorkspace, WeightingDoSampler, anomaly_attribution, arrow_strengths,
    counterfactual_ite, fit_gcm, sample_do,
};
use antecedent_core::{
    CausalRng, CausalSchemaBuilder, ExecutionContext, Intervention, MeasurementSpec, RoleHint,
    SmallRoleSet, Value, ValueType, VariableId,
};
use antecedent_data::{
    Float64Column, OwnedColumn, OwnedColumnarStorage, TabularData, ValidityBitmap,
};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_test_support::{conformance_dir, load_expected_json};

fn fixture_dir(name: &str) -> PathBuf {
    conformance_dir("gcm").join(name)
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
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    (data, g)
}

#[test]
fn gcm_fit_intervene() {
    let expected = load_expected_json(&fixture_dir("gcm_fit_intervene"));
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
    // The 40 points are noiseless, so the fitted mechanism is y = 1 + 2x to rounding and the
    // declared band is loose; hold the sampled mean to the closed form far tighter.
    assert!((mean - true_mean).abs() < 1e-6, "mean={mean} true={true_mean}");
}

#[test]
fn gcm_interventional_distribution_query() {
    use antecedent::gcm::sample_interventional_distribution;
    use antecedent_core::InterventionalDistributionQuery;

    let expected = load_expected_json(&fixture_dir("gcm_fit_intervene"));
    let true_mean = expected["true_interventional_mean_y"].as_f64().unwrap();
    let tol = expected["tolerance"].as_f64().unwrap();
    let (data, g) = chain_data(40, false);
    let fitted = fit_gcm(g, &data).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let mut rng = CausalRng::from_seed(7);
    let q = InterventionalDistributionQuery::new(
        VariableId::from_raw(1),
        [Intervention::set(VariableId::from_raw(0), Value::f64(3.0))],
    );
    let batch = sample_interventional_distribution(&fitted.model, &q, 200, &mut rng, &ctx).unwrap();
    let y = batch.column(1).unwrap();
    let mean = y.iter().sum::<f64>() / y.len() as f64;
    assert!((mean - true_mean).abs() < tol, "mean={mean} true={true_mean}");
    assert!((mean - true_mean).abs() < 1e-6, "mean={mean} true={true_mean}");
}

/// Additive shift `do(X := X + delta)` moves the downstream mean by `coefficient * delta`
/// relative to the (unintervened) baseline, whereas a hard `do(X := delta)` pins the
/// downstream mean to `coefficient * delta` regardless of baseline — the two intervention
/// kinds are observably different, which is the point of exposing `Intervention::Shift`
/// alongside `Intervention::Set`.
#[test]
fn gcm_shift_intervention_differs_from_hard_set() {
    let (data, g) = chain_data(40, false);
    let fitted = fit_gcm(g, &data).unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let x = VariableId::from_raw(0);
    let delta = 5.0;

    let mut rng = CausalRng::from_seed(11);
    let baseline = sample_do(&fitted.model, &[], 200, &mut rng, &ctx).unwrap();
    let baseline_y = baseline.column(1).unwrap();
    let baseline_mean = baseline_y.iter().sum::<f64>() / baseline_y.len() as f64;

    let mut rng = CausalRng::from_seed(11);
    let shifted =
        sample_do(&fitted.model, &[Intervention::shift(x, Value::f64(delta))], 200, &mut rng, &ctx)
            .unwrap();
    let shifted_y = shifted.column(1).unwrap();
    let shifted_mean = shifted_y.iter().sum::<f64>() / shifted_y.len() as f64;

    // y = 1 + 2x (fitted); shifting x by delta should move y's mean by ~2*delta.
    let observed_delta = shifted_mean - baseline_mean;
    let expected_delta = 2.0 * delta;
    assert!(
        (observed_delta - expected_delta).abs() < 1e-6,
        "observed={observed_delta} expected={expected_delta}"
    );

    let mut rng = CausalRng::from_seed(11);
    let hard =
        sample_do(&fitted.model, &[Intervention::set(x, Value::f64(delta))], 200, &mut rng, &ctx)
            .unwrap();
    let hard_y = hard.column(1).unwrap();
    let hard_mean = hard_y.iter().sum::<f64>() / hard_y.len() as f64;

    // Hard set pins x = delta regardless of its baseline distribution; shift instead adds
    // delta on top of the baseline x. On this fixture the two land far apart.
    assert!(
        (hard_mean - shifted_mean).abs() > 1.0,
        "hard={hard_mean} shifted={shifted_mean} should differ"
    );
    // The hard set pins x = 5, so y = 1 + 2 * 5 exactly for the noiseless fit.
    assert!((hard_mean - 11.0).abs() < 1e-6, "hard={hard_mean}");
}

#[test]
fn gcm_path_specific_query() {
    use antecedent::gcm::attribute_path_specific;
    use antecedent_core::PathSpecificEffectQuery;
    use antecedent_model::{CompiledCausalModel, MechanismRegistry, SelectionPolicy};

    let n = 40usize;
    let mut b = CausalSchemaBuilder::new();
    for name in ["t", "m", "y"] {
        b.add_variable(
            name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::OutcomeCandidate),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let tv: Vec<f64> = (0..n).map(|i| i as f64 * 0.1).collect();
    let mv: Vec<f64> = tv.iter().map(|t| 0.5 * t).collect();
    let yv: Vec<f64> = mv.iter().zip(tv.iter()).map(|(m, t)| m + 0.25 * t).collect();
    let validity = ValidityBitmap::all_valid(n);
    let cols = vec![
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(0), Arc::from(tv), validity.clone()).unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(1), Arc::from(mv), validity.clone()).unwrap(),
        ),
        OwnedColumn::Float64(
            Float64Column::new(VariableId::from_raw(2), Arc::from(yv), validity).unwrap(),
        ),
    ];
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut g = Dag::with_variables(3);
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    g.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
    let compiled = CompiledCausalModel::compile(g).unwrap();
    let (store, _) = MechanismRegistry::standard()
        .assign_and_fit(&compiled, &data, SelectionPolicy::BestScore)
        .unwrap();
    let model = compiled.with_mechanisms(store);
    let q = PathSpecificEffectQuery::binary(VariableId::from_raw(0), VariableId::from_raw(2))
        .with_path_nodes([VariableId::from_raw(1)]);
    let result = attribute_path_specific(&model, &q, &ExecutionContext::for_tests(1)).unwrap();
    // One directed path t -> m -> y. The m mechanism is m = 0.5 t and y = m + 0.25 t = 1.5 m
    // on the data, so the binary (0 -> 1) contribution along the path is 0.5 * 1.5.
    assert_eq!(result.path_breakdown.len(), 1);
    let ids: Vec<u32> = result.path_breakdown[0].path.iter().map(|v| v.raw()).collect();
    assert_eq!(ids, [0, 1, 2]);
    assert!((result.path_breakdown[0].contribution - 0.75).abs() < 1e-9);
    assert!((result.total_change - 0.75).abs() < 1e-9, "{}", result.total_change);
}

#[test]
fn gcm_anomaly() {
    let expected = load_expected_json(&fixture_dir("gcm_anomaly"));
    let idx = expected["planted_outlier_index"].as_u64().unwrap() as usize;
    let (data, g) = chain_data(30, true);
    let fitted = fit_gcm(g, &data).unwrap();
    let scores = anomaly_attribution(&fitted.model, &data, [VariableId::from_raw(1)], 100).unwrap();
    assert!(scores[0].scores[idx] > scores[0].scores[0]);
    // Reference values from the definition: score = -ln P(|Z| >= |z|), z = (y - median) /
    // (1.4826 MAD). y = 1 + 0.2 i for i < 29 and 100 at the plant: the median is 3.9 and the
    // MAD 1.5, so rows 7 and 22 sit exactly one MAD out, z = 1/1.4826 = 0.6745, tail 1/2.
    let s = &scores[0].scores;
    for row in [7, 22] {
        assert!((s[row] - std::f64::consts::LN_2).abs() < 1e-3, "row {row}: {}", s[row]);
    }
    assert!(s[idx] > 900.0, "planted outlier score {}", s[idx]);
    assert!(s.iter().enumerate().all(|(i, v)| i == idx || *v < 2.0));
    // The median rows 14 and 15 are 0.1 from the centre, z = 0.1 / (1.4826 * 1.5).
    assert!(s[14] < 0.05 && s[15] < 0.05);
    let arrows = arrow_strengths(&fitted.model).unwrap();
    assert_eq!(arrows.len(), 1);
    // Least squares on the contaminated series: beta = cov(x, y) / var(x), variance
    // contribution beta^2 var(x) (sample variance of x = 0.1 i), residual variance with two
    // fitted parameters, strength = 1/2 ln(1 + beta^2 var(x) / sigma^2).
    let n = 30.0_f64;
    let xs: Vec<f64> = (0..30).map(|i| f64::from(i) * 0.1).collect();
    let ys: Vec<f64> =
        xs.iter().enumerate().map(|(i, x)| if i == 29 { 100.0 } else { 1.0 + 2.0 * x }).collect();
    let (mx, my) = (xs.iter().sum::<f64>() / n, ys.iter().sum::<f64>() / n);
    let sxx: f64 = xs.iter().map(|x| (x - mx).powi(2)).sum();
    let sxy: f64 = xs.iter().zip(&ys).map(|(x, y)| (x - mx) * (y - my)).sum();
    let beta = sxy / sxx;
    let rss: f64 = xs.iter().zip(&ys).map(|(x, y)| (y - my - beta * (x - mx)).powi(2)).sum();
    let vc = beta * beta * sxx / (n - 1.0);
    let strength = 0.5 * (1.0 + vc / (rss / (n - 2.0))).ln();
    assert!((arrows[0].coefficient - beta).abs() < 1e-9, "{}", arrows[0].coefficient);
    assert!((arrows[0].variance_contribution - vc).abs() < 1e-9);
    assert!((arrows[0].strength - strength).abs() < 1e-9, "{} vs {strength}", arrows[0].strength);
}

#[test]
fn gcm_cf_ite() {
    let expected = load_expected_json(&fixture_dir("gcm_cf_ite"));
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
    assert!((ite.mean_ite - true_ite).abs() < tol, "mean_ite={} true={true_ite}", ite.mean_ite);
    // Noiseless y = 1 + 2x is recovered by an invertible mechanism, so the ITE is exact.
    assert!((ite.mean_ite - true_ite).abs() < 1e-9, "mean_ite={}", ite.mean_ite);
    assert_eq!(format!("{:?}", ite.noise_inference), "Invertible");

    let engine = CounterfactualEngine::new(fitted.model);
    let exo = engine
        .abduct(&data, AbductionMissingPolicy::Error, &ExecutionContext::for_tests(1))
        .unwrap();
    let mut ws = MechanismWorkspace::default();
    let worlds = [CounterfactualWorld {
        unit_rows: None,
        interventions: Arc::from([Intervention::set(VariableId::from_raw(0), Value::f64(1.0))]),
    }];
    let res =
        engine.predict(&exo, &worlds, &[VariableId::from_raw(1)], false, &mut ws, &ctx).unwrap();
    // The streaming mean equals the mean of the retained outcome column, read independently.
    let o = DenseNodeId::from_raw(1);
    let column = res.outcome_column(0, o).unwrap();
    let retained = column.iter().sum::<f64>() / column.len() as f64;
    assert!((res.streaming_outcome_mean(0, o) - retained).abs() < 1e-12);
    assert!(expected["streaming_equiv_retained"].as_bool().unwrap());
}

fn binary_treatment_scm(n: usize) -> (antecedent::gcm::CompiledCausalModel, TabularData) {
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
    g.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    let fitted = fit_gcm(g, &data).unwrap();
    (fitted.model, data)
}

#[test]
fn do_sampling_weighting() {
    let expected = load_expected_json(&fixture_dir("do_sampling_weighting"));
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
    let expected = load_expected_json(&fixture_dir("do_sampling_kde"));
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
    // y = 2 t with no noise: do(t = 1) is a point mass at 2.
    let treated = load_expected_json(&fixture_dir("do_sampling_weighting"))["expected_treated_mean"]
        .as_f64()
        .unwrap();
    assert!(kde.values.iter().all(|v| (v - treated).abs() < 1e-6), "{:?}", kde.values);
}

#[test]
fn do_sampling_mcmc() {
    let expected = load_expected_json(&fixture_dir("do_sampling_mcmc"));
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
    let treated = load_expected_json(&fixture_dir("do_sampling_weighting"))["expected_treated_mean"]
        .as_f64()
        .unwrap();
    assert!(mcmc.values.iter().all(|v| (v - treated).abs() < 1e-6), "{:?}", mcmc.values);
}
