//! Feature relevance against closed-form Shapley values of known linear structural models.
//!
//! The value of a coalition `S` is `E[Y | do(X_i = mean_i + delta for i in S)]`. On a linear
//! model with an exactly fitted outcome mechanism that value is an affine function of the
//! intervened levels, so each Shapley value follows from the definition by hand:
//!
//! * additive game, `y = a x + b z` with independent roots: `phi_x = a delta`, `phi_z = b delta`;
//! * mediated game, `m = x`, `y = c m`: setting `m` severs `x`, so `x` matters only when `m` is
//!   free; the test works out `phi_x = phi_m`.
//!
//! The tolerances cover the Monte Carlo error of the sampled root means, which common random
//! numbers keep identical across coalitions but do not remove.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_attribution::{FeatureRelevanceOptions, feature_relevance};
use antecedent_core::{
    CausalSchemaBuilder, ExecutionContext, MeasurementSpec, RoleHint, SmallRoleSet, ValueType,
    VariableId,
};
use antecedent_data::column::{Float64Column, ValidityBitmap};
use antecedent_data::{OwnedColumn, OwnedColumnarStorage, TableView, TabularData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{CompiledCausalModel, MechanismRegistry, SelectionPolicy};

fn build(
    columns: Vec<(&str, Vec<f64>)>,
    edges: &[(u32, u32)],
) -> (CompiledCausalModel, TabularData) {
    let n = columns[0].1.len();
    let mut b = CausalSchemaBuilder::new();
    for (name, _) in &columns {
        b.add_variable(
            *name,
            ValueType::Continuous,
            SmallRoleSet::from_hint(RoleHint::Context),
            None,
            None,
            MeasurementSpec::default(),
        )
        .unwrap();
    }
    let schema = b.build().unwrap();
    let validity = ValidityBitmap::all_valid(n);
    let cols = columns
        .into_iter()
        .enumerate()
        .map(|(i, (_, values))| {
            OwnedColumn::Float64(
                Float64Column::new(
                    VariableId::from_raw(i as u32),
                    Arc::from(values),
                    validity.clone(),
                )
                .unwrap(),
            )
        })
        .collect();
    let data = TabularData::new(OwnedColumnarStorage::try_new(schema, cols, None, None).unwrap());
    let mut g = Dag::with_variables(data.schema().len() as u32);
    for &(a, b) in edges {
        g.insert_directed(DenseNodeId::from_raw(a), DenseNodeId::from_raw(b)).unwrap();
    }
    let compiled = CompiledCausalModel::compile(g).unwrap();
    let (store, _) = MechanismRegistry::standard()
        .assign_and_fit(&compiled, &data, SelectionPolicy::BestScore)
        .unwrap();
    (compiled.with_mechanisms(store), data)
}

/// Deterministic, roughly standard-normal-scaled, uncorrelated series.
fn series(n: usize, freq: f64, phase: f64) -> Vec<f64> {
    (0..n).map(|i| 2.0 * ((i as f64) * freq + phase).sin()).collect()
}

fn scores(
    model: &CompiledCausalModel,
    data: &TabularData,
    outcome: u32,
    features: &[u32],
    delta: f64,
) -> Vec<(u32, f64)> {
    let ids: Vec<_> = features.iter().map(|&f| VariableId::from_raw(f)).collect();
    feature_relevance(
        model,
        data,
        VariableId::from_raw(outcome),
        &ids,
        &FeatureRelevanceOptions { delta, n_samples: 40_000, max_features: 8 },
        &ExecutionContext::for_tests(11),
    )
    .unwrap()
    .into_iter()
    .map(|r| (r.feature.raw(), r.score))
    .collect()
}

fn score_of(scores: &[(u32, f64)], feature: u32) -> f64 {
    scores.iter().find(|(f, _)| *f == feature).unwrap().1
}

#[test]
fn an_additive_linear_game_gives_each_feature_its_own_coefficient_times_the_shift() {
    let n = 400;
    let x = series(n, 0.37, 0.1);
    let z = series(n, 1.13, 0.7);
    let y: Vec<f64> = x.iter().zip(&z).map(|(a, b)| 2.0 * a - 3.0 * b).collect();
    let (model, data) = build(vec![("x", x), ("z", z), ("y", y)], &[(0, 2), (1, 2)]);
    for delta in [1.0, 2.5] {
        let s = scores(&model, &data, 2, &[0, 1], delta);
        assert!((score_of(&s, 0) - 2.0 * delta).abs() < 0.1, "{s:?}");
        // The score is the absolute Shapley value: the negative coefficient is reported positive.
        assert!((score_of(&s, 1) - 3.0 * delta).abs() < 0.1, "{s:?}");
        assert_eq!(s[0].0, 1, "ranked by score, largest first");
    }
}

#[test]
fn intervening_on_a_mediator_severs_its_cause_and_splits_the_effect() {
    // m = x and y = 2 m. v(empty) = E y; v(x) = 2 (mx + d); v(m) = v(x, m) = 2 (mm + d) with
    // mm = mx. phi_x = (2d + 0) / 2 and phi_m = (2d + 0) / 2 add to 2 d (efficiency), so both
    // equal d.
    let n = 400;
    let x = series(n, 0.37, 0.1);
    let m = x.clone();
    let y: Vec<f64> = m.iter().map(|v| 2.0 * v).collect();
    let (model, data) = build(vec![("x", x), ("m", m), ("y", y)], &[(0, 1), (1, 2)]);
    let s = scores(&model, &data, 2, &[0, 1], 1.5);
    assert!((score_of(&s, 0) - 1.5).abs() < 0.1, "{s:?}");
    assert!((score_of(&s, 1) - 1.5).abs() < 0.1, "{s:?}");
}

#[test]
fn a_feature_that_does_not_reach_the_outcome_scores_zero() {
    let n = 400;
    let x = series(n, 0.37, 0.1);
    let z = series(n, 1.13, 0.7);
    let y: Vec<f64> = x.iter().map(|a| 2.0 * a).collect();
    let (model, data) = build(vec![("x", x), ("z", z), ("y", y)], &[(0, 2)]);
    let s = scores(&model, &data, 2, &[0, 1], 1.0);
    assert!(score_of(&s, 1) < 0.05, "{s:?}");
    assert!((score_of(&s, 0) - 2.0).abs() < 0.1, "{s:?}");
}

#[test]
fn more_features_than_the_declared_limit_are_refused() {
    let n = 100;
    let x = series(n, 0.37, 0.1);
    let y = x.clone();
    let (model, data) = build(vec![("x", x), ("y", y)], &[(0, 1)]);
    let result = feature_relevance(
        &model,
        &data,
        VariableId::from_raw(1),
        &[VariableId::from_raw(0)],
        &FeatureRelevanceOptions { delta: 1.0, n_samples: 10, max_features: 0 },
        &ExecutionContext::for_tests(1),
    );
    assert!(result.is_err());
}

#[test]
fn a_repeated_feature_or_the_outcome_among_the_features_is_not_scored_silently() {
    let n = 100;
    let x = series(n, 0.37, 0.1);
    let y: Vec<f64> = x.iter().map(|a| 2.0 * a).collect();
    let (model, data) = build(vec![("x", x), ("y", y)], &[(0, 1)]);
    let options = FeatureRelevanceOptions { delta: 1.0, n_samples: 100, max_features: 8 };
    let ctx = ExecutionContext::for_tests(1);
    let x = VariableId::from_raw(0);
    let y = VariableId::from_raw(1);
    let repeated = feature_relevance(&model, &data, y, &[x, x], &options, &ctx);
    assert!(repeated.is_err(), "a repeated feature would split its effect: {repeated:?}");
    let outcome = feature_relevance(&model, &data, y, &[x, y], &options, &ctx);
    assert!(outcome.is_err(), "the outcome is not a feature of itself: {outcome:?}");
}
