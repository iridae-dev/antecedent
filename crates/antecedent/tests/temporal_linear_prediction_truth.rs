//! Known-truth checks of the compiled lagged linear prediction.
//!
//! The compiled `TemporalLinearPrediction` program regresses a target on listed lagged columns
//! and evaluates the fitted line with one variable's lags held at a level. Each check builds a
//! series from a declared linear lag law and compares what the compiled program returns with the
//! values computed analytically from that law in the test: the generating intercept and slopes,
//! and the conditional prediction `a + b * level + c * z_{t-2}` on every lag-aligned row. A held
//! variable outside the design and a non-finite level are refused, never answered; a cancelled
//! run stops before the fit or before the prediction with no answer.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    reason = "the fixtures build columns from small time indices"
)]

use std::sync::Arc;

use antecedent::analysis::temporal_linear_prediction::{
    TemporalPredictionError, TemporalPredictionRequest, compile_temporal_prediction,
};
use antecedent_core::{
    CancellationToken, ExecutionContext, KernelPolicy, Lag, StreamDomain, VariableId,
};
use antecedent_data::{LaggedColumn, TimeSeriesData};
use antecedent_kernels::standard_normal;

const INTERCEPT: f64 = 1.5;
const X_SLOPE: f64 = 2.0;
const Z_SLOPE: f64 = -0.5;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(41)
}

fn var(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

/// `x` at lag 1 and `z` at lag 2, in coefficient order.
fn parents() -> Arc<[LaggedColumn]> {
    Arc::from([
        LaggedColumn { variable: var(0), lag: Lag::from_raw(1) },
        LaggedColumn { variable: var(1), lag: Lag::from_raw(2) },
    ])
}

/// Deterministic, non-collinear driver columns.
fn drivers(n: usize) -> (Vec<f64>, Vec<f64>) {
    let x = (0..n).map(|t| ((t * 7) % 11) as f64).collect();
    let z = (0..n).map(|t| 0.3 * ((t * 5) % 13) as f64).collect();
    (x, z)
}

/// `y_t = a + b x_{t-1} + c z_{t-2} + noise_t` for `t >= 2`; the first two points have no
/// complete lag window and are never fitted.
fn series(x: &[f64], z: &[f64], noise: &[f64]) -> TimeSeriesData {
    let y: Vec<f64> = (0..x.len())
        .map(|t| {
            if t < 2 { 0.0 } else { INTERCEPT + X_SLOPE * x[t - 1] + Z_SLOPE * z[t - 2] + noise[t] }
        })
        .collect();
    TimeSeriesData::from_f64_columns([("x", x), ("z", z), ("y", y.as_slice())], 1).unwrap()
}

fn predict(data: TimeSeriesData, held: VariableId, level: f64) -> TemporalPredictionRequest {
    TemporalPredictionRequest::predict_conditional(
        data,
        var(2),
        parents(),
        held,
        level,
        KernelPolicy::default_policy(),
    )
}

#[test]
fn a_noise_free_lag_law_is_recovered_with_its_conditional_predictions() {
    let n = 60;
    let (x, z) = drivers(n);
    let data = series(&x, &z, &vec![0.0; n]);
    let level = 3.25;
    let compiled = compile_temporal_prediction(predict(data, var(0), level), &ctx()).unwrap();
    assert_eq!(compiled.lowered().route(), "TemporalLinearPrediction");
    let outcome = compiled.run(&ctx()).unwrap();
    let coefficients = &outcome.predictor().coefficients;
    for (got, want) in coefficients.iter().zip([INTERCEPT, X_SLOPE, Z_SLOPE]) {
        assert!((got - want).abs() < 1e-9, "coefficients {coefficients:?}");
    }
    let predictions = outcome.predictions().expect("a prediction request returns predictions");
    // One prediction per lag-aligned row t = 2..n, with x held at the level.
    assert_eq!(predictions.len(), n - 2);
    let mut truth_sum = 0.0;
    for (row, got) in predictions.iter().enumerate() {
        let t = row + 2;
        let want = INTERCEPT + X_SLOPE * level + Z_SLOPE * z[t - 2];
        truth_sum += want;
        assert!((got - want).abs() < 1e-9, "row {row}: {got} vs {want}");
    }
    let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
    let truth_mean = truth_sum / (n - 2) as f64;
    assert!((mean - truth_mean).abs() < 1e-9, "mean {mean} vs {truth_mean}");
}

#[test]
fn a_noisy_lag_law_is_recovered_within_its_sampling_error() {
    let n = 4000;
    let (x, z) = drivers(n);
    let mut rng = ExecutionContext::for_tests(17).rng.stream_for(StreamDomain::Test, 0x7A11);
    let noise: Vec<f64> = (0..n).map(|_| 0.2 * standard_normal(&mut rng)).collect();
    let data = series(&x, &z, &noise);
    let level = -1.0;
    let outcome = compile_temporal_prediction(predict(data, var(1), level), &ctx())
        .unwrap()
        .run(&ctx())
        .unwrap();
    let coefficients = &outcome.predictor().coefficients;
    // Noise sd 0.2 over ~4000 rows: every coefficient's standard error is below 0.01.
    for (got, want) in coefficients.iter().zip([INTERCEPT, X_SLOPE, Z_SLOPE]) {
        assert!((got - want).abs() < 0.05, "coefficients {coefficients:?}");
    }
    // Holding z: the mean prediction is a + b mean(x_{t-1}) + c level over the fitted rows.
    let predictions = outcome.predictions().unwrap();
    let mean = predictions.iter().sum::<f64>() / predictions.len() as f64;
    let mean_lagged_x = x[1..n - 1].iter().sum::<f64>() / (n - 2) as f64;
    let truth = INTERCEPT + X_SLOPE * mean_lagged_x + Z_SLOPE * level;
    assert!((mean - truth).abs() < 0.05, "mean {mean} vs {truth}");
}

#[test]
fn a_held_variable_outside_the_design_or_a_non_finite_level_is_refused() {
    let n = 30;
    let (x, z) = drivers(n);
    let data = series(&x, &z, &vec![0.0; n]);
    // The target is not one of the predictor's lagged columns.
    let outside = compile_temporal_prediction(predict(data.clone(), var(2), 1.0), &ctx())
        .unwrap()
        .run(&ctx());
    assert!(
        matches!(outside, Err(TemporalPredictionError::Estimation(_))),
        "a held variable outside the design is refused: {outside:?}"
    );
    let infinite = compile_temporal_prediction(predict(data, var(0), f64::INFINITY), &ctx())
        .unwrap()
        .run(&ctx());
    assert!(
        matches!(infinite, Err(TemporalPredictionError::Estimation(_))),
        "a non-finite level is refused: {infinite:?}"
    );
}

#[test]
fn a_cancelled_run_stops_without_an_answer_and_the_same_request_then_runs() {
    let n = 30;
    let (x, z) = drivers(n);
    let compiled =
        compile_temporal_prediction(predict(series(&x, &z, &vec![0.0; n]), var(0), 1.0), &ctx())
            .unwrap();
    // Cancelled before the fit.
    let before = ctx();
    before.cancellation.cancel();
    assert!(matches!(compiled.run(&before), Err(TemporalPredictionError::Cancelled)));
    // The first poll passes and the second, between the fit and the prediction, cancels.
    let mut between = ctx();
    between.cancellation = CancellationToken::cancel_after_checks(1);
    assert!(matches!(compiled.run(&between), Err(TemporalPredictionError::Cancelled)));
    let outcome = compiled.run(&ctx()).unwrap();
    assert_eq!(outcome.predictions().map(|p| p.len()), Some(n - 2));
}
