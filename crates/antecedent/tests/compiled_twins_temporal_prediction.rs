//! The lagged linear predictor and its compiled route are the same computation.
//!
//! `TemporalLinearPredictor::fit` and `predict_conditional` are the direct path; the compiled
//! `TemporalLinearPrediction` program calls them with the same series, target, lagged columns,
//! held variable, level and kernel policy. Each test runs both on a seeded two-parent series and
//! compares, bit for bit, the fitted coefficients, the design the fit retained, every
//! conditional prediction and the mean prediction the Python summary reports. A kernel refusal
//! is the same refusal on both paths. The program claims no uncertainty, so no interval and no
//! calibration key exists to compare; neither path draws from the execution context's seed.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "the bits are compared exactly and the mean divides by a row count"
)]

use std::sync::Arc;

use antecedent::analysis::temporal_linear_prediction::{
    TemporalPredictionError, TemporalPredictionOutcome, TemporalPredictionRequest,
    compile_temporal_prediction,
};
use antecedent_core::{ExecutionContext, KernelPolicy, Lag, StreamDomain, VariableId};
use antecedent_data::{LaggedColumn, TimeSeriesData};
use antecedent_estimate::TemporalLinearPredictor;
use antecedent_kernels::standard_normal;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(29)
}

fn var(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn parents() -> Arc<[LaggedColumn]> {
    Arc::from([
        LaggedColumn { variable: var(0), lag: Lag::from_raw(1) },
        LaggedColumn { variable: var(1), lag: Lag::from_raw(3) },
    ])
}

/// A seeded series: `x` and `z` autoregressive, `y_t = 0.4 + 1.2 x_{t-1} - 0.7 z_{t-3} + e_t`.
fn series() -> TimeSeriesData {
    let n = 500;
    let mut rng = ExecutionContext::for_tests(613).rng.stream_for(StreamDomain::Test, 0x7E2);
    let mut x = vec![0.0; n];
    let mut z = vec![0.0; n];
    let mut y = vec![0.0; n];
    for t in 0..n {
        let previous = |column: &[f64]| if t == 0 { 0.0 } else { column[t - 1] };
        x[t] = 0.5 * previous(&x) + standard_normal(&mut rng);
        z[t] = -0.3 * previous(&z) + standard_normal(&mut rng);
        let lagged_x = if t >= 1 { x[t - 1] } else { 0.0 };
        let lagged_z = if t >= 3 { z[t - 3] } else { 0.0 };
        y[t] = 0.4 + 1.2 * lagged_x - 0.7 * lagged_z + 0.5 * standard_normal(&mut rng);
    }
    TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("z", z.as_slice()), ("y", y.as_slice())],
        1,
    )
    .unwrap()
}

fn bits(values: &[f64]) -> Vec<u64> {
    values.iter().map(|value| value.to_bits()).collect()
}

/// The mean prediction, computed as the Python summary computes it.
fn mean(predictions: &[f64]) -> f64 {
    predictions.iter().sum::<f64>() / predictions.len().max(1) as f64
}

fn assert_same_fit(
    label: &str,
    direct: &TemporalLinearPredictor,
    compiled: &TemporalLinearPredictor,
) {
    assert_eq!(direct.target, compiled.target, "{label}: target");
    assert_eq!(direct.parents, compiled.parents, "{label}: lagged columns");
    assert_eq!(direct.max_lag, compiled.max_lag, "{label}: deepest lag");
    assert_eq!(bits(&direct.coefficients), bits(&compiled.coefficients), "{label}: coefficients");
}

#[test]
fn conditional_prediction_equals_the_compiled_prediction() {
    let data = series();
    let policy = KernelPolicy::default_policy();
    for (held, level) in [(var(0), 1.0), (var(0), -2.5), (var(1), 0.75)] {
        let direct = TemporalLinearPredictor::fit(&data, var(2), parents(), &policy).unwrap();
        let direct_predictions = direct.predict_conditional(&data, held, level, &policy).unwrap();
        let compiled = compile_temporal_prediction(
            TemporalPredictionRequest::predict_conditional(
                data.clone(),
                var(2),
                parents(),
                held,
                level,
                policy,
            ),
            &ctx(),
        )
        .unwrap()
        .run(&ctx())
        .unwrap();
        let TemporalPredictionOutcome::PredictConditional { predictor, predictions } = &compiled
        else {
            panic!("a prediction request returns predictions");
        };
        let label = format!("held {held:?} at {level}");
        assert_same_fit(&label, &direct, predictor);
        assert_eq!(bits(&direct_predictions), bits(predictions), "{label}: predictions");
        assert_eq!(
            mean(&direct_predictions).to_bits(),
            mean(predictions).to_bits(),
            "{label}: mean prediction"
        );
    }
}

#[test]
fn the_fit_equals_the_compiled_fit() {
    let data = series();
    let policy = KernelPolicy::default_policy();
    let direct = TemporalLinearPredictor::fit(&data, var(2), parents(), &policy).unwrap();
    let compiled = compile_temporal_prediction(
        TemporalPredictionRequest::fit(data, var(2), parents(), policy),
        &ctx(),
    )
    .unwrap()
    .run(&ctx())
    .unwrap();
    assert!(compiled.predictions().is_none(), "a fit request returns no predictions");
    assert_same_fit("fit", &direct, compiled.predictor());
}

#[test]
fn a_kernel_refusal_is_the_same_refusal_on_both_paths() {
    let data = series();
    let policy = KernelPolicy::default_policy();
    let direct = TemporalLinearPredictor::fit(&data, var(2), parents(), &policy)
        .unwrap()
        .predict_conditional(&data, var(2), 1.0, &policy)
        .unwrap_err();
    let compiled = compile_temporal_prediction(
        TemporalPredictionRequest::predict_conditional(
            data,
            var(2),
            parents(),
            var(2),
            1.0,
            policy,
        ),
        &ctx(),
    )
    .unwrap()
    .run(&ctx())
    .unwrap_err();
    let TemporalPredictionError::Estimation(compiled) = compiled else {
        panic!("the kernel's own error: {compiled:?}");
    };
    assert_eq!(direct.to_string(), compiled.to_string());
}

#[test]
fn the_program_names_its_route_and_its_identity_is_the_question() {
    let policy = KernelPolicy::default_policy();
    let request = |level: f64| {
        TemporalPredictionRequest::predict_conditional(
            series(),
            var(2),
            parents(),
            var(0),
            level,
            policy,
        )
    };
    let one = compile_temporal_prediction(request(1.0), &ctx()).unwrap();
    let again = compile_temporal_prediction(request(1.0), &ctx()).unwrap();
    let other = compile_temporal_prediction(request(2.0), &ctx()).unwrap();
    let fit = compile_temporal_prediction(
        TemporalPredictionRequest::fit(series(), var(2), parents(), policy),
        &ctx(),
    )
    .unwrap();
    assert_eq!(one.lowered().route(), "TemporalLinearPrediction");
    assert_eq!(fit.lowered().route(), "TemporalLinearPrediction");
    assert_eq!(one.lowered().identity(), again.lowered().identity());
    assert_ne!(one.lowered().identity(), other.lowered().identity());
    assert_ne!(one.lowered().identity(), fit.lowered().identity());
}
