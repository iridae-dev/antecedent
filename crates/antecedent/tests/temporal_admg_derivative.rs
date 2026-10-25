//! A derivative at one horizon on a temporal ADMG, identified by an adjustment set of the
//! unrolled window of its template.
//!
//! The template has an observed confounder `z(t-1)` of the treatment `x(t-1)` and the outcome
//! `y(t)`, and a latent common cause of `w(t-1)` (a parent of `z`) and `y(t)`. The observed
//! confounder separates the treatment from that latent cause, so the structural slope is the
//! coefficient of `x(t-1)` in the regression of `y(t)` on `(1, x(t-1), z(t-1), w(t-1))`, which
//! is computed here by the normal equations, independently of the library. A template whose
//! treatment shares a latent cause with the outcome has a hedge on its window and is refused.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::needless_range_loop,
    clippy::cast_precision_loss,
    reason = "test scaffolding indexes small fixtures"
)]

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, DerivativeScale, ExecutionContext, Lag, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue, TemporalPolicy,
    TemporalResponseSpec, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::TemporalAdmg;

const N: usize = 600;
const X: u32 = 0;
const Y: u32 = 1;
const Z: u32 = 2;
const W: u32 = 3;

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

/// `w = sin(1.7 t + .3) + .8 l`, `z = cos(.41 t) + .5 w`, `x = 2 + .6 z + sin(.71 t) + .2 cos(.13
/// t)`, `y(t) = 5 + 2 x(t-1) + 3 z(t-1) + .7 l(t-1) + e`, with the latent `l = cos(.77 t + 2)`.
/// Returns the columns `x, y, z, w`.
fn columns() -> [Vec<f64>; 4] {
    let mut x = vec![0.0; N];
    let mut y = vec![0.0; N];
    let mut z = vec![0.0; N];
    let mut w = vec![0.0; N];
    let mut latent = vec![0.0; N];
    for t in 0..N {
        let clock = t as f64;
        latent[t] = (clock * 0.77 + 2.0).cos();
        w[t] = (clock * 1.7 + 0.3).sin() + 0.8 * latent[t];
        z[t] = (clock * 0.41).cos() + 0.5 * w[t];
        x[t] = 2.0 + 0.6 * z[t] + (clock * 0.71).sin() + 0.2 * (clock * 0.13).cos();
    }
    for t in 1..N {
        y[t] = 5.0
            + 2.0 * x[t - 1]
            + 3.0 * z[t - 1]
            + 0.7 * latent[t - 1]
            + 0.05 * (t as f64 * 3.1).cos();
    }
    [x, y, z, w]
}

fn series() -> TimeSeriesData {
    let [x, y, z, w] = columns();
    TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("y", y.as_slice()), ("z", z.as_slice()), ("w", w.as_slice())],
        1,
    )
    .unwrap()
}

/// Least-squares coefficients of `y` on an intercept and `regressors` (normal equations,
/// Gaussian elimination with partial pivoting).
fn ols(regressors: &[Vec<f64>], y: &[f64]) -> Vec<f64> {
    let k = regressors.len() + 1;
    let mut a = vec![vec![0.0; k + 1]; k];
    for i in 0..y.len() {
        let row: Vec<f64> =
            std::iter::once(1.0).chain(regressors.iter().map(|column| column[i])).collect();
        for p in 0..k {
            for q in 0..k {
                a[p][q] += row[p] * row[q];
            }
            a[p][k] += row[p] * y[i];
        }
    }
    for col in 0..k {
        let pivot = (col..k).max_by(|&p, &q| a[p][col].abs().total_cmp(&a[q][col].abs())).unwrap();
        a.swap(col, pivot);
        for row in 0..k {
            if row != col {
                let factor = a[row][col] / a[col][col];
                for c in col..=k {
                    let value = a[col][c];
                    a[row][c] -= factor * value;
                }
            }
        }
    }
    (0..k).map(|p| a[p][k] / a[p][p]).collect()
}

/// The structural slope and the confounded one, from the lag-aligned rows of the columns.
fn references() -> (f64, f64) {
    let [x, y, z, w] = columns();
    let lagged = |c: &Vec<f64>| c[..N - 1].to_vec();
    let outcome = y[1..].to_vec();
    let adjusted = ols(&[lagged(&x), lagged(&z), lagged(&w)], &outcome)[1];
    let naive = ols(&[lagged(&x)], &outcome)[1];
    (adjusted, naive)
}

/// `z(t-1) -> x(t-1)`, `z(t-1) -> y(t)`, `x(t-1) -> y(t)`, `w(t-1) -> z(t-1)` and a latent common
/// cause of `w(t-1)` and `y(t)`.
fn template() -> TemporalAdmg {
    let mut graph = TemporalAdmg::empty();
    let x1 = graph.add_lagged(v(X), Lag::from_raw(1)).unwrap();
    let z1 = graph.add_lagged(v(Z), Lag::from_raw(1)).unwrap();
    let w1 = graph.add_lagged(v(W), Lag::from_raw(1)).unwrap();
    let y0 = graph.add_lagged(v(Y), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(z1, x1).unwrap();
    graph.insert_directed(z1, y0).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(w1, z1).unwrap();
    graph.insert_bidirected(w1, y0).unwrap();
    graph
}

fn derivative() -> CausalQuery {
    CausalQuery::Response(
        ResponseQuery::new(ResponseFunctional::PointDerivative {
            outcome: v(Y),
            treatment: v(X),
            at: 2.0,
            order: 1,
            scale: DerivativeScale::Identity,
        })
        .with_temporal(
            TemporalResponseSpec::new(vec![1_u32], TemporalPolicy::pulse(-1), None).unwrap(),
        ),
    )
}

fn value(result: &antecedent::StudyResult) -> f64 {
    let ResponseIdentification::PointIdentified(ResponseValue::Scalar(x)) =
        &result.response().unwrap().estimate
    else {
        panic!("the derivative is not point identified");
    };
    *x
}

#[test]
fn the_window_adjustment_recovers_the_structural_slope_with_a_block_band() {
    let (structural, naive) = references();
    assert!((structural - 2.0).abs() < 0.2, "{structural}");
    assert!((naive - structural).abs() > 0.4, "the confounder must bias {naive} vs {structural}");
    let result = Study::series(series())
        .graph(template())
        .query(derivative())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(48)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(9))
        .unwrap();
    let got = value(&result);
    assert!((got - structural).abs() < 0.15, "{got} vs the regression reference {structural}");
    assert!((got - naive).abs() > 0.3, "{got} is the confounded slope {naive}");
    assert!(
        !matches!(result.response().unwrap().uncertainty, ResponseUncertainty::None),
        "the requested block replicates must publish a band"
    );
}

#[test]
fn the_bayesian_derivative_publishes_a_credible_band_around_the_same_slope() {
    let (structural, _) = references();
    let result = Study::series(series())
        .graph(template())
        .query(derivative())
        .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)))
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(10))
        .unwrap();
    let got = value(&result);
    assert!((got - structural).abs() < 0.2, "{got} vs {structural}");
    match &result.response().unwrap().uncertainty {
        ResponseUncertainty::Scalar { interpretation, lower, upper, .. } => {
            assert_eq!(*interpretation, antecedent_core::IntervalInterpretation::Credible);
            assert!(lower <= upper);
        }
        other => panic!("expected a credible interval, got {other:?}"),
    }
}

#[test]
fn a_treatment_sharing_a_latent_cause_with_the_outcome_is_refused_as_a_hedge() {
    // `x(t-1) -> y(t)` with a latent common cause of `x(t-1)` and `y(t)`: no adjustment set
    // exists on the window and the complete algorithm finds a hedge, so no estimator
    // identifies the slope (Shpitser and Pearl 2006).
    let mut graph = TemporalAdmg::empty();
    let x1 = graph.add_lagged(v(X), Lag::from_raw(1)).unwrap();
    let y0 = graph.add_lagged(v(Y), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_bidirected(x1, y0).unwrap();
    let outcome = Study::series(series())
        .graph(graph)
        .query(derivative())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .and_then(|study| study.run(&ExecutionContext::for_tests(11)));
    let error = outcome.expect_err("a hedge must not be estimated");
    assert!(error.is_identification_refusal(), "{error}");
}

fn derivative_at(horizons: Vec<u32>) -> CausalQuery {
    CausalQuery::Response(
        ResponseQuery::new(ResponseFunctional::PointDerivative {
            outcome: v(Y),
            treatment: v(X),
            at: 2.0,
            order: 1,
            scale: DerivativeScale::Identity,
        })
        .with_temporal(
            TemporalResponseSpec::new(horizons, TemporalPolicy::pulse(-1), None).unwrap(),
        ),
    )
}

#[test]
fn several_horizons_publish_one_slope_each_and_a_horizon_without_a_path_is_zero() {
    // `x(t-1)` reaches `y(t)` (horizon 1) and has no path to `y(t+2)` (horizon 3), whose
    // parents in the template are `x(t+1)` and `z(t+1)`. The independent reference for each
    // horizon is the regression of `y` at that horizon on the treatment and the confounders.
    // Horizon 3 sits past the first lag of the periodic regressors, whose serial correlation
    // would otherwise load a lag-1 horizon's regression with an association the template
    // does not draw.
    let [x, y, z, w] = columns();
    let slope = |horizon: usize| {
        let rows = N - horizon;
        let regress = |c: &Vec<f64>| c[..rows].to_vec();
        ols(&[regress(&x), regress(&z), regress(&w)], &y[horizon..])[1]
    };
    let (first, second) = (slope(1), slope(3));
    assert!((first - 2.0).abs() < 0.2 && second.abs() < 1.0, "{first} {second}");
    let result = Study::series(series())
        .graph(template())
        .query(derivative_at(vec![1, 3]))
        .refute(RefuteSuite::None)
        .bootstrap_replicates(48)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(14))
        .unwrap();
    let ResponseIdentification::PointIdentified(ResponseValue::Vector(values)) =
        &result.response().unwrap().estimate
    else {
        panic!("several horizons publish a vector of slopes");
    };
    assert_eq!(values.len(), 2);
    assert!((values[0] - first).abs() < 0.2, "{} vs {first}", values[0]);
    assert!((values[1] - second).abs() < 0.6, "{} vs {second}", values[1]);
    assert!(matches!(
        result.response().unwrap().uncertainty,
        ResponseUncertainty::PointwiseBand { .. }
    ));
}

#[test]
fn a_prepared_handle_runs_and_refreshes_the_temporal_derivative() {
    let data = series();
    let study = Study::series(data.clone())
        .graph(template())
        .query(derivative())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(15);
    let fresh = study.run(&ctx).unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let click = prepared.estimate_series(&data, &ctx).unwrap();
    assert!((value(&click) - value(&fresh)).abs() < 1e-9, "{} {}", value(&click), value(&fresh));
}

/// `x(t-1) -> m(t-1) -> y(t)` with a latent common cause of `x(t-1)` and `y(t)`, so the window
/// has no adjustment set and the slope is identified through the front door.
fn front_door_columns() -> [Vec<f64>; 3] {
    let mut x = vec![0.0; N];
    let mut m = vec![0.0; N];
    let mut y = vec![0.0; N];
    let mut latent = vec![0.0; N];
    for t in 0..N {
        let clock = t as f64;
        latent[t] = (clock * 0.77 + 2.0).cos();
        x[t] = 1.0 + (clock * 0.71).sin() + 0.2 * (clock * 0.13).cos() + 0.8 * latent[t];
        m[t] = 0.5 + 0.9 * x[t] + 0.5 * (clock * 1.9).sin();
    }
    for t in 1..N {
        y[t] = 3.0 + 2.0 * m[t - 1] + 0.7 * latent[t - 1] + 0.05 * (t as f64 * 3.1).cos();
    }
    [x, m, y]
}

#[test]
fn a_front_door_slope_on_the_window_is_estimated_not_refused() {
    let [x, m, y] = front_door_columns();
    let data = TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("y", y.as_slice()), ("m", m.as_slice())],
        1,
    )
    .unwrap();
    // Reference: slope of the mediator on the treatment times the mediator's coefficient in
    // the outcome regression given the treatment, on the lag-aligned rows.
    let lagged = |c: &Vec<f64>| c[..N - 1].to_vec();
    let outcome = y[1..].to_vec();
    let reference =
        ols(&[lagged(&x)], &lagged(&m))[1] * ols(&[lagged(&m), lagged(&x)], &outcome)[1];
    let naive = ols(&[lagged(&x)], &outcome)[1];
    assert!((reference - 1.8).abs() < 0.05, "{reference}");
    assert!((naive - reference).abs() > 0.2, "the latent cause must bias the naive slope");
    let mut graph = TemporalAdmg::empty();
    let x1 = graph.add_lagged(v(X), Lag::from_raw(1)).unwrap();
    let m1 = graph.add_lagged(v(2), Lag::from_raw(1)).unwrap();
    let y0 = graph.add_lagged(v(Y), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, m1).unwrap();
    graph.insert_directed(m1, y0).unwrap();
    graph.insert_bidirected(x1, y0).unwrap();
    let result = Study::series(data)
        .graph(graph)
        .query(derivative())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(32)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(16))
        .unwrap();
    let got = value(&result);
    assert!((got - reference).abs() < 0.2, "{got} vs the front-door reference {reference}");
    assert!((got - naive).abs() > 0.05, "{got} is the confounded slope {naive}");
    assert!(
        !matches!(result.response().unwrap().uncertainty, ResponseUncertainty::None),
        "the block bootstrap must publish an interval"
    );
}

/// Every derivative family on the temporal window publishes an artifact that an independent
/// consumer reads back, for both inferences.
#[test]
fn every_derivative_functional_on_a_temporal_admg_publishes_a_consumable_artifact() {
    use antecedent_core::DerivativeWeighting;
    use std::sync::Arc;

    let data = series();
    let point = |scale| ResponseFunctional::PointDerivative {
        outcome: v(Y),
        treatment: v(X),
        at: 2.0,
        order: 1,
        scale,
    };
    let functionals = [
        ("point", point(DerivativeScale::Identity)),
        ("elasticity", point(DerivativeScale::LogLog)),
        ("semi_treatment", point(DerivativeScale::LogTreatment)),
        (
            "average",
            ResponseFunctional::AverageDerivative {
                outcome: v(Y),
                treatment: v(X),
                weighting: DerivativeWeighting::Observed,
            },
        ),
        (
            "jacobian",
            ResponseFunctional::Jacobian {
                outcomes: Arc::from([v(Y)]),
                treatments: Arc::from([v(X)]),
                at: Arc::from([2.0]),
                scale: DerivativeScale::Identity,
            },
        ),
        (
            "directional",
            ResponseFunctional::DirectionalDerivative {
                outcomes: Arc::from([v(Y)]),
                treatments: Arc::from([v(X)]),
                at: Arc::from([2.0]),
                direction: Arc::from([1.0]),
            },
        ),
    ];
    let ctx = ExecutionContext::for_tests(21);
    for (mode, inference) in [
        ("frequentist", InferenceMode::Frequentist),
        ("bayesian", InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(16))),
    ] {
        for (key, functional) in &functionals {
            let label = format!("{key} {mode}");
            let query = CausalQuery::Response(
                ResponseQuery::new(functional.clone()).with_temporal(
                    TemporalResponseSpec::new(vec![1_u32], TemporalPolicy::pulse(-1), None)
                        .unwrap(),
                ),
            );
            let study = Study::series(data.clone())
                .graph(template())
                .query(query)
                .inference(inference.clone())
                .refute(RefuteSuite::None)
                .bootstrap_replicates(0)
                .build()
                .unwrap();
            let prepared = study.prepare(&ctx).unwrap_or_else(|e| panic!("{label}: {e}"));
            let result =
                prepared.estimate_series(&data, &ctx).unwrap_or_else(|e| panic!("{label}: {e}"));
            let artifact = prepared
                .encode_contracted_result(&result, &format!("temporal-admg-{key}"), &ctx)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            let consumed = antecedent_io::consume_analysis_result(&artifact)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            assert!(
                consumed.acceptance.accepts_as_verified_program()
                    || !consumed.acceptance.unresolved.is_empty(),
                "{label}: the consumer must verify or explain"
            );
        }
    }
}
