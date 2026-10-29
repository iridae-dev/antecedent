//! Known-truth evidence for derivatives on a supplied or accepted temporal ADMG at one horizon,
//! at validation none, cheap and full.
//!
//! The series is the confounded lag-one law of
//! `conformance/response/temporal_admg_derivative_states/expected.json`: the observed confounder
//! `z(t-1)` separates the treatment `x(t-1)` from the latent cause of `w(t-1)` and `y(t)`, so the
//! structural slope is the coefficient of `x(t-1)` in the regression of `y(t)` on
//! `(1, x(t-1), z(t-1), w(t-1))`, which the fixture's reference solves in exact rational
//! arithmetic. Every derivative coordinate is built under both structure routes, both inference
//! modes and the three validation suites; each run executes from the prepared study with the
//! builder dropped. The published value is compared with the fixture (not the confounded slope),
//! the refuter states (support, the unit-step difference of the identified levels, the
//! Pulse-witness suite, Bayesian predictive checks and prior sensitivity) are checked against it,
//! and the planted violation the fixture declares (an evaluation point beyond the observed
//! treatment range) must be detected.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "test scaffolding builds small fixtures and one scenario is kept in one body"
)]

use antecedent::support::CellStatus;
use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    CausalQuery, DerivativeScale, DerivativeWeighting, ExecutionContext, Lag,
    ResponseFunctional as F, ResponseIdentification, ResponseQuery, ResponseValue, TemporalPolicy,
    TemporalResponseSpec, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::TemporalAdmg;
use serde_json::Value;
use std::sync::Arc;

const N: usize = 600;
const X: u32 = 0;
const Y: u32 = 1;
const Z: u32 = 2;
const W: u32 = 3;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../conformance/response/temporal_admg_derivative_states/expected.json"
    ))
    .expect("temporal_admg_derivative_states fixture parses")
}

fn number(document: &Value, key: &str) -> f64 {
    document[key].as_f64().unwrap_or_else(|| panic!("fixture lacks number `{key}`"))
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

/// `w = sin(1.7 t + .3) + .8 l`, `z = cos(.41 t) + .5 w`, `x = 2 + .6 z + sin(.71 t) + .2 cos(.13
/// t)`, `y(t) = 5 + 2 x(t-1) + 3 z(t-1) + .7 l(t-1) + .5 e` with the latent `l = cos(.77 t + 2)`
/// and `e` the sum of twelve uniforms of one 64-bit LCG stream minus six, one draw per `t >= 1`.
/// Returns the columns `x, y, z, w` as a series.
fn series() -> TimeSeriesData {
    let mut state = 20_240_607_u64;
    let mut gauss = || {
        let mut sum = 0.0;
        for _ in 0..12 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            sum += (state >> 11) as f64 / 9_007_199_254_740_992.0;
        }
        sum - 6.0
    };
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
        y[t] = 5.0 + 2.0 * x[t - 1] + 3.0 * z[t - 1] + 0.7 * latent[t - 1] + 0.5 * gauss();
    }
    TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("y", y.as_slice()), ("z", z.as_slice()), ("w", w.as_slice())],
        1,
    )
    .unwrap()
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

fn point(scale: DerivativeScale, at: f64) -> F {
    F::PointDerivative { outcome: v(Y), treatment: v(X), at, order: 1, scale }
}

fn study(
    data: &TimeSeriesData,
    functional: &F,
    accepted: bool,
    inference: InferenceMode,
    suite: RefuteSuite,
) -> antecedent::StudyBuilder {
    let query = ResponseQuery::new(functional.clone())
        .with_temporal(TemporalResponseSpec::new(vec![1_u32], TemporalPolicy::pulse(-1), None).unwrap());
    let builder = Study::series(data.clone()).query(CausalQuery::Response(query));
    let builder = if accepted {
        builder.graph(AcceptedGraph::from(template()))
    } else {
        builder.graph(template())
    };
    builder.inference(inference).refute(suite).bootstrap_replicates(12)
}

fn published(result: &StudyResult, label: &str) -> Vec<f64> {
    let ResponseIdentification::PointIdentified(value) = &result.response().unwrap().estimate
    else {
        panic!("{label}: the response is not point identified");
    };
    match value {
        ResponseValue::Scalar(x) => vec![*x],
        ResponseValue::Jacobian { values, .. } | ResponseValue::Vector(values) => values.to_vec(),
        other => panic!("{label}: unexpected payload {other:?}"),
    }
}

fn report<'a>(
    result: &'a StudyResult,
    name: &str,
) -> Option<&'a antecedent_validate::RefutationReport> {
    result.refutations.iter().find(|report| report.refuter.as_ref() == name)
}

/// Every functional with the reference values the fixture implies.
fn cases(pin: &Value, at: f64) -> Vec<(&'static str, F, Vec<f64>)> {
    let slope = number(pin, "slope_y");
    vec![
        ("point", point(DerivativeScale::Identity, at), vec![slope]),
        ("elasticity", point(DerivativeScale::LogLog, at), vec![number(pin, "elasticity")]),
        (
            "semi_treatment",
            point(DerivativeScale::LogTreatment, at),
            vec![number(pin, "semi_treatment")],
        ),
        (
            "semi_outcome",
            point(DerivativeScale::LogOutcome, at),
            vec![number(pin, "semi_outcome")],
        ),
        (
            "average",
            F::AverageDerivative {
                outcome: v(Y),
                treatment: v(X),
                weighting: DerivativeWeighting::Observed,
            },
            vec![slope],
        ),
        (
            "jacobian",
            F::Jacobian {
                outcomes: Arc::from([v(Y)]),
                treatments: Arc::from([v(X)]),
                at: Arc::from([at]),
                scale: DerivativeScale::Identity,
            },
            vec![slope],
        ),
        (
            "directional",
            F::DirectionalDerivative {
                outcomes: Arc::from([v(Y)]),
                treatments: Arc::from([v(X)]),
                at: Arc::from([at]),
                direction: Arc::from([1.0]),
            },
            vec![slope],
        ),
    ]
}

#[test]
fn temporal_admg_derivative_routes_and_refuter_states_known_truth() {
    let pin = fixture();
    let data = series();
    let band = number(&pin, "value_relative_tolerance");
    let confounded = number(&pin, "naive_y");
    for (key, functional, truths) in cases(&pin, number(&pin, "at")) {
        for accepted in [false, true] {
            for bayesian in [false, true] {
                for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                    let label =
                        format!("{key} accepted={accepted} bayesian={bayesian} {suite:?}");
                    let inference = if bayesian {
                        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(24))
                    } else {
                        InferenceMode::Frequentist
                    };
                    let builder = study(&data, &functional, accepted, inference, suite)
                        .build()
                        .unwrap_or_else(|e| panic!("{label}: {e}"));
                    assert_eq!(builder.support_status(), Some(CellStatus::Licensed), "{label}");
                    let context = ExecutionContext::for_tests(816);
                    let prepared =
                        builder.prepare(&context).unwrap_or_else(|e| panic!("{label}: {e:?}"));
                    assert!(
                        !prepared.retained_operation().name().is_empty(),
                        "{label}: the prepared study retains no operation"
                    );
                    drop(builder);
                    let result = prepared
                        .estimate_series(&data, &context)
                        .unwrap_or_else(|e| panic!("{label}: {e:?}"));
                    assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
                    assert_eq!(
                        result.logical_plan.estimator.as_deref(),
                        Some(if bayesian { "response.temporal.bayesian" } else { "temporal.response.gcomp" }),
                        "{label}"
                    );
                    let values = published(&result, &label);
                    assert_eq!(values.len(), truths.len(), "{label}");
                    for (value, truth) in values.iter().zip(&truths) {
                        assert!(
                            (value - truth).abs() <= (band * truth.abs()).max(0.1),
                            "{label}: published {value} vs fixture {truth}"
                        );
                    }
                    // The confounded slope is not what a route publishes.
                    if key == "point" || key == "average" {
                        assert!((values[0] - confounded).abs() > 0.3, "{label}");
                    }
                    if suite == RefuteSuite::None {
                        assert!(result.refutations.is_empty(), "{label}");
                        continue;
                    }
                    assert!(
                        report(&result, "derivative.support").is_some_and(|r| r.passed),
                        "{label}"
                    );
                    // The unit-step difference of the identified levels applies to an identity-scale
                    // first derivative.
                    let difference = report(&result, "derivative.finite_difference");
                    assert_eq!(
                        difference.is_some(),
                        matches!(key, "point" | "average" | "jacobian"),
                        "{label}"
                    );
                    if let Some(difference) = difference {
                        assert!(difference.passed, "{label}: {:?}", difference.failure_condition);
                        assert!(
                            (difference.refuted_ate - truths[0]).abs()
                                <= number(&pin, "finite_difference_relative_tolerance")
                                    * truths[0].abs(),
                            "{label}: unit-step difference {} vs fixture {}",
                            difference.refuted_ate,
                            truths[0]
                        );
                    }
                    for name in ["sensitivity.evalue"]
                        .into_iter()
                        .chain((suite == RefuteSuite::Full).then_some("sensitivity.nonparametric"))
                    {
                        let check = report(&result, name)
                            .unwrap_or_else(|| panic!("{label}: no {name} report"));
                        assert!(check.passed, "{label}: {name}: {:?}", check.failure_condition);
                    }
                    if bayesian {
                        for name in ["prior_predictive", "posterior_predictive"] {
                            let check = report(&result, name)
                                .unwrap_or_else(|| panic!("{label}: no {name} check"));
                            assert!(check.passed, "{label}: {name}: {:?}", check.failure_condition);
                        }
                        assert_eq!(
                            report(&result, "prior_sensitivity").is_some(),
                            suite == RefuteSuite::Full,
                            "{label}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn temporal_admg_derivative_planted_off_support_point_is_detected() {
    let pin = fixture();
    let data = series();
    let clean = cases(&pin, number(&pin, "at")).remove(0).1;
    let planted = cases(&pin, number(&pin, "off_support_at")).remove(0).1;
    for bayesian in [false, true] {
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let label = format!("bayesian={bayesian} {suite:?}");
            let run = |functional: &F| {
                let inference = if bayesian {
                    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(24))
                } else {
                    InferenceMode::Frequentist
                };
                study(&data, functional, false, inference, suite)
                    .build()
                    .unwrap()
                    .run(&ExecutionContext::for_tests(817))
            };
            let clean_result = run(&clean).unwrap_or_else(|e| panic!("{label}: {e}"));
            assert!(report(&clean_result, "derivative.support").is_some_and(|r| r.passed), "{label}");
            // A point beyond the observed treatment range is detected: either the support check
            // fails, or the local window around it holds too little weight and the route refuses.
            match run(&planted) {
                Ok(result) => {
                    let support = report(&result, "derivative.support").expect("support report");
                    assert!(!support.passed, "{label}: off-support point not detected");
                }
                Err(error) => assert!(
                    error.to_string().contains("too little effective weight"),
                    "{label}: {error}"
                ),
            }
        }
    }
}
