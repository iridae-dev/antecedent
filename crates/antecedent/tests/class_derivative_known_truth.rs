//! Known-truth evidence for derivatives on a supplied or accepted Cpdag, Pag and latent-confounded
//! Admg, at validation none, cheap and full.
//!
//! The data are the confounded linear law of
//! `conformance/response/class_derivative_states/expected.json`: the observed confounder `z` of
//! the treatment `a` and the outcomes makes the structural derivative the coefficient of `a` in the
//! regression on `(1, a, z)`, which the fixture's reference solves in exact rational arithmetic.
//! Every derivative coordinate is built under both structure routes, both inference modes and the
//! three validation suites; each run executes from the prepared study with the builder dropped.
//! The published value is compared with the fixture (not the confounded slope), the refuter states
//! (support, the finite difference of the identified levels, whole-procedure resampling stability,
//! Bayesian predictive checks and prior sensitivity) are checked against it, and the planted
//! violation the fixture declares (an evaluation point beyond the observed treatment range) must
//! be detected.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    reason = "test scaffolding builds small fixtures and one scenario is kept in one body"
)]

use std::sync::Arc;

use antecedent::support::CellStatus;
use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    CausalQuery, DerivativeScale, DerivativeWeighting, ExecutionContext, ResponseFunctional as F,
    ResponseIdentification, ResponseQuery, ResponseValue, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_graph::{Admg, Cpdag, DenseNodeId, Pag};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../conformance/response/class_derivative_states/expected.json"
    ))
    .expect("class_derivative_states fixture parses")
}

fn number(document: &Value, key: &str) -> f64 {
    document[key].as_f64().unwrap_or_else(|| panic!("fixture lacks number `{key}`"))
}

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

const N: usize = 800;

/// Columns `a, z, y, r, v, w` (indices 0..=5).
///
/// `r` is an exogenous cause of `a`; `w` is exogenous to `z`; a latent `l` moves both `w` and
/// `y`. `z = cos(.41 i) + .5 w`, `a = 2 + .6 z + .5 r + sin(.71 i) + .2 cos(.13 i)`,
/// `y = 5 + 2 a + 3 z + .7 l + e`, `v = 1 + .25 a - 1.5 z + e`.
struct Fixture {
    a: Vec<f64>,
    z: Vec<f64>,
    y: Vec<f64>,
    r: Vec<f64>,
    v: Vec<f64>,
    w: Vec<f64>,
}

impl Fixture {
    /// The fixture's noise: twelve uniforms of one 64-bit LCG stream minus six, drawn for `y`
    /// then `v` on each row.
    fn new() -> Self {
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
        let mut f = Self {
            a: Vec::new(),
            z: Vec::new(),
            y: Vec::new(),
            r: Vec::new(),
            v: Vec::new(),
            w: Vec::new(),
        };
        for i in 0..N {
            let t = i as f64;
            let latent = (t * 0.77 + 2.0).cos();
            let w = (t * 1.7 + 0.3).sin() + 0.8 * latent;
            let r = (t * 0.29 + 1.0).sin();
            let z = (t * 0.41).cos() + 0.5 * w;
            let a = 2.0 + 0.6 * z + 0.5 * r + (t * 0.71).sin() + 0.2 * (t * 0.13).cos();
            let y = 5.0 + 2.0 * a + 3.0 * z + 0.7 * latent + 0.5 * gauss();
            let v = 1.0 + 0.25 * a - 1.5 * z + 0.5 * gauss();
            f.a.push(a);
            f.z.push(z);
            f.y.push(y);
            f.r.push(r);
            f.v.push(v);
            f.w.push(w);
        }
        f
    }

    fn data(&self) -> TabularData {
        TabularData::from_f64_columns([
            ("a", self.a.as_slice()),
            ("z", self.z.as_slice()),
            ("y", self.y.as_slice()),
            ("r", self.r.as_slice()),
            ("v", self.v.as_slice()),
            ("w", self.w.as_slice()),
        ])
        .unwrap()
    }
}

/// Which structure carries the query.
#[derive(Clone, Copy, Debug)]
enum Structure {
    Cpdag,
    Pag,
    Admg,
}

/// `r -> a <- z`, `a -> y`, `z -> y`, `a -> v`, `z -> v` over columns `a, z, y, r, v` (a
/// v-structure, so every edge is compelled and the class has one completion).
fn cpdag() -> Cpdag {
    let mut g = Cpdag::with_variables(6);
    for (from, to) in [(3, 0), (1, 0), (0, 2), (1, 2), (0, 4), (1, 4)] {
        g.insert_directed(d(from), d(to)).unwrap();
    }
    g
}

/// The same edges as a PAG with no circles: `a -> y` is visible because `r` points into `a`
/// and is not adjacent to `y`.
fn pag() -> Pag {
    let mut g = Pag::with_variables(6);
    for (from, to) in [(3, 0), (1, 0), (0, 2), (1, 2), (0, 4), (1, 4)] {
        g.insert_directed(d(from), d(to)).unwrap();
    }
    g
}

/// The same edges plus `w -> z` and a latent common cause of `w` and `y` (`w <-> y`). The
/// observed confounder `z` still separates `a` from that latent cause.
fn admg() -> Admg {
    let mut g = Admg::with_variables(6);
    for (from, to) in [(3, 0), (1, 0), (0, 2), (1, 2), (0, 4), (1, 4), (5, 1)] {
        g.insert_directed(d(from), d(to)).unwrap();
    }
    g.insert_bidirected(d(5), d(2)).unwrap();
    g
}

fn study(
    data: &TabularData,
    structure: Structure,
    accepted: bool,
    functional: &F,
    inference: InferenceMode,
    suite: RefuteSuite,
) -> antecedent::StudyBuilder {
    let builder = Study::tabular(data.clone());
    let builder = match (structure, accepted) {
        (Structure::Cpdag, false) => builder.graph(cpdag()),
        (Structure::Cpdag, true) => builder.graph(AcceptedGraph::from(cpdag())),
        (Structure::Pag, false) => builder.graph(pag()),
        (Structure::Pag, true) => builder.graph(AcceptedGraph::from(pag())),
        (Structure::Admg, false) => builder.graph(admg()),
        (Structure::Admg, true) => builder.graph(AcceptedGraph::from(admg())),
    };
    builder
        .query(CausalQuery::Response(ResponseQuery::new(functional.clone())))
        .response_options(ContinuousResponseOptions {
            bandwidth: Some(0.35),
            ..ContinuousResponseOptions::default()
        })
        .inference(inference)
        .refute(suite)
        .bootstrap_replicates(12)
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

fn expected_estimator(key: &str) -> &'static str {
    match key {
        "average" => "response.riesz_ade",
        "jacobian" | "directional" => "response.gam_derivative",
        _ => "response.kennedy_dr",
    }
}

fn point(scale: DerivativeScale, at: f64) -> F {
    F::PointDerivative { outcome: v(2), treatment: v(0), at, order: 1, scale }
}

fn ids(xs: &[u32]) -> Arc<[VariableId]> {
    xs.iter().map(|&x| v(x)).collect()
}

/// Every functional with the reference values the fixture implies, and whether the unit-step
/// difference of the identified levels applies (an identity-scale or observed-weight first
/// derivative).
fn cases(pin: &Value, at: f64) -> Vec<(&'static str, F, Vec<f64>, bool)> {
    let slope = number(pin, "slope_y");
    vec![
        ("point", point(DerivativeScale::Identity, at), vec![slope], true),
        ("elasticity", point(DerivativeScale::LogLog, at), vec![number(pin, "elasticity")], true),
        (
            "semi_treatment",
            point(DerivativeScale::LogTreatment, at),
            vec![number(pin, "semi_treatment")],
            true,
        ),
        (
            "semi_outcome",
            point(DerivativeScale::LogOutcome, at),
            vec![number(pin, "semi_outcome")],
            true,
        ),
        (
            "average",
            F::AverageDerivative {
                outcome: v(2),
                treatment: v(0),
                weighting: DerivativeWeighting::Observed,
            },
            vec![slope],
            true,
        ),
        (
            "jacobian",
            F::Jacobian {
                outcomes: ids(&[2, 4]),
                treatments: ids(&[0]),
                at: Arc::from([at]),
                scale: DerivativeScale::Identity,
            },
            vec![slope, number(pin, "slope_v")],
            false,
        ),
        (
            "directional",
            F::DirectionalDerivative {
                outcomes: ids(&[2, 4]),
                treatments: ids(&[0]),
                at: Arc::from([at]),
                direction: Arc::from([1.0]),
            },
            vec![slope, number(pin, "slope_v")],
            false,
        ),
    ]
}

#[test]
fn class_derivative_routes_and_refuter_states_known_truth() {
    let pin = fixture();
    let fixture_rows = Fixture::new();
    let data = fixture_rows.data();
    let band = number(&pin, "value_relative_tolerance");
    let confounded = number(&pin, "naive_y");
    for structure in [Structure::Cpdag, Structure::Pag, Structure::Admg] {
        for (key, functional, truths, has_difference) in cases(&pin, number(&pin, "at")) {
            for accepted in [false, true] {
                for bayesian in [false, true] {
                    for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                        let label = format!(
                            "{structure:?} {key} accepted={accepted} bayesian={bayesian} {suite:?}"
                        );
                        let inference = if bayesian {
                            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(24))
                        } else {
                            InferenceMode::Frequentist
                        };
                        let builder =
                            study(&data, structure, accepted, &functional, inference, suite)
                                .build()
                                .unwrap_or_else(|e| panic!("{label}: {e}"));
                        assert_eq!(builder.support_status(), Some(CellStatus::Licensed), "{label}");
                        let context = ExecutionContext::for_tests(814);
                        let prepared =
                            builder.prepare(&context).unwrap_or_else(|e| panic!("{label}: {e:?}"));
                        drop(builder);
                        let result = prepared
                            .estimate(&data, &context)
                            .unwrap_or_else(|e| panic!("{label}: {e:?}"));
                        assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
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
                        assert_eq!(
                            result.logical_plan.estimator.as_deref(),
                            Some(expected_estimator(key)),
                            "{label}"
                        );
                        if suite == RefuteSuite::None {
                            assert!(result.refutations.is_empty(), "{label}");
                            continue;
                        }
                        assert!(
                            report(&result, "derivative.support").is_some_and(|r| r.passed),
                            "{label}"
                        );
                        let difference = report(&result, "derivative.finite_difference");
                        assert_eq!(difference.is_some(), has_difference, "{label}");
                        if let Some(difference) = difference {
                            assert!(
                                difference.passed,
                                "{label}: {:?}",
                                difference.failure_condition
                            );
                            assert!(
                                (difference.refuted_ate - truths[0]).abs()
                                    <= number(&pin, "finite_difference_relative_tolerance")
                                        * truths[0].abs(),
                                "{label}: finite difference {} vs fixture {}",
                                difference.refuted_ate,
                                truths[0]
                            );
                        }
                        let stability = report(&result, "derivative.bootstrap_stability");
                        assert_eq!(stability.is_some(), suite == RefuteSuite::Full, "{label}");
                        if let Some(stability) = stability {
                            assert!(stability.passed && stability.replicates >= 2, "{label}");
                        }
                        if bayesian {
                            let posterior =
                                report(&result, "posterior_predictive").unwrap_or_else(|| {
                                    panic!("{label}: no posterior predictive check")
                                });
                            assert!(posterior.passed, "{label}: {:?}", posterior.failure_condition);
                            // The prior predictive check may only fail by reporting that the weakly
                            // informative default prior is diffuse relative to the data, which its
                            // own message calls a statement about the prior.
                            let prior = report(&result, "prior_predictive")
                                .unwrap_or_else(|| panic!("{label}: no prior predictive check"));
                            assert!(
                                prior.passed
                                    || prior.failure_condition.as_deref().is_some_and(|reason| {
                                        reason.contains("a statement about the prior")
                                    }),
                                "{label}: {:?}",
                                prior.failure_condition
                            );
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
}

#[test]
fn class_derivative_planted_off_support_point_is_detected() {
    let pin = fixture();
    let data = Fixture::new().data();
    let clean = cases(&pin, number(&pin, "at")).remove(0).1;
    let planted = cases(&pin, number(&pin, "off_support_at")).remove(0).1;
    for structure in [Structure::Cpdag, Structure::Pag, Structure::Admg] {
        for bayesian in [false, true] {
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                let label = format!("{structure:?} bayesian={bayesian} {suite:?}");
                let run = |functional: &F| {
                    let inference = if bayesian {
                        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(24))
                    } else {
                        InferenceMode::Frequentist
                    };
                    study(&data, structure, false, functional, inference, suite)
                        .build()
                        .unwrap()
                        .run(&ExecutionContext::for_tests(815))
                };
                let clean_result = run(&clean).unwrap_or_else(|e| panic!("{label}: {e}"));
                assert!(
                    report(&clean_result, "derivative.support").is_some_and(|r| r.passed),
                    "{label}"
                );
                // A point beyond the observed treatment range is detected: either the support
                // check fails, or the local window around it holds too little weight and the route
                // refuses with that typed reason.
                match run(&planted) {
                    Ok(result) => {
                        let support =
                            report(&result, "derivative.support").expect("support report");
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
}
