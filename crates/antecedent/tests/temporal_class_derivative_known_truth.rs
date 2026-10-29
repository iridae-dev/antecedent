//! Known-truth evidence for the temporal derivative refuter states on a supplied or accepted
//! TemporalCpdag and TemporalPag.
//!
//! The class of the lagged template has one completion, so the Pulse witness is that completion's.
//! The series follows the lag-one linear law of
//! `conformance/estimate/temporal_derivative_refuter_states/expected.json` with a deterministic
//! near-Gaussian noise stream the fixture's reference regenerates. Every derivative, elasticity,
//! semi-elasticity, average-derivative, Jacobian and directional-derivative coordinate is built
//! under both structure routes, both inference modes and both refuter suites, then executed from
//! the prepared study with the builder dropped. The published value and the unit-step difference of
//! the identified levels are compared with the fixture's closed-form truth, the TemporalDag
//! Pulse-native suite runs on the Pulse witness, and the planted violation the fixture declares (an
//! evaluation point beyond the observed series) must be detected.
// SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::support::CellStatus;
use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, DerivativeScale, DerivativeWeighting, ExecutionContext, Lag, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, TemporalPolicy, TemporalResponseSpec,
    VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{NodeRef, TemporalCpdag, TemporalDag, TemporalPag, ensure_lagged};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/temporal_derivative_refuter_states/expected.json"
    ))
    .expect("temporal_derivative_refuter_states fixture parses")
}

fn number(document: &Value, key: &str) -> f64 {
    document[key].as_f64().unwrap_or_else(|| panic!("fixture lacks number `{key}`"))
}

fn numbers(document: &Value, key: &str) -> Vec<f64> {
    document[key].as_array().map_or_else(
        || vec![number(document, key)],
        |items| items.iter().map(|item| item.as_f64().unwrap()).collect(),
    )
}

fn series() -> TimeSeriesData {
    let n = 400;
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
    let mut x = vec![0.0; n];
    let mut b = vec![0.0; n];
    let mut y = vec![0.0; n];
    let mut v = vec![0.0; n];
    for t in 0..n {
        b[t] = if gauss() > 0.0 { 1.0 } else { 0.0 };
        x[t] = 1.0 + 0.8 * ((t as f64) * 0.37).sin() + 0.3 * gauss();
        if t > 0 {
            y[t] = 3.0 + 2.0 * x[t - 1] + 0.5 * b[t - 1] + 0.3 * gauss();
            v[t] = 1.0 + x[t - 1] - b[t - 1] + 0.3 * gauss();
        }
    }
    let r: Vec<f64> = (0..n).map(|t| ((t as f64) * 0.17).cos()).collect();
    TimeSeriesData::from_f64_columns(
        [
            ("x", x.as_slice()),
            ("b", b.as_slice()),
            ("y", y.as_slice()),
            ("v", v.as_slice()),
            ("r", r.as_slice()),
        ],
        1,
    )
    .unwrap()
}

fn graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let b1 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(2), Lag::CONTEMPORANEOUS).unwrap();
    let v0 = ensure_lagged(&mut graph, VariableId::from_raw(3), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(b1, y0).unwrap();
    graph.insert_directed(x1, v0).unwrap();
    graph.insert_directed(b1, v0).unwrap();
    graph
}

fn functionals(at: f64) -> Vec<(&'static str, ResponseFunctional, bool)> {
    let point = |scale| ResponseFunctional::PointDerivative {
        outcome: VariableId::from_raw(2),
        treatment: VariableId::from_raw(0),
        at,
        order: 1,
        scale,
    };
    // The third field says whether the unit-step difference of the levels applies (an identity-scale
    // first derivative).
    vec![
        ("point", point(DerivativeScale::Identity), true),
        ("elasticity", point(DerivativeScale::LogLog), false),
        ("semi_treatment", point(DerivativeScale::LogTreatment), false),
        ("semi_outcome", point(DerivativeScale::LogOutcome), false),
        (
            "average",
            ResponseFunctional::AverageDerivative {
                outcome: VariableId::from_raw(2),
                treatment: VariableId::from_raw(0),
                weighting: DerivativeWeighting::Observed,
            },
            true,
        ),
        (
            "jacobian",
            ResponseFunctional::Jacobian {
                outcomes: Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
                treatments: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
                at: Arc::from([at, 1.0]),
                scale: DerivativeScale::Identity,
            },
            true,
        ),
        (
            "directional",
            ResponseFunctional::DirectionalDerivative {
                outcomes: Arc::from([VariableId::from_raw(2), VariableId::from_raw(3)]),
                treatments: Arc::from([VariableId::from_raw(0), VariableId::from_raw(1)]),
                at: Arc::from([at, 1.0]),
                direction: Arc::from([1.0, 2.0]),
            },
            false,
        ),
    ]
}

/// The lagged template as a PAG with no circles, plus a witness `r(t-1)` pointing into the
/// treatment and not adjacent to an outcome, so every treatment-to-outcome edge is visible.
fn directed_pag(dag: &TemporalDag) -> TemporalPag {
    let mut pag = TemporalPag::empty();
    let mut ids = Vec::new();
    for node in dag.nodes() {
        let NodeRef::Lagged { variable, lag } = *node else {
            panic!("temporal dag nodes are lagged");
        };
        ids.push(pag.add_lagged(variable, lag).unwrap());
    }
    for edge in dag.edges() {
        pag.insert_directed(ids[edge.a.as_usize()], ids[edge.b.as_usize()]).unwrap();
    }
    let r1 = pag.add_lagged(VariableId::from_raw(4), Lag::from_raw(1)).unwrap();
    let treatment = dag
        .nodes()
        .iter()
        .position(|node| {
            matches!(
                node,
                NodeRef::Lagged { variable, lag } if variable.raw() == 0 && *lag == Lag::from_raw(1)
            )
        })
        .unwrap();
    pag.insert_directed(r1, ids[treatment]).unwrap();
    pag
}

#[derive(Clone, Copy, Debug)]
enum Class {
    Cpdag,
    Pag,
}

fn study(
    data: &TimeSeriesData,
    class: Class,
    functional: &ResponseFunctional,
    accepted: bool,
    bayesian: bool,
    suite: RefuteSuite,
) -> antecedent::StudyBuilder {
    let query = ResponseQuery::new(functional.clone())
        .with_temporal(TemporalResponseSpec::new([1], TemporalPolicy::pulse(-1), Some(1)).unwrap());
    let builder = Study::series(data.clone()).query(CausalQuery::Response(query));
    let dag = graph();
    let builder = match (class, accepted) {
        (Class::Cpdag, false) => builder.graph(TemporalCpdag::from_temporal_dag(&dag)),
        (Class::Cpdag, true) => {
            builder.graph(AcceptedGraph::from(TemporalCpdag::from_temporal_dag(&dag)))
        }
        (Class::Pag, false) => builder.graph(directed_pag(&dag)),
        (Class::Pag, true) => builder.graph(AcceptedGraph::from(directed_pag(&dag))),
    };
    let builder = builder.refute(suite).bootstrap_replicates(0);
    if bayesian {
        builder.inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(24)))
    } else {
        builder
    }
}

fn published(result: &antecedent::StudyResult, label: &str) -> Vec<f64> {
    match &result.response().expect("response payload").estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => vec![*value],
        ResponseIdentification::PointIdentified(ResponseValue::Vector(values)) => values.to_vec(),
        ResponseIdentification::PointIdentified(ResponseValue::Jacobian { values, .. }) => {
            values.to_vec()
        }
        other => panic!("{label}: unexpected response {other:?}"),
    }
}

fn report<'a>(
    result: &'a antecedent::StudyResult,
    name: &str,
) -> Option<&'a antecedent_validate::RefutationReport> {
    result.refutations.iter().find(|report| report.refuter.as_ref() == name)
}

#[test]
fn temporal_class_derivative_refuter_states_known_truth() {
    let pin = fixture();
    let data = series();
    let value_band = number(&pin, "value_relative_tolerance");
    let difference_band = number(&pin, "finite_difference_relative_tolerance");
    let slope = number(&pin, "point");
    for class in [Class::Cpdag, Class::Pag] {
        for (key, functional, has_difference) in functionals(number(&pin, "at")) {
            let truths = numbers(&pin, key);
            for accepted in [false, true] {
                for bayesian in [false, true] {
                    for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                        let label = format!(
                            "{class:?} {key} accepted={accepted} bayesian={bayesian} {suite:?}"
                        );
                        let builder = study(&data, class, &functional, accepted, bayesian, suite)
                            .build()
                            .unwrap();
                        assert_eq!(builder.support_status(), Some(CellStatus::Licensed), "{label}");
                        let context = ExecutionContext::for_tests(812);
                        let prepared =
                            builder.prepare(&context).unwrap_or_else(|e| panic!("{label}: {e:?}"));
                        assert!(
                            !prepared.retained_operation().name().is_empty(),
                            "{label}: the prepared study retains no operation"
                        );
                        drop(builder);
                        let result = prepared
                            .estimate_retained(&context)
                            .unwrap_or_else(|e| panic!("{label}: {e:?}"));
                        assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
                        assert_eq!(
                            result.logical_plan.estimator.as_deref(),
                            Some(if bayesian {
                                "response.temporal.bayesian"
                            } else {
                                "temporal.response.gcomp"
                            }),
                            "{label}"
                        );
                        let values = published(&result, &label);
                        assert_eq!(values.len(), truths.len(), "{label}");
                        for (value, truth) in values.iter().zip(&truths) {
                            let band = (value_band * truth.abs()).max(0.1);
                            assert!(
                                (value - truth).abs() <= band,
                                "{label}: published {value} vs closed-form {truth}"
                            );
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
                                (difference.refuted_ate - slope).abs() <= difference_band * slope,
                                "{label}: unit-step difference {} vs closed-form {slope}",
                                difference.refuted_ate
                            );
                        }
                        for name in ["placebo.treatment", "data.subset"] {
                            let check = report(&result, name);
                            assert_eq!(
                                check.is_some(),
                                suite == RefuteSuite::Full,
                                "{label}: {name}"
                            );
                            if let Some(check) = check {
                                assert!(
                                    check.passed,
                                    "{label}: {name}: {:?}",
                                    check.failure_condition
                                );
                            }
                        }
                        if bayesian {
                            for name in ["prior_predictive", "posterior_predictive"] {
                                let check = report(&result, name)
                                    .unwrap_or_else(|| panic!("{label}: no {name} check"));
                                assert!(
                                    check.passed,
                                    "{label}: {name}: {:?}",
                                    check.failure_condition
                                );
                            }
                            assert_eq!(
                                report(&result, "prior_sensitivity").is_some(),
                                suite == RefuteSuite::Full,
                                "{label}"
                            );
                        } else if suite == RefuteSuite::Full {
                            let stability = report(&result, "bootstrap.ci_coverage")
                                .unwrap_or_else(|| panic!("{label}: no circular-block stability"));
                            assert!(stability.passed, "{label}");
                        }
                    }
                }
            }
        }
    }
}

#[test]
fn temporal_class_derivative_planted_off_support_point_is_detected() {
    let pin = fixture();
    let data = series();
    let clean = functionals(number(&pin, "at"));
    let planted = functionals(number(&pin, "off_support_at"));
    for class in [Class::Cpdag, Class::Pag] {
        for bayesian in [false, true] {
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                let label = format!("{class:?} bayesian={bayesian} {suite:?}");
                let run = |functional: &ResponseFunctional| {
                    study(&data, class, functional, false, bayesian, suite)
                        .build()
                        .unwrap()
                        .run(&ExecutionContext::for_tests(813))
                        .unwrap_or_else(|e| panic!("{label}: {e}"))
                };
                assert!(
                    report(&run(&clean[0].1), "derivative.support").is_some_and(|r| r.passed),
                    "{label}"
                );
                let result = run(&planted[0].1);
                let support = report(&result, "derivative.support").expect("support report");
                assert!(!support.passed, "{label}: off-support evaluation point not detected");
            }
        }
    }
}
