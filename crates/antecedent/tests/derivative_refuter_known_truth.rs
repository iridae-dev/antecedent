//! Known-truth evidence for the derivative refuter states on a static DAG.
//!
//! The data follow the linear structural law of
//! `conformance/response/derivative_refuter_states/expected.json` with a deterministic
//! near-Gaussian noise stream the fixture's reference regenerates. Every derivative, elasticity,
//! semi-elasticity and average-derivative coordinate is built under both structure routes, both
//! inference modes and both refuter suites; each run executes from its retained operation with the
//! builder dropped. The published value and the finite difference of the identified intervention
//! levels are compared with the fixture's closed-form truth, and the planted violations the
//! fixture declares (an evaluation point off the observed treatment support, a residual law that
//! is not Gaussian) must be detected.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::RetainedOperation;
use antecedent::{AcceptedGraph, BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study};
use antecedent_core::{
    CausalQuery, DerivativeScale, DerivativeWeighting, ExecutionContext, ResponseFunctional as F,
    ResponseIdentification, ResponseQuery, ResponseValue, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_graph::{Dag, DenseNodeId};
use serde_json::Value;

const ROWS: u32 = 800;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../conformance/response/derivative_refuter_states/expected.json"
    ))
    .expect("derivative_refuter_states fixture parses")
}

fn number(document: &Value, key: &str) -> f64 {
    document[key].as_f64().unwrap_or_else(|| panic!("fixture lacks number `{key}`"))
}

/// The fixture's linear-Gaussian law: the Irwin-Hall stream is drawn for `B` over every row, then
/// `Y`, then `V` (the order the fixture's reference regenerates).
fn table(noise_free_outcome: bool) -> TabularData {
    let mut state = 20_240_607_u64;
    let mut gauss = || {
        let mut sum = 0.0;
        for _ in 0..12 {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            #[allow(clippy::cast_precision_loss, reason = "a 53-bit uniform is exact in f64")]
            let uniform = (state >> 11) as f64 / 9_007_199_254_740_992.0;
            sum += uniform;
        }
        sum - 6.0
    };
    let a: Vec<f64> = (0..ROWS)
        .map(|i| 2.0 + (f64::from(i) * 0.71).sin() + 0.2 * (f64::from(i) * 0.13).cos())
        .collect();
    let b: Vec<f64> =
        (0..ROWS).map(|i| 0.7 * (f64::from(i) * 1.13).cos() + 0.4 * gauss()).collect();
    let scale = if noise_free_outcome { 0.0 } else { 0.3 };
    let y: Vec<f64> =
        a.iter().zip(&b).map(|(a, b)| 5.0 + 2.0 * a - 0.5 * b + scale * gauss()).collect();
    let v: Vec<f64> =
        a.iter().zip(&b).map(|(a, b)| 1.0 + 0.25 * a + 1.5 * b + 0.9 * gauss()).collect();
    TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("y", y.as_slice()),
        ("v", v.as_slice()),
    ])
    .unwrap()
}

fn dag() -> Dag {
    let mut graph = Dag::with_variables(4);
    for (source, target) in [(0, 2), (0, 3), (1, 2), (1, 3)] {
        graph
            .insert_directed(DenseNodeId::from_raw(source), DenseNodeId::from_raw(target))
            .unwrap();
    }
    graph
}

fn point(scale: DerivativeScale, at: f64) -> F {
    F::PointDerivative {
        outcome: VariableId::from_raw(2),
        treatment: VariableId::from_raw(0),
        at,
        order: 1,
        scale,
    }
}

fn average() -> F {
    F::AverageDerivative {
        outcome: VariableId::from_raw(2),
        treatment: VariableId::from_raw(0),
        weighting: DerivativeWeighting::Observed,
    }
}

fn study(
    data: &TabularData,
    functional: &F,
    accepted: bool,
    bayesian: bool,
    suite: RefuteSuite,
) -> antecedent::StudyBuilder {
    let base = Study::tabular(data.clone());
    let builder = if accepted { base.graph(AcceptedGraph::from(dag())) } else { base.graph(dag()) };
    let builder = builder
        .query(CausalQuery::Response(ResponseQuery::new(functional.clone())))
        .response_options(ContinuousResponseOptions { bandwidth: Some(0.35), ..Default::default() })
        .refute(suite)
        .bootstrap_replicates(0);
    if bayesian {
        builder.inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(48)))
    } else {
        builder
    }
}

fn values(result: &antecedent::StudyResult, label: &str) -> Vec<f64> {
    match &result.response().expect("response payload").estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(value)) => vec![*value],
        ResponseIdentification::PointIdentified(ResponseValue::Vector(values)) => values.to_vec(),
        ResponseIdentification::PointIdentified(ResponseValue::Jacobian { values, .. }) => {
            values.to_vec()
        }
        other => panic!("{label}: unexpected response {other:?}"),
    }
}

fn numbers(document: &Value, key: &str) -> Vec<f64> {
    document[key].as_array().map_or_else(
        || vec![number(document, key)],
        |items| items.iter().map(|item| item.as_f64().unwrap()).collect(),
    )
}

fn ids(raw: &[u32]) -> std::sync::Arc<[VariableId]> {
    raw.iter().copied().map(VariableId::from_raw).collect()
}

fn report<'a>(
    result: &'a antecedent::StudyResult,
    name: &str,
) -> Option<&'a antecedent_validate::RefutationReport> {
    result.refutations.iter().find(|report| report.refuter.as_ref() == name)
}

#[test]
fn derivative_refuter_states_known_truth() {
    let pin = fixture();
    let data = table(false);
    let value_band = number(&pin, "value_relative_tolerance");
    let difference_band = number(&pin, "finite_difference_relative_tolerance");
    let cases = [
        ("point", point(DerivativeScale::Identity, number(&pin, "at")), "response.kennedy_dr"),
        ("elasticity", point(DerivativeScale::LogLog, number(&pin, "at")), "response.kennedy_dr"),
        (
            "semi_treatment",
            point(DerivativeScale::LogTreatment, number(&pin, "at")),
            "response.kennedy_dr",
        ),
        (
            "semi_outcome",
            point(DerivativeScale::LogOutcome, number(&pin, "at")),
            "response.kennedy_dr",
        ),
        ("average", average(), "response.riesz_ade"),
        (
            "jacobian",
            F::Jacobian {
                outcomes: ids(&[2, 3]),
                treatments: ids(&[0, 1]),
                at: std::sync::Arc::from([2.0, 0.0]),
                scale: DerivativeScale::Identity,
            },
            "response.gam_derivative",
        ),
        (
            "directional",
            F::DirectionalDerivative {
                outcomes: ids(&[2, 3]),
                treatments: ids(&[0, 1]),
                at: std::sync::Arc::from([2.0, 0.0]),
                direction: std::sync::Arc::from([1.0, 2.0]),
            },
            "response.gam_derivative",
        ),
    ];
    for (key, functional, estimator) in &cases {
        let truths = numbers(&pin, key);
        let truth = truths[0];
        let multivariate = truths.len() > 1;
        for accepted in [false, true] {
            for bayesian in [false, true] {
                for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                    let label = format!("{key} accepted={accepted} bayesian={bayesian} {suite:?}");
                    let builder =
                        study(&data, functional, accepted, bayesian, suite).build().unwrap();
                    let context = ExecutionContext::for_tests(809);
                    let prepared =
                        builder.prepare(&context).unwrap_or_else(|e| panic!("{label}: {e:?}"));
                    // A refuted study is not sealed to the derivative operation (that one carries no
                    // refuter state); it runs its estimator route from the lowered program.
                    assert!(
                        matches!(prepared.retained_operation(), RetainedOperation::EstimatorRoute),
                        "{label}: retained operation {:?}",
                        prepared.retained_operation().name()
                    );
                    drop(builder);
                    let result = prepared
                        .estimate(&data, &context)
                        .unwrap_or_else(|e| panic!("{label}: {e:?}"));

                    assert_eq!(
                        result.logical_plan.estimator.as_deref(),
                        Some(*estimator),
                        "{label}"
                    );
                    assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
                    let published = values(&result, &label);
                    assert_eq!(published.len(), truths.len(), "{label}");
                    for (value, truth) in published.iter().zip(&truths) {
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
                    if multivariate {
                        // The contrast-shaped finite difference has no plug-in for a vector.
                        assert!(
                            report(&result, "derivative.finite_difference").is_none(),
                            "{label}"
                        );
                    } else {
                        let difference = report(&result, "derivative.finite_difference")
                            .unwrap_or_else(|| panic!("{label}: no finite-difference report"));
                        assert!(difference.passed, "{label}: {:?}", difference.failure_condition);
                        assert!(
                            (difference.refuted_ate - truth).abs() <= difference_band * truth.abs(),
                            "{label}: finite difference {} vs closed-form {truth}",
                            difference.refuted_ate
                        );
                    }
                    let stability = report(&result, "derivative.bootstrap_stability");
                    assert_eq!(stability.is_some(), suite == RefuteSuite::Full, "{label}");
                    if let Some(stability) = stability {
                        assert!(stability.passed && stability.replicates >= 2, "{label}");
                        assert!(
                            (stability.refuted_ate - truth).abs()
                                <= (value_band * truth.abs()).max(0.1),
                            "{label}"
                        );
                    }
                    if bayesian {
                        let posterior = report(&result, "posterior_predictive")
                            .unwrap_or_else(|| panic!("{label}: no posterior predictive check"));
                        assert!(posterior.passed, "{label}: {:?}", posterior.failure_condition);
                        // The prior predictive check may only fail by reporting that the weakly
                        // informative default prior is diffuse relative to the data, which its own
                        // message calls a statement about the prior.
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

#[test]
fn derivative_refuter_planted_violations_are_detected() {
    let pin = fixture();
    let data = table(false);
    let clean = number(&pin, "at");
    let off_support = number(&pin, "off_support_at");
    for bayesian in [false, true] {
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let label = format!("bayesian={bayesian} {suite:?}");
            let run = |at: f64| {
                study(&data, &point(DerivativeScale::Identity, at), false, bayesian, suite)
                    .build()
                    .unwrap()
                    .run(&ExecutionContext::for_tests(810))
                    .unwrap_or_else(|e| panic!("{label} at {at}: {e}"))
            };
            // The clean case passes the support check; a point beyond the observed treatment range is
            // detected although the plug-in still returns a number there.
            assert!(report(&run(clean), "derivative.support").is_some_and(|r| r.passed), "{label}");
            let planted = run(off_support);
            let support = report(&planted, "derivative.support").expect("support report");
            assert!(!support.passed, "{label}: off-support point not detected");
        }
    }
    // A residual law that is not Gaussian (the outcome has no noise at all, so its residuals are
    // the deterministic misfit of the regression) fails the Bayesian posterior predictive check
    // that the Gaussian-noise data passed above.
    let planted = study(&table(true), &average(), false, true, RefuteSuite::Cheap)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(811))
        .unwrap();
    let check = report(&planted, "posterior_predictive").expect("posterior predictive check");
    assert!(!check.passed, "non-Gaussian residuals not detected");
}
