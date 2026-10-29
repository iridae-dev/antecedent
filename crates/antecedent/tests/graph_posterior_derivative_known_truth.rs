//! Known-truth evidence for derivatives over a graph posterior of Dag atoms, at validation none,
//! cheap and full.
//!
//! The data follow the linear law of
//! `conformance/response/graph_posterior_derivative_states/expected.json`. The posterior holds two
//! DAG atoms that both identify the slope of `y` in `a` (`b` is not a confounder in the first and
//! is in the second) and, in the second posterior, a cyclic sample, which is not a member of the
//! DAG class and is unidentified posterior mass. Every identified atom has the structural
//! derivative the fixture's reference derives, so the posterior-weighted mean over the identified
//! atoms equals it whatever the weights, and the identified and unidentified masses are the sums
//! of the atoms' weights. Every derivative coordinate is built under both inference modes and the
//! three validation suites and executed from the prepared study with the builder dropped; the
//! per-atom refuter reports are mixed by posterior weight, and a planted evaluation point beyond
//! the observed treatment range must be detected.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::support::CellStatus;
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    CausalQuery, DerivativeScale, DerivativeWeighting, ExecutionContext, IdentificationStatus,
    ResponseFunctional as F, ResponseIdentification, ResponseQuery, ResponseValue, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::{GraphPosterior, SamplingSummary, set_edge};
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_prob::InferenceDiagnostics;
use serde_json::Value;

const ROWS: u32 = 800;
const PLAIN_EDGES: [(usize, usize); 4] = [(0, 2), (0, 3), (1, 2), (1, 3)];
const CONFOUNDED_EDGES: [(usize, usize); 5] = [(0, 2), (0, 3), (1, 2), (1, 3), (1, 0)];
/// `a -> y` and `y -> a`: not a DAG, so the sample is not a member of the posterior's class.
const CYCLIC_EDGES: [(usize, usize); 2] = [(0, 2), (2, 0)];

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../conformance/response/graph_posterior_derivative_states/expected.json"
    ))
    .expect("graph_posterior_derivative_states fixture parses")
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

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

/// The fixture's linear-Gaussian law: the Irwin-Hall stream is drawn for `B` over every row, then
/// `Y`, then `V`.
fn table() -> TabularData {
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
    let b: Vec<f64> = (0..ROWS).map(|_| 0.7 * gauss()).collect();
    let y: Vec<f64> =
        a.iter().zip(&b).map(|(a, b)| 5.0 + 2.0 * a - 0.2 * b + 0.3 * gauss()).collect();
    let w: Vec<f64> =
        a.iter().zip(&b).map(|(a, b)| 1.0 + 0.25 * a + 0.3 * b + 0.3 * gauss()).collect();
    TabularData::from_f64_columns([
        ("a", a.as_slice()),
        ("b", b.as_slice()),
        ("y", y.as_slice()),
        ("v", w.as_slice()),
    ])
    .unwrap()
}

fn posterior(atoms: &[&[(usize, usize)]], weights: &[f64]) -> GraphPosterior {
    let masks: Vec<u64> = atoms
        .iter()
        .map(|edges| {
            edges.iter().fold(0_u64, |mask, &(from, to)| set_edge(mask, 4, from, to, true))
        })
        .collect();
    GraphPosterior::new(
        4,
        weights.to_vec(),
        masks,
        vec![0.0; 16],
        vec![0.0; 16],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("graph_posterior_derivative"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
}

fn point(scale: DerivativeScale, at: f64) -> F {
    F::PointDerivative { outcome: v(2), treatment: v(0), at, order: 1, scale }
}

fn cases(at: f64) -> Vec<(&'static str, F, bool)> {
    let outcomes: Arc<[VariableId]> = Arc::from([v(2), v(3)]);
    vec![
        ("point", point(DerivativeScale::Identity, at), true),
        ("elasticity", point(DerivativeScale::LogLog, at), true),
        ("semi_treatment", point(DerivativeScale::LogTreatment, at), true),
        ("semi_outcome", point(DerivativeScale::LogOutcome, at), true),
        (
            "average",
            F::AverageDerivative {
                outcome: v(2),
                treatment: v(0),
                weighting: DerivativeWeighting::Observed,
            },
            true,
        ),
        (
            "jacobian",
            F::Jacobian {
                outcomes: outcomes.clone(),
                treatments: Arc::from([v(0)]),
                at: Arc::from([at]),
                scale: DerivativeScale::Identity,
            },
            false,
        ),
        (
            "directional",
            F::DirectionalDerivative {
                outcomes,
                treatments: Arc::from([v(0)]),
                at: Arc::from([at]),
                direction: Arc::from([1.0]),
            },
            false,
        ),
    ]
}

fn expected_estimator(key: &str) -> &'static str {
    match key {
        "average" => "response.riesz_ade",
        "jacobian" | "directional" => "response.gam_derivative",
        _ => "response.kennedy_dr",
    }
}

fn study(
    gp: GraphPosterior,
    functional: &F,
    bayesian: bool,
    suite: RefuteSuite,
) -> antecedent::StudyBuilder {
    let builder = Study::tabular(table())
        .graph_posterior(gp)
        .query(CausalQuery::Response(ResponseQuery::new(functional.clone())))
        .response_options(ContinuousResponseOptions {
            bandwidth: Some(0.35),
            ..ContinuousResponseOptions::default()
        })
        .refute(suite)
        .bootstrap_replicates(0);
    if bayesian {
        builder.inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(24)))
    } else {
        builder
    }
}

fn published(result: &StudyResult, label: &str) -> Vec<f64> {
    match &result.response().expect("response payload").estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(x)) => vec![*x],
        ResponseIdentification::PointIdentified(
            ResponseValue::Vector(values) | ResponseValue::Jacobian { values, .. },
        ) => values.to_vec(),
        // Lost posterior mass makes the response graph dependent: the mixture over the atoms
        // that identify it is carried by the structural response.
        ResponseIdentification::GraphDependent(_) => {
            match result
                .structural_response
                .as_ref()
                .and_then(|mixture| mixture.conditional_on_identified.clone())
            {
                Some(ResponseValue::Scalar(x)) => vec![x],
                Some(ResponseValue::Vector(values) | ResponseValue::Jacobian { values, .. }) => {
                    values.to_vec()
                }
                other => panic!("{label}: no conditional mixture, got {other:?}"),
            }
        }
        other => panic!("{label}: unexpected response {other:?}"),
    }
}

fn report<'a>(
    result: &'a StudyResult,
    name: &str,
) -> Option<&'a antecedent_validate::RefutationReport> {
    result.refutations.iter().find(|report| report.refuter.as_ref() == name)
}

#[test]
fn graph_posterior_derivative_mixtures_and_refuter_states_known_truth() {
    let pin = fixture();
    let band = number(&pin, "value_relative_tolerance");
    let weights = numbers(&pin["atoms"], "fully_identified_weights");
    for (key, functional, scalar_level) in cases(number(&pin, "at")) {
        let truths = numbers(&pin, key);
        for bayesian in [false, true] {
            for suite in [RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full] {
                let label = format!("{key} bayesian={bayesian} {suite:?}");
                let gp = posterior(&[&PLAIN_EDGES, &CONFOUNDED_EDGES], &weights);
                let builder = study(gp, &functional, bayesian, suite)
                    .build()
                    .unwrap_or_else(|e| panic!("{label}: {e}"));
                assert_eq!(builder.support_status(), Some(CellStatus::Licensed), "{label}");
                let context = ExecutionContext::for_tests(818);
                let prepared =
                    builder.prepare(&context).unwrap_or_else(|e| panic!("{label}: {e:?}"));
                assert!(
                    !prepared.retained_operation().name().is_empty(),
                    "{label}: the prepared study retains no operation"
                );
                drop(builder);
                let result = prepared
                    .estimate(&table(), &context)
                    .unwrap_or_else(|e| panic!("{label}: {e:?}"));
                assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
                assert_eq!(
                    result.logical_plan.estimator.as_deref(),
                    Some(expected_estimator(key)),
                    "{label}"
                );
                let mixture = result.structural_response.as_ref().expect("a structural mixture");
                assert!((mixture.identified_mass - 1.0).abs() < 1e-9, "{label}");
                assert!(mixture.unidentified_mass.abs() < 1e-9, "{label}");
                assert_eq!(mixture.atoms.len(), weights.len(), "{label}");
                let values = published(&result, &label);
                assert_eq!(values.len(), truths.len(), "{label}");
                for (value, truth) in values.iter().zip(&truths) {
                    assert!(
                        (value - truth).abs() <= (band * truth.abs()).max(0.15),
                        "{label}: mixture {value} vs fixture {truth}"
                    );
                }
                if suite == RefuteSuite::None {
                    assert!(result.refutations.is_empty(), "{label}");
                    continue;
                }
                assert!(
                    result
                        .diagnostics
                        .iter()
                        .any(|d| d.code.as_ref() == "refute.graph_posterior.per_atom"),
                    "{label}: per-atom refuter mixture"
                );
                assert!(report(&result, "derivative.support").is_some_and(|r| r.passed), "{label}");
                assert_eq!(
                    report(&result, "derivative.finite_difference").is_some(),
                    scalar_level,
                    "{label}"
                );
                if let Some(difference) = report(&result, "derivative.finite_difference") {
                    assert!(difference.passed, "{label}: {:?}", difference.failure_condition);
                    assert!(
                        (difference.refuted_ate - truths[0]).abs()
                            <= number(&pin, "finite_difference_relative_tolerance")
                                * truths[0].abs(),
                        "{label}: finite difference {} vs fixture {}",
                        difference.refuted_ate,
                        truths[0]
                    );
                }
                assert_eq!(
                    report(&result, "derivative.bootstrap_stability").is_some(),
                    suite == RefuteSuite::Full,
                    "{label}"
                );
                if bayesian {
                    for name in ["prior_predictive", "posterior_predictive"] {
                        let check = report(&result, name)
                            .unwrap_or_else(|| panic!("{label}: no {name} check"));
                        assert!(check.passed, "{label}: {name}: {:?}", check.failure_condition);
                    }
                }
            }
        }
    }
}

#[test]
fn graph_posterior_derivative_lost_mass_is_stated_and_the_identified_mixture_is_the_truth() {
    let pin = fixture();
    let weights = numbers(&pin["atoms"], "weights");
    let (identified, unidentified) =
        (number(&pin["atoms"], "identified_mass"), number(&pin["atoms"], "unidentified_mass"));
    let truth = number(&pin, "point");
    for bayesian in [false, true] {
        let label = format!("bayesian={bayesian}");
        let gp = posterior(&[&PLAIN_EDGES, &CONFOUNDED_EDGES, &CYCLIC_EDGES], &weights);
        let functional = point(DerivativeScale::Identity, number(&pin, "at"));
        let builder = study(gp, &functional, bayesian, RefuteSuite::None).build().unwrap();
        assert_eq!(builder.support_status(), Some(CellStatus::Licensed), "{label}");
        let result = builder.run(&ExecutionContext::for_tests(819)).unwrap();
        let mixture = result.structural_response.as_ref().expect("a structural mixture");
        assert!((mixture.identified_mass - identified).abs() < 1e-9, "{label}");
        assert!((mixture.unidentified_mass - unidentified).abs() < 1e-9, "{label}");
        assert_eq!(result.identification.status, IdentificationStatus::GraphDependent, "{label}");
        let values = published(&result, &label);
        assert!((values[0] - truth).abs() <= 0.1 * truth, "{label}: {} vs {truth}", values[0]);
    }
}

#[test]
fn graph_posterior_derivative_planted_off_support_point_is_detected() {
    let pin = fixture();
    let weights = numbers(&pin["atoms"], "fully_identified_weights");
    for bayesian in [false, true] {
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let label = format!("bayesian={bayesian} {suite:?}");
            let run = |at: f64| {
                let gp = posterior(&[&PLAIN_EDGES, &CONFOUNDED_EDGES], &weights);
                study(gp, &point(DerivativeScale::Identity, at), bayesian, suite)
                    .build()
                    .unwrap()
                    .run(&ExecutionContext::for_tests(820))
            };
            let clean = run(number(&pin, "at")).unwrap_or_else(|e| panic!("{label}: {e}"));
            assert!(report(&clean, "derivative.support").is_some_and(|r| r.passed), "{label}");
            match run(number(&pin, "off_support_at")) {
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
