//! Known-truth evidence for temporal derivatives over a graph posterior of TemporalDag, TemporalCpdag
//! and TemporalPag atoms.
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
use antecedent_core::{IdentificationStatus,
    CausalQuery, DerivativeScale, DerivativeWeighting, ExecutionContext, Lag, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, TemporalPolicy, TemporalResponseSpec,
    VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_discovery::adjacency_mask::set_edge as set_class_edge;
use antecedent_discovery::{
    AdjacencyMask, GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, set_edge,
};
use antecedent_prob::InferenceDiagnostics;
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

fn series(witness: bool) -> TimeSeriesData {
    let n = 400;
    let mut state = 20_240_607_u64;
    let mut gauss = || {
        let mut sum = 0.0;
        for _ in 0..12 {
            state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
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
    let mut columns: Vec<(&str, &[f64])> =
        vec![("x", x.as_slice()), ("b", b.as_slice()), ("y", y.as_slice()), ("v", v.as_slice())];
    if witness {
        columns.push(("r", r.as_slice()));
    }
    TimeSeriesData::from_f64_columns(columns, 1).unwrap()
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


const BASE: usize = 4;
/// `x(t-1) -> y(t)`, `b(t-1) -> y(t)`, `x(t-1) -> v(t)`, `b(t-1) -> v(t)` over the columns `x, b, y, v`.
const LAGGED: [(usize, usize); 4] = [(0, 2), (1, 2), (0, 3), (1, 3)];

fn lag_bit(n: usize, from: usize, to: usize) -> u64 {
    1_u64 << (from * n + to)
}

/// A `TemporalDag` posterior: each template is its contemporaneous edges, lagged edges and weight.
struct Template {
    contemporaneous: Vec<(usize, usize)>,
    lagged: Vec<(usize, usize)>,
    weight: f64,
}

fn dbn_posterior(templates: &[Template]) -> GraphPosterior {
    let n = BASE;
    let adjacency: Vec<u64> = templates
        .iter()
        .map(|t| {
            t.contemporaneous.iter().fold(0_u64, |mask, &(f, to)| set_edge(mask, n, f, to, true))
        })
        .collect();
    let lag_masks: Vec<u64> = templates
        .iter()
        .map(|t| t.lagged.iter().fold(0_u64, |mask, &(f, to)| mask | lag_bit(n, f, to)))
        .collect();
    let weights: Vec<f64> = templates.iter().map(|t| t.weight).collect();
    let cell = n * n;
    GraphPosterior::new(
        n,
        weights,
        adjacency,
        vec![0.0; cell],
        vec![0.0; cell],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("temporal_graph_posterior_derivative"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_lagged_marginals(1, vec![0.0; cell])
    .unwrap()
    .with_lag_masks(lag_masks)
    .unwrap()
}

/// One class atom (a `TemporalCpdag`, or a `TemporalPag` with a witness `r` into the treatment).
fn class_posterior(pag: bool) -> GraphPosterior {
    let n = if pag { BASE + 1 } else { BASE };
    let lag_mask = LAGGED.iter().fold(0_u64, |mask, &(f, to)| mask | lag_bit(n, f, to));
    let mut adjacency = AdjacencyMask::EMPTY;
    if pag {
        adjacency = set_class_edge(adjacency, n, BASE, 0, true);
    }
    let cell = n * n;
    GraphPosterior::from_atoms(
        n,
        vec![1.0],
        vec![adjacency],
        vec![0.0; cell],
        vec![0.0; cell],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("temporal_class_posterior_derivative"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(if pag { GraphPosteriorAtomKind::Pag } else { GraphPosteriorAtomKind::Cpdag })
    .with_lagged_marginals(1, vec![0.0; cell])
    .unwrap()
    .with_lag_masks(vec![lag_mask])
    .unwrap()
}

fn valid(weight: f64) -> Template {
    Template { contemporaneous: vec![], lagged: LAGGED.to_vec(), weight }
}

#[derive(Clone, Copy, Debug)]
enum Atoms {
    Dag,
    Cpdag,
    Pag,
}

fn study(
    data: &TimeSeriesData,
    posterior: GraphPosterior,
    functional: &ResponseFunctional,
    bayesian: bool,
    suite: RefuteSuite,
) -> antecedent::StudyBuilder {
    let query = ResponseQuery::new(functional.clone())
        .with_temporal(TemporalResponseSpec::new([1], TemporalPolicy::pulse(-1), Some(1)).unwrap());
    let builder = Study::series(data.clone())
        .query(CausalQuery::Response(query))
        .graph_posterior(posterior)
        .refute(suite)
        .bootstrap_replicates(0);
    if bayesian {
        builder.inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(99)))
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
    result: &'a antecedent::StudyResult,
    name: &str,
) -> Option<&'a antecedent_validate::RefutationReport> {
    result.refutations.iter().find(|report| report.refuter.as_ref() == name)
}

fn posterior_of(atoms: Atoms) -> GraphPosterior {
    match atoms {
        Atoms::Dag => dbn_posterior(&[valid(1.0)]),
        Atoms::Cpdag => class_posterior(false),
        Atoms::Pag => class_posterior(true),
    }
}

fn atoms_suites(atoms: Atoms) -> Vec<RefuteSuite> {
    match atoms {
        Atoms::Dag => vec![RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full],
        Atoms::Cpdag | Atoms::Pag => vec![RefuteSuite::None, RefuteSuite::Cheap, RefuteSuite::Full],
    }
}

#[test]
fn temporal_graph_posterior_derivative_routes_and_refuter_states_known_truth() {
    let pin = fixture();
    let value_band = number(&pin, "value_relative_tolerance");
    let difference_band = number(&pin, "finite_difference_relative_tolerance");
    let slope = number(&pin, "point");
    for atoms in [Atoms::Dag, Atoms::Cpdag, Atoms::Pag] {
        let data = series(matches!(atoms, Atoms::Pag));
        for (key, functional, has_difference) in functionals(number(&pin, "at")) {
            let truths = numbers(&pin, key);
            for bayesian in [false, true] {
                for suite in atoms_suites(atoms) {
                    let label = format!("{atoms:?} {key} bayesian={bayesian} {suite:?}");
                    let builder = study(&data, posterior_of(atoms), &functional, bayesian, suite)
                        .build()
                        .unwrap_or_else(|e| panic!("{label}: {e}"));
                    assert_eq!(builder.support_status(), Some(CellStatus::Licensed), "{label}");
                    let context = ExecutionContext::for_tests(821);
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
                        Some(if bayesian { "response.temporal.bayesian" } else { "temporal.response.gcomp" }),
                        "{label}"
                    );
                    let mixture = result.structural_response.as_ref();
                    if let (Some(mixture), Atoms::Dag) = (mixture, atoms) {
                        assert!((mixture.identified_mass - 1.0).abs() < 1e-9, "{label}");
                    }
                    let values = published(&result, &label);
                    assert_eq!(values.len(), truths.len(), "{label}");
                    for (value, truth) in values.iter().zip(&truths) {
                        assert!(
                            (value - truth).abs() <= (value_band * truth.abs()).max(0.15),
                            "{label}: mixture {value} vs fixture {truth}"
                        );
                    }
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
                        assert!(difference.passed, "{label}: {:?}", difference.failure_condition);
                        assert!(
                            (difference.refuted_ate - slope).abs() <= difference_band * slope,
                            "{label}: unit-step difference {} vs closed-form {slope}",
                            difference.refuted_ate
                        );
                    }
                    for name in ["placebo.treatment", "data.subset"] {
                        let check = report(&result, name);
                        assert_eq!(check.is_some(), suite == RefuteSuite::Full, "{label}: {name}");
                        if let Some(check) = check {
                            assert!(check.passed, "{label}: {name}: {:?}", check.failure_condition);
                        }
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

#[test]
fn temporal_graph_posterior_lost_mass_is_stated_and_the_identified_mixture_is_the_truth() {
    let pin = fixture();
    let data = series(false);
    let truth = number(&pin, "point");
    // `y -> v` and `v -> y` at the same time: not a DAG, so the sample identifies nothing.
    let cyclic = Template { contemporaneous: vec![(2, 3), (3, 2)], lagged: LAGGED.to_vec(), weight: 0.3 };
    let functional = functionals(number(&pin, "at")).remove(0).1;
    for bayesian in [false, true] {
        let label = format!("bayesian={bayesian}");
        let posterior = dbn_posterior(&[valid(0.7), cyclic_clone(&cyclic)]);
        let builder = study(&data, posterior, &functional, bayesian, RefuteSuite::None)
            .build()
            .unwrap();
        assert_eq!(builder.support_status(), Some(CellStatus::Licensed), "{label}");
        let result = builder.run(&ExecutionContext::for_tests(822)).unwrap();
        let mixture = result.structural_response.as_ref().expect("a structural mixture");
        assert!((mixture.identified_mass - 0.7).abs() < 1e-9, "{label}: {}", mixture.identified_mass);
        assert!((mixture.unidentified_mass - 0.3).abs() < 1e-9, "{label}");
        assert_eq!(result.identification.status, IdentificationStatus::GraphDependent, "{label}");
        let values = published(&result, &label);
        assert!((values[0] - truth).abs() <= 0.1 * truth, "{label}: {} vs {truth}", values[0]);
    }
}

fn cyclic_clone(template: &Template) -> Template {
    Template {
        contemporaneous: template.contemporaneous.clone(),
        lagged: template.lagged.clone(),
        weight: template.weight,
    }
}

#[test]
fn temporal_graph_posterior_planted_off_support_point_is_detected() {
    let pin = fixture();
    let clean = functionals(number(&pin, "at")).remove(0).1;
    let planted = functionals(number(&pin, "off_support_at")).remove(0).1;
    for atoms in [Atoms::Dag, Atoms::Cpdag, Atoms::Pag] {
        let data = series(matches!(atoms, Atoms::Pag));
        for bayesian in [false, true] {
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                let label = format!("{atoms:?} bayesian={bayesian} {suite:?}");
                let run = |functional: &ResponseFunctional| {
                    study(&data, posterior_of(atoms), functional, bayesian, suite)
                        .build()
                        .unwrap()
                        .run(&ExecutionContext::for_tests(823))
                };
                let clean_result = run(&clean).unwrap_or_else(|e| panic!("{label}: {e}"));
                assert!(
                    report(&clean_result, "derivative.support").is_some_and(|r| r.passed),
                    "{label}"
                );
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
