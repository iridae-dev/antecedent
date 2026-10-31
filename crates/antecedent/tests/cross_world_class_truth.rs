//! A counterfactual and a nested natural direct effect on an ADMG, a temporal ADMG, a CPDAG and a
//! PAG, supplied or accepted, on both inference sides, against the exact functionals of the
//! enumerated count tables of `conformance/estimate/cross_world_class_truth`.
//!
//! On a latent-confounded class the counterfactual is the population-level cross-world mean
//! `E[Y_x | X = x'] - E[Y | X = x']` decided by ID*; on a CPDAG it is the per-unit route on every
//! DAG completion. The natural direct effect is the path-specific effect along the edges that
//! avoid the mediator. A mediator-outcome latent confounder refuses the effect as not identified.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    reason = "test fixtures: small exact indices and one coordinate walk per test"
)]

use std::sync::Arc;

use antecedent::analysis::{RetainedOperation, StudyBuilder};
use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, InferenceMode, IntoGraphInput, RefuteSuite, Study,
};
use antecedent_core::{
    CausalQuery, CounterfactualQuery, ExecutionContext, Intervention, NestedCounterfactualQuery,
    Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, Cpdag, DenseNodeId, Pag, TemporalAdmg};

fn pin() -> serde_json::Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/cross_world_class_truth/expected.json"
    ))
    .unwrap()
}

fn expected(pin: &serde_json::Value, case: &str) -> f64 {
    pin["cases"][case]["expected"].as_f64().unwrap()
}

fn d(raw: u32) -> DenseNodeId {
    DenseNodeId::from_raw(raw)
}

fn v(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

/// The columns of a count table: every cell repeated `count` times.
fn columns_of(pin: &serde_json::Value, name: &str) -> (Vec<String>, Vec<Vec<f64>>) {
    let spec = &pin["tables"][name];
    let names: Vec<String> = spec["columns"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| c.as_str().unwrap().to_owned())
        .collect();
    let mut data: Vec<Vec<f64>> = vec![Vec::new(); names.len()];
    for cell in spec["counts"].as_array().unwrap() {
        let levels: Vec<f64> =
            cell["levels"].as_array().unwrap().iter().map(|l| l.as_f64().unwrap()).collect();
        for _ in 0..cell["count"].as_u64().unwrap() {
            for (column, level) in data.iter_mut().zip(&levels) {
                column.push(*level);
            }
        }
    }
    (names, data)
}

fn table(pin: &serde_json::Value, name: &str) -> TabularData {
    let (names, data) = columns_of(pin, name);
    TabularData::from_f64_columns(
        names.iter().zip(&data).map(|(name, column)| (name.as_str(), column.as_slice())),
    )
    .unwrap()
}

fn admg(n: u32, directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::with_variables(n);
    for &(a, b) in directed {
        graph.insert_directed(d(a), d(b)).unwrap();
    }
    for &(a, b) in bidirected {
        graph.insert_bidirected(d(a), d(b)).unwrap();
    }
    graph
}

fn pag(n: u32, directed: &[(u32, u32)]) -> Pag {
    let mut graph = Pag::with_variables(n);
    for &(a, b) in directed {
        graph.insert_directed(d(a), d(b)).unwrap();
    }
    graph
}

fn cpdag(n: u32, directed: &[(u32, u32)], undirected: &[(u32, u32)]) -> Cpdag {
    let mut graph = Cpdag::with_variables(n);
    for &(a, b) in directed {
        graph.insert_directed(d(a), d(b)).unwrap();
    }
    for &(a, b) in undirected {
        graph.insert_undirected(d(a), d(b)).unwrap();
    }
    graph
}

fn counterfactual(treatment: u32, outcome: u32) -> CausalQuery {
    CausalQuery::Counterfactual(CounterfactualQuery::new(
        v(outcome),
        Arc::from([Intervention::set(v(treatment), Value::f64(1.0))]),
    ))
}

fn nested(treatment: u32, mediator: u32, outcome: u32) -> CausalQuery {
    CausalQuery::NestedCounterfactual(
        NestedCounterfactualQuery::new(v(treatment), v(mediator), v(outcome)).unwrap(),
    )
}

fn inference(bayesian: bool) -> InferenceMode {
    if bayesian {
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(200))
    } else {
        InferenceMode::Frequentist
    }
}

fn builder(
    data: TabularData,
    graph: impl IntoGraphInput + Into<AcceptedGraph>,
    accepted: bool,
    query: CausalQuery,
    bayesian: bool,
    suite: RefuteSuite,
) -> StudyBuilder {
    let base = Study::tabular(data);
    let base = if accepted { base.graph(graph.into()) } else { base.graph(graph) };
    base.query(query).inference(inference(bayesian)).refute(suite).bootstrap_replicates(0)
}

/// `x` on a grid, `y = intercept + slope x` exactly (the line of the fixture).
fn line_table(pin: &serde_json::Value) -> TabularData {
    let line = &pin["line"];
    let (rows, intercept, slope) = (
        line["rows"].as_u64().unwrap(),
        line["intercept"].as_f64().unwrap(),
        line["slope"].as_f64().unwrap(),
    );
    let x: Vec<f64> = (0..rows).map(|i| i as f64 / 20.0 - 2.0).collect();
    let y: Vec<f64> = x.iter().map(|x| intercept + slope * x).collect();
    TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap()
}

/// A coordinate of the matrix this file licenses: a graph class, its data, query and truth.
struct Case {
    class: &'static str,
    query_name: &'static str,
    build: Box<dyn Fn(bool, bool, RefuteSuite) -> (StudyBuilder, TabularData)>,
    truth: f64,
    tolerance: f64,
}

fn cases(pin: &serde_json::Value) -> Vec<Case> {
    let front = || table(pin, "front_door");
    let pin = pin.clone();
    let mut out: Vec<Case> = Vec::new();
    let t = |case: &str| expected(&pin, case);
    macro_rules! case {
        ($class:expr, $name:expr, $truth:expr, $tol:expr, $data:expr, $graph:expr, $query:expr) => {{
            let data_pin = pin.clone();
            out.push(Case {
                class: $class,
                query_name: $name,
                build: Box::new(move |accepted, bayesian, suite| {
                    let data = $data(&data_pin);
                    (builder(data.clone(), $graph(), accepted, $query, bayesian, suite), data)
                }),
                truth: $truth,
                tolerance: $tol,
            });
        }};
    }
    let _ = front;
    case!(
        "Admg",
        "Counterfactual",
        t("admg_counterfactual"),
        1e-9,
        |p: &serde_json::Value| table(p, "front_door"),
        || admg(3, &[(0, 1), (1, 2)], &[(0, 2)]),
        counterfactual(0, 2)
    );
    case!(
        "TemporalAdmg",
        "Counterfactual",
        t("admg_counterfactual"),
        1e-9,
        |p: &serde_json::Value| table(p, "front_door"),
        || TemporalAdmg::from_static_contemporaneous(&admg(3, &[(0, 1), (1, 2)], &[(0, 2)]))
            .unwrap(),
        counterfactual(0, 2)
    );
    case!(
        "Pag",
        "Counterfactual",
        t("pag_counterfactual"),
        1e-9,
        |p: &serde_json::Value| table(p, "side_path_chain"),
        || pag(4, &[(0, 1), (0, 2), (2, 3), (1, 3)]),
        counterfactual(1, 3)
    );
    case!(
        "Admg",
        "NestedCounterfactualEffect",
        t("admg_natural_direct"),
        1e-9,
        |p: &serde_json::Value| table(p, "front_door"),
        || admg(3, &[(0, 1), (0, 2), (1, 2)], &[]),
        nested(0, 1, 2)
    );
    case!(
        "TemporalAdmg",
        "NestedCounterfactualEffect",
        t("admg_natural_direct"),
        1e-9,
        |p: &serde_json::Value| table(p, "front_door"),
        || TemporalAdmg::from_static_contemporaneous(&admg(3, &[(0, 1), (0, 2), (1, 2)], &[]))
            .unwrap(),
        nested(0, 1, 2)
    );
    case!(
        "Pag",
        "NestedCounterfactualEffect",
        t("pag_natural_direct"),
        1e-9,
        |p: &serde_json::Value| table(p, "mediated_two_parents"),
        || pag(5, &[(0, 2), (1, 3), (2, 3), (2, 4), (3, 4)]),
        nested(2, 3, 4)
    );
    case!(
        "Cpdag",
        "NestedCounterfactualEffect",
        t("cpdag_path_specific"),
        1e-9,
        |p: &serde_json::Value| table(p, "two_mediators"),
        || cpdag(4, &[(0, 1), (0, 2), (1, 3), (2, 3)], &[]),
        nested(0, 1, 3)
    );
    case!(
        "Cpdag",
        "Counterfactual",
        pin["line"]["mean"].as_f64().unwrap(),
        1e-6,
        |p: &serde_json::Value| line_table(p),
        || cpdag(2, &[], &[(0, 1)]),
        counterfactual(0, 1)
    );
    out
}

fn tolerance(case: &Case, bayesian: bool) -> f64 {
    if bayesian && case.class != "Cpdag" {
        0.03
    } else if bayesian {
        0.05
    } else {
        case.tolerance
    }
}

#[test]
fn class_cross_world_effects_match_the_exact_functionals_of_the_count_tables() {
    let pin = pin();
    let ctx = ExecutionContext::for_tests(5);
    for case in cases(&pin) {
        for accepted in [false, true] {
            for bayesian in [false, true] {
                let label = format!(
                    "{} {} accepted={accepted} bayesian={bayesian}",
                    case.class, case.query_name
                );
                let (b, _) = (case.build)(accepted, bayesian, RefuteSuite::None);
                let study = b.build().unwrap_or_else(|e| panic!("{label}: {e}"));
                assert_eq!(study.support_status(), Some(CellStatus::Licensed), "{label}");
                let result = study.run(&ctx).unwrap_or_else(|e| panic!("{label}: {e}"));
                assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
                let effect = result.effect();
                assert!(
                    (effect - case.truth).abs() <= tolerance(&case, bayesian),
                    "{label}: {effect} vs {}",
                    case.truth
                );
                if case.class == "Cpdag" && case.query_name == "Counterfactual" {
                    // The completions x -> y and y -> x disagree: the enumeration-weighted mean
                    // with the identified set between their values.
                    let mixture = result.structural_response.as_ref().expect(&label);
                    assert_eq!(mixture.atoms.len(), 2, "{label}");
                    let set = mixture.identified_set.as_ref().expect(&label);
                    let want: Vec<f64> = pin["line"]["identified_set"]
                        .as_array()
                        .unwrap()
                        .iter()
                        .map(|b| b.as_f64().unwrap())
                        .collect();
                    assert!((set.lower[0] - want[0]).abs() < 1e-6, "{label}: {:?}", set.lower);
                    assert!((set.upper[0] - want[1]).abs() < 1e-6, "{label}: {:?}", set.upper);
                }
            }
        }
    }
}

#[test]
fn class_cross_world_effects_that_are_not_identified_are_refused_with_the_typed_witness() {
    let pin = pin();
    let ctx = ExecutionContext::for_tests(6);
    // The bow arc x <-> y: the effect on the treated is not identified from the observed law.
    let bow = admg(3, &[(0, 1), (1, 2), (0, 2)], &[(0, 2)]);
    let error = builder(
        table(&pin, "front_door"),
        bow,
        false,
        counterfactual(0, 2),
        false,
        RefuteSuite::None,
    )
    .build()
    .and_then(|study| study.run(&ctx))
    .unwrap_err()
    .to_string();
    assert!(error.contains("route_not_supported: counterfactual_id."), "{error}");
    assert!(error.contains("not a proof of non-identifiability"), "{error}");
    // A mediator-outcome latent confounder leaves the natural direct effect unidentified.
    let confounded = admg(3, &[(0, 1), (0, 2), (1, 2)], &[(1, 2)]);
    let error = builder(
        table(&pin, "front_door"),
        confounded,
        false,
        nested(0, 1, 2),
        false,
        RefuteSuite::None,
    )
    .build()
    .and_then(|study| study.run(&ctx))
    .unwrap_err()
    .to_string();
    assert!(error.contains("not identified"), "{error}");
}

#[test]
fn class_cross_world_effects_execute_from_the_retained_plan_without_the_builder() {
    let pin = pin();
    let ctx = ExecutionContext::for_tests(7);
    for case in cases(&pin) {
        for accepted in [false, true] {
            for bayesian in [false, true] {
                let label = format!(
                    "{} {} accepted={accepted} bayesian={bayesian}",
                    case.class, case.query_name
                );
                let (builder, data) = (case.build)(accepted, bayesian, RefuteSuite::None);
                let study = builder.clone().build().unwrap();
                let prepared = study.prepare(&ctx).unwrap();
                drop(builder);
                drop(study);
                assert!(
                    matches!(prepared.retained_operation(), RetainedOperation::StaticCrossWorld),
                    "{label}"
                );
                let estimator = prepared.plan().logical.record.estimator.clone();
                let expected = if case.class == "Cpdag" && case.query_name == "Counterfactual" {
                    "gcm.fit"
                } else {
                    "functional.effect"
                };
                assert_eq!(estimator.as_deref(), Some(expected), "{label}");
                let result = prepared.estimate(&data, &ctx).unwrap();
                assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
                assert!(
                    (result.effect() - case.truth).abs() <= tolerance(&case, bayesian),
                    "{label}: {}",
                    result.effect()
                );
            }
        }
    }
}

fn report<'a>(
    result: &'a antecedent::StudyResult,
    refuter: &str,
) -> Option<&'a antecedent_validate::RefutationReport> {
    result.refutations.iter().find(|report| report.refuter.as_ref() == refuter)
}

fn refused_refuters(result: &antecedent::StudyResult) -> String {
    result
        .diagnostics
        .iter()
        .filter(|d| d.code.as_ref() == "refute.functional.refuter_refused")
        .map(|d| d.message.to_string())
        .collect::<Vec<_>>()
        .join(" | ")
}

#[test]
fn class_cross_world_cheap_and_full_validation_refutes_the_exact_functionals() {
    let pin = pin();
    let ctx = ExecutionContext::for_tests(8);
    for case in cases(&pin) {
        if case.class == "Pag" && case.query_name == "Counterfactual" {
            continue;
        }
        for accepted in [false, true] {
            for bayesian in [false, true] {
                for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                    let label = format!(
                        "{} {} accepted={accepted} bayesian={bayesian} {suite:?}",
                        case.class, case.query_name
                    );
                    let (b, _) = (case.build)(accepted, bayesian, suite);
                    let study = b.build().unwrap_or_else(|e| panic!("{label}: {e}"));
                    assert_eq!(study.support_status(), Some(CellStatus::Licensed), "{label}");
                    let result = study.run(&ctx).unwrap_or_else(|e| panic!("{label}: {e}"));
                    assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
                    assert!(
                        (result.effect() - case.truth).abs() <= tolerance(&case, bayesian),
                        "{label}: {}",
                        result.effect()
                    );
                    // The permuted treatment removes the effect the identified functional reads.
                    let placebo = report(&result, "target.placebo_treatment").expect(&label);
                    assert!(placebo.passed, "{label}: {}", placebo.refuted_ate);
                    assert!(placebo.refuted_ate.abs() < 0.1, "{label}: {}", placebo.refuted_ate);
                    if suite == RefuteSuite::Full {
                        let outcome = report(&result, "target.placebo_outcome").expect(&label);
                        assert!(outcome.passed && outcome.refuted_ate.abs() < 0.05, "{label}");
                        if !bayesian {
                            let bootstrap =
                                report(&result, "target.bootstrap_stability").expect(&label);
                            assert!(bootstrap.passed, "{label}");
                            assert!(
                                (bootstrap.refuted_ate - bootstrap.original_ate).abs() < 0.02,
                                "{label}"
                            );
                        }
                    }
                    // The cross-world sensitivity range reads the treatment's parents in a supplied
                    // ADMG; every other class says in a typed diagnostic that it did not run.
                    if case.class == "Admg" {
                        let range =
                            report(&result, "target.cross_world_sensitivity").expect(&label);
                        assert!(range.passed, "{label}");
                    } else {
                        assert!(
                            report(&result, "target.cross_world_sensitivity").is_none(),
                            "{label}"
                        );
                        assert!(
                            refused_refuters(&result).contains("cross_world_sensitivity"),
                            "{label}"
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn class_cross_world_validation_flags_a_planted_weak_effect_on_an_admg() {
    let pin = pin();
    let ctx = ExecutionContext::for_tests(9);
    let front = admg(3, &[(0, 1), (1, 2)], &[(0, 2)]);
    let markov = admg(3, &[(0, 1), (0, 2), (1, 2)], &[]);
    for (name, query, graph, truth_case, clean_case) in [
        (
            "Counterfactual",
            counterfactual(0, 2),
            front,
            "admg_counterfactual_weak",
            "admg_counterfactual",
        ),
        (
            "NestedCounterfactualEffect",
            nested(0, 1, 2),
            markov,
            "admg_natural_direct_weak",
            "admg_natural_direct",
        ),
    ] {
        for bayesian in [false, true] {
            let null = builder(
                table(&pin, "weak_front_door"),
                graph.clone(),
                false,
                query.clone(),
                bayesian,
                RefuteSuite::Cheap,
            )
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap();
            assert!(
                (null.effect() - expected(&pin, truth_case)).abs() < 0.03,
                "{name} bayesian={bayesian}: {}",
                null.effect()
            );
            // A weak effect on the treated is sign-fragile under the cross-world dependence; the
            // weak natural direct effect of the same table is only checked against its exact value
            // (its range stays one-signed), and a mediator-outcome confounder is the planted
            // violation the identifier itself refuses.
            if name == "Counterfactual" {
                let flagged = report(&null, "target.cross_world_sensitivity").expect(name);
                assert!(!flagged.passed, "{name} bayesian={bayesian}: sign-fragile");
            }
            let clean = builder(
                table(&pin, "front_door"),
                graph.clone(),
                false,
                query.clone(),
                bayesian,
                RefuteSuite::Cheap,
            )
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap();
            assert!((clean.effect() - expected(&pin, clean_case)).abs() < 0.03, "{name}");
            assert!(report(&clean, "target.cross_world_sensitivity").expect(name).passed, "{name}");
        }
    }
}

#[test]
fn class_cross_world_validation_executes_from_the_retained_plan_without_the_builder() {
    let pin = pin();
    let ctx = ExecutionContext::for_tests(10);
    for case in cases(&pin) {
        if case.class == "Pag" && case.query_name == "Counterfactual" {
            continue;
        }
        for accepted in [false, true] {
            for bayesian in [false, true] {
                for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                    let label = format!(
                        "{} {} accepted={accepted} bayesian={bayesian} {suite:?}",
                        case.class, case.query_name
                    );
                    let (builder, data) = (case.build)(accepted, bayesian, suite);
                    let study = builder.clone().build().unwrap();
                    let prepared = study.prepare(&ctx).unwrap();
                    drop(builder);
                    drop(study);
                    assert!(
                        matches!(
                            prepared.retained_operation(),
                            RetainedOperation::StaticCrossWorld
                        ),
                        "{label}"
                    );
                    let estimator = prepared.plan().logical.record.estimator.clone();
                    let expected = if case.class == "Cpdag" && case.query_name == "Counterfactual" {
                        "gcm.fit"
                    } else {
                        "functional.effect"
                    };
                    assert_eq!(estimator.as_deref(), Some(expected), "{label}");
                    let result =
                        prepared.estimate(&data, &ctx).unwrap_or_else(|e| panic!("{label}: {e}"));
                    assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
                    assert!(
                        (result.effect() - case.truth).abs() <= tolerance(&case, bayesian),
                        "{label}"
                    );
                    assert!(report(&result, "target.placebo_treatment").is_some(), "{label}");
                }
            }
        }
    }
}

#[test]
fn a_counterfactual_on_a_pag_is_refused_under_cheap_and_full_validation() {
    let pin = pin();
    for accepted in [false, true] {
        for bayesian in [false, true] {
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                let (b, _) = cases(&pin)
                    .into_iter()
                    .find(|c| c.class == "Pag" && c.query_name == "Counterfactual")
                    .map(|c| (c.build)(accepted, bayesian, suite))
                    .unwrap();
                let error = b.build().unwrap_err().to_string();
                assert!(error.starts_with("refused:"), "{error}");
                assert!(error.contains("placebo treatment permutes the treatment"), "{error}");
            }
        }
    }
}
