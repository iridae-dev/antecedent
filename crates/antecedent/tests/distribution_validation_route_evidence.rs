//! Cheap and full validation of an interventional distribution on a supplied or accepted ADMG,
//! TemporalAdmg, CPDAG and PAG.
//!
//! The expected law is read from the retained fixtures, where an independent generator derives it
//! from the structural equations (`conformance/estimate/admg_frontdoor_distribution` for the
//! mixed graphs, `conformance/estimate/interventional_distribution_classes` for the classes). The
//! refuter-state expectations are derived from the table alone: the placebo centre is the
//! observed marginal share of the outcome counted from the rows, and the stability refuters
//! must stay at the enumerated law. A planted high-leverage table has its identified value
//! computed by hand from the rows, and either a refuter flags it or the typed `did not run`
//! diagnostic names the refuter that could not speak.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines, reason = "one function walks a coordinate's full lifecycle")]

use antecedent::analysis::RetainedOperation;
use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, EstimatorId, IdentifierId, InferenceMode,
    RefuteSuite, StructureSource, Study,
};
use antecedent_contract::SlotAvailability;
use antecedent_core::{
    CausalQuery, ExecutionContext, Intervention, InterventionalDistributionQuery, Value, VariableId,
};
use antecedent_data::TabularData;
use antecedent_graph::{Admg, Cpdag, DenseNodeId, Pag, TemporalAdmg};
use antecedent_io::consume_analysis_result;
use serde_json::Value as Json;

/// The validation states resample the table, so they run on the exact law at more rows: 5 times
/// the 800 rows of the front-door table, 40 times the 128 rows of the class table.
const FRONT_DOOR_SCALE: u64 = 5;
const CLASS_SCALE: u64 = 40;
/// Dirichlet-posterior-mean tolerance around the enumerated truth on the scaled table.
const BAYESIAN_TOLERANCE: f64 = 2.0e-2;
/// Distance of a refit replicate centre from its expected value.
const REFIT_TOLERANCE: f64 = 2.0e-2;

fn front_door_pin() -> Json {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/admg_frontdoor_distribution/expected.json"
    ))
    .unwrap()
}

fn class_pin() -> Json {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/interventional_distribution_classes/expected.json"
    ))
    .unwrap()
}

fn columns_of(case: &Json) -> Vec<&str> {
    case["columns"].as_array().unwrap().iter().map(|column| column.as_str().unwrap()).collect()
}

fn expand(case: &Json, factor: u64) -> TabularData {
    let columns = columns_of(case);
    let mut values: Vec<Vec<f64>> = vec![Vec::new(); columns.len()];
    for cell in case["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap() * factor).unwrap();
        for (index, name) in columns.iter().enumerate() {
            values[index].extend(std::iter::repeat_n(cell[name].as_f64().unwrap(), count));
        }
    }
    let pairs: Vec<(&str, &[f64])> =
        columns.iter().zip(&values).map(|(name, values)| (*name, values.as_slice())).collect();
    TabularData::from_f64_columns(pairs).unwrap()
}

/// The share of rows with `y = 1`, counted from the contingency table.
fn observed_share(case: &Json, outcome: &str) -> f64 {
    let (mut ones, mut total) = (0.0, 0.0);
    for cell in case["contingency_table"].as_array().unwrap() {
        let count = cell["count"].as_f64().unwrap();
        total += count;
        if cell[outcome].as_f64().unwrap() > 0.5 {
            ones += count;
        }
    }
    ones / total
}

fn node(case: &Json, name: &str) -> u32 {
    let index = columns_of(case).iter().position(|column| *column == name).unwrap();
    u32::try_from(index).unwrap()
}

fn dense(case: &Json, name: &Json) -> DenseNodeId {
    DenseNodeId::from_raw(node(case, name.as_str().unwrap()))
}

fn variable_count(case: &Json) -> u32 {
    u32::try_from(columns_of(case).len()).unwrap()
}

fn front_door_admg(case: &Json) -> Admg {
    let mut graph = Admg::with_variables(variable_count(case));
    for edge in case["graph"]["directed_edges"].as_array().unwrap() {
        graph.insert_directed(dense(case, &edge[0]), dense(case, &edge[1])).unwrap();
    }
    for edge in case["graph"]["bidirected_edges"].as_array().unwrap() {
        graph.insert_bidirected(dense(case, &edge[0]), dense(case, &edge[1])).unwrap();
    }
    graph
}

fn compelled_cpdag(case: &Json) -> Cpdag {
    let mut cpdag = Cpdag::with_variables(variable_count(case));
    for edge in case["graph"]["directed_edges"].as_array().unwrap() {
        cpdag.insert_directed(dense(case, &edge[0]), dense(case, &edge[1])).unwrap();
    }
    cpdag
}

fn decided_pag(case: &Json) -> Pag {
    let mut pag = Pag::with_variables(variable_count(case));
    for edge in case["graph"]["directed_edges"].as_array().unwrap() {
        pag.insert_directed(dense(case, &edge[0]), dense(case, &edge[1])).unwrap();
    }
    pag
}

fn inference_of(bayesian: bool) -> InferenceMode {
    if bayesian {
        InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(256))
    } else {
        InferenceMode::Frequentist
    }
}

fn suite_of(validation: &str) -> RefuteSuite {
    match validation {
        "cheap" => RefuteSuite::Cheap,
        "full" => RefuteSuite::Full,
        other => panic!("unexpected validation {other}"),
    }
}

fn study_of(
    data: &TabularData,
    query: &InterventionalDistributionQuery,
    bayesian: bool,
    suite: RefuteSuite,
) -> antecedent::analysis::StudyBuilder {
    Study::tabular(data.clone())
        .query(CausalQuery::Distribution(query.clone()))
        .identifier(IdentifierId::GeneralId)
        .estimator(EstimatorId::FunctionalDistribution)
        .inference(inference_of(bayesian))
        .refute(suite)
        .bootstrap_replicates(0)
}

fn rebind_estimate(
    prepared: &mut antecedent::PreparedStudy,
    data: TabularData,
    ctx: &ExecutionContext,
) -> Result<antecedent::StudyResult, antecedent::CausalError> {
    let branch = prepared.rebind_data(data, ctx)?.materialize(ctx)?;
    let result = branch.estimate_retained(ctx)?;
    *prepared = branch;
    Ok(result)
}

fn probability_of(result: &antecedent::StudyResult, level: f64) -> f64 {
    result
        .distribution()
        .expect("distribution result")
        .atoms
        .iter()
        .find(|atom| atom.outcomes.iter().any(|(_, value)| *value == Value::f64(level)))
        .unwrap_or_else(|| panic!("no atom at {level}"))
        .probability
}

/// The checks of the refuter state against values derived from the table.
fn assert_refuter_state(
    coordinate: &str,
    result: &antecedent::StudyResult,
    truth_one: f64,
    observed_one: f64,
    bayesian: bool,
    validation: &str,
) {
    let distribution = result.distribution().expect("distribution result");
    // The refuted vector lists the atoms by outcome value, so its first coordinate is the lowest
    // level the table carries.
    let lowest = distribution
        .atoms
        .iter()
        .map(|atom| atom.outcomes[0].1.as_f64().unwrap())
        .fold(f64::INFINITY, f64::min);
    let first_is_one = lowest > 0.5;
    let of_first = |one: f64| if first_is_one { one } else { 1.0 - one };
    let names: Vec<&str> = result.refutations.iter().map(|r| r.refuter.as_ref()).collect();
    // An interventional law has no treatment-keyed null contrast.
    assert!(!names.contains(&"target.placebo_treatment"), "{coordinate}: {names:?}");
    assert!(names.contains(&"target.distribution_placebo"), "{coordinate}: {names:?}");
    assert!(
        names.iter().any(|n| n.ends_with("subset_stability") || n.contains("prior_sensitivity")),
        "{coordinate}: {names:?}"
    );
    if validation == "full" && !bayesian {
        assert!(names.contains(&"target.bootstrap_stability"), "{coordinate}: {names:?}");
    }
    // The intervened law is far from the observed marginal, so the placebo is informative.
    assert!((truth_one - observed_one).abs() > 0.05, "{coordinate}: {truth_one} vs {observed_one}");
    for report in &result.refutations {
        let name = report.refuter.as_ref();
        if name == "target.distribution_placebo" {
            assert!(report.passed, "{coordinate}: {name} {:?}", report.failure_condition);
            let expected = of_first(observed_one);
            assert!(
                (report.refuted_ate - expected).abs() < REFIT_TOLERANCE,
                "{coordinate}: {name} {} vs marginal {expected}",
                report.refuted_ate
            );
        } else if name.ends_with("stability") {
            assert!(report.passed, "{coordinate}: {name} {:?}", report.failure_condition);
            let expected = of_first(truth_one);
            assert!(
                (report.refuted_ate - expected).abs() < REFIT_TOLERANCE,
                "{coordinate}: {name} {} vs {expected} (original {}, {} replicates, first atom is one: \
                 {first_is_one}, all reports {:?})",
                report.refuted_ate,
                report.original_ate,
                report.replicates,
                result
                    .refutations
                    .iter()
                    .map(|r| (r.refuter.to_string(), r.original_ate, r.refuted_ate))
                    .collect::<Vec<_>>()
            );
        }
    }
}

struct Coordinate<'a> {
    class: &'a str,
    accepted: bool,
    bayesian: bool,
    validation: &'a str,
}

fn verify_coordinate(spec: &Coordinate<'_>) {
    let Coordinate { class, accepted, bayesian, validation } = *spec;
    let structure = if accepted { "accepted" } else { "explicit" };
    let inference = if bayesian { "Bayesian" } else { "Frequentist" };
    let coordinate =
        format!("InterventionalDistribution:{class}:{structure}:{inference}:{validation}");
    let fixture;
    let (case, truth_one, outcome, treatment_level) = match class {
        "Admg" | "TemporalAdmg" => {
            fixture = front_door_pin();
            let truth = fixture["truth"]
                .as_array()
                .unwrap()
                .iter()
                .find(|row| row["t"].as_f64() == Some(1.0))
                .unwrap()["p_y1"]
                .as_f64()
                .unwrap();
            (&fixture, truth, "y", 1.0)
        }
        "Cpdag" | "Pag" => {
            fixture = class_pin();
            let case = &fixture["class_point"];
            (case, case["truth"]["y1"].as_f64().unwrap(), "y", 1.0)
        }
        other => panic!("unexpected graph class {other}"),
    };
    let scale =
        if matches!(class, "Admg" | "TemporalAdmg") { FRONT_DOOR_SCALE } else { CLASS_SCALE };
    let data = expand(case, scale);
    let query = InterventionalDistributionQuery::new(
        VariableId::from_raw(node(case, outcome)),
        [Intervention::set(VariableId::from_raw(node(case, "t")), Value::f64(treatment_level))],
    );
    let ctx = ExecutionContext::for_tests(2_130);
    let base = study_of(&data, &query, bayesian, suite_of(validation));
    let builder = match (class, accepted) {
        ("Admg", false) => base.graph(front_door_admg(case)),
        ("Admg", true) => base.graph(AcceptedGraph::from(front_door_admg(case))),
        ("TemporalAdmg", false) => {
            base.graph(TemporalAdmg::from_static_contemporaneous(&front_door_admg(case)).unwrap())
        }
        ("TemporalAdmg", true) => base.graph(AcceptedGraph::from(
            TemporalAdmg::from_static_contemporaneous(&front_door_admg(case)).unwrap(),
        )),
        ("Cpdag", false) => base.graph(compelled_cpdag(case)),
        ("Cpdag", true) => base.graph(AcceptedGraph::from(compelled_cpdag(case))),
        ("Pag", false) => base.graph(decided_pag(case)),
        ("Pag", true) => base.graph(AcceptedGraph::from(decided_pag(case))),
        _ => unreachable!(),
    };
    let study = builder.clone().build().unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    let mut prepared = study.prepare(&ctx).unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    // The retained plan outlives the builder and the study that made it.
    drop(builder);
    drop(study);

    assert_eq!(prepared.query(), &CausalQuery::Distribution(query));
    assert_eq!(
        prepared.structure_source(),
        if accepted { StructureSource::Accepted } else { StructureSource::Explicit },
        "{coordinate} structure source"
    );
    assert_eq!(prepared.support_status(), Some(CellStatus::Licensed), "{coordinate}");
    let contract = prepared.contract().unwrap();
    assert_eq!(contract.query_kind.as_ref(), "InterventionalDistribution", "{coordinate}");
    assert_eq!(contract.estimator.as_deref(), Some("functional.distribution"), "{coordinate}");
    match &contract.reasoning.support {
        SlotAvailability::Available(slot) => assert_eq!(
            slot.matrix_coordinate.as_deref(),
            Some(coordinate.as_str()),
            "prepared handle must bind this exact coordinate"
        ),
        other => panic!("{coordinate} support unavailable: {other:?}"),
    }

    let result =
        prepared.estimate(&data, &ctx).unwrap_or_else(|error| panic!("{coordinate}: {error}"));
    assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{coordinate}");
    let distribution = result.distribution().expect("distribution result");
    let tolerance = if bayesian { BAYESIAN_TOLERANCE } else { 1e-9 };
    let got = probability_of(&result, 1.0);
    assert!((got - truth_one).abs() < tolerance, "{coordinate}: P(Y=1) {got} vs {truth_one}");
    assert!(
        (probability_of(&result, 0.0) - (1.0 - truth_one)).abs() < tolerance,
        "{coordinate}: P(Y=0)"
    );
    assert_refuter_state(
        &coordinate,
        &result,
        truth_one,
        observed_share(case, outcome),
        bayesian,
        validation,
    );

    // The retained program runs the same validation on the refreshed table.
    let refreshed = rebind_estimate(&mut prepared, data.clone(), &ctx)
        .unwrap_or_else(|error| panic!("{coordinate} refresh: {error}"));
    assert!(
        matches!(
            prepared.retained_operation(),
            RetainedOperation::Distribution(..) | RetainedOperation::EstimatorRoute
        ),
        "{coordinate} lost its retained program on refresh"
    );
    let names = |r: &antecedent::StudyResult| {
        let mut names: Vec<String> = r.refutations.iter().map(|x| x.refuter.to_string()).collect();
        names.sort();
        names
    };
    assert_eq!(names(&refreshed), names(&result), "{coordinate}: refresh lost a refuter");
    assert_eq!(
        refreshed.distribution().unwrap().mean.to_bits(),
        distribution.mean.to_bits(),
        "{coordinate} refresh"
    );
    assert_refuter_state(
        &coordinate,
        &refreshed,
        truth_one,
        observed_share(case, outcome),
        bayesian,
        validation,
    );
    let artifact = prepared
        .encode_contracted_result(&refreshed, &format!("distribution-{coordinate}"), &ctx)
        .unwrap();
    consume_analysis_result(&artifact).unwrap();
}

#[test]
fn distribution_admg_explicit_frequentist_cheap() {
    verify_coordinate(&Coordinate {
        class: "Admg",
        accepted: false,
        bayesian: false,
        validation: "cheap",
    });
}

#[test]
fn distribution_admg_explicit_frequentist_full() {
    verify_coordinate(&Coordinate {
        class: "Admg",
        accepted: false,
        bayesian: false,
        validation: "full",
    });
}

#[test]
fn distribution_admg_explicit_bayesian_cheap() {
    verify_coordinate(&Coordinate {
        class: "Admg",
        accepted: false,
        bayesian: true,
        validation: "cheap",
    });
}

#[test]
fn distribution_admg_explicit_bayesian_full() {
    verify_coordinate(&Coordinate {
        class: "Admg",
        accepted: false,
        bayesian: true,
        validation: "full",
    });
}

#[test]
fn distribution_admg_accepted_frequentist_cheap() {
    verify_coordinate(&Coordinate {
        class: "Admg",
        accepted: true,
        bayesian: false,
        validation: "cheap",
    });
}

#[test]
fn distribution_admg_accepted_frequentist_full() {
    verify_coordinate(&Coordinate {
        class: "Admg",
        accepted: true,
        bayesian: false,
        validation: "full",
    });
}

#[test]
fn distribution_admg_accepted_bayesian_cheap() {
    verify_coordinate(&Coordinate {
        class: "Admg",
        accepted: true,
        bayesian: true,
        validation: "cheap",
    });
}

#[test]
fn distribution_admg_accepted_bayesian_full() {
    verify_coordinate(&Coordinate {
        class: "Admg",
        accepted: true,
        bayesian: true,
        validation: "full",
    });
}

#[test]
fn distribution_temporal_admg_explicit_frequentist_cheap() {
    verify_coordinate(&Coordinate {
        class: "TemporalAdmg",
        accepted: false,
        bayesian: false,
        validation: "cheap",
    });
}

#[test]
fn distribution_temporal_admg_explicit_frequentist_full() {
    verify_coordinate(&Coordinate {
        class: "TemporalAdmg",
        accepted: false,
        bayesian: false,
        validation: "full",
    });
}

#[test]
fn distribution_temporal_admg_explicit_bayesian_cheap() {
    verify_coordinate(&Coordinate {
        class: "TemporalAdmg",
        accepted: false,
        bayesian: true,
        validation: "cheap",
    });
}

#[test]
fn distribution_temporal_admg_explicit_bayesian_full() {
    verify_coordinate(&Coordinate {
        class: "TemporalAdmg",
        accepted: false,
        bayesian: true,
        validation: "full",
    });
}

#[test]
fn distribution_temporal_admg_accepted_frequentist_cheap() {
    verify_coordinate(&Coordinate {
        class: "TemporalAdmg",
        accepted: true,
        bayesian: false,
        validation: "cheap",
    });
}

#[test]
fn distribution_temporal_admg_accepted_frequentist_full() {
    verify_coordinate(&Coordinate {
        class: "TemporalAdmg",
        accepted: true,
        bayesian: false,
        validation: "full",
    });
}

#[test]
fn distribution_temporal_admg_accepted_bayesian_cheap() {
    verify_coordinate(&Coordinate {
        class: "TemporalAdmg",
        accepted: true,
        bayesian: true,
        validation: "cheap",
    });
}

#[test]
fn distribution_temporal_admg_accepted_bayesian_full() {
    verify_coordinate(&Coordinate {
        class: "TemporalAdmg",
        accepted: true,
        bayesian: true,
        validation: "full",
    });
}

#[test]
fn distribution_cpdag_explicit_frequentist_cheap() {
    verify_coordinate(&Coordinate {
        class: "Cpdag",
        accepted: false,
        bayesian: false,
        validation: "cheap",
    });
}

#[test]
fn distribution_cpdag_explicit_frequentist_full() {
    verify_coordinate(&Coordinate {
        class: "Cpdag",
        accepted: false,
        bayesian: false,
        validation: "full",
    });
}

#[test]
fn distribution_cpdag_explicit_bayesian_cheap() {
    verify_coordinate(&Coordinate {
        class: "Cpdag",
        accepted: false,
        bayesian: true,
        validation: "cheap",
    });
}

#[test]
fn distribution_cpdag_explicit_bayesian_full() {
    verify_coordinate(&Coordinate {
        class: "Cpdag",
        accepted: false,
        bayesian: true,
        validation: "full",
    });
}

#[test]
fn distribution_cpdag_accepted_frequentist_cheap() {
    verify_coordinate(&Coordinate {
        class: "Cpdag",
        accepted: true,
        bayesian: false,
        validation: "cheap",
    });
}

#[test]
fn distribution_cpdag_accepted_frequentist_full() {
    verify_coordinate(&Coordinate {
        class: "Cpdag",
        accepted: true,
        bayesian: false,
        validation: "full",
    });
}

#[test]
fn distribution_cpdag_accepted_bayesian_cheap() {
    verify_coordinate(&Coordinate {
        class: "Cpdag",
        accepted: true,
        bayesian: true,
        validation: "cheap",
    });
}

#[test]
fn distribution_cpdag_accepted_bayesian_full() {
    verify_coordinate(&Coordinate {
        class: "Cpdag",
        accepted: true,
        bayesian: true,
        validation: "full",
    });
}

#[test]
fn distribution_pag_explicit_frequentist_cheap() {
    verify_coordinate(&Coordinate {
        class: "Pag",
        accepted: false,
        bayesian: false,
        validation: "cheap",
    });
}

#[test]
fn distribution_pag_explicit_frequentist_full() {
    verify_coordinate(&Coordinate {
        class: "Pag",
        accepted: false,
        bayesian: false,
        validation: "full",
    });
}

#[test]
fn distribution_pag_explicit_bayesian_cheap() {
    verify_coordinate(&Coordinate {
        class: "Pag",
        accepted: false,
        bayesian: true,
        validation: "cheap",
    });
}

#[test]
fn distribution_pag_explicit_bayesian_full() {
    verify_coordinate(&Coordinate {
        class: "Pag",
        accepted: false,
        bayesian: true,
        validation: "full",
    });
}

#[test]
fn distribution_pag_accepted_frequentist_cheap() {
    verify_coordinate(&Coordinate {
        class: "Pag",
        accepted: true,
        bayesian: false,
        validation: "cheap",
    });
}

#[test]
fn distribution_pag_accepted_frequentist_full() {
    verify_coordinate(&Coordinate {
        class: "Pag",
        accepted: true,
        bayesian: false,
        validation: "full",
    });
}

#[test]
fn distribution_pag_accepted_bayesian_cheap() {
    verify_coordinate(&Coordinate {
        class: "Pag",
        accepted: true,
        bayesian: true,
        validation: "cheap",
    });
}

#[test]
fn distribution_pag_accepted_bayesian_full() {
    verify_coordinate(&Coordinate {
        class: "Pag",
        accepted: true,
        bayesian: true,
        validation: "full",
    });
}

/// Twelve rows of the front-door design (`t -> m -> y`, `t <-> y`) with one row carrying the only
/// `m = 0` under `t = 1` and the only `y = 0` under `m = 1`: dropping or duplicating it moves the
/// identified law a long way, which is the planted violation of a stable estimate.
fn leverage_table() -> TabularData {
    let t = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0];
    let m = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0];
    let y = [0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0];
    TabularData::from_f64_columns([("t", &t[..]), ("m", &m[..]), ("y", &y[..])]).unwrap()
}

/// `P(Y=1 | do(T=1))` by the front-door formula, computed by hand from the twelve rows.
fn leverage_truth() -> f64 {
    let rows: Vec<(f64, f64, f64)> = vec![
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 0.0, 0.0),
        (0.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 1.0),
        (1.0, 1.0, 0.0),
        (1.0, 0.0, 0.0),
    ];
    let n = rows.len() as f64;
    let share =
        |keep: &dyn Fn(&(f64, f64, f64)) -> bool| rows.iter().filter(|r| keep(r)).count() as f64;
    let mut total = 0.0;
    for m in [0.0, 1.0] {
        let p_m = share(&|r| r.0 == 1.0 && r.1 == m) / share(&|r| r.0 == 1.0);
        let mut inner = 0.0;
        for t in [0.0, 1.0] {
            let p_t = share(&|r| r.0 == t) / n;
            let cell = share(&|r| r.0 == t && r.1 == m);
            let p_y = if cell > 0.0 {
                share(&|r| r.0 == t && r.1 == m && r.2 == 1.0) / cell
            } else {
                0.0
            };
            inner += p_t * p_y;
        }
        total += p_m * inner;
    }
    total
}

/// A high-leverage table is either flagged by a refuter that can speak or, where a resample empties
/// a cell the functional reads, the refuter says in a typed diagnostic that it did not run; the
/// identified value itself is the hand-computed front-door law either way.
#[test]
fn a_high_leverage_table_is_flagged_or_the_refuter_says_it_did_not_run() {
    let truth = leverage_truth();
    let graph = {
        let mut graph = Admg::with_variables(3);
        graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
        graph.insert_directed(DenseNodeId::from_raw(1), DenseNodeId::from_raw(2)).unwrap();
        graph.insert_bidirected(DenseNodeId::from_raw(0), DenseNodeId::from_raw(2)).unwrap();
        graph
    };
    let data = leverage_table();
    let query = InterventionalDistributionQuery::new(
        VariableId::from_raw(2),
        [Intervention::set(VariableId::from_raw(0), Value::f64(1.0))],
    );
    let ctx = ExecutionContext::for_tests(2_131);
    for bayesian in [false, true] {
        let plain = study_of(&data, &query, bayesian, RefuteSuite::None)
            .graph(graph.clone())
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap_or_else(|error| panic!("unvalidated bayesian={bayesian}: {error}"));
        assert!(plain.refutations.is_empty());
        let got = probability_of(&plain, 1.0);
        let tolerance = if bayesian { 0.2 } else { 1e-9 };
        assert!((got - truth).abs() < tolerance, "bayesian={bayesian}: {got} vs {truth}");
        let validated = study_of(&data, &query, bayesian, RefuteSuite::Full)
            .graph(graph.clone())
            .build()
            .unwrap()
            .run(&ctx);
        match validated {
            // Every refuter of the suite refits on resamples that empty a cell the functional
            // reads: the run says so in a typed refusal naming the refuter, never a verdict.
            Err(error) => {
                let text = error.to_string();
                assert!(
                    text.contains("target.subset_stability")
                        && text
                            .contains("a perturbed table leaves a required probability cell empty"),
                    "bayesian={bayesian}: {text}"
                );
            }
            Ok(result) => {
                let flagged = result.refutations.iter().any(|r| !r.passed);
                let refused = result
                    .diagnostics
                    .iter()
                    .any(|d| d.code.as_ref() == "refute.functional.refuter_refused");
                assert!(flagged || refused, "bayesian={bayesian}: neither flagged nor refused");
            }
        }
    }
}
