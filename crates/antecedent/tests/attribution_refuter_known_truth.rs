//! Known-truth evidence for the refuter states of a static-DAG anomaly attribution.
//!
//! The tables follow `conformance/estimate/attribution_refuter_states/expected.json`: a clean
//! 24-row chain and the same chain with one planted high-leverage row, whose fit decides the
//! mechanism of the outcome. The subset-stability refuter of the whole attribution vector must
//! pass the clean table and detect the planted one, under a supplied and an accepted structure,
//! both inference modes and both suites. A masked table must also be attributed as the table of
//! its visible rows, which is what lets a subset refit differ from the full fit at all.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{
    AcceptedGraph, BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study, StudyResult,
};
use antecedent_core::{AnomalyAttributionQuery, CausalQuery, ExecutionContext, Lag, VariableId};
use antecedent_data::column::ValidityBitmap;
use antecedent_data::{TableView, TabularData, TimeSeriesData};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_graph::{TemporalDag, ensure_lagged};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/attribution_refuter_states/expected.json"
    ))
    .expect("attribution_refuter_states fixture parses")
}

/// The fixture's table: `planted` moves the declared row to the declared extreme.
fn table(truth: &Value, planted: bool) -> TabularData {
    let dgp = &truth["dgp"];
    let rows = usize::try_from(dgp["rows"].as_u64().unwrap()).unwrap();
    let row = usize::try_from(dgp["planted_row"].as_u64().unwrap()).unwrap();
    let mut x: Vec<f64> =
        (0..rows).map(|i| (1.37 * i as f64).sin() + 0.5 * (0.91 * i as f64).cos()).collect();
    if planted {
        x[row] = dgp["planted_x"].as_f64().unwrap();
    }
    let mut y: Vec<f64> =
        (0..rows).map(|i| 1.0 + 2.0 * x[i] + 0.3 * (2.17 * i as f64).sin()).collect();
    if planted {
        y[row] += dgp["lift"].as_f64().unwrap();
    }
    TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap()
}

fn dag() -> Dag {
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph
}

fn query() -> CausalQuery {
    CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new([VariableId::from_raw(1)], 100))
}

fn report<'a>(result: &'a StudyResult, refuter: &str) -> &'a antecedent_validate::RefutationReport {
    result
        .refutations
        .iter()
        .find(|report| report.refuter.as_ref() == refuter)
        .unwrap_or_else(|| panic!("no `{refuter}` in {:?}", result.refutations))
}

fn on_large_stack(body: fn()) {
    std::thread::Builder::new()
        .name("attribution-refuter-known-truth".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .expect("spawn evidence thread")
        .join()
        .expect("evidence body panicked");
}

#[test]
fn a_masked_attribution_is_the_attribution_of_the_visible_rows() {
    on_large_stack(masked_body);
}

fn masked_body() {
    let truth = fixture();
    let data = table(&truth, false);
    let ctx = ExecutionContext::for_tests(31);
    let prepared = Study::tabular(data.clone())
        .graph(dag())
        .query(query())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    // The visible rows are the even positions.
    let rows = data.row_count();
    let visible: Vec<usize> = (0..rows).filter(|row| row % 2 == 0).collect();
    let mut bytes = vec![0_u8; rows.div_ceil(8)];
    for &row in &visible {
        bytes[row / 8] |= 1 << (row % 8);
    }
    let masked = data.with_analysis_mask(ValidityBitmap::from_bytes(bytes, rows).unwrap()).unwrap();
    let column = |name: usize| -> Vec<f64> {
        let view = data.column(VariableId::from_raw(name as u32)).unwrap();
        let antecedent_data::ColumnView::Float64(values) = view else { panic!("float column") };
        visible.iter().map(|&row| values.values[row]).collect()
    };
    let (x, y) = (column(0), column(1));
    let gathered =
        TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap();

    let from_mask = prepared.estimate(&masked, &ctx).unwrap();
    let from_rows = prepared.estimate(&gathered, &ctx).unwrap();
    let (a, b) = (&from_mask.anomaly().unwrap()[0], &from_rows.anomaly().unwrap()[0]);
    assert_eq!(a.scores.len(), visible.len(), "only the visible rows are scored");
    assert_eq!(a.rows.as_ref(), visible.as_slice(), "rows are the submitted positions");
    for (got, want) in a.scores.iter().zip(b.scores.iter()) {
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }
    for (got, want) in a.noise_contributions.iter().zip(b.noise_contributions.iter()) {
        assert!((got - want).abs() < 1e-12, "{got} vs {want}");
    }
}

#[test]
fn anomaly_attribution_refuter_states_known_truth() {
    on_large_stack(states_body);
}

fn states_body() {
    let truth = fixture();
    let expected = &truth["expected_verdicts"];
    let clean_passes = expected["clean_subset_stability_passes"].as_bool().unwrap();
    let planted_passes = expected["planted_subset_stability_passes"].as_bool().unwrap();
    assert!(clean_passes && !planted_passes);
    for accepted in [false, true] {
        for inference in [
            InferenceMode::Frequentist,
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(24)),
        ] {
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                let full = matches!(suite, RefuteSuite::Full);
                let run = |planted: bool| {
                    let builder = Study::tabular(table(&truth, planted));
                    let builder = if accepted {
                        builder.graph(AcceptedGraph::from(dag()))
                    } else {
                        builder.graph(dag())
                    };
                    let study = builder
                        .query(query())
                        .inference(inference.clone())
                        .refute(suite)
                        .build()
                        .unwrap();
                    assert_eq!(study.support_status(), Some(CellStatus::Licensed));
                    study.run(&ExecutionContext::for_tests(31)).unwrap()
                };
                let clean = run(false);
                let planted = run(true);
                assert_eq!(
                    report(&clean, "target.subset_stability").passed,
                    clean_passes,
                    "clean table, accepted={accepted}, full={full}"
                );
                assert_eq!(
                    report(&planted, "target.subset_stability").passed,
                    planted_passes,
                    "planted table, accepted={accepted}, full={full}"
                );
                // The planted row is also the top anomaly of the full-sample fit's scores.
                assert!(clean.refutations.iter().all(|r| r.refuter.as_ref() != "target.nope"));
                if full {
                    assert!(report(&clean, "target.bootstrap_stability").passed);
                } else {
                    assert!(
                        clean
                            .refutations
                            .iter()
                            .all(|r| r.refuter.as_ref() != "target.bootstrap_stability")
                    );
                }
            }
        }
    }
}

/// The fixture's series: `planted` moves one period of `x` to the declared extreme and lifts the
/// next period's outcome.
fn series(truth: &Value, planted: bool) -> TimeSeriesData {
    let d = &truth["series"];
    let rows = usize::try_from(d["rows"].as_u64().unwrap()).unwrap();
    let at = usize::try_from(d["planted_position"].as_u64().unwrap()).unwrap();
    let mut x: Vec<f64> =
        (0..rows).map(|t| (0.37 * t as f64).sin() + 0.5 * (1.1 * t as f64).cos()).collect();
    if planted {
        x[at] = d["planted_x"].as_f64().unwrap();
    }
    let y: Vec<f64> = (0..rows)
        .map(|t| {
            let lag = if t > 0 { x[t - 1] } else { 0.0 };
            let base = 1.0 + 2.0 * lag + 0.3 * (2.17 * t as f64).sin();
            if planted && t == at + 1 { base + d["lift"].as_f64().unwrap() } else { base }
        })
        .collect();
    TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

fn temporal_dag() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let parent = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let child = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(parent, child).unwrap();
    graph
}

#[test]
fn temporal_anomaly_attribution_refuter_states_known_truth() {
    on_large_stack(temporal_states_body);
}

fn temporal_states_body() {
    let truth = fixture();
    let expected = &truth["expected_verdicts"];
    let clean_passes = expected["clean_subset_stability_passes"].as_bool().unwrap();
    let planted_passes = expected["planted_subset_stability_passes"].as_bool().unwrap();
    for accepted in [false, true] {
        for inference in [
            InferenceMode::Frequentist,
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(8)),
        ] {
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                let full = matches!(suite, RefuteSuite::Full);
                let run = |planted: bool| {
                    let builder = Study::series(series(&truth, planted));
                    let builder = if accepted {
                        builder.graph(AcceptedGraph::from(temporal_dag()))
                    } else {
                        builder.graph(temporal_dag())
                    };
                    let study = builder
                        .query(query())
                        .inference(inference.clone())
                        .refute(suite)
                        .build()
                        .unwrap();
                    assert_eq!(study.support_status(), Some(CellStatus::Licensed));
                    study.run(&ExecutionContext::for_tests(31)).unwrap()
                };
                let clean = run(false);
                let planted = run(true);
                // The clean series passes: a circular window includes every period equally
                // often, so the replicate mean is the original.
                assert_eq!(
                    report(&clean, "target.subset_stability").passed,
                    clean_passes,
                    "clean series, accepted={accepted}, full={full}"
                );
                assert_eq!(
                    report(&planted, "target.subset_stability").passed,
                    planted_passes,
                    "planted series, accepted={accepted}, full={full}"
                );
                if full {
                    // The block resample is gathered, so it moves the vector at all.
                    let bootstrap = report(&clean, "target.bootstrap_stability");
                    assert!(bootstrap.passed);
                    assert!(
                        (bootstrap.refuted_ate - bootstrap.original_ate).abs() > 1e-6,
                        "the block bootstrap must resample the series"
                    );
                }
            }
        }
    }
}

#[test]
fn a_change_attribution_validation_is_refused_for_want_of_a_detectable_violation() {
    use antecedent_core::{ChangeAttributionQuery, PopulationSelector};
    on_large_stack(|| {
        let truth = fixture();
        let query = CausalQuery::ChangeAttribution(ChangeAttributionQuery::new(
            VariableId::from_raw(1),
            PopulationSelector::TimeRange { start: 0, end: 12 },
            PopulationSelector::TimeRange { start: 12, end: 24 },
        ));
        for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
            let built = Study::tabular(table(&truth, false))
                .graph(dag())
                .query(query.clone())
                .refute(suite)
                .build();
            let message = match built {
                Err(error) => error.to_string(),
                Ok(study) => {
                    assert_eq!(study.support_status(), Some(CellStatus::Refused));
                    study.run(&ExecutionContext::for_tests(31)).unwrap_err().to_string()
                }
            };
            assert!(
                message.contains("Cheap and full validation of a change attribution is refused"),
                "{message}"
            );
        }
    });
}
