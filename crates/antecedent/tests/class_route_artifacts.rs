//! A route that executes on a graph class publishes an artifact that an independent consumer
//! reads back: the contracted result of each class coordinate is encoded from the prepared
//! handle, decoded by the consumer, and the consumer's view of the query and the estimate is
//! the one the route published.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{BayesianConfig, InferenceMode, IntoGraphInput, RefuteSuite, Study};
use antecedent_core::{
    AnomalyAttributionQuery, CausalQuery, ChangeAttributionQuery, ExecutionContext, Lag,
    PopulationSelector, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_graph::{Admg, Cpdag, DenseNodeId, Pag, TemporalAdmg, TemporalCpdag, TemporalPag};
use antecedent_io::consume_analysis_result;

/// Debug builds of the shared dispatcher carry frames close to the default test-thread stack.
fn on_large_stack(name: &str, body: fn()) {
    std::thread::Builder::new()
        .name(name.into())
        .stack_size(16 * 1024 * 1024)
        .spawn(body)
        .expect("spawn evidence thread")
        .join()
        .expect("evidence body panicked");
}

fn node(index: u32) -> DenseNodeId {
    DenseNodeId::from_raw(index)
}

fn variable(index: u32) -> VariableId {
    VariableId::from_raw(index)
}

fn sines(rows: usize) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let sequence = |frequency: f64, phase: f64| -> Vec<f64> {
        (0..rows).map(|i| (frequency * i as f64 + phase).sin()).collect()
    };
    (sequence(0.37, 0.2), sequence(1.13, 0.7), sequence(2.9, 1.3))
}

fn columns(named: &[(&str, &Vec<f64>)]) -> TabularData {
    TabularData::from_f64_columns(named.iter().map(|(name, values)| (*name, values.as_slice())))
        .unwrap()
}

/// Prepare `study`, run it on its retained data, encode the contracted result and read it back.
/// The consumer must decode the artifact and name the query the route answered.
fn round_trip(label: &str, study: &Study) {
    let ctx = ExecutionContext::for_tests(13);
    let prepared = study.prepare(&ctx).unwrap_or_else(|e| panic!("{label}: prepare: {e}"));
    let result = prepared.estimate_retained(&ctx).unwrap_or_else(|e| panic!("{label}: run: {e}"));
    let bytes = prepared
        .encode_contracted_result(&result, &format!("class-route-{label}"), &ctx)
        .unwrap_or_else(|e| panic!("{label}: encode: {e}"));
    let again =
        prepared.encode_contracted_result(&result, &format!("class-route-{label}"), &ctx).unwrap();
    assert_eq!(bytes, again, "{label}: the encoding is deterministic");
    let consumed = consume_analysis_result(&bytes).unwrap_or_else(|e| panic!("{label}: {e}"));
    // A scalar the route did not publish (an attribution has none) is absent from the artifact.
    match result.estimate().as_effect().map(|effect| effect.ate).filter(|ate| ate.is_finite()) {
        Some(ate) => assert_eq!(consumed.body.estimate, Some(ate), "{label}: artifact estimate"),
        None => assert_eq!(consumed.body.estimate, None, "{label}: no scalar published"),
    }
    let mut tampered = bytes;
    let middle = tampered.len() / 2;
    tampered[middle] ^= 0x40;
    assert!(consume_analysis_result(&tampered).is_err(), "{label}: a tampered artifact is refused");
}

fn inferences() -> [(&'static str, InferenceMode); 2] {
    [
        ("frequentist", InferenceMode::Frequentist),
        ("bayesian", InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(4))),
    ]
}

// ------------------------------------------------------------------------- attribution
fn chain_table() -> TabularData {
    let rows = 60;
    let (s1, _, s3) = sines(rows);
    let x: Vec<f64> = s1.iter().map(|v| 2.0 * v).collect();
    let mut y: Vec<f64> = (0..rows).map(|i| 2.0 * x[i] + 0.4 * s3[i]).collect();
    y[rows - 1] += 25.0;
    columns(&[("x", &x), ("y", &y)])
}

fn shifted_chain_table() -> TabularData {
    let rows = 80;
    let (s1, _, s3) = sines(rows);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for population in 0..2 {
        for i in 0..rows {
            let x_i = 2.0 * s1[i];
            x.push(x_i);
            y.push(2.0 * x_i + 0.4 * s3[i] + if population == 1 { 5.0 } else { 0.0 });
        }
    }
    columns(&[("x", &x), ("y", &y)])
}

fn collider_table() -> TabularData {
    let rows = 80;
    let (s1, s2, s3) = sines(rows);
    let a: Vec<f64> = s1.clone();
    let mut b: Vec<f64> = (0..rows).map(|i| -0.6 * s1[i] + 0.8 * s2[i]).collect();
    b[rows - 1] += 6.0;
    let t: Vec<f64> = (0..rows).map(|i| a[i] + 2.0 * b[i] + 0.3 * s3[i]).collect();
    columns(&[("a", &a), ("b", &b), ("t", &t)])
}

fn latent_collider() -> Admg {
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(node(0), node(2)).unwrap();
    graph.insert_directed(node(1), node(2)).unwrap();
    graph.insert_bidirected(node(0), node(1)).unwrap();
    graph
}

fn lagged_series() -> TimeSeriesData {
    let rows = 80;
    let (s1, s2, s3) = sines(rows);
    let x: Vec<f64> = s1.iter().map(|v| 2.0 * v).collect();
    let mut y: Vec<f64> =
        (0..rows).map(|t| 0.8 * if t == 0 { x[0] } else { x[t - 1] } + 0.3 * s2[t]).collect();
    y[rows - 1] += 8.0;
    TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("y", y.as_slice()), ("z", s3.as_slice())],
        1,
    )
    .unwrap()
}

fn anomaly(target: u32) -> CausalQuery {
    CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new([variable(target)], 1000))
}

fn change(outcome: u32, rows: usize) -> CausalQuery {
    CausalQuery::ChangeAttribution(ChangeAttributionQuery::new(
        variable(outcome),
        PopulationSelector::TimeRange { start: 0, end: rows },
        PopulationSelector::TimeRange { start: rows, end: 2 * rows },
    ))
}

fn tabular(
    data: &TabularData,
    graph: impl IntoGraphInput,
    query: CausalQuery,
    inference: &InferenceMode,
) -> Study {
    Study::tabular(data.clone())
        .graph(graph)
        .query(query)
        .inference(inference.clone())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

fn series(
    data: &TimeSeriesData,
    graph: impl IntoGraphInput,
    query: CausalQuery,
    inference: &InferenceMode,
) -> Study {
    Study::series(data.clone())
        .graph(graph)
        .query(query)
        .inference(inference.clone())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

#[test]
fn static_class_attributions_publish_an_artifact_a_consumer_reads() {
    on_large_stack("static-class-attribution", static_attribution_body);
}

fn static_attribution_body() {
    let mut cpdag = Cpdag::with_variables(2);
    cpdag.insert_undirected(node(0), node(1)).unwrap();
    let mut pag = Pag::with_variables(2);
    pag.insert_circle_arrow(node(0), node(1)).unwrap();
    for (mode, inference) in inferences() {
        round_trip(
            &format!("cpdag-anomaly-{mode}"),
            &tabular(&chain_table(), cpdag.clone(), anomaly(1), &inference),
        );
        round_trip(
            &format!("cpdag-change-{mode}"),
            &tabular(&shifted_chain_table(), cpdag.clone(), change(1, 80), &inference),
        );
        round_trip(
            &format!("pag-anomaly-{mode}"),
            &tabular(&chain_table(), pag.clone(), anomaly(1), &inference),
        );
        round_trip(
            &format!("pag-change-{mode}"),
            &tabular(&shifted_chain_table(), pag.clone(), change(1, 80), &inference),
        );
        round_trip(
            &format!("admg-anomaly-{mode}"),
            &tabular(&collider_table(), latent_collider(), anomaly(2), &inference),
        );
        round_trip(
            &format!("admg-change-{mode}"),
            &tabular(&collider_table(), latent_collider(), change(2, 40), &inference),
        );
    }
}

#[test]
fn temporal_class_attributions_publish_an_artifact_a_consumer_reads() {
    on_large_stack("temporal-class-attribution", temporal_attribution_body);
}

fn temporal_attribution_body() {
    let data = lagged_series();
    let mut cpdag = TemporalCpdag::empty();
    let parent = cpdag.add_lagged(variable(0), Lag::from_raw(1)).unwrap();
    let child = cpdag.add_lagged(variable(1), Lag::CONTEMPORANEOUS).unwrap();
    cpdag.insert_directed(parent, child).unwrap();
    let mut admg = TemporalAdmg::empty();
    let lagged_x = admg.add_lagged(variable(0), Lag::from_raw(1)).unwrap();
    let y = admg.add_lagged(variable(1), Lag::CONTEMPORANEOUS).unwrap();
    let x = admg.add_lagged(variable(0), Lag::CONTEMPORANEOUS).unwrap();
    admg.insert_directed(lagged_x, y).unwrap();
    admg.insert_bidirected(x, y).unwrap();
    let mut pag = TemporalPag::empty();
    let parent = pag.add_lagged(variable(0), Lag::from_raw(1)).unwrap();
    let child = pag.add_lagged(variable(1), Lag::CONTEMPORANEOUS).unwrap();
    pag.insert_circle_arrow(parent, child).unwrap();
    for (mode, inference) in inferences() {
        for (class, query) in [("anomaly", anomaly(1)), ("change", change(1, 40))] {
            round_trip(
                &format!("temporal-cpdag-{class}-{mode}"),
                &series(&data, cpdag.clone(), query.clone(), &inference),
            );
            round_trip(
                &format!("temporal-admg-{class}-{mode}"),
                &series(&data, admg.clone(), query.clone(), &inference),
            );
            round_trip(
                &format!("temporal-pag-{class}-{mode}"),
                &series(&data, pag.clone(), query, &inference),
            );
        }
    }
}
