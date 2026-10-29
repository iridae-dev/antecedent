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
fn round_trip(label: &str, study: &Study) -> antecedent::StudyResult {
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
    let error = consume_analysis_result(&tampered).expect_err("a tampered artifact is refused");
    assert!(
        matches!(&error, antecedent_io::IoError::ChecksumMismatch { section }
            if section == "analysis_result.body"),
        "{label}: a flipped body byte is a body checksum mismatch, not another failure: {error:?}"
    );
    result
}

fn close(got: f64, want: f64, label: &str) {
    near(got, want, 1e-8, label);
}

fn near(got: f64, want: f64, tolerance: f64, label: &str) {
    assert!((got - want).abs() < tolerance, "{label}: {got} vs {want}");
}

/// Last-row anomaly truth, derived outside the product (`conformance/estimate/
/// class_attribution_truth/reference.py` functions on these tables): the score is
/// -ln erfc(|y - median| / (1.4826 MAD sqrt 2)) of the planted row, the baseline is the score of the
/// target rebuilt with every noise set to 0, and the Shapley split of score - baseline is
/// over the members' players, merged with equal weights.
fn check_anomaly_truth(
    label: &str,
    result: &antecedent::StudyResult,
    target_slot: usize,
    score: f64,
    baseline: f64,
    by_component: &[(u64, f64)],
) {
    let scores = &result.anomaly().expect("anomaly scores")[target_slot];
    let last = scores.rows.len() - 1;
    close(scores.scores[last], score, &format!("{label}: planted-row score"));
    close(scores.baseline_scores[last], baseline, &format!("{label}: planted-row baseline"));
    let width = scores.noise_components.len();
    assert_eq!(width, by_component.len(), "{label}: players");
    let mut total = 0.0;
    for (id, want) in by_component {
        let column = scores
            .noise_components
            .iter()
            .position(|c| *c == antecedent_core::ComponentId::from_variable(variable(*id as u32)))
            .unwrap_or_else(|| panic!("{label}: component {id}"));
        let got = scores.noise_contributions[last * width + column];
        close(got, *want, &format!("{label}: component {id}"));
        total += got;
    }
    close(total, score - baseline, &format!("{label}: efficiency"));
}

fn check_change_truth(
    label: &str,
    result: &antecedent::StudyResult,
    total: f64,
    by_component: &[(u64, f64)],
    tolerance: f64,
) {
    let change = result.change_attribution().expect("change attribution");
    near(change.total_change, total, tolerance, &format!("{label}: total change"));
    for (id, want) in by_component {
        let got = change
            .contributions
            .iter()
            .find(|c| c.component == antecedent_core::ComponentId::from_variable(variable(*id as u32)))
            .unwrap_or_else(|| panic!("{label}: component {id}"))
            .contribution;
        near(got, *want, tolerance, &format!("{label}: component {id}"));
    }
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
        let result = round_trip(
            &format!("cpdag-anomaly-{mode}"),
            &tabular(&chain_table(), cpdag.clone(), anomaly(1), &inference),
        );
        if mode == "frequentist" {
            // y = 2x + 0.4 sin(2.9 i + 1.3) with +25 planted on the last row: the planted noise
            // is y's own, so y's player carries the score and x's is the small abduction term.
            check_anomaly_truth(
                "cpdag-anomaly",
                &result,
                0,
                22.926_762_990_920_327,
                0.063_641_067_229_894_39,
                &[(0, -0.194_202_758_203_880_3), (1, 23.057_324_681_894_315)],
            );
        }
        let result = round_trip(
            &format!("cpdag-change-{mode}"),
            &tabular(&shifted_chain_table(), cpdag.clone(), change(1, 80), &inference),
        );
        if mode == "frequentist" {
            // The second population is the first with y raised by exactly 5: the whole shift is
            // y's own mechanism and x's mechanism is unchanged.
            check_change_truth("cpdag-change", &result, 5.0, &[(0, 0.0), (1, 5.0)], 1e-8);
        }
        round_trip(
            &format!("pag-anomaly-{mode}"),
            &tabular(&chain_table(), pag.clone(), anomaly(1), &inference),
        );
        round_trip(
            &format!("pag-change-{mode}"),
            &tabular(&shifted_chain_table(), pag.clone(), change(1, 80), &inference),
        );
        let result = round_trip(
            &format!("admg-anomaly-{mode}"),
            &tabular(&collider_table(), latent_collider(), anomaly(2), &inference),
        );
        if mode == "frequentist" {
            // The planted +6 is on b, inside the {a, b} district block, which carries the score.
            check_anomaly_truth(
                "admg-anomaly",
                &result,
                0,
                30.691_544_694_396_732,
                0.053_115_005_623_028_524,
                &[(0, 30.902_994_694_954_167), (2, -0.264_565_006_198_511_4)],
            );
        }
        let result = round_trip(
            &format!("admg-change-{mode}"),
            &tabular(&collider_table(), latent_collider(), change(2, 40), &inference),
        );
        if mode == "frequentist" {
            // The mixed-graph change resamples noise rows (seeded): totals deviate from the
            // closed-form mean difference by 0.047, so the tolerance is 0.06, not exact.
            check_change_truth(
                "admg-change",
                &result,
                0.324_326_898_370_536_63,
                &[(0, 0.336_931_370_249_606_2), (2, -0.012_604_471_879_069_563)],
                0.06,
            );
        }
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
