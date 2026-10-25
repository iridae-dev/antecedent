//! Anomaly and change attribution over a CPDAG, a PAG, a mixed graph and their temporal forms:
//! each member of the class attributed as the DAG route does, the weighted mean with the
//! identified set when they disagree, the noise-block route against known truth, and the typed
//! refusal of a hedge.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::RetainedOperation;
use antecedent::{
    BayesianConfig, CausalError, CellStatus, InferenceMode, IntoGraphInput, RefuteSuite, Study,
};
use antecedent_attribution::{AnomalyScores, HEDGE_WITNESS_ID};
use antecedent_core::{
    AnomalyAttributionQuery, CausalQuery, ChangeAttributionQuery, ComponentId, ExecutionContext,
    Lag, PopulationSelector, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_graph::{
    Admg, Cpdag, Dag, DenseNodeId, Pag, TemporalAdmg, TemporalCpdag, TemporalDag,
};
use antecedent_identify::IdentificationStatus;

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

/// Deterministic, nearly uncorrelated unit-scale sequences.
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

/// `x` and `y = 2x + noise`, with the last row's `y` far out.
fn chain_table(outlier: f64) -> TabularData {
    let rows = 60;
    let (s1, _, s3) = sines(rows);
    let x: Vec<f64> = s1.iter().map(|v| 2.0 * v).collect();
    let mut y: Vec<f64> = (0..rows).map(|i| 2.0 * x[i] + 0.4 * s3[i]).collect();
    y[rows - 1] += outlier;
    columns(&[("x", &x), ("y", &y)])
}

/// Two populations of the chain, the second with its mechanism of `y` shifted by `shift`.
fn shifted_chain_table(shift: f64) -> TabularData {
    let rows = 80;
    let (s1, _, s3) = sines(rows);
    let mut x = Vec::new();
    let mut y = Vec::new();
    for population in 0..2 {
        for i in 0..rows {
            let x_i = 2.0 * s1[i];
            x.push(x_i);
            y.push(2.0 * x_i + 0.4 * s3[i] + if population == 1 { shift } else { 0.0 });
        }
    }
    columns(&[("x", &x), ("y", &y)])
}

fn anomaly_query(target: u32) -> CausalQuery {
    CausalQuery::AnomalyAttribution(AnomalyAttributionQuery::new([variable(target)], 1000))
}

fn change_query(outcome: u32, rows: usize) -> CausalQuery {
    CausalQuery::ChangeAttribution(ChangeAttributionQuery::new(
        variable(outcome),
        PopulationSelector::TimeRange { start: 0, end: rows },
        PopulationSelector::TimeRange { start: rows, end: 2 * rows },
    ))
}

fn tabular_study(
    data: &TabularData,
    graph: impl IntoGraphInput,
    query: &CausalQuery,
    inference: &InferenceMode,
) -> Study {
    Study::tabular(data.clone())
        .graph(graph)
        .query(query.clone())
        .inference(inference.clone())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

fn dag_of(n: u32, edges: &[(u32, u32)]) -> Dag {
    let mut graph = Dag::with_variables(n);
    for &(from, to) in edges {
        graph.insert_directed(node(from), node(to)).unwrap();
    }
    graph
}

/// The contribution of `component` to row `row`, zero for a component that is not a player.
fn contribution(scores: &AnomalyScores, row: usize, component: u32) -> f64 {
    let held = scores.noise_components.len();
    scores
        .noise_components
        .iter()
        .position(|c| *c == ComponentId::from_variable(variable(component)))
        .map_or(0.0, |column| scores.noise_contributions[row * held + column])
}

fn mean_over_rows(scores: &AnomalyScores, component: u32) -> f64 {
    let rows = scores.rows.len();
    (0..rows).map(|row| contribution(scores, row, component)).sum::<f64>() / rows as f64
}

fn frequentist() -> InferenceMode {
    InferenceMode::Frequentist
}

#[test]
fn a_cpdag_anomaly_is_the_mean_of_its_dag_completions() {
    on_large_stack("cpdag-anomaly", cpdag_anomaly_body);
}

fn cpdag_anomaly_body() {
    let ctx = ExecutionContext::for_tests(13);
    let data = chain_table(25.0);
    let query = anomaly_query(1);
    let mut cpdag = Cpdag::with_variables(2);
    cpdag.insert_undirected(node(0), node(1)).unwrap();
    let study = tabular_study(&data, cpdag, &query, &frequentist());
    // No evidence row or coverage record names the cell: it executes and stays unlicensed.
    assert_eq!(study.support_status(), Some(CellStatus::Refused));
    let result = study.run(&ctx).unwrap();
    assert_eq!(result.support_status, Some(CellStatus::Refused));
    let merged = &result.anomaly().expect("anomaly payload")[0];

    // The two DAGs of the Markov class, attributed by the licensed DAG route.
    let forward =
        tabular_study(&data, dag_of(2, &[(0, 1)]), &query, &frequentist()).run(&ctx).unwrap();
    let backward =
        tabular_study(&data, dag_of(2, &[(1, 0)]), &query, &frequentist()).run(&ctx).unwrap();
    let (forward, backward) = (&forward.anomaly().unwrap()[0], &backward.anomaly().unwrap()[0]);
    // `y` is a root in the backward DAG, so `x` is not one of its players there.
    assert_eq!(backward.noise_components.len(), 1);
    assert_eq!(forward.noise_components.len(), 2);
    assert_eq!(merged.noise_components.len(), 2, "the union of the players of the members");
    assert_eq!(merged.scores, forward.scores, "the factual scores do not depend on the graph");
    for row in 0..merged.rows.len() {
        for component in [0, 1] {
            let expected = 0.5
                * (contribution(forward, row, component) + contribution(backward, row, component));
            assert!(
                (contribution(merged, row, component) - expected).abs() < 1e-9,
                "row {row} component {component}"
            );
        }
        let baseline = 0.5 * (forward.baseline_scores[row] + backward.baseline_scores[row]);
        assert!((merged.baseline_scores[row] - baseline).abs() < 1e-9);
    }

    // The members disagree on `x`: the identified set of its contribution runs from the
    // backward DAG (zero) to the forward DAG, and the published value is the mean.
    let mixture = result.structural_response.as_ref().expect("the class mixture");
    assert_eq!(mixture.atoms.len(), 2);
    assert!((mixture.identified_mass - 1.0).abs() < 1e-12);
    let set = mixture.identified_set.as_ref().expect("identified set");
    let column = |component: u32| {
        set.grid.iter().position(|g| (*g - f64::from(component)).abs() < 1e-12).unwrap()
    };
    let upper_x = mean_over_rows(forward, 0);
    // The backward DAG has no player `x` (zero), the forward DAG its mean contribution.
    assert!((set.lower[column(0)] - upper_x.min(0.0)).abs() < 1e-9);
    assert!((set.upper[column(0)] - upper_x.max(0.0)).abs() < 1e-9);
    assert!(upper_x.abs() > 1e-6, "the members must disagree for the set to be a set");
    assert!((mean_over_rows(merged, 0) - 0.5 * upper_x).abs() < 1e-9);
    assert!(result.diagnostics.iter().any(|d| d.code.as_ref() == "attribution.class.envelope"));
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "attribution.class.identified_set")
    );
}

#[test]
fn a_cpdag_change_that_every_member_agrees_on_keeps_one_value() {
    on_large_stack("cpdag-change", cpdag_change_body);
}

fn cpdag_change_body() {
    let ctx = ExecutionContext::for_tests(13);
    let data = shifted_chain_table(5.0);
    let query = change_query(1, 80);
    let mut cpdag = Cpdag::with_variables(2);
    cpdag.insert_undirected(node(0), node(1)).unwrap();
    let result = tabular_study(&data, cpdag, &query, &frequentist()).run(&ctx).unwrap();
    let change = result.change_attribution().expect("change payload");
    // The mechanism of `y` shifts by 5 and `x` does not move: the outcome moves by 5 in
    // both DAGs of the class.
    assert!((change.total_change - 5.0).abs() < 0.5, "total {}", change.total_change);
    let share = |variable_id: u32| {
        change
            .contributions
            .iter()
            .find(|c| c.component == ComponentId::from_variable(variable(variable_id)))
            .map_or(0.0, |c| c.contribution)
    };
    assert!(share(0).abs() < 1e-9, "x does not change: {}", share(0));
    assert!((share(1) - change.total_change).abs() < 1e-6, "y carries the whole change");
    // The members' contributions agree up to the Monte Carlo noise of each, and the graph
    // sensitivity is that spread.
    let spread = change.graph_sensitivity.as_ref().expect("graph sensitivity");
    assert_eq!(spread.len(), change.contributions.len());
    assert!(spread.iter().all(|s| *s < 0.5), "{spread:?}");
    let mixture = result.structural_response.as_ref().unwrap();
    let set = mixture.identified_set.as_ref().unwrap();
    for (lower, upper) in set.lower.iter().zip(set.upper.iter()) {
        assert!(upper - lower < 1.0, "[{lower}, {upper}]");
    }
}

#[test]
fn a_bayesian_cpdag_publishes_a_shared_posterior_and_never_a_mixture() {
    on_large_stack("cpdag-bayesian", cpdag_bayesian_body);
}

fn cpdag_bayesian_body() {
    let ctx = ExecutionContext::for_tests(13);
    let inference = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(4));
    let mut cpdag = Cpdag::with_variables(2);
    cpdag.insert_undirected(node(0), node(1)).unwrap();

    // The mean anomaly score does not depend on the structure: both DAGs of the class publish
    // the same draws, and the class publishes them.
    let data = chain_table(25.0);
    let query = anomaly_query(1);
    let class = tabular_study(&data, cpdag.clone(), &query, &inference).run(&ctx).unwrap();
    let forward = tabular_study(&data, dag_of(2, &[(0, 1)]), &query, &inference).run(&ctx).unwrap();
    let backward =
        tabular_study(&data, dag_of(2, &[(1, 0)]), &query, &inference).run(&ctx).unwrap();
    let draws = |result: &antecedent::StudyResult| result.posterior.as_ref().unwrap().draws.clone();
    assert_eq!(draws(&forward), draws(&backward));
    assert_eq!(draws(&class), draws(&forward));
    assert_eq!(class.support_status, Some(CellStatus::Refused));

    // A change attribution differs between the DAGs of the class: the class publishes no
    // mixture of the two posteriors, keeps each on its atom, and states the envelope of their
    // credible intervals.
    let shifted = shifted_chain_table(5.0);
    let change = change_query(1, 80);
    let class = tabular_study(&shifted, cpdag, &change, &inference).run(&ctx).unwrap();
    assert!(class.posterior.is_none(), "enumeration weights are not posterior mass");
    let mixture = class.structural_response.as_ref().unwrap();
    assert_eq!(mixture.atoms.len(), 2);
    assert!(mixture.atoms.iter().all(|atom| atom.posterior.is_some()));
    assert!(
        class.diagnostics.iter().any(|d| d.code.as_ref() == "attribution.class.posterior_envelope")
    );
    assert!((class.change_attribution().unwrap().total_change - 5.0).abs() < 0.5);
}

/// `a`, `b` (correlated through a shared sine) and `t = a + 2b + noise` over a collider.
fn collider_table(rows: usize, outlier_b: f64) -> TabularData {
    let (s1, s2, s3) = sines(rows);
    let a: Vec<f64> = s1.clone();
    let mut b: Vec<f64> = (0..rows).map(|i| -0.6 * s1[i] + 0.8 * s2[i]).collect();
    b[rows - 1] += outlier_b;
    let t: Vec<f64> = (0..rows).map(|i| a[i] + 2.0 * b[i] + 0.3 * s3[i]).collect();
    columns(&[("a", &a), ("b", &b), ("t", &t)])
}

/// The collider `a -> t <- b` with the latent common cause `a <-> b`.
fn latent_collider() -> Admg {
    let mut graph = Admg::with_variables(3);
    graph.insert_directed(node(0), node(2)).unwrap();
    graph.insert_directed(node(1), node(2)).unwrap();
    graph.insert_bidirected(node(0), node(1)).unwrap();
    graph
}

#[test]
fn an_admg_anomaly_is_attributed_to_the_noise_blocks() {
    on_large_stack("admg-anomaly", admg_anomaly_body);
}

fn admg_anomaly_body() {
    let ctx = ExecutionContext::for_tests(13);
    let data = collider_table(80, 6.0);
    let query = anomaly_query(2);
    let study = tabular_study(&data, latent_collider(), &query, &frequentist());
    assert_eq!(study.support_status(), Some(CellStatus::Refused));
    let result = study.run(&ctx).unwrap();
    assert_eq!(
        result.identification.status,
        IdentificationStatus::IdentifiedUnderParametricRestrictions
    );
    let scores = &result.anomaly().unwrap()[0];
    // Two players: the block {a, b}, named by its smallest member, and the target.
    assert_eq!(
        scores.noise_components.as_ref(),
        &[ComponentId::from_variable(variable(0)), ComponentId::from_variable(variable(2))]
    );
    for row in 0..scores.rows.len() {
        let total: f64 = scores.noise_contributions[row * 2..row * 2 + 2].iter().sum();
        assert!((total - (scores.scores[row] - scores.baseline_scores[row])).abs() < 1e-9);
    }
    assert!(result.diagnostics.iter().any(|d| d.code.as_ref() == "attribution.noise_blocks"));
    // The same factual scores as the DAG route on the same table.
    let plain = tabular_study(&data, dag_of(3, &[(0, 2), (1, 2)]), &query, &frequentist())
        .run(&ctx)
        .unwrap();
    assert_eq!(plain.anomaly().unwrap()[0].scores, scores.scores);
    assert_eq!(plain.anomaly().unwrap()[0].noise_components.len(), 3);

    // Bayesian: the posterior of the mean anomaly score is that of the DAG route.
    let inference = InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(4));
    let bayesian = tabular_study(&data, latent_collider(), &query, &inference).run(&ctx).unwrap();
    let reference =
        tabular_study(&data, dag_of(3, &[(0, 2), (1, 2)]), &query, &inference).run(&ctx).unwrap();
    let got = &bayesian.posterior.as_ref().unwrap().summaries.mean;
    let want = &reference.posterior.as_ref().unwrap().summaries.mean;
    assert_eq!(got.len(), want.len());
    for (a, b) in got.iter().zip(want.iter()) {
        assert!((a - b).abs() < 1e-9, "{a} vs {b}");
    }
}

/// Two populations of the collider, shared noise, the second with `a` and `b` shifted by one and
/// the mechanism of `t` shifted by one half.
fn shifted_collider_table(rows: usize) -> TabularData {
    let (s1, s2, s3) = sines(rows);
    let (mut a, mut b, mut t) = (Vec::new(), Vec::new(), Vec::new());
    for population in 0..2 {
        let (offset, constant) = if population == 0 { (0.0, 0.0) } else { (1.0, 0.5) };
        for i in 0..rows {
            let a_i = s1[i] + offset;
            let b_i = -0.6 * s1[i] + 0.8 * s2[i] + offset;
            a.push(a_i);
            b.push(b_i);
            t.push(a_i + 2.0 * b_i + constant + 0.2 * s3[i]);
        }
    }
    columns(&[("a", &a), ("b", &b), ("t", &t)])
}

#[test]
fn an_admg_change_matches_the_known_shift_of_each_block() {
    on_large_stack("admg-change", admg_change_body);
}

fn admg_change_body() {
    let ctx = ExecutionContext::for_tests(13);
    let rows = 300;
    let data = shifted_collider_table(rows);
    let query = change_query(2, rows);
    let result = tabular_study(&data, latent_collider(), &query, &frequentist()).run(&ctx).unwrap();
    let change = result.change_attribution().unwrap();
    // The block {a, b} shifts t by 1 + 2 = 3 and the mechanism of t by 0.5.
    assert!((change.total_change - 3.5).abs() < 0.3, "total {}", change.total_change);
    let share = |component: u32| {
        change
            .contributions
            .iter()
            .find(|c| c.component == ComponentId::from_variable(variable(component)))
            .map(|c| c.contribution)
            .unwrap()
    };
    assert_eq!(change.contributions.len(), 2);
    assert!((share(0) - 3.0).abs() < 0.3, "block {}", share(0));
    assert!((share(2) - 0.5).abs() < 0.3, "target {}", share(2));
    assert!((change.contribution_sum() - change.total_change).abs() < 1e-9);
}

#[test]
fn a_hedged_mechanism_is_refused_with_the_witness() {
    on_large_stack("admg-hedge", admg_hedge_body);
}

fn admg_hedge_body() {
    let ctx = ExecutionContext::for_tests(13);
    let data = chain_table(10.0);
    // x -> y with x <-> y: the structural function of y is confounded with its noise.
    let mut graph = Admg::with_variables(2);
    graph.insert_directed(node(0), node(1)).unwrap();
    graph.insert_bidirected(node(0), node(1)).unwrap();
    for query in [anomaly_query(1), change_query(1, 30)] {
        let error =
            tabular_study(&data, graph.clone(), &query, &frequentist()).run(&ctx).unwrap_err();
        assert!(matches!(error, CausalError::NotIdentified { .. }), "{error}");
        let text = error.to_string();
        assert!(text.contains(HEDGE_WITNESS_ID), "{text}");
        assert!(text.contains("latent_block_witness.rs"), "{text}");
    }
}

#[test]
fn a_pag_with_a_hedged_completion_keeps_the_identified_mass() {
    on_large_stack("pag-hedge", pag_hedge_body);
}

fn pag_hedge_body() {
    let ctx = ExecutionContext::for_tests(13);
    let data = chain_table(25.0);
    let query = anomaly_query(1);
    // x o-> y: the MAG completions are x -> y (an invisible edge, so a latent common cause may
    // also be present, and the mechanism of y is hedged) and x <-> y (y is a root).
    let mut pag = Pag::with_variables(2);
    pag.insert_circle_arrow(node(0), node(1)).unwrap();
    let result = tabular_study(&data, pag, &query, &frequentist()).run(&ctx).unwrap();
    let mixture = result.structural_response.as_ref().expect("the class mixture");
    assert_eq!(mixture.atoms.len(), 2);
    assert!((mixture.identified_mass - 0.5).abs() < 1e-12);
    assert!((mixture.unidentified_mass - 0.5).abs() < 1e-12);
    assert_eq!(
        mixture.atoms.iter().filter(|a| a.status == IdentificationStatus::NotIdentified).count(),
        1
    );
    assert_eq!(result.identification.status, IdentificationStatus::PartiallyIdentified);
    // The identified completion is the one where y is a root: the DAG route on the empty graph.
    let root = tabular_study(&data, dag_of(2, &[]), &query, &frequentist()).run(&ctx).unwrap();
    let (got, want) = (&result.anomaly().unwrap()[0], &root.anomaly().unwrap()[0]);
    assert_eq!(got.noise_components, want.noise_components);
    for row in 0..got.rows.len() {
        assert!((contribution(got, row, 1) - contribution(want, row, 1)).abs() < 1e-9);
    }
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "attribution.class.noise_block_hedge")
    );
}

#[test]
fn a_prepared_class_attribution_seals_and_repeats_the_run() {
    on_large_stack("class-prepared", prepared_body);
}

fn prepared_body() {
    let ctx = ExecutionContext::for_tests(13);
    let data = chain_table(25.0);
    let query = anomaly_query(1);
    let mut cpdag = Cpdag::with_variables(2);
    cpdag.insert_undirected(node(0), node(1)).unwrap();
    let study = tabular_study(&data, cpdag.clone(), &query, &frequentist());
    let run = study.run(&ctx).unwrap();
    let mut prepared = study.prepare(&ctx).unwrap();
    assert!(matches!(prepared.retained_operation(), RetainedOperation::ClassAttribution));
    let estimated = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(estimated.anomaly().unwrap()[0].scores, run.anomaly().unwrap()[0].scores);
    assert_eq!(
        estimated.anomaly().unwrap()[0].noise_contributions,
        run.anomaly().unwrap()[0].noise_contributions
    );
    let changed = chain_table(30.0);
    let expected = tabular_study(&changed, cpdag, &query, &frequentist()).run(&ctx).unwrap();
    let refreshed = prepared.refresh(changed, &ctx).unwrap();
    assert_eq!(refreshed.anomaly().unwrap(), expected.anomaly().unwrap());
    assert_eq!(refreshed.lowered_program_identity(), prepared.plan().lowered_program_identity());
    let artifact =
        prepared.encode_contracted_result(&refreshed, "retained-class-attribution", &ctx).unwrap();
    assert!(antecedent_io::consume_analysis_result(&artifact).is_ok());
}

/// A series of `x` and `y_t = 0.8 x_{t-1} + noise`, `z_t = noise`.
fn lagged_series(outlier: f64) -> TimeSeriesData {
    let rows = 80;
    let (s1, s2, s3) = sines(rows);
    let x: Vec<f64> = s1.iter().map(|v| 2.0 * v).collect();
    let mut y: Vec<f64> =
        (0..rows).map(|t| 0.8 * if t == 0 { x[0] } else { x[t - 1] } + 0.3 * s2[t]).collect();
    y[rows - 1] += outlier;
    let z: Vec<f64> = s3.clone();
    TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("y", y.as_slice()), ("z", z.as_slice())],
        1,
    )
    .unwrap()
}

fn series_study(data: &TimeSeriesData, graph: impl IntoGraphInput) -> Study {
    Study::series(data.clone())
        .graph(graph)
        .query(anomaly_query(1))
        .inference(frequentist())
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

#[test]
fn a_temporal_cpdag_with_one_completion_is_the_temporal_dag_route() {
    on_large_stack("temporal-cpdag-single", temporal_cpdag_single_body);
}

fn temporal_cpdag_single_body() {
    let ctx = ExecutionContext::for_tests(13);
    let data = lagged_series(8.0);
    // x_{t-1} -> y_t, compelled: the class has one completion.
    let mut cpdag = TemporalCpdag::empty();
    let parent = cpdag.add_lagged(variable(0), Lag::from_raw(1)).unwrap();
    let child = cpdag.add_lagged(variable(1), Lag::CONTEMPORANEOUS).unwrap();
    cpdag.insert_directed(parent, child).unwrap();
    let mut dag = TemporalDag::empty();
    let parent = dag.add_lagged(variable(0), Lag::from_raw(1)).unwrap();
    let child = dag.add_lagged(variable(1), Lag::CONTEMPORANEOUS).unwrap();
    dag.insert_directed(parent, child).unwrap();
    let study = series_study(&data, cpdag.clone());
    let class = study.run(&ctx).unwrap();
    assert_eq!(class.support_status, Some(CellStatus::Refused));
    let reference = series_study(&data, dag).run(&ctx).unwrap();
    assert_eq!(class.anomaly().unwrap(), reference.anomaly().unwrap());
    let mut prepared = study.prepare(&ctx).unwrap();
    assert!(matches!(prepared.retained_operation(), RetainedOperation::ClassAttribution));
    let retained = prepared.estimate_series(&data, &ctx).unwrap();
    assert_eq!(retained.anomaly().unwrap(), class.anomaly().unwrap());
    let changed = lagged_series(10.0);
    let expected = series_study(&changed, cpdag).run(&ctx).unwrap();
    let refreshed = prepared.refresh_series(changed, &ctx).unwrap();
    assert_eq!(refreshed.anomaly().unwrap(), expected.anomaly().unwrap());
    assert_eq!(refreshed.lowered_program_identity(), prepared.plan().lowered_program_identity());
    let artifact = prepared
        .encode_contracted_result(&refreshed, "retained-temporal-class-attribution", &ctx)
        .unwrap();
    assert!(antecedent_io::consume_analysis_result(&artifact).is_ok());
}

#[test]
fn a_temporal_cpdag_is_the_mean_of_its_temporal_dag_completions() {
    on_large_stack("temporal-cpdag-pair", temporal_cpdag_pair_body);
}

fn temporal_cpdag_pair_body() {
    let ctx = ExecutionContext::for_tests(13);
    let data = lagged_series(8.0);
    // x and y joined by an undirected contemporaneous edge: x -> y and y -> x.
    let mut cpdag = TemporalCpdag::empty();
    let x = cpdag.add_lagged(variable(0), Lag::CONTEMPORANEOUS).unwrap();
    let y = cpdag.add_lagged(variable(1), Lag::CONTEMPORANEOUS).unwrap();
    cpdag.insert_undirected(x, y).unwrap();
    let class = series_study(&data, cpdag).run(&ctx).unwrap();
    let member = |from: u32, to: u32| {
        let mut dag = TemporalDag::empty();
        let a = dag.add_lagged(variable(from), Lag::CONTEMPORANEOUS).unwrap();
        let b = dag.add_lagged(variable(to), Lag::CONTEMPORANEOUS).unwrap();
        dag.insert_directed(a, b).unwrap();
        series_study(&data, dag).run(&ctx).unwrap()
    };
    let (forward, backward) = (member(0, 1), member(1, 0));
    let (forward, backward) = (&forward.anomaly().unwrap()[0], &backward.anomaly().unwrap()[0]);
    let merged = &class.anomaly().unwrap()[0];
    for row in 0..merged.rows.len() {
        for component in [0, 1] {
            let expected = 0.5
                * (contribution(forward, row, component) + contribution(backward, row, component));
            assert!((contribution(merged, row, component) - expected).abs() < 1e-9);
        }
    }
    assert_eq!(class.structural_response.as_ref().unwrap().atoms.len(), 2);
}

#[test]
fn a_temporal_admg_whose_latent_edge_misses_the_ancestors_is_the_temporal_dag_route() {
    on_large_stack("temporal-admg-outside", temporal_admg_outside_body);
}

fn temporal_admg_outside_body() {
    let ctx = ExecutionContext::for_tests(13);
    let data = lagged_series(8.0);
    // x_{t-1} -> y_t, and x_t <-> y_t: the latent edge joins x_t and y_t, but x_t is not an
    // ancestor of y_t, so the ancestors of y_t keep singleton blocks.
    let mut template = TemporalAdmg::empty();
    let lagged_x = template.add_lagged(variable(0), Lag::from_raw(1)).unwrap();
    let y = template.add_lagged(variable(1), Lag::CONTEMPORANEOUS).unwrap();
    let x = template.add_lagged(variable(0), Lag::CONTEMPORANEOUS).unwrap();
    template.insert_directed(lagged_x, y).unwrap();
    template.insert_bidirected(x, y).unwrap();
    let class = series_study(&data, template).run(&ctx).unwrap();
    assert_eq!(class.support_status, Some(CellStatus::Refused));

    let mut dag = TemporalDag::empty();
    let parent = dag.add_lagged(variable(0), Lag::from_raw(1)).unwrap();
    let child = dag.add_lagged(variable(1), Lag::CONTEMPORANEOUS).unwrap();
    dag.add_lagged(variable(0), Lag::CONTEMPORANEOUS).unwrap();
    dag.insert_directed(parent, child).unwrap();
    let reference = series_study(&data, dag).run(&ctx).unwrap();
    let (got, want) = (&class.anomaly().unwrap()[0], &reference.anomaly().unwrap()[0]);
    assert_eq!(got.rows, want.rows);
    for row in 0..got.rows.len() {
        assert!((got.scores[row] - want.scores[row]).abs() < 1e-9);
        for component in [0, 1, 3] {
            assert!(
                (contribution(got, row, component) - contribution(want, row, component)).abs()
                    < 1e-9,
                "row {row} component {component}"
            );
        }
    }
}

#[test]
fn a_temporal_admg_with_a_confounded_lag_edge_is_refused_with_the_witness() {
    on_large_stack("temporal-admg-hedge", temporal_admg_hedge_body);
}

fn temporal_admg_hedge_body() {
    let ctx = ExecutionContext::for_tests(13);
    let data = lagged_series(8.0);
    // x_{t-1} -> y_t with x_{t-1} <-> y_t: the lagged cause and its effect share a latent cause.
    let mut template = TemporalAdmg::empty();
    let lagged_x = template.add_lagged(variable(0), Lag::from_raw(1)).unwrap();
    let y = template.add_lagged(variable(1), Lag::CONTEMPORANEOUS).unwrap();
    template.insert_directed(lagged_x, y).unwrap();
    template.insert_bidirected(lagged_x, y).unwrap();
    let error = series_study(&data, template).run(&ctx).unwrap_err();
    assert!(matches!(error, CausalError::NotIdentified { .. }), "{error}");
    assert!(error.to_string().contains(HEDGE_WITNESS_ID));
}
