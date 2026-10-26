//! Anomaly and change attribution over a graph posterior.
//!
//! Each atom is attributed as the supplied structure it stands for is, so every expected value
//! is read from the explicit route of that structure on the same table: the posterior moves
//! only which structure attributes, how the atoms' contributions are weighted, and how much
//! posterior mass attributes nothing.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::{CellStatus, InferenceMode, RefuteSuite, StructuralWeightBasis, Study};
use antecedent_attribution::AnomalyScores;
use antecedent_core::{
    AnomalyAttributionQuery, CausalQuery, ChangeAttributionQuery, ComponentId, ExecutionContext,
    IdentificationStatus, Lag, PopulationSelector, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_discovery::{
    GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, adjacency_mask_from_cpdag, set_edge,
    with_lagged_edge,
};
use antecedent_graph::{Cpdag, Dag, DenseNodeId, TemporalDag};
use antecedent_prob::InferenceDiagnostics;

/// Debug builds of the shared dispatcher carry frames close to the default test-thread stack.
fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("graph-posterior-attribution".into())
        .stack_size(64 * 1024 * 1024)
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

/// `x` and `y = 2x + noise`, with the last row's `y` far out.
fn chain_table(outlier: f64) -> TabularData {
    let rows = 60;
    let (s1, _, s3) = sines(rows);
    let x: Vec<f64> = s1.iter().map(|v| 2.0 * v).collect();
    let mut y: Vec<f64> = (0..rows).map(|i| 2.0 * x[i] + 0.4 * s3[i]).collect();
    y[rows - 1] += outlier;
    TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap()
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
    TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap()
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

fn sampling() -> SamplingSummary {
    SamplingSummary {
        ess: 1.0,
        diagnostics: InferenceDiagnostics::analytic("graph_posterior_attribution"),
        rejected_invalid: 0,
    }
}

fn dag_of(edges: &[(u32, u32)]) -> Dag {
    let mut graph = Dag::with_variables(2);
    for &(from, to) in edges {
        graph.insert_directed(node(from), node(to)).unwrap();
    }
    graph
}

fn mask_of(edges: &[(usize, usize)]) -> u64 {
    edges.iter().fold(0_u64, |mask, &(from, to)| set_edge(mask, 2, from, to, true))
}

fn posterior(kind: GraphPosteriorAtomKind, weights: &[f64], masks: Vec<u64>) -> GraphPosterior {
    GraphPosterior::new(2, weights.to_vec(), masks, vec![0.0; 4], vec![0.0; 4], sampling())
        .unwrap()
        .with_atom_kind(kind)
}

fn posterior_study(data: &TabularData, graphs: GraphPosterior, query: &CausalQuery) -> Study {
    Study::tabular(data.clone())
        .graph_posterior(graphs)
        .query(query.clone())
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
}

fn explicit_study(data: &TabularData, graph: Dag, query: &CausalQuery) -> Study {
    Study::tabular(data.clone())
        .graph(graph)
        .query(query.clone())
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
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

/// The atoms attribute with the weights of their posterior probability: the published
/// contribution of each component is the weighted mean of the DAG routes' own contributions
/// over the atoms that attribute, the cyclic mask is unidentified mass, and nothing is
/// redistributed.
#[test]
fn an_anomaly_is_the_posterior_weighted_mean_of_the_atoms_dag_attributions() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_101);
        let data = chain_table(25.0);
        let query = anomaly_query(1);
        let graphs = posterior(
            GraphPosteriorAtomKind::Dag,
            &[0.6, 0.3, 0.1],
            vec![mask_of(&[(0, 1)]), mask_of(&[(1, 0)]), mask_of(&[(0, 1), (1, 0)])],
        );
        let keys = graphs.graph_keys.clone();
        let built = posterior_study(&data, graphs, &query);
        assert_eq!(built.support_status(), Some(CellStatus::Refused));
        let result = built.run(&ctx).unwrap();
        assert_eq!(result.support_status, Some(CellStatus::Refused));
        let merged = &result.anomaly().expect("anomaly payload")[0];

        let forward = explicit_study(&data, dag_of(&[(0, 1)]), &query).run(&ctx).unwrap();
        let backward = explicit_study(&data, dag_of(&[(1, 0)]), &query).run(&ctx).unwrap();
        let (forward, backward) = (&forward.anomaly().unwrap()[0], &backward.anomaly().unwrap()[0]);
        assert_eq!(merged.scores, forward.scores, "the factual scores do not depend on the graph");
        // The two atoms that attribute carry 0.6 and 0.3 of the 0.9 identified mass.
        let (wf, wb) = (0.6 / 0.9, 0.3 / 0.9);
        for row in 0..merged.rows.len() {
            for component in [0, 1] {
                let expected = wf * contribution(forward, row, component)
                    + wb * contribution(backward, row, component);
                assert!(
                    (contribution(merged, row, component) - expected).abs() < 1e-9,
                    "row {row} component {component}"
                );
            }
        }
        // `x` is a player of the forward DAG only: its contribution is the forward mean shrunk
        // by the forward atom's share of the identified mass.
        assert!(mean_over_rows(forward, 0).abs() > 1e-6);
        assert!((mean_over_rows(merged, 0) - wf * mean_over_rows(forward, 0)).abs() < 1e-9);

        let mixture = result.structural_response.as_ref().expect("the atom mixture");
        assert_eq!(mixture.weight_basis, StructuralWeightBasis::PosteriorProbability);
        assert_eq!(mixture.atoms.len(), 3);
        assert!((mixture.identified_mass - 0.9).abs() < 1e-9, "{}", mixture.identified_mass);
        assert!((mixture.unidentified_mass - 0.1).abs() < 1e-9, "{}", mixture.unidentified_mass);
        assert!(mixture.unevaluable_mass.abs() < 1e-12);
        let weight_of =
            |key: u64| mixture.atoms.iter().find(|a| a.graph_key == key).unwrap().weight;
        assert!((weight_of(keys[0]) - 0.6).abs() < 1e-9);
        assert!((weight_of(keys[1]) - 0.3).abs() < 1e-9);
        assert!((weight_of(keys[2]) - 0.1).abs() < 1e-9);
        assert_eq!(result.identification.status, IdentificationStatus::GraphDependent);
        assert!(result.diagnostics.iter().any(|d| d.code.as_ref() == "attribution.class.envelope"));

        // The prepared handle answers the same attribution.
        let prepared = built.prepare(&ctx).unwrap();
        let again = prepared.estimate(&data, &ctx).unwrap();
        let again = &again.anomaly().unwrap()[0];
        assert!((mean_over_rows(again, 0) - mean_over_rows(merged, 0)).abs() < 1e-9);
    });
}

/// A change attribution publishes the posterior-weighted contributions with the spread across
/// atoms as the graph sensitivity.
#[test]
fn a_change_attribution_carries_the_graph_sensitivity_across_the_atoms() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_102);
        let data = shifted_chain_table(5.0);
        let query = change_query(1, 80);
        let graphs = posterior(
            GraphPosteriorAtomKind::Dag,
            &[0.5, 0.5],
            vec![mask_of(&[(0, 1)]), mask_of(&[(1, 0)])],
        );
        let result = posterior_study(&data, graphs, &query).run(&ctx).unwrap();
        let change = result.change_attribution().expect("change payload");
        // The mechanism of `y` shifts by 5 and `x` does not move.
        assert!((change.total_change - 5.0).abs() < 0.5, "total {}", change.total_change);
        let share = |id: u32| {
            change
                .contributions
                .iter()
                .find(|c| c.component == ComponentId::from_variable(variable(id)))
                .map_or(0.0, |c| c.contribution)
        };
        assert!(share(0).abs() < 1e-9, "x does not change: {}", share(0));
        assert!((share(1) - change.total_change).abs() < 1e-6);
        let spread = change.graph_sensitivity.as_ref().expect("graph sensitivity");
        assert_eq!(spread.len(), change.contributions.len());
        assert!(spread.iter().all(|s| *s < 0.5), "{spread:?}");

        // The same change through the explicit forward DAG: the posterior mean of two atoms
        // that agree is that value.
        let forward = explicit_study(&data, dag_of(&[(0, 1)]), &query).run(&ctx).unwrap();
        let forward = forward.change_attribution().unwrap();
        assert!((forward.total_change - change.total_change).abs() < 0.5);
    });
}

/// A Cpdag atom is attributed member by member over its completions, sharing its posterior
/// probability equally among them, so a one-class posterior equals the explicit class route.
#[test]
fn a_cpdag_atom_is_attributed_over_its_completions_as_the_class_route_does() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_103);
        let data = chain_table(25.0);
        let query = anomaly_query(1);
        let mut cpdag = Cpdag::with_variables(2);
        cpdag.insert_undirected(node(0), node(1)).unwrap();
        let mask = adjacency_mask_from_cpdag(&cpdag).unwrap();
        let graphs = posterior(GraphPosteriorAtomKind::Cpdag, &[1.0], vec![mask]);
        let result = posterior_study(&data, graphs, &query).run(&ctx).unwrap();
        let explicit = Study::tabular(data.clone())
            .graph(cpdag)
            .query(query)
            .refute(RefuteSuite::None)
            .build()
            .unwrap()
            .run(&ctx)
            .unwrap();
        let (ours, theirs) = (&result.anomaly().unwrap()[0], &explicit.anomaly().unwrap()[0]);
        assert_eq!(ours.noise_components, theirs.noise_components);
        for row in 0..ours.rows.len() {
            for component in [0, 1] {
                assert!(
                    (contribution(ours, row, component) - contribution(theirs, row, component))
                        .abs()
                        < 1e-9,
                    "row {row} component {component}"
                );
            }
        }
        let mixture = result.structural_response.as_ref().expect("the atom mixture");
        assert_eq!(mixture.weight_basis, StructuralWeightBasis::PosteriorProbability);
        assert!((mixture.identified_mass - 1.0).abs() < 1e-9);
    });
}

/// A posterior none of whose atoms is a structure of its class attributes nothing.
#[test]
fn a_posterior_with_no_attributable_atom_is_refused() {
    on_large_stack(|| {
        let graphs =
            posterior(GraphPosteriorAtomKind::Dag, &[1.0], vec![mask_of(&[(0, 1), (1, 0)])]);
        let study = posterior_study(&chain_table(25.0), graphs, &anomaly_query(1));
        assert!(study.run(&ExecutionContext::for_tests(8_104)).is_err());
    });
}

/// Cheap and full validation refit the whole attribution vector by the same compiled posterior
/// program on perturbed data: subset stability under cheap, and bootstrap stability under full.
#[test]
fn validation_refits_the_whole_attribution_through_the_posterior_program() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_105);
        let data = chain_table(25.0);
        let query = anomaly_query(1);
        let run = |suite: RefuteSuite| {
            let graphs = posterior(
                GraphPosteriorAtomKind::Dag,
                &[0.6, 0.4],
                vec![mask_of(&[(0, 1)]), mask_of(&[(1, 0)])],
            );
            Study::tabular(data.clone())
                .graph_posterior(graphs)
                .query(query.clone())
                .inference(InferenceMode::Frequentist)
                .refute(suite)
                .build()
                .unwrap_or_else(|e| panic!("{suite:?}: {e}"))
                .run(&ctx)
                .unwrap_or_else(|e| panic!("{suite:?}: {e}"))
        };
        let none = run(RefuteSuite::None);
        assert!(none.refutations.is_empty());
        let cheap = run(RefuteSuite::Cheap);
        assert!(cheap.refutations.iter().any(|r| r.refuter.as_ref() == "target.subset_stability"));
        assert!(
            cheap.refutations.iter().all(|r| r.refuter.as_ref() != "target.bootstrap_stability")
        );
        let full = run(RefuteSuite::Full);
        assert!(
            full.refutations.iter().any(|r| r.refuter.as_ref() == "target.bootstrap_stability")
        );
        // Validation observes the attribution; it does not move it.
        let (a, b) = (&none.anomaly().unwrap()[0], &cheap.anomaly().unwrap()[0]);
        assert!((mean_over_rows(a, 0) - mean_over_rows(b, 0)).abs() < 1e-9);
    });
}

/// A series of `x`, `y_t = 0.8 x_{t-1} + noise` and an unrelated `z`, with the last `y` far out.
fn lagged_series(outlier: f64) -> TimeSeriesData {
    let rows = 80;
    let (s1, s2, s3) = sines(rows);
    let x: Vec<f64> = s1.iter().map(|v| 2.0 * v).collect();
    let mut y: Vec<f64> =
        (0..rows).map(|t| 0.8 * if t == 0 { x[0] } else { x[t - 1] } + 0.3 * s2[t]).collect();
    y[rows - 1] += outlier;
    TimeSeriesData::from_f64_columns(
        [("x", x.as_slice()), ("y", y.as_slice()), ("z", s3.as_slice())],
        1,
    )
    .unwrap()
}

/// A posterior over lag-one templates of `(x, y, z)`: `x(t-1) -> y(t)` and the empty template.
fn template_posterior(weights: &[f64]) -> GraphPosterior {
    let lagged = with_lagged_edge(0, 3, 1, 0, 1, true).unwrap();
    GraphPosterior::from_atoms(
        3,
        weights.to_vec(),
        [0_u64, 0_u64],
        vec![0.0; 9],
        vec![0.0; 9],
        sampling(),
    )
    .unwrap()
    .with_lagged_marginals(1, vec![0.0; 9])
    .unwrap()
    .with_lag_masks(vec![lagged, 0])
    .unwrap()
}

fn series_reference(
    driven: bool,
    data: &TimeSeriesData,
    query: &CausalQuery,
) -> antecedent::StudyResult {
    let mut dag = TemporalDag::empty();
    if driven {
        let parent = dag.add_lagged(variable(0), Lag::from_raw(1)).unwrap();
        let child = dag.add_lagged(variable(1), Lag::CONTEMPORANEOUS).unwrap();
        dag.insert_directed(parent, child).unwrap();
    } else {
        dag.add_lagged(variable(1), Lag::CONTEMPORANEOUS).unwrap();
    }
    Study::series(data.clone())
        .graph(dag)
        .query(query.clone())
        .inference(InferenceMode::Frequentist)
        .refute(RefuteSuite::None)
        .build()
        .unwrap()
        .run(&ExecutionContext::for_tests(8_106))
        .unwrap()
}

/// A posterior of lag templates attributes each template as a supplied temporal DAG does and
/// weights the factual baseline scores by the posterior probability of the templates.
#[test]
fn a_series_posterior_attributes_each_template_as_a_supplied_temporal_dag() {
    on_large_stack(|| {
        let ctx = ExecutionContext::for_tests(8_107);
        let data = lagged_series(8.0);
        let query = anomaly_query(1);
        let graphs = template_posterior(&[0.7, 0.3]);
        let keys = graphs.graph_keys.clone();
        let built = Study::series(data.clone())
            .graph_posterior(graphs)
            .query(query.clone())
            .inference(InferenceMode::Frequentist)
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        assert_eq!(built.support_status(), Some(CellStatus::Refused));
        let result = built.run(&ctx).unwrap();
        let merged = &result.anomaly().expect("anomaly payload")[0];
        let (driven, empty) =
            (series_reference(true, &data, &query), series_reference(false, &data, &query));
        let (driven, empty) = (&driven.anomaly().unwrap()[0], &empty.anomaly().unwrap()[0]);
        assert_eq!(merged.scores, driven.scores, "the factual scores do not depend on the graph");
        for row in 0..merged.rows.len() {
            let expected = 0.7 * driven.baseline_scores[row] + 0.3 * empty.baseline_scores[row];
            assert!((merged.baseline_scores[row] - expected).abs() < 1e-9, "row {row}");
        }
        let mixture = result.structural_response.as_ref().expect("the atom mixture");
        assert_eq!(mixture.atoms.len(), 2);
        assert!((mixture.identified_mass - 1.0).abs() < 1e-9);
        assert!(keys.len() == 2);
    });
}
