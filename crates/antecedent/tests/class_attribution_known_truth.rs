//! Known-truth evidence for anomaly and change attribution over a structure class.
//!
//! Every expected number is read from `conformance/estimate/class_attribution_truth/expected.json`,
//! which `reference.py` derives from the definitions in exact rationals (a least-squares
//! mechanism per node, the abducted residuals as noise, the two-sided Gaussian tail, the exact
//! Shapley value over the members' players, and the mean-level mechanism-swap game for a change),
//! with a Dirichlet row-weight posterior in closed form. A structure that stands for several is
//! attributed member by member and merged: a Cpdag over its DAGs, a Pag over its MAG completions
//! (each read as its most-confounded mixed graph), a mixed graph over the noise blocks of its
//! ancestral sub-graph, and their one-lag temporal forms over the lag-aligned design. The test
//! holds the merged contributions, the identified set, the completion masses and the typed
//! refusals to that derivation, under a supplied and an accepted structure, both inference
//! modes, and again from the retained plan after the builder is gone.
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::RetainedOperation;
use antecedent::{
    AcceptedGraph, BayesianConfig, CausalError, CellStatus, InferenceMode, IntoGraphInput,
    RefuteSuite, Study, StudyResult,
};
use antecedent_attribution::HEDGE_WITNESS_ID;
use antecedent_core::{
    AnomalyAttributionQuery, AnomalyReference, CausalQuery, ChangeAttributionQuery, ComponentId,
    ExecutionContext, IdentificationStatus, IntervalMethod, Lag, OrderedFloatBits,
    PopulationSelector, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_discovery::{
    GraphPosterior, GraphPosteriorAtomKind, LagMask, SamplingSummary, adjacency_mask_from_cpdag,
    set_edge, with_lagged_edge,
};
use antecedent_estimate::CausalPosterior;
use antecedent_graph::{
    Admg, Cpdag, DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag, TemporalAdmg, TemporalCpdag,
    TemporalPag,
};
use antecedent_prob::InferenceDiagnostics;
use serde_json::Value;

fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("class-attribution-known-truth".into())
        .stack_size(64 * 1024 * 1024)
        .spawn(body)
        .expect("spawn evidence thread")
        .join()
        .expect("evidence body panicked");
}

fn fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/class_attribution_truth/expected.json"
    ))
    .expect("class_attribution_truth fixture parses")
}

fn variable(index: u32) -> VariableId {
    VariableId::from_raw(index)
}

fn component(index: u64) -> ComponentId {
    ComponentId::from_variable(variable(u32::try_from(index).unwrap()))
}

fn floats(value: &Value) -> Vec<f64> {
    value.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect()
}

fn close(got: f64, want: f64, tolerance: f64, label: &str) {
    assert!((got - want).abs() <= tolerance, "{label}: {got} vs {want} (tolerance {tolerance})");
}

/// One fixture case.
struct Case<'a> {
    name: &'a str,
    value: &'a Value,
}

impl<'a> Case<'a> {
    fn of(document: &'a Value, name: &'a str) -> Self {
        Self { name, value: &document["cases"][name] }
    }

    fn is_series(&self) -> bool {
        self.value["series"].as_bool().unwrap_or(false)
    }

    fn is_anomaly(&self) -> bool {
        self.value["kind"] == "anomaly"
    }

    fn columns(&self) -> Vec<(String, Vec<f64>)> {
        self.value["column_order"]
            .as_array()
            .unwrap()
            .iter()
            .map(|name| {
                let name = name.as_str().unwrap();
                (name.to_owned(), floats(&self.value["columns"][name]))
            })
            .collect()
    }

    fn target_index(&self) -> u32 {
        let target = self.value["target"].as_str().unwrap();
        let position =
            self.value["column_order"].as_array().unwrap().iter().position(|c| c == target);
        u32::try_from(position.expect("the target is a column")).unwrap()
    }

    fn query(&self) -> CausalQuery {
        let target = variable(self.target_index());
        if self.is_anomaly() {
            let reference = &self.value["reference"];
            let query = AnomalyAttributionQuery::new([target], 1000);
            return CausalQuery::AnomalyAttribution(if reference["kind"] == "fixed" {
                query.with_reference(AnomalyReference::Fixed {
                    center: OrderedFloatBits::from_f64(reference["center"].as_f64().unwrap()),
                    scale: OrderedFloatBits::from_f64(reference["scale"].as_f64().unwrap()),
                })
            } else {
                query
            });
        }
        let range = |index: usize| {
            let bounds = &self.value["populations"][index];
            PopulationSelector::TimeRange {
                start: usize::try_from(bounds[0].as_u64().unwrap()).unwrap(),
                end: usize::try_from(bounds[1].as_u64().unwrap()).unwrap(),
            }
        };
        CausalQuery::ChangeAttribution(ChangeAttributionQuery::new(target, range(0), range(1)))
    }

    fn rows(&self) -> usize {
        self.value["columns"][self.value["column_order"][0].as_str().unwrap()]
            .as_array()
            .unwrap()
            .len()
    }

    fn derived(&self) -> &Value {
        &self.value["derived"]
    }

    fn merged(&self) -> &Value {
        &self.value["derived"]["merged"]
    }

    fn members(&self) -> &Vec<Value> {
        self.value["members"].as_array().unwrap()
    }

    fn hedged(&self) -> usize {
        self.members().iter().filter(|m| m["status"] == "hedged").count()
    }

    fn component_id(&self, name: &Value) -> u64 {
        self.value["component_ids"][name.as_str().unwrap()].as_u64().unwrap()
    }
}

/// The data the case's study reads.
enum Data {
    Table(TabularData),
    Series(TimeSeriesData),
}

fn data_of(case: &Case<'_>) -> Data {
    let columns = case.columns();
    let named: Vec<(&str, &[f64])> =
        columns.iter().map(|(name, values)| (name.as_str(), values.as_slice())).collect();
    if case.is_series() {
        Data::Series(TimeSeriesData::from_f64_columns(named, 1).unwrap())
    } else {
        Data::Table(TabularData::from_f64_columns(named).unwrap())
    }
}

fn node(index: u64) -> DenseNodeId {
    DenseNodeId::from_raw(u32::try_from(index).unwrap())
}

fn endpoint(mark: &Value) -> Endpoint {
    match mark.as_str().unwrap() {
        "tail" => Endpoint::Tail,
        "arrow" => Endpoint::Arrow,
        "circle" => Endpoint::Circle,
        other => panic!("unknown mark {other}"),
    }
}

fn marked(edge: &Value) -> MarkedEdge {
    MarkedEdge {
        a: node(edge[0].as_u64().unwrap()),
        b: node(edge[1].as_u64().unwrap()),
        at_a: endpoint(&edge[2]),
        at_b: endpoint(&edge[3]),
        middle: MiddleMark::Empty,
    }
}

fn lag(value: &Value) -> Lag {
    match value.as_u64().unwrap() {
        0 => Lag::CONTEMPORANEOUS,
        lag => Lag::from_raw(u32::try_from(lag).unwrap()),
    }
}

/// The case's structure, in the class that carries it.
enum Structure {
    Cpdag(Cpdag),
    Pag(Pag),
    Admg(Admg),
    TemporalCpdag(TemporalCpdag),
    TemporalAdmg(TemporalAdmg),
    TemporalPag(TemporalPag),
}

fn pairs<'a>(graph: &'a Value, key: &str) -> impl Iterator<Item = (u64, u64)> + 'a {
    graph[key]
        .as_array()
        .into_iter()
        .flatten()
        .map(|edge| (edge[0].as_u64().unwrap(), edge[1].as_u64().unwrap()))
}

fn structure_of(case: &Case<'_>) -> Structure {
    let graph = &case.value["graph"];
    let count = || u32::try_from(graph["nodes"].as_u64().unwrap()).unwrap();
    match graph["class"].as_str().unwrap() {
        "cpdag" => {
            let mut g = Cpdag::with_variables(count());
            for (a, b) in pairs(graph, "directed") {
                g.insert_directed(node(a), node(b)).unwrap();
            }
            for (a, b) in pairs(graph, "undirected") {
                g.insert_undirected(node(a), node(b)).unwrap();
            }
            Structure::Cpdag(g)
        }
        "pag" => {
            let mut g = Pag::with_variables(count());
            for edge in graph["marked"].as_array().unwrap() {
                g.insert_marked(marked(edge)).unwrap();
            }
            Structure::Pag(g)
        }
        "admg" => {
            let mut g = Admg::with_variables(count());
            for (a, b) in pairs(graph, "directed") {
                g.insert_directed(node(a), node(b)).unwrap();
            }
            for (a, b) in pairs(graph, "bidirected") {
                g.insert_bidirected(node(a), node(b)).unwrap();
            }
            Structure::Admg(g)
        }
        "temporal_cpdag" => {
            let mut g = TemporalCpdag::empty();
            let ids: Vec<_> = graph["nodes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| {
                    g.add_lagged(
                        variable(u32::try_from(n[0].as_u64().unwrap()).unwrap()),
                        lag(&n[1]),
                    )
                    .unwrap()
                })
                .collect();
            for (a, b) in pairs(graph, "directed") {
                g.insert_directed(ids[a as usize], ids[b as usize]).unwrap();
            }
            for (a, b) in pairs(graph, "undirected") {
                g.insert_undirected(ids[a as usize], ids[b as usize]).unwrap();
            }
            Structure::TemporalCpdag(g)
        }
        "temporal_admg" => {
            let mut g = TemporalAdmg::empty();
            let ids: Vec<_> = graph["nodes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| {
                    g.add_lagged(
                        variable(u32::try_from(n[0].as_u64().unwrap()).unwrap()),
                        lag(&n[1]),
                    )
                    .unwrap()
                })
                .collect();
            for (a, b) in pairs(graph, "directed") {
                g.insert_directed(ids[a as usize], ids[b as usize]).unwrap();
            }
            for (a, b) in pairs(graph, "bidirected") {
                g.insert_bidirected(ids[a as usize], ids[b as usize]).unwrap();
            }
            Structure::TemporalAdmg(g)
        }
        "temporal_pag" => {
            let mut g = TemporalPag::empty();
            let ids: Vec<_> = graph["nodes"]
                .as_array()
                .unwrap()
                .iter()
                .map(|n| {
                    g.add_lagged(
                        variable(u32::try_from(n[0].as_u64().unwrap()).unwrap()),
                        lag(&n[1]),
                    )
                    .unwrap()
                })
                .collect();
            for edge in graph["marked"].as_array().unwrap() {
                let mut edge = marked(edge);
                edge.a = ids[edge.a.as_usize()];
                edge.b = ids[edge.b.as_usize()];
                g.insert_marked(edge).unwrap();
            }
            Structure::TemporalPag(g)
        }
        other => panic!("unknown structure class {other}"),
    }
}

fn assemble(
    data: Data,
    graph: impl IntoGraphInput,
    query: CausalQuery,
    inference: &InferenceMode,
    suite: RefuteSuite,
) -> Result<Study, CausalError> {
    let builder = match data {
        Data::Table(table) => Study::tabular(table).graph(graph),
        Data::Series(series) => Study::series(series).graph(graph),
    };
    builder.query(query).inference(inference.clone()).refute(suite).build()
}

/// The study of `case` under a supplied (`accepted = false`) or accepted structure.
fn study_of(
    case: &Case<'_>,
    accepted: bool,
    inference: &InferenceMode,
    suite: RefuteSuite,
) -> Study {
    try_study_of(case, accepted, inference, suite).unwrap()
}

/// As [`study_of`], keeping the refusal of a build.
fn try_study_of(
    case: &Case<'_>,
    accepted: bool,
    inference: &InferenceMode,
    suite: RefuteSuite,
) -> Result<Study, CausalError> {
    let (data, query) = (data_of(case), case.query());
    if !case.value["posterior"].is_null() {
        let builder = match data {
            Data::Table(table) => Study::tabular(table).graph_posterior(posterior_of(case)),
            Data::Series(series) => Study::series(series).graph_posterior(posterior_of(case)),
        };
        return builder.query(query).inference(inference.clone()).refute(suite).build();
    }
    macro_rules! with {
        ($graph:expr) => {
            if accepted {
                assemble(data, AcceptedGraph::from($graph), query, inference, suite)
            } else {
                assemble(data, $graph, query, inference, suite)
            }
        };
    }
    match structure_of(case) {
        Structure::Cpdag(g) => with!(g),
        Structure::Pag(g) => with!(g),
        Structure::Admg(g) => with!(g),
        Structure::TemporalCpdag(g) => with!(g),
        Structure::TemporalAdmg(g) => with!(g),
        Structure::TemporalPag(g) => with!(g),
    }
}

/// The graph posterior of a case: its atoms with their probabilities, and a mask that is no
/// structure of the posterior's kind as the last atom where the case declares outside mass.
fn posterior_of(case: &Case<'_>) -> GraphPosterior {
    let spec = &case.value["posterior"];
    let n = u32::try_from(case.columns().len()).unwrap() as usize;
    let weights = floats(&spec["weights"]);
    let (kind, masks): (GraphPosteriorAtomKind, Vec<u64>) = match spec["kind"].as_str().unwrap() {
        kind @ ("dag" | "admg") => {
            let masks = spec["edges"]
                .as_array()
                .unwrap()
                .iter()
                .map(|edges| {
                    edges.as_array().unwrap().iter().fold(0_u64, |mask, edge| {
                        set_edge(
                            mask,
                            n,
                            usize::try_from(edge[0].as_u64().unwrap()).unwrap(),
                            usize::try_from(edge[1].as_u64().unwrap()).unwrap(),
                            true,
                        )
                    })
                })
                .collect();
            (
                if kind == "dag" {
                    GraphPosteriorAtomKind::Dag
                } else {
                    GraphPosteriorAtomKind::Admg
                },
                masks,
            )
        }
        "cpdag" => {
            let masks = spec["atoms"]
                .as_array()
                .unwrap()
                .iter()
                .map(|atom| {
                    let mut cpdag = Cpdag::with_variables(u32::try_from(n).unwrap());
                    for (a, b) in pairs(atom, "directed") {
                        cpdag.insert_directed(node(a), node(b)).unwrap();
                    }
                    for (a, b) in pairs(atom, "undirected") {
                        cpdag.insert_undirected(node(a), node(b)).unwrap();
                    }
                    adjacency_mask_from_cpdag(&cpdag).unwrap()
                })
                .collect();
            (GraphPosteriorAtomKind::Cpdag, masks)
        }
        "pag" => {
            let edges = |atom: &Value, key: &str| {
                pairs(atom, key).fold(0_u64, |mask, (a, b)| {
                    set_edge(
                        mask,
                        n,
                        usize::try_from(a).unwrap(),
                        usize::try_from(b).unwrap(),
                        true,
                    )
                })
            };
            let atoms = spec["atoms"].as_array().unwrap();
            let adjacency: Vec<u64> = atoms.iter().map(|atom| edges(atom, "edges")).collect();
            let circles: Vec<u64> = atoms.iter().map(|atom| edges(atom, "circles")).collect();
            return finish_posterior(
                case,
                n,
                weights,
                adjacency,
                GraphPosteriorAtomKind::Pag,
                None,
                Some(circles),
            );
        }
        "temporal_pag" => {
            let atoms = spec["atoms"].as_array().unwrap();
            let edges = |atom: &Value, key: &str| {
                pairs(atom, key).fold(0_u64, |mask, (a, b)| {
                    set_edge(
                        mask,
                        n,
                        usize::try_from(a).unwrap(),
                        usize::try_from(b).unwrap(),
                        true,
                    )
                })
            };
            let lag_edges: Vec<u64> = atoms
                .iter()
                .map(|atom| {
                    pairs(atom, "lag_edges").fold(0_u64, |mask, (a, b)| {
                        mask | (1_u64
                            << (usize::try_from(a).unwrap() * n + usize::try_from(b).unwrap()))
                    })
                })
                .collect();
            let adjacency: Vec<u64> = atoms.iter().map(|atom| edges(atom, "edges")).collect();
            let circles: Vec<u64> = atoms.iter().map(|atom| edges(atom, "circles")).collect();
            return finish_posterior(
                case,
                n,
                weights,
                adjacency,
                GraphPosteriorAtomKind::Pag,
                Some(lag_edges),
                Some(circles),
            );
        }
        kind @ ("temporal_dag" | "temporal_cpdag") => {
            let atoms = spec["atoms"].as_array().unwrap();
            let lag_edges: Vec<u64> = atoms
                .iter()
                .map(|atom| {
                    pairs(atom, "lag_edges").fold(0_u64, |mask, (a, b)| {
                        mask | (1_u64
                            << (usize::try_from(a).unwrap() * n + usize::try_from(b).unwrap()))
                    })
                })
                .collect();
            let adjacency: Vec<u64> = atoms
                .iter()
                .map(|atom| {
                    let mut cpdag = Cpdag::with_variables(u32::try_from(n).unwrap());
                    for (a, b) in pairs(atom, "directed") {
                        cpdag.insert_directed(node(a), node(b)).unwrap();
                    }
                    for (a, b) in pairs(atom, "undirected") {
                        cpdag.insert_undirected(node(a), node(b)).unwrap();
                    }
                    adjacency_mask_from_cpdag(&cpdag).unwrap()
                })
                .collect();
            let atom_kind = if kind == "temporal_dag" {
                GraphPosteriorAtomKind::Dag
            } else {
                GraphPosteriorAtomKind::Cpdag
            };
            return finish_posterior(case, n, weights, adjacency, atom_kind, Some(lag_edges), None);
        }
        "temporal_admg" => {
            let atoms = spec["atoms"].as_array().unwrap();
            let lagged = |atom: &Value, key: &str| -> LagMask {
                pairs(atom, key).fold(LagMask::EMPTY, |mask, (a, b)| {
                    with_lagged_edge(
                        mask,
                        n,
                        1,
                        usize::try_from(a).unwrap(),
                        usize::try_from(b).unwrap(),
                        true,
                    )
                    .unwrap()
                })
            };
            let adjacency: Vec<u64> = atoms
                .iter()
                .map(|atom| {
                    pairs(atom, "directed").fold(0_u64, |mask, (a, b)| {
                        set_edge(
                            mask,
                            n,
                            usize::try_from(a).unwrap(),
                            usize::try_from(b).unwrap(),
                            true,
                        )
                    })
                })
                .collect();
            let lag_edges: Vec<LagMask> =
                atoms.iter().map(|atom| lagged(atom, "lag_edges")).collect();
            let latent: Vec<LagMask> =
                atoms.iter().map(|atom| lagged(atom, "lag_latent")).collect();
            let sampling = SamplingSummary {
                ess: 1.0,
                diagnostics: InferenceDiagnostics::analytic("class_attribution_known_truth"),
                rejected_invalid: 0,
            };
            return GraphPosterior::new(
                n,
                weights,
                adjacency,
                vec![0.0; n * n],
                vec![0.0; n * n],
                sampling,
            )
            .unwrap()
            .with_atom_kind(GraphPosteriorAtomKind::TemporalAdmg)
            .with_lagged_marginals(1, vec![0.0; n * n])
            .unwrap()
            .with_lag_masks(lag_edges)
            .unwrap()
            .with_lag_bidirected_masks(latent)
            .unwrap();
        }
        other => panic!("unknown posterior kind {other}"),
    };
    finish_posterior(case, n, weights, masks, kind, None, None)
}

fn finish_posterior(
    case: &Case<'_>,
    n: usize,
    weights: Vec<f64>,
    adjacency: Vec<u64>,
    kind: GraphPosteriorAtomKind,
    lag_edges: Option<Vec<u64>>,
    circles: Option<Vec<u64>>,
) -> GraphPosterior {
    let _ = case;
    let sampling = SamplingSummary {
        ess: 1.0,
        diagnostics: InferenceDiagnostics::analytic("class_attribution_known_truth"),
        rejected_invalid: 0,
    };
    let mut posterior =
        GraphPosterior::new(n, weights, adjacency, vec![0.0; n * n], vec![0.0; n * n], sampling)
            .unwrap()
            .with_atom_kind(kind);
    if let Some(circles) = circles {
        posterior = posterior.with_mark_masks(circles).unwrap();
    }
    if let Some(lag_edges) = lag_edges {
        posterior = posterior
            .with_lagged_marginals(1, vec![0.0; n * n])
            .unwrap()
            .with_lag_masks(lag_edges)
            .unwrap();
    }
    posterior
}

fn frequentist() -> InferenceMode {
    InferenceMode::Frequentist
}

fn bayesian(draws: usize) -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(draws))
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(13)
}

/// Run the study as the builder does.
fn run(case: &Case<'_>, study: &Study) -> StudyResult {
    let _ = case;
    study.run(&ctx()).unwrap()
}

/// The numbers of a payload against the derivation.
fn verify_payload(case: &Case<'_>, result: &StudyResult) {
    if case.is_anomaly() {
        verify_anomaly(case, result);
    } else {
        verify_change(case, result);
    }
    verify_mixture(case, result);
}

fn verify_anomaly(case: &Case<'_>, result: &StudyResult) {
    let merged = case.merged();
    let scores = &result.anomaly().expect("anomaly scores")[0];
    let rows = case.rows();
    assert_eq!(scores.rows.len(), rows, "{}: every row is scored", case.name);
    let first = case.derived()["members"]
        .as_array()
        .unwrap()
        .iter()
        .find(|m| !m.is_null())
        .expect("an identified member");
    let want_scores = floats(&first["scores"]);
    let want_components: Vec<ComponentId> = merged["components"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| component(case.component_id(c)))
        .collect();
    let mut held: Vec<ComponentId> = scores.noise_components.to_vec();
    held.sort();
    let mut wanted = want_components.clone();
    wanted.sort();
    assert_eq!(held, wanted, "{}: the union of the members' players", case.name);
    let width = scores.noise_components.len();
    let baselines = floats(&merged["baseline"]);
    for row in 0..rows {
        close(scores.scores[row], want_scores[row], 1e-9, &format!("{} score {row}", case.name));
        close(scores.baseline_scores[row], baselines[row], 1e-9, "baseline");
        let mut total = 0.0;
        for (column, id) in scores.noise_components.iter().enumerate() {
            let position = want_components.iter().position(|c| c == id).unwrap();
            let want = merged["contributions"][row][position].as_f64().unwrap();
            let got = scores.noise_contributions[row * width + column];
            close(got, want, 1e-9, &format!("{} row {row} component {id:?}", case.name));
            total += got;
        }
        close(total, scores.scores[row] - scores.baseline_scores[row], 1e-9, "efficiency");
    }
}

fn verify_change(case: &Case<'_>, result: &StudyResult) {
    let merged = case.merged();
    let change = result.change_attribution().expect("change attribution");
    close(
        change.total_change,
        merged["total"].as_f64().unwrap(),
        merged["total_tolerance"].as_f64().unwrap(),
        &format!("{} total", case.name),
    );
    if let Some(observed) = change.observed_change {
        close(observed, merged["observed"].as_f64().unwrap(), 1e-9, "observed change");
    }
    let components: Vec<ComponentId> = merged["components"]
        .as_array()
        .unwrap()
        .iter()
        .map(|c| component(case.component_id(c)))
        .collect();
    let mut held: Vec<ComponentId> = change.contributions.iter().map(|c| c.component).collect();
    held.sort();
    let mut wanted = components.clone();
    wanted.sort();
    assert_eq!(held, wanted, "{}: the union of the members' players", case.name);
    for (position, id) in components.iter().enumerate() {
        let got = change.contributions.iter().find(|c| c.component == *id).unwrap().contribution;
        close(
            got,
            merged["contributions"][position].as_f64().unwrap(),
            merged["tolerance"][position].as_f64().unwrap(),
            &format!("{} component {id:?}", case.name),
        );
    }
    close(change.contribution_sum(), change.total_change, 1e-6, "efficiency");
    // The graph sensitivity is the standard deviation of the identified members' contributions.
    let member_values = merged["member_contributions"].as_array().unwrap();
    if let Some(spread) = change.graph_sensitivity.as_ref() {
        assert_eq!(spread.len(), change.contributions.len());
        for (column, contribution) in change.contributions.iter().enumerate() {
            let position = components.iter().position(|c| *c == contribution.component).unwrap();
            let values: Vec<f64> =
                member_values.iter().map(|m| m[position].as_f64().unwrap()).collect();
            let weights: Vec<f64> = case
                .members()
                .iter()
                .filter(|m| m["status"] == "identified")
                .map(|m| m["weight"].as_f64().unwrap())
                .collect();
            let total: f64 = weights.iter().sum();
            let mean = values.iter().zip(&weights).map(|(v, w)| v * w / total).sum::<f64>();
            let sd = values
                .iter()
                .zip(&weights)
                .map(|(v, w)| w / total * (v - mean).powi(2))
                .sum::<f64>()
                .sqrt();
            close(
                spread[column],
                sd,
                2.0 * merged["tolerance"][position].as_f64().unwrap(),
                &format!("{} graph sensitivity", case.name),
            );
        }
    } else {
        assert_eq!(member_values.len(), 1, "{}: members that differ state their spread", case.name);
    }
}

/// The completion envelope, the identified set and the masses of a class.
fn verify_mixture(case: &Case<'_>, result: &StudyResult) {
    let merged = case.merged();
    let members = case.members();
    let Some(mixture) = result.structural_response.as_ref() else {
        assert_eq!(
            members.len(),
            1,
            "{}: a class of several members publishes its mixture",
            case.name
        );
        return;
    };
    let outside = usize::from(case.value["outside_weight"].as_f64().unwrap_or(0.0) > 0.0);
    assert_eq!(mixture.atoms.len(), members.len() + outside, "{}: one atom per member", case.name);
    close(
        mixture.identified_mass,
        merged["identified_mass"].as_f64().unwrap(),
        1e-12,
        "identified mass",
    );
    close(
        mixture.unidentified_mass,
        merged["unidentified_mass"].as_f64().unwrap(),
        1e-12,
        "unidentified mass",
    );
    close(
        mixture.unevaluable_mass,
        merged["unevaluable_mass"].as_f64().unwrap(),
        1e-12,
        "unevaluable mass",
    );
    let failed = case.members().iter().filter(|m| m["status"] == "failed").count();
    assert_eq!(
        mixture.atoms.iter().filter(|a| a.status == IdentificationStatus::NotIdentified).count(),
        case.hedged() + outside,
        "{}: the hedged members are the unidentified atoms",
        case.name
    );
    let want_status = if case.hedged() + outside + failed == 0 {
        IdentificationStatus::IdentifiedUnderParametricRestrictions
    } else if case.value["posterior"].is_null() {
        IdentificationStatus::PartiallyIdentified
    } else {
        IdentificationStatus::GraphDependent
    };
    assert_eq!(result.identification.status, want_status, "{}", case.name);
    let set = mixture.identified_set.as_ref().expect("the identified set");
    let tolerance = |position: usize| {
        if case.is_anomaly() {
            1e-9
        } else {
            let slack = merged["tolerance"][position].as_f64().unwrap();
            2.0 * slack
        }
    };
    for (position, name) in merged["components"].as_array().unwrap().iter().enumerate() {
        let id = f64::from(u32::try_from(case.component_id(name)).unwrap());
        let column = set
            .grid
            .iter()
            .position(|g| (*g - id).abs() < 1e-12)
            .unwrap_or_else(|| panic!("{}: no identified set for {name}", case.name));
        close(
            set.lower[column],
            merged["lower"][position].as_f64().unwrap(),
            tolerance(position),
            "lower",
        );
        close(
            set.upper[column],
            merged["upper"][position].as_f64().unwrap(),
            tolerance(position),
            "upper",
        );
    }
    let envelope = result
        .diagnostics
        .iter()
        .find(|d| d.code.as_ref() == "attribution.class.envelope")
        .expect("the completion envelope diagnostic");
    let field = |key: &str| -> f64 {
        envelope
            .fields
            .iter()
            .find(|(name, _)| name.as_ref() == key)
            .unwrap_or_else(|| panic!("envelope field {key}"))
            .1
            .parse()
            .unwrap()
    };
    close(
        field("identified_mass"),
        merged["identified_mass"].as_f64().unwrap(),
        1e-12,
        "envelope mass",
    );
    close(
        field("unidentified_mass"),
        merged["unidentified_mass"].as_f64().unwrap(),
        1e-12,
        "envelope mass",
    );
    assert!(
        result.diagnostics.iter().any(|d| d.code.as_ref() == "attribution.class.identified_set"),
        "{}: the identified set is stated",
        case.name
    );
}

/// The Bayesian-bootstrap posterior of the mean anomaly score, or of the total change, against
/// its closed-form Dirichlet moments.
fn verify_posterior(case: &Case<'_>, posterior: &CausalPosterior, draws: usize) {
    let moments = &case.merged()["bayes"];
    let (mean, sd) = (moments["mean"].as_f64().unwrap(), moments["sd"].as_f64().unwrap());
    let slack =
        if case.is_anomaly() { 0.0 } else { case.merged()["total_tolerance"].as_f64().unwrap() };
    close(
        posterior.summaries.mean[0],
        mean,
        5.0 * sd / (draws as f64).sqrt() + slack,
        &format!("{} posterior mean", case.name),
    );
    let got = posterior.summaries.sd[0];
    assert!(
        got > 0.75 * sd && got < 1.25 * sd,
        "{}: posterior sd {got} against the Dirichlet sd {sd}",
        case.name
    );
}

/// What a class publishes as a posterior: the members' common draws when they all publish the
/// very same ones, and otherwise nothing mixed, each member's posterior staying on its atom.
fn verify_class_posterior(case: &Case<'_>, result: &StudyResult, draws: usize) {
    if let Some(posterior) = result.posterior.as_ref() {
        verify_posterior(case, posterior, draws);
        // The credible interval of a published posterior is its quantile interval.
        assert_eq!(
            result.interval().as_ref().map(|binding| binding.method),
            Some(IntervalMethod::PosteriorQuantile),
            "{}",
            case.name
        );
        return;
    }
    let mixture = result.structural_response.as_ref().expect("atoms keep their posteriors");
    let mut seen = 0;
    for (atom, member) in mixture.atoms.iter().zip(case.members()) {
        if member["status"] == "identified" {
            verify_posterior(case, atom.posterior.as_ref().expect("the atom's posterior"), draws);
            seen += 1;
        } else {
            assert!(atom.posterior.is_none(), "{}: a refused member has no posterior", case.name);
        }
    }
    assert!(seen >= 2, "{}: no posterior mixture of enumeration weights", case.name);
}

fn assert_licensed(status: Option<CellStatus>, name: &str) {
    assert_eq!(status, Some(CellStatus::Licensed), "{name}");
}

const ANOMALY_CASES: [&str; 9] = [
    "cpdag_pair_anomaly",
    "cpdag_chain_anomaly",
    "cpdag_agreeing_anomaly",
    "pag_circle_arrow_anomaly",
    "pag_circle_circle_anomaly",
    "admg_collider_anomaly",
    "temporal_cpdag_anomaly",
    "temporal_admg_anomaly",
    "temporal_pag_anomaly",
];

const POSTERIOR_CASES: [&str; 18] = [
    "posterior_dag_anomaly",
    "posterior_admg_anomaly",
    "posterior_cpdag_anomaly",
    "posterior_dag_change",
    "posterior_admg_change",
    "posterior_cpdag_change",
    "posterior_dag_unevaluable_anomaly",
    "posterior_dag_unevaluable_change",
    "posterior_pag_anomaly",
    "posterior_pag_change",
    "posterior_temporal_dag_anomaly",
    "posterior_temporal_dag_change",
    "posterior_temporal_cpdag_anomaly",
    "posterior_temporal_cpdag_change",
    "posterior_temporal_pag_anomaly",
    "posterior_temporal_pag_change",
    "posterior_temporal_admg_anomaly",
    "posterior_temporal_admg_change",
];

const CHANGE_CASES: [&str; 7] = [
    "cpdag_pair_change",
    "cpdag_agreeing_change",
    "pag_circle_arrow_change",
    "admg_collider_change",
    "temporal_cpdag_change",
    "temporal_pag_change",
    "temporal_admg_change",
];

fn known_truth(names: &[&str]) {
    let document = fixture();
    for name in names {
        let case = Case::of(&document, name);
        for accepted in [false, true] {
            let study = study_of(&case, accepted, &frequentist(), RefuteSuite::None);
            assert_licensed(study.support_status(), name);
            let result = run(&case, &study);
            assert_licensed(result.support_status(), name);
            verify_payload(&case, &result);
            // An accepted structure is attributed exactly as a supplied one.
            if accepted {
                let supplied =
                    run(&case, &study_of(&case, false, &frequentist(), RefuteSuite::None));
                assert_eq!(result.anomaly(), supplied.anomaly(), "{name}");
            }
        }
        let draws = if case.is_anomaly() { 400 } else { 150 };
        for accepted in [false, true] {
            let study = study_of(&case, accepted, &bayesian(draws), RefuteSuite::None);
            assert_licensed(study.support_status(), name);
            let result = run(&case, &study);
            // The Bayesian-bootstrap draws move the posterior, not the point attribution.
            verify_payload(&case, &result);
            verify_class_posterior(&case, &result, draws);
        }
    }
}

#[test]
fn class_anomaly_attribution_known_truth() {
    on_large_stack(|| known_truth(&ANOMALY_CASES));
}

#[test]
fn class_change_attribution_known_truth() {
    on_large_stack(|| known_truth(&CHANGE_CASES));
}

#[test]
fn graph_posterior_attribution_known_truth() {
    on_large_stack(|| {
        let document = fixture();
        for name in POSTERIOR_CASES {
            let case = Case::of(&document, name);
            let study = study_of(&case, false, &frequentist(), RefuteSuite::None);
            assert_licensed(study.support_status(), name);
            let result = run(&case, &study);
            assert_licensed(result.support_status(), name);
            verify_payload(&case, &result);
            let mixture = result.structural_response.as_ref().unwrap();
            assert_eq!(
                mixture.weight_basis,
                antecedent::StructuralWeightBasis::PosteriorProbability
            );
            // The weights of the atoms are the posterior probabilities, a class atom's shared.
            for (atom, member) in mixture.atoms.iter().zip(case.members()) {
                close(atom.weight, member["weight"].as_f64().unwrap(), 1e-12, "atom weight");
            }
            let draws = if case.is_anomaly() { 400 } else { 150 };
            let study = study_of(&case, false, &bayesian(draws), RefuteSuite::None);
            let result = run(&case, &study);
            verify_payload(&case, &result);
            verify_class_posterior(&case, &result, draws);
        }
    });
}

/// Run the retained plan with the builder gone.
fn retained_run(case: &Case<'_>, accepted: bool, inference: &InferenceMode) -> StudyResult {
    let builder = study_of(case, accepted, inference, RefuteSuite::None);
    let prepared = builder.prepare(&ctx()).unwrap();
    assert!(
        matches!(prepared.retained_operation(), RetainedOperation::ClassAttribution),
        "{}: a sealed class attribution",
        case.name
    );
    drop(builder);
    match data_of(case) {
        Data::Table(table) => prepared.estimate(&table, &ctx()).unwrap(),
        Data::Series(series) => prepared.estimate_series(&series, &ctx()).unwrap(),
    }
}

#[test]
fn class_attribution_executes_from_retained_plan_after_builder_is_dropped() {
    on_large_stack(|| {
        let document = fixture();
        for name in ANOMALY_CASES.iter().chain(CHANGE_CASES.iter()) {
            let case = Case::of(&document, name);
            for accepted in [false, true] {
                let result = retained_run(&case, accepted, &frequentist());
                assert_licensed(result.support_status(), name);
                verify_payload(&case, &result);
            }
            let result = retained_run(&case, false, &bayesian(24));
            assert_licensed(result.support_status(), name);
            verify_payload(&case, &result);
            // A frequentist class attribution is a point value: it reports no interval.
            let point = retained_run(&case, false, &frequentist());
            assert_eq!(
                point.interval().as_ref().map(|binding| binding.method),
                Some(IntervalMethod::None),
                "{name}"
            );
        }
    });
}

/// A graph posterior run from the retained plan with the builder gone.
fn retained_posterior_run(case: &Case<'_>, inference: &InferenceMode) -> StudyResult {
    let builder = study_of(case, false, inference, RefuteSuite::None);
    let prepared = builder.prepare(&ctx()).unwrap();
    // The plan carries the lowered program of the posterior attribution.
    let program = prepared.plan().lowered_program_identity();
    drop(builder);
    let result = match data_of(case) {
        Data::Table(table) => prepared.estimate(&table, &ctx()).unwrap(),
        Data::Series(series) => prepared.estimate_series(&series, &ctx()).unwrap(),
    };
    assert_eq!(result.lowered_program_identity(), program, "{}", case.name);
    result
}

#[test]
fn class_graph_posterior_attribution_executes_from_retained_plan_after_builder_is_dropped() {
    on_large_stack(|| {
        let document = fixture();
        for name in POSTERIOR_CASES {
            let case = Case::of(&document, name);
            let result = retained_posterior_run(&case, &frequentist());
            assert_licensed(result.support_status(), name);
            verify_payload(&case, &result);
            let result = retained_posterior_run(&case, &bayesian(24));
            assert_licensed(result.support_status(), name);
            verify_payload(&case, &result);
        }
    });
}

#[test]
fn a_hedged_class_attribution_is_refused_with_the_witness_and_the_masses_are_stated() {
    on_large_stack(|| {
        let document = fixture();
        // A latent common cause of a mechanism and its parent: no regression identifies it.
        let data = Case::of(&document, "cpdag_pair_anomaly");
        let columns = data.columns();
        let named: Vec<(&str, &[f64])> =
            columns.iter().map(|(n, v)| (n.as_str(), v.as_slice())).collect();
        let table = TabularData::from_f64_columns(named).unwrap();
        let mut hedge = Admg::with_variables(2);
        hedge.insert_directed(node(0), node(1)).unwrap();
        hedge.insert_bidirected(node(0), node(1)).unwrap();
        let queries = [
            data.query(),
            CausalQuery::ChangeAttribution(ChangeAttributionQuery::new(
                variable(1),
                PopulationSelector::TimeRange { start: 0, end: 20 },
                PopulationSelector::TimeRange { start: 20, end: 40 },
            )),
        ];
        for query in queries {
            let study = Study::tabular(table.clone())
                .graph(hedge.clone())
                .query(query)
                .refute(RefuteSuite::None)
                .build()
                .unwrap();
            assert_licensed(study.support_status(), "hedge");
            let error = study.run(&ctx()).unwrap_err();
            assert!(matches!(error, CausalError::NotIdentified { .. }), "{error}");
            let text = error.to_string();
            assert!(text.contains(HEDGE_WITNESS_ID), "{text}");
            assert!(text.contains("latent_block_witness.rs"), "{text}");
        }
        // A PAG whose only completion is the invisible directed edge x -> y (read as possibly
        // confounded) has no identified member: the class is refused, not published empty.
        let mut pag = Pag::with_variables(2);
        pag.insert_marked(MarkedEdge {
            a: node(0),
            b: node(1),
            at_a: Endpoint::Tail,
            at_b: Endpoint::Arrow,
            middle: MiddleMark::Empty,
        })
        .unwrap();
        let study = Study::tabular(table)
            .graph(pag)
            .query(data.query())
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        let error = study.run(&ctx()).unwrap_err();
        assert!(matches!(error, CausalError::NotIdentified { .. }), "{error}");
        // The temporal analogue: a lagged cause and its effect sharing a latent cause.
        let series = Case::of(&document, "temporal_cpdag_anomaly");
        let columns = series.columns();
        let named: Vec<(&str, &[f64])> =
            columns.iter().map(|(n, v)| (n.as_str(), v.as_slice())).collect();
        let series_data = TimeSeriesData::from_f64_columns(named, 1).unwrap();
        let mut template = TemporalAdmg::empty();
        let cause = template.add_lagged(variable(0), Lag::from_raw(1)).unwrap();
        let effect = template.add_lagged(variable(1), Lag::CONTEMPORANEOUS).unwrap();
        template.insert_directed(cause, effect).unwrap();
        template.insert_bidirected(cause, effect).unwrap();
        let study = Study::series(series_data)
            .graph(template)
            .query(series.query())
            .refute(RefuteSuite::None)
            .build()
            .unwrap();
        let error = study.run(&ctx()).unwrap_err();
        assert!(matches!(error, CausalError::NotIdentified { .. }), "{error}");
        assert!(error.to_string().contains(HEDGE_WITNESS_ID));
    });
}

// ---------------------------------------------------------------------------------------------
// Refuter states: the whole attribution vector refitted on subsets of the rows and on resamples.

fn refuter_fixture() -> Value {
    serde_json::from_str(include_str!(
        "../../../conformance/estimate/class_attribution_refuter_states/expected.json"
    ))
    .expect("class_attribution_refuter_states fixture parses")
}

/// A family of the refuter states: the table, the target and the structure it is attributed under.
struct Family {
    name: &'static str,
    table: &'static str,
    target: &'static str,
    series: bool,
    structure: Value,
    posterior: bool,
}

fn families() -> Vec<Family> {
    use serde_json::json;
    let family = |name, table, target, series, structure: Value, posterior| Family {
        name,
        table,
        target,
        series,
        structure,
        posterior,
    };
    vec![
        family(
            "cpdag",
            "xy",
            "y",
            false,
            json!({"class": "cpdag", "nodes": 2, "directed": [], "undirected": [[0, 1]]}),
            false,
        ),
        family(
            "pag",
            "xy",
            "y",
            false,
            json!({"class": "pag", "nodes": 2, "marked": [[0, 1, "circle", "arrow"]]}),
            false,
        ),
        family(
            "admg",
            "abt",
            "t",
            false,
            json!({"class": "admg", "nodes": 3, "directed": [[0, 2], [1, 2]], "bidirected": [[0, 1]]}),
            false,
        ),
        family(
            "temporal_cpdag",
            "series_xy",
            "y",
            true,
            json!({"class": "temporal_cpdag", "nodes": [[0, 1], [0, 0], [1, 0]], "directed": [[0, 1], [0, 2]], "undirected": [[1, 2]]}),
            false,
        ),
        family(
            "temporal_pag",
            "series_xy",
            "y",
            true,
            json!({"class": "temporal_pag", "nodes": [[0, 1], [1, 0]], "marked": [[0, 1, "circle", "arrow"]]}),
            false,
        ),
        family(
            "temporal_admg",
            "series_abt",
            "t",
            true,
            json!({"class": "temporal_admg", "nodes": [[0, 1], [1, 1], [2, 0]], "directed": [[0, 2], [1, 2]], "bidirected": [[0, 1]]}),
            false,
        ),
        family(
            "posterior_dag",
            "xy",
            "y",
            false,
            json!({"kind": "dag", "weights": [0.6, 0.4], "edges": [[[0, 1]], [[1, 0]]]}),
            true,
        ),
        family(
            "posterior_admg",
            "abt",
            "t",
            false,
            json!({"kind": "admg", "weights": [0.6, 0.4], "edges": [[[0, 2], [1, 2], [0, 1], [1, 0]], [[0, 2], [1, 2]]]}),
            true,
        ),
        family(
            "posterior_cpdag",
            "xy",
            "y",
            false,
            json!({"kind": "cpdag", "weights": [0.6, 0.4], "atoms": [{"undirected": [[0, 1]], "directed": []}, {"undirected": [], "directed": [[0, 1]]}]}),
            true,
        ),
        family(
            "posterior_pag",
            "xy",
            "y",
            false,
            json!({"kind": "pag", "weights": [0.6, 0.4], "atoms": [{"edges": [[0, 1]], "circles": [[0, 1]]}, {"edges": [[0, 1], [1, 0]], "circles": []}]}),
            true,
        ),
        family(
            "posterior_temporal_dag",
            "series_xy",
            "y",
            true,
            json!({"kind": "temporal_dag", "weights": [0.7, 0.3], "atoms": [{"lag_edges": [[0, 1]]}, {"lag_edges": []}]}),
            true,
        ),
        family(
            "posterior_temporal_cpdag",
            "series_xy",
            "y",
            true,
            json!({"kind": "temporal_cpdag", "weights": [0.6, 0.4], "atoms": [
                {"undirected": [[0, 1]], "directed": [], "lag_edges": [[0, 0], [0, 1]]},
                {"undirected": [], "directed": [[0, 1]], "lag_edges": [[0, 0], [0, 1]]}]}),
            true,
        ),
        family(
            "posterior_temporal_pag",
            "series_xy",
            "y",
            true,
            json!({"kind": "temporal_pag", "weights": [0.6, 0.4], "atoms": [
                {"edges": [[0, 1]], "circles": [[0, 1]], "lag_edges": [[0, 1]]},
                {"edges": [[0, 1], [1, 0]], "circles": [], "lag_edges": [[0, 1]]}]}),
            true,
        ),
        family(
            "posterior_temporal_admg",
            "series_abt",
            "t",
            true,
            json!({"kind": "temporal_admg", "weights": [0.6, 0.4], "atoms": [
                {"directed": [], "lag_edges": [[0, 2], [1, 2]], "lag_latent": []},
                {"directed": [], "lag_edges": [[0, 2]], "lag_latent": []}]}),
            true,
        ),
    ]
}

/// The case of `family` on the clean or the planted variant of its table.
fn refuter_case(tables: &Value, family: &Family, planted: bool) -> Value {
    use serde_json::json;
    let columns = &tables["tables"][family.table][if planted { "planted" } else { "clean" }];
    let order: Vec<&str> = columns.as_object().unwrap().keys().map(String::as_str).collect();
    let order = match family.table {
        "xy" | "series_xy" => vec!["x", "y"],
        "abt" | "series_abt" => vec!["a", "b", "t"],
        _ => order,
    };
    let mut value = json!({
        "kind": "anomaly",
        "columns": columns,
        "column_order": order,
        "series": family.series,
        "target": family.target,
        "reference": {"kind": "empirical"},
    });
    if family.posterior {
        value["posterior"] = family.structure.clone();
        value["graph"] = json!({"class": "posterior"});
    } else {
        value["graph"] = family.structure.clone();
    }
    value
}

fn refutation<'a>(
    result: &'a StudyResult,
    refuter: &str,
) -> &'a antecedent_validate::RefutationReport {
    result
        .refutations
        .iter()
        .find(|report| report.refuter.as_ref() == refuter)
        .unwrap_or_else(|| panic!("no `{refuter}` in {:?}", result.refutations))
}

fn refuter_states(names: &[&str]) {
    let tables = refuter_fixture();
    let expected = &tables["expected_verdicts"];
    assert!(expected["clean_all_refuters_pass"].as_bool().unwrap());
    assert!(!expected["planted_subset_stability_passes"].as_bool().unwrap());
    for family in families().iter().filter(|f| names.contains(&f.name)) {
        let clean_value = refuter_case(&tables, family, false);
        let planted_value = refuter_case(&tables, family, true);
        let clean_case = Case { name: family.name, value: &clean_value };
        let planted_case = Case { name: family.name, value: &planted_value };
        for accepted in if family.posterior { vec![false] } else { vec![false, true] } {
            for inference in [frequentist(), bayesian(if family.series { 8 } else { 24 })] {
                for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                    let full = matches!(suite, RefuteSuite::Full);
                    let tag = format!(
                        "{} accepted={accepted} bayesian={} full={full}",
                        family.name,
                        matches!(inference, InferenceMode::Bayesian(_))
                    );
                    let clean_study = study_of(&clean_case, accepted, &inference, suite);
                    assert_licensed(clean_study.support_status(), &tag);
                    let clean = run(&clean_case, &clean_study);
                    let planted_study = study_of(&planted_case, accepted, &inference, suite);
                    assert_licensed(planted_study.support_status(), &tag);
                    let planted = run(&planted_case, &planted_study);
                    assert!(
                        clean.refutations.iter().all(|report| report.passed),
                        "clean table passes every refuter, {tag}: {:?}",
                        clean.refutations
                    );
                    assert!(
                        !refutation(&planted, "target.subset_stability").passed,
                        "planted table fails subset stability, {tag}"
                    );
                    assert_eq!(
                        clean
                            .refutations
                            .iter()
                            .any(|r| r.refuter.as_ref() == "target.bootstrap_stability"),
                        full,
                        "{tag}"
                    );
                }
            }
        }
    }
}

#[test]
fn class_anomaly_attribution_refuter_states_known_truth() {
    on_large_stack(|| {
        refuter_states(&[
            "cpdag",
            "pag",
            "admg",
            "temporal_cpdag",
            "temporal_pag",
            "temporal_admg",
        ]);
    });
}

#[test]
fn graph_posterior_anomaly_attribution_refuter_states_known_truth() {
    on_large_stack(|| {
        refuter_states(&[
            "posterior_dag",
            "posterior_admg",
            "posterior_cpdag",
            "posterior_pag",
            "posterior_temporal_dag",
            "posterior_temporal_cpdag",
            "posterior_temporal_pag",
            "posterior_temporal_admg",
        ]);
    });
}

#[test]
fn a_change_attribution_validation_is_refused_for_classes_and_posteriors() {
    on_large_stack(|| {
        let document = fixture();
        for name in [
            "cpdag_pair_change",
            "pag_circle_arrow_change",
            "admg_collider_change",
            "temporal_cpdag_change",
            "temporal_pag_change",
            "temporal_admg_change",
            "posterior_dag_change",
            "posterior_temporal_dag_change",
        ] {
            let case = Case::of(&document, name);
            for accepted in [false, true] {
                for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                    let message = match try_study_of(&case, accepted, &frequentist(), suite) {
                        Err(error) => error.to_string(),
                        Ok(study) => {
                            assert_eq!(study.support_status(), Some(CellStatus::Refused), "{name}");
                            study.run(&ctx()).unwrap_err().to_string()
                        }
                    };
                    assert!(
                        message.contains("Cheap and full validation of a change attribution"),
                        "{name}: {message}"
                    );
                }
            }
        }
    });
}
