//! Cheap and full validation of a graph-posterior `ResponseCurve` over `Admg`, `TemporalAdmg`,
//! `Cpdag` and `Pag` atoms against the identified curves of exact or stated laws.
//!
//! The fixture `conformance/estimate/response_curve_posterior_refuter_states` derives the
//! front-door curve `[0.314, 0.596]` of an exact table (and its mean outcome 0.386, the level of
//! the flat curve a permuted treatment must give), the Cpdag curve `a` and the PAG curve
//! `1 + 2 a`. The functional refuter state refits the whole graph-posterior procedure on
//! perturbed data (placebo treatment: the refitted curve must be flat; placebo outcome; subset
//! and bootstrap stability under Frequentist inference; prior sensitivity and the posterior
//! predictive check under Bayesian inference) and reports the support of the curve over the
//! posterior atoms (`curve.atom_support`): the identified mass, which must equal the posterior
//! weight of the atoms the identifier accepts.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent::{BayesianConfig, CellStatus, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_discovery::adjacency_mask::{
    AdjacencyMask, adjacency_mask_from_admg, adjacency_mask_from_cpdag, adjacency_masks_from_pag,
    set_edge,
};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
use antecedent_graph::{Admg, Cpdag, DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag};
use antecedent_prob::InferenceDiagnostics;

const FIXTURE: &str = include_str!(
    "../../../conformance/estimate/response_curve_posterior_refuter_states/expected.json"
);
const FRONTDOOR: &str =
    include_str!("../../../conformance/estimate/admg_frontdoor_functional/expected.json");
const WEIGHTS: [f64; 2] = [0.65, 0.35];

struct Truth {
    document: serde_json::Value,
    tolerance_exact: f64,
    tolerance_bayes: f64,
    tolerance_sampled: f64,
    null_level: f64,
}

fn truth() -> Truth {
    let document: serde_json::Value = serde_json::from_str(FIXTURE).unwrap();
    let tolerance = |key: &str| document["tolerance"][key].as_f64().unwrap();
    Truth {
        tolerance_exact: tolerance("exact"),
        tolerance_bayes: tolerance("frontdoor_bayesian"),
        tolerance_sampled: tolerance("sampled_curve"),
        null_level: tolerance("placebo_level"),
        document,
    }
}

struct Stream(u64);

impl Stream {
    fn uniform(&mut self) -> f64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        let bits = self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 11;
        (bits as f64 + 0.5) / (1u64 << 53) as f64
    }

    fn normal(&mut self) -> f64 {
        let (u, v) = (self.uniform(), self.uniform());
        (-2.0 * u.ln()).sqrt() * (std::f64::consts::TAU * v).cos()
    }
}

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn curve(outcome: u32, treatment: u32) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: v(outcome),
        treatment: ContinuousDomain::new(v(treatment), GridSpec::Values(Arc::from([0.0, 1.0]))),
    })
}

/// The front-door table and graph of `conformance/estimate/admg_frontdoor_functional`.
fn frontdoor() -> (TabularData, Admg) {
    let pin: serde_json::Value = serde_json::from_str(FRONTDOOR).unwrap();
    let columns: Vec<&str> =
        pin["columns"].as_array().unwrap().iter().map(|c| c.as_str().unwrap()).collect();
    let mut values = vec![Vec::new(); columns.len()];
    for cell in pin["contingency_table"].as_array().unwrap() {
        let count = usize::try_from(cell["count"].as_u64().unwrap()).unwrap();
        for (index, name) in columns.iter().enumerate() {
            values[index].extend(std::iter::repeat_n(cell[*name].as_f64().unwrap(), count));
        }
    }
    let data = TabularData::from_f64_columns(
        columns.iter().zip(&values).map(|(name, column)| (*name, column.as_slice())),
    )
    .unwrap();
    let node =
        |name: &str| d(u32::try_from(columns.iter().position(|c| *c == name).unwrap()).unwrap());
    let mut graph = Admg::with_variables(u32::try_from(columns.len()).unwrap());
    for edge in pin["graph"]["directed_edges"].as_array().unwrap() {
        graph
            .insert_directed(node(edge[0].as_str().unwrap()), node(edge[1].as_str().unwrap()))
            .unwrap();
    }
    for edge in pin["graph"]["bidirected_edges"].as_array().unwrap() {
        graph
            .insert_bidirected(node(edge[0].as_str().unwrap()), node(edge[1].as_str().unwrap()))
            .unwrap();
    }
    (data, graph)
}

/// Columns `t, y, z, w`: `t = 0.8 z + e`, `y = t + z + e`; the curve is `a` in expectation.
fn cpdag_table() -> TabularData {
    let mut stream = Stream(5);
    let n = 1500;
    let (mut t, mut y, mut z, mut w) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = stream.normal();
        w[i] = stream.normal();
        t[i] = 0.8 * z[i] + stream.normal();
        y[i] = t[i] + z[i] + stream.normal();
    }
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
        ("w", w.as_slice()),
    ])
    .unwrap()
}

fn cpdag() -> Cpdag {
    let mut graph = Cpdag::with_variables(4);
    graph.insert_directed(d(2), d(0)).unwrap();
    graph.insert_directed(d(2), d(1)).unwrap();
    graph.insert_directed(d(0), d(1)).unwrap();
    graph.insert_undirected(d(3), d(2)).unwrap();
    graph
}

/// Columns `t, y, z, r`: `t = 0.5 z + 0.4 r + e`, `y = 1 + 2 t + 0.8 z + e`; the curve is `1 + 2 a`.
fn pag_table() -> TabularData {
    let mut stream = Stream(6);
    let n = 1500;
    let (mut t, mut y, mut z, mut r) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = stream.normal();
        r[i] = stream.normal();
        t[i] = 0.5 * z[i] + 0.4 * r[i] + stream.normal();
        y[i] = 1.0 + 2.0 * t[i] + 0.8 * z[i] + stream.normal();
    }
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
        ("r", r.as_slice()),
    ])
    .unwrap()
}

fn pag() -> Pag {
    let mut graph = Pag::with_variables(4);
    for (a, b, at_a, at_b) in [
        (3, 0, Endpoint::Circle, Endpoint::Arrow),
        (2, 0, Endpoint::Circle, Endpoint::Arrow),
        (2, 1, Endpoint::Tail, Endpoint::Arrow),
        (0, 1, Endpoint::Tail, Endpoint::Arrow),
    ] {
        graph
            .insert_marked(MarkedEdge { a: d(a), b: d(b), at_a, at_b, middle: MiddleMark::Empty })
            .unwrap();
    }
    graph
}

fn posterior(
    n: usize,
    kind: GraphPosteriorAtomKind,
    atoms: [AdjacencyMask; 2],
    marks: Option<[AdjacencyMask; 2]>,
) -> GraphPosterior {
    let posterior = GraphPosterior::from_atoms(
        n,
        WEIGHTS.to_vec(),
        atoms.to_vec(),
        vec![0.0; n * n],
        vec![0.0; n * n],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic(
                "response_curve_posterior_validation_truth",
            ),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(kind);
    match marks {
        Some(marks) => posterior.with_mark_atoms(marks.to_vec()).unwrap(),
        None => posterior,
    }
}

/// A directed cycle is not an ADMG: that atom stays unidentified.
fn cycle(n: usize) -> AdjacencyMask {
    let mut mask = set_edge(0, n, 0, 1, true);
    mask = set_edge(mask, n, 1, 2, true);
    set_edge(mask, n, 2, 0, true)
}

#[derive(Clone, Copy, Debug)]
enum Class {
    Admg,
    TemporalAdmg,
    Cpdag,
    Pag,
}

impl Class {
    fn name(self) -> &'static str {
        match self {
            Self::Admg => "Admg",
            Self::TemporalAdmg => "TemporalAdmg",
            Self::Cpdag => "Cpdag",
            Self::Pag => "Pag",
        }
    }
}

/// The data, the posterior and the curve query of one class; `lost` replaces the second atom of
/// an ADMG-like posterior by a directed cycle the identifier refuses.
fn setup(class: Class, lost: bool) -> (TabularData, GraphPosterior, ResponseQuery) {
    match class {
        Class::Admg | Class::TemporalAdmg => {
            let (data, graph) = frontdoor();
            let kind = if matches!(class, Class::Admg) {
                GraphPosteriorAtomKind::Admg
            } else {
                GraphPosteriorAtomKind::TemporalAdmg
            };
            let key = adjacency_mask_from_admg(&graph).unwrap();
            let second = if lost { cycle(3) } else { key.clone() };
            (data, posterior(3, kind, [key, second], None), curve(2, 0))
        }
        Class::Cpdag => {
            let key = adjacency_mask_from_cpdag(&cpdag()).unwrap();
            (
                cpdag_table(),
                posterior(4, GraphPosteriorAtomKind::Cpdag, [key.clone(), key], None),
                curve(1, 0),
            )
        }
        Class::Pag => {
            let (key, marks) = adjacency_masks_from_pag(&pag()).unwrap();
            (
                pag_table(),
                posterior(
                    4,
                    GraphPosteriorAtomKind::Pag,
                    [key.clone(), key],
                    Some([marks.clone(), marks]),
                ),
                curve(1, 0),
            )
        }
    }
}

fn means(result: &StudyResult) -> Vec<f64> {
    match &result.response().expect("response").estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Surface { mean, .. }) => {
            mean.to_vec()
        }
        // A curve that lost mass to a refused atom is graph-dependent: the identified atoms'
        // one curve is kept apart from the unidentified mass.
        ResponseIdentification::GraphDependent(parts) if parts.len() == 1 => match &parts[0].1 {
            ResponseValue::Surface { mean, .. } => mean.to_vec(),
            other => panic!("expected a surface, got {other:?}"),
        },
        other => panic!("expected a point-identified surface, got {other:?}"),
    }
}

fn inference_name(inference: &InferenceMode) -> &'static str {
    if matches!(inference, InferenceMode::Frequentist) { "Frequentist" } else { "Bayesian" }
}

fn suite_name(suite: RefuteSuite) -> &'static str {
    match suite {
        RefuteSuite::Cheap => "cheap",
        RefuteSuite::Full => "full",
        _ => "none",
    }
}

fn mean_outcome(data: &TabularData, outcome: u32) -> f64 {
    let column = match &data.storage().columns()[outcome as usize] {
        antecedent_data::OwnedColumn::Float64(column) => column.values.to_vec(),
        _ => unreachable!("float columns"),
    };
    column.iter().sum::<f64>() / column.len() as f64
}

/// `ResponseCurve` x `Admg`, `TemporalAdmg`, `Cpdag` and `Pag` x graph_posterior x Frequentist
/// and Bayesian x cheap and full.
#[test]
fn response_curve_graph_posterior_cheap_and_full_match_the_identified_curves() {
    let truth = truth();
    let closed = &truth.document["closed_form"];
    let contract = &truth.document["refuter_contract"];
    let listed = |key: &str, name: &str| {
        contract[key].as_array().unwrap().iter().any(|entry| entry.as_str() == Some(name))
    };
    let context = ExecutionContext::for_tests(21);
    for class in [Class::Admg, Class::TemporalAdmg, Class::Cpdag, Class::Pag] {
        let (data, posterior, query) = setup(class, false);
        let (expected, tolerance_f, tolerance_b): (Vec<f64>, f64, f64) = match class {
            Class::Admg | Class::TemporalAdmg => (
                closed["frontdoor_curve"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
                    .collect(),
                truth.tolerance_exact,
                truth.tolerance_bayes,
            ),
            Class::Cpdag => (
                closed["cpdag_curve"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
                    .collect(),
                truth.tolerance_sampled,
                truth.tolerance_sampled,
            ),
            Class::Pag => (
                closed["pag_curve"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .map(|x| x.as_f64().unwrap())
                    .collect(),
                truth.tolerance_sampled,
                truth.tolerance_sampled,
            ),
        };
        let outcome = if matches!(class, Class::Admg | Class::TemporalAdmg) { 2 } else { 1 };
        let flat = mean_outcome(&data, outcome);
        for inference in [
            InferenceMode::Frequentist,
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(64)),
        ] {
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                let label = format!(
                    "{} graph_posterior {} {}",
                    class.name(),
                    inference_name(&inference),
                    suite_name(suite)
                );
                let result = Study::tabular(data.clone())
                    .graph_posterior(posterior.clone())
                    .query(CausalQuery::Response(query.clone()))
                    .inference(inference.clone())
                    .refute(suite)
                    .bootstrap_replicates(0)
                    .build()
                    .unwrap_or_else(|error| panic!("{label}: {error}"))
                    .run(&context)
                    .unwrap_or_else(|error| panic!("{label}: {error}"));
                assert_eq!(result.support_status(), Some(CellStatus::Licensed), "{label}");
                let tolerance = if matches!(inference, InferenceMode::Frequentist) {
                    tolerance_f
                } else {
                    tolerance_b
                };
                for (actual, want) in means(&result).iter().zip(&expected) {
                    assert!((actual - want).abs() < tolerance, "{label}: curve {actual} vs {want}");
                }
                let key = match (&inference, class) {
                    (InferenceMode::Frequentist, _) => "Frequentist",
                    (_, Class::Admg | Class::TemporalAdmg) => "Bayesian_admg",
                    _ => "Bayesian_class",
                };
                for name in contract[key][suite_name(suite)].as_array().unwrap() {
                    let name = name.as_str().unwrap();
                    assert!(
                        result.refutations.iter().any(|report| report.refuter.as_ref() == name),
                        "{label}: refuter {name} did not run"
                    );
                }
                for report in &result.refutations {
                    let name = report.refuter.as_ref();
                    assert!(report.passed, "{label}: {name} flags the clean law");
                    if listed("flat_expected", name) {
                        assert!(
                            (report.refuted_ate - flat).abs() < truth.null_level,
                            "{label}: {name} must be flat at the mean outcome {flat}, got {}",
                            report.refuted_ate
                        );
                    }
                    if name == "curve.atom_support" {
                        assert!((report.original_ate - 1.0).abs() < 1e-12, "{label}");
                    }
                }
            }
        }
    }
}

/// A posterior that loses mass to an atom the identifier refuses is refuted as the same
/// conditional curve, and the support report states the identified mass.
#[test]
fn response_curve_refuters_report_the_identified_mass_of_a_curve_that_lost_an_atom() {
    let truth = truth();
    let expected: Vec<f64> = truth.document["closed_form"]["frontdoor_curve"]
        .as_array()
        .unwrap()
        .iter()
        .map(|x| x.as_f64().unwrap())
        .collect();
    let context = ExecutionContext::for_tests(22);
    for class in [Class::Admg, Class::TemporalAdmg] {
        let (data, posterior, query) = setup(class, true);
        let result = Study::tabular(data)
            .graph_posterior(posterior)
            .query(CausalQuery::Response(query))
            .refute(RefuteSuite::Cheap)
            .bootstrap_replicates(0)
            .build()
            .unwrap()
            .run(&context)
            .unwrap();
        let mixture = result.structural_response.as_ref().expect("atom mixture");
        assert!((mixture.unidentified_mass - WEIGHTS[1]).abs() < 1e-12, "{}", class.name());
        for (actual, want) in means(&result).iter().zip(&expected) {
            assert!((actual - want).abs() < truth.tolerance_exact, "{}", class.name());
        }
        let support = result
            .refutations
            .iter()
            .find(|report| report.refuter.as_ref() == "curve.atom_support")
            .expect("the support of the curve over the atoms");
        assert!((support.original_ate - WEIGHTS[0]).abs() < 1e-12, "{}", class.name());
        for report in &result.refutations {
            assert!(report.passed, "{}: {}", class.name(), report.refuter);
        }
    }
}

/// The retained plan runs the refuter state of every class with the builder dropped.
#[test]
fn response_curve_graph_posterior_executes_from_the_retained_plan_with_the_builder_dropped() {
    let truth = truth();
    let context = ExecutionContext::for_tests(23);
    for class in [Class::Admg, Class::TemporalAdmg, Class::Cpdag, Class::Pag] {
        let (data, posterior, query) = setup(class, false);
        let builder = Study::tabular(data.clone())
            .graph_posterior(posterior)
            .query(CausalQuery::Response(query))
            .refute(RefuteSuite::Cheap)
            .bootstrap_replicates(0);
        let study = builder.clone().build().unwrap();
        let prepared = study.prepare(&context).unwrap();
        drop(builder);
        drop(study);
        assert!(prepared.plan().logical.record.estimator.is_some(), "{}", class.name());
        let result = prepared
            .estimate(&data, &context)
            .unwrap_or_else(|error| panic!("{}: {error}", class.name()));
        assert!(!result.refutations.is_empty(), "{}", class.name());
        assert!(means(&result).iter().all(|m| m.is_finite()));
        let _ = truth.tolerance_exact;
    }
}
