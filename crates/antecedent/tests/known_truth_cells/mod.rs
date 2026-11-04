//! Shared harness for the static known-truth evidence tests.
//!
//! A coordinate is `Query:Graph:Structure:Inference:Validation`. The harness parses the
//! coordinates a family claims, builds the study that coordinate names (graph class,
//! structure source, inference mode, validation suite) around an explicitly selected
//! estimator, and returns the result for the caller to compare with a closed form.
//!
//! The static design is exactly balanced, so every estimator checked on it is exact and the
//! expected value is derived here by hand from the table, never by a production estimator.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(dead_code, reason = "each test target uses a different part of the harness")]

use std::sync::Arc;

use antecedent::analysis::StudyBuilder;
use antecedent::{AcceptedGraph, EstimatorSpec, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{AverageEffectQuery, CausalQuery, ExecutionContext, ResponseValue};
use antecedent_data::{TableView, TabularData};
use antecedent_discovery::adjacency_mask::{adjacency_mask_from_cpdag, adjacency_masks_from_pag};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, set_edge};
use antecedent_graph::{Cpdag, Dag, DenseNodeId, Pag, TieredBackground, WithinTier};
use antecedent_prob::InferenceDiagnostics;

// ------------------------------------------------------------------------ coordinates

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum GraphKind {
    Dag,
    Cpdag,
    Pag,
    Admg,
    Unknown,
    CoDetermined,
    TemporalAdmg,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Structure {
    Explicit,
    Accepted,
    GraphPosterior,
}

#[derive(Clone, Debug)]
pub struct Coordinate {
    pub text: String,
    pub query: String,
    pub graph: GraphKind,
    pub structure: Structure,
    pub bayesian: bool,
    pub suite: RefuteSuite,
    pub suite_label: String,
}

pub fn parse(text: &str) -> Coordinate {
    let parts: Vec<&str> = text.split(':').collect();
    assert_eq!(parts.len(), 5, "coordinate {text}");
    let graph = match parts[1] {
        "Dag" => GraphKind::Dag,
        "Cpdag" => GraphKind::Cpdag,
        "Pag" => GraphKind::Pag,
        "Admg" => GraphKind::Admg,
        "Unknown" => GraphKind::Unknown,
        "CoDetermined" => GraphKind::CoDetermined,
        "TemporalAdmg" => GraphKind::TemporalAdmg,
        other => panic!("graph class {other} in {text}"),
    };
    let structure = match parts[2] {
        "explicit" => Structure::Explicit,
        "accepted" => Structure::Accepted,
        "graph_posterior" => Structure::GraphPosterior,
        other => panic!("structure {other} in {text}"),
    };
    let bayesian = match parts[3] {
        "Frequentist" => false,
        "Bayesian" => true,
        other => panic!("inference {other} in {text}"),
    };
    let suite = match parts[4] {
        "none" => RefuteSuite::None,
        "cheap" => RefuteSuite::Cheap,
        "full" => RefuteSuite::Full,
        other => panic!("validation {other} in {text}"),
    };
    Coordinate {
        text: text.to_string(),
        query: parts[0].to_string(),
        graph,
        structure,
        bayesian,
        suite,
        suite_label: parts[4].to_string(),
    }
}

/// Every coordinate the family `estimator|query` claims, from the committed coordinate list.
pub fn family_coordinates(family: &str) -> Vec<Coordinate> {
    let found: Vec<Coordinate> = include_str!("coordinates.tsv")
        .lines()
        .filter_map(|line| line.split_once('\t'))
        .filter(|(name, _)| *name == family)
        .map(|(_, coordinate)| parse(coordinate))
        .collect();
    assert!(!found.is_empty(), "no coordinates for family {family}");
    found
}

// ------------------------------------------------------------------------ the static design

pub const T: u32 = 0;
pub const Y: u32 = 1;
pub const Z: u32 = 2;
pub const R: u32 = 3;
pub const Q: u32 = 4;

/// Row of the confounded static table.
#[derive(Clone, Copy, Debug)]
pub struct Row {
    pub t: f64,
    pub z: f64,
    pub r: f64,
    pub q: f64,
}

/// The noise-free law `y = 2 t + z`.
pub fn law(t: f64, z: f64) -> f64 {
    2.0 * t + z
}

/// Rows of the balanced confounded design. For `z = -1` each repetition holds 4 treated and
/// 8 control rows, for `z = 1` 8 treated and 4 control rows, so `P(t = 1 | z)` is 1/3 and 2/3
/// and the naive contrast is off by 2/3. The two nuisance indicators `r` and `q` split every
/// `(t, z)` cell into four equal quarters, so they are exactly orthogonal to `t` given `z`
/// (a logistic propensity on any of `z`, `{z, r}`, `{z, q}`, `{z, r, q}` reproduces the cell
/// proportions). The 120 rows are permuted by a stride coprime to their count so no
/// cross-fit fold is a block of one cell.
pub fn rows() -> Vec<Row> {
    let mut rows = Vec::new();
    for _ in 0..5 {
        for z in [-1.0, 1.0] {
            let (treated, control) = if z < 0.0 { (4, 8) } else { (8, 4) };
            for (t, count) in [(0.0, control), (1.0, treated)] {
                for k in 0..count {
                    rows.push(Row { t, z, r: f64::from(k % 2), q: f64::from((k / 2) % 2) });
                }
            }
        }
    }
    let n = rows.len();
    assert_eq!(n, 120);
    (0..n).map(|i| rows[(i * 7) % n]).collect()
}

/// Columns `t, y, z, r, q` (variable ids 0..=4) with the outcome `outcome(t, z)`.
pub fn table_with(outcome: fn(f64, f64) -> f64) -> TabularData {
    let rows = rows();
    let t: Vec<f64> = rows.iter().map(|row| row.t).collect();
    let z: Vec<f64> = rows.iter().map(|row| row.z).collect();
    let r: Vec<f64> = rows.iter().map(|row| row.r).collect();
    let q: Vec<f64> = rows.iter().map(|row| row.q).collect();
    let y: Vec<f64> = rows.iter().map(|row| outcome(row.t, row.z)).collect();
    TabularData::from_f64_columns([
        ("t", t.as_slice()),
        ("y", y.as_slice()),
        ("z", z.as_slice()),
        ("r", r.as_slice()),
        ("q", q.as_slice()),
    ])
    .unwrap()
}

/// [`table_with`] the law `y = 2 t + z`.
pub fn confounded_table() -> TabularData {
    table_with(law)
}

/// The back-door contrast of the table by hand: the `z`-stratified difference of arm means
/// of `outcome`, pooled by stratum size. Independent of every estimator.
pub fn stratified_truth_with(outcome: fn(f64, f64) -> f64) -> f64 {
    let rows = rows();
    let n = rows.len() as f64;
    let mut total = 0.0;
    for z in [-1.0, 1.0] {
        let arm_mean = |t: f64| {
            let cell: Vec<f64> = rows
                .iter()
                .filter(|row| row.z == z && row.t == t)
                .map(|row| outcome(row.t, row.z))
                .collect();
            cell.iter().sum::<f64>() / cell.len() as f64
        };
        let size = rows.iter().filter(|row| row.z == z).count() as f64;
        total += size / n * (arm_mean(1.0) - arm_mean(0.0));
    }
    total
}

pub fn stratified_truth() -> f64 {
    stratified_truth_with(law)
}

/// The unadjusted contrast `E[y | t = 1] - E[y | t = 0]` of the table, by hand.
pub fn naive_contrast() -> f64 {
    let rows = rows();
    let mean = |t: f64| {
        let ys: Vec<f64> =
            rows.iter().filter(|row| row.t == t).map(|row| law(row.t, row.z)).collect();
        ys.iter().sum::<f64>() / ys.len() as f64
    };
    mean(1.0) - mean(0.0)
}

/// Assert the design discriminates adjustment from the naive contrast and return the truth.
pub fn confounded_truth() -> f64 {
    let truth = stratified_truth();
    assert!((truth - 2.0).abs() < 1e-12, "stratified truth {truth}");
    let naive = naive_contrast();
    assert!((naive - (2.0 + 2.0 / 3.0)).abs() < 1e-12, "naive contrast {naive}");
    truth
}

/// Columns `era, t, m, y` of the unknown-order tier design. `m` is a balanced +-1 column
/// crossed fully with every `(era, t)` cell (cells weighted 3:1:1:3 so `t` depends on `era`),
/// so `m` is exactly orthogonal to `t` and `era`; `y = t + 0.2 era` is noise free. Whichever
/// side of `t` the unknown tier order puts `m`, the effect of `t` is the coefficient 1.
pub fn unknown_table() -> TabularData {
    unknown_table_with(|era, t| t + 0.2 * era)
}

/// [`unknown_table`] with the outcome `outcome(era, t)`.
pub fn unknown_table_with(outcome: fn(f64, f64) -> f64) -> TabularData {
    let (mut era, mut t, mut m, mut y) = (vec![], vec![], vec![], vec![]);
    for _ in 0..3 {
        for a in [-1.0, 1.0] {
            for t_i in [0.0, 1.0] {
                let weight = if (a < 0.0) == (t_i < 0.5) { 3 } else { 1 };
                for _ in 0..weight {
                    for e_m in [-1.0, 1.0] {
                        era.push(a);
                        t.push(t_i);
                        m.push(e_m);
                        y.push(outcome(a, t_i));
                    }
                }
            }
        }
    }
    TabularData::from_f64_columns([
        ("era", era.as_slice()),
        ("t", t.as_slice()),
        ("m", m.as_slice()),
        ("y", y.as_slice()),
    ])
    .unwrap()
}

/// The two tier-order scenario effects of [`unknown_table`], by hand: `y = t + 0.2 era` does
/// not involve `m`, so adjusting `{era}` or `{era, m}` both leave the coefficient 1.
pub const UNKNOWN_TRUTH: [f64; 2] = [1.0, 1.0];

// ------------------------------------------------------------------------ designs and graphs

fn dense(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

/// `r -> t <- z`, `z -> y`, `t -> y`: the collider at `t` makes every edge compelled.
pub const ATOM_A: [(u32, u32); 4] = [(R, T), (Z, T), (Z, Y), (T, Y)];
/// The same shape with `q` as the second cause of `t`.
pub const ATOM_B: [(u32, u32); 4] = [(Q, T), (Z, T), (Z, Y), (T, Y)];

/// A table with the graph atoms and tier backgrounds that describe it.
#[derive(Clone)]
pub struct Design {
    /// Table the `Dag`, `Cpdag`, `Pag` and `CoDetermined` coordinates run on.
    pub main: TabularData,
    /// Table the `Unknown` coordinates run on (tiers `unknown_tiers`).
    pub unknown: TabularData,
    /// Number of variables of `main`.
    pub n_vars: usize,
    /// First graph (also the explicit/accepted graph).
    pub atom_a: Vec<(u32, u32)>,
    /// Second graph of the two-atom graph posterior.
    pub atom_b: Vec<(u32, u32)>,
    /// Tiers of the `CoDetermined` coordinate.
    pub codetermined_tiers: Vec<Vec<&'static str>>,
    /// Tiers of the `Unknown` coordinate.
    pub unknown_tiers: Vec<Vec<&'static str>>,
}

impl Design {
    /// The balanced confounded design with the compelled collider graphs.
    pub fn confounded() -> Self {
        Self {
            main: confounded_table(),
            unknown: unknown_table(),
            n_vars: 5,
            atom_a: ATOM_A.to_vec(),
            atom_b: ATOM_B.to_vec(),
            codetermined_tiers: vec![vec!["z", "r", "q"], vec!["t"], vec!["y"]],
            unknown_tiers: vec![vec!["era"], vec!["t", "m"], vec!["y"]],
        }
    }
}

pub fn dag_of(n: usize, edges: &[(u32, u32)]) -> Dag {
    let mut graph = Dag::with_variables(u32::try_from(n).unwrap());
    for &(from, to) in edges {
        graph.insert_directed(dense(from), dense(to)).unwrap();
    }
    graph
}

pub fn cpdag_of(n: usize, edges: &[(u32, u32)]) -> Cpdag {
    Cpdag::from_dag(&dag_of(n, edges))
}

pub fn pag_of(n: usize, edges: &[(u32, u32)]) -> Pag {
    let mut graph = Pag::with_variables(u32::try_from(n).unwrap());
    for &(from, to) in edges {
        graph.insert_directed(dense(from), dense(to)).unwrap();
    }
    graph
}

fn sampling(weights: &[f64]) -> SamplingSummary {
    SamplingSummary {
        ess: 1.0 / weights.iter().map(|w| w * w).sum::<f64>(),
        diagnostics: InferenceDiagnostics::analytic("earn_known_truth"),
        rejected_invalid: 0,
    }
}

pub const POSTERIOR_WEIGHTS: [f64; 2] = [0.6, 0.4];

/// A two-atom graph posterior over the design's two graphs; both atoms identify the same
/// effect on the design's table.
pub fn graph_posterior_of(design: &Design, kind: GraphKind) -> GraphPosterior {
    let weights = POSTERIOR_WEIGHTS.to_vec();
    let n = design.n_vars;
    let atoms = [&design.atom_a, &design.atom_b];
    let zeros = || vec![0.0; n * n];
    match kind {
        GraphKind::Cpdag => {
            let masks: Vec<_> = atoms
                .iter()
                .map(|edges| adjacency_mask_from_cpdag(&cpdag_of(n, edges)).unwrap())
                .collect();
            GraphPosterior::from_atoms(
                n,
                weights.clone(),
                masks,
                zeros(),
                zeros(),
                sampling(&weights),
            )
            .unwrap()
            .with_atom_kind(GraphPosteriorAtomKind::Cpdag)
        }
        GraphKind::Pag => {
            let (mut masks, mut marks) = (Vec::new(), Vec::new());
            for edges in atoms {
                let (adjacency, mark) = adjacency_masks_from_pag(&pag_of(n, edges)).unwrap();
                masks.push(adjacency);
                marks.push(mark);
            }
            GraphPosterior::from_atoms(
                n,
                weights.clone(),
                masks,
                zeros(),
                zeros(),
                sampling(&weights),
            )
            .unwrap()
            .with_atom_kind(GraphPosteriorAtomKind::Pag)
            .with_mark_atoms(marks)
            .unwrap()
        }
        _ => {
            let masks: Vec<u64> = atoms
                .iter()
                .map(|edges| {
                    edges.iter().fold(0_u64, |mask, &(from, to)| {
                        set_edge(mask, n, from as usize, to as usize, true)
                    })
                })
                .collect();
            GraphPosterior::from_atoms(
                n,
                weights.clone(),
                masks,
                zeros(),
                zeros(),
                sampling(&weights),
            )
            .unwrap()
        }
    }
}

/// Attach the structure a coordinate names to a builder that has the data.
pub fn with_structure(
    builder: StudyBuilder,
    coordinate: &Coordinate,
    design: &Design,
) -> StudyBuilder {
    let (n, a) = (design.n_vars, &design.atom_a);
    match (coordinate.graph, coordinate.structure) {
        (GraphKind::Dag, Structure::Explicit) => builder.graph(dag_of(n, a)),
        (GraphKind::Dag, Structure::Accepted) => builder.graph(AcceptedGraph::from(dag_of(n, a))),
        (GraphKind::Cpdag, Structure::Explicit) => builder.graph(cpdag_of(n, a)),
        (GraphKind::Cpdag, Structure::Accepted) => {
            builder.graph(AcceptedGraph::from(cpdag_of(n, a)))
        }
        (GraphKind::Pag, Structure::Explicit) => builder.graph(pag_of(n, a)),
        (GraphKind::Pag, Structure::Accepted) => builder.graph(AcceptedGraph::from(pag_of(n, a))),
        (kind, Structure::GraphPosterior) => {
            builder.graph_posterior(graph_posterior_of(design, kind))
        }
        (kind, structure) => panic!("no static graph builder for {kind:?} {structure:?}"),
    }
}

// ------------------------------------------------------------------------ execution

/// Build the study a coordinate names around `table` (`design.unknown` for `Unknown`), up to
/// but not including the query: the caller adds the query, estimator, inference and validation.
pub fn study_for(coordinate: &Coordinate, design: &Design) -> StudyBuilder {
    match coordinate.graph {
        GraphKind::CoDetermined => {
            let background = TieredBackground::from_named(
                design.main.schema(),
                &design.codetermined_tiers,
                WithinTier::CoDetermined,
            )
            .unwrap();
            Study::tabular(design.main.clone()).tiered_background(background).unwrap()
        }
        GraphKind::Unknown => {
            let background = TieredBackground::from_named(
                design.unknown.schema(),
                &design.unknown_tiers,
                WithinTier::Unknown,
            )
            .unwrap();
            Study::tabular(design.unknown.clone()).tiered_background(background).unwrap()
        }
        _ => with_structure(Study::tabular(design.main.clone()), coordinate, design),
    }
}

/// The table a coordinate runs on.
pub fn table_of<'a>(coordinate: &Coordinate, design: &'a Design) -> &'a TabularData {
    if coordinate.graph == GraphKind::Unknown { &design.unknown } else { &design.main }
}

/// The binary average-effect query of `t` on `y` in the coordinate's table.
pub fn binary_query(coordinate: &Coordinate, design: &Design) -> AverageEffectQuery {
    let schema = table_of(coordinate, design).schema();
    AverageEffectQuery::binary_ate(schema.id_of("t").unwrap(), schema.id_of("y").unwrap())
}

/// Run one coordinate of an `AverageEffect` family with `spec` and `inference`.
pub fn run_average_effect(
    coordinate: &Coordinate,
    spec: &EstimatorSpec,
    inference: &InferenceMode,
    design: &Design,
    seed: u64,
) -> StudyResult {
    let query = CausalQuery::AverageEffect(binary_query(coordinate, design));
    run_query(coordinate, spec, inference, design, query, None, seed)
}

/// Run one coordinate with an explicit `query` (and optionally a declared identifier).
pub fn run_query(
    coordinate: &Coordinate,
    spec: &EstimatorSpec,
    inference: &InferenceMode,
    design: &Design,
    query: CausalQuery,
    identifier: Option<antecedent::IdentifierId>,
    seed: u64,
) -> StudyResult {
    let ctx = ExecutionContext::for_tests(seed);
    let builder = study_for(coordinate, design)
        .query(query)
        .estimator(spec.clone())
        .inference(inference.clone())
        .refute(coordinate.suite);
    let builder = match identifier {
        Some(identifier) => builder.identifier(identifier),
        None => builder,
    };
    let builder = if matches!(spec, EstimatorSpec::Default(_)) && !coordinate.bayesian {
        builder.bootstrap_replicates(0)
    } else {
        builder
    };
    builder
        .build()
        .unwrap_or_else(|e| panic!("{}: build: {e}", coordinate.text))
        .run(&ctx)
        .unwrap_or_else(|e| panic!("{}: run: {e}", coordinate.text))
}

fn values_of(value: &ResponseValue) -> Vec<f64> {
    match value {
        ResponseValue::Scalar(v) => vec![*v],
        ResponseValue::Surface { mean, .. } => mean.to_vec(),
        other => panic!("expected a scalar or surface response, got {other:?}"),
    }
}

/// Values of the graph-posterior atoms that identified the query, in publication order.
pub fn identified_atoms(result: &StudyResult) -> Vec<Vec<f64>> {
    result
        .structural_response
        .as_ref()
        .expect("structural response")
        .atoms
        .iter()
        .filter_map(|atom| atom.value.as_ref().map(values_of))
        .collect()
}

pub fn unidentified_mass(result: &StudyResult) -> f64 {
    result.structural_response.as_ref().expect("structural response").unidentified_mass
}

/// Compare the result of a coordinate with the closed-form `truth` (the scalar effect, or the
/// per-scenario effects for `Unknown`), and check the validation suite the coordinate names
/// executed (or did not).
pub fn assert_coordinate(
    coordinate: &Coordinate,
    estimator_id: &str,
    result: &StudyResult,
    truth: f64,
    unknown_truth: &[f64],
    tolerance: f64,
) {
    let label = &coordinate.text;
    if coordinate.structure != Structure::GraphPosterior {
        assert_eq!(
            result.logical_plan.estimator.as_deref(),
            Some(estimator_id),
            "{label}: the explicitly selected estimator ran"
        );
    }
    match (coordinate.graph, coordinate.structure) {
        (GraphKind::Unknown, _) => {
            let effect = result.estimate().as_effect().unwrap();
            let values = effect.scenarios.effects.as_ref().expect("scenario effects");
            assert_eq!(values.len(), unknown_truth.len(), "{label}: scenario count");
            for (got, want) in values.iter().zip(unknown_truth) {
                assert!((got - want).abs() < tolerance, "{label}: scenario {got} vs {want}");
            }
        }
        (_, Structure::GraphPosterior) => {
            let atoms = identified_atoms(result);
            assert_eq!(atoms.len(), 2, "{label}: two identified atoms");
            for atom in &atoms {
                assert_eq!(atom.len(), 1, "{label}");
                assert!(
                    (atom[0] - truth).abs() < tolerance,
                    "{label}: atom {} vs closed form {truth}",
                    atom[0]
                );
            }
            assert!(unidentified_mass(result) < 1e-12, "{label}: unidentified mass");
        }
        _ => {
            let ate = result.estimate().as_effect().unwrap().ate;
            assert!((ate - truth).abs() < tolerance, "{label}: ate {ate} vs closed form {truth}");
        }
    }
    match coordinate.suite {
        RefuteSuite::None => {
            assert!(result.refutations.is_empty(), "{label}: validation none emits no refuters");
        }
        _ if coordinate.bayesian => assert!(
            !result.refutations.is_empty() || !result.predictive_checks.is_empty(),
            "{label}: validation (refuters or predictive checks) must execute"
        ),
        _ => assert!(!result.refutations.is_empty(), "{label}: refuters must execute"),
    }
}

/// Run `body` on every coordinate of `family`, collecting every failure (a panic with its
/// message) instead of stopping at the first, then fail once listing all of them.
pub fn for_each_coordinate(family: &str, body: impl Fn(&Coordinate)) {
    for_each_matching(family, |_| true, body);
}

/// [`for_each_coordinate`] over the coordinates of `family` that satisfy `keep` (the
/// coordinates the test claims; the rest are recorded as not estimable by the estimator).
pub fn for_each_matching(
    family: &str,
    keep: impl Fn(&Coordinate) -> bool,
    body: impl Fn(&Coordinate),
) {
    // Graph-posterior coordinates are not claimed: the graph-posterior executors fix the
    // estimator (`linear.adjustment.ate` / `bayesian.gcomp` in
    // `analysis/execute/bayesian_path.rs` and `class_posterior.rs`, the default response
    // estimator in `response_path.rs`) and ignore the explicitly selected one.
    let coordinates: Vec<Coordinate> = family_coordinates(family)
        .into_iter()
        .filter(|c| c.structure != Structure::GraphPosterior && keep(c))
        .collect();
    assert!(!coordinates.is_empty(), "no claimed coordinate in {family}");
    let mut failures = Vec::new();
    for coordinate in &coordinates {
        let outcome = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| body(coordinate)));
        if let Err(payload) = outcome {
            let message = payload
                .downcast_ref::<String>()
                .cloned()
                .or_else(|| payload.downcast_ref::<&str>().map(|s| (*s).to_string()))
                .unwrap_or_else(|| "non-string panic".to_string());
            failures.push(format!("{}: {message}", coordinate.text));
        }
    }
    assert!(
        failures.is_empty(),
        "{} of {} coordinates of {family} failed:\n{}",
        failures.len(),
        coordinates.len(),
        failures.join("\n")
    );
}

/// Run `body` on a thread with a stack deep enough for class envelopes in debug builds.
pub fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("earn-known-truth".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap();
}

/// Keep `Arc` in scope for callers that build interventions.
pub type SharedStr = Arc<str>;
