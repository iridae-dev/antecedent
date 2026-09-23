//! Repeated-sampling coverage of the static response coordinates that had
//! no coverage evidence: the Frequentist graph-posterior intervention level,
//! the Bayesian class-aware (CPDAG / PAG) intervention level and response
//! curve, and the CoDetermined same-tier joint intervention level.
//!
//! Every test scores exactly the interval the study reports by default (no
//! response options, no bootstrap or draw overrides, the facade's default
//! Bayesian configuration) at its published level 0.95, and the same
//! construction at the gate's level 0.90 from the same replicates (see
//! `common::reported`). Every tally is keyed as a coverage record through
//! [`keyed`], this file's one emission point; a record declares only its
//! provenance (test, DGP, interval method) and the construction it describes
//! is read from the runtime, bound off every scored execution.
//!
//! Class-aware Bayesian responses publish a band only when every contributing
//! completion returns the same response (they share the adjustment set, so
//! the per-completion posteriors are one posterior). The DGPs below therefore
//! use classes whose identified completions all adjust the same set; a class
//! with disagreeing completions publishes the unbanded identified set, which
//! has no interval to score.
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines, clippy::doc_markdown)]
#![allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

mod common;

use std::sync::Arc;

use antecedent::discovery::GraphPosterior;
use antecedent::{AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Intervention, ResponseFunctional,
    ResponseIdentification, ResponseQuery, ResponseValue, Value, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_graph::{
    Cpdag, DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag, TieredBackground, WithinTier,
};
use common::calibration::{
    CoverageTally, GRID_POINTS, PRECISION_N_SIM, RecordKey, gaussian, grid_n, map_replicates,
    n_sim, stream_seed, unit_uniform,
};
use common::calibration_bind::{bind, bind_all};
use common::reported::{
    GATE_LEVEL, REPORTED_LEVEL, gate, gate_at, n_sim_at_least, record_pair, response_band,
    response_normal_pair, response_scalar, skip_pair,
};
// The continuous response law is shared with the Bayesian static suite;
// one owner, so both measurements of it run on the same replicate data.
use common::static_dgp::response_data;

// ---------------------------------------------------------------- emission

/// Fixed facts of one measured coordinate.
///
/// The construction a record describes (query, graph class, inference,
/// estimator, SE kind, dependence, posterior, identification) is never
/// declared here: it is read from the runtime by [`bind`]. What stays is the
/// record's provenance (the DGP and the interval method the tally scores),
/// plus the sample size and the estimator / graph class the tests assert on.
#[derive(Clone, Copy)]
struct Cell {
    /// Graph class, for assertion messages only.
    graph_class: &'static str,
    /// Estimator the study is pinned to resolve to.
    estimator: &'static str,
    /// Interval method the tally scores, as the runtime reports it.
    interval_method: &'static str,
    /// Bare name of this file's data-generating function.
    dgp: &'static str,
    n: u64,
}

/// This file's single coverage-record emission point: every tally is created
/// here, so re-keying the record is one edit. `test` is the name of the
/// `#[test] fn` that emits the record.
fn keyed(test: &'static str, cell: Cell, level: f64) -> CoverageTally {
    CoverageTally::for_record(
        RecordKey { test, dgp: cell.dgp, interval: cell.interval_method },
        level,
    )
}

/// The reported-level and gate-level tallies of one coordinate. The two
/// records differ by level, so neither needs a label.
fn keyed_pair(test: &'static str, cell: Cell) -> [CoverageTally; 2] {
    [keyed(test, cell, REPORTED_LEVEL), keyed(test, cell, GATE_LEVEL)]
}

/// Label distinguishing the records one band test emits per grid coordinate.
fn band_label(a: f64) -> String {
    format!("a={a}")
}

/// Bind one execution to both levels of a coordinate's tally pair.
fn bind_pair(tallies: &mut [CoverageTally; 2], study: &Study, result: &StudyResult) {
    let [reported, gate_level] = tallies;
    bind_all(&mut [reported, gate_level], study, result);
}

// ---------------------------------------------------------------- helpers

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn table(columns: &[(&str, &[f64])]) -> TabularData {
    TabularData::from_f64_columns(columns.iter().map(|(name, col)| (*name, *col))).unwrap()
}

const GRID: [f64; 5] = [-1.0, -0.5, 0.0, 0.5, 1.0];

fn level_query(level: f64) -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: v(1),
        interventions: Arc::from([Intervention::set(v(0), Value::f64(level))]),
    })
}

fn curve_query() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: v(1),
        treatment: ContinuousDomain::new(v(0), GridSpec::Values(GRID.to_vec().into())),
    })
}

/// The facade's default Bayesian configuration (Python `Bayesian()`):
/// Laplace backend, 1000 draws, prior scale 10.
fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::laplace())
}

/// Accepted or explicit structure for the same class graph.
#[derive(Clone)]
enum Structure {
    Cpdag(Cpdag),
    Pag(Pag),
}

#[derive(Clone, Copy)]
struct Run {
    accepted: bool,
    suite: RefuteSuite,
    /// `Some(level)` sets only `ContinuousResponseOptions::confidence_level`.
    level: Option<f64>,
}

const DEFAULT_RUN: Run = Run { accepted: false, suite: RefuteSuite::None, level: None };

/// Build and run the class-aware study, keeping the [`Study`] so the scored
/// replicate can be bound to its coverage record.
fn run_class(
    data: &TabularData,
    graph: &Structure,
    query: ResponseQuery,
    inference: InferenceMode,
    run: Run,
    seed: u64,
) -> Option<(Study, StudyResult)> {
    let builder = Study::tabular(data.clone());
    let builder = match (graph, run.accepted) {
        (Structure::Cpdag(g), false) => builder.graph(g.clone()),
        (Structure::Cpdag(g), true) => builder.graph(AcceptedGraph::from(g.clone())),
        (Structure::Pag(g), false) => builder.graph(g.clone()),
        (Structure::Pag(g), true) => builder.graph(AcceptedGraph::from(g.clone())),
    };
    let mut builder =
        builder.query(CausalQuery::Response(query)).inference(inference).refute(run.suite);
    if let Some(level) = run.level {
        builder = builder.response_options(ContinuousResponseOptions {
            confidence_level: level,
            ..ContinuousResponseOptions::default()
        });
    }
    let study = builder.build().ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    Some((study, result))
}

fn scalar_value(result: &StudyResult) -> Option<f64> {
    match &result.response.as_ref()?.estimate {
        ResponseIdentification::PointIdentified(ResponseValue::Scalar(v)) => Some(*v),
        _ => None,
    }
}

fn mean_z(data: &TabularData) -> f64 {
    let z = data.float64_values(data.schema().id_of("z").unwrap()).unwrap();
    z.iter().sum::<f64>() / z.len() as f64
}

// ====================================================== graph-posterior level

/// Atom weights of the graph posterior below, over
/// [`common::static_dgp::response_data`]: atom A (`t -> y`, no adjustment)
/// estimates `E[Y | T = 1] = 3 + 0.8·0.3 = 3.24`; atom B (`z -> t`, `z -> y`,
/// `t -> y`) estimates `E[Y | do(T = 1)] = 3`; atom C (`y -> t`) is
/// unidentified. The reported frozen-weight aggregate over identified atoms is
/// [`GP_TRUTH`], `(0.5·3.24 + 0.3·3) / 0.8 = 3.15`, its population value.
const GP_WEIGHTS: [f64; 3] = [0.5, 0.3, 0.2];
const GP_TRUTH: f64 = (0.5 * 3.24 + 0.3 * 3.0) / 0.8;

fn graph_posterior() -> GraphPosterior {
    use antecedent_discovery::set_edge;
    let direct = set_edge(0, 3, 0, 1, true);
    let adjusted = set_edge(set_edge(set_edge(0, 3, 0, 1, true), 3, 2, 0, true), 3, 2, 1, true);
    let unidentified = set_edge(0, 3, 1, 0, true);
    GraphPosterior::new(
        3,
        GP_WEIGHTS.to_vec(),
        vec![direct, adjusted, unidentified],
        vec![0.0; 9],
        vec![0.0; 9],
        1.0 / GP_WEIGHTS.iter().map(|w| w * w).sum::<f64>(),
        antecedent_prob::InferenceDiagnostics::analytic("v110_calibration_response"),
        0,
    )
    .unwrap()
}

/// Build and run the graph-posterior study, keeping the [`Study`] so the
/// scored replicate can be bound to its coverage record.
fn run_graph_posterior(
    data: TabularData,
    suite: RefuteSuite,
    seed: u64,
) -> Option<(Study, StudyResult)> {
    let study = Study::tabular(data)
        .graph_posterior(graph_posterior())
        .query(CausalQuery::Response(level_query(1.0)))
        .refute(suite)
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    Some((study, result))
}

const GP_CELL: Cell = Cell {
    graph_class: "Dag",
    estimator: "response.intervention_gcomp",
    interval_method: "analytic_se",
    dgp: "response_data",
    n: 500,
};

/// Joint-IF scalar interval of the frozen-weight graph-posterior aggregate
/// `E[Y | do(T = 1)]` over two disagreeing identified atoms.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn intervention_response_dag_graph_posterior_frequentist_nominal_coverage() {
    let mut tallies = keyed_pair(
        "intervention_response_dag_graph_posterior_frequentist_nominal_coverage",
        GP_CELL,
    );
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0x110_0001, rep);
        let data = response_data(grid_n(500), seed);
        let (study, result) = run_graph_posterior(data.clone(), RefuteSuite::None, seed)?;
        let pair = response_normal_pair(&result);
        if rep == 0 {
            assert_eq!(
                result.logical_plan.estimator.as_deref(),
                Some(GP_CELL.estimator),
                "the default graph-posterior level estimator"
            );
            assert!(
                result
                    .diagnostics
                    .iter()
                    .any(|d| d.code.as_ref() == "estimate.response.graph_posterior.joint_if_se"),
                "the multi-atom SE must come from the joint IF combiner"
            );
            // Validation cheap / full share the reported interval.
            for suite in [RefuteSuite::Cheap, RefuteSuite::Full] {
                let (_, other) =
                    run_graph_posterior(data.clone(), suite, seed).expect("validated run");
                assert_eq!(response_scalar(&other), response_scalar(&result), "{suite:?}");
            }
        }
        Some((study, result, pair))
    });
    for scored in &runs {
        let Some((study, result, pair)) = scored else {
            skip_pair(&mut tallies);
            continue;
        };
        bind_pair(&mut tallies, study, result);
        record_pair(&mut tallies, *pair, GP_TRUTH);
    }
    gate(&tallies, &[None, None]);
}

// ================================================= Bayesian class responses

/// `z -> t`, `z -> y`, `t -> y`, `w — z` (columns `t, y, z, w`); data from
/// `z ~ N(0,1)`, `w ~ N(0,1)`, `t = 0.8 z + e`, `y = t + z + e`. Both
/// orientations of `w — z` adjust `{z}`, so the two completions return the
/// same posterior. The Bayesian response holds the empirical covariate law
/// fixed, so its target is the sample-covariate level `a + z̄`.
fn cpdag_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut t, mut y, mut z, mut w) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        w[i] = g();
        t[i] = 0.8 * z[i] + g();
        y[i] = t[i] + z[i] + g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z), ("w", &w)])
}

fn agreeing_cpdag() -> Cpdag {
    let mut g = Cpdag::with_variables(4);
    g.insert_directed(d(2), d(0)).unwrap();
    g.insert_directed(d(2), d(1)).unwrap();
    g.insert_directed(d(0), d(1)).unwrap();
    g.insert_undirected(d(3), d(2)).unwrap();
    g
}

/// `r o-> t`, `z o-> t`, `z -> y`, `t -> y` (columns `t, y, z, r`), the PAG
/// of `conformance/response/class_aware_envelope/pag_identified.json`: `t -> y`
/// is visible in every MAG completion and every completion adjusts `{z}`.
/// Data: `z, r ~ N(0,1)`, `t = 0.5 z + 0.4 r + e`, `y = 1 + 2t + 0.8 z + e`;
/// sample-covariate target `1 + 2a + 0.8 z̄`.
fn pag_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut t, mut y, mut z, mut r) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        r[i] = g();
        t[i] = 0.5 * z[i] + 0.4 * r[i] + g();
        y[i] = 1.0 + 2.0 * t[i] + 0.8 * z[i] + g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z), ("r", &r)])
}

fn agreeing_pag() -> Pag {
    let mut pag = Pag::with_variables(4);
    for (a, b, at_a, at_b) in [
        (3, 0, Endpoint::Circle, Endpoint::Arrow),
        (2, 0, Endpoint::Circle, Endpoint::Arrow),
        (2, 1, Endpoint::Tail, Endpoint::Arrow),
        (0, 1, Endpoint::Tail, Endpoint::Arrow),
    ] {
        pag.insert_marked(MarkedEdge { a: d(a), b: d(b), at_a, at_b, middle: MiddleMark::Empty })
            .unwrap();
    }
    pag
}

struct ClassCase {
    cell: Cell,
    graph: fn() -> Structure,
    data: fn(usize, u64) -> TabularData,
    /// Sample-covariate level at `a` given `z̄`.
    truth: fn(f64, f64) -> f64,
    family: u64,
}

fn cpdag_case() -> ClassCase {
    ClassCase {
        cell: Cell {
            graph_class: "Cpdag",
            estimator: "response.bayesian",
            interval_method: "posterior_quantile",
            dgp: "cpdag_data",
            n: 500,
        },
        graph: || Structure::Cpdag(agreeing_cpdag()),
        data: cpdag_data,
        truth: |a, z_bar| a + z_bar,
        family: 0x110_0010,
    }
}

fn pag_case() -> ClassCase {
    ClassCase {
        cell: Cell {
            graph_class: "Pag",
            estimator: "response.bayesian",
            interval_method: "posterior_quantile",
            dgp: "pag_data",
            n: 500,
        },
        graph: || Structure::Pag(agreeing_pag()),
        data: pag_data,
        truth: |a, z_bar| 1.0 + 2.0 * a + 0.8 * z_bar,
        family: 0x110_0020,
    }
}

/// Bayesian class-aware `E[Y | do(T = 1)]`: the 0.95 interval the study
/// reports, and the facade's own 0.90 interval on the same data and seed.
fn class_level_coverage(test: &'static str, case: &ClassCase) {
    let mut tallies = keyed_pair(test, case.cell);
    let graph = (case.graph)();
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(case.family, rep);
        let data = (case.data)(grid_n(case.cell.n as usize), seed);
        let truth = (case.truth)(1.0, mean_z(&data));
        let reported = run_class(&data, &graph, level_query(1.0), bayes(), DEFAULT_RUN, seed);
        let gate_run = run_class(
            &data,
            &graph,
            level_query(1.0),
            bayes(),
            Run { level: Some(GATE_LEVEL), ..DEFAULT_RUN },
            seed,
        );
        let (Some((reported_study, reported)), Some((gate_study, gate_run))) = (reported, gate_run)
        else {
            return None;
        };
        let at_reported = response_scalar(&reported);
        let at_gate = response_scalar(&gate_run);
        if rep == 0 {
            assert_eq!(reported.logical_plan.estimator.as_deref(), Some(case.cell.estimator));
            let structural = reported.structural_response.as_ref().expect("completion atoms");
            assert!(structural.atoms.len() > 1, "the class must enumerate several completions");
            assert!(at_reported.is_some(), "agreeing completions must publish their band");
            assert_eq!(scalar_value(&reported), scalar_value(&gate_run), "same posterior");
            let (_, accepted) = run_class(
                &data,
                &graph,
                level_query(1.0),
                bayes(),
                Run { accepted: true, ..DEFAULT_RUN },
                seed,
            )
            .expect("accepted structure");
            assert_eq!(response_scalar(&accepted), at_reported, "accepted = explicit");
        }
        Some(((reported_study, reported), (gate_study, gate_run), at_reported, at_gate, truth))
    });
    for scored in &runs {
        let Some(((reported_study, reported), (gate_study, gate_run), at_reported, at_gate, truth)) =
            scored
        else {
            skip_pair(&mut tallies);
            continue;
        };
        bind(&mut tallies[0], reported_study, reported);
        bind(&mut tallies[1], gate_study, gate_run);
        for (tally, (interval, level)) in
            tallies.iter_mut().zip([(*at_reported, REPORTED_LEVEL), (*at_gate, GATE_LEVEL)])
        {
            if let Some((_, _, published, _)) = interval {
                assert!((published - level).abs() < 1e-12, "published level {published}");
            }
            tally.record(interval.map(|(lo, hi, _, _)| (lo, hi)), *truth);
        }
    }
    gate(&tallies, &[None, None]);
}

/// Bayesian class-aware pointwise band of `m(a)` on [`GRID`].
///
/// A band's five coordinates move together, so this test measures at
/// [`PRECISION_N_SIM`] replicates (see `common::reported::n_sim_at_least`).
fn class_curve_coverage(
    test: &'static str,
    case: &ClassCase,
    measured: &[[Option<f64>; GRID_POINTS]],
) {
    let mut tallies: Vec<CoverageTally> = GRID
        .iter()
        .flat_map(|&a| {
            let label = band_label(a);
            [
                keyed(test, case.cell, REPORTED_LEVEL).labelled(label.clone()),
                keyed(test, case.cell, GATE_LEVEL).labelled(label),
            ]
        })
        .collect();
    let graph = (case.graph)();
    let runs = map_replicates(n_sim_at_least(PRECISION_N_SIM), |rep| {
        let seed = stream_seed(case.family ^ 0x0C, rep);
        let data = (case.data)(grid_n(case.cell.n as usize), seed);
        let z_bar = mean_z(&data);
        let reported = run_class(&data, &graph, curve_query(), bayes(), DEFAULT_RUN, seed);
        let gate_run = run_class(
            &data,
            &graph,
            curve_query(),
            bayes(),
            Run { level: Some(GATE_LEVEL), ..DEFAULT_RUN },
            seed,
        );
        let (Some((reported_study, reported)), Some((gate_study, gate_run))) = (reported, gate_run)
        else {
            return None;
        };
        let bands = [response_band(&reported), response_band(&gate_run)];
        if rep == 0 {
            assert!(bands[0].is_some(), "agreeing completions must publish their band");
            let (_, accepted) = run_class(
                &data,
                &graph,
                curve_query(),
                bayes(),
                Run { accepted: true, ..DEFAULT_RUN },
                seed,
            )
            .expect("accepted structure");
            assert_eq!(response_band(&accepted), bands[0], "accepted = explicit");
        }
        Some(((reported_study, reported), (gate_study, gate_run), bands, z_bar))
    });
    for scored in &runs {
        let Some(((reported_study, reported), (gate_study, gate_run), bands, z_bar)) = scored
        else {
            for tally in &mut tallies {
                tally.skip();
            }
            continue;
        };
        {
            // Every coordinate's tally scores the band of its own execution:
            // the even tallies the reported-level run, the odd ones the
            // facade's own gate-level run.
            let (mut at_reported, mut at_gate) = (Vec::new(), Vec::new());
            for (i, tally) in tallies.iter_mut().enumerate() {
                if i % 2 == 0 {
                    at_reported.push(tally);
                } else {
                    at_gate.push(tally);
                }
            }
            bind_all(&mut at_reported, reported_study, reported);
            bind_all(&mut at_gate, gate_study, gate_run);
        }
        for (j, &a) in GRID.iter().enumerate() {
            let truth = (case.truth)(a, *z_bar);
            for (k, level) in [REPORTED_LEVEL, GATE_LEVEL].into_iter().enumerate() {
                let interval = bands[k].as_ref().map(|(lower, upper, published)| {
                    assert!((published - level).abs() < 1e-12, "published level {published}");
                    (lower[j], upper[j])
                });
                tallies[2 * j + k].record(interval, truth);
            }
        }
    }
    gate_at(&tallies, measured);
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn intervention_response_cpdag_bayesian_nominal_coverage() {
    class_level_coverage("intervention_response_cpdag_bayesian_nominal_coverage", &cpdag_case());
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn intervention_response_pag_bayesian_nominal_coverage() {
    class_level_coverage("intervention_response_pag_bayesian_nominal_coverage", &pag_case());
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn response_curve_cpdag_bayesian_pointwise_nominal_coverage() {
    class_curve_coverage(
        "response_curve_cpdag_bayesian_pointwise_nominal_coverage",
        &cpdag_case(),
        &CPDAG_CURVE_MEASURED,
    );
}

/// Grid-point-0 floor miss at `a = 1` (0.95): 0.932 against floor 0.936.
/// Grid-point-1 ceiling miss at `a = 1` (0.90): 0.919 against ceiling 0.919.
/// Grid-point-2 misses: `a = 0` 0.95 (0.969), `a = 0` 0.90 (0.929), `a = 0.5`
/// 0.95 (0.964), all 1000-replicate measurements.
const CPDAG_CURVE_MEASURED: [[Option<f64>; 3]; 10] = [
    [None, None, None],
    [None, None, None],
    [None, None, None],
    [None, None, None],
    [None, None, Some(0.969)],
    [None, None, Some(0.929)],
    [None, None, Some(0.964)],
    [None, None, None],
    [Some(0.932), None, None],
    [None, Some(0.919), None],
];

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn response_curve_pag_bayesian_pointwise_nominal_coverage() {
    class_curve_coverage(
        "response_curve_pag_bayesian_pointwise_nominal_coverage",
        &pag_case(),
        &PAG_CURVE_MEASURED,
    );
}

/// Grid-point-0 misses: `a = −0.5` 0.95 (0.933), `a = 0` 0.90 (0.881),
/// `a = 0.5` 0.95 (0.924). Grid-point-1 misses (1000 replicates): `a = 0`
/// 0.95 (0.964), `a = 0.5` 0.90 (0.925), `a = 1` 0.95 (0.964).
const PAG_CURVE_MEASURED: [[Option<f64>; 3]; 10] = [
    [None, None, None],
    [None, None, None],
    [Some(0.933), None, None],
    [None, None, None],
    [None, Some(0.964), None],
    [Some(0.881), None, None],
    [Some(0.924), None, None],
    [None, Some(0.925), None],
    [None, Some(0.964), None],
    [None, None, None],
];

// ============================================ CoDetermined joint intervention

/// Same-tier joint law of `v15_numeric_pins::same_tier_joint_dgp` (columns
/// `z, t1, t2, y`; tiers `{z} | {t1, t2} | {y}`, within-tier CoDetermined): a
/// latent drives both treatments but not the outcome,
/// `P(t1 = 1) = σ(−0.2 + 0.9z + 0.7ℓ)`, `P(t2 = 1) = σ(−0.1 + 0.8z + 0.65ℓ)`,
/// `y = 1.2 t1 + 0.8 t2 + 1.5 t1 t2 + 0.55 z + 0.3 e`.
/// `E[Y | do(t1 = 1, t2 = 1)] = 3.5`.
fn codetermined_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut z, mut t1, mut t2, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let sigmoid = |x: f64| 1.0 / (1.0 + (-x).exp());
    for i in 0..n {
        let zi = g();
        let latent = g();
        z[i] = zi;
        let u1 = unit_uniform(stream_seed(seed, 2 * i as u64));
        let u2 = unit_uniform(stream_seed(seed, 2 * i as u64 + 1));
        t1[i] = f64::from(u1 < sigmoid(-0.2 + 0.9 * zi + 0.7 * latent));
        t2[i] = f64::from(u2 < sigmoid(-0.1 + 0.8 * zi + 0.65 * latent));
        y[i] = 1.2 * t1[i] + 0.8 * t2[i] + 1.5 * t1[i] * t2[i] + 0.55 * zi + 0.3 * g();
    }
    table(&[("z", &z), ("t1", &t1), ("t2", &t2), ("y", &y)])
}

const CODETERMINED_CELL: Cell = Cell {
    graph_class: "CoDetermined",
    estimator: "cell.aipw",
    interval_method: "analytic_se",
    dgp: "codetermined_data",
    n: 1200,
};

/// Build and run the same-tier joint study, keeping the [`Study`] so the
/// scored replicate can be bound to its coverage record.
fn run_codetermined(data: TabularData, seed: u64) -> Option<(Study, StudyResult)> {
    let schema = data.schema().clone();
    let background = TieredBackground::from_named(
        &schema,
        &[vec!["z"], vec!["t1", "t2"], vec!["y"]],
        WithinTier::CoDetermined,
    )
    .unwrap();
    let query = ResponseQuery::new(ResponseFunctional::InterventionResponse {
        outcome: schema.id_of("y").unwrap(),
        interventions: Arc::from([
            Intervention::set(schema.id_of("t1").unwrap(), Value::f64(1.0)),
            Intervention::set(schema.id_of("t2").unwrap(), Value::f64(1.0)),
        ]),
    });
    let study = Study::tabular(data)
        .tiered_background(background)
        .unwrap()
        .query(CausalQuery::Response(query))
        .refute(RefuteSuite::None)
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    Some((study, result))
}

/// `cell.aipw` joint cell mean on the CoDetermined closure, the study default
/// for a same-tier joint intervention.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn intervention_response_codetermined_frequentist_nominal_coverage() {
    let mut tallies = keyed_pair(
        "intervention_response_codetermined_frequentist_nominal_coverage",
        CODETERMINED_CELL,
    );
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0x110_0030, rep);
        let (study, result) = run_codetermined(codetermined_data(grid_n(1200), seed), seed)?;
        if rep == 0 {
            assert_eq!(
                result.logical_plan.estimator.as_deref(),
                Some(CODETERMINED_CELL.estimator),
                "the default same-tier joint estimator"
            );
        }
        Some((study, result))
    });
    for scored in &runs {
        let Some((study, result)) = scored else {
            skip_pair(&mut tallies);
            continue;
        };
        bind_pair(&mut tallies, study, result);
        record_pair(&mut tallies, response_normal_pair(result), 3.5);
    }
    gate(&tallies, &[None, None]);
}

// ------------------------------------------------------------ fast shape pins

/// Every coordinate above publishes the interval its coverage test scores.
#[test]
fn static_response_coordinates_publish_the_scored_interval() {
    let seed = 7;
    let (_, gp) = run_graph_posterior(response_data(300, seed), RefuteSuite::None, seed)
        .expect("graph-posterior run");
    assert!(response_scalar(&gp).is_some_and(|(_, _, level, _)| level == REPORTED_LEVEL));
    let (_, codetermined) =
        run_codetermined(codetermined_data(600, seed), seed).expect("CoDetermined run");
    assert!(response_normal_pair(&codetermined)[0].is_some());
    for case in [cpdag_case(), pag_case()] {
        let data = (case.data)(300, seed);
        let (_, result) =
            run_class(&data, &(case.graph)(), level_query(1.0), bayes(), DEFAULT_RUN, seed)
                .expect("class-aware level run");
        assert!(
            response_scalar(&result).is_some_and(|(_, _, level, _)| level == REPORTED_LEVEL),
            "{}: agreeing completions publish the shared posterior interval",
            case.cell.graph_class
        );
        let (_, curve) =
            run_class(&data, &(case.graph)(), curve_query(), bayes(), DEFAULT_RUN, seed)
                .expect("class-aware curve run");
        assert!(response_band(&curve).is_some(), "{}: curve band", case.cell.graph_class);
    }
}
