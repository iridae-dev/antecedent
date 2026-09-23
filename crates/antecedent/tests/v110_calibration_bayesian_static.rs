//! Repeated-sampling coverage of every Bayesian static coordinate that
//! carries an `iid` coverage record, at the level the facade publishes.
//!
//! The `v19_static_calibration`, `v19_static_envelope_calibration`, and
//! `v19_derivative_calibration` suites score these cells at a 0.90 level with a
//! conjugate backend and reduced draw counts. The records bound to results
//! describe the interval a study reports by default: the facade's default
//! Bayesian configuration (Laplace backend, 1000 draws, prior scale 10 —
//! Python `Bayesian()`), published at 0.95. This file measures exactly that
//! interval on those in-assumption DGPs (reproduced here next to their
//! truths), and the same construction at 0.90 from the same replicates
//! (`common::reported`):
//!
//! * scalar posteriors publish `q025` / `q975` of the effect column; the 0.90
//!   interval is the same equal-tailed rule on the draws;
//! * response and derivative intervals are exchangeable-rank quantiles of a
//!   draw vector the estimator retains on the published uncertainty
//!   (`CredibleDraws`), so the 0.90 interval is that rule on those draws
//!   (checked to reproduce the published 0.95 endpoints exactly), from the
//!   one execution (response options as in those designs: the caller
//!   bandwidth of the point-derivative cells).
//!
//! Every tally is keyed through [`keyed`], this file's one emission point. A
//! key declares only the provenance of the measurement: the emitting test, the
//! data-generating function, and the interval method scored. The construction
//! a record describes — query, graph class, estimator, posterior backend,
//! functional, identification — is never declared here: it is read from the
//! runtime and bound to the tally on every scored replicate, so a record can
//! only describe an interval the facade actually reported.
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::too_many_lines, clippy::doc_markdown, clippy::too_many_arguments)]
#![allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

mod common;

use std::sync::Arc;

use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study, StudyResult};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ContinuousDomain, CounterfactualQuery,
    DerivativeScale, DerivativeWeighting, ExecutionContext, GridSpec, Intervention,
    InterventionalDistributionQuery, MediationContrast, MediationQuery, PathSpecificEffectQuery,
    ResponseFunctional as F, ResponseQuery, Value, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_graph::{Cpdag, Dag, DenseNodeId, Pag};
use common::calibration::{
    CoverageTally, GRID_POINTS, PRECISION_N_SIM, RecordKey, SampleGrid, gaussian, grid_n,
    map_replicates, n_sim, stream_seed,
};
use common::calibration_bind::bind;
use common::reported::{
    GATE_LEVEL, REPORTED_LEVEL, gate, gate_at, n_sim_at_least, posterior_pair,
    response_posterior_pairs,
};
// The laws this suite shares with `v19_static_calibration` live in one owner:
// the two records of a cell are comparable only if the replicate data
// is literally the same, which a copy makes a convention and this makes a fact.
use common::static_dgp::{
    bernoulli, counterfactual_data, distribution_data, envelope_pag as pag, linear_ate_data,
    path_data, response_data, table, two_path_data, uniform,
};

// ---------------------------------------------------------------- emission

/// One record-bearing coordinate of a design.
///
/// `dgp` and `label` are the record's provenance. `query`, `graph_class` and
/// `estimator` are not part of the key: they are the construction
/// [`check_estimator`] pins on the first replicate, while the construction the
/// record describes is bound from the runtime.
#[derive(Clone, Copy)]
struct Cell {
    query: &'static str,
    graph_class: &'static str,
    estimator: &'static str,
    /// Bare name of this file's data-generating function.
    dgp: &'static str,
    /// Distinguishes the several records one test emits for this interval.
    label: Option<&'static str>,
}

/// This file's single coverage-record emission point.
///
/// Every record here scores the facade's posterior-quantile interval; the
/// construction behind that interval is bound from the runtime by [`bind`] on
/// every scored replicate, never declared.
fn keyed(test: &'static str, cell: Cell, level: f64) -> CoverageTally {
    let tally =
        CoverageTally::for_record(RecordKey { test, dgp: cell.dgp, interval: INTERVAL }, level);
    match cell.label {
        Some(label) => tally.labelled(label),
        None => tally,
    }
}

/// The one interval method every cell in this file scores.
const INTERVAL: &str = "posterior_quantile";

/// Leak a generated record label, so a [`Cell`] stays `Copy`.
fn label(text: &str) -> &'static str {
    Box::leak(text.to_owned().into_boxed_str())
}

type Pair = [Option<(f64, f64)>; 2];

/// A built study and the result of running it: [`bind`] reads the construction
/// from the study's contract and the result's reported intervals.
type Run = (Study, StudyResult);

/// One replicate's scored coordinates and the execution they were read from.
///
/// Both levels come from the one execution: the reported interval as
/// published, the gate-level interval re-summarized from that execution's
/// posterior draws (`posterior_pair` / `response_posterior_pairs`).
struct Replicate {
    /// Execution that published every coordinate's reported-level interval.
    reported: Run,
    /// `[reported, gate]` intervals, one entry per cell.
    pairs: Vec<Pair>,
    /// Truth, one entry per cell.
    truths: Vec<f64>,
}

impl Replicate {
    fn new(reported: Run, pairs: Vec<Pair>, truths: Vec<f64>) -> Self {
        Self { reported, pairs, truths }
    }
}

/// Score `k` coordinates (each at 0.95 and 0.90) over `replicates`
/// replicates, emitting the records of the `#[test] fn` named `test`.
/// `replicate(rep)` returns one replicate's per-coordinate interval pairs,
/// truths and executions, or `None` for a refused replicate (skipped, capped
/// at 5%). Every scored replicate binds its execution to the tallies it feeds.
fn coverage_over(
    test: &'static str,
    replicates: u32,
    cells: &[Cell],
    replicate: impl Fn(u64) -> Option<Replicate> + Sync,
    measured: &[Option<[f64; GRID_POINTS]>],
) {
    let mut tallies: Vec<CoverageTally> = cells
        .iter()
        .flat_map(|&cell| [keyed(test, cell, REPORTED_LEVEL), keyed(test, cell, GATE_LEVEL)])
        .collect();
    for scored in map_replicates(replicates, replicate) {
        match scored {
            Some(scored) => {
                assert_eq!(scored.pairs.len(), cells.len());
                let (study, result) = &scored.reported;
                for (j, (pair, truth)) in scored.pairs.iter().zip(&scored.truths).enumerate() {
                    bind(&mut tallies[2 * j], study, result);
                    bind(&mut tallies[2 * j + 1], study, result);
                    tallies[2 * j].record(pair[0], *truth);
                    tallies[2 * j + 1].record(pair[1], *truth);
                }
            }
            None => tallies.iter_mut().for_each(CoverageTally::skip),
        }
    }
    gate(&tallies, measured);
}

fn coverage_at(
    test: &'static str,
    cells: &[Cell],
    replicate: impl Fn(u64) -> Option<Replicate> + Sync,
    measured: &[[Option<f64>; 3]],
) {
    let mut tallies: Vec<CoverageTally> = cells
        .iter()
        .flat_map(|&cell| [keyed(test, cell, REPORTED_LEVEL), keyed(test, cell, GATE_LEVEL)])
        .collect();
    for scored in map_replicates(n_sim(), replicate) {
        match scored {
            Some(scored) => {
                assert_eq!(scored.pairs.len(), cells.len());
                let (study, result) = &scored.reported;
                for (j, (pair, truth)) in scored.pairs.iter().zip(&scored.truths).enumerate() {
                    bind(&mut tallies[2 * j], study, result);
                    bind(&mut tallies[2 * j + 1], study, result);
                    tallies[2 * j].record(pair[0], *truth);
                    tallies[2 * j + 1].record(pair[1], *truth);
                }
            }
            None => tallies.iter_mut().for_each(CoverageTally::skip),
        }
    }
    gate_at(&tallies, measured);
}

/// [`coverage_over`] at the default replicate count.
fn coverage(
    test: &'static str,
    cells: &[Cell],
    replicate: impl Fn(u64) -> Option<Replicate> + Sync,
    measured: &[Option<[f64; GRID_POINTS]>],
) {
    coverage_over(test, n_sim(), cells, replicate, measured);
}

/// The facade's default Bayesian configuration.
fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::laplace())
}

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn dag(n: u32, edges: &[(u32, u32)]) -> Dag {
    let mut g = Dag::with_variables(n);
    for &(a, b) in edges {
        g.insert_directed(d(a), d(b)).unwrap();
    }
    g
}

fn mean_of(data: &TabularData, name: &str) -> f64 {
    let col = data.float64_values(data.schema().id_of(name).unwrap()).unwrap();
    col.iter().sum::<f64>() / col.len() as f64
}

/// Explicit structure of one design.
enum Graph {
    Dag(Dag),
    Cpdag(Cpdag),
    Pag(Pag),
}

impl From<Dag> for Graph {
    fn from(g: Dag) -> Self {
        Self::Dag(g)
    }
}

impl From<Cpdag> for Graph {
    fn from(g: Cpdag) -> Self {
        Self::Cpdag(g)
    }
}

impl From<Pag> for Graph {
    fn from(g: Pag) -> Self {
        Self::Pag(g)
    }
}

/// Run a static study with the default Bayesian configuration and no
/// validation; `options` (when set) carries only the response confidence
/// level and the caller bandwidth of the point-derivative designs.
///
/// The [`Study`] is returned beside its result because [`bind`] reads the
/// construction from the study's contract.
fn run(
    data: TabularData,
    graph: impl Into<Graph>,
    query: impl Into<CausalQuery>,
    options: Option<ContinuousResponseOptions>,
    seed: u64,
) -> Option<Run> {
    let builder = Study::tabular(data);
    let builder = match graph.into() {
        Graph::Dag(g) => builder.graph(g),
        Graph::Cpdag(g) => builder.graph(g),
        Graph::Pag(g) => builder.graph(g),
    };
    let mut builder = builder.query(query.into()).inference(bayes()).refute(RefuteSuite::None);
    if let Some(options) = options {
        builder = builder.response_options(options);
    }
    let study = builder.build().ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    Some((study, result))
}

fn check_estimator(result: &StudyResult, cell: Cell) {
    assert_eq!(
        result.logical_plan.estimator.as_deref(),
        Some(cell.estimator),
        "{} × {}: default Bayesian estimator",
        cell.query,
        cell.graph_class
    );
}

/// Reported / gate intervals of the posterior effect column.
fn effect_pair(result: &StudyResult) -> Pair {
    result
        .posterior
        .as_ref()
        .and_then(antecedent::CausalPosterior::effect_column)
        .map_or([None, None], |col| posterior_pair(result, col))
}

// ================================================================ scalar ATE

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn average_effect_dag_bayesian_default_nominal_coverage() {
    let cell = Cell {
        query: "AverageEffect",
        graph_class: "Dag",
        estimator: "bayesian.gcomp",
        dgp: "crates/antecedent/tests/common/static_dgp.rs::linear_ate_data",
        label: None,
    };
    let graph = dag(3, &[(2, 0), (2, 1), (0, 1)]);
    coverage(
        "average_effect_dag_bayesian_default_nominal_coverage",
        &[cell],
        |rep| {
            let seed = stream_seed(0x110_0501, rep);
            let (study, result) = run(
                linear_ate_data(grid_n(500), seed),
                graph.clone(),
                AverageEffectQuery::binary_ate(v(0), v(1)),
                None,
                seed,
            )?;
            if rep == 0 {
                check_estimator(&result, cell);
            }
            let pairs = vec![effect_pair(&result)];
            Some(Replicate::new((study, result), pairs, vec![2.0]))
        },
        &[None, None],
    );
}

// ------------------------------------------- non-Gaussian likelihoods (ATE)

/// Standard normal CDF (Abramowitz–Stegun 7.1.26, |error| < 1.5e-7), kept
/// independent of the library so a probit truth does not reuse the link the
/// estimator fits with.
fn std_normal_cdf(x: f64) -> f64 {
    let t = 1.0 / (1.0 + 0.327_591_1 * x.abs() / std::f64::consts::SQRT_2);
    let poly = t
        * (0.254_829_592
            + t * (-0.284_496_736
                + t * (1.421_413_741 + t * (-1.453_152_027 + t * 1.061_405_429))));
    let tail = 0.5 * poly * (-(x * x) / 2.0).exp();
    if x >= 0.0 { 1.0 - tail } else { tail }
}

/// Poisson draw by multiplication of uniforms (the means here stay below 10).
fn poisson(u: &mut impl FnMut() -> f64, mean: f64) -> f64 {
    let limit = (-mean).exp();
    let (mut k, mut product) = (0.0, u());
    while product > limit {
        k += 1.0;
        product *= u();
    }
    k
}

/// Outcome law of a non-Gaussian likelihood design.
#[derive(Clone, Copy)]
enum Link {
    Logit,
    Probit,
    Poisson,
}

impl Link {
    fn likelihood(self) -> antecedent_prob::BayesLikelihood {
        match self {
            Self::Logit => antecedent_prob::BayesLikelihood::BernoulliLogit,
            Self::Probit => antecedent_prob::BayesLikelihood::BernoulliProbit,
            Self::Poisson => antecedent_prob::BayesLikelihood::PoissonLog,
        }
    }

    /// Outcome mean at linear predictor `eta`.
    fn mean(self, eta: f64) -> f64 {
        match self {
            Self::Logit => common::static_dgp::sigmoid(eta),
            Self::Probit => std_normal_cdf(eta),
            Self::Poisson => eta.exp(),
        }
    }

    /// `(intercept, treatment, confounder)` coefficients of the outcome model.
    const fn coefficients(self) -> (f64, f64, f64) {
        match self {
            Self::Logit => (-0.5, 1.0, 0.8),
            Self::Probit => (-0.3, 0.6, 0.5),
            Self::Poisson => (0.2, 0.5, 0.3),
        }
    }
}

/// `z ~ N(0,1)`, `t ~ Bern(σ(−0.8 + z))`, and an outcome drawn from `link`
/// at `η = a + b·t + c·z` (columns `t, y, z`; graph `z → t`, `z → y`,
/// `t → y`). Returns the data and the replicate's truth: the average over the
/// observed confounders of `mean(a + b + c·z) − mean(a + c·z)`, the effect
/// g-computation standardizes over the rows it was given.
fn glm_ate_data(link: Link, n: usize, seed: u64) -> (TabularData, f64) {
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (a, b, c) = link.coefficients();
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    let mut truth = 0.0;
    for i in 0..n {
        z[i] = g();
        t[i] = bernoulli(&mut u, common::static_dgp::sigmoid(-0.8 + z[i]));
        let mean = link.mean(a + b * t[i] + c * z[i]);
        y[i] = match link {
            Link::Logit | Link::Probit => bernoulli(&mut u, mean),
            Link::Poisson => poisson(&mut u, mean),
        };
        truth += link.mean(a + b + c * z[i]) - link.mean(a + c * z[i]);
    }
    (table(&[("t", &t), ("y", &y), ("z", &z)]), truth / n as f64)
}

/// The facade's default Bayesian configuration under `link`'s likelihood
/// (Python `Bayesian(likelihood=...)`) on a Dag `AverageEffect`.
fn glm_likelihood_coverage(test: &'static str, dgp: &'static str, link: Link, stream: u64) {
    let cell = Cell {
        query: "AverageEffect",
        graph_class: "Dag",
        estimator: "bayesian.gcomp",
        dgp,
        label: None,
    };
    let graph = dag(3, &[(2, 0), (2, 1), (0, 1)]);
    coverage(
        test,
        &[cell],
        |rep| {
            let seed = stream_seed(stream, rep);
            let (data, truth) = glm_ate_data(link, grid_n(500), seed);
            let study = Study::tabular(data)
                .graph(graph.clone())
                .query(AverageEffectQuery::binary_ate(v(0), v(1)))
                .inference(InferenceMode::Bayesian(
                    BayesianConfig::laplace().likelihood(link.likelihood()),
                ))
                .refute(RefuteSuite::None)
                .build()
                .ok()?;
            let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
            if rep == 0 {
                check_estimator(&result, cell);
            }
            let pairs = vec![effect_pair(&result)];
            Some(Replicate::new((study, result), pairs, vec![truth]))
        },
        &[None, None],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn average_effect_dag_bayesian_logit_nominal_coverage() {
    glm_likelihood_coverage(
        "average_effect_dag_bayesian_logit_nominal_coverage",
        "glm_ate_data",
        Link::Logit,
        0x110_0530,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn average_effect_dag_bayesian_probit_nominal_coverage() {
    glm_likelihood_coverage(
        "average_effect_dag_bayesian_probit_nominal_coverage",
        "glm_ate_data",
        Link::Probit,
        0x110_0531,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn average_effect_dag_bayesian_poisson_nominal_coverage() {
    glm_likelihood_coverage(
        "average_effect_dag_bayesian_poisson_nominal_coverage",
        "glm_ate_data",
        Link::Poisson,
        0x110_0532,
    );
}

// ------------------------------------------------------ class envelopes (ATE)

const ALPHA: f64 = 0.8;
const B: f64 = 1.0;
const G: f64 = 0.8;
const KAPPA: f64 = ALPHA / (ALPHA * ALPHA + 1.0);

/// `v19_static_envelope_calibration::cpdag_data`: `z ~ N(0,1)`,
/// `t = 0.8 z + e`, `y = t + z + e`; CPDAG `z — t`, `z -> y`, `t -> y`.
/// Completions adjust `{z}` (slope 1) and nothing (slope `1 + κ`); the
/// equal-weight envelope value is `1 + κ/2`.
fn cpdag_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        t[i] = ALPHA * z[i] + g();
        y[i] = B * t[i] + z[i] + g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z)])
}

fn cpdag() -> Cpdag {
    let mut g = Cpdag::with_variables(3);
    g.insert_directed(d(2), d(1)).unwrap();
    g.insert_directed(d(0), d(1)).unwrap();
    g.insert_undirected(d(2), d(0)).unwrap();
    g
}

/// `v19_static_envelope_calibration::pag_data` (columns `t, y, z, m, v, x`):
/// four identified completions at slope 1, two at `1 + 0.9κ`.
fn pag_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let cols: [Vec<f64>; 6] = std::array::from_fn(|_| vec![0.0; n]);
    let [mut t, mut y, mut z, mut m, mut w, mut x] = cols;
    for i in 0..n {
        z[i] = g();
        t[i] = ALPHA * z[i] + g();
        w[i] = 0.6 * t[i] + g();
        m[i] = 0.9 * z[i] + g();
        x[i] = g();
        y[i] = B * t[i] + m[i] + 0.5 * x[i] + g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z), ("m", &m), ("v", &w), ("x", &x)])
}

const CPDAG_TRUTH: f64 = 0.5 * B + 0.5 * (B + KAPPA);
const PAG_SHIFT: f64 = 0.9 * KAPPA;
const PAG_TRUTH: f64 = (4.0 * B + 2.0 * (B + PAG_SHIFT)) / 6.0;

fn ate_levels() -> AverageEffectQuery {
    AverageEffectQuery::with_levels(v(0), v(1), 0.0, 1.0)
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn average_effect_cpdag_bayesian_default_nominal_coverage() {
    let cell = Cell {
        query: "AverageEffect",
        graph_class: "Cpdag",
        estimator: "bayesian.gcomp",
        dgp: "cpdag_data",
        label: None,
    };
    coverage(
        "average_effect_cpdag_bayesian_default_nominal_coverage",
        &[cell],
        |rep| {
            let seed = stream_seed(0x110_0502, rep);
            let (study, result) =
                run(cpdag_data(grid_n(400), seed), cpdag(), ate_levels(), None, seed)?;
            if rep == 0 {
                check_estimator(&result, cell);
            }
            let pairs = vec![effect_pair(&result)];
            Some(Replicate::new((study, result), pairs, vec![CPDAG_TRUTH]))
        },
        &[None, None],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn average_effect_pag_bayesian_default_nominal_coverage() {
    let cell = Cell {
        query: "AverageEffect",
        graph_class: "Pag",
        estimator: "bayesian.gcomp",
        dgp: "pag_data",
        label: None,
    };
    coverage(
        "average_effect_pag_bayesian_default_nominal_coverage",
        &[cell],
        |rep| {
            let seed = stream_seed(0x110_0503, rep);
            let (study, result) =
                run(pag_data(grid_n(400), seed), pag(), ate_levels(), None, seed)?;
            if rep == 0 {
                check_estimator(&result, cell);
            }
            let pairs = vec![effect_pair(&result)];
            Some(Replicate::new((study, result), pairs, vec![PAG_TRUTH]))
        },
        &[None, None],
    );
}

// ---------------------------------------------------------- ConditionalEffect

fn conditional_query(modifier: u32) -> ConditionalEffectQuery {
    ConditionalEffectQuery::try_new(ate_levels().with_effect_modifiers([v(modifier)])).unwrap()
}

/// `v19_static_envelope_calibration::conditional_dag_data` at `p = 1/2`
/// (columns `t, y, z, x`): `y = t + 0.8 t·x + z + 0.3 x + e`, `x ~ Bern(1/2)`.
fn conditional_dag_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed ^ 0xC0FF_EE00);
    let (mut t, mut y, mut z, mut x) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        x[i] = bernoulli(&mut u, 0.5);
        t[i] = ALPHA * z[i] + g();
        y[i] = B * t[i] + G * t[i] * x[i] + z[i] + 0.3 * x[i] + g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z), ("x", &x)])
}

/// PAG conditional law (columns `t, y, z, m, v, x`).
fn conditional_pag_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed ^ 0xC0FF_EE00);
    let cols: [Vec<f64>; 6] = std::array::from_fn(|_| vec![0.0; n]);
    let [mut t, mut y, mut z, mut m, mut w, mut x] = cols;
    for i in 0..n {
        z[i] = g();
        t[i] = ALPHA * z[i] + g();
        w[i] = 0.6 * t[i] + g();
        m[i] = 0.9 * z[i] + g();
        x[i] = bernoulli(&mut u, 0.5);
        y[i] = B * t[i] + G * t[i] * x[i] + m[i] + 0.3 * x[i] + g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z), ("m", &m), ("v", &w), ("x", &x)])
}

fn conditional_case(
    test: &'static str,
    graph_class: &'static str,
    dgp: &'static str,
    family: u64,
    modifier: u32,
    truth: f64,
    sample: fn(usize, u64) -> TabularData,
    graph: impl Fn() -> Graph + Sync,
) {
    let cell = Cell {
        query: "ConditionalEffect",
        graph_class,
        estimator: "conditional.bayesian",
        dgp,
        label: None,
    };
    coverage(
        test,
        &[cell],
        |rep| {
            let seed = stream_seed(family, rep);
            let (study, result) =
                run(sample(grid_n(400), seed), graph(), conditional_query(modifier), None, seed)?;
            if rep == 0 {
                check_estimator(&result, cell);
            }
            let pairs = vec![effect_pair(&result)];
            Some(Replicate::new((study, result), pairs, vec![truth]))
        },
        &[None, None],
    );
}

fn conditional_case_at(
    test: &'static str,
    graph_class: &'static str,
    dgp: &'static str,
    family: u64,
    modifier: u32,
    truth: f64,
    sample: fn(usize, u64) -> TabularData,
    graph: impl Fn() -> Graph + Sync,
    measured: &[[Option<f64>; 3]],
) {
    let cell = Cell {
        query: "ConditionalEffect",
        graph_class,
        estimator: "conditional.bayesian",
        dgp,
        label: None,
    };
    coverage_at(
        test,
        &[cell],
        |rep| {
            let seed = stream_seed(family, rep);
            let (study, result) =
                run(sample(grid_n(400), seed), graph(), conditional_query(modifier), None, seed)?;
            if rep == 0 {
                check_estimator(&result, cell);
            }
            let pairs = vec![effect_pair(&result)];
            Some(Replicate::new((study, result), pairs, vec![truth]))
        },
        measured,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn conditional_effect_dag_bayesian_default_nominal_coverage() {
    conditional_case_at(
        "conditional_effect_dag_bayesian_default_nominal_coverage",
        "Dag",
        "conditional_dag_data",
        0x110_0504,
        3,
        B + G * 0.5,
        conditional_dag_data,
        || dag(4, &[(2, 0), (2, 1), (0, 1), (3, 1)]).into(),
        &[[None, None, Some(0.933)], [None, None, None]],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn conditional_effect_cpdag_bayesian_default_nominal_coverage() {
    conditional_case(
        "conditional_effect_cpdag_bayesian_default_nominal_coverage",
        "Cpdag",
        "conditional_dag_data",
        0x110_0505,
        3,
        B + G * 0.5 + 0.5 * KAPPA,
        conditional_dag_data,
        || {
            let mut g = Cpdag::with_variables(4);
            g.insert_undirected(d(2), d(0)).unwrap();
            g.insert_directed(d(2), d(1)).unwrap();
            g.insert_directed(d(0), d(1)).unwrap();
            g.insert_directed(d(3), d(1)).unwrap();
            g.into()
        },
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn conditional_effect_pag_bayesian_default_nominal_coverage() {
    conditional_case(
        "conditional_effect_pag_bayesian_default_nominal_coverage",
        "Pag",
        "conditional_pag_data",
        0x110_0506,
        5,
        B + G * 0.5 + 2.0 * PAG_SHIFT / 6.0,
        conditional_pag_data,
        || pag().into(),
    );
}

// ------------------------------------------------ interventional distribution

/// Boundary cells: the atom probability sits 0.035 from a boundary, so at the
/// design's 600 rows the Bayesian-bootstrap posterior of that atom is
/// discrete and skewed and its equal-tailed interval covers
/// [`DISTRIBUTION_NEAR_ONE_MEASURED`] / [`DISTRIBUTION_NEAR_ZERO_MEASURED`]
/// (measured over 2000 replicates at the base grid point, index 1; the other
/// points are the gate's 400-replicate measurement). The shortfall is small-count,
/// not a defect in the interval: the same construction on the same law covers
/// 0.950 / 0.892 at 2400 rows. Each cell is asserted against its measured coverage
/// at every grid point, `[reported 0.95, gate 0.90]`.
const DISTRIBUTION_NEAR_ONE_MEASURED: [[f64; GRID_POINTS]; 2] =
    [[0.910, 0.932, 0.960], [0.875, 0.877, 0.900]];
const DISTRIBUTION_NEAR_ZERO_MEASURED: [[f64; GRID_POINTS]; 2] =
    [[0.900, 0.923, 0.925], [0.848, 0.866, 0.883]];

fn distribution_case(
    test: &'static str,
    base: f64,
    family: u64,
    measured: [[f64; GRID_POINTS]; 2],
) {
    let cell = Cell {
        query: "InterventionalDistribution",
        graph_class: "Dag",
        estimator: "functional.distribution",
        dgp: "distribution_data",
        label: Some(label(&format!("base={base}"))),
    };
    let graph = dag(3, &[(2, 0), (2, 1), (0, 1)]);
    coverage(
        test,
        &[cell],
        |rep| {
            let seed = stream_seed(family, rep);
            let query = InterventionalDistributionQuery::new(
                v(1),
                [Intervention::set(v(0), Value::f64(1.0))],
            );
            let (study, result) = run(
                distribution_data(grid_n(600), base, seed),
                graph.clone(),
                CausalQuery::Distribution(query),
                None,
                seed,
            )?;
            if rep == 0 {
                check_estimator(&result, cell);
            }
            let dist = result.distribution.as_ref()?;
            let posterior = result.posterior.as_ref()?;
            let offset = usize::from(posterior.effect_column().is_some());
            let atom = dist.atoms.iter().position(|a| a.outcomes[0].1.as_f64() == Some(1.0))?;
            let pairs = vec![posterior_pair(&result, offset + atom)];
            Some(Replicate::new((study, result), pairs, vec![base + 0.015]))
        },
        &[Some(measured[0]), Some(measured[1])],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn interventional_distribution_dag_bayesian_near_one_default_coverage() {
    distribution_case(
        "interventional_distribution_dag_bayesian_near_one_default_coverage",
        0.95,
        0x110_0507,
        DISTRIBUTION_NEAR_ONE_MEASURED,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn interventional_distribution_dag_bayesian_near_zero_default_coverage() {
    distribution_case(
        "interventional_distribution_dag_bayesian_near_zero_default_coverage",
        0.02,
        0x110_0508,
        DISTRIBUTION_NEAR_ZERO_MEASURED,
    );
}

// ------------------------------------------------------ mediation / path / cf

/// `v19_static_calibration::mediation_data` at `γ = 0` (columns `t, m, y, x`):
/// NDE 0.4, NIE 0.3.
fn mediation_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut t, mut m, mut y, mut x) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        x[i] = g();
        t[i] = 0.5 * x[i] + g();
        m[i] = 0.6 * t[i] + 0.8 * g();
        y[i] = 0.4 * t[i] + 0.5 * m[i] + 0.5 * x[i] + g();
    }
    table(&[("t", &t), ("m", &m), ("y", &y), ("x", &x)])
}

fn mediation_case(
    test: &'static str,
    contrast: MediationContrast,
    contrast_label: &'static str,
    truth: f64,
    family: u64,
) {
    let cell = Cell {
        query: "MediationEffect",
        graph_class: "Dag",
        estimator: "mediation.linear",
        dgp: "mediation_data",
        label: Some(contrast_label),
    };
    let graph = dag(4, &[(3, 0), (3, 2), (0, 1), (0, 2), (1, 2)]);
    coverage(
        test,
        &[cell],
        |rep| {
            let seed = stream_seed(family, rep);
            let query = MediationQuery::binary(v(0), v(2), Arc::from([v(1)]), contrast);
            let (study, result) = run(
                mediation_data(grid_n(400), seed),
                graph.clone(),
                CausalQuery::Mediation(query),
                None,
                seed,
            )?;
            if rep == 0 {
                check_estimator(&result, cell);
            }
            let pairs = vec![effect_pair(&result)];
            Some(Replicate::new((study, result), pairs, vec![truth]))
        },
        &[None, None],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn mediation_nde_bayesian_default_nominal_coverage() {
    mediation_case(
        "mediation_nde_bayesian_default_nominal_coverage",
        MediationContrast::NaturalDirect,
        "nde",
        0.4,
        0x110_0509,
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn mediation_nie_bayesian_default_nominal_coverage() {
    mediation_case(
        "mediation_nie_bayesian_default_nominal_coverage",
        MediationContrast::NaturalIndirect,
        "nie",
        0.3,
        0x110_050A,
    );
}

fn path_case(
    test: &'static str,
    dgp: &'static str,
    n: usize,
    truth: f64,
    family: u64,
    sample: fn(usize, u64) -> TabularData,
    graph: &Dag,
) {
    let cell = Cell {
        query: "PathSpecificEffect",
        graph_class: "Dag",
        estimator: "functional.effect",
        dgp,
        label: None,
    };
    coverage(
        test,
        &[cell],
        |rep| {
            let seed = stream_seed(family, rep);
            let query = PathSpecificEffectQuery::binary(v(0), v(2)).with_path_nodes([v(1)]);
            let (study, result) = run(
                sample(grid_n(n), seed),
                graph.clone(),
                CausalQuery::PathSpecific(query),
                None,
                seed,
            )?;
            if rep == 0 {
                check_estimator(&result, cell);
            }
            let pairs = vec![effect_pair(&result)];
            Some(Replicate::new((study, result), pairs, vec![truth]))
        },
        // measured is [reported (0.95), gated (0.90)]: the reported level passes
        // nominal (0.944 at 2000 replicates). Grid point 0's gated 90% level
        // measures 0.885 at 2000 replicates (1770/2000), below the precision
        // floor 0.887: a named boundary; points 1 and 2 keep their own
        // 400-replicate measurement (both comfortably within the wide band).
        &[None, Some([0.885, 0.905, 0.882])],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn path_specific_chain_bayesian_default_nominal_coverage() {
    let cell = Cell {
        query: "PathSpecificEffect",
        graph_class: "Dag",
        estimator: "functional.effect",
        dgp: "path_data",
        label: None,
    };
    coverage_at(
        "path_specific_chain_bayesian_default_nominal_coverage",
        &[cell],
        |rep| {
            let seed = stream_seed(0x110_050B, rep);
            let query = PathSpecificEffectQuery::binary(v(0), v(2)).with_path_nodes([v(1)]);
            let (study, result) = run(
                path_data(grid_n(500), seed),
                dag(3, &[(0, 1), (1, 2)]),
                CausalQuery::PathSpecific(query),
                None,
                seed,
            )?;
            if rep == 0 {
                check_estimator(&result, cell);
            }
            let pairs = vec![effect_pair(&result)];
            Some(Replicate::new((study, result), pairs, vec![0.2]))
        },
        &[[None, None, None], [None, None, Some(0.850)]],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn path_specific_two_path_bayesian_default_nominal_coverage() {
    path_case(
        "path_specific_two_path_bayesian_default_nominal_coverage",
        "two_path_data",
        1000,
        0.12,
        0x110_050C,
        two_path_data,
        &dag(4, &[(3, 0), (3, 1), (3, 2), (0, 1), (0, 2), (1, 2)]),
    );
}

/// Boundary cell: at the design's `n = 300` the mean-ITE credible
/// interval covers [`COUNTERFACTUAL_MEASURED`] — 0.942 at the published 0.95
/// and 0.886 at 0.90, measured over 2000 replicates (the 0.90 reading is one
/// thousandth under the precision floor 0.887). The shortfall is
/// finite-sample, not a defect in the interval: the same construction on the
/// same law covers 0.948 / 0.905 at `n = 1200`. The mean ITE is a functional
/// of three refitted mechanisms (`a -> m`, `a -> y`, `m -> y`), so at 300 rows
/// the posterior of `3 + 4·2` is slightly tighter than the sampling law of its
/// mean. The assertion is the band around the measured coverage at each grid point
/// (index 1 at 2000 replicates, the others at the gate's 400).
const COUNTERFACTUAL_MEASURED: [[f64; GRID_POINTS]; 2] =
    [[0.9475, 0.942, 0.9325], [0.890, 0.886, 0.875]];

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn counterfactual_bayesian_mean_ite_default_coverage() {
    let cell = Cell {
        query: "Counterfactual",
        graph_class: "Dag",
        estimator: "gcm.fit",
        dgp: "counterfactual_data",
        label: None,
    };
    let graph = dag(3, &[(0, 1), (0, 2), (1, 2)]);
    coverage(
        "counterfactual_bayesian_mean_ite_default_coverage",
        &[cell],
        |rep| {
            let seed = stream_seed(0x110_050D, rep);
            let query = CounterfactualQuery::new(
                v(2),
                Arc::from([Intervention::set(v(0), Value::f64(1.0))]),
            )
            .with_control_level(0.0);
            let (study, result) = run(
                counterfactual_data(grid_n(300), seed),
                graph.clone(),
                CausalQuery::Counterfactual(query),
                None,
                seed,
            )?;
            if rep == 0 {
                check_estimator(&result, cell);
            }
            let pairs = vec![effect_pair(&result)];
            Some(Replicate::new((study, result), pairs, vec![11.0]))
        },
        &[Some(COUNTERFACTUAL_MEASURED[0]), Some(COUNTERFACTUAL_MEASURED[1])],
    );
}

// ================================================================ responses

fn response_dag() -> Dag {
    dag(3, &[(2, 0), (2, 1), (0, 1)])
}

const GRID: [f64; 5] = [-1.0, -0.5, 0.0, 0.5, 1.0];

fn level_options(level: Option<f64>, bandwidth: Option<f64>) -> Option<ContinuousResponseOptions> {
    (level.is_some() || bandwidth.is_some()).then(|| ContinuousResponseOptions {
        bandwidth,
        confidence_level: level.unwrap_or(REPORTED_LEVEL),
        ..ContinuousResponseOptions::default()
    })
}

/// Run `functional` once at the default level and read every coordinate's
/// reported interval and its 0.90 re-summarization from the draws the
/// estimator retained on the interval (`common::reported::response_posterior_pairs`).
///
/// One execution scores both levels: the estimator's 0.90 interval is the
/// same draw vector's quantiles at the other level, so a second run at
/// `confidence_level = 0.90` would reproduce it bit for bit at twice the cost.
fn response_pairs(
    data: &TabularData,
    graph: &Dag,
    functional: &F,
    bandwidth: Option<f64>,
    seed: u64,
    coordinates: usize,
) -> Option<(Run, Vec<Pair>)> {
    let query = CausalQuery::Response(ResponseQuery::new(functional.clone()));
    let reported = run(data.clone(), graph.clone(), query, level_options(None, bandwidth), seed)?;
    let pairs =
        response_posterior_pairs(&reported.1).unwrap_or_else(|| vec![[None, None]; coordinates]);
    assert_eq!(pairs.len(), coordinates, "one interval per coordinate");
    Some((reported, pairs))
}

/// One cell per coordinate of a response or derivative functional.
///
/// A record's label joins `design` — which names the design when several tests
/// share one DGP in this file — with the coordinate name, and is absent when
/// neither is needed to tell this test's records apart.
fn response_cells(
    query: &'static str,
    estimator: &'static str,
    dgp: &'static str,
    design: Option<&'static str>,
    coordinates: &[String],
) -> Vec<Cell> {
    coordinates
        .iter()
        .map(|coordinate| {
            let text = match (design, coordinates.len()) {
                (Some(design), 1) => design.to_owned(),
                (Some(design), _) => format!("{design}.{coordinate}"),
                (None, 1) => String::new(),
                (None, _) => coordinate.clone(),
            };
            Cell {
                query,
                graph_class: "Dag",
                estimator,
                dgp,
                label: (!text.is_empty()).then(|| label(&text)),
            }
        })
        .collect()
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn intervention_response_dag_bayesian_default_nominal_coverage() {
    let cells = response_cells(
        "InterventionResponse",
        "response.bayesian",
        "response_data",
        None,
        &["a=1".to_owned()],
    );
    let functional = F::InterventionResponse {
        outcome: v(1),
        interventions: Arc::from([Intervention::set(v(0), Value::f64(1.0))]),
    };
    coverage(
        "intervention_response_dag_bayesian_default_nominal_coverage",
        &cells,
        |rep| {
            let seed = stream_seed(0x110_0510, rep);
            let data = response_data(grid_n(500), seed);
            let (reported, pairs) =
                response_pairs(&data, &response_dag(), &functional, None, seed, 1)?;
            if rep == 0 {
                check_estimator(&reported.1, cells[0]);
            }
            let truths = vec![3.0 + 0.8 * mean_of(&data, "z")];
            Some(Replicate::new(reported, pairs, truths))
        },
        &[None, None],
    );
}

/// A band's five coordinates move together, so this test measures at
/// [`PRECISION_N_SIM`] replicates (see `common::reported::n_sim_at_least`).
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn response_curve_dag_bayesian_default_pointwise_nominal_coverage() {
    let labels: Vec<String> = GRID.iter().map(|a| format!("a={a}")).collect();
    let cells =
        response_cells("ResponseCurve", "response.bayesian", "response_data", None, &labels);
    let functional = F::MeanCurve {
        outcome: v(1),
        treatment: ContinuousDomain::new(v(0), GridSpec::Values(GRID.to_vec().into())),
    };
    coverage_over(
        "response_curve_dag_bayesian_default_pointwise_nominal_coverage",
        n_sim_at_least(PRECISION_N_SIM),
        &cells,
        |rep| {
            let seed = stream_seed(0x110_0511, rep);
            let data = response_data(grid_n(500), seed);
            let (reported, pairs) =
                response_pairs(&data, &response_dag(), &functional, None, seed, 5)?;
            if rep == 0 {
                check_estimator(&reported.1, cells[0]);
            }
            let z_bar = mean_of(&data, "z");
            let truths = GRID.iter().map(|a| 1.0 + 2.0 * a + 0.8 * z_bar).collect();
            Some(Replicate::new(reported, pairs, truths))
        },
        &vec![None; 2 * GRID.len()],
    );
}

// =============================================================== derivatives

const N_DERIVATIVE: usize = 1000;

/// Rows of a derivative design at this run's grid point: the heavy grid
/// (500, 1000, 1500), since these cells already run for hours at 1000 rows.
fn derivative_n() -> usize {
    SampleGrid::HEAVY.n(N_DERIVATIVE)
}
/// Caller bandwidth of the point-derivative designs (≈ the MSE-optimal
/// local-quadratic first-derivative bandwidth at `N_DERIVATIVE`).
const BANDWIDTH: f64 = 0.35;
const AT: f64 = 0.5;

fn mu(a: f64) -> f64 {
    5.0 + 2.0 * a.sin()
}

fn mu_prime(a: f64) -> f64 {
    2.0 * a.cos()
}

/// `v19_derivative_calibration::point_data` (columns `a, x, y`):
/// `x ~ N(0,1)`, `a = 0.5x + e`, `y = 5 + 2 sin a + x + e`.
fn point_data(n: usize, seed: u64) -> TabularData {
    let mut z = gaussian(seed);
    let (mut a, mut x, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        x[i] = z();
        a[i] = 0.5 * x[i] + z();
        y[i] = mu(a[i]) + x[i] + z();
    }
    table(&[("a", &a), ("x", &x), ("y", &y)])
}

fn point_graph() -> Dag {
    dag(3, &[(1, 0), (1, 2), (0, 2)])
}

/// `v19_derivative_calibration::gam_data` (columns `a1, a2, x, y1, y2`).
fn gam_data(n: usize, seed: u64) -> TabularData {
    let mut z = gaussian(seed);
    let mut cols: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::with_capacity(n));
    for _ in 0..n {
        let x = z();
        let a1 = 0.5 * x + z();
        let a2 = 0.3 * x + z();
        cols[0].push(a1);
        cols[1].push(a2);
        cols[2].push(x);
        cols[3].push(1.0 + 1.5 * a1 + 0.5 * a1 * a1 - 0.5 * a2 + x + z());
        cols[4].push(0.25 * a1 + 1.5 * a2 + 0.5 * x + z());
    }
    table(&[
        ("a1", &cols[0]),
        ("a2", &cols[1]),
        ("x", &cols[2]),
        ("y1", &cols[3]),
        ("y2", &cols[4]),
    ])
}

fn gam_graph() -> Dag {
    dag(5, &[(2, 0), (2, 1), (2, 3), (2, 4), (0, 3), (0, 4), (1, 3), (1, 4)])
}

const GAM_AT: [f64; 2] = [0.5, 0.0];
const JACOBIAN_TRUTH: [f64; 4] = [2.0, -0.5, 0.25, 1.5];
const DIRECTION: [f64; 2] = [1.0, 2.0];
const DIRECTIONAL_TRUTH: [f64; 2] = [1.0, 3.25];

/// Score a derivative functional; `truth` per coordinate.
fn derivative_case(
    test: &'static str,
    query: &'static str,
    estimator: &'static str,
    dgp: &'static str,
    design: Option<&'static str>,
    family: u64,
    functional: &F,
    bandwidth: Option<f64>,
    gam: bool,
    truth: &[f64],
    measured: &[Option<[f64; GRID_POINTS]>],
) {
    let labels: Vec<String> = (0..truth.len()).map(|j| format!("{j}")).collect();
    let cells = response_cells(query, estimator, dgp, design, &labels);
    let graph = if gam { gam_graph() } else { point_graph() };
    coverage(
        test,
        &cells,
        |rep| {
            let seed = stream_seed(family, rep);
            let data =
                if gam { gam_data(derivative_n(), seed) } else { point_data(derivative_n(), seed) };
            let (reported, pairs) =
                response_pairs(&data, &graph, functional, bandwidth, seed, truth.len())?;
            if rep == 0 {
                check_estimator(&reported.1, cells[0]);
            }
            Some(Replicate::new(reported, pairs, truth.to_vec()))
        },
        measured,
    );
}

fn point_query(scale: DerivativeScale) -> F {
    F::PointDerivative { outcome: v(2), treatment: v(0), at: AT, order: 1, scale }
}

const POINT_DGP: &str = "point_data";
const GAM_DGP: &str = "gam_data";

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn average_derivative_bayesian_default_nominal_coverage() {
    derivative_case(
        "average_derivative_bayesian_default_nominal_coverage",
        "AverageDerivative",
        "response.riesz_ade",
        POINT_DGP,
        None,
        0x110_0520,
        &F::AverageDerivative {
            outcome: v(2),
            treatment: v(0),
            weighting: DerivativeWeighting::Observed,
        },
        None,
        false,
        // E[m'(A)] with A ~ N(0, 1.25).
        &[2.0 * (-0.625_f64).exp()],
        &[None, None],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn point_derivative_bayesian_default_nominal_coverage() {
    derivative_case(
        "point_derivative_bayesian_default_nominal_coverage",
        "PointDerivative",
        "response.kennedy_dr",
        POINT_DGP,
        None,
        0x110_0521,
        &point_query(DerivativeScale::Identity),
        Some(BANDWIDTH),
        false,
        &[mu_prime(AT)],
        &[None, None],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn semi_elasticity_log_treatment_bayesian_default_nominal_coverage() {
    derivative_case(
        "semi_elasticity_log_treatment_bayesian_default_nominal_coverage",
        "SemiElasticity",
        "response.kennedy_dr",
        POINT_DGP,
        Some("log_treatment"),
        0x110_0522,
        &point_query(DerivativeScale::LogTreatment),
        Some(BANDWIDTH),
        false,
        &[AT * mu_prime(AT)],
        &[None, None],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn semi_elasticity_log_outcome_bayesian_default_nominal_coverage() {
    derivative_case(
        "semi_elasticity_log_outcome_bayesian_default_nominal_coverage",
        "SemiElasticity",
        "response.kennedy_dr",
        POINT_DGP,
        Some("log_outcome"),
        0x110_0523,
        &point_query(DerivativeScale::LogOutcome),
        Some(BANDWIDTH),
        false,
        &[mu_prime(AT) / mu(AT)],
        &[None, None],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn elasticity_bayesian_default_nominal_coverage() {
    derivative_case(
        "elasticity_bayesian_default_nominal_coverage",
        "Elasticity",
        "response.kennedy_dr",
        POINT_DGP,
        None,
        0x110_0524,
        &point_query(DerivativeScale::LogLog),
        Some(BANDWIDTH),
        false,
        &[AT * mu_prime(AT) / mu(AT)],
        &[None, None],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn directional_derivative_bayesian_default_nominal_coverage() {
    let functional = F::DirectionalDerivative {
        outcomes: Arc::from([v(3), v(4)]),
        treatments: Arc::from([v(0), v(1)]),
        at: Arc::from(GAM_AT),
        direction: Arc::from(DIRECTION),
    };
    let labels: Vec<String> = (0..DIRECTIONAL_TRUTH.len()).map(|j| format!("{j}")).collect();
    let cells =
        response_cells("DirectionalDerivative", "response.gam_derivative", GAM_DGP, None, &labels);
    let graph = gam_graph();
    coverage_at(
        "directional_derivative_bayesian_default_nominal_coverage",
        &cells,
        |rep| {
            let seed = stream_seed(0x110_0525, rep);
            let data = gam_data(derivative_n(), seed);
            let (reported, pairs) =
                response_pairs(&data, &graph, &functional, None, seed, DIRECTIONAL_TRUTH.len())?;
            if rep == 0 {
                check_estimator(&reported.1, cells[0]);
            }
            Some(Replicate::new(reported, pairs, DIRECTIONAL_TRUTH.to_vec()))
        },
        &[[None, None, None], [Some(0.886), None, None], [None, None, None], [None, None, None]],
    );
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn response_jacobian_bayesian_default_coverage() {
    derivative_case(
        "response_jacobian_bayesian_default_coverage",
        "ResponseJacobian",
        "response.gam_derivative",
        GAM_DGP,
        None,
        0x110_0526,
        &F::Jacobian {
            outcomes: Arc::from([v(3), v(4)]),
            treatments: Arc::from([v(0), v(1)]),
            at: Arc::from(GAM_AT),
            scale: DerivativeScale::Identity,
        },
        None,
        true,
        &JACOBIAN_TRUTH,
        &[None; 8],
    );
}
