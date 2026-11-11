//! Repeated-sampling coverage of the credible interval the Bayesian temporal response
//! reports on derivative, curve and level queries.
//!
//! Cells: `AverageDerivative`, `PointDerivative`, `Elasticity`, `SemiElasticity`,
//! `ResponseJacobian` and `DirectionalDerivative` on a `TemporalDag`, `TemporalCpdag`,
//! `TemporalPag` (given and as a one-atom graph posterior) and a `TemporalAdmg` (given);
//! `ResponseCurve` and `InterventionResponse` on a `TemporalDag`, `TemporalCpdag` and
//! `TemporalPag` (given and as a one-atom graph posterior). All Bayesian (conjugate,
//! 400 draws, slope prior scale 8), estimator `response.temporal.bayesian`, interval
//! `posterior_quantile`: the equal-tailed 0.95 interval of the retained draws of the
//! functional. The 0.90 interval is re-summarized from the same retained draws.
//!
//! One test measures one (query, graph class, structure family); the validation modes
//! and the accepted twin of an explicit graph are sibling executions on the same data,
//! required to report the base interval bit for bit.
//!
//! Every law is linear-Gaussian with iid innovations, so the truth of each reported
//! functional is closed form. Slopes are sample-independent. A level is not: the
//! Bayesian intercept is marginalized over the fitted rows' adjustment covariates, so
//! the declared functional is the level conditional on the observed covariate average
//! of the lag-aligned rows (anchors `s = 1..n`), evaluated with the true coefficients
//! of the outcome's conditional mean. Each replicate's truth is computed from that
//! replicate's regenerated data (the same seed the execution uses):
//!
//! - derivative law (columns `x, b, y, v`, and a witness `r` for a `TemporalPag`):
//!   `x = 2 + .5 e (+ .3 r)`, `b = 1 + .6 e`,
//!   `y(t) = 3 + 2 x(t-1) + .5 b(t-1) + .5 e`, `v(t) = 1 + x(t-1) - b(t-1) + .5 e`.
//!   The slope of `y` in `x` is 2 everywhere; the Jacobian of `y` in `(x, b)` is
//!   `(2, .5)`; the derivative of `(y, v)` along `(1, 2)` is `(3, -1)`. The level at
//!   `x = 2` is `7 + .5 b̄` with `b̄` the mean of `b(t-1)` over the rows, so the
//!   elasticity there is `2·2/level` and the outcome-log semi-elasticity `2/level`.
//! - `TemporalAdmg` derivative law (columns `x, y, z, w`, latent `l = .5 e`):
//!   `w = .5 e + .8 l`, `z = .5 e + .5 w`, `x = 2 + .6 z + .4 e`,
//!   `y(t) = 5 + 2 x(t-1) + 3 z(t-1) + .7 l(t-1) + .5 e`. The route adjusts
//!   `(z(t-1), w(t-1))`; given them `x(t-1)` carries no information about `l(t-1)`, and
//!   `E[l | w] = (.2/.41) w`, so `E[y(t) | x, z, w] = 5 + 2x + 3z + (14/41) w`. The slope
//!   is 2 and the level at `x = 2` is `9 + 3 z̄ + (14/41) w̄` over the rows.
//! - surface laws (columns `t, y` and a covariate `z` on the class graphs):
//!   `y(t) = 1 + 2 t(t-1) + .5 e` on the `TemporalDag` (level `1 + 2a`, no covariate);
//!   `y(t) = 1 + 2 t(t-1) + 1.5 z(t-1) + .5 e` with independent `t`, `z` on the
//!   `TemporalCpdag` (level `1 + 2a + 1.5 z̄`, `z̄` the mean of `z(t-1)` over the rows);
//!   `t = .6 z + e`, `y(t) = 1 + 2 t(t-1) + 1.5 z(t) + .5 e` on the `TemporalPag` (level
//!   `1 + 2a + 1.5 z̄`, `z̄` the mean of `z(t)` over the rows).
//!
//! The class graphs are fully oriented (one completion), so the class routes publish
//! the completion's interval; the graph posteriors carry all mass on one atom.
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;
#[path = "common/grid_cell.rs"]
mod grid_cell;

use std::sync::Arc;

use antecedent::{
    AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study, StudyBuilder, StudyResult,
};
use antecedent_core::{
    CausalQuery, ContinuousDomain, DerivativeScale, DerivativeWeighting, ExecutionContext,
    GridSpec, Intervention, Lag, ResponseFunctional, ResponseQuery, TemporalPolicy,
    TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::{ColumnView, TableView, TimeSeriesData};
use antecedent_discovery::adjacency_mask::set_edge as set_class_edge;
use antecedent_discovery::{
    AdjacencyMask, GraphPosterior, GraphPosteriorAtomKind, SamplingSummary,
    temporal_cpdag_from_dbn_masks, temporal_pag_from_dbn_masks,
};
use antecedent_graph::{
    NodeRef, TemporalAdmg, TemporalCpdag, TemporalDag, TemporalPag, ensure_lagged,
};
use antecedent_prob::InferenceDiagnostics;
use common::calibration::{gaussian, grid_n, stream_seed};
use common::reported::response_posterior_pairs;
use grid_cell::{Cell, IntervalPair, Target, measure, measure_sample_truths};

const BASE_N: usize = 160;
const BURN: usize = 10;
/// Posterior draws of every execution.
const DRAWS: usize = 400;
/// Scale of the isotropic slope prior.
const PRIOR_SCALE: f64 = 8.0;
/// Resamples a `TemporalAdmg` derivative's full validation reads. Read by validation
/// only, so every sibling carries the same count.
const ADMG_VALIDATION_REPLICATES: u32 = 12;
const ESTIMATOR: &str = "response.temporal.bayesian";
const INTERVAL: &str = "posterior_quantile";

/// How the graph reaches the study.
#[derive(Clone, Copy, Debug)]
enum Structure {
    Explicit,
    Accepted,
    Posterior,
}

/// The temporal graph family of a cell.
#[derive(Clone, Copy, Debug)]
enum Class {
    Dag,
    Cpdag,
    Pag,
}

/// One execution of a coordinate: its structure and validation mode.
type Variant = (Structure, RefuteSuite);

/// A given graph under every validation mode, explicit and accepted.
const GIVEN: &[Variant] = &[
    (Structure::Explicit, RefuteSuite::None),
    (Structure::Explicit, RefuteSuite::Cheap),
    (Structure::Explicit, RefuteSuite::Full),
    (Structure::Accepted, RefuteSuite::None),
    (Structure::Accepted, RefuteSuite::Cheap),
    (Structure::Accepted, RefuteSuite::Full),
];

/// A given graph without the unvalidated accepted execution, which the surface suite of
/// the `TemporalDag` curve measures on its own designs.
const GIVEN_WITHOUT_ACCEPTED_UNVALIDATED: &[Variant] = &[
    (Structure::Explicit, RefuteSuite::None),
    (Structure::Explicit, RefuteSuite::Cheap),
    (Structure::Explicit, RefuteSuite::Full),
    (Structure::Accepted, RefuteSuite::Cheap),
    (Structure::Accepted, RefuteSuite::Full),
];

/// A given graph without the unvalidated explicit execution, which the surface suite of
/// the `TemporalDag` level measures on its own designs.
const GIVEN_WITHOUT_EXPLICIT_UNVALIDATED: &[Variant] = &[
    (Structure::Accepted, RefuteSuite::None),
    (Structure::Explicit, RefuteSuite::Cheap),
    (Structure::Explicit, RefuteSuite::Full),
    (Structure::Accepted, RefuteSuite::Cheap),
    (Structure::Accepted, RefuteSuite::Full),
];

/// A given graph under the validating modes only.
const GIVEN_VALIDATED: &[Variant] = &[
    (Structure::Explicit, RefuteSuite::Cheap),
    (Structure::Explicit, RefuteSuite::Full),
    (Structure::Accepted, RefuteSuite::Cheap),
    (Structure::Accepted, RefuteSuite::Full),
];

/// A graph posterior under every validation mode.
const POSTERIOR: &[Variant] = &[
    (Structure::Posterior, RefuteSuite::None),
    (Structure::Posterior, RefuteSuite::Cheap),
    (Structure::Posterior, RefuteSuite::Full),
];

/// A graph posterior under the validating modes only.
const POSTERIOR_VALIDATED: &[Variant] =
    &[(Structure::Posterior, RefuteSuite::Cheap), (Structure::Posterior, RefuteSuite::Full)];

fn var(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn bayes() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS).prior_scale(PRIOR_SCALE))
}

/// `rows` iid `N(0, sd²)` draws of stream `stream` of the replicate keyed by `seed`.
fn column(rows: usize, sd: f64, seed: u64, stream: u64) -> Vec<f64> {
    let mut draw = gaussian(stream_seed(seed, stream));
    (0..rows).map(|_| sd * draw()).collect()
}

fn lagged(values: &[f64], s: usize) -> f64 {
    s.checked_sub(1).map_or(0.0, |i| values[i])
}

/// The columns without their first `BURN` rows.
fn series(columns: &[(&str, &[f64])]) -> TimeSeriesData {
    let trimmed: Vec<(&str, &[f64])> =
        columns.iter().map(|(name, values)| (*name, &values[BURN..])).collect();
    TimeSeriesData::from_f64_columns(trimmed, 1).unwrap()
}

fn data_column(data: &TimeSeriesData, variable: u32) -> Vec<f64> {
    match data.column(var(variable)).unwrap() {
        ColumnView::Float64(column) => column.values.to_vec(),
        _ => panic!("float column"),
    }
}

#[allow(clippy::cast_precision_loss, reason = "row counts are far below 2^52")]
fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Mean of `variable` read one step before the outcome over the lag-aligned rows
/// (anchors `s = 1..n`): the series' values at `0..n-1`.
fn lagged_row_mean(data: &TimeSeriesData, variable: u32) -> f64 {
    let values = data_column(data, variable);
    mean(&values[..values.len() - 1])
}

/// Mean of `variable` read at the outcome's own period over the lag-aligned rows
/// (anchors `s = 1..n`).
fn contemporaneous_row_mean(data: &TimeSeriesData, variable: u32) -> f64 {
    let values = data_column(data, variable);
    mean(&values[1..])
}

fn lag_bit(n: usize, from: usize, to: usize) -> u64 {
    1_u64 << (from * n + to)
}

fn summary(label: &str) -> SamplingSummary {
    SamplingSummary {
        ess: 1.0,
        diagnostics: InferenceDiagnostics::analytic(label),
        rejected_invalid: 0,
    }
}

/// `[reported, gate]` intervals of every published coordinate, re-summarized from the
/// retained draws; `n` empty pairs when the execution publishes no credible interval.
fn pairs(result: &StudyResult, n: usize) -> Vec<IntervalPair> {
    response_posterior_pairs(result).unwrap_or_else(|| vec![[None, None]; n])
}

/// Each replicate's truths from its replicate seed, for a functional whose true value
/// depends on the sample; `None` when every target's own truth is sample-independent.
type SampleTruths<'a> = Option<&'a (dyn Fn(u64) -> Vec<f64> + Sync)>;

/// Measure one coordinate: `study(replicate seed, structure)` is the replicate's data
/// under one structure, `variants[0]` the base execution. With `truths`, each
/// replicate's truths are `truths(replicate seed)`, computed from the data the same
/// seed regenerates.
#[allow(
    clippy::too_many_arguments,
    reason = "one coordinate's identity, design and execution settings"
)]
fn measure_cell(
    test: &'static str,
    dgp: &'static str,
    seed: u64,
    variants: &'static [Variant],
    targets: &[Target],
    truths: SampleTruths<'_>,
    query: &ResponseQuery,
    replicates: u32,
    study: impl Fn(u64, Structure) -> StudyBuilder + Sync,
) {
    let n = targets.len();
    let cell =
        Cell { test, dgp, interval: INTERVAL, estimator: ESTIMATOR, variants: variants.len() };
    let run = |rep: u64, variant: usize| {
        let (structure, suite) = variants[variant];
        let replicate = stream_seed(seed, rep);
        let study = study(replicate, structure)
            .query(CausalQuery::Response(query.clone()))
            .inference(bayes())
            .refute(suite)
            .bootstrap_replicates(replicates)
            .build()
            .unwrap();
        let result = study.run(&ExecutionContext::for_tests(replicate)).unwrap();
        (study, result)
    };
    let score = |result: &StudyResult| pairs(result, n);
    match truths {
        Some(truths) => {
            measure_sample_truths(cell, targets, |rep| truths(stream_seed(seed, rep)), &run, &score)
        }
        None => measure(cell, targets, &run, &score),
    }
}

// ------------------------------------------------------------------ graphs

/// `from(t-1) -> to(t)` for every listed pair of column indices.
fn lagged_dag(edges: &[(usize, usize)]) -> TemporalDag {
    let mut graph = TemporalDag::empty();
    for &(from, to) in edges {
        let from = u32::try_from(from).unwrap();
        let to = u32::try_from(to).unwrap();
        let from = ensure_lagged(&mut graph, var(from), Lag::from_raw(1)).unwrap();
        let to = ensure_lagged(&mut graph, var(to), Lag::CONTEMPORANEOUS).unwrap();
        graph.insert_directed(from, to).unwrap();
    }
    graph
}

/// The lagged template as a PAG with no circles, plus the witness `witness(t-1)` pointing
/// into the treatment `x(t-1)` and not adjacent to an outcome, so every treatment-to-outcome
/// edge is visible.
fn witness_pag(edges: &[(usize, usize)], witness: u32) -> TemporalPag {
    let dag = lagged_dag(edges);
    let mut pag = TemporalPag::empty();
    let mut ids = Vec::new();
    for node in dag.nodes() {
        let NodeRef::Lagged { variable, lag } = *node else {
            panic!("temporal dag nodes are lagged");
        };
        ids.push(pag.add_lagged(variable, lag).unwrap());
    }
    for edge in dag.edges() {
        pag.insert_directed(ids[edge.a.as_usize()], ids[edge.b.as_usize()]).unwrap();
    }
    let witness = pag.add_lagged(var(witness), Lag::from_raw(1)).unwrap();
    let treatment = dag
        .nodes()
        .iter()
        .position(|node| {
            matches!(
                node,
                NodeRef::Lagged { variable, lag } if variable.raw() == 0 && *lag == Lag::from_raw(1)
            )
        })
        .unwrap();
    pag.insert_directed(witness, ids[treatment]).unwrap();
    pag
}

/// All mass on the lagged template over `n` columns as a `TemporalDag` atom.
fn lagged_dag_posterior(n: usize, edges: &[(usize, usize)]) -> GraphPosterior {
    let lag_mask = edges.iter().fold(0_u64, |mask, &(from, to)| mask | lag_bit(n, from, to));
    GraphPosterior::new(
        n,
        vec![1.0],
        vec![0_u64],
        vec![0.0; n * n],
        vec![0.0; n * n],
        summary("temporal_bayesian_derivative_dag_atom"),
    )
    .unwrap()
    .with_lagged_marginals(1, vec![0.0; n * n])
    .unwrap()
    .with_lag_masks(vec![lag_mask])
    .unwrap()
}

/// All mass on one class atom over `n` columns: the lagged template as a `TemporalCpdag`,
/// or (with a witness column) as a `TemporalPag` with the witness into the treatment.
fn lagged_class_posterior(
    n: usize,
    edges: &[(usize, usize)],
    witness: Option<usize>,
) -> GraphPosterior {
    let lag_mask = edges.iter().fold(0_u64, |mask, &(from, to)| mask | lag_bit(n, from, to));
    let adjacency = witness.map_or(AdjacencyMask::EMPTY, |witness| {
        set_class_edge(AdjacencyMask::EMPTY, n, witness, 0, true)
    });
    GraphPosterior::from_atoms(
        n,
        vec![1.0],
        vec![adjacency],
        vec![0.0; n * n],
        vec![0.0; n * n],
        summary("temporal_bayesian_derivative_class_atom"),
    )
    .unwrap()
    .with_atom_kind(if witness.is_some() {
        GraphPosteriorAtomKind::Pag
    } else {
        GraphPosteriorAtomKind::Cpdag
    })
    .with_lagged_marginals(1, vec![0.0; n * n])
    .unwrap()
    .with_lag_masks(vec![lag_mask])
    .unwrap()
}

// ------------------------------------------------------------------ derivatives

/// The derivative functional a test measures.
#[derive(Clone, Copy, Debug)]
enum Derivative {
    Average,
    Point,
    Elasticity,
    SemiElasticity,
    Jacobian,
    Directional,
}

impl Derivative {
    /// Whether the functional reads the outcome's level, whose truth is sample-dependent.
    fn reads_level(self) -> bool {
        matches!(self, Self::Elasticity | Self::SemiElasticity)
    }
}

/// Treatment value the pointwise derivatives are evaluated at: the treatment's mean
/// under both derivative laws.
const AT: f64 = 2.0;
/// Second treatment's evaluation value under the derivative law: its mean.
const AT_B: f64 = 1.0;
/// `dy/dx` under both derivative laws.
const SLOPE: f64 = 2.0;
/// Coefficient of `w(t-1)` in `E[y(t) | x(t-1), z(t-1), w(t-1)]` under the `TemporalAdmg`
/// law: `.7 · Cov(l, w) / Var(w) = .7 · .2 / .41`.
const ADMG_W_COEFFICIENT: f64 = 14.0 / 41.0;

/// Columns of the derivative law without the witness: `x, b, y, v`.
const DERIVATIVE_WIDTH: usize = 4;
/// `x(t-1) -> y(t)`, `b(t-1) -> y(t)`, `x(t-1) -> v(t)`, `b(t-1) -> v(t)`.
const DERIVATIVE_LAGGED: [(usize, usize); 4] = [(0, 2), (1, 2), (0, 3), (1, 3)];

fn spec(history: Option<u32>) -> TemporalResponseSpec {
    TemporalResponseSpec::new(vec![1_u32], TemporalPolicy::pulse(-1), history).unwrap()
}

/// Columns `x, b, y, v` and, with `witness`, an exogenous `r` entering `x`.
fn derivative_series(seed: u64, witness: bool) -> TimeSeriesData {
    let n = grid_n(BASE_N) + BURN;
    let r = column(n, 1.0, seed, 5);
    let gx = column(n, 0.5, seed, 1);
    let x: Vec<f64> =
        (0..n).map(|s| 2.0 + gx[s] + if witness { 0.3 * r[s] } else { 0.0 }).collect();
    let b: Vec<f64> = column(n, 0.6, seed, 2).iter().map(|g| 1.0 + g).collect();
    let ey = column(n, 0.5, seed, 3);
    let ev = column(n, 0.5, seed, 4);
    let y: Vec<f64> =
        (0..n).map(|s| 3.0 + SLOPE * lagged(&x, s) + 0.5 * lagged(&b, s) + ey[s]).collect();
    let v: Vec<f64> = (0..n).map(|s| 1.0 + lagged(&x, s) - lagged(&b, s) + ev[s]).collect();
    if witness {
        series(&[("x", &x), ("b", &b), ("y", &y), ("v", &v), ("r", &r)])
    } else {
        series(&[("x", &x), ("b", &b), ("y", &y), ("v", &v)])
    }
}

/// `E[y(t) | do(x(t-1) = AT)]` conditional on the rows' `b(t-1)` average:
/// `3 + 2·2 + .5 b̄`.
fn derivative_level(data: &TimeSeriesData) -> f64 {
    3.0 + SLOPE * AT + 0.5 * lagged_row_mean(data, 1)
}

fn derivative_study(class: Class, seed: u64, structure: Structure) -> StudyBuilder {
    let base = Study::series(derivative_series(seed, matches!(class, Class::Pag)));
    let edges = &DERIVATIVE_LAGGED;
    let witness = u32::try_from(DERIVATIVE_WIDTH).unwrap();
    match (class, structure) {
        (Class::Dag, Structure::Explicit) => base.graph(lagged_dag(edges)),
        (Class::Dag, Structure::Accepted) => base.graph(AcceptedGraph::from(lagged_dag(edges))),
        (Class::Dag, Structure::Posterior) => {
            base.graph_posterior(lagged_dag_posterior(DERIVATIVE_WIDTH, edges))
        }
        (Class::Cpdag, Structure::Explicit) => {
            base.graph(TemporalCpdag::from_temporal_dag(&lagged_dag(edges)))
        }
        (Class::Cpdag, Structure::Accepted) => {
            base.graph(AcceptedGraph::from(TemporalCpdag::from_temporal_dag(&lagged_dag(edges))))
        }
        (Class::Cpdag, Structure::Posterior) => {
            base.graph_posterior(lagged_class_posterior(DERIVATIVE_WIDTH, edges, None))
        }
        (Class::Pag, Structure::Explicit) => base.graph(witness_pag(edges, witness)),
        (Class::Pag, Structure::Accepted) => {
            base.graph(AcceptedGraph::from(witness_pag(edges, witness)))
        }
        (Class::Pag, Structure::Posterior) => base.graph_posterior(lagged_class_posterior(
            DERIVATIVE_WIDTH + 1,
            edges,
            Some(DERIVATIVE_WIDTH),
        )),
    }
}

/// The scaled derivative's truth at [`AT`] given the outcome's level there.
fn scaled_truth(kind: Derivative, level: f64) -> f64 {
    match kind {
        Derivative::Elasticity => AT * SLOPE / level,
        Derivative::SemiElasticity => SLOPE / level,
        _ => SLOPE,
    }
}

/// The functional over `outcome` and `treatment` (and, for the two multi-coordinate
/// functionals of the derivative law, the second outcome and treatment) and its targets.
/// The slope targets carry their constant truth; an elasticity's target is scored on the
/// replicate's own truth ([`scaled_truth`] of its level), so its `truth` is unread.
fn derivative_functional(
    kind: Derivative,
    outcome: u32,
    treatment: u32,
    second: Option<(u32, u32)>,
) -> (ResponseFunctional, Vec<Target>) {
    let point = |scale| ResponseFunctional::PointDerivative {
        outcome: var(outcome),
        treatment: var(treatment),
        at: AT,
        order: 1,
        scale,
    };
    match kind {
        Derivative::Average => (
            ResponseFunctional::AverageDerivative {
                outcome: var(outcome),
                treatment: var(treatment),
                weighting: DerivativeWeighting::Observed,
            },
            Target::scalar(SLOPE),
        ),
        Derivative::Point => (point(DerivativeScale::Identity), Target::scalar(SLOPE)),
        Derivative::Elasticity => (point(DerivativeScale::LogLog), Target::scalar(0.0)),
        Derivative::SemiElasticity => (point(DerivativeScale::LogOutcome), Target::scalar(0.0)),
        Derivative::Jacobian => match second {
            // `d y / d (x, b)` at `(AT, AT_B)`.
            Some((_, second_treatment)) => (
                ResponseFunctional::Jacobian {
                    outcomes: Arc::from([var(outcome)]),
                    treatments: Arc::from([var(treatment), var(second_treatment)]),
                    at: Arc::from([AT, AT_B]),
                    scale: DerivativeScale::Identity,
                },
                vec![Target::labelled("dy_dx", SLOPE), Target::labelled("dy_db", 0.5)],
            ),
            None => (
                ResponseFunctional::Jacobian {
                    outcomes: Arc::from([var(outcome)]),
                    treatments: Arc::from([var(treatment)]),
                    at: Arc::from([AT]),
                    scale: DerivativeScale::Identity,
                },
                Target::scalar(SLOPE),
            ),
        },
        Derivative::Directional => match second {
            // `(y, v)` along `(1, 2)` in `(x, b)`: `2 + 2·.5` and `1 - 2`.
            Some((second_outcome, second_treatment)) => (
                ResponseFunctional::DirectionalDerivative {
                    outcomes: Arc::from([var(outcome), var(second_outcome)]),
                    treatments: Arc::from([var(treatment), var(second_treatment)]),
                    at: Arc::from([AT, AT_B]),
                    direction: Arc::from([1.0, 2.0]),
                },
                vec![Target::labelled("y", 3.0), Target::labelled("v", -1.0)],
            ),
            None => (
                ResponseFunctional::DirectionalDerivative {
                    outcomes: Arc::from([var(outcome)]),
                    treatments: Arc::from([var(treatment)]),
                    at: Arc::from([AT]),
                    direction: Arc::from([1.0]),
                },
                Target::scalar(SLOPE),
            ),
        },
    }
}

fn derivative_cell(
    test: &'static str,
    kind: Derivative,
    class: Class,
    variants: &'static [Variant],
    seed: u64,
) {
    let (functional, targets) = derivative_functional(kind, 2, 0, Some((3, 1)));
    let query = ResponseQuery::new(functional).with_temporal(spec(Some(1)));
    let witness = matches!(class, Class::Pag);
    let dgp = if witness { "derivative_series_with_witness" } else { "derivative_series" };
    let truths = move |replicate: u64| -> Vec<f64> {
        vec![scaled_truth(kind, derivative_level(&derivative_series(replicate, witness)))]
    };
    measure_cell(
        test,
        dgp,
        seed,
        variants,
        &targets,
        kind.reads_level().then_some(&truths as &(dyn Fn(u64) -> Vec<f64> + Sync)),
        &query,
        0,
        |replicate, structure| derivative_study(class, replicate, structure),
    );
}

/// Columns `x, y, z, w` with a latent `l` entering `w` and, one step later, `y`.
fn admg_derivative_series(seed: u64) -> TimeSeriesData {
    let n = grid_n(BASE_N) + BURN;
    let latent = column(n, 0.5, seed, 1);
    let ew = column(n, 0.5, seed, 2);
    let ez = column(n, 0.5, seed, 3);
    let ex = column(n, 0.4, seed, 4);
    let ey = column(n, 0.5, seed, 5);
    let w: Vec<f64> = (0..n).map(|s| ew[s] + 0.8 * latent[s]).collect();
    let z: Vec<f64> = (0..n).map(|s| ez[s] + 0.5 * w[s]).collect();
    let x: Vec<f64> = (0..n).map(|s| 2.0 + 0.6 * z[s] + ex[s]).collect();
    let y: Vec<f64> = (0..n)
        .map(|s| {
            5.0 + SLOPE * lagged(&x, s) + 3.0 * lagged(&z, s) + 0.7 * lagged(&latent, s) + ey[s]
        })
        .collect();
    series(&[("x", &x), ("y", &y), ("z", &z), ("w", &w)])
}

/// `E[y(t) | do(x(t-1) = AT)]` conditional on the rows' adjustment averages:
/// `5 + 2·2 + 3 z̄ + (14/41) w̄`.
fn admg_level(data: &TimeSeriesData) -> f64 {
    5.0 + SLOPE * AT
        + 3.0 * lagged_row_mean(data, 2)
        + ADMG_W_COEFFICIENT * lagged_row_mean(data, 3)
}

/// `z(t-1) -> x(t-1)`, `z(t-1) -> y(t)`, `x(t-1) -> y(t)`, `w(t-1) -> z(t-1)` and a latent
/// common cause of `w(t-1)` and `y(t)`.
fn admg_template() -> TemporalAdmg {
    let mut graph = TemporalAdmg::empty();
    let x1 = graph.add_lagged(var(0), Lag::from_raw(1)).unwrap();
    let z1 = graph.add_lagged(var(2), Lag::from_raw(1)).unwrap();
    let w1 = graph.add_lagged(var(3), Lag::from_raw(1)).unwrap();
    let y0 = graph.add_lagged(var(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(z1, x1).unwrap();
    graph.insert_directed(z1, y0).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(w1, z1).unwrap();
    graph.insert_bidirected(w1, y0).unwrap();
    graph
}

fn admg_derivative_cell(
    test: &'static str,
    kind: Derivative,
    variants: &'static [Variant],
    seed: u64,
) {
    let (functional, targets) = derivative_functional(kind, 1, 0, None);
    let query = ResponseQuery::new(functional).with_temporal(spec(None));
    let truths = move |replicate: u64| -> Vec<f64> {
        vec![scaled_truth(kind, admg_level(&admg_derivative_series(replicate)))]
    };
    measure_cell(
        test,
        "admg_derivative_series",
        seed,
        variants,
        &targets,
        kind.reads_level().then_some(&truths as &(dyn Fn(u64) -> Vec<f64> + Sync)),
        &query,
        ADMG_VALIDATION_REPLICATES,
        |replicate, structure| {
            let base = Study::series(admg_derivative_series(replicate));
            match structure {
                Structure::Explicit => base.graph(admg_template()),
                Structure::Accepted => base.graph(AcceptedGraph::from(admg_template())),
                Structure::Posterior => {
                    unreachable!("a TemporalAdmg cell is measured on a given graph")
                }
            }
        },
    );
}

// --------------------------------------------------------------- curves and levels

/// The surface a test measures: the dose curve or the level under one intervention.
#[derive(Clone, Copy, Debug)]
enum Surface {
    Curve,
    Level,
}

const DOSES: [f64; 3] = [-1.0, 0.0, 1.0];
/// The intervention level of the `InterventionResponse` cells.
const SET_DOSE: f64 = 1.0;

/// `E[y(t) | do(t(t-1) = a)]` on the `TemporalDag` surface law.
fn level(dose: f64) -> f64 {
    1.0 + 2.0 * dose
}

/// Columns `t, y` on the `TemporalDag`; `t, y, z` on the class graphs.
fn surface_series(class: Class, seed: u64) -> TimeSeriesData {
    let n = grid_n(BASE_N) + BURN;
    let e = column(n, 0.5, seed, 2);
    match class {
        Class::Dag => {
            let t = column(n, 0.8, seed, 1);
            let y: Vec<f64> = (0..n).map(|s| 1.0 + 2.0 * lagged(&t, s) + e[s]).collect();
            series(&[("t", &t), ("y", &y)])
        }
        Class::Cpdag => {
            let t = column(n, 1.0, seed, 1);
            let z = column(n, 1.0, seed, 3);
            let y: Vec<f64> =
                (0..n).map(|s| 1.0 + 2.0 * lagged(&t, s) + 1.5 * lagged(&z, s) + e[s]).collect();
            series(&[("t", &t), ("y", &y), ("z", &z)])
        }
        Class::Pag => {
            let u = column(n, 1.0, seed, 1);
            let z = column(n, 1.0, seed, 3);
            let t: Vec<f64> = (0..n).map(|s| 0.6 * z[s] + u[s]).collect();
            let y: Vec<f64> =
                (0..n).map(|s| 1.0 + 2.0 * lagged(&t, s) + 1.5 * z[s] + e[s]).collect();
            series(&[("t", &t), ("y", &y), ("z", &z)])
        }
    }
}

/// The covariate's contribution to the class laws' level conditional on the rows'
/// covariate average: `1.5 z̄`, with `z(t-1)` on the `TemporalCpdag` and `z(t)` on the
/// `TemporalPag`.
fn surface_covariate_level(class: Class, data: &TimeSeriesData) -> f64 {
    match class {
        Class::Dag => 0.0,
        Class::Cpdag => 1.5 * lagged_row_mean(data, 2),
        Class::Pag => 1.5 * contemporaneous_row_mean(data, 2),
    }
}

fn surface_variables() -> Vec<VariableId> {
    vec![var(0), var(1), var(2)]
}

/// `t(t-1) -> y(t)`.
fn surface_dag() -> TemporalDag {
    lagged_dag(&[(0, 1)])
}

/// `t@1 -> y`, `z@1 -> y` with no contemporaneous structure.
fn surface_cpdag() -> TemporalCpdag {
    let lag = lag_bit(3, 0, 1) | lag_bit(3, 2, 1);
    temporal_cpdag_from_dbn_masks(0, lag, 3, 1, &surface_variables()).unwrap()
}

/// `z -> t`, `z -> y` contemporaneously and `t@1 -> y`.
fn surface_pag() -> TemporalPag {
    let contemporaneous = set_class_edge(set_class_edge(0, 3, 2, 0, true), 3, 2, 1, true)
        .to_word()
        .expect("a three-variable graph fits one word");
    temporal_pag_from_dbn_masks(contemporaneous, lag_bit(3, 0, 1), 0, 3, 1, &surface_variables())
        .unwrap()
}

/// All mass on the lag-one `TemporalDag` atom `t@1 -> y`.
fn surface_dag_posterior() -> GraphPosterior {
    GraphPosterior::from_atoms(
        2,
        vec![1.0],
        vec![0],
        vec![0.0; 4],
        vec![0.0; 4],
        summary("temporal_bayesian_surface_dag_atom"),
    )
    .unwrap()
    .with_lagged_marginals(1, vec![0.0, 1.0, 0.0, 0.0])
    .unwrap()
    .with_lag_masks(vec![lag_bit(2, 0, 1)])
    .unwrap()
}

/// All mass on one class atom: [`surface_cpdag`] or [`surface_pag`].
fn surface_class_posterior(kind: GraphPosteriorAtomKind) -> GraphPosterior {
    let n = 3;
    let (contemporaneous, lag) = match kind {
        GraphPosteriorAtomKind::Pag => {
            (set_class_edge(set_class_edge(0, 3, 2, 0, true), 3, 2, 1, true), lag_bit(3, 0, 1))
        }
        _ => (AdjacencyMask::EMPTY, lag_bit(3, 0, 1) | lag_bit(3, 2, 1)),
    };
    GraphPosterior::from_atoms(
        n,
        vec![1.0],
        vec![contemporaneous],
        vec![0.0; n * n],
        vec![0.0; n * n],
        summary("temporal_bayesian_surface_class_atom"),
    )
    .unwrap()
    .with_atom_kind(kind)
    .with_lagged_marginals(1, vec![0.0; n * n])
    .unwrap()
    .with_lag_masks(vec![lag])
    .unwrap()
}

fn surface_study(class: Class, seed: u64, structure: Structure) -> StudyBuilder {
    let base = Study::series(surface_series(class, seed));
    match (class, structure) {
        (Class::Dag, Structure::Explicit) => base.graph(surface_dag()),
        (Class::Dag, Structure::Accepted) => base.graph(AcceptedGraph::from(surface_dag())),
        (Class::Dag, Structure::Posterior) => base.graph_posterior(surface_dag_posterior()),
        (Class::Cpdag, Structure::Explicit) => base.graph(surface_cpdag()),
        (Class::Cpdag, Structure::Accepted) => base.graph(AcceptedGraph::from(surface_cpdag())),
        (Class::Cpdag, Structure::Posterior) => {
            base.graph_posterior(surface_class_posterior(GraphPosteriorAtomKind::Cpdag))
        }
        (Class::Pag, Structure::Explicit) => base.graph(surface_pag()),
        (Class::Pag, Structure::Accepted) => base.graph(AcceptedGraph::from(surface_pag())),
        (Class::Pag, Structure::Posterior) => {
            base.graph_posterior(surface_class_posterior(GraphPosteriorAtomKind::Pag))
        }
    }
}

/// A surface cell. The `TemporalDag` law has no covariate, so its levels are constants;
/// the class laws' levels carry the replicate's covariate average.
fn surface_cell(
    test: &'static str,
    surface: Surface,
    class: Class,
    variants: &'static [Variant],
    seed: u64,
) {
    let doses: &'static [f64] = match surface {
        Surface::Curve => &DOSES,
        Surface::Level => &[SET_DOSE],
    };
    let (functional, targets) = match surface {
        Surface::Curve => (
            ResponseFunctional::MeanCurve {
                outcome: var(1),
                treatment: ContinuousDomain::new(var(0), GridSpec::Values(Arc::from(DOSES))),
            },
            DOSES.iter().map(|&dose| Target::labelled(format!("a={dose}"), level(dose))).collect(),
        ),
        Surface::Level => (
            ResponseFunctional::InterventionResponse {
                outcome: var(1),
                interventions: Arc::from([Intervention::set(var(0), Value::f64(SET_DOSE))]),
            },
            Target::scalar(level(SET_DOSE)),
        ),
    };
    // The class routes condition on one step of history; the `TemporalDag` route reads
    // the graph's own lags.
    let history = if matches!(class, Class::Dag) { None } else { Some(1) };
    let query = ResponseQuery::new(functional).with_temporal(spec(history));
    let dgp = match class {
        Class::Dag => "surface_series_dag",
        Class::Cpdag => "surface_series_cpdag",
        Class::Pag => "surface_series_pag",
    };
    let truths = move |replicate: u64| -> Vec<f64> {
        let shift = surface_covariate_level(class, &surface_series(class, replicate));
        doses.iter().map(|&dose| level(dose) + shift).collect()
    };
    measure_cell(
        test,
        dgp,
        seed,
        variants,
        &targets,
        (!matches!(class, Class::Dag)).then_some(&truths as &(dyn Fn(u64) -> Vec<f64> + Sync)),
        &query,
        0,
        |replicate, structure| surface_study(class, replicate, structure),
    );
}

// ------------------------------------------------------------------------ cells

/// `AverageDerivative` on a `TemporalDag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_average_derivative_temporal_dag_given() {
    derivative_cell(
        "bayesian_average_derivative_temporal_dag_given",
        Derivative::Average,
        Class::Dag,
        GIVEN,
        1_931_000,
    );
}

/// `AverageDerivative` on a `TemporalDag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_average_derivative_temporal_dag_graph_posterior() {
    derivative_cell(
        "bayesian_average_derivative_temporal_dag_graph_posterior",
        Derivative::Average,
        Class::Dag,
        POSTERIOR,
        1_932_000,
    );
}

/// `AverageDerivative` on a `TemporalCpdag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_average_derivative_temporal_cpdag_given() {
    derivative_cell(
        "bayesian_average_derivative_temporal_cpdag_given",
        Derivative::Average,
        Class::Cpdag,
        GIVEN,
        1_933_000,
    );
}

/// `AverageDerivative` on a `TemporalCpdag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_average_derivative_temporal_cpdag_graph_posterior() {
    derivative_cell(
        "bayesian_average_derivative_temporal_cpdag_graph_posterior",
        Derivative::Average,
        Class::Cpdag,
        POSTERIOR,
        1_934_000,
    );
}

/// `AverageDerivative` on a `TemporalPag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_average_derivative_temporal_pag_given() {
    derivative_cell(
        "bayesian_average_derivative_temporal_pag_given",
        Derivative::Average,
        Class::Pag,
        GIVEN,
        1_935_000,
    );
}

/// `AverageDerivative` on a `TemporalPag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_average_derivative_temporal_pag_graph_posterior() {
    derivative_cell(
        "bayesian_average_derivative_temporal_pag_graph_posterior",
        Derivative::Average,
        Class::Pag,
        POSTERIOR,
        1_936_000,
    );
}

/// `AverageDerivative` on a `TemporalAdmg` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_average_derivative_temporal_admg_given() {
    admg_derivative_cell(
        "bayesian_average_derivative_temporal_admg_given",
        Derivative::Average,
        GIVEN,
        1_937_000,
    );
}

/// `PointDerivative` on a `TemporalDag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_point_derivative_temporal_dag_given() {
    derivative_cell(
        "bayesian_point_derivative_temporal_dag_given",
        Derivative::Point,
        Class::Dag,
        GIVEN,
        1_938_000,
    );
}

/// `PointDerivative` on a `TemporalDag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_point_derivative_temporal_dag_graph_posterior() {
    derivative_cell(
        "bayesian_point_derivative_temporal_dag_graph_posterior",
        Derivative::Point,
        Class::Dag,
        POSTERIOR,
        1_939_000,
    );
}

/// `PointDerivative` on a `TemporalCpdag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_point_derivative_temporal_cpdag_given() {
    derivative_cell(
        "bayesian_point_derivative_temporal_cpdag_given",
        Derivative::Point,
        Class::Cpdag,
        GIVEN,
        1_940_000,
    );
}

/// `PointDerivative` on a `TemporalCpdag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_point_derivative_temporal_cpdag_graph_posterior() {
    derivative_cell(
        "bayesian_point_derivative_temporal_cpdag_graph_posterior",
        Derivative::Point,
        Class::Cpdag,
        POSTERIOR,
        1_941_000,
    );
}

/// `PointDerivative` on a `TemporalPag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_point_derivative_temporal_pag_given() {
    derivative_cell(
        "bayesian_point_derivative_temporal_pag_given",
        Derivative::Point,
        Class::Pag,
        GIVEN,
        1_942_000,
    );
}

/// `PointDerivative` on a `TemporalPag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_point_derivative_temporal_pag_graph_posterior() {
    derivative_cell(
        "bayesian_point_derivative_temporal_pag_graph_posterior",
        Derivative::Point,
        Class::Pag,
        POSTERIOR,
        1_943_000,
    );
}

/// `PointDerivative` on a `TemporalAdmg` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_point_derivative_temporal_admg_given() {
    admg_derivative_cell(
        "bayesian_point_derivative_temporal_admg_given",
        Derivative::Point,
        GIVEN,
        1_944_000,
    );
}

/// `Elasticity` on a `TemporalDag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_elasticity_temporal_dag_given() {
    derivative_cell(
        "bayesian_elasticity_temporal_dag_given",
        Derivative::Elasticity,
        Class::Dag,
        GIVEN,
        1_945_000,
    );
}

/// `Elasticity` on a `TemporalDag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_elasticity_temporal_dag_graph_posterior() {
    derivative_cell(
        "bayesian_elasticity_temporal_dag_graph_posterior",
        Derivative::Elasticity,
        Class::Dag,
        POSTERIOR,
        1_946_000,
    );
}

/// `Elasticity` on a `TemporalCpdag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_elasticity_temporal_cpdag_given() {
    derivative_cell(
        "bayesian_elasticity_temporal_cpdag_given",
        Derivative::Elasticity,
        Class::Cpdag,
        GIVEN,
        1_947_000,
    );
}

/// `Elasticity` on a `TemporalCpdag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_elasticity_temporal_cpdag_graph_posterior() {
    derivative_cell(
        "bayesian_elasticity_temporal_cpdag_graph_posterior",
        Derivative::Elasticity,
        Class::Cpdag,
        POSTERIOR,
        1_948_000,
    );
}

/// `Elasticity` on a `TemporalPag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_elasticity_temporal_pag_given() {
    derivative_cell(
        "bayesian_elasticity_temporal_pag_given",
        Derivative::Elasticity,
        Class::Pag,
        GIVEN,
        1_949_000,
    );
}

/// `Elasticity` on a `TemporalPag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_elasticity_temporal_pag_graph_posterior() {
    derivative_cell(
        "bayesian_elasticity_temporal_pag_graph_posterior",
        Derivative::Elasticity,
        Class::Pag,
        POSTERIOR,
        1_950_000,
    );
}

/// `Elasticity` on a `TemporalAdmg` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_elasticity_temporal_admg_given() {
    admg_derivative_cell(
        "bayesian_elasticity_temporal_admg_given",
        Derivative::Elasticity,
        GIVEN,
        1_951_000,
    );
}

/// `SemiElasticity` on a `TemporalDag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_semi_elasticity_temporal_dag_given() {
    derivative_cell(
        "bayesian_semi_elasticity_temporal_dag_given",
        Derivative::SemiElasticity,
        Class::Dag,
        GIVEN,
        1_952_000,
    );
}

/// `SemiElasticity` on a `TemporalDag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_semi_elasticity_temporal_dag_graph_posterior() {
    derivative_cell(
        "bayesian_semi_elasticity_temporal_dag_graph_posterior",
        Derivative::SemiElasticity,
        Class::Dag,
        POSTERIOR,
        1_953_000,
    );
}

/// `SemiElasticity` on a `TemporalCpdag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_semi_elasticity_temporal_cpdag_given() {
    derivative_cell(
        "bayesian_semi_elasticity_temporal_cpdag_given",
        Derivative::SemiElasticity,
        Class::Cpdag,
        GIVEN,
        1_954_000,
    );
}

/// `SemiElasticity` on a `TemporalCpdag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_semi_elasticity_temporal_cpdag_graph_posterior() {
    derivative_cell(
        "bayesian_semi_elasticity_temporal_cpdag_graph_posterior",
        Derivative::SemiElasticity,
        Class::Cpdag,
        POSTERIOR,
        1_955_000,
    );
}

/// `SemiElasticity` on a `TemporalPag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_semi_elasticity_temporal_pag_given() {
    derivative_cell(
        "bayesian_semi_elasticity_temporal_pag_given",
        Derivative::SemiElasticity,
        Class::Pag,
        GIVEN,
        1_956_000,
    );
}

/// `SemiElasticity` on a `TemporalPag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_semi_elasticity_temporal_pag_graph_posterior() {
    derivative_cell(
        "bayesian_semi_elasticity_temporal_pag_graph_posterior",
        Derivative::SemiElasticity,
        Class::Pag,
        POSTERIOR,
        1_957_000,
    );
}

/// `SemiElasticity` on a `TemporalAdmg` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_semi_elasticity_temporal_admg_given() {
    admg_derivative_cell(
        "bayesian_semi_elasticity_temporal_admg_given",
        Derivative::SemiElasticity,
        GIVEN,
        1_958_000,
    );
}

/// `ResponseJacobian` on a `TemporalDag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_response_jacobian_temporal_dag_given() {
    derivative_cell(
        "bayesian_response_jacobian_temporal_dag_given",
        Derivative::Jacobian,
        Class::Dag,
        GIVEN,
        1_959_000,
    );
}

/// `ResponseJacobian` on a `TemporalDag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_response_jacobian_temporal_dag_graph_posterior() {
    derivative_cell(
        "bayesian_response_jacobian_temporal_dag_graph_posterior",
        Derivative::Jacobian,
        Class::Dag,
        POSTERIOR,
        1_960_000,
    );
}

/// `ResponseJacobian` on a `TemporalCpdag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_response_jacobian_temporal_cpdag_given() {
    derivative_cell(
        "bayesian_response_jacobian_temporal_cpdag_given",
        Derivative::Jacobian,
        Class::Cpdag,
        GIVEN,
        1_961_000,
    );
}

/// `ResponseJacobian` on a `TemporalCpdag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_response_jacobian_temporal_cpdag_graph_posterior() {
    derivative_cell(
        "bayesian_response_jacobian_temporal_cpdag_graph_posterior",
        Derivative::Jacobian,
        Class::Cpdag,
        POSTERIOR,
        1_962_000,
    );
}

/// `ResponseJacobian` on a `TemporalPag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_response_jacobian_temporal_pag_given() {
    derivative_cell(
        "bayesian_response_jacobian_temporal_pag_given",
        Derivative::Jacobian,
        Class::Pag,
        GIVEN,
        1_963_000,
    );
}

/// `ResponseJacobian` on a `TemporalPag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_response_jacobian_temporal_pag_graph_posterior() {
    derivative_cell(
        "bayesian_response_jacobian_temporal_pag_graph_posterior",
        Derivative::Jacobian,
        Class::Pag,
        POSTERIOR,
        1_964_000,
    );
}

/// `ResponseJacobian` on a `TemporalAdmg` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_response_jacobian_temporal_admg_given() {
    admg_derivative_cell(
        "bayesian_response_jacobian_temporal_admg_given",
        Derivative::Jacobian,
        GIVEN,
        1_965_000,
    );
}

/// `DirectionalDerivative` on a `TemporalDag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_directional_derivative_temporal_dag_given() {
    derivative_cell(
        "bayesian_directional_derivative_temporal_dag_given",
        Derivative::Directional,
        Class::Dag,
        GIVEN,
        1_966_000,
    );
}

/// `DirectionalDerivative` on a `TemporalDag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_directional_derivative_temporal_dag_graph_posterior() {
    derivative_cell(
        "bayesian_directional_derivative_temporal_dag_graph_posterior",
        Derivative::Directional,
        Class::Dag,
        POSTERIOR,
        1_967_000,
    );
}

/// `DirectionalDerivative` on a `TemporalCpdag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_directional_derivative_temporal_cpdag_given() {
    derivative_cell(
        "bayesian_directional_derivative_temporal_cpdag_given",
        Derivative::Directional,
        Class::Cpdag,
        GIVEN,
        1_968_000,
    );
}

/// `DirectionalDerivative` on a `TemporalCpdag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_directional_derivative_temporal_cpdag_graph_posterior() {
    derivative_cell(
        "bayesian_directional_derivative_temporal_cpdag_graph_posterior",
        Derivative::Directional,
        Class::Cpdag,
        POSTERIOR,
        1_969_000,
    );
}

/// `DirectionalDerivative` on a `TemporalPag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_directional_derivative_temporal_pag_given() {
    derivative_cell(
        "bayesian_directional_derivative_temporal_pag_given",
        Derivative::Directional,
        Class::Pag,
        GIVEN,
        1_970_000,
    );
}

/// `DirectionalDerivative` on a `TemporalPag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_directional_derivative_temporal_pag_graph_posterior() {
    derivative_cell(
        "bayesian_directional_derivative_temporal_pag_graph_posterior",
        Derivative::Directional,
        Class::Pag,
        POSTERIOR,
        1_971_000,
    );
}

/// `DirectionalDerivative` on a `TemporalAdmg` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_directional_derivative_temporal_admg_given() {
    admg_derivative_cell(
        "bayesian_directional_derivative_temporal_admg_given",
        Derivative::Directional,
        GIVEN,
        1_972_000,
    );
}

/// `ResponseCurve` on a `TemporalDag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_response_curve_temporal_dag_given() {
    surface_cell(
        "bayesian_response_curve_temporal_dag_given",
        Surface::Curve,
        Class::Dag,
        GIVEN_WITHOUT_ACCEPTED_UNVALIDATED,
        1_973_000,
    );
}

/// `ResponseCurve` on a `TemporalDag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_response_curve_temporal_dag_graph_posterior() {
    surface_cell(
        "bayesian_response_curve_temporal_dag_graph_posterior",
        Surface::Curve,
        Class::Dag,
        POSTERIOR_VALIDATED,
        1_974_000,
    );
}

/// `ResponseCurve` on a `TemporalCpdag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_response_curve_temporal_cpdag_given() {
    surface_cell(
        "bayesian_response_curve_temporal_cpdag_given",
        Surface::Curve,
        Class::Cpdag,
        GIVEN_VALIDATED,
        1_975_000,
    );
}

/// `ResponseCurve` on a `TemporalCpdag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_response_curve_temporal_cpdag_graph_posterior() {
    surface_cell(
        "bayesian_response_curve_temporal_cpdag_graph_posterior",
        Surface::Curve,
        Class::Cpdag,
        POSTERIOR_VALIDATED,
        1_976_000,
    );
}

/// `ResponseCurve` on a `TemporalPag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_response_curve_temporal_pag_given() {
    surface_cell(
        "bayesian_response_curve_temporal_pag_given",
        Surface::Curve,
        Class::Pag,
        GIVEN_VALIDATED,
        1_977_000,
    );
}

/// `ResponseCurve` on a `TemporalPag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_response_curve_temporal_pag_graph_posterior() {
    surface_cell(
        "bayesian_response_curve_temporal_pag_graph_posterior",
        Surface::Curve,
        Class::Pag,
        POSTERIOR_VALIDATED,
        1_978_000,
    );
}

/// `InterventionResponse` on a `TemporalDag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_intervention_response_temporal_dag_given() {
    surface_cell(
        "bayesian_intervention_response_temporal_dag_given",
        Surface::Level,
        Class::Dag,
        GIVEN_WITHOUT_EXPLICIT_UNVALIDATED,
        1_979_000,
    );
}

/// `InterventionResponse` on a `TemporalDag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_intervention_response_temporal_dag_graph_posterior() {
    surface_cell(
        "bayesian_intervention_response_temporal_dag_graph_posterior",
        Surface::Level,
        Class::Dag,
        POSTERIOR_VALIDATED,
        1_980_000,
    );
}

/// `InterventionResponse` on a `TemporalCpdag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_intervention_response_temporal_cpdag_given() {
    surface_cell(
        "bayesian_intervention_response_temporal_cpdag_given",
        Surface::Level,
        Class::Cpdag,
        GIVEN_VALIDATED,
        1_981_000,
    );
}

/// `InterventionResponse` on a `TemporalCpdag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_intervention_response_temporal_cpdag_graph_posterior() {
    surface_cell(
        "bayesian_intervention_response_temporal_cpdag_graph_posterior",
        Surface::Level,
        Class::Cpdag,
        POSTERIOR_VALIDATED,
        1_982_000,
    );
}

/// `InterventionResponse` on a `TemporalPag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_intervention_response_temporal_pag_given() {
    surface_cell(
        "bayesian_intervention_response_temporal_pag_given",
        Surface::Level,
        Class::Pag,
        GIVEN_VALIDATED,
        1_983_000,
    );
}

/// `InterventionResponse` on a `TemporalPag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn bayesian_intervention_response_temporal_pag_graph_posterior() {
    surface_cell(
        "bayesian_intervention_response_temporal_pag_graph_posterior",
        Surface::Level,
        Class::Pag,
        POSTERIOR_VALIDATED,
        1_984_000,
    );
}
