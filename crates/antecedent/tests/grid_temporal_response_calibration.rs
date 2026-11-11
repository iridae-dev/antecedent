//! Repeated-sampling coverage of the circular-block interval the temporal response
//! g-computation reports on derivative, curve and level queries.
//!
//! Cells: `AverageDerivative`, `PointDerivative`, `Elasticity`, `SemiElasticity`,
//! `ResponseJacobian` and `DirectionalDerivative` on a `TemporalDag`, `TemporalCpdag`,
//! `TemporalPag` (given and as a one-atom graph posterior) and a `TemporalAdmg` (given);
//! `ResponseCurve` and `InterventionResponse` on a `TemporalDag`, `TemporalCpdag` and
//! `TemporalPag` (given and as a one-atom graph posterior). All Frequentist, estimator
//! `temporal.response.gcomp`, interval `circular_block_se`: the joint circular-block
//! bootstrap of lag-aligned tuples with 199 replicates, published as `estimate ± z·se`
//! at 0.95. The 0.90 interval is the same construction at the gate level, formed from
//! the published midpoint and the SE the published half-width states.
//!
//! One test measures one (query, graph class, structure family); the validation modes
//! and the accepted twin of an explicit graph are sibling executions on the same data,
//! required to report the base interval bit for bit.
//!
//! Every law is linear-Gaussian with iid innovations, so the truth of each reported
//! functional is closed form:
//!
//! - derivative law (columns `x, b, y, v`, and a witness `r` for a `TemporalPag`):
//!   `x = 2 + .5 e (+ .3 r)`, `b = 1 + .6 e`,
//!   `y(t) = 3 + 2 x(t-1) + .5 b(t-1) + .5 e`, `v(t) = 1 + x(t-1) - b(t-1) + .5 e`.
//!   The slope of `y` in `x` is 2 everywhere; the level at `x = 2` is
//!   `3 + 4 + .5 E[b] = 7.5`, so the elasticity there is `2·2/7.5` and the
//!   outcome-log semi-elasticity `2/7.5`; the Jacobian of `y` in `(x, b)` is
//!   `(2, .5)`; the derivative of `(y, v)` along `(1, 2)` is `(3, -1)`.
//! - `TemporalAdmg` derivative law (columns `x, y, z, w`, latent `l`):
//!   `w = .5 e + .8 l`, `z = .5 e + .5 w`, `x = 2 + .6 z + .4 e`,
//!   `y(t) = 5 + 2 x(t-1) + 3 z(t-1) + .7 l(t-1) + .5 e`, `l = .5 e`. The slope is 2
//!   and the level at `x = 2` is `5 + 4 = 9`.
//! - surface laws (columns `t, y` and a covariate `z` on the class graphs):
//!   `y(t) = 1 + 2 t(t-1) + .5 e` on the `TemporalDag`;
//!   `y(t) = 1 + 2 t(t-1) + 1.5 z(t-1) + .5 e` with independent `t`, `z` on the
//!   `TemporalCpdag`; `t = .6 z + e`, `y(t) = 1 + 2 t(t-1) + 1.5 z(t) + .5 e` on the
//!   `TemporalPag`. `E[z] = 0`, so the level under `do(t(t-1) = a)` is `1 + 2a`.
//!
//! The class graphs are fully oriented (one completion), so the class routes publish
//! the completion's band; the graph posteriors carry all mass on one atom.
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;
#[path = "common/grid_cell.rs"]
mod grid_cell;

use std::sync::Arc;

use antecedent::{AcceptedGraph, InferenceMode, RefuteSuite, Study, StudyBuilder, StudyResult};
use antecedent_core::{
    CausalQuery, ContinuousDomain, DerivativeScale, DerivativeWeighting, ExecutionContext,
    GridSpec, Intervention, Lag, ResponseFunctional, ResponseQuery, ResponseUncertainty,
    TemporalPolicy, TemporalResponseSpec, Value, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_discovery::adjacency_mask::set_edge as set_class_edge;
use antecedent_discovery::{
    AdjacencyMask, GraphPosterior, GraphPosteriorAtomKind, SamplingSummary, set_edge,
    temporal_cpdag_from_dbn_masks, temporal_pag_from_dbn_masks,
};
use antecedent_graph::{
    NodeRef, TemporalAdmg, TemporalCpdag, TemporalDag, TemporalPag, ensure_lagged,
};
use antecedent_prob::InferenceDiagnostics;
use common::calibration::{REPORTED_LEVEL, Z95, gaussian, grid_n, stream_seed};
use common::reported::{GATE_LEVEL, normal_at, response_normal_pair};
use grid_cell::{Cell, IntervalPair, Target, measure};

const BASE_N: usize = 160;
const BURN: usize = 10;
const BOOT: u32 = 199;
const ESTIMATOR: &str = "temporal.response.gcomp";
const INTERVAL: &str = "circular_block_se";

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

/// A given graph without the unvalidated explicit execution, which the surface suite
/// of the `TemporalDag` response measures on its own designs.
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

/// `[reported, gate]` intervals of every published coordinate; `n` empty pairs when the
/// execution publishes neither a scalar interval nor a pointwise band.
fn pairs(result: &StudyResult, n: usize) -> Vec<IntervalPair> {
    let Some(response) = result.response() else {
        return vec![[None, None]; n];
    };
    match &response.uncertainty {
        ResponseUncertainty::Scalar { .. } => vec![response_normal_pair(result)],
        ResponseUncertainty::PointwiseBand { level, lower, upper, .. } => {
            assert!((level - REPORTED_LEVEL).abs() < 1e-12, "response band level {level}");
            lower
                .iter()
                .zip(upper.iter())
                .map(|(&lo, &hi)| {
                    let se = (hi - lo) / (2.0 * Z95);
                    [Some((lo, hi)), normal_at(0.5 * (lo + hi), se, GATE_LEVEL)]
                })
                .collect()
        }
        _ => vec![[None, None]; n],
    }
}

/// Measure one coordinate: `study(replicate seed, structure)` is the replicate's data
/// under one structure, `variants[0]` the base execution.
fn measure_cell(
    test: &'static str,
    dgp: &'static str,
    seed: u64,
    variants: &'static [Variant],
    targets: &[Target],
    query: &ResponseQuery,
    study: impl Fn(u64, Structure) -> StudyBuilder + Sync,
) {
    let n = targets.len();
    measure(
        Cell { test, dgp, interval: INTERVAL, estimator: ESTIMATOR, variants: variants.len() },
        targets,
        |rep, variant| {
            let (structure, suite) = variants[variant];
            let replicate = stream_seed(seed, rep);
            let study = study(replicate, structure)
                .query(CausalQuery::Response(query.clone()))
                .inference(InferenceMode::Frequentist)
                .refute(suite)
                .bootstrap_replicates(BOOT)
                .build()
                .unwrap();
            let result = study.run(&ExecutionContext::for_tests(replicate)).unwrap();
            (study, result)
        },
        |result| pairs(result, n),
    );
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

/// Treatment value the pointwise derivatives are evaluated at: the treatment's mean
/// under both derivative laws.
const AT: f64 = 2.0;
/// Second treatment's evaluation value under the derivative law: its mean.
const AT_B: f64 = 1.0;
/// `dy/dx` under both derivative laws.
const SLOPE: f64 = 2.0;
/// `E[y(t) | do(x(t-1) = AT)]` under the derivative law: `3 + 2·2 + .5·E[b]`.
const LEVEL_AT: f64 = 7.5;
/// `E[y(t) | do(x(t-1) = AT)]` under the `TemporalAdmg` law: `5 + 2·2`.
const ADMG_LEVEL_AT: f64 = 9.0;

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

/// `x(t-1) -> y(t)`, `b(t-1) -> y(t)`, `x(t-1) -> v(t)`, `b(t-1) -> v(t)`.
const DERIVATIVE_LAGGED: [(usize, usize); 4] = [(0, 2), (1, 2), (0, 3), (1, 3)];

fn derivative_dag() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, var(0), Lag::from_raw(1)).unwrap();
    let b1 = ensure_lagged(&mut graph, var(1), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, var(2), Lag::CONTEMPORANEOUS).unwrap();
    let v0 = ensure_lagged(&mut graph, var(3), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(b1, y0).unwrap();
    graph.insert_directed(x1, v0).unwrap();
    graph.insert_directed(b1, v0).unwrap();
    graph
}

/// The lagged template as a PAG with no circles, plus the witness `r(t-1)` pointing into
/// the treatment and not adjacent to an outcome, so every treatment-to-outcome edge is
/// visible.
fn derivative_pag() -> TemporalPag {
    let dag = derivative_dag();
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
    let r1 = pag.add_lagged(var(4), Lag::from_raw(1)).unwrap();
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
    pag.insert_directed(r1, ids[treatment]).unwrap();
    pag
}

/// All mass on the lagged template as a `TemporalDag` atom.
fn derivative_dag_posterior() -> GraphPosterior {
    let n = 4;
    let lag_mask = DERIVATIVE_LAGGED.iter().fold(0_u64, |mask, &(f, to)| mask | lag_bit(n, f, to));
    GraphPosterior::new(
        n,
        vec![1.0],
        vec![0_u64],
        vec![0.0; n * n],
        vec![0.0; n * n],
        summary("temporal_response_derivative_dag_atom"),
    )
    .unwrap()
    .with_lagged_marginals(1, vec![0.0; n * n])
    .unwrap()
    .with_lag_masks(vec![lag_mask])
    .unwrap()
}

/// All mass on one class atom: the lagged template as a `TemporalCpdag`, or as a
/// `TemporalPag` with the witness `r` into the treatment.
fn derivative_class_posterior(pag: bool) -> GraphPosterior {
    let n = if pag { 5 } else { 4 };
    let lag_mask = DERIVATIVE_LAGGED.iter().fold(0_u64, |mask, &(f, to)| mask | lag_bit(n, f, to));
    let mut adjacency = AdjacencyMask::EMPTY;
    if pag {
        adjacency = set_class_edge(adjacency, n, 4, 0, true);
    }
    GraphPosterior::from_atoms(
        n,
        vec![1.0],
        vec![adjacency],
        vec![0.0; n * n],
        vec![0.0; n * n],
        summary("temporal_response_derivative_class_atom"),
    )
    .unwrap()
    .with_atom_kind(if pag { GraphPosteriorAtomKind::Pag } else { GraphPosteriorAtomKind::Cpdag })
    .with_lagged_marginals(1, vec![0.0; n * n])
    .unwrap()
    .with_lag_masks(vec![lag_mask])
    .unwrap()
}

fn derivative_study(class: Class, seed: u64, structure: Structure) -> StudyBuilder {
    let base = Study::series(derivative_series(seed, matches!(class, Class::Pag)));
    match (class, structure) {
        (Class::Dag, Structure::Explicit) => base.graph(derivative_dag()),
        (Class::Dag, Structure::Accepted) => base.graph(AcceptedGraph::from(derivative_dag())),
        (Class::Dag, Structure::Posterior) => base.graph_posterior(derivative_dag_posterior()),
        (Class::Cpdag, Structure::Explicit) => {
            base.graph(TemporalCpdag::from_temporal_dag(&derivative_dag()))
        }
        (Class::Cpdag, Structure::Accepted) => {
            base.graph(AcceptedGraph::from(TemporalCpdag::from_temporal_dag(&derivative_dag())))
        }
        (Class::Cpdag, Structure::Posterior) => {
            base.graph_posterior(derivative_class_posterior(false))
        }
        (Class::Pag, Structure::Explicit) => base.graph(derivative_pag()),
        (Class::Pag, Structure::Accepted) => base.graph(AcceptedGraph::from(derivative_pag())),
        (Class::Pag, Structure::Posterior) => {
            base.graph_posterior(derivative_class_posterior(true))
        }
    }
}

/// The functional over `outcome` and `treatment` (and, for the two multi-coordinate
/// functionals of the derivative law, the second outcome and treatment) and the truth
/// of each published coordinate. `level` is the outcome's level at [`AT`].
fn derivative_functional(
    kind: Derivative,
    outcome: u32,
    treatment: u32,
    level: f64,
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
        Derivative::Elasticity => {
            (point(DerivativeScale::LogLog), Target::scalar(AT * SLOPE / level))
        }
        Derivative::SemiElasticity => {
            (point(DerivativeScale::LogOutcome), Target::scalar(SLOPE / level))
        }
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
    let (functional, targets) = derivative_functional(kind, 2, 0, LEVEL_AT, Some((3, 1)));
    let query = ResponseQuery::new(functional).with_temporal(spec(Some(1)));
    let dgp = if matches!(class, Class::Pag) {
        "derivative_series_with_witness"
    } else {
        "derivative_series"
    };
    measure_cell(test, dgp, seed, variants, &targets, &query, |replicate, structure| {
        derivative_study(class, replicate, structure)
    });
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
    let (functional, targets) = derivative_functional(kind, 1, 0, ADMG_LEVEL_AT, None);
    let query = ResponseQuery::new(functional).with_temporal(spec(None));
    measure_cell(
        test,
        "admg_derivative_series",
        seed,
        variants,
        &targets,
        &query,
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

/// `E[y(t) | do(t(t-1) = a)]` under every surface law.
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

fn surface_variables() -> Vec<VariableId> {
    vec![var(0), var(1), var(2)]
}

/// `t(t-1) -> y(t)`.
fn surface_dag() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let t1 = ensure_lagged(&mut graph, var(0), Lag::from_raw(1)).unwrap();
    let y0 = ensure_lagged(&mut graph, var(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(t1, y0).unwrap();
    graph
}

/// `t@1 -> y`, `z@1 -> y` with no contemporaneous structure.
fn surface_cpdag() -> TemporalCpdag {
    let lag = lag_bit(3, 0, 1) | lag_bit(3, 2, 1);
    temporal_cpdag_from_dbn_masks(0, lag, 3, 1, &surface_variables()).unwrap()
}

/// `z -> t`, `z -> y` contemporaneously and `t@1 -> y`.
fn surface_pag() -> TemporalPag {
    let contemporaneous = set_edge(set_edge(0, 3, 2, 0, true), 3, 2, 1, true);
    temporal_pag_from_dbn_masks(contemporaneous, lag_bit(3, 0, 1), 0, 3, 1, &surface_variables())
        .unwrap()
}

/// All mass on the lag-one `TemporalDag` atom `t@1 -> y`.
fn surface_dag_posterior() -> GraphPosterior {
    GraphPosterior::from_atoms(
        2,
        vec![1.0],
        vec![0_u64],
        vec![0.0; 4],
        vec![0.0; 4],
        summary("temporal_response_surface_dag_atom"),
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
            (set_edge(set_edge(0, 3, 2, 0, true), 3, 2, 1, true), lag_bit(3, 0, 1))
        }
        _ => (0_u64, lag_bit(3, 0, 1) | lag_bit(3, 2, 1)),
    };
    GraphPosterior::from_atoms(
        n,
        vec![1.0],
        vec![contemporaneous],
        vec![0.0; n * n],
        vec![0.0; n * n],
        summary("temporal_response_surface_class_atom"),
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

fn surface_cell(
    test: &'static str,
    surface: Surface,
    class: Class,
    variants: &'static [Variant],
    seed: u64,
) {
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
                interventions: Arc::from([Intervention::set(var(0), Value::f64(1.0))]),
            },
            Target::scalar(level(1.0)),
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
    measure_cell(test, dgp, seed, variants, &targets, &query, |replicate, structure| {
        surface_study(class, replicate, structure)
    });
}

// ------------------------------------------------------------------------ cells

/// `AverageDerivative` on a `TemporalDag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_average_derivative_temporal_dag_given() {
    derivative_cell(
        "gcomp_average_derivative_temporal_dag_given",
        Derivative::Average,
        Class::Dag,
        GIVEN,
        771_000,
    );
}

/// `AverageDerivative` on a `TemporalDag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_average_derivative_temporal_dag_graph_posterior() {
    derivative_cell(
        "gcomp_average_derivative_temporal_dag_graph_posterior",
        Derivative::Average,
        Class::Dag,
        POSTERIOR,
        772_000,
    );
}

/// `AverageDerivative` on a `TemporalCpdag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_average_derivative_temporal_cpdag_given() {
    derivative_cell(
        "gcomp_average_derivative_temporal_cpdag_given",
        Derivative::Average,
        Class::Cpdag,
        GIVEN,
        773_000,
    );
}

/// `AverageDerivative` on a `TemporalCpdag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_average_derivative_temporal_cpdag_graph_posterior() {
    derivative_cell(
        "gcomp_average_derivative_temporal_cpdag_graph_posterior",
        Derivative::Average,
        Class::Cpdag,
        POSTERIOR,
        774_000,
    );
}

/// `AverageDerivative` on a `TemporalPag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_average_derivative_temporal_pag_given() {
    derivative_cell(
        "gcomp_average_derivative_temporal_pag_given",
        Derivative::Average,
        Class::Pag,
        GIVEN,
        775_000,
    );
}

/// `AverageDerivative` on a `TemporalPag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_average_derivative_temporal_pag_graph_posterior() {
    derivative_cell(
        "gcomp_average_derivative_temporal_pag_graph_posterior",
        Derivative::Average,
        Class::Pag,
        POSTERIOR,
        776_000,
    );
}

/// `AverageDerivative` on a `TemporalAdmg` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_average_derivative_temporal_admg_given() {
    admg_derivative_cell(
        "gcomp_average_derivative_temporal_admg_given",
        Derivative::Average,
        GIVEN,
        777_000,
    );
}

/// `PointDerivative` on a `TemporalDag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_point_derivative_temporal_dag_given() {
    derivative_cell(
        "gcomp_point_derivative_temporal_dag_given",
        Derivative::Point,
        Class::Dag,
        GIVEN,
        778_000,
    );
}

/// `PointDerivative` on a `TemporalDag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_point_derivative_temporal_dag_graph_posterior() {
    derivative_cell(
        "gcomp_point_derivative_temporal_dag_graph_posterior",
        Derivative::Point,
        Class::Dag,
        POSTERIOR,
        779_000,
    );
}

/// `PointDerivative` on a `TemporalCpdag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_point_derivative_temporal_cpdag_given() {
    derivative_cell(
        "gcomp_point_derivative_temporal_cpdag_given",
        Derivative::Point,
        Class::Cpdag,
        GIVEN,
        780_000,
    );
}

/// `PointDerivative` on a `TemporalCpdag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_point_derivative_temporal_cpdag_graph_posterior() {
    derivative_cell(
        "gcomp_point_derivative_temporal_cpdag_graph_posterior",
        Derivative::Point,
        Class::Cpdag,
        POSTERIOR,
        781_000,
    );
}

/// `PointDerivative` on a `TemporalPag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_point_derivative_temporal_pag_given() {
    derivative_cell(
        "gcomp_point_derivative_temporal_pag_given",
        Derivative::Point,
        Class::Pag,
        GIVEN,
        782_000,
    );
}

/// `PointDerivative` on a `TemporalPag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_point_derivative_temporal_pag_graph_posterior() {
    derivative_cell(
        "gcomp_point_derivative_temporal_pag_graph_posterior",
        Derivative::Point,
        Class::Pag,
        POSTERIOR,
        783_000,
    );
}

/// `PointDerivative` on a `TemporalAdmg` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_point_derivative_temporal_admg_given() {
    admg_derivative_cell(
        "gcomp_point_derivative_temporal_admg_given",
        Derivative::Point,
        GIVEN,
        784_000,
    );
}

/// `Elasticity` on a `TemporalDag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_elasticity_temporal_dag_given() {
    derivative_cell(
        "gcomp_elasticity_temporal_dag_given",
        Derivative::Elasticity,
        Class::Dag,
        GIVEN,
        785_000,
    );
}

/// `Elasticity` on a `TemporalDag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_elasticity_temporal_dag_graph_posterior() {
    derivative_cell(
        "gcomp_elasticity_temporal_dag_graph_posterior",
        Derivative::Elasticity,
        Class::Dag,
        POSTERIOR,
        786_000,
    );
}

/// `Elasticity` on a `TemporalCpdag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_elasticity_temporal_cpdag_given() {
    derivative_cell(
        "gcomp_elasticity_temporal_cpdag_given",
        Derivative::Elasticity,
        Class::Cpdag,
        GIVEN,
        787_000,
    );
}

/// `Elasticity` on a `TemporalCpdag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_elasticity_temporal_cpdag_graph_posterior() {
    derivative_cell(
        "gcomp_elasticity_temporal_cpdag_graph_posterior",
        Derivative::Elasticity,
        Class::Cpdag,
        POSTERIOR,
        788_000,
    );
}

/// `Elasticity` on a `TemporalPag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_elasticity_temporal_pag_given() {
    derivative_cell(
        "gcomp_elasticity_temporal_pag_given",
        Derivative::Elasticity,
        Class::Pag,
        GIVEN,
        789_000,
    );
}

/// `Elasticity` on a `TemporalPag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_elasticity_temporal_pag_graph_posterior() {
    derivative_cell(
        "gcomp_elasticity_temporal_pag_graph_posterior",
        Derivative::Elasticity,
        Class::Pag,
        POSTERIOR,
        790_000,
    );
}

/// `Elasticity` on a `TemporalAdmg` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_elasticity_temporal_admg_given() {
    admg_derivative_cell(
        "gcomp_elasticity_temporal_admg_given",
        Derivative::Elasticity,
        GIVEN,
        791_000,
    );
}

/// `SemiElasticity` on a `TemporalDag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_semi_elasticity_temporal_dag_given() {
    derivative_cell(
        "gcomp_semi_elasticity_temporal_dag_given",
        Derivative::SemiElasticity,
        Class::Dag,
        GIVEN,
        792_000,
    );
}

/// `SemiElasticity` on a `TemporalDag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_semi_elasticity_temporal_dag_graph_posterior() {
    derivative_cell(
        "gcomp_semi_elasticity_temporal_dag_graph_posterior",
        Derivative::SemiElasticity,
        Class::Dag,
        POSTERIOR,
        793_000,
    );
}

/// `SemiElasticity` on a `TemporalCpdag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_semi_elasticity_temporal_cpdag_given() {
    derivative_cell(
        "gcomp_semi_elasticity_temporal_cpdag_given",
        Derivative::SemiElasticity,
        Class::Cpdag,
        GIVEN,
        794_000,
    );
}

/// `SemiElasticity` on a `TemporalCpdag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_semi_elasticity_temporal_cpdag_graph_posterior() {
    derivative_cell(
        "gcomp_semi_elasticity_temporal_cpdag_graph_posterior",
        Derivative::SemiElasticity,
        Class::Cpdag,
        POSTERIOR,
        795_000,
    );
}

/// `SemiElasticity` on a `TemporalPag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_semi_elasticity_temporal_pag_given() {
    derivative_cell(
        "gcomp_semi_elasticity_temporal_pag_given",
        Derivative::SemiElasticity,
        Class::Pag,
        GIVEN,
        796_000,
    );
}

/// `SemiElasticity` on a `TemporalPag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_semi_elasticity_temporal_pag_graph_posterior() {
    derivative_cell(
        "gcomp_semi_elasticity_temporal_pag_graph_posterior",
        Derivative::SemiElasticity,
        Class::Pag,
        POSTERIOR,
        797_000,
    );
}

/// `SemiElasticity` on a `TemporalAdmg` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_semi_elasticity_temporal_admg_given() {
    admg_derivative_cell(
        "gcomp_semi_elasticity_temporal_admg_given",
        Derivative::SemiElasticity,
        GIVEN,
        798_000,
    );
}

/// `ResponseJacobian` on a `TemporalDag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_response_jacobian_temporal_dag_given() {
    derivative_cell(
        "gcomp_response_jacobian_temporal_dag_given",
        Derivative::Jacobian,
        Class::Dag,
        GIVEN,
        799_000,
    );
}

/// `ResponseJacobian` on a `TemporalDag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_response_jacobian_temporal_dag_graph_posterior() {
    derivative_cell(
        "gcomp_response_jacobian_temporal_dag_graph_posterior",
        Derivative::Jacobian,
        Class::Dag,
        POSTERIOR,
        800_000,
    );
}

/// `ResponseJacobian` on a `TemporalCpdag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_response_jacobian_temporal_cpdag_given() {
    derivative_cell(
        "gcomp_response_jacobian_temporal_cpdag_given",
        Derivative::Jacobian,
        Class::Cpdag,
        GIVEN,
        801_000,
    );
}

/// `ResponseJacobian` on a `TemporalCpdag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_response_jacobian_temporal_cpdag_graph_posterior() {
    derivative_cell(
        "gcomp_response_jacobian_temporal_cpdag_graph_posterior",
        Derivative::Jacobian,
        Class::Cpdag,
        POSTERIOR,
        802_000,
    );
}

/// `ResponseJacobian` on a `TemporalPag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_response_jacobian_temporal_pag_given() {
    derivative_cell(
        "gcomp_response_jacobian_temporal_pag_given",
        Derivative::Jacobian,
        Class::Pag,
        GIVEN,
        803_000,
    );
}

/// `ResponseJacobian` on a `TemporalPag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_response_jacobian_temporal_pag_graph_posterior() {
    derivative_cell(
        "gcomp_response_jacobian_temporal_pag_graph_posterior",
        Derivative::Jacobian,
        Class::Pag,
        POSTERIOR,
        804_000,
    );
}

/// `ResponseJacobian` on a `TemporalAdmg` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_response_jacobian_temporal_admg_given() {
    admg_derivative_cell(
        "gcomp_response_jacobian_temporal_admg_given",
        Derivative::Jacobian,
        GIVEN,
        805_000,
    );
}

/// `DirectionalDerivative` on a `TemporalDag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_directional_derivative_temporal_dag_given() {
    derivative_cell(
        "gcomp_directional_derivative_temporal_dag_given",
        Derivative::Directional,
        Class::Dag,
        GIVEN,
        806_000,
    );
}

/// `DirectionalDerivative` on a `TemporalDag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_directional_derivative_temporal_dag_graph_posterior() {
    derivative_cell(
        "gcomp_directional_derivative_temporal_dag_graph_posterior",
        Derivative::Directional,
        Class::Dag,
        POSTERIOR,
        807_000,
    );
}

/// `DirectionalDerivative` on a `TemporalCpdag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_directional_derivative_temporal_cpdag_given() {
    derivative_cell(
        "gcomp_directional_derivative_temporal_cpdag_given",
        Derivative::Directional,
        Class::Cpdag,
        GIVEN,
        808_000,
    );
}

/// `DirectionalDerivative` on a `TemporalCpdag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_directional_derivative_temporal_cpdag_graph_posterior() {
    derivative_cell(
        "gcomp_directional_derivative_temporal_cpdag_graph_posterior",
        Derivative::Directional,
        Class::Cpdag,
        POSTERIOR,
        809_000,
    );
}

/// `DirectionalDerivative` on a `TemporalPag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_directional_derivative_temporal_pag_given() {
    derivative_cell(
        "gcomp_directional_derivative_temporal_pag_given",
        Derivative::Directional,
        Class::Pag,
        GIVEN,
        810_000,
    );
}

/// `DirectionalDerivative` on a `TemporalPag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_directional_derivative_temporal_pag_graph_posterior() {
    derivative_cell(
        "gcomp_directional_derivative_temporal_pag_graph_posterior",
        Derivative::Directional,
        Class::Pag,
        POSTERIOR,
        811_000,
    );
}

/// `DirectionalDerivative` on a `TemporalAdmg` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_directional_derivative_temporal_admg_given() {
    admg_derivative_cell(
        "gcomp_directional_derivative_temporal_admg_given",
        Derivative::Directional,
        GIVEN,
        812_000,
    );
}

/// `ResponseCurve` on a `TemporalDag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_response_curve_temporal_dag_given() {
    surface_cell(
        "gcomp_response_curve_temporal_dag_given",
        Surface::Curve,
        Class::Dag,
        GIVEN_WITHOUT_EXPLICIT_UNVALIDATED,
        813_000,
    );
}

/// `ResponseCurve` on a `TemporalDag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_response_curve_temporal_dag_graph_posterior() {
    surface_cell(
        "gcomp_response_curve_temporal_dag_graph_posterior",
        Surface::Curve,
        Class::Dag,
        POSTERIOR_VALIDATED,
        814_000,
    );
}

/// `ResponseCurve` on a `TemporalCpdag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_response_curve_temporal_cpdag_given() {
    surface_cell(
        "gcomp_response_curve_temporal_cpdag_given",
        Surface::Curve,
        Class::Cpdag,
        GIVEN_VALIDATED,
        815_000,
    );
}

/// `ResponseCurve` on a `TemporalCpdag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_response_curve_temporal_cpdag_graph_posterior() {
    surface_cell(
        "gcomp_response_curve_temporal_cpdag_graph_posterior",
        Surface::Curve,
        Class::Cpdag,
        POSTERIOR_VALIDATED,
        816_000,
    );
}

/// `ResponseCurve` on a `TemporalPag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_response_curve_temporal_pag_given() {
    surface_cell(
        "gcomp_response_curve_temporal_pag_given",
        Surface::Curve,
        Class::Pag,
        GIVEN_VALIDATED,
        817_000,
    );
}

/// `ResponseCurve` on a `TemporalPag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_response_curve_temporal_pag_graph_posterior() {
    surface_cell(
        "gcomp_response_curve_temporal_pag_graph_posterior",
        Surface::Curve,
        Class::Pag,
        POSTERIOR_VALIDATED,
        818_000,
    );
}

/// `InterventionResponse` on a `TemporalDag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_intervention_response_temporal_dag_given() {
    surface_cell(
        "gcomp_intervention_response_temporal_dag_given",
        Surface::Level,
        Class::Dag,
        GIVEN_WITHOUT_EXPLICIT_UNVALIDATED,
        819_000,
    );
}

/// `InterventionResponse` on a `TemporalDag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_intervention_response_temporal_dag_graph_posterior() {
    surface_cell(
        "gcomp_intervention_response_temporal_dag_graph_posterior",
        Surface::Level,
        Class::Dag,
        POSTERIOR_VALIDATED,
        820_000,
    );
}

/// `InterventionResponse` on a `TemporalCpdag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_intervention_response_temporal_cpdag_given() {
    surface_cell(
        "gcomp_intervention_response_temporal_cpdag_given",
        Surface::Level,
        Class::Cpdag,
        GIVEN_VALIDATED,
        821_000,
    );
}

/// `InterventionResponse` on a `TemporalCpdag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_intervention_response_temporal_cpdag_graph_posterior() {
    surface_cell(
        "gcomp_intervention_response_temporal_cpdag_graph_posterior",
        Surface::Level,
        Class::Cpdag,
        POSTERIOR_VALIDATED,
        822_000,
    );
}

/// `InterventionResponse` on a `TemporalPag` (explicit and accepted).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_intervention_response_temporal_pag_given() {
    surface_cell(
        "gcomp_intervention_response_temporal_pag_given",
        Surface::Level,
        Class::Pag,
        GIVEN_VALIDATED,
        823_000,
    );
}

/// `InterventionResponse` on a `TemporalPag` (one-atom graph posterior).
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn gcomp_intervention_response_temporal_pag_graph_posterior() {
    surface_cell(
        "gcomp_intervention_response_temporal_pag_graph_posterior",
        Surface::Level,
        Class::Pag,
        POSTERIOR_VALIDATED,
        824_000,
    );
}
