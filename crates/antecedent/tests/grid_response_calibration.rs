//! Repeated-sampling coverage of the cross-fitted response estimators on every graph
//! class, structure route and validation mode they are offered under.
//!
//! One test measures one coordinate: a query (point derivative, elasticity,
//! semi-elasticity, average derivative, Jacobian, directional derivative, or the mean
//! curve), a graph class, an inference mode and a structure family (a given graph, or a
//! graph posterior). The validation modes and, for a given graph, its accepted twin are
//! sibling executions of that coordinate: the driver measures the first listed execution
//! on every replicate and requires every sibling to report its interval bit for bit.
//!
//! Every law has a closed-form population truth:
//!
//! * [`point_data`]: `X ~ N(0,1)`, `A = 0.5X + N(0,1)`, `Y = 5 + 2 sin A + X + N(0,1)`, so
//!   `m(a) = 5 + 2 sin a`, `m'(a) = 2 cos a` and `E[m'(A)] = 2 exp(-Var(A)/2)`.
//! * [`gam_data`]: two treatments confounded by `X`, `Y1` quadratic in `A1`, `Y2`
//!   linear; the Jacobian at `(0.5, 0)` is `[[2, -0.5], [0.25, 1.5]]`.
//! * [`class_point_data`] and [`class_gam_data`]: the same response laws on the six
//!   columns `a, z, y, r, v, w` of the class graphs, where `r` is an exogenous cause of
//!   `a`, `z` confounds `a` and the outcomes, `w` causes `z` and a latent cause moves
//!   both `w` and `y`. `a` is independent of that latent cause given `z`, and the
//!   outcome is additive in `a`, so every valid adjustment set returns the same
//!   structural derivative.
//! * The mean-curve laws are linear in the dose (`m(a) = 1 + 2a`, or `m(a) = a` on
//!   [`cpdag_data`]).
//!
//! A graph posterior holds one atom: one identified atom publishes its own sampling
//! interval, whereas a mixture over atoms is a distribution over graphs.
//!
//! The driver scores the reported 0.95 interval and the 0.90 interval of the same
//! execution. A Bayesian interval is re-summarized from the draws the estimator
//! retained and a symmetric Frequentist scalar from its standard error. An interval
//! that publishes neither (a Frequentist pointwise band, the Fieller elasticity
//! interval) is refitted on the same data and seed at the 0.90 level, which changes
//! only the interval quantile.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::too_many_lines,
    reason = "test scaffolding compares exact constants and keeps one design table in one body"
)]

mod common;
#[path = "common/grid_cell.rs"]
mod grid_cell;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use antecedent::{
    AcceptedGraph, BayesianConfig, InferenceMode, RefuteSuite, Study, StudyBuilder, StudyResult,
};
use antecedent_core::{
    CausalQuery, ContinuousDomain, DerivativeScale, DerivativeWeighting, ExecutionContext,
    GridSpec, Lag, ResponseFunctional as F, ResponseIdentification, ResponseQuery,
    ResponseUncertainty, ResponseValue, VariableId,
};
use antecedent_data::TabularData;
use antecedent_discovery::adjacency_mask::{
    AdjacencyMask, adjacency_mask_from_admg, adjacency_mask_from_cpdag, adjacency_masks_from_pag,
    set_edge,
};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
use antecedent_estimate::ContinuousResponseOptions;
use antecedent_graph::{
    Admg, Cpdag, Dag, DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag, TemporalAdmg,
};
use antecedent_prob::InferenceDiagnostics;
use common::calibration::{Z95, gaussian, grid_n, stream_seed};
use common::reported::{
    GATE_LEVEL, REPORTED_LEVEL, response_normal_pair, response_posterior_pairs,
};
use common::static_dgp::response_data;
use grid_cell::{Cell, IntervalPair, Target, measure};

/// Base sample size of the derivative designs.
const N_DERIVATIVE: usize = 1000;
/// Base sample size of the mean-curve designs.
const N_CURVE: usize = 500;
/// Fixed bandwidth of the local-polynomial point derivative.
const BANDWIDTH: f64 = 0.35;
/// Evaluation dose of the point derivative, the Jacobian's first treatment and the class
/// Jacobian's only treatment.
const AT: f64 = 0.5;
/// Posterior draws of a Bayesian execution.
const DRAWS: usize = 200;
/// Resamples of the whole-procedure stability check a full validation runs. It is read
/// by validation only, so every sibling carries the same count.
const STABILITY_REPLICATES: u32 = 12;
/// Doses of the mean curve.
const CURVE_GRID: [f64; 3] = [-1.0, 0.0, 1.0];

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn ids(xs: &[u32]) -> Arc<[VariableId]> {
    xs.iter().map(|&x| v(x)).collect()
}

fn table(columns: &[(&str, &[f64])]) -> TabularData {
    TabularData::from_f64_columns(columns.iter().map(|(name, col)| (*name, *col))).unwrap()
}

// ------------------------------------------------------------------ data laws

/// Structural mean `m(a) = E[Y | do(A = a)] = 5 + 2 sin a` of [`point_data`] and
/// [`class_point_data`] (every other cause of the outcome has mean zero).
fn mu(a: f64) -> f64 {
    5.0 + 2.0 * a.sin()
}

fn mu_prime(a: f64) -> f64 {
    2.0 * a.cos()
}

/// `E[m'(A)] = 2 E[cos A] = 2 exp(-variance / 2)` for `A ~ N(0, variance)`.
fn average_slope(variance: f64) -> f64 {
    2.0 * (-0.5 * variance).exp()
}

/// Columns `a`(0), `x`(1), `y`(2): `X ~ N(0,1)`, `A = 0.5X + N(0,1)`,
/// `Y = 5 + 2 sin A + X + N(0,1)`; `Var(A) = 1.25`.
fn point_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut a, mut x, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        x[i] = g();
        a[i] = 0.5 * x[i] + g();
        y[i] = mu(a[i]) + x[i] + g();
    }
    table(&[("a", &a), ("x", &x), ("y", &y)])
}

const POINT_EDGES: [(usize, usize); 3] = [(1, 0), (1, 2), (0, 2)];
const POINT_TREATMENT_VARIANCE: f64 = 1.25;

/// Columns `a1`(0), `a2`(1), `x`(2), `y1`(3), `y2`(4); `X` confounds both treatments:
/// `A1 = 0.5X + e`, `A2 = 0.3X + e`, `Y1 = 1 + 1.5 A1 + 0.5 A1² − 0.5 A2 + X + e`,
/// `Y2 = 0.25 A1 + 1.5 A2 + 0.5 X + e`.
fn gam_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut cols: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::with_capacity(n));
    for _ in 0..n {
        let x = g();
        let a1 = 0.5 * x + g();
        let a2 = 0.3 * x + g();
        cols[0].push(a1);
        cols[1].push(a2);
        cols[2].push(x);
        cols[3].push(1.0 + 1.5 * a1 + 0.5 * a1 * a1 - 0.5 * a2 + x + g());
        cols[4].push(0.25 * a1 + 1.5 * a2 + 0.5 * x + g());
    }
    table(&[
        ("a1", &cols[0]),
        ("a2", &cols[1]),
        ("x", &cols[2]),
        ("y1", &cols[3]),
        ("y2", &cols[4]),
    ])
}

const GAM_EDGES: [(usize, usize); 8] =
    [(2, 0), (2, 1), (2, 3), (2, 4), (0, 3), (0, 4), (1, 3), (1, 4)];
/// The second treatment's evaluation dose.
const GAM_AT_SECOND: f64 = 0.0;
/// Row-major (outcomes × treatments) Jacobian of [`gam_data`] at `(AT, GAM_AT_SECOND)`:
/// `dY1/dA1 = 1.5 + A1 = 2`.
const GAM_JACOBIAN: [f64; 4] = [2.0, -0.5, 0.25, 1.5];
const GAM_DIRECTION: [f64; 2] = [1.0, 2.0];
/// `GAM_JACOBIAN · GAM_DIRECTION`.
const GAM_DIRECTIONAL: [f64; 2] = [1.0, 3.25];

/// The exogenous and confounding part of the class laws: returns `(z, r, w, latent)` with
/// `latent ~ N(0,1)`, `w = latent + e`, `r ~ N(0,1)`, `z = 0.5 w + e`.
fn class_causes(g: &mut impl FnMut() -> f64) -> (f64, f64, f64, f64) {
    let latent = g();
    let w = latent + g();
    let r = g();
    let z = 0.5 * w + g();
    (z, r, w, latent)
}

/// `Var(A)` of the class laws, `A = 0.5 z + 0.5 r + e` with `Var(z) = 0.25·2 + 1`.
const CLASS_TREATMENT_VARIANCE: f64 = 0.25 * 1.5 + 0.25 + 1.0;

fn class_table(cols: &[Vec<f64>; 6]) -> TabularData {
    table(&[
        ("a", &cols[0]),
        ("z", &cols[1]),
        ("y", &cols[2]),
        ("r", &cols[3]),
        ("v", &cols[4]),
        ("w", &cols[5]),
    ])
}

/// Columns `a, z, y, r, v, w` (indices 0..=5): [`class_causes`], `a = 0.5 z + 0.5 r + e`,
/// `y = 5 + 2 sin a + z + 0.5 latent + e`, `v = 1 + 0.25 a − 0.5 z + e`.
///
/// `a` depends on the latent cause only through `z`, so adjusting for `z` (with or
/// without `r` or `w`) leaves `m(a) = 5 + 2 sin a`.
fn class_point_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut cols: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::with_capacity(n));
    for _ in 0..n {
        let (z, r, w, latent) = class_causes(&mut g);
        let a = 0.5 * z + 0.5 * r + g();
        cols[0].push(a);
        cols[1].push(z);
        cols[2].push(mu(a) + z + 0.5 * latent + g());
        cols[3].push(r);
        cols[4].push(1.0 + 0.25 * a - 0.5 * z + g());
        cols[5].push(w);
    }
    class_table(&cols)
}

/// [`class_point_data`] with a quadratic outcome: `y = 1 + 1.5 a + 0.5 a² + z +
/// 0.5 latent + e`, `v = 0.25 a − 0.5 z + e`; `dy/da = 1.5 + a`, `dv/da = 0.25`.
fn class_gam_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut cols: [Vec<f64>; 6] = std::array::from_fn(|_| Vec::with_capacity(n));
    for _ in 0..n {
        let (z, r, w, latent) = class_causes(&mut g);
        let a = 0.5 * z + 0.5 * r + g();
        cols[0].push(a);
        cols[1].push(z);
        cols[2].push(1.0 + 1.5 * a + 0.5 * a * a + z + 0.5 * latent + g());
        cols[3].push(r);
        cols[4].push(0.25 * a - 0.5 * z + g());
        cols[5].push(w);
    }
    class_table(&cols)
}

/// Jacobian of [`class_gam_data`]'s outcomes `(y, v)` in `a` at [`AT`].
const CLASS_JACOBIAN: [f64; 2] = [1.5 + AT, 0.25];
const CLASS_DIRECTION: f64 = 2.0;

/// `r -> a <- z`, `a -> y`, `z -> y`, `a -> v`, `z -> v` over the class columns: a
/// v-structure at `a`, so every edge is compelled and the class has one completion.
const CLASS_EDGES: [(usize, usize); 6] = [(3, 0), (1, 0), (0, 2), (1, 2), (0, 4), (1, 4)];

fn class_cpdag() -> Cpdag {
    let mut g = Cpdag::with_variables(6);
    for (from, to) in CLASS_EDGES {
        g.insert_directed(d(from as u32), d(to as u32)).unwrap();
    }
    g
}

/// The class edges as a PAG with no circles: `a -> y` is visible because `r` points into
/// `a` and is not adjacent to `y`.
fn class_pag() -> Pag {
    let mut g = Pag::with_variables(6);
    for (from, to) in CLASS_EDGES {
        g.insert_directed(d(from as u32), d(to as u32)).unwrap();
    }
    g
}

/// The class edges plus `w -> z` and the latent common cause of `w` and `y` (`w <-> y`).
fn class_admg() -> Admg {
    let mut g = Admg::with_variables(6);
    for (from, to) in CLASS_EDGES {
        g.insert_directed(d(from as u32), d(to as u32)).unwrap();
    }
    g.insert_directed(d(5), d(1)).unwrap();
    g.insert_bidirected(d(5), d(2)).unwrap();
    g
}

/// [`class_admg`] as a `TemporalAdmg` over one contemporaneous node per class column
/// (variable order). Each variable appears once, so the graph projects to
/// [`class_admg`] and a tabular execution runs the static response estimators on it.
fn class_temporal_admg() -> TemporalAdmg {
    let mut g = TemporalAdmg::empty();
    let nodes: Vec<DenseNodeId> =
        (0..6).map(|i| g.add_lagged(v(i), Lag::CONTEMPORANEOUS).unwrap()).collect();
    for (from, to) in CLASS_EDGES {
        g.insert_directed(nodes[from], nodes[to]).unwrap();
    }
    g.insert_directed(nodes[5], nodes[1]).unwrap();
    g.insert_bidirected(nodes[5], nodes[2]).unwrap();
    g
}

/// Columns `t, y, z`: `t` and `z` independent standard normals, `y = 1 + 2t + 0.8 z + e`,
/// matching the one-atom posterior `t -> y`; `m(a) = 1 + 2a`.
fn independent_response_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        t[i] = g();
        z[i] = g();
        y[i] = 1.0 + 2.0 * t[i] + 0.8 * z[i] + g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z)])
}

/// `z -> t`, `z -> y`, `t -> y` over [`response_data`]'s columns `t, y, z`.
const RESPONSE_EDGES: [(usize, usize); 3] = [(2, 0), (2, 1), (0, 1)];
/// `t -> y` over [`independent_response_data`]'s columns.
const INDEPENDENT_EDGES: [(usize, usize); 1] = [(0, 1)];

/// Columns `t, y, z, w`: `z, w ~ N(0,1)`, `t = 0.8 z + e`, `y = t + z + e`. Both
/// orientations of `w — z` in [`agreeing_cpdag`] adjust `{z}`, so `m(a) = a`.
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

/// `z -> t`, `z -> y`, `t -> y`, `w — z` over [`cpdag_data`]'s columns.
fn agreeing_cpdag() -> Cpdag {
    let mut g = Cpdag::with_variables(4);
    g.insert_directed(d(2), d(0)).unwrap();
    g.insert_directed(d(2), d(1)).unwrap();
    g.insert_directed(d(0), d(1)).unwrap();
    g.insert_undirected(d(3), d(2)).unwrap();
    g
}

/// Columns `t, y, z, r`: `z, r ~ N(0,1)`, `t = 0.5 z + 0.4 r + e`,
/// `y = 1 + 2t + 0.8 z + e`; every completion of [`agreeing_pag`] adjusts `{z}`, so
/// `m(a) = 1 + 2a`.
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

/// `r o-> t`, `z o-> t`, `z -> y`, `t -> y` over [`pag_data`]'s columns: `t -> y` is
/// visible in every MAG completion.
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

// ---------------------------------------------------------------- structures

/// The graph class of a coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    Dag,
    Cpdag,
    Pag,
    Admg,
    TemporalAdmg,
}

/// How an execution is given its structure.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Via {
    Explicit,
    Accepted,
    Posterior,
}

use Via::{Accepted, Explicit, Posterior};

/// One execution of a coordinate: its structure route and validation mode.
type Variant = (Via, RefuteSuite);

/// The structure of a design, as a given graph and as a one-atom graph posterior.
enum Structure {
    Dag {
        nodes: usize,
        edges: &'static [(usize, usize)],
    },
    Cpdag(Cpdag),
    /// A PAG whose marks are all tails and arrowheads: its atom is the directed edges.
    DirectedPag(Pag, &'static [(usize, usize)]),
    /// A PAG with circle marks: its atom carries the mark mask.
    MarkedPag(Pag),
    Admg(Admg),
    /// The latent-confounded class graph as a temporal-class atom (a graph posterior) or,
    /// given, as [`class_temporal_admg`].
    TemporalAdmg(Admg),
}

fn directed_mask(nodes: usize, edges: &[(usize, usize)]) -> AdjacencyMask {
    edges
        .iter()
        .fold(AdjacencyMask::EMPTY, |mask, &(from, to)| set_edge(mask, nodes, from, to, true))
}

impl Structure {
    fn dag(nodes: usize, edges: &[(usize, usize)]) -> Dag {
        let mut graph = Dag::with_variables(nodes as u32);
        for &(from, to) in edges {
            graph.insert_directed(d(from as u32), d(to as u32)).unwrap();
        }
        graph
    }

    fn posterior(&self) -> GraphPosterior {
        let (nodes, atom, kind, marks) = match self {
            Self::Dag { nodes, edges } => (*nodes, directed_mask(*nodes, edges), None, None),
            Self::Cpdag(graph) => (
                graph.node_count(),
                adjacency_mask_from_cpdag(graph).unwrap(),
                Some(GraphPosteriorAtomKind::Cpdag),
                None,
            ),
            Self::DirectedPag(graph, edges) => (
                graph.node_count(),
                directed_mask(graph.node_count(), edges),
                Some(GraphPosteriorAtomKind::Pag),
                None,
            ),
            Self::MarkedPag(graph) => {
                let (adjacency, marks) = adjacency_masks_from_pag(graph).unwrap();
                (graph.node_count(), adjacency, Some(GraphPosteriorAtomKind::Pag), Some(marks))
            }
            Self::Admg(graph) => (
                graph.node_count(),
                adjacency_mask_from_admg(graph).unwrap(),
                Some(GraphPosteriorAtomKind::Admg),
                None,
            ),
            Self::TemporalAdmg(graph) => (
                graph.node_count(),
                adjacency_mask_from_admg(graph).unwrap(),
                Some(GraphPosteriorAtomKind::TemporalAdmg),
                None,
            ),
        };
        let posterior = GraphPosterior::from_atoms(
            nodes,
            vec![1.0],
            vec![atom],
            vec![0.0; nodes * nodes],
            vec![0.0; nodes * nodes],
            SamplingSummary {
                ess: 1.0,
                diagnostics: InferenceDiagnostics::analytic("single_atom_response"),
                rejected_invalid: 0,
            },
        )
        .unwrap();
        let posterior = match kind {
            Some(kind) => posterior.with_atom_kind(kind),
            None => posterior,
        };
        match marks {
            Some(marks) => posterior.with_mark_atoms(vec![marks]).unwrap(),
            None => posterior,
        }
    }

    fn attach(&self, builder: StudyBuilder, via: Via) -> StudyBuilder {
        match (self, via) {
            (_, Posterior) => builder.graph_posterior(self.posterior()),
            (Self::Dag { nodes, edges }, Explicit) => builder.graph(Self::dag(*nodes, edges)),
            (Self::Dag { nodes, edges }, Accepted) => {
                builder.graph(AcceptedGraph::from(Self::dag(*nodes, edges)))
            }
            (Self::Cpdag(graph), Explicit) => builder.graph(graph.clone()),
            (Self::Cpdag(graph), Accepted) => builder.graph(AcceptedGraph::from(graph.clone())),
            (Self::DirectedPag(graph, _) | Self::MarkedPag(graph), Explicit) => {
                builder.graph(graph.clone())
            }
            (Self::DirectedPag(graph, _) | Self::MarkedPag(graph), Accepted) => {
                builder.graph(AcceptedGraph::from(graph.clone()))
            }
            (Self::Admg(graph), Explicit) => builder.graph(graph.clone()),
            (Self::Admg(graph), Accepted) => builder.graph(AcceptedGraph::from(graph.clone())),
            (Self::TemporalAdmg(_), Explicit) => builder.graph(class_temporal_admg()),
            (Self::TemporalAdmg(_), Accepted) => {
                builder.graph(AcceptedGraph::from(class_temporal_admg()))
            }
        }
    }
}

// ------------------------------------------------------------------- designs

/// The inference mode of a coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Frequentist,
    Bayesian,
}

impl Mode {
    fn inference(self) -> InferenceMode {
        match self {
            Self::Frequentist => InferenceMode::Frequentist,
            Self::Bayesian => InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(DRAWS)),
        }
    }

    /// `IntervalMethod::as_str()` of the interval the mode reports for these estimators:
    /// the influence-function interval, or the quantiles of the posterior draws.
    fn interval(self) -> &'static str {
        match self {
            Self::Frequentist => "analytic_se",
            Self::Bayesian => "posterior_quantile",
        }
    }
}

/// The query of a coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Query {
    Point,
    Elasticity,
    /// The outcome-log semi-elasticity `m'(a) / m(a)`.
    SemiElasticity,
    Average,
    Jacobian,
    Directional,
    Curve,
}

/// Everything one coordinate executes and scores.
struct Design {
    /// The generator's name.
    dgp: &'static str,
    data: fn(usize, u64) -> TabularData,
    base_n: usize,
    structure: Structure,
    functional: F,
    bandwidth: Option<f64>,
    estimator: &'static str,
    /// Reported coordinates scored, in `targets` order.
    scored: Vec<usize>,
    targets: Vec<Target>,
}

fn point(outcome: u32, scale: DerivativeScale) -> F {
    F::PointDerivative { outcome: v(outcome), treatment: v(0), at: AT, order: 1, scale }
}

/// The mean-curve design of `class` under a given graph or a graph posterior.
fn curve_design(class: Class, posterior: bool) -> Design {
    let (dgp, data, structure, truth): (_, fn(usize, u64) -> TabularData, _, fn(f64) -> f64) =
        match (class, posterior) {
            (Class::Dag, false) => (
                "response_data",
                response_data,
                Structure::Dag { nodes: 3, edges: &RESPONSE_EDGES },
                |a| 1.0 + 2.0 * a,
            ),
            (Class::Dag, true) => (
                "independent_response_data",
                independent_response_data,
                Structure::Dag { nodes: 3, edges: &INDEPENDENT_EDGES },
                |a| 1.0 + 2.0 * a,
            ),
            (Class::Cpdag, _) => {
                ("cpdag_data", cpdag_data, Structure::Cpdag(agreeing_cpdag()), |a| a)
            }
            (Class::Pag, _) => {
                ("pag_data", pag_data, Structure::MarkedPag(agreeing_pag()), |a| 1.0 + 2.0 * a)
            }
            (Class::Admg | Class::TemporalAdmg, _) => {
                panic!("no mean-curve coordinate is measured here for {class:?}")
            }
        };
    Design {
        dgp,
        data,
        base_n: N_CURVE,
        structure,
        functional: F::MeanCurve {
            outcome: v(1),
            treatment: ContinuousDomain::new(v(0), GridSpec::Values(CURVE_GRID.to_vec().into())),
        },
        bandwidth: None,
        estimator: "response.kennedy_dr",
        scored: (0..CURVE_GRID.len()).collect(),
        targets: CURVE_GRID.iter().map(|&a| Target::labelled(format!("a={a}"), truth(a))).collect(),
    }
}

/// The design of `query` on `class`: the plain-DAG laws for a DAG, the six-column class
/// laws otherwise.
fn design(class: Class, query: Query, posterior: bool) -> Design {
    if query == Query::Curve {
        return curve_design(class, posterior);
    }
    let dag = class == Class::Dag;
    let vector = matches!(query, Query::Jacobian | Query::Directional);
    let (dgp, data): (_, fn(usize, u64) -> TabularData) = match (dag, vector) {
        (true, false) => ("point_data", point_data),
        (true, true) => ("gam_data", gam_data),
        (false, false) => ("class_point_data", class_point_data),
        (false, true) => ("class_gam_data", class_gam_data),
    };
    let structure = match class {
        Class::Dag if vector => Structure::Dag { nodes: 5, edges: &GAM_EDGES },
        Class::Dag => Structure::Dag { nodes: 3, edges: &POINT_EDGES },
        Class::Cpdag => Structure::Cpdag(class_cpdag()),
        Class::Pag => Structure::DirectedPag(class_pag(), &CLASS_EDGES),
        Class::Admg => Structure::Admg(class_admg()),
        Class::TemporalAdmg => Structure::TemporalAdmg(class_admg()),
    };
    // The scalar outcome is column `y` (2) in both scalar laws.
    let treatment_variance = if dag { POINT_TREATMENT_VARIANCE } else { CLASS_TREATMENT_VARIANCE };
    let (functional, bandwidth, estimator, scored, targets) = match query {
        Query::Point => (
            point(2, DerivativeScale::Identity),
            Some(BANDWIDTH),
            "response.kennedy_dr",
            vec![0],
            Target::scalar(mu_prime(AT)),
        ),
        Query::Elasticity => (
            point(2, DerivativeScale::LogLog),
            Some(BANDWIDTH),
            "response.kennedy_dr",
            vec![0],
            Target::scalar(AT * mu_prime(AT) / mu(AT)),
        ),
        Query::SemiElasticity => (
            point(2, DerivativeScale::LogOutcome),
            Some(BANDWIDTH),
            "response.kennedy_dr",
            vec![0],
            Target::scalar(mu_prime(AT) / mu(AT)),
        ),
        Query::Average => (
            F::AverageDerivative {
                outcome: v(2),
                treatment: v(0),
                weighting: DerivativeWeighting::Observed,
            },
            None,
            "response.riesz_ade",
            vec![0],
            Target::scalar(average_slope(treatment_variance)),
        ),
        Query::Jacobian if dag => (
            F::Jacobian {
                outcomes: ids(&[3, 4]),
                treatments: ids(&[0, 1]),
                at: Arc::from([AT, GAM_AT_SECOND]),
                scale: DerivativeScale::Identity,
            },
            None,
            "response.gam_derivative",
            // Three of the four row-major coordinates: both partials of `y1` and the
            // own-treatment partial of `y2`.
            vec![0, 1, 3],
            vec![
                Target::labelled("dy1/da1", GAM_JACOBIAN[0]),
                Target::labelled("dy1/da2", GAM_JACOBIAN[1]),
                Target::labelled("dy2/da2", GAM_JACOBIAN[3]),
            ],
        ),
        Query::Jacobian => (
            F::Jacobian {
                outcomes: ids(&[2, 4]),
                treatments: ids(&[0]),
                at: Arc::from([AT]),
                scale: DerivativeScale::Identity,
            },
            None,
            "response.gam_derivative",
            vec![0, 1],
            vec![
                Target::labelled("dy/da", CLASS_JACOBIAN[0]),
                Target::labelled("dv/da", CLASS_JACOBIAN[1]),
            ],
        ),
        Query::Directional if dag => (
            F::DirectionalDerivative {
                outcomes: ids(&[3, 4]),
                treatments: ids(&[0, 1]),
                at: Arc::from([AT, GAM_AT_SECOND]),
                direction: Arc::from(GAM_DIRECTION),
            },
            None,
            "response.gam_derivative",
            vec![0, 1],
            vec![
                Target::labelled("y1", GAM_DIRECTIONAL[0]),
                Target::labelled("y2", GAM_DIRECTIONAL[1]),
            ],
        ),
        Query::Directional => (
            F::DirectionalDerivative {
                outcomes: ids(&[2, 4]),
                treatments: ids(&[0]),
                at: Arc::from([AT]),
                direction: Arc::from([CLASS_DIRECTION]),
            },
            None,
            "response.gam_derivative",
            vec![0, 1],
            vec![
                Target::labelled("y", CLASS_DIRECTION * CLASS_JACOBIAN[0]),
                Target::labelled("v", CLASS_DIRECTION * CLASS_JACOBIAN[1]),
            ],
        ),
        Query::Curve => unreachable!("handled above"),
    };
    Design {
        dgp,
        data,
        base_n: N_DERIVATIVE,
        structure,
        functional,
        bandwidth,
        estimator,
        scored,
        targets,
    }
}

// ----------------------------------------------------------------- execution

/// Build and run one execution of `design` at interval `level`.
fn execute(
    design: &Design,
    mode: Mode,
    (via, suite): Variant,
    data: &TabularData,
    seed: u64,
    level: f64,
) -> (Study, StudyResult) {
    // The stability resamples belong to the derivative refuters; the curve refuters
    // take none.
    let replicates =
        if matches!(design.functional, F::MeanCurve { .. }) { 0 } else { STABILITY_REPLICATES };
    let study = design
        .structure
        .attach(Study::tabular(data.clone()), via)
        .query(CausalQuery::Response(ResponseQuery::new(design.functional.clone())))
        .response_options(ContinuousResponseOptions {
            bandwidth: design.bandwidth,
            confidence_level: level,
            ..ContinuousResponseOptions::default()
        })
        .inference(mode.inference())
        .refute(suite)
        .bootstrap_replicates(replicates)
        .build()
        .unwrap_or_else(|e| panic!("{} {via:?} {suite:?}: {e}", design.dgp));
    let result = study
        .run(&ExecutionContext::for_tests(seed))
        .unwrap_or_else(|e| panic!("{} {via:?} {suite:?}: {e}", design.dgp));
    (study, result)
}

/// Every reported coordinate of a point-identified response.
fn values(result: &StudyResult) -> Vec<f64> {
    let response = result.response().expect("a response result");
    let ResponseIdentification::PointIdentified(value) = &response.estimate else {
        panic!("the response is not point identified: {:?}", response.estimate);
    };
    match value {
        ResponseValue::Scalar(x) => vec![*x],
        ResponseValue::Surface { mean, .. } => mean.to_vec(),
        ResponseValue::Jacobian { values, .. } | ResponseValue::Vector(values) => values.to_vec(),
        other @ ResponseValue::Envelope(_) => panic!("unexpected payload {other:?}"),
    }
}

/// The published interval of every reported coordinate, which must be at `level`;
/// `None` per coordinate when the execution publishes no interval.
fn bounds(result: &StudyResult, level: f64) -> Vec<Option<(f64, f64)>> {
    let coordinates = values(result).len();
    let (published, intervals): (f64, Vec<Option<(f64, f64)>>) =
        match &result.response().expect("a response result").uncertainty {
            ResponseUncertainty::Scalar { level, lower, upper, .. } => {
                (*level, vec![Some((*lower, *upper))])
            }
            ResponseUncertainty::PointwiseBand { level, lower, upper, .. } => {
                (*level, lower.iter().zip(upper.iter()).map(|(&lo, &hi)| Some((lo, hi))).collect())
            }
            _ => return vec![None; coordinates],
        };
    assert!((published - level).abs() < 1e-12, "published level {published}, expected {level}");
    assert_eq!(intervals.len(), coordinates, "one interval per reported coordinate");
    intervals
}

/// `[reported, gate]` intervals when the gate-level interval is the estimator's own
/// construction re-read from one execution: the retained draws of a credible interval,
/// or the standard error of a scalar interval that is `± z·se` around its midpoint.
fn derived(result: &StudyResult) -> Option<Vec<IntervalPair>> {
    let uncertainty = &result.response()?.uncertainty;
    if uncertainty.credible_draws().is_some() {
        return response_posterior_pairs(result);
    }
    if let ResponseUncertainty::Scalar { standard_error, level, lower, upper, .. } = uncertainty
        && (level - REPORTED_LEVEL).abs() < 1e-12
        && ((upper - lower) - 2.0 * Z95 * standard_error).abs() <= 1e-9 * (1.0 + standard_error)
    {
        return Some(vec![response_normal_pair(result)]);
    }
    None
}

/// Identity of an execution's reported estimate and interval: the bits of every value
/// and every bound.
fn fingerprint(result: &StudyResult) -> Vec<u64> {
    let mut bits: Vec<u64> = values(result).iter().map(|x| x.to_bits()).collect();
    for interval in bounds(result, REPORTED_LEVEL) {
        match interval {
            Some((lo, hi)) => bits.extend([1, lo.to_bits(), hi.to_bits()]),
            None => bits.push(0),
        }
    }
    bits
}

type GateIntervals = Mutex<HashMap<Vec<u64>, Vec<Option<(f64, f64)>>>>;

/// Gate-level intervals of the executions that were refitted at the gate level, keyed
/// by the reported execution's [`fingerprint`]. A sibling that reports the base
/// execution's estimate and interval bit for bit reads the base execution's entry; one
/// that does not finds none and fails the driver's sibling comparison.
fn gate_intervals() -> &'static GateIntervals {
    static STORE: OnceLock<GateIntervals> = OnceLock::new();
    STORE.get_or_init(|| Mutex::new(HashMap::new()))
}

/// `[reported, gate]` interval pairs of the scored coordinates.
fn score(result: &StudyResult, scored: &[usize]) -> Vec<IntervalPair> {
    let pairs = derived(result).unwrap_or_else(|| {
        let gate = gate_intervals().lock().unwrap().get(&fingerprint(result)).cloned();
        bounds(result, REPORTED_LEVEL)
            .into_iter()
            .enumerate()
            .map(|(j, reported)| [reported, gate.as_ref().and_then(|gate| gate[j])])
            .collect()
    });
    scored.iter().map(|&j| pairs[j]).collect()
}

/// Measure one coordinate: `variants[0]` on every replicate, the rest as its siblings.
fn measure_coordinate(
    test: &'static str,
    family: u64,
    class: Class,
    query: Query,
    mode: Mode,
    variants: &[Variant],
) {
    let design = design(class, query, variants[0].0 == Posterior);
    let cell = Cell {
        test,
        dgp: design.dgp,
        interval: mode.interval(),
        estimator: design.estimator,
        variants: variants.len(),
    };
    measure(
        cell,
        &design.targets,
        |rep, variant| {
            let seed = stream_seed(family, rep);
            let data = (design.data)(grid_n(design.base_n), seed);
            let (study, result) =
                execute(&design, mode, variants[variant], &data, seed, REPORTED_LEVEL);
            if variant == 0 && derived(&result).is_none() {
                // The level enters only the interval quantile, so the same data and seed
                // at the gate level give the gate-level interval of this execution.
                let (_, at_gate) = execute(&design, mode, variants[0], &data, seed, GATE_LEVEL);
                assert_eq!(
                    values(&at_gate),
                    values(&result),
                    "{test}: the interval level moved the estimate"
                );
                gate_intervals()
                    .lock()
                    .unwrap()
                    .insert(fingerprint(&result), bounds(&at_gate, GATE_LEVEL));
            }
            (study, result)
        },
        |result| score(result, &design.scored),
    );
}

macro_rules! coverage_tests {
    ($(
        $name:ident: $family:literal, $class:ident, $query:ident, $mode:ident,
        [$(($via:ident, $suite:ident)),+ $(,)?];
    )+) => {$(
        #[test]
        #[ignore = "calibration: run via scripts/measure_calibration.sh"]
        fn $name() {
            measure_coordinate(
                stringify!($name),
                $family,
                Class::$class,
                Query::$query,
                Mode::$mode,
                &[$(($via, RefuteSuite::$suite)),+],
            );
        }
    )+};
}

coverage_tests! {
    point_derivative_dag_given_frequentist_coverage: 0x52E5_0000, Dag, Point, Frequentist, [
        (Accepted, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full),
        (Accepted, Full),
    ];
    point_derivative_dag_given_bayesian_coverage: 0x52E5_0001, Dag, Point, Bayesian, [
        (Accepted, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full),
        (Accepted, Full),
    ];
    point_derivative_dag_graph_posterior_frequentist_coverage: 0x52E5_0002, Dag, Point, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    point_derivative_dag_graph_posterior_bayesian_coverage: 0x52E5_0003, Dag, Point, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    point_derivative_cpdag_given_frequentist_coverage: 0x52E5_0004, Cpdag, Point, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    point_derivative_cpdag_given_bayesian_coverage: 0x52E5_0005, Cpdag, Point, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    point_derivative_cpdag_graph_posterior_frequentist_coverage: 0x52E5_0006, Cpdag, Point, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    point_derivative_cpdag_graph_posterior_bayesian_coverage: 0x52E5_0007, Cpdag, Point, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    point_derivative_pag_given_frequentist_coverage: 0x52E5_0008, Pag, Point, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    point_derivative_pag_given_bayesian_coverage: 0x52E5_0009, Pag, Point, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    point_derivative_pag_graph_posterior_frequentist_coverage: 0x52E5_000A, Pag, Point, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    point_derivative_pag_graph_posterior_bayesian_coverage: 0x52E5_000B, Pag, Point, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    point_derivative_admg_given_frequentist_coverage: 0x52E5_000C, Admg, Point, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    point_derivative_admg_given_bayesian_coverage: 0x52E5_000D, Admg, Point, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    point_derivative_admg_graph_posterior_frequentist_coverage: 0x52E5_000E, Admg, Point, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    point_derivative_admg_graph_posterior_bayesian_coverage: 0x52E5_000F, Admg, Point, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    point_derivative_temporal_admg_graph_posterior_frequentist_coverage: 0x52E5_0010, TemporalAdmg, Point, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    point_derivative_temporal_admg_graph_posterior_bayesian_coverage: 0x52E5_0011, TemporalAdmg, Point, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    plain_elasticity_dag_given_frequentist_coverage: 0x52E5_0012, Dag, Elasticity, Frequentist, [
        (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full), (Accepted, Full),
    ];
    plain_elasticity_dag_given_bayesian_coverage: 0x52E5_0013, Dag, Elasticity, Bayesian, [
        (Accepted, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full),
        (Accepted, Full),
    ];
    plain_elasticity_dag_graph_posterior_frequentist_coverage: 0x52E5_0014, Dag, Elasticity, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    plain_elasticity_dag_graph_posterior_bayesian_coverage: 0x52E5_0015, Dag, Elasticity, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    plain_elasticity_cpdag_given_frequentist_coverage: 0x52E5_0016, Cpdag, Elasticity, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    plain_elasticity_cpdag_given_bayesian_coverage: 0x52E5_0017, Cpdag, Elasticity, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    plain_elasticity_cpdag_graph_posterior_frequentist_coverage: 0x52E5_0018, Cpdag, Elasticity, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    plain_elasticity_cpdag_graph_posterior_bayesian_coverage: 0x52E5_0019, Cpdag, Elasticity, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    plain_elasticity_pag_given_frequentist_coverage: 0x52E5_001A, Pag, Elasticity, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    plain_elasticity_pag_given_bayesian_coverage: 0x52E5_001B, Pag, Elasticity, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    plain_elasticity_pag_graph_posterior_frequentist_coverage: 0x52E5_001C, Pag, Elasticity, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    plain_elasticity_pag_graph_posterior_bayesian_coverage: 0x52E5_001D, Pag, Elasticity, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    plain_elasticity_admg_given_frequentist_coverage: 0x52E5_001E, Admg, Elasticity, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    plain_elasticity_admg_given_bayesian_coverage: 0x52E5_001F, Admg, Elasticity, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    plain_elasticity_admg_graph_posterior_frequentist_coverage: 0x52E5_0020, Admg, Elasticity, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    plain_elasticity_admg_graph_posterior_bayesian_coverage: 0x52E5_0021, Admg, Elasticity, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    plain_elasticity_temporal_admg_graph_posterior_frequentist_coverage: 0x52E5_0022, TemporalAdmg, Elasticity, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    plain_elasticity_temporal_admg_graph_posterior_bayesian_coverage: 0x52E5_0023, TemporalAdmg, Elasticity, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    semi_elasticity_dag_given_frequentist_coverage: 0x52E5_0024, Dag, SemiElasticity, Frequentist, [
        (Accepted, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full),
        (Accepted, Full),
    ];
    semi_elasticity_dag_given_bayesian_coverage: 0x52E5_0025, Dag, SemiElasticity, Bayesian, [
        (Accepted, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full),
        (Accepted, Full),
    ];
    semi_elasticity_dag_graph_posterior_frequentist_coverage: 0x52E5_0026, Dag, SemiElasticity, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    semi_elasticity_dag_graph_posterior_bayesian_coverage: 0x52E5_0027, Dag, SemiElasticity, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    semi_elasticity_cpdag_given_frequentist_coverage: 0x52E5_0028, Cpdag, SemiElasticity, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    semi_elasticity_cpdag_given_bayesian_coverage: 0x52E5_0029, Cpdag, SemiElasticity, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    semi_elasticity_cpdag_graph_posterior_frequentist_coverage: 0x52E5_002A, Cpdag, SemiElasticity, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    semi_elasticity_cpdag_graph_posterior_bayesian_coverage: 0x52E5_002B, Cpdag, SemiElasticity, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    semi_elasticity_pag_given_frequentist_coverage: 0x52E5_002C, Pag, SemiElasticity, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    semi_elasticity_pag_given_bayesian_coverage: 0x52E5_002D, Pag, SemiElasticity, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    semi_elasticity_pag_graph_posterior_frequentist_coverage: 0x52E5_002E, Pag, SemiElasticity, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    semi_elasticity_pag_graph_posterior_bayesian_coverage: 0x52E5_002F, Pag, SemiElasticity, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    semi_elasticity_admg_given_frequentist_coverage: 0x52E5_0030, Admg, SemiElasticity, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    semi_elasticity_admg_given_bayesian_coverage: 0x52E5_0031, Admg, SemiElasticity, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    semi_elasticity_admg_graph_posterior_frequentist_coverage: 0x52E5_0032, Admg, SemiElasticity, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    semi_elasticity_admg_graph_posterior_bayesian_coverage: 0x52E5_0033, Admg, SemiElasticity, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    semi_elasticity_temporal_admg_graph_posterior_frequentist_coverage: 0x52E5_0034, TemporalAdmg, SemiElasticity, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    semi_elasticity_temporal_admg_graph_posterior_bayesian_coverage: 0x52E5_0035, TemporalAdmg, SemiElasticity, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    average_derivative_dag_given_frequentist_coverage: 0x52E5_0036, Dag, Average, Frequentist, [
        (Accepted, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full),
        (Accepted, Full),
    ];
    average_derivative_dag_given_bayesian_coverage: 0x52E5_0037, Dag, Average, Bayesian, [
        (Accepted, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full),
        (Accepted, Full),
    ];
    average_derivative_dag_graph_posterior_frequentist_coverage: 0x52E5_0038, Dag, Average, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    average_derivative_dag_graph_posterior_bayesian_coverage: 0x52E5_0039, Dag, Average, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    average_derivative_cpdag_given_frequentist_coverage: 0x52E5_003A, Cpdag, Average, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    average_derivative_cpdag_given_bayesian_coverage: 0x52E5_003B, Cpdag, Average, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    average_derivative_cpdag_graph_posterior_frequentist_coverage: 0x52E5_003C, Cpdag, Average, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    average_derivative_cpdag_graph_posterior_bayesian_coverage: 0x52E5_003D, Cpdag, Average, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    average_derivative_pag_given_frequentist_coverage: 0x52E5_003E, Pag, Average, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    average_derivative_pag_given_bayesian_coverage: 0x52E5_003F, Pag, Average, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    average_derivative_pag_graph_posterior_frequentist_coverage: 0x52E5_0040, Pag, Average, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    average_derivative_pag_graph_posterior_bayesian_coverage: 0x52E5_0041, Pag, Average, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    average_derivative_admg_given_frequentist_coverage: 0x52E5_0042, Admg, Average, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    average_derivative_admg_given_bayesian_coverage: 0x52E5_0043, Admg, Average, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    average_derivative_admg_graph_posterior_frequentist_coverage: 0x52E5_0044, Admg, Average, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    average_derivative_admg_graph_posterior_bayesian_coverage: 0x52E5_0045, Admg, Average, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    average_derivative_temporal_admg_graph_posterior_frequentist_coverage: 0x52E5_0046, TemporalAdmg, Average, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    average_derivative_temporal_admg_graph_posterior_bayesian_coverage: 0x52E5_0047, TemporalAdmg, Average, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    response_jacobian_dag_given_frequentist_coverage: 0x52E5_0048, Dag, Jacobian, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    response_jacobian_dag_given_bayesian_coverage: 0x52E5_0049, Dag, Jacobian, Bayesian, [
        (Accepted, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full),
        (Accepted, Full),
    ];
    response_jacobian_dag_graph_posterior_frequentist_coverage: 0x52E5_004A, Dag, Jacobian, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    response_jacobian_dag_graph_posterior_bayesian_coverage: 0x52E5_004B, Dag, Jacobian, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    response_jacobian_cpdag_given_frequentist_coverage: 0x52E5_004C, Cpdag, Jacobian, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    response_jacobian_cpdag_given_bayesian_coverage: 0x52E5_004D, Cpdag, Jacobian, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    response_jacobian_cpdag_graph_posterior_frequentist_coverage: 0x52E5_004E, Cpdag, Jacobian, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    response_jacobian_cpdag_graph_posterior_bayesian_coverage: 0x52E5_004F, Cpdag, Jacobian, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    response_jacobian_pag_given_frequentist_coverage: 0x52E5_0050, Pag, Jacobian, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    response_jacobian_pag_given_bayesian_coverage: 0x52E5_0051, Pag, Jacobian, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    response_jacobian_pag_graph_posterior_frequentist_coverage: 0x52E5_0052, Pag, Jacobian, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    response_jacobian_pag_graph_posterior_bayesian_coverage: 0x52E5_0053, Pag, Jacobian, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    response_jacobian_admg_given_frequentist_coverage: 0x52E5_0054, Admg, Jacobian, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    response_jacobian_admg_given_bayesian_coverage: 0x52E5_0055, Admg, Jacobian, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    response_jacobian_admg_graph_posterior_frequentist_coverage: 0x52E5_0056, Admg, Jacobian, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    response_jacobian_admg_graph_posterior_bayesian_coverage: 0x52E5_0057, Admg, Jacobian, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    response_jacobian_temporal_admg_graph_posterior_frequentist_coverage: 0x52E5_0058, TemporalAdmg, Jacobian, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    response_jacobian_temporal_admg_graph_posterior_bayesian_coverage: 0x52E5_0059, TemporalAdmg, Jacobian, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    directional_derivative_dag_given_frequentist_coverage: 0x52E5_005A, Dag, Directional, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    directional_derivative_dag_given_bayesian_coverage: 0x52E5_005B, Dag, Directional, Bayesian, [
        (Accepted, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full),
        (Accepted, Full),
    ];
    directional_derivative_dag_graph_posterior_frequentist_coverage: 0x52E5_005C, Dag, Directional, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    directional_derivative_dag_graph_posterior_bayesian_coverage: 0x52E5_005D, Dag, Directional, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    directional_derivative_cpdag_given_frequentist_coverage: 0x52E5_005E, Cpdag, Directional, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    directional_derivative_cpdag_given_bayesian_coverage: 0x52E5_005F, Cpdag, Directional, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    directional_derivative_cpdag_graph_posterior_frequentist_coverage: 0x52E5_0060, Cpdag, Directional, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    directional_derivative_cpdag_graph_posterior_bayesian_coverage: 0x52E5_0061, Cpdag, Directional, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    directional_derivative_pag_given_frequentist_coverage: 0x52E5_0062, Pag, Directional, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    directional_derivative_pag_given_bayesian_coverage: 0x52E5_0063, Pag, Directional, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    directional_derivative_pag_graph_posterior_frequentist_coverage: 0x52E5_0064, Pag, Directional, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    directional_derivative_pag_graph_posterior_bayesian_coverage: 0x52E5_0065, Pag, Directional, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    directional_derivative_admg_given_frequentist_coverage: 0x52E5_0066, Admg, Directional, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    directional_derivative_admg_given_bayesian_coverage: 0x52E5_0067, Admg, Directional, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    directional_derivative_admg_graph_posterior_frequentist_coverage: 0x52E5_0068, Admg, Directional, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    directional_derivative_admg_graph_posterior_bayesian_coverage: 0x52E5_0069, Admg, Directional, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    directional_derivative_temporal_admg_graph_posterior_frequentist_coverage: 0x52E5_006A, TemporalAdmg, Directional, Frequentist, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    directional_derivative_temporal_admg_graph_posterior_bayesian_coverage: 0x52E5_006B, TemporalAdmg, Directional, Bayesian, [
        (Posterior, None), (Posterior, Cheap), (Posterior, Full),
    ];
    response_curve_dag_given_frequentist_coverage: 0x52E5_006C, Dag, Curve, Frequentist, [
        (Accepted, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full),
        (Accepted, Full),
    ];
    response_curve_dag_graph_posterior_frequentist_coverage: 0x52E5_006D, Dag, Curve, Frequentist, [
        (Posterior, Cheap), (Posterior, Full),
    ];
    response_curve_cpdag_given_frequentist_coverage: 0x52E5_006E, Cpdag, Curve, Frequentist, [
        (Accepted, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full),
        (Accepted, Full),
    ];
    response_curve_cpdag_graph_posterior_frequentist_coverage: 0x52E5_006F, Cpdag, Curve, Frequentist, [
        (Posterior, Cheap), (Posterior, Full),
    ];
    response_curve_pag_given_frequentist_coverage: 0x52E5_0070, Pag, Curve, Frequentist, [
        (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full), (Accepted, Full),
    ];
    response_curve_pag_graph_posterior_frequentist_coverage: 0x52E5_0071, Pag, Curve, Frequentist, [
        (Posterior, Cheap), (Posterior, Full),
    ];
    point_derivative_temporal_admg_given_frequentist_coverage: 0x52E5_0072, TemporalAdmg, Point, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    point_derivative_temporal_admg_given_bayesian_coverage: 0x52E5_0073, TemporalAdmg, Point, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    plain_elasticity_temporal_admg_given_frequentist_coverage: 0x52E5_0074, TemporalAdmg, Elasticity, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    plain_elasticity_temporal_admg_given_bayesian_coverage: 0x52E5_0075, TemporalAdmg, Elasticity, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    semi_elasticity_temporal_admg_given_frequentist_coverage: 0x52E5_0076, TemporalAdmg, SemiElasticity, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    semi_elasticity_temporal_admg_given_bayesian_coverage: 0x52E5_0077, TemporalAdmg, SemiElasticity, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    average_derivative_temporal_admg_given_frequentist_coverage: 0x52E5_0078, TemporalAdmg, Average, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    average_derivative_temporal_admg_given_bayesian_coverage: 0x52E5_0079, TemporalAdmg, Average, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    response_jacobian_temporal_admg_given_frequentist_coverage: 0x52E5_007A, TemporalAdmg, Jacobian, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    response_jacobian_temporal_admg_given_bayesian_coverage: 0x52E5_007B, TemporalAdmg, Jacobian, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    directional_derivative_temporal_admg_given_frequentist_coverage: 0x52E5_007C, TemporalAdmg, Directional, Frequentist, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
    directional_derivative_temporal_admg_given_bayesian_coverage: 0x52E5_007D, TemporalAdmg, Directional, Bayesian, [
        (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
        (Explicit, Full), (Accepted, Full),
    ];
}
