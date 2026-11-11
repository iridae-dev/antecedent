//! Repeated-sampling coverage of the mechanism and design coordinates on every graph class,
//! structure route and validation mode they are offered under: anomaly and change attribution
//! (`gcm.fit`, `gcm.fit.bayesian`, `gcm.attribution.bayesian`), the per-unit counterfactual's
//! mean effect (`gcm.fit`), the trial-to-target transported effect (`transport.trial_ipw`,
//! `transport.trial_bayesian_bootstrap`) and the network-interference exposure contrast
//! (`interference.ht_hajek`, `interference.cluster_neyman`, `interference.observational_ipw`,
//! `interference.bayesian_gaussian`, `interference.cluster_bayesian_bootstrap`).
//!
//! One test measures one coordinate: a query, a graph class, an inference mode, an estimator and
//! a structure family (a given graph, or a graph posterior). The validation modes and, for a
//! given graph, its accepted twin are sibling executions of that coordinate: the driver
//! (`common/grid_cell.rs`) measures the first listed execution on every replicate and requires
//! every sibling to report its interval bit for bit.
//!
//! Every truth is the exact population value of the reported functional:
//!
//! * Anomaly attribution scores the target against a fixed reference equal to the target's true
//!   marginal location and scale. The standardized deviation is then standard normal, the
//!   two-sided tail probability `U = 2Φ(−|z|)` is uniform, and the mean score `E[−log U]` is
//!   exactly 1 whatever the structure.
//! * Change attribution contrasts two stacked populations (or two time ranges of one series)
//!   that differ only in the target's intercept, so the total change of the target's mean is the
//!   intercept shift.
//! * The counterfactual's mean effect of `do(a = 1)` against `do(a = 0)` on a linear outcome is
//!   the outcome's coefficient on `a`.
//! * The transported effect is `E[τ(x) | S = 0]`: `1 − 2·E[x σ(0.5x)]` when trial membership
//!   depends on the effect modifier `x` (computed by quadrature of the closed-form integrand),
//!   and the constant effect when membership is random.
//! * Every interference contrast is a constant structural effect of the stated exposure change.
//!
//! A graph posterior holds one atom, except where one atom publishes no interval: a frequentist
//! counterfactual's interval over a posterior is the spread of the mixture of several atoms
//! re-run on shared resampled tables, so that posterior holds two atoms that both identify the
//! same effect.
//!
//! The driver scores the reported 0.95 interval and the 0.90 interval of the same execution: a
//! posterior-quantile interval re-summarized from the retained draws, a normal interval re-formed
//! from the published standard error.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_lines,
    reason = "test scaffolding compares exact constants, indexes rows with small integers and keeps one design table in one body"
)]

mod common;
#[path = "common/grid_cell.rs"]
mod grid_cell;

use std::sync::Arc;

use antecedent::{
    AcceptedGraph, BayesianConfig, CausalPosterior, InferenceMode, InterferenceSpec, RefuteSuite,
    Study, StudyBuilder, StudyResult, TransportTrialSpec,
};
use antecedent_core::{
    AnomalyAttributionQuery, AnomalyReference, AssignmentDesign, CausalQuery,
    ChangeAttributionQuery, ContinuousDomain, CounterfactualQuery, ExecutionContext, ExposureLevel,
    ExposureMapping, ExposurePropensityProvenance, GridSpec, InterferenceFunctional,
    InterferenceQuery, Intervention, Lag, PopulationSelector, ResponseFunctional, ResponseQuery,
    TransportQuery, Value, VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData, TimeSeriesData};
use antecedent_discovery::{
    GraphPosterior, GraphPosteriorAtomKind, LagMask, SamplingSummary, adjacency_mask_from_admg,
    adjacency_mask_from_cpdag, adjacency_masks_from_pag, set_edge, with_lagged_edge,
};
use antecedent_graph::{
    Admg, Cpdag, Dag, DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag, TemporalAdmg,
    TemporalCpdag, TemporalDag, TemporalPag, ensure_lagged,
};
use antecedent_prob::InferenceDiagnostics;
use common::calibration::{GRID_POINTS, gaussian, grid_n, stream_seed, unit_uniform};
use common::reported::{posterior_pair, scalar_normal_pair};
use common::static_dgp::{sigmoid, uniform};
use grid_cell::{Cell, IntervalPair, Target, measure, measure_at};

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn table(columns: &[&str], values: &[Vec<f64>]) -> TabularData {
    TabularData::from_f64_columns(
        columns.iter().zip(values).map(|(name, col)| (*name, col.as_slice())),
    )
    .unwrap()
}

fn series(columns: &[&str], values: &[Vec<f64>]) -> TimeSeriesData {
    TimeSeriesData::from_f64_columns(
        columns.iter().zip(values).map(|(name, col)| (*name, col.as_slice())),
        1,
    )
    .unwrap()
}

fn sampling() -> SamplingSummary {
    SamplingSummary {
        ess: 1.0,
        diagnostics: InferenceDiagnostics::analytic("grid_mechanism_design_atom"),
        rejected_invalid: 0,
    }
}

/// Directed edges `from -> to` packed into one adjacency word on `nodes` variables.
fn directed_mask(nodes: usize, edges: &[(usize, usize)]) -> u64 {
    edges.iter().fold(0_u64, |mask, &(from, to)| set_edge(mask, nodes, from, to, true))
}

/// A graph posterior of one or more atoms of `kind` over `nodes` variables: contemporaneous
/// adjacency words, optional PAG circle words, and lag-one directed edges shared by every atom
/// (a lagged temporal-ADMG atom also carries an empty lagged latent-confounder set).
fn posterior(
    nodes: usize,
    weights: &[f64],
    atoms: Vec<u64>,
    kind: GraphPosteriorAtomKind,
    circles: Option<u64>,
    lags: &[(usize, usize)],
) -> GraphPosterior {
    let temporal_admg = kind == GraphPosteriorAtomKind::TemporalAdmg;
    let count = atoms.len();
    let mut graph = GraphPosterior::new(
        nodes,
        weights.to_vec(),
        atoms,
        vec![0.0; nodes * nodes],
        vec![0.0; nodes * nodes],
        sampling(),
    )
    .unwrap()
    .with_atom_kind(kind);
    if let Some(circles) = circles {
        graph = graph.with_mark_masks(vec![circles; count]).unwrap();
    }
    if !lags.is_empty() {
        let lagged = lags.iter().fold(LagMask::EMPTY, |mask, &(from, to)| {
            with_lagged_edge(mask, nodes, 1, from, to, true).unwrap()
        });
        graph = graph
            .with_lagged_marginals(1, vec![0.0; nodes * nodes])
            .unwrap()
            .with_lag_masks(vec![lagged; count])
            .unwrap();
        if temporal_admg {
            graph = graph.with_lag_bidirected_masks(vec![LagMask::EMPTY; count]).unwrap();
        }
    }
    graph
}

// ---------------------------------------------------------------- structures

/// The graph class of a coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    Dag,
    Cpdag,
    Pag,
    Admg,
    TemporalDag,
    TemporalCpdag,
    TemporalPag,
    TemporalAdmg,
}

impl Class {
    fn atom_kind(self) -> GraphPosteriorAtomKind {
        match self {
            Self::Dag | Self::TemporalDag => GraphPosteriorAtomKind::Dag,
            Self::Cpdag | Self::TemporalCpdag => GraphPosteriorAtomKind::Cpdag,
            Self::Pag | Self::TemporalPag => GraphPosteriorAtomKind::Pag,
            Self::Admg => GraphPosteriorAtomKind::Admg,
            Self::TemporalAdmg => GraphPosteriorAtomKind::TemporalAdmg,
        }
    }
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

/// The inference mode of a coordinate.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Frequentist,
    Bayesian,
}

/// A given structure in its own class.
enum Structure {
    Dag(Dag),
    Cpdag(Cpdag),
    Pag(Pag),
    Admg(Admg),
    TemporalDag(TemporalDag),
    TemporalCpdag(TemporalCpdag),
    TemporalPag(TemporalPag),
    TemporalAdmg(TemporalAdmg),
}

macro_rules! supplied {
    ($builder:expr, $graph:expr, $accepted:expr) => {
        if $accepted {
            $builder.graph(AcceptedGraph::from($graph.clone()))
        } else {
            $builder.graph($graph.clone())
        }
    };
}

impl Structure {
    fn bind(&self, builder: StudyBuilder, accepted: bool) -> StudyBuilder {
        match self {
            Self::Dag(g) => supplied!(builder, g, accepted),
            Self::Cpdag(g) => supplied!(builder, g, accepted),
            Self::Pag(g) => supplied!(builder, g, accepted),
            Self::Admg(g) => supplied!(builder, g, accepted),
            Self::TemporalDag(g) => supplied!(builder, g, accepted),
            Self::TemporalCpdag(g) => supplied!(builder, g, accepted),
            Self::TemporalPag(g) => supplied!(builder, g, accepted),
            Self::TemporalAdmg(g) => supplied!(builder, g, accepted),
        }
    }
}

/// The structure of a coordinate: a given graph (explicit or accepted) or a graph posterior.
enum Graph {
    Given(Structure),
    Posterior(GraphPosterior),
}

impl Graph {
    fn attach(&self, builder: StudyBuilder, via: Via) -> StudyBuilder {
        match (self, via) {
            (Self::Posterior(graph), Posterior) => builder.graph_posterior(graph.clone()),
            (Self::Given(structure), Explicit) => structure.bind(builder, false),
            (Self::Given(structure), Accepted) => structure.bind(builder, true),
            (Self::Given(_), Posterior) | (Self::Posterior(_), Explicit | Accepted) => {
                panic!("a coordinate's variants share its structure family")
            }
        }
    }
}

fn dag(nodes: u32, edges: &[(u32, u32)]) -> Dag {
    let mut graph = Dag::with_variables(nodes);
    for &(from, to) in edges {
        graph.insert_directed(d(from), d(to)).unwrap();
    }
    graph
}

fn admg(nodes: u32, directed: &[(u32, u32)], bidirected: &[(u32, u32)]) -> Admg {
    let mut graph = Admg::with_variables(nodes);
    for &(from, to) in directed {
        graph.insert_directed(d(from), d(to)).unwrap();
    }
    for &(left, right) in bidirected {
        graph.insert_bidirected(d(left), d(right)).unwrap();
    }
    graph
}

fn marked(a: DenseNodeId, b: DenseNodeId, at_a: Endpoint, at_b: Endpoint) -> MarkedEdge {
    MarkedEdge { a, b, at_a, at_b, middle: MiddleMark::Empty }
}

/// The data a study reads.
enum Rows {
    Table(TabularData),
    Series(TimeSeriesData),
}

impl Rows {
    fn builder(&self) -> StudyBuilder {
        match self {
            Self::Table(data) => Study::tabular(data.clone()),
            Self::Series(data) => Study::series(data.clone()),
        }
    }
}

fn execute(test: &str, builder: StudyBuilder, variant: Variant, seed: u64) -> (Study, StudyResult) {
    execute_in(test, builder, variant, &ExecutionContext::for_tests(seed))
}

fn execute_in(
    test: &str,
    builder: StudyBuilder,
    (via, suite): Variant,
    ctx: &ExecutionContext,
) -> (Study, StudyResult) {
    let study =
        builder.refute(suite).build().unwrap_or_else(|e| panic!("{test} {via:?} {suite:?}: {e}"));
    let result = study.run(ctx).unwrap_or_else(|e| panic!("{test} {via:?} {suite:?}: {e}"));
    (study, result)
}

fn effect_pair(result: &StudyResult) -> IntervalPair {
    result
        .posterior
        .as_ref()
        .and_then(CausalPosterior::effect_column)
        .map_or([None, None], |column| posterior_pair(result, column))
}

// ================================================================ attribution

/// Residual scale of every attribution target.
const NOISE: f64 = 0.7;
/// The anomaly target's intercept: the fixed reference's center.
const CENTER: f64 = 1.0;
/// The anomaly mean score against the target's true location and scale: `E[−log U]` for a
/// uniform two-sided tail probability `U`, i.e. the mean of an `Exp(1)` variable.
const ANOMALY_TRUTH: f64 = 1.0;
/// Intercepts of the baseline and comparison populations of a change attribution.
const BASELINE: f64 = 1.0;
const COMPARISON: f64 = 3.0;
/// Rows of an anomaly design.
const N_ANOMALY: usize = 1000;
/// Rows of each population of a change design.
const N_POPULATION: usize = 600;
/// Posterior draws of a Bayesian attribution.
const ATTRIBUTION_DRAWS: usize = 400;
/// Circular-block bootstrap replicates of a frequentist attribution interval.
const ATTRIBUTION_REPLICATES: u32 = 200;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum AttrKind {
    Anomaly,
    Change,
}

/// The data laws of the attribution designs. In every law the target is the last column and
/// `c` is its intercept (constant for an anomaly design, the population's for a change design);
/// `e` is a fresh standard normal draw.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Law {
    /// `z, y`: `y = c + 2z + 0.7e`.
    Chain,
    /// `z, w, y`: `y = c + z + w + 0.7e`.
    Collider,
    /// `a, b, t`: a latent `u` moves `a = u + e` and `b = u + e`; `t = c + a + b + 0.7e`.
    ConfoundedCollider,
    /// `x, y`: a latent `u` moves `x = u + e` and `y = c + u + 0.7e`.
    ConfoundedPair,
    /// Series `x, y`: `y_t = c + 2x_{t−1} + 0.7e`.
    LagChain,
    /// Series `x, y`: a latent `u_t` moves `x_t = u_t + e` and `y_t = c + 2x_{t−1} + u_t + 0.7e`.
    LagConfoundedPair,
    /// Series `a, b, t`: a latent `u_t` moves `a_t = u_t + e`, `b_t = u_t + e`;
    /// `t_t = c + a_{t−1} + b_{t−1} + 0.7e`.
    LagConfoundedCollider,
    /// Series `a, b, t`: `t_t = c + a_{t−1} + b_t + 0.7e`.
    LagMixed,
}

impl Law {
    fn name(self) -> &'static str {
        match self {
            Self::Chain => "attribution_chain",
            Self::Collider => "attribution_collider",
            Self::ConfoundedCollider => "attribution_confounded_collider",
            Self::ConfoundedPair => "attribution_confounded_pair",
            Self::LagChain => "attribution_lag_chain",
            Self::LagConfoundedPair => "attribution_lag_confounded_pair",
            Self::LagConfoundedCollider => "attribution_lag_confounded_collider",
            Self::LagMixed => "attribution_lag_mixed",
        }
    }

    fn columns(self) -> &'static [&'static str] {
        match self {
            Self::Chain => &["z", "y"],
            Self::Collider => &["z", "w", "y"],
            Self::ConfoundedPair | Self::LagChain | Self::LagConfoundedPair => &["x", "y"],
            Self::ConfoundedCollider | Self::LagConfoundedCollider | Self::LagMixed => {
                &["a", "b", "t"]
            }
        }
    }

    fn target(self) -> u32 {
        self.columns().len() as u32 - 1
    }

    fn is_series(self) -> bool {
        matches!(
            self,
            Self::LagChain | Self::LagConfoundedPair | Self::LagConfoundedCollider | Self::LagMixed
        )
    }

    /// Exact variance of the target's (stationary) marginal law.
    fn target_variance(self) -> f64 {
        let noise = NOISE * NOISE;
        match self {
            // 4·Var z (or 4·Var x_{t−1}) plus the residual.
            Self::Chain | Self::LagChain => 4.0 + noise,
            Self::Collider => 2.0 + noise,
            // Var(a + b) = Var(2u + e1 + e2) = 4 + 2.
            Self::ConfoundedCollider | Self::LagConfoundedCollider => 6.0 + noise,
            Self::ConfoundedPair => 1.0 + noise,
            // 4·Var x_{t−1} = 4·2, plus Var u_t = 1 (independent of x_{t−1}).
            Self::LagConfoundedPair => 8.0 + 1.0 + noise,
            Self::LagMixed => 2.0 + noise,
        }
    }

    /// `rows` rows of the law, the target's intercept `intercept(row)`. A series starts from a
    /// draw of its stationary law, so every row's target has the stated marginal law.
    fn rows(self, rows: usize, intercept: impl Fn(usize) -> f64, seed: u64) -> Rows {
        let mut g = gaussian(seed);
        let width = self.columns().len();
        let mut cols: Vec<Vec<f64>> = (0..width).map(|_| Vec::with_capacity(rows)).collect();
        let mut previous: [f64; 2] = match self {
            Self::LagChain | Self::LagMixed => [g(), 0.0],
            Self::LagConfoundedPair => {
                let u = g();
                [u + g(), 0.0]
            }
            Self::LagConfoundedCollider => {
                let u = g();
                [u + g(), u + g()]
            }
            _ => [0.0, 0.0],
        };
        for i in 0..rows {
            let c = intercept(i);
            match self {
                Self::Chain => {
                    let z = g();
                    cols[0].push(z);
                    cols[1].push(c + 2.0 * z + NOISE * g());
                }
                Self::Collider => {
                    let (z, w) = (g(), g());
                    cols[0].push(z);
                    cols[1].push(w);
                    cols[2].push(c + z + w + NOISE * g());
                }
                Self::ConfoundedCollider => {
                    let u = g();
                    let (a, b) = (u + g(), u + g());
                    cols[0].push(a);
                    cols[1].push(b);
                    cols[2].push(c + a + b + NOISE * g());
                }
                Self::ConfoundedPair => {
                    let u = g();
                    cols[0].push(u + g());
                    cols[1].push(c + u + NOISE * g());
                }
                Self::LagChain => {
                    let x = g();
                    cols[0].push(x);
                    cols[1].push(c + 2.0 * previous[0] + NOISE * g());
                    previous[0] = x;
                }
                Self::LagConfoundedPair => {
                    let u = g();
                    let x = u + g();
                    cols[0].push(x);
                    cols[1].push(c + 2.0 * previous[0] + u + NOISE * g());
                    previous[0] = x;
                }
                Self::LagConfoundedCollider => {
                    let u = g();
                    let (a, b) = (u + g(), u + g());
                    cols[0].push(a);
                    cols[1].push(b);
                    cols[2].push(c + previous[0] + previous[1] + NOISE * g());
                    previous = [a, b];
                }
                Self::LagMixed => {
                    let (a, b) = (g(), g());
                    cols[0].push(a);
                    cols[1].push(b);
                    cols[2].push(c + previous[0] + b + NOISE * g());
                    previous[0] = a;
                }
            }
        }
        if self.is_series() {
            Rows::Series(series(self.columns(), &cols))
        } else {
            Rows::Table(table(self.columns(), &cols))
        }
    }
}

/// `z -> y`.
fn chain_dag() -> Dag {
    dag(2, &[(0, 1)])
}

/// `z -> y <- w` (columns `z, w, y`): the compelled collider, a class of one DAG.
fn collider_cpdag() -> Cpdag {
    Cpdag::from_dag(&dag(3, &[(0, 2), (1, 2)]))
}

/// `a -> t <- b` with `a <-> b`.
fn confounded_collider_admg() -> Admg {
    admg(3, &[(0, 2), (1, 2)], &[(0, 1)])
}

/// The given structure of an attribution coordinate and the law it is measured on.
fn attribution_given(class: Class) -> (Law, Structure) {
    let lag = || Lag::from_raw(1);
    let now = || Lag::CONTEMPORANEOUS;
    match class {
        Class::Dag => (Law::Chain, Structure::Dag(chain_dag())),
        Class::Cpdag => (Law::Collider, Structure::Cpdag(collider_cpdag())),
        // `z o-> y`: one completion is the chain, the other a confounded pair whose mechanism a
        // hedge leaves unidentified; the identified member publishes the posterior.
        Class::Pag => {
            let mut pag = Pag::with_variables(2);
            pag.insert_marked(marked(d(0), d(1), Endpoint::Circle, Endpoint::Arrow)).unwrap();
            (Law::Chain, Structure::Pag(pag))
        }
        Class::Admg => (Law::ConfoundedCollider, Structure::Admg(confounded_collider_admg())),
        Class::TemporalDag => {
            let mut graph = TemporalDag::empty();
            let parent = ensure_lagged(&mut graph, v(0), lag()).unwrap();
            let child = ensure_lagged(&mut graph, v(1), now()).unwrap();
            graph.insert_directed(parent, child).unwrap();
            (Law::LagChain, Structure::TemporalDag(graph))
        }
        Class::TemporalCpdag => {
            let mut graph = TemporalCpdag::empty();
            let parent = graph.add_lagged(v(0), lag()).unwrap();
            let child = graph.add_lagged(v(1), now()).unwrap();
            graph.insert_directed(parent, child).unwrap();
            (Law::LagChain, Structure::TemporalCpdag(graph))
        }
        Class::TemporalPag => {
            let mut graph = TemporalPag::empty();
            let parent = graph.add_lagged(v(0), lag()).unwrap();
            let child = graph.add_lagged(v(1), now()).unwrap();
            graph.insert_marked(marked(parent, child, Endpoint::Circle, Endpoint::Arrow)).unwrap();
            (Law::LagChain, Structure::TemporalPag(graph))
        }
        // `a_{t−1} -> t_t <- b_{t−1}` with `a <-> b` in every period.
        Class::TemporalAdmg => {
            let mut graph = TemporalAdmg::empty();
            let a = graph.add_lagged(v(0), lag()).unwrap();
            let b = graph.add_lagged(v(1), lag()).unwrap();
            let t = graph.add_lagged(v(2), now()).unwrap();
            graph.insert_directed(a, t).unwrap();
            graph.insert_directed(b, t).unwrap();
            graph.insert_bidirected(a, b).unwrap();
            (Law::LagConfoundedCollider, Structure::TemporalAdmg(graph))
        }
    }
}

/// The one-atom graph posterior of an attribution coordinate and the law it is measured on.
fn attribution_posterior(class: Class) -> (Law, GraphPosterior) {
    let kind = class.atom_kind();
    // Both directed bits of a pair: a bidirected edge.
    let confounded = directed_mask(2, &[(0, 1), (1, 0)]);
    match class {
        Class::Dag => {
            (Law::Chain, posterior(2, &[1.0], vec![directed_mask(2, &[(0, 1)])], kind, None, &[]))
        }
        Class::Cpdag => {
            let atom = adjacency_mask_from_cpdag(&collider_cpdag()).unwrap();
            (Law::Collider, posterior(3, &[1.0], vec![atom], kind, None, &[]))
        }
        Class::Pag => {
            (Law::ConfoundedPair, posterior(2, &[1.0], vec![confounded], kind, Some(0), &[]))
        }
        Class::Admg => (
            Law::ConfoundedCollider,
            posterior(
                3,
                &[1.0],
                vec![adjacency_mask_from_admg(&confounded_collider_admg()).unwrap()],
                kind,
                None,
                &[],
            ),
        ),
        Class::TemporalDag | Class::TemporalCpdag => {
            (Law::LagChain, posterior(2, &[1.0], vec![0], kind, None, &[(0, 1)]))
        }
        // `x_t <-> y_t` and `x_{t−1} -> y_t`.
        Class::TemporalPag => (
            Law::LagConfoundedPair,
            posterior(2, &[1.0], vec![confounded], kind, Some(0), &[(0, 1)]),
        ),
        // `b_t -> t_t` and `a_{t−1} -> t_t`.
        Class::TemporalAdmg => (
            Law::LagMixed,
            posterior(3, &[1.0], vec![directed_mask(3, &[(1, 2)])], kind, None, &[(0, 2)]),
        ),
    }
}

/// Measure one attribution coordinate.
fn attribution(
    test: &'static str,
    family: u64,
    kind: AttrKind,
    class: Class,
    mode: Mode,
    variants: &[Variant],
) {
    let (law, graph) = if variants[0].0 == Posterior {
        let (law, graph) = attribution_posterior(class);
        (law, Graph::Posterior(graph))
    } else {
        let (law, structure) = attribution_given(class);
        (law, Graph::Given(structure))
    };
    let target = v(law.target());
    let (truth, estimator) = match (kind, mode) {
        (AttrKind::Anomaly, Mode::Bayesian) => (ANOMALY_TRUTH, "gcm.fit.bayesian"),
        (AttrKind::Change, Mode::Bayesian) => (COMPARISON - BASELINE, "gcm.attribution.bayesian"),
        (AttrKind::Anomaly, Mode::Frequentist) => (ANOMALY_TRUTH, "gcm.fit"),
        (AttrKind::Change, Mode::Frequentist) => (COMPARISON - BASELINE, "gcm.fit"),
    };
    let cell = Cell {
        test,
        dgp: law.name(),
        interval: match mode {
            // The whole-procedure circular-block bootstrap of the lag-aligned times.
            Mode::Frequentist => "circular_block_se",
            Mode::Bayesian => "posterior_quantile",
        },
        estimator,
        variants: variants.len(),
    };
    let inference = match mode {
        Mode::Frequentist => InferenceMode::Frequentist,
        Mode::Bayesian => {
            InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(ATTRIBUTION_DRAWS))
        }
    };
    measure(
        cell,
        &Target::scalar(truth),
        |rep, variant| {
            let seed = stream_seed(family, rep);
            let (rows, query) = match kind {
                AttrKind::Anomaly => (
                    law.rows(grid_n(N_ANOMALY), |_| CENTER, seed),
                    CausalQuery::AnomalyAttribution(
                        AnomalyAttributionQuery::new([target], 1_000_000).with_reference(
                            AnomalyReference::fixed(CENTER, law.target_variance().sqrt()),
                        ),
                    ),
                ),
                AttrKind::Change => {
                    let pop = grid_n(N_POPULATION);
                    (
                        law.rows(2 * pop, |i| if i < pop { BASELINE } else { COMPARISON }, seed),
                        CausalQuery::ChangeAttribution(ChangeAttributionQuery::new(
                            target,
                            PopulationSelector::TimeRange { start: 0, end: pop },
                            PopulationSelector::TimeRange { start: pop, end: 2 * pop },
                        )),
                    )
                }
            };
            let mut builder = graph
                .attach(rows.builder(), variants[variant].0)
                .query(query)
                .inference(inference.clone());
            if mode == Mode::Frequentist {
                builder = builder.bootstrap_replicates(ATTRIBUTION_REPLICATES);
            }
            execute(test, builder, variants[variant], seed)
        },
        |result| match mode {
            Mode::Bayesian => vec![posterior_pair(result, 0)],
            Mode::Frequentist => vec![scalar_normal_pair(result)],
        },
    );
}

// ============================================================= counterfactual

/// The outcome's coefficient on `a`: the mean effect of `do(a = 1)` against `do(a = 0)`.
const CF_EFFECT: f64 = 1.5;
/// Rows of a counterfactual design.
const N_COUNTERFACTUAL: usize = 500;
/// Posterior draws of a Bayesian counterfactual.
const CF_DRAWS: usize = 400;
/// Shared resampled tables behind a frequentist mixture's standard error.
const CF_REPLICATES: u32 = 100;

/// Columns `a, z, w, y`: `a, z, w ~ N(0,1)` independent, `y = 0.5 + 1.5a + z + e`. `w` is an
/// independent covariate with no effect on `y`.
fn counterfactual_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut cols: Vec<Vec<f64>> = (0..4).map(|_| Vec::with_capacity(n)).collect();
    for _ in 0..n {
        let (a, z, w) = (g(), g(), g());
        cols[0].push(a);
        cols[1].push(z);
        cols[2].push(w);
        cols[3].push(0.5 + CF_EFFECT * a + z + g());
    }
    table(&["a", "z", "w", "y"], &cols)
}

/// `a -> y <- z`, and with `with_w` also `w -> y` (its coefficient is zero): both are causal
/// structures of [`counterfactual_data`] with the same effect of `a` on `y`.
fn counterfactual_dag(with_w: bool) -> Dag {
    let mut graph = Dag::with_variables(4);
    for &(from, to) in counterfactual_edges(with_w) {
        graph.insert_directed(d(from as u32), d(to as u32)).unwrap();
    }
    graph
}

/// The edges of [`counterfactual_dag`].
fn counterfactual_edges(with_w: bool) -> &'static [(usize, usize)] {
    if with_w { &[(0, 3), (1, 3), (2, 3)] } else { &[(0, 3), (1, 3)] }
}

fn counterfactual_atom(class: Class, with_w: bool) -> u64 {
    match class {
        Class::Dag => directed_mask(4, counterfactual_edges(with_w)),
        Class::Cpdag => {
            adjacency_mask_from_cpdag(&Cpdag::from_dag(&counterfactual_dag(with_w))).unwrap()
        }
        other => panic!("no counterfactual coordinate on {other:?}"),
    }
}

/// Measure one counterfactual coordinate.
fn counterfactual(test: &'static str, family: u64, class: Class, mode: Mode, variants: &[Variant]) {
    let graph = match (variants[0].0, mode) {
        (Posterior, Mode::Bayesian) => Graph::Posterior(posterior(
            4,
            &[1.0],
            vec![counterfactual_atom(class, false)],
            class.atom_kind(),
            None,
            &[],
        )),
        // One atom publishes no frequentist interval: two atoms that identify the same effect.
        (Posterior, Mode::Frequentist) => Graph::Posterior(posterior(
            4,
            &[0.6, 0.4],
            vec![counterfactual_atom(class, false), counterfactual_atom(class, true)],
            class.atom_kind(),
            None,
            &[],
        )),
        (_, _) => Graph::Given(match class {
            Class::Dag => Structure::Dag(counterfactual_dag(false)),
            Class::Cpdag => Structure::Cpdag(Cpdag::from_dag(&counterfactual_dag(false))),
            other => panic!("no counterfactual coordinate on {other:?}"),
        }),
    };
    let cell = Cell {
        test,
        dgp: "counterfactual_data",
        interval: match mode {
            Mode::Frequentist => "bootstrap_se",
            Mode::Bayesian => "posterior_quantile",
        },
        estimator: "gcm.fit",
        variants: variants.len(),
    };
    let query = CausalQuery::Counterfactual(
        CounterfactualQuery::new(v(3), Arc::from([Intervention::set(v(0), Value::f64(1.0))]))
            .with_control_level(0.0),
    );
    measure(
        cell,
        &Target::scalar(CF_EFFECT),
        |rep, variant| {
            let seed = stream_seed(family, rep);
            let data = counterfactual_data(grid_n(N_COUNTERFACTUAL), seed);
            let builder =
                graph.attach(Study::tabular(data), variants[variant].0).query(query.clone());
            let builder = match mode {
                Mode::Frequentist => builder
                    .inference(InferenceMode::Frequentist)
                    .bootstrap_replicates(CF_REPLICATES),
                Mode::Bayesian => builder.inference(InferenceMode::Bayesian(
                    BayesianConfig::laplace().n_draws(CF_DRAWS),
                )),
            };
            execute(test, builder, variants[variant], seed)
        },
        |result| match mode {
            Mode::Bayesian => vec![effect_pair(result)],
            Mode::Frequentist => vec![scalar_normal_pair(result)],
        },
    );
}

// ================================================================ transport

/// Rows of a frequentist transport design (trial and target drawn together).
const N_TRANSPORT: usize = 1000;
/// Trial rows, and as many target rows, of a Bayesian transport design.
const N_TRANSPORT_EACH: usize = 2000;
/// Posterior draws of the Bayesian bootstrap.
const TRANSPORT_DRAWS: usize = 1000;
/// The constant effect of the random-membership design.
const DIRECT_EFFECT: f64 = 1.5;

/// Columns `a(0), y(1), trial(2), s(3), e(4), x(5)` of every transport design.
const TRANSPORT_COLUMNS: [&str; 6] = ["a", "y", "trial", "s", "e", "x"];

/// Population `x ~ N(0,1)`. With `modified`, trial membership is `S | x ~ Bern(σ(0.5x))` and
/// the outcome `y = 1 + a(1 + x) + x + ε`, so the effect `τ(x) = 1 + x` is modified by the
/// covariate whose law differs between trial and target; otherwise membership is `Bern(1/2)`
/// independent of `x` and `y = 1 + 1.5a + ε`.
fn selection_probability(x: f64, modified: bool) -> f64 {
    if modified { sigmoid(0.5 * x) } else { 0.5 }
}

fn transport_outcome(a: f64, x: f64, noise: f64, modified: bool) -> f64 {
    if modified { 1.0 + a * (1.0 + x) + x + noise } else { 1.0 + DIRECT_EFFECT * a + noise }
}

fn push_transport_row(cols: &mut [Vec<f64>], trial: bool, a: f64, y: f64, s: f64, x: f64) {
    for (col, value) in cols.iter_mut().zip([a, y, f64::from(u8::from(trial)), s, 0.5, x]) {
        col.push(value);
    }
}

/// `n` independent population rows: each joins the trial with its selection probability; a
/// trial row is randomized `a ~ Bern(1/2)` (known, column `e`) and observed; a target row
/// carries no treatment or outcome.
fn transport_iid(n: usize, seed: u64, modified: bool) -> TabularData {
    let mut g = gaussian(seed);
    let mut cols: Vec<Vec<f64>> = (0..6).map(|_| Vec::with_capacity(n)).collect();
    for i in 0..n {
        let x = g();
        let s = selection_probability(x, modified);
        let trial = unit_uniform(stream_seed(seed, 2 * i as u64)) < s;
        let a = if trial {
            f64::from(u8::from(unit_uniform(stream_seed(seed, 2 * i as u64 + 1)) < 0.5))
        } else {
            0.0
        };
        let noise = g();
        let y = if trial { transport_outcome(a, x, noise, modified) } else { 0.0 };
        push_transport_row(&mut cols, trial, a, y, s, x);
    }
    table(&TRANSPORT_COLUMNS, &cols)
}

/// One population draw of `x` conditional on membership: rejection from the population law with
/// acceptance probability `s(x)` (trial) or `1 − s(x)` (target).
fn member_x(
    g: &mut impl FnMut() -> f64,
    u: &mut impl FnMut() -> f64,
    trial: bool,
    modified: bool,
) -> f64 {
    loop {
        let x = g();
        let s = selection_probability(x, modified);
        let accept = u();
        if (trial && accept < s) || (!trial && accept >= s) {
            return x;
        }
    }
}

/// `n` trial rows drawn from `x | S = 1` and `n` target rows from `x | S = 0`, interleaved.
///
/// The Bayesian bootstrap conditions on the trial and target sizes, and the trial-to-target
/// weighting divides the selection-odds-weighted trial sum by the target size. With both sizes
/// fixed in the population ratio `P(S = 1) / P(S = 0)` (one: `E[σ(0.5x)] = 1/2` by symmetry),
/// the weighting's expectation is exactly `E[τ(x) | S = 0]` and the only sampling variation is
/// the trial rows' — the uncertainty the posterior propagates. The target rows enter the
/// weighting only through their count.
fn transport_fixed(n: usize, seed: u64, modified: bool) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(stream_seed(seed, 0x7A5E));
    let mut cols: Vec<Vec<f64>> = (0..6).map(|_| Vec::with_capacity(2 * n)).collect();
    for _ in 0..n {
        let x = member_x(&mut g, &mut u, false, modified);
        push_transport_row(&mut cols, false, 0.0, 0.0, selection_probability(x, modified), x);
        let x = member_x(&mut g, &mut u, true, modified);
        let a = f64::from(u8::from(u() < 0.5));
        let y = transport_outcome(a, x, g(), modified);
        push_transport_row(&mut cols, true, a, y, selection_probability(x, modified), x);
    }
    table(&TRANSPORT_COLUMNS, &cols)
}

/// `E[τ(x) | S = 0] = 1 − 2·E[x σ(0.5x)]` for `x ~ N(0,1)`, by the trapezoid rule on
/// `[−12, 12]`.
fn transport_truth() -> f64 {
    let steps = 240_000;
    let h = 24.0 / f64::from(steps);
    let mut acc = 0.0;
    for k in 0..=steps {
        let x = -12.0 + h * f64::from(k);
        let w = if k == 0 || k == steps { 0.5 } else { 1.0 };
        let phi = (-0.5 * x * x).exp() / (2.0 * std::f64::consts::PI).sqrt();
        acc += w * x * sigmoid(0.5 * x) * phi;
    }
    1.0 - 2.0 * acc * h
}

/// `a -> y <- x` on the six transport columns.
fn transport_dag() -> Dag {
    dag(6, &[(0, 1), (5, 1)])
}

fn transport_admg() -> Admg {
    admg(6, &[(0, 1), (5, 1)], &[])
}

/// `a o-o y`: every completion, and every ADMG sharing its m-separation model, certifies the
/// direct formula when no selection target is declared.
fn transport_pag() -> Pag {
    let mut pag = Pag::with_variables(6);
    pag.insert_circle_circle(d(0), d(1)).unwrap();
    pag
}

/// Measure one transport coordinate. The Pag coordinates are measured on the random-membership
/// design with no selection target (the direct formula); every other class on the
/// effect-modified design with the covariate `x` as the selection target (an S-admissible
/// standardization over `x`).
fn transport(test: &'static str, family: u64, class: Class, mode: Mode, variants: &[Variant]) {
    let modified = class != Class::Pag;
    let selection: Arc<[VariableId]> = if modified { Arc::from([v(5)]) } else { Arc::from([]) };
    let graph = if variants[0].0 == Posterior {
        let kind = class.atom_kind();
        Graph::Posterior(match class {
            Class::Dag => {
                posterior(6, &[1.0], vec![directed_mask(6, &[(0, 1), (5, 1)])], kind, None, &[])
            }
            Class::Admg | Class::TemporalAdmg => posterior(
                6,
                &[1.0],
                vec![adjacency_mask_from_admg(&transport_admg()).unwrap()],
                kind,
                None,
                &[],
            ),
            Class::Cpdag => posterior(
                6,
                &[1.0],
                vec![adjacency_mask_from_cpdag(&Cpdag::from_dag(&transport_dag())).unwrap()],
                kind,
                None,
                &[],
            ),
            Class::Pag => {
                let (adjacency, circles) = adjacency_masks_from_pag(&transport_pag()).unwrap();
                posterior(6, &[1.0], vec![adjacency], kind, Some(circles), &[])
            }
            other => panic!("no transport coordinate on {other:?}"),
        })
    } else {
        Graph::Given(match class {
            Class::Dag => Structure::Dag(transport_dag()),
            Class::Admg => Structure::Admg(transport_admg()),
            Class::TemporalAdmg => Structure::TemporalAdmg(
                TemporalAdmg::from_static_contemporaneous(&transport_admg()).unwrap(),
            ),
            Class::Cpdag => Structure::Cpdag(Cpdag::from_dag(&transport_dag())),
            Class::Pag => Structure::Pag(transport_pag()),
            other => panic!("no transport coordinate on {other:?}"),
        })
    };
    let truth = if modified { transport_truth() } else { DIRECT_EFFECT };
    let (dgp, estimator, interval, inference) = match mode {
        Mode::Frequentist => (
            if modified { "transport_iid_modified" } else { "transport_iid_random_membership" },
            "transport.trial_ipw",
            "analytic_se",
            InferenceMode::Frequentist,
        ),
        Mode::Bayesian => (
            if modified { "transport_fixed_modified" } else { "transport_fixed_random_membership" },
            "transport.trial_bayesian_bootstrap",
            "posterior_quantile",
            InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(TRANSPORT_DRAWS)),
        ),
    };
    let cell = Cell { test, dgp, interval, estimator, variants: variants.len() };
    let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: v(1),
        treatment: ContinuousDomain::new(v(0), GridSpec::Values(Arc::from([0.0, 1.0]))),
    });
    let query = CausalQuery::Transport(TransportQuery::new(response, "trial", "target", [v(0)]));
    measure(
        cell,
        &Target::scalar(truth),
        |rep, variant| {
            let seed = stream_seed(family, rep);
            let data = match mode {
                Mode::Frequentist => transport_iid(grid_n(N_TRANSPORT), seed, modified),
                Mode::Bayesian => transport_fixed(grid_n(N_TRANSPORT_EACH), seed, modified),
            };
            let builder = graph
                .attach(Study::tabular(data), variants[variant].0)
                .query(query.clone())
                .selection_targets(Arc::clone(&selection))
                .transport_trial(TransportTrialSpec {
                    trial: v(2),
                    selection_probability: v(3),
                    treatment_probability: v(4),
                })
                .inference(inference.clone())
                .bootstrap_replicates(0);
            execute(test, builder, variants[variant], seed)
        },
        |result| match mode {
            Mode::Bayesian => vec![posterior_pair(result, 0)],
            Mode::Frequentist => vec![scalar_normal_pair(result)],
        },
    );
}

// ============================================================== interference

/// Units of a ring design.
const RING_UNITS: usize = 400;
/// Two-unit clusters of a cluster or observed-exposure design.
const CLUSTERS: usize = 200;
/// The direct effect `(own 0, neighbors 0) -> (own 1, neighbors 0)` of the ring designs.
const RING_DIRECT: f64 = 2.0;
/// The ring designs' effect of each treated neighbor.
const RING_NEIGHBOR: f64 = 0.5;
/// The total effect `(0, 0) -> (1, 1)` of a cluster-randomized design.
const CLUSTER_EFFECT: f64 = 3.0;
/// Own and neighbor effects of the observed-exposure design: the `(0, 0) -> (1, 1)` contrast is
/// their sum.
const OWN_EFFECT: f64 = 2.0;
const NEIGHBOR_EFFECT: f64 = 1.5;
/// Posterior draws of a Bayesian interference execution.
const INTERFERENCE_DRAWS: usize = 2_000;
/// Diffuse isotropic prior of the conjugate Gaussian exposure model.
const PRIOR_SCALE: f64 = 50.0;
/// Exposure-probability work ceiling of every interference execution. The `full` suite's
/// randomization-unbiasedness check re-draws the assignment 199 times (the default when the
/// study requests no replicates) and recomputes both exposure levels' probabilities by 10,000
/// Monte Carlo draws each. The procedure quote is
/// `(n * (bits(n) + 4) + 2 * edges) * draws * levels` with `levels = 2 * (199 + 1)`: at the
/// largest grid point (800 ring units, 1,600 edges) `14,400 * 10,000 * 400 = 5.76e10`, past
/// the 4e9 default, and the two-unit cluster design (800 units, 800 edges) quotes `5.12e10`.
/// The ceiling authorizes that work without changing a draw, a re-randomization or the design.
const EXPOSURE_PROBABILITY_WORK: u64 = 64_000_000_000;
/// Measured coverage of the Horvitz–Thompson contrast's conservative Young bound: the bound
/// replaces the unobservable joint-exposure covariance by an upper bound, and with an exposure
/// probability of 1/8 dividing every scored outcome its interval is about twenty times the
/// effect wide. Every replicate covers.
const YOUNG_BOUND_MEASURED: [f64; GRID_POINTS] = [1.0; GRID_POINTS];

/// The interference estimator a coordinate measures, each on the design that routes to it.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Design {
    /// Bernoulli assignment on a ring, fixed potential outcomes: `interference.ht_hajek`.
    HtHajek,
    /// Complete randomization of two-unit clusters: `interference.cluster_neyman`.
    ClusterNeyman,
    /// Supplied known exposure probabilities on two-unit clusters:
    /// `interference.observational_ipw`.
    Observational,
    /// Bernoulli assignment on a ring, Gaussian outcomes: `interference.bayesian_gaussian`.
    BayesianGaussian,
    /// Complete randomization of two-unit clusters under Bayesian inference:
    /// `interference.cluster_bayesian_bootstrap`.
    ClusterBootstrap,
}

fn ring_edges(units: usize) -> Vec<NetworkEdge> {
    let n = units as u64;
    (0..n)
        .flat_map(|i| {
            [NetworkEdge::new((i + 1) % n, i, 1.0), NetworkEdge::new((i + n - 1) % n, i, 1.0)]
        })
        .collect()
}

fn pair_edges(clusters: usize) -> Vec<NetworkEdge> {
    (0..clusters as u64)
        .flat_map(|c| {
            [NetworkEdge::new(2 * c, 2 * c + 1, 1.0), NetworkEdge::new(2 * c + 1, 2 * c, 1.0)]
        })
        .collect()
}

fn treated_neighbors(assignment: &[bool], i: usize) -> f64 {
    let n = assignment.len();
    f64::from(u8::from(assignment[(i + 1) % n]) + u8::from(assignment[(i + n - 1) % n]))
}

/// The unit table: the outcome alone, or with the two covariate columns a graph posterior's
/// atoms are drawn over.
fn unit_table(y: &[f64], covariates: bool) -> TabularData {
    if !covariates {
        return TabularData::from_f64_columns([("y", y)]).unwrap();
    }
    let z: Vec<f64> = (0..y.len()).map(|row| (row % 2) as f64).collect();
    let w: Vec<f64> = (0..y.len()).map(|row| ((row / 2) % 2) as f64).collect();
    TabularData::from_f64_columns([("y", y), ("z", z.as_slice()), ("w", w.as_slice())]).unwrap()
}

/// Exactly `k / 2` of `k` clusters treated, uniformly over the allocations.
fn complete_randomization(k: usize, seed: u64) -> Vec<bool> {
    let mut order: Vec<usize> = (0..k).collect();
    for i in (1..k).rev() {
        let j = ((unit_uniform(stream_seed(seed, i as u64)) * (i + 1) as f64) as usize).min(i);
        order.swap(i, j);
    }
    let mut treated = vec![false; k];
    for &cluster in &order[..k / 2] {
        treated[cluster] = true;
    }
    treated
}

impl Design {
    fn estimator(self) -> &'static str {
        match self {
            Self::HtHajek => "interference.ht_hajek",
            Self::ClusterNeyman => "interference.cluster_neyman",
            Self::Observational => "interference.observational_ipw",
            Self::BayesianGaussian => "interference.bayesian_gaussian",
            Self::ClusterBootstrap => "interference.cluster_bayesian_bootstrap",
        }
    }

    fn dgp(self) -> &'static str {
        match self {
            Self::HtHajek => "ring_fixed_outcomes",
            Self::BayesianGaussian => "ring_gaussian_outcomes",
            Self::ClusterNeyman | Self::ClusterBootstrap => "paired_cluster_randomization",
            Self::Observational => "paired_known_exposure",
        }
    }

    fn bayesian(self) -> bool {
        matches!(self, Self::BayesianGaussian | Self::ClusterBootstrap)
    }

    fn truth(self) -> f64 {
        match self {
            Self::HtHajek | Self::BayesianGaussian => RING_DIRECT,
            Self::ClusterNeyman | Self::ClusterBootstrap => CLUSTER_EFFECT,
            Self::Observational => OWN_EFFECT + NEIGHBOR_EFFECT,
        }
    }

    fn inference(self) -> InferenceMode {
        match self {
            Self::HtHajek | Self::ClusterNeyman | Self::Observational => InferenceMode::Frequentist,
            Self::BayesianGaussian => InferenceMode::Bayesian(
                BayesianConfig::conjugate().n_draws(INTERFERENCE_DRAWS).prior_scale(PRIOR_SCALE),
            ),
            // The cluster bootstrap reads no prior: its prior is the vanishing-concentration
            // Dirichlet process on the cluster law.
            Self::ClusterBootstrap => {
                InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(INTERFERENCE_DRAWS))
            }
        }
    }

    /// One replicate's unit outcomes, query and spec.
    ///
    /// * Ring designs: a ring of units, unit `i` exposed to `i − 1` and `i + 1`; each replicate
    ///   draws a fresh Bernoulli(1/2) assignment. [`Self::HtHajek`] keeps the finite population's
    ///   potential outcomes `y_i(own, k) = α_i + 2·own + 0.5·k` fixed (`α_i` drawn once);
    ///   [`Self::BayesianGaussian`] draws `y_i = 1 + 2·own + 0.5·k + ε_i`, the estimator's own
    ///   Gaussian exposure model. Both contrasts `(0, 0) -> (1, 0)` are 2.
    /// * Cluster designs: two-unit clusters with mutual edges; each replicate draws cluster
    ///   effects `α_c ~ N(0,1)`, unit noise and a complete randomization of half the clusters;
    ///   `y = α_c + 0.5ε + 3·treated`. The total effect `(0, 0) -> (1, 1)` is 3 for every unit.
    /// * Observed exposure: two-unit clusters, each unit's own treatment a fair coin, so the
    ///   exposures `(0, 0)` and `(1, 1)` each have known probability 1/4;
    ///   `y = α_c + 0.5ε + 2·own + 1.5·neighbor`. The contrast `(0, 0) -> (1, 1)` is 3.5.
    fn draw(
        self,
        seed: u64,
        covariates: bool,
    ) -> (TabularData, InterferenceQuery, InterferenceSpec) {
        let contrast = |from: ExposureLevel, to: ExposureLevel| {
            InterferenceFunctional::ExposureContrast { outcome: v(0), from, to }
        };
        let none = ExposureLevel { own: 0.0, neighbors: 0.0 };
        let direct = ExposureLevel { own: 1.0, neighbors: 0.0 };
        let total = ExposureLevel { own: 1.0, neighbors: 1.0 };
        let (y, assignment, edges, query) = match self {
            Self::HtHajek | Self::BayesianGaussian => {
                let units = grid_n(RING_UNITS);
                let assignment: Vec<bool> =
                    (0..units).map(|i| unit_uniform(stream_seed(seed, i as u64)) < 0.5).collect();
                let y: Vec<f64> = if self == Self::HtHajek {
                    let mut g = gaussian(0x1F7E_A1FA);
                    (0..units)
                        .map(|i| {
                            let alpha = 1.0 + 0.5 * (i as f64).sin() + 0.5 * g();
                            alpha
                                + RING_DIRECT * f64::from(u8::from(assignment[i]))
                                + RING_NEIGHBOR * treated_neighbors(&assignment, i)
                        })
                        .collect()
                } else {
                    let mut g = gaussian(stream_seed(seed, 0x9E37_79B9));
                    (0..units)
                        .map(|i| {
                            1.0 + RING_DIRECT * f64::from(u8::from(assignment[i]))
                                + RING_NEIGHBOR * treated_neighbors(&assignment, i)
                                + g()
                        })
                        .collect()
                };
                let query = InterferenceQuery::new(
                    AssignmentDesign::Bernoulli { probabilities: Arc::from([0.5]) },
                    ExposureMapping::NeighborCount,
                    contrast(none, direct),
                );
                (y, assignment, ring_edges(units), query)
            }
            Self::ClusterNeyman | Self::ClusterBootstrap => {
                let k = grid_n(CLUSTERS);
                let treated = complete_randomization(k, stream_seed(seed, 0xC1));
                let mut g = gaussian(seed);
                let mut y = Vec::with_capacity(2 * k);
                let mut assignment = Vec::with_capacity(2 * k);
                for &arm in &treated {
                    let alpha = g();
                    for _ in 0..2 {
                        y.push(alpha + 0.5 * g() + CLUSTER_EFFECT * f64::from(u8::from(arm)));
                        assignment.push(arm);
                    }
                }
                let labels: Vec<u32> = (0..2 * k).map(|unit| (unit / 2) as u32).collect();
                let query = InterferenceQuery::new(
                    AssignmentDesign::ClusterRandomization {
                        clusters: Arc::from(labels),
                        treated_clusters: k / 2,
                    },
                    ExposureMapping::NeighborFraction,
                    contrast(none, total),
                );
                (y, assignment, pair_edges(k), query)
            }
            Self::Observational => {
                let k = grid_n(CLUSTERS);
                let units = 2 * k;
                let assignment: Vec<bool> =
                    (0..units).map(|i| unit_uniform(stream_seed(seed, i as u64)) < 0.5).collect();
                let mut g = gaussian(seed);
                let mut y = Vec::with_capacity(units);
                for c in 0..k {
                    let alpha = g();
                    for unit in [2 * c, 2 * c + 1] {
                        y.push(
                            alpha
                                + 0.5 * g()
                                + OWN_EFFECT * f64::from(u8::from(assignment[unit]))
                                + NEIGHBOR_EFFECT * f64::from(u8::from(assignment[unit ^ 1])),
                        );
                    }
                }
                let labels: Vec<u32> = (0..units).map(|unit| (unit / 2) as u32).collect();
                let query = InterferenceQuery::new(
                    AssignmentDesign::ObservedExposure {
                        clusters: Arc::from(labels),
                        propensity_from: Arc::from(vec![0.25; units]),
                        propensity_to: Arc::from(vec![0.25; units]),
                        provenance: ExposurePropensityProvenance::Known,
                        assume_network_exchangeability: true,
                    },
                    ExposureMapping::NeighborCount,
                    contrast(none, total),
                );
                (y, assignment, pair_edges(k), query)
            }
        };
        let units = unit_table(&y, covariates);
        let network = NetworkData::try_new(units.clone(), edges).unwrap();
        (units, query, InterferenceSpec { network, assignment: Arc::from(assignment) })
    }
}

/// Measure one interference coordinate. The exposure contrast is identified by the known
/// assignment mechanism and the exposure mapping, so the structure binds only the unit schema: a
/// given structure is the one-variable graph of its class over the outcome, a graph posterior one
/// atom of its kind over the outcome and two covariates.
fn interference(
    test: &'static str,
    family: u64,
    class: Class,
    design: Design,
    variants: &[Variant],
) {
    let on_posterior = variants[0].0 == Posterior;
    let graph = if on_posterior {
        let atom = directed_mask(3, &[(0, 1)]);
        Graph::Posterior(posterior(3, &[1.0], vec![atom], class.atom_kind(), None, &[]))
    } else {
        Graph::Given(match class {
            Class::Dag => Structure::Dag(Dag::with_variables(1)),
            Class::Admg => Structure::Admg(Admg::with_variables(1)),
            Class::TemporalAdmg => Structure::TemporalAdmg(
                TemporalAdmg::from_static_contemporaneous(&Admg::with_variables(1)).unwrap(),
            ),
            Class::Cpdag => Structure::Cpdag(Cpdag::with_variables(1)),
            Class::Pag => Structure::Pag(Pag::with_variables(1)),
            other => panic!("no interference coordinate on {other:?}"),
        })
    };
    let cell = Cell {
        test,
        dgp: design.dgp(),
        interval: if design.bayesian() { "posterior_quantile" } else { "analytic_se" },
        estimator: design.estimator(),
        variants: variants.len(),
    };
    let inference = design.inference();
    let run = |rep: u64, variant: usize| {
        let seed = stream_seed(family, rep);
        let (units, query, spec) = design.draw(seed, on_posterior);
        let builder = graph
            .attach(Study::tabular(units), variants[variant].0)
            .query(CausalQuery::Interference(query))
            .interference(spec)
            .inference(inference.clone());
        let ctx = ExecutionContext::for_tests(seed)
            .with_exposure_probability_work_limit(EXPOSURE_PROBABILITY_WORK);
        execute_in(test, builder, variants[variant], &ctx)
    };
    let score = |result: &StudyResult| -> Vec<IntervalPair> {
        if design.bayesian() {
            vec![posterior_pair(result, 0)]
        } else {
            vec![scalar_normal_pair(result)]
        }
    };
    let targets = Target::scalar(design.truth());
    if design == Design::HtHajek {
        measure_at(
            cell,
            &targets,
            run,
            score,
            &[[YOUNG_BOUND_MEASURED.map(Some), YOUNG_BOUND_MEASURED.map(Some)]],
        );
    } else {
        measure(cell, &targets, run, score);
    }
}

// ===================================================================== tests

macro_rules! coverage_tests {
    ($(
        $name:ident: $runner:ident($($arg:expr),+ $(,)?) [$(($via:ident, $suite:ident)),+ $(,)?];
    )+) => {$(
        #[test]
        #[ignore = "calibration: run via scripts/measure_calibration.sh"]
        fn $name() {
            $runner(stringify!($name), $($arg),+, &[$(($via, RefuteSuite::$suite)),+]);
        }
    )+};
}

coverage_tests! {
    anomaly_attribution_dag_given_bayesian_coverage:
        attribution(0x4D45_0000, AttrKind::Anomaly, Class::Dag, Mode::Bayesian) [
            (Accepted, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full),
            (Accepted, Full),
        ];
    anomaly_attribution_dag_graph_posterior_bayesian_coverage:
        attribution(0x4D45_0001, AttrKind::Anomaly, Class::Dag, Mode::Bayesian) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    anomaly_attribution_cpdag_given_bayesian_coverage:
        attribution(0x4D45_0002, AttrKind::Anomaly, Class::Cpdag, Mode::Bayesian) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    anomaly_attribution_cpdag_graph_posterior_bayesian_coverage:
        attribution(0x4D45_0003, AttrKind::Anomaly, Class::Cpdag, Mode::Bayesian) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    anomaly_attribution_pag_given_bayesian_coverage:
        attribution(0x4D45_0004, AttrKind::Anomaly, Class::Pag, Mode::Bayesian) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    anomaly_attribution_pag_graph_posterior_bayesian_coverage:
        attribution(0x4D45_0005, AttrKind::Anomaly, Class::Pag, Mode::Bayesian) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    anomaly_attribution_admg_given_bayesian_coverage:
        attribution(0x4D45_0006, AttrKind::Anomaly, Class::Admg, Mode::Bayesian) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    anomaly_attribution_admg_graph_posterior_bayesian_coverage:
        attribution(0x4D45_0007, AttrKind::Anomaly, Class::Admg, Mode::Bayesian) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    anomaly_attribution_temporal_dag_given_bayesian_coverage:
        attribution(0x4D45_0008, AttrKind::Anomaly, Class::TemporalDag, Mode::Bayesian) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    anomaly_attribution_temporal_dag_given_frequentist_coverage:
        attribution(0x4D45_0009, AttrKind::Anomaly, Class::TemporalDag, Mode::Frequentist) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    anomaly_attribution_temporal_dag_graph_posterior_bayesian_coverage:
        attribution(0x4D45_000A, AttrKind::Anomaly, Class::TemporalDag, Mode::Bayesian) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    anomaly_attribution_temporal_cpdag_given_bayesian_coverage:
        attribution(0x4D45_000B, AttrKind::Anomaly, Class::TemporalCpdag, Mode::Bayesian) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    anomaly_attribution_temporal_cpdag_graph_posterior_bayesian_coverage:
        attribution(0x4D45_000C, AttrKind::Anomaly, Class::TemporalCpdag, Mode::Bayesian) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    anomaly_attribution_temporal_pag_given_bayesian_coverage:
        attribution(0x4D45_000D, AttrKind::Anomaly, Class::TemporalPag, Mode::Bayesian) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    anomaly_attribution_temporal_pag_graph_posterior_bayesian_coverage:
        attribution(0x4D45_000E, AttrKind::Anomaly, Class::TemporalPag, Mode::Bayesian) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    anomaly_attribution_temporal_admg_given_bayesian_coverage:
        attribution(0x4D45_000F, AttrKind::Anomaly, Class::TemporalAdmg, Mode::Bayesian) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    anomaly_attribution_temporal_admg_graph_posterior_bayesian_coverage:
        attribution(0x4D45_0010, AttrKind::Anomaly, Class::TemporalAdmg, Mode::Bayesian) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    change_attribution_dag_given_bayesian_coverage:
        attribution(0x4D45_0011, AttrKind::Change, Class::Dag, Mode::Bayesian) [
            (Accepted, None),
        ];
    change_attribution_dag_graph_posterior_bayesian_coverage:
        attribution(0x4D45_0012, AttrKind::Change, Class::Dag, Mode::Bayesian) [
            (Posterior, None),
        ];
    change_attribution_cpdag_given_bayesian_coverage:
        attribution(0x4D45_0013, AttrKind::Change, Class::Cpdag, Mode::Bayesian) [
            (Explicit, None), (Accepted, None),
        ];
    change_attribution_cpdag_graph_posterior_bayesian_coverage:
        attribution(0x4D45_0014, AttrKind::Change, Class::Cpdag, Mode::Bayesian) [
            (Posterior, None),
        ];
    change_attribution_pag_given_bayesian_coverage:
        attribution(0x4D45_0015, AttrKind::Change, Class::Pag, Mode::Bayesian) [
            (Explicit, None), (Accepted, None),
        ];
    change_attribution_pag_graph_posterior_bayesian_coverage:
        attribution(0x4D45_0016, AttrKind::Change, Class::Pag, Mode::Bayesian) [
            (Posterior, None),
        ];
    change_attribution_admg_given_bayesian_coverage:
        attribution(0x4D45_0017, AttrKind::Change, Class::Admg, Mode::Bayesian) [
            (Explicit, None), (Accepted, None),
        ];
    change_attribution_admg_graph_posterior_bayesian_coverage:
        attribution(0x4D45_0018, AttrKind::Change, Class::Admg, Mode::Bayesian) [
            (Posterior, None),
        ];
    change_attribution_temporal_dag_given_bayesian_coverage:
        attribution(0x4D45_0019, AttrKind::Change, Class::TemporalDag, Mode::Bayesian) [
            (Explicit, None), (Accepted, None),
        ];
    change_attribution_temporal_dag_given_frequentist_coverage:
        attribution(0x4D45_001A, AttrKind::Change, Class::TemporalDag, Mode::Frequentist) [
            (Explicit, None), (Accepted, None),
        ];
    change_attribution_temporal_dag_graph_posterior_bayesian_coverage:
        attribution(0x4D45_001B, AttrKind::Change, Class::TemporalDag, Mode::Bayesian) [
            (Posterior, None),
        ];
    change_attribution_temporal_cpdag_given_bayesian_coverage:
        attribution(0x4D45_001C, AttrKind::Change, Class::TemporalCpdag, Mode::Bayesian) [
            (Explicit, None), (Accepted, None),
        ];
    change_attribution_temporal_cpdag_graph_posterior_bayesian_coverage:
        attribution(0x4D45_001D, AttrKind::Change, Class::TemporalCpdag, Mode::Bayesian) [
            (Posterior, None),
        ];
    change_attribution_temporal_pag_given_bayesian_coverage:
        attribution(0x4D45_001E, AttrKind::Change, Class::TemporalPag, Mode::Bayesian) [
            (Explicit, None), (Accepted, None),
        ];
    change_attribution_temporal_pag_graph_posterior_bayesian_coverage:
        attribution(0x4D45_001F, AttrKind::Change, Class::TemporalPag, Mode::Bayesian) [
            (Posterior, None),
        ];
    change_attribution_temporal_admg_given_bayesian_coverage:
        attribution(0x4D45_0020, AttrKind::Change, Class::TemporalAdmg, Mode::Bayesian) [
            (Explicit, None), (Accepted, None),
        ];
    change_attribution_temporal_admg_graph_posterior_bayesian_coverage:
        attribution(0x4D45_0021, AttrKind::Change, Class::TemporalAdmg, Mode::Bayesian) [
            (Posterior, None),
        ];
    counterfactual_dag_given_bayesian_coverage:
        counterfactual(0x4D45_0022, Class::Dag, Mode::Bayesian) [
            (Accepted, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full),
            (Accepted, Full),
        ];
    counterfactual_dag_graph_posterior_bayesian_coverage:
        counterfactual(0x4D45_0023, Class::Dag, Mode::Bayesian) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    counterfactual_dag_graph_posterior_frequentist_coverage:
        counterfactual(0x4D45_0024, Class::Dag, Mode::Frequentist) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    counterfactual_cpdag_given_bayesian_coverage:
        counterfactual(0x4D45_0025, Class::Cpdag, Mode::Bayesian) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    counterfactual_cpdag_graph_posterior_bayesian_coverage:
        counterfactual(0x4D45_0026, Class::Cpdag, Mode::Bayesian) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    counterfactual_cpdag_graph_posterior_frequentist_coverage:
        counterfactual(0x4D45_0027, Class::Cpdag, Mode::Frequentist) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    transport_dag_given_bayesian_coverage:
        transport(0x4D45_0028, Class::Dag, Mode::Bayesian) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    transport_dag_given_frequentist_coverage:
        transport(0x4D45_0029, Class::Dag, Mode::Frequentist) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    transport_dag_graph_posterior_bayesian_coverage:
        transport(0x4D45_002A, Class::Dag, Mode::Bayesian) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    transport_dag_graph_posterior_frequentist_coverage:
        transport(0x4D45_002B, Class::Dag, Mode::Frequentist) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    transport_cpdag_given_bayesian_coverage:
        transport(0x4D45_002C, Class::Cpdag, Mode::Bayesian) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    transport_cpdag_given_frequentist_coverage:
        transport(0x4D45_002D, Class::Cpdag, Mode::Frequentist) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    transport_cpdag_graph_posterior_bayesian_coverage:
        transport(0x4D45_002E, Class::Cpdag, Mode::Bayesian) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    transport_cpdag_graph_posterior_frequentist_coverage:
        transport(0x4D45_002F, Class::Cpdag, Mode::Frequentist) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    transport_pag_given_bayesian_coverage:
        transport(0x4D45_0030, Class::Pag, Mode::Bayesian) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    transport_pag_given_frequentist_coverage:
        transport(0x4D45_0031, Class::Pag, Mode::Frequentist) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    transport_pag_graph_posterior_bayesian_coverage:
        transport(0x4D45_0032, Class::Pag, Mode::Bayesian) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    transport_pag_graph_posterior_frequentist_coverage:
        transport(0x4D45_0033, Class::Pag, Mode::Frequentist) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    transport_admg_given_bayesian_coverage:
        transport(0x4D45_0034, Class::Admg, Mode::Bayesian) [
            (Accepted, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full),
            (Accepted, Full),
        ];
    transport_admg_given_frequentist_coverage:
        transport(0x4D45_0035, Class::Admg, Mode::Frequentist) [
            (Accepted, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full),
            (Accepted, Full),
        ];
    transport_admg_graph_posterior_bayesian_coverage:
        transport(0x4D45_0036, Class::Admg, Mode::Bayesian) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    transport_admg_graph_posterior_frequentist_coverage:
        transport(0x4D45_0037, Class::Admg, Mode::Frequentist) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    transport_temporal_admg_given_bayesian_coverage:
        transport(0x4D45_0038, Class::TemporalAdmg, Mode::Bayesian) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    transport_temporal_admg_given_frequentist_coverage:
        transport(0x4D45_0039, Class::TemporalAdmg, Mode::Frequentist) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    transport_temporal_admg_graph_posterior_bayesian_coverage:
        transport(0x4D45_003A, Class::TemporalAdmg, Mode::Bayesian) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    transport_temporal_admg_graph_posterior_frequentist_coverage:
        transport(0x4D45_003B, Class::TemporalAdmg, Mode::Frequentist) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_dag_given_bayesian_gaussian_coverage:
        interference(0x4D45_003C, Class::Dag, Design::BayesianGaussian) [
            (Accepted, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full),
            (Accepted, Full),
        ];
    interference_dag_given_cluster_bayesian_bootstrap_coverage:
        interference(0x4D45_003D, Class::Dag, Design::ClusterBootstrap) [
            (Accepted, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full),
            (Accepted, Full),
        ];
    interference_dag_given_ht_hajek_coverage:
        interference(0x4D45_003E, Class::Dag, Design::HtHajek) [
            (Accepted, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full),
            (Accepted, Full),
        ];
    interference_dag_given_cluster_neyman_coverage:
        interference(0x4D45_003F, Class::Dag, Design::ClusterNeyman) [
            (Accepted, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full),
            (Accepted, Full),
        ];
    interference_dag_given_observational_ipw_coverage:
        interference(0x4D45_0040, Class::Dag, Design::Observational) [
            (Accepted, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full),
            (Accepted, Full),
        ];
    interference_dag_graph_posterior_bayesian_gaussian_coverage:
        interference(0x4D45_0041, Class::Dag, Design::BayesianGaussian) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_dag_graph_posterior_cluster_bayesian_bootstrap_coverage:
        interference(0x4D45_0042, Class::Dag, Design::ClusterBootstrap) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_dag_graph_posterior_ht_hajek_coverage:
        interference(0x4D45_0043, Class::Dag, Design::HtHajek) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_dag_graph_posterior_cluster_neyman_coverage:
        interference(0x4D45_0044, Class::Dag, Design::ClusterNeyman) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_dag_graph_posterior_observational_ipw_coverage:
        interference(0x4D45_0045, Class::Dag, Design::Observational) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_cpdag_given_bayesian_gaussian_coverage:
        interference(0x4D45_0046, Class::Cpdag, Design::BayesianGaussian) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    interference_cpdag_given_cluster_bayesian_bootstrap_coverage:
        interference(0x4D45_0047, Class::Cpdag, Design::ClusterBootstrap) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    interference_cpdag_given_ht_hajek_coverage:
        interference(0x4D45_0048, Class::Cpdag, Design::HtHajek) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    interference_cpdag_given_cluster_neyman_coverage:
        interference(0x4D45_0049, Class::Cpdag, Design::ClusterNeyman) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    interference_cpdag_given_observational_ipw_coverage:
        interference(0x4D45_004A, Class::Cpdag, Design::Observational) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    interference_cpdag_graph_posterior_bayesian_gaussian_coverage:
        interference(0x4D45_004B, Class::Cpdag, Design::BayesianGaussian) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_cpdag_graph_posterior_cluster_bayesian_bootstrap_coverage:
        interference(0x4D45_004C, Class::Cpdag, Design::ClusterBootstrap) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_cpdag_graph_posterior_ht_hajek_coverage:
        interference(0x4D45_004D, Class::Cpdag, Design::HtHajek) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_cpdag_graph_posterior_cluster_neyman_coverage:
        interference(0x4D45_004E, Class::Cpdag, Design::ClusterNeyman) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_cpdag_graph_posterior_observational_ipw_coverage:
        interference(0x4D45_004F, Class::Cpdag, Design::Observational) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_pag_given_bayesian_gaussian_coverage:
        interference(0x4D45_0050, Class::Pag, Design::BayesianGaussian) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    interference_pag_given_cluster_bayesian_bootstrap_coverage:
        interference(0x4D45_0051, Class::Pag, Design::ClusterBootstrap) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    interference_pag_given_ht_hajek_coverage:
        interference(0x4D45_0052, Class::Pag, Design::HtHajek) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    interference_pag_given_cluster_neyman_coverage:
        interference(0x4D45_0053, Class::Pag, Design::ClusterNeyman) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    interference_pag_given_observational_ipw_coverage:
        interference(0x4D45_0054, Class::Pag, Design::Observational) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    interference_pag_graph_posterior_bayesian_gaussian_coverage:
        interference(0x4D45_0055, Class::Pag, Design::BayesianGaussian) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_pag_graph_posterior_cluster_bayesian_bootstrap_coverage:
        interference(0x4D45_0056, Class::Pag, Design::ClusterBootstrap) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_pag_graph_posterior_ht_hajek_coverage:
        interference(0x4D45_0057, Class::Pag, Design::HtHajek) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_pag_graph_posterior_cluster_neyman_coverage:
        interference(0x4D45_0058, Class::Pag, Design::ClusterNeyman) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_pag_graph_posterior_observational_ipw_coverage:
        interference(0x4D45_0059, Class::Pag, Design::Observational) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_admg_given_bayesian_gaussian_coverage:
        interference(0x4D45_005A, Class::Admg, Design::BayesianGaussian) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    interference_admg_given_cluster_bayesian_bootstrap_coverage:
        interference(0x4D45_005B, Class::Admg, Design::ClusterBootstrap) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    interference_admg_given_ht_hajek_coverage:
        interference(0x4D45_005C, Class::Admg, Design::HtHajek) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    interference_admg_given_cluster_neyman_coverage:
        interference(0x4D45_005D, Class::Admg, Design::ClusterNeyman) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    interference_admg_given_observational_ipw_coverage:
        interference(0x4D45_005E, Class::Admg, Design::Observational) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    interference_admg_graph_posterior_bayesian_gaussian_coverage:
        interference(0x4D45_005F, Class::Admg, Design::BayesianGaussian) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_admg_graph_posterior_cluster_bayesian_bootstrap_coverage:
        interference(0x4D45_0060, Class::Admg, Design::ClusterBootstrap) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_admg_graph_posterior_ht_hajek_coverage:
        interference(0x4D45_0061, Class::Admg, Design::HtHajek) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_admg_graph_posterior_cluster_neyman_coverage:
        interference(0x4D45_0062, Class::Admg, Design::ClusterNeyman) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_admg_graph_posterior_observational_ipw_coverage:
        interference(0x4D45_0063, Class::Admg, Design::Observational) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_temporal_admg_given_bayesian_gaussian_coverage:
        interference(0x4D45_0064, Class::TemporalAdmg, Design::BayesianGaussian) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    interference_temporal_admg_given_cluster_bayesian_bootstrap_coverage:
        interference(0x4D45_0065, Class::TemporalAdmg, Design::ClusterBootstrap) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    interference_temporal_admg_given_ht_hajek_coverage:
        interference(0x4D45_0066, Class::TemporalAdmg, Design::HtHajek) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    interference_temporal_admg_given_cluster_neyman_coverage:
        interference(0x4D45_0067, Class::TemporalAdmg, Design::ClusterNeyman) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    interference_temporal_admg_given_observational_ipw_coverage:
        interference(0x4D45_0068, Class::TemporalAdmg, Design::Observational) [
            (Explicit, None), (Accepted, None), (Explicit, Cheap), (Accepted, Cheap),
            (Explicit, Full), (Accepted, Full),
        ];
    interference_temporal_admg_graph_posterior_bayesian_gaussian_coverage:
        interference(0x4D45_0069, Class::TemporalAdmg, Design::BayesianGaussian) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_temporal_admg_graph_posterior_cluster_bayesian_bootstrap_coverage:
        interference(0x4D45_006A, Class::TemporalAdmg, Design::ClusterBootstrap) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_temporal_admg_graph_posterior_ht_hajek_coverage:
        interference(0x4D45_006B, Class::TemporalAdmg, Design::HtHajek) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_temporal_admg_graph_posterior_cluster_neyman_coverage:
        interference(0x4D45_006C, Class::TemporalAdmg, Design::ClusterNeyman) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
    interference_temporal_admg_graph_posterior_observational_ipw_coverage:
        interference(0x4D45_006D, Class::TemporalAdmg, Design::Observational) [
            (Posterior, None), (Posterior, Cheap), (Posterior, Full),
        ];
}
