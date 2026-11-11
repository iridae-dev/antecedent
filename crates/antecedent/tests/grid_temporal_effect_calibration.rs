//! Repeated-sampling coverage of the intervals the temporal linear adjustment, the Bayesian
//! temporal g-computation, the sequential g-computation of a multi-step sustained contrast, the
//! temporal mediation estimators, and the lagged-instrument, GLM link-contrast and
//! white-innovation routes report on temporal effect queries.
//!
//! Cells: `AverageEffect` (the lag-0 pulse), `PulseEffect`, `SustainedEffect`,
//! `TemporalConditionalPulse`, `TemporalConditionalSustained` and the cross-world effects
//! (`NestedCounterfactualEffect`, `PathSpecificEffect`, `Counterfactual`,
//! `InterventionalDistribution`) on a `TemporalDag`, `TemporalCpdag` and `TemporalPag`, and
//! `TemporalMediationEffect` on a `TemporalDag` and `TemporalCpdag`; each given (explicit and
//! accepted) and as a one-atom graph posterior, Frequentist and Bayesian. The Frequentist
//! interval of the adjustment routes is the circular-block SE interval (`estimate ± z·se` at 0.95 from 199 replicates;
//! the 0.90 interval is the same construction at the gate level); the Bayesian interval is the
//! equal-tailed posterior quantile interval (conjugate, 1000 draws, prior scale 8), at 0.90
//! re-summarized from the same draws.
//!
//! One test measures one (query, graph class, inference, estimator, structure family); the validation modes
//! and the accepted twin of an explicit graph are sibling executions on the same data, required
//! to report the base interval bit for bit.
//!
//! Every series law is linear-Gaussian with iid innovations (the conditional law adds a
//! Bernoulli modifier), so the truth of each reported functional is closed form:
//!
//! - lag-0 law (columns `x, y, z`): `x(t) = .6 z(t-1) + e`, `y(t) = 2 x(t) + .8 z(t-1) + .5 e`;
//!   the `TemporalPag` reads the same law without `z(t-1) -> y(t)`, so `z(t-1)` is the witness
//!   that makes `x -> y` visible. `E[y | do(x = 1)] - E[y | do(x = 0)] = 2`.
//! - pulse law (columns `t, y, z, w`): `z = e`, `w = .4 z + e`, `t = .5 z + .4 w + e`,
//!   `y(t) = 1 + 2 t(t-1) + .8 z(t-1) + e`. Adjusting for `z(t-1)` identifies the lag-1 slope 2;
//!   `w` enters only the treatment and is its visibility witness under a PAG.
//! - sustained law: the pulse law with `y(t) = 1 + 2 t(t-1) + t(t-2) + .8 z(t-1) + e`; holding
//!   `t = 1` at lags 1 and 2 against `t = 0` moves `y` by `2 + 1 = 3`.
//! - mediation law (columns `t, m, y`): `t = e`, `m(t) = .8 t(t-1) + .4 e`,
//!   `y(t) = .25 t(t-1) + .55 m(t) + .4 e`; the mediated contrast is the path product `.44`.
//! - conditional law (columns `x, y, w, r`): `x = .8 r + e`, `w ~ Bernoulli(.4)`,
//!   `y(t) = 1.5 x(t-1) + .2 w(t) + .8 x(t-1) w(t) + .5 e`; the pulse slope at the modifier is
//!   `1.5 + .8 w`, so the effect marginal over the modifier is `1.5 + .8 · .4 = 1.82` (the
//!   one-step sustained window is the same contrast). `r` is the treatment's PAG witness.
//! - cross-world law (columns `x, m, y`, and a witness `r` with `x = .6 r + e` on the
//!   `TemporalPag`): `m(t) = .5 x(t-1) + e`, `y(t) = 1.5 x(t-1) + 4 m(t) + e`. The nested
//!   direct effect is 1.5, the path-specific effect through `m` is `.5 · 4 = 2`, the unit
//!   counterfactual contrast of `x = 1` against `x = 0` is 3.5, and the interventional mean of
//!   `y` under `do(x = 1)` is `1.5 + 4 · .5 = 3.5` (every innovation has mean 0).
//! - one-step sustained window: the pulse law under the schedule that holds `t = 1` at lag 1
//!   only, answered by the temporal linear adjustment and the Bayesian temporal g-computation;
//!   the contrast is the lag-1 slope 2.
//! - two-step conditional law: the conditional law with `y(t)` gaining `1.0 x(t-2)`; holding
//!   `x = 1` at lags 1 and 2 against `x = 0`, marginal over the modifier, moves `y` by
//!   `1.5 + .8 · .4 + 1 = 2.82`, answered by the sequential g-computation.
//!
//! Three routes are selected explicitly on a `TemporalDag` and report the scalar contrast with
//! an analytic interval (Frequentist) or posterior draws (Bayesian):
//!
//! - lagged instrument (columns `x, y, z, u`, every input iid normal): `x(s) = .9 z(s-1) +
//!   .8 u(s) + .5 e`, `y(t) = 1 + 2 x(t-1) + u(t-1) + .5 e`. `u(t-1)` confounds treatment and
//!   outcome; `z(t-2)` is the instrument the unfolded window certifies. The Wald contrast is 2;
//!   the homoskedastic Wald standard error is valid because the aligned rows are independent.
//! - GLM link contrast (a panel of independent subjects, columns `x, y, c`, one decision row
//!   each): `c ~ N(0, 1)`, `x ~ Bernoulli(expit(.6 c))`, `y ~ Bernoulli(expit(-.5 + x + .8 c))`.
//!   The route reports the probability-scale contrast averaged over the observed `c` with its
//!   Fisher delta-method (or Laplace) uncertainty, which holds `c` fixed, so the truth is that
//!   average over the replicate's own subjects: `mean_i[expit(.5 + .8 c_i) - expit(-.5 + .8 c_i)]`.
//! - white-innovation linear posterior: the pulse law (its outcome innovation is iid) with the
//!   declared white-innovation premise; the Ljung–Box gate runs at level `1e-6` so a replicate is
//!   refused only on overwhelming residual autocorrelation.
//!
//! The class graphs are fully oriented (one completion); a Bayesian class effect that mixes
//! completion posteriors is given the one-completion class prior `[1]`. Every graph posterior
//! carries all mass on one atom.
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;
#[path = "common/grid_cell.rs"]
mod grid_cell;

use std::sync::Arc;

use antecedent::estimate::InnovationWhitenessGate;
use antecedent::{
    AcceptedGraph, BayesianConfig, CausalPosterior, ClassPrior, EstimatorId, IdentifierId,
    InferenceMode, RefuteSuite, Study, StudyResult,
};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, CounterfactualQuery, ExecutionContext, Intervention,
    InterventionalDistributionQuery, Lag, MediationContrast, MediationQuery,
    NestedCounterfactualQuery, PathSpecificEffectQuery, TemporalEffectQuery, TemporalPolicy, Value,
    VariableId,
};
use antecedent_data::{PanelData, PanelUnit, TimeSeriesData};
use antecedent_discovery::adjacency_mask::set_edge;
use antecedent_discovery::{
    AdjacencyMask, GraphPosterior, GraphPosteriorAtomKind, SamplingSummary,
};
use antecedent_graph::{DenseNodeId, TemporalCpdag, TemporalDag, TemporalPag, ensure_lagged};
use antecedent_prob::{BayesLikelihood, InferenceDiagnostics};
use common::calibration::{gaussian, grid_n, stream_seed, uniform};
use common::reported::{posterior_pair, scalar_normal_pair};
use grid_cell::{Cell, IntervalPair, Target, measure, measure_sample_truths};

/// Rows dropped from the head of every simulated series, so the lagged terms are populated.
const BURN: usize = 10;
/// Circular-block replicates of a Frequentist execution.
const BOOT: u32 = 199;
/// Base series length of the lagged-instrument law.
const INSTRUMENT_BASE_N: usize = 200;
/// Base number of independent subjects of the GLM link panel.
const LINK_BASE_N: usize = 400;
/// Level of the Ljung–Box gate of the white-innovation premise.
const WHITENESS_LEVEL: f64 = 1e-6;
/// GLM link law: intercept, treatment and covariate coefficients on the logit scale.
const LINK_LOGIT: (f64, f64, f64) = (-0.5, 1.0, 0.8);

/// How the graph reaches the study.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Structure {
    Explicit,
    Accepted,
    Posterior,
}

/// The temporal graph family of a cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Class {
    Dag,
    Cpdag,
    Pag,
}

/// The inference family of a cell.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Mode {
    Frequentist,
    Bayesian,
}

/// The estimator route of a cell: the default adjustment the plan selects, or one of the
/// routes a `TemporalDag` study selects explicitly, each on a law it is valid for.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Route {
    Adjustment,
    Instrument,
    Link,
    White,
}

/// The query of a cell, with its law, graphs and truth.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Family {
    Average,
    Pulse,
    Sustained,
    /// The one-step sustained window over the pulse law.
    SustainedStep,
    Mediation,
    ConditionalPulse,
    ConditionalSustained,
    /// The two-step sustained window with an effect modifier over the two-step conditional law.
    ConditionalMultiStep,
    Nested,
    PathSpecific,
    Counterfactual,
    Distribution,
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

/// A given graph without the unvalidated accepted execution, which is measured on its own.
const GIVEN_WITHOUT_ACCEPTED_UNVALIDATED: &[Variant] = &[
    (Structure::Explicit, RefuteSuite::None),
    (Structure::Explicit, RefuteSuite::Cheap),
    (Structure::Explicit, RefuteSuite::Full),
    (Structure::Accepted, RefuteSuite::Cheap),
    (Structure::Accepted, RefuteSuite::Full),
];

/// A given graph without the unvalidated explicit execution, which is measured on its own.
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

/// A directed edge of a temporal template: `((variable, lag), (variable, lag))`.
type Edge = ((u32, u32), (u32, u32));

/// A lag-`lag` edge `from -> to` of a graph-posterior atom: `(lag, from, to)`.
type LagEdge = (usize, usize, usize);

fn var(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

/// `rows` iid standard-normal draws of stream `stream` of the replicate keyed by `seed`.
fn noise(rows: usize, seed: u64, stream: u64) -> Vec<f64> {
    let mut draw = gaussian(stream_seed(seed, stream));
    (0..rows).map(|_| draw()).collect()
}

/// `values` `k` steps before row `s` (zero before the first row; those rows are burned).
fn at(values: &[f64], s: usize, k: usize) -> f64 {
    s.checked_sub(k).map_or(0.0, |i| values[i])
}

/// The columns without their first `BURN` rows.
fn series(columns: &[(&str, &[f64])]) -> TimeSeriesData {
    let trimmed: Vec<(&str, &[f64])> =
        columns.iter().map(|(name, values)| (*name, &values[BURN..])).collect();
    TimeSeriesData::from_f64_columns(trimmed, 1).unwrap()
}

/// Columns `x, y, z`: `x(t) = .6 z(t-1) + e`, `y(t) = 2 x(t) + gamma z(t-1) + .5 e`.
fn average_series(rows: usize, seed: u64, gamma: f64) -> TimeSeriesData {
    let z = noise(rows, seed, 1);
    let ex = noise(rows, seed, 2);
    let ey = noise(rows, seed, 3);
    let x: Vec<f64> = (0..rows).map(|s| 0.6 * at(&z, s, 1) + ex[s]).collect();
    let y: Vec<f64> = (0..rows).map(|s| 2.0 * x[s] + gamma * at(&z, s, 1) + 0.5 * ey[s]).collect();
    series(&[("x", &x), ("y", &y), ("z", &z)])
}

/// Columns `t, y, z, w`: `z = e`, `w = .4 z + e`, `t = .5 z + .4 w + e`,
/// `y(t) = 1 + 2 t(t-1) + lag2 t(t-2) + .8 z(t-1) + e`.
fn pulse_series(rows: usize, seed: u64, lag2: f64) -> TimeSeriesData {
    let z = noise(rows, seed, 1);
    let ew = noise(rows, seed, 2);
    let et = noise(rows, seed, 3);
    let ey = noise(rows, seed, 4);
    let w: Vec<f64> = (0..rows).map(|s| 0.4 * z[s] + ew[s]).collect();
    let t: Vec<f64> = (0..rows).map(|s| 0.5 * z[s] + 0.4 * w[s] + et[s]).collect();
    let y: Vec<f64> = (0..rows)
        .map(|s| 1.0 + 2.0 * at(&t, s, 1) + lag2 * at(&t, s, 2) + 0.8 * at(&z, s, 1) + ey[s])
        .collect();
    series(&[("t", &t), ("y", &y), ("z", &z), ("w", &w)])
}

/// Columns `t, m, y`: `t = e`, `m(t) = .8 t(t-1) + .4 e`, `y(t) = .25 t(t-1) + .55 m(t) + .4 e`.
fn mediation_series(rows: usize, seed: u64) -> TimeSeriesData {
    let t = noise(rows, seed, 1);
    let em = noise(rows, seed, 2);
    let ey = noise(rows, seed, 3);
    let m: Vec<f64> = (0..rows).map(|s| 0.8 * at(&t, s, 1) + 0.4 * em[s]).collect();
    let y: Vec<f64> = (0..rows).map(|s| 0.25 * at(&t, s, 1) + 0.55 * m[s] + 0.4 * ey[s]).collect();
    series(&[("t", &t), ("m", &m), ("y", &y)])
}

/// Columns `x, y, w, r`: `x = .8 r + e`, `w ~ Bernoulli(.4)`,
/// `y(t) = 1.5 x(t-1) + .2 w(t) + .8 x(t-1) w(t) + lag2 x(t-2) + .5 e`.
fn conditional_series(rows: usize, seed: u64, lag2: f64) -> TimeSeriesData {
    let r = noise(rows, seed, 1);
    let ex = noise(rows, seed, 2);
    let ey = noise(rows, seed, 3);
    let mut coin = uniform(seed, 4);
    let w: Vec<f64> = (0..rows).map(|_| f64::from(u8::from(coin() < 0.4))).collect();
    let x: Vec<f64> = (0..rows).map(|s| 0.8 * r[s] + ex[s]).collect();
    let y: Vec<f64> = (0..rows)
        .map(|s| {
            let x1 = at(&x, s, 1);
            1.5 * x1 + 0.2 * w[s] + 0.8 * x1 * w[s] + 0.5 * ey[s] + lag2 * at(&x, s, 2)
        })
        .collect();
    series(&[("x", &x), ("y", &y), ("w", &w), ("r", &r)])
}

/// Columns `x, y, z, u`, every input iid normal: `x(s) = .9 z(s-1) + .8 u(s) + .5 e`,
/// `y(t) = 1 + 2 x(t-1) + u(t-1) + .5 e`.
fn instrument_series(rows: usize, seed: u64) -> TimeSeriesData {
    let z = noise(rows, seed, 1);
    let u = noise(rows, seed, 2);
    let ex = noise(rows, seed, 3);
    let ey = noise(rows, seed, 4);
    let x: Vec<f64> = (0..rows).map(|s| 0.9 * at(&z, s, 1) + 0.8 * u[s] + 0.5 * ex[s]).collect();
    let y: Vec<f64> =
        (0..rows).map(|s| 1.0 + 2.0 * at(&x, s, 1) + at(&u, s, 1) + 0.5 * ey[s]).collect();
    series(&[("x", &x), ("y", &y), ("z", &z), ("u", &u)])
}

fn expit(eta: f64) -> f64 {
    1.0 / (1.0 + (-eta).exp())
}

/// `units` independent subjects with columns `x, y, c` over three rows: `c` and `x` on the first
/// row, `y` on the second (the decision row), the third empty. `c ~ N(0, 1)`,
/// `x ~ Bernoulli(expit(.6 c))`, `y ~ Bernoulli(expit(-.5 + x + .8 c))`. Returns the panel and the
/// probability-scale contrast averaged over its subjects' `c`.
fn link_panel(units: usize, seed: u64) -> (PanelData, f64) {
    let (intercept, slope, loading) = LINK_LOGIT;
    let c = noise(units, seed, 1);
    let mut coin = uniform(seed, 2);
    let mut members = Vec::with_capacity(units);
    let mut contrast = 0.0;
    for (index, &covariate) in c.iter().enumerate() {
        let treated = coin() < expit(0.6 * covariate);
        let active = expit(intercept + slope + loading * covariate);
        let control = expit(intercept + loading * covariate);
        contrast += active - control;
        let responded = coin() < if treated { active } else { control };
        let x = [f64::from(u8::from(treated)), 0.0, 0.0];
        let y = [0.0, f64::from(u8::from(responded)), 0.0];
        let c = [covariate, 0.0, 0.0];
        let series = TimeSeriesData::from_f64_columns(
            [("x", x.as_slice()), ("y", y.as_slice()), ("c", c.as_slice())],
            1,
        )
        .unwrap();
        members.push(PanelUnit { unit_id: u32::try_from(index).unwrap(), series });
    }
    let count = f64::from(u32::try_from(units).unwrap());
    (PanelData::try_new(Arc::from(members)).unwrap(), contrast / count)
}

/// Columns `x, m, y` (and, with `witness`, `r` with `x = .6 r + e`):
/// `m(t) = .5 x(t-1) + e`, `y(t) = 1.5 x(t-1) + 4 m(t) + e`.
fn cross_world_series(rows: usize, seed: u64, witness: bool) -> TimeSeriesData {
    let r = noise(rows, seed, 1);
    let ex = noise(rows, seed, 2);
    let em = noise(rows, seed, 3);
    let ey = noise(rows, seed, 4);
    let loading = if witness { 0.6 } else { 0.0 };
    let x: Vec<f64> = (0..rows).map(|s| loading * r[s] + ex[s]).collect();
    let m: Vec<f64> = (0..rows).map(|s| 0.5 * at(&x, s, 1) + em[s]).collect();
    let y: Vec<f64> = (0..rows).map(|s| 1.5 * at(&x, s, 1) + 4.0 * m[s] + ey[s]).collect();
    if witness {
        series(&[("x", &x), ("m", &m), ("y", &y), ("r", &r)])
    } else {
        series(&[("x", &x), ("m", &m), ("y", &y)])
    }
}

fn dag_of(edges: &[Edge]) -> TemporalDag {
    let mut graph = TemporalDag::empty();
    for &((from, from_lag), (to, to_lag)) in edges {
        let a = ensure_lagged(&mut graph, var(from), Lag::from_raw(from_lag)).unwrap();
        let b = ensure_lagged(&mut graph, var(to), Lag::from_raw(to_lag)).unwrap();
        graph.insert_directed(a, b).unwrap();
    }
    graph
}

fn pag_node(
    graph: &mut TemporalPag,
    ids: &mut Vec<((u32, u32), DenseNodeId)>,
    key: (u32, u32),
) -> DenseNodeId {
    if let Some(&(_, id)) = ids.iter().find(|(seen, _)| *seen == key) {
        return id;
    }
    let id = graph.add_lagged(var(key.0), Lag::from_raw(key.1)).unwrap();
    ids.push((key, id));
    id
}

/// The template as a PAG with no circles: every edge directed.
fn pag_of(edges: &[Edge]) -> TemporalPag {
    let mut graph = TemporalPag::empty();
    let mut ids = Vec::new();
    for &(from, to) in edges {
        let a = pag_node(&mut graph, &mut ids, from);
        let b = pag_node(&mut graph, &mut ids, to);
        graph.insert_directed(a, b).unwrap();
    }
    graph
}

/// All mass on one atom of `kind` over `n` variables: the contemporaneous edges
/// `(from, to)` and the lagged edges `(lag, from, to)`.
fn atom_posterior(
    kind: GraphPosteriorAtomKind,
    n: usize,
    contemporaneous: &[(usize, usize)],
    lagged: &[LagEdge],
) -> GraphPosterior {
    let adjacency = contemporaneous
        .iter()
        .fold(AdjacencyMask::EMPTY, |mask, &(from, to)| set_edge(mask, n, from, to, true));
    let max_lag = lagged.iter().map(|&(lag, _, _)| lag).max().unwrap_or(1);
    let lag_mask = lagged.iter().fold(0_u64, |mask, &(lag, from, to)| {
        let bit = (lag - 1) * n * n + from * n + to;
        mask | (1_u64 << bit)
    });
    GraphPosterior::from_atoms(
        n,
        vec![1.0],
        vec![adjacency],
        vec![0.0; n * n],
        vec![0.0; n * n],
        SamplingSummary {
            ess: 1.0,
            diagnostics: InferenceDiagnostics::analytic("temporal_effect_single_atom"),
            rejected_invalid: 0,
        },
    )
    .unwrap()
    .with_atom_kind(kind)
    .with_lagged_marginals(u32::try_from(max_lag).unwrap(), vec![0.0; max_lag * n * n])
    .unwrap()
    .with_lag_masks(vec![lag_mask])
    .unwrap()
}

/// `z(t-1) -> x(t)`, `z(t-1) -> y(t)`, `x(t) -> y(t)`.
const AVERAGE_EDGES: &[Edge] = &[((2, 1), (0, 0)), ((2, 1), (1, 0)), ((0, 0), (1, 0))];
/// `z(t-1) -> x(t) -> y(t)`, with `z(t-1)` not adjacent to `y(t)`.
const AVERAGE_PAG_EDGES: &[Edge] = &[((2, 1), (0, 0)), ((0, 0), (1, 0))];
/// `z -> t`, `w -> t` within the lag-1 slice, `t(t-1) -> y(t)`, `z(t-1) -> y(t)`.
const PULSE_EDGES: &[Edge] =
    &[((2, 1), (0, 1)), ((3, 1), (0, 1)), ((0, 1), (1, 0)), ((2, 1), (1, 0))];
/// The pulse template with the lag-2 slice and `t(t-2) -> y(t)`.
const SUSTAINED_EDGES: &[Edge] = &[
    ((2, 1), (0, 1)),
    ((3, 1), (0, 1)),
    ((2, 2), (0, 2)),
    ((3, 2), (0, 2)),
    ((0, 1), (1, 0)),
    ((0, 2), (1, 0)),
    ((2, 1), (1, 0)),
];
/// `t(t-1) -> m(t)`, `t(t-1) -> y(t)`, `m(t) -> y(t)`.
const MEDIATION_EDGES: &[Edge] = &[((0, 1), (1, 0)), ((0, 1), (2, 0)), ((1, 0), (2, 0))];
/// `x(t-1) -> y(t)`, `w(t) -> y(t)`.
const CONDITIONAL_EDGES: &[Edge] = &[((0, 1), (1, 0)), ((2, 0), (1, 0))];
/// The conditional template with the witness `r(t-1) -> x(t-1)`.
const CONDITIONAL_PAG_EDGES: &[Edge] = &[((3, 1), (0, 1)), ((0, 1), (1, 0)), ((2, 0), (1, 0))];
/// `x(t-1) -> m(t)`, `x(t-1) -> y(t)`, `m(t) -> y(t)`.
const CROSS_WORLD_EDGES: &[Edge] = &[((0, 1), (1, 0)), ((0, 1), (2, 0)), ((1, 0), (2, 0))];
/// The cross-world template with the witness `r(t-1) -> x(t-1)`.
const CROSS_WORLD_PAG_EDGES: &[Edge] =
    &[((3, 1), (0, 1)), ((0, 1), (1, 0)), ((0, 1), (2, 0)), ((1, 0), (2, 0))];
/// The conditional template with `x(t-2) -> y(t)`.
const CONDITIONAL_MULTI_STEP_EDGES: &[Edge] =
    &[((0, 1), (1, 0)), ((0, 2), (1, 0)), ((2, 0), (1, 0))];
/// `z(t-2) -> x(t-1) -> y(t)`, `u(t-1) -> x(t-1)`, `u(t-1) -> y(t)`.
const INSTRUMENT_EDGES: &[Edge] =
    &[((2, 2), (0, 1)), ((0, 1), (1, 0)), ((3, 1), (0, 1)), ((3, 1), (1, 0))];
/// `c(t-1) -> x(t-1)`, `c(t-1) -> y(t)`, `x(t-1) -> y(t)`.
const LINK_EDGES: &[Edge] = &[((2, 1), (0, 1)), ((2, 1), (1, 0)), ((0, 1), (1, 0))];

fn pulse_query() -> TemporalEffectQuery {
    TemporalEffectQuery::pulse(var(0), var(1), 1.0)
        .with_policy(TemporalPolicy::pulse(-1))
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1))
}

fn conditional_query(sustained: bool) -> TemporalEffectQuery {
    let base = TemporalEffectQuery::pulse(var(0), var(1), 1.0)
        .with_horizon_steps(1)
        .with_max_history_lag(Some(1))
        .with_effect_modifiers([var(2)]);
    if sustained {
        base.with_policy(TemporalPolicy::sustained(-1, -1))
    } else {
        base.with_policy(TemporalPolicy::pulse(-1))
    }
}

impl Family {
    const fn base_n(self) -> usize {
        match self {
            Self::ConditionalPulse | Self::ConditionalSustained | Self::ConditionalMultiStep => 240,
            _ => 160,
        }
    }

    /// A Bayesian effect on a given class graph publishes a posterior only as a class-prior
    /// mixture of its completion posteriors.
    const fn mixes_class_prior(self) -> bool {
        matches!(self, Self::Average | Self::Pulse | Self::Sustained | Self::SustainedStep)
    }

    fn dgp(self, class: Class) -> &'static str {
        match (self, class) {
            (Self::Average, Class::Pag) => "average_series_without_confounder_edge",
            (Self::Average, _) => "average_series",
            (Self::Pulse | Self::SustainedStep, _) => "pulse_series",
            (Self::Sustained, _) => "sustained_series",
            (Self::ConditionalMultiStep, _) => "conditional_series_with_lag_two",
            (Self::Mediation, _) => "mediation_series",
            (Self::ConditionalPulse | Self::ConditionalSustained, _) => "conditional_series",
            (_, Class::Pag) => "cross_world_series_with_witness",
            _ => "cross_world_series",
        }
    }

    /// The population value of the reported functional under the family's law.
    fn truth(self) -> f64 {
        match self {
            Self::Average | Self::Pulse | Self::SustainedStep | Self::PathSpecific => 2.0,
            Self::Sustained => 3.0,
            Self::Mediation => 0.8 * 0.55,
            Self::ConditionalPulse | Self::ConditionalSustained => 1.5 + 0.8 * 0.4,
            Self::ConditionalMultiStep => 1.5 + 0.8 * 0.4 + 1.0,
            Self::Nested => 1.5,
            Self::Counterfactual | Self::Distribution => 1.5 + 4.0 * 0.5,
        }
    }

    fn data(self, class: Class, seed: u64) -> TimeSeriesData {
        let rows = grid_n(self.base_n()) + BURN;
        match self {
            Self::Average => {
                average_series(rows, seed, if class == Class::Pag { 0.0 } else { 0.8 })
            }
            Self::Pulse | Self::SustainedStep => pulse_series(rows, seed, 0.0),
            Self::Sustained => pulse_series(rows, seed, 1.0),
            Self::Mediation => mediation_series(rows, seed),
            Self::ConditionalPulse | Self::ConditionalSustained => {
                conditional_series(rows, seed, 0.0)
            }
            Self::ConditionalMultiStep => conditional_series(rows, seed, 1.0),
            _ => cross_world_series(rows, seed, class == Class::Pag),
        }
    }

    fn edges(self, class: Class) -> &'static [Edge] {
        match (self, class) {
            (Self::Average, Class::Pag) => AVERAGE_PAG_EDGES,
            (Self::Average, _) => AVERAGE_EDGES,
            (Self::Pulse | Self::SustainedStep, _) => PULSE_EDGES,
            (Self::Sustained, _) => SUSTAINED_EDGES,
            (Self::ConditionalMultiStep, _) => CONDITIONAL_MULTI_STEP_EDGES,
            (Self::Mediation, _) => MEDIATION_EDGES,
            (Self::ConditionalPulse | Self::ConditionalSustained, Class::Pag) => {
                CONDITIONAL_PAG_EDGES
            }
            (Self::ConditionalPulse | Self::ConditionalSustained, _) => CONDITIONAL_EDGES,
            (_, Class::Pag) => CROSS_WORLD_PAG_EDGES,
            _ => CROSS_WORLD_EDGES,
        }
    }

    fn posterior(self, class: Class) -> GraphPosterior {
        let kind = match class {
            Class::Dag => GraphPosteriorAtomKind::Dag,
            Class::Cpdag => GraphPosteriorAtomKind::Cpdag,
            Class::Pag => GraphPosteriorAtomKind::Pag,
        };
        match self {
            Self::Average if class == Class::Pag => {
                atom_posterior(kind, 3, &[(0, 1)], &[(1, 2, 0)])
            }
            Self::Average => atom_posterior(kind, 3, &[(0, 1)], &[(1, 2, 0), (1, 2, 1)]),
            Self::Pulse | Self::SustainedStep => {
                atom_posterior(kind, 4, &[(2, 0), (3, 0)], &[(1, 0, 1), (1, 2, 1)])
            }
            Self::Sustained => {
                atom_posterior(kind, 4, &[(2, 0), (3, 0)], &[(1, 0, 1), (1, 2, 1), (2, 0, 1)])
            }
            Self::Mediation => atom_posterior(kind, 3, &[(1, 2)], &[(1, 0, 1), (1, 0, 2)]),
            Self::ConditionalPulse | Self::ConditionalSustained => {
                atom_posterior(kind, 4, &[(2, 1), (3, 0)], &[(1, 0, 1)])
            }
            Self::ConditionalMultiStep => {
                atom_posterior(kind, 4, &[(2, 1), (3, 0)], &[(1, 0, 1), (2, 0, 1)])
            }
            _ if class == Class::Pag => {
                atom_posterior(kind, 4, &[(1, 2), (3, 0)], &[(1, 0, 1), (1, 0, 2)])
            }
            _ => atom_posterior(kind, 3, &[(1, 2)], &[(1, 0, 1), (1, 0, 2)]),
        }
    }

    fn query(self) -> CausalQuery {
        let (x, m, y) = (var(0), var(1), var(2));
        match self {
            Self::Average => {
                CausalQuery::AverageEffect(AverageEffectQuery::with_levels(x, var(1), 0.0, 1.0))
            }
            Self::Pulse => CausalQuery::TemporalEffect(pulse_query()),
            Self::SustainedStep => CausalQuery::TemporalEffect(
                pulse_query().with_policy(TemporalPolicy::sustained(-1, -1)),
            ),
            Self::Sustained => CausalQuery::TemporalEffect(
                TemporalEffectQuery::sustained(var(0), var(1), -2, 1.0)
                    .with_policy(TemporalPolicy::sustained(-2, -1))
                    .with_horizon_steps(1)
                    .with_max_history_lag(Some(2)),
            ),
            Self::Mediation => CausalQuery::Mediation(
                MediationQuery::binary(var(0), var(2), [var(1)], MediationContrast::Mediated)
                    .with_horizons(vec![1])
                    .unwrap(),
            ),
            Self::ConditionalPulse => CausalQuery::TemporalEffect(conditional_query(false)),
            Self::ConditionalSustained => CausalQuery::TemporalEffect(conditional_query(true)),
            Self::ConditionalMultiStep => CausalQuery::TemporalEffect(
                TemporalEffectQuery::sustained(var(0), var(1), -2, 1.0)
                    .with_policy(TemporalPolicy::sustained(-2, -1))
                    .with_horizon_steps(1)
                    .with_max_history_lag(Some(2))
                    .with_effect_modifiers([var(2)]),
            ),
            Self::Nested => {
                CausalQuery::NestedCounterfactual(NestedCounterfactualQuery::new(x, m, y).unwrap())
            }
            Self::PathSpecific => CausalQuery::PathSpecific(
                PathSpecificEffectQuery::binary(x, y).with_path_nodes([m]),
            ),
            Self::Counterfactual => CausalQuery::Counterfactual(
                CounterfactualQuery::new(y, Arc::from([Intervention::set(x, Value::f64(1.0))]))
                    .with_control_level(0.0),
            ),
            Self::Distribution => CausalQuery::Distribution(InterventionalDistributionQuery::new(
                y,
                [Intervention::set(x, Value::f64(1.0))],
            )),
        }
    }
}

fn bayes_config() -> BayesianConfig {
    BayesianConfig::conjugate().n_draws(1000).prior_scale(8.0)
}

/// A cell's coordinate: the query family, graph class, inference and estimator route.
type Coordinate = (Family, Class, Mode, Route);

impl Route {
    /// The Bayesian configuration of the route: the Laplace Bernoulli-logit posterior for the
    /// link contrast, the declared white-innovation premise, the conjugate posterior otherwise.
    fn bayes(self) -> InferenceMode {
        InferenceMode::Bayesian(match self {
            Self::Adjustment | Self::Instrument => bayes_config(),
            Self::Link => BayesianConfig::laplace()
                .likelihood(BayesLikelihood::BernoulliLogit)
                .prior_scale(8.0)
                .n_draws(1000),
            Self::White => bayes_config()
                .white_innovation(InnovationWhitenessGate { max_lag: 6, level: WHITENESS_LEVEL }),
        })
    }

    /// The family's query; the instrument's window is not capped at one lag, since the
    /// certified instrument sits two steps before the outcome.
    fn query(self, family: Family) -> CausalQuery {
        match (self, family) {
            (Self::Instrument, Family::Pulse | Family::SustainedStep) => {
                let base = TemporalEffectQuery::pulse(var(0), var(1), 1.0).with_horizon_steps(1);
                CausalQuery::TemporalEffect(if family == Family::Pulse {
                    base.with_policy(TemporalPolicy::pulse(-1))
                } else {
                    base.with_policy(TemporalPolicy::sustained(-1, -1))
                })
            }
            _ => family.query(),
        }
    }

    /// The generator the record names.
    fn dgp(self, family: Family, class: Class) -> &'static str {
        match self {
            Self::Adjustment | Self::White => family.dgp(class),
            Self::Instrument => "instrument_series",
            Self::Link => "link_panel",
        }
    }

    /// The `IntervalMethod::as_str()` of the scored interval: the circular-block SE of the
    /// adjustment routes, the analytic SE of the Wald and link contrasts, the posterior quantile
    /// interval of every Bayesian route.
    const fn interval(self, mode: Mode) -> &'static str {
        match (mode, self) {
            (Mode::Bayesian, _) => "posterior_quantile",
            (Mode::Frequentist, Self::Adjustment) => "circular_block_se",
            (Mode::Frequentist, _) => "analytic_se",
        }
    }
}

/// The study input of one replicate of a coordinate.
fn study_input((family, class, _, route): Coordinate, seed: u64) -> antecedent::StudyBuilder {
    match route {
        Route::Adjustment | Route::White => Study::series(family.data(class, seed)),
        Route::Instrument => {
            Study::series(instrument_series(grid_n(INSTRUMENT_BASE_N) + BURN, seed))
        }
        Route::Link => Study::panel(link_panel(grid_n(LINK_BASE_N), seed).0),
    }
}

/// One execution of a coordinate on the replicate's data under one structure and validation mode.
fn execute(
    coordinate: Coordinate,
    (structure, suite): Variant,
    replicate: u64,
) -> (Study, StudyResult) {
    let (family, class, mode, route) = coordinate;
    let base = study_input(coordinate, replicate);
    let edges = match route {
        Route::Adjustment | Route::White => family.edges(class),
        Route::Instrument => INSTRUMENT_EDGES,
        Route::Link => LINK_EDGES,
    };
    let base = match (structure, class) {
        (Structure::Posterior, _) => base.graph_posterior(family.posterior(class)),
        (Structure::Explicit, Class::Dag) => base.graph(dag_of(edges)),
        (Structure::Accepted, Class::Dag) => base.graph(AcceptedGraph::from(dag_of(edges))),
        (Structure::Explicit, Class::Cpdag) => {
            base.graph(TemporalCpdag::from_temporal_dag(&dag_of(edges)))
        }
        (Structure::Accepted, Class::Cpdag) => {
            base.graph(AcceptedGraph::from(TemporalCpdag::from_temporal_dag(&dag_of(edges))))
        }
        (Structure::Explicit, Class::Pag) => base.graph(pag_of(edges)),
        (Structure::Accepted, Class::Pag) => base.graph(AcceptedGraph::from(pag_of(edges))),
    };
    let base = match mode {
        // The Wald and link contrasts publish their analytic interval and resample nothing.
        Mode::Frequentist => base
            .inference(InferenceMode::Frequentist)
            .bootstrap_replicates(if route == Route::Adjustment { BOOT } else { 0 }),
        Mode::Bayesian => {
            let base = base.inference(route.bayes()).bootstrap_replicates(0);
            if family.mixes_class_prior()
                && class != Class::Dag
                && structure != Structure::Posterior
            {
                base.class_prior(ClassPrior::from_ordered([1.0]).unwrap())
            } else {
                base
            }
        }
    };
    let bayesian = mode == Mode::Bayesian;
    let base = match route {
        Route::Instrument => {
            base.identifier(IdentifierId::TemporalIvUnfolded).estimator(if bayesian {
                EstimatorId::BayesianTemporalLaggedInstrument
            } else {
                EstimatorId::TemporalLaggedInstrument
            })
        }
        Route::Link => base.estimator(if bayesian {
            EstimatorId::BayesianTemporalGlmLinkContrast
        } else {
            EstimatorId::TemporalGlmLinkContrast
        }),
        Route::Adjustment if family == Family::ConditionalMultiStep => {
            base.estimator(EstimatorId::TemporalSequentialGcomp)
        }
        // The white-innovation premise of the Bayesian configuration selects its estimator.
        Route::Adjustment | Route::White => base,
    };
    let label = format!("{family:?} {class:?} {mode:?} {route:?} {structure:?} {suite:?}");
    let study = base
        .query(route.query(family))
        .refute(suite)
        .build()
        .unwrap_or_else(|error| panic!("{label}: {error}"));
    let result = study
        .run(&ExecutionContext::for_tests(replicate))
        .unwrap_or_else(|error| panic!("{label}: {error}"));
    (study, result)
}

/// `[reported, gate]` interval of the scalar the execution reports.
fn score(mode: Mode, result: &StudyResult) -> Vec<IntervalPair> {
    vec![match mode {
        Mode::Frequentist => scalar_normal_pair(result),
        Mode::Bayesian => {
            let column =
                result.posterior.as_ref().and_then(CausalPosterior::effect_column).unwrap_or(0);
            posterior_pair(result, column)
        }
    }]
}

/// Measure one coordinate of the default adjustment route.
fn measure_cell(
    test: &'static str,
    (family, class, mode): (Family, Class, Mode),
    estimator: &'static str,
    variants: &'static [Variant],
    seed: u64,
) {
    measure_route(test, (family, class, mode, Route::Adjustment), estimator, variants, seed);
}

/// Measure one coordinate: `variants[0]` is the base execution, the rest its siblings.
fn measure_route(
    test: &'static str,
    coordinate: Coordinate,
    estimator: &'static str,
    variants: &'static [Variant],
    seed: u64,
) {
    let (family, class, mode, route) = coordinate;
    let cell = Cell {
        test,
        dgp: route.dgp(family, class),
        interval: route.interval(mode),
        estimator,
        variants: variants.len(),
    };
    let run = |rep, variant: usize| execute(coordinate, variants[variant], stream_seed(seed, rep));
    match route {
        // The link contrast averages over the replicate's own covariates.
        Route::Link => measure_sample_truths(
            cell,
            &Target::scalar(0.0),
            |rep| vec![link_panel(grid_n(LINK_BASE_N), stream_seed(seed, rep)).1],
            run,
            |result| score(mode, result),
        ),
        Route::Instrument => {
            measure(cell, &Target::scalar(2.0), run, |result| score(mode, result));
        }
        Route::Adjustment | Route::White => {
            measure(cell, &Target::scalar(family.truth()), run, |result| score(mode, result));
        }
    }
}

/// `AverageEffect` on a `TemporalDag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn average_effect_temporal_dag_frequentist_given_coverage() {
    measure_cell(
        "average_effect_temporal_dag_frequentist_given_coverage",
        (Family::Average, Class::Dag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_000,
    );
}

/// `AverageEffect` on a `TemporalDag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn average_effect_temporal_dag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "average_effect_temporal_dag_frequentist_graph_posterior_coverage",
        (Family::Average, Class::Dag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_001,
    );
}

/// `AverageEffect` on a `TemporalDag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn average_effect_temporal_dag_bayesian_given_coverage() {
    measure_cell(
        "average_effect_temporal_dag_bayesian_given_coverage",
        (Family::Average, Class::Dag, Mode::Bayesian),
        "bayesian.temporal.gcomp",
        GIVEN,
        0x7EE_002,
    );
}

/// `AverageEffect` on a `TemporalDag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn average_effect_temporal_dag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "average_effect_temporal_dag_bayesian_graph_posterior_coverage",
        (Family::Average, Class::Dag, Mode::Bayesian),
        "bayesian.temporal.gcomp",
        POSTERIOR,
        0x7EE_003,
    );
}

/// `AverageEffect` on a `TemporalCpdag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn average_effect_temporal_cpdag_frequentist_given_coverage() {
    measure_cell(
        "average_effect_temporal_cpdag_frequentist_given_coverage",
        (Family::Average, Class::Cpdag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_004,
    );
}

/// `AverageEffect` on a `TemporalCpdag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn average_effect_temporal_cpdag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "average_effect_temporal_cpdag_frequentist_graph_posterior_coverage",
        (Family::Average, Class::Cpdag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_005,
    );
}

/// `AverageEffect` on a `TemporalCpdag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn average_effect_temporal_cpdag_bayesian_given_coverage() {
    measure_cell(
        "average_effect_temporal_cpdag_bayesian_given_coverage",
        (Family::Average, Class::Cpdag, Mode::Bayesian),
        "bayesian.temporal.gcomp",
        GIVEN,
        0x7EE_006,
    );
}

/// `AverageEffect` on a `TemporalCpdag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn average_effect_temporal_cpdag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "average_effect_temporal_cpdag_bayesian_graph_posterior_coverage",
        (Family::Average, Class::Cpdag, Mode::Bayesian),
        "bayesian.temporal.gcomp",
        POSTERIOR,
        0x7EE_007,
    );
}

/// `AverageEffect` on a `TemporalPag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn average_effect_temporal_pag_frequentist_given_coverage() {
    measure_cell(
        "average_effect_temporal_pag_frequentist_given_coverage",
        (Family::Average, Class::Pag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_008,
    );
}

/// `AverageEffect` on a `TemporalPag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn average_effect_temporal_pag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "average_effect_temporal_pag_frequentist_graph_posterior_coverage",
        (Family::Average, Class::Pag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_009,
    );
}

/// `AverageEffect` on a `TemporalPag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn average_effect_temporal_pag_bayesian_given_coverage() {
    measure_cell(
        "average_effect_temporal_pag_bayesian_given_coverage",
        (Family::Average, Class::Pag, Mode::Bayesian),
        "bayesian.temporal.gcomp",
        GIVEN,
        0x7EE_00A,
    );
}

/// `AverageEffect` on a `TemporalPag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn average_effect_temporal_pag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "average_effect_temporal_pag_bayesian_graph_posterior_coverage",
        (Family::Average, Class::Pag, Mode::Bayesian),
        "bayesian.temporal.gcomp",
        POSTERIOR,
        0x7EE_00B,
    );
}

/// `PulseEffect` on a `TemporalDag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn pulse_effect_temporal_dag_frequentist_given_coverage() {
    measure_cell(
        "pulse_effect_temporal_dag_frequentist_given_coverage",
        (Family::Pulse, Class::Dag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN_WITHOUT_EXPLICIT_UNVALIDATED,
        0x7EE_00C,
    );
}

/// `PulseEffect` on a `TemporalDag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn pulse_effect_temporal_dag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "pulse_effect_temporal_dag_frequentist_graph_posterior_coverage",
        (Family::Pulse, Class::Dag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR_VALIDATED,
        0x7EE_00D,
    );
}

/// `PulseEffect` on a `TemporalDag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn pulse_effect_temporal_dag_bayesian_given_coverage() {
    measure_cell(
        "pulse_effect_temporal_dag_bayesian_given_coverage",
        (Family::Pulse, Class::Dag, Mode::Bayesian),
        "bayesian.temporal.gcomp",
        GIVEN_VALIDATED,
        0x7EE_00E,
    );
}

/// `PulseEffect` on a `TemporalDag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn pulse_effect_temporal_dag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "pulse_effect_temporal_dag_bayesian_graph_posterior_coverage",
        (Family::Pulse, Class::Dag, Mode::Bayesian),
        "bayesian.temporal.gcomp",
        POSTERIOR_VALIDATED,
        0x7EE_00F,
    );
}

/// `PulseEffect` on a `TemporalCpdag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn pulse_effect_temporal_cpdag_frequentist_given_coverage() {
    measure_cell(
        "pulse_effect_temporal_cpdag_frequentist_given_coverage",
        (Family::Pulse, Class::Cpdag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN_WITHOUT_ACCEPTED_UNVALIDATED,
        0x7EE_010,
    );
}

/// `PulseEffect` on a `TemporalCpdag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn pulse_effect_temporal_cpdag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "pulse_effect_temporal_cpdag_frequentist_graph_posterior_coverage",
        (Family::Pulse, Class::Cpdag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR_VALIDATED,
        0x7EE_011,
    );
}

/// `PulseEffect` on a `TemporalCpdag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn pulse_effect_temporal_cpdag_bayesian_given_coverage() {
    measure_cell(
        "pulse_effect_temporal_cpdag_bayesian_given_coverage",
        (Family::Pulse, Class::Cpdag, Mode::Bayesian),
        "bayesian.temporal.gcomp",
        GIVEN_WITHOUT_ACCEPTED_UNVALIDATED,
        0x7EE_012,
    );
}

/// `PulseEffect` on a `TemporalCpdag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn pulse_effect_temporal_cpdag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "pulse_effect_temporal_cpdag_bayesian_graph_posterior_coverage",
        (Family::Pulse, Class::Cpdag, Mode::Bayesian),
        "bayesian.temporal.gcomp",
        POSTERIOR_VALIDATED,
        0x7EE_013,
    );
}

/// `PulseEffect` on a `TemporalPag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn pulse_effect_temporal_pag_frequentist_given_coverage() {
    measure_cell(
        "pulse_effect_temporal_pag_frequentist_given_coverage",
        (Family::Pulse, Class::Pag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN_WITHOUT_ACCEPTED_UNVALIDATED,
        0x7EE_014,
    );
}

/// `PulseEffect` on a `TemporalPag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn pulse_effect_temporal_pag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "pulse_effect_temporal_pag_frequentist_graph_posterior_coverage",
        (Family::Pulse, Class::Pag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR_VALIDATED,
        0x7EE_015,
    );
}

/// `PulseEffect` on a `TemporalPag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn pulse_effect_temporal_pag_bayesian_given_coverage() {
    measure_cell(
        "pulse_effect_temporal_pag_bayesian_given_coverage",
        (Family::Pulse, Class::Pag, Mode::Bayesian),
        "bayesian.temporal.gcomp",
        GIVEN_WITHOUT_ACCEPTED_UNVALIDATED,
        0x7EE_016,
    );
}

/// `PulseEffect` on a `TemporalPag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn pulse_effect_temporal_pag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "pulse_effect_temporal_pag_bayesian_graph_posterior_coverage",
        (Family::Pulse, Class::Pag, Mode::Bayesian),
        "bayesian.temporal.gcomp",
        POSTERIOR_VALIDATED,
        0x7EE_017,
    );
}

/// `SustainedEffect` on a `TemporalDag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_dag_frequentist_given_coverage() {
    measure_cell(
        "sustained_effect_temporal_dag_frequentist_given_coverage",
        (Family::Sustained, Class::Dag, Mode::Frequentist),
        "temporal.sequential.gcomp",
        GIVEN_WITHOUT_EXPLICIT_UNVALIDATED,
        0x7EE_018,
    );
}

/// `SustainedEffect` on a `TemporalDag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_dag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "sustained_effect_temporal_dag_frequentist_graph_posterior_coverage",
        (Family::Sustained, Class::Dag, Mode::Frequentist),
        "temporal.sequential.gcomp",
        POSTERIOR_VALIDATED,
        0x7EE_019,
    );
}

/// `SustainedEffect` on a `TemporalDag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_dag_bayesian_given_coverage() {
    measure_cell(
        "sustained_effect_temporal_dag_bayesian_given_coverage",
        (Family::Sustained, Class::Dag, Mode::Bayesian),
        "temporal.sequential.gcomp",
        GIVEN_WITHOUT_ACCEPTED_UNVALIDATED,
        0x7EE_01A,
    );
}

/// `SustainedEffect` on a `TemporalDag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_dag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "sustained_effect_temporal_dag_bayesian_graph_posterior_coverage",
        (Family::Sustained, Class::Dag, Mode::Bayesian),
        "temporal.sequential.gcomp",
        POSTERIOR_VALIDATED,
        0x7EE_01B,
    );
}

/// `SustainedEffect` on a `TemporalCpdag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_cpdag_frequentist_given_coverage() {
    measure_cell(
        "sustained_effect_temporal_cpdag_frequentist_given_coverage",
        (Family::Sustained, Class::Cpdag, Mode::Frequentist),
        "temporal.sequential.gcomp",
        GIVEN_WITHOUT_ACCEPTED_UNVALIDATED,
        0x7EE_01C,
    );
}

/// `SustainedEffect` on a `TemporalCpdag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_cpdag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "sustained_effect_temporal_cpdag_frequentist_graph_posterior_coverage",
        (Family::Sustained, Class::Cpdag, Mode::Frequentist),
        "temporal.sequential.gcomp",
        POSTERIOR_VALIDATED,
        0x7EE_01D,
    );
}

/// `SustainedEffect` on a `TemporalCpdag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_cpdag_bayesian_given_coverage() {
    measure_cell(
        "sustained_effect_temporal_cpdag_bayesian_given_coverage",
        (Family::Sustained, Class::Cpdag, Mode::Bayesian),
        "temporal.sequential.gcomp",
        GIVEN_WITHOUT_ACCEPTED_UNVALIDATED,
        0x7EE_01E,
    );
}

/// `SustainedEffect` on a `TemporalCpdag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_cpdag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "sustained_effect_temporal_cpdag_bayesian_graph_posterior_coverage",
        (Family::Sustained, Class::Cpdag, Mode::Bayesian),
        "temporal.sequential.gcomp",
        POSTERIOR_VALIDATED,
        0x7EE_01F,
    );
}

/// `SustainedEffect` on a `TemporalPag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_pag_frequentist_given_coverage() {
    measure_cell(
        "sustained_effect_temporal_pag_frequentist_given_coverage",
        (Family::Sustained, Class::Pag, Mode::Frequentist),
        "temporal.sequential.gcomp",
        GIVEN_WITHOUT_ACCEPTED_UNVALIDATED,
        0x7EE_020,
    );
}

/// `SustainedEffect` on a `TemporalPag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_pag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "sustained_effect_temporal_pag_frequentist_graph_posterior_coverage",
        (Family::Sustained, Class::Pag, Mode::Frequentist),
        "temporal.sequential.gcomp",
        POSTERIOR_VALIDATED,
        0x7EE_021,
    );
}

/// `SustainedEffect` on a `TemporalPag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_pag_bayesian_given_coverage() {
    measure_cell(
        "sustained_effect_temporal_pag_bayesian_given_coverage",
        (Family::Sustained, Class::Pag, Mode::Bayesian),
        "temporal.sequential.gcomp",
        GIVEN_WITHOUT_ACCEPTED_UNVALIDATED,
        0x7EE_022,
    );
}

/// `SustainedEffect` on a `TemporalPag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_pag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "sustained_effect_temporal_pag_bayesian_graph_posterior_coverage",
        (Family::Sustained, Class::Pag, Mode::Bayesian),
        "temporal.sequential.gcomp",
        POSTERIOR_VALIDATED,
        0x7EE_023,
    );
}

/// `TemporalMediationEffect` on a `TemporalDag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_mediation_effect_temporal_dag_frequentist_given_coverage() {
    measure_cell(
        "temporal_mediation_effect_temporal_dag_frequentist_given_coverage",
        (Family::Mediation, Class::Dag, Mode::Frequentist),
        "temporal.mediation",
        GIVEN_WITHOUT_EXPLICIT_UNVALIDATED,
        0x7EE_024,
    );
}

/// `TemporalMediationEffect` on a `TemporalDag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_mediation_effect_temporal_dag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "temporal_mediation_effect_temporal_dag_frequentist_graph_posterior_coverage",
        (Family::Mediation, Class::Dag, Mode::Frequentist),
        "temporal.mediation",
        POSTERIOR_VALIDATED,
        0x7EE_025,
    );
}

/// `TemporalMediationEffect` on a `TemporalDag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_mediation_effect_temporal_dag_bayesian_given_coverage() {
    measure_cell(
        "temporal_mediation_effect_temporal_dag_bayesian_given_coverage",
        (Family::Mediation, Class::Dag, Mode::Bayesian),
        "temporal.mediation.bayesian",
        GIVEN_WITHOUT_EXPLICIT_UNVALIDATED,
        0x7EE_026,
    );
}

/// `TemporalMediationEffect` on a `TemporalDag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_mediation_effect_temporal_dag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "temporal_mediation_effect_temporal_dag_bayesian_graph_posterior_coverage",
        (Family::Mediation, Class::Dag, Mode::Bayesian),
        "temporal.mediation.bayesian",
        POSTERIOR_VALIDATED,
        0x7EE_027,
    );
}

/// `TemporalMediationEffect` on a `TemporalCpdag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_mediation_effect_temporal_cpdag_frequentist_given_coverage() {
    measure_cell(
        "temporal_mediation_effect_temporal_cpdag_frequentist_given_coverage",
        (Family::Mediation, Class::Cpdag, Mode::Frequentist),
        "temporal.mediation",
        GIVEN_WITHOUT_EXPLICIT_UNVALIDATED,
        0x7EE_028,
    );
}

/// `TemporalMediationEffect` on a `TemporalCpdag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_mediation_effect_temporal_cpdag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "temporal_mediation_effect_temporal_cpdag_frequentist_graph_posterior_coverage",
        (Family::Mediation, Class::Cpdag, Mode::Frequentist),
        "temporal.mediation",
        POSTERIOR_VALIDATED,
        0x7EE_029,
    );
}

/// `TemporalMediationEffect` on a `TemporalCpdag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_mediation_effect_temporal_cpdag_bayesian_given_coverage() {
    measure_cell(
        "temporal_mediation_effect_temporal_cpdag_bayesian_given_coverage",
        (Family::Mediation, Class::Cpdag, Mode::Bayesian),
        "temporal.mediation.bayesian",
        GIVEN_VALIDATED,
        0x7EE_02A,
    );
}

/// `TemporalMediationEffect` on a `TemporalCpdag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_mediation_effect_temporal_cpdag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "temporal_mediation_effect_temporal_cpdag_bayesian_graph_posterior_coverage",
        (Family::Mediation, Class::Cpdag, Mode::Bayesian),
        "temporal.mediation.bayesian",
        POSTERIOR_VALIDATED,
        0x7EE_02B,
    );
}

/// `TemporalConditionalPulse` on a `TemporalDag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_pulse_temporal_dag_frequentist_given_coverage() {
    measure_cell(
        "temporal_conditional_pulse_temporal_dag_frequentist_given_coverage",
        (Family::ConditionalPulse, Class::Dag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_02C,
    );
}

/// `TemporalConditionalPulse` on a `TemporalDag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_pulse_temporal_dag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "temporal_conditional_pulse_temporal_dag_frequentist_graph_posterior_coverage",
        (Family::ConditionalPulse, Class::Dag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_02D,
    );
}

/// `TemporalConditionalPulse` on a `TemporalDag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_pulse_temporal_dag_bayesian_given_coverage() {
    measure_cell(
        "temporal_conditional_pulse_temporal_dag_bayesian_given_coverage",
        (Family::ConditionalPulse, Class::Dag, Mode::Bayesian),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_02E,
    );
}

/// `TemporalConditionalPulse` on a `TemporalDag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_pulse_temporal_dag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "temporal_conditional_pulse_temporal_dag_bayesian_graph_posterior_coverage",
        (Family::ConditionalPulse, Class::Dag, Mode::Bayesian),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_02F,
    );
}

/// `TemporalConditionalPulse` on a `TemporalCpdag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_pulse_temporal_cpdag_frequentist_given_coverage() {
    measure_cell(
        "temporal_conditional_pulse_temporal_cpdag_frequentist_given_coverage",
        (Family::ConditionalPulse, Class::Cpdag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_030,
    );
}

/// `TemporalConditionalPulse` on a `TemporalCpdag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_pulse_temporal_cpdag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "temporal_conditional_pulse_temporal_cpdag_frequentist_graph_posterior_coverage",
        (Family::ConditionalPulse, Class::Cpdag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_031,
    );
}

/// `TemporalConditionalPulse` on a `TemporalCpdag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_pulse_temporal_cpdag_bayesian_given_coverage() {
    measure_cell(
        "temporal_conditional_pulse_temporal_cpdag_bayesian_given_coverage",
        (Family::ConditionalPulse, Class::Cpdag, Mode::Bayesian),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_032,
    );
}

/// `TemporalConditionalPulse` on a `TemporalCpdag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_pulse_temporal_cpdag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "temporal_conditional_pulse_temporal_cpdag_bayesian_graph_posterior_coverage",
        (Family::ConditionalPulse, Class::Cpdag, Mode::Bayesian),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_033,
    );
}

/// `TemporalConditionalPulse` on a `TemporalPag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_pulse_temporal_pag_frequentist_given_coverage() {
    measure_cell(
        "temporal_conditional_pulse_temporal_pag_frequentist_given_coverage",
        (Family::ConditionalPulse, Class::Pag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_034,
    );
}

/// `TemporalConditionalPulse` on a `TemporalPag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_pulse_temporal_pag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "temporal_conditional_pulse_temporal_pag_frequentist_graph_posterior_coverage",
        (Family::ConditionalPulse, Class::Pag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_035,
    );
}

/// `TemporalConditionalPulse` on a `TemporalPag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_pulse_temporal_pag_bayesian_given_coverage() {
    measure_cell(
        "temporal_conditional_pulse_temporal_pag_bayesian_given_coverage",
        (Family::ConditionalPulse, Class::Pag, Mode::Bayesian),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_036,
    );
}

/// `TemporalConditionalPulse` on a `TemporalPag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_pulse_temporal_pag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "temporal_conditional_pulse_temporal_pag_bayesian_graph_posterior_coverage",
        (Family::ConditionalPulse, Class::Pag, Mode::Bayesian),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_037,
    );
}

/// `TemporalConditionalSustained` on a `TemporalDag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_sustained_temporal_dag_frequentist_given_coverage() {
    measure_cell(
        "temporal_conditional_sustained_temporal_dag_frequentist_given_coverage",
        (Family::ConditionalSustained, Class::Dag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_038,
    );
}

/// `TemporalConditionalSustained` on a `TemporalDag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_sustained_temporal_dag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "temporal_conditional_sustained_temporal_dag_frequentist_graph_posterior_coverage",
        (Family::ConditionalSustained, Class::Dag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_039,
    );
}

/// `TemporalConditionalSustained` on a `TemporalDag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_sustained_temporal_dag_bayesian_given_coverage() {
    measure_cell(
        "temporal_conditional_sustained_temporal_dag_bayesian_given_coverage",
        (Family::ConditionalSustained, Class::Dag, Mode::Bayesian),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_03A,
    );
}

/// `TemporalConditionalSustained` on a `TemporalDag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_sustained_temporal_dag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "temporal_conditional_sustained_temporal_dag_bayesian_graph_posterior_coverage",
        (Family::ConditionalSustained, Class::Dag, Mode::Bayesian),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_03B,
    );
}

/// `TemporalConditionalSustained` on a `TemporalCpdag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_sustained_temporal_cpdag_frequentist_given_coverage() {
    measure_cell(
        "temporal_conditional_sustained_temporal_cpdag_frequentist_given_coverage",
        (Family::ConditionalSustained, Class::Cpdag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_03C,
    );
}

/// `TemporalConditionalSustained` on a `TemporalCpdag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_sustained_temporal_cpdag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "temporal_conditional_sustained_temporal_cpdag_frequentist_graph_posterior_coverage",
        (Family::ConditionalSustained, Class::Cpdag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_03D,
    );
}

/// `TemporalConditionalSustained` on a `TemporalCpdag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_sustained_temporal_cpdag_bayesian_given_coverage() {
    measure_cell(
        "temporal_conditional_sustained_temporal_cpdag_bayesian_given_coverage",
        (Family::ConditionalSustained, Class::Cpdag, Mode::Bayesian),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_03E,
    );
}

/// `TemporalConditionalSustained` on a `TemporalCpdag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_sustained_temporal_cpdag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "temporal_conditional_sustained_temporal_cpdag_bayesian_graph_posterior_coverage",
        (Family::ConditionalSustained, Class::Cpdag, Mode::Bayesian),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_03F,
    );
}

/// `TemporalConditionalSustained` on a `TemporalPag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_sustained_temporal_pag_frequentist_given_coverage() {
    measure_cell(
        "temporal_conditional_sustained_temporal_pag_frequentist_given_coverage",
        (Family::ConditionalSustained, Class::Pag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_040,
    );
}

/// `TemporalConditionalSustained` on a `TemporalPag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_sustained_temporal_pag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "temporal_conditional_sustained_temporal_pag_frequentist_graph_posterior_coverage",
        (Family::ConditionalSustained, Class::Pag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_041,
    );
}

/// `TemporalConditionalSustained` on a `TemporalPag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_sustained_temporal_pag_bayesian_given_coverage() {
    measure_cell(
        "temporal_conditional_sustained_temporal_pag_bayesian_given_coverage",
        (Family::ConditionalSustained, Class::Pag, Mode::Bayesian),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_042,
    );
}

/// `TemporalConditionalSustained` on a `TemporalPag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_sustained_temporal_pag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "temporal_conditional_sustained_temporal_pag_bayesian_graph_posterior_coverage",
        (Family::ConditionalSustained, Class::Pag, Mode::Bayesian),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_043,
    );
}

/// `NestedCounterfactualEffect` on a `TemporalDag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn nested_counterfactual_effect_temporal_dag_frequentist_given_coverage() {
    measure_cell(
        "nested_counterfactual_effect_temporal_dag_frequentist_given_coverage",
        (Family::Nested, Class::Dag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_044,
    );
}

/// `NestedCounterfactualEffect` on a `TemporalDag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn nested_counterfactual_effect_temporal_dag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "nested_counterfactual_effect_temporal_dag_frequentist_graph_posterior_coverage",
        (Family::Nested, Class::Dag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_045,
    );
}

/// `NestedCounterfactualEffect` on a `TemporalDag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn nested_counterfactual_effect_temporal_dag_bayesian_given_coverage() {
    measure_cell(
        "nested_counterfactual_effect_temporal_dag_bayesian_given_coverage",
        (Family::Nested, Class::Dag, Mode::Bayesian),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_046,
    );
}

/// `NestedCounterfactualEffect` on a `TemporalDag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn nested_counterfactual_effect_temporal_dag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "nested_counterfactual_effect_temporal_dag_bayesian_graph_posterior_coverage",
        (Family::Nested, Class::Dag, Mode::Bayesian),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_047,
    );
}

/// `NestedCounterfactualEffect` on a `TemporalCpdag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn nested_counterfactual_effect_temporal_cpdag_frequentist_given_coverage() {
    measure_cell(
        "nested_counterfactual_effect_temporal_cpdag_frequentist_given_coverage",
        (Family::Nested, Class::Cpdag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_048,
    );
}

/// `NestedCounterfactualEffect` on a `TemporalCpdag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn nested_counterfactual_effect_temporal_cpdag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "nested_counterfactual_effect_temporal_cpdag_frequentist_graph_posterior_coverage",
        (Family::Nested, Class::Cpdag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_049,
    );
}

/// `NestedCounterfactualEffect` on a `TemporalCpdag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn nested_counterfactual_effect_temporal_cpdag_bayesian_given_coverage() {
    measure_cell(
        "nested_counterfactual_effect_temporal_cpdag_bayesian_given_coverage",
        (Family::Nested, Class::Cpdag, Mode::Bayesian),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_04A,
    );
}

/// `NestedCounterfactualEffect` on a `TemporalCpdag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn nested_counterfactual_effect_temporal_cpdag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "nested_counterfactual_effect_temporal_cpdag_bayesian_graph_posterior_coverage",
        (Family::Nested, Class::Cpdag, Mode::Bayesian),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_04B,
    );
}

/// `NestedCounterfactualEffect` on a `TemporalPag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn nested_counterfactual_effect_temporal_pag_frequentist_given_coverage() {
    measure_cell(
        "nested_counterfactual_effect_temporal_pag_frequentist_given_coverage",
        (Family::Nested, Class::Pag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_04C,
    );
}

/// `NestedCounterfactualEffect` on a `TemporalPag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn nested_counterfactual_effect_temporal_pag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "nested_counterfactual_effect_temporal_pag_frequentist_graph_posterior_coverage",
        (Family::Nested, Class::Pag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_04D,
    );
}

/// `NestedCounterfactualEffect` on a `TemporalPag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn nested_counterfactual_effect_temporal_pag_bayesian_given_coverage() {
    measure_cell(
        "nested_counterfactual_effect_temporal_pag_bayesian_given_coverage",
        (Family::Nested, Class::Pag, Mode::Bayesian),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_04E,
    );
}

/// `NestedCounterfactualEffect` on a `TemporalPag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn nested_counterfactual_effect_temporal_pag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "nested_counterfactual_effect_temporal_pag_bayesian_graph_posterior_coverage",
        (Family::Nested, Class::Pag, Mode::Bayesian),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_04F,
    );
}

/// `PathSpecificEffect` on a `TemporalDag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn path_specific_effect_temporal_dag_frequentist_given_coverage() {
    measure_cell(
        "path_specific_effect_temporal_dag_frequentist_given_coverage",
        (Family::PathSpecific, Class::Dag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_050,
    );
}

/// `PathSpecificEffect` on a `TemporalDag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn path_specific_effect_temporal_dag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "path_specific_effect_temporal_dag_frequentist_graph_posterior_coverage",
        (Family::PathSpecific, Class::Dag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_051,
    );
}

/// `PathSpecificEffect` on a `TemporalDag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn path_specific_effect_temporal_dag_bayesian_given_coverage() {
    measure_cell(
        "path_specific_effect_temporal_dag_bayesian_given_coverage",
        (Family::PathSpecific, Class::Dag, Mode::Bayesian),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_052,
    );
}

/// `PathSpecificEffect` on a `TemporalDag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn path_specific_effect_temporal_dag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "path_specific_effect_temporal_dag_bayesian_graph_posterior_coverage",
        (Family::PathSpecific, Class::Dag, Mode::Bayesian),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_053,
    );
}

/// `PathSpecificEffect` on a `TemporalCpdag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn path_specific_effect_temporal_cpdag_frequentist_given_coverage() {
    measure_cell(
        "path_specific_effect_temporal_cpdag_frequentist_given_coverage",
        (Family::PathSpecific, Class::Cpdag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_054,
    );
}

/// `PathSpecificEffect` on a `TemporalCpdag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn path_specific_effect_temporal_cpdag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "path_specific_effect_temporal_cpdag_frequentist_graph_posterior_coverage",
        (Family::PathSpecific, Class::Cpdag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_055,
    );
}

/// `PathSpecificEffect` on a `TemporalCpdag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn path_specific_effect_temporal_cpdag_bayesian_given_coverage() {
    measure_cell(
        "path_specific_effect_temporal_cpdag_bayesian_given_coverage",
        (Family::PathSpecific, Class::Cpdag, Mode::Bayesian),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_056,
    );
}

/// `PathSpecificEffect` on a `TemporalCpdag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn path_specific_effect_temporal_cpdag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "path_specific_effect_temporal_cpdag_bayesian_graph_posterior_coverage",
        (Family::PathSpecific, Class::Cpdag, Mode::Bayesian),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_057,
    );
}

/// `PathSpecificEffect` on a `TemporalPag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn path_specific_effect_temporal_pag_frequentist_given_coverage() {
    measure_cell(
        "path_specific_effect_temporal_pag_frequentist_given_coverage",
        (Family::PathSpecific, Class::Pag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_058,
    );
}

/// `PathSpecificEffect` on a `TemporalPag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn path_specific_effect_temporal_pag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "path_specific_effect_temporal_pag_frequentist_graph_posterior_coverage",
        (Family::PathSpecific, Class::Pag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_059,
    );
}

/// `PathSpecificEffect` on a `TemporalPag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn path_specific_effect_temporal_pag_bayesian_given_coverage() {
    measure_cell(
        "path_specific_effect_temporal_pag_bayesian_given_coverage",
        (Family::PathSpecific, Class::Pag, Mode::Bayesian),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_05A,
    );
}

/// `PathSpecificEffect` on a `TemporalPag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn path_specific_effect_temporal_pag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "path_specific_effect_temporal_pag_bayesian_graph_posterior_coverage",
        (Family::PathSpecific, Class::Pag, Mode::Bayesian),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_05B,
    );
}

/// `Counterfactual` on a `TemporalDag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn counterfactual_temporal_dag_frequentist_given_coverage() {
    measure_cell(
        "counterfactual_temporal_dag_frequentist_given_coverage",
        (Family::Counterfactual, Class::Dag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_05C,
    );
}

/// `Counterfactual` on a `TemporalDag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn counterfactual_temporal_dag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "counterfactual_temporal_dag_frequentist_graph_posterior_coverage",
        (Family::Counterfactual, Class::Dag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_05D,
    );
}

/// `Counterfactual` on a `TemporalDag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn counterfactual_temporal_dag_bayesian_given_coverage() {
    measure_cell(
        "counterfactual_temporal_dag_bayesian_given_coverage",
        (Family::Counterfactual, Class::Dag, Mode::Bayesian),
        "bayesian.temporal.gcomp",
        GIVEN,
        0x7EE_05E,
    );
}

/// `Counterfactual` on a `TemporalDag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn counterfactual_temporal_dag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "counterfactual_temporal_dag_bayesian_graph_posterior_coverage",
        (Family::Counterfactual, Class::Dag, Mode::Bayesian),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_05F,
    );
}

/// `Counterfactual` on a `TemporalCpdag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn counterfactual_temporal_cpdag_frequentist_given_coverage() {
    measure_cell(
        "counterfactual_temporal_cpdag_frequentist_given_coverage",
        (Family::Counterfactual, Class::Cpdag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_060,
    );
}

/// `Counterfactual` on a `TemporalCpdag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn counterfactual_temporal_cpdag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "counterfactual_temporal_cpdag_frequentist_graph_posterior_coverage",
        (Family::Counterfactual, Class::Cpdag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_061,
    );
}

/// `Counterfactual` on a `TemporalCpdag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn counterfactual_temporal_cpdag_bayesian_given_coverage() {
    measure_cell(
        "counterfactual_temporal_cpdag_bayesian_given_coverage",
        (Family::Counterfactual, Class::Cpdag, Mode::Bayesian),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_062,
    );
}

/// `Counterfactual` on a `TemporalCpdag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn counterfactual_temporal_cpdag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "counterfactual_temporal_cpdag_bayesian_graph_posterior_coverage",
        (Family::Counterfactual, Class::Cpdag, Mode::Bayesian),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_063,
    );
}

/// `Counterfactual` on a `TemporalPag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn counterfactual_temporal_pag_frequentist_given_coverage() {
    measure_cell(
        "counterfactual_temporal_pag_frequentist_given_coverage",
        (Family::Counterfactual, Class::Pag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_064,
    );
}

/// `Counterfactual` on a `TemporalPag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn counterfactual_temporal_pag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "counterfactual_temporal_pag_frequentist_graph_posterior_coverage",
        (Family::Counterfactual, Class::Pag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_065,
    );
}

/// `Counterfactual` on a `TemporalPag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn counterfactual_temporal_pag_bayesian_given_coverage() {
    measure_cell(
        "counterfactual_temporal_pag_bayesian_given_coverage",
        (Family::Counterfactual, Class::Pag, Mode::Bayesian),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_066,
    );
}

/// `Counterfactual` on a `TemporalPag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn counterfactual_temporal_pag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "counterfactual_temporal_pag_bayesian_graph_posterior_coverage",
        (Family::Counterfactual, Class::Pag, Mode::Bayesian),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_067,
    );
}

/// `InterventionalDistribution` on a `TemporalDag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn interventional_distribution_temporal_dag_frequentist_given_coverage() {
    measure_cell(
        "interventional_distribution_temporal_dag_frequentist_given_coverage",
        (Family::Distribution, Class::Dag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_068,
    );
}

/// `InterventionalDistribution` on a `TemporalDag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn interventional_distribution_temporal_dag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "interventional_distribution_temporal_dag_frequentist_graph_posterior_coverage",
        (Family::Distribution, Class::Dag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_069,
    );
}

/// `InterventionalDistribution` on a `TemporalDag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn interventional_distribution_temporal_dag_bayesian_given_coverage() {
    measure_cell(
        "interventional_distribution_temporal_dag_bayesian_given_coverage",
        (Family::Distribution, Class::Dag, Mode::Bayesian),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_06A,
    );
}

/// `InterventionalDistribution` on a `TemporalDag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn interventional_distribution_temporal_dag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "interventional_distribution_temporal_dag_bayesian_graph_posterior_coverage",
        (Family::Distribution, Class::Dag, Mode::Bayesian),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_06B,
    );
}

/// `InterventionalDistribution` on a `TemporalCpdag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn interventional_distribution_temporal_cpdag_frequentist_given_coverage() {
    measure_cell(
        "interventional_distribution_temporal_cpdag_frequentist_given_coverage",
        (Family::Distribution, Class::Cpdag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_06C,
    );
}

/// `InterventionalDistribution` on a `TemporalCpdag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn interventional_distribution_temporal_cpdag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "interventional_distribution_temporal_cpdag_frequentist_graph_posterior_coverage",
        (Family::Distribution, Class::Cpdag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_06D,
    );
}

/// `InterventionalDistribution` on a `TemporalCpdag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn interventional_distribution_temporal_cpdag_bayesian_given_coverage() {
    measure_cell(
        "interventional_distribution_temporal_cpdag_bayesian_given_coverage",
        (Family::Distribution, Class::Cpdag, Mode::Bayesian),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_06E,
    );
}

/// `InterventionalDistribution` on a `TemporalCpdag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn interventional_distribution_temporal_cpdag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "interventional_distribution_temporal_cpdag_bayesian_graph_posterior_coverage",
        (Family::Distribution, Class::Cpdag, Mode::Bayesian),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_06F,
    );
}

/// `InterventionalDistribution` on a `TemporalPag`, Frequentist, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn interventional_distribution_temporal_pag_frequentist_given_coverage() {
    measure_cell(
        "interventional_distribution_temporal_pag_frequentist_given_coverage",
        (Family::Distribution, Class::Pag, Mode::Frequentist),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_070,
    );
}

/// `InterventionalDistribution` on a `TemporalPag`, Frequentist, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn interventional_distribution_temporal_pag_frequentist_graph_posterior_coverage() {
    measure_cell(
        "interventional_distribution_temporal_pag_frequentist_graph_posterior_coverage",
        (Family::Distribution, Class::Pag, Mode::Frequentist),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_071,
    );
}

/// `InterventionalDistribution` on a `TemporalPag`, Bayesian, explicit and accepted.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn interventional_distribution_temporal_pag_bayesian_given_coverage() {
    measure_cell(
        "interventional_distribution_temporal_pag_bayesian_given_coverage",
        (Family::Distribution, Class::Pag, Mode::Bayesian),
        "temporal.linear.adjustment",
        GIVEN,
        0x7EE_072,
    );
}

/// `InterventionalDistribution` on a `TemporalPag`, Bayesian, a one-atom graph posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn interventional_distribution_temporal_pag_bayesian_graph_posterior_coverage() {
    measure_cell(
        "interventional_distribution_temporal_pag_bayesian_graph_posterior_coverage",
        (Family::Distribution, Class::Pag, Mode::Bayesian),
        "temporal.linear.adjustment",
        POSTERIOR,
        0x7EE_073,
    );
}

/// `PulseEffect` (the lag-1 pulse) on a `TemporalDag`, Frequentist, explicit and accepted, by the
/// lagged instrument.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn pulse_effect_temporal_dag_frequentist_given_lagged_instrument() {
    measure_route(
        "pulse_effect_temporal_dag_frequentist_given_lagged_instrument",
        (Family::Pulse, Class::Dag, Mode::Frequentist, Route::Instrument),
        "temporal.lagged_instrument",
        GIVEN_WITHOUT_EXPLICIT_UNVALIDATED,
        0x7EE_100,
    );
}

/// `PulseEffect` (the lag-1 pulse) on a `TemporalDag`, Bayesian, explicit and accepted, by the
/// lagged instrument.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn pulse_effect_temporal_dag_bayesian_given_lagged_instrument() {
    measure_route(
        "pulse_effect_temporal_dag_bayesian_given_lagged_instrument",
        (Family::Pulse, Class::Dag, Mode::Bayesian, Route::Instrument),
        "bayesian.temporal.lagged_instrument",
        GIVEN_VALIDATED,
        0x7EE_101,
    );
}

/// `PulseEffect` (the lag-1 pulse) on a `TemporalDag`, Frequentist, explicit and accepted, by the
/// GLM link contrast.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn pulse_effect_temporal_dag_frequentist_given_glm_link() {
    measure_route(
        "pulse_effect_temporal_dag_frequentist_given_glm_link",
        (Family::Pulse, Class::Dag, Mode::Frequentist, Route::Link),
        "temporal.glm_link_contrast",
        GIVEN_WITHOUT_EXPLICIT_UNVALIDATED,
        0x7EE_102,
    );
}

/// `PulseEffect` (the lag-1 pulse) on a `TemporalDag`, Bayesian, explicit and accepted, by the GLM
/// link contrast.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn pulse_effect_temporal_dag_bayesian_given_glm_link() {
    measure_route(
        "pulse_effect_temporal_dag_bayesian_given_glm_link",
        (Family::Pulse, Class::Dag, Mode::Bayesian, Route::Link),
        "bayesian.temporal.glm_link_contrast",
        GIVEN_VALIDATED,
        0x7EE_103,
    );
}

/// `PulseEffect` (the lag-1 pulse) on a `TemporalDag`, Bayesian, explicit and accepted, by the
/// white-innovation linear posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn pulse_effect_temporal_dag_bayesian_given_white_innovation() {
    measure_route(
        "pulse_effect_temporal_dag_bayesian_given_white_innovation",
        (Family::Pulse, Class::Dag, Mode::Bayesian, Route::White),
        "bayesian.temporal.white_innovation_linear",
        GIVEN_VALIDATED,
        0x7EE_104,
    );
}

/// `SustainedEffect` (the one-step window) on a `TemporalDag`, Frequentist, explicit and accepted,
/// by the lagged instrument.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_dag_frequentist_given_lagged_instrument() {
    measure_route(
        "sustained_effect_temporal_dag_frequentist_given_lagged_instrument",
        (Family::SustainedStep, Class::Dag, Mode::Frequentist, Route::Instrument),
        "temporal.lagged_instrument",
        GIVEN_WITHOUT_EXPLICIT_UNVALIDATED,
        0x7EE_105,
    );
}

/// `SustainedEffect` (the one-step window) on a `TemporalDag`, Bayesian, explicit and accepted, by
/// the lagged instrument.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_dag_bayesian_given_lagged_instrument() {
    measure_route(
        "sustained_effect_temporal_dag_bayesian_given_lagged_instrument",
        (Family::SustainedStep, Class::Dag, Mode::Bayesian, Route::Instrument),
        "bayesian.temporal.lagged_instrument",
        GIVEN_WITHOUT_ACCEPTED_UNVALIDATED,
        0x7EE_106,
    );
}

/// `SustainedEffect` (the one-step window) on a `TemporalDag`, Frequentist, explicit and accepted,
/// by the GLM link contrast.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_dag_frequentist_given_glm_link() {
    measure_route(
        "sustained_effect_temporal_dag_frequentist_given_glm_link",
        (Family::SustainedStep, Class::Dag, Mode::Frequentist, Route::Link),
        "temporal.glm_link_contrast",
        GIVEN_WITHOUT_EXPLICIT_UNVALIDATED,
        0x7EE_107,
    );
}

/// `SustainedEffect` (the one-step window) on a `TemporalDag`, Bayesian, explicit and accepted, by
/// the GLM link contrast.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_dag_bayesian_given_glm_link() {
    measure_route(
        "sustained_effect_temporal_dag_bayesian_given_glm_link",
        (Family::SustainedStep, Class::Dag, Mode::Bayesian, Route::Link),
        "bayesian.temporal.glm_link_contrast",
        GIVEN_WITHOUT_ACCEPTED_UNVALIDATED,
        0x7EE_108,
    );
}

/// `SustainedEffect` (the one-step window) on a `TemporalDag`, Bayesian, explicit and accepted, by
/// the white-innovation linear posterior.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_dag_bayesian_given_white_innovation() {
    measure_route(
        "sustained_effect_temporal_dag_bayesian_given_white_innovation",
        (Family::SustainedStep, Class::Dag, Mode::Bayesian, Route::White),
        "bayesian.temporal.white_innovation_linear",
        GIVEN_WITHOUT_ACCEPTED_UNVALIDATED,
        0x7EE_109,
    );
}

/// `SustainedEffect` (the one-step window) on a `TemporalDag`, Frequentist, explicit and accepted,
/// by `temporal.linear.adjustment`.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_dag_frequentist_given_one_step_window() {
    measure_route(
        "sustained_effect_temporal_dag_frequentist_given_one_step_window",
        (Family::SustainedStep, Class::Dag, Mode::Frequentist, Route::Adjustment),
        "temporal.linear.adjustment",
        GIVEN_WITHOUT_EXPLICIT_UNVALIDATED,
        0x7EE_10A,
    );
}

/// `SustainedEffect` (the one-step window) on a `TemporalDag`, Frequentist, a one-atom graph
/// posterior, by `temporal.linear.adjustment`.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_dag_frequentist_graph_posterior_one_step_window() {
    measure_route(
        "sustained_effect_temporal_dag_frequentist_graph_posterior_one_step_window",
        (Family::SustainedStep, Class::Dag, Mode::Frequentist, Route::Adjustment),
        "temporal.linear.adjustment",
        POSTERIOR_VALIDATED,
        0x7EE_10B,
    );
}

/// `SustainedEffect` (the one-step window) on a `TemporalDag`, Bayesian, explicit and accepted, by
/// `bayesian.temporal.gcomp`.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_dag_bayesian_given_one_step_window() {
    measure_route(
        "sustained_effect_temporal_dag_bayesian_given_one_step_window",
        (Family::SustainedStep, Class::Dag, Mode::Bayesian, Route::Adjustment),
        "bayesian.temporal.gcomp",
        GIVEN_WITHOUT_ACCEPTED_UNVALIDATED,
        0x7EE_10C,
    );
}

/// `SustainedEffect` (the one-step window) on a `TemporalDag`, Bayesian, a one-atom graph
/// posterior, by `bayesian.temporal.gcomp`.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_dag_bayesian_graph_posterior_one_step_window() {
    measure_route(
        "sustained_effect_temporal_dag_bayesian_graph_posterior_one_step_window",
        (Family::SustainedStep, Class::Dag, Mode::Bayesian, Route::Adjustment),
        "bayesian.temporal.gcomp",
        POSTERIOR_VALIDATED,
        0x7EE_10D,
    );
}

/// `SustainedEffect` (the one-step window) on a `TemporalCpdag`, Frequentist, explicit and
/// accepted, by `temporal.linear.adjustment`.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_cpdag_frequentist_given_one_step_window() {
    measure_route(
        "sustained_effect_temporal_cpdag_frequentist_given_one_step_window",
        (Family::SustainedStep, Class::Cpdag, Mode::Frequentist, Route::Adjustment),
        "temporal.linear.adjustment",
        GIVEN_WITHOUT_ACCEPTED_UNVALIDATED,
        0x7EE_10E,
    );
}

/// `SustainedEffect` (the one-step window) on a `TemporalCpdag`, Frequentist, a one-atom graph
/// posterior, by `temporal.linear.adjustment`.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_cpdag_frequentist_graph_posterior_one_step_window() {
    measure_route(
        "sustained_effect_temporal_cpdag_frequentist_graph_posterior_one_step_window",
        (Family::SustainedStep, Class::Cpdag, Mode::Frequentist, Route::Adjustment),
        "temporal.linear.adjustment",
        POSTERIOR_VALIDATED,
        0x7EE_10F,
    );
}

/// `SustainedEffect` (the one-step window) on a `TemporalCpdag`, Bayesian, explicit and accepted,
/// by `bayesian.temporal.gcomp`.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_cpdag_bayesian_given_one_step_window() {
    measure_route(
        "sustained_effect_temporal_cpdag_bayesian_given_one_step_window",
        (Family::SustainedStep, Class::Cpdag, Mode::Bayesian, Route::Adjustment),
        "bayesian.temporal.gcomp",
        GIVEN_WITHOUT_ACCEPTED_UNVALIDATED,
        0x7EE_110,
    );
}

/// `SustainedEffect` (the one-step window) on a `TemporalCpdag`, Bayesian, a one-atom graph
/// posterior, by `bayesian.temporal.gcomp`.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_cpdag_bayesian_graph_posterior_one_step_window() {
    measure_route(
        "sustained_effect_temporal_cpdag_bayesian_graph_posterior_one_step_window",
        (Family::SustainedStep, Class::Cpdag, Mode::Bayesian, Route::Adjustment),
        "bayesian.temporal.gcomp",
        POSTERIOR_VALIDATED,
        0x7EE_111,
    );
}

/// `SustainedEffect` (the one-step window) on a `TemporalPag`, Frequentist, explicit and accepted,
/// by `temporal.linear.adjustment`.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_pag_frequentist_given_one_step_window() {
    measure_route(
        "sustained_effect_temporal_pag_frequentist_given_one_step_window",
        (Family::SustainedStep, Class::Pag, Mode::Frequentist, Route::Adjustment),
        "temporal.linear.adjustment",
        GIVEN_WITHOUT_ACCEPTED_UNVALIDATED,
        0x7EE_112,
    );
}

/// `SustainedEffect` (the one-step window) on a `TemporalPag`, Frequentist, a one-atom graph
/// posterior, by `temporal.linear.adjustment`.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_pag_frequentist_graph_posterior_one_step_window() {
    measure_route(
        "sustained_effect_temporal_pag_frequentist_graph_posterior_one_step_window",
        (Family::SustainedStep, Class::Pag, Mode::Frequentist, Route::Adjustment),
        "temporal.linear.adjustment",
        POSTERIOR_VALIDATED,
        0x7EE_113,
    );
}

/// `SustainedEffect` (the one-step window) on a `TemporalPag`, Bayesian, explicit and accepted, by
/// `bayesian.temporal.gcomp`.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_pag_bayesian_given_one_step_window() {
    measure_route(
        "sustained_effect_temporal_pag_bayesian_given_one_step_window",
        (Family::SustainedStep, Class::Pag, Mode::Bayesian, Route::Adjustment),
        "bayesian.temporal.gcomp",
        GIVEN_WITHOUT_ACCEPTED_UNVALIDATED,
        0x7EE_114,
    );
}

/// `SustainedEffect` (the one-step window) on a `TemporalPag`, Bayesian, a one-atom graph
/// posterior, by `bayesian.temporal.gcomp`.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn sustained_effect_temporal_pag_bayesian_graph_posterior_one_step_window() {
    measure_route(
        "sustained_effect_temporal_pag_bayesian_graph_posterior_one_step_window",
        (Family::SustainedStep, Class::Pag, Mode::Bayesian, Route::Adjustment),
        "bayesian.temporal.gcomp",
        POSTERIOR_VALIDATED,
        0x7EE_115,
    );
}

/// `TemporalConditionalSustained` (the two-step window marginal over the modifier) on a
/// `TemporalDag`, Frequentist, explicit and accepted, by `temporal.sequential.gcomp`.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_sustained_temporal_dag_frequentist_given_sequential_gcomp() {
    measure_route(
        "temporal_conditional_sustained_temporal_dag_frequentist_given_sequential_gcomp",
        (Family::ConditionalMultiStep, Class::Dag, Mode::Frequentist, Route::Adjustment),
        "temporal.sequential.gcomp",
        GIVEN,
        0x7EE_116,
    );
}

/// `TemporalConditionalSustained` (the two-step window marginal over the modifier) on a
/// `TemporalDag`, Bayesian, explicit and accepted, by `temporal.sequential.gcomp`.
#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn temporal_conditional_sustained_temporal_dag_bayesian_given_sequential_gcomp() {
    measure_route(
        "temporal_conditional_sustained_temporal_dag_bayesian_given_sequential_gcomp",
        (Family::ConditionalMultiStep, Class::Dag, Mode::Bayesian, Route::Adjustment),
        "temporal.sequential.gcomp",
        GIVEN,
        0x7EE_117,
    );
}
