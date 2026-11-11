//! Repeated-sampling coverage of the static effect, conditional-effect, intervention
//! response, mean-curve, interventional-distribution and mediation estimators on every
//! graph class, structure route and validation mode they are offered under.
//!
//! One test measures one coordinate: a query, a graph class, an inference mode, an
//! estimator and a structure family (a given graph or tiered background, or a graph
//! posterior). The validation modes and, for a given graph, its accepted twin are sibling
//! executions of that coordinate: the driver measures the first listed execution on every
//! replicate and requires every sibling to report its interval bit for bit.
//!
//! Every law has a closed-form population truth:
//!
//! * Average effects: a binary treatment confounded by `z` with a homogeneous effect of
//!   `2` ([`linear_ate_data`], [`cpdag_ate_data`], [`pag_ate_data`]); the design
//!   estimators each have their own law (front door, sharp discontinuity, instrument,
//!   randomized median shift) documented at its generator.
//! * Estimators whose estimate carries no influence function (propensity weighting,
//!   matching, stratification, distance matching, GLM adjustment) are measured on a
//!   one-member CPDAG or PAG ([`compelled_ate_data`]): on a class with several members the
//!   runtime publishes no standard error for them.
//! * GLM adjustment: a logistic binary outcome; its target is the risk difference averaged
//!   over the sample's confounder values ([`sample_risk_difference`]).
//! * The finite-discrete `functional.effect` plug-in: a binary confounded law with effect
//!   `0.4` in every stratum ([`discrete_effect_data`]).
//! * Conditional effects: `y` carries `0.8 t x` with a binary modifier `x`, so the
//!   modifier-averaged effect is the base slope plus `0.8 P(x = 1)`.
//! * Intervention responses and mean curves: linear in the dose, `m(a) = 1 + 2a`
//!   (`m(a) = a` on the CPDAG law).
//! * Interventional distributions: binary outcomes whose `P(Y = 1 | do(t = 1))` is a
//!   sum of products of the generating probabilities.
//! * Mediation: the linear law of [`mediation_data`], natural indirect effect `0.3`.
//! * Tiered backgrounds: the co-determined closure functional (`2`), the two canonical
//!   orientation scenarios of an undecided tier (`1` and `-1`), and the co-determined
//!   joint level (`0.9`).
//!
//! A Bayesian response holds the empirical covariate law fixed, so its target is the
//! response at the sample covariate means. Its laws therefore centre every exogenous
//! covariate on its sample mean, which makes that target the population response
//! exactly while leaving the conditional law of the outcome untouched.
//!
//! A graph posterior holds one atom: one identified atom publishes its own sampling
//! interval, whereas a mixture over atoms is a distribution over graphs.
//!
//! The driver scores the reported 0.95 interval and the 0.90 interval of the same
//! execution: a symmetric Frequentist interval from its standard error, a posterior
//! interval from the draws the estimator retained, a bootstrap probability interval from
//! its logit construction, and the Anderson–Rubin set by inverting the same statistic on
//! the same rows at the 0.90 level.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::too_many_lines,
    clippy::too_many_arguments,
    reason = "test scaffolding compares exact constants and keeps one design table in one body"
)]

mod common;
#[path = "common/grid_cell.rs"]
mod grid_cell;

use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};

use antecedent::{
    AcceptedGraph, BayesianConfig, DesignPosteriorModel, EstimatorId, IdentifierId, InferenceMode,
    RefuteSuite, Study, StudyBuilder, StudyResult,
};
use antecedent_core::{
    AverageEffectQuery, CausalQuery, ConditionalEffectQuery, ContinuousDomain, ExecutionContext,
    GridSpec, Intervention, InterventionalDistributionQuery, MediationContrast, MediationQuery,
    OutcomeFunctional, ResponseFunctional, ResponseQuery, Value, VariableId,
};
use antecedent_data::{TableView, TabularData};
use antecedent_discovery::adjacency_mask::{
    AdjacencyMask, adjacency_mask_from_admg, adjacency_mask_from_cpdag, adjacency_masks_from_pag,
    set_edge,
};
use antecedent_discovery::{GraphPosterior, GraphPosteriorAtomKind, SamplingSummary};
use antecedent_estimate::logit_probability_interval;
use antecedent_graph::{
    Admg, Cpdag, Dag, DenseNodeId, Endpoint, MarkedEdge, MiddleMark, Pag, TemporalAdmg,
    TieredBackground, WithinTier,
};
use antecedent_prob::InferenceDiagnostics;
use antecedent_stats::{
    FaerBackend, IvBlocks, LeastSquaresWorkspace, anderson_rubin_confidence_set,
};
use common::calibration::{gaussian, grid_n, stream_seed};
use common::reported::{
    GATE_LEVEL, REPORTED_LEVEL, posterior_pair, response_normal_pair, response_posterior_pairs,
    scalar_normal_pair,
};
use common::static_dgp::{
    bernoulli, distribution_data, linear_ate_data, response_data, sigmoid, table, uniform,
};
use grid_cell::{Cell, IntervalPair, Target, measure, measure_sample_truths};

/// Posterior draws of a Bayesian execution.
const DRAWS: usize = 400;
/// Dose of the scalar intervention response.
const LEVEL: f64 = 1.0;
/// Doses of the mean curve.
const CURVE_GRID: [f64; 3] = [-1.0, 0.0, 1.0];

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

/// Subtract the sample mean, so the column's sample mean is zero.
fn centre(column: &mut [f64]) {
    let mean = column.iter().sum::<f64>() / column.len() as f64;
    for value in column {
        *value -= mean;
    }
}

/// `n` standard normal draws, centred on their sample mean when `centred`.
fn exogenous(g: &mut impl FnMut() -> f64, n: usize, centred: bool) -> Vec<f64> {
    let mut column: Vec<f64> = (0..n).map(|_| g()).collect();
    if centred {
        centre(&mut column);
    }
    column
}

// ------------------------------------------------------------------ data laws

/// Columns `t, y, z, w`: `z, w ~ N(0,1)`, `t ~ Bern(sigmoid(0.8 z - 0.2))`,
/// `y = 1 + 2t + z + e`. Both orientations of `w — z` in [`agreeing_cpdag`] adjust
/// `{z}`; the average effect is `2`.
fn cpdag_ate_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (mut t, mut y, mut z, mut w) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        w[i] = g();
        t[i] = bernoulli(&mut u, sigmoid(0.8 * z[i] - 0.2));
        y[i] = 1.0 + 2.0 * t[i] + z[i] + g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z), ("w", &w)])
}

/// Columns `t, y, z, r`: `z, r ~ N(0,1)`, `t ~ Bern(sigmoid(0.6 z + 0.5 r))`,
/// `y = 1 + 2t + 0.8 z + e`. Every completion of [`agreeing_pag`] adjusts `{z}` (with or
/// without the outcome-irrelevant `r`); the average effect is `2`.
fn pag_ate_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (mut t, mut y, mut z, mut r) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        r[i] = g();
        t[i] = bernoulli(&mut u, sigmoid(0.6 * z[i] + 0.5 * r[i]));
        y[i] = 1.0 + 2.0 * t[i] + 0.8 * z[i] + g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z), ("r", &r)])
}

/// Columns `t, y, r, w`: `r, w ~ N(0,1)`, `t ~ Bern(sigmoid(0.6 r + 0.6 w - 0.2))`,
/// `y = 1 + 2t + 0.8 w + e`. The collider `r -> t <- w` compels every edge of
/// [`compelled_cpdag`] and [`decided_pag`], so each class has the one member that adjusts
/// `{w}`; the average effect is `2`.
fn compelled_ate_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (mut t, mut y, mut r, mut w) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        r[i] = g();
        w[i] = g();
        t[i] = bernoulli(&mut u, sigmoid(0.6 * r[i] + 0.6 * w[i] - 0.2));
        y[i] = 1.0 + 2.0 * t[i] + 0.8 * w[i] + g();
    }
    table(&[("t", &t), ("y", &y), ("r", &r), ("w", &w)])
}

/// Logit intercept, treatment and confounder coefficients of the binary-outcome laws.
const GLM_LOGIT: [f64; 3] = [-0.3, 0.9, 0.8];

/// The g-computation risk difference of the binary-outcome laws averaged over the
/// sample's confounder values: the functional the GLM adjustment reports, whose delta-method
/// standard error holds the covariates fixed.
fn sample_risk_difference(confounder: &[f64]) -> f64 {
    let [intercept, effect, slope] = GLM_LOGIT;
    confounder
        .iter()
        .map(|c| sigmoid(intercept + effect + slope * c) - sigmoid(intercept + slope * c))
        .sum::<f64>()
        / confounder.len() as f64
}

/// `(t, y, z)`: `z ~ N(0,1)`, `t ~ Bern(sigmoid(0.8 z - 0.2))`,
/// `y ~ Bern(sigmoid(-0.3 + 0.9 t + 0.8 z))`.
fn binary_outcome_columns(n: usize, seed: u64) -> [Vec<f64>; 3] {
    let [intercept, effect, slope] = GLM_LOGIT;
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        t[i] = bernoulli(&mut u, sigmoid(0.8 * z[i] - 0.2));
        y[i] = bernoulli(&mut u, sigmoid(intercept + effect * t[i] + slope * z[i]));
    }
    [t, y, z]
}

fn binary_outcome_data(n: usize, seed: u64) -> TabularData {
    let [t, y, z] = binary_outcome_columns(n, seed);
    table(&[("t", &t), ("y", &y), ("z", &z)])
}

fn binary_outcome_truth(n: usize, seed: u64) -> Vec<f64> {
    vec![sample_risk_difference(&binary_outcome_columns(n, seed)[2])]
}

/// `(t, y, r, w)`: [`compelled_ate_data`]'s design with the binary outcome
/// `y ~ Bern(sigmoid(-0.3 + 0.9 t + 0.8 w))`.
fn compelled_binary_outcome_columns(n: usize, seed: u64) -> [Vec<f64>; 4] {
    let [intercept, effect, slope] = GLM_LOGIT;
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (mut t, mut y, mut r, mut w) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        r[i] = g();
        w[i] = g();
        t[i] = bernoulli(&mut u, sigmoid(0.6 * r[i] + 0.6 * w[i] - 0.2));
        y[i] = bernoulli(&mut u, sigmoid(intercept + effect * t[i] + slope * w[i]));
    }
    [t, y, r, w]
}

fn compelled_binary_outcome_data(n: usize, seed: u64) -> TabularData {
    let [t, y, r, w] = compelled_binary_outcome_columns(n, seed);
    table(&[("t", &t), ("y", &y), ("r", &r), ("w", &w)])
}

fn compelled_binary_outcome_truth(n: usize, seed: u64) -> Vec<f64> {
    vec![sample_risk_difference(&compelled_binary_outcome_columns(n, seed)[3])]
}

/// Average effect of [`discrete_effect_data`]: risks `0.1 + 0.4 t + 0.4 z`, so `0.4` in
/// every stratum.
const DISCRETE_EFFECT_TRUTH: f64 = 0.4;

/// Columns `t, y, z`, all binary: `z ~ Bern(1/2)`, `t ~ Bern(0.3 + 0.4 z)`,
/// `y ~ Bern(0.1 + 0.4 t + 0.4 z)`; the finite-discrete plug-in law.
fn discrete_effect_data(n: usize, seed: u64) -> TabularData {
    let mut u = uniform(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = bernoulli(&mut u, 0.5);
        t[i] = bernoulli(&mut u, 0.3 + 0.4 * z[i]);
        y[i] = bernoulli(&mut u, 0.1 + 0.4 * t[i] + 0.4 * z[i]);
    }
    table(&[("t", &t), ("y", &y), ("z", &z)])
}

/// Share of the binary effect modifier.
const MODIFIER_SHARE: f64 = 0.5;
/// Interaction of the treatment with the modifier.
const MODIFIER_SLOPE: f64 = 0.8;

/// Columns `t, y, z, x`: `z ~ N(0,1)`, `x ~ Bern(1/2)`, `t = 0.8 z + e`,
/// `y = t + 0.8 t x + z + 0.3 x + e`; the modifier-averaged effect is `1 + 0.8 / 2`.
fn conditional_dag_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (mut t, mut y, mut z, mut x) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        x[i] = bernoulli(&mut u, MODIFIER_SHARE);
        t[i] = 0.8 * z[i] + g();
        y[i] = t[i] + MODIFIER_SLOPE * t[i] * x[i] + z[i] + 0.3 * x[i] + g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z), ("x", &x)])
}

/// `(t, y, z, x)`: `z ~ N(0,1)`, `x ~ Bern(1/2)`, `t ~ Bern(sigmoid(0.8 z - 0.2))`,
/// `y = 1 + t + 0.8 t x + z + 0.3 x + e`: the effect at modifier `x` is `1 + 0.8 x`.
fn binary_conditional_columns(n: usize, seed: u64) -> [Vec<f64>; 4] {
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (mut t, mut y, mut z, mut x) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        x[i] = bernoulli(&mut u, MODIFIER_SHARE);
        t[i] = bernoulli(&mut u, sigmoid(0.8 * z[i] - 0.2));
        y[i] = 1.0 + t[i] + MODIFIER_SLOPE * t[i] * x[i] + z[i] + 0.3 * x[i] + g();
    }
    [t, y, z, x]
}

fn binary_conditional_data(n: usize, seed: u64) -> TabularData {
    let [t, y, z, x] = binary_conditional_columns(n, seed);
    table(&[("t", &t), ("y", &y), ("z", &z), ("x", &x)])
}

/// The modifier-averaged effect over the sample's modifier values, `1 + 0.8 mean(x)`: the
/// basis g-computation posterior averages over the empirical covariate law.
fn binary_conditional_truth(n: usize, seed: u64) -> Vec<f64> {
    let x = &binary_conditional_columns(n, seed)[3];
    vec![1.0 + MODIFIER_SLOPE * x.iter().sum::<f64>() / x.len() as f64]
}

/// [`conditional_dag_data`] with a fifth column `w ~ N(0,1)` adjacent to `z` only
/// (columns `t, y, z, x, w`).
fn conditional_cpdag_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let mut cols: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::with_capacity(n));
    for _ in 0..n {
        let (z, w) = (g(), g());
        let x = bernoulli(&mut u, MODIFIER_SHARE);
        let t = 0.8 * z + g();
        cols[0].push(t);
        cols[1].push(t + MODIFIER_SLOPE * t * x + z + 0.3 * x + g());
        cols[2].push(z);
        cols[3].push(x);
        cols[4].push(w);
    }
    table(&[("t", &cols[0]), ("y", &cols[1]), ("z", &cols[2]), ("x", &cols[3]), ("w", &cols[4])])
}

/// Columns `t, y, z, r, x`: `z, r ~ N(0,1)`, `x ~ Bern(1/2)`, `t = 0.5 z + 0.4 r + e`,
/// `y = 1 + 2t + 0.8 t x + 0.8 z + 0.3 x + e`; the modifier-averaged effect is
/// `2 + 0.8 / 2`.
fn conditional_pag_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let mut cols: [Vec<f64>; 5] = std::array::from_fn(|_| Vec::with_capacity(n));
    for _ in 0..n {
        let (z, r) = (g(), g());
        let x = bernoulli(&mut u, MODIFIER_SHARE);
        let t = 0.5 * z + 0.4 * r + g();
        cols[0].push(t);
        cols[1].push(1.0 + 2.0 * t + MODIFIER_SLOPE * t * x + 0.8 * z + 0.3 * x + g());
        cols[2].push(z);
        cols[3].push(r);
        cols[4].push(x);
    }
    table(&[("t", &cols[0]), ("y", &cols[1]), ("z", &cols[2]), ("r", &cols[3]), ("x", &cols[4])])
}

/// Columns `t, y, z`: `t = 0.3 z + sqrt(0.91) e`, `y = 1 + 2t + 0.8 z + e` with `z`
/// centred on its sample mean; `m(a) = 1 + 2a` at the sample covariate mean.
fn centred_response_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let z = exogenous(&mut g, n, true);
    let (mut t, mut y) = (vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        t[i] = 0.3 * z[i] + 0.91_f64.sqrt() * g();
        y[i] = 1.0 + 2.0 * t[i] + 0.8 * z[i] + g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z)])
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

/// Columns `t, y, z, w`: `z, w ~ N(0,1)`, `t = 0.8 z + e`, `y = t + z + e`; `m(a) = a`.
fn cpdag_response(n: usize, seed: u64, centred: bool) -> TabularData {
    let mut g = gaussian(seed);
    let z = exogenous(&mut g, n, centred);
    let w = exogenous(&mut g, n, centred);
    let (mut t, mut y) = (vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        t[i] = 0.8 * z[i] + g();
        y[i] = t[i] + z[i] + g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z), ("w", &w)])
}

fn cpdag_response_data(n: usize, seed: u64) -> TabularData {
    cpdag_response(n, seed, false)
}

fn centred_cpdag_response_data(n: usize, seed: u64) -> TabularData {
    cpdag_response(n, seed, true)
}

/// Columns `t, y, z, r`: `z, r ~ N(0,1)`, `t = 0.5 z + 0.4 r + e`,
/// `y = 1 + 2t + 0.8 z + e`; `m(a) = 1 + 2a`.
fn pag_response(n: usize, seed: u64, centred: bool) -> TabularData {
    let mut g = gaussian(seed);
    let z = exogenous(&mut g, n, centred);
    let r = exogenous(&mut g, n, centred);
    let (mut t, mut y) = (vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        t[i] = 0.5 * z[i] + 0.4 * r[i] + g();
        y[i] = 1.0 + 2.0 * t[i] + 0.8 * z[i] + g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z), ("r", &r)])
}

fn pag_response_data(n: usize, seed: u64) -> TabularData {
    pag_response(n, seed, false)
}

fn centred_pag_response_data(n: usize, seed: u64) -> TabularData {
    pag_response(n, seed, true)
}

/// Joint level of [`cell_data`] at `(t1, t2) = (1, 1)`: `1 + 1 + 0.5 + 0.7`.
const CELL_TRUTH: f64 = 3.2;

/// Columns `t1, t2, y, z`: `z ~ N(0,1)`; the treatment cell `2 t1 + t2` is multinomial
/// logit in `z` with intercepts `(0, -0.2, 0.1, -0.3)` and slopes `(0, 0.5, -0.4, 0.8)`;
/// `y = 1 + t1 + 0.5 t2 + 0.7 t1 t2 + z + 0.2 z t1 + e`.
fn cell_data(n: usize, seed: u64) -> TabularData {
    const INTERCEPT: [f64; 4] = [0.0, -0.2, 0.1, -0.3];
    const SLOPE: [f64; 4] = [0.0, 0.5, -0.4, 0.8];
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (mut t1, mut t2, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        z[i] = g();
        let weights: [f64; 4] = std::array::from_fn(|k| (INTERCEPT[k] + SLOPE[k] * z[i]).exp());
        let mut remaining = u() * weights.iter().sum::<f64>();
        let mut cell = 3;
        for (k, weight) in weights.iter().enumerate() {
            if remaining < *weight {
                cell = k;
                break;
            }
            remaining -= weight;
        }
        t1[i] = f64::from(cell >= 2);
        t2[i] = f64::from(cell % 2 == 1);
        y[i] = 1.0 + t1[i] + 0.5 * t2[i] + 0.7 * t1[i] * t2[i] + z[i] + 0.2 * z[i] * t1[i] + g();
    }
    table(&[("t1", &t1), ("t2", &t2), ("y", &y), ("z", &z)])
}

/// Columns `t, m, y, x`: `x ~ N(0,1)`, `t = 0.5 x + e`, `m = 0.6 t + 0.8 e`,
/// `y = 0.4 t + 0.5 m + 0.5 x + e`; the natural indirect effect is `0.6 · 0.5`.
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

/// `P(Y = 1 | do(t = 1))` of [`distribution_data`] at base `0.45`.
const DAG_DISTRIBUTION_TRUTH: f64 = 0.465;

fn dag_distribution_data(n: usize, seed: u64) -> TabularData {
    distribution_data(n, 0.45, seed)
}

/// `P(Y = 1 | do(t = 1))` of [`front_door_binary_data`]: `0.1 + 0.4 · 0.8 + 0.3 · 0.5`.
const FRONT_DOOR_DISTRIBUTION_TRUTH: f64 = 0.57;

/// Columns `t, m, y` with a latent `u ~ Bern(1/2)`: `t ~ Bern(0.3 + 0.4 u)`,
/// `m ~ Bern(0.2 + 0.6 t)`, `y ~ Bern(0.1 + 0.4 m + 0.3 u)`.
fn front_door_binary_data(n: usize, seed: u64) -> TabularData {
    let mut u = uniform(seed);
    let (mut t, mut m, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let latent = bernoulli(&mut u, 0.5);
        t[i] = bernoulli(&mut u, 0.3 + 0.4 * latent);
        m[i] = bernoulli(&mut u, 0.2 + 0.6 * t[i]);
        y[i] = bernoulli(&mut u, 0.1 + 0.4 * m[i] + 0.3 * latent);
    }
    table(&[("t", &t), ("m", &m), ("y", &y)])
}

/// `P(Y = 1 | do(t = 1))` of [`compelled_binary_data`]: `0.2 + 0.4 + 0.2 · 0.5`.
const COMPELLED_DISTRIBUTION_TRUTH: f64 = 0.7;

/// Columns `t, y, r, w`, all binary: `r, w ~ Bern(1/2)`,
/// `t ~ Bern(0.2 + 0.3 r + 0.3 w)`, `y ~ Bern(0.2 + 0.4 t + 0.2 w)`.
fn compelled_binary_data(n: usize, seed: u64) -> TabularData {
    let mut u = uniform(seed);
    let (mut t, mut y, mut r, mut w) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        r[i] = bernoulli(&mut u, 0.5);
        w[i] = bernoulli(&mut u, 0.5);
        t[i] = bernoulli(&mut u, 0.2 + 0.3 * r[i] + 0.3 * w[i]);
        y[i] = bernoulli(&mut u, 0.2 + 0.4 * t[i] + 0.2 * w[i]);
    }
    table(&[("t", &t), ("y", &y), ("r", &r), ("w", &w)])
}

/// `P(Y = 1 | do(t = 1))` of [`collider_binary_data`]: `0.2 + 0.4`.
const COLLIDER_DISTRIBUTION_TRUTH: f64 = 0.6;

/// Columns `t, y, r, w`, all binary: `r, w ~ Bern(1/2)`, `t ~ Bern(0.2 + 0.3 r + 0.3 w)`,
/// `y ~ Bern(0.2 + 0.4 t)`. The law of [`collider_pag`]: `r` and `w` are independent
/// causes of `t` and reach `y` only through it, so nothing confounds `t` and `y`.
fn collider_binary_data(n: usize, seed: u64) -> TabularData {
    let mut u = uniform(seed);
    let (mut t, mut y, mut r, mut w) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        r[i] = bernoulli(&mut u, 0.5);
        w[i] = bernoulli(&mut u, 0.5);
        t[i] = bernoulli(&mut u, 0.2 + 0.3 * r[i] + 0.3 * w[i]);
        y[i] = bernoulli(&mut u, 0.2 + 0.4 * t[i]);
    }
    table(&[("t", &t), ("y", &y), ("r", &r), ("w", &w)])
}

/// Columns `t, y, m` with a latent `u ~ N(0,1)`: `t = u + 0.8 e`, `m = t + 0.7 e`,
/// `y = 2m + 1.5 u + 0.5 e`; the front-door effect of a unit of `t` is `2`.
fn front_door_linear_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut t, mut y, mut m) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let latent = g();
        t[i] = latent + 0.8 * g();
        m[i] = t[i] + 0.7 * g();
        y[i] = 2.0 * m[i] + 1.5 * latent + 0.5 * g();
    }
    table(&[("t", &t), ("y", &y), ("m", &m)])
}

/// Front-door effect of [`front_door_functional_data`]: `2 · 0.4 · (0.5 + P(u = 1))`.
const FRONT_DOOR_FUNCTIONAL_TRUTH: f64 = 0.8;

/// Columns `t, y, m` with a latent `u ~ Bern(1/2)`: `t ~ Bern(0.3 + 0.4 u)`,
/// `m = 1 + 0.4 t + e`, `y = 2m (0.5 + u) + 0.5 u + 0.6 e`. The outcome is linear in the
/// mediator within each arm.
fn front_door_functional_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (mut t, mut y, mut m) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let latent = bernoulli(&mut u, 0.5);
        t[i] = bernoulli(&mut u, 0.3 + 0.4 * latent);
        m[i] = 1.0 + 0.4 * t[i] + g();
        y[i] = 2.0 * m[i] * (0.5 + latent) + 0.5 * latent + 0.6 * g();
    }
    table(&[("t", &t), ("y", &y), ("m", &m)])
}

/// Jump of [`discontinuity_data`] at the cutoff.
const DISCONTINUITY_TRUTH: f64 = 2.5;
/// Bandwidth of the local-linear discontinuity fit.
const DISCONTINUITY_BANDWIDTH: f64 = 0.8;

/// Columns `t, y, r`: `r ~ U(-1, 1)`, `t = 1{r >= 0}`,
/// `y = 1 + 2.5 t + 0.8 r + 1.2 t r + 0.5 e`: linear on each side of the cutoff.
fn discontinuity_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (mut t, mut y, mut r) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        r[i] = 2.0 * u() - 1.0;
        t[i] = f64::from(r[i] >= 0.0);
        y[i] = 1.0 + 2.5 * t[i] + 0.8 * r[i] + 1.2 * t[i] * r[i] + 0.5 * g();
    }
    table(&[("t", &t), ("y", &y), ("r", &r)])
}

/// `(t, y, z)` of the instrument law: `z ~ Bern(1/2)`, a latent `u ~ N(0,1)`,
/// `t = z + u + 0.5 e`, `y = 2t + 0.8 u + 0.5 e`; the structural effect is `2`.
fn instrument_columns(n: usize, seed: u64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let latent = g();
        z[i] = bernoulli(&mut u, 0.5);
        t[i] = z[i] + latent + 0.5 * g();
        y[i] = 2.0 * t[i] + 0.8 * latent + 0.5 * g();
    }
    (t, y, z)
}

fn instrument_data(n: usize, seed: u64) -> TabularData {
    let (t, y, z) = instrument_columns(n, seed);
    table(&[("t", &t), ("y", &y), ("z", &z)])
}

/// The Anderson–Rubin set of [`instrument_data`] at `level`, from the same statistic the
/// estimator inverts (one instrument, an intercept as the only exogenous column).
fn instrument_anderson_rubin(n: usize, seed: u64, level: f64) -> Option<(f64, f64)> {
    let (t, y, z) = instrument_columns(n, seed);
    let intercept = vec![1.0; n];
    anderson_rubin_confidence_set(
        &IvBlocks {
            nrows: n,
            instruments_colmajor: &z,
            z_ncols: 1,
            exogenous_colmajor: &intercept,
            x_ncols: 1,
        },
        &y,
        &t,
        level,
        &FaerBackend,
        &mut LeastSquaresWorkspace::default(),
    )
    .unwrap()
    .0
    .map(|(lower, upper, _)| (lower, upper))
}

/// Columns `t, y, z`: `z ~ Bern(1/2)`, `t = z + v`, `y = 1 + 2t + v + e` with `v` the
/// shared disturbance; the structural effect is `2`.
fn joint_instrument_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (mut t, mut y, mut z) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let shared = g();
        z[i] = bernoulli(&mut u, 0.5);
        t[i] = z[i] + shared;
        y[i] = 1.0 + 2.0 * t[i] + shared + g();
    }
    table(&[("t", &t), ("y", &y), ("z", &z)])
}

/// Median shift of [`randomized_median_data`].
const MEDIAN_TRUTH: f64 = 0.5;

/// Threshold grid of the quantile posterior. It holds each arm's median (`0` and `0.5`), so
/// the piecewise-linear grid distribution function equals the arm's distribution function
/// there and the grid median is the arm median exactly.
const MEDIAN_GRID: [f64; 13] =
    [-3.0, -2.5, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0];
/// Symmetric Dirichlet concentration of every grid bin.
const MEDIAN_GRID_CONCENTRATION: f64 = 0.5;

/// Columns `t, y`: `t ~ Bern(1/2)`, `y = 0.5 t + (1 + 0.3 t) e`; each arm is symmetric
/// about its location, so the median effect is `0.5`.
fn randomized_median_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let (mut t, mut y) = (vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        t[i] = bernoulli(&mut u, 0.5);
        y[i] = MEDIAN_TRUTH * t[i] + (1.0 + 0.3 * t[i]) * g();
    }
    table(&[("t", &t), ("y", &y)])
}

/// Columns `z, u, t, y`: `z` and `u` share a common cause (`0.7 c + 0.7 e` each),
/// `t ~ Bern(sigmoid(0.6 z - 0.4 u))`, `y = 2t + z - 0.5 u + e`; the closure functional
/// of the co-determined tier `{z, u}` is the average effect `2`.
fn codetermined_effect_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut uniform01 = uniform(seed);
    let (mut z, mut u, mut t, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        let common = g();
        z[i] = 0.7 * common + 0.7 * g();
        u[i] = 0.7 * common + 0.7 * g();
        t[i] = bernoulli(&mut uniform01, sigmoid(0.6 * z[i] - 0.4 * u[i]));
        y[i] = 2.0 * t[i] + z[i] - 0.5 * u[i] + g();
    }
    table(&[("z", &z), ("u", &u), ("t", &t), ("y", &y)])
}

/// Effects of `t` on `y` under the two canonical orientations of the undecided tier
/// `{t, m}` of [`undecided_tier_data`]: `t -> m` (total effect `-1 + 2`) and `m -> t`
/// (direct coefficient `-1`).
const SCENARIO_TRUTH: [f64; 2] = [1.0, -1.0];

/// Columns `era, t, m, y`: `era ~ N(0,1)`, `t = 0.8 era + e`,
/// `m = t + 0.2 era + 0.5 e`, `y = -t + 2m + 0.2 era + e`.
fn undecided_tier_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut era, mut t, mut m, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        era[i] = g();
        t[i] = 0.8 * era[i] + g();
        m[i] = t[i] + 0.2 * era[i] + 0.5 * g();
        y[i] = -t[i] + 2.0 * m[i] + 0.2 * era[i] + g();
    }
    table(&[("era", &era), ("t", &t), ("m", &m), ("y", &y)])
}

/// Joint level of [`codetermined_response_data`] at `(t1, t2) = (1, 1)` and the sample
/// mean of `z`: `0.2 + 0.4 + 0.3`.
const CODETERMINED_LEVEL_TRUTH: f64 = 0.9;

/// Columns `z, t1, t2, y`: `z ~ N(0,1)` centred on its sample mean,
/// `t1 ~ Bern(sigmoid(0.5 z))`, `t2 ~ Bern(sigmoid(-0.4 z))`,
/// `y = 0.2 + 0.4 t1 + 0.3 t2 + 0.2 z + 0.5 e`: additive in the co-determined treatments.
fn codetermined_response_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut u = uniform(seed);
    let z = exogenous(&mut g, n, true);
    let (mut t1, mut t2, mut y) = (vec![0.0; n], vec![0.0; n], vec![0.0; n]);
    for i in 0..n {
        t1[i] = bernoulli(&mut u, sigmoid(0.5 * z[i]));
        t2[i] = bernoulli(&mut u, sigmoid(-0.4 * z[i]));
        y[i] = 0.2 + 0.4 * t1[i] + 0.3 * t2[i] + 0.2 * z[i] + 0.5 * g();
    }
    table(&[("z", &z), ("t1", &t1), ("t2", &t2), ("y", &y)])
}

// -------------------------------------------------------------------- graphs

/// `z -> t`, `z -> y`, `t -> y` over columns `t, y, z`.
const CONFOUNDED_EDGES: [(usize, usize); 3] = [(2, 0), (2, 1), (0, 1)];
/// `t -> y` over columns `t, y[, z]`.
const DIRECT_EDGES: [(usize, usize); 1] = [(0, 1)];
/// [`CONFOUNDED_EDGES`] plus the modifier `x -> y` over columns `t, y, z, x`.
const CONDITIONAL_EDGES: [(usize, usize); 4] = [(2, 0), (2, 1), (0, 1), (3, 1)];
/// `z -> t1`, `z -> t2`, `z -> y`, `t1 -> t2`, `t1 -> y`, `t2 -> y` over `t1, t2, y, z`.
const CELL_EDGES: [(usize, usize); 6] = [(3, 0), (3, 1), (3, 2), (0, 1), (0, 2), (1, 2)];
/// `x -> t`, `x -> y`, `t -> m`, `t -> y`, `m -> y` over columns `t, m, y, x`.
const MEDIATION_EDGES: [(usize, usize); 5] = [(3, 0), (3, 2), (0, 1), (0, 2), (1, 2)];
/// `t -> m -> y` over columns `t, y, m`; the latent confounder is absent from the graph.
const FRONT_DOOR_EDGES: [(usize, usize); 2] = [(0, 2), (2, 1)];
/// `r -> t`, `t -> y`, `r -> y` over columns `t, y, r`.
const DISCONTINUITY_EDGES: [(usize, usize); 3] = [(2, 0), (0, 1), (2, 1)];
/// `z -> t -> y` over columns `t, y, z`.
const INSTRUMENT_EDGES: [(usize, usize); 2] = [(2, 0), (0, 1)];
/// `r -> t <- w`, `w -> y`, `t -> y` over columns `t, y, r, w`: the collider at `t`
/// compels every edge, and `r` makes `t -> y` visible.
const COMPELLED_EDGES: [(usize, usize); 4] = [(2, 0), (3, 0), (3, 1), (0, 1)];
/// `r -> t <- w`, `t -> y` over columns `t, y, r, w`: the known causes `r` and `w` of `t`
/// are not adjacent to `y`, so `t -> y` is visible and `y` is in no district but its own.
const COLLIDER_EDGES: [(usize, usize); 3] = [(2, 0), (3, 0), (0, 1)];

/// `z -> t`, `z -> y`, `t -> y`, `w — z` over columns `t, y, z, w`.
fn agreeing_cpdag() -> Cpdag {
    let mut g = Cpdag::with_variables(4);
    g.insert_directed(d(2), d(0)).unwrap();
    g.insert_directed(d(2), d(1)).unwrap();
    g.insert_directed(d(0), d(1)).unwrap();
    g.insert_undirected(d(3), d(2)).unwrap();
    g
}

/// [`agreeing_cpdag`] with the modifier `x -> y` over columns `t, y, z, x, w`.
fn agreeing_conditional_cpdag() -> Cpdag {
    let mut g = Cpdag::with_variables(5);
    g.insert_directed(d(2), d(0)).unwrap();
    g.insert_directed(d(2), d(1)).unwrap();
    g.insert_directed(d(0), d(1)).unwrap();
    g.insert_directed(d(3), d(1)).unwrap();
    g.insert_undirected(d(4), d(2)).unwrap();
    g
}

/// `r o-> t`, `z o-> t`, `z -> y`, `t -> y` over columns `t, y, z, r`, plus the modifier
/// `x -> y` (column 4) when `modifier`: `t -> y` is visible in every MAG completion.
fn agreeing_pag(modifier: bool) -> Pag {
    let mut pag = Pag::with_variables(if modifier { 5 } else { 4 });
    let mut edges = vec![
        (3, 0, Endpoint::Circle, Endpoint::Arrow),
        (2, 0, Endpoint::Circle, Endpoint::Arrow),
        (2, 1, Endpoint::Tail, Endpoint::Arrow),
        (0, 1, Endpoint::Tail, Endpoint::Arrow),
    ];
    if modifier {
        edges.push((4, 1, Endpoint::Tail, Endpoint::Arrow));
    }
    for (a, b, at_a, at_b) in edges {
        pag.insert_marked(MarkedEdge { a: d(a), b: d(b), at_a, at_b, middle: MiddleMark::Empty })
            .unwrap();
    }
    pag
}

fn compelled_cpdag() -> Cpdag {
    let mut g = Cpdag::with_variables(4);
    for (from, to) in COMPELLED_EDGES {
        g.insert_directed(d(from as u32), d(to as u32)).unwrap();
    }
    g
}

fn decided_pag() -> Pag {
    let mut g = Pag::with_variables(4);
    for (from, to) in COMPELLED_EDGES {
        g.insert_directed(d(from as u32), d(to as u32)).unwrap();
    }
    g
}

/// The decided PAG of [`COLLIDER_EDGES`]. Its one MAG completion keeps every directed edge
/// in every ADMG candidate (dropping `r -> t` or `w -> t` would lose the ancestry the tail
/// asserts), and the only candidate edges are `r <-> t` and `w <-> t` beside the two
/// invisible edges out of the roots; `t -> y` is visible. So every candidate has the same
/// directed part and `{y}` as a district, and ID gives each one the same functional
/// `P(y | r, t, w)` with `r, w` free (line 3 adds them, line 6 factors `{y}`): one member,
/// by construction rather than by the sample.
fn collider_pag() -> Pag {
    let mut g = Pag::with_variables(4);
    for (from, to) in COLLIDER_EDGES {
        g.insert_directed(d(from as u32), d(to as u32)).unwrap();
    }
    g
}

/// `t -> m -> y` with `t <-> y` over columns `t, m, y`.
fn front_door_admg() -> Admg {
    let mut g = Admg::with_variables(3);
    g.insert_directed(d(0), d(1)).unwrap();
    g.insert_directed(d(1), d(2)).unwrap();
    g.insert_bidirected(d(0), d(2)).unwrap();
    g
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
    /// The latent-confounded graph read as a temporal-class graph of contemporaneous
    /// edges.
    TemporalAdmg(Admg),
    /// Ordered tiers of named columns; offered as an explicit structure only.
    Tiered(&'static [&'static [&'static str]], WithinTier),
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
            Self::Tiered(..) => panic!("a tiered background is offered as a given structure only"),
        };
        let posterior = GraphPosterior::from_atoms(
            nodes,
            vec![1.0],
            vec![atom],
            vec![0.0; nodes * nodes],
            vec![0.0; nodes * nodes],
            SamplingSummary {
                ess: 1.0,
                diagnostics: InferenceDiagnostics::analytic("single_atom_static"),
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

    fn attach(&self, builder: StudyBuilder, via: Via, data: &TabularData) -> StudyBuilder {
        match (self, via) {
            (Self::Tiered(tiers, within), Explicit) => {
                let tiers: Vec<Vec<&str>> = tiers.iter().map(|tier| tier.to_vec()).collect();
                let background =
                    TieredBackground::from_named(data.schema(), &tiers, *within).unwrap();
                builder.tiered_background(background).unwrap()
            }
            (Self::Tiered(..), Accepted | Posterior) => {
                panic!("a tiered background is offered as an explicit structure only")
            }
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
            (Self::TemporalAdmg(graph), Explicit) => {
                builder.graph(TemporalAdmg::from_static_contemporaneous(graph).unwrap())
            }
            (Self::TemporalAdmg(graph), Accepted) => builder.graph(AcceptedGraph::from(
                TemporalAdmg::from_static_contemporaneous(graph).unwrap(),
            )),
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

/// The conjugate posterior the static effect estimators are measured under.
fn conjugate() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS).prior_scale(10.0))
}

/// The conjugate posterior of a tiered background, under a diffuse prior.
fn diffuse_conjugate() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS).prior_scale(1_000.0))
}

/// The conjugate posterior at its default prior.
fn default_conjugate() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS))
}

/// The Laplace posterior the class responses are measured under.
fn laplace() -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(DRAWS))
}

/// Everything one coordinate executes and scores.
struct Design {
    /// The generator's name.
    dgp: &'static str,
    data: fn(usize, u64) -> TabularData,
    base_n: usize,
    structure: Structure,
    query: CausalQuery,
    inference: InferenceMode,
    /// The estimator requested, when the coordinate's default is another one.
    request: Option<EstimatorId>,
    identifier: Option<IdentifierId>,
    /// Bandwidth of a discontinuity design on column 2 with its cutoff at zero.
    discontinuity: Option<f64>,
    /// Bootstrap replicates requested; `None` leaves the builder's default.
    bootstrap: Option<u32>,
    /// `IntervalMethod::as_str()` of the scored interval.
    interval: &'static str,
    /// The estimator the plan reports.
    estimator: &'static str,
    targets: Vec<Target>,
    score: fn(&StudyResult) -> Vec<IntervalPair>,
    /// The scored interval at a level, recomputed from the replicate's rows, for an
    /// interval the result publishes at the reported level only.
    recomputed: Option<fn(usize, u64, f64) -> Option<(f64, f64)>>,
    /// One truth per target from the replicate's `(n, seed)`, for a functional averaged over
    /// the sample's covariates; `targets[..].truth` is then not read.
    sample_truth: Option<fn(usize, u64) -> Vec<f64>>,
}

impl Design {
    /// A design on a given structure with the defaults of an analytic scalar effect.
    fn new(
        dgp: &'static str,
        data: fn(usize, u64) -> TabularData,
        base_n: usize,
        structure: Structure,
        query: CausalQuery,
        mode: Mode,
        estimator: &'static str,
        truth: f64,
    ) -> Self {
        let (inference, interval, score): (_, _, fn(&StudyResult) -> Vec<IntervalPair>) = match mode
        {
            Mode::Frequentist => (InferenceMode::Frequentist, "analytic_se", score_normal),
            Mode::Bayesian => (conjugate(), "posterior_quantile", score_posterior),
        };
        Self {
            dgp,
            data,
            base_n,
            structure,
            query,
            inference,
            request: None,
            identifier: None,
            discontinuity: None,
            bootstrap: Some(0),
            interval,
            estimator,
            targets: Target::scalar(truth),
            score,
            recomputed: None,
            sample_truth: None,
        }
    }

    fn request(mut self, estimator: EstimatorId) -> Self {
        self.request = Some(estimator);
        self
    }

    fn identifier(mut self, identifier: IdentifierId) -> Self {
        self.identifier = Some(identifier);
        self
    }

    fn inference(mut self, inference: InferenceMode) -> Self {
        self.inference = inference;
        self
    }
}

fn effect_query(treatment: u32, outcome: u32) -> AverageEffectQuery {
    AverageEffectQuery::with_levels(v(treatment), v(outcome), 0.0, 1.0)
}

fn binary_effect(treatment: u32, outcome: u32) -> CausalQuery {
    CausalQuery::AverageEffect(AverageEffectQuery::binary_ate(v(treatment), v(outcome)))
}

/// The estimator a mode runs when none is requested.
fn default_effect_estimator(mode: Mode) -> &'static str {
    match mode {
        Mode::Frequentist => "linear.adjustment.ate",
        Mode::Bayesian => "bayesian.gcomp",
    }
}

/// The average effect of a binary treatment on the class's confounded law, by the
/// mode's default estimator or the requested one.
fn average_effect(
    class: Class,
    mode: Mode,
    request: Option<(EstimatorId, &'static str)>,
) -> Design {
    let (dgp, data, structure): (_, fn(usize, u64) -> TabularData, _) = match class {
        Class::Dag => (
            "linear_ate_data",
            linear_ate_data,
            Structure::Dag { nodes: 3, edges: &CONFOUNDED_EDGES },
        ),
        Class::Cpdag => ("cpdag_ate_data", cpdag_ate_data, Structure::Cpdag(agreeing_cpdag())),
        Class::Pag => ("pag_ate_data", pag_ate_data, Structure::MarkedPag(agreeing_pag(false))),
        Class::Admg | Class::TemporalAdmg => panic!("no average-effect design on {class:?}"),
    };
    let estimator = request.map_or(default_effect_estimator(mode), |(_, name)| name);
    let design = Design::new(dgp, data, 500, structure, binary_effect(0, 1), mode, estimator, 2.0);
    match request {
        Some((id, _)) => design.request(id),
        None => design,
    }
}

/// The class structure with a single member: the compelled CPDAG or the fully decided PAG
/// over [`compelled_ate_data`]'s columns.
fn single_member_structure(class: Class) -> Structure {
    match class {
        Class::Cpdag => Structure::Cpdag(compelled_cpdag()),
        Class::Pag => Structure::DirectedPag(decided_pag(), &COMPELLED_EDGES),
        _ => panic!("no single-member class design on {class:?}"),
    }
}

/// The average effect on a one-member CPDAG or PAG by an estimator whose estimate carries no
/// influence function (propensity weighting, matching, stratification, distance matching).
///
/// A class with several members is a mixture of member fits; without per-member influence
/// functions the runtime cannot form the mixture's sampling covariance and publishes no
/// standard error (`atom_mix.rs`, `mix_weighted_analytic_se` refuses more than one atom).
/// A one-member class publishes the member's own interval, which this measures.
fn single_member_effect(class: Class, request: (EstimatorId, &'static str)) -> Design {
    Design::new(
        "compelled_ate_data",
        compelled_ate_data,
        500,
        single_member_structure(class),
        binary_effect(0, 1),
        Mode::Frequentist,
        request.1,
        2.0,
    )
    .request(request.0)
}

/// The logistic GLM adjustment of a binary outcome: the confounded binary-outcome law on a
/// DAG, its compelled twin on a one-member CPDAG or PAG (the GLM estimate carries no
/// influence function either). The truth is the risk difference averaged over the sample's
/// confounder values.
fn glm_adjustment(class: Class) -> Design {
    let (dgp, data, structure, truth): (
        _,
        fn(usize, u64) -> TabularData,
        _,
        fn(usize, u64) -> Vec<f64>,
    ) = match class {
        Class::Dag => (
            "binary_outcome_data",
            binary_outcome_data,
            Structure::Dag { nodes: 3, edges: &CONFOUNDED_EDGES },
            binary_outcome_truth,
        ),
        Class::Cpdag | Class::Pag => (
            "compelled_binary_outcome_data",
            compelled_binary_outcome_data,
            single_member_structure(class),
            compelled_binary_outcome_truth,
        ),
        Class::Admg | Class::TemporalAdmg => panic!("no GLM adjustment design on {class:?}"),
    };
    let mut design = Design::new(
        dgp,
        data,
        500,
        structure,
        binary_effect(0, 1),
        Mode::Frequentist,
        "glm.adjustment",
        0.0,
    )
    .request(EstimatorId::GlmAdjustment);
    design.sample_truth = Some(truth);
    design
}

/// The finite-discrete plug-in `functional.effect` on the binary confounded law: its
/// bootstrap interval (builder's default count) or its Dirichlet posterior.
fn functional_effect(mode: Mode) -> Design {
    let mut design = Design::new(
        "discrete_effect_data",
        discrete_effect_data,
        600,
        Structure::Dag { nodes: 3, edges: &CONFOUNDED_EDGES },
        binary_effect(0, 1),
        mode,
        "functional.effect",
        DISCRETE_EFFECT_TRUTH,
    )
    .identifier(IdentifierId::GeneralId)
    .request(EstimatorId::FunctionalEffect);
    match mode {
        Mode::Frequentist => {
            design.bootstrap = None;
            design.interval = "bootstrap_se";
        }
        Mode::Bayesian => design.inference = default_conjugate(),
    }
    design
}

/// The linear front-door effect by the two-stage estimator.
///
/// The route fits the two-stage estimator at its own default of 200 bootstrap replicates
/// whatever the builder's count, so the published interval is the bootstrap one.
fn front_door_two_stage() -> Design {
    let mut design = Design::new(
        "front_door_linear_data",
        front_door_linear_data,
        400,
        Structure::Dag { nodes: 3, edges: &FRONT_DOOR_EDGES },
        CausalQuery::AverageEffect(effect_query(0, 1)),
        Mode::Frequentist,
        "frontdoor.linear_two_stage",
        2.0,
    )
    .identifier(IdentifierId::Frontdoor)
    .request(EstimatorId::FrontDoorTwoStage);
    design.interval = "bootstrap_se";
    design
}

/// The front-door functional of a binary treatment with an arm-linear outcome.
fn front_door_functional() -> Design {
    Design::new(
        "front_door_functional_data",
        front_door_functional_data,
        600,
        Structure::Dag { nodes: 3, edges: &FRONT_DOOR_EDGES },
        CausalQuery::AverageEffect(effect_query(0, 1)),
        Mode::Frequentist,
        "frontdoor.functional",
        FRONT_DOOR_FUNCTIONAL_TRUTH,
    )
    .identifier(IdentifierId::Frontdoor)
    .request(EstimatorId::FrontDoorFunctional)
}

/// The sharp discontinuity, Frequentist local-linear or its Bayesian counterpart.
fn discontinuity(mode: Mode) -> Design {
    let (request, estimator) = match mode {
        Mode::Frequentist => (EstimatorId::RdSharp, "rd.sharp"),
        Mode::Bayesian => (EstimatorId::BayesianRdLocalLinear, "rd.bayesian_local_linear"),
    };
    let mut design = Design::new(
        "discontinuity_data",
        discontinuity_data,
        400,
        Structure::Dag { nodes: 3, edges: &DISCONTINUITY_EDGES },
        binary_effect(0, 1),
        mode,
        estimator,
        DISCONTINUITY_TRUTH,
    )
    .identifier(IdentifierId::RdSharp)
    .request(request);
    design.discontinuity = Some(DISCONTINUITY_BANDWIDTH);
    if mode == Mode::Bayesian {
        design.inference = default_conjugate();
    }
    design
}

/// The instrumented effect with its Anderson–Rubin set.
fn instrument(request: EstimatorId, estimator: &'static str) -> Design {
    let mut design = Design::new(
        "instrument_data",
        instrument_data,
        300,
        Structure::Dag { nodes: 3, edges: &INSTRUMENT_EDGES },
        CausalQuery::AverageEffect(effect_query(0, 1)),
        Mode::Frequentist,
        estimator,
        2.0,
    )
    .identifier(IdentifierId::Iv)
    .request(request);
    design.interval = "anderson_rubin";
    design.score = score_anderson_rubin;
    design.recomputed = Some(instrument_anderson_rubin);
    design
}

/// The joint-linear Bayesian instrument posterior.
fn joint_instrument() -> Design {
    Design::new(
        "joint_instrument_data",
        joint_instrument_data,
        180,
        Structure::Dag { nodes: 3, edges: &INSTRUMENT_EDGES },
        CausalQuery::AverageEffect(effect_query(0, 1)),
        Mode::Bayesian,
        "iv.bayesian_joint_linear",
        2.0,
    )
    .identifier(IdentifierId::Iv)
    .request(EstimatorId::BayesianIvJointLinear)
    .inference(default_conjugate())
}

/// The randomized median effect.
fn randomized_median(mode: Mode) -> Design {
    let mut query = AverageEffectQuery::binary_ate(v(0), v(1));
    query.outcome_functional = OutcomeFunctional::quantile(0.5);
    let (request, estimator) = match mode {
        Mode::Frequentist => (EstimatorId::RandomizedQuantile, "randomized.quantile"),
        Mode::Bayesian => (EstimatorId::BayesianRandomizedQuantile, "bayesian.randomized_quantile"),
    };
    let design = Design::new(
        "randomized_median_data",
        randomized_median_data,
        400,
        Structure::Dag { nodes: 2, edges: &DIRECT_EDGES },
        CausalQuery::AverageEffect(query),
        mode,
        estimator,
        MEDIAN_TRUTH,
    )
    .request(request);
    match mode {
        Mode::Frequentist => design,
        // The quantile posterior is the Dirichlet-multinomial law of each arm on a declared
        // threshold grid.
        Mode::Bayesian => design.inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(DRAWS).design_model(
                DesignPosteriorModel::QuantileGrid {
                    thresholds: Arc::from(MEDIAN_GRID),
                    bin_concentration: MEDIAN_GRID_CONCENTRATION,
                },
            ),
        )),
    }
}

/// The closure functional of a co-determined tier.
fn codetermined_effect() -> Design {
    Design::new(
        "codetermined_effect_data",
        codetermined_effect_data,
        600,
        Structure::Tiered(&[&["z", "u"], &["t"], &["y"]], WithinTier::CoDetermined),
        binary_effect(2, 3),
        Mode::Bayesian,
        "bayesian.gcomp",
        2.0,
    )
    .inference(diffuse_conjugate())
}

/// The canonical orientation scenarios of an undecided tier, each with its own interval.
fn undecided_tier_effect() -> Design {
    let mut design = Design::new(
        "undecided_tier_data",
        undecided_tier_data,
        400,
        Structure::Tiered(&[&["era"], &["t", "m"], &["y"]], WithinTier::Unknown),
        binary_effect(1, 3),
        Mode::Bayesian,
        "bayesian.gcomp",
        0.0,
    )
    .inference(diffuse_conjugate());
    design.interval = "simultaneous_band";
    design.score = score_scenarios;
    design.targets = SCENARIO_TRUTH
        .iter()
        .enumerate()
        .map(|(j, truth)| Target::labelled(format!("scenario={j}"), *truth))
        .collect();
    design
}

/// The modifier-averaged conditional effect on the class's law.
fn conditional_effect(class: Class, mode: Mode) -> Design {
    let (dgp, data, structure, modifier, base): (_, fn(usize, u64) -> TabularData, _, _, _) =
        match class {
            Class::Dag => (
                "conditional_dag_data",
                conditional_dag_data,
                Structure::Dag { nodes: 4, edges: &CONDITIONAL_EDGES },
                3,
                1.0,
            ),
            Class::Cpdag => (
                "conditional_cpdag_data",
                conditional_cpdag_data,
                Structure::Cpdag(agreeing_conditional_cpdag()),
                3,
                1.0,
            ),
            Class::Pag => (
                "conditional_pag_data",
                conditional_pag_data,
                Structure::MarkedPag(agreeing_pag(true)),
                4,
                2.0,
            ),
            Class::Admg | Class::TemporalAdmg => panic!("no conditional design on {class:?}"),
        };
    let query = CausalQuery::ConditionalEffect(
        ConditionalEffectQuery::try_new(effect_query(0, 1).with_effect_modifiers([v(modifier)]))
            .unwrap(),
    );
    let estimator = match mode {
        Mode::Frequentist => "conditional.linear.adjustment",
        Mode::Bayesian => "conditional.bayesian",
    };
    Design::new(
        dgp,
        data,
        400,
        structure,
        query,
        mode,
        estimator,
        base + MODIFIER_SLOPE * MODIFIER_SHARE,
    )
}

/// The conditional effect by the basis g-computation posterior, which needs a binary
/// treatment: the binary-treatment twin of [`conditional_dag_data`], scored against the
/// modifier-averaged effect over the sample's modifier values.
fn conditional_basis() -> Design {
    let query = CausalQuery::ConditionalEffect(
        ConditionalEffectQuery::try_new(effect_query(0, 1).with_effect_modifiers([v(3)])).unwrap(),
    );
    let mut design = Design::new(
        "binary_conditional_data",
        binary_conditional_data,
        400,
        Structure::Dag { nodes: 4, edges: &CONDITIONAL_EDGES },
        query,
        Mode::Bayesian,
        "bayesian.basis.gcomp",
        0.0,
    )
    .request(EstimatorId::BayesianBasisGcomp);
    design.sample_truth = Some(binary_conditional_truth);
    design
}

/// The response law of a class: its generators (population and sample-centred), its
/// structure and its structural mean.
fn response_law(
    class: Class,
    posterior: bool,
    mode: Mode,
) -> (&'static str, fn(usize, u64) -> TabularData, Structure, fn(f64) -> f64) {
    let bayesian = mode == Mode::Bayesian;
    match (class, bayesian) {
        (Class::Dag, false) => (
            "response_data",
            response_data,
            Structure::Dag { nodes: 3, edges: &CONFOUNDED_EDGES },
            |a| 1.0 + 2.0 * a,
        ),
        (Class::Dag, true) if posterior => (
            "independent_response_data",
            independent_response_data,
            Structure::Dag { nodes: 3, edges: &DIRECT_EDGES },
            |a| 1.0 + 2.0 * a,
        ),
        (Class::Dag, true) => (
            "centred_response_data",
            centred_response_data,
            Structure::Dag { nodes: 3, edges: &CONFOUNDED_EDGES },
            |a| 1.0 + 2.0 * a,
        ),
        (Class::Cpdag, false) => {
            ("cpdag_response_data", cpdag_response_data, Structure::Cpdag(agreeing_cpdag()), |a| a)
        }
        (Class::Cpdag, true) => (
            "centred_cpdag_response_data",
            centred_cpdag_response_data,
            Structure::Cpdag(agreeing_cpdag()),
            |a| a,
        ),
        (Class::Pag, false) => (
            "pag_response_data",
            pag_response_data,
            Structure::MarkedPag(agreeing_pag(false)),
            |a| 1.0 + 2.0 * a,
        ),
        (Class::Pag, true) => (
            "centred_pag_response_data",
            centred_pag_response_data,
            Structure::MarkedPag(agreeing_pag(false)),
            |a| 1.0 + 2.0 * a,
        ),
        (Class::Admg | Class::TemporalAdmg, _) => panic!("no response design on {class:?}"),
    }
}

/// The Bayesian response posterior of a class: conjugate on a DAG, Laplace on a class.
fn response_inference(class: Class, mode: Mode) -> InferenceMode {
    match (mode, class) {
        (Mode::Frequentist, _) => InferenceMode::Frequentist,
        (Mode::Bayesian, Class::Dag) => conjugate(),
        (Mode::Bayesian, _) => laplace(),
    }
}

/// The intervention response at [`LEVEL`].
fn intervention_response(class: Class, mode: Mode, posterior: bool) -> Design {
    let (dgp, data, structure, truth) = response_law(class, posterior, mode);
    let query =
        CausalQuery::Response(ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: v(1),
            interventions: Arc::from([Intervention::set(v(0), Value::f64(LEVEL))]),
        }));
    let (estimator, score): (_, fn(&StudyResult) -> Vec<IntervalPair>) = match mode {
        Mode::Frequentist => ("response.intervention_gcomp", score_response_normal),
        Mode::Bayesian => ("response.bayesian", score_response_level_posterior),
    };
    let mut design = Design::new(dgp, data, 500, structure, query, mode, estimator, truth(LEVEL));
    design.inference = response_inference(class, mode);
    design.score = score;
    design
}

/// The joint level of two treatments by cell AIPW.
fn cell_response() -> Design {
    let query =
        CausalQuery::Response(ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: v(2),
            interventions: Arc::from([
                Intervention::set(v(0), Value::f64(1.0)),
                Intervention::set(v(1), Value::f64(1.0)),
            ]),
        }));
    let mut design = Design::new(
        "cell_data",
        cell_data,
        800,
        Structure::Dag { nodes: 4, edges: &CELL_EDGES },
        query,
        Mode::Frequentist,
        "cell.aipw",
        CELL_TRUTH,
    )
    .request(EstimatorId::CellAipw);
    design.score = score_response_normal;
    design
}

/// The joint level of two co-determined treatments.
fn codetermined_response() -> Design {
    let query =
        CausalQuery::Response(ResponseQuery::new(ResponseFunctional::InterventionResponse {
            outcome: v(3),
            interventions: Arc::from([
                Intervention::set(v(1), Value::f64(1.0)),
                Intervention::set(v(2), Value::f64(1.0)),
            ]),
        }));
    let mut design = Design::new(
        "codetermined_response_data",
        codetermined_response_data,
        500,
        Structure::Tiered(&[&["z"], &["t1", "t2"], &["y"]], WithinTier::CoDetermined),
        query,
        Mode::Bayesian,
        "response.bayesian",
        CODETERMINED_LEVEL_TRUTH,
    )
    .inference(default_conjugate());
    design.score = score_response_level_posterior;
    design
}

/// The Bayesian mean curve over [`CURVE_GRID`].
fn mean_curve(class: Class, posterior: bool) -> Design {
    let (dgp, data, structure, truth) = response_law(class, posterior, Mode::Bayesian);
    let query = CausalQuery::Response(ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: v(1),
        treatment: ContinuousDomain::new(v(0), GridSpec::Values(CURVE_GRID.to_vec().into())),
    }));
    let mut design =
        Design::new(dgp, data, 500, structure, query, Mode::Bayesian, "response.bayesian", 0.0);
    design.inference = response_inference(class, Mode::Bayesian);
    design.score = score_response_curve_posterior;
    design.targets =
        CURVE_GRID.iter().map(|&a| Target::labelled(format!("a={a}"), truth(a))).collect();
    design
}

/// `P(Y = 1 | do(t = 1))` on the class's binary law.
fn distribution(class: Class, mode: Mode) -> Design {
    let (dgp, data, base_n, structure, outcome, truth): (
        _,
        fn(usize, u64) -> TabularData,
        _,
        _,
        _,
        _,
    ) = match class {
        Class::Dag => (
            "distribution_data",
            dag_distribution_data,
            600,
            Structure::Dag { nodes: 3, edges: &CONFOUNDED_EDGES },
            1,
            DAG_DISTRIBUTION_TRUTH,
        ),
        Class::Cpdag => (
            "compelled_binary_data",
            compelled_binary_data,
            600,
            Structure::Cpdag(compelled_cpdag()),
            1,
            COMPELLED_DISTRIBUTION_TRUTH,
        ),
        Class::Pag => (
            "collider_binary_data",
            collider_binary_data,
            600,
            Structure::DirectedPag(collider_pag(), &COLLIDER_EDGES),
            1,
            COLLIDER_DISTRIBUTION_TRUTH,
        ),
        Class::Admg => (
            "front_door_binary_data",
            front_door_binary_data,
            1000,
            Structure::Admg(front_door_admg()),
            2,
            FRONT_DOOR_DISTRIBUTION_TRUTH,
        ),
        Class::TemporalAdmg => (
            "front_door_binary_data",
            front_door_binary_data,
            1000,
            Structure::TemporalAdmg(front_door_admg()),
            2,
            FRONT_DOOR_DISTRIBUTION_TRUTH,
        ),
    };
    let query = CausalQuery::Distribution(InterventionalDistributionQuery::new(
        v(outcome),
        [Intervention::set(v(0), Value::f64(1.0))],
    ));
    let mut design =
        Design::new(dgp, data, base_n, structure, query, mode, "functional.distribution", truth);
    match mode {
        Mode::Frequentist => {
            // The probability interval is the bootstrap's; the builder's default count.
            design.bootstrap = None;
            design.interval = "bootstrap_se";
            design.score = score_distribution_bootstrap;
        }
        Mode::Bayesian => design.score = score_distribution_posterior,
    }
    if matches!(class, Class::Cpdag | Class::Pag | Class::TemporalAdmg) {
        design.identifier = Some(IdentifierId::GeneralId);
        design.request = Some(EstimatorId::FunctionalDistribution);
    }
    design
}

/// The natural indirect effect of the linear mediation law.
fn mediation(mode: Mode) -> Design {
    let query = CausalQuery::Mediation(MediationQuery::binary(
        v(0),
        v(2),
        Arc::from([v(1)]),
        MediationContrast::NaturalIndirect,
    ));
    let mut design = Design::new(
        "mediation_data",
        mediation_data,
        400,
        Structure::Dag { nodes: 4, edges: &MEDIATION_EDGES },
        query,
        mode,
        "mediation.linear",
        0.3,
    );
    if mode == Mode::Frequentist {
        // The product of coefficients publishes a bootstrap standard error.
        design.bootstrap = Some(StudyBuilder::OMITTED_BOOTSTRAP);
        design.interval = "bootstrap_se";
    }
    design
}

// ------------------------------------------------------------------- scoring

fn score_normal(result: &StudyResult) -> Vec<IntervalPair> {
    vec![scalar_normal_pair(result)]
}

fn score_posterior(result: &StudyResult) -> Vec<IntervalPair> {
    let column = result.posterior.as_ref().and_then(|p| p.effect_column()).unwrap_or(0);
    vec![posterior_pair(result, column)]
}

fn score_response_normal(result: &StudyResult) -> Vec<IntervalPair> {
    vec![response_normal_pair(result)]
}

fn score_response_level_posterior(result: &StudyResult) -> Vec<IntervalPair> {
    response_posterior_pairs(result).unwrap_or_else(|| vec![[None, None]])
}

fn score_response_curve_posterior(result: &StudyResult) -> Vec<IntervalPair> {
    response_posterior_pairs(result).unwrap_or_else(|| vec![[None, None]; CURVE_GRID.len()])
}

fn score_distribution_bootstrap(result: &StudyResult) -> Vec<IntervalPair> {
    let Some(distribution) = result.distribution() else {
        return vec![[None, None]];
    };
    vec![[
        distribution.mean_interval.and_then(|interval| interval.bounds()),
        logit_probability_interval(distribution.mean, distribution.se_bootstrap, GATE_LEVEL)
            .bounds(),
    ]]
}

/// The posterior of the `Y = 1` atom: the atoms follow the effect column when the
/// posterior carries one.
fn score_distribution_posterior(result: &StudyResult) -> Vec<IntervalPair> {
    let (Some(distribution), Some(posterior)) = (result.distribution(), result.posterior.as_ref())
    else {
        return vec![[None, None]];
    };
    let offset = usize::from(posterior.effect_column().is_some());
    let atom = distribution
        .atoms
        .iter()
        .position(|a| a.outcomes[0].1.as_f64() == Some(1.0))
        .expect("Y = 1 atom");
    vec![posterior_pair(result, offset + atom)]
}

/// The nearest-rank quantile the scenario intervals are published with.
fn nearest_rank(sorted: &[f64], q: f64) -> f64 {
    let last = sorted.len().saturating_sub(1);
    let position = (last as f64 * q).round() as usize;
    sorted.get(position.min(last)).copied().unwrap_or(f64::NAN)
}

/// One credible interval per canonical scenario: the published one, and the same rule
/// at the gate level from the scenario's retained posterior.
fn score_scenarios(result: &StudyResult) -> Vec<IntervalPair> {
    let published =
        result.estimate().as_effect().and_then(|effect| effect.scenarios.intervals.clone());
    let atoms = result.structural_response.as_ref().map(|mixture| &mixture.atoms);
    (0..SCENARIO_TRUTH.len())
        .map(|j| {
            let reported = published.as_ref().and_then(|intervals| intervals.get(j).copied());
            let gate = atoms
                .and_then(|atoms| atoms.get(j))
                .and_then(|atom| atom.posterior.as_ref())
                .and_then(|posterior| {
                    let column = posterior.effect_column()?;
                    let mut draws = posterior.draws.column(column).ok()?.to_vec();
                    draws.sort_by(f64::total_cmp);
                    assert_eq!(
                        Some((nearest_rank(&draws, 0.025), nearest_rank(&draws, 0.975))),
                        reported,
                        "scenario {j}: the published interval is not the nearest-rank \
                         quantiles of the retained draws"
                    );
                    let tail = 0.5 * (1.0 - GATE_LEVEL);
                    Some((nearest_rank(&draws, tail), nearest_rank(&draws, 1.0 - tail)))
                });
            [reported, gate]
        })
        .collect()
}

/// The Anderson–Rubin set a result publishes.
fn published_anderson_rubin(result: &StudyResult) -> Option<(f64, f64)> {
    result
        .estimate()
        .as_effect()?
        .diagnostics
        .first_stage
        .as_ref()?
        .anderson_rubin
        .map(|(lower, upper, _)| (lower, upper))
}

type Recomputed = Mutex<HashMap<[u64; 2], Option<(f64, f64)>>>;

/// Gate-level intervals recomputed from a replicate's rows, keyed by the bits of the
/// reported interval of the same rows.
fn recomputed_intervals() -> &'static Recomputed {
    static CELL: OnceLock<Recomputed> = OnceLock::new();
    CELL.get_or_init(|| Mutex::new(HashMap::new()))
}

fn score_anderson_rubin(result: &StudyResult) -> Vec<IntervalPair> {
    let reported = published_anderson_rubin(result);
    let gate = reported.and_then(|(lower, upper)| {
        recomputed_intervals()
            .lock()
            .unwrap()
            .get(&[lower.to_bits(), upper.to_bits()])
            .copied()
            .flatten()
    });
    vec![[reported, gate]]
}

// ----------------------------------------------------------------- execution

fn execute(
    design: &Design,
    (via, suite): Variant,
    data: &TabularData,
    seed: u64,
) -> (Study, StudyResult) {
    let mut builder = design
        .structure
        .attach(Study::tabular(data.clone()), via, data)
        .query(design.query.clone())
        .inference(design.inference.clone())
        .refute(suite);
    if let Some(identifier) = design.identifier {
        builder = builder.identifier(identifier);
    }
    if let Some(estimator) = design.request {
        builder = builder.estimator(estimator);
    }
    if let Some(bandwidth) = design.discontinuity {
        builder = builder.rd_config(v(2), 0.0, bandwidth);
    }
    if let Some(replicates) = design.bootstrap {
        builder = builder.bootstrap_replicates(replicates);
    }
    let study = builder.build().unwrap_or_else(|e| panic!("{} {via:?} {suite:?}: {e}", design.dgp));
    let result = study
        .run(&ExecutionContext::for_tests(seed))
        .unwrap_or_else(|e| panic!("{} {via:?} {suite:?}: {e}", design.dgp));
    (study, result)
}

/// Measure one coordinate: the first variant on every replicate, its siblings on the
/// driver's sibling replicates.
fn measure_coordinate(test: &'static str, family: u64, design: &Design, variants: &[Variant]) {
    let cell = Cell {
        test,
        dgp: design.dgp,
        interval: design.interval,
        estimator: design.estimator,
        variants: variants.len(),
    };
    let run = |rep: u64, variant: usize| {
        let seed = stream_seed(family, rep);
        let n = grid_n(design.base_n);
        let data = (design.data)(n, seed);
        let (study, result) = execute(design, variants[variant], &data, seed);
        if rep == 0 && design.estimator == "functional.distribution" {
            // A class with one identified member publishes that member's interval; the
            // members' envelope carries none, so falling into it fails here by name.
            assert!(
                result
                    .diagnostics
                    .iter()
                    .all(|d| d.code.as_ref() != "estimate.distribution.class_envelope"),
                "{test}: the class distribution published the members' envelope"
            );
        }
        if variant == 0
            && let Some(recompute) = design.recomputed
            && let Some((lower, upper)) = published_anderson_rubin(&result)
        {
            // The level enters only the critical value, so the same statistic on the
            // same rows at the gate level is this execution's gate-level set.
            let rebuilt = recompute(n, seed, REPORTED_LEVEL);
            assert!(
                rebuilt.is_some_and(|(lo, hi)| {
                    (lo - lower).abs() <= 1e-9 * lower.abs().max(1.0)
                        && (hi - upper).abs() <= 1e-9 * upper.abs().max(1.0)
                }),
                "{test}: the recomputed set {rebuilt:?} is not the published \
                 [{lower}, {upper}]"
            );
            recomputed_intervals()
                .lock()
                .unwrap()
                .insert([lower.to_bits(), upper.to_bits()], recompute(n, seed, GATE_LEVEL));
        }
        (study, result)
    };
    match design.sample_truth {
        Some(truth) => measure_sample_truths(
            cell,
            &design.targets,
            |rep| truth(grid_n(design.base_n), stream_seed(family, rep)),
            run,
            design.score,
        ),
        None => measure(cell, &design.targets, run, design.score),
    }
}

macro_rules! coverage_tests {
    ($(
        $name:ident: $family:literal, $design:expr,
        [$(($via:ident, $suite:ident)),+ $(,)?];
    )+) => {$(
        #[test]
        #[ignore = "calibration: run via scripts/measure_calibration.sh"]
        fn $name() {
            measure_coordinate(
                stringify!($name),
                $family,
                &$design,
                &[$(($via, RefuteSuite::$suite)),+],
            );
        }
    )+};
}

use Class::TemporalAdmg as OnTemporalAdmg;
use Class::{Admg as OnAdmg, Cpdag as OnCpdag, Dag as OnDag, Pag as OnPag};
use Mode::{Bayesian, Frequentist};

/// A requested estimator and the name the plan reports for it.
const fn by(id: EstimatorId, name: &'static str) -> Option<(EstimatorId, &'static str)> {
    Some((id, name))
}

coverage_tests! {
    intervention_response_dag_frequentist_graph_posterior: 0x57A7_0001, intervention_response(OnDag, Frequentist, true),
        [(Posterior, None), (Posterior, Cheap)];
    average_effect_dag_frequentist_given_primary: 0x57A7_0002, average_effect(OnDag, Frequentist, None),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_dag_frequentist_given_aipw: 0x57A7_0003, average_effect(OnDag, Frequentist, by(EstimatorId::Aipw, "aipw")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_dag_frequentist_given_frontdoor_linear_two_stage: 0x57A7_0004, front_door_two_stage(),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_dag_frequentist_given_glm_adjustment: 0x57A7_0005, glm_adjustment(OnDag),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_dag_frequentist_given_iv_2sls: 0x57A7_0006, instrument(EstimatorId::Iv2Sls, "iv.2sls"),
        [(Accepted, None), (Explicit, Cheap), (Explicit, Full), (Accepted, Cheap), (Accepted, Full)];
    average_effect_dag_frequentist_given_iv_wald: 0x57A7_0007, instrument(EstimatorId::IvWald, "iv.wald"),
        [(Accepted, None), (Explicit, Cheap), (Explicit, Full), (Accepted, Cheap), (Accepted, Full)];
    average_effect_dag_frequentist_given_propensity_matching: 0x57A7_0008, average_effect(OnDag, Frequentist, by(EstimatorId::PropensityMatching, "propensity.matching")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_dag_frequentist_given_propensity_weighting: 0x57A7_0009, average_effect(OnDag, Frequentist, by(EstimatorId::PropensityWeighting, "propensity.weighting")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_dag_frequentist_given_rd_sharp: 0x57A7_000A, discontinuity(Frequentist),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_dag_frequentist_given_dml: 0x57A7_000B, average_effect(OnDag, Frequentist, by(EstimatorId::Dml, "dml")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_dag_frequentist_given_dr_learner: 0x57A7_000C, average_effect(OnDag, Frequentist, by(EstimatorId::DrLearner, "dr.learner")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_dag_frequentist_given_causal_forest: 0x57A7_000D, average_effect(OnDag, Frequentist, by(EstimatorId::CausalForest, "causal.forest")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_dag_frequentist_given_propensity_stratification: 0x57A7_000E, average_effect(OnDag, Frequentist, by(EstimatorId::PropensityStratification, "propensity.stratification")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_dag_frequentist_given_distance_matching: 0x57A7_000F, average_effect(OnDag, Frequentist, by(EstimatorId::DistanceMatching, "distance.matching")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_dag_frequentist_given_frontdoor_functional: 0x57A7_0010, front_door_functional(),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_dag_frequentist_given_randomized_quantile: 0x57A7_0011, randomized_median(Frequentist),
        [(Accepted, None)];
    conditional_effect_dag_frequentist_given: 0x57A7_0012, conditional_effect(OnDag, Frequentist),
        [(Accepted, None), (Explicit, Cheap), (Explicit, Full), (Accepted, Cheap), (Accepted, Full)];
    intervention_response_dag_frequentist_given_primary: 0x57A7_0013, intervention_response(OnDag, Frequentist, false),
        [(Accepted, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full), (Accepted, Full)];
    intervention_response_dag_frequentist_given_cell_aipw: 0x57A7_0014, cell_response(),
        [(Accepted, None), (Explicit, Cheap), (Accepted, Cheap), (Explicit, Full), (Accepted, Full)];
    average_effect_dag_bayesian_given_primary: 0x57A7_0015, average_effect(OnDag, Bayesian, None),
        [(Accepted, None)];
    average_effect_dag_bayesian_given_bayesian_robust_ate: 0x57A7_0016, average_effect(OnDag, Bayesian, by(EstimatorId::BayesianRobustAte, "bayesian.robust_ate")),
        [(Accepted, None)];
    average_effect_dag_bayesian_given_bayesian_basis_gcomp: 0x57A7_0017, average_effect(OnDag, Bayesian, by(EstimatorId::BayesianBasisGcomp, "bayesian.basis.gcomp")),
        [(Accepted, None)];
    average_effect_dag_bayesian_given_bayesian_randomized_quantile: 0x57A7_0018, randomized_median(Bayesian),
        [(Accepted, None)];
    average_effect_dag_bayesian_given_rd_bayesian_local_linear: 0x57A7_0019, discontinuity(Bayesian),
        [(Accepted, None)];
    average_effect_dag_bayesian_given_iv_bayesian_joint_linear: 0x57A7_001A, joint_instrument(),
        [(Accepted, None)];
    average_effect_pag_frequentist_given_primary: 0x57A7_001B, average_effect(OnPag, Frequentist, None),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_pag_frequentist_given_aipw: 0x57A7_001C, average_effect(OnPag, Frequentist, by(EstimatorId::Aipw, "aipw")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_pag_frequentist_given_dml: 0x57A7_001D, average_effect(OnPag, Frequentist, by(EstimatorId::Dml, "dml")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_pag_frequentist_given_dr_learner: 0x57A7_001E, average_effect(OnPag, Frequentist, by(EstimatorId::DrLearner, "dr.learner")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_pag_frequentist_given_causal_forest: 0x57A7_001F, average_effect(OnPag, Frequentist, by(EstimatorId::CausalForest, "causal.forest")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_pag_frequentist_given_glm_adjustment: 0x57A7_0020, glm_adjustment(OnPag),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_pag_frequentist_given_propensity_weighting: 0x57A7_0021, single_member_effect(OnPag, (EstimatorId::PropensityWeighting, "propensity.weighting")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_pag_frequentist_given_propensity_matching: 0x57A7_0022, single_member_effect(OnPag, (EstimatorId::PropensityMatching, "propensity.matching")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_pag_frequentist_given_propensity_stratification: 0x57A7_0023, single_member_effect(OnPag, (EstimatorId::PropensityStratification, "propensity.stratification")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_pag_frequentist_given_distance_matching: 0x57A7_0024, single_member_effect(OnPag, (EstimatorId::DistanceMatching, "distance.matching")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_pag_bayesian_given: 0x57A7_0025, average_effect(OnPag, Bayesian, None),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_cpdag_frequentist_given_primary: 0x57A7_0026, average_effect(OnCpdag, Frequentist, None),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_cpdag_frequentist_given_aipw: 0x57A7_0027, average_effect(OnCpdag, Frequentist, by(EstimatorId::Aipw, "aipw")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_cpdag_frequentist_given_dml: 0x57A7_0028, average_effect(OnCpdag, Frequentist, by(EstimatorId::Dml, "dml")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_cpdag_frequentist_given_dr_learner: 0x57A7_0029, average_effect(OnCpdag, Frequentist, by(EstimatorId::DrLearner, "dr.learner")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_cpdag_frequentist_given_causal_forest: 0x57A7_002A, average_effect(OnCpdag, Frequentist, by(EstimatorId::CausalForest, "causal.forest")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_cpdag_frequentist_given_glm_adjustment: 0x57A7_002B, glm_adjustment(OnCpdag),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_cpdag_frequentist_given_propensity_weighting: 0x57A7_002C, single_member_effect(OnCpdag, (EstimatorId::PropensityWeighting, "propensity.weighting")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_cpdag_frequentist_given_propensity_matching: 0x57A7_002D, single_member_effect(OnCpdag, (EstimatorId::PropensityMatching, "propensity.matching")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_cpdag_frequentist_given_propensity_stratification: 0x57A7_002E, single_member_effect(OnCpdag, (EstimatorId::PropensityStratification, "propensity.stratification")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_cpdag_frequentist_given_distance_matching: 0x57A7_002F, single_member_effect(OnCpdag, (EstimatorId::DistanceMatching, "distance.matching")),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_cpdag_bayesian_given: 0x57A7_0030, average_effect(OnCpdag, Bayesian, None),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    intervention_response_cpdag_frequentist_given: 0x57A7_0031, intervention_response(OnCpdag, Frequentist, false),
        [(Accepted, None), (Explicit, Cheap), (Explicit, Full), (Accepted, Cheap), (Accepted, Full)];
    mean_curve_cpdag_bayesian_given: 0x57A7_0032, mean_curve(OnCpdag, false),
        [(Accepted, None), (Explicit, Cheap), (Explicit, Full), (Accepted, Cheap), (Accepted, Full)];
    mean_curve_pag_bayesian_given: 0x57A7_0033, mean_curve(OnPag, false),
        [(Accepted, None), (Explicit, Cheap), (Explicit, Full), (Accepted, Cheap), (Accepted, Full)];
    intervention_response_cpdag_bayesian_given: 0x57A7_0034, intervention_response(OnCpdag, Bayesian, false),
        [(Accepted, None), (Explicit, Cheap), (Explicit, Full), (Accepted, Cheap), (Accepted, Full)];
    intervention_response_pag_bayesian_given: 0x57A7_0035, intervention_response(OnPag, Bayesian, false),
        [(Accepted, None), (Explicit, Cheap), (Explicit, Full), (Accepted, Cheap), (Accepted, Full)];
    average_effect_dag_frequentist_graph_posterior: 0x57A7_0036, average_effect(OnDag, Frequentist, None),
        [(Posterior, Cheap), (Posterior, Full)];
    average_effect_dag_bayesian_graph_posterior: 0x57A7_0037, average_effect(OnDag, Bayesian, None),
        [(Posterior, Cheap), (Posterior, Full)];
    average_effect_cpdag_frequentist_graph_posterior: 0x57A7_0038, average_effect(OnCpdag, Frequentist, None),
        [(Posterior, Cheap), (Posterior, Full)];
    average_effect_pag_frequentist_graph_posterior: 0x57A7_0039, average_effect(OnPag, Frequentist, None),
        [(Posterior, Cheap), (Posterior, Full)];
    average_effect_cpdag_bayesian_graph_posterior: 0x57A7_003A, average_effect(OnCpdag, Bayesian, None),
        [(Posterior, Cheap), (Posterior, Full)];
    average_effect_pag_bayesian_graph_posterior: 0x57A7_003B, average_effect(OnPag, Bayesian, None),
        [(Posterior, Cheap), (Posterior, Full)];
    distribution_dag_frequentist_given: 0x57A7_003C, distribution(OnDag, Frequentist),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    conditional_effect_dag_bayesian_given_primary: 0x57A7_003D, conditional_effect(OnDag, Bayesian),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    conditional_effect_dag_bayesian_given_bayesian_basis_gcomp: 0x57A7_003E, conditional_basis(),
        [(Accepted, None)];
    conditional_effect_cpdag_frequentist_given: 0x57A7_003F, conditional_effect(OnCpdag, Frequentist),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    conditional_effect_cpdag_bayesian_given: 0x57A7_0040, conditional_effect(OnCpdag, Bayesian),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    conditional_effect_pag_frequentist_given: 0x57A7_0041, conditional_effect(OnPag, Frequentist),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    conditional_effect_pag_bayesian_given: 0x57A7_0042, conditional_effect(OnPag, Bayesian),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    mean_curve_dag_bayesian_given: 0x57A7_0043, mean_curve(OnDag, false),
        [(Accepted, None), (Explicit, Cheap), (Explicit, Full), (Accepted, Cheap), (Accepted, Full)];
    intervention_response_dag_bayesian_given: 0x57A7_0044, intervention_response(OnDag, Bayesian, false),
        [(Accepted, None), (Explicit, Cheap), (Explicit, Full), (Accepted, Cheap), (Accepted, Full)];
    mediation_dag_frequentist_given: 0x57A7_0045, mediation(Frequentist),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_codetermined_tier_bayesian_given: 0x57A7_0046, codetermined_effect(),
        [(Explicit, None)];
    average_effect_undecided_tier_bayesian_given: 0x57A7_0047, undecided_tier_effect(),
        [(Explicit, None)];
    distribution_dag_bayesian_given: 0x57A7_0048, distribution(OnDag, Bayesian),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    mediation_dag_bayesian_given: 0x57A7_0049, mediation(Bayesian),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    conditional_effect_dag_frequentist_graph_posterior: 0x57A7_004A, conditional_effect(OnDag, Frequentist),
        [(Posterior, Cheap), (Posterior, Full)];
    conditional_effect_dag_bayesian_graph_posterior: 0x57A7_004B, conditional_effect(OnDag, Bayesian),
        [(Posterior, Cheap), (Posterior, Full)];
    distribution_admg_bayesian_given: 0x57A7_004C, distribution(OnAdmg, Bayesian),
        [(Accepted, None), (Explicit, Cheap), (Explicit, Full), (Accepted, Cheap), (Accepted, Full)];
    conditional_effect_cpdag_frequentist_graph_posterior: 0x57A7_004D, conditional_effect(OnCpdag, Frequentist),
        [(Posterior, Cheap), (Posterior, Full)];
    conditional_effect_cpdag_bayesian_graph_posterior: 0x57A7_004E, conditional_effect(OnCpdag, Bayesian),
        [(Posterior, Cheap), (Posterior, Full)];
    intervention_response_cpdag_frequentist_graph_posterior: 0x57A7_004F, intervention_response(OnCpdag, Frequentist, true),
        [(Posterior, None), (Posterior, Cheap)];
    intervention_response_cpdag_bayesian_graph_posterior: 0x57A7_0050, intervention_response(OnCpdag, Bayesian, true),
        [(Posterior, None), (Posterior, Cheap)];
    conditional_effect_pag_frequentist_graph_posterior: 0x57A7_0051, conditional_effect(OnPag, Frequentist),
        [(Posterior, Cheap), (Posterior, Full)];
    conditional_effect_pag_bayesian_graph_posterior: 0x57A7_0052, conditional_effect(OnPag, Bayesian),
        [(Posterior, Cheap), (Posterior, Full)];
    intervention_response_pag_frequentist_graph_posterior: 0x57A7_0053, intervention_response(OnPag, Frequentist, true),
        [(Posterior, None), (Posterior, Cheap)];
    intervention_response_pag_bayesian_graph_posterior: 0x57A7_0054, intervention_response(OnPag, Bayesian, true),
        [(Posterior, None), (Posterior, Cheap)];
    distribution_temporal_admg_frequentist_given: 0x57A7_0055, distribution(OnTemporalAdmg, Frequentist),
        [(Explicit, None), (Accepted, None), (Explicit, Cheap), (Explicit, Full), (Accepted, Cheap), (Accepted, Full)];
    distribution_temporal_admg_bayesian_given: 0x57A7_0056, distribution(OnTemporalAdmg, Bayesian),
        [(Explicit, None), (Accepted, None), (Explicit, Cheap), (Explicit, Full), (Accepted, Cheap), (Accepted, Full)];
    intervention_response_codetermined_tier_bayesian_given: 0x57A7_0057, codetermined_response(),
        [(Explicit, None)];
    intervention_response_dag_bayesian_graph_posterior: 0x57A7_0058, intervention_response(OnDag, Bayesian, true),
        [(Posterior, Cheap), (Posterior, Full)];
    intervention_response_pag_frequentist_given: 0x57A7_0059, intervention_response(OnPag, Frequentist, false),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, Cheap), (Accepted, Full)];
    mean_curve_cpdag_bayesian_graph_posterior: 0x57A7_005A, mean_curve(OnCpdag, true),
        [(Posterior, Cheap), (Posterior, Full)];
    mean_curve_pag_bayesian_graph_posterior: 0x57A7_005B, mean_curve(OnPag, true),
        [(Posterior, Cheap), (Posterior, Full)];
    distribution_cpdag_frequentist_given: 0x57A7_005C, distribution(OnCpdag, Frequentist),
        [(Explicit, None), (Accepted, None), (Explicit, Cheap), (Explicit, Full), (Accepted, Cheap), (Accepted, Full)];
    distribution_cpdag_bayesian_given: 0x57A7_005D, distribution(OnCpdag, Bayesian),
        [(Explicit, None), (Accepted, None), (Explicit, Cheap), (Explicit, Full), (Accepted, Cheap), (Accepted, Full)];
    distribution_pag_frequentist_given: 0x57A7_005E, distribution(OnPag, Frequentist),
        [(Explicit, None), (Accepted, None), (Explicit, Cheap), (Explicit, Full), (Accepted, Cheap), (Accepted, Full)];
    distribution_pag_bayesian_given: 0x57A7_005F, distribution(OnPag, Bayesian),
        [(Explicit, None), (Accepted, None), (Explicit, Cheap), (Explicit, Full), (Accepted, Cheap), (Accepted, Full)];
    distribution_admg_frequentist_given: 0x57A7_0060, distribution(OnAdmg, Frequentist),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, Cheap), (Accepted, Full)];
    distribution_dag_frequentist_graph_posterior: 0x57A7_0061, distribution(OnDag, Frequentist),
        [(Posterior, None), (Posterior, Cheap), (Posterior, Full)];
    distribution_dag_bayesian_graph_posterior: 0x57A7_0062, distribution(OnDag, Bayesian),
        [(Posterior, None), (Posterior, Cheap), (Posterior, Full)];
    distribution_admg_frequentist_graph_posterior: 0x57A7_0063, distribution(OnAdmg, Frequentist),
        [(Posterior, None), (Posterior, Cheap), (Posterior, Full)];
    distribution_admg_bayesian_graph_posterior: 0x57A7_0064, distribution(OnAdmg, Bayesian),
        [(Posterior, None), (Posterior, Cheap), (Posterior, Full)];
    distribution_temporal_admg_frequentist_graph_posterior: 0x57A7_0065, distribution(OnTemporalAdmg, Frequentist),
        [(Posterior, None), (Posterior, Cheap), (Posterior, Full)];
    distribution_temporal_admg_bayesian_graph_posterior: 0x57A7_0066, distribution(OnTemporalAdmg, Bayesian),
        [(Posterior, None), (Posterior, Cheap), (Posterior, Full)];
    distribution_cpdag_frequentist_graph_posterior: 0x57A7_0067, distribution(OnCpdag, Frequentist),
        [(Posterior, None), (Posterior, Cheap), (Posterior, Full)];
    distribution_cpdag_bayesian_graph_posterior: 0x57A7_0068, distribution(OnCpdag, Bayesian),
        [(Posterior, None), (Posterior, Cheap), (Posterior, Full)];
    distribution_pag_frequentist_graph_posterior: 0x57A7_0069, distribution(OnPag, Frequentist),
        [(Posterior, None), (Posterior, Cheap), (Posterior, Full)];
    distribution_pag_bayesian_graph_posterior: 0x57A7_006A, distribution(OnPag, Bayesian),
        [(Posterior, None), (Posterior, Cheap), (Posterior, Full)];
    mediation_dag_frequentist_graph_posterior: 0x57A7_006B, mediation(Frequentist),
        [(Posterior, None), (Posterior, Cheap), (Posterior, Full)];
    mediation_dag_bayesian_graph_posterior: 0x57A7_006C, mediation(Bayesian),
        [(Posterior, None), (Posterior, Cheap), (Posterior, Full)];
    mean_curve_dag_bayesian_graph_posterior: 0x57A7_006D, mean_curve(OnDag, true),
        [(Posterior, Cheap), (Posterior, Full)];
    average_effect_dag_frequentist_given_functional_effect: 0x57A7_006E, functional_effect(Frequentist),
        [(Explicit, Cheap), (Explicit, Full), (Accepted, None), (Accepted, Cheap), (Accepted, Full)];
    average_effect_dag_bayesian_given_functional_effect: 0x57A7_006F, functional_effect(Bayesian),
        [(Accepted, None)];
}
