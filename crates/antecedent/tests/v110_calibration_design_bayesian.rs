//! Repeated-sampling coverage of the Bayesian design-based coordinates: the
//! conjugate-Gaussian network-interference exposure contrast on a Dag, and the
//! Rubin Bayesian-bootstrap trial-to-target transported effect on an explicit
//! selection-diagram ADMG.
//!
//! Each test scores exactly the posterior-quantile credible interval the study
//! reports by default at 0.95, and the same construction at 0.90 re-derived from
//! the retained draws (`common::reported::posterior_pair`). Every tally is keyed
//! through [`keyed`], this file's one emission point: the key names only the
//! emitting test, its DGP and the interval method it scores. The construction
//! behind a record (query, graph class, estimator, inference, identification) is
//! never declared here — it is read from the runtime by binding every scored
//! replicate's execution with `common::calibration_bind`.
//!
//! Ignored tests run via `scripts/gate_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::doc_markdown)]
#![allow(
    clippy::cast_possible_truncation,
    clippy::float_cmp,
    reason = "test scaffolding compares exact constants and indexes with small literals"
)]

mod common;

use std::sync::Arc;

use antecedent::{
    BayesianConfig, InferenceMode, InterferenceSpec, RefuteSuite, Study, StudyResult,
    TransportTrialSpec,
};
use antecedent_core::{
    AssignmentDesign, CausalQuery, ContinuousDomain, ExecutionContext, ExposureLevel,
    ExposureMapping, GridSpec, InterferenceFunctional, InterferenceQuery, ResponseFunctional,
    ResponseQuery, TransportQuery, VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_graph::{Admg, Dag, DenseNodeId};
use common::calibration::{
    CoverageTally, RecordKey, gaussian, grid_n, map_replicates, n_sim, stream_seed, unit_uniform,
};
use common::calibration_bind::{bind_all, constructions};
use common::reported::{GATE_LEVEL, REPORTED_LEVEL, gate, posterior_pair, record_pair, skip_pair};

// ---------------------------------------------------------------- emission

/// The provenance of one design coordinate: the DGP its replicates are drawn
/// from and the estimator the plan must select. Everything else about the
/// record comes from the runtime.
#[derive(Clone, Copy)]
struct Cell {
    estimator: &'static str,
    dgp: &'static str,
}

/// This file's single coverage-record emission point. `test` is the name of the
/// `#[test] fn` that emits the record; both coordinates score the
/// posterior-quantile credible interval the facade reports.
fn keyed(test: &'static str, cell: Cell, level: f64) -> CoverageTally {
    CoverageTally::for_record(
        RecordKey { test, dgp: cell.dgp, interval: "posterior_quantile" },
        level,
    )
}

/// The reported-level and gate-level tallies of one coordinate, scored on the
/// same replicates. The two records differ by level, so they need no label.
fn keyed_pair(test: &'static str, cell: Cell) -> [CoverageTally; 2] {
    [keyed(test, cell, REPORTED_LEVEL), keyed(test, cell, GATE_LEVEL)]
}

/// Bind one replicate's execution to both levels' tallies, as `record_pair`
/// scores both from that execution.
fn bind_pair(tallies: &mut [CoverageTally; 2], study: &Study, result: &StudyResult) {
    let [reported, gated] = tallies;
    bind_all(&mut [reported, gated], study, result);
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn sigmoid(x: f64) -> f64 {
    1.0 / (1.0 + (-x).exp())
}

/// The interval the coverage test scores must be the posterior quantile the
/// facade reports, or a later run would score a construction the study never
/// published.
fn assert_posterior_quantile(study: &Study, result: &StudyResult) {
    let contract = study.inspect().expect("inspect");
    let methods: Vec<String> = constructions(&contract, result)
        .into_iter()
        .map(|(construction, _)| construction.interval_method)
        .collect();
    assert!(
        methods.iter().any(|m| m == "posterior_quantile"),
        "the execution reported {methods:?}, not posterior_quantile"
    );
}

// ============================================================== interference

const P_TREAT: f64 = 0.5;
const UNITS: usize = 400;
/// Shared intercept, direct effect, and neighbor effect of the fixed-network
/// Gaussian potential-outcome model. A *single* shared intercept (no per-unit
/// random effects) keeps the estimator's `[1, own, g]` design correctly
/// specified with residual variance exactly 1.
const ALPHA0: f64 = 1.0;
const BETA0: f64 = 2.0;
const GAMMA0: f64 = 0.5;
const INTERFERENCE_DRAWS: usize = 2_000;
/// Diffuse isotropic prior so the variance-1 conjugate likelihood dominates and
/// the credible interval is not shrunk toward zero at finite N.
const PRIOR_SCALE: f64 = 50.0;

/// Units of the finite population at this run's sample-size grid point.
fn units() -> usize {
    grid_n(UNITS)
}

fn ring_edges() -> Vec<NetworkEdge> {
    let n = units() as u32;
    (0..n)
        .flat_map(|i| {
            [
                NetworkEdge { from: (i + 1) % n, to: i, weight: 1.0 },
                NetworkEdge { from: (i + n - 1) % n, to: i, weight: 1.0 },
            ]
        })
        .collect()
}

/// The study and its result, so a scored replicate can be bound to its tally.
///
/// Fixed ring network of [`units`] units, neighbor set `{i−1, i+1}`. Each
/// replicate draws a fresh Bernoulli(1/2) assignment `z`, forms the
/// `NeighborCount` exposure `g_i = z_{i−1} + z_{i+1} ∈ {0,1,2}`, and draws
/// outcomes `y_i = ALPHA0 + BETA0·z_i + GAMMA0·g_i + ε_i` with `ε_i ~ N(0,1)`.
/// This is exactly the estimator's own fixed-network Gaussian likelihood, so the
/// conjugate posterior on the contrast is correctly specified.
fn run_interference(seed: u64) -> Option<(Study, StudyResult)> {
    let units = units();
    let assignment: Vec<bool> =
        (0..units).map(|i| unit_uniform(stream_seed(seed, i as u64)) < P_TREAT).collect();
    let mut noise = gaussian(stream_seed(seed, 0x9E37_79B9_7F4A_7C15));
    let y: Vec<f64> = (0..units)
        .map(|i| {
            let k = usize::from(assignment[(i + 1) % units])
                + usize::from(assignment[(i + units - 1) % units]);
            ALPHA0 + BETA0 * f64::from(u8::from(assignment[i])) + GAMMA0 * k as f64 + noise()
        })
        .collect();
    let units_data = TabularData::from_f64_columns([("y", y.as_slice())]).unwrap();
    let network = NetworkData::try_new(units_data.clone(), ring_edges()).unwrap();
    let query = InterferenceQuery::new(
        AssignmentDesign::Bernoulli { probabilities: Arc::from([P_TREAT]) },
        ExposureMapping::NeighborCount,
        InterferenceFunctional::ExposureContrast {
            outcome: v(0),
            from: ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: ExposureLevel { own: 1.0, neighbors: 0.0 },
        },
    );
    let study = Study::tabular(units_data)
        .graph(Dag::with_variables(1))
        .query(CausalQuery::Interference(query))
        .interference(InterferenceSpec { network, assignment: Arc::from(assignment) })
        .inference(InferenceMode::Bayesian(
            BayesianConfig::conjugate().n_draws(INTERFERENCE_DRAWS).prior_scale(PRIOR_SCALE),
        ))
        .refute(RefuteSuite::None)
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    Some((study, result))
}

const INTERFERENCE_CELL: Cell =
    Cell { estimator: "interference.bayesian_gaussian", dgp: "run_interference" };

/// Conjugate-Gaussian posterior of the finite-network exposure contrast.
///
/// The contrast is `(own 0, nb 0) → (own 1, nb 0)`. Under the structural model
/// `y_i = ALPHA0 + BETA0·own_i + GAMMA0·g_i + ε_i` the contrast equals
/// `BETA0·(1 − 0) + GAMMA0·(0 − 0) = BETA0`. Truth = `BETA0 = 2.0`. The
/// estimator's likelihood is this exact model (residual variance fixed to 1),
/// so with a diffuse prior and a large ring the credible interval is
/// frequency-calibrated.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn interference_query_dag_bayesian_neighbor_count_nominal_coverage() {
    // contrast (own 0, nb 0) -> (own 1, nb 0) = BETA0 (the neighbor term cancels).
    let truth = BETA0;
    let mut tallies =
        keyed_pair("interference_query_dag_bayesian_neighbor_count_nominal_coverage", INTERFERENCE_CELL);
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0x110_0401, rep);
        let (study, result) = run_interference(seed)?;
        if rep == 0 {
            assert_eq!(result.logical_plan.estimator.as_deref(), Some(INTERFERENCE_CELL.estimator));
            assert_posterior_quantile(&study, &result);
            assert!(
                posterior_pair(&result, 0)[0].is_some(),
                "the interference posterior contrast must publish its interval"
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
        record_pair(&mut tallies, posterior_pair(result, 0), truth);
    }
    eprintln!("info interference_query: contrast truth {truth:.6}");
    gate(&tallies, &[None, None]);
}

// ================================================================ transport

const TRANSPORT_DRAWS: usize = 2_000;
/// Individuals drawn to seed the frozen target population; those with `S = 0`
/// become the fixed target rows.
const TARGET_POOL: usize = 8_000;
const TARGET_SEED: u64 = 0x110_0402_FEED;
/// Individuals drawn per replicate; those with `S = 1` become that replicate's
/// fresh trial sample.
const TRIAL_POOL: usize = 2_000;

/// One individual of the transport DGP (`transport_data` in the frequentist
/// twin): `x ~ N(0,1)`; trial membership `S | x ~ Bern(σ(0.5x))`; inside the
/// trial `a ~ Bern(1/2)` and `y = 1 + a(1 + x) + x + ε`, `ε ~ N(0,1)`; the CATE
/// is `τ(x) = 1 + x`. Target individuals (`S = 0`) carry no treatment or
/// outcome.
struct Individual {
    x: f64,
    trial: bool,
    a: f64,
    y: f64,
}

fn draw_pool(pool: usize, seed: u64) -> Vec<Individual> {
    let mut g = gaussian(seed);
    (0..pool)
        .map(|i| {
            let x = g();
            let s = sigmoid(0.5 * x);
            let trial = unit_uniform(stream_seed(seed, 2 * i as u64)) < s;
            let a = if trial {
                f64::from(unit_uniform(stream_seed(seed, 2 * i as u64 + 1)) < 0.5)
            } else {
                0.0
            };
            let noise = g();
            let y = if trial { 1.0 + a * (1.0 + x) + x + noise } else { 0.0 };
            Individual { x, trial, a, y }
        })
        .collect()
}

/// The fixed target rows (`S = 0`) drawn once from [`TARGET_SEED`], held frozen
/// across every replicate.
fn frozen_target() -> Vec<Individual> {
    draw_pool(TARGET_POOL, TARGET_SEED).into_iter().filter(|ind| !ind.trial).collect()
}

/// Columns `a, y, trial, s, e, x`: the frozen target rows followed by this
/// replicate's fresh trial rows (`S = 1`). Only the trial is redrawn, so the
/// Bayesian bootstrap conditions on exactly the trial-sampling uncertainty it
/// propagates.
fn transport_frozen_data(target: &[Individual], seed: u64) -> TabularData {
    let n_trial_pool = grid_n(TRIAL_POOL);
    let trial_rows: Vec<Individual> =
        draw_pool(n_trial_pool, seed).into_iter().filter(|ind| ind.trial).collect();
    let total = target.len() + trial_rows.len();
    let mut cols: [Vec<f64>; 6] = std::array::from_fn(|_| vec![0.0; total]);
    for (row, ind) in target.iter().chain(trial_rows.iter()).enumerate() {
        cols[0][row] = ind.a;
        cols[1][row] = ind.y;
        cols[2][row] = f64::from(ind.trial);
        cols[3][row] = sigmoid(0.5 * ind.x);
        cols[4][row] = 0.5;
        cols[5][row] = ind.x;
    }
    TabularData::from_f64_columns([
        ("a", cols[0].as_slice()),
        ("y", cols[1].as_slice()),
        ("trial", cols[2].as_slice()),
        ("s", cols[3].as_slice()),
        ("e", cols[4].as_slice()),
        ("x", cols[5].as_slice()),
    ])
    .unwrap()
}

/// The study and its result. Selection diagram ADMG `S -> x, x -> y, a -> y`
/// (S-admissible standardization over `x`); the transported mean curve is routed
/// under Bayesian inference to `transport.trial_bayesian_bootstrap`.
fn run_transport(data: TabularData, seed: u64) -> Option<(Study, StudyResult)> {
    let mut admg = Admg::with_variables(6);
    admg.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    admg.insert_directed(DenseNodeId::from_raw(5), DenseNodeId::from_raw(1)).unwrap();
    let response = ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: v(1),
        treatment: ContinuousDomain::new(v(0), GridSpec::Values(Arc::from([0.0, 1.0]))),
    });
    let study = Study::tabular(data)
        .graph(admg)
        .query(CausalQuery::Transport(TransportQuery::new(response, "trial", "target", [v(0)])))
        .selection_targets(Arc::from([v(5)]))
        .transport_trial(TransportTrialSpec {
            trial: v(2),
            selection_probability: v(3),
            treatment_probability: v(4),
        })
        .inference(InferenceMode::Bayesian(BayesianConfig::laplace().n_draws(TRANSPORT_DRAWS)))
        .refute(RefuteSuite::None)
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    Some((study, result))
}

const TRANSPORT_CELL: Cell =
    Cell { estimator: "transport.trial_bayesian_bootstrap", dgp: "transport_data_frozen_target" };

/// Transported effect over the frozen target rows: `truth = (1/n_target)·Σ_j
/// τ(x_j) = 1 + mean_target(x_j)`, since `τ(x) = 1 + x`. Computed exactly from
/// the frozen target `x` column (no integral).
fn transport_truth_frozen(target: &[Individual]) -> f64 {
    let mean_x = target.iter().map(|ind| ind.x).sum::<f64>() / target.len() as f64;
    1.0 + mean_x
}

/// Rubin Bayesian bootstrap of the trial-to-target transported effect, target
/// rows frozen. The posterior conditions on the target sample and the known
/// selection/treatment probabilities, so its estimand is the finite-target
/// conditional effect the truth above computes exactly; the Bayesian bootstrap
/// captures exactly the trial-sampling uncertainty it conditions on, so
/// `posterior_quantile` coverage is nominal.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn transport_query_admg_bayesian_bootstrap_nominal_coverage() {
    let target = frozen_target();
    // truth = 1 + mean over the frozen target x column (tau(x) = 1 + x).
    let truth = transport_truth_frozen(&target);
    let mut tallies =
        keyed_pair("transport_query_admg_bayesian_bootstrap_nominal_coverage", TRANSPORT_CELL);
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0x110_0403, rep);
        let (study, result) = run_transport(transport_frozen_data(&target, seed), seed)?;
        if rep == 0 {
            assert_eq!(result.logical_plan.estimator.as_deref(), Some(TRANSPORT_CELL.estimator));
            assert_posterior_quantile(&study, &result);
            assert!(
                posterior_pair(&result, 0)[0].is_some(),
                "the transported posterior must publish its interval"
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
        record_pair(&mut tallies, posterior_pair(result, 0), truth);
    }
    eprintln!("info transport_query: frozen-target transported truth {truth:.6}");
    gate(&tallies, &[None, None]);
}
