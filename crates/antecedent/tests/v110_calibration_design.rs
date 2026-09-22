//! Repeated-sampling coverage of the design-based coordinates: the
//! trial-to-target transported ATE on an explicit selection-diagram ADMG and
//! the randomized network-interference exposure contrast on a Dag.
//!
//! Every test scores exactly the interval the study reports by default at
//! 0.95, and the same construction at 0.90 from the same replicates
//! (`common::reported`). Every tally is keyed through [`keyed`], this file's
//! one emission point: the key names only the emitting test, its DGP and the
//! interval method it scores. The construction behind a record (query, graph
//! class, estimator, SE kind, dependence, identification) is never declared
//! here — it is read from the runtime by binding every scored replicate's
//! execution with `common::calibration_bind`.
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

use antecedent::{InterferenceSpec, RefuteSuite, Study, StudyResult, TransportTrialSpec};
use antecedent_core::{
    AssignmentDesign, CausalQuery, ContinuousDomain, ExecutionContext, ExposureLevel,
    ExposureMapping, GridSpec, InterferenceFunctional, InterferenceQuery, ResponseFunctional,
    ResponseQuery, TransportQuery, VariableId,
};
use antecedent_data::{NetworkData, NetworkEdge, TabularData};
use antecedent_graph::{Admg, Dag, DenseNodeId};
use common::calibration::{
    CoverageTally, GRID_POINTS, RecordKey, gaussian, grid_n, map_replicates, n_sim, stream_seed,
    unit_uniform,
};
use common::calibration_bind::bind_all;
use common::reported::{
    GATE_LEVEL, REPORTED_LEVEL, gate, record_pair, scalar_normal_pair, skip_pair,
};

// ---------------------------------------------------------------- emission

/// The provenance of one design coordinate: the DGP its replicates are drawn
/// from and the estimator the plan must select. Everything else about the
/// record comes from the runtime.
#[derive(Clone, Copy)]
struct Cell {
    estimator: &'static str,
    dgp: &'static str,
}

/// This file's single coverage-record emission point. `test` is the name of
/// the `#[test] fn` that emits the record; both coordinates score the
/// analytic-SE interval the facade reports.
fn keyed(test: &'static str, cell: Cell, level: f64) -> CoverageTally {
    CoverageTally::for_record(RecordKey { test, dgp: cell.dgp, interval: "analytic_se" }, level)
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

// ================================================================ transport

/// Columns `a, y, trial, s, e, x`. Population `x ~ N(0,1)`; trial membership
/// `S | x ~ Bern(s(x))`, `s(x) = σ(0.5x)` (known, column `s`); inside the
/// trial `a ~ Bern(1/2)` (known, column `e`); `y = 1 + a(1 + x) + x + ε`.
/// Target rows (`S = 0`) carry no treatment or outcome. The effect is modified
/// by `x`, whose law differs between trial and target, so the transported
/// target ATE is `1 + E[x | S = 0] = 1 − 2·E[x σ(0.5x)]` (`E[s] = 1/2`), not
/// the trial ATE `1 + E[x | S = 1]`.
fn transport_data(n: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let mut cols: [Vec<f64>; 6] = std::array::from_fn(|_| vec![0.0; n]);
    for i in 0..n {
        let x = g();
        let s = sigmoid(0.5 * x);
        let trial = unit_uniform(stream_seed(seed, 2 * i as u64)) < s;
        let a = if trial {
            f64::from(unit_uniform(stream_seed(seed, 2 * i as u64 + 1)) < 0.5)
        } else {
            0.0
        };
        let noise = g();
        cols[0][i] = a;
        cols[1][i] = if trial { 1.0 + a * (1.0 + x) + x + noise } else { 0.0 };
        cols[2][i] = f64::from(trial);
        cols[3][i] = s;
        cols[4][i] = 0.5;
        cols[5][i] = x;
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

/// `1 − 2·E[x σ(0.5x)]` by the trapezoid rule on `[−12, 12]`.
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

/// The study and its result, so a scored replicate can be bound to its tally.
fn run_transport(data: TabularData, seed: u64) -> Option<(Study, StudyResult)> {
    // Selection diagram: S -> x, x -> y, a -> y (S-admissible standardization over x).
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
        .refute(RefuteSuite::None)
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    Some((study, result))
}

const TRANSPORT_CELL: Cell = Cell { estimator: "transport.trial_ipw", dgp: "transport_data" };

/// Dahabreh trial-to-target IPW with its ratio-of-means delta-method SE.
#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn transport_admg_trial_ipw_frequentist_nominal_coverage() {
    let truth = transport_truth();
    let mut tallies =
        keyed_pair("transport_admg_trial_ipw_frequentist_nominal_coverage", TRANSPORT_CELL);
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0x110_0201, rep);
        let (study, result) = run_transport(transport_data(grid_n(1000), seed), seed)?;
        if rep == 0 {
            assert_eq!(result.logical_plan.estimator.as_deref(), Some(TRANSPORT_CELL.estimator));
        }
        Some((study, result))
    });
    for scored in &runs {
        let Some((study, result)) = scored else {
            skip_pair(&mut tallies);
            continue;
        };
        bind_pair(&mut tallies, study, result);
        record_pair(&mut tallies, scalar_normal_pair(result), truth);
    }
    eprintln!("info transport: target ATE truth {truth:.6}");
    gate(&tallies, &[None, None]);
}

// ============================================================== interference

const UNITS: usize = 400;

/// Units of the finite population at this run's sample-size grid point.
fn units() -> usize {
    grid_n(UNITS)
}
const P_TREAT: f64 = 0.5;
const DIRECT_EFFECT: f64 = 2.0;

/// Finite population of [`UNITS`] units on a ring: unit `i` receives exposure
/// edges from `i − 1` and `i + 1`, so its `NeighborCount` exposure is 0, 1 or
/// 2. Fixed potential outcomes `y_i(own, k) = α_i + 2·own + 0.5·k` with
/// `α_i = 1 + 0.5 sin(i) + 0.5 ε_i` (drawn once, not per replicate). Each
/// replicate re-randomizes the Bernoulli(1/2) assignment. The contrast
/// `(own 0, 0 neighbors) → (own 1, 0 neighbors)` is 2 for every unit, so the
/// finite-population estimand is exactly 2.
fn alpha() -> Vec<f64> {
    let mut g = gaussian(0x1F7E_A1FA);
    (0..units()).map(|i| 1.0 + 0.5 * (i as f64).sin() + 0.5 * g()).collect()
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
fn run_interference(alpha: &[f64], seed: u64) -> Option<(Study, StudyResult)> {
    let units = units();
    let assignment: Vec<bool> =
        (0..units).map(|i| unit_uniform(stream_seed(seed, i as u64)) < P_TREAT).collect();
    let y: Vec<f64> = (0..units)
        .map(|i| {
            let k = usize::from(assignment[(i + 1) % units])
                + usize::from(assignment[(i + units - 1) % units]);
            alpha[i] + DIRECT_EFFECT * f64::from(u8::from(assignment[i])) + 0.5 * k as f64
        })
        .collect();
    let units = TabularData::from_f64_columns([("y", y.as_slice())]).unwrap();
    let network = NetworkData::try_new(units.clone(), ring_edges()).unwrap();
    let query = InterferenceQuery::new(
        AssignmentDesign::Bernoulli { probabilities: Arc::from([P_TREAT]) },
        ExposureMapping::NeighborCount,
        InterferenceFunctional::ExposureContrast {
            outcome: v(0),
            from: ExposureLevel { own: 0.0, neighbors: 0.0 },
            to: ExposureLevel { own: 1.0, neighbors: 0.0 },
        },
    );
    let study = Study::tabular(units)
        .graph(Dag::with_variables(1))
        .query(CausalQuery::Interference(query))
        .interference(InterferenceSpec { network, assignment: Arc::from(assignment) })
        .refute(RefuteSuite::None)
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    Some((study, result))
}

/// The replicate's data is the fixed [`alpha`] potential outcomes on
/// [`ring_edges`] under a freshly drawn assignment, so [`run_interference`] is
/// the function that generates it.
const INTERFERENCE_CELL: Cell =
    Cell { estimator: "interference.ht_hajek", dgp: "run_interference" };

/// Boundary cell: the design-based Horvitz–Thompson exposure contrast is
/// published with the conservative Young variance bound, which bounds the
/// unobservable joint-exposure covariance from above instead of estimating
/// it. On this finite population every replicate's interval contains the
/// estimand ([`INTERFERENCE_MEASURED`]); its mean length is about twenty
/// times the direct effect, because an exposure probability of 1/8 divides
/// every scored outcome. The cell is asserted against that measured
/// coverage, not nominal: an interval that started to miss would be a
/// regression in the bound, and the bound cannot be sharpened into a
/// nominal interval without the Aronow–Samii joint-exposure variance.
const INTERFERENCE_MEASURED: [f64; GRID_POINTS] = [1.0; GRID_POINTS];

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn interference_dag_bernoulli_neighbor_count_conservative_bound_boundary() {
    let alpha = alpha();
    let mut tallies = keyed_pair(
        "interference_dag_bernoulli_neighbor_count_conservative_bound_boundary",
        INTERFERENCE_CELL,
    );
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0x110_0202, rep);
        let (study, result) = run_interference(&alpha, seed)?;
        if rep == 0 {
            assert_eq!(result.logical_plan.estimator.as_deref(), Some(INTERFERENCE_CELL.estimator));
        }
        Some((study, result))
    });
    for scored in &runs {
        let Some((study, result)) = scored else {
            skip_pair(&mut tallies);
            continue;
        };
        bind_pair(&mut tallies, study, result);
        record_pair(&mut tallies, scalar_normal_pair(result), DIRECT_EFFECT);
    }
    gate(&tallies, &[Some(INTERFERENCE_MEASURED), Some(INTERFERENCE_MEASURED)]);
}

/// Each coordinate publishes the interval its coverage test scores.
#[test]
fn design_coordinates_publish_the_scored_interval() {
    let (_, transport) =
        run_transport(transport_data(400, 5), 5).expect("the transport study must run");
    assert!(
        transport.estimate.se_analytic.is_finite() && transport.estimate.se_analytic > 0.0,
        "the transported IPW must publish its SE"
    );
    assert!(scalar_normal_pair(&transport)[0].is_some());
    let (_, interference) = run_interference(&alpha(), 5).expect("the interference study must run");
    assert!(scalar_normal_pair(&interference)[0].is_some());
}
