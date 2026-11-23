//! Coverage of the 95% interval of a linear composition across the horizons of one temporal
//! response, against the closed-form composed truth.
//!
//! The procedure scored is the compiled temporal mean curve with its joint horizon replicates
//! retained (`Study::retain_joint_horizons`) and the interval the route publishes for the
//! composition (`horizon_composition_interval`): the composed point value with the spread of the
//! composed replicates at the dispersion of the composed influence. The frequentist route
//! resamples circular blocks of the lag-aligned rows and refits every horizon on the same blocks
//! (fixed-b normal reference); the Bayesian route draws one Exp(1) weight per block and refits
//! every horizon by weighted least squares (batch-means Student-t reference).
//!
//! Law: `x` is iid `1 + 0.5 e`, `y(t) = 1 + 2 x(t-1) + 3 x(t-2) + 0.5 e'`. A pulse at step 0 read
//! at horizon `h` is `E[y(t + h - 1) | do(x(t) = a)]` (the outcome offset of a horizon is
//! `h - 1`), so horizon 1 is the contemporaneous level, which no edge carries:
//! `E[y] = 1 + 2 + 3 = 6` at every dose. Horizon 2 reads the lag-one effect with the other lag
//! at its mean 1: `4 + 2 a`. At dose 1.25 the cumulative effect `h1 + h2` is 12.5 and the
//! contrast `h2 - h1` is 0.5.
//!
//! The composed key is the one the claim route derives (`composed_basis` of the composed
//! coordinate, which carries the horizon axis, the aligned joint-law route and the construction
//! the interval names). The horizon composition has no resolver of its own for that key.
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "bounded replicate indices in a calibration fixture"
)]
#![allow(clippy::cast_precision_loss, reason = "bounded simulation sizes and small indices")]

mod common;

use std::sync::Arc;

use antecedent::analysis::horizon_composition::{
    HORIZON_BLOCK_BOOTSTRAP_INTERVAL, HORIZON_BLOCK_WEIGHT_INTERVAL, horizon_composition_interval,
};
use antecedent::analysis::joint_law_composed_basis::composed_basis;
use antecedent::{BayesianConfig, InferenceMode, RefuteSuite, Study};
use antecedent_contract::{PropagationDependence, PropagationOperation};
use antecedent_core::{
    CalibrationBasis, CausalQuery, ContinuousDomain, ExecutionContext, GridSpec, Lag,
    ResponseFunctional, ResponseQuery, TemporalPolicy, TemporalResponseSpec, VariableId,
};
use antecedent_data::TimeSeriesData;
use antecedent_graph::{TemporalDag, ensure_lagged};
use common::calibration::{
    Construction, CoverageTally, RecordKey, ScopeFacts, gaussian, grid_n, grid_seed,
    map_replicates, n_sim, stream_seed,
};

const LEVEL: f64 = 0.95;
const BASE_ROWS: usize = 160;
const BURN: usize = 10;
const DRAWS: u32 = 199;
const DOSE: f64 = 1.25;
const STACK_BYTES: usize = 32 * 1024 * 1024;

/// The mean level of horizon 1 (outcome at the pulse's own step): `1 + (2 + 3) E[x]`, the same at
/// every dose because `x(t)` has no contemporaneous effect on `y(t)`.
fn horizon_one(_dose: f64) -> f64 {
    6.0
}

/// The mean level of horizon 2 (outcome one step after the pulse) at `dose`:
/// `1 + 2 dose + 3 E[x]`.
fn horizon_two(dose: f64) -> f64 {
    4.0 + 2.0 * dose
}

fn column(rows: usize, sd: f64, seed: u64, stream: u64) -> Vec<f64> {
    let mut draw = gaussian(stream_seed(seed, stream));
    (0..rows).map(|_| sd * draw()).collect()
}

fn lagged(values: &[f64], s: usize, lag: usize) -> f64 {
    s.checked_sub(lag).map_or(0.0, |i| values[i])
}

fn series(seed: u64) -> TimeSeriesData {
    let n = grid_n(BASE_ROWS) + BURN;
    let x: Vec<f64> = column(n, 0.5, seed, 1).iter().map(|g| 1.0 + g).collect();
    let noise = column(n, 0.5, seed, 2);
    let y: Vec<f64> =
        (0..n).map(|s| 1.0 + 2.0 * lagged(&x, s, 1) + 3.0 * lagged(&x, s, 2) + noise[s]).collect();
    TimeSeriesData::from_f64_columns([("x", &x[BURN..]), ("y", &y[BURN..])], 1).unwrap()
}

fn graph() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let x1 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let x2 = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(2)).unwrap();
    let y0 = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(x1, y0).unwrap();
    graph.insert_directed(x2, y0).unwrap();
    graph
}

fn query() -> ResponseQuery {
    ResponseQuery::new(ResponseFunctional::MeanCurve {
        outcome: VariableId::from_raw(1),
        treatment: ContinuousDomain::new(
            VariableId::from_raw(0),
            GridSpec::Values(Arc::from([0.75, DOSE])),
        ),
    })
    .with_temporal(TemporalResponseSpec::new(vec![1, 2], TemporalPolicy::pulse(0), None).unwrap())
}

/// Run `work` on a thread with a stack large enough for the compiled study.
fn on_large_stack<T: Send>(work: impl FnOnce() -> T + Send) -> T {
    std::thread::scope(|scope| {
        let handle = std::thread::Builder::new()
            .stack_size(STACK_BYTES)
            .spawn_scoped(scope, work)
            .expect("spawn the replicate thread");
        match handle.join() {
            Ok(value) => value,
            Err(payload) => std::panic::resume_unwind(payload),
        }
    })
}

/// One replicate's composed interval and the key it is measured under.
struct Reading {
    interval: (f64, f64),
    basis: CalibrationBasis,
}

fn replicate(
    seed: u64,
    bayesian: bool,
    coefficients: [f64; 2],
    operation: PropagationOperation,
) -> Option<Reading> {
    on_large_stack(move || {
        let rows = grid_n(BASE_ROWS);
        let (inference, bootstrap) = if bayesian {
            (InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS as usize)), 0)
        } else {
            (InferenceMode::Frequentist, DRAWS)
        };
        let result = Study::series(series(seed))
            .graph(graph())
            .query(CausalQuery::Response(query()))
            .inference(inference)
            .refute(RefuteSuite::None)
            .bootstrap_replicates(bootstrap)
            .retain_joint_horizons(true)
            .build()
            .ok()?
            .run(&ExecutionContext::for_tests(seed))
            .ok()?;
        let interval =
            horizon_composition_interval(result.response()?, 1, &coefficients, operation).ok()?;
        let basis = composed_basis(
            &interval.composed.coordinate.id(),
            interval.composed.plan.route.as_str(),
            &PropagationDependence::AlignedJointLaw,
            interval.construction,
            rows as u64,
        );
        Some(Reading { interval: (interval.lower, interval.upper), basis })
    })
}

fn bind_basis(tally: &mut CoverageTally, basis: &CalibrationBasis) {
    tally.bind_validation(
        &Construction {
            query: basis.query.to_string(),
            graph_class: basis.graph_class.to_string(),
            structure: basis.structure.to_string(),
            modality: basis.modality.to_string(),
            inference: basis.inference.to_string(),
            estimator: basis.estimator.to_string(),
            interval_method: basis.interval_method.to_string(),
            se_kind: basis.se_kind.to_string(),
            dependence: basis.dependence.to_string(),
            posterior: basis.posterior.to_string(),
            functional: basis.functional.to_string(),
            identification: basis.identification.to_string(),
            reported_level: basis.level,
        },
        ScopeFacts {
            row_count: basis.row_count,
            replicates_ok: basis.replicates_ok,
            posterior_draws: basis.posterior_draws,
            unidentified_mass: basis.unidentified_mass,
        },
        &basis.validation,
    );
}

fn measure(
    test: &'static str,
    dgp: &'static str,
    bayesian: bool,
    coefficients: [f64; 2],
    operation: PropagationOperation,
    truth: f64,
    seed_base: u64,
) {
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test,
            dgp,
            interval: if bayesian {
                HORIZON_BLOCK_WEIGHT_INTERVAL
            } else {
                HORIZON_BLOCK_BOOTSTRAP_INTERVAL
            },
        },
        LEVEL,
    );
    let readings = map_replicates(n_sim(), |rep| {
        replicate(grid_seed(seed_base + rep), bayesian, coefficients, operation)
    });
    for reading in readings {
        let Some(reading) = reading else {
            tally.skip();
            continue;
        };
        bind_basis(&mut tally, &reading.basis);
        tally.record(Some(reading.interval), truth);
    }
    tally.assert();
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn horizon_cumulative_effect_block_bootstrap_l95() {
    measure(
        "horizon_cumulative_effect_block_bootstrap_l95",
        "iid_treatment_two_lag_linear_outcome_cumulative_effect_at_dose_1_25_circular_block_bootstrap",
        false,
        [1.0, 1.0],
        PropagationOperation::Sum,
        horizon_one(DOSE) + horizon_two(DOSE),
        0x4D20_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn horizon_contrast_block_bootstrap_l95() {
    measure(
        "horizon_contrast_block_bootstrap_l95",
        "iid_treatment_two_lag_linear_outcome_horizon_contrast_at_dose_1_25_circular_block_bootstrap",
        false,
        [-1.0, 1.0],
        PropagationOperation::Contrast,
        horizon_two(DOSE) - horizon_one(DOSE),
        0x4D21_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn horizon_cumulative_effect_block_weight_posterior_l95() {
    measure(
        "horizon_cumulative_effect_block_weight_posterior_l95",
        "iid_treatment_two_lag_linear_outcome_cumulative_effect_at_dose_1_25_block_weight_posterior",
        true,
        [1.0, 1.0],
        PropagationOperation::Sum,
        horizon_one(DOSE) + horizon_two(DOSE),
        0x4D22_0000,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn horizon_contrast_block_weight_posterior_l95() {
    measure(
        "horizon_contrast_block_weight_posterior_l95",
        "iid_treatment_two_lag_linear_outcome_horizon_contrast_at_dose_1_25_block_weight_posterior",
        true,
        [-1.0, 1.0],
        PropagationOperation::Contrast,
        horizon_two(DOSE) - horizon_one(DOSE),
        0x4D23_0000,
    );
}
