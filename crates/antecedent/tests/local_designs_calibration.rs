//! Repeated-sampling coverage of the local-ratio design routes: the fuzzy regression
//! discontinuity and regression kink with a selected or diagnosed bandwidth, the same designs
//! with calendar time as the running variable, and their Bayesian counterparts.
//!
//! Every design runs the public route on a simulated dataset and scores the interval the route
//! publishes at 95% against the analytic target of the simulation:
//!
//! * the robust bias-corrected HC0 delta interval of independent rows, at the MSE-optimal plug-in
//!   bandwidth and at a declared bandwidth with the density and balance diagnostics (read from the
//!   rollout payload's overall effect);
//! * the within-side moving-block bootstrap Student-t interval of a discontinuity or kink in
//!   time under serially dependent errors (read from the same payload);
//! * the fixed-bandwidth Bayesian local-linear posterior of the jump or kink ratio, on data drawn
//!   from the model's own Gaussian local-linear likelihood, and the within-side block Bayesian
//!   bootstrap posterior in time (read from the posterior's `q025` / `q975`).
//!
//! The target of every design is the ratio of the outcome's jump (or slope change) to the
//! treatment's jump (or slope change) at the cutoff. The outcome is generated as a fixed
//! multiple of the received treatment plus a smooth trend and noise, so the ratio is that
//! multiple.
//!
//! The frequentist design-layer intervals ride on the rollout payload and the result reports no
//! scalar interval for them, so the runtime's own calibration key has no interval method for
//! them. Each of those records takes the rest of its construction from the runtime's key
//! ([`StudyResult::calibration_bases`] through `common::calibration_bind`) and names the
//! interval it scores. The Bayesian designs bind the posterior-quantile construction the
//! runtime reports. The runtime has no query name for the local ratio, so every record names
//! it, and every record is keyed on the design axes (structure `design`, validation
//! `not_applicable`), as the randomized design routes are: these routes have no graph and run
//! no validation suite.
//!
//! The designs are `#[ignore]`d measurement designs, run by `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent::{BayesianConfig, InferenceMode, Study, StudyResult};
use antecedent_core::{
    ExecutionContext, LocalBandwidthRule, LocalPolynomialRatioQuery, LocalRatioOptions, VariableId,
};
use antecedent_data::TabularData;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, ar1_noise, gaussian,
    grid_n, map_replicates, n_sim, stream_seed, uniform,
};
use common::calibration_bind::{bind_reported, constructions};
use common::reported::{GATE_LEVEL, gate, posterior_pair};

const TREATMENT: f64 = 2.0;
const POSTERIOR_QUANTILE: &str = "posterior_quantile";
/// Query name of every record.
const QUERY: &str = "LocalPolynomialRatio";
/// A design route runs no validation suite.
const NO_VALIDATION: &str = "not_applicable";

fn variable(index: u32) -> VariableId {
    VariableId::from_raw(index)
}

fn tabular(running: &[f64], treatment: &[f64], outcome: &[f64]) -> TabularData {
    TabularData::from_f64_columns([("x", running), ("t", treatment), ("y", outcome)])
        .expect("the simulated columns are finite and aligned")
}

fn query(
    cutoff: f64,
    bandwidth: f64,
    kink: bool,
    options: Option<LocalRatioOptions>,
) -> LocalPolynomialRatioQuery {
    LocalPolynomialRatioQuery {
        outcome: variable(2),
        treatment: variable(1),
        running: variable(0),
        cutoff,
        bandwidth,
        kink,
        options,
    }
}

/// The study and its result; the first replicate must run, so a design bug is loud, and a later
/// refusal is a replicate that publishes nothing.
fn run_study(
    data: TabularData,
    query: LocalPolynomialRatioQuery,
    inference: Option<InferenceMode>,
    seed: u64,
    first: bool,
) -> Option<(Study, StudyResult)> {
    let built = match inference {
        Some(mode) => Study::tabular(data).query(query).inference(mode).build(),
        None => Study::tabular(data).query(query).build(),
    };
    let study = match built {
        Ok(study) => study,
        Err(error) if first => panic!("the design declares a valid query: {error:?}"),
        Err(_) => return None,
    };
    let result = match study.run(&ExecutionContext::for_tests(seed)) {
        Ok(result) => result,
        Err(error) if first => panic!("the route runs on the simulated data: {error:?}"),
        Err(_) => return None,
    };
    Some((study, result))
}

// ------------------------------------------------------------------ data

/// Rows of independent observations on an even grid of the running variable over `[-1, 1]`:
/// a binary treatment received with probability `0.1 + 0.02 x + 0.8 [x > 0]`, and the outcome
/// `0.7 + 0.5 x + 0.3 x^2 + 0.1 x^3 + 2 t + noise`. The local ratio is the treatment effect 2.
fn fuzzy_rows(seed: u64) -> (TabularData, f64) {
    let n = grid_n(1_200);
    let half = n as f64 / 2.0 + 0.5;
    let running: Vec<f64> = (1..=n).map(|i| -1.0 + i as f64 / half).collect();
    let mut assign = uniform(seed, 0x51);
    let mut noise = gaussian(stream_seed(seed, 0x52));
    let treatment: Vec<f64> = running
        .iter()
        .map(|&x| {
            let probability = 0.1 + 0.02 * x + if x > 0.0 { 0.8 } else { 0.0 };
            f64::from(assign() < probability)
        })
        .collect();
    let outcome: Vec<f64> = running
        .iter()
        .zip(&treatment)
        .map(|(&x, &t)| 0.7 + 0.5 * x + 0.3 * x * x + 0.1 * x.powi(3) + TREATMENT * t + noise())
        .collect();
    (tabular(&running, &treatment, &outcome), TREATMENT)
}

/// Rows of independent observations on the same grid with a kinked dose
/// `1 + 0.2 x + 3 max(x, 0) + 0.02 noise` and the outcome
/// `0.7 + 0.5 x + 0.3 x^2 + 0.1 x^3 + 2 dose + 0.5 noise`; the slope-change ratio is 2.
///
/// The route refuses a first stage its 95% interval does not separate from zero, and the
/// selected bandwidth falls to about a third of the range, where the quartic slope change has
/// a standard error of 17 to 34 times the dose noise. The dose noise keeps the slope change
/// of 3 more than four standard errors from zero there at every grid size.
fn kink_rows(seed: u64) -> (TabularData, f64) {
    let n = grid_n(1_200);
    let half = n as f64 / 2.0 + 0.5;
    let running: Vec<f64> = (1..=n).map(|i| -1.0 + i as f64 / half).collect();
    let mut dose_noise = gaussian(stream_seed(seed, 0x53));
    let mut noise = gaussian(stream_seed(seed, 0x54));
    let treatment: Vec<f64> =
        running.iter().map(|&x| 1.0 + 0.2 * x + 3.0 * x.max(0.0) + 0.02 * dose_noise()).collect();
    let outcome: Vec<f64> = running
        .iter()
        .zip(&treatment)
        .map(|(&x, &t)| {
            0.7 + 0.5 * x + 0.3 * x * x + 0.1 * x.powi(3) + TREATMENT * t + 0.5 * noise()
        })
        .collect();
    (tabular(&running, &treatment, &outcome), TREATMENT)
}

/// Periods `1..=n` with a sharp treatment from the middle period on and AR(1) errors (`0.7`,
/// marginal standard deviation `0.7`) around a linear trend; the jump at the cutoff is 1.
fn time_jump_rows(seed: u64) -> (TabularData, LocalPolynomialRatioQuery, f64) {
    let n = grid_n(2_000);
    let cutoff = n as f64 / 2.0;
    let time: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let error = ar1_noise(n, 0.7, 0.7, stream_seed(seed, 0x55));
    let treatment: Vec<f64> = time.iter().map(|&t| f64::from(t >= cutoff)).collect();
    let outcome: Vec<f64> =
        (0..n).map(|i| 2.0 * time[i] / n as f64 + treatment[i] + error[i]).collect();
    let options =
        LocalRatioOptions { temporal_replicates: Some(199), ..LocalRatioOptions::default() };
    (
        tabular(&time, &treatment, &outcome),
        query(cutoff, 0.15 * n as f64, false, Some(options)),
        1.0,
    )
}

/// Periods `1..=n` with a dose that kinks at the middle period
/// (`1 + 0.5 s + 3 max(s, 0) + 0.02 noise`, `s` the time from the cutoff in bandwidths) and an
/// outcome `0.7 + 0.5 s + 0.3 s^2 + 0.1 s^3 + 2 dose` plus AR(1) errors (`0.7`, marginal
/// standard deviation `0.5`); the slope-change ratio is 2.
///
/// The route refuses a first stage its 95% interval does not separate from zero. The quartic
/// slope change has a standard error of 4.5 to 9 times the dose noise in bandwidth units over
/// the grid, so the dose noise keeps the slope change of 3 more than sixteen standard errors
/// from zero and the ratio well inside its declared strong-first-stage condition.
fn time_kink_rows(seed: u64) -> (TabularData, LocalPolynomialRatioQuery, f64) {
    let n = grid_n(2_000);
    let cutoff = n as f64 / 2.0;
    let bandwidth = 0.15 * n as f64;
    let time: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let error = ar1_noise(n, 0.7, 0.5, stream_seed(seed, 0x56));
    let mut dose_noise = gaussian(stream_seed(seed, 0x57));
    let scaled: Vec<f64> = time.iter().map(|&t| (t - cutoff) / bandwidth).collect();
    let treatment: Vec<f64> =
        scaled.iter().map(|&s| 1.0 + 0.5 * s + 3.0 * s.max(0.0) + 0.02 * dose_noise()).collect();
    let outcome: Vec<f64> = (0..n)
        .map(|i| {
            let s = scaled[i];
            0.7 + 0.5 * s + 0.3 * s * s + 0.1 * s.powi(3) + TREATMENT * treatment[i] + error[i]
        })
        .collect();
    let options =
        LocalRatioOptions { temporal_replicates: Some(199), ..LocalRatioOptions::default() };
    (tabular(&time, &treatment, &outcome), query(cutoff, bandwidth, true, Some(options)), TREATMENT)
}

/// A fixed-bandwidth Bayesian fuzzy jump drawn from the model's own likelihood: on an even grid
/// over `[-1, 1]` the dose is `0.2 + 0.1 x + 0.6 [x >= 0] + 0.3 noise` and the outcome
/// `1 + 0.3 x + 2 dose + 0.2 noise`, linear on each side of the cutoff; the jump ratio is 2.
fn bayesian_jump_rows(seed: u64) -> (TabularData, LocalPolynomialRatioQuery, f64) {
    let n = grid_n(400);
    let running: Vec<f64> = (0..n).map(|i| -1.0 + 2.0 * (i as f64 + 0.5) / n as f64).collect();
    let mut dose_noise = gaussian(stream_seed(seed, 0x58));
    let mut noise = gaussian(stream_seed(seed, 0x59));
    let treatment: Vec<f64> = running
        .iter()
        .map(|&x| 0.2 + 0.1 * x + if x >= 0.0 { 0.6 } else { 0.0 } + 0.3 * dose_noise())
        .collect();
    let outcome: Vec<f64> = running
        .iter()
        .zip(&treatment)
        .map(|(&x, &t)| 1.0 + 0.3 * x + TREATMENT * t + 0.2 * noise())
        .collect();
    (tabular(&running, &treatment, &outcome), query(0.0, 0.8, false, None), TREATMENT)
}

/// A fixed-bandwidth Bayesian regression kink drawn from the model's own likelihood: the dose is
/// `0.2 + 0.5 x + 0.6 max(x, 0) + 0.1 noise` and the outcome `1 + 2 dose + 0.2 noise`, linear on
/// each side of the cutoff; the slope-change ratio is 2.
fn bayesian_kink_rows(seed: u64) -> (TabularData, LocalPolynomialRatioQuery, f64) {
    let n = grid_n(400);
    let running: Vec<f64> = (0..n).map(|i| -1.0 + 2.0 * (i as f64 + 0.5) / n as f64).collect();
    let mut dose_noise = gaussian(stream_seed(seed, 0x5A));
    let mut noise = gaussian(stream_seed(seed, 0x5B));
    let treatment: Vec<f64> =
        running.iter().map(|&x| 0.2 + 0.5 * x + 0.6 * x.max(0.0) + 0.1 * dose_noise()).collect();
    let outcome: Vec<f64> =
        treatment.iter().map(|&t| 1.0 + TREATMENT * t + 0.2 * noise()).collect();
    (tabular(&running, &treatment, &outcome), query(0.0, 0.8, true, None), TREATMENT)
}

// ------------------------------------------------------- frequentist design layer

/// What a frequentist design-layer coordinate scores.
#[derive(Clone, Copy)]
struct Cell {
    test: &'static str,
    dgp: &'static str,
    /// Estimator the compiled plan must select.
    estimator: &'static str,
    /// Interval method the record names.
    interval: &'static str,
    /// Dependence the interval allows for.
    dependence: &'static str,
}

/// One replicate: the payload interval, the runtime-keyed construction and scope facts.
struct Scored {
    interval: Option<[f64; 2]>,
    construction: Construction,
    scope: ScopeFacts,
    validation: &'static str,
}

fn score_design(
    cell: Cell,
    data: TabularData,
    query: LocalPolynomialRatioQuery,
    seed: u64,
    first: bool,
) -> Option<Scored> {
    let (study, result) = run_study(data, query, None, seed, first)?;
    if first {
        assert_eq!(result.logical_plan.estimator.as_deref(), Some(cell.estimator));
    }
    let rollout = result.quasi_rollout()?;
    let interval = rollout.overall().and_then(|effect| effect.interval);
    let (mut construction, mut scope, _) =
        constructions(&study.inspect().ok()?, &result).into_iter().next()?;
    construction.query = QUERY.into();
    construction.structure = "design".into();
    construction.interval_method = cell.interval.into();
    construction.se_kind = String::new();
    construction.dependence = cell.dependence.into();
    construction.reported_level = REPORTED_LEVEL;
    scope.replicates_ok = (rollout.replicates_requested > 0).then_some(rollout.replicates_ok);
    Some(Scored { interval, construction, scope, validation: NO_VALIDATION })
}

fn measure_design(
    cell: Cell,
    truth: f64,
    seed_base: u64,
    build: impl Fn(u64) -> (TabularData, LocalPolynomialRatioQuery) + Sync,
) {
    let mut tally = CoverageTally::for_record(
        RecordKey { test: cell.test, dgp: cell.dgp, interval: cell.interval },
        REPORTED_LEVEL,
    );
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(seed_base, rep);
        let (data, query) = build(seed);
        score_design(cell, data, query, seed, rep == 0)
    });
    for run in runs {
        // A refusal or a withheld interval publishes nothing: a skip, never a free pass.
        let Some(Scored { interval: Some([lower, upper]), construction, scope, validation }) = run
        else {
            tally.skip();
            continue;
        };
        tally.bind_validation(&construction, scope, validation);
        tally.record(Some((lower, upper)), truth);
    }
    tally.assert();
}

const FUZZY_MSE: Cell = Cell {
    test: "fuzzy_rd_mse_optimal_rbc_interval_covers_the_local_ratio",
    dgp: "fuzzy_rows",
    estimator: "quasi.fuzzy_rd.mse_optimal_bandwidth_rbc",
    interval: "analytic_se",
    dependence: "iid",
};

/// Fuzzy jump at the MSE-optimal plug-in bandwidth (capped at the declared 0.5): the robust
/// bias-corrected HC0 delta interval against the treatment effect.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn fuzzy_rd_mse_optimal_rbc_interval_covers_the_local_ratio() {
    measure_design(FUZZY_MSE, TREATMENT, 0x1D_0001, |seed| {
        let (data, _) = fuzzy_rows(seed);
        let options = LocalRatioOptions {
            bandwidth_rule: LocalBandwidthRule::MseOptimal,
            ..LocalRatioOptions::default()
        };
        (data, query(0.0, 0.5, false, Some(options)))
    });
}

const KINK_MSE: Cell = Cell {
    test: "regression_kink_mse_optimal_rbc_interval_covers_the_slope_ratio",
    dgp: "kink_rows",
    estimator: "quasi.regression_kink.mse_optimal_bandwidth_rbc",
    interval: "analytic_se",
    dependence: "iid",
};

/// Regression kink at the MSE-optimal plug-in bandwidth (capped at the declared 1.0).
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn regression_kink_mse_optimal_rbc_interval_covers_the_slope_ratio() {
    measure_design(KINK_MSE, TREATMENT, 0x1D_0002, |seed| {
        let (data, _) = kink_rows(seed);
        let options = LocalRatioOptions {
            bandwidth_rule: LocalBandwidthRule::MseOptimal,
            ..LocalRatioOptions::default()
        };
        (data, query(0.0, 1.0, true, Some(options)))
    });
}

const FUZZY_DECLARED: Cell = Cell {
    test: "fuzzy_rd_declared_bandwidth_rbc_with_diagnostics_interval_covers_the_local_ratio",
    dgp: "fuzzy_rows",
    estimator: "quasi.fuzzy_rd.declared_bandwidth_rbc_with_diagnostics",
    interval: "analytic_se",
    dependence: "iid",
};

/// Fuzzy jump at the declared bandwidth 0.5 with the density test on.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn fuzzy_rd_declared_bandwidth_rbc_with_diagnostics_interval_covers_the_local_ratio() {
    measure_design(FUZZY_DECLARED, TREATMENT, 0x1D_0003, |seed| {
        let (data, _) = fuzzy_rows(seed);
        let options = LocalRatioOptions {
            bandwidth_rule: LocalBandwidthRule::Declared,
            density_test: true,
            ..LocalRatioOptions::default()
        };
        (data, query(0.0, 0.5, false, Some(options)))
    });
}

const TIME_JUMP: Cell = Cell {
    test: "temporal_fuzzy_rd_block_bootstrap_interval_covers_the_jump",
    dgp: "time_jump_rows",
    estimator: "quasi.fuzzy_rd.declared_bandwidth_rbc_block_bootstrap",
    interval: "bootstrap_se",
    dependence: "moving_block",
};

/// A sharp discontinuity in time under AR(1) errors: the moving-block bootstrap Student-t
/// interval.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_fuzzy_rd_block_bootstrap_interval_covers_the_jump() {
    measure_design(TIME_JUMP, 1.0, 0x1D_0004, |seed| {
        let (data, query, _) = time_jump_rows(seed);
        (data, query)
    });
}

const TIME_KINK: Cell = Cell {
    test: "temporal_regression_kink_block_bootstrap_interval_covers_the_slope_ratio",
    dgp: "time_kink_rows",
    estimator: "quasi.regression_kink.declared_bandwidth_rbc_block_bootstrap",
    interval: "bootstrap_se",
    dependence: "moving_block",
};

/// A kink in time under AR(1) errors: the moving-block bootstrap Student-t interval.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_regression_kink_block_bootstrap_interval_covers_the_slope_ratio() {
    measure_design(TIME_KINK, TREATMENT, 0x1D_0005, |seed| {
        let (data, query, _) = time_kink_rows(seed);
        (data, query)
    });
}

// ------------------------------------------------------------------ Bayesian

const POSTERIOR_DRAWS: usize = 2_000;
const TEMPORAL_DRAWS: usize = 500;
/// Diffuse coefficient prior, the one the route's own evidence tests declare.
const PRIOR_SCALE: f64 = 100.0;

/// A Bayesian coordinate: its record keys and the estimator the plan must select.
#[derive(Clone, Copy)]
struct BayesCell {
    test: &'static str,
    dgp: &'static str,
    estimator: &'static str,
}

fn keyed(cell: BayesCell, level: f64) -> CoverageTally {
    CoverageTally::for_record(
        RecordKey { test: cell.test, dgp: cell.dgp, interval: POSTERIOR_QUANTILE },
        level,
    )
}

fn measure_posterior(
    cell: BayesCell,
    truth: f64,
    seed_base: u64,
    draws: usize,
    build: impl Fn(u64) -> (TabularData, LocalPolynomialRatioQuery) + Sync,
) {
    let mut tallies = [keyed(cell, REPORTED_LEVEL), keyed(cell, GATE_LEVEL)];
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(seed_base, rep);
        let (data, query) = build(seed);
        let mode = InferenceMode::Bayesian(
            BayesianConfig::conjugate().prior_scale(PRIOR_SCALE).n_draws(draws),
        );
        let (study, result) = run_study(data, query, Some(mode), seed, rep == 0)?;
        if rep == 0 {
            assert_eq!(result.logical_plan.estimator.as_deref(), Some(cell.estimator));
        }
        let reported: Vec<_> = constructions(&study.inspect().ok()?, &result)
            .into_iter()
            .map(|(mut construction, scope, _)| {
                construction.query = QUERY.into();
                construction.structure = "design".into();
                (construction, scope, NO_VALIDATION)
            })
            .collect();
        // A weak first stage withholds the ratio draws, so no effect column and no interval.
        let intervals = result
            .posterior
            .as_ref()
            .and_then(|posterior| posterior.effect_column())
            .map(|column| posterior_pair(&result, column));
        Some((reported, intervals))
    });
    for scored in &runs {
        let Some((reported, Some(intervals))) = scored else {
            for tally in &mut tallies {
                tally.skip();
            }
            continue;
        };
        for (tally, interval) in tallies.iter_mut().zip(*intervals) {
            bind_reported(tally, reported);
            tally.record(interval, truth);
        }
    }
    gate(&tallies, &[None, None]);
}

/// Fixed-bandwidth fuzzy jump, Gaussian local-linear posterior of the ratio.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_fuzzy_rd_posterior_interval_covers_the_local_ratio() {
    let cell = BayesCell {
        test: "bayesian_fuzzy_rd_posterior_interval_covers_the_local_ratio",
        dgp: "bayesian_jump_rows",
        estimator: "bayesian.local_ratio",
    };
    measure_posterior(cell, TREATMENT, 0x1D_0101, POSTERIOR_DRAWS, |seed| {
        let (data, query, _) = bayesian_jump_rows(seed);
        (data, query)
    });
}

/// Fixed-bandwidth regression kink, Gaussian local-linear posterior of the slope-change ratio.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_regression_kink_posterior_interval_covers_the_slope_ratio() {
    let cell = BayesCell {
        test: "bayesian_regression_kink_posterior_interval_covers_the_slope_ratio",
        dgp: "bayesian_kink_rows",
        estimator: "bayesian.local_ratio",
    };
    measure_posterior(cell, TREATMENT, 0x1D_0102, POSTERIOR_DRAWS, |seed| {
        let (data, query, _) = bayesian_kink_rows(seed);
        (data, query)
    });
}

/// A sharp discontinuity in time under AR(1) errors: the within-side block Bayesian bootstrap
/// credible interval.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_fuzzy_rd_block_posterior_covers_the_jump() {
    let cell = BayesCell {
        test: "bayesian_temporal_fuzzy_rd_block_posterior_covers_the_jump",
        dgp: "time_jump_rows",
        estimator: "bayesian.local_ratio",
    };
    measure_posterior(cell, 1.0, 0x1D_0103, TEMPORAL_DRAWS, |seed| {
        let (data, query, _) = time_jump_rows(seed);
        (data, query)
    });
}

/// A kink in time under AR(1) errors: the within-side block Bayesian bootstrap credible interval.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_regression_kink_block_posterior_covers_the_slope_ratio() {
    let cell = BayesCell {
        test: "bayesian_temporal_regression_kink_block_posterior_covers_the_slope_ratio",
        dgp: "time_kink_rows",
        estimator: "bayesian.local_ratio",
    };
    measure_posterior(cell, TREATMENT, 0x1D_0104, TEMPORAL_DRAWS, |seed| {
        let (data, query, _) = time_kink_rows(seed);
        (data, query)
    });
}
