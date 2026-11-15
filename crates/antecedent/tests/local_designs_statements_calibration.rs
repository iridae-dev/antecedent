//! Coverage of the further statements of the local-ratio design routes: the kink at a declared
//! bandwidth with diagnostics, the discontinuity and kink in time at the selected bandwidth,
//! a fuzzy step in time, second-order autoregressive errors, and the Bayesian counterparts.
//! The target is the ratio of the outcome's jump (or slope change) to the treatment's.
//!
//! Builders mirror the family's first harness file, parametrized where a statement needs it.
//! Records are keyed on the design axes (structure `design`, validation `not_applicable`). A
//! replicate that publishes nothing to score is a skip (a miss).
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
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

/// Stationary AR(2) noise `e_t = phi1 e_{t-1} + phi2 e_{t-2} + innovation` with marginal
/// standard deviation `sd`, after a burn-in of 500 periods.
///
/// The marginal variance of an AR(2) is `s^2 (1 - phi2) / ((1 + phi2) ((1 - phi2)^2 - phi1^2))`
/// for innovation variance `s^2`, which fixes the innovation scale.
fn ar2_noise(n: usize, phi1: f64, phi2: f64, sd: f64, seed: u64) -> Vec<f64> {
    const BURN_IN: usize = 500;
    let mut z = gaussian(seed);
    let innovation =
        sd * ((1.0 + phi2) * ((1.0 - phi2).powi(2) - phi1 * phi1) / (1.0 - phi2)).sqrt();
    let (mut previous, mut before) = (0.0, 0.0);
    let mut out = Vec::with_capacity(n);
    for period in 0..BURN_IN + n {
        let e = phi1 * previous + phi2 * before + innovation * z();
        before = previous;
        previous = e;
        if period >= BURN_IN {
            out.push(e);
        }
    }
    out
}

/// The sharp step of [`time_jump_rows`] with AR(2) errors (`0.5`, `0.3`, marginal standard
/// deviation `0.7`): dependence beyond the first-order autoregression the interval models.
/// The jump is the coefficient 1 of the step, whatever the law of the errors.
fn time_jump_ar2_rows(seed: u64) -> (TabularData, LocalPolynomialRatioQuery, f64) {
    let n = grid_n(2_000);
    let cutoff = n as f64 / 2.0;
    let time: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let error = ar2_noise(n, 0.5, 0.3, 0.7, stream_seed(seed, 0x5C));
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

/// The kink of [`time_kink_rows`] with AR(2) errors (`0.5`, `0.3`, marginal standard deviation
/// `0.5`): dependence beyond the first-order autoregression, whose slope-change ratio is 2.
fn time_kink_ar2_rows(seed: u64) -> (TabularData, LocalPolynomialRatioQuery, f64) {
    let n = grid_n(2_000);
    let cutoff = n as f64 / 2.0;
    let bandwidth = 0.15 * n as f64;
    let time: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let error = ar2_noise(n, 0.5, 0.3, 0.5, stream_seed(seed, 0x5F));
    let mut dose_noise = gaussian(stream_seed(seed, 0x60));
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

/// Periods `1..=n` with a fuzzy step at the middle period: the treatment is received with
/// probability `0.1` before it and `0.9` from it on, independently by period, and the outcome
/// is a linear trend plus `2 t` plus AR(1) errors (`0.7`, marginal standard deviation `0.7`).
/// The outcome jumps by `2 * 0.8` and the uptake by `0.8`, so the ratio is 2.
fn time_fuzzy_rows(seed: u64) -> (TabularData, LocalPolynomialRatioQuery, f64) {
    let n = grid_n(2_000);
    let cutoff = n as f64 / 2.0;
    let time: Vec<f64> = (1..=n).map(|i| i as f64).collect();
    let error = ar1_noise(n, 0.7, 0.7, stream_seed(seed, 0x5D));
    let mut assign = uniform(seed, 0x5E);
    let treatment: Vec<f64> =
        time.iter().map(|&t| f64::from(assign() < if t >= cutoff { 0.9 } else { 0.1 })).collect();
    let outcome: Vec<f64> =
        (0..n).map(|i| 2.0 * time[i] / n as f64 + TREATMENT * treatment[i] + error[i]).collect();
    let options =
        LocalRatioOptions { temporal_replicates: Some(199), ..LocalRatioOptions::default() };
    (
        tabular(&time, &treatment, &outcome),
        query(cutoff, 0.15 * n as f64, false, Some(options)),
        TREATMENT,
    )
}

/// The same design at the MSE-optimal plug-in bandwidth, capped at the declared one. The
/// target is unchanged: the ratio at the cutoff does not depend on the window.
fn at_selected_bandwidth(mut query: LocalPolynomialRatioQuery) -> LocalPolynomialRatioQuery {
    if let Some(options) = query.options.as_mut() {
        options.bandwidth_rule = LocalBandwidthRule::MseOptimal;
    }
    query
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

const KINK_DECLARED: Cell = Cell {
    test: "regression_kink_declared_bandwidth_rbc_with_diagnostics_interval_covers_the_slope_ratio",
    dgp: "kink_rows",
    estimator: "quasi.regression_kink.declared_bandwidth_rbc_with_diagnostics",
    interval: "analytic_se",
    dependence: "iid",
};

/// Regression kink at the declared bandwidth 1.0 with the density test on. The outcome is a
/// cubic plus twice a piecewise-linear dose, which the quartic fit reproduces at any window,
/// so the slope-change ratio is 2.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn regression_kink_declared_bandwidth_rbc_with_diagnostics_interval_covers_the_slope_ratio() {
    measure_design(KINK_DECLARED, TREATMENT, 0x1D_0006, |seed| {
        let (data, _) = kink_rows(seed);
        let options = LocalRatioOptions {
            bandwidth_rule: LocalBandwidthRule::Declared,
            density_test: true,
            ..LocalRatioOptions::default()
        };
        (data, query(0.0, 1.0, true, Some(options)))
    });
}

const TIME_JUMP_SELECTED: Cell = Cell {
    test: "temporal_fuzzy_rd_selected_bandwidth_block_bootstrap_interval_covers_the_jump",
    dgp: "time_jump_rows",
    estimator: "quasi.fuzzy_rd.mse_optimal_bandwidth_rbc_block_bootstrap",
    interval: "bootstrap_se",
    dependence: "moving_block",
};

/// A sharp discontinuity in time at the selected bandwidth: the block bootstrap interval of
/// the fit at the MSE-optimal window, capped at the declared one.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_fuzzy_rd_selected_bandwidth_block_bootstrap_interval_covers_the_jump() {
    measure_design(TIME_JUMP_SELECTED, 1.0, 0x1D_0007, |seed| {
        let (data, query, _) = time_jump_rows(seed);
        (data, at_selected_bandwidth(query))
    });
}

const TIME_KINK_SELECTED: Cell = Cell {
    test: "temporal_regression_kink_selected_bandwidth_block_bootstrap_interval_covers_the_slope_ratio",
    dgp: "time_kink_rows",
    estimator: "quasi.regression_kink.mse_optimal_bandwidth_rbc_block_bootstrap",
    interval: "bootstrap_se",
    dependence: "moving_block",
};

/// A kink in time at the selected bandwidth.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_regression_kink_selected_bandwidth_block_bootstrap_interval_covers_the_slope_ratio() {
    measure_design(TIME_KINK_SELECTED, TREATMENT, 0x1D_0008, |seed| {
        let (data, query, _) = time_kink_rows(seed);
        (data, at_selected_bandwidth(query))
    });
}

const TIME_FUZZY: Cell = Cell {
    test: "temporal_fuzzy_step_block_bootstrap_interval_covers_the_local_ratio",
    dgp: "time_fuzzy_rows",
    estimator: "quasi.fuzzy_rd.declared_bandwidth_rbc_block_bootstrap",
    interval: "bootstrap_se",
    dependence: "moving_block",
};

/// A fuzzy step in time under AR(1) errors: the block bootstrap redraws the uptake errors as
/// well as the outcome's.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_fuzzy_step_block_bootstrap_interval_covers_the_local_ratio() {
    measure_design(TIME_FUZZY, TREATMENT, 0x1D_0009, |seed| {
        let (data, query, _) = time_fuzzy_rows(seed);
        (data, query)
    });
}

const TIME_JUMP_AR2: Cell = Cell {
    test: "temporal_fuzzy_rd_block_bootstrap_interval_covers_the_jump_under_ar2_errors",
    dgp: "time_jump_ar2_rows",
    estimator: "quasi.fuzzy_rd.declared_bandwidth_rbc_block_bootstrap",
    interval: "bootstrap_se",
    dependence: "moving_block",
};

/// A sharp discontinuity in time under AR(2) errors, which the first-order prewhitening
/// leaves dependent; only the blocks carry the rest.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_fuzzy_rd_block_bootstrap_interval_covers_the_jump_under_ar2_errors() {
    measure_design(TIME_JUMP_AR2, 1.0, 0x1D_000A, |seed| {
        let (data, query, _) = time_jump_ar2_rows(seed);
        (data, query)
    });
}
const TIME_KINK_AR2: Cell = Cell {
    test: "temporal_regression_kink_block_bootstrap_interval_covers_the_slope_ratio_under_ar2_errors",
    dgp: "time_kink_ar2_rows",
    estimator: "quasi.regression_kink.declared_bandwidth_rbc_block_bootstrap",
    interval: "bootstrap_se",
    dependence: "moving_block",
};

/// A kink in time under AR(2) errors at the declared bandwidth.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn temporal_regression_kink_block_bootstrap_interval_covers_the_slope_ratio_under_ar2_errors() {
    measure_design(TIME_KINK_AR2, TREATMENT, 0x1D_000B, |seed| {
        let (data, query, _) = time_kink_ar2_rows(seed);
        (data, query)
    });
}
// ------------------------------------------------------------------ Bayesian

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
            .and_then(antecedent::CausalPosterior::effect_column)
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

/// A sharp discontinuity in time at the selected bandwidth: the block Bayesian bootstrap at the
/// MSE-optimal window, capped at the declared one.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_fuzzy_rd_selected_bandwidth_block_posterior_covers_the_jump() {
    let cell = BayesCell {
        test: "bayesian_temporal_fuzzy_rd_selected_bandwidth_block_posterior_covers_the_jump",
        dgp: "time_jump_rows",
        estimator: "bayesian.local_ratio",
    };
    measure_posterior(cell, 1.0, 0x1D_0105, TEMPORAL_DRAWS, |seed| {
        let (data, query, _) = time_jump_rows(seed);
        (data, at_selected_bandwidth(query))
    });
}

/// A kink in time at the selected bandwidth.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_regression_kink_selected_bandwidth_block_posterior_covers_the_slope_ratio() {
    let cell = BayesCell {
        test: "bayesian_temporal_regression_kink_selected_bandwidth_block_posterior_covers_the_slope_ratio",
        dgp: "time_kink_rows",
        estimator: "bayesian.local_ratio",
    };
    measure_posterior(cell, TREATMENT, 0x1D_0106, TEMPORAL_DRAWS, |seed| {
        let (data, query, _) = time_kink_rows(seed);
        (data, at_selected_bandwidth(query))
    });
}

/// A fuzzy step in time under AR(1) errors: the block Bayesian bootstrap redraws the uptake
/// errors as well as the outcome's.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_fuzzy_step_block_posterior_covers_the_local_ratio() {
    let cell = BayesCell {
        test: "bayesian_temporal_fuzzy_step_block_posterior_covers_the_local_ratio",
        dgp: "time_fuzzy_rows",
        estimator: "bayesian.local_ratio",
    };
    measure_posterior(cell, TREATMENT, 0x1D_0107, TEMPORAL_DRAWS, |seed| {
        let (data, query, _) = time_fuzzy_rows(seed);
        (data, query)
    });
}

/// A sharp discontinuity in time under AR(2) errors at the declared bandwidth: the block
/// Bayesian bootstrap with the autoregressive order chosen from the data.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_fuzzy_rd_block_posterior_covers_the_jump_under_ar2_errors() {
    let cell = BayesCell {
        test: "bayesian_temporal_fuzzy_rd_block_posterior_covers_the_jump_under_ar2_errors",
        dgp: "time_jump_ar2_rows",
        estimator: "bayesian.local_ratio",
    };
    measure_posterior(cell, 1.0, 0x1D_0108, TEMPORAL_DRAWS, |seed| {
        let (data, query, _) = time_jump_ar2_rows(seed);
        (data, query)
    });
}

/// A kink in time under AR(2) errors at the declared bandwidth.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn bayesian_temporal_regression_kink_block_posterior_covers_the_slope_ratio_under_ar2_errors() {
    let cell = BayesCell {
        test: "bayesian_temporal_regression_kink_block_posterior_covers_the_slope_ratio_under_ar2_errors",
        dgp: "time_kink_ar2_rows",
        estimator: "bayesian.local_ratio",
    };
    measure_posterior(cell, TREATMENT, 0x1D_0109, TEMPORAL_DRAWS, |seed| {
        let (data, query, _) = time_kink_ar2_rows(seed);
        (data, query)
    });
}
