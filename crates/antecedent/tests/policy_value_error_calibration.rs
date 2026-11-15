//! The Monte Carlo standard error a policy value publishes against the spread of the estimate it
//! describes, and the coverage of the Monte Carlo interval of the simulated expectation.
//!
//! A policy value draws `n` rows of the fitted model under the policy, replays the same noise
//! with no intervention, and reports the policy mean, the baseline mean and their paired
//! difference, each with its simulation standard error. At a fixed fitted model that error is
//! the whole uncertainty of the number (the model is the truth, the fit is not re-estimated), so
//! the claim is checkable against repeated runs on independent seeds:
//!
//! * The mean over seeds of the published standard error equals the standard deviation of the
//!   estimate across seeds. A standard deviation estimated from `SEEDS` independent draws has
//!   relative standard deviation `1 / sqrt(2 (SEEDS - 1))` (the chi-square spread of a sample
//!   variance), so the ratio is held to `1 +- Z / sqrt(2 (SEEDS - 1))`.
//! * The interval `estimate +- 1.96 se` holds the model's exact expectation, taken from one run
//!   of `REFERENCE_DRAWS` draws whose own error is a few percent of the error of one run, in at
//!   least the nominal share of the seeds, less `Z` binomial standard deviations of that share.
//!
//! The row-bootstrap standard error and the Dirichlet row-weight posterior of the same value
//! are measured by `simulate_policy_value_dag_frequentist_bootstrap_coverage` and
//! `simulate_policy_value_dag_bayesian_dirichlet_coverage` in `fitted_model_calibration`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::gcm::{FittedGcm, fit_gcm, policy_value};
use antecedent_core::{ExecutionContext, Intervention, Value, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};

/// Independent simulation seeds.
const SEEDS: u64 = 200;
/// Draws of one policy-value run.
const DRAWS: usize = 400;
/// Draws of the reference run that stands for the exact expectation of the fitted model.
const REFERENCE_DRAWS: usize = 200_000;
/// Half-width of every acceptance band, in standard deviations of the statistic it bounds.
const Z: f64 = 4.0;

/// Two variables, `price -> demand`, with a fixed deterministic table fitted once.
fn fitted_model() -> FittedGcm {
    let price: Vec<f64> =
        (0..600_u32).map(|i| 11.0 + (f64::from((i * 37) % 101) - 50.0) * 0.04).collect();
    let demand: Vec<f64> = (0..600_u32)
        .zip(&price)
        .map(|(i, p)| 50.0 - 0.8 * p + (f64::from((i * 13) % 23) - 11.0) * 0.05)
        .collect();
    let data =
        TabularData::from_f64_columns([("price", price.as_slice()), ("demand", demand.as_slice())])
            .unwrap();
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    fit_gcm(graph, &data).unwrap()
}

/// The three published quantities of one run: (estimate, standard error) of the policy mean, the
/// baseline mean and the paired difference.
type Run = [(f64, f64); 3];

fn run(fitted: &FittedGcm, program: &[Intervention], outcome: VariableId, n: usize, seed: u64) -> Run {
    let value = policy_value(&fitted.model, program, outcome, n, &ExecutionContext::for_tests(seed))
        .unwrap();
    [
        (value.policy_mean, value.policy_simulation_stderr),
        (value.baseline_mean, 0.0),
        (value.mean_difference, value.difference_simulation_stderr),
    ]
}

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

fn sample_sd(values: &[f64]) -> f64 {
    let m = mean(values);
    (values.iter().map(|v| (v - m) * (v - m)).sum::<f64>() / (values.len() as f64 - 1.0)).sqrt()
}

struct Case {
    fitted: FittedGcm,
    program: Vec<Intervention>,
    outcome: VariableId,
}

/// A hard level of the price read on demand: the policy mean is the fitted line at the level
/// plus the fitted noise mean, and the baseline is the unintervened demand on the same noise.
fn case() -> Case {
    Case {
        fitted: fitted_model(),
        program: vec![Intervention::set(VariableId::from_raw(0), Value::f64(14.0))],
        outcome: VariableId::from_raw(1),
    }
}

#[test]
fn the_published_simulation_error_matches_the_spread_of_the_estimate_across_seeds() {
    let case = case();
    let runs: Vec<Run> =
        (0..SEEDS).map(|s| run(&case.fitted, &case.program, case.outcome, DRAWS, 10_000 + s)).collect();
    let band = Z / (2.0 * (SEEDS as f64 - 1.0)).sqrt();
    // The policy mean and the paired difference publish an error; the baseline is a mean of the
    // same noise and shares the policy mean's spread, so it is read through the difference.
    for (index, name) in [(0, "policy_mean"), (2, "mean_difference")] {
        let estimates: Vec<f64> = runs.iter().map(|r| r[index].0).collect();
        let published = mean(&runs.iter().map(|r| r[index].1).collect::<Vec<_>>());
        let spread = sample_sd(&estimates);
        assert!(spread > 0.0, "{name}: the estimate must vary across seeds");
        let ratio = published / spread;
        assert!(
            (ratio - 1.0).abs() <= band,
            "{name}: mean published error {published} against spread {spread} (ratio {ratio}, band {band})"
        );
    }
}

#[test]
fn the_simulation_interval_holds_the_exact_expectation_at_its_nominal_share() {
    let case = case();
    let reference = run(&case.fitted, &case.program, case.outcome, REFERENCE_DRAWS, 1);
    let nominal = 0.95_f64;
    let floor = nominal - Z * (nominal * (1.0 - nominal) / SEEDS as f64).sqrt();
    let runs: Vec<Run> =
        (0..SEEDS).map(|s| run(&case.fitted, &case.program, case.outcome, DRAWS, 20_000 + s)).collect();
    for (index, name) in [(0, "policy_mean"), (2, "mean_difference")] {
        let truth = reference[index].0;
        let held = runs
            .iter()
            .filter(|r| (r[index].0 - truth).abs() <= 1.959_963_984_540_054 * r[index].1)
            .count();
        let share = held as f64 / SEEDS as f64;
        assert!(share >= floor, "{name}: the interval held the expectation in {share}, below {floor}");
    }
}
