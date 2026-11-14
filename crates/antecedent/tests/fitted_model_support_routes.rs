//! Retained fitted-model attribution routes and the scope their intervals state.
//!
//! Each test prepares a fitted-model attribution, checks that the handle retains the sealed
//! attribution operation, executes it, compares the published point with the known change of
//! the generating law, and reads the `gcm.model.interval_scope` diagnostic: inside the
//! conditions the coverage harness ran under the result says so, and outside them (too few
//! draws, replicates, rows or blocks) it states that the interval has no coverage measurement.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::RetainedOperation;
use antecedent::{BayesianConfig, EstimatorId, InferenceMode, RefuteSuite, StudyResult};
use antecedent::{IdentifierId, Study};
use antecedent_core::{
    AttributeQuery, AttributeTarget, CausalQuery, ChangeAttributionQuery, ChangeEstimator,
    ChangeMeasure, ExecutionContext, Lag, PopulationSelector, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_graph::{Dag, DenseNodeId, TemporalDag, ensure_lagged};

/// Intercept shift of the outcome between the two populations.
const SHIFT: f64 = 2.0;
/// Slope of the cause.
const SLOPE: f64 = 2.0;
/// Residual scale.
const NOISE: f64 = 0.7;

/// A fixed-seed standard normal stream (splitmix64 and Box-Muller).
struct Normals(u64);

impl Normals {
    fn uniform(&mut self) -> f64 {
        self.0 = self.0.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.0;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        // 53 random bits in (0, 1).
        ((z >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
    }

    fn next(&mut self) -> f64 {
        let radius = (-2.0 * self.uniform().ln()).sqrt();
        radius * (std::f64::consts::TAU * self.uniform()).cos()
    }
}

/// `x ~ N(0, 1)` and `y = c + 2 x + 0.7 e`, with `c` shifted by [`SHIFT`] from row `pop` on. With
/// `lagged`, `y` reads the previous row's `x`.
fn columns(pop: usize, seed: u64, lagged: bool) -> (Vec<f64>, Vec<f64>) {
    let mut normal = Normals(seed);
    let (mut x, mut y) = (Vec::with_capacity(2 * pop), Vec::with_capacity(2 * pop));
    let mut previous = normal.next();
    for row in 0..2 * pop {
        let cause = normal.next();
        let intercept = if row < pop { 1.0 } else { 1.0 + SHIFT };
        let read = if lagged { previous } else { cause };
        x.push(cause);
        y.push(intercept + SLOPE * read + NOISE * normal.next());
        previous = cause;
    }
    (x, y)
}

fn table(pop: usize, seed: u64) -> TabularData {
    let (x, y) = columns(pop, seed, false);
    TabularData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())]).unwrap()
}

fn series(pop: usize, seed: u64) -> TimeSeriesData {
    let (x, y) = columns(pop, seed, true);
    TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

fn dag() -> Dag {
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph
}

/// `x_{t-1} -> y_t`.
fn lag_dag() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let parent = ensure_lagged(&mut graph, VariableId::from_raw(0), Lag::from_raw(1)).unwrap();
    let child = ensure_lagged(&mut graph, VariableId::from_raw(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(parent, child).unwrap();
    graph
}

/// The mean-difference change of `y` between the first and the second `pop` rows.
fn change_query(pop: usize) -> CausalQuery {
    CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::DistributionChange {
        change: ChangeAttributionQuery::new(
            VariableId::from_raw(1),
            PopulationSelector::TimeRange { start: 0, end: pop },
            PopulationSelector::TimeRange { start: pop, end: 2 * pop },
        ),
        estimator: ChangeEstimator::Sampled {
            measure: ChangeMeasure::MeanDifference,
            n_samples: 2_000,
            seed: 3,
        },
    }))
}

fn bayesian(draws: usize) -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(draws))
}

/// The value of the `coverage_scope` field of the result's interval-scope diagnostic.
fn coverage_scope(result: &StudyResult) -> String {
    let diagnostic = result
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code.as_ref() == "gcm.model.interval_scope")
        .expect("a fitted-model interval states its scope");
    diagnostic
        .fields
        .iter()
        .find(|(key, _)| key.as_ref() == "coverage_scope")
        .map(|(_, value)| value.to_string())
        .expect("the scope diagnostic carries its coverage scope")
}

/// Run `body` on a thread with the stack the prepared dispatcher needs in debug builds.
fn on_large_stack(name: &str, body: fn()) {
    std::thread::Builder::new()
        .name(name.into())
        .stack_size(16 * 1024 * 1024)
        .spawn(body)
        .expect("spawn evidence thread")
        .join()
        .expect("evidence body panicked");
}

/// Prepare and execute a static change attribution; the handle retains the attribution
/// operation with `estimator`.
fn run_static(
    pop: usize,
    inference: InferenceMode,
    replicates: u32,
    estimator: EstimatorId,
) -> StudyResult {
    let ctx = ExecutionContext::for_tests(41);
    let data = table(pop, 0xF17_5EED);
    let query = change_query(pop);
    let study = Study::tabular(data.clone())
        .graph(dag())
        .query(query.clone())
        .inference(inference)
        .bootstrap_replicates(replicates)
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let RetainedOperation::Attribution(plan) = prepared.retained_operation() else {
        panic!("a fitted-model attribution with uncertainty retains the attribution operation")
    };
    assert_eq!(plan.query, query);
    assert_eq!(plan.identifier, IdentifierId::GcmParametric);
    assert_eq!(plan.estimator, estimator);
    prepared.estimate(&data, &ctx).unwrap()
}

/// Prepare and execute a change attribution on a lag-one series.
fn run_series(
    pop: usize,
    inference: InferenceMode,
    replicates: u32,
    estimator: EstimatorId,
) -> StudyResult {
    let ctx = ExecutionContext::for_tests(43);
    let data = series(pop, 0xF17_5EED);
    let query = change_query(pop);
    let study = Study::series(data.clone())
        .graph(lag_dag())
        .query(query.clone())
        .inference(inference)
        .bootstrap_replicates(replicates)
        .refute(RefuteSuite::None)
        .build()
        .unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let RetainedOperation::Attribution(plan) = prepared.retained_operation() else {
        panic!("a temporal fitted-model attribution retains the attribution operation")
    };
    assert_eq!(plan.query, query);
    assert_eq!(plan.estimator, estimator);
    prepared.estimate_series(&data, &ctx).unwrap()
}

#[test]
fn a_row_weight_change_posterior_runs_retained_and_states_its_measured_scope() {
    on_large_stack("fitted-model-static-posterior", || {
        let inside = run_static(300, bayesian(1_000), 0, EstimatorId::GcmFitBayesian);
        let posterior = inside.posterior.as_ref().expect("a posterior over the mechanisms");
        assert_eq!(posterior.draws.n_draws, 1_000);
        // The total, then the mechanisms of x (unchanged) and y (the whole shift). The total
        // also carries the difference of the two populations' means of x: its sampling
        // standard deviation is sqrt(2 * (4 + 0.49) / 300) = 0.17.
        assert!((posterior.summaries.mean[0] - SHIFT).abs() < 0.6);
        assert!(posterior.summaries.mean[1].abs() < 0.6);
        assert!((posterior.summaries.mean[2] - SHIFT).abs() < 0.3);
        // Sampling standard deviation of the shift is 0.7 * sqrt(2 / 300) = 0.057.
        assert!((posterior.summaries.sd[2] - 0.057).abs() < 0.02);
        assert_eq!(coverage_scope(&inside), "measured_conditions");

        // Fewer draws than the coverage harness ran: the interval states no measurement.
        let few_draws = run_static(300, bayesian(24), 0, EstimatorId::GcmFitBayesian);
        assert!((few_draws.posterior.as_ref().unwrap().summaries.mean[0] - SHIFT).abs() < 0.6);
        assert_eq!(coverage_scope(&few_draws), "not_measured");

        // Fewer rows per population than the smallest measured grid point.
        let few_rows = run_static(80, bayesian(1_000), 0, EstimatorId::GcmFitBayesian);
        assert!((few_rows.posterior.as_ref().unwrap().summaries.mean[0] - SHIFT).abs() < 1.2);
        assert_eq!(coverage_scope(&few_rows), "not_measured");
    });
}

#[test]
fn a_block_bootstrap_change_interval_runs_retained_and_states_its_measured_scope() {
    on_large_stack("fitted-model-temporal-bootstrap", || {
        let inside = run_series(301, InferenceMode::Frequentist, 200, EstimatorId::GcmFit);
        let effect = inside.estimate().as_effect().expect("the total change");
        // Sampling standard deviation of the total change: sqrt(2 * 4.49 / 300) = 0.17.
        assert!((effect.ate - SHIFT).abs() < 0.6);
        let se = effect.se_bootstrap().expect("a block-bootstrap standard error");
        assert!((se - 0.17).abs() < 0.06);
        assert_eq!(coverage_scope(&inside), "measured_conditions");

        // Fewer replicates than the coverage harness ran.
        let few = run_series(301, InferenceMode::Frequentist, 20, EstimatorId::GcmFit);
        assert!((few.estimate().as_effect().unwrap().ate - SHIFT).abs() < 0.6);
        assert_eq!(coverage_scope(&few), "not_measured");
    });
}

#[test]
fn a_block_weight_change_posterior_runs_retained_and_states_its_measured_scope() {
    on_large_stack("fitted-model-temporal-posterior", || {
        let inside = run_series(301, bayesian(1_000), 0, EstimatorId::GcmFitBayesian);
        let posterior = inside.posterior.as_ref().expect("a block-weight posterior");
        assert!((posterior.summaries.mean[0] - SHIFT).abs() < 0.6);
        assert!((posterior.summaries.sd[0] - 0.17).abs() < 0.06);
        assert_eq!(coverage_scope(&inside), "measured_conditions");

        // Too short a series for the smallest measured block count per population.
        let short = run_series(60, bayesian(1_000), 0, EstimatorId::GcmFitBayesian);
        assert!((short.posterior.as_ref().unwrap().summaries.mean[0] - SHIFT).abs() < 1.4);
        assert_eq!(coverage_scope(&short), "not_measured");
    });
}
