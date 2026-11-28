//! Repeated-sampling coverage of the uncertainty a fitted-model attribution, simulation or
//! structure search publishes beside its point answer.
//!
//! Every truth is the exact population value of the published quantity on a linear-Gaussian
//! law:
//!
//! * A change attribution contrasts two stacked populations (or two time ranges of one series)
//!   that differ only in the target's intercept: the total change is the intercept shift, the
//!   mechanism of the unchanged cause carries none of it and the mechanism of the target all of
//!   it.
//! * A path decomposition of `x -> y` is the slope; the mean anomaly score against a fixed
//!   reference equal to the target's true location and scale is the mean of an `Exp(1)`
//!   variable, 1.
//! * The feature relevance of `x` for `y` sets `x` to its observed mean plus one and holds that
//!   level fixed across refits, so its target is the effect of that fixed level against the
//!   natural law of `x` (mean 0): the slope times one plus the observed mean of `x`.
//! * On `z -> t -> y`, `z -> y` the variance of `y` splits into the squared total effects of the
//!   three noise terms (3.24, 2.25, 1), `do(t = 1)` moves the mean of `y` by 1.5 and a shift of
//!   `t` by 0.5 moves it by 0.75.
//! * A temporal rollout that holds `x` at 1 puts `y` at `intercept + slope` from the second
//!   step on.
//! * A structure posterior is scored by whether its smallest set of at least the stated mass
//!   holds the true structure; the posterior of a mechanism change is scored the same way on the
//!   indicator of the change, drawn from the prior the query declares.
//!
//! Tests that score an interval the result binds read the construction behind each record from
//! the runtime's calibration key. These routes have no support-matrix cell, so the runtime
//! reports no validation mode; the records are bound with validation `not_applicable`, as the
//! other routes off the matrix are. The posterior of an intrinsic influence is held to at least
//! nominal: every refit draw carries its own nested-simulation noise, which only widens the
//! interval. The statements a route publishes in its payload only (the whole-procedure interval
//! of an intrinsic influence, the posterior probability of a mechanism change, the structure
//! posteriors) have no runtime calibration key: their records name the construction here, as
//! `measurement_composition_calibration.rs` does for the measurement routes.
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

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

use antecedent::discovery::{BayesianDiscoverParams, GraphMcmcSchedule};
use antecedent::{
    AttributionOutcome, BayesianConfig, CompiledExecution, DiscoveredStructure, DiscoverySpec,
    InferenceMode, RefuteSuite, Study, StudyBuilder, StudyResult,
};
use antecedent_attribution::ComponentStatus;
use antecedent_core::{
    AnomalyAttributionQuery, AnomalyReference, AttributeQuery, AttributeTarget, CausalQuery,
    ChangeAttributionQuery, ChangeEstimator, ChangeMeasure, ExecutionContext, InfluenceAllocation,
    Intervention, InterventionSequence, IntrinsicInfluenceTarget, Lag, MechanismChangeQuery,
    MechanismChangeTest, NoiseDeclaration, PopulationSelector, SequencedIntervention,
    SimulateQuery, SimulationTarget, TemporalMaterializationLimits, TemporalPolicy, Value,
    VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_discovery::adjacency_mask_from_cpdag;
use antecedent_graph::{Cpdag, Dag, DenseNodeId, TemporalDag, ensure_lagged};
use common::calibration::{
    Construction, CoverageSide, CoverageTally, RecordKey, ScopeFacts, gaussian, grid_n,
    map_replicates, n_sim, stream_seed, unit_uniform,
};
use common::calibration_bind::constructions;
use common::reported::{GATE_LEVEL, REPORTED_LEVEL, gate, posterior_pair, scalar_normal_pair};
use grid_cell::{Cell, IntervalPair, Target};

/// Validation mode of every record of this file: the routes have no support-matrix cell.
const NO_VALIDATION: &str = "not_applicable";

/// Measure `cell` over `n_sim()` replicates and gate every record at nominal.
///
/// `run(replicate, 0)` executes the route on the replicate's dataset and `score` reads one
/// interval pair per target, in `targets` order. The construction of each record is the one the
/// runtime reports for the scored interval method.
fn measure(
    cell: Cell,
    capability: Option<&str>,
    targets: &[Target],
    run: impl Fn(u64, usize) -> (Study, StudyResult) + Sync,
    score: impl Fn(&StudyResult) -> Vec<IntervalPair> + Sync,
) {
    let fixed: Vec<f64> = targets.iter().map(|target| target.truth).collect();
    measure_sample_truths(cell, capability, targets, |_| fixed.clone(), run, score);
}

/// [`measure`] for a functional whose true value depends on the replicate's sample:
/// `truths(replicate)` gives one truth per target, in `targets` order.
fn measure_sample_truths(
    cell: Cell,
    capability: Option<&str>,
    targets: &[Target],
    truths: impl Fn(u64) -> Vec<f64> + Sync,
    run: impl Fn(u64, usize) -> (Study, StudyResult) + Sync,
    score: impl Fn(&StudyResult) -> Vec<IntervalPair> + Sync,
) {
    let runs = map_replicates(n_sim(), |rep| {
        let truth = truths(rep);
        assert_eq!(truth.len(), targets.len(), "{}: one truth per target", cell.test);
        let (study, result) = run(rep, 0);
        assert_eq!(
            result.logical_plan.estimator.as_deref(),
            Some(cell.estimator),
            "{}: estimator target",
            cell.test
        );
        let intervals = score(&result);
        assert_eq!(intervals.len(), targets.len(), "{}: one interval pair per target", cell.test);
        let (construction, scope, _) = constructions(&study.inspect().expect("inspect"), &result)
            .into_iter()
            .find(|(construction, _, _)| construction.interval_method == cell.interval)
            .unwrap_or_else(|| panic!("{}: no {} interval was reported", cell.test, cell.interval));
        (construction, scope, intervals, truth)
    });
    let key = RecordKey { test: cell.test, dgp: cell.dgp, interval: cell.interval };
    let mut tallies: Vec<CoverageTally> = Vec::new();
    for (index, target) in targets.iter().enumerate() {
        for (slot, level) in [REPORTED_LEVEL, GATE_LEVEL].into_iter().enumerate() {
            let mut tally = CoverageTally::for_record(key, level);
            if let Some(label) = &target.label {
                tally = tally.labelled(label.clone());
            }
            if let Some(capability) = capability {
                tally = tally.measures("capability", capability);
            }
            for (construction, scope, intervals, truth) in &runs {
                tally.bind_validation(construction, *scope, NO_VALIDATION);
                tally.record(intervals[index][slot], truth[index]);
            }
            tallies.push(tally);
        }
    }
    gate(&tallies, &vec![None; tallies.len()]);
}

/// The construction of a statement a route publishes in its payload only, on a supplied
/// structure and tabular or series rows.
#[allow(
    clippy::too_many_arguments,
    reason = "one value per axis of the calibration key, all of them required"
)]
fn payload_construction(
    query: &str,
    graph_class: &str,
    modality: &str,
    inference: &str,
    estimator: &str,
    interval: &str,
    se_kind: &str,
    posterior: &str,
    functional: &str,
) -> Construction {
    Construction {
        query: query.into(),
        graph_class: graph_class.into(),
        structure: "fixed".into(),
        modality: modality.into(),
        inference: inference.into(),
        estimator: estimator.into(),
        interval_method: interval.into(),
        se_kind: se_kind.into(),
        dependence: "iid".into(),
        posterior: posterior.into(),
        functional: functional.into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
        unit: None,
    }
}

/// Scope facts of a payload statement measured on `rows` rows.
fn payload_scope(rows: usize, replicates: Option<u32>, draws: Option<u32>) -> ScopeFacts {
    ScopeFacts {
        row_count: rows as u64,
        replicates_ok: replicates,
        posterior_draws: draws,
        unidentified_mass: 0.0,
    }
}

/// Hold every tally to at least nominal coverage of `target`, printing every line before
/// failing on any of them.
fn gate_at_least(tallies: &[CoverageTally], target: &str) {
    let failures: Vec<String> = tallies
        .iter()
        .filter_map(|tally| {
            // The truth is a point, so either side names it.
            std::panic::catch_unwind(|| tally.assert_one_sided(CoverageSide::Lower, target))
                .err()
                .map(|panic| {
                    panic
                        .downcast_ref::<String>()
                        .cloned()
                        .or_else(|| panic.downcast_ref::<&str>().map(|s| (*s).to_owned()))
                        .unwrap_or_else(|| "coverage failure".into())
                })
        })
        .collect();
    assert!(failures.is_empty(), "{}", failures.join("; "));
}

fn d(i: u32) -> DenseNodeId {
    DenseNodeId::from_raw(i)
}

fn v(i: u32) -> VariableId {
    VariableId::from_raw(i)
}

fn dag(nodes: u32, edges: &[(u32, u32)]) -> Dag {
    let mut graph = Dag::with_variables(nodes);
    for &(from, to) in edges {
        graph.insert_directed(d(from), d(to)).unwrap();
    }
    graph
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

/// `x_{t-1} -> y_t`.
fn lag_dag() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let parent = ensure_lagged(&mut graph, v(0), Lag::from_raw(1)).unwrap();
    let child = ensure_lagged(&mut graph, v(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(parent, child).unwrap();
    graph
}

fn bayesian(draws: usize) -> InferenceMode {
    InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(draws))
}

/// Build and run one study; a refusal is a test failure, not a skipped replicate.
fn execute(test: &str, builder: StudyBuilder, seed: u64) -> (Study, StudyResult) {
    execute_in(test, builder, &ExecutionContext::for_tests(seed))
}

/// [`execute`] in a context the caller built.
fn execute_in(test: &str, builder: StudyBuilder, ctx: &ExecutionContext) -> (Study, StudyResult) {
    let study = builder.refute(RefuteSuite::None).build().unwrap_or_else(|e| panic!("{test}: {e}"));
    let result = study.run(ctx).unwrap_or_else(|e| panic!("{test}: {e}"));
    (study, result)
}

/// Reported and gate-level interval of each posterior column in `columns`.
fn posterior_columns(result: &StudyResult, width: usize, columns: &[usize]) -> Vec<IntervalPair> {
    let posterior = result.posterior.as_ref().expect("a posterior over the mechanisms");
    assert_eq!(posterior.draws.schema.n_quantities(), width, "posterior columns");
    columns.iter().map(|&column| posterior_pair(result, column)).collect()
}

// ====================================================================== data laws

/// Residual scale of every law.
const NOISE: f64 = 0.7;
/// Slope of `x -> y`.
const SLOPE: f64 = 2.0;
/// Intercepts of the baseline and comparison populations of a change attribution.
const BASELINE: f64 = 1.0;
const COMPARISON: f64 = 3.0;
/// Intercept of a single population.
const CENTER: f64 = 1.0;

/// The columns of [`pair_rows`].
fn pair_columns(rows: usize, seed: u64, intercept: impl Fn(usize) -> f64) -> [Vec<f64>; 2] {
    let mut g = gaussian(seed);
    let (mut x, mut y) = (Vec::with_capacity(rows), Vec::with_capacity(rows));
    for i in 0..rows {
        let xi = g();
        x.push(xi);
        y.push(intercept(i) + SLOPE * xi + NOISE * g());
    }
    [x, y]
}

/// `x ~ N(0, 1)`, `y = c + 2x + 0.7e` with the intercept `c` of each row.
fn pair_rows(rows: usize, seed: u64, intercept: impl Fn(usize) -> f64) -> TabularData {
    table(&["x", "y"], &pair_columns(rows, seed, intercept))
}

/// `z ~ N(0, 1)`, `t = 0.8z + e`, `y = 1.5t + 0.6z + e'`.
fn chain_rows(rows: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut z, mut t, mut y) =
        (Vec::with_capacity(rows), Vec::with_capacity(rows), Vec::with_capacity(rows));
    for _ in 0..rows {
        let zi = g();
        let ti = 0.8 * zi + g();
        let yi = 1.5 * ti + 0.6 * zi + g();
        z.push(zi);
        t.push(ti);
        y.push(yi);
    }
    table(&["z", "t", "y"], &[z, t, y])
}

/// `z -> t -> y`, `z -> y`.
fn chain_dag() -> Dag {
    dag(3, &[(0, 1), (1, 2), (0, 2)])
}

/// `x_t ~ N(0, 1)`, `y_t = c + 2 x_{t-1} + 0.7e`, started from a draw of the stationary law.
fn lag_chain_rows(
    rows: usize,
    seed: u64,
    intercept: impl Fn(usize) -> f64,
    noise: f64,
) -> TimeSeriesData {
    let mut g = gaussian(seed);
    let (mut x, mut y) = (Vec::with_capacity(rows), Vec::with_capacity(rows));
    let mut previous = g();
    for i in 0..rows {
        let xi = g();
        x.push(xi);
        y.push(intercept(i) + SLOPE * previous + noise * g());
        previous = xi;
    }
    series(&["x", "y"], &[x, y])
}

// ================================================================ change attribution

/// Rows of each population of a change design.
const POPULATION: usize = 600;
/// Posterior draws of a Bayesian attribution or simulation: the count a study gets when it sets
/// none. The published limits are the interpolated order statistics of the draws; at `B` draws
/// the lower one sits at expected rank `(0.025 (B - 1) + 1) / (B + 1)`, so an exact posterior
/// covers 0.9405 at 200 draws and 0.9481 at 1000.
const DRAWS: usize = 1_000;
/// Row (or block) bootstrap replicates of a frequentist attribution.
const REPLICATES: u32 = 200;

/// A change attribution of the outcome `y` between the first and second half of `2 * pop` rows.
fn change_query(rooted: bool, pop: usize) -> CausalQuery {
    let change = ChangeAttributionQuery::new(
        v(1),
        PopulationSelector::TimeRange { start: 0, end: pop },
        PopulationSelector::TimeRange { start: pop, end: 2 * pop },
    );
    let estimator = ChangeEstimator::Sampled {
        measure: ChangeMeasure::MeanDifference,
        n_samples: 2_000,
        seed: 3,
    };
    CausalQuery::Attribute(AttributeQuery::new(if rooted {
        AttributeTarget::RootCauses { change, estimator }
    } else {
        AttributeTarget::DistributionChange { change, estimator }
    }))
}

fn change_rows(seed: u64) -> (TabularData, usize) {
    let pop = grid_n(POPULATION);
    let data = pair_rows(2 * pop, seed, |i| if i < pop { BASELINE } else { COMPARISON });
    (data, pop)
}

/// The mechanism-refit row bootstrap standard error of the total change.
fn attribute_change_frequentist(test: &'static str, family: u64, rooted: bool) {
    let cell = Cell {
        test,
        dgp: "pair_rows",
        interval: "bootstrap_se",
        estimator: "gcm.fit",
        variants: 1,
    };
    measure(
        cell,
        None,
        &Target::scalar(COMPARISON - BASELINE),
        |rep, _| {
            let seed = stream_seed(family, rep);
            let (data, pop) = change_rows(seed);
            let builder = Study::tabular(data)
                .graph(dag(2, &[(0, 1)]))
                .query(change_query(rooted, pop))
                .inference(InferenceMode::Frequentist)
                .bootstrap_replicates(REPLICATES);
            execute(test, builder, seed)
        },
        |result| vec![scalar_normal_pair(result)],
    );
}

/// The Dirichlet row-weight posterior over the mechanisms: the total change, the mechanism of
/// `x` (no change) and the mechanism of `y` (the whole change), in the order the posterior lists
/// its components.
fn attribute_change_bayesian(test: &'static str, family: u64, rooted: bool) {
    let cell = Cell {
        test,
        dgp: "pair_rows",
        interval: "posterior_quantile",
        estimator: "gcm.fit.bayesian",
        variants: 1,
    };
    let total = COMPARISON - BASELINE;
    let targets = vec![
        Target::labelled("total_change", total),
        Target::labelled("mechanism_x", 0.0),
        Target::labelled("mechanism_y", total),
    ];
    measure(
        cell,
        None,
        &targets,
        |rep, _| {
            let seed = stream_seed(family, rep);
            let (data, pop) = change_rows(seed);
            let builder = Study::tabular(data)
                .graph(dag(2, &[(0, 1)]))
                .query(change_query(rooted, pop))
                .inference(bayesian(DRAWS))
                .bootstrap_replicates(0);
            execute(test, builder, seed)
        },
        |result| posterior_columns(result, 3, &[0, 1, 2]),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn attribute_distribution_change_dag_frequentist_bootstrap_coverage() {
    attribute_change_frequentist(
        "attribute_distribution_change_dag_frequentist_bootstrap_coverage",
        0xF17_0001,
        false,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn attribute_root_causes_dag_frequentist_bootstrap_coverage() {
    attribute_change_frequentist(
        "attribute_root_causes_dag_frequentist_bootstrap_coverage",
        0xF17_0002,
        true,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn attribute_distribution_change_dag_bayesian_dirichlet_coverage() {
    attribute_change_bayesian(
        "attribute_distribution_change_dag_bayesian_dirichlet_coverage",
        0xF17_0003,
        false,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn attribute_root_causes_dag_bayesian_dirichlet_coverage() {
    attribute_change_bayesian(
        "attribute_root_causes_dag_bayesian_dirichlet_coverage",
        0xF17_0004,
        true,
    );
}

// ============================================ whole-table attribution targets (posterior)

/// Rows of a single-population design.
const ROWS: usize = 600;

/// One whole-table attribution target on `x -> y`, measured as a posterior over the mechanisms.
/// `truths` gives the truth of each labelled quantity from the replicate's observed `x` column.
fn attribute_whole_table_bayesian(
    test: &'static str,
    family: u64,
    target: fn() -> AttributeTarget,
    labels: &[&'static str],
    truths: impl Fn(&[f64]) -> Vec<f64> + Sync,
) {
    let cell = Cell {
        test,
        dgp: "pair_rows",
        interval: "posterior_quantile",
        estimator: "gcm.fit.bayesian",
        variants: 1,
    };
    // The truth of each target is read per replicate, not from the target.
    let targets: Vec<Target> =
        labels.iter().map(|&label| Target::labelled(label, f64::NAN)).collect();
    let width = targets.len();
    let columns: Vec<usize> = (0..width).collect();
    measure_sample_truths(
        cell,
        None,
        &targets,
        |rep| {
            let [x, _] = pair_columns(grid_n(ROWS), stream_seed(family, rep), |_| CENTER);
            truths(&x)
        },
        |rep, _| {
            let seed = stream_seed(family, rep);
            let data = pair_rows(grid_n(ROWS), seed, |_| CENTER);
            let builder = Study::tabular(data)
                .graph(dag(2, &[(0, 1)]))
                .query(CausalQuery::Attribute(AttributeQuery::new(target())))
                .inference(bayesian(DRAWS))
                .bootstrap_replicates(0);
            execute(test, builder, seed)
        },
        |result| posterior_columns(result, width, &columns),
    );
}

fn paths_target() -> AttributeTarget {
    AttributeTarget::Paths {
        sources: Arc::from([v(0)]),
        outcome: v(1),
        max_paths: 8,
        max_len: 4,
        through: Arc::from([]),
    }
}

fn feature_relevance_target() -> AttributeTarget {
    AttributeTarget::FeatureRelevance {
        outcome: v(1),
        features: Arc::from([v(0)]),
        delta: 1.0,
        // Large enough that the common-random-number Monte Carlo offset of the two coalition
        // values stays small against the refit spread of the score.
        n_samples: 50_000,
        max_features: 4,
    }
}

fn anomaly_target() -> AttributeTarget {
    let scale = (SLOPE * SLOPE + NOISE * NOISE).sqrt();
    AttributeTarget::Anomaly(
        AnomalyAttributionQuery::new([v(1)], 1_000_000)
            .with_reference(AnomalyReference::fixed(CENTER, scale)),
    )
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn attribute_paths_dag_bayesian_dirichlet_coverage() {
    // Total, source component and the one path are each the slope.
    attribute_whole_table_bayesian(
        "attribute_paths_dag_bayesian_dirichlet_coverage",
        0xF17_0005,
        paths_target,
        &["total", "source_x", "path_x_y"],
        |_| vec![SLOPE; 3],
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn attribute_feature_relevance_dag_bayesian_dirichlet_coverage() {
    // The score is |E[y | do(x = m + 1)] - E[y]| with `m` the observed mean of `x`, held at
    // that value in every refit: slope * (m + 1 - E[x]) with E[x] = 0.
    attribute_whole_table_bayesian(
        "attribute_feature_relevance_dag_bayesian_dirichlet_coverage",
        0xF17_0006,
        feature_relevance_target,
        &["feature_x"],
        |x| vec![SLOPE * (1.0 + x.iter().sum::<f64>() / x.len() as f64)],
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn attribute_anomaly_dag_bayesian_dirichlet_coverage() {
    // Mean of -log U for a uniform two-sided tail probability U.
    attribute_whole_table_bayesian(
        "attribute_anomaly_dag_bayesian_dirichlet_coverage",
        0xF17_0007,
        anomaly_target,
        &["mean_anomaly_score"],
        |_| vec![1.0],
    );
}

// ============================================================== intrinsic influence

/// Rows of the chain designs.
const CHAIN_ROWS: usize = 300;
/// Squared total effects of the noise of `z`, `t` and `y` on `y`, with unit noise variances.
const INFLUENCE_TRUTH: [f64; 3] = [3.24, 2.25, 1.0];

fn intrinsic_target(fit_bootstrap_replicates: u32) -> CausalQuery {
    CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::IntrinsicInfluence(
        IntrinsicInfluenceTarget {
            target: v(2),
            allocation: InfluenceAllocation::Exact,
            outer_draws: 100,
            inner_draws: 20,
            replicates: 3,
            fit_bootstrap_replicates,
            level: REPORTED_LEVEL,
            max_sources: 8,
            seed: 11,
            noise: NoiseDeclaration::Independent,
        },
    )))
}

/// The posterior over the mechanisms of the variance contribution of each noise source, held to
/// at least nominal.
///
/// Every refit draw is the refit's contribution plus the Monte Carlo error of its own nested
/// simulation, drawn afresh and independently of the refit. The draws are therefore the
/// posterior over the mechanisms convolved with independent zero-mean noise, and an equal-tailed
/// interval of such a convolution is at least as wide as the posterior's own (Anderson's
/// inequality for symmetric unimodal laws): the interval is conservative, by an amount set by the
/// simulation budget.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn attribute_intrinsic_influence_dag_bayesian_dirichlet_coverage() {
    let test = "attribute_intrinsic_influence_dag_bayesian_dirichlet_coverage";
    let key = RecordKey { test, dgp: "chain_rows", interval: "posterior_quantile" };
    let names = ["noise_z", "noise_t", "noise_y"];
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0xF17_0008, rep);
        let builder = Study::tabular(chain_rows(grid_n(CHAIN_ROWS), seed))
            .graph(chain_dag())
            .query(intrinsic_target(0))
            .inference(bayesian(DRAWS))
            .bootstrap_replicates(0);
        let (study, result) = execute(test, builder, seed);
        assert_eq!(
            result.logical_plan.estimator.as_deref(),
            Some("gcm.fit.bayesian"),
            "{test}: estimator target"
        );
        let bound = constructions(&study.inspect().expect("inspect"), &result)
            .into_iter()
            .find(|(construction, _, _)| construction.interval_method == key.interval)
            .unwrap_or_else(|| panic!("{test}: no {} interval was reported", key.interval));
        (bound, posterior_columns(&result, 3, &[0, 1, 2]))
    });
    let mut tallies: Vec<CoverageTally> = Vec::new();
    for (source, name) in names.into_iter().enumerate() {
        for (slot, level) in [REPORTED_LEVEL, GATE_LEVEL].into_iter().enumerate() {
            let mut tally = CoverageTally::for_record(key, level).labelled(name);
            for ((construction, scope, _), intervals) in &runs {
                tally.bind_validation(construction, *scope, NO_VALIDATION);
                tally.record(intervals[source][slot], INFLUENCE_TRUTH[source]);
            }
            tallies.push(tally);
        }
    }
    gate_at_least(
        &tallies,
        "variance contribution of the noise source; refit draws carry independent \
         nested-simulation noise, so the interval is conservative",
    );
}

/// Full row-bootstrap refits behind a whole-procedure interval.
const WHOLE_PROCEDURE_REFITS: u32 = 40;

/// One replicate's whole-procedure interval of one source.
#[derive(Clone, Copy)]
struct Whole {
    estimate: f64,
    low: f64,
    high: f64,
    stderr: f64,
    df: f64,
}

/// The whole-procedure interval of each source's variance contribution, from the spread of full
/// row-bootstrap refits with mechanism re-selection and fresh simulation. The interval is a
/// Student-t interval at the query's level; the gate-level interval is re-formed from the
/// published estimate, standard error and degrees of freedom.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn intrinsic_influence_whole_procedure_interval_coverage() {
    let names = ["noise_z", "noise_t", "noise_y"];
    let key = RecordKey {
        test: "intrinsic_influence_whole_procedure_interval_coverage",
        dgp: "chain_rows",
        interval: "whole_procedure_student_t",
    };
    let construction = payload_construction(
        "Attribute",
        "Dag",
        "tabular",
        "Frequentist",
        "gcm.fit",
        key.interval,
        "row_bootstrap_refit",
        "",
        "attribute.intrinsic_influence",
    );
    let scope = payload_scope(grid_n(CHAIN_ROWS), Some(WHOLE_PROCEDURE_REFITS), None);
    let mut tallies: Vec<CoverageTally> = Vec::new();
    for name in names {
        for level in [REPORTED_LEVEL, GATE_LEVEL] {
            tallies.push(CoverageTally::for_record(key, level).labelled(name));
        }
    }
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0xF17_0009, rep);
        let study = Study::tabular(chain_rows(grid_n(CHAIN_ROWS), seed))
            .graph(chain_dag())
            .query(intrinsic_target(WHOLE_PROCEDURE_REFITS))
            .inference(InferenceMode::Frequentist)
            .refute(RefuteSuite::None)
            .build()
            .ok()?;
        let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
        let Some(AttributionOutcome::IntrinsicInfluence(influence)) =
            result.attribution().map(|payload| payload.outcome.clone())
        else {
            return None;
        };
        assert_eq!(influence.sources.len(), names.len(), "one source per noise term");
        let wholes: Vec<Option<Whole>> = influence
            .sources
            .iter()
            .enumerate()
            .map(|(index, source)| {
                assert_eq!(source.source.variable().raw() as usize, index, "source order");
                match &source.whole_procedure {
                    ComponentStatus::Quantified { stderr, ci_low, ci_high, df, .. } => {
                        Some(Whole {
                            estimate: source.contribution,
                            low: *ci_low,
                            high: *ci_high,
                            stderr: *stderr,
                            df: *df,
                        })
                    }
                    _ => None,
                }
            })
            .collect();
        Some(wholes)
    });
    for scored in &runs {
        for (source, truth) in INFLUENCE_TRUTH.iter().enumerate() {
            let whole = scored.as_ref().and_then(|wholes| wholes[source]);
            let [reported, gated] = &mut tallies[2 * source..2 * source + 2] else {
                unreachable!("two tallies per source")
            };
            let Some(whole) = whole else {
                reported.skip();
                gated.skip();
                continue;
            };
            reported.bind_validation(&construction, scope, NO_VALIDATION);
            gated.bind_validation(&construction, scope, NO_VALIDATION);
            reported.record(Some((whole.low, whole.high)), *truth);
            let t = antecedent_stats::student_t_ppf(0.5 + 0.5 * GATE_LEVEL, whole.df);
            gated.record(
                Some((whole.estimate - t * whole.stderr, whole.estimate + t * whole.stderr)),
                *truth,
            );
        }
    }
    gate(&tallies, &vec![None; tallies.len()]);
}

// ====================================================================== simulation

/// Rows of the simulation designs.
const SIM_ROWS: usize = 500;
/// Draws of one policy-value batch. Every refit simulates on the same noise, so the Monte Carlo
/// error of a simulated mean is an offset the refit spread does not see: its variance is the
/// sampling variance of that mean times `rows / draws`, 1% here at the base sample size.
const POLICY_DRAWS: usize = 50_000;

fn do_t() -> Arc<[Intervention]> {
    Arc::from([Intervention::set(v(1), Value::f64(1.0))])
}

fn shift_t() -> Arc<[Intervention]> {
    Arc::from([Intervention::shift(v(1), Value::f64(0.5))])
}

fn policy_query(program: fn() -> Arc<[Intervention]>) -> CausalQuery {
    CausalQuery::Simulate(SimulateQuery::new(SimulationTarget::PolicyValue {
        interventions: program(),
        outcome: v(2),
        draws: POLICY_DRAWS,
    }))
}

/// The posterior over the mechanisms of the policy mean, the baseline mean and their paired
/// difference. `effect` is the policy's effect on the mean of `y`; the baseline mean is 0.
fn policy_value_bayesian(
    test: &'static str,
    family: u64,
    program: fn() -> Arc<[Intervention]>,
    effect: f64,
) {
    let cell = Cell {
        test,
        dgp: "chain_rows",
        interval: "posterior_quantile",
        estimator: "gcm.fit.bayesian",
        variants: 1,
    };
    let targets = vec![
        Target::labelled("policy_mean", effect),
        Target::labelled("baseline_mean", 0.0),
        Target::labelled("mean_difference", effect),
    ];
    measure(
        cell,
        None,
        &targets,
        |rep, _| {
            let seed = stream_seed(family, rep);
            let builder = Study::tabular(chain_rows(grid_n(SIM_ROWS), seed))
                .graph(chain_dag())
                .query(policy_query(program))
                .inference(bayesian(DRAWS))
                .bootstrap_replicates(0);
            execute(test, builder, seed)
        },
        |result| posterior_columns(result, 3, &[0, 1, 2]),
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn simulate_policy_value_dag_bayesian_dirichlet_coverage() {
    policy_value_bayesian(
        "simulate_policy_value_dag_bayesian_dirichlet_coverage",
        0xF17_000A,
        do_t,
        1.5,
    );
}

/// A shift program runs through the same simulation route as a hard set.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn intervention_program_shift_dag_bayesian_dirichlet_coverage() {
    policy_value_bayesian(
        "intervention_program_shift_dag_bayesian_dirichlet_coverage",
        0xF17_000B,
        shift_t,
        0.75,
    );
}

/// The replicate-count interval of a policy value: the point is the paired mean difference and
/// the standard error the mechanism-refit bootstrap spread the route publishes beside it.
fn policy_value_frequentist(
    test: &'static str,
    family: u64,
    program: fn() -> Arc<[Intervention]>,
    effect: f64,
) {
    let cell = Cell {
        test,
        dgp: "chain_rows",
        interval: "bootstrap_se",
        estimator: "gcm.fit",
        variants: 1,
    };
    measure(
        cell,
        None,
        &Target::scalar(effect),
        |rep, _| {
            let seed = stream_seed(family, rep);
            let builder = Study::tabular(chain_rows(grid_n(SIM_ROWS), seed))
                .graph(chain_dag())
                .query(policy_query(program))
                .inference(InferenceMode::Frequentist)
                .bootstrap_replicates(REPLICATES);
            execute(test, builder, seed)
        },
        |result| vec![scalar_normal_pair(result)],
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn simulate_policy_value_dag_frequentist_bootstrap_coverage() {
    policy_value_frequentist(
        "simulate_policy_value_dag_frequentist_bootstrap_coverage",
        0xF17_000C,
        do_t,
        1.5,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn intervention_program_shift_dag_frequentist_bootstrap_coverage() {
    policy_value_frequentist(
        "intervention_program_shift_dag_frequentist_bootstrap_coverage",
        0xF17_000D,
        shift_t,
        0.75,
    );
}

// ======================================================================== temporal

/// Rows of each population of a temporal change design.
const SERIES_POPULATION: usize = 600;
/// Steps and rows of a temporal rollout. The rollout's seed is the query's, so the simulated
/// innovations are the same in every refit and replicate and the mean of a step sits a fixed
/// `sigma / sqrt(rows)` offset from its model value: 0.0022 at this row count, against a refit
/// spread of 0.04 to 0.08.
const ROLLOUT_STEPS: usize = 4;
const ROLLOUT_ROWS: usize = 100_000;
/// Temporal work the rollout test authorizes: every refit reads one history row per simulated
/// row and node.
const ROLLOUT_WORK: u64 = 4_000_000_000;
/// Length of the series a rollout is fitted on.
const ROLLOUT_SERIES: usize = 300;

fn temporal_change_rows(seed: u64) -> (TimeSeriesData, usize) {
    let pop = grid_n(SERIES_POPULATION);
    let data =
        lag_chain_rows(2 * pop, seed, |i| if i < pop { BASELINE } else { COMPARISON }, NOISE);
    (data, pop)
}

/// The circular-block mechanism-refit bootstrap standard error of the total change of a lag-one
/// chain.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn attribute_temporal_dag_frequentist_block_bootstrap_coverage() {
    let test = "attribute_temporal_dag_frequentist_block_bootstrap_coverage";
    let cell = Cell {
        test,
        dgp: "lag_chain_rows",
        interval: "circular_block_se",
        estimator: "gcm.fit",
        variants: 1,
    };
    measure(
        cell,
        Some("gcm.attribute.mechanism_refit"),
        &Target::scalar(COMPARISON - BASELINE),
        |rep, _| {
            let seed = stream_seed(0xF17_000E, rep);
            let (data, pop) = temporal_change_rows(seed);
            let builder = Study::series(data)
                .graph(lag_dag())
                .query(change_query(false, pop))
                .inference(InferenceMode::Frequentist)
                .bootstrap_replicates(REPLICATES);
            execute(test, builder, seed)
        },
        |result| vec![scalar_normal_pair(result)],
    );
}

/// The block-weight posterior over the mechanisms of the total change of a lag-one chain.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn attribute_temporal_dag_bayesian_block_weights_coverage() {
    let test = "attribute_temporal_dag_bayesian_block_weights_coverage";
    let cell = Cell {
        test,
        dgp: "lag_chain_rows",
        interval: "posterior_quantile",
        estimator: "gcm.fit.bayesian",
        variants: 1,
    };
    let targets = vec![Target::labelled("total_change", COMPARISON - BASELINE)];
    measure(
        cell,
        Some("gcm.attribute.mechanism_refit"),
        &targets,
        |rep, _| {
            let seed = stream_seed(0xF17_000F, rep);
            let (data, pop) = temporal_change_rows(seed);
            let builder = Study::series(data)
                .graph(lag_dag())
                .query(change_query(false, pop))
                .inference(bayesian(DRAWS))
                .bootstrap_replicates(0);
            execute(test, builder, seed)
        },
        |result| {
            let posterior = result.posterior.as_ref().expect("a block-weight posterior");
            assert!(posterior.draws.schema.n_quantities() >= 1, "posterior columns");
            vec![posterior_pair(result, 0)]
        },
    );
}

/// The block-weight posterior of the per-step means of a rollout that holds `x` at 1: `y` is
/// `intercept + slope = 3` from the second step on. The rollout lists the mean of each variable
/// at each step (step-major), then one simulated path per variable and step.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn simulate_temporal_rollout_bayesian_block_weights_coverage() {
    let test = "simulate_temporal_rollout_bayesian_block_weights_coverage";
    let cell = Cell {
        test,
        dgp: "lag_chain_rows",
        interval: "posterior_quantile",
        estimator: "gcm.fit.bayesian",
        variants: 1,
    };
    let held = CENTER + SLOPE;
    let targets = vec![
        Target::labelled("step1_y", held),
        Target::labelled("step2_y", held),
        Target::labelled("step3_y", held),
    ];
    measure(
        cell,
        Some("gcm.simulate.mechanism_refit"),
        &targets,
        |rep, _| {
            let seed = stream_seed(0xF17_0010, rep);
            let data = lag_chain_rows(grid_n(ROLLOUT_SERIES), seed, |_| CENTER, NOISE);
            let schedule = InterventionSequence::new(vec![SequencedIntervention::new(
                Intervention::set(v(0), Value::f64(1.0)),
                TemporalPolicy::Sustained {
                    from: 0,
                    until: i32::try_from(ROLLOUT_STEPS).unwrap() - 1,
                },
            )]);
            let target = SimulationTarget::TemporalRollout {
                schedule,
                n_rows: ROLLOUT_ROWS,
                n_steps: ROLLOUT_STEPS,
                window: 0,
                history: Arc::from([]),
                seed: 9,
            };
            let builder = Study::series(data)
                .graph(lag_dag())
                .query(CausalQuery::Simulate(SimulateQuery::new(target)))
                .inference(bayesian(DRAWS))
                .bootstrap_replicates(0);
            let limits =
                TemporalMaterializationLimits { max_work: ROLLOUT_WORK, ..Default::default() };
            let ctx =
                ExecutionContext::for_tests(seed).with_temporal_materialization_limits(limits);
            execute_in(test, builder, &ctx)
        },
        // Two variables per step: the mean of `y` at step `s` is column `2s + 1`; the means and
        // the simulated paths together are sixteen columns.
        |result| posterior_columns(result, 16, &[3, 5, 7]),
    );
}

// ============================================================ mechanism-change posterior

/// Prior probability of a change the query declares.
const CHANGE_PRIOR: f64 = 0.5;
/// Intercept shift of the target's mechanism when it changes.
const CHANGE_SHIFT: f64 = 1.0;
/// Rows of each population of a mechanism-change design.
const CHANGE_ROWS: usize = 150;

/// Whether the smallest credible set of mass `level` over `{no change, change}` misses the truth:
/// the set is the more probable value alone once its mass reaches `level`, else both values.
fn credible_set_misses(probability: f64, change: bool, level: f64) -> bool {
    let truth = if change { probability } else { 1.0 - probability };
    1.0 - truth >= level
}

/// The posterior probability of a change under the declared prior, scored as a credible set for
/// the change indicator: the indicator is drawn from the declared prior, so a posterior that is
/// right about the law of the data holds the truth in its credible set at least as often as the
/// level. The set is encoded as the interval `[0, 1]` when it holds the truth and `[2, 3]` when
/// it does not, so the tally counts set coverage.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn mechanism_change_posterior_probability_credible_set_coverage() {
    let key = RecordKey {
        test: "mechanism_change_posterior_probability_credible_set_coverage",
        dgp: "pair_rows",
        interval: "change_indicator_credible_set",
    };
    let construction = payload_construction(
        "Attribute",
        "Dag",
        "tabular",
        "Bayesian",
        "gcm.fit",
        key.interval,
        "",
        "declared_change_prior.normal_inverse_gamma",
        "attribute.mechanism_change",
    );
    let scope = payload_scope(2 * grid_n(CHANGE_ROWS), None, None);
    let mut tallies = [
        CoverageTally::for_record(key, REPORTED_LEVEL),
        CoverageTally::for_record(key, GATE_LEVEL),
    ];
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0xF17_0011, rep);
        let change = unit_uniform(seed) < CHANGE_PRIOR;
        let n = grid_n(CHANGE_ROWS);
        let shift = if change { CHANGE_SHIFT } else { 0.0 };
        let data = pair_rows(2 * n, seed, |i| if i < n { CENTER } else { CENTER + shift });
        let query = MechanismChangeQuery::new(
            [v(1)],
            PopulationSelector::TimeRange { start: 0, end: n },
            PopulationSelector::TimeRange { start: n, end: 2 * n },
            0.05,
            4,
        )
        .with_change_prior(CHANGE_PRIOR);
        let study = Study::tabular(data)
            .graph(dag(2, &[(0, 1)]))
            .query(CausalQuery::Attribute(AttributeQuery::new(AttributeTarget::MechanismChange {
                query,
                test: MechanismChangeTest::LikelihoodRatio,
            })))
            .inference(bayesian(8))
            .refute(RefuteSuite::None)
            .bootstrap_replicates(0)
            .build()
            .ok()?;
        let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
        let probability = match result.attribution().map(|payload| payload.outcome.clone()) {
            Some(AttributionOutcome::MechanismChange(found)) => {
                found.first()?.posterior_probability?
            }
            _ => return None,
        };
        Some((change, probability))
    });
    for scored in &runs {
        for (tally, level) in tallies.iter_mut().zip([REPORTED_LEVEL, GATE_LEVEL]) {
            match scored {
                None => tally.skip(),
                Some((change, probability)) => {
                    let miss = credible_set_misses(*probability, *change, level);
                    tally.bind_validation(&construction, scope, NO_VALIDATION);
                    tally.record(Some(if miss { (2.0, 3.0) } else { (0.0, 1.0) }), 1.0);
                }
            }
        }
    }
    gate_at_least(
        &tallies,
        "change indicator drawn from the declared prior; the credible set is the more probable \
         value alone or both values, so it holds at least its stated mass",
    );
}

// ============================================================ structure posteriors

/// Rows of a structure-search design.
const SEARCH_ROWS: usize = 150;
/// Length of the series a temporal structure search reads.
const SEARCH_SERIES: usize = 200;

/// Whether the smallest set of atoms holding at least `level` of the posterior mass contains an
/// atom `hit` accepts: atoms are taken in decreasing weight until the mass reaches `level`.
fn credible_set_holds(weights: &[f64], hit: impl Fn(usize) -> bool, level: f64) -> bool {
    let mut order: Vec<usize> = (0..weights.len()).collect();
    order.sort_by(|&a, &b| weights[b].total_cmp(&weights[a]));
    let mut mass = 0.0;
    for index in order {
        if hit(index) {
            return true;
        }
        mass += weights[index];
        if mass >= level {
            return false;
        }
    }
    false
}

/// Interval method of a structure posterior scored as a credible set.
const STRUCTURE_SET: &str = "structure_credible_set";
/// Why a structure credible set is held to at least nominal.
const STRUCTURE_SET_TARGET: &str = "true structure; a credible set of whole atoms holds at least \
                                    its stated mass";

/// Scored tallies of a structure-posterior test at the reported and the gate level.
fn set_tallies(test: &'static str, dgp: &'static str) -> [CoverageTally; 2] {
    let key = RecordKey { test, dgp, interval: STRUCTURE_SET };
    [CoverageTally::for_record(key, REPORTED_LEVEL), CoverageTally::for_record(key, GATE_LEVEL)]
}

fn score_sets(
    tallies: &mut [CoverageTally; 2],
    scored: Option<[bool; 2]>,
    construction: &Construction,
    scope: ScopeFacts,
) {
    for (index, tally) in tallies.iter_mut().enumerate() {
        match scored {
            None => tally.skip(),
            Some(holds) => {
                tally.bind_validation(construction, scope, NO_VALIDATION);
                tally.record(Some(if holds[index] { (0.0, 1.0) } else { (2.0, 3.0) }), 1.0);
            }
        }
    }
}

/// `z -> t -> y` with Gaussian noise and no direct edge.
fn pure_chain_rows(rows: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut z, mut t, mut y) =
        (Vec::with_capacity(rows), Vec::with_capacity(rows), Vec::with_capacity(rows));
    for _ in 0..rows {
        let zi = g();
        let ti = 0.8 * zi + g();
        let yi = 1.5 * ti + g();
        z.push(zi);
        t.push(ti);
        y.push(yi);
    }
    table(&["z", "t", "y"], &[z, t, y])
}

/// The class posterior of the exact DAG posterior: the posterior over Markov equivalence classes
/// of three variables, scored by whether its smallest set of at least the stated mass holds the
/// class of the true chain. A chain has no collider, so its class leaves both edges undirected.
/// The exact posterior lists every class, so the true class is always an atom.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn discovery_static_exact_class_posterior_credible_set_coverage() {
    let mut chain_class = Cpdag::with_variables(3);
    chain_class.insert_undirected(d(0), d(1)).unwrap();
    chain_class.insert_undirected(d(1), d(2)).unwrap();
    let truth = adjacency_mask_from_cpdag(&chain_class).unwrap();
    let mut tallies = set_tallies(
        "discovery_static_exact_class_posterior_credible_set_coverage",
        "pure_chain_rows",
    );
    let construction = payload_construction(
        "Discover",
        "Cpdag",
        "tabular",
        "Bayesian",
        "discovery.exact_dag_posterior",
        STRUCTURE_SET,
        "",
        "exact_dag_posterior.equivalence_classes.gaussian_bic",
        "discover.class_posterior",
    );
    let scope = payload_scope(grid_n(SEARCH_ROWS), None, None);
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0xF17_0012, rep);
        let spec = DiscoverySpec::exact_dag_posterior(BayesianDiscoverParams::default())
            .equivalence_classes();
        let study = Study::tabular(pure_chain_rows(grid_n(SEARCH_ROWS), seed))
            .discovery(spec, vec![v(0), v(1), v(2)])
            .build()
            .ok()?;
        let execution = CompiledExecution::run(&study, &ExecutionContext::for_tests(seed)).ok()?;
        let Some(DiscoveredStructure::Posterior(classes)) =
            execution.result().discovery().map(|payload| payload.structure.clone())
        else {
            return None;
        };
        let weights = classes.weights();
        let atoms = classes.atoms();
        assert!(
            atoms.iter().any(|atom| atom.to_word() == Some(truth)),
            "the exact class posterior lists the true class"
        );
        let hit = |index: usize| atoms[index].to_word() == Some(truth);
        Some([
            credible_set_holds(weights, hit, REPORTED_LEVEL),
            credible_set_holds(weights, hit, GATE_LEVEL),
        ])
    });
    for scored in &runs {
        score_sets(&mut tallies, *scored, &construction, scope);
    }
    gate_at_least(&tallies, STRUCTURE_SET_TARGET);
}

/// The posterior over dynamic-Bayesian-network templates of two variables with one lag, scored
/// by whether its smallest set of at least the stated mass holds the true template: no
/// contemporaneous edge and the lagged edge `x_{t-1} -> y_t`.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn discovery_temporal_dbn_posterior_credible_set_coverage() {
    // Bit `(lag - 1) * n^2 + from * n + to` of the lag mask with two variables.
    let truth_lag: u64 = 1 << 1;
    let mut tallies =
        set_tallies("discovery_temporal_dbn_posterior_credible_set_coverage", "lag_chain_rows");
    let construction = payload_construction(
        "Discover",
        "TemporalDag",
        "series",
        "Bayesian",
        "discovery.dbn_posterior",
        STRUCTURE_SET,
        "",
        "dbn_posterior.gaussian_bic.mcmc_4x400",
        "discover.dbn_template_posterior",
    );
    let scope = payload_scope(grid_n(SEARCH_SERIES), None, Some(1_600));
    let runs = map_replicates(n_sim(), |rep| {
        let seed = stream_seed(0xF17_0013, rep);
        let schedule = GraphMcmcSchedule { n_chains: 4, n_warmup: 200, n_draws: 400, thin: 1 };
        let spec =
            DiscoverySpec::dbn_posterior(BayesianDiscoverParams::default(), 1, false, schedule);
        let study = Study::series(lag_chain_rows(grid_n(SEARCH_SERIES), seed, |_| 0.0, 0.5))
            .discovery(spec, vec![v(0), v(1)])
            .build()
            .ok()?;
        let execution = CompiledExecution::run(&study, &ExecutionContext::for_tests(seed)).ok()?;
        let Some(DiscoveredStructure::Posterior(templates)) =
            execution.result().discovery().map(|payload| payload.structure.clone())
        else {
            return None;
        };
        let weights = templates.weights();
        let atoms = templates.atoms();
        let lags = templates.lag_masks().expect("a template posterior carries its lag masks");
        let hit = |index: usize| {
            atoms[index].to_word() == Some(0) && lags[index].to_word() == Some(truth_lag)
        };
        Some([
            credible_set_holds(weights, hit, REPORTED_LEVEL),
            credible_set_holds(weights, hit, GATE_LEVEL),
        ])
    });
    for scored in &runs {
        score_sets(&mut tallies, *scored, &construction, scope);
    }
    gate_at_least(&tallies, STRUCTURE_SET_TARGET);
}
