//! Repeated-sampling coverage of the further uncertainty statements of the fitted-model
//! attribution and simulation routes: the mechanism-refit bootstrap error of the path, feature,
//! anomaly, unit-change, structure-change and robust-change targets, the posterior over the
//! mechanisms of a unit change, a structure change and a mean anomaly score against the observed
//! reference, the mechanism components and the intrinsic influence on the unfolding of a series,
//! and both resamplings of an interventional sample, unit effects, a counterfactual and a
//! temporal counterfactual.
//!
//! Every law is linear-Gaussian, so each truth is exact:
//!
//! * On `x -> y` with `y = c + 2x + 0.7e` a path decomposition is the slope. The feature
//!   relevance of `x` holds the observed mean `m` of `x` fixed across refits, so its target is
//!   `2 (1 + m)`. The mean anomaly score against the true location and scale of `y` is the mean
//!   of an `Exp(1)` variable, 1; against the observed median and scaled median absolute deviation
//!   it is the population mean of the score at that realized reference, integrated numerically
//!   per replicate. The mean unit change of the first rows is the slope times the distance of
//!   their mean `x` from the observed mean of `x`, the reference every refit keeps.
//! * A change between two stacked populations that differ only in the intercept of `y` is the
//!   intercept shift under the robust estimator. A structure change removes the edge `x -> y`
//!   in the second population (`y = 3 + 0.7e` there), so with `E[x] = 0` the one structure
//!   player carries the whole change of the mean, 2.
//! * On a lag-one chain whose intercept shifts, the mechanism of `y` carries the whole change
//!   and the mechanism of the lagged cause none; the variance of `y` splits into `4 Var(x)` from
//!   the lagged cause and `0.49` from its own noise.
//! * On `z -> t -> y`, `z -> y` (`y = 1.5t + 0.6z + e`) a sample under `do(t = 1)` has mean 0
//!   for `z` and 1.5 for `y`, and every unit effect of `t` from 0 to 1 is 1.5. A counterfactual
//!   replays the observed units, so its target is the mean of their true counterfactual
//!   outcomes, `y_i + 1.5 (1 - t_i)`, on the rows of the table or on each step of a factual
//!   panel.
//!
//! A statement whose interval the result binds (a posterior, or a bootstrap standard error beside
//! a finite point estimate) is keyed by the runtime's calibration key. A bootstrap standard
//! error the route publishes only as a field of its bootstrap diagnostic, beside a point in the
//! payload, has no runtime key: the normal interval the diagnostic describes is formed here and
//! its record names the construction, as the first harness of this family does for payload
//! statements. The routes have no support-matrix cell, so every record is bound with validation
//! `not_applicable`. A replicate the route refuses is a skip; one that runs and publishes
//! nothing to score is a miss.
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_possible_truncation,
    clippy::cast_possible_wrap,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    clippy::too_many_arguments,
    clippy::too_many_lines,
    reason = "test scaffolding compares exact constants, indexes rows with small integers and keeps one design in one body"
)]

mod common;

use std::sync::Arc;

use antecedent::{
    BayesianConfig, InferenceMode, RefuteSuite, SimulationOutcome, Study, StudyBuilder, StudyResult,
};
use antecedent_core::{
    AllocationMethod, AnomalyAttributionQuery, AnomalyReference, AttributeQuery, AttributeTarget,
    AttributionComponents, CausalQuery, ChangeAttributionQuery, ChangeEstimator, ChangeMeasure,
    ExecutionContext, InfluenceAllocation, Intervention, InterventionSequence,
    IntrinsicInfluenceTarget, Lag, NoiseDeclaration, PopulationSelector, SequencedIntervention,
    ShapleyConfig, SimulateQuery, SimulationTarget, TemporalMaterializationLimits, TemporalPolicy,
    UnitChangeQuery, Value, VariableId,
};
use antecedent_data::{TabularData, TimeSeriesData};
use antecedent_graph::{Dag, DenseNodeId, TemporalDag, ensure_lagged};
use antecedent_prob::PosteriorQuantityKind;
use common::calibration::{
    Construction, CoverageSide, CoverageTally, RecordKey, ScopeFacts, gaussian, grid_n,
    map_replicates, n_sim, stream_seed,
};
use common::calibration_bind::constructions;
use common::reported::{
    GATE_LEVEL, IntervalPair, REPORTED_LEVEL, gate, normal_at, posterior_pair, scalar_normal_pair,
};

/// Validation mode of every record of this file: the routes have no support-matrix cell.
const NO_VALIDATION: &str = "not_applicable";

// ====================================================================== measurement

/// The uncertainty statement a test scores.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Statement {
    /// Equal-tailed intervals of named columns of the posterior over the mechanisms.
    Posterior,
    /// The normal interval of the result's point estimate and bootstrap standard error.
    BootstrapEstimate,
    /// Normal intervals of named quantities: the point in the payload, the bootstrap standard
    /// error in the field of the bootstrap diagnostic that names the quantity.
    BootstrapFields,
}

impl Statement {
    const fn interval(self) -> &'static str {
        match self {
            Self::Posterior => "posterior_quantile",
            Self::BootstrapEstimate | Self::BootstrapFields => "bootstrap_se",
        }
    }

    const fn estimator(self) -> &'static str {
        match self {
            Self::Posterior => "gcm.fit.bayesian",
            Self::BootstrapEstimate | Self::BootstrapFields => "gcm.fit",
        }
    }
}

/// One replicate the route executed.
struct Scored {
    /// Construction and scope of the scored intervals; `None` when the execution published no
    /// interval of the statement, which every target then counts as a miss.
    bound: Option<(Construction, ScopeFacts)>,
    /// One `[reported, gate]` pair per target.
    intervals: Vec<IntervalPair>,
    /// One truth per target.
    truth: Vec<f64>,
}

/// How a test's tallies are gated.
#[derive(Clone, Copy)]
enum Gate {
    /// At the nominal level.
    Nominal,
    /// At no less than the nominal level, for the stated reason.
    AtLeast(&'static str),
}

/// Measure one statement over `n_sim()` replicates. `run(replicate)` is `None` for a refused
/// replicate. `labels` names the targets in the order `run` scores them; a single target may be
/// left unlabelled with an empty label.
fn measure(
    test: &'static str,
    dgp: &'static str,
    statement: Statement,
    labels: &[&'static str],
    gating: Gate,
    run: impl Fn(u64) -> Option<Scored> + Sync,
) {
    let runs = map_replicates(n_sim(), run);
    let key = RecordKey { test, dgp, interval: statement.interval() };
    let mut tallies: Vec<CoverageTally> = Vec::new();
    for (index, label) in labels.iter().enumerate() {
        for (slot, level) in [REPORTED_LEVEL, GATE_LEVEL].into_iter().enumerate() {
            let mut tally = CoverageTally::for_record(key, level);
            if !label.is_empty() {
                tally = tally.labelled(*label);
            }
            for scored in &runs {
                let Some(scored) = scored else {
                    tally.skip();
                    continue;
                };
                assert_eq!(scored.intervals.len(), labels.len(), "{test}: one interval per target");
                assert_eq!(scored.truth.len(), labels.len(), "{test}: one truth per target");
                match &scored.bound {
                    Some((construction, scope)) => {
                        tally.bind_validation(construction, *scope, NO_VALIDATION);
                        tally.record(scored.intervals[index][slot], scored.truth[index]);
                    }
                    None => tally.record(None, scored.truth[index]),
                }
            }
            tallies.push(tally);
        }
    }
    match gating {
        Gate::Nominal => gate(&tallies, &vec![None; tallies.len()]),
        Gate::AtLeast(target) => gate_at_least(&tallies, target),
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

/// Build and run one study without a validation suite; `None` when the route refuses it.
fn execute(
    test: &str,
    builder: StudyBuilder,
    ctx: &ExecutionContext,
) -> Option<(Study, StudyResult)> {
    let study = match builder.refute(RefuteSuite::None).build() {
        Ok(study) => study,
        Err(error) => {
            eprintln!("{test}: refused at build: {error}");
            return None;
        }
    };
    match study.run(ctx) {
        Ok(result) => Some((study, result)),
        Err(error) => {
            eprintln!("{test}: refused at run: {error}");
            None
        }
    }
}

/// The inference and resampling a statement runs under.
fn resampled(builder: StudyBuilder, statement: Statement) -> StudyBuilder {
    match statement {
        Statement::Posterior => builder
            .inference(InferenceMode::Bayesian(BayesianConfig::conjugate().n_draws(DRAWS)))
            .bootstrap_replicates(0),
        Statement::BootstrapEstimate | Statement::BootstrapFields => {
            builder.inference(InferenceMode::Frequentist).bootstrap_replicates(REPLICATES)
        }
    }
}

/// The runtime's construction and scope of the `statement` interval of `result`, when it bound
/// one.
fn runtime_bound(
    test: &str,
    statement: Statement,
    study: &Study,
    result: &StudyResult,
) -> Option<(Construction, ScopeFacts)> {
    constructions(&study.inspect().expect("inspect"), result)
        .into_iter()
        .find(|(construction, _, _)| construction.interval_method == statement.interval())
        .map(|(construction, scope, _)| (construction, scope))
        .or_else(|| {
            eprintln!("{test}: the execution bound no {} interval", statement.interval());
            None
        })
}

/// Names of the posterior's columns, in column order; empty without a posterior.
fn posterior_names(result: &StudyResult) -> Vec<String> {
    result.posterior.as_ref().map_or_else(Vec::new, |posterior| {
        posterior
            .draws
            .schema
            .quantities
            .iter()
            .map(|quantity| match quantity {
                PosteriorQuantityKind::Effect { name } | PosteriorQuantityKind::Scalar { name } => {
                    name.to_string()
                }
                _ => String::new(),
            })
            .collect()
    })
}

/// Reported and gate-level interval of the posterior column called `name`; no interval when the
/// execution has no posterior.
///
/// # Panics
///
/// When a posterior is published without that column: the test scored the wrong quantity.
fn posterior_named(test: &str, result: &StudyResult, name: &str) -> IntervalPair {
    if result.posterior.is_none() {
        return [None, None];
    }
    let names = posterior_names(result);
    let column = names
        .iter()
        .position(|column| column == name)
        .unwrap_or_else(|| panic!("{test}: no posterior column {name} among {names:?}"));
    posterior_pair(result, column)
}

/// The value of the field `key` of the bootstrap diagnostic of `result`.
fn bootstrap_field<'a>(result: &'a StudyResult, key: &str) -> Option<&'a str> {
    result
        .diagnostics
        .iter()
        .find(|diagnostic| diagnostic.code.as_ref() == "gcm.frequentist_bootstrap")?
        .fields
        .iter()
        .find(|(field, _)| field.as_ref() == key)
        .map(|(_, value)| &**value)
}

/// The normal intervals of the named quantities of a refit bootstrap whose standard errors the
/// route publishes as fields of its bootstrap diagnostic: `point ± z · se` with the point from
/// `points` (the payload) and the standard error from the field `bootstrap_se.<name>`. The
/// construction is named here, with the axes the runtime keys the bound bootstrap intervals of
/// the same routes under.
///
/// # Panics
///
/// When the payload has no quantity of a requested name.
fn bootstrap_fields(
    test: &str,
    query: &str,
    functional: &str,
    rows: usize,
    result: &StudyResult,
    points: &[(String, f64)],
    names: &[&str],
) -> (Option<(Construction, ScopeFacts)>, Vec<IntervalPair>) {
    let Some(replicates_ok) =
        bootstrap_field(result, "replicates_ok").and_then(|count| count.parse::<u32>().ok())
    else {
        eprintln!("{test}: the execution published no bootstrap accounting");
        return (None, vec![[None, None]; names.len()]);
    };
    let intervals = names
        .iter()
        .map(|name| {
            let point = points
                .iter()
                .find(|(quantity, _)| quantity == name)
                .unwrap_or_else(|| panic!("{test}: no payload quantity {name} among {points:?}"))
                .1;
            // An unquantified standard error does not parse: no interval, a miss.
            match bootstrap_field(result, &format!("bootstrap_se.{name}"))
                .and_then(|se| se.parse::<f64>().ok())
            {
                Some(se) => {
                    [normal_at(point, se, REPORTED_LEVEL), normal_at(point, se, GATE_LEVEL)]
                }
                None => [None, None],
            }
        })
        .collect();
    let construction = Construction {
        query: query.into(),
        graph_class: "Dag".into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: Statement::BootstrapFields.estimator().into(),
        interval_method: Statement::BootstrapFields.interval().into(),
        se_kind: String::new(),
        dependence: "iid".into(),
        posterior: String::new(),
        functional: functional.into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
    };
    let scope = ScopeFacts {
        row_count: rows as u64,
        replicates_ok: Some(replicates_ok),
        posterior_draws: None,
        unidentified_mass: 0.0,
    };
    (Some((construction, scope)), intervals)
}

/// Score `names` of `result` under `statement`. `points` reads the payload's named point values
/// and is called for [`Statement::BootstrapFields`] only; [`Statement::BootstrapEstimate`] scores
/// the single quantity the result's estimate is.
fn score(
    test: &str,
    statement: Statement,
    query: &str,
    functional: &str,
    rows: usize,
    study: &Study,
    result: &StudyResult,
    points: impl FnOnce(&StudyResult) -> Vec<(String, f64)>,
    names: &[&str],
) -> (Option<(Construction, ScopeFacts)>, Vec<IntervalPair>) {
    assert_eq!(
        result.logical_plan.estimator.as_deref(),
        Some(statement.estimator()),
        "{test}: estimator target"
    );
    match statement {
        Statement::Posterior => (
            runtime_bound(test, statement, study, result),
            names.iter().map(|name| posterior_named(test, result, name)).collect(),
        ),
        Statement::BootstrapEstimate => {
            assert_eq!(names.len(), 1, "{test}: the estimate is one quantity");
            (runtime_bound(test, statement, study, result), vec![scalar_normal_pair(result)])
        }
        Statement::BootstrapFields => {
            bootstrap_fields(test, query, functional, rows, result, &points(result), names)
        }
    }
}

/// The named point values of an attribution payload, as the route names its refit quantities.
fn attribution_points(result: &StudyResult) -> Vec<(String, f64)> {
    let payload = result.attribution().expect("an attribution payload");
    antecedent::analysis::structure_route::attribution_quantities(&payload.outcome)
        .unwrap_or_default()
}

/// The named point values of a simulation payload, as the route names its refit quantities: the
/// mean of each variable of a draw batch, of each outcome in each world of a counterfactual and
/// of each variable at each step of a panel replay.
fn simulation_points(result: &StudyResult) -> Vec<(String, f64)> {
    let payload = result.simulation().expect("a simulation payload");
    match &payload.outcome {
        SimulationOutcome::Sample(batch) => (0..batch.n_nodes)
            .map(|node| {
                (format!("draw_mean[{node}]"), mean(batch.column(node).expect("a drawn column")))
            })
            .collect(),
        SimulationOutcome::Counterfactual(found) => {
            let mut out = Vec::new();
            for world in 0..found.n_worlds {
                for &outcome in found.outcomes.iter() {
                    let column = found.outcome_column(world, outcome).expect("an outcome column");
                    out.push((
                        format!("world[{world}].outcome[{}]", outcome.as_usize()),
                        mean(column),
                    ));
                }
            }
            out
        }
        SimulationOutcome::Temporal(path) => {
            let mut out = Vec::new();
            for (step, batch) in path.steps.iter().enumerate() {
                for node in 0..path.n_nodes {
                    out.push((
                        format!("step[{step}].node[{node}]"),
                        mean(batch.column(node).expect("a replayed column")),
                    ));
                }
            }
            out
        }
        _ => Vec::new(),
    }
}

// ====================================================================== data laws

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

fn mean(values: &[f64]) -> f64 {
    values.iter().sum::<f64>() / values.len() as f64
}

/// Residual scale of `y` on the pair and lag-chain laws.
const NOISE: f64 = 0.7;
/// Slope of `x -> y`.
const SLOPE: f64 = 2.0;
/// Intercepts of the baseline and comparison populations of a change design.
const BASELINE: f64 = 1.0;
const COMPARISON: f64 = 3.0;
/// Intercept of a single population.
const CENTER: f64 = 1.0;
/// Coefficients of the chain `z -> t -> y`, `z -> y`.
const T_ON_Z: f64 = 0.8;
const Y_ON_T: f64 = 1.5;
const Y_ON_Z: f64 = 0.6;

/// Posterior draws of a Bayesian run: the count the measured posteriors of these routes ran.
const DRAWS: usize = 1_000;
/// Refit replicates of a frequentist run: the count the measured bootstraps of these routes ran.
const REPLICATES: u32 = 200;

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
fn pair_rows(columns: &[Vec<f64>; 2]) -> TabularData {
    table(&["x", "y"], columns)
}

/// `x ~ N(0, 1)` throughout; `y = 1 + 2x + 0.7e` in the first `pop` rows and `y = 3 + 0.7e`,
/// with no dependence on `x`, in the other `pop`.
fn structure_rows(pop: usize, seed: u64) -> TabularData {
    let mut g = gaussian(seed);
    let (mut x, mut y) = (Vec::with_capacity(2 * pop), Vec::with_capacity(2 * pop));
    for i in 0..2 * pop {
        let xi = g();
        x.push(xi);
        y.push(if i < pop { BASELINE + SLOPE * xi } else { COMPARISON } + NOISE * g());
    }
    table(&["x", "y"], &[x, y])
}

/// The columns of [`chain_rows`], `z`, `t`, `y`.
fn chain_columns(rows: usize, seed: u64) -> [Vec<f64>; 3] {
    let mut g = gaussian(seed);
    let (mut z, mut t, mut y) =
        (Vec::with_capacity(rows), Vec::with_capacity(rows), Vec::with_capacity(rows));
    for _ in 0..rows {
        let zi = g();
        let ti = T_ON_Z * zi + g();
        let yi = Y_ON_T * ti + Y_ON_Z * zi + g();
        z.push(zi);
        t.push(ti);
        y.push(yi);
    }
    [z, t, y]
}

/// `z ~ N(0, 1)`, `t = 0.8z + e`, `y = 1.5t + 0.6z + e'`.
fn chain_rows(columns: &[Vec<f64>; 3]) -> TabularData {
    table(&["z", "t", "y"], columns)
}

/// `z -> t -> y`, `z -> y`.
fn chain_dag() -> Dag {
    dag(3, &[(0, 1), (1, 2), (0, 2)])
}

/// `x_t ~ N(0, 1)`, `y_t = c + 2 x_{t-1} + 0.7e`, started from a draw of the stationary law.
fn lag_chain_rows(rows: usize, seed: u64, intercept: impl Fn(usize) -> f64) -> TimeSeriesData {
    let mut g = gaussian(seed);
    let (mut x, mut y) = (Vec::with_capacity(rows), Vec::with_capacity(rows));
    let mut previous = g();
    for i in 0..rows {
        let xi = g();
        x.push(xi);
        y.push(intercept(i) + SLOPE * previous + NOISE * g());
        previous = xi;
    }
    TimeSeriesData::from_f64_columns([("x", x.as_slice()), ("y", y.as_slice())], 1).unwrap()
}

/// `x_{t-1} -> y_t`.
fn lag_dag() -> TemporalDag {
    let mut graph = TemporalDag::empty();
    let parent = ensure_lagged(&mut graph, v(0), Lag::from_raw(1)).unwrap();
    let child = ensure_lagged(&mut graph, v(1), Lag::CONTEMPORANEOUS).unwrap();
    graph.insert_directed(parent, child).unwrap();
    graph
}

// ============================================================ whole-table attribution targets

/// Rows of a single-population design.
const ROWS: usize = 600;
/// Units whose change a unit-change target attributes: the first rows of the table.
const UNITS: usize = 40;

/// One whole-table attribution target on `x -> y`. `truths` gives the truth of each quantity in
/// `names` from the replicate's observed `x` and `y` columns.
fn attribute_whole_table(
    test: &'static str,
    family: u64,
    statement: Statement,
    target: fn() -> AttributeTarget,
    names: &'static [&'static str],
    truths: impl Fn(&[f64], &[f64]) -> Vec<f64> + Sync,
) {
    let functional = format!("attribute.{}", target().as_str());
    let labels = vec![""; names.len()];
    measure(test, "pair_rows", statement, &labels, Gate::Nominal, |rep| {
        let seed = stream_seed(family, rep);
        let rows = grid_n(ROWS);
        let columns = pair_columns(rows, seed, |_| CENTER);
        let truth = truths(&columns[0], &columns[1]);
        let builder = Study::tabular(pair_rows(&columns))
            .graph(dag(2, &[(0, 1)]))
            .query(CausalQuery::Attribute(AttributeQuery::new(target())));
        let (study, result) =
            execute(test, resampled(builder, statement), &ExecutionContext::for_tests(seed))?;
        let (bound, intervals) = score(
            test,
            statement,
            "Attribute",
            &functional,
            rows,
            &study,
            &result,
            attribution_points,
            names,
        );
        Some(Scored { bound, intervals, truth })
    });
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

/// Standard deviation of `y` on the pair law.
fn pair_outcome_sd() -> f64 {
    (SLOPE * SLOPE + NOISE * NOISE).sqrt()
}

/// The mean anomaly score of `y` against a fixed reference at its true location and scale.
fn anomaly_fixed_target() -> AttributeTarget {
    AttributeTarget::Anomaly(
        AnomalyAttributionQuery::new([v(1)], 1_000_000)
            .with_reference(AnomalyReference::fixed(CENTER, pair_outcome_sd())),
    )
}

/// The mean anomaly score of `y` against the reference the route reads off the rows.
fn anomaly_empirical_target() -> AttributeTarget {
    AttributeTarget::Anomaly(AnomalyAttributionQuery::new([v(1)], 1_000_000))
}

fn unit_change_target() -> AttributeTarget {
    AttributeTarget::UnitChange(
        UnitChangeQuery::new(v(1), UNITS)
            .with_unit_rows((0..UNITS).collect::<Vec<usize>>())
            // One parent, so the exact allocation is two evaluations a unit.
            .with_allocation(AllocationMethod::Shapley { approximation: ShapleyConfig::exact() }),
    )
}

/// The location and scale an empirical anomaly reference reads off `values`: the median and
/// the median absolute deviation scaled to the standard deviation of a normal law.
fn robust_reference(values: &[f64]) -> (f64, f64) {
    let mut sorted = values.to_vec();
    sorted.sort_by(f64::total_cmp);
    let center = antecedent_kernels::median_sorted(&sorted);
    let mut deviations: Vec<f64> = sorted.iter().map(|value| (value - center).abs()).collect();
    deviations.sort_by(f64::total_cmp);
    (center, 1.482_602_218_505_602 * antecedent_kernels::median_sorted(&deviations))
}

/// `E[-ln P(|Z| >= |Y - center| / scale)]` for `Y ~ N(location, sd^2)` and standard normal `Z`:
/// the population mean of the anomaly score against the reference `(center, scale)`. Simpson's
/// rule over ten standard deviations either side at a step of `5e-4`; the integrand has one kink
/// (at `Y = center`), which costs the rule an error of the order of the squared step. At
/// `center = location`, `scale = sd` the value is the mean of an `Exp(1)` variable, 1.
fn mean_anomaly_score(center: f64, scale: f64, location: f64, sd: f64) -> f64 {
    const HALF_STEPS: i64 = 20_000;
    const SPAN: f64 = 10.0;
    let step = SPAN / HALF_STEPS as f64;
    let integrand = |u: f64| {
        let z = ((location + sd * u - center) / scale).abs();
        let score = -(2.0 * antecedent_kernels::norm_sf(z)).ln();
        score * (-0.5 * u * u).exp() / std::f64::consts::TAU.sqrt()
    };
    let mut sum = 0.0;
    for i in -HALF_STEPS..=HALF_STEPS {
        let weight = if i.abs() == HALF_STEPS {
            1.0
        } else if (i + HALF_STEPS) % 2 == 1 {
            4.0
        } else {
            2.0
        };
        sum += weight * integrand(i as f64 * step);
    }
    sum * step / 3.0
}

/// The truth of a mean anomaly score against the observed reference: the population mean of the
/// score at the median and scaled median absolute deviation of the replicate's `y`. The route
/// holds that reference fixed across refits and states its interval as conditional on it.
fn anomaly_empirical_truth(y: &[f64]) -> f64 {
    let (center, scale) = robust_reference(y);
    mean_anomaly_score(center, scale, CENTER, pair_outcome_sd())
}

/// The mean true change of the first [`UNITS`] units: each unit's `y` moves by the slope times
/// the distance of its `x` from the observed mean of `x`, the reference every refit keeps.
fn unit_change_truth(x: &[f64]) -> f64 {
    SLOPE * (mean(&x[..UNITS]) - mean(x))
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn attribute_paths_dag_frequentist_bootstrap_coverage() {
    // The total of the decomposition is the one path's product of coefficients, the slope.
    attribute_whole_table(
        "attribute_paths_dag_frequentist_bootstrap_coverage",
        0xF5A_0001,
        Statement::BootstrapEstimate,
        paths_target,
        &["total_change"],
        |_, _| vec![SLOPE],
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn attribute_feature_relevance_dag_frequentist_bootstrap_coverage() {
    // The score is |E[y | do(x = m + 1)] - E[y]| with `m` the observed mean of `x`, held at
    // that value in every refit: slope * (m + 1 - E[x]) with E[x] = 0.
    attribute_whole_table(
        "attribute_feature_relevance_dag_frequentist_bootstrap_coverage",
        0xF5A_0002,
        Statement::BootstrapFields,
        feature_relevance_target,
        &["feature[0]"],
        |x, _| vec![SLOPE * (1.0 + mean(x))],
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn attribute_anomaly_dag_frequentist_bootstrap_coverage() {
    // Mean of -log U for a uniform two-sided tail probability U.
    attribute_whole_table(
        "attribute_anomaly_dag_frequentist_bootstrap_coverage",
        0xF5A_0003,
        Statement::BootstrapFields,
        anomaly_fixed_target,
        &["mean_anomaly_score[1]"],
        |_, _| vec![1.0],
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn attribute_anomaly_empirical_reference_dag_frequentist_bootstrap_coverage() {
    attribute_whole_table(
        "attribute_anomaly_empirical_reference_dag_frequentist_bootstrap_coverage",
        0xF5A_0004,
        Statement::BootstrapFields,
        anomaly_empirical_target,
        &["mean_anomaly_score[1]"],
        |_, y| vec![anomaly_empirical_truth(y)],
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn attribute_unit_change_dag_frequentist_bootstrap_coverage() {
    attribute_whole_table(
        "attribute_unit_change_dag_frequentist_bootstrap_coverage",
        0xF5A_0005,
        Statement::BootstrapFields,
        unit_change_target,
        &["unit_mean[0]"],
        |x, _| vec![unit_change_truth(x)],
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn attribute_unit_change_dag_bayesian_dirichlet_coverage() {
    attribute_whole_table(
        "attribute_unit_change_dag_bayesian_dirichlet_coverage",
        0xF5A_0006,
        Statement::Posterior,
        unit_change_target,
        &["unit_mean[0]"],
        |x, _| vec![unit_change_truth(x)],
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn attribute_anomaly_empirical_reference_dag_bayesian_dirichlet_coverage() {
    attribute_whole_table(
        "attribute_anomaly_empirical_reference_dag_bayesian_dirichlet_coverage",
        0xF5A_0007,
        Statement::Posterior,
        anomaly_empirical_target,
        &["mean_anomaly_score[1]"],
        |_, y| vec![anomaly_empirical_truth(y)],
    );
}

// ================================================================ change attribution

/// Rows of each population of the robust change design.
const POPULATION: usize = 600;
/// Rows of each population of the structure-change design.
const STRUCTURE_POPULATION: usize = 300;
/// Draws of each coalition's outcome law in a structure change. The baseline law of `y` is
/// simulated through `x` and the comparison law is not, so the Monte Carlo error of the two
/// simulated means does not cancel in their difference; every refit simulates on the same
/// stream, so that error is an offset the refit spread does not see. Its variance is at most
/// the sampling variance of the change times `rows per population / draws`: 1.5% at the base
/// sample size.
const STRUCTURE_SAMPLES: usize = 20_000;

fn populations(pop: usize) -> ChangeAttributionQuery {
    ChangeAttributionQuery::new(
        v(1),
        PopulationSelector::TimeRange { start: 0, end: pop },
        PopulationSelector::TimeRange { start: pop, end: 2 * pop },
    )
}

/// The mechanism-refit row bootstrap standard error of the total change under the robust
/// estimator, which composes the fitted linear mechanisms exactly: the intercept shift.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn attribute_distribution_change_robust_dag_frequentist_bootstrap_coverage() {
    let test = "attribute_distribution_change_robust_dag_frequentist_bootstrap_coverage";
    let statement = Statement::BootstrapEstimate;
    measure(test, "pair_rows", statement, &[""], Gate::Nominal, |rep| {
        let seed = stream_seed(0xF5A_0008, rep);
        let pop = grid_n(POPULATION);
        let columns = pair_columns(2 * pop, seed, |i| if i < pop { BASELINE } else { COMPARISON });
        let target = AttributeTarget::DistributionChange {
            change: populations(pop),
            estimator: ChangeEstimator::Robust { max_rows: 10_000, n_samples: 2_000, seed: 3 },
        };
        let builder = Study::tabular(pair_rows(&columns))
            .graph(dag(2, &[(0, 1)]))
            .query(CausalQuery::Attribute(AttributeQuery::new(target)));
        let (study, result) =
            execute(test, resampled(builder, statement), &ExecutionContext::for_tests(seed))?;
        let (bound, intervals) = score(
            test,
            statement,
            "Attribute",
            "attribute.distribution_change",
            2 * pop,
            &study,
            &result,
            attribution_points,
            &["total_change"],
        );
        Some(Scored { bound, intervals, truth: vec![COMPARISON - BASELINE] })
    });
}

/// A structure change from `x -> y` to no edge: the one structure player, `y`, swaps its parent
/// set and its mechanism, so the change it explains is the mean of `y` in the comparison
/// population less its mean under the baseline model, `3 - (1 + 2 E[x]) = 2`. `names` are the
/// quantities scored, each with that truth.
fn attribute_structure_change(
    test: &'static str,
    family: u64,
    statement: Statement,
    names: &'static [&'static str],
    labels: &[&'static str],
) {
    measure(test, "structure_rows", statement, labels, Gate::Nominal, |rep| {
        let seed = stream_seed(family, rep);
        let pop = grid_n(STRUCTURE_POPULATION);
        let target = AttributeTarget::StructureChange {
            change: populations(pop).with_components(AttributionComponents::Structure),
            comparison: Arc::from([]),
            measure: ChangeMeasure::MeanDifference,
            n_samples: STRUCTURE_SAMPLES,
            seed: 5,
        };
        let builder = Study::tabular(structure_rows(pop, seed))
            .graph(dag(2, &[(0, 1)]))
            .query(CausalQuery::Attribute(AttributeQuery::new(target)));
        let (study, result) =
            execute(test, resampled(builder, statement), &ExecutionContext::for_tests(seed))?;
        let (bound, intervals) = score(
            test,
            statement,
            "Attribute",
            "attribute.structure_change",
            2 * pop,
            &study,
            &result,
            attribution_points,
            names,
        );
        Some(Scored { bound, intervals, truth: vec![COMPARISON - BASELINE; names.len()] })
    });
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn attribute_structure_change_dag_frequentist_bootstrap_coverage() {
    attribute_structure_change(
        "attribute_structure_change_dag_frequentist_bootstrap_coverage",
        0xF5A_0009,
        Statement::BootstrapEstimate,
        &["total_change"],
        &[""],
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn attribute_structure_change_dag_bayesian_dirichlet_coverage() {
    attribute_structure_change(
        "attribute_structure_change_dag_bayesian_dirichlet_coverage",
        0xF5A_000A,
        Statement::Posterior,
        &["total_change", "component[1]"],
        &["total_change", "structure_y"],
    );
}

// ======================================================================== temporal

/// Rows of each population of the temporal change design.
const SERIES_POPULATION: usize = 600;
/// Length of the series an intrinsic influence is fitted on.
const INFLUENCE_SERIES: usize = 300;
/// Temporal work an intrinsic-influence test authorizes, so that the number of block-weight
/// refits is never what refuses a replicate.
const INFLUENCE_WORK: u64 = 4_000_000_000;

/// The names of the posterior columns that start with `prefix`, split into the column of `y`
/// (variable 1, which keeps its series id on the unfolding) and the single other one, the lagged
/// cause.
///
/// # Panics
///
/// When the posterior does not hold exactly those two columns of that kind.
fn own_and_lagged(test: &str, result: &StudyResult, prefix: &str) -> (String, String) {
    let names = posterior_names(result);
    let own = format!("{prefix}[1]");
    let others: Vec<&String> =
        names.iter().filter(|name| name.starts_with(prefix) && **name != own).collect();
    assert!(
        names.contains(&own) && others.len() == 1,
        "{test}: expected {own} and one lagged {prefix} column among {names:?}"
    );
    (own, others[0].clone())
}

/// The block-weight posterior of the mechanism components of a change of a lag-one chain whose
/// intercept shifts: the mechanism of `y` carries the whole change, the mechanism of the lagged
/// cause none.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn attribute_temporal_dag_bayesian_block_weights_component_coverage() {
    let test = "attribute_temporal_dag_bayesian_block_weights_component_coverage";
    let statement = Statement::Posterior;
    let labels = ["mechanism_y", "mechanism_lagged_x"];
    measure(test, "lag_chain_rows", statement, &labels, Gate::Nominal, |rep| {
        let seed = stream_seed(0xF5A_000B, rep);
        let pop = grid_n(SERIES_POPULATION);
        let data = lag_chain_rows(2 * pop, seed, |i| if i < pop { BASELINE } else { COMPARISON });
        let target = AttributeTarget::DistributionChange {
            change: populations(pop),
            estimator: ChangeEstimator::Sampled {
                measure: ChangeMeasure::MeanDifference,
                n_samples: 2_000,
                seed: 3,
            },
        };
        let builder = Study::series(data)
            .graph(lag_dag())
            .query(CausalQuery::Attribute(AttributeQuery::new(target)));
        let (study, result) =
            execute(test, resampled(builder, statement), &ExecutionContext::for_tests(seed))?;
        let truth = vec![COMPARISON - BASELINE, 0.0];
        if result.posterior.is_none() {
            return Some(Scored { bound: None, intervals: vec![[None, None]; 2], truth });
        }
        let (own, lagged) = own_and_lagged(test, &result, "component");
        let (bound, intervals) = score(
            test,
            statement,
            "Attribute",
            "attribute.distribution_change",
            2 * pop,
            &study,
            &result,
            attribution_points,
            &[own.as_str(), lagged.as_str()],
        );
        Some(Scored { bound, intervals, truth })
    });
}

/// The block-weight posterior of the variance contribution of each noise source of `y` on a
/// lag-one chain, held to at least nominal: `0.49` from the noise of `y` and `4 Var(x) = 4` from
/// the lagged cause.
///
/// Every refit draw is the refit's contribution plus the Monte Carlo error of its own nested
/// simulation, drawn afresh and independently of the refit, so an equal-tailed interval of the
/// draws is at least as wide as the posterior's own: the interval is conservative, by an amount
/// set by the simulation budget, as on a static table.
#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn attribute_temporal_intrinsic_influence_bayesian_block_weights_coverage() {
    let test = "attribute_temporal_intrinsic_influence_bayesian_block_weights_coverage";
    let statement = Statement::Posterior;
    let labels = ["noise_y", "noise_lagged_x"];
    let gating = Gate::AtLeast(
        "variance contribution of the noise source; refit draws carry independent \
         nested-simulation noise, so the interval is conservative",
    );
    measure(test, "lag_chain_rows", statement, &labels, gating, |rep| {
        let seed = stream_seed(0xF5A_000C, rep);
        let rows = grid_n(INFLUENCE_SERIES);
        let target = AttributeTarget::IntrinsicInfluence(IntrinsicInfluenceTarget {
            target: v(1),
            allocation: InfluenceAllocation::Exact,
            outer_draws: 100,
            inner_draws: 20,
            replicates: 3,
            fit_bootstrap_replicates: 0,
            level: REPORTED_LEVEL,
            max_sources: 8,
            seed: 11,
            noise: NoiseDeclaration::Independent,
        });
        let builder = Study::series(lag_chain_rows(rows, seed, |_| CENTER))
            .graph(lag_dag())
            .query(CausalQuery::Attribute(AttributeQuery::new(target)));
        let limits =
            TemporalMaterializationLimits { max_work: INFLUENCE_WORK, ..Default::default() };
        let ctx = ExecutionContext::for_tests(seed).with_temporal_materialization_limits(limits);
        let (study, result) = execute(test, resampled(builder, statement), &ctx)?;
        let truth = vec![NOISE * NOISE, SLOPE * SLOPE];
        if result.posterior.is_none() {
            return Some(Scored { bound: None, intervals: vec![[None, None]; 2], truth });
        }
        let (own, lagged) = own_and_lagged(test, &result, "source");
        let (bound, intervals) = score(
            test,
            statement,
            "Attribute",
            "attribute.intrinsic_influence",
            rows,
            &study,
            &result,
            attribution_points,
            &[own.as_str(), lagged.as_str()],
        );
        Some(Scored { bound, intervals, truth })
    });
}

// ====================================================================== simulation

/// Rows of the simulation designs.
const SIM_ROWS: usize = 300;
/// Draws of an interventional sample. Every refit simulates on the same stream, so the Monte
/// Carlo error of a simulated mean is an offset the refit spread does not see: its variance is
/// at most the sampling variance of that mean times `rows / draws`, 1% at the base sample size.
const SAMPLE_DRAWS: usize = 30_000;
/// Units and steps of the factual panel of a temporal counterfactual.
const PANEL_UNITS: usize = 40;
const PANEL_STEPS: usize = 2;

fn do_t() -> Arc<[Intervention]> {
    Arc::from([Intervention::set(v(1), Value::f64(1.0))])
}

/// The mean true outcome of the units `(t, y)` had `t` been 1: each unit keeps its `z` and its
/// noise, so its `y` moves by `1.5 (1 - t)`.
fn counterfactual_mean(t: &[f64], y: &[f64]) -> f64 {
    mean(y) + Y_ON_T * (1.0 - mean(t))
}

/// One simulation target on the chain. `design` gives the target and the truth of each quantity
/// in `names` from the replicate's seed and its observed `z`, `t`, `y` columns.
fn simulate(
    test: &'static str,
    family: u64,
    statement: Statement,
    names: &'static [&'static str],
    labels: &[&'static str],
    design: impl Fn(u64, &[Vec<f64>; 3]) -> (SimulationTarget, Vec<f64>) + Sync,
) {
    measure(test, "chain_rows", statement, labels, Gate::Nominal, |rep| {
        let seed = stream_seed(family, rep);
        let rows = grid_n(SIM_ROWS);
        let columns = chain_columns(rows, seed);
        let (target, truth) = design(seed, &columns);
        let functional = format!("simulate.{}", target.as_str());
        let builder = Study::tabular(chain_rows(&columns))
            .graph(chain_dag())
            .query(CausalQuery::Simulate(SimulateQuery::new(target)));
        let (study, result) =
            execute(test, resampled(builder, statement), &ExecutionContext::for_tests(seed))?;
        let (bound, intervals) = score(
            test,
            statement,
            "Simulate",
            &functional,
            rows,
            &study,
            &result,
            simulation_points,
            names,
        );
        Some(Scored { bound, intervals, truth })
    });
}

/// Draws under `do(t = 1)`: the mean of `z` is 0 and the mean of `y` is 1.5. The mean of `t`
/// is the level it is held at in every refit and is not scored.
fn interventional_sample(test: &'static str, family: u64, statement: Statement) {
    simulate(
        test,
        family,
        statement,
        &["draw_mean[0]", "draw_mean[2]"],
        &["mean_z", "mean_y"],
        |seed, _| {
            let target = SimulationTarget::InterventionalSample {
                interventions: do_t(),
                draws: SAMPLE_DRAWS,
                // A stream per replicate, so the simulation offset is not one draw shared by all.
                seed: stream_seed(seed, 1),
            };
            (target, vec![0.0, Y_ON_T])
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn simulate_interventional_sample_dag_bayesian_dirichlet_coverage() {
    interventional_sample(
        "simulate_interventional_sample_dag_bayesian_dirichlet_coverage",
        0xF5A_000D,
        Statement::Posterior,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn simulate_interventional_sample_dag_frequentist_bootstrap_coverage() {
    interventional_sample(
        "simulate_interventional_sample_dag_frequentist_bootstrap_coverage",
        0xF5A_000E,
        Statement::BootstrapFields,
    );
}

/// The mean effect on `y` of moving each unit's `t` from 0 to 1 after abduction: 1.5 for every
/// unit.
fn unit_effects(test: &'static str, family: u64, statement: Statement) {
    simulate(test, family, statement, &["mean_ite"], &[""], |_, _| {
        let target = SimulationTarget::UnitEffects {
            treatment: v(1),
            outcome: v(2),
            active: 1.0,
            control: 0.0,
        };
        (target, vec![Y_ON_T])
    });
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn simulate_unit_effects_dag_bayesian_dirichlet_coverage() {
    unit_effects(
        "simulate_unit_effects_dag_bayesian_dirichlet_coverage",
        0xF5A_000F,
        Statement::Posterior,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn simulate_unit_effects_dag_frequentist_bootstrap_coverage() {
    unit_effects(
        "simulate_unit_effects_dag_frequentist_bootstrap_coverage",
        0xF5A_0010,
        Statement::BootstrapEstimate,
    );
}

/// The mean of `y` over the rows of the table had `t` been 1, scored against the mean of the
/// rows' true counterfactual outcomes.
fn counterfactual(test: &'static str, family: u64, statement: Statement) {
    simulate(test, family, statement, &["world[0].outcome[2]"], &[""], |_, columns| {
        let target =
            SimulationTarget::Counterfactual { interventions: do_t(), outcomes: Arc::from([v(2)]) };
        (target, vec![counterfactual_mean(&columns[1], &columns[2])])
    });
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn simulate_counterfactual_dag_bayesian_dirichlet_coverage() {
    counterfactual(
        "simulate_counterfactual_dag_bayesian_dirichlet_coverage",
        0xF5A_0011,
        Statement::Posterior,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn simulate_counterfactual_dag_frequentist_bootstrap_coverage() {
    counterfactual(
        "simulate_counterfactual_dag_frequentist_bootstrap_coverage",
        0xF5A_0012,
        Statement::BootstrapFields,
    );
}

/// A factual panel of [`PANEL_UNITS`] units over [`PANEL_STEPS`] steps, each step an independent
/// draw of the chain law, replayed with `t` held at 1 at every step: the mean of `y` at each
/// step, scored against the mean of that step's true counterfactual outcomes. The panel is laid
/// out as the query reads it, step by step with one block of values per variable. The means of
/// `z` (replayed unchanged) and of `t` (held) do not vary across refits and are not scored.
fn temporal_counterfactual(test: &'static str, family: u64, statement: Statement) {
    simulate(
        test,
        family,
        statement,
        &["step[0].node[2]", "step[1].node[2]"],
        &["step0_y", "step1_y"],
        |seed, _| {
            let mut panel = Vec::with_capacity(3 * PANEL_UNITS * PANEL_STEPS);
            let mut truth = Vec::with_capacity(PANEL_STEPS);
            for step in 0..PANEL_STEPS {
                let [z, t, y] = chain_columns(PANEL_UNITS, stream_seed(seed, 2 + step as u64));
                truth.push(counterfactual_mean(&t, &y));
                panel.extend(z);
                panel.extend(t);
                panel.extend(y);
            }
            let schedule = InterventionSequence::new(vec![SequencedIntervention::new(
                Intervention::set(v(1), Value::f64(1.0)),
                TemporalPolicy::Sustained { from: 0, until: PANEL_STEPS as i32 - 1 },
            )]);
            let target = SimulationTarget::TemporalCounterfactual {
                schedule,
                n_rows: PANEL_UNITS,
                n_steps: PANEL_STEPS,
                window: 0,
                panel: Arc::from(panel),
                history: Arc::from([]),
            };
            (target, truth)
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn simulate_temporal_counterfactual_dag_bayesian_dirichlet_coverage() {
    temporal_counterfactual(
        "simulate_temporal_counterfactual_dag_bayesian_dirichlet_coverage",
        0xF5A_0013,
        Statement::Posterior,
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn simulate_temporal_counterfactual_dag_frequentist_bootstrap_coverage() {
    temporal_counterfactual(
        "simulate_temporal_counterfactual_dag_frequentist_bootstrap_coverage",
        0xF5A_0014,
        Statement::BootstrapFields,
    );
}
