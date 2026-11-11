//! Whole-method coverage of the interval of the frequentist `randomized.quantile` estimator
//! on a randomized binary treatment, scored through the public Study API.
//!
//! The procedure scored is the estimator as the facade runs it on a supplied Dag `t -> y`
//! with no adjustment set: the piecewise-linear inversion of the arm distribution functions
//! at the declared quantile, the influence-function standard error with each arm's density
//! taken as its Siddiqui (1960) sparsity estimate at the Hall and Sheather (1988) bandwidth
//! (not the slope of the pooled grid bracket), and the normal interval
//! `estimate +- 1.959963984540054 * se` (`analytic_se`). Its estimate, standard error and
//! interval are pinned against a hand-derived table in
//! `randomized_quantile_known_truth.rs`; this file measures whether the interval covers.
//!
//! Design and truth: treatment `t ~ Bernoulli(1/2)` independent of everything else, and the
//! outcome `y = 0.5 t + (1 + 0.3 t) e` with `e ~ N(0, 1)`. Each arm's outcome law is
//! symmetric about its location, so the arm medians are `0` (control) and `0.5` (treated)
//! and the median treatment effect is exactly `0.5`. Rows are on the three-point grid
//! around 400 (200, 400, 800), `n_sim()` datasets. A dataset the facade refuses is a skip
//! and counts as a miss, never a free pass.
//!
//! Assertion: `CoverageTally::assert`, the harness band around the nominal 0.95 for the
//! first `n_sim()` replicates, rechecked at 2000 near the edge. The quantile is the median
//! because the estimator's interval rests on a smooth density at the quantile, which the
//! Gaussian arms have at the median; no tail quantile or discrete outcome is claimed here.
//!
//! The record is keyed by the construction the facade reports for the executed study (read
//! from its own contract by `bind_reported`), so it can only describe the interval the
//! route actually publishes.
//!
//! Ignored tests run via `scripts/measure_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

mod common;

use antecedent::{EstimatorId, RefuteSuite, Study};
use antecedent_core::{AverageEffectQuery, ExecutionContext, OutcomeFunctional, VariableId};
use antecedent_data::TabularData;
use antecedent_graph::{Dag, DenseNodeId};
use common::calibration::{
    CoverageTally, REPORTED_LEVEL, RecordKey, Z95, gaussian, grid_n, grid_seed, map_replicates,
    n_sim, normal_interval, stream_seed, unit_uniform,
};
use common::calibration_bind::{bind_reported, constructions};

const BASE_ROWS: usize = 400;
const SEED_BASE: u64 = 0x0C47_0001;
/// The median treatment effect of the design: the treated arm's median minus the control's.
const TRUTH: f64 = 0.5;

fn vid(raw: u32) -> VariableId {
    VariableId::from_raw(raw)
}

fn dataset(n: usize, seed: u64) -> TabularData {
    let mut noise = gaussian(seed);
    let mut treatment = vec![0.0; n];
    let mut outcome = vec![0.0; n];
    for i in 0..n {
        let treated = unit_uniform(stream_seed(seed, i as u64)) < 0.5;
        treatment[i] = f64::from(treated);
        let shift = if treated { 0.5 } else { 0.0 };
        let scale = if treated { 1.3 } else { 1.0 };
        outcome[i] = shift + scale * noise();
    }
    TabularData::from_f64_columns([("t", treatment.as_slice()), ("y", outcome.as_slice())])
        .expect("the simulated columns are finite and aligned")
}

fn median_effect_query() -> AverageEffectQuery {
    AverageEffectQuery::binary_ate(vid(0), vid(1))
        .with_outcome_functional(OutcomeFunctional::quantile(0.5))
}

fn graph() -> Dag {
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    graph
}

type Reported =
    Vec<(common::calibration::Construction, common::calibration::ScopeFacts, &'static str)>;

/// The estimate, its analytic standard error and the construction the facade reported, or
/// `None` when the study is refused.
fn execute(n: usize, seed: u64) -> Option<(f64, f64, Reported)> {
    let study = Study::tabular(dataset(n, seed))
        .query(median_effect_query())
        .estimator(EstimatorId::RandomizedQuantile)
        .graph(graph())
        .refute(RefuteSuite::None)
        .bootstrap_replicates(0)
        .build()
        .ok()?;
    let result = study.run(&ExecutionContext::for_tests(seed)).ok()?;
    let reported = constructions(&study.inspect().ok()?, &result);
    let estimate = result.estimate();
    let effect = estimate.as_effect()?;
    Some((effect.ate, effect.uncertainty.analytic.se, reported))
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn randomized_quantile_median_effect_analytic_interval() {
    let n = grid_n(BASE_ROWS);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "randomized_quantile_median_effect_analytic_interval",
            dgp: "bernoulli_half_gaussian_arms_known_median_effect",
            interval: "analytic_se",
        },
        REPORTED_LEVEL,
    );
    let runs = map_replicates(n_sim(), |rep| execute(n, grid_seed(SEED_BASE + rep)));
    for run in &runs {
        let Some((ate, se, reported)) = run else {
            tally.skip();
            continue;
        };
        bind_reported(&mut tally, reported);
        tally.record(normal_interval(*ate, Some(*se), Z95), TRUTH);
    }
    tally.assert();
}

/// One replicate, quickly: the estimate sits near the known median effect, the standard
/// error is positive and the normal interval it publishes is ordered and brackets the truth.
#[test]
fn one_replicate_reports_a_valid_interval_near_the_known_median_effect() {
    let (ate, se, reported) = execute(BASE_ROWS, grid_seed(SEED_BASE)).expect("the study runs");
    assert!(!reported.is_empty(), "the facade reports its construction");
    assert!(se.is_finite() && se > 0.0, "standard error {se}");
    let (lower, upper) = normal_interval(ate, Some(se), Z95).expect("a finite interval");
    assert!(lower < ate && ate < upper, "[{lower}, {upper}] around {ate}");
    assert!((ate - TRUTH).abs() < 5.0 * se, "estimate {ate} vs truth {TRUTH}, se {se}");
}
