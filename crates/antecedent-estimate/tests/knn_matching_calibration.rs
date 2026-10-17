//! Coverage of the analytic interval of covariate-distance matching with three nearest donors.
//!
//! Population: eight standard-normal covariates, treatment `T ~ Bernoulli(σ(−0.8 + 0.3 Σ zₖ))`
//! and outcome `Y = 0.5 T + 0.4 Σ zₖ + N(0, 1)`, so the effect on the treated is `0.5` and the
//! covariates confound. The interval is `ATT ± z₀.₉₇₅ · se` from
//! [`KnnDistanceMatching`] with three neighbours, scored at 10,000, 25,000 and 100,000 rows
//! (the grid points the gate runs on its own seeds). Averaging three donors is a separate
//! construction from one-nearest-neighbour matching, so a coverage measured for the latter
//! says nothing about it; this design measures it. Besides coverage at the nominal level, the
//! design asserts the precision target (median interval width at most 1.5 times the effect) and
//! that at least 90 percent of replicates report an interval.
//!
//! The ignored design runs through `scripts/gate_calibration.sh`; it emits no coverage record
//! (the estimator is not reachable through the facade) and gates pass or fail.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_precision_loss, reason = "bounded measurement dimensions")]

#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, CausalRng, ExecutionContext, TargetPopulation, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::{KnnDistanceMatching, PropensityEstimationWorkspace};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_kernels::{sigmoid, standard_normal};
use calibration::{
    CoverageTally, REPORTED_LEVEL, SampleGrid, Z95, grid_seed, map_replicates, n_sim,
};
use std::sync::Arc;

const EFFECT: f64 = 0.5;
const COVARIATES: usize = 8;
const NEIGHBORS: usize = 3;
/// Seeds of the two stated sizes' populations; replicate `r` draws from `SEED + r`.
const SEED: u64 = 3_151_358;
/// 10,000, 25,000 and 100,000 rows.
const ROWS: SampleGrid =
    SampleGrid { name: "knn_matching_rows", factors: [(1, 10), (1, 4), (1, 1)] };
const MAX_ROWS: usize = 100_000;
/// Median interval width, as a multiple of the effect, the precision target allows.
const MAX_MEDIAN_WIDTH_OVER_EFFECT: f64 = 1.5;
const MIN_REPORTING_RATE: f64 = 0.9;

fn draw(rows: usize, seed: u64) -> (TabularData, IdentifiedEstimand) {
    let mut rng = CausalRng::from_seed(grid_seed(seed));
    let mut columns: Vec<Vec<f64>> = (0..COVARIATES).map(|_| Vec::with_capacity(rows)).collect();
    let (mut treatment, mut outcome) = (Vec::with_capacity(rows), Vec::with_capacity(rows));
    for _ in 0..rows {
        let mut sum = 0.0;
        for column in &mut columns {
            let z = standard_normal(&mut rng);
            column.push(z);
            sum += z;
        }
        let treated = f64::from(rng.next_f64() < sigmoid(-0.8 + 0.3 * sum));
        treatment.push(treated);
        outcome.push(EFFECT * treated + 0.4 * sum + standard_normal(&mut rng));
    }
    let names: Vec<String> = (0..COVARIATES).map(|k| format!("z{k}")).collect();
    let mut table: Vec<(&str, &[f64])> =
        vec![("t", treatment.as_slice()), ("y", outcome.as_slice())];
    table.extend(
        names.iter().zip(&columns).map(|(name, column)| (name.as_str(), column.as_slice())),
    );
    let data = TabularData::from_f64_columns(table).expect("equal-length columns");
    let adjustment: Vec<VariableId> =
        (2..2 + u32::try_from(COVARIATES).unwrap()).map(VariableId::from_raw).collect();
    let estimand = IdentifiedEstimand::backdoor(
        "backdoor.adjustment",
        Arc::from(adjustment),
        ExprId::from_raw(0),
    );
    (data, estimand)
}

/// ATT and its analytic standard error from the three-neighbour estimator, `None` on a refusal.
fn replicate(rows: usize, seed: u64) -> Option<(f64, f64)> {
    let (data, estimand) = draw(rows, seed);
    let knn = KnnDistanceMatching::new(NEIGHBORS).ok()?;
    let query = AverageEffectQuery::binary_ate(VariableId::from_raw(0), VariableId::from_raw(1))
        .with_target_population(TargetPopulation::Treated);
    let prepared = knn.prepare(&data, &estimand, &query).ok()?;
    let effect = knn
        .fit(
            &prepared,
            &mut PropensityEstimationWorkspace::default(),
            &ExecutionContext::for_tests(1),
            AssumptionSet::new(),
        )
        .ok()?;
    Some((effect.ate, effect.se_analytic()))
}

fn interval((ate, se): (f64, f64)) -> Option<(f64, f64)> {
    (ate.is_finite() && se.is_finite() && se > 0.0).then_some((ate - Z95 * se, ate + Z95 * se))
}

/// The interval of a sample is finite, ordered and centred near the effect on the treated.
#[test]
fn knn_matching_replicates_are_finite_and_centred_on_the_effect() {
    let fits: Vec<(f64, f64)> =
        (0..24).map(|r| replicate(2_000, SEED + r).expect("a fit")).collect();
    let mean = fits.iter().map(|&(ate, _)| ate).sum::<f64>() / fits.len() as f64;
    assert!((mean - EFFECT).abs() < 0.1, "mean ATT {mean}");
    for fit in fits {
        let (lo, hi) = interval(fit).expect("a finite interval");
        assert!(lo < hi && hi - lo < 1.0, "[{lo}, {hi}]");
    }
}

#[test]
#[ignore = "calibration: run via scripts/gate_calibration.sh"]
fn knn_matching_interval_ci_coverage() {
    let rows = ROWS.n(MAX_ROWS);
    let mut tally = CoverageTally::new("knn_matching_interval", REPORTED_LEVEL);
    let mut widths = Vec::new();
    let mut reported = 0u32;
    let fits = map_replicates(n_sim(), |rep| replicate(rows, SEED + rep));
    let attempts = fits.len();
    for fit in fits {
        match fit.and_then(interval) {
            Some((lo, hi)) => {
                reported += 1;
                widths.push(hi - lo);
                tally.record(Some((lo, hi)), EFFECT);
            }
            None => tally.skip(),
        }
    }
    widths.sort_by(f64::total_cmp);
    let median_width = widths.get(widths.len() / 2).copied().unwrap_or(f64::INFINITY);
    eprintln!(
        "knn matching interval: rows {rows}, reported {reported} of {attempts}, median width \
         {median_width:.4}"
    );
    assert!(
        f64::from(reported) >= MIN_REPORTING_RATE * attempts as f64,
        "only {reported} of {attempts} replicates reported an interval"
    );
    assert!(
        median_width <= MAX_MEDIAN_WIDTH_OVER_EFFECT * EFFECT,
        "median width {median_width} exceeds {MAX_MEDIAN_WIDTH_OVER_EFFECT} times the effect"
    );
    tally.assert();
}
