//! Whole-method coverage of the delta-method Wald interval of a descriptive comparison on a
//! transformed scale, against the known population value on that scale.
//!
//! The procedure scored is `transform_mean_pair` end to end for a pair of arm proportions with
//! their sampling variances (independent arms, so the pair's covariance is zero) and
//! `interval(i, 0.95)`, the equal-tailed delta-method Wald interval of contrast `i`. Its
//! construction is pinned against the hand-derived gradients and variances in
//! `descriptive_comparison.rs`; this file measures whether the interval covers.
//!
//! Design and truth: two independent Bernoulli arms with proportions `0.6` (active) and `0.4`
//! (control), `n` rows per arm on the three-point grid around 200 (100, 200, 400), `n_sim()`
//! datasets. The pair is `(p1_hat, p0_hat)` with variances `p_hat (1 - p_hat) / n`. Two scales
//! are scored on the same datasets, each as its own record: the risk difference `0.6 - 0.4 =
//! 0.2` and the log risk ratio `ln(0.6 / 0.4)`. A dataset whose arm proportion is 0 or 1
//! (transform undefined) is a skip and counts as a miss; at these sizes it has negligible
//! probability.
//!
//! Assertion: `CoverageTally::assert`, the harness band around the nominal 0.95 for the first
//! `n_sim()` replicates, rechecked at 2000 near the edge. The interval rests on asymptotic
//! normality and a linearization away from the boundary of the scale; nothing is claimed near
//! a proportion of 0 or 1.
//!
//! Ignored tests run via `scripts/measure_calibration_groups.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;

use antecedent_estimate::{MeanPair, ReportingScale, transform_mean_pair};
use calibration::{
    Construction, CoverageTally, RecordKey, ScopeFacts, grid_n, grid_seed, map_replicates, n_sim,
    stream_seed, unit_uniform,
};

const LEVEL: f64 = 0.95;
const BASE_ROWS: usize = 200;
const ACTIVE: f64 = 0.6;
const CONTROL: f64 = 0.4;
const SEED_BASE: u64 = 0x5D1C_0001;

fn proportion(p: f64, n: usize, seed: u64, arm: u64) -> f64 {
    let successes =
        (0..n).filter(|&i| unit_uniform(stream_seed(seed ^ (arm << 40), i as u64)) < p).count();
    successes as f64 / n as f64
}

/// The interval of `scale` for one simulated pair of arms of `n` rows, or `None` when the pair
/// has no defined transform.
fn interval(scale: ReportingScale, n: usize, seed: u64) -> Option<(f64, f64)> {
    let (p1, p0) = (proportion(ACTIVE, n, seed, 1), proportion(CONTROL, n, seed, 2));
    let variance = |p: f64| p * (1.0 - p) / n as f64;
    let pair = MeanPair::new(p1, p0, Some([variance(p1), 0.0, variance(p0)]));
    let transform = transform_mean_pair(&pair, &[scale]).ok()?;
    transform.interval(0, LEVEL)
}

fn construction(functional: &str) -> Construction {
    Construction {
        query: "DescriptiveComparison".into(),
        graph_class: "NotApplicable".into(),
        structure: "fixed".into(),
        modality: "summary".into(),
        inference: "Frequentist".into(),
        estimator: "descriptive.delta_method_transform".into(),
        interval_method: "delta_method_wald".into(),
        se_kind: "delta_method".into(),
        dependence: "independent_arms".into(),
        posterior: String::new(),
        functional: functional.into(),
        identification: "point".into(),
        reported_level: LEVEL,
    }
}

fn measure(test: &'static str, scale: ReportingScale, functional: &str, truth: f64) {
    let n = grid_n(BASE_ROWS);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test,
            dgp: "two_bernoulli_arms_known_population_proportions",
            interval: "delta_method_wald",
        },
        LEVEL,
    );
    let runs = map_replicates(n_sim(), |rep| interval(scale, n, grid_seed(SEED_BASE + rep)));
    for run in runs {
        let Some(band) = run else {
            tally.skip();
            continue;
        };
        tally.bind_validation(
            &construction(functional),
            ScopeFacts {
                row_count: 2 * n as u64,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
            "not_applicable",
        );
        tally.record(Some(band), truth);
    }
    tally.assert();
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration_groups.sh"]
fn descriptive_risk_difference_delta_method_interval() {
    measure(
        "descriptive_risk_difference_delta_method_interval",
        ReportingScale::RiskDifference,
        "risk_difference",
        ACTIVE - CONTROL,
    );
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration_groups.sh"]
fn descriptive_log_risk_ratio_delta_method_interval() {
    measure(
        "descriptive_log_risk_ratio_delta_method_interval",
        ReportingScale::LogRiskRatio,
        "log_risk_ratio",
        (ACTIVE / CONTROL).ln(),
    );
}
