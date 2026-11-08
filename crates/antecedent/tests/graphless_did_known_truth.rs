//! Independent known-truth checks for the three retained graphless difference-in-differences
//! rows: the balanced two-period panel, the four-cell repeated cross-section and the
//! never-treated staggered event study.
//!
//! Each fixture is built so that the answer is analytic, not a function of the production code:
//!
//! * the data are an exact additive model (unit level + period shift + a known adoption effect)
//!   plus a cluster-level error whose sign alternates inside every arm (or cell), so every
//!   group mean of the error is exactly zero and the estimand is recovered exactly;
//! * the cluster score of an observation is therefore `+/- error / group_size`, and the CR1
//!   variance is `G / (G - 1) * sum_c score_c^2`, written out here from the error vector alone;
//! * the pointwise 95% interval is `estimate +/- 1.959963984540054 * se` with the quantile
//!   hard-coded, not read from the library.
//!
//! These are exact-value checks of the estimator, its standard error and its interval
//! construction. The repeated-sampling coverage tests in `panel_did_interval_calibration.rs`
//! are calibration evidence and are a separate claim.
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_lossless,
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "fixture derives f64 values from small nonnegative integer counters"
)]

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{PanelDidQuery, Study};
use antecedent_core::VariableId;
use antecedent_data::TabularData;

const TRUTH: f64 = 2.0;
const Z975: f64 = 1.959_963_984_540_054;
const TOL: f64 = 1e-11;

/// A fixed pseudo-random value in (0.5, 1.5) from an integer key (splitmix64 finalizer).
fn spread(key: u64) -> f64 {
    let mut z = key.wrapping_add(0x9E37_79B9_7F4A_7C15);
    z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
    z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
    z ^= z >> 31;
    0.5 + (z >> 11) as f64 / (1u64 << 53) as f64
}

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= TOL * (1.0 + want.abs()), "{what}: got {got}, want {want}");
}

/// `G / (G - 1) * sum score^2` square-rooted, for one score per cluster.
fn cr1_se(scores: &[f64]) -> f64 {
    let g = scores.len() as f64;
    (g / (g - 1.0) * scores.iter().map(|s| s * s).sum::<f64>()).sqrt()
}

#[test]
fn panel_2x2_matches_the_analytic_effect_cluster_sandwich_and_interval() {
    let per_arm = 30_usize;
    let mut subjects = Vec::new();
    let mut clusters = Vec::new();
    let mut treated = Vec::new();
    let mut post = Vec::new();
    let mut outcome = Vec::new();
    let mut error_by_cluster = Vec::new();
    for cluster in 0..2 * per_arm {
        let active = cluster >= per_arm;
        // Alternating sign inside the arm: the arm mean of the error is exactly zero.
        let within_arm = cluster % per_arm;
        let sign = if within_arm.is_multiple_of(2) { 1.0 } else { -1.0 };
        let error = sign * spread(1_000 + (within_arm / 2) as u64 + 100 * u64::from(active));
        error_by_cluster.push((active, error));
        for subject in 0..2_usize {
            let level = 10.0 * spread((cluster * 2 + subject) as u64);
            for after in [false, true] {
                subjects.push(Arc::<str>::from(format!("s-{cluster}-{subject}")));
                clusters.push(Arc::<str>::from(format!("c-{cluster}")));
                treated.push(active);
                post.push(after);
                let shift =
                    if after { 0.5 + if active { TRUTH } else { 0.0 } + error } else { 0.0 };
                outcome.push(level + shift);
            }
        }
    }
    let query = PanelDidQuery::new(VariableId::from_raw(0), treated, post, subjects, clusters);
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let ctx = ExecutionContext::for_tests(11);
    let prepared = Study::tabular(data).query(query).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared
        .estimate(&TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap(), &ctx)
        .unwrap();
    let fit = result.panel_did().unwrap();

    // Each arm has 60 subjects; two subjects per cluster share the cluster error.
    let n_group = (2 * per_arm) as f64;
    let scores: Vec<f64> = error_by_cluster
        .iter()
        .map(|&(active, e)| (if active { 1.0 } else { -1.0 }) * 2.0 * e / n_group)
        .collect();
    let se = cr1_se(&scores);
    close(fit.effect, TRUTH, "panel effect equals the planted adoption effect");
    close(fit.standard_error, se, "panel CR1 standard error");
    let interval = fit.interval_95.expect("30 clusters per arm publish an interval");
    close(interval[0], TRUTH - Z975 * se, "panel interval lower");
    close(interval[1], TRUTH + Z975 * se, "panel interval upper");
    assert_eq!((fit.treated_subjects, fit.comparison_subjects, fit.clusters), (60, 60, 60));
}

#[test]
fn repeated_cross_section_matches_the_analytic_effect_cluster_sandwich_and_interval() {
    let rows = 30_usize; // rows (one cluster each) per group-by-period cell
    let mut treated = Vec::new();
    let mut post = Vec::new();
    let mut subjects = Vec::new();
    let mut clusters = Vec::new();
    let mut outcome = Vec::new();
    let mut scores = Vec::new();
    for group in [false, true] {
        for after in [false, true] {
            for row in 0..rows {
                let sign = if row % 2 == 0 { 1.0 } else { -1.0 };
                let key =
                    5_000 + (row / 2) as u64 + 100 * u64::from(group) + 1_000 * u64::from(after);
                let error = sign * spread(key);
                treated.push(group);
                post.push(after);
                subjects.push(Arc::<str>::from(format!("s-{group}-{after}-{row}")));
                clusters.push(Arc::<str>::from(format!("c-{group}-{after}-{row}")));
                outcome.push(
                    3.0 * f64::from(group)
                        + 0.5 * f64::from(after)
                        + TRUTH * f64::from(group && after)
                        + error,
                );
                // The cell mean of the error is exactly zero, so the score is error / cell size.
                let diagonal = group == after;
                scores.push((if diagonal { 1.0 } else { -1.0 }) * error / rows as f64);
            }
        }
    }
    let query = PanelDidQuery::repeated_cross_section(
        VariableId::from_raw(0),
        treated,
        post,
        subjects,
        clusters,
    );
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let ctx = ExecutionContext::for_tests(12);
    let prepared = Study::tabular(data).query(query).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared
        .estimate(&TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap(), &ctx)
        .unwrap();
    let fit = result.panel_did().unwrap();

    let se = cr1_se(&scores);
    close(fit.effect, TRUTH, "repeated cross-section effect equals the planted effect");
    close(fit.standard_error, se, "repeated cross-section CR1 standard error");
    let interval = fit.interval_95.expect("30 clusters per cell publish an interval");
    close(interval[0], TRUTH - Z975 * se, "repeated cross-section interval lower");
    close(interval[1], TRUTH + Z975 * se, "repeated cross-section interval upper");
}

#[test]
fn staggered_event_study_matches_the_analytic_event_time_effects_and_intervals() {
    let per_group = 24_usize;
    // Period weights on the unit error: the contrast of period p against the baseline p = 2
    // carries error * (w_p - w_2).
    let weight = [0.0, 1.0, 0.25, 2.0, -1.0]; // index 0 unused; periods 1..=4
    let mut subjects = Vec::new();
    let mut clusters = Vec::new();
    let mut periods = Vec::new();
    let mut cohorts = Vec::new();
    let mut outcome = Vec::new();
    let mut errors = [Vec::new(), Vec::new()];
    for (group, cohort) in [0_i64, 3].into_iter().enumerate() {
        for unit in 0..per_group {
            let sign = if unit % 2 == 0 { 1.0 } else { -1.0 };
            let error = sign * spread(9_000 + (unit / 2) as u64 + 100 * group as u64);
            errors[group].push(error);
            let level = 7.0 * spread(20_000 + (unit + 100 * group) as u64);
            let id = Arc::<str>::from(format!("u-{group}-{unit}"));
            for period in 1..=4_i64 {
                subjects.push(id.clone());
                clusters.push(id.clone());
                periods.push(period);
                cohorts.push(cohort);
                let adoption = if cohort == 3 && period >= 3 {
                    TRUTH + 0.5 * (period - 3) as f64
                } else {
                    0.0
                };
                outcome
                    .push(level + 0.3 * period as f64 + adoption + error * weight[period as usize]);
            }
        }
    }
    let query = PanelDidQuery::staggered_event_study(
        VariableId::from_raw(0),
        subjects,
        clusters,
        periods,
        cohorts,
    );
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let ctx = ExecutionContext::for_tests(13);
    let prepared = Study::tabular(data).query(query).build().unwrap().prepare(&ctx).unwrap();
    let result = prepared
        .estimate(&TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap(), &ctx)
        .unwrap();
    let fit = result.panel_did().unwrap();

    // Contrasts against baseline period 2 exist for periods 1, 3 and 4 of cohort 3.
    let expected_periods = [1_i64, 3, 4];
    assert_eq!(fit.event_time_effects.len(), expected_periods.len());
    assert_eq!(fit.event_time_intervals_95.len(), expected_periods.len());
    for (k, &period) in expected_periods.iter().enumerate() {
        let effect = &fit.event_time_effects[k];
        assert_eq!((effect.cohort, effect.period, effect.event_time), (3, period, period - 3));
        let truth = if period >= 3 { TRUTH + 0.5 * (period - 3) as f64 } else { 0.0 };
        let delta_weight = weight[period as usize] - weight[2];
        let n = per_group as f64;
        // Control and treated errors each sum to zero, so the group means of the error cancel
        // and the score of a unit is +/- error * delta_weight / n; every unit is its own cluster.
        let scores: Vec<f64> = errors[0]
            .iter()
            .map(|e| -e * delta_weight / n)
            .chain(errors[1].iter().map(|e| e * delta_weight / n))
            .collect();
        let se = cr1_se(&scores);
        close(effect.effect, truth, &format!("event-time effect at period {period}"));
        close(effect.standard_error, se, &format!("event-time CR1 se at period {period}"));
        assert_eq!(effect.clusters, 2 * per_group);
        match fit.event_time_intervals_95[k] {
            Some(interval) => {
                assert!(period >= 3, "only post-adoption contrasts publish an interval");
                close(interval[0], truth - Z975 * se, "event-time interval lower");
                close(interval[1], truth + Z975 * se, "event-time interval upper");
            }
            None => assert!(period < 3, "post-adoption contrast at 24+24 clusters publishes"),
        }
    }
}
