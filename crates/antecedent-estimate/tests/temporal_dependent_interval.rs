//! Dependence-preserving interval of the finite two-step sequence (2.3A, X5).
//!
//! The truth is an enumerated two-step dynamic SCM written out below: a pre-action
//! state `s0`, a first action `a1` randomized given `s0`, a step-2 covariate `l`
//! drawn from `P(l = 1 | s0, a1)`, a second action `a2` randomized given the
//! history, and a binary outcome with mean `m(s0, a1, l, a2)`. The panel is an
//! exact-count one: every unit holds all four `(s0, a1)` blocks in the SCM's
//! proportions, and a unit-level effect shifts every cell mean by plus or minus
//! 0.2 (equal numbers of each sign), so units are strongly dependent and the
//! population mean is exactly the SCM's.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "enumerated binary SCM fixtures use bounded nonnegative counts and indices"
)]

use antecedent_core::ExecutionContext;
use antecedent_estimate::EstimationError;
use antecedent_estimate::temporal_dependent_interval::{
    DependentIntervalConfig, INTERVAL_CALIBRATION_STATUS, INTERVAL_CLAIM, IntervalMethod,
    ObservedStateSequence, SequenceHistory, TemporalUnitPanel, UnitHistories,
    dependent_unit_interval, route_frozen_refusal,
};
use antecedent_estimate::temporal_initial_state::{
    FixedStateQuery, InitialStateLaw, InitialStatePopulation, InitialStateSpec, MarginalizedQuery,
};

/// `P(l = 1 | s0, a1)` in tenths, indexed `[s0][a1]`.
const P_L1: [[u64; 2]; 2] = [[3, 7], [2, 8]];
/// Outcome mean in tenths, indexed `[s0][a1][l][a2]`.
const M10: [[[[i64; 2]; 2]; 2]; 2] =
    [[[[3, 4], [4, 6]], [[3, 5], [5, 7]]], [[[4, 3], [7, 6]], [[6, 3], [5, 7]]]];
const SEQUENCE: [u32; 2] = [0, 0];
/// The target initial-state law: state 1 is the mode.
const TARGET: [(u32, f64); 2] = [(0, 0.3), (1, 0.7)];

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn refused(error: EstimationError) -> (&'static str, String) {
    match error {
        EstimationError::Refused { code, message } => (code, message),
        other => panic!("expected a registered refusal, got {other:?}"),
    }
}

/// Response given `s0` enumerated from the SCM tables.
fn truth_given_state(sequence: [u32; 2], s0: usize) -> f64 {
    let (a1, a2) = (sequence[0] as usize, sequence[1] as usize);
    let p1 = P_L1[s0][a1] as f64 / 10.0;
    (0..2)
        .map(|l| {
            let pl = if l == 1 { p1 } else { 1.0 - p1 };
            pl * M10[s0][a1][l][a2] as f64 / 10.0
        })
        .sum()
}

/// All histories of unit `unit` (sign of the unit effect from its parity).
fn unit_rows(unit: usize) -> Vec<(u32, u32, u32, u32, f64)> {
    let shift: i64 = if unit % 2 == 0 { 2 } else { -2 };
    let mut rows = Vec::new();
    for s0 in 0..2_usize {
        for a1 in 0..2_usize {
            let n_one = 20 * P_L1[s0][a1];
            for l in 0..2_usize {
                let n_l = if l == 1 { n_one } else { 200 - n_one };
                for a2 in 0..2_usize {
                    let n_cell = n_l / 2;
                    let ones = n_cell / 10 * (M10[s0][a1][l][a2] + shift) as u64;
                    for k in 0..n_cell {
                        let y = if k < ones { 1.0 } else { 0.0 };
                        rows.push((s0 as u32, a1 as u32, l as u32, a2 as u32, y));
                    }
                }
            }
        }
    }
    rows
}

fn history(time_id: u64, row: (u32, u32, u32, u32, f64)) -> SequenceHistory {
    SequenceHistory { time_id, s0: row.0, a1: row.1, l2: row.2, a2: row.3, y: row.4 }
}

/// Complete repeated-unit histories for the studentized SCM fixture.
fn panel(units: usize) -> TemporalUnitPanel {
    let out = (0..units)
        .map(|unit| {
            let histories = unit_rows(unit)
                .into_iter()
                .enumerate()
                .map(|(t, row)| history(t as u64, row))
                .collect();
            UnitHistories { unit_id: unit as u64, histories }
        })
        .collect();
    TemporalUnitPanel::new("dynamic-scm-panel", Some(out)).unwrap()
}

fn config(seed: u64) -> DependentIntervalConfig {
    DependentIntervalConfig { replicates: 200, seed, ..DependentIntervalConfig::default() }
}

fn target_query() -> MarginalizedQuery {
    let law = InitialStateLaw::new(InitialStatePopulation::Target, "target-state", TARGET.to_vec())
        .unwrap();
    MarginalizedQuery::new(SEQUENCE, InitialStateSpec::Law(law)).unwrap()
}

fn target_truth() -> f64 {
    TARGET.iter().map(|(s0, mass)| mass * truth_given_state(SEQUENCE, *s0 as usize)).sum()
}

#[test]
fn x5_dynamic_scm_interval_point_equals_enumerated_truth() {
    let p = panel(30);
    assert_eq!(p.unit_count(), 30);
    assert_eq!(p.history_count(), 30 * 800);
    for (s0, hand) in [(0_u32, 0.33), (1, 0.46)] {
        let fixed = FixedStateQuery { sequence: SEQUENCE, s0 }.effect(&p).unwrap().value;
        assert!((fixed - truth_given_state(SEQUENCE, s0 as usize)).abs() < 1e-12);
        assert!((fixed - hand).abs() < 1e-12, "{fixed} vs hand-enumerated {hand}");
    }
    let interval = dependent_unit_interval(
        &p,
        &antecedent_estimate::temporal_dependent_interval::BalancedTemporalEstimator::Marginalized(
            target_query(),
        ),
        &config(7),
        &ctx(),
    )
    .unwrap();
    assert!((target_truth() - 0.421).abs() < 1e-12);
    assert!((interval.point - 0.421).abs() < 1e-12, "point {}", interval.point);
    assert_eq!(interval.estimand, "marginalized_initial_state");
}

#[test]
fn x5_dynamic_scm_studentized_interval_covers_independent_truth() {
    let truth = target_truth();
    let by_unit = dependent_unit_interval(
        &panel(30),
        &antecedent_estimate::temporal_dependent_interval::BalancedTemporalEstimator::Marginalized(
            target_query(),
        ),
        &config(7),
        &ctx(),
    )
    .unwrap();

    assert_eq!(by_unit.units, 30);
    assert_eq!(by_unit.failed, 0);
    assert_eq!(by_unit.replicates.len(), 200);
    assert!((by_unit.point - truth).abs() < 1e-12);
    // Analytic direction: the unit effect moves the whole value by +-0.2, so the
    // replicate sd across 30 whole units is about 0.2 * 2 * sqrt(0.25 / 30) = 0.037
    // (95% width about 0.14).
    assert!(by_unit.width() > 0.08 && by_unit.width() < 0.25, "width {}", by_unit.width());
    assert!(by_unit.contains(truth), "[{}, {}] misses {truth}", by_unit.lower, by_unit.upper);
    assert_eq!(by_unit.claim, INTERVAL_CLAIM);
    assert_eq!(by_unit.calibration, INTERVAL_CALIBRATION_STATUS);
    assert_eq!(by_unit.calibration, "unmeasured");
    assert!((by_unit.level - 0.95).abs() < 1e-12);
}

#[test]
fn x5_dynamic_scm_interval_replicate_ids_and_digests_reproduce() {
    let p = panel(30);
    let first = dependent_unit_interval(
        &p,
        &antecedent_estimate::temporal_dependent_interval::BalancedTemporalEstimator::Marginalized(
            target_query(),
        ),
        &config(11),
        &ctx(),
    )
    .unwrap();
    let again = dependent_unit_interval(
        &p,
        &antecedent_estimate::temporal_dependent_interval::BalancedTemporalEstimator::Marginalized(
            target_query(),
        ),
        &config(11),
        &ctx(),
    )
    .unwrap();
    assert_eq!(first.replicates, again.replicates);
    assert_eq!(first.replicate_digest(), again.replicate_digest());
    assert_eq!(first, again);
    let ids =
        first.replicates.iter().map(|r| r.replicate_id).collect::<std::collections::BTreeSet<_>>();
    assert_eq!(ids.len(), 200, "replicate ids are distinct");
    let other_seed = dependent_unit_interval(
        &p,
        &antecedent_estimate::temporal_dependent_interval::BalancedTemporalEstimator::Marginalized(
            target_query(),
        ),
        &config(12),
        &ctx(),
    )
    .unwrap();
    assert_ne!(first.replicate_digest(), other_seed.replicate_digest());

    // Changed unit identity: the same values under a renamed unit.
    let mut units = p.units().to_vec();
    units[3].unit_id = 9_999;
    let renamed = TemporalUnitPanel::new("dynamic-scm-panel", Some(units)).unwrap();
    assert_ne!(renamed.digest(), p.digest());
    let moved = dependent_unit_interval(
        &renamed,
        &antecedent_estimate::temporal_dependent_interval::BalancedTemporalEstimator::Marginalized(
            target_query(),
        ),
        &config(11),
        &ctx(),
    )
    .unwrap();
    assert_ne!(first.replicate_digest(), moved.replicate_digest());

    // Changed time identity and changed snapshot identity.
    let mut units = p.units().to_vec();
    units[0].histories.last_mut().unwrap().time_id += 1_000_000;
    let retimed = TemporalUnitPanel::new("dynamic-scm-panel", Some(units)).unwrap();
    assert_ne!(retimed.digest(), p.digest());
    let resnapped = TemporalUnitPanel::new("other-snapshot", Some(p.units().to_vec())).unwrap();
    assert_ne!(resnapped.digest(), p.digest());
}

#[test]
fn x5_dynamic_scm_interval_observed_state_estimand_is_labelled_distinctly() {
    let p = panel(30);
    let observed = dependent_unit_interval(
        &p,
        &antecedent_estimate::temporal_dependent_interval::BalancedTemporalEstimator::Observed(
            ObservedStateSequence { sequence: SEQUENCE },
        ),
        &config(5),
        &ctx(),
    )
    .unwrap();
    // The panel is half s0 = 0, half s0 = 1: 0.5 * 0.33 + 0.5 * 0.46.
    assert!((observed.point - 0.395).abs() < 1e-12, "{}", observed.point);
    assert_eq!(observed.estimand, "observed_initial_state");
    let fixed = dependent_unit_interval(
        &p,
        &antecedent_estimate::temporal_dependent_interval::BalancedTemporalEstimator::Fixed(
            FixedStateQuery { sequence: SEQUENCE, s0: 1 },
        ),
        &config(5),
        &ctx(),
    )
    .unwrap();
    assert_eq!(fixed.estimand, "fixed_initial_state");
    assert!((fixed.point - 0.46).abs() < 1e-12);
}

#[test]
fn x5_dynamic_scm_interval_refuses_bounds_unsupported_histories_and_cancellation() {
    let p = panel(30);
    let too_many = DependentIntervalConfig { replicates: 2001, ..config(1) };
    let (code, message) =
        refused(dependent_unit_interval(&p, &antecedent_estimate::temporal_dependent_interval::BalancedTemporalEstimator::Marginalized(target_query()), &too_many, &ctx()).unwrap_err());
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("temporal_interval.too_many_replicates"), "{message}");

    let few = panel(5);
    let (code, message) =
        refused(dependent_unit_interval(&few, &antecedent_estimate::temporal_dependent_interval::BalancedTemporalEstimator::Marginalized(target_query()), &config(1), &ctx()).unwrap_err());
    assert_eq!(code, "too_few_clusters");
    assert!(message.contains("temporal_interval.too_few_units"), "{message}");

    // Every history takes a1 = 1, so the sequence (0, 0) has no support at all.
    let units = (0..30_u64)
        .map(|u| UnitHistories {
            unit_id: u,
            histories: vec![history(0, (0, 1, 0, 0, 1.0)), history(1, (1, 1, 1, 1, 0.0))],
        })
        .collect();
    let unsupported = TemporalUnitPanel::new("no-support", Some(units)).unwrap();
    let query = FixedStateQuery { sequence: SEQUENCE, s0: 0 };
    let (code, message) =
        refused(dependent_unit_interval(&unsupported, &query, &config(1), &ctx()).unwrap_err());
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("temporal_interval.unsupported_history"), "{message}");

    let stopped = ctx();
    stopped.cancellation.cancel();
    let (code, _) =
        refused(dependent_unit_interval(&p, &antecedent_estimate::temporal_dependent_interval::BalancedTemporalEstimator::Marginalized(target_query()), &config(1), &stopped).unwrap_err());
    assert_eq!(code, "transport_budget_cancel");
}

#[test]
fn x5_unknown_unit_dependence_refuses_an_absent_map_and_incompatible_units() {
    let (code, message) = refused(TemporalUnitPanel::new("rows-only", None).unwrap_err());
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("temporal_interval.unknown_units"), "{message}");

    let one = |id: u64, times: &[u64]| UnitHistories {
        unit_id: id,
        histories: times.iter().map(|t| history(*t, (0, 0, 0, 0, 1.0))).collect(),
    };
    let cases = [
        ("duplicate unit ids", vec![one(1, &[0]), one(1, &[0])]),
        ("a unit without histories", vec![one(1, &[])]),
        ("non-ordered time", vec![one(1, &[2, 1])]),
        ("repeated time id", vec![one(1, &[1, 1])]),
    ];
    for (name, units) in cases {
        let (code, message) = refused(TemporalUnitPanel::new("s", Some(units)).unwrap_err());
        assert_eq!(code, "route_not_supported", "{name}");
        assert!(message.contains("temporal_interval.unknown_units"), "{name}: {message}");
    }
    let (_, message) = refused(TemporalUnitPanel::new("", Some(vec![one(1, &[0])])).unwrap_err());
    assert!(message.contains("temporal_interval.unknown_units"), "{message}");

    let nan = UnitHistories { unit_id: 1, histories: vec![history(0, (0, 0, 0, 0, f64::NAN))] };
    assert!(TemporalUnitPanel::new("s", Some(vec![nan])).is_err());
}

#[test]
fn x5_unknown_unit_dependence_public_route_stays_closed() {
    let (code, message) = refused(route_frozen_refusal());
    assert_eq!(code, "cell_not_licensed");
    assert!(message.contains("temporal_interval.route_frozen"), "{message}");
}

#[test]
fn studentized_balanced_unit_variance_and_pivots_match_independent_two_point_algebra() {
    let p = panel(30);
    let result = dependent_unit_interval(
        &p,
        &antecedent_estimate::temporal_dependent_interval::BalancedTemporalEstimator::Marginalized(
            target_query(),
        ),
        &DependentIntervalConfig { method: IntervalMethod::Studentized, ..config(719) },
        &ctx(),
    )
    .unwrap();
    let receipt = result.studentization.as_ref().unwrap();
    let truth = target_truth();
    assert!((result.point - truth).abs() < 1e-12);
    assert!((receipt.standard_error - 0.2 / (29_f64).sqrt()).abs() < 1e-12);
    let mut pivots = Vec::new();
    for ((record, se), pivot) in
        result.replicates.iter().zip(&receipt.replicate_standard_errors).zip(&receipt.pivots)
    {
        let point = record.point.unwrap();
        // For scores theta +/- .2, unbiased resample variance uses the actual
        // selected sign frequency, not the original variance or bootstrap SE.
        let expected_se = ((0.04 - (point - truth).powi(2)) / 29.0).sqrt();
        assert!((se.unwrap() - expected_se).abs() < 1e-12);
        let expected_pivot = (point - truth) / expected_se;
        assert!((pivot.unwrap() - expected_pivot).abs() < 1e-11);
        pivots.push(expected_pivot);
    }
    pivots.sort_by(f64::total_cmp);
    let interpolate = |probability: f64| {
        let h = probability * (pivots.len() - 1) as f64;
        let index = h.floor() as usize;
        pivots[index]
            + (h - index as f64) * (pivots[(index + 1).min(pivots.len() - 1)] - pivots[index])
    };
    assert!((result.lower - (truth - interpolate(0.975) * receipt.standard_error)).abs() < 1e-12);
    assert!((result.upper - (truth - interpolate(0.025) * receipt.standard_error)).abs() < 1e-12);
}

#[test]
fn studentized_refuses_unbalanced_units_and_uncertified_estimator() {
    struct Arbitrary(std::cell::Cell<usize>);
    impl antecedent_estimate::temporal_dependent_interval::TemporalEstimator for Arbitrary {
        fn label(&self) -> &'static str {
            "arbitrary"
        }
        fn estimate(&self, units: &[&UnitHistories]) -> Result<f64, EstimationError> {
            self.0.set(self.0.get() + 1);
            Ok(units.iter().map(|u| u.histories[0].y).sum::<f64>() / units.len() as f64)
        }
    }
    let p = panel(30);
    let mut units = p.units().to_vec();
    units[0].histories.pop();
    let changed = TemporalUnitPanel::new("unbalanced", Some(units)).unwrap();
    let config = DependentIntervalConfig { method: IntervalMethod::Studentized, ..config(91) };
    assert!(
        dependent_unit_interval(&changed, &antecedent_estimate::temporal_dependent_interval::BalancedTemporalEstimator::Marginalized(target_query()), &config, &ctx())
            .unwrap_err()
            .to_string()
            .contains("studentized_unbalanced")
    );
    let arbitrary = Arbitrary(std::cell::Cell::new(0));
    assert!(
        dependent_unit_interval(&p, &arbitrary, &config, &ctx())
            .unwrap_err()
            .to_string()
            .contains("studentized_estimator_not_certified")
    );
    assert_eq!(arbitrary.0.get(), 1, "uncertified callback must refuse before any resample");
}

#[test]
fn studentized_refuses_zero_original_unit_variance() {
    let original = panel(30);
    let mut units = original.units().to_vec();
    for unit in &mut units {
        for history in &mut unit.histories {
            history.y = 1.0;
        }
    }
    let constant = TemporalUnitPanel::new("constant_unit_scores", Some(units)).unwrap();
    let estimator =
        antecedent_estimate::temporal_dependent_interval::BalancedTemporalEstimator::Marginalized(
            target_query(),
        );
    let config = DependentIntervalConfig { method: IntervalMethod::Studentized, ..config(91) };
    assert!(
        dependent_unit_interval(&constant, &estimator, &config, &ctx())
            .unwrap_err()
            .to_string()
            .contains("studentized_zero_variance")
    );
}

#[test]
fn direct_intervals_reject_retired_methods_before_estimation() {
    for method in [IntervalMethod::Percentile, IntervalMethod::Basic] {
        let error = dependent_unit_interval(
            &panel(30),
            &target_query(),
            &DependentIntervalConfig { method, ..config(1) },
            &ctx(),
        )
        .unwrap_err();
        assert!(error.to_string().contains("temporal_interval.retired_method"));
    }
}
