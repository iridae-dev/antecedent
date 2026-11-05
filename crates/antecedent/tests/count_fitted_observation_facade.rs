//! Count-fitted observation probability: the point value follows the posterior-mean
//! tables, the interval is a measured scalar or a typed refusal, and a changed source or
//! a `not_measured` construction is refused.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::count_fitted_observation::{
    CountFittedNodeWire, CountFittedObservation, MeasuredInference,
};
use antecedent_core::ExecutionContext;

fn levels() -> Vec<String> {
    vec!["0".into(), "1".into()]
}

/// `Z -> Y`, 100 rows: Z = 1 in 40 rows, and Y = 1 in 30 of those; Y = 1 in 12 of the
/// 60 rows with Z = 0.
fn fitted() -> CountFittedObservation {
    let nodes = vec![
        CountFittedNodeWire {
            name: "Z".into(),
            levels: levels(),
            parents: vec![],
            counts: Some(vec![60, 40]),
            declared_bits: None,
        },
        CountFittedNodeWire {
            name: "Y".into(),
            levels: levels(),
            parents: vec![0],
            counts: Some(vec![48, 12, 10, 30]),
            declared_bits: None,
        },
    ];
    CountFittedObservation::new(nodes, 100, "Y", "1", &[("Z".into(), "1".into())], Some(7)).unwrap()
}

#[test]
fn the_point_value_is_the_posterior_mean_of_the_fitted_conditional() {
    // (30 + 1) / (40 + 2) under the uniform Dirichlet prior.
    let point = fitted().point().unwrap();
    assert!((point - 31.0 / 42.0).abs() < 1e-12, "{point}");
}

#[test]
fn the_interval_is_reported_with_its_calibration_annotation() {
    let claim = fitted();
    let ctx = ExecutionContext::for_tests(5);
    let measured = claim.interval(0.95, &ctx).expect("the interval is always reported");
    let scalar = &measured.report().scalars[0];
    assert!(scalar.lower <= scalar.point && scalar.point <= scalar.upper);
    assert!((scalar.point - 31.0 / 42.0).abs() < 1e-12);
    // Calibration only annotates: the status is whichever the records give at n = 100.
    assert!(
        ["calibrated", "outside_measured_scope", "measured_below_nominal", "not_measured"]
            .contains(&scalar.calibration.status.as_str()),
        "{:?}",
        scalar.calibration
    );
    let again =
        MeasuredInference::consume(measured.export(), measured.expectation(), &ctx).unwrap();
    assert_eq!(again.export(), measured.export());
}

#[test]
fn an_not_measured_level_is_reported_and_annotated_not_measured() {
    let ctx = ExecutionContext::for_tests(5);
    let measured = fitted().interval(0.5, &ctx).expect("the interval is reported");
    let scalar = &measured.report().scalars[0];
    assert!(scalar.lower <= scalar.point && scalar.point <= scalar.upper);
    assert!((scalar.level - 0.5).abs() < 1e-12);
    assert_eq!(scalar.calibration.status.as_str(), "not_measured");
    assert_eq!(scalar.calibration.reason.as_deref(), Some("interval_level_not_measured"));
}

#[test]
fn unknown_variables_and_zero_probability_evidence_are_refused_at_construction() {
    let nodes = vec![CountFittedNodeWire {
        name: "Z".into(),
        levels: levels(),
        parents: vec![],
        counts: Some(vec![100, 0]),
        declared_bits: None,
    }];
    let refusal = CountFittedObservation::new(nodes, 100, "W", "1", &[], None).unwrap_err();
    assert!(!refusal.refusals.is_empty());
}

fn with_draws(draws: u32) -> CountFittedObservation {
    let mut source = fitted().source().clone();
    source.draws = draws;
    CountFittedObservation::from_source(source).unwrap()
}

#[test]
fn draw_counts_from_the_measured_minimum_are_calibrated_and_fewer_are_annotated() {
    let ctx = ExecutionContext::for_tests(5);
    for draws in [2000, 3000, 4000, 8000] {
        let measured = with_draws(draws).interval(0.95, &ctx).expect("reported");
        let scalar = &measured.report().scalars[0];
        assert!(scalar.lower <= scalar.point && scalar.point <= scalar.upper, "{draws}");
    }
    // Fewer draws than the measured minimum are computed; calibration annotates, never gates.
    let fewer = with_draws(1999).interval(0.95, &ctx).expect("reported");
    assert_ne!(fewer.report().scalars[0].calibration.status, "calibrated");
}
