//! Count-fitted observation probability: the point value follows the posterior-mean
//! tables, the interval is a measured scalar or a typed refusal, and a changed source or
//! an unmeasured construction is refused.
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
fn the_interval_is_bound_to_its_record_or_refused_for_want_of_one() {
    let claim = fitted();
    let ctx = ExecutionContext::for_tests(5);
    match claim.interval(0.95, &ctx) {
        Ok(measured) => {
            let scalar = &measured.report().scalars[0];
            assert_eq!(scalar.calibration.status, "calibrated");
            assert!(scalar.lower <= scalar.point && scalar.point <= scalar.upper);
            assert!((scalar.point - 31.0 / 42.0).abs() < 1e-12);
            let again = MeasuredInference::consume(measured.export(), measured.expectation(), &ctx)
                .unwrap();
            assert_eq!(again.export(), measured.export());
        }
        Err(refusal) => {
            assert_eq!(refusal.code(), "cell_not_licensed");
            assert!(
                refusal.details()[0].contains("record_not_attesting")
                    || refusal.details()[0].contains("protocol_not_measured"),
                "{refusal}"
            );
        }
    }
}

#[test]
fn an_unmeasured_level_or_a_two_level_mismatch_refuses_cell_not_licensed() {
    let ctx = ExecutionContext::for_tests(5);
    let refusal = fitted().interval(0.5, &ctx).unwrap_err();
    assert_eq!(refusal.code(), "cell_not_licensed");
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
fn draw_counts_from_the_supported_minimum_are_supported_and_fewer_are_refused() {
    let ctx = ExecutionContext::for_tests(5);
    for draws in [2000, 3000, 4000, 8000] {
        match with_draws(draws).interval(0.95, &ctx) {
            Ok(measured) => {
                let scalar = &measured.report().scalars[0];
                assert_eq!(scalar.calibration.status, "calibrated", "{draws}");
                assert!(scalar.lower <= scalar.point && scalar.point <= scalar.upper);
            }
            Err(refusal) => {
                // Only the absence of a currently attesting record may stand in the way.
                assert_eq!(refusal.code(), "cell_not_licensed");
                let detail = &refusal.details()[0];
                assert!(
                    !detail.contains("protocol_not_measured")
                        && !detail.contains("draws_not_converged"),
                    "{draws}: {detail}"
                );
            }
        }
    }
    let refusal = with_draws(1999).interval(0.95, &ctx).unwrap_err();
    assert_eq!(refusal.code(), "cell_not_licensed");
    assert!(refusal.details()[0].contains("protocol_not_measured"), "{refusal}");
}
