//! The winner's-curse bound against 20-digit quadrature of the expected maximum of independent
//! Gaussians (`oracle_data/winners_curse_reference.py`).
//!
//! Mixed standard errors (a spread of five orders of magnitude), unseen candidates padded at the
//! largest supplied standard error up to ten million considered candidates, candidates with a
//! zero standard error, and the dependence-free union bound. The bound reported is the exact
//! maximum only for declared independent streams.

use antecedent_design::decision_reliability::{
    BoundKind, McEstimate, NoiseDependence, StandardErrorMethod, winners_curse_bound,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/winners_curse_reference.json"))
        .expect("frozen reference")
}

fn estimates(ses: &Value) -> Vec<McEstimate> {
    ses.as_array()
        .unwrap()
        .iter()
        .map(|s| McEstimate {
            value: 0.0,
            standard_error: s.as_f64().unwrap(),
            effective_sample_size: 100.0,
            n_draws: 100,
            method: StandardErrorMethod::IidSampleVariance,
        })
        .collect()
}

#[test]
fn the_exact_maximum_and_the_union_bound_equal_the_quadrature_reference() {
    let mut failures = Vec::new();
    for case in reference().as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let n = case["n_considered"].as_u64().unwrap();
        let est = estimates(&case["ses"]);
        let bound = winners_curse_bound(&est, n, NoiseDependence::IndependentStreams, 1.5).unwrap();
        let gaussian = case["gaussian"].as_f64().unwrap();
        let union = case["union"].as_f64().unwrap();
        let got = bound.gaussian_expected_maximum.expect("within the group cap");
        if (got - gaussian).abs() > 2e-8 * (1.0 + gaussian) {
            failures.push(format!("{name}: expected maximum {got} vs {gaussian}"));
        }
        assert!(
            (bound.union_bound - union).abs() <= 1e-12 * (1.0 + union),
            "{name}: union {} vs {union}",
            bound.union_bound
        );
        assert_eq!(bound.padded_candidates, n - est.len() as u64, "{name}");
        // The reported bound is the exact one for independent streams (it never exceeds the
        // union bound), and the adjustment is the in-sample value less the bound.
        assert_eq!(bound.bound_kind, BoundKind::GaussianExpectedMaximum, "{name}");
        assert!((bound.adjusted_value - (1.5 - bound.bound)).abs() < 1e-15);
        // Every other dependence declaration reports the dependence-free bound.
        for dependence in
            [NoiseDependence::SampleCovariancesNonNegative, NoiseDependence::Unassessed]
        {
            let other = winners_curse_bound(&est, n, dependence, 1.5).unwrap();
            assert_eq!(other.bound_kind, BoundKind::UnionBound, "{name}");
            assert_eq!(other.bound, other.union_bound, "{name}");
        }
    }
    assert!(failures.is_empty(), "{failures:#?}");
}

#[test]
fn malformed_inputs_are_refused() {
    let est = estimates(&serde_json::json!([1.0, 2.0]));
    assert!(winners_curse_bound(&est, 1, NoiseDependence::IndependentStreams, 0.0).is_err());
    assert!(winners_curse_bound(&[], 3, NoiseDependence::IndependentStreams, 0.0).is_err());
    let negative = estimates(&serde_json::json!([1.0, -0.5]));
    assert!(winners_curse_bound(&negative, 2, NoiseDependence::IndependentStreams, 0.0).is_err());
    assert!(winners_curse_bound(&est, 2, NoiseDependence::IndependentStreams, f64::NAN).is_err());
}
