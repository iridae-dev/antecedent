//! Time-uniform cell bounds at a stopping time and the exact whole-procedure exclusion
//! probability, against a 60-digit evaluation of the mixture radius and exact rational path
//! enumeration (`oracle_data/procedure_anytime_reference.py`).
//!
//! The reference sums the exact probabilities of the terminal states at which some observed cell's
//! bound `[max(m - r, 0), min(m + r, 1)]`, at level `alpha / cells`, misses that cell's truth; the
//! procedures cover Thompson sampling with a posterior stop, a three-arm UCB run under a resource
//! cap, lagged epsilon-greedy with random loss, exposure failure, and a truth on the boundary of
//! the support. The radius itself is reviewed under the confidence-sequence record; this file checks
//! the route's own arithmetic (the level split over cells, unobserved cells, clipping to the unit
//! interval, which cells are tested against which truth).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "support/adaptive_procedure_reference.rs"]
mod support;

use antecedent_stats::adaptive_inference::{
    AdaptiveInferenceError, DERIVED_UNMEASURED, anytime_exclusion_probability, anytime_valid_report,
};
use serde_json::Value;
use support::{fracs, never, spec};

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/procedure_anytime_reference.json"))
        .expect("frozen reference")
}

fn level(value: &Value) -> f64 {
    value.as_str().map_or_else(|| value.as_f64().unwrap(), |t| t.parse().unwrap())
}

#[test]
fn exclusion_probabilities_match_the_exact_enumeration() {
    let reference = reference();
    for case in reference["exclusions"].as_array().expect("cases") {
        let name = case["name"].as_str().unwrap();
        let spec = spec(&case["declaration"]);
        let got = anytime_exclusion_probability(
            &spec,
            &fracs(&case["truth"]),
            level(&case["alpha"]),
            level(&case["tuning"]),
            &never,
        )
        .expect("a probability");
        let want = case["exclusion_probability"].as_f64().unwrap();
        assert!((got - want).abs() <= 1e-12, "{name}: {got} vs {want}");
    }
}

#[test]
fn reported_bounds_match_the_reference_radius() {
    let reference = reference();
    let declaration = serde_json::json!({
        "arms": 2,
        "assignment": {"kind": "fixed", "probabilities": ["1/2", "1/2"]},
        "enrolment": [2, 2],
        "stopping": {"kind": "none"},
        "selection": {"kind": "none"}
    });
    let spec = spec(&declaration);
    for row in reference["reports"].as_array().expect("reports") {
        let counts: Vec<u16> =
            row["counts"].as_array().unwrap().iter().map(|n| n.as_u64().unwrap() as u16).collect();
        let alpha = level(&row["alpha"]);
        let report = anytime_valid_report(&spec, &counts, alpha, level(&row["tuning"])).unwrap();
        assert_eq!(report.calibration, DERIVED_UNMEASURED);
        assert!((report.alpha_per_cell - row["alpha_per_cell"].as_f64().unwrap()).abs() < 1e-15);
        for (cell, want) in report.bounds.iter().zip(row["cells"].as_array().unwrap()) {
            assert_eq!(u64::from(cell.units), want["units"].as_u64().unwrap());
            assert_eq!(cell.mean, want["mean"].as_f64());
            assert!(
                (cell.lower - want["lower"].as_f64().unwrap()).abs() <= 1e-12,
                "{counts:?} lower"
            );
            assert!(
                (cell.upper - want["upper"].as_f64().unwrap()).abs() <= 1e-12,
                "{counts:?} upper"
            );
        }
    }
    // A level outside (0, 1), a count vector of the wrong length and a truth of the wrong shape
    // are refused, not defaulted.
    assert!(anytime_valid_report(&spec, &[1, 1, 1, 1], 1.0, 10.0).is_err());
    assert!(anytime_valid_report(&spec, &[1, 1, 1, 1], 0.05, 0.0).is_err());
    assert!(anytime_valid_report(&spec, &[1, 1, 1], 0.05, 10.0).is_err());
    assert!(matches!(
        anytime_exclusion_probability(&spec, &[0.5], 0.05, 10.0, &never),
        Err(AdaptiveInferenceError::Shape(_))
    ));
}
