//! The mixture-Hoeffding and predictable plug-in empirical Bernstein confidence sequences against
//! 80-digit evaluation of their formulas (`oracle_data/confidence_sequence_reference.py`).
//!
//! Covers fixed streams (constant, alternating, a jump that violates the common-mean premise,
//! participant proportions, near-zero variance, rare events) at three levels, radii out to fifty
//! million observations, and the exact miss probability over every Bernoulli path, which Ville's
//! inequality bounds by the level.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::adaptive_inference::{SequenceKind, SequenceState, mixture_hoeffding_radius};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/confidence_sequence_reference.json"))
        .expect("frozen reference")
}

fn kind_of(label: &str) -> SequenceKind {
    match label {
        "plug_in" => SequenceKind::PredictablePlugInBernstein,
        _ => SequenceKind::MixtureHoeffding,
    }
}

fn number(value: &Value) -> f64 {
    match value {
        Value::String(text) => text.parse().expect("number"),
        other => other.as_f64().expect("number"),
    }
}

#[test]
fn fixed_streams_match_the_reference_at_every_checkpoint() {
    let reference = reference();
    let mut compared = 0;
    for case in reference["streams"].as_array().expect("streams") {
        let kind = kind_of(case["kind"].as_str().expect("kind"));
        let mut state = SequenceState::of_kind(
            kind,
            case["alpha"].as_f64().expect("alpha"),
            case["tuning"].as_f64().expect("tuning"),
        )
        .expect("sequence");
        let values: Vec<f64> =
            case["values"].as_array().expect("values").iter().map(number).collect();
        let readings = case["readings"].as_array().expect("readings");
        let mut next = 0;
        for (index, &x) in values.iter().enumerate() {
            state.push(x).expect("outcome");
            let Some(want) = readings.get(next) else { break };
            if want["count"].as_u64() != Some(index as u64 + 1) {
                continue;
            }
            next += 1;
            let label = format!(
                "{} {} alpha {} n {}",
                case["name"],
                case["kind"],
                case["alpha"],
                index + 1
            );
            if want["empty"].as_bool() == Some(true) {
                assert!(state.reading().is_err(), "an empty intersection is an error: {label}");
                continue;
            }
            let got = state.reading().expect("reading");
            let (lower, upper) = (number(&want["lower"]), number(&want["upper"]));
            assert!(
                (got.lower - lower).abs() <= 1e-13 + 1e-12 * lower.abs(),
                "{label}: lower {} vs {lower}",
                got.lower
            );
            assert!(
                (got.upper - upper).abs() <= 1e-13 + 1e-12 * upper.abs(),
                "{label}: upper {} vs {upper}",
                got.upper
            );
            compared += 1;
        }
        assert_eq!(next, readings.len(), "every checkpoint reached");
    }
    assert!(compared > 300, "{compared}");
}

#[test]
fn mixture_radius_matches_the_reference_to_fifty_million_observations() {
    let reference = reference();
    for case in reference["radius"].as_array().expect("radius") {
        let got = mixture_hoeffding_radius(
            case["count"].as_u64().expect("count"),
            case["alpha"].as_f64().expect("alpha"),
            case["tuning"].as_f64().expect("tuning"),
        )
        .expect("radius");
        let want = number(&case["radius"]);
        assert!((got - want).abs() <= 1e-13 * want, "{case}: {got} vs {want}");
    }
}

/// Probability that the running intersection ever excludes `p` (or empties) over every
/// Bernoulli(`p`) path of `depth` more outcomes.
fn miss_probability(state: SequenceState, p: f64, depth: usize) -> f64 {
    if depth == 0 {
        return 0.0;
    }
    let mut missed = 0.0;
    for (x, weight) in [(0.0, 1.0 - p), (1.0, p)] {
        let mut next = state;
        next.push(x).expect("outcome");
        let outside = match next.reading() {
            Ok(r) => !(r.lower <= p && p <= r.upper),
            Err(_) => true,
        };
        missed += weight * if outside { 1.0 } else { miss_probability(next, p, depth - 1) };
    }
    missed
}

#[test]
fn exact_miss_probability_equals_the_reference_and_respects_the_level() {
    let reference = reference();
    for case in reference["miss"].as_array().expect("miss") {
        let alpha = case["alpha"].as_f64().expect("alpha");
        let p = case["p"].as_f64().expect("p");
        let state = SequenceState::of_kind(
            kind_of(case["kind"].as_str().expect("kind")),
            alpha,
            case["tuning"].as_f64().expect("tuning"),
        )
        .expect("sequence");
        let got = miss_probability(state, p, case["n"].as_u64().expect("n") as usize);
        let want = number(&case["probability"]);
        assert!((got - want).abs() <= 1e-12, "{case}: {got} vs {want}");
        assert!(got <= alpha, "Ville's inequality bounds the miss probability: {case}");
    }
}
