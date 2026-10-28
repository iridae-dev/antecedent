//! Stagewise-ordered tail probabilities and whole-procedure inversion against exact rational
//! enumeration (`oracle_data/procedure_ordered_reference.py`).
//!
//! The reference path-enumerates the law of fixed-randomization Wald-boundary procedures (two-sided
//! and upper boundaries, pair and single-unit looks, so that arms without observations give
//! undefined estimates), ranks every terminal state by the stagewise ordering written from its
//! definition (plus-arm rule stops first and earlier higher, minus-arm rule stops last and later
//! higher, the rest by the estimate), and takes the tail masses at every defined terminal state of
//! three truths. The inversion accepts a contrast on a dyadic grid when the supremum over the
//! baselines of each exact tail is at least alpha / 2; the accepted set and the number of laws
//! evaluated are compared exactly.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "support/adaptive_procedure_reference.rs"]
mod support;

use antecedent_stats::adaptive_inference::{
    InversionGrid, ObservedTerminal, enumerate_law, inversion_interval, ordering_p_values,
};
use serde_json::Value;
use support::{frac, fracs, kernel, never, spec};

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/procedure_ordered_reference.json"))
        .expect("frozen reference")
}

fn observed(value: &Value) -> ObservedTerminal {
    ObservedTerminal {
        look: value["look"].as_u64().unwrap() as usize,
        rule_stopped: value["stopped"].as_bool().unwrap(),
        decision: value["decision"].as_u64().map(|d| d as usize),
        estimate: frac(&value["estimate"]),
    }
}

#[test]
fn tail_probabilities_match_the_exact_stagewise_ordering() {
    let reference = reference();
    let mut compared = 0;
    for case in reference["cases"].as_array().expect("cases") {
        let name = case["name"].as_str().unwrap();
        let spec = spec(&case["declaration"]);
        let truth = serde_json::json!({"kind": "fixed", "values": case["truth"]});
        let law = enumerate_law(&spec, &kernel(&truth), &never).expect("a law");
        for state in case["observed"].as_array().unwrap() {
            let p = ordering_p_values(&law, 1, 0, &observed(state)).expect("p-values");
            let label = format!("{name} {state}");
            assert!(
                (p.upper - state["upper"].as_f64().unwrap()).abs() <= 1e-12,
                "{label}: upper {}",
                p.upper
            );
            assert!(
                (p.lower - state["lower"].as_f64().unwrap()).abs() <= 1e-12,
                "{label}: lower {}",
                p.lower
            );
            assert!(
                (p.undefined_mass - state["undefined"].as_f64().unwrap()).abs() <= 1e-12,
                "{label}"
            );
            compared += 1;
        }
    }
    assert!(compared > 300);
}

#[test]
fn the_inversion_accepts_exactly_the_contrasts_the_exact_tails_allow() {
    let reference = reference();
    let spec = spec(&reference["declaration"]);
    for inversion in reference["inversions"].as_array().expect("inversions") {
        let alpha = frac(&inversion["alpha"]);
        let grid = InversionGrid {
            contrasts: fracs(&inversion["contrasts"]),
            baselines: fracs(&inversion["baselines"]),
        };
        let result =
            inversion_interval(&spec, &observed(&inversion["observed"]), &grid, alpha, &never)
                .expect("an interval");
        let want = fracs(&inversion["accepted"]);
        assert_eq!(result.accepted, want, "accepted set at alpha {alpha}");
        assert_eq!(result.laws_evaluated, inversion["laws_evaluated"].as_u64().unwrap() as usize);
        assert_eq!(result.lower, want.first().copied());
        assert_eq!(result.upper, want.last().copied());
        assert_eq!(result.calibration, antecedent_stats::adaptive_inference::DERIVED_UNMEASURED);
    }
}
