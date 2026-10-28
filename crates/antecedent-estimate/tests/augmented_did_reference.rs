//! Augmented two-period panel ATT against an exact rational reference.
//!
//! `oracle_data/augmented_did_reference.py` (frozen in the JSON beside it) forms the treated
//! mean of the untreated-change residual minus the odds-weighted comparison mean with the
//! weights normalised to sum to one, in `Fraction` arithmetic. Cases cover moderate and
//! extreme propensity odds, an outcome offset of 1e8, a prediction wrong by a constant of 5
//! (which must cancel) and four subjects.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::augmented_panel_did::estimate;
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-300), "{what}: got {got} want {want}");
}

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/augmented_did_reference.json")).unwrap()
        ["cases"]
        .as_array()
        .unwrap()
        .clone()
}

#[test]
fn the_augmented_effect_and_comparison_size_match_the_exact_reference() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let treated: Vec<bool> =
            case["treated"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect();
        let fit = estimate(
            &floats(&case["pre"]),
            &floats(&case["post"]),
            &treated,
            &floats(&case["p"]),
            &floats(&case["m"]),
        )
        .unwrap();
        // The 1e8 offset costs the changes about 1e-8 absolute; the effect is of order 2.
        close(fit.effect, case["effect"].as_f64().unwrap(), 2e-7, &format!("{name} effect"));
        close(
            fit.effective_control_sample_size,
            case["ess"].as_f64().unwrap(),
            1e-12,
            &format!("{name} comparison size"),
        );
    }
}

#[test]
fn a_constant_error_in_the_supplied_prediction_cancels() {
    let case = cases().into_iter().find(|c| c["name"] == "biased_prediction").unwrap();
    let treated: Vec<bool> =
        case["treated"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect();
    let (pre, post, p) = (floats(&case["pre"]), floats(&case["post"]), floats(&case["p"]));
    let m = floats(&case["m"]);
    let shifted: Vec<f64> = m.iter().map(|v| v + 1000.0).collect();
    let a = estimate(&pre, &post, &treated, &p, &m).unwrap().effect;
    let b = estimate(&pre, &post, &treated, &p, &shifted).unwrap().effect;
    assert!((a - b).abs() < 1e-9, "{a} {b}");
}
