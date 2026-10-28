//! The lagged-instrument Wald contrast against a 40-digit reference.
//!
//! `oracle_data/lagged_instrument_wald_reference.py` (frozen in the JSON beside it) aligns the
//! instrument `lag` steps back, forms the Wald ratio, the first stage, the structural intercept,
//! the reduced-form outcome slope, the homoskedastic standard error and the first-stage F. A weak
//! lag (F near its threshold) and a shifted alignment are included so the alignment and the
//! threshold rule are exercised, not only an easy instrument.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::lagged_instrument::{LaggedInstrumentSample, fit_lagged_instrument};
use serde_json::Value;

fn fixture() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/lagged_instrument_wald_reference.json"))
        .unwrap()
        .as_array()
        .cloned()
        .unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn text(v: &Value) -> f64 {
    v.as_str().unwrap().parse().unwrap()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    let tol = rel * want.abs().max(1e-300);
    assert!((got - want).abs() <= tol, "{what}: got {got} want {want} tol {tol}");
}

#[test]
fn wald_effect_first_stage_standard_error_and_f_match_the_multiprecision_fit() {
    for case in fixture() {
        let name = case["name"].as_str().unwrap();
        let (z, t, y) =
            (floats(&case["instrument"]), floats(&case["treatment"]), floats(&case["outcome"]));
        let lag = case["lag"].as_u64().unwrap() as usize;
        let fit = fit_lagged_instrument(
            &LaggedInstrumentSample { instrument: &z, treatment: &t, outcome: &y, lag },
            0.0,
        )
        .unwrap();
        close(fit.effect, text(&case["effect"]), 1e-10, name);
        close(fit.intercept, text(&case["intercept"]), 1e-9, name);
        close(fit.first_stage_slope, text(&case["first_stage_slope"]), 1e-10, name);
        close(fit.first_stage_intercept, text(&case["first_stage_intercept"]), 1e-9, name);
        close(fit.reduced_form_outcome_slope, text(&case["reduced_form_slope"]), 1e-10, name);
        close(fit.homoskedastic_se, text(&case["se"]), 1e-10, name);
        close(fit.first_stage_f, text(&case["first_stage_f"]), 1e-9, name);
        assert_eq!(fit.aligned_rows, case["aligned_rows"].as_u64().unwrap() as usize, "{name}");
        assert_eq!(fit.lag, lag);
        close(fit.effect * fit.first_stage_slope, fit.reduced_form_outcome_slope, 1e-12, name);
    }
}

#[test]
fn the_weak_instrument_threshold_refuses_a_finite_f_below_it_and_accepts_one_at_it() {
    for case in fixture() {
        let name = case["name"].as_str().unwrap();
        let (z, t, y) =
            (floats(&case["instrument"]), floats(&case["treatment"]), floats(&case["outcome"]));
        let lag = case["lag"].as_u64().unwrap() as usize;
        let f = text(&case["first_stage_f"]);
        let sample = LaggedInstrumentSample { instrument: &z, treatment: &t, outcome: &y, lag };
        assert!(fit_lagged_instrument(&sample, f * (1.0 - 1e-9)).is_ok(), "{name} below F");
        let error = fit_lagged_instrument(&sample, f * (1.0 + 1e-9)).unwrap_err().to_string();
        assert!(error.contains("weak_instrument"), "{name}: {error}");
    }
}
