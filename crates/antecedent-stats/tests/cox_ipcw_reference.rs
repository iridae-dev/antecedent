//! The Cox/Breslow fit and the Cox inverse-censoring weights against a 40-digit reference.
//!
//! `oracle_data/cox_ipcw_reference.py` (frozen in the JSON beside it) maximises the Breslow
//! partial likelihood of 64 rows with 50 tied times (a quarter-unit time grid), a continuous and a
//! binary covariate, by Newton's method with the exact score and information; forms the Breslow
//! cumulative baseline at the covariate means, its left and right limits, and the conditional
//! censoring survival just before each time; and weighs observed rows by its inverse.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{cox_breslow, cox_ipcw};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("oracle_data/cox_ipcw_reference.json")).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn texts(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().parse().unwrap()).collect()
}

fn text(v: &Value) -> f64 {
    v.as_str().unwrap().parse().unwrap()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    let tol = rel * want.abs().max(1e-300);
    assert!((got - want).abs() <= tol, "{what}: got {got} want {want} tol {tol}");
}

fn column_major(fx: &Value) -> Vec<f64> {
    let rows: Vec<Vec<f64>> = fx["covariates"].as_array().unwrap().iter().map(floats).collect();
    (0..2).flat_map(|j| rows.iter().map(move |r| r[j])).collect()
}

#[test]
fn breslow_coefficients_baseline_and_limits_match_the_multiprecision_fit() {
    let fx = fixture();
    assert!(fx["ties"].as_u64().unwrap() >= 30, "the design must carry tied times");
    let failure: Vec<bool> =
        fx["failure"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect();
    let fit = cox_breslow(&floats(&fx["time"]), &failure, &column_major(&fx), 2).unwrap();
    let want = &fx["breslow"];
    for (got, w) in fit.coefficients.iter().zip(texts(&want["coefficients"])) {
        close(*got, w, 1e-7, "coefficient");
    }
    close(fit.center, text(&want["center"]), 1e-7, "center");
    assert_eq!(fit.jump_times, floats(&want["jump_times"]));
    for (got, w) in fit.cumulative_baseline.iter().zip(texts(&want["cumulative_baseline"])) {
        close(*got, w, 1e-7, "cumulative baseline");
    }
    for row in want["baseline_before"].as_array().unwrap() {
        close(fit.baseline_before(row["t"].as_f64().unwrap()), text(&row["value"]), 1e-7, "before");
    }
    for row in want["baseline_at"].as_array().unwrap() {
        close(fit.baseline_at(row["t"].as_f64().unwrap()), text(&row["value"]), 1e-7, "at");
    }
    // The linear predictor is the raw-scale predictor minus the centre.
    let eta = fit.linear_predictor(&[1.0, 1.0]);
    close(eta, fit.coefficients[0] + fit.coefficients[1] - fit.center, 1e-14, "predictor");
}

#[test]
fn inverse_censoring_weights_match_the_multiprecision_conditional_survival() {
    let fx = fixture();
    let time = floats(&fx["time"]);
    let event = floats(&fx["event"]);
    let fit = cox_ipcw(&time, &event, &column_major(&fx), 2, 0.01).unwrap();
    let want = &fx["ipcw"];
    for (got, w) in fit.coefficients.iter().zip(texts(&want["coefficients"])) {
        close(*got, w, 1e-6, "censoring coefficient");
    }
    for (i, (got, w)) in fit.survival_before.iter().zip(texts(&want["survival_before"])).enumerate()
    {
        close(*got, w, 1e-6, &format!("survival {i}"));
    }
    for (i, (got, w)) in fit.weights.iter().zip(texts(&want["weights"])).enumerate() {
        if w == 0.0 {
            assert_eq!(*got, 0.0, "censored row {i}");
        } else {
            close(*got, w, 1e-6, &format!("weight {i}"));
        }
    }
    // A floor above the smallest observed-row survival refuses instead of clamping.
    let smallest = fit
        .survival_before
        .iter()
        .zip(&event)
        .filter(|(_, e)| **e == 1.0)
        .map(|(s, _)| *s)
        .fold(f64::INFINITY, f64::min);
    assert!(format!("{:?}", cox_ipcw(&time, &event, &column_major(&fx), 2, (smallest * 1.01).min(0.99)).err()).contains("below the configured positivity floor"), "below the configured positivity floor");
}
