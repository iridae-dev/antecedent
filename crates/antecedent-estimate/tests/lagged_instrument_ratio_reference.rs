//! The lagged-instrument Wald contrast and its reduced-form ratio posterior against an exact
//! reference.
//!
//! `oracle_data/lagged_instrument_ratio_reference.py` (frozen in the JSON beside it) aligns the lagged
//! instrument, forms the Wald effect, first-stage F and homoskedastic standard error, builds the
//! posterior precision `kron(S^-1, X'X) + I / tau^2` of the two reduced forms from the plug-in
//! residual covariance (a different construction from the row-by-row accumulation in the module)
//! and evaluates the exact distribution function of the ratio `gamma / pi` of two jointly Normal
//! coefficients by quadrature. The sorted ratio draws must reproduce that distribution function.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::lagged_instrument::{
    LaggedInstrumentSample, fit_lagged_instrument, fit_lagged_instrument_posterior,
};
use antecedent_estimate::util::DrawSpec;
use serde_json::Value;

const DRAWS: usize = 400_000;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!(
        "oracle_data/lagged_instrument_ratio_reference.json"
    ))
    .unwrap()
    .as_array()
    .unwrap()
    .clone()
}

fn relative(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= 1e-9 * expected.abs().max(1e-12),
        "{what}: {actual} vs {expected}"
    );
}

#[test]
fn the_wald_contrast_matches_the_exact_reference() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let (z, t, y) =
            (floats(&case["instrument"]), floats(&case["treatment"]), floats(&case["outcome"]));
        let sample = LaggedInstrumentSample {
            instrument: &z,
            treatment: &t,
            outcome: &y,
            lag: case["lag"].as_u64().unwrap() as usize,
        };
        let wald = fit_lagged_instrument(&sample, 0.0).unwrap();
        assert_eq!(wald.aligned_rows as u64, case["aligned_rows"].as_u64().unwrap(), "{name}");
        relative(wald.effect, case["effect"].as_f64().unwrap(), "effect");
        relative(wald.intercept, case["intercept"].as_f64().unwrap(), "intercept");
        relative(wald.first_stage_slope, case["first_stage_slope"].as_f64().unwrap(), "slope");
        relative(wald.first_stage_f, case["first_stage_f"].as_f64().unwrap(), "F");
        relative(wald.homoskedastic_se, case["homoskedastic_se"].as_f64().unwrap(), "se");
    }
}

#[test]
fn the_ratio_draws_follow_the_exact_distribution_function_of_the_gaussian_ratio() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let (z, t, y) =
            (floats(&case["instrument"]), floats(&case["treatment"]), floats(&case["outcome"]));
        let sample = LaggedInstrumentSample {
            instrument: &z,
            treatment: &t,
            outcome: &y,
            lag: case["lag"].as_u64().unwrap() as usize,
        };
        let fit = fit_lagged_instrument_posterior(
            &sample,
            case["prior_sd"].as_f64().unwrap(),
            DrawSpec::new(DRAWS, 3),
            0.0,
        )
        .unwrap();
        assert_eq!(fit.effect_draws.len(), DRAWS);
        for point in case["cdf"].as_array().unwrap() {
            let (x, p) = (point["x"].as_f64().unwrap(), point["p"].as_f64().unwrap());
            let below = fit.effect_draws.partition_point(|v| *v <= x) as f64 / DRAWS as f64;
            let tolerance = 5.0 * (p * (1.0 - p) / DRAWS as f64).sqrt() + 1e-4;
            assert!((below - p).abs() < tolerance, "{name} P(ratio <= {x}) = {below} vs {p}");
        }
    }
}

#[test]
fn a_zero_lag_and_a_short_series_are_refused() {
    let case = &cases()[0];
    let (z, t, y) =
        (floats(&case["instrument"]), floats(&case["treatment"]), floats(&case["outcome"]));
    for lag in [0, z.len() - 3] {
        let sample = LaggedInstrumentSample { instrument: &z, treatment: &t, outcome: &y, lag };
        assert!(fit_lagged_instrument(&sample, 0.0).is_err(), "lag {lag}");
    }
}
