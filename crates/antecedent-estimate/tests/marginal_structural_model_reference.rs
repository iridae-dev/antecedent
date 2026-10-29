//! Additive marginal structural model against an exact-rational weighted least-squares reference.
//!
//! `oracle_data/marginal_structural_model_reference.py` (frozen in the JSON beside it) builds the
//! stabilised treatment-and-censoring weights from the supplied probabilities, solves the weighted
//! normal equations by exact Gauss-Jordan elimination, forms the one-subject-per-cluster sandwich
//! with the `m / (m - p)` factor and the Kish effective sample size. Cases cover varying
//! propensities, censoring with unobserved subjects, probabilities at the positivity floor, an
//! outcome offset of 1e7, a rare treatment period and the smallest admissible sample.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::longitudinal_regime::SubjectHistories;
use antecedent_estimate::marginal_structural_model::{
    MsmSummary, fit_binary_msm, pointwise_intervals_95,
};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-300), "{what}: got {got} want {want}");
}

fn fit(case: &Value) -> MsmSummary {
    let treatment: Vec<bool> =
        case["treatment"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect();
    let observed: Vec<bool> =
        case["observed"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect();
    let (p, c) = (floats(&case["treatment_probability"]), floats(&case["censoring_probability"]));
    fit_binary_msm(
        &floats(&case["y"]),
        &SubjectHistories {
            treatment: &treatment,
            treatment_probability: &p,
            outcome_observed: &observed,
            censoring_probability: &c,
            periods: case["periods"].as_u64().unwrap() as usize,
            minimum_probability: case["minimum_probability"].as_f64().unwrap(),
        },
        &floats(&case["numerator"]),
    )
    .unwrap()
}

#[test]
fn coefficients_variances_and_weights_match_the_exact_reference() {
    let cases = serde_json::from_str::<Value>(include_str!(
        "oracle_data/marginal_structural_model_reference.json"
    ))
    .unwrap();
    for case in cases["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let want = &case["expected"];
        let got = fit(case);
        let beta = floats(&want["beta"]);
        let var = floats(&want["var"]);
        let rel = if name.contains("offset") { 1e-7 } else { 1e-10 };
        close(got.intercept, beta[0], rel, &format!("{name} intercept"));
        close(
            got.intercept_standard_error.powi(2),
            var[0],
            1e-8 * if name.contains("offset") { 1e2 } else { 1.0 },
            &format!("{name} intercept variance"),
        );
        for t in 0..got.period_effects.len() {
            close(got.period_effects[t], beta[t + 1], rel, &format!("{name} effect {t}"));
            close(
                got.standard_errors[t].powi(2),
                var[t + 1],
                1e-8 * if name.contains("offset") { 1e2 } else { 1.0 },
                &format!("{name} variance {t}"),
            );
        }
        close(got.effective_sample_size, want["ess"].as_f64().unwrap(), 1e-12, name);
        close(got.maximum_weight, want["max_weight"].as_f64().unwrap(), 1e-12, name);
        assert_eq!(got.observed_subjects as u64, want["observed"].as_u64().unwrap(), "{name}");
    }
}

#[test]
fn intervals_require_every_support_floor_and_use_the_normal_quantile() {
    let fit = |subjects_obs: usize, ess: f64, se: f64| MsmSummary {
        intercept: 1.0,
        intercept_standard_error: se,
        period_effects: vec![2.0, -3.0],
        standard_errors: vec![se, 2.0 * se],
        effective_sample_size: ess,
        maximum_weight: 3.0,
        observed_subjects: subjects_obs,
    };
    let ok = fit(200, 150.0, 0.25);
    let bands = pointwise_intervals_95(&ok, 300).expect("floors met exactly");
    let z = 1.959_963_984_540_054;
    close(bands.intercept[0], 1.0 - z * 0.25, 1e-14, "lower");
    close(bands.intercept[1], 1.0 + z * 0.25, 1e-14, "upper");
    close(bands.period_effects[1][1], -3.0 + z * 0.5, 1e-14, "second effect upper");
    assert!(pointwise_intervals_95(&ok, 299).is_none(), "subjects floor");
    assert!(pointwise_intervals_95(&fit(199, 150.0, 0.25), 300).is_none(), "observed floor");
    assert!(pointwise_intervals_95(&fit(200, 149.999, 0.25), 300).is_none(), "ESS floor");
    assert!(pointwise_intervals_95(&fit(200, 150.0, 0.0), 300).is_none(), "degenerate SE");
    assert!(pointwise_intervals_95(&fit(200, 150.0, f64::NAN), 300).is_none(), "non-finite SE");
}

#[test]
fn declarations_that_would_be_ignored_are_refused() {
    let cases = serde_json::from_str::<Value>(include_str!(
        "oracle_data/marginal_structural_model_reference.json"
    ))
    .unwrap();
    let case = &cases["cases"][0];
    let treatment: Vec<bool> =
        case["treatment"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect();
    let observed: Vec<bool> =
        case["observed"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect();
    let (p, c) = (floats(&case["treatment_probability"]), floats(&case["censoring_probability"]));
    let y = floats(&case["y"]);
    let run = |numerator: &[f64], floor: f64, p: &[f64], obs: &[bool], periods: usize| {
        fit_binary_msm(
            &y,
            &SubjectHistories {
                treatment: &treatment,
                treatment_probability: p,
                outcome_observed: obs,
                censoring_probability: &c,
                periods,
                minimum_probability: floor,
            },
            numerator,
        )
    };
    assert!(run(&[0.5, 0.5], 0.01, &p, &observed, 2).is_ok());
    let message = |error: antecedent_estimate::longitudinal_regime::RegimeKernelError| error.message().to_owned();
    assert_eq!(
        message(run(&[0.5, 0.5, 0.5], 0.01, &p, &observed, 2).unwrap_err()),
        "MSM arrays must have matching subject and period dimensions",
        "numerator length"
    );
    assert_eq!(
        message(run(&[0.5, 0.005], 0.01, &p, &observed, 2).unwrap_err()),
        "stabilizing numerator probabilities violate the declared positivity floor"
    );
    assert_eq!(
        message(run(&[0.5, 0.5], 0.6, &p, &observed, 2).unwrap_err()),
        "minimum probability must be finite and in (0, 0.5]"
    );
    assert_eq!(
        message(run(&[0.5, 0.5], 0.01, &p, &observed[1..], 2).unwrap_err()),
        "MSM arrays must have matching subject and period dimensions",
        "observed length"
    );
    let mut low = p.clone();
    low[7] = 0.005;
    assert_eq!(
        message(run(&[0.5, 0.5], 0.01, &low, &observed, 2).unwrap_err()),
        "sequential treatment positivity is violated"
    );
    let none = vec![false; observed.len()];
    assert_eq!(
        message(run(&[0.5, 0.5], 0.01, &p, &none, 2).unwrap_err()),
        "MSM requires more observed subjects than coefficients"
    );
}
