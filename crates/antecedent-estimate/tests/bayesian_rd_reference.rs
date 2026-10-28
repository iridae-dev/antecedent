//! The Bayesian sharp-RD posterior against an exact local-linear reference.
//!
//! `oracle_data/bayesian_rd_reference.py` (frozen in the JSON beside it) forms the window, the
//! ridge-form posterior of the four local-linear coefficients under a coefficient prior scaled by
//! the plug-in residual variance, and the Normal posterior of the jump, in 40-digit arithmetic.
//! Cases cover a wide and a tight prior, a narrow window and a shifted cutoff, with a curved
//! mean so the plug-in variance is not trivial. The scale test pins the declared convention that
//! the prior scale is in units of the residual sd.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::bayesian_rd::{RdSample, fit_bayesian_sharp_rd};
use antecedent_estimate::util::DrawSpec;
use serde_json::Value;

const DRAWS: usize = 300_000;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/bayesian_rd_reference.json"))
        .unwrap()
        .as_array()
        .unwrap()
        .clone()
}

#[test]
fn the_jump_posterior_matches_the_exact_normal_reference() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let (running, treatment, outcome) =
            (floats(&case["running"]), floats(&case["treatment"]), floats(&case["outcome"]));
        let fit = fit_bayesian_sharp_rd(
            &RdSample { running: &running, treatment: &treatment, outcome: &outcome },
            case["cutoff"].as_f64().unwrap(),
            case["bandwidth"].as_f64().unwrap(),
            case["prior_sd"].as_f64().unwrap(),
            DrawSpec::new(DRAWS, 5),
        )
        .unwrap();
        let (mean, sd) = (case["mean"].as_f64().unwrap(), case["sd"].as_f64().unwrap());
        assert_eq!(fit.n_window as u64, case["n_window"].as_u64().unwrap(), "{name} window");
        assert!((fit.mean - mean).abs() < 5.0 * sd / (DRAWS as f64).sqrt(), "{name} mean");
        assert!((fit.lower - case["lower"].as_f64().unwrap()).abs() < 0.03 * sd, "{name} lower");
        assert!((fit.upper - case["upper"].as_f64().unwrap()).abs() < 0.03 * sd, "{name} upper");
    }
}

#[test]
fn rescaling_the_outcome_rescales_the_posterior_because_the_prior_is_in_residual_units() {
    let case = cases().into_iter().find(|c| c["name"] == "tight_prior").unwrap();
    let (running, treatment, outcome) =
        (floats(&case["running"]), floats(&case["treatment"]), floats(&case["outcome"]));
    let scaled: Vec<f64> = outcome.iter().map(|v| 1000.0 * v).collect();
    let fit = |y: &[f64]| {
        fit_bayesian_sharp_rd(
            &RdSample { running: &running, treatment: &treatment, outcome: y },
            0.0,
            0.8,
            0.5,
            DrawSpec::new(2000, 9),
        )
        .unwrap()
    };
    let (a, b) = (fit(&outcome), fit(&scaled));
    // The same seed gives the same standard-normal draws, so the posteriors scale exactly.
    assert!((b.mean / a.mean - 1000.0).abs() < 1e-9, "{} {}", a.mean, b.mean);
    assert!((b.lower / a.lower - 1000.0).abs() < 1e-9);
    assert!((b.upper / a.upper - 1000.0).abs() < 1e-9);
}
