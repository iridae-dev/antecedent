//! The joint fixed-loading Bayesian IV posterior against an exact reference.
//!
//! `oracle_data/bayesian_iv_reference.py` (frozen in the JSON beside it) writes the log posterior
//! of the two-equation system from the factorisation `p(T | Z, W) p(Y | T, Z, W)`, differentiates
//! it numerically and takes one Newton step, so the posterior mean and covariance of the
//! structural coefficient are obtained without the stacked-regression algebra the product uses.
//! The single-instrument function, the general design function (with an instrument block and
//! exogenous covariates) and a tight prior are compared, as is the first-stage F statistic that
//! screens a weak instrument, and the weak-instrument refusal threshold is pinned.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::bayesian_iv::{
    IvDesignSample, IvSample, StageVariances, fit_bayesian_iv_joint_fixed_loading,
    fit_bayesian_iv_joint_fixed_loading_design,
};
use antecedent_estimate::util::DrawSpec;
use serde_json::Value;

const DRAWS: usize = 300_000;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/bayesian_iv_reference.json"))
        .unwrap()
        .as_array()
        .unwrap()
        .clone()
}

fn columns(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().flat_map(floats).collect()
}

fn variances(case: &Value) -> StageVariances {
    StageVariances {
        treatment: case["treatment_variance"].as_f64().unwrap(),
        outcome: case["outcome_variance"].as_f64().unwrap(),
    }
}

fn check(name: &str, fit: &antecedent_estimate::bayesian_iv::BayesianIvEstimate, case: &Value) {
    let (mean, sd) = (case["mean"].as_f64().unwrap(), case["sd"].as_f64().unwrap());
    assert!((fit.mean - mean).abs() < 5.0 * sd / (DRAWS as f64).sqrt(), "{name} mean {}", fit.mean);
    assert!((fit.lower - case["lower"].as_f64().unwrap()).abs() < 0.03 * sd, "{name} lower");
    assert!((fit.upper - case["upper"].as_f64().unwrap()).abs() < 0.03 * sd, "{name} upper");
    let f = case["first_stage_f"].as_f64().unwrap();
    assert!((fit.first_stage_f / f - 1.0).abs() < 1e-8, "{name} F {} vs {f}", fit.first_stage_f);
}

#[test]
fn the_design_function_matches_the_exact_posterior_with_instrument_blocks_and_covariates() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let treatment = floats(&case["treatment"]);
        let outcome = floats(&case["outcome"]);
        let instruments = columns(&case["instruments"]);
        let covariates = columns(&case["covariates"]);
        let fit = fit_bayesian_iv_joint_fixed_loading_design(
            &IvDesignSample {
                instruments: &instruments,
                n_instruments: case["instruments"].as_array().unwrap().len(),
                covariates: &covariates,
                n_covariates: case["covariates"].as_array().unwrap().len(),
                treatment: &treatment,
                outcome: &outcome,
            },
            case["prior_sd"].as_f64().unwrap(),
            variances(&case),
            DrawSpec::new(DRAWS, 2),
            0.0,
        )
        .unwrap();
        check(name, &fit, &case);
    }
}

#[test]
fn the_single_instrument_function_matches_the_exact_posterior() {
    let case = cases().into_iter().find(|c| c["name"] == "single_instrument").unwrap();
    let instrument = floats(&case["instruments"][0]);
    let (treatment, outcome) = (floats(&case["treatment"]), floats(&case["outcome"]));
    let fit = fit_bayesian_iv_joint_fixed_loading(
        &IvSample { instrument: &instrument, treatment: &treatment, outcome: &outcome },
        case["prior_sd"].as_f64().unwrap(),
        variances(&case),
        DrawSpec::new(DRAWS, 2),
        0.0,
    )
    .unwrap();
    check("single_instrument", &fit, &case);
}

#[test]
fn the_weak_instrument_screen_refuses_exactly_above_the_reference_statistic() {
    let case = cases().into_iter().find(|c| c["name"] == "single_instrument").unwrap();
    let instrument = floats(&case["instruments"][0]);
    let (treatment, outcome) = (floats(&case["treatment"]), floats(&case["outcome"]));
    let f = case["first_stage_f"].as_f64().unwrap();
    let run = |threshold: f64| {
        fit_bayesian_iv_joint_fixed_loading(
            &IvSample { instrument: &instrument, treatment: &treatment, outcome: &outcome },
            case["prior_sd"].as_f64().unwrap(),
            variances(&case),
            DrawSpec::new(1000, 2),
            threshold,
        )
    };
    assert!(run(f * 0.999).is_ok());
    assert!(run(f * 1.001).is_err());
}
