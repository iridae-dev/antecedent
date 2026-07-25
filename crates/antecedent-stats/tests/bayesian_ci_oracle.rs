//! Frozen clean-room conjugate and posterior-predictive Bayesian CI parity.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::fs;
use std::path::PathBuf;

use antecedent_core::ExecutionContext;
use antecedent_stats::{
    BayesFactorCi, CiBatchRequest, CiQuery, CiWorkspace, ConditionalIndependenceTest,
    ConfidenceMethod, PosteriorDependenceCi, PosteriorPredictiveCi, SignificanceMethod,
};
use serde_json::Value as JsonValue;

fn fixture() -> JsonValue {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/discovery/bayesian_ci/expected.json");
    serde_json::from_str(&fs::read_to_string(path).expect("Bayesian CI fixture"))
        .expect("parse Bayesian CI fixture")
}

fn columns(case: &JsonValue) -> (Vec<Vec<f64>>, Vec<usize>) {
    let columns = case["columns"]
        .as_array()
        .unwrap()
        .iter()
        .map(|column| column.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
        .collect();
    let z = case["z_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    (columns, z)
}

fn run(test: &dyn ConditionalIndependenceTest, case: &JsonValue, seed: u64) -> (f64, f64) {
    let (owned, z) = columns(case);
    let refs: Vec<&[f64]> = owned.iter().map(Vec::as_slice).collect();
    let query = [CiQuery { x: 0, y: 1, z_start: 0, z_len: z.len() }];
    let request = CiBatchRequest {
        columns: &refs,
        queries: &query,
        z_flat: &z,
        significance: SignificanceMethod::Analytic,
        confidence: ConfidenceMethod::None,
    };
    let mut workspace = CiWorkspace::default();
    let output = test
        .test_batch_adhoc(&request, &mut workspace, &ExecutionContext::for_tests(seed))
        .unwrap();
    (output.results[0].statistic, output.results[0].p_value)
}

#[test]
fn conjugate_bayes_factor_and_posterior_probability_match_clean_room_oracle() {
    let fixture = fixture();
    let bf_atol = fixture["tolerances"]["log_bf_atol"].as_f64().unwrap();
    let probability_atol = fixture["tolerances"]["probability_atol"].as_f64().unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let reference = &case["reference"];
        let (log_bf, independence) = run(&BayesFactorCi::new(), case, 1);
        assert!(
            (log_bf - reference["log_bf10"].as_f64().unwrap()).abs() <= bf_atol,
            "{} log BF",
            case["name"]
        );
        assert!(
            (independence - reference["posterior_independence"].as_f64().unwrap()).abs()
                <= probability_atol,
            "{} independence mass",
            case["name"]
        );
        let (dependence, complement) = run(&PosteriorDependenceCi::new(), case, 2);
        assert!(
            (dependence - reference["posterior_dependence"].as_f64().unwrap()).abs()
                <= probability_atol,
            "{} dependence mass",
            case["name"]
        );
        assert!((dependence + complement - 1.0).abs() <= 1e-15);
    }
}

#[test]
fn posterior_predictive_statistic_and_tail_probability_match_external_calibration() {
    let fixture = fixture();
    let stat_atol = fixture["tolerances"]["ppc_statistic_atol"].as_f64().unwrap();
    let multiplier = fixture["tolerances"]["ppc_mcse_multiplier"].as_f64().unwrap();
    let floor = fixture["tolerances"]["ppc_absolute_floor"].as_f64().unwrap();
    for (index, case) in fixture["cases"].as_array().unwrap().iter().enumerate() {
        let reference = &case["reference"]["posterior_predictive"];
        let test = PosteriorPredictiveCi::new(1_999).with_seed(0xB4_C1 + index as u64);
        let (statistic, p_value) = run(&test, case, 3);
        let expected_statistic = reference["observed_abs_residual_correlation"].as_f64().unwrap();
        assert!(
            (statistic - expected_statistic).abs() <= stat_atol,
            "{} PPC statistic {statistic} != {expected_statistic}",
            case["name"]
        );
        let expected_p = reference["tail_probability"].as_f64().unwrap();
        let oracle_mcse = reference["mcse"].as_f64().unwrap();
        let native_mcse = (p_value * (1.0 - p_value) / 2_000.0).max(1.0 / 4_000_000.0).sqrt();
        let band = floor.max(multiplier * (oracle_mcse.powi(2) + native_mcse.powi(2)).sqrt());
        assert!(
            (p_value - expected_p).abs() <= band,
            "{} PPC p={p_value}, oracle={expected_p}, band={band}",
            case["name"]
        );
    }
}

#[test]
fn bayesian_ci_fixture_records_model_pins_and_no_generator() {
    let fixture = fixture();
    let oracle = &fixture["oracle"];
    assert_eq!(oracle["packages"]["numpy"]["version"], "2.1.3");
    assert_eq!(oracle["packages"]["scipy"]["version"], "1.14.1");
    assert_eq!(oracle["generation_location"], "temporary external harness; not retained");
    assert_eq!(fixture["model"]["alpha0"], 1e-3);
    assert_eq!(fixture["model"]["coefficient_prior_precision"], 0.01);
    for package in ["numpy", "scipy"] {
        assert_eq!(oracle["packages"][package]["metadata_sha256"].as_str().unwrap().len(), 64);
    }
}
