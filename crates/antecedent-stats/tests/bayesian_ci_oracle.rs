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

/// Partial correlation of columns 0 and 1 given `z`, from an independent dense OLS: normal
/// equations on `[1 | Z]` solved by Gauss-Jordan with partial pivoting, then Pearson of the
/// two residual vectors.
fn partial_correlation(columns: &[Vec<f64>], z: &[usize]) -> f64 {
    let n = columns[0].len();
    let q = 1 + z.len();
    let design = |r: usize, j: usize| if j == 0 { 1.0 } else { columns[z[j - 1]][r] };
    let residuals = |target: &[f64]| -> Vec<f64> {
        let mut a = vec![vec![0.0; q + 1]; q];
        for i in 0..q {
            for j in 0..q {
                a[i][j] = (0..n).map(|r| design(r, i) * design(r, j)).sum();
            }
            a[i][q] = (0..n).map(|r| design(r, i) * target[r]).sum();
        }
        for c in 0..q {
            let piv = (c..q).max_by(|&p, &s| a[p][c].abs().total_cmp(&a[s][c].abs())).unwrap();
            a.swap(c, piv);
            let d = a[c][c];
            for j in 0..=q {
                a[c][j] /= d;
            }
            for r in 0..q {
                if r != c {
                    let f = a[r][c];
                    for j in 0..=q {
                        let v = a[c][j];
                        a[r][j] -= f * v;
                    }
                }
            }
        }
        (0..n).map(|r| target[r] - (0..q).map(|j| a[j][q] * design(r, j)).sum::<f64>()).collect()
    };
    let (rx, ry) = (residuals(&columns[0]), residuals(&columns[1]));
    let nf = n as f64;
    let (mx, my) = (rx.iter().sum::<f64>() / nf, ry.iter().sum::<f64>() / nf);
    let sxy: f64 = rx.iter().zip(&ry).map(|(a, b)| (a - mx) * (b - my)).sum();
    let sxx: f64 = rx.iter().map(|a| (a - mx) * (a - mx)).sum();
    let syy: f64 = ry.iter().map(|b| (b - my) * (b - my)).sum();
    sxy / (sxx * syy).sqrt()
}

/// The Bayes factor is a Zellner g-prior (`g = n`) factor on the partial regression
/// coefficient, `log BF10 = -1/2 ln(1+g) - 1/2 (n-1-|Z|) ln(1 - g r^2/(1+g))`, a function of the
/// partial correlation only. Expected values are computed from the fixture columns with an
/// independent OLS, not read from the NIG-model oracle values (which price a different prior and
/// are kept in the fixture for the posterior-predictive model).
#[test]
fn g_prior_bayes_factor_and_posterior_probability_match_closed_form() {
    let fixture = fixture();
    let bf_atol = fixture["tolerances"]["log_bf_atol"].as_f64().unwrap();
    let probability_atol = fixture["tolerances"]["probability_atol"].as_f64().unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let (owned, z) = columns(case);
        let n = owned[0].len();
        let r = partial_correlation(&owned, &z);
        let g = n as f64;
        let d = n as f64 - 1.0 - z.len() as f64;
        let expected_log_bf = -0.5 * (1.0 + g).ln() - 0.5 * d * (1.0 - g / (1.0 + g) * r * r).ln();
        let expected_independence = 1.0 / (1.0 + expected_log_bf.exp());
        let expected_dependence = 1.0 / (1.0 + (-expected_log_bf).exp());

        let (log_bf, independence) = run(&BayesFactorCi::new(), case, 1);
        assert!(
            (log_bf - expected_log_bf).abs() <= bf_atol,
            "{} log BF {log_bf} != {expected_log_bf} (r={r})",
            case["name"]
        );
        assert!(
            (independence - expected_independence).abs() <= probability_atol,
            "{} independence mass",
            case["name"]
        );
        let (dependence, complement) = run(&PosteriorDependenceCi::new(), case, 2);
        assert!(
            (dependence - expected_dependence).abs() <= probability_atol,
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
