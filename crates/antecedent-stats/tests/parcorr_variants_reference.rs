//! Block, weighted and normal-score partial-correlation tests against 60-digit references.
//!
//! `oracle_data/parcorr_variants_reference.py` residualizes, forms Wilks' Lambda and the
//! canonical correlations, the weighted Pearson statistic with its Kish degrees of freedom and
//! the van der Waerden scores from their definitions in multiprecision arithmetic.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_stats::{
    BlockTest, CiBatchRequest, CiQuery, CiWorkspace, ConditionalIndependenceTest, ConfidenceMethod,
    MultivariatePartialCorrelation, RobustPartialCorrelation, SignificanceMethod,
    WeightedPartialCorrelation,
};
use serde_json::Value;

fn doc() -> Value {
    serde_json::from_str(include_str!("oracle_data/parcorr_variants_reference.json")).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn columns(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().unwrap().iter().map(floats).collect()
}

fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(f64::MIN_POSITIVE)
}

#[test]
fn block_test_matches_wilks_lambda_bartlett_p_and_leading_canonical_correlation() {
    for case in doc()["block"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (x, y, z) = (columns(&case["X"]), columns(&case["Y"]), columns(&case["Z"]));
        let all: Vec<&[f64]> = x.iter().chain(&y).chain(&z).map(Vec::as_slice).collect();
        let x_cols: Vec<usize> = (0..x.len()).collect();
        let y_cols: Vec<usize> = (x.len()..x.len() + y.len()).collect();
        let z_cols: Vec<usize> = (x.len() + y.len()..all.len()).collect();
        let got = MultivariatePartialCorrelation::new()
            .test_blocks(
                &all,
                &BlockTest { x_cols: &x_cols, y_cols: &y_cols, z_flat: &z_cols },
                SignificanceMethod::Analytic,
                &mut CiWorkspace::default(),
                &ExecutionContext::for_tests(3),
            )
            .unwrap();
        assert_eq!(got.df, case["df"].as_f64().unwrap(), "{name}");
        assert!(
            rel(got.p_value, case["p"].as_f64().unwrap()) < 1e-9,
            "{name}: p {} vs {}",
            got.p_value,
            case["p"]
        );
        assert!(
            (got.statistic - case["rho"].as_f64().unwrap()).abs() < 1e-10,
            "{name}: leading canonical correlation {} vs {}",
            got.statistic,
            case["rho"]
        );
    }
}

#[test]
fn weighted_partial_correlation_matches_weighted_least_squares_and_kish_df() {
    for case in doc()["weighted"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (x, y, z) = (floats(&case["x"]), floats(&case["y"]), columns(&case["Z"]));
        let all: Vec<&[f64]> =
            [x.as_slice(), y.as_slice()].into_iter().chain(z.iter().map(Vec::as_slice)).collect();
        let z_flat: Vec<usize> = (2..all.len()).collect();
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: z_flat.len() }];
        let req = CiBatchRequest {
            columns: &all,
            queries: &queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::None,
        };
        let got = WeightedPartialCorrelation::new(floats(&case["w"]))
            .test_batch_adhoc(&req, &mut CiWorkspace::default(), &ExecutionContext::for_tests(3))
            .unwrap()
            .results
            .remove(0);
        assert!((got.statistic - case["r"].as_f64().unwrap()).abs() < 1e-11, "{name}: r");
        assert!((got.df - case["df"].as_f64().unwrap()).abs() < 1e-9, "{name}: df {}", got.df);
        assert!(rel(got.p_value, case["p"].as_f64().unwrap()) < 1e-8, "{name}: p");
    }
}

#[test]
fn robust_partial_correlation_matches_van_der_waerden_scores() {
    for case in doc()["robust"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (x, y, z) = (floats(&case["x"]), floats(&case["y"]), columns(&case["Z"]));
        let all: Vec<&[f64]> =
            [x.as_slice(), y.as_slice()].into_iter().chain(z.iter().map(Vec::as_slice)).collect();
        let z_flat: Vec<usize> = (2..all.len()).collect();
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: z_flat.len() }];
        let req = CiBatchRequest {
            columns: &all,
            queries: &queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::None,
        };
        let got = RobustPartialCorrelation::new()
            .test_batch_adhoc(&req, &mut CiWorkspace::default(), &ExecutionContext::for_tests(3))
            .unwrap()
            .results
            .remove(0);
        assert!(
            (got.statistic - case["r"].as_f64().unwrap()).abs() < 1e-9,
            "{name}: r {} vs {}",
            got.statistic,
            case["r"]
        );
        assert_eq!(got.df, case["df"].as_f64().unwrap(), "{name}");
        assert!(
            rel(got.p_value, case["p"].as_f64().unwrap()) < 1e-6,
            "{name}: p {} vs {}",
            got.p_value,
            case["p"]
        );
    }
}
