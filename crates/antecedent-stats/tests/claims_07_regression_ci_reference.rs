//! `RegressionCi` and scalar-block `MultivariatePartialCorrelation` against the same 60-digit
//! partial-correlation references as `PartialCorrelation` (`oracle_data/ci_reference.py`).
//!
//! The scalar equivalence is established against multiprecision residualization, not by
//! comparing the entry points with each other.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_stats::{
    CiBatchRequest, CiQuery, CiWorkspace, ConditionalIndependenceTest, ConfidenceMethod,
    MultivariatePartialCorrelation, RegressionCi, SignificanceMethod,
};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn rel(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(f64::MIN_POSITIVE)
}

fn check(test: &dyn ConditionalIndependenceTest, label: &str) {
    let doc: Value = serde_json::from_str(include_str!("oracle_data/ci_reference.json")).unwrap();
    let mut failures = Vec::new();
    let mut cases = 0;
    for case in doc["parcorr"].as_array().unwrap() {
        cases += 1;
        let name = case["name"].as_str().unwrap();
        let owned: Vec<Vec<f64>> = case["columns"].as_array().unwrap().iter().map(floats).collect();
        let cols: Vec<&[f64]> = owned.iter().map(Vec::as_slice).collect();
        let z: Vec<usize> =
            case["z"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as usize).collect();
        let want_r = case["r"].as_f64().unwrap();
        let want_p = case["p"].as_f64().unwrap();
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: z.len() }];
        let req = CiBatchRequest {
            columns: &cols,
            queries: &queries,
            z_flat: &z,
            significance: SignificanceMethod::Analytic,
            confidence: ConfidenceMethod::None,
        };
        let mut ws = CiWorkspace::default();
        let got = test
            .test_batch_adhoc(&req, &mut ws, &ExecutionContext::for_tests(7))
            .unwrap()
            .results
            .remove(0);
        let r_tol = 5e-13 * (1.0 - want_r.abs()).max(1e-300) + 4e-16;
        if (got.statistic - want_r).abs() > r_tol {
            failures.push(format!("{label}/{name}: r {} vs {want_r}", got.statistic));
        }
        if rel(got.p_value, want_p) > 1e-8 {
            failures.push(format!("{label}/{name}: p {:e} vs {want_p:e}", got.p_value));
        }
        if (got.df - case["df"].as_f64().unwrap()).abs() > 0.0 {
            failures.push(format!("{label}/{name}: df {} vs {}", got.df, case["df"]));
        }
    }
    assert!(cases >= 5, "reference must carry the parcorr grid");
    assert!(failures.is_empty(), "{}", failures.join("\n"));
}

#[test]
fn claims_07_regression_ci_matches_multiprecision_partial_correlation() {
    check(&RegressionCi::new(), "regression");
}

#[test]
fn claims_07_scalar_block_matches_multiprecision_partial_correlation() {
    check(&MultivariatePartialCorrelation::new(), "scalar-block");
}
