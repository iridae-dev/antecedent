//! Distance-correlation and kNN permutation tests against an exhaustively enumerated null with ties.
//!
//! `oracle_data/continuous_null_reference.py` (frozen in the JSON beside it) evaluates the distance
//! correlation and the kNN distance statistic at 60 digits and enumerates every arrangement of Y,
//! so the exact permutation p-value is a ratio of integers in which arrangements that tie the
//! observed statistic as real numbers count. The product's add-one Monte Carlo p-value has
//! expectation `(1 + R p) / (1 + R)`; with R = 20000 replicates it must lie within five binomial
//! standard errors of that. Statistics are asserted to 1e-12 (the distance correlation as its square).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_stats::{
    CiBatchRequest, CiQuery, CiWorkspace, ConditionalIndependenceTest, ConfidenceMethod, Gpdc,
    KnnDependence, SignificanceMethod,
};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn check(kind: &str) {
    const REPLICATES: u32 = 20_000;
    let doc: Value =
        serde_json::from_str(include_str!("oracle_data/continuous_null_reference.json")).unwrap();
    for (index, case) in doc["cases"].as_array().unwrap().iter().enumerate() {
        if case["kind"].as_str().unwrap() != kind {
            continue;
        }
        let name = case["name"].as_str().unwrap();
        let (x, y) = (floats(&case["x"]), floats(&case["y"]));
        let columns: Vec<&[f64]> = vec![&x, &y];
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: 0 }];
        let request = CiBatchRequest {
            columns: &columns,
            queries: &queries,
            z_flat: &[],
            significance: SignificanceMethod::BlockShuffle {
                replicates: REPLICATES,
                block_size: 1,
            },
            confidence: ConfidenceMethod::None,
        };
        let test: Box<dyn ConditionalIndependenceTest> = if kind == "dcor" {
            Box::new(Gpdc::new())
        } else {
            Box::new(KnnDependence::new(case["k"].as_u64().unwrap() as usize))
        };
        let result = test
            .test_batch_adhoc(
                &request,
                &mut CiWorkspace::default(),
                &ExecutionContext::for_tests(31 + index as u64),
            )
            .unwrap()
            .results
            .remove(0);
        let want = case["statistic"].as_f64().unwrap();
        // The distance correlation is the square root of a difference of nearly equal sums, so
        // its rounding error is about 1e-8 where the true value is 0; compare its square.
        let (got, want) = if kind == "dcor" {
            (result.statistic.powi(2), want * want)
        } else {
            (result.statistic, want)
        };
        assert!((got - want).abs() <= 1e-12 * want.abs().max(1.0), "{name}: {got} vs {want}");
        let exact = case["exact_p"].as_f64().unwrap();
        let replicates = f64::from(REPLICATES);
        let expected = (1.0 + replicates * exact) / (1.0 + replicates);
        let sigma = (exact * (1.0 - exact) / replicates).sqrt().max(1.0 / replicates);
        assert!(
            (result.p_value - expected).abs() <= 5.0 * sigma,
            "{name}: p {} vs expected {expected} (exact {exact}, five sigma {})",
            result.p_value,
            5.0 * sigma
        );
    }
}

#[test]
fn distance_correlation_permutation_p_value_counts_tied_arrangements_like_the_enumerated_null() {
    check("dcor");
}

#[test]
fn knn_distance_permutation_p_value_counts_tied_arrangements_like_the_enumerated_null() {
    check("knn");
}
