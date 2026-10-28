//! Symbolic conditional mutual information against an exhaustively enumerated permutation null.
//!
//! `oracle_data/symbolic_cmi_reference.py` (frozen in the JSON beside it) evaluates the
//! stratum-size-weighted plug-in mutual information at 60 digits and enumerates every
//! arrangement of Y within the conditioning strata, so the exact permutation p-value is a ratio
//! of integers in which ties between mathematically equal statistics count. The product's
//! add-one Monte Carlo p-value has expectation `(1 + R p) / (1 + R)`; with R = 20000 replicates it
//! must lie within five binomial standard errors of that. Statistics are asserted to 1e-12.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_stats::{
    CiBatchRequest, CiQuery, CiWorkspace, ConditionalIndependenceTest, ConfidenceMethod, GSquared,
    SignificanceMethod, SymbolicCmi,
};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

#[test]
fn symbolic_statistic_and_permutation_p_value_match_the_enumerated_null() {
    check(&SymbolicCmi::new(), "cases", "statistic", "exact_p", 1e-12);
}

#[test]
fn g_squared_permutation_p_value_counts_tied_tables_like_the_enumerated_null() {
    check(&GSquared::new(), "g2_cases", "statistic", "exact_p", 1e-10);
}

fn check(
    test: &dyn ConditionalIndependenceTest,
    set: &str,
    statistic: &str,
    exact_key: &str,
    tolerance: f64,
) {
    const REPLICATES: u32 = 20_000;
    let doc: Value =
        serde_json::from_str(include_str!("oracle_data/symbolic_cmi_reference.json")).unwrap();
    for (index, case) in doc[set].as_array().unwrap().iter().enumerate() {
        let name = case["name"].as_str().unwrap();
        let (x, y, z) = (floats(&case["x"]), floats(&case["y"]), floats(&case["z"]));
        let conditioned = z.iter().any(|v| *v != z[0]);
        let columns: Vec<&[f64]> = vec![&x, &y, &z];
        let z_flat: Vec<usize> = if conditioned { vec![2] } else { Vec::new() };
        let queries = [CiQuery { x: 0, y: 1, z_start: 0, z_len: z_flat.len() }];
        let request = CiBatchRequest {
            columns: &columns,
            queries: &queries,
            z_flat: &z_flat,
            significance: SignificanceMethod::BlockShuffle {
                replicates: REPLICATES,
                block_size: 1,
            },
            confidence: ConfidenceMethod::None,
        };
        let result = test
            .test_batch_adhoc(
                &request,
                &mut CiWorkspace::default(),
                &ExecutionContext::for_tests(11 + index as u64),
            )
            .unwrap()
            .results
            .remove(0);
        let want = case[statistic].as_f64().unwrap();
        assert!(
            (result.statistic - want).abs() <= tolerance * want.abs().max(1.0),
            "{name}: {} vs {want}",
            result.statistic
        );
        let exact = case[exact_key].as_f64().unwrap();
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
