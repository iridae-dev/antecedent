//! Frozen Tigramite parity checks for symbolic CMI and GPDC.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, clippy::similar_names)]

use std::fs;
use std::path::PathBuf;

use antecedent_core::ExecutionContext;
use antecedent_stats::{
    CiBatchRequest, CiQuery, CiWorkspace, ConditionalIndependenceTest, ConfidenceMethod, Gpdc,
    SignificanceMethod, SymbolicCmi,
};
use serde_json::Value as JsonValue;

fn fixture() -> JsonValue {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/discovery/advanced_ci/expected.json");
    serde_json::from_str(&fs::read_to_string(path).expect("advanced CI fixture"))
        .expect("parse advanced CI fixture")
}

fn run_case(
    test: &dyn ConditionalIndependenceTest,
    case: &JsonValue,
    replicates: u32,
    seed: u64,
) -> (f64, f64) {
    let owned: Vec<Vec<f64>> = case["columns"]
        .as_array()
        .unwrap()
        .iter()
        .map(|col| col.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
        .collect();
    let columns: Vec<&[f64]> = owned.iter().map(Vec::as_slice).collect();
    let z_flat: Vec<usize> = case["z_indices"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let query = [CiQuery { x: 0, y: 1, z_start: 0, z_len: z_flat.len() }];
    let request = CiBatchRequest {
        columns: &columns,
        queries: &query,
        z_flat: &z_flat,
        significance: SignificanceMethod::BlockShuffle { replicates, block_size: 1 },
        confidence: ConfidenceMethod::None,
    };
    let mut workspace = CiWorkspace::default();
    let result = test
        .test_batch_adhoc(&request, &mut workspace, &ExecutionContext::for_tests(seed))
        .unwrap();
    (result.results[0].statistic, result.results[0].p_value)
}

#[test]
fn symbolic_cmi_matches_tigramite_statistics_and_decisions() {
    let fixture = fixture();
    let cases = fixture["symbolic_cases"].as_array().unwrap();
    let atol = fixture["tolerances"]["symbolic_stat_atol"].as_f64().unwrap();
    let p_atol = fixture["tolerances"]["symbolic_p_atol"].as_f64().unwrap();
    let test = SymbolicCmi::new();
    let mut observed = Vec::new();

    for (i, case) in cases.iter().enumerate() {
        let (statistic, p_value) = run_case(&test, case, 499, 0x51C0 + i as u64);
        let reference = &case["reference"];
        let expected_statistic = reference["statistic"].as_f64().unwrap();
        let expected_p = reference["p_value"].as_f64().unwrap();
        assert!(
            (statistic - expected_statistic).abs() <= atol,
            "{} statistic {statistic} != Tigramite {expected_statistic}",
            case["name"]
        );
        assert!(
            (p_value - expected_p).abs() <= p_atol,
            "{} p-value {p_value} too far from Tigramite {expected_p}",
            case["name"]
        );
        observed.push((case["name"].as_str().unwrap(), statistic, p_value));
    }

    let dep = observed.iter().find(|v| v.0 == "conditional_dependent").unwrap();
    let null = observed.iter().find(|v| v.0 == "conditional_null").unwrap();
    let relabel = observed.iter().find(|v| v.0 == "relabelled_conditional_dependent").unwrap();
    assert!(dep.1 > null.1 && dep.2 < 0.05 && null.2 > 0.05);
    assert!((dep.1 - relabel.1).abs() <= atol, "numeric symbol relabeling changed CMI");
}

#[test]
fn gpdc_matches_oracle_decisions_after_stable_residualization() {
    // MM-008: after centered Cholesky residualization, conditional nulls stay
    // above α=0.05 and the two-conditioner alternative rejects below it.
    let fixture = fixture();
    let cases = fixture["gpdc_cases"].as_array().unwrap();
    let atol = fixture["tolerances"]["gpdc_stat_atol"].as_f64().unwrap();
    let rtol = fixture["tolerances"]["gpdc_stat_rtol"].as_f64().unwrap();
    let test = Gpdc::new();
    let mut observed = Vec::new();

    for (i, case) in cases.iter().enumerate() {
        let (statistic, p_value) = run_case(&test, case, 199, 0x69DC + i as u64);
        let reference = &case["reference"];
        let expected_statistic = reference["statistic"].as_f64().unwrap();
        let expected_p = reference["p_value"].as_f64().unwrap();
        assert!(
            (statistic - expected_statistic).abs() <= atol + rtol * expected_statistic.abs(),
            "{} statistic {statistic} outside native-backend band around Tigramite {expected_statistic}",
            case["name"]
        );
        observed.push((case["name"].as_str().unwrap(), statistic, p_value, expected_p));
    }

    let conditional_dep =
        observed.iter().find(|v| v.0 == "conditional_nonlinear_dependent").unwrap();
    let conditional_null = observed.iter().find(|v| v.0 == "conditional_nonlinear_null").unwrap();
    let unconditional_dep =
        observed.iter().find(|v| v.0 == "unconditional_nonlinear_dependent").unwrap();
    let two_conditioners = observed.iter().find(|v| v.0 == "two_conditioners_dependent").unwrap();
    assert!(conditional_dep.1 > conditional_null.1);
    assert!(conditional_dep.2 < 0.05);
    assert!(unconditional_dep.2 < 0.05);
    assert!(conditional_null.2 > 0.05 && conditional_null.3 > 0.05);
    assert!(two_conditioners.2 < 0.05 && two_conditioners.3 < 0.05);
}

#[test]
fn advanced_ci_fixture_has_pinned_reproducible_provenance() {
    let fixture = fixture();
    let oracle = &fixture["oracle"];
    assert_eq!(
        oracle["generation_location"], "temporary external harness; not retained",
        "the repository must retain artifacts, not the external generator"
    );
    assert_eq!(oracle["packages"]["tigramite"]["version"], "5.2.1.30");
    assert_eq!(oracle["packages"]["numpy"]["version"], "1.23.5");
    assert!(oracle["command"].as_str().unwrap().contains("tigramite==5.2.1.30"));
    for package in ["tigramite", "numpy", "scipy", "scikit-learn", "dcor"] {
        assert_eq!(oracle["packages"][package]["metadata_sha256"].as_str().unwrap().len(), 64);
    }
}
