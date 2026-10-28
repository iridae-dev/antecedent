//! The neighbourhood-truncated leave-one-out kernel smoother against a brute-force 60-digit
//! evaluation of its declared definition
//! (`oracle_data/nonparametric_sensitivity_neighborhood_reference.py`).
//!
//! Compiled as a child of `sensitivity` (`#[path]` there) because the smoother is private.
//! The reference selects, for every row, the `neighbors` nearest other covariate cells plus the
//! row's own cell when it holds other rows, weights each by exp(-(d^2 - d_min^2)/(2 h^2)) and
//! forms the multiplicity-weighted mean with the row itself removed. Regimes: covariate cells
//! formed by repeated rows, a neighbour count larger than the number of cells (the exact
//! smoother), bandwidths from 1e-3 to 25, one to three covariate dimensions, a two-row sample
//! whose rows are identical, and a one-row sample that has no other row to smooth over.

use antecedent_core::ExecutionContext;
use serde_json::Value;

use super::neighborhood_loo_predict_pair;

fn reference() -> Value {
    serde_json::from_str(include_str!(
        "../tests/oracle_data/nonparametric_sensitivity_neighborhood_reference.json"
    ))
    .expect("frozen reference")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(|x| x.as_f64().expect("f64")).collect()
}

#[test]
fn truncated_smoother_equals_the_brute_force_reference() {
    let ctx = ExecutionContext::for_tests(1);
    for case in reference().as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let cov: Vec<f64> =
            case["cov"].as_array().unwrap().iter().flat_map(|row| floats(row)).collect();
        let (y1, y2) = (floats(&case["y1"]), floats(&case["y2"]));
        let dim = case["dim"].as_u64().unwrap() as usize;
        let (h, k) = (case["h"].as_f64().unwrap(), case["neighbors"].as_u64().unwrap() as usize);
        let (hat1, hat2) = neighborhood_loo_predict_pair(&y1, &y2, &cov, dim, h, k, &ctx)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        for (got, want) in
            [(hat1, floats(&case["expected"][0])), (hat2, floats(&case["expected"][1]))]
        {
            assert_eq!(got.len(), want.len(), "{name}");
            for (i, (g, w)) in got.iter().zip(&want).enumerate() {
                assert!((g - w).abs() <= 1e-10 * w.abs().max(1.0), "{name}[{i}]: {g} vs {w}");
            }
        }
    }
}

#[test]
fn one_row_has_nothing_to_smooth_over() {
    let ctx = ExecutionContext::for_tests(1);
    assert!(neighborhood_loo_predict_pair(&[1.0], &[2.0], &[0.5], 1, 1.0, 3, &ctx).is_err());
}

#[test]
fn the_neighbour_count_changes_the_answer() {
    // A declared option must be used: one neighbour and eight neighbours differ on the same data.
    let ctx = ExecutionContext::for_tests(1);
    let frozen = reference();
    let case = &frozen.as_array().unwrap()[1];
    let cov: Vec<f64> = case["cov"].as_array().unwrap().iter().flat_map(|r| floats(r)).collect();
    let (y1, y2) = (floats(&case["y1"]), floats(&case["y2"]));
    let run = |k| neighborhood_loo_predict_pair(&y1, &y2, &cov, 2, 25.0, k, &ctx).unwrap().0;
    let (a, b) = (run(1), run(8));
    assert!(a.iter().zip(&b).any(|(x, y)| (x - y).abs() > 1e-6));
}
