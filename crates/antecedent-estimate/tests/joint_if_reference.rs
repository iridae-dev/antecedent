//! Joint influence-function covariance, weighted mean and frozen-weight mixture against exact
//! rational references.
//!
//! `oracle_data/joint_if_reference.py` (frozen in the JSON beside it) evaluates, with every
//! binary64 input converted to an exact fraction, the Hajek mean, its influence function
//! `n (w_i / W)(phi_i - theta)`, the joint covariance `sum psi_ij psi_ik / (n (n - 1))` and the
//! frozen-weight mixture `sum_a (g_a / sum g) phi_a`. Cases cover two rows, a 1e8 offset,
//! weights spanning eighteen orders of magnitude with tied scores, zero-weight rows, 1e150
//! weights, collinear columns and zero-weight atoms. This reviews the linear combination and
//! the covariance arithmetic, not the nuisance fits that produce the scores.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::{
    frozen_weight_mixture_scores, joint_influence_covariance, weighted_mean,
};
use serde_json::Value;

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/joint_if_reference.json")).unwrap()
        ["cases"]
        .as_array()
        .unwrap()
        .clone()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-300), "{what}: got {got} want {want}");
}

fn tolerance(name: &str) -> f64 {
    if name.contains("offset") { 1e-6 } else { 1e-11 }
}

#[test]
fn means_and_joint_covariance_match_the_exact_rational_reference() {
    let mut seen = 0;
    for case in cases() {
        if case["kind"] != "joint" {
            continue;
        }
        seen += 1;
        let name = case["name"].as_str().unwrap();
        let cols: Vec<Vec<f64>> = case["cols"].as_array().unwrap().iter().map(floats).collect();
        let weights = (!case["weights"].is_null()).then(|| floats(&case["weights"]));
        let refs: Vec<&[f64]> = cols.iter().map(Vec::as_slice).collect();
        let rel = tolerance(name);
        let means = floats(&case["means"]);
        for (j, col) in cols.iter().enumerate() {
            let got = weighted_mean(col, weights.as_deref()).unwrap();
            close(got, means[j], rel, &format!("{name} mean {j}"));
        }
        let cov = joint_influence_covariance(&refs, weights.as_deref()).unwrap();
        let want: Vec<Vec<f64>> = case["cov"].as_array().unwrap().iter().map(floats).collect();
        for i in 0..cols.len() {
            for j in 0..cols.len() {
                let (g, w) = (cov.get(i, j), want[i][j]);
                // a singular covariance has exact zeros the product reaches only to rounding
                let floor = want[i][i].abs().max(want[j][j].abs()).sqrt() * 1e-9;
                assert!(
                    (g - w).abs() <= rel * w.abs() + floor * f64::from(u8::from(w == 0.0)),
                    "{name} cov[{i}][{j}]: got {g} want {w}"
                );
                assert_eq!(cov.get(i, j), cov.get(j, i), "{name}: covariance is symmetric");
            }
            assert!((cov.se(i) - want[i][i].sqrt()).abs() <= rel * want[i][i].sqrt() + 1e-300);
        }
    }
    assert_eq!(seen, 10);
}

#[test]
fn the_mixture_column_and_its_standard_error_match_the_exact_reference() {
    let mut seen = 0;
    for case in cases() {
        if case["kind"] != "mixture" {
            continue;
        }
        seen += 1;
        let name = case["name"].as_str().unwrap();
        let atoms: Vec<Vec<f64>> = case["atoms"].as_array().unwrap().iter().map(floats).collect();
        let weights = floats(&case["weights"]);
        let refs: Vec<&[f64]> = atoms.iter().map(Vec::as_slice).collect();
        let mixed = frozen_weight_mixture_scores(&refs, &weights).unwrap();
        for (r, (g, w)) in mixed.iter().zip(floats(&case["mixed"])).enumerate() {
            close(*g, w, 1e-13, &format!("{name} row {r}"));
        }
        let se = joint_influence_covariance(&[mixed.as_slice()], None).unwrap().se(0);
        close(se * se, case["mixed_se2"].as_f64().unwrap(), tolerance(name), name);
    }
    assert_eq!(seen, 5);
}

#[test]
fn an_atom_with_zero_weight_is_not_mixed_even_when_its_scores_are_not_finite() {
    let ok = [1.0, 2.0, 3.0];
    let bad = [f64::NAN, 1.0, 2.0];
    let mixed = frozen_weight_mixture_scores(&[&ok, &bad], &[1.0, 0.0]).unwrap();
    assert_eq!(mixed, ok);
    assert!(frozen_weight_mixture_scores(&[&ok, &bad], &[1.0, 1.0]).is_err());
}

#[test]
fn invalid_declarations_are_refused_not_ignored() {
    let a = [1.0, 2.0, 4.0];
    let short = [1.0, 2.0];
    assert!(joint_influence_covariance(&[], None).is_err());
    assert!(joint_influence_covariance(&[&[1.0]], None).is_err(), "one row has no variance");
    assert!(joint_influence_covariance(&[&a, &short], None).is_err());
    assert!(joint_influence_covariance(&[&a], Some(&[1.0, 1.0])).is_err(), "weight length");
    assert!(joint_influence_covariance(&[&a], Some(&[1.0, -1.0, 1.0])).is_err());
    assert!(joint_influence_covariance(&[&a], Some(&[0.0, 0.0, 0.0])).is_err());
    assert!(joint_influence_covariance(&[&a], Some(&[1.0, f64::NAN, 1.0])).is_err());
    assert!(weighted_mean(&[], None).is_err());
    assert!(weighted_mean(&[1.0, f64::INFINITY], None).is_err());
    assert!(frozen_weight_mixture_scores(&[&a], &[0.0]).is_err());
    assert!(frozen_weight_mixture_scores(&[&a, &short], &[1.0, 1.0]).is_err());
    assert!(frozen_weight_mixture_scores(&[&a, &a], &[1.0, -1.0]).is_err());
    assert!(frozen_weight_mixture_scores(&[&a, &a], &[1.0]).is_err());
    assert!(frozen_weight_mixture_scores(&[&a], &[f64::INFINITY]).is_err());
}
