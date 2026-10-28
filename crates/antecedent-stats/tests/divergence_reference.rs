//! Divergence and two-sample statistics against multiprecision and exact-enumeration references.
//!
//! The frozen values in `oracle_data/divergence_reference.json` come from
//! `oracle_data/divergence_reference.py`: 100-digit Gaussian KL, Welch and chi-square tails, a
//! Mann-Whitney tie-corrected variance verified against the exact permutation variance of U,
//! and exact randomization p-values over every partition of the pooled values for the
//! small-sample likelihood-ratio and kernel tests.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::divergence::{
    classifier_two_sample, gaussian_kl, kernel_two_sample_with_permutations, mean_diff_two_sample,
    residual_likelihood_ratio,
};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("oracle_data/divergence_reference.json")).unwrap()
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

fn vec_f(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(f).collect()
}

fn close(got: f64, want: f64, rel: f64, abs: f64, what: &str) {
    assert!(
        (got - want).abs() <= rel * want.abs() + abs,
        "{what}: got {got:e}, want {want:e}"
    );
}

#[test]
fn gaussian_kl_matches_multiprecision_including_near_equal_and_overflowing_cases() {
    for case in fixture()["kl"].as_array().unwrap() {
        let args = vec_f(&case["args"]);
        let got = gaussian_kl(args[0], args[1], args[2], args[3]).unwrap();
        // Near-equal variances must keep full relative accuracy (no log1p cancellation).
        close(got, f(&case["kl"]), 1e-13, 0.0, &format!("kl{args:?}"));
    }
}

#[test]
fn welch_statistic_and_p_value_match_multiprecision() {
    for case in fixture()["welch"].as_array().unwrap() {
        let (stat, p) = mean_diff_two_sample(&vec_f(&case["a"]), &vec_f(&case["b"])).unwrap();
        close(stat, f(&case["stat"]), 1e-13, 0.0, "welch statistic");
        // p down to 8e-12 (a 30-sigma-scale split with five per arm) keeps relative accuracy.
        close(p, f(&case["p"]), 1e-9, 0.0, "welch p");
    }
}

#[test]
fn mann_whitney_p_uses_the_exact_tie_corrected_permutation_variance() {
    for case in fixture()["mann_whitney"].as_array().unwrap() {
        let (stat, p) = classifier_two_sample(&vec_f(&case["a"]), &vec_f(&case["b"])).unwrap();
        close(stat, f(&case["stat"]), 1e-14, 1e-15, "auc distance");
        close(p, f(&case["p"]), 1e-10, 1e-15, "mann-whitney p");
    }
}

#[test]
fn asymptotic_likelihood_ratio_matches_statistic_and_chi_square_tail() {
    for case in fixture()["lr_asymptotic"].as_array().unwrap() {
        let (stat, p) =
            residual_likelihood_ratio(&vec_f(&case["left"]), &vec_f(&case["right"])).unwrap();
        close(stat, f(&case["stat"]), 1e-11, 1e-12, "lr statistic");
        // The tail goes down to 7e-70; it must not be clamped or lose relative accuracy.
        close(p, f(&case["p"]), 1e-9, 0.0, "lr p");
    }
}

#[test]
fn small_segment_likelihood_ratio_p_is_the_exact_randomization_p_within_sampling_error() {
    let permutations = 999.0_f64;
    for case in fixture()["lr_exact"].as_array().unwrap() {
        let (stat, p) =
            residual_likelihood_ratio(&vec_f(&case["left"]), &vec_f(&case["right"])).unwrap();
        close(stat, f(&case["stat"]), 1e-11, 1e-12, "lr statistic");
        // Add-one Monte Carlo p-value: mean (1 + B p) / (B + 1); five binomial sigmas of slack.
        let exact = f(&case["p_exact"]);
        let mean = (1.0 + permutations * exact) / (permutations + 1.0);
        let sigma = (exact * (1.0 - exact) / permutations).sqrt();
        assert!(
            (p - mean).abs() <= 5.0 * sigma + 1.0 / (permutations + 1.0),
            "p {p} vs exact {exact} (sigma {sigma})"
        );
    }
}

#[test]
fn kernel_statistic_and_permutation_p_match_exact_enumeration() {
    let permutations = 4000usize;
    for case in fixture()["mmd"].as_array().unwrap() {
        let r = kernel_two_sample_with_permutations(
            &vec_f(&case["a"]),
            &vec_f(&case["b"]),
            5,
            permutations,
        )
        .unwrap();
        close(r.statistic, f(&case["mmd2"]), 1e-12, 1e-15, "mmd2");
        let b = permutations as f64;
        let exact = f(&case["p_exact"]);
        let mean = (1.0 + b * exact) / (b + 1.0);
        let sigma = (exact * (1.0 - exact) / b).sqrt();
        assert!(
            (r.p_value - mean).abs() <= 5.0 * sigma + 1.0 / (b + 1.0),
            "p {} vs exact {exact} (sigma {sigma})",
            r.p_value
        );
    }
}
