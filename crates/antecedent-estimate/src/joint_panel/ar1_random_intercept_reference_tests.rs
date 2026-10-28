// SPDX-License-Identifier: MIT OR Apache-2.0
//! The closed-form stationary AR(1) plus random-intercept marginal law against a
//! dense-covariance reference.
//!
//! `tests/oracle_data/joint_panel_ar1_reference.py` (frozen in the JSON beside it) builds the
//! covariance `S = Cov(AR(1)) + tau^2 11'` of a group's residuals and returns the marginal
//! log-likelihood and the conditional mean and variance of the intercept by dense 120-digit
//! linear algebra, without the precision-matrix and rank-one-update algebra the product uses.
//! Cases cover a single observation, strong positive and negative serial correlation, tiny and
//! huge intercept and innovation scales, residuals offset by 1e6 and one long group.

use super::likelihood::ar1_random_intercept;
use serde_json::Value;

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!(
        "../../tests/oracle_data/joint_panel_ar1_reference.json"
    ))
    .unwrap()
    .as_array()
    .unwrap()
    .clone()
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

fn close(got: f64, want: f64, rel: f64, abs: f64) -> bool {
    (got - want).abs() <= abs + rel * want.abs()
}

#[test]
fn closed_form_matches_the_dense_covariance_reference() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let r: Vec<f64> = case["r"].as_array().unwrap().iter().map(f).collect();
        let got = ar1_random_intercept(&r, f(&case["rho"]), f(&case["sigma"]), f(&case["tau"]))
            .unwrap_or_else(|| panic!("{name}: the product refused a valid group"));
        let (ll, m, v) = (
            f(&case["log_likelihood"]),
            f(&case["intercept_mean"]),
            f(&case["intercept_variance"]),
        );
        eprintln!(
            "{name}: ll {:+.3e} (rel {:.1e}) mean {:+.3e} var {:+.3e}",
            got.log_likelihood - ll,
            (got.log_likelihood - ll).abs() / ll.abs().max(1.0),
            got.intercept_mean - m,
            got.intercept_variance - v
        );
        // The log-likelihood is a sum of O(n) terms of size up to |ll|; a few hundred ulps of
        // that magnitude is the honest f64 envelope (the offset case subtracts 1e12-sized terms).
        assert!(
            close(got.log_likelihood, ll, 1e-11, 1e-11),
            "{name} ll: {} vs {ll}",
            got.log_likelihood
        );
        assert!(
            close(got.intercept_mean, m, 1e-9, 1e-12),
            "{name} mean: {} vs {m}",
            got.intercept_mean
        );
        assert!(
            close(got.intercept_variance, v, 1e-9, 1e-14),
            "{name} var: {} vs {v}",
            got.intercept_variance
        );
    }
}

#[test]
fn invalid_parameters_and_residuals_are_refused_not_evaluated() {
    let r = [0.1, -0.2, 0.3];
    assert!(ar1_random_intercept(&r, 0.4, 0.2, 0.3).is_some());
    for (rho, sigma, tau) in [
        (1.0, 0.2, 0.3),
        (-1.0, 0.2, 0.3),
        (1.5, 0.2, 0.3),
        (f64::NAN, 0.2, 0.3),
        (0.4, 0.0, 0.3),
        (0.4, -0.2, 0.3),
        (0.4, 0.2, 0.0),
        (0.4, 0.2, f64::INFINITY),
    ] {
        assert!(ar1_random_intercept(&r, rho, sigma, tau).is_none(), "{rho} {sigma} {tau}");
    }
    assert!(ar1_random_intercept(&[], 0.4, 0.2, 0.3).is_none());
    assert!(ar1_random_intercept(&[0.1, f64::NAN], 0.4, 0.2, 0.3).is_none());
}

#[test]
fn the_likelihood_is_time_reversible_but_not_permutation_invariant() {
    // reversing a stationary AR(1) series leaves its law unchanged (time reversibility)
    let r = [0.3, -0.1, 0.8, 0.2, -0.5];
    let rev: Vec<f64> = r.iter().rev().copied().collect();
    let a = ar1_random_intercept(&r, 0.6, 0.3, 0.4).unwrap();
    let b = ar1_random_intercept(&rev, 0.6, 0.3, 0.4).unwrap();
    assert!((a.log_likelihood - b.log_likelihood).abs() < 1e-12);
    assert!((a.intercept_mean - b.intercept_mean).abs() < 1e-12);
    // a shuffle that is not a reversal changes it
    let shuffled = [0.8, 0.3, -0.5, -0.1, 0.2];
    let c = ar1_random_intercept(&shuffled, 0.6, 0.3, 0.4).unwrap();
    assert!((a.log_likelihood - c.log_likelihood).abs() > 1e-6);
}
