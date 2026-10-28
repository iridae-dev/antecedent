//! Ridge, lasso and elastic-net fits against exact and multiprecision references.
//!
//! `oracle_data/regularized_reference.py` solves ridge in rational arithmetic and the elastic net
//! by enumerating every active set and sign pattern with Karush-Kuhn-Tucker verification in
//! 80-digit arithmetic, so no iterative solver is shared with the implementation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{
    LassoFit, LassoOptions, RidgeIntercept, fit_lasso, fit_ridge_with, predict_lasso, soft_threshold,
};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("oracle_data/regularized_reference.json")).unwrap()
}

fn f(v: &Value) -> f64 {
    v.as_f64().unwrap()
}

fn vec_f(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(f).collect()
}

fn cols(v: &Value) -> Vec<Vec<f64>> {
    v.as_array().unwrap().iter().map(vec_f).collect()
}

fn us(v: &Value) -> usize {
    v.as_u64().unwrap() as usize
}

fn flatten(cols: &[Vec<f64>]) -> Vec<f64> {
    cols.iter().flatten().copied().collect()
}

#[test]
fn ridge_matches_the_rational_normal_equations() {
    for (k, case) in fixture()["ridge"].as_array().unwrap().iter().enumerate() {
        let intercept = if case["unpenalized_leading"].as_bool().unwrap() {
            RidgeIntercept::LeadingColumn
        } else {
            RidgeIntercept::None
        };
        let (n, p) = (us(&case["n"]), us(&case["p"]));
        let fit = fit_ridge_with(
            &flatten(&cols(&case["cols"])),
            n,
            p,
            &vec_f(&case["y"]),
            f(&case["lambda"]),
            intercept,
        )
        .unwrap();
        // The nearly collinear case (columns correlated to 1 - 1e-6, lambda 1e-6) has
        // coefficients near +-340 that amplify rounding in X'X by its condition number.
        let rel = if k == 2 { 1e-6 } else { 1e-11 };
        for (j, (got, want)) in fit.coefficients.iter().zip(vec_f(&case["coef"])).enumerate() {
            assert!(
                (got - want).abs() <= rel * want.abs().max(1.0),
                "ridge case {k} coef {j}: {got} vs {want}"
            );
        }
    }
}

#[test]
fn elastic_net_converged_to_tight_tolerance_matches_the_exact_active_set_solution() {
    for (k, case) in fixture()["enet"].as_array().unwrap().iter().enumerate() {
        let (n, p) = (us(&case["n"]), us(&case["p"]));
        let columns = cols(&case["cols"]);
        let want = vec_f(&case["coef"]);
        let options = LassoOptions {
            lambda: f(&case["lambda"]),
            l1_ratio: f(&case["l1_ratio"]),
            fit_intercept: case["fit_intercept"].as_bool().unwrap(),
            standardize: case["standardize"].as_bool().unwrap(),
            max_iter: 200_000,
            tol: 1e-13,
        };
        let fit = fit_lasso(&flatten(&columns), n, p, &vec_f(&case["y"]), &options).unwrap();
        assert!(fit.converged, "enet case {k} did not converge");
        // Columns span two orders of magnitude, so compare in fitted-value units:
        // |dbeta| times the column's root mean square.
        for j in 0..p {
            let rms = (columns[j].iter().map(|v| v * v).sum::<f64>() / n as f64).sqrt();
            let err = (fit.coefficients[j] - want[j]).abs() * rms;
            assert!(err <= 1e-9, "enet case {k} coef {j}: {} vs {}", fit.coefficients[j], want[j]);
            // A coordinate the exact solution zeroes is exactly zero (soft thresholding).
            if want[j] == 0.0 {
                assert_eq!(fit.coefficients[j], 0.0, "enet case {k} coef {j} should be zero");
            }
        }
        let a0 = f(&case["intercept"]);
        assert!((fit.intercept - a0).abs() <= 1e-8 * a0.abs().max(1.0), "enet case {k} intercept");
    }
}

#[test]
fn default_tolerance_holds_fitted_values_near_the_exact_solution_on_correlated_designs() {
    for (k, case) in fixture()["correlated"].as_array().unwrap().iter().enumerate() {
        let (n, p) = (us(&case["n"]), us(&case["p"]));
        let options = LassoOptions { lambda: f(&case["lambda"]), ..LassoOptions::default() };
        let x = flatten(&cols(&case["cols"]));
        let y = vec_f(&case["y"]);
        let fit = fit_lasso(&x, n, p, &y, &options).unwrap();
        assert!(fit.converged, "correlated case {k}");
        let got = predict_lasso(&fit, &x, n, p).unwrap();
        let exact = LassoFit {
            intercept: f(&case["intercept"]),
            coefficients: vec_f(&case["coef"]),
            ..fit.clone()
        };
        let want = predict_lasso(&exact, &x, n, p).unwrap();
        let ybar = y.iter().sum::<f64>() / n as f64;
        let scale = y.iter().map(|v| (v - ybar).powi(2)).sum::<f64>().sqrt();
        let gap = got.iter().zip(&want).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
        // The tolerance bounds how far one sweep moves the fit, not the distance to the
        // minimizer; with predictors correlated at 0.99 the error is a few multiples of it.
        assert!(gap <= 20.0 * options.tol * scale, "case {k}: fitted-value error {}", gap / scale);
    }
}

#[test]
fn soft_threshold_is_the_proximal_operator_of_the_l1_norm() {
    for &(z, g, want) in &[(3.0, 1.0, 2.0), (-3.0, 1.0, -2.0), (0.5, 1.0, 0.0), (-1.0, 1.0, 0.0), (1.0, 0.0, 1.0)] {
        assert_eq!(soft_threshold(z, g), want);
    }
}
