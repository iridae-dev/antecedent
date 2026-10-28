//! The point-only posterior means against an exact rational reference.
//!
//! `oracle_data/posterior_mean_reference.py` computes each posterior mean in `fractions.Fraction`
//! arithmetic (normal-normal update of a sample or contrast mean, known-variance update, covariance
//! entries, Dirichlet and supported-Dirichlet means, and the ridge normal equations solved by exact
//! Gauss-Jordan elimination) and freezes the binary64 roundings in the JSON beside it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::posterior_mean::{
    DirichletNamespace, PosteriorRequest, ScalarNamespace, ScoreNamespace,
    covariance_entry_posterior, dirichlet_posterior_mean, dirichlet_supported_mean,
    known_variance_posterior, normal_mean_posterior, score_contrast_posterior,
};
use antecedent_estimate::vector_treatment::{DesignSummary, full_coefficient_posterior_mean};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("oracle_data/posterior_mean_reference.json")).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn num(v: &Value, key: &str) -> f64 {
    v[key].as_f64().unwrap_or_else(|| panic!("{key}"))
}

fn normal(scale: f64) -> PosteriorRequest {
    PosteriorRequest::IndependentNormal { scale }
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    let tol = rel * want.abs().max(1e-300);
    assert!((got - want).abs() <= tol, "{what}: got {got} want {want} tol {tol}");
}

#[test]
fn normal_mean_and_score_contrast_equal_the_exact_conjugate_update() {
    let fx = fixture();
    for case in fx["normal_mean"].as_array().unwrap() {
        let got = normal_mean_posterior(
            &floats(&case["values"]),
            normal(num(case, "scale")),
            ScalarNamespace::Refresh,
        )
        .unwrap();
        close(got, num(case, "want"), 1e-11, "normal_mean");
    }
    for case in fx["score_contrast"].as_array().unwrap() {
        let got = score_contrast_posterior(
            &floats(&case["control"]),
            &floats(&case["treated"]),
            normal(num(case, "scale")),
            ScoreNamespace::Penalized,
        )
        .unwrap();
        close(got, num(case, "want"), 1e-9, "score_contrast");
    }
}

#[test]
fn known_variance_and_covariance_entries_equal_the_exact_update() {
    let fx = fixture();
    for case in fx["known_variance"].as_array().unwrap() {
        let got = known_variance_posterior(
            num(case, "point"),
            num(case, "variance"),
            normal(num(case, "scale")),
            ScalarNamespace::Constancy,
        )
        .unwrap();
        close(got, num(case, "want"), 1e-14, "known_variance");
    }
    for case in fx["covariance"].as_array().unwrap() {
        let want = floats(&case["want"]);
        let got = covariance_entry_posterior(
            &floats(&case["entries"]),
            &floats(&case["variances"]),
            normal(num(case, "scale")),
        )
        .unwrap();
        for (g, w) in got.iter().zip(&want) {
            close(*g, *w, 1e-14, "covariance entry");
        }
    }
}

#[test]
fn dirichlet_means_equal_the_exact_pseudo_count_update_and_keep_structural_zeros() {
    let fx = fixture();
    for case in fx["dirichlet"].as_array().unwrap() {
        let got = dirichlet_posterior_mean(
            &floats(&case["weights"]),
            normal(num(case, "concentration")),
            DirichletNamespace::Decision,
        )
        .unwrap();
        for (g, w) in got.iter().zip(floats(&case["want"])) {
            close(*g, w, 1e-14, "dirichlet");
        }
        assert!((got.iter().sum::<f64>() - 1.0).abs() < 1e-14);
    }
    for case in fx["dirichlet_supported"].as_array().unwrap() {
        let probs = floats(&case["probabilities"]);
        let got = dirichlet_supported_mean(
            &probs,
            normal(num(case, "concentration")),
            DirichletNamespace::Msm,
        )
        .unwrap();
        for ((g, w), p) in got.iter().zip(floats(&case["want"])).zip(&probs) {
            if *p == 0.0 {
                assert_eq!(*g, 0.0);
            } else {
                close(*g, w, 1e-14, "dirichlet_supported");
            }
        }
    }
}

#[test]
fn ridge_coefficient_posterior_equals_the_exact_penalized_normal_equations() {
    for case in fixture()["ridge"].as_array().unwrap() {
        let xty = floats(&case["xty"]);
        let summary = DesignSummary {
            n_rows: case["n_rows"].as_u64().unwrap() as usize,
            first_treatment: 1,
            gram: floats(&case["gram"]),
            xty,
            rss: num(case, "rss"),
        };
        let got = full_coefficient_posterior_mean(&summary, num(case, "scale")).unwrap();
        let want = floats(&case["want"]);
        let scale = want.iter().fold(0.0_f64, |m, w| m.max(w.abs()));
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-9 * scale.max(1.0), "ridge: got {g} want {w}");
        }
    }
}

#[test]
fn the_updates_have_the_prior_and_flat_limits_and_refuse_invalid_scales() {
    let values = [1.0, 2.0, 4.0, 8.0];
    let mean = 3.75;
    let wide = normal_mean_posterior(&values, normal(1e9), ScalarNamespace::Refresh).unwrap();
    close(wide, mean, 1e-12, "flat prior limit");
    let tight = normal_mean_posterior(&values, normal(1e-9), ScalarNamespace::Refresh).unwrap();
    assert!(tight.abs() < 1e-8, "{tight}");
    for scale in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(normal_mean_posterior(&values, normal(scale), ScalarNamespace::Refresh).is_err());
        assert!(
            dirichlet_posterior_mean(&[1.0, 2.0], normal(scale), DirichletNamespace::Decision)
                .is_err()
        );
    }
    assert!(known_variance_posterior(1.0, 0.0, normal(1.0), ScalarNamespace::Constancy).is_err());
    assert!(
        dirichlet_posterior_mean(&[1.0, -0.5], normal(1.0), DirichletNamespace::Decision).is_err()
    );
    assert!(normal_mean_posterior(&[2.0, 2.0, 2.0], normal(1.0), ScalarNamespace::Refresh).is_err());
}
