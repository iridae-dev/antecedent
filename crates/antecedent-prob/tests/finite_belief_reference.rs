//! The aligned finite Bayesian update under a known-variance Gaussian experimental-mean
//! likelihood against 80-digit evaluation of the stated model
//! (`oracle_data/finite_belief_reference.py`), and the linear functional of aligned rows.
//!
//! Covers negative and mixed-sign affine slopes, intercepts, a zero-mass slot, a posterior
//! pushed 30 standard errors into one tail, a sample of 1e8 and a variance of 1e-9, forty seeded
//! random slot sets, and the refusals of the contract.

use antecedent_prob::finite_belief::{AffineGaussianExperiment, condition, linear_functional_rows};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/finite_belief_reference.json"))
        .expect("frozen reference")
}

fn floats(value: &Value) -> Vec<f64> {
    value.as_array().expect("array").iter().map(|v| v.as_f64().expect("number")).collect()
}

fn text_floats(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("array")
        .iter()
        .map(|v| v.as_str().expect("text").parse().expect("number"))
        .collect()
}

#[test]
fn the_update_equals_the_normalised_gaussian_likelihood_times_prior() {
    let reference = reference();
    let cases = reference["conditioning"].as_array().expect("cases");
    assert!(cases.len() >= 40);
    for case in cases {
        let (targets, intercepts, slopes) =
            (floats(&case["targets"]), floats(&case["intercepts"]), floats(&case["slopes"]));
        let experiment = AffineGaussianExperiment {
            observed_mean: case["observed_mean"].as_f64().expect("mean"),
            observation_variance: case["variance"].as_f64().expect("variance"),
            sample_size: case["n"].as_u64().expect("n"),
            target_values: &targets,
            intercepts: &intercepts,
            slopes: &slopes,
        };
        let got = experiment.condition(&floats(&case["prior"])).expect("update");
        let want = text_floats(&case["posterior"]);
        assert_eq!(got.len(), want.len());
        assert!((got.iter().sum::<f64>() - 1.0).abs() < 1e-14, "{}: mass", case["name"]);
        for (slot, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-12 * w.abs() + 1e-300,
                "{} slot {slot}: {g} vs {w}",
                case["name"]
            );
        }
    }
}

#[test]
fn linear_functional_equals_the_exact_row_dot_product() {
    let reference = reference();
    for case in reference["functional"].as_array().expect("cases") {
        let got = linear_functional_rows(&floats(&case["rows"]), &floats(&case["coefficients"]))
            .expect("functional");
        let want = text_floats(&case["values"]);
        for (g, w) in got.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-12 * w.abs().max(1.0), "{g} vs {w}");
        }
    }
}

fn refused<T: std::fmt::Debug, E: std::fmt::Display>(result: Result<T, E>, needle: &str) {
    let message = result.expect_err("a refusal").to_string();
    assert!(message.contains(needle), "{message:?} does not name {needle:?}");
}

#[test]
fn malformed_updates_are_refused_and_never_reweighted() {
    let aligned = "invalid aligned finite belief/update";
    let zero_mass = "zero mass under every retained state";
    refused(condition(&[], &[]), aligned);
    refused(condition(&[0.5, 0.5], &[0.0]), aligned);
    refused(condition(&[0.5, -0.1], &[0.0, 0.0]), aligned);
    refused(condition(&[0.5, f64::NAN], &[0.0, 0.0]), aligned);
    refused(condition(&[0.5, 0.5], &[f64::NAN, 0.0]), aligned);
    refused(condition(&[0.5, 0.5], &[f64::INFINITY, 0.0]), aligned);
    refused(condition(&[0.5, 0.5], &[f64::NEG_INFINITY, f64::NEG_INFINITY]), zero_mass);
    refused(condition(&[0.0, 0.0], &[0.0, 0.0]), zero_mass);
    // A slot of zero likelihood is dropped without disturbing the rest.
    let got = condition(&[0.25, 0.75], &[f64::NEG_INFINITY, -3.0]).expect("update");
    assert_eq!(got, vec![0.0, 1.0]);
    let base = AffineGaussianExperiment {
        observed_mean: 0.0,
        observation_variance: 1.0,
        sample_size: 1,
        target_values: &[0.0, 1.0],
        intercepts: &[0.0, 0.0],
        slopes: &[1.0, 1.0],
    };
    assert!(base.condition(&[0.5, 0.5]).is_ok());
    let paired = "invalid paired affine Gaussian experiment";
    refused(base.condition(&[1.0]), paired);
    for broken in [
        AffineGaussianExperiment { sample_size: 0, ..base },
        AffineGaussianExperiment { observation_variance: 0.0, ..base },
        AffineGaussianExperiment { observation_variance: -1.0, ..base },
        AffineGaussianExperiment { observation_variance: f64::INFINITY, ..base },
        AffineGaussianExperiment { observed_mean: f64::NAN, ..base },
        AffineGaussianExperiment { slopes: &[1.0, 0.0], ..base },
        AffineGaussianExperiment { intercepts: &[0.0, f64::NAN], ..base },
        AffineGaussianExperiment { target_values: &[0.0], ..base },
    ] {
        refused(broken.condition(&[0.5, 0.5]), paired);
    }
    refused(
        AffineGaussianExperiment { observation_variance: 5e-324, sample_size: 3, ..base }
            .condition(&[0.5, 0.5]),
        "variance underflow/overflow",
    );
    refused(linear_functional_rows(&[1.0, 2.0, 3.0], &[1.0, 1.0]), "invalid finite-law functional axes");
    refused(linear_functional_rows(&[1.0, 2.0], &[]), "invalid finite-law functional axes");
    refused(linear_functional_rows(&[f64::MAX, f64::MAX], &[2.0, 2.0]), "functional overflow");
}
