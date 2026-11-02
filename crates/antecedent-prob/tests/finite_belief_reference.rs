//! The aligned finite Bayesian update under a known-variance Gaussian experimental-mean
//! likelihood against 80-digit evaluation of the stated model
//! (`oracle_data/finite_belief_reference.py`), and the linear functional of aligned rows.
//!
//! Covers negative and mixed-sign affine slopes, intercepts, a zero-mass slot, a posterior
//! pushed 30 standard errors into one tail, a sample of 1e8 and a variance of 1e-9, forty seeded
//! random slot sets, and the refusals of the contract.

use antecedent_prob::finite_belief::{
    AffineGaussianExperiment, JointAffineGaussianExperiment, condition, linear_functional_rows,
};
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
    refused(
        linear_functional_rows(&[1.0, 2.0, 3.0], &[1.0, 1.0]),
        "invalid finite-law functional axes",
    );
    refused(linear_functional_rows(&[1.0, 2.0], &[]), "invalid finite-law functional axes");
    refused(linear_functional_rows(&[f64::MAX, f64::MAX], &[2.0, 2.0]), "functional overflow");
}

/// Independent estimators reduce to the product of the marginal likelihoods the scalar update
/// already owns; dependence moves the posterior to the closed-form bivariate law.
#[test]
fn joint_update_is_the_marginal_product_when_independent_and_the_bivariate_law_otherwise() {
    let targets = [vec![0.2, 0.9, -0.4, 0.5], vec![1.0, 0.1, 0.8, -0.3]];
    let intercepts = [vec![0.0, 0.1, 0.0, -0.1], vec![0.2, 0.0, 0.1, 0.0]];
    let slopes = [vec![1.0, 1.2, 0.8, -1.1], vec![0.9, 1.0, 1.3, 1.0]];
    let observed = [0.4, 0.7];
    let prior = [0.1, 0.2, 0.3, 0.4];
    let joint = |covariance: &[f64]| {
        JointAffineGaussianExperiment {
            observed_means: &observed,
            sampling_covariance: covariance,
            target_values: &targets,
            intercepts: &intercepts,
            slopes: &slopes,
        }
        .condition(&prior)
        .expect("joint update")
    };
    // Independent: condition on component 0, then on component 1, with the scalar owner.
    let scalar = |c: usize, belief: &[f64]| {
        AffineGaussianExperiment {
            observed_mean: observed[c],
            observation_variance: [0.04, 0.09][c],
            sample_size: 1,
            target_values: &targets[c],
            intercepts: &intercepts[c],
            slopes: &slopes[c],
        }
        .condition(belief)
        .expect("scalar update")
    };
    let sequential = scalar(1, &scalar(0, &prior));
    for (a, b) in joint(&[0.04, 0.0, 0.0, 0.09]).iter().zip(&sequential) {
        assert!((a - b).abs() < 1e-12, "{a} vs {b}");
    }
    // Correlated: closed-form bivariate quadratic form, normalized.
    let (v1, c, v2) = (0.04, 0.03, 0.09);
    let det = v1 * v2 - c * c;
    let log_weights = (0..4)
        .map(|s| {
            let r1 = observed[0] - (targets[0][s] - intercepts[0][s]) / slopes[0][s];
            let r2 = observed[1] - (targets[1][s] - intercepts[1][s]) / slopes[1][s];
            f64::ln(prior[s]) - 0.5 * (v2 * r1 * r1 - 2.0 * c * r1 * r2 + v1 * r2 * r2) / det
        })
        .collect::<Vec<_>>();
    let top = log_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mass = log_weights.iter().map(|w| (w - top).exp()).sum::<f64>();
    let correlated = joint(&[v1, c, c, v2]);
    for (actual, log_weight) in correlated.iter().zip(&log_weights) {
        assert!((actual - (log_weight - top).exp() / mass).abs() < 1e-12);
    }
    let moved = correlated.iter().zip(&sequential).map(|(a, b)| (a - b).abs()).sum::<f64>();
    assert!(moved > 1e-3, "dependence must change the posterior, moved {moved}");
}

#[test]
fn joint_update_refuses_what_is_not_a_joint_sampling_law() {
    let (targets, intercepts, slopes) = (
        [vec![0.0, 1.0], vec![1.0, 0.0]],
        [vec![0.0, 0.0], vec![0.0, 0.0]],
        [vec![1.0, 1.0], vec![1.0, 1.0]],
    );
    let base = JointAffineGaussianExperiment {
        observed_means: &[0.5, 0.5],
        sampling_covariance: &[1.0, 0.5, 0.5, 1.0],
        target_values: &targets,
        intercepts: &intercepts,
        slopes: &slopes,
    };
    assert!(base.condition(&[0.5, 0.5]).is_ok());
    for broken in [
        JointAffineGaussianExperiment { sampling_covariance: &[1.0, 1.0, 1.0, 1.0], ..base },
        JointAffineGaussianExperiment { sampling_covariance: &[1.0, 0.9, 0.1, 1.0], ..base },
        JointAffineGaussianExperiment { sampling_covariance: &[1.0, 0.5, 0.5], ..base },
        JointAffineGaussianExperiment {
            sampling_covariance: &[1.0, f64::NAN, f64::NAN, 1.0],
            ..base
        },
        JointAffineGaussianExperiment { observed_means: &[0.5], ..base },
        JointAffineGaussianExperiment { slopes: &[vec![1.0, 0.0], vec![1.0, 1.0]], ..base },
    ] {
        assert!(broken.condition(&[0.5, 0.5]).is_err());
    }
    assert!(base.condition(&[0.5]).is_err());
}
