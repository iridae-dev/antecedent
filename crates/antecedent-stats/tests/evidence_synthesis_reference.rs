//! Summary-data synthesis against 50-digit evaluation of the models
//! (`oracle_data/evidence_synthesis_reference.py`).
//!
//! GLS pooling with correlated, near-singular and widely scaled covariances; DerSimonian-Laird and
//! restricted-maximum-likelihood random effects (the reference takes the REML stationary point from
//! the derivative of the restricted likelihood itself); hierarchical borrowing by direct numerical
//! integration over the common mean; and the equicorrelation closed form and its envelope over a
//! correlation range, with correlations at 0.95 of the way to the positive-definite limit and at
//! 1 - 1e-6.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::synthesis::{
    MeanPrior, TauPrior, equicorrelated_gls, fixed_effect_gls, fixed_effect_inverse_variance,
    hierarchical_normal_borrowing, pooled_dependence_envelope, random_effects_dersimonian_laird,
    random_effects_reml,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/evidence_synthesis_reference.json"))
        .expect("frozen reference")
}

fn number(value: &Value) -> f64 {
    match value {
        Value::String(text) => text.parse().expect("number"),
        other => other.as_f64().expect("number"),
    }
}

fn numbers(value: &Value) -> Vec<f64> {
    value.as_array().expect("array").iter().map(number).collect()
}

fn close(got: f64, want: f64, relative: f64, what: &str) {
    assert!((got - want).abs() <= 1e-300 + relative * want.abs(), "{what}: {got} vs {want}");
}

fn close_all(got: &[f64], want: &[f64], relative: f64, floor: f64, what: &str) {
    assert_eq!(got.len(), want.len(), "{what}");
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() <= floor + relative * w.abs(), "{what}: {g} vs {w}");
    }
}

#[test]
fn gls_pooling_matches_the_explicit_covariance_solve() {
    let reference = reference();
    for case in reference["gls"].as_array().expect("gls") {
        let y = numbers(&case["estimates"]);
        let covariance = numbers(&case["covariance"]);
        let got = match fixed_effect_gls(&y, &covariance) {
            Ok(fit) => fit,
            Err(e) => panic!("{case}: {e}"),
        };
        let tolerance = if covariance.iter().any(|c| *c == 1e8) { 1e-7 } else { 1e-9 };
        close(got.estimate, number(&case["estimate"]), tolerance, &format!("estimate {case}"));
        close(got.variance, number(&case["variance"]), tolerance, &format!("variance {case}"));
        close_all(
            &got.weights,
            &numbers(&case["weights"]),
            tolerance,
            1e-9,
            &format!("weights {case}"),
        );
        close(got.heterogeneity_q, number(&case["q"]), 1e-6, &format!("q {case}"));
    }
}

#[test]
fn random_effects_match_the_moment_and_likelihood_references() {
    let reference = reference();
    for case in reference["random_effects"].as_array().expect("random_effects") {
        let (y, v) = (numbers(&case["estimates"]), numbers(&case["variances"]));
        let fixed = fixed_effect_inverse_variance(&y, &v).expect("fixed effect");
        close(fixed.heterogeneity_q, number(&case["q"]), 1e-10, &format!("q {case}"));
        for (name, fit) in [
            ("dl", random_effects_dersimonian_laird(&y, &v).expect("dl")),
            ("reml", random_effects_reml(&y, &v).expect("reml")),
        ] {
            let want = &case[name];
            let what = format!("{name} {case}");
            let tau = number(&want["tau2"]);
            assert!(
                (fit.tau_squared - tau).abs() <= 1e-12 * (1.0 + tau.abs()) + 1e-10 * tau,
                "tau2 {what}: {} vs {tau}",
                fit.tau_squared
            );
            close(fit.estimate, number(&want["estimate"]), 1e-9, &format!("estimate {what}"));
            close(fit.variance, number(&want["variance"]), 1e-9, &format!("variance {what}"));
            close_all(
                &fit.weights,
                &numbers(&want["weights"]),
                1e-9,
                1e-12,
                &format!("weights {what}"),
            );
            close(fit.i_squared, number(&case["i_squared"]), 1e-10, &format!("i2 {what}"));
        }
    }
}

#[test]
fn hierarchical_borrowing_matches_direct_integration_over_the_common_mean() {
    let reference = reference();
    for case in reference["hierarchical"].as_array().expect("hierarchical") {
        let mean_prior = match case["mean_prior"].as_array() {
            None => MeanPrior::Flat,
            Some(p) => MeanPrior::Normal { mean: number(&p[0]), variance: number(&p[1]) },
        };
        let prior = TauPrior {
            support: numbers(&case["support"]),
            probabilities: numbers(&case["probabilities"]),
        };
        let got = hierarchical_normal_borrowing(
            &numbers(&case["estimates"]),
            &numbers(&case["variances"]),
            mean_prior,
            &prior,
        )
        .expect("borrowing");
        let what = format!("{case}");
        close(got.mean, number(&case["mean"]), 1e-9, &format!("mean {what}"));
        close(got.variance, number(&case["variance"]), 1e-8, &format!("variance {what}"));
        close_all(
            &got.study_means,
            &numbers(&case["study_means"]),
            1e-9,
            1e-12,
            &format!("study means {what}"),
        );
        close_all(
            &got.study_variances,
            &numbers(&case["study_variances"]),
            1e-7,
            1e-12,
            &format!("study variances {what}"),
        );
        close_all(
            &got.tau_posterior,
            &numbers(&case["tau_posterior"]),
            1e-9,
            1e-12,
            &format!("tau posterior {what}"),
        );
    }
}

#[test]
fn equicorrelated_pooling_matches_the_explicit_covariance_solve() {
    let reference = reference();
    for case in reference["dependence"]["points"].as_array().expect("points") {
        let (estimate, variance) = equicorrelated_gls(
            &numbers(&case["estimates"]),
            &numbers(&case["se"]),
            case["rho"].as_f64().expect("rho"),
        )
        .expect("equicorrelated pooling");
        close(estimate, number(&case["estimate"]), 1e-9, &format!("estimate {case}"));
        close(variance, number(&case["variance"]), 1e-9, &format!("variance {case}"));
    }
}

#[test]
fn the_dependence_envelope_attains_the_extremes_of_the_reference() {
    let reference = reference();
    for case in reference["dependence"]["envelopes"].as_array().expect("envelopes") {
        let envelope = pooled_dependence_envelope(
            &numbers(&case["estimates"]),
            &numbers(&case["se"]),
            case["lo"].as_f64().expect("lo"),
            case["hi"].as_f64().expect("hi"),
        )
        .expect("envelope");
        let what = format!("{case}");
        close(
            envelope.estimate_min,
            number(&case["estimate"]["min"]),
            1e-9,
            &format!("estimate min {what}"),
        );
        close(
            envelope.estimate_max,
            number(&case["estimate"]["max"]),
            1e-9,
            &format!("estimate max {what}"),
        );
        close(
            envelope.variance_min,
            number(&case["variance"]["min"]),
            1e-9,
            &format!("variance min {what}"),
        );
        close(
            envelope.variance_max,
            number(&case["variance"]["max"]),
            1e-9,
            &format!("variance max {what}"),
        );
    }
}

#[test]
fn singularity_is_judged_on_correlations_not_on_the_variance_ratio() {
    use antecedent_stats::synthesis::SynthesisError;
    // Independent studies whose variances differ by sixteen orders of magnitude are well
    // conditioned once scaled; collinear studies are singular at any scale.
    let wide = [1e-8, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1e8];
    let fit = fixed_effect_gls(&[1.0, 2.0, 3.0], &wide).expect("diagonal covariance");
    close(fit.estimate, 1.0, 1e-7, "the precise study dominates");
    for scale in [1.0, 1e-9, 1e9] {
        let collinear = [scale, scale, scale, scale * (1.0 + 1e-13)];
        assert!(matches!(
            fixed_effect_gls(&[1.0, 2.0], &collinear),
            Err(SynthesisError::Singular { .. })
        ));
        let collinear_scaled = [scale, 3.0 * scale, 3.0 * scale, 9.0 * scale * (1.0 + 1e-13)];
        assert!(matches!(
            fixed_effect_gls(&[1.0, 2.0], &collinear_scaled),
            Err(SynthesisError::Singular { .. })
        ));
    }
}
