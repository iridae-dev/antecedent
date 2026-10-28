//! Latent-class (finite mixture of Gaussian regressions) fit against an independent 40-digit EM.
//!
//! `oracle_data/latent_class_reference.py` (frozen in the JSON beside it) does not reuse the
//! product's initialisation: it starts EM from random hard labellings of its own generator, keeps
//! the best of 24 starts and polishes it at 40 digits to a stationary point, then evaluates the
//! observed-data log-likelihood, the BIC, the responsibilities, the effective class sizes, the
//! separation and the mixture-average effect there, with the classes in canonical order. The test
//! requires the product's selected fit to be that stationary point (parameters, likelihood and
//! every posterior responsibility) on four well-separated designs: a binary treatment with one
//! covariate, three classes with a continuous treatment and two covariates, unbalanced weights at
//! an outcome offset of 1e4 and a covariate offset of 100, and no covariates. It also checks the
//! likelihood trace is non-decreasing and that the declared-premise refusals fire. Global
//! optimality is not proved; the bootstrap standard errors are not reviewed here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::latent_class_effects::{
    LatentClassConfig, LatentClassData, fit_latent_class_effects,
};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-3), "{what}: got {got} want {want}");
}

fn config(case: &Value) -> LatentClassConfig {
    let mut config = LatentClassConfig::new(
        case["classes"].as_u64().unwrap() as usize,
        case["seed"].as_u64().unwrap(),
    )
    .declare_conditional_randomization();
    config.bootstrap_replicates = 0;
    config.tolerance = 1e-14;
    config.max_iterations = 20_000;
    config
}

#[test]
fn the_selected_fit_is_the_independent_stationary_point() {
    let reference: Value =
        serde_json::from_str(include_str!("oracle_data/latent_class_reference.json")).unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let want = &case["expected"];
        let covariates: Vec<Vec<f64>> =
            case["covariates"].as_array().unwrap().iter().map(floats).collect();
        let outcome = floats(&case["y"]);
        let treatment = floats(&case["treatment"]);
        let result = fit_latent_class_effects(
            &LatentClassData { outcome: &outcome, treatment: &treatment, covariates: &covariates },
            &config(case),
        )
        .unwrap();
        let (weights, sigma2) = (floats(&want["weights"]), floats(&want["sigma2"]));
        let effective = floats(&want["effective_n"]);
        for (k, class) in result.classes.iter().enumerate() {
            let beta = floats(&want["beta"][k]);
            close(class.weight, weights[k], 1e-6, &format!("{name} weight {k}"));
            close(class.intercept, beta[0], 1e-6, &format!("{name} intercept {k}"));
            close(class.effect, beta[1], 1e-6, &format!("{name} effect {k}"));
            for (g, w) in class.covariate_coefficients.iter().zip(&beta[2..]) {
                close(*g, *w, 1e-6, &format!("{name} gamma {k}"));
            }
            close(class.residual_variance, sigma2[k], 1e-6, &format!("{name} variance {k}"));
            close(class.effective_n, effective[k], 1e-6, &format!("{name} size {k}"));
        }
        close(result.log_likelihood, want["log_likelihood"].as_f64().unwrap(), 1e-9, name);
        close(result.bic, want["bic"].as_f64().unwrap(), 1e-9, name);
        close(result.separation, want["separation"].as_f64().unwrap(), 1e-7, name);
        close(
            result.mixture_average_effect,
            want["mixture_average_effect"].as_f64().unwrap(),
            1e-6,
            name,
        );
        for (i, row) in want["responsibilities"].as_array().unwrap().iter().enumerate() {
            for (k, w) in floats(row).into_iter().enumerate() {
                assert!(
                    (result.responsibilities[i][k] - w).abs() <= 1e-6,
                    "{name} responsibility [{i}][{k}]"
                );
            }
        }
        assert!(
            result
                .log_likelihood_trace
                .windows(2)
                .all(|w| w[1] >= w[0] - 1e-8 * (1.0 + w[0].abs())),
            "{name}: EM never decreases the likelihood"
        );
        assert!(result.bootstrap.is_none(), "{name}: no replicates were requested");
        assert!(result.classes.iter().all(|c| c.effect_se.is_none()));
    }
}

#[test]
fn premises_the_data_cannot_supply_are_refused_when_undeclared() {
    let reference: Value =
        serde_json::from_str(include_str!("oracle_data/latent_class_reference.json")).unwrap();
    let case = &reference["cases"][0];
    let covariates: Vec<Vec<f64>> =
        case["covariates"].as_array().unwrap().iter().map(floats).collect();
    let (outcome, treatment) = (floats(&case["y"]), floats(&case["treatment"]));
    let data =
        LatentClassData { outcome: &outcome, treatment: &treatment, covariates: &covariates };
    let undeclared = LatentClassConfig::new(2, 7);
    let error = fit_latent_class_effects(&data, &undeclared).unwrap_err().to_string();
    assert!(error.contains("randomization_not_declared"), "{error}");
    let single = LatentClassConfig::new(1, 7).declare_conditional_randomization();
    assert!(fit_latent_class_effects(&data, &single).is_err(), "one class is not a mixture");
    let constant = vec![1.0; outcome.len()];
    let flat = LatentClassData { outcome: &outcome, treatment: &constant, covariates: &covariates };
    let declared = LatentClassConfig::new(2, 7).declare_conditional_randomization();
    let error = fit_latent_class_effects(&flat, &declared).unwrap_err().to_string();
    assert!(error.contains("treatment_constant"), "{error}");
}
