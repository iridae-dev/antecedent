//! The Bayesian difference-in-differences posteriors against exact Student-t references.
//!
//! `oracle_data/bayesian_panel_did_reference.py` (frozen in the JSON beside it) fits the
//! repeated cross-section 2x2, the two-period change-score and the covariate-augmented arm
//! regressions in 40-digit arithmetic and gives the exact posterior mean, standard deviation and
//! equal-tailed 95% interval of the effect under a diffuse and an informative prior. The
//! augmented effect is a difference of two independent Student-t variables, so its interval comes
//! from numerical convolution. The sampler is compared at Monte Carlo tolerances.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::bayesian_panel_did::{
    DidPrior, fit_bayesian_augmented_panel_did, fit_bayesian_cross_section_did,
    fit_bayesian_panel_did,
};
use antecedent_estimate::util::DrawSpec;
use serde_json::Value;

const DRAWS: usize = 300_000;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn flags(v: &Value) -> Vec<bool> {
    v.as_array().unwrap().iter().map(|x| x.as_bool().unwrap()).collect()
}

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/bayesian_panel_did_reference.json"))
        .unwrap()
        .as_array()
        .unwrap()
        .clone()
}

#[test]
fn every_form_matches_the_exact_posterior_under_both_priors() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let prior = DidPrior {
            coefficient_scale: case["coefficient_scale"].as_f64().unwrap(),
            variance_shape: case["variance_shape"].as_f64().unwrap(),
            variance_scale: case["variance_scale"].as_f64().unwrap(),
        };
        let sampling = DrawSpec::new(DRAWS, 17);
        let fit = match case["form"].as_str().unwrap() {
            "cross_section" => {
                let y = floats(&case["y"]);
                let ids: Vec<String> = (0..y.len()).map(|i| format!("r{i}")).collect();
                fit_bayesian_cross_section_did(
                    &y,
                    &flags(&case["group"]),
                    &flags(&case["post"]),
                    &ids,
                    prior,
                    sampling,
                )
                .unwrap()
            }
            "panel" => {
                let pre = floats(&case["pre"]);
                let ids: Vec<String> = (0..pre.len()).map(|i| format!("u{i}")).collect();
                fit_bayesian_panel_did(
                    &pre,
                    &floats(&case["post"]),
                    &flags(&case["treated"]),
                    &ids,
                    prior,
                    sampling,
                )
                .unwrap()
            }
            _ => {
                let pre = floats(&case["pre"]);
                let ids: Vec<String> = (0..pre.len()).map(|i| format!("u{i}")).collect();
                let covariates: Vec<Vec<f64>> =
                    case["covariates"].as_array().unwrap().iter().map(floats).collect();
                fit_bayesian_augmented_panel_did(
                    &pre,
                    &floats(&case["post"]),
                    &flags(&case["treated"]),
                    &covariates,
                    &ids,
                    prior,
                    sampling,
                )
                .unwrap()
            }
        };
        let (mean, sd) = (case["mean"].as_f64().unwrap(), case["sd"].as_f64().unwrap());
        // Five Monte Carlo standard errors on the mean, one percent on the spread (heavy
        // tails make the sample sd noisier than the mean), 3% of the sd on the interval ends.
        assert!((fit.mean - mean).abs() < 5.0 * sd / (DRAWS as f64).sqrt(), "{name} mean");
        assert!((fit.sd / sd - 1.0).abs() < 0.02, "{name} sd {} vs {sd}", fit.sd);
        assert!((fit.lower - case["lower"].as_f64().unwrap()).abs() < 0.03 * sd, "{name} lower");
        assert!((fit.upper - case["upper"].as_f64().unwrap()).abs() < 0.03 * sd, "{name} upper");
    }
}

#[test]
fn the_prior_moves_the_posterior_as_the_reference_says() {
    // The diffuse and informative references differ in mean by more than the tolerance, so a
    // sampler ignoring the declared prior scale could not pass the test above.
    let all = cases();
    for form in ["cross_section", "panel", "augmented"] {
        let pick = |tag: &str| {
            all.iter().find(|c| c["name"] == format!("{form}_{tag}")).unwrap()["mean"]
                .as_f64()
                .unwrap()
        };
        assert!((pick("diffuse") - pick("informative")).abs() > 0.05, "{form}");
    }
}
