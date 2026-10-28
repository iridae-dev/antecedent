//! The Bayesian randomized-trial arm contrasts against exact arm laws.
//!
//! `oracle_data/randomized_posterior_reference.py` (frozen in the JSON beside it) derives, in
//! 30-digit arithmetic, the contrast laws for Bernoulli arms with a Beta prior and Gaussian arms
//! with a Normal-Inverse-Gamma prior under both estimands: the superpopulation contrast (a
//! difference of independent Beta or Student-t variables, by numerical integration of the
//! distribution function) and the finite-population contrast (an exactly enumerated
//! Beta-Binomial convolution, or a Student-t predictive sum for Gaussian arms).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::randomized_posterior::{
    ArmBetaPrior, ArmContrastRequest, ArmNormalPrior, ArmOutcomeModel, RandomizedEstimand,
    posterior_arm_contrasts,
};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

#[test]
fn every_contrast_law_matches_the_exact_reference() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("oracle_data/randomized_posterior_reference.json"))
            .unwrap();
    assert_eq!(cases.len(), 7);
    for case in cases {
        let prior = floats(&case["prior"]);
        let model = if case["model"] == "bernoulli" {
            ArmOutcomeModel::Bernoulli(ArmBetaPrior { alpha: prior[0], beta: prior[1] })
        } else {
            ArmOutcomeModel::Gaussian(ArmNormalPrior {
                mean: prior[0],
                pseudo_units: prior[1],
                variance_shape: prior[2],
                variance_scale: prior[3],
            })
        };
        let finite = case["estimand"] == "finite_population";
        let outcome = floats(&case["outcome"]);
        let assignment: Vec<usize> = case["assignment"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| usize::try_from(v.as_u64().unwrap()).unwrap())
            .collect();
        let weights = floats(&case["weights"]);
        let arms = weights.len();
        let probabilities = vec![vec![1.0 / arms as f64; arms]; outcome.len()];
        let contrasts = vec![weights.clone()];
        let name = format!("{} {} {:?}", case["model"], case["estimand"], weights);
        let fit = posterior_arm_contrasts(&ArmContrastRequest {
            outcome: &outcome,
            assignment: &assignment,
            probabilities: &probabilities,
            model,
            contrasts: &contrasts,
            estimand: if finite {
                RandomizedEstimand::FinitePopulation
            } else {
                RandomizedEstimand::Superpopulation
            },
            credible_level: 0.95,
        })
        .unwrap();
        for (arm, (m, sd)) in fit
            .arms
            .iter()
            .zip(floats(&case["arm_means"]).into_iter().zip(floats(&case["arm_sds"])))
        {
            assert!((arm.mean - m).abs() < 1e-10, "{name} arm mean");
            assert!((arm.standard_deviation - sd).abs() < 1e-9 * sd.max(1.0), "{name} arm sd");
        }
        let got = &fit.contrasts[0];
        let (mean, sd) = (case["mean"].as_f64().unwrap(), case["sd"].as_f64().unwrap());
        // The lattice is a discretization with sixteen cells per arm scale, so the continuous
        // laws are compared to a fraction of a percent of the sd; the discrete one is exact.
        let tol = if finite && case["model"] == "bernoulli" { 1e-9 } else { 0.01 * sd };
        assert!((got.mean - mean).abs() < tol.max(1e-9), "{name} mean {} vs {mean}", got.mean);
        assert!(
            (got.standard_deviation / sd - 1.0).abs() < 0.01,
            "{name} sd {} vs {sd}",
            got.standard_deviation
        );
        assert!(
            (got.interval[0] - case["lower"].as_f64().unwrap()).abs() < tol.max(1e-9) * 2.0,
            "{name} lower {}",
            got.interval[0]
        );
        assert!(
            (got.interval[1] - case["upper"].as_f64().unwrap()).abs() < tol.max(1e-9) * 2.0,
            "{name} upper {}",
            got.interval[1]
        );
        assert!(
            (got.probability_positive - case["positive"].as_f64().unwrap()).abs() < 2e-3,
            "{name} positive {}",
            got.probability_positive
        );
    }
}
