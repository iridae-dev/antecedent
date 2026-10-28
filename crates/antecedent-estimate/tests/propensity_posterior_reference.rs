//! The Laplace posterior of a static logistic propensity against an exact reference.
//!
//! `oracle_data/propensity_posterior_reference.py` (frozen in the JSON beside it) finds the
//! posterior mode of an 80-row logistic design with two covariates under a Gaussian coefficient
//! prior (intercept included) by 40-digit Newton iteration, forms the Laplace covariance at the
//! mode, and evaluates for four probe rows the exact posterior mean and standard deviation of the
//! score (quadrature of the logistic function against the Normal linear predictor), its value at
//! the mode and the probability that it lies outside a positivity band (a Normal tail on the
//! logit scale), for three prior scales.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::penalized_propensity_posterior::{
    PropensityCoefficientPrior, StaticPropensityDesign,
};
use serde_json::Value;

const DRAWS: usize = 200_000;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

#[test]
fn the_mode_and_the_score_summaries_match_the_exact_laplace_posterior() {
    let data: Value =
        serde_json::from_str(include_str!("oracle_data/propensity_posterior_reference.json"))
            .unwrap();
    let rows: Vec<Vec<f64>> = data["rows"].as_array().unwrap().iter().map(floats).collect();
    let n = rows.len();
    let p = rows[0].len();
    let mut x_colmajor = Vec::with_capacity(n * p);
    for j in 0..p {
        x_colmajor.extend(rows.iter().map(|r| r[j]));
    }
    let design =
        StaticPropensityDesign::try_new(x_colmajor, floats(&data["treatment"]), n).unwrap();
    for case in data["cases"].as_array().unwrap() {
        let sd_prior = case["prior_sd"].as_f64().unwrap();
        let clip = case["clip"].as_f64().unwrap();
        let fit =
            PropensityCoefficientPrior::new(sd_prior, DRAWS, 11).unwrap().fit(&design).unwrap();
        for (got, want) in fit.mode.iter().zip(floats(&case["mode"])) {
            assert!((got - want).abs() < 1e-6, "prior {sd_prior} mode {got} vs {want}");
        }
        let probes = case["probes"].as_array().unwrap();
        for (probe, want) in probes.iter().zip(case["scores"].as_array().unwrap()) {
            let got = fit.score_at(&floats(probe), clip).unwrap();
            let what = format!("prior {sd_prior} probe {probe}");
            let (mean, sd, outside) = (
                want["mean"].as_f64().unwrap(),
                want["sd"].as_f64().unwrap(),
                want["outside_band"].as_f64().unwrap(),
            );
            assert!(
                (got.at_mode - want["at_mode"].as_f64().unwrap()).abs() < 1e-6,
                "{what} at mode"
            );
            assert!((got.mean - mean).abs() < 5.0 * sd / (DRAWS as f64).sqrt(), "{what} mean");
            assert!((got.sd / sd - 1.0).abs() < 0.01, "{what} sd {} vs {sd}", got.sd);
            let se = (outside * (1.0 - outside) / DRAWS as f64).sqrt();
            assert!((got.outside_band - outside).abs() < 5.0 * se + 1e-6, "{what} outside band");
        }
    }
}
