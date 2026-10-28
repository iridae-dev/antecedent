//! The temporal GLM decision-time contrast against a 40-digit reference.
//!
//! `oracle_data/temporal_link_reference.py` (frozen in the JSON beside it) maximises the Bernoulli
//! likelihood by Newton's method for a logit and a probit link, forms the average mean-scale
//! contrast of the active and control regimes with the lag held at its observed value, the Fisher
//! delta-method standard error, the Laplace mode and observed-Hessian covariance under a Gaussian
//! prior, the contrast at the mode and the exact posterior-mean contrast of that Gaussian by
//! quadrature of each row's scalar linear predictor.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::temporal_link::{
    BayesianTemporalLink, FrequentistTemporalLink, RegimeColumn, TemporalContrastDesign,
    TemporalOutcomeLink, TrajectoryDependence,
};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("oracle_data/temporal_link_reference.json")).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn text(v: &Value) -> f64 {
    v.as_str().unwrap().parse().unwrap()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    let tol = rel * want.abs().max(1e-300);
    assert!((got - want).abs() <= tol, "{what}: got {got} want {want} tol {tol}");
}

fn design(fx: &Value) -> TemporalContrastDesign {
    let (a, z, y) = (floats(&fx["treatment"]), floats(&fx["lag"]), floats(&fx["outcome"]));
    let n = y.len();
    let interaction: Vec<f64> = a.iter().zip(&z).map(|(a, z)| a * z).collect();
    let mut x = vec![1.0; n];
    x.extend(&a);
    x.extend(&z);
    x.extend(&interaction);
    TemporalContrastDesign::try_new(
        x,
        y,
        n,
        vec![
            RegimeColumn { column: 1, active: vec![1.0; n], control: vec![0.0; n] },
            RegimeColumn { column: 3, active: z.clone(), control: vec![0.0; n] },
        ],
        TrajectoryDependence::IndependentSubjects,
    )
    .unwrap()
}

fn link(name: &str) -> TemporalOutcomeLink {
    match name {
        "logit" => TemporalOutcomeLink::BernoulliLogit,
        _ => TemporalOutcomeLink::BernoulliProbit,
    }
}

#[test]
fn frequentist_contrast_and_fisher_delta_se_match_the_multiprecision_fit() {
    let fx = fixture();
    let d = design(&fx);
    for case in fx["cases"].as_array().unwrap() {
        let name = case["link"].as_str().unwrap();
        let fit = FrequentistTemporalLink::new().fit(&d, link(name)).unwrap();
        close(fit.estimate, text(&case["estimate"]), 1e-9, name);
        close(fit.standard_error, text(&case["se"]), 1e-8, name);
        let mle = case["mle"].as_array().unwrap();
        assert_eq!(fit.regime_coefficients.len(), 2);
        close(fit.regime_coefficients[0], text(&mle[1]), 1e-8, name);
        close(fit.regime_coefficients[1], text(&mle[3]), 1e-8, name);
        assert_eq!(fit.lags, 1);
        assert_eq!(fit.trajectories, 140);
    }
}

#[test]
fn laplace_mode_contrast_and_posterior_mean_match_the_exact_gaussian_integral() {
    let fx = fixture();
    let d = design(&fx);
    for case in fx["cases"].as_array().unwrap() {
        let name = case["link"].as_str().unwrap();
        let sd = case["prior_sd"].as_f64().unwrap();
        let draws = 60_000;
        let posterior =
            BayesianTemporalLink::new(sd, draws, 20261028).unwrap().fit(&d, link(name)).unwrap();
        close(posterior.functional_at_mode, text(&case["functional_at_mode"]), 1e-7, name);
        assert_eq!(posterior.draws.len(), draws);
        let n = draws as f64;
        let mean = posterior.draws.iter().sum::<f64>() / n;
        close(posterior.posterior_mean, mean, 1e-12, "published mean is the draw mean");
        let sd_draws =
            (posterior.draws.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
        let want = text(&case["posterior_mean"]);
        assert!(
            (mean - want).abs() < 5.0 * sd_draws / n.sqrt(),
            "{name} sd {sd}: mean {mean} vs {want}"
        );
        // The mean of the draws is not the contrast at the mode: with a curved link they differ.
        assert!((want - text(&case["functional_at_mode"])).abs() > 1e-6);
    }
}
