//! The temporal propensity ridge mode and Laplace posterior against a 40-digit reference.
//!
//! `oracle_data/temporal_propensity_reference.py` (frozen in the JSON beside it) maximises the
//! Bernoulli log likelihood with the slopes (not the intercept) penalised on the supplied column
//! scale, for three penalties, and forms the Laplace posterior under independent Gaussian priors
//! with separate intercept and slope scales: mode, observed-information covariance, the score at
//! the mode and the exact posterior-mean score by quadrature of the scalar linear predictor.
//! One lag column is on a different scale from the other so the column-scale convention matters.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::temporal_penalized_score::{
    BayesianTemporalPropensity, FrequentistTemporalPropensity, TemporalPropensityDesign,
    TrajectoryDependence,
};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("oracle_data/temporal_propensity_reference.json")).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn texts(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_str().unwrap().parse().unwrap()).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    let tol = rel * want.abs().max(1e-300);
    assert!((got - want).abs() <= tol, "{what}: got {got} want {want} tol {tol}");
}

fn design(fx: &Value) -> TemporalPropensityDesign {
    let rows: Vec<Vec<f64>> = fx["rows"].as_array().unwrap().iter().map(floats).collect();
    let n = rows.len();
    let mut x = Vec::with_capacity(n * 3);
    for column in 0..3 {
        x.extend(rows.iter().map(|r| r[column]));
    }
    TemporalPropensityDesign::try_new(
        x,
        floats(&fx["treatment"]),
        n,
        TrajectoryDependence::IndependentSubjects,
    )
    .unwrap()
}

fn histories(fx: &Value) -> Vec<Vec<f64>> {
    fx["histories"].as_array().unwrap().iter().map(floats).collect()
}

#[test]
fn ridge_mode_and_score_match_the_multiprecision_penalised_fit() {
    let fx = fixture();
    let d = design(&fx);
    for case in fx["ridge"].as_array().unwrap() {
        let penalty = case["penalty"].as_f64().unwrap();
        let score = FrequentistTemporalPropensity::new(penalty).unwrap().fit(&d).unwrap();
        for (got, want) in score.coefficients.iter().zip(texts(&case["coefficients"])) {
            close(*got, want, 1e-7, &format!("ridge {penalty}"));
        }
        for (history, want) in histories(&fx).iter().zip(texts(&case["scores"])) {
            close(score.at(history).unwrap(), want, 1e-8, &format!("ridge score {penalty}"));
        }
    }
}

#[test]
fn laplace_mode_covariance_and_posterior_mean_score_match_the_gaussian_integral() {
    let fx = fixture();
    let d = design(&fx);
    let draws = 60_000;
    for case in fx["posterior"].as_array().unwrap() {
        let (sd0, sd1) =
            (case["intercept_sd"].as_f64().unwrap(), case["slope_sd"].as_f64().unwrap());
        let label = format!("prior {sd0}/{sd1}");
        let posterior =
            BayesianTemporalPropensity::new(sd0, sd1, draws, 20261028).unwrap().fit(&d).unwrap();
        for (got, want) in posterior.mode.iter().zip(texts(&case["mode"])) {
            close(*got, want, 1e-6, &label);
        }
        for (history, want) in histories(&fx).iter().zip(texts(&case["mode_scores"])) {
            close(posterior.at_mode(history).unwrap(), want, 1e-6, &label);
        }
        // The draws are N(mode, covariance): compare the sample covariance entry by entry.
        let n = draws as f64;
        let mean: Vec<f64> = (0..3)
            .map(|c| (0..draws).map(|k| posterior.draws[k * 3 + c]).sum::<f64>() / n)
            .collect();
        let want_cov: Vec<Vec<f64>> =
            case["covariance"].as_array().unwrap().iter().map(texts).collect();
        for i in 0..3 {
            for j in 0..3 {
                let cov = (0..draws)
                    .map(|k| {
                        (posterior.draws[k * 3 + i] - mean[i])
                            * (posterior.draws[k * 3 + j] - mean[j])
                    })
                    .sum::<f64>()
                    / (n - 1.0);
                // Sampling error of a covariance entry: sqrt((c_ii c_jj + c_ij^2) / n).
                let se = ((want_cov[i][i] * want_cov[j][j] + want_cov[i][j].powi(2)) / n).sqrt();
                assert!((cov - want_cov[i][j]).abs() < 5.0 * se, "{label} cov[{i}][{j}]");
            }
            let se = (want_cov[i][i] / n).sqrt();
            assert!((mean[i] - texts(&case["mode"])[i]).abs() < 5.0 * se, "{label} mean {i}");
        }
        for (history, want) in histories(&fx).iter().zip(texts(&case["posterior_mean_scores"])) {
            let got = posterior.at(history).unwrap();
            // A sigmoid has slope at most one quarter, so its draw standard deviation is below
            // a quarter of the linear predictor's.
            assert!(
                (got - want).abs() < 5.0 * 0.25 * 0.6 / n.sqrt() + 1e-4,
                "{label} {got} {want}"
            );
        }
    }
}
