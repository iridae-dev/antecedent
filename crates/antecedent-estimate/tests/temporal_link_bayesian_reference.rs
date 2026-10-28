//! The Laplace temporal GLM contrast with two held lags, extreme scales and draw-level moments.
//!
//! `oracle_data/temporal_link_bayesian_reference.py` (frozen in the JSON beside it) takes designs
//! with an intercept, a treatment regime column, two lags held at their observed values and a
//! treatment-by-first-lag interaction regime column, fits the Bernoulli likelihood by Newton's
//! method at 40 digits (unpenalised, and with a Gaussian prior on every coefficient), forms the
//! average mean-scale contrast, its Fisher delta-method standard error, the contrast at the
//! Laplace mode and the mean and standard deviation of the contrast over the Laplace normal by a
//! 13-point Gauss-Hermite rule in five dimensions. This extends `temporal_link_reference.rs`
//! (one lag, one prior scale per link, posterior mean only) to two lags, prior scales from 0.15
//! to 20, a lag offset by 100 at scale 0.5, an unbalanced treatment arm, a probit link and the
//! draw-level standard deviation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::temporal_link::{
    BayesianTemporalLink, FrequentistTemporalLink, RegimeColumn, TemporalContrastDesign,
    TemporalOutcomeLink, TrajectoryDependence,
};
use serde_json::Value;

const DRAWS: usize = 200_000;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(|x| x.as_f64().expect("number")).collect()
}

fn design(case: &Value) -> TemporalContrastDesign {
    let rows: Vec<Vec<f64>> = case["x"].as_array().unwrap().iter().map(floats).collect();
    let n = rows.len();
    let p = rows[0].len();
    let mut x = vec![0.0; n * p];
    for (i, row) in rows.iter().enumerate() {
        for (j, value) in row.iter().enumerate() {
            x[j * n + i] = *value;
        }
    }
    let act: Vec<Vec<f64>> = case["act"].as_array().unwrap().iter().map(floats).collect();
    let ctl: Vec<Vec<f64>> = case["ctl"].as_array().unwrap().iter().map(floats).collect();
    let regime = |column: usize| RegimeColumn {
        column,
        active: act.iter().map(|r| r[column]).collect(),
        control: ctl.iter().map(|r| r[column]).collect(),
    };
    TemporalContrastDesign::try_new(
        x,
        floats(&case["y"]),
        n,
        vec![regime(1), regime(4)],
        TrajectoryDependence::IndependentSubjects,
    )
    .unwrap()
}

fn link(case: &Value) -> TemporalOutcomeLink {
    if case["link"] == "probit" {
        TemporalOutcomeLink::BernoulliProbit
    } else {
        TemporalOutcomeLink::BernoulliLogit
    }
}

fn rel(got: f64, want: f64, tol: f64, what: &str) {
    assert!((got - want).abs() <= tol * want.abs().max(1e-3), "{what}: {got} vs {want}");
}

#[test]
fn maximum_likelihood_contrast_and_fisher_standard_error_with_two_lags() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("oracle_data/temporal_link_bayesian_reference.json"))
            .unwrap();
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let fit = FrequentistTemporalLink::new().fit(&design(case), link(case)).unwrap();
        assert_eq!(fit.lags, 2, "{name}");
        rel(fit.estimate, case["mle_estimate"].as_f64().unwrap(), 1e-7, name);
        rel(fit.standard_error, case["mle_se"].as_f64().unwrap(), 1e-6, name);
        let coefficients = floats(&case["mle_coefficients"]);
        // The regime coefficients are the treatment and the interaction coefficient.
        rel(fit.regime_coefficients[0], coefficients[1], 1e-6, name);
        rel(fit.regime_coefficients[1], coefficients[4], 1e-6, name);
    }
}

#[test]
fn laplace_posterior_mean_and_draw_standard_deviation_equal_the_gauss_hermite_moments() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("oracle_data/temporal_link_bayesian_reference.json"))
            .unwrap();
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let bayes =
            BayesianTemporalLink::new(case["prior_sd"].as_f64().unwrap(), DRAWS, 5).unwrap();
        let fit = bayes.fit(&design(case), link(case)).unwrap();
        assert_eq!(fit.draws.len(), DRAWS);
        rel(fit.functional_at_mode, case["mode_contrast"].as_f64().unwrap(), 1e-7, name);
        let mean = case["posterior_mean"].as_f64().unwrap();
        let sd = case["posterior_sd"].as_f64().unwrap();
        // Five Monte Carlo standard errors, plus the quadrature error of the reference.
        assert!(
            (fit.posterior_mean - mean).abs() <= 5.0 * sd / (DRAWS as f64).sqrt() + 1e-6,
            "{name} mean: {} vs {mean}",
            fit.posterior_mean
        );
        let draw_mean = fit.draws.iter().sum::<f64>() / DRAWS as f64;
        assert!((draw_mean - fit.posterior_mean).abs() < 1e-12, "{name}");
        let draw_sd = (fit.draws.iter().map(|d| (d - draw_mean).powi(2)).sum::<f64>()
            / (DRAWS as f64 - 1.0))
            .sqrt();
        assert!((draw_sd - sd).abs() <= 0.02 * sd, "{name} sd: {draw_sd} vs {sd}");
        // The posterior mean of the contrast is not the contrast at the mode: the reference gap
        // is reproduced.
        let gap = mean - case["mode_contrast"].as_f64().unwrap();
        assert!(
            ((fit.posterior_mean - fit.functional_at_mode) - gap).abs()
                <= 5.0 * sd / (DRAWS as f64).sqrt() + 1e-6,
            "{name} gap"
        );
    }
}

#[test]
fn declarations_change_the_answer_or_are_refused() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("oracle_data/temporal_link_bayesian_reference.json"))
            .unwrap();
    let case = &cases[0];
    let d = design(case);
    let base = BayesianTemporalLink::new(1.0, 2000, 5).unwrap().fit(&d, link(case)).unwrap();
    let tight = BayesianTemporalLink::new(0.05, 2000, 5).unwrap().fit(&d, link(case)).unwrap();
    assert!((tight.functional_at_mode - base.functional_at_mode).abs() > 1e-3);
    let reseeded = BayesianTemporalLink::new(1.0, 2000, 6).unwrap().fit(&d, link(case)).unwrap();
    assert_eq!(reseeded.functional_at_mode, base.functional_at_mode);
    assert_ne!(reseeded.draws, base.draws);
    // The same design under the probit link is a different object.
    let probit = BayesianTemporalLink::new(1.0, 2000, 5)
        .unwrap()
        .fit(&d, TemporalOutcomeLink::BernoulliProbit)
        .unwrap();
    assert_ne!(probit.functional_at_mode, base.functional_at_mode);
    for bad in [0.0, -1.0, f64::NAN, f64::INFINITY] {
        assert!(BayesianTemporalLink::new(bad, 100, 1).is_err());
    }
    assert!(BayesianTemporalLink::new(1.0, 0, 1).is_err());
}
