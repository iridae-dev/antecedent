//! Proper scores and paired score-difference standard errors against exact-rational and 50-digit
//! evaluation of their definitions (`oracle_data/proper_scores_reference.py`).
//!
//! Gaussian log score and CRPS from -40 to +40 standard deviations and for scales from 1e-8 to
//! 1e5; finite-law and ensemble CRPS (empirical-law and fair forms) with ensembles offset by up to
//! 1e12 from the origin; Brier and log scores of Bernoulli and categorical laws; the energy score
//! of finite and sampled joint laws; and the iid, Bartlett long-run and cluster-robust variances
//! of a mean difference, with truncation lags up to n - 1.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_validate::predictive_assessment::{
    DependenceDeclaration, EnsembleCrpsForm, ForecastLaw, Outcome, ScoreKind, score_case,
    variance_of_mean,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/proper_scores_reference.json"))
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

fn close(got: f64, want: f64, relative: f64, absolute: f64, what: &str) {
    assert!((got - want).abs() <= absolute + relative * want.abs(), "{what}: {got} vs {want}");
}

fn score(kind: ScoreKind, form: EnsembleCrpsForm, law: &ForecastLaw, outcome: &Outcome) -> f64 {
    score_case(kind, form, law, outcome).expect("score")
}

#[test]
fn gaussian_scores_match_the_reference_and_the_closed_form_equals_the_integral() {
    let reference = reference();
    for case in reference["gaussian"].as_array().expect("gaussian") {
        let (mean, sd, y) = (number(&case["mean"]), number(&case["sd"]), number(&case["y"]));
        let law = ForecastLaw::Gaussian { mean, sd };
        let log = score(ScoreKind::Log, EnsembleCrpsForm::EmpiricalLaw, &law, &Outcome::Real(y));
        close(log, number(&case["log_score"]), 1e-13, 1e-13, &format!("log score {case}"));
        let crps = score(ScoreKind::Crps, EnsembleCrpsForm::EmpiricalLaw, &law, &Outcome::Real(y));
        close(crps, number(&case["crps"]), 1e-12, 1e-15 * sd, &format!("crps {case}"));
    }
    for case in reference["gaussian_crps_quadrature"].as_array().expect("quadrature") {
        let (want, closed) = (number(&case["crps"]), number(&case["closed"]));
        close(
            closed,
            want,
            1e-15,
            1e-15,
            "the closed form is the integral of the squared difference",
        );
        let law = ForecastLaw::Gaussian { mean: number(&case["mean"]), sd: number(&case["sd"]) };
        let got = score(
            ScoreKind::Crps,
            EnsembleCrpsForm::EmpiricalLaw,
            &law,
            &Outcome::Real(number(&case["y"])),
        );
        close(got, want, 1e-12, 0.0, &format!("{case}"));
    }
}

#[test]
fn brier_and_log_scores_of_discrete_laws_match_the_reference() {
    let reference = reference();
    for case in reference["brier"].as_array().expect("brier") {
        let probabilities = numbers(&case["probabilities"]);
        let outcome = case["outcome"].as_u64().expect("outcome") as u32;
        let law = ForecastLaw::Categorical { probabilities: probabilities.clone() };
        let brier = score(
            ScoreKind::Brier,
            EnsembleCrpsForm::EmpiricalLaw,
            &law,
            &Outcome::Category(outcome),
        );
        close(brier, number(&case["brier"]), 1e-13, 1e-15, &format!("{case}"));
        if probabilities.len() == 2 {
            // The Bernoulli Brier score and CRPS are half the two-category sum.
            let bernoulli = ForecastLaw::Bernoulli { probability: probabilities[1] };
            for kind in [ScoreKind::Brier, ScoreKind::Crps] {
                let got = score(
                    kind,
                    EnsembleCrpsForm::EmpiricalLaw,
                    &bernoulli,
                    &Outcome::Category(outcome),
                );
                close(
                    got,
                    0.5 * number(&case["brier"]),
                    1e-13,
                    1e-15,
                    &format!("bernoulli {case}"),
                );
            }
        }
        match case["log_score"].as_str() {
            Some(text) => {
                let log = score(
                    ScoreKind::Log,
                    EnsembleCrpsForm::EmpiricalLaw,
                    &law,
                    &Outcome::Category(outcome),
                );
                close(log, text.parse().expect("number"), 1e-12, 1e-14, &format!("{case}"));
            }
            None => assert!(
                score_case(
                    ScoreKind::Log,
                    EnsembleCrpsForm::EmpiricalLaw,
                    &law,
                    &Outcome::Category(outcome)
                )
                .is_err(),
                "a zero-probability outcome has no finite log score: {case}"
            ),
        }
    }
}

#[test]
fn finite_and_sampled_crps_match_the_energy_form_in_exact_arithmetic() {
    let reference = reference();
    for case in reference["finite"].as_array().expect("finite") {
        let law = ForecastLaw::FiniteReal {
            support: numbers(&case["support"]),
            probabilities: numbers(&case["probabilities"]),
        };
        let y = number(&case["y"]);
        let got = score(ScoreKind::Crps, EnsembleCrpsForm::EmpiricalLaw, &law, &Outcome::Real(y));
        close(got, number(&case["crps"]), 1e-9, 1e-13, &format!("{case}"));
    }
    for case in reference["sample"].as_array().expect("sample") {
        let law = ForecastLaw::EmpiricalSample { draws: numbers(&case["draws"]) };
        let y = number(&case["y"]);
        let empirical =
            score(ScoreKind::Crps, EnsembleCrpsForm::EmpiricalLaw, &law, &Outcome::Real(y));
        close(empirical, number(&case["empirical"]), 1e-9, 1e-12, &format!("empirical {case}"));
        let fair = score(ScoreKind::Crps, EnsembleCrpsForm::Fair, &law, &Outcome::Real(y));
        close(fair, number(&case["fair"]), 1e-9, 1e-12, &format!("fair {case}"));
    }
}

#[test]
fn energy_scores_match_the_reference() {
    let reference = reference();
    for case in reference["energy"].as_array().expect("energy") {
        let dimension = case["dimension"].as_u64().expect("dimension") as usize;
        let atoms = numbers(&case["atoms"]);
        let probabilities = numbers(&case["probabilities"]);
        let y = Outcome::Vector(numbers(&case["y"]));
        let want = number(&case["energy"]);
        let finite = ForecastLaw::JointFinite {
            dimension,
            atoms: atoms.clone(),
            probabilities: probabilities.clone(),
        };
        close(
            score(ScoreKind::Energy, EnsembleCrpsForm::EmpiricalLaw, &finite, &y),
            want,
            1e-12,
            1e-14,
            &format!("{case}"),
        );
        if probabilities.iter().all(|p| (p - probabilities[0]).abs() < 1e-15) {
            let sample = ForecastLaw::JointSample { dimension, draws: atoms };
            close(
                score(ScoreKind::Energy, EnsembleCrpsForm::EmpiricalLaw, &sample, &y),
                want,
                1e-12,
                1e-14,
                &format!("sample {case}"),
            );
        }
    }
}

#[test]
fn mean_difference_variances_match_exact_rational_arithmetic() {
    let reference = reference();
    for case in reference["variance"].as_array().expect("variance") {
        let d = numbers(&case["d"]);
        let want = number(&case["variance"]);
        let got = match case["kind"].as_str().expect("kind") {
            "iid" => variance_of_mean(&d, &DependenceDeclaration::Iid, None),
            "serial" => variance_of_mean(
                &d,
                &DependenceDeclaration::Serial {
                    time_index: (0..d.len() as u64).collect(),
                    truncation_lag: case["lag"].as_u64().expect("lag") as u32,
                },
                None,
            ),
            _ => {
                let labels: Vec<u32> = case["labels"]
                    .as_array()
                    .expect("labels")
                    .iter()
                    .map(|l| l.as_u64().expect("label") as u32)
                    .collect();
                variance_of_mean(
                    &d,
                    &DependenceDeclaration::Cluster { labels: labels.clone() },
                    Some(&labels),
                )
            }
        }
        .expect("variance");
        close(got, want, 1e-9, 0.0, &format!("{} n {}", case["kind"], d.len()));
    }
}
