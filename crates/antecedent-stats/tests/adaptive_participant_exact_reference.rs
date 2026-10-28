//! The participant-level posterior, the beta-binomial mass, the borrowed prior and the exploration
//! floor against exact rational arithmetic (`oracle_data/participant_exact_reference.py`).
//!
//! With a declared correlation and integer Beta shapes the posterior density of an arm is a
//! polynomial in the success probability, so the reference obtains the posterior mean and variance,
//! the probability that each arm is largest, the expected loss and the exceedance of a margin as
//! exact `Fraction` integrals of polynomials, with no quadrature and no special function. Cases:
//! three to eight responses per participant, correlations 0 to 0.9, two to four arms, an empty
//! history, an all-failure arm against an all-success arm (best probabilities near 1e-42), and
//! adjacent arms with 80 participants. The mass function is compared for 2 to 32 responses at means
//! 0, 0.01, 0.5, 0.99 and 1 and correlations up to 0.999; the power-prior arithmetic with discounts
//! from 0 to 1; and the exploration floor of two to six arms against its definition as
//! `max(floor, c p_i)` with the unique normalizing `c`.
//!
//! The estimated-correlation posterior is not covered by this file (its integrand is not a
//! polynomial); see the limitations of the record.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "support/adaptive_procedure_reference.rs"]
mod support;

use antecedent_stats::adaptive_inference::{
    AdaptiveInferenceError as Error, AnytimeSettings, ArmEvidence, ArmPrior, BetaPrior,
    BorrowingPlan, CorrelationModel, DecisionRules, HistoricalPrior, LookDecision,
    ParticipantDeclaration, ParticipantHistory, ParticipantSpec, PosteriorRequest, StopReason,
    apply_exploration_floor, beta_binomial_pmf, exact_posterior,
};
use serde_json::Value;
use support::{frac, fracs, never};

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/participant_exact_reference.json"))
        .expect("frozen reference")
}

fn numbers(value: &Value) -> Vec<f64> {
    value.as_array().expect("array").iter().map(|v| v.as_f64().expect("number")).collect()
}

#[test]
fn the_declared_correlation_posterior_matches_the_exact_polynomial_integrals() {
    let reference = reference();
    let mut worst = 0.0_f64;
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let counts: Vec<Vec<u64>> = case["arms"]
            .as_array()
            .unwrap()
            .iter()
            .map(|a| a["counts"].as_array().unwrap().iter().map(|n| n.as_u64().unwrap()).collect())
            .collect();
        let arms: Vec<ArmEvidence<'_>> = case["arms"]
            .as_array()
            .unwrap()
            .iter()
            .zip(&counts)
            .map(|(a, c)| ArmEvidence {
                prior_alpha: a["alpha"].as_f64().unwrap(),
                prior_beta: a["beta"].as_f64().unwrap(),
                counts: c,
            })
            .collect();
        let request = PosteriorRequest {
            responses: case["responses"].as_u64().unwrap() as u32,
            correlation: CorrelationModel::Declared(frac(&case["rho"])),
            arms: &arms,
            exceedance_margin: case["margin"].as_str().map(|_| frac(&case["margin"])),
        };
        let got = exact_posterior(&request, &never).expect("a posterior");
        let mut check = |label: &str, got: &[f64], want: Vec<f64>| {
            assert_eq!(got.len(), want.len(), "{name} {label}");
            for (arm, (g, w)) in got.iter().zip(&want).enumerate() {
                let error = (g - w).abs();
                assert!(error <= 1e-9, "{name} {label} arm {arm}: {g} vs {w}");
                worst = worst.max(error);
            }
        };
        check("mean", &got.mean, numbers(&case["mean"]));
        let sd: Vec<f64> = got.sd.iter().map(|s| s * s).collect();
        check("variance", &sd, numbers(&case["variance"]));
        check("best", &got.probability_best, numbers(&case["best"]));
        check("loss", &got.expected_loss, numbers(&case["loss"]));
        check("exceedance", &got.exceedance, numbers(&case["exceedance"]));
        assert!(
            got.probability_error <= 1e-9,
            "{name}: probability error {}",
            got.probability_error
        );
    }
    assert!(worst <= 1e-9);
}

#[test]
fn the_beta_binomial_mass_matches_the_exact_product_form() {
    let reference = reference();
    for row in reference["pmfs"].as_array().unwrap() {
        let m = row["responses"].as_u64().unwrap() as u32;
        let (mean, rho) = (frac(&row["mean"]), frac(&row["rho"]));
        let got = beta_binomial_pmf(m, mean, rho);
        let want = numbers(&row["pmf"]);
        for (s, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-12 + 1e-9 * w,
                "m={m} mean={mean} rho={rho} s={s}: {g} vs {w}"
            );
        }
    }
}

#[test]
fn borrowed_priors_resolve_to_the_power_prior_shapes() {
    let reference = reference();
    for row in reference["borrowing"].as_array().unwrap() {
        let initial = BetaPrior {
            alpha: row["initial"][0].as_f64().unwrap(),
            beta: row["initial"][1].as_f64().unwrap(),
        };
        let historical = row["sources"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, s)| HistoricalPrior {
                source: format!("source {i}"),
                successes: s["successes"].as_u64().unwrap(),
                trials: s["trials"].as_u64().unwrap(),
                discount: frac(&s["discount"]),
            })
            .collect();
        let resolved = ArmPrior { initial, historical }.resolve().expect("a valid prior");
        assert!((resolved.prior.alpha - row["alpha"].as_f64().unwrap()).abs() <= 1e-9);
        assert!((resolved.prior.beta - row["beta"].as_f64().unwrap()).abs() <= 1e-9);
        assert!(
            (resolved.effective_sample_size - row["effective_sample_size"].as_f64().unwrap()).abs()
                <= 1e-9
        );
        assert!(
            (resolved.borrowed_sample_size - row["borrowed_sample_size"].as_f64().unwrap()).abs()
                <= 1e-9
        );
    }
}

#[test]
fn the_exploration_floor_is_the_normalized_clamp_of_its_definition() {
    let reference = reference();
    for row in reference["floors"].as_array().unwrap() {
        let mut got = fracs(&row["row"]);
        apply_exploration_floor(&mut got, frac(&row["floor"]));
        let want = numbers(&row["result"]);
        for (arm, (g, w)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - w).abs() <= 1e-12,
                "row {} floor {} arm {arm}: {g} vs {w}",
                row["row"],
                row["floor"]
            );
        }
        assert!((got.iter().sum::<f64>() - 1.0).abs() <= 1e-12);
    }
}

/// One completed participant of arm 1 with no success among two responses leaves the control a
/// posterior expected loss of exactly 1/20 (the exact value of E[(theta_1 - theta_0)+] for
/// Beta(1, 1) against Beta(1, 3)); the integral evaluates it to 0.04999999999999927. A threshold
/// of exactly 1/20 must not stop at "loss below the threshold" on the strength of that rounding,
/// and a threshold a little above must.
#[test]
fn an_expected_loss_threshold_equal_to_the_exact_loss_does_not_stop() {
    let spec = |loss_threshold: f64| {
        ParticipantSpec::new(ParticipantDeclaration {
            arms: 2,
            responses: Some(2),
            correlation: Some(0.0),
            correlation_prior: None,
            borrowing: Some(BorrowingPlan::uniform(2, BetaPrior::UNIFORM)),
            enrolment: Some(vec![1, 1]),
            lag_looks: 0,
            min_exploration: Some(0.1),
            decisions: Some(DecisionRules {
                loss_threshold,
                futility: None,
                participant_cap: None,
            }),
            anytime: Some(AnytimeSettings { alpha: 0.05, tuning_samples: 30.0 }),
        })
        .expect("a valid declaration")
    };
    let history = ParticipantHistory {
        batches: vec![1],
        arms: vec![1],
        successes: vec![Some(0)],
        assignment: vec![vec![0.5, 0.5]],
    };
    // At the threshold the exact reading is "continue", so a history that ends here is not the
    // history of this procedure.
    match spec(0.05).read_history(&history, &never) {
        Err(Error::PremiseFailed { detail, .. }) => {
            assert_eq!(detail, "adaptive_inference.stopping_not_replayed");
        }
        other => panic!("a loss equal to the threshold must not stop the procedure: {other:?}"),
    }
    let above = spec(0.050_001).read_history(&history, &never).expect("a replayed first look");
    assert_eq!(above[0].decision, LookDecision::Stop { reason: StopReason::ExpectedLoss, arm: 0 });
}
