//! The Bayesian-bootstrap trial-to-target draw against the exact Dirichlet simplex-spline law.
//!
//! `oracle_data/trial_bayesian_bootstrap_reference.py` (frozen in the JSON beside it) forms the
//! per-row scores, the exact mean and variance of `(m / n_t) sum pi_i psi_i` for
//! `pi ~ Dirichlet(1, ..., 1)`, and its exact distribution function by the Hermite-Genocchi
//! divided-difference formula (cross-checked by simulation in the script itself); the blocked
//! case with equal trial counts per block has Dirichlet block weights, and the blocked case with
//! unequal counts is a ratio of exponentials whose distribution function is the partial-fraction
//! formula of a weighted sum of exponentials.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_estimate::transport::{
    trial_to_target_bayesian_bootstrap, trial_to_target_bayesian_bootstrap_blocks,
};
use antecedent_estimate::util::DrawSpec;
use antecedent_identify::{
    PopulationFactor, TransportCertificate, TransportFormula, TransportIdentification,
};
use serde_json::Value;

fn identification() -> TransportIdentification {
    TransportIdentification::Transportable {
        formula: TransportFormula::Direct(PopulationFactor {
            regime: None,
            population: Arc::from("source"),
            variables: Arc::from([]),
            conditioned_on: Arc::from([]),
            interventions: Arc::from([]),
        }),
        certificate: TransportCertificate {
            rule: Arc::from("transport.sid.direct"),
            selection_targets: Arc::from([]),
            premises: Arc::from([]),
        },
    }
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn flags(v: &Value) -> Vec<bool> {
    v.as_array().unwrap().iter().map(|x| x.as_bool().unwrap()).collect()
}

fn text(v: &Value) -> f64 {
    v.as_str().unwrap().parse().unwrap()
}

#[test]
fn draws_follow_the_exact_dirichlet_law_free_and_blocked() {
    let cases: Vec<Value> = serde_json::from_str::<Value>(include_str!(
        "oracle_data/trial_bayesian_bootstrap_reference.json"
    ))
    .unwrap()
    .as_array()
    .cloned()
    .unwrap();
    let draws = 200_000;
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let (outcome, treatment, trial) =
            (floats(&case["outcome"]), flags(&case["treatment"]), flags(&case["trial"]));
        let (selection, propensity) = (floats(&case["selection"]), floats(&case["propensity"]));
        let sampling = DrawSpec { draws, seed: 20261028 };
        let out = match case["block"].as_u64() {
            None => trial_to_target_bayesian_bootstrap(
                &identification(),
                &outcome,
                &treatment,
                &trial,
                &selection,
                &propensity,
                sampling,
            ),
            Some(block) => trial_to_target_bayesian_bootstrap_blocks(
                &identification(),
                &outcome,
                &treatment,
                &trial,
                &selection,
                &propensity,
                (sampling, Some(block as usize)),
            ),
        }
        .unwrap();
        let n = draws as f64;
        let mean = out.iter().sum::<f64>() / n;
        let var = out.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
        if !case["mean"].is_null() {
            let (want_mean, want_var) = (text(&case["mean"]), text(&case["variance"]));
            assert!((mean - want_mean).abs() < 5.0 * (want_var / n).sqrt(), "{name} mean {mean}");
            // Relative standard error of a sample variance of a bounded, light-tailed draw is below
            // 0.5%; 3% leaves a wide margin.
            assert!((var - want_var).abs() < 0.03 * want_var, "{name} variance {var}");
        }
        assert!(var.is_finite());
        // Every draw lies in the convex hull of the row scores: weights are positive and sum to one.
        assert!(out.iter().all(|v| *v >= text(&case["min"]) && *v <= text(&case["max"])));
        for point in case["cdf"].as_array().unwrap() {
            let x = text(&point["x"]);
            let p = text(&point["p"]);
            let empirical = out.iter().filter(|v| **v <= x).count() as f64 / n;
            let tol = 5.0 * (p * (1.0 - p) / n).sqrt();
            assert!((empirical - p).abs() < tol, "{name} F({x}) = {empirical} vs {p}");
        }
    }
}

#[test]
fn a_single_arm_trial_and_too_few_draws_are_refused() {
    let outcome = [1.0, 2.0, 0.0];
    let trial = [true, true, false];
    let all_treated = [true, true, false];
    let sel = [0.5; 3];
    let prop = [0.5; 3];
    let single_arm = trial_to_target_bayesian_bootstrap(
        &identification(),
        &outcome,
        &all_treated,
        &trial,
        &sel,
        &prop,
        DrawSpec { draws: 100, seed: 1 },
    )
    .err()
    .expect("a single-arm trial is refused")
    .to_string();
    assert!(single_arm.contains("requires both observed treatment arms"), "{single_arm}");
    let one_draw = trial_to_target_bayesian_bootstrap(
        &identification(),
        &outcome,
        &[true, false, false],
        &trial,
        &sel,
        &prop,
        DrawSpec { draws: 1, seed: 1 },
    )
    .err()
    .expect("one draw is refused")
    .to_string();
    assert!(one_draw.contains("requires at least two draws"), "{one_draw}");
}
