//! The Bayesian-bootstrap policy value against exact Dirichlet-weighted score laws.
//!
//! With Dirichlet(1, ..., 1) row weights the posterior of a score mean has the B-spline law
//! `P(S <= x) = 1 - sum_i (s_i - x)_+^(n-1) / prod_{j != i} (s_i - s_j)` for distinct scores.
//! `oracle_data/policy_value_posterior_reference.py` writes the binary augmented scores of a
//! policy and of its complement from the estimand, evaluates the law of the policy value, of the
//! reference value and of their paired difference in exact rational arithmetic, and freezes the
//! distribution functions on a grid. The three posteriors share draws, so the difference law is
//! the law of the differenced scores, not of a difference of independent laws.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_estimate::policy_value::BinaryPolicyDesign;
use antecedent_estimate::policy_value_posterior::bayesian_binary_policy_value;
use antecedent_estimate::score_posterior::ScorePosteriorOptions;
use serde_json::Value;

const DRAWS: usize = 200_000;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn bools(v: &Value) -> Vec<bool> {
    v.as_array().unwrap().iter().map(|x| x.as_bool().unwrap()).collect()
}

fn empirical(draws: &[f64], x: f64) -> f64 {
    draws.iter().filter(|d| **d <= x).count() as f64 / draws.len() as f64
}

#[test]
fn the_policy_reference_and_paired_difference_laws_match_the_exact_distribution_functions() {
    let data: Value =
        serde_json::from_str(include_str!("oracle_data/policy_value_posterior_reference.json"))
            .unwrap();
    let n = data["y"].as_array().unwrap().len();
    let (assignment, actions, reference) =
        (bools(&data["assigned"]), bools(&data["actions"]), bools(&data["reference_actions"]));
    let propensity = floats(&data["propensity"]);
    let costs = vec![data["cost"].as_f64().unwrap(); n];
    let reference_costs = vec![data["reference_cost"].as_f64().unwrap(); n];
    let design = BinaryPolicyDesign {
        assignment: &assignment,
        actions: &actions,
        propensity: &propensity,
        reference: &reference,
        costs: &costs,
        reference_costs: &reference_costs,
    };
    let (mu0, mu1) = (floats(&data["mu0"]), floats(&data["mu1"]));
    let fit = bayesian_binary_policy_value(
        &floats(&data["y"]),
        &design,
        Some((&mu0, &mu1)),
        ScorePosteriorOptions { draws: DRAWS, coverage: 0.95 },
        &ExecutionContext::for_tests(29),
    )
    .unwrap();
    for (name, posterior) in
        [("policy", &fit.policy), ("reference", &fit.reference), ("incremental", &fit.incremental)]
    {
        let want = &data[name];
        assert!((posterior.score_mean - want["mean"].as_f64().unwrap()).abs() < 1e-12, "{name}");
        for (x, f) in floats(&want["grid"]).into_iter().zip(floats(&want["cdf"])) {
            let got = empirical(&posterior.draws, x);
            let se = (f * (1.0 - f) / DRAWS as f64).sqrt();
            assert!((got - f).abs() < 5.0 * se + 1e-9, "{name} cdf at {x}: {got} vs {f}");
        }
    }
    for k in 0..200 {
        let paired = fit.policy.draws[k] - fit.reference.draws[k];
        assert!((fit.incremental.draws[k] - paired).abs() < 1e-9);
    }
}
