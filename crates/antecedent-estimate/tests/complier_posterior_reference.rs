//! The complier-effect posterior against the exact Dirichlet-weighted score laws.
//!
//! With Dirichlet(1, ..., 1) row weights the posterior of a score mean has the B-spline law
//! `P(S <= x) = 1 - sum_i (c_i - x)_+^(n-1) / prod_{j != i} (c_i - c_j)`. For one-sided
//! compliance the receipt score is nonnegative, so the Wald-ratio law is `P(R <= r) =
//! P(S(a - r b) <= 0)`. `oracle_data/score_posterior_reference.py` evaluates both in exact
//! rational arithmetic on ten rows with a known and with a row-varying assignment probability
//! and freezes the distribution functions and the 2.5% and 97.5% points of the ratio.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_estimate::complier_posterior::bayesian_complier_effect;
use antecedent_estimate::score_posterior::ScorePosteriorOptions;
use serde_json::Value;

const DRAWS: usize = 200_000;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn empirical(draws: &[f64], x: f64) -> f64 {
    draws.iter().filter(|d| **d <= x).count() as f64 / draws.len() as f64
}

fn within(got: f64, want: f64, what: &str) {
    let se = (want * (1.0 - want) / DRAWS as f64).sqrt();
    assert!((got - want).abs() < 5.0 * se + 1e-9, "{what}: {got} vs {want}");
}

#[test]
fn the_ratio_and_intention_to_treat_laws_match_the_exact_distribution_functions() {
    let cases: Vec<Value> =
        serde_json::from_str::<Value>(include_str!("oracle_data/score_posterior_reference.json"))
            .unwrap()
            .as_array()
            .unwrap()
            .clone();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let assigned: Vec<bool> =
            case["assigned"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect();
        let received: Vec<bool> =
            case["received"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect();
        let fit = bayesian_complier_effect(
            &floats(&case["y"]),
            &assigned,
            &received,
            &floats(&case["probabilities"]),
            ScorePosteriorOptions { draws: DRAWS, coverage: 0.95 },
            &ExecutionContext::for_tests(41),
        )
        .unwrap();
        assert!(
            (fit.complier.score_mean - case["point"].as_f64().unwrap()).abs() < 1e-12,
            "{name}"
        );
        assert!(
            (fit.intention_to_treat.score_mean - case["itt_mean"].as_f64().unwrap()).abs() < 1e-12
        );
        assert!(
            (fit.first_stage.score_mean - case["first_stage_mean"].as_f64().unwrap()).abs() < 1e-12
        );
        for (r, f) in floats(&case["ratio_grid"]).into_iter().zip(floats(&case["ratio_cdf"])) {
            within(empirical(&fit.complier.draws, r), f, &format!("{name} ratio cdf at {r}"));
        }
        for (x, f) in floats(&case["itt_grid"]).into_iter().zip(floats(&case["itt_cdf"])) {
            within(
                empirical(&fit.intention_to_treat.draws, x),
                f,
                &format!("{name} itt cdf at {x}"),
            );
        }
        within(empirical(&fit.complier.draws, case["ratio_q025"].as_f64().unwrap()), 0.025, name);
        within(empirical(&fit.complier.draws, case["ratio_q975"].as_f64().unwrap()), 0.975, name);
    }
}
