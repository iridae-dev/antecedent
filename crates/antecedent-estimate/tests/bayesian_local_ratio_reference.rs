//! The Bayesian local ratio against the exact bivariate-Normal ratio law.
//!
//! `oracle_data/bayesian_local_ratio_reference.py` (frozen in the JSON beside it) forms the window,
//! the ridge-form posterior means of the outcome and treatment coefficients, the plug-in residual
//! covariance (floors and correlation clamp included) and the coefficient pair's bivariate
//! Normal law, then integrates the exact distribution function of the ratio of the paired
//! coefficients, in 30-digit arithmetic. Cases: a strong fuzzy jump, a regression kink and a
//! first stage so weak that the interval must be withheld.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::bayesian_local_ratio::{
    LocalRatioKind, LocalRatioSample, LocalRatioSpec, fit_bayesian_local_ratio,
};
use antecedent_estimate::util::DrawSpec;
use serde_json::Value;

const DRAWS: usize = 200_000;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn within(got: f64, want: f64, what: &str) {
    let se = (want * (1.0 - want) / DRAWS as f64).sqrt();
    assert!((got - want).abs() < 5.0 * se + 1e-6, "{what}: {got} vs {want}");
}

#[test]
fn the_paired_draws_follow_the_exact_numerator_denominator_and_ratio_laws() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("oracle_data/bayesian_local_ratio_reference.json"))
            .unwrap();
    for case in cases {
        let name = case["law"].as_str().unwrap().to_owned();
        let (running, treatment, outcome) =
            (floats(&case["running"]), floats(&case["treatment"]), floats(&case["outcome"]));
        let fit = fit_bayesian_local_ratio(
            &LocalRatioSample { running: &running, treatment: &treatment, outcome: &outcome },
            &LocalRatioSpec {
                kind: if case["kind"] == "jump" {
                    LocalRatioKind::FuzzyJump
                } else {
                    LocalRatioKind::Kink
                },
                cutoff: case["cutoff"].as_f64().unwrap(),
                bandwidth: case["bandwidth"].as_f64().unwrap(),
                prior_sd: case["prior_sd"].as_f64().unwrap(),
                first_stage_floor: case["floor"].as_f64().unwrap(),
            },
            DrawSpec::new(DRAWS, 3),
        )
        .unwrap();
        let (mean, sd) =
            (case["denominator_mean"].as_f64().unwrap(), case["denominator_sd"].as_f64().unwrap());
        assert_eq!(fit.n_window as u64, case["n_window"].as_u64().unwrap(), "{name}");
        assert!((fit.first_stage_mean - mean).abs() < 5.0 * sd / (DRAWS as f64).sqrt(), "{name}");
        assert!((fit.first_stage_sd / sd - 1.0).abs() < 0.01, "{name} sd");
        within(
            fit.first_stage_positive_probability,
            case["denominator_positive"].as_f64().unwrap(),
            &format!("{name} positive"),
        );
        within(
            fit.first_stage_near_zero_mass,
            case["near_zero_mass"].as_f64().unwrap(),
            &format!("{name} near zero"),
        );
        assert_eq!(fit.weak_first_stage, case["weak"].as_bool().unwrap(), "{name} weak flag");
        assert_eq!(fit.ratio_interval.is_none(), fit.weak_first_stage, "{name} interval withheld");
        for (r, f) in floats(&case["ratio_grid"]).into_iter().zip(floats(&case["ratio_cdf"])) {
            let got = fit.ratio_draws.iter().filter(|d| **d <= r).count() as f64 / DRAWS as f64;
            within(got, f, &format!("{name} ratio cdf at {r}"));
        }
        // The coefficient draws are paired: the ratio is the draw-wise quotient.
        for k in 0..100 {
            let want = fit.reduced_form_draws[k] / fit.first_stage_draws[k];
            assert!((fit.ratio_draws[k] - want).abs() <= 1e-12 * want.abs().max(1.0));
        }
    }
}
