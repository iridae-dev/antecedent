//! The piecewise-exponential survival posterior against exact gamma-mixture moments.
//!
//! `oracle_data/survival_posterior_reference.py` (frozen in the JSON beside it) forms each
//! interval's events and exact exposure (including delayed entry and partial intervals), the
//! gamma posterior of every hazard, and the exact mean and variance of the horizon survival and
//! of the restricted mean survival time from the moments of the independent gamma hazards, in
//! 40-digit arithmetic. The conjugate parameters and the closed-form means are asserted to
//! 1e-9; the draws behind the intervals are compared on mean and standard deviation.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::survival::SurvivalRows;
use antecedent_estimate::survival_posterior::{
    HazardGammaPrior, randomized_piecewise_survival_posterior,
};
use serde_json::Value;

const DRAWS: usize = 200_000;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-12), "{what}: {got} vs {want}");
}

fn mean_sd(values: &[f64]) -> (f64, f64) {
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let var = values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0);
    (mean, var.sqrt())
}

#[test]
fn conjugate_updates_means_and_draw_moments_match_the_exact_reference() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("oracle_data/survival_posterior_reference.json"))
            .unwrap();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let duration = floats(&case["duration"]);
        let event: Vec<i64> =
            case["event"].as_array().unwrap().iter().map(|v| v.as_i64().unwrap()).collect();
        let treated: Vec<bool> =
            case["treated"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect();
        let entry = case["entry"].as_array().map(|_| floats(&case["entry"]));
        let prior = floats(&case["prior"]);
        let fit = randomized_piecewise_survival_posterior(
            &SurvivalRows {
                duration: &duration,
                event_code: &event,
                treated: &treated,
                delayed_entry: entry.as_deref(),
            },
            &floats(&case["cuts"]),
            HazardGammaPrior { shape: prior[0], rate: prior[1] },
            DRAWS,
            19,
        )
        .unwrap();
        for (label, got, want) in [
            ("control", &fit.control, &case["control"]),
            ("treated", &fit.treated, &case["treated_arm"]),
        ] {
            for (k, (p, w)) in got.iter().zip(want.as_array().unwrap()).enumerate() {
                assert_eq!(p.events as u64, w["events"].as_u64().unwrap(), "{name} {label} {k}");
                close(
                    p.exposure,
                    w["exposure"].as_f64().unwrap(),
                    1e-12,
                    &format!("{name} {label} exposure {k}"),
                );
                close(p.shape, w["shape"].as_f64().unwrap(), 1e-12, name);
                close(p.rate, w["rate"].as_f64().unwrap(), 1e-12, name);
            }
        }
        close(fit.control_survival_mean, case["control_survival"].as_f64().unwrap(), 1e-9, name);
        close(fit.treated_survival_mean, case["treated_survival"].as_f64().unwrap(), 1e-9, name);
        close(
            fit.survival_difference_mean,
            case["survival_difference"].as_f64().unwrap(),
            1e-9,
            name,
        );
        close(fit.rmst_difference_mean, case["rmst_difference"].as_f64().unwrap(), 1e-8, name);
        let (m, s) = mean_sd(&fit.survival_difference_draws);
        let want_sd = case["survival_difference_sd"].as_f64().unwrap();
        assert!(
            (m - case["survival_difference"].as_f64().unwrap()).abs()
                < 5.0 * want_sd / (DRAWS as f64).sqrt(),
            "{name} draw mean"
        );
        assert!((s / want_sd - 1.0).abs() < 0.015, "{name} survival sd {s} vs {want_sd}");
        let (m, s) = mean_sd(&fit.rmst_difference_draws);
        let want_sd = case["rmst_difference_sd"].as_f64().unwrap();
        assert!(
            (m - case["rmst_difference"].as_f64().unwrap()).abs()
                < 5.0 * want_sd / (DRAWS as f64).sqrt(),
            "{name} rmst draw mean"
        );
        assert!((s / want_sd - 1.0).abs() < 0.015, "{name} rmst sd {s} vs {want_sd}");
    }
}
