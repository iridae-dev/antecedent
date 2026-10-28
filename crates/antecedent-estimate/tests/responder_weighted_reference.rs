//! The responder-weighted contrast against an exact stacked M-estimation reference.
//!
//! `oracle_data/responder_weighted_reference.py` (frozen in the JSON beside it) fits each arm's
//! observation logistic model by 40-digit Newton iteration, forms the Hajek mean, and obtains the
//! variance from the stacked sandwich of the logistic score and the weighted estimating equation
//! (rescaled to the sample-variance divisor n - 1 the product uses). An arm in which everyone
//! responds has no observation-model term. The product's influence-function variance, which
//! subtracts the estimated-probability term, must agree.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::randomized_missing::responder_weighted_itt;
use serde_json::Value;

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-12), "{what}: {got} vs {want}");
}

#[test]
fn means_variances_and_weight_diagnostics_match_the_exact_stacked_estimator() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("oracle_data/responder_weighted_reference.json"))
            .unwrap();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let rows = case["rows"].as_array().unwrap();
        let n = rows.len();
        let outcome: Vec<f64> = rows.iter().map(|r| r["y"].as_f64().unwrap()).collect();
        let observed: Vec<bool> = rows.iter().map(|r| r["observed"].as_bool().unwrap()).collect();
        let treated: Vec<bool> = rows.iter().map(|r| r["treated"].as_bool().unwrap()).collect();
        let mut covariates = Vec::with_capacity(2 * n);
        for key in ["x1", "x2"] {
            covariates.extend(rows.iter().map(|r| r[key].as_f64().unwrap()));
        }
        let fit = responder_weighted_itt(&outcome, &observed, &treated, &covariates, 2).unwrap();
        for (label, got, want) in [
            ("control", &fit.arms[0], &case["control"]),
            ("treated", &fit.arms[1], &case["treated"]),
        ] {
            let what = format!("{name} {label}");
            assert_eq!(got.assigned as u64, want["assigned"].as_u64().unwrap(), "{what}");
            assert_eq!(got.responders as u64, want["responders"].as_u64().unwrap(), "{what}");
            close(got.mean, want["mean"].as_f64().unwrap(), 1e-9, &format!("{what} mean"));
            close(
                got.variance,
                want["variance"].as_f64().unwrap(),
                1e-6,
                &format!("{what} variance"),
            );
            close(
                got.min_observation_probability,
                want["min_probability"].as_f64().unwrap(),
                1e-8,
                &format!("{what} min probability"),
            );
            close(
                got.effective_responders,
                want["ess"].as_f64().unwrap(),
                1e-8,
                &format!("{what} ess"),
            );
        }
        close(fit.effect, case["effect"].as_f64().unwrap(), 1e-9, name);
        close(fit.variance, case["variance"].as_f64().unwrap(), 1e-6, name);
    }
}

#[test]
fn fitted_probabilities_above_the_upper_clip_are_not_clipped() {
    // The control arm of the first case has a fitted observation probability of about 0.991.
    // Clipping it to 0.98 (the stats fit's symmetric clip) moved the Hajek mean in the fourth
    // significant digit; the exact fit does not clip.
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("oracle_data/responder_weighted_reference.json"))
            .unwrap();
    assert!(cases[0]["control"]["max_probability"].as_f64().unwrap() > 0.98);
}
