//! Regime survival (always-treat and never-treat) against a multiprecision g-formula reference.
//!
//! `oracle_data/temporal_regime_survival_reference.py` (frozen in the JSON beside it) simulates
//! person-period trajectories with time-varying confounding and censoring, fits the pooled
//! logistic hazard (uncensored rows), the Gaussian linear confounder transition and the pooled
//! logistic treatment model by maximum likelihood in multiprecision arithmetic and integrates the
//! g-formula over the Gaussian innovations by nested Gauss-Hermite quadrature, for follow-up
//! truncated at the horizon, with no censoring at all, under strong confounding and with 36
//! subjects. The product evaluates the same integral on 16 fixed Monte Carlo paths per subject,
//! so its point values are held to the quadrature values within the Monte Carlo tolerance the
//! oracle measures for exactly that scheme (five standard deviations); the smallest regime
//! probability is deterministic and held to 1e-6. Its subject bootstrap is exercised only for
//! shape because its draws come from a seeded stream.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_estimate::temporal_design::survival::{
    MIN_REPLICATES, Trajectories, TrajectoryRows, fit_regime_survival,
};
use serde_json::Value;

fn close(actual: f64, expected: f64, tol: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= tol * expected.abs().max(1e-3),
        "{what}: {actual} vs {expected}"
    );
}

#[test]
fn regime_survival_matches_the_multiprecision_reference() {
    let reference: Value =
        serde_json::from_str(include_str!("oracle_data/temporal_regime_survival_reference.json"))
            .unwrap();
    let mut refused: Vec<String> = Vec::new();
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let rows = case["rows"].as_array().unwrap();
        let unit: Vec<usize> =
            rows.iter().map(|r| usize::try_from(r["unit"].as_u64().unwrap()).unwrap()).collect();
        let period: Vec<i64> = rows.iter().map(|r| r["period"].as_i64().unwrap()).collect();
        let flag =
            |key: &str| -> Vec<bool> { rows.iter().map(|r| r[key].as_bool().unwrap()).collect() };
        let (treatment, censored, event) = (flag("treatment"), flag("censored"), flag("event"));
        let confounder: Vec<f64> = rows.iter().map(|r| r["L"].as_f64().unwrap()).collect();
        let horizon = usize::try_from(case["horizon"].as_u64().unwrap()).unwrap();
        let data = Trajectories::new(
            TrajectoryRows {
                unit: &unit,
                period: &period,
                treatment: &treatment,
                censored: &censored,
                event: &event,
                confounders: &[confounder],
            },
            horizon,
        )
        .unwrap();
        assert_eq!(data.subjects() as u64, case["subjects"].as_u64().unwrap(), "{name}");
        assert_eq!(
            data.person_periods() as u64,
            case["person_periods"].as_u64().unwrap(),
            "{name}"
        );
        let outcome = fit_regime_survival(
            &data,
            case["floor"].as_f64().unwrap(),
            MIN_REPLICATES,
            &ExecutionContext::for_tests(5),
        );
        let fit = match outcome {
            Ok(fit) => fit,
            Err(error) => {
                // The interval refits the models in every subject-bootstrap replicate and the
                // fit refuses when more than five percent of them cannot be refitted; the point
                // is then not published. Only designs thin enough to lose support may refuse.
                let text = error.to_string();
                assert!(
                    text.contains("regime_survival.bootstrap_failures")
                        && ["small_sample"].contains(&name),
                    "{name}: unexpected refusal {text}"
                );
                refused.push(name.to_owned());
                continue;
            }
        };
        let want = case["estimate"].as_array().unwrap();
        let tolerance = case["monte_carlo_tolerance"].as_array().unwrap();
        for (j, label) in ["difference", "always_treat", "never_treat"].iter().enumerate() {
            let (expected, allowed) = (want[j].as_f64().unwrap(), tolerance[j].as_f64().unwrap());
            assert!(
                (fit.estimate[j] - expected).abs() <= allowed,
                "{name} {label}: {} vs {expected} (Monte Carlo tolerance {allowed})",
                fit.estimate[j]
            );
        }
        close(
            fit.minimum_regime_probability,
            case["minimum_regime_probability"].as_f64().unwrap(),
            1e-6,
            &format!("{name} smallest regime probability"),
        );
        // The interval is a bootstrap percentile band: finite, ordered and inside the range of
        // a survival difference.
        let (ok, failed) = fit.replicates;
        assert_eq!(ok + failed, MIN_REPLICATES, "{name}");
        for ((lo, hi), se) in fit.interval.unwrap().iter().zip(fit.standard_error.unwrap()) {
            assert!(lo <= hi && *lo >= -1.0 - 1e-12 && *hi <= 1.0 + 1e-12 && se >= 0.0, "{name}");
        }
    }
    eprintln!("refused for bootstrap support: {refused:?}");
    assert!(refused.len() < 2, "{refused:?}");
}
