//! The Bayesian event-time contrasts against exact Student-t references.
//!
//! `oracle_data/bayesian_event_study_reference.py` (frozen in the JSON beside it) builds a
//! balanced panel with never-treated subjects and two adoption cohorts, forms for each
//! cohort-by-period pair the change score from the cohort's reference period `g - 1`, and gives
//! the exact Normal-Inverse-Gamma Student-t posterior of the group coefficient in 40-digit
//! arithmetic. Rows are passed in reverse order to show that subjects and periods are keyed, not
//! positional.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::bayesian_panel_did::DidPrior;
use antecedent_estimate::bayesian_staggered_event_study::fit_bayesian_event_study;
use antecedent_estimate::util::DrawSpec;
use serde_json::Value;

const DRAWS: usize = 200_000;

#[test]
fn every_cohort_period_contrast_matches_the_exact_posterior() {
    let data: Value =
        serde_json::from_str(include_str!("oracle_data/bayesian_event_study_reference.json"))
            .unwrap();
    let rows = data["rows"].as_array().unwrap();
    let prior_values = data["prior"].as_array().unwrap();
    let prior = DidPrior {
        coefficient_scale: prior_values[0].as_f64().unwrap(),
        variance_shape: prior_values[1].as_f64().unwrap(),
        variance_scale: prior_values[2].as_f64().unwrap(),
    };
    let order: Vec<&Value> = rows.iter().rev().collect();
    let outcomes: Vec<f64> = order.iter().map(|r| r["y"].as_f64().unwrap()).collect();
    let subjects: Vec<String> =
        order.iter().map(|r| r["subject"].as_str().unwrap().to_owned()).collect();
    let periods: Vec<i64> = order.iter().map(|r| r["period"].as_i64().unwrap()).collect();
    let cohorts: Vec<i64> = order.iter().map(|r| r["cohort"].as_i64().unwrap()).collect();
    let fits = fit_bayesian_event_study(
        &outcomes,
        &subjects,
        &periods,
        &cohorts,
        &subjects,
        prior,
        DrawSpec::new(DRAWS, 23),
    )
    .unwrap();
    let want = data["contrasts"].as_array().unwrap();
    assert_eq!(fits.len(), want.len(), "one contrast per cohort and non-reference period");
    for w in want {
        let (cohort, period) = (w["cohort"].as_i64().unwrap(), w["period"].as_i64().unwrap());
        let fit = fits
            .iter()
            .find(|f| f.cohort == cohort && f.period == period)
            .unwrap_or_else(|| panic!("missing contrast {cohort}/{period}"));
        let name = format!("cohort {cohort} period {period}");
        let sd = w["sd"].as_f64().unwrap();
        assert_eq!(fit.event_time, w["event_time"].as_i64().unwrap(), "{name}");
        assert_eq!(fit.treated_subjects as u64, w["treated"].as_u64().unwrap(), "{name}");
        assert_eq!(fit.comparison_subjects as u64, w["comparison"].as_u64().unwrap(), "{name}");
        assert!(
            (fit.mean - w["mean"].as_f64().unwrap()).abs() < 5.0 * sd / (DRAWS as f64).sqrt(),
            "{name} mean"
        );
        assert!((fit.sd / sd - 1.0).abs() < 0.02, "{name} sd");
        assert!((fit.lower - w["lower"].as_f64().unwrap()).abs() < 0.03 * sd, "{name} lower");
        assert!((fit.upper - w["upper"].as_f64().unwrap()).abs() < 0.03 * sd, "{name} upper");
    }
    assert!(!fits.iter().any(|f| f.event_time == -1), "the reference period is omitted");
}
