//! Kaplan-Meier, Aalen-Johansen and restricted mean survival against exact rational arithmetic.
//!
//! `oracle_data/product_limit_reference.py` evaluates the risk-set definitions in
//! `fractions.Fraction` (no rounding) on seeded data with tied event times, tied events and
//! censorings, delayed entry, two competing causes, events exactly at the horizon and
//! censoring before it; the frozen curves are rounded once to binary64.

use antecedent_estimate::survival::{
    SurvivalEndpoint, SurvivalRows, randomized_survival_summary,
};
use serde_json::Value;

fn floats(value: &Value) -> Vec<f64> {
    value.as_array().expect("array").iter().map(|v| v.as_f64().expect("number")).collect()
}

#[test]
fn risk_set_curves_and_restricted_means_match_exact_arithmetic() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("oracle_data/product_limit_reference.json"))
            .expect("frozen reference");
    assert!(cases.len() >= 20, "reference should retain many datasets");
    let mut delayed_seen = 0;
    let mut competing_seen = 0;
    for case in &cases {
        let seed = case["seed"].as_u64().expect("seed");
        let duration = floats(&case["duration"]);
        let code: Vec<i64> =
            case["event_code"].as_array().expect("codes").iter().map(|v| v.as_i64().expect("i")).collect();
        let treated: Vec<bool> =
            case["treated"].as_array().expect("treated").iter().map(|v| v.as_bool().expect("b")).collect();
        let entry = case["entry"].as_array().map(|_| floats(&case["entry"]));
        let competing = case["competing"].as_bool().expect("competing");
        delayed_seen += usize::from(entry.is_some());
        competing_seen += usize::from(competing);
        let rows = SurvivalRows {
            duration: &duration,
            event_code: &code,
            treated: &treated,
            delayed_entry: entry.as_deref(),
        };
        let endpoint = if competing {
            SurvivalEndpoint::CumulativeIncidence { target_cause: 1 }
        } else {
            SurvivalEndpoint::Survival
        };
        let got = randomized_survival_summary(&rows, case["tau"].as_f64().expect("tau"), endpoint)
            .unwrap_or_else(|e| panic!("seed {seed}: {e}"));
        let times = floats(&case["times"]);
        assert_eq!(got.times, times, "seed {seed}: event-time grid");
        for (name, got, want) in [
            ("control", &got.control, floats(&case["control"])),
            ("treated", &got.treated, floats(&case["treated_curve"])),
        ] {
            for (k, (a, b)) in got.iter().zip(&want).enumerate() {
                assert!((a - b).abs() <= 1e-14, "seed {seed} {name}[{k}]: {a} vs {b}");
            }
        }
        if competing {
            assert!(got.rmst_control.is_none() && got.rmst_treated.is_none());
        } else {
            for (got, key) in
                [(got.rmst_control, "rmst_control"), (got.rmst_treated, "rmst_treated")]
            {
                let want = case[key].as_f64().expect("rmst");
                let got = got.expect("rmst");
                assert!((got - want).abs() <= 1e-13 * want.abs().max(1.0), "seed {seed} {key}");
            }
        }
        assert_eq!(
            got.minimum_event_risk_set.map(|v| v as u64),
            case["minimum_event_risk_set"].as_u64(),
            "seed {seed}: smallest event risk set"
        );
    }
    assert!(delayed_seen >= 5 && competing_seen >= 5, "{delayed_seen} {competing_seen}");
}
