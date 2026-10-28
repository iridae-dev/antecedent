//! Cohort-by-period contrasts on serially dependent panels against an exact rational reference.
//!
//! `oracle_data/staggered_did_serial_reference.py` (frozen in the JSON beside it) draws every
//! unit's outcomes around its own level and slope with stationary AR(1) errors (coefficients 0.6
//! and 0.9), then forms each cohort-by-period contrast and its cluster sandwich in `Fraction`
//! arithmetic. A contrast reads a unit only through the unit's own change between the baseline
//! and the period, so serial dependence inside a unit enters the variance through that change;
//! the product must reproduce the exact effect and standard error on these panels, including an
//! unbalanced one read as available cases.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::staggered_event_study::{estimate_with, group_time_att};
use serde_json::Value;

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 1e-9 * want.abs().max(1e-300), "{what}: got {got} want {want}");
}

struct Panel {
    y: Vec<f64>,
    unit: Vec<String>,
    period: Vec<i64>,
    cohort: Vec<i64>,
    cluster: Vec<String>,
}

fn panel(case: &Value) -> Panel {
    let mut p = Panel { y: vec![], unit: vec![], period: vec![], cohort: vec![], cluster: vec![] };
    for row in case["rows"].as_array().unwrap() {
        p.y.push(row[0].as_f64().unwrap());
        p.unit.push(row[1].as_str().unwrap().into());
        p.period.push(row[2].as_i64().unwrap());
        p.cohort.push(row[3].as_i64().unwrap());
        p.cluster.push(row[4].as_str().unwrap().into());
    }
    p
}

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/staggered_did_serial_reference.json"))
        .unwrap()["cases"]
        .as_array()
        .unwrap()
        .clone()
}

#[test]
fn cells_01_staggered_contrasts_on_ar1_panels_match_the_exact_rational_reference() {
    let all = cases();
    assert_eq!(all.len(), 3);
    for case in all {
        let name = case["name"].as_str().unwrap();
        let p = panel(&case);
        let available = case["available_case"].as_bool().unwrap();
        let got = estimate_with(&p.y, &p.unit, &p.period, &p.cohort, &p.cluster, available)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let want = case["event"].as_array().unwrap();
        assert_eq!(got.len(), want.len(), "{name}: contrast count");
        for (g, w) in got.iter().zip(want) {
            let tag = format!("{name} cohort {} period {}", g.cohort, g.period);
            assert_eq!(
                (g.cohort, g.period),
                (w["cohort"].as_i64().unwrap(), w["period"].as_i64().unwrap()),
                "{tag}"
            );
            assert_eq!(g.treated_subjects, w["treated"].as_u64().unwrap() as usize, "{tag}");
            assert_eq!(g.comparison_subjects, w["comparison"].as_u64().unwrap() as usize, "{tag}");
            assert_eq!(g.clusters, w["clusters"].as_u64().unwrap() as usize, "{tag}");
            close(g.effect, w["effect"].as_f64().unwrap(), &format!("{tag} effect"));
            close(g.standard_error, w["se"].as_f64().unwrap(), &format!("{tag} se"));
        }
        if let Some(group_time) = case.get("group_time") {
            let got = group_time_att(&p.y, &p.unit, &p.period, &p.cohort, &p.cluster).unwrap();
            let want = group_time.as_array().unwrap();
            assert_eq!(got.len(), want.len(), "{name}: group-time count");
            for (g, w) in got.iter().zip(want) {
                let tag = format!("{name} group-time cohort {} period {}", g.cohort, g.period);
                close(g.effect, w["effect"].as_f64().unwrap(), &format!("{tag} effect"));
                close(g.standard_error, w["se"].as_f64().unwrap(), &format!("{tag} se"));
            }
        }
    }
}
