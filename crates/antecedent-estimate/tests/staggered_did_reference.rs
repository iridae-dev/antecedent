//! Cohort-by-period difference-in-differences contrasts against an exact rational reference.
//!
//! `oracle_data/staggered_did_reference.py` (frozen in the JSON beside it) forms every
//! cohort-by-period contrast as a difference of mean changes from the baseline period over the
//! units observed at both periods, with the cluster sandwich of the difference of two means,
//! in `Fraction` arithmetic. Cases cover three cohorts, clusters that span the treated and
//! never-treated groups, an outcome offset of 1e9 with effects of order 1e-3, two clusters
//! per group, and unbalanced panels read as available cases.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::staggered_event_study::{estimate, estimate_with, group_time_att};
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
    serde_json::from_str::<Value>(include_str!("oracle_data/staggered_did_reference.json")).unwrap()
        ["cases"]
        .as_array()
        .unwrap()
        .clone()
}

#[test]
fn event_time_contrasts_match_the_exact_rational_reference() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let p = panel(&case);
        let available = case["available_case"].as_bool().unwrap();
        let got = estimate_with(&p.y, &p.unit, &p.period, &p.cohort, &p.cluster, available)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        let want = case["event"].as_array().unwrap();
        assert_eq!(got.len(), want.len(), "{name}: contrast count");
        for (g, w) in got.iter().zip(want) {
            let tag = format!("{name} cohort {} period {}", g.cohort, g.period);
            assert_eq!(g.cohort, w["cohort"].as_i64().unwrap(), "{tag}");
            assert_eq!(g.period, w["period"].as_i64().unwrap(), "{tag}");
            assert_eq!(g.event_time, w["event_time"].as_i64().unwrap(), "{tag}");
            assert_eq!(g.treated_subjects, w["treated"].as_u64().unwrap() as usize, "{tag}");
            assert_eq!(g.comparison_subjects, w["comparison"].as_u64().unwrap() as usize, "{tag}");
            assert_eq!(g.clusters, w["clusters"].as_u64().unwrap() as usize, "{tag}");
            close(g.effect, w["effect"].as_f64().unwrap(), &format!("{tag} effect"));
            close(g.standard_error, w["se"].as_f64().unwrap(), &format!("{tag} se"));
        }
        if !available {
            let balanced = estimate(&p.y, &p.unit, &p.period, &p.cohort, &p.cluster).unwrap();
            assert_eq!(balanced, got, "{name}: estimate is estimate_with(false)");
        }
    }
}

#[test]
fn group_time_cells_match_the_exact_rational_reference() {
    for case in cases().into_iter().filter(|c| c.get("group_time").is_some()) {
        let name = case["name"].as_str().unwrap();
        let p = panel(&case);
        let got = group_time_att(&p.y, &p.unit, &p.period, &p.cohort, &p.cluster).unwrap();
        let want = case["group_time"].as_array().unwrap();
        assert_eq!(got.len(), want.len(), "{name}");
        for (g, w) in got.iter().zip(want) {
            let tag = format!("{name} cohort {} period {}", g.cohort, g.period);
            assert_eq!(
                (g.cohort, g.period),
                (w["cohort"].as_i64().unwrap(), w["period"].as_i64().unwrap())
            );
            assert_eq!(g.treated_subjects, w["treated"].as_u64().unwrap() as usize, "{tag}");
            assert_eq!(g.clusters, w["clusters"].as_u64().unwrap() as usize, "{tag}");
            close(g.effect, w["effect"].as_f64().unwrap(), &format!("{tag} effect"));
            close(g.standard_error, w["se"].as_f64().unwrap(), &format!("{tag} se"));
        }
    }
}

#[test]
fn a_missing_unit_period_refuses_without_available_case_and_one_cluster_group_refuses() {
    let case = cases().into_iter().find(|c| c["name"] == "unbalanced_available_case").unwrap();
    let p = panel(&case);
    assert!(estimate(&p.y, &p.unit, &p.period, &p.cohort, &p.cluster).is_err());
    assert!(group_time_att(&p.y, &p.unit, &p.period, &p.cohort, &p.cluster).is_err());
    // Every treated unit in one cluster: the sandwich is undefined and is refused.
    let case = cases().into_iter().find(|c| c["name"] == "two_clusters_per_group").unwrap();
    let mut p = panel(&case);
    for (cluster, cohort) in p.cluster.iter_mut().zip(&p.cohort) {
        if *cohort == 3 {
            *cluster = "all".into();
        }
    }
    assert!(estimate(&p.y, &p.unit, &p.period, &p.cohort, &p.cluster).is_err());
}
