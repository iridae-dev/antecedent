//! The Imbens-Manski critical value and finite-identified-set intervals against an exact
//! reference.
//!
//! `oracle_data/identified_set_reference.py` (frozen in the JSON beside it) solves
//! `Phi(c + r) - Phi(-c) = level` by root-finding in 40-digit arithmetic and rebuilds both
//! frequentist constructions from their definitions: the influence interval (centred-score
//! standard errors, the width standard error from the highest-minus-lowest completion's
//! influence, the `sqrt(ln n)` retention rule and the per-completion endpoints) and the
//! shared-replicate interval (scaled replicate standard deviations with non-finite replicates
//! dropped for every completion). The cases include a width that is retained with an interior
//! critical value, a width below the retention threshold, a single completion and dropped
//! replicates.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{
    IdentifiedSetInterval, imbens_manski_critical_value, imbens_manski_influence,
    imbens_manski_shared_replicates,
};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap_or(f64::NAN)).collect()
}

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/identified_set_reference.json")).unwrap()
}

fn close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= 1e-9 * expected.abs().max(1.0),
        "{what}: {actual} vs {expected}"
    );
}

fn check(name: &str, fit: &IdentifiedSetInterval, case: &Value) {
    assert_eq!(fit.width_retained, case["width_retained"].as_bool().unwrap(), "{name} retained");
    for (label, value) in [
        ("critical", fit.critical_value),
        ("lower", fit.lower),
        ("upper", fit.upper),
        ("lower_se", fit.lower_se),
        ("upper_se", fit.upper_se),
        ("bound_lower", fit.bound_lower),
        ("bound_upper", fit.bound_upper),
    ] {
        let key = if label == "critical" { "critical" } else { label };
        close(value, case[key].as_f64().unwrap(), &format!("{name} {label}"));
    }
}

#[test]
fn the_critical_value_solves_the_imbens_manski_equation() {
    for point in reference()["critical"].as_array().unwrap() {
        let (level, r, c) = (
            point["level"].as_f64().unwrap(),
            point["r"].as_f64().unwrap(),
            point["c"].as_f64().unwrap(),
        );
        close(imbens_manski_critical_value(level, r), c, &format!("level {level} r {r}"));
    }
}

#[test]
fn the_influence_interval_matches_the_exact_construction() {
    for case in reference()["influence"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let points = floats(&case["points"]);
        let scores: Vec<Vec<f64>> = case["scores"].as_array().unwrap().iter().map(floats).collect();
        let columns: Vec<&[f64]> = scores.iter().map(Vec::as_slice).collect();
        let fit = imbens_manski_influence(&points, &columns, case["level"].as_f64().unwrap())
            .unwrap_or_else(|| panic!("{name}"));
        check(name, &fit, case);
    }
}

#[test]
fn the_shared_replicate_interval_matches_the_exact_construction() {
    for case in reference()["replicates"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let points = floats(&case["points"]);
        let draws: Vec<Vec<f64>> = case["draws"].as_array().unwrap().iter().map(floats).collect();
        let fit = imbens_manski_shared_replicates(
            &points,
            &draws,
            case["scale"].as_f64().unwrap(),
            case["rows"].as_u64().unwrap() as usize,
            case["level"].as_f64().unwrap(),
        )
        .unwrap_or_else(|| panic!("{name}"));
        assert_eq!(fit.replicates as u64, case["kept"].as_u64().unwrap(), "{name} kept");
        check(name, &fit, case);
    }
}

#[test]
fn a_single_replicate_or_a_non_finite_point_gives_no_interval() {
    assert!(
        imbens_manski_shared_replicates(&[0.0, 1.0], &[vec![0.0, 1.0]], 1.0, 50, 0.95).is_none()
    );
    assert!(imbens_manski_influence(&[f64::NAN], &[&[0.0, 1.0, 2.0]], 0.95).is_none());
    assert!(imbens_manski_influence(&[0.0], &[&[0.0, 1.0, 2.0]], 1.0).is_none());
}
