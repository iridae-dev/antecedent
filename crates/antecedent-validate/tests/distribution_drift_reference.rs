//! The PIT-window drift diagnostic against exact and 60-digit evaluation of its definitions
//! (`oracle_data/distribution_drift_reference.py`): the two-sample Kolmogorov-Smirnov distance
//! in rational arithmetic and the population stability index with its half-count pseudocount.
//!
//! Covers identical and disjoint windows, ties at bin edges and at 0 and 1, unequal window sizes,
//! a single-case window and sixty seeded random pairs, and the invariants that the distance is
//! symmetric, the index is symmetric and non-negative, and neither depends on the order of cases.

use antecedent_validate::predictive_assessment::distribution_drift;
use serde_json::Value;

fn reference() -> Vec<Value> {
    serde_json::from_str(include_str!("oracle_data/distribution_drift_reference.json"))
        .expect("frozen reference")
}

fn window(value: &Value) -> Vec<f64> {
    value
        .as_array()
        .expect("window")
        .iter()
        .map(|v| v.as_str().expect("decimal").parse::<f64>().expect("number"))
        .collect()
}

#[test]
fn ks_distance_and_population_stability_index_equal_their_definitions() {
    let cases = reference();
    assert!(cases.len() >= 60);
    for case in &cases {
        let a = window(&case["reference"]);
        let b = window(&case["comparison"]);
        let bins = case["bins"].as_u64().expect("bins") as usize;
        let got = distribution_drift(&a, &b, bins).expect("drift");
        let ks = case["ks"].as_f64().expect("ks");
        let psi: f64 = case["psi"].as_str().expect("psi").parse().expect("number");
        assert!(
            (got.ks_distance - ks).abs() <= 1e-13,
            "{}: ks {} vs {ks}",
            case["name"],
            got.ks_distance
        );
        assert!(
            (got.population_stability_index - psi).abs() <= 1e-11 * psi.abs().max(1.0),
            "{}: psi {} vs {psi}",
            case["name"],
            got.population_stability_index
        );
        assert_eq!((got.reference_cases, got.comparison_cases, got.bins), (a.len(), b.len(), bins));
    }
}

#[test]
fn swapping_the_windows_or_permuting_cases_changes_nothing_that_should_not_change() {
    for case in &reference() {
        let a = window(&case["reference"]);
        let b = window(&case["comparison"]);
        let bins = case["bins"].as_u64().expect("bins") as usize;
        let forward = distribution_drift(&a, &b, bins).expect("drift");
        let backward = distribution_drift(&b, &a, bins).expect("drift");
        assert!((forward.ks_distance - backward.ks_distance).abs() <= 1e-15);
        assert!(
            (forward.population_stability_index - backward.population_stability_index).abs()
                <= 1e-12 * forward.population_stability_index.max(1.0)
        );
        assert!(forward.population_stability_index >= 0.0);
        let mut reversed = a.clone();
        reversed.reverse();
        let again = distribution_drift(&reversed, &b, bins).expect("drift");
        assert_eq!(again.ks_distance, forward.ks_distance);
        assert_eq!(again.population_stability_index, forward.population_stability_index);
    }
}
