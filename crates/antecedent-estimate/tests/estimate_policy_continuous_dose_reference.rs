//! Inverse-density Epanechnikov dose response and fixed group-dose policy against exact rationals.
//!
//! `oracle_data/estimate_policy_continuous_dose_reference.py` (frozen in the JSON beside it)
//! evaluates, with Fractions on dyadic inputs, the local weight (3/4)(1 - s^2)/f on the open window
//! |s| < 1, the weighted response, the Kish size, the weighted outcome deviation, the per-group
//! fixed-policy value, its influence scores, the paired variances with the smallest-group
//! correction and the support diagnostics. Cases cover three unbalanced groups of 4, 9 and 30 rows
//! at an outcome offset of 1e9 with densities over four decades and doses exactly on the window
//! edge, a design where one row carries 99.99 percent of the weight, and a 2 x 350 row design in
//! which the pointwise intervals are published (known density) or withheld (density not known).
//! The refusals of an unmapped or duplicated group and of too little local support are executed.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::continuous_dose::{
    DoseRows, DoseRules, DoseWindow, conditional_dose_response, fixed_dose_policy_value,
};
use serde_json::Value;

fn num(v: &Value) -> f64 {
    v.as_str().unwrap().parse().unwrap()
}

fn nums(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(num).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!(
        (got - want).abs() <= rel * want.abs().max(1e-300),
        "{what}: got {got:.17e} want {want:.17e}"
    );
}

fn rule(v: &Value) -> Vec<(String, f64)> {
    v.as_object().unwrap().iter().map(|(g, d)| (g.clone(), num(d))).collect()
}

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!(
        "oracle_data/estimate_policy_continuous_dose_reference.json"
    ))
    .unwrap()["cases"]
        .as_array()
        .unwrap()
        .clone()
}

#[test]
fn the_dose_response_and_fixed_policy_match_the_exact_rational_reference() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let groups: Vec<String> = case["groups"]
            .as_array()
            .unwrap()
            .iter()
            .map(|g| g.as_str().unwrap().to_owned())
            .collect();
        let (dose, density, outcome) =
            (nums(&case["dose"]), nums(&case["density"]), nums(&case["outcome"]));
        let h = num(&case["bandwidth"]);
        let support = usize::try_from(case["min_local_support"].as_u64().unwrap()).unwrap();
        let targets = nums(&case["targets"]);

        let points =
            conditional_dose_response(&outcome, &dose, &groups, &density, &targets, h, support)
                .unwrap();
        let want = case["response"].as_array().unwrap();
        assert_eq!(points.len(), want.len(), "{name}: cells");
        for (got, w) in points.iter().zip(want) {
            let cell = format!("{name} {} @ {}", got.baseline_group, got.target_dose);
            assert_eq!(got.baseline_group, w["group"].as_str().unwrap());
            close(got.response, num(&w["response"]), 3e-16, &format!("{cell} response"));
            assert_eq!(got.local_rows as u64, w["local_rows"].as_u64().unwrap(), "{cell}");
            close(got.effective_sample_size, num(&w["ess"]), 1e-12, &format!("{cell} ess"));
            close(got.minimum_dose_density, num(&w["min_density"]), 1e-15, &cell);
            close(
                got.maximum_normalized_weight,
                num(&w["max_norm_weight"]),
                1e-12,
                &format!("{cell} max weight"),
            );
            close(got.local_outcome_sd, num(&w["local_sd"]), 1e-9, &format!("{cell} sd"));
        }

        let policy = rule(&case["policy"]);
        let reference = rule(&case["reference"]);
        let value = fixed_dose_policy_value(
            &DoseRows { outcome: &outcome, dose: &dose, groups: &groups, density: &density },
            &DoseRules { policy: &policy, reference: &reference },
            &DoseWindow {
                bandwidth: h,
                min_local_support: support,
                density_known: case["density_known"].as_bool().unwrap(),
            },
        )
        .unwrap();
        let f = &case["fixed"];
        // The values sit at an outcome offset of 1e9 in one case, so they are held to the
        // double-precision resolution of the offset (one unit in the last place of 1e9 is
        // 1.2e-7); the paired difference of two such values carries two of those.
        close(value.policy_value, num(&f["policy_value"]), 3e-16, &format!("{name} policy"));
        close(value.reference_value, num(&f["reference_value"]), 3e-16, &format!("{name} ref"));
        let ulp = f64::EPSILON * num(&f["policy_value"]).abs();
        assert!(
            (value.incremental_value - num(&f["incremental_value"])).abs() <= 2.0 * ulp,
            "{name}: incremental {} vs {}",
            value.incremental_value,
            num(&f["incremental_value"])
        );
        for (got, key) in [
            (value.policy_variance, "policy_variance"),
            (value.reference_variance, "reference_variance"),
            (value.incremental_variance, "incremental_variance"),
        ] {
            // One row holds 99.997 percent of the weight in one case: the score of every other row
            // is a cancellation of two values near the dominant outcome.
            let rel = if name == "one_row_dominates" { 1e-8 } else { 1e-12 };
            close(got, num(&f[key]), rel, &format!("{name} {key}"));
        }
        assert_eq!(value.minimum_local_rows as u64, f["minimum_local_rows"].as_u64().unwrap());
        close(
            value.minimum_effective_sample_size,
            num(&f["minimum_effective_sample_size"]),
            1e-12,
            name,
        );
        close(value.maximum_normalized_weight, num(&f["maximum_normalized_weight"]), 1e-12, name);
        close(value.minimum_dose_density, num(&f["minimum_dose_density"]), 1e-15, name);
        let published = f["interval_supported"].as_bool().unwrap();
        for (interval, centre, variance) in [
            (value.policy_interval_95, value.policy_value, value.policy_variance),
            (value.reference_interval_95, value.reference_value, value.reference_variance),
            (value.incremental_interval_95, value.incremental_value, value.incremental_variance),
        ] {
            assert_eq!(interval.is_some(), published, "{name}: interval publication");
            if let Some([lo, hi]) = interval {
                let radius = 1.959_963_984_540_054 * variance.sqrt();
                close(hi - lo, 2.0 * radius, 1e-12, &format!("{name} width"));
                close((hi + lo) / 2.0, centre, 1e-12, &format!("{name} centre"));
            }
        }
    }
}

#[test]
fn a_rule_that_misnames_a_group_or_a_window_with_too_little_support_is_refused() {
    let groups: Vec<String> = ["a", "a", "a", "b", "b", "b"].map(String::from).to_vec();
    let dose = [0.0, 0.1, 0.2, 0.0, 0.1, 0.2];
    let outcome = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
    let density = [1.0; 6];
    let rows = DoseRows { outcome: &outcome, dose: &dose, groups: &groups, density: &density };
    let window = DoseWindow { bandwidth: 0.5, min_local_support: 2, density_known: true };
    let full = [("a".to_owned(), 0.1), ("b".to_owned(), 0.1)];
    assert!(
        fixed_dose_policy_value(&rows, &DoseRules { policy: &full, reference: &full }, &window)
            .is_ok()
    );
    let missing = [("a".to_owned(), 0.1)];
    let duplicated = [("a".to_owned(), 0.1), ("a".to_owned(), 0.2), ("b".to_owned(), 0.1)];
    let unknown = [("a".to_owned(), 0.1), ("b".to_owned(), 0.1), ("c".to_owned(), 0.1)];
    let nonfinite = [("a".to_owned(), f64::NAN), ("b".to_owned(), 0.1)];
    for bad in [&missing[..], &duplicated[..], &unknown[..], &nonfinite[..]] {
        assert!(
            fixed_dose_policy_value(&rows, &DoseRules { policy: bad, reference: &full }, &window)
                .is_err()
        );
        assert!(
            fixed_dose_policy_value(&rows, &DoseRules { policy: &full, reference: bad }, &window)
                .is_err()
        );
    }
    // A target dose with fewer than two rows in the open window is a support failure, and a row
    // exactly on the window edge is outside it.
    let far = [("a".to_owned(), 5.0), ("b".to_owned(), 0.1)];
    assert!(
        fixed_dose_policy_value(&rows, &DoseRules { policy: &far, reference: &full }, &window)
            .is_err()
    );
    let edge = [("a".to_owned(), 0.6), ("b".to_owned(), 0.1)];
    assert!(
        fixed_dose_policy_value(&rows, &DoseRules { policy: &edge, reference: &full }, &window)
            .is_err()
    );
    assert!(conditional_dose_response(&outcome, &dose, &groups, &density, &[0.6], 0.5, 3).is_err());
}
