//! The Kaplan-Meier censoring weights and the selected-outcome pseudo-values against exact
//! references.
//!
//! `oracle_data/observation_ipcw_reference.py` (frozen in the JSON beside it) computes the
//! censoring product-limit in rational arithmetic from the definition (risk set by brute-force
//! count, left limits by scanning the jumps) for distinct times, tied times, delayed entry and an
//! administrative-censoring tail, and checks the Satten-Datta identity (the weighted observed
//! mass up to `t` equals one minus the outcome Kaplan-Meier at `t`) for distinct times. The
//! selected-outcome pseudo-values are checked here by exact enumeration: averaging the IPW and
//! the augmented pseudo-value over every one of the `2^n` selection patterns, weighted by their
//! probabilities, returns the full-data mean for any outcome model.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::{
    censoring_tail_restriction, kaplan_meier_ipcw, selected_outcome_pseudo_values,
};
use serde_json::Value;

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/observation_ipcw_reference.json"))
        .unwrap()
        .as_array()
        .cloned()
        .unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

#[test]
fn weights_and_tail_restriction_equal_the_rational_product_limit() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let (time, event) = (floats(&case["time"]), floats(&case["event"]));
        let entry = case["entry"].as_array().map(|_| floats(&case["entry"]));
        let floor = case["floor"].as_f64().unwrap();
        let got = kaplan_meier_ipcw(&time, &event, entry.as_deref(), floor).unwrap();
        for (i, (g, w)) in got.weights.iter().zip(floats(&case["weights"])).enumerate() {
            assert!((g - w).abs() <= 1e-13 * w.abs().max(1.0), "{name} row {i}: {g} vs {w}");
        }
        let want_tail = case["tail_restriction"].as_f64();
        assert_eq!(got.tail_restriction, want_tail, "{name}");
        if entry.is_none() {
            assert_eq!(
                censoring_tail_restriction(&time, &event, floor).unwrap(),
                want_tail,
                "{name}"
            );
        }
        if let Some(checks) = case["satten_datta"].as_array() {
            // The weights reproduce one minus the outcome Kaplan-Meier at every observed time.
            let n = time.len() as f64;
            for check in checks {
                let t = check["t"].as_f64().unwrap();
                let mass: f64 = got
                    .weights
                    .iter()
                    .zip(time.iter().zip(&event))
                    .filter(|(_, (ti, di))| **ti <= t && **di == 1.0)
                    .map(|(w, _)| w)
                    .sum::<f64>()
                    / n;
                let want = check["one_minus_km"].as_f64().unwrap();
                assert!((mass - want).abs() < 1e-12, "{name} t={t}: {mass} vs {want}");
            }
        }
    }
}

#[test]
fn delayed_entry_weights_divide_the_left_limit_at_entry_by_the_left_limit_at_the_time() {
    let case = &cases()[2];
    let (time, event) = (floats(&case["time"]), floats(&case["event"]));
    let entry = floats(&case["entry"]);
    let with = kaplan_meier_ipcw(&time, &event, Some(&entry), 0.01).unwrap();
    // Entry before every censoring time reduces to the no-entry weights.
    let early = vec![-1.0e9; time.len()];
    let none = kaplan_meier_ipcw(&time, &event, None, 0.01).unwrap();
    let early_entry = kaplan_meier_ipcw(&time, &event, Some(&early), 0.01).unwrap();
    for (a, b) in none.weights.iter().zip(&early_entry.weights) {
        assert!((a - b).abs() < 1e-14);
    }
    // A late entrant is up-weighted by the survival to its entry, so no weight is below one.
    assert!(with.weights.iter().all(|w| *w == 0.0 || *w >= 1.0 - 1e-12));
}

#[test]
fn selected_outcome_pseudo_values_are_exactly_unbiased_over_every_selection_pattern() {
    let y = [2.0, -1.0, 0.5, 4.0, 3.0, -2.5];
    let p = [0.2, 0.5, 0.8, 0.35, 0.6, 0.9];
    let m = [1.0, 0.0, -0.5, 2.0, 5.0, 0.3];
    let n = y.len();
    let target = y.iter().sum::<f64>() / n as f64;
    for model in [None, Some(&m[..])] {
        let mut expected = vec![0.0; n];
        let mut total_probability = 0.0;
        for pattern in 0..(1_u32 << n) {
            let indicator: Vec<f64> = (0..n).map(|i| f64::from((pattern >> i) & 1)).collect();
            let probability: f64 =
                (0..n).map(|i| if indicator[i] == 1.0 { p[i] } else { 1.0 - p[i] }).product();
            total_probability += probability;
            // Unselected outcomes are ignored: scramble them to prove it.
            let observed: Vec<f64> =
                (0..n).map(|i| if indicator[i] == 1.0 { y[i] } else { f64::NAN }).collect();
            let pseudo = selected_outcome_pseudo_values(&observed, &indicator, &p, model).unwrap();
            for (e, v) in expected.iter_mut().zip(&pseudo) {
                *e += probability * v;
            }
        }
        assert!((total_probability - 1.0).abs() < 1e-12);
        for (e, truth) in expected.iter().zip(&y) {
            assert!((e - truth).abs() < 1e-12, "row expectation {e} vs {truth}");
        }
        assert!((expected.iter().sum::<f64>() / n as f64 - target).abs() < 1e-12);
    }
}

#[test]
fn pseudo_values_match_the_closed_forms_and_refuse_invalid_inputs() {
    let observed = [3.0, 99.0, 1.0];
    let indicator = [1.0, 0.0, 1.0];
    let p = [0.5, 0.25, 0.2];
    let m = [2.0, 1.5, 0.0];
    let ipw = selected_outcome_pseudo_values(&observed, &indicator, &p, None).unwrap();
    assert_eq!(ipw, vec![6.0, 0.0, 5.0]);
    let aipw = selected_outcome_pseudo_values(&observed, &indicator, &p, Some(&m)).unwrap();
    assert_eq!(aipw, vec![2.0 + (3.0 - 2.0) / 0.5, 1.5, 0.0 + (1.0 - 0.0) / 0.2]);
    for bad in [[0.0, 0.5, 0.2], [1.5, 0.5, 0.2], [f64::NAN, 0.5, 0.2]] {
        assert!(selected_outcome_pseudo_values(&observed, &indicator, &bad, None).is_err());
    }
    assert!(selected_outcome_pseudo_values(&observed, &[1.0, 0.5, 1.0], &p, None).is_err());
    assert!(selected_outcome_pseudo_values(&[f64::NAN, 0.0, 1.0], &indicator, &p, None).is_err());
}
