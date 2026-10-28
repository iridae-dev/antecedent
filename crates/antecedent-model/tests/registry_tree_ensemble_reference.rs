//! Tree-ensemble mechanism slots against exactly evaluated frozen tree maps.
//!
//! The ensembles are grown by the pinned providers (forust-ml, smartcore) and exported to a
//! provider-independent tree map; this file does not test that growth. It tests what the
//! repository itself does with a map, against `oracle_data/registry_tree_ensemble_reference.py`
//! (frozen in the JSON beside it, exact rationals plus a 60-digit sigmoid): the walk of each tree
//! (strict and inclusive thresholds, values equal to a threshold, NaN down the missing branch, a
//! leaf that casts no vote), the sum and the valid-vote mean, the logistic link and the
//! probability clip, the Gaussian density of the additive-noise slot, the renormalised
//! one-vs-rest class probabilities of the categorical slot (binary and three levels), its
//! `2^-52` floor, and `-inf` outside the support.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_learn::PortablePredictor;
use antecedent_model::batch::MechanismWorkspace;
use antecedent_model::{
    MechanismSlot, ParentBatch, TreeEnsemble, evaluate_column, log_prob_column,
};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn ensemble(v: &Value) -> TreeEnsemble {
    let predictor: PortablePredictor = serde_json::from_value(v.clone()).unwrap();
    TreeEnsemble::new(predictor).unwrap()
}

/// Column-major parent values of the reference rows (`null` is NaN).
fn column_major(rows: &Value) -> (Vec<f64>, usize) {
    let rows = rows.as_array().unwrap();
    let n = rows.len();
    let mut values = vec![0.0; n * 2];
    for (r, row) in rows.iter().enumerate() {
        for (c, v) in row.as_array().unwrap().iter().enumerate() {
            values[c * n + r] = v.as_f64().unwrap_or(f64::NAN);
        }
    }
    (values, n)
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!(
        got == want || (got - want).abs() <= rel * want.abs().max(1.0),
        "{what}: {got} vs {want}"
    );
}

#[test]
fn registry_tree_ensemble_reference_maps_densities_and_class_probabilities_match_exact_values() {
    let d: Value =
        serde_json::from_str(include_str!("oracle_data/registry_tree_ensemble_reference.json"))
            .unwrap();
    let (values, n) = column_major(&d["rows"]);
    let batch = ParentBatch { n_rows: n, n_parents: 2, values: &values };

    for case in d["regression"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let ens = ensemble(&case["predictor"]);
        let mut got = vec![0.0; n];
        ens.predict_column(batch, &mut got).unwrap();
        for (r, w) in floats(&case["predictions"]).into_iter().enumerate() {
            close(got[r], w, 1e-15, &format!("{name}: row {r}"));
        }
    }

    // Additive-noise slot: evaluation adds the noise; the density is Gaussian about the map.
    for case in d["additive_density"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let slot = MechanismSlot::TreeAdditive {
            ensemble: ensemble(&case["predictor"]),
            sigma: case["sigma"].as_f64().unwrap(),
        };
        let observations = floats(&case["observations"]);
        let mut lp = vec![0.0; n];
        log_prob_column(&slot, &observations, batch, &mut lp).unwrap();
        for (r, w) in floats(&case["log_density"]).into_iter().enumerate() {
            close(lp[r], w, 1e-12, &format!("{name}: log density row {r}"));
        }
        let mut mean = vec![0.0; n];
        evaluate_column(&slot, batch, &vec![0.0; n], &mut mean, &mut MechanismWorkspace::default())
            .unwrap();
        for (r, w) in floats(&case["predictions"]).into_iter().enumerate() {
            close(mean[r], w, 1e-15, &format!("{name}: evaluation row {r}"));
        }
    }

    // Binary categorical slot: one classifier for the second level.
    for case in d["binary"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let support = floats(&case["support"]);
        let slot = MechanismSlot::TreeCategorical {
            support: support.clone().into(),
            probs: vec![0.5, 0.5].into(),
            classifiers: vec![ensemble(&case["predictor"])].into(),
        };
        let wanted = case["log_probability"].as_array().unwrap();
        for (level, value) in support.iter().enumerate() {
            let mut lp = vec![0.0; n];
            log_prob_column(&slot, &vec![*value; n], batch, &mut lp).unwrap();
            for r in 0..n {
                close(
                    lp[r],
                    wanted[r].as_array().unwrap()[level].as_f64().unwrap(),
                    // 1 - (1 - 1e-9) is formed in binary64: the complement of a clipped score
                    // carries the rounding of the score (a relative 3e-8 near 1e-9).
                    1e-8,
                    &format!("{name}: level {level} row {r}"),
                );
            }
        }
        // A value outside the support has no probability.
        let mut lp = vec![0.0; n];
        log_prob_column(&slot, &vec![15.0; n], batch, &mut lp).unwrap();
        assert!(lp.iter().all(|v| *v == f64::NEG_INFINITY), "{name}: outside the support");
        // A map that reaches exactly 0 or 1 scores the explicit floor, not minus infinity; the
        // provider maps never do (they clip at 1e-9).
        if name == "unclipped_indicator" {
            let floors = wanted
                .iter()
                .flat_map(|row| row.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()))
                .filter(|v| *v < -30.0)
                .count();
            assert!(floors > 0, "the indicator map must reach the floor");
            for row in wanted {
                for v in row.as_array().unwrap() {
                    let v = v.as_f64().unwrap();
                    assert!(v >= f64::EPSILON.ln(), "never below the floor: {v}");
                }
            }
        }
    }

    // Three levels: one-vs-rest classifiers renormalised to a distribution.
    for case in d["three_level"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let support = floats(&case["support"]);
        let classifiers: Vec<TreeEnsemble> =
            case["predictors"].as_array().unwrap().iter().map(ensemble).collect();
        let slot = MechanismSlot::TreeCategorical {
            support: support.clone().into(),
            probs: vec![1.0 / 3.0; 3].into(),
            classifiers: classifiers.into(),
        };
        let wanted = case["log_probability"].as_array().unwrap();
        let probabilities = case["probabilities"].as_array().unwrap();
        for (level, value) in support.iter().enumerate() {
            let mut lp = vec![0.0; n];
            log_prob_column(&slot, &vec![*value; n], batch, &mut lp).unwrap();
            for r in 0..n {
                close(
                    lp[r],
                    wanted[r].as_array().unwrap()[level].as_f64().unwrap(),
                    // 1 - (1 - 1e-9) is formed in binary64: the complement of a clipped score
                    // carries the rounding of the score (a relative 3e-8 near 1e-9).
                    1e-8,
                    &format!("{name}: level {level} row {r}"),
                );
            }
        }
        for row in probabilities {
            let total: f64 = floats(row).iter().sum();
            assert!((total - 1.0).abs() < 1e-14, "{name}: a distribution");
        }
    }
}
