//! Quantile inversion of an interventional CDF against an exact rational reference.
//!
//! `oracle_data/quantile_inversion_reference.py` (frozen in the JSON beside it) finds the
//! quantile of a piecewise-linear CDF by scanning its segments in exact rational arithmetic
//! and differentiates the whole map with dual numbers. The cases cover grid units from 1e-6
//! to 1e6, a level at and just above a grid CDF value, a CDF plateau, both end segments, a
//! near-jump bracket and brackets either side of the density guard, with weighted score
//! tables at weight units from 1e-200 to 1e200.

use antecedent_estimate::quantile::{invert_supported_cdf, quantile_arm};
use antecedent_estimate::scores::{ScoreColumn, ScoreTable};
use serde_json::Value;
use std::sync::Arc;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/quantile_inversion_reference.json"))
        .expect("frozen reference parses")
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().expect("array").iter().map(|x| x.as_f64().expect("number")).collect()
}

fn close(got: f64, want: f64, rel: f64, abs: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs() + abs, "{what}: {got} vs {want}");
}

#[test]
fn direct_inversion_matches_exact_value_influence_and_refusals() {
    let data = reference();
    for case in data["direct"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let x = floats(&case["x"]);
        let cdf = floats(&case["cdf"]);
        let phi: Vec<Vec<f64>> = case["phi"].as_array().expect("phi").iter().map(floats).collect();
        let tau = case["tau"].as_f64().expect("tau");
        let supported = vec![true; x.len()];
        let got = invert_supported_cdf(&x, &cdf, &phi, &supported, tau);
        match case["expect"].as_str().expect("expect") {
            "ok" => {
                let arm = got.unwrap_or_else(|e| panic!("{name}: {e}"));
                let span = x[x.len() - 1] - x[0];
                close(arm.value, case["value"].as_f64().unwrap(), 1e-11, 1e-12 * span, name);
                close(arm.density, case["density"].as_f64().unwrap(), 1e-11, 0.0, name);
                let want = floats(&case["influence"]);
                assert_eq!(arm.influence.len(), want.len(), "{name}");
                let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
                for (g, w) in arm.influence.iter().zip(&want) {
                    close(*g, *w, 1e-9, 1e-12 * scale, name);
                }
            }
            "range" => {
                let e = got.expect_err(name).to_string();
                assert!(e.contains("CDF range"), "{name}: {e}");
            }
            "density" => {
                let e = got.expect_err(name).to_string();
                assert!(e.contains("density"), "{name}: {e}");
            }
            other => panic!("{name}: unknown expectation {other}"),
        }
    }
}

#[test]
fn unsupported_bracket_and_projection_are_refused_where_the_reference_has_none() {
    let x = [0.0, 1.0, 2.0, 3.0, 4.0, 5.0];
    let phi = vec![vec![0.01, -0.01]; 6];
    let ok = [true; 6];
    // A monotone CDF with an unsupported bracket endpoint is refused; elsewhere it is not.
    let cdf = [0.1, 0.3, 0.5, 0.7, 0.85, 0.95];
    let mut bad = ok;
    bad[3] = false;
    let e = invert_supported_cdf(&x, &cdf, &phi, &bad, 0.6).unwrap_err().to_string();
    assert_eq!(
        e,
        "quantile inversion refused: insufficient tail or overlap support at the bracket"
    );
    assert!(invert_supported_cdf(&x, &cdf, &phi, &bad, 0.4).is_ok());
    // A violation away from the bracket is projected without touching the bracket.
    let raw = [0.1, 0.3, 0.7, 0.6, 0.65, 0.9];
    let arm = invert_supported_cdf(&x, &raw, &phi, &ok, 0.2).expect("bracket untouched");
    assert!((arm.value - 0.5).abs() < 1e-12, "{}", arm.value);
    // A violation inside the bracket is refused, whichever side of it the level falls on.
    for tau in [0.4, 0.5, 0.65] {
        let e = invert_supported_cdf(&x, &raw, &phi, &ok, tau).unwrap_err().to_string();
        assert_eq!(
            e, "quantile uncertainty refused: isotonic projection changes the inversion bracket",
            "tau {tau}"
        );
    }
    // The pooled plateau (0.65) is exact at its last point, so the last segment is untouched.
    assert!(invert_supported_cdf(&x, &raw, &phi, &ok, 0.8).is_ok());
}

#[test]
fn weighted_score_table_matches_the_exact_ratio_estimator_derivative() {
    let data = reference();
    for case in data["table"].as_array().expect("cases") {
        let name = case["name"].as_str().expect("name");
        let thresholds = floats(&case["thresholds"]);
        let outcome = floats(&case["outcome"]);
        let n = outcome.len();
        let columns: Vec<Vec<f64>> =
            case["columns"].as_array().unwrap().iter().map(floats).collect();
        // Column-major storage: one block of n scores per column.
        let scores: Vec<f64> = columns.iter().flatten().copied().collect();
        let table = ScoreTable {
            observed_arm: vec![0; n].into(),
            propensities: vec![0.5; columns.len() * n].into(),
            observed_outcome: outcome.into(),
            n_rows: n,
            row_index: (0..n)
                .map(|i| antecedent_data::GlobalId::new(u64::try_from(i).unwrap()))
                .collect(),
            fold_ids: (0..n).map(|i| u32::try_from(i % 2).unwrap()).collect(),
            n_folds: 2,
            scores: scores.into(),
            columns: thresholds
                .iter()
                .map(|t| ScoreColumn { arm: 0, threshold: Some(*t) })
                .collect::<Vec<_>>()
                .into(),
            adjustment_set: Arc::from([]),
            nuisance_provenance: Arc::from("reference"),
            propensity_clip: None,
            treatment: antecedent_core::VariableId::from_raw(0),
            intervened: Arc::from([]),
            preparation: None,
        };
        let weights = if case["weights"].is_null() { None } else { Some(floats(&case["weights"])) };
        let tau = case["tau"].as_f64().unwrap();
        let arm = quantile_arm(&table, weights.as_deref(), tau, 0)
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        close(arm.value, case["value"].as_f64().unwrap(), 1e-11, 1e-12, name);
        let want = floats(&case["influence"]);
        let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        for (g, w) in arm.influence.iter().zip(&want) {
            close(*g, *w, 1e-9, 1e-11 * scale, name);
        }
    }
}
