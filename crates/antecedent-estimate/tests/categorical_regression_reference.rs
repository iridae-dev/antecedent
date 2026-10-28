//! Dummy-coded categorical regression against an exact rational reference.
//!
//! `oracle_data/categorical_regression_reference.py` (frozen in the JSON beside it) solves the
//! least-squares problem with an adjustment column in `Fraction` arithmetic for model-based, HC0,
//! HC1, HC2 and HC3 covariances, and forms the level and pairwise contrasts, their normal
//! p-values (50-digit mpmath), the Holm adjustment over the declared family, the chi-square
//! omnibus and the oriented minimum-step monotonicity statistic. Cases cover unbalanced levels,
//! a reference inside the level list, an outcome offset of 1e8, an adjustment nearly equal to a
//! level indicator and levels with two rows.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::categorical_treatment::{
    CategoricalTreatmentInput, CategoricalTreatmentSpec, LevelScale, MonotonicityDirection,
    fit_categorical_treatment,
};
use antecedent_estimate::vector_treatment::{NamedColumn, VectorCovariance};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!(
        (got - want).abs() <= rel * want.abs().max(1e-300),
        "{what}: got {got:e} want {want:e}"
    );
}

/// Relative to a scale, for entries that may be exactly zero in exact arithmetic.
fn close_scaled(got: f64, want: f64, scale: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * scale, "{what}: got {got:e} want {want:e}");
}

#[test]
fn categorical_fit_matches_the_exact_regression_for_every_covariance() {
    let reference: Value =
        serde_json::from_str(include_str!("oracle_data/categorical_regression_reference.json"))
            .unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        // Conditioning: the near-collinear adjustment and the 1e8 offset cost digits.
        let tol = match name {
            "near_collinear_adjustment" => 1e-6,
            "outcome_offset" => 3e-6,
            _ => 1e-10,
        };
        let strings = |v: &Value| -> Vec<String> {
            v.as_array().unwrap().iter().map(|s| s.as_str().unwrap().to_string()).collect()
        };
        let adjustment = case["adjustment"]
            .as_array()
            .map(|_| vec![NamedColumn::new("g", floats(&case["adjustment"]))])
            .unwrap_or_default();
        let input = CategoricalTreatmentInput {
            outcome: floats(&case["outcomes"]).into(),
            row_snapshot: "snap".into(),
            adjustment,
            levels: strings(&case["row_levels"]).into(),
        };
        let covariance = match case["covariance"].as_str().unwrap() {
            "model_based" => VectorCovariance::ModelBased,
            "hc0" => VectorCovariance::Hc0,
            "hc1" => VectorCovariance::Hc1,
            "hc2" => VectorCovariance::Hc2,
            _ => VectorCovariance::Hc3,
        };
        let monotonicity = match case["monotonicity"].as_str() {
            Some("non_decreasing") => Some(MonotonicityDirection::NonDecreasing),
            Some(_) => Some(MonotonicityDirection::NonIncreasing),
            None => None,
        };
        let pairwise: Vec<(String, String)> = case["pairs"]
            .as_array()
            .unwrap()
            .iter()
            .map(|p| {
                let p = strings(p);
                (p[0].clone(), p[1].clone())
            })
            .collect();
        let spec = CategoricalTreatmentSpec {
            declared_levels: strings(&case["levels"]),
            scale: LevelScale::Ordered,
            reference: case["reference"].as_str().unwrap().to_string(),
            min_level_rows: 2,
            pairwise,
            monotonicity,
            covariance,
        };
        let fit =
            fit_categorical_treatment(&input, &spec).unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let want = &case["expected"];
        // coefficients: [intercept, adjustment?, level dummies]; the fit reports the dummies
        let all = floats(&want["coefficients"]);
        let dummies = &all[all.len() - fit.coefficients.len()..];
        let p = (all.len() as f64).sqrt();
        let _ = p;
        for (c, w) in fit.coefficients.iter().zip(dummies) {
            close(c.estimate, *w, tol, &format!("{name} coefficient {}", c.name));
        }
        let wcov = floats(&want["covariance"]);
        let full = (wcov.len() as f64).sqrt() as usize;
        let k = fit.coefficients.len();
        let scale =
            (0..k).map(|i| wcov[(full - k + i) * full + full - k + i].abs()).fold(0.0, f64::max);
        for i in 0..k {
            for j in 0..k {
                close_scaled(
                    fit.covariance[i * k + j],
                    wcov[(full - k + i) * full + full - k + j],
                    scale,
                    tol,
                    &format!("{name} covariance[{i}][{j}]"),
                );
            }
        }
        let contrasts = want["contrasts"].as_array().unwrap();
        let level_count = fit.level_contrasts.len();
        for (c, w) in fit.level_contrasts.iter().zip(contrasts) {
            let what = format!("{name} level {}", c.level);
            close(c.estimate, w["estimate"].as_f64().unwrap(), tol, &what);
            close(c.standard_error, w["variance"].as_f64().unwrap().sqrt(), tol, &what);
            close(c.z, w["z"].as_f64().unwrap(), tol, &what);
            close(c.p_value, w["p"].as_f64().unwrap(), 1e-6_f64.max(tol), &what);
            close(c.p_holm, w["p_holm"].as_f64().unwrap(), 1e-6_f64.max(tol), &what);
        }
        for (c, w) in fit.pairwise.iter().zip(&contrasts[level_count..]) {
            let what = format!("{name} pair {}->{}", c.from, c.to);
            close(c.estimate, w["estimate"].as_f64().unwrap(), tol, &what);
            close(c.standard_error, w["variance"].as_f64().unwrap().sqrt(), tol, &what);
            close(c.p_value, w["p"].as_f64().unwrap(), 1e-6_f64.max(tol), &what);
            close(c.p_holm, w["p_holm"].as_f64().unwrap(), 1e-6_f64.max(tol), &what);
        }
        close(
            fit.omnibus.statistic,
            want["omnibus"]["statistic"].as_f64().unwrap(),
            tol,
            &format!("{name} omnibus"),
        );
        close(
            fit.omnibus.p_value,
            want["omnibus"]["p"].as_f64().unwrap(),
            1e-6_f64.max(tol),
            &format!("{name} omnibus p"),
        );
        match (&fit.monotonicity, want.get("monotonicity")) {
            (Some(m), Some(w)) => {
                close(m.statistic, w["statistic"].as_f64().unwrap(), tol, &format!("{name} T"));
                close(
                    m.p_value,
                    w["p"].as_f64().unwrap(),
                    1e-6_f64.max(tol),
                    &format!("{name} mono p"),
                );
                for (s, w) in m.steps.iter().zip(floats(&w["steps"])) {
                    close(s.difference, w, tol, &format!("{name} step"));
                }
            }
            (None, None) => {}
            _ => panic!("{name}: monotonicity presence disagrees"),
        }
    }
}
