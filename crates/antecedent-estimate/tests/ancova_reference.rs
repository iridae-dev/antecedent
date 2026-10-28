//! Randomized ANCOVA: HC0 variance, HC2 variance and Bell-McCaffrey degrees of freedom against
//! exact rational arithmetic.
//!
//! `oracle_data/ancova_reference.py` (frozen in the JSON beside it) fits each table by exact
//! rational least squares, forms the HC0 and HC2 variances of the assignment coefficient, builds
//! the n-by-n matrix `M D M` explicitly for the Satterthwaite degrees of freedom, and solves the
//! 97.5% Student quantile at 50 digits. Cases cover the difference in means (Welch), a balanced
//! trial, two arms of 2 and 14 rows, a high-leverage covariate value, an outcome offset of 1e7
//! with a covariate offset of 1e3, near-collinear covariates and a design with a row of
//! leverage one, where HC2 is undefined and no interval is published.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::ancova::{fit_ancova, small_arm_interval_95};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-300), "{what}: got {got} want {want}");
}

#[test]
fn hc0_hc2_and_degrees_of_freedom_match_the_exact_rational_fit() {
    let reference: Value =
        serde_json::from_str(include_str!("oracle_data/ancova_reference.json")).unwrap();
    for case in reference["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let y = floats(&case["y"]);
        let t: Vec<bool> = case["t"].as_array().unwrap().iter().map(|v| v.as_bool().unwrap()).collect();
        let x: Vec<Vec<f64>> = case["x"].as_array().unwrap().iter().map(floats).collect();
        let covariates: Vec<&[f64]> = x.iter().map(Vec::as_slice).collect();
        // The offset case loses digits to the conditioning of the data, not to the formulas.
        let rel = if name.contains("offset") { 1e-5 } else if name.contains("collinear") { 1e-6 } else { 1e-9 };
        if !covariates.is_empty() {
            let fit = fit_ancova(&y, &t, &covariates).unwrap();
            close(fit.effect, case["effect"].as_f64().unwrap(), rel, name);
            close(fit.hc0_variance, case["hc0"].as_f64().unwrap(), rel, name);
            for (got, want) in fit.covariate_coefficients.iter().zip(floats(&case["covariate_coefficients"])) {
                close(*got, want, rel.max(1e-8), name);
            }
        }
        let interval = small_arm_interval_95(&y, &t, &covariates);
        if case["undefined"].as_bool().unwrap() {
            assert!(interval.is_none(), "{name}: HC2 is undefined at a row of leverage one");
            continue;
        }
        let interval = interval.unwrap_or_else(|| panic!("{name}: no interval"));
        close(interval.effect, case["effect"].as_f64().unwrap(), rel, name);
        close(interval.variance, case["hc2"].as_f64().unwrap(), rel, name);
        close(interval.degrees_of_freedom, case["df"].as_f64().unwrap(), rel, name);
        let half = (interval.interval_95[1] - interval.interval_95[0]) / 2.0;
        close(half, case["half_width"].as_f64().unwrap(), rel.max(1e-8), name);
    }
}
