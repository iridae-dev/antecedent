//! The joint vector-treatment fit against a 40-digit reference.
//!
//! `oracle_data/vector_treatment_reference.py` (frozen in the JSON beside it) solves the
//! ordinary least squares of a heteroskedastic 48-row design with an intercept, two adjustment
//! columns and three treatments, forms the model-based, HC0, HC1, HC2 and HC3 covariances of the
//! treatment block, three declared contrasts with their covariance-aware and naive standard
//! errors, two-sided normal p-values and Holm adjustment, and the joint and contrast Wald
//! statistics with their chi-square upper-tail probabilities.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::vector_treatment::{
    Contrast, NamedColumn, TreatmentColumn, VectorCovariance, VectorTreatmentInput,
    VectorTreatmentOptions, fit_vector_treatment,
};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("oracle_data/vector_treatment_reference.json")).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn text(v: &Value) -> f64 {
    v.as_str().unwrap().parse().unwrap()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    let tol = rel * want.abs().max(1e-300);
    assert!((got - want).abs() <= tol, "{what}: got {got} want {want} tol {tol}");
}

fn contrasts(fx: &Value) -> Vec<Contrast> {
    let names = ["t1", "t2", "t3"];
    fx["contrasts"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
        .map(|(index, row)| Contrast {
            name: format!("c{index}"),
            weights: floats(row)
                .into_iter()
                .zip(names)
                .filter(|(w, _)| *w != 0.0)
                .map(|(w, name)| (name.to_owned(), w))
                .collect(),
        })
        .collect()
}

#[test]
fn every_covariance_kind_matches_the_multiprecision_fit() {
    let fx = fixture();
    let adjustment: Vec<Vec<f64>> =
        fx["adjustment"].as_array().unwrap().iter().map(floats).collect();
    let treatments: Vec<Vec<f64>> =
        fx["treatments"].as_array().unwrap().iter().map(floats).collect();
    let declared = contrasts(&fx);
    for (key, kind) in [
        ("model_based", VectorCovariance::ModelBased),
        ("hc0", VectorCovariance::Hc0),
        ("hc1", VectorCovariance::Hc1),
        ("hc2", VectorCovariance::Hc2),
        ("hc3", VectorCovariance::Hc3),
    ] {
        let set = ["a1", "a2"];
        let input = VectorTreatmentInput {
            outcome: floats(&fx["outcome"]).into(),
            row_snapshot: "snap".into(),
            adjustment: set
                .iter()
                .zip(&adjustment)
                .map(|(name, v)| NamedColumn::new(*name, v.clone()))
                .collect(),
            treatments: ["t1", "t2", "t3"]
                .iter()
                .zip(&treatments)
                .map(|(name, v)| TreatmentColumn {
                    name: (*name).into(),
                    values: v.clone().into(),
                    adjustment_set: set.iter().map(|s| (*s).into()).collect(),
                    row_snapshot: "snap".into(),
                })
                .collect(),
        };
        let fit = fit_vector_treatment(
            &input,
            &VectorTreatmentOptions { covariance: kind, contrasts: declared.clone() },
        )
        .unwrap();
        let want = &fx["kinds"][key];
        assert_eq!(fit.residual_df, 48 - 6);
        close(fit.residual_variance, text(&want["residual_variance"]), 1e-10, key);
        for (i, c) in want["coefficients"].as_array().unwrap().iter().enumerate() {
            let got = &fit.coefficients[i];
            close(got.estimate, text(&c["estimate"]), 1e-9, key);
            close(got.standard_error, text(&c["se"]), 1e-9, key);
            close(got.z, text(&c["z"]), 1e-9, key);
            close(got.p_value, text(&c["p"]), 1e-7, key);
        }
        for (i, row) in want["covariance"].as_array().unwrap().iter().enumerate() {
            for (j, v) in row.as_array().unwrap().iter().enumerate() {
                close(fit.covariance[i * 3 + j], text(v), 1e-9, key);
            }
        }
        for (got, c) in fit.contrasts.iter().zip(want["contrasts"].as_array().unwrap()) {
            close(got.estimate, text(&c["estimate"]), 1e-9, key);
            close(got.standard_error, text(&c["se"]), 1e-9, key);
            close(got.naive_independent_standard_error, text(&c["naive_se"]), 1e-9, key);
            close(got.z, text(&c["z"]), 1e-9, key);
            close(got.p_value, text(&c["p"]), 1e-7, key);
            close(got.p_holm, text(&c["p_holm"]), 1e-7, key);
        }
        close(fit.joint_wald.statistic, text(&want["joint_wald"]["statistic"]), 1e-9, key);
        close(fit.joint_wald.p_value, text(&want["joint_wald"]["p"]), 1e-7, key);
        assert_eq!(fit.joint_wald.degrees_of_freedom, 3);
        let wald = fit.wald_test(&declared[..2]).unwrap();
        close(wald.statistic, text(&want["contrast_wald"]["statistic"]), 1e-9, key);
        close(wald.p_value, text(&want["contrast_wald"]["p"]), 1e-7, key);
        assert_eq!(wald.degrees_of_freedom, 2);
    }
}
