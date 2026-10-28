//! The retained adjusted-model contrast against a 50-digit refit and delta-method reference.
//!
//! `oracle_data/adjusted_contrast_reference.py` refits each model from the data in multiprecision
//! arithmetic (OLS with model-based covariance, binomial-logit and Poisson-log by Newton's method
//! with the inverse Fisher information) and forms the response-scale contrast over the target law
//! with its gradient and standard error; the product's fit and contrast are compared to it.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::adjustment_resume::AdjustedFit;
use antecedent_estimate::vector_treatment::{
    NamedColumn, TreatmentColumn, VectorCovariance, VectorTreatmentInput, VectorTreatmentOptions,
};
use antecedent_stats::{GlmFamily, GlmOptions};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    let tol = rel * want.abs().max(1e-12);
    assert!((got - want).abs() <= tol, "{what}: got {got} want {want} tol {tol}");
}

fn input(case: &Value) -> VectorTreatmentInput<'static> {
    let adjustment = case["adjustment"].as_array().unwrap();
    let treatments = case["treatments"].as_array().unwrap();
    VectorTreatmentInput {
        outcome: floats(&case["outcome"]).into(),
        row_snapshot: "snap".into(),
        adjustment: adjustment
            .iter()
            .enumerate()
            .map(|(i, z)| NamedColumn::new(format!("z{i}"), floats(z)))
            .collect(),
        treatments: treatments
            .iter()
            .enumerate()
            .map(|(i, t)| TreatmentColumn {
                name: format!("t{i}").into(),
                values: floats(t).into(),
                adjustment_set: (0..adjustment.len()).map(|i| format!("z{i}").into()).collect(),
                row_snapshot: "snap".into(),
            })
            .collect(),
    }
}

#[test]
fn contrast_effect_and_delta_se_match_the_multiprecision_reference() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("oracle_data/adjusted_contrast_reference.json")).unwrap();
    assert_eq!(cases.len(), 5);
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let data = input(case);
        let fit = match case["family"].as_str().unwrap() {
            "gaussian" => AdjustedFit::linear(
                &data,
                &VectorTreatmentOptions { covariance: VectorCovariance::ModelBased, contrasts: vec![] },
            ),
            "logit" => AdjustedFit::glm(&data, GlmFamily::BinomialLogit, &GlmOptions::default()),
            "poisson" => AdjustedFit::glm(&data, GlmFamily::PoissonLog, &GlmOptions::default()),
            other => panic!("{other}"),
        }
        .unwrap_or_else(|e| panic!("{name}: {e}"));
        let weights = case["weights"].as_array().map(|_| floats(&case["weights"]));
        let (effect, se) = fit
            .contrast(&floats(&case["active"]), &floats(&case["control"]), weights.as_deref())
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        close(effect, case["effect"].as_f64().unwrap(), 1e-6, &format!("{name} effect"));
        close(se, case["se"].as_f64().unwrap(), 1e-5, &format!("{name} se"));
    }
}

#[test]
fn contrast_refuses_a_mismatched_dimension_and_invalid_target_weights() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("oracle_data/adjusted_contrast_reference.json")).unwrap();
    let case = &cases[2];
    let data = input(case);
    let fit = AdjustedFit::glm(&data, GlmFamily::BinomialLogit, &GlmOptions::default()).unwrap();
    assert!(fit.contrast(&[1.0, 1.0], &[0.0, 0.0], None).is_err());
    let n = data.outcome.len();
    assert!(fit.contrast(&[1.0], &[0.0], Some(&vec![1.0; n - 1])).is_err());
    let mut negative = vec![1.0; n];
    negative[0] = -1.0;
    assert!(fit.contrast(&[1.0], &[0.0], Some(&negative)).is_err());
    assert!(fit.contrast(&[1.0], &[0.0], Some(&vec![0.0; n])).is_err());
    assert!(fit.contrast(&[2.0], &[0.0], None).is_err(), "outside the observed binary support");
}
