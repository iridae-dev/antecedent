//! Nonlinear continuous-mediator natural effects against an 80-digit reference on noisy data.
//!
//! `oracle_data/nonlinear_mediation_reference.py` (frozen in the JSON beside it) fits the
//! mediator and the polynomial outcome by the normal equations at 80 digits and integrates the
//! outcome polynomial over the fitted Gaussian mediator law in closed form (exact normal
//! moments), where the product uses Gauss-Hermite quadrature. Cases cover degree two to four,
//! a mediator offset of 1e6, a covariate offset of 1e3, an outcome offset of 1e7, an
//! exponential (skewed) mediator and a mediator residual 1e-3 of its scale.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::nonlinear_mediation::{
    estimate_nonlinear_mediation, NonlinearMediationConfig, NonlinearMediationInput,
    NonlinearMediationPremises,
};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-300), "{what}: got {got} want {want}");
}

#[test]
fn natural_effects_and_mediator_fit_match_the_exact_gaussian_integral() {
    let file: Value =
        serde_json::from_str(include_str!("oracle_data/nonlinear_mediation_reference.json"))
            .unwrap();
    for case in file["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (a, m, y) = (floats(&case["a"]), floats(&case["m"]), floats(&case["y"]));
        let x: Vec<Vec<f64>> = case["x"].as_array().unwrap().iter().map(floats).collect();
        let covariates: Vec<&[f64]> = x.iter().map(Vec::as_slice).collect();
        let input = NonlinearMediationInput {
            treatment: &a,
            mediator: &m,
            outcome: &y,
            covariates: &covariates,
        };
        let config = NonlinearMediationConfig {
            outcome_degree: case["degree"].as_u64().unwrap() as usize,
            bootstrap_replicates: 0,
            min_arm_count: 5,
            max_support_violation: 1.0,
            ..NonlinearMediationConfig::default()
        };
        let got = estimate_nonlinear_mediation(
            &input,
            &NonlinearMediationPremises::sequentially_ignorable(),
            &config,
        )
        .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        // The outcome offset of 1e7 costs the least-squares fit about 1e-9 relative.
        let rel = 1e-6;
        close(got.natural_direct, case["direct"].as_f64().unwrap(), rel, &format!("{name} NDE"));
        close(
            got.natural_indirect,
            case["indirect"].as_f64().unwrap(),
            rel,
            &format!("{name} NIE"),
        );
        close(got.total, case["total"].as_f64().unwrap(), rel, &format!("{name} TE"));
        close(
            got.mediator.residual_variance,
            case["residual_variance"].as_f64().unwrap(),
            1e-9,
            &format!("{name} mediator residual variance"),
        );
        for (g, w) in got.mediator.coefficients.iter().zip(floats(&case["med_coefficients"])) {
            close(*g, w, 1e-6, &format!("{name} mediator coefficient"));
        }
    }
}
