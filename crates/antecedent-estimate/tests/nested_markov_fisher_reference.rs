//! Expected-information covariance of the Verma pilot against high-precision differentiation.
//!
//! `oracle_data/nested_markov_fisher_reference.py` (frozen in the JSON beside it) takes the
//! SLSQP maximum-likelihood parameters of integer count tables (tiny counts, 2e9 counts over nine
//! decades of cell scale, a table near the parameter boundary, random tables), differentiates the
//! cell law and the do-means by 60-digit mpmath central differences, forms
//! `I = N sum dp dp^T / p`, inverts it and propagates `J I^-1 J^T`. The product's analytic
//! derivatives, Cholesky inverse and delta propagation must agree.

use antecedent_core::ExecutionContext;
use antecedent_estimate::nested_markov_binary::{
    AdmgDeclaration, FitOptions, NestedMarkovInput, Regime, RegimeCounts,
};
use antecedent_estimate::nested_markov_uncertainty::nested_markov_fisher_internal;
use serde_json::Value as Json;

#[test]
fn delta_covariance_matches_high_precision_information_inverse() {
    let cases: Json =
        serde_json::from_str(include_str!("oracle_data/nested_markov_fisher_reference.json"))
            .unwrap();
    let ctx = ExecutionContext::for_tests(31);
    let mut checked = 0;
    for case in cases["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let cells: Vec<f64> =
            case["counts"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let input = NestedMarkovInput {
            graph: AdmgDeclaration::selected(),
            regimes: vec![RegimeCounts {
                regime: Regime::Observational,
                levels: vec![2; 4],
                cells,
            }],
        };
        let estimate =
            match nested_markov_fisher_internal(&input, &FitOptions::default(), 0.95, &ctx) {
                Ok(estimate) => estimate,
                Err(error) => panic!("{name}: {error}"),
            };
        assert_eq!(estimate.sample_count, case["sample_count"].as_f64().unwrap());
        for i in 0..3 {
            for j in 0..3 {
                let reference = case["effect_covariance"][i][j].as_f64().unwrap();
                let got = estimate.effect_covariance[i][j];
                let scale = case["effect_covariance"][i][i]
                    .as_f64()
                    .unwrap()
                    .max(case["effect_covariance"][j][j].as_f64().unwrap());
                assert!(
                    (got - reference).abs() <= 2e-4 * scale,
                    "{name}[{i}][{j}]: {got} vs {reference}"
                );
            }
        }
        assert!(estimate.inverse_residual < 1e-8, "{name}");
        // interval candidates are mean +/- z * sd with z the 0.975 normal quantile
        for i in 0..3 {
            let sd = estimate.effect_covariance[i][i].sqrt();
            let mid =
                0.5 * (estimate.interval_candidates[i][0] + estimate.interval_candidates[i][1]);
            let half =
                0.5 * (estimate.interval_candidates[i][1] - estimate.interval_candidates[i][0]);
            assert!((half / sd - 1.959_963_984_540_054).abs() < 1e-9, "{name} z");
            assert!(
                (mid - case["mean_values"][i].as_f64().unwrap()).abs() <= 1e-4,
                "{name}: centre {i}"
            );
        }
        checked += 1;
    }
    assert_eq!(checked, 10);
}
