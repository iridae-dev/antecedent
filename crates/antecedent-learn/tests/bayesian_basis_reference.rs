//! Quadratic-basis Bayesian g-computation against exact Normal-Inverse-Gamma moments.
//!
//! `oracle_data/bayesian_basis_reference.py` (frozen in the JSON beside it) builds the basis
//! `(1, t, z_j, t z_j, z_j^2)` and gives, in 40-digit arithmetic, the mean and standard deviation
//! of every linear functional of the Student-t coefficient vector: the ATE (the row effect
//! `b_t + sum_j b_tz_j z_j` averaged over the observed rows), the row-level effects and the
//! row-level control outcomes. The estimator's shared draws must reproduce them, which checks the
//! basis layout, the prior scale convention and the per-row pushforward.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_learn::{
    BasisData, BasisTargetPopulation, BayesianBasisGComputation, BayesianBasisSpec,
};
use serde_json::Value;

const DRAWS: usize = 80_000;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn moments(values: impl Iterator<Item = f64>) -> (f64, f64) {
    let values: Vec<f64> = values.collect();
    let n = values.len() as f64;
    let mean = values.iter().sum::<f64>() / n;
    let sd = (values.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (n - 1.0)).sqrt();
    (mean, sd)
}

#[test]
fn ate_row_effects_and_control_outcomes_match_the_exact_posterior() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("oracle_data/bayesian_basis_reference.json")).unwrap();
    let root = (DRAWS as f64).sqrt();
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let (t, y) = (floats(&case["treatment"]), floats(&case["outcome"]));
        let covariates: Vec<f64> =
            case["covariates"].as_array().unwrap().iter().flat_map(floats).collect();
        let fit = BayesianBasisGComputation
            .fit(
                &BasisData {
                    treatment: &t,
                    outcome: &y,
                    covariates_colmajor: &covariates,
                    n_covariates: 2,
                    row_ids: None,
                    fold_ids: None,
                },
                BasisTargetPopulation::AllObserved,
                BayesianBasisSpec {
                    prior_sd: case["prior_sd"].as_f64().unwrap(),
                    n_draws: DRAWS,
                    seed: 29,
                },
                &ExecutionContext::for_tests(1),
            )
            .unwrap();
        let (mean, sd) = moments(fit.ate_draws.iter().copied());
        let (r_mean, r_sd) = (case["ate_mean"].as_f64().unwrap(), case["ate_sd"].as_f64().unwrap());
        assert!((mean - r_mean).abs() < 5.0 * r_sd / root, "{name} ATE mean {mean} vs {r_mean}");
        assert!((sd / r_sd - 1.0).abs() < 8.0 / root, "{name} ATE sd {sd} vs {r_sd}");
        assert!((fit.ate_mean - mean).abs() < 1e-12);

        for row in case["rows"].as_array().unwrap() {
            let r = row["row"].as_u64().unwrap() as usize;
            for (target, key) in [(2, "effect"), (0, "control")] {
                let (m, s) =
                    moments((0..DRAWS).map(|d| fit.predictions.draw_target(d, target).unwrap()[r]));
                let (rm, rs) = (
                    row[format!("{key}_mean")].as_f64().unwrap(),
                    row[format!("{key}_sd")].as_f64().unwrap(),
                );
                assert!((m - rm).abs() < 5.0 * rs / root, "{name} row {r} {key} mean {m} vs {rm}");
                assert!((s / rs - 1.0).abs() < 8.0 / root, "{name} row {r} {key} sd {s} vs {rs}");
            }
        }
    }
}
