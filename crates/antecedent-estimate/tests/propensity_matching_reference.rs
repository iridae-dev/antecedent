//! Propensity matching against an exhaustive 60-digit search.
//!
//! `oracle_data/propensity_matching_reference.py` (frozen in the JSON beside it) fits the
//! logistic propensity by Newton's method, compares every query with every donor of the other
//! arm on the logit or the clipped probability, averages all donors tied at the smallest
//! distance, applies the caliper (a distance equal to it matches), removes the donor-arm
//! least-squares bias term and forms the matched effects and the homoskedastic matching
//! variance from their definitions. Cases cover both scales, a caliper on each, clipped scores
//! that tie, extreme unclipped scores, a covariate with five levels, an outcome offset of 1e7
//! and a ten-row table.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "oracle_data/table_support.rs"]
mod table_support;

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, RoleHint, TargetPopulation,
};
use antecedent_estimate::{
    CaliperScale, OverlapPolicy, PropensityEstimationWorkspace, PropensityMatching,
};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use serde_json::Value;
use table_support::{id, table};

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/propensity_matching_reference.json"))
        .unwrap()["cases"]
        .as_array()
        .unwrap()
        .clone()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-300), "{what}: got {got} want {want}");
}

#[test]
fn matched_effect_and_standard_error_match_the_exhaustive_reference() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let (t, y) = (floats(&case["t"]), floats(&case["y"]));
        let x: Vec<Vec<f64>> = case["x"].as_array().unwrap().iter().map(floats).collect();
        let data = table(&[
            ("t", RoleHint::TreatmentCandidate, t),
            ("y", RoleHint::OutcomeCandidate, y),
            ("x", RoleHint::Context, x.iter().map(|row| row[0]).collect()),
        ]);
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from([id(2)]),
            ExprId::from_raw(0),
        );
        let overlap = OverlapPolicy::RequireDiagnostics { clip: case["clip"].as_f64(), trim: None };
        let scale = match case["scale"].as_str().unwrap() {
            "raw" => CaliperScale::Raw,
            _ => CaliperScale::Logit,
        };
        for (key, population) in [
            ("att", TargetPopulation::Treated),
            ("atc", TargetPopulation::Untreated),
            ("ate", TargetPopulation::AllObserved),
        ] {
            let query =
                AverageEffectQuery::binary_ate(id(0), id(1)).with_target_population(population);
            let mut estimator = PropensityMatching::new()
                .with_bootstrap_replicates(0)
                .with_overlap(overlap)
                .with_caliper_scale(scale);
            if let Some(caliper) = case["caliper"].as_f64() {
                estimator = estimator.with_caliper(caliper);
            }
            let prepared = estimator.prepare(&data, &estimand, &query).unwrap();
            let got = estimator
                .fit(
                    &prepared,
                    &mut PropensityEstimationWorkspace::default(),
                    &ExecutionContext::for_tests(3),
                    AssumptionSet::new(),
                )
                .unwrap_or_else(|e| panic!("{name} {key}: {e:?}"));
            let want = &case["targets"][key];
            close(got.ate, want["ate"].as_f64().unwrap(), 1e-8, &format!("{name} {key} effect"));
            close(got.se_analytic(), want["se"].as_f64().unwrap(), 1e-6, &format!("{name} {key} se"));
            assert_eq!(got.provenance.n_obs, want["matched"].as_u64(), "{name} {key} matched");
        }
    }
}
