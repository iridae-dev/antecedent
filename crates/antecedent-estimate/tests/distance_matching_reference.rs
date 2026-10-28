//! Covariate-distance matching against an exhaustive 60-digit search.
//!
//! `oracle_data/distance_matching_reference.py` (frozen in the JSON beside it) z-scores every
//! covariate, compares each query with every donor of the other arm by Euclidean distance,
//! averages all donors tied at the smallest distance, applies the caliper (a distance equal to it
//! matches), removes the donor-arm least-squares bias term and forms the matched effects and the
//! homoskedastic matching variance from their definitions. Cases cover two and three covariates,
//! a caliper, duplicated rows that tie, an outcome offset of 1e7, a covariate offset of 1e6 and
//! covariates in units of 1e-14 and 1e14 (the match must not depend on units), near-collinear
//! covariates and a sixteen-row table.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "oracle_data/table_support.rs"]
mod table_support;

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, RoleHint, TargetPopulation,
};
use antecedent_estimate::{DistanceMatching, OverlapPolicy, PropensityEstimationWorkspace};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use serde_json::Value;
use table_support::{id, table};

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/distance_matching_reference.json"))
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
fn distance_matched_effect_and_standard_error_match_the_exhaustive_reference() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let (t, y) = (floats(&case["t"]), floats(&case["y"]));
        let x: Vec<Vec<f64>> = case["x"].as_array().unwrap().iter().map(floats).collect();
        let p = x[0].len();
        let mut columns =
            vec![("t", RoleHint::TreatmentCandidate, t), ("y", RoleHint::OutcomeCandidate, y)];
        for k in 0..p {
            columns.push(("x", RoleHint::Context, x.iter().map(|row| row[k]).collect()));
        }
        let data = table(&columns);
        let adjustment: Vec<_> = (0..p).map(|k| id(2 + u32::try_from(k).unwrap())).collect();
        let estimand = IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from(adjustment),
            ExprId::from_raw(0),
        );
        let overlap = OverlapPolicy::RequireDiagnostics { clip: Some(0.01), trim: None };
        for (key, population) in [
            ("att", TargetPopulation::Treated),
            ("atc", TargetPopulation::Untreated),
            ("ate", TargetPopulation::AllObserved),
        ] {
            let query =
                AverageEffectQuery::binary_ate(id(0), id(1)).with_target_population(population);
            let mut estimator =
                DistanceMatching::new().with_bootstrap_replicates(0).with_overlap(overlap);
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
