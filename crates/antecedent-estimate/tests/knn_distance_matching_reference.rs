//! M-nearest-neighbour covariate-distance matching against an exhaustive 60-digit search.
//!
//! `oracle_data/knn_distance_matching_reference.py` (frozen in the JSON beside it) z-scores every
//! covariate, takes for each query the M donors of the other arm with the smallest (distance,
//! donor position) pairs, applies the caliper to the M-th nearest (a distance equal to it
//! matches), removes the donor-arm least-squares bias term of the donor mean and forms the
//! matched effects and the Abadie-Imbens donor-reuse variance for M matches (use counts over M,
//! outcome noise `Var(effect) / (1 + 1/M)`) from their definitions. Cases cover M of 2, 3, 4 and
//! 5, a caliper, duplicated rows that tie, covariate units of 1e14 and a small table.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "oracle_data/table_support.rs"]
mod table_support;

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, RoleHint, TargetPopulation,
};
use antecedent_estimate::{KnnDistanceMatching, OverlapPolicy, PropensityEstimationWorkspace};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use serde_json::Value;
use table_support::{id, table};

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/knn_distance_matching_reference.json"))
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
fn knn_matched_effect_and_standard_error_match_the_exhaustive_reference() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let (t, y) = (floats(&case["t"]), floats(&case["y"]));
        let x: Vec<Vec<f64>> = case["x"].as_array().unwrap().iter().map(floats).collect();
        let p = x[0].len();
        let neighbors = usize::try_from(case["neighbors"].as_u64().unwrap()).unwrap();
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
            let mut estimator = KnnDistanceMatching::new(neighbors).unwrap();
            estimator.matching =
                estimator.matching.clone().with_bootstrap_replicates(0).with_overlap(overlap);
            if let Some(caliper) = case["caliper"].as_f64() {
                estimator.matching = estimator.matching.clone().with_caliper(caliper);
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
            close(
                got.se_analytic(),
                want["se"].as_f64().unwrap(),
                1e-6,
                &format!("{name} {key} se"),
            );
            assert_eq!(got.provenance.n_obs, want["matched"].as_u64(), "{name} {key} matched");
        }
    }
}
