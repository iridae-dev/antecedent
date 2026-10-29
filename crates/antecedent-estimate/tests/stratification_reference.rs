//! Propensity stratification against a 60-digit refit, the empirical quantile classes and a
//! case-weight derivative of the pooled contrast.
//!
//! `oracle_data/stratification_reference.py` (frozen in the JSON beside it) fits the logistic
//! propensity by Newton's method, cuts the clipped scores at the type-1 empirical quantiles
//! (the order statistic `x_(ceil(s n / k))`, equal scores sharing a class), pools the
//! within-class contrasts by the mass of the target arms and reads both pieces of every row's
//! influence off derivatives of the pooled contrast. Cases put 100 rows in five classes of
//! exactly twenty, a remainder, a clip that ties sixty scores into one class, a trim, an
//! outcome offset of 1e8 and declared sampling weights.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "oracle_data/table_support.rs"]
mod table_support;

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, RoleHint, TargetPopulation,
};
use antecedent_estimate::{
    OverlapPolicy, PropensityEstimationWorkspace, PropensityStratification, SamplingWeights,
};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use serde_json::Value;
use table_support::{id, table};

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/stratification_reference.json"))
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
fn pooled_effect_and_standard_error_match_the_reference() {
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
        let overlap = OverlapPolicy::RequireDiagnostics {
            clip: case["clip"].as_f64(),
            trim: case["trim"].as_f64(),
        };
        let strata = u32::try_from(case["strata"].as_u64().unwrap()).unwrap();
        for (key, population) in [
            ("ate", TargetPopulation::AllObserved),
            ("att", TargetPopulation::Treated),
            ("atc", TargetPopulation::Untreated),
        ] {
            let query =
                AverageEffectQuery::binary_ate(id(0), id(1)).with_target_population(population);
            let mut estimator = PropensityStratification::new()
                .with_n_strata(strata)
                .with_bootstrap_replicates(0)
                .with_overlap(overlap);
            if !case["weights"].is_null() {
                estimator = estimator
                    .with_sampling_weights(SamplingWeights::new(floats(&case["weights"])).unwrap());
            }
            let prepared = estimator.prepare(&data, &estimand, &query).unwrap();
            let got = estimator.fit(
                &prepared,
                &mut PropensityEstimationWorkspace::default(),
                &ExecutionContext::for_tests(3),
                AssumptionSet::new(),
            );
            let want = &case["targets"][key];
            // Every retained target is estimable (no fixture case records a refusal), so a null
            // reference would silently skip the comparison; it must not.
            assert!(!want.is_null(), "{name} {key}: the reference holds no value for this target");
            let got = got.unwrap_or_else(|e| panic!("{name} {key}: {e:?}"));
            close(got.ate, want["ate"].as_f64().unwrap(), 1e-8, &format!("{name} {key} effect"));
            close(got.se_analytic(), want["se"].as_f64().unwrap(), 1e-6, &format!("{name} {key} se"));
            assert_eq!(got.provenance.n_obs, want["rows"].as_u64(), "{name} {key} rows");
        }
    }
}
