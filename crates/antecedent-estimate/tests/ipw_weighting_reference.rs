//! Hajek inverse-probability weighting against a 60-digit refit and case-weight derivative.
//!
//! `oracle_data/ipw_weighting_reference.py` (frozen in the JSON beside it) fits the logistic
//! propensity by Newton's method in multiprecision arithmetic, forms the clipped or trimmed
//! Hajek ATE, ATT and ATC, and takes the standard error from the derivative of the whole
//! refitted estimator with respect to each row's case weight, so the propensity-estimation term
//! is not written down anywhere in the reference. Cases cover a binding clip, a trim, two nearly
//! collinear covariates, an outcome offset of 1e7, a covariate offset of 1e3 at scale 250 and a
//! nine-row table.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "oracle_data/table_support.rs"]
mod table_support;

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, RoleHint, TargetPopulation,
};
use antecedent_estimate::{OverlapPolicy, PropensityEstimationWorkspace, PropensityWeighting};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use serde_json::Value;
use table_support::{id, table};

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/ipw_weighting_reference.json"))
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
fn weighted_effect_and_standard_error_match_the_multiprecision_refit() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let (t, y) = (floats(&case["t"]), floats(&case["y"]));
        let x: Vec<Vec<f64>> = case["x"].as_array().unwrap().iter().map(floats).collect();
        let p = x[0].len();
        let mut columns = vec![
            ("t", RoleHint::TreatmentCandidate, t),
            ("y", RoleHint::OutcomeCandidate, y),
        ];
        for k in 0..p {
            columns.push(("x", RoleHint::Context, x.iter().map(|row| row[k]).collect()));
        }
        let data = table(&columns);
        let adjustment: Vec<_> = (0..p).map(|k| id(2 + u32::try_from(k).unwrap())).collect();
        let estimand =
            IdentifiedEstimand::backdoor("backdoor.adjustment", Arc::from(adjustment), ExprId::from_raw(0));
        let bound = |v: &Value| v.as_f64();
        let overlap = OverlapPolicy::RequireDiagnostics {
            clip: bound(&case["clip"]),
            trim: bound(&case["trim"]),
        };
        for (key, population) in [
            ("ate", TargetPopulation::AllObserved),
            ("att", TargetPopulation::Treated),
            ("atc", TargetPopulation::Untreated),
        ] {
            let query = AverageEffectQuery::binary_ate(id(0), id(1)).with_target_population(population);
            let estimator = PropensityWeighting::new().with_bootstrap_replicates(0).with_overlap(overlap);
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
        }
    }
}
