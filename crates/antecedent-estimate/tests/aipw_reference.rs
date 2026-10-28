//! Full-sample AIPW (trimmed ATE, ATT, ATC) against 60-digit refits.
//!
//! `oracle_data/aipw_reference.py` (frozen in the JSON beside it) fits the logistic propensity on
//! all rows and the arm regressions in multiprecision arithmetic. For a trimmed ATE the standard
//! error is the derivative of the whole refitted estimator with respect to every row's case mass,
//! so the rows trimmed from the average still enter through the propensity they helped fit; for
//! ATT and ATC it is the centred efficient influence function the product documents. Cases cover a
//! trim, a trim with a clip, an outcome offset of 1e7 and strong selection. The untrimmed ATE is the
//! cross-fitted score; its recorded-fold references are the AIPW rows of `dml_reference.json`
//! (per-fold least-squares arms and unpenalised logistic propensity at 60 digits).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "oracle_data/table_support.rs"]
mod table_support;

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, AverageEffectQuery, ExecutionContext, RoleHint, TargetPopulation,
};
use antecedent_estimate::{AipwAte, AipwWorkspace, OverlapPolicy};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use serde_json::Value;
use table_support::{id, table};

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/aipw_reference.json")).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-300), "{what}: got {got} want {want}");
}

fn fit(case: &Value, population: TargetPopulation) -> antecedent_estimate::EffectEstimate {
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
    let query = AverageEffectQuery::binary_ate(id(0), id(1)).with_target_population(population);
    let estimator = AipwAte::new().with_bootstrap_replicates(0).with_overlap(
        OverlapPolicy::RequireDiagnostics { clip: case["clip"].as_f64(), trim: case["trim"].as_f64() },
    );
    let problem = estimator.prepare(&data, &estimand, &query).unwrap();
    estimator
        .fit(
            &problem,
            &mut AipwWorkspace::default(),
            &ExecutionContext::for_tests(3),
            AssumptionSet::new(),
        )
        .unwrap()
}

#[test]
fn trimmed_ate_matches_the_refit_and_its_case_mass_derivative() {
    for case in reference()["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let want = &case["trimmed_ate"];
        let got = fit(case, TargetPopulation::AllObserved);
        close(got.ate, want["ate"].as_f64().unwrap(), 1e-8, &format!("{name} effect"));
        close(got.se_analytic(), want["se"].as_f64().unwrap(), 1e-6, &format!("{name} se"));
    }
}

#[test]
fn att_and_atc_match_the_refit_and_the_centred_efficient_influence_function() {
    for case in reference()["population_cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        for (key, population) in
            [("att", TargetPopulation::Treated), ("atc", TargetPopulation::Untreated)]
        {
            let want = &case[key];
            let got = fit(case, population);
            close(got.ate, want["ate"].as_f64().unwrap(), 1e-8, &format!("{name} {key} effect"));
            close(got.se_analytic(), want["se"].as_f64().unwrap(), 1e-6, &format!("{name} {key} se"));
        }
    }
}

#[test]
fn untrimmed_ate_is_the_cross_fitted_score_on_the_recorded_folds() {
    let reference: Value =
        serde_json::from_str(include_str!("oracle_data/dml_reference.json")).unwrap();
    for case in reference["cases"].as_array().unwrap() {
        // The estimator cross-fits over five folds; a trimmed fit is the full-sample route.
        if !case["trim"].is_null() || case["k"].as_u64() != Some(5) {
            continue;
        }
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
        let query = AverageEffectQuery::binary_ate(id(0), id(1));
        let estimator = AipwAte::new()
            .with_bootstrap_replicates(0)
            .with_overlap(OverlapPolicy::RequireDiagnostics { clip: case["clip"].as_f64(), trim: None });
        let mut problem = estimator.prepare(&data, &estimand, &query).unwrap();
        let folds: Vec<u32> =
            case["folds"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        problem.fold_assignment = Some(Arc::from(folds));
        let got = estimator
            .fit(
                &problem,
                &mut AipwWorkspace::default(),
                &ExecutionContext::for_tests(3),
                AssumptionSet::new(),
            )
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        let want = &case["aipw"];
        close(got.ate, want["ate"].as_f64().unwrap(), 1e-6, &format!("{name} effect"));
        close(got.se_analytic(), want["se"].as_f64().unwrap(), 1e-5, &format!("{name} se"));
    }
}
