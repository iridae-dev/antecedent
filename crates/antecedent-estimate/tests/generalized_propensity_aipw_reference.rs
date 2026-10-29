//! Multi-level cross-fitted AIPW against a fixed-fold exact reference.
//!
//! `oracle_data/generalized_propensity_aipw_reference.py` (frozen in the JSON beside it) takes the
//! fold of every row as part of the case (the test asserts that the product's level-stratified
//! plan for the recorded seed is exactly that plan), solves the unpenalised multinomial-logit
//! generalized propensity and the per-level least-squares outcome regressions on each fold's
//! training rows at 60 digits, forms the per-level scores with the propensity floor, and reports
//! each level's standardized mean and every pairwise contrast with its iid influence standard
//! error. Cases cover three and four levels, three to five folds, a floor that binds on
//! propensities down to 0.01, an outcome offset of 1e7, a covariate offset of 1e3 at scale 250
//! and nearly collinear covariates. Further tests check that an undeclared level is refused
//! rather than dropped, and that a trim is refused rather than ignored.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "oracle_data/table_support.rs"]
mod table_support;

use std::sync::Arc;

use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, RoleHint};
use antecedent_estimate::{MultiLevelAipw, OverlapPolicy};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use serde_json::Value;
use table_support::{id, table};

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-9), "{what}: got {got} want {want}");
}

fn data_for(cat: &[f64], y: &[f64], z: &[Vec<f64>]) -> antecedent_data::TabularData {
    let mut columns = vec![
        ("t", RoleHint::TreatmentCandidate, cat.to_vec()),
        ("y", RoleHint::OutcomeCandidate, y.to_vec()),
    ];
    for j in 0..z[0].len() {
        columns.push(("z", RoleHint::Context, z.iter().map(|r| r[j]).collect()));
    }
    table(&columns)
}

fn estimand(p: usize) -> IdentifiedEstimand {
    let adjustment: Vec<_> = (0..p).map(|k| id(2 + u32::try_from(k).unwrap())).collect();
    IdentifiedEstimand::backdoor("backdoor.adjustment", Arc::from(adjustment), ExprId::from_raw(0))
}

#[test]
fn level_means_and_pairwise_contrasts_match_the_fixed_fold_references() {
    let doc: Value = serde_json::from_str(include_str!(
        "oracle_data/generalized_propensity_aipw_reference.json"
    ))
    .unwrap();
    let mut plans = Vec::new();
    for case in doc["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let cat = floats(&case["cat"]);
        let y = floats(&case["y"]);
        let z: Vec<Vec<f64>> = case["z"].as_array().unwrap().iter().map(floats).collect();
        let k = case["K"].as_u64().unwrap() as usize;
        let data = data_for(&cat, &y, &z);
        let levels: Vec<f64> = (0..k).map(|c| c as f64).collect();
        let overlap = OverlapPolicy::RequireDiagnostics { clip: case["clip"].as_f64(), trim: None };
        let estimator = MultiLevelAipw::full_population(levels)
            .with_folds(case["k"].as_u64().unwrap() as usize)
            .with_overlap(overlap);
        let ctx = ExecutionContext::for_tests(11);
        for contrast in case["contrasts"].as_array().unwrap() {
            let (a, b) =
                (contrast["active"].as_f64().unwrap(), contrast["control"].as_f64().unwrap());
            let query = AverageEffectQuery::with_levels(id(0), id(1), b, a);
            let got = match estimator.fit(
                &data,
                &estimand(z[0].len()),
                &query,
                &ctx,
                AssumptionSet::new(),
            ) {
                Ok(got) => got,
                // The multinomial IRLS may refuse (never return a wrong number) on a design
                // whose covariates differ by 1e-4; the refusal is the recorded behaviour.
                Err(error) if case["may_refuse"].as_bool() == Some(true) => {
                    assert!(
                        format!("{error:?}").contains("NonConverged"),
                        "{name}: refused for another reason: {error:?}"
                    );
                    continue;
                }
                Err(error) => panic!("{name}: {error:?}"),
            };
            let table = got.effect.scores.table.as_ref().expect("score table");
            let product_folds: Vec<u64> = table.fold_ids.iter().map(|&f| u64::from(f)).collect();
            let want_folds: Vec<u64> =
                case["folds"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
            if product_folds != want_folds {
                plans.push(format!("{name}: {product_folds:?}"));
                continue;
            }
            close(
                got.effect.ate,
                contrast["effect"].as_f64().unwrap(),
                1e-6,
                &format!("{name} {a}-{b} effect"),
            );
            close(
                got.effect.se_analytic(),
                contrast["se"].as_f64().unwrap(),
                1e-5,
                &format!("{name} {a}-{b} se"),
            );
            for (c, want) in floats(&case["means"]).iter().enumerate() {
                close(got.level_means[c].mean, *want, 1e-6, &format!("{name} mean of level {c}"));
            }
        }
    }
    assert!(
        plans.is_empty(),
        "the product fold plan differs from the recorded one:\n{}",
        plans.join("\n")
    );
}

#[test]
fn an_undeclared_level_is_refused_and_a_trim_is_refused() {
    let doc: Value = serde_json::from_str(include_str!(
        "oracle_data/generalized_propensity_aipw_reference.json"
    ))
    .unwrap();
    let case = &doc["cases"][0];
    let cat = floats(&case["cat"]);
    let y = floats(&case["y"]);
    let z: Vec<Vec<f64>> = case["z"].as_array().unwrap().iter().map(floats).collect();
    let data = data_for(&cat, &y, &z);
    let ctx = ExecutionContext::for_tests(11);
    let query = AverageEffectQuery::with_levels(id(0), id(1), 0.0, 1.0);
    // Level 2 is observed but not declared: the population would silently change.
    let narrow = MultiLevelAipw::full_population(vec![0.0, 1.0]);
    assert!(format!("{:?}", narrow.fit(&data, &estimand(2), &query, &ctx, AssumptionSet::new()).err()).contains("treatment_level_undeclared"), "treatment_level_undeclared");
    let trimmed = MultiLevelAipw::full_population(vec![0.0, 1.0, 2.0])
        .with_overlap(OverlapPolicy::RequireDiagnostics { clip: Some(0.05), trim: Some(0.1) });
    assert!(format!("{:?}", trimmed.fit(&data, &estimand(2), &query, &ctx, AssumptionSet::new()).err()).contains("generalized-propensity estimators do not trim"), "generalized-propensity estimators do not trim");
}
