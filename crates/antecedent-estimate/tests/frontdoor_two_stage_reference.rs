//! Front-door path-sum estimator and its stacked sandwich against an 80-digit reference.
//!
//! `oracle_data/frontdoor_two_stage_reference.py` (frozen in the JSON beside it) solves the
//! `k` mediator regressions and the outcome regression at 80 digits, stacks their estimating
//! equations and forms `delta * grad' A^-1 B A^-1 grad` for the HC0, HC1 and cluster meats with
//! the per-equation small-sample convention the product documents. Cases cover one to three
//! mediators, a treatment-level difference of two, and offsets of 1e3 (treatment), 1e5
//! (mediators) and 1e6 (outcome).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "oracle_data/table_support.rs"]
mod table_support;

use std::sync::Arc;

use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, RoleHint};
use antecedent_estimate::{AnalyticSeKind, FrontDoorTwoStage, FrontDoorWorkspace};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use serde_json::Value;
use table_support::{id, table};

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-300), "{what}: got {got} want {want}");
}

#[test]
fn path_sum_and_stacked_standard_errors_match_the_exact_reference() {
    let file: Value =
        serde_json::from_str(include_str!("oracle_data/frontdoor_two_stage_reference.json"))
            .unwrap();
    for case in file["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let mediators: Vec<Vec<f64>> = case["m"].as_array().unwrap().iter().map(floats).collect();
        let k = mediators.len();
        let mut columns = vec![
            ("t", RoleHint::TreatmentCandidate, floats(&case["t"])),
            ("y", RoleHint::OutcomeCandidate, floats(&case["y"])),
        ];
        for m in &mediators {
            columns.push(("m", RoleHint::Context, m.clone()));
        }
        let data = table(&columns);
        let ids: Vec<_> = (0..k).map(|j| id(2 + u32::try_from(j).unwrap())).collect();
        let estimand =
            IdentifiedEstimand::frontdoor("frontdoor", Arc::from(ids), ExprId::from_raw(0));
        let low = case["t_low"].as_f64().unwrap();
        let query = AverageEffectQuery::with_levels(
            id(0),
            id(1),
            low,
            low + case["delta"].as_f64().unwrap(),
        );
        let clusters: Vec<u32> = case["clusters"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap() as u32)
            .collect();
        for (kind, key) in [
            (AnalyticSeKind::Hc0, "se_hc0"),
            (AnalyticSeKind::Hc1, "se_hc1"),
            (AnalyticSeKind::Cluster, "se_cluster"),
        ] {
            let estimator = FrontDoorTwoStage::new()
                .with_bootstrap_replicates(0)
                .with_se_kind(kind)
                .with_cluster_ids(clusters.clone());
            let prep = estimator.prepare(&data, &estimand, &query).unwrap();
            let got = estimator
                .fit(
                    &prep,
                    &mut FrontDoorWorkspace::default(),
                    &ExecutionContext::for_tests(1),
                    AssumptionSet::new(),
                )
                .unwrap_or_else(|e| panic!("{name} {key}: {e:?}"));
            // Uncentred offsets of 1e3 to 1e6 leave the design within 1e-5 of collinear with the
            // intercept, so the least-squares residuals carry a conditioning-squared error of
            // about 1e-6 relative that the standard error inherits (the effect keeps 1e-7).
            let (effect_tol, se_tol) = if name == "offsets" { (1e-7, 1e-5) } else { (1e-9, 1e-9) };
            close(got.ate, case["effect"].as_f64().unwrap(), effect_tol, &format!("{name} effect"));
            close(got.se_analytic(), case[key].as_f64().unwrap(), se_tol, &format!("{name} {key}"));
        }
    }
}
