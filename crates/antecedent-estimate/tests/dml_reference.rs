//! Cross-fitted AIPW and partially linear DML against fixed-fold exact least-squares references.
//!
//! `oracle_data/dml_reference.py` (frozen in the JSON beside it) takes the fold of every row as
//! part of the case, solves the per-arm outcome regressions, the all-row outcome regression and
//! the unpenalised logistic propensity on each fold's training rows at 60 digits, and forms the
//! AIPW and Robinson scores, the trim and clip, and the standard error as the derivative of the
//! plug-in estimator with respect to the row masses. It reviews the orthogonal score and its
//! influence function, not a flexible learner. Cases cover five, three and two folds, a trim, a
//! clip that binds, both together, an outcome offset of 1e7, a covariate offset of 1e3 at scale
//! 250 and nearly collinear covariates.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "oracle_data/table_support.rs"]
mod table_support;

use std::sync::Arc;

use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, RoleHint};
use antecedent_estimate::{DmlAte, DmlScore, OverlapPolicy};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_learn::{LearnerSpec, LinearSpec, LogisticSpec};
use serde_json::Value;
use table_support::{id, table};

fn cases() -> Vec<Value> {
    serde_json::from_str::<Value>(include_str!("oracle_data/dml_reference.json")).unwrap()["cases"]
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
fn orthogonal_scores_and_standard_errors_match_the_fixed_fold_references() {
    for case in cases() {
        let name = case["name"].as_str().unwrap();
        let (t, y) = (floats(&case["t"]), floats(&case["y"]));
        let x: Vec<Vec<f64>> = case["x"].as_array().unwrap().iter().map(floats).collect();
        let folds: Vec<u32> =
            case["folds"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
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
        for (score, key) in [(DmlScore::Aipw, "aipw"), (DmlScore::PartiallyLinear, "plr")] {
            let mut estimator = DmlAte::new();
            estimator.folds = case["k"].as_u64().unwrap() as usize;
            estimator.outcome = LearnerSpec::Linear(LinearSpec::default());
            estimator.treatment = LearnerSpec::Logistic(LogisticSpec { ridge_lambda: 0.0 });
            estimator.score = score;
            estimator.overlap = OverlapPolicy::RequireDiagnostics {
                clip: case["clip"].as_f64(),
                trim: case["trim"].as_f64(),
            };
            let mut problem = estimator.prepare(&data, &estimand, &query).unwrap();
            problem.fold_assignment = Some(Arc::from(folds.clone()));
            let got = estimator
                .fit(&problem, &ExecutionContext::for_tests(3), AssumptionSet::new())
                .unwrap_or_else(|e| panic!("{name} {key}: {e:?}"));
            let want = &case[key];
            close(got.ate, want["ate"].as_f64().unwrap(), 1e-6, &format!("{name} {key} effect"));
            close(got.se_analytic(), want["se"].as_f64().unwrap(), 1e-5, &format!("{name} {key} se"));
        }
    }
}
