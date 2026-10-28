//! DR-Learner against a fixed-fold exact reference, and its sampling-weight declaration.
//!
//! `oracle_data/dr_learner_reference.py` (frozen in the JSON beside it) takes the fold of every row
//! as part of the case, forms the cross-fitted AIPW pseudo-outcome from exact least-squares and
//! unpenalised logistic nuisances at 60 digits (the shared cross-fit machinery is reviewed in the
//! AIPW/DML record), and then evaluates this route's own arithmetic: the marginal effect and its
//! influence standard error, the ordinary-least-squares final stage on the retained
//! (common-support) rows, the fitted CATE at every row and its HC0 sandwich standard error.
//! Cases cover a trim, a clip that binds, both together, an outcome offset of 1e7, a covariate
//! offset of 1e3 at scale 250, nearly collinear covariates and a small sample. A separate test
//! checks that a declared integer sampling weight equals row replication for the point estimates,
//! so the declaration is neither ignored nor mis-scaled.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "oracle_data/table_support.rs"]
mod table_support;

use std::sync::Arc;

use antecedent_core::{AssumptionSet, AverageEffectQuery, ExecutionContext, RoleHint};
use antecedent_estimate::{DrLearner, OverlapPolicy, SamplingWeights};
use antecedent_expr::{ExprId, IdentifiedEstimand};
use antecedent_learn::{LearnerSpec, LinearSpec, LogisticSpec};
use serde_json::Value;
use table_support::{id, table};

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-9), "{what}: got {got} want {want}");
}

fn learner(case: &Value) -> DrLearner {
    let mut learner = DrLearner::new();
    learner.folds = case["k"].as_u64().unwrap() as usize;
    learner.outcome = LearnerSpec::Linear(LinearSpec::default());
    learner.treatment = LearnerSpec::Logistic(LogisticSpec { ridge_lambda: 0.0 });
    learner.final_learner = LearnerSpec::Linear(LinearSpec::default());
    learner.overlap = OverlapPolicy::RequireDiagnostics {
        clip: case["clip"].as_f64(),
        trim: case["trim"].as_f64(),
    };
    learner
}

fn columns(t: &[f64], y: &[f64], x: &[Vec<f64>]) -> Vec<(&'static str, RoleHint, Vec<f64>)> {
    let mut columns = vec![
        ("t", RoleHint::TreatmentCandidate, t.to_vec()),
        ("y", RoleHint::OutcomeCandidate, y.to_vec()),
    ];
    for k in 0..x[0].len() {
        columns.push(("x", RoleHint::Context, x.iter().map(|row| row[k]).collect()));
    }
    columns
}

fn setup(p: usize) -> (IdentifiedEstimand, AverageEffectQuery) {
    let adjustment: Vec<_> = (0..p).map(|k| id(2 + u32::try_from(k).unwrap())).collect();
    (
        IdentifiedEstimand::backdoor(
            "backdoor.adjustment",
            Arc::from(adjustment),
            ExprId::from_raw(0),
        ),
        AverageEffectQuery::binary_ate(id(0), id(1)),
    )
}

#[test]
fn effect_cate_and_hc0_standard_errors_match_the_fixed_fold_references() {
    let doc: Value =
        serde_json::from_str(include_str!("oracle_data/dr_learner_reference.json")).unwrap();
    for case in doc["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (t, y) = (floats(&case["t"]), floats(&case["y"]));
        let x: Vec<Vec<f64>> = case["x"].as_array().unwrap().iter().map(floats).collect();
        let folds: Vec<u32> =
            case["folds"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
        let data = table(&columns(&t, &y, &x));
        let (estimand, query) = setup(x[0].len());
        let estimator = learner(case);
        let mut problem = estimator.prepare(&data, &estimand, &query).unwrap();
        problem.fold_assignment = Some(Arc::from(folds));
        let got = estimator
            .fit(&problem, &ExecutionContext::for_tests(3), AssumptionSet::new())
            .unwrap_or_else(|e| panic!("{name}: {e:?}"));
        close(got.ate, case["ate"].as_f64().unwrap(), 1e-6, &format!("{name} effect"));
        close(got.se_analytic(), case["se"].as_f64().unwrap(), 1e-5, &format!("{name} se"));
        let cate = got.heterogeneity.cate.as_ref().expect("cate");
        let cate_se = got.heterogeneity.cate_se.as_ref().expect("cate se");
        let (want_cate, want_se) = (floats(&case["cate"]), floats(&case["cate_se"]));
        assert_eq!(cate.len(), want_cate.len());
        let scale = want_cate.iter().fold(0.0_f64, |a, v| a.max(v.abs()));
        for i in 0..cate.len() {
            assert!(
                (cate[i] - want_cate[i]).abs() <= 1e-5 * scale,
                "{name} cate[{i}]: {} vs {}",
                cate[i],
                want_cate[i]
            );
            close(cate_se[i], want_se[i], 1e-4, &format!("{name} cate_se[{i}]"));
        }
    }
}

#[test]
fn an_integer_sampling_weight_equals_row_replication_for_the_point_estimates() {
    let doc: Value =
        serde_json::from_str(include_str!("oracle_data/dr_learner_reference.json")).unwrap();
    let case = &doc["cases"][0];
    let (t, y) = (floats(&case["t"]), floats(&case["y"]));
    let x: Vec<Vec<f64>> = case["x"].as_array().unwrap().iter().map(floats).collect();
    let folds: Vec<u32> =
        case["folds"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect();
    let n = t.len();
    let mult: Vec<usize> = (0..n).map(|i| 1 + i % 3).collect();
    let (mut t2, mut y2, mut x2, mut f2) = (vec![], vec![], vec![], vec![]);
    for i in 0..n {
        for _ in 0..mult[i] {
            t2.push(t[i]);
            y2.push(y[i]);
            x2.push(x[i].clone());
            f2.push(folds[i]);
        }
    }
    let (estimand, query) = setup(x[0].len());
    let ctx = ExecutionContext::for_tests(3);

    let replicated = table(&columns(&t2, &y2, &x2));
    let estimator = learner(case);
    let mut problem = estimator.prepare(&replicated, &estimand, &query).unwrap();
    problem.fold_assignment = Some(Arc::from(f2));
    let want = estimator.fit(&problem, &ctx, AssumptionSet::new()).unwrap();

    let weighted = table(&columns(&t, &y, &x));
    let weights: Vec<f64> = mult.iter().map(|&m| m as f64).collect();
    let estimator = learner(case).with_sampling_weights(SamplingWeights::new(weights).unwrap());
    let mut problem = estimator.prepare(&weighted, &estimand, &query).unwrap();
    problem.fold_assignment = Some(Arc::from(folds));
    let got = estimator.fit(&problem, &ctx, AssumptionSet::new()).unwrap();
    let unweighted = {
        let estimator = learner(case);
        let mut problem = estimator.prepare(&weighted, &estimand, &query).unwrap();
        problem.fold_assignment = Some(Arc::from(floats_u32(case)));
        estimator.fit(&problem, &ctx, AssumptionSet::new()).unwrap()
    };
    assert!((unweighted.ate - want.ate).abs() > 1e-4, "the weights must matter on this case");
    close(got.ate, want.ate, 1e-6, "weighted effect");
    let (gc, wc) = (got.heterogeneity.cate.unwrap(), want.heterogeneity.cate.unwrap());
    let mut at = 0;
    for i in 0..n {
        close(gc[i], wc[at], 1e-6, &format!("weighted cate[{i}]"));
        at += mult[i];
    }
}

fn floats_u32(case: &Value) -> Vec<u32> {
    case["folds"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u32).collect()
}

#[test]
fn exact_support_profiles_report_the_hc0_estimate_of_the_matching_row() {
    use antecedent_estimate::CateProfile;
    let doc: Value =
        serde_json::from_str(include_str!("oracle_data/dr_learner_reference.json")).unwrap();
    let case = doc["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "discrete_covariate_profiles")
        .unwrap();
    let (t, y) = (floats(&case["t"]), floats(&case["y"]));
    let x: Vec<Vec<f64>> = case["x"].as_array().unwrap().iter().map(floats).collect();
    let folds: Vec<u32> = floats_u32(case);
    let data = table(&columns(&t, &y, &x));
    let (estimand, query) = setup(1);
    let estimator = learner(case);
    let mut problem = estimator.prepare(&data, &estimand, &query).unwrap();
    problem.fold_assignment = Some(Arc::from(folds));
    let profiles: Vec<CateProfile> =
        [0.0, 1.0, 2.0].iter().map(|&v| CateProfile { values: Arc::from([v]) }).collect();
    let (_, results) = estimator
        .fit_pointwise_profiles(
            &problem,
            &profiles,
            &ExecutionContext::for_tests(3),
            AssumptionSet::new(),
        )
        .unwrap();
    let (want_cate, want_se) = (floats(&case["cate"]), floats(&case["cate_se"]));
    for (profile, result) in profiles.iter().zip(results.iter()) {
        let row = x.iter().position(|r| r[0] == profile.values[0]).unwrap();
        close(result.estimate, want_cate[row], 1e-6, "profile estimate");
        close(result.standard_error, want_se[row], 1e-4, "profile hc0 standard error");
        assert!(result.control_count > 0 && result.treated_count > 0);
    }
    // A profile off the observed support is refused rather than extrapolated.
    let off = [CateProfile { values: Arc::from([0.5]) }];
    assert!(
        estimator
            .fit_pointwise_profiles(
                &problem,
                &off,
                &ExecutionContext::for_tests(3),
                AssumptionSet::new()
            )
            .is_err()
    );
}
