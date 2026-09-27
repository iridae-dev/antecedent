//! End-to-end retained randomized policy-value execution.

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{PrimaryEstimate, Study};
use antecedent_core::{CausalQuery, MultiActionPolicyInputs, PolicyValueQuery, VariableId};
use antecedent_data::TabularData;

fn query() -> PolicyValueQuery {
    PolicyValueQuery {
        outcome: VariableId::from_raw(0),
        assignment: Arc::from([false, true, false, true]),
        propensity: Arc::from([0.5]),
        actions: Arc::from([false, true, false, true]),
        reference: Arc::from([false; 4]),
        mu0: Arc::from([1.0; 4]),
        mu1: Arc::from([3.0; 4]),
        costs: Arc::from([0.0]),
        reference_costs: Arc::from([0.0]),
        evaluation_subject_ids: Arc::from(["a", "b", "c", "d"].map(Arc::<str>::from)),
        disjoint_training_subjects: true,
        crossfit_fold_ownership_valid: false,
        multi_action: None,
        uplift_bins: Arc::from([]),
        uplift_bin_count: 0,
        uplift_training_subject_ids: Arc::from([]),
    }
}

#[test]
fn policy_answer_is_not_an_ate_and_survives_retained_reexecution() {
    let y = [1.0, 3.0, 1.0, 3.0];
    let data = TabularData::from_f64_columns([("outcome", &y[..])]).unwrap();
    let study = Study::tabular(data.clone()).query(CausalQuery::PolicyValue(query())).build().unwrap();
    let ctx = ExecutionContext::for_tests(41);
    let mut prepared = study.prepare(&ctx).unwrap();
    let first = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(first.estimand.method.as_ref(), "randomized.policy_value");
    assert!(first.identification.derivation.steps.iter().all(|step| step.rule.as_ref() != "randomized.itt"));
    let answer = first.policy_value.as_ref().unwrap();
    assert!((answer.policy_value - 2.0).abs() < 1e-12);
    assert!((answer.reference_value - 1.0).abs() < 1e-12);
    assert!((answer.incremental_value - 1.0).abs() < 1e-12);
    assert!(answer.incremental_standard_error.is_finite());
    let bytes = prepared.encode_contracted_result(&first, "policy-value", &ctx).unwrap();
    let (_, _, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let policy_wire = body.policy_value.as_ref().unwrap();
    assert_eq!(policy_wire.policy_value, answer.policy_value);
    assert_eq!(policy_wire.incremental_standard_error, answer.incremental_standard_error);
    assert_eq!(policy_wire.uncertainty, "row_score_standard_error_independent_subjects");
    assert_eq!(body.estimate, None, "policy values must never be serialized as ATEs");
    assert!(matches!(first.estimate, PrimaryEstimate::NotAnEffect));
    assert_eq!(first.effect().is_nan(), true);
    let refreshed = prepared.refresh(data, &ctx).unwrap();
    assert_eq!(first.policy_value, refreshed.policy_value);
    assert_eq!(first.logical_plan.plan_id, refreshed.logical_plan.plan_id);
}

#[test]
fn retained_uplift_bins_recover_ranked_randomized_contrasts_and_round_trip() {
    let mut ranked = query();
    ranked.mu0 = Arc::from([]);
    ranked.mu1 = Arc::from([]);
    ranked.disjoint_training_subjects = false;
    ranked.uplift_bins = Arc::from([0, 0, 1, 1]);
    ranked.uplift_bin_count = 2;
    ranked.uplift_training_subject_ids = Arc::from([Arc::<str>::from("rank-train")]);
    let data = TabularData::from_f64_columns([("outcome", &[1.0, 3.0, 1.0, 1.0][..])]).unwrap();
    let study = Study::tabular(data.clone()).query(CausalQuery::PolicyValue(ranked.clone())).build().unwrap();
    let ctx = ExecutionContext::for_tests(31);
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let bins = &result.policy_value.as_ref().unwrap().uplift_bins;
    assert_eq!(bins.len(), 2);
    assert!((bins[0].effect - 2.0).abs() < 1e-12);
    assert!((bins[1].effect - 0.0).abs() < 1e-12);
    assert_eq!(bins[0].evaluation_rows, 2);
    assert!(bins.iter().all(|bin| bin.standard_error.is_finite()));
    let bytes = prepared.encode_contracted_result(&result, "ranked-policy", &ctx).unwrap();
    let (_, _, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.policy_value.unwrap().uplift_bins[0].effect, 2.0);
    ranked.uplift_training_subject_ids = Arc::from([Arc::<str>::from("a")]);
    assert!(ranked.validate().is_err());
}

#[test]
fn policy_query_refuses_missing_ownership_or_overlap() {
    let mut invalid = query();
    invalid.disjoint_training_subjects = false;
    assert!(invalid.validate().is_err());
    invalid = query();
    invalid.propensity = Arc::from([0.0]);
    assert!(invalid.validate().is_err());
}

#[test]
fn randomized_ipw_policy_round_trips_without_nuisance_predictions() {
    let y = [1.0, 3.0, 1.0, 3.0];
    let data = TabularData::from_f64_columns([("outcome", &y[..])]).unwrap();
    let mut q = query();
    q.mu0 = Arc::from([]);
    q.mu1 = Arc::from([]);
    q.disjoint_training_subjects = false;
    let study = Study::tabular(data.clone()).query(CausalQuery::PolicyValue(q)).build().unwrap();
    let ctx = ExecutionContext::for_tests(42);
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(result.estimand.method.as_ref(), "randomized.policy_value");
    let answer = result.policy_value.as_ref().unwrap();
    assert!((answer.policy_value - 4.0).abs() < 1e-12);
    assert!((answer.reference_value - 1.0).abs() < 1e-12);
    assert!((answer.incremental_value - 3.0).abs() < 1e-12);
    assert_eq!(answer.prediction_ownership.as_ref(), "no_outcome_nuisance_predictions");
    assert_eq!(answer.uncertainty.as_ref(), "ipw_row_score_standard_error_independent_subjects");
    let bytes = prepared.encode_contracted_result(&result, "policy-ipw", &ctx).unwrap();
    let (_, _, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.policy_value.as_ref().unwrap().uncertainty, answer.uncertainty.as_ref());
    assert_eq!(body.estimate, None);
}

#[test]
fn retained_multi_action_value_matches_known_randomized_truth_and_round_trips() {
    let y = [1.0, 2.0, 4.0, 1.0, 2.0, 4.0, 1.0, 2.0, 4.0];
    let data = TabularData::from_f64_columns([("outcome", &y[..])]).unwrap();
    let mut q = query();
    q.assignment = Vec::new().into();
    q.propensity = Vec::new().into();
    q.actions = Vec::new().into();
    q.reference = Vec::new().into();
    q.mu0 = Vec::new().into();
    q.mu1 = Vec::new().into();
    q.costs = Vec::new().into();
    q.reference_costs = Vec::new().into();
    q.evaluation_subject_ids = (0..9).map(|i| Arc::<str>::from(format!("s{i}"))).collect::<Vec<_>>().into();
    q.disjoint_training_subjects = false;
    q.multi_action = Some(MultiActionPolicyInputs {
        action_labels: ["control", "A", "B"].map(Arc::<str>::from).into(),
        assignment: [0, 1, 2, 0, 1, 2, 0, 1, 2].into(),
        actions: [0, 0, 0, 1, 1, 1, 2, 2, 2].into(),
        reference: [0; 9].into(),
        propensities: vec![1.0 / 3.0; 27].into(),
        available: vec![true; 27].into(),
        costs: [0.0; 3].into(),
        reference_costs: [0.0; 3].into(),
        capacities: [9; 3].into(),
        reference_capacities: [9; 3].into(),
        budget: None,
        reference_budget: None,
    });
    let study = Study::tabular(data.clone()).query(CausalQuery::PolicyValue(q.clone())).build().unwrap();
    let ctx = ExecutionContext::for_tests(43);
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(result.estimand.method.as_ref(), "randomized.policy_value");
    assert_eq!(result.identification.derivation.steps[0].rule.as_ref(), "randomized.policy_value");
    let answer = result.policy_value.as_ref().unwrap();
    assert!((answer.policy_value - 7.0 / 3.0).abs() < 1e-12);
    assert!((answer.reference_value - 1.0).abs() < 1e-12);
    assert!((answer.incremental_value - 4.0 / 3.0).abs() < 1e-12);
    assert!((answer.treatment_rate - 2.0 / 3.0).abs() < 1e-12);
    assert_eq!(answer.uncertainty.as_ref(), "multi_action_ipw_row_score_standard_error_independent_subjects");
    let artifact = prepared.encode_contracted_result(&result, "multi-policy", &ctx).unwrap();
    let (_, _, body) = antecedent_io::decode_analysis_result_artifact(&artifact).unwrap();
    assert_eq!(body.policy_value.as_ref().unwrap().policy_value, answer.policy_value);
    assert_eq!(body.query, antecedent_io::causal_query_to_wire(&CausalQuery::PolicyValue(q.clone())).unwrap());
    q.multi_action.as_mut().unwrap().capacities = [9, 2, 9].into();
    assert!(q.validate().is_err());
    q.multi_action.as_mut().unwrap().capacities = [9; 3].into();
    q.multi_action.as_mut().unwrap().propensities = vec![0.0; 27].into();
    assert!(q.validate().is_err());
}
