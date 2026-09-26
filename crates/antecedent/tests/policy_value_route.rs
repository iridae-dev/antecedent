//! End-to-end retained randomized policy-value execution.

use std::sync::Arc;

use antecedent::prelude::ExecutionContext;
use antecedent::{PrimaryEstimate, Study};
use antecedent_core::{CausalQuery, PolicyValueQuery, VariableId};
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
fn policy_query_refuses_missing_ownership_or_overlap() {
    let mut invalid = query();
    invalid.disjoint_training_subjects = false;
    assert!(invalid.validate().is_err());
    invalid = query();
    invalid.propensity = Arc::from([0.0]);
    assert!(invalid.validate().is_err());
}
