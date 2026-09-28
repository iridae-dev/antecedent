//! End-to-end retained randomized policy-value execution.
#![allow(
    clippy::float_cmp,
    clippy::bool_assert_comparison,
    reason = "integration test asserts exact deterministic estimates and an explicit boolean predicate"
)]

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
        global_constraints_present: false,
        multi_action: None,
        uplift_bins: Arc::from([]),
        uplift_bin_count: 0,
        uplift_training_subject_ids: Arc::from([]),
        regret: None,
    }
}

#[test]
fn retained_finite_class_regret_round_trips_and_rejects_forged_bounds() {
    let n = 400;
    let assignment = (0..n).map(|i| i % 4 < 2).collect::<Vec<_>>();
    let outcome = assignment
        .iter()
        .enumerate()
        .map(|(i, &assigned)| {
            1.0 + if assigned { if i % 2 == 0 { 2.0 } else { -1.0 } } else { 0.0 }
                + (i % 7) as f64 * 0.01
        })
        .collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let candidates = vec![
        Arc::<[bool]>::from(vec![false; n]),
        Arc::<[bool]>::from(vec![true; n]),
        Arc::<[bool]>::from((0..n).map(|i| i % 2 == 0).collect::<Vec<_>>()),
        Arc::<[bool]>::from((0..n).map(|i| i % 2 == 1).collect::<Vec<_>>()),
    ];
    let mut q = query();
    q.assignment = assignment.into();
    q.propensity = Arc::from([0.5]);
    q.actions = candidates[1].clone();
    q.reference = Arc::from(vec![false; n]);
    q.costs = Arc::from([0.2]);
    q.mu0 = Arc::from([]);
    q.mu1 = Arc::from([]);
    q.disjoint_training_subjects = false;
    q.evaluation_subject_ids =
        (0..n).map(|i| Arc::<str>::from(format!("eval-{i}"))).collect::<Vec<_>>().into();
    q.regret = Some(antecedent_core::FixedCandidateRegretInputs {
        candidates: candidates.into(),
        selected_index: 1,
        training_subject_ids: Arc::from([Arc::<str>::from("selection-training")]),
    });
    let ctx = ExecutionContext::for_tests(82);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::PolicyValue(q.clone()))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let policy = result.policy_value.as_ref().unwrap();
    let regret = policy.regret.as_ref().unwrap();
    assert!(regret.regret > 0.0);
    assert!(regret.interval_95[0] <= regret.regret && regret.regret <= regret.interval_95[1]);
    // The finite-class simultaneous regret bound is now an exact graphless row.
    assert_eq!(result.support_status, Some(antecedent::support::CellStatus::Licensed));
    let bytes = prepared.encode_contracted_result(&result, "fixed-class-regret", &ctx).unwrap();
    let (_, header, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.policy_value.as_ref().unwrap().regret.as_ref().unwrap().selected_index, 1);
    assert_eq!(
        body.policy_value.as_ref().unwrap().graphless_support_status.as_deref(),
        Some("licensed")
    );
    let mut forged = body.clone();
    forged.policy_value.as_mut().unwrap().regret.as_mut().unwrap().interval_95[1] += 0.5;
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged,
            header.variable_names.clone(),
            "forged-regret-bound"
        )
        .is_err()
    );
    // Forging the licensed status onto a scalar-only payload must also be rejected.
    let mut forged_status = body.clone();
    forged_status.policy_value.as_mut().unwrap().regret = None;
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &forged_status,
            header.variable_names,
            "forged-regret-status"
        )
        .is_err()
    );
    q.regret.as_mut().unwrap().training_subject_ids = Arc::from([Arc::<str>::from("eval-0")]);
    assert!(q.validate().is_err());
}

#[test]
fn policy_answer_is_not_an_ate_and_survives_retained_reexecution() {
    let y = [1.0, 3.0, 1.0, 3.0];
    let data = TabularData::from_f64_columns([("outcome", &y[..])]).unwrap();
    let study =
        Study::tabular(data.clone()).query(CausalQuery::PolicyValue(query())).build().unwrap();
    let ctx = ExecutionContext::for_tests(41);
    let mut prepared = study.prepare(&ctx).unwrap();
    let first = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(first.estimand.method.as_ref(), "randomized.policy_value");
    assert!(
        first
            .identification
            .derivation
            .steps
            .iter()
            .all(|step| step.rule.as_ref() != "randomized.itt")
    );
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
    let study = Study::tabular(data.clone())
        .query(CausalQuery::PolicyValue(ranked.clone()))
        .build()
        .unwrap();
    let ctx = ExecutionContext::for_tests(31);
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let bins = &result.policy_value.as_ref().unwrap().uplift_bins;
    assert_eq!(bins.len(), 2);
    assert!((bins[0].effect - 2.0).abs() < 1e-12);
    assert!((bins[1].effect - 0.0).abs() < 1e-12);
    assert_eq!(bins[0].evaluation_rows, 2);
    assert!(bins.iter().all(|bin| bin.standard_error.is_finite()));
    assert!(bins.iter().all(|bin| bin.interval_95.is_none()));
    let bytes = prepared.encode_contracted_result(&result, "ranked-policy", &ctx).unwrap();
    let (_, _, body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.policy_value.unwrap().uplift_bins[0].effect, 2.0);
    ranked.uplift_training_subject_ids = Arc::from([Arc::<str>::from("a")]);
    assert!(ranked.validate().is_err());
}

#[test]
fn retained_held_out_uplift_intervals_round_trip_and_refuse_forged_bounds() {
    let n = 600;
    let assignment = (0..n).map(|i| i % 2 == 0).collect::<Vec<_>>();
    let bin_ids = (0..n).map(|i| usize::from(i >= 300)).collect::<Vec<_>>();
    let outcomes = (0..n)
        .map(|i| 1.0 + if assignment[i] { if bin_ids[i] == 0 { 2.0 } else { 0.5 } } else { 0.0 })
        .collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap();
    let mut ranked = query();
    ranked.assignment = assignment.into();
    ranked.actions = (0..n).map(|i| i < 300).collect::<Vec<_>>().into();
    ranked.reference = vec![false; n].into();
    ranked.mu0 = Arc::from([]);
    ranked.mu1 = Arc::from([]);
    ranked.disjoint_training_subjects = false;
    ranked.evaluation_subject_ids =
        (0..n).map(|i| Arc::<str>::from(format!("eval-{i}"))).collect::<Vec<_>>().into();
    ranked.uplift_bins = bin_ids.into();
    ranked.uplift_bin_count = 2;
    ranked.uplift_training_subject_ids = Arc::from([Arc::<str>::from("rank-train")]);
    let ctx = ExecutionContext::for_tests(61);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::PolicyValue(ranked.clone()))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let bins = &result.policy_value.as_ref().unwrap().uplift_bins;
    assert_eq!(bins.len(), 2);
    for (index, &truth) in [2.0, 0.5].iter().enumerate() {
        let bounds = bins[index].interval_95.unwrap();
        assert!(bounds[0] < truth && truth < bounds[1]);
        assert_eq!(bins[index].evaluation_rows, 300);
    }
    let artifact = prepared.encode_contracted_result(&result, "ranked-interval", &ctx).unwrap();
    let (_, _, mut body) = antecedent_io::decode_analysis_result_artifact(&artifact).unwrap();
    assert_eq!(body.policy_value.as_ref().unwrap().uplift_bins[0].interval_95, bins[0].interval_95);
    assert_eq!(result.support_status, Some(antecedent::support::CellStatus::Licensed));
    assert_eq!(
        body.policy_value.as_ref().unwrap().graphless_support_status.as_deref(),
        Some("licensed")
    );
    body.policy_value.as_mut().unwrap().uplift_bins[0].interval_95 = Some([0.0, 0.0]);
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &body,
            vec!["outcome".into()],
            "forged-uplift"
        )
        .is_err()
    );

    ranked.mu0 = vec![1.0; n].into();
    ranked.mu1 = vec![3.0; n].into();
    ranked.disjoint_training_subjects = true;
    let aipw = Study::tabular(data.clone())
        .query(CausalQuery::PolicyValue(ranked))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let aipw_result = aipw.estimate(&data, &ctx).unwrap();
    assert_eq!(aipw_result.support_status, Some(antecedent::support::CellStatus::Licensed));
    assert!(
        aipw_result
            .policy_value
            .as_ref()
            .unwrap()
            .uplift_bins
            .iter()
            .all(|bin| bin.interval_95.is_some())
    );
    let aipw_artifact =
        aipw.encode_contracted_result(&aipw_result, "aipw-ranked-interval", &ctx).unwrap();
    let (_, _, aipw_body) = antecedent_io::decode_analysis_result_artifact(&aipw_artifact).unwrap();
    assert_eq!(
        aipw_body.policy_value.unwrap().graphless_support_status.as_deref(),
        Some("licensed")
    );
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
fn retained_held_out_policy_intervals_round_trip_and_reject_forgery() {
    let n = 300;
    let assignment = (0..n).map(|i| i % 2 == 1).collect::<Vec<_>>();
    let actions = (0..n).map(|i| i % 3 == 0).collect::<Vec<_>>();
    let outcomes = (0..n)
        .map(|i| 1.0 + 2.0 * f64::from(assignment[i]) + (i % 5) as f64 / 10.0)
        .collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap();
    let mut q = query();
    q.assignment = assignment.into();
    q.actions = actions.into();
    q.reference = vec![false; n].into();
    q.mu0 = vec![1.0; n].into();
    q.mu1 = vec![3.0; n].into();
    q.evaluation_subject_ids =
        (0..n).map(|i| Arc::<str>::from(format!("evaluation-{i}"))).collect::<Vec<_>>().into();
    let ctx = ExecutionContext::for_tests(44);
    let study =
        Study::tabular(data.clone()).query(CausalQuery::PolicyValue(q.clone())).build().unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let value = result.policy_value.as_ref().unwrap();
    let policy_ci = value.policy_interval_95.unwrap();
    let incremental_ci = value.incremental_interval_95.unwrap();
    assert!(policy_ci[0] < value.policy_value && value.policy_value < policy_ci[1]);
    assert!(
        incremental_ci[0] < value.incremental_value && value.incremental_value < incremental_ci[1]
    );
    assert_eq!(result.support_status, Some(antecedent::support::CellStatus::Licensed));
    let bytes = prepared.encode_contracted_result(&result, "policy-interval", &ctx).unwrap();
    let (_, _, mut body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.policy_value.as_ref().unwrap().policy_interval_95, Some(policy_ci));
    assert_eq!(body.policy_value.as_ref().unwrap().incremental_interval_95, Some(incremental_ci));
    assert_eq!(
        body.policy_value.as_ref().unwrap().graphless_support_status.as_deref(),
        Some("licensed")
    );
    body.policy_value.as_mut().unwrap().incremental_interval_95 = Some([0.0, 0.0]);
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &body,
            vec!["outcome".into()],
            "forged-policy"
        )
        .is_err()
    );

    let mut constrained_query = query();
    constrained_query.assignment = (0..n).map(|i| i % 2 == 1).collect::<Vec<_>>().into();
    constrained_query.actions = (0..n).map(|i| i % 3 == 0).collect::<Vec<_>>().into();
    constrained_query.reference = vec![false; n].into();
    constrained_query.mu0 = vec![1.0; n].into();
    constrained_query.mu1 = vec![3.0; n].into();
    constrained_query.evaluation_subject_ids =
        (0..n).map(|i| Arc::<str>::from(format!("evaluation-{i}"))).collect::<Vec<_>>().into();
    constrained_query.global_constraints_present = true;
    let constrained = Study::tabular(data.clone())
        .query(CausalQuery::PolicyValue(constrained_query))
        .build()
        .unwrap();
    let constrained_result = constrained.prepare(&ctx).unwrap().estimate(&data, &ctx).unwrap();
    assert!(constrained_result.policy_value.unwrap().policy_interval_95.is_none());
}

#[test]
fn retained_crossfit_policy_intervals_round_trip_and_refuse_forgery() {
    let n = 300;
    let assignment = (0..n).map(|i| i % 2 == 1).collect::<Vec<_>>();
    let actions = (0..n).map(|i| i % 3 == 0).collect::<Vec<_>>();
    let outcomes = (0..n)
        .map(|i| 1.0 + 2.0 * f64::from(assignment[i]) + (i % 5) as f64 / 10.0)
        .collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("outcome", outcomes.as_slice())]).unwrap();
    let mut q = query();
    q.assignment = assignment.into();
    q.actions = actions.into();
    q.reference = vec![false; n].into();
    q.mu0 = vec![1.0; n].into();
    q.mu1 = vec![3.0; n].into();
    // Cross-fitted nuisance ownership: caller-declared excluded folds, not
    // disjoint training subjects. The interval is licensed on this row.
    q.disjoint_training_subjects = false;
    q.crossfit_fold_ownership_valid = true;
    q.evaluation_subject_ids =
        (0..n).map(|i| Arc::<str>::from(format!("evaluation-{i}"))).collect::<Vec<_>>().into();
    let ctx = ExecutionContext::for_tests(77);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::PolicyValue(q.clone()))
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let value = result.policy_value.as_ref().unwrap();
    let policy_ci =
        value.policy_interval_95.expect("cross-fitted AIPW publishes a policy interval");
    let incremental_ci =
        value.incremental_interval_95.expect("cross-fitted AIPW publishes an incremental interval");
    assert!(policy_ci[0] < value.policy_value && value.policy_value < policy_ci[1]);
    assert!(
        incremental_ci[0] < value.incremental_value && value.incremental_value < incremental_ci[1]
    );
    assert_eq!(
        value.prediction_ownership.as_ref(),
        "caller_declared_cross_fitted_excluded_fold_ids"
    );
    assert_eq!(result.support_status, Some(antecedent::support::CellStatus::Licensed));
    let bytes = prepared.encode_contracted_result(&result, "crossfit-policy", &ctx).unwrap();
    let (_, _, mut body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(
        body.policy_value.as_ref().unwrap().graphless_support_status.as_deref(),
        Some("licensed")
    );
    assert_eq!(body.policy_value.as_ref().unwrap().policy_interval_95, Some(policy_ci));
    body.policy_value.as_mut().unwrap().incremental_interval_95 = Some([0.0, 0.0]);
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &body,
            vec!["outcome".into()],
            "forged-crossfit"
        )
        .is_err()
    );
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
#[allow(clippy::too_many_lines)] // one end-to-end route fixture; splitting would fragment the flow
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
    q.evaluation_subject_ids =
        (0..9).map(|i| Arc::<str>::from(format!("s{i}"))).collect::<Vec<_>>().into();
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
        cate_groups: Arc::from([]),
    });
    let study =
        Study::tabular(data.clone()).query(CausalQuery::PolicyValue(q.clone())).build().unwrap();
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
    assert_eq!(
        answer.uncertainty.as_ref(),
        "multi_action_ipw_row_score_standard_error_independent_subjects"
    );
    let artifact = prepared.encode_contracted_result(&result, "multi-policy", &ctx).unwrap();
    let (_, _, body) = antecedent_io::decode_analysis_result_artifact(&artifact).unwrap();
    assert_eq!(body.policy_value.as_ref().unwrap().policy_value, answer.policy_value);
    assert_eq!(
        body.query,
        antecedent_io::causal_query_to_wire(&CausalQuery::PolicyValue(q.clone())).unwrap()
    );
    q.multi_action.as_mut().unwrap().cate_groups =
        ["x", "x", "x", "y", "y", "y", "z", "z", "z"].map(Arc::<str>::from).into();
    let grouped =
        Study::tabular(data.clone()).query(CausalQuery::PolicyValue(q.clone())).build().unwrap();
    let grouped_prepared = grouped.prepare(&ctx).unwrap();
    let grouped_result = grouped_prepared.estimate(&data, &ctx).unwrap();
    let points = &grouped_result.policy_value.as_ref().unwrap().multi_action_cate;
    assert_eq!(points.len(), 6);
    for point in points {
        assert_eq!(point.evaluation_rows, 3);
        assert_eq!(point.observed_action_rows, 1);
        assert_eq!(point.observed_control_rows, 1);
        assert!((point.effect - if point.action == "A" { 1.0 } else { 3.0 }).abs() < 1e-12);
        assert!(point.interval_95.is_none());
    }
    let grouped_artifact =
        grouped_prepared.encode_contracted_result(&grouped_result, "grouped-policy", &ctx).unwrap();
    let (_, _, mut grouped_body) =
        antecedent_io::decode_analysis_result_artifact(&grouped_artifact).unwrap();
    assert_eq!(grouped_body.policy_value.as_ref().unwrap().multi_action_cate[0].effect, 1.0);
    grouped_body.policy_value.as_mut().unwrap().graphless_support_status = Some("licensed".into());
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &grouped_body,
            vec!["outcome".into()],
            "forged-policy-license"
        )
        .is_err()
    );
    assert_eq!(
        grouped_body.query,
        antecedent_io::causal_query_to_wire(&CausalQuery::PolicyValue(q.clone())).unwrap()
    );
    q.multi_action.as_mut().unwrap().cate_groups = Arc::from([Arc::<str>::from("x")]);
    assert!(q.validate().is_err());
    q.multi_action.as_mut().unwrap().cate_groups =
        ["x", "y", "y", "x", "y", "y", "x", "y", "y"].map(Arc::<str>::from).into();
    assert!(q.validate().is_ok());
    let unsupported =
        Study::tabular(data.clone()).query(CausalQuery::PolicyValue(q.clone())).build().unwrap();
    let unsupported_prepared = unsupported.prepare(&ctx).unwrap();
    assert!(
        unsupported_prepared.estimate(&data, &ctx).is_err(),
        "a stratum without every observed action must refuse CATE estimation"
    );
    q.multi_action.as_mut().unwrap().cate_groups = Arc::from([]);
    q.multi_action.as_mut().unwrap().capacities = [9, 2, 9].into();
    assert!(q.validate().is_err());
    q.multi_action.as_mut().unwrap().capacities = [9; 3].into();
    q.multi_action.as_mut().unwrap().propensities = vec![0.0; 27].into();
    assert!(q.validate().is_err());
}

#[test]
fn multi_action_intervals_require_nonbinding_global_constraints() {
    let n = 300;
    let assigned = (0..n).map(|i| i % 3).collect::<Vec<_>>();
    let y =
        (0..n).map(|i| [1.0, 2.0, 4.0][assigned[i]] + (i % 7) as f64 / 10.0).collect::<Vec<_>>();
    let data = TabularData::from_f64_columns([("outcome", y.as_slice())]).unwrap();
    let mut q = query();
    q.assignment = Arc::from([]);
    q.propensity = Arc::from([]);
    q.actions = Arc::from([]);
    q.reference = Arc::from([]);
    q.mu0 = Arc::from([]);
    q.mu1 = Arc::from([]);
    q.costs = Arc::from([]);
    q.reference_costs = Arc::from([]);
    q.disjoint_training_subjects = false;
    q.evaluation_subject_ids =
        (0..n).map(|i| Arc::<str>::from(format!("multi-{i}"))).collect::<Vec<_>>().into();
    q.multi_action = Some(MultiActionPolicyInputs {
        action_labels: ["control", "A", "B"].map(Arc::<str>::from).into(),
        assignment: assigned.clone().into(),
        actions: assigned.into(),
        reference: vec![0; n].into(),
        propensities: vec![1.0 / 3.0; n * 3].into(),
        available: vec![true; n * 3].into(),
        costs: [0.0, 0.1, 0.2].into(),
        reference_costs: [0.0; 3].into(),
        capacities: [n; 3].into(),
        reference_capacities: [n; 3].into(),
        budget: Some(n as f64 * 0.2),
        reference_budget: None,
        cate_groups: vec![Arc::<str>::from("g0"); n].into(),
    });
    let ctx = ExecutionContext::for_tests(45);
    let study =
        Study::tabular(data.clone()).query(CausalQuery::PolicyValue(q.clone())).build().unwrap();
    let prepared = study.prepare(&ctx).unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let value = result.policy_value.as_ref().unwrap();
    assert!(value.policy_interval_95.is_some());
    assert!(value.incremental_interval_95.is_some());
    assert_eq!(result.support_status, Some(antecedent::support::CellStatus::Licensed));
    assert_eq!(value.multi_action_cate.len(), 2);
    for (point, truth) in value.multi_action_cate.iter().zip([1.0, 3.0]) {
        let bounds = point.interval_95.expect("300 randomized subjects and 100 observed per arm");
        assert!(bounds[0] < truth && truth < bounds[1]);
    }
    let bytes = prepared.encode_contracted_result(&result, "multi-interval", &ctx).unwrap();
    let (_, _, mut body) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    assert_eq!(body.policy_value.as_ref().unwrap().policy_interval_95, value.policy_interval_95);
    assert_eq!(
        body.policy_value.as_ref().unwrap().multi_action_cate[0].interval_95,
        value.multi_action_cate[0].interval_95
    );
    body.policy_value.as_mut().unwrap().multi_action_cate[0].interval_95 = Some([0.0, 0.0]);
    assert!(
        antecedent_io::encode_analysis_result_artifact(
            &body,
            vec!["outcome".into()],
            "forged-cate"
        )
        .is_err()
    );

    q.multi_action.as_mut().unwrap().budget = Some(40.0);
    let constrained =
        Study::tabular(data.clone()).query(CausalQuery::PolicyValue(q.clone())).build().unwrap();
    let constrained_value =
        constrained.prepare(&ctx).unwrap().estimate(&data, &ctx).unwrap().policy_value.unwrap();
    assert!(constrained_value.policy_interval_95.is_none());
    q.multi_action.as_mut().unwrap().budget = None;
    q.multi_action.as_mut().unwrap().capacities = [n, n, n / 3].into();
    let capped = Study::tabular(data.clone()).query(CausalQuery::PolicyValue(q)).build().unwrap();
    let capped_value =
        capped.prepare(&ctx).unwrap().estimate(&data, &ctx).unwrap().policy_value.unwrap();
    assert!(capped_value.policy_interval_95.is_none());
}
