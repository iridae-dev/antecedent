//! The facade policy-value evaluators and the compiled `PolicyValue` route give the same
//! numbers.
//!
//! * Doubly robust binary value: `evaluate_policy_value_scores` is the kernel call the compiled
//!   route makes, so every value, standard error and interval bound is compared on `to_bits()`.
//! * Held-out uplift bins: `uplift_by_score_bin` and the compiled bins form the same row scores
//!   and the same bin means and standard errors, compared on `to_bits()`. They differ only in
//!   what they refuse (a one-row bin is a `NaN` error on the direct path and a refusal on the
//!   compiled route).
//! * Inverse-probability binary and multi-action value: the direct evaluators and the compiled
//!   route reach the same estimand through different kernels (a sum divided by `n` minus the
//!   mean cost, against the mean of per-row scores), so they agree to rounding, not always to
//!   the last bit. The fixtures with dyadic values compare on `to_bits()`; the others use a
//!   relative tolerance.
#![allow(
    clippy::float_cmp,
    clippy::cast_lossless,
    reason = "bit-identical equality is the property under test; fixtures widen small integers"
)]
#![allow(deprecated, reason = "exercises the deprecated direct facade path")]

use std::sync::Arc;

use antecedent::estimate::policy_value::{
    BinaryIpwDesign, BinaryPolicyLimits, MultiActionIpwDesign, MultiActionLimits,
    evaluate_binary_ipw_policy, evaluate_multi_action_ipw_policy, evaluate_policy_value_scores,
    uplift_by_score_bin,
};
use antecedent::prelude::ExecutionContext;
use antecedent::{PolicyValueEstimate, Study};
use antecedent_core::{CausalQuery, MultiActionPolicyInputs, PolicyValueQuery, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::policy_value::{BinaryPolicyDesign, pointwise_intervals_95};

fn base_query(n: usize) -> PolicyValueQuery {
    PolicyValueQuery {
        outcome: VariableId::from_raw(0),
        assignment: Arc::from(vec![false; n]),
        propensity: Arc::from([0.5]),
        actions: Arc::from(vec![false; n]),
        reference: Arc::from(vec![false; n]),
        mu0: Arc::from([]),
        mu1: Arc::from([]),
        costs: Arc::from([0.0]),
        reference_costs: Arc::from([0.0]),
        evaluation_subject_ids: (0..n).map(|i| Arc::<str>::from(format!("eval-{i}"))).collect(),
        disjoint_training_subjects: false,
        crossfit_fold_ownership_valid: false,
        global_constraints_present: false,
        multi_action: None,
        uplift_bins: Arc::from([]),
        uplift_bin_count: 0,
        uplift_training_subject_ids: Arc::from([]),
        regret: None,
    }
}

fn run(outcome: &[f64], query: PolicyValueQuery) -> PolicyValueEstimate {
    let data = TabularData::from_f64_columns([("outcome", outcome)]).unwrap();
    let context = ExecutionContext::for_tests(31);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::PolicyValue(query))
        .build()
        .unwrap()
        .prepare(&context)
        .unwrap();
    let result = prepared.estimate(&data, &context).unwrap();
    result.policy_value().unwrap().clone()
}

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 1e-12 * want.abs().max(1.0), "{what}: got {got}, want {want}");
}

struct Binary {
    outcome: Vec<f64>,
    assignment: Vec<bool>,
    actions: Vec<bool>,
    reference: Vec<bool>,
}

fn binary(n: usize) -> Binary {
    let assignment: Vec<bool> = (0..n).map(|i| i % 2 == 1).collect();
    Binary {
        outcome: (0..n)
            .map(|i| 1.0 + 2.0 * f64::from(assignment[i]) + (i % 5) as f64 / 10.0)
            .collect(),
        actions: (0..n).map(|i| i % 3 == 0).collect(),
        reference: (0..n).map(|i| i % 4 == 0).collect(),
        assignment,
    }
}

#[test]
fn compiled_doubly_robust_policy_value_reproduces_the_direct_scores_and_intervals() {
    let n = 400;
    let fixture = binary(n);
    let mu0: Vec<f64> = (0..n).map(|i| 1.0 + 0.01 * (i % 7) as f64).collect();
    let mu1: Vec<f64> = (0..n).map(|i| 3.0 + 0.02 * (i % 5) as f64).collect();
    let propensity = [0.5];
    let costs = [0.1];
    let direct = evaluate_policy_value_scores(
        &fixture.outcome,
        &BinaryPolicyDesign {
            assignment: &fixture.assignment,
            actions: &fixture.actions,
            propensity: &propensity,
            reference: &fixture.reference,
            costs: &costs,
            reference_costs: &costs,
        },
        &mu0,
        &mu1,
    )
    .unwrap();

    let mut query = base_query(n);
    query.assignment = fixture.assignment.clone().into();
    query.actions = fixture.actions.clone().into();
    query.reference = fixture.reference.clone().into();
    query.mu0 = mu0.into();
    query.mu1 = mu1.into();
    query.costs = Arc::from(costs);
    query.reference_costs = Arc::from(costs);
    query.disjoint_training_subjects = true;
    let compiled = run(&fixture.outcome, query);

    let pairs = [
        (compiled.policy_value, direct.policy_value, "policy value"),
        (compiled.reference_value, direct.reference_value, "reference value"),
        (compiled.incremental_value, direct.incremental_value, "incremental value"),
        (compiled.relative_value_gap, direct.relative_value_gap, "relative gap"),
        (compiled.treatment_rate, direct.treatment_rate, "treatment rate"),
        (compiled.total_cost, direct.total_cost, "total cost"),
        (compiled.policy_standard_error, direct.policy_standard_error, "policy se"),
        (compiled.reference_standard_error, direct.reference_standard_error, "reference se"),
        (compiled.incremental_standard_error, direct.incremental_standard_error, "incremental se"),
        (compiled.propensity_min, direct.propensity_min, "propensity min"),
        (compiled.propensity_max, direct.propensity_max, "propensity max"),
    ];
    for (got, want, what) in pairs {
        assert_eq!(got.to_bits(), want.to_bits(), "{what}");
    }

    // The interval is the kernel interval of the direct scores; the direct path publishes none.
    let matches = |policy: &[bool]| {
        fixture
            .assignment
            .iter()
            .zip(policy)
            .filter(|(assigned, action)| assigned == action)
            .count()
    };
    let kernel =
        pointwise_intervals_95(&direct, n, matches(&fixture.actions), matches(&fixture.reference));
    assert_eq!(compiled.policy_interval_95.is_some(), kernel.is_some());
    if let Some(kernel) = kernel {
        let policy = compiled.policy_interval_95.unwrap();
        let incremental = compiled.incremental_interval_95.unwrap();
        assert_eq!(policy.map(f64::to_bits), kernel.policy.map(f64::to_bits));
        assert_eq!(incremental.map(f64::to_bits), kernel.incremental.map(f64::to_bits));
    }
}

#[test]
fn compiled_ipw_policy_value_reproduces_the_direct_point_value() {
    // Dyadic outcomes, propensity and costs make every sum exact, so both kernels agree to
    // the last bit.
    let n = 64;
    let assignment: Vec<bool> = (0..n).map(|i| i % 2 == 1).collect();
    let outcome: Vec<f64> = (0..n).map(|i| 1.0 + 0.25 * (i % 4) as f64).collect();
    let actions: Vec<bool> = (0..n).map(|i| i % 4 < 2).collect();
    let reference = vec![false; n];
    let costs = [0.25];
    let direct = evaluate_binary_ipw_policy(
        &outcome,
        &BinaryIpwDesign {
            assignment: &assignment,
            actions: &actions,
            propensity: &[0.5],
            reference: None,
            costs: &costs,
            reference_costs: None,
            available: None,
        },
        &BinaryPolicyLimits::default(),
    )
    .unwrap();
    let mut query = base_query(n);
    query.assignment = assignment.into();
    query.actions = actions.into();
    query.reference = reference.into();
    query.costs = Arc::from(costs);
    query.reference_costs = Arc::from(costs);
    let compiled = run(&outcome, query);
    assert_eq!(compiled.policy_value.to_bits(), direct.policy_net.to_bits());
    assert_eq!(compiled.reference_value.to_bits(), direct.reference_net.to_bits());
    assert_eq!(compiled.incremental_value.to_bits(), direct.incremental_value.to_bits());
    assert_eq!(compiled.relative_value_gap.to_bits(), direct.regret.to_bits());
    assert_eq!(compiled.treatment_rate.to_bits(), direct.treatment_rate.to_bits());
    assert_eq!(compiled.total_cost.to_bits(), direct.total_cost.to_bits());
}

#[test]
fn compiled_ipw_policy_value_agrees_with_the_direct_value_to_rounding() {
    let n = 300;
    let fixture = binary(n);
    let propensity: Vec<f64> = (0..n).map(|i| 0.3 + 0.01 * (i % 9) as f64).collect();
    let costs = [0.1];
    let direct = evaluate_binary_ipw_policy(
        &fixture.outcome,
        &BinaryIpwDesign {
            assignment: &fixture.assignment,
            actions: &fixture.actions,
            propensity: &propensity,
            reference: Some(&fixture.reference),
            costs: &costs,
            reference_costs: None,
            available: None,
        },
        &BinaryPolicyLimits::default(),
    )
    .unwrap();
    let mut query = base_query(n);
    query.assignment = fixture.assignment.clone().into();
    query.actions = fixture.actions.clone().into();
    query.reference = fixture.reference.clone().into();
    query.propensity = propensity.into();
    query.costs = Arc::from(costs);
    query.reference_costs = Arc::from(costs);
    let compiled = run(&fixture.outcome, query);
    close(compiled.policy_value, direct.policy_net, "policy value");
    close(compiled.reference_value, direct.reference_net, "reference value");
    close(compiled.incremental_value, direct.incremental_value, "incremental value");
    close(compiled.relative_value_gap, direct.regret, "relative gap");
    assert_eq!(compiled.treatment_rate.to_bits(), direct.treatment_rate.to_bits());
}

#[test]
fn compiled_uplift_bins_reproduce_the_direct_bin_contrasts() {
    let n = 600;
    let fixture = binary(n);
    let bins = 4;
    let score_bin: Vec<usize> = (0..n).map(|i| (i * bins) / n).collect();
    let propensity = [0.5];
    let direct =
        uplift_by_score_bin(&fixture.outcome, &fixture.assignment, &score_bin, &propensity, bins)
            .unwrap();
    let mut query = base_query(n);
    query.assignment = fixture.assignment.clone().into();
    query.actions = fixture.actions.clone().into();
    query.reference = fixture.reference.clone().into();
    query.uplift_bins = score_bin.into();
    query.uplift_bin_count = bins;
    query.uplift_training_subject_ids = Arc::from([Arc::<str>::from("rank-train")]);
    let compiled = run(&fixture.outcome, query);
    assert_eq!(compiled.uplift_bins.len(), direct.len());
    for (bin, expected) in compiled.uplift_bins.iter().zip(&direct) {
        assert_eq!(bin.rank, expected.index);
        assert_eq!(bin.evaluation_rows, expected.rows);
        assert_eq!(bin.effect.to_bits(), expected.effect.to_bits());
        assert_eq!(bin.standard_error.to_bits(), expected.standard_error.to_bits());
    }
}

#[test]
fn compiled_multi_action_value_reproduces_the_direct_point_value() {
    // Dyadic outcomes and propensities (one half, one quarter, one quarter) keep every sum
    // exact.
    let n = 64;
    let k = 3;
    let outcome: Vec<f64> = (0..n).map(|i| 1.0 + 0.5 * (i % 4) as f64).collect();
    let assigned: Vec<usize> = (0..n).map(|i| [0, 0, 1, 2][i % 4]).collect();
    let actions: Vec<usize> = (0..n).map(|i| [0, 1, 1, 2][i % 4]).collect();
    let probabilities: Vec<f64> = (0..n).flat_map(|_| [0.5, 0.25, 0.25]).collect();
    let costs = [0.0, 0.25, 0.5];
    let direct = evaluate_multi_action_ipw_policy(
        &outcome,
        &MultiActionIpwDesign {
            assigned: &assigned,
            actions: &actions,
            probabilities: &probabilities,
            probability_rows: n,
            action_count: k,
            reference: None,
            costs: Some(&costs),
            reference_costs: None,
            available: None,
        },
        &MultiActionLimits::default(),
    )
    .unwrap();
    let mut query = base_query(n);
    query.assignment = Vec::new().into();
    query.propensity = Vec::new().into();
    query.actions = Vec::new().into();
    query.reference = Vec::new().into();
    query.costs = Vec::new().into();
    query.reference_costs = Vec::new().into();
    query.multi_action = Some(MultiActionPolicyInputs {
        action_labels: ["control", "A", "B"].map(Arc::<str>::from).into(),
        assignment: assigned.into(),
        actions: actions.into(),
        reference: vec![0; n].into(),
        propensities: probabilities.into(),
        available: vec![true; n * k].into(),
        costs: costs.into(),
        reference_costs: costs.into(),
        capacities: [n; 3].into(),
        reference_capacities: [n; 3].into(),
        budget: None,
        reference_budget: None,
        cate_groups: Arc::from([]),
    });
    let compiled = run(&outcome, query);
    assert_eq!(compiled.policy_value.to_bits(), direct.policy_value.to_bits());
    assert_eq!(compiled.reference_value.to_bits(), direct.reference_value.to_bits());
    assert_eq!(compiled.incremental_value.to_bits(), direct.incremental_value.to_bits());
    assert_eq!(compiled.treatment_rate.to_bits(), direct.non_control_rate.to_bits());
    assert_eq!(compiled.total_cost.to_bits(), direct.policy_cost.to_bits());
}
