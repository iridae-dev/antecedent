//! The capacity, budget and availability checks of the point-only policy evaluators on the
//! compiled net-value routes.
//!
//! * Binary: `evaluate_binary_ipw_policy` takes the policy's and the reference's capacity and
//!   budget and a per-row availability mask. A study states them with
//!   `StudyBuilder::binary_policy_constraints`, and the compiled `PolicyValue` route under
//!   `randomized.ipw_policy_net` passes them to that same kernel: every broken limit is refused
//!   with the kernel's own message, every policy that keeps its limits has the values the kernel
//!   returns (and the values the unconstrained route returns), to the bit. The declaration enters
//!   the program identity only when it is made, and is refused where nothing would check it.
//! * Multi-action: `evaluate_multi_action_ipw_policy` takes availability, per-action capacities
//!   and budgets; the multi-action query carries all of them and its validation refuses the same
//!   conditions before the route runs, while a policy that keeps them has the kernel's values.
//!
//! None of these kernels draws randomness, so there is no seed to pin.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    deprecated,
    reason = "the direct facade kernels are the paths under comparison and become deprecated wrappers"
)]
#![allow(clippy::float_cmp, reason = "bit-identical equality is the property under test")]

use std::sync::Arc;

use antecedent::analysis::binary_policy_constraints::BinaryPolicyConstraints;
use antecedent::estimate::policy_value::{
    BinaryIpwDesign, BinaryPolicyLimits, MultiActionIpwDesign, MultiActionLimits,
    evaluate_binary_ipw_policy, evaluate_multi_action_ipw_policy,
};
use antecedent::prelude::ExecutionContext;
use antecedent::{CausalError, EstimatorId, PolicyValueEstimate, Study, StudyResult};
use antecedent_core::{CausalQuery, MultiActionPolicyInputs, PolicyValueQuery, VariableId};
use antecedent_data::TabularData;
use antecedent_estimate::policy_value::{BinaryIpwPolicy, MultiActionIpwPolicy};

const N: usize = 300;
const COST: f64 = 0.1;
const REFERENCE_COST: f64 = 0.07;

fn labels(prefix: &str, n: usize) -> Vec<Arc<str>> {
    (0..n).map(|i| Arc::<str>::from(format!("{prefix}-{i}"))).collect()
}

fn same(left: f64, right: f64, what: &str) {
    assert_eq!(left.to_bits(), right.to_bits(), "{what}: {left} != {right}");
}

fn same_interval(left: Option<[f64; 2]>, right: Option<[f64; 2]>, what: &str) {
    match (left, right) {
        (Some(a), Some(b)) => {
            same(a[0], b[0], what);
            same(a[1], b[1], what);
        }
        (None, None) => {}
        other => panic!("{what}: interval presence differs: {other:?}"),
    }
}

/// A binary fixture whose policy and reference treat only available rows.
struct Binary {
    outcome: Vec<f64>,
    assignment: Vec<bool>,
    actions: Vec<bool>,
    reference: Vec<bool>,
    propensity: Vec<f64>,
    available: Vec<bool>,
}

fn binary() -> Binary {
    let available: Vec<bool> = (0..N).map(|i| i % 7 != 6).collect();
    let assignment: Vec<bool> = (0..N).map(|i| (i * 7) % 5 < 2).collect();
    Binary {
        outcome: (0..N)
            .map(|i| 1.0 + 0.37 * (i % 11) as f64 + 0.13 * f64::from(u8::from(assignment[i])))
            .collect(),
        actions: (0..N).map(|i| i % 3 != 0 && available[i]).collect(),
        reference: (0..N).map(|i| i % 4 == 0 && available[i]).collect(),
        propensity: (0..N).map(|i| 0.3 + 0.01 * (i % 9) as f64).collect(),
        assignment,
        available,
    }
}

impl Binary {
    fn query(&self) -> PolicyValueQuery {
        PolicyValueQuery {
            outcome: VariableId::from_raw(0),
            assignment: self.assignment.clone().into(),
            propensity: self.propensity.clone().into(),
            actions: self.actions.clone().into(),
            reference: self.reference.clone().into(),
            mu0: Arc::from([]),
            mu1: Arc::from([]),
            costs: Arc::from([COST]),
            reference_costs: Arc::from([REFERENCE_COST]),
            evaluation_subject_ids: labels("eval", N).into(),
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

    /// The direct point-only evaluator under `constraints`.
    fn direct(
        &self,
        constraints: &BinaryPolicyConstraints,
    ) -> Result<BinaryIpwPolicy, &'static str> {
        evaluate_binary_ipw_policy(
            &self.outcome,
            &BinaryIpwDesign {
                assignment: &self.assignment,
                actions: &self.actions,
                propensity: &self.propensity,
                reference: Some(&self.reference),
                costs: &[COST],
                reference_costs: Some(&[REFERENCE_COST]),
                available: constraints.available.as_deref(),
            },
            &BinaryPolicyLimits {
                capacity: constraints.capacity,
                budget: constraints.budget,
                reference_capacity: constraints.reference_capacity,
                reference_budget: constraints.reference_budget,
            },
        )
    }

    fn treated(&self) -> usize {
        self.actions.iter().filter(|&&treat| treat).count()
    }

    fn reference_treated(&self) -> usize {
        self.reference.iter().filter(|&&treat| treat).count()
    }

    /// Limits every one of which the fixture's policy and reference meet with equality.
    fn binding(&self) -> BinaryPolicyConstraints {
        let unconstrained = self.direct(&BinaryPolicyConstraints::new()).unwrap();
        BinaryPolicyConstraints::new()
            .with_capacity(self.treated())
            .with_budget(unconstrained.total_cost)
            .with_reference_capacity(self.reference_treated())
            .with_reference_budget(REFERENCE_COST * self.reference_treated() as f64)
            .with_available(self.available.clone())
    }
}

/// Build, prepare and estimate a policy-value study on `outcome`.
fn run(
    outcome: &[f64],
    query: PolicyValueQuery,
    estimator: Option<EstimatorId>,
    constraints: Option<BinaryPolicyConstraints>,
) -> Result<StudyResult, CausalError> {
    let data = TabularData::from_f64_columns([("outcome", outcome)]).unwrap();
    let context = ExecutionContext::for_tests(31);
    let mut builder = Study::tabular(data.clone()).query(CausalQuery::PolicyValue(query));
    if let Some(estimator) = estimator {
        builder = builder.estimator(estimator);
    }
    if let Some(constraints) = constraints {
        builder = builder.binary_policy_constraints(constraints);
    }
    let prepared = builder.build()?.prepare(&context)?;
    prepared.estimate(&data, &context)
}

fn value(result: StudyResult) -> PolicyValueEstimate {
    result.into_policy_value().expect("a policy-value result")
}

fn assert_same_estimate(left: &PolicyValueEstimate, right: &PolicyValueEstimate, what: &str) {
    same(left.policy_value, right.policy_value, &format!("{what}: policy value"));
    same(left.reference_value, right.reference_value, &format!("{what}: reference value"));
    same(left.incremental_value, right.incremental_value, &format!("{what}: incremental value"));
    same(left.relative_value_gap, right.relative_value_gap, &format!("{what}: relative gap"));
    same(left.treatment_rate, right.treatment_rate, &format!("{what}: treatment rate"));
    same(left.total_cost, right.total_cost, &format!("{what}: total cost"));
    same(left.policy_standard_error, right.policy_standard_error, &format!("{what}: policy SE"));
    same(
        left.incremental_standard_error,
        right.incremental_standard_error,
        &format!("{what}: incremental SE"),
    );
    same_interval(
        left.policy_interval_95,
        right.policy_interval_95,
        &format!("{what}: policy interval"),
    );
    same_interval(
        left.incremental_interval_95,
        right.incremental_interval_95,
        &format!("{what}: incremental interval"),
    );
}

#[test]
fn binary_limits_that_hold_leave_the_net_value_bit_identical() {
    let fixture = binary();
    let constraints = fixture.binding();
    let direct = fixture.direct(&constraints).expect("the fixture keeps every limit");
    let constrained = value(
        run(
            &fixture.outcome,
            fixture.query(),
            Some(EstimatorId::RandomizedIpwPolicyNet),
            Some(constraints),
        )
        .unwrap(),
    );
    same(constrained.policy_value, direct.policy_net, "policy value");
    same(constrained.reference_value, direct.reference_net, "reference value");
    same(constrained.incremental_value, direct.incremental_value, "incremental value");
    same(constrained.relative_value_gap, direct.regret, "relative gap");
    same(constrained.treatment_rate, direct.treatment_rate, "treatment rate");
    same(constrained.total_cost, direct.total_cost, "total cost");
    // The limits are checks: an accepted policy has the unconstrained route's estimate,
    // standard errors and intervals.
    let unconstrained = value(
        run(&fixture.outcome, fixture.query(), Some(EstimatorId::RandomizedIpwPolicyNet), None)
            .unwrap(),
    );
    assert!(unconstrained.policy_interval_95.is_some(), "the fixture publishes intervals");
    assert_same_estimate(&constrained, &unconstrained, "constrained versus unconstrained");
}

#[test]
fn each_broken_binary_limit_is_refused_with_the_direct_kernel_message() {
    let fixture = binary();
    let binding = fixture.binding();
    let mut unavailable_policy_row = vec![true; N];
    let treated_row = fixture.actions.iter().position(|&treat| treat).unwrap();
    assert!(!fixture.reference[treated_row]);
    unavailable_policy_row[treated_row] = false;
    let mut unavailable_reference_row = vec![true; N];
    let reference_row = fixture
        .reference
        .iter()
        .zip(&fixture.actions)
        .position(|(&reference, &policy)| reference && !policy)
        .unwrap();
    unavailable_reference_row[reference_row] = false;
    let cases = [
        ("capacity", BinaryPolicyConstraints::new().with_capacity(fixture.treated() - 1)),
        ("budget", BinaryPolicyConstraints::new().with_budget(binding.budget.unwrap() - 0.5)),
        (
            "reference capacity",
            BinaryPolicyConstraints::new().with_reference_capacity(fixture.reference_treated() - 1),
        ),
        (
            "reference budget",
            BinaryPolicyConstraints::new()
                .with_reference_budget(binding.reference_budget.unwrap() - 0.5),
        ),
        (
            "policy availability",
            BinaryPolicyConstraints::new().with_available(unavailable_policy_row),
        ),
        (
            "reference availability",
            BinaryPolicyConstraints::new().with_available(unavailable_reference_row),
        ),
        ("availability length", BinaryPolicyConstraints::new().with_available(vec![true; N - 1])),
        ("negative budget", binding.clone().with_budget(-1.0)),
        ("non-finite reference budget", binding.clone().with_reference_budget(f64::NAN)),
        ("capacity beside holding limits", binding.with_capacity(fixture.treated() - 1)),
    ];
    for (what, constraints) in cases {
        let direct = fixture.direct(&constraints).expect_err(what);
        let compiled = run(
            &fixture.outcome,
            fixture.query(),
            Some(EstimatorId::RandomizedIpwPolicyNet),
            Some(constraints),
        )
        .expect_err(what);
        assert!(
            matches!(&compiled, CausalError::Unsupported { message } if *message == direct),
            "{what}: the compiled route refused with {compiled:?}, the direct kernel with {direct:?}"
        );
    }
}

#[test]
fn declared_limits_enter_the_program_identity_only_when_declared() {
    let fixture = binary();
    let identity = |constraints: Option<BinaryPolicyConstraints>| {
        run(
            &fixture.outcome,
            fixture.query(),
            Some(EstimatorId::RandomizedIpwPolicyNet),
            constraints,
        )
        .unwrap()
        .lowered_program_identity()
        .expect("the run is lowered")
    };
    let undeclared = identity(None);
    assert_eq!(undeclared, identity(None));
    let declared = identity(Some(fixture.binding()));
    assert_ne!(declared, undeclared, "declared limits change the program");
    assert_eq!(declared, identity(Some(fixture.binding())), "the same limits, the same program");
    let looser = fixture.binding().with_capacity(fixture.treated() + 1);
    assert_ne!(identity(Some(looser)), declared, "each limit is part of the program");
}

#[test]
fn binary_limits_are_refused_where_nothing_checks_them() {
    let fixture = binary();
    let refused = |query: PolicyValueQuery, estimator: Option<EstimatorId>| {
        let data =
            TabularData::from_f64_columns([("outcome", fixture.outcome.as_slice())]).unwrap();
        let mut builder = Study::tabular(data)
            .query(CausalQuery::PolicyValue(query))
            .binary_policy_constraints(fixture.binding());
        if let Some(estimator) = estimator {
            builder = builder.estimator(estimator);
        }
        builder.build().is_err()
    };
    assert!(refused(fixture.query(), None), "the default estimator does not read the limits");
    assert!(
        refused(fixture.query(), Some(EstimatorId::RandomizedIpwPolicy)),
        "the row-score estimator does not read the limits"
    );
    let (_, multi) = multi_action();
    assert!(
        refused(multi.query(), Some(EstimatorId::RandomizedMultiActionIpwPolicyNet)),
        "a multi-action policy states its limits in its inputs"
    );
    assert!(!refused(fixture.query(), Some(EstimatorId::RandomizedIpwPolicyNet)));
}

/// A three-action fixture whose policy is the only one to use action `B`, which is available
/// exactly where the policy recommends it.
#[derive(Clone)]
struct MultiAction {
    assigned: Vec<usize>,
    actions: Vec<usize>,
    reference: Vec<usize>,
    probabilities: Vec<f64>,
    available: Vec<bool>,
    costs: [f64; 3],
    reference_costs: [f64; 3],
    capacities: [usize; 3],
    reference_capacities: [usize; 3],
    budget: Option<f64>,
    reference_budget: Option<f64>,
}

const K: usize = 3;

fn multi_action() -> (Vec<f64>, MultiAction) {
    let outcome = (0..N).map(|i| 1.0 + 0.31 * (i % 7) as f64 + 0.11 * (i % 3) as f64).collect();
    let actions: Vec<usize> = (0..N).map(|i| [0, 1, 1, 2][i % 4]).collect();
    let available = actions.iter().flat_map(|&action| [true, true, action == 2]).collect();
    (
        outcome,
        MultiAction {
            assigned: (0..N).map(|i| [0, 1, 2, 0, 2][i % 5]).collect(),
            reference: (0..N).map(|i| usize::from(i % 6 == 0)).collect(),
            probabilities: (0..N)
                .flat_map(|i| if i % 2 == 0 { [0.4, 0.35, 0.25] } else { [0.5, 0.3, 0.2] })
                .collect(),
            actions,
            available,
            costs: [0.0, 0.15, 0.4],
            reference_costs: [0.0, 0.1, 0.3],
            capacities: [N; K],
            reference_capacities: [N; K],
            budget: None,
            reference_budget: None,
        },
    )
}

fn counts(policy: &[usize]) -> [usize; K] {
    let mut counts = [0; K];
    for &action in policy {
        counts[action] += 1;
    }
    counts
}

fn cost(policy: &[usize], costs: &[f64; K]) -> f64 {
    policy.iter().map(|&action| costs[action]).sum()
}

impl MultiAction {
    /// Limits every one of which the fixture's policy and reference meet with equality.
    fn binding(mut self) -> Self {
        self.capacities = counts(&self.actions);
        self.reference_capacities = counts(&self.reference);
        self.budget = Some(cost(&self.actions, &self.costs));
        self.reference_budget = Some(cost(&self.reference, &self.reference_costs));
        self
    }

    fn direct(&self, outcome: &[f64]) -> Result<MultiActionIpwPolicy, &'static str> {
        let available: Vec<Vec<bool>> =
            self.available.chunks_exact(K).map(<[bool]>::to_vec).collect();
        evaluate_multi_action_ipw_policy(
            outcome,
            &MultiActionIpwDesign {
                assigned: &self.assigned,
                actions: &self.actions,
                probabilities: &self.probabilities,
                probability_rows: N,
                action_count: K,
                reference: Some(&self.reference),
                costs: Some(&self.costs),
                reference_costs: Some(&self.reference_costs),
                available: Some(&available),
            },
            &MultiActionLimits {
                capacities: Some(&self.capacities),
                reference_capacities: Some(&self.reference_capacities),
                budget: self.budget,
                reference_budget: self.reference_budget,
            },
        )
    }

    fn query(&self) -> PolicyValueQuery {
        let mut query = binary().query();
        query.assignment = Vec::new().into();
        query.propensity = Vec::new().into();
        query.actions = Vec::new().into();
        query.reference = Vec::new().into();
        query.costs = Vec::new().into();
        query.reference_costs = Vec::new().into();
        query.multi_action = Some(MultiActionPolicyInputs {
            action_labels: ["control", "A", "B"].map(Arc::<str>::from).into(),
            assignment: self.assigned.clone().into(),
            actions: self.actions.clone().into(),
            reference: self.reference.clone().into(),
            propensities: self.probabilities.clone().into(),
            available: self.available.clone().into(),
            costs: self.costs.into(),
            reference_costs: self.reference_costs.into(),
            capacities: self.capacities.into(),
            reference_capacities: self.reference_capacities.into(),
            budget: self.budget,
            reference_budget: self.reference_budget,
            cate_groups: Arc::from([]),
        });
        query
    }
}

#[test]
fn multi_action_limits_that_bind_leave_the_net_value_bit_identical() {
    let (outcome, fixture) = multi_action();
    let fixture = fixture.binding();
    let direct = fixture.direct(&outcome).expect("the fixture keeps every limit");
    let compiled = value(
        run(&outcome, fixture.query(), Some(EstimatorId::RandomizedMultiActionIpwPolicyNet), None)
            .unwrap(),
    );
    same(compiled.policy_value, direct.policy_value, "policy value");
    same(compiled.reference_value, direct.reference_value, "reference value");
    same(compiled.incremental_value, direct.incremental_value, "incremental value");
    same(compiled.treatment_rate, direct.non_control_rate, "non-control rate");
    same(compiled.total_cost, direct.policy_cost, "total cost");
    // Capacities below the row count couple the recommendations across rows: the route keeps
    // the point values and withholds the independent-row intervals.
    assert!(compiled.policy_interval_95.is_none());
    assert!(compiled.incremental_interval_95.is_none());
}

#[test]
fn each_broken_multi_action_limit_is_refused_by_both_paths() {
    let (outcome, fixture) = multi_action();
    let binding = fixture.binding();
    let broken: [(&str, fn(&mut MultiAction)); 6] = [
        ("policy availability", |f| f.available[K + 1] = false),
        ("reference availability", |f| f.available[1] = false),
        ("capacity", |f| f.capacities[1] -= 1),
        ("reference capacity", |f| f.reference_capacities[1] -= 1),
        ("budget", |f| f.budget = f.budget.map(|limit| limit - 0.5)),
        ("reference budget", |f| f.reference_budget = f.reference_budget.map(|limit| limit - 0.5)),
    ];
    for (what, breaking) in broken {
        let mut case = binding.clone();
        breaking(&mut case);
        assert!(case.direct(&outcome).is_err(), "{what}: the direct kernel accepts");
        assert!(
            run(&outcome, case.query(), Some(EstimatorId::RandomizedMultiActionIpwPolicyNet), None)
                .is_err(),
            "{what}: the compiled route accepts"
        );
    }
}
