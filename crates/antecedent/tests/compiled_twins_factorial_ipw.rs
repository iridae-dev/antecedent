//! Twin equality of the independent-factor 2x2 and point-only policy-value kernels and the
//! compiled routes their declared estimators select.
//!
//! * `randomized.factorial_bernoulli_ht` on a four-cell multi-arm `RandomizedEffectQuery` calls
//!   `factorial_bernoulli::estimate_factorial_2x2_cells` with each row's cell and that cell's
//!   probability; with probability rows stated as the products `qa * qb` it returns the numbers
//!   of `estimate_factorial_2x2` to the bit: the three contrasts, their bounds, the cell means
//!   and supports, and the intervals `effect -/+ z * sqrt(bound)`.
//! * `randomized.ipw_policy_net` and `randomized.multi_action_ipw_policy_net` on a
//!   `PolicyValueQuery` take their point values from `evaluate_binary_ipw_policy` and
//!   `evaluate_multi_action_ipw_policy` (bit-identical), their standard errors from the row
//!   scores of the default route (bit-identical to `evaluate_policy_value_ipw_scores` and
//!   `evaluate_multi_action_policy_value_scores`), and their intervals from
//!   `pointwise_intervals_95` around the net points.
//!
//! None of these kernels draws randomness, so there is no seed to pin.
// SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    deprecated,
    clippy::unusual_byte_groupings,
    reason = "the direct facade kernels are the paths under comparison and become deprecated wrappers; the execution seeds spell the test family's number"
)]
#![allow(clippy::float_cmp, reason = "bit-identical equality is the property under test")]
#[path = "common/retained_graphless.rs"]
mod retained_graphless;

use std::sync::Arc;

use antecedent::analysis::RetainedOperation;
use antecedent::estimate::policy_value::{
    BinaryIpwDesign, BinaryPolicyLimits, MultiActionIpwDesign, MultiActionLimits,
    evaluate_binary_ipw_policy, evaluate_multi_action_ipw_policy,
};
use antecedent::prelude::ExecutionContext;
use antecedent::{
    EstimatorId, PolicyValueEstimate, RandomizedEffectEstimate, RandomizedEffectQuery, Study,
};
use antecedent_core::{
    CausalQuery, MultiActionPolicyInputs, PolicyValueQuery, RandomizationDesign, VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::policy_value::{
    BinaryPolicyDesign, PolicyValueScores, evaluate_multi_action_policy_value_scores,
    evaluate_policy_value_ipw_scores, pointwise_intervals_95,
};

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

/// A two-factor design: per-row factor probabilities, the realized factors, and outcomes.
struct Factorial {
    y: Vec<f64>,
    a: Vec<bool>,
    b: Vec<bool>,
    pa: Vec<f64>,
    pb: Vec<f64>,
}

fn factorial(n: usize) -> Factorial {
    let a = (0..n).map(|i| (i % 4) / 2 == 1).collect::<Vec<_>>();
    let b = (0..n).map(|i| i % 2 == 1).collect::<Vec<_>>();
    let pa = (0..n).map(|i| [0.35, 0.5, 0.65][i % 3]).collect::<Vec<_>>();
    let pb = (0..n).map(|i| [0.3, 0.45, 0.6][(i / 3) % 3]).collect::<Vec<_>>();
    let y = (0..n)
        .map(|i| {
            1.0 + 0.8 * f64::from(u8::from(a[i]))
                + 0.5 * f64::from(u8::from(b[i]))
                + 0.3 * f64::from(u8::from(a[i] && b[i]))
                + 0.4 * (0.21 * i as f64).sin()
        })
        .collect::<Vec<_>>();
    Factorial { y, a, b, pa, pb }
}

/// The four-cell multi-arm query of `design`: cells in the order `(a=0,b=0)`, `(a=0,b=1)`,
/// `(a=1,b=0)`, `(a=1,b=1)`, every probability the product `qa * qb` the kernel forms.
fn four_cell_query(design: &Factorial) -> RandomizedEffectQuery {
    let n = design.y.len();
    let cells =
        (0..n).map(|i| 2 * usize::from(design.a[i]) + usize::from(design.b[i])).collect::<Vec<_>>();
    let rows = (0..n)
        .map(|i| {
            [(false, false), (false, true), (true, false), (true, true)]
                .map(|(level_a, level_b)| {
                    let qa = if level_a { design.pa[i] } else { 1.0 - design.pa[i] };
                    let qb = if level_b { design.pb[i] } else { 1.0 - design.pb[i] };
                    qa * qb
                })
                .to_vec()
        })
        .collect::<Vec<_>>();
    let assigned_probability = (0..n).map(|i| rows[i][cells[i]]).collect::<Vec<_>>();
    RandomizedEffectQuery::with_design(
        RandomizationDesign::MultiArm {
            assignment: Arc::from(cells.clone()),
            probabilities: rows.into(),
            arms: ["a0b0", "a0b1", "a1b0", "a1b1"].map(Arc::<str>::from).into(),
        },
        VariableId::from_raw(0),
        cells.iter().map(|cell| *cell != 0).collect::<Vec<_>>(),
        assigned_probability,
        labels("unit", n),
        labels("row", n),
        ("a0b0", "a0b1"),
    )
}

fn run_factorial(
    y: &[f64],
    query: RandomizedEffectQuery,
) -> Result<RandomizedEffectEstimate, antecedent::CausalError> {
    let data = TabularData::from_f64_columns([("outcome", y)]).unwrap();
    let ctx = ExecutionContext::for_tests(0x7717_0B);
    let result = Study::tabular(data)
        .query(query)
        .estimator(EstimatorId::RandomizedFactorialBernoulliHt)
        .build()?
        .run(&ctx)?;
    Ok(result.into_randomized_effect().expect("a randomized result"))
}

fn assert_factorial_twin(design: &Factorial, fit: &RandomizedEffectEstimate) {
    let direct = antecedent::estimate::factorial_bernoulli::estimate_factorial_2x2(
        &design.y, &design.a, &design.b, &design.pa, &design.pb,
    )
    .unwrap();
    same(fit.effect, direct.factor_a_effect, "factor A effect");
    same(fit.second_factor_effect.unwrap(), direct.factor_b_effect, "factor B effect");
    same(fit.factorial_interaction.unwrap(), direct.interaction_effect, "interaction");
    same(fit.variance_upper_bound, direct.factor_a_variance_bound, "factor A bound");
    same(fit.second_factor_variance.unwrap(), direct.factor_b_variance_bound, "factor B bound");
    same(
        fit.factorial_interaction_variance.unwrap(),
        direct.interaction_variance_bound,
        "interaction bound",
    );
    let n = design.y.len();
    let cells =
        (0..n).map(|i| 2 * usize::from(design.a[i]) + usize::from(design.b[i])).collect::<Vec<_>>();
    let q = (0..n)
        .map(|i| {
            let qa = if design.a[i] { design.pa[i] } else { 1.0 - design.pa[i] };
            let qb = if design.b[i] { design.pb[i] } else { 1.0 - design.pb[i] };
            qa * qb
        })
        .collect::<Vec<_>>();
    let cell_fit = antecedent_estimate::factorial_bernoulli::estimate_factorial_2x2_cells(
        &design.y, &cells, &q,
    )
    .unwrap();
    assert_eq!(cell_fit.contrasts, direct, "the cell entry is the factor entry");
    assert_eq!(fit.multi_arm_values.len(), 4);
    for (cell, (_, mean, variance, rows)) in fit.multi_arm_values.iter().enumerate() {
        same(*mean, direct.cell_means[cell], "cell mean");
        same(*variance, cell_fit.cell_variances[cell], "cell variance");
        assert_eq!(*rows, direct.cell_support[cell], "cell support");
    }
    assert_eq!(fit.control_units, direct.cell_support[0] + direct.cell_support[1]);
    assert_eq!(fit.treatment_units, direct.cell_support[2] + direct.cell_support[3]);
    assert_eq!(&*fit.assignment_design, "factorial_2x2_bernoulli");
    assert_eq!(&*fit.estimand, "factorial_primary_main_effect");
    let publish = direct.cell_support.iter().all(|rows| *rows >= 30);
    let interval = |effect: f64, bound: f64| {
        publish.then(|| {
            let span = antecedent_stats::NORMAL_Q975 * bound.sqrt();
            [effect - span, effect + span]
        })
    };
    same_interval(
        fit.interval_95,
        interval(direct.factor_a_effect, direct.factor_a_variance_bound),
        "factor A interval",
    );
    same_interval(
        fit.second_factor_interval_95,
        interval(direct.factor_b_effect, direct.factor_b_variance_bound),
        "factor B interval",
    );
    same_interval(
        fit.factorial_interaction_interval_95,
        interval(direct.interaction_effect, direct.interaction_variance_bound),
        "interaction interval",
    );
    match fit.standard_error {
        Some(se) => same(se, direct.factor_a_variance_bound.sqrt(), "standard error"),
        None => assert!(!publish, "a published interval reports its standard error"),
    }
}

#[test]
fn declared_factorial_reproduces_the_direct_kernel_bit_for_bit() {
    let design = factorial(160);
    let fit = run_factorial(&design.y, four_cell_query(&design)).unwrap();
    assert!(fit.interval_95.is_some(), "forty rows in every cell publish the three intervals");
    assert_factorial_twin(&design, &fit);
}

#[test]
fn declared_factorial_with_thin_cells_is_point_only_and_bit_identical() {
    let design = factorial(12);
    let fit = run_factorial(&design.y, four_cell_query(&design)).unwrap();
    assert!(fit.interval_95.is_none() && fit.standard_error.is_none());
    assert_eq!(&*fit.uncertainty, "factorial_bernoulli_ht_young_variance_bound_no_interval");
    assert_factorial_twin(&design, &fit);
}

#[test]
fn the_default_multi_arm_route_is_unchanged_by_the_declared_reading() {
    let design = factorial(160);
    let data = TabularData::from_f64_columns([("outcome", design.y.as_slice())]).unwrap();
    let ctx = ExecutionContext::for_tests(0x7717_0B);
    let default = Study::tabular(data)
        .query(four_cell_query(&design))
        .build()
        .unwrap()
        .run(&ctx)
        .unwrap()
        .into_randomized_effect()
        .unwrap();
    assert_eq!(&*default.assignment_design, "multi_arm");
    assert_eq!(&*default.estimand, "multi_arm_itt");
}

#[test]
fn declared_factorial_refuses_a_design_that_is_not_four_cells() {
    let n = 60;
    let three_arm = RandomizedEffectQuery::with_design(
        RandomizationDesign::MultiArm {
            assignment: (0..n).map(|i| i % 3).collect::<Vec<_>>().into(),
            probabilities: vec![vec![0.4, 0.3, 0.3]; n].into(),
            arms: ["c", "t1", "t2"].map(Arc::<str>::from).into(),
        },
        VariableId::from_raw(0),
        (0..n).map(|i| i % 3 != 0).collect::<Vec<_>>(),
        (0..n).map(|i| [0.4, 0.3, 0.3][i % 3]).collect::<Vec<_>>(),
        labels("unit", n),
        labels("row", n),
        ("c", "t1"),
    );
    let y = (0..n).map(|i| (i % 7) as f64).collect::<Vec<_>>();
    assert!(run_factorial(&y, three_arm).is_err());
    let bernoulli = RandomizedEffectQuery::bernoulli_itt(
        VariableId::from_raw(0),
        (0..n).map(|i| i % 2 == 0).collect::<Vec<_>>(),
        vec![0.5; n],
        labels("unit", n),
        labels("row", n),
        ("control", "treated"),
    );
    assert!(run_factorial(&y, bernoulli).is_err());
}

/// The independent-factor intervals carry their own estimator in the calibration key, so no
/// coverage record measured for another randomized estimator or design annotates them: until
/// one is measured for this estimator they report coverage not measured.
#[test]
fn declared_factorial_intervals_read_no_coverage_measured_for_another_estimator() {
    let design = factorial(160);
    let data = TabularData::from_f64_columns([("outcome", design.y.as_slice())]).unwrap();
    let ctx = ExecutionContext::for_tests(0x7717_0D);
    let prepared = Study::tabular(data.clone())
        .query(four_cell_query(&design))
        .estimator(EstimatorId::RandomizedFactorialBernoulliHt)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert!(result.randomized_effect().unwrap().interval_95.is_some());
    let contract = prepared.contract_for_result(&result).unwrap();
    let bases = result.calibration_bases(&contract).unwrap();
    assert!(!bases.is_empty());
    for basis in &bases {
        assert_eq!(basis.key.estimator, "randomized.factorial_bernoulli_ht");
        let slot = antecedent_io::calibration::calibration_slot(basis);
        assert_eq!(slot.status.as_str(), "not_measured", "{slot:#?}");
        assert!(slot.record_id.is_none(), "{slot:#?}");
    }
}

/// Whether `artifact` is refused by the encoder's own recompute.
fn refused(artifact: &antecedent_io::AnalysisResultWire, names: &[String], label: &str) -> bool {
    antecedent_io::encode_analysis_result_artifact(artifact, names.to_vec(), label).is_err()
}

#[test]
fn declared_factorial_artifact_round_trips_and_refuses_forgery() {
    let design = factorial(160);
    let data = TabularData::from_f64_columns([("outcome", design.y.as_slice())]).unwrap();
    let ctx = ExecutionContext::for_tests(0x7717_0C);
    let prepared = Study::tabular(data.clone())
        .query(four_cell_query(&design))
        .estimator(EstimatorId::RandomizedFactorialBernoulliHt)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    assert!(matches!(prepared.retained_operation(), RetainedOperation::Randomized));
    let program = prepared.plan_view().unwrap().identity();
    let result = prepared.estimate(&data, &ctx).unwrap();
    assert_eq!(result.lowered_program_identity(), Some(program));
    retained_graphless::assert_retained_graphless_program(
        "factorial-bernoulli",
        &prepared,
        &result,
        &ctx,
    );
    let fit = result.randomized_effect().unwrap().clone();
    assert_factorial_twin(&design, &fit);
    let bytes = prepared.encode_contracted_result(&result, "factorial-bernoulli", &ctx).unwrap();
    let (_, header, artifact) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let names = header.variable_names.clone();
    let stored = artifact.randomized_effect.as_ref().unwrap();
    assert_eq!(stored.assignment_design, "factorial_2x2_bernoulli");
    assert_eq!(stored.effect.to_bits(), fit.effect.to_bits());
    assert_eq!(stored.interval_95, fit.interval_95);
    assert_eq!(stored.factorial_interaction_interval_95, fit.factorial_interaction_interval_95);
    assert!(!refused(&artifact, &names, "factorial-bernoulli-reencode"));

    let mut forged_effect = artifact.clone();
    forged_effect.randomized_effect.as_mut().unwrap().effect += 0.5;
    forged_effect.estimate = forged_effect.randomized_effect.as_ref().map(|fit| fit.effect);
    assert!(refused(&forged_effect, &names, "forged-effect"));

    let mut dropped_interval = artifact.clone();
    dropped_interval.randomized_effect.as_mut().unwrap().factorial_interaction_interval_95 = None;
    assert!(refused(&dropped_interval, &names, "dropped-interval"));

    let mut forged_bound = artifact.clone();
    forged_bound.randomized_effect.as_mut().unwrap().multi_arm_values[3].2 *= 0.5;
    assert!(refused(&forged_bound, &names, "forged-cell-variance"));

    let mut forged_support = artifact.clone();
    let status = forged_support.randomized_effect.as_mut().unwrap();
    status.graphless_support_status = if status.graphless_support_status.is_some() {
        Some("not_supported".into())
    } else {
        Some("supported".into())
    };
    assert!(refused(&forged_support, &names, "forged-support"));
}

fn base_policy_query(n: usize) -> PolicyValueQuery {
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
        evaluation_subject_ids: labels("eval", n).into(),
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

fn run_policy(
    outcome: &[f64],
    query: PolicyValueQuery,
    estimator: EstimatorId,
) -> Result<PolicyValueEstimate, antecedent::CausalError> {
    let data = TabularData::from_f64_columns([("outcome", outcome)]).unwrap();
    let context = ExecutionContext::for_tests(31);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::PolicyValue(query))
        .estimator(estimator)
        .build()?
        .prepare(&context)?;
    let result = prepared.estimate(&data, &context)?;
    Ok(result.into_policy_value().expect("a policy-value result"))
}

/// The intervals the compiled route forms: the default rule around the net points.
fn net_intervals(
    scores: PolicyValueScores,
    policy_value: f64,
    incremental_value: f64,
    n: usize,
    matches: (usize, usize),
) -> Option<antecedent_estimate::policy_value::PolicyValueIntervals95> {
    let mut centred = scores;
    centred.policy_value = policy_value;
    centred.incremental_value = incremental_value;
    pointwise_intervals_95(&centred, n, matches.0, matches.1)
}

#[test]
fn declared_binary_net_value_reproduces_the_point_only_evaluator_bit_for_bit() {
    let n = 300;
    let assignment: Vec<bool> = (0..n).map(|i| (i * 7) % 5 < 2).collect();
    let actions: Vec<bool> = (0..n).map(|i| i % 3 != 0).collect();
    let reference: Vec<bool> = (0..n).map(|i| i % 4 == 0).collect();
    let propensity: Vec<f64> = (0..n).map(|i| 0.3 + 0.01 * (i % 9) as f64).collect();
    let outcome: Vec<f64> = (0..n)
        .map(|i| 1.0 + 0.37 * (i % 11) as f64 + 0.13 * f64::from(u8::from(assignment[i])))
        .collect();
    let costs = [0.1];
    let reference_costs = [0.07];
    let direct = evaluate_binary_ipw_policy(
        &outcome,
        &BinaryIpwDesign {
            assignment: &assignment,
            actions: &actions,
            propensity: &propensity,
            reference: Some(&reference),
            costs: &costs,
            reference_costs: Some(&reference_costs),
            available: None,
        },
        &BinaryPolicyLimits::default(),
    )
    .unwrap();
    let scores = evaluate_policy_value_ipw_scores(
        &outcome,
        &BinaryPolicyDesign {
            assignment: &assignment,
            actions: &actions,
            propensity: &propensity,
            reference: &reference,
            costs: &costs,
            reference_costs: &reference_costs,
        },
    )
    .unwrap();
    let mut query = base_policy_query(n);
    query.assignment = assignment.clone().into();
    query.actions = actions.clone().into();
    query.reference = reference.clone().into();
    query.propensity = propensity.into();
    query.costs = Arc::from(costs);
    query.reference_costs = Arc::from(reference_costs);
    let compiled = run_policy(&outcome, query, EstimatorId::RandomizedIpwPolicyNet).unwrap();
    same(compiled.policy_value, direct.policy_net, "policy value");
    same(compiled.reference_value, direct.reference_net, "reference value");
    same(compiled.incremental_value, direct.incremental_value, "incremental value");
    same(compiled.relative_value_gap, direct.regret, "relative gap");
    same(compiled.treatment_rate, direct.treatment_rate, "treatment rate");
    same(compiled.total_cost, direct.total_cost, "total cost");
    same(compiled.policy_standard_error, scores.policy_standard_error, "policy SE");
    same(compiled.reference_standard_error, scores.reference_standard_error, "reference SE");
    same(compiled.incremental_standard_error, scores.incremental_standard_error, "incremental SE");
    let matches = |policy: &[bool]| assignment.iter().zip(policy).filter(|(a, b)| a == b).count();
    let expected = net_intervals(
        scores,
        direct.policy_net,
        direct.incremental_value,
        n,
        (matches(&actions), matches(&reference)),
    );
    assert!(expected.is_some(), "the fixture publishes intervals");
    same_interval(compiled.policy_interval_95, expected.map(|i| i.policy), "policy interval");
    same_interval(
        compiled.incremental_interval_95,
        expected.map(|i| i.incremental),
        "incremental interval",
    );
}

#[test]
fn declared_multi_action_net_value_reproduces_the_point_only_evaluator_bit_for_bit() {
    let n = 300;
    let k = 3;
    let outcome: Vec<f64> =
        (0..n).map(|i| 1.0 + 0.31 * (i % 7) as f64 + 0.11 * (i % 3) as f64).collect();
    let assigned: Vec<usize> = (0..n).map(|i| [0, 1, 2, 0, 2][i % 5]).collect();
    let actions: Vec<usize> = (0..n).map(|i| [0, 1, 1, 2][i % 4]).collect();
    let reference: Vec<usize> = (0..n).map(|i| usize::from(i % 6 == 0)).collect();
    let probabilities: Vec<f64> =
        (0..n).flat_map(|i| if i % 2 == 0 { [0.4, 0.35, 0.25] } else { [0.5, 0.3, 0.2] }).collect();
    let costs = [0.0, 0.15, 0.4];
    let reference_costs = [0.0, 0.1, 0.3];
    let direct = evaluate_multi_action_ipw_policy(
        &outcome,
        &MultiActionIpwDesign {
            assigned: &assigned,
            actions: &actions,
            probabilities: &probabilities,
            probability_rows: n,
            action_count: k,
            reference: Some(&reference),
            costs: Some(&costs),
            reference_costs: Some(&reference_costs),
            available: None,
        },
        &MultiActionLimits::default(),
    )
    .unwrap();
    let inputs = MultiActionPolicyInputs {
        action_labels: ["control", "A", "B"].map(Arc::<str>::from).into(),
        assignment: assigned.clone().into(),
        actions: actions.clone().into(),
        reference: reference.clone().into(),
        propensities: probabilities.into(),
        available: vec![true; n * k].into(),
        costs: costs.into(),
        reference_costs: reference_costs.into(),
        capacities: [n; 3].into(),
        reference_capacities: [n; 3].into(),
        budget: None,
        reference_budget: None,
        cate_groups: Arc::from([]),
    };
    let scores = evaluate_multi_action_policy_value_scores(&outcome, &inputs).unwrap();
    let mut query = base_policy_query(n);
    query.assignment = Vec::new().into();
    query.propensity = Vec::new().into();
    query.actions = Vec::new().into();
    query.reference = Vec::new().into();
    query.costs = Vec::new().into();
    query.reference_costs = Vec::new().into();
    query.multi_action = Some(inputs);
    let compiled =
        run_policy(&outcome, query, EstimatorId::RandomizedMultiActionIpwPolicyNet).unwrap();
    same(compiled.policy_value, direct.policy_value, "policy value");
    same(compiled.reference_value, direct.reference_value, "reference value");
    same(compiled.incremental_value, direct.incremental_value, "incremental value");
    same(compiled.relative_value_gap, direct.reference_value - direct.policy_value, "relative gap");
    same(compiled.treatment_rate, direct.non_control_rate, "non-control rate");
    same(compiled.total_cost, direct.policy_cost, "total cost");
    same(compiled.policy_standard_error, scores.policy_standard_error, "policy SE");
    same(compiled.incremental_standard_error, scores.incremental_standard_error, "incremental SE");
    let matches = |policy: &[usize]| assigned.iter().zip(policy).filter(|(a, b)| a == b).count();
    let expected = net_intervals(
        scores,
        direct.policy_value,
        direct.incremental_value,
        n,
        (matches(&actions), matches(&reference)),
    );
    assert!(expected.is_some(), "the fixture publishes intervals");
    same_interval(compiled.policy_interval_95, expected.map(|i| i.policy), "policy interval");
    same_interval(
        compiled.incremental_interval_95,
        expected.map(|i| i.incremental),
        "incremental interval",
    );
}

#[test]
fn declared_binary_net_value_artifact_round_trips_under_the_ipw_support_row() {
    let n = 160;
    let assignment: Vec<bool> = (0..n).map(|i| i % 2 == 0).collect();
    let outcome: Vec<f64> = (0..n)
        .map(|i| 1.0 + 0.23 * (i % 9) as f64 + 0.4 * f64::from(u8::from(assignment[i])))
        .collect();
    let mut query = base_policy_query(n);
    query.assignment = assignment.into();
    query.actions = (0..n).map(|i| i % 3 == 0).collect::<Vec<_>>().into();
    query.reference = (0..n).map(|i| i % 5 == 0).collect::<Vec<_>>().into();
    query.costs = Arc::from([0.1]);
    query.reference_costs = Arc::from([0.1]);
    let data = TabularData::from_f64_columns([("outcome", outcome.as_slice())]).unwrap();
    let ctx = ExecutionContext::for_tests(0x7717_0D);
    let prepared = Study::tabular(data.clone())
        .query(CausalQuery::PolicyValue(query))
        .estimator(EstimatorId::RandomizedIpwPolicyNet)
        .build()
        .unwrap()
        .prepare(&ctx)
        .unwrap();
    let result = prepared.estimate(&data, &ctx).unwrap();
    let value = result.policy_value().unwrap().clone();
    assert!(value.policy_interval_95.is_some());
    let bytes = prepared.encode_contracted_result(&result, "ipw-policy-net", &ctx).unwrap();
    let (_, header, artifact) = antecedent_io::decode_analysis_result_artifact(&bytes).unwrap();
    let stored = artifact.policy_value.as_ref().unwrap();
    same(stored.policy_value, value.policy_value, "stored policy value");
    same(stored.incremental_value, value.incremental_value, "stored incremental value");
    assert!(!refused(&artifact, &header.variable_names, "ipw-policy-net-reencode"));
    let mut forged = artifact.clone();
    forged.policy_value.as_mut().unwrap().policy_interval_95 =
        stored.policy_interval_95.map(|[lower, upper]| [lower - 0.5, upper + 0.5]);
    assert!(refused(&forged, &header.variable_names, "forged-policy-interval"));
}

#[test]
fn a_net_value_estimator_refuses_a_query_it_does_not_answer() {
    let n = 40;
    let outcome: Vec<f64> = (0..n).map(|i| (i % 5) as f64).collect();
    let mut doubly_robust = base_policy_query(n);
    doubly_robust.assignment = (0..n).map(|i| i % 2 == 0).collect::<Vec<_>>().into();
    doubly_robust.mu0 = vec![1.0; n].into();
    doubly_robust.mu1 = vec![2.0; n].into();
    doubly_robust.disjoint_training_subjects = true;
    assert!(run_policy(&outcome, doubly_robust, EstimatorId::RandomizedIpwPolicyNet).is_err());
    let mut binary = base_policy_query(n);
    binary.assignment = (0..n).map(|i| i % 2 == 0).collect::<Vec<_>>().into();
    assert!(
        run_policy(&outcome, binary.clone(), EstimatorId::RandomizedMultiActionIpwPolicyNet)
            .is_err()
    );
    assert!(run_policy(&outcome, binary, EstimatorId::RandomizedIpwPolicyNet).is_ok());
}
