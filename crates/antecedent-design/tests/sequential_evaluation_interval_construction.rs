//! Independent construction check of the interval of an off-policy sequential evaluation.
//!
//! The estimate is the importance-weighted mean `m = mean(w_i G_i)` of the logged returns
//! `G_i`, with weight `w_i = pi(a_i) / mu(a_i)` of the target policy `pi` over the logging
//! policy `mu` at the single decision. Its interval is the Wald interval
//! `m +- z * sqrt(s^2 / n)` where `s^2` is the sample variance (denominator `n - 1`) of the
//! weighted returns and `z = 1.959963984540054` is the 97.5% standard-normal quantile.
//!
//! The design is one decision between two commit actions, `left` and `right`. The logging
//! policy picks each with probability 1/2; the target picks `left` with probability 4/5. The
//! log holds four trajectories: `left` with returns 2 and 4, `right` with returns 6 and 10.
//! By hand: the weights are 4/5 / 1/2 = 8/5 for `left` and 1/5 / 1/2 = 2/5 for `right`, so the
//! weighted returns are 16/5, 32/5, 12/5 and 4. Their mean is `m = 4`; the deviations are
//! -4/5, 12/5, -8/5 and 0, whose squares sum to `(16 + 144 + 64) / 25 = 224/25`; hence
//! `s^2 = 224/75` and the standard error is `sqrt(224/300) = sqrt(56/75)`. The expected limits
//! below are computed here from those fractions and the literal `z`, never by calling the
//! production interval helper.
//!
//! Tolerance: the production values come from four products and sums of order-one doubles and
//! a normal quantile accurate to a few ulps (about `1e-15`), so the limits agree with the hand
//! values to about `1e-14`. An absolute tolerance of `1e-12` leaves two orders of margin and a
//! different construction (a population variance, a Student-t quantile, an unweighted mean)
//! moves a limit by at least `1e-2`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_contract::evidence_ledger::EvidenceLedger;
use antecedent_core::ExecutionContext;
use antecedent_design::sequential_evaluation::{
    ActionRole, ActionSpec, ActionSupport, CensoringRule, ContingentPolicy, EvaluationLimits,
    FiniteSequentialModel, HistoryKey, LoggedStep, LoggedTrajectory, OFF_POLICY_INTERVAL_LEVEL,
    OffPolicyEstimate, evaluate_off_policy,
};
use antecedent_io_core::contract_section::CalibrationStatus;

const LEFT: usize = 0;
const RIGHT: usize = 1;
const Z_975: f64 = 1.959_963_984_540_054;
const TOLERANCE: f64 = 1e-12;

fn model() -> FiniteSequentialModel {
    let commit = |label: &str| ActionSpec {
        label: label.into(),
        role: ActionRole::Commit,
        utility_cost: 0.0,
        resource_use: 0.0,
        transition: vec![1.0, 0.0, 0.0, 1.0],
        channel: None,
        immediate_utility: vec![0.0, 0.0],
        delayed: None,
    };
    FiniteSequentialModel {
        label: "two_commits".into(),
        population: "plant_a".into(),
        regime: "operations".into(),
        period: "2026".into(),
        state_variable: "unit_condition".into(),
        states: vec!["sound".into(), "faulty".into()],
        initial: vec![0.5, 0.5],
        horizon: 1,
        actions: vec![commit("left"), commit("right")],
        terminal_utility: vec![0.0, 0.0],
        censoring: CensoringRule::Refuse,
        resource_budget: None,
        stopping: Vec::new(),
        support: ActionSupport::everywhere(2),
    }
}

fn policy(version: &str, left: f64, right: f64) -> ContingentPolicy {
    ContingentPolicy::new(version)
        .decide_stochastic(HistoryKey::root(), vec![(LEFT, left), (RIGHT, right)])
        .unwrap()
}

fn trajectory(id: &str, action: usize, outcome: f64) -> LoggedTrajectory {
    LoggedTrajectory {
        id: id.into(),
        steps: vec![LoggedStep { action, observation: None, assignment_probability: 0.5 }],
        outcome_utility: outcome,
    }
}

fn evaluate(log: &[LoggedTrajectory]) -> OffPolicyEstimate {
    evaluate_off_policy(
        &model(),
        &policy("target_v1", 0.8, 0.2),
        &policy("logging_v1", 0.5, 0.5),
        log,
        &mut EvidenceLedger::new(),
        EvaluationLimits::default(),
        &ExecutionContext::for_tests(3),
    )
    .unwrap()
}

#[test]
fn the_interval_is_the_wald_interval_of_the_weighted_returns() {
    let log = [
        trajectory("l1", LEFT, 2.0),
        trajectory("l2", LEFT, 4.0),
        trajectory("r1", RIGHT, 6.0),
        trajectory("r2", RIGHT, 10.0),
    ];
    let estimate = evaluate(&log);

    // Hand construction from the fractions in the module documentation.
    let weighted = [16.0 / 5.0, 32.0 / 5.0, 12.0 / 5.0, 4.0];
    let mean = weighted.iter().sum::<f64>() / 4.0;
    assert!((mean - 4.0).abs() < TOLERANCE);
    let sum_squares: f64 = weighted.iter().map(|x| (x - mean) * (x - mean)).sum();
    assert!((sum_squares - 224.0 / 25.0).abs() < TOLERANCE);
    let standard_error = (sum_squares / 3.0 / 4.0).sqrt();
    assert!((standard_error - (56.0_f64 / 75.0).sqrt()).abs() < TOLERANCE);
    let lower = mean - Z_975 * standard_error;
    let upper = mean + Z_975 * standard_error;

    assert!((estimate.ordinary - mean).abs() < TOLERANCE);
    assert!((estimate.uncalibrated_standard_error.unwrap() - standard_error).abs() < TOLERANCE);
    let interval = estimate.interval.expect("an interval above one trajectory");
    assert!((interval.level - 0.95).abs() < TOLERANCE);
    assert!((OFF_POLICY_INTERVAL_LEVEL - 0.95).abs() < TOLERANCE);
    assert!((interval.lower - lower).abs() < TOLERANCE, "{} vs {lower}", interval.lower);
    assert!((interval.upper - upper).abs() < TOLERANCE, "{} vs {upper}", interval.upper);
    // No coverage record measures this construction: the interval says so.
    assert_eq!(interval.calibration.status, CalibrationStatus::NotMeasured);
}

#[test]
fn one_trajectory_has_no_standard_error_and_so_no_interval() {
    let estimate = evaluate(&[trajectory("l1", LEFT, 2.0)]);
    assert!(estimate.uncalibrated_standard_error.is_none());
    assert!(estimate.interval.is_none());
}
