//! Whole-procedure coverage of the Wald interval of an off-policy sequential evaluation
//! against the known value of the target policy.
//!
//! The procedure scored is `evaluate_off_policy` end to end: positivity check on every
//! history the target reaches, the importance-weighted mean `m = mean(w_i G_i)` of the
//! logged returns and the interval `m +- z * sqrt(s^2 / n)` (`OFF_POLICY_INTERVAL_LEVEL`,
//! 0.95). Its construction is pinned independently by
//! `sequential_evaluation_interval_construction.rs`; this file measures whether the
//! interval it publishes covers.
//!
//! Design: one decision between two commit actions, `left` and `right`. The logging
//! policy picks `left` with probability 1/2 and the target with probability 4/5. A
//! `left` return is `N(2, 1)` and a `right` return is `N(6, 1)`, so the target policy's
//! value is the known `0.8 * 2 + 0.2 * 6 = 2.8`. Each replicate logs `n` trajectories,
//! `n` on the three-point grid around 200 (100, 200, 400); the logged assignment
//! probability is the logging policy's 1/2. The weights are 8/5 and 2/5 and the weighted
//! returns have a finite variance, so the Wald interval is asymptotically exact and its
//! Monte Carlo coverage at 100 trajectories or more is held to the harness band
//! (`CoverageTally::assert`: nominal `0.95` within the precision floor of the first
//! `n_sim()` replicates, rechecked at 2000 when near the band edge). The record is keyed
//! by the construction declared in [`construction`]; it names the importance-weighted
//! sample-variance Wald interval, an iid logged sample and the target value as the truth.
//!
//! The non-ignored test anchors the truth: with one `left` and one `right` trajectory at
//! the returns 2 and 6 the importance-weighted estimate is `(8/5 * 2 + 2/5 * 6) / 2 = 2.8`
//! exactly (tolerance `1e-12`, two orders above the rounding of four products and sums of
//! order-one doubles), the value the coverage tests score against.
//!
//! Ignored tests run via `scripts/measure_calibration.sh` (release build).
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    reason = "bounded simulation sizes and small indices"
)]

#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;

use antecedent_contract::evidence_ledger::EvidenceLedger;
use antecedent_core::ExecutionContext;
use antecedent_design::sequential_evaluation::{
    ActionRole, ActionSpec, ActionSupport, CensoringRule, ContingentPolicy, EvaluationLimits,
    FiniteSequentialModel, HistoryKey, LoggedStep, LoggedTrajectory, evaluate_off_policy,
};
use calibration::{
    Construction, CoverageTally, RecordKey, ScopeFacts, gaussian, grid_n, grid_seed,
    map_replicates, n_sim, stream_seed, unit_uniform,
};

const LEVEL: f64 = 0.95;
const LEFT: usize = 0;
const RIGHT: usize = 1;
const LOGGING_LEFT: f64 = 0.5;
const TARGET_LEFT: f64 = 0.8;
const LEFT_MEAN: f64 = 2.0;
const RIGHT_MEAN: f64 = 6.0;
const BASE_TRAJECTORIES: usize = 200;
const SEED_BASE: u64 = 0x0FF9_0001;

/// The target policy's known value: `0.8 * 2 + 0.2 * 6`.
const TRUTH: f64 = TARGET_LEFT * LEFT_MEAN + (1.0 - TARGET_LEFT) * RIGHT_MEAN;

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

fn trajectory(id: String, action: usize, outcome: f64) -> LoggedTrajectory {
    LoggedTrajectory {
        id,
        steps: vec![LoggedStep { action, observation: None, assignment_probability: LOGGING_LEFT }],
        outcome_utility: outcome,
    }
}

/// The interval limits of the off-policy evaluation of `log`, or `None` when it is refused or
/// publishes no interval.
fn evaluate(log: &[LoggedTrajectory], seed: u64) -> Option<(f64, Option<(f64, f64)>)> {
    let estimate = evaluate_off_policy(
        &model(),
        &policy("target_v1", TARGET_LEFT, 1.0 - TARGET_LEFT),
        &policy("logging_v1", LOGGING_LEFT, 1.0 - LOGGING_LEFT),
        log,
        &mut EvidenceLedger::new(),
        EvaluationLimits::default(),
        &ExecutionContext::for_tests(seed),
    )
    .ok()?;
    Some((estimate.ordinary, estimate.interval.map(|interval| (interval.lower, interval.upper))))
}

/// One simulated log of `n` trajectories drawn from the logging policy.
fn simulated_log(n: usize, seed: u64) -> Vec<LoggedTrajectory> {
    let mut noise = gaussian(seed);
    (0..n)
        .map(|i| {
            let left = unit_uniform(stream_seed(seed, i as u64)) < LOGGING_LEFT;
            let (action, mean) = if left { (LEFT, LEFT_MEAN) } else { (RIGHT, RIGHT_MEAN) };
            trajectory(format!("traj-{i}"), action, mean + noise())
        })
        .collect()
}

fn construction() -> Construction {
    Construction {
        query: "SequentialPolicyValue".into(),
        graph_class: "NotApplicable".into(),
        structure: "fixed".into(),
        modality: "logged_trajectories".into(),
        inference: "Frequentist".into(),
        estimator: "sequential_evaluation.off_policy_importance_weighting".into(),
        interval_method: "wald_normal".into(),
        se_kind: "importance_weighted_sample_variance".into(),
        dependence: "iid".into(),
        posterior: String::new(),
        functional: "target_policy_value".into(),
        identification: "point".into(),
        reported_level: LEVEL,
    }
}

#[test]
fn the_known_value_is_the_importance_weighted_estimate_of_the_balanced_pair() {
    let log = [trajectory("a".into(), LEFT, LEFT_MEAN), trajectory("b".into(), RIGHT, RIGHT_MEAN)];
    let (estimate, _) = evaluate(&log, 1).expect("the evaluation runs");
    assert!((estimate - TRUTH).abs() < 1e-12, "{estimate} vs {TRUTH}");
    assert!((TRUTH - 2.8).abs() < 1e-12);
}

#[test]
#[ignore = "calibration: run via scripts/measure_calibration.sh"]
fn off_policy_importance_weighted_wald_interval() {
    let n = grid_n(BASE_TRAJECTORIES);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "off_policy_importance_weighted_wald_interval",
            dgp: "two_action_gaussian_returns_known_target_value",
            interval: "wald_normal",
        },
        LEVEL,
    );
    let runs = map_replicates(n_sim(), |rep| {
        let seed = grid_seed(SEED_BASE + rep);
        evaluate(&simulated_log(n, seed), seed)
    });
    for run in runs {
        let Some((_, Some(band))) = run else {
            tally.skip();
            continue;
        };
        tally.bind_validation(
            &construction(),
            ScopeFacts {
                row_count: n as u64,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
            "not_applicable",
        );
        tally.record(Some(band), TRUTH);
    }
    tally.assert();
}
