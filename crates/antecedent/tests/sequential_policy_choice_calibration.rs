//! Prospective known-optimum sequential policy choice studies. Each ignored
//! grid reports the scored and refused replicates and the correct-choice count
//! under one exact design scope. No decision-reliability record is earned.
mod common;

use antecedent::analysis::sequential_policy_reliability::{
    SequentialReliabilityMode, compile_sequential_policy_reliability,
};
use antecedent_core::{ExecutionContext, SemanticDigest};
use antecedent_design::sequential_evaluation::{
    ActionRole, ActionSpec, ActionSupport, CensoringRule, ContingentPolicy, FiniteSequentialModel,
    HistoryKey, LoggedStep, LoggedTrajectory, PolicySpec, RolloutConfig,
};
use common::calibration::{grid_n, grid_seed, map_replicates, n_sim, uniform};

fn design() -> (FiniteSequentialModel, Vec<PolicySpec>, String, ContingentPolicy) {
    let action = |label: &str, utility: Vec<f64>| ActionSpec {
        label: label.into(),
        role: ActionRole::Commit,
        utility_cost: 0.0,
        resource_use: 0.0,
        transition: vec![1.0, 0.0, 0.0, 1.0],
        channel: None,
        immediate_utility: utility,
        delayed: None,
    };
    let model = FiniteSequentialModel {
        label: "known-choice".into(),
        population: "p".into(),
        regime: "r".into(),
        period: "t".into(),
        state_variable: "state".into(),
        states: vec!["a".into(), "b".into()],
        initial: vec![0.5, 0.5],
        horizon: 1,
        actions: vec![action("a", vec![2.0, 0.0]), action("b", vec![0.0, 1.0])],
        terminal_utility: vec![0.0, 0.0],
        censoring: CensoringRule::Refuse,
        resource_budget: None,
        stopping: vec![],
        support: ActionSupport::everywhere(2),
    };
    let best = ContingentPolicy::new("a").decide(HistoryKey::root(), 0);
    let best_id = best.identity_hex();
    let policies = vec![
        PolicySpec::Contingent(best),
        PolicySpec::Contingent(ContingentPolicy::new("b").decide(HistoryKey::root(), 1)),
    ];
    let behaviour = ContingentPolicy::new("logging")
        .decide_stochastic(HistoryKey::root(), vec![(0, 0.5), (1, 0.5)])
        .unwrap();
    (model, policies, best_id, behaviour)
}

fn replicate(seed: u64, n: usize, logged: bool) -> Option<(SemanticDigest, bool)> {
    let (model, policies, best_id, behaviour) = design();
    let mode = if logged {
        let mut rng = uniform(seed, 0x5E91_0001);
        let log: Vec<_> = (0..n)
            .map(|i| {
                let action = usize::from(rng() >= 0.5);
                let first_state = rng() < 0.5;
                let reward = if action == 0 {
                    if first_state { 2.0 } else { 0.0 }
                } else if first_state {
                    0.0
                } else {
                    1.0
                };
                LoggedTrajectory {
                    id: format!("row-{i}"),
                    steps: vec![LoggedStep {
                        action,
                        observation: None,
                        assignment_probability: 0.5,
                    }],
                    outcome_utility: reward,
                }
            })
            .collect();
        SequentialReliabilityMode::OffPolicy {
            behaviour,
            log,
            seed: grid_seed(seed),
            replicates: 100,
            minimum_effective_sample_size: 2.0,
        }
    } else {
        SequentialReliabilityMode::Rollout(RolloutConfig {
            seed: grid_seed(seed),
            rollouts: n as u64,
        })
    };
    let plan =
        compile_sequential_policy_reliability(model, policies, "utility".into(), mode).ok()?;
    let scope = plan.decision_reliability_scope().ok()?;
    let result = plan.execute(&ExecutionContext::for_tests(grid_seed(seed))).ok()?;
    Some((scope, result.report.decided_id == best_id))
}

fn run_grid(logged: bool) {
    let n = grid_n(120);
    let mut scope = None;
    let (mut scored, mut refused, mut correct) = (0_u32, 0_u32, 0_u32);
    for result in map_replicates(n_sim(), |rep| replicate(0x5E91_8001 + rep, n, logged)) {
        if let Some((coordinate, chose_best)) = result {
            if let Some(previous) = scope {
                assert_eq!(previous, coordinate);
            } else {
                scope = Some(coordinate);
            }
            scored += 1;
            correct += u32::from(chose_best);
        } else {
            refused += 1;
        }
    }
    assert!(scored > 0, "no usable sequential evaluations");
    eprintln!(
        "prospective known-optimum choice: {correct}/{scored}, refused {refused}; no record earned"
    );
}

#[test]
#[ignore = "calibration: prospective known-law action-choice study; no reliability record earned"]
fn sequential_known_law_action_choice() {
    run_grid(false);
}
#[test]
#[ignore = "calibration: prospective logged-bootstrap action-choice study; no reliability record earned"]
fn sequential_logged_bootstrap_action_choice() {
    run_grid(true);
}
