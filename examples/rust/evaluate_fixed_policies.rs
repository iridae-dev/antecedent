// SPDX-License-Identifier: MIT OR Apache-2.0
//! Evaluate fixed rules using held-out action logs.
//!
//! Each synthetic log records the assignment probability. Resampling keeps each complete history together.

use antecedent::analysis::sequential_policy_reliability::{
    SequentialReliabilityMode, compile_sequential_policy_reliability,
    consume_sequential_policy_reliability,
};
use antecedent_core::ExecutionContext;
use antecedent_design::sequential_evaluation::{
    ActionRole, ActionSpec, ActionSupport, CensoringRule, ContingentPolicy, FiniteSequentialModel,
    HistoryKey, LoggedStep, LoggedTrajectory, PolicySpec,
};

fn fixed_rules() -> (FiniteSequentialModel, Vec<PolicySpec>) {
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
        label: "held-out decision log".into(),
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
    let policies = vec![
        PolicySpec::Contingent(ContingentPolicy::new("a").decide(HistoryKey::root(), 0)),
        PolicySpec::Contingent(ContingentPolicy::new("b").decide(HistoryKey::root(), 1)),
    ];
    (model, policies)
}

fn main() {
    let (model, policies) = fixed_rules();
    let behaviour = ContingentPolicy::new("logging")
        .decide_stochastic(HistoryKey::root(), vec![(0, 0.5), (1, 0.5)])
        .unwrap();
    let log = (0..16)
        .map(|i| LoggedTrajectory {
            id: format!("{i}"),
            steps: vec![LoggedStep {
                action: i % 2,
                observation: None,
                assignment_probability: 0.5,
            }],
            outcome_utility: (i % 4) as f64,
        })
        .collect();
    let plan = compile_sequential_policy_reliability(
        model,
        policies,
        "utility".into(),
        SequentialReliabilityMode::OffPolicy {
            behaviour,
            log,
            seed: 3,
            replicates: 64,
            minimum_effective_sample_size: 2.0,
        },
    )
    .unwrap();
    let ctx = ExecutionContext::for_tests(1);
    let result = plan.execute(&ctx).unwrap();
    let replay =
        consume_sequential_policy_reliability(&plan.export(&result, &ctx).unwrap(), None, &ctx)
            .unwrap();
    assert_eq!(result.draws, replay.draws);
    println!("Compared {} fixed policies on held-out histories", result.draws.candidates.len());
    println!("Reliability diagnostics: {:?}", result.report);
}
