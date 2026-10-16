//! Selective execution of the fixed-population temporal row.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_counterfactual::selective::{StageStatus, TemporalExecutor};
use antecedent_counterfactual::temporal_cross_world::{
    FactualUnitHistory, HistoryId, NamedActionHistory, NodeMechanism, SnapshotId,
    TemporalCounterfactualSpec, TemporalGraph, TemporalMechanismFit, TemporalNode, UnitId,
    evaluate_temporal_counterfactual,
};

use TemporalNode::{Action0, Action1, Covariate0, Covariate1, Outcome};

// Known closed-form two-slice SCM:
//   L0 = 1.0 + u0
//   L1 = 0.5 + 0.8 L0 + 1.5 A0 + u1
//   Y  = -1.0 + 0.5 L0 + 0.7 A0 + 2.0 L1 + 1.2 A1 + u2
fn simulate(noise: [f64; 3], a: [f64; 2]) -> [f64; 5] {
    let l0 = 1.0 + noise[0];
    let l1 = 0.5 + 0.8 * l0 + 1.5 * a[0] + noise[1];
    let y = -1.0 + 0.5 * l0 + 0.7 * a[0] + 2.0 * l1 + 1.2 * a[1] + noise[2];
    [l0, a[0], l1, a[1], y]
}

const NOISE: [[f64; 3]; 4] =
    [[0.3, -0.2, 0.5], [-0.4, 0.6, -0.1], [0.0, 0.1, 0.9], [1.1, -0.7, -0.3]];
const FACTUAL_ACTIONS: [[f64; 2]; 4] = [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]];
const PLUS: [f64; 2] = [1.0, 1.0];
const MINUS: [f64; 2] = [0.0, 0.0];

fn graph() -> TemporalGraph {
    TemporalGraph {
        horizon: 2,
        edges: vec![
            (Covariate0, Covariate1),
            (Action0, Covariate1),
            (Covariate0, Outcome),
            (Action0, Outcome),
            (Covariate1, Outcome),
            (Action1, Outcome),
        ],
        latent_confounding: false,
    }
}

fn fit(halfwidth: Option<f64>) -> TemporalMechanismFit {
    TemporalMechanismFit {
        fit_id: "fit-known-coefficients".to_owned(),
        mechanisms: vec![
            NodeMechanism {
                node: Covariate0,
                intercept: 1.0,
                parent_coefficients: vec![],
                noise_halfwidth: halfwidth,
            },
            NodeMechanism {
                node: Covariate1,
                intercept: 0.5,
                parent_coefficients: vec![(Covariate0, 0.8), (Action0, 1.5)],
                noise_halfwidth: halfwidth,
            },
            NodeMechanism {
                node: Outcome,
                intercept: -1.0,
                parent_coefficients: vec![
                    (Covariate0, 0.5),
                    (Action0, 0.7),
                    (Covariate1, 2.0),
                    (Action1, 1.2),
                ],
                noise_halfwidth: halfwidth,
            },
        ],
    }
}

fn unit(i: usize) -> UnitId {
    UnitId::new(format!("unit-{i}"))
}

fn spec() -> TemporalCounterfactualSpec {
    let factual = (0..4)
        .map(|i| FactualUnitHistory {
            unit: unit(i),
            history: HistoryId::new(format!("hist-{i}")),
            times: [0, 1],
            values: simulate(NOISE[i], FACTUAL_ACTIONS[i]),
        })
        .collect();
    let units: Vec<UnitId> = (0..4).map(unit).collect();
    TemporalCounterfactualSpec {
        graph: graph(),
        fit: fit(None),
        snapshot: SnapshotId::new("snapshot-1"),
        factual,
        plus: NamedActionHistory {
            name: "always_treat".to_owned(),
            times: [0, 1],
            actions: PLUS,
            units: units.clone(),
        },
        minus: NamedActionHistory {
            name: "never_treat".to_owned(),
            times: [0, 1],
            actions: MINUS,
            units,
        },
    }
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(7)
}

#[test]
fn staged_execution_is_bit_identical_to_the_one_shot_evaluator() {
    let reference = evaluate_temporal_counterfactual(&spec(), &ctx()).unwrap();
    let mut exec = TemporalExecutor::new();
    let staged = exec.execute(&spec(), &ctx()).unwrap();
    assert_eq!(staged.result, reference);
    assert_eq!(staged.work.abductions, 4);
    assert_eq!(staged.work.world_predictions, 8);
    assert_eq!(staged.work.mechanism_fits, 0);
    assert_eq!(staged.stages[1].status, StageStatus::Computed);
}

// Action-only change: new named histories reuse the abduction and the
// per-unit exogenous draws, and agree bit for bit with a full rerun.
#[test]
fn a_changed_action_history_reuses_the_abduction() {
    let mut exec = TemporalExecutor::new();
    exec.execute(&spec(), &ctx()).unwrap();
    let mut changed = spec();
    changed.plus.name = "treat_first".to_owned();
    changed.plus.actions = [1.0, 0.0];
    let reused = exec.execute(&changed, &ctx()).unwrap();
    assert_eq!(reused.work.abductions, 0);
    assert_eq!(reused.work.abductions_reused, 4);
    assert_eq!(reused.work.world_predictions, 8);
    assert_eq!(reused.stages[1].status, StageStatus::Reused);
    assert_eq!(reused.stages[2].status, StageStatus::Reused);
    let full = evaluate_temporal_counterfactual(&changed, &ctx()).unwrap();
    assert_eq!(reused.result, full);
    // Hand truth: plus = (1, 0), minus = (0, 0) differ by A0 only.
    for (i, row) in reused.result.units.iter().enumerate() {
        let truth = simulate(NOISE[i], [1.0, 0.0])[4];
        assert!((row.plus_outcome - truth).abs() < 1e-9);
    }
}

#[test]
fn changed_factual_evidence_or_fit_invalidates_the_abduction() {
    let mut exec = TemporalExecutor::new();
    exec.execute(&spec(), &ctx()).unwrap();

    let mut evidence = spec();
    evidence.factual[2].values[4] += 0.25;
    let a = exec.execute(&evidence, &ctx()).unwrap();
    assert_eq!(a.work.abductions, 4);
    assert_eq!(a.work.abductions_reused, 0);
    assert_eq!(a.result, evaluate_temporal_counterfactual(&evidence, &ctx()).unwrap());

    let mut refit = spec();
    refit.fit.mechanisms[2].parent_coefficients[3].1 = 1.25;
    let b = exec.execute(&refit, &ctx()).unwrap();
    assert_eq!(b.work.abductions, 4);
    assert_eq!(b.result, evaluate_temporal_counterfactual(&refit, &ctx()).unwrap());

    let mut snapshot = spec();
    snapshot.snapshot = SnapshotId::new("snapshot-2");
    let c = exec.execute(&snapshot, &ctx()).unwrap();
    assert_eq!(c.work.abductions, 4);
    assert_ne!(
        c.result.receipt.digest,
        exec.execute(&spec(), &ctx()).unwrap().result.receipt.digest
    );
}

#[test]
fn refusals_and_cancellation_leave_the_retained_state_unchanged() {
    let mut exec = TemporalExecutor::new();
    exec.execute(&spec(), &ctx()).unwrap();
    let (retained, counters) = (exec.retained_len(), exec.counters());
    let mut unpaired = spec();
    unpaired.plus.units.pop();
    let error = exec.execute(&unpaired, &ctx()).expect_err("unpaired");
    assert_eq!(error.refusal().unwrap().detail, "temporal_counterfactual.unpaired_histories");
    let cancelled = ExecutionContext::for_tests(1);
    cancelled.cancellation.cancel();
    assert!(exec.execute(&spec(), &cancelled).is_err());
    assert_eq!((exec.retained_len(), exec.counters()), (retained, counters));
}
