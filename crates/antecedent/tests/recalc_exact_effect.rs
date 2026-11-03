//! Selective recalculation on a constant-effect design whose answer is known exactly.
//!
//! `y = 1 + effect * t + 0.3 z` with no noise and a confounded binary `t`: both arm-wise
//! outcome regressions are exactly linear, so every row's contrast score is `mu_1 - mu_0 =
//! effect`, and the average effect equals `effect` for the observed population and for any
//! positive target weights. The net benefit of the rule is `benefit * effect - cost`. Each step
//! below states, by hand from the specification, which stages must run (5 folds x 2 nuisance
//! sets = 10 fold fits on a first fit, none on a reweight or a utility change) and what number
//! comes out; none of it is read back from the production path.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::recalc_receipt::{
    RecalcOutcome, RecalcRequest, RecalcSession, TargetWeights, UtilitySpec, execute_with_receipt,
};
use antecedent::analysis::recalc_cell::{
    ScoreQuantity, ScoreResumeRequest, ScoreResumeSession, execute_resumed_retarget,
    freeze_crossfit_scores,
};
use antecedent_core::recalc::RetargetSupport;
use antecedent_core::{ExecutionContext, StreamDomain, VariableId};
use antecedent_estimate::AipwAte;
use antecedent_kernels::standard_normal;

const T: u32 = 0;
const Y: u32 = 1;
const Z: u32 = 2;
const N: usize = 400;
const SEED: u64 = 71;

fn columns(effect: f64) -> Vec<(String, Vec<f64>)> {
    let mut rng = ExecutionContext::for_tests(SEED).rng.stream_for(StreamDomain::Test, 0xC3);
    let (mut t, mut y, mut z) = (vec![0.0; N], vec![0.0; N], vec![0.0; N]);
    for i in 0..N {
        let zi = standard_normal(&mut rng);
        let p = 1.0 / (1.0 + (-(-0.2 + 0.8 * zi)).exp());
        z[i] = zi;
        t[i] = f64::from(rng.next_f64() < p);
        y[i] = 1.0 + effect * t[i] + 0.3 * zi;
    }
    vec![("t".into(), t), ("y".into(), y), ("z".into(), z)]
}

fn request(effect: f64, benefit: f64, cost: f64) -> RecalcRequest {
    RecalcRequest {
        columns: columns(effect).into(),
        edges: vec![(Z, T), (Z, Y), (T, Y)],
        treatment: T,
        outcome: Y,
        estimator: AipwAte::new().with_bootstrap_replicates(0),
        target: None,
        utility: UtilitySpec { benefit_per_unit: benefit, cost },
    }
}

fn go(session: &mut RecalcSession, request: &RecalcRequest) -> RecalcOutcome {
    execute_with_receipt(session, request, &ExecutionContext::for_tests(SEED)).unwrap()
}

fn work(o: &RecalcOutcome) -> (u64, u64, u64, u64, u64) {
    let c = o.receipt.totals();
    (c.identifications, c.fold_fits, c.score_computations, c.reweights, c.decisions)
}

fn close(a: f64, b: f64, what: &str) {
    assert!((a - b).abs() <= 1e-7, "{what}: {a} vs {b}");
}

#[test]
fn selective_recalculation_gives_the_hand_values_and_refits_only_the_hand_enumerated_stages() {
    let mut session = RecalcSession::new();
    let base = request(2.0, 2.0, 0.5);

    // First run: everything runs once; effect 2, net benefit 2 * 2 - 0.5 = 3.5, treat.
    let first = go(&mut session, &base);
    assert_eq!(work(&first), (1, 10, 1, 1, 1));
    close(first.law.ate, 2.0, "first effect");
    close(first.decision.net_benefit, 3.5, "first net benefit");
    assert!(first.decision.treat);

    // Utility only: no fit of any kind; 0.2 * 2 - 0.5 = -0.1, do not treat.
    let mut utility = base.clone();
    utility.utility = UtilitySpec { benefit_per_unit: 0.2, cost: 0.5 };
    let second = go(&mut session, &utility);
    assert_eq!(work(&second), (0, 0, 0, 0, 1));
    close(second.law.ate, 2.0, "effect is untouched by a utility change");
    close(second.decision.net_benefit, -0.1, "second net benefit");
    assert!(!second.decision.treat);

    // Target weights: the frozen scores are reweighted, nothing is refitted; a constant
    // effect stays 2 under any positive weights.
    let mut target = utility.clone();
    let z = &target.columns[Z as usize].1;
    let weights: Vec<f64> = z.iter().map(|v| (0.4 * v).exp()).collect();
    target.target = Some(TargetWeights { weights, depends_on: vec![VariableId::from_raw(Z)] });
    let third = go(&mut session, &target);
    assert_eq!(work(&third), (0, 0, 0, 1, 1));
    close(third.law.ate, 2.0, "constant effect under reweighting");
    close(third.decision.net_benefit, -0.1, "third net benefit");

    // New outcomes with effect 3: refit the cross-fitted design (a multiple of 5 fold fits,
    // at least 10) without re-identifying; the effect is 3 and the net benefit 0.2 * 3 - 0.5.
    let fresh = request(3.0, 0.2, 0.5);
    let fourth = go(&mut session, &fresh);
    let (identifications, fits, scores, reweights, decisions) = work(&fourth);
    assert_eq!((identifications, scores, reweights, decisions), (0, 1, 1, 1));
    assert!(fits >= 10 && fits % 5 == 0, "{fits}");
    close(fourth.law.ate, 3.0, "new-outcome effect");
    close(fourth.decision.net_benefit, 0.1, "fourth net benefit");
    assert!(fourth.decision.treat);
}

#[test]
fn receipt_counts_for_each_step_equal_the_hand_enumerated_work_and_identities_track_changes() {
    let mut session = RecalcSession::new();
    let base = request(2.0, 2.0, 0.5);
    let first = go(&mut session, &base);
    // A repeat of an unchanged request fits, identifies, scores and reweights nothing.
    let again = go(&mut session, &base);
    let (identifications, fits, scores, reweights, _) = work(&again);
    assert_eq!((identifications, fits, scores, reweights), (0, 0, 0, 0));
    close(again.law.ate, first.law.ate, "unchanged law");
    // The effect changes only with the outcome column; the receipt of the changed-outcome run
    // differs from the first.
    let changed = go(&mut session, &request(3.0, 2.0, 0.5));
    assert_ne!(changed.receipt.identity(), first.receipt.identity());
    assert_eq!(again.receipt.totals().identifications, 0);
    close(changed.decision.net_benefit, 2.0 * 3.0 - 0.5, "net benefit follows the effect");
}

#[test]
fn frozen_scores_resume_in_a_fresh_session_to_the_hand_effect_and_hand_decision() {
    let mut session = RecalcSession::new();
    let base = request(2.0, 2.0, 0.5);
    let first = go(&mut session, &base);
    assert_eq!(first.receipt.totals().fold_fits, 10);
    let artifact = freeze_crossfit_scores(&session, N as u64, RetargetSupport::Supported).unwrap();
    let bytes = artifact.to_bytes("exact-effect-scores").unwrap();

    // A fresh process sees only the bytes: no data, no learner, no fit.
    let mut fresh =
        ScoreResumeSession::resume_from_score_bytes(&bytes, Some(artifact.identity())).unwrap();
    {
        let table = fresh.score_table();
        let column = |arm: u32| {
            let at = table
                .columns
                .iter()
                .position(|c| c.arm == arm && c.threshold.is_none())
                .expect("an arm column");
            table.column(at).unwrap().to_vec()
        };
        let (control, active) = (column(0), column(1));
        assert_eq!(control.len(), N);
        // Every stored row contrast is the hand effect 2.
        for (a, c) in active.iter().zip(&control) {
            assert!((a - c - 2.0).abs() < 1e-7, "{a} - {c}");
        }
    }
    let mut weights_base = base.clone();
    let z = &weights_base.columns[Z as usize].1;
    let weights: Vec<f64> = z.iter().map(|v| (0.4 * v).exp()).collect();
    weights_base.target =
        Some(TargetWeights { weights, depends_on: vec![VariableId::from_raw(Z)] });
    let resume = ScoreResumeRequest {
        n_variables: 3,
        edges: vec![(Z, T), (Z, Y), (T, Y)],
        target: weights_base.target.clone(),
        target_row_ids: Some(fresh.score_table().row_index.iter().map(|id| id.get()).collect()),
        utility: UtilitySpec { benefit_per_unit: 4.0, cost: 1.0 },
        quantity: ScoreQuantity::AverageEffect,
        changed_inputs: Vec::new(),
    };
    let resumed = execute_resumed_retarget(&mut fresh, &resume).unwrap();
    assert_eq!(resumed.receipt.totals().fold_fits, 0);
    // Reweighting a constant effect leaves it at 2; the rule gives 4 * 2 - 1 = 7.
    close(resumed.law.ate, 2.0, "resumed effect");
    close(resumed.decision.net_benefit, 7.0, "resumed net benefit");
    assert!(resumed.decision.treat);
}
