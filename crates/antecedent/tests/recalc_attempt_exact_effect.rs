//! Execution-attempt accounting for a recalculation on an exactly solvable design.
//!
//! `y = 1 + 2 t + 0.3 z` with no noise and a confounded binary `t`: every contrast score is
//! exactly 2. The attempt observer must report, by the specification, a first run in which each
//! original component (identification, score construction, law summary, decision) is attempted
//! once and completed; a utility-only change attempts the decision alone; an unchanged repeat
//! attempts nothing. The numbers produced (effect 2, net benefit `2 * 2 - 0.5`) are hand values.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::recalc_attempt::{Operation, execute_with_attempt};
use antecedent::analysis::recalc_receipt::{
    RecalcRequest, RecalcSession, UtilitySpec, execute_with_receipt,
};
use antecedent_core::{ExecutionContext, StreamDomain};
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


#[test]
fn attempts_are_the_hand_enumerated_components_and_the_values_are_the_hand_values() {
    let ctx = ExecutionContext::for_tests(SEED);
    let mut session = RecalcSession::new();
    let base = request(2.0, 2.0, 0.5);

    let first = execute_with_attempt(|| execute_with_receipt(&mut session, &base, &ctx));
    let outcome = first.result.unwrap();
    assert!((outcome.law.ate - 2.0).abs() < 1e-7);
    assert!((outcome.decision.net_benefit - 3.5).abs() < 1e-7);
    for operation in [
        Operation::Identification,
        Operation::ScoreConstruction,
        Operation::LawSummary,
        Operation::Decision,
    ] {
        let count = first.report.counts(operation);
        assert_eq!((count.attempted, count.completed, count.failed), (1, 1, 0), "{operation:?}");
    }
    // The 10 fold fits (5 folds x 2 nuisance sets) are counted by the receipt. The attempt
    // instrument counts only fits of factories resolved through the learner registry; the
    // default AIPW nuisances (logistic propensity, arm-wise least squares) are direct concrete
    // fits outside it, so no learner fit is attempted.
    assert_eq!(outcome.receipt.totals().fold_fits, 10);
    assert_eq!(first.report.counts(Operation::LearnerFit).attempted, 0);
    assert!(first.report.is_complete());

    // A utility-only change attempts the decision alone.
    let mut utility = base.clone();
    utility.utility = UtilitySpec { benefit_per_unit: 0.2, cost: 0.5 };
    let second = execute_with_attempt(|| execute_with_receipt(&mut session, &utility, &ctx));
    let outcome = second.result.unwrap();
    assert!((outcome.decision.net_benefit + 0.1).abs() < 1e-7);
    assert_eq!(second.report.counts(Operation::Decision).completed, 1);
    for operation in
        [Operation::Identification, Operation::ScoreConstruction, Operation::LearnerFit]
    {
        assert_eq!(second.report.counts(operation).attempted, 0, "{operation:?}");
    }

    // An unchanged repeat attempts nothing at all.
    let third = execute_with_attempt(|| execute_with_receipt(&mut session, &utility, &ctx));
    assert!(third.result.is_ok());
    assert!(third.report.is_empty());
}
