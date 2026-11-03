//! Cell-saturated AIPW on a design whose interaction effect is known exactly.
//!
//! `y = effect * a * d + 0.5 * a + 0.2 * z` with no noise, two confounded binary treatments `a`
//! and `d`, and a continuous adjustment `z`. Every cell outcome regression `E[y | a, d, z]` is
//! exactly linear in `z`, so the cross-fitted interaction contrast score of every row is
//! `mu_11 - mu_10 - mu_01 + mu_00 = effect`, and the interaction equals `effect` for the
//! observed population and for any positive target weights. The first run fits 5 folds x (1
//! multinomial propensity + 4 cell outcome regressions) = 25 cell models by the specification;
//! a utility change fits nothing. The expected numbers are written out by hand here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::recalc_cell::{
    CellRequest, CellSession, CellSpec, ScoreQuantity, execute_cell_with_receipt,
};
use antecedent::analysis::recalc_receipt::{TargetWeights, UtilitySpec};
use antecedent_core::{ExecutionContext, StreamDomain, VariableId};
use antecedent_estimate::CellSaturatedAipw;
use antecedent_estimate::cell_aipw::count_cell_model_fits;
use antecedent_kernels::standard_normal;

const A: u32 = 0;
const D: u32 = 1;
const Y: u32 = 2;
const Z: u32 = 3;
const N: usize = 1200;
const SEED: u64 = 71;

fn columns(effect: f64) -> Vec<(String, Vec<f64>)> {
    let mut rng = ExecutionContext::for_tests(SEED).rng.stream_for(StreamDomain::Test, 0xC3);
    let (mut a, mut d, mut y, mut z) = (vec![0.0; N], vec![0.0; N], vec![0.0; N], vec![0.0; N]);
    for i in 0..N {
        let zi = standard_normal(&mut rng);
        let p = 1.0 / (1.0 + (-(0.5 * zi)).exp());
        z[i] = zi;
        a[i] = f64::from(rng.next_f64() < p);
        d[i] = f64::from(rng.next_f64() < 0.5);
        y[i] = effect * a[i] * d[i] + 0.5 * a[i] + 0.2 * zi;
    }
    vec![("a".into(), a), ("d".into(), d), ("y".into(), y), ("z".into(), z)]
}

fn request(effect: f64, benefit: f64, cost: f64) -> CellRequest {
    CellRequest {
        columns: columns(effect),
        spec: CellSpec {
            edges: vec![(Z, A), (Z, D), (Z, Y), (A, Y), (D, Y)],
            treatments: vec![A, D],
            outcome: Y,
            adjustment: vec![Z],
            estimator: CellSaturatedAipw::new(),
            quantity: ScoreQuantity::Interaction,
            target: None,
            utility: UtilitySpec { benefit_per_unit: benefit, cost },
        },
    }
}

fn close(a: f64, b: f64, what: &str) {
    assert!((a - b).abs() <= 1e-7, "{what}: {a} vs {b}");
}

#[test]
fn cell_aipw_interaction_equals_the_hand_effect_with_25_first_run_fits_and_none_on_reuse() {
    let ctx = ExecutionContext::for_tests(SEED);
    let mut session = CellSession::new();
    let base = request(1.5, 2.0, 0.5);

    let (first, measured) =
        count_cell_model_fits(|| execute_cell_with_receipt(&mut session, &base, &ctx));
    let first = first.unwrap();
    // 5 folds x (1 propensity + 4 cell regressions) = 25, measured from outside the executor.
    assert_eq!(measured, 25);
    assert_eq!(first.receipt.totals().fold_fits, 25);
    close(first.law.ate, 1.5, "interaction");
    // The rule: 2 * 1.5 - 0.5 = 2.5.
    close(first.decision.net_benefit, 2.5, "net benefit");
    assert!(first.decision.treat);

    // Utility only: zero cell fits; 0.2 * 1.5 - 0.5 = -0.2, do not treat.
    let mut utility = base;
    utility.spec.utility = UtilitySpec { benefit_per_unit: 0.2, cost: 0.5 };
    let (second, measured) =
        count_cell_model_fits(|| execute_cell_with_receipt(&mut session, &utility, &ctx));
    let second = second.unwrap();
    assert_eq!(measured, 0);
    close(second.law.ate, 1.5, "interaction after a utility change");
    close(second.decision.net_benefit, -0.2, "second net benefit");
    assert!(!second.decision.treat);

    // Target weights: reweight only; a constant interaction stays 1.5.
    let mut target = utility;
    let z = &target.columns[Z as usize].1;
    let weights: Vec<f64> = z.iter().map(|v| (0.4 * v).exp()).collect();
    target.spec.target = Some(TargetWeights { weights, depends_on: vec![VariableId::from_raw(Z)] });
    let (third, measured) =
        count_cell_model_fits(|| execute_cell_with_receipt(&mut session, &target, &ctx));
    let third = third.unwrap();
    assert_eq!(measured, 0);
    close(third.law.ate, 1.5, "interaction under reweighting");
}
