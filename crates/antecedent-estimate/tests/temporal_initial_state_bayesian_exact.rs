//! The uncertain-initial-state route under a Dirichlet posterior mean of the target
//! initial-state law, against hand values.
//!
//! The enumerated two-step SCM of `temporal_initial_state.rs` has response 0.33 at `s0 = 0`
//! and 0.46 at `s0 = 1` for the sequence (0, 0). With target weights (0.3, 0.7) and a Dirichlet
//! concentration of 0.5 per state, the posterior mean is `(w + 0.5) / (1 + 2 * 0.5)` =
//! (0.4, 0.6), so the marginalized effect is `0.4 * 0.33 + 0.6 * 0.46 = 0.408`, computed by
//! hand here and compared with the route run on the updated law and the unit panel.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "enumerated binary SCM fixtures use bounded nonnegative counts and indices"
)]

use antecedent_estimate::posterior_mean::{
    DirichletNamespace, PosteriorRequest, dirichlet_supported_mean,
};
use antecedent_estimate::temporal_dependent_interval::{
    SequenceHistory, TemporalUnitPanel, UnitHistories,
};
use antecedent_estimate::temporal_initial_state::{
    InitialStateLaw, InitialStatePopulation, InitialStateSpec, MarginalizedQuery,
};

/// `P(l = 1 | s0, a1)` in tenths, indexed `[s0][a1]`.
const P_L1: [[u64; 2]; 2] = [[3, 7], [2, 8]];
/// Outcome mean in tenths, indexed `[s0][a1][l][a2]`.
const M10: [[[[i64; 2]; 2]; 2]; 2] =
    [[[[3, 4], [4, 6]], [[3, 5], [5, 7]]], [[[4, 3], [7, 6]], [[6, 3], [5, 7]]]];
const SEQUENCE: [u32; 2] = [0, 0];

fn panel() -> TemporalUnitPanel {
    let mut units = Vec::new();
    for unit in 0..20_usize {
        let shift: i64 = if unit % 2 == 0 { 2 } else { -2 };
        let mut histories = Vec::new();
        for s0 in 0..2_usize {
            for a1 in 0..2_usize {
                let n_one = 20 * P_L1[s0][a1];
                for l in 0..2_usize {
                    let n_l = if l == 1 { n_one } else { 200 - n_one };
                    for a2 in 0..2_usize {
                        let n_cell = n_l / 2;
                        let ones = n_cell / 10 * (M10[s0][a1][l][a2] + shift) as u64;
                        for k in 0..n_cell {
                            histories.push(SequenceHistory::two_step(
                                histories.len() as u64,
                                s0 as u32,
                                a1 as u32,
                                l as u32,
                                a2 as u32,
                                if k < ones { 1.0 } else { 0.0 },
                            ));
                        }
                    }
                }
            }
        }
        units.push(UnitHistories { unit_id: unit as u64, histories });
    }
    TemporalUnitPanel::new("initial-state-panel", Some(units)).unwrap()
}


#[test]
fn dirichlet_updated_initial_state_law_gives_the_hand_marginalized_effect() {
    let weights = [0.3, 0.7];
    let mean = dirichlet_supported_mean(
        &weights,
        PosteriorRequest::IndependentNormal { scale: 0.5 },
        DirichletNamespace::InitialState,
    )
    .unwrap();
    assert!((mean[0] - 0.4).abs() < 1e-12 && (mean[1] - 0.6).abs() < 1e-12, "{mean:?}");
    let updated = InitialStateLaw::new(
        InitialStatePopulation::Target,
        "target-state",
        vec![(0, mean[0]), (1, mean[1])],
    )
    .unwrap();
    let result = MarginalizedQuery::new(SEQUENCE, InitialStateSpec::Law(updated))
        .unwrap()
        .effect(&panel())
        .unwrap();
    let hand = 0.4 * 0.33 + 0.6 * 0.46;
    assert!((hand - 0.408_f64).abs() < 1e-12);
    assert!((result.value - hand).abs() < 1e-12, "{} vs {hand}", result.value);
    // The untouched law gives 0.421, so the update visibly moved the answer.
    assert!((result.value - 0.421).abs() > 0.01);
}
