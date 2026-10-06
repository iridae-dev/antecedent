//! Uncertain pre-action state of the finite two-step sequence (2.3A, X5).
//!
//! The truth is the enumerated two-step dynamic SCM of the dependent-interval
//! test: a pre-action state `s0`, a first action `a1`, a step-2 covariate `l`
//! drawn from `P(l = 1 | s0, a1)`, a second action `a2` and a binary outcome with
//! mean `m(s0, a1, l, a2)`. For the sequence `(0, 0)` the response given `s0` is
//! `0.7 * 0.3 + 0.3 * 0.4 = 0.33` at `s0 = 0` and `0.8 * 0.4 + 0.2 * 0.7 = 0.46`
//! at `s0 = 1`. The target initial-state law is `P(s0 = 1) = 0.7`, so the
//! marginalized effect is `0.3 * 0.33 + 0.7 * 0.46 = 0.421`, whereas fixing `s0` at
//! its mode gives `0.46`. The panel itself is half `s0 = 0`, so its observed-state
//! mean (`0.395`) is a third, different quantity.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use antecedent_estimate::EstimationError;
use antecedent_estimate::temporal_dependent_interval::{
    ObservedStateSequence, SequenceHistory, TemporalEstimator, TemporalUnitPanel, UnitHistories,
};
use antecedent_estimate::temporal_initial_state::{
    FixedStateEffect, FixedStateQuery, INITIAL_STATE_MAX_STATES, InitialStateLaw,
    InitialStatePopulation, InitialStateSpec, MarginalizedEffect, MarginalizedQuery,
    route_frozen_refusal,
};

/// `P(l = 1 | s0, a1)` in tenths, indexed `[s0][a1]`.
const P_L1: [[u64; 2]; 2] = [[3, 7], [2, 8]];
/// Outcome mean in tenths, indexed `[s0][a1][l][a2]`.
const M10: [[[[i64; 2]; 2]; 2]; 2] = [
    [[[3, 4], [4, 6]], [[3, 5], [5, 7]]],
    [[[4, 3], [7, 6]], [[6, 3], [5, 7]]],
];
const SEQUENCE: [u32; 2] = [0, 0];

fn refused(error: EstimationError) -> (&'static str, String) {
    match error {
        EstimationError::Refused { code, message } => (code, message),
        other => panic!("expected a registered refusal, got {other:?}"),
    }
}

/// Response given `s0` enumerated from the SCM tables.
fn truth_given_state(s0: usize) -> f64 {
    let p1 = P_L1[s0][0] as f64 / 10.0;
    (0..2)
        .map(|l| {
            let pl = if l == 1 { p1 } else { 1.0 - p1 };
            pl * M10[s0][0][l][0] as f64 / 10.0
        })
        .sum()
}

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
                            histories.push(SequenceHistory {
                                time_id: histories.len() as u64,
                                s0: s0 as u32,
                                a1: a1 as u32,
                                l2: l as u32,
                                a2: a2 as u32,
                                y: if k < ones { 1.0 } else { 0.0 },
                            });
                        }
                    }
                }
            }
        }
        units.push(UnitHistories { unit_id: unit as u64, histories });
    }
    TemporalUnitPanel::new("initial-state-panel", Some(units)).unwrap()
}

fn law(population: InitialStatePopulation, snapshot: &str, p1: f64) -> InitialStateLaw {
    InitialStateLaw::new(population, snapshot, vec![(0, 1.0 - p1), (1, p1)]).unwrap()
}

fn target_query(p1: f64) -> MarginalizedQuery {
    let law = law(InitialStatePopulation::Target, "target-state", p1);
    MarginalizedQuery::new(SEQUENCE, InitialStateSpec::Law(law)).unwrap()
}

#[test]
fn x5_initial_state_sum_equals_the_hand_enumerated_target_sum() {
    let p = panel();
    let result = target_query(0.7).effect(&p).unwrap();
    let hand = 0.3 * 0.33 + 0.7 * 0.46;
    let enumerated = 0.3 * truth_given_state(0) + 0.7 * truth_given_state(1);
    assert!((hand - 0.421_f64).abs() < 1e-12 && (enumerated - 0.421_f64).abs() < 1e-12);
    assert!((result.value - 0.421).abs() < 1e-12, "{}", result.value);
    assert_eq!(result.contributions.len(), 2);
    for (contribution, (s0, mass, response)) in
        result.contributions.iter().zip([(0_u32, 0.3, 0.33), (1, 0.7, 0.46)])
    {
        assert_eq!(contribution.s0, s0);
        assert!((contribution.mass - mass).abs() < 1e-12);
        assert!((contribution.response - response).abs() < 1e-12);
    }
    assert_eq!(result.state_snapshot_id, "target-state");
    assert_eq!(result.panel_snapshot_id, "initial-state-panel");
    assert_eq!(result.inference_claim, "point_only");
    assert_eq!(result.sequence, SEQUENCE);
}

#[test]
fn x5_initial_state_sum_differs_from_fixing_the_state_at_its_mode() {
    let p = panel();
    let marginalized = target_query(0.7).effect(&p).unwrap();
    // The mode of the target law is s0 = 1; fixing it is the wrong answer to the
    // target-marginal question.
    let at_mode = FixedStateQuery { sequence: SEQUENCE, s0: 1 }.effect(&p).unwrap();
    assert!((at_mode.value - 0.46).abs() < 1e-12);
    assert!((marginalized.value - at_mode.value).abs() > 0.03, "fixing the state is wrong here");
    let at_zero = FixedStateQuery { sequence: SEQUENCE, s0: 0 }.effect(&p).unwrap();
    assert!((at_zero.value - 0.33).abs() < 1e-12);
    assert!(at_zero.value < marginalized.value && marginalized.value < at_mode.value);
    // The panel's own observed-state law is yet another quantity.
    let observed = ObservedStateSequence { sequence: SEQUENCE };
    let units = p.units().iter().collect::<Vec<_>>();
    let observed = observed.estimate(&units).unwrap();
    assert!((observed - 0.395).abs() < 1e-12);
    assert!((observed - marginalized.value).abs() > 0.02);
}

#[test]
fn x5_initial_state_sum_results_are_typed_and_labelled_distinctly() {
    let p = panel();
    let fixed = FixedStateQuery { sequence: SEQUENCE, s0: 1 }.effect(&p).unwrap();
    let marginalized = target_query(0.7).effect(&p).unwrap();
    assert_eq!(fixed.label(), "fixed_initial_state");
    assert_eq!(marginalized.label(), "marginalized_initial_state");
    assert_ne!(FixedStateEffect::LABEL, MarginalizedEffect::LABEL);
    assert_eq!(fixed.s0, 1);
    assert_eq!(fixed.sequence, SEQUENCE);
    let fixed_estimator = FixedStateQuery { sequence: SEQUENCE, s0: 1 };
    let marginal_estimator = target_query(0.7);
    assert_eq!(TemporalEstimator::label(&fixed_estimator), "fixed_initial_state");
    assert_eq!(TemporalEstimator::label(&marginal_estimator), "marginalized_initial_state");
}

#[test]
fn x5_initial_state_sum_follows_the_law_snapshot_and_zero_mass_states() {
    let p = panel();
    let base = target_query(0.7);
    // A different target law moves the value: 0.5 * 0.33 + 0.5 * 0.46.
    let even = target_query(0.5).effect(&p).unwrap();
    assert!((even.value - 0.395).abs() < 1e-12);
    let a = base.effect(&p).unwrap();
    assert_ne!(a.state_law_digest, even.state_law_digest);
    // The same masses under another snapshot id have another identity.
    let moved = law(InitialStatePopulation::Target, "target-state-v2", 0.7);
    assert_ne!(moved.digest(), base.law().digest());
    assert_eq!(moved.states(), base.law().states());
    // A state with zero target mass needs no history support.
    let with_zero = InitialStateLaw::new(
        InitialStatePopulation::Target,
        "target-state",
        vec![(0, 0.3), (1, 0.7), (9, 0.0)],
    )
    .unwrap();
    let query = MarginalizedQuery::new(SEQUENCE, InitialStateSpec::Law(with_zero)).unwrap();
    assert!((query.effect(&p).unwrap().value - 0.421).abs() < 1e-12);
}

#[test]
fn x5_source_state_only_refuses_a_source_law_and_a_point_state() {
    let source = law(InitialStatePopulation::Source, "source-state", 0.5);
    assert_eq!(source.population(), InitialStatePopulation::Source);
    let (code, message) =
        refused(MarginalizedQuery::new(SEQUENCE, InitialStateSpec::Law(source)).unwrap_err());
    assert_eq!(code, "transport_missing_evidence");
    assert!(message.contains("initial_state.target_law_missing"), "{message}");

    let (code, message) =
        refused(MarginalizedQuery::new(SEQUENCE, InitialStateSpec::Point(1)).unwrap_err());
    assert_eq!(code, "transport_missing_evidence");
    assert!(message.contains("initial_state.target_law_missing"), "{message}");
}

#[test]
fn x5_source_state_only_refuses_support_gaps_bounds_and_malformed_laws() {
    // State 7 has positive target mass but no history starts there.
    let gap = InitialStateLaw::new(
        InitialStatePopulation::Target,
        "target-state",
        vec![(0, 0.5), (7, 0.5)],
    )
    .unwrap();
    let query = MarginalizedQuery::new(SEQUENCE, InitialStateSpec::Law(gap)).unwrap();
    let (code, message) = refused(query.effect(&panel()).unwrap_err());
    assert_eq!(code, "transport_support_failure");
    assert!(message.contains("initial_state.support_gap"), "{message}");
    let (_, message) =
        refused(FixedStateQuery { sequence: SEQUENCE, s0: 7 }.effect(&panel()).unwrap_err());
    assert!(message.contains("initial_state.support_gap"), "{message}");

    let many = (0..=INITIAL_STATE_MAX_STATES as u32)
        .map(|s| (s, 1.0 / (INITIAL_STATE_MAX_STATES + 1) as f64))
        .collect();
    let (code, message) = refused(
        InitialStateLaw::new(InitialStatePopulation::Target, "many", many).unwrap_err(),
    );
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("initial_state.too_many_states"), "{message}");
    let at_cap = (0..INITIAL_STATE_MAX_STATES as u32)
        .map(|s| (s, 1.0 / INITIAL_STATE_MAX_STATES as f64))
        .collect();
    assert!(InitialStateLaw::new(InitialStatePopulation::Target, "cap", at_cap).is_ok());

    for bad in [
        vec![],
        vec![(0, 0.5), (1, 0.4)],
        vec![(0, 0.5), (0, 0.5)],
        vec![(0, 1.5), (1, -0.5)],
        vec![(0, f64::NAN), (1, 1.0)],
    ] {
        assert!(InitialStateLaw::new(InitialStatePopulation::Target, "s", bad).is_err());
    }
    assert!(InitialStateLaw::new(InitialStatePopulation::Target, "", vec![(0, 1.0)]).is_err());

    let (code, message) = refused(route_frozen_refusal());
    assert_eq!(code, "cell_not_licensed");
    assert!(message.contains("initial_state.route_frozen"), "{message}");
}
