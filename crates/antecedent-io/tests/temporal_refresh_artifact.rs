//! 2.3A X5 `new_period_refresh`: the refresh artifact re-decides and replays from its
//! embedded identities and panel summaries, and refuses every resealed mutation of the
//! period, snapshot, units, decision, receipt or interval flag.
//!
//! The held result is the target-marginal sum `0.421` of the enumerated two-step SCM on
//! the old period; the replacement panel's cell means are each `0.1` higher, so the
//! re-evaluated value is `0.3 * 0.43 + 0.7 * 0.56 = 0.521`.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss
)]

use std::collections::BTreeSet;

use antecedent_estimate::temporal_dependent_interval::{
    SequenceHistory, TemporalUnitPanel, UnitHistories,
};
use antecedent_estimate::temporal_initial_state::{
    InitialStateLaw, InitialStatePopulation, InitialStateSpec, MarginalizedEffect,
    MarginalizedQuery,
};
use antecedent_estimate::temporal_refresh::{
    HeldTemporalResult, ObservationPeriod, RefreshDecision, RefreshInvalidation,
    TemporalWindowIdentity, refresh_held,
};
use antecedent_io::IoError;
use antecedent_io::temporal_initial_state_artifact::{
    TemporalInitialStateConsumeLimits, TemporalPremisesWire,
};
use antecedent_io::temporal_refresh_artifact::{
    RefreshInputs, TEMPORAL_REFRESH_ARTIFACT_FEATURE, TemporalRefreshArtifactWire,
};

/// `P(l = 1 | s0, a1)` in tenths, indexed `[s0][a1]`.
const P_L1: [[u64; 2]; 2] = [[3, 7], [2, 8]];
/// Outcome mean in tenths, indexed `[s0][a1][l][a2]`.
const M10: [[[[u64; 2]; 2]; 2]; 2] =
    [[[[3, 4], [4, 6]], [[3, 5], [5, 7]]], [[[4, 3], [7, 6]], [[6, 3], [5, 7]]]];
const SEQUENCE: [u32; 2] = [0, 0];

fn limits() -> TemporalInitialStateConsumeLimits {
    TemporalInitialStateConsumeLimits::default()
}

/// Two identical units; every cell mean is raised by `lift` tenths; time ids start at `start`.
fn panel(snapshot: &str, first_unit: u64, start: u64, lift: u64) -> TemporalUnitPanel {
    let mut units = Vec::new();
    for unit in 0..2_u64 {
        let mut histories = Vec::new();
        for s0 in 0..2_usize {
            for a1 in 0..2_usize {
                let n_one = 20 * P_L1[s0][a1];
                for l in 0..2_usize {
                    let n_l = if l == 1 { n_one } else { 200 - n_one };
                    for a2 in 0..2_usize {
                        let n_cell = n_l / 2;
                        let ones = n_cell * (M10[s0][a1][l][a2] + lift) / 10;
                        for k in 0..n_cell {
                            histories.push(SequenceHistory {
                                time_id: start + histories.len() as u64,
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
        units.push(UnitHistories { unit_id: first_unit + unit, histories });
    }
    TemporalUnitPanel::new(snapshot, Some(units)).unwrap()
}

fn set(items: &[&str]) -> BTreeSet<String> {
    items.iter().map(|s| (*s).to_owned()).collect()
}

fn identity(snapshot: &str, period: (i64, i64), units: &[&str]) -> TemporalWindowIdentity {
    TemporalWindowIdentity {
        graph_id: "two_slice_graph_v1".into(),
        horizon: 2,
        lag_alignment: vec![
            ("s0".into(), 1),
            ("a1".into(), 1),
            ("l2".into(), 2),
            ("a2".into(), 2),
            ("y".into(), 2),
        ],
        intervention_history: vec!["a1=0".into(), "a2=0".into()],
        selection_targets: set(&["s_t1", "s_t2"]),
        regimes: set(&["source", "target"]),
        period: ObservationPeriod { start: period.0, end: period.1 },
        unit_ids: set(units),
        snapshot_id: snapshot.into(),
        proof_id: "proof_1".into(),
    }
}

fn old_identity() -> TemporalWindowIdentity {
    identity("snap_old", (0, 1000), &["0", "1"])
}

fn new_identity() -> TemporalWindowIdentity {
    identity("snap_new", (1000, 2000), &["10", "11"])
}

fn premises() -> TemporalPremisesWire {
    TemporalPremisesWire {
        initial_state_variable: "s0".into(),
        time_order: ["s0", "a1", "l2", "a2", "y"].iter().map(|s| (*s).to_owned()).collect(),
        source_regime: "source".into(),
        target_regime: "target".into(),
        graph_id: "two_slice_graph_v1".into(),
        proof_id: "proof_1".into(),
    }
}

fn query() -> MarginalizedQuery {
    let law = InitialStateLaw::new(
        InitialStatePopulation::Target,
        "target-state",
        vec![(0, 0.3), (1, 0.7)],
    )
    .unwrap();
    MarginalizedQuery::new(SEQUENCE, InitialStateSpec::Law(law)).unwrap()
}

fn build(
    new: &TemporalWindowIdentity,
    interval: bool,
) -> Result<TemporalRefreshArtifactWire, IoError> {
    let (old_panel, new_panel) = (panel("snap_old", 0, 0, 0), panel("snap_new", 10, 1000, 1));
    TemporalRefreshArtifactWire::checked(&RefreshInputs {
        premises: &premises(),
        sequence: SEQUENCE,
        query: &query(),
        old_identity: &old_identity(),
        old_panel: &old_panel,
        new_identity: new,
        new_panel: &new_panel,
        interval_invalidated: interval,
    })
}

fn artifact() -> TemporalRefreshArtifactWire {
    build(&new_identity(), true).unwrap()
}

fn reseal(mut wire: TemporalRefreshArtifactWire) -> Vec<u8> {
    wire.seal = wire.compute_seal().unwrap();
    wire.export().unwrap()
}

fn consume(bytes: &[u8]) -> IoError {
    TemporalRefreshArtifactWire::consume(bytes, &limits()).unwrap_err()
}

fn convert_detail(error: IoError) -> String {
    match error {
        IoError::Convert(message) => message,
        other => panic!("expected a conversion failure, got {other:?}"),
    }
}

fn refused(error: IoError) -> (&'static str, String) {
    match error {
        IoError::Refused { code, message } => (code, message),
        other => panic!("expected a coded refusal, got {other:?}"),
    }
}

#[test]
fn x5_period_replace_artifact_replays_the_re_evaluated_point() {
    let bytes = artifact().export().unwrap();
    let (wire, replay) = TemporalRefreshArtifactWire::consume(&bytes, &limits()).unwrap();
    assert_eq!(replay.decision, RefreshDecision::Reusable);
    assert!((replay.old_value - 0.421).abs() < 1e-12, "{}", replay.old_value);
    let new_value = replay.new_value.unwrap();
    assert!((new_value - 0.521).abs() < 1e-12, "{new_value}");
    // Re-evaluated, not copied.
    assert!((new_value - replay.old_value).abs() > 0.05);
    assert_eq!(wire.decision, "reusable");
    assert_eq!(wire.invalidation, None);
    assert!(wire.interval_invalidated);
    let refreshed = wire.refreshed.as_ref().unwrap();
    assert_eq!(refreshed.label, MarginalizedEffect::LABEL);
    assert!(!refreshed.interval_present);
    assert_eq!(new_value.to_bits(), refreshed.value.to_bits());
    let receipt = wire.receipt.as_ref().unwrap();
    assert!(receipt.interval_invalidated);
    assert_eq!(receipt.inference_claim, "point_only");
    assert_eq!(receipt.old_period, (0, 1000));
    assert_eq!(receipt.new_period, (1000, 2000));
    assert_eq!(receipt.new_snapshot_id, "snap_new");
    assert_eq!(wire.new_identity_digest().unwrap(), receipt.new_identity_digest);
}

#[test]
fn x5_period_replace_artifact_receipt_is_the_core_refresh_receipt() {
    let q = query();
    let old_panel = panel("snap_old", 0, 0, 0);
    let new_panel = panel("snap_new", 10, 1000, 1);
    let held = HeldTemporalResult {
        identity: old_identity(),
        value: q.effect(&old_panel).unwrap(),
        interval_present: true,
    };
    let (refreshed, core_receipt) =
        refresh_held(&held, new_identity(), |_| q.effect(&new_panel)).unwrap();
    let wire = artifact();
    let receipt = wire.receipt.unwrap();
    assert_eq!(receipt.digest, core_receipt.digest);
    assert_eq!(receipt.new_identity_digest, core_receipt.new_identity_digest);
    assert_eq!(wire.refreshed.unwrap().value.to_bits(), refreshed.value.value.to_bits());
}

#[test]
fn x5_period_replace_artifact_without_a_prior_interval_records_none() {
    let wire = build(&new_identity(), false).unwrap();
    assert!(!wire.interval_invalidated);
    assert!(!wire.receipt.as_ref().unwrap().interval_invalidated);
    TemporalRefreshArtifactWire::consume(&wire.export().unwrap(), &limits()).unwrap();
}

type IdentityEdit = (fn(&mut TemporalWindowIdentity), RefreshInvalidation);

#[test]
fn x5_third_period_artifact_records_a_typed_invalidation_and_no_result() {
    let edits: [IdentityEdit; 4] = [
        (|i| i.horizon = 3, RefreshInvalidation::HorizonChanged),
        (|i| i.lag_alignment[2].1 = 1, RefreshInvalidation::LagAlignmentChanged),
        (
            |i| i.intervention_history[1] = "a2=1".into(),
            RefreshInvalidation::InterventionHistoryChanged,
        ),
        (|i| i.proof_id = "proof_2".into(), RefreshInvalidation::PremisesChanged),
    ];
    for (edit, reason) in edits {
        let mut new = new_identity();
        edit(&mut new);
        let wire = build(&new, true).unwrap();
        assert_eq!(wire.decision, "invalidated");
        assert_eq!(wire.invalidation.as_deref(), Some(reason.detail()));
        assert!(wire.refreshed.is_none() && wire.receipt.is_none());
        let (_, replay) =
            TemporalRefreshArtifactWire::consume(&wire.export().unwrap(), &limits()).unwrap();
        assert_eq!(replay.decision, RefreshDecision::Invalidated(reason));
        assert_eq!(replay.new_value, None);
        assert!((replay.old_value - 0.421).abs() < 1e-12);
    }
}

#[test]
fn x5_period_reseal_refuses_unsealed_edits_and_a_foreign_feature() {
    let mut wire = artifact();
    wire.new_window.period_end = 2500;
    let message = convert_detail(consume(&wire.export().unwrap()));
    assert!(message.contains("temporal_refresh_artifact.seal"), "{message}");

    let mut foreign = artifact();
    foreign.required_features = vec!["temporal_initial_state_target_law_v1".into()];
    assert_ne!(TEMPORAL_REFRESH_ARTIFACT_FEATURE, "temporal_initial_state_target_law_v1");
    let message = convert_detail(
        TemporalRefreshArtifactWire::decode(&foreign.export().unwrap()).unwrap_err(),
    );
    assert!(message.contains("unsupported_semantics"), "{message}");
}

#[test]
fn x5_period_reseal_refuses_a_changed_period_snapshot_or_units() {
    // Period no longer covering the new panel's time ids.
    let mut period = artifact();
    period.new_window.period_end = 1500;
    let (code, message) = refused(consume(&reseal(period)));
    assert_eq!(code, "invalid_argument");
    assert!(message.contains("temporal_refresh.invalid_replacement"), "{message}");

    // A wider period still covers the panel but is not the one the receipt binds.
    let mut wider = artifact();
    wider.new_window.period_end = 3000;
    let (code, message) = refused(consume(&reseal(wider)));
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("temporal_refresh.receipt_mismatch"), "{message}");

    // A snapshot id that is not the panel's.
    let mut snapshot = artifact();
    snapshot.new_window.snapshot_id = "snap_forged".into();
    let (code, message) = refused(consume(&reseal(snapshot)));
    assert_eq!(code, "invalid_argument");
    assert!(message.contains("temporal_refresh.invalid_replacement"), "{message}");

    // Units that are not the panel's units.
    let mut units = artifact();
    units.new_window.unit_ids = vec!["10".into(), "12".into()];
    let (code, message) = refused(consume(&reseal(units)));
    assert_eq!(code, "invalid_argument");
    assert!(message.contains("temporal_refresh.invalid_replacement"), "{message}");

    // The old window's period rewritten: not the one the receipt binds.
    let mut old_period = artifact();
    old_period.old_window.period_end = 1005;
    let (code, message) = refused(consume(&reseal(old_period)));
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("temporal_refresh.receipt_mismatch"), "{message}");
}

#[test]
fn x5_period_reseal_refuses_a_replacement_that_is_the_old_snapshot() {
    // Both the window and its panel claim the old snapshot: the identities are
    // self-consistent but the refresh is stale, so a stored `reusable` is refused.
    let mut wire = artifact();
    wire.new_window.snapshot_id = "snap_old".into();
    wire.new_panel.snapshot_id = "snap_old".into();
    let (code, message) = refused(consume(&reseal(wire)));
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("temporal_refresh.stale_snapshot"), "{message}");
}

#[test]
fn x5_period_reseal_refuses_a_resealed_change_of_decision_receipt_or_interval() {
    let mut flipped = artifact();
    flipped.decision = "invalidated".into();
    flipped.invalidation = Some("temporal_refresh.horizon_changed".into());
    let message = convert_detail(consume(&reseal(flipped)));
    assert!(message.contains("temporal_refresh_artifact.decision"), "{message}");

    let mut interval_flag = artifact();
    interval_flag.interval_invalidated = false;
    let (_, message) = refused(consume(&reseal(interval_flag)));
    assert!(message.contains("temporal_refresh.receipt_mismatch"), "{message}");

    let mut receipt = artifact();
    receipt.receipt.as_mut().unwrap().proof_id = "proof_forged".into();
    let (_, message) = refused(consume(&reseal(receipt)));
    assert!(message.contains("temporal_refresh.receipt_digest_mismatch"), "{message}");

    let mut stale = artifact();
    stale.refreshed.as_mut().unwrap().interval_present = true;
    let (code, message) = refused(consume(&reseal(stale)));
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("temporal_refresh.stale_interval"), "{message}");

    let mut dropped = artifact();
    dropped.receipt = None;
    let message = convert_detail(consume(&reseal(dropped)));
    assert!(message.contains("reusable_without_result"), "{message}");
}

#[test]
fn x5_period_reseal_refuses_a_resealed_change_of_value_law_or_premises() {
    let mut copied = artifact();
    copied.refreshed.as_mut().unwrap().value = copied.old_value;
    let message = convert_detail(consume(&reseal(copied)));
    assert!(message.contains("temporal_refresh_artifact.new_value"), "{message}");

    let mut old_value = artifact();
    old_value.old_value = 0.5;
    let message = convert_detail(consume(&reseal(old_value)));
    assert!(message.contains("temporal_refresh_artifact.old_value"), "{message}");

    let mut source = artifact();
    source.law.population = "source".into();
    let (code, message) = refused(consume(&reseal(source)));
    assert_eq!(code, "transport_missing_evidence");
    assert!(message.contains("initial_state.target_law_missing"), "{message}");

    let mut graph = artifact();
    graph.premises.graph_id = "other_graph".into();
    let message = convert_detail(consume(&reseal(graph)));
    assert!(message.contains("temporal_refresh_artifact.premises"), "{message}");

    let mut label = artifact();
    label.refreshed.as_mut().unwrap().label = "fixed_initial_state".into();
    let message = convert_detail(consume(&reseal(label)));
    assert!(message.contains("temporal_refresh_artifact.label"), "{message}");
}

#[test]
fn x5_period_reseal_refuses_an_invalidated_record_that_carries_a_result() {
    let mut new = new_identity();
    new.horizon = 3;
    let mut wire = build(&new, false).unwrap();
    let good = artifact();
    wire.refreshed = good.refreshed;
    wire.receipt = good.receipt;
    let message = convert_detail(consume(&reseal(wire)));
    assert!(message.contains("invalidated_carries_result"), "{message}");
}
