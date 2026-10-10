//! 2.3A X5 `uncertain_initial_state`: the target-marginal initial-state artifact replays
//! from its embedded law and panel summary, and refuses every resealed mutation.
//!
//! The truth is the enumerated two-step dynamic SCM of the core tests: for the sequence
//! `(0, 0)` the response given `s0` is `0.33` at `s0 = 0` and `0.46` at `s0 = 1`; the
//! target law `P(s0 = 1) = 0.7` gives the integrated value `0.3 * 0.33 + 0.7 * 0.46 =
//! 0.421`, while fixing the state at its mode gives `0.46`.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "enumerated binary SCM fixtures use bounded nonnegative counts and indices"
)]

use antecedent_estimate::temporal_dependent_interval::{
    SequenceHistory, TemporalUnitPanel, UnitHistories,
};
use antecedent_estimate::temporal_initial_state::{
    FixedStateEffect, FixedStateQuery, InitialStateLaw, InitialStatePopulation, InitialStateSpec,
    MarginalizedEffect, MarginalizedQuery,
};
use antecedent_io::IoError;
use antecedent_io::temporal_initial_state_artifact::{
    InitialStateLawWire, TEMPORAL_INITIAL_STATE_ARTIFACT_FEATURE, TemporalInitialStateArtifactWire,
    TemporalInitialStateConsumeLimits, TemporalPremisesWire,
};
use antecedent_io::temporal_transport_artifact::{
    TEMPORAL_TRANSPORT_ARTIFACT_FEATURE, TemporalSequenceArtifactWire,
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

/// Two identical units, each holding every `(s0, a1)` block in the SCM's proportions.
fn panel(snapshot: &str) -> TemporalUnitPanel {
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
                        let ones = n_cell * M10[s0][a1][l][a2] / 10;
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
        units.push(UnitHistories { unit_id: unit, histories });
    }
    TemporalUnitPanel::new(snapshot, Some(units)).unwrap()
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

fn target_law(snapshot: &str, p1: f64) -> InitialStateLaw {
    InitialStateLaw::new(InitialStatePopulation::Target, snapshot, vec![(0, 1.0 - p1), (1, p1)])
        .unwrap()
}

fn artifact() -> TemporalInitialStateArtifactWire {
    let query =
        MarginalizedQuery::new(SEQUENCE, InitialStateSpec::Law(target_law("target-state", 0.7)))
            .unwrap();
    let fixed = FixedStateQuery { sequence: SEQUENCE, s0: 1 };
    TemporalInitialStateArtifactWire::checked(
        premises(),
        SEQUENCE,
        &query,
        Some(&fixed),
        &panel("state-panel"),
    )
    .unwrap()
}

fn reseal(mut wire: TemporalInitialStateArtifactWire) -> Vec<u8> {
    wire.seal = wire.compute_seal().unwrap();
    wire.export().unwrap()
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
fn x5_initial_state_artifact_replays_the_integrated_value_and_the_fixed_mode() {
    let bytes = artifact().export().unwrap();
    let (wire, replay) = TemporalInitialStateArtifactWire::consume(&bytes, &limits()).unwrap();
    assert!((replay.value - 0.421).abs() < 1e-12, "{}", replay.value);
    let (state, fixed) = replay.fixed.unwrap();
    assert_eq!(state, 1);
    assert!((fixed - 0.46).abs() < 1e-12, "{fixed}");
    assert!((replay.value - fixed).abs() > 0.03, "fixing the state is a different answer");
    assert_eq!(replay.value.to_bits(), wire.marginalized.value.to_bits());
    assert_eq!(replay.contributions.len(), 2);
    assert!((replay.contributions[0].response - 0.33).abs() < 1e-12);
    assert!((replay.contributions[1].response - 0.46).abs() < 1e-12);
    // The two result labels are fixed strings and never swapped.
    assert_eq!(wire.marginalized.label, MarginalizedEffect::LABEL);
    assert_eq!(wire.fixed.as_ref().unwrap().label, FixedStateEffect::LABEL);
    assert_ne!(wire.marginalized.label, wire.fixed.as_ref().unwrap().label);
    assert_eq!(wire.marginalized.inference_claim, "point_only");
    assert_eq!(wire.law.population, "target");
    assert_eq!(wire.law.snapshot_id, "target-state");
    assert_eq!(wire.panel.snapshot_id, "state-panel");
    assert_eq!(wire.premises.initial_state_variable, "s0");
    assert_eq!(wire.premises.time_order[0], "s0");
}

#[test]
fn x5_initial_state_artifact_matches_the_core_evaluation_bit_for_bit() {
    let p = panel("state-panel");
    let query =
        MarginalizedQuery::new(SEQUENCE, InitialStateSpec::Law(target_law("target-state", 0.7)))
            .unwrap();
    let core = query.effect(&p).unwrap();
    let (_, replay) =
        TemporalInitialStateArtifactWire::consume(&artifact().export().unwrap(), &limits())
            .unwrap();
    assert_eq!(replay.value.to_bits(), core.value.to_bits());
    for (replayed, original) in replay.contributions.iter().zip(&core.contributions) {
        assert_eq!(replayed.state, original.s0);
        assert_eq!(replayed.response.to_bits(), original.response.to_bits());
    }
}

#[test]
fn x5_initial_state_artifact_refuses_unsealed_edits() {
    let mut wire = artifact();
    wire.marginalized.value += 0.001;
    let message = convert_detail(
        TemporalInitialStateArtifactWire::consume(&wire.export().unwrap(), &limits()).unwrap_err(),
    );
    assert!(message.contains("temporal_initial_state_artifact.seal"), "{message}");
}

#[test]
fn x5_initial_state_artifact_refuses_a_resealed_changed_value_or_swapped_labels() {
    let mut value = artifact();
    value.marginalized.value = 0.46;
    let message = convert_detail(
        TemporalInitialStateArtifactWire::consume(&reseal(value), &limits()).unwrap_err(),
    );
    assert!(message.contains("temporal_initial_state_artifact.value"), "{message}");

    // The fixed-mode value presented as the marginalized one.
    let mut relabeled = artifact();
    let fixed = relabeled.fixed.clone().unwrap();
    relabeled.marginalized.value = fixed.value;
    let message = convert_detail(
        TemporalInitialStateArtifactWire::consume(&reseal(relabeled), &limits()).unwrap_err(),
    );
    assert!(message.contains("temporal_initial_state_artifact.value"), "{message}");

    let mut swapped = artifact();
    std::mem::swap(&mut swapped.marginalized.label, &mut swapped.fixed.as_mut().unwrap().label);
    let message = convert_detail(
        TemporalInitialStateArtifactWire::consume(&reseal(swapped), &limits()).unwrap_err(),
    );
    assert!(message.contains("temporal_initial_state_artifact.label"), "{message}");

    let mut claim = artifact();
    claim.marginalized.inference_claim = "calibrated".into();
    let message = convert_detail(
        TemporalInitialStateArtifactWire::consume(&reseal(claim), &limits()).unwrap_err(),
    );
    assert!(message.contains("temporal_initial_state_artifact.inference_claim"), "{message}");
}

#[test]
fn x5_initial_state_artifact_refuses_a_resealed_changed_law_snapshot_or_population() {
    // Changed masses under a recomputed law digest: the stored result no longer matches.
    let mut moved = artifact();
    moved.law = InitialStateLawWire::from_law(&target_law("target-state", 0.5));
    let message = convert_detail(
        TemporalInitialStateArtifactWire::consume(&reseal(moved), &limits()).unwrap_err(),
    );
    assert!(message.contains("temporal_initial_state_artifact.identity"), "{message}");

    // Masses edited under the stale digest.
    let mut stale = artifact();
    stale.law.states[0].mass = 0.4;
    stale.law.states[1].mass = 0.6;
    let message = convert_detail(
        TemporalInitialStateArtifactWire::consume(&reseal(stale), &limits()).unwrap_err(),
    );
    assert!(message.contains("temporal_initial_state_artifact.law_digest"), "{message}");

    // Another law snapshot id under the stale digest.
    let mut snapshot = artifact();
    snapshot.law.snapshot_id = "target-state-v2".into();
    let message = convert_detail(
        TemporalInitialStateArtifactWire::consume(&reseal(snapshot), &limits()).unwrap_err(),
    );
    assert!(message.contains("temporal_initial_state_artifact.law_digest"), "{message}");

    // A source-labelled law cannot answer a target-marginal query.
    let mut source = artifact();
    source.law.population = "source".into();
    let (code, message) =
        refused(TemporalInitialStateArtifactWire::consume(&reseal(source), &limits()).unwrap_err());
    assert_eq!(code, "transport_missing_evidence");
    assert!(message.contains("initial_state.target_law_missing"), "{message}");

    // The stored result bound to another panel snapshot.
    let mut other_panel = artifact();
    other_panel.panel.snapshot_id = "other-panel".into();
    let message = convert_detail(
        TemporalInitialStateArtifactWire::consume(&reseal(other_panel), &limits()).unwrap_err(),
    );
    assert!(message.contains("temporal_initial_state_artifact.identity"), "{message}");
}

#[test]
fn x5_initial_state_artifact_refuses_a_resealed_changed_panel_summary() {
    let mut tallies = artifact();
    tallies.panel.rows[0].y_sum += 5.0;
    let message = convert_detail(
        TemporalInitialStateArtifactWire::consume(&reseal(tallies), &limits()).unwrap_err(),
    );
    assert!(message.contains("temporal_initial_state_artifact.value"), "{message}");

    let mut counts = artifact();
    counts.panel.rows[0].n_covariate += 10;
    counts.panel.histories += 10;
    let message = convert_detail(
        TemporalInitialStateArtifactWire::consume(&reseal(counts), &limits()).unwrap_err(),
    );
    assert!(message.contains("temporal_initial_state_artifact.value"), "{message}");

    let mut unsorted = artifact();
    unsorted.panel.rows.reverse();
    let message = convert_detail(
        TemporalInitialStateArtifactWire::consume(&reseal(unsorted), &limits()).unwrap_err(),
    );
    assert!(message.contains("panel_summary_not_canonical"), "{message}");

    // A tightened consumer bound refuses an otherwise valid artifact.
    let small = TemporalInitialStateConsumeLimits { max_summary_rows: 1, max_units: 100_000 };
    let message = convert_detail(
        TemporalInitialStateArtifactWire::consume(&artifact().export().unwrap(), &small)
            .unwrap_err(),
    );
    assert!(message.contains("limits_exceeded"), "{message}");
}

#[test]
fn x5_initial_state_artifact_is_not_the_specified_initial_state_artifact() {
    assert_ne!(TEMPORAL_INITIAL_STATE_ARTIFACT_FEATURE, TEMPORAL_TRANSPORT_ARTIFACT_FEATURE);
    let bytes = artifact().export().unwrap();
    // The 2.2 sequence artifact decoder refuses these bytes outright.
    assert!(TemporalSequenceArtifactWire::decode(&bytes).is_err());
    // A foreign feature marker or kind is refused by this decoder.
    let mut foreign = artifact();
    foreign.required_features = vec![TEMPORAL_TRANSPORT_ARTIFACT_FEATURE.into()];
    let message = convert_detail(
        TemporalInitialStateArtifactWire::decode(&foreign.export().unwrap()).unwrap_err(),
    );
    assert!(message.contains("unsupported_semantics"), "{message}");
    let mut kind = artifact();
    kind.kind = "temporal_transport_sequence".into();
    assert!(TemporalInitialStateArtifactWire::decode(&kind.export().unwrap()).is_err());
    let mut version = artifact();
    version.version = 2;
    assert!(matches!(
        TemporalInitialStateArtifactWire::decode(&version.export().unwrap()),
        Err(IoError::UnsupportedVersion { version: 2 })
    ));
}

#[test]
fn x5_initial_state_artifact_is_not_built_for_a_source_law_or_a_support_gap() {
    let source = InitialStateLaw::new(
        InitialStatePopulation::Source,
        "source-state",
        vec![(0, 0.5), (1, 0.5)],
    )
    .unwrap();
    assert!(MarginalizedQuery::new(SEQUENCE, InitialStateSpec::Law(source)).is_err());

    let gap = InitialStateLaw::new(
        InitialStatePopulation::Target,
        "target-state",
        vec![(0, 0.5), (7, 0.5)],
    )
    .unwrap();
    let query = MarginalizedQuery::new(SEQUENCE, InitialStateSpec::Law(gap)).unwrap();
    let (code, message) = refused(
        TemporalInitialStateArtifactWire::checked(
            premises(),
            SEQUENCE,
            &query,
            None,
            &panel("state-panel"),
        )
        .unwrap_err(),
    );
    assert_eq!(code, "transport_support_failure");
    assert!(message.contains("initial_state.support_gap"), "{message}");

    let mut bad = premises();
    bad.time_order[0] = "a1".into();
    let query =
        MarginalizedQuery::new(SEQUENCE, InitialStateSpec::Law(target_law("target-state", 0.7)))
            .unwrap();
    let (code, message) = refused(
        TemporalInitialStateArtifactWire::checked(bad, SEQUENCE, &query, None, &panel("p"))
            .unwrap_err(),
    );
    assert_eq!(code, "invalid_argument");
    assert!(message.contains("initial_state.invalid_premises"), "{message}");
}
