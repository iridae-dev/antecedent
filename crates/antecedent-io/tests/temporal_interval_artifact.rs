//! 2.3A X5 `dependent_temporal_interval`: the internal interval artifact replays the whole
//! unit resampling from its embedded panel and refuses every resealed mutation of a unit,
//! time id, value, snapshot, design, estimator or stored replicate. The public interval
//! route stays closed; this artifact is internal and records calibration `unmeasured`.
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_wrap,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "enumerated binary SCM fixtures use bounded nonnegative counts and indices"
)]

use std::collections::BTreeSet;

use antecedent_core::ExecutionContext;
use antecedent_estimate::temporal_dependent_interval::{
    DependentIntervalConfig, INTERVAL_CALIBRATION_STATUS, INTERVAL_CLAIM, IntervalMethod,
    ObservedStateSequence, SequenceHistory, TemporalEstimator, TemporalUnitPanel, UnitHistories,
    dependent_unit_interval,
};
use antecedent_estimate::temporal_initial_state::{
    InitialStateLaw, InitialStatePopulation, InitialStateSpec, MarginalizedQuery,
};
use antecedent_estimate::temporal_refresh::{
    HeldTemporalResult, ObservationPeriod, TemporalWindowIdentity, refresh_held,
};
use antecedent_io::IoError;
use antecedent_io::temporal_interval_artifact::{
    IntervalEstimatorWire, TemporalIntervalArtifactWire, TemporalIntervalConsumeLimits,
};
use antecedent_io::temporal_refresh_artifact::ReceiptWire;

const SEQUENCE: [u32; 2] = [0, 0];
const SNAPSHOT: &str = "interval-panel";

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn limits() -> TemporalIntervalConsumeLimits {
    TemporalIntervalConsumeLimits::default()
}

/// 24 units; every unit holds all four `(s0, l2)` cells under both second actions, with an
/// outcome that depends on the unit, so units are dependent and every resample is supported.
fn panel() -> TemporalUnitPanel {
    let mut units = Vec::new();
    for unit in 0..24_u64 {
        let mut histories = Vec::new();
        for s0 in 0..2_u32 {
            for l2 in 0..2_u32 {
                for a2 in 0..2_u32 {
                    let level =
                        (unit * 7 + u64::from(s0) * 3 + u64::from(l2) * 5 + u64::from(a2)) % 10;
                    for a1 in 0..2 {
                        histories.push(SequenceHistory {
                            time_id: histories.len() as u64,
                            s0,
                            a1,
                            l2,
                            a2,
                            y: level as f64 / 10.0,
                        });
                    }
                }
            }
        }
        units.push(UnitHistories { unit_id: unit, histories });
    }
    TemporalUnitPanel::new(SNAPSHOT, Some(units)).unwrap()
}

fn config() -> DependentIntervalConfig {
    DependentIntervalConfig { replicates: 40, seed: 7, ..DependentIntervalConfig::default() }
}

fn marginalized() -> MarginalizedQuery {
    let law = InitialStateLaw::new(
        InitialStatePopulation::Target,
        "target-state",
        vec![(0, 0.3), (1, 0.7)],
    )
    .unwrap();
    MarginalizedQuery::new(SEQUENCE, InitialStateSpec::Law(law)).unwrap()
}

fn artifact_for(
    _estimator: &dyn TemporalEstimator,
    wire: IntervalEstimatorWire,
) -> TemporalIntervalArtifactWire {
    let p = panel();
    let estimator = wire.to_estimator().unwrap();
    let interval = dependent_unit_interval(&p, &*estimator, &config(), &ctx()).unwrap();
    TemporalIntervalArtifactWire::checked(&p, wire, &config(), &interval, None, &ctx()).unwrap()
}

fn artifact() -> TemporalIntervalArtifactWire {
    artifact_for(
        &ObservedStateSequence { sequence: SEQUENCE },
        IntervalEstimatorWire::observed(SEQUENCE),
    )
}

fn reseal(mut wire: TemporalIntervalArtifactWire) -> Vec<u8> {
    wire.seal = wire.compute_seal().unwrap();
    wire.export().unwrap()
}

fn consume(bytes: &[u8]) -> IoError {
    TemporalIntervalArtifactWire::consume(bytes, &limits(), &ctx()).unwrap_err()
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
fn x5_dependent_interval_artifact_replays_the_whole_resampling() {
    let p = panel();
    let estimator =
        antecedent_estimate::temporal_dependent_interval::BalancedTemporalEstimator::Observed(
            ObservedStateSequence { sequence: SEQUENCE },
        );
    let interval = dependent_unit_interval(&p, &estimator, &config(), &ctx()).unwrap();
    let wire = artifact();
    let (decoded, replayed) =
        TemporalIntervalArtifactWire::consume(&wire.export().unwrap(), &limits(), &ctx()).unwrap();
    assert_eq!(replayed, interval);
    assert_eq!(decoded.result.replicates.len(), 40);
    assert_eq!(decoded.result.calibration, "unmeasured");
    assert_eq!(decoded.result.calibration, INTERVAL_CALIBRATION_STATUS);
    assert_eq!(decoded.result.claim, INTERVAL_CLAIM);
    assert_eq!(decoded.result.units, 24);
    assert_eq!(decoded.result.snapshot_id, SNAPSHOT);
    assert!(decoded.refresh_receipt.is_none());
    // Whole-unit replicate ids and unit-selection digests are recorded and distinct.
    let ids = decoded.result.replicates.iter().map(|r| r.replicate_id).collect::<BTreeSet<_>>();
    let selections =
        decoded.result.replicates.iter().map(|r| r.selection_digest).collect::<BTreeSet<_>>();
    assert_eq!(ids.len(), 40);
    assert_eq!(selections.len(), 40);
    assert!(decoded.result.lower < decoded.result.upper);
    assert_eq!(decoded.result.estimand, "observed_initial_state");
}

#[test]
fn x5_dependent_interval_artifact_replays_the_marginalized_estimator() {
    let query = marginalized();
    let wire = artifact_for(&query, IntervalEstimatorWire::marginalized(SEQUENCE, &query));
    assert_eq!(wire.result.estimand, "marginalized_initial_state");
    let (_, replayed) =
        TemporalIntervalArtifactWire::consume(&wire.export().unwrap(), &limits(), &ctx()).unwrap();
    assert_eq!(replayed.estimand, "marginalized_initial_state");
    assert_eq!(replayed.calibration, "unmeasured");
}

#[test]
fn x5_dependent_interval_reseal_refuses_unsealed_edits() {
    let mut wire = artifact();
    wire.result.lower -= 0.01;
    let message = convert_detail(consume(&wire.export().unwrap()));
    assert!(message.contains("temporal_interval_artifact.seal"), "{message}");
}

#[test]
fn x5_dependent_interval_reseal_refuses_a_changed_unit_time_value_or_snapshot() {
    let mut unit = artifact();
    unit.panel.units[0].unit_id = 99;
    let message = convert_detail(consume(&reseal(unit)));
    assert!(message.contains("temporal_interval_artifact.interval_replay"), "{message}");

    let mut time = artifact();
    let last = time.panel.units[3].histories.len() - 1;
    time.panel.units[3].histories[last].time_id += 100;
    let message = convert_detail(consume(&reseal(time)));
    assert!(message.contains("temporal_interval_artifact.interval_replay"), "{message}");

    let mut value = artifact();
    value.panel.units[5].histories[2].y = 0.95;
    let message = convert_detail(consume(&reseal(value)));
    assert!(message.contains("temporal_interval_artifact.interval_replay"), "{message}");

    let mut snapshot = artifact();
    snapshot.panel.snapshot_id = "other-panel".into();
    let message = convert_detail(consume(&reseal(snapshot)));
    assert!(message.contains("temporal_interval_artifact.interval_replay"), "{message}");

    // Dropping a unit changes the resampling population.
    let mut dropped = artifact();
    dropped.panel.units.pop();
    let message = convert_detail(consume(&reseal(dropped)));
    assert!(message.contains("temporal_interval_artifact.interval_replay"), "{message}");
}

#[test]
fn x5_dependent_interval_reseal_refuses_ill_formed_units_through_the_core() {
    let mut duplicate = artifact();
    duplicate.panel.units[1].unit_id = duplicate.panel.units[0].unit_id;
    let (code, message) = refused(consume(&reseal(duplicate)));
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("temporal_interval.unknown_units"), "{message}");

    let mut unordered = artifact();
    unordered.panel.units[0].histories[1].time_id = 0;
    let (code, message) = refused(consume(&reseal(unordered)));
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("temporal_interval.unknown_units"), "{message}");

    let mut few = artifact();
    few.panel.units.truncate(5);
    let (code, message) = refused(consume(&reseal(few)));
    assert_eq!(code, "too_few_clusters");
    assert!(message.contains("temporal_interval.too_few_units"), "{message}");
}

#[test]
fn x5_dependent_interval_reseal_refuses_a_changed_design_estimator_or_stored_replicate() {
    let mut seed = artifact();
    seed.config.seed += 1;
    let message = convert_detail(consume(&reseal(seed)));
    assert!(message.contains("temporal_interval_artifact.interval_replay"), "{message}");

    let mut digest = artifact();
    digest.result.replicates[3].selection_digest ^= 1;
    let message = convert_detail(consume(&reseal(digest)));
    assert!(message.contains("temporal_interval_artifact.interval_replay"), "{message}");

    let mut point = artifact();
    point.result.point += 0.001;
    let message = convert_detail(consume(&reseal(point)));
    assert!(message.contains("temporal_interval_artifact.interval_replay"), "{message}");

    let mut estimator = artifact();
    estimator.estimator.kind = "fixed_initial_state".into();
    let message = convert_detail(consume(&reseal(estimator)));
    assert!(message.contains("temporal_interval_artifact.estimator"), "{message}");

    let mut method = artifact();
    method.config.method = "bca".into();
    let message = convert_detail(consume(&reseal(method)));
    assert!(message.contains("temporal_interval_artifact.config"), "{message}");

    let mut calibration = artifact();
    calibration.result.calibration = "calibrated".into();
    let message = convert_detail(consume(&reseal(calibration)));
    assert!(message.contains("temporal_interval_artifact.calibration"), "{message}");

    let mut claim = artifact();
    claim.result.claim = "calibrated_dependent_interval".into();
    let message = convert_detail(consume(&reseal(claim)));
    assert!(message.contains("temporal_interval_artifact.calibration"), "{message}");
}

#[test]
fn x5_dependent_interval_reseal_refuses_a_source_law_and_over_limit_replays() {
    let query = marginalized();
    let mut source = artifact_for(&query, IntervalEstimatorWire::marginalized(SEQUENCE, &query));
    source.estimator.law.as_mut().unwrap().population = "source".into();
    let (code, message) = refused(consume(&reseal(source)));
    assert_eq!(code, "transport_missing_evidence");
    assert!(message.contains("initial_state.target_law_missing"), "{message}");

    let small = TemporalIntervalConsumeLimits { max_replicates: 10, ..limits() };
    let bytes = artifact().export().unwrap();
    let message =
        convert_detail(TemporalIntervalArtifactWire::consume(&bytes, &small, &ctx()).unwrap_err());
    assert!(message.contains("limits_exceeded"), "{message}");
}

fn identity(snapshot: &str, period: (i64, i64)) -> TemporalWindowIdentity {
    TemporalWindowIdentity {
        graph_id: "g".into(),
        horizon: 2,
        lag_alignment: vec![("s0".into(), 1)],
        intervention_history: vec!["a1=0".into(), "a2=0".into()],
        selection_targets: BTreeSet::from(["s_t1".to_owned()]),
        regimes: BTreeSet::from(["source".to_owned(), "target".to_owned()]),
        period: ObservationPeriod { start: period.0, end: period.1 },
        unit_ids: BTreeSet::from(["0".to_owned()]),
        snapshot_id: snapshot.into(),
        proof_id: "p".into(),
    }
}

#[test]
fn x5_dependent_interval_artifact_links_the_refresh_that_produced_its_panel() {
    let held = HeldTemporalResult {
        identity: identity("interval-panel-old", (0, 10)),
        value: 0.0_f64,
        interval_present: true,
    };
    let (_, receipt) = refresh_held(&held, identity(SNAPSHOT, (10, 20)), |_| Ok(1.0)).unwrap();
    let p = panel();
    let estimator =
        antecedent_estimate::temporal_dependent_interval::BalancedTemporalEstimator::Observed(
            ObservedStateSequence { sequence: SEQUENCE },
        );
    let interval = dependent_unit_interval(&p, &estimator, &config(), &ctx()).unwrap();
    let wire = TemporalIntervalArtifactWire::checked(
        &p,
        IntervalEstimatorWire::observed(SEQUENCE),
        &config(),
        &interval,
        Some(&receipt),
        &ctx(),
    )
    .unwrap();
    assert!(wire.refresh_receipt.as_ref().unwrap().interval_invalidated);
    TemporalIntervalArtifactWire::consume(&wire.export().unwrap(), &limits(), &ctx()).unwrap();

    // A receipt for another snapshot is not this panel's refresh.
    let mut elsewhere = wire.clone();
    let mut foreign = receipt.clone();
    foreign.new_snapshot_id = "elsewhere".into();
    foreign.digest = foreign.compute_digest();
    elsewhere.refresh_receipt = Some(ReceiptWire::from_receipt(&foreign));
    let message = convert_detail(consume(&reseal(elsewhere)));
    assert!(message.contains("temporal_interval_artifact.refresh_link"), "{message}");

    // A receipt whose seal does not match its fields is refused.
    let mut forged = wire;
    forged.refresh_receipt.as_mut().unwrap().proof_id = "forged".into();
    let (code, message) = refused(consume(&reseal(forged)));
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("temporal_refresh.receipt_digest_mismatch"), "{message}");
}

#[test]
fn x5_dependent_interval_artifact_refuses_an_interval_of_another_panel() {
    let p = panel();
    let estimator =
        antecedent_estimate::temporal_dependent_interval::BalancedTemporalEstimator::Observed(
            ObservedStateSequence { sequence: SEQUENCE },
        );
    let interval = dependent_unit_interval(&p, &estimator, &config(), &ctx()).unwrap();
    let mut units = p.units().to_vec();
    units.pop();
    let other = TemporalUnitPanel::new(SNAPSHOT, Some(units)).unwrap();
    let message = convert_detail(
        TemporalIntervalArtifactWire::checked(
            &other,
            IntervalEstimatorWire::observed(SEQUENCE),
            &config(),
            &interval,
            None,
            &ctx(),
        )
        .unwrap_err(),
    );
    assert!(message.contains("temporal_interval_artifact.foreign_interval"), "{message}");
}

#[test]
fn studentized_artifact_replays_original_variance_and_rejects_resealed_pivot_mutation() {
    let panel = panel();
    let config = DependentIntervalConfig { method: IntervalMethod::Studentized, ..config() };
    let estimator = marginalized();
    let result = dependent_unit_interval(
        &panel,
        &antecedent_estimate::temporal_dependent_interval::BalancedTemporalEstimator::Marginalized(
            estimator.clone(),
        ),
        &config,
        &ctx(),
    )
    .unwrap();
    let mut wire = TemporalIntervalArtifactWire::checked(
        &panel,
        IntervalEstimatorWire::marginalized(SEQUENCE, &estimator),
        &config,
        &result,
        None,
        &ctx(),
    )
    .unwrap();
    let (_, consumed) =
        TemporalIntervalArtifactWire::consume(&wire.export().unwrap(), &limits(), &ctx()).unwrap();
    assert_eq!(consumed, result);
    wire.result.studentization.as_mut().unwrap().pivots[0] = Some(123.0);
    assert!(TemporalIntervalArtifactWire::consume(&reseal(wire), &limits(), &ctx()).is_err());
}

#[test]
fn direct_artifact_rejects_resealed_retired_methods() {
    for method in ["percentile", "basic"] {
        let mut wire = artifact();
        wire.config.method = method.into();
        wire.result.method = method.into();
        assert!(convert_detail(consume(&reseal(wire))).contains("retired_method"));
    }
}
