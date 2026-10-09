//! Internal preparation of the real facade producer and portable independent consumer.
//! Neither test measures coverage nor activates a released interval route.
#![cfg(feature = "calibration-internal")]

use antecedent::{IntervalEstimand, temporal_dependent_interval_candidate};
use antecedent_core::ExecutionContext;
use antecedent_estimate::temporal_dependent_interval::{
    DependentIntervalConfig, IntervalMethod, SequenceHistory, UnitHistories,
};
use antecedent_estimate::temporal_initial_state::{
    InitialStateLaw, InitialStatePopulation, InitialStateSpec,
};
use antecedent_io::temporal_interval_artifact::{
    TemporalIntervalArtifactWire, TemporalIntervalConsumeLimits,
};

fn panel() -> Vec<UnitHistories> {
    (0..100_u64)
        .map(|unit| {
            let mut histories = Vec::new();
            let shock = if unit % 2 == 0 { -0.2 } else { 0.2 };
            for s0 in 0..2_u32 {
                for a1 in 0..2_u32 {
                    for l2 in 0..2_u32 {
                        for a2 in 0..2_u32 {
                            let mean = 0.3
                                + 0.1 * f64::from(l2)
                                + 0.1 * f64::from(s0)
                                + 0.2 * f64::from(s0 * l2)
                                + 0.05 * f64::from(a1 + a2);
                            histories.push(SequenceHistory {
                                time_id: u64::try_from(histories.len()).unwrap(),
                                s0,
                                a1,
                                l2,
                                a2,
                                y: mean + shock,
                            });
                        }
                    }
                }
            }
            UnitHistories { unit_id: unit, histories }
        })
        .collect()
}
fn prepare(method: IntervalMethod) -> TemporalIntervalArtifactWire {
    let law = InitialStateLaw::new(
        InitialStatePopulation::Target,
        "fixed_target",
        vec![(0, 0.3), (1, 0.7)],
    )
    .unwrap();
    temporal_dependent_interval_candidate(
        "activation-unit-panel",
        Some(panel()),
        [0, 0],
        IntervalEstimand::Marginalized(InitialStateSpec::Law(law)),
        &DependentIntervalConfig {
            replicates: 500,
            seed: 901,
            method,
            ..DependentIntervalConfig::default()
        },
        &ExecutionContext::for_tests(901),
    )
    .unwrap()
}

#[test]
fn temporal_actual_facade_candidate_and_independent_consumer_preserve_frozen_coordinates() {
    for method in [IntervalMethod::Percentile, IntervalMethod::Basic] {
        let wire = prepare(method);
        // Independent SCM integration: .3*.35 + .7*.55 = .49; unit shocks sum to zero.
        assert!((wire.result.point - 0.49).abs() < 1e-12);
        assert_eq!(wire.result.calibration, "unmeasured");
        assert_eq!(wire.result.claim, "dependence_preserving_calibration_unmeasured");
        assert_eq!(wire.result.replicates.len(), 500);
        assert_eq!(wire.result.failed, 0);
        let bytes = wire.export().unwrap();
        let (fresh, interval) = TemporalIntervalArtifactWire::consume(
            &bytes,
            &TemporalIntervalConsumeLimits::default(),
            &ExecutionContext::for_tests(901),
        )
        .unwrap();
        assert_eq!(fresh, wire);
        assert!((interval.point - 0.49).abs() < 1e-12);
    }
}

#[test]
fn temporal_facade_candidate_replay_refuses_resealed_panel_estimator_and_seed_mutations() {
    let original = prepare(IntervalMethod::Percentile);
    for field in 0..6 {
        let mut changed = original.clone();
        match field {
            0 => changed.panel.snapshot_id.push_str("-changed"),
            1 => changed.panel.units[0].unit_id = 10_000,
            2 => changed.panel.units[0].histories[0].time_id = 90_000,
            3 => changed.panel.units[0].histories[0].y += 0.1,
            4 => changed.estimator.sequence = [1, 1],
            _ => changed.config.seed += 1,
        }
        changed.seal = changed.compute_seal().unwrap();
        assert!(
            TemporalIntervalArtifactWire::consume(
                &changed.export().unwrap(),
                &TemporalIntervalConsumeLimits::default(),
                &ExecutionContext::for_tests(901),
            )
            .is_err()
        );
    }
}
