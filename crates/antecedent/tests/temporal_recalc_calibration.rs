//! Whole-method coverage candidates for actual checked temporal adapter execution.
//! Independent units each have every binary S0/A1/L2/A2 history. A shared ±.1
//! latent unit shock induces dependence; outcomes are Bernoulli with means
//! .3+.1L2+.1S0+.2S0L2+.05(A1+A2)+shock. Target P(S0=1)=.7 is fixed.
//! Original checked identification/fit precedes actual whole-unit resample/refit.
//! Public inference remains closed; these ignored measurements run only at final cut.
#![allow(clippy::cast_precision_loss, reason = "bounded unit/history simulation")]
#[path = "common/calibration.rs"]
mod calibration;
#[path = "../../antecedent-estimate/tests/common/candidate_calibration.rs"]
mod candidate;
use antecedent::analysis::recalc_temporal::{
    TemporalFunctional, TemporalHistoryWire, TemporalRequest, TemporalSession, TemporalUnitWire,
    execute_temporal_with_receipt,
};
use antecedent_core::ExecutionContext;
use antecedent_estimate::temporal_dependent_interval::{DependentIntervalConfig, IntervalMethod};
use calibration::{CoverageTally, RecordKey, grid_n, grid_seed, map_replicates, n_sim};

fn request(n: usize, seed: u64, functional: TemporalFunctional) -> TemporalRequest {
    let mut rng = candidate::Generator::new(seed);
    let mut units = Vec::with_capacity(n);
    for unit in 0..n {
        let shock = if rng.binary(0.5) == 1 { 0.1 } else { -0.1 };
        let mut histories = Vec::with_capacity(16);
        for s0 in 0..2 {
            for a1 in 0..2 {
                for l2 in 0..2 {
                    for a2 in 0..2 {
                        let probability = 0.3
                            + 0.1 * f64::from(l2)
                            + 0.1 * f64::from(s0)
                            + 0.2 * f64::from(s0 * l2)
                            + 0.05 * f64::from(a1 + a2)
                            + shock;
                        histories.push(TemporalHistoryWire {
                            time_id: 3 * histories.len() as u64,
                            s0,
                            a1,
                            l2,
                            a2,
                            y: rng.binary(probability) as f64,
                        });
                    }
                }
            }
        }
        units.push(TemporalUnitWire { unit_id: unit as u64, histories });
    }
    TemporalRequest {
        edges: vec![(0, 1), (0, 2), (0, 3), (0, 4), (1, 2), (1, 3), (1, 4), (2, 3), (2, 4), (3, 4)],
        bidirected: vec![],
        selection_targets: vec![0],
        horizon: 2,
        lag_alignment: [0, 1, 2, 2, 2],
        period: (0, 49),
        units,
        snapshot_id: format!("independent_binary_temporal_units:{seed}"),
        initial_state: [0.3, 0.7],
        initial_state_id: "fixed_target_initial_law".into(),
        functional,
        benefit_per_unit: 1.,
        cost: 0.,
    }
}
fn measure(
    test: &'static str,
    id: &str,
    functional: &TemporalFunctional,
    method: IntervalMethod,
    truth: f64,
) {
    let n = grid_n(128);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test,
            dgp: "independent_units_binary_shared_shock_checked_adapter",
            interval: match method {
                IntervalMethod::Percentile => "bootstrap_percentile",
                IntervalMethod::Basic => "bootstrap_basic",
                IntervalMethod::Studentized => "bootstrap_studentized",
            },
        },
        0.95,
    );
    let results = map_replicates(n_sim(), |rep| {
        let seed = grid_seed(0x317b_0000 + rep);
        let context = ExecutionContext::for_tests(seed);
        let mut session = TemporalSession::new();
        execute_temporal_with_receipt(
            &mut session,
            &request(n, seed, functional.clone()),
            &context,
        )?;
        session.candidate_interval_internal(
            &DependentIntervalConfig {
                replicates: 500,
                seed: seed + 100_000,
                method,
                ..DependentIntervalConfig::default()
            },
            &context,
        )
    });
    for result in results {
        match result {
            Ok(interval) => {
                candidate::bind(&mut tally, &interval.calibration_basis());
                tally.record(Some((interval.lower, interval.upper)), truth);
            }
            Err(_) => tally.skip(),
        }
    }
    assert_eq!(tally.record_id().as_deref(), Some(id));
    tally.assert();
}
#[test]
#[ignore = "calibration: final measurement only"]
fn temporal_checked_response_percentile_l95() {
    measure(
        "temporal_checked_response_percentile_l95",
        "cov.temporal_transport.selection_admg.frequentist.bootstrap_percentile.l95.temporal_checked_response_percentile_l95",
        &TemporalFunctional::Response { sequence: [0, 0] },
        IntervalMethod::Percentile,
        0.49,
    );
}
#[test]
#[ignore = "calibration: final measurement only"]
fn temporal_checked_response_basic_l95() {
    measure(
        "temporal_checked_response_basic_l95",
        "cov.temporal_transport.selection_admg.frequentist.bootstrap_basic.l95.temporal_checked_response_basic_l95",
        &TemporalFunctional::Response { sequence: [0, 0] },
        IntervalMethod::Basic,
        0.49,
    );
}
#[test]
#[ignore = "calibration: final measurement only"]
fn temporal_checked_effect_percentile_l95() {
    measure(
        "temporal_checked_effect_percentile_l95",
        "cov.temporal_transport.selection_admg.frequentist.bootstrap_percentile.l95.temporal_checked_effect_percentile_l95",
        &TemporalFunctional::Effect { active: [1, 1], control: [0, 0] },
        IntervalMethod::Percentile,
        0.1,
    );
}
#[test]
#[ignore = "calibration: final measurement only"]
fn temporal_checked_effect_basic_l95() {
    measure(
        "temporal_checked_effect_basic_l95",
        "cov.temporal_transport.selection_admg.frequentist.bootstrap_basic.l95.temporal_checked_effect_basic_l95",
        &TemporalFunctional::Effect { active: [1, 1], control: [0, 0] },
        IntervalMethod::Basic,
        0.1,
    );
}

#[test]
fn checked_temporal_calibration_oracle_matches_enumerated_scm() {
    // Quadrature of the same marginal SCM probabilities, not a measurement run.
    std::thread::Builder::new()
        .stack_size(8 * 1024 * 1024)
        .spawn(|| {
            let mut input = request(1, 17, TemporalFunctional::Response { sequence: [0, 0] });
            input.units.clear();
            for shock in [-0.1, 0.1] {
                for noise in 0..20 {
                    let mut histories = Vec::new();
                    for s0 in 0..2 {
                        for a1 in 0..2 {
                            for l2 in 0..2 {
                                for a2 in 0..2 {
                                    let probability = 0.3
                                        + 0.1 * f64::from(l2)
                                        + 0.1 * f64::from(s0)
                                        + 0.2 * f64::from(s0 * l2)
                                        + 0.05 * f64::from(a1 + a2)
                                        + shock;
                                    let y = if (f64::from(noise) + 0.5) / 20. < probability {
                                        1.
                                    } else {
                                        0.
                                    };
                                    histories.push(TemporalHistoryWire {
                                        time_id: 3 * histories.len() as u64,
                                        s0,
                                        a1,
                                        l2,
                                        a2,
                                        y,
                                    });
                                }
                            }
                        }
                    }
                    input
                        .units
                        .push(TemporalUnitWire { unit_id: input.units.len() as u64, histories });
                }
            }
            for (functional, expected) in [
                (TemporalFunctional::Response { sequence: [0, 0] }, 0.49),
                (TemporalFunctional::Response { sequence: [1, 1] }, 0.59),
                (TemporalFunctional::Effect { active: [1, 1], control: [0, 0] }, 0.1),
            ] {
                input.functional = functional;
                let value = execute_temporal_with_receipt(
                    &mut TemporalSession::new(),
                    &input,
                    &ExecutionContext::for_tests(17),
                )
                .unwrap();
                assert!((value.recalc.law.ate - expected).abs() < 1e-12);
            }
        })
        .unwrap()
        .join()
        .unwrap();
}

#[test]
#[ignore = "calibration: final measurement only"]
fn temporal_checked_response_studentized_l95() {
    measure(
        "temporal_checked_response_studentized_l95",
        "cov.temporal_transport.selection_admg.frequentist.bootstrap_studentized.l95.temporal_checked_response_studentized_l95",
        &TemporalFunctional::Response { sequence: [0, 0] },
        IntervalMethod::Studentized,
        0.49,
    );
}
#[test]
#[ignore = "calibration: final measurement only"]
fn temporal_checked_effect_studentized_l95() {
    measure(
        "temporal_checked_effect_studentized_l95",
        "cov.temporal_transport.selection_admg.frequentist.bootstrap_studentized.l95.temporal_checked_effect_studentized_l95",
        &TemporalFunctional::Effect { active: [1, 1], control: [0, 0] },
        IntervalMethod::Studentized,
        0.1,
    );
}

#[test]
fn checked_temporal_studentized_unit_scores_match_independent_paired_scm_algebra() {
    let ctx = ExecutionContext::for_tests(179);
    for functional in [
        TemporalFunctional::Response { sequence: [0, 0] },
        TemporalFunctional::Effect { active: [1, 1], control: [0, 0] },
    ] {
        let input = request(24, 179, functional.clone());
        let sequence_score = |unit: &TemporalUnitWire, seq: [u32; 2]| {
            unit.histories
                .iter()
                .filter(|h| h.a1 == seq[0] && h.a2 == seq[1])
                .map(|h| h.y * if h.s0 == 0 { 0.15 } else { 0.35 })
                .sum::<f64>()
        };
        let scores = input
            .units
            .iter()
            .map(|u| match functional {
                TemporalFunctional::Response { sequence } => sequence_score(u, sequence),
                TemporalFunctional::Effect { active, control } => {
                    sequence_score(u, active) - sequence_score(u, control)
                }
            })
            .collect::<Vec<_>>();
        let mean = scores.iter().sum::<f64>() / 24.0;
        let se = (scores.iter().map(|v| (v - mean).powi(2)).sum::<f64>() / (23.0 * 24.0)).sqrt();
        let mut session = TemporalSession::new();
        execute_temporal_with_receipt(&mut session, &input, &ctx).unwrap();
        let result = session
            .candidate_interval_internal(
                &DependentIntervalConfig {
                    method: IntervalMethod::Studentized,
                    replicates: 100,
                    seed: 137,
                    ..DependentIntervalConfig::default()
                },
                &ctx,
            )
            .unwrap();
        let receipt = result.studentization.as_ref().unwrap();
        assert!((result.point - mean).abs() < 1e-12);
        assert!((receipt.standard_error - se).abs() < 1e-12);
        for (actual, expected) in receipt.unit_scores.iter().zip(scores) {
            assert!((actual - expected).abs() < 1e-12);
        }
    }
}
