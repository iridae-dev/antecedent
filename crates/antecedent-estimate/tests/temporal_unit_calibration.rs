//! Frozen whole-unit two-step candidate design. Every independent unit has all
//! 16 (state, first action, intermediate state, second action) histories. A shared
//! unit shock and independent history errors induce genuine within-unit dependence.
//! Target initial-state probabilities are supplied, fixed and exact. Both actual
//! bootstrap constructions are measured separately; ordinary tests never measure.
#![allow(clippy::cast_precision_loss, reason = "bounded simulated unit/history indices")]
#[path = "../../antecedent/tests/common/calibration.rs"]
mod calibration;
#[path = "common/candidate_calibration.rs"]
mod candidate;
use antecedent_core::ExecutionContext;
use antecedent_estimate::temporal_dependent_interval::{
    DependentIntervalConfig, IntervalMethod, SequenceHistory, TemporalUnitPanel, UnitHistories,
    dependent_unit_interval,
};
use antecedent_estimate::temporal_initial_state::{
    InitialStateLaw, InitialStatePopulation, InitialStateSpec, MarginalizedQuery,
};
use calibration::{CoverageTally, RecordKey, grid_n, grid_seed, map_replicates, n_sim};

fn panel(n: usize, seed: u64) -> TemporalUnitPanel {
    let mut rng = candidate::Generator::new(seed);
    let mut units = Vec::with_capacity(n);
    for unit in 0..n {
        let shock = if rng.binary(0.5) == 1 { 0.2 } else { -0.2 };
        let mut histories = Vec::with_capacity(16);
        for s0 in 0..2 {
            for a1 in 0..2 {
                for l2 in 0..2 {
                    for a2 in 0..2 {
                        // Under do(0,0), state means .35 and .55, marginalized to .49.
                        let base = 0.3
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
                            y: base + shock + 0.1 * rng.normal(),
                        });
                    }
                }
            }
        }
        units.push(UnitHistories { unit_id: u64::try_from(unit).unwrap(), histories });
    }
    TemporalUnitPanel::new(format!("independent-unit-scm-{seed}"), Some(units)).unwrap()
}
fn measure(test: &'static str, expected_id: &str, method: IntervalMethod) {
    let n = grid_n(100);
    let law = InitialStateLaw::new(
        InitialStatePopulation::Target,
        "fixed_target",
        vec![(0, 0.3), (1, 0.7)],
    )
    .unwrap();
    let query = MarginalizedQuery::new([0, 0], InitialStateSpec::Law(law)).unwrap();
    let interval_name = match method {
        IntervalMethod::Percentile => "bootstrap_percentile",
        IntervalMethod::Basic => "bootstrap_basic",
        IntervalMethod::Studentized => "bootstrap_studentized",
    };
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test,
            dgp: "independent_units_shared_shock_fixed_target",
            interval: interval_name,
        },
        0.95,
    );
    let results = map_replicates(n_sim(), |rep| {
        let seed = grid_seed(0x45ab_0000 + rep);
        dependent_unit_interval(
            &panel(n, seed),
            &antecedent_estimate::temporal_dependent_interval::BalancedTemporalEstimator::Marginalized(query.clone()),
            &DependentIntervalConfig {
                replicates: 500,
                seed: seed + 100_000,
                method,
                ..DependentIntervalConfig::default()
            },
            &ExecutionContext::for_tests(seed),
        )
    });
    for result in results {
        match result {
            Ok(result) => {
                candidate::bind(&mut tally, &result.calibration_basis());
                tally.record(Some((result.lower, result.upper)), 0.49);
            }
            Err(_) => tally.skip(),
        }
    }
    assert_eq!(tally.record_id().as_deref(), Some(expected_id));
    tally.assert();
}
#[test]
#[ignore = "calibration: final measurement only"]
fn temporal_two_step_units_percentile_l95() {
    measure(
        "temporal_two_step_units_percentile_l95",
        "cov.temporal_transport.selection_admg.frequentist.bootstrap_percentile.l95.temporal_two_step_units_percentile_l95",
        IntervalMethod::Percentile,
    );
}
#[test]
#[ignore = "calibration: final measurement only"]
fn temporal_two_step_units_basic_l95() {
    measure(
        "temporal_two_step_units_basic_l95",
        "cov.temporal_transport.selection_admg.frequentist.bootstrap_basic.l95.temporal_two_step_units_basic_l95",
        IntervalMethod::Basic,
    );
}

#[test]
#[ignore = "calibration: final measurement only"]
fn temporal_two_step_units_studentized_l95() {
    measure(
        "temporal_two_step_units_studentized_l95",
        "cov.temporal_transport.selection_admg.frequentist.bootstrap_studentized.l95.temporal_two_step_units_studentized_l95",
        IntervalMethod::Studentized,
    );
}
