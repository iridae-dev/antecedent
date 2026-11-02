//! Pending whole-method measurement for a fixed, treated-arm MNAR shift.
//!
//! Each independent arm has Bernoulli(0.5) observed-pattern outcomes, a 0.8
//! observation probability and a known one-unit missing-pattern shift in the
//! treated arm. Its full-population contrast is 1.2. The result is specific to
//! this data-generating process, grid and native delta-method construction.
//! It grants no authority until measured and reviewed.

#![cfg(feature = "calibration-internal")]

#[path = "common/calibration.rs"]
mod calibration;
#[path = "../../antecedent-estimate/tests/common/candidate_calibration.rs"]
mod candidate;

use antecedent_validate::mnar_sensitivity::{MnarData, mnar_shift_delta_calibration_candidate};
use calibration::{CoverageTally, RecordKey, grid_n, grid_seed, map_replicates, n_sim, uniform};

#[test]
#[ignore = "calibration: run through scripts/gate_calibration.sh"]
fn mnar_shift_delta_interval() {
    let n = grid_n(200);
    let mut tally = CoverageTally::for_record(
        RecordKey {
            test: "mnar_shift_delta_interval",
            dgp: "independent_bernoulli_observed_patterns_fixed_shift_one",
            interval: "delta_normal",
        },
        0.95,
    );
    let samples = map_replicates(n_sim(), |replicate| {
        let seed = grid_seed(0x6d6e_0000 + replicate);
        let mut observed = uniform(seed, 11);
        let mut outcomes = uniform(seed, 17);
        let mut treatment = Vec::with_capacity(2 * n);
        let mut outcome = Vec::with_capacity(2 * n);
        for arm in [0_u8, 1_u8] {
            for _ in 0..n {
                treatment.push(f64::from(arm));
                let y = f64::from(arm) + f64::from(u8::from(outcomes() < 0.5));
                outcome.push((observed() < 0.8).then_some(y));
            }
        }
        mnar_shift_delta_calibration_candidate(&MnarData::new(&treatment, &outcome)).ok()
    });
    for sample in samples {
        let Some((interval, basis)) = sample else {
            tally.skip();
            continue;
        };
        candidate::bind(&mut tally, &basis);
        tally.record(Some((interval.lower, interval.upper)), 1.2);
    }
    assert_eq!(
        tally.record_id().as_deref(),
        Some(
            "cov.average_effect.not_applicable.frequentist.delta_normal.l95.mnar_shift_delta_interval.v2_validation_not_applicable"
        )
    );
    tally.assert();
}
