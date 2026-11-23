//! Whole-method measurement of the declared-shift MNAR delta interval over its scope.
//!
//! One record is measured per cell of the declared shift (in pooled standard
//! deviations) by the largest arm missingness, at three row counts from 50 to
//! 10000. Each replicate draws an arbitrary treatment effect, allocation between
//! 30% and 70% treated, a shift and missingness uniformly inside the cell, and
//! observed-pattern outcomes that are Gaussian or centred exponential with unit
//! standard deviation. Missing treated outcomes are shifted by the declared amount
//! and missing control outcomes are not. The truth is the full-population contrast.
//! A replicate whose realized cell differs from the one measured is redrawn, so
//! each record describes the executions the runtime would key into that cell.

#![cfg(feature = "calibration-internal")]
#![allow(
    clippy::cast_possible_truncation,
    clippy::cast_precision_loss,
    clippy::cast_sign_loss,
    reason = "bands, row counts and shares are small and non-negative"
)]

#[path = "common/calibration.rs"]
mod calibration;
#[path = "../../antecedent-estimate/tests/common/candidate_calibration.rs"]
mod candidate;

use antecedent::Study;
use antecedent::analysis::mnar_fixed_shift::consume_source;
use antecedent_core::ExecutionContext;
use antecedent_core::mnar_fixed_shift::{
    FixedShiftSource, MISSING_BAND_WIDTH, SHIFT_BAND_WIDTH, SamplingPremise, SourceRow,
    scope_functional,
};
use calibration::{
    CoverageTally, RecordKey, SampleGrid, grid_seed, map_replicates, n_sim, stream_seed, uniform,
};

// 50, 1000 and 10000 rows bound the measured range, since a record covers only the row
// counts it measured; a run outside every record refuses as unsupported.
const SIZES: SampleGrid = SampleGrid { name: "mnar_scope", factors: [(1, 20), (1, 1), (10, 1)] };
const BASE_ROWS: usize = 1000;
const ATTEMPTS: u64 = 200;
/// Fewest assigned rows per arm a drawn design keeps, so every arm has observed outcomes.
const MIN_ARM_ROWS: usize = 15;
/// Declared-shift bands measured: `[0, 2)` pooled standard deviations.
const SHIFT_BANDS: usize = 4;
/// Missingness bands measured: `[0, 0.5)` of an arm.
const MISSING_BANDS: usize = 4;

struct Sample {
    interval: (f64, f64),
    basis: antecedent_core::CalibrationBasis,
    truth: f64,
}

fn draw(
    seed: u64,
    replicate: u64,
    rows: usize,
    shift_band: usize,
    missing_band: usize,
) -> Option<Sample> {
    let label = scope_functional(shift_band, missing_band);
    for attempt in 0..ATTEMPTS {
        let mut u = uniform(stream_seed(seed, attempt), 0x6d6e);
        let allocation = 0.3 + 0.4 * u();
        let treated_rows =
            ((allocation * rows as f64).round() as usize).clamp(MIN_ARM_ROWS, rows - MIN_ARM_ROWS);
        let control_mean = 3.0 * (u() - 0.5);
        let effect = 4.0 * u() - 2.0;
        let magnitude = (shift_band as f64 + u()) * SHIFT_BAND_WIDTH;
        let shift = if u() < 0.5 { -magnitude } else { magnitude };
        let lower = missing_band as f64 * MISSING_BAND_WIDTH;
        let heaviest = lower + u() * MISSING_BAND_WIDTH;
        let other = heaviest * u();
        let (control_missing, treated_missing) =
            if u() < 0.5 { (heaviest, other) } else { (other, heaviest) };
        let skewed = replicate % 2 == 1;
        let mut noise = || {
            if skewed {
                -(1.0 - u()).ln() - 1.0
            } else {
                // Box-Muller from the same stream.
                let (a, b) = (1.0 - u(), u());
                (-2.0 * a.ln()).sqrt() * (std::f64::consts::TAU * b).cos()
            }
        };
        let mut gate = uniform(stream_seed(seed, attempt), 0x6d6f);
        let data: Vec<SourceRow> = (0..rows)
            .map(|index| {
                let treated = index >= rows - treated_rows;
                let missing = if treated { treated_missing } else { control_missing };
                let mean = control_mean + if treated { effect } else { 0.0 };
                let value = mean + noise();
                SourceRow {
                    subject_id: index as u64,
                    treated,
                    outcome: (gate() >= missing).then_some(value),
                }
            })
            .collect();
        let Ok(source) = FixedShiftSource::new(
            "calibration_population".into(),
            0,
            1,
            SamplingPremise::IndependentRandomizedArms,
            shift,
            data,
        ) else {
            continue;
        };
        if source.scope_functional() != label {
            continue;
        }
        let ctx = ExecutionContext::for_tests(seed);
        let Ok(prepared) = Study::prepare_mnar_fixed_shift(source, &ctx) else {
            continue;
        };
        let Ok(artifact) = prepared.export_source(&ctx) else {
            continue;
        };
        let Ok(replayed) = consume_source(artifact.bytes(), artifact.expectation(), &ctx) else {
            continue;
        };
        let Ok((interval, basis)) = replayed.interval_calibration_candidate(&ctx) else {
            continue;
        };
        // The population contrast: the treated arm's missing share carries the shift.
        let truth = effect + treated_missing * shift;
        return Some(Sample { interval: (interval.lower, interval.upper), basis, truth });
    }
    None
}

#[test]
#[ignore = "calibration: run through scripts/measure_calibration.sh"]
fn mnar_compiled_shift_delta_io_interval() {
    let rows = SIZES.n(BASE_ROWS);
    for shift_band in 0..SHIFT_BANDS {
        for missing_band in 0..MISSING_BANDS {
            let mut tally = CoverageTally::for_record(
                RecordKey {
                    test: "mnar_compiled_shift_delta_io_interval",
                    dgp: "independent_arms_declared_treated_shift_gaussian_or_exponential",
                    interval: "delta_normal",
                },
                0.95,
            )
            .labelled(scope_functional(shift_band, missing_band));
            let cell = (shift_band * MISSING_BANDS + missing_band) as u64;
            let samples = map_replicates(n_sim(), |replicate| {
                let seed = stream_seed(grid_seed(0x6d6e_0000 + replicate), cell);
                draw(seed, replicate, rows, shift_band, missing_band)
            });
            for sample in samples {
                let Some(sample) = sample else {
                    tally.skip();
                    continue;
                };
                candidate::bind(&mut tally, &sample.basis);
                tally.record(Some(sample.interval), sample.truth);
            }
            let id = tally.record_id().expect("every cell binds at least one execution");
            assert!(
                id.starts_with(
                    "cov.average_effect.not_applicable.frequentist.delta_normal.l95.mnar_compiled_shift_delta_io_interval.v2_validation_not_applicable."
                ),
                "{id}"
            );
            tally.assert();
        }
    }
}
