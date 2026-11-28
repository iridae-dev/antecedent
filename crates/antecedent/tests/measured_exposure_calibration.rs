//! Repeated-sampling coverage of the pooled interval of a measured exposure's effect, scored
//! against a planted truth, for every declaration.
//!
//! Each replicate plants a dataset, declares the measurement on the exposure, imputes the
//! exposure draws, runs the ordinary frequentist average-effect study on every completed table
//! and reports the Rubin pooled `t` interval at 0.95 (`StudyBuilder::measured_exposure` and
//! `PreparedMeasuredExposure::estimate`). A replicate whose imputation or study refuses is a
//! skip and counts as a miss.
//!
//! * shared misclassification: sensitivity 0.85, specificity 0.9, rates declared with variance
//!   `1e-4`, continuous outcome, planted effect 2;
//! * outcome-differential misclassification: cases (0.9, 0.8), controls (0.7, 0.95), each
//!   declared as a validation study's estimate (planted rate plus an error of the declared
//!   variance `1e-4`), a 0/1 outcome, truth the planted risk difference of a logistic outcome integrated over the
//!   covariate;
//! * per-unit probabilities calibrated given the covariate, planted effect 2;
//! * covariate-dependent detection `expit(-1 + 0.8 c)` of a count `c ~ Poisson(3)` with its two
//!   parameters carried with their variances, planted effect 2;
//! * three conditionally independent codings of a latent binary facet (sensitivity 0.8,
//!   specificity 0.85), planted effect 2.
//!
//! The imputation and the pooling are conservative by construction when the imputation outcome
//! model is richer than needed; coverage at or above nominal is a pass.
//!
//! Ignored tests run via `scripts/measure_calibration.sh`.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::cast_sign_loss,
    clippy::cast_possible_truncation,
    reason = "bounded replicate indices in a calibration fixture"
)]
#![allow(clippy::cast_precision_loss, reason = "bounded simulation sizes and small indices")]

mod common;

use antecedent::StudyBuilder;
use antecedent_core::ExecutionContext;
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, grid_n, grid_seed,
    map_replicates, n_sim,
};
use common::measured_exposure_dgp::{Scenario, analyse, plant, request};

const DRAWS: usize = 12;
const ROWS: u64 = 1500;

/// One replicate's reported interval, the construction it was reported under and its row count.
struct Reading {
    interval: (f64, f64),
    construction: Construction,
    rows: u64,
}

fn construction(estimator: &str, dependence: &str) -> Construction {
    Construction {
        query: "MeasurementCorrectedFunctional".into(),
        graph_class: "Dag".into(),
        structure: "fixed".into(),
        modality: "tabular".into(),
        inference: "Frequentist".into(),
        estimator: estimator.into(),
        interval_method: "rubin_t_interval".into(),
        se_kind: "rubin_total_variance".into(),
        dependence: dependence.into(),
        posterior: String::new(),
        functional: "measured_exposure_effect".into(),
        identification: "point".into(),
        reported_level: REPORTED_LEVEL,
        unit: None,
    }
}

/// Score `replicate` over `n_sim()` datasets against each dataset's planted truth and assert
/// nominal coverage.
fn measure(
    test: &'static str,
    dgp: &'static str,
    seed_base: u64,
    replicate: impl Fn(u64) -> Option<(Reading, f64)> + Sync,
) {
    let mut tally = CoverageTally::for_record(
        RecordKey { test, dgp, interval: "rubin_t_interval" },
        REPORTED_LEVEL,
    );
    let readings = map_replicates(n_sim(), |rep| replicate(grid_seed(seed_base + rep)));
    for reading in readings {
        let Some((reading, truth)) = reading else {
            tally.skip();
            continue;
        };
        tally.bind_validation(
            &reading.construction,
            ScopeFacts {
                row_count: reading.rows,
                replicates_ok: None,
                posterior_draws: None,
                unidentified_mass: 0.0,
            },
            "not_applicable",
        );
        tally.record(Some(reading.interval), truth);
    }
    tally.assert();
}

fn replicate(
    scenario: Scenario,
    estimator: &str,
    dependence: &str,
    seed: u64,
) -> Option<(Reading, f64)> {
    let n = grid_n(ROWS as usize);
    // The study of a completed table needs the stack a prepared study result needs in a debug
    // build.
    let estimator = estimator.to_owned();
    let dependence = dependence.to_owned();
    antecedent::support::caller_thread()
        .stack_size(16 * 1024 * 1024)
        .spawn(move || {
            let planted = plant(scenario, n, seed);
            let ctx = ExecutionContext::for_tests(seed);
            let study = StudyBuilder::measured_exposure(
                &planted.declaration,
                planted.columns.clone(),
                request(DRAWS, seed ^ 0x5EED),
                &ctx,
            )
            .ok()?;
            let estimate = study.estimate(|table| analyse(&planted, table, &ctx)).ok()?;
            Some((
                Reading {
                    interval: estimate.effect.interval(REPORTED_LEVEL),
                    construction: construction(&estimator, &dependence),
                    rows: n as u64,
                },
                planted.truth,
            ))
        })
        .ok()?
        .join()
        .ok()?
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measured_exposure_shared_misclassification_rubin_l95() {
    measure(
        "measured_exposure_shared_misclassification_rubin_l95",
        "shared_misclassified_exposure_effect_2",
        0x4E11_0000,
        |seed| {
            replicate(
                Scenario::Shared,
                "measurement.measured_exposure_effect.misclassified",
                "declared_non_differential",
                seed,
            )
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measured_exposure_differential_misclassification_rubin_l95() {
    measure(
        "measured_exposure_differential_misclassification_rubin_l95",
        "outcome_differential_misclassified_exposure_risk_difference",
        0x4E12_0000,
        |seed| {
            replicate(
                Scenario::DifferentialValidated,
                "measurement.measured_exposure_effect.misclassified",
                "declared_differential_by_outcome",
                seed,
            )
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measured_exposure_soft_probabilities_rubin_l95() {
    measure(
        "measured_exposure_soft_probabilities_rubin_l95",
        "calibrated_exposure_probability_effect_2",
        0x4E13_0000,
        |seed| {
            replicate(
                Scenario::Soft,
                "measurement.measured_exposure_effect.soft",
                "declared_non_differential",
                seed,
            )
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measured_exposure_detection_rubin_l95() {
    measure(
        "measured_exposure_detection_rubin_l95",
        "covariate_dependent_detection_effect_2",
        0x4E14_0000,
        |seed| {
            replicate(
                Scenario::Detected,
                "measurement.measured_exposure_effect.detected",
                "declared_non_differential",
                seed,
            )
        },
    );
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn measured_exposure_repeat_codings_rubin_l95() {
    measure(
        "measured_exposure_repeat_codings_rubin_l95",
        "three_binary_codings_of_a_latent_facet_effect_2",
        0x4E15_0000,
        |seed| {
            replicate(
                Scenario::Repeat,
                "measurement.measured_exposure_effect.repeat_coded",
                "declared_non_differential",
                seed,
            )
        },
    );
}
