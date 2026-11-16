//! A measured exposure declared on the study's treatment recovers the planted effect.
//!
//! Each test plants a causal effect of an exposure that is then recorded with error (shared and
//! outcome-specific misclassification, per-unit probabilities, covariate-dependent detection,
//! repeat codings), declares the measurement, and reads the effect through the facade:
//! `StudyBuilder::measured_exposure` imputes the exposure draws, the ordinary frequentist
//! average-effect study runs on every completed table and the effects are pooled by Rubin's
//! rules. The planted effect must lie within four pooled standard errors and inside the pooled
//! interval; where the recorded column exists, taking it at face value must miss the planted
//! effect. The pooled values are also checked against Rubin's rules written out here.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(
    clippy::float_cmp,
    clippy::cast_precision_loss,
    reason = "integration test with planted truths and a hand-written reference"
)]

mod common;

use antecedent::StudyBuilder;
use antecedent::measured_exposure::{
    PreparedMeasuredExposure, consume_exposure_draws, export_exposure_draws,
};
use antecedent_core::ExecutionContext;
use common::measured_exposure_dgp::{Planted, Scenario, analyse, plant, request};

const DRAWS: usize = 16;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0x4D45_0001)
}

/// Run `body` on a thread with the stack a prepared study result needs in a debug build.
fn on_large_stack(body: impl FnOnce() + Send + 'static) {
    std::thread::Builder::new()
        .name("measured-exposure".into())
        .stack_size(16 * 1024 * 1024)
        .spawn(body)
        .unwrap()
        .join()
        .unwrap();
}

fn prepared(planted: &Planted, seed: u64) -> PreparedMeasuredExposure {
    StudyBuilder::measured_exposure(
        &planted.declaration,
        planted.columns.clone(),
        request(DRAWS, seed),
        &ctx(),
    )
    .unwrap()
}

/// Rubin's rules, written out.
fn rubin(points: &[f64], variances: &[f64]) -> (f64, f64, f64, f64) {
    let m = points.len() as f64;
    let mean = points.iter().sum::<f64>() / m;
    let within = variances.iter().sum::<f64>() / m;
    let between = points.iter().map(|q| (q - mean).powi(2)).sum::<f64>() / (m - 1.0);
    (mean, within, between, within + (1.0 + 1.0 / m) * between)
}

fn recovers(scenario: Scenario, n: usize, seed: u64, naive_gap: Option<f64>) {
    let planted = plant(scenario, n, seed);
    let study = prepared(&planted, seed ^ 0xA5);
    let estimate = study.estimate(|table| analyse(&planted, table, &ctx())).unwrap();
    let effect = &estimate.effect;
    let se = effect.standard_error();
    assert!(
        (effect.pooled.point - planted.truth).abs() < 4.0 * se + 0.03,
        "{scenario:?}: pooled {} against planted {} (se {se})",
        effect.pooled.point,
        planted.truth
    );
    let (lower, upper) = effect.interval(0.95);
    assert!(lower <= planted.truth && planted.truth <= upper, "{scenario:?}: [{lower}, {upper}]");
    // The pooling is Rubin's rules over the per-draw effects the study reported.
    let (mean, within, between, total) = rubin(&effect.points, &effect.variances);
    assert!((effect.pooled.point - mean).abs() < 1e-12);
    assert!((effect.pooled.within - within).abs() < 1e-12);
    assert!((effect.pooled.between - between).abs() < 1e-12);
    assert!((effect.pooled.total - total).abs() < 1e-12);
    match (naive_gap, effect.naive) {
        (Some(gap), Some(naive)) => assert!(
            (naive - planted.truth).abs() > gap,
            "{scenario:?}: the recorded column gives {naive}, near the planted {}",
            planted.truth
        ),
        (None, None) => {}
        (gap, naive) => panic!("{scenario:?}: naive {naive:?} against expected gap {gap:?}"),
    }
}

#[test]
fn shared_misclassification_recovers_the_planted_effect() {
    on_large_stack(|| recovers(Scenario::Shared, 3000, 11, Some(0.3)));
}

#[test]
fn outcome_differential_misclassification_recovers_the_planted_risk_difference() {
    on_large_stack(|| {
        let planted = plant(Scenario::Differential, 6000, 21);
        let study = prepared(&planted, 77);
        let estimate = study.estimate(|table| analyse(&planted, table, &ctx())).unwrap();
        let effect = &estimate.effect;
        assert!(
            (effect.pooled.point - planted.truth).abs() < 4.0 * effect.standard_error() + 0.03,
            "pooled {} against planted {}",
            effect.pooled.point,
            planted.truth
        );
        let (lower, upper) = effect.interval(0.95);
        assert!(lower <= planted.truth && planted.truth <= upper);
    });
}

#[test]
fn per_unit_exposure_probabilities_recover_the_planted_effect() {
    on_large_stack(|| recovers(Scenario::Soft, 3000, 41, None));
}

#[test]
fn covariate_dependent_detection_recovers_the_planted_effect() {
    on_large_stack(|| recovers(Scenario::Detected, 3000, 51, Some(0.3)));
}

#[test]
fn repeat_codings_recover_the_planted_effect_of_the_latent_facet() {
    on_large_stack(|| recovers(Scenario::Repeat, 3000, 61, None));
}

#[test]
fn the_exposure_draws_export_as_a_posterior_artifact_and_read_back() {
    on_large_stack(|| {
        let planted = plant(Scenario::Shared, 400, 91);
        let study = prepared(&planted, 5);
        let draws = study.draws();
        assert_eq!(draws.draw_count, DRAWS);
        assert!(draws.draws.iter().all(|&v| v == 0.0 || v == 1.0));
        assert!(draws.probability.iter().all(|p| (0.0..=1.0).contains(p)));
        let bytes = study.export_draws().unwrap();
        let back = consume_exposure_draws(&bytes, "x", Some(&draws.declaration_identity)).unwrap();
        assert_eq!(back.draws, draws.draws);
        assert_eq!(back.declaration_identity, draws.declaration_identity);
        assert_eq!(back.data_identity, draws.data_identity);
        assert!(consume_exposure_draws(&bytes, "other", None).is_err());
        assert!(consume_exposure_draws(&bytes, "x", Some(&"0".repeat(64))).is_err());
        assert_eq!(export_exposure_draws("x", &back).unwrap(), bytes);
        // The completed table carries the draw in place of the recorded column.
        assert_eq!(antecedent::data::TableView::row_count(&study.completed_table(2).unwrap()), 400);
        // The same seed reproduces the draws; another seed does not.
        assert_eq!(prepared(&planted, 5).draws().draws, draws.draws);
        assert_ne!(prepared(&planted, 6).draws().draws, draws.draws);
    });
}

#[test]
fn an_unidentified_error_declaration_refuses_before_any_draw() {
    use antecedent::estimate::measurement::{
        ExposureErrorDeclaration, ExposureMeasurement, RatePair,
    };
    let planted = plant(Scenario::Shared, 200, 101);
    let ExposureMeasurement::Misclassified { recorded, .. } = &planted.declaration else {
        unreachable!("the shared scenario is a misclassification");
    };
    let weak = ExposureMeasurement::Misclassified {
        recorded: recorded.clone(),
        error: ExposureErrorDeclaration::Shared(RatePair::declared(0.55, 0.5, 0.0, 0.0).unwrap()),
        min_youden: 0.1,
    };
    let error =
        StudyBuilder::measured_exposure(&weak, planted.columns.clone(), request(4, 1), &ctx())
            .unwrap_err();
    assert!(error.to_string().contains("youden_too_small"), "{error}");
}

#[test]
fn an_outcome_specific_error_needs_a_binary_outcome() {
    use antecedent::estimate::measurement::{
        ExposureErrorDeclaration, ExposureMeasurement, RatePair,
    };
    let planted = plant(Scenario::Shared, 200, 111);
    let ExposureMeasurement::Misclassified { recorded, .. } = &planted.declaration else {
        unreachable!("the shared scenario is a misclassification");
    };
    let declaration = ExposureMeasurement::Misclassified {
        recorded: recorded.clone(),
        error: ExposureErrorDeclaration::ByOutcome {
            cases: RatePair::declared(0.9, 0.8, 0.0, 0.0).unwrap(),
            controls: RatePair::declared(0.7, 0.95, 0.0, 0.0).unwrap(),
        },
        min_youden: 0.1,
    };
    let error = StudyBuilder::measured_exposure(
        &declaration,
        planted.columns.clone(),
        request(4, 1),
        &ctx(),
    )
    .unwrap_err();
    assert!(error.to_string().contains("binary_outcome_continuous_error"), "{error}");
}
