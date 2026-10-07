//! 2.3 B1 (X4): the named-functional dose-response row against closed-form truth.
//!
//! Oracle (independent of the estimator): the response is the quadratic
//! `m(d) = 1 + 0.5 d + 0.25 d^2`, so `m'(d) = 0.5 + 0.5 d`, written out per grid
//! dose below, and `m(3) - m(1) = 4.75 - 1.75 = 3`. Doses sit on an exact uniform
//! design over `[0, 4]`. A Gaussian-kernel local quadratic reproduces a quadratic
//! exactly when the data carry no noise, so the noise-free design checks the
//! numerical error; an alternating deterministic perturbation of the outcome checks
//! a stated tolerance. No coverage is claimed anywhere.
#![allow(
    clippy::cast_precision_loss,
    clippy::float_cmp,
    reason = "exact deterministic design and closed-form truth"
)]

use antecedent_core::SupportStatus;
use antecedent_estimate::EstimationError;
use antecedent_estimate::dose_grid_functional::{
    ClaimSet, DoseDesign, DoseFunctional, DoseGridRequest, DoseGridRow, IntervalCalibration,
    dose_support_report, estimate_dose_grid_functional,
};

const N: usize = 2001;

fn truth_level(d: f64) -> f64 {
    1.0 + 0.5 * d + 0.25 * d * d
}

fn truth_derivative(d: f64) -> f64 {
    0.5 + 0.5 * d
}

fn design(noise: f64) -> (Vec<f64>, Vec<f64>) {
    let dose: Vec<f64> = (0..N).map(|i| 4.0 * i as f64 / (N - 1) as f64).collect();
    let outcome = dose
        .iter()
        .enumerate()
        .map(|(i, &d)| truth_level(d) + if i % 2 == 0 { noise } else { -noise })
        .collect();
    (dose, outcome)
}

fn claims(level: bool, derivative: bool) -> ClaimSet {
    ClaimSet { pointwise_level: level, derivative, simultaneous_band: false }
}

fn request<'a>(
    dose: &'a [f64],
    outcome: &'a [f64],
    grid: &'a [f64],
    functional: DoseFunctional,
    claims: ClaimSet,
) -> DoseGridRequest<'a> {
    DoseGridRequest {
        dose,
        outcome,
        grid,
        bandwidth: 0.4,
        bandwidth_range: (0.1, 1.0),
        minimum_local_ess: 50.0,
        functional,
        claims,
        design: DoseDesign::RandomizedDose,
    }
}

fn reason(error: &EstimationError) -> String {
    error.refusal_fields().and_then(|fields| fields.reason.clone()).unwrap_or_default()
}

const GRID: [f64; 4] = [0.5, 1.0, 2.0, 3.5];

fn run(functional: DoseFunctional, claim_set: ClaimSet, noise: f64) -> DoseGridRow {
    let (dose, outcome) = design(noise);
    let row =
        estimate_dose_grid_functional(&request(&dose, &outcome, &GRID, functional, claim_set));
    assert!(row.is_ok(), "row must evaluate: {:?}", row.as_ref().err());
    row.unwrap()
}

#[test]
fn b1_level_recovers_closed_form_exactly_without_noise() {
    // Truth per dose: m(0.5)=1.3125, m(1)=1.75, m(2)=3, m(3.5)=5.8125.
    let row = run(DoseFunctional::Level, claims(true, false), 0.0);
    let expected = [1.3125, 1.75, 3.0, 5.8125];
    assert_eq!(row.levels.len(), 4);
    for (level, truth) in row.levels.iter().zip(expected) {
        assert!((level.level - truth).abs() < 1e-7, "{} vs {truth}", level.level);
        assert!((truth - truth_level(level.dose)).abs() < 1e-12);
        assert_eq!(level.support, SupportStatus::Supported);
    }
}

#[test]
fn b1_derivative_recovers_closed_form_exactly_without_noise() {
    // Truth per dose: m'(0.5)=0.75, m'(1)=1, m'(2)=1.5, m'(3.5)=2.25.
    let row = run(DoseFunctional::Derivative, claims(false, true), 0.0);
    let expected = [0.75, 1.0, 1.5, 2.25];
    assert_eq!(row.derivatives.len(), 4);
    for (item, truth) in row.derivatives.iter().zip(expected) {
        assert!((item.derivative - truth).abs() < 1e-6, "{} vs {truth}", item.derivative);
        assert!((truth - truth_derivative(item.dose)).abs() < 1e-12);
    }
}

#[test]
fn b1_level_and_derivative_recover_truth_under_alternating_noise_within_tolerance() {
    let level_row = run(DoseFunctional::Level, claims(true, false), 0.1);
    for level in &level_row.levels {
        assert!((level.level - truth_level(level.dose)).abs() < 0.02, "level at {}", level.dose);
        assert!(level.standard_error.is_finite() && level.standard_error > 0.0);
    }
    let derivative_row = run(DoseFunctional::Derivative, claims(false, true), 0.1);
    for item in &derivative_row.derivatives {
        assert!(
            (item.derivative - truth_derivative(item.dose)).abs() < 0.05,
            "derivative at {}",
            item.dose
        );
        assert!(item.standard_error.is_finite() && item.standard_error > 0.0);
    }
}

#[test]
fn b1_contrast_between_named_doses_matches_closed_form() {
    // m(3) - m(1) = 4.75 - 1.75 = 3.
    let (dose, outcome) = design(0.0);
    let contrast = DoseFunctional::Contrast { from: 1.0, to: 3.0 };
    let exact = estimate_dose_grid_functional(&request(
        &dose,
        &outcome,
        &[],
        contrast,
        claims(true, false),
    ));
    assert!(exact.is_ok());
    let exact = exact.unwrap().contrast.unwrap();
    assert!((exact.estimate - 3.0).abs() < 1e-7);
    assert_eq!(exact.from_support, SupportStatus::Supported);
    assert_eq!(exact.to_support, SupportStatus::Supported);

    let (dose, outcome) = design(0.1);
    let noisy = estimate_dose_grid_functional(&request(
        &dose,
        &outcome,
        &[],
        contrast,
        claims(true, false),
    ));
    assert!(noisy.is_ok());
    let noisy = noisy.unwrap().contrast.unwrap();
    assert!((noisy.estimate - 3.0).abs() < 0.02);
    assert!(noisy.standard_error.is_finite() && noisy.standard_error > 0.0);
    assert_eq!(noisy.interval.calibration, IntervalCalibration::Unmeasured);
    assert!(noisy.interval.lower < noisy.estimate && noisy.estimate < noisy.interval.upper);
    assert!(!noisy.interval.smoothing_bias_included);
}

#[test]
fn b1_claims_are_separate_typed_fields_and_band_stays_closed() {
    let level = run(DoseFunctional::Level, claims(true, false), 0.1);
    assert!(level.derivatives.is_empty() && level.contrast.is_none());
    assert!(level.simultaneous_band.is_none());
    let derivative = run(DoseFunctional::Derivative, claims(false, true), 0.1);
    assert!(derivative.levels.is_empty() && derivative.contrast.is_none());
    assert!(derivative.simultaneous_band.is_none());
    for item in &derivative.derivatives {
        assert_eq!(item.interval.calibration, IntervalCalibration::Unmeasured);
    }

    let (dose, outcome) = design(0.1);
    let band = ClaimSet { pointwise_level: true, derivative: true, simultaneous_band: true };
    let refused = estimate_dose_grid_functional(&request(
        &dose,
        &outcome,
        &GRID,
        DoseFunctional::Level,
        band,
    ));
    assert!(refused.is_err());
    assert_eq!(reason(&refused.unwrap_err()), "dose_grid.simultaneous_band_closed");
}

#[test]
fn b1_derivative_without_derivative_claim_is_refused() {
    let (dose, outcome) = design(0.1);
    // A pointwise level claim does not license a derivative.
    let refused = estimate_dose_grid_functional(&request(
        &dose,
        &outcome,
        &GRID,
        DoseFunctional::Derivative,
        claims(true, false),
    ));
    assert!(refused.is_err());
    assert_eq!(reason(&refused.unwrap_err()), "dose_grid.derivative_without_claim");
    // A derivative claim does not license a level.
    let refused = estimate_dose_grid_functional(&request(
        &dose,
        &outcome,
        &GRID,
        DoseFunctional::Level,
        claims(false, true),
    ));
    assert!(refused.is_err());
    assert_eq!(reason(&refused.unwrap_err()), "dose_grid.level_without_claim");
}

#[test]
fn b1_dose_outside_support_is_labelled_and_refused() {
    let (dose, outcome) = design(0.1);
    let points = [1.0, 5.0];
    let report = dose_support_report(&dose, &points, 0.4, 50.0);
    assert!(report.is_ok());
    let report = report.unwrap();
    let labels = report.point_status.clone().unwrap();
    assert_eq!(labels[0], SupportStatus::Supported);
    assert_eq!(labels[1], SupportStatus::OutsideEmpiricalSupport);
    assert_eq!(report.status, SupportStatus::OutsideEmpiricalSupport);

    let grid = [1.0, 5.0];
    let refused = estimate_dose_grid_functional(&request(
        &dose,
        &outcome,
        &grid,
        DoseFunctional::Level,
        claims(true, false),
    ));
    assert!(refused.is_err());
    assert_eq!(reason(&refused.unwrap_err()), "dose_grid.unsupported_dose");
}

#[test]
fn b1_insufficient_local_weight_is_labelled_and_refused() {
    let (dose, outcome) = design(0.1);
    // The effective size can never exceed the row count, so this minimum is unmeetable.
    let mut asked = request(&dose, &outcome, &GRID, DoseFunctional::Level, claims(true, false));
    asked.minimum_local_ess = 5000.0;
    let report = dose_support_report(&dose, &GRID, 0.4, 5000.0).unwrap();
    assert_eq!(report.status, SupportStatus::WeakOverlap);
    let refused = estimate_dose_grid_functional(&asked);
    assert!(refused.is_err());
    assert_eq!(reason(&refused.unwrap_err()), "dose_grid.insufficient_local_weight");
}

#[test]
fn b1_bandwidth_outside_declared_range_is_refused() {
    let (dose, outcome) = design(0.1);
    for bandwidth in [0.05, 2.0] {
        let mut asked = request(&dose, &outcome, &GRID, DoseFunctional::Level, claims(true, false));
        asked.bandwidth = bandwidth;
        let refused = estimate_dose_grid_functional(&asked);
        assert!(refused.is_err());
        assert_eq!(reason(&refused.unwrap_err()), "dose_grid.bandwidth_outside_range");
    }
}

#[test]
fn b1_observational_dose_is_not_certified() {
    let (dose, outcome) = design(0.1);
    let mut asked = request(&dose, &outcome, &GRID, DoseFunctional::Level, claims(true, false));
    asked.design = DoseDesign::Observational;
    let refused = estimate_dose_grid_functional(&asked);
    assert!(refused.is_err());
    assert_eq!(reason(&refused.unwrap_err()), "dose_grid.graph_not_certified");
}

#[test]
fn b1_numerical_error_is_measured_and_negligible() {
    let row = run(DoseFunctional::Level, claims(true, false), 0.1);
    assert_eq!(row.numerical.fits, GRID.len());
    // The linearized level influences sum to zero in exact arithmetic.
    assert!(row.numerical.max_abs_influence_sum < 1e-8, "{}", row.numerical.max_abs_influence_sum);
    let exact = run(DoseFunctional::Level, claims(true, false), 0.0);
    let worst = exact
        .levels
        .iter()
        .map(|level| (level.level - truth_level(level.dose)).abs())
        .fold(0.0, f64::max);
    assert!(worst < 1e-7, "{worst}");
}
