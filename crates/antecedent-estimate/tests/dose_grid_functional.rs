//! The named-functional dose-response row against closed-form truth.
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

/// Independent local-quadratic reference at `x0`: Gaussian-kernel weighted least squares of the
/// outcome on `[1, d - x0, (d - x0)^2]`, solved from the 3 by 3 normal equations. Returns the
/// level, the derivative, and the HC0 sandwich standard errors `sqrt(sum_i (l_i e_i)^2)` of each,
/// where `l_i` are the equivalent-kernel weights of the coefficient and `e_i` the residuals.
fn local_quadratic_reference(
    dose: &[f64],
    outcome: &[f64],
    x0: f64,
    bandwidth: f64,
) -> (f64, f64, f64, f64, Vec<f64>) {
    let w: Vec<f64> = dose.iter().map(|d| (-0.5 * ((d - x0) / bandwidth).powi(2)).exp()).collect();
    let basis = |d: f64| [1.0, d - x0, (d - x0).powi(2)];
    let mut a = [[0.0_f64; 3]; 3];
    let mut b = [0.0_f64; 3];
    for i in 0..dose.len() {
        let x = basis(dose[i]);
        for r in 0..3 {
            for c in 0..3 {
                a[r][c] += w[i] * x[r] * x[c];
            }
            b[r] += w[i] * x[r] * outcome[i];
        }
    }
    let inverse = invert3(a);
    let beta: Vec<f64> =
        (0..3).map(|r| (0..3).map(|c| inverse[r][c] * b[c]).sum::<f64>()).collect();
    let mut var = [0.0_f64; 2];
    let mut influence = Vec::with_capacity(dose.len());
    for i in 0..dose.len() {
        let x = basis(dose[i]);
        let fitted: f64 = (0..3).map(|k| beta[k] * x[k]).sum();
        let e = outcome[i] - fitted;
        for (slot, row) in [0_usize, 1].into_iter().enumerate() {
            let l: f64 = (0..3).map(|c| inverse[row][c] * w[i] * x[c]).sum();
            var[slot] += (l * e).powi(2);
            if row == 0 {
                influence.push(l * e);
            }
        }
    }
    (beta[0], beta[1], var[0].sqrt(), var[1].sqrt(), influence)
}

fn invert3(m: [[f64; 3]; 3]) -> [[f64; 3]; 3] {
    let det = m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
        - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
        + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0]);
    let mut out = [[0.0; 3]; 3];
    for r in 0..3 {
        for c in 0..3 {
            let (r1, r2) = ((c + 1) % 3, (c + 2) % 3);
            let (c1, c2) = ((r + 1) % 3, (r + 2) % 3);
            out[r][c] = (m[r1][c1] * m[r2][c2] - m[r1][c2] * m[r2][c1]) / det;
        }
    }
    out
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
fn level_recovers_closed_form_exactly_without_noise() {
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
fn derivative_recovers_closed_form_exactly_without_noise() {
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
fn level_and_derivative_recover_truth_under_alternating_noise_within_tolerance() {
    let (dose, outcome) = design(0.1);
    let level_row = run(DoseFunctional::Level, claims(true, false), 0.1);
    for level in &level_row.levels {
        assert!((level.level - truth_level(level.dose)).abs() < 0.02, "level at {}", level.dose);
        // Independent Gaussian-kernel local-quadratic fit and HC0 sandwich on the same rows.
        let (reference, _, se, _, _) = local_quadratic_reference(&dose, &outcome, level.dose, 0.4);
        assert!((level.level - reference).abs() < 1e-9, "level {} vs {reference}", level.level);
        assert!((level.standard_error - se).abs() < 1e-9, "se {} vs {se}", level.standard_error);
    }
    let derivative_row = run(DoseFunctional::Derivative, claims(false, true), 0.1);
    for item in &derivative_row.derivatives {
        assert!(
            (item.derivative - truth_derivative(item.dose)).abs() < 0.05,
            "derivative at {}",
            item.dose
        );
        let (_, reference, _, se, _) = local_quadratic_reference(&dose, &outcome, item.dose, 0.4);
        assert!((item.derivative - reference).abs() < 1e-9, "{} vs {reference}", item.derivative);
        assert!((item.standard_error - se).abs() < 1e-9, "se {} vs {se}", item.standard_error);
    }
}

#[test]
fn contrast_between_named_doses_matches_closed_form() {
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
    let (to, from) = (
        local_quadratic_reference(&dose, &outcome, 3.0, 0.4),
        local_quadratic_reference(&dose, &outcome, 1.0, 0.4),
    );
    let difference = to.0 - from.0;
    assert!((noisy.estimate - difference).abs() < 1e-9, "{} vs {difference}", noisy.estimate);
    let se = to.4.iter().zip(&from.4).map(|(a, b)| (a - b).powi(2)).sum::<f64>().sqrt();
    assert!((noisy.standard_error - se).abs() < 1e-9, "se {} vs {se}", noisy.standard_error);
    assert_eq!(noisy.interval.calibration, IntervalCalibration::Unmeasured);
    assert!(noisy.interval.lower < noisy.estimate && noisy.estimate < noisy.interval.upper);
    assert!(!noisy.interval.smoothing_bias_included);
}

#[test]
fn claims_are_separate_typed_fields_and_band_stays_closed() {
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
fn derivative_without_derivative_claim_is_refused() {
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
fn dose_outside_support_is_labelled_and_refused() {
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
fn insufficient_local_weight_is_labelled_and_refused() {
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
fn support_report_uses_the_same_local_ess_floor_as_estimation() {
    let (dose, outcome) = design(0.1);
    let mut asked = request(&dose, &outcome, &GRID, DoseFunctional::Level, claims(true, false));
    for minimum in [-1.0, 0.0, 2.0] {
        asked.minimum_local_ess = minimum;
        assert_eq!(
            reason(&dose_support_report(&dose, &GRID, 0.4, minimum).unwrap_err()),
            "dose_grid.invalid_request"
        );
        assert_eq!(
            reason(&estimate_dose_grid_functional(&asked).unwrap_err()),
            "dose_grid.invalid_request"
        );
    }
}

#[test]
fn bandwidth_outside_declared_range_is_refused() {
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
fn observational_dose_is_not_certified() {
    let (dose, outcome) = design(0.1);
    let mut asked = request(&dose, &outcome, &GRID, DoseFunctional::Level, claims(true, false));
    asked.design = DoseDesign::Observational;
    let refused = estimate_dose_grid_functional(&asked);
    assert!(refused.is_err());
    assert_eq!(reason(&refused.unwrap_err()), "dose_grid.graph_not_certified");
}

#[test]
fn numerical_error_is_measured_and_negligible() {
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

#[test]
fn exact_quadratic_scope_derives_zero_bias_without_claiming_sampling_calibration() {
    use antecedent_estimate::dose_grid_functional::estimate_dose_grid_quadratic_mean;
    let oracle: serde_json::Value = serde_json::from_str(include_str!(
        "../../../conformance/transport/quadratic_dose/expected.json"
    ))
    .unwrap();
    let coefficients = oracle["conditional_mean_coefficients"].as_array().unwrap();
    let b: Vec<f64> = coefficients.iter().map(|value| value.as_f64().unwrap()).collect();
    let (dose, outcome) = design(0.0);
    for bandwidth in [0.2, 0.4, 0.8] {
        for functional in [
            DoseFunctional::Level,
            DoseFunctional::Derivative,
            DoseFunctional::Contrast { from: 1.0, to: 3.0 },
        ] {
            let mut req = request(&dose, &outcome, &GRID, functional, claims(true, true));
            req.bandwidth = bandwidth;
            let scoped = estimate_dose_grid_quadratic_mean(&req).unwrap();
            let generic = estimate_dose_grid_functional(&req).unwrap();
            for p in &scoped.levels {
                assert!((p.level - (b[0] + b[1] * p.dose + b[2] * p.dose.powi(2))).abs() < 1e-6);
                assert!(p.interval.smoothing_bias_included);
                assert_eq!(p.interval.calibration, IntervalCalibration::Unmeasured);
            }
            for p in &scoped.derivatives {
                assert!((p.derivative - (b[1] + 2.0 * b[2] * p.dose)).abs() < 1e-5);
                assert!(p.interval.smoothing_bias_included);
            }
            if let Some(c) = &scoped.contrast {
                assert!((c.estimate - oracle["contrast"].as_f64().unwrap()).abs() < 1e-6);
                assert!(c.interval.smoothing_bias_included);
            }
            assert!(generic.levels.iter().all(|p| !p.interval.smoothing_bias_included));
            assert!(generic.derivatives.iter().all(|p| !p.interval.smoothing_bias_included));
            assert!(generic.contrast.is_none_or(|c| !c.interval.smoothing_bias_included));
        }
    }
}

#[test]
fn a_missing_dose_or_outcome_is_refused_instead_of_the_row_being_dropped() {
    let grid = [1.0, 2.0];
    let (dose, outcome) = design(0.0);
    let ok = estimate_dose_grid_functional(&request(
        &dose,
        &outcome,
        &grid,
        DoseFunctional::Level,
        claims(true, false),
    ));
    assert!(ok.is_ok());
    for column in ["dose", "outcome"] {
        let (mut dose, mut outcome) = (dose.clone(), outcome.clone());
        if column == "dose" {
            dose[17] = f64::NAN;
        } else {
            outcome[17] = f64::NAN;
        }
        let error = estimate_dose_grid_functional(&request(
            &dose,
            &outcome,
            &grid,
            DoseFunctional::Level,
            claims(true, false),
        ))
        .expect_err("a row with a missing value was used");
        assert!(error.to_string().contains("finite"), "{column}: {error}");
    }
}
