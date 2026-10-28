//! The exact-quadratic dose row: reproduction of a quadratic mean at any bandwidth, and
//! agreement of its level and derivative with the 40-digit unpenalised Gaussian local fit.
//!
//! The reproduction argument is that a weighted least-squares quadratic fit reproduces any exact
//! quadratic conditional mean for every positive weighting, so with noise-free data the level,
//! the derivative and a level difference are the truth up to rounding whatever the bandwidth and
//! the dose design. The 40-digit references are the unpenalised level and derivative in
//! `oracle_data/dose_posterior_reference.json` (frozen by `dose_posterior_reference.py`), which
//! are computed from the weighted normal equations and not from this module.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::dose_grid_functional::{
    ClaimSet, DoseDesign, DoseFunctional, DoseGridRequest, estimate_dose_grid_functional,
    estimate_dose_grid_quadratic_mean,
};
use serde_json::Value;

fn request<'a>(
    dose: &'a [f64],
    outcome: &'a [f64],
    grid: &'a [f64],
    bandwidth: f64,
    functional: DoseFunctional,
) -> DoseGridRequest<'a> {
    DoseGridRequest {
        dose,
        outcome,
        grid,
        bandwidth,
        bandwidth_range: (bandwidth * 0.5, bandwidth * 2.0),
        minimum_local_ess: 3.0,
        functional,
        claims: ClaimSet { pointwise_level: true, derivative: true, simultaneous_band: false },
        design: DoseDesign::RandomizedDose,
    }
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn text(v: &Value) -> f64 {
    v.as_str().unwrap().parse().unwrap()
}

#[test]
fn a_noise_free_quadratic_is_reproduced_at_every_bandwidth_and_offset() {
    // A deterministic, unevenly spaced design.
    let base: Vec<f64> =
        (0..40).map(|i| f64::from(i) * 0.31 + 0.2 * (f64::from(i) * 1.7).sin()).collect();
    for offset in [0.0, 1.0e4] {
        let dose: Vec<f64> = base.iter().map(|d| d + offset).collect();
        let (b0, b1, b2) = (1.5, -0.8, 0.07);
        let truth = |d: f64| b0 + b1 * (d - offset) + b2 * (d - offset).powi(2);
        let slope = |d: f64| b1 + 2.0 * b2 * (d - offset);
        let outcome: Vec<f64> = dose.iter().map(|d| truth(*d)).collect();
        let grid = [offset + 4.0, offset + 7.5];
        for bandwidth in [0.8, 2.5, 40.0] {
            let level = estimate_dose_grid_quadratic_mean(&request(
                &dose,
                &outcome,
                &grid,
                bandwidth,
                DoseFunctional::Level,
            ))
            .unwrap();
            let derivative = estimate_dose_grid_quadratic_mean(&request(
                &dose,
                &outcome,
                &grid,
                bandwidth,
                DoseFunctional::Derivative,
            ))
            .unwrap();
            for (i, g) in grid.iter().enumerate() {
                let l = &level.levels[i];
                assert!(
                    (l.level - truth(*g)).abs() < 1e-8 * (1.0 + truth(*g).abs()),
                    "offset {offset} h {bandwidth}: level {} vs {}",
                    l.level,
                    truth(*g)
                );
                assert!(l.interval.smoothing_bias_included);
                let d = &derivative.derivatives[i];
                assert!(
                    (d.derivative - slope(*g)).abs() < 1e-7,
                    "offset {offset} h {bandwidth}: slope {} vs {}",
                    d.derivative,
                    slope(*g)
                );
                assert!(d.interval.smoothing_bias_included);
            }
            let contrast = estimate_dose_grid_quadratic_mean(&request(
                &dose,
                &outcome,
                &grid,
                bandwidth,
                DoseFunctional::Contrast { from: grid[0], to: grid[1] },
            ))
            .unwrap()
            .contrast
            .unwrap();
            let want = truth(grid[1]) - truth(grid[0]);
            assert!((contrast.estimate - want).abs() < 1e-8 * (1.0 + want.abs()), "contrast");
            assert!(contrast.interval.smoothing_bias_included);
        }
    }
}

#[test]
fn the_unrestricted_row_does_not_claim_the_bias_is_included() {
    let dose: Vec<f64> = (0..30).map(|i| f64::from(i) * 0.4).collect();
    let outcome: Vec<f64> =
        dose.iter().map(|d| 1.0 + d * d * 0.02 + (d * 3.0).sin() * 0.1).collect();
    let grid = [5.0];
    let plain =
        estimate_dose_grid_functional(&request(&dose, &outcome, &grid, 1.5, DoseFunctional::Level))
            .unwrap();
    let quadratic = estimate_dose_grid_quadratic_mean(&request(
        &dose,
        &outcome,
        &grid,
        1.5,
        DoseFunctional::Level,
    ))
    .unwrap();
    assert!(!plain.levels[0].interval.smoothing_bias_included);
    assert!(quadratic.levels[0].interval.smoothing_bias_included);
    // The premise changes the label, not the numbers.
    assert_eq!(plain.levels[0].level.to_bits(), quadratic.levels[0].level.to_bits());
    assert_eq!(
        plain.levels[0].standard_error.to_bits(),
        quadratic.levels[0].standard_error.to_bits()
    );
}

#[test]
fn noisy_level_and_derivative_equal_the_multiprecision_unpenalised_fit() {
    let fx: Value =
        serde_json::from_str(include_str!("oracle_data/dose_posterior_reference.json")).unwrap();
    for (index, case) in fx["local_quadratic"].as_array().unwrap().iter().enumerate() {
        let dose = floats(&case["dose"]);
        let outcome = floats(&case["outcome"]);
        let at = case["at"].as_f64().unwrap();
        let bandwidth = case["bandwidth"].as_f64().unwrap();
        let grid = [at];
        let level = estimate_dose_grid_quadratic_mean(&request(
            &dose,
            &outcome,
            &grid,
            bandwidth,
            DoseFunctional::Level,
        ))
        .unwrap();
        let slope = estimate_dose_grid_quadratic_mean(&request(
            &dose,
            &outcome,
            &grid,
            bandwidth,
            DoseFunctional::Derivative,
        ))
        .unwrap();
        let want_level = text(&case["unpenalized_level"]);
        let want_slope = text(&case["unpenalized_derivative"]);
        assert!(
            (level.levels[0].level - want_level).abs() < 1e-9 * want_level.abs().max(1.0),
            "case {index} level"
        );
        assert!(
            (slope.derivatives[0].derivative - want_slope).abs() < 1e-9 * want_slope.abs().max(1.0),
            "case {index} derivative"
        );
    }
}
