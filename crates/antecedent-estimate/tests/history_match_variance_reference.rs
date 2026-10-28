//! Period-matched history contrast: standard error against exhaustive noise enumeration.
//!
//! `oracle_data/history_match_variance_reference.py` (frozen in the JSON beside it) fixes scores,
//! arms, periods and weights, matches each treated unit to the nearest control of its period, and
//! lists the exact Abadie-Imbens standard error for every sign pattern of the outcome noise. It
//! asserts, exactly, that the expected squared standard error over those patterns equals the
//! conditional variance of the contrast (weights squared times the noise variance plus squared
//! donated shares times the noise variance), which is the property a pooled pair-difference
//! estimate loses when one control serves many treated units. This test requires the product to
//! reproduce the standard error and contrast on every pattern and to return no standard error
//! exactly where the reference has no within-arm neighbour.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_estimate::history_match::{HistoryUnit, PeriodHistoryContrast};
use serde_json::Value;

#[test]
fn standard_error_matches_every_noise_pattern_and_is_unbiased_for_the_conditional_variance() {
    let designs: Vec<Value> = serde_json::from_str::<Value>(include_str!(
        "oracle_data/history_match_variance_reference.json"
    ))
    .unwrap()
    .as_array()
    .unwrap()
    .clone();
    assert_eq!(designs.len(), 3);
    for design in designs {
        let name = design["name"].as_str().unwrap();
        let units = design["units"].as_array().unwrap();
        let mut mean_square = 0.0;
        let mut defined = 0_usize;
        let patterns = design["realizations"].as_array().unwrap();
        for real in patterns {
            let outcomes: Vec<f64> =
                real["outcomes"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
            let rows: Vec<HistoryUnit> = units
                .iter()
                .zip(&outcomes)
                .map(|(u, &y)| {
                    HistoryUnit::try_new(
                        &[u["score"].as_f64().unwrap()],
                        y,
                        u["treated"].as_bool().unwrap(),
                        u["weight"].as_f64().unwrap(),
                    )
                    .unwrap()
                    .in_period(u["period"].as_i64().unwrap())
                })
                .collect();
            let fit = PeriodHistoryContrast::try_new(None).unwrap().contrast(&rows).unwrap();
            let want_contrast = real["contrast"].as_f64().unwrap();
            assert!(
                (fit.contrast - want_contrast).abs() <= 1e-12 * want_contrast.abs().max(1.0),
                "{name}: contrast {} vs {want_contrast}",
                fit.contrast
            );
            match (fit.standard_error, real["standard_error"].as_f64()) {
                (None, None) => {}
                (Some(got), Some(want)) => {
                    assert!((got - want).abs() <= 1e-12 * want.max(1e-300), "{name}: {got} {want}");
                    mean_square += got * got;
                    defined += 1;
                }
                (got, want) => panic!("{name}: standard error {got:?} but reference {want:?}"),
            }
        }
        if defined > 0 {
            assert_eq!(defined, patterns.len());
            let truth = design["true_variance"].as_f64().unwrap();
            let mean = mean_square / defined as f64;
            assert!((mean - truth).abs() <= 1e-12 * truth, "{name}: E[se^2] {mean} vs {truth}");
        }
    }
}
