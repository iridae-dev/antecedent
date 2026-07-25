//! Particle-filter calibration against frozen exact scalar Kalman targets.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::fs;
use std::path::PathBuf;

use antecedent_state::{LgssmParams, ParticleFilterState};
use serde_json::Value as JsonValue;

fn fixture() -> JsonValue {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/design_state/particle_filter_kalman/expected.json");
    serde_json::from_str(&fs::read_to_string(path).expect("particle-filter fixture"))
        .expect("parse particle-filter fixture")
}

fn weighted_moments(state: &ParticleFilterState) -> (f64, f64) {
    let max = state.log_weights.iter().copied().fold(f64::NEG_INFINITY, f64::max);
    let mut weights: Vec<f64> = state.log_weights.iter().map(|value| (value - max).exp()).collect();
    let sum: f64 = weights.iter().sum();
    for weight in &mut weights {
        *weight /= sum;
    }
    let mean: f64 = weights.iter().zip(&state.particles).map(|(w, x)| w * x).sum();
    let variance: f64 =
        weights.iter().zip(&state.particles).map(|(w, x)| w * (x - mean).powi(2)).sum();
    (mean, variance)
}

#[test]
fn particle_filter_moments_match_exact_kalman_filter() {
    let fixture = fixture();
    let acceptance = &fixture["acceptance"];
    let particles = acceptance["particles"].as_u64().unwrap() as usize;
    let seeds: Vec<u64> =
        acceptance["seeds"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
    let checkpoints: Vec<usize> = acceptance["checkpoints"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    let mean_atol = acceptance["mean_atol"].as_f64().unwrap();
    let variance_atol = acceptance["variance_atol"].as_f64().unwrap();

    for case in fixture["cases"].as_array().unwrap() {
        let params = LgssmParams {
            a: case["params"]["a"].as_f64().unwrap(),
            process_std: case["params"]["process_std"].as_f64().unwrap(),
            obs_std: case["params"]["obs_std"].as_f64().unwrap(),
        };
        let observations: Vec<f64> =
            case["observations"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        let exact_means = case["reference"]["filtered_means"].as_array().unwrap();
        let exact_variances = case["reference"]["filtered_variances"].as_array().unwrap();
        for &seed in &seeds {
            let mut state = ParticleFilterState::init(particles, params, 1, seed).unwrap();
            for (time, &observation) in observations.iter().enumerate() {
                state.step(observation).unwrap();
                if checkpoints.contains(&time) {
                    let (mean, variance) = weighted_moments(&state);
                    let expected_mean = exact_means[time].as_f64().unwrap();
                    let expected_variance = exact_variances[time].as_f64().unwrap();
                    assert!(
                        (mean - expected_mean).abs() <= mean_atol,
                        "{} seed={seed} t={time}: mean {mean} != {expected_mean}",
                        case["name"]
                    );
                    assert!(
                        (variance - expected_variance).abs() <= variance_atol,
                        "{} seed={seed} t={time}: variance {variance} != {expected_variance}",
                        case["name"]
                    );
                }
            }
            let replay =
                ParticleFilterState::run_batch(&observations, particles, params, 1, seed).unwrap();
            assert_eq!(state, replay, "fixed-seed batch/stream replay must be exact");
        }
    }
}

#[test]
fn particle_filter_fixture_records_exact_pins_and_no_generator() {
    let fixture = fixture();
    assert_eq!(fixture["oracle"]["packages"]["numpy"]["version"], "2.1.3");
    assert_eq!(
        fixture["oracle"]["generation_location"],
        "temporary external harness; not retained"
    );
    assert_eq!(
        fixture["oracle"]["packages"]["numpy"]["metadata_sha256"].as_str().unwrap().len(),
        64
    );
}
