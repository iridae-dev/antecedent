//! Frozen statsmodels parity for adjustment primitives and discovery-family routing.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation)]

use std::fs;
use std::path::PathBuf;

use antecedent_core::{Lag, VariableId};
use antecedent_discovery::{LaggedLink, ScoredLink, threshold_scored_links};
use antecedent_stats::{FdrAdjustment, MultipleTestingMethod, adjust_pvalues};
use serde_json::Value as JsonValue;

fn fixture() -> JsonValue {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../conformance/discovery/multiplicity/expected.json");
    serde_json::from_str(&fs::read_to_string(path).expect("multiplicity fixture"))
        .expect("parse multiplicity fixture")
}

fn method(name: &str) -> MultipleTestingMethod {
    match name {
        "benjamini_hochberg" => MultipleTestingMethod::BenjaminiHochberg,
        "benjamini_yekutieli" => MultipleTestingMethod::BenjaminiYekutieli,
        "bonferroni" => MultipleTestingMethod::Bonferroni,
        "holm" => MultipleTestingMethod::Holm,
        _ => panic!("unknown method {name}"),
    }
}

fn close(actual: f64, expected: f64, atol: f64, rtol: f64) -> bool {
    (actual - expected).abs() <= atol + rtol * expected.abs()
}

#[test]
fn all_adjustment_methods_match_statsmodels() {
    let fixture = fixture();
    let atol = fixture["tolerance"]["atol"].as_f64().unwrap();
    let rtol = fixture["tolerance"]["rtol"].as_f64().unwrap();
    for case in fixture["cases"].as_array().unwrap() {
        let p_values: Vec<f64> =
            case["p_values"].as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect();
        for (name, output) in case["outputs"].as_object().unwrap() {
            let actual = adjust_pvalues(&p_values, method(name));
            let expected: Vec<f64> = output["adjusted"]
                .as_array()
                .unwrap()
                .iter()
                .map(|v| v.as_f64().unwrap())
                .collect();
            assert_eq!(actual.len(), expected.len());
            for (index, (&a, &e)) in actual.iter().zip(&expected).enumerate() {
                assert!(
                    close(a, e, atol, rtol),
                    "{} {name}[{index}] {a} != statsmodels {e}",
                    case["name"]
                );
            }
        }
    }
}

fn routing_links(routing: &JsonValue) -> Vec<ScoredLink> {
    routing["p_values"]
        .as_array()
        .unwrap()
        .iter()
        .zip(routing["source_lags"].as_array().unwrap())
        .enumerate()
        .map(|(index, (p, lag))| ScoredLink {
            link: LaggedLink {
                source: VariableId::from_raw(index as u32),
                source_lag: Lag::from_raw(lag.as_u64().unwrap() as u32),
                target: VariableId::from_raw(index as u32 + 20),
                target_lag: Lag::CONTEMPORANEOUS,
            },
            statistic: index as f64,
            p_value: p.as_f64().unwrap(),
            adjusted_p_value: None,
        })
        .collect()
}

#[test]
fn discovery_family_routing_matches_statsmodels_with_and_without_lag_zero() {
    let fixture = fixture();
    let routing = &fixture["discovery_routing"];
    let alpha = routing["alpha"].as_f64().unwrap();
    for exclude in [true, false] {
        let expected = &routing["exclude_contemporaneous"][exclude.to_string()];
        let config = FdrAdjustment {
            method: MultipleTestingMethod::BenjaminiHochberg,
            exclude_contemporaneous: exclude,
        };
        let adjusted = threshold_scored_links(routing_links(routing), Some(config), 1.0);
        let expected_adjusted = expected["adjusted_by_link"].as_array().unwrap();
        for link in &adjusted {
            let index = link.link.source.raw() as usize;
            match expected_adjusted[index].as_f64() {
                Some(value) => assert!(
                    close(link.adjusted_p_value.unwrap(), value, 1e-15, 1e-13),
                    "exclude={exclude}, index={index}"
                ),
                None => assert_eq!(link.adjusted_p_value, None),
            }
        }

        let retained = threshold_scored_links(routing_links(routing), Some(config), alpha);
        let retained_indices: Vec<u64> =
            retained.iter().map(|link| u64::from(link.link.source.raw())).collect();
        let expected_indices: Vec<u64> = expected["retained_indices"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().unwrap())
            .collect();
        assert_eq!(retained_indices, expected_indices, "exclude_contemporaneous={exclude}");
    }
}

#[test]
fn multiplicity_fixture_records_pins_and_no_generator() {
    let fixture = fixture();
    let oracle = &fixture["oracle"];
    assert_eq!(oracle["packages"]["statsmodels"]["version"], "0.14.4");
    assert_eq!(oracle["packages"]["numpy"]["version"], "2.1.3");
    assert_eq!(oracle["generation_location"], "temporary external harness; not retained");
    for package in ["statsmodels", "numpy", "scipy"] {
        assert_eq!(oracle["packages"][package]["metadata_sha256"].as_str().unwrap().len(), 64);
    }
}
