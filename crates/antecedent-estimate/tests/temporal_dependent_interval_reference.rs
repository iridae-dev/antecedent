//! Bootstrap-t interval over seeded unit resamples against an independent multiprecision
//! construction.
//!
//! `oracle_data/temporal_dependent_interval_reference.py` (frozen in the JSON beside it) builds
//! panels of complete binary two-step histories with unequal cell multiplicities and
//! unit-specific outcomes, derives each unit's score and the pooled estimate from the histories,
//! regenerates every unit draw from the product's replicate ids with its own `SplitMix64`, and
//! recomputes the original standard error, each resample's standard error, the pivots, the
//! type-7 quantiles and the equal-tailed endpoints at 50 digits, with a resample of exactly zero
//! variance dropped. The test regenerates the product's replicate stream, asserts it equals the
//! frozen one, and asserts every derived quantity of the product equals the reference. The
//! keyed replicate identity words themselves are checked by the golden vectors in
//! `temporal_dependent_interval.rs`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::cast_precision_loss,
    clippy::cast_possible_truncation,
    clippy::cast_sign_loss,
    reason = "reference fixtures use bounded counts and indices"
)]

use antecedent_core::ExecutionContext;
use antecedent_estimate::EstimationError;
use antecedent_estimate::temporal_dependent_interval::{
    BalancedTemporalEstimator, DependentInterval, DependentIntervalConfig, IntervalMethod,
    ObservedStateSequence, SequenceHistory, TemporalEstimator, TemporalUnitPanel, UnitHistories,
    dependent_unit_interval,
};
use antecedent_estimate::temporal_initial_state::{
    FixedStateQuery, InitialStateLaw, InitialStatePopulation, InitialStateSpec, MarginalizedQuery,
};
use serde_json::Value;

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/temporal_dependent_interval_reference.json"))
        .expect("frozen reference parses")
}

fn panel_of(spec: &Value) -> TemporalUnitPanel {
    let units = spec["units"]
        .as_array()
        .unwrap()
        .iter()
        .map(|unit| UnitHistories {
            unit_id: unit["unit_id"].as_u64().unwrap(),
            histories: unit["rows"]
                .as_array()
                .unwrap()
                .iter()
                .map(|row| {
                    let level = |k: usize| u32::try_from(row[k].as_u64().unwrap()).unwrap();
                    SequenceHistory::two_step(
                        row[0].as_u64().unwrap(),
                        level(1),
                        level(2),
                        level(3),
                        level(4),
                        row[5].as_f64().unwrap(),
                    )
                })
                .collect(),
        })
        .collect();
    TemporalUnitPanel::new(spec["snapshot"].as_str().unwrap(), Some(units)).unwrap()
}

fn sequence_of(estimator: &Value) -> Vec<u32> {
    estimator["sequence"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| u32::try_from(a.as_u64().unwrap()).unwrap())
        .collect()
}

fn estimator_of(estimator: &Value) -> BalancedTemporalEstimator {
    match estimator["kind"].as_str().unwrap() {
        "observed" => BalancedTemporalEstimator::Observed(ObservedStateSequence {
            sequence: sequence_of(estimator),
        }),
        "fixed" => BalancedTemporalEstimator::Fixed(FixedStateQuery {
            sequence: sequence_of(estimator),
            s0: u32::try_from(estimator["s0"].as_u64().unwrap()).unwrap(),
        }),
        "marginalized" => {
            let states = estimator["law"]
                .as_array()
                .unwrap()
                .iter()
                .map(|pair| {
                    (u32::try_from(pair[0].as_u64().unwrap()).unwrap(), pair[1].as_f64().unwrap())
                })
                .collect();
            let law = InitialStateLaw::new(InitialStatePopulation::Target, "target-state", states)
                .unwrap();
            BalancedTemporalEstimator::Marginalized(
                MarginalizedQuery::new(sequence_of(estimator), InitialStateSpec::Law(law)).unwrap(),
            )
        }
        other => panic!("unknown estimator {other}"),
    }
}

fn config_of(config: &Value) -> DependentIntervalConfig {
    DependentIntervalConfig {
        replicates: usize::try_from(config["replicates"].as_u64().unwrap()).unwrap(),
        seed: config["seed"].as_u64().unwrap(),
        level: config["level"].as_f64().unwrap(),
        method: IntervalMethod::Studentized,
        min_units: usize::try_from(config["min_units"].as_u64().unwrap()).unwrap(),
        max_failed_fraction: config["max_failed_fraction"].as_f64().unwrap(),
    }
}

fn refusal_message(error: EstimationError) -> (&'static str, String) {
    match error {
        EstimationError::Refused { code, message } => (code, message.into_string()),
        other => panic!("expected a registered refusal, got {other:?}"),
    }
}

fn run(data: &Value, scenario: &Value) -> Result<DependentInterval, EstimationError> {
    let panel = panel_of(&data["panels"][scenario["panel"].as_str().unwrap()]);
    dependent_unit_interval(
        &panel,
        &estimator_of(&scenario["estimator"]),
        &config_of(&scenario["config"]),
        &ctx(),
    )
}

/// Regenerates the frozen replicate stream: reads the scenarios the reference script prints with
/// `inputs` from `INTERVAL_INPUTS` and writes each scenario's replicate ids and points to
/// `INTERVAL_STREAM`.
#[test]
#[ignore = "regenerates the frozen replicate stream of the reference; run by hand"]
fn dump_temporal_interval_stream() {
    let path = std::env::var("INTERVAL_INPUTS").expect("INTERVAL_INPUTS");
    let out = std::env::var("INTERVAL_STREAM").expect("INTERVAL_STREAM");
    let inputs: Value = serde_json::from_str(&std::fs::read_to_string(path).unwrap()).unwrap();
    let mut dumped = serde_json::Map::new();
    for scenario in inputs["scenarios"].as_array().unwrap() {
        let mut permissive = scenario.clone();
        permissive["config"]["max_failed_fraction"] = serde_json::json!(0.99);
        let interval = run(&inputs, &permissive).unwrap();
        dumped.insert(
            scenario["name"].as_str().unwrap().to_owned(),
            serde_json::json!({
                "replicate_ids": interval.replicates.iter().map(|r| r.replicate_id).collect::<Vec<_>>(),
                "points": interval.replicates.iter().map(|r| r.point).collect::<Vec<_>>(),
            }),
        );
    }
    std::fs::write(out, serde_json::to_string(&Value::Object(dumped)).unwrap()).unwrap();
}

fn close(got: f64, want: f64, what: &str) {
    assert!((got - want).abs() <= 1e-9 * want.abs().max(1.0), "{what}: {got} vs {want}");
}

fn parsed(value: &Value) -> Option<f64> {
    value.as_str().map(|text| text.parse::<f64>().unwrap())
}

#[test]
fn bootstrap_t_scores_standard_errors_pivots_and_endpoints_equal_the_multiprecision_construction() {
    let data = reference();
    let mut dropped_somewhere = 0;
    for scenario in data["scenarios"].as_array().unwrap() {
        let name = scenario["name"].as_str().unwrap();
        let interval = run(&data, scenario).unwrap();
        let expected = &scenario["expected"];
        let stream = &scenario["stream"];

        // The regenerated replicate stream is the frozen one.
        let ids = stream["replicate_ids"].as_array().unwrap();
        let points = stream["points"].as_array().unwrap();
        assert_eq!(interval.replicates.len(), ids.len(), "{name}");
        for ((record, id), point) in interval.replicates.iter().zip(ids).zip(points) {
            assert_eq!(record.replicate_id, id.as_u64().unwrap(), "{name}: replicate id");
            close(record.point.unwrap(), point.as_f64().unwrap(), "replicate point");
        }

        close(interval.point, expected["point"].as_f64().unwrap(), "point");
        let receipt = interval.studentization.as_ref().expect("studentization receipt");
        let scores = expected["unit_scores"].as_array().unwrap();
        assert_eq!(receipt.unit_scores.len(), scores.len(), "{name}");
        for (got, want) in receipt.unit_scores.iter().zip(scores) {
            close(*got, want.as_f64().unwrap(), "unit score");
        }
        close(receipt.standard_error, parsed(&expected["standard_error"]).unwrap(), "original se");

        let errors = expected["replicate_standard_errors"].as_array().unwrap();
        let pivots = expected["pivots"].as_array().unwrap();
        assert_eq!(receipt.replicate_standard_errors.len(), errors.len(), "{name}");
        for (got, want) in receipt.replicate_standard_errors.iter().zip(errors) {
            match (got, parsed(want)) {
                (Some(g), Some(w)) => close(*g, w, "resample se"),
                (None, None) => {}
                (g, w) => panic!("{name}: resample se {g:?} vs reference {w:?}"),
            }
        }
        for (got, want) in receipt.pivots.iter().zip(pivots) {
            match (got, parsed(want)) {
                (Some(g), Some(w)) => close(*g, w, "pivot"),
                (None, None) => {}
                (g, w) => panic!("{name}: pivot {g:?} vs reference {w:?}"),
            }
        }
        assert_eq!(interval.failed as u64, expected["failed"].as_u64().unwrap(), "{name}");
        dropped_somewhere += interval.failed;
        close(interval.lower, parsed(&expected["lower"]).unwrap(), "lower endpoint");
        close(interval.upper, parsed(&expected["upper"]).unwrap(), "upper endpoint");
        assert!((interval.level - scenario["config"]["level"].as_f64().unwrap()).abs() < 1e-15);
        assert_eq!(
            interval.units,
            data["panels"][scenario["panel"].as_str().unwrap()]["units"].as_array().unwrap().len()
        );
        assert!(interval.lower < interval.point && interval.point < interval.upper, "{name}");
    }
    assert!(dropped_somewhere > 0, "the tied-score scenario must exercise the dropped resamples");
}

#[test]
fn the_declared_level_changes_the_endpoints_and_the_replicate_count_is_the_stream_length() {
    let data = reference();
    let scenario = &data["scenarios"][0];
    let narrow = {
        let mut changed = scenario.clone();
        changed["config"]["level"] = serde_json::json!(0.5);
        run(&data, &changed).unwrap()
    };
    let wide = run(&data, scenario).unwrap();
    assert!(narrow.width() < wide.width(), "{} vs {}", narrow.width(), wide.width());
    assert_eq!(wide.replicates.len(), 200);
    // The nominal level reaches the receipt and the calibration basis counts the kept replicates.
    assert!((narrow.level - 0.5).abs() < 1e-15);
    assert_eq!(narrow.calibration_basis().replicates_ok, Some(200));
}

#[test]
fn dropped_zero_variance_resamples_are_counted_and_bounded_by_the_declared_fraction() {
    let data = reference();
    let tied = data["scenarios"]
        .as_array()
        .unwrap()
        .iter()
        .find(|s| s["name"] == "tied_scores_b")
        .unwrap();
    let failed = tied["expected"]["failed"].as_u64().unwrap();
    assert!(failed > 0);
    let mut strict = tied.clone();
    strict["config"]["max_failed_fraction"] = serde_json::json!(0.0);
    let (code, message) = refusal_message(run(&data, &strict).unwrap_err());
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("temporal_interval.unsupported_history"), "{message}");
    assert!(message.contains(&format!("{failed} of 200")), "{message}");

    // A tolerated fraction just above the observed one passes, one just below refuses.
    let observed = failed as f64 / 200.0;
    let mut at = tied.clone();
    at["config"]["max_failed_fraction"] = serde_json::json!(observed);
    assert_eq!(run(&data, &at).unwrap().failed as u64, failed);
    let mut under = tied.clone();
    under["config"]["max_failed_fraction"] = serde_json::json!(observed - 1e-9);
    assert!(run(&data, &under).is_err());
}

#[test]
fn the_declared_minimum_unit_count_is_read_and_never_drops_below_two() {
    let data = reference();
    let mut scenario = data["scenarios"][0].clone();
    scenario["config"]["min_units"] = serde_json::json!(25);
    let (code, message) = refusal_message(run(&data, &scenario).unwrap_err());
    assert_eq!(code, "too_few_clusters");
    assert!(message.contains("24 repeated units, at least 25 required"), "{message}");
    // A requested minimum of zero or one is raised to two: one unit has no variance to resample.
    let one = {
        let mut panel = data["panels"]["A"].clone();
        panel["units"].as_array_mut().unwrap().truncate(1);
        panel
    };
    let single = panel_of(&one);
    let estimator = estimator_of(&data["scenarios"][0]["estimator"]);
    let mut config = config_of(&data["scenarios"][0]["config"]);
    config.min_units = 0;
    let (code, message) =
        refusal_message(dependent_unit_interval(&single, &estimator, &config, &ctx()).unwrap_err());
    assert_eq!(code, "too_few_clusters");
    assert!(message.contains("at least 2 required"), "{message}");
}

#[test]
fn a_studentized_request_beyond_the_certified_unit_range_is_refused_not_rerouted() {
    struct Counting(std::cell::Cell<usize>);
    impl TemporalEstimator for Counting {
        fn label(&self) -> &'static str {
            "counting"
        }
        fn estimate(&self, units: &[&UnitHistories]) -> Result<f64, EstimationError> {
            self.0.set(self.0.get() + 1);
            Ok(units.iter().map(|u| u.histories[0].y).sum::<f64>() / units.len() as f64)
        }
    }
    let units = (0..4097_u64)
        .map(|id| UnitHistories {
            unit_id: id,
            histories: vec![SequenceHistory::two_step(0, 0, 1, 0, 1, (id % 7) as f64)],
        })
        .collect();
    let panel = TemporalUnitPanel::new("large-panel", Some(units)).unwrap();
    let estimator = Counting(std::cell::Cell::new(0));
    let config = DependentIntervalConfig {
        replicates: 20,
        method: IntervalMethod::Studentized,
        ..DependentIntervalConfig::default()
    };
    let error = dependent_unit_interval(&panel, &estimator, &config, &ctx())
        .expect_err("a request for the bootstrap-t must not return another construction");
    let (code, message) = refusal_message(error);
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("temporal_interval.studentized_bounds_exceeded"), "{message}");
    assert_eq!(estimator.0.get(), 0, "no resampling work before the refusal");
}
