//! The adjusted-effect regression of the sample-split routes against exact rational least squares.
//!
//! `oracle_data/sample_split_effect_reference.py` (frozen in the JSON beside it) solves the normal
//! equations of `y ~ 1 + x + z...` over the rationals for the exact binary values given to the
//! product and evaluates the Student quantile at 60 digits. The product's `adjusted_effect` is the
//! numerical kernel of both the discovered-graph effect and the parent-set effect; it must
//! reproduce the coefficient, standard error, degrees of freedom and interval at several levels,
//! including columns offset by 1e6 to 1e9, a covariate within 1e-4 and 1e-6 of the treatment, the
//! smallest admissible sample and an exactly collinear design, which it must refuse.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_data::TabularData;
use antecedent_io::measured_inference::split_effect::adjusted_effect;
use serde_json::Value;

fn column(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn rel_err(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(f64::MIN_POSITIVE)
}

#[test]
fn adjusted_effect_equals_exact_least_squares_including_offsets_and_near_collinearity() {
    let cases: Vec<Value> =
        serde_json::from_str(include_str!("oracle_data/sample_split_effect_reference.json"))
            .unwrap();
    assert_eq!(cases.len(), 9);
    let mut worst: Vec<(String, f64)> = Vec::new();
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let y = column(&case["y"]);
        let x = column(&case["x"]);
        let zs: Vec<Vec<f64>> = case["z"].as_array().unwrap().iter().map(column).collect();
        let mut names = vec!["y".to_owned(), "x".to_owned()];
        names.extend((0..zs.len()).map(|j| format!("z{j}")));
        let mut columns = vec![y, x];
        columns.extend(zs.iter().cloned());
        let frame = TabularData::from_f64_columns(
            names.iter().map(String::as_str).zip(columns.iter().map(Vec::as_slice)),
        )
        .unwrap();
        let adjustment: Vec<u32> = (0..zs.len() as u32).map(|j| j + 2).collect();
        if case["singular"].as_bool().unwrap() {
            assert!(
                adjusted_effect(&frame, 1, 0, &adjustment, 0.95).is_err(),
                "{name}: a design that is singular in exact arithmetic must be refused"
            );
            continue;
        }
        let mut case_worst = 0.0_f64;
        for interval in case["intervals"].as_array().unwrap() {
            let level = interval["level"].as_f64().unwrap();
            let got = adjusted_effect(&frame, 1, 0, &adjustment, level)
                .unwrap_or_else(|e| panic!("{name}: {e:?}"));
            assert_eq!(got.degrees_of_freedom, case["df"].as_f64().unwrap(), "{name}: df");
            assert_eq!(got.rows, case["rows"].as_u64().unwrap(), "{name}: rows");
            let errors = [
                rel_err(got.point, case["point"].as_f64().unwrap()),
                rel_err(got.standard_error, case["se"].as_f64().unwrap()),
                rel_err(got.lower, interval["lower"].as_f64().unwrap()),
                rel_err(got.upper, interval["upper"].as_f64().unwrap()),
            ];
            case_worst = case_worst.max(errors.iter().copied().fold(0.0, f64::max));
        }
        worst.push((name.to_owned(), case_worst));
    }
    let failures: Vec<String> = worst
        .iter()
        .filter(|(_, e)| *e > 1e-6)
        .map(|(n, e)| format!("{n}: relative error {e:.3e}"))
        .collect();
    assert!(
        failures.is_empty(),
        "adjusted effect departs from exact least squares: {failures:?} (all: {worst:?})"
    );
}

/// The whole route on frozen data: the seeded split equals the definition (splitmix64-driven
/// Fisher-Yates, reproduced in the reference), discovery orients the treatment's collider so the
/// adjustment set is the true parents, and the interval equals exact least squares on the
/// reference's analysis rows.
#[test]
fn the_split_effect_route_equals_the_split_by_definition_and_exact_least_squares() {
    use antecedent_core::{ExecutionContext, VariableId};
    use antecedent_discovery::selective::reuse::{SplitRule, split_rows};
    use antecedent_io::measured_inference::split_effect::{
        DISCOVERY_FRACTION, SplitEffectSource, run,
    };

    let doc: Value =
        serde_json::from_str(include_str!("oracle_data/sample_split_pipeline_reference.json"))
            .unwrap();
    let case = &doc["effect"];
    let columns: Vec<Vec<f64>> = case["columns"].as_array().unwrap().iter().map(column).collect();
    let names: Vec<String> = (0..columns.len()).map(|i| format!("v{i}")).collect();
    let seed = case["split_seed"].as_u64().unwrap();
    let source = SplitEffectSource::new(
        names.clone(),
        columns.clone(),
        case["treatment"].as_u64().unwrap() as u32,
        case["outcome"].as_u64().unwrap() as u32,
        seed,
        case["discovery_seed"].as_u64().unwrap(),
    );

    let frame = TabularData::from_f64_columns(
        names.iter().map(String::as_str).zip(columns.iter().map(Vec::as_slice)),
    )
    .unwrap();
    let variables: Vec<VariableId> = (0..columns.len() as u32).map(VariableId::from_raw).collect();
    let split = split_rows(
        &frame,
        &variables,
        SplitRule::SeededPermutation { seed, discovery_fraction: DISCOVERY_FRACTION },
    )
    .unwrap();
    let expected_rows: Vec<usize> = case["analysis_row_ids"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as usize)
        .collect();
    assert_eq!(
        split.analysis_row_ids, expected_rows,
        "the seeded split departs from its definition"
    );

    let ran = run(&source, &ExecutionContext::for_tests(1)).unwrap();
    assert_eq!(ran.analysis_rows, case["analysis_rows"].as_u64().unwrap());
    assert_eq!(ran.discovery_rows, case["discovery_rows"].as_u64().unwrap());
    let expected_adjustment: Vec<u32> = case["expected_adjustment"]
        .as_array()
        .unwrap()
        .iter()
        .map(|v| v.as_u64().unwrap() as u32)
        .collect();
    assert_eq!(ran.effect.adjustment_set, expected_adjustment);
    let reference = &case["reference"];
    assert_eq!(ran.effect.degrees_of_freedom, reference["df"].as_f64().unwrap());
    for (got, key) in [
        (ran.effect.point, "point"),
        (ran.effect.standard_error, "se"),
        (ran.effect.lower, "lower"),
        (ran.effect.upper, "upper"),
    ] {
        assert!(rel_err(got, reference[key].as_f64().unwrap()) < 1e-9, "{key}: {got}");
    }
}
