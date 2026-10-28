//! The two-sample mechanism discrepancy diagnostic against exact rational algebra and 50-digit
//! evaluation of its stated construction (`oracle_data/mechanism_discrepancy_reference.py`).
//!
//! Thirty seeded designs of one to three parents (observations on a binary grid, so the
//! sufficient statistics are exact): coefficients, the Wald statistic and its chi-square tail,
//! per-coefficient standard errors, z, p, Holm-adjusted p and minimal detectable differences, with
//! and without the intercept in the compared set and at three levels and powers.

use antecedent_estimate::mechanism_discrepancy::{
    DiscrepancyOptions, MechanismMeasurement, ParentSpec, PopulationSample, SampleDependence,
    test_mechanism_discrepancy,
};
use serde_json::Value;

fn document() -> Value {
    serde_json::from_str(include_str!("oracle_data/mechanism_discrepancy_reference.json"))
        .expect("frozen reference")
}

fn reference() -> Vec<Value> {
    document()["cases"].as_array().expect("cases").clone()
}

fn text(value: &Value) -> f64 {
    value.as_str().expect("text").parse().expect("number")
}

fn population(label: &str, x: &Value, y: &Value, k: usize) -> PopulationSample {
    let rows: Vec<Vec<f64>> = x
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect())
        .collect();
    let names = ["a", "b", "c"];
    PopulationSample {
        label: label.into(),
        measurement: MechanismMeasurement {
            node: "v".into(),
            node_unit: "u".into(),
            parents: names[..k]
                .iter()
                .map(|n| ParentSpec { name: (*n).into(), unit: "u".into() })
                .collect(),
            protocol_id: "protocol".into(),
        },
        outcome: y.as_array().unwrap().iter().map(|v| v.as_f64().unwrap()).collect(),
        parent_values: (0..k).map(|j| rows.iter().map(|r| r[j]).collect()).collect(),
        unit_ids: vec![],
    }
}

fn near(got: f64, want: f64, tolerance: f64, what: &str) {
    assert!((got - want).abs() <= tolerance * want.abs().max(1e-12), "{what}: {got:e} vs {want:e}");
}

#[test]
fn fits_wald_holm_and_detectable_differences_equal_the_rational_reference() {
    let cases = reference();
    assert!(cases.len() >= 30);
    for (index, case) in cases.iter().enumerate() {
        let k = case["k"].as_u64().unwrap() as usize;
        let source = population("source", &case["x_source"], &case["y_source"], k);
        let target = population("target", &case["x_target"], &case["y_target"], k);
        let options = DiscrepancyOptions {
            compare_intercept: case["compare_intercept"].as_bool().unwrap(),
            alpha: case["alpha"].as_f64().unwrap(),
            power: case["power"].as_f64().unwrap(),
            dependence: SampleDependence::Independent,
        };
        let result = test_mechanism_discrepancy(&source, &target, &options).expect("result");
        let label = |what: &str| format!("case {index} {what}");
        for (got, want) in
            result.source.coefficients.iter().zip(case["beta_source"].as_array().unwrap())
        {
            near(*got, text(want), 1e-11, &label("source beta"));
        }
        for (got, want) in
            result.target.coefficients.iter().zip(case["beta_target"].as_array().unwrap())
        {
            near(*got, text(want), 1e-11, &label("target beta"));
        }
        assert_eq!(result.test.degrees_of_freedom, case["df"].as_u64().unwrap() as usize);
        near(result.test.statistic, text(&case["wald"]), 1e-10, &label("wald"));
        near(result.test.p_value, text(&case["wald_p"]), 1e-8, &label("wald p"));
        near(result.detectability_factor, text(&case["factor"]), 1e-9, &label("factor"));
        let rows = case["coefficients"].as_array().unwrap();
        assert_eq!(result.coefficients.len(), rows.len());
        for (got, want) in result.coefficients.iter().zip(rows) {
            near(got.difference, text(&want["difference"]), 1e-10, &label("difference"));
            near(got.standard_error, text(&want["se"]), 1e-10, &label("se"));
            near(got.z, text(&want["z"]), 1e-9, &label("z"));
            near(got.p_value, text(&want["p"]), 1e-8, &label("p"));
            near(got.p_holm, text(&want["p_holm"]), 1e-8, &label("holm"));
            near(got.minimal_detectable_difference, text(&want["mdd"]), 1e-8, &label("mdd"));
            assert_eq!(got.rejected, got.p_holm < options.alpha);
        }
    }
}

#[test]
fn a_large_response_offset_costs_digits_through_the_summary_and_is_bounded() {
    let document = document();
    for case in document["offset_cases"].as_array().unwrap() {
        let offset = case["offset"].as_f64().unwrap();
        let source = population("source", &case["x_source"], &case["y_source"], 1);
        let target = population("target", &case["x_target"], &case["y_target"], 1);
        let options = DiscrepancyOptions {
            compare_intercept: true,
            alpha: 0.05,
            power: 0.8,
            dependence: SampleDependence::Independent,
        };
        let outcome = test_mechanism_discrepancy(&source, &target, &options);
        let want = text(&case["wald"]);
        if offset >= 1e6 {
            // The residual sum of squares is below 1e-12 of y'y: the summary no longer holds
            // enough digits to separate it from rounding, and the diagnostic refuses.
            let error = outcome.expect_err("refused");
            assert!(error.to_string().contains("degenerate_covariance"), "{offset}: {error}");
            continue;
        }
        let result = outcome.expect("result");
        // y'y - beta'X'y cancels about log10(y'y / RSS) digits of a summary held in doubles.
        let tolerance = if offset <= 1e3 { 1e-8 } else { 1e-5 };
        near(result.test.statistic, want, tolerance, &format!("wald at offset {offset}"));
        near(
            result.coefficients[1].standard_error,
            text(&case["se_slope"]),
            tolerance,
            &format!("slope se at offset {offset}"),
        );
    }
}
