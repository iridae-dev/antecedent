//! Misclassification corrections against exact-rational and 60-digit evaluation of their
//! definitions (`oracle_data/measurement_correction_reference.py`).
//!
//! Binary: corrected prevalence and risk difference (shared and per-arm rates) and the delta-method
//! variance split into its sampling and rate parts, including a perfect test, a Youden index of
//! 0.15 and counts of zero and all. Categorical: corrected proportions, outcome means, both
//! covariances and the 1-norm condition number for two to four categories, against an analytic
//! Jacobian evaluated at 60 digits. A two-category correction must also reproduce the binary one.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::measurement_error::{
    BinaryErrorRates, ConfusionMatrix, ObservedClassSummary, TwoByTwoCounts, corrected_categories,
    corrected_prevalence, corrected_risk_difference_differential,
    corrected_risk_difference_nondifferential,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/measurement_correction_reference.json"))
        .expect("frozen reference")
}

fn number(value: &Value) -> f64 {
    match value {
        Value::String(text) => text.parse().expect("number"),
        other => other.as_f64().expect("number"),
    }
}

fn numbers(value: &Value) -> Vec<f64> {
    value.as_array().expect("array").iter().map(number).collect()
}

fn close(got: f64, want: f64, relative: f64, what: &str) {
    assert!((got - want).abs() <= 1e-15 + relative * want.abs(), "{what}: {got} vs {want}");
}

fn rates(values: &[f64]) -> BinaryErrorRates {
    BinaryErrorRates::new(values[0], values[1], values[2], values[3]).expect("rates")
}

#[test]
fn corrected_prevalence_matches_exact_rational_arithmetic() {
    let reference = reference();
    for case in reference["prevalence"].as_array().expect("prevalence") {
        let r = BinaryErrorRates::new(
            number(&case["se"]),
            number(&case["sp"]),
            number(&case["vse"]),
            number(&case["vsp"]),
        )
        .expect("rates");
        let got = corrected_prevalence(
            case["events"].as_u64().expect("events"),
            case["total"].as_u64().expect("total"),
            &r,
            0.01,
        )
        .expect("prevalence");
        close(got.corrected, number(&case["corrected"]), 1e-13, &format!("{case}"));
        close(got.youden, number(&case["youden"]), 1e-13, &format!("{case}"));
        close(got.variance, number(&case["variance"]), 1e-12, &format!("{case}"));
        close(got.variance_sampling, number(&case["sampling"]), 1e-12, &format!("{case}"));
        close(got.variance_rates, number(&case["rates"]), 1e-12, &format!("{case}"));
        assert_eq!(got.outside_unit_interval, !(0.0..=1.0).contains(&number(&case["corrected"])));
    }
}

#[test]
fn risk_differences_match_exact_rational_arithmetic() {
    let reference = reference();
    for case in reference["risk_difference"].as_array().expect("risk_difference") {
        let t: Vec<u64> = case["table"]
            .as_array()
            .expect("table")
            .iter()
            .map(|v| v.as_u64().expect("count"))
            .collect();
        let table = TwoByTwoCounts {
            exposed_events: t[0],
            exposed_total: t[1],
            unexposed_events: t[2],
            unexposed_total: t[3],
        };
        let r = rates(&[
            number(&case["se"]),
            number(&case["sp"]),
            number(&case["vse"]),
            number(&case["vsp"]),
        ]);
        let got = corrected_risk_difference_nondifferential(&table, &r, 0.01).expect("difference");
        close(got.corrected, number(&case["corrected"]), 1e-13, &format!("{case}"));
        close(got.variance, number(&case["variance"]), 1e-12, &format!("{case}"));
        close(got.variance_sampling, number(&case["sampling"]), 1e-12, &format!("{case}"));
        close(got.variance_rates, number(&case["rates"]), 1e-12, &format!("{case}"));
        close(got.risk_exposed, number(&case["risk_exposed"]), 1e-13, &format!("{case}"));
        close(got.risk_unexposed, number(&case["risk_unexposed"]), 1e-13, &format!("{case}"));
    }
    for case in reference["risk_difference_differential"].as_array().expect("differential") {
        let t: Vec<u64> = case["table"]
            .as_array()
            .expect("table")
            .iter()
            .map(|v| v.as_u64().expect("count"))
            .collect();
        let table = TwoByTwoCounts {
            exposed_events: t[0],
            exposed_total: t[1],
            unexposed_events: t[2],
            unexposed_total: t[3],
        };
        let got = corrected_risk_difference_differential(
            &table,
            &rates(&numbers(&case["exposed"])),
            &rates(&numbers(&case["unexposed"])),
            0.01,
        )
        .expect("difference");
        close(got.corrected, number(&case["corrected"]), 1e-13, &format!("{case}"));
        close(got.variance, number(&case["variance"]), 1e-12, &format!("{case}"));
        close(got.variance_sampling, number(&case["sampling"]), 1e-12, &format!("{case}"));
        close(got.variance_rates, number(&case["rates"]), 1e-12, &format!("{case}"));
    }
}

fn categorical_inputs(case: &Value) -> (Vec<ObservedClassSummary>, ConfusionMatrix) {
    let counts: Vec<u64> = case["counts"]
        .as_array()
        .expect("counts")
        .iter()
        .map(|v| v.as_u64().expect("count"))
        .collect();
    let means = numbers(&case["means"]);
    let variances = numbers(&case["variances"]);
    let classes = (0..counts.len())
        .map(|i| ObservedClassSummary { count: counts[i], mean: means[i], variance: variances[i] })
        .collect();
    let validation: Vec<u64> = case["validation"]
        .as_array()
        .expect("validation")
        .iter()
        .map(|v| v.as_u64().expect("count"))
        .collect();
    let confusion = ConfusionMatrix::from_validation_counts(
        case["levels"].as_u64().expect("levels") as usize,
        &validation,
    )
    .expect("confusion");
    (classes, confusion)
}

#[test]
fn categorical_corrections_match_the_analytic_delta_method() {
    let reference = reference();
    for case in reference["categorical"].as_array().expect("categorical") {
        let (classes, confusion) = categorical_inputs(case);
        let got = corrected_categories(&classes, &confusion, 1e6).expect("correction");
        let scale = |want: &[f64]| want.iter().fold(0.0_f64, |m, v| m.max(v.abs())).max(1e-300);
        for (name, got, want) in [
            ("proportions", &got.proportions, numbers(&case["proportions"])),
            ("outcome means", &got.outcome_means, numbers(&case["outcome_means"])),
            (
                "proportion covariance",
                &got.proportion_covariance,
                numbers(&case["proportion_covariance"]),
            ),
            (
                "outcome covariance",
                &got.outcome_mean_covariance,
                numbers(&case["outcome_mean_covariance"]),
            ),
        ] {
            let tolerance = if name.contains("covariance") { 1e-8 } else { 1e-11 };
            let s = scale(&want);
            for (g, w) in got.iter().zip(&want) {
                assert!((g - w).abs() <= tolerance * s, "{name} {case}: {g} vs {w}");
            }
        }
        close(got.condition_number, number(&case["condition"]), 1e-12, &format!("{case}"));
    }
}

#[test]
fn two_categories_reproduce_the_binary_correction() {
    // Class 1 is "positive": sensitivity C[1][1], specificity C[0][0].
    let validation = [45_u64, 15, 5, 135]; // observed x true: true 0 column (45, 5), true 1 column (15, 135)
    let confusion = ConfusionMatrix::from_validation_counts(2, &validation).expect("confusion");
    let classes = [
        ObservedClassSummary { count: 700, mean: 0.0, variance: 1.0 },
        ObservedClassSummary { count: 300, mean: 1.0, variance: 1.0 },
    ];
    let categorical = corrected_categories(&classes, &confusion, 1e6).expect("categorical");
    let binary_rates = BinaryErrorRates::from_validation_counts(135, 15, 45, 5).expect("rates");
    let binary = corrected_prevalence(300, 1000, &binary_rates, 0.01).expect("binary");
    close(categorical.proportions[1], binary.corrected, 1e-13, "corrected prevalence");
    close(
        categorical.proportion_covariance[3],
        binary.variance,
        1e-12,
        "variance of the prevalence",
    );
    close(categorical.proportions[0], 1.0 - binary.corrected, 1e-13, "complement");
}
