//! The numerical statistics of the target-native refuters against exact rational and 50-digit
//! evaluation of their definitions (`oracle_data/target_native_numerics_reference.py`):
//! the joint Mahalanobis distance with its chi-square tail, the Richardson-extrapolated central
//! difference with its truncation and rounding standardisation, and the local-linearity measure.
//!
//! The response in the derivative checks is a lookup of the exact function values the reference
//! used, so each comparison isolates the statistic and not the test function's rounding.

use antecedent_validate::target_native::{
    DerivativeCheckPoint, finite_difference_consistency, joint_mahalanobis, local_linearity,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/target_native_numerics_reference.json"))
        .expect("frozen reference")
}

fn floats(value: &Value) -> Vec<f64> {
    value.as_array().expect("array").iter().map(|v| v.as_f64().expect("number")).collect()
}

fn text(value: &Value) -> f64 {
    value.as_str().expect("text").parse().expect("number")
}

fn relative(got: f64, want: f64) -> f64 {
    (got - want).abs() / want.abs().max(1e-300)
}

#[test]
fn joint_mahalanobis_equals_the_rational_quadratic_form_and_its_chi_square_tail() {
    let reference = reference();
    let cases = reference["joint"].as_array().expect("cases");
    assert!(cases.len() >= 24);
    for (index, case) in cases.iter().enumerate() {
        let k = case["k"].as_u64().unwrap() as usize;
        let stat =
            joint_mahalanobis(&floats(&case["expected"]), &floats(&case["replicates"]), k).unwrap();
        let (d2, p) = (text(&case["d2"]), text(&case["p"]));
        // A conditioning-limited quadratic form: the covariance is formed in doubles.
        let tolerance = if index + 1 == cases.len() { 1e-5 } else { 1e-9 };
        assert!(relative(stat.value, d2) <= tolerance, "case {index}: {} vs {d2}", stat.value);
        assert_eq!(stat.df, k);
        let got = stat.p_value.expect("p-value");
        assert!(
            (got - p).abs() <= 100.0 * tolerance * p.max(1e-12) + 1e-14,
            "case {index}: p {got:e} vs {p:e} (D2 {d2})"
        );
    }
}

fn lookup(values: &Value, x: f64, point: f64, step: f64) -> f64 {
    let key = if x == point + step {
        "p"
    } else if x == point - step {
        "m"
    } else if x == point + step / 2.0 {
        "p2"
    } else if x == point - step / 2.0 {
        "m2"
    } else if x == point {
        "c"
    } else {
        panic!("unexpected abscissa {x}")
    };
    values[key].as_f64().expect("value")
}

#[test]
fn richardson_difference_truncation_and_standardisation_equal_their_definitions() {
    let reference = reference();
    for case in reference["finite_difference"].as_array().unwrap() {
        let (x, step) = (case["x"].as_f64().unwrap(), case["step"].as_f64().unwrap());
        let check = finite_difference_consistency(
            |t| Ok(lookup(&case["values"], t, x, step)),
            &[DerivativeCheckPoint { x, reported: case["reported"].as_f64().unwrap() }],
            step,
            case["tolerance"].as_f64().unwrap(),
        )
        .unwrap();
        let want = case["richardson"].as_f64().unwrap();
        assert!(relative(check.finite_difference[0], want) <= 1e-12, "{case}");
        assert!(
            (check.truncation_error[0] - case["truncation"].as_f64().unwrap()).abs()
                <= 1e-12 * want.abs().max(1.0),
            "{case}"
        );
        let std = case["standardized"].as_f64().unwrap();
        // The numerator is a difference of two doubles near the derivative, so its rounding
        // (a few units of eps in the derivative) over the scale (at least tolerance) bounds the error.
        let tolerance = case["tolerance"].as_f64().unwrap();
        let bound = 64.0 * f64::EPSILON / tolerance + 1e-9 * std;
        assert!(
            (check.standardized[0] - std).abs() <= bound,
            "{} vs {std}: {case}",
            check.standardized[0]
        );
        assert_eq!(check.passed(), std <= 1.0);
    }
}

#[test]
fn local_linearity_equals_the_second_difference_definition() {
    let reference = reference();
    for case in reference["local_linearity"].as_array().unwrap() {
        let (x, step) = (case["x"].as_f64().unwrap(), case["step"].as_f64().unwrap());
        let result = local_linearity(
            |t| Ok(lookup(&case["values"], t, x, step)),
            &[x],
            step,
            0.1,
            case["floor"].as_f64().unwrap(),
        )
        .unwrap();
        assert!(relative(result.slope[0], case["slope"].as_f64().unwrap()) <= 1e-12, "{case}");
        let curvature = case["curvature"].as_f64().unwrap();
        assert!(
            (result.curvature[0] - curvature).abs() <= 1e-9 * curvature.abs().max(1.0),
            "{case}"
        );
        let nl = case["nonlinearity"].as_f64().unwrap();
        assert!(
            relative(result.nonlinearity[0], nl) <= 1e-8,
            "{} vs {nl}: {case}",
            result.nonlinearity[0]
        );
    }
}
