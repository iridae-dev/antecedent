//! Gaussian-kernel local quadratic regression against 60-digit evaluation of its definition
//! (`oracle_data/local_quadratic_precision_reference.py`) across scales: doses spread over 1e-4,
//! 1 and 1e4, bandwidths from a tenth of the spread to fifty times it, a response offset of 1e6, a
//! dose offset of 1e6 and heavily tied doses. The level, slope and curvature, the observation
//! influences and robust standard error, the plug-in standard errors and the effective size are
//! compared, so the unit-scale checks of `local_quadratic_reference.rs` also hold off the unit scale.

use antecedent_stats::gaussian_local_quadratic_influence;
use serde_json::Value;

fn reference() -> Vec<Value> {
    serde_json::from_str(include_str!("oracle_data/local_quadratic_precision_reference.json"))
        .expect("frozen reference")
}

fn text(value: &Value) -> f64 {
    value.as_str().expect("text").parse().expect("number")
}

fn floats(value: &Value) -> Vec<f64> {
    value.as_array().expect("array").iter().map(|v| v.as_f64().expect("number")).collect()
}

#[test]
fn estimates_influences_and_standard_errors_equal_the_high_precision_definition() {
    let cases = reference();
    assert!(cases.len() >= 14);
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let fit = gaussian_local_quadratic_influence(
            &floats(&case["x"]),
            &floats(&case["y"]),
            case["at"].as_f64().unwrap(),
            case["bandwidth"].as_f64().unwrap(),
        )
        .expect(name);
        let check = |what: &str, got: f64, want: f64| {
            let tolerance = 1e-10;
            assert!(
                (got - want).abs() <= tolerance * want.abs().max(f64::MIN_POSITIVE),
                "{name} {what}: {got:e} vs {want:e}"
            );
        };
        check("level", fit.point.value, text(&case["value"]));
        check("slope", fit.point.first_derivative, text(&case["slope"]));
        check("curvature", fit.point.second_derivative, text(&case["curvature"]));
        check("robust se", fit.robust_standard_error, text(&case["robust_se"]));
        check("plug-in se", fit.point.standard_error, text(&case["plugin_se"]));
        check(
            "plug-in slope se",
            fit.point.first_derivative_standard_error,
            text(&case["plugin_slope_se"]),
        );
        check("ess", fit.point.local_ess, text(&case["ess"]));
        let want: Vec<f64> = case["influences"].as_array().unwrap().iter().map(text).collect();
        let scale = want.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        for (g, w) in fit.influences.iter().zip(&want) {
            assert!((g - w).abs() <= 1e-10 * scale, "{name} influence {g:e} vs {w:e}");
        }
    }
}
