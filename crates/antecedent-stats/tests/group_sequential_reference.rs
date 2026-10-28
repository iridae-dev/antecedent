//! Group-sequential boundaries, stage-wise crossing probabilities, stage-wise p-values and
//! inference against an independent recursion (`oracle_data/group_sequential_reference.py`).
//!
//! The reference recurses on the score scale with its own Gauss-Legendre panels sized to the
//! narrowest look-to-look transition and 20-digit arithmetic, over seven information-fraction
//! designs (one to five looks; unequal, tightly clustered and nearly final looks), both spending
//! functions, both sidedness conventions, levels 0.05 and 0.001 (0.05 only for designs of more
//! than four looks), drifts 0 and 1.5 for crossing probabilities and 0 and 1 for stage-wise
//! p-values. Two-look probabilities are also compared with a direct integral of the bivariate
//! normal, down to looks at 0.999 and 1 of the information (where the transition has a standard
//! deviation of 0.03 and the panels must follow it), and looks too close for the quadrature are
//! refused.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_stats::adaptive_inference::{
    Sidedness, SpendingFunction, StagewiseLook, boundaries, crossing_probabilities,
    stagewise_inference, stagewise_upper_p_value,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/group_sequential_reference.json"))
        .expect("frozen reference")
}

fn never() -> bool {
    false
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

fn spending(label: &str) -> SpendingFunction {
    if label == "obf" { SpendingFunction::OBrienFlemingType } else { SpendingFunction::PocockType }
}

fn side(two: &Value) -> Sidedness {
    if two.as_bool() == Some(true) {
        Sidedness::TwoSidedSymmetric
    } else {
        Sidedness::OneSidedUpper
    }
}

#[test]
fn boundaries_and_crossing_probabilities_match_the_independent_recursion() {
    let reference = reference();
    for design in reference["boundaries"].as_array().expect("designs") {
        let fractions = numbers(&design["fractions"]);
        let alpha = design["alpha"].as_f64().expect("alpha");
        let sidedness = side(&design["two_sided"]);
        let label = format!(
            "{fractions:?} alpha {alpha} {} two_sided {}",
            design["spending"], design["two_sided"]
        );
        let got = boundaries(
            &fractions,
            alpha,
            spending(design["spending"].as_str().expect("spending")),
            sidedness,
            &never,
        )
        .unwrap_or_else(|e| panic!("{label}: {e}"));
        let want = numbers(&design["critical"]);
        for (k, (g, w)) in got.critical.iter().zip(&want).enumerate() {
            assert!((g - w).abs() <= 1e-9, "{label}: boundary {k}: {g} vs {w}");
        }
        for drift in design["drifts"].as_array().expect("drifts") {
            let d = drift["drift"].as_f64().expect("drift");
            let probabilities = crossing_probabilities(&fractions, &want, sidedness, d, &never)
                .unwrap_or_else(|e| panic!("{label}: {e}"));
            for (k, (g, w)) in
                probabilities.iter().zip(numbers(&drift["probabilities"])).enumerate()
            {
                assert!((g - w).abs() <= 1e-9, "{label} drift {d}: look {k}: {g} vs {w}");
            }
        }
        // With no drift the crossings are the increments of the spending function.
        for (g, s) in crossing_probabilities(&fractions, &got.critical, sidedness, 0.0, &never)
            .expect("null crossings")
            .iter()
            .zip(&got.spend_increments)
        {
            assert!((g - s).abs() <= 1e-9 * s.max(1e-3), "{label}: spent {s}, crossed {g}");
        }
    }
}

#[test]
fn stagewise_p_values_match_the_independent_recursion() {
    let reference = reference();
    for case in reference["stagewise"].as_array().expect("stagewise") {
        let fractions = numbers(&case["fractions"]);
        let critical = numbers(&case["critical"]);
        let look = case["look"].as_u64().expect("look") as usize;
        let observed = StagewiseLook { look, statistic: case["statistic"].as_f64().expect("stat") };
        let got = stagewise_upper_p_value(
            &fractions,
            &critical,
            side(&case["two_sided"]),
            observed,
            case["drift"].as_f64().expect("drift"),
            &never,
        )
        .expect("p-value");
        let want = number(&case["p"]);
        assert!((got - want).abs() <= 1e-9, "{case}: {got} vs {want}");
    }
}

#[test]
fn two_look_probabilities_match_the_bivariate_normal_integral() {
    let reference = reference();
    for case in reference["two_look"].as_array().expect("two_look") {
        let fractions = numbers(&case["fractions"]);
        let critical = numbers(&case["critical"]);
        let got = crossing_probabilities(
            &fractions,
            &critical,
            Sidedness::TwoSidedSymmetric,
            case["drift"].as_f64().expect("drift"),
            &never,
        )
        .expect("crossings")[1];
        let want = number(&case["exact"]);
        assert!((got - want).abs() <= 1e-9, "{case}: product {got} vs bivariate integral {want}");
        let recursion = number(&case["recursion"]);
        assert!((recursion - want).abs() <= 1e-9, "the reference recursion itself: {case}");
    }
}

#[test]
fn stagewise_limits_solve_their_defining_equations() {
    let fractions = [1.0 / 3.0, 2.0 / 3.0, 1.0];
    let b = boundaries(
        &fractions,
        0.05,
        SpendingFunction::OBrienFlemingType,
        Sidedness::TwoSidedSymmetric,
        &never,
    )
    .expect("boundaries");
    for (look, statistic) in [(0, 4.5), (1, 3.0), (2, 2.1), (2, 0.4)] {
        let observed = StagewiseLook { look, statistic };
        let inference = stagewise_inference(
            &fractions,
            &b.critical,
            Sidedness::TwoSidedSymmetric,
            observed,
            0.95,
            &never,
        )
        .expect("inference");
        let p = |theta: f64| {
            stagewise_upper_p_value(
                &fractions,
                &b.critical,
                Sidedness::TwoSidedSymmetric,
                observed,
                theta,
                &never,
            )
            .expect("p")
        };
        assert!((p(inference.median_unbiased) - 0.5).abs() < 1e-8, "{look} {statistic}");
        assert!((p(inference.lower) - 0.025).abs() < 1e-8, "{look} {statistic}");
        assert!((p(inference.upper) - 0.975).abs() < 1e-8, "{look} {statistic}");
        assert!(inference.lower < inference.median_unbiased);
        assert!(inference.median_unbiased < inference.upper);
        assert!((inference.p_value_at_null - p(0.0)).abs() < 1e-15);
    }
}

#[test]
fn looks_too_close_for_the_quadrature_are_refused_not_summed_inaccurately() {
    use antecedent_stats::adaptive_inference::AdaptiveInferenceError;
    let close_pair = [0.999_999, 1.0];
    let outcome = boundaries(
        &close_pair,
        0.05,
        SpendingFunction::OBrienFlemingType,
        Sidedness::TwoSidedSymmetric,
        &never,
    );
    assert!(matches!(outcome, Err(AdaptiveInferenceError::Limit(_))), "{outcome:?}");
    let outcome =
        crossing_probabilities(&close_pair, &[2.5, 2.0], Sidedness::OneSidedUpper, 0.0, &never);
    assert!(matches!(outcome, Err(AdaptiveInferenceError::Limit(_))), "{outcome:?}");
}
