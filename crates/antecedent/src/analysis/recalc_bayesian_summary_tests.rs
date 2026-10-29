//! The projections of retained effect draws against an exact rational reference.
//!
//! `tests/oracle_data/recalc_posterior_summary_reference.py` forms the mean, standard deviation,
//! type-7 interpolated quantiles and strict-inequality proportion of each draw set in exact
//! arithmetic; `summarize_issued_values` is compared to its frozen rounding.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use super::{PosteriorSummarySpec, RecalcRequestCode, RecalcRunError, summarize_issued_values};
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    let tol = rel * want.abs().max(1e-300);
    assert!((got - want).abs() <= tol, "{what}: got {got} want {want} tol {tol}");
}

#[test]
fn projections_equal_the_exact_rational_mean_sd_type7_quantiles_and_strict_proportion() {
    let cases: Vec<Value> = serde_json::from_str(include_str!(
        "../../tests/oracle_data/recalc_posterior_summary_reference.json"
    ))
    .unwrap();
    assert_eq!(cases.len(), 8);
    for case in &cases {
        let name = case["name"].as_str().unwrap();
        let values = floats(&case["values"]);
        let spec = PosteriorSummarySpec {
            lower_probability: case["lower"].as_f64().unwrap(),
            upper_probability: case["upper"].as_f64().unwrap(),
            threshold: case["threshold"].as_f64().unwrap(),
        };
        let law = summarize_issued_values(&values, spec).unwrap();
        assert_eq!(law.draws, values.len(), "{name}");
        let scale = values.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        let want_mean = case["mean"].as_f64().unwrap();
        assert!((law.mean - want_mean).abs() <= 4.0 * f64::EPSILON * scale.max(1e-300), "{name} mean");
        close(law.standard_deviation, case["sd"].as_f64().unwrap(), 1e-9, &format!("{name} sd"));
        let q = |key: &str| case[key].as_f64().unwrap();
        assert!(
            (law.lower_quantile - q("lower_quantile")).abs() <= 4.0 * f64::EPSILON * scale.max(1e-300),
            "{name} lower: {} vs {}",
            law.lower_quantile,
            q("lower_quantile")
        );
        assert!(
            (law.upper_quantile - q("upper_quantile")).abs() <= 4.0 * f64::EPSILON * scale.max(1e-300),
            "{name} upper"
        );
        assert_eq!(law.probability_below, q("below"), "{name} proportion below");
    }
}

#[test]
fn projections_refuse_too_few_or_non_finite_draws_and_invalid_summaries() {
    let spec = PosteriorSummarySpec::default();
    for draws in [&[1.0][..], &[1.0, f64::NAN][..], &[1.0, f64::INFINITY][..]] {
        let refused = summarize_issued_values(draws, spec).err().expect("refused");
        assert!(matches!(
            refused,
            RecalcRunError::Request(RecalcRequestCode::BayesianEffectUnavailable)
        ));
        assert_eq!(refused.to_string(), "recalc.bayesian_effect_unavailable", "{draws:?}");
    }
    for (lo, hi) in [(0.0, 0.9), (0.1, 1.0), (0.6, 0.4), (0.5, 0.5), (f64::NAN, 0.9)] {
        let bad =
            PosteriorSummarySpec { lower_probability: lo, upper_probability: hi, threshold: 0.0 };
        let refused = bad.validate().err().unwrap_or_else(|| panic!("{lo} {hi} must be refused"));
        assert!(matches!(
            refused,
            RecalcRunError::Request(RecalcRequestCode::BayesianInvalidSummary)
        ));
        assert_eq!(refused.to_string(), "recalc.bayesian_invalid_summary", "{lo} {hi}");
    }
}
