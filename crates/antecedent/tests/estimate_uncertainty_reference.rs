//! The published scalar standard error, its method label and its 95% interval against an
//! enumerated rule table and 50-digit quantiles.
//!
//! `oracle_data/estimate_uncertainty_reference.py` (frozen in the JSON beside it) enumerates every
//! combination of an analytic SE, a bootstrap SE and first-stage state (absent, present without
//! an Anderson-Rubin set, valid and invalid sets) and decides the published outcome by the stated
//! rule, and computes the two-sided 0.95 critical value (normal, or Student-t at the declared
//! reference degrees of freedom) and the interval by root finding on the exact distribution
//! functions. The product selector and interval must match.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::PublishedScalarUncertainty;
use antecedent_core::{AssumptionSet, IntervalMethod};
use antecedent_estimate::{EffectEstimate, FirstStageDiagnostics, OverlapPolicy};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("oracle_data/estimate_uncertainty_reference.json")).unwrap()
}

fn num(v: &Value) -> f64 {
    match v {
        Value::String(s) if s == "nan" => f64::NAN,
        Value::String(s) if s == "inf" => f64::INFINITY,
        Value::String(s) if s == "-inf" => f64::NEG_INFINITY,
        other => other.as_f64().unwrap(),
    }
}

fn estimate(ate: f64, analytic: f64, bootstrap: Option<f64>) -> EffectEstimate {
    let mut estimate = EffectEstimate::new(
        ate,
        analytic,
        AssumptionSet::default(),
        OverlapPolicy::ExplicitOverride,
    );
    estimate.uncertainty.bootstrap.se = bootstrap;
    estimate
}

fn method_tag(method: IntervalMethod) -> &'static str {
    match method {
        IntervalMethod::AnalyticSe => "analytic_se",
        IntervalMethod::BootstrapSe => "bootstrap_se",
        IntervalMethod::AndersonRubin => "anderson_rubin",
        IntervalMethod::None => "none",
        other => panic!("unexpected published method {other:?}"),
    }
}

#[test]
fn the_selector_matches_the_enumerated_rule_for_every_combination() {
    let fixture = fixture();
    let cases = fixture["selection"].as_array().unwrap();
    assert_eq!(cases.len(), 5 * 4 * 7);
    for case in cases {
        let name = case["name"].as_str().unwrap();
        let bootstrap = (!case["bootstrap"].is_null()).then(|| num(&case["bootstrap"]));
        let mut est = estimate(1.0, num(&case["analytic"]), bootstrap);
        if !case["first_stage"].is_null() {
            let ar =
                case["first_stage"]["ar"].as_array().map(|a| (num(&a[0]), num(&a[1]), num(&a[2])));
            est = est.with_first_stage_diagnostics(Some(FirstStageDiagnostics {
                f_statistic: 5.0,
                df1: 1,
                df2: 98,
                partial_r2: 0.05,
                anderson_rubin: ar,
                uncertainty_withheld: None,
            }));
        }
        let got = PublishedScalarUncertainty::select(&est);
        let want = &case["expected"];
        assert_eq!(method_tag(got.method), want["method"].as_str().unwrap(), "{name} method");
        match want["se"].as_f64() {
            Some(se) => assert_eq!(got.standard_error, Some(se), "{name} se"),
            None => assert!(got.standard_error.is_none(), "{name} se"),
        }
        assert_eq!(got.withheld_reason.is_some(), want["withheld"].as_bool().unwrap(), "{name}");
        match want["method"].as_str().unwrap() {
            "none" => assert!(got.level.is_nan(), "{name} level"),
            _ => assert!((got.level - want["level"].as_f64().unwrap()).abs() < 1e-15, "{name}"),
        }
        match (&want["lower"], got.lower) {
            (Value::Null, lower) => assert!(lower.is_none(), "{name} lower"),
            (w, Some(lower)) => assert_eq!(lower, num(w), "{name} lower"),
            (_, None) => panic!("{name}: missing lower"),
        }
        match (&want["upper"], got.upper) {
            (Value::Null, upper) => assert!(upper.is_none(), "{name} upper"),
            (w, Some(upper)) => assert_eq!(upper, num(w), "{name} upper"),
            (_, None) => panic!("{name}: missing upper"),
        }
    }
}

#[test]
fn the_critical_value_and_interval_match_the_exact_quantiles() {
    let fixture = fixture();
    for case in fixture["critical"].as_array().unwrap() {
        let (ate, se) = (case["ate"].as_f64().unwrap(), case["se"].as_f64().unwrap());
        let mut est = estimate(ate, se, None);
        est.uncertainty.analytic.reference_df = case["df"].as_f64();
        let published = PublishedScalarUncertainty::select(&est);
        let critical = published.critical_value(&est);
        let want = case["critical"].as_f64().unwrap();
        assert!(
            (critical - want).abs() <= 1e-9 * want,
            "df {:?}: {critical} vs {want}",
            case["df"]
        );
        let (lower, upper) = published.interval(&est).unwrap();
        let (wl, wu) = (case["lower"].as_f64().unwrap(), case["upper"].as_f64().unwrap());
        let tol = 1e-9 * (se * want).abs() + 4.0 * f64::EPSILON * ate.abs();
        assert!((lower - wl).abs() <= tol, "lower {lower} vs {wl}");
        assert!((upper - wu).abs() <= tol, "upper {upper} vs {wu}");
    }
    let z = fixture["z"].as_f64().unwrap();
    assert!((antecedent::result::reported_se_interval_z() - z).abs() < 1e-12);
}

#[test]
fn a_bootstrap_interval_ignores_declared_reference_degrees_of_freedom() {
    // The Student-t reference belongs to the analytic SE only; a bootstrap SE keeps the normal
    // quantile even if the estimate also declares degrees of freedom.
    let mut est = estimate(1.0, 0.1, Some(0.2));
    est.uncertainty.analytic.reference_df = Some(3.0);
    let published = PublishedScalarUncertainty::select(&est);
    assert_eq!(published.method, IntervalMethod::BootstrapSe);
    let z = fixture()["z"].as_f64().unwrap();
    assert!((published.critical_value(&est) - z).abs() < 1e-12);
}
