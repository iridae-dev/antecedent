//! Native linear conditional-effect fit and its cohort-mean interval: closed-form
//! oracles, exact recovery on noiseless data, and the declared-export and shifted-cohort
//! refusals. No coverage is measured here; see `cohort_linear_effect_calibration.rs`.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0
#![allow(
    clippy::float_cmp,
    reason = "bounded indices and exact enumerated comparisons in a calibration fixture"
)]

use antecedent_core::ExecutionContext;
use antecedent_estimate::cohort_linear_effect::{
    DETAIL_NOT_LICENSED, DETAIL_REQUIRES_NATIVE_FIT, FIT_PROCEDURE_ID, NativeCohortFit,
    NativeFitRows, declared_export_refusal, fit_cohort_linear_effect, native_cohort_interval,
    native_cohort_interval_for,
};
use antecedent_io_core::cohort_decision::{
    CATE_FUNCTIONAL_ID, CohortBinding, CohortColumn, CohortDecisionSource, CohortOptions,
    ContrastDeclaration, CovariateShiftDeclaration, RetainedEffectModel,
};
use antecedent_io_core::compact_export::{
    CompactExport, ExportInput, ExportSpec, InputSupport, TermSpec,
};
use antecedent_io_core::quantity_wire::ScientificQuantityWire;

fn wire(id: &str, role: &str, functional: &str, regime: &str) -> ScientificQuantityWire {
    ScientificQuantityWire {
        version: 1,
        variable_id: id.into(),
        variable_name: id.into(),
        role: role.into(),
        units: "unit".into(),
        population_id: "target".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: functional.into(),
        conditioning: vec![],
        transform_id: "identity".into(),
    }
}

fn lcg(state: &mut u64) -> f64 {
    *state = state.wrapping_mul(6_364_136_223_846_793_005).wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 11) as f64 + 0.5) / 9_007_199_254_740_992.0
}

fn response() -> ScientificQuantityWire {
    wire("y", "outcome", CATE_FUNCTIONAL_ID, &ContrastDeclaration::regime_of("t", 1.0, 0.0))
}

/// Noisy data with one modifier: tau(x) = 1 + 0.5 x, baseline 2 + x.
fn one_modifier(n: usize, noise: f64) -> (Vec<f64>, Vec<f64>, Vec<f64>) {
    let mut s = 17_u64;
    let (mut x, mut t, mut y) = (vec![], vec![], vec![]);
    for i in 0..n {
        let xi = 4.0 * lcg(&mut s) - 2.0;
        let ti = f64::from(u8::from(i % 2 == 0));
        x.push(xi);
        t.push(ti);
        y.push(2.0 + xi + ti * (1.0 + 0.5 * xi) + noise * (lcg(&mut s) - 0.5));
    }
    (x, t, y)
}

#[test]
fn the_fit_matches_the_two_group_closed_form_for_the_sandwich() {
    // No modifiers: design (1, t); b = ybar_1 - ybar_0 and, with h_i = 1/n_g,
    // HC3 variance = sum_g sum_i e_i^2 / (n_g - 1)^2.
    let mut s = 5_u64;
    let n = 40;
    let t: Vec<f64> = (0..n).map(|i| f64::from(u8::from(i % 2 == 0))).collect();
    let y: Vec<f64> = t.iter().map(|ti| 1.0 + 2.0 * ti + (lcg(&mut s) - 0.5)).collect();
    let fit = fit_cohort_linear_effect(&y, &t, &[]).unwrap();
    let mean = |arm: f64| {
        let v: Vec<f64> = y.iter().zip(&t).filter(|(_, a)| **a == arm).map(|(v, _)| *v).collect();
        (v.iter().sum::<f64>() / v.len() as f64, v)
    };
    let (m1, v1) = mean(1.0);
    let (m0, v0) = mean(0.0);
    assert!((fit.coefficients[0] - (m1 - m0)).abs() < 1e-9);
    let group = |m: f64, v: &[f64]| {
        v.iter().map(|y| (y - m) * (y - m)).sum::<f64>() / ((v.len() - 1) as f64).powi(2)
    };
    let expected = group(m1, &v1) + group(m0, &v0);
    assert!((fit.covariance[0] - expected).abs() < 1e-9 * expected.max(1.0));
    assert_eq!((fit.rows, fit.width), (n, 2));
}

#[test]
fn noiseless_data_recover_the_effect_coefficients_exactly() {
    let (x, t, y) = one_modifier(60, 0.0);
    let fit = fit_cohort_linear_effect(&y, &t, &[&x]).unwrap();
    assert!((fit.coefficients[0] - 1.0).abs() < 1e-8, "{:?}", fit.coefficients);
    assert!((fit.coefficients[1] - 0.5).abs() < 1e-8, "{:?}", fit.coefficients);
    assert!(fit.covariance.iter().all(|v| v.abs() < 1e-12));
    assert_eq!((fit.rows, fit.width), (60, 4));
}

#[test]
fn a_fit_with_too_few_rows_or_a_singular_design_is_refused() {
    let (x, t, y) = one_modifier(60, 0.1);
    assert!(fit_cohort_linear_effect(&y[..4], &t[..4], &[&x[..4]]).is_err());
    let constant = vec![1.0; 60];
    assert!(fit_cohort_linear_effect(&y, &t, &[&constant]).is_err());
}

fn native_of(x: &[f64], t: &[f64], y: &[f64]) -> NativeCohortFit {
    NativeCohortFit::fit(y, t, &[x], &response(), &[wire("x", "covariate", "mean", "observed")])
        .unwrap()
}

fn cohort_of(x: &[f64], population: &str) -> CohortBinding {
    CohortBinding::new(
        "cohort",
        population,
        (0..x.len()).map(|i| format!("r{i}")).collect(),
        vec!["x".into()],
        vec![CohortColumn::Numeric(x.to_vec())],
        CohortBinding::uniform_weights(x.len()).unwrap(),
    )
    .unwrap()
}

#[test]
fn the_native_interval_is_a_symmetric_student_t_band_over_the_recomputed_standard_error() {
    let (x, t, y) = one_modifier(120, 1.0);
    let native = native_of(&x, &t, &y);
    let size = native.size();
    assert_eq!((size.rows, size.width), (120, 4));
    let cohort = cohort_of(&x[..60], "target");
    let (source, interval) =
        native_cohort_interval_for(&native, &cohort, "unit", 0.95, &ExecutionContext::for_tests(1))
            .unwrap();
    assert!((interval.point - source.mean_contrast()).abs() < 1e-12);
    assert!(
        (interval.standard_error - source.uncalibrated_standard_error().unwrap().value).abs()
            < 1e-15
    );
    assert_eq!(interval.degrees_of_freedom, 116);
    let expected = antecedent_stats::special::student_t_ppf(0.975, 116.0);
    assert!((interval.critical_value - expected).abs() < 1e-12);
    assert!(((interval.upper - interval.point) - (interval.point - interval.lower)).abs() < 1e-12);
    // Wider at 0.95 than at 0.90.
    let narrow = native_cohort_interval(&native, &source, 0.90).unwrap();
    assert!(narrow.upper - narrow.lower < interval.upper - interval.lower);
    assert!(native_cohort_interval(&native, &source, 1.0).is_err());
}

#[test]
fn a_declared_export_cannot_claim_a_native_fit() {
    let (x, t, y) = one_modifier(120, 1.0);
    let native = native_of(&x, &t, &y);
    let fit = fit_cohort_linear_effect(&y, &t, &[&x]).unwrap();
    let declared = CompactExport::build(ExportSpec {
        response: response(),
        inputs: vec![ExportInput {
            quantity: wire("x", "covariate", "mean", "observed"),
            support: InputSupport::Range { lo: -2.0, hi: 2.0 },
        }],
        terms: vec![TermSpec::Intercept, TermSpec::Linear { quantity: "x".into() }],
        coefficients: fit.coefficients.clone(),
        covariance: fit.covariance.clone(),
        mask: vec![],
    })
    .unwrap();
    assert_eq!(declared_export_refusal().detail, DETAIL_REQUIRES_NATIVE_FIT);
    assert_eq!(declared_export_refusal().code, "cell_not_licensed");
    // The same numbers carry a different identity, and replaying the real rows against
    // the declared export is refused.
    assert_ne!(native.export().identity(), declared.identity());
    let rows = NativeFitRows {
        outcome: y.clone(),
        treatment: t.clone(),
        features: vec![x.clone()],
        feature_ids: vec!["x".into()],
    };
    assert!(NativeCohortFit::replay(rows.clone(), &declared).is_err());
    // A forged marker over declared numbers is refused: the replay recomputes it.
    let mut forged = response();
    forged.conditioning = native.export().body().response.conditioning.clone();
    let forged_export = CompactExport::build(ExportSpec {
        response: forged,
        inputs: vec![ExportInput {
            quantity: wire("x", "covariate", "mean", "observed"),
            support: InputSupport::Range { lo: -2.0, hi: 2.0 },
        }],
        terms: vec![TermSpec::Intercept, TermSpec::Linear { quantity: "x".into() }],
        coefficients: vec![9.0, 9.0],
        covariance: vec![0.0001, 0.0, 0.0, 0.0001],
        mask: vec![],
    })
    .unwrap();
    assert!(NativeCohortFit::replay(rows.clone(), &forged_export).is_err());
    // The genuine export replays to itself.
    let replayed = NativeCohortFit::replay(rows, native.export()).unwrap();
    assert_eq!(replayed.export().identity(), native.export().identity());
    assert_eq!(native.export().body().response.conditioning.len(), 4);
    assert!(
        native.export().body().response.conditioning.iter().any(|c| c.value_id == FIT_PROCEDURE_ID)
    );
}

#[test]
fn a_shifted_cohort_stays_point_only() {
    let (x, t, y) = one_modifier(120, 1.0);
    let native = native_of(&x, &t, &y);
    let model = RetainedEffectModel::from_compact_export(native.export()).unwrap();
    let options = CohortOptions {
        shift: Some(CovariateShiftDeclaration::new("target", "elsewhere", &["x"])),
        expected_contrast_regime: None,
    };
    let source = CohortDecisionSource::build(
        &model,
        &cohort_of(&x[..60], "elsewhere"),
        "unit",
        &options,
        &ExecutionContext::for_tests(1),
    )
    .unwrap();
    let error = native_cohort_interval(&native, &source, 0.95).unwrap_err();
    assert_eq!(error.detail, DETAIL_NOT_LICENSED);
}

// -- exact multiprecision reference --------------------------------------------------------------
// `oracle_data/cohort_linear_effect_reference.py` refits each design by OLS with the HC3 sandwich
// in 50-digit arithmetic (on the uncentred basis), forms the cohort-mean contrast and its standard
// error, inverts the Student-t distribution function from the incomplete beta function, and
// freezes the binary64 roundings.

fn reference() -> serde_json::Value {
    serde_json::from_str(include_str!("oracle_data/cohort_linear_effect_reference.json")).unwrap()
}

fn column(v: &serde_json::Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn rel_close(got: f64, want: f64, rel: f64, what: &str) {
    let tol = rel * want.abs().max(1e-300);
    assert!((got - want).abs() <= tol, "{what}: got {got} want {want} tol {tol}");
}

#[test]
fn coefficients_and_hc3_covariance_equal_the_multiprecision_fit_with_zero_one_and_two_modifiers() {
    let fx = reference();
    for case in fx["fits"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let features: Vec<Vec<f64>> =
            case["features"].as_array().unwrap().iter().map(column).collect();
        let columns: Vec<&[f64]> = features.iter().map(Vec::as_slice).collect();
        let fit =
            fit_cohort_linear_effect(&column(&case["outcome"]), &column(&case["treatment"]), &columns)
                .unwrap();
        assert_eq!(fit.width as u64, case["width"].as_u64().unwrap(), "{name}");
        let coefficients = column(&case["coefficients"]);
        let covariance = column(&case["covariance"]);
        let scale = coefficients.iter().fold(0.0_f64, |m, c| m.max(c.abs()));
        for (got, want) in fit.coefficients.iter().zip(&coefficients) {
            assert!((got - want).abs() <= 1e-9 * scale.max(1.0), "{name} coefficient {got} vs {want}");
        }
        let vscale = covariance.iter().fold(0.0_f64, |m, c| m.max(c.abs()));
        for (got, want) in fit.covariance.iter().zip(&covariance) {
            assert!((got - want).abs() <= 1e-8 * vscale, "{name} covariance {got} vs {want}");
        }
    }
}

#[test]
fn student_t_critical_values_equal_the_inverted_incomplete_beta() {
    for q in reference()["t_quantiles"].as_array().unwrap() {
        let df = q["df"].as_f64().unwrap();
        let p = q["p"].as_f64().unwrap();
        rel_close(antecedent_stats::special::student_t_ppf(p, df), q["want"].as_f64().unwrap(), 1e-9, &format!("t({p}; {df})"));
    }
}

#[test]
fn the_cohort_mean_interval_equals_the_reference_point_standard_error_and_bands() {
    let fx = reference();
    let case = &fx["fits"][1];
    let (x, t, y) = (
        column(&case["features"][0]),
        column(&case["treatment"]),
        column(&case["outcome"]),
    );
    let native = native_of(&x, &t, &y);
    let interval = &fx["interval"];
    let cohort = column(&interval["cohort"]);
    let (source, first) =
        native_cohort_interval_for(&native, &cohort_of(&cohort, "target"), "unit", 0.95, &ExecutionContext::for_tests(1))
            .unwrap();
    rel_close(source.mean_contrast(), interval["point"].as_f64().unwrap(), 1e-9, "cohort point");
    rel_close(first.standard_error, interval["se"].as_f64().unwrap(), 1e-8, "cohort standard error");
    assert_eq!(first.degrees_of_freedom as u64, interval["df"].as_u64().unwrap());
    for band in interval["levels"].as_array().unwrap() {
        let level = band["level"].as_f64().unwrap();
        let got = native_cohort_interval(&native, &source, level).unwrap();
        rel_close(got.critical_value, band["critical"].as_f64().unwrap(), 1e-9, "critical");
        rel_close(got.lower, band["lower"].as_f64().unwrap(), 1e-8, "lower");
        rel_close(got.upper, band["upper"].as_f64().unwrap(), 1e-8, "upper");
    }
}
