//! Cross-fitted Gaussian-score Riesz average derivative against an exact rational pipeline.
//!
//! `oracle_data/estimate_response_riesz_ade_reference.py` (frozen in the JSON beside it) follows the
//! estimator's definition on dyadic data: index-modulo folds, the penalized additive spline
//! nuisances of the outcome and of the treatment on the training rows at their unique exact
//! solution, the derivative of the outcome's treatment smooth at the held-out treatment (zero
//! beyond the training knot interior), the treatment scale with the effective-degrees-of-freedom
//! divisor (or the n - 1 divisor without adjusters), the Riesz weight (a - m) / sigma^2, the score
//! derivative + weight * residual, its mean, its (n - 1)-divisor variance, the normal interval at
//! the stated level and the Kish effective size of the weights. The product's nuisances are the
//! backfit to its stopping tolerance, so scores are compared at that resolution, per row scaled by
//! the row's Riesz weight, not to rounding. The exported influence is the centered score.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::{
    AssumptionSet, DerivativeWeighting, IdentificationStatus, IntervalInterpretation,
    ResponseFunctional, ResponseIdentification, ResponseQuery, ResponseUncertainty, ResponseValue,
    VariableId,
};
use antecedent_data::TabularData;
use antecedent_estimate::response::ContinuousResponseEstimator;
use serde_json::Value;

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| num(x)).collect()
}

fn num(v: &Value) -> f64 {
    v.as_str().map_or_else(|| v.as_f64().unwrap(), |s| s.parse().unwrap())
}

fn fixture() -> Value {
    serde_json::from_str(include_str!("oracle_data/estimate_response_riesz_ade_reference.json"))
        .unwrap()
}

fn estimator_for(case: &Value) -> (ContinuousResponseEstimator, ResponseQuery, TabularData) {
    let a = floats(&case["a"]);
    let y = floats(&case["y"]);
    let adjusters = case["adjusters"].as_u64().unwrap();
    let data = if adjusters == 1 {
        let x = floats(&case["x"]);
        TabularData::from_f64_columns([
            ("a", a.as_slice()),
            ("y", y.as_slice()),
            ("x", x.as_slice()),
        ])
        .unwrap()
    } else {
        TabularData::from_f64_columns([("a", a.as_slice()), ("y", y.as_slice())]).unwrap()
    };
    let mut estimator = ContinuousResponseEstimator::new(
        (0..adjusters).map(|_| VariableId::from_raw(2)).collect::<Vec<_>>(),
    );
    estimator.options.folds = usize::try_from(case["folds"].as_u64().unwrap()).unwrap();
    estimator.options.nuisance_basis = usize::try_from(case["basis"].as_u64().unwrap()).unwrap();
    estimator.options.nuisance_lambda = case["lambda"].as_f64().unwrap();
    estimator.options.confidence_level = case["level"].as_f64().unwrap();
    estimator.options.minimum_local_ess = case["min_ess"].as_f64().unwrap();
    let query = ResponseQuery::new(ResponseFunctional::AverageDerivative {
        outcome: VariableId::from_raw(1),
        treatment: VariableId::from_raw(0),
        weighting: DerivativeWeighting::Observed,
    });
    (estimator, query, data)
}

#[test]
fn estimate_response_riesz_ade_scores_estimate_interval_and_overlap_match_exact_pipeline() {
    for case in fixture()["cases"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let (estimator, query, data) = estimator_for(case);
        let response = estimator
            .estimate_identified(
                &data,
                &query,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .unwrap_or_else(|e| panic!("{name}: {e}"));
        assert_eq!(response.provenance_id.as_ref(), "estimate.response.riesz_ade");
        let scores = floats(&case["scores"]);
        let riesz = floats(&case["riesz"]);
        let n = scores.len() as f64;
        let y = floats(&case["y"]);
        let spread = y.iter().copied().fold(f64::NEG_INFINITY, f64::max)
            - y.iter().copied().fold(f64::INFINITY, f64::min);
        let magnitude = y.iter().fold(0.0_f64, |m, v| m.max(v.abs()));
        // The backfit stops on a change relative to the largest fitted value, intercept included.
        let base = 1e-6 * spread.max(1.0) + 2e-9 * magnitude;
        let row_tolerance = |i: usize| base * (1.0 + riesz[i].abs());

        // Influence export: the centered score of the product's own cross-fit.
        let influence = estimator
            .derivative_influence(&data, &query)
            .unwrap()
            .unwrap_or_else(|| panic!("{name}: an influence column"));
        assert_eq!(influence.columns.len(), 1, "{name}");
        let column = &influence.columns[0];
        assert_eq!(column.len(), scores.len(), "{name}");
        let want_mean = scores.iter().sum::<f64>() / n;
        let got_mean_of_influence = column.iter().sum::<f64>() / n;
        assert!(got_mean_of_influence.abs() <= 1e-10 * (1.0 + magnitude), "{name}: centered");
        let mut worst = 0.0_f64;
        for i in 0..scores.len() {
            let want = scores[i] - want_mean;
            let miss = (column[i] - want).abs();
            worst = worst.max(miss / row_tolerance(i));
            assert!(
                miss <= 2.0 * row_tolerance(i),
                "{name} row {i}: centered score {} vs {want}",
                column[i]
            );
        }
        eprintln!("{name}: worst score miss / tolerance = {worst:.3}");

        let ResponseIdentification::PointIdentified(ResponseValue::Scalar(estimate)) =
            response.estimate
        else {
            panic!("{name}: a scalar");
        };
        let ResponseUncertainty::Scalar {
            standard_error,
            level,
            lower,
            upper,
            interpretation,
            draws,
        } = response.uncertainty
        else {
            panic!("{name}: a scalar interval");
        };
        assert_eq!(level, case["level"].as_f64().unwrap(), "{name}: the stated level is used");
        assert_eq!(interpretation, IntervalInterpretation::Confidence, "{name}");
        assert!(draws.is_none(), "{name}");
        let mean_tolerance = base * (1.0 + riesz.iter().map(|r| r.abs()).sum::<f64>() / n);
        let (want_estimate, want_se) = (num(&case["estimate"]), num(&case["se"]));
        assert!((estimate - want_estimate).abs() <= mean_tolerance, "{name}: estimate");
        assert!(
            (standard_error - want_se).abs() <= mean_tolerance / n.sqrt() * 8.0 + 1e-9 * want_se,
            "{name}: se {standard_error} vs {want_se}"
        );
        assert!((lower - num(&case["lower"])).abs() <= 4.0 * mean_tolerance, "{name}: lower");
        assert!((upper - num(&case["upper"])).abs() <= 4.0 * mean_tolerance, "{name}: upper");
        // The interval is the estimate with a symmetric normal half-width at the stated level.
        assert!(((upper - estimate) - (estimate - lower)).abs() <= 1e-9 * (1.0 + upper.abs()));

        // Kish effective size of the representer, and the warning it drives.
        let diag = response
            .support
            .diagnostics
            .iter()
            .find(|d| d.id.as_ref() == "response.weighted_effective_sample_size")
            .unwrap_or_else(|| panic!("{name}: the effective sample size"));
        let want_ess = num(&case["ess"]);
        assert!((diag.values[0] - want_ess).abs() <= 1e-3 * want_ess, "{name}: ess");
        assert_eq!(diag.values[1], n, "{name}: rows");
        let weak = want_ess < case["min_ess"].as_f64().unwrap();
        let warned = response
            .support
            .warnings
            .iter()
            .any(|w| w.code.as_ref() == "response.weak_riesz_overlap");
        assert_eq!(warned, weak, "{name}: weak-overlap warning");
        assert_eq!(
            matches!(response.support.status, antecedent_core::SupportStatus::WeakOverlap),
            weak,
            "{name}: support status"
        );
    }
}

#[test]
fn estimate_response_riesz_ade_refuses_other_weightings_and_excess_folds() {
    let frozen = fixture();
    let case = &frozen["cases"][0];
    let (mut estimator, query, data) = estimator_for(case);
    for weighting in [
        DerivativeWeighting::Uniform { lower: -0.5, upper: 0.5 },
        DerivativeWeighting::Custom(Arc::from(vec![1.0; 60])),
    ] {
        let other = ResponseQuery::new(ResponseFunctional::AverageDerivative {
            outcome: VariableId::from_raw(1),
            treatment: VariableId::from_raw(0),
            weighting,
        });
        let error = estimator
            .estimate_identified(
                &data,
                &other,
                IdentificationStatus::NonparametricallyIdentified,
                AssumptionSet::new(),
            )
            .expect_err("a weighting other than the observed law");
        assert!(error.to_string().contains("observed-law"), "{error}");
        assert!(estimator.derivative_influence(&data, &other).unwrap().is_none());
    }
    // Folds beyond the rows are refused rather than silently clamped.
    estimator.options.folds = 1000;
    let error = estimator
        .estimate_identified(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
        .expect_err("more folds than rows");
    assert!(error.to_string().contains("folds"), "{error}");
}

/// A noise-free linear law on a skewed treatment, no adjusters.
///
/// Sample-quantile knots of a skewed treatment are far from equally spaced, where a coefficient
/// second-difference penalty is not zero on a straight line and shrank the fitted smooth's
/// derivative by several percent; the knot-aware penalty leaves a line unpenalised, so the fitted
/// smooth is exactly the line inside each fold's training range and its derivative is the slope.
/// The one place the cross-fitted average is not the slope is the documented boundary rule: a
/// held-out treatment at or beyond its fold's training maximum (or below its minimum) has
/// derivative zero and its residual is the distance to the clamped line, which the Riesz term
/// then carries. The expected value below is that closed form per fold (training mean and
/// n - 1 variance for the Riesz weight, no fitted nuisance), not a number read off the product.
#[test]
fn estimate_response_riesz_ade_of_a_linear_law_on_a_skewed_treatment_equals_the_closed_form() {
    const SLOPE: f64 = 2.0;
    let n = 150_usize;
    let folds = 5_usize;
    let a: Vec<f64> = (0..n).map(|i| (i as f64 / (n - 1) as f64).powi(3) * 4.0).collect();
    let y: Vec<f64> = a.iter().map(|v| 1.0 + SLOPE * v).collect();
    let mut expected = 0.0;
    for fold in 0..folds {
        let train: Vec<f64> = (0..n).filter(|i| i % folds != fold).map(|i| a[i]).collect();
        let (lo, hi) = train.iter().fold((f64::INFINITY, f64::NEG_INFINITY), |(l, h), &v| {
            (l.min(v), h.max(v))
        });
        let mean = train.iter().sum::<f64>() / train.len() as f64;
        let sigma2 = train.iter().map(|v| (v - mean).powi(2)).sum::<f64>()
            / (train.len() as f64 - 1.0);
        for i in (0..n).filter(|i| i % folds == fold) {
            let derivative = if a[i] < lo || a[i] >= hi { 0.0 } else { SLOPE };
            let residual = y[i] - (1.0 + SLOPE * a[i].clamp(lo, hi));
            expected += derivative + (a[i] - mean) / sigma2 * residual;
        }
    }
    expected /= n as f64;
    let data = TabularData::from_f64_columns([("a", &a[..]), ("y", &y[..])]).unwrap();
    let query = ResponseQuery::new(ResponseFunctional::AverageDerivative {
        outcome: VariableId::from_raw(1),
        treatment: VariableId::from_raw(0),
        weighting: DerivativeWeighting::Observed,
    });
    let response = ContinuousResponseEstimator::new(Vec::new())
        .estimate_identified(
            &data,
            &query,
            IdentificationStatus::NonparametricallyIdentified,
            AssumptionSet::new(),
        )
        .unwrap();
    let ResponseIdentification::PointIdentified(ResponseValue::Scalar(estimate)) =
        response.estimate
    else {
        panic!("expected a point-identified scalar");
    };
    assert!((estimate - expected).abs() < 1e-6, "{estimate} vs the closed form {expected}");
    // The coefficient second difference gave 2.1347 here, 0.16 above this closed form.
}
