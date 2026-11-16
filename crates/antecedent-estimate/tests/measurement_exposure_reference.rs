//! Exposure misclassification, soft exposures, covariate-dependent detection and latent-class
//! repeat codings against independent oracles.
//!
//! `oracle_data/measurement_exposure_reference.py` (frozen in the JSON beside it) computes, in
//! 50-digit and exact rational arithmetic and without the library's closed forms: the corrected
//! exposure log odds ratio with its delta-method variance by numerical differentiation of the
//! explicit function; exact rational least squares and Rubin's rules for a soft exposure; exact
//! reliability bins; the prevalence under a logistic detection function with the estimator's
//! gradient by numerical differentiation; the sharp bounds of a monotone envelope; and the
//! maximum-likelihood latent-class fit of two integer pattern-count tables by EM to 1e-30.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#![allow(clippy::cast_possible_truncation, reason = "oracle pattern indices are small")]

use antecedent_core::ExecutionContext;
use antecedent_core::measurement_model::ParameterUncertainty;
use antecedent_estimate::measurement::{
    SoftExposureCombination, correct_detection_dependent_prevalence,
    correct_exposure_misclassification, correct_soft_exposure_effect, fit_repeat_codings,
    soft_exposure_reliability_report,
};
use antecedent_stats::measurement_error::{
    BinaryErrorRates, DetectionAdjusted, DetectionAnchor, DetectionData, DetectionDeclaration,
    DetectionDirection, DetectionFunction, ExposureErrorStructure, ExposureOutcomeCounts,
    FalsePositiveRate, LatentClassConstraints, LatentClassData, LatentClassSettings,
    SoftExposureData, exposure_rates_from_validation,
};
use serde_json::Value;

fn reference() -> Value {
    serde_json::from_str(include_str!("oracle_data/measurement_exposure_reference.json")).unwrap()
}

fn floats(v: &Value) -> Vec<f64> {
    v.as_array().unwrap().iter().map(|x| x.as_f64().unwrap()).collect()
}

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(1)
}

fn close(got: f64, want: f64, rel: f64, what: &str) {
    assert!((got - want).abs() <= rel * want.abs().max(1e-300), "{what}: got {got} want {want}");
}

fn rates(se: f64, sp: f64, se_var: f64, sp_var: f64) -> BinaryErrorRates {
    BinaryErrorRates::new(se, sp, se_var, sp_var).unwrap()
}

#[test]
fn exposure_correction_matches_the_numerically_differentiated_oracle() {
    for case in reference()["exposure"].as_array().unwrap() {
        let name = case["name"].as_str().unwrap();
        let c: Vec<u64> =
            case["counts"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        let counts = ExposureOutcomeCounts {
            exposed_cases: c[0],
            cases: c[1],
            exposed_controls: c[2],
            controls: c[3],
        };
        let r = floats(&case["rates"]);
        let v = floats(&case["rate_variances"]);
        let structure = if case["shared"].as_bool().unwrap() {
            ExposureErrorStructure::Shared(rates(r[0], r[1], v[0], v[1]))
        } else if case.get("validation_cases").is_some() {
            let tuple = |key: &str| {
                let a: Vec<u64> =
                    case[key].as_array().unwrap().iter().map(|x| x.as_u64().unwrap()).collect();
                (a[0], a[1], a[2], a[3])
            };
            exposure_rates_from_validation(tuple("validation_controls"), tuple("validation_cases"))
                .unwrap()
        } else {
            ExposureErrorStructure::ByOutcome {
                cases: rates(r[0], r[1], v[0], v[1]),
                controls: rates(r[2], r[3], v[2], v[3]),
            }
        };
        let fit = correct_exposure_misclassification(&counts, &structure, 0.1, &ctx()).unwrap();
        let a = &fit.association;
        close(a.log_odds_ratio, case["log_odds_ratio"].as_f64().unwrap(), 1e-12, name);
        close(a.variance_sampling, case["variance_sampling"].as_f64().unwrap(), 1e-9, name);
        close(a.variance_rates, case["variance_rates"].as_f64().unwrap(), 1e-8, name);
        let total = a.variance_sampling + a.variance_rates;
        close(fit.estimate.standard_error.unwrap(), total.sqrt(), 1e-12, name);
        assert_eq!(fit.estimate.parameter_uncertainty, ParameterUncertainty::Propagated);
        let (lo, hi) = fit.estimate.wald_interval(0.95).unwrap();
        close((lo + hi) / 2.0, a.log_odds_ratio, 1e-12, name);
        assert!(lo < a.log_odds_ratio && a.log_odds_ratio < hi);
    }
}

#[test]
fn unidentified_exposure_rates_refuse_with_the_registered_code() {
    let counts = ExposureOutcomeCounts {
        exposed_cases: 480,
        cases: 1000,
        exposed_controls: 180,
        controls: 1000,
    };
    let weak = ExposureErrorStructure::ByOutcome {
        cases: rates(0.9, 0.8, 0.0, 0.0),
        controls: rates(0.55, 0.5, 0.0, 0.0),
    };
    let err = correct_exposure_misclassification(&counts, &weak, 0.1, &ctx()).unwrap_err();
    assert_eq!(err.code, antecedent_core::reason_code!("measurement_not_identified"));
    assert_eq!(err.detail, antecedent_core::reason_details::measurement_model::YOUDEN_TOO_SMALL);
}

#[test]
fn soft_exposure_substitution_and_rubin_match_exact_rational_arithmetic() {
    let soft = &reference()["soft"];
    let outcome = floats(&soft["outcome"]);
    let probability = floats(&soft["probability"]);
    let covariate = floats(&soft["covariate"]);
    let data = SoftExposureData {
        outcome: &outcome,
        probability: &probability,
        covariates: &covariate,
        covariate_count: 1,
    };
    let fit = correct_soft_exposure_effect(&data, &SoftExposureCombination::PosteriorMean, &ctx())
        .unwrap();
    for (got, want) in fit.coefficients.iter().zip(floats(&soft["coefficients"])) {
        close(*got, want, 1e-9, "substitution coefficient");
    }
    let substitution = fit.substitution.as_ref().unwrap();
    close(substitution.variance, soft["hc0_variance"].as_f64().unwrap(), 1e-9, "HC0 variance");
    close(
        substitution.variance_exposure_uncertainty,
        soft["exposure_term"].as_f64().unwrap(),
        1e-9,
        "exposure uncertainty term",
    );
    close(
        substitution.mean_posterior_variance,
        soft["mean_posterior_variance"].as_f64().unwrap(),
        1e-12,
        "mean posterior variance",
    );
    let draws: Vec<f64> = soft["draws"].as_array().unwrap().iter().flat_map(floats).collect();
    let imputed = correct_soft_exposure_effect(
        &data,
        &SoftExposureCombination::SuppliedDraws { draws: &draws, draw_count: 3 },
        &ctx(),
    )
    .unwrap();
    let combined = imputed.imputations.as_ref().unwrap();
    let d = combined.dim;
    close(
        combined.coefficients[1],
        soft["rubin_coefficient"].as_f64().unwrap(),
        1e-9,
        "Rubin point",
    );
    close(combined.within[d + 1], soft["rubin_within"].as_f64().unwrap(), 1e-9, "within");
    close(combined.between[d + 1], soft["rubin_between"].as_f64().unwrap(), 1e-9, "between");
    close(combined.covariance[d + 1], soft["rubin_total"].as_f64().unwrap(), 1e-9, "total");
    assert_eq!(imputed.estimate.parameter_uncertainty, ParameterUncertainty::Propagated);
    // One imputation cannot be combined.
    assert!(
        correct_soft_exposure_effect(
            &data,
            &SoftExposureCombination::SuppliedDraws { draws: &draws[..12], draw_count: 1 },
            &ctx()
        )
        .is_err()
    );
}

#[test]
fn seeded_imputation_matches_the_independent_port_draw_for_draw() {
    let case = &reference()["imputation"];
    let outcome = floats(&case["outcome"]);
    let probability = floats(&case["probability"]);
    let covariate = floats(&case["covariate"]);
    let draw_count = case["draw_count"].as_u64().unwrap() as usize;
    let data = SoftExposureData {
        outcome: &outcome,
        probability: &probability,
        covariates: &covariate,
        covariate_count: 1,
    };
    let generated = antecedent_stats::measurement_error::soft_exposure_outcome_conditioned_draws(
        &data,
        draw_count,
        case["seed"].as_u64().unwrap(),
    )
    .unwrap();
    for (got, want) in generated.coefficients.iter().zip(floats(&case["full_coefficients"])) {
        close(*got, want, 1e-6, "EM coefficient");
    }
    close(
        generated.residual_variance,
        case["full_residual_variance"].as_f64().unwrap(),
        1e-6,
        "EM residual variance",
    );
    let want: Vec<f64> = case["draws"].as_array().unwrap().iter().flat_map(floats).collect();
    assert_eq!(generated.draws, want, "every imputed exposure equals the port's");
    let effect = correct_soft_exposure_effect(
        &data,
        &SoftExposureCombination::MultipleImputation {
            draw_count,
            seed: case["seed"].as_u64().unwrap(),
        },
        &ctx(),
    )
    .unwrap();
    let combined = effect.imputations.as_ref().unwrap();
    let d = combined.dim;
    close(combined.coefficients[1], case["coefficient"].as_f64().unwrap(), 1e-9, "Rubin point");
    close(combined.within[d + 1], case["within"].as_f64().unwrap(), 1e-9, "within");
    close(combined.between[d + 1], case["between"].as_f64().unwrap(), 1e-9, "between");
    close(combined.covariance[d + 1], case["total"].as_f64().unwrap(), 1e-9, "total");
}

#[test]
fn malformed_soft_exposures_and_a_single_imputation_refuse() {
    let outcome = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
    let covariate = vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0];
    let bad = vec![0.1, 0.2, 1.5, 0.4, 0.5, 0.6, 0.7, 0.8];
    let data = SoftExposureData {
        outcome: &outcome,
        probability: &bad,
        covariates: &covariate,
        covariate_count: 1,
    };
    let err = correct_soft_exposure_effect(&data, &SoftExposureCombination::PosteriorMean, &ctx())
        .unwrap_err();
    assert_eq!(err.code, antecedent_core::reason_code!("invalid_argument"));
    assert_eq!(err.detail, antecedent_core::reason_details::measurement_model::INPUT_INVALID);
    let good = vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8];
    let data = SoftExposureData { probability: &good, ..data };
    let err = correct_soft_exposure_effect(
        &data,
        &SoftExposureCombination::SuppliedDraws { draws: &[0.0; 8], draw_count: 1 },
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(err.code, antecedent_core::reason_code!("measurement_not_identified"));
    assert_eq!(
        err.detail,
        antecedent_core::reason_details::measurement_model::INSUFFICIENT_REPLICATES
    );
}

#[test]
fn reliability_bins_match_exact_rational_arithmetic() {
    let case = &reference()["reliability"];
    let probability = floats(&case["probability"]);
    let rows: Vec<Vec<u8>> = case["codings"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| r.as_array().unwrap().iter().map(|v| v.as_u64().unwrap() as u8).collect())
        .collect();
    let sizes: Vec<u32> = rows.iter().map(|r| r.len() as u32).collect();
    let codings: Vec<u8> = rows.into_iter().flatten().collect();
    let reference_rates = floats(&case["reference"]);
    let reference_error = rates(reference_rates[0], reference_rates[1], 0.0, 0.0);
    let bins = case["bins"].as_u64().unwrap() as usize;
    let report = soft_exposure_reliability_report(
        &probability,
        &codings,
        &sizes,
        bins,
        Some(&reference_error),
        &ctx(),
    )
    .unwrap();
    let want = case["result"].as_array().unwrap();
    assert_eq!(report.bins.len(), want.len());
    for (got, want) in report.bins.iter().zip(want) {
        assert_eq!(got.units as u64, want["units"].as_u64().unwrap());
        close(got.mean_expected, want["mean_expected"].as_f64().unwrap(), 1e-12, "expected");
        close(got.mean_observed, want["mean_observed"].as_f64().unwrap(), 1e-12, "observed");
        close(got.gap, want["gap"].as_f64().unwrap(), 1e-9, "gap");
        close(got.standard_error.unwrap(), want["standard_error"].as_f64().unwrap(), 1e-9, "se");
    }
}

fn detection_rows(per: usize, hits: &[u64]) -> (Vec<bool>, Vec<f64>) {
    let mut detected = Vec::new();
    let mut covariate = Vec::new();
    for (c, &h) in hits.iter().enumerate() {
        for k in 0..per {
            detected.push((k as u64) < h);
            covariate.push(c as f64);
        }
    }
    (detected, covariate)
}

#[test]
fn logistic_detection_matches_the_numerically_differentiated_oracle() {
    let case = &reference()["logistic"];
    let hits: Vec<u64> =
        case["hits"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
    let (detected, covariate) = detection_rows(case["per"].as_u64().unwrap() as usize, &hits);
    let data = DetectionData { detected: &detected, covariate: &covariate };
    let declaration = DetectionDeclaration::Full(DetectionFunction::Logistic {
        intercept: case["intercept"].as_f64().unwrap(),
        slope: case["slope"].as_f64().unwrap(),
        intercept_variance: case["intercept_variance"].as_f64().unwrap(),
        covariance: case["covariance"].as_f64().unwrap(),
        slope_variance: case["slope_variance"].as_f64().unwrap(),
    });
    let fp = FalsePositiveRate {
        rate: case["false_positive"].as_f64().unwrap(),
        variance: case["false_positive_variance"].as_f64().unwrap(),
    };
    let result =
        correct_detection_dependent_prevalence(&data, &declaration, &fp, 0.05, &ctx()).unwrap();
    let DetectionAdjusted::Point(point) = result.adjusted else { panic!("a point is expected") };
    close(point.prevalence, case["prevalence"].as_f64().unwrap(), 1e-12, "prevalence");
    close(point.variance_sampling, case["variance_sampling"].as_f64().unwrap(), 1e-9, "sampling");
    close(
        point.variance_parameters,
        case["variance_parameters"].as_f64().unwrap(),
        1e-7,
        "parameters",
    );
    let estimate = result.estimate.expect("a fully declared function has a point estimate");
    assert_eq!(estimate.parameter_uncertainty, ParameterUncertainty::Propagated);
}

#[test]
fn monotone_envelope_bounds_match_exact_rational_arithmetic() {
    let case = &reference()["bounds"];
    let hits: Vec<u64> =
        case["hits"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
    let (detected, covariate) = detection_rows(case["per"].as_u64().unwrap() as usize, &hits);
    let data = DetectionData { detected: &detected, covariate: &covariate };
    let anchors = case["anchors"]
        .as_array()
        .unwrap()
        .iter()
        .map(|a| {
            let a = floats(a);
            DetectionAnchor { covariate: a[0], lower: a[1], upper: a[2] }
        })
        .collect();
    let declaration =
        DetectionDeclaration::Partial { direction: DetectionDirection::Nondecreasing, anchors };
    let result = correct_detection_dependent_prevalence(
        &data,
        &declaration,
        &FalsePositiveRate::NONE,
        0.1,
        &ctx(),
    )
    .unwrap();
    let DetectionAdjusted::Bounds(bounds) = result.adjusted else { panic!("bounds are expected") };
    close(bounds.lower, case["lower"].as_f64().unwrap(), 1e-12, "lower");
    close(bounds.upper, case["upper"].as_f64().unwrap(), 1e-12, "upper");
    close(bounds.mean_detection_lower, case["mean_lower"].as_f64().unwrap(), 1e-12, "mean lower");
    close(bounds.mean_detection_upper, case["mean_upper"].as_f64().unwrap(), 1e-12, "mean upper");
    assert!(result.estimate.is_none(), "bounds carry no point estimate");
}

#[test]
fn non_identifying_detection_refuses_with_the_registered_code() {
    let (detected, covariate) = detection_rows(100, &[10, 10, 10, 10]);
    let data = DetectionData { detected: &detected, covariate: &covariate };
    let weak = DetectionDeclaration::Full(DetectionFunction::Table {
        knots: vec![0.0, 3.0],
        values: vec![0.05, 0.08],
    });
    let err =
        correct_detection_dependent_prevalence(&data, &weak, &FalsePositiveRate::NONE, 0.1, &ctx())
            .unwrap_err();
    assert_eq!(err.code, antecedent_core::reason_code!("measurement_not_identified"));
    assert_eq!(err.detail, antecedent_core::reason_details::measurement_model::YOUDEN_TOO_SMALL);
}

#[test]
fn latent_class_fit_matches_the_multiprecision_maximum_likelihood_oracle() {
    for case in reference()["latent_class"].as_array().unwrap() {
        let k = case["k"].as_u64().unwrap() as usize;
        let c = case["c"].as_u64().unwrap() as usize;
        let j = case["j"].as_u64().unwrap() as usize;
        let counts: Vec<u64> =
            case["counts"].as_array().unwrap().iter().map(|v| v.as_u64().unwrap()).collect();
        // Patterns are enumerated with the first coding most significant, as the oracle does.
        let mut codings: Vec<u8> = Vec::new();
        let mut units = 0_usize;
        for (pattern, &count) in counts.iter().enumerate() {
            for _ in 0..count {
                for coding in 0..k {
                    let digit = (pattern / c.pow((k - 1 - coding) as u32)) % c;
                    codings.push(digit as u8);
                }
                units += 1;
            }
        }
        let data = LatentClassData {
            codings: &codings,
            units,
            coding_count: k,
            categories: c,
            latent_levels: j,
        };
        let settings =
            LatentClassSettings { tolerance: 1e-14, max_iterations: 50_000, ..Default::default() };
        let fit = fit_repeat_codings(&data, &LatentClassConstraints::default(), &settings, &ctx())
            .unwrap();
        for (got, want) in fit.prevalence.iter().zip(floats(&case["prevalence"])) {
            close(*got, want, 1e-5, "prevalence");
        }
        let response = case["response"].as_array().unwrap();
        for coding in 0..k {
            for level in 0..j {
                let want = floats(&response[coding].as_array().unwrap()[level]);
                for (category, w) in want.iter().enumerate() {
                    close(fit.response_of(coding, level, category), *w, 1e-4, "response");
                }
            }
        }
        close(fit.log_likelihood, case["log_likelihood"].as_f64().unwrap(), 1e-9, "log likelihood");
    }
}

#[test]
fn repeat_codings_that_cannot_identify_the_latent_classes_refuse_with_the_registered_code() {
    // Two binary codings with no declared constraints.
    let codings: Vec<u8> =
        (0..400).flat_map(|i| [u8::from(i % 3 == 0), u8::from(i % 5 == 0)]).collect();
    let data = LatentClassData {
        codings: &codings,
        units: 400,
        coding_count: 2,
        categories: 2,
        latent_levels: 2,
    };
    let err = fit_repeat_codings(
        &data,
        &LatentClassConstraints::default(),
        &LatentClassSettings::default(),
        &ctx(),
    )
    .unwrap_err();
    assert_eq!(err.code, antecedent_core::reason_code!("measurement_not_identified"));
    assert_eq!(
        err.detail,
        antecedent_core::reason_details::measurement_model::INSUFFICIENT_REPLICATES
    );
}
