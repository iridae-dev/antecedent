//! 2.2 E7: descriptive raw-versus-adjusted comparison and declared reporting-scale
//! transforms, against hand-computed values and an independent finite-difference Jacobian,
//! plus the typed refusals and typed-unavailable fields.

#![allow(clippy::float_cmp, reason = "exact comparisons of values built from the same operations")]

use std::sync::Arc;

use antecedent_core::ExecutionContext;
use antecedent_estimate::{
    AdjustedEstimate, Availability, EstimationError, JointCovariance, MeanPair, ReportingScale,
    compare_raw_adjusted, raw_contrast, refuse_column_attribution, refuse_transform_interval,
    transform_mean_pair,
};

fn ctx() -> ExecutionContext {
    ExecutionContext::for_tests(0)
}

fn close(actual: f64, expected: f64, what: &str) {
    assert!(
        (actual - expected).abs() <= 1e-12 * expected.abs().max(1.0),
        "{what}: {actual} vs {expected}"
    );
}

fn refused(error: EstimationError) -> (&'static str, String) {
    let EstimationError::Refused { code, message } = error else {
        panic!("expected a coded refusal, got {error:?}");
    };
    (code, message)
}

/// Control `[1, 2, 3]` (mean 2, s^2 1), active `[4, 6, 8]` (mean 6, s^2 4).
fn two_arms() -> (Vec<f64>, Vec<f64>) {
    (vec![1.0, 2.0, 3.0, 4.0, 6.0, 8.0], vec![0.0, 0.0, 0.0, 1.0, 1.0, 1.0])
}

/// Active `[1, 1, 0, 1]` (p 0.75), control `[0, 0, 1, 0]` (p 0.25): both s^2 = 0.25, n = 4.
fn binary_arms() -> (Vec<f64>, Vec<f64>) {
    (vec![1.0, 1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0], vec![1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0])
}

#[test]
fn the_raw_contrast_matches_the_hand_computation() {
    let (y, t) = two_arms();
    let raw = raw_contrast(&y, &t, &ctx()).unwrap();
    assert_eq!((raw.active.n, raw.control.n), (3, 3));
    close(raw.active.mean, 6.0, "active mean");
    close(raw.control.mean, 2.0, "control mean");
    close(raw.active.m2, 8.0, "active m2");
    close(raw.control.m2, 2.0, "control m2");
    close(raw.difference, 4.0, "raw difference");
    // Welch: sqrt(4/3 + 1/3) = sqrt(5/3).
    close(*raw.se.available().unwrap(), (5.0_f64 / 3.0).sqrt(), "Welch SE");
}

#[test]
fn the_gap_is_raw_minus_adjusted_and_carries_no_interval() {
    let (y, t) = two_arms();
    let comparison =
        compare_raw_adjusted(&y, &t, AdjustedEstimate::mean_difference(3.5, Some(0.4)), &ctx())
            .unwrap();
    close(comparison.gap, 0.5, "gap");
    close(comparison.raw.difference, 4.0, "raw");
    close(comparison.adjusted.estimate, 3.5, "adjusted");
    // The covariance of the raw and the adjusted estimate is not carried: the gap has a typed
    // reason instead of a standard error, and the adjusted SE is never folded into it.
    assert_eq!(comparison.gap_interval.code, "cell_not_licensed");
    assert_eq!(comparison.gap_interval.detail, "descriptive_comparison.gap_interval_unavailable");
}

#[test]
fn a_gap_is_zero_when_the_adjusted_estimate_equals_the_raw_contrast() {
    let (y, t) = two_arms();
    let comparison =
        compare_raw_adjusted(&y, &t, AdjustedEstimate::mean_difference(4.0, None), &ctx()).unwrap();
    close(comparison.gap, 0.0, "gap");
}

#[test]
fn an_adjusted_estimate_of_another_coding_or_population_is_not_compared() {
    let (y, t) = two_arms();
    for (adjusted, code, detail) in [
        (
            AdjustedEstimate { active: 2.0, ..AdjustedEstimate::mean_difference(3.0, None) },
            "route_not_supported",
            "descriptive_comparison.estimand_coding_mismatch",
        ),
        (
            AdjustedEstimate { control: 1.0, ..AdjustedEstimate::mean_difference(3.0, None) },
            "route_not_supported",
            "descriptive_comparison.estimand_coding_mismatch",
        ),
        (
            AdjustedEstimate {
                scale: ReportingScale::LogRiskRatio,
                ..AdjustedEstimate::mean_difference(0.3, None)
            },
            "route_not_supported",
            "descriptive_comparison.estimand_coding_mismatch",
        ),
        (
            AdjustedEstimate {
                all_observed_population: false,
                ..AdjustedEstimate::mean_difference(3.0, None)
            },
            "population_not_estimable",
            "descriptive_comparison.population_not_all_observed",
        ),
        (
            AdjustedEstimate::mean_difference(f64::NAN, None),
            "invalid_argument",
            "descriptive_comparison.invalid_data",
        ),
        (
            AdjustedEstimate::mean_difference(3.0, Some(-1.0)),
            "invalid_argument",
            "descriptive_comparison.invalid_data",
        ),
    ] {
        let (got, message) = refused(compare_raw_adjusted(&y, &t, adjusted, &ctx()).unwrap_err());
        assert_eq!(got, code, "{detail}");
        assert!(message.starts_with(detail), "{message}");
    }
}

#[test]
fn malformed_rows_and_empty_arms_are_refused() {
    let invalid = "descriptive_comparison.invalid_data";
    for (y, t) in [
        (vec![], vec![]),
        (vec![1.0, 2.0], vec![0.0]),
        (vec![1.0, f64::NAN], vec![0.0, 1.0]),
        (vec![1.0, 2.0, 3.0], vec![0.0, 1.0, 2.0]),
        (vec![1.0, 2.0], vec![0.0, f64::NAN]),
    ] {
        let (code, message) = refused(raw_contrast(&y, &t, &ctx()).unwrap_err());
        assert_eq!(code, "invalid_argument");
        assert!(message.starts_with(invalid), "{message}");
    }
    let (code, message) = refused(raw_contrast(&[1.0, 2.0], &[1.0, 1.0], &ctx()).unwrap_err());
    assert_eq!(code, "arm_not_populated");
    assert!(message.starts_with("descriptive_comparison.arm_not_populated"), "{message}");
}

#[test]
fn a_one_row_arm_has_a_typed_reason_instead_of_a_standard_error() {
    let raw = raw_contrast(&[1.0, 2.0, 5.0], &[0.0, 0.0, 1.0], &ctx()).unwrap();
    close(raw.difference, 5.0 - 1.5, "difference");
    let reason = raw.se.unavailable().expect("a one-row arm has no sample variance");
    assert_eq!(reason.code, "invalid_argument");
    assert_eq!(reason.detail, "descriptive_comparison.arm_too_small");
    assert_eq!(raw.mean_pair().covariance.unavailable(), Some(reason));
}

#[test]
fn log_risk_ratio_and_its_delta_method_variance_match_the_hand_computation() {
    // Means (0.4, 0.2), Sigma = [[0.01, 0.002], [0.002, 0.004]].
    // ln(0.4/0.2) = ln 2; gradient (1/0.4, -1/0.2) = (2.5, -5);
    // variance = 6.25 * 0.01 + 2 * (2.5 * -5) * 0.002 + 25 * 0.004 = 0.1125.
    let pair = MeanPair::new(0.4, 0.2, Some([0.01, 0.002, 0.004]));
    let transform = transform_mean_pair(&pair, &[ReportingScale::LogRiskRatio]).unwrap();
    close(transform.contrasts[0].value, 2.0_f64.ln(), "log risk ratio");
    assert_eq!(transform.contrasts[0].gradient, [1.0 / 0.4, -1.0 / 0.2]);
    let Availability::Available(covariance) = &transform.covariance else {
        panic!("the pair carries its joint covariance");
    };
    close(covariance[0], 0.1125, "variance");
    close(transform.se(0).unwrap(), 0.1125_f64.sqrt(), "se");
}

#[test]
fn a_transform_rejects_more_scales_than_the_declared_bound() {
    let pair = MeanPair::new(0.4, 0.2, None);
    let scales = [ReportingScale::MeanDifference; 5];
    let (code, message) = refused(transform_mean_pair(&pair, &scales).unwrap_err());
    assert_eq!(code, "invalid_argument");
    assert!(message.contains("at most four reporting scales"), "{message}");
}

#[test]
fn the_family_covariance_matches_an_independent_finite_difference_jacobian() {
    let (mu1, mu0) = (0.4, 0.2);
    let sigma = [0.01, 0.002, 0.004];
    let pair = MeanPair::new(mu1, mu0, Some(sigma));
    let scales = [
        ReportingScale::MeanDifference,
        ReportingScale::RiskDifference,
        ReportingScale::LogRiskRatio,
        ReportingScale::LogOddsRatio,
    ];
    let transform = transform_mean_pair(&pair, &scales).unwrap();
    // Independent oracle: each transform written directly, differentiated by central
    // differences, and J Sigma J' formed from the numerical Jacobian.
    let functions: [fn(f64, f64) -> f64; 4] = [
        |a, c| a - c,
        |a, c| a - c,
        |a, c| (a / c).ln(),
        |a, c| (a / (1.0 - a)).ln() - (c / (1.0 - c)).ln(),
    ];
    let h = 1e-6;
    let jacobian: Vec<[f64; 2]> = functions
        .iter()
        .map(|f| {
            [
                (f(mu1 + h, mu0) - f(mu1 - h, mu0)) / (2.0 * h),
                (f(mu1, mu0 + h) - f(mu1, mu0 - h)) / (2.0 * h),
            ]
        })
        .collect();
    let Availability::Available(covariance) = &transform.covariance else {
        panic!("the pair carries its joint covariance");
    };
    let k = scales.len();
    for (i, f) in functions.iter().enumerate() {
        assert!((transform.contrasts[i].value - f(mu1, mu0)).abs() < 1e-12, "value {i}");
        for j in 0..k {
            let expected = jacobian[i][0] * jacobian[j][0] * sigma[0]
                + (jacobian[i][0] * jacobian[j][1] + jacobian[i][1] * jacobian[j][0]) * sigma[1]
                + jacobian[i][1] * jacobian[j][1] * sigma[2];
            let actual = covariance[j * k + i];
            assert!(
                (actual - expected).abs() <= 1e-6 * expected.abs().max(1e-3),
                "covariance ({i}, {j}): {actual} vs {expected}"
            );
            assert_eq!(actual, covariance[i * k + j], "covariance is symmetric");
        }
    }
    // The mean and risk differences are the same linear functional of the pair.
    close(covariance[1], covariance[0], "risk difference variance equals mean difference");
}

#[test]
fn raw_arm_means_carry_the_independent_groups_covariance_into_a_transform() {
    // Active p = 0.75, control p = 0.25, each s^2/n = 0.0625 (see `binary_arms`).
    let (y, t) = binary_arms();
    let raw = raw_contrast(&y, &t, &ctx()).unwrap();
    let pair = raw.mean_pair();
    assert_eq!(pair.covariance, Availability::Available([0.0625, 0.0, 0.0625]));
    let transform =
        transform_mean_pair(&pair, &[ReportingScale::LogRiskRatio, ReportingScale::LogOddsRatio])
            .unwrap();
    close(transform.contrasts[0].value, 3.0_f64.ln(), "log risk ratio");
    close(transform.contrasts[1].value, 2.0 * 3.0_f64.ln(), "log odds ratio");
    // Hand: log RR gradient (4/3, -4), variance 16/9 * 1/16 + 16 * 1/16 = 10/9.
    close(transform.se(0).unwrap(), (10.0_f64 / 9.0).sqrt(), "log RR se");
    // Hand: log OR gradient (16/3, -16/3), variance 2 * (256/9) * 1/16 = 32/9.
    close(transform.se(1).unwrap(), (32.0_f64 / 9.0).sqrt(), "log OR se");
}

#[test]
fn a_missing_joint_covariance_gives_points_and_a_typed_reason_never_a_zero() {
    let pair = MeanPair::new(0.4, 0.2, None);
    let transform =
        transform_mean_pair(&pair, &[ReportingScale::LogRiskRatio, ReportingScale::RiskDifference])
            .unwrap();
    close(transform.contrasts[0].value, 2.0_f64.ln(), "point");
    close(transform.contrasts[1].value, 0.2, "point");
    let reason = transform.covariance.unavailable().expect("no covariance was carried");
    assert_eq!(reason.code, "required_option_missing");
    assert_eq!(reason.detail, "descriptive_comparison.joint_covariance_unavailable");
    assert_eq!(transform.se(0), None);
}

#[test]
fn a_joint_covariance_is_read_at_the_named_entries() {
    // Three functionals; the pair is functionals 2 (active) and 0 (control).
    let values: Vec<f64> = vec![0.004, 0.0, 0.002, 0.0, 9.0, 0.0, 0.002, 0.0, 0.01];
    let joint = JointCovariance { dim: 3, values: Arc::from(values) };
    let pair = MeanPair::from_joint_covariance(0.4, 0.2, &joint, 2, 0).unwrap();
    assert_eq!(pair.covariance, Availability::Available([0.01, 0.002, 0.004]));
    for (i, j) in [(1, 1), (0, 3), (3, 0)] {
        let (code, message) =
            refused(MeanPair::from_joint_covariance(0.4, 0.2, &joint, i, j).unwrap_err());
        assert_eq!(code, "invalid_argument");
        assert!(message.starts_with("descriptive_comparison.invalid_covariance"), "{message}");
    }
    let asymmetric =
        JointCovariance { dim: 2, values: Arc::from(vec![0.01_f64, 0.002, 0.003, 0.004]) };
    let (code, _) =
        refused(MeanPair::from_joint_covariance(0.4, 0.2, &asymmetric, 0, 1).unwrap_err());
    assert_eq!(code, "invalid_argument");
}

#[test]
fn a_transform_outside_its_domain_or_on_a_bad_covariance_is_refused() {
    let log_rr = [ReportingScale::LogRiskRatio];
    for (pair, scales, detail) in [
        (MeanPair::new(0.4, 0.0, None), &log_rr[..], "descriptive_comparison.transform_domain"),
        (MeanPair::new(-0.1, 0.2, None), &log_rr[..], "descriptive_comparison.transform_domain"),
        (
            MeanPair::new(1.0, 0.2, None),
            &[ReportingScale::LogOddsRatio][..],
            "descriptive_comparison.transform_domain",
        ),
        (
            MeanPair::new(1.2, 0.2, None),
            &[ReportingScale::RiskDifference][..],
            "descriptive_comparison.transform_domain",
        ),
        (MeanPair::new(f64::NAN, 0.2, None), &log_rr[..], "descriptive_comparison.invalid_data"),
        (MeanPair::new(0.4, 0.2, None), &[][..], "descriptive_comparison.invalid_data"),
        (
            MeanPair::new(0.4, 0.2, Some([0.01, 0.02, 0.004])),
            &log_rr[..],
            "descriptive_comparison.invalid_covariance",
        ),
        (
            MeanPair::new(0.4, 0.2, Some([-0.01, 0.0, 0.004])),
            &log_rr[..],
            "descriptive_comparison.invalid_covariance",
        ),
        (
            MeanPair::new(0.4, 0.2, Some([f64::NAN, 0.0, 0.004])),
            &log_rr[..],
            "descriptive_comparison.invalid_covariance",
        ),
    ] {
        let (code, message) = refused(transform_mean_pair(&pair, scales).unwrap_err());
        assert_eq!(code, "invalid_argument", "{detail}");
        assert!(message.starts_with(detail), "{message}");
    }
    // A mean difference has no domain restriction: a continuous-outcome pair is fine.
    let transform =
        transform_mean_pair(&MeanPair::new(-3.0, 12.5, None), &[ReportingScale::MeanDifference])
            .unwrap();
    close(transform.contrasts[0].value, -15.5, "mean difference");
}

#[test]
fn scale_names_round_trip_and_an_unknown_scale_is_refused() {
    for scale in [
        ReportingScale::MeanDifference,
        ReportingScale::RiskDifference,
        ReportingScale::LogRiskRatio,
        ReportingScale::LogOddsRatio,
    ] {
        assert_eq!(ReportingScale::parse(scale.name()).unwrap(), scale);
    }
    let (code, message) = refused(ReportingScale::parse("hazard_ratio").unwrap_err());
    assert_eq!(code, "route_not_supported");
    assert!(message.starts_with("descriptive_comparison.scale_not_supported"), "{message}");
}

#[test]
fn an_interval_and_a_column_attribution_are_closed_routes() {
    let (code, message) = refused(refuse_transform_interval());
    assert_eq!(code, "cell_not_licensed");
    assert!(message.starts_with("descriptive_comparison.interval_withheld"), "{message}");
    let (code, message) = refused(refuse_column_attribution());
    assert_eq!(code, "effect_not_identified");
    assert!(
        message.starts_with("descriptive_comparison.column_attribution_not_identified"),
        "{message}"
    );
}
