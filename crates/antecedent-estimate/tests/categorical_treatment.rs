//! B4 categorical treatments: group-mean closed forms for the dummy-coded regression, contrast
//! variances from the full covariance, Holm multiplicity written out, sparse/absent level
//! refusals, unordered level-permutation invariance, and the declared monotonicity test with
//! hand-derived statistics. Calibration (Type I error, power) is deliberately NOT measured here.
//!
//! Frozen dataset A (rows interleaved on purpose): `a = [1,3]` (mean 2, SS 2),
//! `b = [4,6,5]` (mean 5, SS 2), `c = [9,7,11,9]` (mean 9, SS 8); `n = 9`, `J = 3`,
//! `RSS = 12`, `sigma^2 = 12 / 6 = 2`. With reference `a`: `b_b = 3`, `b_c = 7`,
//! `V = sigma^2 [[1/3 + 1/2, 1/2], [1/2, 1/4 + 1/2]] = [[5/3, 1], [1, 3/2]]`.
//!
//! Monotonicity datasets: three ordered levels `lo < mid < hi`, three rows each, outcome
//! `mean_g + [-1, 0, 1]`: `sigma^2 = 6 / 6 = 1`, every adjacent-step variance is
//! `sigma^2 (1/3 + 1/3) = 2/3` (the step covariance `-1/3` cancels in `Var(b_hi - b_mid)`).

use antecedent_estimate::EstimationError;
use antecedent_estimate::categorical_treatment::{
    CategoricalTreatmentInput, CategoricalTreatmentSpec, LevelScale,
    MONOTONICITY_NON_DECREASING_NULL, MONOTONICITY_NON_INCREASING_NULL, MonotonicityDirection,
    fit_categorical_treatment,
};
use antecedent_estimate::effect_constancy::CalibrationStatus;
use antecedent_estimate::vector_treatment::VectorCovariance;
use antecedent_kernels::erfc;

fn close(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() <= 1e-9, "{what}: {actual} vs {expected}");
}

fn refusal(error: EstimationError) -> (&'static str, String) {
    let EstimationError::Refused { code, message } = error else {
        panic!("expected a coded refusal, got {error:?}");
    };
    (code, message)
}

fn normal_p(z: f64) -> f64 {
    erfc(z.abs() / 2.0_f64.sqrt())
}

fn abc_input() -> CategoricalTreatmentInput {
    let rows = [
        ("c", 9.0),
        ("a", 1.0),
        ("b", 4.0),
        ("c", 7.0),
        ("b", 6.0),
        ("a", 3.0),
        ("c", 11.0),
        ("b", 5.0),
        ("c", 9.0),
    ];
    CategoricalTreatmentInput {
        outcome: rows.iter().map(|r| r.1).collect(),
        row_snapshot: "snap".into(),
        adjustment: vec![],
        levels: rows.iter().map(|r| r.0.to_string()).collect(),
    }
}

fn names(levels: &[&str]) -> Vec<String> {
    levels.iter().map(|l| (*l).to_string()).collect()
}

fn spec(levels: &[&str], scale: LevelScale, reference: &str) -> CategoricalTreatmentSpec {
    CategoricalTreatmentSpec {
        declared_levels: names(levels),
        scale,
        reference: reference.into(),
        min_level_rows: 2,
        pairwise: vec![],
        monotonicity: None,
        covariance: VectorCovariance::ModelBased,
    }
}

#[test]
fn b4_categorical_dummy_regression_covariance_matches_group_mean_algebra() {
    let fit = fit_categorical_treatment(
        &abc_input(),
        &spec(&["a", "b", "c"], LevelScale::Unordered, "a"),
    )
    .unwrap();
    assert_eq!(fit.level_order, ["a", "b", "c"]);
    assert_eq!(fit.counts.iter().map(|c| c.rows).collect::<Vec<_>>(), [2, 3, 4]);
    assert_eq!(fit.coefficients[0].name, "level:b");
    assert_eq!(fit.coefficients[1].name, "level:c");
    close(fit.coefficients[0].estimate, 3.0, "b - a");
    close(fit.coefficients[1].estimate, 7.0, "c - a");
    close(fit.covariance[0], 5.0 / 3.0, "V_bb");
    close(fit.covariance[1], 1.0, "V_bc");
    close(fit.covariance[2], 1.0, "V_cb");
    close(fit.covariance[3], 1.5, "V_cc");
    assert_eq!(fit.level_contrasts.len(), 2);
    assert_eq!(fit.level_contrasts[0].level, "b");
    assert_eq!(fit.level_contrasts[0].reference, "a");
    close(fit.level_contrasts[0].standard_error, (5.0_f64 / 3.0).sqrt(), "se b");
    close(fit.level_contrasts[1].standard_error, 1.5_f64.sqrt(), "se c");
    close(fit.level_contrasts[1].z, 7.0 / 1.5_f64.sqrt(), "z c");
    close(fit.level_contrasts[1].p_value, normal_p(7.0 / 1.5_f64.sqrt()), "p c");
    assert_eq!(fit.calibration, CalibrationStatus::Unmeasured);
    assert!(fit.monotonicity.is_none());
}

#[test]
fn b4_categorical_omnibus_wald_matches_closed_form() {
    let fit = fit_categorical_treatment(
        &abc_input(),
        &spec(&["a", "b", "c"], LevelScale::Unordered, "a"),
    )
    .unwrap();
    // b = (3, 7), V = [[5/3, 1], [1, 3/2]], det V = 3/2, V^-1 = [[3/2, -1], [-1, 5/3]] / (3/2).
    let statistic = (1.5 * 9.0 - 2.0 * 21.0 + 5.0 / 3.0 * 49.0) / 1.5;
    close(statistic, 319.0 / 9.0, "closed form");
    close(fit.omnibus.statistic, statistic, "omnibus");
    assert_eq!(fit.omnibus.degrees_of_freedom, 2);
    close(fit.omnibus.p_value, (-statistic / 2.0).exp(), "omnibus p (df 2)");
}

#[test]
fn b4_categorical_pairwise_contrast_uses_the_dummy_covariance() {
    let mut s = spec(&["a", "b", "c"], LevelScale::Unordered, "a");
    s.pairwise = vec![("b".into(), "c".into())];
    let fit = fit_categorical_treatment(&abc_input(), &s).unwrap();
    let pair = &fit.pairwise[0];
    assert_eq!((pair.from.as_str(), pair.to.as_str()), ("b", "c"));
    close(pair.estimate, 4.0, "c - b");
    // Direct two-sample variance: sigma^2 (1/4 + 1/3) = 7/6; the dummy covariance gives
    // 5/3 + 3/2 - 2 * 1 = 7/6 only because the off-diagonal is used (the naive sum is 19/6).
    close(pair.standard_error.powi(2), 7.0 / 6.0, "pair variance");
    assert!((pair.standard_error.powi(2) - 19.0 / 6.0).abs() > 1.0);
    close(pair.z, 4.0 / (7.0_f64 / 6.0).sqrt(), "pair z");
}

#[test]
fn b4_categorical_non_alphabetical_reference_changes_the_covariance_as_derived() {
    let fit = fit_categorical_treatment(
        &abc_input(),
        &spec(&["a", "b", "c"], LevelScale::Unordered, "b"),
    )
    .unwrap();
    assert_eq!(fit.coefficients[0].name, "level:a");
    assert_eq!(fit.coefficients[1].name, "level:c");
    close(fit.coefficients[0].estimate, -3.0, "a - b");
    close(fit.coefficients[1].estimate, 4.0, "c - b");
    // Var(a - b) = 2 (1/2 + 1/3), Var(c - b) = 2 (1/4 + 1/3), Cov = sigma^2 / n_b = 2/3.
    close(fit.covariance[0], 5.0 / 3.0, "V_aa");
    close(fit.covariance[1], 2.0 / 3.0, "V_ac");
    close(fit.covariance[3], 7.0 / 6.0, "V_cc");
}

#[test]
fn b4_categorical_holm_across_the_declared_family_is_written_out() {
    let mut s = spec(&["a", "b", "c"], LevelScale::Unordered, "a");
    s.pairwise = vec![("b".into(), "c".into())];
    let fit = fit_categorical_treatment(&abc_input(), &s).unwrap();
    assert_eq!(fit.family_size, 3);
    let (b, c) = (&fit.level_contrasts[0], &fit.level_contrasts[1]);
    let pair = &fit.pairwise[0];
    // |z|: c = 7 / sqrt(1.5) = 5.71 > pair = 4 / sqrt(7/6) = 3.70 > b = 3 / sqrt(5/3) = 2.32,
    // so p_c < p_pair < p_b and Holm multiplies by 3, 2, 1 with a running maximum.
    close(c.p_value, normal_p(7.0 / 1.5_f64.sqrt()), "p_c");
    close(pair.p_value, normal_p(4.0 / (7.0_f64 / 6.0).sqrt()), "p_pair");
    close(b.p_value, normal_p(3.0 / (5.0_f64 / 3.0).sqrt()), "p_b");
    assert!(c.p_value < pair.p_value && pair.p_value < b.p_value);
    let holm_c = (3.0 * c.p_value).min(1.0);
    let holm_pair = (2.0 * pair.p_value).min(1.0).max(holm_c);
    let holm_b = b.p_value.min(1.0).max(holm_pair);
    close(c.p_holm, holm_c, "holm c");
    close(pair.p_holm, holm_pair, "holm pair");
    close(b.p_holm, holm_b, "holm b");
    assert!(b.p_holm >= b.p_value && c.p_holm >= c.p_value);
}

#[test]
fn b4_categorical_unordered_level_permutation_gives_identical_contrasts() {
    let mut base = spec(&["a", "b", "c"], LevelScale::Unordered, "a");
    base.pairwise = vec![("b".into(), "c".into()), ("c".into(), "a".into())];
    let mut permuted = base.clone();
    permuted.declared_levels = names(&["c", "a", "b"]);
    let one = fit_categorical_treatment(&abc_input(), &base).unwrap();
    let two = fit_categorical_treatment(&abc_input(), &permuted).unwrap();
    assert_eq!(one, two);
    let mut reversed = base.clone();
    reversed.declared_levels = names(&["c", "b", "a"]);
    assert_eq!(one, fit_categorical_treatment(&abc_input(), &reversed).unwrap());
}

#[test]
fn b4_categorical_ordered_levels_keep_the_declared_order() {
    // Declared order lo < mid < hi is not alphabetical; the scale must survive.
    let fit = fit_categorical_treatment(
        &grouped([0.0, 2.0, 4.0]),
        &spec(&["lo", "mid", "hi"], LevelScale::Ordered, "lo"),
    )
    .unwrap();
    assert_eq!(fit.level_order, ["lo", "mid", "hi"]);
    assert_eq!(fit.scale, LevelScale::Ordered);
    assert_eq!(fit.coefficients[0].name, "level:mid");
    assert_eq!(fit.coefficients[1].name, "level:hi");
    close(fit.coefficients[0].estimate, 2.0, "mid - lo");
    close(fit.coefficients[1].estimate, 4.0, "hi - lo");
}

#[test]
fn b4_categorical_two_level_factor_is_supported() {
    let rows = [("x", 1.0), ("y", 4.0), ("x", 3.0), ("y", 6.0), ("x", 2.0), ("y", 5.0)];
    let input = CategoricalTreatmentInput {
        outcome: rows.iter().map(|r| r.1).collect(),
        row_snapshot: "snap".into(),
        adjustment: vec![],
        levels: rows.iter().map(|r| r.0.to_string()).collect(),
    };
    let fit =
        fit_categorical_treatment(&input, &spec(&["x", "y"], LevelScale::Unordered, "x")).unwrap();
    // Means 2 and 5; each group SS = 2, sigma^2 = 4 / 4 = 1, Var(b) = 1 (1/3 + 1/3).
    close(fit.coefficients[0].estimate, 3.0, "y - x");
    close(fit.covariance[0], 2.0 / 3.0, "Var");
    assert_eq!(fit.omnibus.degrees_of_freedom, 1);
    close(fit.omnibus.statistic, 9.0 / (2.0 / 3.0), "z squared");
}

/// Three ordered levels `lo, mid, hi`, three rows each: `mean_g + [-1, 0, 1]`.
fn grouped(means: [f64; 3]) -> CategoricalTreatmentInput {
    let labels = ["lo", "mid", "hi"];
    let mut outcome = Vec::new();
    let mut levels = Vec::new();
    for (label, mean) in labels.iter().zip(means) {
        for offset in [-1.0, 0.0, 1.0] {
            outcome.push(mean + offset);
            levels.push((*label).to_string());
        }
    }
    CategoricalTreatmentInput { outcome, row_snapshot: "snap".into(), adjustment: vec![], levels }
}

fn monotone_spec(direction: MonotonicityDirection) -> CategoricalTreatmentSpec {
    let mut s = spec(&["lo", "mid", "hi"], LevelScale::Ordered, "lo");
    s.monotonicity = Some(direction);
    s
}

fn monotonicity(
    means: [f64; 3],
    direction: MonotonicityDirection,
) -> antecedent_estimate::categorical_treatment::MonotonicityResult {
    fit_categorical_treatment(&grouped(means), &monotone_spec(direction))
        .unwrap()
        .monotonicity
        .unwrap()
}

#[test]
fn b4_categorical_monotonicity_constant_effect_is_never_rejected() {
    let step_se = (2.0_f64 / 3.0).sqrt();
    for direction in [MonotonicityDirection::NonDecreasing, MonotonicityDirection::NonIncreasing] {
        let result = monotonicity([5.0, 5.0, 5.0], direction);
        assert_eq!(result.steps.len(), 2);
        assert_eq!((result.steps[0].from.as_str(), result.steps[0].to.as_str()), ("lo", "mid"));
        assert_eq!((result.steps[1].from.as_str(), result.steps[1].to.as_str()), ("mid", "hi"));
        close(result.steps[0].difference, 0.0, "d1");
        close(result.steps[1].difference, 0.0, "d2");
        close(result.steps[0].standard_error, step_se, "se d1");
        close(result.steps[1].standard_error, step_se, "se d2");
        close(result.statistic, 0.0, "T");
        // p = min(1, 2 Phi(0)) = 1.
        close(result.p_value, 1.0, "p");
        assert!(result.conservative);
        assert_eq!(result.calibration, CalibrationStatus::Unmeasured);
    }
}

#[test]
fn b4_categorical_monotonicity_increasing_effect() {
    // d = (2, 2), se = sqrt(2/3), z = sqrt(6).
    let z = 6.0_f64.sqrt();
    let up = monotonicity([0.0, 2.0, 4.0], MonotonicityDirection::NonDecreasing);
    assert_eq!(up.null, MONOTONICITY_NON_DECREASING_NULL);
    close(up.steps[0].difference, 2.0, "d1");
    close(up.steps[1].difference, 2.0, "d2");
    close(up.steps[0].z, z, "z1");
    close(up.statistic, z, "T");
    // 2 Phi(sqrt 6) > 1, so the bound is capped at one.
    close(up.p_value, 1.0, "p");
    // The opposite null (non-increasing) is rejected: T = -sqrt(6), p = 2 Phi(-sqrt 6) = erfc(sqrt 3).
    let down = monotonicity([0.0, 2.0, 4.0], MonotonicityDirection::NonIncreasing);
    assert_eq!(down.null, MONOTONICITY_NON_INCREASING_NULL);
    close(down.statistic, -z, "T");
    close(down.p_value, erfc(3.0_f64.sqrt()), "p");
    assert!(down.p_value < 0.05);
}

#[test]
fn b4_categorical_monotonicity_decreasing_effect() {
    let z = 6.0_f64.sqrt();
    let up = monotonicity([4.0, 2.0, 0.0], MonotonicityDirection::NonDecreasing);
    close(up.steps[0].difference, -2.0, "d1");
    close(up.statistic, -z, "T");
    close(up.p_value, erfc(3.0_f64.sqrt()), "p");
    let down = monotonicity([4.0, 2.0, 0.0], MonotonicityDirection::NonIncreasing);
    close(down.statistic, z, "T");
    close(down.p_value, 1.0, "p");
}

#[test]
fn b4_categorical_monotonicity_non_monotone_effect_rejects_both_directions() {
    // d = (3, -3), se = sqrt(2/3), |z| = 3 / sqrt(2/3) = sqrt(13.5).
    let z = 13.5_f64.sqrt();
    let expected_p = erfc(6.75_f64.sqrt()); // 2 Phi(-sqrt 13.5) = erfc(sqrt(13.5 / 2))
    for direction in [MonotonicityDirection::NonDecreasing, MonotonicityDirection::NonIncreasing] {
        let result = monotonicity([0.0, 3.0, 0.0], direction);
        close(result.steps[0].difference, 3.0, "d1");
        close(result.steps[1].difference, -3.0, "d2");
        close(result.steps[0].z.abs(), z, "|z1|");
        close(result.steps[1].z.abs(), z, "|z2|");
        close(result.statistic, -z, "T");
        close(result.p_value, expected_p, "p");
    }
}

#[test]
fn b4_categorical_monotonicity_single_step_is_the_exact_one_sided_test() {
    // Two levels: m = 1, so p = Phi(T) with no Bonferroni inflation.
    let rows = [("x", 1.0), ("y", 4.0), ("x", 3.0), ("y", 6.0), ("x", 2.0), ("y", 5.0)];
    let input = CategoricalTreatmentInput {
        outcome: rows.iter().map(|r| r.1).collect(),
        row_snapshot: "snap".into(),
        adjustment: vec![],
        levels: rows.iter().map(|r| r.0.to_string()).collect(),
    };
    let mut s = spec(&["x", "y"], LevelScale::Ordered, "x");
    s.monotonicity = Some(MonotonicityDirection::NonIncreasing);
    let result = fit_categorical_treatment(&input, &s).unwrap().monotonicity.unwrap();
    // d = 3, se = sqrt(2/3); oriented z = -3 / sqrt(2/3); p = Phi(z) = erfc(|z| / sqrt 2) / 2.
    let z = 3.0 / (2.0_f64 / 3.0).sqrt();
    close(result.statistic, -z, "T");
    close(result.p_value, 0.5 * erfc(z / 2.0_f64.sqrt()), "p");
}

#[test]
fn b4_categorical_refuses_monotonicity_on_unordered_levels() {
    let mut s = spec(&["a", "b", "c"], LevelScale::Unordered, "a");
    s.monotonicity = Some(MonotonicityDirection::NonDecreasing);
    let (code, message) = refusal(fit_categorical_treatment(&abc_input(), &s).unwrap_err());
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("categorical_treatment.monotonicity_requires_ordered"), "{message}");
}

#[test]
fn b4_categorical_absent_level_is_refused_by_name() {
    let s = spec(&["a", "b", "c", "d"], LevelScale::Unordered, "a");
    let (code, message) = refusal(fit_categorical_treatment(&abc_input(), &s).unwrap_err());
    assert_eq!(code, "arm_not_populated");
    assert!(message.contains("categorical_treatment.absent_level"), "{message}");
    assert!(message.contains("`d`"), "{message}");
}

#[test]
fn b4_categorical_sparse_level_is_refused_by_name_never_dropped() {
    let mut s = spec(&["a", "b", "c"], LevelScale::Unordered, "a");
    s.min_level_rows = 3; // level `a` has only two rows
    let (code, message) = refusal(fit_categorical_treatment(&abc_input(), &s).unwrap_err());
    assert_eq!(code, "arm_not_populated");
    assert!(message.contains("categorical_treatment.sparse_level"), "{message}");
    assert!(message.contains("`a`"), "{message}");
    assert!(message.contains("2 rows"), "{message}");
    // The same data pass once the declared minimum is met.
    s.min_level_rows = 2;
    assert!(fit_categorical_treatment(&abc_input(), &s).is_ok());
}

#[test]
fn b4_categorical_refuses_malformed_declarations() {
    let run = |s: &CategoricalTreatmentSpec, input: &CategoricalTreatmentInput| {
        refusal(fit_categorical_treatment(input, s).unwrap_err())
    };
    let base = spec(&["a", "b", "c"], LevelScale::Unordered, "a");

    let mut unknown_reference = base.clone();
    unknown_reference.reference = "z".into();
    let (code, message) = run(&unknown_reference, &abc_input());
    assert_eq!(code, "invalid_argument");
    assert!(message.contains("categorical_treatment.unknown_reference"), "{message}");

    let undeclared = spec(&["a", "b"], LevelScale::Unordered, "a");
    let (_, message) = run(&undeclared, &abc_input());
    assert!(message.contains("categorical_treatment.undeclared_level"), "{message}");
    assert!(message.contains("`c`"), "{message}");

    let one_level = spec(&["a"], LevelScale::Unordered, "a");
    let (_, message) = run(&one_level, &abc_input());
    assert!(message.contains("categorical_treatment.too_few_levels"), "{message}");

    let mut zero_min = base.clone();
    zero_min.min_level_rows = 0;
    let (_, message) = run(&zero_min, &abc_input());
    assert!(message.contains("categorical_treatment.invalid_min_level_rows"), "{message}");

    let mut bad_pair = base.clone();
    bad_pair.pairwise = vec![("a".into(), "a".into())];
    let (_, message) = run(&bad_pair, &abc_input());
    assert!(message.contains("categorical_treatment.invalid_pair"), "{message}");

    let mut unknown_pair = base.clone();
    unknown_pair.pairwise = vec![("a".into(), "q".into())];
    let (_, message) = run(&unknown_pair, &abc_input());
    assert!(message.contains("categorical_treatment.unknown_level"), "{message}");

    let mut short = abc_input();
    short.levels.pop();
    let (code, message) = run(&base, &short);
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("categorical_treatment.row_count_mismatch"), "{message}");
}
