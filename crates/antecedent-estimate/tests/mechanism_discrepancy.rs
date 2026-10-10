//! B3 mechanism discrepancy diagnostic: hand-derived OLS fits, summed covariances, the Wald
//! statistic with df 1 and 2 closed-form tails, the minimal detectable differences, the
//! comparability and dependence refusals, and the never-certifies-invariance fields. Type I error
//! and power are deliberately NOT measured here.
//!
//! Dataset A (one parent `x`, n = 4): `x = [0,1,2,3]`, `y_s = [1,3,2,4]`. `X'X = [[4,6],[6,14]]`,
//! `det = 20`, `X'y = [10,19]`, so `b = [1.3, 0.8]`; residuals `[-.3,.9,-.9,.3]`, `RSS = 1.8`,
//! `sigma^2 = 0.9`, `V = 0.9 (X'X)^-1 = [[0.63,-0.27],[-0.27,0.18]]`. Adding `c0 + c1 x` to the
//! outcome shifts the OLS coefficients by exactly `(c0, c1)` and leaves residuals and `V`
//! unchanged, so the target covariance equals the source covariance and the summed covariance is
//! `2 V = [[1.26,-0.54],[-0.54,0.36]]` (`det = 0.162`).
//!
//! Dataset B (two parents, n = 6): the `vector_treatment` frozen data, `a = [1,0,1,0,0,1]`,
//! `b = [1,1,1,0,0,0]`, `y = [3,2,5,1,0,4]`, `b_hat = [0.75, 2.75, 0.75]`, `sigma^2 = 3.25/3`,
//! slope block of `V` = `(3.25/36) [[9,-3],[-3,9]]`; the summed block is
//! `(3.25/18) [[9,-3],[-3,9]]`.

use antecedent_core::reason_code;
use antecedent_estimate::EstimationError;
use antecedent_estimate::effect_constancy::CalibrationStatus;
use antecedent_estimate::mechanism_discrepancy::{
    DiscrepancyConclusion, DiscrepancyOptions, MECHANISM_DISCREPANCY_NULL, MechanismMeasurement,
    MechanismSummary, ParentSpec, PopulationSample, SampleDependence, Z_ALPHA_TWO_SIDED_05,
    Z_POWER_80, normal_upper_quantile, summarize_population, test_mechanism_discrepancy,
    test_mechanism_discrepancy_from_summaries,
};
use antecedent_kernels::erfc;

const X: [f64; 4] = [0.0, 1.0, 2.0, 3.0];
const YS: [f64; 4] = [1.0, 3.0, 2.0, 4.0];
/// `YS + 0.5 x`: slope shifted by 0.5, intercept unchanged.
const YT_SLOPE: [f64; 4] = [1.0, 3.5, 3.0, 5.5];
/// `YS + 0.5 + 0.5 x`: intercept and slope both shifted by 0.5.
const YT_BOTH: [f64; 4] = [1.5, 4.0, 3.5, 6.0];
/// `YS + 5 x`.
const YT_BIG: [f64; 4] = [1.0, 8.0, 12.0, 19.0];

const A: [f64; 6] = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
const B: [f64; 6] = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
const Y6: [f64; 6] = [3.0, 2.0, 5.0, 1.0, 0.0, 4.0];
/// `Y6 + 1 a + 0.5 b`.
const Y6_TARGET: [f64; 6] = [4.5, 2.5, 6.5, 1.0, 0.0, 5.0];

fn close(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() <= 1e-9, "{what}: {actual} vs {expected}");
}

fn refusal(error: EstimationError) -> (&'static str, String) {
    let EstimationError::Refused { code, message } = error else {
        panic!("expected a coded refusal, got {error:?}");
    };
    (code, message)
}

fn measurement(protocol: &str, node_unit: &str, parents: &[(&str, &str)]) -> MechanismMeasurement {
    MechanismMeasurement {
        node: "V".into(),
        node_unit: node_unit.into(),
        parents: parents
            .iter()
            .map(|(name, unit)| ParentSpec { name: (*name).into(), unit: (*unit).into() })
            .collect(),
        protocol_id: protocol.into(),
    }
}

fn sample(label: &str, y: &[f64], parents: &[(&str, &str, &[f64])]) -> PopulationSample {
    sample_with(label, "protocol-1", "mg", y, parents)
}

fn sample_with(
    label: &str,
    protocol: &str,
    node_unit: &str,
    y: &[f64],
    parents: &[(&str, &str, &[f64])],
) -> PopulationSample {
    let specs: Vec<(&str, &str)> = parents.iter().map(|(n, u, _)| (*n, *u)).collect();
    PopulationSample {
        label: label.into(),
        measurement: measurement(protocol, node_unit, &specs),
        outcome: y.to_vec(),
        parent_values: parents.iter().map(|(_, _, v)| v.to_vec()).collect(),
        unit_ids: vec![],
    }
}

fn opts(compare_intercept: bool) -> DiscrepancyOptions {
    DiscrepancyOptions {
        compare_intercept,
        alpha: 0.05,
        power: 0.8,
        dependence: SampleDependence::Independent,
    }
}

fn one_parent(label: &str, y: &[f64]) -> PopulationSample {
    sample(label, y, &[("x", "cm", &X)])
}

fn two_parents(label: &str, y: &[f64]) -> PopulationSample {
    sample(label, y, &[("a", "u", &A), ("b", "u", &B)])
}

#[test]
fn b3_discrepancy_ols_fits_and_summed_covariance_match_hand_algebra() {
    let r = test_mechanism_discrepancy(
        &one_parent("source", &YS),
        &one_parent("target", &YT_BOTH),
        &opts(true),
    )
    .unwrap();
    assert_eq!(r.coefficient_names, ["(intercept)", "x"]);
    close(r.source.coefficients[0], 1.3, "source intercept");
    close(r.source.coefficients[1], 0.8, "source slope");
    close(r.target.coefficients[0], 1.8, "target intercept");
    close(r.target.coefficients[1], 1.3, "target slope");
    assert_eq!(r.source.n, 4);
    assert_eq!(r.source.residual_df, 2);
    close(r.source.residual_variance, 0.9, "sigma^2");
    close(r.target.residual_variance, 0.9, "target sigma^2");
    close(r.source.standard_errors[0], 0.63_f64.sqrt(), "se intercept");
    close(r.source.standard_errors[1], 0.18_f64.sqrt(), "se slope");
    assert_eq!(r.null, MECHANISM_DISCREPANCY_NULL);
    assert_eq!(r.calibration, CalibrationStatus::Unmeasured);
}

#[test]
#[allow(clippy::float_cmp, reason = "identical sufficient statistics give bitwise-equal fits")]
fn b3_discrepancy_identical_fits_give_statistic_zero_exactly() {
    let same = test_mechanism_discrepancy(
        &one_parent("source", &YS),
        &one_parent("target", &YS),
        &opts(true),
    )
    .unwrap();
    assert_eq!(same.test.statistic, 0.0);
    assert_eq!(same.test.p_value, 1.0);
    assert!(same.coefficients.iter().all(|c| c.difference == 0.0 && !c.rejected));
    assert_eq!(same.conclusion, DiscrepancyConclusion::NotRejected);

    // Same OLS fit from the rows in reverse order: identical sufficient statistics.
    let reversed = sample("target", &[4.0, 2.0, 3.0, 1.0], &[("x", "cm", &[3.0, 2.0, 1.0, 0.0])]);
    let r = test_mechanism_discrepancy(&one_parent("source", &YS), &reversed, &opts(true)).unwrap();
    assert_eq!(r.test.statistic, 0.0);
    assert_eq!(r.test.p_value, 1.0);
}

#[test]
fn b3_discrepancy_slope_shift_matches_df1_closed_form() {
    let r = test_mechanism_discrepancy(
        &one_parent("source", &YS),
        &one_parent("target", &YT_SLOPE),
        &opts(false),
    )
    .unwrap();
    // delta = 0.5, v_s + v_t = 0.36: W = delta^2 / 0.36.
    let w = 0.25 / 0.36;
    assert_eq!(r.test.degrees_of_freedom, 1);
    close(r.test.statistic, w, "W");
    // df = 1: P(chi2_1 > w) = erfc(sqrt(w / 2)).
    close(r.test.p_value, erfc((w / 2.0).sqrt()), "p (df 1)");
    assert_eq!(r.coefficients.len(), 1);
    let c = &r.coefficients[0];
    assert_eq!(c.name, "x");
    close(c.difference, 0.5, "delta");
    close(c.standard_error, 0.6, "se");
    close(c.z, 0.5 / 0.6, "z");
    close(c.p_value, erfc((0.5_f64 / 0.6).abs() / 2.0_f64.sqrt()), "p_j");
    // One coefficient: Holm is the identity.
    close(c.p_holm, c.p_value, "holm");
    assert_eq!(r.conclusion, DiscrepancyConclusion::NotRejected);
}

#[test]
fn b3_discrepancy_intercept_and_slope_shift_matches_df2_closed_form() {
    let r = test_mechanism_discrepancy(
        &one_parent("source", &YS),
        &one_parent("target", &YT_BOTH),
        &opts(true),
    )
    .unwrap();
    // d = (0.5, 0.5); W = d' (2V)^-1 d = 0.25 (0.36 + 2 * 0.54 + 1.26) / 0.162 = 25 / 6.
    close(r.test.statistic, 25.0 / 6.0, "W");
    assert_eq!(r.test.degrees_of_freedom, 2);
    // df = 2: P(chi2_2 > w) = exp(-w / 2).
    close(r.test.p_value, (-25.0_f64 / 12.0).exp(), "p (df 2)");
    let intercept = &r.coefficients[0];
    let slope = &r.coefficients[1];
    assert_eq!(intercept.name, "(intercept)");
    close(intercept.standard_error, 1.26_f64.sqrt(), "se intercept");
    close(slope.standard_error, 0.6, "se slope");
    // Holm over two: the smaller raw p (slope) doubles; the other is at least that.
    assert!(slope.p_value < intercept.p_value);
    close(slope.p_holm, (2.0 * slope.p_value).min(1.0), "holm slope");
    close(intercept.p_holm, slope.p_holm.max(intercept.p_value), "holm intercept");
}

#[test]
fn b3_discrepancy_two_parent_statistic_matches_slope_block_algebra() {
    let r = test_mechanism_discrepancy(
        &two_parents("source", &Y6),
        &two_parents("target", &Y6_TARGET),
        &opts(false),
    )
    .unwrap();
    close(r.source.coefficients[0], 0.75, "intercept");
    close(r.source.coefficients[1], 2.75, "a");
    close(r.source.coefficients[2], 0.75, "b");
    // d = (1, 0.5); summed block s [[9,-3],[-3,9]], s = 3.25/18; inverse = [[9,3],[3,9]] / (72 s).
    let w = (9.0 + 3.0 + 2.25) / (72.0 * 3.25 / 18.0);
    close(r.test.statistic, w, "W");
    assert_eq!(r.test.degrees_of_freedom, 2);
    close(r.test.p_value, (-w / 2.0).exp(), "p (df 2)");
    close(r.coefficients[0].difference, 1.0, "a diff");
    close(r.coefficients[1].difference, 0.5, "b diff");
    close(r.coefficients[0].standard_error, (9.0 * 3.25 / 18.0_f64).sqrt(), "se a");
}

#[test]
fn b3_discrepancy_minimal_detectable_difference_uses_normal_quantiles() {
    // The six-decimal constants are the quantiles of the standard normal (checked against erfc).
    let tail = |z: f64| 0.5 * erfc(z / 2.0_f64.sqrt());
    assert!((tail(Z_ALPHA_TWO_SIDED_05) - 0.025).abs() < 1e-7);
    assert!((tail(Z_POWER_80) - 0.2).abs() < 1e-7);
    assert!((normal_upper_quantile(0.025) - 1.959_964).abs() < 1e-6);
    assert!((normal_upper_quantile(0.2) - 0.841_621).abs() < 1e-6);
    assert!(normal_upper_quantile(0.5).abs() < 1e-12);

    let r = test_mechanism_discrepancy(
        &one_parent("source", &YS),
        &one_parent("target", &YT_BOTH),
        &opts(true),
    )
    .unwrap();
    let factor = Z_ALPHA_TWO_SIDED_05 + Z_POWER_80;
    assert!((r.detectability_factor - factor).abs() < 1e-6);
    for c in &r.coefficients {
        assert!(
            (c.minimal_detectable_difference - factor * c.standard_error).abs() < 1e-6,
            "{}: {}",
            c.name,
            c.minimal_detectable_difference
        );
    }
    // Slope: (z_{0.975} + z_{0.80}) * sqrt(0.18 + 0.18).
    assert!((r.coefficients[1].minimal_detectable_difference - factor * 0.6).abs() < 1e-6);
    assert!(r.power_statement.contains("not detectable"));
    assert!(r.power_statement.contains("x:"));
}

#[test]
fn b3_discrepancy_non_rejection_never_certifies_invariance() {
    let r = test_mechanism_discrepancy(
        &one_parent("source", &YS),
        &one_parent("target", &YT_SLOPE),
        &opts(false),
    )
    .unwrap();
    assert_eq!(r.conclusion, DiscrepancyConclusion::NotRejected);
    assert!(!r.non_rejection_certifies_invariance);
    assert!(r.caveat.contains("does not certify"));
    assert!(r.power_caveat.contains("unmeasured"));
    assert_eq!(r.calibration, CalibrationStatus::Unmeasured);
    assert_eq!(r.informs_selection_on, ["V"]);
    assert!(r.alignment.contains("non-rejection leaves a selection node"));
    assert!(r.alignment.contains("cannot be excluded"));
    // The minimal detectable slope difference exceeds the observed shift: the shift was not
    // detectable, which is exactly what the power limit reports.
    assert!(r.coefficients[0].minimal_detectable_difference > r.coefficients[0].difference.abs());

    let big = test_mechanism_discrepancy(
        &one_parent("source", &YS),
        &one_parent("target", &YT_BIG),
        &opts(false),
    )
    .unwrap();
    assert_eq!(big.conclusion, DiscrepancyConclusion::Rejected);
    assert!(big.coefficients[0].rejected);
    assert!(!big.non_rejection_certifies_invariance);
    assert_eq!(big.informs_selection_on, ["V"]);
}

#[test]
fn b3_discrepancy_parent_order_is_irrelevant() {
    let base = test_mechanism_discrepancy(
        &two_parents("source", &Y6),
        &two_parents("target", &Y6_TARGET),
        &opts(true),
    )
    .unwrap();
    let permuted_target = sample("target", &Y6_TARGET, &[("b", "u", &B), ("a", "u", &A)]);
    let permuted_source = sample("source", &Y6, &[("b", "u", &B), ("a", "u", &A)]);
    for (source, target) in [
        (two_parents("source", &Y6), permuted_target.clone()),
        (permuted_source, two_parents("target", &Y6_TARGET)),
    ] {
        let r = test_mechanism_discrepancy(&source, &target, &opts(true)).unwrap();
        assert_eq!(r.coefficient_names, ["(intercept)", "a", "b"]);
        assert_eq!(r.test, base.test);
        assert_eq!(r.coefficients, base.coefficients);
        assert_eq!(r.source.coefficients, base.source.coefficients);
    }
}

#[test]
fn b3_discrepancy_incomparable_measurements_refuse() {
    let incomparable = reason_code!("route_not_supported");
    let detail = "mechanism_discrepancy.incomparable_measurements:";
    let source = one_parent("source", &YS);
    let cases = [
        // Parent unit differs.
        sample("target", &YT_BOTH, &[("x", "m", &X)]),
        // Node unit differs.
        sample_with("target", "protocol-1", "g", &YT_BOTH, &[("x", "cm", &X)]),
        // Measurement protocol id differs.
        sample_with("target", "protocol-2", "mg", &YT_BOTH, &[("x", "cm", &X)]),
        // Parent name differs.
        sample("target", &YT_BOTH, &[("w", "cm", &X)]),
        // Extra parent in the target.
        sample("target", &YT_BOTH, &[("x", "cm", &X), ("w", "cm", &[1.0, 0.0, 2.0, 5.0])]),
        // Blank unit.
        sample("target", &YT_BOTH, &[("x", "", &X)]),
    ];
    for target in cases {
        let (code, message) =
            refusal(test_mechanism_discrepancy(&source, &target, &opts(true)).unwrap_err());
        assert_eq!(code, incomparable);
        assert!(message.starts_with(detail), "{message}");
    }
    let mut other_node = one_parent("target", &YT_BOTH);
    other_node.measurement.node = "W".into();
    let (code, message) =
        refusal(test_mechanism_discrepancy(&source, &other_node, &opts(true)).unwrap_err());
    assert_eq!(code, incomparable);
    assert!(message.starts_with(detail));
}

#[test]
fn b3_discrepancy_dependence_unknown_refuses() {
    let source = one_parent("source", &YS);
    let target = one_parent("target", &YT_BOTH);
    for dependence in [SampleDependence::Unknown, SampleDependence::SharedUnits] {
        let options = DiscrepancyOptions { dependence, ..opts(true) };
        let (code, message) =
            refusal(test_mechanism_discrepancy(&source, &target, &options).unwrap_err());
        assert_eq!(code, reason_code!("route_not_supported"));
        assert!(message.starts_with("mechanism_discrepancy.dependence_unknown:"), "{message}");
    }
    // A shared row identity refuses even when the declaration says independent.
    let ids = |names: [&str; 4]| names.iter().map(|s| (*s).to_owned()).collect::<Vec<_>>();
    let mut shared_source = source.clone();
    shared_source.unit_ids = ids(["u1", "u2", "u3", "u4"]);
    let mut shared_target = target.clone();
    shared_target.unit_ids = ids(["u9", "u3", "u8", "u7"]);
    let (_, message) = refusal(
        test_mechanism_discrepancy(&shared_source, &shared_target, &opts(true)).unwrap_err(),
    );
    assert!(message.starts_with("mechanism_discrepancy.dependence_unknown:"), "{message}");
    // Disjoint ids are fine.
    shared_target.unit_ids = ids(["t1", "t2", "t3", "t4"]);
    assert!(test_mechanism_discrepancy(&shared_source, &shared_target, &opts(true)).is_ok());
}

#[test]
fn b3_discrepancy_degenerate_inputs_refuse() {
    let invalid = reason_code!("invalid_argument");
    let detail_of = |error: EstimationError| {
        let (code, message) = refusal(error);
        assert_eq!(code, invalid);
        message.split(':').next().unwrap_or_default().to_owned()
    };
    let source = one_parent("source", &YS);

    // Tiny sample: n = 3, p = 2 leaves one residual degree of freedom.
    let tiny = sample("target", &[1.0, 3.0, 2.0], &[("x", "cm", &[0.0, 1.0, 2.0])]);
    assert_eq!(
        detail_of(test_mechanism_discrepancy(&source, &tiny, &opts(true)).unwrap_err()),
        "mechanism_discrepancy.sample_too_small"
    );
    // Rank deficient: a constant parent is collinear with the intercept.
    let constant = sample("target", &YT_BOTH, &[("x", "cm", &[2.0; 4])]);
    assert_eq!(
        detail_of(test_mechanism_discrepancy(&source, &constant, &opts(true)).unwrap_err()),
        "mechanism_discrepancy.rank_deficient_design"
    );
    // Non-finite value.
    let nan = sample("target", &[1.0, f64::NAN, 3.0, 4.0], &[("x", "cm", &X)]);
    assert_eq!(
        detail_of(test_mechanism_discrepancy(&source, &nan, &opts(true)).unwrap_err()),
        "mechanism_discrepancy.non_finite_value"
    );
    // Both populations fit exactly: the summed covariance is zero.
    let exact_s = one_parent("source", &[0.0, 1.0, 2.0, 3.0]);
    let exact_t = one_parent("target", &[1.0, 3.0, 5.0, 7.0]);
    assert_eq!(
        detail_of(test_mechanism_discrepancy(&exact_s, &exact_t, &opts(true)).unwrap_err()),
        "mechanism_discrepancy.degenerate_covariance"
    );
    // Row-count mismatch.
    let ragged = sample("target", &YT_BOTH, &[("x", "cm", &[0.0, 1.0, 2.0])]);
    assert_eq!(
        detail_of(test_mechanism_discrepancy(&source, &ragged, &opts(true)).unwrap_err()),
        "mechanism_discrepancy.row_count_mismatch"
    );
    // Levels.
    let bad_alpha = DiscrepancyOptions { alpha: 1.0, ..opts(true) };
    assert_eq!(
        detail_of(test_mechanism_discrepancy(&source, &source, &bad_alpha).unwrap_err()),
        "mechanism_discrepancy.invalid_alpha"
    );
    let bad_power = DiscrepancyOptions { power: 0.3, ..opts(true) };
    assert_eq!(
        detail_of(test_mechanism_discrepancy(&source, &source, &bad_power).unwrap_err()),
        "mechanism_discrepancy.invalid_power"
    );
    // A parent-free node with the intercept excluded compares nothing.
    let bare = |label: &str| sample(label, &YS, &[]);
    assert_eq!(
        detail_of(test_mechanism_discrepancy(&bare("s"), &bare("t"), &opts(false)).unwrap_err()),
        "mechanism_discrepancy.no_compared_coefficients"
    );
}

#[test]
#[allow(clippy::float_cmp, reason = "the summary entries are integer-valued sums")]
fn b3_discrepancy_summaries_replay_the_raw_data_result() {
    let source = one_parent("source", &YS);
    let target = one_parent("target", &YT_BOTH);
    let raw = test_mechanism_discrepancy(&source, &target, &opts(true)).unwrap();
    let s = summarize_population(&source).unwrap();
    let t = summarize_population(&target).unwrap();
    // X'X = [[4,6],[6,14]], X'y = [10,19], y'y = 30.
    assert_eq!(s.n, 4);
    assert_eq!(s.xtx, [4.0, 6.0, 6.0, 14.0]);
    assert_eq!(s.xty, [10.0, 19.0]);
    assert_eq!(s.yty, 30.0);
    let replay = test_mechanism_discrepancy_from_summaries(&s, &t, &opts(true)).unwrap();
    assert_eq!(replay, raw);
    // An inconsistent summary (intercept sum != n) refuses.
    let mut broken: MechanismSummary = s;
    broken.xtx[0] = 5.0;
    let (code, message) =
        refusal(test_mechanism_discrepancy_from_summaries(&broken, &t, &opts(true)).unwrap_err());
    assert_eq!(code, reason_code!("invalid_argument"));
    assert!(message.starts_with("mechanism_discrepancy.inconsistent_summary:"), "{message}");
}
