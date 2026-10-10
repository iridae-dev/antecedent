//! B4 joint vector-treatment coefficients: hand-derived OLS coefficients, the full covariance
//! matrix `sigma^2 (X'X)^-1` (off-diagonals included), contrast variances `w' V w`, HC sandwiches
//! written out row by row, a Frisch--Waugh closed-form check with an adjustment column, the
//! joint Wald statistic, and the typed refusals. Calibration (coverage, Type I error, power) is
//! deliberately NOT measured here.
//!
//! Frozen dataset (n = 6, p = 3, columns `1, t1, t2`):
//! `t1 = [1,0,1,0,0,1]`, `t2 = [1,1,1,0,0,0]`, `y = [3,2,5,1,0,4]`.
//! `X'X = [[6,3,3],[3,3,2],[3,2,3]]`, `det = 12`, `(X'X)^-1 = [[5,-3,-3],[-3,9,-3],[-3,-3,9]] / 12`,
//! `X'y = [15,12,10]`, so `b = [0.75, 2.75, 0.75]`; residuals `[-1.25,0.5,0.75,0.25,-0.75,0.5]`,
//! `RSS = 3.25`, `sigma^2 = 3.25 / 3`; the treatment block of the model-based covariance is
//! `(3.25 / 36) [[9,-3],[-3,9]]`.

use antecedent_estimate::EstimationError;
use antecedent_estimate::effect_constancy::CalibrationStatus;
use antecedent_estimate::vector_treatment::{
    Contrast, NamedColumn, TreatmentColumn, VECTOR_TREATMENT_NULL, VectorCovariance,
    VectorTreatmentInput, VectorTreatmentOptions, fit_vector_treatment,
};
use antecedent_kernels::erfc;

const X1: [f64; 6] = [1.0, 0.0, 1.0, 0.0, 0.0, 1.0];
const X2: [f64; 6] = [1.0, 1.0, 1.0, 0.0, 0.0, 0.0];
const Y: [f64; 6] = [3.0, 2.0, 5.0, 1.0, 0.0, 4.0];
const S: f64 = 3.25 / 36.0;

fn close(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() <= 1e-9, "{what}: {actual} vs {expected}");
}

fn refusal(error: EstimationError) -> (&'static str, String) {
    let EstimationError::Refused { code, message } = error else {
        panic!("expected a coded refusal, got {error:?}");
    };
    (code, message)
}

fn treatment(name: &str, values: &[f64]) -> TreatmentColumn {
    TreatmentColumn {
        name: name.into(),
        values: values.to_vec(),
        adjustment_set: vec![],
        row_snapshot: "snap".into(),
    }
}

fn base_input() -> VectorTreatmentInput {
    VectorTreatmentInput {
        outcome: Y.to_vec(),
        row_snapshot: "snap".into(),
        adjustment: vec![],
        treatments: vec![treatment("t1", &X1), treatment("t2", &X2)],
    }
}

fn contrast(name: &str, weights: &[(&str, f64)]) -> Contrast {
    Contrast {
        name: name.into(),
        weights: weights.iter().map(|(n, w)| ((*n).to_string(), *w)).collect(),
    }
}

fn options(covariance: VectorCovariance, contrasts: Vec<Contrast>) -> VectorTreatmentOptions {
    VectorTreatmentOptions { covariance, contrasts }
}

#[test]
fn b4_vector_coefficients_and_full_covariance_match_hand_algebra() {
    let fit = fit_vector_treatment(&base_input(), &options(VectorCovariance::ModelBased, vec![]))
        .unwrap();
    assert_eq!(fit.names, ["t1", "t2"]);
    assert_eq!(fit.dimension(), 2);
    close(fit.coefficients[0].estimate, 2.75, "b1");
    close(fit.coefficients[1].estimate, 0.75, "b2");
    assert_eq!(fit.coefficients[0].name, "t1");
    assert_eq!(fit.n_rows, 6);
    assert_eq!(fit.residual_df, 3);
    close(fit.residual_variance, 3.25 / 3.0, "sigma^2");
    close(fit.covariance[0], 9.0 * S, "V11");
    close(fit.covariance[1], -3.0 * S, "V12");
    close(fit.covariance[2], -3.0 * S, "V21");
    close(fit.covariance[3], 9.0 * S, "V22");
    close(fit.covariance_entry(0, 1).unwrap(), -3.0 * S, "entry(0,1)");
    assert!(fit.covariance_entry(2, 0).is_none());
    close(fit.coefficients[0].standard_error, (9.0 * S).sqrt(), "se1");
    close(fit.coefficients[0].z, 2.75 / (9.0 * S).sqrt(), "z1");
    close(fit.coefficients[0].p_value, erfc(fit.coefficients[0].z.abs() / 2.0_f64.sqrt()), "p1");
    assert_eq!(fit.null, VECTOR_TREATMENT_NULL);
    assert_eq!(fit.calibration, CalibrationStatus::Unmeasured);
    assert_eq!(fit.covariance_kind, VectorCovariance::ModelBased);
}

#[test]
fn b4_vector_contrast_se_uses_the_off_diagonal_not_the_independent_sum() {
    let contrasts = vec![
        contrast("diff", &[("t1", 1.0), ("t2", -1.0)]),
        contrast("sum", &[("t1", 1.0), ("t2", 1.0)]),
    ];
    let fit =
        fit_vector_treatment(&base_input(), &options(VectorCovariance::ModelBased, contrasts))
            .unwrap();
    let diff = &fit.contrasts[0];
    let sum = &fit.contrasts[1];
    assert_eq!(diff.name, "diff");
    close(diff.estimate, 2.0, "b1 - b2");
    // Var(b1 - b2) = V11 + V22 - 2 V12 = 24 s; the naive independent variance is 18 s.
    close(diff.standard_error * diff.standard_error, 24.0 * S, "diff variance");
    close(
        diff.naive_independent_standard_error * diff.naive_independent_standard_error,
        18.0 * S,
        "naive diff variance",
    );
    // The off-diagonal correction is exactly -2 V12 = +6 s here.
    close(
        diff.standard_error.powi(2) - diff.naive_independent_standard_error.powi(2),
        6.0 * S,
        "correction = -2 V12",
    );
    // Var(b1 + b2) = 18 s - 6 s = 12 s: the correction has the opposite sign.
    close(sum.estimate, 3.5, "b1 + b2");
    close(sum.standard_error.powi(2), 12.0 * S, "sum variance");
    close(
        sum.standard_error.powi(2) - sum.naive_independent_standard_error.powi(2),
        -6.0 * S,
        "sum correction",
    );
    close(diff.z, 2.0 / (24.0 * S).sqrt(), "diff z");
    close(diff.p_value, erfc(diff.z.abs() / 2.0_f64.sqrt()), "diff p");
}

#[test]
fn b4_vector_contrast_family_gets_holm_adjustment() {
    let contrasts = vec![
        contrast("diff", &[("t1", 1.0), ("t2", -1.0)]),
        contrast("sum", &[("t1", 1.0), ("t2", 1.0)]),
    ];
    let fit =
        fit_vector_treatment(&base_input(), &options(VectorCovariance::ModelBased, contrasts))
            .unwrap();
    let (diff, sum) = (&fit.contrasts[0], &fit.contrasts[1]);
    // |z_sum| = 3.5 / sqrt(12 s) > |z_diff| = 2 / sqrt(24 s), so p_sum < p_diff.
    assert!(sum.p_value < diff.p_value);
    let holm_sum = (2.0 * sum.p_value).min(1.0);
    let holm_diff = diff.p_value.min(1.0).max(holm_sum);
    close(sum.p_holm, holm_sum, "holm sum");
    close(diff.p_holm, holm_diff, "holm diff");
}

#[test]
fn b4_vector_joint_wald_matches_closed_form_quadratic() {
    let fit = fit_vector_treatment(&base_input(), &options(VectorCovariance::ModelBased, vec![]))
        .unwrap();
    // V = s [[9,-3],[-3,9]], V^-1 = [[9,3],[3,9]] / (72 s); b = (2.75, 0.75).
    let b = (2.75_f64, 0.75_f64);
    let statistic = (9.0 * b.0 * b.0 + 6.0 * b.0 * b.1 + 9.0 * b.1 * b.1) / (72.0 * S);
    close(statistic, 85.5 / 6.5, "closed-form statistic");
    close(fit.joint_wald.statistic, statistic, "Wald statistic");
    assert_eq!(fit.joint_wald.degrees_of_freedom, 2);
    // chi-square df 2 survival function is exp(-x / 2).
    close(fit.joint_wald.p_value, (-statistic / 2.0).exp(), "Wald p");
}

#[test]
fn b4_vector_wald_test_over_declared_contrasts() {
    let fit = fit_vector_treatment(&base_input(), &options(VectorCovariance::ModelBased, vec![]))
        .unwrap();
    let diff = contrast("diff", &[("t1", 1.0), ("t2", -1.0)]);
    let one = fit.wald_test(std::slice::from_ref(&diff)).unwrap();
    // One restriction: (b1 - b2)^2 / (24 s); chi-square df 1 survival is erfc(sqrt(x / 2)).
    close(one.statistic, 4.0 / (24.0 * S), "contrast Wald");
    assert_eq!(one.degrees_of_freedom, 1);
    close(one.p_value, erfc((one.statistic / 2.0).sqrt()), "contrast Wald p");
    // Two proportional contrasts make the restriction covariance singular.
    let negated = contrast("neg", &[("t1", -1.0), ("t2", 1.0)]);
    let (code, message) = refusal(fit.wald_test(&[diff, negated]).unwrap_err());
    assert_eq!(code, "invalid_argument");
    assert!(message.contains("vector_treatment.degenerate_covariance"), "{message}");
    let (_, empty) = refusal(fit.wald_test(&[]).unwrap_err());
    assert!(empty.contains("vector_treatment.empty_contrast"), "{empty}");
}

/// Treatment block of `Ginv M Ginv` with `M = sum_i w_i e_i^2 x_i x_i'`, all written out.
fn sandwich_block(weights: &[f64; 6]) -> [f64; 4] {
    let ginv = [[5.0, -3.0, -3.0], [-3.0, 9.0, -3.0], [-3.0, -3.0, 9.0]];
    let e = [-1.25, 0.5, 0.75, 0.25, -0.75, 0.5];
    let mut meat = [[0.0_f64; 3]; 3];
    for i in 0..6 {
        let x = [1.0, X1[i], X2[i]];
        for a in 0..3 {
            for b in 0..3 {
                meat[a][b] += weights[i] * e[i] * e[i] * x[a] * x[b];
            }
        }
    }
    let mut out = [0.0_f64; 4];
    for (slot, (a, b)) in [(1, 1), (1, 2), (2, 1), (2, 2)].into_iter().enumerate() {
        let mut sum = 0.0;
        for (l, meat_row) in meat.iter().enumerate() {
            for m in 0..3 {
                sum += ginv[a][l] / 12.0 * meat_row[m] * ginv[m][b] / 12.0;
            }
        }
        out[slot] = sum;
    }
    out
}

#[test]
fn b4_vector_robust_sandwiches_match_row_by_row_algebra() {
    let leverage = [5.0 / 12.0, 8.0 / 12.0, 5.0 / 12.0, 5.0 / 12.0, 5.0 / 12.0, 8.0 / 12.0];
    let hc0 = [1.0; 6];
    let hc1 = [2.0; 6]; // n / (n - p) = 6 / 3
    let mut hc2 = [0.0; 6];
    let mut hc3 = [0.0; 6];
    for i in 0..6 {
        hc2[i] = 1.0 / (1.0 - leverage[i]);
        hc3[i] = 1.0 / ((1.0 - leverage[i]) * (1.0 - leverage[i]));
    }
    let cases = [
        (VectorCovariance::Hc0, hc0),
        (VectorCovariance::Hc1, hc1),
        (VectorCovariance::Hc2, hc2),
        (VectorCovariance::Hc3, hc3),
    ];
    for (kind, weights) in cases {
        let fit = fit_vector_treatment(&base_input(), &options(kind, vec![])).unwrap();
        let expected = sandwich_block(&weights);
        for (slot, value) in expected.iter().enumerate() {
            close(fit.covariance[slot], *value, kind.as_str());
        }
        assert_eq!(fit.covariance_kind, kind);
        // The point estimates do not depend on the covariance estimator.
        close(fit.coefficients[0].estimate, 2.75, "b1");
        close(fit.coefficients[1].estimate, 0.75, "b2");
    }
}

/// Residuals of `v` on `[1, z]` by the closed-form simple regression.
fn residualize(v: &[f64], z: &[f64]) -> Vec<f64> {
    let n = v.len() as f64;
    let (vbar, zbar) = (v.iter().sum::<f64>() / n, z.iter().sum::<f64>() / n);
    let szz: f64 = z.iter().map(|a| (a - zbar) * (a - zbar)).sum();
    let szv: f64 = z.iter().zip(v).map(|(a, b)| (a - zbar) * (b - vbar)).sum();
    let slope = szv / szz;
    z.iter().zip(v).map(|(a, b)| b - vbar - slope * (a - zbar)).collect()
}

#[test]
fn b4_vector_adjustment_matches_frisch_waugh_closed_form() {
    let z = [0.0, 1.0, 2.0, 0.0, 1.0, 2.0];
    let mut input = base_input();
    input.adjustment = vec![NamedColumn { name: "z".into(), values: z.to_vec() }];
    for t in &mut input.treatments {
        t.adjustment_set = vec!["z".into()];
    }
    let fit = fit_vector_treatment(&input, &options(VectorCovariance::ModelBased, vec![])).unwrap();
    let (r1, r2, ry) = (residualize(&X1, &z), residualize(&X2, &z), residualize(&Y, &z));
    let a = r1.iter().map(|v| v * v).sum::<f64>();
    let d = r2.iter().map(|v| v * v).sum::<f64>();
    let b = r1.iter().zip(&r2).map(|(u, v)| u * v).sum::<f64>();
    let det = a * d - b * b;
    let c1 = r1.iter().zip(&ry).map(|(u, v)| u * v).sum::<f64>();
    let c2 = r2.iter().zip(&ry).map(|(u, v)| u * v).sum::<f64>();
    let (b1, b2) = ((d * c1 - b * c2) / det, (a * c2 - b * c1) / det);
    let rss: f64 = (0..6).map(|i| (ry[i] - b1 * r1[i] - b2 * r2[i]).powi(2)).sum();
    let sigma2 = rss / 2.0; // n - p = 6 - 4
    close(fit.coefficients[0].estimate, b1, "b1");
    close(fit.coefficients[1].estimate, b2, "b2");
    close(fit.covariance[0], sigma2 * d / det, "V11");
    close(fit.covariance[1], -sigma2 * b / det, "V12");
    close(fit.covariance[3], sigma2 * a / det, "V22");
    assert_eq!(fit.residual_df, 2);
    close(fit.residual_variance, sigma2, "sigma^2");
}

#[test]
fn b4_vector_treatment_order_permutes_the_covariance_consistently() {
    let mut swapped = base_input();
    swapped.treatments.reverse();
    let contrasts = vec![contrast("diff", &[("t1", 1.0), ("t2", -1.0)])];
    let a = fit_vector_treatment(
        &base_input(),
        &options(VectorCovariance::ModelBased, contrasts.clone()),
    )
    .unwrap();
    let b =
        fit_vector_treatment(&swapped, &options(VectorCovariance::ModelBased, contrasts)).unwrap();
    assert_eq!(b.names, ["t2", "t1"]);
    close(b.covariance[0], a.covariance[3], "V22 moves to the front");
    close(b.covariance[1], a.covariance[1], "off-diagonal is symmetric under swap");
    close(b.covariance[3], a.covariance[0], "V11 moves to the back");
    close(b.contrasts[0].standard_error, a.contrasts[0].standard_error, "contrast se");
    close(b.joint_wald.statistic, a.joint_wald.statistic, "Wald statistic");
}

#[test]
fn b4_vector_refuses_different_adjustment_sets() {
    let mut input = base_input();
    input.adjustment =
        vec![NamedColumn { name: "z".into(), values: vec![0.0, 1.0, 2.0, 0.0, 1.0, 3.0] }];
    input.treatments[0].adjustment_set = vec!["z".into()];
    // t2 was declared against an empty set while the shared block holds `z`.
    let (code, message) = refusal(
        fit_vector_treatment(&input, &options(VectorCovariance::ModelBased, vec![])).unwrap_err(),
    );
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("vector_treatment.adjustment_set_mismatch"), "{message}");
    assert!(message.contains("`t2`"), "{message}");
    // A treatment that names an adjustment column the shared block does not hold also refuses.
    let mut other = base_input();
    other.treatments[1].adjustment_set = vec!["w".into()];
    let (_, message) = refusal(
        fit_vector_treatment(&other, &options(VectorCovariance::ModelBased, vec![])).unwrap_err(),
    );
    assert!(message.contains("vector_treatment.adjustment_set_mismatch"), "{message}");
}

#[test]
fn b4_vector_refuses_different_row_snapshots_and_row_counts() {
    let mut input = base_input();
    input.treatments[1].row_snapshot = "other".into();
    let (code, message) = refusal(
        fit_vector_treatment(&input, &options(VectorCovariance::ModelBased, vec![])).unwrap_err(),
    );
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("vector_treatment.row_snapshot_mismatch"), "{message}");
    let mut short = base_input();
    short.treatments[1].values.pop();
    let (code, message) = refusal(
        fit_vector_treatment(&short, &options(VectorCovariance::ModelBased, vec![])).unwrap_err(),
    );
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("vector_treatment.row_count_mismatch"), "{message}");
}

#[test]
fn b4_vector_refuses_constant_collinear_and_rank_deficient_designs() {
    let run = |input: &VectorTreatmentInput| {
        refusal(
            fit_vector_treatment(input, &options(VectorCovariance::ModelBased, vec![]))
                .unwrap_err(),
        )
    };
    let mut constant = base_input();
    constant.treatments[1] = treatment("t2", &[1.0; 6]);
    let (code, message) = run(&constant);
    assert_eq!(code, "design_rank_deficient");
    assert!(message.contains("vector_treatment.treatment_without_variation"), "{message}");
    assert!(message.contains("`t2`"), "{message}");

    let mut collinear = base_input();
    collinear.treatments[1] = treatment("t2", &[2.0, 0.0, 2.0, 0.0, 0.0, 2.0]);
    let (code, message) = run(&collinear);
    assert_eq!(code, "design_rank_deficient");
    assert!(message.contains("vector_treatment.collinear_treatments"), "{message}");
    assert!(message.contains("`t2`"), "{message}");

    let mut duplicated_adjustment = base_input();
    duplicated_adjustment.adjustment = vec![NamedColumn { name: "w".into(), values: vec![1.0; 6] }];
    for t in &mut duplicated_adjustment.treatments {
        t.adjustment_set = vec!["w".into()];
    }
    let (code, message) = run(&duplicated_adjustment);
    assert_eq!(code, "design_rank_deficient");
    assert!(message.contains("vector_treatment.rank_deficient_adjustment"), "{message}");
}

#[test]
fn b4_vector_refuses_malformed_requests() {
    let run = |input: &VectorTreatmentInput, contrasts: Vec<Contrast>| {
        refusal(
            fit_vector_treatment(input, &options(VectorCovariance::ModelBased, contrasts))
                .unwrap_err(),
        )
    };
    let mut single = base_input();
    single.treatments.pop();
    let (code, message) = run(&single, vec![]);
    assert_eq!(code, "invalid_argument");
    assert!(message.contains("vector_treatment.too_few_treatments"), "{message}");

    let mut duplicate = base_input();
    duplicate.treatments[1].name = "t1".into();
    let (_, message) = run(&duplicate, vec![]);
    assert!(message.contains("vector_treatment.duplicate_name"), "{message}");

    let mut non_finite = base_input();
    non_finite.outcome[0] = f64::NAN;
    let (_, message) = run(&non_finite, vec![]);
    assert!(message.contains("vector_treatment.non_finite_value"), "{message}");

    let unknown = vec![contrast("bad", &[("t1", 1.0), ("nope", -1.0)])];
    let (_, message) = run(&base_input(), unknown);
    assert!(message.contains("vector_treatment.unknown_contrast_coefficient"), "{message}");

    // Three rows cannot support an intercept plus two treatments and a residual degree of freedom.
    let tiny = VectorTreatmentInput {
        outcome: vec![1.0, 2.0, 4.0],
        row_snapshot: "snap".into(),
        adjustment: vec![],
        treatments: vec![treatment("t1", &[1.0, 0.0, 1.0]), treatment("t2", &[0.0, 1.0, 1.0])],
    };
    let (_, message) = run(&tiny, vec![]);
    assert!(message.contains("vector_treatment.too_few_rows"), "{message}");
}
