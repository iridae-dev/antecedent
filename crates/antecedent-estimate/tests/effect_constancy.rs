//! F18 `EffectConstancy`: hand-computed heterogeneity statistics and p-values against closed-form
//! chi-square survival functions (df 1: `erfc(sqrt(x / 2))`, df 2: `exp(-x / 2)`), Holm values
//! checked by hand, refusals, and permutation invariance. Calibration (null Type I error and
//! power) is deliberately NOT measured here.

use antecedent_estimate::EstimationError;
use antecedent_estimate::effect_constancy::{
    CalibrationStatus, ConstancyConclusion, ContrastFamily, EFFECT_CONSTANCY_NULL,
    EffectEstimandIdentity, HeterogeneityStatistic, PartitionDependence, PartitionEstimate,
    PartitionSupport, holm_adjust, test_effect_constancy,
};

fn estimand() -> EffectEstimandIdentity {
    EffectEstimandIdentity {
        estimand: "ate_difference".into(),
        units: "outcome_units".into(),
        regime: "treat_vs_control".into(),
        population: "all_observed_h2".into(),
    }
}

fn part(label: &str, effect: f64, se: f64) -> PartitionEstimate {
    PartitionEstimate {
        label: label.into(),
        coordinate: format!("period:{label}"),
        support: PartitionSupport::Supported,
        estimand: estimand(),
        effect,
        standard_error: se,
    }
}

fn close(actual: f64, expected: f64, what: &str) {
    assert!((actual - expected).abs() <= 1e-9, "{what}: {actual} vs {expected}");
}

fn refusal(error: EstimationError) -> (&'static str, String) {
    let EstimationError::Refused { code, message } = error else {
        panic!("expected a coded refusal, got {error:?}");
    };
    (code, message)
}

fn run_independent(
    parts: &[PartitionEstimate],
) -> Result<antecedent_estimate::effect_constancy::EffectConstancyResult, EstimationError> {
    test_effect_constancy(parts, &PartitionDependence::Independent, &ContrastFamily::AllPairs, 0.05)
}

#[test]
fn f18_constancy_oracle_equal_effects_have_zero_statistic() {
    // Frozen acceptance: effects 1 and 1, variance 1/4 each, zero covariance.
    let result = run_independent(&[part("p1", 1.0, 0.5), part("p2", 1.0, 0.5)]).unwrap();
    assert_eq!(result.null, EFFECT_CONSTANCY_NULL);
    assert_eq!(result.test.statistic_kind, HeterogeneityStatistic::CochranQ);
    close(result.test.statistic, 0.0, "Q");
    assert_eq!(result.test.degrees_of_freedom, 1);
    close(result.test.p_value, 1.0, "p");
    assert_eq!(result.conclusion, ConstancyConclusion::NotRejected);
    assert_eq!(result.calibration, CalibrationStatus::Unmeasured);
    assert!(result.non_rejection_caveat.contains("does not prove"));
    assert!(!result.power_caveat.is_empty());
    close(result.pooled_effect.unwrap(), 1.0, "pooled");
}

#[test]
fn f18_constancy_oracle_one_versus_two_has_statistic_two() {
    // Q = (2 - 1)^2 / (1/4 + 1/4) = 2; df 1: sf = erfc(sqrt(1)) = erfc(1).
    let result = run_independent(&[part("p1", 1.0, 0.5), part("p2", 2.0, 0.5)]).unwrap();
    close(result.test.statistic, 2.0, "Q");
    close(result.test.p_value, 0.157_299_207_050_285_13, "p = erfc(1)");
    assert_eq!(result.conclusion, ConstancyConclusion::NotRejected);
    close(result.pooled_effect.unwrap(), 1.5, "pooled");
    assert_eq!(result.contrasts.len(), 1);
    // Single contrast: z = (1 - 2) / sqrt(1/2) = -sqrt(2), p = erfc(1), Holm m = 1 leaves it.
    close(result.contrasts[0].p_value, 0.157_299_207_050_285_13, "contrast p");
    close(result.contrasts[0].p_holm, 0.157_299_207_050_285_13, "contrast holm");
    close(result.contrasts[0].difference, -1.0, "difference");
}

#[test]
fn f18_constancy_oracle_three_partitions_df_two_and_holm() {
    // Effects 0, 1, 2 with unit variance: pooled 1, Q = 1 + 0 + 1 = 2, df 2, p = exp(-1).
    let parts = [part("a", 0.0, 1.0), part("b", 1.0, 1.0), part("c", 2.0, 1.0)];
    let result = run_independent(&parts).unwrap();
    close(result.test.statistic, 2.0, "Q");
    assert_eq!(result.test.degrees_of_freedom, 2);
    close(result.test.p_value, (-1.0_f64).exp(), "p = exp(-1)");
    // Pairs (a,b), (a,c), (b,c): variance 2 each.
    //   a-b: z = -1/sqrt(2), p = erfc(1/2) = 0.4795001221869535
    //   a-c: z = -sqrt(2),   p = erfc(1)   = 0.15729920705028513
    //   b-c: z = -1/sqrt(2), p = erfc(1/2)
    // Holm (m = 3): a-c 3 * 0.1573 = 0.4718976211508554; next 2 * 0.4795 = 0.959000244373907;
    // last 1 * 0.4795 lifted to the running max 0.959000244373907.
    let c = &result.contrasts;
    assert_eq!((c[0].left.as_str(), c[0].right.as_str()), ("a", "b"));
    assert_eq!((c[1].left.as_str(), c[1].right.as_str()), ("a", "c"));
    assert_eq!((c[2].left.as_str(), c[2].right.as_str()), ("b", "c"));
    close(c[0].p_value, 0.479_500_122_186_953_5, "ab p");
    close(c[1].p_value, 0.157_299_207_050_285_13, "ac p");
    close(c[2].p_value, 0.479_500_122_186_953_5, "bc p");
    close(c[0].p_holm, 0.959_000_244_373_907, "ab holm");
    close(c[1].p_holm, 0.471_897_621_150_855_4, "ac holm");
    close(c[2].p_holm, 0.959_000_244_373_907, "bc holm");
    assert!(c.iter().all(|r| !r.rejected));
    assert_eq!(result.family, ContrastFamily::AllPairs);
}

#[test]
fn f18_constancy_oracle_holm_adjust_by_hand() {
    // p = (0.01, 0.04, 0.03, 0.5): sorted 0.01, 0.03, 0.04, 0.5 -> 0.04, 0.09, 0.08, 0.5;
    // running max 0.04, 0.09, 0.09, 0.5. Original order: 0.04, 0.09, 0.09, 0.5.
    let adjusted = holm_adjust(&[0.01, 0.04, 0.03, 0.5]);
    for (a, e) in adjusted.iter().zip([0.04, 0.09, 0.09, 0.5]) {
        close(*a, e, "holm");
    }
    // Capped at one.
    close(holm_adjust(&[0.6, 0.7])[1], 1.0, "capped");
}

#[test]
fn f18_constancy_oracle_varying_effects_rejected() {
    // Q = 25 / (0.25 + 0.25) = 50; sf = erfc(5) ~ 1.5e-12.
    let result = run_independent(&[part("a", 0.0, 0.5), part("b", 5.0, 0.5)]).unwrap();
    close(result.test.statistic, 50.0, "Q");
    assert!(result.test.p_value < 1e-9);
    assert_eq!(result.conclusion, ConstancyConclusion::Rejected);
    assert!(result.contrasts[0].rejected);
}

#[test]
fn f18_constancy_oracle_dependent_covariance_differs_by_analytic_amount() {
    // Estimates 1 and 2, variance 1/4 each, covariance 1/8.
    // Var(e2 - e1) = 1/4 + 1/4 - 2/8 = 1/4, so Wald = 1 / (1/4) = 4 versus Q = 2 under
    // independence; df 1: sf = erfc(sqrt(2)) = 0.04550026389635842.
    let parts = [part("a", 1.0, 0.5), part("b", 2.0, 0.5)];
    let covariance = PartitionDependence::Covariance(vec![0.25, 0.125, 0.125, 0.25]);
    let result =
        test_effect_constancy(&parts, &covariance, &ContrastFamily::AllPairs, 0.05).unwrap();
    assert_eq!(result.test.statistic_kind, HeterogeneityStatistic::WaldChiSquare);
    close(result.test.statistic, 4.0, "Wald");
    close(result.test.p_value, 0.045_500_263_896_358_42, "p = erfc(sqrt 2)");
    assert_eq!(result.conclusion, ConstancyConclusion::Rejected);
    assert!(result.pooled_effect.is_none());
    close(result.contrasts[0].standard_error, 0.5, "contrast se");
    let independent = run_independent(&parts).unwrap();
    close(independent.test.statistic, 2.0, "Q");
    close(result.test.statistic / independent.test.statistic, 2.0, "analytic ratio");
}

#[test]
fn f18_constancy_oracle_diagonal_covariance_equals_cochran_q() {
    let parts = [part("a", 0.0, 1.0), part("b", 1.0, 1.0), part("c", 2.0, 1.0)];
    let diagonal =
        PartitionDependence::Covariance(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]);
    let result = test_effect_constancy(&parts, &diagonal, &ContrastFamily::AllPairs, 0.05).unwrap();
    close(result.test.statistic, 2.0, "Wald = Q");
    close(result.test.p_value, (-1.0_f64).exp(), "p = exp(-1)");
}

#[test]
fn f18_constancy_oracle_reference_family_is_reported() {
    let parts = [part("a", 0.0, 1.0), part("b", 1.0, 1.0), part("c", 2.0, 1.0)];
    let family = ContrastFamily::AgainstReference("a".into());
    let result =
        test_effect_constancy(&parts, &PartitionDependence::Independent, &family, 0.05).unwrap();
    assert_eq!(result.family, family);
    assert_eq!(result.contrasts.len(), 2);
    assert!(result.contrasts.iter().all(|c| c.right == "a"));
    // b-a: z = 1/sqrt 2, p = erfc(1/2); c-a: z = sqrt 2, p = erfc(1). Holm m = 2:
    // c-a 2 * 0.1573 = 0.31459841410057026; b-a max(0.4795, 0.3146) = 0.4795.
    close(result.contrasts[0].p_holm, 0.479_500_122_186_953_5, "b-a holm");
    close(result.contrasts[1].p_holm, 0.314_598_414_100_570_26, "c-a holm");
}

#[test]
fn f18_incompatible_partition_estimand_units_and_regime_refuse() {
    for mutate in [
        |e: &mut EffectEstimandIdentity| e.estimand = "risk_ratio".into(),
        |e: &mut EffectEstimandIdentity| e.units = "log_scale".into(),
        |e: &mut EffectEstimandIdentity| e.regime = "other_regime".into(),
        |e: &mut EffectEstimandIdentity| e.population = "other_population".into(),
    ] {
        let mut changed = part("b", 1.0, 0.5);
        mutate(&mut changed.estimand);
        let error = run_independent(&[part("a", 1.0, 0.5), changed]).unwrap_err();
        let (code, message) = refusal(error);
        assert_eq!(code, "route_not_supported");
        assert!(message.contains("effect_constancy.incompatible_partitions"), "{message}");
    }
}

#[test]
fn f18_incompatible_partition_unsupported_coordinate_and_duplicate_label_refuse() {
    let mut unsupported = part("b", 1.0, 0.5);
    unsupported.support = PartitionSupport::Unsupported;
    let (code, message) =
        refusal(run_independent(&[part("a", 1.0, 0.5), unsupported]).unwrap_err());
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("effect_constancy.unsupported_partition"));
    let (code, message) =
        refusal(run_independent(&[part("a", 1.0, 0.5), part("a", 2.0, 0.5)]).unwrap_err());
    assert_eq!(code, "route_not_supported");
    assert!(message.contains("effect_constancy.incompatible_partitions"));
    let mut partial = part("b", 1.0, 0.5);
    partial.support = PartitionSupport::Partial;
    let result = run_independent(&[part("a", 1.0, 0.5), partial]).unwrap();
    assert_eq!(result.partitions[1].support, PartitionSupport::Partial);
}

#[test]
fn f18_incompatible_partition_too_few_and_non_finite_refuse() {
    let (code, message) = refusal(run_independent(&[part("a", 1.0, 0.5)]).unwrap_err());
    assert_eq!(code, "invalid_argument");
    assert!(message.contains("effect_constancy.too_few_partitions"));
    assert!(run_independent(&[]).is_err());
    for bad in [f64::NAN, f64::INFINITY] {
        let (code, message) =
            refusal(run_independent(&[part("a", 1.0, 0.5), part("b", bad, 0.5)]).unwrap_err());
        assert_eq!(code, "invalid_argument");
        assert!(message.contains("effect_constancy.non_finite_estimate"));
    }
    let (_, message) =
        refusal(run_independent(&[part("a", 1.0, 0.5), part("b", 1.0, f64::NAN)]).unwrap_err());
    assert!(message.contains("effect_constancy.non_finite_estimate"));
    let (_, message) =
        refusal(run_independent(&[part("a", 1.0, 0.5), part("b", 1.0, 0.0)]).unwrap_err());
    assert!(message.contains("effect_constancy.invalid_standard_error"));
}

#[test]
fn f18_incompatible_partition_invalid_covariance_refuses() {
    let parts = [part("a", 1.0, 0.5), part("b", 2.0, 0.5)];
    let cases: [Vec<f64>; 5] = [
        vec![0.25, 0.0, 0.0],         // wrong size
        vec![0.25, 0.1, 0.2, 0.25],   // not symmetric
        vec![0.25, 0.5, 0.5, 0.25],   // not positive definite (correlation 2)
        vec![0.25, 0.25, 0.25, 0.25], // singular
        vec![1.0, 0.0, 0.0, 0.25],    // diagonal does not match the standard error
    ];
    for covariance in cases {
        let error = test_effect_constancy(
            &parts,
            &PartitionDependence::Covariance(covariance),
            &ContrastFamily::AllPairs,
            0.05,
        )
        .unwrap_err();
        let (code, message) = refusal(error);
        assert_eq!(code, "invalid_argument");
        assert!(message.contains("effect_constancy.invalid_covariance"), "{message}");
    }
    let error = test_effect_constancy(
        &parts,
        &PartitionDependence::Covariance(vec![0.25, f64::NAN, f64::NAN, 0.25]),
        &ContrastFamily::AllPairs,
        0.05,
    )
    .unwrap_err();
    assert!(refusal(error).1.contains("effect_constancy.invalid_covariance"));
}

#[test]
fn f18_incompatible_partition_unknown_reference_and_alpha_refuse() {
    let parts = [part("a", 1.0, 0.5), part("b", 2.0, 0.5)];
    let family = ContrastFamily::AgainstReference("zzz".into());
    let error = test_effect_constancy(&parts, &PartitionDependence::Independent, &family, 0.05)
        .unwrap_err();
    assert!(refusal(error).1.contains("effect_constancy.unknown_reference"));
    let error = test_effect_constancy(
        &parts,
        &PartitionDependence::Independent,
        &ContrastFamily::AllPairs,
        1.5,
    )
    .unwrap_err();
    assert!(refusal(error).1.contains("effect_constancy.invalid_alpha"));
}

#[test]
fn f18_partition_permutation_gives_bit_identical_statistic() {
    let a = part("a", 0.3, 0.4);
    let b = part("b", 1.1, 0.6);
    let c = part("c", -0.2, 0.5);
    let forward = run_independent(&[a.clone(), b.clone(), c.clone()]).unwrap();
    let reversed = run_independent(&[c.clone(), b.clone(), a.clone()]).unwrap();
    assert_eq!(forward, reversed);

    // Dependent: permute estimates and the covariance consistently.
    let cov = |order: &[usize]| {
        let full = [[0.16, 0.05, -0.02], [0.05, 0.36, 0.03], [-0.02, 0.03, 0.25]];
        let mut flat = Vec::new();
        for &i in order {
            for &j in order {
                flat.push(full[i][j]);
            }
        }
        flat
    };
    let parts = [a, b, c];
    let run = |order: [usize; 3]| {
        let permuted: Vec<PartitionEstimate> = order.iter().map(|&i| parts[i].clone()).collect();
        test_effect_constancy(
            &permuted,
            &PartitionDependence::Covariance(cov(&order)),
            &ContrastFamily::AllPairs,
            0.05,
        )
        .unwrap()
    };
    let base = run([0, 1, 2]);
    for order in [[2, 1, 0], [1, 2, 0], [0, 2, 1]] {
        let other = run(order);
        assert_eq!(base.test, other.test, "{order:?}");
        assert_eq!(base.contrasts, other.contrasts, "{order:?}");
        assert_eq!(base.partitions, other.partitions, "{order:?}");
    }
}

#[test]
fn covariance_validation_is_invariant_to_measurement_units() {
    for scale in [1.0, 1e-12, 1e12] {
        let parts = [part("a", scale, scale), part("b", 2.0 * scale, scale)];
        let s2 = scale * scale;
        let covariance = PartitionDependence::Covariance(vec![s2, 0.5 * s2, 0.1 * s2, s2]);
        assert!(
            test_effect_constancy(&parts, &covariance, &ContrastFamily::AllPairs, 0.05).is_err()
        );
        let valid = PartitionDependence::Covariance(vec![s2, 0.5 * s2, 0.5 * s2, s2]);
        let result =
            test_effect_constancy(&parts, &valid, &ContrastFamily::AllPairs, 0.05).unwrap();
        close(result.test.statistic, 1.0, "scale-invariant Wald");
    }
}

#[test]
fn overflow_underflow_and_blank_identities_refuse() {
    for se in [1e-300, 1e300] {
        assert!(run_independent(&[part("a", 0.0, se), part("b", 1.0, se)]).is_err());
    }
    assert!(run_independent(&[part("a", -1e308, 1.0), part("b", 1e308, 1.0)]).is_err());
    assert!(run_independent(&[part(" ", 0.0, 1.0), part("b", 1.0, 1.0)]).is_err());
    let mut parts = [part("a", 0.0, 1.0), part("b", 1.0, 1.0)];
    for p in &mut parts {
        p.estimand.population.clear();
    }
    assert!(run_independent(&parts).is_err());
}
