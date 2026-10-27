//! Sensitivity analyses beyond the marginal sensitivity model, each an assumption range
//! bound to its target: E-values, omitted-variable bias with robustness values, Rosenbaum
//! bounds and the mediation correlation curve.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent_core::ExecutionContext;
use antecedent_core::execution::AdmissionPool;
use antecedent_validate::rosenbaum::DEFAULT_GAMMA_GRID;
use antecedent_validate::target_native::TargetCertificationError;
use antecedent_validate::{
    EffectInterval, EffectScale, LinearAdjustmentSummary, MatchedSet, MatchedSetScore,
    RefutationTarget, SensitivityAssumption, TargetFamily, TargetQuadrant, ValidationError,
    evalue_range, matched_sets_break_even, mediation_range, mediation_sensitivity_fit, ovb_range,
    rosenbaum_range,
};

fn scalar(source: u8) -> RefutationTarget {
    RefutationTarget::scalar_average_effect(TargetQuadrant::STATIC_FREQUENTIST, [source; 32])
}

#[test]
#[allow(clippy::float_cmp, reason = "no confounding returns the input estimate unchanged")]
fn evalue_range_states_the_assumption_the_range_and_the_break_even_values() {
    let (bounds, range) = evalue_range(
        scalar(1),
        EffectInterval::Ratio { estimate: 3.9, lower: Some(1.8), upper: Some(8.7) },
        EffectScale::RiskRatio,
    )
    .unwrap();
    assert_eq!(range.assumption, SensitivityAssumption::UnmeasuredConfoundingRiskRatio);
    assert_eq!(range.break_even, Some(bounds.point));
    assert_eq!(range.break_even_significance, bounds.limit);
    // No confounding leaves the estimate; a confounder at the E-value would reach the null,
    // so every probed strength below it leaves the adjusted risk ratio above one.
    assert_eq!(range.tested[0].target, 3.9);
    assert!(range.tested.iter().filter(|p| p.parameter < bounds.point).all(|p| p.target > 1.0));
    assert!(range.tested.windows(2).all(|w| w[0].target >= w[1].target));
}

#[test]
fn a_range_cannot_be_cited_by_a_claim_on_another_target() {
    let (_, range) = evalue_range(
        scalar(1),
        EffectInterval::Ratio { estimate: 2.0, lower: Some(1.2), upper: Some(3.0) },
        EffectScale::OddsRatio { common_outcome: true },
    )
    .unwrap();
    assert!(range.bound_to(&scalar(1)).is_ok());
    let conditional = RefutationTarget::new(
        TargetFamily::ConditionalEffect,
        TargetQuadrant::STATIC_FREQUENTIST,
        ["x<0", "x>=0"],
        [1; 32],
    )
    .unwrap();
    assert!(matches!(
        range.bound_to(&conditional),
        Err(TargetCertificationError::TargetMismatch { .. })
    ));
    // The target identity is part of the range identity.
    let (_, other) = evalue_range(
        scalar(2),
        EffectInterval::Ratio { estimate: 2.0, lower: Some(1.2), upper: Some(3.0) },
        EffectScale::OddsRatio { common_outcome: true },
    )
    .unwrap();
    assert_ne!(range.identity, other.identity);
}

#[test]
fn ovb_range_reports_the_extreme_scenario_and_robustness_values() {
    // Cinelli and Hazlett (2020) Darfur summary statistics.
    let summary = LinearAdjustmentSummary::new(0.097_315_82, 0.023_256_54, 783.0).unwrap();
    let range = ovb_range(scalar(3), &summary, 1.0, 0.05, 1.0).unwrap();
    assert_eq!(range.assumption, SensitivityAssumption::UnmeasuredConfoundingPartialR2);
    assert!((range.break_even.unwrap() - 0.1388).abs() < 5e-5);
    assert!((range.break_even_significance.unwrap() - 0.0763).abs() < 5e-5);
    // An extreme confounder (all residual outcome variance) explaining 5% of the residual
    // treatment variance already reverses the estimate: R2_{Y~D|X} is only 2.2%.
    let at_five = range.tested.iter().find(|p| (p.parameter - 0.05).abs() < 1e-12).unwrap();
    assert!(at_five.target < 0.0);
    let at_one = range.tested.iter().find(|p| (p.parameter - 0.01).abs() < 1e-12).unwrap();
    assert!(at_one.target > 0.0);
}

#[test]
fn rosenbaum_range_tracks_the_upper_bound_and_gamma_star() {
    let d = [0.8, 1.1, -0.2, 0.5, 1.9, 0.3, 1.4, 0.9, -0.4, 1.2, 0.7, 2.2, 0.6, 1.0, -0.1, 1.6];
    let (analysis, range) =
        rosenbaum_range(scalar(4), &d, &DEFAULT_GAMMA_GRID, 0.05, None).unwrap();
    assert!(analysis.exact);
    assert_eq!(range.assumption, SensitivityAssumption::HiddenBiasInMatchedAssignment);
    let gamma_star = range.break_even_significance.unwrap();
    assert!(gamma_star > 1.0);
    for point in range.tested.iter() {
        assert_eq!(point.parameter <= gamma_star, point.target <= 0.05, "{point:?}");
    }
}

#[test]
fn the_exact_signed_rank_table_leases_before_allocating() {
    let d: Vec<f64> =
        (1..=200).map(|i| f64::from(i) * if i % 7 == 0 { -1.0 } else { 1.0 }).collect();
    let pool = AdmissionPool::fixed(1, 1024);
    let before = pool.available();
    let ctx = ExecutionContext::for_tests(1).with_admission(pool.clone());
    let error = rosenbaum_range(scalar(5), &d, &[1.0, 2.0], 0.05, Some(&ctx)).unwrap_err();
    assert!(matches!(error, ValidationError::Resource { .. }), "{error:?}");
    assert_eq!(pool.available(), before);
}

#[test]
fn matched_sets_with_several_controls_have_a_break_even_gamma() {
    let sets: Vec<MatchedSet> = (0..30)
        .map(|i| {
            let base = f64::from(i % 5);
            MatchedSet { outcomes: vec![base + 1.5, base, base + 0.2, base - 0.1], treated: 0 }
        })
        .collect();
    let gamma = matched_sets_break_even(&sets, MatchedSetScore::AlignedRanks, 0.05).unwrap();
    assert!(gamma.is_some_and(|g| g > 1.0));
}

#[test]
fn mediation_range_breaks_even_at_the_residual_correlation() {
    let mut t = Vec::new();
    let mut m = Vec::new();
    let mut y = Vec::new();
    for i in 0..120 {
        let a = f64::from(i);
        let ti = f64::from(i % 2);
        let mi = 0.7 * ti + 0.5 * (a * 1.3).sin();
        t.push(ti);
        m.push(mi);
        y.push(0.2 * ti + 0.9 * mi + 0.4 * (a * 0.7).cos());
    }
    let summary = mediation_sensitivity_fit(&t, &m, &y, &[]).unwrap();
    let mediation = RefutationTarget::new(
        TargetFamily::Mediation,
        TargetQuadrant::STATIC_FREQUENTIST,
        ["total", "direct", "indirect"],
        [6; 32],
    )
    .unwrap();
    let range = mediation_range(mediation, &summary).unwrap();
    assert_eq!(range.break_even, Some(summary.rho_tilde));
    assert!(summary.rho_tilde > 0.0);
    let at_zero = range.tested.iter().find(|p| p.parameter == 0.0).unwrap();
    assert!(at_zero.target > 0.0);
    assert!(
        range.tested.iter().filter(|p| p.parameter > summary.rho_tilde).all(|p| p.target < 0.0)
    );
}
