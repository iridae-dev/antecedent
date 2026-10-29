//! Closed-form checks of the analytic planner against published values, the
//! solves in both directions, and the refusals.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use std::sync::Arc;

use antecedent_core::VariableId;
use antecedent_stats::adaptive_inference::group_sequential::SpendingFunction;

use super::*;
use crate::candidate::CandidateDesign;

fn power(target: f64) -> PlanningCriterion {
    PlanningCriterion::Power { target }
}

fn size(max_units: u64) -> PlanningGoal {
    PlanningGoal::SampleSize { max_units }
}

fn plan(
    design: PlannedDesign,
    assumptions: PlanningAssumptions,
    criterion: PlanningCriterion,
    goal: PlanningGoal,
) -> Result<PlanningResult, PlanningRefusal> {
    plan_analytic(&PlanningRequest::new(design, assumptions, criterion, goal))
}

#[test]
fn two_sample_t_sizes_match_cohen() {
    // Cohen (1988), two-sided 0.05, power 0.80: d = 0.5 needs 64 per group and
    // d = 0.8 needs 26 per group (exact noncentral t).
    for (d, per_group) in [(0.5, 64), (0.8, 26)] {
        let result = plan(
            PlannedDesign::Complete,
            PlanningAssumptions::continuous(1.0).with_effect(d),
            power(0.8),
            size(100_000),
        )
        .unwrap();
        let total = i64::try_from(result.units).unwrap();
        assert!((total - 2 * per_group).abs() <= 1, "d = {d}: {:?}", result.arm_units);
        assert!(result.analytic.value >= 0.8);
    }
    let small = plan(
        PlannedDesign::Complete,
        PlanningAssumptions::continuous(1.0).with_effect(0.2),
        power(0.8),
        size(100_000),
    )
    .unwrap();
    assert!(small.units.abs_diff(788) <= 4, "{}", small.units);
}

#[test]
fn proportion_sizes_match_the_unpooled_formula() {
    // 0.5 against 0.6, two-sided 0.05, power 0.80: 385 per group (unpooled Wald).
    let result = plan(
        PlannedDesign::Complete,
        PlanningAssumptions::binary(0.5).with_effect(0.1),
        power(0.8),
        size(100_000),
    )
    .unwrap();
    let per_group = result.arm_units[0];
    assert!((384..=386).contains(&per_group), "{per_group}");
    assert_eq!(result.analytic.reference_df, None);
}

#[test]
fn cluster_design_effect_matches_the_closed_forms() {
    let equal = plan(
        PlannedDesign::Cluster { mean_cluster_size: 20.0, icc: 0.05, cluster_size_cv: 0.0 },
        PlanningAssumptions::continuous(1.0).with_effect(0.5),
        power(0.8),
        size(10_000),
    )
    .unwrap();
    let de = equal.analytic.design_effect.unwrap();
    assert!((de - 1.95).abs() < 1e-12, "{de}");
    let (control, treated) = (equal.arm_units[0] as f64, equal.arm_units[1] as f64);
    let expected = de / 20.0 * (1.0 / control + 1.0 / treated);
    assert!((equal.analytic.estimator_variance - expected).abs() < 1e-12);
    // 64 individuals per arm inflate to 124.8, so at least 7 clusters per arm.
    assert!(equal.arm_units[0] >= 7);
    // Eldridge, Ashby and Kerry (2006): 1 + ((1 + CV²) m − 1) ρ.
    let unequal = plan(
        PlannedDesign::Cluster { mean_cluster_size: 20.0, icc: 0.05, cluster_size_cv: 0.5 },
        PlanningAssumptions::continuous(1.0).with_effect(0.5).with_control_mean(0.0),
        power(0.8),
        PlanningGoal::Evaluate { units: 20 },
    )
    .unwrap();
    assert!((unequal.analytic.design_effect.unwrap() - 2.2).abs() < 1e-12);
}

#[test]
fn familywise_error_at_the_null_is_alpha() {
    let assumptions = PlanningAssumptions::continuous(1.0).with_effect(0.0);
    let any = |multiplicity| {
        plan(
            PlannedDesign::MultiArm {
                treatment_arms: 3,
                comparison: ArmComparison::ManyToOne { multiplicity, power: FamilyPower::Any },
            },
            assumptions,
            power(0.8),
            PlanningGoal::Evaluate { units: 200 },
        )
        .unwrap()
        .analytic
        .value
    };
    let dunnett = any(Multiplicity::Dunnett);
    let bonferroni = any(Multiplicity::Bonferroni);
    let holm = any(Multiplicity::Holm);
    assert!((dunnett - 0.05).abs() < 1e-4, "{dunnett}");
    // Bonferroni tests each comparison at the Student-t level alpha / 6 (two-sided,
    // 196 df, critical 2.4147). The three statistics share the control (ρ = 1/2) and
    // the variance estimate, so the familywise error is below alpha: 0.04437 by the
    // joint law, against 0.0492 for independent normal comparisons.
    assert!(bonferroni < dunnett && (bonferroni - 0.04437).abs() < 2e-4, "{bonferroni}");
    assert!((holm - bonferroni).abs() < 1e-9, "Holm's first step is Bonferroni");
}

#[test]
fn holm_dominates_bonferroni_and_conjunctive_is_hardest() {
    let result = |multiplicity, family| {
        plan(
            PlannedDesign::MultiArm {
                treatment_arms: 3,
                comparison: ArmComparison::ManyToOne { multiplicity, power: family },
            },
            PlanningAssumptions::continuous(1.0).with_effect(0.5),
            power(0.8),
            PlanningGoal::Evaluate { units: 240 },
        )
        .unwrap()
    };
    let holm = result(Multiplicity::Holm, FamilyPower::All);
    let bonferroni = result(Multiplicity::Bonferroni, FamilyPower::All);
    let family = holm.analytic.family.as_ref().unwrap();
    assert!(family.all <= family.per_comparison && family.per_comparison <= family.any);
    assert!(holm.analytic.value > bonferroni.analytic.value);
}

#[test]
fn one_look_is_the_fixed_z_design_and_inflation_matches_jennison_turnbull() {
    let assumptions = PlanningAssumptions::continuous(1.0).with_effect(0.3);
    let one = plan(
        PlannedDesign::GroupSequential {
            information_fractions: vec![1.0],
            spending: SpendingFunction::OBrienFlemingType,
        },
        assumptions,
        power(0.9),
        PlanningGoal::Evaluate { units: 400 },
    )
    .unwrap();
    let z = 0.3 / (4.0_f64 / 400.0).sqrt();
    let fixed = antecedent_stats::power::t_test_power(z, None, 0.05, true);
    assert!((one.analytic.value - fixed).abs() < 1e-6);
    // Jennison and Turnbull (2000), table 2.4, K = 3, two-sided 0.05, power 0.9:
    // O'Brien-Fleming R = 1.022, Pocock R = 1.166. The spending-function
    // boundaries approximate them (observed 1.0225 and 1.154), so each band is a
    // few times the observed deviation and excludes the other family's value.
    for (spending, low, high) in [
        (SpendingFunction::OBrienFlemingType, 1.022 - 0.004, 1.022 + 0.004),
        (SpendingFunction::PocockType, 1.154 - 0.008, 1.154 + 0.008),
    ] {
        let result = plan(
            PlannedDesign::GroupSequential {
                information_fractions: vec![1.0 / 3.0, 2.0 / 3.0, 1.0],
                spending,
            },
            assumptions,
            power(0.9),
            size(1_000_000),
        )
        .unwrap();
        let sequential = result.analytic.sequential.as_ref().unwrap();
        let inflation = result.units as f64 / sequential.fixed_design_units.unwrap() as f64;
        assert!((low..high).contains(&inflation), "{spending:?}: {inflation}");
        assert!(sequential.expected_units_alternative < result.units as f64);
    }
}

#[test]
fn sample_size_is_monotone_and_inverse_consistent() {
    let mut previous = u64::MAX;
    for effect in [0.2, 0.3, 0.4, 0.6] {
        let assumptions = PlanningAssumptions::continuous(1.0)
            .with_effect(effect)
            .with_allocation_ratio(2.0)
            .with_attrition(0.1);
        let planned =
            plan(PlannedDesign::Complete, assumptions, power(0.85), size(1_000_000)).unwrap();
        assert!(planned.units <= previous);
        previous = planned.units;
        let again = plan(
            PlannedDesign::Complete,
            assumptions,
            power(0.85),
            PlanningGoal::Evaluate { units: planned.units },
        )
        .unwrap();
        assert!(again.analytic.value >= 0.85 && again.target_met);
        let before = plan(
            PlannedDesign::Complete,
            assumptions,
            power(0.85),
            PlanningGoal::Evaluate { units: planned.units - 1 },
        )
        .unwrap();
        assert!(before.analytic.value < 0.85);
        let mde = plan(
            PlannedDesign::Complete,
            PlanningAssumptions { effect: None, ..assumptions },
            power(0.85),
            PlanningGoal::MinimumDetectableEffect { units: planned.units },
        )
        .unwrap();
        assert!(mde.effect.unwrap() <= effect + 1e-9);
        assert!((mde.analytic.value - 0.85).abs() < 1e-6);
    }
}

#[test]
fn covariate_adjustment_and_bernoulli_assignment_move_power_as_they_should() {
    let base = PlanningAssumptions::continuous(1.0).with_effect(0.3);
    let at = |design, assumptions| {
        plan(design, assumptions, power(0.8), PlanningGoal::Evaluate { units: 200 })
            .unwrap()
            .analytic
            .value
    };
    let complete = at(PlannedDesign::Complete, base);
    let adjusted = at(PlannedDesign::Complete, base.with_covariates(0.5, 1));
    let bernoulli = at(PlannedDesign::Bernoulli, base);
    assert!(adjusted > complete);
    // ANCOVA with R² = 0.5 halves the residual variance of the contrast.
    let variance = |design, assumptions| {
        plan(design, assumptions, power(0.8), PlanningGoal::Evaluate { units: 200 })
            .unwrap()
            .analytic
            .estimator_variance
    };
    let ratio = variance(PlannedDesign::Complete, base.with_covariates(0.5, 1))
        / variance(PlannedDesign::Complete, base);
    assert!((ratio - 0.5).abs() < 1e-9, "{ratio}");
    assert!(bernoulli < complete && complete - bernoulli < 0.01, "{bernoulli} {complete}");
}

#[test]
fn switchback_variance_has_its_closed_form() {
    let result = plan(
        PlannedDesign::Switchback {
            sequences: 10,
            periods_per_block: 1,
            autocorrelation: 0.0,
            carryover: 0.0,
            washout_periods: 0,
        },
        PlanningAssumptions::continuous(2.0).with_effect(0.5).with_control_mean(0.0),
        power(0.8),
        PlanningGoal::Evaluate { units: 20 },
    )
    .unwrap();
    // One period per block, p = 1/2, zero level: (4σ² + τ²) / blocks.
    let expected = (4.0 * 4.0 + 0.25) / 200.0;
    assert!((result.analytic.estimator_variance - expected).abs() < 1e-12);
    assert_eq!(result.analytic.reference_df, Some(9.0));
}

#[test]
fn posterior_precision_solves_the_conjugate_bound() {
    let prior = NormalPrior { mean: 0.0, sd: 1.0 };
    let result = plan(
        PlannedDesign::Complete,
        PlanningAssumptions::continuous(1.0),
        PlanningCriterion::PosteriorPrecision { analysis_prior: prior, max_posterior_sd: 0.12 },
        size(100_000),
    )
    .unwrap();
    assert_eq!(result.units, 274);
    let posterior_sd = 1.0 / (1.0 + 68.5_f64).sqrt();
    assert!((result.analytic.value - posterior_sd).abs() < 1e-12);
}

#[test]
fn conjugate_assurance_matches_the_hybrid_form_under_a_vague_prior() {
    let design_prior = NormalPrior { mean: 0.3, sd: 0.15 };
    let assumptions =
        PlanningAssumptions::continuous(1.0).with_test(0.025, TestSidedness::OneSided);
    let at = |success| {
        plan(
            PlannedDesign::Complete,
            assumptions,
            PlanningCriterion::Assurance { design_prior, success, target: 0.7 },
            PlanningGoal::Evaluate { units: 4_000 },
        )
        .unwrap()
        .analytic
        .value
    };
    let hybrid = at(SuccessCriterion::Significance);
    let posterior = at(SuccessCriterion::Posterior {
        analysis_prior: NormalPrior { mean: 0.0, sd: 1e4 },
        threshold: 0.0,
        probability: 0.975,
    });
    // Closed form of the vague-prior posterior rule: Φ((m − z √v) / √(s² + v)).
    let v = 4.0_f64 / 4_000.0;
    let z = antecedent_kernels::norm_inv(0.975);
    let closed = antecedent_kernels::norm_cdf((0.3 - z * v.sqrt()) / (0.15_f64.powi(2) + v).sqrt());
    assert!((posterior - closed).abs() < 1e-6, "{posterior} vs {closed}");
    assert!((hybrid - posterior).abs() < 2e-3, "{hybrid} vs {posterior}");
    let point = plan(
        PlannedDesign::Complete,
        assumptions,
        PlanningCriterion::Assurance {
            design_prior: NormalPrior { mean: 0.3, sd: 0.0 },
            success: SuccessCriterion::Significance,
            target: 0.7,
        },
        PlanningGoal::Evaluate { units: 400 },
    )
    .unwrap();
    let fixed = plan(
        PlannedDesign::Complete,
        assumptions.with_effect(0.3),
        power(0.7),
        PlanningGoal::Evaluate { units: 400 },
    )
    .unwrap();
    assert!((point.analytic.value - fixed.analytic.value).abs() < 1e-12);
}

#[test]
fn unreachable_assurance_is_refused_with_its_limit() {
    let refusal = plan(
        PlannedDesign::Complete,
        PlanningAssumptions::continuous(1.0).with_test(0.025, TestSidedness::OneSided),
        PlanningCriterion::Assurance {
            design_prior: NormalPrior { mean: 0.1, sd: 0.2 },
            success: SuccessCriterion::Significance,
            target: 0.9,
        },
        size(100_000),
    )
    .unwrap_err();
    assert_eq!(refusal.code, "evidence_cannot_reach_level");
    assert_eq!(refusal.next_attempt, Some(RELAX_TARGET));
    assert!(refusal.message.contains("approaches"));
}

#[test]
fn designs_without_operating_characteristics_are_refused() {
    let assumptions = PlanningAssumptions::continuous(1.0).with_effect(0.3);
    let refuse = |design| plan(design, assumptions, power(0.8), size(1_000)).unwrap_err();
    let interference = refuse(PlannedDesign::Interference { exposure: None });
    assert_eq!(interference.code, "required_option_missing");
    assert_eq!(interference.next_attempt, Some(DECLARE_EXPOSURE_MODEL));
    let adaptive = refuse(PlannedDesign::OutcomeAdaptive { rule: None });
    assert_eq!(adaptive.code, "adaptive_contract_incomplete");
    assert_eq!(adaptive.next_attempt, Some(PRESPECIFY_STOPPING_RULE));
    let declared = refuse(PlannedDesign::OutcomeAdaptive { rule: Some("thompson".into()) });
    assert_eq!(declared.next_attempt, Some(SIMULATE_ADAPTIVE_PROCEDURE));
    let carryover = plan(
        PlannedDesign::Switchback {
            sequences: 10,
            periods_per_block: 4,
            autocorrelation: 0.3,
            carryover: 0.5,
            washout_periods: 0,
        },
        assumptions.with_control_mean(1.0),
        power(0.8),
        size(1_000),
    )
    .unwrap_err();
    assert_eq!(carryover.code, "effect_not_identified");
    assert_eq!(carryover.next_attempt, Some(DECLARE_WASHOUT));
    let unadjustable = plan(
        PlannedDesign::Cluster { mean_cluster_size: 10.0, icc: 0.1, cluster_size_cv: 0.0 },
        assumptions.with_covariates(0.3, 1),
        power(0.8),
        size(1_000),
    )
    .unwrap_err();
    assert_eq!(unadjustable.next_attempt, Some(COVARIATE_ADJUSTED_DESIGN));
    let contained = plan(
        PlannedDesign::Interference {
            exposure: Some(InterferenceExposure::ContainedInClusters {
                mean_cluster_size: 10.0,
                icc: 0.1,
                cluster_size_cv: 0.0,
            }),
        },
        assumptions,
        power(0.8),
        size(10_000),
    )
    .unwrap();
    assert_eq!(contained.analysis, "complete_cluster_itt");
}

#[test]
fn identity_retains_every_assumption() {
    let base = PlanningRequest::new(
        PlannedDesign::Complete,
        PlanningAssumptions::continuous(1.0).with_effect(0.3),
        power(0.8),
        size(1_000),
    );
    let changed = [
        PlanningRequest { assumptions: base.assumptions.with_attrition(0.1), ..base.clone() },
        PlanningRequest {
            assumptions: base.assumptions.with_allocation_ratio(2.0),
            ..base.clone()
        },
        PlanningRequest {
            assumptions: base.assumptions.with_test(0.05, TestSidedness::OneSided),
            ..base.clone()
        },
        PlanningRequest { criterion: power(0.9), ..base.clone() },
        base.clone().with_simulation(100, 1),
    ];
    for request in changed {
        assert_ne!(request.identity(), base.identity());
    }
}

#[test]
fn a_plan_feeds_design_ranking_candidates() {
    let result = plan(
        PlannedDesign::Cluster { mean_cluster_size: 20.0, icc: 0.05, cluster_size_cv: 0.0 },
        PlanningAssumptions::continuous(1.0).with_effect(0.5),
        power(0.8),
        size(10_000),
    )
    .unwrap();
    let targets: Arc<[VariableId]> = Arc::from(vec![VariableId::from_raw(0)]);
    let CandidateDesign::Intervene(experiment) = result.experiment_candidate(targets, 2.0, 7)
    else {
        panic!("an intervention candidate");
    };
    assert_eq!(experiment.cost.sample_budget, result.units * 20);
    assert!((experiment.cost.amount - 2.0 * (result.units * 20) as f64).abs() < 1e-9);
    let future = result.future_data().unwrap();
    let variance_of_mean = future.noise_variance() / future.sample_size() as f64;
    assert!((variance_of_mean - result.analytic.estimator_variance).abs() < 1e-12);
}
