//! Known-truth checks of the compiled experiment planner: sample sizes and powers
//! of declared randomized designs against published tables and closed forms
//! (Cohen's two-sample t table, the unpooled two-proportion formula, the
//! cluster design effect of Eldridge, Ashby and Kerry), through the compiled
//! program and its retained execution rather than the analytic solver alone.
//!
// SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::planning::compile_planning;
use antecedent::design::{
    PlannedDesign, PlanningAssumptions, PlanningCriterion, PlanningGoal, PlanningRequest,
    PlanningResult,
};
use antecedent_core::ExecutionContext;

fn run(request: &PlanningRequest) -> PlanningResult {
    let ctx = ExecutionContext::for_tests(7);
    let plan = compile_planning(request, &ctx).expect("compiles");
    let executed = plan.execute(&ctx).expect("executes");
    assert_eq!(executed.program_identity, plan.identity());
    assert_eq!(executed.result.calibration, "not_measured");
    executed.result
}

fn power(target: f64) -> PlanningCriterion {
    PlanningCriterion::Power { target }
}

#[test]
fn sample_sizes_match_cohens_two_sample_t_table() {
    // Two-sided 0.05, power 0.80, unit variance (Cohen 1988): d = 0.5 needs 64 per group
    // and d = 0.8 needs 26 per group; d = 0.2 needs 394 per group (788 in total).
    for (d, per_group) in [(0.5, 64_i64), (0.8, 26), (0.2, 394)] {
        let result = run(&PlanningRequest::new(
            PlannedDesign::Complete,
            PlanningAssumptions::continuous(1.0).with_effect(d),
            power(0.8),
            PlanningGoal::SampleSize { max_units: 100_000 },
        ));
        let total = i64::try_from(result.units).unwrap();
        assert!((total - 2 * per_group).abs() <= 4, "d = {d}: {:?}", result.arm_units);
        assert!(result.target_met && result.analytic.value >= 0.8);
    }
}

#[test]
fn power_at_a_given_size_matches_the_noncentral_t_value() {
    // 64 per group at d = 0.5 has power 0.8015 (exact noncentral t; the z approximation
    // is 0.807).
    let result = run(&PlanningRequest::new(
        PlannedDesign::Complete,
        PlanningAssumptions::continuous(1.0).with_effect(0.5),
        power(0.8),
        PlanningGoal::Evaluate { units: 128 },
    ));
    assert!((0.795..0.81).contains(&result.analytic.value), "{}", result.analytic.value);
}

#[test]
fn proportions_follow_the_unpooled_formula() {
    // 0.5 against 0.6, two-sided 0.05, power 0.80: n = (1.96 + 0.8416)^2 (0.25 + 0.24) / 0.01
    // = 384.6, so 385 per group.
    let result = run(&PlanningRequest::new(
        PlannedDesign::Complete,
        PlanningAssumptions::binary(0.5).with_effect(0.1),
        power(0.8),
        PlanningGoal::SampleSize { max_units: 100_000 },
    ));
    assert!((384..=386).contains(&result.arm_units[0]), "{}", result.arm_units[0]);
}

#[test]
fn cluster_design_effect_is_the_closed_form() {
    // 1 + (m - 1) rho for equal clusters of 20 at rho = 0.05 is 1.95; with a cluster-size
    // coefficient of variation 0.5 it is 1 + ((1 + CV^2) m - 1) rho = 2.2.
    let equal = run(&PlanningRequest::new(
        PlannedDesign::Cluster { mean_cluster_size: 20.0, icc: 0.05, cluster_size_cv: 0.0 },
        PlanningAssumptions::continuous(1.0).with_effect(0.5),
        power(0.8),
        PlanningGoal::SampleSize { max_units: 10_000 },
    ));
    assert!((equal.analytic.design_effect.unwrap() - 1.95).abs() < 1e-12);
    let unequal = run(&PlanningRequest::new(
        PlannedDesign::Cluster { mean_cluster_size: 20.0, icc: 0.05, cluster_size_cv: 0.5 },
        PlanningAssumptions::continuous(1.0).with_effect(0.5).with_control_mean(0.0),
        power(0.8),
        PlanningGoal::Evaluate { units: 20 },
    ));
    assert!((unequal.analytic.design_effect.unwrap() - 2.2).abs() < 1e-12);
}

#[test]
fn the_three_goals_invert_one_another() {
    // The smallest size reaching power 0.85 reaches it, one fewer unit does not, and the
    // minimum detectable effect at the planned size recovers the planned effect.
    let assumptions = PlanningAssumptions::continuous(1.0).with_effect(0.4).with_attrition(0.1);
    let planned = run(&PlanningRequest::new(
        PlannedDesign::Complete,
        assumptions,
        power(0.85),
        PlanningGoal::SampleSize { max_units: 1_000_000 },
    ));
    let at = |units: u64| {
        run(&PlanningRequest::new(
            PlannedDesign::Complete,
            assumptions,
            power(0.85),
            PlanningGoal::Evaluate { units },
        ))
    };
    assert!(at(planned.units).analytic.value >= 0.85);
    assert!(at(planned.units - 1).analytic.value < 0.85);
    let mde = run(&PlanningRequest::new(
        PlannedDesign::Complete,
        PlanningAssumptions { effect: None, ..assumptions },
        power(0.85),
        PlanningGoal::MinimumDetectableEffect { units: planned.units },
    ));
    assert!(mde.effect.unwrap() <= 0.4 + 1e-9);
    assert!((mde.analytic.value - 0.85).abs() < 1e-6);
}
