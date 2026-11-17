//! A continuous program search is a bounded local poll over compiled programs: it publishes the
//! best feasible point it found and every attempt, never a global optimality gap or a global
//! certificate. On the complete aligned fitted law it also publishes the reliability of the
//! selected decision: a simultaneous interval for the selected program's value over every
//! retained proposal, with its calibration annotation. On point means there is no law to
//! carry it and the search says so.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

#[path = "../../../examples/rust/joint_panel_fixture.rs"]
mod fixture;

use antecedent::{
    Study,
    analysis::joint_panel::{
        JointPanelContinuousSearchRequest, JointPanelResponseFunctional, JointPanelScientificScope,
        JointPanelSearchUncertainty,
    },
};
use antecedent_core::ExecutionContext;
use antecedent_design::{
    decision_contract::{ProgramObjective, ProgramSearchContract, ProgramSearchSpace, UtilityExpr},
    program_search::{
        FiniteProgramDomain, ProgramControl,
        continuous::{ContinuousSearchOptions, ContinuousSearchReliability},
    },
};
use antecedent_estimate::joint_panel::PanelInference;

fn search_reliability(
    uncertainty: JointPanelSearchUncertainty,
) -> antecedent::analysis::joint_panel::JointPanelContinuousSearchResult {
    let ctx = ExecutionContext::for_tests(20_261_021);
    let data = fixture::source(false);
    let request = fixture::request(&data, false, PanelInference::Frequentist { draws: 20 });
    let result = Study::prepare_joint_panel(data, request, &ctx).unwrap().estimate(&ctx).unwrap();

    // Every scheduled control may move within a box above the base world's own program; the
    // starting point is that program, which is feasible (no budget, no restriction).
    let request = result.request();
    let base = &request.worlds[0];
    let mut initial = Vec::new();
    let mut controls = Vec::new();
    for (channel, variable) in base.program.channels.iter().zip(&request.columns.channels) {
        for (period, value) in channel.values.iter().copied().enumerate() {
            initial.push(value);
            controls.push(ProgramControl {
                variable: *variable,
                offset: i32::try_from(period).unwrap(),
                lower: value,
                upper: value + 1.0,
                unit_cost: 0.0,
            });
        }
    }
    let search = result
        .compile_continuous_search(
            JointPanelContinuousSearchRequest {
                base_world: base.world,
                scope: JointPanelScientificScope::ModelConditional,
                functional: JointPanelResponseFunctional::TerminalMean,
                uncertainty,
                contract: ProgramSearchContract {
                    space: ProgramSearchSpace {
                        id: "continuous_search_reliability".into(),
                        variables: vec!["response".into()],
                    },
                    objective: ProgramObjective::ExpectedUtility(UtilityExpr::Input(0)),
                    constraints: vec![],
                },
                domain: FiniteProgramDomain { controls, budget: None, restrictions: vec![] },
                utility_units: "outcome".into(),
                initial,
                options: ContinuousSearchOptions {
                    max_evaluations: 12,
                    max_iterations: 2,
                    initial_step: 0.5,
                    step_tolerance: 1e-6,
                    objective_tolerance: 1e-10,
                },
            },
            &ctx,
        )
        .unwrap();
    search.execute(&ctx).unwrap()
}

#[test]
fn continuous_program_search_on_point_means_says_it_needs_the_aligned_law() {
    let found = search_reliability(JointPanelSearchUncertainty::PointMeans);
    assert!(!found.search.attempts.is_empty());
    assert!(matches!(found.search.reliability, ContinuousSearchReliability::RequiresAlignedLaw));
    assert!(found.search.gap.is_none());
    assert!(!found.search.global_certificate);
}

#[test]
fn continuous_program_search_on_the_aligned_law_reports_the_selected_decision_reliability() {
    let found = search_reliability(JointPanelSearchUncertainty::AlignedLaw);
    assert!(!found.search.attempts.is_empty());
    assert!(found.search.gap.is_none());
    assert!(!found.search.global_certificate);
    let ContinuousSearchReliability::Interval(reliability) = &found.search.reliability else {
        panic!("the aligned law reports an interval, got {}", found.search.reliability.label());
    };
    let interval = &reliability.interval;
    let best = found.search.best.as_ref().expect("a feasible program");
    assert!(interval.lower <= interval.value && interval.value <= interval.upper);
    assert!((interval.value - best.value.unwrap()).abs() < 1e-6);
    assert!(interval.critical_value >= 0.0);
    assert!((interval.level - 0.95).abs() < 1e-12);
    assert!(interval.n_considered >= interval.n_retained as u64);
    assert!(interval.coordinate.adaptively_generated);
    // The calibration is an annotation of the published interval, never a reason to withhold it.
    assert!(!interval.calibration.status.as_str().is_empty());
    assert_eq!(reliability.draws.n_candidates(), interval.n_retained);
}
