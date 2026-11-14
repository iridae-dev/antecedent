//! A continuous program search is a bounded local poll over compiled programs: it publishes the
//! best feasible point it found and every attempt, never a global optimality gap, a global
//! certificate or a reliability figure for the selected decision.
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
    program_search::{FiniteProgramDomain, ProgramControl, continuous::ContinuousSearchOptions},
};
use antecedent_estimate::joint_panel::PanelInference;

#[test]
fn continuous_program_search_publishes_no_gap_certificate_or_reliability() {
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
                uncertainty: JointPanelSearchUncertainty::PointMeans,
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
    let found = search.execute(&ctx).unwrap();

    assert!(!found.search.attempts.is_empty());
    assert_eq!(found.search.reliability, "not_measured");
    assert!(found.search.gap.is_none());
    assert!(!found.search.global_certificate);
}
