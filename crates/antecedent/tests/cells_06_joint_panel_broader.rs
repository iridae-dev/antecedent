// SPDX-License-Identifier: MIT OR Apache-2.0
//! A held-out whole-program intervention on the broader panel family: a world whose complete
//! program no source group received is not certified by the design, however well the mechanism
//! fits, and the model-conditional response there stays unlicensed.
#[path = "../../../examples/rust/joint_panel_fixture.rs"]
mod fixture;

use antecedent::Study;
use antecedent::analysis::joint_panel::{
    JointPanelDesignOutcome, JointPanelFunctional, declare_joint_panel_exogenous_context,
    joint_panel_target,
};
use antecedent_core::{ExecutionContext, InferenceSelection, ScientificWorldId};
use antecedent_estimate::joint_panel::{
    PanelAdstock, PanelFamily, PanelInference, PanelOptimizerOptions, PanelSaturation,
};

fn broader_family() -> PanelFamily {
    PanelFamily {
        adstock: PanelAdstock::Geometric,
        saturation: PanelSaturation::Hill,
        scope: PanelFamily::per_group_saturation(4),
        spatial: None,
    }
}

#[test]
fn cells_06_a_held_out_whole_program_is_not_certified_by_the_broader_family_fit() {
    let ctx = ExecutionContext::for_tests(20_261_021);
    let bounded = PanelOptimizerOptions { max_iterations: 600, gradient_tolerance: 1e-2 };
    let request = |held_out: bool| {
        let data = fixture::whole_program_source();
        let mut request = fixture::request(&data, false, PanelInference::Frequentist { draws: 0 });
        if held_out {
            // The treated worlds are declared at dose 2; the experiment assigned dose 4, and no
            // group ever received dose 2. The target is rebuilt for the new worlds.
            for world in &mut request.worlds[1..] {
                for channel in &mut world.program.channels {
                    if channel.values[0] == 4.0 {
                        channel.values = vec![2.0; 24];
                    }
                }
            }
            request.target = joint_panel_target(
                data.schema(),
                &request.columns,
                2,
                &request.worlds,
                JointPanelFunctional::Interaction {
                    joint: ScientificWorldId::from_raw(3),
                    a_at_reference: ScientificWorldId::from_raw(1),
                    b_at_reference: ScientificWorldId::from_raw(2),
                    reference: ScientificWorldId::from_raw(0),
                },
                InferenceSelection::Frequentist,
            )
            .unwrap();
        }
        request.target = declare_joint_panel_exogenous_context(&request.target);
        request.assignment = Some(fixture::whole_program_randomization());
        request.options.optimizer = bounded;
        request.family = broader_family();
        (data, request)
    };

    // Control: the declared doses are the assigned doses, so the broader family inherits the
    // design claim for every world.
    let (data, control) = request(false);
    let control = Study::prepare_joint_panel(data, control, &ctx).unwrap();
    assert!(control.inspect().identification.is_identified_claim());
    let inspection = control.inspect();
    let attempt = inspection
        .design_attempts
        .iter()
        .find(|attempt| attempt.id == "randomized.multi_arm_itt.whole_program")
        .unwrap();
    assert!(matches!(attempt.outcome, JointPanelDesignOutcome::Identified));
    assert!(attempt.worlds.iter().all(|w| w.identified && w.matched_groups.len() == 1));

    // Held out: the same fit procedure, one world at doses no group received.
    let (data, held_out) = request(true);
    let held_out = Study::prepare_joint_panel(data, held_out, &ctx).unwrap();
    let inspection = held_out.inspect();
    assert_eq!(inspection.fits_performed, 0);
    assert!(!inspection.identification.is_identified_claim());
    let rejected = inspection
        .design_attempts
        .iter()
        .find(|attempt| attempt.id == "randomized.multi_arm_itt.whole_program")
        .unwrap();
    assert!(matches!(rejected.outcome, JointPanelDesignOutcome::Rejected));
    assert!(rejected.detail.contains("did not receive its assigned complete program"));
    assert!(rejected.next_attempt.is_some());
    assert!(rejected.proof.is_none());
    assert!(rejected.worlds.iter().all(|w| !w.identified));
    for world in &rejected.worlds[1..] {
        assert!(world.matched_groups.is_empty(), "no group received the held-out program");
    }
    assert_eq!(rejected.worlds[0].matched_groups.len(), 1, "the reference was observed");

    // The fit still produces a model-conditional response at the held-out doses, finite and
    // without a causal license.
    let result = held_out.estimate(&ctx).unwrap();
    assert!(result.point.iter().all(|world| world.means.iter().all(|m| m.is_finite())));
    assert!(result.require_identified().is_err());
}
