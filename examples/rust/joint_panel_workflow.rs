// SPDX-License-Identifier: MIT OR Apache-2.0
//! Actual compiler-owned fits of the independent panel fixture.
//! The example reports numerical diagnostics and model-conditional response only.
mod joint_panel_fixture;
use antecedent::{Study, analysis::joint_panel::JointPanelFit};
use antecedent_core::ExecutionContext;
use antecedent_estimate::joint_panel::{PanelInference, PanelPrior};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    for inference in [
        PanelInference::Frequentist { draws: 200 },
        PanelInference::Bayesian {
            prior: PanelPrior::default(),
            sampler: joint_panel_fixture::posterior_sampling(),
        },
    ] {
        let ctx = ExecutionContext::production(20_261_021, 1);
        let prepared = {
            let source = joint_panel_fixture::source(false);
            let mut request = joint_panel_fixture::request(&source, false, inference);
            request.assignment = Some(joint_panel_fixture::period_randomization());
            request.target =
                antecedent::analysis::joint_panel::declare_joint_panel_exogenous_context(
                    &request.target,
                );
            Study::prepare_joint_panel(source, request, &ctx)?
        };
        println!("compiled without fitting: {} fits", prepared.fits_performed());
        let result = prepared.estimate(&ctx)?;
        println!(
            "model-conditional worlds: {}, retained draw slots: {}",
            result.point.len(),
            result.draws.len()
        );
        match &result.fit {
            JointPanelFit::Invariant(fit) => {
                println!("actual fit diagnostics: {:?}", fit.diagnostics)
            }
            JointPanelFit::Varying(fit) => println!("actual change attempts: {:?}", fit.attempts),
        }
        // Re-evaluation reuses the checked fit rather than reconstructing a builder.
        let again = prepared.estimate(&ctx)?;
        assert_eq!(prepared.fits_performed(), 1);
        assert_eq!(result.point, again.point);
        println!(
            "fit count after reuse: {}; calibration remains unmeasured",
            prepared.fits_performed()
        );
    }
    let ctx = ExecutionContext::production(20_261_021, 1);
    let data = joint_panel_fixture::whole_program_source();
    let mut request =
        joint_panel_fixture::request(&data, false, PanelInference::Frequentist { draws: 0 });
    request.target =
        antecedent::analysis::joint_panel::declare_joint_panel_exogenous_context(&request.target);
    request.assignment = Some(joint_panel_fixture::whole_program_randomization());
    let experiment = Study::prepare_joint_panel(data, request, &ctx)?;
    assert!(experiment.inspect().identification.is_identified_claim());
    println!(
        "whole-program randomization identifies the worlds; model fitting/calibration earns no extra license"
    );
    Ok(())
}
