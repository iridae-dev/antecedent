// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fit a joint panel response model once and evaluate whole intervention programs under it.
//!
//! Question: what does a two-channel panel with carryover predict for complete intervention
//! programs, and which assignment design identifies those programs?
//!
//! Data: the retained conformance panels under `conformance/response/joint_panel/`, parsed by
//! `joint_panel_fixture.rs`. The first panel assigns doses period by period; the second assigns
//! whole programs to independent groups. The fixture hides no fitting.
//!
//! Steps: for frequentist and for Bayesian inference, `Study::prepare_joint_panel` compiles the
//! request without fitting, `estimate` fits and evaluates the declared worlds, and a second
//! `estimate` reuses the checked fit. The last block prepares the whole-program experiment and
//! reads its identification.
//!
//! Asserted: two estimates perform exactly one fit and return the same points; the
//! whole-program request is an identified claim.
//!
//! What this establishes: the printed responses of the first panel are conditional on the fitted
//! mechanism. Whole-program randomization with a declared exogenous context identifies the
//! program worlds, and the model fit adds no identification. The intervals are not checked
//! against the fixture's truth here.
//!
//! Run: `cargo run -p antecedent --example joint_panel_workflow`
//! Source: `examples/rust/joint_panel_workflow.rs`
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
        // A second evaluation reuses the checked fit: the fit count stays at one and the
        // points are identical.
        let again = prepared.estimate(&ctx)?;
        assert_eq!(prepared.fits_performed(), 1);
        assert_eq!(result.point, again.point);
        println!("fit count after reuse: {}", prepared.fits_performed());
    }
    // Assigning complete programs at random, with the context declared exogenous, is the design
    // that identifies a program's effect. Preparation decides this before any fit.
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
        "whole-program randomization identifies the worlds; the model fit adds no identification"
    );
    Ok(())
}
