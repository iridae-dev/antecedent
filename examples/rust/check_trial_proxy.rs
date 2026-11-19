// SPDX-License-Identifier: MIT OR Apache-2.0
//! Check an early measure against a later outcome in another population.
//!
//! Two independent sets of synthetic trial effects make the target relation checkable.

use antecedent::analysis::bridge_transport_premise_assessment::{
    BridgeTransportPremiseAssessmentRequest, compile_bridge_transport_premise_assessment,
    consume_bridge_transport_premise_assessment,
};
use antecedent_contract::bridge::{
    BridgeScope, TransportabilityPremise, TrialEffectPair, TrialLevelValidation,
};
use antecedent_core::ExecutionContext;

fn main() {
    let trials = |target: bool| TrialLevelValidation {
        trials: (0..20)
            .map(|i| {
                let x = -1.0 + 2.0 * i as f64 / 19.0;
                let shift = if target { 0.05 } else { 0.0 };
                TrialEffectPair {
                    surrogate_effect: x,
                    terminal_effect: 0.3
                        + 2.0 * x
                        + shift
                        + (if i % 2 == 0 { 0.01 } else { -0.01 }),
                }
            })
            .collect(),
        min_trials: 3,
        min_r_squared: 0.0,
    };
    let request = BridgeTransportPremiseAssessmentRequest {
        premise: TransportabilityPremise {
            id: "surrogate-line-invariant".into(),
            statement: "the trial-level surrogate relation transfers".into(),
            established_in: BridgeScope {
                population_id: "source".into(),
                regime_id: "trial".into(),
            },
            applies_to: BridgeScope { population_id: "target".into(), regime_id: "trial".into() },
            evidence: Some("independent-target-validation".into()),
        },
        source: trials(false),
        target: trials(true),
        source_trial_ids: (0..20).map(|i| format!("source-trial-{i}")).collect(),
        target_trial_ids: (0..20).map(|i| format!("target-trial-{i}")).collect(),
        surrogate_grid: vec![-0.8, 0.0, 0.8],
        equivalence_margin: 0.2,
    };
    let ctx = ExecutionContext::for_tests(73);
    let plan = compile_bridge_transport_premise_assessment(request.clone()).unwrap();
    assert_eq!(plan.plan_view().route().as_str(), "BridgeTransportPremiseAssessment");
    let result = plan.execute(&ctx).unwrap();
    assert!(result.equivalence_screen_passed);
    assert_eq!(result.calibration, "not_measured");
    let bytes = plan.export(&result, &ctx).unwrap();
    let replay =
        consume_bridge_transport_premise_assessment(&bytes, result.lowered_program_identity, &ctx)
            .unwrap();
    assert_eq!(replay, result);
    println!("Screen passed: {}", result.equivalence_screen_passed);
    println!("Grid discrepancy and support: {:?}", result);
    let mut larger_shift = request;
    for trial in &mut larger_shift.target.trials {
        trial.terminal_effect += 0.4;
    }
    let changed =
        compile_bridge_transport_premise_assessment(larger_shift).unwrap().execute(&ctx).unwrap();
    assert!(!changed.equivalence_screen_passed);
    println!("Screen with a larger target shift: {}", changed.equivalence_screen_passed);
}
