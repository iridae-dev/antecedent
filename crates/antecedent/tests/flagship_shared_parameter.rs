//! A shared parameter between the cluster margin and the cost contrast turns
//! the mixed-module policy into inference, not a covariance edit.
//!
//! The margin estimates a shared parameter `theta` and the cost contrast is
//! `theta + phi`. The margin-cost correlation is the exact posterior
//! correlation of `theta` and `theta + phi` under the policy: the coherent
//! joint model and a declared feedback let the cost data inform `theta`; a cut
//! learns `theta` from the margin alone but still carries `theta` forward into
//! the contrast, so the dependence it leaves is real and nonzero. The decision
//! variance read back from the exported artifact is the diagonal minus twice
//! that correlation times the routes' published standard deviations.
//!
//! SPDX-License-Identifier: MIT OR Apache-2.0

use antecedent::analysis::{FlagshipDependence, FlagshipError, FlagshipWorkflow, load_decision};
use antecedent_contract::mixed_module::{
    CutFeedback, GaussianLaw, MixedModulePolicy, SharedParameterModel,
};
use antecedent_core::ExecutionContext;

fn model(feedback_weight: f64) -> SharedParameterModel {
    SharedParameterModel {
        margin_prior: GaussianLaw { mean: 0.5, variance: 2.0 },
        cost_prior: GaussianLaw { mean: -0.4, variance: 1.0 },
        pooling_weight: 0.4,
        margin_observation: GaussianLaw { mean: 1.3, variance: 0.25 },
        cost_observation: GaussianLaw { mean: 2.1, variance: 0.5 },
        specific_prior: GaussianLaw { mean: 0.3, variance: 1.5 },
        feedback_weight,
    }
}

fn dependence(model: SharedParameterModel) -> FlagshipDependence {
    FlagshipDependence::SharedParameter {
        id: "shared-theta".to_owned(),
        model,
        margin_temporal: 0.0,
        temporal_cost: 0.0,
    }
}

fn variance_under(
    ctx: &ExecutionContext,
    model: SharedParameterModel,
    policy: &MixedModulePolicy,
) -> (f64, f64, String) {
    let workflow =
        FlagshipWorkflow::execute_with_policy(ctx, dependence(model), policy).expect("workflow");
    let artifact = workflow.export_decision().expect("export");
    let authority = workflow.into_authority();
    let loaded =
        load_decision(&artifact.bytes, Some(&artifact.identity), &authority, ctx).expect("load");
    (loaded.mean, loaded.variance, artifact.identity)
}

#[test]
fn each_policy_reloads_the_decision_variance_its_posterior_correlation_implies() {
    let ctx = ExecutionContext::for_tests(0xF1A6);
    let base = FlagshipWorkflow::execute(&ctx, FlagshipDependence::Independent).expect("base");
    let decision = base.decision().expect("decision");
    let (margin, temporal, cost) =
        (decision.margin.variance, decision.temporal_variance, decision.cost_variance);
    let diagonal = margin + temporal + cost;
    let base_mean = decision.joint.value("mean").expect("mean");

    let model = model(0.5);
    let cut = MixedModulePolicy::declared_modular("pooled theta, margin leads", CutFeedback::Cut)
        .expect("policy");
    let feedback =
        MixedModulePolicy::declared_modular("pooled theta, partial return", CutFeedback::Feedback)
            .expect("policy");
    let joint = MixedModulePolicy::coherent_joint();

    let mut variances = Vec::new();
    for (policy, eta) in [(&joint, 1.0), (&cut, 0.0), (&feedback, 0.5)] {
        let correlation = model.posterior(eta).expect("posterior").theta_contrast_correlation();
        let expected = diagonal - 2.0 * correlation * (margin * cost).sqrt();
        let (mean, variance, identity) = variance_under(&ctx, model, policy);
        assert!(
            (variance - expected).abs() <= 1.0e-9 * expected.abs(),
            "eta {eta}: {variance} vs {expected}"
        );
        // The dependence changes the spread, never the combined mean.
        assert!((mean - base_mean).abs() <= 1.0e-12 * base_mean.abs().max(1.0));
        variances.push((variance, identity));
    }
    // The three policies are three different decisions with three identities.
    for (left, right) in [(0, 1), (0, 2), (1, 2)] {
        assert!((variances[left].0 - variances[right].0).abs() > 1.0e-9 * variances[left].0);
        assert_ne!(variances[left].1, variances[right].1);
    }
    // A cut is not independence: the dependence that flows forward remains.
    assert!((variances[1].0 - diagonal).abs() > 1.0e-6 * diagonal);
}

#[test]
fn the_cut_dependence_responds_to_the_second_module_precision() {
    let ctx = ExecutionContext::for_tests(0xF1A6);
    let cut = MixedModulePolicy::declared_modular("pooled theta", CutFeedback::Cut).unwrap();
    let base = model(1.0);
    let noisier =
        SharedParameterModel { cost_observation: GaussianLaw { mean: 2.1, variance: 4.0 }, ..base };
    let (_, tight, _) = variance_under(&ctx, base, &cut);
    let (_, loose, _) = variance_under(&ctx, noisier, &cut);
    // The cut posterior of theta ignores the second observation, but phi's
    // conditional on theta does not: a noisier second module weakens the link.
    assert!((tight - loose).abs() > 1.0e-9 * tight);
}

#[test]
fn a_malformed_shared_parameter_declaration_is_refused() {
    let ctx = ExecutionContext::for_tests(0xF1A6);
    let joint = MixedModulePolicy::coherent_joint();
    let bad = SharedParameterModel { pooling_weight: 1.5, ..model(1.0) };
    assert!(matches!(
        FlagshipWorkflow::execute_with_policy(&ctx, dependence(bad), &joint),
        Err(FlagshipError::Dependence(_))
    ));
    let unnamed = FlagshipDependence::SharedParameter {
        id: String::new(),
        model: model(1.0),
        margin_temporal: 0.0,
        temporal_cost: 0.0,
    };
    assert!(matches!(
        FlagshipWorkflow::execute_with_policy(&ctx, unnamed, &joint),
        Err(FlagshipError::Dependence(_))
    ));
}
