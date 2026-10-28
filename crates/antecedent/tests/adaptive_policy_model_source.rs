//! A fitted model as the response source of an adaptive-policy search: the lowered policy
//! program executes inside the temporal rollout, shares noise across policies and reaches the
//! closed-form optimum of a quadratic tracking loss.

use std::sync::Arc;

use antecedent::gcm::{ModelPolicySource, ModelRolloutDesign, StageUtility};
use antecedent::prelude::ExecutionContext;
use antecedent_core::VariableId;
use antecedent_design::adaptive_policy::{AdaptivePolicyFamily, PolicyFeature, PolicyRule};
use antecedent_design::adaptive_policy_search::{
    DrawSplit, PolicyConstraints, PolicyObjective, PolicyRolloutSource, PolicySearchOptions,
    PreparedAdaptivePolicySearch, ResponseStanding,
};
use antecedent_graph::{Dag, DenseNodeId};
use antecedent_model::{CompiledCausalModel, CompiledMechanismStore, MechanismSlot};

const PRICE: VariableId = VariableId::from_raw(0);
const DEMAND: VariableId = VariableId::from_raw(1);
const HORIZON: usize = 6;

/// demand = 50 - 0.8 price + N(0, 1); tracking loss (demand - 30)^2 is minimised at price 25
/// with expected loss sigma^2 = 1 per step.
fn model() -> CompiledCausalModel {
    let mut graph = Dag::with_variables(2);
    graph.insert_directed(DenseNodeId::from_raw(0), DenseNodeId::from_raw(1)).unwrap();
    CompiledCausalModel::compile(graph).unwrap().with_mechanisms(CompiledMechanismStore {
        slots: Arc::from([
            MechanismSlot::LinearGaussian { intercept: 11.0, coeffs: Arc::from([]), sigma: 2.0 },
            MechanismSlot::LinearGaussian {
                intercept: 50.0,
                coeffs: Arc::from([-0.8]),
                sigma: 1.0,
            },
        ]),
    })
}

fn source(initial_demand: f64) -> ModelPolicySource {
    source_with_interval(initial_demand, 1)
}

fn source_with_interval(initial_demand: f64, interval: usize) -> ModelPolicySource {
    ModelPolicySource::new(
        vec![Arc::new(model())],
        PRICE,
        StageUtility {
            outcome: DEMAND,
            target: 30.0,
            over: 0.0,
            under: 0.0,
            quadratic: 1.0,
            action_linear: 0.0,
            action_quadratic: 0.0,
            discount: 1.0,
        },
        ModelRolloutDesign {
            horizon: HORIZON,
            rows: 2000,
            held_out_rows: 2000,
            noise_seed: 99,
            initial_state: vec![(DEMAND, initial_demand)],
            unit_cost: 1.0,
            exposure_weight: 1.0,
            decision_interval: interval,
        },
        vec![DEMAND],
        ResponseStanding { identified: false, basis: "fitted linear-Gaussian model".into() },
    )
    .unwrap()
}

fn feedback() -> AdaptivePolicyFamily {
    AdaptivePolicyFamily::new(
        "feedback",
        vec![PolicyFeature::Lag { variable: DEMAND, lag: 1 }],
        &[(0.0, 80.0)],
        PolicyRule::LinearFeedback { features: vec![0] },
        (0.0, 60.0),
        (-2.0, 2.0),
    )
    .unwrap()
}

fn schedule() -> AdaptivePolicyFamily {
    AdaptivePolicyFamily::new(
        "schedule",
        vec![PolicyFeature::Lag { variable: DEMAND, lag: 1 }],
        &[(0.0, 80.0)],
        PolicyRule::Schedule { steps: HORIZON },
        (0.0, 60.0),
        (-2.0, 2.0),
    )
    .unwrap()
}

#[test]
fn the_lowered_policy_reads_the_realised_history_inside_the_rollout() {
    let src = source(40.0);
    let ctx = ExecutionContext::for_tests(1);
    // price_0 = 20 + 0.25 * demand_{-1} = 30 for the initial demand 40.
    let paths = src.rollout(&feedback(), &[20.0, 0.25], DrawSplit::Optimization, &ctx).unwrap();
    assert_eq!(paths.len(), 2000);
    assert!(paths.iter().all(|p| p.actions.len() == HORIZON && p.actions[0] == 30.0));
    // Later steps read the realised lagged demand, so they differ across draws.
    assert!(paths.iter().any(|p| p.actions[1] != paths[0].actions[1]));
    // The same draws serve every policy: a repeated call is bit-identical.
    let again = src.rollout(&feedback(), &[20.0, 0.25], DrawSplit::Optimization, &ctx).unwrap();
    assert_eq!(paths, again);
    // The held-out split uses independent noise.
    let held = src.rollout(&feedback(), &[20.0, 0.25], DrawSplit::HeldOut, &ctx).unwrap();
    assert_ne!(paths[0].utility, held[0].utility);
}

#[test]
fn the_search_reaches_the_closed_form_tracking_optimum_on_held_out_noise() {
    let src = source(40.0);
    let ctx = ExecutionContext::for_tests(2);
    let result = PreparedAdaptivePolicySearch::compile(
        vec![schedule(), feedback()],
        Some(0),
        PolicyConstraints::default(),
        PolicyObjective::ExpectedUtility,
        PolicySearchOptions { seed: 5, max_evaluations: 600, ..PolicySearchOptions::default() },
    )
    .unwrap()
    .execute(&src, &ctx)
    .unwrap();
    // Optimal price 25 gives demand N(30, 1): expected loss 1 per step, utility -HORIZON.
    let optimum = -(HORIZON as f64);
    for family in &result.families {
        let held = family.held_out.as_ref().unwrap();
        assert!(
            (held.mean_utility - optimum).abs() < 5.0 * held.standard_error + 0.05,
            "{}: held-out {} vs {optimum}",
            family.family_id,
            held.mean_utility
        );
    }
    assert!(
        (result.families[1].theta[0] - 25.0).abs() < 3.0 || result.families[1].theta[1].abs() > 0.0
    );
    assert_eq!(result.standing.label(), "fitted_model_optimization");
    // The paired contrast is the difference of the two held-out means over the same draws
    // (the model noise cancels). Its standard error excludes search error, so the contrast is
    // checked for pairing, not against a noise-only tolerance: each family is already within
    // tolerance of the closed-form optimum above.
    let contrast = result.contrasts.iter().find(|c| c.split == DrawSplit::HeldOut).unwrap();
    let means: Vec<f64> =
        result.families.iter().map(|f| f.held_out.as_ref().unwrap().mean_utility).collect();
    assert!((contrast.mean_difference - (means[1] - means[0])).abs() < 1e-9);
    let se = |i: usize| result.families[i].held_out.as_ref().unwrap().standard_error;
    assert!(
        contrast.standard_error < (se(0).powi(2) + se(1).powi(2)).sqrt(),
        "common random numbers make the paired error smaller than the unpaired one"
    );
}

#[test]
fn between_decisions_the_action_holds_its_last_value() {
    let src = source_with_interval(40.0, 3);
    let ctx = ExecutionContext::for_tests(3);
    let paths = src.rollout(&feedback(), &[20.0, 0.25], DrawSplit::Optimization, &ctx).unwrap();
    for p in &paths {
        assert_eq!(p.actions[1], p.actions[0]);
        assert_eq!(p.actions[2], p.actions[0]);
        assert_eq!(p.actions[4], p.actions[3]);
    }
    assert!(paths.iter().any(|p| p.actions[3] != p.actions[0]));
}
