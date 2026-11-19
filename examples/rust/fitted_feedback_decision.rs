// SPDX-License-Identifier: MIT OR Apache-2.0
//! Fit a feedback loop, compare actions and replay the decision.
//!
//! The synthetic campaign rows fix the ad setting and supply wishlist and traffic outcomes.

use antecedent::analysis::cyclic_action_world_decision::CyclicDecisionOutput;
use antecedent::analysis::cyclic_posterior_action_decision::{
    CyclicPosteriorDecisionRequest, compile_cyclic_posterior_action_decision,
    consume_cyclic_posterior_action_decision,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::ExecutionContext;
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_estimate::cyclic_likelihood::CyclicParameter;
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_model::nonlinear_equilibrium::{
    Activation, Clamp, Equation, EquilibriumModel, RegimeRow, SolvePolicy,
};

fn quantity(regime: &str) -> ScientificQuantity {
    ScientificQuantity {
        variable_id: "wishlists".into(),
        variable_name: "wishlists".into(),
        role: QuantityRole::Outcome,
        units: "count".into(),
        population_id: "game".into(),
        regime_id: regime.into(),
        horizon: 0,
        functional_id: "outcome".into(),
        conditioning: Vec::new(),
        transform_id: "identity".into(),
    }
}

fn request() -> CyclicPosteriorDecisionRequest {
    let training = (0..260)
        .map(|i| {
            let ad = (i % 5) as f64 * 0.4;
            let ew = (i as f64 * 0.19).sin() * 0.2;
            let et = (i as f64 * 0.23).cos() * 0.2;
            let wishlists = (0.2 * ad + ew + 0.3 * et) / (1.0 - 0.06);
            let traffic = 0.2 * wishlists + et;
            RegimeRow {
                values: vec![wishlists, traffic, ad],
                clamps: vec![Clamp { variable: 2, value: ad }],
            }
        })
        .collect();
    CyclicPosteriorDecisionRequest {
        model: EquilibriumModel {
            equations: vec![
                Equation {
                    id: "wishlists".into(),
                    intercept: 0.0,
                    weights: vec![0.0, 0.3, 0.2],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: Vec::new(),
                },
                Equation {
                    id: "traffic".into(),
                    intercept: 0.0,
                    weights: vec![0.2, 0.0, 0.0],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: Vec::new(),
                },
                Equation {
                    id: "ads".into(),
                    intercept: 0.0,
                    weights: vec![0.0, 0.0, 0.0],
                    activation: Activation::Identity,
                    expression: None,
                    parameters: Vec::new(),
                },
            ],
            noise_covariance: vec![0.04, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.04],
        },
        parameters: vec![CyclicParameter::Weight { equation: 0, parent: 1 }],
        training,
        prior_sd: 1.0,
        chains: 2,
        warmup: 300,
        draws_per_chain: 300,
        proposal_scale: 0.08,
        worlds: vec![
            vec![Clamp { variable: 2, value: 0.0 }],
            vec![Clamp { variable: 2, value: 1.0 }],
        ],
        initial: vec![0.0; 3],
        solve_policy: SolvePolicy::default(),
    }
}

fn outputs() -> Vec<CyclicDecisionOutput> {
    ["ordinary", "promoted"]
        .iter()
        .enumerate()
        .map(|(world, regime)| CyclicDecisionOutput {
            world,
            equation: 0,
            quantity: ScientificQuantityWire::from(&quantity(regime)),
        })
        .collect()
}

fn contract() -> DecisionContract {
    DecisionContract {
        actions: ["ordinary", "promoted"]
            .iter()
            .map(|regime| DecisionAction {
                id: (*regime).into(),
                kind: ActionKind::Intervention,
                inputs: vec![quantity(regime)],
                utility: UtilityExpr::Input(0),
            })
            .collect(),
        utility_units: "wishlist_count".into(),
        criterion: DecisionCriterion::ExpectedRegret,
        constraints: Vec::new(),
        target_population: "game".into(),
        horizon: 0,
        structural_policy: StructuralPolicy::RequireInvariantBestAction,
    }
}

fn main() {
    let ctx = ExecutionContext::for_tests(47);
    let plan = compile_cyclic_posterior_action_decision(request(), outputs(), contract(), &ctx)
        .expect("posterior action compiler");
    assert_eq!(plan.plan_view().route().as_str(), "CyclicPosteriorActionDecision");
    let result = plan.execute(&ctx).expect("joint fitted predictive worlds");
    assert!(result.posterior.diagnostics_pass);
    assert_eq!(result.decision.decision.n_draws, 600);
    let bytes =
        plan.export(&result, "cyclic-fitted-worlds", &ctx).expect("portable fitted decision");
    let replay = consume_cyclic_posterior_action_decision(&bytes, plan.lowered().identity(), &ctx)
        .expect("independent fitted decision replay");
    assert_eq!(replay.decision.decision, result.decision.decision);
    let mut changed = request();
    changed.worlds[1][0].value = 1.5;
    let changed_plan =
        compile_cyclic_posterior_action_decision(changed, outputs(), contract(), &ctx).unwrap();
    assert_ne!(plan.lowered().identity(), changed_plan.lowered().identity());
    assert!(
        consume_cyclic_posterior_action_decision(&bytes, changed_plan.lowered().identity(), &ctx)
            .is_err()
    );

    println!("Compared {} paired outcome draws per action", result.decision.decision.n_draws);
    println!("Decision: {:?}", replay.decision.decision.verdict);
}
