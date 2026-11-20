//! Prospective known-optimum target-feedback action choice. Every replicate
//! redraws source disturbances and target baselines, refits the cyclic model,
//! and checks a fixed design-level scope. No decision record is earned.
mod common;
use antecedent::analysis::cyclic_action_world_decision::CyclicDecisionOutput;
use antecedent::analysis::cyclic_posterior_action_decision::CyclicPosteriorDecisionRequest;
use antecedent::analysis::cyclic_target_feedback_decision::{
    CyclicTargetFeedbackRequest, TargetFeedbackFrame, compile_cyclic_target_feedback_decision,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{ExecutionContext, SemanticDigest};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_design::decision_eval::Verdict;
use antecedent_estimate::cyclic_likelihood::CyclicParameter;
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_model::nonlinear_equilibrium::{
    Activation, Clamp, Equation, EquilibriumModel, RegimeRow, SolvePolicy,
};
use common::calibration::{gaussian, grid_n, grid_seed, map_replicates, n_sim, uniform};

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
#[allow(clippy::too_many_lines, reason = "one replicate builds the whole simulated study")]
fn replicate(seed: u64, n: usize) -> Option<(SemanticDigest, bool)> {
    let mut baseline_rng = uniform(seed, 0xC1C1_A100);
    let mut target_rng = uniform(seed, 0xC1C1_A101);
    let mut ew = gaussian(seed ^ 0xC1C1_A102);
    let mut et = gaussian(seed ^ 0xC1C1_A103);
    let training = (0..n)
        .map(|i| {
            let ad = (i % 2) as f64;
            let baseline = if i == 0 {
                0.0
            } else if i == 1 {
                1.0
            } else {
                baseline_rng()
            };
            let w_noise = 0.2 * ew();
            let t_noise = 0.2 * et();
            let wishlist = (0.2 * ad + 0.4 * baseline + w_noise + 0.3 * t_noise) / 0.94;
            RegimeRow {
                values: vec![wishlist, 0.2 * wishlist + t_noise, ad, baseline],
                clamps: vec![
                    Clamp { variable: 2, value: ad },
                    Clamp { variable: 3, value: baseline },
                ],
            }
        })
        .collect();
    let equations = vec![
        Equation {
            id: "wishlists".into(),
            intercept: 0.0,
            weights: vec![0.0, 0.3, 0.2, 0.4],
            activation: Activation::Identity,
            expression: None,
            parameters: Vec::new(),
        },
        Equation {
            id: "traffic".into(),
            intercept: 0.0,
            weights: vec![0.2, 0.0, 0.0, 0.0],
            activation: Activation::Identity,
            expression: None,
            parameters: Vec::new(),
        },
        Equation {
            id: "ads".into(),
            intercept: 0.0,
            weights: vec![0.0; 4],
            activation: Activation::Identity,
            expression: None,
            parameters: Vec::new(),
        },
        Equation {
            id: "baseline".into(),
            intercept: 0.0,
            weights: vec![0.0; 4],
            activation: Activation::Identity,
            expression: None,
            parameters: Vec::new(),
        },
    ];
    let source = CyclicPosteriorDecisionRequest {
        model: EquilibriumModel {
            equations,
            noise_covariance: vec![
                0.04, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.0, 0.04, 0.0, 0.0, 0.0, 0.0, 0.04,
            ],
        },
        parameters: vec![CyclicParameter::Weight { equation: 0, parent: 1 }],
        training,
        prior_sd: 1.0,
        chains: 4,
        warmup: 500,
        draws_per_chain: 1_500,
        proposal_scale: 0.14,
        worlds: vec![
            vec![Clamp { variable: 2, value: 0.0 }],
            vec![Clamp { variable: 2, value: 1.0 }],
        ],
        initial: vec![0.0; 4],
        solve_policy: SolvePolicy::default(),
    };
    let outputs = ["ordinary", "promoted"]
        .iter()
        .enumerate()
        .map(|(world, regime)| CyclicDecisionOutput {
            world,
            equation: 0,
            quantity: ScientificQuantityWire::from(&quantity(regime)),
        })
        .collect();
    let contract = DecisionContract {
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
    };
    let target = TargetFeedbackFrame {
        source_sample: "trial".into(),
        target_sample: "target".into(),
        target_population: "game".into(),
        baseline_indices: vec![3],
        unit_ids: (0..n).map(|i| format!("target-{i}")).collect(),
        rows: (0..n).map(|_| vec![0.2 + 0.6 * target_rng()]).collect(),
    };
    let ctx = ExecutionContext::for_tests(grid_seed(seed));
    let plan = compile_cyclic_target_feedback_decision(
        CyclicTargetFeedbackRequest { source, target, outputs },
        contract,
        &ctx,
    )
    .ok()?;
    let scope = plan.decision_reliability_scope().ok()?;
    let result = plan.execute(&ctx).ok()?;
    if !result.posterior.diagnostics_pass {
        return None;
    }
    assert_eq!(result.calibration, "not_measured");
    let correct = matches!(&result.decision.decision.verdict, Verdict::UniquelyOptimal(id) if id == "promoted");
    Some((scope, correct))
}
#[test]
#[ignore = "calibration: prospective target-feedback known-optimum choice; no record earned"]
fn cyclic_target_feedback_action_choice() {
    let n = grid_n(240);
    let (mut scored, mut refused, mut correct) = (0_u32, 0_u32, 0_u32);
    let mut scope = None;
    for result in map_replicates(n_sim(), |rep| replicate(0xC1C1_A200 + rep, n)) {
        if let Some((coordinate, chose_best)) = result {
            if let Some(previous) = scope {
                assert_eq!(previous, coordinate);
            } else {
                scope = Some(coordinate);
            }
            scored += 1;
            correct += u32::from(chose_best);
        } else {
            refused += 1;
        }
    }
    assert!(scored > 0, "no usable feedback fits");
    eprintln!(
        "prospective target-feedback choices: {correct}/{scored}, refused {refused}; no record earned"
    );
}
