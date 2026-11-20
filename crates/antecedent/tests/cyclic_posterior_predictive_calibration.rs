//! Whole-route population prediction coverage for a connected Gaussian feedback
//! model. Every replicate generates new intervention-labelled training rows,
//! refits the joint cyclic parameter posterior, draws paired predictive
//! equilibria and scores one outcome interval against an independent future
//! equilibrium under the same specified model.

mod common;

use antecedent::analysis::cyclic_action_world_decision::CyclicDecisionOutput;
use antecedent::analysis::cyclic_posterior_action_decision::{
    CyclicPosteriorDecisionRequest, compile_cyclic_posterior_action_decision,
};
use antecedent_contract::{QuantityRole, ScientificQuantity};
use antecedent_core::{CalibrationBasis, ExecutionContext};
use antecedent_design::decision_contract::{
    ActionKind, DecisionAction, DecisionContract, DecisionCriterion, StructuralPolicy, UtilityExpr,
};
use antecedent_estimate::cyclic_likelihood::CyclicParameter;
use antecedent_io::quantity_wire::ScientificQuantityWire;
use antecedent_model::nonlinear_equilibrium::{
    Activation, Clamp, Equation, EquilibriumModel, RegimeRow, SolvePolicy,
};
use common::calibration::{
    Construction, CoverageTally, REPORTED_LEVEL, RecordKey, ScopeFacts, gaussian, grid_n,
    grid_seed, map_replicates, n_sim, uniform,
};

const TEST: &str = "cyclic_posterior_population_outcome_l95";
const DGP: &str = "two_equation_gaussian_feedback_known_covariance";
const INTERVAL: &str = "nearest_rank_95";

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

fn replicate(seed: u64, n: usize) -> Option<((f64, f64), f64, CalibrationBasis)> {
    let mut assignment = uniform(seed, 0xC1C1_A001);
    let mut noise_w = gaussian(seed ^ 0xC1C1_A002);
    let mut noise_t = gaussian(seed ^ 0xC1C1_A003);
    let training: Vec<RegimeRow> = (0..n)
        .map(|_| {
            let ad = if assignment() < 0.5 { 0.0 } else { 1.0 };
            let ew = 0.2 * noise_w();
            let et = 0.2 * noise_t();
            let wishlists = (0.2 * ad + ew + 0.3 * et) / 0.94;
            let traffic = 0.2 * wishlists + et;
            RegimeRow {
                values: vec![wishlists, traffic, ad],
                clamps: vec![Clamp { variable: 2, value: ad }],
            }
        })
        .collect();
    let model = EquilibriumModel {
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
    };
    let request = CyclicPosteriorDecisionRequest {
        model,
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
        initial: vec![0.0; 3],
        solve_policy: SolvePolicy::default(),
    };
    let outputs: Vec<CyclicDecisionOutput> = ["ordinary", "promoted"]
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
    let ctx = ExecutionContext::for_tests(grid_seed(seed));
    let plan = compile_cyclic_posterior_action_decision(request, outputs, contract, &ctx).ok()?;
    let candidate = plan.calibration_candidate(0, &ctx).ok()?;
    assert!(candidate.result.posterior.diagnostics_pass);
    assert_eq!(candidate.basis.interval_method.as_ref(), INTERVAL);
    let future_w = 0.2 * noise_w();
    let future_t = 0.2 * noise_t();
    let future_wishlists = (future_w + 0.3 * future_t) / 0.94;
    Some((candidate.interval, future_wishlists, candidate.basis))
}

#[test]
#[ignore = "coverage: measure with scripts/measure_calibration.sh"]
fn cyclic_posterior_population_outcome_l95() {
    let n = grid_n(240);
    let mut tally = CoverageTally::for_record(
        RecordKey { test: TEST, dgp: DGP, interval: INTERVAL },
        REPORTED_LEVEL,
    );
    for result in map_replicates(n_sim(), |rep| replicate(0xC1C1_7001 + rep, n)) {
        if let Some((interval, truth, basis)) = result {
            let construction = Construction {
                query: basis.query.to_string(),
                graph_class: basis.graph_class.to_string(),
                structure: basis.structure.to_string(),
                modality: basis.modality.to_string(),
                inference: basis.inference.to_string(),
                estimator: basis.estimator.to_string(),
                interval_method: basis.interval_method.to_string(),
                se_kind: basis.se_kind.to_string(),
                dependence: basis.dependence.to_string(),
                posterior: basis.posterior.to_string(),
                functional: basis.functional.to_string(),
                identification: basis.identification.to_string(),
                reported_level: basis.level,
            };
            tally.bind_validation(
                &construction,
                ScopeFacts {
                    row_count: basis.row_count,
                    replicates_ok: basis.replicates_ok,
                    posterior_draws: basis.posterior_draws,
                    unidentified_mass: basis.unidentified_mass,
                },
                &basis.validation,
            );
            tally.record(Some(interval), truth);
        } else {
            tally.skip();
        }
    }
    tally.assert();
}
